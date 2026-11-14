"""RSMT-Q pooled panel: grid, labels, offsets, weights, drop accounting (synthetic, offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_prices

from quantcio.data.base import PricePanel
from quantcio.models import Market
from quantcio.quant.features import ewma_vol, regime_features
from quantcio.quant.panel import (
    OFFSET_MISSING_REASON,
    PanelData,
    build_panel,
    market_moments,
    market_mu_prior,
    moments_from_settings,
)

H = 20


def _panel(
    n_tickers: int = 12,
    n_days: int = 900,
    seed: int = 0,
    extra: dict | None = None,
    bench_of: dict | None = None,
    failed: dict | None = None,
) -> PricePanel:
    bench = make_prices(n=n_days, start_price=2500.0, drift=0.0002, vol=0.011, seed=500 + seed)
    prices = {
        f"{100000 + i:06d}": make_prices(
            n=n_days, drift=0.0003 * (i % 3 - 1), vol=0.012 + 0.002 * (i % 5), seed=seed * 100 + i
        )
        for i in range(n_tickers)
    }
    prices.update(extra or {})
    tb = {t: "KOSPI" for t in prices}
    tb.update(bench_of or {})
    return PricePanel(Market.KR, prices, {"KOSPI": bench}, tb, failed=dict(failed or {}))


@pytest.fixture(scope="module")
def pp() -> PricePanel:
    return _panel()


@pytest.fixture(scope="module")
def pdata(pp, settings_mod) -> PanelData:
    return build_panel(pp, settings_mod)


@pytest.fixture(scope="module")
def settings_mod():
    from quantcio.config import load_settings

    # the non-overlapping grid (overlap_step 0) - the overlapping grid has its own test below
    return load_settings(
        rsmt={"overlap_step": 0, "markets": {"KR": {"overlap_step": None}, "US": {"overlap_step": None}}}
    )


def test_grid_anchored_on_latest_date_and_non_overlapping(pp, pdata, settings_mod):
    step = max(settings_mod.rsmt.sample_step, H)
    calendar = pd.DatetimeIndex(sorted(set().union(*(p.index for p in pp.prices.values()))))
    grid = pdata.grid_dates
    assert grid[-1] == calendar[-1] == pdata.as_of
    pos = calendar.get_indexer(grid)
    assert (np.diff(pos) == step).all()  # every step-th session, counted backwards
    assert pos[-1] == len(calendar) - 1
    assert set(pdata.train["date"]).issubset(set(grid))
    assert pdata.train["date"].max() < grid[-1]  # latest date is the prediction date
    # label known: t + h <= last date, exactly h sessions ahead
    tpos = calendar.get_indexer(pd.DatetimeIndex(pdata.train["date"]))
    epos = calendar.get_indexer(pd.DatetimeIndex(pdata.train["t_end"]))
    assert (epos - tpos == H).all() and (epos <= len(calendar) - 1).all()
    # prediction rows: latest date per ticker, one per ticker
    assert list(pdata.pred["ticker"]) == sorted(pp.prices)
    for row in pdata.pred.itertuples(index=False):
        assert row.date == pp.prices[row.ticker].index[-1]


def test_labels_beta_and_excess_return(pp, pdata, settings_mod):
    row = pdata.train.iloc[len(pdata.train) // 2]
    px = pp.prices[row.ticker]
    bench = pp.benchmarks["KOSPI"]
    i = px.index.get_loc(row.date)
    r = np.log(px["close"].iloc[i + H] / px["close"].iloc[i])
    m = np.log(bench["close"].loc[px.index[i + H]] / bench["close"].loc[row.date])
    feats = regime_features(px, bench)
    k = settings_mod.rsmt.beta_shrink
    beta = (1 - k) * feats["beta_252"].loc[row.date] + k
    assert row.r == pytest.approx(r)
    assert row.m == pytest.approx(m)
    assert row.beta == pytest.approx(beta)
    assert row.y_e == pytest.approx(r - beta * m)
    for c in settings_mod.rsmt.partition_features:
        assert getattr(row, c) == pytest.approx(feats[c].loc[row.date])
    assert row.bench_vol == pytest.approx(ewma_vol(bench["close"]).loc[row.date])


def test_no_look_ahead_future_prices_do_not_change_features(pp, settings_mod):
    step = max(settings_mod.rsmt.sample_step, H)
    calendar = next(iter(pp.prices.values())).index
    cut = calendar[-1 - 3 * step]
    rng = np.random.default_rng(1)
    shocked = {}
    for t, px in pp.prices.items():
        q = px.copy()
        after = q.index > cut
        q.loc[after, ["open", "high", "low", "close"]] *= np.exp(rng.normal(0, 0.3, after.sum()))[:, None]
        shocked[t] = q
    alt = PricePanel(Market.KR, shocked, pp.benchmarks, pp.ticker_benchmark)
    a = build_panel(pp, settings_mod, as_of=cut)
    b = build_panel(alt, settings_mod, as_of=cut)
    pd.testing.assert_frame_equal(a.train, b.train)
    pd.testing.assert_frame_equal(a.pred.drop(columns="degraded_reasons"), b.pred.drop(columns="degraded_reasons"))
    assert a.as_of == cut and (a.train["t_end"] <= cut).all()
    # full-history panels: features of rows on/before the cut are identical, only labels may differ
    fa, fb = build_panel(pp, settings_mod), build_panel(alt, settings_mod)
    feat = [*settings_mod.rsmt.partition_features, "mom_12_1", "ewma_vol", "xs_mom_z", "beta"]
    ma = fa.train[fa.train["date"] <= cut].set_index(["ticker", "date"])[feat]
    mb = fb.train[fb.train["date"] <= cut].set_index(["ticker", "date"])[feat]
    pd.testing.assert_frame_equal(ma, mb)


def test_xs_momentum_z_scored_and_winsorised(settings_mod):
    n_days = 900
    rocket = make_prices(n=n_days, drift=0.0, vol=0.012, seed=999)
    rocket[["open", "high", "low", "close"]] *= np.exp(np.linspace(0, 3.0, n_days))[:, None]  # huge momentum
    panel = _panel(n_tickers=19, n_days=n_days, extra={"999999": rocket})
    pdata = build_panel(panel, settings_mod)
    tr = pdata.train
    assert tr["xs_mom_z"].between(-3.0, 3.0).all()
    rocket_z = tr.loc[tr["ticker"] == "999999", "xs_mom_z"]
    assert (rocket_z == 3.0).any() and (rocket_z > 1.0).all()  # winsorised at +3
    # independent recomputation on every training date
    mom = pd.DataFrame(
        {t: regime_features(px, panel.benchmarks["KOSPI"])["mom_12_1"] for t, px in panel.prices.items()}
    )
    for d, g in tr.groupby("date"):
        x = mom.loc[d]
        expected = ((x - x.mean()) / x.std(ddof=1)).clip(-3, 3)
        np.testing.assert_allclose(g.set_index("ticker")["xs_mom_z"], expected.loc[g["ticker"]].to_numpy())
    assert (tr["z"] == tr["xs_mom_z"]).all()
    for _, g in tr.groupby("date"):
        if g["z"].abs().max() < 3.0:
            assert abs(g["z"].mean()) < 1e-9


def test_fewer_than_ten_tickers_gives_nan_z_and_is_reported(settings_mod):
    with pytest.raises(ValueError, match="학습 행 없음") as exc:
        build_panel(_panel(n_tickers=9), settings_mod)
    assert "'z'" in str(exc.value)


def test_v_centering_weights_and_drop_accounting(pdata, settings_mod):
    tr = pdata.train
    w = tr["w"].to_numpy()
    assert (w * tr["v"]).sum() / w.sum() == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(tr["v"], np.log(tr["ewma_vol"]) - pdata.c_v)
    np.testing.assert_allclose(pdata.pred["v"], np.log(pdata.pred["ewma_vol"]) - pdata.c_v)
    hl = settings_mod.rsmt.markets["KR"].half_life_days
    age = (pdata.as_of - pd.DatetimeIndex(tr["date"])).days.to_numpy()
    np.testing.assert_allclose(w, 0.5 ** (age / hl))
    assert (pdata.pred["w"] == 1.0).all()
    # every dropped row is counted and reported
    assert pdata.dropped["total"] == pdata.n_candidates - len(tr) > 0
    assert pdata.dropped["vol_pct_3y"] > 0 and pdata.dropped["z"] > 0
    required = [*settings_mod.rsmt.partition_features, "z", "v", "beta", "bench_vol", "y_e", "m"]
    assert not tr[required].isna().any().any()


def test_market_moments_and_benchmark_ewma(pp, pdata, settings_mod):
    d = pdata.train.drop_duplicates(["bench", "date"])
    hist = (d["w"] * d["m"]).sum() / d["w"].sum()
    s = settings_mod.rsmt.market_mu_shrink
    prior = np.log1p(settings_mod.rsmt.market_mu_prior_annual) * H / 252
    expected = (1 - s) * hist + s * prior
    assert pdata.market_mu_hist == pytest.approx(expected)
    assert pdata.market_mu_by_bench["KOSPI"] == pytest.approx(expected)
    assert pdata.market_mu_raw == pytest.approx(hist)
    assert pdata.market_mu_raw_by_bench["KOSPI"] == pytest.approx(hist)
    assert pdata.market_mu_prior == pytest.approx(prior) and pdata.market_mu_shrink == s
    kr = settings_mod.rsmt.markets["KR"]
    assert kr.nu_min <= pdata.market_nu <= kr.nu_max
    bvol = ewma_vol(pp.benchmarks["KOSPI"]["close"]).iloc[-1]
    assert all(v == pytest.approx(bvol) for v in pdata.benchmark_ewma.values())
    mm = market_moments(pdata.train, kr.nu_min, kr.nu_max, mu_prior=prior, shrink=s)
    assert mm.mu == pytest.approx(pdata.market_mu_hist) and mm.n_obs == len(d)
    assert mm.mu_hist == pytest.approx(hist)
    ms = moments_from_settings(pdata.train, settings_mod, Market.KR, H)
    assert ms.mu == pytest.approx(mm.mu) and ms.mu_by_bench == pytest.approx(mm.mu_by_bench)


def test_market_drift_prior_does_not_extrapolate_a_rally(settings_mod):
    """Regression (live 2026-09-24): the old rule mu_m = 0.5 * time-decayed mean served +1.45%/20d of
    KOSPI drift (~20%/yr) after a rally. mu_m = (1-s)*hist + s*log(1+prior)*h/252 keeps it near the
    long-run prior whatever the recent benchmark did."""
    rally = make_prices(n=900, start_price=2500.0, drift=0.004, vol=0.011, seed=901)  # ~+8%/20d
    pp = _panel(n_tickers=12, seed=9)
    pp.benchmarks["KOSPI"] = rally
    pdata = build_panel(pp, settings_mod)
    s = settings_mod.rsmt.market_mu_shrink
    prior = market_mu_prior(settings_mod.rsmt.market_mu_prior_annual, H)
    assert prior == pytest.approx(np.log(1.06) * 20 / 252)
    hist = pdata.market_mu_raw
    assert hist > 0.05  # the recent rally
    assert pdata.market_mu_hist == pytest.approx((1 - s) * hist + s * prior)
    assert pdata.market_mu_hist < 0.5 * hist  # below even the old 50% rule
    assert abs(pdata.market_mu_hist - prior) == pytest.approx((1 - s) * abs(hist - prior))
    # the two extremes of the shrink weight
    kr = settings_mod.rsmt.markets["KR"]
    assert market_moments(pdata.train, kr.nu_min, kr.nu_max, mu_prior=prior, shrink=1.0).mu == pytest.approx(prior)
    assert market_moments(pdata.train, kr.nu_min, kr.nu_max, mu_prior=prior, shrink=0.0).mu == pytest.approx(hist)
    # the config is honoured (no hidden constant)
    from quantcio.config import load_settings

    s2 = load_settings(rsmt={"market_mu_shrink": 0.5, "market_mu_prior_annual": 0.0})
    p2 = build_panel(pp, s2)
    assert p2.market_mu_hist == pytest.approx(0.5 * hist)


def test_market_drift_prior_inputs_are_validated(pdata, settings_mod):
    kr = settings_mod.rsmt.markets["KR"]
    with pytest.raises(TypeError):
        market_moments(pdata.train, kr.nu_min, kr.nu_max)  # the prior can never be skipped silently
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="market_mu_shrink"):
            market_moments(pdata.train, kr.nu_min, kr.nu_max, mu_prior=0.004, shrink=bad)
    with pytest.raises(ValueError):
        market_moments(pdata.train, kr.nu_min, kr.nu_max, mu_prior=float("nan"), shrink=0.8)
    for bad in (-1.0, -2.0, float("inf")):
        with pytest.raises(ValueError, match="market_mu_prior_annual"):
            market_mu_prior(bad, H)
    with pytest.raises(ValueError):
        market_mu_prior(0.06, 0)


def test_prediction_row_with_missing_offset_is_kept_and_degraded(settings_mod):
    short = make_prices(n=230, seed=77)  # < 252 bars: no 12-1 momentum -> z NaN
    pdata = build_panel(_panel(extra={"777777": short}), settings_mod)
    row = pdata.pred.set_index("ticker").loc["777777"]
    assert bool(row["degraded"]) is True
    assert any(r.startswith(OFFSET_MISSING_REASON) and "xs_mom_z" in r for r in row["degraded_reasons"])
    assert row["z"] == 0.0  # set explicitly, never silently dropped
    assert any("분할 변수 누락" in r for r in row["degraded_reasons"])
    assert any("베타 누락" in r for r in row["degraded_reasons"]) and row["beta"] == 1.0
    assert "777777" not in set(pdata.train["ticker"])  # no usable training rows (counted in dropped)
    healthy = pdata.pred[pdata.pred["ticker"] != "777777"]
    assert not healthy["degraded"].any()


def test_missing_benchmark_stale_ticker_and_failures_are_reported(settings_mod):
    stale = make_prices(n=895, seed=88, end="2026-09-16")  # ends 5 sessions early
    orphan = make_prices(n=900, seed=89)
    pdata = build_panel(
        _panel(
            extra={"888888": stale, "999998": orphan},
            bench_of={"999998": "KOSDAQ"},
            failed={"123456": "HTTPError: 500", "benchmark:KOSDAQ": "timeout"},
        ),
        settings_mod,
    )
    pred = pdata.pred.set_index("ticker")
    assert any("가격 지연" in r for r in pred.loc["888888", "degraded_reasons"])
    assert any("벤치마크 변동성 누락" in r for r in pred.loc["999998", "degraded_reasons"])
    assert any("999998" in n and "KOSDAQ" in n for n in pdata.notes)
    assert "123456" in pdata.failed_tickers and "benchmark:KOSDAQ" not in pdata.failed_tickers
    assert pdata.n_tickers == 12 + 2 + 1
    assert pdata.failed_ratio == pytest.approx(1 / 15)


def test_halted_ticker_label_window_longer_than_h_is_dropped_and_counted(settings_mod):
    """Regression: the label is h rows ahead on the ticker's OWN bars, so across a trading halt it
    spanned 35 market sessions (not h=20) and overlapped the next grid sample. Such rows are dropped
    and counted in dropped['label_span']; a short gap (<= LABEL_SPAN_TOL sessions) is kept."""
    halted = make_prices(n=900, seed=77)
    halted = halted.drop(halted.index[400:415])  # 15-session halt
    blip = make_prices(n=900, seed=78)
    blip = blip.drop(blip.index[[500]])  # one missing bar
    pdata = build_panel(_panel(extra={"555555": halted, "666666": blip}), settings_mod)
    cal = pd.DatetimeIndex(sorted(set().union(*(p.index for p in _panel().prices.values()))))
    tr = pdata.train
    span = cal.get_indexer(pd.DatetimeIndex(tr["t_end"])) - cal.get_indexer(pd.DatetimeIndex(tr["date"]))
    assert span.max() <= H + 2 and (span >= H).all()
    assert pdata.dropped.get("label_span", 0) >= 1
    assert pdata.dropped["total"] == pdata.n_candidates - len(tr)
    for t in ("555555", "666666"):  # consecutive samples never overlap
        g = tr[tr["ticker"] == t].sort_values("date")
        assert (
            cal.get_indexer(pd.DatetimeIndex(g["t_end"]))[:-1] <= cal.get_indexer(pd.DatetimeIndex(g["date"]))[1:] + 2
        ).all()
    assert (tr["ticker"] == "666666").sum() >= (tr["ticker"] == "100000").sum() - 1  # a blip costs <= 1 row


def test_as_of_cutoff_ignores_later_bars(pp, settings_mod):
    cut = pd.Timestamp("2026-06-30")
    pdata = build_panel(pp, settings_mod, as_of=cut)
    assert pdata.as_of <= cut
    assert (pdata.pred["date"] <= cut).all() and (pdata.train["t_end"] <= cut).all()


def test_overlapping_grid_nests_the_sample_grid_with_h_blocks_and_scale(pp):
    """rsmt.overlap_step = 5 (docs/RSMT.md): training rows every 5 sessions, the non-overlapping sample grid is the
    `on_grid` subset, `block` = h-session blocks counted back from the latest date, `scale_h` = EWMA vol *
    sqrt(h/252)."""
    from quantcio.config import load_settings

    s0 = load_settings(rsmt={"overlap_step": 0, "markets": {"KR": {"overlap_step": None}}})
    s5 = load_settings(rsmt={"overlap_step": 5, "markets": {"KR": {"overlap_step": None}}})
    a, b = build_panel(pp, s0), build_panel(pp, s5)
    calendar = pd.DatetimeIndex(sorted(set().union(*(p.index for p in pp.prices.values()))))
    fine = calendar.get_indexer(pd.DatetimeIndex(sorted(b.train["date"].unique())))
    assert (np.diff(fine) == 5).all() and (len(calendar) - 1 - fine[-1]) % 5 == 0
    assert set(b.train.loc[b.train["on_grid"], "date"]) == set(a.train["date"]) and a.train["on_grid"].all()
    assert b.grid_dates.equals(a.grid_dates) and b.overlap_step == 5 and b.block_sessions == H
    pos = calendar.get_indexer(pd.DatetimeIndex(b.train["date"]))
    np.testing.assert_array_equal(b.train["block"].to_numpy(), (pos - (len(calendar) - 1)) // H)
    np.testing.assert_allclose(b.train["scale_h"], b.train["ewma_vol"] * np.sqrt(H / 252))
    np.testing.assert_allclose(b.pred["scale_h"], np.exp(b.pred["v"] + b.c_v) * np.sqrt(H / 252))
    # the market leg ignores the overlapping rows
    assert b.market_mu_hist == pytest.approx(a.market_mu_hist)


def test_sigma_term_columns_missing_inputs_and_fitted_centering():
    """HAR / calendar columns: a configured term missing in training drops the row (counted); missing at serving
    time it is set to 0 (neutral) and the row is flagged; `c_v` of a fitted model is honoured."""
    from quantcio.config import load_settings
    from quantcio.quant.panel import SIGMA_TERM_COLUMNS, SIGMA_TERM_MISSING_REASON

    s = load_settings(
        rsmt={
            "overlap_step": 0,
            "markets": {
                "KR": {"overlap_step": None, "sigma_terms": ["har_252", "cal_s4"]},
                "US": {"overlap_step": None},
            },
        }
    )
    young = make_prices(n=240, seed=78)  # < 252 returns: har_252 missing on every row
    pd_ = build_panel(_panel(extra={"778888": young}), s)
    assert set(SIGMA_TERM_COLUMNS) <= set(pd_.train.columns) and set(SIGMA_TERM_COLUMNS) <= set(pd_.pred.columns)
    assert pd_.train["har_252"].notna().all() and pd_.dropped.get("har_252", 0) > 0
    row = pd_.pred.set_index("ticker").loc["778888"]
    assert row["har_252"] == 0.0 and bool(row["degraded"])
    assert any(r.startswith(SIGMA_TERM_MISSING_REASON) and "har_252" in r for r in row["degraded_reasons"])
    doy = pd.DatetimeIndex(pd_.train["date"]).dayofyear.to_numpy() - 1.0
    np.testing.assert_allclose(pd_.train["cal_s4"], np.sin(2 * np.pi * 4 * doy / 365.25), atol=1e-12)
    fixed = build_panel(_panel(), s, c_v=-1.5)
    assert fixed.c_v == -1.5
    np.testing.assert_allclose(fixed.train["v"], np.log(fixed.train["scale_h"] / np.sqrt(H / 252)) + 1.5, atol=1e-9)
