"""`quantcio evaluate`: ranking / calibration / barrier statistics, look-ahead guards and the end-to-end
out-of-sample evaluation on synthetic panels (offline)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from conftest import make_prices
from typer.testing import CliRunner

from quantcio import evaluation as ev
from quantcio.config import load_settings
from quantcio.data.base import PricePanel
from quantcio.models import Instrument, Market
from quantcio.quant import rsmt
from quantcio.quant.distribution import ReturnDistribution
from quantcio.quant.indicators import atr as atr_series
from quantcio.quant.panel import build_panel

# ------------------------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def eval_settings(tmp_path_factory):
    return load_settings(data={"cache_dir": str(tmp_path_factory.mktemp("evalcache"))})


def xs_signal_panel(
    n_tickers: int = 30, n_days: int = 1300, k: float = 0.0006, seed: int = 0, noise_only: bool = False
) -> PricePanel:
    """Panel whose residual daily drift is k * (cross-sectional z of 12-1 momentum at t), computed from the
    simulated closes up to t: a planted, causal cross-sectional signal (k = 0 / noise_only -> none).

    r_i,t = beta_i * m_t + k * z_i,t + 0.018 * eps_i,t.
    """
    rng = np.random.default_rng(seed)
    bench = make_prices(n=n_days, start_price=2500.0, drift=0.0002, vol=0.011, seed=20_000 + seed)
    blr = np.log(bench["close"]).diff().fillna(0.0).to_numpy()
    betas = 0.6 + 0.8 * rng.random(n_tickers)
    eps = rng.normal(0.0, 0.018, (n_days, n_tickers))
    logp = np.zeros((n_days, n_tickers))
    logp[0] = np.log(50_000.0)
    for t in range(1, n_days):
        drift = np.zeros(n_tickers)
        if not noise_only and t > 253:
            mom = logp[t - 22] - logp[t - 253]  # 12-1 momentum known at t-1
            z = (mom - mom.mean()) / mom.std(ddof=1)
            drift = k * np.clip(z, -3, 3)
        logp[t] = logp[t - 1] + betas * blr[t] + drift + eps[t]
    prices, tb = {}, {}
    for i in range(n_tickers):
        base = make_prices(n=n_days, drift=0.0, vol=0.018, seed=seed * 1000 + i)
        close = np.exp(logp[:, i])
        ratio = close / base["close"].to_numpy()
        px = base.copy()
        px[["open", "high", "low", "close"]] = px[["open", "high", "low", "close"]].to_numpy() * ratio[:, None]
        t = f"{300000 + i:06d}"
        prices[t], tb[t] = px, "KOSPI"
    return PricePanel(Market.KR, prices, {"KOSPI": bench}, tb)


def _xs_frame(n_dates: int, n_names: int, ic: float, seed: int) -> pd.DataFrame:
    """Long frame (date, score, y_e): y = ic-correlated with the score plus a common date shock."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2020-01-01", periods=n_dates, freq="20B"):
        s = rng.normal(size=n_names)
        y = ic * s + math.sqrt(1 - ic * ic) * rng.normal(size=n_names) + rng.normal() * 3.0  # date shock
        rows.append(pd.DataFrame({"date": d, "score": s, "y_e": 0.05 * y}))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------ B: ranking statistics


def test_cross_sectional_ic_detects_a_planted_signal():
    f = _xs_frame(60, 40, ic=0.15, seed=1)
    st = ev.ranking_stats(ev.cross_sectional_ic(f, "score"), horizon=20)
    assert st["n_dates"] == 60
    assert 0.10 < st["mean_ic"] < 0.20 and st["t_ic"] > 4 and st["share_ic_pos"] > 0.75
    assert st["spread_mean"] > 0 and st["spread_t"] > 3 and st["top_excess_mean"] > 0
    assert st["spread_annual"] == pytest.approx(st["spread_mean"] * 252 / 20)
    # the common date shock cancels in ranks and in the top-minus-bottom spread
    shocked = f.assign(y_e=f["y_e"] + f["date"].map({d: 10.0 * i for i, d in enumerate(f["date"].unique())}))
    st2 = ev.ranking_stats(ev.cross_sectional_ic(shocked, "score"), horizon=20)
    assert st2["mean_ic"] == pytest.approx(st["mean_ic"]) and st2["spread_mean"] == pytest.approx(st["spread_mean"])


def test_cross_sectional_ic_of_noise_is_near_zero_and_the_t_is_calibrated():
    f = _xs_frame(60, 40, ic=0.0, seed=2)
    st = ev.ranking_stats(ev.cross_sectional_ic(f, "score"), horizon=20)
    assert abs(st["mean_ic"]) < 0.04 and abs(st["t_ic"]) < 2.5 and 0.3 < st["share_ic_pos"] < 0.7
    band = ev.random_null_band(f, "y_e", n_perm=300, seed=0)
    assert -2.8 < band["t_q025"] < -1.4 and 1.4 < band["t_q975"] < 2.8
    # false-positive rate of |t| >= 2 across independent noise panels is ~5%
    hits = 0
    for seed in range(120):
        g = _xs_frame(25, 20, ic=0.0, seed=100 + seed)
        hits += abs(ev.ranking_stats(ev.cross_sectional_ic(g, "score"), 20)["t_ic"]) >= 2.0
    assert 1 <= hits <= 14  # 120 x 5% = 6 expected


def test_quantile_buckets_with_ties_and_constant_scores():
    d1, d2 = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-01")
    f = pd.concat(
        [
            pd.DataFrame({"date": d1, "score": [0.0] * 10 + [1.0] * 10, "y_e": np.arange(20, dtype=float)}),
            pd.DataFrame({"date": d2, "score": 7.0, "y_e": np.arange(20, dtype=float)}),  # constant score
        ],
        ignore_index=True,
    )
    per = ev.cross_sectional_ic(f, "score")
    r1 = per.set_index("date").loc[d1]
    # equal-sized buckets (4 names each) drawn inside the tie groups: top = 4 of the score-1 names
    assert r1["top"] >= 10 and r1["bottom"] <= 9 and r1["spread"] > 0 and r1["ic"] > 0.8
    r2 = per.set_index("date").loc[d2]
    assert math.isnan(r2["ic"]) and math.isnan(r2["spread"])  # no ranking information, never a fake spread
    assert ev.ranking_stats(per, 20)["n_dates"] == 1
    # dates with fewer than min_names names are skipped
    assert ev.cross_sectional_ic(f.iloc[:5], "score").empty


def test_paired_ic_difference_uses_common_dates():
    a = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]), "ic": [0.2, 0.1, 0.3]})
    b = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-04-01"]), "ic": [0.1, 0.1, 0.9]})
    out = ev.paired_ic_difference(a, b)
    assert out["n_dates"] == 2 and out["mean_diff"] == pytest.approx(0.05)


# ------------------------------------------------------------------------- bootstrap / calibration


def test_block_bootstrap_resamples_whole_dates():
    rng = np.random.default_rng(0)
    dates = np.repeat(pd.bdate_range("2021-01-01", periods=40).to_numpy(), 50)
    shock = np.repeat(rng.normal(0, 1, 40), 50)  # every row on a date shares a shock
    v = shock + rng.normal(0, 0.2, dates.size)
    hat, draws, g = ev.block_bootstrap_means(v, dates, 999, seed=0)
    assert g == 40 and hat[0] == pytest.approx(v.mean()) and draws.shape == (999, 1)
    sd_dates = draws[:, 0].std()
    assert sd_dates == pytest.approx(shock[::50].std(ddof=1) / math.sqrt(40), rel=0.2)  # ~ date-level SE
    assert sd_dates > 3 * v.std() / math.sqrt(v.size)  # a row bootstrap would be far too narrow
    lo, hi = ev.ci90(hat[0], draws[:, 0], g)
    assert lo < hat[0] < hi
    q05, q95 = np.quantile(draws[:, 0], [0.05, 0.95])
    assert hi - lo > q95 - q05  # small-cluster widening (c_G > 1)
    _, _, g3 = ev.block_bootstrap_means(v, dates, 10, seed=0, block_dates=3)
    assert g3 == math.ceil(40 / 3)
    with pytest.raises(ValueError):
        ev.block_bootstrap_means(np.array([1.0, np.nan]), dates[:2], 10)
    with pytest.raises(ValueError):
        ev.block_bootstrap_means(v[:3], dates[:2], 10)
    assert all(math.isnan(x) for x in ev.ci90(0.1, np.array([]), 40))


def test_calibration_summary_flags_overconfident_probabilities():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.2, 0.8, 20000)
    y = (rng.random(p.size) < p).astype(float)
    good = ev.calibration_summary(p, y)
    assert good["ece"] < 0.02 and 0.85 < good["slope"] < 1.15 and abs(good["intercept"]) < 0.1
    lp = np.log(p / (1 - p))
    over = 1 / (1 + np.exp(-2.5 * lp))  # same ranking, far too extreme
    bad = ev.calibration_summary(over, y)
    assert bad["slope"] < 0.5 and bad["ece"] > good["ece"]
    table = ev.reliability_table(p, y, 10)
    assert len(table) == 10 and sum(r["n"] for r in table) == p.size
    assert all(abs(r["gap"]) < 0.04 for r in table)


# ------------------------------------------------------------------------------- D: barrier replay


def test_stop_take_rule(eval_settings):
    assert ev.stop_take(100.0, 1.0, eval_settings) == pytest.approx((-0.04, 0.06))  # 2*ATR < 4% floor
    assert ev.stop_take(100.0, 3.0, eval_settings) == pytest.approx((-0.06, 0.09))
    assert ev.stop_take(100.0, 10.0, eval_settings) == pytest.approx((-0.15, 0.225))  # capped at 15%
    with pytest.raises(ValueError):
        ev.stop_take(100.0, float("nan"), eval_settings)


def test_realised_barrier_first_touch_on_closes():
    assert ev.realised_barrier(np.array([101.0, 104.0, 111.0, 80.0]), 100.0, -0.05, 0.10) == (1, pytest.approx(0.11))
    # a gap through the stop keeps the gap (exit at the breach-day close)
    assert ev.realised_barrier(np.array([99.0, 94.0, 120.0]), 100.0, -0.05, 0.10) == (-1, pytest.approx(-0.06))
    assert ev.realised_barrier(np.array([101.0, 99.0, 102.0]), 100.0, -0.05, 0.10) == (0, pytest.approx(0.02))
    with pytest.raises(ValueError):
        ev.realised_barrier(np.array([]), 100.0, -0.05, 0.10)


def test_realised_barrier_matches_the_distribution_barrier_rule():
    """The realised replay and ReturnDistribution.barrier_trades apply the same first-touch rule."""
    dist = ReturnDistribution(0.01, 0.10, 0.004, 0.05, 6.0, 8.0, 1.1, 0.2, 20, 300, seed=5, hold_days=60)
    outcome, trade = dist.barrier_trades(-0.07, 0.10)
    closes = 100.0 * np.exp(dist.r_paths.astype("float64"))
    for i in range(dist.n_paths):
        o, r = ev.realised_barrier(closes[i], 100.0, -0.07, 0.10)
        assert o == outcome[i] and r == pytest.approx(trade[i], abs=1e-9)


def _gbm_panel(n_tickers: int, n_days: int, m_d: float, s_d: float, seed: int) -> PricePanel:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-09-23", periods=n_days)
    prices = {}
    for i in range(n_tickers):
        close = 100.0 * np.exp(np.cumsum(rng.normal(m_d, s_d, n_days)))
        prices[f"G{i:03d}"] = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6}, index=idx
        )
    bench = make_prices(n=n_days, seed=seed + 1)
    return PricePanel(Market.US, prices, {"SPY": bench}, {t: "SPY" for t in prices})


def test_barrier_calibration_on_gbm_matches_known_probabilities(eval_settings):
    """GBM prices with known daily (m, s): the predicted first-touch probabilities of the TRUE distribution
    match the realised frequencies; a mis-specified twin (no drift, 1.8x vol) is worse on Brier."""
    h, hold, m_d, s_d = 20, 60, 0.0008, 0.02
    panel = _gbm_panel(40, 1600, m_d, s_d, seed=11)
    idx = next(iter(panel.prices.values())).index
    dates = idx[300::60]  # non-overlapping 60-day windows per ticker; the last ones lack 60 future bars
    true = {"mu_e": m_d * h, "sigma_e": s_d * math.sqrt(h), "nu": 1e6, "rho": 0.0}
    wrong = {"base_mu_e": 0.0, "base_sigma_e": 1.8 * s_d * math.sqrt(h), "base_nu": 1e6, "base_rho": 0.0}
    oos = pd.DataFrame(
        [
            {"date": d, "ticker": t, "fold": 1, **true, **wrong, "mu_m": 0.0, "sigma_m": 0.0, "nu_m": 1e6, "beta": 0.0}
            for t in panel.prices
            for d in dates
        ]
    )
    frame, excluded = ev.barrier_frame(oos, panel, eval_settings, h, hold=hold, n_paths=600, workers=2)
    assert excluded["no_future"] == 40 * sum(idx.get_loc(d) + hold >= len(idx) for d in dates)
    assert len(frame) + sum(excluded.values()) == len(oos)
    assert np.allclose(frame[["p_tp", "p_sl", "p_time"]].sum(axis=1), 1.0)
    real_tp = (frame["outcome"] == 1).mean()
    real_sl = (frame["outcome"] == -1).mean()
    se = math.sqrt(0.25 / len(frame))
    assert abs(frame["p_tp"].mean() - real_tp) < 4 * se + 0.01
    assert abs(frame["p_sl"].mean() - real_sl) < 4 * se + 0.01
    assert frame["e_trade"].mean() == pytest.approx(frame["trade_ret"].mean(), abs=0.01)
    summ = ev.barrier_summary(frame, n_boot=199, seed=0, block_dates=1)
    assert summ["brier3"]["model"] < summ["brier3"]["base"] and summ["brier3"]["gain"]["hat"] > 0
    assert summ["tp"]["real"]["hat"] == pytest.approx(real_tp)
    assert len(summ["reliability_tp"]) == ev.BARRIER_BINS
    # stops/take-profits come from ATR at t (bars <= t)
    px = panel.prices["G000"]
    d0 = dates[0]
    i0 = px.index.get_loc(d0)
    a0 = float(atr_series(px.iloc[: i0 + 1], eval_settings.risk.atr_period).iloc[-1])
    row = frame[(frame["ticker"] == "G000") & (frame["date"] == d0)].iloc[0]
    assert (row["stop_pct"], row["take_pct"]) == pytest.approx(ev.stop_take(px["close"].iloc[i0], a0, eval_settings))


# ----------------------------------------------------------------------------- look-ahead guards


def test_atr_at_t_ignores_future_bars():
    px = make_prices(n=600, seed=4)
    full = atr_series(px, 14)
    for i in (50, 300, 599):
        assert full.iloc[i] == pytest.approx(float(atr_series(px.iloc[: i + 1], 14).iloc[-1]), rel=0, abs=1e-12)


def test_point_in_time_factor_scores_use_only_bars_up_to_the_date():
    px = make_prices(n=1500, seed=6)
    bench = make_prices(n=1500, seed=7, start_price=2500.0, vol=0.011)
    inst = Instrument(ticker="X", market=Market.KR, name="X", currency="KRW", benchmark="KOSPI", benchmark_name="KOSPI")
    dates = [px.index[900], px.index[1200], px.index[1499]]
    base = ev.point_in_time_factor_scores(px, bench, dates, inst)
    assert base[list(ev.PRICE_FACTORS)].notna().all().all() and (base["error"] == "").all()
    # identical with the full history (the PIT_ROWS window loses nothing) ...
    full = ev.point_in_time_factor_scores(px, bench, dates, inst, pit_rows=10**6)
    np.testing.assert_allclose(base[list(ev.PRICE_FACTORS)], full[list(ev.PRICE_FACTORS)], rtol=0, atol=1e-9)
    # ... and unaffected by anything after the date (future bars replaced by a crash)
    for d in dates[:2]:
        crashed = px.copy()
        crashed.loc[crashed.index > d, ["open", "high", "low", "close"]] *= 0.3
        bcrash = bench.copy()
        bcrash.loc[bcrash.index > d, ["open", "high", "low", "close"]] *= 3.0
        again = ev.point_in_time_factor_scores(crashed, bcrash, [d], inst)
        np.testing.assert_allclose(
            again[list(ev.PRICE_FACTORS)].to_numpy()[0],
            base.loc[base["date"] == d, list(ev.PRICE_FACTORS)].to_numpy()[0],
            rtol=0,
            atol=1e-9,
        )
    missing = ev.point_in_time_factor_scores(px, bench, [pd.Timestamp("1990-01-01")], inst)
    assert missing["error"].iloc[0] == "가격 봉 없음" and missing[list(ev.PRICE_FACTORS)].isna().all().all()


def test_cv_oos_rows_expose_the_simulated_distribution(eval_settings):
    """rsmt.cross_validate_layouts: per-row distribution parameters in the OOS frame, consistent with the
    served mean, and the same primary metrics as cross_validate (same seeds)."""
    pdata = build_panel(xs_signal_panel(n_tickers=20, n_days=1000, seed=4), eval_settings)
    runs = rsmt.cross_validate_layouts(pdata, eval_settings, [4])
    res = runs[4]
    assert res.available and res.n_folds_primary == 4 and res.seconds > 0
    o = res.oos
    assert list(o.columns) == list(rsmt.OOS_COLUMNS)
    np.testing.assert_allclose(o["mu"], o["beta"] * o["mu_m"] + o["mu_e"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(o["base_mu"], o["beta"] * o["mu_m"] + o["base_mu_e"], rtol=0, atol=1e-12)
    assert set(np.unique(o["exc95"])) <= {0.0, 1.0} and set(np.unique(o["in80"])) <= {0.0, 1.0}
    assert 0.0 <= o["exc95"].mean() <= 0.2 and 0.5 <= o["in80"].mean() <= 1.0
    assert (o["sigma_e"] > 0).all() and (o["base_sigma_e"] > 0).all()
    info = {f["fold"]: f for f in res.folds}
    for (fold, leaf), _ in o.groupby(["fold", "leaf_id"]):
        assert int(leaf) in info[int(fold)]["leaf_paths"]
        assert (o.loc[o["fold"] == fold, "date"] >= pd.Timestamp(info[int(fold)]["test_start"])).all()
    # every OOS row is a panel training row of a test block (one to one)
    merged = o.merge(pdata.train[["ticker", "date", "r"]], on=["ticker", "date"], how="left", validate="one_to_one")
    np.testing.assert_allclose(merged["r_x"], merged["r_y"])
    primary = rsmt.cross_validate(pdata, eval_settings, robust_folds=[4])
    assert primary.metrics["auc"] == res.metrics["auc"] and primary.metrics["crps"] == res.metrics["crps"]


# ------------------------------------------------------------------------------- verdict helpers


def test_recommend_pillar_weights_never_moves_weight_to_an_untested_pillar():
    cur = {"factor": 0.35, "ml": 0.25, "llm": 0.40}
    ev_all = {"factor": ev.TIER_EVIDENCE, "ml": ev.TIER_EVIDENCE, "llm": ev.TIER_UNTESTABLE}
    assert ev.recommend_pillar_weights(cur, ev_all) == pytest.approx(cur)
    # no pillar with evidence: nowhere to move weight -> unchanged (never into the untested LLM pillar)
    none = {"factor": ev.TIER_NONE, "ml": ev.TIER_WEAK, "llm": ev.TIER_UNTESTABLE}
    assert ev.recommend_pillar_weights(cur, none) == pytest.approx(cur)
    rec = ev.recommend_pillar_weights(cur, {"factor": ev.TIER_NONE, "ml": ev.TIER_EVIDENCE, "llm": ev.TIER_UNTESTABLE})
    assert rec["llm"] == pytest.approx(0.40)  # untested: kept
    assert rec["factor"] == pytest.approx(0.35 * 0.3) and rec["ml"] == pytest.approx(0.25 + 0.35 * 0.7)
    assert sum(rec.values()) == pytest.approx(1.0)
    rec2 = ev.recommend_pillar_weights(cur, {"factor": ev.TIER_EVIDENCE, "ml": ev.TIER_WEAK, "llm": ev.TIER_EVIDENCE})
    assert rec2["ml"] == pytest.approx(0.15) and rec2["factor"] == pytest.approx(0.35 + 0.10 * 0.35 / 0.75)
    assert rec2["llm"] == pytest.approx(0.40 + 0.10 * 0.40 / 0.75)
    with pytest.raises(ValueError):
        ev.recommend_pillar_weights(cur, {"factor": "?", "ml": ev.TIER_NONE, "llm": ev.TIER_NONE})


def test_eval_settings_validates_and_does_not_mutate(eval_settings):
    s = ev.eval_settings(eval_settings, Market.KR, universe=25, years=4)
    assert s.rsmt.markets["KR"].universe_size == 25 and s.history_years == 4
    assert eval_settings.rsmt.markets["KR"].universe_size == 60 and eval_settings.history_years == 7
    with pytest.raises(ValueError):
        ev.eval_settings(eval_settings, Market.KR, universe=5)
    with pytest.raises(ValueError):
        ev.eval_settings(eval_settings, Market.KR, years=2)


# ------------------------------------------------------------------------------ end to end (CLI)


def test_cli_evaluate_end_to_end_on_a_planted_cross_sectional_signal(eval_settings, monkeypatch, tmp_path):
    """Full `quantcio evaluate` on a synthetic panel whose residual drift follows 12-1 momentum: the
    selection section finds it (momentum and RSMT-Q mu_excess IC t > 2), random stays inside the null,
    and the .md / .json reports carry every section."""
    import quantcio.cli as cli

    panel = xs_signal_panel(n_tickers=30, n_days=1300, k=0.0008, seed=0)
    seen = {}

    def fake_load(market, settings, cache, *, use_cache=True):
        seen["universe"] = settings.rsmt.markets[market.value].universe_size
        seen["years"] = settings.history_years
        return panel, sorted(panel.prices), False

    monkeypatch.setattr(ev, "load_eval_panel", fake_load)
    monkeypatch.setattr(cli, "load_settings", lambda config=None: eval_settings)
    monkeypatch.setattr(cli, "_setup_logging", lambda settings, verbose: tmp_path / "test.log")
    runner = CliRunner()
    args = [
        "evaluate",
        "--market",
        "KR",
        "--universe",
        "30",
        "--years",
        "5",
        "--folds",
        "3,4",
        "--universe-mode",
        "current",
    ]
    res = runner.invoke(cli.app, [*args, "--out", str(tmp_path), "--no-ml"])
    assert res.exit_code == 0, res.output
    assert seen == {"universe": 30, "years": 5}
    md_path, json_path = tmp_path / "evaluation_KR_30u_5y.md", tmp_path / "evaluation_KR_30u_5y.json"
    assert md_path.exists() and json_path.exists()
    assert "운영 스킬 게이트" in res.stdout and "G) 결론" in res.stdout

    data = json.loads(json_path.read_text(encoding="utf-8"))
    sec = data["sections"]
    assert set(sec) == set("ABCDEG")  # F (per-ticker ML ensemble) was removed; --no-ml is still accepted
    assert [lay["folds"] for lay in sec["A"]["layouts"]] == [3, 4]
    assert sec["A"]["production"]["layouts"] == [3, 4, 5]
    sig = {s["key"]: s for s in sec["B"]["signals"]}
    assert sig["momentum"]["t_ic"] > 2 and sig["momentum"]["spread_mean"] > 0
    assert sig["mu_excess"]["t_ic"] > 2
    assert abs(sig["random"]["t_ic"]) < 2.5
    assert data["meta"]["n_oos_dates"] >= 20 and data["meta"]["oos_start"] > data["meta"]["first_train_date"]
    # every OOS row is strictly after the first training block, barrier rows exclude missing futures
    assert sec["D"]["n"] > 0 and sec["D"]["excluded"]["no_future"] > 0
    assert abs(sec["D"]["tp"]["pred"] + sec["D"]["sl"]["pred"] + sec["D"]["time"]["pred"] - 1.0) < 1e-9
    fac = {s["key"]: s for s in sec["E"]["signals"]}
    assert set(fac) == {"momentum", "volatility", "trend", "price_composite"}
    assert fac["momentum"]["full"]["t_ic"] > 0
    weights = sec["G"]["weights"]
    assert sum(w["recommended"] for w in weights) == pytest.approx(1.0)
    llm = next(w for w in weights if w["pillar"] == "llm")
    assert llm["recommended"] == pytest.approx(llm["current"])  # untested pillar kept
    text = md_path.read_text(encoding="utf-8")
    for head in ("## 핵심 요약", "## A.", "## B.", "## C.", "## D.", "## E.", "## G.", "권장 필러 가중치"):
        assert head in text
    assert "## F." not in text and "앙상블" not in text  # section F was removed with the per-ticker ensemble
    assert "생존·선택 편향" in text and "밸류에이션·퀄리티" in text
    assert "생존 편향 진단(이 표본)" in text and "다중검정" in text and "상위−전체 t [" in text
    assert data["meta"]["universe_listed"] == 30 and not any("목록 상한" in n for n in data["notes"])
    # the reported total is the whole run: never below the load or below the stages it contains
    tm = data["timings"]
    stages = sum(v for k, v in tm.items() if k not in ("total", "load"))
    assert tm["total"] >= tm["load"] and tm["total"] == pytest.approx(stages + tm["load"], abs=0.5)
    # a short provider list is stated in the title (and, in run_evaluation, as the first note)
    short = ev.EvaluationResult(
        data["market"], {**data["meta"], "universe_listed": 20}, sec, data["notes"], data["timings"]
    )
    assert "유니버스 요청 30 → 제공 목록 20종목" in ev.render_markdown(short)
    # the same result labelled as a point-in-time run: PIT wording, no current-list survivorship paragraph
    pit = ev.EvaluationResult(
        data["market"],
        {**data["meta"], "universe_mode": "pit", "pit_start": "2021-01-01", "holdout_guard": {"xsh_names": 3}},
        sec,
        data["notes"],
        data["timings"],
    )
    pit_md = ev.render_markdown(pit)
    assert "시점별(PIT) 유니버스 30종목" in pit_md and "과거 승자만 남아" not in pit_md
    assert "홀드아웃 제외" in pit_md and "xsh_names 3" in pit_md
    assert data["meta"]["universe_mode"] == "current" and "현재 목록" in text

    # bad inputs are Korean errors with exit code 2, never a traceback
    assert runner.invoke(cli.app, ["evaluate", "--market", "JP"]).exit_code == 2
    assert runner.invoke(cli.app, ["evaluate", "--market", "KR", "--folds", "3,x"]).exit_code == 2
    bad = runner.invoke(cli.app, ["evaluate", "--market", "KR", "--universe", "5", "--out", str(tmp_path)])
    assert bad.exit_code == 2
    bad_mode = runner.invoke(
        cli.app, ["evaluate", "--market", "KR", "--universe-mode", "today", "--out", str(tmp_path)]
    )
    assert bad_mode.exit_code == 2


def test_run_evaluation_pit_mode_uses_the_pit_panel_and_the_holdout_guard(eval_settings, monkeypatch, tmp_path):
    """--universe-mode pit: PIT panel (members per month-end), holdout guard applied, `_pit` file names; the default
    mode follows pit.universe_mode."""
    from quantcio import holdout

    panel = xs_signal_panel(n_tickers=12, n_days=400, k=0.0, seed=1)
    order = sorted(panel.prices)[::-1]
    seen: dict = {}

    def fake_pit(market, settings, cache, *, today=None):
        seen["n"] = settings.rsmt.markets[market.value].universe_size
        return panel, order, {"pit_start": "2020-01-01", "pit_entities": 12, "pit_last_rebalance": "2026-05-29"}

    def fake_guard(p, settings):
        seen["guarded"] = True
        return p, {"xsh_names": 0, "xsh_rows": 0, "forward_rows": 0, "benchmark_rows": 0, "membership_rows": 0}

    def fake_eval(price_panel, settings, **kw):
        seen["meta"] = kw["meta"]
        return ev.EvaluationResult("KR", {}, {}, [], {"total": 1.0})

    written = {}
    monkeypatch.setattr(ev, "load_pit_eval_panel", fake_pit)
    monkeypatch.setattr(ev, "load_eval_panel", lambda *a, **k: pytest.fail("current-list panel loaded in pit mode"))
    monkeypatch.setattr(holdout, "guard_panel", fake_guard)
    monkeypatch.setattr(ev, "evaluate_panel", fake_eval)
    monkeypatch.setattr(ev, "write_outputs", lambda r, md, js: written.update(md=md, js=js))
    art = ev.run_evaluation(Market.KR, eval_settings, universe=20, years=4, out_dir=tmp_path, universe_mode="pit")
    assert seen["n"] == 20 and seen["guarded"]
    assert seen["meta"]["universe_mode"] == "pit" and seen["meta"]["pit_entities"] == 12
    assert written["md"].name == "evaluation_KR_20u_4y_pit.md" and art.json_path.name.endswith("_pit.json")
    assert ev.resolve_universe_mode(eval_settings, None) == "pit"
    assert (
        ev.resolve_universe_mode(
            eval_settings.model_copy(update={"pit": eval_settings.pit.model_copy(update={"universe_mode": "current"})}),
            None,
        )
        == "current"
    )
    with pytest.raises(ValueError):
        ev.resolve_universe_mode(eval_settings, "today")


def test_significantly_reversed_signals_are_called_out():
    rows = [
        {"label": "가격 팩터 합성", "full": {"t_ic": -2.3}},
        {"label": "추세 점수", "full": {"t_ic": -1.2}},
    ]
    note = ev._reverse_note(rows)
    assert "역방향" in note and "가격 팩터 합성(t=-2.30)" in note and "추세" not in note
    assert ev._reverse_note([{"label": "x", "t_ic": 2.5}], full=False) == ""
    assert ev._reverse_note([{"label": "x", "t_ic": float("nan")}], full=False) == ""

    assert "반대" in ev._slope_text(-0.5) and "과신" in ev._slope_text(0.7) and "몰림" in ev._slope_text(1.4)


# ------------------------------------------------------------------ audit regressions (2026-09-25)


def test_top_and_bottom_quintile_buckets_have_the_same_size():
    """n = 59 used to give a 12-name bottom and an 11-name top bucket (i*q//n); both are now round(n/q)."""
    assert [ev.quantile_bucket_size(n) for n in (10, 56, 59, 60, 135)] == [2, 11, 12, 12, 27]
    assert ev.quantile_bucket_size(3) == 1
    with pytest.raises(ValueError):
        ev.quantile_bucket_size(1)
    d = pd.Timestamp("2024-01-02")
    f = pd.DataFrame({"date": d, "score": np.arange(59.0), "y_e": np.arange(59.0)})
    r = ev.cross_sectional_ic(f, "score").iloc[0]
    assert r["top"] == pytest.approx(np.arange(47, 59).mean())  # 12 names
    assert r["bottom"] == pytest.approx(np.arange(0, 12).mean())  # 12 names
    assert r["top_excess"] == pytest.approx(np.arange(47, 59).mean() - 29.0)


def test_random_null_band_covers_the_spread_and_top_minus_all_t():
    """The permutation band now also bounds the spread t and the top-minus-all t, so a random row's
    single t (e.g. +2.59 on KR 150) can be read against its own chance range."""
    f = _xs_frame(60, 40, ic=0.0, seed=5)
    band = ev.random_null_band(f, "y_e", n_perm=400, seed=0)
    for key in ("t", "spread_t", "top_excess_t"):
        lo, hi = band[f"{key}_q025"], band[f"{key}_q975"]
        assert -2.8 < lo < -1.4 and 1.4 < hi < 2.8, (key, lo, hi)
    # the band is the distribution of a random score's t: ~5% of fresh random scores fall outside it
    rng = np.random.default_rng(1)
    outside = 0
    for _ in range(80):
        st = ev.ranking_stats(ev.cross_sectional_ic(f.assign(rnd=rng.random(len(f))), "rnd"), 20)
        outside += not (band["top_excess_t_q025"] <= st["top_excess_t"] <= band["top_excess_t_q975"])
    assert outside <= 12  # 80 x 5% = 4 expected
    assert math.isnan(ev.random_null_band(f.iloc[:30], "y_e", n_perm=50)["spread_t_q025"])  # < 3 dates


def _layout(k: int, splits: dict[int, int]) -> dict:
    return {"folds": k, "available": True, "splits_by_fold": splits}


def test_split_summary_does_not_claim_all_folds_when_another_layout_split():
    """KR 60 x 7y: the primary 4-fold layout never split but the 3-fold (fold 3) and 5-fold (fold 4)
    layouts did once each; the headline said '모든 CV 폴드에서 ... 분할이 없었음'."""
    sa = {
        "primary_folds": 4,
        "layouts": [_layout(3, {1: 0, 2: 0, 3: 1}), _layout(4, {1: 0, 2: 0, 3: 0, 4: 0}), _layout(5, {4: 1, 5: 0})],
    }
    line = ev.split_summary_line(sa)
    assert line.startswith("기준 4폴드 배치의 모든 CV 폴드에서 트리 분할 없음")
    assert "3폴드 배치 폴드3 1회" in line and "5폴드 배치 폴드4 1회" in line and "모든 폴드 배치" not in line
    none = {"primary_folds": 4, "layouts": [_layout(3, {1: 0}), _layout(4, {1: 0, 2: 0})]}
    assert ev.split_summary_line(none).startswith("평가한 모든 폴드 배치(3, 4)의 모든 CV 폴드")
    assert ev.split_summary_line({"primary_folds": 4, "layouts": [_layout(4, {1: 0, 2: 2})]}) == ""


def _barrier_section(slope_tp: float, slope_sl: float, gap: float = 0.01) -> dict:
    return {
        "n": 3000,
        "tp": {"pred": 0.48, "real": {"hat": 0.48 - gap}},
        "sl": {"pred": 0.49, "real": {"hat": 0.49 + gap}},
        "calibration_tp": {"slope": slope_tp},
        "calibration_sl": {"slope": slope_sl},
    }


def test_barrier_verdict_needs_per_row_calibration_not_just_the_average():
    """KR/US: mean P(TP)/P(SL) matched the realised rates within 3pp but the recalibration slopes were
    0.47/0.72 (KR 60), -0.04/-0.34 (US 60) with AUC ~0.5 - reported as '보정 양호' before."""
    assert ev.barrier_verdict(_barrier_section(1.0, 0.9)) == (ev.CAL_OK, True, True)
    assert ev.barrier_verdict(_barrier_section(0.47, 0.72)) == (ev.CAL_LEVEL, True, False)
    assert ev.barrier_verdict(_barrier_section(-0.04, -0.34))[0] == ev.CAL_LEVEL
    assert ev.barrier_verdict(_barrier_section(1.0, 1.0, gap=0.05)) == (ev.CAL_BAD, False, True)
    assert ev.barrier_verdict(_barrier_section(float("nan"), 1.0))[0] == ev.CAL_LEVEL
    # the bottom line no longer calls level-only barrier probabilities a risk-management basis
    sections = {"A": {"layouts": []}}
    tiers = {"factor": ev.TIER_NONE, "ml": ev.TIER_NONE, "llm": ev.TIER_UNTESTABLE}
    lines = ev._bottom_line(sections, tiers, False, False, _barrier_section(0.47, 0.72))
    assert not any("손절·익절 선도달 확률(평균 수준과" in ln for ln in lines)
    assert any("평균 빈도만" in ln and "근거로는 부족" in ln for ln in lines)
    ok = ev._bottom_line(sections, tiers, False, True, _barrier_section(1.0, 1.0))
    assert any("손절·익절 선도달 확률(평균 수준과 종목별 순서 모두" in ln and "손절·익절 설정" in ln for ln in ok)
    # a better-than-root distribution alone supports VaR / Kelly sizing, not the stop / take-profit levels
    dist_only = ev._bottom_line(sections, tiers, True, False, _barrier_section(0.47, 0.72))
    risk = next(ln for ln in dist_only if ln.startswith("표본 외에서 확인된 항목"))
    assert "VaR, 켈리 상한" in risk and "손절·익절 설정" not in risk


def test_survivorship_diagnostic_measures_drift_that_grows_with_lookback():
    """Selection on TODAY's market cap: the further back, the more today's winners led (synthetic)."""
    asof = pd.Timestamp("2026-09-23")
    dates = pd.bdate_range("2019-01-01", "2026-08-01", freq="20B")
    rng = np.random.default_rng(0)
    rows = []
    for d in dates:
        back = (asof - d).days / 365.25
        for t in range(40):
            rows.append({"date": d, "ticker": t, "y_e": 0.002 * back + rng.normal(0, 0.02)})
    train = pd.DataFrame(rows)
    oos = pd.DataFrame({"mu_e": [0.01, 0.01], "base_mu_e": [0.008, 0.008], "y_e": [0.0, 0.002]})
    out = ev.survivorship_diagnostic(train, oos, asof, 20)
    assert out["n_dates"] == len(dates)
    assert out["slope_per_year_back"] == pytest.approx(0.002 * 252 / 20, rel=0.25) and out["slope_t"] > 3
    assert out["mean_ye_annual"] > 0 and out["mean_ye_t"] > 3
    assert out["oos_base_mu_e_annual"] == pytest.approx(0.008 * 252 / 20)
    assert out["oos_realised_ye_annual"] == pytest.approx(0.001 * 252 / 20)
    flat = ev.survivorship_diagnostic(train.assign(y_e=rng.normal(0, 0.08, len(train))), oos, asof, 20)
    assert abs(flat["slope_t"]) < 3
    assert math.isnan(ev.survivorship_diagnostic(train.iloc[0:0], oos, asof, 20)["slope_t"])


def test_universe_shortfall_is_stated_not_silent():
    """US --universe 150 ran on the static 107-name list with '실패 0' and no mention of the other 43."""
    note = ev.universe_shortfall_note(Market.US, 150, 107)
    assert "150" in note and "107" in note and "43종목" in note and "로드 실패가 아님" in note
    assert ev.universe_shortfall_note(Market.KR, 60, 60) == ""


def test_multiple_testing_summary_and_zero_pass_binomial_text():
    sections = {
        "B": {
            "signals": [
                {"key": "mu_excess", "t_ic": 0.8, "spread_t": 0.3, "top_excess_t": -0.1},
                {"key": "random", "t_ic": 3.0, "spread_t": 0.2, "top_excess_t": 2.6},  # excluded (it is the null)
            ],
            "by_layout": [
                {"folds": 3, "available": True, "mu_excess": {"t_ic": 1.6}, "prob_outperform": {"t_ic": 0.2}}
            ],
        },
        "E": {
            "signals": [
                {
                    "key": "price_composite",
                    "full": {"t_ic": -2.22, "spread_t": -1.95},
                    "oos_period": {"t_ic": float("nan")},
                }
            ]
        },
    }
    mt = ev.multiple_testing_summary(sections)
    assert mt["n_t"] == 7 and mt["n_abs_t_ge_2"] == 1 and mt["expected_by_chance"] == pytest.approx(0.35)
    assert mt["hits"] == ["E price_composite IC t=-2.22"]
