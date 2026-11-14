"""PIT factor panel (quantcio.backtest.factors / sectors): offline, synthetic universes only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcio.backtest.factors import (
    FACTORS,
    add_zscores,
    compute_factor_panel,
    panel_coverage,
    uniform_rank,
    us_market_caps,
)
from quantcio.backtest.sectors import (
    OTHER,
    kr_sector_panel,
    ksic_to_sector,
    sector_coverage,
    sic_to_gics,
    us_entity_sectors,
)
from quantcio.config import load_settings
from quantcio.data.pit import FUNDAMENTAL_COLUMNS, PITFundamentals, PITUniverse
from quantcio.models import PRICE_COLUMNS, Market


def make_bt_universe(
    n_ent: int = 30,
    n_months: int = 30,
    seed: int = 0,
    n_sectors: int = 3,
    market: Market = Market.KR,
    start: str = "2019-01-01",
) -> tuple[PITUniverse, PITFundamentals, pd.DataFrame]:
    """Synthetic PIT universe (random-walk prices, monthly membership = everyone, meta sectors), quarterly PIT
    fundamentals (available 45 days after period end) and wide market caps (close x fixed shares)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_months * 21 + 170)
    s = pd.Series(idx, index=idx)
    month_ends = pd.DatetimeIndex(s.groupby(idx.to_period("M")).max().to_numpy())
    month_ends = month_ends[(month_ends >= idx[140]) & (month_ends < idx[-5])]
    ents = [f"CIK{k:010d}" if market is Market.US else f"{k:06d}" for k in range(n_ent)]
    prices: dict[str, pd.DataFrame] = {}
    shares = {}
    for k, e in enumerate(ents):
        r = rng.normal(0.0003, 0.02, len(idx))
        close = 1000.0 * (1 + k / 10) * np.exp(np.cumsum(r))
        vol = rng.lognormal(10, 0.3, len(idx))
        prices[e] = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": vol}, index=idx
        )[PRICE_COLUMNS]
        shares[e] = float(rng.uniform(1e6, 1e7))
    caps = pd.DataFrame(
        {e: prices[e]["close"].reindex(month_ends).to_numpy() * shares[e] for e in ents}, index=month_ends
    )
    rows = []
    for d in month_ends:
        c = caps.loc[d].sort_values(ascending=False)
        for rank, (e, v) in enumerate(c.items(), start=1):
            rows.append(
                {
                    "date": d,
                    "entity": e,
                    "ticker": e.removeprefix("CIK")[-6:],
                    "name": f"N{e}",
                    "rank": rank,
                    "rank_value": float(v),
                }
            )
    membership = pd.DataFrame(rows)
    meta = pd.DataFrame(
        {
            "entity": ents,
            "ticker": [e.removeprefix("CIK")[-6:] for e in ents],
            "name": [f"N{e}" for e in ents],
            "sector": [f"S{k % n_sectors}" for k in range(len(ents))],
            "delisted": False,
        }
    )
    frows = []
    qends = pd.date_range(pd.Timestamp(start) - pd.Timedelta(400, "D"), idx[-1], freq="QE")
    for e in ents:
        base = float(caps[e].mean())
        for pe in qends:
            frows.append(
                {
                    "entity": e,
                    "period_end": pe,
                    "period_type": "Q",
                    "available_from": pe + pd.Timedelta(45, "D"),
                    "currency": "KRW",
                    "revenue_ttm": base,
                    "op_income_ttm": base * 0.1,
                    "ni_parent_ttm": base * float(rng.normal(0.06, 0.05)),
                    "equity_parent": base * float(rng.uniform(0.3, 1.5)),
                    "shares_out": shares[e],
                    "source": "synthetic",
                    "restated_later": False,
                }
            )
    fund = PITFundamentals(market=market, panel=pd.DataFrame(frows)[FUNDAMENTAL_COLUMNS])
    bench = prices[ents[0]].copy()
    uni = PITUniverse(
        market=market,
        n=n_ent,
        membership=membership,
        prices=prices,
        benchmarks={"BM": bench},
        entity_benchmark={e: "BM" for e in ents},
        meta=meta,
    )
    return uni, fund, caps


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def synth():
    return make_bt_universe()


def test_panel_columns_and_momentum_definition(settings, synth):
    uni, fund, caps = synth
    p = compute_factor_panel(uni, fund, settings, None, market_caps=caps)
    for c in ("date", "entity", "ticker", "sector", "per", "pbr", "ep", "bp", "volume_trend", "mom_3m", "mom_6m"):
        assert c in p
    for f in FACTORS:
        assert f"z_{f}" in p and f"zr_{f}" in p
    t = sorted(p["date"].unique())[10]
    e = "000005"
    px = uni.prices[e]["close"]
    i = px.index.get_loc(t)
    row = p[(p["date"] == t) & (p["entity"] == e)].iloc[0]
    assert row["mom_3m"] == pytest.approx(px.iloc[i] / px.iloc[i - 63] - 1)
    assert row["mom_6m"] == pytest.approx(px.iloc[i] / px.iloc[i - 126] - 1)
    dv = uni.prices[e]["close"] * uni.prices[e]["volume"]
    assert row["volume_trend"] == pytest.approx(
        np.log(dv.iloc[i - 19 : i + 1].mean() / dv.iloc[i - 119 : i + 1].mean())
    )
    # sector momentum = equal-weight mean of the sector's 3M momentum
    g = p[(p["date"] == t) & (p["sector"] == row["sector"])]
    assert row["sector_mom_3m"] == pytest.approx(g["mom_3m"].mean())


def test_value_uses_pit_market_cap_and_available_from(settings, synth):
    uni, fund, caps = synth
    p = compute_factor_panel(uni, fund, settings, None, market_caps=caps)
    t = sorted(p["date"].unique())[12]
    e = "000003"
    fp = fund.panel[(fund.panel["entity"] == e) & (fund.panel["available_from"] <= t)]
    last = fp.sort_values(["available_from", "period_end"]).iloc[-1]
    row = p[(p["date"] == t) & (p["entity"] == e)].iloc[0]
    mcap = caps.loc[t, e]
    assert row["market_cap"] == pytest.approx(mcap)
    if last["ni_parent_ttm"] > 0:
        assert row["ep"] == pytest.approx(last["ni_parent_ttm"] / mcap)
        assert row["per"] == pytest.approx(mcap / last["ni_parent_ttm"])
    else:
        assert np.isnan(row["ep"]) and bool(row["ep_negative"]) and np.isnan(row["per"])
    assert row["bp"] == pytest.approx(last["equity_parent"] / mcap)


def test_shuffling_future_fundamentals_and_prices_does_not_change_past_signals(settings, synth):
    uni, fund, caps = synth
    base = compute_factor_panel(uni, fund, settings, None, market_caps=caps)
    t0 = sorted(base["date"].unique())[15]
    rng = np.random.default_rng(7)
    fp = fund.panel.copy()
    fut = fp["available_from"] > t0
    fp.loc[fut, "ni_parent_ttm"] = rng.permutation(fp.loc[fut, "ni_parent_ttm"].to_numpy()) * -3
    fp.loc[fut, "equity_parent"] = rng.permutation(fp.loc[fut, "equity_parent"].to_numpy()) * 7
    prices = {}
    for e, px in uni.prices.items():
        px = px.copy()
        later = px.index > t0 + pd.Timedelta(40, "D")  # beyond the holding period of t0
        px.loc[later, "close"] = px.loc[later, "close"].to_numpy()[::-1]
        prices[e] = px
    uni2 = PITUniverse(uni.market, uni.n, uni.membership, prices, uni.benchmarks, uni.entity_benchmark, uni.meta)
    shuffled = compute_factor_panel(uni2, PITFundamentals(fund.market, fp), settings, None, market_caps=caps)
    cols = ["ep", "bp", "per", "pbr", "mom_3m", "mom_6m", "volume_trend", *[f"z_{f}" for f in FACTORS]]
    a = base[base["date"] <= t0][cols].reset_index(drop=True)
    b = shuffled[shuffled["date"] <= t0][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    # and the future really changed
    assert not base[base["date"] > t0 + pd.Timedelta(60, "D")]["ep"].equals(
        shuffled[shuffled["date"] > t0 + pd.Timedelta(60, "D")]["ep"]
    )


def test_negative_earnings_rank_last_and_sector_neutral(settings, synth):
    uni, fund, caps = synth
    p = compute_factor_panel(uni, fund, settings, None, market_caps=caps)
    neg = p[p["ep_negative"]]
    assert len(neg) > 0 and neg["ep"].isna().all() and neg["per"].isna().all()
    for (_, _grp), g in p.groupby(["date", "sector_group"]):
        z = g["z_value_ep"].dropna()
        if g["ep_negative"].any() and (~g["ep_negative"] & g["ep"].notna()).any():
            assert z[g["ep_negative"]].max() < z[~g["ep_negative"] & g["ep"].notna()].min()
    # sector-neutral z spans (-1, 1) inside each sector: symmetric ranks -> mean 0
    for _, g in p.groupby(["date", "sector_group"]):
        z = g["z_mom_3m"].dropna()
        assert abs(z.mean()) < 1e-9 and z.max() < 1 and z.min() > -1


def test_small_sectors_merge_into_other(settings):
    panel = pd.DataFrame(
        {
            "date": pd.Timestamp("2020-01-31"),
            "entity": [f"e{i}" for i in range(12)],
            "sector": ["A"] * 8 + ["B"] * 3 + [OTHER],
        }
    )
    for c in ("ep", "bp", "volume_trend", "mom_3m", "mom_6m", "sector_mom_3m"):
        panel[c] = np.arange(12, dtype=float)
    panel["ep_negative"] = False
    panel["bp_negative"] = False
    out = add_zscores(panel, sector_neutral=True, min_names=5)
    assert (out.loc[out["sector"] == "B", "sector_group"] == OTHER).all()
    assert (out.loc[out["sector"] == "A", "sector_group"] == "A").all()
    other = out[out["sector_group"] == OTHER]
    assert len(other) == 4 and abs(other["z_mom_3m"].mean()) < 1e-12
    flat = add_zscores(panel, sector_neutral=False, min_names=5)
    assert np.allclose(flat["z_mom_3m"], flat["zr_mom_3m"])


def test_uniform_rank_bounds_and_ties():
    z = uniform_rank(pd.Series([1.0, 2.0, 2.0, np.nan, -np.inf]))
    assert np.isnan(z.iloc[3])
    assert z.iloc[4] == z.min() and z.iloc[1] == z.iloc[2]
    assert z.max() < 1 and z.min() > -1


def test_fwd_ret_next_session_entry_and_delisting_terminal(settings):
    uni, fund, caps = make_bt_universe(n_ent=12, n_months=14, seed=3)
    dates = sorted(uni.membership["date"].unique())
    t, t1 = dates[5], dates[6]
    # entity 000001 stops trading mid-month after t (delisted; KR frame ends at its last 정리매매 session)
    e = "000001"
    px = uni.prices[e]
    cut = px.index[px.index > t][8]
    uni.prices[e] = px[px.index <= cut]
    uni.meta.loc[uni.meta["entity"] == e, "delisted"] = True
    mem = uni.membership
    uni.membership = mem[~((mem["entity"] == e) & (mem["date"] > t))].reset_index(drop=True)
    p = compute_factor_panel(uni, fund, settings, None, market_caps=caps)
    row = p[(p["date"] == t) & (p["entity"] == e)].iloc[0]
    entry = uni.prices[e]["close"][uni.prices[e].index > t].iloc[0]
    assert row["fwd_status"] == "delisted"
    assert row["fwd_ret"] == pytest.approx(uni.prices[e]["close"].iloc[-1] / entry - 1)
    # a normal name: close of the first session after t -> close on t1
    e2 = "000002"
    r2 = p[(p["date"] == t) & (p["entity"] == e2)].iloc[0]
    c = uni.prices[e2]["close"]
    assert r2["fwd_status"] == "ok"
    assert r2["fwd_ret"] == pytest.approx(c.asof(t1) / c[c.index > t].iloc[0] - 1)
    # the last date has no forward return
    last = p[p["date"] == dates[-1]]
    assert last["fwd_ret"].isna().all() and (last["fwd_status"] == "last_period").all()
    # US-style terminal mark on the frame basis replaces the last close
    uni.meta["terminal_date"] = pd.NaT
    uni.meta["terminal_price"] = np.nan
    uni.meta["terminal_basis"] = None
    k = uni.meta.index[uni.meta["entity"] == e][0]
    uni.meta.loc[k, ["terminal_date", "terminal_price", "terminal_basis"]] = [cut + pd.Timedelta(3, "D"), 1.0, "frame"]
    p2 = compute_factor_panel(uni, fund, settings, None, market_caps=caps)
    row2 = p2[(p2["date"] == t) & (p2["entity"] == e)].iloc[0]
    assert row2["fwd_status"] == "delisted_terminal"
    assert row2["fwd_ret"] == pytest.approx(1.0 / entry - 1)
    cov = panel_coverage(p2)
    assert cov["fwd_status"]["delisted_terminal"] == 1
    assert cov["member_months"] == len(p2)


def test_us_market_caps_drop_imputed_ranks(settings):
    uni, fund, _ = make_bt_universe(n_ent=10, n_months=8, market=Market.US)
    e = uni.membership["entity"].iloc[0]
    fp = fund.panel.copy()
    fp.loc[fp["entity"] == e, "shares_out"] = np.nan
    wide = us_market_caps(uni, PITFundamentals(Market.US, fp), settings)
    assert wide[e].isna().all()
    other = [c for c in wide.columns if c != e][0]
    m = uni.membership[uni.membership["entity"] == other].set_index("date")["rank_value"]
    assert np.allclose(wide[other].dropna().to_numpy(), m.reindex(wide[other].dropna().index).to_numpy())
    assert wide.attrs["no_shares_member_months"] == int((uni.membership["entity"] == e).sum())


def test_ksic_and_sic_mappings():
    assert ksic_to_sector("261") == "반도체"
    assert ksic_to_sector("264") == "IT·전자"
    assert ksic_to_sector("649") == "금융"
    assert ksic_to_sector("701") == "제약·바이오"
    assert ksic_to_sector("702") == "산업재"
    assert ksic_to_sector("") is None and ksic_to_sector("999") is None
    assert sic_to_gics("3674") == "Information Technology"
    assert sic_to_gics(2834) == "Health Care"
    assert sic_to_gics("6798") == "Real Estate"
    assert sic_to_gics("6021") == "Financials"
    assert sic_to_gics("1311") == "Energy"
    assert sic_to_gics("5331") == "Consumer Staples"
    assert sic_to_gics(None) is None and sic_to_gics("0") is None


def test_us_entity_sectors_wiki_then_sic(settings):
    sec, src, _ = us_entity_sectors(
        settings,
        ["CIK0000000001", "CIK0000000002", "CIK0000000003"],
        wiki={1: "Information Technology"},
        sic={2: "2834", 3: None},
    )
    assert sec == {"CIK0000000001": "정보기술", "CIK0000000002": "헬스케어", "CIK0000000003": OTHER}
    assert src == {"CIK0000000001": "wiki_gics", "CIK0000000002": "sec_sic", "CIK0000000003": "none"}


def test_kr_sector_panel_asof_backfill_and_counts():
    bulk = pd.DataFrame(
        {
            "code": ["000001", "000001", "000002"],
            "period_end": pd.to_datetime(["2018-12-31", "2019-12-31", "2021-12-31"]),
            "ksic": ["261", "212", "303"],
            "ksic_name": ["반도체 제조업", "의약품 제조업", "자동차 부품"],
        }
    )
    pairs = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-06-28", "2020-06-30", "2019-06-28", "2019-06-28"]),
            "entity": ["000001", "000001", "000002#1", "000009"],
        }
    )
    out = kr_sector_panel(pairs, bulk, pd.DataFrame({"code": ["000009"], "industry": ["반도체 제조업"]}))
    assert out["sector"].tolist() == ["반도체", "제약·바이오", "자동차", "반도체"]
    assert out["sector_source"].tolist() == ["dart_ksic", "dart_ksic", "dart_ksic_backfill", "kind_current"]
    none = kr_sector_panel(pairs.assign(entity=["x1", "x2", "x3", "x4"]), bulk)
    assert (none["sector"] == OTHER).all()
    cov = sector_coverage(none)
    assert cov["unclassified_member_months"] == 4 and cov["by_year"]["2019"]["unclassified"] == 3


def test_kr_halt_at_period_end_is_charged_at_next_trade(settings):
    uni, fund, caps = make_bt_universe(n_ent=12, n_months=14, seed=4)
    dates = sorted(uni.membership["date"].unique())
    t, t1 = dates[4], dates[5]
    e = "000007"
    px = uni.prices[e].copy()
    after = px.index[px.index > t]
    halt = (px.index > after[5]) & (px.index <= t1 + pd.Timedelta(20, "D"))
    px.loc[halt, "volume"] = 0.0
    px.loc[halt, "close"] = px["close"][~halt & (px.index <= after[5])].iloc[-1]
    resume = px.index[px.index > t1 + pd.Timedelta(20, "D")][0]
    px.loc[resume, "close"] = px.loc[after[5], "close"] * 0.1  # 정리매매-like crash when trading resumes
    uni.prices[e] = px
    p = compute_factor_panel(uni, fund, settings, None, market_caps=caps)
    row = p[(p["date"] == t) & (p["entity"] == e)].iloc[0]
    assert row["fwd_status"] == "halted_exit_next_trade"
    assert row["fwd_ret"] == pytest.approx(px.loc[resume, "close"] / px["close"][after[0]] - 1)
    # halted through the whole next holding period: cannot be bought
    nxt = p[(p["date"] == t1) & (p["entity"] == e)].iloc[0]
    assert nxt["fwd_status"] in ("halted_no_entry", "halted_exit_next_trade", "ok")
