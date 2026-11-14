"""PIT factor backtest engine (quantcio.backtest.engine / report): offline, synthetic panels only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from test_backtest_factors import make_bt_universe

from quantcio.backtest.engine import (
    BacktestResult,
    _place,
    _score_outside,
    benjamini_hochberg,
    factor_table,
    load_cached_backtest,
    report_summary,
    run_factor_backtest,
    score_frame,
    score_tickers,
)
from quantcio.backtest.factors import FACTORS
from quantcio.backtest.model import _turnover, perf_stats, quantile_portfolios
from quantcio.backtest.report import write_backtest_report
from quantcio.config import load_settings
from quantcio.data.cache import DiskCache
from quantcio.models import FactorModelScore, Market


def make_panel(n_dates: int = 60, n: int = 120, beta: float = 0.02, planted: str = "mom_3m", seed: int = 0):
    """Synthetic factor panel: every factor is uniform noise except ``planted``, which drives fwd_ret."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=n_dates, freq="ME")
    parts = []
    for t in dates:
        df = pd.DataFrame({"date": t, "entity": [f"e{i:03d}" for i in range(n)]})
        df["sector"] = [f"S{i % 4}" for i in range(n)]
        df["sector_group"] = df["sector"]
        for f in FACTORS:
            x = rng.uniform(-1, 1, n)
            raw = {"value_ep": "ep", "value_bp": "bp"}.get(f, f)
            df[raw] = x
            df[f"z_{f}"] = x
            df[f"zr_{f}"] = x
        df["ep_negative"] = False
        df["bp_negative"] = False
        df["fwd_ret"] = beta * df[f"z_{planted}"] + rng.normal(0, 0.05, n)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def test_planted_factor_has_positive_ic_and_long_short():
    rows, curves = factor_table(make_panel(), q=5, cost_bps=10.0)
    by = {(r["factor"], r["variant"]): r for r in rows}
    hit = by[("mom_3m", "sector_neutral")]
    assert hit["ic_mean"] > 0.15 and hit["ic_t"] > 8 and hit["bh_significant"]
    assert hit["ls_net"]["ann_return"] > 0.1 and hit["ls_net"]["sharpe"] > 1
    assert hit["direction_ok"] is True
    for (f, _v), r in by.items():
        if f not in ("mom_3m",):
            assert abs(r["ic_t"]) < 3.5, (f, r["ic_t"])
    # sector momentum is evaluated only on the universe-wide ranking
    assert ("sector_mom_3m", "sector_neutral") not in by
    assert len(rows) == 11 and rows[0]["bh_tests"] == 11
    assert "mom_3m|sector_neutral|ls_net" in curves


def test_noise_only_panel_has_no_significant_factor():
    rows, _ = factor_table(make_panel(beta=0.0, seed=11), q=5, cost_bps=10.0)
    assert not any(r["bh_significant"] for r in rows)
    assert all(abs(r["ic_mean"]) < 0.03 for r in rows)


def test_turnover_and_cost_arithmetic():
    # drifted book: I +10%, J 0% -> weights 0.5238 / 0.4762; new book J, H -> one-way turnover 0.5238
    to = _turnover({"J": 0.5, "H": 0.5}, {"I": 0.5, "J": 0.5}, {"I": 0.10, "J": 0.0})
    assert to == pytest.approx(0.5 * (0.55 / 1.05 + abs(0.5 - 0.5 / 1.05) + 0.5))
    assert _turnover({"A": 1.0}, {}, {}) == pytest.approx(0.5)  # initial build from cash
    assert _turnover({"A": 1.0}, {"A": 1.0}, {"A": 0.3}) == pytest.approx(0.0)
    # 10 names, q=5: bottom = a0,a1 / top = a8,a9; period 2 swaps the top book completely
    d1, d2 = pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")
    ents = [f"a{i}" for i in range(10)]
    p1 = pd.DataFrame({"date": d1, "entity": ents, "sig": np.arange(10.0), "fwd_ret": np.linspace(-0.05, 0.04, 10)})
    sig2 = np.arange(10.0)
    sig2[[6, 7, 8, 9]] = [8.5, 9.5, 0.5, 1.5]  # a6, a7 become the top; a8 drops to the bottom book
    p2 = pd.DataFrame({"date": d2, "entity": ents, "sig": sig2, "fwd_ret": 0.01})
    qp = quantile_portfolios(pd.concat([p1, p2], ignore_index=True), "sig", q=5, cost_bps=100.0)
    r1, r2 = qp.iloc[0], qp.iloc[1]
    assert r1["to_long"] == pytest.approx(0.5) and r1["to_short"] == pytest.approx(0.5)
    assert r1["ls"] == pytest.approx(np.mean([0.03, 0.04]) - np.mean([-0.05, -0.04]))
    assert r1["ls_net"] == pytest.approx(r1["ls"] - (0.5 + 0.5) * 0.01)
    assert r2["to_long"] == pytest.approx(1.0)  # complete replacement of the top book
    assert r2["lo_net"] == pytest.approx(r2["lo"] - r2["to_long"] * 0.01)
    assert r2["univ"] == pytest.approx(0.01)


def test_benjamini_hochberg_step_up():
    assert benjamini_hochberg([0.001, 0.02, 0.03, 0.5], 0.10) == [True, True, True, False]
    assert benjamini_hochberg([0.2, 0.3, np.nan], 0.10) == [False, False, False]
    assert benjamini_hochberg([0.04, 0.2], 0.10) == [True, False]  # 0.04 <= 0.1 * 1/2


def test_perf_stats_known_series():
    s = perf_stats(pd.Series([0.1, -0.1, 0.1, -0.1]))
    assert s["ann_return"] == pytest.approx((0.99**2) ** 3 - 1)  # 4 months -> growth^(12/4)
    assert s["mdd"] == pytest.approx(0.99**2 / 1.1 - 1)
    assert s["win_rate"] == pytest.approx(0.5)
    assert perf_stats(pd.Series(dtype=float))["n"] == 0


def test_place_and_outside_scoring():
    vals = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert _place(vals, 2.5) == pytest.approx(2 * (2 + 0.5) / 5 - 1)
    assert _place(vals, -np.inf) == _place(vals, 0.0) < _place(vals, 1.0) < _place(vals, 1.5)
    latest = make_panel(n_dates=1, n=40)
    latest = score_frame(latest, {f: 1 / 6 for f in FACTORS})
    ex = {"ep": 5.0, "bp": 5.0, "volume_trend": 5.0, "mom_3m": 5.0, "mom_6m": 5.0, "sector": "S1"}
    z, score, pct = _score_outside(ex, latest, {f: 1 / 6 for f in FACTORS}, min_names=5, sector_neutral=True)
    assert all(v > 0.9 for f, v in z.items() if f != "sector_mom_3m")
    assert pct > 0.9 and score > 0.7
    ex_neg = {"ep": None, "ep_negative": True, "sector": "S2"}
    z2, _, _ = _score_outside(ex_neg, latest, {f: 1 / 6 for f in FACTORS}, min_names=5, sector_neutral=True)
    assert z2["value_ep"] < -0.9 and z2["mom_3m"] is None


def _synthetic_run(tmp_path, settings, market=Market.US, use_cache=True):
    uni, fund, caps = make_bt_universe(n_ent=40, n_months=40, seed=5, market=market, start="2018-01-01")
    cache = DiskCache(tmp_path / "cache")
    s = settings.model_copy(deep=True)
    s.backtest.us_start = "2018-01-01"
    s.backtest.kr_start = "2018-01-01"
    s.backtest.model.min_months = 12
    res = run_factor_backtest(
        market,
        s,
        cache,
        use_cache=use_cache,
        universe=uni,
        fundamentals=fund,
        market_caps=None if market is Market.US else caps,
    )
    return res, s, cache, uni


def test_run_factor_backtest_end_to_end_and_serving(tmp_path, settings):
    res, s, cache, uni = _synthetic_run(tmp_path, settings)
    assert isinstance(res, BacktestResult)
    blob = json.dumps(res.to_dict(), ensure_ascii=False)
    d = json.loads(blob)
    assert d["n_months"] == res.panel["date"].nunique()
    assert {r["variant"] for r in d["factor_table"]} == {"raw", "sector_neutral"}
    assert "ic_t" in d["model"]["oos"] and isinstance(d["model"]["skill_ok"], bool)
    assert d["coverage"]["member_months"] == len(res.panel)
    assert any("Benjamini" in n for n in d["notes"])
    # the live snapshot re-measures the last members after the last month-end
    assert res.live["as_of"] > res.live["rebalance_date"]
    assert res.latest["score"].notna().all()
    # cached: the pointer finds it without any data
    again = load_cached_backtest(Market.US, s, cache)
    assert again is not None and again.data_snapshot == res.data_snapshot
    tick = str(res.latest["ticker"].iloc[0])
    out = score_tickers(Market.US, [tick], s, cache)
    fm = out[tick]
    assert isinstance(fm, FactorModelScore)
    assert 0 <= fm.percentile <= 1 and -1 <= fm.pillar <= 1
    assert sum(fm.contributions.values()) == pytest.approx(fm.score, abs=1e-5)
    mult = 1.0 if fm.skill_ok else 0.35
    assert fm.pillar == pytest.approx(np.clip(2 * (fm.percentile - 0.5), -1, 1) * mult, abs=1e-5)
    assert set(fm.zscores) == set(FACTORS)
    # the pillar side test of the served model's OOS scores rides along (pre-holdout months decide)
    assert {"pre", "holdout", "all", "weak_side", "rule"} <= set(fm.side_test)
    assert fm.weak_side == fm.side_test["weak_side"] and fm.side_test["pre"]["n"] > 0
    json.dumps(fm.side_test)
    # month-end cross-section on request
    first_model = res.model_result.weights.index[3]
    old = score_tickers(Market.US, [tick], s, cache, as_of=first_model.date())
    assert old[tick].as_of == first_model.date()
    summ = report_summary(res)
    json.dumps(summ)
    assert len(summ["factors"]) == 11 and summ["model"]["skill_ok"] == res.model["skill_ok"]
    md, js = write_backtest_report(res, tmp_path / "reports")
    text = md.read_text(encoding="utf-8")
    assert "팩터별 결과" in text and "워크포워드" in text and "커버리지" in text
    assert json.loads(js.read_text(encoding="utf-8"))["market"] == "US"


def test_score_tickers_without_cache_returns_empty(tmp_path, settings):
    assert score_tickers(Market.KR, ["005930"], settings, DiskCache(tmp_path / "empty")) == {}


def test_kr_run_with_injected_market_caps(tmp_path, settings):
    res, *_ = _synthetic_run(tmp_path, settings, market=Market.KR)
    assert res.market == "KR" and res.cost_bps == pytest.approx(settings.backtest.cost_bps_kr)
    assert res.coverage["fwd_status"]["ok"] > 0
