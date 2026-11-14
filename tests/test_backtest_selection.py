"""Factor-model candidate set, RSMT-style IC tree and selection rule (quantcio.backtest.candidates / tree).

Offline, synthetic panels only (docs/BACKTEST_MACRO.md section 3.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_backtest_engine import _synthetic_run, make_panel

from quantcio.backtest import candidates as cand
from quantcio.backtest import engine
from quantcio.backtest.candidates import (
    CANDIDATES,
    N_PARAMS,
    build_inputs,
    candidate_weights,
    compare_candidates,
    ridge_fit,
    select_candidate,
)
from quantcio.backtest.engine import load_cached_backtest, market_state_vars, score_tickers
from quantcio.backtest.factors import FACTORS
from quantcio.backtest.report import render_markdown
from quantcio.backtest.tree import fit_ic_tree, simes
from quantcio.config import load_settings
from quantcio.models import Market


@pytest.fixture(scope="module")
def settings():
    s = load_settings()
    s.backtest.model.min_months = 12
    return s


def _ar1(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    x = np.zeros(n)
    e = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def _regime_panel(n_dates: int = 72, n: int = 60, seed: int = 11):
    """mom_3m pays when the market-vol variable is high and reverses when it is low; macro regime = random."""
    panel = make_panel(n_dates=n_dates, n=n, beta=0.0, seed=seed)
    dates = sorted(panel["date"].unique())
    rng = np.random.default_rng(seed)
    vol = pd.Series(_ar1(rng, n_dates, 0.8), index=dates)
    sv = pd.DataFrame({"mkt_ret_60d": rng.normal(size=n_dates), "mkt_vol_60d": vol.to_numpy()}, index=dates)
    sign = np.where(vol > vol.median(), 1.0, -1.0)
    panel["fwd_ret"] = panel["fwd_ret"] + 0.03 * panel["z_mom_3m"] * panel["date"].map(pd.Series(sign, index=dates))
    regime = pd.DataFrame({"date": dates, "region": "kr", "score": rng.normal(0, 0.4, n_dates)})
    return panel, sv, regime


# --------------------------------------------------------------------------------------------- look-ahead


@pytest.mark.parametrize("name", CANDIDATES)
def test_every_candidate_is_look_ahead_free(settings, name):
    panel, sv, regime = _regime_panel()
    dates = sorted(panel["date"].unique())
    inp = build_inputs(panel, settings, regime, "kr", split_vars=sv)
    w, _ = candidate_weights(name, inp)
    assert list(w.index) == inp.model_dates and w.index[0] == dates[12]  # same first model date for every candidate
    assert np.allclose(w.abs().sum(axis=1), 1.0)
    t = dates[60]
    # rewrite every return from t on, and every split / regime value after t
    fut = panel.copy()
    later = fut["date"] >= t
    fut.loc[later, "fwd_ret"] = np.random.default_rng(5).normal(0, 0.3, int(later.sum()))
    sv2 = sv.copy()
    sv2.loc[sv2.index > t] = np.random.default_rng(6).normal(size=(int((sv2.index > t).sum()), sv2.shape[1]))
    reg2 = regime.copy()
    reg2.loc[reg2["date"] > t, "score"] = -reg2.loc[reg2["date"] > t, "score"] + 3.0
    w2, _ = candidate_weights(name, build_inputs(fut, settings, reg2, "kr", split_vars=sv2))
    pd.testing.assert_frame_equal(w.loc[:t], w2.loc[:t])
    if name != "equal_weight":  # the fitted candidates do react to the new months after t
        assert not np.allclose(w.loc[w.index > t].to_numpy(), w2.loc[w2.index > t].to_numpy())


def test_rsmt_tree_walk_forward_learns_the_planted_regime(settings):
    panel, sv, regime = _regime_panel(n_dates=84, n=80)
    inp = build_inputs(panel, settings, regime, "kr", split_vars=sv)
    w, diag = candidate_weights("rsmt_tree", inp)
    assert diag["dates_with_split"] > 0 and diag["latest_tree"]["split"]["var"] == "mkt_vol_60d"
    # once split, the mom_3m weight follows the date's volatility regime
    late = [t for t in w.index if t >= inp.dates[60]]
    hi = [t for t in late if sv.at[t, "mkt_vol_60d"] > diag["latest_tree"]["split"]["threshold"]]
    lo = [t for t in late if t not in hi]
    assert w.loc[hi, "mom_3m"].mean() > w.loc[lo, "mom_3m"].mean() + 0.1


# --------------------------------------------------------------------------------------------- the tree


def test_ic_tree_splits_on_a_planted_regime_ic():
    right = 0
    for s in range(10):
        rng = np.random.default_rng(100 + s)
        n = 96
        vol = _ar1(rng, n, 0.9)
        ic = 0.05 * rng.standard_normal((n, 6))
        ic[:, 3] += np.where(vol > np.median(vol), 0.05, -0.05)
        tree = fit_ic_tree(ic, {"ret": _ar1(rng, n, 0.8), "vol": vol, "macro": _ar1(rng, n, 0.95)})
        right += tree.var == "vol"
        assert tree.left is None or (tree.left.n >= 24 and tree.right.n >= 24)
    assert right >= 9


def test_ic_tree_rarely_splits_on_noise_and_respects_limits():
    splits = 0
    for s in range(40):
        rng = np.random.default_rng(s)
        n = 96
        ic = np.column_stack([0.03 * _ar1(rng, n, 0.3) for _ in range(6)])  # serially correlated, no regime
        tree = fit_ic_tree(ic, {"ret": _ar1(rng, n, 0.8), "vol": _ar1(rng, n, 0.9), "macro": _ar1(rng, n, 0.95)})
        splits += tree.n_splits > 0
        assert tree.n_leaves <= 4
    assert splits <= 4  # nominal 5% (Simes over 3 variables)
    # fewer than 2 x min_leaf months: never tested
    small = fit_ic_tree(np.ones((40, 6)) + np.arange(40)[:, None], {"v": np.arange(40.0)})
    assert small.n_splits == 0 and small.tests == {}


def test_ic_tree_missing_split_values_and_routing():
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(size=n)
    ic = 0.03 * rng.standard_normal((n, 6))
    ic[:, 0] += np.where(x > 0, 0.08, -0.08)
    x_nan = x.copy()
    x_nan[:10] = np.nan  # unknown months stay in the parent only
    tree = fit_ic_tree(ic, {"x": x_nan})
    assert tree.var == "x" and tree.left.n + tree.right.n == n - 10
    assert tree.route({"x": 2.0}).mean[0] > 0 > tree.route({"x": -2.0}).mean[0]
    assert tree.route({"x": np.nan}) is tree and tree.route({}) is tree
    # leaf means are shrunk towards the parent: |leaf| < |raw leaf mean|
    leaf = tree.route({"x": 2.0})
    assert abs(leaf.mean[0] - tree.mean[0]) < abs(leaf.raw_mean[0] - tree.mean[0])
    assert simes([0.04, 0.5, 0.9]) == pytest.approx(0.12) and simes([]) == 1.0


# --------------------------------------------------------------------------------------------- ridge


def test_ridge_interpolates_between_ols_and_ic_means():
    rng = np.random.default_rng(0)
    blocks = []
    for _ in range(24):
        x = rng.uniform(-1, 1, (150, 3))
        x[:, 1] = 0.9 * x[:, 0] + 0.1 * rng.uniform(-1, 1, 150)  # collinear copy
        y = 0.5 * x[:, 0] + rng.normal(0, 1, 150)
        yr = cand.uniform_rank_array(y)
        blocks.append((x, yr))
    beta, alpha, cv = ridge_fit(blocks)
    assert alpha in cand.RIDGE_ALPHAS and set(cv) == set(cand.RIDGE_ALPHAS)
    a = sum(x.T @ x for x, _ in blocks)
    b = sum(x.T @ y for x, y in blocks)
    n = sum(len(y) for _, y in blocks)
    big = np.linalg.solve(a + 1e6 * n / 3 * np.eye(3), b)
    xy = b / np.abs(b).sum()
    assert np.allclose(big / np.abs(big).sum(), xy, atol=1e-3)  # lambda -> inf: proportional to X'y
    assert beta[0] > 0


# --------------------------------------------------------------------------------------------- selection


def _row(ic, t, available=True):
    return {"available": available, "pre": {"ic_mean": ic, "ic_t": t}}


def test_selection_rule_primary_ties_and_simplicity():
    rows = {"equal_weight": _row(0.010, 1.0), "ic_weighted": _row(0.030, 2.0), "ridge": _row(0.020, 3.0)}
    assert select_candidate(rows, N_PARAMS)[0] == "ic_weighted"
    # within 0.002 of the best IC: the higher IC t wins
    rows = {"ic_weighted": _row(0.030, 2.0), "ridge": _row(0.0285, 2.5), "fama_macbeth": _row(0.0270, 9.0)}
    assert select_candidate(rows, N_PARAMS)[0] == "ridge"
    # IC and t tied: the simpler (fewer parameters, then table order) wins
    rows = {"rsmt_tree": _row(0.031, 2.55), "ridge": _row(0.030, 2.5), "ic_weighted_60": _row(0.0295, 2.47)}
    assert select_candidate(rows, N_PARAMS)[0] == "ic_weighted_60"
    rows = {"ic_weighted_raw": _row(0.030, 2.5), "ic_weighted": _row(0.030, 2.5)}
    assert select_candidate(rows, N_PARAMS)[0] == "ic_weighted"
    # unavailable / NaN candidates are skipped; nothing available -> None
    rows = {"macro_conditional": _row(0.9, 9.0, available=False), "ic_weighted": _row(0.01, 1.0)}
    assert select_candidate(rows, N_PARAMS)[0] == "ic_weighted"
    assert select_candidate({"ridge": _row(float("nan"), 1.0)}, N_PARAMS)[0] is None
    rows = {"ridge": _row(0.03, float("nan")), "ic_weighted": _row(0.03, float("nan"))}  # no t: simplest wins
    assert select_candidate(rows, N_PARAMS)[0] == "ic_weighted"


def test_holdout_months_never_change_the_choice(settings):
    panel = make_panel(n_dates=72, n=100, beta=0.02, seed=4)
    s = settings.model_copy(deep=True)
    s.backtest.model.holdout_months = 18
    served, results = compare_candidates(panel, s, cost_bps=10.0)
    sel = served.selection
    assert sel["holdout_months"] == 18 and sel["pre_months"] + 18 == sel["common_months"]
    hold_start = pd.Timestamp(sel["holdout_period"][0])
    # scramble every return inside the holdout: the pre-holdout statistics and the choice cannot move
    fut = panel.copy()
    m = fut["date"] >= hold_start
    fut.loc[m, "fwd_ret"] = np.random.default_rng(1).normal(0, 0.2, int(m.sum()))
    served2, _ = compare_candidates(fut, s, cost_bps=10.0)
    for name in results:
        a, b = sel["rows"][name]["pre"], served2.selection["rows"][name]["pre"]
        assert a["ic_mean"] == pytest.approx(b["ic_mean"]) and a["ic_t"] == pytest.approx(b["ic_t"])
    assert served2.method == served.method == sel["selected"]
    assert (
        sel["rows"][served.method]["holdout"]["ic_mean"]
        != served2.selection["rows"][served.method]["holdout"]["ic_mean"]
    )
    # skill is the existing rule on the selected candidate's full OOS IC t
    assert served.skill_ok == (served.oos["ic_t"] >= s.backtest.model.skill_min_ic_t)


def test_fixed_method_serves_that_candidate(settings):
    s = settings.model_copy(deep=True)
    s.backtest.model.method = "equal_weight"
    served, _ = compare_candidates(make_panel(n_dates=48, n=80, beta=0.02, seed=2), s)
    assert served.method == "equal_weight" and served.selection["mode"] == "fixed"
    assert served.latest_weights["volume_trend"] == 0.0 and served.latest_weights["mom_3m"] == pytest.approx(0.2)
    # a fixed candidate that cannot run (no regime) falls back to the rule's choice
    s.backtest.model.method = "macro_conditional"
    served, _ = compare_candidates(make_panel(n_dates=48, n=80, beta=0.02, seed=2), s)
    assert served.method == served.selection["rule_winner"] != "macro_conditional"


# --------------------------------------------------------------------------------------------- serving


def test_serving_uses_the_selected_candidate(tmp_path, settings):
    res, s, cache, _ = _synthetic_run(tmp_path, settings)
    m = res.model
    sel = m["selection"]
    assert m["method"] == sel["selected"] and sel["mode"] == "auto"
    assert set(sel["rows"]) == set(CANDIDATES)
    assert res.model_result.method == m["method"]
    assert res.live["weights"] == pytest.approx(res.model_result.latest_weights)
    tick = str(res.latest["ticker"].iloc[0])
    fm = score_tickers(Market.US, [tick], s, cache)[tick]
    assert fm.weights == pytest.approx({f: round(v, 6) for f, v in res.model_result.latest_weights.items()})
    row = res.latest[res.latest["ticker"] == tick].iloc[0]
    assert fm.score == pytest.approx(sum(res.model_result.latest_weights[f] * row[f"z_{f}"] for f in FACTORS), abs=1e-5)
    summ = engine.report_summary(res)
    assert summ["model"]["method"] == m["method"] and len(summ["model"]["candidates"]) == len(CANDIDATES)
    text = render_markdown(res)
    assert "모델 비교" in text and "승자의 저주" in text and "홀드아웃" in text
    # the market split variables come from the benchmark and use closes <= t only
    assert "mkt_vol_60d" in (sel["rows"]["rsmt_tree"]["diagnostics"]["split_variables"])
    # the cache keys carry the candidate set / selection version
    k1 = engine._latest_key(Market.US, s)
    k2 = engine._backtest_key(Market.US, s, res.data_snapshot, None)
    assert load_cached_backtest(Market.US, s, cache) is not None
    old = engine.SELECTION_VERSION
    try:
        engine.SELECTION_VERSION = old + 1
        assert engine._latest_key(Market.US, s) != k1
        assert engine._backtest_key(Market.US, s, res.data_snapshot, None) != k2
        assert load_cached_backtest(Market.US, s, cache) is None
    finally:
        engine.SELECTION_VERSION = old
    s2 = s.model_copy(deep=True)
    s2.backtest.model.method = "ridge"
    assert engine._latest_key(Market.US, s2) != k1


def test_market_state_vars_are_point_in_time():
    from test_backtest_factors import make_bt_universe

    uni, _, _ = make_bt_universe(n_ent=10, n_months=24, seed=1, market=Market.US, start="2018-01-01")
    bench = next(iter(uni.benchmarks.values()))
    dates = sorted(uni.membership["date"].unique())
    mv = market_state_vars(uni, dates)
    t = pd.Timestamp(dates[10])
    close = bench["close"]
    c = close[close.index <= t]
    assert mv.at[t, "mkt_ret_60d"] == pytest.approx(c.iloc[-1] / c.iloc[-61] - 1)
    lr = np.log(c).diff().iloc[-60:]
    assert mv.at[t, "mkt_vol_60d"] == pytest.approx(lr.std() * np.sqrt(252))
    # later prices do not move the value at t
    uni.benchmarks = {
        k: v.assign(close=np.where(v.index > t, v["close"] * 3, v["close"])) for k, v in uni.benchmarks.items()
    }
    assert market_state_vars(uni, dates).at[t, "mkt_ret_60d"] == pytest.approx(mv.at[t, "mkt_ret_60d"])
    uni.benchmarks = {}
    assert market_state_vars(uni, dates) is None
