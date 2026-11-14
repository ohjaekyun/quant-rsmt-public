"""Walk-forward multi-factor model (quantcio.backtest.model): offline, synthetic panels only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_backtest_engine import make_panel

from quantcio.backtest.factors import FACTORS
from quantcio.backtest.model import (
    SIDE_MIN_T,
    cross_sectional_ic,
    fit_weights,
    pillar_side_test,
    regime_state,
    walk_forward_model,
)
from quantcio.config import load_settings


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def _settings(base, **model):
    s = base.model_copy(deep=True)
    for k, v in model.items():
        setattr(s.backtest.model, k, v)
    return s


def test_fit_weights_sign_constraint_normalisation_and_shrink():
    ic = pd.Series(
        {
            "value_ep": 0.03,
            "value_bp": -0.02,
            "volume_trend": -0.01,
            "mom_3m": 0.01,
            "mom_6m": 0.0,
            "sector_mom_3m": 0.02,
        }
    )
    w0 = fit_weights(ic, shrink=0.0)
    assert w0["value_bp"] == 0.0  # opposite of the a-priori (+) direction
    assert w0["volume_trend"] == pytest.approx(-0.01 / 0.07)  # no prior: keeps its sign
    assert w0["value_ep"] == pytest.approx(0.03 / 0.07)
    assert sum(abs(v) for v in w0.values()) == pytest.approx(1.0)
    w1 = fit_weights(ic, shrink=1.0)
    assert w1["value_bp"] == pytest.approx(1 / 6) and w1["volume_trend"] == pytest.approx(-1 / 6)
    wh = fit_weights(ic, shrink=0.5)
    assert wh["value_bp"] == pytest.approx(0.5 / 6 / sum(abs(0.5 * w0[f] + 0.5 * w1[f]) for f in FACTORS))
    assert sum(abs(v) for v in wh.values()) == pytest.approx(1.0)


def test_walk_forward_uses_only_past_months(settings):
    s = _settings(settings, min_months=12, lookback_months=36, method="ic_weighted")
    panel = make_panel(n_dates=40, n=80, seed=2)
    m = walk_forward_model(panel, s)
    dates = sorted(panel["date"].unique())
    assert m.weights.index[0] == dates[12]  # min_months past cross-sections before the first model
    t = dates[25]
    # the weights at t are the fit of the ICs of strictly earlier months
    ic = pd.DataFrame({f: cross_sectional_ic(panel, f"z_{f}") for f in FACTORS})
    expect = fit_weights(ic[ic.index < t].iloc[-36:].mean(), s.backtest.model.shrink_to_equal)
    for f in FACTORS:
        assert m.weights.loc[t, f] == pytest.approx(expect[f])
    # rewriting the returns of t and later must not change the weights (or scores) at t
    fut = panel.copy()
    later = fut["date"] >= t
    fut.loc[later, "fwd_ret"] = np.random.default_rng(9).normal(0, 0.3, int(later.sum()))
    m2 = walk_forward_model(fut, s)
    pd.testing.assert_series_equal(m.weights.loc[t], m2.weights.loc[t])
    a = m.scores[m.scores["date"] <= t].reset_index(drop=True)
    b = m2.scores[m2.scores["date"] <= t].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    # no model before min_months
    short = walk_forward_model(panel[panel["date"] < dates[10]], s)
    assert short.weights.empty and not short.skill_ok and short.latest_weights == {}


def test_planted_signal_gives_oos_skill(settings):
    s = _settings(settings, min_months=12, method="ic_weighted")
    m = walk_forward_model(make_panel(n_dates=60, n=150, beta=0.02, seed=4), s, cost_bps=10.0)
    assert m.skill_ok and m.oos["ic_t"] > 5 and m.oos["ic_mean"] > 0.05
    assert max(m.latest_weights, key=lambda f: abs(m.latest_weights[f])) == "mom_3m"
    assert m.oos["ls_net"]["ann_return"] > 0 and m.oos["lo_net"]["ann_return"] > 0
    d = m.to_dict()
    assert d["skill_ok"] is True and d["first_model_date"] is not None


def test_noise_panel_fails_skill_gate(settings):
    s = _settings(settings, min_months=12)
    m = walk_forward_model(make_panel(n_dates=48, n=100, beta=0.0, seed=1), s)
    assert not m.skill_ok and abs(m.oos["ic_t"]) < 3


def test_fama_macbeth_method(settings):
    s = _settings(settings, min_months=12, method="fama_macbeth")
    m = walk_forward_model(make_panel(n_dates=48, n=120, beta=0.02, seed=5), s)
    assert m.method == "fama_macbeth" and m.skill_ok
    assert max(m.latest_weights, key=lambda f: abs(m.latest_weights[f])) == "mom_3m"
    # fama_macbeth learns signs: a planted NEGATIVE slope gets a negative weight (no a-priori constraint)
    neg = make_panel(n_dates=48, n=120, beta=-0.02, seed=5)
    m2 = walk_forward_model(neg, s)
    assert (
        m2.latest_weights["mom_3m"] < 0 and max(m2.latest_weights, key=lambda f: abs(m2.latest_weights[f])) == "mom_3m"
    )


def test_macro_conditioned_candidate_wins_on_a_planted_regime(settings):
    s = _settings(settings, min_months=12)
    panel = make_panel(n_dates=72, n=120, beta=0.0, seed=6)
    dates = sorted(panel["date"].unique())
    rng = np.random.default_rng(1)
    state = pd.Series(np.where(rng.random(len(dates)) < 0.5, 1.0, -1.0), index=dates)
    # the planted factor pays in risk-on months and reverses in risk-off months
    panel["fwd_ret"] = panel["fwd_ret"] + 0.03 * panel["z_mom_3m"] * panel["date"].map(state)
    regime = pd.DataFrame({"date": dates, "region": "kr", "score": state.to_numpy() * 0.5})
    m = walk_forward_model(panel, s, regime, region="kr")
    assert m.macro is not None
    assert m.macro["conditional"]["ic_t"] > m.macro["unconditional"]["ic_t"]
    # the regime-aware candidates beat the unconditional ones; one of them is selected
    assert m.method in {"macro_conditional", "rsmt_tree"} and m.macro["used"] is (m.method == "macro_conditional")
    rows = m.selection["rows"]
    assert rows["macro_conditional"]["pre"]["ic_mean"] > rows["ic_weighted"]["pre"]["ic_mean"] + 0.05
    # without a regime history the macro candidate is reported as unavailable, never selected
    m2 = walk_forward_model(make_panel(n_dates=72, n=120, beta=0.02, seed=6), s, None, region="kr")
    assert m2.selection["rows"]["macro_conditional"]["available"] is False and m2.method != "macro_conditional"
    assert m2.macro is None


def test_regime_state_matches_calendar_month():
    dates = [pd.Timestamp("2024-06-28"), pd.Timestamp("2024-07-31"), pd.Timestamp("2024-09-30")]
    regime = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-06-30", "2024-07-31", "2024-06-30"]),
            "region": ["kr", "kr", "us"],
            "score": [0.3, -0.4, -0.9],
        }
    )
    st = regime_state(dates, regime, "kr")
    assert st.iloc[0] == 1.0 and st.iloc[1] == -1.0 and np.isnan(st.iloc[2])
    assert regime_state(dates, None, "kr").isna().all()


# ----------------------------------------------------------------------------- pillar side test


def _side_panel(
    slope_pos: float, slope_neg: float, months: int = 60, n: int = 80, seed: int = 0, start: str = "2018-01-31"
):
    """Scores and forward returns whose conditional mean is slope_pos·p+ + slope_neg·p- (p = 2·(pct − 0.5))."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=months, freq="ME")
    sc, pn = [], []
    for t in dates:
        score = rng.normal(size=n)
        pct = (pd.Series(score).rank() - 0.5) / n
        p = 2.0 * (pct.to_numpy() - 0.5)
        ret = slope_pos * np.maximum(p, 0) + slope_neg * np.minimum(p, 0) + rng.normal(0, 0.03, n)
        ents = [f"E{i}" for i in range(n)]
        sc.append(pd.DataFrame({"date": t, "entity": ents, "score": score}))
        pn.append(pd.DataFrame({"date": t, "entity": ents, "fwd_ret": ret}))
    return pd.concat(sc, ignore_index=True), pd.concat(pn, ignore_index=True), dates


def test_pillar_side_test_flags_the_weak_side_only_when_significant():
    sc, pn, _ = _side_panel(0.03, 0.0)  # the low scores carry nothing
    out = pillar_side_test(sc, pn)
    assert out["weak_side"] == "neg" and out["pre"]["t_diff"] >= SIDE_MIN_T
    assert out["pre"]["b_pos"] == pytest.approx(0.03, abs=0.006) and abs(out["pre"]["b_neg"]) < 0.006
    assert pillar_side_test(*_side_panel(0.0, 0.03)[:2])["weak_side"] == "pos"
    sym = pillar_side_test(*_side_panel(0.02, 0.02)[:2])
    assert sym["weak_side"] is None and abs(sym["pre"]["t_diff"]) < SIDE_MIN_T
    assert sym["pre"]["n"] == 60 and sym["holdout"] is None


def test_pillar_side_test_decides_on_pre_holdout_months_only():
    sc1, pn1, d1 = _side_panel(0.02, 0.02, months=36)
    sc2, pn2, d2 = _side_panel(0.04, 0.0, months=24, seed=1, start="2021-01-31")  # holdout: strongly asymmetric
    sc, pn = pd.concat([sc1, sc2], ignore_index=True), pd.concat([pn1, pn2], ignore_index=True)
    out = pillar_side_test(sc, pn, d2[0])
    assert out["pre"]["n"] == 36 and out["holdout"]["n"] == 24 and out["all"]["n"] == 60
    assert out["holdout"]["t_diff"] >= SIDE_MIN_T  # the holdout alone would flag the negative side ...
    assert out["weak_side"] is None  # ... but it is reported, never used for the verdict
