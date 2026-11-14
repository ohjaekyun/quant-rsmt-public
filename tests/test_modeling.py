"""Stock-selection modeling (quantcio.modeling), RSMT-Q v2, PIT panel membership and the serving path (offline).

Synthetic panels with a PLANTED signal: y_e = a * x_sig + common date shock + noise, so a model that learns the
signal must reach a within-date AUC well above 0.5 while the momentum baseline (pure noise here) stays at 0.5.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import types
from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import make_prices
from sklearn.metrics import roc_auc_score

from quantcio.config import load_settings
from quantcio.data.base import PricePanel
from quantcio.data.cache import DiskCache
from quantcio.modeling import candidates as cmod
from quantcio.modeling import events as ev
from quantcio.modeling import features as feat
from quantcio.modeling import metrics as mt
from quantcio.modeling import selection as sel
from quantcio.modeling.dataset import ModelingDataset, _labels_for_entity, add_date_targets, grid_dates, members_on_grid
from quantcio.models import ExcessForecast, Fundamentals, Instrument, Market, MarketData

H = 20


@pytest.fixture(scope="module")
def msettings(tmp_path_factory):
    return load_settings(data={"cache_dir": str(tmp_path_factory.mktemp("mcache"))}, macro={"enabled": False})


# ------------------------------------------------------------------------------------------------ synthetic data


def synthetic_dataset(
    n_names: int = 40, years: int = 8, signal: float = 0.03, seed: int = 0, market: Market = Market.KR
) -> ModelingDataset:
    """Weekly + month-end grid, planted signal in x_sig (and in sx_ep for the factor candidates)."""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2015-01-01", periods=252 * years + 40)
    s = pd.Series(cal, index=cal)
    month_ends = pd.DatetimeIndex(s.groupby([cal.year, cal.month]).max().to_numpy())[:-1]
    grid = grid_dates(cal, month_ends, cal[0], weekly=True)
    grid = grid[grid <= cal[-1]]
    rows = []
    pos = pd.Series(np.arange(len(cal)), index=cal)
    ents = [f"{100000 + i:06d}" for i in range(n_names)]
    for d in grid:
        p = int(pos[d])
        te = cal[p + H] if p + H < len(cal) else pd.NaT
        common = rng.normal(0.0, 0.05)
        m = rng.normal(0.005, 0.04)
        x_sig = rng.normal(size=n_names)
        noise = rng.normal(0.0, 0.08, n_names)
        y = signal * x_sig + common + noise
        for i, e in enumerate(ents):
            rows.append(
                {
                    "date": d,
                    "entity": e,
                    "ticker": e,
                    "name": f"n{e}",
                    "sector": "A" if i % 2 else "B",
                    "sector_group": "A" if i % 2 else "B",
                    "is_me": d in month_ends,
                    "t_end": te,
                    "y_e": y[i] if not pd.isna(te) else np.nan,
                    "r": (y[i] + m) if not pd.isna(te) else np.nan,
                    "m": m,
                    "beta_s": 1.0,
                    "vol60": 0.3 + 0.1 * rng.random(),
                    "x_sig": x_sig[i],
                    "x_noise": rng.normal(),
                }
            )
    f = pd.DataFrame(rows)
    f["labelled"] = f["y_e"].notna()
    f = add_date_targets(f)
    # factor-candidate columns: sx_ep carries the signal, the others are noise
    for c in ["x_mom_12_1", "sx_bp", "sx_volume_trend", "sx_mom_3m", "sx_mom_6m", "x_sector_mom_3m", "sx_ret_1m"]:
        f[c] = rng.normal(size=len(f))
    f["sx_ep"] = f["x_sig"] + rng.normal(0, 0.5, len(f))
    f["x_ret_1w"] = rng.normal(size=len(f))
    f["x_ret_1m"] = rng.normal(size=len(f))
    dates = pd.DatetimeIndex(sorted(f["date"].unique()))
    reg = pd.DataFrame({"r_mkt_dd": rng.uniform(-0.3, 0.0, len(dates))}, index=dates)
    f = f.merge(reg, left_on="date", right_index=True)
    dfr = reg.copy()
    dfr["m_primary"] = [f.loc[f["date"] == d, "m"].iloc[0] if (f["date"] == d).any() else np.nan for d in dates]
    tt = f.groupby("date")["t_end"].first()
    dfr["t_end"] = tt.reindex(dates)
    dfr.loc[dfr["t_end"].isna(), "m_primary"] = np.nan
    dfr["is_me"] = dates.isin(month_ends)
    xs = ["x_sig", "x_noise", "x_mom_12_1", "sx_ep", "sx_bp", "sx_volume_trend", "sx_mom_3m", "sx_mom_6m"]
    xs += ["x_sector_mom_3m", "sx_ret_1m", "x_ret_1w", "x_ret_1m"]
    return ModelingDataset(
        market=market,
        horizon=H,
        frame=f.sort_values(["date", "entity"]).reset_index(drop=True),
        xs_cols=xs,
        regime_cols=["r_mkt_dd"],
        as_of=pd.Timestamp(cal[-1]) if cal[-1] in set(grid) else pd.Timestamp(grid[-1]),
        calendar=cal,
        month_ends=month_ends,
        events=ev.build_calendar(market, {}),
        dates_frame=dfr,
        snapshot=f"synthetic-{seed}-{signal}",
        cost_bps=30.0,
    )


def _cands(*names: str) -> list[cmod.Candidate]:
    return [c for c in cmod.default_candidates() if c.name in names]


@pytest.fixture(scope="module")
def planted():
    return synthetic_dataset()


@pytest.fixture(scope="module")
def planted_result(planted, msettings):
    cache = DiskCache(msettings.cache_path)
    return sel.run_selection(
        planted,
        msettings,
        cache,
        candidates=_cands("mom_12_1", "ridge", "logit", "rsmt_v3"),
        threads=2,
        use_cache=False,
    )


# ------------------------------------------------------------------------------------------------ metrics


def test_per_date_auc_and_ic_match_sklearn_and_scipy():
    rng = np.random.default_rng(1)
    rows = []
    for d in pd.date_range("2020-01-31", periods=5, freq="ME"):
        s = rng.normal(size=60)
        y = 0.5 * s + rng.normal(size=60) - 0.3
        rows += [{"date": d, "score": a, "y_e": b} for a, b in zip(s, y)]
    f = pd.DataFrame(rows)
    auc = mt.per_date_auc(f)
    ic = mt.per_date_ic(f)
    for d, g in f.groupby("date"):
        assert auc[d] == pytest.approx(roc_auc_score(g["y_e"] > 0, g["score"]))
        assert ic[d] == pytest.approx(g["score"].rank().corr(g["y_e"].rank()))


def test_block_bootstrap_is_reproducible_and_brackets_the_mean():
    x = np.random.default_rng(0).normal(0.02, 0.05, 80)
    a = mt.block_bootstrap_means(x, seed=3)
    b = mt.block_bootstrap_means(x, seed=3)
    assert np.array_equal(a, b)
    s = mt.summarize(x)
    assert s["lo"] < s["mean"] < s["hi"] and s["n"] == 80


def test_top_quintile_net_subtracts_turnover_cost():
    f = pd.DataFrame(
        {
            "date": np.repeat(pd.date_range("2021-01-31", periods=2, freq="ME"), 25),
            "entity": [f"e{i}" for i in range(25)] * 2,
            "score": list(range(25)) + list(range(25))[::-1],
            "r": 0.01,
            "y_e": 0.0,
        }
    )
    tq = mt.top_quintile(f, cost_bps=30.0)
    assert tq["turnover"].tolist() == [1.0, 1.0]  # the second date's top 5 are all new names
    assert tq["net"].iloc[1] == pytest.approx(tq["gross"].iloc[1] - 30.0 / 1e4)


# ------------------------------------------------------------------------------------------------ events


def test_event_calendar_named_and_detected_shocks():
    idx = pd.bdate_range("2020-01-01", periods=400)
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 400))), index=idx)
    close.iloc[300:] *= 0.85  # a -16% day
    bench = pd.DataFrame({"close": close})
    kr = ev.build_calendar(Market.KR, {"KOSPI": bench})
    us = ev.build_calendar(Market.US, {})
    assert "russia_ukraine_2022" in set(kr.windows["key"]) and "russia_ukraine_2022" in set(us.windows["key"])
    assert "kr_martial_law_2024" in set(kr.windows["key"]) and "kr_martial_law_2024" not in set(us.windows["key"])
    shocks = kr.windows[kr.windows["kind"] == "shock"]
    assert list(shocks["start"]) == [idx[300]]
    # overlap: a label window touching the invasion; windows ending after the cutoff are unknown at the cutoff
    s, e = pd.Series([pd.Timestamp("2022-02-20")]), pd.Series([pd.Timestamp("2022-03-10")])
    assert kr.overlap_mask(s, e)[0]
    assert not kr.overlap_mask(s, e, known_before=pd.Timestamp("2022-03-15"))[0]


def test_event_down_scheme_weights_only_overlapping_rows():
    cal = ev.build_calendar(Market.US, {})
    f = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-02", "2021-06-01"]),
            "t_end": pd.to_datetime(["2020-03-30", "2021-06-29"]),
        }
    )
    w = ev.scheme_weights(f, "event_down", cal, pd.Timestamp("2023-01-01"))
    assert w.tolist() == [ev.EVENT_WEIGHT, 1.0]
    assert ev.scheme_weights(f, "none", cal, None).tolist() == [1.0, 1.0]
    with pytest.raises(ValueError):
        ev.scheme_weights(f, "bogus", cal, None)


# ------------------------------------------------------------------------------------------------ features / labels


def test_price_features_are_look_ahead_free():
    px = make_prices(n=700, seed=5)
    bench = make_prices(n=700, start_price=2500.0, seed=6)
    full = feat.entity_price_features(px, bench)
    for cut in (400, 550, 699):
        t = px.index[cut]
        part = feat.entity_price_features(px.loc[:t], bench.loc[:t])
        a, b = full.loc[t, feat.PRICE_FEATURES], part.loc[t, feat.PRICE_FEATURES]
        assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


def test_v2_price_features_are_look_ahead_free_and_defined():
    """skew60 / resid_ret_1m / seas at t: unchanged when the prices after t are dropped OR replaced."""
    px = make_prices(n=900, seed=7)
    bench = make_prices(n=900, start_price=2500.0, seed=8)
    full = feat.entity_price_features(px, bench)
    v2 = [c for c in feat.FEATURES_V2 if c in feat.PRICE_FEATURES]
    assert v2 == ["skew60", "resid_ret_1m", "seas"]
    for cut in (300, 620, 899):
        t = px.index[cut]
        part = feat.entity_price_features(px.loc[:t], bench.loc[:t])
        alt = px.copy()
        alt.loc[alt.index > t, ["open", "high", "low", "close"]] *= 3.0  # a wild future
        alt.loc[alt.index > t, "volume"] = 1.0
        other = feat.entity_price_features(alt, bench)
        a = full.loc[t, v2].to_numpy(dtype=float)
        assert np.isfinite(a).all()  # defined here (the check is not vacuous)
        assert np.allclose(a, part.loc[t, v2].to_numpy(dtype=float))
        assert np.allclose(a, other.loc[t, v2].to_numpy(dtype=float))
    # definitions
    t = px.index[700]
    c, b = px["close"], feat.align_close(px.index, bench)
    lr = np.log(c).diff()
    assert full.loc[t, "skew60"] == pytest.approx(lr.loc[:t].iloc[-60:].skew())
    beta = full.loc[t, "beta"]
    want = math.log(c.loc[t] / c.iloc[700 - 21]) - beta * math.log(b.loc[t] / b.iloc[700 - 21])
    assert full.loc[t, "resid_ret_1m"] == pytest.approx(want)
    lags = [math.log(c.iloc[700 - 252 * k + 20] / c.iloc[700 - 252 * k]) for k in (1, 2)]  # 700 bars: 2 lags
    assert full.loc[t, "seas"] == pytest.approx(np.mean(lags))


def test_v2_fundamentals_are_point_in_time_and_from_one_filing():
    """Prev-year TTM values: only when known by the current filing; YoY pairs come from ONE filing row."""
    panel = pd.DataFrame(
        {
            "entity": ["A"] * 4,
            "period_end": pd.to_datetime(["2020-03-31", "2020-12-31", "2021-03-31", "2021-06-30"]),
            "available_from": pd.to_datetime(["2020-05-15", "2021-03-15", "2021-05-17", "2021-08-16"]),
            "ni_parent_ttm": [8.0, 10.0, 12.0, 20.0],
            "equity_parent": [100.0] * 4,
            "revenue_ttm": [40.0, 50.0, 60.0, 70.0],
            "op_income_ttm": [4.0, 5.0, 6.0, 9.0],
        }
    )
    p = feat.add_prev_year(panel)
    q121 = p.index[p["period_end"] == pd.Timestamp("2021-03-31")][0]
    assert p.loc[q121, "op_income_ttm_prev"] == 4.0 and p.loc[q121, "ni_parent_ttm_prev"] == 8.0
    # 2021-06-30 has no filed period a year earlier -> no prev value (and no YoY from a mismatched filing)
    assert np.isnan(p.loc[p.index[-1], "op_income_ttm_prev"])
    # a prior-year value first filed AFTER the current row is never used
    late = panel.copy()
    late.loc[0, "available_from"] = pd.Timestamp("2021-06-01")
    assert np.isnan(feat.add_prev_year(late).loc[q121, "op_income_ttm_prev"])
    rows = pd.DataFrame({"date": pd.to_datetime(["2021-05-14", "2021-05-17", "2021-08-16"]), "entity": ["A"] * 3})
    mcap = pd.Series([1000.0] * 3)
    out = feat.fundamental_features(rows, p, mcap, 400)
    assert np.isnan(out.loc[0, "op_yoy"])  # before 2021-05-17 no filing with a prev-year value is known
    assert out.loc[1, "op_yoy"] == pytest.approx((6.0 - 4.0) / 4.0)
    assert out.loc[1, "dep"] == pytest.approx((12.0 - 8.0) / 1000.0)
    assert out.loc[1, "op_yield"] == pytest.approx(0.006) and out.loc[1, "sp"] == pytest.approx(0.06)
    # 2021-08-16: the latest filing has no prev -> the YoY stays that of the 2021-05-17 filing (same-filing pair)
    assert out.loc[2, "op_yoy"] == pytest.approx(0.5) and out.loc[2, "op_yield"] == pytest.approx(0.009)
    assert set(feat.FEATURES_V2) <= set(feat.XS_FEATURES)
    assert {"resid_ret_1m", "op_yield", "sp", "op_yoy", "dep"} <= set(feat.SECTOR_RELATIVE)


def test_fundamentals_use_only_rows_available_at_t():
    panel = pd.DataFrame(
        {
            "entity": ["A", "A"],
            "period_end": pd.to_datetime(["2020-12-31", "2021-03-31"]),
            "available_from": pd.to_datetime(["2021-03-15", "2021-05-17"]),
            "ni_parent_ttm": [10.0, 99.0],
            "equity_parent": [100.0, 100.0],
            "revenue_ttm": [50.0, 60.0],
            "op_income_ttm": [5.0, 6.0],
        }
    )
    rows = pd.DataFrame({"date": pd.to_datetime(["2021-05-14", "2021-05-17"]), "entity": ["A", "A"]})
    out = feat.fundamental_features(rows, feat.add_revenue_prev(panel), pd.Series([1000.0, 1000.0]), 200)
    assert out["ep"].tolist() == [pytest.approx(0.01), pytest.approx(0.099)]  # the 05-17 filing only from 05-17


def test_rank_gauss_per_date_is_monotone_and_keeps_nan():
    v = pd.Series([3.0, 1.0, np.nan, 2.0, 10.0, -1.0])
    d = pd.Series(["a", "a", "a", "a", "b", "b"])
    z = feat.rank_gauss(v, d)
    assert np.isnan(z[2]) and z[1] < z[3] < z[0] and z[5] < z[4]
    assert z[[0, 1, 3]].sum() == pytest.approx(0.0, abs=1e-12)


def test_labels_delisting_gap_and_halt():
    idx = pd.bdate_range("2021-01-01", periods=60)
    close = pd.Series(np.linspace(100, 159, 60), index=idx)
    feats = pd.DataFrame({"close": close, "traded": True}, index=idx)
    bench = pd.Series(1000.0, index=pd.bdate_range("2021-01-01", periods=120))
    d = np.array([idx[10]], dtype="datetime64[ns]")
    te = np.array([pd.bdate_range("2021-01-01", periods=120)[80]], dtype="datetime64[ns]")  # after the frame ends
    lab = _labels_for_entity(d, te, feats, bench, delisted=True, terminal=None, halts=False)
    assert lab["status"][0] == "delisted" and lab["r"][0] == pytest.approx(math.log(159 / 110))
    lab = _labels_for_entity(d, te, feats, bench, delisted=False, terminal=None, halts=False)
    assert lab["status"][0] == "price_gap_end" and np.isnan(lab["r"][0])
    term = (pd.bdate_range("2021-01-01", periods=120)[70], 80.0)
    lab = _labels_for_entity(d, te, feats, bench, delisted=True, terminal=term, halts=False)
    assert lab["status"][0] == "delisted_terminal" and lab["r"][0] == pytest.approx(math.log(80 / 110))
    halted = feats.copy()
    halted.loc[idx[30:35], "traded"] = False
    te2 = np.array([idx[32]], dtype="datetime64[ns]")
    lab = _labels_for_entity(d, te2, halted, bench, delisted=False, terminal=None, halts=True)
    assert lab["status"][0] == "halted_exit_next_trade" and lab["r"][0] == pytest.approx(math.log(close.iloc[35] / 110))


def test_members_on_grid_uses_latest_rebalance():
    mem = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-29", "2021-01-29", "2021-02-26"]),
            "entity": ["A", "B", "C"],
            "ticker": ["A", "B", "C"],
        }
    )
    grid = pd.to_datetime(["2021-01-15", "2021-02-05", "2021-02-26", "2021-03-05"])
    out = members_on_grid(mem, pd.DatetimeIndex(grid))
    got = out.groupby("date")["entity"].apply(list).to_dict()
    assert pd.Timestamp("2021-01-15") not in got
    assert got[pd.Timestamp("2021-02-05")] == ["A", "B"] and got[pd.Timestamp("2021-03-05")] == ["C"]


# ------------------------------------------------------------------------------------------------ protocol


def test_v2_candidates_learn_the_planted_signal(planted, msettings):
    data = sel._Data(planted)
    cut = data.me_lab[60]
    ctx = sel._ctx(data, cut, msettings, 2)
    test = data.eval_rows(data.me_lab[62:80])
    for c in _cands("logit_w", "ridge_rg"):
        tr = data.train(sel.purge_end(planted, cut), c.rows)
        f = c.fit(tr, dict(c.grid[1]), "none", ctx)
        a = mt.per_date_auc_median(test.assign(score=f.predict(test))).mean()
        assert a > 0.56, c.name
    # v3 (docs/MODELING.md 16.4): 23 eligible + 2 diagnostics; rsmt_v2 removed; v4.1 (docs 82): + 3 *_bo
    assert _cands("logit_w")[0].rows == "W" and len(cmod.default_candidates()) == 35  # v4.2 (docs 83): + 7 *_bo
    assert "rsmt_v2" not in {c.name for c in cmod.default_candidates()}
    assert cmod.STACKED == ("rsmt_v3", "rsmt_v3_bo")
    assert all(len(c.grid) <= 4 for c in cmod.default_candidates())
    diag = {c.name for c in cmod.default_candidates() if c.diagnostic}
    assert diag == {"logit_w_core", "lgbm_clf_core"}


def test_blend_is_the_fixed_rank_mean_of_logit_and_lgbm_clf(planted, msettings):
    data = sel._Data(planted)
    outer, _, _, _ = sel.make_contexts(data)
    res = sel.run_context(data, outer[-1], _cands("logit", "lgbm_clf"), msettings, 2, lambda _m: None)
    p = res.predictions
    assert res.configs["blend"] == {"members": ["logit", "lgbm_clf"], "weights": [1.0, 1.0]}
    want = (feat.uniform_rank(p["logit"], p["date"]) + feat.uniform_rank(p["lgbm_clf"], p["date"])) / 2
    assert np.allclose(p["blend"], want)
    assert sel.composite_members("blend", res.configs) == ["logit", "lgbm_clf"]
    assert sel.composite_members("logit", res.configs) == ["logit"]
    alone = sel.run_context(data, outer[-1], _cands("logit"), msettings, 2, lambda _m: None)
    assert "blend" not in alone.predictions  # only when every member ran


@pytest.mark.parametrize(
    "name",
    [
        "enet_logit",
        "rf",
        "et",
        "xgb_reg",
        "xgb_clf",
        "xgb_clf_w",
        "xgb_rank",
        "cat_clf",
        "cat_rank",
        "mlp",
        "lgbm_rank_bo",
        "lgbm_reg_bo",
        "et_bo",
        "logit_bo",
        "ridge_bo",
        "rf_bo",
        "xgb_clf_bo",
        "cat_clf_bo",
        "mlp_bo",
    ],
)
def test_v3_candidates_learn_the_planted_signal(planted, msettings, name):
    data = sel._Data(planted)
    cut = data.me_lab[60]
    ctx = sel._ctx(data, cut, msettings, 2)
    test = data.eval_rows(data.me_lab[62:80])
    c = _cands(name)[0]
    tr = data.train(sel.purge_end(planted, cut), c.rows)
    f = c.fit(tr, dict(c.grid[0]), c.schemes[-1], ctx)
    score = f.predict(test)
    assert len(score) == len(test) and np.isfinite(score).all()
    assert mt.per_date_auc_median(test.assign(score=score)).mean() > 0.54, name
    assert mt.per_date_primary(test.assign(score=score)).mean() > 0, name  # v4: the money metric too
    imp = f.importance()
    assert imp
    if name not in ("mlp", "mlp_bo"):  # first-layer weight mass is only a rough display proxy for the neural net
        assert max(imp, key=imp.get) in {"x_sig", "sx_ep"}, (name, sorted(imp.items(), key=lambda kv: -kv[1])[:3])


def test_composite_rule_members_and_weights():
    cands = cmod.default_candidates()
    inner = {  # protocol v4: "inner_auc" holds the inner PRIMARY score (top-quintile net excess, no skill = 0)
        "logit": {"inner_auc": 0.006, "inner_ic": 0.05},
        "logit_w": {"inner_auc": 0.007, "inner_ic": 0.06},
        "lgbm_clf": {"inner_auc": 0.005, "inner_ic": 0.04},
        "xgb_rank": {"inner_auc": 0.004, "inner_ic": 0.03},
        "rf": {"inner_auc": 0.003, "inner_ic": 0.02},
        "mlp": {"inner_auc": -0.001, "inner_ic": -0.01},
        "logit_w_core": {"inner_auc": 0.010, "inner_ic": 0.09},  # diagnostic: never a member
        "rsmt_v3": {"inner_auc": 0.020, "inner_ic": 0.1},  # stacked: never a member
        "mom_12_1": {"inner_auc": 0.015, "inner_ic": 0.1},  # baseline (untuned): never a member
    }
    assert cmod.composite_rule("ensemble", inner, cands) == (["logit_w", "logit", "lgbm_clf"], [1.0] * 3)
    m5, _ = cmod.composite_rule("ensemble5", inner, cands)
    assert m5 == ["logit_w", "logit", "lgbm_clf", "xgb_rank", "rf"]
    mw, w = cmod.composite_rule("ensemble_w", inner, cands)
    assert "mlp" not in mw and w[0] == pytest.approx(0.007) and len(mw) == len(w) == 5
    assert cmod.composite_rule("family_blend", inner, cands)[0] == ["logit_w", "lgbm_clf", "rf", "mlp"]
    assert cmod.composite_rule("blend", inner, cands) == (["logit", "lgbm_clf"], [1.0, 1.0])
    assert cmod.composite_rule("blend", {"logit": inner["logit"]}, cands) == ([], [])


def test_weighted_rank_mean():
    p = pd.DataFrame({"date": [1, 1, 1], "a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    eq = sel.rank_mean(p, ["a", "b"])
    assert np.allclose(eq, 0.5)
    w = sel.rank_mean(p, ["a", "b"], [3.0, 1.0])
    assert w.iloc[2] > w.iloc[0]


def test_diagnostics_are_reported_never_selected_and_measure_alt_value(msettings):
    ds = synthetic_dataset(seed=3)
    ds.core_xs_cols = [c for c in ds.xs_cols if c not in ("x_sig", "sx_ep")]  # the "new" inputs carry the signal
    ds.core_regime_cols = list(ds.regime_cols)
    cache = DiskCache(msettings.cache_path)
    res = sel.run_selection(
        ds, msettings, cache, candidates=_cands("mom_12_1", "logit_w", "logit_w_core"), threads=2, use_cache=False
    )
    rows = {c["name"]: c for c in res.candidates}
    assert rows["logit_w_core"]["diagnostic"] is True and rows["logit_w"]["diagnostic"] is False
    assert res.selected != "logit_w_core"
    assert abs(rows["logit_w_core"]["outer"][mt.PRIMARY_AUC]["mean"] - 0.5) < 0.04  # it cannot see the signal
    av = res.protocol["alt_value"]["logit_w - logit_w_core"]["outer"]
    assert av["mean"] > 0 and av["t"] > 3  # v4: the paired difference of the money metric
    assert res.protocol["eligible_candidates"] == len(res.candidates) - 1
    for comp in ("ensemble", "ensemble5", "ensemble_w", "family_blend"):
        assert "logit_w_core" not in (res.final.configs.get(comp) or {}).get("members", [])


def test_contexts_purge_and_embargo(planted):
    data = sel._Data(planted)
    outer, hold, final, info = sel.make_contexts(data)
    assert len(hold.test_dates) == sel.HOLDOUT_MONTHS
    for c in [*outer, hold]:
        tr = data.train(c.train_end, "W")
        assert tr["date"].max() <= c.train_end
        assert tr["t_end"].max() < c.test_dates[0]
        gap = int(planted.pos(np.datetime64(c.test_dates[0], "ns"))) - int(
            planted.pos(np.datetime64(tr["t_end"].max(), "ns"))
        )
        assert gap >= planted.horizon  # label end -> test start >= h sessions (embargo)
        for te, val in sel.inner_folds(data, c.train_end):
            assert val.max() <= c.train_end and data.train(te, "W")["t_end"].max() < val[0]
    assert outer[-1].test_dates[-1] < hold.test_dates[0]
    assert info["outer_folds"] == len(outer)


def test_selection_finds_planted_signal_and_momentum_does_not(planted_result):
    res = planted_result
    auc = {c["name"]: c["outer"]["auc"]["mean"] for c in res.candidates}
    assert auc["logit"] > 0.56 and auc["ridge"] > 0.56
    assert abs(auc["mom_12_1"] - 0.5) < 0.03
    assert res.selected in {"logit", "ridge", "ensemble", "ensemble5", "ensemble_w", "family_blend", "rsmt_v3"}
    assert res.gate["passed"] and res.gate["beats_momentum"]
    assert res.gate["metric"] == mt.PRIMARY and res.gate["post_hoc"] is True
    assert res.gate["value"] == pytest.approx(res.candidate(res.selected)["outer"][mt.PRIMARY]["mean"])
    assert res.gate["value"] > 0 and res.gate["lo"] > 0 and res.gate["null"] == 0.0
    assert res.gate["auc"] == pytest.approx(res.candidate(res.selected)["outer"][mt.PRIMARY_AUC]["mean"])
    h = res.candidate(res.selected)["holdout"]
    assert h["auc"]["mean"] > 0.55 and h[mt.PRIMARY_AUC]["mean"] > 0.55  # the untouched holdout confirms
    assert res.calibration["slope"] > 0 and res.calibration["shrink"] > 0.9


def test_rsmt_v3_is_stacked_checkpointed_and_carries_the_planted_ranking(planted_result, planted, msettings):
    res = planted_result
    c = res.candidate("rsmt_v3")
    assert c["family"] == "rsmt" and c["rows"] == "W"
    assert c["outer"]["auc"]["mean"] > 0.55 and c["holdout"]["auc"]["mean"] > 0.55
    # stacked on the context's best inner-CV candidate, never tuned / fitted as a base candidate
    stack_bases = {x["name"]: x["stack_base"] for x in res.contexts}
    for ctx_name, cfg in c["configs"].items():
        if cfg is None:
            continue
        assert cfg["stack_base"] == stack_bases[ctx_name]
        assert cfg["params"]["multiple_testing"] == "simes" and cfg["params"]["target_scale"] == "vol"
    assert "rsmt_v3" not in (res.final.ensemble_members or [])
    fin = res.final.fitted["rsmt_v3"].model
    assert fin.version == 3 and fin.tree.scaled and fin.tree.gamma_mode == "leaf"
    assert fin.tree.multiple_testing == "simes" and fin.tree.gamma > 0
    assert fin.describe()[0].startswith("RSMT-Q v3")
    # the stacked candidate has its own checkpoint: a cached rerun reuses it
    cache = DiskCache(msettings.cache_path)
    cands = _cands("mom_12_1", "logit", "rsmt_v3")
    seen: list[str] = []
    sel.run_selection(planted, msettings, cache, candidates=cands, threads=2, use_cache=True, max_outer=1)
    again = sel.run_selection(
        planted, msettings, cache, candidates=cands, threads=2, use_cache=True, max_outer=1, progress=seen.append
    )
    assert any("rsmt_v3 checkpoint reused" in x for x in seen)
    assert again.candidate("rsmt_v3")["outer"]["auc"]["mean"] is not None


def test_holdout_labels_never_change_the_selection(planted, msettings):
    """Scrambling every label inside the holdout window leaves the selection and outer metrics unchanged."""
    cache = DiskCache(msettings.cache_path)
    cands = _cands("mom_12_1", "ridge", "logit")
    a = sel.run_selection(planted, msettings, cache, candidates=cands, threads=2, use_cache=False)
    b_ds = synthetic_dataset()
    h0 = pd.Timestamp(a.protocol["holdout"][0])
    f = b_ds.frame
    m = f["date"] >= h0  # rows before h0 (incl. the last outer test block) keep their labels
    rng = np.random.default_rng(99)
    f.loc[m, "y_e"] = rng.permutation(f.loc[m, "y_e"].to_numpy())
    b_ds.frame = add_date_targets(f)
    b = sel.run_selection(b_ds, msettings, cache, candidates=cands, threads=2, use_cache=False)
    assert a.selected == b.selected
    for n in ("ridge", "logit"):
        assert a.candidate(n)["outer"]["auc"]["mean"] == pytest.approx(b.candidate(n)["outer"]["auc"]["mean"])


def test_every_fit_before_the_final_context_is_purged(planted, msettings, monkeypatch):
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    orig = cmod.Candidate.fit

    def spy(self, train, params, scheme, ctx):
        seen.append((ctx.cutoff, train["t_end"].max()))
        return orig(self, train, params, scheme, ctx)

    monkeypatch.setattr(cmod.Candidate, "fit", spy)
    sel.run_selection(planted, msettings, DiskCache(msettings.cache_path), candidates=_cands("ridge"), use_cache=False)
    non_final = [(c, t) for c, t in seen if c < planted.as_of]
    assert non_final and all(t < c for c, t in non_final)


def test_selection_rule_is_deterministic_and_breaks_ties_by_ic():
    def m(prim, ic, top):
        return {mt.PRIMARY: {"mean": prim}, "ic": {"mean": ic}, "top_net": {"mean": top}}

    assert sel.TIE_TOL == pytest.approx(0.0002)  # per month (protocol v4)
    table = {"a": m(0.0030, 0.01, 0.0), "b": m(0.00305, 0.02, 0.0), "c": m(0.0020, 0.09, 0.1)}
    assert sel.select_best(table) == "b"  # |0.00305 - 0.0030| < TIE_TOL -> IC decides
    table2 = {"a": m(0.0030, 0.05, 0.0), "b": m(0.00305, 0.02, 0.0)}
    assert sel.select_best(table2) == "a"
    assert sel.select_best(dict(reversed(list(table2.items())))) == "a"  # order-independent
    table3 = {"a": m(0.0030, 0.05, 0.01), "b": m(0.0030, 0.05, 0.02)}
    assert sel.select_best(table3) == "b"  # then the raw top-quintile net excess
    assert sel.select_best({"x": m(0.001, 0.0, 0.0), "y": m(0.004, -0.1, -1.0)}) == "y"  # a real gap wins


def test_protocol_v4_primary_metric_drives_selection_and_gate():
    """The v4 PRIMARY (top-quintile net excess on the simple label) decides; the AUCs are only reported."""
    assert mt.PRIMARY == "top_ye_net" and sel.SELECTION_RULE["primary_metric"] == mt.PRIMARY
    assert mt.PRIMARY_AUC == "auc_median" and sel.PROTOCOL_VERSION == 4
    assert sel.SELECTION_RULE["post_hoc"] is True
    table = {  # "a" ranks names well (high AUC) but its top quintile earns less than "b"'s
        "a": {mt.PRIMARY: {"mean": 0.001}, mt.PRIMARY_AUC: {"mean": 0.56}, "ic": {"mean": 0.0}},
        "b": {mt.PRIMARY: {"mean": 0.004}, mt.PRIMARY_AUC: {"mean": 0.52}, "ic": {"mean": 0.0}},
    }
    assert sel.select_best(table) == "b"
    assert sel.select_best(table, key=mt.PRIMARY_AUC, tie_tol=0.0005) == "a"  # the v3 rule picks the other one
    assert mt.metric_label(mt.PRIMARY, "ko") == "상위 20% 베타 조정 초과수익(비용 후, 월)"
    assert mt.auc_label("auc_median", "ko") == "날짜 내 AUC(중앙값 초과)"
    assert mt.auc_label("auc_median", "en") == "within-date AUC (above median)"
    assert "y_e" in mt.auc_label("auc", "en")


def test_primary_auc_is_the_above_median_auc():
    rng = np.random.default_rng(3)
    rows = []
    for d in pd.date_range("2020-01-31", periods=4, freq="ME"):
        s = rng.normal(size=50)
        y = 0.4 * s + rng.normal(size=50) + 0.8  # most names beat 0: the two labels differ
        rows += [{"date": d, "score": a, "y_e": b} for a, b in zip(s, y)]
    f = pd.DataFrame(rows)
    prim = mt.per_date_auc_median(f)
    for d, g in f.groupby("date"):
        assert prim[d] == pytest.approx(roc_auc_score(g["y_e"] > g["y_e"].median(), g["score"]))
    ev = mt.evaluate(f.assign(entity=[f"e{i % 50}" for i in range(len(f))], r=f["y_e"]), 0.0)
    assert ev[mt.PRIMARY_AUC]["mean"] == pytest.approx(prim.mean())


def test_inner_cv_scores_the_primary(planted, msettings, monkeypatch):
    """``_score_inner`` averages the per-date v4 PRIMARY (top-quintile net excess, the dataset's cost) over the
    inner folds - not an AUC."""
    data = sel._Data(planted)
    outer, _, _, _ = sel.make_contexts(data)
    folds = sel.inner_folds(data, outer[0].train_end)
    seen: list[pd.Series] = []
    costs: list[float] = []
    orig = mt.per_date_top_ye_net

    def spy(frame, score="score", cost_bps=0.0, label="y_e", q=5):
        out = orig(frame, score, cost_bps, label, q)
        seen.append(out)
        costs.append(cost_bps)
        return out

    monkeypatch.setattr(mt, "per_date_top_ye_net", spy)
    ridge = _cands("ridge")[0]
    a, _ = sel._score_inner(data, ridge, {"a": 0.1}, "none", folds, msettings, 2)
    assert len(seen) == len(folds) and a == pytest.approx(pd.concat(seen).mean())
    assert all(c == planted.cost_bps for c in costs)


def test_serving_gate_requires_both_conditions():
    k = mt.PRIMARY
    base = {k: {"mean": 0.001, "lo": -0.002}, mt.PRIMARY_AUC: {"mean": 0.60, "lo": 0.59}}
    g = sel.serving_gate({k: {"mean": 0.004, "lo": 0.001}, mt.PRIMARY_AUC: {"mean": 0.40, "lo": 0.30}}, base)
    assert g["passed"] and g["metric"] == k and g["post_hoc"] is True and g["needs_confirmation"] is True
    assert g["lo95_above_null"] and g["value"] == 0.004 and g["baseline"] == 0.001 and g["auc"] == 0.40
    assert g["value_ann"] == pytest.approx(0.048)
    assert not sel.serving_gate({k: {"mean": 0.004, "lo": -0.0001}}, base)["passed"]  # lower bound must be > 0
    assert not sel.serving_gate({k: {"mean": 0.0008, "lo": 0.0001}}, base)["passed"]  # must beat momentum
    # a great AUC does not count without the money
    assert not sel.serving_gate({mt.PRIMARY_AUC: {"mean": 0.60, "lo": 0.58}}, base)["passed"]


# ------------------------------------------------------------------------------------------------ serving


def _md(ticker: str, market: Market = Market.KR) -> MarketData:
    px = make_prices(n=400, seed=hash(ticker) % 1000)
    inst = Instrument(
        ticker=ticker,
        market=market,
        name=ticker,
        currency="KRW",
        benchmark="KOSPI",
        benchmark_name="KOSPI",
    )
    return MarketData(inst, px, make_prices(n=400, start_price=2500.0, seed=4), Fundamentals(per=10.0, pbr=1.2))


def test_artifact_scores_members_and_outsiders(planted_result, planted, msettings, monkeypatch):
    from quantcio.modeling import serve

    art = serve.build_artifact(planted_result, planted, msettings)
    # the gate passed, but its label is post hoc: without confirmation the artifact is not "skilled"
    assert len(art.latest) == 40 and art.gate_passed and art.post_hoc and not art.skill_ok
    assert art.multiplier == pytest.approx(art.calibration["shrink"] * sel.UNSKILLED_SHRINK)
    assert art.metrics["primary_auc"] == mt.PRIMARY_AUC and "중앙값" in art.metrics["auc_label_ko"]
    assert np.isclose(art.latest["mu_excess"].mean(), 0.0, atol=1e-3)
    top = art.latest.sort_values("score").iloc[-1]
    assert top["mu_excess"] > 0 and top["percentile"] > 0.9
    cache = DiskCache(msettings.cache_path)
    serve.save_artifact(art, msettings, cache)
    member = str(art.latest["ticker"].iloc[0])
    out, notes, got = serve.score_for_report(
        Market.KR, [(member, _md(member)), ("999999", _md("999999"))], msettings, cache
    )
    assert got is not None and set(out) == {member, "999999"}
    fc = out[member]
    assert isinstance(fc, ExcessForecast) and fc.in_universe and fc.model == art.selected
    assert fc.mu_excess == pytest.approx(float(art.latest.loc[art.latest["ticker"] == member, "mu_excess"].iloc[0]))
    assert not out["999999"].in_universe and any("PIT" in n for n in out["999999"].notes)
    ov = serve.excess_override(fc)
    assert ov.mu_excess == fc.mu_excess
    assert ov.market_p_up == (fc.market_p_up if fc.market_model_used else None)


@pytest.mark.parametrize(
    "behaviour, confirmed",
    [("missing", False), ("none", False), ("raises", False), ("negative", False), ("dict", True)],
)
def test_post_hoc_gate_needs_xsh_confirmation_at_scoring_time(
    planted_result, planted, msettings, tmp_path, monkeypatch, behaviour, confirmed
):
    from quantcio import holdout
    from quantcio.modeling import serve

    s = msettings.model_copy(deep=True)
    s.data.cache_dir = str(tmp_path / "c")
    cache = DiskCache(tmp_path / "c")
    art = serve.build_artifact(planted_result, planted, s)
    serve.save_artifact(art, s, cache)
    calls: list[tuple] = []

    def fake(market, settings):
        calls.append((str(market), settings is s))
        if behaviour == "raises":
            raise RuntimeError("registry unreadable")
        return {"none": None, "negative": {"id": "x", "confirmed": False}, "dict": {"id": "XSH-KR-1"}}[behaviour]

    if behaviour == "missing":
        monkeypatch.delattr(holdout, "xsh_confirmation", raising=False)
    else:
        monkeypatch.setattr(holdout, "xsh_confirmation", fake, raising=False)
    member = str(art.latest["ticker"].iloc[-1])
    out, _, got = serve.score_for_report(Market.KR, [(member, _md(member))], s, cache)
    fc = out[member]
    shrink = art.calibration["shrink"]
    slope = art.calibration["slope"]
    assert (behaviour == "missing") or calls == [("KR", True)]
    assert fc.skill_ok is confirmed and got.skill_ok is confirmed and fc.gate["confirmed"] is confirmed
    assert fc.gate["gate_passed"] and fc.gate["post_hoc"]
    mult = shrink * (1.0 if confirmed else sel.UNSKILLED_SHRINK)
    assert got.multiplier == pytest.approx(mult) and fc.mu_excess == pytest.approx(mult * slope * fc.score_z)
    note = "게이트 통과(사후 지표) — 미열람 자료(전진 홀드아웃) 확인 전까지 ×0.35"
    assert (note in fc.notes) is (not confirmed)
    assert load_cached_skill(serve, s, cache) is False  # the cached artifact never bakes the confirmation in
    en = s.model_copy(deep=True)
    en.report.language = "en"
    out_en, _, _ = serve.score_for_report(Market.KR, [(member, _md(member))], en, cache)
    note_en = "gate passed (post-hoc metric) - x0.35 until confirmed on untouched data (forward holdout)"
    assert (note_en in out_en[member].notes) is (not confirmed)


def load_cached_skill(serve, settings, cache) -> bool:
    return bool(serve.load_artifact(Market.KR, settings, cache).skill_ok)


def test_failed_gate_note_names_the_metric(planted_result, planted, msettings, tmp_path):
    from quantcio.modeling import serve

    s = msettings.model_copy(deep=True)
    s.data.cache_dir = str(tmp_path / "c")
    cache = DiskCache(tmp_path / "c")
    art = serve.build_artifact(planted_result, planted, s)
    art.gate = {**art.gate, "passed": False, "lo95_above_null": False}
    art.gate_passed = False
    serve.save_artifact(art, s, cache)
    member = str(art.latest["ticker"].iloc[0])
    fc = serve.score_for_report(Market.KR, [(member, _md(member))], s, cache)[0][member]
    assert not fc.skill_ok and any("상위 20% 베타 조정 초과수익(비용 후, 월) 하한>0 False" in n for n in fc.notes)


def test_report_labels_every_auc_and_flags_post_hoc(planted_result, msettings):
    from quantcio.modeling.report import render_markdown, summary_lines

    ko, en = summary_lines(planted_result)
    # v4: the money metric leads, the (v3 primary) above-median AUC is reported next to it
    assert "상위 20% 베타 조정 초과수익(비용 후, 월)" in ko and "날짜 내 AUC(중앙값 초과)" in ko
    assert "top-quintile beta-adjusted excess" in en and "within-date AUC (above median)" in en and "post-hoc" in en
    md = render_markdown(planted_result, msettings)
    assert "사후 결정 경고" in md and "날짜 내 AUC(중앙값 초과)" in md
    assert "| blend |" not in md and "**blend**" not in md  # the fixed blend needs logit + lgbm_clf


def test_ventile_of_matches_the_88_bins():
    from quantcio.modeling.serve import ventile_of

    assert [ventile_of(p) for p in (0.0, 0.0499, 0.05, 0.5, 0.9499, 0.95, 0.9999, 1.0)] == [1, 1, 2, 11, 19, 20, 20, 20]
    n = 328
    u = (np.arange(1, n + 1) - 0.5) / n
    assert [ventile_of(p) for p in u] == list(np.clip(np.floor(u * 20), 0, 19).astype(int) + 1)


def test_order_artifact_is_reference_only(planted_result, planted, msettings, tmp_path):
    """docs/MODELING.md 93, 94.3: the reference artifacts live in their own namespace (the served artifact is
    untouched), one per model in display order; order_for_report gives each model's view: each requested name's
    ventile (outsiders placed from their own prices) + the D20 list."""
    from quantcio.modeling import serve

    s = msettings.model_copy(deep=True)
    s.data.cache_dir = str(tmp_path / "c")
    cache = DiskCache(tmp_path / "c")
    assert serve.order_for_report(Market.KR, [("000001", _md("000001"))], s, cache) == []
    art = serve.build_artifact(planted_result, planted, s)
    art.metrics = {**art.metrics, "ventile": {"top_ann": 0.4}}
    serve.save_order_artifact(art, s, cache)
    assert serve.load_artifact(Market.KR, s, cache) is None and serve.load_order_artifact(Market.KR, s, cache)
    lat = art.latest.sort_values("score")
    lo, hi = str(lat["ticker"].iloc[0]), str(lat["ticker"].iloc[-1])
    vs = serve.order_for_report(Market.KR, [(lo, _md(lo)), (hi, _md(hi)), ("999999", _md("999999"))], s, cache)
    assert len(vs) == 1
    v = vs[0]
    assert v["model"] == art.selected and v["n"] == len(art.latest) == 40 and v["k"] == 20 and v["horizon"] == 20
    t = v["tickers"]
    assert t[lo]["ventile"] == 1 and t[hi]["ventile"] == 20 and t[lo]["in_universe"] and t[hi]["in_universe"]
    assert not t["999999"]["in_universe"] and 1 <= t["999999"]["ventile"] <= 20
    assert t["999999"]["ventile"] == serve.ventile_of(t["999999"]["percentile"])
    assert [n["ticker"] for n in v["top"]] == [str(x) for x in lat["ticker"].iloc[-2:][::-1]]  # 40 names: D20 = 2
    assert v["stats"] == {"top_ann": 0.4} and v["as_of"] == str(art.as_of.date())
    # a second model: its own slot, shown after the first; re-saving keeps the order; removal hides it
    other = copy.deepcopy(art)
    other.selected = "other_model"
    other.latest = art.latest.assign(score=-art.latest["score"], percentile=1.0 - art.latest["percentile"])
    serve.save_order_artifact(other, s, cache)
    serve.save_order_artifact(art, s, cache)
    assert serve.order_models(Market.KR, s, cache) == [art.selected, "other_model"]
    assert serve.load_order_artifact(Market.KR, s, cache).selected == art.selected
    assert serve.load_order_artifact(Market.KR, s, cache, "other_model").selected == "other_model"
    assert serve.load_order_artifact(Market.KR, s, cache, "missing") is None
    vs = serve.order_for_report(Market.KR, [(lo, _md(lo))], s, cache)
    assert [x["model"] for x in vs] == [art.selected, "other_model"]
    assert vs[0]["tickers"][lo]["ventile"] == 1 and vs[1]["tickers"][lo]["ventile"] == 20
    assert serve.remove_order_artifact(Market.KR, "other_model", s, cache)
    assert not serve.remove_order_artifact(Market.KR, "other_model", s, cache)
    assert [x["model"] for x in serve.order_for_report(Market.KR, [(lo, _md(lo))], s, cache)] == [art.selected]
    assert serve.load_artifact(Market.KR, s, cache) is None  # the served slot is never written


def test_missing_artifact_is_a_note_not_an_error(msettings, tmp_path):
    from quantcio.modeling import serve

    s = msettings.model_copy(deep=True)
    s.data.cache_dir = str(tmp_path / "empty")
    out, notes, art = serve.score_for_report(
        Market.US, [("AAPL", _md("AAPL", Market.US))], s, DiskCache(tmp_path / "empty")
    )
    assert out == {} and art is None and any("modeling select" in n for n in notes)


# ------------------------------------------------------------------------------------ RSMT-Q v2 / serving override


def test_rsmt_ranker_recovers_the_stacked_offset(msettings):
    from quantcio.quant.rsmt import RSMTRanker

    rng = np.random.default_rng(4)
    cal = pd.bdate_range("2018-01-01", periods=900)
    dates = cal[::5][:150]
    n = 60
    f = pd.DataFrame(
        {
            "date": np.repeat(dates, n),
            "stack_z": rng.normal(size=len(dates) * n),
            "x_ret_1m": rng.normal(size=len(dates) * n),
            "r_mkt_dd": np.repeat(rng.uniform(-0.3, 0, len(dates)), n),
            "vol60": rng.uniform(0.2, 0.5, len(dates) * n),
        }
    )
    f["y_dm"] = 0.02 * f["stack_z"] + rng.normal(0, 0.05, len(f))
    f["w_date"] = 1.0 / n
    rk = RSMTRanker(["x_ret_1m"], ["r_mkt_dd"], boot_sims=99, cache_dir=msettings.cache_path / "rsmt_null").fit(
        f, cal, y_col="y_dm"
    )
    assert rk.tree.gamma == pytest.approx(0.02, abs=0.004)
    p = rk.predict(f)
    assert np.corrcoef(p["mu_e"], f["stack_z"])[0, 1] > 0.9
    assert rk.describe()[0].startswith("RSMT-Q v2")
    np.testing.assert_array_equal(p["score"], p["mu_e"])  # v2 ranks by mu_e
    # split-test bootstrap clusters = runs of 3 h-blocks (overlapping weekly labels), min_leaf_blocks counts h-blocks
    assert rk.tree.n_clusters == rk.n_blocks and np.unique(rk.tree._clusters).size == -(-rk.n_blocks // 3)


def test_rsmt_ranker_v3_standardised_target_leaf_gamma_and_simes(msettings):
    """v3: the target y / (vol * sqrt(h/252)) - the stacked offset is recovered in vol units, the ranking score is
    the standardised mean (not re-ranked by volatility), mu_e is back in returns, the gamma is leaf-local."""
    from quantcio.quant.rsmt import RSMTRanker

    rng = np.random.default_rng(6)
    cal = pd.bdate_range("2018-01-01", periods=900)
    dates = cal[::5][:150]
    n = 60
    f = pd.DataFrame(
        {
            "date": np.repeat(dates, n),
            "stack_z": rng.normal(size=len(dates) * n),
            "x_ret_1m": rng.normal(size=len(dates) * n),
            "r_mkt_dd": np.repeat(rng.uniform(-0.3, 0, len(dates)), n),
            "r_mkt_vol_ratio": np.repeat(rng.uniform(0.7, 1.3, len(dates)), n),
            "vol60": rng.uniform(0.2, 0.6, len(dates) * n),
        }
    )
    s = f["vol60"] * np.sqrt(20 / 252)
    f["y_dm"] = s * (0.2 * f["stack_z"] + rng.normal(0, 1.0, len(f)))
    f["w_date"] = 1.0 / n
    kw = dict(boot_sims=99, cache_dir=msettings.cache_path / "rsmt_null", horizon=20)
    rk = RSMTRanker(
        ["x_ret_1m"],
        ["r_mkt_dd", "r_mkt_vol_ratio"],
        multiple_testing="simes",
        target_scale="vol",
        gamma_mode="leaf",
        **kw,
    ).fit(f, cal, y_col="y_dm")
    assert rk.version == 3 and rk.tree.scaled and rk.tree.multiple_testing == "simes"
    assert rk.tree.gamma == pytest.approx(0.2, abs=0.04)  # in vol units
    assert abs(rk.tree.eta_equivalent - 1.0) < 0.2  # the vol60 scale is the right scale: residual eta ~ 0
    p = rk.predict(f)
    np.testing.assert_allclose(p["mu_e"], p["score"] * s.to_numpy())
    assert stats_corr(p["score"], f["stack_z"]) > 0.95 and stats_corr(p["score"], f["vol60"]) < 0.1
    assert all(np.isfinite(lf.gamma) for lf in rk.tree.leaves)
    assert rk.describe()[0].startswith("RSMT-Q v3") and "simes" in rk.describe()[0]
    pca = RSMTRanker(["x_ret_1m"], ["r_mkt_dd", "r_mkt_vol_ratio"], multiple_testing="pca", pca_components=1, **kw)
    pca.fit(f, cal, y_col="y_dm")
    assert pca.tree.split_features == ["x_ret_1m", "PC1"]  # only the regime variables are reduced
    assert set(pca.importance()) >= {"x_ret_1m", "PC1"}
    with pytest.raises(ValueError, match="target_scale"):
        RSMTRanker(["x_ret_1m"], ["r_mkt_dd"], target_scale="ewma")


def stats_corr(a, b) -> float:
    return float(np.corrcoef(np.asarray(a, dtype=float), np.asarray(b, dtype=float))[0, 1])


def test_market_model_serves_the_excess_override():
    from test_rsmt import synthetic_price_panel

    from quantcio.quant.panel import build_panel
    from quantcio.quant.rsmt import ExcessOverride, fit_market_model

    s = load_settings()
    model = fit_market_model(build_panel(synthetic_price_panel(n_tickers=15, n_days=900, seed=8), s), s, run_cv=False)
    t = model.panel.pred["ticker"].iloc[0]
    ov = ExcessOverride(mu_excess=0.004, skill_ok=True, model="logit", metrics={"excess_oos_auc": 0.53})
    out = model.predict_many([t], overrides={t: ov}, hold_days=20)
    dist, fc = out[t]
    assert fc.blend == {"logit": 1.0}
    # role separation (docs/RSMT.md): skill_ok certifies RSMT-Q's distribution, the direction skill is the model's
    assert fc.direction_skill_ok is True and fc.skill_ok == model.skill_ok and fc.relative_edge == 0.0
    assert fc.offsets["mu_excess_model"] == pytest.approx(0.004)
    assert fc.cv_metrics["excess_oos_auc"] == pytest.approx(0.53)
    # the served excess mean is the override (the plausibility clamp does not bind at 0.4%)
    assert fc.mu_excess == pytest.approx(0.004, abs=1e-9)
    plain = model.predict_many([t], hold_days=20)[t][1]
    assert plain.blend.get("rsmt") == 1.0 and plain.skill_ok == model.skill_ok
    ov_mkt = ExcessOverride(mu_excess=0.0, skill_ok=False, model="x", market_p_up=0.7)
    fc2 = model.predict_many([t], overrides={t: ov_mkt}, hold_days=20)[t][1]
    from scipy import stats

    assert fc2.mu_market > 0
    assert fc2.offsets["mu_market_served"] == pytest.approx(fc2.sigma_market * stats.norm.ppf(0.7), rel=1e-6)


# ------------------------------------------------------------------------------------------------ PIT panel membership


def test_build_panel_honours_pit_membership():
    from quantcio.quant.panel import build_panel

    n_days = 900
    bench = make_prices(n=n_days, start_price=2500.0, seed=11)
    prices = {f"{100000 + i:06d}": make_prices(n=n_days, seed=40 + i) for i in range(14)}
    prices["REQ"] = make_prices(n=n_days, seed=99)
    prices["DEL"] = make_prices(n=n_days, seed=98).iloc[:500]  # an early member that was later delisted
    cal = bench.index
    reb = pd.DatetimeIndex(pd.Series(cal, index=cal).groupby([cal.year, cal.month]).max().to_numpy())
    rows = []
    for d in reb:
        members = (
            [f"{100000 + i:06d}" for i in range(12)] + ["DEL"]
            if d < reb[len(reb) // 2]
            else [f"{100000 + i:06d}" for i in range(2, 14)]
        )
        rows += [{"date": d, "ticker": t} for t in members]
    mem = pd.DataFrame(rows)
    pp = PricePanel(Market.KR, prices, {"KOSPI": bench}, {t: "KOSPI" for t in prices}, membership=mem)
    s = load_settings()
    pdata = build_panel(pp, s)
    assert pdata.pit and pdata.n_non_member > 0
    tr = pdata.train
    assert "REQ" not in set(tr["ticker"])  # never a member -> never a training row
    early = tr[tr["date"] < reb[len(reb) // 2]]
    assert set(early["ticker"]) <= {f"{100000 + i:06d}" for i in range(12)} | {"DEL"}
    late = tr[tr["date"] > reb[len(reb) // 2 + 1]]
    assert not {"100000", "100001"} & set(late["ticker"])
    served = set(pdata.pred["ticker"])
    assert "REQ" in served and "100013" in served
    assert "DEL" not in served  # a delisted former member is not served
    assert "100000" in served  # a former member that still trades may be requested: served
    req = pdata.pred.set_index("ticker").loc["REQ"]
    assert not req["degraded"]  # a requested non-member is standardised against the members, not dropped
    assert "member" not in tr.columns
    assert any("PIT" in n for n in pdata.notes)


def test_pipeline_uses_pit_universe_and_falls_back_loudly(monkeypatch, tmp_path):
    import quantcio.data.pit as pit_mod
    from quantcio import pipeline

    s = load_settings(data={"cache_dir": str(tmp_path / "c")}, macro={"enabled": False})
    cache = DiskCache(tmp_path / "c")
    n_days = 900
    bench = make_prices(n=n_days, start_price=2500.0, seed=12)
    prices = {f"{100000 + i:06d}": make_prices(n=n_days, seed=60 + i) for i in range(20)}
    cal = bench.index
    reb = pd.DatetimeIndex(pd.Series(cal, index=cal).groupby([cal.year, cal.month]).max().to_numpy())
    mem = pd.DataFrame(
        [{"date": d, "entity": t, "ticker": t, "name": t, "rank": 1, "rank_value": 1.0} for d in reb for t in prices]
    )
    uni = pit_mod.PITUniverse(
        Market.KR, 20, mem, prices, {"KOSPI": bench}, {t: "KOSPI" for t in prices}, pd.DataFrame(), {}, ["note"]
    )
    monkeypatch.setattr(pit_mod, "load_pit_universe", lambda *a, **k: uni)
    panel, notes, universe = pipeline.training_panel(Market.KR, s, cache, ())
    assert universe == set(prices) and panel.membership is not None and any("PIT" in n for n in notes)

    def boom(*a, **k):
        raise RuntimeError("marcap offline")

    monkeypatch.setattr(pit_mod, "load_pit_universe", boom)
    fallback = PricePanel(Market.KR, prices, {"KOSPI": bench}, {t: "KOSPI" for t in prices})
    monkeypatch.setattr(pipeline.data_base, "load_price_panel", lambda *a, **k: fallback)
    panel, notes, universe = pipeline.training_panel(Market.KR, s, cache, ())
    assert universe is None and panel is fallback
    assert any("PIT" in n and "marcap offline" in n for n in notes)


def test_i18n_notes_in_english(monkeypatch, tmp_path):
    from quantcio import pipeline

    s = load_settings(report={"language": "en"}, data={"cache_dir": str(tmp_path / "c")})
    assert pipeline.regime_text(make_prices(n=260, start_price=2500.0, seed=3), s).split("MA200 ")[1] in {
        "above",
        "below",
    }
    fit = pipeline.MarketFit(market=Market.KR, error="boom")
    assert pipeline.market_diagnostics(fit, s)[0].startswith("⚠ RSMT-Q KR fit failed")
    assert "--no-llm" in pipeline.NO_LLM_NOTE_EN
    assert date.today()  # keep the import used


# ------------------------------------------------------------------------------------------------ protocol v4


def test_v4_label_is_the_simple_beta_adjusted_excess():
    from quantcio.modeling.dataset import DATASET_VERSION, excess_labels

    r, m, b = (
        pd.Series([math.log(1.10), math.log(0.80)]),
        pd.Series([math.log(1.02), math.log(0.95)]),
        pd.Series([1.5, 0.5]),
    )
    y, y_log = excess_labels(r, m, b)
    assert y.tolist() == pytest.approx([0.10 - 1.5 * 0.02, -0.20 - 0.5 * (-0.05)])
    assert y_log.tolist() == pytest.approx(
        [math.log(1.10) - 1.5 * math.log(1.02), math.log(0.80) - 0.5 * math.log(0.95)]
    )
    assert DATASET_VERSION == 5


def test_v4_primary_is_the_top_quintile_net_excess_by_hand():
    dates = pd.to_datetime(["2021-01-29", "2021-02-26"])
    top_sets = ({0, 1, 2, 3, 4}, {2, 3, 4, 5, 6})  # two new names on date 2 -> one-way turnover 2/5
    rows = []
    for d, top in zip(dates, top_sets, strict=True):
        for i in range(25):
            rows.append({"date": d, "entity": f"e{i}", "score": 1.0 + i / 100 if i in top else -i, "y_e": 0.01 * i})
    f = pd.DataFrame(rows)
    got = mt.per_date_top_ye_net(f, "score", cost_bps=30.0)
    for d, top, turnover in zip(dates, top_sets, (1.0, 2 / 5), strict=True):
        g = f[f["date"] == d]
        want = 0.01 * np.mean(sorted(top)) - g["y_e"].mean() - turnover * 30.0 / 1e4
        assert got[d] == pytest.approx(want)
    ev = mt.evaluate(f.assign(r=f["y_e"]), 30.0)
    assert ev[mt.PRIMARY]["mean"] == pytest.approx(got.mean()) and ev[f"{mt.PRIMARY}_ann"] == pytest.approx(
        got.mean() * 12
    )
    assert mt.per_date_primary(f, "score", 30.0).equals(got)


def test_v42_equal_scores_are_no_view_not_a_row_order_book():
    dates = pd.to_datetime(["2021-01-29", "2021-02-26", "2021-03-31"])
    rows = []
    for j, d in enumerate(dates):
        for i in range(25):  # date 2: every score equal (an all-zero L1 logit); y_e rises with row order
            rows.append({"date": d, "entity": f"e{i}", "score": 0.0 if j == 1 else float(i), "y_e": 0.01 * i})
    got = mt.per_date_top_ye_net(pd.DataFrame(rows), "score", cost_bps=30.0)
    assert got[dates[1]] == 0.0  # not the mean of the first five rows minus the date mean
    want = 0.01 * np.mean(range(20, 25)) - 0.01 * np.mean(range(25)) - 1.0 * 30.0 / 1e4  # full rebuild after it
    assert got[dates[2]] == pytest.approx(want)


def test_v42_inner_cv_voids_a_configuration_without_a_view(planted, msettings, monkeypatch):
    data = sel._Data(planted)
    outer, _, _, _ = sel.make_contexts(data)
    end = outer[-1].train_end
    folds = sel.inner_folds(data, end, sel.BO_INNER_BLOCKS)
    c = next(x for x in cmod.default_candidates() if x.name == "logit_bo")

    class Flat:
        def predict(self, frame):
            return np.zeros(len(frame))

    monkeypatch.setattr(cmod.Candidate, "fit", lambda self, train, params, scheme, ctx: Flat())
    prim, ic = sel._inner_series(data, c, {"C": 1e-4, "l1_ratio": 1.0}, "none", folds, msettings, 2)
    assert prim.empty and ic.empty
    assert np.isnan(sel._score_inner(data, c, {"C": 1e-4, "l1_ratio": 1.0}, "none", folds, msettings, 2)[0])
    cfg = sel._bo_config(
        [{"params": {"C": 1e-4}, "scheme": "none", "mean": float("nan"), "series": prim, "ic": ic}], c, 3
    )
    assert cfg.params == dict(c.grid[0]) and not cfg.bo  # a void-only record falls back, never picks it


def test_v4_magnitude_scheme_weights_by_distance_from_the_median():
    from quantcio.modeling.events import SCHEMES, scheme_weights

    assert "magnitude" in SCHEMES
    assert "magnitude" in cmod._RANK_SCHEMES and "magnitude" in cmod._REG_SCHEMES
    d = pd.to_datetime(["2021-01-29"] * 41 + ["2021-02-26"] * 41)
    y = np.r_[np.linspace(-0.2, 0.2, 41), np.linspace(-0.02, 0.02, 41)]
    f = pd.DataFrame({"date": d, "y_e": y})
    w = scheme_weights(f, "magnitude", None, None)
    for day in pd.unique(f["date"]):
        m = (f["date"] == day).to_numpy()
        assert w[m].mean() == pytest.approx(1.0)  # per-date mean 1: a calm date counts as much as a wild one
        mid = np.argmin(np.abs(y[m]))
        assert w[m][mid] == pytest.approx(0.0) and w[m].argmax() in (0, 40)  # the median row weighs nothing


def test_kr_modeling_universe_is_its_own_setting():
    """docs/MODELING.md 80.8: the stock-selection model uses modeling.kr_universe_n / kr_cost_bps; the factor backtest
    keeps backtest.kr_universe_n."""
    from quantcio.config import load_settings
    from quantcio.modeling.dataset import cost_bps_for, universe_params

    s = load_settings(modeling={"kr_universe_n": 333, "kr_cost_bps": 44.0}, backtest={"kr_universe_n": 150})
    assert universe_params(Market.KR, s)[0] == 333 and cost_bps_for(Market.KR, s) == 44.0
    assert universe_params(Market.US, s)[0] == s.backtest.us_universe_n
    assert cost_bps_for(Market.US, s) == s.backtest.cost_bps_us
    assert load_settings().modeling.kr_universe_n == 400 and load_settings().backtest.kr_universe_n == 200


# ------------------------------------------------------------------------------------------------ protocol v4.1 (82)


def test_bo_candidates_search_regularised_spaces():
    bo = {c.name: c for c in cmod.default_candidates() if c.search}
    assert set(bo) == {
        "lgbm_rank_bo",
        "lgbm_reg_bo",
        "et_bo",
        "logit_bo",
        "ridge_bo",
        "rf_bo",
        "xgb_clf_bo",
        "cat_clf_bo",
        "mlp_bo",
        "rsmt_v3_bo",
    }
    for c in bo.values():
        assert c.n_trials == 40 and not c.diagnostic and c.simplicity
        if c.name == "rsmt_v3_bo":  # stacked: tuned inside run_rsmt on the stacked score, weekly rows like rsmt_v3
            assert not c.tuned and c.rows == "W" and c.name in cmod.STACKED and "alpha" in c.search
            continue
        assert c.tuned and c.rows == "M" and "scheme" in c.search and "magnitude" in c.search["scheme"][1]
    assert {"C", "l1_ratio"} <= set(bo["logit_bo"].search) and "a" in bo["ridge_bo"].search
    assert {"l1", "l2", "min_gain", "bagging", "feature_frac"} <= set(bo["xgb_clf_bo"].search)
    assert {"l2", "depth", "feature_frac"} <= set(bo["cat_clf_bo"].search) and "alpha" in bo["mlp_bo"].search
    assert {"min_leaf_blocks", "mu_shrink_k", "gamma_shrink_k", "max_depth"} <= set(bo["rsmt_v3_bo"].search)
    lg = bo["lgbm_rank_bo"].search
    for key in ("l1", "l2", "leaf_frac", "feature_frac", "bagging", "max_depth", "min_gain", "lr", "n_estimators"):
        assert key in lg
    assert {"max_features", "leaf_frac", "max_depth", "bootstrap", "max_samples"} <= set(bo["et_bo"].search)


def test_lgbm_defaults_are_unchanged_for_the_old_candidates():
    """The regularisation keys of docs 82 default to the values every other LightGBM candidate always used."""
    ctx = types.SimpleNamespace(threads=2, seed=0)
    kw = cmod._lgbm_common({"leaves": 15, "leaf_frac": 0.01}, 10_000, ctx)
    assert kw["subsample"] == 0.8 and kw["colsample_bytree"] == 0.8 and kw["reg_lambda"] == 1.0
    assert kw["reg_alpha"] == 0.0 and kw["max_depth"] == -1 and kw["min_split_gain"] == 0.0
    kw2 = cmod._lgbm_common({"l1": 2.0, "l2": 5.0, "max_depth": 4, "min_gain": 0.05}, 10_000, ctx, gain_scale=2.0)
    assert kw2["reg_alpha"] == 2.0 and kw2["reg_lambda"] == 5.0 and kw2["max_depth"] == 4
    assert kw2["min_split_gain"] == pytest.approx(0.1)


def test_one_se_rule_prefers_the_simplest_near_tie():
    entries = [
        {"params": {"leaf_frac": 0.005, "leaves": 31}, "mean": 0.010},
        {"params": {"leaf_frac": 0.030, "leaves": 15}, "mean": 0.009},
        {"params": {"leaf_frac": 0.030, "leaves": 7}, "mean": 0.008},
    ]
    assert sel._simplest(entries, (("leaf_frac", +1), ("leaves", -1))) == 2
    assert sel._simplest(entries[:2], (("leaf_frac", +1), ("leaves", -1))) == 1


def test_tune_bayes_is_nested_deterministic_and_inside_the_space(planted, msettings, monkeypatch):
    data = sel._Data(planted)
    outer, _, _, _ = sel.make_contexts(data)
    end = outer[-1].train_end
    c = dataclasses.replace(next(x for x in cmod.default_candidates() if x.name == "lgbm_rank_bo"), n_trials=6)
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    orig = cmod.Candidate.fit

    def spy(self, train, params, scheme, ctx):
        seen.append((train["t_end"].max(), ctx.cutoff))
        return orig(self, train, params, scheme, ctx)

    monkeypatch.setattr(cmod.Candidate, "fit", spy)
    a = sel.tune(data, c, end, msettings, 2)
    b = sel.tune(data, c, end, msettings, 2)
    assert a.params == b.params and a.scheme == b.scheme  # fixed TPE seed
    assert a.bo["trials"] == 6 and a.bo["inner_blocks"] == len(sel.inner_folds(data, end, sel.BO_INNER_BLOCKS))
    assert a.bo["within_1se"] >= 1 and a.inner_auc <= a.bo["best_inner"] + 1e-12
    assert seen and all(t < cut for t, cut in seen)  # every inner fit's labels end before its validation block
    assert all(cut <= end for _, cut in seen)  # validation blocks lie inside the outer training window
    for k, spec in c.search.items():
        v = a.scheme if k == "scheme" else a.params[k]
        if spec[0] == "cat":
            assert v in spec[1]
        else:
            assert spec[1] <= v <= spec[2]


def test_v42_regularisation_keys_default_to_the_old_values():
    ctx = types.SimpleNamespace(threads=2, seed=0)
    xk = cmod._xgb_common({"depth": 3}, 10_000, ctx)
    assert xk["subsample"] == 0.8 and xk["colsample_bytree"] == 0.8 and xk["reg_lambda"] == 1.0
    assert xk["reg_alpha"] == 0.0 and xk["gamma"] == 0.0
    ck = cmod._cat_common({"depth": 4}, ctx)
    assert ck["l2_leaf_reg"] == 3.0 and ck["rsm"] == 1.0
    assert cmod._cat_common({"l2": 10.0, "feature_frac": 0.5}, ctx)["rsm"] == 0.5


def test_rsmt_v3_bo_is_tuned_on_the_stacked_score(planted, msettings, monkeypatch):
    cands = _cands("mom_12_1", "ridge", "rsmt_v3_bo")
    cands = [dataclasses.replace(c, n_trials=3) if c.name == "rsmt_v3_bo" else c for c in cands]
    res = sel.run_selection(
        planted, msettings, DiskCache(msettings.cache_path), candidates=cands, threads=2, use_cache=False, max_outer=1
    )
    row = res.candidate("rsmt_v3_bo")
    cfgs = [c for c in row["configs"].values() if c]
    assert cfgs and all(c["bo"]["trials"] == 3 for c in cfgs)
    space = next(c for c in cmod.default_candidates() if c.name == "rsmt_v3_bo").search
    for cfg in cfgs:
        for k, spec in space.items():
            v = cfg["params"][k]
            assert v in spec[1] if spec[0] == "cat" else spec[1] <= v <= spec[2]
    assert row["outer"][mt.PRIMARY]["mean"] is not None


def _fake_inner(table: dict, n: int = 24):
    """``selection._inner_series`` stand-in: (sorted params, scheme) -> (mean, sd) of n per-date values."""

    def fake(data, cand, params, scheme, folds, settings, threads):
        mean, sd = table[(tuple(sorted(params.items())), scheme)]
        idx = pd.date_range("2020-01-31", periods=n, freq="ME")
        z = np.tile([1.0, -1.0], n // 2)
        return pd.Series(mean + sd * z / np.std(z, ddof=1), index=idx), pd.Series(0.01, index=idx)

    return fake


def test_v43_grid_tuner_keeps_the_simplest_within_one_se(planted, msettings, monkeypatch):
    """docs/MODELING.md 84.1: stage 1 keeps the simplest grid point within one SE of the best (scheme 'none'), stage 2
    keeps 'none' unless it is more than one SE below the best scheme."""
    data = sel._Data(planted)
    outer, _, _, _ = sel.make_contexts(data)
    ridge = next(c for c in cmod.default_candidates() if c.name == "ridge")  # alpha: larger = simpler
    sd = 0.01  # SE = sd / sqrt(24) = 0.0020
    t = {
        ((("a", 0.01),), "none"): (0.0100, sd),
        ((("a", 0.1),), "none"): (0.0090, sd),
        ((("a", 1.0),), "none"): (0.0085, sd),  # within one SE of the best: the simplest point
        ((("a", 1.0),), "event_down"): (0.0095, sd),  # better than 'none' by less than one SE: 'none' kept
        ((("a", 1.0),), "winsor"): (0.0080, sd),
        ((("a", 1.0),), "magnitude"): (0.0090, sd),
    }
    monkeypatch.setattr(sel, "_inner_series", _fake_inner(t))
    cfg = sel.tune(data, ridge, outer[-1].train_end, msettings, 2)
    assert cfg.params == {"a": 1.0} and cfg.scheme == "none" and cfg.inner_auc == pytest.approx(0.0085)
    g, s = cfg.one_se["grid"], cfg.one_se["scheme"]
    assert g["best_params"] == {"a": 0.01} and g["within_1se"] == 3 and g["se"] == pytest.approx(sd / np.sqrt(24))
    assert s["best_scheme"] == "event_down" and s["within_1se"] == 4 and cfg.tried == 6
    t[((("a", 1.0),), "magnitude")] = (0.0130, sd)  # more than one SE above 'none': adopted
    cfg = sel.tune(data, ridge, outer[-1].train_end, msettings, 2)
    assert cfg.params == {"a": 1.0} and cfg.scheme == "magnitude"


def test_v43_bo_prefers_scheme_none_within_one_se():
    c = next(x for x in cmod.default_candidates() if x.name == "lgbm_reg_bo")
    idx = pd.date_range("2020-01-31", periods=36, freq="ME")
    z = np.tile([1.0, -1.0], 18) / np.std(np.tile([1.0, -1.0], 18), ddof=1)

    def rec(p, scheme, mean):
        return {"params": p, "scheme": scheme, "mean": mean, "series": pd.Series(mean + 0.01 * z, index=idx), "ic": z}

    cfg = sel._bo_config(
        [
            rec({"leaf_frac": 0.05, "leaves": 7}, "magnitude", 0.0100),  # best, simplest parameters, but weighted
            rec({"leaf_frac": 0.002, "leaves": 63}, "none", 0.0090),  # within one SE (0.0017) and unweighted
            rec({"leaf_frac": 0.01, "leaves": 15}, "none", 0.0050),
        ],
        c,
        3,
    )
    assert cfg.scheme == "none" and cfg.params == {"leaf_frac": 0.002, "leaves": 63}
    assert cfg.bo["within_1se"] == 2 and cfg.bo["best_scheme"] == "magnitude" and cfg.bo["trials"] == 3


def test_v43_factor_fm_whole_history_is_the_simplest_window():
    assert sel._simple_value("lookback", 0) == np.inf and sel._simple_value("lookback", 36) == 36.0
    c = next(x for x in cmod.default_candidates() if x.name == "factor_fm")
    recs = [
        {"params": {"lookback": lb}, "scheme": "none", "mean": m} for lb, m in ((36, 0.010), (60, 0.009), (0, 0.009))
    ]
    assert sel._simplest(recs, c.simplicity) == 2


def test_v43_every_tuned_grid_declares_its_simplicity_order():
    for c in cmod.default_candidates():
        if not c.tuned or c.search:
            continue
        vary = {k for k in set().union(*[g.keys() for g in c.grid]) if len({repr(g.get(k)) for g in c.grid}) > 1}
        assert vary <= {k for k, _ in c.simplicity}, c.name


def test_v43_dataset_fingerprint_covers_the_market_loaders():
    """docs/MODELING.md 84.5: a loader fix changes the data, so the dataset cache key and the frozen-window digest
    identity (both ``_code_fingerprint(market)``) cover that market's loader modules."""
    from pathlib import Path

    from quantcio.modeling import dataset as dsm

    kr, us = dsm._code_fingerprint(Market.KR), dsm._code_fingerprint("US")
    assert len({kr, us, dsm._code_fingerprint()}) == 3
    data = Path(dsm.__file__).resolve().parent.parent / "data"
    for files in dsm.LOADER_MODULES.values():
        assert all((data / f).is_file() for f in files)
