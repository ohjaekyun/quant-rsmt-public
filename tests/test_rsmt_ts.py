"""Research fork of RSMT-Q time-series extensions (scripts/rsmt_ts.py, docs/RSMT.md section 6) - offline, synthetic."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import make_prices

import quantcio.quant.rsmt as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rsmt_ts as T  # noqa: E402 - the research module lives in scripts/


def _frame(n_dates: int = 120, per: int = 25, seed: int = 0, oblique: bool = False) -> pd.DataFrame:
    """Pooled synthetic panel: n_dates clusters x per rows, offsets z / v, two regime inputs x1 / x2 and noise x3.
    oblique: sigma doubles on the diagonal x1 + x2 > 0 (no single axis carries it fully)."""
    rng = np.random.default_rng(seed)
    n = n_dates * per
    dates = np.repeat(pd.bdate_range("2019-01-01", periods=n_dates, freq="20B"), per)
    x1, x2, x3 = rng.normal(size=(3, n))
    z = rng.normal(size=n)
    v = rng.normal(0, 0.3, n)
    sig = 0.05 * np.exp(v) * (np.where(x1 + x2 > 0, 2.0, 1.0) if oblique else 1.0)
    shock = np.repeat(rng.normal(0, 0.01, n_dates), per)
    y = 0.002 * z + shock + sig * rng.standard_normal(n)
    return pd.DataFrame(
        {"date": dates, "x1": x1, "x2": x2, "x3": x3, "z": z, "v": v, "y_e": y, "w": 1.0, "m": rng.normal(0, 0.03, n)}
    )


def _kw(tmp_path: Path, feats: list[str]) -> dict:
    return dict(
        partition_features=feats,
        alpha=0.05,
        min_leaf=300,
        max_depth=1,
        null_sims=2000,
        boot_sims=199,
        min_leaf_dates=10,
        cache_dir=tmp_path,
        multiple_testing="simes",
    )


def test_fit_offsets_multi_with_one_column_is_the_production_fit():
    rng = np.random.default_rng(1)
    n = 3000
    z, v = rng.normal(size=n), rng.normal(0, 0.4, n)
    leaf = (rng.random(n) > 0.5).astype(np.int64)
    y = 0.01 * leaf + 0.003 * z + 0.04 * np.exp(0.8 * v) * rng.standard_normal(n)
    w = np.ones(n)
    ref = R.fit_offsets(y, z, v, w, leaf, 0.0, 1.0, 50, 1e-10, (-0.5, 2.0))
    g, e, _, ok = T.fit_offsets_multi(y, z, v[:, None], w, leaf, 0.0, np.array([1.0]), [(-0.5, 2.0)], tol=1e-10)
    assert ok and g == pytest.approx(ref.gamma, abs=1e-7) and e[0] == pytest.approx(ref.eta, abs=1e-6)


def test_fit_offsets_multi_recovers_the_sigma_equation():
    rng = np.random.default_rng(2)
    n = 40000
    z, v, a, b = rng.normal(size=n), rng.normal(0, 0.3, n), rng.normal(size=n), rng.normal(size=n)
    y = 0.002 * z + 0.05 * np.exp(1.0 * v + 0.3 * a - 0.2 * b) * rng.standard_normal(n)
    vm = np.column_stack([v, a, b])
    g, e, _, ok = T.fit_offsets_multi(
        y, z, vm, np.ones(n), np.zeros(n, np.int64), 0.0, np.array([1.0, 0.0, 0.0]), [(-0.5, 2.0)] + [(-3.0, 3.0)] * 2
    )
    assert ok
    assert e == pytest.approx([1.0, 0.3, -0.2], abs=0.03)


def test_rsmtreex_without_switches_is_the_production_tree(tmp_path):
    df = _frame(seed=3)
    ref = R.RSMTree(**_kw(tmp_path, ["x1", "x2", "x3"])).fit_frame(df)
    with T.installed():
        tx = R.RSMTree(**_kw(tmp_path, ["x1", "x2", "x3"])).fit_frame(df)
    assert isinstance(tx, T.RSMTreeX) and not tx.extended
    a = ref.predict_params(df, df["z"], df["v"])
    b = tx.predict_params(df, df["z"], df["v"])
    for k in ("mu_e", "sigma_e", "leaf_id"):
        np.testing.assert_array_equal(a[k], b[k])
    assert R.RSMTree is T._BaseTree  # restored


def test_sigma_terms_are_fitted_and_served(tmp_path):
    df = _frame(seed=4)
    rng = np.random.default_rng(5)
    df["h1"] = rng.normal(size=len(df))
    df["y_e"] = 0.002 * df["z"] + 0.05 * np.exp(df["v"] + 0.4 * df["h1"]) * rng.standard_normal(len(df))
    with T.installed(sigma_terms=["h1"]):
        tx = R.RSMTree(**_kw(tmp_path, ["x3"])).fit_frame(df)
    assert tx.sigma_equation()["h1"] == pytest.approx(0.4, abs=0.05)
    p = tx.predict_params(df, df["z"], df["v"])
    ratio = p["sigma_e"] / p["sigma_leaf"]
    np.testing.assert_allclose(np.log(ratio), tx.eta_v * df["v"] + tx.x_theta[0] * df["h1"], atol=1e-10)


def test_oblique_group_finds_the_diagonal_split(tmp_path):
    df = _frame(seed=6, oblique=True)
    with T.installed(groups=[("x1", "x2")]):
        tx = R.RSMTree(**_kw(tmp_path, ["x3"])).fit_frame(df)
    assert tx.n_splits == 1
    assert tx.root.split_var == "(+0.71·x1 +0.71·x2)"  # 45 degrees = the true boundary direction
    tests = {t.variable: t for t in tx.root.tests}
    assert tests["(+0.71·x1 +0.71·x2)"].null == "joint-bootstrap" and tests["(+0.71·x1 +0.71·x2)"].p_adj < 0.05
    p = tx.predict_params(df, df["z"], df["v"])  # routing new rows through the stored standardisation
    hi = (df["x1"] + df["x2"]).to_numpy() > 0.3
    lo = (df["x1"] + df["x2"]).to_numpy() < -0.3
    assert p["sigma_leaf"][hi].mean() > 1.6 * p["sigma_leaf"][lo].mean()


def test_joint_max_bootstrap_dominates_and_is_calibrated():
    dirs = T.direction_grid(2)
    rng = np.random.default_rng(7)
    pvals = []
    for rep in range(120):
        n = 400
        y = rng.standard_normal(n)
        s, _, _ = R.node_scores(y)
        bih = R.inv_sqrt_psd(s.T @ s / n)
        x = rng.normal(size=(n, 2)) @ dirs.T
        clusters = np.repeat(np.arange(40), 10)
        stat = max(R.sup_lm(x[:, j], s @ bih, 0.1)[0] for j in range(x.shape[1]))
        p, draws = T.joint_max_bootstrap(x, s, bih, np.ones(n), np.ones(n), clusters, 0.1, 199, rep, stat)
        pvals.append(p)
        if rep == 0:  # the max over directions dominates one direction under the same multipliers
            _, one = T.joint_max_bootstrap(x[:, :1], s, bih, np.ones(n), np.ones(n), clusters, 0.1, 199, rep, stat)
            assert (draws >= one - 1e-12).all()
    rate = float(np.mean(np.array(pvals) <= 0.10))
    assert 0.02 <= rate <= 0.20, rate


def test_direction_grid_and_features():
    d2, d3 = T.direction_grid(2), T.direction_grid(3)
    assert d2.shape == (8, 2) and d3.shape == (13, 3)
    np.testing.assert_allclose(np.linalg.norm(d3, axis=1), 1.0)
    cal = T.calendar_columns(pd.Series(pd.to_datetime(["2024-01-01", "2024-04-01"])))
    assert cal.loc[0, "cal_s1"] == pytest.approx(0.0) and cal.loc[0, "cal_c1"] == pytest.approx(1.0)
    assert cal.loc[1, "cal_s4"] == pytest.approx(math.sin(2 * math.pi * 4 * 91 / 365.25))
    px = make_prices(n=600, seed=8)
    bench = make_prices(n=600, seed=9)
    e = T.ticker_extras(px, bench)
    assert e[["har_5", "har_66", "har_252", "d20_vol_ratio", "d20_mkt_vol", "d20_dist_ma200"]].iloc[-1].notna().all()
    assert e["har_252"].iloc[:250].isna().all()  # 252 returns needed, data <= t only


def test_joint_max_bootstrap_row_path_matches_the_matmul_path(monkeypatch):
    rng = np.random.default_rng(11)
    n = 300
    s, _, _ = R.node_scores(rng.standard_normal(n))
    bih = R.inv_sqrt_psd(s.T @ s / n)
    x = rng.normal(size=(n, 2)) @ T.direction_grid(2).T
    clusters = np.repeat(np.arange(30), 10)
    args = (x, s, bih, np.ones(n), np.ones(n), clusters, 0.1, 199, 3, 5.0)
    p1, d1 = T.joint_max_bootstrap(*args)
    monkeypatch.setattr(T, "BOOT_FLOATS", 1)  # forces per-row multipliers, one draw per batch
    p2, d2 = T.joint_max_bootstrap(*args)
    assert p1 == p2
    np.testing.assert_allclose(d1, d2, rtol=1e-9)


def test_event_columns_are_point_in_time():
    days = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    us_days = pd.bdate_range("2024-01-01", "2024-03-31")
    events = pd.concat(
        {
            "us": pd.DataFrame({"vix": np.arange(len(us_days), dtype=float), "vrp": 0.0}, index=us_days),
            "gpr": pd.DataFrame({"gpr": np.arange(len(days), dtype=float)}, index=days),
            "epu": pd.DataFrame({"epu": np.arange(len(days), dtype=float)}, index=days),
        },
        axis=1,
    )
    t = pd.Series(pd.to_datetime(["2024-02-14", "2024-02-14"]))
    us = T.event_columns(t.iloc[:1], "US", events)
    kr = T.event_columns(t.iloc[1:], "KR", events)
    vix_on = float(events["us"]["vix"].loc["2024-02-14"])
    assert us["ev_vix"].iloc[0] == vix_on  # US: that session's close
    assert kr["ev_vix"].iloc[0] == vix_on - 1  # KR: the previous US session (it closes after Seoul)
    assert us["ev_gpr"].iloc[0] == float(events["gpr"]["gpr"].loc["2024-02-06"])  # 8 calendar days
    assert us["ev_epu"].iloc[0] == float(events["epu"]["epu"].loc["2024-02-12"])  # 2 calendar days


def test_step_mask_selects_the_every_fifth_session_grid():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import rsmt_history as H

    dates = pd.bdate_range("2024-01-01", periods=60)
    tr = pd.DataFrame({"date": np.repeat(dates, 2), "ticker": ["A", "B"] * 60})
    back = np.repeat(np.arange(60)[::-1], 2)
    tr["on_grid"] = back % 20 == 0
    m = H.step_mask(tr, 5)
    assert m.sum() == 24 and (back[m] % 5 == 0).all() and m[tr["on_grid"].to_numpy()].all()
