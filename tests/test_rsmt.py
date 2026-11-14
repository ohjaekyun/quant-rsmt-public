"""RSMT-Q MOB score tree, sup-LM null, offsets, CV and the served MarketModel (synthetic, offline)."""

from __future__ import annotations

import copy
import math
import re
import time

import numpy as np
import pandas as pd
import pytest
from conftest import make_prices
from scipy import stats

from quantcio.config import load_settings
from quantcio.data.base import PricePanel
from quantcio.models import DistributionForecast, Market
from quantcio.quant import distribution_ml_pillar
from quantcio.quant import rsmt as rsmt_mod
from quantcio.quant.distribution import ReturnDistribution, apply_edge, total_log_sd
from quantcio.quant.panel import RSMT_VARIANT_KEYS, build_panel
from quantcio.quant.rsmt import (
    MU_PLAUSIBLE_K,
    ZERO_SPLIT_INFO,
    CVResult,
    LayoutGate,
    MarketModel,
    RSMTOffsetError,
    RSMTree,
    _auc,
    _bb_sup_points,
    _evaluate,
    _logloss,
    apply_robust_gate,
    cluster_bootstrap_null,
    cross_validate,
    cv_splits,
    date_block_bootstrap,
    fit_market_model,
    fit_offsets,
    inv_sqrt_psd,
    node_scores,
    null_cache_file,
    plausibility_shift,
    robust_fold_layouts,
    skill_gate,
    split_scan,
    sup_lm,
    sup_lm_null,
)

# ----------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def null_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("rsmt_null")


# the ORIGINAL model (every docs/RSMT.md switch off, direction AUC gate): the tests below written for it pin these
# switches explicitly, whatever config/default.yaml selects (backward compatibility of the old path)
OLD_PATH = {
    "skill_metric": "direction",
    "overlap_step": 0,
    "multiple_testing": "bonferroni",
    "target_scale": "none",
    "gamma_mode": "global",
    "markets": {mk: dict.fromkeys(RSMT_VARIANT_KEYS) for mk in ("KR", "US")},
}


def _old_path(extra: dict | None = None) -> dict:
    """OLD_PATH with `extra` rsmt keys merged in (markets merged per market)."""
    out = copy.deepcopy(OLD_PATH)
    for k, v in (extra or {}).items():
        if k == "markets":
            for mk, mv in v.items():
                out["markets"].setdefault(mk, {}).update(mv)
        else:
            out[k] = v
    return out


@pytest.fixture(scope="session")
def rsmt_settings(tmp_path_factory):
    return load_settings(data={"cache_dir": str(tmp_path_factory.mktemp("qcache"))}, rsmt=OLD_PATH)


def _tree(names, null_dir, **kw) -> RSMTree:
    base = dict(alpha=0.05, min_leaf=50, max_depth=3, cache_dir=null_dir)
    base.update(kw)
    return RSMTree(names, **base)


def synthetic_price_panel(
    n_tickers: int = 40,
    n_days: int = 1800,
    planted: bool = True,
    seed: int = 0,
    extra: dict | None = None,
    failed: dict | None = None,
) -> PricePanel:
    """Market-model panel built from conftest.make_prices: r = beta*m + e per day.

    planted=True: e follows a persistent 2-state regime (calm: +0.12%/day, 1.2% vol; stressed:
    -0.30%/day, 3.0% vol, P(stay)=0.985) - a regime effect visible to the partition variables.
    planted=False: e is an iid random walk (no structure).
    """
    rng = np.random.default_rng(seed)
    bench = make_prices(n=n_days, start_price=2500.0, drift=0.0002, vol=0.011, seed=10_000 + seed)
    blr = np.log(bench["close"]).diff().fillna(0.0).to_numpy()
    prices, tb = {}, {}
    for i in range(n_tickers):
        base = make_prices(n=n_days, drift=0.0, vol=0.018, seed=seed * 1000 + i)
        eps = np.log(base["close"]).diff().fillna(0.0).to_numpy() / 0.018
        beta = 0.6 + 0.8 * rng.random()
        if planted:
            state = np.empty(n_days, dtype=int)
            s = int(rng.random() < 0.5)
            for d in range(n_days):
                if rng.random() > 0.985:
                    s = 1 - s
                state[d] = s
            e = np.where(state == 1, -0.0030, 0.0012) + np.where(state == 1, 0.030, 0.012) * eps
        else:
            e = 0.0003 + 0.018 * eps
        lr = beta * blr + e
        lr[0] = 0.0
        ratio = base["close"].iloc[0] * np.exp(np.cumsum(lr)) / base["close"].to_numpy()
        px = base.copy()
        px[["open", "high", "low", "close"]] = px[["open", "high", "low", "close"]].to_numpy() * ratio[:, None]
        t = f"{100000 + i:06d}"
        prices[t], tb[t] = px, "KOSPI"
    for t, px in (extra or {}).items():
        prices[t], tb[t] = px, "KOSPI"
    return PricePanel(Market.KR, prices, {"KOSPI": bench}, tb, failed=dict(failed or {}))


def clustered_null_price_panel(seed: int, n_tickers: int = 30, n_days: int = 1300, f_vol: float = 0.008) -> PricePanel:
    """NO conditional structure, but every name loads on one common daily residual factor
    (sector/size-like, intra-date ICC of y_e ~ 0.1-0.2 - realistic for 20-day residual returns)."""
    rng = np.random.default_rng(seed)
    bench = make_prices(n=n_days, start_price=2500.0, drift=0.0002, vol=0.011, seed=10_000 + seed)
    blr = np.log(bench["close"]).diff().fillna(0.0).to_numpy()
    factor = rng.normal(0.0, 1.0, n_days)
    prices, tb = {}, {}
    for i in range(n_tickers):
        base = make_prices(n=n_days, drift=0.0, vol=0.018, seed=seed * 1000 + i)
        eps = np.log(base["close"]).diff().fillna(0.0).to_numpy() / 0.018
        lr = (0.6 + 0.8 * rng.random()) * blr + 0.0002 + 0.016 * eps + f_vol * rng.normal(1.0, 0.5) * factor
        lr[0] = 0.0
        ratio = base["close"].iloc[0] * np.exp(np.cumsum(lr)) / base["close"].to_numpy()
        px = base.copy()
        px[["open", "high", "low", "close"]] = px[["open", "high", "low", "close"]].to_numpy() * ratio[:, None]
        t = f"{100000 + i:06d}"
        prices[t], tb[t] = px, "KOSPI"
    return PricePanel(Market.KR, prices, {"KOSPI": bench}, tb)


@pytest.fixture(scope="module")
def small_panel(rsmt_settings):
    short = make_prices(n=230, seed=77)  # too short for 12-1 momentum -> offset input missing
    return build_panel(
        synthetic_price_panel(n_tickers=20, n_days=1000, seed=3, extra={"777777": short}),
        rsmt_settings,
    )


@pytest.fixture(scope="module")
def small_model(small_panel, rsmt_settings) -> MarketModel:
    return fit_market_model(small_panel, rsmt_settings, run_cv=False)


# ---------------------------------------------------------------- (a) scores at the MLE


def test_node_scores_sum_to_zero_at_mle_with_offsets():
    rng = np.random.default_rng(0)
    n = 500
    y, z, v = rng.normal(0.01, 0.07, n), rng.normal(size=n), rng.normal(0, 0.4, n)
    w = rng.uniform(0.1, 1.0, n)
    S, mu, sigma2 = node_scores(y, z, v, w, gamma=0.013, eta=0.8)
    assert S.shape == (n, 2)
    scale = np.abs(S).sum(axis=0)
    np.testing.assert_allclose(S.sum(axis=0) / scale, 0.0, atol=1e-12)
    s2 = np.exp(1.6 * v)
    assert mu == pytest.approx(np.sum(w * (y - 0.013 * z) / s2) / np.sum(w / s2))
    # away from the MLE the score sum is not 0
    res = (y - 0.013 * z) - (mu + 0.01)
    assert abs(np.sum(w * res / s2)) / scale[0] > 1e-3


def test_every_fitted_node_is_at_its_mle(null_dir):
    rng = np.random.default_rng(1)
    n = 3000
    X = rng.uniform(size=(n, 3))
    z, v = rng.normal(size=n), rng.normal(0, 0.3, n)
    y = 0.03 * (X[:, 0] > 0.5) + 0.01 * z + 0.05 * np.exp(0.5 * v) * rng.normal(size=n)
    w = 0.5 ** (rng.uniform(0, 2000, n) / 504)
    tree = _tree(["a", "b", "c"], null_dir, min_leaf=150).fit(X, y, z, v, w)
    assert tree.n_splits >= 1
    stack = [tree.root]
    while stack:
        node = stack.pop()
        S, mu, sigma2 = node_scores(y[node.rows], z[node.rows], v[node.rows], w[node.rows], tree.gamma, tree.eta)
        np.testing.assert_allclose(S.sum(axis=0) / np.abs(S).sum(axis=0), 0.0, atol=1e-12)
        assert mu == pytest.approx(node.mu_raw) and sigma2 == pytest.approx(node.sigma2)
        if not node.is_leaf:
            stack += [node.left, node.right]


# ------------------------------------------------------------ (b) Q(k) via cumulative sums


def test_split_scan_matches_brute_force_at_every_k():
    rng = np.random.default_rng(2)
    n = 140
    x = np.round(rng.normal(size=n), 1)  # plenty of ties
    y = rng.normal(size=n) + 0.5 * (x > 0.3)
    S, _, _ = node_scores(y, w=rng.uniform(0.2, 1.0, n))
    scan = split_scan(x, S, min_leaf=10)
    B = S.T @ S / n
    Binv = np.linalg.inv(B)
    uniq = np.unique(x)
    mids = (uniq[1:] + uniq[:-1]) / 2
    expected = []
    for thr in mids:
        left = x <= thr
        nl = int(left.sum())
        if nl < 10 or n - nl < 10:
            continue
        sl = S[left].sum(axis=0)
        expected.append((nl, thr, n / (nl * (n - nl)) * sl @ Binv @ sl))
    assert len(expected) == scan.k.size > 5
    for (nl, thr, q), k, qq, val in zip(expected, scan.k, scan.q, scan.values):
        assert k == nl
        assert val == pytest.approx(thr)
        assert qq == pytest.approx(q, rel=1e-9)
    # identical to the LM_k of the fluctuation test at the same boundary: Q(k) = |W_k|^2 / (t(1-t))
    from quantcio.quant.rsmt import inv_sqrt_psd

    stat, t = sup_lm(x, S @ inv_sqrt_psd(B), trim=10 / n)
    assert stat == pytest.approx(max(q for _, _, q in expected), rel=1e-9)


# ---------------------------------------------------------------- null distribution


def test_null_cache_file_key_and_reuse(null_dir):
    path = null_cache_file(null_dir, 2, 0.10, 1000, 20000, 0)
    for token in ("k2", "trim0.1000", "grid1000", "sims20000", "seed0"):
        assert token in path.name
    null = sup_lm_null(2, 0.10, 20000, 1000, 0, cache_dir=null_dir)
    assert path.exists()
    assert null.shape == (20000,) and (np.diff(null) >= 0).all()
    np.testing.assert_array_equal(np.load(path), null)
    # Andrews (1993) sup-LM, 2 parameters, trimming 0.10: 5% critical value ~ 12
    assert 11.3 < np.quantile(null, 0.95) < 12.8
    other = sup_lm_null(2, 0.20, 2000, 1000, 0, cache_dir=null_dir)
    assert null_cache_file(null_dir, 2, 0.20, 1000, 2000, 0).exists()
    assert np.quantile(other, 0.95) < np.quantile(null, 0.95)  # the null depends on trim


def test_null_is_generated_in_batches_quickly(tmp_path):
    t0 = time.perf_counter()
    sup_lm_null(2, 0.10, 20000, 1000, 7, cache_dir=tmp_path)
    assert time.perf_counter() - t0 < 10.0


# --------------------------------------------------------------------- (c) calibration


def test_null_calibration_uniform_p_and_no_splits(null_dir):
    rng = np.random.default_rng(3)
    p_first, no_split = [], 0
    reps = 200
    for _ in range(reps):
        n = 600
        X = rng.normal(size=(n, 3))
        y = rng.normal(0.0, 0.05, n)
        z, v = rng.normal(size=n), rng.normal(0, 0.3, n)
        tree = _tree(["a", "b", "c"], null_dir).fit(X, y, z, v)
        p_first.append(tree.root_tests[0].p)
        no_split += tree.n_splits == 0
    assert stats.kstest(p_first, "uniform").pvalue > 0.01
    assert no_split / reps >= 0.90


def test_null_calibration_with_time_decay_weights_is_not_badly_inflated(null_dir):
    rng = np.random.default_rng(4)
    splits = 0
    reps = 200
    for _ in range(reps):
        n = 1500
        X = rng.normal(size=(n, 3))
        y = rng.standard_t(8, size=n) * 0.05
        w = 0.5 ** (rng.uniform(0, 2500, n) / 504)
        tree = _tree(["a", "b", "c"], null_dir, max_depth=1).fit(X, y, rng.normal(size=n), rng.normal(0, 0.3, n), w)
        splits += tree.n_splits > 0
    assert splits / reps <= 0.12


def test_clustered_null_panel_does_not_split_on_date_noise(rsmt_settings):
    """Regression: with a common residual shock per date, the independent-rows bridge null split on
    date-level variables (mkt_ret_60 / mkt_vol_60) in 12/12 null panels (p at the MC floor). The
    panel path (fit_frame -> date clusters -> multiplier bootstrap) must hold its size."""
    s = rsmt_settings
    boot_splits, bb_splits = 0, 0
    for seed in range(4):
        tr = build_panel(clustered_null_price_panel(100 + seed), s).train
        g = tr.groupby("date")["y_e"]
        icc_between = g.mean().var() - g.var().mean() / g.size().mean()
        assert icc_between > 0  # the dates really are clustered
        boot = RSMTree.from_settings(s, Market.US, max_depth=1).fit_frame(tr)
        assert boot.null_used == "bootstrap" and boot.n_clusters == tr["date"].nunique()
        assert {t.null for t in boot.root_tests} <= {"bootstrap", "none"}
        boot_splits += boot.n_splits
        # the same rows as independent observations (the old behaviour) - documents the failure mode
        bb = RSMTree.from_settings(s, Market.US, max_depth=1, null_method="bb").fit(
            tr[s.rsmt.partition_features], tr["y_e"], tr["z"], tr["v"], tr["w"], tr["m"]
        )
        bb_splits += bb.n_splits
    assert boot_splits == 0
    assert bb_splits >= 3


def test_weighted_null_with_time_ordered_variable_is_calibrated(null_dir):
    """Regression: time-decay weights + a time-ordered split variable (no clusters) inflated a nominal
    5% test to ~28% under the bridge null; the weighted fit now uses the multiplier bootstrap."""
    rng = np.random.default_rng(21)
    p_boot, p_bb = [], []
    for _ in range(120):
        n = 2000
        age = rng.uniform(0, 2555, n)
        w = 0.5 ** (age / 504)
        X = np.column_stack([age + rng.normal(0, 30, n), rng.normal(size=n)])
        y = rng.standard_t(8, n) * 0.05
        tree = _tree(["time", "iid"], null_dir, min_leaf=150, max_depth=1, use_offsets=False).fit(X, y, w=w)
        assert tree.null_used == "bootstrap"
        p_boot.append(tree.root_tests[0].p)
        # the same root statistic referred to the independent-rows bridge null (old behaviour)
        S, _, _ = node_scores(y, w=w)
        stat, _ = sup_lm(X[:, 0], S @ inv_sqrt_psd(S.T @ S / n), 0.10)
        assert stat == pytest.approx(tree.root_tests[0].stat)
        p_bb.append(float(np.mean(sup_lm_null(2, 0.10, 20000, 1000, 0, null_dir) >= stat)))
    assert np.mean(np.array(p_boot) < 0.05) <= 0.10
    assert np.mean(np.array(p_bb) < 0.05) >= 0.15  # the old null is badly inflated here


def test_cluster_bootstrap_reduces_to_bridge_null_and_paths_agree():
    rng = np.random.default_rng(22)
    n = 1500
    x = np.column_stack([rng.normal(size=n), np.round(rng.normal(size=n), 1), np.zeros(n)])
    S, _, _ = node_scores(rng.normal(size=n))
    bih = inv_sqrt_psd(S.T @ S / n)
    ones = np.ones(n)
    nulls = cluster_bootstrap_null(x, S, bih, ones, ones, np.arange(n), 0.10, 3000, seed=1)
    assert nulls[2] is None  # constant column: no admissible boundary
    for j in (0, 1):
        _, times = sup_lm(x[:, j], S @ bih, 0.10)
        ref = _bb_sup_points(times, 2, 20000, 3)
        # singleton clusters + unit weights: the discrete Brownian-bridge null (up to MC error and
        # the finite-sample spread of the squared-residual scores)
        assert np.quantile(nulls[j].draws, 0.95) == pytest.approx(np.quantile(ref, 0.95), rel=0.08)
    # cluster-level matmul path == row-level cumsum path (same multipliers)
    import quantcio.quant.rsmt as rsmt_mod

    codes = np.repeat(np.arange(75), 20)
    a = cluster_bootstrap_null(x[:, :2], S, bih, ones, ones, codes, 0.10, 500, seed=4)
    old = rsmt_mod.BOOT_BATCH_FLOATS
    try:
        rsmt_mod.BOOT_BATCH_FLOATS = 1000  # forces the row path
        b = cluster_bootstrap_null(x[:, :2], S, bih, ones, ones, codes, 0.10, 500, seed=4)
    finally:
        rsmt_mod.BOOT_BATCH_FLOATS = old
    for j in (0, 1):
        np.testing.assert_allclose(a[j].draws, b[j].draws, rtol=1e-9, atol=1e-9)
    # sequential (Besag-Clifford) p: a tiny statistic stops early with p ~ 1, a huge one uses all draws
    res = cluster_bootstrap_null(
        x[:, :2],
        S,
        bih,
        ones,
        ones,
        np.arange(n),
        0.10,
        999,
        seed=5,
        stats_obs=[0.0, 1e9],
    )
    assert res[0].stopped_early and res[0].p == pytest.approx(1.0) and res[0].draws.size < 999
    assert not res[1].stopped_early and res[1].p == pytest.approx(1 / 1000)


# ------------------------------------------------------------------------- (d) power


def test_power_mean_shift_selects_variable_and_break(null_dir):
    rng = np.random.default_rng(5)
    n = 2000
    X = rng.uniform(size=(n, 5))
    y = 0.6 * (X[:, 2] > 0.6) + rng.normal(size=n)
    tree = _tree([f"x{i}" for i in range(5)], null_dir, max_depth=1).fit(
        X, y, rng.normal(size=n), rng.normal(0, 0.2, n)
    )
    assert tree.root.split_var == "x2"
    assert tree.root.split_value == pytest.approx(0.6, abs=0.05)
    assert tree.root_p < 1e-3
    assert tree.describe()[0].startswith("x2 ≤ 0.")


def test_power_variance_shift_selects_variable_and_break(null_dir):
    rng = np.random.default_rng(6)
    n = 2000
    X = rng.uniform(size=(n, 5))
    y = rng.normal(size=n) * np.where(X[:, 4] > 0.35, 2.5, 1.0)
    tree = _tree([f"x{i}" for i in range(5)], null_dir, max_depth=1).fit(
        X, y, rng.normal(size=n), rng.normal(0, 0.2, n)
    )
    assert tree.root.split_var == "x4"
    assert tree.root.split_value == pytest.approx(0.35, abs=0.05)
    left, right = tree.leaves
    assert right.sigma / left.sigma == pytest.approx(2.5, rel=0.15)
    assert abs(left.mu - right.mu) < 0.3


# ------------------------------------------------------------ (e) selection unbiasedness


def test_selection_not_biased_toward_many_distinct_values(null_dir):
    rng = np.random.default_rng(7)
    reps, many = 300, 0
    kinds = set()
    for _ in range(reps):
        n = 600
        X = np.column_stack([rng.uniform(size=n), rng.integers(0, 5, size=n).astype(float)])
        tree = _tree(["many", "few"], null_dir, max_depth=1, use_offsets=False).fit(X, rng.normal(size=n))
        t = tree.root_tests
        kinds |= {t[0].null, t[1].null}
        many += min(range(2), key=lambda j: (t[j].p, -t[j].stat)) == 0
    assert {
        "continuum",
        "discrete",
    } <= kinds  # few-valued variable uses its exact discrete null
    assert 0.40 <= many / reps <= 0.60


# ----------------------------------------------------------------------- (f) offsets


def test_offsets_recovered(null_dir):
    rng = np.random.default_rng(8)
    n = 6000
    X = rng.uniform(size=(n, 3))
    z, v = rng.normal(size=n), rng.normal(0, 0.4, n)
    gamma, eta, sigma = 0.02, 0.6, 0.07
    mu_leaf = np.where(X[:, 0] > 0.5, 0.03, -0.01)
    noise = rng.standard_t(6, size=n) / math.sqrt(6 / 4)
    y = mu_leaf + gamma * z + sigma * np.exp(eta * v) * noise
    tree = _tree(["a", "b", "c"], null_dir, min_leaf=300, max_depth=2).fit(X, y, z, v)
    assert tree.gamma == pytest.approx(gamma, abs=0.004)
    assert tree.eta == pytest.approx(eta, abs=0.08)
    assert tree.offset_fit is not None and tree.offset_fit.converged
    assert tree.root.split_var == "a"
    p = tree.predict_params(X[:5], z[:5], v[:5])
    np.testing.assert_allclose(p["mu_e"], p["mu_leaf"] + tree.gamma * z[:5])
    np.testing.assert_allclose(p["sigma_e"], p["sigma_leaf"] * np.exp(tree.eta * v[:5]))
    for leaf in tree.leaves:
        assert 4.5 <= leaf.nu <= 12.0  # t6 tails: nu ~ 6 (kurtosis-matched)


def test_offsets_converge_when_z_is_correlated_with_leaves():
    """Regression: the coordinate (mu_L, gamma) step zig-zagged when z tracks leaf membership
    (momentum vs dd_250 / dist_ma200 leaves): 50 iterations, not converged, gamma 0.5% short of the
    joint MLE. The Frisch-Waugh step has the same fixed point and converges in a few iterations."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(23)
    n = 5000
    leaf = rng.integers(0, 4, n)
    z = 0.95 * (leaf - 1.5) / 1.1 + math.sqrt(1 - 0.95**2) * rng.normal(size=n)
    v = rng.normal(0, 0.4, n)
    v -= v.mean()
    w = 0.5 ** (rng.uniform(0, 2500, n) / 504)
    y = (
        np.array([-0.02, 0.0, 0.01, 0.03])[leaf]
        + 0.02 * z
        + 0.06 * np.exp(0.7 * v) * rng.standard_t(6, n) / math.sqrt(1.5)
    )
    fit = fit_offsets(y, z, v, w, leaf, 0.0, 1.0, max_iter=50, tol=1e-8)
    assert fit.converged and fit.n_iter <= 15

    def nll(p):  # independent joint MLE
        r = y - p[8] * z - p[:4][leaf]
        var = np.exp(p[4:8][leaf] + 2 * p[9] * v)
        return 0.5 * np.sum(w * (np.log(var) + r * r / var))

    ref = minimize(
        nll,
        np.r_[np.zeros(4), np.full(4, math.log(0.0036)), 0.0, 1.0],
        method="L-BFGS-B",
        options=dict(maxiter=20000, ftol=1e-15, gtol=1e-10),
    ).x
    assert fit.gamma == pytest.approx(ref[8], abs=2e-5)
    assert fit.eta == pytest.approx(ref[9], abs=2e-4)


def test_offset_errors_are_raised_never_silent(null_dir):
    rng = np.random.default_rng(9)
    n = 400
    X, y = rng.uniform(size=(n, 2)), rng.normal(size=n)
    z, v = rng.normal(size=n), rng.normal(size=n)
    nan = np.full(n, np.nan)
    for zz, vv in ((nan, v), (z, nan), (None, v), (z, None), (np.full(n, 0.3), v)):
        with pytest.raises(RSMTOffsetError):
            _tree(["a", "b"], null_dir).fit(X, y, zz, vv)
    partial = z.copy()
    partial[:7] = np.nan
    with pytest.raises(RSMTOffsetError, match="NaN 7/400"):
        _tree(["a", "b"], null_dir).fit(X, y, partial, v)
    tree = _tree(["a", "b"], null_dir).fit(X, y, z, v)
    with pytest.raises(RSMTOffsetError):
        tree.predict_params(X[:3], np.array([0.1, np.nan, 0.2]), v[:3])
    with pytest.raises(ValueError):
        bad = X.copy()
        bad[0, 0] = np.nan
        _tree(["a", "b"], null_dir).fit(bad, y, z, v)
    # the no-offset baseline needs neither
    assert _tree(["a", "b"], null_dir, use_offsets=False).fit(X, y).gamma == 0.0


# --------------------------------------------------------------------- (g) Bonferroni


def test_bonferroni_noise_variables_raise_p_adj(null_dir):
    rng = np.random.default_rng(10)
    n = 800
    signal = rng.uniform(size=n)
    y = 0.22 * (signal > 0.5) + rng.normal(size=n)
    noise = rng.uniform(size=(n, 9))
    small = _tree(["s", "n0"], null_dir, max_depth=1, use_offsets=False).fit(np.column_stack([signal, noise[:, 0]]), y)
    big = _tree(["s", *[f"n{i}" for i in range(9)]], null_dir, max_depth=1, use_offsets=False).fit(
        np.column_stack([signal, noise]), y
    )
    ts, tb = small.root_tests[0], big.root_tests[0]
    assert ts.p == tb.p  # same scores, same statistic, same null
    assert ts.p_adj == pytest.approx(min(1.0, 2 * ts.p))
    assert tb.p_adj == pytest.approx(min(1.0, 10 * tb.p))
    assert tb.p_adj > ts.p_adj
    assert big.root_p >= small.root_p


# ---------------------------------------------------------------- tree API / printing


def test_describe_rsmt_style_and_nan_routing(null_dir):
    rng = np.random.default_rng(11)
    n = 3000
    X = rng.uniform(size=(n, 3))
    y = 0.05 * (X[:, 0] > 0.5) - 0.05 * (X[:, 1] > 0.5) * (X[:, 0] > 0.5) + 0.05 * rng.normal(size=n)
    tree = _tree(["vol_ratio_20_120", "ret_1m", "rsi14"], null_dir, min_leaf=150).fit(
        X, y, rng.normal(size=n), rng.normal(0, 0.3, n), m=rng.normal(size=n)
    )
    lines = tree.describe()
    assert re.match(r"^vol_ratio_20_120 ≤ 0\.\d+ \(p=\d\.\de-\d\d, n=3000\)$", lines[0])
    assert any(
        re.match(
            r"^(│  |   )*(├─|└─) 리프#1 n=\d+ μe [+-]\d+\.\d% σe \d+\.\d% ν \d+\.\d$",
            ln,
        )
        for ln in lines
    )
    assert sum("리프#" in ln for ln in lines) == tree.n_leaves >= 3
    assert all(re.search(r"(├─|└─)", ln) for ln in lines[1:])
    # leaf paths and NaN routing (goes to the larger child, flagged)
    leaf_ids, missing = tree.apply(np.array([[0.9, 0.9, 0.5], [np.nan, 0.1, 0.5]]))
    assert tree.leaf_path(int(leaf_ids[0]))[0].startswith("vol_ratio_20_120 >")
    assert missing.tolist() == [False, True]
    assert all(-0.99 <= lf.rho <= 0.99 for lf in tree.leaves)


def test_zero_split_describe(null_dir):
    rng = np.random.default_rng(12)
    tree = _tree(["a"], null_dir).fit(
        rng.normal(size=(200, 1)),
        rng.normal(size=200),
        rng.normal(size=200),
        rng.normal(size=200),
    )
    assert tree.n_splits == 0
    assert tree.describe()[0].startswith("리프#1 n=200") and "분할 없음" in tree.describe()[0]


def test_tree_fit_speed_6000_rows_9_vars(null_dir):
    rng = np.random.default_rng(13)
    n = 6000
    X = rng.normal(size=(n, 9))
    y = 0.02 * (X[:, 0] > 0.3) + 0.07 * rng.standard_t(6, size=n) / math.sqrt(1.5)
    sup_lm_null(2, 0.10, 20000, 1000, 0, cache_dir=null_dir)  # warm cache (generation timed separately)
    t0 = time.perf_counter()
    tree = RSMTree([f"x{i}" for i in range(9)], cache_dir=null_dir).fit(
        X,
        y,
        rng.normal(size=n),
        rng.normal(0, 0.3, n),
        0.5 ** (rng.uniform(0, 2500, n) / 504),
        rng.normal(size=n),
    )
    assert time.perf_counter() - t0 < 5.0
    assert tree.n_train == n


# ------------------------------------------------------------------------- (j) CV purge


def test_cv_purge_no_label_overlap(small_panel, rsmt_settings):
    folds = cv_splits(small_panel.train, rsmt_settings.rsmt.cv_folds)
    tr = small_panel.train
    assert len(folds) == rsmt_settings.rsmt.cv_folds
    sizes = []
    for f in folds:
        train, test = tr.iloc[f.train_idx], tr.iloc[f.test_idx]
        assert len(train) and len(test)
        assert (train["t_end"] < f.test_start).all()  # no training label window reaches the test block
        assert (train["date"] < f.test_start).all()
        assert test["date"].between(f.test_start, f.test_end).all()
        assert f.n_purged > 0  # the last pre-block grid date is purged
        sizes.append(len(train))
    assert sizes == sorted(sizes)  # expanding window
    assert [f.test_start for f in folds] == sorted(f.test_start for f in folds)


# ------------------------------------------------------------------------- (i) health


def test_missing_offset_input_row_is_served_degraded(small_model):
    dist, fc = small_model.predict("777777")
    assert isinstance(dist, ReturnDistribution) and isinstance(fc, DistributionForecast)
    assert fc.degraded is True
    assert any(r.startswith("오프셋 입력 누락(z|v)") for r in fc.degraded_reasons)
    assert fc.offsets["z"] == 0.0
    ok = small_model.predict("100001")[1]
    assert ok.degraded is False and not ok.degraded_reasons
    health = small_model.health()
    assert any(re.match(r"^\[저하\] 오프셋 미적용 1/\d+행", h) and "777777" in h for h in health)


def test_collapsed_predictions_flag_every_row_degraded(small_model, small_panel, rsmt_settings):
    collapsed = copy.copy(small_panel)
    pred = small_panel.pred.copy()
    cols = [*small_panel.partition_features, "z", "v"]
    pred[cols] = pred[cols].iloc[[0] * len(pred)].to_numpy()
    pred.loc[:, "z"] = 0.0
    pred.loc[:, "v"] = 0.0
    collapsed.pred = pred
    model = MarketModel(collapsed, small_model.tree, CVResult(), rsmt_settings)
    assert model.degraded
    for t in ("100000", "100005", "100010", "100015"):
        fc = model.predict(t)[1]
        assert fc.degraded and any("예측 붕괴" in r for r in fc.degraded_reasons)
    assert any(h.startswith("[저하] 예측 붕괴") for h in model.health())


def test_collapse_among_served_rows_is_detected(small_model, small_panel, rsmt_settings):
    panel = copy.copy(small_panel)
    pred = small_panel.pred.copy()
    same = ["100000", "100001", "100002", "100003"]
    cols = [*small_panel.partition_features, "z", "v"]
    pred.loc[pred["ticker"].isin(same), cols] = pred.loc[pred["ticker"] == "100000", cols].to_numpy()
    panel.pred = pred
    model = MarketModel(panel, small_model.tree, CVResult(), rsmt_settings)
    assert not model.degraded  # universe as a whole is fine (4 of 21 identical)
    served = [model.predict(t)[1] for t in same]
    assert model.degraded
    assert any("예측 붕괴(서빙 종목)" in h for h in model.health())
    # regression: EVERY served row is flagged - the 4th row that triggers the check and the three
    # forecasts returned before it (flagged in place), never only rows served afterwards
    for fc in served:
        assert fc.degraded and any("예측 붕괴(서빙 종목)" in r for r in fc.degraded_reasons)
    assert all(s.degraded for s in model._served.values())
    assert model.predict("100004")[1].degraded  # later rows too
    # the collapse text carries counts (1/4 -> 2/5); earlier forecasts hold ONE current copy
    for fc in served:
        collapse = [r for r in fc.degraded_reasons if r.startswith("예측 붕괴(서빙 종목)")]
        assert len(collapse) == 1 and "2/5" in collapse[0]


def test_missing_benchmark_moments_are_flagged_not_silently_pooled(small_model, small_panel, rsmt_settings):
    panel = copy.copy(small_panel)
    panel.market_mu_by_bench = {}  # e.g. every row of that benchmark was dropped
    panel.market_nu_by_bench = {}
    model = MarketModel(panel, small_model.tree, CVResult(), rsmt_settings)
    fc = model.predict("100001")[1]
    assert fc.degraded and any("통합 시장 μ/ν 사용" in r for r in fc.degraded_reasons)
    assert fc.mu_market == pytest.approx(panel.market_mu_hist)
    assert not small_model.predict("100001")[1].degraded


def test_eta_at_its_bound_is_reported(small_model, small_panel, rsmt_settings):
    tree = copy.copy(small_model.tree)
    tree.eta = tree.eta_bounds[1]  # e.g. true eta 2.5 -> clipped to 2.0
    model = MarketModel(small_panel, tree, CVResult(), rsmt_settings)
    assert any(h.startswith("[정보] η=2.00 경계 도달") for h in model.health())
    assert not any("경계 도달" in h for h in small_model.health())


def test_leaf_id_out_of_range_raises(small_model):
    tree = small_model.tree
    with pytest.raises(KeyError):
        tree.leaf(0)  # used to wrap around to the LAST leaf
    with pytest.raises(KeyError):
        tree.leaf(tree.n_leaves + 1)
    assert tree.leaf(1).leaf_id == 1


def test_failed_panel_share_and_zero_split_info(rsmt_settings):
    s = load_settings(
        data={"cache_dir": rsmt_settings.data.cache_dir},
        rsmt=_old_path({"markets": {"KR": {"alpha": 1e-12}}}),
    )
    failed = {f"9{i:05d}": "HTTPError: 500" for i in range(10)}
    panel = build_panel(
        synthetic_price_panel(n_tickers=20, n_days=900, planted=False, seed=4, failed=failed),
        s,
    )
    model = fit_market_model(panel, s, run_cv=False)
    assert model.tree.n_splits == 0
    health = model.health()
    assert any(h.startswith("[정보]") and ZERO_SPLIT_INFO in h for h in health)  # info, not degraded
    assert any(h.startswith("[저하] 패널 로드 실패 10/30") for h in health)
    fc = model.predict("100000")[1]
    assert fc.degraded and any("패널 로드 실패" in r for r in fc.degraded_reasons)
    assert not any(ZERO_SPLIT_INFO in r for r in fc.degraded_reasons)


def test_zero_split_alone_is_not_degraded(rsmt_settings):
    s = load_settings(
        data={"cache_dir": rsmt_settings.data.cache_dir},
        rsmt=_old_path({"markets": {"KR": {"alpha": 1e-12}}}),
    )
    panel = build_panel(synthetic_price_panel(n_tickers=20, n_days=900, planted=False, seed=5), s)
    model = fit_market_model(panel, s, run_cv=False)
    assert model.tree.n_splits == 0 and not model.degraded
    assert model.predict("100003")[1].degraded is False


# ------------------------------------------------------------------- served mean / edge


def test_served_mean_and_direction_edge(small_model, small_panel, rsmt_settings):
    _, alone = small_model.predict("100002")
    rsmt_total = alone.beta * alone.mu_market + alone.offsets["mu_excess_rsmt"]
    assert alone.blend == {"rsmt": 1.0}  # the served mean is 100% RSMT-Q (no ensemble leg since 2026-10-02)
    assert alone.mu == pytest.approx(rsmt_total)
    assert alone.mu == pytest.approx(alone.beta * alone.mu_market + alone.mu_excess)
    # offsets are inside mu/sigma
    assert alone.offsets["z_mu"] == pytest.approx(small_model.gamma * alone.offsets["z"])
    assert alone.sigma_excess == pytest.approx(alone.offsets["sigma_leaf"] * alone.offsets["v_sigma"])
    # delta: only direction probabilities move
    edged = MarketModel(
        small_panel,
        small_model.tree,
        CVResult(direction_edge=0.3, relative_edge=0.1),
        rsmt_settings,
    )
    d_raw, f_raw = small_model.predict("100002")
    d_e, f_e = edged.predict("100002")
    assert f_e.prob_up == pytest.approx(apply_edge(d_raw.prob_up_raw, 0.3, d_raw.n_paths))
    assert f_e.prob_outperform == pytest.approx(apply_edge(d_raw.prob_outperform_raw, 0.1, d_raw.n_paths))
    for key in (
        "mu",
        "sigma",
        "expected_return",
        "q10",
        "q50",
        "q90",
        "var95",
        "cvar95",
    ):
        assert getattr(f_e, key) == getattr(f_raw, key)
    # deterministic seed: same ticker/as_of -> identical forecast
    assert small_model.predict("100002")[1] == f_raw


def test_unknown_ticker_raises(small_model):
    with pytest.raises(KeyError, match="RSMT-Q"):
        small_model.predict("999999")


# -------------------------------------------------------------------- (k) end to end


def test_end_to_end_planted_regime_has_skill(rsmt_settings):
    panel = build_panel(
        synthetic_price_panel(n_tickers=40, n_days=1800, planted=True, seed=0),
        rsmt_settings,
    )
    t0 = time.perf_counter()
    model = fit_market_model(panel, rsmt_settings)
    assert time.perf_counter() - t0 < 60.0
    assert model.skill_ok is True
    m = model.cv.metrics
    assert m["auc"] - m["base_auc"] >= 0.01 and m["logloss"] < m["base_logloss"]
    assert m["n_oos"] > 1000 and 0.6 < m["cover80"] < 0.95
    assert m["auc_gain_p05"] > 0 and m["n_oos_dates"] >= 40 and m["auc_rel"] > m["base_auc_rel"]
    assert abs(m["delta"]) <= 0.5 and abs(m["delta_rel"]) <= 0.5
    assert m["crps"] < m["base_crps"] and m["ic"] > 0
    assert model.tree.n_splits >= 1
    lines = model.diagnostics()
    assert re.match(r"^\w+ ≤ -?[\d.e+-]+ \(p=\d\.\de-\d\d, n=\d+\)$", lines[0])
    assert any(re.search(r"(├─|└─) 리프#\d+ n=\d+ μe [+-]\d+\.\d% σe \d+\.\d% ν \d+\.\d$", ln) for ln in lines)
    text = "\n".join(lines)
    for token in (
        "root p=",
        "γ=",
        "η=",
        "δ=",
        "AUC",
        "logloss",
        "CRPS",
        "cover80",
        "기준",
        "제외 행",
        "로드 실패",
        "날짜 군집 승수 부트스트랩",
    ):
        assert token in text
    assert model.tree.null_used == "bootstrap"  # panel path: date-clustered null
    # date-aware skill gate: every component passes, bootstrap over whole test dates (boot_sims)
    assert model.cv.gate == {
        "auc_gain": True,
        "logloss": True,
        "date_ci": True,
        "date_ci_required": True,
    }
    # fold-robust verdict: every layout of rsmt.robust_cv_folds ran, a majority passed
    cv = model.cv
    assert sorted(cv.layouts) == sorted(rsmt_settings.rsmt.robust_cv_folds) == [3, 4, 5]
    assert cv.skill_ok_primary and cv.robust_majority and cv.robust_pass >= 2
    primary = cv.layouts[rsmt_settings.rsmt.cv_folds]  # the primary run is reused, not re-run
    assert primary.auc_gain == m["auc_gain"] and primary.auc_gain_p05 == m["auc_gain_p05"]
    for k, lay in cv.layouts.items():
        assert lay.available and lay.n_oos > 0 and m[f"gate_folds_{k}"] == float(lay.skill_ok)
        assert m[f"auc_gain_p05_folds_{k}"] == lay.auc_gain_p05
    assert m["gate_folds_pass"] == cv.robust_pass and m["gate_folds_n"] == 3 and m["gate_folds_need"] == 2
    assert m["skill_ok_robust"] == 1.0 and m["skill_ok_primary"] == 1.0
    assert cv.seconds > 0
    assert m["n_boot"] == rsmt_settings.rsmt.boot_sims
    assert m["auc_gain"] == pytest.approx(m["auc"] - m["base_auc"])
    assert m["logloss_gain"] == pytest.approx(m["base_logloss"] - m["logloss"])
    assert m["logloss_gain_p05"] < m["logloss_gain"] and m["auc_gain_p05"] < m["auc_gain"]
    gate = next(ln for ln in lines if ln.startswith("스킬 게이트"))
    assert gate.startswith(f"스킬 게이트 통과(폴드 배치 과반 {cv.robust_pass}/3)")
    robust = next(ln for ln in lines if ln.startswith("폴드 강건성 게이트"))
    assert robust.startswith("폴드 강건성 게이트 통과") and all(f"{k}폴드" in robust for k in (3, 4, 5))
    assert any(ln.startswith("시장 드리프트 μ_m(20일)") for ln in lines)
    assert any(ln.startswith("μ 타당성 상한") for ln in lines)
    assert f"{m['auc_gain_p05']:+.3f}" in gate and "5% 하한" in gate
    assert f"B={rsmt_settings.rsmt.boot_sims}" in gate
    # every leaf spans >= min_leaf_dates sample dates
    assert all(lf.n_dates >= rsmt_settings.rsmt.min_leaf_dates for lf in model.tree.leaves)
    dist, fc = model.predict("100000")
    assert fc.skill_ok and fc.cv_metrics["auc"] == m["auc"] and fc.leaf_path
    assert fc.cv_metrics["auc_gain_p05"] == m["auc_gain_p05"]
    assert {f"gate_folds_{k}" for k in (3, 4, 5)} <= set(fc.cv_metrics)
    assert "logloss_gain_p05" in fc.cv_metrics
    # served over the holding horizon: the barrier covers fusion.target_horizon_days
    assert dist.hold_days == rsmt_settings.fusion.target_horizon_days
    b = dist.barrier(-0.07, 0.12)
    assert b.p_take_profit + b.p_stop_loss + b.p_time_exit == 1.0
    assert b.horizon_days == rsmt_settings.fusion.target_horizon_days
    assert fc.horizon_days == rsmt_settings.horizon_days


def test_end_to_end_random_walk_has_no_skill(rsmt_settings):
    panel = build_panel(
        synthetic_price_panel(n_tickers=40, n_days=1800, planted=False, seed=1),
        rsmt_settings,
    )
    model = fit_market_model(panel, rsmt_settings)
    assert model.skill_ok is False
    assert model.predict("100000")[1].skill_ok is False
    assert model.cv.robust_n == 3 and not model.cv.robust_majority
    assert model.predict("100000")[1].cv_metrics["skill_ok_robust"] == 0.0
    assert any("스킬 게이트 미달" in h for h in model.health())
    assert not model.cv.gate["date_ci"] and model.cv.metrics["auc_gain_p05"] <= 0
    assert any(ln.startswith("스킬 게이트 미달") for ln in model.diagnostics())
    # the failed components are named in the health line
    assert any(h.startswith("[정보] CV 스킬 게이트 미달(") and "날짜 부트스트랩" in h for h in model.health())


# ------------------------------------------------------------- date-aware skill gate


def test_skill_gate_requires_the_date_bootstrap_bound():
    base = {
        "auc": 0.56,
        "base_auc": 0.54,
        "logloss": 0.690,
        "base_logloss": 0.692,
        "auc_gain_p05": -0.004,
    }
    ok, gate = skill_gate(base, 0.01, True)
    assert not ok and gate == {
        "auc_gain": True,
        "logloss": True,
        "date_ci": False,
        "date_ci_required": True,
    }
    assert skill_gate(base, 0.01, False)[0] is True  # CI reported, not required
    assert skill_gate({**base, "auc_gain_p05": 0.002}, 0.01, True)[0] is True
    assert skill_gate({**base, "auc_gain_p05": 0.002}, 0.03, True)[0] is False
    assert skill_gate({**base, "logloss": 0.692}, 0.01, False)[0] is False
    # a missing / NaN metric fails its component, never passes by default
    assert skill_gate({**base, "auc_gain_p05": float("nan")}, 0.01, True)[0] is False
    no_ci = dict(base)
    no_ci.pop("auc_gain_p05")
    assert skill_gate(no_ci, 0.01, True)[0] is False
    assert skill_gate({}, 0.01, False)[0] is False


def test_date_block_bootstrap_equals_resampling_whole_dates():
    rng = np.random.default_rng(31)
    n_dates, n = 25, 600
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.sort(rng.integers(0, n_dates, n)) * 28, "D")
    y = (rng.random(n) < 0.55).astype(float)
    p = np.round(np.clip(0.5 + 0.1 * (y - 0.5) + rng.normal(0, 0.1, n), 0.01, 0.99), 3)
    pb = np.round(np.clip(0.5 + rng.normal(0, 0.1, n), 0.01, 0.99), 3)  # many ties
    codes, uniq = pd.factorize(pd.Series(dates))
    counts = rng.multinomial(uniq.size, np.full(uniq.size, 1 / uniq.size), size=6)
    bs = date_block_bootstrap(p, pb, y, dates, 0, counts=counts, eps=1e-6)
    for b in range(6):
        idx = np.concatenate([np.tile(np.nonzero(codes == d)[0], int(c)) for d, c in enumerate(counts[b])])
        auc_gain = _auc(p[idx], y[idx]) - _auc(pb[idx], y[idx])
        ll_gain = _logloss(pb[idx], y[idx], 1e-6) - _logloss(p[idx], y[idx], 1e-6)
        assert bs.auc_gain[b] == pytest.approx(auc_gain, abs=1e-12)
        assert bs.logloss_gain[b] == pytest.approx(ll_gain, abs=1e-12)
    # full-sample gains (every date once) are the plain pooled metrics
    assert bs.auc_gain_hat == pytest.approx(_auc(p, y) - _auc(pb, y), abs=1e-12)
    assert bs.logloss_gain_hat == pytest.approx(_logloss(pb, y, 1e-6) - _logloss(p, y, 1e-6), abs=1e-12)
    # random draws: deterministic per seed, n_boot draws, informative model -> bound > 0
    a = date_block_bootstrap(p, pb, y, dates, 999, seed=3)
    assert a.n_boot == 999 and a.auc_gain.size == 999 and a.n_dates == uniq.size
    np.testing.assert_array_equal(a.auc_gain, date_block_bootstrap(p, pb, y, dates, 999, seed=3).auc_gain)
    assert a.auc_gain_p05 > 0 and a.logloss_gain_p05 > 0
    # small-cluster correction: the percentile's distance to the full-sample gain is widened by
    # c_G = sqrt(G/(G-1)) * t_{.95,G-1} / z_{.95} (> 1, -> 1 as G grows)
    g = uniq.size
    c = math.sqrt(g / (g - 1)) * stats.t.ppf(0.95, g - 1) / stats.norm.ppf(0.95)
    assert a.small_cluster_factor == pytest.approx(c) and c > 1.0
    assert a.auc_gain_q05 == pytest.approx(np.quantile(a.auc_gain, 0.05))
    assert a.auc_gain_p05 == pytest.approx(a.auc_gain_hat - c * (a.auc_gain_hat - a.auc_gain_q05))
    assert a.auc_gain_p05 < a.auc_gain_q05 < a.auc_gain_hat
    assert a.logloss_gain_p05 == pytest.approx(a.logloss_gain_hat - c * (a.logloss_gain_hat - a.logloss_gain_q05))
    one = date_block_bootstrap(p, pb, y, np.zeros(n), 999)  # a single date: no interval
    assert math.isnan(one.auc_gain_p05) and one.n_dates == 1


def test_date_block_bound_is_calibrated_where_a_row_bootstrap_is_not():
    """Regression: 1{r>0} is ~one independent draw per DATE (all names share the market move). A
    model whose extra signal is date-level noise must pass the 5% bound about 5% of the time when
    whole dates are resampled; resampling rows treats 1000 rows as independent and passes ~20%."""
    rng = np.random.default_rng(1)
    n_dates, n_names = 40, 25
    date_hits, row_hits = [], []
    for rep in range(150):
        m = rng.normal(0.004, 0.049, n_dates)
        beta = rng.uniform(0.6, 1.4, n_names)
        r = beta[None, :] * m[:, None] + rng.normal(0.006, 0.08, (n_dates, n_names))
        y = (r > 0).astype(float).ravel()
        vol = rng.lognormal(0.0, 0.3, n_dates)
        base = np.clip(0.5 + 0.05 / vol[:, None] + 0.01 * (beta[None, :] - 1), 0.01, 0.99)
        noise = 0.03 * rng.normal(size=(n_dates, 1)) + 0.005 * rng.normal(
            size=(n_dates, n_names)
        )  # no information about r
        p = np.round(np.clip(base + noise, 0.01, 0.99).ravel(), 3)
        pb = np.round(base.ravel(), 3)
        dates = np.repeat(np.arange(n_dates), n_names)
        date_hits.append(date_block_bootstrap(p, pb, y, dates, 299, seed=rep).auc_gain_p05 > 0)
        row_hits.append(date_block_bootstrap(p, pb, y, np.arange(y.size), 299, seed=rep).auc_gain_p05 > 0)
    assert np.mean(date_hits) <= 0.10
    assert np.mean(row_hits) >= 0.15


# ------------------------------------------------------------- min dates per child


def test_split_scan_requires_min_distinct_dates_per_child():
    rng = np.random.default_rng(41)
    n = 900
    x = np.round(rng.normal(size=n), 1)
    dates = rng.integers(0, 40, n)
    S, _, _ = node_scores(rng.normal(size=n) + 0.3 * (x > 0.5))
    free = split_scan(x, S, min_leaf=30)
    scan = split_scan(x, S, min_leaf=30, clusters=dates, min_clusters=36)
    assert 0 < scan.k.size < free.k.size
    order = np.argsort(x, kind="stable")
    for k in free.k:
        left, right = dates[order[:k]], dates[order[k:]]
        admissible = np.unique(left).size >= 36 and np.unique(right).size >= 36
        assert (k in scan.k) == admissible
    for k, dl, dr in zip(scan.k, scan.dates_left, scan.dates_right):
        assert dl == np.unique(dates[order[:k]]).size >= 36
        assert dr == np.unique(dates[order[k:]]).size >= 36
    # Q(k) is unchanged on the admissible boundaries
    q_free = dict(zip(free.k.tolist(), free.q.tolist()))
    for k, q in zip(scan.k, scan.q):
        assert q == pytest.approx(q_free[int(k)], rel=1e-12)
    assert split_scan(x, S, 30, clusters=dates, min_clusters=41).k.size == 0


def test_tree_never_isolates_a_handful_of_dates(null_dir):
    """Regression: a split on a date-level variable could isolate a few sample dates (enough ROWS,
    but a few dates = a few market shocks). Each child must span >= min_leaf_dates dates."""
    rng = np.random.default_rng(42)
    n_dates, per = 60, 40
    date = np.repeat(np.arange(n_dates), per)
    x_date = date.astype(float) + 0.5  # date-level variable (like mkt_vol_60)
    # effect on the last 8 dates (320 rows >= min_leaf; break at t=52/60, inside the sup-LM trim)
    y = rng.normal(size=date.size) + np.where(date >= 52, 1.0, 0.0)
    X = np.column_stack([x_date, rng.normal(size=date.size)])
    kw = dict(min_leaf=100, max_depth=1, use_offsets=False)
    free = _tree(["mkt", "noise"], null_dir, **kw).fit(X, y, cluster=date)
    assert free.root.split_var == "mkt" and free.root.right.n_dates == 8
    tree = _tree(["mkt", "noise"], null_dir, min_leaf_dates=12, **kw).fit(X, y, cluster=date)
    assert tree.n_splits == 1 and tree.root.split_var == "mkt"
    assert tree.root.left.n_dates >= 12 and tree.root.right.n_dates >= 12
    assert free.root.split_value == pytest.approx(52.0)  # isolates dates 52..59
    assert tree.root.split_value <= 48.0  # >= 12 dates right of the split
    # a node with < 2*min_leaf_dates dates is not split at all
    few = _tree(["mkt", "noise"], null_dir, min_leaf_dates=31, **kw).fit(X, y, cluster=date)
    assert few.n_splits == 0 and few.root.stop_reason == "dates < 2*min_leaf_dates"
    assert few.root.n_dates == n_dates


def test_from_settings_wires_min_leaf_dates_and_boot_sims(rsmt_settings):
    s = load_settings(
        data={"cache_dir": rsmt_settings.data.cache_dir},
        rsmt=_old_path({"min_leaf_dates": 7, "boot_sims": 333}),
    )
    tree = RSMTree.from_settings(s, Market.KR)
    assert tree.min_leaf_dates == 7 and tree.boot_sims == 333
    default = RSMTree.from_settings(rsmt_settings, Market.KR)
    assert default.min_leaf_dates == rsmt_settings.rsmt.min_leaf_dates
    assert default.boot_sims == rsmt_settings.rsmt.boot_sims


# ------------------------------------------------------------- holding horizon + batch serving


def _fresh_model(small_panel, small_model, rsmt_settings, pred=None) -> MarketModel:
    panel = copy.copy(small_panel)
    if pred is not None:
        panel.pred = pred
    return MarketModel(panel, small_model.tree, CVResult(), rsmt_settings)


def test_predict_simulates_over_the_holding_horizon(small_model, small_panel, rsmt_settings):
    model = _fresh_model(small_panel, small_model, rsmt_settings)
    H = rsmt_settings.fusion.target_horizon_days
    dist, fc = model.predict("100001")
    assert model.hold_days == H and dist.hold_days == H and dist.horizon == 20
    assert dist.r_paths.shape == (rsmt_settings.rsmt.n_paths, H)
    assert dist.barrier(-0.07, 0.12).horizon_days == H
    short, fc20 = model.predict("100001", hold_days=20)
    assert short.hold_days == 20 and short.barrier(-0.07, 0.12).horizon_days == 20
    assert fc20 == fc  # every h-horizon quantity is identical for any hold_days
    with pytest.raises(ValueError):
        model.predict("100001", hold_days=0)


def test_predict_many_serves_the_batch_and_returns_errors(small_model, small_panel, rsmt_settings):
    model = _fresh_model(small_panel, small_model, rsmt_settings)
    out = model.predict_many(["100001", "999999", "100002.KS", "100001", "777777"])
    assert list(out) == [
        "100001",
        "999999",
        "100002.KS",
        "777777",
    ]  # input order, deduped
    assert isinstance(out["999999"], KeyError) and "RSMT-Q" in str(out["999999"])
    d1, f1 = out["100001"]
    assert f1.blend == {"rsmt": 1.0}
    d2, f2 = out["100002.KS"]  # normalised lookup
    assert f2.blend == {"rsmt": 1.0}
    assert d1.hold_days == d2.hold_days == rsmt_settings.fusion.target_horizon_days
    assert not f1.degraded and not f2.degraded
    _, f7 = out["777777"]
    assert f7.degraded and any(r.startswith("오프셋 입력 누락") for r in f7.degraded_reasons)
    # identical to serving the same rows one at a time
    solo = _fresh_model(small_panel, small_model, rsmt_settings)
    assert solo.predict("100001")[1] == f1
    assert solo.predict("100002")[1] == f2
    assert model.predict_many([]) == {}


def test_predict_many_flags_a_batch_collapse_before_returning(small_model, small_panel, rsmt_settings):
    """Regression: with one-at-a-time serving the collapse among served rows is only visible from
    the 4th row, so the first rows were handed out un-degraded and patched in place later. The batch
    runs the checks over the whole batch first: every returned forecast is final."""
    pred = small_panel.pred.copy()
    same = ["100000", "100001", "100002", "100003"]
    cols = [*small_panel.partition_features, "z", "v"]
    pred.loc[pred["ticker"].isin(same), cols] = pred.loc[pred["ticker"] == "100000", cols].to_numpy()
    model = _fresh_model(small_panel, small_model, rsmt_settings, pred)
    out = model.predict_many([*same, "100004"])
    reasons = {t: list(out[t][1].degraded_reasons) for t in out}
    collapse = {t: [r for r in rs if r.startswith("예측 붕괴(서빙 종목)")] for t, rs in reasons.items()}
    assert all(len(c) == 1 and "2/5" in c[0] for c in collapse.values())
    assert len({c[0] for c in collapse.values()}) == 1  # one consistent batch-wide reason
    assert all(out[t][1].degraded for t in out)
    # nothing is patched afterwards: the forecasts are unchanged by later health checks
    model.health()
    assert {t: list(out[t][1].degraded_reasons) for t in out} == reasons
    assert any("예측 붕괴(서빙 종목)" in h for h in model.health())
    # a healthy batch of the same size is not flagged
    ok = _fresh_model(small_panel, small_model, rsmt_settings).predict_many(
        ["100005", "100006", "100007", "100008", "100009"]
    )
    assert not any(v[1].degraded for v in ok.values())


def test_null_memory_hit_still_writes_the_requested_cache_dir(tmp_path):
    """Integration regression: the in-process memory cache served a second cache_dir without writing its
    .npy, so test_null_cache_file_key_and_reuse failed whenever another test had computed the same null
    first (order-dependent full-suite failure after tests/test_pipeline.py was added)."""
    a, b = tmp_path / "a", tmp_path / "b"
    first = sup_lm_null(2, 0.10, 20000, 1000, 0, cache_dir=a)
    second = sup_lm_null(2, 0.10, 20000, 1000, 0, cache_dir=b)
    assert second is first
    for d in (a, b):
        assert null_cache_file(d, 2, 0.10, 1000, 20000, 0).exists()


# ------------------------------------------------------------ market drift prior (serving + CV)


def test_served_market_leg_uses_the_prior_shrunk_drift(small_model, small_panel, rsmt_settings):
    _, fc = small_model.predict("100003")
    s = rsmt_settings.rsmt.market_mu_shrink
    prior = math.log1p(rsmt_settings.rsmt.market_mu_prior_annual) * 20 / 252
    raw = small_panel.market_mu_raw_by_bench["KOSPI"]
    assert fc.mu_market == pytest.approx((1 - s) * raw + s * prior)
    line = small_model.drift_line()
    assert line.startswith("시장 드리프트 μ_m(20일) = 0.20×과거") and "KOSPI" in line and "연 6%" in line


def test_cv_folds_use_the_same_drift_rule(small_panel, rsmt_settings):
    """The CV folds build the market leg with the serving rule: with shrink = 1 every fold's mu_m is
    exactly the prior, so the baseline twin's mean (beta * mu_m + root mu) moves with the prior."""
    lo = load_settings(
        data={"cache_dir": rsmt_settings.data.cache_dir},
        rsmt=_old_path({"market_mu_shrink": 1.0, "market_mu_prior_annual": 0.0}),
    )
    hi = load_settings(
        data={"cache_dir": rsmt_settings.data.cache_dir},
        rsmt=_old_path({"market_mu_shrink": 1.0, "market_mu_prior_annual": 0.5}),
    )
    a = cross_validate(small_panel, lo, n_paths=200, robust_folds=[3])
    b = cross_validate(small_panel, hi, n_paths=200, robust_folds=[3])
    shift = math.log1p(0.5) * 20 / 252
    d = b.oos["base_mu"].to_numpy() - a.oos["base_mu"].to_numpy()
    beta = small_panel.train.set_index(["ticker", "date"]).loc[list(zip(a.oos["ticker"], a.oos["date"])), "beta"]
    np.testing.assert_allclose(d, beta.to_numpy() * shift, rtol=1e-9, atol=1e-12)


# ------------------------------------------------------------ plausibility clamp on mu


def test_plausibility_shift():
    assert plausibility_shift(0.01, 0.10) == 0.0
    assert plausibility_shift(0.05, 0.10) == 0.0  # exactly at the cap: not binding
    assert plausibility_shift(0.08, 0.10) == pytest.approx(-0.03)
    assert plausibility_shift(-0.09, 0.10) == pytest.approx(0.04)
    assert plausibility_shift(0.08, 0.10, k=1.0) == 0.0
    for bad in ((float("nan"), 0.1), (0.1, float("nan")), (0.1, -0.1)):
        with pytest.raises(ValueError):
            plausibility_shift(*bad)


def test_implausible_mean_is_clamped_and_reported(small_model, small_panel, rsmt_settings):
    """Regression: gamma*z, a small leaf's mean or an extreme stock-selection mean could push the served mean to
    any size (live 005930 E[R] +5%/20d at the Kelly cap). The served total mean is clamped to
    |mu| <= MU_PLAUSIBLE_K * sigma_h, recorded in offsets and in health - never silently."""
    model = _fresh_model(small_panel, small_model, rsmt_settings)
    _, ok = model.predict("100004")
    assert ok.offsets["mu_clamp_shift"] == 0.0 and ok.offsets["mu_cap"] == pytest.approx(MU_PLAUSIBLE_K * ok.sigma)
    assert not any("μ 타당성 상한" in h for h in model.health())

    def extreme(mu: float) -> rsmt_mod.ExcessOverride:
        return rsmt_mod.ExcessOverride(mu_excess=mu, skill_ok=True, model="x")

    up_d, up = model.predict("100005", override=extreme(1.0))  # +100% in 20 days
    dn_d, dn = model.predict("100006", override=extreme(-1.0))
    for d, fc, sign in ((up_d, up, 1.0), (dn_d, dn, -1.0)):
        assert fc.mu == pytest.approx(sign * MU_PLAUSIBLE_K * fc.sigma)  # exactly at the cap
        assert fc.sigma == pytest.approx(d.sd_log)
        assert fc.offsets["mu_cap"] == pytest.approx(MU_PLAUSIBLE_K * fc.sigma)
        assert fc.offsets["mu_total_unclamped"] == pytest.approx(fc.beta * fc.mu_market + sign)
        assert fc.offsets["mu_clamp_shift"] == pytest.approx(fc.mu - fc.offsets["mu_total_unclamped"])
        assert sign * fc.offsets["mu_clamp_shift"] < 0
        assert not fc.degraded  # a plausibility guard, not a data failure
    line = next(h for h in model.health() if "μ 타당성 상한" in h)
    assert line.startswith("[정보]") and "2/3" in line and "100005" in line and "100006" in line
    # re-serving the same ticker without the extreme mean clears it
    model.predict("100005")
    assert "100005" not in next(h for h in model.health() if "μ 타당성 상한" in h)


def test_implausible_leaf_mean_is_clamped_in_serving_and_in_cv(small_model, small_panel, rsmt_settings):
    tree = copy.deepcopy(small_model.tree)
    for lf in tree.leaves:
        lf.mu = 0.9  # a bubble leaf: +90% log return in 20 days
    model = MarketModel(small_panel, tree, CVResult(), rsmt_settings)
    hits, n = model.universe_clamp_count()
    assert hits == n == len(small_panel.pred)
    assert any(
        ln.startswith(f"μ 타당성 상한 |μ| ≤ {MU_PLAUSIBLE_K:g}·σ_h: 유니버스 최종일 {n}/{n}행")
        for ln in model.diagnostics()
    )
    _, fc = model.predict("100001")
    assert fc.mu == pytest.approx(MU_PLAUSIBLE_K * fc.sigma) and fc.expected_return < 0.25
    # the CV evaluates with the same clamp (so its metrics measure what is served)
    params = {
        "mu_e": np.array([0.9, 0.001]),
        "sigma_e": np.array([0.08, 0.08]),
        "mu_m": np.array([0.004, 0.004]),
        "sigma_m": np.array([0.05, 0.05]),
        "nu": np.array([8.0, 8.0]),
        "nu_m": np.array([10.0, 10.0]),
        "beta": np.array([1.1, 1.1]),
        "rho": np.array([0.1, 0.1]),
    }
    out = _evaluate(params, [1, 2], np.array([0.01, -0.02]), 20, 400)
    sd = total_log_sd(1.1, 0.08, 0.05, 0.1, 8.0, 10.0, 20)
    assert out["clamped"].tolist() == [1.0, 0.0]
    assert out["mu"][0] == pytest.approx(MU_PLAUSIBLE_K * sd)
    assert out["mu"][1] == pytest.approx(1.1 * 0.004 + 0.001)


# ------------------------------------------------------------ fold-robust skill gate


def _cv_primary(passing: bool) -> CVResult:
    m = {
        "auc": 0.56,
        "base_auc": 0.54,
        "logloss": 0.690,
        "base_logloss": 0.692,
        "auc_gain": 0.02,
        "logloss_gain": 0.002,
        "auc_gain_p05": 0.003 if passing else -0.004,
        "logloss_gain_p05": 0.0,
        "n_oos": 3000.0,
        "n_folds": 4.0,
        "n_oos_dates": 56.0,
        "n_boot": 999.0,
    }
    ok, gate = skill_gate(m, 0.01, True)
    assert ok is passing
    return CVResult(metrics=m, skill_ok=ok, gate=gate, n_folds_primary=4)


def _lay(k: int, ok: bool, available: bool = True) -> LayoutGate:
    return LayoutGate(
        k, ok and available, available, auc_gain=0.02 if ok else 0.004, auc_gain_p05=0.002 if ok else -0.01
    )


def test_apply_robust_gate_needs_a_strict_majority_of_layouts():
    # live KR 2026-09-24: 4 folds pass (bound +0.0006), 3 and 5 fail -> no skill
    cv = apply_robust_gate(_cv_primary(True), {3: _lay(3, False), 4: _lay(4, True), 5: _lay(5, False)})
    assert cv.skill_ok is False and cv.skill_ok_primary is True
    assert (cv.robust_pass, cv.robust_n, cv.robust_majority) == (1, 3, False)
    m = cv.metrics
    assert (m["gate_folds_3"], m["gate_folds_4"], m["gate_folds_5"]) == (0.0, 1.0, 0.0)
    assert m["gate_folds_pass"] == 1.0 and m["gate_folds_n"] == 3.0 and m["gate_folds_need"] == 2.0
    assert m["skill_ok_primary"] == 1.0 and m["skill_ok_robust"] == 0.0
    assert m["auc_gain_p05_folds_3"] == -0.01 and m["auc_gain_folds_4"] == 0.02
    assert any("기준 4폴드 배치는 게이트 통과이나 폴드 배치 과반 판정 1/3 → 스킬 불인정" in n for n in cv.notes)
    # a majority passes even though the primary layout itself failed
    cv = apply_robust_gate(_cv_primary(False), {3: _lay(3, True), 4: _lay(4, False), 5: _lay(5, True)})
    assert cv.skill_ok is True and cv.skill_ok_primary is False and cv.metrics["gate_folds_4"] == 0.0
    # 2 of 4 is not a strict majority; an unavailable layout counts as a failure
    four = {k: _lay(k, k in (3, 4)) for k in (3, 4, 5, 6)}
    cv = apply_robust_gate(_cv_primary(True), four)
    assert cv.skill_ok is False and cv.metrics["gate_folds_need"] == 3.0
    cv = apply_robust_gate(_cv_primary(True), {3: _lay(3, True), 4: _lay(4, True), 5: _lay(5, True, available=False)})
    assert cv.skill_ok is True and cv.metrics["gate_folds_5"] == 0.0
    assert any(n.startswith("5폴드 배치 CV 불가") for n in cv.notes)
    # no layout at all, or no primary metrics: never passes by default, an unavailable CV stays unavailable
    assert apply_robust_gate(_cv_primary(True), {}).skill_ok is False
    empty = apply_robust_gate(CVResult(), {3: _lay(3, True), 4: _lay(4, True), 5: _lay(5, True)})
    assert empty.skill_ok is False and not empty.available and empty.metrics == {}


def test_robust_fold_layouts_from_config(rsmt_settings):
    assert robust_fold_layouts(rsmt_settings) == [3, 4, 5]
    s = load_settings(data={"cache_dir": rsmt_settings.data.cache_dir}, rsmt=_old_path({"robust_cv_folds": [5, 3, 5]}))
    assert robust_fold_layouts(s) == [3, 5]
    s = load_settings(data={"cache_dir": rsmt_settings.data.cache_dir}, rsmt=_old_path({"robust_cv_folds": []}))
    assert robust_fold_layouts(s) == [s.rsmt.cv_folds]
    s = load_settings(data={"cache_dir": rsmt_settings.data.cache_dir}, rsmt=_old_path({"robust_cv_folds": [0, 4]}))
    with pytest.raises(ValueError, match="robust_cv_folds"):
        robust_fold_layouts(s)


def test_cross_validate_runs_every_layout_and_reuses_the_primary(small_panel, rsmt_settings, monkeypatch):
    cv = cross_validate(small_panel, rsmt_settings, n_paths=200)
    assert sorted(cv.layouts) == [3, 4, 5] and cv.n_folds_primary == 4
    assert cv.metrics["n_folds"] <= 4 and cv.layouts[4].auc_gain == cv.metrics["auc_gain"]
    assert cv.skill_ok == (cv.available and 2 * sum(lay.skill_ok for lay in cv.layouts.values()) > 3)
    # the layouts run concurrently; the result does not depend on it
    monkeypatch.setattr(rsmt_mod, "CV_LAYOUT_WORKERS", 1)
    seq = cross_validate(small_panel, rsmt_settings, n_paths=200)
    np.testing.assert_equal(seq.metrics, cv.metrics)
    assert {k: (v.skill_ok, v.auc_gain_p05) for k, v in seq.layouts.items()} == {
        k: (v.skill_ok, v.auc_gain_p05) for k, v in cv.layouts.items()
    }
    # an explicit single layout = the old single-layout gate; an empty config list is noted, not silent
    one = cross_validate(small_panel, rsmt_settings, n_paths=200, robust_folds=[4])
    assert one.skill_ok == one.skill_ok_primary and list(one.layouts) == [4]
    s = load_settings(data={"cache_dir": rsmt_settings.data.cache_dir}, rsmt=_old_path({"robust_cv_folds": []}))
    none = cross_validate(small_panel, s, n_paths=200)
    assert list(none.layouts) == [4] and any("robust_cv_folds 비어 있음" in n for n in none.notes)


def test_served_ml_pillar_multiplier_uses_the_robust_gate(small_model, small_panel, rsmt_settings):
    """Task: the served DistributionForecast.skill_ok (what quant.distribution_ml_pillar and fusion
    read) is the fold-robust verdict, not the primary layout's single-layout gate."""
    lone = apply_robust_gate(_cv_primary(True), {3: _lay(3, False), 4: _lay(4, True), 5: _lay(5, False)})
    model = MarketModel(small_panel, small_model.tree, lone, rsmt_settings)
    _, fc = model.predict("100002")
    assert model.skill_ok is False and fc.skill_ok is False
    raw = 0.5 * math.tanh(2 * fc.mu / fc.sigma) + 0.5 * (2 * fc.prob_up - 1)
    assert distribution_ml_pillar(fc) == pytest.approx(raw * 0.35)
    assert fc.cv_metrics["skill_ok_primary"] == 1.0 and fc.cv_metrics["skill_ok_robust"] == 0.0
    assert any(
        h.startswith("[정보] CV 스킬 게이트 미달(폴드 배치 과반 미달 1/3 (3폴드 미달 · 4폴드 통과 · 5폴드 미달))")
        for h in model.health()
    )
    assert model.gate_line().startswith("스킬 게이트 미달(폴드 배치 과반 1/3): 기준 4폴드 AUC 개선 +0.020 ≥ 0.010 충족")
    assert model.robust_line().startswith(
        "폴드 강건성 게이트 미달: 3폴드 미달(AUC 개선 +0.004, 하한 -0.010) · 4폴드 통과"
    )
    # the majority passes although the primary layout failed -> full multiplier
    maj = apply_robust_gate(_cv_primary(False), {3: _lay(3, True), 4: _lay(4, False), 5: _lay(5, True)})
    model2 = MarketModel(small_panel, small_model.tree, maj, rsmt_settings)
    _, fc2 = model2.predict("100002")
    assert fc2.skill_ok is True and distribution_ml_pillar(fc2) == pytest.approx(
        max(-1.0, min(1.0, 0.5 * math.tanh(2 * fc2.mu / fc2.sigma) + 0.5 * (2 * fc2.prob_up - 1)))
    )
    health = model2.health()
    assert any("기준 4폴드 배치는 게이트 미달이나 폴드 배치 과반 판정 2/3 → 스킬 인정" in h for h in health)
    assert not any("날짜 CI 미요구 설정" in h for h in health)  # the CI IS required: no false claim


def test_served_gate_keys_read_consistently_by_the_report(small_model, small_panel, rsmt_settings):
    """Contract with report/render.py ('모델 신뢰도' line): the served cv_metrics gate_folds_* keys give
    the same verdict as DistributionForecast.skill_ok (the renderer flags any disagreement)."""
    render = pytest.importorskip("quantcio.report.render")
    trust_view = getattr(render, "_trust_view", None)
    if trust_view is None:
        pytest.skip("report/render.py has no _trust_view")
    lone = apply_robust_gate(_cv_primary(True), {3: _lay(3, False), 4: _lay(4, True), 5: _lay(5, False)})
    _, fc = MarketModel(small_panel, small_model.tree, lone, rsmt_settings).predict("100002")
    view = trust_view("KR", fc)
    assert view["text"] == "3개 폴드 구성 중 1개만 통과 → 스킬 미확인" and view["cls"] == "no"
    assert view["detail"] == "3폴드 ✗ · 4폴드 ✓ · 5폴드 ✗"
    maj = apply_robust_gate(_cv_primary(False), {3: _lay(3, True), 4: _lay(4, False), 5: _lay(5, True)})
    _, fc2 = MarketModel(small_panel, small_model.tree, maj, rsmt_settings).predict("100002")
    view = trust_view("KR", fc2)
    assert view["text"] == "3개 폴드 구성 중 2개 통과 → 스킬 확인" and view["cls"] == "ok"


# ------------------------------------------------------------ report language (ko / en)

_HANGUL = re.compile(r"[가-힣]")


def _model_texts(model: MarketModel, served: list[str]) -> list[str]:
    for t in served:
        model.predict(t)
    return [*model.health(), *model.diagnostics(), *model.degraded_reasons, *model.panel.notes]


def test_english_model_texts_have_no_korean_and_keep_the_renderer_highlighting(rsmt_settings):
    """`--lang en`: every RSMT-Q health / diagnostics line, degraded reason and panel note is English, line for
    line the Korean output translated - same count, and the same report/render.py classes (pinned head line,
    always-kept line, highlighted alarm), so the English diagnostics box shows and flags the same lines."""
    from quantcio.report.render import _HEAD_LINE, _HOT_LINE, _KEEP_LINE

    en = load_settings(data={"cache_dir": rsmt_settings.data.cache_dir}, report={"language": "en"}, rsmt=OLD_PATH)
    short = make_prices(n=230, seed=77)
    failed = {"999990": "HTTPError: 500"}
    texts = {}
    for lang, s in (("ko", rsmt_settings), ("en", en)):
        pp = synthetic_price_panel(n_tickers=20, n_days=1000, seed=3, extra={"777777": short}, failed=failed)
        pd_ = build_panel(pp, s)
        lone = apply_robust_gate(_cv_primary(True), {3: _lay(3, False), 4: _lay(4, True), 5: _lay(5, False)}, lang=lang)
        lone.metrics.update(
            brier=0.249, base_brier=0.25, crps=0.03, base_crps=0.031, cover80=0.8, base_cover80=0.81, ic=0.02,
            auc_rel=0.52, base_auc_rel=0.5,
        )  # fmt: skip
        tree = RSMTree.from_settings(s, Market.KR).fit_frame(pd_.train)
        assert tree.lang == lang
        texts[lang] = _model_texts(MarketModel(pd_, tree, lone, s), ["777777", "100001"])
        texts[lang] += [*MarketModel(pd_, tree, CVResult(notes=["x"]), s).diagnostics()]
    ko, eng = texts["ko"], texts["en"]
    assert len(ko) == len(eng) and any(_HANGUL.search(x) for x in ko)
    assert not [x for x in eng if _HANGUL.search(x)]
    for a, b in zip(ko, eng):
        cls = [(bool(_HEAD_LINE.match(x)), bool(_KEEP_LINE.search(x)), bool(_HOT_LINE.search(x))) for x in (a, b)]
        assert cls[0] == cls[1], (a, b)
    # the missing-offset row is recognised in English too (served degraded, counted in the health line)
    assert any(x.startswith("[degraded] offsets not applied to 1/2 rows (served): 777777") for x in eng)
    assert any(x.startswith("[info] CV skill gate not passed (no fold-layout majority 1/3") for x in eng)
    assert any(x.startswith("RSMT-Q KR v1 · h=20d (stop / take-profit paths 60d)") for x in eng)
    assert any(x.startswith("load failed 1/22: 999990: HTTPError: 500") for x in eng)


def test_english_served_collapse_text_is_replaced_not_piled_up(small_model, small_panel, rsmt_settings):
    en = load_settings(data={"cache_dir": rsmt_settings.data.cache_dir}, report={"language": "en"}, rsmt=OLD_PATH)
    panel = copy.copy(small_panel)
    pred = small_panel.pred.copy()
    same = ["100000", "100001", "100002", "100003"]
    cols = [*small_panel.partition_features, "z", "v"]
    pred.loc[pred["ticker"].isin(same), cols] = pred.loc[pred["ticker"] == "100000", cols].to_numpy()
    panel.pred = pred
    model = MarketModel(panel, small_model.tree, CVResult(), en)
    served = [model.predict(t)[1] for t in same]
    model.predict("100004")
    for fc in served:
        collapse = [r for r in fc.degraded_reasons if r.startswith(rsmt_mod.SERVED_COLLAPSE_PREFIX_EN)]
        assert len(collapse) == 1 and "2/5" in collapse[0]
    assert any(h.startswith("[degraded] forecast collapse (served names)") for h in model.health())
    with pytest.raises(KeyError, match="no prediction row in the RSMT-Q KR model"):
        model.predict("999999")


def test_tree_describe_follows_the_language(small_model):
    ko = small_model.tree.describe("ko")
    en = small_model.tree.describe("en")
    assert ko == small_model.tree.describe()  # built with Korean settings
    assert len(ko) == len(en) and not any(_HANGUL.search(x) for x in en)
    assert any("leaf #" in x for x in en) and any("리프#" in x for x in ko)


# ------------------------------------------------ (v1.1) model switches and the distribution gate (docs/RSMT.md)


def _new_settings(tmp_cache: str, **rsmt_kw) -> object:
    """Settings with every docs/RSMT.md switch at the original value except `rsmt_kw` (distribution gate)."""
    return load_settings(data={"cache_dir": tmp_cache}, rsmt=_old_path({"skill_metric": "distribution", **rsmt_kw}))


def test_simes_adjustment_is_the_bh_step_up_and_rejects_where_bonferroni_does_not(null_dir):
    p = [0.02, 0.021, 0.022, 0.9]
    bonf = _tree(["a", "b", "c", "d"], null_dir).adjust_p(p)
    np.testing.assert_allclose(bonf, np.minimum(1.0, 4 * np.array(p)))
    bh = _tree(["a", "b", "c", "d"], null_dir, multiple_testing="simes").adjust_p(p)
    # BH step-up: min_{i >= rank} m p_(i) / i
    np.testing.assert_allclose(bh, [4 * 0.022 / 3, 4 * 0.022 / 3, 4 * 0.022 / 3, 0.9])
    simes = min(4 * pi / (i + 1) for i, pi in enumerate(sorted(p)))
    assert min(bh) == pytest.approx(simes) and simes < 0.05 < min(bonf)
    with pytest.raises(ValueError, match="multiple_testing"):
        _tree(["a"], null_dir, multiple_testing="holm")


def test_simes_null_calibration_is_not_inflated(null_dir):
    """Correlated regime variables (the positive-dependence case Simes is valid for): no conditional structure,
    the Simes node test must hold its level."""
    rng = np.random.default_rng(12)
    splits, reps = 0, 150
    for _ in range(reps):
        n = 600
        common = rng.normal(size=n)
        X = np.column_stack([common + rng.normal(0, 0.6, n) for _ in range(4)])
        y = rng.normal(0.0, 0.05, n)
        tree = _tree(["a", "b", "c", "d"], null_dir, max_depth=1, multiple_testing="simes").fit(
            X, y, rng.normal(size=n), rng.normal(0, 0.3, n)
        )
        splits += tree.n_splits > 0
    assert splits / reps <= 0.09  # nominal alpha = 0.05


def test_pca_split_variables_use_training_loadings(null_dir):
    rng = np.random.default_rng(5)
    n = 4000
    f = rng.normal(size=n)  # the latent regime factor behind three correlated variables
    X = np.column_stack([f + rng.normal(0, 0.3, n), f + rng.normal(0, 0.3, n), -f + rng.normal(0, 0.3, n)])
    X = np.column_stack([X, rng.normal(size=n)])
    y = np.where(f > 0.3, 0.03, -0.02) + rng.normal(0, 0.05, n)
    tree = _tree(["a", "b", "c", "d"], null_dir, multiple_testing="pca", pca_components=2, max_depth=1)
    tree.fit(X, y, rng.normal(size=n), rng.normal(0, 0.3, n))
    assert tree.split_features == ["PC1", "PC2"] and tree.root.split_var == "PC1"
    assert len(tree.root_tests) == 2 and all(t.p_adj == pytest.approx(min(1.0, 2 * t.p)) for t in tree.root_tests)
    pca = tree.pca
    np.testing.assert_allclose(pca["mean"], X.mean(axis=0))  # fitted on the training rows only
    assert abs(pca["loadings"][:, 0] @ np.array([1, 1, -1, 0])) > 1.5  # PC1 = the common factor
    assert tree.pca_lines()[0].startswith("PC1 (") and any(ln.startswith("PC1 (") for ln in tree.describe())
    # new rows are routed with the STORED loadings; a missing input is imputed at the mean and flagged
    new = X[:3].copy()
    leaf, missing = tree.apply(new)
    np.testing.assert_array_equal(leaf, tree.apply(X)[0][:3])
    new[0, 1] = np.nan
    leaf2, missing2 = tree.apply(new)
    assert not missing.any() and missing2.tolist() == [True, False, False]


def test_target_scale_splits_on_the_standardised_return_and_serves_returns(null_dir):
    rng = np.random.default_rng(9)
    n = 6000
    X = rng.uniform(size=(n, 2))
    v = rng.normal(0, 0.4, n)
    scale = 0.08 * np.exp(v)  # horizon vol known at t
    z = rng.normal(size=n)
    eps = np.where(X[:, 0] > 0.5, 0.25, -0.1) + 0.1 * z + rng.normal(size=n)
    y = scale * eps  # the leaf effect is a Sharpe-like shift: proportional to vol
    tree = _tree(["a", "b"], null_dir, min_leaf=300, max_depth=1, target_scale="ewma", eta0=0.0, eta_bounds=(-1.5, 1.0))
    tree.fit(X, y, z, v, scale=scale)
    assert tree.root.split_var == "a"
    assert tree.gamma == pytest.approx(0.1, abs=0.03)  # in scale units
    assert abs(tree.eta) < 0.1 and tree.eta_equivalent == pytest.approx(1.0 + tree.eta)  # EWMA vol ~ the right scale
    p = tree.predict_params(X[:5], z[:5], v[:5], scale[:5])
    np.testing.assert_allclose(p["mu_e"], scale[:5] * (p["mu_leaf"] + p["gamma"] * z[:5]))
    np.testing.assert_allclose(p["sigma_e"], scale[:5] * p["sigma_leaf"] * np.exp(tree.eta * v[:5]))
    assert "μe/σv" in tree.describe()[1]
    with pytest.raises(RSMTOffsetError):
        tree.predict_params(X[:5], z[:5], v[:5])  # the scale is required, never assumed
    with pytest.raises(ValueError, match="scale"):
        _tree(["a", "b"], null_dir, target_scale="ewma").fit(X, y, z, v)
    # constant scale, no offsets: identical served distribution to the raw-target fit
    raw = _tree(["a", "b"], null_dir, max_depth=0, use_offsets=False).fit(X, y)
    sc = _tree(["a", "b"], null_dir, max_depth=0, use_offsets=False, target_scale="ewma").fit(
        X, y, scale=np.full(n, 0.1)
    )
    a, b = raw.predict_params(X[:3]), sc.predict_params(X[:3], scale=np.full(3, 0.1))
    np.testing.assert_allclose(a["mu_e"], b["mu_e"]) and np.testing.assert_allclose(a["sigma_e"], b["sigma_e"])


def test_leaf_gamma_is_shrunk_toward_the_parent(null_dir):
    rng = np.random.default_rng(11)
    n_dates, per = 200, 40
    date = np.repeat(np.arange(n_dates), per)
    n = date.size
    X = np.column_stack([np.repeat(rng.uniform(size=n_dates), per), rng.normal(size=n)])  # a date-level regime
    z = rng.normal(size=n)
    hi = X[:, 0] > 0.5
    y = np.where(hi, 0.02, -0.02) + np.where(hi, 0.03, -0.01) * z + np.where(hi, 0.06, 0.04) * rng.normal(size=n)
    kw = dict(min_leaf=300, max_depth=1, min_leaf_dates=12)
    glob = _tree(["regime", "noise"], null_dir, **kw).fit(X, y, z, rng.normal(0, 0.1, n), cluster=date)
    loc = _tree(["regime", "noise"], null_dir, gamma_mode="leaf", gamma_shrink_k=50.0, **kw).fit(
        X, y, z, rng.normal(0, 0.1, n), cluster=date
    )
    assert loc.root.split_var == "regime" and loc.n_leaves == 2
    assert loc.root.gamma == pytest.approx(loc.gamma)  # the root keeps the global gamma
    for leaf in loc.leaves:
        n_eff = leaf.n_dates
        expect = (n_eff * leaf.gamma_hat + 50.0 * loc.gamma) / (n_eff + 50.0)
        assert leaf.gamma == pytest.approx(expect)
        assert min(leaf.gamma_hat, loc.gamma) - 1e-12 <= leaf.gamma <= max(leaf.gamma_hat, loc.gamma) + 1e-12
    by_side = {leaf.path[0].split()[1]: leaf for leaf in loc.leaves}
    assert by_side[">"].gamma > 0.018 and by_side["≤"].gamma < 0.0  # momentum acts differently by regime
    assert all(lf.gamma == pytest.approx(glob.gamma) for lf in glob.leaves)  # global mode: one gamma
    p = loc.predict_params(X[:4], z[:4], np.zeros(4))
    np.testing.assert_allclose(p["mu_e"], p["mu_leaf"] + p["gamma"] * z[:4])
    assert "γ +" in "\n".join(loc.describe()) or "γ -" in "\n".join(loc.describe())
    # a huge prior weight gives the global gamma back
    stiff = _tree(["regime", "noise"], null_dir, gamma_mode="leaf", gamma_shrink_k=1e9, **kw).fit(
        X, y, z, rng.normal(0, 0.1, n), cluster=date
    )
    assert all(lf.gamma == pytest.approx(stiff.gamma, abs=1e-6) for lf in stiff.leaves)


def test_overlapping_panel_keeps_folds_test_rows_and_market_leg(rsmt_settings, tmp_path):
    pp = synthetic_price_panel(n_tickers=20, n_days=1100, planted=True, seed=6)
    s0 = _new_settings(str(tmp_path))
    s1 = _new_settings(str(tmp_path), overlap_step=5)
    a, b = build_panel(pp, s0), build_panel(pp, s1)
    assert a.overlap_step == 0 and b.overlap_step == 5 and len(b.train) > 3.5 * len(a.train)
    assert a.train["on_grid"].all() and b.train["on_grid"].sum() == len(a.train)
    cols = ["ticker", "date", "y_e", "z", "v", "w", "scale_h", "block"]
    on = b.train[b.train["on_grid"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(on[cols[:4]], a.train[cols[:4]])
    # blocks: one sample date per block on the non-overlapping grid, h/step dates per block with overlap
    assert a.train.groupby("block")["date"].nunique().max() == 1
    assert b.train.groupby("block")["date"].nunique().max() == 4
    # the market leg only counts the non-overlapping rows
    assert b.market_mu_hist == pytest.approx(a.market_mu_hist) and b.market_nu == pytest.approx(a.market_nu)
    # CV: identical folds and test rows, overlapping rows only add training rows (all purged before the test)
    for k in (3, 4):
        fa, fb = cv_splits(a.train, k), cv_splits(b.train, k)
        for x, y in zip(fa, fb):
            assert x.test_start == y.test_start and len(x.test_idx) == len(y.test_idx)
            assert b.train.iloc[y.test_idx]["on_grid"].all()
            assert len(y.train_idx) > len(x.train_idx)
            assert (pd.to_datetime(b.train.iloc[y.train_idx]["t_end"]) < y.test_start).all()
    # the tree clusters on h-session blocks; min_leaf is counted in non-overlapping rows
    tree = RSMTree.from_settings(s1, Market.KR).fit_frame(b.train)
    assert tree.null_used == "bootstrap" and tree.n_clusters == b.train["block"].nunique()
    assert tree.row_multiplicity == 4.0 and tree.min_leaf == 4 * s1.rsmt.markets["KR"].min_leaf
    with pytest.raises(ValueError, match="divide"):
        build_panel(pp, _new_settings(str(tmp_path), overlap_step=3))


def test_overlapping_null_panel_does_not_split_on_noise(tmp_path):
    """Overlapping labels + a common residual factor, no structure: the block-cluster bootstrap holds its size."""
    s = _new_settings(str(tmp_path), overlap_step=5)
    splits = 0
    for seed in range(3):
        tr = build_panel(clustered_null_price_panel(200 + seed), s).train
        tree = RSMTree.from_settings(s, Market.US, max_depth=1).fit_frame(tr)
        assert tree.null_used == "bootstrap" and tree.n_clusters == tr["block"].nunique() < tr["date"].nunique()
        splits += tree.n_splits
    assert splits == 0


def test_per_market_switch_overrides_and_validation(tmp_path):
    from quantcio.quant.panel import overlap_factor, rsmt_option

    s = load_settings(
        data={"cache_dir": str(tmp_path)},
        rsmt=_old_path(
            {"multiple_testing": "simes", "markets": {"US": {"multiple_testing": "pca", "overlap_step": 5}}}
        ),
    )
    assert rsmt_option(s, Market.KR, "multiple_testing") == "simes"
    assert rsmt_option(s, Market.US, "multiple_testing") == "pca"
    assert overlap_factor(s, "KR") == 1.0 and overlap_factor(s, "US") == 4.0
    assert RSMTree.from_settings(s, Market.US).multiple_testing == "pca"
    with pytest.raises(KeyError):
        rsmt_option(s, Market.KR, "alpha")
    with pytest.raises(ValueError):
        load_settings(rsmt={"markets": {"KR": {"gamma_shrink_k": 0}}})
    with pytest.raises(ValueError):
        load_settings(rsmt={"skill_cover80_band": [0.9, 0.8]})


def test_distribution_gate_components():
    m = {"crps_gain_p05": 0.0001, "cover80": 0.79, "exc95": 0.05}
    assert rsmt_mod.distribution_gate(m) == (True, {"crps_ci": True, "cover80": True, "exc95": True})
    assert rsmt_mod.distribution_gate({**m, "crps_gain_p05": -1e-6})[1]["crps_ci"] is False
    assert rsmt_mod.distribution_gate({**m, "cover80": 0.86})[0] is False
    assert rsmt_mod.distribution_gate({**m, "exc95": 0.019})[0] is False
    assert rsmt_mod.distribution_gate({**m, "exc95": 0.09}, exc95_band=(0.0, 0.1))[0] is True
    assert rsmt_mod.distribution_gate({})[0] is False  # missing metrics never pass


def test_date_block_mean_bootstrap_equals_resampling_whole_dates():
    rng = np.random.default_rng(2)
    dates = np.repeat(pd.date_range("2024-01-01", periods=30, freq="7D"), rng.integers(5, 15, 30))
    vals = rng.normal(0.0001, 0.001, dates.size)
    b = rsmt_mod.date_block_mean_bootstrap(vals, dates, 400, seed=3)
    assert b.hat == pytest.approx(vals.mean()) and b.n_dates == 30 and b.draws.size == 400
    # one draw by hand: stack the rows of the drawn dates
    codes, _ = pd.factorize(pd.Series(dates))
    cnt = np.random.default_rng(np.random.SeedSequence([3, 0xC425])).multinomial(30, np.full(30, 1 / 30))
    rows = np.concatenate([np.repeat(vals[codes == c][None, :], k, axis=0).ravel() for c, k in enumerate(cnt) if k])
    assert b.draws[0] == pytest.approx(rows.mean())
    assert b.p05 < b.q05 < b.hat  # small-cluster widening


def test_distribution_skill_metric_serves_both_verdicts(small_model, small_panel, rsmt_settings, tmp_path):
    """skill_metric = distribution: DistributionForecast.skill_ok = the distribution verdict (also under an
    ExcessOverride), direction_skill_ok = RSMT-Q's direction gate (or the override's skill), and the directional ML
    pillar multiplier reads direction_skill_ok only."""
    s = _new_settings(str(tmp_path))
    dist_ok = CVResult(
        metrics={"cover80": 0.8, "exc95": 0.05, "crps_gain_p05": 1e-4, "auc_gain_p05": -0.01},
        skill_ok=True,
        skill_metric="distribution",
        distribution_ok=True,
        direction_ok=False,
        dist_gate={"crps_ci": True, "cover80": True, "exc95": True},
        n_folds_primary=4,
    )
    lays = {
        k: LayoutGate(k, True, True, direction_ok=False, distribution_ok=True, crps_gain=1e-4, crps_gain_p05=5e-5)
        for k in (3, 4, 5)
    }
    apply_robust_gate(dist_ok, lays)
    m = dist_ok.metrics
    assert dist_ok.skill_ok and dist_ok.distribution_ok and not dist_ok.direction_ok
    assert m["gate_folds_pass"] == 3 and m["distribution_gate"] == 1.0 and m["direction_gate"] == 0.0
    assert m["distribution_gate_folds_4"] == 1.0 and m["direction_gate_folds_4"] == 0.0
    assert m["skill_metric_distribution"] == 1.0
    model = MarketModel(small_panel, small_model.tree, dist_ok, s)
    _, fc = model.predict("100002")
    assert fc.skill_ok is True and fc.direction_skill_ok is False
    raw = 0.5 * math.tanh(2 * fc.mu / fc.sigma) + 0.5 * (2 * fc.prob_up - 1)
    assert distribution_ml_pillar(fc) == pytest.approx(raw * 0.35)  # never un-shrunk by the distribution verdict
    assert model.gate_line().startswith("스킬 게이트(분포) 통과(폴드 배치 과반 3/3)")
    assert model.robust_line().startswith("폴드 강건성 게이트(분포) 통과")
    assert any(h.startswith("[정보] 방향(AUC) 게이트 미달") for h in model.health())
    # the stock-selection model's excess mean: 100% of the mean, its skill = the direction verdict
    ov = rsmt_mod.ExcessOverride(mu_excess=0.004, skill_ok=True, model="logit")
    t = "100003"
    fc2 = model.predict_many([t], overrides={t: ov})[t][1]
    assert fc2.blend == {"logit": 1.0} and fc2.mu_excess == pytest.approx(0.004, abs=1e-9)
    assert fc2.skill_ok is True and fc2.direction_skill_ok is True
    assert distribution_ml_pillar(fc2) == pytest.approx(
        max(-1.0, min(1.0, 0.5 * math.tanh(2 * fc2.mu / fc2.sigma) + 0.5 * (2 * fc2.prob_up - 1)))
    )
    # a failing distribution gate is named component by component
    bad = CVResult(
        metrics={"cover80": 0.9, "exc95": 0.05, "crps_gain_p05": -1e-4},
        skill_metric="distribution",
        dist_gate={"crps_ci": False, "cover80": False, "exc95": True},
        n_folds_primary=4,
    )
    apply_robust_gate(bad, {k: LayoutGate(k, False, True) for k in (3, 4, 5)})
    hl = MarketModel(small_panel, small_model.tree, bad, s).health()
    line = next(h for h in hl if h.startswith("[정보] 분포 스킬 게이트 미달("))
    assert "CRPS 개선 날짜 부트스트랩 하한" in line and "cover80 0.900 ∉ [0.75, 0.85]" in line


def test_cv_with_every_switch_runs_the_distribution_gate(tmp_path):
    """V6-style fit end to end: overlapping rows, Simes, standardised target, leaf gamma - CV test rows are the
    non-overlapping ones, both gates are computed, the served forecast is in return units."""
    s = _new_settings(str(tmp_path), overlap_step=5, multiple_testing="simes", target_scale="ewma", gamma_mode="leaf")
    pp = synthetic_price_panel(n_tickers=24, n_days=1300, planted=True, seed=2)
    panel = build_panel(pp, s)
    model = fit_market_model(panel, s)
    cv = model.cv
    assert cv.skill_metric == "distribution" and cv.available
    assert cv.metrics["n_oos"] == len(cv.oos) and set(cv.oos["date"]) <= set(panel.grid_dates)
    assert {"crps_gain", "crps_gain_p05", "exc95", "base_exc95", "direction_gate", "distribution_gate"} <= set(
        cv.metrics
    )
    assert cv.skill_ok == cv.distribution_ok and cv.metrics["gate_folds_pass"] == cv.robust_pass_of("distribution")
    assert model.tree.scaled and model.tree.gamma_mode == "leaf" and model.tree.multiple_testing == "simes"
    t = panel.pred["ticker"].iloc[0]
    _, fc = model.predict(t)
    row = panel.pred.set_index("ticker").loc[t]
    assert fc.offsets["target_scale"] == pytest.approx(row["scale_h"])
    assert fc.sigma_excess == pytest.approx(fc.offsets["sigma_leaf"] * fc.offsets["v_sigma"])
    text = "\n".join(model.diagnostics())
    assert "블록 군집 승수 부트스트랩" in text and "Simes/BH m=9" in text and "모델 스위치: 5거래일 간격" in text
    # a train / test split with the CV machinery (the comparison's final holdout)
    tr = panel.train
    first = pd.Timestamp(sorted(tr.loc[tr["on_grid"], "date"].unique())[-12])
    met, fitted, oos = rsmt_mod.evaluate_split(
        panel, s, (pd.to_datetime(tr["t_end"]) < first).to_numpy(), (tr["date"] >= first).to_numpy(), n_paths=200
    )
    assert met["n_oos"] == len(oos) > 0 and (oos["date"] >= first).all() and "distribution_gate" in met


# ----------------------------------------------------------------------------- sigma equation (docs/RSMT.md section 8)


def _sigma_frame(seed: int = 0, n_dates: int = 120, per: int = 25, theta: float = 0.4) -> pd.DataFrame:
    """Pooled synthetic frame: log sigma = v + theta * h1 (h1 = a sigma term), one partition input of noise."""
    rng = np.random.default_rng(seed)
    n = n_dates * per
    v = rng.normal(0, 0.3, n)
    h1 = rng.normal(size=n)
    z = rng.normal(size=n)
    shock = np.repeat(rng.normal(0, 0.005, n_dates), per)
    y = 0.002 * z + shock + 0.05 * np.exp(v + theta * h1) * rng.standard_normal(n)
    return pd.DataFrame(
        {
            "date": np.repeat(pd.bdate_range("2019-01-01", periods=n_dates, freq="20B"), per),
            "x": rng.normal(size=n),
            "har_66": h1,
            "z": z,
            "v": v,
            "y_e": y,
            "w": 1.0,
        }
    )


def test_fit_offsets_multi_with_one_column_is_fit_offsets():
    rng = np.random.default_rng(1)
    n = 3000
    z, v = rng.normal(size=n), rng.normal(0, 0.4, n)
    leaf = (rng.random(n) > 0.5).astype(np.int64)
    y = 0.01 * leaf + 0.003 * z + 0.04 * np.exp(0.8 * v) * rng.standard_normal(n)
    ref = fit_offsets(y, z, v, np.ones(n), leaf, 0.0, 1.0, 50, 1e-10, (-0.5, 2.0))
    g, e, _, ok = rsmt_mod.fit_offsets_multi(
        y, z, v[:, None], np.ones(n), leaf, 0.0, np.array([1.0]), [(-0.5, 2.0)], tol=1e-10
    )
    assert ok and g == pytest.approx(ref.gamma, abs=1e-7) and e[0] == pytest.approx(ref.eta, abs=1e-6)


def test_sigma_terms_are_fitted_and_served(null_dir):
    df = _sigma_frame()
    tree = RSMTree(
        ["x"], alpha=0.05, min_leaf=300, max_depth=1, cache_dir=null_dir, boot_sims=199, sigma_terms=["har_66"]
    )
    tree.fit_frame(df)
    eq = tree.sigma_equation()
    assert eq["har_66"] == pytest.approx(0.4, abs=0.05) and eq["eta"] == pytest.approx(1.0, abs=0.08)
    p = tree.predict_params(df, df["z"], df["v"])
    np.testing.assert_allclose(
        np.log(p["sigma_e"] / p["sigma_leaf"]), tree.eta * df["v"] + tree.theta[0] * df["har_66"]
    )
    assert any("har_66" in line for line in tree.describe("en"))
    with pytest.raises(RSMTOffsetError):  # the served frame must carry the sigma inputs
        tree.predict_params(df.drop(columns=["har_66"]), df["z"], df["v"])
    with pytest.raises(RSMTOffsetError):
        tree.predict_params(df.assign(har_66=np.nan), df["z"], df["v"])
    # without the term the tree is the original one; the baseline twin never uses sigma terms
    plain = RSMTree(["x"], alpha=0.05, min_leaf=300, max_depth=1, cache_dir=null_dir, boot_sims=199).fit_frame(df)
    assert plain.sigma_equation() == {"eta": pytest.approx(plain.eta)} and plain.theta.size == 0
    twin = RSMTree(["x"], cache_dir=null_dir, **{**rsmt_mod.BASE_TWIN, "sigma_terms": ["har_66"]})
    assert twin.sigma_terms == ()


def test_split_test_clusters_span_boot_block_horizons(null_dir):
    """Overlapping rows: bootstrap clusters = runs of boot_block_horizons h-blocks; min_leaf_dates counts h-blocks."""
    df = _sigma_frame(n_dates=240, per=10)
    df["block"] = np.repeat(np.arange(-240, 0), 10)
    kw = dict(alpha=0.05, min_leaf=100, max_depth=1, cache_dir=null_dir, boot_sims=99, row_multiplicity=4.0)
    t3 = RSMTree(["x"], boot_block_horizons=3, **kw).fit_frame(df)
    assert t3.n_clusters == 240 and np.unique(t3._clusters).size == 80
    t1 = RSMTree(["x"], **kw).fit_frame(df)
    assert np.unique(t1._clusters).size == 240
    # non-overlapping rows (row multiplicity 1): nothing to widen
    t_grid = RSMTree(["x"], boot_block_horizons=3, **{**kw, "row_multiplicity": 1.0}).fit_frame(df)
    assert np.unique(t_grid._clusters).size == 240


def test_sigma_terms_and_history_config_validation(tmp_path):
    from quantcio.quant.panel import sigma_terms_of

    s = load_settings(data={"cache_dir": str(tmp_path)})
    assert sigma_terms_of(s, "KR") == ["har_5", "har_66", "har_252"]
    assert sigma_terms_of(s, "US") == ["cal_s1", "cal_c1", "cal_s4", "cal_c4"]
    assert str(s.rsmt.markets["KR"].history_start) == "1995-09-01" and s.rsmt.boot_block_horizons == 3
    assert RSMTree.from_settings(s, Market.US).sigma_terms == ("cal_s1", "cal_c1", "cal_s4", "cal_c4")
    for bad in (
        {"sigma_terms": ["har_7"]},
        {"markets": {"KR": {"sigma_terms": ["har_5", "har_5"]}}},
        {"boot_block_horizons": 0},
        {"model_max_age_days": 0},
    ):
        with pytest.raises(ValueError):
            load_settings(rsmt=bad)


def test_served_offsets_include_the_sigma_equation(tmp_path):
    """KR: HAR terms in the sigma equation; the served sigma = sigma_leaf x v_sigma with v_sigma carrying them."""
    s = load_settings(data={"cache_dir": str(tmp_path)})
    s.rsmt.overlap_step = 5
    panel = build_panel(synthetic_price_panel(), s)
    model = fit_market_model(panel, s, run_cv=False)
    assert model.tree.sigma_terms == ("har_5", "har_66", "har_252") and model.tree.theta.size == 3
    t = panel.pred["ticker"].iloc[0]
    _, fc = model.predict(t)
    row = panel.pred.set_index("ticker").loc[t]
    assert fc.sigma_excess == pytest.approx(fc.offsets["sigma_leaf"] * fc.offsets["v_sigma"])
    terms = float(np.dot(row[list(model.tree.sigma_terms)].to_numpy(dtype=float), model.tree.theta))
    assert fc.offsets["log_sigma_terms"] == pytest.approx(terms, abs=1e-9)
    assert any("변동성 식" in line for line in model.diagnostics() + model.tree.describe("ko"))
