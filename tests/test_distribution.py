"""ReturnDistribution: one path matrix -> every derived quantity (synthetic, offline)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import stats

from quantcio.models import BarrierOutcome, DistributionForecast
from quantcio.quant.distribution import (
    ReturnDistribution,
    apply_edge,
    daily_nu,
    effective_rho,
    exact_shares,
    kelly_optimal,
    nu_from_excess_kurtosis,
    path_seed,
    sample_crps,
    total_log_sd,
    weighted_excess_kurtosis,
)


def _dist(**kw) -> ReturnDistribution:
    base = dict(
        mu_e=0.004,
        sigma_e=0.07,
        mu_m=0.006,
        sigma_m=0.045,
        nu=8.0,
        nu_m=12.0,
        beta=1.1,
        rho=0.2,
        horizon=20,
        n_paths=20000,
        seed=123,
    )
    base.update(kw)
    return ReturnDistribution(**base)


# ------------------------------------------------------------------------------ barrier


@pytest.mark.parametrize(
    "kw, stop, take",
    [
        ({}, -0.07, 0.12),
        ({"sigma_e": 0.15, "nu": 4.5}, -0.05, 0.05),
        ({"mu_e": -0.08}, -0.10, 0.30),
        ({"n_paths": 997, "seed": 5}, -0.04, 0.06),
        ({"n_paths": 3, "seed": 9}, -0.02, 0.02),
        ({"hold_days": 60}, -0.07, 0.12),
        ({"n_paths": 997, "seed": 5, "hold_days": 45}, -0.04, 0.10),
    ],
)
def test_barrier_probabilities_sum_to_exactly_one(kw, stop, take):
    d = _dist(**kw)
    b = d.barrier(stop, take)
    assert isinstance(b, BarrierOutcome)
    parts = [b.p_take_profit, b.p_stop_loss, b.p_time_exit]
    for perm in itertools.permutations(parts):
        assert perm[0] + perm[1] + perm[2] == 1.0  # exact, in any summation order
    assert sum(parts) == 1.0 and math.fsum(parts) == 1.0
    outcome, _ = d.barrier_trades(stop, take)
    n = d.n_paths
    assert abs(b.p_take_profit - (outcome == 1).sum() / n) < 1e-12  # they ARE counts / n_paths
    assert abs(b.p_stop_loss - (outcome == -1).sum() / n) < 1e-12
    assert abs(b.p_time_exit - (outcome == 0).sum() / n) < 1e-12
    assert all(0.0 <= p <= 1.0 for p in parts)
    assert b.n_paths == n
    assert b.horizon_days == d.hold_days == kw.get("hold_days", 20)


def test_exact_shares_edge_cases():
    rng = np.random.default_rng(0)
    for _ in range(300):
        counts = list(rng.integers(0, 50_000, size=3))
        if sum(counts) == 0:
            continue
        shares = exact_shares(counts)
        for perm in itertools.permutations(shares):
            assert perm[0] + perm[1] + perm[2] == 1.0
        assert all(s >= 0 for s in shares)
        assert all(abs(s - c / sum(counts)) < 1e-12 for s, c in zip(shares, counts))
    assert exact_shares([0, 5, 0]) == [0.0, 1.0, 0.0]
    assert exact_shares([1, 1, 2]) == [0.25, 0.25, 0.5]
    with pytest.raises(ValueError):
        exact_shares([0, 0, 0])


def test_barrier_gap_risk_exit_at_breach_close():
    # fat tails + high vol so a close can jump through the stop
    d = _dist(sigma_e=0.20, nu=4.5, seed=7)
    stop, take = -0.05, 0.10
    outcome, trade = d.barrier_trades(stop, take)
    sl, tp, tm = outcome == -1, outcome == 1, outcome == 0
    assert sl.any() and tp.any() and tm.any()
    assert (trade[sl] <= stop).all()  # breach-day close is at or beyond the stop
    assert trade[sl].min() < stop - 0.01  # gap risk: strictly worse than the stop exists
    assert (trade[tp] >= take).all()
    assert ((trade[tm] > stop) & (trade[tm] < take)).all()
    # time exits are exactly the horizon close (the stored matrix is float32; the barrier evaluates
    # it in float64)
    paths64 = d.r_paths.astype(np.float64)
    np.testing.assert_array_equal(trade[tm], np.expm1(paths64[tm, -1]))
    # first touch: before the exit day the path never touched either barrier
    simple = np.expm1(paths64)
    for i in np.nonzero(sl)[0][:200]:
        day = int(np.argmax(simple[i] <= stop))
        assert (simple[i, :day] > stop).all() and (simple[i, :day] < take).all()
        assert trade[i] == simple[i, day]
    b = d.barrier(stop, take)
    assert b.expected_trade_return == pytest.approx(float(trade.mean()))


def test_barrier_input_validation():
    d = _dist(n_paths=100)
    for stop, take in [(0.05, 0.1), (-0.05, -0.01), (-1.5, 0.1)]:
        with pytest.raises(ValueError):
            d.barrier(stop, take)


# ------------------------------------------------------------------------ determinism


def test_deterministic_under_same_seed():
    a, b = _dist(seed=42), _dist(seed=42)
    np.testing.assert_array_equal(a.r_paths, b.r_paths)
    np.testing.assert_array_equal(a.m_terminal, b.m_terminal)
    assert a.barrier(-0.06, 0.1) == b.barrier(-0.06, 0.1)
    assert a.summary() == b.summary()
    c = _dist(seed=43)
    assert not np.array_equal(a.r_paths, c.r_paths)


def test_path_seed_is_sha1_of_ticker_and_date():
    import hashlib

    s = path_seed("005930", "2026-09-23")
    assert s == int(hashlib.sha1(b"005930|2026-09-23").hexdigest()[:16], 16)
    assert s == path_seed("005930", "2026-09-23")
    assert s != path_seed("005930", "2026-09-22") and s != path_seed("000660", "2026-09-23")


def test_paths_are_read_only():
    d = _dist(n_paths=50)
    with pytest.raises(ValueError):
        d.r_paths[0, 0] = 1.0


# ------------------------------------------------------------------------ tails / moments


@pytest.mark.parametrize("nu, h", [(24.0, 4), (16.0, 2)])
def test_terminal_kurtosis_matches_t_nu(nu, h):
    d = ReturnDistribution(0.0, 0.05, 0.0, 0.0, nu, 30.0, 0.0, 0.0, h, 400_000, seed=11)
    nu_d = daily_nu(nu, h)
    assert nu_d == pytest.approx(4 + (nu - 4) / h)
    target = 6.0 / (nu - 4.0)  # excess kurtosis of t_nu
    assert (6.0 / (nu_d - 4.0)) / h == pytest.approx(target)
    k = stats.kurtosis(d.r_terminal, fisher=True)
    assert k == pytest.approx(target, abs=0.08)
    assert d.r_terminal.std() == pytest.approx(0.05, rel=0.01)  # unit-variance scaling of the t's
    assert d.r_terminal.mean() == pytest.approx(0.0, abs=0.0005)


def test_daily_nu_floor():
    assert daily_nu(4.5, 20) == 4.05
    assert daily_nu(30.0, 20) == pytest.approx(5.3)
    assert daily_nu(8.0, 1) == 8.0


def test_analytic_moments_match_simulation():
    d = _dist(n_paths=200_000, seed=3, rho=0.5, nu=6.0, nu_m=6.0)
    assert d.mean_log == pytest.approx(1.1 * 0.006 + 0.004)
    assert d.r_terminal.mean() == pytest.approx(d.mean_log, abs=0.001)
    assert d.r_terminal.std() == pytest.approx(d.sd_log, rel=0.01)
    e_t = d.r_terminal - d.beta * d.m_terminal
    corr = np.corrcoef(e_t, d.m_terminal)[0, 1]
    assert corr == pytest.approx(effective_rho(0.5, d.nu_d, d.nu_d_m), abs=0.01)
    assert abs(effective_rho(0.5, d.nu_d, d.nu_d_m)) < 0.5  # separate chi-square mixers dilute rho


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"rho": -0.4, "nu": 4.5, "nu_m": 30.0},
        {"beta": 0.0},
        {"horizon": 1, "sigma_m": 0.0},
        {"beta": 2.5, "rho": 0.9},
    ],
)
def test_total_log_sd_is_the_distribution_sd(kw):
    """The serving plausibility clamp computes sigma_h BEFORE building the distribution; it must be
    exactly the sd the path matrix has (`sd_log`)."""
    d = _dist(n_paths=2, **kw)
    assert total_log_sd(d.beta, d.sigma_e, d.sigma_m, d.rho, d.nu, d.nu_m, d.horizon) == d.sd_log
    big = _dist(n_paths=100_000, seed=11, **kw)
    assert big.r_terminal.std() == pytest.approx(big.sd_log, rel=0.02)


# ---------------------------------------------------------------------------- edges


def test_direction_edge_only_moves_direction_probabilities():
    plain = _dist(seed=21)
    edged = _dist(seed=21, direction_edge=0.3, relative_edge=-0.2)
    assert edged.prob_up_raw == plain.prob_up_raw
    assert edged.prob_up == pytest.approx(apply_edge(plain.prob_up, 0.3, plain.n_paths))
    assert edged.prob_up > plain.prob_up
    assert edged.prob_outperform < plain.prob_outperform
    # everything else is edge-free and identical
    assert edged.expected_return == plain.expected_return
    np.testing.assert_array_equal(edged.quantile([0.05, 0.1, 0.5, 0.9]), plain.quantile([0.05, 0.1, 0.5, 0.9]))
    assert edged.var95 == plain.var95 and edged.cvar95 == plain.cvar95
    assert edged.barrier(-0.07, 0.12) == plain.barrier(-0.07, 0.12)
    assert edged.kelly_optimal() == plain.kelly_optimal()
    fe, fp = edged.summary(), plain.summary()
    for key in (
        "mu",
        "sigma",
        "expected_return",
        "q10",
        "q50",
        "q90",
        "var95",
        "cvar95",
        "mu_excess",
    ):
        assert getattr(fe, key) == getattr(fp, key)
    assert fe.prob_up != fp.prob_up and fe.direction_edge == 0.3 and fe.relative_edge == -0.2


def test_apply_edge_handles_monte_carlo_certainty():
    assert apply_edge(1.0, 0.0, 1000) == 1.0
    assert 0.99 < apply_edge(1.0, 0.1, 1000) < 1.0
    assert 0.0 < apply_edge(0.0, -0.1, 1000) < 0.01


# ---------------------------------------------------------------------------- kelly


def test_kelly_bounds_and_negative_edge():
    neg = _dist(mu_e=-0.06, mu_m=-0.01)
    assert neg.expected_return < 0
    assert neg.kelly_optimal() == 0.0
    assert neg.barrier(-0.05, 0.08).kelly_optimal == 0.0
    mid = ReturnDistribution(0.0, 0.2, 0.0, 0.0, 30.0, 30.0, 0.0, 0.0, 20, 20000, seed=1)
    k = mid.kelly_optimal()
    assert 0.0 < k < 1.0
    assert k == pytest.approx(0.5, abs=0.12)  # ~ E[R]/Var[R] for a lognormal-ish return
    assert 0.0 <= _dist().barrier(-0.07, 0.12).kelly_optimal <= 1.0
    grid_ok = round(k / 0.005) * 0.005
    assert k == pytest.approx(grid_ok)


def test_kelly_grid_search_is_argmax():
    rng = np.random.default_rng(0)
    r = rng.normal(0.02, 0.15, 5000)
    f = kelly_optimal(r)
    grid = np.arange(0, 1.0001, 0.005)
    vals = [np.mean(np.log1p(g * r)) for g in grid]
    assert f == pytest.approx(grid[int(np.argmax(vals))])
    with pytest.raises(ValueError):
        kelly_optimal(np.array([]))


# ------------------------------------------------------------------------- misc API


def test_shift_mu_same_seed_shifts_every_path():
    d = _dist(seed=8)
    s = d.shift_mu(0.02)
    assert s.seed == d.seed and s.mu_e == pytest.approx(d.mu_e + 0.02)
    np.testing.assert_allclose(s.r_terminal - d.r_terminal, 0.02, atol=1e-12)
    assert s.mean_log == pytest.approx(d.mean_log + 0.02)
    assert s.prob_up_raw >= d.prob_up_raw


def test_summary_is_consistent_forecast():
    d = _dist(direction_edge=0.1)
    fc = d.summary(
        leaf_id=3,
        leaf_path=["a ≤ 1"],
        leaf_n=500,
        offsets={"gamma": 0.01},
        cv_metrics={"auc": 0.55},
        skill_ok=True,
        blend={"logit_w": 1.0},
    )
    assert isinstance(fc, DistributionForecast)
    assert fc.mu == pytest.approx(d.beta * d.mu_m + d.mu_e)
    assert fc.q10 < fc.q50 < fc.q90
    assert fc.cvar95 <= fc.var95 <= fc.q10
    assert fc.expected_return == pytest.approx(float(np.mean(np.expm1(d.r_terminal))))
    assert fc.var95 == pytest.approx(float(np.quantile(np.expm1(d.r_terminal), 0.05)))
    assert fc.prob_up == pytest.approx(d.prob_up) and fc.degraded is False
    assert fc.leaf_path == ["a ≤ 1"] and fc.blend == {"logit_w": 1.0}
    deg = d.summary(degraded_reasons=["오프셋 입력 누락(z|v): xs_mom_z → 0으로 설정"])
    assert deg.degraded is True and deg.degraded_reasons
    nan_metric = d.summary(cv_metrics={"auc": 0.55, "ic": float("nan")})
    assert nan_metric.cv_metrics == {"auc": 0.55}
    assert DistributionForecast.model_validate_json(fc.model_dump_json()) == fc  # JSON round-trip


@pytest.mark.parametrize(
    "bad",
    [
        {"sigma_e": float("nan")},
        {"sigma_e": 0.0},
        {"rho": 1.5},
        {"nu": 2.0},
        {"horizon": 0},
        {"n_paths": 1},
        {"mu_e": float("inf")},
        {"beta": None},
    ],
)
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        _dist(n_paths=100, **bad) if "n_paths" not in bad else _dist(**bad)


def test_crps_and_kurtosis_helpers():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200_000)
    # CRPS of N(0,1) at y=0 is (sqrt(2) - 1)/sqrt(pi)
    assert sample_crps(x, 0.0) == pytest.approx((math.sqrt(2) - 1) / math.sqrt(math.pi), abs=0.005)
    assert weighted_excess_kurtosis(x) == pytest.approx(0.0, abs=0.05)
    assert nu_from_excess_kurtosis(1.0, 4.5, 30) == pytest.approx(10.0)
    assert nu_from_excess_kurtosis(-0.2, 4.5, 30) == 30.0
    assert nu_from_excess_kurtosis(100.0, 4.5, 30) == 4.5


def test_memory_and_speed_serving_size():
    import time

    t0 = time.perf_counter()
    d = _dist(n_paths=20000)
    d.barrier(-0.07, 0.12)
    d.summary()
    assert time.perf_counter() - t0 < 2.0
    assert d.r_paths.nbytes <= 20000 * 20 * 8


# ------------------------------------------------------------- holding horizon (hold_days)


def test_hold_days_keeps_every_horizon_quantity_bit_identical():
    """One matrix of max(h, hold_days) steps: the first h days come from the seed's own stream, so
    every h-horizon quantity is identical for any hold_days (and to the plain h-day simulation)."""
    base = _dist(seed=31)
    for hold in (20, 60, 5):
        d = _dist(seed=31, hold_days=hold)
        assert d.hold_days == hold and d.n_steps == max(20, hold)
        assert d.r_paths.shape == (20000, max(20, hold))
        np.testing.assert_array_equal(d.r_terminal, base.r_terminal)
        np.testing.assert_array_equal(d.m_terminal, base.m_terminal)
        np.testing.assert_array_equal(d.r_paths[:, :20], base.r_paths)
        assert d.summary() == base.summary()
        assert d.summary().horizon_days == 20
        assert d.prob_up == base.prob_up and d.var95 == base.var95
        assert d.kelly_optimal() == base.kelly_optimal()  # buy-and-hold to h
    # day-h column of the stored (float32) matrix is the float64 terminal, rounded
    d = _dist(seed=31, hold_days=60)
    np.testing.assert_allclose(d.r_paths[:, 19], d.r_terminal, atol=1e-6)
    # deterministic, read-only, shift_mu keeps hold_days and the draws
    np.testing.assert_array_equal(d.r_paths, _dist(seed=31, hold_days=60).r_paths)
    with pytest.raises(ValueError):
        d.r_paths[0, 30] = 1.0
    s = d.shift_mu(0.01)
    assert s.hold_days == 60 and s.r_paths.shape == d.r_paths.shape


def test_hold_days_extension_uses_the_same_daily_parameters():
    # 60 daily steps of mean mu/h and sd sigma/sqrt(h): day-60 mean 3*mu, sd sqrt(3)*sigma
    d = ReturnDistribution(0.01, 0.06, 0.004, 0.04, 30.0, 30.0, 0.8, 0.3, 20, 200_000, seed=5, hold_days=60)
    x = d.r_paths[:, -1].astype(np.float64)
    assert x.mean() == pytest.approx(3 * d.mean_log, abs=0.0015)
    assert x.std() == pytest.approx(math.sqrt(3) * d.sd_log, rel=0.01)
    # increments after day h are independent of the first h days
    inc = x - d.r_terminal
    assert abs(np.corrcoef(inc, d.r_terminal)[0, 1]) < 0.01
    assert inc.std() == pytest.approx(math.sqrt(2) * d.sd_log, rel=0.01)


def test_barrier_runs_over_the_holding_horizon():
    """Regression: the take-profit comes from a target at fusion.target_horizon_days (60) while the
    distribution is for h=20; a 20-day barrier made P(TP first) structurally low (most paths
    time-exited before the target was reachable)."""
    d = _dist(mu_e=0.02, seed=17, hold_days=60)
    take = math.expm1(3 * d.mean_log)  # the 60-day expected move
    stop = -0.08
    hold = d.barrier(stop, take)
    short = d.barrier(stop, take, days=20)
    assert hold.horizon_days == 60 and short.horizon_days == 20
    for b in (hold, short):
        assert b.p_take_profit + b.p_stop_loss + b.p_time_exit == 1.0
    assert hold.p_take_profit > short.p_take_profit + 0.10
    assert hold.p_time_exit < short.p_time_exit - 0.20
    # the h-day barrier on a hold_days=60 matrix == the barrier of the plain h-day distribution
    plain = _dist(mu_e=0.02, seed=17)
    assert short == plain.barrier(stop, take)
    assert plain.barrier(stop, take).horizon_days == 20
    # time exits of the holding barrier are the day-60 close
    outcome, trade = d.barrier_trades(stop, take)
    tm = outcome == 0
    np.testing.assert_array_equal(trade[tm], np.expm1(d.r_paths[tm, 59].astype(np.float64)))
    with pytest.raises(ValueError):
        d.barrier(stop, take, days=61)
    with pytest.raises(ValueError):
        d.barrier(stop, take, days=0)
    with pytest.raises(ValueError):
        _dist(n_paths=100, hold_days=0)


def test_hold_days_memory_and_speed_serving_size():
    import time
    import tracemalloc

    tracemalloc.start()
    try:
        t0 = time.perf_counter()
        d = _dist(n_paths=20000, hold_days=60)
        d.barrier(-0.07, 0.12)
        d.summary()
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert elapsed < 2.0
    assert d.r_paths.dtype == np.float32
    assert d.r_paths.nbytes == 20000 * 60 * 4  # 4.8 MB retained path matrix
    assert current < 12e6  # retained: matrix + float64 day-h terminals (+ barrier temporaries freed)
    assert peak < 40e6  # transient working set (chunked extension, float64 barrier view)
