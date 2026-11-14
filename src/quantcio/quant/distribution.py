"""RSMT-Q return distribution: ONE simulated path matrix -> every derived quantity.

`ReturnDistribution` simulates `n_paths` daily paths of the stock's cumulative log return,
``r_t = beta * m_t + e_t``, where ``m`` is the benchmark and ``e`` the beta-adjusted excess return.
P(up), P(beat benchmark), E[return], quantiles, VaR/CVaR, the stop-loss / take-profit barrier
outcome and the Kelly fraction are all read off that single matrix, so they can never contradict
each other (the RSMT analogue: win/draw/loss, O/U and exact scores from one score matrix).

Two horizons, one matrix:
- ``horizon`` (h, the model horizon = settings.horizon_days): every forecast quantity (prob_up,
  prob_outperform, expected_return, quantiles, VaR/CVaR, CRPS, buy-and-hold Kelly, `summary`) reads
  the path matrix at day h.
- ``hold_days`` (the trade's holding horizon, normally fusion.target_horizon_days = 60): the matrix
  has ``max(h, hold_days)`` daily steps with the SAME daily parameters, and `barrier` runs over
  ``hold_days`` by default. The take-profit is set from a 60-day target, so a 20-day barrier would
  make P(TP first) structurally low (most paths would "time-exit" before the target is reachable).
  The first h steps are drawn from the seed's own stream and the extension from an independent
  spawned stream, so every h-horizon quantity is bit-identical for any hold_days.

Simulation (all mu / sigma inputs are HORIZON-h log-return moments):
- daily mean = mu / h, daily sd = sigma / sqrt(h) (also on the days after h);
- Student-t innovations with ``nu_d = max(4.05, 4 + (nu - 4) / h)`` so that the h-day sum has the
  excess kurtosis of a t_nu (6 / (nu - 4)); scaled to unit variance by sqrt((nu_d - 2) / nu_d);
- the residual dependence ``rho`` is applied to the underlying normals BEFORE the chi-square scale
  mixing, with separate chi-square draws for e and m;
- the seed is deterministic (see `path_seed`: sha1(ticker|as_of)).

Memory: the stored path matrix is float32 (20000 paths x 60 days = 4.8 MB); the day-h terminal
returns of the stock and the benchmark are kept in float64 (every h-horizon quantity uses them).
The extension days are simulated in path chunks, so the transient working set stays bounded.

The direction edges (delta, delta_rel) are logit shifts applied ONLY to `prob_up` and
`prob_outperform`; the mean, quantiles, VaR, barrier and Kelly never include them.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Mapping, Sequence
from functools import cached_property

import numpy as np

from quantcio.models import BarrierOutcome, DistributionForecast

log = logging.getLogger(__name__)

NU_D_FLOOR = 4.05  # daily t dof floor (4th moment must exist)
KELLY_STEP = 0.005  # Kelly grid step on f in [0, 1]
PATH_DTYPE = np.float32  # stored path matrix (terminal day-h vectors stay float64)
EXT_CHUNK_FLOATS = 1_000_000  # bounds the per-chunk working set of the extension days (~8 MB)
_EXT_STREAM = 1  # spawn key of the independent RNG stream used for the days after the horizon
_SHARE_UNITS = 1 << 40  # barrier probabilities are exact multiples of 2^-40 (sum to 1.0 in any order)


# ----------------------------------------------------------------------------- helpers


def path_seed(ticker: str, as_of: object) -> int:
    """Deterministic 64-bit simulation seed from sha1(ticker|as_of)."""
    digest = hashlib.sha1(f"{ticker}|{as_of}".encode()).hexdigest()
    return int(digest[:16], 16)


def daily_nu(nu: float, horizon: int) -> float:
    """Daily t dof whose h-day sum has the excess kurtosis of a t_nu: max(4.05, 4 + (nu - 4) / h)."""
    return max(NU_D_FLOOR, 4.0 + (float(nu) - 4.0) / int(horizon))


def _inv_sqrt_chi_mean(nu: float) -> float:
    """E[(chi2_nu / nu)^(-1/2)] = sqrt(nu/2) * Gamma((nu-1)/2) / Gamma(nu/2)."""
    return math.sqrt(nu / 2.0) * math.exp(math.lgamma((nu - 1.0) / 2.0) - math.lgamma(nu / 2.0))


def effective_rho(rho: float, nu_d_e: float, nu_d_m: float) -> float:
    """Correlation of the unit-variance t innovations when rho is applied to the underlying normals
    and the chi-square mixers are independent: rho * E[s_e] * E[s_m] (s = unit-variance t scale)."""
    scale_e = math.sqrt((nu_d_e - 2.0) / nu_d_e) * _inv_sqrt_chi_mean(nu_d_e)
    scale_m = math.sqrt((nu_d_m - 2.0) / nu_d_m) * _inv_sqrt_chi_mean(nu_d_m)
    return float(rho) * scale_e * scale_m


def total_log_sd(
    beta: float,
    sigma_e: float,
    sigma_m: float,
    rho: float,
    nu: float,
    nu_m: float,
    horizon: int,
) -> float:
    """Exact sd of the simulated horizon log return r_h = beta*m_h + e_h (same formula as
    `ReturnDistribution.sd_log`, usable before a distribution is built): the correlation is the
    effective correlation of the unit-variance daily t innovations (`effective_rho`)."""
    rho_eff = effective_rho(rho, daily_nu(nu, horizon), daily_nu(nu_m, horizon))
    var = (beta * sigma_m) ** 2 + sigma_e**2 + 2.0 * beta * rho_eff * sigma_e * sigma_m
    return math.sqrt(max(var, 0.0))


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def apply_edge(p: float, delta: float, n_paths: int) -> float:
    """sigmoid(logit(p) + delta); p is clipped to [0.5/n, 1 - 0.5/n] first (a Monte Carlo 0/1 is not
    a certainty). delta == 0 returns p unchanged."""
    if delta == 0.0:
        return float(p)
    eps = 0.5 / max(int(n_paths), 1)
    pc = min(max(float(p), eps), 1.0 - eps)
    return sigmoid(logit(pc) + float(delta))


def exact_shares(counts: Sequence[int]) -> list[float]:
    """Turn integer counts into probabilities that sum to exactly 1.0 in ANY summation order.

    Each share is a multiple of 2^-40 (largest-remainder rounding of count/total), so every partial
    sum is exactly representable. The deviation from count/total is below 1e-12; when the total is a
    power of two (<= 2^40) the shares equal count/total exactly.
    """
    counts = [int(c) for c in counts]
    if any(c < 0 for c in counts):
        raise ValueError(f"negative count in {counts}")
    total = sum(counts)
    if total <= 0:
        raise ValueError("exact_shares needs a positive total")
    base = [c * _SHARE_UNITS // total for c in counts]
    rem = [c * _SHARE_UNITS % total for c in counts]
    short = _SHARE_UNITS - sum(base)
    for i in sorted(range(len(counts)), key=lambda j: -rem[j])[:short]:
        base[i] += 1
    return [b / _SHARE_UNITS for b in base]


def kelly_optimal(returns: np.ndarray, step: float = KELLY_STEP, chunk: int = 32) -> float:
    """argmax_f mean(log(1 + f*R)) over f in {0, step, ..., 1} (first maximiser; 0 for a negative edge)."""
    r = np.asarray(returns, dtype="float64").ravel()
    if r.size == 0 or not np.isfinite(r).all():
        raise ValueError("kelly_optimal needs a non-empty array of finite returns")
    grid = np.round(np.arange(0.0, 1.0 + step / 2.0, step), 10)
    values = np.empty(grid.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        for start in range(0, grid.size, chunk):
            fs = grid[start : start + chunk]
            values[start : start + fs.size] = np.log1p(np.multiply.outer(fs, r)).mean(axis=1)
    values = np.where(np.isnan(values), -np.inf, values)
    return float(grid[int(np.argmax(values))])


def weighted_excess_kurtosis(x: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted excess kurtosis m4 / m2^2 - 3 (weighted mean removed). NaN when undefined."""
    x = np.asarray(x, dtype="float64")
    w = np.ones_like(x) if w is None else np.asarray(w, dtype="float64")
    sw = w.sum()
    if x.size < 4 or sw <= 0:
        return float("nan")
    mu = (w * x).sum() / sw
    d = x - mu
    m2 = (w * d * d).sum() / sw
    if m2 <= 0:
        return float("nan")
    m4 = (w * d**4).sum() / sw
    return float(m4 / (m2 * m2) - 3.0)


def nu_from_excess_kurtosis(kappa: float, nu_min: float, nu_max: float) -> float:
    """Moment-matched t dof: nu = 4 + 6/kappa when kappa > 0, else nu_max; clipped to [nu_min, nu_max]."""
    if not math.isfinite(kappa) or kappa <= 0:
        return float(nu_max)
    return float(min(max(4.0 + 6.0 / kappa, nu_min), nu_max))


def sample_crps(draws: np.ndarray, y: float) -> float:
    """Sample CRPS = E|X - y| - 0.5 E|X - X'| (PWM form on the sorted draws)."""
    x = np.sort(np.asarray(draws, dtype="float64"))
    n = x.size
    e_xy = np.abs(x - y).mean()
    coef = 2.0 * np.arange(1, n + 1) - n - 1.0
    e_xx = 2.0 * (coef * x).sum() / (n * n)
    return float(e_xy - 0.5 * e_xx)


# ------------------------------------------------------------------------ distribution


class ReturnDistribution:
    """One simulated path matrix of daily log returns; every derived quantity reads from it.

    Args:
        mu_e, sigma_e: horizon mean / sd of the beta-adjusted excess log return e.
        mu_m, sigma_m: horizon mean / sd of the benchmark log return m.
        nu, nu_m: horizon Student-t tail parameters of e and m.
        beta: loading in r = beta*m + e.
        rho: correlation of the underlying normals of e and m.
        horizon: the model horizon h (daily steps the moments above refer to); every forecast
            quantity reads the matrix at day h.
        n_paths: number of simulated paths.
        seed: RNG seed (use `path_seed(ticker, as_of)` when serving).
        direction_edge: logit shift applied ONLY to prob_up.
        relative_edge: logit shift applied ONLY to prob_outperform.
        hold_days: holding horizon of the trade (default: h). The matrix has max(h, hold_days)
            daily steps with the same daily parameters; `barrier` runs over hold_days by default.
    """

    def __init__(
        self,
        mu_e: float,
        sigma_e: float,
        mu_m: float,
        sigma_m: float,
        nu: float,
        nu_m: float,
        beta: float,
        rho: float,
        horizon: int,
        n_paths: int = 20000,
        seed: int = 0,
        direction_edge: float = 0.0,
        relative_edge: float = 0.0,
        hold_days: int | None = None,
    ):
        named = {
            "mu_e": mu_e,
            "sigma_e": sigma_e,
            "mu_m": mu_m,
            "sigma_m": sigma_m,
            "nu": nu,
            "nu_m": nu_m,
            "beta": beta,
            "rho": rho,
            "direction_edge": direction_edge,
            "relative_edge": relative_edge,
        }
        for name, value in named.items():
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"ReturnDistribution: {name} must be finite, got {value!r}")
        if int(horizon) < 1:
            raise ValueError(f"ReturnDistribution: horizon must be >= 1, got {horizon}")
        if hold_days is not None and int(hold_days) < 1:
            raise ValueError(f"ReturnDistribution: hold_days must be >= 1, got {hold_days}")
        if int(n_paths) < 2:
            raise ValueError(f"ReturnDistribution: n_paths must be >= 2, got {n_paths}")
        if sigma_e <= 0 or sigma_m < 0:
            raise ValueError(f"ReturnDistribution: need sigma_e > 0 and sigma_m >= 0 (got {sigma_e}, {sigma_m})")
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"ReturnDistribution: rho must be in [-1, 1], got {rho}")
        if nu <= 2 or nu_m <= 2:
            raise ValueError(f"ReturnDistribution: nu and nu_m must be > 2 (got {nu}, {nu_m})")

        self.mu_e = float(mu_e)
        self.sigma_e = float(sigma_e)
        self.mu_m = float(mu_m)
        self.sigma_m = float(sigma_m)
        self.nu = float(nu)
        self.nu_m = float(nu_m)
        self.beta = float(beta)
        self.rho = float(rho)
        self.horizon = int(horizon)
        self.hold_days = self.horizon if hold_days is None else int(hold_days)
        self.n_steps = max(self.horizon, self.hold_days)
        self.n_paths = int(n_paths)
        self.seed = int(seed)
        self.direction_edge = float(direction_edge)
        self.relative_edge = float(relative_edge)
        self.nu_d = daily_nu(self.nu, self.horizon)
        self.nu_d_m = daily_nu(self.nu_m, self.horizon)
        self._r, self._r_terminal, self._m_terminal = self._simulate()

    # ------------------------------------------------------------------ simulation

    def _cum_block(self, rng: np.random.Generator, n: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
        """(n, steps) cumulative stock and benchmark log returns from 0, float64, in place.

        Draw order per block: normals (2, n, steps), chi2 for e, chi2 for m - the horizon block is
        bit-identical to the original single-horizon simulation.
        """
        h = self.horizon
        normals = rng.standard_normal((2, n, steps))
        z_m, t_e = normals[0], normals[1]
        # z_e = rho*z_m + sqrt(1-rho^2)*z_2 (in place into normals[1])
        t_e *= math.sqrt(max(0.0, 1.0 - self.rho * self.rho))
        t_e += self.rho * z_m
        # unit-variance t: z / sqrt(chi/nu_d) * sqrt((nu_d-2)/nu_d) == z * sqrt((nu_d-2)/chi)
        chi = rng.chisquare(self.nu_d, size=(n, steps))
        np.divide(self.nu_d - 2.0, chi, out=chi)
        np.sqrt(chi, out=chi)
        t_e *= chi
        chi = rng.chisquare(self.nu_d_m, size=(n, steps))
        np.divide(self.nu_d_m - 2.0, chi, out=chi)
        np.sqrt(chi, out=chi)
        t_m = z_m
        t_m *= chi
        del chi
        sq = math.sqrt(h)
        t_m *= self.sigma_m / sq
        t_m += self.mu_m / h
        np.cumsum(t_m, axis=1, out=t_m)  # m_cum
        t_e *= self.sigma_e / sq
        t_e += self.mu_e / h
        np.cumsum(t_e, axis=1, out=t_e)  # e_cum
        t_e += self.beta * t_m  # r_cum = e_cum + beta * m_cum
        return t_e, t_m

    def _simulate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, h, steps = self.n_paths, self.horizon, self.n_steps
        paths = np.empty((n, steps), dtype=PATH_DTYPE)
        # horizon block: the seed's own stream (identical draws for any hold_days)
        r_cum, m_cum = self._cum_block(np.random.default_rng(self.seed), n, h)
        paths[:, :h] = r_cum
        r_terminal = r_cum[:, -1].copy()
        m_terminal = m_cum[:, -1].copy()
        del r_cum, m_cum
        ext = steps - h
        if ext > 0:
            # days after the horizon: same daily parameters, an independent spawned stream, drawn
            # in path chunks so the transient working set stays bounded
            rng_ext = np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(_EXT_STREAM,)))
            chunk = max(1, EXT_CHUNK_FLOATS // (2 * ext))
            for start in range(0, n, chunk):
                stop = min(n, start + chunk)
                r_ext, _ = self._cum_block(rng_ext, stop - start, ext)
                r_ext += r_terminal[start:stop, None]
                paths[start:stop, h:] = r_ext
                del r_ext
        for arr in (paths, r_terminal, m_terminal):
            arr.setflags(write=False)
        return paths, r_terminal, m_terminal

    # ------------------------------------------------------------------ raw matrix

    @property
    def r_paths(self) -> np.ndarray:
        """(n_paths, max(horizon, hold_days)) cumulative stock log returns on each daily close
        (float32, read-only). Column h-1 is the model horizon, column hold_days-1 the holding one."""
        return self._r

    @property
    def r_terminal(self) -> np.ndarray:
        """Stock log return at the model horizon h (float64; the day-h column of the matrix)."""
        return self._r_terminal

    @property
    def m_terminal(self) -> np.ndarray:
        """Benchmark log return at the model horizon h (float64)."""
        return self._m_terminal

    @cached_property
    def simple_terminal(self) -> np.ndarray:
        """exp(r_T) - 1 per path."""
        out = np.expm1(self.r_terminal)
        out.setflags(write=False)
        return out

    # ------------------------------------------------------------------ analytic moments

    @property
    def mean_log(self) -> float:
        """Exact mean of the simulated horizon log return: beta*mu_m + mu_e."""
        return self.beta * self.mu_m + self.mu_e

    @property
    def sd_log(self) -> float:
        """Exact sd of the simulated horizon log return (uses the effective t-innovation correlation)."""
        return total_log_sd(self.beta, self.sigma_e, self.sigma_m, self.rho, self.nu, self.nu_m, self.horizon)

    # ------------------------------------------------------------------ probabilities

    @cached_property
    def prob_up_raw(self) -> float:
        return float(np.mean(self.r_terminal > 0.0))

    @property
    def prob_up(self) -> float:
        """P(r_T > 0) with the direction edge delta (logit shift)."""
        return apply_edge(self.prob_up_raw, self.direction_edge, self.n_paths)

    @cached_property
    def prob_outperform_raw(self) -> float:
        return float(np.mean(self.r_terminal > self.m_terminal))

    @property
    def prob_outperform(self) -> float:
        """P(r_T > m_T) with the relative edge delta_rel (logit shift)."""
        return apply_edge(self.prob_outperform_raw, self.relative_edge, self.n_paths)

    # ------------------------------------------------------------------ edge-free quantities

    @cached_property
    def expected_return(self) -> float:
        """E[exp(r_T) - 1] (simple, decimal). No edge applied."""
        return float(self.simple_terminal.mean())

    def quantile(self, q: float | Sequence[float]) -> float | np.ndarray:
        """Quantile(s) of the simple horizon return exp(r_T) - 1."""
        out = np.quantile(self.simple_terminal, q)
        return float(out) if np.ndim(out) == 0 else out

    def quantile_log(self, q: float | Sequence[float]) -> float | np.ndarray:
        """Quantile(s) of the horizon log return r_T."""
        out = np.quantile(self.r_terminal, q)
        return float(out) if np.ndim(out) == 0 else out

    @cached_property
    def var95(self) -> float:
        """5% quantile of the simple horizon return (negative decimal for a loss)."""
        return float(np.quantile(self.simple_terminal, 0.05))

    @cached_property
    def cvar95(self) -> float:
        """Mean simple return of the paths at or below var95."""
        tail = self.simple_terminal[self.simple_terminal <= self.var95]
        return float(tail.mean()) if tail.size else self.var95

    def crps(self, y_log: float) -> float:
        """Sample CRPS of the terminal log-return draws against a realised log return."""
        return sample_crps(self.r_terminal, float(y_log))

    def kelly_optimal(self) -> float:
        """Kelly fraction of a buy-and-hold position to the horizon (grid search, before any cap)."""
        return kelly_optimal(self.simple_terminal)

    # ------------------------------------------------------------------ barrier

    def _barrier_days(self, days: int | None) -> int:
        d = self.hold_days if days is None else int(days)
        if not 1 <= d <= self.n_steps:
            raise ValueError(f"barrier days must be in 1..{self.n_steps} (simulated steps), got {days}")
        return d

    def barrier_trades(
        self, stop_pct: float, take_pct: float, days: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-path managed-trade outcome on DAILY CLOSES (종가 기준 손절) over `days` sessions
        (default: hold_days).

        Returns (outcome, trade_return): outcome is +1 (take-profit touched first), -1 (stop-loss
        touched first) or 0 (time exit at the close of day `days`); trade_return is the simple
        return (float64) at the exit close. The exit is the close on the breach day, so gap risk is
        kept (a stop exit can be worse than stop_pct).
        """
        stop_pct, take_pct = float(stop_pct), float(take_pct)
        if not (stop_pct < 0.0 < take_pct) or stop_pct <= -1.0:
            raise ValueError(f"barrier needs -1 < stop_pct < 0 < take_pct (got {stop_pct}, {take_pct})")
        d = self._barrier_days(days)
        # (n, d) simple return on each close, evaluated in float64
        simple = np.expm1(self._r[:, :d], dtype=np.float64)
        hit_sl = simple <= stop_pct
        hit_tp = simple >= take_pct
        first_sl = np.where(hit_sl.any(axis=1), hit_sl.argmax(axis=1), d)
        first_tp = np.where(hit_tp.any(axis=1), hit_tp.argmax(axis=1), d)
        tp_first = first_tp < first_sl  # the two can't coincide (stop < 0 < take)
        sl_first = first_sl < first_tp
        exit_day = np.where(tp_first, first_tp, np.where(sl_first, first_sl, d - 1))
        trade = simple[np.arange(self.n_paths), exit_day]
        outcome = np.where(tp_first, 1, np.where(sl_first, -1, 0)).astype(np.int8)
        return outcome, trade

    def barrier(self, stop_pct: float, take_pct: float, days: int | None = None) -> BarrierOutcome:
        """Stop-loss / take-profit first-touch outcome on DAILY CLOSES (종가 기준 손절) over the
        holding horizon (`days`, default hold_days; recorded in BarrierOutcome.horizon_days).

        The exit return is the close on the breach day, so gap risk is kept (a stop exit can be worse
        than `stop_pct`). Paths touching neither barrier exit at the close of the last holding day.
        The three probabilities are path counts turned into exact shares that sum to exactly 1 (see
        `exact_shares`); kelly_optimal is the grid-search Kelly of the managed-trade returns.
        """
        d = self._barrier_days(days)
        outcome, trade = self.barrier_trades(stop_pct, take_pct, d)
        n_tp, n_sl = int((outcome == 1).sum()), int((outcome == -1).sum())
        p_tp, p_sl, p_time = exact_shares([n_tp, n_sl, self.n_paths - n_tp - n_sl])
        return BarrierOutcome(
            p_take_profit=p_tp,
            p_stop_loss=p_sl,
            p_time_exit=p_time,
            expected_trade_return=float(trade.mean()),
            kelly_optimal=kelly_optimal(trade),
            n_paths=self.n_paths,
            horizon_days=d,
        )

    # ------------------------------------------------------------------ transforms / summary

    def shift_mu(self, delta: float) -> ReturnDistribution:
        """New instance with mu_e shifted by `delta` (total mean shifts by delta); same seed and draws."""
        return ReturnDistribution(
            self.mu_e + float(delta),
            self.sigma_e,
            self.mu_m,
            self.sigma_m,
            self.nu,
            self.nu_m,
            self.beta,
            self.rho,
            self.horizon,
            self.n_paths,
            self.seed,
            self.direction_edge,
            self.relative_edge,
            hold_days=self.hold_days,
        )

    def summary(
        self,
        *,
        leaf_id: int = -1,
        leaf_path: Sequence[str] = (),
        leaf_n: int = 0,
        offsets: Mapping[str, float] | None = None,
        cv_metrics: Mapping[str, float] | None = None,
        skill_ok: bool = False,
        degraded: bool = False,
        degraded_reasons: Sequence[str] = (),
        blend: Mapping[str, float] | None = None,
        feature_set: str = "v1",
        model: str = "RSMT-Q",
        direction_skill_ok: bool | None = None,
    ) -> DistributionForecast:
        """Headline forecast; every number comes from this instance's path matrix / parameters.

        direction_skill_ok: the skill behind the served mean's direction (DistributionForecast.direction_skill_ok);
        None = `skill_ok` (callers that do not separate the two verdicts)."""
        q10, q50, q90 = (float(v) for v in self.quantile([0.10, 0.50, 0.90]))
        reasons = list(degraded_reasons)
        return DistributionForecast(
            model=model,
            feature_set=feature_set,
            horizon_days=self.horizon,
            mu=self.mean_log,
            sigma=self.sd_log,
            nu=self.nu,
            beta=self.beta,
            mu_excess=self.mu_e,
            sigma_excess=self.sigma_e,
            mu_market=self.mu_m,
            sigma_market=self.sigma_m,
            rho=self.rho,
            blend=dict(blend or {}),
            prob_up=self.prob_up,
            prob_outperform=self.prob_outperform,
            expected_return=self.expected_return,
            q10=q10,
            q50=q50,
            q90=q90,
            var95=self.var95,
            cvar95=self.cvar95,
            direction_edge=self.direction_edge,
            relative_edge=self.relative_edge,
            leaf_id=int(leaf_id),
            leaf_path=list(leaf_path),
            leaf_n=int(leaf_n),
            offsets={k: float(v) for k, v in (offsets or {}).items()},
            # non-finite metrics (e.g. an undefined IC) stay in the CV diagnostics; JSON would turn them
            # into null, which does not round-trip through the float-valued contract field
            cv_metrics={k: float(v) for k, v in (cv_metrics or {}).items() if math.isfinite(float(v))},
            skill_ok=bool(skill_ok),
            direction_skill_ok=bool(skill_ok if direction_skill_ok is None else direction_skill_ok),
            degraded=bool(degraded or reasons),
            degraded_reasons=reasons,
        )

    def __repr__(self) -> str:
        return (
            f"ReturnDistribution(mu_e={self.mu_e:+.4f}, sigma_e={self.sigma_e:.4f}, mu_m={self.mu_m:+.4f}, "
            f"sigma_m={self.sigma_m:.4f}, nu={self.nu:.2f}, nu_m={self.nu_m:.2f}, beta={self.beta:.2f}, "
            f"rho={self.rho:+.2f}, h={self.horizon}, hold={self.hold_days}, n_paths={self.n_paths}, "
            f"seed={self.seed})"
        )
