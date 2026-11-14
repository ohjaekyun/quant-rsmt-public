"""RSMT-Q: a Zeileis-Hothorn-Hornik model-based (MOB) score tree whose leaves hold return distributions.

Role (2026-09-29, docs/RSMT.md): RSMT-Q is the RISK / DISTRIBUTION engine. The served excess MEAN comes 100% from
the selected stock-selection model when its serving artifact exists (`ExcessOverride`: no RSMT-Q leaf mean);
RSMT-Q's own leaf mean (100%) is only the fallback without an artifact. Its skill gate is
therefore a DISTRIBUTION gate by default (rsmt.skill_metric, see "CV skill gate" below); the direction (AUC) gate
stays computed and reported (`direction_skill_ok`).

Model switches (rsmt.<key>, per market rsmt.markets.<M>.<key>; defaults = the original model, config/default.yaml
holds the variant chosen by the pre-declared comparison of docs/RSMT.md):
- overlap_step (sample efficiency): training rows every k sessions (quant.panel), labels overlap; the split test
  clusters on h-session BLOCKS (the multiplier bootstrap below with one multiplier per block = a Bartlett-kernel
  HAC / Driscoll-Kraay variance of the cross-sectionally summed scores, since two overlapping h-day windows at lag
  d share (h-d)/h of their days = the Bartlett weight, which the h-block assignment reproduces on average);
  leaf shrinkage counts n / (sample_step / k) effective rows and min_leaf non-overlapping rows. CV folds / test rows
  / the market leg stay those of the non-overlapping grid (`on_grid`).
- multiple_testing: bonferroni | simes (Simes' global p over the variable tests, valid under positive dependence;
  per-variable BH p) | pca (the 9 correlated regime variables -> K principal components fitted on the training rows;
  the stored loadings route new rows; Bonferroni over K).
- target_scale = ewma: split on e / sigma_v,h (EWMA vol at t * sqrt(h/252), the panel's scale_h), serve back in
  returns; the eta offset stays as the RESIDUAL volatility elasticity (raw-model eta = 1 + eta).
- gamma_mode = leaf: leaf-local momentum gamma, empirical-Bayes shrunk toward the parent node's
  ((n_eff g_hat + k g_parent) / (n_eff + k), n_eff = independent dates / blocks, k = gamma_shrink_k).

Adaptation of the user's RSMT sports model (leaves = Poisson lambdas) to h-day equity returns. Each
leaf holds (mu, sigma^2) of the beta-adjusted excess log return e = r - beta*m plus a tail nu and the
residual dependence rho with the market. Downstream every quantity comes from ONE simulated path
matrix (`quant.distribution.ReturnDistribution`).

Node model: Gaussian QMLE on the offset-adjusted target ytil = y - gamma*z with scale s = exp(eta*v):
    mu = sum(w*ytil/s^2) / sum(w/s^2),   sigma^2 = sum(w*(ytil-mu)^2/s^2) / sum(w)
Scores (N x 2): S_i = w_i * [(ytil_i-mu)/s_i^2, (ytil_i-mu)^2/s_i^2 - sigma^2], which sum to 0 at the
node MLE. Meat B = S'S/N; standardisation with the symmetric B^{-1/2} (NOT the sandwich) as in RSMT.

Growing (per node):
1. Variable test, no split point tried: stable-sort by x, W_k = N^{-1/2} sum_{i<=k} B^{-1/2} S_(i),
   LM_k = |W_k|^2 / (t_k(1-t_k)), t_k = k/N, evaluated at tie-group boundaries inside
   [trim*N, (1-trim)*N]; supLM = max LM_k; p from the Monte Carlo Brownian-bridge null,
   p = (1 + #{null >= stat}) / (n_sims + 1); Bonferroni p_adj = min(1, m*p). Leaf if min p_adj >= alpha.
   When a variable has only a few admissible boundaries (<= DISCRETE_NULL_MAX_POINTS) the null is the
   sup of the bridge at exactly those t's (exact, so few-valued variables are not penalised and
   many-valued variables are not preferred).
   Pooled panel (rows clustered by date, time-decay weights): the bridge null is invalid there (a
   market-wide shock per date made the test split on date-level noise in 12/12 null panels; weights
   correlated with a time-ordered x inflated a 5% test to 28%). The SAME statistic is then referred
   to a date-cluster multiplier bootstrap with the Hessian-fraction bridge (`cluster_bootstrap_null`,
   Besag-Clifford early stop). `fit_frame` passes the panel's h-session `block` (= the sample date on the
   non-overlapping grid) or the `date` column as the cluster. rsmt.multiple_testing replaces Bonferroni by
   Simes / BH or the tested variables by K principal components (see "Model switches").
2. Split scan in the selected variable only: Q(k) = N/(N_L N_R) * S_L' B^{-1} S_L from cumulative
   sums (S_R = -S_L), tie boundaries with N_L, N_R >= min_leaf AND (pooled panel) each child
   spanning >= min_leaf_dates distinct sample dates (rows on one date share the market shock, so a
   child of a few dates is a few observations); split value = midpoint of the two adjacent
   distinct values. Stop at max_depth, n < 2*min_leaf or dates < 2*min_leaf_dates.

CV skill gate (`cross_validate`), per fold layout, two gates (both always computed):
- direction: auc - base_auc >= skill_min_auc_gain, logloss < base_logloss and (skill_require_date_ci) the 5% bound
  of the AUC gain from a bootstrap over whole CV test DATES > 0;
- distribution (rsmt.skill_metric = "distribution", the default): the date-bootstrap 5% bound of the CRPS gain over
  the baseline twin > 0 AND cover80 in rsmt.skill_cover80_band ([0.75, 0.85]) AND exc95 (VaR95 exceedance) in
  rsmt.skill_exc95_band ([0.02, 0.08]).
skill_ok = the rsmt.skill_metric gate. Fold-robust verdict: skill_ok holds only when that gate passes for a strict
MAJORITY of the fold layouts rsmt.robust_cv_folds (each a full expanding-window CV); direction_ok /
distribution_ok are the same majority verdicts of each gate. Live 2026-09-24 the KR direction gate passed with 4
folds (bound +0.0006) and failed with 3 and 5 (bounds -0.0098 / -0.0095): a single-layout pass is the fold
boundaries' noise. The primary metrics, delta and the OOS frame stay those of rsmt.cv_folds; every layout's
verdict is in `CVResult.layouts` and cv_metrics (gate_folds_<k> = 1/0 of skill_metric's gate,
direction_gate_folds_<k> / distribution_gate_folds_<k>, direction_gate / distribution_gate). Served:
DistributionForecast.skill_ok = the robust skill_metric verdict; DistributionForecast.direction_skill_ok = the
direction verdict (or the stock-selection model's served skill under an ExcessOverride) - the directional ML-pillar
multiplier reads direction_skill_ok, so a distribution pass never un-shrinks a direction.

Serving (`MarketModel.predict_many` / `predict`): one ReturnDistribution per ticker at the model
horizon h, simulated over the holding horizon (fusion.target_horizon_days) for the trade barrier.
Market leg: mu_m = (1 - s) * time-decayed historical mean + s * log(1 + prior_annual) * h / 252
(`quant.panel.market_moments`; the same rule in every CV fold). Plausibility clamp: the served total
mean satisfies |mu| <= MU_PLAUSIBLE_K * sigma_h (sigma_h = the exact sd of the simulated h-day log
return); a binding clamp is recorded in `offsets` (mu_clamp_shift, mu_cap, mu_total_unclamped) and in
health / diagnostics, and the CV applies the same clamp so the metrics measure what is served.

Leaves: mu shrunk toward the (shrunk) parent mu, (n*mu + k*mu_parent)/(n+k); sigma^2 as is;
nu = 4 + 6/kappa of the standardised residuals (nu_max if kappa <= 0) clipped per market;
rho = weighted corr(standardised residual, standardised m) * n/(n + k).

Offsets (global gamma on mu, eta on log sigma; not split variables): 2 outer rounds of
[grow with offsets fixed (start gamma=0, eta=1)] -> [alternate leaf closed form / gamma WLS /
bounded eta Newton until convergence]. Missing offsets raise `RSMTOffsetError` - never dropped silently.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from quantcio.config import Settings
from quantcio.data.base import normalise_ticker
from quantcio.i18n import lang_of, tr
from quantcio.models import DistributionForecast, Market
from quantcio.quant.distribution import (
    ReturnDistribution,
    apply_edge,
    nu_from_excess_kurtosis,
    path_seed,
    total_log_sd,
    weighted_excess_kurtosis,
)
from quantcio.quant.panel import (
    PanelData,
    is_offset_missing,
    market_config,
    moments_from_settings,
    on_grid_rows,
    overlap_factor,
    rsmt_option,
)

log = logging.getLogger(__name__)

TRADING_DAYS = 252
N_SCORE = 2  # score columns: (mu, sigma^2)
DISCRETE_NULL_MAX_POINTS = 128  # <= this many admissible boundaries -> exact discrete bridge null
NULL_BATCH = 500
BOOT_SIMS = 999  # multiplier-bootstrap draws of the sup-LM null (clustered / weighted fits)
BOOT_BATCH_FLOATS = 2_000_000  # bounds the bootstrap working set (~16 MB of float64)
BOOT_STOP_EXCEED = 10  # Besag-Clifford: stop once this many draws exceed the statistic
FAILED_PANEL_MAX = 0.30  # health: > 30% of the panel failed to load -> degraded
# user-facing texts: Korean / English pairs (quantcio.i18n; the model keeps the report language it was built with)
ZERO_SPLIT_INFO = "분할 검정 미통과 — 표본 확대 필요"
ZERO_SPLIT_INFO_EN = "no split passed the test — a larger sample is needed"
SERVED_SCOPE = "서빙 종목"
SERVED_SCOPE_EN = "served names"
SERVED_COLLAPSE_PREFIX = f"예측 붕괴({SERVED_SCOPE})"
SERVED_COLLAPSE_PREFIX_EN = f"forecast collapse ({SERVED_SCOPE_EN})"
CV_PATHS = 1000
# Plausibility clamp on the served total h-day log mean: |mu| <= MU_PLAUSIBLE_K * sigma_h. At h = 20
# k = 0.5 is an annualised Sharpe of 0.5 * sqrt(252/20) = 1.8 for ONE stock's conditional mean, far
# beyond any credible forecast; live 2026-09-24 the served universes topped out at |mu|/sigma_h = 0.19
# (KR) and 0.14 (US), so the clamp only catches pathological fits (a small leaf's bubble mean, a blown-up
# momentum offset gamma*z, an extreme stock-selection mean) and never shapes normal forecasts.
MU_PLAUSIBLE_K = 0.5
# fold layouts of the robust gate run concurrently: ~95% of a layout's time is the Monte Carlo of the
# test rows (numpy RNG releases the GIL); live KR 3 layouts: 16.0 s sequential -> 6.5 s, bit-identical
CV_LAYOUT_WORKERS = 3
MU_CLAMP_INFO = "μ 타당성 상한"
MU_CLAMP_INFO_EN = "μ plausibility cap"
# pooled CV OOS rows (CVResult.oos): realised labels, the model's and the baseline twin's forecasts, and
# the parameters of each row's simulated distribution (mu_e = excess mean actually simulated, after the
# plausibility clamp; beta / market leg shared by both). leaf ids are per fold (CVResult.folds[i]
# ["leaf_paths"] names them). Consumers: audits, `quantcio evaluate` (ranking, calibration, barrier replay).
OOS_COLUMNS = (
    "date",
    "ticker",
    "fold",
    "r",
    "m",
    "p_up",
    "base_p_up",
    "p_out",
    "base_p_out",
    "mu",
    "base_mu",
    "mu_e",
    "base_mu_e",
    "mu_leaf",
    "base_mu_leaf",
    "sigma_e",
    "base_sigma_e",
    "nu",
    "base_nu",
    "rho",
    "base_rho",
    "beta",
    "mu_m",
    "sigma_m",
    "nu_m",
    "leaf_id",
    "base_leaf_id",
    "crps",
    "base_crps",
    "in80",
    "base_in80",
    "exc95",
    "base_exc95",
    "clamped",
    "base_clamped",
)


def plausibility_shift(mu_total: float, sd_total: float, k: float = MU_PLAUSIBLE_K) -> float:
    """Shift that brings the total h-day log mean inside [-k*sd, +k*sd] (0.0 when it does not bind).

    Raises:
        ValueError: non-finite mu / sd, or a negative sd or k (never clamps against a NaN cap).
    """
    mu, sd, kk = float(mu_total), float(sd_total), float(k)
    if not (math.isfinite(mu) and math.isfinite(sd) and sd >= 0.0 and math.isfinite(kk) and kk >= 0.0):
        raise ValueError(f"plausibility_shift needs finite mu, sd >= 0 and k >= 0 (got {mu_total}, {sd_total}, {k})")
    cap = kk * sd
    if mu > cap:
        return cap - mu
    if mu < -cap:
        return -cap - mu
    return 0.0


class RSMTOffsetError(ValueError):
    """Offset inputs (z for gamma, v for eta) are missing/unusable. Raised - never dropped silently."""


# =============================================================================== null


_NULL_MEM: dict[tuple, np.ndarray] = {}
_DISCRETE_MEM: OrderedDict[tuple, np.ndarray] = OrderedDict()
_DISCRETE_MEM_MAX = 512
_NULL_LOCK = threading.RLock()  # markets may be fitted from a thread pool


def null_cache_file(cache_dir: Path, k: int, trim: float, n_grid: int, n_sims: int, seed: int) -> Path:
    """Cache file name; k, trim, n_grid, n_sims and seed are ALL part of it (the null depends on trim)."""
    return Path(cache_dir) / f"suplm_null_k{k}_trim{trim:.4f}_grid{n_grid}_sims{n_sims}_seed{seed}.npy"


def _bb_sup_grid(k: int, trim: float, n_sims: int, n_grid: int, seed: int, batch: int = NULL_BATCH) -> np.ndarray:
    """sup_{t in [trim, 1-trim]} |BB(t)|^2 / (t(1-t)) for a k-dim Brownian bridge on a regular grid."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, n_grid + 1, dtype="float64") / n_grid
    mask = (t >= trim - 1e-12) & (t <= 1.0 - trim + 1e-12)
    te = t[mask]
    denom = te * (1.0 - te)
    sd = math.sqrt(1.0 / n_grid)
    out = np.empty(n_sims)
    for start in range(0, n_sims, batch):
        b = min(batch, n_sims - start)
        walk = np.cumsum(rng.standard_normal((b, n_grid, k)) * sd, axis=1)
        bridge = walk[:, mask, :] - te[None, :, None] * walk[:, -1:, :]
        out[start : start + b] = ((bridge**2).sum(axis=2) / denom).max(axis=1)
    return np.sort(out)


def _bb_sup_points(times: np.ndarray, k: int, n_sims: int, seed: int, batch_floats: int = 4_000_000) -> np.ndarray:
    """Exact null of max_j |BB(t_j)|^2/(t_j(1-t_j)) at the given t's (independent Gaussian increments)."""
    rng = np.random.default_rng(seed)
    pts = np.append(times, 1.0)
    sd = np.sqrt(np.diff(np.concatenate([[0.0], pts])))
    g = times.size
    denom = times * (1.0 - times)
    batch = max(1, batch_floats // ((g + 1) * k))
    out = np.empty(n_sims)
    for start in range(0, n_sims, batch):
        b = min(batch, n_sims - start)
        walk = np.cumsum(rng.standard_normal((b, g + 1, k)) * sd[None, :, None], axis=1)
        bridge = walk[:, :g, :] - times[None, :, None] * walk[:, -1:, :]
        out[start : start + b] = ((bridge**2).sum(axis=2) / denom).max(axis=1)
    return np.sort(out)


def sup_lm_null(
    k: int = 2,
    trim: float = 0.10,
    n_sims: int = 20000,
    n_grid: int = 1000,
    seed: int = 0,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Sorted Monte Carlo sample of sup_{t in [trim,1-trim]} |BB(t)|^2/(t(1-t)) (Andrews 1993).

    Generated in batches (bounded memory) and cached as .npy under `cache_dir` (normally
    settings.cache_path/'rsmt_null') with k, trim, n_grid, n_sims and seed in the file name.
    """
    if not 0 < trim < 0.5:
        raise ValueError(f"trim must be in (0, 0.5), got {trim}")
    key = (int(k), round(float(trim), 6), int(n_grid), int(n_sims), int(seed))
    with _NULL_LOCK:
        path = null_cache_file(cache_dir, *key) if cache_dir is not None else None
        if key in _NULL_MEM:
            # an in-memory hit still honours the on-disk cache of THIS cache_dir (the result must not
            # depend on which cache_dir asked first in the process)
            if path is not None and not path.exists():
                _save_null(path, _NULL_MEM[key])
            return _NULL_MEM[key]
        null: np.ndarray | None = None
        if path is not None and path.exists():
            try:
                loaded = np.load(path)
                if loaded.shape == (n_sims,) and np.isfinite(loaded).all():
                    null = np.sort(loaded)
                else:
                    log.warning(
                        "sup-LM null cache %s has shape %s / non-finite values - regenerating",
                        path,
                        loaded.shape,
                    )
            except Exception as exc:  # noqa: BLE001 - corrupt cache is regenerated, and logged
                log.warning("sup-LM null cache %s unreadable (%s) - regenerating", path, exc)
        if null is None:
            null = _bb_sup_grid(int(k), float(trim), int(n_sims), int(n_grid), int(seed))
            if path is not None:
                _save_null(path, null)
        null.setflags(write=False)
        _NULL_MEM[key] = null
        return null


def _save_null(path: Path, null: np.ndarray) -> None:
    """Atomic .npy write of a null sample; a failure is logged and the in-memory copy is used."""
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.npy")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(tmp, null)
        tmp.replace(path)
    except OSError as exc:
        log.warning("sup-LM null not cached to %s (%s) - using the in-memory copy", path, exc)
        tmp.unlink(missing_ok=True)


def discrete_null(times: np.ndarray, k: int, n_sims: int, seed: int) -> np.ndarray:
    """Sorted null of the sup-LM restricted to the given boundary t's (in-memory LRU cache)."""
    times = np.asarray(times, dtype="float64")
    blob = np.round(times, 12).tobytes()
    key = (int(k), int(n_sims), int(seed), blob)
    with _NULL_LOCK:
        hit = _DISCRETE_MEM.get(key)
        if hit is not None:
            _DISCRETE_MEM.move_to_end(key)
            return hit
    sub_seed = int(np.random.SeedSequence([int(seed), zlib.crc32(blob)]).generate_state(1)[0])
    null = _bb_sup_points(times, k, n_sims, seed=sub_seed)
    null.setflags(write=False)
    with _NULL_LOCK:
        _DISCRETE_MEM[key] = null
        while len(_DISCRETE_MEM) > _DISCRETE_MEM_MAX:
            _DISCRETE_MEM.popitem(last=False)
    return null


def p_value(null_sorted: np.ndarray, stat: float) -> float:
    """(1 + #{null >= stat}) / (n_sims + 1)."""
    n = null_sorted.size
    ge = n - int(np.searchsorted(null_sorted, stat, side="left"))
    return (1.0 + ge) / (n + 1.0)


# ========================================================================= node model


def node_mle(ytil: np.ndarray, s2: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Closed-form node QMLE: mu = sum(w ytil/s^2)/sum(w/s^2), sigma^2 = sum(w (ytil-mu)^2/s^2)/sum(w)."""
    a = w / s2
    mu = float((a * ytil).sum() / a.sum())
    sigma2 = float((w * (ytil - mu) ** 2 / s2).sum() / w.sum())
    return mu, sigma2


def _scores(ytil: np.ndarray, s2: np.ndarray, w: np.ndarray, mu: float, sigma2: float) -> np.ndarray:
    res = ytil - mu
    return np.column_stack([w * res / s2, w * (res * res / s2 - sigma2)])


def node_scores(
    y: np.ndarray,
    z: np.ndarray | None = None,
    v: np.ndarray | None = None,
    w: np.ndarray | None = None,
    gamma: float = 0.0,
    eta: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Score matrix S (N x 2) at the node MLE given offsets. Returns (S, mu, sigma^2); S sums to ~0."""
    y = np.asarray(y, dtype="float64")
    z = np.zeros_like(y) if z is None else np.asarray(z, dtype="float64")
    v = np.zeros_like(y) if v is None else np.asarray(v, dtype="float64")
    w = np.ones_like(y) if w is None else np.asarray(w, dtype="float64")
    ytil = y - gamma * z
    s2 = np.exp(2.0 * eta * v)
    mu, sigma2 = node_mle(ytil, s2, w)
    return _scores(ytil, s2, w, mu, sigma2), mu, sigma2


def inv_sqrt_psd(b: np.ndarray, rel_tol: float = 1e-12) -> np.ndarray | None:
    """Symmetric inverse square root B^{-1/2} (eigen root); None when B is (near) singular."""
    vals, vecs = np.linalg.eigh(b)
    if not np.isfinite(vals).all() or vals.max() <= 0 or vals.min() <= rel_tol * vals.max():
        return None
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def _sorted_cumsum(x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sorted x, cumulative sums of u in that order, boundary positions k = N_L at tie-group ends)."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    cs = np.cumsum(u[order], axis=0)
    k = np.nonzero(xs[1:] > xs[:-1])[0] + 1
    return xs, cs, k


def sup_lm(x: np.ndarray, u: np.ndarray, trim: float) -> tuple[float, np.ndarray]:
    """sup-LM statistic of standardised scores U = S B^{-1/2} ordered by x.

    Returns (supLM, t's of the admissible tie boundaries). No admissible boundary -> (0.0, empty).
    """
    n = x.size
    _, cs, k = _sorted_cumsum(x, u)
    lo, hi = math.ceil(trim * n - 1e-9), math.floor((1.0 - trim) * n + 1e-9)
    k = k[(k >= lo) & (k <= hi)]
    if k.size == 0:
        return 0.0, np.empty(0)
    t = k / n
    lm = (cs[k - 1] ** 2).sum(axis=1) / n / (t * (1.0 - t))
    return float(lm.max()), t


@dataclass
class SplitScan:
    """Admissible split candidates of one variable: N_L, Q(k) and the midpoint split values.

    `dates_left` / `dates_right`: distinct cluster (sample date) counts of each child, when the scan
    was given clusters (else None).
    """

    k: np.ndarray
    q: np.ndarray
    values: np.ndarray
    dates_left: np.ndarray | None = None
    dates_right: np.ndarray | None = None

    @property
    def best(self) -> int:
        return int(np.argmax(self.q))


def _distinct_prefix_suffix(codes_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(left[i], right[i]) = number of distinct codes in codes_sorted[:i] and codes_sorted[i:],
    for i = 0..n (O(n) after one unique per direction)."""
    n = codes_sorted.size
    first = np.zeros(n, dtype=np.int64)
    first[np.unique(codes_sorted, return_index=True)[1]] = 1
    last = np.zeros(n, dtype=np.int64)
    last[n - 1 - np.unique(codes_sorted[::-1], return_index=True)[1]] = 1
    left = np.concatenate([[0], np.cumsum(first)])
    right = np.concatenate([np.cumsum(last[::-1])[::-1], [0]])
    return left, right


def split_scan(
    x: np.ndarray,
    s: np.ndarray,
    min_leaf: int,
    b_inv_sqrt: np.ndarray | None = None,
    clusters: np.ndarray | None = None,
    min_clusters: int = 0,
) -> SplitScan:
    """Q(k) = N/(N_L N_R) * S_L' B^{-1} S_L at every tie boundary with N_L, N_R >= min_leaf.

    With `clusters` (the panel's sample date per row), a boundary is admissible only when BOTH
    children also span >= `min_clusters` distinct clusters: rows on one date share the market-wide
    shock, so a child made of a handful of dates is a handful of observations however many rows
    it has (settings.rsmt.min_leaf_dates).

    O(N log N): one stable sort + cumulative sums, no MLE refit (S_R = -S_L because sum(S) = 0).
    """
    x = np.asarray(x, dtype="float64")
    n = x.size
    empty = SplitScan(np.empty(0, int), np.empty(0), np.empty(0))
    if b_inv_sqrt is None:
        b_inv_sqrt = inv_sqrt_psd(s.T @ s / n)
        if b_inv_sqrt is None:
            return empty
    order = np.argsort(x, kind="stable")
    xs = x[order]
    cs = np.cumsum((s @ b_inv_sqrt)[order], axis=0)
    k = np.nonzero(xs[1:] > xs[:-1])[0] + 1
    keep = (k >= min_leaf) & (k <= n - min_leaf)
    d_left = d_right = None
    if clusters is not None:
        codes = np.asarray(clusters).ravel()
        if codes.size != n:
            raise ValueError("split_scan: clusters must have one label per row")
        left, right = _distinct_prefix_suffix(codes[order])
        d_left, d_right = left[k], right[k]
        keep &= (d_left >= min_clusters) & (d_right >= min_clusters)
        d_left, d_right = d_left[keep], d_right[keep]
    k = k[keep]
    if k.size == 0:
        return empty
    q = n / (k * (n - k)) * (cs[k - 1] ** 2).sum(axis=1)
    values = (xs[k - 1] + xs[k]) / 2.0
    values = np.where(values >= xs[k], xs[k - 1], values)  # float midpoint collapse guard
    return SplitScan(k, q, values, d_left, d_right)


@dataclass
class BootNull:
    """Multiplier-bootstrap reference sample of one variable's sup-LM statistic."""

    draws: np.ndarray  # sorted bootstrap statistics actually drawn
    p: float  # Monte Carlo p-value of the observed statistic (NaN when none was given)
    stopped_early: bool = False  # Besag-Clifford sequential stop (p clearly large)


def cluster_bootstrap_null(
    x: np.ndarray,
    s: np.ndarray,
    b_inv_sqrt: np.ndarray,
    h_mu: np.ndarray,
    h_sigma: np.ndarray,
    clusters: np.ndarray,
    trim: float,
    n_boot: int = BOOT_SIMS,
    seed: int = 0,
    stats_obs: Sequence[float] | None = None,
    stop_exceed: int = BOOT_STOP_EXCEED,
) -> list[BootNull | None]:
    """Multiplier-bootstrap null of the SAME sup-LM statistic, one per column of x.

    Why: the Brownian-bridge null assumes independent, equally weighted rows. The pooled panel has
    neither: rows on one date share the market-wide shock (a date-level variable such as mkt_vol_60
    then splits on noise almost surely), and the time-decay weights are correlated with any
    time-ordered variable (partial sums are no longer a bridge in t = k/N). Both were measured to
    inflate the size of a nominal 5% test to 28-100%.

    Each draw multiplies every cluster's scores by one N(0, 1) multiplier g_c (within-cluster
    dependence kept), forms the partial sums in the variable's order and removes the estimation
    effect with the Hessian-fraction bridge  S*_k - diag(a_mu_k, a_sigma_k) S*_N,  where
    a_mu_k = sum_{i<=k} w/s^2 / sum w/s^2 and a_sigma_k = sum_{i<=k} w / sum w (the node Hessian of
    the (mu, sigma^2) scores is diagonal). The statistic is standardised exactly as the data
    statistic (B^{-1/2}, t_k = k/N, same tie boundaries and trimming), so only its reference
    distribution changes. With singleton clusters and unit weights this reduces to a Monte Carlo
    of the discrete Brownian-bridge null.

    With `stats_obs`, the p-value is (1 + #{draw >= stat}) / (n_boot + 1), or the Besag-Clifford
    sequential p = stop_exceed / draws as soon as `stop_exceed` draws exceed the statistic (a
    valid Monte Carlo p-value; only clearly non-significant variables stop early).

    Returns:
        per column a `BootNull`, or None when the column has no admissible tie boundary.
    """
    x = np.asarray(x, dtype="float64")
    if x.ndim == 1:
        x = x[:, None]
    n, n_var = x.shape
    codes = np.asarray(clusters, dtype=np.int64).ravel()
    if codes.size != n or s.shape[0] != n:
        raise ValueError("cluster_bootstrap_null: x, s and clusters must have one row each")
    n_clu = int(codes.max()) + 1
    lo, hi = math.ceil(trim * n - 1e-9), math.floor((1.0 - trim) * n + 1e-9)
    sum_mu, sum_sig = float(h_mu.sum()), float(h_sigma.sum())
    cluster_tot = np.column_stack([np.bincount(codes, s[:, 0], n_clu), np.bincount(codes, s[:, 1], n_clu)])
    b00, b01, b10, b11 = (float(v) for v in np.asarray(b_inv_sqrt).ravel())
    out: list[BootNull | None] = []
    for j in range(n_var):
        order = np.argsort(x[:, j], kind="stable")
        xs = x[order, j]
        k = np.nonzero(xs[1:] > xs[:-1])[0] + 1
        k = k[(k >= lo) & (k <= hi)]
        if k.size == 0:
            out.append(None)
            continue
        t = k / n
        denom = n * t * (1.0 - t)
        a_mu = (np.cumsum(h_mu[order])[k - 1] / sum_mu)[None, :]
        a_sig = (np.cumsum(h_sigma[order])[k - 1] / sum_sig)[None, :]
        s_sorted, c_sorted = s[order], codes[order]
        # few clusters (panel dates): partial sums per cluster at each boundary, then one matmul
        # per batch; many clusters (e.g. singletons): per-row multipliers and a cumulative sum
        matmul = n * n_clu <= BOOT_BATCH_FLOATS and k.size * n_clu <= BOOT_BATCH_FLOATS
        if matmul:
            mk = []
            for col in (0, 1):
                acc = np.zeros((n, n_clu))
                acc[np.arange(n), c_sorted] = s_sorted[:, col]
                mk.append(np.cumsum(acc, axis=0)[k - 1].T.copy())  # (C, K)
        stat = None if stats_obs is None else float(stats_obs[j])
        rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, j]).generate_state(2))
        draws: list[np.ndarray] = []
        done, exceed, p, early = 0, 0, float("nan"), False
        batch = 64
        cap = max(1, BOOT_BATCH_FLOATS // max(1, 2 * (k.size if matmul else n)))
        while done < n_boot:
            b = min(batch, cap, n_boot - done)
            g = rng.standard_normal((b, n_clu))
            tot = g @ cluster_tot  # (b, 2) bootstrap score totals
            if matmul:
                c0 = g @ mk[0]
                c1 = g @ mk[1]
            else:
                g_rows = g[:, c_sorted]  # (b, n) multipliers in this variable's order
                c0 = np.cumsum(g_rows * s_sorted[:, 0], axis=1)[:, k - 1]
                c1 = np.cumsum(g_rows * s_sorted[:, 1], axis=1)[:, k - 1]
            c0 -= a_mu * tot[:, :1]
            c1 -= a_sig * tot[:, 1:]
            u0 = c0 * b00 + c1 * b10
            u1 = c0 * b01 + c1 * b11
            sup = ((u0 * u0 + u1 * u1) / denom).max(axis=1)
            draws.append(sup)
            if stat is not None:
                hits = np.cumsum(sup >= stat)
                if exceed + int(hits[-1]) >= stop_exceed:
                    at = int(np.searchsorted(hits, stop_exceed - exceed)) + 1
                    p, early = stop_exceed / float(done + at), True
                    done += b
                    break
                exceed += int(hits[-1])
            done += b
            batch *= 2
        if stat is not None and not early:
            p = (1.0 + exceed) / (n_boot + 1.0)
        out.append(BootNull(np.sort(np.concatenate(draws)), p, early))
    return out


# ============================================================================= offsets


@dataclass
class OffsetFit:
    gamma: float
    eta: float
    n_iter: int
    converged: bool


def fit_offsets(
    y: np.ndarray,
    z: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    leaf: np.ndarray,
    gamma0: float = 0.0,
    eta0: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-8,
    eta_bounds: tuple[float, float] = (-0.5, 2.0),
) -> OffsetFit:
    """Global offsets with the leaves fixed: alternate leaf (mu, sigma^2) closed form, gamma by WLS
    gamma = sum(a z (y-mu_L)) / sum(a z^2) with a = w/(sigma_L^2 s^2), and a bounded Newton step for
    eta on l(eta) = sum w[-eta v - r^2 / (2 sigma_L^2 e^{2 eta v})].

    The (mu_L, gamma) block is solved jointly each iteration (Frisch-Waugh: gamma from z and y
    centred on their a-weighted leaf means, then mu_L = ybar_L - gamma zbar_L). This has the same
    fixed point as the coordinate WLS step above, but it does not zig-zag when z is correlated
    with leaf membership (momentum vs dd_250 / dist_ma200 leaves). The plain coordinate step needed
    > 50 iterations at corr 0.95 and stopped 0.5% short of the joint MLE."""
    leaf = np.asarray(leaf, dtype=np.int64)
    nl = int(leaf.max()) + 1
    gamma, eta = float(gamma0), float(eta0)
    lo, hi = eta_bounds
    sw_leaf = np.bincount(leaf, w, nl)
    for it in range(1, max_iter + 1):
        s2 = np.exp(2.0 * eta * v)
        ytil = y - gamma * z
        a = w / s2
        sa = np.bincount(leaf, a, nl)
        mu_l = np.bincount(leaf, a * ytil, nl) / sa
        res = ytil - mu_l[leaf]
        sig2_l = np.bincount(leaf, w * res * res / s2, nl) / sw_leaf
        sig2_l = np.maximum(sig2_l, 1e-300)
        aa = w / (sig2_l[leaf] * s2)
        # within a leaf aa is proportional to a, so the a-weighted leaf means are the aa ones
        zc = z - (np.bincount(leaf, a * z, nl) / sa)[leaf]
        yc = y - (np.bincount(leaf, a * y, nl) / sa)[leaf]
        den = float((aa * zc * zc).sum())
        gamma_new = float((aa * zc * yc).sum() / den) if den > 0 else gamma
        mu_l = np.bincount(leaf, a * (y - gamma_new * z), nl) / sa
        r2 = (y - gamma_new * z - mu_l[leaf]) ** 2
        q = r2 * np.exp(-2.0 * eta * v) / sig2_l[leaf]
        grad = float((w * (q * v - v)).sum())
        hess = float(-2.0 * (w * q * v * v).sum())
        eta_new = min(max(eta - grad / hess, lo), hi) if hess < 0 else eta
        step = abs(gamma_new - gamma) + abs(eta_new - eta)
        gamma, eta = gamma_new, eta_new
        if step < tol:
            return OffsetFit(gamma, eta, it, True)
    log.warning(
        "RSMT-Q offsets did not converge in %d iterations (gamma=%.6g, eta=%.6g)",
        max_iter,
        gamma,
        eta,
    )
    return OffsetFit(gamma, eta, max_iter, False)


def fit_offsets_multi(
    y: np.ndarray,
    z: np.ndarray,
    vm: np.ndarray,
    w: np.ndarray,
    leaf: np.ndarray,
    gamma0: float,
    eta0: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[float, np.ndarray, int, bool]:
    """`fit_offsets` with a vector sigma equation log s = vm @ eta (column 0 = v, then the sigma terms): the same
    joint (mu_L, gamma) block, then a bounded (componentwise clipped) Newton step for eta on
    l(eta) = sum w [-vm eta - r^2 / (2 sigma_L^2 e^{2 vm eta})], halved while the likelihood does not improve.
    With one column it has the fixed point of `fit_offsets` (docs/RSMT.md section 8). Returns (gamma, eta, iterations,
    converged)."""
    leaf = np.asarray(leaf, dtype=np.int64)
    nl = int(leaf.max()) + 1
    gamma, eta = float(gamma0), np.asarray(eta0, dtype="float64").copy()
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    sw_leaf = np.bincount(leaf, w, nl)

    def loglik(g: float, e: np.ndarray, mu_l: np.ndarray, sig2_l: np.ndarray) -> float:
        lv = vm @ e
        r2 = (y - g * z - mu_l[leaf]) ** 2
        return float((w * (-lv - 0.5 * r2 * np.exp(-2.0 * lv) / sig2_l[leaf])).sum())

    for it in range(1, max_iter + 1):
        lv = vm @ eta
        s2 = np.exp(2.0 * lv)
        ytil = y - gamma * z
        a = w / s2
        sa = np.bincount(leaf, a, nl)
        mu_l = np.bincount(leaf, a * ytil, nl) / sa
        res = ytil - mu_l[leaf]
        sig2_l = np.maximum(np.bincount(leaf, w * res * res / s2, nl) / sw_leaf, 1e-300)
        aa = w / (sig2_l[leaf] * s2)
        zc = z - (np.bincount(leaf, a * z, nl) / sa)[leaf]
        yc = y - (np.bincount(leaf, a * y, nl) / sa)[leaf]
        den = float((aa * zc * zc).sum())
        gamma_new = float((aa * zc * yc).sum() / den) if den > 0 else gamma
        mu_l = np.bincount(leaf, a * (y - gamma_new * z), nl) / sa
        r2 = (y - gamma_new * z - mu_l[leaf]) ** 2
        q = r2 * np.exp(-2.0 * lv) / sig2_l[leaf]
        grad = vm.T @ (w * (q - 1.0))
        hess = -2.0 * (vm * (w * q)[:, None]).T @ vm
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.zeros_like(eta)
        base = loglik(gamma_new, eta, mu_l, sig2_l)
        eta_new = np.clip(eta - step, lo, hi)
        for _ in range(20):
            if loglik(gamma_new, eta_new, mu_l, sig2_l) >= base - 1e-12:
                break
            eta_new = eta + 0.5 * (eta_new - eta)
        change = abs(gamma_new - gamma) + float(np.abs(eta_new - eta).sum())
        gamma, eta = gamma_new, eta_new
        if change < tol:
            return gamma, eta, it, True
    log.warning(
        "RSMT-Q sigma-equation offsets did not converge in %d iterations (gamma=%.6g, eta=%s)", max_iter, gamma, eta
    )
    return gamma, eta, max_iter, False


# ================================================================================ tree


@dataclass
class VariableTest:
    variable: str
    stat: float
    p: float
    p_adj: float
    n_points: int
    null: str  # "continuum" | "discrete" | "bootstrap" | "none"
    null_q99: float = float("nan")  # 99th percentile of the null used (tie-break scale)


@dataclass
class TreeNode:
    node_id: int
    depth: int
    rows: np.ndarray = field(repr=False)
    path: tuple[str, ...] = ()
    mu_raw: float = float("nan")
    mu: float = float("nan")
    sigma2: float = float("nan")
    nu: float = float("nan")
    rho: float = 0.0
    gamma: float = float("nan")  # momentum offset of the node (global gamma, or the shrunk leaf gamma)
    gamma_hat: float = float("nan")  # the node's own WLS gamma before shrinkage (gamma_mode = "leaf")
    tests: list[VariableTest] = field(default_factory=list)
    p: float = float("nan")  # node p: min Bonferroni p, or the Simes p (NaN if not tested)
    split_var: str | None = None
    split_value: float = float("nan")
    q_max: float = float("nan")
    left: TreeNode | None = None
    right: TreeNode | None = None
    leaf_id: int = 0
    stop_reason: str = ""
    n_dates: int = 0  # distinct clusters (sample dates) among the node's rows

    @property
    def n(self) -> int:
        return int(self.rows.size)

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    @property
    def sigma(self) -> float:
        return math.sqrt(self.sigma2)


def _fmt_value(x: float) -> str:
    return f"{x:.3g}"


class RSMTree:
    """MOB score tree over (mu, sigma^2) of the excess return with global offsets (see module doc).

    null_method: "auto" (default) refers the sup-LM statistic to the Brownian-bridge null for
    independent, equally weighted rows and to the cluster multiplier bootstrap (`boot_sims` draws,
    Besag-Clifford early stop) when `fit(cluster=...)` is given or the weights vary; "bb" /
    "bootstrap" force one. The fitted choice is `null_used` (shown in MarketModel.diagnostics).

    min_leaf_dates: every child of a split must also span >= this many distinct clusters (the
    panel's sample dates; without `cluster` each row is its own cluster, so it acts like a row
    minimum). A node with < 2*min_leaf_dates dates is not split.

    lang: language of the user-facing texts (offset errors, `describe`): "ko" (default) or "en";
    `from_settings` takes it from settings.report.language.

    Model switches (docs/RSMT.md; defaults = the original model):
    - multiple_testing: "bonferroni" (p_adj = m p), "simes" (node p = Simes' global p = min_j m p_(j) / j, valid
      under positive dependence of the variable tests; per-variable p_adj = the Benjamini-Hochberg step-up p) or
      "pca" (the `pca_features` - default every partition variable - are replaced by their first `pca_components`
      principal components, fitted on the training rows only (standardised, correlation-matrix eigenvectors); the
      stored loadings route new rows; Bonferroni over the K components plus any variable not reduced).
    - target_scale: "none" or "ewma" - `fit(scale=...)` splits on y / scale (the panel's `scale_h` = EWMA vol at t *
      sqrt(h/252)); every leaf / offset parameter is then in scale units and `predict_params(scale=...)` converts
      back: mu_e = scale * (mu_leaf + gamma z), sigma_e = scale * sigma_leaf * exp(eta v). The eta offset stays as
      a RESIDUAL volatility elasticity (`eta_equivalent` = 1 + eta, the raw-mode eta; bounds shifted by -1).
    - gamma_mode: "global" or "leaf" - after the tree and the global (gamma, eta) are fitted, every node gets its own
      WLS gamma_hat (within-node intercept, weights w / s^2), shrunk top-down toward its parent's:
      gamma_node = (n_eff gamma_hat + k gamma_parent) / (n_eff + k), n_eff = the node's distinct clusters
      (independent sample dates / h-session blocks), k = gamma_shrink_k; the root keeps the global gamma, so a
      tree without splits is unchanged. Leaf means / sigmas are then refitted with the leaf gammas. The split test
      itself still uses the global gamma offset (gamma is not a score column).
    - row_multiplicity: training rows per independent observation (overlapping samples: sample_step /
      overlap_step); the leaf-mean and rho shrinkage use n_eff = n / row_multiplicity.
    """

    lang: str = "ko"  # class default: trees pickled before the attribute existed stay Korean
    # class defaults: trees pickled before the switches existed behave as the original model
    multiple_testing: str = "bonferroni"
    pca_components: int = 3
    pca_features: list[str] | None = None
    target_scale: str = "none"
    gamma_mode: str = "global"
    gamma_shrink_k: float = 50.0
    row_multiplicity: float = 1.0
    pca: dict[str, Any] | None = None
    _split_names: list[str] | None = None

    def __init__(
        self,
        partition_features: Sequence[str],
        *,
        alpha: float = 0.05,
        min_leaf: int = 150,
        max_depth: int = 4,
        trim: float = 0.10,
        mu_shrink_k: float = 200.0,
        nu_min: float = 4.5,
        nu_max: float = 30.0,
        null_sims: int = 20000,
        n_grid: int = 1000,
        null_seed: int = 0,
        cache_dir: Path | None = None,
        use_offsets: bool = True,
        outer_rounds: int = 2,
        gamma0: float = 0.0,
        eta0: float = 1.0,
        offset_max_iter: int = 50,
        offset_tol: float = 1e-8,
        eta_bounds: tuple[float, float] = (-0.5, 2.0),
        null_method: str = "auto",
        boot_sims: int = BOOT_SIMS,
        min_leaf_dates: int = 0,
        lang: str = "ko",
        multiple_testing: str = "bonferroni",
        pca_components: int = 3,
        pca_features: Sequence[str] | None = None,
        target_scale: str = "none",
        gamma_mode: str = "global",
        gamma_shrink_k: float = 50.0,
        row_multiplicity: float = 1.0,
        sigma_terms: Sequence[str] = (),
        theta_bounds: tuple[float, float] = (-3.0, 3.0),
        boot_block_horizons: int = 1,
    ):
        if not partition_features:
            raise ValueError("RSMTree needs at least one partition feature")
        if int(boot_sims) < 1:
            raise ValueError(f"boot_sims must be >= 1, got {boot_sims}")
        if null_method not in ("auto", "bb", "bootstrap"):
            raise ValueError(f"null_method must be 'auto', 'bb' or 'bootstrap', got {null_method!r}")
        if multiple_testing not in ("bonferroni", "simes", "pca"):
            raise ValueError(f"multiple_testing must be 'bonferroni', 'simes' or 'pca', got {multiple_testing!r}")
        if target_scale not in ("none", "ewma", "vol"):
            raise ValueError(f"target_scale must be 'none', 'ewma' or 'vol', got {target_scale!r}")
        if gamma_mode not in ("global", "leaf"):
            raise ValueError(f"gamma_mode must be 'global' or 'leaf', got {gamma_mode!r}")
        if int(pca_components) < 1 or not float(gamma_shrink_k) > 0 or not float(row_multiplicity) >= 1.0:
            raise ValueError(
                f"need pca_components >= 1, gamma_shrink_k > 0 and row_multiplicity >= 1 (got {pca_components}, "
                f"{gamma_shrink_k}, {row_multiplicity})"
            )
        unknown = sorted(set(pca_features or ()) - set(partition_features))
        if unknown:
            raise ValueError(f"pca_features must be partition features, unknown: {unknown}")
        self.multiple_testing = multiple_testing
        self.pca_components = int(pca_components)
        self.pca_features = None if pca_features is None else list(pca_features)
        self.target_scale = target_scale
        self.gamma_mode = gamma_mode
        self.gamma_shrink_k = float(gamma_shrink_k)
        self.row_multiplicity = float(row_multiplicity)
        # sigma equation: log sigma_e = log sigma_leaf + eta * v + theta' V (V = the sigma_terms columns; offsets only)
        self.sigma_terms: tuple[str, ...] = tuple(sigma_terms) if use_offsets else ()
        self.theta_bounds = theta_bounds
        # overlapping rows: bootstrap clusters = this many consecutive h-session blocks (fit_frame); min_leaf_dates and
        # the leaf-gamma n_eff keep counting h-blocks
        self.boot_block_horizons = max(1, int(boot_block_horizons))
        self.theta = np.zeros(len(self.sigma_terms))
        self._v2: np.ndarray | None = None
        self.pca = None
        self._split_names = None
        self.partition_features = list(partition_features)
        self.alpha = float(alpha)
        self.min_leaf = int(min_leaf)
        self.max_depth = int(max_depth)
        self.trim = float(trim)
        self.mu_shrink_k = float(mu_shrink_k)
        self.nu_min, self.nu_max = float(nu_min), float(nu_max)
        self.null_sims, self.n_grid, self.null_seed = (
            int(null_sims),
            int(n_grid),
            int(null_seed),
        )
        self.cache_dir = cache_dir
        self.use_offsets = bool(use_offsets)
        self.outer_rounds = max(1, int(outer_rounds))
        self.gamma0, self.eta0 = float(gamma0), float(eta0)
        self.offset_max_iter, self.offset_tol = int(offset_max_iter), float(offset_tol)
        self.eta_bounds = eta_bounds
        # "auto": Brownian-bridge null for independent, equally weighted rows; the cluster multiplier
        # bootstrap (`cluster_bootstrap_null`) when rows are clustered (panel dates) or weighted
        self.null_method = null_method
        self.boot_sims = int(boot_sims)
        self.min_leaf_dates = max(0, int(min_leaf_dates))
        self.lang = lang_of(lang)
        # fitted state
        self.null_used: str = ""  # "bb" | "bootstrap", set by fit()
        self.n_clusters: int = 0
        self._clusters: np.ndarray | None = None  # bootstrap clusters (None: bridge null)
        self._date_codes: np.ndarray | None = None  # cluster code per row (min_leaf_dates)
        self._round = 0
        self.root: TreeNode | None = None
        self.gamma: float = 0.0
        self.eta: float = 0.0
        self.offset_fit: OffsetFit | None = None
        self.n_train: int = 0
        self._leaves: list[TreeNode] = []
        self._null: np.ndarray | None = None

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_settings(cls, settings: Settings, market: Market | str, **overrides: Any) -> RSMTree:
        """The market's tree with every switch resolved per market (`quant.panel.rsmt_option`).

        Overlapping samples (overlap_step): row_multiplicity = sample_step / overlap_step and min_leaf is counted
        in non-overlapping rows (min_leaf * row_multiplicity training rows). target_scale = "ewma": the eta offset
        is the residual elasticity (start 0, bounds (-1.5, 1.0) = raw (-0.5, 2.0)) unless overridden."""
        mcfg = market_config(settings, market)
        mult = overlap_factor(settings, market)
        kw: dict[str, Any] = dict(
            partition_features=list(settings.rsmt.partition_features),
            alpha=mcfg.alpha,
            min_leaf=int(round(mcfg.min_leaf * mult)),
            max_depth=mcfg.max_depth,
            trim=settings.rsmt.trim,
            mu_shrink_k=settings.rsmt.mu_shrink_k,
            nu_min=mcfg.nu_min,
            nu_max=mcfg.nu_max,
            null_sims=settings.rsmt.null_sims,
            cache_dir=null_cache_dir(settings),
            boot_sims=settings.rsmt.boot_sims,
            min_leaf_dates=settings.rsmt.min_leaf_dates,
            lang=lang_of(settings),
            multiple_testing=rsmt_option(settings, market, "multiple_testing"),
            pca_components=int(rsmt_option(settings, market, "pca_components")),
            target_scale=rsmt_option(settings, market, "target_scale"),
            gamma_mode=rsmt_option(settings, market, "gamma_mode"),
            gamma_shrink_k=float(rsmt_option(settings, market, "gamma_shrink_k")),
            row_multiplicity=mult,
            sigma_terms=tuple(rsmt_option(settings, market, "sigma_terms") or ()),
            boot_block_horizons=int(settings.rsmt.boot_block_horizons),
        )
        kw.update(overrides)
        if kw["target_scale"] != "none" and "eta0" not in overrides:
            kw["eta0"] = 0.0
            kw.setdefault("eta_bounds", (-1.5, 1.0))
        return cls(**kw)

    # ------------------------------------------------------------------ split variables

    @property
    def split_features(self) -> list[str]:
        """Names of the variables the tree tests and splits on (the partition features, or with
        multiple_testing = "pca" the variables not reduced followed by PC1..PCK)."""
        return list(self._split_names) if self._split_names else list(self.partition_features)

    @property
    def scaled(self) -> bool:
        """The target is divided by a per-row volatility scale (target_scale "ewma" / "vol")."""
        return self.target_scale != "none"

    @property
    def eta_equivalent(self) -> float:
        """The eta of the raw-return model: eta, or 1 + eta when the target is scaled by the volatility."""
        return self.eta + (1.0 if self.scaled else 0.0)

    def _pca_columns(self) -> list[int]:
        cols = self.pca_features if self.pca_features is not None else self.partition_features
        return [self.partition_features.index(c) for c in cols]

    def _fit_pca(self, raw: np.ndarray) -> None:
        """Standardise the reduced columns on the training rows and keep the top-K correlation eigenvectors
        (sign: the largest absolute loading positive)."""
        cols = self._pca_columns()
        sub = raw[:, cols]
        mean = sub.mean(axis=0)
        sd = sub.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        zs = (sub - mean) / sd
        vals, vecs = np.linalg.eigh(zs.T @ zs / max(1, zs.shape[0]))
        k = min(self.pca_components, len(cols))
        order = np.argsort(vals)[::-1][:k]
        load = vecs[:, order]
        big = np.argmax(np.abs(load), axis=0)
        load = load * np.where(load[big, np.arange(k)] < 0, -1.0, 1.0)
        total = float(vals.sum())
        keep = [j for j in range(len(self.partition_features)) if j not in cols]
        self.pca = {
            "features": [self.partition_features[j] for j in cols],
            "mean": mean,
            "scale": sd,
            "loadings": load,
            "explained": vals[order] / total if total > 0 else np.zeros(k),
            "keep": keep,
        }
        self._split_names = [*(self.partition_features[j] for j in keep), *(f"PC{i + 1}" for i in range(k))]

    def _split_matrix(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(matrix of the split variables, rows with a missing input). With PCA a missing reduced value is set to
        its training mean (0 after standardisation) and the row is flagged; otherwise NaN stays NaN."""
        missing = ~np.isfinite(raw).all(axis=1)
        if self.pca is None:
            return raw, missing
        pc = self.pca
        cols = self._pca_columns()
        zs = np.nan_to_num((raw[:, cols] - pc["mean"]) / pc["scale"], nan=0.0, posinf=0.0, neginf=0.0)
        return np.column_stack([raw[:, pc["keep"]], zs @ pc["loadings"]]), missing

    def sigma_lines(self, lang: str = "ko") -> list[str]:
        """The sigma equation's extra terms (none without sigma_terms)."""
        if not self.sigma_terms:
            return []
        terms = " ".join(f"{c:+.3f}·{t}" for t, c in zip(self.sigma_terms, self.theta, strict=True))
        return [
            tr(
                lang,
                f"변동성 식: log σ += {self.eta:+.3f}·v {terms}",
                f"sigma equation: log σ += {self.eta:+.3f}·v {terms}",
            )
        ]

    def pca_lines(self) -> list[str]:
        """One line per principal component: explained share and the three largest loadings."""
        if self.pca is None:
            return []
        out = []
        names = self.pca["features"]
        for i in range(self.pca["loadings"].shape[1]):
            col = self.pca["loadings"][:, i]
            top = np.argsort(-np.abs(col))[:3]
            terms = " ".join(f"{col[j]:+.2f}·{names[j]}" for j in top)
            out.append(f"PC{i + 1} ({float(self.pca['explained'][i]):.0%}): {terms}")
        return out

    # ------------------------------------------------------------------ inputs

    def _matrix(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            missing = [c for c in self.partition_features if c not in x.columns]
            if missing:
                raise KeyError(f"partition features missing from frame: {missing}")
            arr = x[self.partition_features].to_numpy(dtype="float64")
        else:
            arr = np.asarray(x, dtype="float64")
            if arr.ndim == 1:
                arr = arr[:, None]
        if arr.shape[1] != len(self.partition_features):
            raise ValueError(f"X has {arr.shape[1]} columns, expected {len(self.partition_features)}")
        return arr

    def _check_offset(self, name: str, arr: np.ndarray | None, w: np.ndarray) -> np.ndarray:
        if arr is None:
            raise RSMTOffsetError(
                tr(
                    self.lang,
                    f"RSMT-Q 오프셋 입력 {name} 없음 — 오프셋을 조용히 생략하지 않음",
                    f"RSMT-Q offset input {name} is missing — the offset is never dropped silently",
                )
            )
        a = np.asarray(arr, dtype="float64").ravel()
        finite = np.isfinite(a)
        if not finite.any():
            raise RSMTOffsetError(
                tr(
                    self.lang,
                    f"RSMT-Q 오프셋 입력 {name}에 유한값이 없음 (n={a.size})",
                    f"RSMT-Q offset input {name} has no finite value (n={a.size})",
                )
            )
        if not finite.all():
            n_nan = int((~finite).sum())
            raise RSMTOffsetError(
                tr(
                    self.lang,
                    f"RSMT-Q 오프셋 입력 {name}에 NaN {n_nan}/{a.size}행 — 상류(panel)에서 제외·집계 필요",
                    f"RSMT-Q offset input {name} has NaN in {n_nan}/{a.size} rows — drop and count them upstream "
                    "(panel)",
                )
            )
        mean = (w * a).sum() / w.sum()
        if math.sqrt((w * (a - mean) ** 2).sum() / w.sum()) < 1e-12:
            raise RSMTOffsetError(
                tr(
                    self.lang,
                    f"RSMT-Q 오프셋 입력 {name}에 변동이 없음 (상수) — 오프셋 식별 불가",
                    f"RSMT-Q offset input {name} does not vary (constant) — the offset is not identified",
                )
            )
        return a

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series,
        z: np.ndarray | pd.Series | None = None,
        v: np.ndarray | pd.Series | None = None,
        w: np.ndarray | pd.Series | None = None,
        m: np.ndarray | pd.Series | None = None,
        cluster: np.ndarray | pd.Series | None = None,
        scale: np.ndarray | pd.Series | None = None,
        boot_cluster: np.ndarray | pd.Series | None = None,
    ) -> RSMTree:
        """Fit structure, leaf parameters and (unless use_offsets=False) the global offsets.

        Args:
            cluster: optional cluster label per row (the panel passes the h-session BLOCK = the sample
                date on the non-overlapping grid: every name on a date shares the market-wide shock, and
                overlapping labels of one block share their window). With clusters or non-uniform weights
                the variable test's p-values come from the cluster multiplier bootstrap (see
                `cluster_bootstrap_null`) instead of the independent-rows Brownian-bridge null.
            scale: target scale per row (target_scale = "ewma": y is divided by it; required then).
            boot_cluster: optional coarser cluster label for the split-test bootstrap only (overlapping rows: a run of
                boot_block_horizons h-blocks, see `fit_frame`); `cluster` still counts min_leaf_dates.

        Raises:
            RSMTOffsetError: z or v missing, all/partly NaN or constant (use_offsets=True).
            ValueError: NaN in X, y, w, m or cluster, non-positive weights, or a missing / non-positive
                scale with target_scale = "ewma".
        """
        xa = self._matrix(X)
        ya = np.asarray(y, dtype="float64").ravel()
        n = ya.size
        wa = np.ones(n) if w is None else np.asarray(w, dtype="float64").ravel()
        if xa.shape[0] != n or wa.size != n:
            raise ValueError("X, y and w must have the same number of rows")
        if n < 2:
            raise ValueError(f"RSMTree.fit needs at least 2 rows, got {n}")
        for name, arr in (("X", xa), ("y", ya), ("w", wa)):
            bad = ~np.isfinite(arr)
            if bad.any():
                raise ValueError(f"RSMTree.fit: {int(bad.sum())} non-finite values in {name} — drop and count upstream")
        if (wa < 0).any() or wa.sum() <= 0:
            raise ValueError("RSMTree.fit: weights must be >= 0 with a positive sum")
        if self.scaled:
            sc = None if scale is None else np.asarray(scale, dtype="float64").ravel()
            if sc is None or sc.size != n or not (np.isfinite(sc).all() and (sc > 0).all()):
                raise ValueError(
                    f"RSMTree.fit: target_scale={self.target_scale!r} needs a finite positive scale per row"
                )
            ya = ya / sc
        if self.multiple_testing == "pca":
            self._fit_pca(xa)
            xa = self._split_matrix(xa)[0]
        else:
            self.pca, self._split_names = None, None
        if self.use_offsets:
            za = self._check_offset("z", None if z is None else np.asarray(z, dtype="float64"), wa)
            va = self._check_offset("v", None if v is None else np.asarray(v, dtype="float64"), wa)
            if za.size != n or va.size != n:
                raise ValueError("z and v must have one value per row")
        else:
            za = np.zeros(n)
            va = np.zeros(n)
        vm = va[:, None]
        if self.sigma_terms:
            v2 = self._v2
            if v2 is None or v2.shape != (n, len(self.sigma_terms)) or not np.isfinite(v2).all():
                raise RSMTOffsetError(
                    tr(
                        self.lang,
                        f"RSMT-Q 변동성 식 입력 {list(self.sigma_terms)} 없음/NaN — fit_frame으로 적합하고 "
                        "상류에서 제외할 것",
                        f"RSMT-Q sigma-equation inputs {list(self.sigma_terms)} missing / NaN — fit through fit_frame "
                        "and drop them upstream",
                    )
                )
            vm = np.column_stack([va, v2])
        ma = None
        if m is not None:
            ma = np.asarray(m, dtype="float64").ravel()
            if ma.size != n or not np.isfinite(ma).all():
                raise ValueError("RSMTree.fit: m must be finite with one value per row")
        codes: np.ndarray | None = None
        if cluster is not None:
            carr = pd.Series(np.asarray(cluster).ravel())
            if carr.size != n or carr.isna().any():
                raise ValueError("RSMTree.fit: cluster must have one non-missing label per row")
            codes = pd.factorize(carr, sort=True)[0].astype(np.int64)
        clustered = codes is not None and int(codes.max()) + 1 < n
        weighted = bool(np.ptp(wa) > 1e-12 * float(wa.max()))
        use_boot = self.null_method == "bootstrap" or (self.null_method == "auto" and (clustered or weighted))
        self.null_used = "bootstrap" if use_boot else "bb"
        self._date_codes = codes if codes is not None else np.arange(n, dtype=np.int64)
        boot_codes = self._date_codes
        if boot_cluster is not None:
            barr = pd.Series(np.asarray(boot_cluster).ravel())
            if barr.size != n or barr.isna().any():
                raise ValueError("RSMTree.fit: boot_cluster must have one non-missing label per row")
            boot_codes = pd.factorize(barr, sort=True)[0].astype(np.int64)
        self._clusters = boot_codes if use_boot else None
        self.n_clusters = int(codes.max()) + 1 if codes is not None else n

        self._null = sup_lm_null(2, self.trim, self.null_sims, self.n_grid, self.null_seed, self.cache_dir)
        self.n_train = n
        gamma = self.gamma0 if self.use_offsets else 0.0
        eta = np.zeros(vm.shape[1])
        eta[0] = self.eta0 if self.use_offsets else 0.0
        offset_fit: OffsetFit | None = None
        for _round in range(self.outer_rounds if self.use_offsets else 1):
            self._round = _round
            ytil = ya - gamma * za
            s2 = np.exp(2.0 * (vm @ eta))
            self._next_id = 0
            root = self._grow(np.arange(n), 0, (), xa, ytil, s2, wa)
            leaves = self._collect_leaves(root)
            if self.use_offsets:
                leaf_index = np.empty(n, dtype=np.int64)
                for i, node in enumerate(leaves):
                    leaf_index[node.rows] = i
                if vm.shape[1] == 1:
                    offset_fit = fit_offsets(
                        ya,
                        za,
                        va,
                        wa,
                        leaf_index,
                        gamma,
                        float(eta[0]),
                        self.offset_max_iter,
                        self.offset_tol,
                        self.eta_bounds,
                    )
                    gamma, eta = offset_fit.gamma, np.array([offset_fit.eta])
                else:
                    gamma, eta, it, ok = fit_offsets_multi(
                        ya,
                        za,
                        vm,
                        wa,
                        leaf_index,
                        gamma,
                        eta,
                        [tuple(self.eta_bounds), *([tuple(self.theta_bounds)] * (vm.shape[1] - 1))],
                        max(self.offset_max_iter, 100),
                        self.offset_tol,
                    )
                    offset_fit = OffsetFit(gamma, float(eta[0]), it, ok)
        self.root = root
        self.gamma, self.eta, self.offset_fit = float(gamma), float(eta[0]), offset_fit
        self.theta = np.asarray(eta[1:], dtype="float64")
        lv = vm @ eta  # log-scale offset index of every training row
        self._leaves = self._collect_leaves(root)
        for i, node in enumerate(self._leaves, start=1):
            node.leaf_id = i
        if self.use_offsets and self.gamma_mode == "leaf":
            g_row = self._fit_node_gammas(root, ya, za, lv, wa)
        else:
            g_row = np.full(n, self.gamma)
            stack = [root]
            while stack:
                node = stack.pop()
                node.gamma = self.gamma
                if not node.is_leaf:
                    stack.extend([node.left, node.right])  # type: ignore[list-item]
        ytil = ya - g_row * za
        s2 = np.exp(2.0 * lv)
        self._finalize(root, None, ytil, s2, wa, ma)
        self._v2 = None  # training inputs are not kept (the fitted model is cached / pickled)
        return self

    def _fit_node_gammas(
        self, root: TreeNode, y: np.ndarray, z: np.ndarray, lv: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        """Leaf-local momentum gamma (gamma_mode = "leaf"), top-down empirical-Bayes shrinkage toward the parent:
        gamma_node = (n_eff gamma_hat + k gamma_parent) / (n_eff + k), gamma_hat = the node's WLS slope of y on z
        with its own intercept (weights w / s^2), n_eff = the node's distinct clusters (independent dates /
        blocks), k = gamma_shrink_k; the root keeps the global gamma. Returns the leaf gamma of every row."""
        a = w / np.exp(2.0 * lv)  # lv = the log-scale offset index (eta * v + theta' V)
        k = self.gamma_shrink_k
        g_row = np.full(y.size, self.gamma)
        stack: list[tuple[TreeNode, float | None]] = [(root, None)]
        while stack:
            node, parent = stack.pop()
            r = node.rows
            ar, zr, yr = a[r], z[r], y[r]
            sa = float(ar.sum())
            zc = zr - float((ar * zr).sum()) / sa
            yc = yr - float((ar * yr).sum()) / sa
            den = float((ar * zc * zc).sum())
            node.gamma_hat = float((ar * zc * yc).sum() / den) if den > 0 else float("nan")
            if parent is None:
                node.gamma = self.gamma
            elif math.isfinite(node.gamma_hat):
                n_eff = float(max(1, node.n_dates))
                node.gamma = (n_eff * node.gamma_hat + k * parent) / (n_eff + k)
            else:
                node.gamma = parent
            if node.is_leaf:
                g_row[r] = node.gamma
            else:
                stack.extend([(node.left, node.gamma), (node.right, node.gamma)])  # type: ignore[list-item]
        return g_row

    def fit_frame(self, df: pd.DataFrame, y_col: str = "y_e") -> RSMTree:
        """Fit from a panel frame (partition features, y_e, z, v, w, m, scale_h columns).

        The cluster label is the panel's `block` (h-session block; on the non-overlapping grid one block = one
        sample date, so this is the date cluster) or, without it, the `date` column: rows on one date share the
        market-wide shock (and overlapping labels share their window), so the variable test is calibrated with
        the cluster bootstrap. target_scale = "ewma" divides y_e by `scale_h`.

        Overlapping rows (row_multiplicity > 1): the split-test bootstrap clusters are runs of boot_block_horizons
        consecutive h-blocks. Labels of neighbouring h-blocks overlap, so a multiplier per h-block keeps only the
        within-block covariances - a Bartlett window of h, ~2/3 of the variance of MA(h-1) scores (null panels split
        in ~10% of the cases at a 5% test, docs/RSMT.md section 8); 3 horizons keep ~89%.
        """
        cluster_col = "block" if "block" in df else ("date" if "date" in df else None)
        scale_col = "scale_h" if self.scaled and "scale_h" in df else None
        boot = None
        if cluster_col == "block" and self.boot_block_horizons > 1 and self.row_multiplicity > 1:
            boot = np.floor_divide(df["block"].to_numpy(dtype=np.int64), self.boot_block_horizons)
        if self.sigma_terms:
            missing = [c for c in self.sigma_terms if c not in df.columns]
            if missing:
                raise RSMTOffsetError(f"sigma-equation columns missing from the frame: {missing}")
            self._v2 = df[list(self.sigma_terms)].to_numpy(dtype="float64")
        return self.fit(
            df[self.partition_features],
            df[y_col].to_numpy(),
            df["z"].to_numpy() if "z" in df else None,
            df["v"].to_numpy() if "v" in df else None,
            df["w"].to_numpy() if "w" in df else None,
            df["m"].to_numpy() if "m" in df else None,
            df[cluster_col].to_numpy() if cluster_col else None,
            df[scale_col].to_numpy() if scale_col else None,
            boot,
        )

    # ------------------------------------------------------------------ growing

    def _test_variables(
        self,
        xn: np.ndarray,
        u: np.ndarray,
        boot: dict[str, Any] | None = None,
    ) -> list[VariableTest]:
        names = self.split_features
        m = len(names)
        out: list[VariableTest] = []
        observed = [sup_lm(xn[:, j], u, self.trim) for j in range(m)]
        boot_null: list[BootNull | None] = []
        if boot is not None:
            boot_null = cluster_bootstrap_null(
                xn,
                boot["s"],
                boot["b_ih"],
                boot["h_mu"],
                boot["h_sigma"],
                boot["clusters"],
                self.trim,
                self.boot_sims,
                boot["seed"],
                stats_obs=[st for st, _ in observed],
            )
        for j, name in enumerate(names):
            stat, times = observed[j]
            null: np.ndarray | None = None
            bn = boot_null[j] if boot is not None else None
            if times.size == 0:
                p, kind = 1.0, "none"
            elif bn is not None:
                null, kind = bn.draws, "bootstrap"
            elif times.size <= DISCRETE_NULL_MAX_POINTS:
                null = discrete_null(times, N_SCORE, self.null_sims, self.null_seed)
                kind = "discrete"
            else:
                null, kind = self._null, "continuum"
            if null is not None:
                p = bn.p if bn is not None else p_value(null, stat)
                q99 = float(np.quantile(null, 0.99))
            else:
                p, q99 = 1.0, float("nan")
            out.append(VariableTest(name, stat, p, float("nan"), int(times.size), kind, q99))
        for test, p_adj in zip(out, self.adjust_p([t.p for t in out])):
            test.p_adj = p_adj
        return out

    def adjust_p(self, p: Sequence[float]) -> np.ndarray:
        """Multiplicity-adjusted p-values of one node's variable tests.

        bonferroni / pca: min(1, m p). simes: the Benjamini-Hochberg step-up p, min_{i >= rank} m p_(i) / i
        (their minimum is Simes' global p, the node's p; valid under positive regression dependence)."""
        pa = np.asarray(p, dtype="float64")
        m = pa.size
        if self.multiple_testing != "simes" or m == 0:
            return np.minimum(1.0, m * pa)
        order = np.argsort(pa, kind="stable")
        ranked = pa[order] * m / np.arange(1, m + 1)
        out = np.empty(m)
        out[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
        return out

    def _grow(
        self,
        rows: np.ndarray,
        depth: int,
        path: tuple[str, ...],
        xa: np.ndarray,
        ytil: np.ndarray,
        s2: np.ndarray,
        w: np.ndarray,
    ) -> TreeNode:
        node = TreeNode(node_id=self._next_id, depth=depth, rows=rows, path=path)
        self._next_id += 1
        yt, s2n, wn = ytil[rows], s2[rows], w[rows]
        node.mu_raw, node.sigma2 = node_mle(yt, s2n, wn)
        node_dates = self._date_codes[rows] if self._date_codes is not None else None
        node.n_dates = int(np.unique(node_dates).size) if node_dates is not None else rows.size
        if depth >= self.max_depth:
            node.stop_reason = "max_depth"
            return node
        if rows.size < 2 * self.min_leaf:
            node.stop_reason = "n < 2*min_leaf"
            return node
        if node.n_dates < 2 * self.min_leaf_dates:
            node.stop_reason = "dates < 2*min_leaf_dates"
            return node
        s = _scores(yt, s2n, wn, node.mu_raw, node.sigma2)
        b_ih = inv_sqrt_psd(s.T @ s / rows.size)
        if b_ih is None:
            node.stop_reason = "singular score covariance"
            return node
        xn = xa[rows]
        boot: dict[str, Any] | None = None
        if self._clusters is not None:
            boot = {
                "s": s,
                "b_ih": b_ih,
                "h_mu": wn / s2n,
                "h_sigma": wn,
                "clusters": np.unique(self._clusters[rows], return_inverse=True)[1],
                "seed": int(
                    np.random.SeedSequence(
                        [self.null_seed, 0xB007, self._round, node.node_id, rows.size]
                    ).generate_state(1)[0]
                ),
            }
        node.tests = self._test_variables(xn, s @ b_ih, boot)

        # select on the UNADJUSTED p: Bonferroni / BH are monotone, but clipping at 1 creates ties that a
        # stat tie-break would hand to the many-valued variable (selection bias under the null). node.p = the
        # selected variable's p_adj = min Bonferroni p, or Simes' global p (the minimum BH p).
        # Ties (e.g. several variables at the Monte Carlo floor) go to the largest stat relative to
        # its OWN null scale - a raw-stat tie-break would favour variables with a wider null.
        def _key(j: int) -> tuple[float, float]:
            t = node.tests[j]
            scale = t.null_q99 if t.null_q99 > 0 else 1.0
            return (t.p, -t.stat / scale)

        best = min(range(len(node.tests)), key=_key)
        node.p = node.tests[best].p_adj
        if node.p >= self.alpha:
            node.stop_reason = "not significant"
            return node
        scan = split_scan(
            xn[:, best],
            s,
            self.min_leaf,
            b_ih,
            clusters=node_dates,
            min_clusters=self.min_leaf_dates,
        )
        if scan.k.size == 0:
            node.stop_reason = (
                "no admissible split (min_leaf / min_leaf_dates)" if self.min_leaf_dates > 1 else "no admissible split"
            )
            return node
        i = scan.best
        name = self.split_features[best]
        node.split_var, node.split_value, node.q_max = (
            name,
            float(scan.values[i]),
            float(scan.q[i]),
        )
        go_left = xn[:, best] <= node.split_value
        node.left = self._grow(
            rows[go_left],
            depth + 1,
            (*path, f"{name} ≤ {_fmt_value(node.split_value)}"),
            xa,
            ytil,
            s2,
            w,
        )
        node.right = self._grow(
            rows[~go_left],
            depth + 1,
            (*path, f"{name} > {_fmt_value(node.split_value)}"),
            xa,
            ytil,
            s2,
            w,
        )
        return node

    @staticmethod
    def _collect_leaves(root: TreeNode) -> list[TreeNode]:
        out: list[TreeNode] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                out.append(node)
            else:
                stack.append(node.right)  # type: ignore[arg-type]
                stack.append(node.left)  # type: ignore[arg-type]
        return out

    def _finalize(
        self,
        node: TreeNode,
        parent_mu: float | None,
        ytil: np.ndarray,
        s2: np.ndarray,
        w: np.ndarray,
        m: np.ndarray | None,
    ) -> None:
        rows = node.rows
        yt, s2n, wn = ytil[rows], s2[rows], w[rows]
        node.mu_raw, node.sigma2 = node_mle(yt, s2n, wn)
        k = self.mu_shrink_k
        n_eff = node.n / self.row_multiplicity  # overlapping rows: independent-observation equivalents
        node.mu = node.mu_raw if parent_mu is None else (n_eff * node.mu_raw + k * parent_mu) / (n_eff + k)
        if node.is_leaf:
            eps = (yt - node.mu_raw) / np.sqrt(node.sigma2 * s2n) if node.sigma2 > 0 else np.zeros_like(yt)
            node.nu = nu_from_excess_kurtosis(weighted_excess_kurtosis(eps, wn), self.nu_min, self.nu_max)
            node.rho = 0.0
            if m is not None and node.n >= 3:
                mm = m[rows]
                sw = wn.sum()
                md = mm - (wn * mm).sum() / sw
                ed = eps - (wn * eps).sum() / sw
                den = math.sqrt((wn * md * md).sum() * (wn * ed * ed).sum())
                if den > 0:
                    rho = float((wn * md * ed).sum() / den)
                    node.rho = float(np.clip(rho * n_eff / (n_eff + k), -0.99, 0.99))
        else:
            self._finalize(node.left, node.mu, ytil, s2, w, m)  # type: ignore[arg-type]
            self._finalize(node.right, node.mu, ytil, s2, w, m)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ accessors

    def _require_fit(self) -> TreeNode:
        if self.root is None:
            raise RuntimeError("RSMTree is not fitted")
        return self.root

    @property
    def leaves(self) -> list[TreeNode]:
        self._require_fit()
        return list(self._leaves)

    @property
    def n_leaves(self) -> int:
        return len(self.leaves)

    @property
    def n_splits(self) -> int:
        return self.n_leaves - 1

    @property
    def root_p(self) -> float:
        return self._require_fit().p

    @property
    def root_tests(self) -> list[VariableTest]:
        return list(self._require_fit().tests)

    def leaf(self, leaf_id: int) -> TreeNode:
        """Leaf by its 1-based id (KeyError for an unknown id - never a wrapped-around leaf)."""
        leaves = self.leaves
        lid = int(leaf_id)
        if not 1 <= lid <= len(leaves):
            raise KeyError(f"leaf id {lid} not in 1..{len(leaves)}")
        return leaves[lid - 1]

    def leaf_path(self, leaf_id: int) -> list[str]:
        return list(self.leaf(leaf_id).path)

    # ------------------------------------------------------------------ prediction

    def apply(self, X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Leaf id (1-based) per row and a mask of rows routed with a missing split value.

        A NaN split value goes to the child with more training rows; callers must flag such rows.
        """
        root = self._require_fit()
        xa, input_missing = self._split_matrix(self._matrix(X))
        n = xa.shape[0]
        leaf = np.zeros(n, dtype=np.int64)
        # PCA: a missing input is imputed at its training mean - flag the row (the raw features route NaN below)
        missing = input_missing.copy() if self.pca is not None else np.zeros(n, dtype=bool)
        stack: list[tuple[TreeNode, np.ndarray]] = [(root, np.arange(n))]
        col = {name: j for j, name in enumerate(self.split_features)}
        while stack:
            node, idx = stack.pop()
            if node.is_leaf:
                leaf[idx] = node.leaf_id
                continue
            x = xa[idx, col[node.split_var]]  # type: ignore[index]
            nan = ~np.isfinite(x)
            go_left = x <= node.split_value
            if nan.any():
                missing[idx[nan]] = True
                go_left[nan] = node.left.n >= node.right.n  # type: ignore[union-attr]
            stack.append((node.left, idx[go_left]))  # type: ignore[arg-type]
            stack.append((node.right, idx[~go_left]))  # type: ignore[arg-type]
        return leaf, missing

    def predict_params(
        self,
        X: pd.DataFrame | np.ndarray,
        z: np.ndarray | pd.Series | None = None,
        v: np.ndarray | pd.Series | None = None,
        scale: np.ndarray | pd.Series | None = None,
    ) -> dict[str, np.ndarray]:
        """Per-row leaf parameters with offsets: mu_e = s * (mu_leaf + gamma_leaf*z), sigma_e = s * sigma_leaf *
        exp(eta*v), with s = `scale` when target_scale = "ewma" (mu_leaf / sigma_leaf / gamma then in scale
        units) and s = 1 otherwise; gamma_leaf = the global gamma unless gamma_mode = "leaf".

        Raises:
            RSMTOffsetError: offsets are used but z / v are missing or non-finite, or the target is scaled and
                `scale` is missing / not finite and positive (the caller must substitute explicitly and flag
                the row, never silently).
        """
        leaf, missing = self.apply(X)
        n = leaf.size
        if self.use_offsets:
            if z is None or v is None:
                raise RSMTOffsetError(
                    tr(
                        self.lang,
                        "predict_params: z와 v가 필요함 (오프셋 모델)",
                        "predict_params: z and v are required (offset model)",
                    )
                )
            za = np.asarray(z, dtype="float64").ravel()
            va = np.asarray(v, dtype="float64").ravel()
            if not (np.isfinite(za).all() and np.isfinite(va).all()):
                raise RSMTOffsetError(
                    tr(
                        self.lang,
                        "predict_params: z/v에 NaN — 명시적으로 대체하고 degraded로 표시할 것",
                        "predict_params: NaN in z/v — substitute explicitly and flag the row degraded",
                    )
                )
        else:
            za = np.zeros(n)
            va = np.zeros(n)
        if self.scaled:
            sc = None if scale is None else np.asarray(scale, dtype="float64").ravel()
            if sc is None or sc.size != n or not (np.isfinite(sc).all() and (sc > 0).all()):
                raise RSMTOffsetError(
                    tr(
                        self.lang,
                        "predict_params: 목표 척도(scale_h)가 없거나 양의 유한값이 아님 — 명시적으로 대체할 것",
                        "predict_params: target scale (scale_h) missing or not finite and positive — substitute "
                        "explicitly",
                    )
                )
        else:
            sc = np.ones(n)
        leaves = self.leaves
        mu_l = np.array([nd.mu for nd in leaves])[leaf - 1]
        sig_l = np.array([nd.sigma for nd in leaves])[leaf - 1]
        g_l = np.array([nd.gamma if math.isfinite(nd.gamma) else self.gamma for nd in leaves])[leaf - 1]
        return {
            "leaf_id": leaf,
            "routed_missing": missing,
            "mu_leaf": mu_l,
            "sigma_leaf": sig_l,
            "gamma": g_l,
            "scale": sc,
            "mu_e": sc * (mu_l + g_l * za),
            "sigma_e": sc * sig_l * np.exp(self.eta * va + self._sigma_terms_index(X, n)),
            "nu": np.array([nd.nu for nd in leaves])[leaf - 1],
            "rho": np.array([nd.rho for nd in leaves])[leaf - 1],
            "leaf_n": np.array([nd.n for nd in leaves])[leaf - 1],
        }

    def _sigma_terms_index(self, X: pd.DataFrame | np.ndarray, n: int) -> np.ndarray:
        """theta' V of the rows of X (0 without sigma terms); X must be a frame with the terms (the panel's pred /
        test rows), missing or non-finite values raise (the panel substitutes and flags them)."""
        if not self.sigma_terms or not self.theta.size:
            return np.zeros(n)
        if not isinstance(X, pd.DataFrame) or any(c not in X.columns for c in self.sigma_terms):
            raise RSMTOffsetError(
                tr(
                    self.lang,
                    f"predict_params: 변동성 식 입력 {list(self.sigma_terms)}이 있는 프레임이 필요함",
                    f"predict_params: a frame with the sigma-equation inputs {list(self.sigma_terms)} is required",
                )
            )
        v2 = X[list(self.sigma_terms)].to_numpy(dtype="float64")
        if not np.isfinite(v2).all():
            raise RSMTOffsetError(
                tr(
                    self.lang,
                    "predict_params: 변동성 식 입력에 NaN — 명시적으로 대체하고 degraded로 표시할 것",
                    "predict_params: NaN in the sigma-equation inputs — substitute explicitly and flag the row "
                    "degraded",
                )
            )
        return v2 @ self.theta

    def sigma_equation(self) -> dict[str, float]:
        """eta (on v) and theta per sigma term of the fitted model."""
        return {"eta": float(self.eta), **{t: float(c) for t, c in zip(self.sigma_terms, self.theta, strict=True)}}

    # ------------------------------------------------------------------ printing

    def _leaf_text(self, node: TreeNode, lang: str = "ko") -> str:
        word = tr(lang, "리프#", "leaf #")
        if self.scaled:  # leaf parameters in units of the horizon volatility
            body = f"μe/σv {node.mu:+.2f} σe/σv {node.sigma:.2f} ν {node.nu:.1f}"
        else:
            body = f"μe {node.mu * 100:+.1f}% σe {node.sigma * 100:.1f}% ν {node.nu:.1f}"
        if self.gamma_mode == "leaf" and math.isfinite(node.gamma):
            body += f" γ {node.gamma:+.4f}"
        return f"{word}{node.leaf_id} n={node.n} {body}"

    def describe(self, lang: str | None = None) -> list[str]:
        """Tree print in the RSMT style (first child = '≤' branch, second = '>' branch).

        lang: "ko" / "en" (default: the tree's own `lang`)."""
        root = self._require_fit()
        lang = lang_of(self.lang if lang is None else lang)
        if root.is_leaf:
            p = "n/a" if math.isnan(root.p) else f"{root.p:.1e}"
            no_split = tr(lang, "분할 없음", "no split")
            return [
                f"{self._leaf_text(root, lang)} ({no_split}: root p={p}, {root.stop_reason})",
                *self.pca_lines(),
                *self.sigma_lines(lang),
            ]
        lines: list[str] = []

        def rec(node: TreeNode, prefix: str, connector: str, child_prefix: str) -> None:
            if node.is_leaf:
                lines.append(prefix + connector + self._leaf_text(node, lang))
                return
            lines.append(
                f"{prefix}{connector}{node.split_var} ≤ {_fmt_value(node.split_value)} (p={node.p:.1e}, n={node.n})"
            )
            nxt = prefix + child_prefix
            rec(node.left, nxt, "├─ ", "│  ")  # type: ignore[arg-type]
            rec(node.right, nxt, "└─ ", "   ")  # type: ignore[arg-type]

        rec(root, "", "", "")
        return [*lines, *self.pca_lines(), *self.sigma_lines(lang)]


def null_cache_dir(settings: Settings) -> Path:
    return settings.cache_path / "rsmt_null"


# ================================================================================== CV


@dataclass
class CVFold:
    fold: int
    train_idx: np.ndarray = field(repr=False)
    test_idx: np.ndarray = field(repr=False)
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_purged: int


def cv_splits(train: pd.DataFrame, n_folds: int) -> list[CVFold]:
    """Expanding-window folds by date: grid dates in n_folds+1 chronological blocks; fold f trains on
    blocks <= f with the purge t_end < test start (label window never reaches the test block) and
    tests on block f+1.

    Overlapping samples (an `on_grid` column): the blocks are cut on the non-overlapping grid dates and only
    on_grid rows are test rows, so the folds and the test rows are those of the non-overlapping panel; every
    earlier row (overlapping ones included) with t_end < test start trains."""
    grid = train["on_grid"].to_numpy(dtype=bool) if "on_grid" in train else np.ones(len(train), dtype=bool)
    d = train["date"].to_numpy()
    dates = np.sort(pd.unique(d[grid]))
    if n_folds < 1 or dates.size < n_folds + 1:
        raise ValueError(f"CV needs >= {n_folds + 1} distinct dates, got {dates.size}")
    blocks = np.array_split(dates, n_folds + 1)
    t_end = train["t_end"].to_numpy()
    folds: list[CVFold] = []
    for f in range(1, n_folds + 1):
        start, end = blocks[f][0], blocks[f][-1]
        before = d < start
        keep = before & (t_end < start)
        test = grid & (d >= start) & (d <= end)
        folds.append(
            CVFold(
                f,
                np.nonzero(keep)[0],
                np.nonzero(test)[0],
                pd.Timestamp(start),
                pd.Timestamp(end),
                int(before.sum() - keep.sum()),
            )
        )
    return folds


@dataclass
class LayoutGate:
    """Skill-gate verdict of ONE expanding-window fold layout (a full CV with `n_folds` folds).

    `skill_ok` is the gate of rsmt.skill_metric; `direction_ok` / `distribution_ok` are both gates."""

    n_folds: int
    skill_ok: bool
    available: bool
    gate: dict[str, bool] = field(default_factory=dict)  # direction-gate components
    auc_gain: float = float("nan")
    auc_gain_p05: float = float("nan")  # date-bootstrap 5% bound (small-cluster corrected)
    logloss_gain: float = float("nan")
    n_oos: int = 0
    n_oos_dates: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    direction_ok: bool = False
    distribution_ok: bool = False
    dist_gate: dict[str, bool] = field(default_factory=dict)  # distribution-gate components
    crps_gain: float = float("nan")  # base_crps - crps (positive = better)
    crps_gain_p05: float = float("nan")  # date-bootstrap 5% bound (small-cluster corrected)
    cover80: float = float("nan")
    exc95: float = float("nan")


@dataclass
class CVResult:
    """Expanding-window CV of RSMT-Q vs the root-only / no-offset baseline.

    `metrics`, `gate`, the edges, `folds` and `oos` are those of the primary layout (rsmt.cv_folds).
    `skill_ok` is the FOLD-ROBUST verdict: the primary layout's CV is available and the single-layout
    gate passes for a strict majority of `layouts` (rsmt.robust_cv_folds); `skill_ok_primary` is the
    primary layout's own single-layout gate.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    direction_edge: float = 0.0
    relative_edge: float = 0.0
    skill_ok: bool = False
    folds: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # DIRECTION-gate components of the primary layout: {"auc_gain": bool, "logloss": bool, "date_ci": bool,
    # "date_ci_required": bool}
    gate: dict[str, bool] = field(default_factory=dict)
    # pooled OOS rows (OOS_COLUMNS: date, ticker, fold, labels, forecasts and each row's simulated
    # distribution parameters, model and base_ twin) for audit / tuning / evaluation; None when CV did not run
    oos: pd.DataFrame | None = field(default=None, repr=False)
    skill_ok_primary: bool = False
    layouts: dict[int, LayoutGate] = field(default_factory=dict)  # n_folds -> verdict (robust set)
    n_folds_primary: int = 0
    seconds: float = 0.0  # wall time of every layout's CV
    # rsmt.skill_metric: "direction" (skill_ok = the AUC / log-loss gate) or "distribution" (skill_ok = the CRPS /
    # coverage gate). Both verdicts are always computed: direction_ok / distribution_ok (single layout, then the
    # fold-robust majority after apply_robust_gate); dist_gate = the primary layout's distribution components
    skill_metric: str = "direction"
    direction_ok: bool = False
    distribution_ok: bool = False
    dist_gate: dict[str, bool] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return bool(self.metrics)

    @property
    def robust_pass(self) -> int:
        return sum(1 for lay in self.layouts.values() if lay.skill_ok)

    def robust_pass_of(self, kind: str) -> int:
        """Layouts passing the "direction" or the "distribution" gate (for rsmt.skill_metric's own kind a layout's
        `skill_ok` IS that gate)."""
        attr = {"direction": "direction_ok", "distribution": "distribution_ok"}[kind]
        if kind == self.skill_metric:
            attr = "skill_ok"
        return sum(1 for lay in self.layouts.values() if getattr(lay, attr))

    @property
    def robust_n(self) -> int:
        return len(self.layouts)

    @property
    def robust_majority(self) -> bool:
        """Strict majority of the fold layouts pass (False without any layout)."""
        return self.robust_n > 0 and 2 * self.robust_pass > self.robust_n


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    pos = label.astype(bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = stats.rankdata(score)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _logloss(p: np.ndarray, y: np.ndarray, eps: float) -> float:
    pc = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))


def _fit_edge(p: np.ndarray, y: np.ndarray, eps: float) -> float:
    """delta = argmin logloss(sigmoid(logit(p) + delta)), bounded to +-0.5."""
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))

    def loss(d: float) -> float:
        return _logloss(1.0 / (1.0 + np.exp(-(lp + d))), y, 1e-12)

    res = optimize.minimize_scalar(loss, bounds=(-0.5, 0.5), method="bounded", options={"xatol": 1e-6})
    return float(np.clip(res.x, -0.5, 0.5))


def _grouped_auc(
    score: np.ndarray, label: np.ndarray, groups: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-(group, tie-group) positive / negative row counts of `score`, tie groups in score order.

    Returns (pos, neg), each (n_groups, n_ties): the sufficient statistics of the AUC under any
    frequency weighting of whole groups (see `_weighted_auc_draws`).
    """
    order = np.argsort(score, kind="stable")
    s = score[order]
    tie = np.concatenate([[0], np.cumsum(s[1:] > s[:-1])])
    n_ties = int(tie[-1]) + 1 if tie.size else 0
    g = groups[order]
    lab = label[order].astype(bool)
    flat = g * n_ties + tie
    size = n_groups * n_ties
    pos = np.bincount(flat[lab], minlength=size).reshape(n_groups, n_ties)
    neg = np.bincount(flat[~lab], minlength=size).reshape(n_groups, n_ties)
    return pos.astype("float64"), neg.astype("float64")


def _weighted_auc_draws(counts: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """AUC (ties count 1/2) of every bootstrap draw; counts (B, n_groups) = times each group is drawn.

    Exactly the AUC of the rows of the resampled groups stacked together (a group drawn c times
    contributes its rows c times). NaN when a draw has no positive or no negative row.
    """
    wp = counts @ pos  # (B, n_ties) positive weight per tie group
    wn = counts @ neg
    below = np.cumsum(wn, axis=1) - wn  # negative weight strictly below each tie group
    num = (wp * (below + 0.5 * wn)).sum(axis=1)
    den = wp.sum(axis=1) * wn.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


@dataclass
class DateBootstrap:
    """Date-block bootstrap of the CV gains over the baseline twin (whole test dates resampled).

    The 5% lower bounds (`auc_gain_p05`, `logloss_gain_p05`) are the bootstrap 5th percentiles
    with the usual small-cluster correction: the distance from the full-sample gain to the
    percentile is widened by c_G = sqrt(G/(G-1)) * t_{0.95, G-1} / z_{0.95} (G = test dates;
    CR1 variance factor and t(G-1) critical value; c_G -> 1 as G grows, 1.025 at G = 59). With
    ~60 dates the plain percentile passed pure-noise panels ~8% of the time at a nominal 5%.
    """

    auc_gain: np.ndarray  # (B,) auc(p) - auc(p_base) per draw (finite draws only)
    logloss_gain: np.ndarray  # (B,) logloss(p_base) - logloss(p) per draw (positive = better)
    n_dates: int
    n_boot: int
    auc_gain_hat: float = float("nan")  # full-sample (every date once) AUC gain
    logloss_gain_hat: float = float("nan")  # full-sample log-loss gain

    @property
    def small_cluster_factor(self) -> float:
        return small_cluster_factor(self.n_dates)

    @staticmethod
    def _q05(draws: np.ndarray) -> float:
        return float(np.quantile(draws, 0.05)) if draws.size else float("nan")

    def _bound(self, hat: float, draws: np.ndarray) -> float:
        q = self._q05(draws)
        if not (math.isfinite(q) and math.isfinite(hat)):
            return float("nan")
        return hat - self.small_cluster_factor * (hat - q)

    @property
    def auc_gain_q05(self) -> float:
        """Plain bootstrap 5th percentile of the AUC gain (no small-cluster correction)."""
        return self._q05(self.auc_gain)

    @property
    def logloss_gain_q05(self) -> float:
        return self._q05(self.logloss_gain)

    @property
    def auc_gain_p05(self) -> float:
        """5% lower bound of the AUC gain (small-cluster corrected) - the skill-gate quantity."""
        return self._bound(self.auc_gain_hat, self.auc_gain)

    @property
    def logloss_gain_p05(self) -> float:
        """5% lower bound of the log-loss gain (small-cluster corrected)."""
        return self._bound(self.logloss_gain_hat, self.logloss_gain)


def small_cluster_factor(g: int) -> float:
    """c_G = sqrt(G/(G-1)) * t_{0.95, G-1} / z_{0.95} (CR1 variance factor and t(G-1) critical value; NaN for G < 2)."""
    if g < 2:
        return float("nan")
    return math.sqrt(g / (g - 1.0)) * float(stats.t.ppf(0.95, g - 1)) / float(stats.norm.ppf(0.95))


@dataclass
class MeanBootstrap:
    """Date-block bootstrap of a per-row gain's mean (e.g. the CRPS gain base_crps - crps): whole dates resampled.

    p05 = the small-cluster corrected 5% lower bound hat - c_G (hat - q05), as `DateBootstrap`."""

    draws: np.ndarray
    n_dates: int
    n_boot: int
    hat: float = float("nan")

    @property
    def q05(self) -> float:
        return float(np.quantile(self.draws, 0.05)) if self.draws.size else float("nan")

    @property
    def p05(self) -> float:
        q = self.q05
        if not (math.isfinite(q) and math.isfinite(self.hat)):
            return float("nan")
        return self.hat - small_cluster_factor(self.n_dates) * (self.hat - q)


def date_block_mean_bootstrap(
    values: np.ndarray, dates: np.ndarray, n_boot: int, seed: int = 0, batch: int = 256
) -> MeanBootstrap:
    """Resample whole dates with replacement (`n_boot` draws) and recompute the mean of `values` over the rows of
    the drawn dates (rows of a date share the market move: ~one independent draw per date). Vectorised: a draw is
    a multinomial count vector over dates applied to per-date sums and row counts."""
    vals = np.asarray(values, dtype="float64")
    codes, uniq = pd.factorize(pd.Series(np.asarray(dates)))
    n_dates = int(uniq.size)
    if n_dates < 2 or int(n_boot) < 1 or vals.size == 0:
        return MeanBootstrap(np.empty(0), n_dates, int(n_boot), float(vals.mean()) if vals.size else float("nan"))
    sums = np.bincount(codes, vals, n_dates)
    rows = np.bincount(codes, minlength=n_dates).astype("float64")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, 0xC425]))
    out = []
    for start in range(0, int(n_boot), batch):
        b = min(batch, int(n_boot) - start)
        cb = rng.multinomial(n_dates, np.full(n_dates, 1.0 / n_dates), size=b).astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append((cb @ sums) / (cb @ rows))
    draws = np.concatenate(out)
    return MeanBootstrap(draws[np.isfinite(draws)], n_dates, int(n_boot), float(sums.sum() / rows.sum()))


def date_block_bootstrap(
    p: np.ndarray,
    p_base: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    n_boot: int,
    seed: int = 0,
    eps: float = 1e-12,
    batch: int = 128,
    counts: np.ndarray | None = None,
) -> DateBootstrap:
    """Resample whole CV test DATES with replacement (`n_boot` draws) and recompute the AUC gain and
    the log-loss gain of `p` over `p_base` on the stacked rows of the drawn dates.

    Why dates: every name on a date shares the market move, so 1{r > 0} carries about one
    independent draw per DATE, not per row. A row-level interval (or none) let pure-noise panels
    pass the skill gate. Vectorised exactly: a draw is a multinomial count vector over dates, the
    AUC comes from per-(date, tie-group) counts (`_weighted_auc_draws`), the log loss from per-date
    loss sums; `n_boot` = settings.rsmt.boot_sims.

    Args:
        counts: optional explicit draws (B, n_dates) - times each date (in order of first
            appearance in `dates`) is drawn; replaces the random draws (tests / paired analyses).
    """
    p = np.asarray(p, dtype="float64")
    p_base = np.asarray(p_base, dtype="float64")
    y = np.asarray(y, dtype="float64")
    codes, uniq = pd.factorize(pd.Series(np.asarray(dates)))
    n_dates = int(uniq.size)
    if counts is not None:
        counts = np.asarray(counts, dtype="float64")
        if counts.ndim != 2 or counts.shape[1] != n_dates:
            raise ValueError(f"counts must be (B, {n_dates}) date multiplicities, got {counts.shape}")
        n_boot = counts.shape[0]
    if n_dates < 2 or n_boot < 1:
        return DateBootstrap(np.empty(0), np.empty(0), n_dates, int(n_boot))
    pos, neg = _grouped_auc(p, y, codes, n_dates)
    pos_b, neg_b = _grouped_auc(p_base, y, codes, n_dates)

    def _loss(q: np.ndarray) -> np.ndarray:
        qc = np.clip(q, eps, 1.0 - eps)
        return -(y * np.log(qc) + (1.0 - y) * np.log(1.0 - qc))

    gain_sum = np.bincount(codes, _loss(p_base) - _loss(p), n_dates)
    rows = np.bincount(codes, minlength=n_dates).astype("float64")
    once = np.ones((1, n_dates))
    auc_hat = float((_weighted_auc_draws(once, pos, neg) - _weighted_auc_draws(once, pos_b, neg_b))[0])
    ll_hat = float(gain_sum.sum() / rows.sum())
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, 0xDA7E]))
    auc_g, ll_g = [], []
    for start in range(0, int(n_boot), batch):
        b = min(batch, int(n_boot) - start)
        if counts is None:
            cb = rng.multinomial(n_dates, np.full(n_dates, 1.0 / n_dates), size=b)
            cb = cb.astype("float64")
        else:
            cb = counts[start : start + b]
        auc_g.append(_weighted_auc_draws(cb, pos, neg) - _weighted_auc_draws(cb, pos_b, neg_b))
        with np.errstate(invalid="ignore", divide="ignore"):
            ll_g.append((cb @ gain_sum) / (cb @ rows))
    auc_gain = np.concatenate(auc_g)
    ll_gain = np.concatenate(ll_g)
    return DateBootstrap(
        auc_gain[np.isfinite(auc_gain)],
        ll_gain[np.isfinite(ll_gain)],
        n_dates,
        int(n_boot),
        auc_hat,
        ll_hat,
    )


def skill_gate(
    metrics: Mapping[str, float], min_auc_gain: float, require_date_ci: bool
) -> tuple[bool, dict[str, bool]]:
    """Date-aware CV skill gate (all quantities before any direction edge).

    skill_ok = (auc - base_auc >= min_auc_gain) and (logloss < base_logloss)
               and (not require_date_ci or auc_gain_p05 > 0)

    A missing / non-finite metric fails its component (never passes by default).

    Returns:
        (skill_ok, {"auc_gain", "logloss", "date_ci", "date_ci_required"} component flags).
    """

    def _get(key: str) -> float:
        v = metrics.get(key, float("nan"))
        return float(v) if v is not None else float("nan")

    gain = _get("auc") - _get("base_auc")
    ll, base_ll, lo = _get("logloss"), _get("base_logloss"), _get("auc_gain_p05")
    gate = {
        "auc_gain": bool(math.isfinite(gain) and gain >= float(min_auc_gain)),
        "logloss": bool(math.isfinite(ll) and math.isfinite(base_ll) and ll < base_ll),
        "date_ci": bool(math.isfinite(lo) and lo > 0.0),
        "date_ci_required": bool(require_date_ci),
    }
    ok = gate["auc_gain"] and gate["logloss"] and (gate["date_ci"] or not require_date_ci)
    return bool(ok), gate


def distribution_gate(
    metrics: Mapping[str, float],
    cover80_band: Sequence[float] = (0.75, 0.85),
    exc95_band: Sequence[float] = (0.02, 0.08),
) -> tuple[bool, dict[str, bool]]:
    """Distribution (risk-engine) skill gate of one layout (docs/RSMT.md):

    skill_ok = crps_gain_p05 > 0 (date-block bootstrap 5% bound of base_crps - crps over the baseline twin)
               and cover80 in [lo, hi] and exc95 in [lo, hi] (80% band coverage, VaR95 exceedance rate).

    A missing / non-finite metric fails its component. Returns (skill_ok, {"crps_ci", "cover80", "exc95"})."""

    def _get(key: str) -> float:
        v = metrics.get(key, float("nan"))
        return float(v) if v is not None else float("nan")

    def _inside(x: float, band: Sequence[float]) -> bool:
        return bool(math.isfinite(x) and float(band[0]) <= x <= float(band[1]))

    lo = _get("crps_gain_p05")
    gate = {
        "crps_ci": bool(math.isfinite(lo) and lo > 0.0),
        "cover80": _inside(_get("cover80"), cover80_band),
        "exc95": _inside(_get("exc95"), exc95_band),
    }
    return bool(all(gate.values())), gate


# the baseline twin of every CV / holdout comparison: root only, no offsets, raw target, fitted on the
# non-overlapping rows - identical for every model variant
BASE_TWIN: dict[str, Any] = dict(
    use_offsets=False,
    sigma_terms=(),
    max_depth=0,
    target_scale="none",
    gamma_mode="global",
    multiple_testing="bonferroni",
    row_multiplicity=1.0,
)


def _row_params(
    tree: RSMTree, frame: pd.DataFrame, moments: Any, horizon: int, with_offsets: bool
) -> dict[str, np.ndarray]:
    """Per-row distribution parameters of CV test rows (market leg from the fold's own moments)."""
    scale = frame["scale_h"].to_numpy(dtype="float64") if tree.scaled else None
    if with_offsets:
        p = tree.predict_params(frame, frame["z"].to_numpy(), frame["v"].to_numpy(), scale)
    else:
        p = tree.predict_params(frame, scale=scale)
    p["beta"] = frame["beta"].to_numpy(dtype="float64")
    p["mu_m"] = frame["bench"].map(moments.mu_by_bench).fillna(moments.mu).to_numpy(dtype="float64")
    p["nu_m"] = frame["bench"].map(moments.nu_by_bench).fillna(moments.nu).to_numpy(dtype="float64")
    p["sigma_m"] = frame["bench_vol"].to_numpy(dtype="float64") * math.sqrt(horizon / TRADING_DAYS)
    return p


def _evaluate(
    params: dict[str, np.ndarray],
    seeds: Sequence[int],
    r: np.ndarray,
    horizon: int,
    n_paths: int,
    mu_k: float = MU_PLAUSIBLE_K,
) -> dict[str, np.ndarray]:
    """Streamed per-row evaluation: one ReturnDistribution (the serving class) per test row, reduced
    to scalars immediately, so memory stays O(n_paths * horizon). The serving plausibility clamp
    |mu| <= mu_k * sigma_h is applied first (`clamped` = 1 where it bound; `mu_e` is the excess mean
    actually simulated, i.e. after the clamp). `exc95` = 1 where r fell below the 5% log quantile
    (VaR95 exceedance; the target rate is 5%)."""
    n = r.size
    out = {k: np.empty(n) for k in ("p_up", "p_out", "crps", "in80", "exc95", "mu", "mu_e", "clamped")}
    for i in range(n):
        beta_i = float(params["beta"][i])
        sd_i = total_log_sd(
            beta_i,
            float(params["sigma_e"][i]),
            float(params["sigma_m"][i]),
            float(params["rho"][i]),
            float(params["nu"][i]),
            float(params["nu_m"][i]),
            horizon,
        )
        clamp = plausibility_shift(beta_i * float(params["mu_m"][i]) + float(params["mu_e"][i]), sd_i, mu_k)
        dist = ReturnDistribution(
            params["mu_e"][i] + clamp,
            params["sigma_e"][i],
            params["mu_m"][i],
            params["sigma_m"][i],
            params["nu"][i],
            params["nu_m"][i],
            params["beta"][i],
            params["rho"][i],
            horizon,
            n_paths,
            seeds[i],
        )
        q05, q10, q90 = dist.quantile_log([0.05, 0.10, 0.90])
        out["p_up"][i] = dist.prob_up_raw
        out["p_out"][i] = dist.prob_outperform_raw
        out["crps"][i] = dist.crps(r[i])
        out["in80"][i] = float(q10 <= r[i] <= q90)
        out["exc95"][i] = float(r[i] < q05)
        out["mu"][i] = dist.mean_log
        out["mu_e"][i] = dist.mu_e
        out["clamped"][i] = float(clamp != 0.0)
    return out


def robust_fold_layouts(settings: Settings) -> list[int]:
    """The fold layouts of the robust skill gate: sorted unique rsmt.robust_cv_folds, or [cv_folds]
    when the list is empty (robustness switched off - `cross_validate` notes it).

    Raises:
        ValueError: a layout < 1 (a config error must not silently count as a failing layout).
    """
    return _clean_layouts(settings.rsmt.robust_cv_folds) or [int(settings.rsmt.cv_folds)]


def _clean_layouts(values: Sequence[int]) -> list[int]:
    ks = sorted({int(k) for k in values})
    bad = [k for k in ks if k < 1]
    if bad:
        raise ValueError(f"rsmt.robust_cv_folds must be >= 1, got {bad}")
    return ks


def layout_gate(result: CVResult, n_folds: int, seconds: float = 0.0) -> LayoutGate:
    """The single-layout verdict of ONE fold layout's CV (`result.skill_ok` read as its own gate)."""
    m = result.metrics
    nan = float("nan")
    return LayoutGate(
        n_folds=int(n_folds),
        skill_ok=bool(result.available and result.skill_ok),
        available=result.available,
        gate=dict(result.gate),
        auc_gain=float(m.get("auc_gain", nan)),
        auc_gain_p05=float(m.get("auc_gain_p05", nan)),
        logloss_gain=float(m.get("logloss_gain", nan)),
        n_oos=int(m.get("n_oos", 0)),
        n_oos_dates=int(m.get("n_oos_dates", 0)),
        seconds=round(float(seconds), 3),
        notes=list(result.notes),
        direction_ok=bool(result.available and result.direction_ok),
        distribution_ok=bool(result.available and result.distribution_ok),
        dist_gate=dict(result.dist_gate),
        crps_gain=float(m.get("crps_gain", nan)),
        crps_gain_p05=float(m.get("crps_gain_p05", nan)),
        cover80=float(m.get("cover80", nan)),
        exc95=float(m.get("exc95", nan)),
    )


def apply_robust_gate(result: CVResult, layouts: Mapping[int, LayoutGate], *, lang: str = "ko") -> CVResult:
    """Set the fold-robust verdict on the primary-layout `result` (in place; also returned). Apply it
    ONCE to a fresh single-layout result (its `skill_ok` is read as the primary layout's gate).

    skill_ok = primary CV available and the single-layout gate (of rsmt.skill_metric) passes for a strict
    majority of `layouts`; skill_ok_primary keeps the primary layout's own gate. direction_ok /
    distribution_ok become the same majority verdicts of the direction and the distribution gates. The
    per-layout verdicts go to cv_metrics - gate_folds_<k> = 1/0 per k-fold layout (the skill_metric gate),
    gate_folds_n (layouts), gate_folds_pass (passes), gate_folds_need (strict majority n//2 + 1), the keys
    report/render.py reads for the '모델 신뢰도' line - plus direction_gate_folds_<k> / distribution_gate_folds_<k>
    (+ _pass), direction_gate / distribution_gate (1/0 majority verdicts), skill_metric_distribution (1/0),
    auc_gain_folds_<k>, auc_gain_p05_folds_<k>, crps_gain_folds_<k>, crps_gain_p05_folds_<k>, skill_ok_primary
    and skill_ok_robust (only when the primary CV produced metrics, so an unavailable CV stays unavailable). A
    verdict that differs from the primary layout's gate is noted (in `lang`: "ko" / "en").
    """
    result.skill_ok_primary = bool(result.available and result.skill_ok)
    result.layouts = {int(k): lay for k, lay in sorted(layouts.items())}
    result.skill_ok = bool(result.available and result.robust_majority)
    n_lay = result.robust_n
    result.direction_ok = bool(result.available and n_lay > 0 and 2 * result.robust_pass_of("direction") > n_lay)
    result.distribution_ok = bool(result.available and n_lay > 0 and 2 * result.robust_pass_of("distribution") > n_lay)
    if result.metrics:
        m = result.metrics
        for k, lay in result.layouts.items():
            m[f"gate_folds_{k}"] = 1.0 if lay.skill_ok else 0.0
            m[f"auc_gain_folds_{k}"] = lay.auc_gain
            m[f"auc_gain_p05_folds_{k}"] = lay.auc_gain_p05
            m[f"crps_gain_folds_{k}"] = lay.crps_gain
            m[f"crps_gain_p05_folds_{k}"] = lay.crps_gain_p05
            m[f"direction_gate_folds_{k}"] = 1.0 if lay.direction_ok else 0.0
            m[f"distribution_gate_folds_{k}"] = 1.0 if lay.distribution_ok else 0.0
        m["gate_folds_n"] = float(result.robust_n)
        m["gate_folds_pass"] = float(result.robust_pass)
        m["gate_folds_need"] = float(result.robust_n // 2 + 1)
        m["direction_gate_folds_pass"] = float(result.robust_pass_of("direction"))
        m["distribution_gate_folds_pass"] = float(result.robust_pass_of("distribution"))
        m["direction_gate"] = 1.0 if result.direction_ok else 0.0
        m["distribution_gate"] = 1.0 if result.distribution_ok else 0.0
        m["skill_metric_distribution"] = 1.0 if result.skill_metric == "distribution" else 0.0
        m["skill_ok_primary"] = 1.0 if result.skill_ok_primary else 0.0
        m["skill_ok_robust"] = 1.0 if result.skill_ok else 0.0
    for k, lay in result.layouts.items():
        if not lay.available and k != result.n_folds_primary:
            why = "; ".join(lay.notes) or tr(lang, "사유 미기재", "reason not given")
            result.notes.append(
                tr(
                    lang,
                    f"{k}폴드 배치 CV 불가(게이트 미달 처리): {why}",
                    f"{k}-fold layout CV unavailable (counted as not passing the gate): {why}",
                )
            )
    if result.available and result.skill_ok_primary != result.skill_ok:
        k0, rp, rn = result.n_folds_primary, result.robust_pass, result.robust_n
        if lang_of(lang) == "en":
            verdict = "passes" if result.skill_ok_primary else "does not pass"
            skill = "recognised" if result.skill_ok else "not recognised"
            note = (
                f"the primary {k0}-fold layout {verdict} the gate but the fold-layout majority is {rp}/{rn} → "
                f"skill {skill}"
            )
        else:
            verdict = "통과" if result.skill_ok_primary else "미달"
            note = (
                f"기준 {k0}폴드 배치는 게이트 {verdict}이나 폴드 배치 과반 판정 "
                f"{rp}/{rn} → 스킬 {'인정' if result.skill_ok else '불인정'}"
            )
        result.notes.append(note)
    return result


def cross_validate(
    panel: PanelData,
    settings: Settings,
    n_paths: int = CV_PATHS,
    *,
    robust_folds: Sequence[int] | None = None,
) -> CVResult:
    """Expanding-window CV (purged) of the full RSMT-Q vs a root-only, no-offset baseline, repeated
    for every fold layout of the fold-robust skill gate.

    Metrics (pooled OOS rows of the primary layout rsmt.cv_folds): auc / logloss / brier of P(up) vs
    1{r>0}, sample CRPS of the terminal log-return draws, cover80 (r inside [q10, q90]),
    ic = Spearman(mu, r); `base_*` twins; the direction edges delta (on 1{r>0}) and delta_rel (on
    1{r>m}). Every layout uses the serving rules (market drift prior, plausibility clamp).

    Date-aware single-layout gate (all before any edge):
        gate = auc - base_auc >= rsmt.skill_min_auc_gain
               and logloss < base_logloss
               and (not rsmt.skill_require_date_ci or auc_gain_p05 > 0)
    where auc_gain_p05 / logloss_gain_p05 are the 5% lower bounds of the gains from a bootstrap
    that resamples whole CV test DATES (`date_block_bootstrap`, rsmt.boot_sims draws; small-cluster
    corrected, the plain percentiles are reported as *_q05): every name on a date shares the market
    move, so the pooled rows are ~one independent draw per date.

    Fold-robust verdict: skill_ok = the gate passes for a strict MAJORITY of the layouts
    `robust_folds` (default `robust_fold_layouts(settings)` = rsmt.robust_cv_folds); the primary
    layout's own run is reused when it is one of them (see `apply_robust_gate`); the layouts run
    concurrently (`_run_layouts`).

    Measured 2026-09-25 (synthetic 40 names x 1800 sessions; layouts 3/4/5; single = 4-fold gate):
    pure random walk single 3/40, robust 2/40 (5.0%); common residual factor per date (30 x 1300)
    single 1/20, robust 2/20; planted regimes single 17/20, robust 18/20 (power 90%). The layouts
    share most test rows, so on clear-cut panels their verdicts agree; the majority rule matters on
    marginal cases such as live KR 2026-09-24 (only the 4-fold layout passed, bound +0.001 -> no
    skill). Wall time: live KR/US panels ~5 s per layout, 3 layouts 10-16 s (was 4 s for one).
    """
    t0 = time.perf_counter()
    primary_k = int(settings.rsmt.cv_folds)
    ks = _clean_layouts(settings.rsmt.robust_cv_folds if robust_folds is None else robust_folds)
    empty = not ks
    if empty:
        ks = [primary_k]
    runs = _run_layouts(panel, settings, [primary_k, *(k for k in ks if k != primary_k)], n_paths)
    result = runs[primary_k][0]
    result.n_folds_primary = primary_k
    layouts = {k: layout_gate(runs[k][0], k, runs[k][1]) for k in ks}
    if empty:
        result.notes.append(
            tr(
                settings,
                f"rsmt.robust_cv_folds 비어 있음 → 기준 {primary_k}폴드 단일 배치 게이트",
                f"rsmt.robust_cv_folds is empty → single-layout gate on the primary {primary_k} folds",
            )
        )
    apply_robust_gate(result, layouts, lang=lang_of(settings))
    result.seconds = round(time.perf_counter() - t0, 3)
    log.info(
        "RSMT-Q %s CV: skill_ok=%s (layouts %s pass %d/%d, primary %d folds %s) in %.1fs",
        panel.market.value,
        result.skill_ok,
        {k: lay.skill_ok for k, lay in layouts.items()},
        result.robust_pass,
        result.robust_n,
        primary_k,
        result.skill_ok_primary,
        result.seconds,
    )
    return result


def cross_validate_layouts(
    panel: PanelData, settings: Settings, layouts: Sequence[int], n_paths: int = CV_PATHS
) -> dict[int, CVResult]:
    """Full single-layout CV of every fold layout in `layouts` ({n_folds: CVResult}), run concurrently.

    Each result is ONE expanding-window CV (its own metrics, gate, edges, per-fold info and OOS frame);
    `skill_ok` is that layout's single-layout gate, `n_folds_primary` its fold count and `seconds` its
    wall time. No robust verdict is applied: combine them with `layout_gate` + `apply_robust_gate` on a
    copy (as `cross_validate` does) when the fold-robust verdict is needed.

    Raises:
        ValueError: a layout < 1.
    """
    ks = _clean_layouts(layouts)
    out: dict[int, CVResult] = {}
    for k, (res, secs) in _run_layouts(panel, settings, ks, n_paths).items():
        res.n_folds_primary = k
        res.seconds = round(float(secs), 3)
        out[k] = res
    return out


def _run_layouts(
    panel: PanelData, settings: Settings, ks: Sequence[int], n_paths: int
) -> dict[int, tuple[CVResult, float]]:
    """{n_folds: (CVResult, seconds)} of every layout, run concurrently (CV_LAYOUT_WORKERS threads).

    Each layout is deterministic on its own (seeded fits and paths), so the results do not depend on
    the scheduling. An exception in any layout propagates (never swallowed).
    """

    def one(k: int) -> tuple[CVResult, float]:
        t = time.perf_counter()
        res = _cross_validate_layout(panel, settings, k, n_paths)
        return res, time.perf_counter() - t

    ks = list(dict.fromkeys(int(k) for k in ks))
    workers = max(1, min(CV_LAYOUT_WORKERS, len(ks)))
    if workers == 1:
        return {k: one(k) for k in ks}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rsmt-cv") as pool:
        futures = {k: pool.submit(one, k) for k in ks}
        return {k: fut.result() for k, fut in futures.items()}


def _fold_forecasts(
    panel: PanelData, settings: Settings, trn: pd.DataFrame, te: pd.DataFrame, n_paths: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], RSMTree]:
    """Fit the model on `trn` (every row, overlapping ones included) and the baseline twin (`BASE_TWIN`, on the
    non-overlapping rows) and evaluate both on the test rows `te` with the serving rules.

    Returns (model evaluation, twin evaluation, model row parameters, twin row parameters, fitted model)."""
    h = panel.horizon
    model = RSMTree.from_settings(settings, panel.market).fit_frame(trn)
    base = RSMTree.from_settings(settings, panel.market, **BASE_TWIN).fit_frame(on_grid_rows(trn))
    moments = moments_from_settings(trn, settings, panel.market, h)
    seeds = [path_seed(t, pd.Timestamp(d).date().isoformat()) for t, d in zip(te["ticker"], te["date"])]
    r = te["r"].to_numpy(dtype="float64")
    prm = _row_params(model, te, moments, h, True)
    prm_b = _row_params(base, te, moments, h, False)
    return _evaluate(prm, seeds, r, h, n_paths), _evaluate(prm_b, seeds, r, h, n_paths), prm, prm_b, model


def _pool_rows(
    pooled: dict[str, list[np.ndarray]],
    ev: dict[str, np.ndarray],
    evb: dict[str, np.ndarray],
    prm: dict[str, np.ndarray],
    prm_b: dict[str, np.ndarray],
    te: pd.DataFrame,
    fold: int,
) -> None:
    for key, arr in (*ev.items(), *(("base_" + k, a) for k, a in evb.items())):
        pooled.setdefault(key, []).append(arr)
    # the simulated distribution of every test row (audit / evaluation: barrier replays, leaf groups)
    for key in ("mu_leaf", "sigma_e", "nu", "rho", "leaf_id"):
        pooled.setdefault(key, []).append(np.asarray(prm[key]))
        pooled.setdefault("base_" + key, []).append(np.asarray(prm_b[key]))
    for key in ("beta", "mu_m", "sigma_m", "nu_m"):
        pooled.setdefault(key, []).append(np.asarray(prm[key], dtype="float64"))
    pooled.setdefault("r", []).append(te["r"].to_numpy(dtype="float64"))
    pooled.setdefault("m", []).append(te["m"].to_numpy(dtype="float64"))
    pooled.setdefault("date", []).append(te["date"].to_numpy())
    pooled.setdefault("ticker", []).append(te["ticker"].to_numpy())
    pooled.setdefault("fold", []).append(np.full(len(te), fold))


def _pooled_metrics(P: Mapping[str, np.ndarray], settings: Settings, n_paths: int) -> tuple[dict[str, float], Any]:
    """CV / holdout metrics of pooled OOS rows (model and base_ twin) + the direction date bootstrap."""
    up = (P["r"] > 0).astype(float)
    beat = (P["r"] > P["m"]).astype(float)
    eps = 0.5 / n_paths
    metrics: dict[str, float] = {
        "n_oos": float(P["r"].size),
        "n_mu_clamped": float(P["clamped"].sum()),
        "base_n_mu_clamped": float(P["base_clamped"].sum()),
    }
    for prefix in ("", "base_"):
        p = P[prefix + "p_up"]
        metrics[prefix + "auc"] = _auc(p, up)
        metrics[prefix + "logloss"] = _logloss(p, up, eps)
        metrics[prefix + "brier"] = float(np.mean((p - up) ** 2))
        metrics[prefix + "crps"] = float(np.mean(P[prefix + "crps"]))
        metrics[prefix + "cover80"] = float(np.mean(P[prefix + "in80"]))
        metrics[prefix + "exc95"] = float(np.mean(P[prefix + "exc95"]))
        ic = stats.spearmanr(P[prefix + "mu"], P["r"]).statistic if np.ptp(P[prefix + "mu"]) > 0 else float("nan")
        metrics[prefix + "ic"] = float(ic)
    delta = _fit_edge(P["p_up"], up, eps)
    delta_rel = _fit_edge(P["p_out"], beat, eps)
    metrics["delta"] = delta
    metrics["delta_rel"] = delta_rel
    metrics["logloss_edge"] = _logloss(np.array([apply_edge(p, delta, n_paths) for p in P["p_up"]]), up, eps)
    # relative AUC (diagnostic): all names share the market move, the excess-return label is the
    # cross-sectional part of the skill
    metrics["auc_rel"] = _auc(P["p_out"], beat)
    metrics["base_auc_rel"] = _auc(P["base_p_out"], beat)
    rcfg = settings.rsmt
    boot = date_block_bootstrap(P["p_up"], P["base_p_up"], up, P["date"], rcfg.boot_sims, seed=0, eps=eps)
    metrics["n_oos_dates"] = float(boot.n_dates)
    metrics["n_boot"] = float(boot.n_boot)
    metrics["auc_gain"] = metrics["auc"] - metrics["base_auc"]
    metrics["logloss_gain"] = metrics["base_logloss"] - metrics["logloss"]
    metrics["auc_gain_p05"] = boot.auc_gain_p05  # small-cluster corrected 5% bound (gate)
    metrics["logloss_gain_p05"] = boot.logloss_gain_p05
    metrics["auc_gain_q05"] = boot.auc_gain_q05  # plain bootstrap percentile (reference)
    metrics["logloss_gain_q05"] = boot.logloss_gain_q05
    metrics["skill_min_auc_gain"] = float(rcfg.skill_min_auc_gain)
    # distribution skill: CRPS gain over the twin (positive = better), bootstrap over whole test dates
    cb = date_block_mean_bootstrap(P["base_crps"] - P["crps"], P["date"], rcfg.boot_sims, seed=0)
    metrics["crps_gain"] = metrics["base_crps"] - metrics["crps"]
    metrics["crps_gain_rel"] = metrics["crps_gain"] / metrics["base_crps"] if metrics["base_crps"] else float("nan")
    metrics["crps_gain_p05"] = cb.p05
    metrics["crps_gain_q05"] = cb.q05
    return metrics, boot


def _cross_validate_layout(panel: PanelData, settings: Settings, n_folds: int, n_paths: int) -> CVResult:
    """ONE expanding-window CV layout with `n_folds` folds; `skill_ok` is its single-layout gate of
    rsmt.skill_metric (both the direction and the distribution gates are computed)."""
    mcfg = market_config(settings, panel.market)
    train = panel.train
    rcfg = settings.rsmt
    result = CVResult(skill_metric=str(rcfg.skill_metric))
    try:
        folds = cv_splits(train, n_folds)
    except ValueError as exc:
        result.notes.append(tr(settings, f"CV 불가: {exc}", f"CV unavailable: {exc}"))
        log.warning("RSMT-Q %s CV unavailable: %s", panel.market.value, exc)
        return result

    pooled: dict[str, list[np.ndarray]] = {}
    min_train = max(2 * mcfg.min_leaf, 30)
    for fold in folds:
        trn = train.iloc[fold.train_idx]
        te = train.iloc[fold.test_idx]
        info: dict[str, Any] = {
            "fold": fold.fold,
            "n_train": len(trn),
            "n_test": len(te),
            "n_purged": fold.n_purged,
            "test_start": str(fold.test_start.date()),
        }
        if len(te) == 0 or len(on_grid_rows(trn)) < min_train:
            info["skipped"] = tr(settings, "학습/검증 표본 부족", "too few train/test rows")
            result.folds.append(info)
            result.notes.append(
                tr(
                    settings,
                    f"CV fold {fold.fold} 생략: 학습 {len(trn)}행 / 검증 {len(te)}행",
                    f"CV fold {fold.fold} skipped: train {len(trn)} rows / test {len(te)} rows",
                )
            )
            continue
        ev, evb, prm, prm_b, model = _fold_forecasts(panel, settings, trn, te, n_paths)
        info["n_splits"] = model.n_splits
        info["leaf_paths"] = {int(lf.leaf_id): " / ".join(lf.path) for lf in model.leaves}
        result.folds.append(info)
        _pool_rows(pooled, ev, evb, prm, prm_b, te, fold.fold)

    if not pooled:
        result.notes.append(
            tr(settings, "CV 불가: 평가 가능한 fold 없음", "CV unavailable: no fold could be evaluated")
        )
        return result
    P = {k: np.concatenate(v) for k, v in pooled.items()}
    result.oos = pd.DataFrame({k: P[k] for k in OOS_COLUMNS})
    metrics, boot = _pooled_metrics(P, settings, n_paths)
    metrics["n_folds"] = float(sum("skipped" not in f for f in result.folds))
    result.direction_ok, result.gate = skill_gate(metrics, rcfg.skill_min_auc_gain, rcfg.skill_require_date_ci)
    result.distribution_ok, result.dist_gate = distribution_gate(
        metrics, rcfg.skill_cover80_band, rcfg.skill_exc95_band
    )
    result.skill_ok = result.distribution_ok if result.skill_metric == "distribution" else result.direction_ok
    if rcfg.skill_require_date_ci and not math.isfinite(metrics["auc_gain_p05"]):
        result.notes.append(
            tr(
                settings,
                f"날짜 부트스트랩 불가(OOS 날짜 {boot.n_dates}개) — 방향 스킬 게이트 미달 처리",
                f"date bootstrap unavailable ({boot.n_dates} OOS dates) — counted as not passing the direction skill "
                "gate",
            )
        )
    result.metrics = metrics
    result.direction_edge = metrics["delta"]
    result.relative_edge = metrics["delta_rel"]
    return result


def evaluate_split(
    panel: PanelData,
    settings: Settings,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    n_paths: int = CV_PATHS,
) -> tuple[dict[str, float], RSMTree, pd.DataFrame]:
    """One train / test evaluation with the CV machinery (e.g. a final holdout: fit on every pre-holdout row whose
    label ended before the holdout, evaluate once on the holdout's on_grid rows).

    Returns (metrics incl. the direction / distribution gate components as 1/0, the fitted model, OOS rows).

    Raises:
        ValueError: no training or no test row.
    """
    train = panel.train
    trn = train[np.asarray(train_mask, dtype=bool)]
    te = on_grid_rows(train[np.asarray(test_mask, dtype=bool)])
    if trn.empty or te.empty:
        raise ValueError(f"evaluate_split: {len(trn)} training / {len(te)} test rows")
    ev, evb, prm, prm_b, model = _fold_forecasts(panel, settings, trn, te, n_paths)
    pooled: dict[str, list[np.ndarray]] = {}
    _pool_rows(pooled, ev, evb, prm, prm_b, te, 0)
    P = {k: np.concatenate(v) for k, v in pooled.items()}
    metrics, _ = _pooled_metrics(P, settings, n_paths)
    rcfg = settings.rsmt
    d_ok, d_gate = skill_gate(metrics, rcfg.skill_min_auc_gain, rcfg.skill_require_date_ci)
    s_ok, s_gate = distribution_gate(metrics, rcfg.skill_cover80_band, rcfg.skill_exc95_band)
    metrics["direction_gate"] = float(d_ok)
    metrics["distribution_gate"] = float(s_ok)
    metrics.update({f"gate_{k}": float(v) for k, v in {**d_gate, **s_gate}.items()})
    metrics["n_splits"] = float(model.n_splits)
    return metrics, model, pd.DataFrame({k: P[k] for k in OOS_COLUMNS})


# ======================================================================== market model


@dataclass
class ExcessOverride:
    """The selected stock-selection model's excess mean for one served ticker (quantcio.modeling.serve).

    The served distribution keeps everything RSMT-Q v1 estimates for risk (sigma_e with the eta offset, nu, rho,
    beta, the market leg's sigma / nu), but its beta-adjusted excess MEAN is ``mu_excess`` (not the RSMT-Q leaf mean).
    ``market_p_up`` (only when the market-direction model passed its rule) moves the market leg's mean so that
    P(m > 0) matches it under a normal approximation; otherwise the prior-shrunk drift stays. ``skill_ok`` is the
    modeling serving gate and becomes DistributionForecast.skill_ok.
    """

    mu_excess: float
    skill_ok: bool
    model: str
    market_p_up: float | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)


@dataclass
class _Served:
    mu_e: float
    sigma_e: float
    offsets_applied: bool
    degraded: bool
    reasons: list[str]


@dataclass
class _Prepared:
    """Everything needed to serve one ticker except the model-level (batch-wide) reasons."""

    ticker: str
    row: pd.Series
    i: int
    reasons: list[str]  # row-level reasons (panel inputs, benchmark fallbacks)
    beta: float
    mu_m: float
    nu_m: float
    sigma_m: float
    mu_e_rsmt: float
    sigma_e: float
    shift: float
    blend: dict[str, float]
    leaf_id: int
    offsets_applied: bool
    mu_cap: float = float("nan")  # MU_PLAUSIBLE_K * sigma_h
    mu_clamp: float = 0.0  # plausibility shift applied to the served total mean (0 = not binding)
    mu_unclamped: float = float("nan")  # served total mean before the clamp
    override: ExcessOverride | None = None  # selected stock-selection model's excess mean (replaces RSMT-Q's)
    mu_m_rsmt: float = float("nan")  # the prior-shrunk market drift before a market-model override


class MarketModel:
    """Fitted RSMT-Q for one market: serves one ReturnDistribution per ticker, with health checks.

    Serving: `predict_many` (preferred - the whole market batch; health checks over the batch run
    BEFORE anything is returned, so every forecast is final) or `predict` (one ticker; a collapse
    detected later flags forecasts already returned in place). The distribution is simulated over
    `hold_days` (default settings.fusion.target_horizon_days) so the trade barrier covers the
    holding horizon; every forecast quantity is still the model horizon h.

    Every user-facing text (health / diagnostics lines, degraded reasons, errors) is in the report language
    settings.report.language at construction (`lang`, "ko" / "en"); the pipeline's model cache key includes it.
    """

    lang: str = "ko"  # class default: models pickled before the attribute existed stay Korean

    def __init__(self, panel: PanelData, tree: RSMTree, cv: CVResult, settings: Settings):
        self.panel = panel
        self.tree = tree
        self.cv = cv
        self.lang = lang_of(settings)
        self.market = panel.market
        self.horizon = panel.horizon
        self.as_of = panel.as_of
        self.hold_days = max(1, int(settings.fusion.target_horizon_days))
        self.skill_min_auc_gain = float(settings.rsmt.skill_min_auc_gain)
        self.n_paths = int(settings.rsmt.n_paths)
        self.feature_set = settings.rsmt.feature_set
        self.collapse_min_unique_ratio = float(settings.rsmt.collapse_min_unique_ratio)
        self.cover80_band = tuple(float(x) for x in settings.rsmt.skill_cover80_band)
        self.exc95_band = tuple(float(x) for x in settings.rsmt.skill_exc95_band)
        self.mu_plausible_k = MU_PLAUSIBLE_K
        self._pred = panel.pred.set_index("ticker", drop=False)
        self._universe = tree.predict_params(
            self._pred, self._pred["z"].to_numpy(), self._pred["v"].to_numpy(), self._pred_scale()
        )
        self._served: dict[str, _Served] = {}
        self._returned: list[tuple[str, DistributionForecast]] = []
        self._clamped: dict[str, tuple[float, float]] = {}  # served ticker -> (unclamped mu, served mu)
        self._model_reasons = self._model_level_checks()
        for reason in self._model_reasons:
            log.warning("RSMT-Q %s degraded: %s", self.market.value, reason)

    # ------------------------------------------------------------------ properties

    def _t(self, ko: str, en: str) -> str:
        """The Korean or English text for this model's report language."""
        return tr(self.lang, ko, en)

    def _pred_scale(self) -> np.ndarray | None:
        """Target scale of the prediction rows (target_scale = "ewma"): the panel's scale_h, or - for a panel
        built before the column existed - exp(v + c_v) * sqrt(h/252) (v missing was set to 0 and flagged)."""
        if getattr(self.tree, "target_scale", "none") == "none":
            return None
        if "scale_h" in self._pred:
            return self._pred["scale_h"].to_numpy(dtype="float64")
        v = self._pred["v"].to_numpy(dtype="float64")
        return np.exp(v + self.panel.c_v) * math.sqrt(self.horizon / TRADING_DAYS)

    @property
    def gamma(self) -> float:
        """The global momentum offset (gamma_mode = "leaf": the root's; each leaf's is in the tree / offsets)."""
        return self.tree.gamma

    @property
    def eta(self) -> float:
        """The volatility elasticity of the raw-return model (target_scale = "ewma": 1 + the residual eta)."""
        return self.tree.eta_equivalent

    @property
    def skill_metric(self) -> str:
        return str(getattr(self.cv, "skill_metric", "direction"))

    @property
    def direction_skill_ok(self) -> bool:
        """The fold-robust DIRECTION (AUC / log-loss) verdict - reported with every forecast as
        DistributionForecast.direction_skill_ok when RSMT-Q's own mean is served (the directional ML-pillar
        multiplier reads it; with skill_metric = "distribution" `skill_ok` certifies the distribution instead)."""
        return bool(getattr(self.cv, "direction_ok", self.cv.skill_ok))

    @property
    def direction_edge(self) -> float:
        return self.cv.direction_edge

    @property
    def relative_edge(self) -> float:
        return self.cv.relative_edge

    @property
    def skill_ok(self) -> bool:
        """The FOLD-ROBUST CV verdict of rsmt.skill_metric (majority of rsmt.robust_cv_folds layouts; see
        `apply_robust_gate`): the distribution gate (CRPS gain bound, 80% coverage, VaR95 exceedance) by default,
        the AUC gate with skill_metric = "direction". Every served DistributionForecast.skill_ok carries it (also
        under an ExcessOverride: it certifies RSMT-Q's distribution); the directional ML-pillar multiplier reads
        DistributionForecast.direction_skill_ok instead."""
        return bool(self.cv.skill_ok)

    @property
    def degraded_reasons(self) -> list[str]:
        """Model-level reasons (applied to every served row)."""
        reasons = list(self._model_reasons)
        served_collapse = self._served_collapse_reason()
        if served_collapse and served_collapse not in reasons:
            reasons.append(served_collapse)
        return reasons

    @property
    def degraded(self) -> bool:
        return bool(self.degraded_reasons)

    # ------------------------------------------------------------------ checks

    def _collapse_reason(self, mu: np.ndarray, sigma: np.ndarray, scope: str) -> str | None:
        n = mu.size
        if n < 4:
            return None
        pairs = {(round(float(a), 5), round(float(b), 5)) for a, b in zip(mu, sigma)}
        ratio = len(pairs) / n
        if ratio < self.collapse_min_unique_ratio:
            lim = self.collapse_min_unique_ratio
            return self._t(
                f"예측 붕괴({scope}): 고유 (μ,σ) {len(pairs)}/{n} = {ratio:.2f} < {lim:.2f} — 오프셋/분할 미작동 의심",
                f"forecast collapse ({scope}): unique (μ,σ) {len(pairs)}/{n} = {ratio:.2f} < {lim:.2f} — offsets / "
                "splits suspected inactive",
            )
        return None

    def _model_level_checks(self) -> list[str]:
        reasons: list[str] = []
        scope = self._t("유니버스 최종일", "universe, last date")
        collapse = self._collapse_reason(self._universe["mu_e"], self._universe["sigma_e"], scope)
        if collapse:
            reasons.append(collapse)
        if self.panel.failed_ratio > FAILED_PANEL_MAX:
            nf, nt, fr = len(self.panel.failed_tickers), self.panel.n_tickers, self.panel.failed_ratio
            reasons.append(
                self._t(
                    f"패널 로드 실패 {nf}/{nt} ({fr:.0%}) > {FAILED_PANEL_MAX:.0%}",
                    f"panel load failed {nf}/{nt} ({fr:.0%}) > {FAILED_PANEL_MAX:.0%}",
                )
            )
        return reasons

    def _served_collapse_reason(self) -> str | None:
        if len(self._served) < 4:
            return None
        mu = np.array([s.mu_e for s in self._served.values()])
        sig = np.array([s.sigma_e for s in self._served.values()])
        return self._collapse_reason(mu, sig, self._t(SERVED_SCOPE, SERVED_SCOPE_EN))

    # ------------------------------------------------------------------ serving

    def _hold(self, hold_days: int | None) -> int:
        hold = self.hold_days if hold_days is None else int(hold_days)
        if hold < 1:
            raise ValueError(f"hold_days must be >= 1, got {hold_days}")
        return hold

    def _prepare(self, ticker: str, override: ExcessOverride | None = None) -> _Prepared:
        """Leaf params + offsets and market leg of one ticker; the served mean is RSMT-Q's own (100%) or the selected
        stock-selection model's excess mean when ``override`` is given (see ExcessOverride).

        Raises:
            KeyError: the ticker is not in the panel (e.g. failed to load).
            ValueError: no benchmark volatility anywhere in the universe.
        """
        t = normalise_ticker(ticker)
        if t not in self._pred.index:
            why = self.panel.failed.get(t, self._t("패널에 없음", "not in the panel"))
            raise KeyError(
                self._t(
                    f"{t}: RSMT-Q {self.market.value} 모델에 예측 행 없음 ({why})",
                    f"{t}: no prediction row in the RSMT-Q {self.market.value} model ({why})",
                )
            )
        i = int(self._pred.index.get_loc(t))
        row = self._pred.iloc[i]
        U = self._universe
        reasons = list(row["degraded_reasons"])

        beta = float(row["beta"])
        mu_m, nu_m, sigma_m, leg_reasons = self._market_leg(row, t)
        reasons.extend(leg_reasons)

        mu_e_rsmt = float(U["mu_e"][i])
        sigma_e = float(U["sigma_e"][i])
        mu_m_rsmt = mu_m
        if override is not None and override.market_p_up is not None and 0.0 < float(override.market_p_up) < 1.0:
            mu_m = float(sigma_m * stats.norm.ppf(float(override.market_p_up)))
        mu_total_rsmt = beta * mu_m + mu_e_rsmt
        if override is not None and math.isfinite(float(override.mu_excess)):
            target = beta * mu_m + float(override.mu_excess)
            blend = {str(override.model): 1.0}
        else:
            target = mu_total_rsmt
            blend = {"rsmt": 1.0}
        # plausibility clamp on the served total mean: |mu| <= k * sigma_h
        sd_total = total_log_sd(beta, sigma_e, sigma_m, float(U["rho"][i]), float(U["nu"][i]), nu_m, self.horizon)
        clamp = plausibility_shift(target, sd_total, self.mu_plausible_k)
        shift = target + clamp - mu_total_rsmt
        return _Prepared(
            ticker=t,
            row=row,
            i=i,
            reasons=reasons,
            beta=beta,
            mu_m=mu_m,
            nu_m=nu_m,
            sigma_m=sigma_m,
            mu_e_rsmt=mu_e_rsmt,
            sigma_e=sigma_e,
            shift=shift,
            blend=blend,
            leaf_id=int(U["leaf_id"][i]),
            offsets_applied=not any(is_offset_missing(r) for r in reasons),
            mu_cap=self.mu_plausible_k * sd_total,
            mu_clamp=clamp,
            mu_unclamped=target,
            override=override if override is not None and math.isfinite(float(override.mu_excess)) else None,
            mu_m_rsmt=mu_m_rsmt,
        )

    def _market_leg(self, row: pd.Series, ticker: str) -> tuple[float, float, float, list[str]]:
        """(mu_m, nu_m, sigma_m, reasons) of one prediction row's benchmark leg.

        Raises:
            ValueError: no benchmark volatility anywhere in the universe.
        """
        reasons: list[str] = []
        bench = str(row["bench"])
        mu_m = float(self.panel.market_mu_by_bench.get(bench, self.panel.market_mu_hist))
        nu_m = float(self.panel.market_nu_by_bench.get(bench, self.panel.market_nu))
        if bench not in self.panel.market_mu_by_bench:
            reasons.append(
                self._t(
                    f"벤치마크({bench or '없음'}) 학습 표본 없음 → 통합 시장 μ/ν 사용",
                    f"no training sample for benchmark {bench or '(none)'} → pooled market μ/ν used",
                )
            )
        bench_vol = float(row["bench_vol"])
        if not math.isfinite(bench_vol):
            vols = [v for v in self.panel.benchmark_ewma.values() if math.isfinite(v)]
            if not vols:
                raise ValueError(
                    self._t(
                        f"{ticker}: 벤치마크 변동성을 구할 수 없음 (유니버스 전체 결측)",
                        f"{ticker}: benchmark volatility unavailable (absent for the whole universe)",
                    )
                )
            bench_vol = float(np.median(vols))
        return mu_m, nu_m, bench_vol * math.sqrt(self.horizon / TRADING_DAYS), reasons

    def universe_clamp_count(self) -> tuple[int, int]:
        """(rows whose RSMT-only total mean would hit the plausibility clamp, rows) on the last date."""
        U = self._universe
        hits = n = 0
        for i in range(len(self._pred)):
            row = self._pred.iloc[i]
            try:
                mu_m, nu_m, sigma_m, _ = self._market_leg(row, str(row["ticker"]))
            except ValueError:
                continue
            beta = float(row["beta"])
            sd = total_log_sd(
                beta, float(U["sigma_e"][i]), sigma_m, float(U["rho"][i]), float(U["nu"][i]), nu_m, self.horizon
            )
            n += 1
            hits += plausibility_shift(beta * mu_m + float(U["mu_e"][i]), sd, self.mu_plausible_k) != 0.0
        return hits, n

    def _register(self, prep: _Prepared) -> None:
        """Count the row as served (the served-collapse check covers every registered row)."""
        self._served[prep.ticker] = _Served(
            prep.mu_e_rsmt,
            prep.sigma_e,
            prep.offsets_applied,
            bool(prep.reasons),
            list(prep.reasons),
        )

    def _build(
        self, prep: _Prepared, model_reasons: list[str], hold: int
    ) -> tuple[ReturnDistribution, DistributionForecast]:
        """ONE ReturnDistribution (model horizon h, simulated over `hold` days) + its forecast."""
        t, row, i, U = prep.ticker, prep.row, prep.i, self._universe
        reasons = list(prep.reasons)
        reasons.extend(r for r in model_reasons if r not in reasons)
        dist = ReturnDistribution(
            prep.mu_e_rsmt + prep.shift,
            prep.sigma_e,
            prep.mu_m,
            prep.sigma_m,
            float(U["nu"][i]),
            prep.nu_m,
            prep.beta,
            float(U["rho"][i]),
            self.horizon,
            self.n_paths,
            path_seed(t, pd.Timestamp(row["date"]).date().isoformat()),
            direction_edge=self.direction_edge,
            relative_edge=0.0 if prep.override is not None else self.relative_edge,
            hold_days=hold,
        )
        g_row, sc = float(U["gamma"][i]), float(U["scale"][i])
        offsets = {
            "gamma": g_row,
            "gamma_global": self.gamma,
            "eta": self.eta,
            "z": float(row["z"]),
            "v": float(row["v"]),
            "z_mu": sc * g_row * float(row["z"]),
            # sigma_e / sigma_leaf: scale x exp(eta v + the sigma-equation terms theta' V)
            "v_sigma": float(U["sigma_e"][i]) / float(U["sigma_leaf"][i]),
            "log_sigma_terms": math.log(float(U["sigma_e"][i]) / (float(U["sigma_leaf"][i]) * sc))
            - self.tree.eta * float(row["v"]),
            "target_scale": sc,
            "mu_leaf": float(U["mu_leaf"][i]),
            "sigma_leaf": float(U["sigma_leaf"][i]),
            "mu_excess_rsmt": prep.mu_e_rsmt,
            "blend_shift": prep.shift,
            "mu_total_unclamped": prep.mu_unclamped,
            "mu_cap": prep.mu_cap,
            "mu_clamp_shift": prep.mu_clamp,
        }
        cv_metrics = self.cv.metrics
        direction_ok = self.direction_skill_ok
        if prep.override is not None:
            ov = prep.override
            offsets["mu_excess_model"] = float(ov.mu_excess)
            offsets["mu_market_rsmt"] = prep.mu_m_rsmt
            offsets["mu_market_served"] = prep.mu_m
            cv_metrics = {**self.cv.metrics}
            for k, v in ov.metrics.items():
                if v is not None and math.isfinite(float(v)):
                    cv_metrics[k] = float(v)
            cv_metrics["excess_skill_ok"] = 1.0 if ov.skill_ok else 0.0
            cv_metrics["rsmt_skill_ok"] = 1.0 if self.skill_ok else 0.0
            # role separation: the served mean's direction is the selection model's, so is its skill
            direction_ok = bool(ov.skill_ok)
        forecast = dist.summary(
            leaf_id=prep.leaf_id,
            leaf_path=self.tree.leaf_path(prep.leaf_id),
            leaf_n=int(U["leaf_n"][i]),
            offsets=offsets,
            cv_metrics=cv_metrics,
            skill_ok=self.skill_ok,
            direction_skill_ok=direction_ok,
            degraded=bool(reasons),
            degraded_reasons=reasons,
            blend=prep.blend,
            feature_set=self.feature_set,
        )
        self._served[t] = _Served(prep.mu_e_rsmt, prep.sigma_e, prep.offsets_applied, bool(reasons), reasons)
        if reasons:
            log.warning("RSMT-Q %s %s degraded: %s", self.market.value, t, "; ".join(reasons))
        if prep.mu_clamp != 0.0:
            self._clamped[t] = (prep.mu_unclamped, prep.mu_unclamped + prep.mu_clamp)
            log.warning(
                "RSMT-Q %s %s: plausibility clamp |mu| <= %.2f*sigma_h bound (mu %+.4f -> %+.4f)",
                self.market.value,
                t,
                self.mu_plausible_k,
                prep.mu_unclamped,
                prep.mu_unclamped + prep.mu_clamp,
            )
        else:
            self._clamped.pop(t, None)
        return dist, forecast

    def predict(
        self,
        ticker: str,
        *,
        hold_days: int | None = None,
        override: ExcessOverride | None = None,
    ) -> tuple[ReturnDistribution, DistributionForecast]:
        """Serve one ticker: leaf params + offsets, market leg, then delta.

        The served total mean is RSMT-Q's own (`blend` = {"rsmt": 1.0}) or, with ``override``, the stock-selection
        model's excess mean, applied by shifting mu_e (so `mu_excess` in the forecast is the served excess mean; the
        RSMT-Q leaf value is in offsets['mu_excess_rsmt']). The path matrix covers `hold_days` (default
        settings.fusion.target_horizon_days) for the trade barrier; every forecast quantity is at the model horizon h.

        The served-collapse check includes this row; when it fires, the forecasts returned earlier
        by this model are flagged degraded in place as well (see `_propagate_model_reasons`). Prefer
        `predict_many` for a market batch: it checks the whole batch before returning.

        Raises:
            KeyError: the ticker is not in the panel (e.g. failed to load) - never silently skipped.
        """
        hold = self._hold(hold_days)
        prep = self._prepare(ticker, override)
        # register this row BEFORE the model-level checks so the served-collapse check includes it
        # (the 4th identical row must itself be flagged, not only the rows served after it)
        self._register(prep)
        model_reasons = self.degraded_reasons
        dist, forecast = self._build(prep, model_reasons, hold)
        self._propagate_model_reasons(model_reasons)
        self._returned.append((prep.ticker, forecast))
        return dist, forecast

    def predict_many(
        self,
        tickers: Sequence[str],
        *,
        hold_days: int | None = None,
        overrides: Mapping[str, ExcessOverride] | None = None,
    ) -> dict[str, tuple[ReturnDistribution, DistributionForecast] | Exception]:
        """Serve a whole batch (normally every requested ticker of this market) in one pass.

        Every row is prepared and registered first, then the model-level health checks (served
        collapse over the WHOLE batch, failed-panel share, universe collapse) run once, and only
        then are the distributions built - so every returned forecast already carries its final
        `degraded` flag and reasons (callers never need a retroactive update). Forecasts returned
        earlier by `predict` are still flagged in place when the batch reveals a collapse.

        Args:
            tickers: tickers to serve (duplicates are served once).
            hold_days: holding horizon for the barrier (default settings.fusion.target_horizon_days).
            overrides: {ticker: ExcessOverride} - the selected stock-selection model's excess mean replaces the
                RSMT-Q leaf mean for that ticker (risk parameters unchanged; see ExcessOverride).

        Returns:
            {ticker (as given): (ReturnDistribution, DistributionForecast) or the Exception}. A
            ticker that is not in the panel maps to a KeyError instance (not raised); any other
            per-ticker failure maps to its exception, and every failure is logged.
        """
        hold = self._hold(hold_days)
        ovs = dict(overrides or {})
        out: dict[str, tuple[ReturnDistribution, DistributionForecast] | Exception] = {}
        prepared: dict[str, _Prepared] = {}
        for tk in tickers:
            if tk in out:
                continue
            ov = ovs.get(tk) or ovs.get(normalise_ticker(tk))
            try:
                prepared[tk] = self._prepare(tk, ov)
                out[tk] = None  # type: ignore[assignment] - placeholder, filled below (keeps order)
            except Exception as exc:  # noqa: BLE001 - returned per ticker and logged, never silent
                out[tk] = exc
                self._log_batch_failure(tk, exc)
        for prep in prepared.values():
            self._register(prep)
        model_reasons = self.degraded_reasons  # batch-wide checks BEFORE anything is returned
        for tk, prep in prepared.items():
            try:
                dist, forecast = self._build(prep, model_reasons, hold)
            except Exception as exc:  # noqa: BLE001 - returned per ticker and logged, never silent
                out[tk] = exc
                self._served.pop(prep.ticker, None)  # not served: keep health counts honest
                self._log_batch_failure(tk, exc)
                continue
            out[tk] = (dist, forecast)
            self._returned.append((prep.ticker, forecast))
        self._propagate_model_reasons(model_reasons)
        return out

    def _log_batch_failure(self, ticker: str, exc: Exception) -> None:
        if isinstance(exc, KeyError | ValueError):
            log.warning("RSMT-Q %s %s not served: %s", self.market.value, ticker, exc)
        else:  # unexpected: keep the traceback in the log
            log.error(
                "RSMT-Q %s %s failed: %r",
                self.market.value,
                ticker,
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    def _propagate_model_reasons(self, model_reasons: list[str]) -> None:
        """A model-level reason found late (served collapse) also flags forecasts ALREADY returned.

        The collapse among served rows can only be seen once >= 4 rows are served, i.e. after the
        first rows were handed out. Those DistributionForecast objects (held by the caller / the
        report) are flagged in place, so no served row stays silently un-degraded.
        """
        prefixes = (SERVED_COLLAPSE_PREFIX, SERVED_COLLAPSE_PREFIX_EN)
        refresh = any(r.startswith(prefixes) for r in model_reasons)

        def _merge(existing: list[str]) -> list[str]:
            # the served-collapse text carries counts (1/4, 1/5, ...): replace, never pile up
            kept = [r for r in existing if not (refresh and r.startswith(prefixes))]
            return kept + [r for r in model_reasons if r not in kept]

        for t, fc in self._returned:
            merged = _merge(fc.degraded_reasons)
            if merged == fc.degraded_reasons:
                continue
            first = not fc.degraded
            fc.degraded_reasons[:] = merged
            fc.degraded = True
            served = self._served.get(t)
            if served is not None:
                served.degraded = True
                served.reasons[:] = _merge(served.reasons)
            if first:
                log.warning(
                    "RSMT-Q %s %s flagged degraded after serving: %s",
                    self.market.value,
                    t,
                    "; ".join(model_reasons),
                )

    # ------------------------------------------------------------------ reporting

    def health(self) -> list[str]:
        """Health lines: '[저하] ...' (degraded), '[정보] ...' (info) or '[정상] ...'; English models use
        '[degraded] ...', '[info] ...' and '[ok] ...' (report/render.py keys its highlighting on both)."""
        en = self.lang == "en"
        deg, info, ok = ("[degraded]", "[info]", "[ok]") if en else ("[저하]", "[정보]", "[정상]")
        lines: list[str] = []
        pred = self.panel.pred
        if self._served:
            miss = [t for t, s in self._served.items() if not s.offsets_applied]
            n = len(self._served)
            scope = self._t("서빙", "served")
        else:
            miss = [
                str(r.ticker)
                for r in pred.itertuples(index=False)
                if any(is_offset_missing(x) for x in r.degraded_reasons)
            ]
            n = len(pred)
            scope = self._t("유니버스", "universe")
        if miss:
            shown = ", ".join(miss[:8])
            lines.append(
                self._t(
                    f"{deg} 오프셋 미적용 {len(miss)}/{n}행({scope}): {shown}",
                    f"{deg} offsets not applied to {len(miss)}/{n} rows ({scope}): {shown}",
                )
            )
        else:
            lines.append(
                self._t(
                    f"{ok} 오프셋 적용 {n}/{n}행({scope}) (γ={self.gamma:+.4f}, η={self.eta:.3f})",
                    f"{ok} offsets applied to {n}/{n} rows ({scope}) (γ={self.gamma:+.4f}, η={self.eta:.3f})",
                )
            )
        for reason in self.degraded_reasons:
            lines.append(f"{deg} {reason}")
        other = [
            (str(r.ticker), r.degraded_reasons)
            for r in pred.itertuples(index=False)
            if r.degraded_reasons and not any(is_offset_missing(x) for x in r.degraded_reasons)
        ]
        if other:
            sep = " " if en else ""
            shown = ", ".join(f"{t}{sep}({'; '.join(rs)})" for t, rs in other[:5])
            lines.append(self._t(f"{deg} 입력 결측 종목: {shown}", f"{deg} names with incomplete inputs: {shown}"))
        if self.tree.n_splits == 0:
            zs = self._t(ZERO_SPLIT_INFO, ZERO_SPLIT_INFO_EN)
            lines.append(f"{info} {zs} (root p={self.tree.root_p:.2g}, n={self.tree.n_train})")
        fit = self.tree.offset_fit
        if fit is not None and not fit.converged:
            lines.append(
                self._t(
                    f"{info} 오프셋 수렴 미완 ({fit.n_iter}회)",
                    f"{info} offsets not converged ({fit.n_iter} iterations)",
                )
            )
        lo_eta, hi_eta = self.tree.eta_bounds
        if fit is not None and min(abs(self.tree.eta - lo_eta), abs(self.tree.eta - hi_eta)) < 1e-9:
            shift = self.eta - self.tree.eta  # the scaled target fits a residual eta: show raw-model bounds
            lo_eta, hi_eta = lo_eta + shift, hi_eta + shift
            lines.append(
                self._t(
                    f"{info} η={self.eta:.2f} 경계 도달(허용 [{lo_eta}, {hi_eta}]) — 변동성 오프셋이 제약됨",
                    f"{info} η={self.eta:.2f} at its bound (allowed [{lo_eta}, {hi_eta}]) — the volatility offset is "
                    "constrained",
                )
            )
        for note in self.cv.notes:
            lines.append(f"{info} {note}")
        if self._clamped:
            shown = ", ".join(f"{t} {a * 100:+.1f}%→{b * 100:+.1f}%" for t, (a, b) in list(self._clamped.items())[:6])
            k, nc, ns = self.mu_plausible_k, len(self._clamped), len(self._served)
            lines.append(
                self._t(
                    f"{info} {MU_CLAMP_INFO}(|μ| ≤ {k:g}·σ_h) 적용 {nc}/{ns}종목(서빙): {shown}",
                    f"{info} {MU_CLAMP_INFO_EN} (|μ| ≤ {k:g}·σ_h) applied to {nc}/{ns} served names: {shown}",
                )
            )
        if self.cv.available and not self.cv.skill_ok and self.skill_metric == "distribution":
            failed = "; ".join(self._gate_failures())
            lines.append(
                self._t(
                    f"{info} 분포 스킬 게이트 미달"
                    + (f"({failed})" if failed else "")
                    + " — 위험 분포가 기준 쌍둥이보다 낫다는 근거 없음",
                    f"{info} distribution skill gate not passed"
                    + (f" ({failed})" if failed else "")
                    + " — no evidence that the risk distribution beats the baseline twin",
                )
            )
        elif self.cv.available and not self.cv.skill_ok:
            failed = "; ".join(self._gate_failures())
            lines.append(
                self._t(
                    f"{info} CV 스킬 게이트 미달" + (f"({failed})" if failed else "") + " — ML 필러 가중 축소",
                    f"{info} CV skill gate not passed"
                    + (f" ({failed})" if failed else "")
                    + " — ML pillar weight reduced",
                )
            )
        if self.cv.available and self.skill_metric == "distribution" and not self.direction_skill_ok:
            lines.append(
                self._t(
                    f"{info} 방향(AUC) 게이트 미달 — RSMT-Q 자체 평균 서빙 시(종목 선택 모델 없음) ML 필러 가중 축소",
                    f"{info} direction (AUC) gate not passed — ML pillar weight reduced when RSMT-Q's own mean is "
                    "served (no stock-selection model)",
                )
            )
        lo = self.cv.metrics.get("auc_gain_p05", float("nan"))
        # only when the CI is not required: with the fold-robust verdict a majority of layouts can pass
        # while the primary layout's bound is <= 0 (that case is stated in the CV notes instead)
        direction_mode = self.skill_metric == "direction"
        if direction_mode and self.cv.skill_ok and not self.cv.gate.get("date_ci_required", True) and not (lo > 0):
            lines.append(
                self._t(
                    f"{info} 스킬 게이트 통과(날짜 CI 미요구 설정)이나 날짜 부트스트랩 AUC 개선 하한(5%) "
                    f"{lo:+.3f} ≤ 0 — 시장 방향 잡음 가능",
                    f"{info} skill gate passed (date CI not required by the config) but the date-bootstrap 5% bound "
                    f"of the AUC gain is {lo:+.3f} ≤ 0 — possibly market-direction noise",
                )
            )
        return lines

    # ------------------------------------------------------------------ skill gate text

    def _gate_failures(self) -> list[str]:
        """Reasons (in the model's language) for each failed (required) skill-gate component: the fold-robust
        majority, then the primary layout's components (when the primary layout itself failed)."""
        if self.skill_metric == "distribution":
            return self._dist_gate_failures()
        m, g = self.cv.metrics, self.cv.gate
        if not m or not g:
            return []
        out: list[str] = []
        nan = float("nan")
        if self.cv.layouts and not self.cv.robust_majority:
            rp, rn, flags = self.cv.robust_pass, self.cv.robust_n, self._layout_flags()
            out.append(
                self._t(f"폴드 배치 과반 미달 {rp}/{rn} ({flags})", f"no fold-layout majority {rp}/{rn} ({flags})")
            )
        if self.cv.skill_ok_primary or (self.cv.layouts and self.cv.robust_majority):
            return out
        if not g.get("auc_gain", False):
            gain, lim = m.get("auc_gain", nan), self.skill_min_auc_gain
            out.append(self._t(f"AUC 개선 {gain:+.3f} < {lim:.3f}", f"AUC gain {gain:+.3f} < {lim:.3f}"))
        if not g.get("logloss", False):
            a, b = m["logloss"], m["base_logloss"]
            out.append(self._t(f"logloss {a:.4f} ≥ 기준 {b:.4f}", f"logloss {a:.4f} ≥ baseline {b:.4f}"))
        if g.get("date_ci_required", False) and not g.get("date_ci", False):
            lo = m.get("auc_gain_p05", nan)
            out.append(
                self._t(
                    f"날짜 부트스트랩 AUC 개선 하한 {lo:+.3f} ≤ 0",
                    f"date-bootstrap bound of the AUC gain {lo:+.3f} ≤ 0",
                )
            )
        return out

    def _bands(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return getattr(self, "cover80_band", (0.75, 0.85)), getattr(self, "exc95_band", (0.02, 0.08))

    def _dist_gate_failures(self) -> list[str]:
        """Failed distribution-gate components: the fold-robust majority, then the primary layout's CRPS bound /
        coverage bands (when the primary layout itself failed)."""
        m, g = self.cv.metrics, self.cv.dist_gate
        if not m or not g:
            return []
        out: list[str] = []
        nan = float("nan")
        if self.cv.layouts and not self.cv.robust_majority:
            rp, rn, flags = self.cv.robust_pass, self.cv.robust_n, self._layout_flags()
            out.append(
                self._t(f"폴드 배치 과반 미달 {rp}/{rn} ({flags})", f"no fold-layout majority {rp}/{rn} ({flags})")
            )
        if self.cv.skill_ok_primary or (self.cv.layouts and self.cv.robust_majority):
            return out
        (c_lo, c_hi), (e_lo, e_hi) = self._bands()
        if not g.get("crps_ci", False):
            lo = m.get("crps_gain_p05", nan)
            out.append(
                self._t(
                    f"CRPS 개선 날짜 부트스트랩 하한 {lo:+.5f} ≤ 0",
                    f"date-bootstrap bound of the CRPS gain {lo:+.5f} ≤ 0",
                )
            )
        if not g.get("cover80", False):
            c = m.get("cover80", nan)
            out.append(self._t(f"cover80 {c:.3f} ∉ [{c_lo:g}, {c_hi:g}]", f"cover80 {c:.3f} ∉ [{c_lo:g}, {c_hi:g}]"))
        if not g.get("exc95", False):
            e = m.get("exc95", nan)
            out.append(
                self._t(
                    f"VaR95 초과율 {e:.3f} ∉ [{e_lo:g}, {e_hi:g}]", f"VaR95 exceedance {e:.3f} ∉ [{e_lo:g}, {e_hi:g}]"
                )
            )
        return out

    def _verdict(self, passed: bool) -> str:
        return self._t("통과" if passed else "미달", "passed" if passed else "not passed")

    def _layout_flags(self) -> str:
        if self.lang == "en":
            return " · ".join(
                f"{k}-fold {'passed' if lay.skill_ok else ('CV unavailable' if not lay.available else 'not passed')}"
                for k, lay in self.cv.layouts.items()
            )
        return " · ".join(
            f"{k}폴드 {'통과' if lay.skill_ok else ('CV 불가' if not lay.available else '미달')}"
            for k, lay in self.cv.layouts.items()
        )

    def robust_line(self) -> str:
        """One line with every fold layout's gate (AUC gain, date-bootstrap bound) and the verdict."""
        cv = self.cv
        if not cv.layouts:
            return self._t(
                "폴드 강건성 게이트: 배치별 결과 없음 → 미달", "Fold-robustness gate: no result per layout → not passed"
            )
        dist = self.skill_metric == "distribution"
        parts = []
        for k, lay in cv.layouts.items():
            if not lay.available:
                parts.append(self._t(f"{k}폴드 CV 불가", f"{k}-fold CV unavailable"))
                continue
            if dist:
                parts.append(
                    self._t(
                        f"{k}폴드 {self._verdict(lay.skill_ok)}(CRPS 개선 {lay.crps_gain:+.5f}, 하한 "
                        f"{lay.crps_gain_p05:+.5f}, cover80 {lay.cover80:.3f}, VaR95 초과 {lay.exc95:.3f})",
                        f"{k}-fold {self._verdict(lay.skill_ok)} (CRPS gain {lay.crps_gain:+.5f}, bound "
                        f"{lay.crps_gain_p05:+.5f}, cover80 {lay.cover80:.3f}, VaR95 exceedance {lay.exc95:.3f})",
                    )
                )
                continue
            parts.append(
                self._t(
                    f"{k}폴드 {self._verdict(lay.skill_ok)}(AUC 개선 {lay.auc_gain:+.3f}, 하한 "
                    f"{lay.auc_gain_p05:+.3f})",
                    f"{k}-fold {self._verdict(lay.skill_ok)} (AUC gain {lay.auc_gain:+.3f}, bound "
                    f"{lay.auc_gain_p05:+.3f})",
                )
            )
        need = cv.robust_n // 2 + 1
        verdict = self._verdict(cv.robust_majority)
        body = " · ".join(parts)
        kind = self._t("(분포)", " (distribution)") if dist else ""
        return self._t(
            f"폴드 강건성 게이트{kind} {verdict}: {body} → {cv.robust_pass}/{cv.robust_n} (과반 {need} 필요) · "
            f"CV {cv.seconds:.1f}s",
            f"Fold-robustness gate{kind} {verdict}: {body} → {cv.robust_pass}/{cv.robust_n} ({need} needed for a "
            f"majority) · CV {cv.seconds:.1f}s",
        )

    def gate_line(self) -> str:
        """One line with the skill gate: the fold-robust verdict, then the primary layout's date-aware
        components (each with its threshold and result)."""
        dist = self.skill_metric == "distribution"
        m, g = self.cv.metrics, (self.cv.dist_gate if dist else self.cv.gate)
        if not m or not g:
            return self._t("스킬 게이트: CV 결과 없음 → 미달", "Skill gate: no result from the CV → not passed")

        def ok(flag: bool) -> str:
            return self._t("충족" if flag else "미충족", "met" if flag else "not met")

        if dist:
            return self._dist_gate_line(ok)
        if self.lang == "en":
            ci = (
                f"{int(m['n_oos_dates'])}-date block bootstrap (B={int(m['n_boot'])}) 5% bound "
                f"AUC {m['auc_gain_p05']:+.3f} > 0 {ok(g['date_ci'])}"
                + ("" if g["date_ci_required"] else " (reference only, not gated)")
                + f" · logloss gain bound {m['logloss_gain_p05']:+.4f}"
            )
            robust = f" (fold-layout majority {self.cv.robust_pass}/{self.cv.robust_n})" if self.cv.layouts else ""
            primary = f"primary {self.cv.n_folds_primary}-fold " if self.cv.n_folds_primary else ""
            return (
                f"Skill gate {self._verdict(self.skill_ok)}{robust}: {primary}"
                f"AUC gain {m['auc_gain']:+.3f} ≥ {self.skill_min_auc_gain:.3f} {ok(g['auc_gain'])} · "
                f"logloss gain {m['logloss_gain']:+.4f} > 0 {ok(g['logloss'])} · {ci}"
            )
        ci = (
            f"날짜 {int(m['n_oos_dates'])}개 블록 부트스트랩(B={int(m['n_boot'])}) 5% 하한 "
            f"AUC {m['auc_gain_p05']:+.3f} > 0 {ok(g['date_ci'])}"
            + ("" if g["date_ci_required"] else "(참고, 게이트 미적용)")
            + f" · logloss 개선 하한 {m['logloss_gain_p05']:+.4f}"
        )
        robust = f"(폴드 배치 과반 {self.cv.robust_pass}/{self.cv.robust_n})" if self.cv.layouts else ""
        primary = f"기준 {self.cv.n_folds_primary}폴드 " if self.cv.n_folds_primary else ""
        return (
            f"스킬 게이트 {'통과' if self.skill_ok else '미달'}{robust}: {primary}"
            f"AUC 개선 {m['auc_gain']:+.3f} ≥ {self.skill_min_auc_gain:.3f} {ok(g['auc_gain'])} · "
            f"logloss 개선 {m['logloss_gain']:+.4f} > 0 {ok(g['logloss'])} · {ci}"
        )

    def _dist_gate_line(self, ok: Callable[[bool], str]) -> str:
        """gate_line of the distribution metric: CRPS-gain bound, coverage bands, and the direction gate (reference)."""
        m, g, cv = self.cv.metrics, self.cv.dist_gate, self.cv
        (c_lo, c_hi), (e_lo, e_hi) = self._bands()
        nan = float("nan")
        robust = self._t(
            f"(폴드 배치 과반 {cv.robust_pass}/{cv.robust_n})",
            f" (fold-layout majority {cv.robust_pass}/{cv.robust_n})",
        )
        robust = robust if cv.layouts else ""
        k0 = cv.n_folds_primary
        n_dates, n_boot = int(m.get("n_oos_dates", 0)), int(m.get("n_boot", 0))
        crps, lo = m.get("crps_gain", nan), m.get("crps_gain_p05", nan)
        c80, e95 = m.get("cover80", nan), m.get("exc95", nan)
        auc_g, auc_lo = m.get("auc_gain", nan), m.get("auc_gain_p05", nan)
        d_verdict = self._verdict(self.direction_skill_ok)
        if self.lang == "en":
            primary = f"primary {k0}-fold " if k0 else ""
            return (
                f"Skill gate (distribution) {self._verdict(self.skill_ok)}{robust}: {primary}CRPS gain {crps:+.5f} "
                f"({n_dates}-date block bootstrap B={n_boot} 5% bound {lo:+.5f} > 0 {ok(g.get('crps_ci', False))}) · "
                f"cover80 {c80:.3f} ∈ [{c_lo:g}, {c_hi:g}] {ok(g.get('cover80', False))} · VaR95 exceedance "
                f"{e95:.3f} ∈ [{e_lo:g}, {e_hi:g}] {ok(g.get('exc95', False))} · direction (reference): AUC gain "
                f"{auc_g:+.3f}, bound {auc_lo:+.3f} → direction gate {d_verdict}"
            )
        primary = f"기준 {k0}폴드 " if k0 else ""
        return (
            f"스킬 게이트(분포) {self._verdict(self.skill_ok)}{robust}: {primary}CRPS 개선 {crps:+.5f}"
            f"(날짜 {n_dates}개 블록 부트스트랩 B={n_boot} 5% 하한 {lo:+.5f} > 0 {ok(g.get('crps_ci', False))}) · "
            f"cover80 {c80:.3f} ∈ [{c_lo:g}, {c_hi:g}] {ok(g.get('cover80', False))} · VaR95 초과율 {e95:.3f} ∈ "
            f"[{e_lo:g}, {e_hi:g}] {ok(g.get('exc95', False))} · 방향(참고): AUC 개선 {auc_g:+.3f}, 하한 "
            f"{auc_lo:+.3f} → 방향 게이트 {d_verdict}"
        )

    def switches_line(self) -> str:
        """The model switches of this fit (docs/RSMT.md): sampling / clusters, multiple testing, target scale,
        gamma mode and the skill metric."""
        t = self.tree
        step = int(getattr(self.panel, "overlap_step", 0) or 0)
        h = self.horizon
        mt = getattr(t, "multiple_testing", "bonferroni")
        m = len(t.split_features)
        if mt == "pca":
            mt_txt = f"PCA K={m - len(t.pca['keep']) if t.pca else m} → Bonferroni m={m}"
        elif mt == "simes":
            mt_txt = f"Simes/BH m={m}"
        else:
            mt_txt = f"Bonferroni m={m}"
        leaf_g = getattr(t, "gamma_mode", "global") == "leaf"
        scaled = getattr(t, "target_scale", "none") != "none"
        dist = self.skill_metric == "distribution"
        if self.lang == "en":
            samp = (
                f"rows every {step} sessions ({h}-session block clusters)" if step else f"non-overlapping {h}-day rows"
            )
            return (
                f"Model switches: {samp} · {mt_txt} · target {'e/σ_v (EWMA vol)' if scaled else 'e (raw)'} · "
                f"{f'leaf γ (k={t.gamma_shrink_k:g})' if leaf_g else 'global γ'} · skill metric "
                f"{'distribution (CRPS / coverage)' if dist else 'direction (AUC)'}"
            )
        samp = f"{step}거래일 간격 표본({h}거래일 블록 군집)" if step else f"비중첩 {h}거래일 표본"
        return (
            f"모델 스위치: {samp} · {mt_txt} · 목표 {'e/σ_v(EWMA 변동성)' if scaled else 'e(원 수익률)'} · "
            f"{f'리프별 γ(k={t.gamma_shrink_k:g})' if leaf_g else '전역 γ'} · 스킬 지표 "
            f"{'분포(CRPS·커버리지)' if dist else '방향(AUC)'}"
        )

    def drift_line(self) -> str:
        """Line with the served market drift per benchmark: prior-shrunk historical mean."""
        pn = self.panel
        s, prior = pn.market_mu_shrink, pn.market_mu_prior
        en = self.lang == "en"
        parts = []
        for bench, mu in sorted(pn.market_mu_by_bench.items()):
            raw = pn.market_mu_raw_by_bench.get(bench, float("nan"))
            parts.append(
                f"{bench} {mu * 100:+.2f}% (history {raw * 100:+.2f}%)"
                if en
                else f"{bench} {mu * 100:+.2f}%(과거 {raw * 100:+.2f}%)"
            )
        annual = math.expm1(prior * TRADING_DAYS / self.horizon) if self.horizon else float("nan")
        benches = " · ".join(parts) or self._t("벤치마크 없음", "no benchmark")
        return self._t(
            f"시장 드리프트 μ_m({self.horizon}일) = {1 - s:.2f}×과거(시간감쇠 평균) + {s:.2f}×사전 "
            f"{prior * 100:+.2f}%(연 {annual:.0%}): {benches}",
            f"Market drift μ_m ({self.horizon}d) = {1 - s:.2f}×history (time-decayed mean) + {s:.2f}×prior "
            f"{prior * 100:+.2f}% ({annual:.0%}/yr): {benches}",
        )

    def diagnostics(self) -> list[str]:
        """Tree print, n / leaves / root p, gamma / eta / delta, CV vs baseline, dropped rows, failures."""
        tree = self.tree
        root_p = tree.root_p
        lines = list(tree.describe(self.lang))
        rp = "n/a" if math.isnan(root_p) else f"{root_p:.1e}"
        head = f"RSMT-Q {self.market.value} {self.feature_set}"
        lines.append(
            self._t(
                f"{head} · h={self.horizon}일(손절·익절 경로 {self.hold_days}일) · "
                f"n={tree.n_train} · 리프 {tree.n_leaves} · root p={rp}",
                f"{head} · h={self.horizon}d (stop / take-profit paths {self.hold_days}d) · "
                f"n={tree.n_train} · leaves {tree.n_leaves} · root p={rp}",
            )
        )
        edges = f"δ={self.direction_edge:+.3f} · δ_rel={self.relative_edge:+.3f}"
        lines.append(
            self._t(
                f"오프셋 γ={self.gamma:+.4f}(모멘텀 z{', σ_v 단위' if tree.scaled else ''}) · "
                f"η={self.eta:.3f}(log σ) · {edges}",
                f"offsets γ={self.gamma:+.4f} (momentum z{', σ_v units' if tree.scaled else ''}) · η={self.eta:.3f} "
                f"(log σ) · {edges}",
            )
        )
        m_bonf = len(tree.split_features)
        mt = getattr(tree, "multiple_testing", "bonferroni")
        mt_txt = f"Simes/BH m={m_bonf}" if mt == "simes" else f"Bonferroni m={m_bonf}"
        blocks = int(getattr(self.panel, "overlap_step", 0) or 0) > 0
        if tree.null_used == "bootstrap" and blocks:
            # clusters = runs of boot_block_horizons h-blocks with overlapping rows (RSMTree.fit_frame)
            k = int(getattr(tree, "boot_block_horizons", 1)) if getattr(tree, "row_multiplicity", 1.0) > 1 else 1
            span = self.horizon * k
            clusters = getattr(tree, "_clusters", None)
            n_clu = int(np.unique(clusters).size) if clusters is not None else tree.n_clusters
            lines.append(
                self._t(
                    f"분할 검정 귀무분포: {span}거래일 블록 군집 승수 부트스트랩(군집 {n_clu}개, "
                    f"B≤{tree.boot_sims}; Bartlett/Driscoll-Kraay HAC, 창 {span}거래일) · {mt_txt}",
                    f"split-test null: {span}-session block multiplier bootstrap ({n_clu} clusters, "
                    f"B≤{tree.boot_sims}; a Bartlett / Driscoll-Kraay HAC, window {span} sessions) · {mt_txt}",
                )
            )
        elif tree.null_used == "bootstrap":
            lines.append(
                self._t(
                    f"분할 검정 귀무분포: 날짜 군집 승수 부트스트랩(군집 {tree.n_clusters}개, "
                    f"B≤{tree.boot_sims}) · {mt_txt}",
                    f"split-test null: date-cluster multiplier bootstrap ({tree.n_clusters} clusters, "
                    f"B≤{tree.boot_sims}) · {mt_txt}",
                )
            )
        else:
            lines.append(
                self._t(
                    f"분할 검정 귀무분포: 브라운 브리지 sup-LM(trim {tree.trim:.2f}) · {mt_txt}",
                    f"split-test null: Brownian-bridge sup-LM (trim {tree.trim:.2f}) · {mt_txt}",
                )
            )
        lines.append(self.switches_line())
        m = self.cv.metrics
        if m:
            scores = (
                f"logloss {m['logloss']:.4f} vs {m['base_logloss']:.4f} · Brier {m['brier']:.4f} vs "
                f"{m['base_brier']:.4f}"
            )
            lines.append(
                self._t(
                    f"CV(OOS {int(m['n_oos'])}행, {int(m['n_folds'])}폴드) AUC {m['auc']:.3f} vs 기준 "
                    f"{m['base_auc']:.3f} · {scores}",
                    f"CV (OOS {int(m['n_oos'])} rows, {int(m['n_folds'])} folds) AUC {m['auc']:.3f} vs baseline "
                    f"{m['base_auc']:.3f} · {scores}",
                )
            )
            dist = (
                f"CRPS {m['crps']:.4f} vs {m['base_crps']:.4f} · cover80 {m['cover80']:.2f} vs "
                f"{m['base_cover80']:.2f} · IC {m['ic']:+.3f}"
            )
            rel = f"AUC {m['auc_rel']:.3f} vs {m['base_auc_rel']:.3f}"
            lines.append(self._t(f"{dist} · 초과수익 {rel}", f"{dist} · excess-return {rel}"))
            lines.append(self.gate_line())
            if self.cv.layouts:
                lines.append(self.robust_line())
        else:
            why = "; ".join(self.cv.notes) or self._t("미실행", "not run")
            lines.append(self._t(f"CV 결과 없음: {why}", f"CV gave no result: {why}"))
            lines.append(self.gate_line())
        lines.append(self.drift_line())
        hits, n_rows = self.universe_clamp_count()
        k = self.mu_plausible_k
        if m and "n_mu_clamped" in m:
            oos = self._t(
                f"(CV OOS {int(m['n_mu_clamped'])}/{int(m['n_oos'])}행)",
                f" (CV OOS {int(m['n_mu_clamped'])}/{int(m['n_oos'])} rows)",
            )
        else:
            oos = ""
        lines.append(
            self._t(
                f"{MU_CLAMP_INFO} |μ| ≤ {k:g}·σ_h: 유니버스 최종일 {hits}/{n_rows}행 적용{oos}",
                f"{MU_CLAMP_INFO_EN} |μ| ≤ {k:g}·σ_h: applied to {hits}/{n_rows} rows on the universe's last date{oos}",
            )
        )
        d = self.panel.dropped
        detail = ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]) if k != "total")
        detail = f" ({detail})" if detail else ""
        total, cand = d.get("total", 0), self.panel.n_candidates
        lines.append(self._t(f"제외 행 {total}/{cand}{detail}", f"dropped rows {total}/{cand}{detail}"))
        failed = self.panel.failed_tickers
        nt = self.panel.n_tickers
        if failed:
            shown = ", ".join(f"{t}: {e[:40]}" for t, e in list(failed.items())[:6])
            lines.append(self._t(f"로드 실패 {len(failed)}/{nt}: {shown}", f"load failed {len(failed)}/{nt}: {shown}"))
        else:
            lines.append(self._t(f"로드 실패 0/{nt}", f"load failed 0/{nt}"))
        return lines


def fit_market_model(
    panel_data: PanelData, settings: Settings, *, run_cv: bool = True, cv: CVResult | None = None
) -> MarketModel:
    """Run the expanding-window CV (metrics, delta, fold-robust skill gate), then the final fit on all
    training rows.

    `cv`: a CV already computed on another panel of the same market; it replaces the CV run. (pipeline.fit_market
    fits the whole model on the universe-only panel, so nothing depends on which other tickers were requested.)

    Raises:
        RSMTOffsetError: offsets unusable in the training panel (never dropped silently).
        ValueError: empty training panel.
    """
    if panel_data.train.empty:
        raise ValueError("fit_market_model: empty training panel")
    if cv is None:
        cv = (
            cross_validate(panel_data, settings)
            if run_cv
            else CVResult(notes=[tr(settings, "CV 미실행(run_cv=False)", "CV not run (run_cv=False)")])
        )
    tree = RSMTree.from_settings(settings, panel_data.market).fit_frame(panel_data.train)
    model = MarketModel(panel_data, tree, cv, settings)
    log.info(
        "RSMT-Q %s fitted: n=%d leaves=%d gamma=%.4g eta=%.3g skill_ok=%s (fold layouts %d/%d, primary %s)",
        panel_data.market.value,
        tree.n_train,
        tree.n_leaves,
        tree.gamma,
        tree.eta,
        cv.skill_ok,
        cv.robust_pass,
        cv.robust_n,
        cv.skill_ok_primary,
    )
    return model


# ============================================================================= RSMT-Q v2 (stock selection)

V2_BOOT_SIMS = 499
# split-test bootstrap clusters = this many consecutive h-session blocks: the weekly rows' 20-session labels overlap
# across neighbouring blocks too, so one multiplier per block was anti-conservative (docs/RSMT.md section 8)
V2_BOOT_BLOCK_HORIZONS = 3
V2_MIN_LEAF_FRAC = 0.05  # every leaf holds >= 5% of the training rows (and >= V2_MIN_LEAF_ROWS)
V2_MIN_LEAF_ROWS = 500


class RSMTRanker:
    """RSMT-Q v2: a prediction-oriented, time-series-aware MOB tree for STOCK SELECTION.

    What changed vs v1 (``MarketModel``, which still serves sigma / nu / rho / VaR of every served forecast):

    - Target and use: the leaf model is still (mu, sigma^2, nu) of an excess return, fitted to RANK names within
      a date. The stock-selection candidate uses the PER-DATE DEMEANED beta-adjusted excess return (the common
      move of the date removed, so the tree cannot spend its splits on market timing) and ranks by mu_e (leaf
      mean + gamma * z). P(e > 0) = 1 - T_nu(-mu_e / scale_e) is returned as well. Ranking by P(y_e > 0) on the
      raw excess was measured to reward volatility whenever the average excess of a period is negative
      (Phi(mu / sigma) with mu < 0 rises with sigma): KR 2019-24 OOS rank IC -0.017.
    - Partition variables = a few CROSS-SECTIONAL ranks (``xs_features``: relative state of a name on its date)
      plus a few TIME-SERIES regime / stress variables (``regime_features``: identical for all names of a date).
      A regime split therefore separates market states, an xs split separates relative states inside them, and
      their interaction (a split on an xs rank below a regime split) is how the ranking can differ by regime.
    - mu offset gamma on a STACKED ML score (``z``: the per-date rank-gauss of the best ML candidate's
      out-of-sample score - like RSMT's FIP offset, never an in-sample fit), sigma offset eta on log volatility.
    - Finer sampling (weekly rows, overlapping h-session labels): rows are clustered into BLOCKS of
      ``block_sessions`` consecutive market sessions, and the split test uses the existing cluster multiplier
      bootstrap with those blocks (overlapping labels of neighbouring weeks share a block), so the test is not
      over-confident; ``min_leaf_blocks`` blocks per child.
    - Time-decay weights 0.5 ** (age / half_life_days) on top of the per-date equal weights.

    Missing partition values: xs ranks -> 0 (the median of a rank-gauss column); regime values -> the training
    median of that column (counted in ``n_filled``). Nothing is dropped silently.

    RSMT-Q v3 (candidate ``rsmt_v3``, docs/RSMT.md) = v2 + the v1.1 switches of `RSMTree`:
    - ``multiple_testing``: "simes" (Simes' global p / BH over the partition variables) or "pca" (the REGIME
      variables reduced to ``pca_components`` principal components fitted on the training rows; the xs ranks stay).
    - ``target_scale = "vol"``: the tree splits on y / s with s = vol60 * sqrt(h/252) (the 60-day realised vol known
      at t; NaN -> the training median), the eta offset becomes the residual elasticity. ``predict`` returns
      ``mu_e`` in return units (s * standardised mean) and ranks by ``score`` = the STANDARDISED mean (mu_leaf +
      gamma z): P(y beats the date median) ~ Phi(mu_e / sigma_e) does not depend on s, whereas s * mean would
      re-rank names by volatility.
    - ``gamma_mode = "leaf"``: leaf-local stacked-score gamma, shrunk toward the parent's with ``gamma_shrink_k``
      h-session blocks of prior weight.
    The h-session block clusters are the same as v2's.
    """

    # class defaults: rankers pickled before the v3 switches existed behave as v2
    multiple_testing: str = "bonferroni"
    pca_components: int = 2
    target_scale: str = "none"
    gamma_mode: str = "global"
    gamma_shrink_k: float = 50.0
    horizon: int = 20
    vol_fill: float = float("nan")

    def __init__(
        self,
        xs_features: Sequence[str],
        regime_features: Sequence[str],
        *,
        alpha: float = 0.05,
        max_depth: int = 3,
        min_leaf: int | None = None,
        min_leaf_blocks: int = 12,
        trim: float = 0.10,
        mu_shrink_k: float = 500.0,
        nu_min: float = 4.5,
        nu_max: float = 30.0,
        half_life_days: float = 756.0,
        block_sessions: int = 20,
        boot_sims: int = V2_BOOT_SIMS,
        null_sims: int = 20000,
        cache_dir: Path | None = None,
        seed: int = 0,
        multiple_testing: str = "bonferroni",
        pca_components: int = 2,
        target_scale: str = "none",
        gamma_mode: str = "global",
        gamma_shrink_k: float = 50.0,
        horizon: int = 20,
        boot_block_horizons: int = V2_BOOT_BLOCK_HORIZONS,
    ):
        if target_scale not in ("none", "vol"):
            raise ValueError(f"RSMTRanker target_scale must be 'none' or 'vol', got {target_scale!r}")
        self.xs_features = list(xs_features)
        self.regime_features = list(regime_features)
        self.partition = [*self.xs_features, *self.regime_features]
        if not self.partition:
            raise ValueError("RSMTRanker needs at least one partition feature")
        self.alpha = float(alpha)
        self.max_depth = int(max_depth)
        self.min_leaf = None if min_leaf is None else int(min_leaf)
        self.min_leaf_blocks = int(min_leaf_blocks)
        self.trim = float(trim)
        self.mu_shrink_k = float(mu_shrink_k)
        self.nu_min, self.nu_max = float(nu_min), float(nu_max)
        self.half_life_days = float(half_life_days)
        self.block_sessions = max(1, int(block_sessions))
        self.boot_sims = int(boot_sims)
        self.null_sims = int(null_sims)
        self.cache_dir = cache_dir
        self.seed = int(seed)
        self.multiple_testing = multiple_testing
        self.pca_components = int(pca_components)
        self.target_scale = target_scale
        self.gamma_mode = gamma_mode
        self.gamma_shrink_k = float(gamma_shrink_k)
        self.horizon = max(1, int(horizon))
        self.boot_block_horizons = max(1, int(boot_block_horizons))
        self.tree: RSMTree | None = None
        self.c_v = 0.0
        self.vol_fill = float("nan")  # training median of the volatility column (scale / offset of a missing vol)
        self.fill: dict[str, float] = {}
        self.n_filled: dict[str, int] = {}
        self.n_blocks = 0

    @property
    def version(self) -> int:
        """3 when any v3 switch is on, else 2."""
        v3 = (self.multiple_testing, self.target_scale, self.gamma_mode) != ("bonferroni", "none", "global")
        return 3 if v3 else 2

    def _scale(self, vol: pd.Series) -> np.ndarray:
        """Horizon volatility s = vol * sqrt(h/252) (vol missing / non-positive -> the training median)."""
        v = vol.astype(float).where(vol.astype(float) > 0).fillna(self.vol_fill)
        return v.to_numpy(dtype=float) * math.sqrt(self.horizon / TRADING_DAYS)

    def _x(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        x = frame[self.partition].astype(float).copy()
        for c in self.xs_features:
            if fit:
                self.n_filled[c] = int(x[c].isna().sum())
            x[c] = x[c].fillna(0.0)
        for c in self.regime_features:
            if fit:
                med = float(x[c].median()) if x[c].notna().any() else 0.0
                self.fill[c] = med
                self.n_filled[c] = int(x[c].isna().sum())
            x[c] = x[c].fillna(self.fill.get(c, 0.0))
        return x

    def fit(
        self,
        frame: pd.DataFrame,
        calendar: pd.DatetimeIndex,
        *,
        y_col: str = "y_e",
        z_col: str = "stack_z",
        vol_col: str = "vol60",
        weight_col: str | None = "w_date",
    ) -> RSMTRanker:
        """Fit on ``frame`` (partition columns, y, stacked score z, volatility, date).

        Raises:
            ValueError: empty frame / non-finite target.
            RSMTOffsetError: z or v unusable (never dropped silently).
        """
        if frame.empty:
            raise ValueError("RSMTRanker.fit: empty training frame")
        y = frame[y_col].to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("RSMTRanker.fit: non-finite target - drop and count upstream")
        x = self._x(frame, fit=True)
        dates = pd.DatetimeIndex(frame["date"])
        age = (dates.max() - dates).days.to_numpy(dtype=float)
        w = 0.5 ** (age / self.half_life_days)
        if weight_col is not None and weight_col in frame:
            w = w * frame[weight_col].fillna(0.0).to_numpy(dtype=float)
        w = w / w.mean()
        vol = frame[vol_col].astype(float)
        self.vol_fill = float(vol[vol > 0].median())
        log_v = np.log(vol.where(vol > 0)).fillna(np.log(self.vol_fill))
        self.c_v = float((w * log_v).sum() / w.sum())
        v = log_v.to_numpy() - self.c_v
        cal = calendar.to_numpy(dtype="datetime64[ns]")
        pos = np.searchsorted(cal, dates.to_numpy(dtype="datetime64[ns]"), "right") - 1
        blocks = pos // self.block_sessions
        self.n_blocks = int(np.unique(blocks).size)
        n = len(frame)
        min_leaf = self.min_leaf or max(V2_MIN_LEAF_ROWS, int(V2_MIN_LEAF_FRAC * n))
        scaled = self.target_scale != "none"
        self.tree = RSMTree(
            self.partition,
            alpha=self.alpha,
            min_leaf=min_leaf,
            max_depth=self.max_depth,
            trim=self.trim,
            mu_shrink_k=self.mu_shrink_k,
            nu_min=self.nu_min,
            nu_max=self.nu_max,
            null_sims=self.null_sims,
            cache_dir=self.cache_dir,
            null_method="bootstrap",
            boot_sims=self.boot_sims,
            min_leaf_dates=self.min_leaf_blocks,
            null_seed=self.seed,
            multiple_testing=self.multiple_testing,
            pca_components=self.pca_components,
            pca_features=self.regime_features if self.multiple_testing == "pca" else None,
            target_scale=self.target_scale,
            gamma_mode=self.gamma_mode,
            gamma_shrink_k=self.gamma_shrink_k,
            eta0=0.0 if scaled else 1.0,
            eta_bounds=(-1.5, 1.0) if scaled else (-0.5, 2.0),
        ).fit(
            x,
            y,
            frame[z_col].to_numpy(dtype=float),
            v,
            w,
            None,
            blocks,
            self._scale(vol) if scaled else None,
            np.floor_divide(blocks, getattr(self, "boot_block_horizons", 1)),
        )
        return self

    def predict(self, frame: pd.DataFrame, *, z_col: str = "stack_z", vol_col: str = "vol60") -> dict[str, np.ndarray]:
        """mu_e, sigma_e, nu, P(y_e > 0), the leaf id and the ranking ``score`` per row (z NaN -> 0 = median,
        vol NaN -> exp(c_v) / the training median vol). score = mu_e, or with a scaled target the standardised
        mean mu_e / s (see the class docstring)."""
        if self.tree is None:
            raise RuntimeError("RSMTRanker is not fitted")
        x = self._x(frame, fit=False)
        z = frame[z_col].astype(float).fillna(0.0).to_numpy()
        vol = frame[vol_col].astype(float)
        v = (np.log(vol.where(vol > 0)) - self.c_v).fillna(0.0).to_numpy()
        scaled = getattr(self, "target_scale", "none") != "none"
        s = self._scale(vol) if scaled else None
        p = self.tree.predict_params(x, z, v, s)
        nu = np.maximum(p["nu"], 2.05)
        scale = p["sigma_e"] * np.sqrt((nu - 2.0) / nu)
        p_beat = stats.t.sf(-p["mu_e"] / np.where(scale > 0, scale, np.nan), nu)
        return {
            "mu_e": p["mu_e"],
            "sigma_e": p["sigma_e"],
            "nu": p["nu"],
            "p_beat": np.nan_to_num(p_beat, nan=0.5),
            "leaf_id": p["leaf_id"],
            "score": p["mu_e"] / p["scale"] if scaled else p["mu_e"],
        }

    def describe(self, lang: str = "ko") -> list[str]:
        """Header (English in both languages) + the tree print in ``lang`` ("ko" / "en")."""
        ver = getattr(self, "version", 2)
        if self.tree is None:
            return [f"RSMT-Q v{ver}: not fitted"]
        head = (
            f"RSMT-Q v{ver}: n={self.tree.n_train} · blocks={self.n_blocks} · leaves={self.tree.n_leaves} · "
            f"gamma={self.tree.gamma:+.4f} (stacked ML z) · eta={self.tree.eta_equivalent:.3f}"
        )
        if ver == 3:
            head += (
                f" · {self.multiple_testing} · target {'y/s (vol60)' if self.target_scale != 'none' else 'y'} · "
                f"gamma {self.gamma_mode}"
            )
        return [head, *self.tree.describe(lang)]

    def importance(self) -> dict[str, float]:
        """Split gain proxy: sum over internal nodes of the split statistic q_max (plus |gamma| for the offset)."""
        out = dict.fromkeys(self.tree.split_features if self.tree is not None else self.partition, 0.0)
        if self.tree is None or self.tree.root is None:
            return out
        stack = [self.tree.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                continue
            out[node.split_var] = out.get(node.split_var, 0.0) + float(node.q_max)  # type: ignore[index]
            stack.extend([node.left, node.right])  # type: ignore[list-item]
        out["stack_z (gamma)"] = abs(self.tree.gamma)
        return out
