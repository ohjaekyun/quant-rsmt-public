"""Strictly out-of-sample evaluation of every QuantCIO signal: ``quantcio evaluate --market KR|US``.

Writes ``evaluation_<market>_<N>u_<Y>y.md`` (Korean, human-readable) and ``.json`` into ``--out``
(default ``<report.output_dir>/evaluation``). Sections:

A. RSMT-Q vs its baseline twin (root-only, no offsets) for every expanding-window fold layout in
   ``--folds``: AUC, log loss, Brier, CRPS, 80% band coverage and VaR95 exceedance (target 5%), with
   date-block bootstrap 90% CIs of the gains, each layout's single-layout skill gate and the fold-robust
   verdict exactly as served (``rsmt.robust_cv_folds``).
B. Stock selection: on every OOS test date rank the names by ``mu_excess`` (and separately by
   ``prob_outperform``); Spearman rank IC against the realised beta-adjusted excess return
   ``y_e = r - beta*m``; mean IC, IC t-stat across dates (the grid is every h sessions, so the 20-day
   windows do not overlap), share of dates with IC > 0 and the top-minus-bottom quintile spread (mean, t,
   annualised). Baselines: 12-1 momentum z (``xs_mom_z``), 1-month reversal (``-ret_1m``) and random.
C. Calibration: P(up) and P(outperform) reliability deciles (raw, before the direction edge delta, which
   is fitted on these same rows), 80% band coverage and VaR95 exceedance by period, market regime and leaf.
D. Barrier calibration (validates Module 4): entry = close_t, stop = entry - clip(2*ATR14_t, 4%, 15%)
   (``risk.*``; ATR from bars <= t), take-profit = entry + 1.5 * stop distance; predicted P(TP first) /
   P(SL first) / P(time exit) from the row's own OOS ``ReturnDistribution`` over 60 sessions (1000 paths)
   vs the realised first touch on the next 60 daily closes (rows without 60 future bars are excluded and
   counted); Brier, reliability and AUC vs the root-only twin.
E. Point-in-time price factors: momentum, volatility(stability), trend and their price-only composite,
   computed with ``quant.indicators.compute_technicals`` and the ``quant.factors`` sub-score functions on
   bars <= t at every grid date; the same ranking statistics as B. Value / quality are excluded because
   their inputs are TODAY's fundamentals snapshot (no point-in-time history).
F. Removed 2026-10-02 (the per-ticker HGB+Ridge ensemble was deleted); the letters of the other sections are kept.
G. Plain-Korean verdict (what has evidence, what does not) and recommended pillar weights; the config is
   never changed.

No look-ahead (tests/test_evaluation.py checks each item):
- RSMT-Q rows are the CV test rows of ``rsmt.cross_validate_layouts``: each fold's tree, offsets, market
  drift and baseline are fitted only on rows whose label window ends before the test block
  (``rsmt.cv_splits`` purge) and the serving plausibility clamp is applied.
- Every input at t (regime features, beta, xs momentum z, ATR14, factor scores) uses bars <= t; the
  realised outcomes (``y_e``, the barrier first touch) are the only data after t.
- Biases that remain are stated in the report: with ``--universe-mode current`` the universe is TODAY's top-N
  (survivorship / selection); with ``--universe-mode pit`` (default when ``pit.universe_mode == "pit"``) it is the
  point-in-time top-N per month-end incl. delisted names (``data.pit``). The hyper-parameters were chosen by the
  developers on overlapping data; KR prices are not dividend adjusted.
- Registered holdouts are removed first (``holdout.guard_panel``): cross-sectional holdout (XSH) names and every
  bar on/after the earliest forward-holdout (FTH) start, so an evaluation never looks at held-out data.

Nothing fails silently: every skipped row, failed name or unavailable layout is counted and printed in the
report's notes (and logged).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import optimize, stats

from quantcio import __version__
from quantcio.config import Settings
from quantcio.data import base as data_base
from quantcio.data.base import PricePanel
from quantcio.data.cache import DiskCache
from quantcio.models import Fundamentals, Instrument, Market, MarketData
from quantcio.quant import rsmt
from quantcio.quant.distribution import ReturnDistribution, apply_edge, path_seed, total_log_sd
from quantcio.quant.factors import (
    FACTOR_LABELS,
    momentum_subscores,
    trend_subscores,
    volatility_subscores,
)
from quantcio.quant.indicators import atr as atr_series
from quantcio.quant.indicators import compute_technicals
from quantcio.quant.panel import XS_MIN_TICKERS, PanelData, build_panel, on_grid_rows

log = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")
TRADING_DAYS = 252
EVAL_CACHE_NS = "eval_panel"
EVAL_CACHE_VERSION = 1
OUT_SUBDIR = "evaluation"
UNIVERSE_MODES = ("pit", "current")
MIN_YEARS = 3  # 12-1 momentum + the 3y vol percentile warm up ~1.2y; CV needs dates after that
BARRIER_WORKERS = 4
BARRIER_PATHS = 1000
QUANTILES = 5  # top-minus-bottom quintile spread
RELIABILITY_BINS = 10
BARRIER_BINS = 5
LEAF_MIN_ROWS = 30  # leaf coverage rows shown only with at least this many OOS rows
RANDOM_NULL_PERMS = 500
# bars kept for one point-in-time factor snapshot: the 756-session vol-percentile window over 60-day vol
# needs 816 bars; with 830 the snapshot equals the full-history computation (ATR/RSI seeds decay (13/14)^800)
PIT_ROWS = 830
T_EVIDENCE = 2.0  # |t| >= 2: evidence (about 5% two-sided on independent dates)
T_WEAK = 1.0
TIER_EVIDENCE, TIER_WEAK, TIER_NONE, TIER_UNTESTABLE = "근거 있음", "근거 약함", "근거 없음", "검증 불가"
# recommended pillar weight = current weight x multiplier, renormalised (a stated policy, not a fit)
TIER_MULT = {TIER_EVIDENCE: 1.0, TIER_WEAK: 0.6, TIER_NONE: 0.3, TIER_UNTESTABLE: 0.6}
PRICE_FACTORS = ("momentum", "volatility", "trend")
PILLAR_LABELS = {"factor": "팩터", "ml": "ML(RSMT-Q)", "llm": "LLM"}
# calibration verdicts (not evidence tiers): tolerances for "보정 양호"
CAL_OK, CAL_BAD = "보정 양호", "보정 미흡"
# the average predicted frequency matches the realised one, but the per-row probabilities do not order the
# outcomes (recalibration slope outside SLOPE_BAND): right on average, no per-name information
CAL_LEVEL = "평균만 보정"
ECE_TOL = 0.02
SLOPE_BAND = (0.7, 1.3)
COVER_TOL = 0.03  # |cover80 - 0.80|
EXC_TOL = 0.02  # |VaR95 exceedance - 0.05|
BARRIER_TOL = 0.03  # |mean predicted - realised| of P(TP first) / P(SL first)

# section B score columns -> Korean labels (order = table order)
RANK_SIGNALS: dict[str, str] = {
    "mu_excess": "RSMT-Q μ_excess (베타조정 초과수익 기대)",
    "prob_outperform": "RSMT-Q P(BM 초과)",
    "mu_leaf": "RSMT-Q 리프 μ (모멘텀 오프셋 제외)",
    "momentum": "기준: 12-1 모멘텀 z",
    "reversal": "기준: 단기 반전(−1개월 수익률)",
    "random": "기준: 무작위",
}
FACTOR_SIGNALS: dict[str, str] = {
    "momentum": f"{FACTOR_LABELS['momentum']} 점수",
    "volatility": f"{FACTOR_LABELS['volatility']} 점수",
    "trend": f"{FACTOR_LABELS['trend']} 점수",
    "price_composite": "가격 팩터 합성(모멘텀·변동성·추세 가중)",
}


class EvaluationError(RuntimeError):
    """The evaluation cannot run (no panel, no CV result). The message is Korean."""


# ====================================================================================== results


@dataclass
class EvaluationResult:
    """Everything the evaluation measured. `sections` ("A".."G") and `meta` are JSON-ready dicts;
    `frames` keeps the row-level frames (OOS rows, barrier replays, factor scores) for audits/tests."""

    market: str
    meta: dict[str, Any]
    sections: dict[str, dict[str, Any]]
    notes: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def to_json(self) -> dict[str, Any]:
        return _jsonable(
            {
                "market": self.market,
                "meta": self.meta,
                "sections": self.sections,
                "notes": self.notes,
                "timings": self.timings,
            }
        )


@dataclass
class EvaluationArtifacts:
    md_path: Path
    json_path: Path
    result: EvaluationResult


# ====================================================================================== helpers


def _nan() -> float:
    return float("nan")


def _fin(x: Any) -> float:
    """float(x) or NaN (None / non-numeric / non-finite)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _jsonable(x: Any) -> Any:
    """Recursively convert to JSON-safe values (NaN/inf -> None, numpy scalars -> Python, dates -> ISO)."""
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple | set):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, bool | np.bool_):
        return bool(x)
    if isinstance(x, int | np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, pd.Timestamp | datetime | date):
        return x.isoformat()
    return x


def small_cluster_factor(n_clusters: int) -> float:
    """c_G = sqrt(G/(G-1)) * t_{0.95, G-1} / z_{0.95}: widens a bootstrap percentile interval of a mean over
    G clusters (CR1 variance factor and t critical value; same correction as rsmt.DateBootstrap)."""
    g = int(n_clusters)
    if g < 2:
        return float("nan")
    return math.sqrt(g / (g - 1.0)) * float(stats.t.ppf(0.95, g - 1)) / float(stats.norm.ppf(0.95))


def ci90(hat: float, draws: np.ndarray, n_clusters: int) -> tuple[float, float]:
    """Two-sided 90% interval: bootstrap 5%/95% percentiles, their distance to `hat` widened by the
    small-cluster factor c_G. (NaN, NaN) without finite draws."""
    d = np.asarray(draws, dtype="float64")
    d = d[np.isfinite(d)]
    h = _fin(hat)
    c = small_cluster_factor(n_clusters)
    if d.size == 0 or not math.isfinite(h) or not math.isfinite(c):
        return float("nan"), float("nan")
    q05, q95 = (float(v) for v in np.quantile(d, [0.05, 0.95]))
    return h - c * (h - q05), h + c * (q95 - h)


def block_bootstrap_means(
    values: np.ndarray,
    dates: np.ndarray,
    n_boot: int,
    seed: int = 0,
    block_dates: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Cluster bootstrap of per-row means, resampling whole blocks of consecutive test DATES.

    Rows on one date share the market move (~one independent draw per date), so dates, not rows, are
    resampled. `block_dates` > 1 joins that many consecutive dates into one block (used when the outcome
    window spans several grid steps, e.g. a 60-session barrier on a 20-session grid). A draw is a
    multinomial count vector over blocks; its mean is sum(c*block_sum) / sum(c*block_rows), which equals
    stacking the drawn blocks' rows.

    Args:
        values: (n,) or (n, k) finite per-row values.
        dates: (n,) row dates.

    Returns:
        (hat (k,), draws (n_boot, k), n_blocks).

    Raises:
        ValueError: shape mismatch, non-finite values or block_dates < 1.
    """
    v = np.asarray(values, dtype="float64")
    if v.ndim == 1:
        v = v[:, None]
    d = pd.to_datetime(pd.Series(np.asarray(dates))).to_numpy()
    if v.shape[0] != d.size:
        raise ValueError(f"values ({v.shape[0]} rows) and dates ({d.size}) differ in length")
    if not np.isfinite(v).all():
        raise ValueError("block_bootstrap_means: values must be finite (drop NaN rows first)")
    if int(block_dates) < 1:
        raise ValueError(f"block_dates must be >= 1, got {block_dates}")
    k = v.shape[1]
    if v.shape[0] == 0:
        return np.full(k, np.nan), np.empty((0, k)), 0
    uniq = np.unique(d)
    block = np.searchsorted(uniq, d) // int(block_dates)
    n_blocks = int(block.max()) + 1
    sums = np.zeros((n_blocks, k))
    np.add.at(sums, block, v)
    rows = np.bincount(block, minlength=n_blocks).astype("float64")
    hat = sums.sum(axis=0) / rows.sum()
    if n_blocks < 2 or n_boot < 1:
        return hat, np.empty((0, k)), n_blocks
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, 0xE7A1]))
    out = []
    for start in range(0, int(n_boot), 256):
        b = min(256, int(n_boot) - start)
        c = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=b).astype("float64")
        out.append((c @ sums) / (c @ rows)[:, None])
    return hat, np.vstack(out), n_blocks


def _mean_ci(values: np.ndarray, dates: np.ndarray, n_boot: int, seed: int, block_dates: int = 1) -> dict[str, Any]:
    """{hat, ci90, n_blocks} of the mean of `values` (date-block bootstrap)."""
    v = np.asarray(values, dtype="float64")
    ok = np.isfinite(v)
    hat, draws, g = block_bootstrap_means(v[ok], np.asarray(dates)[ok], n_boot, seed, block_dates)
    lo, hi = ci90(hat[0], draws[:, 0], g) if draws.size else (float("nan"), float("nan"))
    return {"hat": float(hat[0]), "ci90": [lo, hi], "n_blocks": g}


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    s = np.asarray(score, dtype="float64")
    y = np.asarray(label, dtype=bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = stats.rankdata(s)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _t_of_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    sd = float(x.std(ddof=1))
    if sd <= 0:
        return float("nan")
    return float(x.mean() / (sd / math.sqrt(x.size)))


# =============================================================================== settings / data


def eval_settings(
    settings: Settings, market: Market, universe: int | None = None, years: int | None = None
) -> Settings:
    """A deep copy of `settings` with the evaluation's universe size and history length.

    Raises:
        ValueError: unknown market, universe < XS_MIN_TICKERS (cross-sectional z and rank IC need
            that many names per date) or years < MIN_YEARS.
    """
    s = settings.model_copy(deep=True)
    key = market.value
    if key not in s.rsmt.markets:
        raise ValueError(f"rsmt.markets 에 {key} 설정이 없습니다")
    if universe is not None:
        if int(universe) < XS_MIN_TICKERS:
            raise ValueError(f"--universe 는 {XS_MIN_TICKERS} 이상이어야 합니다(횡단면 z·순위 IC): {universe}")
        s.rsmt.markets[key].universe_size = int(universe)
    if years is not None:
        if int(years) < MIN_YEARS:
            raise ValueError(f"--years 는 {MIN_YEARS} 이상이어야 합니다(모멘텀·변동성 백분위 워밍업 후 CV): {years}")
        s.history_years = int(years)
    return s


def load_eval_panel(
    market: Market, settings: Settings, cache: DiskCache, *, use_cache: bool = True
) -> tuple[PricePanel, list[str], bool]:
    """The market universe's price panel (``data.base.load_price_panel``), cached for the day.

    Returns (panel, universe order as the provider lists it (market cap for KR), cached?). The panel is
    cached in DiskCache namespace ``eval_panel`` for ``data.price_ttl_hours`` under (market, N, years, day).
    """
    n = settings.rsmt.markets[market.value].universe_size
    key = DiskCache.key("eval-panel", EVAL_CACHE_VERSION, market.value, n, settings.history_years, date.today())
    hit = cache.get(EVAL_CACHE_NS, key, settings.data.price_ttl_hours) if use_cache else None
    if isinstance(hit, dict) and isinstance(hit.get("panel"), PricePanel):
        return hit["panel"], list(hit.get("order") or sorted(hit["panel"].prices)), True
    provider = data_base.get_provider(market, settings, cache)
    order = [data_base.normalise_ticker(t) for t in provider.universe(n)]
    panel = data_base.load_price_panel(market, settings, cache)
    if use_cache:
        try:
            cache.set(EVAL_CACHE_NS, key, {"panel": panel, "order": order})
        except Exception as exc:  # noqa: BLE001 - a cache write failure only costs a reload next time
            log.warning("evaluation panel cache write failed: %s", exc)
    return panel, order, False


def resolve_universe_mode(settings: Settings, mode: str | None) -> str:
    """ "pit" or "current"; None -> "pit" when ``pit.enabled`` and ``pit.universe_mode == "pit"``.

    Raises:
        ValueError: an unknown mode.
    """
    if mode is None:
        return "pit" if settings.pit.enabled and settings.pit.universe_mode == "pit" else "current"
    m = str(mode).strip().lower()
    if m not in UNIVERSE_MODES:
        raise ValueError(f"--universe-mode 는 pit 또는 current 여야 합니다: {mode}")
    return m


def load_pit_eval_panel(
    market: Market, settings: Settings, cache: DiskCache, *, today: date | None = None
) -> tuple[PricePanel, list[str], dict[str, Any]]:
    """The point-in-time universe (top ``rsmt.markets.<M>.universe_size`` per month-end since ``history_years``
    ago, delisted names with their terminal prices) as a ``PricePanel`` whose ``membership`` restricts the RSMT-Q
    rows to the members of each date (``data.pit``; cached by the PIT builders).

    Returns (panel, the latest month-end's members by rank (section F picks from it), info).
    """
    from datetime import timedelta

    from quantcio.data.pit import load_pit_universe

    n = int(settings.rsmt.markets[market.value].universe_size)
    start = (today or date.today()) - timedelta(days=int(365.25 * settings.history_years) + 30)
    uni = load_pit_universe(market, settings, cache, n, start)
    panel = uni.to_price_panel()
    mem = uni.membership
    last = mem["date"].max()
    order = [str(e) for e in mem.loc[mem["date"] == last].sort_values("rank")["entity"] if e in panel.prices]
    info = {
        "pit_start": start.isoformat(),
        "pit_entities": len(panel.prices),
        "pit_rebalances": int(mem["date"].nunique()),
        "pit_last_rebalance": str(pd.Timestamp(last).date()) if len(mem) else None,
        "pit_delisted": int(uni.meta["delisted"].fillna(False).astype(bool).sum())
        if uni.meta is not None and "delisted" in uni.meta
        else None,
    }
    return panel, order, info


# =========================================================================== B: ranking statistics


def quantile_bucket_size(n: int, quantiles: int = QUANTILES) -> int:
    """Names in the top (= bottom) bucket of an n-name cross-section: round(n / quantiles), at least 1 and
    at most n // 2 (the two buckets never overlap)."""
    n, q = int(n), int(quantiles)
    if n < 2 or q < 2:
        raise ValueError(f"quantile_bucket_size needs n >= 2 and quantiles >= 2 (got {n}, {q})")
    return max(1, min(int(round(n / q)), n // 2))


def cross_sectional_ic(
    frame: pd.DataFrame,
    score: str,
    target: str = "y_e",
    date_col: str = "date",
    min_names: int = XS_MIN_TICKERS,
    quantiles: int = QUANTILES,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-date Spearman rank IC of `score` vs `target` and the top-minus-bottom quantile spread.

    Rows with a NaN score or target are ignored; dates with fewer than `min_names` names are skipped.
    The IC uses average ranks (ties handled as Spearman does). The top and bottom buckets have the SAME
    size k = round(n / quantiles) (`quantile_bucket_size`; with n = 59 the old `i*q//n` rule gave a
    12-name bottom and an 11-name top): names are ordered by score and ties are broken by a seeded
    random draw (a few-valued score such as a leaf mean would otherwise leave a bucket empty or
    oversized). A date whose score is constant carries no ranking information: its IC and spread are
    NaN. Returns one row per date: date, n, ic, top, bottom, spread (mean target of the top bucket minus
    the bottom bucket), top_excess (top bucket minus the date mean).
    """
    cols = [date_col, score, target]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, 0x7B1E]))
    rows: list[dict[str, Any]] = []
    for d, g in frame[cols].dropna().groupby(date_col, sort=True):
        n = len(g)
        if n < min_names:
            continue
        s = g[score].to_numpy(dtype="float64")
        y = g[target].to_numpy(dtype="float64")
        tiebreak = rng.random(n)
        if np.ptp(s) == 0:
            rows.append(
                {
                    "date": d,
                    "n": n,
                    "ic": _nan(),
                    "top": _nan(),
                    "bottom": _nan(),
                    "spread": _nan(),
                    "top_excess": _nan(),
                }
            )
            continue
        s_rank = stats.rankdata(s)
        y_rank = stats.rankdata(y)
        ic = float(np.corrcoef(s_rank, y_rank)[0, 1]) if np.ptp(y_rank) > 0 else _nan()
        order = np.lexsort((tiebreak, s))  # ascending score, random among ties
        k = quantile_bucket_size(n, quantiles)
        top = y[order[-k:]]
        bot = y[order[:k]]
        t_mean = float(top.mean()) if top.size else _nan()
        b_mean = float(bot.mean()) if bot.size else _nan()
        rows.append(
            {
                "date": d,
                "n": n,
                "ic": ic,
                "top": t_mean,
                "bottom": b_mean,
                "spread": t_mean - b_mean,
                "top_excess": t_mean - float(y.mean()),
            }
        )
    return pd.DataFrame(rows, columns=["date", "n", "ic", "top", "bottom", "spread", "top_excess"])


def ranking_stats(per_date: pd.DataFrame, horizon: int) -> dict[str, float]:
    """Summary of `cross_sectional_ic` output: mean IC, IC t across dates, share IC > 0 and the quintile
    spread (mean per h days, t, simple annualisation x252/h, annualised Sharpe)."""
    ic = per_date["ic"].to_numpy(dtype="float64") if len(per_date) else np.empty(0)
    sp = per_date["spread"].to_numpy(dtype="float64") if len(per_date) else np.empty(0)
    te = per_date["top_excess"].to_numpy(dtype="float64") if "top_excess" in per_date and len(per_date) else np.empty(0)
    ic, sp, te = ic[np.isfinite(ic)], sp[np.isfinite(sp)], te[np.isfinite(te)]
    periods = TRADING_DAYS / float(horizon)
    sd_sp = float(sp.std(ddof=1)) if sp.size >= 2 else _nan()
    return {
        "n_dates": int(ic.size),
        "mean_ic": float(ic.mean()) if ic.size else _nan(),
        "sd_ic": float(ic.std(ddof=1)) if ic.size >= 2 else _nan(),
        "t_ic": _t_of_mean(ic),
        "share_ic_pos": float((ic > 0).mean()) if ic.size else _nan(),
        "spread_mean": float(sp.mean()) if sp.size else _nan(),
        "spread_t": _t_of_mean(sp),
        "spread_annual": float(sp.mean() * periods) if sp.size else _nan(),
        "spread_sharpe": float(sp.mean() / sd_sp * math.sqrt(periods)) if sp.size >= 2 and sd_sp > 0 else _nan(),
        "share_spread_pos": float((sp > 0).mean()) if sp.size else _nan(),
        "top_excess_mean": float(np.nanmean(te)) if te.size and np.isfinite(te).any() else _nan(),
        "top_excess_t": _t_of_mean(te),
        "mean_names": float(per_date["n"].mean()) if len(per_date) else _nan(),
    }


def paired_ic_difference(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float]:
    """Mean and t of the per-date IC difference a - b over their common dates."""
    m = a[["date", "ic"]].merge(b[["date", "ic"]], on="date", suffixes=("_a", "_b")).dropna()
    diff = (m["ic_a"] - m["ic_b"]).to_numpy(dtype="float64")
    return {"n_dates": int(diff.size), "mean_diff": float(diff.mean()) if diff.size else _nan(), "t": _t_of_mean(diff)}


def random_null_band(
    frame: pd.DataFrame, target: str = "y_e", n_perm: int = RANDOM_NULL_PERMS, seed: int = 0
) -> dict[str, float]:
    """Permutation null of random rankings on the same dates: the 2.5% / 97.5% quantiles of the IC t, of
    the top-minus-bottom spread t and of the top-minus-all t (buckets of `quantile_bucket_size`), i.e. how
    large each |t| gets by chance with this many dates and names. A random score's own row in section B
    is ONE draw from these distributions (a single t of +2.6 is a ~1% draw, not a bug)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, 0x4A11]))
    ics, spreads, tops = [], [], []
    for _, g in frame[["date", target]].dropna().groupby("date", sort=True):
        n = len(g)
        if n < XS_MIN_TICKERS:
            continue
        yv = g[target].to_numpy(dtype="float64")
        y = g[target].rank(method="average").to_numpy()
        y = y - y.mean()
        ny = float(np.sqrt((y * y).sum()))
        if ny == 0:
            continue
        base = np.arange(1, n + 1, dtype="float64") - (n + 1) / 2.0
        perms = rng.permuted(np.tile(base, (n_perm, 1)), axis=1)
        ics.append(perms @ y / (np.sqrt((base * base).sum()) * ny))
        # the permuted rank positions double as a random ordering of the names: bottom k / top k buckets
        k = quantile_bucket_size(n)
        pos = np.argsort(perms, axis=1)
        top = yv[pos[:, -k:]].mean(axis=1)
        spreads.append(top - yv[pos[:, :k]].mean(axis=1))
        tops.append(top - yv.mean())
    nan_out = {
        "n_perm": int(n_perm),
        "t_q025": _nan(),
        "t_q975": _nan(),
        "share_abs_t_ge_2": _nan(),
        "spread_t_q025": _nan(),
        "spread_t_q975": _nan(),
        "top_excess_t_q025": _nan(),
        "top_excess_t_q975": _nan(),
    }
    if len(ics) < 3:
        return nan_out

    def t_of(rows: list[np.ndarray]) -> np.ndarray:
        mat = np.vstack(rows)  # (dates, perms)
        return mat.mean(axis=0) / (mat.std(axis=0, ddof=1) / math.sqrt(mat.shape[0]))

    t, ts, tt = t_of(ics), t_of(spreads), t_of(tops)
    return {
        "n_perm": int(n_perm),
        "t_q025": float(np.quantile(t, 0.025)),
        "t_q975": float(np.quantile(t, 0.975)),
        "share_abs_t_ge_2": float((np.abs(t) >= T_EVIDENCE).mean()),
        "spread_t_q025": float(np.quantile(ts, 0.025)),
        "spread_t_q975": float(np.quantile(ts, 0.975)),
        "top_excess_t_q025": float(np.quantile(tt, 0.025)),
        "top_excess_t_q975": float(np.quantile(tt, 0.975)),
    }


# ============================================================================ C: calibration


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = RELIABILITY_BINS) -> list[dict[str, float]]:
    """Equal-count bins of the predicted probability: n, mean predicted, realised frequency, gap."""
    p = np.asarray(p, dtype="float64")
    y = np.asarray(y, dtype="float64")
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if p.size == 0:
        return []
    ranks = stats.rankdata(p, method="average")
    b = np.minimum(np.floor((ranks - 0.5) / p.size * n_bins).astype(int), n_bins - 1)
    out = []
    for k in range(n_bins):
        sel = b == k
        if not sel.any():
            continue
        pm, yr = float(p[sel].mean()), float(y[sel].mean())
        out.append(
            {
                "bin": k + 1,
                "n": int(sel.sum()),
                "p_min": float(p[sel].min()),
                "p_max": float(p[sel].max()),
                "pred": pm,
                "real": yr,
                "gap": yr - pm,
            }
        )
    return out


def calibration_summary(p: np.ndarray, y: np.ndarray, n_bins: int = RELIABILITY_BINS) -> dict[str, float]:
    """ECE over equal-count bins, mean predicted vs realised rate, and the logistic recalibration
    sigmoid(a + b*logit(p)) fitted by maximum likelihood (a ~ 0 and b ~ 1 when calibrated; b < 1 means the
    probabilities are too extreme, b > 1 too timid)."""
    p = np.asarray(p, dtype="float64")
    y = np.asarray(y, dtype="float64")
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if p.size < 10:
        return {
            "n": int(p.size),
            "ece": _nan(),
            "mean_pred": _nan(),
            "rate": _nan(),
            "slope": _nan(),
            "intercept": _nan(),
        }
    table = reliability_table(p, y, n_bins)
    ece = sum(r["n"] * abs(r["gap"]) for r in table) / p.size
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    lp = np.log(pc / (1 - pc))

    def nll(ab: np.ndarray) -> float:
        z = ab[0] + ab[1] * lp
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    slope = intercept = _nan()
    if np.ptp(lp) > 1e-9:
        res = optimize.minimize(nll, np.array([0.0, 1.0]), method="BFGS")
        if res.success or np.isfinite(res.fun):
            intercept, slope = float(res.x[0]), float(res.x[1])
    return {
        "n": int(p.size),
        "ece": float(ece),
        "mean_pred": float(p.mean()),
        "rate": float(y.mean()),
        "slope": slope,
        "intercept": intercept,
    }


def coverage_groups(
    frame: pd.DataFrame, group_col: str, labels: Mapping[Any, str] | None = None
) -> list[dict[str, Any]]:
    """80% band coverage and VaR95 exceedance (model and baseline twin) per group of OOS rows."""
    out = []
    for key, g in frame.groupby(group_col, sort=True, observed=True):
        out.append(
            {
                "group": (labels or {}).get(key, str(key)),
                "n": int(len(g)),
                "n_dates": int(g["date"].nunique()),
                "cover80": float(g["in80"].mean()),
                "base_cover80": float(g["base_in80"].mean()),
                "exc95": float(g["exc95"].mean()),
                "base_exc95": float(g["base_exc95"].mean()),
                "mean_sigma": float(g["sigma_total"].mean()) if "sigma_total" in g else _nan(),
            }
        )
    return out


def survivorship_diagnostic(
    train: pd.DataFrame, oos: pd.DataFrame, as_of: pd.Timestamp | None, horizon: int
) -> dict[str, float]:
    """How much excess drift TODAY's universe shows in the past (a survivorship / selection signature).

    The universe is today's top-N by market cap, i.e. selected on the outcome. On a null market (no alpha,
    no momentum) that selection alone gives the kept names a positive mean beta-adjusted excess return
    that grows the further back the date is (audit simulation 2026-09-25: top 60 of 600 by final cap over
    7 years ~ +13%/yr; top 150 of 1500 over 12 years ~ +12%/yr), and it pushes the 12-1 momentum IC DOWN
    (names that lagged early must have rebounded to be selected; IC t about -0.6 over 47 dates, -1.0 to
    -1.3 over 86-138 dates) while leaving the 1-month reversal IC unbiased.

    Returns (annualised = x 252/h): mean of the per-date cross-sectional mean y_e over the training dates
    and its t; the OLS slope of that per-date mean on 'years before as_of' (annualised drift per year back)
    and its t; and on the OOS rows the model's mean mu_e, the root-only twin's mean mu_e (the drift the
    root learns) and the realised mean y_e.
    """
    periods = TRADING_DAYS / float(horizon)
    out = {
        "n_dates": 0,
        "mean_ye_annual": _nan(),
        "mean_ye_t": _nan(),
        "slope_per_year_back": _nan(),
        "slope_t": _nan(),
        "oos_model_mu_e_annual": _nan(),
        "oos_base_mu_e_annual": _nan(),
        "oos_realised_ye_annual": _nan(),
    }
    if len(oos):
        out["oos_model_mu_e_annual"] = float(oos["mu_e"].mean() * periods)
        out["oos_base_mu_e_annual"] = float(oos["base_mu_e"].mean() * periods)
        out["oos_realised_ye_annual"] = float(oos["y_e"].mean() * periods)
    if train.empty or as_of is None:
        return out
    daily = train.groupby("date")["y_e"].mean().dropna()
    out["n_dates"] = int(daily.size)
    if daily.size < 3:
        return out
    out["mean_ye_annual"] = float(daily.mean() * periods)
    out["mean_ye_t"] = _t_of_mean(daily.to_numpy())
    back = ((pd.Timestamp(as_of) - pd.DatetimeIndex(daily.index)).days / 365.25).to_numpy(dtype="float64")
    if np.ptp(back) > 0:
        fit = stats.linregress(back, daily.to_numpy(dtype="float64"))
        out["slope_per_year_back"] = float(fit.slope * periods)
        out["slope_t"] = float(fit.slope / fit.stderr) if fit.stderr > 0 else _nan()
    return out


# ============================================================================ D: barrier replay


def stop_take(entry: float, atr: float, settings: Settings) -> tuple[float, float]:
    """(stop_pct < 0, take_pct > 0): stop distance = clip(stop_atr_mult*ATR, min_stop_pct*entry,
    max_stop_pct*entry); take-profit = min_reward_risk * stop distance (risk.* config).

    Raises:
        ValueError: non-finite / non-positive entry or ATR.
    """
    e, a = _fin(entry), _fin(atr)
    if not (e > 0 and a > 0):
        raise ValueError(f"stop_take needs entry > 0 and ATR > 0 (got {entry}, {atr})")
    r = settings.risk
    dist = min(max(r.stop_atr_mult * a, r.min_stop_pct * e), r.max_stop_pct * e)
    return -dist / e, r.min_reward_risk * dist / e


def realised_barrier(future_close: np.ndarray, entry: float, stop_pct: float, take_pct: float) -> tuple[int, float]:
    """First touch on DAILY CLOSES (the same rule as ReturnDistribution.barrier_trades): +1 take-profit first,
    -1 stop-loss first, 0 time exit at the last close; returns (outcome, simple return at the exit close)."""
    c = np.asarray(future_close, dtype="float64")
    if c.size == 0 or not np.isfinite(c).all():
        raise ValueError("realised_barrier needs a non-empty array of finite future closes")
    simple = c / float(entry) - 1.0
    hit_sl = simple <= stop_pct
    hit_tp = simple >= take_pct
    d = simple.size
    first_sl = int(hit_sl.argmax()) if hit_sl.any() else d
    first_tp = int(hit_tp.argmax()) if hit_tp.any() else d
    if first_tp < first_sl:
        return 1, float(simple[first_tp])
    if first_sl < first_tp:
        return -1, float(simple[first_sl])
    return 0, float(simple[-1])


_DIST_COLS = ("mu_e", "sigma_e", "mu_m", "sigma_m", "nu", "nu_m", "beta", "rho")


def predicted_barrier(
    params: Mapping[str, float],
    seed: int,
    horizon: int,
    hold: int,
    n_paths: int,
    stop_pct: float,
    take_pct: float,
) -> tuple[float, float, float, float]:
    """(p_tp, p_sl, p_time, E[trade]) of ONE row's ReturnDistribution (the serving class) over `hold` days."""
    dist = ReturnDistribution(
        *(float(params[k]) for k in _DIST_COLS[:6]),
        float(params["beta"]),
        float(params["rho"]),
        horizon,
        n_paths,
        seed,
        hold_days=hold,
    )
    b = dist.barrier(stop_pct, take_pct)
    return b.p_take_profit, b.p_stop_loss, b.p_time_exit, b.expected_trade_return


def barrier_frame(
    oos: pd.DataFrame,
    price_panel: PricePanel,
    settings: Settings,
    horizon: int,
    *,
    hold: int | None = None,
    n_paths: int = BARRIER_PATHS,
    workers: int = BARRIER_WORKERS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replay the stop-loss / take-profit plan on every OOS row: predicted probabilities from the row's
    model and baseline distributions, realised first touch on the next `hold` daily closes.

    Returns (frame, excluded counts {"no_future": rows without `hold` future bars, "no_atr": ATR
    unavailable at t, "no_prices": ticker not in the price panel}).
    """
    hold = int(hold if hold is not None else settings.fusion.target_horizon_days)
    excluded = {"no_future": 0, "no_atr": 0, "no_prices": 0}
    jobs: list[dict[str, Any]] = []
    for ticker, g in oos.groupby("ticker", sort=True):
        px = price_panel.prices.get(str(ticker))
        if px is None or px.empty:
            excluded["no_prices"] += len(g)
            continue
        close = px["close"].to_numpy(dtype="float64")
        atr_v = atr_series(px, settings.risk.atr_period).to_numpy(dtype="float64")  # causal: value at t uses <= t
        pos = pd.Series(np.arange(len(px)), index=px.index)
        for row in g.itertuples(index=False):
            d = pd.Timestamp(row.date)
            if d not in pos.index:
                excluded["no_prices"] += 1
                continue
            i = int(pos[d])
            if i + hold >= len(close):
                excluded["no_future"] += 1
                continue
            entry, a = close[i], atr_v[i]
            if not (math.isfinite(a) and a > 0):
                excluded["no_atr"] += 1
                continue
            sp, tp = stop_take(entry, a, settings)
            outcome, trade = realised_barrier(close[i + 1 : i + 1 + hold], entry, sp, tp)
            jobs.append(
                {
                    "date": d,
                    "ticker": str(ticker),
                    "fold": int(row.fold),
                    "stop_pct": sp,
                    "take_pct": tp,
                    "outcome": outcome,
                    "trade_ret": trade,
                    "seed": path_seed(str(ticker), d.date().isoformat()),
                    "model": {k: getattr(row, k) for k in _DIST_COLS},
                    "base": {
                        **{k: getattr(row, k) for k in ("mu_m", "sigma_m", "nu_m", "beta")},
                        "mu_e": row.base_mu_e,
                        "sigma_e": row.base_sigma_e,
                        "nu": row.base_nu,
                        "rho": row.base_rho,
                    },
                }
            )

    def run(chunk: list[dict[str, Any]]) -> list[tuple[float, ...]]:
        out = []
        for j in chunk:
            m = predicted_barrier(j["model"], j["seed"], horizon, hold, n_paths, j["stop_pct"], j["take_pct"])
            b = predicted_barrier(j["base"], j["seed"], horizon, hold, n_paths, j["stop_pct"], j["take_pct"])
            out.append((*m, *b))
        return out

    size = max(1, math.ceil(len(jobs) / max(1, workers * 4)))
    chunks = [jobs[i : i + size] for i in range(0, len(jobs), size)]
    if workers > 1 and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eval-barrier") as pool:
            results = [r for part in pool.map(run, chunks) for r in part]
    else:
        results = [r for c in chunks for r in run(c)]
    cols = ["p_tp", "p_sl", "p_time", "e_trade", "base_p_tp", "base_p_sl", "base_p_time", "base_e_trade"]
    frame = pd.DataFrame(
        [{k: j[k] for k in ("date", "ticker", "fold", "stop_pct", "take_pct", "outcome", "trade_ret")} for j in jobs],
        columns=["date", "ticker", "fold", "stop_pct", "take_pct", "outcome", "trade_ret"],
    )
    pred = pd.DataFrame(results, columns=cols) if results else pd.DataFrame(columns=cols)
    return pd.concat([frame, pred], axis=1), excluded


def _brier3(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    """Per-row 3-class Brier score of (p_tp, p_sl, p_time) against the realised outcome."""
    o = frame["outcome"].to_numpy()
    y = np.stack([(o == 1), (o == -1), (o == 0)], axis=1).astype("float64")
    p = frame[[f"{prefix}p_tp", f"{prefix}p_sl", f"{prefix}p_time"]].to_numpy(dtype="float64")
    return ((p - y) ** 2).sum(axis=1)


def barrier_summary(frame: pd.DataFrame, n_boot: int, seed: int, block_dates: int) -> dict[str, Any]:
    """Predicted vs realised TP / SL / time-exit frequencies, Brier (model vs root-only twin), AUCs,
    reliability bins and the expected vs realised trade return."""
    if frame.empty:
        return {"n": 0}
    dates = frame["date"].to_numpy()
    o = frame["outcome"].to_numpy()
    out: dict[str, Any] = {"n": int(len(frame)), "n_dates": int(frame["date"].nunique()), "block_dates": block_dates}
    for name, code, col in (("tp", 1, "p_tp"), ("sl", -1, "p_sl"), ("time", 0, "p_time")):
        real = (o == code).astype("float64")
        out[name] = {
            "real": _mean_ci(real, dates, n_boot, seed, block_dates),
            "pred": float(frame[col].mean()),
            "base_pred": float(frame["base_" + col].mean()),
        }
    b_model, b_base = _brier3(frame, ""), _brier3(frame, "base_")
    out["brier3"] = {
        "model": float(b_model.mean()),
        "base": float(b_base.mean()),
        "gain": _mean_ci(b_base - b_model, dates, n_boot, seed, block_dates),
    }
    for name, code in (("tp", 1), ("sl", -1)):
        y = (o == code).astype("float64")
        p, pb = frame[f"p_{name}"].to_numpy(dtype="float64"), frame[f"base_p_{name}"].to_numpy(dtype="float64")
        out[f"auc_{name}"] = {"model": _auc(p, y), "base": _auc(pb, y)}
        out[f"brier_{name}"] = {"model": float(np.mean((p - y) ** 2)), "base": float(np.mean((pb - y) ** 2))}
        out[f"reliability_{name}"] = reliability_table(p, y, BARRIER_BINS)
        out[f"calibration_{name}"] = calibration_summary(p, y, BARRIER_BINS)
    out["trade"] = {
        "pred": float(frame["e_trade"].mean()),
        "base_pred": float(frame["base_e_trade"].mean()),
        "real": _mean_ci(frame["trade_ret"].to_numpy(dtype="float64"), dates, n_boot, seed, block_dates),
    }
    out["stop_pct_mean"] = float(frame["stop_pct"].mean())
    out["take_pct_mean"] = float(frame["take_pct"].mean())
    return out


# ============================================================================ E: point-in-time factors


def _factor_score(subs: Sequence[Any]) -> float:
    """Factor score = mean of the available sub-scores clipped to [0, 100] (the quant.factors rule); NaN if none."""
    vals = [s.score for s in subs if s.score is not None]
    return float(np.clip(np.mean(vals), 0.0, 100.0)) if vals else _nan()


def point_in_time_factor_scores(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    dates: Sequence[pd.Timestamp],
    instrument: Instrument,
    atr_period: int = 14,
    pit_rows: int = PIT_ROWS,
) -> pd.DataFrame:
    """Momentum / volatility(stability) / trend scores at each date from bars <= date only.

    Each snapshot runs the production path (``compute_technicals`` -> ``momentum_subscores`` /
    ``volatility_subscores`` / ``trend_subscores``) on the last `pit_rows` bars up to the date and the
    benchmark up to the date. Fundamentals are empty on purpose (price factors only). A date that is not
    a bar of the stock or whose snapshot fails gets NaN scores and an `error` text (never dropped silently).
    """
    idx = prices.index
    bench = benchmark if benchmark is not None else pd.DataFrame()
    rows = []
    for d in dates:
        d = pd.Timestamp(d)
        row: dict[str, Any] = {"date": d, **{f: _nan() for f in PRICE_FACTORS}, "error": ""}
        if d not in idx:
            row["error"] = "가격 봉 없음"
            rows.append(row)
            continue
        i = int(idx.get_loc(d))
        sl = prices.iloc[max(0, i + 1 - pit_rows) : i + 1]
        b = bench.loc[sl.index[0] - pd.Timedelta(10, "D") : d] if not bench.empty else bench
        try:
            tech = compute_technicals(sl, b, atr_period)
            md = MarketData(instrument=instrument, prices=sl, benchmark=b, fundamentals=Fundamentals())
            row["momentum"] = _factor_score(momentum_subscores(tech))
            row["volatility"] = _factor_score(volatility_subscores(md, tech))
            row["trend"] = _factor_score(trend_subscores(md, tech))
        except ValueError as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:80]}"
        rows.append(row)
    return pd.DataFrame(rows, columns=["date", *PRICE_FACTORS, "error"])


def factor_frame(
    price_panel: PricePanel,
    train: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Point-in-time price factor scores on every (ticker, grid date) training row of the RSMT-Q panel,
    with the realised beta-adjusted excess return y_e and the price-only composite (factor weights of the
    three price factors, renormalised over the available ones).

    Sequential on purpose: the snapshots are many small pandas operations (GIL-bound); measured on 30
    names x 49 dates, 8.2 s sequential vs 11.1 s with 4 threads (~5.6 ms per snapshot).
    """
    market = price_panel.market

    def one(ticker: str) -> pd.DataFrame:
        px = price_panel.prices[ticker]
        bid = price_panel.ticker_benchmark.get(ticker, "")
        inst = Instrument(
            ticker=ticker,
            market=market,
            name=ticker,
            currency="KRW" if market is Market.KR else "USD",
            benchmark=bid or "-",
            benchmark_name=bid or "-",
        )
        dates = sorted(pd.DatetimeIndex(train.loc[train["ticker"] == ticker, "date"]).unique())
        out = point_in_time_factor_scores(px, price_panel.benchmarks.get(bid), dates, inst, settings.risk.atr_period)
        out.insert(0, "ticker", ticker)
        return out

    parts = [one(t) for t in sorted(train["ticker"].unique()) if t in price_panel.prices]
    cols = ["ticker", "date", *PRICE_FACTORS, "error"]
    fac = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    fac = fac.merge(train[["ticker", "date", "y_e"]], on=["ticker", "date"], how="left")
    w = {f: float(settings.factors.weights.get(f, 0.0)) for f in PRICE_FACTORS}
    num = sum(w[f] * fac[f].fillna(0.0) for f in PRICE_FACTORS)
    den = sum(w[f] * fac[f].notna() for f in PRICE_FACTORS)
    fac["price_composite"] = (num / den.where(den > 0)).astype("float64")
    return fac


# ================================================================================ evaluation


def _layout_metrics(res: rsmt.CVResult, k: int, n_boot: int, seed: int) -> dict[str, Any]:
    """Section A numbers of ONE fold layout (every gain = better-than-baseline is positive)."""
    o, m = res.oos, res.metrics
    up = (o["r"].to_numpy() > 0).astype("float64")
    dates = o["date"].to_numpy()
    eps = 0.5 / rsmt.CV_PATHS
    boot = rsmt.date_block_bootstrap(
        o["p_up"].to_numpy(), o["base_p_up"].to_numpy(), up, dates, n_boot, seed=seed, eps=eps
    )
    g = boot.n_dates
    p, pb = o["p_up"].to_numpy(dtype="float64"), o["base_p_up"].to_numpy(dtype="float64")
    per_row = np.column_stack(
        [
            (pb - up) ** 2 - (p - up) ** 2,  # Brier gain
            o["base_crps"].to_numpy(dtype="float64") - o["crps"].to_numpy(dtype="float64"),  # CRPS gain
            o["in80"].to_numpy(dtype="float64"),
            o["base_in80"].to_numpy(dtype="float64"),
            o["exc95"].to_numpy(dtype="float64"),
            o["base_exc95"].to_numpy(dtype="float64"),
        ]
    )
    hat, draws, gb = block_bootstrap_means(per_row, dates, n_boot, seed)

    def ci(j: int) -> list[float]:
        return list(ci90(hat[j], draws[:, j], gb)) if draws.size else [_nan(), _nan()]

    rows = [
        {
            "key": "auc",
            "label": "AUC P(상승)",
            "better": "high",
            "model": m["auc"],
            "base": m["base_auc"],
            "gain": boot.auc_gain_hat,
            "ci90": list(ci90(boot.auc_gain_hat, boot.auc_gain, g)),
        },
        {
            "key": "logloss",
            "label": "로그손실",
            "better": "low",
            "model": m["logloss"],
            "base": m["base_logloss"],
            "gain": boot.logloss_gain_hat,
            "ci90": list(ci90(boot.logloss_gain_hat, boot.logloss_gain, g)),
        },
        {
            "key": "brier",
            "label": "Brier",
            "better": "low",
            "model": m["brier"],
            "base": m["base_brier"],
            "gain": float(hat[0]),
            "ci90": ci(0),
        },
        {
            "key": "crps",
            "label": "CRPS(로그수익률)",
            "better": "low",
            "model": m["crps"],
            "base": m["base_crps"],
            "gain": float(hat[1]),
            "ci90": ci(1),
        },
    ]
    coverage = [
        {
            "key": "cover80",
            "label": "80% 구간 적중률",
            "target": 0.80,
            "model": float(hat[2]),
            "model_ci90": ci(2),
            "base": float(hat[3]),
            "base_ci90": ci(3),
        },
        {
            "key": "exc95",
            "label": "VaR95 초과율",
            "target": 0.05,
            "model": float(hat[4]),
            "model_ci90": ci(4),
            "base": float(hat[5]),
            "base_ci90": ci(5),
        },
    ]
    return {
        "folds": int(k),
        "available": True,
        "n_oos": int(len(o)),
        "n_dates": int(g),
        "test_start": str(pd.Timestamp(o["date"].min()).date()),
        "test_end": str(pd.Timestamp(o["date"].max()).date()),
        "seconds": float(res.seconds),
        "rows": rows,
        "coverage": coverage,
        "ic": {"model": m.get("ic", _nan()), "base": m.get("base_ic", _nan())},
        "auc_rel": {"model": m.get("auc_rel", _nan()), "base": m.get("base_auc_rel", _nan())},
        "gate_pass": bool(res.skill_ok),
        "gate": dict(res.gate),
        "direction_gate_pass": bool(res.direction_ok),
        "distribution_gate_pass": bool(res.distribution_ok),
        "dist_gate": dict(res.dist_gate),
        "crps_gain_p05": m.get("crps_gain_p05", _nan()),
        "auc_gain_p05": m.get("auc_gain_p05", _nan()),
        "n_mu_clamped": m.get("n_mu_clamped", _nan()),
        "delta": m.get("delta", _nan()),
        "splits_by_fold": {int(f["fold"]): int(f.get("n_splits", 0)) for f in res.folds if "skipped" not in f},
        "notes": list(res.notes),
    }


def _section_a(
    runs: Mapping[int, rsmt.CVResult], settings: Settings, folds: Sequence[int], n_boot: int, seed: int
) -> tuple[dict[str, Any], rsmt.CVResult]:
    primary = int(settings.rsmt.cv_folds)
    robust = rsmt.robust_fold_layouts(settings)
    layouts = []
    for k in folds:
        res = runs[k]
        if res.available and res.oos is not None:
            layouts.append(_layout_metrics(res, k, n_boot, seed))
        else:
            layouts.append({"folds": int(k), "available": False, "gate_pass": False, "notes": list(res.notes)})
    prod = rsmt.apply_robust_gate(
        copy.deepcopy(runs[primary]), {k: rsmt.layout_gate(runs[k], k, runs[k].seconds) for k in robust}
    )
    n_eval = len(layouts)
    n_eval_pass = sum(1 for lay in layouts if lay.get("gate_pass"))
    return (
        {
            "primary_folds": primary,
            "layouts": layouts,
            "production": {
                "layouts": robust,
                "pass": int(prod.robust_pass),
                "n": int(prod.robust_n),
                "need": int(prod.robust_n // 2 + 1),
                "skill_ok": bool(prod.skill_ok),
                "skill_ok_primary": bool(prod.skill_ok_primary),
                "direction_skill_ok": bool(prod.direction_ok),
                "distribution_skill_ok": bool(prod.distribution_ok),
                "direction_pass": int(prod.robust_pass_of("direction")),
                "distribution_pass": int(prod.robust_pass_of("distribution")),
            },
            "eval_majority": {
                "layouts": [int(k) for k in folds],
                "pass": n_eval_pass,
                "n": n_eval,
                "need": n_eval // 2 + 1,
                "skill_ok": bool(n_eval and 2 * n_eval_pass > n_eval),
            },
            "skill_min_auc_gain": float(settings.rsmt.skill_min_auc_gain),
            "require_date_ci": bool(settings.rsmt.skill_require_date_ci),
            "skill_metric": str(settings.rsmt.skill_metric),
            "cover80_band": list(settings.rsmt.skill_cover80_band),
            "exc95_band": list(settings.rsmt.skill_exc95_band),
        },
        prod,
    )


def oos_frame(res: rsmt.CVResult, pdata: PanelData, seed: int) -> pd.DataFrame:
    """Primary-layout OOS rows joined with the panel's row inputs (y_e, xs momentum z, ret_1m, regimes)."""
    cols = ["ticker", "date", "y_e", "z", "ret_1m", "mkt_ret_60", "mkt_vol_60", "vol_ratio_20_120"]
    tr = pdata.train[[c for c in cols if c in pdata.train.columns]]
    o = res.oos.merge(tr, on=["ticker", "date"], how="left", validate="one_to_one")
    if o["y_e"].isna().any():
        raise EvaluationError(f"OOS 행 {int(o['y_e'].isna().sum())}개가 패널 학습 행과 매칭되지 않음")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, 0x5C0E]))
    o["mu_excess"] = o["mu_e"]
    o["prob_outperform"] = o["p_out"]
    o["momentum"] = o["z"]
    o["reversal"] = -o["ret_1m"]
    o["random"] = rng.random(len(o))
    o["beat"] = (o["r"] > o["m"]).astype("float64")
    o["up"] = (o["r"] > 0).astype("float64")
    o["sigma_total"] = [
        total_log_sd(b, se, sm, rho, nu, nm, pdata.horizon)
        for b, se, sm, rho, nu, nm in zip(o["beta"], o["sigma_e"], o["sigma_m"], o["rho"], o["nu"], o["nu_m"])
    ]
    return o


def _section_b(
    o: pd.DataFrame, runs: Mapping[int, rsmt.CVResult], pdata: PanelData, folds: Sequence[int], seed: int
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    h = pdata.horizon
    per: dict[str, pd.DataFrame] = {}
    signals = []
    for key, label in RANK_SIGNALS.items():
        pdf = cross_sectional_ic(o, key)
        per[key] = pdf
        signals.append(
            {
                "key": key,
                "label": label,
                **ranking_stats(pdf, h),
                "ic_by_date": [[str(pd.Timestamp(d).date()), _fin(v)] for d, v in zip(pdf["date"], pdf["ic"])],
            }
        )
    paired = {
        "mu_excess_vs_momentum": paired_ic_difference(per["mu_excess"], per["momentum"]),
        "prob_outperform_vs_momentum": paired_ic_difference(per["prob_outperform"], per["momentum"]),
    }
    by_layout = []
    for k in folds:
        res = runs[k]
        if not (res.available and res.oos is not None):
            by_layout.append({"folds": int(k), "available": False})
            continue
        ok = res.oos.merge(pdata.train[["ticker", "date", "y_e"]], on=["ticker", "date"], how="left")
        ok["mu_excess"], ok["prob_outperform"] = ok["mu_e"], ok["p_out"]
        a = ranking_stats(cross_sectional_ic(ok, "mu_excess"), h)
        b = ranking_stats(cross_sectional_ic(ok, "prob_outperform"), h)
        by_layout.append({"folds": int(k), "available": True, "mu_excess": a, "prob_outperform": b})
    return (
        {
            "target": "y_e = r − β·m (베타조정 초과 로그수익률, h일)",
            "horizon": h,
            "signals": signals,
            "paired": paired,
            "by_layout": by_layout,
            "random_null": random_null_band(o, "y_e", RANDOM_NULL_PERMS, seed),
        },
        per,
    )


def _with_edge(p: pd.Series, delta: float) -> np.ndarray:
    """The served probability: sigmoid(logit(p) + delta) (distribution.apply_edge, CV path count)."""
    return np.array([apply_edge(float(v), float(delta), rsmt.CV_PATHS) for v in p], dtype="float64")


def _section_c(o: pd.DataFrame, res: rsmt.CVResult) -> dict[str, Any]:
    fold_info = {int(f["fold"]): f for f in res.folds}
    delta = _fin(res.metrics.get("delta", 0.0))
    delta_rel = _fin(res.metrics.get("delta_rel", 0.0))
    delta = delta if math.isfinite(delta) else 0.0
    delta_rel = delta_rel if math.isfinite(delta_rel) else 0.0
    o = o.copy()
    o["period"] = o["fold"].astype(int)
    period_labels = {
        k: f"검증 블록 {k} ({fold_info.get(k, {}).get('test_start', '?')}~)" for k in sorted(o["period"].unique())
    }
    q = o["mkt_vol_60"].quantile([1 / 3, 2 / 3]).to_numpy()
    o["vol_regime"] = np.where(o["mkt_vol_60"] <= q[0], 0, np.where(o["mkt_vol_60"] <= q[1], 1, 2))
    vol_labels = {
        0: f"시장 변동성 하위 1/3 (≤{q[0]:.0%})",
        1: f"시장 변동성 중간 1/3 ({q[0]:.0%}~{q[1]:.0%})",
        2: f"시장 변동성 상위 1/3 (>{q[1]:.0%})",
    }
    o["trend_regime"] = (o["mkt_ret_60"] > 0).astype(int)
    trend_labels = {0: "직전 60일 시장 하락", 1: "직전 60일 시장 상승"}
    leaf_rows = []
    for (fold, leaf), g in o.groupby(["fold", "leaf_id"], sort=True):
        if len(g) < LEAF_MIN_ROWS:
            continue
        path = fold_info.get(int(fold), {}).get("leaf_paths", {}).get(int(leaf), "")
        leaf_rows.append(
            {
                "group": f"{int(fold)}폴드 리프#{int(leaf)}",
                "path": path or "루트(분할 없음)",
                "n": int(len(g)),
                "n_dates": int(g["date"].nunique()),
                "cover80": float(g["in80"].mean()),
                "base_cover80": float(g["base_in80"].mean()),
                "exc95": float(g["exc95"].mean()),
                "base_exc95": float(g["base_exc95"].mean()),
                "mean_sigma": float(g["sigma_total"].mean()),
            }
        )
    n_small = int(sum(len(g) < LEAF_MIN_ROWS for _, g in o.groupby(["fold", "leaf_id"])))
    return {
        "p_up": {
            "model": calibration_summary(o["p_up"], o["up"]),
            "base": calibration_summary(o["base_p_up"], o["up"]),
            "served": calibration_summary(_with_edge(o["p_up"], delta), o["up"]),
            "edge": delta,
            "table": reliability_table(o["p_up"], o["up"]),
        },
        "p_out": {
            "model": calibration_summary(o["p_out"], o["beat"]),
            "base": calibration_summary(o["base_p_out"], o["beat"]),
            "served": calibration_summary(_with_edge(o["p_out"], delta_rel), o["beat"]),
            "edge": delta_rel,
            "table": reliability_table(o["p_out"], o["beat"]),
        },
        "coverage_period": coverage_groups(o, "period", period_labels),
        "coverage_vol": coverage_groups(o, "vol_regime", vol_labels),
        "coverage_trend": coverage_groups(o, "trend_regime", trend_labels),
        "coverage_leaf": leaf_rows,
        "leaf_groups_hidden": n_small,
        "overall": {
            "cover80": float(o["in80"].mean()),
            "base_cover80": float(o["base_in80"].mean()),
            "exc95": float(o["exc95"].mean()),
            "base_exc95": float(o["base_exc95"].mean()),
        },
    }


def _section_e(
    fac: pd.DataFrame, oos_start: pd.Timestamp | None, h: int
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    per: dict[str, pd.DataFrame] = {}
    rows = []
    for key, label in FACTOR_SIGNALS.items():
        full = cross_sectional_ic(fac, key)
        per[key] = full
        oos = full[full["date"] >= oos_start] if oos_start is not None else full.iloc[0:0]
        rows.append(
            {
                "key": key,
                "label": label,
                "full": ranking_stats(full, h),
                "oos_period": ranking_stats(oos, h),
                "coverage": float(fac[key].notna().mean()) if len(fac) else _nan(),
            }
        )
    errors = fac.loc[fac["error"].astype(bool), "error"] if len(fac) else pd.Series(dtype=str)
    return (
        {
            "signals": rows,
            "n_rows": int(len(fac)),
            "n_errors": int(len(errors)),
            "error_examples": sorted(set(errors.tolist()))[:3],
            "oos_start": str(oos_start.date()) if oos_start is not None else None,
            "excluded": "밸류에이션·퀄리티: 현재 시점 펀더멘털 스냅샷만 있어 시점별(point-in-time) 재현 불가 → 제외",
        },
        per,
    )


def _tier_from_t(t: float) -> str:
    if not math.isfinite(t):
        return TIER_NONE
    if t >= T_EVIDENCE:
        return TIER_EVIDENCE
    if t >= T_WEAK:
        return TIER_WEAK
    return TIER_NONE


_TIER_RANK = {TIER_NONE: 0, TIER_UNTESTABLE: 1, TIER_WEAK: 1, TIER_EVIDENCE: 2}


def recommend_pillar_weights(current: Mapping[str, float], tiers: Mapping[str, str]) -> dict[str, float]:
    """Recommended pillar weights (a stated policy; the config is never changed).

    - An untestable pillar keeps its current (normalised) weight: this evaluation has no evidence to raise
      or lower it.
    - A tested pillar without evidence is cut to current x TIER_MULT[tier] (weak 0.6, none 0.3).
    - The weight freed by those cuts goes ONLY to the pillars with evidence (TIER_EVIDENCE), in proportion
      to their current weights. With no such pillar there is nowhere to move it and the current weights
      are returned (the report then recommends stricter thresholds instead).

    Raises:
        ValueError: a pillar without a tier, an unknown tier or no positive current weight.
    """
    cur = {p: max(float(w), 0.0) for p, w in current.items()}
    total = sum(cur.values())
    if total <= 0:
        raise ValueError("recommend_pillar_weights: no positive weight")
    cur = {p: w / total for p, w in cur.items()}
    for p in cur:
        if tiers.get(p) not in TIER_MULT:
            raise ValueError(f"pillar {p!r}: unknown evidence tier {tiers.get(p)!r}")
    receivers = [p for p in cur if tiers[p] == TIER_EVIDENCE and cur[p] > 0]
    if not receivers:
        return cur
    out = dict(cur)
    freed = 0.0
    for p, w in cur.items():
        if tiers[p] in (TIER_WEAK, TIER_NONE):
            out[p] = w * TIER_MULT[tiers[p]]
            freed += w - out[p]
    base = sum(cur[p] for p in receivers)
    for p in receivers:
        out[p] = cur[p] + freed * cur[p] / base
    return out


def _reverse_note(rows: Sequence[Mapping[str, Any]], full: bool = True) -> str:
    """' ... 유의하게 역방향 ...' for signals whose IC t <= -T_EVIDENCE (a score that ranks names backwards is
    worse than no score: using it as-is loses money); empty otherwise."""
    bad = []
    for r in rows:
        t = _fin((r.get("full", {}) if full else r).get("t_ic"))
        if t <= -T_EVIDENCE:
            bad.append(f"{r.get('label', r.get('key', '?'))}(t={t:+.2f})")
    if not bad:
        return ""
    return (
        f" ⚠ 유의하게 역방향: {', '.join(bad)} — 이 기간에는 점수가 높은 종목이 오히려 부진했음"
        "(그대로 쓰면 손해; 표본 기간 특이성인지 추가 기간으로 재확인 필요)."
    )


def _slope_text(slope: float) -> str:
    """Korean reading of a logistic recalibration slope."""
    if not math.isfinite(slope):
        return "(재보정 기울기 산출 불가)."
    if slope < 0:
        return "(음수: 예측 확률의 높낮이가 실현과 반대 — 방향 정보 없음/역방향)."
    if slope < 1:
        return "(0~1: 확률이 과신 — 극단값을 믿으면 안 됨)."
    return "(1 초과: 확률이 50%에 너무 몰림)."


def _band_direction(ov: Mapping[str, Any]) -> str:
    """Korean note on which way the 80% band / VaR95 miss (empty when within tolerance)."""
    parts = []
    cover, exc = _fin(ov.get("cover80")), _fin(ov.get("exc95"))
    if cover > 0.80 + COVER_TOL:
        parts.append("80% 구간이 넓음(보수적)")
    elif cover < 0.80 - COVER_TOL:
        parts.append("80% 구간이 좁음(위험 과소평가)")
    if exc < 0.05 - EXC_TOL:
        parts.append("VaR95가 보수적(실제 손실이 VaR를 넘는 빈도가 5%보다 적음)")
    elif exc > 0.05 + EXC_TOL:
        parts.append("VaR95가 위험을 과소평가")
    return f" — {', '.join(parts)}" if parts else ""


def _bottom_line(
    sections: Mapping[str, Any], tiers: Mapping[str, str], dist_ok: bool, band_ok: bool, sd: Mapping[str, Any]
) -> list[str]:
    """The few sentences a CIO reads first (rule-based on the verdicts above)."""
    out = []
    tested = {p: t for p, t in tiers.items() if t != TIER_UNTESTABLE}
    if not any(t == TIER_EVIDENCE for t in tested.values()):
        out.append(
            "방향·종목 선택에 대해 표본 외 근거(t ≥ 2 또는 강건성 게이트 통과)를 가진 정량 필러가 없음 → 종합점수 기반 "
            "매수/매도 신호는 상당 부분 잡음일 수 있음. 가중 재배분보다 매수 문턱 상향·포지션 축소를 우선 검토(권고)."
        )
    else:
        good = [PILLAR_LABELS.get(p, p) for p, t in tested.items() if t == TIER_EVIDENCE]
        out.append(f"표본 외 근거가 있는 정량 필러: {', '.join(good)}.")
    risk_ok, uses = [], []
    if dist_ok:
        risk_ok.append("수익률 분포의 변동성 규모(CRPS가 기준보다 우수 — 개선 폭은 작음)")
        uses += ["VaR", "켈리 상한"]
    barrier_tier = barrier_verdict(sd)[0] if sd.get("n") else ""
    if barrier_tier == CAL_OK:
        risk_ok.append("손절·익절 선도달 확률(평균 수준과 종목별 순서 모두 실현과 일치)")
        uses.insert(0, "손절·익절 설정")
    if risk_ok:
        out.append(
            f"표본 외에서 확인된 항목: {' · '.join(risk_ok)} → 리스크 관리({', '.join(uses)}) 용도로는 사용 근거 있음"
            + ("." if band_ok else " (단, 구간·VaR 보정은 허용폭 밖 — 섹션 A·C 참조).")
        )
    if barrier_tier == CAL_LEVEL:
        out.append(
            "손절·익절 선도달 확률은 평균 빈도만 실현과 맞고 종목별 순서는 맞히지 못함(순위 AUC≈0.5, 재보정 기울기 "
            "허용폭 밖) → 포트폴리오 전체의 평균 손절 빈도 추정에는 쓸 수 있으나, 종목별 배리어 가드레일(매수 강등)의 "
            "근거로는 부족."
        )
    split_line = split_summary_line(sections.get("A", {}))
    if split_line:
        out.append(split_line)
    return out


def split_summary_line(sa: Mapping[str, Any]) -> str:
    """Korean sentence on how often the CV trees split, over EVERY evaluated fold layout (not only the
    primary one: '모든 CV 폴드에서 분할 없음' must not be claimed when another layout's fold split).
    Empty when some primary-layout fold split (the tree is then described in section A)."""
    layouts = [lay for lay in sa.get("layouts", []) if lay.get("available")]
    prim = next((lay for lay in layouts if lay.get("folds") == sa.get("primary_folds")), None)
    if prim is None:
        return ""
    prim_splits = list(prim.get("splits_by_fold", {}).values())
    if not prim_splits or any(prim_splits):
        return ""
    others = [
        (lay["folds"], {f: n for f, n in lay.get("splits_by_fold", {}).items() if n})
        for lay in layouts
        if lay is not prim
    ]
    split_elsewhere = [(k, d) for k, d in others if d]
    tail = "모형 = 루트 분포 + 모멘텀·변동성 오프셋(γ·z, η·v); 국면(레짐) 조건부 효과는 이 시장에서 검출되지 않음."
    if not split_elsewhere:
        names = ", ".join(str(lay["folds"]) for lay in layouts)
        return f"평가한 모든 폴드 배치({names})의 모든 CV 폴드에서 RSMT-Q 트리 분할이 없었음 → {tail}"
    detail = " · ".join(
        f"{k}폴드 배치 " + ", ".join(f"폴드{f} {n}회" for f, n in sorted(d.items())) for k, d in split_elsewhere
    )
    return (
        f"기준 {prim['folds']}폴드 배치의 모든 CV 폴드에서 트리 분할 없음(다른 배치에서는 {detail}) → 분할이 드물고 "
        f"폴드 배치마다 달라 안정적이지 않음. 사실상 {tail}"
    )


def multiple_testing_summary(sections: Mapping[str, Any]) -> dict[str, Any]:
    """Count the ranking t-statistics this report prints (B: IC / spread / top-minus-all t of every
    non-random signal and the per-layout IC t; E: full-period IC / spread t and the B-period IC t) and how
    many reach |t| >= 2, next to the ~5% of them expected by chance alone (the tests overlap, so this is a
    rough yardstick, not an exact family-wise rate). A lone |t| ~ 2 among dozens is not evidence."""
    hits: list[str] = []
    n = 0

    def add(label: str, t: Any) -> None:
        nonlocal n
        v = _fin(t)
        if math.isfinite(v):
            n += 1
            if abs(v) >= T_EVIDENCE:
                hits.append(f"{label} t={v:+.2f}")

    sb, se = sections.get("B", {}), sections.get("E", {})
    for s in sb.get("signals", []):
        if s.get("key") == "random":
            continue
        for key, what in (("t_ic", "IC"), ("spread_t", "스프레드"), ("top_excess_t", "상위−전체")):
            add(f"B {s.get('key')} {what}", s.get(key))
    for lay in sb.get("by_layout", []):
        if lay.get("available"):
            for sig in ("mu_excess", "prob_outperform"):
                add(f"B {lay['folds']}폴드 {sig} IC", lay.get(sig, {}).get("t_ic"))
    for s in se.get("signals", []):
        add(f"E {s.get('key')} IC", s.get("full", {}).get("t_ic"))
        add(f"E {s.get('key')} 스프레드", s.get("full", {}).get("spread_t"))
        add(f"E {s.get('key')} IC(B 기간)", s.get("oos_period", {}).get("t_ic"))
    return {"n_t": n, "n_abs_t_ge_2": len(hits), "expected_by_chance": 0.05 * n, "hits": hits}


def barrier_verdict(sd: Mapping[str, Any]) -> tuple[str, bool, bool]:
    """(tier, level_ok, shape_ok) of the section-D barrier probabilities.

    level_ok: |mean predicted - realised| <= BARRIER_TOL for P(TP first) and P(SL first).
    shape_ok: the logistic recalibration slope of BOTH probabilities lies in SLOPE_BAND (the per-row
    probabilities order the outcomes about as much as they claim; the same rule as the P(up) verdict).
    tier: CAL_OK when both hold, CAL_LEVEL when only the level holds, CAL_BAD otherwise. A matching
    average alone must not read as '보정 양호': with AUC ~0.5 the per-name numbers carry no information.
    """
    tp, sl = sd["tp"], sd["sl"]
    level_ok = abs(tp["pred"] - tp["real"]["hat"]) <= BARRIER_TOL and abs(sl["pred"] - sl["real"]["hat"]) <= BARRIER_TOL
    slopes = [_fin(sd.get(f"calibration_{k}", {}).get("slope")) for k in ("tp", "sl")]
    shape_ok = all(SLOPE_BAND[0] <= s <= SLOPE_BAND[1] for s in slopes)
    if level_ok and shape_ok:
        return CAL_OK, True, True
    return (CAL_LEVEL if level_ok else CAL_BAD), level_ok, shape_ok


def _sig(sections: Mapping[str, Any], sec: str, key: str) -> dict[str, Any]:
    return next((s for s in sections.get(sec, {}).get("signals", []) if s["key"] == key), {})


def _section_g(sections: Mapping[str, Any], settings: Settings) -> dict[str, Any]:
    """Plain-Korean verdict lines and the recommended pillar weights (rule-based on the numbers above)."""
    lines: list[dict[str, str]] = []
    sa, sc, sd = (sections.get(k, {}) for k in ("A", "C", "D"))

    # --- RSMT-Q direction (A)
    prod = sa.get("production", {})
    prim = next((lay for lay in sa.get("layouts", []) if lay.get("folds") == sa.get("primary_folds")), None)
    if prim is None:
        prim = next((lay for lay in sa.get("layouts", []) if lay.get("available")), {})
    auc_row = next((r for r in prim.get("rows", []) if r["key"] == "auc"), {}) if prim else {}
    crps_row = next((r for r in prim.get("rows", []) if r["key"] == "crps"), {}) if prim else {}
    # the DIRECTION verdict (the production gate may be the distribution gate: rsmt.skill_metric)
    d_ok = prod.get("direction_skill_ok", prod.get("skill_ok"))
    d_pass = prod.get("direction_pass", prod.get("pass", 0))
    a_tier = TIER_EVIDENCE if d_ok else (TIER_WEAK if d_pass > 0 else TIER_NONE)
    auc_ci = auc_row.get("ci90", [_nan(), _nan()])
    lines.append(
        {
            "topic": "RSMT-Q 방향 예측 P(상승) — 기준 모형 대비",
            "tier": a_tier,
            "text": (
                f"방향 폴드 강건성 게이트 {'통과' if d_ok else '미달'} "
                f"({d_pass}/{prod.get('n', 0)}, 과반 {prod.get('need', 0)} 필요). "
                f"기준 {sa.get('primary_folds')}폴드 AUC 개선 {_fin(auc_row.get('gain')):+.4f} "
                f"(90% CI {_fin(auc_ci[0]):+.4f} ~ {_fin(auc_ci[1]):+.4f})."
                + (
                    " 0을 포함하는 구간이면 루트 분포(무조건부)보다 낫다는 증거가 없음."
                    if not (_fin(auc_ci[0]) > 0)
                    else ""
                )
            ),
        }
    )
    # --- distribution quality (A/C)
    ov = sc.get("overall", {})
    crps_ci = crps_row.get("ci90", [_nan(), _nan()])
    dist_ok = _fin(crps_ci[0]) > 0
    crps_base = _fin(crps_row.get("base"))
    crps_rel_gain = _fin(crps_row.get("gain")) / crps_base if crps_base > 0 else _nan()
    band_ok = abs(_fin(ov.get("cover80")) - 0.80) <= COVER_TOL and abs(_fin(ov.get("exc95")) - 0.05) <= EXC_TOL
    lines.append(
        {
            "topic": "RSMT-Q 분포(구간·꼬리) — 기준 모형 대비",
            "tier": TIER_EVIDENCE if dist_ok else TIER_NONE,
            "text": (
                f"CRPS 개선 {_fin(crps_row.get('gain')):+.5f} "
                f"(90% CI {_fin(crps_ci[0]):+.5f} ~ {_fin(crps_ci[1]):+.5f}; 기준 CRPS 대비 "
                f"{crps_rel_gain:+.1%}"
                " — 기준 쌍둥이는 변동성 조정이 없는 루트 분포라, 이 개선에는 EWMA 변동성 조건화(η·v) 효과가 "
                "포함되며 방향·종목 선택 능력을 뜻하지 않음); "
                f"80% 구간 적중률 {_fin(ov.get('cover80')):.1%} (기준 {_fin(ov.get('base_cover80')):.1%}, 목표 80%), "
                f"VaR95 초과율 {_fin(ov.get('exc95')):.1%} (기준 {_fin(ov.get('base_exc95')):.1%}, 목표 5%) → "
                f"구간·꼬리 {CAL_OK if band_ok else CAL_BAD}{_band_direction(ov)}."
            ),
        }
    )
    # --- selection (B)
    mu = _sig(sections, "B", "mu_excess")
    po = _sig(sections, "B", "prob_outperform")
    mom = _sig(sections, "B", "momentum")
    paired = sections.get("B", {}).get("paired", {}).get("mu_excess_vs_momentum", {})
    b_tier = _tier_from_t(_fin(mu.get("t_ic")))
    lines.append(
        {
            "topic": "종목 선택(횡단면 순위) — RSMT-Q",
            "tier": b_tier,
            "text": (
                f"μ_excess 순위 IC 평균 {_fin(mu.get('mean_ic')):+.3f}, t={_fin(mu.get('t_ic')):+.2f} "
                f"({mu.get('n_dates', 0)}개 날짜, IC>0 비율 {_fin(mu.get('share_ic_pos')):.0%}); "
                f"상위−하위 5분위 {_fin(mu.get('spread_mean')):+.2%}/{sections.get('B', {}).get('horizon', '')}일 "
                f"(t={_fin(mu.get('spread_t')):+.2f}). P(BM 초과) IC t={_fin(po.get('t_ic')):+.2f}. "
                f"모멘텀 단독 IC t={_fin(mom.get('t_ic')):+.2f}; "
                f"μ_excess − 모멘텀 IC 차이 t={_fin(paired.get('t')):+.2f}." + _reverse_note([mu, po], full=False)
            ),
        }
    )
    # --- calibration (C)
    pu = sc.get("p_up", {}).get("model", {})
    served = sc.get("p_up", {}).get("served", {})
    cal_ok = _fin(pu.get("ece")) < ECE_TOL and SLOPE_BAND[0] <= _fin(pu.get("slope")) <= SLOPE_BAND[1]
    slope = _fin(pu.get("slope"))
    lines.append(
        {
            "topic": "P(상승) 확률 보정",
            "tier": CAL_OK if cal_ok else CAL_BAD,
            "text": (
                f"ECE {_fin(pu.get('ece')):.3f}, 평균 예측 {_fin(pu.get('mean_pred')):.1%} "
                f"vs 실현 {_fin(pu.get('rate')):.1%}, 재보정 기울기 {slope:.2f} "
                + _slope_text(slope)
                + f" 서빙 확률(δ 적용, 표본 내 1모수) ECE {_fin(served.get('ece')):.3f}."
            ),
        }
    )
    # --- barrier (D)
    if sd.get("n"):
        tp, sl = sd["tp"], sd["sl"]
        gap_sl = sl["pred"] - sl["real"]["hat"]
        gain = sd["brier3"]["gain"]
        d_tier, level_ok, shape_ok = barrier_verdict(sd)
        if gap_sl > BARRIER_TOL:
            impl = " 손절 확률을 과대 예측 → 배리어 가드레일(매수 강등)·켈리 비중이 보수적으로 작동."
        elif gap_sl < -BARRIER_TOL:
            impl = " 손절 확률을 과소 예측 → 배리어 가드레일·켈리 비중이 위험을 과소평가(낙관적)."
        elif not level_ok:
            impl = " 익절 확률의 예측-실현 차이가 허용폭을 넘음."
        else:
            impl = ""
        if level_ok and not shape_ok:
            impl += (
                " 평균 빈도는 맞지만 종목별 확률의 높낮이는 실현과 무관 → 3분류 Brier 개선은 평균 수준 맞춤에서 온 것."
            )
        cal_tp, cal_sl = sd.get("calibration_tp", {}), sd.get("calibration_sl", {})
        lines.append(
            {
                "topic": "손절·익절 선도달 확률(모듈 4)",
                "tier": d_tier,
                "text": (
                    f"P(익절 먼저) 예측 {tp['pred']:.1%} vs 실현 {tp['real']['hat']:.1%}, "
                    f"P(손절 먼저) 예측 {sl['pred']:.1%} vs 실현 {sl['real']['hat']:.1%} ({sd['n']}건). "
                    f"순위 AUC 익절 {_num(sd.get('auc_tp', {}).get('model'))} · 손절 "
                    f"{_num(sd.get('auc_sl', {}).get('model'))}, 재보정 기울기 {_num(cal_tp.get('slope'), 2)} · "
                    f"{_num(cal_sl.get('slope'), 2)} (허용 {SLOPE_BAND[0]:g}~{SLOPE_BAND[1]:g}). "
                    f"3분류 Brier 개선(기준 대비) {gain['hat']:+.4f} (90% CI {_fin(gain['ci90'][0]):+.4f} ~ "
                    f"{_fin(gain['ci90'][1]):+.4f}).{impl}"
                ),
            }
        )
    # --- price factors (E)
    e_rows = sections.get("E", {}).get("signals", [])
    comp = next((r for r in e_rows if r["key"] == "price_composite"), {})
    f_t = _fin(comp.get("full", {}).get("t_ic"))
    parts = ", ".join(f"{r['label']} t={_fin(r['full'].get('t_ic')):+.2f}" for r in e_rows)
    lines.append(
        {
            "topic": "가격 기반 팩터(모멘텀·변동성·추세)",
            "tier": _tier_from_t(f_t),
            "text": f"전체 기간 순위 IC: {parts}. 밸류·퀄리티는 시점별 데이터가 없어 검증 불가."
            + _reverse_note(e_rows),
        }
    )
    lines.append(
        {
            "topic": "LLM 뉴스·공시 필러",
            "tier": TIER_UNTESTABLE,
            "text": "과거 시점의 뉴스·공시 원문과 LLM 응답이 없어 이 평가로 검증할 수 없음(현재 가중은 정책 판단).",
        }
    )

    # --- recommended pillar weights
    ml_tier = max(b_tier, a_tier, key=lambda t: _TIER_RANK[t])
    tiers = {"factor": _tier_from_t(f_t), "ml": ml_tier, "llm": TIER_UNTESTABLE}
    cur = dict(settings.fusion.weights)
    for p in cur:
        tiers.setdefault(p, TIER_UNTESTABLE)
    rec = recommend_pillar_weights(cur, tiers)
    basis = {
        "factor": f"가격 팩터 합성 IC t={f_t:+.2f} (밸류·퀄리티 미검증)",
        "ml": f"μ_excess IC t={_fin(mu.get('t_ic')):+.2f} · 강건성 게이트 {prod.get('pass', 0)}/{prod.get('n', 0)}",
        "llm": "과거 시점 텍스트 없음",
    }
    weights = [
        {
            "pillar": p,
            "label": PILLAR_LABELS.get(p, p),
            "current": float(cur[p]),
            "tier": tiers[p],
            "basis": basis.get(p, ""),
            "recommended": float(rec[p]),
        }
        for p in cur
    ]
    bottom = _bottom_line(sections, tiers, dist_ok, band_ok, sd)
    return {
        "lines": lines,
        "bottom_line": bottom,
        "multiple_testing": multiple_testing_summary(sections),
        "weights": weights,
        "rule": (
            "검증 불가 필러는 현재 가중 유지(근거 없이 올리거나 내리지 않음). 근거가 약하거나 없는 필러는 "
            "현재 가중 × 배수(약함 0.6 · 없음 0.3)로 줄이고, 줄어든 가중은 '근거 있음' 필러에만 현재 가중 "
            "비율로 배분. '근거 있음' 필러가 없으면 재배분할 곳이 없으므로 가중을 바꾸지 않음. "
            "t ≥ 2는 근거 있음, 1 ≤ t < 2는 약함. 설정 파일은 변경하지 않음."
        ),
        "ml_multiplier_note": (
            "운영 중인 ML 필러는 스킬 게이트 미달 시 이미 ×0.35로 축소됨 — 위 권장은 필러 가중 자체에 대한 것."
        ),
    }


def evaluate_panel(
    price_panel: PricePanel,
    settings: Settings,
    *,
    folds: Sequence[int] | None = None,
    barrier_paths: int = BARRIER_PATHS,
    n_boot: int | None = None,
    seed: int = 0,
    workers: int = 4,
    progress: Callable[[str], None] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> EvaluationResult:
    """Run sections A-E and G on a loaded price panel (see the module docstring; F was removed).

    Args:
        folds: fold layouts for section A (default rsmt.robust_cv_folds). The primary layout
            (rsmt.cv_folds) and the production robust layouts are always run as well: B-D use the
            primary layout's OOS rows and the served verdict needs the robust ones.
        n_boot: bootstrap draws for every CI (default rsmt.boot_sims).

    Raises:
        EvaluationError: the panel cannot be built or the primary CV produced no OOS rows.
        ValueError: a fold layout < 1.
    """
    say = progress or (lambda _m: None)
    timings: dict[str, float] = {}
    notes: list[str] = []
    t_all = time.perf_counter()
    n_boot = int(settings.rsmt.boot_sims if n_boot is None else n_boot)
    eval_folds = sorted({int(k) for k in (folds if folds is not None else rsmt.robust_fold_layouts(settings))})
    if not eval_folds or min(eval_folds) < 1:
        raise ValueError(f"--folds 는 1 이상의 정수 목록이어야 합니다: {folds}")
    primary = int(settings.rsmt.cv_folds)
    robust = rsmt.robust_fold_layouts(settings)
    all_folds = sorted({*eval_folds, *robust, primary})

    t0 = time.perf_counter()
    try:
        pdata = build_panel(price_panel, settings)
    except ValueError as exc:
        raise EvaluationError(f"패널 구성 실패: {exc}") from exc
    timings["panel"] = round(time.perf_counter() - t0, 2)
    h = pdata.horizon
    if pdata.failed_tickers:
        notes.append(
            f"가격 로드/패널 실패 {len(pdata.failed_tickers)}/{pdata.n_tickers}종목: "
            + ", ".join(f"{t}({e[:30]})" for t, e in list(pdata.failed_tickers.items())[:8])
        )
    if pdata.dropped.get("total"):
        notes.append(
            f"패널 학습 후보 {pdata.n_candidates}행 중 {pdata.dropped['total']}행 제외(결측·라벨 구간): "
            + ", ".join(f"{k} {v}" for k, v in pdata.dropped.items() if k != "total")
        )
    notes.extend(pdata.notes)

    # ---- A: CV layouts
    say(f"A) RSMT-Q 교차검증 {len(all_folds)}개 폴드 배치({', '.join(map(str, all_folds))}) 실행 중")
    t0 = time.perf_counter()
    runs = rsmt.cross_validate_layouts(pdata, settings, all_folds)
    res = runs[primary]
    if not (res.available and res.oos is not None and len(res.oos)):
        raise EvaluationError(f"기준 {primary}폴드 CV 결과 없음: {'; '.join(res.notes) or '사유 미상'}")
    sections: dict[str, dict[str, Any]] = {}
    sections["A"], prod_cv = _section_a(runs, settings, eval_folds, n_boot, seed)
    timings["cv"] = round(time.perf_counter() - t0, 2)  # the section-A summary (bootstraps) belongs to this stage
    for k in all_folds:
        for n in runs[k].notes:
            notes.append(f"{k}폴드 CV: {n}")

    # current model (final fit on every training row) for context: tree print, drift, served verdict
    t0 = time.perf_counter()
    try:
        model = rsmt.fit_market_model(pdata, settings, cv=prod_cv)
        sections["A"]["model_lines"] = [*model.tree.describe()[:12], model.drift_line(), model.robust_line()]
    except Exception as exc:  # noqa: BLE001 - context only, reported
        log.exception("evaluation: final model fit failed")
        notes.append(f"최종 모형 적합 실패(맥락 정보만 누락): {type(exc).__name__}: {str(exc)[:120]}")
    timings["final_fit"] = round(time.perf_counter() - t0, 2)

    # ---- B / C
    say("B) 횡단면 순위 IC · C) 확률 보정 계산 중")
    t0 = time.perf_counter()
    o = oos_frame(res, pdata, seed)
    sections["B"], per_b = _section_b(o, runs, pdata, eval_folds, seed)
    sections["C"] = _section_c(o, res)
    # date-level statistics use the non-overlapping sample grid (rsmt.overlap_step adds dependent dates)
    grid_train = on_grid_rows(pdata.train)
    sections["C"]["survivorship"] = survivorship_diagnostic(grid_train, o, pdata.as_of, h)
    timings["ranking_calibration"] = round(time.perf_counter() - t0, 2)

    # ---- D: barrier replay
    hold = int(settings.fusion.target_horizon_days)
    say(f"D) 손절·익절 재현: OOS {len(o)}행 × 경로 {barrier_paths} × {hold}일")
    t0 = time.perf_counter()
    bf, excluded = barrier_frame(o, price_panel, settings, h, hold=hold, n_paths=barrier_paths, workers=workers)
    block = max(1, math.ceil(hold / max(1, settings.rsmt.sample_step)))
    sections["D"] = {
        **barrier_summary(bf, n_boot, seed, block),
        "hold_days": hold,
        "n_paths": int(barrier_paths),
        "excluded": excluded,
        "rule": (
            f"진입=t일 종가, 손절폭=clip({settings.risk.stop_atr_mult:g}×ATR{settings.risk.atr_period}, "
            f"{settings.risk.min_stop_pct:.0%}, {settings.risk.max_stop_pct:.0%}), "
            f"익절=진입+{settings.risk.min_reward_risk:g}×손절폭, 종가 기준 {hold}거래일"
        ),
    }
    if sum(excluded.values()):
        notes.append(
            f"D) 제외 {sum(excluded.values())}행: 미래 {hold}봉 부족 {excluded['no_future']} · ATR 없음 "
            f"{excluded['no_atr']} · 가격 없음 {excluded['no_prices']}"
        )
    timings["barrier"] = round(time.perf_counter() - t0, 2)

    # ---- E: point-in-time price factors
    say(f"E) 시점별 가격 팩터 계산 중 ({len(grid_train)}행)")
    t0 = time.perf_counter()
    fac = factor_frame(price_panel, grid_train, settings)
    oos_start = pd.Timestamp(o["date"].min())
    sections["E"], per_e = _section_e(fac, oos_start, h)
    if sections["E"]["n_errors"]:
        notes.append(f"E) 팩터 스냅샷 실패 {sections['E']['n_errors']}행: {'; '.join(sections['E']['error_examples'])}")
    timings["factors"] = round(time.perf_counter() - t0, 2)

    sections["G"] = _section_g(sections, settings)
    total = round(time.perf_counter() - t_all, 2)
    # Work between the timed stages (setup, notes, section G) gets its own bucket so the parts add up to the total.
    timings["other"] = max(round(total - sum(timings.values()), 2), 0.0)
    timings["total"] = total

    grid = pd.DatetimeIndex(pdata.grid_dates)
    info = {
        "market": price_panel.market.value,
        "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "version": __version__,
        "config_digest": settings.digest(),
        "as_of": str(pdata.as_of.date()) if pdata.as_of is not None else None,
        "horizon_days": h,
        "sample_step": int(settings.rsmt.sample_step),
        "history_years": int(settings.history_years),
        "universe_size": int(settings.rsmt.markets[price_panel.market.value].universe_size),
        "n_tickers_panel": len(pdata.tickers),
        "n_failed": len(pdata.failed_tickers),
        "n_train_rows": int(len(pdata.train)),
        "n_grid_dates": int(len(grid)),
        "first_train_date": str(pd.Timestamp(pdata.train["date"].min()).date()),
        "oos_start": str(oos_start.date()),
        "oos_end": str(pd.Timestamp(o["date"].max()).date()),
        "n_oos_rows": int(len(o)),
        "n_oos_dates": int(o["date"].nunique()),
        "folds_evaluated": eval_folds,
        "n_boot": n_boot,
        "barrier_paths": int(barrier_paths),
        **dict(meta or {}),
    }
    frames = {"oos": o, "barrier": bf, "factors": fac}
    frames.update({f"ic_{k}": v for k, v in per_b.items()})
    frames.update({f"factor_ic_{k}": v for k, v in per_e.items()})
    log.info("evaluation %s done in %.1fs %s", price_panel.market.value, timings["total"], timings)
    return EvaluationResult(price_panel.market.value, info, sections, notes, timings, frames)


# ================================================================================== rendering


def _pct(x: Any, d: int = 1, signed: bool = True) -> str:
    v = _fin(x)
    if not math.isfinite(v):
        return "—"
    return f"{v * 100:+.{d}f}%" if signed else f"{v * 100:.{d}f}%"


def _num(x: Any, d: int = 3, signed: bool = False) -> str:
    v = _fin(x)
    if not math.isfinite(v):
        return "—"
    return f"{v:+.{d}f}" if signed else f"{v:.{d}f}"


def _ci_text(ci: Sequence[Any] | None, d: int = 4, pct: bool = False) -> str:
    if not ci or len(ci) != 2:
        return "—"
    lo, hi = _fin(ci[0]), _fin(ci[1])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "—"
    if pct:
        return f"[{lo * 100:.1f}%, {hi * 100:.1f}%]"
    return f"[{lo:+.{d}f}, {hi:+.{d}f}]"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _mark(lo: float, hi: float) -> str:
    """✔ when the 90% CI of a gain is above 0, ✘ below 0, · when it straddles 0."""
    if math.isfinite(lo) and lo > 0:
        return "✔ 개선"
    if math.isfinite(hi) and hi < 0:
        return "✘ 악화"
    return "· 차이 불확실"


def render_markdown(result: EvaluationResult) -> str:
    """Korean, human-readable report of every section."""
    m, S = result.meta, result.sections
    sa, sb, sc, sd, se, sg = (S.get(k, {}) for k in "ABCDEG")
    md: list[str] = []
    listed = m.get("universe_listed")
    pit_mode = m.get("universe_mode") == "pit"
    uni = (
        f"유니버스 요청 {m['universe_size']} → 제공 목록 {listed}종목"
        if isinstance(listed, int) and listed < int(m["universe_size"]) and not pit_mode
        else f"유니버스 {m['universe_size']}종목"
    )
    uni = f"{'시점별(PIT) ' if pit_mode else '현재 목록 '}{uni}" if m.get("universe_mode") else uni
    md.append(f"# QuantCIO 신호 평가 리포트 — {m['market']} ({uni} · {m['history_years']}년)")
    md.append("")
    md.append(
        f"- 생성 {m['generated_at']} · 데이터 기준일 {m['as_of']} · 설정 digest `{m['config_digest']}` · "
        f"quantcio {m['version']} · 소요 {result.timings.get('total', 0):.0f}초"
    )
    md.append(
        f"- 패널 {m['n_tickers_panel']}종목(실패 {m['n_failed']}) · 학습 행 {m['n_train_rows']:,} "
        f"({m['first_train_date']}~) · 표본 간격 {m['sample_step']}거래일(겹치지 않는 {m['horizon_days']}일 창)"
    )
    md.append(
        f"- 표본 외(OOS) 검증 기간 {m['oos_start']} ~ {m['oos_end']} · {m['n_oos_dates']}개 날짜 × 평균 "
        f"{m['n_oos_rows'] / max(1, m['n_oos_dates']):.0f}종목 = {m['n_oos_rows']:,}행 "
        f"(기준 {sa.get('primary_folds')}폴드 확장 윈도 CV)"
    )
    md.append("")
    md.append("## 핵심 요약")
    md.append("")
    for b in sg.get("bottom_line", []):
        md.append(f"> {b}")
        md.append(">")
    if sg.get("bottom_line"):
        md.pop()
        md.append("")
    for ln in sg.get("lines", []):
        md.append(f"- **{ln['topic']}** — {ln['tier']}: {ln['text']}")
    md.append("")
    md.append("## 읽는 법과 한계")
    md.append("")
    md.append(
        "- 모든 수치는 **표본 외(OOS)** 입니다. RSMT-Q 행은 각 폴드에서 검증 구간 이전에 라벨이 끝나는 행으로만 학습한 "
        "모형의 예측이며(퍼지), 시점 t의 입력(특징·베타·ATR·팩터 점수)은 t 이하 데이터만 씁니다."
    )
    md.append(
        "- t-통계량은 날짜 간 평균의 t입니다(20거래일 간격 → 수익률 창이 겹치지 않아 날짜끼리 거의 독립). "
        "|t| ≥ 2가 대략 5% 유의수준입니다. 90% 신뢰구간은 날짜(블록) 부트스트랩, 소표본 보정 포함."
    )
    if pit_mode:
        md.append(
            f"- **유니버스**: 시점별(PIT) — 매 월말 그 시점 정보로 정한 시가총액 상위 {m['universe_size']}종목"
            f"(상장폐지·편출 종목 포함, 폐지 종목은 실제 마지막 가격까지; {m.get('pit_start')}~, 엔티티 "
            f"{m.get('pit_entities')}개, 리밸런싱 {m.get('pit_rebalances')}회). 각 날짜의 학습·평가 행은 그 날짜 "
            "구성종목만 사용하므로 오늘의 승자만 고르는 생존·선택 편향이 없습니다(F 섹션만 최근 대형주 선택)."
        )
        md.append(
            "- **남아 있는 편향**: ① 하이퍼파라미터(트리 α·리프 크기·드리프트 사전·폴드 배치 등)는 개발 과정에서 같은 "
            "기간 데이터를 보고 정함. ② 한국 가격은 배당 미반영."
        )
    else:
        md.append(
            "- **남아 있는 편향**: ① 유니버스가 *오늘* 기준 시가총액 상위 종목(생존·선택 편향) — 과거 승자만 남아 "
            "평균 초과수익(드리프트)이 부풀려지고(무알파·무모멘텀 시장 시뮬레이션에서 선택만으로 연 +12~13%p), "
            "반대로 12-1 모멘텀 IC는 아래로 치우침(같은 시뮬레이션에서 IC t 약 −0.6~−1.3: 초기에 부진했던 종목은 "
            "이후 반등해야만 오늘 상위에 들 수 있음; 1개월 반전 IC는 영향 없음). ② 하이퍼파라미터(트리 α·리프 크기·"
            "드리프트 사전·폴드 배치 등)는 개발 과정에서 같은 기간 데이터를 보고 정함. ③ 한국 가격은 배당 미반영"
            "(네이버 수정주가). `--universe-mode pit`로 시점별 유니버스 평가 가능."
        )
    hg = {k: v for k, v in (m.get("holdout_guard") or {}).items() if v}
    if hg:
        md.append(
            "- **홀드아웃 제외**: 등록된 홀드아웃을 평가 전에 제거 — "
            + ", ".join(f"{k} {v:,}" for k, v in hg.items())
            + " (docs/HOLDOUT.md)"
        )
    sv = sc.get("survivorship", {})
    if sv.get("n_dates"):
        md.append(
            f"- **생존 편향 진단(이 표본)**: 오늘 유니버스의 학습 기간 날짜별 평균 초과수익 y_e 연 "
            f"{_pct(sv.get('mean_ye_annual'), 1)} (t {_num(sv.get('mean_ye_t'), 2, True)}, {sv['n_dates']}개 날짜); "
            f"과거로 1년 갈수록 연 {_pct(sv.get('slope_per_year_back'), 1)}씩 변함(t "
            f"{_num(sv.get('slope_t'), 2, True)}; 양수 = 먼 과거일수록 오늘의 승자가 더 앞섰음 → 선택 편향 신호). "
            f"OOS 행: 기준 쌍둥이(루트)가 학습한 초과 드리프트 연 {_pct(sv.get('oos_base_mu_e_annual'), 1)}, 모형 μ_e "
            f"연 {_pct(sv.get('oos_model_mu_e_annual'), 1)}, 실현 y_e 연 {_pct(sv.get('oos_realised_ye_annual'), 1)}."
        )
    mt = sg.get("multiple_testing", {})
    if mt.get("n_t"):
        md.append(
            f"- **다중검정**: 이 리포트의 순위 t-통계량(B·E) {mt['n_t']}개 중 |t| ≥ 2는 {mt['n_abs_t_ge_2']}개 — "
            f"신호가 "
            f"전혀 없어도 약 {mt['expected_by_chance']:.1f}개는 우연히 나옴(검정끼리 겹치므로 대략치). 사전에 정한 주 "
            "지표(μ_excess IC, 가격 팩터 합성 IC, 강건성 게이트)만 판정에 쓰며, 그 밖의 |t| ≈ 2 하나는 근거로 보지 "
            "않음." + (f" (|t| ≥ 2: {'; '.join(mt['hits'][:6])})" if mt.get("hits") else "")
        )
    md.append("")

    # ---------------- A
    md.append("## A. RSMT-Q 교차검증 vs 기준 쌍둥이(루트 분포·오프셋 없음)")
    md.append("")
    prod, ev = sa.get("production", {}), sa.get("eval_majority", {})
    dist_metric = sa.get("skill_metric") == "distribution"
    # the directional ML-pillar multiplier reads the DIRECTION verdict (quant.distribution_ml_pillar)
    dir_ok = prod.get("direction_skill_ok", prod.get("skill_ok"))
    lays = ", ".join(map(str, prod.get("layouts", [])))
    if dist_metric:
        mult = 1.0 if dir_ok else 0.35
        tail = f"· 방향 게이트 {'통과' if dir_ok else '미달'} → 자체 평균 서빙 시 ML 필러 배수 ×{mult:g}"
    else:
        tail = f"→ 서빙 ML 필러 배수 ×{1.0 if prod.get('skill_ok') else 0.35:g}"
    md.append(
        f"- **운영 스킬 게이트{'(분포)' if dist_metric else ''}(폴드 배치 {lays} 과반)**: "
        f"{'통과' if prod.get('skill_ok') else '미달'} {prod.get('pass')}/{prod.get('n')} (과반 "
        f"{prod.get('need')} 필요) {tail}"
    )
    if dist_metric:
        c_lo, c_hi = sa.get("cover80_band", [0.75, 0.85])
        e_lo, e_hi = sa.get("exc95_band", [0.02, 0.08])
        rule = (
            f"CRPS 개선 날짜 부트스트랩 5% 하한 > 0 · cover80 ∈ [{c_lo:g}, {c_hi:g}] · VaR95 초과율 ∈ [{e_lo:g}, "
            f"{e_hi:g}] (방향 AUC·로그손실은 보고만)"
        )
    else:
        rule = (
            f"AUC 개선 ≥ {sa.get('skill_min_auc_gain', 0):.2f} · 로그손실 개선 > 0 · 날짜 부트스트랩 AUC 개선 "
            f"5% 하한 > 0{'' if sa.get('require_date_ci') else '(미요구)'}"
        )
    md.append(
        f"- 평가 배치({', '.join(map(str, ev.get('layouts', [])))}) 과반: {'통과' if ev.get('skill_ok') else '미달'} "
        f"{ev.get('pass')}/{ev.get('n')} · 단일 배치 게이트 = {rule}"
    )
    md.append("")
    for lay in sa.get("layouts", []):
        k = lay["folds"]
        if not lay.get("available"):
            md.append(f"### {k}폴드 배치 — CV 불가: {'; '.join(lay.get('notes', [])) or '사유 미상'}")
            md.append("")
            continue
        md.append(
            f"### {k}폴드 배치 — 게이트 {'통과' if lay['gate_pass'] else '미달'} "
            f"(OOS {lay['n_oos']:,}행 · {lay['n_dates']}개 날짜 · {lay['test_start']}~{lay['test_end']} · "
            f"{lay['seconds']:.1f}s)"
        )
        md.append("")
        rows = []
        for r in lay["rows"]:
            lo, hi = _fin(r["ci90"][0]), _fin(r["ci90"][1])
            d = 5 if r["key"] == "crps" else 4
            rows.append(
                [
                    r["label"],
                    _num(r["model"], d),
                    _num(r["base"], d),
                    _num(r["gain"], d, True),
                    _ci_text(r["ci90"], d),
                    _mark(lo, hi),
                ]
            )
        md += _table(["지표", "RSMT-Q", "기준", "개선(+가 좋음)", "90% CI", "판정"], rows)
        md.append("")
        rows = []
        for r in lay["coverage"]:
            rows.append(
                [
                    r["label"],
                    _pct(r["model"], 1, False),
                    _ci_text(r["model_ci90"], pct=True),
                    _pct(r["base"], 1, False),
                    _ci_text(r["base_ci90"], pct=True),
                    _pct(r["target"], 0, False),
                ]
            )
        md += _table(["구간 지표", "RSMT-Q", "90% CI", "기준", "90% CI", "목표"], rows)
        md.append("")
        md.append(
            f"순위 IC(풀링, μ vs r) {_num(lay['ic']['model'], 3, True)} (기준 {_num(lay['ic']['base'], 3, True)}) · "
            f"초과수익 AUC {_num(lay['auc_rel']['model'])} vs {_num(lay['auc_rel']['base'])} · "
            f"게이트 AUC 하한(5%) {_num(lay['auc_gain_p05'], 4, True)} · μ 상한 적용 "
            f"{int(_fin(lay['n_mu_clamped']) or 0)}행 · 폴드별 트리 분할 수 "
            + (", ".join(f"{f}:{n}" for f, n in lay.get("splits_by_fold", {}).items()) or "—")
        )
        md.append("")
    if sa.get("model_lines"):
        md.append("<details><summary>현재 적합 모형(전체 학습 행) — 트리·드리프트·게이트</summary>")
        md.append("")
        md.append("```")
        md += sa["model_lines"]
        md.append("```")
        md.append("</details>")
        md.append("")

    # ---------------- B
    md.append("## B. 횡단면 종목 선택(순위) — CIO의 핵심 질문")
    md.append("")
    md.append(
        f"각 OOS 날짜에서 종목을 신호로 순위화하고, 실현 **{sb.get('target', '')}**와의 스피어만 순위 IC를 계산. "
        "5분위 스프레드 = 상위 20% 평균 − 하위 20% 평균(동일가중, 거래비용 제외)."
    )
    md.append("")
    rows = []
    for s in sb.get("signals", []):
        rows.append(
            [
                s["label"],
                _num(s["mean_ic"], 3, True),
                _num(s["t_ic"], 2, True),
                _pct(s["share_ic_pos"], 0, False),
                _pct(s["spread_mean"], 2),
                _num(s["spread_t"], 2, True),
                _pct(s["spread_annual"], 1),
                _num(s["spread_sharpe"], 2, True),
                f"{_pct(s.get('top_excess_mean'), 2)} (t {_num(s.get('top_excess_t'), 2, True)})",
                s["n_dates"],
            ]
        )
    md += _table(
        [
            "신호",
            "평균 IC",
            "IC t",
            "IC>0 날짜",
            f"5분위 스프레드({sb.get('horizon')}일)",
            "스프레드 t",
            "연율(단순)",
            "연율 샤프",
            "상위 5분위−전체",
            "날짜",
        ],
        rows,
    )
    md.append("")
    flat = [s["label"] for s in sb.get("signals", []) if not s["n_dates"]]
    if flat:
        md.append(
            f"- 산출 불가: {', '.join(flat)} — 모든 OOS 날짜에서 종목 간 값이 같음(폴드 트리에 분할이 없거나 "
            "시장 변수로만 분할) → 횡단면 순위 정보 없음."
        )
    n_max = max((int(s["n_dates"]) for s in sb.get("signals", [])), default=0)
    sparse = [s for s in sb.get("signals", []) if 0 < int(s["n_dates"]) < 0.8 * n_max]
    for s in sparse:
        md.append(
            f"- 주의: '{s['label']}' 행은 {s['n_dates']}개 날짜만 산출(다른 행 최대 {n_max}개) — 트리가 종목을 가른 "
            "폴드의 날짜만 남은 다른 표본이므로 다른 행과 직접 비교하면 안 됨."
        )
    md.append("- '상위 5분위−전체' = 롱온리 관점: 상위 20% 평균 초과수익 − 같은 날 전체 평균(t는 날짜 간).")
    nb = sb.get("random_null", {})
    pr = sb.get("paired", {})
    md.append(
        f"- 무작위 순위 {nb.get('n_perm')}회의 우연 범위(95%): IC t [{_num(nb.get('t_q025'), 2, True)}, "
        f"{_num(nb.get('t_q975'), 2, True)}] · 스프레드 t [{_num(nb.get('spread_t_q025'), 2, True)}, "
        f"{_num(nb.get('spread_t_q975'), 2, True)}] · 상위−전체 t [{_num(nb.get('top_excess_t_q025'), 2, True)}, "
        f"{_num(nb.get('top_excess_t_q975'), 2, True)}] — 이 범위 안의 t는 우연과 구분되지 않음. '기준: 무작위' 행은 "
        "이 분포에서 뽑은 한 번의 표본이라 범위 밖(약 5% 확률)으로 나와도 오류가 아님."
    )
    mv = pr.get("mu_excess_vs_momentum", {})
    pv = pr.get("prob_outperform_vs_momentum", {})
    md.append(
        f"- 모멘텀 대비 증분(날짜별 IC 차이): μ_excess − 모멘텀 {_num(mv.get('mean_diff'), 3, True)} "
        f"(t={_num(mv.get('t'), 2, True)}), "
        f"P(BM 초과) − 모멘텀 {_num(pv.get('mean_diff'), 3, True)} (t={_num(pv.get('t'), 2, True)}). "
        "μ_excess에는 모멘텀 오프셋 γ·z가 들어 있어 두 신호가 겹칩니다."
    )
    lay_parts = []
    for bl in sb.get("by_layout", []):
        if bl.get("available"):
            lay_parts.append(
                f"{bl['folds']}폴드 μ_excess t={_num(bl['mu_excess']['t_ic'], 2, True)} · "
                f"P(BM 초과) t={_num(bl['prob_outperform']['t_ic'], 2, True)}"
            )
    if lay_parts:
        md.append("- 폴드 배치별 강건성: " + " / ".join(lay_parts))
    md.append("")

    # ---------------- C
    md.append("## C. 확률 보정(calibration)")
    md.append("")
    md.append("P는 방향 엣지 δ 적용 **전** 원시 확률입니다(δ는 이 OOS 행들로 적합되므로 적용하면 표본 내가 됨).")
    md.append("")
    for key, title in (("p_up", "P(상승) = P(r > 0)"), ("p_out", "P(BM 초과) = P(r > m)")):
        blk = sc.get(key, {})
        mm, bb = blk.get("model", {}), blk.get("base", {})
        md.append(
            f"**{title}** — ECE {_num(mm.get('ece'))} (기준 {_num(bb.get('ece'))}) · 평균 예측 "
            f"{_pct(mm.get('mean_pred'), 1, False)} "
            f"vs 실현 {_pct(mm.get('rate'), 1, False)} · 재보정 기울기 {_num(mm.get('slope'), 2)} "
            f"(기준 {_num(bb.get('slope'), 2)}) · 절편 {_num(mm.get('intercept'), 2, True)}"
        )
        md.append("")
        sv = blk.get("served", {})
        md.append(
            f"서빙 확률(방향 엣지 δ={_num(blk.get('edge'), 3, True)} 적용 — δ가 이 행들로 적합된 1모수라 표본 내): "
            f"ECE {_num(sv.get('ece'))} · 평균 예측 {_pct(sv.get('mean_pred'), 1, False)} · "
            f"재보정 기울기 {_num(sv.get('slope'), 2)}"
        )
        md.append("")
        rows = [
            [
                r["bin"],
                r["n"],
                f"{r['p_min']:.2f}~{r['p_max']:.2f}",
                _pct(r["pred"], 1, False),
                _pct(r["real"], 1, False),
                _pct(r["gap"], 1),
            ]
            for r in blk.get("table", [])
        ]
        md += _table(["십분위", "행", "예측 범위", "평균 예측", "실현 빈도", "실현−예측"], rows)
        md.append("")
    md.append("**80% 구간 적중률 · VaR95 초과율 (그룹별; 목표 80% · 5%)**")
    md.append("")
    grp_rows = []
    for title, key in (("기간", "coverage_period"), ("시장 변동성", "coverage_vol"), ("시장 추세", "coverage_trend")):
        for r in sc.get(key, []):
            grp_rows.append(
                [
                    title,
                    r["group"],
                    r["n"],
                    _pct(r["cover80"], 1, False),
                    _pct(r["base_cover80"], 1, False),
                    _pct(r["exc95"], 1, False),
                    _pct(r["base_exc95"], 1, False),
                ]
            )
    md += _table(
        ["구분", "그룹", "행", "80% 적중(모형)", "80% 적중(기준)", "VaR95 초과(모형)", "VaR95 초과(기준)"], grp_rows
    )
    md.append("")
    if sc.get("coverage_leaf"):
        md.append(
            f"**리프별** (행 ≥ {LEAF_MIN_ROWS}; 리프 번호는 폴드마다 다른 트리의 것, 소규모 "
            f"{sc.get('leaf_groups_hidden', 0)}개 생략)"
        )
        md.append("")
        rows = [
            [
                r["group"],
                r["path"],
                r["n"],
                _pct(r["mean_sigma"], 1, False),
                _pct(r["cover80"], 1, False),
                _pct(r["base_cover80"], 1, False),
                _pct(r["exc95"], 1, False),
            ]
            for r in sc["coverage_leaf"]
        ]
        md += _table(["리프", "경로", "행", "평균 σ_h", "80% 적중", "기준", "VaR95 초과"], rows)
        md.append("")

    # ---------------- D
    md.append("## D. 손절·익절 선도달 확률 검증(모듈 4)")
    md.append("")
    if sd.get("n"):
        md.append(
            f"규칙: {sd['rule']}. 예측 = 해당 행의 OOS 분포(경로 {sd['n_paths']}개, 방향 엣지 미적용), 실현 "
            "= 실제 이후 "
            f"{sd['hold_days']}거래일 종가로 첫 도달 판정. 평균 손절 {_pct(sd['stop_pct_mean'])} · 익절 "
            f"{_pct(sd['take_pct_mean'])}. "
            f"{sd['n']:,}건({sd['n_dates']}개 날짜); 신뢰구간은 {sd['block_dates']}개 연속 날짜 블록 부트스트랩"
            f"(60일 창이 겹치므로)."
        )
        md.append("")
        rows = []
        for key, label in (("tp", "익절 먼저"), ("sl", "손절 먼저"), ("time", "시간 청산")):
            blk = sd[key]
            rows.append(
                [
                    label,
                    _pct(blk["pred"], 1, False),
                    _pct(blk["base_pred"], 1, False),
                    _pct(blk["real"]["hat"], 1, False),
                    _ci_text(blk["real"]["ci90"], pct=True),
                ]
            )
        tr = sd["trade"]
        rows.append(
            [
                "기대 거래수익률",
                _pct(tr["pred"], 2),
                _pct(tr["base_pred"], 2),
                _pct(tr["real"]["hat"], 2),
                _ci_text([_fin(tr["real"]["ci90"][0]), _fin(tr["real"]["ci90"][1])], pct=True),
            ]
        )
        md += _table(["결과", "예측(RSMT-Q)", "예측(기준)", "실현", "실현 90% CI"], rows)
        md.append("")
        g3 = sd["brier3"]
        md.append(
            f"- 3분류 Brier: 모형 {_num(g3['model'], 4)} vs 기준 {_num(g3['base'], 4)} → 개선 "
            f"{_num(g3['gain']['hat'], 4, True)} "
            f"{_ci_text(g3['gain']['ci90'])} {_mark(_fin(g3['gain']['ci90'][0]), _fin(g3['gain']['ci90'][1]))}"
        )
        md.append(
            f"- 순위 능력 AUC: 익절 {_num(sd['auc_tp']['model'])} (기준 {_num(sd['auc_tp']['base'])}) · "
            f"손절 {_num(sd['auc_sl']['model'])} (기준 {_num(sd['auc_sl']['base'])})"
        )
        md.append("")
        for key, label in (("tp", "P(익절 먼저)"), ("sl", "P(손절 먼저)")):
            cal = sd[f"calibration_{key}"]
            md.append(
                f"**{label} 신뢰도 5분위** — ECE {_num(cal.get('ece'))}, 재보정 기울기 {_num(cal.get('slope'), 2)}"
            )
            md.append("")
            rows = [
                [r["bin"], r["n"], _pct(r["pred"], 1, False), _pct(r["real"], 1, False), _pct(r["gap"], 1)]
                for r in sd[f"reliability_{key}"]
            ]
            md += _table(["5분위", "건수", "평균 예측", "실현 빈도", "실현−예측"], rows)
            md.append("")
    else:
        md.append("평가 가능한 행 없음.")
        md.append("")

    # ---------------- E
    md.append("## E. 가격 기반 팩터 백테스트(시점별, point-in-time)")
    md.append("")
    md.append(
        f"각 그리드 날짜에서 그날까지의 가격만으로 `compute_technicals` → 팩터 하위 점수(운영 코드 그대로)를 계산, "
        f"다음 {m['horizon_days']}일 베타조정 초과수익(y_e)과 순위 IC. {se.get('excluded', '')}"
    )
    md.append("")
    rows = []
    for s in se.get("signals", []):
        f_, o_ = s["full"], s["oos_period"]
        rows.append(
            [
                s["label"],
                _num(f_["mean_ic"], 3, True),
                _num(f_["t_ic"], 2, True),
                _pct(f_["share_ic_pos"], 0, False),
                _pct(f_["spread_mean"], 2),
                _num(f_["spread_t"], 2, True),
                _pct(f_["spread_annual"], 1),
                f_["n_dates"],
                _num(o_["mean_ic"], 3, True),
                _num(o_["t_ic"], 2, True),
            ]
        )
    md += _table(
        [
            "팩터",
            "평균 IC",
            "IC t",
            "IC>0",
            "5분위 스프레드",
            "t",
            "연율",
            "날짜",
            f"IC(B 기간 {se.get('oos_start')}~)",
            "t(B 기간)",
        ],
        rows,
    )
    md.append("")
    md.append(
        "해석: 팩터 점수는 '높을수록 좋음'으로 설계됨(변동성 점수가 높을수록 안정적). 음의 t는 이 기간에 "
        "반대로 작동했다는 뜻."
    )
    md.append("")

    # ---------------- G
    md.append("## G. 결론 — 근거가 있는 것과 없는 것")
    md.append("")
    for ln in sg.get("lines", []):
        md.append(f"- **{ln['topic']}** ({ln['tier']}): {ln['text']}")
    md.append("")
    if sg.get("bottom_line"):
        md.append("### 한 줄 결론")
        md.append("")
        md += [f"- {b}" for b in sg["bottom_line"]]
        md.append("")
    md.append("### 권장 필러 가중치 (권고만 — 설정은 변경하지 않음)")
    md.append("")
    rows = [
        [w["label"], _pct(w["current"], 0, False), w["tier"], w["basis"], _pct(w["recommended"], 0, False)]
        for w in sg.get("weights", [])
    ]
    md += _table(["필러", "현재", "근거 등급", "근거", "권장"], rows)
    md.append("")
    md.append(f"- 규칙: {sg.get('rule', '')}")
    md.append(f"- {sg.get('ml_multiplier_note', '')}")
    md.append("")
    if result.notes:
        md.append("## 알림(제외·실패 — 누락 없이 기록)")
        md.append("")
        md += [f"- {n}" for n in result.notes]
        md.append("")
    md.append("## 소요 시간")
    md.append("")
    md.append(" · ".join(f"{k} {v:.1f}s" for k, v in result.timings.items()))
    md.append("")
    return "\n".join(md)


# ================================================================================== entry point


def universe_shortfall_note(market: Market, requested: int, listed: int) -> str:
    """Korean note when the provider lists fewer universe names than requested (e.g. the static US list has
    107 names: --universe 150 used to run on 107 with '실패 0' and no word about the other 43). Empty when
    the list is complete. This is not a load failure, so it is not counted in '실패'."""
    if int(listed) >= int(requested):
        return ""
    return (
        f"유니버스 요청 {int(requested)}종목 중 {market.value} 제공 목록은 {int(listed)}종목뿐 — "
        f"{int(requested) - int(listed)}종목은 목록 상한으로 빠짐(로드 실패가 아님; 실제 평가 유니버스 "
        f"{int(listed)}종목)"
    )


def output_paths(out_dir: Path, market: Market, universe: int, years: int, mode: str = "current") -> tuple[Path, Path]:
    """``evaluation_<M>_<N>u_<Y>y.md|json`` for the current-list universe, ``..._pit`` for the PIT universe."""
    stem = f"evaluation_{market.value}_{int(universe)}u_{int(years)}y" + ("_pit" if mode == "pit" else "")
    return out_dir / f"{stem}.md", out_dir / f"{stem}.json"


def write_outputs(result: EvaluationResult, md_path: Path, json_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(result.to_json(), ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")


def run_evaluation(
    market: Market,
    settings: Settings,
    *,
    universe: int | None = None,
    years: int | None = None,
    folds: Sequence[int] | None = None,
    out_dir: Path | None = None,
    cache: DiskCache | None = None,
    use_cache: bool = True,
    progress: Callable[[str], None] | None = None,
    universe_mode: str | None = None,
) -> EvaluationArtifacts:
    """Load the market universe, remove registered holdouts, run `evaluate_panel`, write the .md and .json.

    ``universe_mode``: "pit" (point-in-time top-N per month-end incl. delisted names; default when
    ``pit.universe_mode == "pit"``) or "current" (today's top-N / static list, cached for the day; survivorship
    biased). Registered holdouts are removed from either panel (``holdout.guard_panel``) and reported in the notes.

    Raises:
        EvaluationError: nothing could be loaded or the primary CV produced no OOS rows.
        ValueError: invalid universe / years / folds / universe mode.
    """
    from quantcio import holdout

    s = eval_settings(settings, market, universe, years)
    mode = resolve_universe_mode(s, universe_mode)
    say = progress or (lambda _m: None)
    n = s.rsmt.markets[market.value].universe_size
    cache = cache or DiskCache(s.cache_path)
    label = "시점별(PIT)" if mode == "pit" else "현재 목록"
    say(f"{market.value} 유니버스 {label} {n}종목 · {s.history_years}년 가격 로드 중")
    t0 = time.perf_counter()
    pit_info: dict[str, Any] = {}
    if mode == "pit":
        panel, order, pit_info = load_pit_eval_panel(market, s, cache)
        cached = False
    else:
        panel, order, cached = load_eval_panel(market, s, cache, use_cache=use_cache)
    try:
        panel, guard = holdout.guard_panel(panel, s)
    except holdout.HoldoutError as exc:  # an unreadable registry must stop the run: it could leak held-out data
        raise EvaluationError(f"홀드아웃 레지스트리 확인 실패: {exc}") from exc
    if not panel.prices:
        raise EvaluationError(f"{market.value} 가격 패널이 비어 있음 (실패 {len(panel.failed)}종목: {panel.failed})")
    load_s = round(time.perf_counter() - t0, 2)
    say(f"가격 {len(panel.prices)}종목 로드({'캐시' if cached else '신규'}, {load_s:.1f}s; 실패 {len(panel.failed)})")
    shortfall = universe_shortfall_note(market, n, len(order)) if mode == "current" else ""
    if shortfall:
        log.warning("evaluation %s: %s", market.value, shortfall)
        say(shortfall)
    result = evaluate_panel(
        panel,
        s,
        folds=folds,
        progress=progress,
        meta={
            "panel_cached": cached,
            "load_seconds": load_s,
            "universe_listed": len(order),
            "universe_mode": mode,
            "holdout_guard": dict(guard),
            **pit_info,
        },
    )
    guard_notes = [x for x in panel.notes if x.startswith("[홀드아웃]")]
    for gn in reversed(guard_notes):
        result.notes.insert(0, gn)
    if shortfall:
        result.notes.insert(0, shortfall)
    # 'total' = the whole run including the price load (the evaluate_panel stages alone were reported as the
    # total before, e.g. 178 s in the .md vs 259.5 s on the console for KR 150 x 12y)
    result.timings = {"load": load_s, **result.timings}
    result.timings["total"] = round(float(result.timings.get("total", 0.0)) + load_s, 2)
    root = Path(out_dir) if out_dir is not None else Path(s.report.output_dir) / OUT_SUBDIR
    md_path, json_path = output_paths(root, market, n, s.history_years, mode)
    write_outputs(result, md_path, json_path)
    return EvaluationArtifacts(md_path, json_path, result)


# ============================================================================ holdout hooks (quantcio holdout ...)


def xsh_candidate_names(market: Market, settings: Settings, cache: DiskCache) -> list[str]:
    """Never-tuned names for a cross-sectional holdout (``quantcio.xsh.xsh_candidate_names``)."""
    from quantcio.xsh import xsh_candidate_names as candidates

    return candidates(market, settings, cache)


def run_xsh_evaluation(entry: Mapping[str, Any], settings: Settings, cache: DiskCache) -> dict[str, Any]:
    """One look at a registered cross-sectional holdout (``quantcio.xsh.run_xsh_evaluation``)."""
    from quantcio.xsh import run_xsh_evaluation as run

    return run(entry, settings, cache, progress=lambda m: log.info("xsh: %s", m))
