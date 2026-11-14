"""Nested, purged, expanding walk-forward model selection - PROTOCOL v2 (declared 2026-09-27 before any v2 run).

Protocol v2 (pre-declared here, docs/MODELING.md "검증 프로토콜 v2"):

0. PRIMARY METRIC (``metrics.PRIMARY_AUC`` = "auc_median"): the within-date AUC of the score against the per-date
   MEDIAN label 1{y_e > median_date(y_e)} - balanced classes on every date, the target of a long/short
   stock-selection book. v1 used the AUC against 1{y_e > 0}; it and the rank IC stay reported as SECONDARY
   metrics. POST HOC: this switch was decided AFTER the v1 results were seen (v1 showed the median-label AUC
   as a diagnostic, higher than the v1 primary), so every gate / report carries ``post_hoc: True`` and a gate
   pass under v2 needs CONFIRMATION ON UNTOUCHED DATA before full serving (step 7).
1. Evaluation dates = labelled month-end rebalance dates (h = 20-session labels barely overlap across months).
   The LAST ``HOLDOUT_MONTHS`` (24) of them are the final HOLDOUT: never used for any choice. It was already
   looked at in v1, so it is reported but is no longer untouched data (not a confirmation).
2. Outer walk-forward on the pre-holdout dates: test blocks of ``OUTER_BLOCK_MONTHS`` (12) month-ends starting
   after ``OUTER_MIN_TRAIN_MONTHS`` (36). Training rows of a fold have date <= the session 2h before the test
   start (purge: their label windows end >= h sessions = the embargo before the test starts).
3. Inside every outer training window: ``INNER_BLOCKS`` (2) inner validation blocks of 12 month-ends (the last
   two years), purged the same way. Stage 1 picks each candidate's grid point with scheme 'none'; stage 2 tries
   the other training-weight schemes (events.SCHEMES) at that grid point. Inner score = mean PRIMARY AUC
   (ties -> the rank IC, then 'none', then the earlier grid point).
4. The chosen configuration is refitted on the whole outer training window and predicts the test block.
   RSMT-Q v2 and v3 (``candidates.STACKED``, each with its own checkpoint) are stacked on the best inner-CV
   candidate's walk-forward out-of-sample score; the ensemble is the
   rank average of the top-3 candidates by inner score; ``blend`` is the FIXED rank average of logit and
   lgbm_clf (members pre-declared, each tuned by its own inner CV). Baselines are refitted walk-forward with no
   tuning.
5. Holdout: every candidate is tuned by inner CV on the pre-holdout data only, fitted once on all pre-holdout
   rows (purged) and frozen for the whole 24-month holdout.
6. SELECTION RULE (``select_best``): the candidate with the highest mean PRIMARY AUC over the pre-holdout outer
   out-of-sample dates; a difference below ``TIE_TOL`` is a tie, broken by the mean rank IC, then by the mean
   top-quintile net excess. Baselines are eligible.
7. Serving gate (``serving_gate``): the selected model's pre-holdout mean PRIMARY AUC is higher than the 12-1
   momentum baseline's AND the 95% block-bootstrap lower bound of its mean PRIMARY AUC is > 0.5. Serving
   (``modeling.serve``): the full calibrated excess forecast only when the gate passed AND the result is
   confirmed on untouched data - looked up AT SCORING TIME via ``quantcio.holdout.xsh_confirmation(market,
   settings)`` (the cross-sectional holdout: KR names ranked 201-400 never in the top-200 modeling universe,
   US names outside the S&P 500). Gate passed but unconfirmed -> x ``UNSKILLED_SHRINK`` and flagged; gate
   failed -> x ``UNSKILLED_SHRINK`` and flagged unskilled.
8. One v2 run per market (features v2 + candidates v2 declared in docs/MODELING.md before it). At most one
   revision round, only for a clearly broken feature (a bug, an all-NaN column) - never to improve the outer-CV
   numbers. Every configuration run is reported.

PROTOCOL v3 (docs/MODELING.md 16, declared 2026-10-03 before the v3 run) keeps steps 0-7 and changes:
- candidates v3 (``candidates.default_candidates``): + enet_logit, rf, et, xgb_reg, xgb_clf, xgb_clf_w, xgb_rank,
  cat_clf, cat_rank, mlp; - rsmt_v2. Composites (``candidates.composite_rule``): ensemble (top 3), ensemble5,
  ensemble_w (weights max(inner AUC - 0.5, 0); v4: max(inner primary, 0)), family_blend, blend.
- features v3 (``modeling.altfeatures``): alternative data + per-stock macro exposures + vix_term.
- diagnostics (``Candidate.diagnostic``: logit_w_core, lgbm_clf_core): evaluated like every candidate but never
  selected, never a composite member / stack base; ``alt_value`` reports the paired per-date primary-AUC
  difference full-input model - core-input model.
- confirmation: a NEW untouched cross-sectional holdout (KR PIT ranks 401-600, never in the top 400), registered
  before the run and looked at once after it; US: forward holdout only.

PROTOCOL v4 (docs/MODELING.md 79, declared 2026-10-25 before the v4 run - POST HOC, after the v3 results and the
decile / hedged backtests of docs 77-78 were seen) keeps steps 1-5 and changes:
- label (dataset v5): y_e = SIMPLE beta-adjusted excess (e^r - 1) - beta (e^m - 1); the log label made the rank
  metrics reward low-volatility picks that earned no mean return (78.1).
- PRIMARY METRIC (``metrics.PRIMARY`` = "top_ye_net"): per date, mean y_e of the score's top quintile minus the
  date's mean y_e, minus one-way turnover x the round-trip cost (``ModelingDataset.cost_bps``) - the net excess of
  "top quintile + index beta hedge". The AUCs and the rank IC are secondary. No skill = 0 (``PRIMARY_NULL``).
- inner score = mean PRIMARY over the inner validation dates (ties -> rank IC); composites rank / weight members
  by it (``ensemble_w``: max(inner - PRIMARY_NULL, 0)); training-weight scheme ``magnitude`` (events.SCHEMES).
- selection: highest mean outer PRIMARY, ties (< ``TIE_TOL``, per month) -> rank IC -> raw top-quintile net.
- gate: PRIMARY(selected) > PRIMARY(mom_12_1) on the same dates AND its 95% block-bootstrap lower bound > 0.
- confirmation: forward holdout only (the XSH names 401-600 were looked at for v3; the changed code / config
  fingerprints void the v3 confirmation automatically).

PROTOCOL v4.3 (docs/MODELING.md 84, declared 2026-10-29 before the run - post hoc, after the 83.3 rerun flipped the KR
pick on revised data) keeps v4 and changes step 3: every tuning step uses the 1-standard-error rule toward the
simplest configuration - grid candidates keep the simplest grid point (``Candidate.simplicity``) within one SE of the
best with scheme 'none', then keep 'none' unless it is more than one SE below the best scheme; BO candidates prefer
scheme 'none', then ``Candidate.simplicity``. The dataset reads the inputs that sources revise through frozen
first-seen vintages (``quantcio.data.vintage``) and traded value is the day's own (``value`` column).

Every context is checkpointed to the disk cache (resumable after an interruption).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantcio.config import Settings
from quantcio.data.cache import DiskCache
from quantcio.modeling import metrics as mt
from quantcio.modeling.candidates import (
    COMPOSITE_CANDIDATES,
    COMPOSITES,
    STACKED,
    Candidate,
    FitContext,
    Fitted,
    composite_rule,
    default_candidates,
)
from quantcio.modeling.dataset import ModelingDataset
from quantcio.modeling.features import rank_gauss, uniform_rank
from quantcio.modeling.metrics import PRIMARY, PRIMARY_AUC, PRIMARY_NULL

log = logging.getLogger(__name__)

SELECTION_VERSION = 4
PROTOCOL_VERSION = 4
POST_HOC_LABEL = True  # v4 (like v2) was decided after earlier results were seen (module docstring)
HOLDOUT_MONTHS = 24
OUTER_MIN_TRAIN_MONTHS = 36
OUTER_BLOCK_MONTHS = 12
INNER_BLOCKS = 2
INNER_BLOCK_MONTHS = 12
INNER_MIN_TRAIN_MONTHS = 12
STACK_MIN_TRAIN_MONTHS = 24
ENSEMBLE_K = 3  # ``ensemble`` = top 3 (``candidates.composite_rule``)
TIE_TOL = 0.0002  # v4: per-month net excess (v3: 0.0005 AUC)
BO_INNER_BLOCKS = 3  # docs/MODELING.md 82: the *_bo candidates validate on three inner annual blocks
BO_SEED = 20261026
BO_STARTUP = 10  # random trials before the TPE model
UNSKILLED_SHRINK = 0.35
BASELINE = "mom_12_1"
CTX_NS = "modeling_ctx"
RSMT_NS = "modeling_rsmt"
RSMT = "rsmt_v3"  # the stacked candidate (STACKED)
DIAGNOSTIC_PAIRS = (("logit_w", "logit_w_core"), ("lgbm_clf", "lgbm_clf_core"))  # alt_value: full - core inputs
RESULT_NS = "modeling_selection"
SELECTION_RULE = {
    "protocol_version": PROTOCOL_VERSION,
    "primary_metric": PRIMARY,
    "primary": (
        "mean per-date top-quintile net excess on the simple beta-adjusted label y_e (top-quintile mean y_e - date "
        "mean y_e - one-way turnover x round-trip cost) over pre-holdout outer OOS month-end dates"
    ),
    "secondary": [
        "within-date AUC of 1{y_e > median_date(y_e)} (the v2/v3 primary)",
        "within-date AUC of 1{y_e>0} (the v1 primary)",
        "mean within-date rank IC",
        "raw top-quintile net excess",
    ],
    "tie_tolerance": TIE_TOL,
    "tie_breaks": ["mean rank IC", "mean raw top-quintile net excess"],
    "gate": (
        "primary(selected) > primary(mom_12_1) on the same dates AND the primary's 95% block-bootstrap lower bound > 0"
    ),
    "serving": (
        "full multiplier only when the gate passed AND quantcio.holdout.xsh_confirmation(market, settings) confirms "
        "it on untouched names (looked up at scoring time); otherwise x UNSKILLED_SHRINK"
    ),
    "post_hoc": POST_HOC_LABEL,
    "diagnostics": "diagnostic=True candidates (v2 inputs only) are reported, never selected (docs/MODELING.md 16.4)",
    "post_hoc_note": (
        "v4's simple label and money metric were chosen AFTER the v3 results and docs 77-78 were seen; the "
        "24-month holdout was also seen before - confirmation needs untouched data (forward holdout)"
    ),
}


@dataclass
class Context:
    name: str
    train_end: pd.Timestamp  # training rows have date <= train_end (purged: t_end < first test date - h)
    test_dates: pd.DatetimeIndex
    final: bool = False  # serving context: predict the latest cross-section (unlabelled rows)


@dataclass
class Config:
    params: dict[str, Any]
    scheme: str
    inner_auc: float = float("nan")  # the inner PRIMARY score (v4: top-quintile net excess; key name kept from v3)
    inner_ic: float = float("nan")
    tried: int = 0
    bo: dict[str, Any] = field(default_factory=dict)  # Bayesian-optimisation record (docs/MODELING.md 82)
    one_se: dict[str, Any] = field(default_factory=dict)  # grid tuner's 1-SE record per stage (docs/MODELING.md 84.1)


@dataclass
class ContextResult:
    name: str
    predictions: pd.DataFrame  # date, entity, <candidate> scores
    configs: dict[str, dict[str, Any]]
    stack_base: str | None
    ensemble_members: list[str]
    seconds: float
    fitted: dict[str, Fitted] = field(default_factory=dict)  # final context only
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------------------------------ data access


class _Data:
    """Slim, float32 view of the dataset with fast row selection."""

    def __init__(self, ds: ModelingDataset):
        f = ds.frame
        keep = [
            "date",
            "entity",
            "ticker",
            "is_me",
            "labelled",
            "t_end",
            "y_e",
            "y_dm",
            "y_cls",
            "y_q",
            "w_date",
            "r",
            "vol60",
            *ds.xs_cols,
            *ds.regime_cols,
        ]
        keep = list(dict.fromkeys(c for c in keep if c in f))
        self.f = f[keep].copy()
        for c in [*ds.xs_cols, *ds.regime_cols]:
            self.f[c] = self.f[c].astype(np.float32)
        self.date = self.f["date"].to_numpy(dtype="datetime64[ns]")
        self.lab = self.f["labelled"].to_numpy(dtype=bool)
        self.me = self.f["is_me"].to_numpy(dtype=bool)
        self.ds = ds
        me_dates = self.f.loc[self.lab & self.me, "date"]
        self.me_lab = pd.DatetimeIndex(sorted(me_dates.unique()))

    def train(self, train_end: pd.Timestamp, rows: str) -> pd.DataFrame:
        m = self.lab & (self.date <= np.datetime64(train_end, "ns"))
        if rows == "M":
            m &= self.me
        return self.f[m]

    def rows_between(self, start: pd.Timestamp, end: pd.Timestamp, labelled: bool = True) -> pd.DataFrame:
        m = (self.date >= np.datetime64(start, "ns")) & (self.date <= np.datetime64(end, "ns"))
        if labelled:
            m &= self.lab
        return self.f[m]

    def eval_rows(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        m = self.lab & self.me & np.isin(self.date, dates.to_numpy(dtype="datetime64[ns]"))
        return self.f[m]

    def latest_rows(self) -> pd.DataFrame:
        return self.f[self.date == np.datetime64(self.ds.as_of, "ns")]


def purge_end(ds: ModelingDataset, first_test: pd.Timestamp) -> pd.Timestamp:
    """Last training date for a test block starting at ``first_test``: 2h sessions earlier (label end + embargo h)."""
    return ds.session_offset(first_test, -2 * ds.horizon)


def make_contexts(data: _Data) -> tuple[list[Context], Context, Context, dict[str, Any]]:
    """Outer walk-forward contexts, the holdout context and the final (serving) context."""
    ds = data.ds
    me = data.me_lab
    if len(me) < OUTER_MIN_TRAIN_MONTHS + HOLDOUT_MONTHS + 6:
        raise ValueError(f"modeling {ds.market.value}: only {len(me)} labelled month-ends - too short to select")
    hold = me[-HOLDOUT_MONTHS:]
    pre = me[:-HOLDOUT_MONTHS]
    tests = pre[OUTER_MIN_TRAIN_MONTHS:]
    outer = []
    for k, i in enumerate(range(0, len(tests), OUTER_BLOCK_MONTHS), start=1):
        block = tests[i : i + OUTER_BLOCK_MONTHS]
        outer.append(Context(f"outer_{k}", purge_end(ds, block[0]), block))
    holdout = Context("holdout", purge_end(ds, hold[0]), hold)
    final = Context("final", ds.session_offset(ds.as_of, -ds.horizon), pd.DatetimeIndex([ds.as_of]), final=True)
    info = {
        "labelled_month_ends": len(me),
        "first_month_end": str(me[0].date()),
        "pre_holdout": [str(pre[0].date()), str(pre[-1].date())],
        "outer_test": [str(tests[0].date()), str(tests[-1].date())],
        "outer_folds": len(outer),
        "holdout": [str(hold[0].date()), str(hold[-1].date())],
        "holdout_months": HOLDOUT_MONTHS,
        "embargo_sessions": ds.horizon,
        "purge_rule": "train rows: date <= session(test_start) - 2h (label end + h-session embargo before the test)",
        "inner": (
            f"{INNER_BLOCKS} blocks x {INNER_BLOCK_MONTHS} month-ends, same purge, "
            f">= {INNER_MIN_TRAIN_MONTHS} months of training"
        ),
        "final_train_end": str(final.train_end.date()),
    }
    return outer, holdout, final, info


def inner_folds(
    data: _Data, train_end: pd.Timestamp, blocks: int | None = None
) -> list[tuple[pd.Timestamp, pd.DatetimeIndex]]:
    me = data.me_lab[data.me_lab <= train_end]
    out = []
    for b in range(INNER_BLOCKS if blocks is None else int(blocks), 0, -1):
        lo = len(me) - b * INNER_BLOCK_MONTHS
        val = me[max(lo, 0) : len(me) - (b - 1) * INNER_BLOCK_MONTHS]
        if len(val) == 0:
            continue
        te = purge_end(data.ds, val[0])
        if (me <= te).sum() < INNER_MIN_TRAIN_MONTHS:
            continue
        out.append((te, val))
    return out


# ------------------------------------------------------------------------------------------------ fitting


def _ctx(data: _Data, cutoff: pd.Timestamp | None, settings: Settings, threads: int) -> FitContext:
    ds = data.ds
    return FitContext(
        xs_cols=list(ds.xs_cols),
        regime_cols=list(ds.regime_cols),
        events=ds.events,
        cutoff=cutoff,
        calendar=ds.calendar,
        horizon=ds.horizon,
        seed=0,
        threads=threads,
        cache_dir=settings.cache_path / "rsmt_null",
        backtest_shrink=float(settings.backtest.model.shrink_to_equal),
        backtest_lookback=int(settings.backtest.model.lookback_months),
        core_xs_cols=list(ds.core_xs_cols) if getattr(ds, "core_xs_cols", None) else None,
        core_regime_cols=list(ds.core_regime_cols) if getattr(ds, "core_regime_cols", None) else None,
    )


def _inner_series(
    data: _Data, cand: Candidate, params: dict, scheme: str, folds: list, settings: Settings, threads: int
) -> tuple[pd.Series, pd.Series]:
    """Per-date PRIMARY (protocol v4: top-quintile net excess) and rank IC over the inner validation blocks.
    A configuration whose scores are all equal on every date of an inner block has no view (e.g. an L1 logit
    with every coefficient at 0): both series come back empty, so it never wins (docs/MODELING.md 83.2)."""
    aucs, ics = [], []
    for te, val in folds:
        tr = data.train(te, cand.rows)
        va = data.eval_rows(val)
        if tr.empty or va.empty:
            continue
        fitted = cand.fit(tr, params, scheme, _ctx(data, val[0], settings, threads))
        va = va.assign(score=fitted.predict(va))
        if va.groupby("date")["score"].nunique().max() <= 1:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        aucs.append(mt.per_date_primary(va, cost_bps=data.ds.cost_bps))
        ics.append(mt.per_date_ic(va))
    if not aucs:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return pd.concat(aucs), pd.concat(ics)


def _score_inner(
    data: _Data, cand: Candidate, params: dict, scheme: str, folds: list, settings: Settings, threads: int
) -> tuple[float, float]:
    a, i = _inner_series(data, cand, params, scheme, folds, settings, threads)
    return float(a.mean()) if len(a) else float("nan"), float(i.mean()) if len(i) else float("nan")


def _suggest(trial: Any, space: dict[str, tuple]) -> tuple[dict[str, Any], str]:
    """(params, scheme) of one Optuna trial from a ``Candidate.search`` space."""
    params: dict[str, Any] = {}
    scheme = "none"
    for k, spec in space.items():
        kind = spec[0]
        if kind == "int":
            v: Any = trial.suggest_int(k, int(spec[1]), int(spec[2]))
        elif kind == "int_log":
            v = trial.suggest_int(k, int(spec[1]), int(spec[2]), log=True)
        elif kind == "float":
            v = trial.suggest_float(k, float(spec[1]), float(spec[2]))
        elif kind == "float_log":
            v = trial.suggest_float(k, float(spec[1]), float(spec[2]), log=True)
        elif kind == "cat":
            v = trial.suggest_categorical(k, list(spec[1]))
        else:
            raise ValueError(f"unknown search kind {kind!r} for {k}")
        if k == "scheme":
            scheme = str(v)
        else:
            params[k] = v
    return params, scheme


def _simple_value(name: str, value: Any) -> float:
    """A parameter on its simplicity scale; ``factor_fm``'s lookback 0 is the whole history, the longest window."""
    x = float(value)
    return float(np.inf) if name == "lookback" and x == 0 else x


def _simplest(cands: list[dict[str, Any]], simplicity: tuple[tuple[str, int], ...]) -> int:
    """Index of the simplest entry: training-weight scheme 'none' first (docs/MODELING.md 84.1), then the
    ``Candidate.simplicity`` order, then the higher inner mean (then the earlier entry)."""

    def key(i: int) -> tuple:
        p = cands[i]["params"]
        return (
            cands[i].get("scheme", "none") == "none",
            *(sign * _simple_value(name, p.get(name, 0.0)) for name, sign in simplicity),
            cands[i]["mean"],
        )

    return max(range(len(cands)), key=key)


def _one_se(
    rec: list[dict[str, Any]], simplicity: tuple[tuple[str, int], ...]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The 1-standard-error pick of scored configurations (docs/MODELING.md 82, 84.1): among the finite records
    within one standard error (of the best record's per-date inner PRIMARY values) of the best mean, the simplest
    (``_simplest``). Returns (the pick - None when no record is finite -, the rule's record)."""
    ok = [r for r in rec if np.isfinite(r["mean"])]
    if not ok:
        return None, {}
    best = max(ok, key=lambda r: r["mean"])
    x = best["series"].to_numpy(dtype=float)
    se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    near = [r for r in ok if r["mean"] >= best["mean"] - se]
    pick = near[_simplest(near, simplicity)]
    return pick, {
        "best_inner": best["mean"],
        "best_params": best["params"],
        "best_scheme": best["scheme"],
        "se": se,
        "within_1se": len(near),
        "picked_inner": pick["mean"],
    }


def _record(
    data: _Data, cand: Candidate, params: dict, scheme: str, folds: list, settings: Settings, threads: int
) -> dict[str, Any]:
    """One scored configuration: params, scheme, inner PRIMARY mean (NaN = no view) and the per-date series."""
    prim, ic = _inner_series(data, cand, params, scheme, folds, settings, threads)
    m = float(prim.mean()) if len(prim) else float("nan")
    return {"params": dict(params), "scheme": scheme, "mean": m, "series": prim, "ic": ic}


def tune_bayes(data: _Data, cand: Candidate, train_end: pd.Timestamp, settings: Settings, threads: int) -> Config:
    """docs/MODELING.md 82: TPE over ``cand.search`` on ``BO_INNER_BLOCKS`` purged inner blocks of THIS training
    window (objective = mean inner PRIMARY), then the 1-standard-error rule toward the simplest configuration."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = inner_folds(data, train_end, BO_INNER_BLOCKS)
    if not folds:
        return Config(dict(cand.grid[0]), "none")
    rec: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        params, scheme = _suggest(trial, cand.search)
        prim, ic = _inner_series(data, cand, params, scheme, folds, settings, threads)
        m = float(prim.mean()) if len(prim) else float("nan")
        rec.append({"params": params, "scheme": scheme, "mean": m, "series": prim, "ic": ic})
        return m if np.isfinite(m) else -1e9

    sampler = optuna.samplers.TPESampler(seed=BO_SEED, n_startup_trials=BO_STARTUP, multivariate=True)
    optuna.create_study(direction="maximize", sampler=sampler).optimize(objective, n_trials=int(cand.n_trials))
    return _bo_config(rec, cand, len(folds))


def _bo_config(rec: list[dict[str, Any]], cand: Candidate, n_folds: int) -> Config:
    """The 1-standard-error pick of a BO record (``tune_bayes`` / ``tune_rsmt_bayes``; scheme 'none' first)."""
    pick, rule = _one_se(rec, cand.simplicity)
    if pick is None:
        return Config(dict(cand.grid[0]), "none", tried=len(rec))
    ic = pick["ic"]
    cfg = Config(
        dict(pick["params"]), pick["scheme"], pick["mean"], float(ic.mean()) if len(ic) else float("nan"), len(rec)
    )
    cfg.bo = {"trials": len(rec), "inner_blocks": n_folds, **rule}
    return cfg


def tune(data: _Data, cand: Candidate, train_end: pd.Timestamp, settings: Settings, threads: int) -> Config:
    """Two-stage inner CV with the 1-standard-error rule (docs/MODELING.md 84.1): every grid point with scheme
    'none', keeping the simplest (``Candidate.simplicity``) within one SE of the best; then the other schemes at
    that point, keeping 'none' unless it is more than one SE below the best. A candidate with a search space is
    tuned by Bayesian optimisation instead (``tune_bayes``)."""
    if not cand.tuned:
        return Config(dict(cand.grid[0]), "none")
    if cand.search:
        return tune_bayes(data, cand, train_end, settings, threads)
    folds = inner_folds(data, train_end)
    if not folds:
        return Config(dict(cand.grid[0]), "none")
    grid = [_record(data, cand, params, "none", folds, settings, threads) for params in cand.grid]
    pick, grid_rule = _one_se(grid, cand.simplicity)
    if pick is None:
        return Config(dict(cand.grid[0]), "none", tried=len(grid))
    stage2 = [pick] + [
        _record(data, cand, pick["params"], scheme, folds, settings, threads)
        for scheme in cand.schemes
        if scheme != "none"
    ]
    scheme_rule: dict[str, Any] = {}
    if len(stage2) > 1:
        pick, scheme_rule = _one_se(stage2, ())
        assert pick is not None  # stage2[0] is finite
    ic = pick["ic"]
    cfg = Config(
        dict(pick["params"]),
        pick["scheme"],
        pick["mean"],
        float(ic.mean()) if len(ic) else float("nan"),
        len(grid) + len(stage2) - 1,
    )
    cfg.one_se = {"grid": grid_rule, "scheme": scheme_rule}
    return cfg


def tune_rsmt_bayes(
    data: _Data, rs: Candidate, tr: pd.DataFrame, train_end: pd.Timestamp, settings: Settings, threads: int
) -> Config:
    """docs/MODELING.md 83: BO of a stacked RSMT-Q candidate. ``tr`` = the context's training rows with the stacked
    walk-forward base score ``stack_z``; each inner block is predicted by a fit on the rows before its purge end."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = inner_folds(data, train_end, BO_INNER_BLOCKS)
    if not folds:
        return Config(dict(rs.grid[0]), "none")
    rec: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        params, _ = _suggest(trial, rs.search)
        prims, ics = [], []
        for te, val in folds:
            tri = tr[tr["date"] <= te]
            va = data.eval_rows(val)
            va = va.assign(stack_z=tr["stack_z"].reindex(va.index))
            va = va[va["stack_z"].notna()]
            if tri["stack_z"].notna().sum() == 0 or va.empty:
                continue
            try:
                f = rs.fit(tri, params, "none", _ctx(data, val[0], settings, threads))
            except ValueError:  # too few stacked rows in an early inner window
                continue
            sc = va.assign(score=f.predict(va))
            prims.append(mt.per_date_primary(sc, cost_bps=data.ds.cost_bps))
            ics.append(mt.per_date_ic(sc))
        prim = pd.concat(prims) if prims else pd.Series(dtype=float)
        ic = pd.concat(ics) if ics else pd.Series(dtype=float)
        m = float(prim.mean()) if len(prim) else float("nan")
        rec.append({"params": params, "scheme": "none", "mean": m, "series": prim, "ic": ic})
        return m if np.isfinite(m) else -1e9

    sampler = optuna.samplers.TPESampler(seed=BO_SEED, n_startup_trials=BO_STARTUP, multivariate=True)
    optuna.create_study(direction="maximize", sampler=sampler).optimize(objective, n_trials=int(rs.n_trials))
    return _bo_config(rec, rs, len(folds))


def stack_scores(
    data: _Data, base: Candidate, cfg: Config, train_end: pd.Timestamp, settings: Settings, threads: int
) -> pd.Series:
    """Walk-forward OOS scores of the stacking base on every labelled grid row with date <= train_end
    (yearly refits after STACK_MIN_TRAIN_MONTHS); per-date rank-gauss. Index = rows of ``data.f``."""
    me = data.me_lab[data.me_lab <= train_end]
    parts = []
    starts = list(range(STACK_MIN_TRAIN_MONTHS, len(me), INNER_BLOCK_MONTHS))
    for j, i in enumerate(starts):
        b0 = me[i]
        b1 = me[starts[j + 1]] if j + 1 < len(starts) else None
        tr = data.train(purge_end(data.ds, b0), base.rows)
        if tr.empty:
            continue
        end = train_end if b1 is None else data.ds.session_offset(b1, -1)
        rows = data.rows_between(b0, min(end, train_end))
        if rows.empty:
            continue
        fitted = base.fit(tr, cfg.params, cfg.scheme, _ctx(data, b0, settings, threads))
        parts.append(pd.Series(fitted.predict(rows), index=rows.index))
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts)
    return rank_gauss(s, data.f.loc[s.index, "date"])


def rank_mean(preds: pd.DataFrame, members: list[str], weights: list[float] | None = None) -> pd.Series:
    """Per-date (weighted) mean of the members' uniform ranks (0, 1) - the composite score."""
    ranks = pd.concat([uniform_rank(preds[m], preds["date"]) for m in members], axis=1)
    if not weights:
        return ranks.mean(axis=1)
    w = np.asarray(weights, dtype=float)
    return pd.Series(ranks.to_numpy(dtype=float) @ (w / w.sum()), index=ranks.index)


def composite_members(name: str, configs: dict[str, dict[str, Any]]) -> list[str]:
    """Members of a composite candidate (ensemble / blend) in one context, else ``[name]``."""
    if name in COMPOSITES:
        return list((configs.get(name) or {}).get("members", []))
    return [name]


def run_context(
    data: _Data,
    ctx: Context,
    cands: list[Candidate],
    settings: Settings,
    threads: int,
    say: Callable[[str], None],
) -> ContextResult:
    """Tune, fit and predict every candidate in one context (see module docstring, steps 3-5)."""
    t0 = time.monotonic()
    test = data.latest_rows() if ctx.final else data.eval_rows(ctx.test_dates)
    preds = test[["date", "entity"]].copy()
    configs: dict[str, dict[str, Any]] = {}
    fitted_keep: dict[str, Fitted] = {}
    notes: list[str] = []
    fitted_all: dict[str, Fitted] = {}
    cutoff = ctx.test_dates[0]
    for cand in cands:
        if cand.name in STACKED:
            continue  # stacked on this context's result: run_rsmt (own checkpoint)
        tc = time.monotonic()
        cfg = tune(data, cand, ctx.train_end, settings, threads)
        tr = data.train(ctx.train_end, cand.rows)
        fitted = cand.fit(tr, cfg.params, cfg.scheme, _ctx(data, cutoff, settings, threads))
        preds[cand.name] = fitted.predict(test)
        fitted_all[cand.name] = fitted
        configs[cand.name] = {
            "params": cfg.params,
            "scheme": cfg.scheme,
            "inner_auc": cfg.inner_auc,
            "inner_ic": cfg.inner_ic,
            "configs_tried": cfg.tried,
            "n_train": len(tr),
            "seconds": round(time.monotonic() - tc, 1),
        }
        if cfg.one_se:  # docs 84.1: the grid tuner's 1-SE record (best vs picked, per stage)
            configs[cand.name]["one_se"] = cfg.one_se
        if cand.search:  # docs 82: the BO record + in-sample (training month-end) fit, to see the overfitting gap
            tr_me = tr[tr["is_me"]] if "is_me" in tr else tr
            ins = tr_me.assign(score=fitted.predict(tr_me))
            configs[cand.name]["bo"] = cfg.bo
            configs[cand.name]["train_auc"] = float(mt.per_date_auc_median(ins).mean())
            configs[cand.name]["train_primary"] = float(mt.per_date_primary(ins, cost_bps=data.ds.cost_bps).mean())
        say(
            f"{ctx.name}: {cand.name} {cfg.params} {cfg.scheme} inner AUC {cfg.inner_auc:.4f} "
            f"({configs[cand.name]['seconds']}s)"
        )
    inner = {
        c.name: {"inner_auc": configs[c.name]["inner_auc"], "inner_ic": configs[c.name]["inner_ic"]}
        for c in cands
        if c.name in configs
    }
    tuned = [
        c
        for c in cands
        if c.tuned and not c.diagnostic and c.name in configs and np.isfinite(configs[c.name]["inner_auc"])
    ]
    ranked = sorted(tuned, key=lambda c: (-configs[c.name]["inner_auc"], -np.nan_to_num(configs[c.name]["inner_ic"])))
    stack_base = ranked[0].name if ranked else None
    members = [c.name for c in ranked[:ENSEMBLE_K]]
    # ---- composites (candidates.composite_rule): ensemble, ensemble5, ensemble_w, family_blend, blend
    for comp in COMPOSITE_CANDIDATES:
        mem, w = composite_rule(comp.name, inner, cands)
        if mem and all(m in preds for m in mem):
            preds[comp.name] = rank_mean(preds, mem, w if any(x != 1.0 for x in w) else None)
            configs[comp.name] = {"members": mem, "weights": w}
    if ctx.final:
        fitted_keep = fitted_all
    return ContextResult(
        ctx.name,
        preds.reset_index(drop=True),
        configs,
        stack_base,
        members,
        round(time.monotonic() - t0, 1),
        fitted_keep,
        notes,
    )


def run_rsmt(
    data: _Data,
    ctx: Context,
    base_res: ContextResult,
    cands: list[Candidate],
    settings: Settings,
    threads: int,
    say: Callable[[str], None],
    name: str = RSMT,
) -> dict[str, Any]:
    """The stacked RSMT-Q candidate ``name`` (``STACKED``: rsmt_v2 / rsmt_v3) of one context, stacked on the
    context's best inner-CV candidate (walk-forward OOS scores of that candidate on the training rows; its outer
    prediction on the test rows). Returns predictions (aligned to ``base_res.predictions``), config, fitted model
    (final context) and notes."""
    rs = next((c for c in cands if c.name == name), None)
    base_name = base_res.stack_base
    out: dict[str, Any] = {"pred": None, "config": None, "fitted": None, "notes": []}
    if rs is None or base_name is None:
        return out
    tc = time.monotonic()
    base = next(c for c in cands if c.name == base_name)
    bc = base_res.configs[base_name]
    bcfg = Config(bc["params"], bc["scheme"])
    test = data.latest_rows() if ctx.final else data.eval_rows(ctx.test_dates)
    preds = base_res.predictions
    key = test[["date", "entity"]].reset_index().merge(preds[["date", "entity", base_name]], on=["date", "entity"])
    try:
        sz = stack_scores(data, base, bcfg, ctx.train_end, settings, threads)
        tr = data.train(ctx.train_end, rs.rows).copy()
        tr["stack_z"] = sz.reindex(tr.index)
        te = test.loc[key["index"].to_numpy()].copy()
        te["stack_z"] = rank_gauss(pd.Series(key[base_name].to_numpy(), index=te.index), te["date"])
        rcfg = (
            tune_rsmt_bayes(data, rs, tr, ctx.train_end, settings, threads)
            if rs.search
            else Config(dict(rs.grid[0]), "none")
        )
        fitted = rs.fit(tr, dict(rcfg.params), "none", _ctx(data, ctx.test_dates[0], settings, threads))
        scored = te[["date", "entity"]].assign(score=fitted.predict(te))
        out["pred"] = preds[["date", "entity"]].merge(scored, on=["date", "entity"], how="left")["score"].to_numpy()
        out["config"] = {
            "params": dict(rcfg.params),
            "scheme": "none",
            "bo": rcfg.bo,
            "inner_auc": rcfg.inner_auc,
            "stack_base": base_name,
            "n_train": int(tr["stack_z"].notna().sum()),
            "tree": fitted.model.describe(),
            "seconds": round(time.monotonic() - tc, 1),
        }
        out["fitted"] = fitted if ctx.final else None
        say(f"{ctx.name}: {name} on {base_name} n={out['config']['n_train']} ({out['config']['seconds']}s)")
    except Exception as exc:  # noqa: BLE001 - recorded, never silent
        log.exception("%s failed in %s", name, ctx.name)
        out["notes"].append(f"{name} failed in {ctx.name}: {type(exc).__name__}: {exc}")
    return out


# ------------------------------------------------------------------------------------------------ selection


def _with_rsmt(res: ContextResult, rres: dict[str, Any], name: str = RSMT) -> ContextResult:
    """A copy of the context result with a stacked candidate's (``name``) predictions / config / fitted model
    merged in."""
    preds = res.predictions.copy()
    preds[name] = rres["pred"] if rres.get("pred") is not None else np.nan
    configs = {**res.configs}
    if rres.get("config") is not None:
        configs[name] = rres["config"]
    fitted = {**res.fitted}
    if rres.get("fitted") is not None:
        fitted[name] = rres["fitted"]
    return ContextResult(
        res.name,
        preds,
        configs,
        res.stack_base,
        res.ensemble_members,
        res.seconds,
        fitted,
        [*res.notes, *rres.get("notes", [])],
    )


def select_best(table: dict[str, dict[str, Any]], tie_tol: float = TIE_TOL, key: str = PRIMARY) -> str:
    """The pre-declared selection rule over ``{name: outer-OOS metrics}`` (module docstring, step 6 / v4): the
    highest mean ``key`` (the PRIMARY metric), ties by rank IC, then raw top-quintile net excess."""

    def val(m: dict[str, Any], k: str) -> float:
        v = (m.get(k) or {}).get("mean")
        return float(v) if v is not None and np.isfinite(v) else -np.inf

    names = [n for n in table if np.isfinite(val(table[n], key))]
    if not names:
        raise ValueError(f"select_best: no candidate has an out-of-sample {key}")
    top = max(val(table[n], key) for n in names)
    tied = [n for n in names if top - val(table[n], key) < tie_tol]
    tied.sort(key=lambda n: (-val(table[n], key), n))
    return max(tied, key=lambda n: (val(table[n], "ic"), val(table[n], "top_net"), -tied.index(n)))


def serving_gate(selected: dict[str, Any], baseline: dict[str, Any], key: str = PRIMARY) -> dict[str, Any]:
    """The pre-declared serving gate on the PRIMARY metric (module docstring, step 7 / v4): ``value`` / ``lo`` /
    ``baseline`` are that metric (``null`` = its no-skill value); ``auc`` / ``auc_lo`` / ``baseline_auc`` are the
    reported AUC (``PRIMARY_AUC``, secondary since v4). ``post_hoc``: a pass still needs confirmation on untouched
    data (``needs_confirmation``, looked up at scoring time)."""
    null = PRIMARY_NULL if key == PRIMARY else 0.5
    a = (selected.get(key) or {}).get("mean")
    lo = (selected.get(key) or {}).get("lo")
    b = (baseline.get(key) or {}).get("mean")
    beats = bool(a is not None and b is not None and a > b)
    ci_ok = bool(lo is not None and lo > null)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "metric": key,
        "metric_label_ko": mt.metric_label(key, "ko"),
        "metric_label_en": mt.metric_label(key, "en"),
        "beats_momentum": beats,
        "lo95_above_null": ci_ok,
        "passed": beats and ci_ok,
        "value": a,
        "lo": lo,
        "baseline": b,
        "null": null,
        "value_ann": None if a is None else a * mt.PERIODS_PER_YEAR,
        "auc": (selected.get(PRIMARY_AUC) or {}).get("mean"),
        "auc_lo": (selected.get(PRIMARY_AUC) or {}).get("lo"),
        "baseline_auc": (baseline.get(PRIMARY_AUC) or {}).get("mean"),
        "post_hoc": POST_HOC_LABEL,
        "needs_confirmation": POST_HOC_LABEL,
        "confirmation": "quantcio.holdout.xsh_confirmation(market, settings) at scoring time (untouched XSH names)",
    }


def calibration(oos: pd.DataFrame, score: str) -> dict[str, Any]:
    """Per-date OLS slope of the demeaned excess return on the per-date rank-gauss score (date-clustered t)."""
    d = oos[["date", score, "y_e"]].dropna()
    if d.empty:
        return {"slope": 0.0, "t": None, "n_dates": 0, "shrink": 0.0}
    z = rank_gauss(d[score], d["date"])
    y = d["y_e"] - d["y_e"].groupby(d["date"]).transform("mean")
    g = pd.DataFrame({"date": d["date"], "zy": z * y, "zz": z * z})
    agg = g.groupby("date").agg(zy=("zy", "sum"), zz=("zz", "sum"), n=("zz", "size"))
    slopes = (agg["zy"] / agg["zz"]).where(agg["n"] >= mt.MIN_NAMES).dropna()
    s = mt.summarize(slopes, 0.0)
    t = s["t"] or 0.0
    return {
        "slope": s["mean"] or 0.0,
        "t": s["t"],
        "lo": s["lo"],
        "hi": s["hi"],
        "n_dates": s["n"],
        "shrink": float(t * t / (1.0 + t * t)) if t > 0 else 0.0,
        "rule": "mu_excess = shrink * slope * z (z = per-date rank-gauss score); shrink = t^2/(1+t^2), 0 if t <= 0",
    }


def event_robustness(oos: pd.DataFrame, ds: ModelingDataset, names: list[str]) -> dict[str, Any]:
    """Per candidate: metrics on event-overlap vs calm dates; per named event: mean primary / IC over its dates.
    "auc" keys = the PRIMARY metric (``auc_metric`` in the output; v4: top-quintile net excess)."""
    ev = ds.events
    dates = pd.DatetimeIndex(sorted(oos["date"].unique()))
    t_end = ds.dates_frame["t_end"].reindex(dates)
    flag = pd.Series(ev.overlap_mask(dates, t_end.fillna(pd.Series(dates, index=dates))), index=dates)
    out: dict[str, Any] = {
        "n_event_dates": int(flag.sum()),
        "n_calm_dates": int((~flag).sum()),
        "auc_metric": PRIMARY,
        "by_candidate": {},
        "by_event": [],
    }
    per_auc = {n: mt.per_date_primary(oos, n, ds.cost_bps) for n in names}
    per_ic = {n: mt.per_date_ic(oos, n) for n in names}
    for n in names:
        a, i = per_auc[n], per_ic[n]
        fa = flag.reindex(a.index).fillna(False).astype(bool)
        fi = flag.reindex(i.index).fillna(False).astype(bool)
        out["by_candidate"][n] = {
            "event_auc": mt.summarize(a[fa], PRIMARY_NULL),
            "calm_auc": mt.summarize(a[~fa], PRIMARY_NULL),
            "event_ic": mt.summarize(i[fi], 0.0),
            "calm_ic": mt.summarize(i[~fi], 0.0),
        }
    for w in ev.windows.itertuples(index=False):
        hit = [d for d in dates if (d <= w.end) and (t_end.get(d, d) >= w.start)]
        if not hit:
            continue
        row = {"key": w.key, "kind": w.kind, "label_ko": w.label_ko, "label_en": w.label_en, "n_dates": len(hit)}
        for n in names:
            row[f"auc_{n}"] = float(per_auc[n].reindex(hit).mean()) if len(per_auc[n].reindex(hit).dropna()) else None
            row[f"ic_{n}"] = float(per_ic[n].reindex(hit).mean()) if len(per_ic[n].reindex(hit).dropna()) else None
        out["by_event"].append(row)
    return out


def _ast(src: str) -> bytes:
    import ast
    import textwrap

    return ast.dump(ast.parse(textwrap.dedent(src))).encode()


def _fingerprint(rsmt: bool = False) -> str:
    """Code identity of a checkpoint, hashed on the AST (formatting-only edits and edits to reporting / serving
    code keep the checkpoints valid).

    Base contexts: protocol constants, candidates.py WITHOUT the RSMT-Q v2 fitter, and the functions that
    produce a context. RSMT-Q v2 (``rsmt=True``): the base identity + its fitter, ``RSMTRanker``,
    ``stack_scores`` and ``run_rsmt``.
    """
    import ast
    import inspect

    from quantcio.modeling import candidates as cmod
    from quantcio.quant.rsmt import RSMTRanker

    tree = ast.parse(Path(cmod.__file__).read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == "_fit_rsmt")]
    h = hashlib.sha1(str(SELECTION_VERSION).encode())
    consts = (HOLDOUT_MONTHS, OUTER_MIN_TRAIN_MONTHS, OUTER_BLOCK_MONTHS, INNER_BLOCKS, INNER_BLOCK_MONTHS)
    h.update(
        repr(
            (*consts, INNER_MIN_TRAIN_MONTHS, STACK_MIN_TRAIN_MONTHS, ENSEMBLE_K, PRIMARY, PROTOCOL_VERSION)
            + (BO_INNER_BLOCKS, BO_SEED, BO_STARTUP)
        ).encode()
    )
    h.update(ast.dump(tree).encode())
    objs = (
        run_context,
        rank_mean,
        tune,
        tune_bayes,
        _bo_config,
        _suggest,
        _simple_value,
        _simplest,
        _one_se,
        _record,
        _inner_series,
        inner_folds,
        make_contexts,
        purge_end,
        _score_inner,
        _ctx,
        _Data,
    )
    for obj in objs:
        h.update(_ast(inspect.getsource(obj)))
    if rsmt:
        for obj in (cmod._fit_rsmt, RSMTRanker, stack_scores, run_rsmt, tune_rsmt_bayes, _bo_config, _one_se):
            h.update(_ast(inspect.getsource(obj)))
    return h.hexdigest()[:12]


@dataclass
class SelectionResult:
    market: str
    created_at: str
    as_of: str
    protocol: dict[str, Any]
    candidates: list[dict[str, Any]]
    selected: str
    gate: dict[str, Any]
    market_model: dict[str, Any]
    events: dict[str, Any]
    calibration: dict[str, Any]
    importances: dict[str, float]
    reference: dict[str, Any]
    dataset: dict[str, Any]
    contexts: list[dict[str, Any]]
    notes: list[dict[str, str]]
    timings: dict[str, float]
    snapshot: str = ""
    settings_digest: str = ""
    # not serialised
    oos: pd.DataFrame | None = None
    holdout_preds: pd.DataFrame | None = None
    final: ContextResult | None = None

    def to_dict(self) -> dict[str, Any]:
        from quantcio.backtest.model import _jsonable

        return _jsonable(
            {
                k: getattr(self, k)
                for k in (
                    "market",
                    "created_at",
                    "as_of",
                    "protocol",
                    "candidates",
                    "selected",
                    "gate",
                    "market_model",
                    "events",
                    "calibration",
                    "importances",
                    "reference",
                    "dataset",
                    "contexts",
                    "notes",
                    "timings",
                    "snapshot",
                    "settings_digest",
                )
            }
        )

    def candidate(self, name: str) -> dict[str, Any]:
        return next(c for c in self.candidates if c["name"] == name)


def _reference_backtest(
    ds: ModelingDataset, settings: Settings, cache: DiskCache, eval_rows: pd.DataFrame
) -> dict[str, Any]:
    """The factor-backtest agent's cached walk-forward OOS scores evaluated with the same metrics (reference)."""
    try:
        from quantcio.backtest import load_cached_backtest

        bt = load_cached_backtest(ds.market, settings, cache)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    if bt is None or bt.model_result is None:
        return {"available": False, "reason": "no cached factor backtest"}
    sc = bt.model_result.scores.rename(columns={"score": "bt_score"})
    j = eval_rows[["date", "entity", "y_e", "r"]].merge(sc, on=["date", "entity"], how="inner")
    if j.empty:
        return {"available": False, "reason": "no overlapping dates"}
    return {"available": True, "method": bt.model_result.method, "metrics": mt.evaluate(j, ds.cost_bps, "bt_score")}


def run_selection(
    ds: ModelingDataset,
    settings: Settings,
    cache: DiskCache,
    *,
    candidates: list[Candidate] | None = None,
    threads: int = 8,
    use_cache: bool = True,
    progress: Callable[[str], None] | None = None,
    max_outer: int | None = None,
) -> SelectionResult:
    """Run the full protocol on ``ds`` (see module docstring) and return the result (not yet written to disk)."""
    say = progress or (lambda m: log.info("selection: %s", m))
    t0 = time.monotonic()
    cands = list(candidates or default_candidates())
    if BASELINE not in {c.name for c in cands}:  # the pre-declared gate compares against it: always evaluated
        cands.insert(0, next(c for c in default_candidates() if c.name == BASELINE))
    names = [c.name for c in cands]
    data = _Data(ds)
    outer, holdout, final, info = make_contexts(data)
    if max_outer is not None:
        outer = outer[-int(max_outer) :]
    fp = _fingerprint()
    fp_rsmt = _fingerprint(rsmt=True)
    base_names = [n for n in names if n not in STACKED]
    results: dict[str, ContextResult] = {}
    timings: dict[str, float] = {}
    for ctx in [*outer, holdout, final]:
        ident = [ds.snapshot, ctx.name, str(ctx.train_end), [str(d) for d in ctx.test_dates]]
        key = DiskCache.key(CTX_NS, fp, *ident, base_names)
        hit = cache.get(CTX_NS, key, None) if use_cache else None
        if isinstance(hit, ContextResult):
            say(f"{ctx.name}: checkpoint reused")
            res = hit
        else:
            test_span = f"{ctx.test_dates[0].date()}..{ctx.test_dates[-1].date()}"
            say(f"{ctx.name}: train <= {ctx.train_end.date()}, test {test_span}")
            res = run_context(data, ctx, cands, settings, threads, say)
            timings[ctx.name] = res.seconds
            if use_cache:
                cache.set(CTX_NS, key, res)
        for sname in (n for n in STACKED if n in names):
            sb = res.stack_base
            rkey = DiskCache.key(RSMT_NS, fp_rsmt, *ident, base_names, sb, res.configs.get(sb) if sb else None, sname)
            rhit = cache.get(RSMT_NS, rkey, None) if use_cache else None
            if isinstance(rhit, dict) and "pred" in rhit:
                say(f"{ctx.name}: {sname} checkpoint reused")
                rres = rhit
            else:
                rres = run_rsmt(data, ctx, res, cands, settings, threads, say, name=sname)
                if use_cache:
                    cache.set(RSMT_NS, rkey, rres)
            res = _with_rsmt(res, rres, sname)
        results[ctx.name] = res
    all_names = [*names, *COMPOSITES]
    by_name = {c.name: c for c in [*cands, *COMPOSITE_CANDIDATES]}
    lab = data.f[data.lab & data.me][["date", "entity", "y_e", "r"]]
    oos = pd.concat([results[c.name].predictions for c in outer], ignore_index=True).merge(lab, on=["date", "entity"])
    hold = results["holdout"].predictions.merge(lab, on=["date", "entity"])
    table: dict[str, dict[str, Any]] = {}
    hold_table: dict[str, dict[str, Any]] = {}
    rows = []
    base_auc = mt.per_date_primary(oos, BASELINE, ds.cost_bps)
    for n in all_names:
        if n not in oos:
            continue
        table[n] = mt.evaluate(oos, ds.cost_bps, n)
        hold_table[n] = mt.evaluate(hold, ds.cost_bps, n) if n in hold else {}
        cand = by_name[n]
        rows.append(
            {
                "name": n,
                "family": cand.family,
                "diagnostic": bool(cand.diagnostic),
                "label_ko": cand.label_ko,
                "label_en": cand.label_en,
                "rows": cand.rows,
                "grid_points": len(cand.grid),
                "schemes": list(cand.schemes),
                "outer": table[n],
                "holdout": hold_table[n],
                "holdout_note": "not used for selection",
                # PRIMARY-metric difference vs the momentum baseline (per date, common dates; key kept from v3)
                "vs_momentum_auc": (
                    mt.paired_difference(mt.per_date_primary(oos, n, ds.cost_bps), base_auc) if n != BASELINE else None
                ),
                "configs": {r: results[r].configs.get(n) for r in results},
            }
        )
    selected = select_best({n: m for n, m in table.items() if not by_name[n].diagnostic})
    gate = serving_gate(table[selected], table[BASELINE])
    alt_value = {}
    for full, core in DIAGNOSTIC_PAIRS:
        if full in oos and core in oos:
            alt_value[f"{full} - {core}"] = {
                "outer": mt.paired_difference(
                    mt.per_date_primary(oos, full, ds.cost_bps), mt.per_date_primary(oos, core, ds.cost_bps)
                ),
                "holdout": (
                    mt.paired_difference(
                        mt.per_date_primary(hold, full, ds.cost_bps), mt.per_date_primary(hold, core, ds.cost_bps)
                    )
                    if full in hold and core in hold
                    else None
                ),
            }
    say(f"selected {selected}: outer {PRIMARY} {table[selected][PRIMARY]['mean']:.5f}, gate {gate}")
    # market direction
    from quantcio.modeling.market import market_direction

    mkt = market_direction(ds.dates_frame, ds.calendar, ds.horizon, pd.Timestamp(holdout.test_dates[0]))
    allo = pd.concat([oos, hold], ignore_index=True)
    ev = event_robustness(allo, ds, [n for n in all_names if n in allo])
    cal = calibration(allo, selected)
    fin = results["final"]
    imp: dict[str, float] = {}
    members = composite_members(selected, fin.configs)
    for mname in members:
        f = fin.fitted.get(mname)
        if f is None:
            continue
        raw = f.importance()
        tot = sum(abs(v) for v in raw.values()) or 1.0
        for k, v in raw.items():
            imp[k] = imp.get(k, 0.0) + abs(v) / tot / max(len(members), 1)
    imp = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:20])
    n_configs = sum(
        (c.n_trials if c.search else len(c.grid) + max(0, len(c.schemes) - 1)) for c in cands if c.tuned
    ) + sum(1 for c in cands if not c.tuned)
    n_eligible = sum(1 for r in rows if not r["diagnostic"])
    protocol = {
        **info,
        "protocol_version": PROTOCOL_VERSION,
        "primary_metric": PRIMARY,
        "post_hoc": POST_HOC_LABEL,
        "rule": SELECTION_RULE,
        "candidates": len(rows),
        "eligible_candidates": n_eligible,
        "alt_value": alt_value,
        "alt_features": list(getattr(ds, "alt_raw", []) or []),
        "alt_coverage": getattr(ds, "alt_coverage", {}) or {},
        "configurations_per_fold": n_configs,
        "multiple_testing": (
            f"{n_eligible} candidates ({n_configs} configurations per fold, tuned by inner CV) compete on the same "
            f"pre-holdout dates: the winner's outer primary is optimistically biased (winner's curse). The "
            f"{HOLDOUT_MONTHS}-month holdout was already seen and the v4 metric is post hoc, so neither is an "
            "untouched confirmation: that needs a forward holdout registered after the run."
        ),
        "fingerprint": fp,
    }
    res = SelectionResult(
        market=ds.market.value,
        created_at=datetime.now().isoformat(timespec="seconds"),
        as_of=str(ds.as_of.date()),
        protocol=protocol,
        candidates=rows,
        selected=selected,
        gate=gate,
        market_model={k: v for k, v in mkt.items() if k != "predictions"},
        events=ev,
        calibration=cal,
        importances=imp,
        reference=_reference_backtest(ds, settings, cache, data.f[data.lab & data.me]),
        dataset={"coverage": ds.coverage, "universe": ds.universe_desc, "cost_bps": ds.cost_bps, "notes": ds.notes},
        contexts=[
            {
                "name": r.name,
                "seconds": r.seconds,
                "stack_base": r.stack_base,
                "ensemble": r.ensemble_members,
                "notes": r.notes,
            }
            for r in results.values()
        ],
        notes=[],
        timings={**timings, "total": round(time.monotonic() - t0, 1)},
        snapshot=ds.snapshot,
        settings_digest=settings.digest(),
        oos=oos,
        holdout_preds=hold,
        final=fin,
    )
    return res


__all__ = [
    "BASELINE",
    "HOLDOUT_MONTHS",
    "POST_HOC_LABEL",
    "PRIMARY_AUC",
    "PROTOCOL_VERSION",
    "SELECTION_RULE",
    "TIE_TOL",
    "UNSKILLED_SHRINK",
    "Context",
    "ContextResult",
    "SelectionResult",
    "calibration",
    "composite_members",
    "event_robustness",
    "inner_folds",
    "make_contexts",
    "purge_end",
    "rank_mean",
    "run_selection",
    "select_best",
    "serving_gate",
]
