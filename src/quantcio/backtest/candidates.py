"""Pre-declared candidate set of walk-forward multi-factor models + the selection rule.

docs/BACKTEST_MACRO.md section 3.1 (written before any real-data run). Every candidate estimates the weights of
rebalance date t from months strictly before t (the cross-section of month s is complete once its forward return is
realised at the next rebalance date <= t) and outputs weights on the same dates (>= ``min_months`` past IC months).
All candidates are scored out of sample on the common months; the last ``holdout_months`` are a holdout that is
reported but never used for the choice. The choice: highest mean monthly OOS rank IC over the pre-holdout months;
ties (< ``TIE_IC``) -> higher IC t (ties < ``TIE_T``) -> simpler (fewer fitted parameters, then the table order).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from quantcio.backtest.factors import EXPECTED_SIGN, FACTORS
from quantcio.backtest.model import (
    REGIME_SHRINK_MONTHS,
    ModelResult,
    _evaluate,
    _jsonable,
    _scores,
    _weights_path,
    cross_sectional_ic,
    fit_weights,
    fm_slopes,
    ic_summary,
    perf_stats,
    regime_score,
    regime_state,
    t_stat,
)
from quantcio.backtest.tree import fit_ic_tree
from quantcio.config import Settings

SELECTION_VERSION = 1
CANDIDATES: tuple[str, ...] = (
    "equal_weight",
    "ic_weighted",
    "ic_weighted_60",
    "ic_weighted_raw",
    "fama_macbeth",
    "ridge",
    "macro_conditional",
    "rsmt_tree",
)
N_PARAMS = {
    "equal_weight": 0,
    "ic_weighted": 6,
    "ic_weighted_60": 6,
    "ic_weighted_raw": 6,
    "fama_macbeth": 6,
    "ridge": 7,
    "macro_conditional": 12,
    "rsmt_tree": 6,  # 6 x leaves + splits of the latest tree (set at run time)
}
LABELS = {
    "equal_weight": "동일가중(사전 부호, 적합 없음)",
    "ic_weighted": "IC 가중 36개월(부호 제약·50% 축소, 현행)",
    "ic_weighted_60": "IC 가중 60개월(부호 제약·50% 축소)",
    "ic_weighted_raw": "IC 가중 36개월(부호 학습·축소 없음)",
    "fama_macbeth": "Fama–MacBeth 36개월(t²/(1+t²) 축소)",
    "ridge": "풀링 능형회귀 36개월(순위 목표, λ 내부 CV)",
    "macro_conditional": "매크로 조건부 IC 가중(위험선호/회피)",
    "rsmt_tree": "RSMT식 IC 트리(시장 수익률·변동성·매크로 분할)",
}
LABELS_EN = {
    "equal_weight": "equal weight (prior signs, no fit)",
    "ic_weighted": "IC-weighted 36M (sign-constrained, 50% shrink; current)",
    "ic_weighted_60": "IC-weighted 60M (sign-constrained, 50% shrink)",
    "ic_weighted_raw": "IC-weighted 36M (learned signs, no shrink)",
    "fama_macbeth": "Fama-MacBeth 36M (t^2/(1+t^2) shrink)",
    "ridge": "pooled ridge 36M (rank target, inner-CV lambda)",
    "macro_conditional": "macro-conditioned IC weights (risk-on/off)",
    "rsmt_tree": "RSMT-style IC tree (market return / vol / macro splits)",
}
REFERENCE = "ic_weighted"  # the current model: paired IC differences are reported against it
TIE_IC = 0.002
TIE_T = 0.10
LOOKBACK_60 = 60
FM_LOOKBACK = 36
RIDGE_LOOKBACK = 36
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)  # lambda = alpha * N / 3 (N = pooled rows; Var(uniform z) = 1/3)
SPLIT_VARS = ("mkt_ret_60d", "mkt_vol_60d", "macro_score")


@dataclass
class Inputs:
    panel: pd.DataFrame
    factors: list[str]
    zcols: list[str]
    dates: list[pd.Timestamp]  # every formation date of the panel
    model_dates: list[pd.Timestamp]  # dates with >= min_months past monthly ICs
    ic: pd.DataFrame  # monthly factor Spearman IC (index formation date, columns factors)
    lookback: int
    min_months: int
    shrink: float
    state: pd.Series | None  # macro regime +1 / -1 / NaN by date (None: no regime history)
    split_vars: pd.DataFrame  # index dates, date-level variables known at t (rsmt_tree)


# --------------------------------------------------------------------------------------------- helpers


def _normalise(w: pd.Series) -> dict[str, float]:
    w = w.astype(float).fillna(0.0)
    tot = float(w.abs().sum())
    return {f: (float(v) / tot if tot > 0 else 0.0) for f, v in w.items()}


def _frame(rows: dict[pd.Timestamp, dict[str, float]], factors: list[str]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(rows, orient="index", columns=factors).sort_index()


def _past(hist: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    return hist[hist.index < t].dropna(how="all")


def model_dates(dates: list[pd.Timestamp], ic: pd.DataFrame, min_months: int) -> list[pd.Timestamp]:
    idx = ic.dropna(how="all").index
    return [t for t in dates if int(np.searchsorted(idx.values, np.datetime64(t), side="left")) >= min_months]


def uniform_rank_array(x: np.ndarray) -> np.ndarray:
    n = x.size
    return 2.0 * (rankdata(x, method="average") - 0.5) / n - 1.0 if n else x


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


# --------------------------------------------------------------------------------------------- candidates


def weights_equal(inp: Inputs) -> pd.DataFrame:
    w = _normalise(pd.Series({f: float(EXPECTED_SIGN.get(f, 0)) for f in inp.factors}))
    return _frame({t: w for t in inp.model_dates}, inp.factors)


def weights_ic(inp: Inputs, lookback: int, shrink: float, state: pd.Series | None = None) -> pd.DataFrame:
    w = _weights_path(inp.dates, inp.ic, inp.factors, lookback, inp.min_months, shrink, state)
    return w.reindex(inp.model_dates)


def weights_ic_raw(inp: Inputs) -> pd.DataFrame:
    rows = {}
    for t in inp.model_dates:
        mean = _past(inp.ic, t).iloc[-inp.lookback :].mean()
        rows[t] = _normalise(mean.reindex(inp.factors))
    return _frame(rows, inp.factors)


def weights_fama_macbeth(inp: Inputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """w ∝ mean slope x t^2 / (1 + t^2) over the trailing FM_LOOKBACK months (no sign constraint)."""
    slopes = fm_slopes(inp.panel, inp.zcols)
    slopes.columns = inp.factors
    rows = {}
    for t in inp.model_dates:
        past = _past(slopes, t).iloc[-FM_LOOKBACK:]
        if len(past) < 3:
            rows[t] = {f: 0.0 for f in inp.factors}
            continue
        m = past.mean()
        sd = past.std(ddof=1)
        tt = (m / (sd / np.sqrt(past.count()))).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rows[t] = _normalise(m * tt**2 / (1.0 + tt**2))
    return _frame(rows, inp.factors), slopes


def _ridge_blocks(inp: Inputs) -> dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]:
    """Per formation date: (X = z with missing -> 0, y = uniform rank of fwd_ret) of the rows with a return."""
    out = {}
    for t, g in inp.panel.groupby("date", sort=True):
        g = g[g["fwd_ret"].notna()]
        if len(g) < 10:
            continue
        x = np.nan_to_num(g[inp.zcols].to_numpy(dtype=float), nan=0.0)
        out[pd.Timestamp(t)] = (x, uniform_rank_array(g["fwd_ret"].to_numpy(dtype=float)))
    return out


def ridge_fit(blocks: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, float, dict[float, float]]:
    """(beta, chosen alpha, alpha -> mean leave-one-month-out rank IC) of the pooled ridge on ``blocks``."""
    k = blocks[0][0].shape[1]
    a_m = [x.T @ x for x, _ in blocks]
    b_m = [x.T @ y for x, y in blocks]
    a, b = sum(a_m), sum(b_m)
    n_rows = sum(len(y) for _, y in blocks)
    eye = np.eye(k)
    cv: dict[float, float] = {}
    for alpha in RIDGE_ALPHAS:
        lam = alpha * n_rows / 3.0
        ics = []
        if len(blocks) >= 2:
            for (x, y), am, bm in zip(blocks, a_m, b_m, strict=True):
                beta = np.linalg.solve(a - am + lam * eye, b - bm)
                ic = _spearman(x @ beta, y)
                if np.isfinite(ic):
                    ics.append(ic)
        cv[alpha] = float(np.mean(ics)) if ics else float("nan")
    finite = {al: v for al, v in cv.items() if np.isfinite(v)}
    # highest CV IC; ties -> the larger alpha (more shrinkage)
    alpha = max(finite, key=lambda al: (finite[al], al)) if finite else max(RIDGE_ALPHAS)
    beta = np.linalg.solve(a + alpha * n_rows / 3.0 * eye, b)
    return beta, float(alpha), cv


def weights_ridge(inp: Inputs) -> tuple[pd.DataFrame, pd.Series]:
    blocks = _ridge_blocks(inp)
    bdates = sorted(blocks)
    rows, alphas = {}, {}
    for t in inp.model_dates:
        win = [d for d in bdates if d < t][-RIDGE_LOOKBACK:]
        if not win:
            rows[t] = {f: 0.0 for f in inp.factors}
            continue
        beta, alpha, _ = ridge_fit([blocks[d] for d in win])
        rows[t] = _normalise(pd.Series(beta, index=inp.factors))
        alphas[t] = alpha
    return _frame(rows, inp.factors), pd.Series(alphas, dtype=float)


def weights_tree(inp: Inputs) -> tuple[pd.DataFrame, dict[str, Any]]:
    """rsmt_tree: expanding-window IC tree fitted at every date on months < t; the date is routed by its own split
    variables and the leaf mean gets the ic_weighted sign constraint + shrink to equal weights."""
    ic_c = inp.ic.dropna()
    pos_of = {t: i for i, t in enumerate(inp.dates)}
    pos = np.array([pos_of.get(t, i) for i, t in enumerate(ic_c.index)], dtype=int)
    cols = [c for c in inp.split_vars.columns if inp.split_vars[c].notna().any()]
    sv = inp.split_vars.reindex(ic_c.index)
    rows, path = {}, []
    tree = None
    for t in inp.model_dates:
        m = np.asarray(ic_c.index < t)
        if m.sum() == 0:
            mean = _past(inp.ic, t).mean().reindex(inp.factors)
            rows[t] = fit_weights(mean, inp.shrink, inp.factors)
            path.append({"date": t, "n": 0, "splits": 0, "leaf_n": 0, "vars": []})
            continue
        tree = fit_ic_tree(ic_c.to_numpy()[m], {c: sv[c].to_numpy(dtype=float)[m] for c in cols}, month_pos=pos[m])
        x_t = {c: float(inp.split_vars.at[t, c]) if t in inp.split_vars.index else np.nan for c in cols}
        leaf = tree.route(x_t)
        rows[t] = fit_weights(pd.Series(leaf.mean, index=inp.factors), inp.shrink, inp.factors)
        path.append(
            {"date": t, "n": int(m.sum()), "splits": tree.n_splits, "leaf_n": leaf.n, "vars": tree.split_vars()}
        )
    split_share = float(np.mean([p["splits"] > 0 for p in path])) if path else 0.0
    var_counts: dict[str, int] = {}
    for p in path:
        for v in p["vars"]:
            var_counts[v] = var_counts.get(v, 0) + 1
    diag = {
        "split_variables": cols,
        "dates": len(path),
        "dates_with_split": int(sum(p["splits"] > 0 for p in path)),
        "split_share": split_share,
        "split_variable_counts": var_counts,
        "latest_tree": None if tree is None else tree.describe(inp.factors),
        "latest_leaves": 1 if tree is None else tree.n_leaves,
        "latest_splits": 0 if tree is None else tree.n_splits,
        "path": [{**p, "date": str(pd.Timestamp(p["date"]).date())} for p in path],
    }
    return _frame(rows, inp.factors), diag


# --------------------------------------------------------------------------------------------- evaluation


def period_stats(ic: pd.Series, qp: pd.DataFrame, months: pd.Index) -> dict[str, Any]:
    """IC summary + net long-short / top-minus-universe performance and long-book turnover over ``months``."""
    out: dict[str, Any] = ic_summary(ic.reindex(months).dropna())
    if qp is None or qp.empty:
        return {**out, "ls_net": perf_stats(pd.Series(dtype=float)), "lo_net": perf_stats(pd.Series(dtype=float))}
    sel = qp[qp["date"].isin(months)]
    later = sel[sel["date"] > qp["date"].min()]  # the first portfolio month has no previous book
    out["ls_net"] = perf_stats(sel["ls_net"])
    out["lo_net"] = perf_stats(sel["lo_net"])
    out["turnover_long"] = float(later["to_long"].mean()) if len(later) else None
    return out


def select_candidate(rows: dict[str, dict[str, Any]], n_params: dict[str, int]) -> tuple[str | None, list[str]]:
    """Apply the pre-declared rule to per-candidate pre-holdout stats ({'pre': {'ic_mean', 'ic_t'}})."""
    order = [c for c in CANDIDATES if c in rows]
    ok = [c for c in order if rows[c].get("available") and np.isfinite(_f(rows[c]["pre"].get("ic_mean")))]
    if not ok:
        return None, ["선택 가능한 후보 없음"]
    best = max(_f(rows[c]["pre"]["ic_mean"]) for c in ok)
    t1 = [c for c in ok if best - _f(rows[c]["pre"]["ic_mean"]) < TIE_IC]
    tval = {c: _f(rows[c]["pre"].get("ic_t"), -np.inf) for c in t1}
    tmax = max(tval.values())
    t2 = [c for c in t1 if tmax - tval[c] < TIE_T] if np.isfinite(tmax) else list(t1)
    win = min(t2, key=lambda c: (n_params.get(c, 99), CANDIDATES.index(c)))
    trace = [
        f"최고 홀드아웃 전 IC 평균 {best:.4f}",
        f"IC 동률(차이 < {TIE_IC}) 후보: {', '.join(t1)}",
        f"IC t 동률(차이 < {TIE_T}) 후보: {', '.join(t2)}",
        f"선택: {win}",
    ]
    return win, trace


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def build_inputs(
    panel: pd.DataFrame,
    settings: Settings,
    regime: pd.DataFrame | None,
    region: str,
    factors: list[str] | None = None,
    split_vars: pd.DataFrame | None = None,
) -> Inputs:
    cfg = settings.backtest.model
    factors = factors or list(FACTORS)
    zcols = [f"z_{f}" for f in factors]
    dates = sorted(pd.to_datetime(panel["date"].unique()))
    ic = pd.DataFrame({f: cross_sectional_ic(panel, z) for f, z in zip(factors, zcols, strict=True)}).sort_index()
    ic = ic.reindex(columns=factors)
    have_regime = regime is not None and len(regime) > 0
    state = regime_state(dates, regime, region) if have_regime else None
    sv = pd.DataFrame(index=pd.DatetimeIndex(dates))
    if split_vars is not None and len(split_vars):
        base = split_vars.copy()
        base.index = pd.DatetimeIndex(base.index)
        sv = base.reindex(sv.index)
    else:
        # fallback without a benchmark: the equal-weight 3M return of the universe (known at t)
        sv["mkt_ret_60d"] = panel.groupby("date")["mom_3m"].mean().reindex(sv.index) if "mom_3m" in panel else np.nan
    if have_regime:
        sv["macro_score"] = regime_score(dates, regime, region).to_numpy()
    sv = sv[[c for c in SPLIT_VARS if c in sv.columns] + [c for c in sv.columns if c not in SPLIT_VARS]]
    return Inputs(
        panel=panel,
        factors=factors,
        zcols=zcols,
        dates=dates,
        model_dates=model_dates(dates, ic, int(cfg.min_months)),
        ic=ic,
        lookback=int(cfg.lookback_months),
        min_months=int(cfg.min_months),
        shrink=float(cfg.shrink_to_equal),
        state=state,
        split_vars=sv,
    )


def candidate_weights(name: str, inp: Inputs) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Weights path (index = model dates) + diagnostics of one candidate."""
    if name == "equal_weight":
        return weights_equal(inp), {}
    if name == "ic_weighted":
        return weights_ic(inp, inp.lookback, inp.shrink), {"lookback": inp.lookback, "shrink": inp.shrink}
    if name == "ic_weighted_60":
        return weights_ic(inp, LOOKBACK_60, inp.shrink), {"lookback": LOOKBACK_60, "shrink": inp.shrink}
    if name == "ic_weighted_raw":
        return weights_ic_raw(inp), {"lookback": inp.lookback, "shrink": 0.0}
    if name == "fama_macbeth":
        w, _ = weights_fama_macbeth(inp)
        return w, {"lookback": FM_LOOKBACK}
    if name == "ridge":
        w, alphas = weights_ridge(inp)
        counts = {str(a): int((alphas == a).sum()) for a in RIDGE_ALPHAS}
        return w, {
            "lookback": RIDGE_LOOKBACK,
            "alpha_counts": counts,
            "latest_alpha": _f(alphas.iloc[-1]) if len(alphas) else None,
        }
    if name == "macro_conditional":
        if inp.state is None:
            raise ValueError("macro_conditional needs a regime history")
        w = weights_ic(inp, inp.lookback, inp.shrink, inp.state)
        known = inp.state.reindex(inp.model_dates)
        return w, {
            "shrink_months": REGIME_SHRINK_MONTHS,
            "months_risk_on": int((known == 1).sum()),
            "months_risk_off": int((known == -1).sum()),
            "months_unknown": int(known.isna().sum()),
        }
    if name == "rsmt_tree":
        return weights_tree(inp)
    raise ValueError(f"unknown factor-model candidate: {name}")


def run_candidate(name: str, inp: Inputs, settings: Settings, cost_bps: float) -> tuple[ModelResult, dict[str, Any]]:
    weights, diag = candidate_weights(name, inp)
    weights = weights.dropna(how="all")
    q = int(settings.backtest.quantiles)
    scores = _scores(inp.panel, weights, inp.zcols, inp.factors)
    if weights.empty:
        ic, oos, qp = pd.Series(dtype=float), {**ic_summary(pd.Series(dtype=float))}, pd.DataFrame()
    else:
        ic, oos, qp = _evaluate(inp.panel, scores, q, cost_bps)
    t = oos.get("ic_t")
    skill_ok = bool(t is not None and np.isfinite(t) and t >= float(settings.backtest.model.skill_min_ic_t))
    res = ModelResult(
        method=name,
        factors=inp.factors,
        weights=weights,
        scores=scores,
        oos_ic=ic,
        oos=oos,
        portfolios=qp,
        skill_ok=skill_ok,
        latest_weights=weights.iloc[-1].to_dict() if len(weights) else {},
        latest_date=weights.index[-1] if len(weights) else None,
        stat_history=inp.ic,
        diagnostics=diag,
    )
    return res, diag


def compare_candidates(
    panel: pd.DataFrame,
    settings: Settings,
    regime: pd.DataFrame | None = None,
    *,
    region: str = "kr",
    cost_bps: float = 0.0,
    factors: list[str] | None = None,
    split_vars: pd.DataFrame | None = None,
) -> tuple[ModelResult, dict[str, ModelResult]]:
    """Run every candidate, apply the selection rule and return (served ModelResult with ``selection``, all)."""
    cfg = settings.backtest.model
    inp = build_inputs(panel, settings, regime, region, factors, split_vars)
    results: dict[str, ModelResult] = {}
    unavailable: dict[str, str] = {}
    for name in CANDIDATES:
        if name == "macro_conditional" and inp.state is None:
            unavailable[name] = "레짐 이력 없음(--no-macro 또는 매크로 비활성)"
            continue
        results[name], _ = run_candidate(name, inp, settings, cost_bps)
    n_params = dict(N_PARAMS)
    if "rsmt_tree" in results:
        d = results["rsmt_tree"].diagnostics or {}
        n_params["rsmt_tree"] = 6 * int(d.get("latest_leaves", 1)) + int(d.get("latest_splits", 0))
    # common OOS months and the holdout
    common: pd.Index | None = None
    for r in results.values():
        idx = r.oos_ic.dropna().index
        common = idx if common is None else common.intersection(idx)
    common = pd.DatetimeIndex(sorted(common)) if common is not None else pd.DatetimeIndex([])
    h = max(int(cfg.holdout_months), 0)
    if len(common) < 2 * h:
        h = len(common) // 2
    pre, hold = common[: len(common) - h], common[len(common) - h :]
    ref_ic = results[REFERENCE].oos_ic if REFERENCE in results else None
    rows: dict[str, dict[str, Any]] = {}
    for name in CANDIDATES:
        if name in unavailable:
            rows[name] = {"available": False, "reason": unavailable[name], "label": LABELS[name]}
            continue
        r = results[name]
        row: dict[str, Any] = {
            "available": bool(len(r.weights)),
            "label": LABELS[name],
            "label_en": LABELS_EN[name],
            "n_params": n_params[name],
            "pre": period_stats(r.oos_ic, r.portfolios, pre),
            "holdout": period_stats(r.oos_ic, r.portfolios, hold),
            "full": period_stats(r.oos_ic, r.portfolios, common),
            "full_all_months": {k: r.oos.get(k) for k in ("n_months", "ic_mean", "ic_t", "ic_hit")},
            "skill_ok": r.skill_ok,
            "latest_weights": r.latest_weights,
        }
        if ref_ic is not None and name != REFERENCE:
            diff = (r.oos_ic.reindex(pre) - ref_ic.reindex(pre)).dropna()
            m, t, n = t_stat(diff)
            row["pre_vs_reference"] = {"mean_diff": m, "t": t, "n": n}
        if r.diagnostics:
            row["diagnostics"] = {k: v for k, v in r.diagnostics.items() if k != "path"}
        rows[name] = row
    fixed = cfg.method if cfg.method != "auto" else None
    winner, trace = select_candidate(rows, n_params)
    served = winner
    notes: list[str] = []
    if fixed is not None:
        if fixed in results:
            served = fixed
            trace.append(f"설정 backtest.model.method={fixed} → 선택 규칙 대신 고정 후보 서빙")
        else:
            served = winner or REFERENCE
            why = unavailable.get(fixed, "결과 없음")
            trace.append(f"고정 후보 {fixed}를 쓸 수 없어({why}) 규칙 선택 {served} 서빙")
    if served is None:
        served = REFERENCE
    out = results[served]
    selection = {
        "version": SELECTION_VERSION,
        "mode": "fixed" if fixed is not None else "auto",
        "candidates": list(CANDIDATES),
        "selected": served,
        "rule_winner": winner,
        "rule": {
            "primary": "홀드아웃 전 월평균 OOS 순위 IC 최대",
            "tie_ic": TIE_IC,
            "tie_t": TIE_T,
            "then": "적합 모수가 적은 후보(같으면 후보 표 순서)",
        },
        "trace": trace,
        "holdout_months": int(h),
        "common_months": len(common),
        "pre_months": len(pre),
        "pre_period": [str(pre[0].date()), str(pre[-1].date())] if len(pre) else None,
        "holdout_period": [str(hold[0].date()), str(hold[-1].date())] if len(hold) else None,
        "rows": rows,
        "skill_rule": f"선택 후보의 전체 OOS IC t ≥ {float(cfg.skill_min_ic_t):.1f} → 필러 전체 가중, 미달 ×0.35",
    }
    if served in rows and rows[served].get("available"):
        sel = rows[served]
        notes.append(
            f"후보 {len([r for r in rows.values() if r.get('available')])}개 비교 → {served} 선택"
            f"(홀드아웃 전 {len(pre)}개월 IC {_f(sel['pre']['ic_mean']):.4f}, t {_f(sel['pre']['ic_t']):.2f}; "
            f"홀드아웃 {len(hold)}개월 IC {_f(sel['holdout']['ic_mean']):.4f}, t {_f(sel['holdout']['ic_t']):.2f})."
        )
        notes.append(
            "승자의 저주: 여러 후보 중 최댓값을 골랐으므로 홀드아웃 전 IC는 위로 치우칩니다. 편향 없는 확인값은 "
            "홀드아웃 IC입니다."
        )
    out.selection = _jsonable(selection)
    out.notes = notes + list(out.notes)
    if "macro_conditional" in results and REFERENCE in results:
        mc, ref = results["macro_conditional"], results[REFERENCE]
        cm = mc.oos_ic.dropna().index.intersection(ref.oos_ic.dropna().index)
        d = mc.diagnostics or {}
        out.macro = {
            "region": region,
            "months_compared": len(cm),
            "months_risk_on": d.get("months_risk_on"),
            "months_risk_off": d.get("months_risk_off"),
            "months_unknown": d.get("months_unknown"),
            "unconditional": ic_summary(ref.oos_ic.reindex(cm)),
            "conditional": ic_summary(mc.oos_ic.reindex(cm)),
            "conditional_portfolio": {k: mc.oos.get(k) for k in ("ls_net", "lo_net")},
            "used": served == "macro_conditional",
            "shrink_months": REGIME_SHRINK_MONTHS,
        }
    return out, results


__all__ = [
    "CANDIDATES",
    "LABELS",
    "N_PARAMS",
    "SELECTION_VERSION",
    "Inputs",
    "build_inputs",
    "candidate_weights",
    "compare_candidates",
    "ridge_fit",
    "run_candidate",
    "select_candidate",
]
