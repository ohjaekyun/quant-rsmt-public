"""Walk-forward multi-factor model (docs/BACKTEST_MACRO.md section 3).

At every rebalance date t the factor weights are estimated from the monthly cross-sections BEFORE t only (the
cross-section of month s is complete once its forward return is realised at the next rebalance date <= t). This
module holds the statistics (monthly IC, t-stats, quantile portfolios, performance), the ``ic_weighted`` weight fit
(:func:`fit_weights`: sign constraint to the a-priori direction, |w| normalisation, shrink to equal weights), the
Fama-MacBeth slopes and the macro regime state / score of a formation date.

:func:`walk_forward_model` runs the pre-declared candidate set of ``backtest.candidates`` (equal weight, three IC
weightings, Fama-MacBeth, pooled ridge, macro-conditioned IC weights, an RSMT-style IC tree), evaluates each out of
sample on the common months with the last ``holdout_months`` held out, applies the selection rule (or the fixed
``backtest.model.method``) and returns the served candidate's :class:`ModelResult` with the comparison in
``selection``. No weights (and no score) exist before ``min_months`` past months. The score is sum_f w_f * z_f (a
missing z counts as 0 = neutral).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantcio.backtest.factors import EXPECTED_SIGN
from quantcio.config import Settings

MIN_IC_NAMES = 10  # a monthly IC needs at least this many names with a signal and a return
REGIME_SHRINK_MONTHS = 12.0  # conditional IC mean = (n_c * m_c + k * m_all) / (n_c + k)
REGIME_MAX_LAG_DAYS = 10
# pillar side test (pre-declared 2026-09-30, docs/BACKTEST_MACRO.md 3.3): one side of the served pillar is "weak" when
# the pre-holdout Fama-MacBeth slope difference b+ - b- has |t| >= this
SIDE_MIN_T = 2.0


# --------------------------------------------------------------------------------------------- statistics


def cross_sectional_ic(panel: pd.DataFrame, col: str, min_names: int = MIN_IC_NAMES) -> pd.Series:
    """Monthly Spearman IC of ``col`` vs ``fwd_ret`` (index: formation date)."""
    d = panel[["date", col, "fwd_ret"]].dropna()
    if d.empty:
        return pd.Series(dtype=float)
    out = {}
    for t, g in d.groupby("date"):
        if len(g) < min_names or g[col].nunique() < 2 or g["fwd_ret"].nunique() < 2:
            continue
        out[t] = float(g[col].rank().corr(g["fwd_ret"].rank()))
    return pd.Series(out, dtype=float).sort_index()


def t_stat(x: pd.Series) -> tuple[float, float, int]:
    """(mean, t, n) of a series (t = mean / (sd / sqrt(n)); NaN when n < 3 or sd == 0)."""
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 3:
        return (float(x.mean()) if n else float("nan")), float("nan"), n
    sd = float(x.std(ddof=1))
    m = float(x.mean())
    return m, (m / (sd / np.sqrt(n)) if sd > 0 else float("nan")), n


def p_value_two_sided(t: float, n: int) -> float:
    if not np.isfinite(t) or n < 3:
        return float("nan")
    from scipy import stats

    return float(2.0 * stats.t.sf(abs(t), df=n - 1))


def ic_summary(ic: pd.Series) -> dict[str, Any]:
    m, t, n = t_stat(ic)
    hit = float((ic.dropna() > 0).mean()) if n else float("nan")
    return {"n_months": n, "ic_mean": m, "ic_t": t, "ic_hit": hit, "ic_p": p_value_two_sided(t, n)}


def perf_stats(r: pd.Series) -> dict[str, Any]:
    """Annualised geometric return, volatility, Sharpe (mean*12 / vol), max drawdown and monthly win rate."""
    r = pd.Series(r).dropna()
    n = len(r)
    if n == 0:
        return {"n": 0, "ann_return": None, "ann_vol": None, "sharpe": None, "mdd": None, "win_rate": None}
    growth = float(np.prod(1.0 + r.to_numpy()))
    ann = growth ** (12.0 / n) - 1.0 if growth > 0 else -1.0
    vol = float(r.std(ddof=1) * np.sqrt(12)) if n > 1 else float("nan")
    sharpe = float(r.mean() * 12 / vol) if n > 1 and vol > 0 else float("nan")
    wealth = np.cumprod(1.0 + r.to_numpy())
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    mdd = float(np.min(wealth / peak - 1.0))
    return {
        "n": n,
        "ann_return": float(ann),
        "ann_vol": vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": float((r > 0).mean()),
    }


def bucket_of(x: pd.Series, q: int) -> pd.Series:
    """Quantile bucket 1..q from average ranks (ties share a bucket)."""
    n = x.notna().sum()
    r = x.rank(method="average")
    return np.floor((r - 0.5) / max(n, 1) * q).clip(0, q - 1) + 1


def _turnover(new: dict[str, float], old: dict[str, float], old_ret: dict[str, float]) -> float:
    """One-way turnover 0.5 * sum|w_new - w_old_drifted| (the old book drifted by its realised returns)."""
    if old:
        grown = {e: w * (1.0 + old_ret.get(e, 0.0)) for e, w in old.items()}
        tot = sum(grown.values())
        drifted = {e: v / tot for e, v in grown.items()} if tot > 0 else {}
    else:
        drifted = {}
    keys = set(new) | set(drifted)
    return 0.5 * sum(abs(new.get(e, 0.0) - drifted.get(e, 0.0)) for e in keys)


def quantile_portfolios(panel: pd.DataFrame, col: str, q: int, cost_bps: float) -> pd.DataFrame:
    """Monthly (formation date) returns of the quantile portfolios of ``col``.

    Columns: q1..q<q> (equal-weight mean fwd_ret), univ (equal-weight all members with a return), n, ls, lo (top -
    universe), to_long, to_short (one-way turnover of the top / bottom book), ls_net, lo_net (after
    turnover * cost_bps / 10,000 on each traded book), end (holding-period end = next formation date).
    """
    dates = sorted(panel["date"].unique())
    end = dict(zip(dates[:-1], dates[1:], strict=False))
    rows = []
    prev_top: dict[str, float] = {}
    prev_bot: dict[str, float] = {}
    prev_ret: dict[str, float] = {}
    c = float(cost_bps) / 10_000.0
    for t, g in panel.groupby("date", sort=True):
        univ = g["fwd_ret"].dropna()
        e = g[[col, "fwd_ret", "entity"]].dropna()
        if len(e) < q * 2 or e[col].nunique() < 2:
            continue
        b = bucket_of(e[col], q)
        row: dict[str, Any] = {"date": t, "end": end.get(t), "n": len(e), "univ": float(univ.mean())}
        for k in range(1, q + 1):
            row[f"q{k}"] = float(e.loc[b == k, "fwd_ret"].mean()) if (b == k).any() else np.nan
        top = e.loc[b == q, "entity"].tolist()
        bot = e.loc[b == 1, "entity"].tolist()
        new_top = {x: 1.0 / len(top) for x in top} if top else {}
        new_bot = {x: 1.0 / len(bot) for x in bot} if bot else {}
        row["to_long"] = _turnover(new_top, prev_top, prev_ret)
        row["to_short"] = _turnover(new_bot, prev_bot, prev_ret)
        row["ls"] = row[f"q{q}"] - row["q1"]
        row["lo"] = row[f"q{q}"] - row["univ"]
        row["ls_net"] = row["ls"] - (row["to_long"] + row["to_short"]) * c
        row["lo_net"] = row["lo"] - row["to_long"] * c
        rows.append(row)
        prev_top, prev_bot = new_top, new_bot
        prev_ret = dict(zip(e["entity"], e["fwd_ret"], strict=False))
    return pd.DataFrame(rows)


def portfolio_summary(qp: pd.DataFrame, q: int) -> dict[str, Any]:
    """Performance of the long-short and long-only-excess series (gross / net) + average turnover."""
    if qp.empty:
        return {"ls": perf_stats(pd.Series(dtype=float)), "ls_net": perf_stats(pd.Series(dtype=float))}
    later = qp.iloc[1:] if len(qp) > 1 else qp
    out = {
        "ls": perf_stats(qp["ls"]),
        "ls_net": perf_stats(qp["ls_net"]),
        "lo": perf_stats(qp["lo"]),
        "lo_net": perf_stats(qp["lo_net"]),
        "top": perf_stats(qp[f"q{q}"]),
        "universe": perf_stats(qp["univ"]),
        "turnover_long": float(later["to_long"].mean()),
        "turnover_short": float(later["to_short"].mean()),
        "quantile_mean_monthly": {f"q{k}": float(qp[f"q{k}"].mean()) for k in range(1, q + 1)},
    }
    return out


def pillar_side_test(
    scores: pd.DataFrame, panel: pd.DataFrame, holdout_start: pd.Timestamp | None = None
) -> dict[str, Any]:
    """Does the served factor pillar carry information on BOTH sides of the median?

    Per formation date: OLS of the demeaned ``fwd_ret`` on p+ = max(p, 0) and p- = min(p, 0), where
    p = 2·(percentile − 0.5) is the served pillar before the skill shrink (``score_frame`` percentile). Fama-MacBeth
    mean / t of b+, b- and b+ − b- over the pre-holdout months decide (the same months as the candidate choice); the
    holdout and all months are reported only. ``weak_side`` = 'neg' when t(b+ − b-) >= SIDE_MIN_T (the model's low
    scores say little), 'pos' when <= −SIDE_MIN_T, else None (symmetric pillar)."""
    d = scores[["date", "entity", "score"]].merge(panel[["date", "entity", "fwd_ret"]], on=["date", "entity"])
    d = d.dropna(subset=["score", "fwd_ret"])
    rows = []
    for t, g in d.groupby("date", sort=True):
        if len(g) < MIN_IC_NAMES or g["score"].nunique() < 2:
            continue
        pct = (g["score"].rank(method="average") - 0.5) / len(g)
        p = (2.0 * (pct - 0.5)).clip(-1.0, 1.0).to_numpy()
        x = np.column_stack([np.ones(len(g)), np.maximum(p, 0.0), np.minimum(p, 0.0)])
        y = (g["fwd_ret"] - g["fwd_ret"].mean()).to_numpy(dtype=float)
        b = np.linalg.lstsq(x, y, rcond=None)[0]
        rows.append((pd.Timestamp(t), float(b[1]), float(b[2])))
    s = pd.DataFrame(rows, columns=["date", "b_pos", "b_neg"])

    def block(g: pd.DataFrame) -> dict[str, Any]:
        bp, tp, n = t_stat(g["b_pos"])
        bn, tn, _ = t_stat(g["b_neg"])
        df, td, _ = t_stat(g["b_pos"] - g["b_neg"])
        return {"n": n, "b_pos": bp, "t_pos": tp, "b_neg": bn, "t_neg": tn, "diff": df, "t_diff": td}

    h0 = pd.Timestamp(holdout_start) if holdout_start is not None else None
    pre = s[s["date"] < h0] if h0 is not None else s
    out: dict[str, Any] = {"pre": block(pre), "holdout": block(s[s["date"] >= h0]) if h0 is not None else None}
    out["all"] = block(s)
    td = out["pre"]["t_diff"]
    out["weak_side"] = (
        None if not np.isfinite(td) else ("neg" if td >= SIDE_MIN_T else "pos" if td <= -SIDE_MIN_T else None)
    )
    out["rule"] = f"pre-holdout t(b+ - b-) vs ±{SIDE_MIN_T:g}"
    return out


# --------------------------------------------------------------------------------------------- weights


def fit_weights(
    stat_mean: pd.Series,
    shrink: float,
    factors: list[str] | None = None,
    equal_over: str = "all",
) -> dict[str, float]:
    """IC (or slope) means -> sign-constrained, |w|-normalised weights shrunk towards equal weights.

    ``equal_over='all'`` (the model): the equal-weight target covers every factor in its a-priori direction
    (``volume_trend``: the sign of its fitted statistic), so a factor zeroed by the sign constraint keeps
    shrink / k. ``'admissible'`` (sensitivity only): equal weights over the factors that passed the sign constraint.
    """
    factors = factors or list(stat_mean.index)
    m = stat_mean.reindex(factors).astype(float).fillna(0.0)
    raw = {}
    for f in factors:
        prior = EXPECTED_SIGN.get(f, 0)
        v = float(m[f])
        raw[f] = 0.0 if prior != 0 and np.sign(v) != prior else v
    tot = sum(abs(v) for v in raw.values())
    fitted = {f: v / tot for f, v in raw.items()} if tot > 0 else {f: 0.0 for f in factors}
    eq_sign = {}
    for f in factors:
        prior = EXPECTED_SIGN.get(f, 0)
        eq_sign[f] = float(prior if prior != 0 else np.sign(float(m[f])))
        if equal_over == "admissible" and raw[f] == 0.0:
            eq_sign[f] = 0.0
    k = sum(1 for v in eq_sign.values() if v != 0) or 1
    equal = {f: eq_sign[f] / k for f in factors}
    s = float(np.clip(shrink, 0.0, 1.0))
    w = {f: (1.0 - s) * fitted[f] + s * equal[f] for f in factors}
    tot = sum(abs(v) for v in w.values())
    return {f: (v / tot if tot > 0 else 0.0) for f, v in w.items()}


def fm_slopes(panel: pd.DataFrame, zcols: list[str]) -> pd.DataFrame:
    """Monthly cross-sectional OLS slopes of fwd_ret on the z-scores (missing z -> 0), index formation date."""
    out = {}
    for t, g in panel.groupby("date"):
        g = g.dropna(subset=["fwd_ret"])
        if len(g) < max(MIN_IC_NAMES, len(zcols) + 5):
            continue
        X = np.column_stack([np.ones(len(g)), g[zcols].fillna(0.0).to_numpy(dtype=float)])
        y = g["fwd_ret"].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[t] = beta[1:]
    return pd.DataFrame.from_dict(out, orient="index", columns=zcols).sort_index()


def regime_score(dates: list[pd.Timestamp], regime: pd.DataFrame | None, region: str) -> pd.Series:
    """Macro regime score (NaN unknown) of each formation date.

    The regime row of the same calendar month is used (``macro.regime_history`` stamps calendar month-ends, which
    carry no data after the last session of the month); otherwise the latest row at most 10 days before the date.
    """
    idx = pd.DatetimeIndex(dates)
    if regime is None or regime.empty:
        return pd.Series(np.nan, index=idx)
    r = regime[regime["region"].astype(str).str.lower() == region.lower()][["date", "score"]].dropna()
    if r.empty:
        return pd.Series(np.nan, index=idx)
    r = r.assign(date=pd.to_datetime(r["date"]).dt.normalize()).sort_values("date")
    by_month = r.groupby(r["date"].dt.to_period("M"))["score"].last()
    score = pd.Series(idx.to_period("M").map(lambda p: by_month.get(p, np.nan)), index=idx, dtype=float)
    miss = score.isna()
    if miss.any():
        left = pd.DataFrame({"t": idx[miss.to_numpy()]}).sort_values("t")
        j = pd.merge_asof(left, r, left_on="t", right_on="date", direction="backward")
        fresh = (j["t"] - j["date"]).dt.days <= REGIME_MAX_LAG_DAYS
        score[pd.DatetimeIndex(j["t"])] = np.where(fresh, j["score"], np.nan)
    return score


def regime_state(dates: list[pd.Timestamp], regime: pd.DataFrame | None, region: str) -> pd.Series:
    """+1 risk-on (score >= 0) / -1 risk-off / NaN unknown for each formation date (see :func:`regime_score`)."""
    score = regime_score(dates, regime, region)
    return pd.Series(np.where(score.isna(), np.nan, np.where(score >= 0, 1.0, -1.0)), index=score.index)


# --------------------------------------------------------------------------------------------- model


@dataclass
class ModelResult:
    method: str
    factors: list[str]
    weights: pd.DataFrame  # index formation date, columns factors (only dates with a model)
    scores: pd.DataFrame  # date, entity, score
    oos_ic: pd.Series  # monthly IC of the model score
    oos: dict[str, Any]  # IC summary + portfolio summary
    portfolios: pd.DataFrame  # quantile_portfolios of the score
    skill_ok: bool
    latest_weights: dict[str, float]
    latest_date: pd.Timestamp | None
    stat_history: pd.DataFrame  # monthly factor IC (or FM slope) used for fitting
    macro: dict[str, Any] | None = None
    sensitivity: dict[str, Any] | None = None  # diagnostics only, never used for serving
    notes: list[str] = field(default_factory=list)
    selection: dict[str, Any] | None = None  # candidate comparison + selection (backtest.candidates)
    diagnostics: dict[str, Any] | None = None  # candidate-specific (ridge alphas, tree splits, ...)

    def to_dict(self) -> dict[str, Any]:
        w = self.weights
        return {
            "method": self.method,
            "factors": list(self.factors),
            "skill_ok": bool(self.skill_ok),
            "oos": _jsonable(self.oos),
            "latest_date": None if self.latest_date is None else str(pd.Timestamp(self.latest_date).date()),
            "latest_weights": {k: float(v) for k, v in self.latest_weights.items()},
            "first_model_date": None if w.empty else str(w.index[0].date()),
            "weights_history": {
                str(pd.Timestamp(t).date()): {f: round(float(v), 4) for f, v in row.items()} for t, row in w.iterrows()
            },
            "macro": _jsonable(self.macro) if self.macro else None,
            "sensitivity": _jsonable(self.sensitivity) if self.sensitivity else None,
            "notes": list(self.notes),
            "selection": _jsonable(self.selection) if self.selection else None,
            "diagnostics": _jsonable(self.diagnostics) if self.diagnostics else None,
        }


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    if isinstance(x, np.bool_ | bool):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        v = float(x)
        return None if not np.isfinite(v) else v
    return x


def _weights_path(
    dates: list[pd.Timestamp],
    hist: pd.DataFrame,
    factors: list[str],
    lookback: int,
    min_months: int,
    shrink: float,
    state: pd.Series | None = None,
    equal_over: str = "all",
) -> pd.DataFrame:
    """Weights at every date from the monthly statistics of strictly earlier formation dates."""
    rows = {}
    for t in dates:
        past = hist[hist.index < t].dropna(how="all")
        if len(past) < min_months:
            continue
        past = past.iloc[-lookback:]
        mean = past.mean()
        if state is not None and pd.notna(state.get(t)):
            st = state.reindex(past.index)
            same = past[st == state[t]]
            n_c = len(same)
            if n_c:
                mean = (n_c * same.mean() + REGIME_SHRINK_MONTHS * mean) / (n_c + REGIME_SHRINK_MONTHS)
        mean.index = factors
        rows[t] = fit_weights(mean, shrink, factors, equal_over)
    return pd.DataFrame.from_dict(rows, orient="index", columns=factors).sort_index()


def _scores(panel: pd.DataFrame, weights: pd.DataFrame, zcols: list[str], factors: list[str]) -> pd.DataFrame:
    parts = []
    groups = dict(tuple(panel.groupby("date")))
    for t, w in weights.iterrows():
        g = groups.get(t)
        if g is None or g.empty:
            continue
        z = g[zcols].to_numpy(dtype=float)
        allnan = np.isnan(z).all(axis=1)
        s = np.nan_to_num(z, nan=0.0) @ w[factors].to_numpy(dtype=float)
        s[allnan] = np.nan
        parts.append(pd.DataFrame({"date": t, "entity": g["entity"].to_numpy(), "score": s}))
    if not parts:
        return pd.DataFrame(columns=["date", "entity", "score"])
    return pd.concat(parts, ignore_index=True)


def _evaluate(panel: pd.DataFrame, scores: pd.DataFrame, q: int, cost_bps: float) -> tuple[pd.Series, dict, Any]:
    sp = panel[["date", "entity", "fwd_ret"]].merge(scores, on=["date", "entity"], how="inner")
    ic = cross_sectional_ic(sp, "score")
    qp = quantile_portfolios(sp, "score", q, cost_bps)
    oos = {**ic_summary(ic), **portfolio_summary(qp, q)}
    return ic, oos, qp


def walk_forward_model(
    panel: pd.DataFrame,
    settings: Settings,
    regime: pd.DataFrame | None = None,
    *,
    region: str | None = None,
    cost_bps: float = 0.0,
    factors: list[str] | None = None,
    split_vars: pd.DataFrame | None = None,
) -> ModelResult:
    """Walk-forward multi-factor model on a factor panel (``compute_factor_panel`` output; uses the ``z_<f>``
    columns): every pre-declared candidate is fitted and evaluated, and the selected (``backtest.model.method:
    auto``) or fixed candidate is returned with the comparison in ``selection``.

    ``regime`` (date, region, score) enables ``macro_conditional`` and the macro split variable of ``rsmt_tree`` for
    ``region`` ('kr' / 'us'; default from the panel's attrs or 'kr'). ``split_vars`` (index formation date; columns
    ``mkt_ret_60d``, ``mkt_vol_60d``) are the market variables known at t (default: the universe's equal-weight 3M
    return only). ``cost_bps`` is the round-trip cost per 100% turnover."""
    from quantcio.backtest.candidates import compare_candidates

    region = (region or str(panel.attrs.get("region") or "kr")).lower()
    served, _ = compare_candidates(
        panel, settings, regime, region=region, cost_bps=cost_bps, factors=factors, split_vars=split_vars
    )
    if served.weights.empty:
        n_min = settings.backtest.model.min_months
        served.notes.append(f"과거 월별 단면이 {n_min}개월 미만이라 모델 점수를 만들지 않았습니다.")
    return served


__all__ = [
    "ModelResult",
    "bucket_of",
    "cross_sectional_ic",
    "fit_weights",
    "ic_summary",
    "perf_stats",
    "portfolio_summary",
    "quantile_portfolios",
    "regime_score",
    "regime_state",
    "walk_forward_model",
]
