"""Within-date (cross-sectional) evaluation metrics with date-clustered inference.

Every metric is computed PER DATE first (the common market move of a date cancels inside it), then averaged
over dates; the date is the independent unit. Inference: t = mean / (sd / sqrt(n_dates)) of the per-date
series, and a 95% percentile interval from a circular moving-block bootstrap over dates (block = 3 months,
so short-range autocorrelation of the per-date series is kept inside a block).

- ``auc_median`` (PRIMARY since protocol v2, ``PRIMARY_AUC``): within-date AUC of the score against the per-date
  MEDIAN label 1{y_e > median_date(y_e)} - balanced classes on every date (the target of a long/short book).
  Chosen after the v1 results were seen (post hoc): see ``modeling.selection``.
- ``auc`` (secondary): within-date AUC of the score against 1{y_e > 0} (beta-adjusted excess return beats 0).
  0.5 = no skill for both.
- ``ic`` (secondary): within-date Spearman rank IC of the score vs y_e.
- ``top_excess_net``: equal-weight top-quintile simple return minus the equal-weight universe return of the
  date, minus one-way turnover of the top-quintile book x cost_bps / 10,000 (turnover measured against the
  previous evaluation date's book). Reported per period and annualised (x 12 for month-end dates).
- ``top_ye_net`` (PRIMARY since protocol v4, ``PRIMARY``, docs/MODELING.md 79): the same top-quintile book on the
  label y_e (since dataset v5 the SIMPLE beta-adjusted excess return): mean y_e of the top quintile minus the
  date's mean y_e, minus turnover x cost - the net excess of "buy the top quintile, hedge beta with the index".
  Rank metrics (AUC, IC) reward picking names that rarely fall below the median (low volatility) even when that
  earns no mean return (docs/MODELING.md 78.1); this one counts the money. No skill = 0.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

BOOT_BLOCK = 3
BOOT_SIMS = 2000
MIN_NAMES = 20
PERIODS_PER_YEAR = 12
PRIMARY_AUC = "auc_median"  # the reported AUC (protocol v2/v3 primary; secondary since v4); v1 used "auc"
SECONDARY_AUC = "auc"
PRIMARY = "top_ye_net"  # protocol v4 primary metric key (``evaluate`` output, per month)
PRIMARY_NULL = 0.0  # the no-skill value of PRIMARY (an AUC's is 0.5)
AUC_LABELS: dict[str, tuple[str, str]] = {  # metric key -> (Korean, English) display label
    "auc_median": ("날짜 내 AUC(중앙값 초과)", "within-date AUC (above median)"),
    "auc": ("날짜 내 AUC(y_e>0)", "within-date AUC (y_e > 0)"),
}
METRIC_LABELS: dict[str, tuple[str, str]] = {
    **AUC_LABELS,
    "top_ye_net": (
        "상위 20% 베타 조정 초과수익(비용 후, 월)",
        "top-quintile beta-adjusted excess (net of cost, per month)",
    ),
}


def auc_label(key: str = PRIMARY_AUC, lang: str = "ko") -> str:
    """Display label of an AUC metric key (always says which label the AUC is computed against)."""
    ko, en = AUC_LABELS.get(key, (key, key))
    return en if lang == "en" else ko


def metric_label(key: str = PRIMARY, lang: str = "ko") -> str:
    """Display label of any metric key (the AUCs and the v4 primary)."""
    ko, en = METRIC_LABELS.get(key, (key, key))
    return en if lang == "en" else ko


def per_date_auc(
    frame: pd.DataFrame, score: str = "score", label: str = "y_e", min_names: int = MIN_NAMES
) -> pd.Series:
    """AUC(score, 1{label > 0}) per date (NaN when a date has < min_names rows or a single class)."""
    d = frame[["date", score, label]].dropna()
    if d.empty:
        return pd.Series(dtype=float)
    pos = (d[label] > 0).astype(float)
    rk = d[score].groupby(d["date"]).rank(method="average")
    g = pd.DataFrame({"date": d["date"], "pos": pos, "rkpos": rk * pos})
    agg = g.groupby("date").agg(n=("pos", "size"), n1=("pos", "sum"), s=("rkpos", "sum"))
    n0 = agg["n"] - agg["n1"]
    auc = (agg["s"] - agg["n1"] * (agg["n1"] + 1) / 2.0) / (agg["n1"] * n0)
    ok = (agg["n"] >= min_names) & (agg["n1"] > 0) & (n0 > 0)
    return auc.where(ok).dropna().sort_index()


def per_date_auc_median(frame: pd.DataFrame, score: str = "score", label: str = "y_e") -> pd.Series:
    """PRIMARY (protocol v2): within-date AUC against 1{label > date median} (balanced classes on every date)."""
    d = frame[["date", score, label]].dropna().copy()
    if d.empty:
        return pd.Series(dtype=float)
    d["_c"] = d[label] - d[label].groupby(d["date"]).transform("median")
    return per_date_auc(d, score, "_c")


def per_date_top_ye_net(
    frame: pd.DataFrame, score: str = "score", cost_bps: float = 0.0, label: str = "y_e", q: int = 5
) -> pd.Series:
    """PRIMARY (protocol v4): per date, mean label of the top 1/q by score minus the date's mean label, minus the
    one-way turnover of that book (vs the previous date in ``frame``; the first date counts a full build) x
    cost_bps / 10,000. A date whose scores are all equal is "no view" (docs/MODELING.md 83.2): 0 (the universe
    itself), and the next book counts a full build - never a top fifth filled in row order."""
    d = frame[["date", "entity", score, label]].dropna()
    out: dict[pd.Timestamp, float] = {}
    prev: set[str] = set()
    for t, g in d.groupby("date", sort=True):
        if len(g) < MIN_NAMES:
            continue
        if g[score].nunique() <= 1:
            out[t] = 0.0
            prev = set()
            continue
        k = max(1, int(round(len(g) / q)))
        top = g.nlargest(k, score)
        names = set(top["entity"])
        turnover = 1.0 if not prev else len(names - prev) / len(names)
        out[t] = float(top[label].mean() - g[label].mean() - turnover * float(cost_bps) / 1e4)
        prev = names
    return pd.Series(out, dtype=float).sort_index()


def per_date_primary(frame: pd.DataFrame, score: str = "score", cost_bps: float = 0.0) -> pd.Series:
    """Per-date values of the PRIMARY metric (``PRIMARY``) of ``score`` vs y_e (``cost_bps``: the v4 book cost)."""
    if PRIMARY == "top_ye_net":
        return per_date_top_ye_net(frame, score, cost_bps)
    return {"auc": per_date_auc, "auc_median": per_date_auc_median}[PRIMARY](frame, score)


def per_date_ic(frame: pd.DataFrame, score: str = "score", label: str = "y_e", min_names: int = MIN_NAMES) -> pd.Series:
    """Spearman rank correlation of score and label per date."""
    d = frame[["date", score, label]].dropna()
    if d.empty:
        return pd.Series(dtype=float)
    a = d[score].groupby(d["date"]).rank(method="average")
    b = d[label].groupby(d["date"]).rank(method="average")
    a = a - a.groupby(d["date"]).transform("mean")
    b = b - b.groupby(d["date"]).transform("mean")
    g = pd.DataFrame({"date": d["date"], "ab": a * b, "aa": a * a, "bb": b * b})
    agg = g.groupby("date").agg(n=("ab", "size"), ab=("ab", "sum"), aa=("aa", "sum"), bb=("bb", "sum"))
    ic = agg["ab"] / np.sqrt(agg["aa"] * agg["bb"])
    return ic.where((agg["n"] >= min_names) & (agg["aa"] > 0) & (agg["bb"] > 0)).dropna().sort_index()


def top_quintile(frame: pd.DataFrame, cost_bps: float, score: str = "score", q: int = 5) -> pd.DataFrame:
    """Per date: top-quintile EW simple return minus the universe EW return (gross, turnover, net) and the
    same on the beta-adjusted excess y_e (``gross_ye``)."""
    d = frame[["date", "entity", score, "r", "y_e"]].dropna(subset=[score, "r"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["gross", "turnover", "net", "gross_ye", "spread", "n_top"])
    d["R"] = np.expm1(d["r"])
    rows = []
    prev: set[str] = set()
    for t, g in d.groupby("date", sort=True):
        if len(g) < MIN_NAMES:
            continue
        k = max(1, int(round(len(g) / q)))
        top = g.nlargest(k, score)
        names = set(top["entity"])
        turnover = 1.0 if not prev else len(names - prev) / len(names)
        gross = float(top["R"].mean() - g["R"].mean())
        gross_ye = float(top["y_e"].mean() - g["y_e"].mean())
        bottom = g.nsmallest(k, score)
        spread = float(top["R"].mean() - bottom["R"].mean())
        rows.append(
            {"date": t, "gross": gross, "turnover": turnover, "gross_ye": gross_ye, "spread": spread, "n_top": k}
        )
        prev = names
    out = pd.DataFrame(rows).set_index("date")
    out["net"] = out["gross"] - out["turnover"] * float(cost_bps) / 1e4
    return out


def block_bootstrap_means(x: np.ndarray, block: int = BOOT_BLOCK, n_boot: int = BOOT_SIMS, seed: int = 0) -> np.ndarray:
    """Circular moving-block bootstrap draws of the mean of a per-date series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.array([])
    b = max(1, min(int(block), n))
    n_blocks = int(math.ceil(n / b))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % n
    idx = idx.reshape(n_boot, -1)[:, :n]
    return x[idx].mean(axis=1)


def summarize(series: pd.Series | np.ndarray, null: float = 0.0, seed: int = 0) -> dict[str, Any]:
    """mean, t vs ``null`` (date-clustered), 95% block-bootstrap interval, share above null, n dates."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {"mean": None, "t": None, "lo": None, "hi": None, "n": 0, "hit": None}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    t = (mean - null) / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
    draws = block_bootstrap_means(x, seed=seed)
    lo, hi = (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))) if draws.size else (None, None)
    return {
        "mean": mean,
        "t": None if not math.isfinite(t) else float(t),
        "lo": lo,
        "hi": hi,
        "n": n,
        "hit": float((x > null).mean()),
    }


def evaluate(frame: pd.DataFrame, cost_bps: float, score: str = "score", seed: int = 0) -> dict[str, Any]:
    """Within-date AUC, rank IC and top-quintile excess (net of cost) of ``score`` on the rows given.

    ``frame`` needs date, entity, score, y_e and r (month-end OOS rows).
    """
    auc = per_date_auc(frame, score)
    ic = per_date_ic(frame, score)
    tq = top_quintile(frame, cost_bps, score)
    out = {
        "auc": summarize(auc, 0.5, seed),
        "ic": summarize(ic, 0.0, seed + 1),
        "top_net": summarize(tq["net"] if len(tq) else [], 0.0, seed + 2),
        "top_gross": summarize(tq["gross"] if len(tq) else [], 0.0, seed + 3),
        "top_ye": summarize(tq["gross_ye"] if len(tq) else [], 0.0, seed + 4),
        "q5_q1": summarize(tq["spread"] if len(tq) else [], 0.0, seed + 5),
        "auc_median": summarize(per_date_auc_median(frame, score), 0.5, seed + 6),
        "top_ye_net": summarize(per_date_top_ye_net(frame, score, cost_bps), 0.0, seed + 7),
        "turnover": float(tq["turnover"].mean()) if len(tq) else None,
        "n_dates": int(len(auc)),
        "n_rows": int(frame[score].notna().sum()),
        "first_date": str(pd.Timestamp(frame["date"].min()).date()) if len(frame) else None,
        "last_date": str(pd.Timestamp(frame["date"].max()).date()) if len(frame) else None,
    }
    tn = out["top_net"]
    out["top_net_ann"] = None if tn["mean"] is None else tn["mean"] * PERIODS_PER_YEAR
    ty = out["top_ye_net"]
    out["top_ye_net_ann"] = None if ty["mean"] is None else ty["mean"] * PERIODS_PER_YEAR
    return out


def paired_difference(a: pd.Series, b: pd.Series, seed: int = 0) -> dict[str, Any]:
    """Per-date difference a - b over common dates (e.g. AUC of a model minus a baseline)."""
    common = a.index.intersection(b.index)
    return summarize((a.reindex(common) - b.reindex(common)).to_numpy(), 0.0, seed)


__all__ = [
    "AUC_LABELS",
    "BOOT_BLOCK",
    "BOOT_SIMS",
    "METRIC_LABELS",
    "PRIMARY",
    "PRIMARY_AUC",
    "PRIMARY_NULL",
    "SECONDARY_AUC",
    "auc_label",
    "block_bootstrap_means",
    "evaluate",
    "metric_label",
    "paired_difference",
    "per_date_auc",
    "per_date_auc_median",
    "per_date_ic",
    "per_date_primary",
    "per_date_top_ye_net",
    "summarize",
    "top_quintile",
]
