"""Market-direction component: P(benchmark h-session return > 0) from observable stress / macro features.

Kept separate from stock selection on purpose: the common market move dominates the old pooled P(up) AUC and
stock features cannot predict it. Here one row per month-end date (non-overlapping h-session labels):
features = a small pre-declared set of date-level regime features (``MARKET_FEATURES``), label = 1{m > 0}
of the primary benchmark (KOSPI / SPY).

Models (walk-forward, refit every month on dates whose label ended at least one session before t):
- ``base``: the expanding base rate mean(1{m > 0}).
- ``logit``: L2 logistic regression (C = 0.1) on the expanding-standardised features (NaN -> 0 = the mean).

Honest decision rule (pre-declared): the model is used for serving only when its log-loss improvement over the
base rate has a 95% block-bootstrap lower bound > 0 on the pre-holdout dates AND the improvement is > 0 on the
untouched holdout. Otherwise the served market component is the base rate / the RSMT-Q prior-shrunk drift.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from quantcio.modeling.metrics import block_bootstrap_means, summarize

MARKET_FEATURES = [
    "r_mkt_ret_1m",
    "r_mkt_ret_3m",
    "r_mkt_dd",
    "r_mkt_vol20",
    "r_mkt_vol_ratio",
    "r_breadth",
    "r_vix_z",
    "r_credit_z",
    "r_curve_z",
    "r_score_region",
]
MIN_TRAIN_MONTHS = 36
LOGIT_C = 0.1
EPS = 1e-6


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats

    pos = y.astype(bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(p)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _auc_ci(p: np.ndarray, y: np.ndarray, block: int = 3, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    n = len(p)
    if n < 10:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = min(block, n)
    nb = int(math.ceil(n / b))
    draws = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = ((starts[:, None] + np.arange(b)[None, :]) % n).ravel()[:n]
        a = _auc(p[idx], y[idx])
        if math.isfinite(a):
            draws.append(a)
    if not draws:
        return float("nan"), float("nan")
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def walk_forward(dates_frame: pd.DataFrame, calendar: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    """Monthly walk-forward predictions: date, up, m, p_base, p_logit (rows from MIN_TRAIN_MONTHS on)."""
    from sklearn.linear_model import LogisticRegression

    d = dates_frame[dates_frame["is_me"] & dates_frame["m_primary"].notna()].sort_index()
    feats = [c for c in MARKET_FEATURES if c in d]
    cal = calendar.to_numpy(dtype="datetime64[ns]")
    rows = []
    for i, t in enumerate(d.index):
        if i < MIN_TRAIN_MONTHS:
            continue
        cut_pos = int(np.searchsorted(cal, np.datetime64(t, "ns"), "right")) - 1 - (horizon + 1)
        if cut_pos < 0:
            continue
        cut = pd.Timestamp(cal[cut_pos])
        tr = d[d.index <= cut]
        if len(tr) < MIN_TRAIN_MONTHS // 2:
            continue
        y = (tr["m_primary"] > 0).astype(int).to_numpy()
        base = float(y.mean())
        X = tr[feats].to_numpy(dtype=float)
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd[~(sd > 0)] = 1.0
        mu = np.nan_to_num(mu)
        x_std = np.nan_to_num((X - mu) / sd)
        xt = np.nan_to_num((d.loc[[t], feats].to_numpy(dtype=float) - mu) / sd)
        if 0 < y.sum() < len(y):
            m = LogisticRegression(C=LOGIT_C, max_iter=1000).fit(x_std, y)
            p = float(m.predict_proba(xt)[0, 1])
        else:
            p = base
        rows.append(
            {
                "date": t,
                "m": float(d.at[t, "m_primary"]),
                "up": int(d.at[t, "m_primary"] > 0),
                "p_base": base,
                "p_logit": p,
            }
        )
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["m", "up", "p_base", "p_logit"])


def _period_metrics(w: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    if w.empty:
        return {"n": 0}
    y = w["up"].to_numpy(dtype=float)
    pb, pm = w["p_base"].to_numpy(dtype=float), w["p_logit"].to_numpy(dtype=float)
    ll_gain = _logloss(pb, y) - _logloss(pm, y)
    br_gain = (pb - y) ** 2 - (pm - y) ** 2
    lo, hi = _auc_ci(pm, y, seed=seed)
    ll = summarize(ll_gain, 0.0, seed)
    return {
        "n": int(len(w)),
        "up_rate": float(y.mean()),
        "auc_logit": _auc(pm, y),
        "auc_logit_lo": lo,
        "auc_logit_hi": hi,
        "logloss_base": float(_logloss(pb, y).mean()),
        "logloss_logit": float(_logloss(pm, y).mean()),
        "logloss_gain": ll,
        "brier_gain": summarize(br_gain, 0.0, seed + 1),
        "first": str(w.index.min().date()),
        "last": str(w.index.max().date()),
    }


def market_direction(
    dates_frame: pd.DataFrame, calendar: pd.DatetimeIndex, horizon: int, holdout_start: pd.Timestamp
) -> dict[str, Any]:
    """Walk-forward evaluation (pre-holdout / holdout) and the serving decision (see module docstring)."""
    w = walk_forward(dates_frame, calendar, horizon)
    pre = w[w.index < holdout_start]
    hold = w[w.index >= holdout_start]
    pre_m = _period_metrics(pre, 0)
    hold_m = _period_metrics(hold, 7)
    lo = (pre_m.get("logloss_gain") or {}).get("lo")
    hold_gain = (hold_m.get("logloss_gain") or {}).get("mean")
    use = bool(lo is not None and lo > 0 and hold_gain is not None and hold_gain > 0)
    d = dates_frame[dates_frame["is_me"] & dates_frame["m_primary"].notna()]
    base_all = float((d["m_primary"] > 0).mean()) if len(d) else float("nan")
    latest = None
    if len(dates_frame):
        from sklearn.linear_model import LogisticRegression

        feats = [c for c in MARKET_FEATURES if c in dates_frame]
        X = d[feats].to_numpy(dtype=float)
        y = (d["m_primary"] > 0).astype(int).to_numpy()
        mu = np.nan_to_num(np.nanmean(X, axis=0))
        sd = np.nanstd(X, axis=0)
        sd[~(sd > 0)] = 1.0
        if 0 < y.sum() < len(y):
            m = LogisticRegression(C=LOGIT_C, max_iter=1000).fit(np.nan_to_num((X - mu) / sd), y)
            xl = np.nan_to_num((dates_frame[feats].iloc[[-1]].to_numpy(dtype=float) - mu) / sd)
            latest = {"date": str(dates_frame.index[-1].date()), "p_logit": float(m.predict_proba(xl)[0, 1])}
    return {
        "features": [c for c in MARKET_FEATURES if c in dates_frame],
        "rule": "use logit only if pre-holdout log-loss gain 95% lower bound > 0 AND holdout gain > 0",
        "pre_holdout": pre_m,
        "holdout": hold_m,
        "use_model": use,
        "base_rate_all": base_all,
        "latest": latest,
        "served_p_up": (latest or {}).get("p_logit") if use else base_all,
        "predictions": w.reset_index().assign(date=lambda x: x["date"].astype(str)).to_dict("records")
        if len(w)
        else [],
    }


__all__ = ["MARKET_FEATURES", "market_direction", "walk_forward", "block_bootstrap_means"]
