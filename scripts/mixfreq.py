"""Mixed-frequency and missing-data rules (docs/MODELING.md 71, declared 2026-10-20). Research only.

Every series is a frame with `date` (reference period end), `value` and `available` (first day the value may be used).
`asof` puts it on any row dates with the staleness cap of its frequency; `zchange` turns it into a frequency-matched
change z-scored against its own past (no future data, minimum one year of history). Missing values are never filled
from later observations; counts that mean "none" are zero-filled by the caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_AGE = {"D": 7, "W": 14, "M": 62, "Q": 140}  # 1.5 x the release interval, in days
LAG = {"D": 1, "W": 9, "M": 61, "Q": 150}  # conservative availability when the release date is unknown
Z_WINDOW = {"D": 756, "W": 156, "M": 36, "Q": 12}  # three years of the series' own steps
Z_MIN = {"D": 252, "W": 52, "M": 12, "Q": 4}


def with_lag(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Fill a missing `available` with the conservative lag of the frequency."""
    out = df.copy()
    if "available" not in out:
        out["available"] = pd.NaT
    out["available"] = out["available"].fillna(pd.to_datetime(out["date"]) + pd.Timedelta(LAG[freq], "D"))
    return out


def zchange(df: pd.DataFrame, freq: str, steps: int, log: bool = True, yoy: bool = False) -> pd.DataFrame:
    """Change over `steps` periods (or the same period a year earlier when yoy) z-scored by the trailing window of the
    same change. Returns date, available, z (the value usable from `available`)."""
    d = with_lag(df, freq).sort_values("date").drop_duplicates("date", keep="last")
    v = d["value"].astype(float)
    if log:
        v = np.log(v.where(v > 0))
    lag = {"D": 252, "W": 52, "M": 12, "Q": 4}[freq] if yoy else steps
    ch = v - v.shift(lag)
    mu = ch.rolling(Z_WINDOW[freq], min_periods=Z_MIN[freq]).mean().shift(1)
    sd = ch.rolling(Z_WINDOW[freq], min_periods=Z_MIN[freq]).std().shift(1)
    d = d.assign(z=(ch - mu) / sd.where(sd > 0))
    return d[["date", "available", "z"]].dropna(subset=["z"])


def level_z(df: pd.DataFrame, freq: str, smooth: int = 1) -> pd.DataFrame:
    """Rolling mean of the last `smooth` values against the trailing window (e.g. daily: 4-week mean vs 52 weeks)."""
    d = with_lag(df, freq).sort_values("date").drop_duplicates("date", keep="last")
    v = d["value"].astype(float).rolling(smooth, min_periods=max(1, smooth // 2)).mean()
    win = {"D": 252, "W": 52, "M": 36, "Q": 12}[freq]
    mu = v.rolling(win, min_periods=win // 2).mean().shift(smooth)
    sd = v.rolling(win, min_periods=win // 2).std().shift(smooth)
    return d.assign(z=(v - mu) / sd.where(sd > 0))[["date", "available", "z"]].dropna(subset=["z"])


def asof(series: pd.DataFrame, col: str, dates: np.ndarray, freq: str) -> np.ndarray:
    """Value of `col` known on each date (latest `available` <= date), NaN when older than MAX_AGE[freq]."""
    s = series.dropna(subset=[col, "available"]).sort_values("available")
    s = s[~s["available"].duplicated(keep="last")]
    idx = s["available"].to_numpy(dtype="datetime64[ns]")
    vals = s[col].to_numpy(dtype=float)
    out = np.full(len(dates), np.nan)
    if not len(idx):
        return out
    k = np.searchsorted(idx, dates, "right") - 1
    ok = (k >= 0) & ((dates - idx[np.clip(k, 0, None)]) <= np.timedelta64(MAX_AGE[freq], "D"))
    out[ok] = vals[k[ok]]
    return out


def mad_outliers(x: pd.Series, by: pd.Series, k: float = 10.0) -> pd.Series:
    """NaN where a value lies more than k MADs from its group's (date's) median (71.5 (d))."""
    med = x.groupby(by).transform("median")
    mad = (x - med).abs().groupby(by).transform("median")
    return x.where((x - med).abs() <= k * mad.where(mad > 0, np.inf))


__all__ = ["LAG", "MAX_AGE", "asof", "level_z", "mad_outliers", "with_lag", "zchange"]
