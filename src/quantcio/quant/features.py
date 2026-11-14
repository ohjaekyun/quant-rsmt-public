"""Shared, look-ahead-free per-ticker features (used by the RSMT-Q panel, indicators and factors).

Every column at date t is computed from prices/benchmark with index <= t only.
Rolling windows use `min_periods` = window, so early rows are NaN rather than biased.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EWMA_LAMBDA = 0.94


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def wilder_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss > 0, 100.0)  # only gains -> 100
    rsi = rsi.where((avg_loss > 0) | (avg_gain > 0), 50.0)  # flat (0/0) -> neutral 50
    return rsi.where(avg_gain.notna())


def ewma_vol(close: pd.Series, lam: float = EWMA_LAMBDA, min_periods: int = 60) -> pd.Series:
    """Annualised RiskMetrics EWMA volatility of daily log returns (value at t uses returns <= t)."""
    r2 = log_returns(close) ** 2
    var = r2.ewm(alpha=1.0 - lam, adjust=False, min_periods=min_periods).mean()
    return np.sqrt(var * TRADING_DAYS)


def rolling_beta(ret: pd.Series, bench_ret: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    cov = ret.rolling(window, min_periods=window).cov(bench_ret)
    var = bench_ret.rolling(window, min_periods=window).var()
    return cov / var.replace(0.0, np.nan)


def align_benchmark(prices: pd.DataFrame, benchmark: pd.DataFrame) -> pd.Series:
    """Benchmark close on the stock's calendar (forward-fill at most 3 missing sessions, never back-fill)."""
    if benchmark is None or benchmark.empty:
        return pd.Series(np.nan, index=prices.index)
    return benchmark["close"].reindex(prices.index.union(benchmark.index)).ffill(limit=3).reindex(prices.index)


def regime_features(prices: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Per-date feature frame for one ticker. Columns = FEATURE_LABELS keys except xs_mom_z (panel-level)."""
    close = prices["close"]
    lr = log_returns(close)
    bclose = align_benchmark(prices, benchmark)
    blr = np.log(bclose).diff()

    vol20 = lr.rolling(20, min_periods=20).std()
    vol60 = lr.rolling(60, min_periods=60).std()
    vol120 = lr.rolling(120, min_periods=120).std()
    ma200 = close.rolling(200, min_periods=200).mean()
    roll_max = close.rolling(250, min_periods=250).max()
    lvol = np.log1p(prices["volume"].clip(lower=0.0))

    out = pd.DataFrame(index=prices.index)
    out["vol_ratio_20_120"] = vol20 / vol120
    # own-history percentile of current 60d vol within the trailing 3y (past values only, incl. today)
    out["vol_pct_3y"] = vol60.rolling(756, min_periods=252).rank(pct=True)
    out["ret_1m"] = close / close.shift(21) - 1.0
    out["dist_ma200"] = close / ma200 - 1.0
    out["rsi14"] = wilder_rsi(close, 14)
    out["dd_250"] = close / roll_max - 1.0
    out["volume_z20"] = (lvol - lvol.rolling(20, min_periods=20).mean()) / lvol.rolling(20, min_periods=20).std()
    out["mkt_ret_60"] = bclose / bclose.shift(60) - 1.0
    out["mkt_vol_60"] = blr.rolling(60, min_periods=60).std() * np.sqrt(TRADING_DAYS)
    out["mom_12_1"] = close.shift(21) / close.shift(252) - 1.0
    out["ewma_vol"] = ewma_vol(close)
    out["beta_252"] = rolling_beta(lr, blr)
    return out.replace([np.inf, -np.inf], np.nan)


def forward_log_return(close: pd.Series, h: int) -> pd.Series:
    """Label at t: log(close[t+h] / close[t]). NaN for the last h rows (unknown future)."""
    return np.log(close.shift(-h) / close)


def shrink_beta(beta: pd.Series | float, k: float) -> pd.Series | float:
    """Blume-style shrink toward 1: (1-k)*beta + k."""
    return (1.0 - k) * beta + k
