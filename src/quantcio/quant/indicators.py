"""Deterministic technical indicators -> models.Technicals.

Conventions (see models.py docstring):
- Input frames follow the PRICE_COLUMNS contract (tz-naive, ascending, unique dates).
- Every value is "as of" the last stock row t and uses rows with index <= t only. Benchmark rows
  after t never enter a calculation (inner join / reindex on the stock calendar), so appending
  future benchmark data cannot change the result.
- Returns are decimals; volatilities are annualised with sqrt(252); drawdowns are <= 0.

Definitions (ARCHITECTURE.md, section quant/indicators.py):
- True range   TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|); NaN on the first row.
- ATR          Wilder's RMA of TR: seed = simple mean of the first n TRs, then
               ATR_t = ATR_{t-1} + (TR_t - ATR_{t-1}) / n  (identical to TA-Lib ATR).
- RSI14        features.wilder_rsi (shared definition; also used by the ML/RSMT features), except
               that a never-moving close (0/0) gives None instead of 100.
- MA50/MA200   simple means of the close; None when history is shorter than the window.
- realized_vol_60d  std(ddof=1) of the last 60 daily log returns x sqrt(252).
- beta_1y      cov/var of daily log returns over the last 252 sessions, on dates common to the
               stock and the benchmark (inner join, returns taken on the joined series so both
               legs span identical intervals). None when the benchmark is empty or the overlap
               has fewer than BETA_MIN_OBS returns.
- max_drawdown_1y   min(close / running max - 1) over the last 253 closes (252 returns), <= 0.
- ret_1m / ret_3m   close[t]/close[t-21] - 1, close[t]/close[t-63] - 1.
- ret_12m_ex1m close[t-21]/close[t-252] - 1 (same as features.mom_12_1).
- rel_strength_3m   ret_3m - benchmark ret_3m (benchmark via features.align_benchmark).
- high_52w / low_52w  max(high) / min(low) over the last 252 sessions.
- support / resistance  min(low, 20 sessions) / max(high, 60 sessions).

Degraded paths are never silent: each one is logged at WARNING level and, when the caller passes
a `warnings` list (normally ``md.warnings``), a Korean note is appended to it (deduplicated).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from quantcio.models import PRICE_COLUMNS, Technicals
from quantcio.quant.features import (
    TRADING_DAYS,
    align_benchmark,
    log_returns,
    wilder_rsi,
)

log = logging.getLogger(__name__)

MIN_HISTORY = 21  # below this (or atr_period + 1) compute_technicals raises ValueError
BETA_MIN_OBS = 120  # minimum common daily returns for a beta estimate
RSI_PERIOD = 14
MA_SHORT, MA_LONG = 50, 200
RET_1M, RET_3M = 21, 63
VOL_WINDOW = 60
SUPPORT_WINDOW, RESISTANCE_WINDOW = 20, 60
_REL_TOL = 1e-9  # float noise tolerance when detecting inconsistent high/low values

__all__ = [
    "MIN_HISTORY",
    "BETA_MIN_OBS",
    "valid_prices",
    "sanitize_high_low",
    "true_range",
    "wilder",
    "atr",
    "rsi",
    "realized_vol",
    "beta",
    "max_drawdown",
    "compute_technicals",
]


# ----------------------------------------------------------------------------- helpers


def _finite(x: object) -> float | None:
    """float(x) when it is a finite number, else None (None/NaN/inf/non-numeric)."""
    if x is None:
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _simple_return(series: pd.Series, lag: int, end_offset: int = 0) -> float | None:
    """series[t - end_offset] / series[t - end_offset - lag] - 1 on positional rows (None if unavailable)."""
    need = lag + end_offset + 1
    if len(series) < need:
        return None
    end = _finite(series.iloc[-1 - end_offset])
    start = _finite(series.iloc[-need])
    if end is None or start is None or start <= 0:
        return None
    return end / start - 1.0


def _note(msg: str, notes: list[str]) -> None:
    if msg not in notes:
        notes.append(msg)


def valid_prices(prices: pd.DataFrame | None) -> pd.DataFrame:
    """Rows whose close is finite and > 0 (every indicator needs the close); empty frame if none."""
    if prices is None or prices.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS, dtype="float64")
    close = pd.to_numeric(prices["close"], errors="coerce")
    ok = close.notna() & np.isfinite(close) & (close > 0)
    return prices.loc[ok]


def sanitize_high_low(prices: pd.DataFrame) -> tuple[pd.Series, pd.Series, int]:
    """High/low made consistent with the close: (high, low, n_rows_repaired).

    A missing, non-finite or non-positive high/low (e.g. trading-halt rows with O/H/L = 0, or a
    corrupt +inf tick) is replaced by the close; a high below the close / a low above the close is
    widened to the close. The number of repaired rows is returned so callers can report it (never
    silent).
    """
    close = prices["close"].astype("float64")
    raw_high = pd.to_numeric(prices["high"], errors="coerce").astype("float64")
    raw_low = pd.to_numeric(prices["low"], errors="coerce").astype("float64")
    # +inf passes a plain `> 0` test and would become high_52w / resistance = inf
    high = raw_high.where(np.isfinite(raw_high) & (raw_high > 0))
    low = raw_low.where(np.isfinite(raw_low) & (raw_low > 0))
    bad = high.isna() | low.isna() | (high < close * (1.0 - _REL_TOL)) | (low > close * (1.0 + _REL_TOL))
    high = np.fmax(high, close)  # NaN high -> close
    low = np.fmin(low, close)  # NaN low  -> close
    return high, low, int(bad.sum())


# -------------------------------------------------------------------------- indicators


def true_range(prices: pd.DataFrame) -> pd.Series:
    """TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|); NaN on the first row (no previous close)."""
    high, low, _ = sanitize_high_low(prices)
    prev_close = prices["close"].astype("float64").shift(1)
    parts = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return parts.max(axis=1, skipna=False).rename("tr")


def wilder(values: pd.Series, n: int) -> pd.Series:
    """Wilder's running moving average (RMA), seeded with the simple mean of the first n valid values.

    y_seed = mean(first n finite x); afterwards y_t = y_{t-1} + (x_t - y_{t-1}) / n. Rows before the
    seed are NaN; a NaN input after the seed carries the previous value forward. Causal: y_t uses
    x_0..x_t only.
    """
    if n < 1:
        raise ValueError(f"wilder: n must be >= 1, got {n}")
    x = values.to_numpy(dtype="float64")
    out = np.full(len(x), np.nan)
    count, total, state = 0, 0.0, math.nan
    for i, v in enumerate(x):
        finite = math.isfinite(v)
        if count < n:
            if finite:
                count += 1
                total += v
                if count == n:
                    state = total / n
                    out[i] = state
            continue
        if finite:
            state += (v - state) / n
        out[i] = state
    return pd.Series(out, index=values.index, name=values.name)


def atr(prices: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR (RMA of the true range), in price units. First value at row `period`."""
    return wilder(true_range(prices), period).rename("atr")


def rsi(close: pd.Series, n: int = RSI_PERIOD) -> pd.Series:
    """Wilder RSI; the shared features.wilder_rsi so every module uses one definition, except 0/0.

    When the smoothed gain AND the smoothed loss are both exactly 0 (the close has not moved at all
    since the series began), RSI is undefined: this returns NaN there, whereas wilder_rsi returns 100,
    which the trend factor would score as maximally overbought. All other values are identical.
    """
    out = wilder_rsi(close, n)
    moved = close.diff().abs().ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return out.where(~(moved == 0.0))


def realized_vol(close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """Rolling annualised volatility: std(ddof=1) of `window` daily log returns x sqrt(252).

    NaN until `window` returns exist (no partial windows).
    """
    return log_returns(close).rolling(window, min_periods=window).std() * math.sqrt(TRADING_DAYS)


def beta(
    close: pd.Series,
    bench_close: pd.Series | None,
    window: int = TRADING_DAYS,
    min_obs: int = BETA_MIN_OBS,
) -> float | None:
    """cov/var of daily log returns over the last `window` stock sessions, on common dates only.

    The stock's last `window + 1` closes are inner-joined with the benchmark closes; returns are taken
    on the joined series, so both legs always span identical intervals (a missing benchmark day
    produces one 2-day return on both sides instead of a fake zero). Returns None when the benchmark
    is empty, the overlap has fewer than `min_obs` returns, or the benchmark variance is zero.
    """
    if bench_close is None or len(bench_close) == 0 or len(close) < 2:
        return None
    stock = close.iloc[-(window + 1) :].astype("float64")
    joined = pd.concat({"s": stock, "b": bench_close.astype("float64")}, axis=1, join="inner")
    joined = joined[(joined > 0).all(axis=1)].dropna()
    rets = pd.DataFrame({"s": log_returns(joined["s"]), "b": log_returns(joined["b"])}).dropna()
    if len(rets) < min_obs:
        return None
    var = _finite(rets["b"].var())
    cov = _finite(rets["s"].cov(rets["b"]))
    if var is None or cov is None or var <= 0:
        return None
    return cov / var


def max_drawdown(close: pd.Series, window: int | None = TRADING_DAYS) -> float:
    """Worst peak-to-trough decline, min(close / running max - 1), over the last `window` returns.

    Uses the last `window + 1` closes (all when window is None). Always <= 0; 0.0 for empty input.
    """
    x = close.dropna().astype("float64")
    if window is not None:
        x = x.iloc[-(window + 1) :]
    if x.empty:
        return 0.0
    dd = x / x.cummax() - 1.0
    return float(min(dd.min(), 0.0))


# ------------------------------------------------------------------------------ public


def compute_technicals(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    atr_period: int = 14,
    *,
    warnings: list[str] | None = None,
) -> Technicals:
    """Technical snapshot as of the last row of `prices` (see module docstring for definitions).

    Args:
        prices: stock OHLCV (PRICE_COLUMNS contract). Rows with a missing/non-positive close are dropped
            (reported).
        benchmark: benchmark OHLCV; may be None/empty (beta_1y and rel_strength_3m are then None).
        atr_period: Wilder ATR period (settings.risk.atr_period).
        warnings: optional list (normally ``md.warnings``) that receives Korean notes for every
            degraded path; the same notes are always logged at WARNING level.

    Raises:
        ValueError: fewer than max(MIN_HISTORY, atr_period + 1) valid price rows.
    """
    notes: list[str] = []
    px = valid_prices(prices)
    n_dropped = (0 if prices is None else len(prices)) - len(px)
    if n_dropped:
        _note(f"기술지표: 종가 결측/비정상 {n_dropped}행 제외", notes)
    min_rows = max(MIN_HISTORY, atr_period + 1)
    if len(px) < min_rows:
        raise ValueError(f"compute_technicals: {len(px)} valid price rows, need >= {min_rows}")

    close = px["close"].astype("float64")
    high, low, n_fixed = sanitize_high_low(px)
    if n_fixed:
        _note(f"기술지표: 고가/저가 이상값 {n_fixed}행을 종가 기준으로 보정", notes)
    n = len(close)
    last_close = float(close.iloc[-1])

    atr_val = float(atr(px, atr_period).iloc[-1])

    ma50 = _finite(close.rolling(MA_SHORT, min_periods=MA_SHORT).mean().iloc[-1])
    ma200 = _finite(close.rolling(MA_LONG, min_periods=MA_LONG).mean().iloc[-1])

    if n < TRADING_DAYS + 1:
        _note(
            f"기술지표: 가격 이력 {n}거래일(1년 미만) - 52주 고저·최대낙폭은 가용 기간 기준, 12-1M 수익률 미산출",
            notes,
        )
    high_52w = float(high.iloc[-TRADING_DAYS:].max())
    low_52w = float(low.iloc[-TRADING_DAYS:].min())

    rv = _finite(realized_vol(close, VOL_WINDOW).iloc[-1])
    if rv is None:
        lr = log_returns(close).dropna()
        rv = float(lr.std() * math.sqrt(TRADING_DAYS))
        _note(f"기술지표: 60일 미만 이력({len(lr)}일 수익률)으로 실현변동성 산출", notes)

    ret_1m = _simple_return(close, RET_1M)
    ret_3m = _simple_return(close, RET_3M)
    ret_12m_ex1m = _simple_return(close, TRADING_DAYS - RET_1M, end_offset=RET_1M)

    bench_empty = benchmark is None or benchmark.empty
    beta_1y: float | None = None
    rel_strength_3m: float | None = None
    if bench_empty:
        _note("기술지표: 벤치마크 데이터 없음 - 베타·3개월 상대강도 미산출", notes)
    else:
        bench_close = pd.to_numeric(benchmark["close"], errors="coerce")
        beta_1y = _finite(beta(close, bench_close))
        if beta_1y is None:
            _note(
                f"기술지표: 벤치마크 공통 거래일 부족(<{BETA_MIN_OBS}일) - 베타 미산출",
                notes,
            )
        bench_on_stock = align_benchmark(px, benchmark)
        bench_ret_3m = _simple_return(bench_on_stock, RET_3M)
        if ret_3m is not None and bench_ret_3m is not None:
            rel_strength_3m = ret_3m - bench_ret_3m
        elif ret_3m is not None:
            _note("기술지표: 벤치마크 최근/3개월 전 종가 없음 - 상대강도 미산출", notes)

    rsi14 = _finite(rsi(close, RSI_PERIOD).iloc[-1])
    if rsi14 is None and n > RSI_PERIOD:
        _note("기술지표: 가격 변동 없음 - RSI 산출 불가", notes)

    tech = Technicals(
        last_close=last_close,
        atr=atr_val,
        atr_pct=atr_val / last_close,
        rsi14=rsi14,
        ma50=ma50,
        ma200=ma200,
        high_52w=high_52w,
        low_52w=low_52w,
        realized_vol_60d=rv,
        beta_1y=beta_1y,
        max_drawdown_1y=max_drawdown(close, TRADING_DAYS),
        ret_1m=ret_1m,
        ret_3m=ret_3m,
        ret_12m_ex1m=ret_12m_ex1m,
        rel_strength_3m=rel_strength_3m,
        support=float(low.iloc[-SUPPORT_WINDOW:].min()),
        resistance=float(high.iloc[-RESISTANCE_WINDOW:].max()),
    )

    for msg in notes:
        log.warning("%s", msg)
        if warnings is not None and msg not in warnings:
            warnings.append(msg)
    return tech
