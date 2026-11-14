"""KR cash dividends for total-return labels (docs/MODELING.md 45 / 52.1).

The KR price chain (marcap base prices) and the naver prices are PRICE returns: a cash dividend shows up as a drop
on the ex-dividend session. The stock-selection labels add the dividend back: for a label window (t, t_end] every
ex-dividend session inside it contributes log(1 + D / C_prev), D the cash dividend per share and C_prev the close of
the session before the ex-date, both from the same Yahoo series (``.KS`` / ``.KQ``, not dividend-adjusted closes;
split-consistent within Yahoo). Names Yahoo does not carry (mostly delisted) keep price-return labels - counted by
the caller. Labels are realised future returns, so using dividends announced after t is correct here (they are
not features).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from quantcio.config import Settings
from quantcio.data.cache import DiskCache

log = logging.getLogger(__name__)

CACHE_NS = "kr_dividends"
CACHE_TTL_HOURS = 24.0 * 7
MAX_YIELD = 0.5  # a single dividend above half the price is a data error (clipped, logged)


def yahoo_symbol(code: str, market: str | None) -> list[str]:
    """Yahoo symbols to try for a KRX code (the listing market first)."""
    first, second = (".KQ", ".KS") if str(market or "").upper() == "KOSDAQ" else (".KS", ".KQ")
    return [f"{code}{first}", f"{code}{second}"]


def _yahoo_history(symbol: str) -> pd.DataFrame | None:
    import yfinance as yf

    h = yf.Ticker(symbol).history(period="max", auto_adjust=False, actions=True)
    if h is None or h.empty or "Dividends" not in h:
        return None
    idx = pd.DatetimeIndex(h.index)
    idx = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    return pd.DataFrame(
        {"close": h["Close"].to_numpy(dtype=float), "div": h["Dividends"].to_numpy(dtype=float)}, index=idx
    )


def dividend_yields(hist: pd.DataFrame) -> pd.Series:
    """Ex-date -> D / previous close (0 < yield <= MAX_YIELD) of one Yahoo history (close, div)."""
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    h = hist.sort_index()
    prev = h["close"].shift(1)
    ev = h[(h["div"] > 0) & (prev > 0)]
    if ev.empty:
        return pd.Series(dtype=float)
    y = ev["div"] / prev.loc[ev.index]
    bad = y > MAX_YIELD
    if bad.any():
        log.warning("KR dividends: %d yields above %.0f%% clipped", int(bad.sum()), MAX_YIELD * 100)
    return y.clip(upper=MAX_YIELD).astype(float)


def load_kr_dividends(
    codes_markets: dict[str, str | None],
    settings: Settings,
    cache: DiskCache,
    *,
    fetch: Callable[[str], pd.DataFrame | None] | None = None,
) -> tuple[dict[str, pd.Series], dict[str, int]]:
    """code -> ex-date dividend yields (cached per code); coverage counts. ``fetch`` is an injection point."""
    fetch = fetch or _yahoo_history
    out: dict[str, pd.Series] = {}
    cov: dict = {"codes": len(codes_markets), "with_yahoo": 0, "with_dividends": 0, "failed": 0}
    yahoo_codes: list[str] = []
    for code, market in codes_markets.items():
        key = DiskCache.key("kr_div", code)
        hist = cache.get(CACHE_NS, key, CACHE_TTL_HOURS)
        if hist is None:
            hist = pd.DataFrame(columns=["close", "div"])
            for sym in yahoo_symbol(code, market):
                try:
                    h = fetch(sym)
                except Exception as exc:  # noqa: BLE001 - one symbol failing leaves that code on price returns
                    log.info("KR dividends %s: %s", sym, type(exc).__name__)
                    h = None
                    cov["failed"] += 1
                if h is not None and len(h):
                    hist = h
                    break
            cache.set(CACHE_NS, key, hist)
        if len(hist):
            cov["with_yahoo"] += 1
            yahoo_codes.append(code)
        y = dividend_yields(hist)
        if len(y):
            cov["with_dividends"] += 1
            out[code] = y
    cov["yahoo_codes"] = yahoo_codes  # codes with a Yahoo history (the dataset's dividend vintage pops it)
    return out, cov


def window_dividend_log(yields: pd.Series, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Per row: sum of log(1 + yield) over ex-dates in (t0, t1]; 0 without ex-dates, NaT t1 -> 0."""
    out = np.zeros(len(t0))
    if yields is None or not len(yields):
        return out
    ex = pd.DatetimeIndex(yields.index).to_numpy(dtype="datetime64[ns]")
    order = np.argsort(ex)
    ex = ex[order]
    cum = np.concatenate([[0.0], np.cumsum(np.log1p(yields.to_numpy(dtype=float)[order]))])
    ok = ~np.isnat(t1)
    lo = np.searchsorted(ex, t0[ok], "right")
    hi = np.searchsorted(ex, t1[ok], "right")
    out[ok] = cum[hi] - cum[lo]
    return out


__all__ = ["dividend_yields", "load_kr_dividends", "window_dividend_log", "yahoo_symbol"]
