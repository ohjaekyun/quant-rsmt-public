"""US market provider: prices, fundamentals, valuation history, news, SEC EDGAR filings, FX, universe.

Ported from the live-verified reference modules in ``docs/sources/reference/`` (us_market.py,
sec_edgar.py, edgar_text.py, xbrl_facts.py, fool_transcripts.py); see ARCHITECTURE.md
"Verified data sources & gotchas". Pure parsing helpers are module-level functions so they can be
tested offline on saved fixtures. ``USProvider`` adds the fetching, caching and fallback chains.

Source chains
-------------
- **prices / benchmark**: yfinance ``history(auto_adjust=True)``, then the raw Yahoo v8 chart
  (short UA, OHLC adjusted by the adjclose/close ratio), then the Nasdaq API. Nasdaq is
  split-adjusted but NOT dividend-adjusted. A Nasdaq frame carries ``attrs["adjusted"] = False``
  (``attrs`` survive ``clean_prices`` and the disk cache) and records a Korean degradation note.
  The half-formed bar of a running session is always dropped. Sessions a stock frame lacks relative
  to the SPY calendar (Yahoo ``close=None`` holes; live 2026-09-24: 43 of the 60 universe names lacked
  2026-09-22) are repaired bar by bar: the raw Yahoo v8 chart, then Nasdaq, each bar scaled onto the
  frame's price basis by the neighbouring-session ratio (ex-dividend / split dates from the chart
  decide which side applies) and validated (``scale_missing_bars``). Repaired dates are listed in
  ``attrs["repaired_sessions"]`` and noted; sessions no source has (halts) stay in
  ``attrs["missing_sessions"]`` and are noted. The series itself is never replaced by another source.
- **fundamentals**: yfinance ``.info`` with unit fixes. ``dividendYield`` is in PERCENT, and
  ``recommendationMean`` uses 1 = Strong Buy. SEC XBRL companyfacts cross-checks the YoY growth
  numbers and fills fields Yahoo lacks. When ``.info`` fails entirely, an XBRL-only snapshot is
  served and flagged. ``next_earnings_date`` (ET date) comes from ``.info``
  (``earningsTimestampStart`` / ``isEarningsDateEstimate``) cross-checked against the scraped Yahoo
  earnings-calendar page; disagreement -> earliest date, flagged as an estimate. ``Ticker.calendar``
  is NOT used: it converts epochs in the machine's timezone (off by one day when run in Seoul).
- **valuation_history**: trailing P/E = split-adjusted raw close / diluted-EPS TTM from XBRL. Each
  XBRL fact is split-adjusted by its own ``filed`` date, and the TTM is joined point-in-time on
  ``first_filed`` + 1 day, so there is no look-ahead. Only ``per`` is provided (no ``pbr``).
- **news**: yfinance ``get_news(count=40)`` on a fresh Ticker, plus Google News RSS. Yahoo RSS is
  used when yfinance fails OR returns nothing: Yahoo's news endpoint can answer HTTP 502 while
  yfinance silently returns ``[]``.
- **filings**: SEC EDGAR, with the User-Agent from ``settings.data.sec_user_agent`` (missing ->
  RuntimeError), throttled to 8 req/s globally. Also the latest Motley Fool earnings-call
  transcript.
- **fx**: the last *completed* Yahoo ``KRW=X`` bar, aligned to the last completed US session. The
  fallback is Naver (Hana Bank first-notice rate), a different definition, so its use is flagged.
  This is THE USD/KRW definition of the project: ``KRProvider.fx_rate`` calls the same functions
  (``cached_usdkrw`` / ``usdkrw_fx``) and shares the cache entry, so both markets convert with the
  identical rate and KRW->USD is exactly ``1 / (USD->KRW)``.

Degradation notes
-----------------
A fallback that changes data quality is never silent. It raises, sets an explicit flag
(``DataFrame.attrs``), and/or records a Korean note, retrievable with
``USProvider.data_warnings(ticker)`` (``MarketProvider`` API; ``drain_notes`` is the clearing alias)
and also logged at WARNING. The shared loader merges these notes into ``MarketData.warnings``.
Price / benchmark frames carry the same notes in ``attrs["warnings"]`` (always a list).
"""

from __future__ import annotations

import copy
import html as _html
import json
import logging
import math
import random
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, TypeVar

import feedparser
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from lxml import html as lxml_html

from quantcio.config import Settings
from quantcio.data.base import MarketProvider
from quantcio.data.cache import DiskCache
from quantcio.data.http import BROWSER_UA, PoliteSession
from quantcio.i18n import lang_of
from quantcio.i18n import tr as _tr
from quantcio.models import (
    PRICE_COLUMNS,
    FilingItem,
    Fundamentals,
    Instrument,
    Market,
    NewsItem,
    text_id,
)

log = logging.getLogger(__name__)

T = TypeVar("T")

_CACHE_VERSION = 1
ET = "America/New_York"
BENCHMARK = "SPY"
BENCHMARK_NAME = "S&P 500 (SPY)"
# Yahoo answers 429 to plain requests with a full Chrome UA (TLS-fingerprint mismatch) but 200 to this.
UA_YAHOO_SHORT = "Mozilla/5.0 AppleWebKit/537.36"
SEC_MAX_RPS = 8.0  # SEC fair-access limit is 10 req/s across all sec.gov hosts
_SEC_TEXT_TTL_HOURS = 24 * 30  # archived EDGAR documents never change
_DAY_TTL_HOURS = 24.0
_GLOBAL_NOTES = "*"
_TRANSCRIPT_ER_SLACK_DAYS = 7  # 8-K 2.02 deadline is 4 business days after the release
_FX_TZ = "Europe/London"  # Yahoo's exchange timezone for KRW=X daily bars


def _configure_yfinance() -> None:
    """yfinance 1.x hides HTTP errors as EMPTY frames by default; make them raise."""
    try:
        yf.config.debug.hide_exceptions = False
        yf.config.network.retries = 2
    except AttributeError:  # pragma: no cover - older yfinance
        log.warning(
            "yfinance %s has no config API: errors may surface as empty frames",
            yf.__version__,
        )


_configure_yfinance()


# =============================================================================== small utils


def _num(value: Any) -> float | None:
    """Finite float or None (Yahoo mixes None, 'Infinity' and strings)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _first(*values: float | None) -> float | None:
    return next((v for v in values if v is not None), None)


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _to_utc_ts(value: Any) -> pd.Timestamp | None:
    """Epoch seconds / ISO string / Timestamp / datetime -> tz-aware UTC Timestamp (None if unparseable)."""
    if value is None:
        return None
    try:
        if isinstance(value, int | float | np.integer | np.floating) and not isinstance(value, bool):
            ts = pd.Timestamp(int(value), unit="s", tz="UTC")
        else:
            ts = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(ts):
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _as_utc_datetime(value: datetime) -> datetime:
    """Naive datetimes are interpreted as UTC (models.py: text datetimes are tz-aware UTC)."""
    ts = _to_utc_ts(value)
    if ts is None:
        raise ValueError(f"invalid datetime: {value!r}")
    return ts.to_pydatetime()


def today_et(now: pd.Timestamp | None = None) -> date:
    return (now if now is not None else _now_utc()).tz_convert(ET).date()


def last_completed_us_session(now: pd.Timestamp | None = None, buffer_min: int = 20) -> date:
    """Latest weekday whose regular NYSE session (16:00 ET + buffer) has ended.

    Exchange holidays are not modelled. On a holiday this returns a date with no bar, which is harmless:
    it is used as a cache-key tag and as an "at or before" bound.
    """
    et = (now if now is not None else _now_utc()).tz_convert(ET)
    day = et.tz_localize(None).normalize()
    if et.weekday() < 5 and et.hour * 60 + et.minute >= 16 * 60 + buffer_min:
        return day.date()
    day -= pd.Timedelta(1, "D")
    while day.weekday() >= 5:
        day -= pd.Timedelta(1, "D")
    return day.date()


_SHARE_CLASS = re.compile(r"^([A-Z]{1,5})[./]([A-Z])$")


def normalise_us_ticker(ticker: str) -> str:
    """Upper-case, and write share classes with '-' as Yahoo and SEC do ('BRK.B' -> 'BRK-B')."""
    t = (ticker or "").strip().upper()
    m = _SHARE_CLASS.match(t)
    return f"{m.group(1)}-{m.group(2)}" if m else t


class _Throttle:
    """Process-wide minimum spacing between calls (thread-safe slot reservation)."""

    def __init__(self, min_interval: float):
        self.min_interval = float(min_interval)
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self.min_interval
        if wait > 0:
            time.sleep(wait)


_KEY_LOCKS: dict[str, threading.Lock] = {}
_KEY_LOCKS_GUARD = threading.Lock()


def _key_lock(key: str) -> threading.Lock:
    with _KEY_LOCKS_GUARD:
        return _KEY_LOCKS.setdefault(key, threading.Lock())


def cache_get_or_fetch[T](
    cache: DiskCache,
    namespace: str,
    key: str,
    ttl_hours: float | None,
    fetch: Callable[[], T],
) -> T:
    """``DiskCache.get_or_fetch`` with a process-wide per-key lock and a non-fatal write.

    The lock means parallel loaders (panel threads, sector peers, SPY, both market providers) fetch a
    key once. A failed write is logged, and the fetched value is still returned.
    """
    hit = cache.get(namespace, key, ttl_hours)
    if hit is not None:
        return hit
    with _key_lock(f"{cache.root}|{namespace}|{key}"):
        hit = cache.get(namespace, key, ttl_hours)
        if hit is not None:
            return hit
        value = fetch()
        if value is not None:
            try:
                cache.set(namespace, key, value)
            except OSError as exc:
                log.warning("cache write failed (%s/%s): %s", namespace, key, exc)
        return value


_SEC_THROTTLE = _Throttle(1.0 / SEC_MAX_RPS)  # shared by data.sec.gov + www.sec.gov
_YAHOO_THROTTLE = _Throttle(0.15)  # smooth bursts from the parallel panel loader
_FOOL_THROTTLE = _Throttle(1.0)  # public-site scraping: >= 1 s spacing
_NASDAQ_THROTTLE = _Throttle(0.2)  # api.nasdaq.com: keyless, be polite to it


def _is_permanent_error(exc: BaseException) -> bool:
    if isinstance(exc, LookupError):  # unknown symbol / missing key: retrying cannot help
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        if exc.response.status_code in (400, 401, 403, 404):
            return True
    msg = str(exc)
    return "404" in msg or "Not Found" in msg or "delisted" in msg


def _is_rate_limit(exc: BaseException) -> bool:
    rl = getattr(getattr(yf, "exceptions", None), "YFRateLimitError", None)
    msg = str(exc)
    return (rl is not None and isinstance(exc, rl)) or "429" in msg or "Too Many" in msg


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, pd.DataFrame | pd.Series):
        return value.empty
    if isinstance(value, list | dict | tuple):
        return len(value) == 0
    return False


def with_backoff[T](
    fn: Callable[[], T],
    *,
    tries: int = 3,
    base: float = 1.5,
    allow_empty: bool = False,
    what: str = "",
) -> T:
    """Call ``fn`` and retry transient failures with exponential backoff and jitter (tripled on 429).

    404 / not-found / delisted errors are raised at once. An empty result counts as a failure unless
    ``allow_empty``: yfinance turns some upstream errors into empty frames.
    """
    last: BaseException | None = None
    for i in range(tries):
        try:
            out = fn()
            if not allow_empty and _is_empty(out):
                raise ValueError(f"{what or 'call'}: empty result")
            return out
        except Exception as exc:  # noqa: BLE001 - classified below
            if _is_permanent_error(exc):
                raise
            last = exc
            if i < tries - 1:
                time.sleep(base * (2**i) * (3 if _is_rate_limit(exc) else 1) + random.uniform(0, 0.5))
    assert last is not None
    raise last


# =============================================================================== prices

_DEFAULT_CLOSE = {
    "America/New_York": (16, 0),
    "America/Chicago": (15, 15),
    "Europe/London": (16, 30),
    "Asia/Seoul": (15, 30),
    "Asia/Tokyo": (15, 30),
}


def drop_unfinished_bar(
    df: pd.DataFrame,
    tz: str,
    regular_end: Any = None,
    now: Any = None,
    buffer_min: int = 20,
) -> pd.DataFrame:
    """Drop the last row if it is *today* (exchange-local) and the regular session has not ended yet.

    ``df`` has a tz-naive, exchange-local, normalised date index. ``regular_end`` is the provider's
    current regular-session end (epoch seconds or Timestamp). When it is missing or refers to another
    day, a per-timezone default close is used. The bar counts as final ``buffer_min`` minutes after
    the close.
    """
    if df.empty:
        return df
    now_utc = _to_utc_ts(now) if now is not None else _now_utc()
    assert now_utc is not None
    local_now = now_utc.tz_convert(tz)
    today = local_now.tz_localize(None).normalize()
    if df.index[-1] != today:
        return df
    end = _to_utc_ts(regular_end)
    if end is None or end.tz_convert(tz).tz_localize(None).normalize() != today:
        h, m = _DEFAULT_CLOSE.get(tz, (16, 0))
        end = (today + pd.Timedelta(h * 60 + m, "m")).tz_localize(tz)
    if local_now < end + pd.Timedelta(buffer_min, "m"):
        return df.iloc[:-1]
    return df


def _finish_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["close"].notna()].copy()
    df.index.name = "date"
    return df


def frame_from_yf_history(raw: pd.DataFrame, regular_end: Any = None, now: Any = None) -> pd.DataFrame:
    """yfinance ``Ticker.history`` output -> PRICE_COLUMNS frame (tz-naive exchange dates, live bar dropped)."""
    if raw is None or raw.empty:
        raise ValueError("yfinance history: empty frame")
    tz = str(raw.index.tz) if getattr(raw.index, "tz", None) is not None else ET
    df = raw.rename(columns=lambda c: str(c).strip().lower())
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"yfinance history: missing columns {missing}")
    df = df[PRICE_COLUMNS].astype("float64")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)  # drop tz, keep exchange-local wall time
    df.index = idx.normalize()
    df = drop_unfinished_bar(_finish_frame(df), tz, regular_end, now)
    df.attrs.update(source="yfinance", adjusted=True, tz=tz)
    return df


def parse_yahoo_chart(res: dict, now: Any = None, adjust: bool = True) -> pd.DataFrame:
    """Yahoo v8 chart ``result[0]`` -> PRICE_COLUMNS frame.

    ``adjust=True`` scales OHLC by adjclose/close (split + dividend adjusted, same as
    ``auto_adjust=True``). ``adjust=False`` keeps Yahoo's raw prices, which are split-adjusted only
    (used for P/E history). Dates are computed in the exchange timezone, which is correct for FX too.
    """
    meta = res.get("meta") or {}
    tz = meta.get("exchangeTimezoneName") or ET
    stamps = res.get("timestamp") or []
    if not stamps:
        raise ValueError("Yahoo chart: no bars")
    idx = pd.to_datetime(stamps, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    df = pd.DataFrame({c: quote.get(c) for c in PRICE_COLUMNS}, index=idx, dtype="float64")
    adjusted = False
    if adjust:
        adj_list = ((res.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
        if adj_list is not None and len(adj_list) == len(df):
            ratio = pd.Series(adj_list, index=idx, dtype="float64") / df["close"]
            for c in ("open", "high", "low", "close"):
                df[c] = df[c] * ratio
            adjusted = True
        else:
            log.warning(
                "Yahoo chart %s: adjclose missing - returning unadjusted prices",
                meta.get("symbol"),
            )
    df = _finish_frame(df)
    df["volume"] = df["volume"].fillna(0.0)
    reg = ((meta.get("currentTradingPeriod") or {}).get("regular") or {}).get("end")
    df = drop_unfinished_bar(df, tz, reg, now)
    df.attrs.update(source="yahoo_chart", adjusted=adjusted, split_adjusted=True, tz=tz)
    return df


def chart_splits(res: dict) -> pd.Series:
    """Split events of a Yahoo chart result -> Series(date -> ratio), e.g. 2024-06-10 -> 10.0."""
    tz = (res.get("meta") or {}).get("exchangeTimezoneName") or ET
    events = ((res.get("events") or {}).get("splits")) or {}
    rows: dict[pd.Timestamp, float] = {}
    for ev in events.values():
        num, den = _num(ev.get("numerator")), _num(ev.get("denominator"))
        ts = _to_utc_ts(ev.get("date"))
        if num and den and ts is not None:
            rows[ts.tz_convert(tz).tz_localize(None).normalize()] = num / den
    return pd.Series(rows, dtype="float64").sort_index()


def parse_nasdaq_rows(rows: list[dict], now: Any = None) -> pd.DataFrame:
    """api.nasdaq.com ``tradesTable.rows`` -> PRICE_COLUMNS frame.

    The prices are SPLIT-adjusted but NOT dividend-adjusted, so ``attrs["adjusted"] = False``.
    Values arrive as strings such as '$337.02' and '31,658,820'; index volume is '--'.
    """
    df = pd.DataFrame(rows)
    if df.empty or "date" not in df.columns:
        raise ValueError("nasdaq: no rows")
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    for c in PRICE_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"nasdaq: missing column {c}")
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    df = df.set_index("date")[PRICE_COLUMNS].astype("float64")
    df = _finish_frame(df)
    df["volume"] = df["volume"].fillna(0.0)
    df = drop_unfinished_bar(df, ET, None, now)
    df.attrs.update(source="nasdaq", adjusted=False, split_adjusted=True, tz=ET)
    return df


def missing_sessions(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Sessions of a reference calendar (SPY) since the frame's first bar that the frame lacks.

    Yahoo sometimes returns a session with ``close=None`` for many NYSE names (seen live for
    2026-09-22: KO, XOM, GE, PG, CVX, LIN, NEE) while Nasdaq has the bar. yfinance then silently drops
    the row. A trading halt also shows up here. A missing LAST session means the latest price is stale.
    """
    if df is None or df.empty or calendar is None or len(calendar) == 0:
        return []
    ref = pd.DatetimeIndex(calendar)
    return list(ref[ref >= df.index[0]].difference(df.index))


def chart_event_dates(res: dict) -> pd.DatetimeIndex:
    """Ex-dividend and split dates (exchange-local) of a Yahoo chart result: the only days on which the
    ratio between an adjusted and a raw price series can change."""
    tz = (res.get("meta") or {}).get("exchangeTimezoneName") or ET
    events = res.get("events") or {}
    out: set[pd.Timestamp] = set()
    for kind in ("dividends", "splits"):
        for ev in (events.get(kind) or {}).values():
            ts = _to_utc_ts((ev or {}).get("date"))
            if ts is not None:
                out.add(ts.tz_convert(tz).tz_localize(None).normalize())
    return pd.DatetimeIndex(sorted(out))


#: max |log| move of a repaired close vs the neighbouring closes of the frame (a wrong / mislabelled
#: bar is rejected, the hole stays reported). Yahoo's narrow-window daily query was seen live
#: (2026-09-24) stamping the running session's partial OHLV with an earlier date.
_REPAIR_MAX_ABS_LOG_MOVE = 0.4


def scale_missing_bars(
    df: pd.DataFrame,
    missing: Iterable[pd.Timestamp],
    candidate: pd.DataFrame,
    events: pd.DatetimeIndex | None,
    *,
    lang: str = "ko",
) -> tuple[dict[pd.Timestamp, list[float]], dict[pd.Timestamp, str]]:
    """Bars of ``candidate`` for the ``missing`` sessions of ``df``, scaled onto ``df``'s price basis.

    ``df`` is the served series (e.g. yfinance, split + dividend adjusted). ``candidate`` is another
    source with the bar: the raw Yahoo chart (same basis, ratio ~1) or Nasdaq (split-adjusted only,
    ratio = Yahoo's dividend factor). The ratio ``df.close / candidate.close`` is read on the nearest
    common sessions before (p) and after (n) the hole. It can only change on an ex-dividend / split
    date, so:

    - p and n agree (within cent rounding) -> their mean;
    - they disagree -> n's ratio when no event lies in (d, n], p's when none lies in (p, d];
      ``events`` (from the Yahoo chart) decides, and without it the bar is NOT repaired;
    - one side only -> that side, again only when ``events`` shows no event in between.

    A candidate bar must have finite positive, consistent OHLC and stay within
    ``_REPAIR_MAX_ABS_LOG_MOVE`` of the frame's neighbouring closes. Volume is taken as reported.
    Returns ``(bars, rejected)``: ``{date: [open, high, low, close, volume]}`` and ``{date: reason}``.
    """
    bars: dict[pd.Timestamp, list[float]] = {}
    rejected: dict[pd.Timestamp, str] = {}
    if candidate is None or candidate.empty:
        return bars, {pd.Timestamp(d): _tr(lang, "원천 데이터 없음", "no source data") for d in missing}
    cand = candidate[PRICE_COLUMNS].astype("float64")
    cand = cand[~cand.index.duplicated(keep="last")].sort_index()
    good = cand["close"].notna() & (cand["close"] > 0)
    base_close = df["close"].astype("float64")
    common = df.index.intersection(cand.index[good.to_numpy()]).sort_values()
    ev = pd.DatetimeIndex(events) if events is not None else None

    def _event_in(lo: pd.Timestamp, hi: pd.Timestamp) -> bool | None:
        """Is there an event in (lo, hi]? None when unknown."""
        if ev is None:
            return None
        return bool(((ev > lo) & (ev <= hi)).any())

    for d in sorted(pd.Timestamp(x) for x in missing):
        if d not in cand.index or not bool(good.get(d, False)):
            rejected[d] = _tr(lang, "원천에도 봉 없음", "no bar in the source either")
            continue
        o, h, lo_, c, v = (float(cand.at[d, k]) for k in PRICE_COLUMNS)
        ohlc = np.array([o, h, lo_, c])
        if not (np.isfinite(ohlc).all() and (ohlc > 0).all()):
            rejected[d] = _tr(lang, "OHLC 결측", "OHLC missing")
            continue
        if h < max(o, c, lo_) * (1 - 1e-6) or lo_ > min(o, c, h) * (1 + 1e-6):
            rejected[d] = _tr(lang, "OHLC 불일치", "OHLC inconsistent")
            continue
        before, after = common[common < d], common[common > d]
        p = before[-1] if len(before) else None
        n = after[0] if len(after) else None
        r_p = float(base_close[p] / cand.at[p, "close"]) if p is not None else None
        r_n = float(base_close[n] / cand.at[n, "close"]) if n is not None else None
        ref_close = min(float(cand.at[x, "close"]) for x in (p, n, d) if x is not None)
        tol = max(3e-4, 0.011 / ref_close)  # cent rounding on both sources
        ratio: float | None = None
        if r_p is not None and r_n is not None:
            if abs(r_p / r_n - 1.0) <= tol:
                ratio = 0.5 * (r_p + r_n)
            elif _event_in(d, n) is False:
                ratio = r_n
            elif _event_in(p, d) is False:
                ratio = r_p
        elif r_p is not None and _event_in(p, d) is False:
            ratio = r_p
        elif r_n is not None and _event_in(d, n) is False:
            ratio = r_n
        if ratio is None:
            rejected[d] = (
                _tr(lang, "인접 거래일 없음", "no adjacent sessions")
                if r_p is None and r_n is None
                else _tr(lang, "배당/분할 조정 비율 불명확", "dividend/split adjustment ratio unclear")
            )
            continue
        if not (0.2 < ratio < 5.0):
            rejected[d] = _tr(lang, f"조정 비율 이상({ratio:.3f})", f"abnormal adjustment ratio ({ratio:.3f})")
            continue
        close = c * ratio
        nb = [
            float(base_close.iloc[i])
            for i in (
                base_close.index.searchsorted(d) - 1,
                base_close.index.searchsorted(d),
            )
            if 0 <= i < len(base_close)
        ]
        if any(abs(math.log(close / x)) > _REPAIR_MAX_ABS_LOG_MOVE for x in nb if x > 0):
            rejected[d] = _tr(lang, "인접일 대비 급변 — 검증 실패", "abrupt move vs adjacent days — validation failed")
            continue
        vol = v if math.isfinite(v) and v >= 0 else 0.0
        bars[d] = [o * ratio, h * ratio, lo_ * ratio, close, vol]
    return bars, rejected


def apply_repaired_bars(df: pd.DataFrame, bars: dict[pd.Timestamp, list[float]]) -> pd.DataFrame:
    """``df`` with the repaired bars inserted (sorted, unique index, attrs kept)."""
    if not bars:
        return df
    add = pd.DataFrame.from_dict(
        {pd.Timestamp(k): v for k, v in bars.items()},
        orient="index",
        columns=PRICE_COLUMNS,
        dtype="float64",
    )
    add = add[~add.index.isin(df.index)]
    out = pd.concat([df[PRICE_COLUMNS].astype("float64"), add]).sort_index()
    out.index = pd.DatetimeIndex(out.index)
    out.index.name = df.index.name
    out.attrs = copy.deepcopy(dict(df.attrs))
    return out


def _yf_history(symbol: str, start: date, *, auto_adjust: bool = True, actions: bool = False):
    """yfinance history with backoff. Returns (raw frame, current regular-session end)."""
    tk = yf.Ticker(symbol)

    def call() -> pd.DataFrame:
        _YAHOO_THROTTLE.wait()
        return tk.history(start=start.isoformat(), auto_adjust=auto_adjust, actions=actions)

    raw = with_backoff(call, what=f"yfinance history {symbol}")
    try:
        reg_end = tk.history_metadata["currentTradingPeriod"]["regular"]["end"]
    except Exception:  # noqa: BLE001 - metadata is optional; a default close time is used instead
        reg_end = None
    return raw, reg_end


def fetch_yahoo_chart(symbol: str, start: date) -> dict:
    """Raw Yahoo v8 chart JSON ``result[0]`` (daily bars since ``start``, dividend + split events)."""
    params = {
        "period1": int(pd.Timestamp(start).tz_localize("UTC").timestamp()),
        "period2": int(time.time()),
        "interval": "1d",
        "includeAdjustedClose": "true",
        "events": "div,splits",
    }
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"

    def call() -> dict:
        _YAHOO_THROTTLE.wait()
        r = requests.get(url, params=params, headers={"User-Agent": UA_YAHOO_SHORT}, timeout=20)
        r.raise_for_status()
        chart = r.json().get("chart") or {}
        if chart.get("error"):
            raise ValueError(f"Yahoo chart error: {chart['error']}")
        result = chart.get("result") or []
        if not result:
            raise ValueError("Yahoo chart: empty result")
        return result[0]

    return with_backoff(call, what=f"yahoo chart {symbol}")


def fetch_nasdaq_rows(symbol: str, start: date, assetclass: str) -> list[dict]:
    """api.nasdaq.com historical rows (keyless, ~10y max). assetclass: 'stocks' | 'etf' | 'index'."""
    params = {
        "assetclass": assetclass,
        "fromdate": start.isoformat(),
        "todate": today_et().isoformat(),
        "limit": 9999,
    }
    url = f"https://api.nasdaq.com/api/quote/{symbol.replace('-', '.')}/historical"

    def call() -> list[dict]:
        _NASDAQ_THROTTLE.wait()
        r = requests.get(
            url,
            params=params,
            headers={"User-Agent": BROWSER_UA, "Accept": "application/json"},
            timeout=30,
        )
        r.raise_for_status()
        j = r.json()
        data = j.get("data")
        if not data:  # e.g. {"rCode": 400, ... "Symbol not exists."} for the wrong asset class
            raise LookupError(f"nasdaq {assetclass}: {j.get('status')}")
        return ((data.get("tradesTable") or {}).get("rows")) or []

    return with_backoff(call, what=f"nasdaq {symbol}")


# =============================================================================== fundamentals

_RATING_BANDS = (
    (1.5, "강력매수"),
    (2.5, "매수"),
    (3.5, "중립"),
    (4.5, "비중축소"),
    (math.inf, "매도"),
)
_RATING_KEYS = {
    "strong_buy": "강력매수",
    "buy": "매수",
    "hold": "중립",
    "underperform": "비중축소",
    "sell": "매도",
    "strong_sell": "매도",
}


def recommendation_label(mean: Any, key: Any = None) -> str | None:
    """Yahoo ``recommendationMean`` (1 = Strong Buy ... 5 = Strong Sell) -> e.g. '2.20 매수 (1=강력매수)'.

    The scale is stated because KR ratings use the opposite direction (5 = strong buy).
    """
    m = _num(mean)
    if m is not None and 1.0 <= m <= 5.0:
        label = next(lbl for bound, lbl in _RATING_BANDS if m < bound)
        return f"{m:.2f} {label} (1=강력매수)"
    if isinstance(key, str) and key.lower() in _RATING_KEYS:
        return _RATING_KEYS[key.lower()]
    return None


def info_price(info: dict) -> float | None:
    return _first(
        _num(info.get("currentPrice")),
        _num(info.get("regularMarketPrice")),
        _num(info.get("previousClose")),
    )


def session_anchor_price(info: dict, now: Any = None, buffer_min: int = 20) -> tuple[float | None, float | None]:
    """``(anchor, quote)``: the close of the last COMPLETED regular session, and the quote Yahoo priced
    its ``.info`` ratios at (``info_price``).

    ``trailingPE``, ``forwardPE``, ``priceToBook``, ``marketCap`` and ``dividendRate / price`` use
    ``currentPrice``. During the regular session (22:30-05:00 KST, prime time for a Korean user) that is
    the LIVE intraday quote, while the price series ends at the last completed close (the live bar is
    dropped). Live 2026-09-24 10:38 ET: TSLA trailingPE = 378.175 / 1.10 while the series closed at
    380.12, so every valuation input and the valuation target mixed two timestamps and two runs on the
    same price snapshot disagreed. When the quote belongs to a session that has not completed
    (``regularMarketTime`` later than :func:`last_completed_us_session`), the anchor is
    ``regularMarketPreviousClose``; otherwise the quote is itself a completed close (pre-market,
    after the close + buffer, weekends).
    """
    quote = info_price(info)
    prev = _first(_num(info.get("regularMarketPreviousClose")), _num(info.get("previousClose")))
    t = _num(info.get("regularMarketTime"))
    if quote is None or prev is None or prev <= 0 or t is None or t <= 0:
        return quote, quote
    tz = str(info.get("exchangeTimezoneName") or ET)
    quote_day = pd.Timestamp(int(t), unit="s", tz="UTC").tz_convert(tz).date()
    if quote_day > last_completed_us_session(_to_utc_ts(now) if now is not None else None, buffer_min):
        return prev, quote
    return quote, quote


def dividend_yield_from_info(info: dict, price: float | None = None) -> float | None:
    """Dividend yield as a decimal.

    ``info['dividendYield']`` is in PERCENT (AAPL 0.32 means 0.32%). ``dividendRate / price`` is
    preferred, because ``trailingAnnualDividendYield`` lags a dividend change.
    """
    price = price if price is not None else info_price(info)
    rate = _num(info.get("dividendRate"))
    if rate is not None and price and price > 0:
        return rate / price
    pct = _num(info.get("dividendYield"))
    if pct is not None:
        return pct / 100.0
    trailing = _num(info.get("trailingAnnualDividendYield"))  # already a decimal
    if trailing is not None:
        return trailing
    if _num(info.get("trailingAnnualDividendRate")) == 0.0:
        return 0.0
    return None


def fundamentals_from_info(info: dict, as_of: date | None = None, anchor: float | None = None) -> Fundamentals:
    """Normalise yfinance ``.info`` into ``Fundamentals`` (ratios as decimals, local currency).

    Unit notes (verified live): returnOnEquity, operatingMargins, revenueGrowth and earningsGrowth
    are already decimals (ROE can exceed 1, e.g. AAPL 1.49). dividendYield is a percent. When
    trailingPE is missing (Yahoo drops it for loss-makers), P/E is derived from trailingEps, so a
    loss shows up as a negative P/E.

    ``anchor`` (see :func:`session_anchor_price`): the price-based fields (P/E, forward P/E, P/B,
    market cap, dividend yield) are re-priced from the ``.info`` quote to this completed close, so
    they match the price series the rest of the system uses.
    """
    g = lambda k: _num(info.get(k))  # noqa: E731
    quote = info_price(info)
    price = anchor if anchor is not None and anchor > 0 else quote
    k = price / quote if price and quote else 1.0  # 1.0 unless the quote is a live intraday price
    rebase = lambda v: v * k if v is not None else None  # noqa: E731
    eps = _first(g("trailingEps"), g("epsTrailingTwelveMonths"))
    forward_eps = g("forwardEps")
    if anchor is not None and price and eps and eps > 0:
        # Yahoo's own definition (trailingPE = currentPrice / trailingEps, verified live to 1e-7), re-priced
        # exactly: trailingPE·k inherits Yahoo's slightly out-of-sync PE/quote snapshots (0.01% run to run)
        per = price / eps
    else:
        per = rebase(g("trailingPE"))
        if per is None and price and eps:
            per = price / eps
    if anchor is not None and price and forward_eps and forward_eps > 0:
        forward_per = price / forward_eps
    else:
        forward_per = rebase(g("forwardPE"))
        if forward_per is None and price and forward_eps:
            forward_per = price / forward_eps
    count = g("numberOfAnalystOpinions")
    return Fundamentals(
        as_of=as_of,
        source="yfinance",
        market_cap=rebase(g("marketCap")),
        per=per,
        forward_per=forward_per,
        pbr=(  # Yahoo's priceToBook = currentPrice / bookValue, re-priced exactly like the P/E
            price / g("bookValue")
            if anchor is not None and price and (g("bookValue") or 0) > 0
            else rebase(g("priceToBook"))
        ),
        eps=eps,
        forward_eps=forward_eps,
        bps=g("bookValue"),
        roe=g("returnOnEquity"),
        operating_margin=g("operatingMargins"),
        dividend_yield=dividend_yield_from_info(info, price),
        ev_ebitda=g("enterpriseToEbitda"),
        revenue_growth_yoy=g("revenueGrowth"),
        earnings_growth_yoy=g("earningsGrowth"),
        analyst_target_price=g("targetMeanPrice"),
        analyst_target_high=g("targetHighPrice"),
        analyst_target_low=g("targetLowPrice"),
        analyst_count=int(count) if count is not None else None,
        analyst_rating=recommendation_label(info.get("recommendationMean"), info.get("recommendationKey")),
    )


# =============================================================================== earnings calendar

#: A "next" earnings date further ahead than this is a stale placeholder (quarterly / half-year filers).
EARNINGS_MAX_AHEAD_DAYS = 200


def epoch_local_date(value: Any, tz: str) -> date | None:
    """Yahoo epoch seconds -> the EXCHANGE-local calendar date (None for missing / pre-2000 values).

    Yahoo stores date-only earnings events at 20:00 UTC (16:00 EDT / 15:00 EST), so the date has to be
    taken in the exchange timezone. yfinance's ``Ticker.calendar`` uses the machine's local timezone
    (``datetime.fromtimestamp``) instead: run in Seoul it moves every after-close US date one day
    forward (live 2026-09-24: AAPL 10-29 -> 10-30, NVDA 11-17 -> 11-18). It also comes from the same
    quoteSummary data as ``.info``, so it is deliberately not used.
    """
    ts = _to_utc_ts(value)
    if ts is None or ts.year < 2000:
        return None
    return ts.tz_convert(tz).date()


def parse_earnings_calendar(df: pd.DataFrame | None, tz: str) -> dict[str, list[date]]:
    """yfinance ``get_earnings_dates`` frame (scraped Yahoo earnings-calendar page, tz-aware index)
    -> {"upcoming": dates not yet reported, "reported": dates with a reported EPS}, exchange-local,
    ascending. The page is a separate pipeline from quoteSummary, so it serves as a cross-check."""
    out: dict[str, list[date]] = {"upcoming": [], "reported": []}
    if df is None or df.empty:
        return out
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(tz)
    rep = (
        pd.to_numeric(df["Reported EPS"], errors="coerce")
        if "Reported EPS" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    done = rep.notna().to_numpy()
    out["upcoming"] = sorted({d.date() for d in idx[~done] if not pd.isna(d)})
    out["reported"] = sorted({d.date() for d in idx[done] if not pd.isna(d)})
    return out


def resolve_next_earnings(
    info: dict | None,
    calendar: dict[str, list[date]] | None,
    today: date,
    tz: str = ET,
) -> tuple[date | None, bool | None, str]:
    """Next scheduled earnings release (exchange-local date) -> (date, is_estimate, detail).

    Candidates on/after ``today`` and at most ``EARNINGS_MAX_AHEAD_DAYS`` ahead:
    - quote info: ``earningsTimestampStart`` (and ``earningsTimestamp`` when it points ahead);
    - the earnings-calendar page: its first unreported row.
    A date the calendar page already shows as REPORTED is dropped (the release is out; Yahoo rolls the
    quote fields later). Sources that disagree are resolved conservatively: the EARLIEST date is used
    and flagged as an estimate. ``is_estimate`` is False only when info says
    ``isEarningsDateEstimate: False``, its window is a single day and no source disagrees; a date
    from the calendar page alone (no confirmation flag) is an estimate. ``detail`` is for logs.
    """
    horizon = today + pd.Timedelta(EARNINGS_MAX_AHEAD_DAYS, "D").to_pytimedelta()
    reported = set((calendar or {}).get("reported") or [])

    def usable(d: date | None) -> bool:
        return d is not None and today <= d <= horizon and d not in reported

    cands: dict[str, date] = {}
    window = False
    flag: bool | None = None
    if info:
        start = epoch_local_date(info.get("earningsTimestampStart"), tz)
        end = epoch_local_date(info.get("earningsTimestampEnd"), tz)
        ts = epoch_local_date(info.get("earningsTimestamp"), tz)
        ahead = sorted({d for d in (start, ts) if usable(d)})
        if ahead:
            cands["info"] = ahead[0]
            # a start..end range, or two different "next" dates inside info, is not a confirmed date
            window = len(ahead) > 1 or (start is not None and end is not None and end != start)
        raw_flag = info.get("isEarningsDateEstimate")
        flag = raw_flag if isinstance(raw_flag, bool) else None
    upcoming = [d for d in (calendar or {}).get("upcoming") or [] if usable(d)]
    if upcoming:
        cands["calendar"] = min(upcoming)
    if not cands:
        return None, None, "no upcoming earnings date"
    nxt = min(cands.values())
    disagree = len(set(cands.values())) > 1
    is_est = disagree or window or "info" not in cands or flag is not False  # unknown flag = unconfirmed
    detail = ", ".join(f"{k}={v.isoformat()}" for k, v in sorted(cands.items()))
    detail += f", yahoo_estimate={flag}" + (" (DISAGREE)" if disagree else "")
    return nxt, is_est, detail


_EXCHANGE_MAP = {
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NAS": "NASDAQ",
    "NASDAQ": "NASDAQ",
    "NYQ": "NYSE",
    "NYS": "NYSE",
    "NYSE": "NYSE",
    "ASE": "NYSE American",
    "AMEX": "NYSE American",
    "PCX": "NYSE Arca",
    "ARCA": "NYSE Arca",
    "BTS": "Cboe BZX",
    "BATS": "Cboe BZX",
    "PNK": "OTC",
    "OQB": "OTC",
    "OQX": "OTC",
}
_SUPPORTED_QUOTE_TYPES = {"EQUITY", "ETF"}


class UnsupportedInstrumentError(ValueError):
    """The symbol exists but is not a stock/ETF (mutual fund, index, currency, ...)."""


def _check_quote_type(symbol: str, quote_type: Any, *, lang: str = "ko") -> None:
    qt = str(quote_type or "").upper()
    if qt and qt not in _SUPPORTED_QUOTE_TYPES:
        raise UnsupportedInstrumentError(
            _tr(
                lang,
                f"{symbol}: 지원하지 않는 종목 유형 {qt} (주식/ETF만 지원)",
                f"{symbol}: unsupported instrument type {qt} (stocks/ETFs only)",
            )
        )


def instrument_from_info(symbol: str, info: dict, *, lang: str = "ko") -> Instrument:
    _check_quote_type(symbol, info.get("quoteType"), lang=lang)
    name = info.get("longName") or info.get("shortName") or info.get("displayName")
    if not name and not info.get("quoteType"):
        raise ValueError(
            _tr(
                lang,
                f"{symbol}: Yahoo 종목 정보 없음 (잘못된 티커?)",
                f"{symbol}: no Yahoo instrument info (wrong ticker?)",
            )
        )
    code = str(info.get("exchange") or "").upper()
    return Instrument(
        ticker=symbol,
        market=Market.US,
        name=str(name or symbol),
        exchange=_EXCHANGE_MAP.get(code) or info.get("fullExchangeName") or code or None,
        currency=str(info.get("currency") or "USD"),
        sector=info.get("sector") or _STATIC_SECTOR_YAHOO.get(symbol),
        yf_symbol=symbol,
        benchmark=BENCHMARK,
        benchmark_name=BENCHMARK_NAME,
    )


def instrument_from_chart_meta(symbol: str, meta: dict, *, lang: str = "ko") -> Instrument:
    """Fallback when ``.info`` fails: the chart metadata has name/exchange/currency (no sector)."""
    _check_quote_type(symbol, meta.get("instrumentType"), lang=lang)
    code = str(meta.get("exchangeName") or "").upper()
    return Instrument(
        ticker=symbol,
        market=Market.US,
        name=str(meta.get("longName") or meta.get("shortName") or symbol),
        exchange=_EXCHANGE_MAP.get(code) or meta.get("fullExchangeName") or code or None,
        currency=str(meta.get("currency") or "USD"),
        sector=_STATIC_SECTOR_YAHOO.get(symbol),
        yf_symbol=symbol,
        benchmark=BENCHMARK,
        benchmark_name=BENCHMARK_NAME,
    )


# =============================================================================== EDGAR text

_BLOCK = {
    "p",
    "div",
    "br",
    "tr",
    "li",
    "ul",
    "ol",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "section",
    "article",
    "hr",
    "title",
    "center",
    "blockquote",
    "pre",
    "dt",
    "dd",
    "dl",
    "table",
    "body",
    "page",
}
_DROP_TAGS = {"script", "style", "head", "noscript", "object", "embed", "svg"}
_HIDDEN_RE = re.compile(r"display\s*:\s*none", re.I)
# unicode spaces / zero-width chars built with chr() so the source stays pure ASCII
_SPACES = "".join(map(chr, [0xA0, 0x2002, 0x2003, 0x2009, 0x200A, 0x202F, 0x205F, 0x3000]))
_WS_RE = re.compile("[ \t\r\f\v" + _SPACES + "]+")
_ZW_RE = re.compile("[" + "".join(map(chr, [0x200B, 0x200C, 0x200D, 0xFEFF, 0xAD])) + "]")
_QUOTES = str.maketrans({0x2019: "'", 0x2018: "'", 0x201C: '"', 0x201D: '"', 0x2013: "-", 0x2014: "-"})
_PAGENUM_RE = re.compile(r"(page\s*)?-?\s*\d{1,3}\s*-?", re.I)
_FOOTER_RE = re.compile(r"[^|]{2,80}\|[^|]{0,40}Form\s+10-[KQ]\s*\|\s*\d{1,3}", re.I)
_RUNHDR_RE = re.compile(
    r"(part\s+[ivx]+(\s*,\s*[ivx]+)*\s*)?(item\s+\d{1,2}[a-c]?(\s*,\s*\d{1,2}[a-c]?)*)?",
    re.I,
)


def _local(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""  # comments / processing instructions
    t = tag.lower()
    return t.split("}", 1)[1] if "}" in t else t


def _walk(el: Any, out: list) -> None:
    tag = _local(el.tag)
    if tag == "table":
        out.append("\n" + _table_to_text(el) + "\n")
    else:
        block = tag in _BLOCK
        if block:
            out.append("\n")
        if el.text and isinstance(el.tag, str):
            out.append(el.text)
        for ch in el:
            _walk(ch, out)
        if block:
            out.append("\n")
    if el.tail:
        out.append(el.tail)


def _cell_text(td: Any) -> str:
    parts: list = [td.text] if td.text else []
    for ch in td:  # <br>/<div>/<p> inside a cell -> spaces ("June 27,<br>2026")
        _walk(ch, parts)
    return _WS_RE.sub(" ", _ZW_RE.sub("", "".join(parts)).replace("\n", " ")).strip()


def _table_to_text(tbl: Any) -> str:
    """Render a table as 'a | b | c' rows; merge EDGAR's split '$', ')' and '%' cells."""
    lines = []
    for tr in tbl.iter():
        if _local(tr.tag) != "tr":
            continue
        merged: list[str] = []
        for td in tr:
            if _local(td.tag) not in ("td", "th"):
                continue
            c = _cell_text(td)
            if not c:
                continue
            if merged and (c in (")", "%", ")%", "%)") or c.startswith(")")):
                merged[-1] += c
            elif merged and merged[-1] in ("$", "(", "$(", "($"):
                merged[-1] += c
            else:
                merged.append(c)
        if merged:
            lines.append(" | ".join(merged))
    return "\n".join(lines)


def edgar_html_to_text(raw: bytes | str, drop_page_furniture: bool = True) -> str:
    """Convert an EDGAR .htm (incl. inline XBRL, Workiva/DFIN/Toppan output) to plain text.

    - drops <ix:header> hidden XBRL facts, display:none blocks, <script>/<style>/<head>
    - strips the SGML ``<DOCUMENT><TYPE>..`` wrapper of non-inline exhibits and the ``<?xml ..?>``
      declaration (lxml refuses a str with an encoding declaration)
    - tables -> one row per line, cells joined by ' | ' (with '$', ')' and '%' merged)
    - normalises unicode spaces/quotes, removes page numbers, 'X | Form 10-Q | 12' footers and
      running headers ('Item 1A', 'PART II') that repeat >= 3x
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    raw = re.sub(r"^\s*<\?xml[^>]*\?>", "", raw)
    m = re.search(r"<html", raw[:5000], re.I)
    if m and m.start() > 0:
        raw = raw[m.start() :]
    if not re.search(r"<(html|body|div|p|table)\b", raw[:20000], re.I):  # plain-text (.txt) filing
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"<[^>]+>", "", raw)).strip()
    doc = lxml_html.fromstring(raw)
    to_drop = []
    for el in doc.iter():
        if not isinstance(el.tag, str):
            to_drop.append(el)  # comments / PIs
            continue
        lt = el.tag.lower()
        if lt in _DROP_TAGS or lt == "ix:header" or (lt.endswith("}header") and "inlinexbrl" in lt):
            to_drop.append(el)
        else:
            st = el.get("style")
            if st and _HIDDEN_RE.search(st):
                to_drop.append(el)
    for el in to_drop:
        parent = el.getparent()
        if parent is None:
            continue
        if el.tail:  # keep the text that follows the removed element
            prev = el.getprevious()
            if prev is not None:
                prev.tail = (prev.tail or "") + el.tail
            else:
                parent.text = (parent.text or "") + el.tail
        parent.remove(el)

    out: list = []
    _walk(doc, out)
    text = _ZW_RE.sub("", "".join(out)).translate(_QUOTES)
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.split("\n")]
    if drop_page_furniture:
        cnt = Counter(lines)
        keep = []
        for ln in lines:
            if ln and (_PAGENUM_RE.fullmatch(ln) or _FOOTER_RE.fullmatch(ln)):
                continue
            if ln and _RUNHDR_RE.fullmatch(ln) and cnt[ln] >= 3:
                continue
            if ln.lower() in ("table of contents", "index"):
                continue
            keep.append(ln)
        lines = keep
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


_SEP = r"[\s|.:\-]*"
_SECTION_PATTERNS = {
    # (start regexes, end regexes); all anchored at line start (re.M)
    "10-K": {
        "risk_factors": (
            [rf"^item{_SEP}1a{_SEP}risk\s+factors"],
            [rf"^item{_SEP}1b\b", rf"^item{_SEP}1c\b", rf"^item{_SEP}2{_SEP}propert"],
        ),
        "mdna": (
            [rf"^item{_SEP}7{_SEP}management'?s\s+discussion"],
            [rf"^item{_SEP}7a\b", rf"^item{_SEP}8{_SEP}financial\s+statements"],
        ),
    },
    "10-Q": {
        "mdna": (
            [rf"^item{_SEP}2{_SEP}management'?s\s+discussion"],
            [rf"^item{_SEP}3{_SEP}quantitative", rf"^item{_SEP}4{_SEP}controls"],
        ),
        "risk_factors": (
            [rf"^item{_SEP}1a{_SEP}risk\s+factors"],
            [
                rf"^item{_SEP}2{_SEP}unregistered",
                rf"^item{_SEP}3{_SEP}defaults",
                rf"^item{_SEP}4{_SEP}mine",
                rf"^item{_SEP}5{_SEP}other\s+information",
                rf"^item{_SEP}6{_SEP}exhibits",
            ],
        ),
    },
}


def extract_section(text: str, form: str, section: str, max_chars: int | None = 60_000) -> str:
    """Extract ``section`` ('risk_factors' | 'mdna') from ``edgar_html_to_text`` output.

    ``form`` is '10-K' or '10-Q' (amendments map to the base form). For every heading match the span
    up to the next end heading is taken, and the LONGEST span wins. This skips table-of-contents rows
    and in-text cross references without per-filer rules. Returns '' when the section is not found.
    """
    base = "10-K" if form.upper().startswith("10-K") else "10-Q"
    starts, ends = _SECTION_PATTERNS[base][section]
    flags = re.I | re.M
    s_pos = sorted({m.start() for p in starts for m in re.finditer(p, text, flags)})
    e_pos = sorted({m.start() for p in ends for m in re.finditer(p, text, flags)})
    best = (0, 0)
    for s in s_pos:
        nxt = [e for e in e_pos if e > s + 50]
        e = nxt[0] if nxt else min(len(text), s + 150_000)  # no end heading -> cap
        if e - s > best[1] - best[0]:
            best = (s, e)
    if best == (0, 0):
        return ""
    sec = text[best[0] : best[1]].strip()
    if len(sec) < 3000:  # bank-style filers (JPM/BAC/C): section is only a sub-TOC 'Title | page'
        sec = _expand_sub_toc(text, sec, best[1], ends) or sec
    tail_lines = sec.split("\n")
    while tail_lines and (not tail_lines[-1].strip() or _RUNHDR_RE.fullmatch(tail_lines[-1].strip())):
        tail_lines.pop()  # drop trailing running header like 'Item 3, 4'
    sec = "\n".join(tail_lines)
    if max_chars and len(sec) > max_chars:
        sec = truncate_head_tail(sec, max_chars)
    return sec


def _expand_sub_toc(text: str, sec: str, after: int, ends: list) -> str:
    entries = [e.strip() for e in re.findall(r"^(.{3,120}?) \| \d{1,3}$", sec, re.M)]
    if len(entries) < 3:
        return ""

    def _find(title: str, start: int) -> int | None:
        m = re.compile(r"^" + re.escape(title) + r"\b.{0,40}$", re.I | re.M).search(text, start)
        return m.start() if m else None

    s = next((p for p in (_find(t, after) for t in entries[:3]) if p is not None), None)
    if s is None:
        return ""
    last = (
        next(
            (p for p in (_find(t, s + 1) for t in reversed(entries)) if p is not None),
            None,
        )
        or s
    )
    stops = list(ends) + [
        r"^consolidated (statements? of|balance sheets?)",
        r"^report of independent",
        r"^.{0,60}\bconsolidated statements of (income|operations)\b",
    ]
    e_m = [m.start() + 50 for p in stops for m in re.finditer(p, text[last + 50 :], re.I | re.M)]
    e = last + min(min(e_m) if e_m else 12_000, 12_000)  # last sub-section is short (e.g. FLS)
    return text[s:e].strip()


def truncate_head_tail(text: str, max_chars: int, head_frac: float = 0.7) -> str:
    """Keep the head (70%) and tail (30%), cut on line boundaries, with an explicit omission marker.

    The result is guaranteed to be at most ``max_chars`` characters, marker included.
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = "\n\n[... {n:,} chars truncated ...]\n\n"
    budget = max_chars - len(marker.format(n=len(text)))
    if budget <= 0:
        return text[:max_chars]
    head_n = int(budget * head_frac)
    tail_n = budget - head_n
    head = text[:head_n]
    cut = head.rfind("\n")
    head = head[:cut] if cut > head_n * 0.5 else head
    tail = text[-tail_n:] if tail_n > 0 else ""
    cut = tail.find("\n")
    tail = tail[cut + 1 :] if 0 <= cut < tail_n * 0.5 else tail
    return f"{head}{marker.format(n=len(text) - len(head) - len(tail))}{tail}"


def risk_factor_headlines(risk_text: str, max_items: int = 60) -> list[str]:
    """Risk-factor titles = standalone one-sentence lines (40-400 chars) ending in '.'.

    Clean on 10-Ks. On 10-Qs some body sentences slip in, so treat the list as a hint.
    """
    out = []
    for ln in risk_text.split("\n"):
        ln = ln.strip().lstrip(chr(0x2022) + chr(0xB7) + "o- ").strip()
        if (
            40 <= len(ln) <= 400
            and ln.endswith(".")
            and ". " not in ln
            and "|" not in ln
            and not re.match(
                r"(other than|this section|the following|for all of these|there have been)",
                ln,
                re.I,
            )
        ):
            out.append(ln)
    return out[:max_items]


_KEYWORDS = (
    "expect",
    "outlook",
    "guidance",
    "anticipate",
    "demand",
    "supply",
    "constrain",
    "margin",
    "headwind",
    "tailwind",
    "decline",
    "decrease",
    "increase",
    "growth",
    "tariff",
    "pricing",
    "backlog",
    "impairment",
    "restructur",
    "litigation",
    "regulat",
    "export",
    "china",
    "liquidity",
    "repurchase",
    "capital expenditure",
    "uncertain",
    "risk",
)


def select_paragraphs(
    text: str,
    budget_chars: int = 12_000,
    keywords: Iterable[str] = _KEYWORDS,
    keep_first: int = 1,
    min_len: int = 120,
) -> str:
    """Budgeted digest for LLM input: keep the first ``keep_first`` paragraphs, then the paragraphs
    with the highest keyword density (tables and boilerplate skipped), emitted in original order."""
    kws = tuple(keywords)
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    scored = []
    for i, p in enumerate(paras):
        if len(p) < min_len or p.count("|") > 3:
            continue
        if re.search(
            r"forward-looking statements|should be read in conjunction|incorporated by reference",
            p,
            re.I,
        ):
            continue
        low = p.lower()
        score = sum(low.count(k) for k in kws) / (len(p) ** 0.5)
        scored.append((i < keep_first, score, i, p))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    chosen, used = [], 0
    for _, _, i, p in scored:
        if used + len(p) > budget_chars:
            continue
        chosen.append((i, p))
        used += len(p) + 2
    return "\n\n".join(p for _, p in sorted(chosen))


def body_after_items(text: str, max_chars: int = 6000) -> str:
    """8-K main document: skip the cover page (start at the first 'Item x.xx'), stop at SIGNATURES."""
    m = re.search(r"^item\s*\d\.\d\d", text, re.I | re.M)
    body = text[m.start() :] if m else text
    body = re.split(r"^signatures?\s*$", body, flags=re.I | re.M)[0].strip()
    return truncate_head_tail(body, max_chars)


def _allocate(lengths: list[int], total: int, weights: list[float] | None = None) -> list[int]:
    """Split ``total`` characters across parts by weight. A part shorter than its share keeps its full
    length, and the unused budget flows to the others."""
    n = len(lengths)
    w = list(weights) if weights else [1.0] * n
    alloc = [0] * n
    active = {i for i in range(n) if lengths[i] > 0}
    remaining = max(0, total)
    while active and remaining > 0:
        wsum = sum(w[i] for i in active) or 1.0
        share = {i: remaining * w[i] / wsum for i in active}
        done = [i for i in active if lengths[i] - alloc[i] <= share[i]]
        if not done:
            for i in active:
                alloc[i] += int(share[i])
            break
        for i in done:
            remaining -= lengths[i] - alloc[i]
            alloc[i] = lengths[i]
            active.discard(i)
    return alloc


def compose_sections(parts: list[tuple[str, str]], max_chars: int, weights: list[float] | None = None) -> str:
    """Join labelled parts ('[label]\\nbody') within ``max_chars``, truncating each part head+tail."""
    kept = [(label, body.strip()) for label, body in parts if body and body.strip()]
    if not kept:
        return ""
    w = None
    if weights and len(weights) == len(parts):
        w = [weights[i] for i, (_, body) in enumerate(parts) if body and body.strip()]
    sep = "\n\n"
    overhead = sum(len(f"[{label}]\n") for label, _ in kept) + len(sep) * (len(kept) - 1)
    budgets = _allocate([len(b) for _, b in kept], max_chars - overhead, w)
    out = [f"[{label}]\n{truncate_head_tail(body, b)}" for (label, body), b in zip(kept, budgets) if b > 0]
    return truncate_head_tail(sep.join(out), max_chars)


# =============================================================================== SEC EDGAR

EIGHT_K_FORMS = ("8-K", "8-K/A")
PERIODIC_FORMS = ("10-Q", "10-K")

ITEM_8K_KO = {
    "1.01": "중요 계약 체결",
    "1.02": "중요 계약 해지",
    "1.03": "파산·법정관리",
    "1.04": "광산 안전",
    "1.05": "중대 사이버보안 사고",
    "2.01": "자산 인수·처분 완료",
    "2.02": "실적 발표",
    "2.03": "직접 금융채무 발생",
    "2.04": "채무 가속 사유 발생",
    "2.05": "구조조정 비용",
    "2.06": "중대 손상차손",
    "3.01": "상장폐지 통지·상장요건 미충족",
    "3.02": "미등록 지분증권 발행",
    "3.03": "주주 권리 변경",
    "4.01": "외부감사인 변경",
    "4.02": "기발표 재무제표 신뢰 불가(정정)",
    "5.01": "경영권 변동",
    "5.02": "임원·이사 변동/보상",
    "5.03": "정관·회계연도 변경",
    "5.04": "종업원 연금 거래 정지",
    "5.05": "윤리강령 변경",
    "5.07": "주주총회 표결 결과",
    "5.08": "주주 이사 후보 추천",
    "7.01": "공정공시(Reg FD)",
    "8.01": "기타 주요 사항",
    "9.01": "재무제표·첨부서류",
}
# rough event-risk weight (negative = adverse tail risk); <= -2 is prioritised in filings()
ITEM_8K_SEVERITY = {
    "1.03": -3,
    "4.02": -3,
    "3.01": -2,
    "1.05": -2,
    "2.06": -2,
    "2.04": -2,
    "4.01": -1,
    "2.05": -1,
    "5.02": -1,
    "1.02": -1,
}


def parse_8k_items(items: Any) -> list[str]:
    return [i.strip() for i in str(items or "").split(",") if i.strip()]


def item_severity(items: Iterable[str]) -> int:
    return min([ITEM_8K_SEVERITY.get(i, 0) for i in items] or [0])


def parse_company_tickers(j: dict) -> dict[str, tuple[str, str]]:
    """SEC company_tickers.json -> {'AAPL': ('0000320193', 'Apple Inc.'), 'BRK-B': (...)}."""
    out: dict[str, tuple[str, str]] = {}
    for v in j.values():
        t = str(v.get("ticker", "")).upper().strip()
        if t:
            out.setdefault(t, (str(v["cik_str"]).zfill(10), str(v.get("title", ""))))
    return out


_SUBMISSION_TEXT_COLS = (
    "accessionNumber",
    "filingDate",
    "reportDate",
    "acceptanceDateTime",
    "form",
    "items",
    "primaryDocument",
    "primaryDocDescription",
)


def parse_submissions(j: dict) -> tuple[dict, pd.DataFrame]:
    """data.sec.gov submissions JSON -> (company meta, filings newest-first).

    Adds ``accepted_utc`` (acceptanceDateTime is TRUE UTC), ``accepted_et``, ``filed_utc``
    (accepted_utc, else the filing date at 00:00 ET), ``url`` (primary document) and ``index_url``
    (``<acc>-index.htm``, the only keyless place listing exhibit TYPES such as EX-99.1).
    """
    meta = {
        k: j.get(k)
        for k in (
            "cik",
            "name",
            "tickers",
            "exchanges",
            "sic",
            "sicDescription",
            "fiscalYearEnd",
            "category",
        )
    }
    df = pd.DataFrame(((j.get("filings") or {}).get("recent")) or {})
    for c in _SUBMISSION_TEXT_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df.drop_duplicates("accessionNumber")
    for c in _SUBMISSION_TEXT_COLS:
        df[c] = df[c].fillna("").astype(str)
    acc_dt = df["acceptanceDateTime"].where(df["acceptanceDateTime"] != "")
    df["accepted_utc"] = pd.to_datetime(acc_dt, utc=True, errors="coerce", format="ISO8601")
    df["accepted_et"] = df["accepted_utc"].dt.tz_convert(ET)
    df["filingDate"] = pd.to_datetime(
        df["filingDate"].where(df["filingDate"] != ""),
        errors="coerce",
        format="%Y-%m-%d",
    )
    fallback = df["filingDate"].dt.tz_localize(ET).dt.tz_convert("UTC")
    df["filed_utc"] = df["accepted_utc"].fillna(fallback)
    ncik = int(j.get("cik") or 0)
    acc = df["accessionNumber"].str.replace("-", "", regex=False)
    base = f"https://www.sec.gov/Archives/edgar/data/{ncik}/"
    df["url"] = base + acc + "/" + df["primaryDocument"]
    df["index_url"] = base + acc + "/" + df["accessionNumber"] + "-index.htm"
    df = df.sort_values("filed_utc", ascending=False, kind="stable", na_position="last")
    return meta, df.reset_index(drop=True)


def parse_filing_index(content: bytes | str) -> pd.DataFrame:
    """``<acc>-index.htm`` -> DataFrame[seq, description, document, type, size, url].

    The '/ix?doc=' inline-XBRL viewer prefix is stripped so ``url`` points at the raw document.
    """
    doc = lxml_html.fromstring(content)
    rows = []
    for tbl in doc.xpath('//table[contains(@class,"tableFile")]'):
        for tr in tbl.xpath(".//tr")[1:]:
            tds = tr.xpath("./td")
            if len(tds) < 5:
                continue
            a = tds[2].xpath(".//a")
            href = (a[0].get("href") if a else "") or ""
            href = href.replace("/ix?doc=", "")
            doc_txt = tds[2].text_content().strip()
            rows.append(
                {
                    "seq": tds[0].text_content().strip(),
                    "description": tds[1].text_content().strip(),
                    "document": doc_txt.split()[0] if doc_txt else "",
                    "type": tds[3].text_content().strip(),
                    "size": tds[4].text_content().strip(),
                    "url": ("https://www.sec.gov" + href) if href.startswith("/") else href,
                }
            )
    return pd.DataFrame(rows, columns=["seq", "description", "document", "type", "size", "url"])


def exhibits_99(index: pd.DataFrame) -> list[dict]:
    """EX-99.x text exhibits of a filing index (EX-99.1 first)."""
    if index.empty:
        return []
    ex = index[
        index["type"].str.upper().str.startswith("EX-99")
        & index["url"].str.lower().str.endswith((".htm", ".html", ".txt"))
    ]
    return ex.sort_values("type", kind="stable").to_dict("records")


# =============================================================================== XBRL facts

XBRL_CONCEPTS = {
    # metric: ordered us-gaap concept candidates (first one wins per period)
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "RevenuesNetOfInterestExpense",
    ],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "operating_income": ["OperatingIncomeLoss"],
}
_PER_SHARE_METRICS = {"eps_diluted"}
_XBRL_PERIODIC_FORMS = {
    "10-K",
    "10-Q",
    "10-K/A",
    "10-Q/A",
    "10-KT",
    "10-QT",
    "20-F",
    "20-F/A",
    "40-F",
    "6-K",
}
# duration buckets (days): 0 = quarter, 1 = 6M YTD, 2 = 9M YTD, 3 = fiscal year
_BUCKETS = [(80, 100), (170, 200), (260, 285), (350, 380)]


def _days(n: int) -> pd.Timedelta:
    return pd.Timedelta(n, "D")


def split_factor_after(filed: pd.Series, splits: pd.Series | None) -> pd.Series:
    """Product of the split ratios effective AFTER each filing date.

    Filings retroactively restate per-share numbers for splits that happened before they were filed,
    so a value filed before a later split must be divided by that split's ratio.
    """
    factor = np.ones(len(filed))
    if splits is not None and len(splits):
        filed_v = pd.to_datetime(filed).to_numpy()
        for d, ratio in splits.items():
            r = _num(ratio)
            if r and r > 0 and r != 1.0:
                factor = np.where(filed_v < np.datetime64(pd.Timestamp(d)), factor * r, factor)
    return pd.Series(factor, index=filed.index)


def trim_companyfacts(facts: dict) -> dict:
    """Keep only the us-gaap concepts used here (the full JSON is 4-5 MB)."""
    wanted = {c for cs in XBRL_CONCEPTS.values() for c in cs}
    g = (facts.get("facts") or {}).get("us-gaap") or {}
    return {
        "cik": facts.get("cik"),
        "entityName": facts.get("entityName"),
        "facts": {"us-gaap": {c: g[c] for c in wanted if c in g}},
    }


def _pick_unit(units: pd.Series) -> str:
    """The metric's reporting unit: the one with the most facts, USD preferred on a tie.

    Foreign filers tag both their own currency and a USD convenience translation for recent periods
    (BABA: 'CNY/shares' + 'USD/shares'). Mixing them in one series would make YoY and TTM meaningless.
    """
    counts = units.value_counts()
    top = counts.max()
    tied = sorted(u for u, n in counts.items() if n == top)
    return next((u for u in tied if u.upper().startswith("USD")), tied[0])


def _xbrl_facts_frame(
    facts: dict,
    concepts: list[str],
    vintage: str = "latest",
    splits: pd.Series | None = None,
    per_share: bool = False,
) -> pd.DataFrame:
    rows = []
    g = (facts.get("facts") or {}).get("us-gaap") or {}
    for prio, c in enumerate(concepts):
        for unit, arr in (g.get(c) or {}).get("units", {}).items():
            for f in arr:
                if "start" not in f:  # instant facts (balance sheet) not handled here
                    continue
                rows.append({**f, "concept": c, "unit": unit, "prio": prio})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df["form"].isin(_XBRL_PERIODIC_FORMS)].copy()  # drop DEF 14A / 8-K / S-4 copies of facts
    if df.empty:
        return df
    if df["unit"].nunique() > 1:
        unit = _pick_unit(df["unit"])
        log.info(
            "XBRL %s: units %s -> using %s only",
            concepts[0],
            sorted(df["unit"].unique()),
            unit,
        )
        df = df[df["unit"] == unit].copy()
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df["days"] = (df["end"] - df["start"]).dt.days
    # first date this period's number was public (point-in-time availability, no look-ahead)
    df["first_filed"] = df.groupby(["start", "end"])["filed"].transform("min")
    if vintage == "first":  # as originally reported (ignores later restatements)
        df = df.sort_values(["start", "end", "filed", "prio"])
    else:  # latest restated value, best-priority concept
        df = df.sort_values(["start", "end", "prio", "filed"], ascending=[True, True, True, False])
    df = df.drop_duplicates(["start", "end"], keep="first").copy()
    if per_share and splits is not None and len(splits):
        # adjust each fact to today's share basis BEFORE deriving quarters: a derived Q4 (FY - 9M)
        # must not mix a post-split FY value with a pre-split 9M value
        df["val"] = df["val"].astype(float) / split_factor_after(df["filed"], splits).to_numpy()
    df["bucket"] = df["days"].map(_bucket)
    return df[df["bucket"].notna()]


def _bucket(days: int) -> int | None:
    for i, (lo, hi) in enumerate(_BUCKETS):
        if lo <= days <= hi:
            return i
    return None


def _derived_row(a: Any, start: pd.Timestamp, val: float, first_filed: pd.Timestamp) -> dict:
    return {
        "start": start,
        "end": a.end,
        "val": val,
        "concept": a.concept,
        "form": a.form,
        "accn": a.accn,
        "filed": a.filed,
        "first_filed": first_filed,
        "unit": a.unit,
        "prio": a.prio,
        "derived": True,
    }


def quarterly_series(
    facts: dict,
    metric: str = "revenue",
    derive: bool = True,
    vintage: str = "latest",
    splits: pd.Series | None = None,
) -> pd.DataFrame:
    """Quarterly XBRL series: DataFrame[start, end, value, yoy, qoq, derived, concept, unit, form, accn,
    filed, first_filed].

    - Quarters are 80..100-day durations. Missing quarters are derived from consecutive YTD facts
      (Q2 = 6M - Q1, Q3 = 9M - 6M, Q4 = FY - 9M) and flagged ``derived``.
    - ``yoy`` compares with the quarter ending 364 +/- 16 days earlier (NaN on a base <= 0). ``qoq``
      is filled only for adjacent quarters.
    - ``first_filed`` is the first public date of the period. Use it, never ``end`` or ``filed``, as the
      availability date when joining to prices.
    - Per-share metrics: pass ``splits`` (date -> ratio). Old facts that were never restated keep their
      pre-split values in the "latest" vintage (NVDA 2019-2023), so each fact is divided by the splits
      that happened after its own filing date.
    """
    df = _xbrl_facts_frame(facts, XBRL_CONCEPTS[metric], vintage, splits, metric in _PER_SHARE_METRICS)
    if df.empty:
        return df
    q = df[df.bucket == 0].copy()
    q["derived"] = False
    if derive:
        add: list = []
        for b in (1, 2, 3):
            known = pd.concat([q, pd.DataFrame(add)], ignore_index=True) if add else q
            for _, a in df[df.bucket == b].iterrows():
                if ((known.end - a.end).abs().dt.days <= 3).any():
                    continue  # quarter already reported (or derived)
                prev = df[
                    (df.bucket == b - 1) & ((df.start - a.start).abs().dt.days <= 3) & (df.end < a.end - _days(60))
                ]
                if len(prev):
                    p = prev.sort_values("end").iloc[-1]
                    add.append(
                        _derived_row(
                            a,
                            p.end + _days(1),
                            a.val - p.val,
                            max(a.first_filed, p.first_filed),
                        )
                    )
                    continue
                qs = known[(known.start >= a.start - _days(3)) & (known.end < a.end - _days(20))]
                if len(qs) == b:  # sum of the b earlier quarters of this fiscal year
                    add.append(
                        _derived_row(
                            a,
                            qs.end.max() + _days(1),
                            a.val - qs.val.sum(),
                            max([a.first_filed] + list(qs.first_filed)),
                        )
                    )
        if add:
            q = pd.concat([q, pd.DataFrame(add)], ignore_index=True)
    q = q.sort_values(["end", "prio", "derived"]).drop_duplicates("end", keep="first").reset_index(drop=True)
    q = q.rename(columns={"val": "value"})
    q["value"] = q["value"].astype(float).round(6)  # YTD subtraction float noise
    prev_q = q[["end", "value"]].rename(columns={"end": "end_prev", "value": "value_prev"})
    q["_key"] = q["end"] - _days(364)
    q = pd.merge_asof(
        q.sort_values("_key"),
        prev_q.sort_values("end_prev"),
        left_on="_key",
        right_on="end_prev",
        direction="nearest",
        tolerance=_days(16),
    )
    q["yoy"] = q["value"] / q["value_prev"] - 1
    q.loc[q["value_prev"].fillna(0) <= 0, "yoy"] = np.nan  # meaningless on a negative base
    q = q.sort_values("end").reset_index(drop=True)
    gap = q["end"].diff().dt.days
    q["qoq"] = (q["value"] / q["value"].shift(1) - 1).where(gap.between(80, 100))
    q.loc[q["value"].shift(1).fillna(0) <= 0, "qoq"] = np.nan
    return q[
        [
            "start",
            "end",
            "value",
            "yoy",
            "qoq",
            "derived",
            "concept",
            "unit",
            "form",
            "accn",
            "filed",
            "first_filed",
        ]
    ]


def ttm_series(q: pd.DataFrame) -> pd.DataFrame:
    """Trailing-twelve-month sums over 4 CONSECUTIVE quarters -> DataFrame[end, ttm, available].

    ``available`` is the latest ``first_filed`` among the four quarters.
    """
    if q is None or q.empty or len(q) < 4:
        return pd.DataFrame(columns=["end", "ttm", "available"])
    q = q.sort_values("end").reset_index(drop=True)
    gaps = q["end"].diff().dt.days
    rows = []
    for i in range(3, len(q)):
        if not gaps.iloc[i - 2 : i + 1].between(80, 100).all():
            continue
        win = q.iloc[i - 3 : i + 1]
        rows.append(
            {
                "end": win["end"].iloc[-1],
                "ttm": float(win["value"].sum()),
                "available": win["first_filed"].max(),
            }
        )
    return pd.DataFrame(rows, columns=["end", "ttm", "available"])


def xbrl_snapshot(facts: dict, as_of: date, max_age_days: int = 200) -> dict[str, float]:
    """Latest-quarter YoY growth, TTM operating margin and TTM diluted EPS from companyfacts.

    ``earnings_growth_yoy`` uses NET INCOME, which is free of share-count/split effects. Values from a
    quarter that ended more than ``max_age_days`` before ``as_of`` are ignored (stale filer).
    """
    out: dict[str, float] = {}
    cutoff = pd.Timestamp(as_of) - _days(max_age_days)

    def fresh(q: pd.DataFrame) -> bool:
        return q is not None and not q.empty and q["end"].iloc[-1] >= cutoff

    def last_yoy(q: pd.DataFrame) -> float | None:
        return _num(q["yoy"].iloc[-1]) if fresh(q) else None

    rev = quarterly_series(facts, "revenue")
    ni = quarterly_series(facts, "net_income")
    op = quarterly_series(facts, "operating_income")
    eps = quarterly_series(facts, "eps_diluted")
    if (v := last_yoy(rev)) is not None:
        out["revenue_growth_yoy"] = v
    if (v := last_yoy(ni)) is not None:
        out["earnings_growth_yoy"] = v
    rev_t, op_t = ttm_series(rev), ttm_series(op)
    if fresh(rev) and fresh(op) and not rev_t.empty and not op_t.empty and _same_unit(rev, op):
        if rev_t["end"].iloc[-1] == op_t["end"].iloc[-1] and rev_t["ttm"].iloc[-1] > 0:
            out["operating_margin"] = float(op_t["ttm"].iloc[-1] / rev_t["ttm"].iloc[-1])
    eps_t = ttm_series(eps)
    if fresh(eps) and not eps_t.empty and eps_t["end"].iloc[-1] == eps["end"].iloc[-1]:
        if _is_usd_per_share(eps):  # compared with a USD price downstream
            out["eps_ttm"] = float(eps_t["ttm"].iloc[-1])
        else:
            log.info(
                "XBRL EPS unit %s is not USD/shares - eps_ttm skipped",
                eps["unit"].iloc[-1],
            )
    return out


def _same_unit(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return set(a["unit"]) == set(b["unit"]) and a["unit"].nunique() == 1


def _is_usd_per_share(q: pd.DataFrame) -> bool:
    return not q.empty and set(q["unit"].str.upper()) == {"USD/SHARES"}


_XBRL_CROSSCHECK_TOL = {
    "revenue_growth_yoy": 0.05,
    "earnings_growth_yoy": 0.15,
    "operating_margin": 0.05,
}


def merge_xbrl_snapshot(fund: Fundamentals, snap: dict[str, float], price: float | None, label: str = "") -> list[str]:
    """Fill missing ``fund`` fields from an XBRL snapshot (in place). Returns the filled field names.

    Fields that Yahoo already has are only cross-checked. A large disagreement is logged, and the
    Yahoo value is kept because it may include a quarter whose 10-Q is not filed yet.
    """
    filled: list[str] = []
    for fld, tol in _XBRL_CROSSCHECK_TOL.items():
        x = snap.get(fld)
        if x is None:
            continue
        y = getattr(fund, fld)
        if y is None:
            setattr(fund, fld, x)
            filled.append(fld)
        elif abs(y - x) > tol:
            log.warning(
                "%s %s: yfinance %.4f vs SEC XBRL %.4f (kept yfinance)",
                label,
                fld,
                y,
                x,
            )
    if fund.eps is None and snap.get("eps_ttm") is not None:
        fund.eps = snap["eps_ttm"]
        filled.append("eps")
    if fund.per is None and fund.eps and price:
        fund.per = price / fund.eps
        filled.append("per")
    return filled


def trailing_pe_history(
    facts: dict, close: pd.Series, splits: pd.Series | None, max_age_days: int = 200
) -> pd.DataFrame | None:
    """Point-in-time trailing P/E: ``close / diluted-EPS TTM``.

    ``close`` must be split-adjusted but NOT dividend-adjusted (a dividend-adjusted close biases old
    P/Es low, badly so for high-yielders). The TTM becomes usable on ``first_filed + 1 day`` (a filing
    after the close is tradable the next session). The P/E is NaN when the TTM EPS is <= 0 (loss) or
    older than ``max_age_days``.
    """
    q = quarterly_series(facts, "eps_diluted", splits=splits)
    if not q.empty and not _is_usd_per_share(q):
        log.warning("P/E history skipped: EPS unit %s is not USD/shares", sorted(set(q["unit"])))
        return None
    ttm = ttm_series(q)
    if ttm.empty or close is None or close.empty:
        return None
    ttm = ttm.assign(available=pd.to_datetime(ttm["available"]) + _days(1))
    ttm = ttm.sort_values(["available", "end"]).drop_duplicates("available", keep="last")
    s_eps = pd.Series(ttm["ttm"].to_numpy(), index=pd.DatetimeIndex(ttm["available"]))
    s_av = pd.Series(pd.DatetimeIndex(ttm["available"]), index=pd.DatetimeIndex(ttm["available"]))
    grid = close.index.union(s_eps.index)
    eps_on = s_eps.reindex(grid).ffill().reindex(close.index)
    av_on = s_av.reindex(grid).ffill().reindex(close.index)
    age = (close.index.to_series() - av_on).dt.days
    per = (close / eps_on).where((eps_on > 0) & (age <= max_age_days))
    out = pd.DataFrame({"per": per.replace([np.inf, -np.inf], np.nan)}).dropna()
    out.index.name = "date"
    return out if not out.empty else None


# =============================================================================== Motley Fool

# Slugs: '<company>-<ticker>-q2-2027-earnings-call-transcript' and, since 2026, '...-earnings-transcript'.
_FOOL_LINK_RE = re.compile(
    r"/earnings/call-transcripts/(\d{4})/(\d\d)/(\d\d)/"
    r"([a-z0-9\-]+?-q([1-4])-(\d{4})-earnings(?:-call)?-transcript)/"
)
_CHIP_RE = re.compile(r"\(\s*(?:NASDAQ|NYSE)?:?\s*[A-Z.]{1,6}\s*[-+]?\d+(?:\.\d+)?%\s*\)")
_FOOL_TICKER_RE = re.compile(r"[A-Za-z][A-Za-z.\-]{0,5}")


def _fool_slug_tickers(ticker: str) -> set[str]:
    """Ticker spellings in Fool slugs: 'brk-b' (and the other share class 'brk-a'), 'googl' <-> 'goog'."""
    t = normalise_us_ticker(ticker).lower().replace(".", "-")
    out = {t}
    m = re.fullmatch(r"([a-z]+)-([a-z])", t)
    if m:
        out |= {f"{m.group(1)}-a", f"{m.group(1)}-b", m.group(1)}
    if t in ("goog", "googl"):
        out |= {"goog", "googl"}
    return out


def parse_fool_links(page: str, ticker: str | None = None) -> list[dict]:
    """Transcript links on a fool.com quote page, newest FISCAL period first.

    The URL date is the Fool publish date, not the call date, and old calls are sometimes
    re-published, so the sort is by (fiscal_year, fiscal_quarter, published). Quote pages also link
    OTHER companies' transcripts (related articles), so with ``ticker`` only slugs of the form
    '<company>-<ticker>-qN-YYYY-...' are kept. Without that filter a peer's newer fiscal period
    (e.g. NVDA FY2027 on MSFT's page) would outrank the company's own call.
    """
    tickers = _fool_slug_tickers(ticker) if ticker else None
    seen, out = set(), []
    for m in _FOOL_LINK_RE.finditer(page or ""):
        slug = m.group(4)
        if slug in seen:
            continue
        seen.add(slug)
        if tickers is not None and not any(re.search(rf"(?:^|-){re.escape(t)}-q[1-4]-\d{{4}}-", slug) for t in tickers):
            continue
        out.append(
            {
                "url": "https://www.fool.com" + m.group(0),
                "published": f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
                "fiscal_quarter": int(m.group(5)),
                "fiscal_year": int(m.group(6)),
            }
        )
    return sorted(
        out,
        key=lambda d: (d["fiscal_year"], d["fiscal_quarter"], d["published"]),
        reverse=True,
    )


def parse_fool_transcript(page: str, url: str = "") -> dict:
    """fool.com transcript page -> {'title','sections','takeaways','risks','summary','transcript',
    'prepared','qa','chars'}.

    Q&A starts at the first 'Operator:' line that opens the question queue AFTER the prepared remarks
    (> 2000 chars). The opening Operator line often mentions the 'question-and-answer session' and
    must not count.
    """
    soup = BeautifulSoup(page, "lxml")
    body = soup.select_one("div.article-body") or soup.select_one("article")
    if body is None:
        raise ValueError(f"fool.com layout changed: no article body in {url}")
    for bad in body.select("script, style, aside, figure, img"):
        bad.decompose()
    sections: dict[str, list[str]] = {}
    cur = "_intro"
    for el in body.find_all(["h2", "h3", "p", "li"]):
        if el.name == "li" and el.find_parent("li"):
            continue
        txt = _CHIP_RE.sub("", el.get_text(" ", strip=True))
        txt = re.sub(r"\s+([,.;:%)])", r"\1", re.sub(r"\s{2,}", " ", txt)).strip()
        if not txt:
            continue
        if el.name == "h2":
            cur = txt
            continue
        sections.setdefault(cur, []).append(("- " if el.name == "li" else "") + txt)
    joined = {k: "\n".join(v) for k, v in sections.items()}
    tkey = next((k for k in joined if "transcript" in k.lower()), None)
    transcript = joined.get(tkey, "") if tkey else "\n".join(joined.values())
    m = next(
        (
            x
            for x in re.finditer(
                r"^Operator:.*?(question|first caller|Q&A|queue)",
                transcript,
                re.I | re.M,
            )
            if x.start() > 2000
        ),
        None,
    )
    prepared, qa = (transcript[: m.start()], transcript[m.start() :]) if m else (transcript, "")

    def get(name: str) -> str:
        return next((v for k, v in joined.items() if k.upper().startswith(name)), "")

    return {
        "url": url,
        "title": soup.title.get_text(strip=True) if soup.title else "",
        "sections": joined,
        "takeaways": get("TAKEAWAYS"),
        "risks": get("RISKS"),
        "summary": get("SUMMARY"),
        "transcript": transcript,
        "prepared": prepared,
        "qa": qa,
        "chars": len(transcript),
    }


def transcript_digest(t: dict, max_chars: int) -> str:
    """Prepared remarks (~55%) + Q&A (~45%) digest. Each part keeps its first lines plus the lines
    with the most guidance/demand/margin keywords, in original order."""
    prepared, qa = (t.get("prepared") or "").strip(), (t.get("qa") or "").strip()
    overhead = 40

    def digest(text: str, budget: int) -> str:
        paras = "\n\n".join(ln.strip() for ln in text.split("\n") if ln.strip())
        if len(paras) <= budget:
            return paras
        return select_paragraphs(paras, budget, keep_first=2, min_len=80) or truncate_head_tail(paras, budget)

    b_prep, b_qa = _allocate([len(prepared), len(qa)], max_chars - overhead, [0.55, 0.45])
    return compose_sections(
        [("Prepared remarks", digest(prepared, b_prep)), ("Q&A", digest(qa, b_qa))],
        max_chars,
    )


# =============================================================================== news

_TAG_RE = re.compile(r"<[^>]+>")
_COMPANY_NOISE = re.compile(
    r"\b(inc|incorporated|corp|corporation|company|companies|co|ltd|limited|plc|holdings?|group|n\.?v|"
    r"s\.?a|class\s+[a-c]|common\s+stock)\b\.?",
    re.I,
)
_GENERIC_FIRST_WORDS = {
    "american",
    "general",
    "united",
    "first",
    "bank",
    "international",
    "national",
    "global",
    "new",
    "southern",
    "eli",
    "walt",
    "johnson",
    "texas",
    "applied",
    "advanced",
    "home",
    "public",
    "digital",
    "marathon",
    "union",
    "realty",
    "charles",
    "duke",
    "constellation",
    "crown",
    "air",
    "martin",
    "philip",
    "intuitive",
    "booking",
    "lowe's",
    "the",
    "royal",
    "western",
    "eastern",
    "north",
    "south",
}
_EXTRA_ALIASES: dict[str, tuple[str, ...]] = {
    "AAPL": ("Apple", "iPhone"),
    "MSFT": ("Microsoft",),
    "NVDA": ("Nvidia",),
    "AMZN": ("Amazon", "AWS"),
    "GOOGL": ("Google", "Alphabet"),
    "GOOG": ("Google", "Alphabet"),
    "META": ("Meta", "Facebook", "Instagram"),
    "TSLA": ("Tesla",),
    "BRK-B": ("Berkshire",),
    "BRK-A": ("Berkshire",),
    "DIS": ("Disney",),
    "LLY": ("Lilly",),
    "XOM": ("Exxon", "ExxonMobil"),
    "TMUS": ("T-Mobile",),
    "T": ("AT&T",),
    "PM": ("Philip Morris",),
    "TMO": ("Thermo Fisher",),
    "HD": ("Home Depot",),
    "JPM": ("JPMorgan", "JP Morgan"),
    "GS": ("Goldman",),
    "MS": ("Morgan Stanley",),
    "BAC": ("Bank of America", "BofA"),
    "WFC": ("Wells Fargo",),
    "KO": ("Coca-Cola",),
    "PG": ("Procter & Gamble", "P&G"),
    "JNJ": ("Johnson & Johnson", "J&J"),
    "UNH": ("UnitedHealth",),
    "WMT": ("Walmart",),
    "V": ("Visa",),
    "MA": ("Mastercard",),
}


def _clean_text(value: Any) -> str:
    if not value:
        return ""
    s = _html.unescape(_TAG_RE.sub(" ", str(value))).translate(_QUOTES)
    return re.sub(r"\s+", " ", s).strip()


def clean_company_name(name: str | None) -> str:
    """'Apple Inc.' -> 'Apple', 'Amazon.com, Inc.' -> 'Amazon', 'JPMorgan Chase & Co.' -> 'JPMorgan Chase'."""
    if not name:
        return ""
    s = str(name).translate(_QUOTES)
    s = re.sub(r"\.com\b", "", s, flags=re.I)
    s = re.sub(r"^\s*the\s+", "", s, flags=re.I)
    s = re.sub(r"\s+new\s*$", "", s, flags=re.I)
    s = _COMPANY_NOISE.sub(" ", s)
    s = re.sub(r"[,()]|\.(?=\s|$)", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" &-.")
    s = re.sub(r"\s+(and|&|the)$", "", s, flags=re.I).strip(" &-.")
    return s


def company_aliases(ticker: str, company: str | None) -> list[str]:
    """Case-insensitive name aliases (cleaned name, distinctive first word, well-known brands)."""
    out: list[str] = []
    base = clean_company_name(company)
    if base:
        out.append(base)
        first = base.split()[0]
        if len(first) >= 4 and first.lower() not in _GENERIC_FIRST_WORDS and first.lower() != base.lower():
            out.append(first)
    out.extend(_EXTRA_ALIASES.get(ticker.upper(), ()))
    # the ticker has its own (case-sensitive) pattern; a 1-2 letter alias would match any word fragment
    t = ticker.upper()
    return list(dict.fromkeys(a for a in out if len(a) >= 3 and a.upper() not in (t, t.replace("-", "."))))


class NewsRelevance:
    """Is an article about this company? Checks the ticker (case-sensitive) and company aliases.

    Tickers of 1-2 letters (T, V, O, MA...) are English words, so for them only explicit ticker
    notations count: '$T', '(T)', 'NYSE: T'.
    """

    def __init__(self, ticker: str, company: str | None):
        t = normalise_us_ticker(ticker)
        variants = {t, t.replace("-", "."), t.replace("-", "/")}
        self.patterns: list[re.Pattern] = []
        for v in sorted(variants):
            core = re.sub(r"[^A-Z]", "", v)
            if len(core) <= 2:
                self.patterns.append(
                    re.compile(
                        rf"(?:\$|\(|(?:NYSE|NASDAQ|Nasdaq|NYSEARCA|NYSE American)\s*:\s*){re.escape(v)}(?![A-Za-z0-9])"
                    )
                )
            else:
                self.patterns.append(re.compile(rf"(?<![A-Za-z0-9]){re.escape(v)}(?![A-Za-z0-9])"))
        for alias in company_aliases(t, company):
            self.patterns.append(re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.I))

    def is_relevant(self, text: str) -> bool:
        s = (text or "").translate(_QUOTES)
        return any(p.search(s) for p in self.patterns)


def _parse_time(value: Any) -> datetime | None:
    ts = _to_utc_ts(value)
    return ts.to_pydatetime() if ts is not None else None


def parse_yf_news(items: Iterable[dict]) -> list[dict]:
    """yfinance 1.x ``get_news`` items ({'id','content':{...}}) -> normalised rows (UTC datetimes)."""
    rows = []
    for it in items or []:
        c = it.get("content") if isinstance(it.get("content"), dict) else it
        title = _clean_text(c.get("title"))
        ts = _parse_time(c.get("pubDate") or c.get("displayTime") or c.get("providerPublishTime"))
        if not title or ts is None:
            continue
        url = (
            (c.get("canonicalUrl") or {}).get("url")
            or (c.get("clickThroughUrl") or {}).get("url")
            or c.get("link")
            or ""
        )
        provider = c.get("provider") if isinstance(c.get("provider"), dict) else {}
        publisher = str(provider.get("displayName") or c.get("publisher") or "")
        rows.append(
            {
                "title": title,
                "summary": _clean_text(c.get("summary") or c.get("description")),
                "url": str(url),
                "source": publisher or "Yahoo Finance",
                "publisher": publisher,
                "published_at": ts,
                "origin": "yahoo",
            }
        )
    return rows


def parse_rss(content: bytes | str, origin: str) -> list[dict]:
    """Google News / Yahoo Finance RSS -> normalised rows.

    Google titles end in ' - Publisher' (stripped), and its 'summary' is only an anchor tag, so it is
    dropped.
    """
    feed = feedparser.parse(content)
    if getattr(feed, "bozo", False) and not feed.entries:
        raise ValueError(f"RSS parse error ({origin}): {getattr(feed, 'bozo_exception', '')}")
    rows = []
    for e in feed.entries:
        title = _clean_text(e.get("title"))
        publisher = ((e.get("source") or {}).get("title") or "") if origin == "google" else ""
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(publisher) - 3].rstrip()
        if e.get("published_parsed"):
            ts: datetime | None = datetime(*e["published_parsed"][:6], tzinfo=UTC)
        else:
            ts = _parse_time(e.get("published") or e.get("updated"))
        if not title or ts is None:
            continue
        rows.append(
            {
                "title": title,
                "summary": "" if origin == "google" else _clean_text(e.get("summary")),
                "url": str(e.get("link") or ""),
                "source": publisher or ("Google News" if origin == "google" else "Yahoo Finance"),
                "publisher": publisher,
                "published_at": ts,
                "origin": origin,
            }
        )
    return rows


def self_published_matcher(ticker: str, company: str | None) -> Callable[[str], bool]:
    """True for the company's OWN channels ('nvidia.com', 'developer.nvidia.com', 'NVIDIA Blog',
    'Apple Newsroom', and a bare 'NVIDIA', which is how Google News labels nvidia.com pages). They are
    marketing, not independent news, and would bias sentiment upward. Independent titles that merely
    contain the name ('AppleInsider') are kept."""
    names = {re.sub(r"[^a-z0-9]", "", a.lower()) for a in company_aliases(ticker, company)}
    names = {n for n in names if len(n) >= 3}
    pats = []
    for n in sorted(names):
        pats.append(re.compile(rf"^(?:[a-z0-9-]+\.)*{re.escape(n)}\.(?:com|net|org|co|io|ai)$"))
        pats.append(
            re.compile(  # 'Microsoft Source' / 'Microsoft Azure' / 'Google Cloud' seen live
                rf"^(?:about\s*)?{re.escape(n)}\s*(?:blog|newsroom|news|investor relations|investors|ir|press|"
                rf"source|stories|developer|azure|cloud)?$"
            )
        )

    def match(publisher: str) -> bool:
        p = (publisher or "").strip().lower()
        compact = re.sub(r"[^a-z0-9. ]", "", p)  # 'Coca-Cola' / 'AT&T' -> alias spelling without symbols
        return bool(p) and any(pt.match(p) or pt.match(compact) or pt.match(compact.replace(" ", "")) for pt in pats)

    return match


def _dedupe_key(title: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", "", (title or "").lower().translate(_QUOTES))
    return re.sub(r"\s+", " ", s).strip()[:60]


def merge_news(
    rows: Iterable[dict],
    since: datetime,
    limit: int,
    ticker: str,
    company: str | None,
    now: datetime | None = None,
) -> list[NewsItem]:
    """Relevance filter, ``since`` window, removal of the company's own channels, de-duplication (by
    normalised title and by URL; a duplicate lends its summary to the kept item), newest first, at most
    ``limit`` items."""
    since_utc = _as_utc_datetime(since)
    horizon = _as_utc_datetime(now) if now else datetime.now(UTC)
    horizon = horizon + pd.Timedelta(1, "h").to_pytimedelta()  # tolerate small clock skew only
    rel = NewsRelevance(ticker, company)
    own = self_published_matcher(ticker, company)
    cand = [
        dict(r)
        for r in rows
        if since_utc <= r["published_at"] <= horizon and rel.is_relevant(f"{r['title']} {r['summary']}")
    ]
    n_own = sum(own(r.get("publisher", "")) for r in cand)
    if n_own:
        log.info(
            "%s news: %d self-published items (company blog/site) excluded",
            ticker,
            n_own,
        )
        cand = [r for r in cand if not own(r.get("publisher", ""))]
    cand.sort(key=lambda r: r["published_at"], reverse=True)
    kept: list[dict] = []
    seen: dict[str, int] = {}
    for r in cand:
        keys = [k for k in (_dedupe_key(r["title"]), r["url"]) if k]
        hit = next((seen[k] for k in keys if k in seen), None)
        if hit is not None:
            if not kept[hit]["summary"] and r["summary"]:
                kept[hit]["summary"] = r["summary"]
            continue
        for k in keys:
            seen[k] = len(kept)
        kept.append(r)
    out = []
    for r in kept[: max(0, limit)]:
        pub = r["published_at"]
        out.append(
            NewsItem(
                id=text_id(r["url"]) if r["url"] else text_id(r["title"], pub.isoformat()),
                title=r["title"],
                summary=r["summary"],
                url=r["url"],
                source=r["source"],
                published_at=pub,
                lang="en",
            )
        )
    return out


def google_news_query(ticker: str, company: str | None, days: int) -> str:
    """'"<Company>" OR "<TICKER> stock" when:<days>d' (the cleaned company name, not the legal name)."""
    base = clean_company_name(company)
    terms = [f'"{base}"'] if base and base.upper() != ticker.upper() else []
    terms.append(f'"{ticker.replace("-", ".")} stock"')  # share classes are written 'BRK.B' in news
    return " OR ".join(terms) + f" when:{max(1, min(int(days), 60))}d"


# =============================================================================== FX


def pick_completed_fx(
    series: pd.Series, tz: str, session: date, now: Any = None, *, lang: str = "ko"
) -> tuple[float, date]:
    """Last COMPLETED daily FX bar (bar date < today in the quote's own timezone), aligned to
    ``session`` (the last completed stock session). KRW=X trades on US holidays, and its bar for
    today is live."""
    now_ts = _to_utc_ts(now) if now is not None else _now_utc()
    assert now_ts is not None
    local_today = now_ts.tz_convert(tz).tz_localize(None).normalize()
    s = series.dropna()
    done = s[s.index < local_today]
    aligned = done[done.index <= pd.Timestamp(session)]
    use = aligned if not aligned.empty else done
    if use.empty:
        raise ValueError(_tr(lang, "완료된 USD/KRW 일봉 없음", "no completed USD/KRW daily bar"))
    rate = float(use.iloc[-1])
    if not 500.0 < rate < 3000.0:
        raise ValueError(_tr(lang, f"USD/KRW 이상치: {rate}", f"USD/KRW outlier: {rate}"))
    return rate, use.index[-1].date()


def parse_naver_fx(rows: list[dict]) -> pd.Series:
    """Naver FX_USDKRW prices JSON (Hana Bank first-notice rate, KST dates) -> Series."""
    if not isinstance(rows, list):
        raise ValueError(f"naver fx: expected a list of rows, got {type(rows).__name__}")
    df = pd.DataFrame(rows)
    if df.empty or "closePrice" not in df.columns:
        raise ValueError("naver fx: no rows")
    s = pd.Series(
        pd.to_numeric(df["closePrice"].astype(str).str.replace(",", ""), errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(df["localTradedAt"])).normalize(),
        name="usdkrw",
    ).sort_index()
    return s[~s.index.duplicated(keep="last")].dropna()


# ---- the ONE USD/KRW definition, shared by USProvider and KRProvider ------------------------------
#
# rate = the last COMPLETED Yahoo ``KRW=X`` daily bar (London bar date < today in London), at or before
# the last completed US session (``last_completed_us_session``); never the live partial bar. Fallback:
# the latest Naver/Hana Bank first-notice rate dated <= today KST (a different definition, always
# flagged). Both providers read and write the SAME cache entry (namespace + key below), so a KR and a US
# ticker converted in one run use the identical rate, and KRW->USD is exactly ``1.0 / (USD->KRW)``.

FX_CACHE_NAMESPACE = "us_fx"
NAVER_FX_PRICES = "https://api.stock.naver.com/marketindex/exchange/FX_USDKRW/prices"
NAVER_FX_PARAMS = {"page": 1, "pageSize": 10}
FX_YF_TRIES = 3  # with_backoff attempts for yfinance KRW=X (tests lower it)


class UnsupportedCurrencyPairError(NotImplementedError, ValueError):
    """Only USD<->KRW is supported (a ValueError for KR callers, NotImplementedError for US ones)."""


def fetch_krw_series_yf(yf_module: Any = None, *, lang: str = "ko") -> tuple[pd.Series, str]:
    """Daily ``KRW=X`` closes of the last month -> (Series on tz-naive London dates, quote timezone).

    ``yf_module`` defaults to ``yfinance``; the KR provider passes its own (patchable) handle.
    """
    mod = yf_module if yf_module is not None else yf

    def call() -> pd.DataFrame:
        _YAHOO_THROTTLE.wait()
        return mod.Ticker("KRW=X").history(period="1mo", auto_adjust=True, actions=False)

    h = with_backoff(call, tries=max(1, int(FX_YF_TRIES)), what="yfinance KRW=X")
    if "Close" not in getattr(h, "columns", ()):
        raise ValueError(_tr(lang, "yfinance KRW=X: Close 열 없음", "yfinance KRW=X: no Close column"))
    tz = str(h.index.tz) if getattr(h.index, "tz", None) is not None else _FX_TZ
    s = h["Close"].astype("float64").copy()
    idx = pd.DatetimeIndex(s.index)
    s.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    return s[~s.index.duplicated(keep="last")], tz


def usdkrw_payload(
    fetch_yahoo: Callable[[], tuple[pd.Series, str]],
    fetch_naver: Callable[[], pd.Series],
    now: Any = None,
    *,
    lang: str = "ko",
) -> dict:
    """Fetch USD/KRW under the shared definition (see above) -> ``{rate, date, source[, error]}``.

    ``source`` is ``"yahoo"`` or ``"naver"`` (flag the latter with :func:`usdkrw_fallback_note`).
    Raises RuntimeError when both sources fail.
    """
    now_ts = _to_utc_ts(now) if now is not None else _now_utc()
    assert now_ts is not None
    session = last_completed_us_session(now_ts)
    yahoo_err = ""
    try:
        series, tz = fetch_yahoo()
        rate, d = pick_completed_fx(series, tz, session, now=now_ts, lang=lang)
        return {"rate": rate, "date": d.isoformat(), "source": "yahoo"}
    except Exception as exc:  # noqa: BLE001 - fallback, flagged via the returned source
        log.warning("USD/KRW via Yahoo failed: %s -> Naver", exc)
        yahoo_err = f"{type(exc).__name__}: {str(exc)[:120]}"
    try:
        s = fetch_naver()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            _tr(
                lang,
                f"USD/KRW 환율 수집 실패 (yahoo: {yahoo_err}; naver: {type(exc).__name__}: {exc})",
                f"USD/KRW FX rate unavailable (yahoo: {yahoo_err}; naver: {type(exc).__name__}: {exc})",
            )
        ) from exc
    today_kst = now_ts.tz_convert("Asia/Seoul").tz_localize(None).normalize()
    s = s[s.index <= today_kst]
    s = s[(s > 500.0) & (s < 3000.0)]
    if s.empty:
        raise RuntimeError(
            _tr(
                lang,
                f"USD/KRW 환율 수집 실패 (yahoo: {yahoo_err}; naver: 유효한 값 없음)",
                f"USD/KRW FX rate unavailable (yahoo: {yahoo_err}; naver: no valid value)",
            )
        )
    return {
        "rate": float(s.iloc[-1]),
        "date": s.index[-1].date().isoformat(),
        "source": "naver",
        "error": yahoo_err,
    }


def usdkrw_cache_key(now: Any = None) -> str:
    """Cache key of the shared USD/KRW entry: the last completed US session + today's London date.

    KRW=X bars are London days: between the US close and London midnight the session's own bar is
    still live and the previous one is served. Keying on the London date as well means that stand-in
    is replaced once the session's bar completes, instead of living for the whole TTL.
    """
    now_ts = _to_utc_ts(now) if now is not None else _now_utc()
    assert now_ts is not None
    session = last_completed_us_session(now_ts)
    fx_day = now_ts.tz_convert(_FX_TZ).date()
    return DiskCache.key("fx", _CACHE_VERSION, "USDKRW", session.isoformat(), fx_day.isoformat())


def cached_usdkrw(
    cache: DiskCache,
    ttl_hours: float | None,
    fetch_yahoo: Callable[[], tuple[pd.Series, str]],
    fetch_naver: Callable[[], pd.Series],
    now: Any = None,
    *,
    lang: str = "ko",
) -> dict:
    """:func:`usdkrw_payload` through the shared disk-cache entry (one fetch per key per process)."""
    now_ts = _to_utc_ts(now) if now is not None else _now_utc()
    return cache_get_or_fetch(
        cache,
        FX_CACHE_NAMESPACE,
        usdkrw_cache_key(now_ts),
        ttl_hours,
        lambda: usdkrw_payload(fetch_yahoo, fetch_naver, now_ts, lang=lang),
    )


def usdkrw_fallback_note(payload: dict, *, lang: str = "ko") -> str | None:
    """Korean report note when the rate did NOT come from the primary definition (else None)."""
    if payload.get("source") == "yahoo":
        return None
    return _tr(
        lang,
        f"환율: Yahoo(KRW=X) 실패 → 네이버 고시환율(하나은행) USD/KRW {float(payload['rate']):,.2f} "
        f"({payload.get('date')}) 사용 — 정의 차이 최대 ~24원",
        f"FX: Yahoo (KRW=X) failed → using the Naver posted rate (Hana Bank) USD/KRW {float(payload['rate']):,.2f} "
        f"({payload.get('date')}) — definitions differ by up to ~24 KRW",
    )


def usdkrw_fx(from_ccy: str, to_ccy: str, usdkrw: Callable[[], float], *, lang: str = "ko") -> float:
    """Amount multiplier ``from_ccy -> to_ccy`` (the ``fx_rate`` of BOTH providers).

    ``usdkrw`` is called only for a supported, non-trivial pair. KRW->USD is exactly ``1.0 / rate``.
    """
    f, t = from_ccy.upper(), to_ccy.upper()
    if f == t:
        return 1.0
    if {f, t} != {"USD", "KRW"}:
        raise UnsupportedCurrencyPairError(
            _tr(lang, f"환율 {f}->{t} 미지원 (USD/KRW만 지원)", f"FX {f}->{t} not supported (USD/KRW only)")
        )
    rate = float(usdkrw())
    return rate if f == "USD" else 1.0 / rate


# =============================================================================== universe

# GICS sector -> large caps, roughly by market cap. Static on purpose: deterministic, no ETFs.
US_UNIVERSE: dict[str, tuple[str, ...]] = {
    "Information Technology": (
        "AAPL",
        "MSFT",
        "NVDA",
        "AVGO",
        "ORCL",
        "CSCO",
        "AMD",
        "CRM",
        "IBM",
        "ADBE",
        "QCOM",
        "TXN",
    ),
    "Financials": (
        "JPM",
        "V",
        "MA",
        "BRK-B",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "AXP",
        "BLK",
        "SPGI",
    ),
    "Health Care": (
        "LLY",
        "JNJ",
        "ABBV",
        "UNH",
        "MRK",
        "ABT",
        "TMO",
        "ISRG",
        "AMGN",
        "PFE",
    ),
    "Consumer Discretionary": (
        "AMZN",
        "TSLA",
        "HD",
        "MCD",
        "BKNG",
        "TJX",
        "LOW",
        "SBUX",
        "NKE",
    ),
    "Communication Services": (
        "GOOGL",
        "META",
        "NFLX",
        "TMUS",
        "DIS",
        "VZ",
        "T",
        "CMCSA",
    ),
    "Industrials": (
        "GE",
        "CAT",
        "RTX",
        "UNP",
        "ETN",
        "HON",
        "DE",
        "LMT",
        "UPS",
        "ADP",
        "BA",
    ),
    "Consumer Staples": ("WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "MDLZ", "CL"),
    "Energy": ("XOM", "CVX", "COP", "EOG", "SLB", "WMB", "KMI", "PSX", "MPC", "OXY"),
    "Utilities": ("NEE", "SO", "DUK", "CEG", "AEP", "SRE", "D", "EXC", "XEL"),
    "Real Estate": ("PLD", "AMT", "EQIX", "WELL", "SPG", "PSA", "O", "DLR", "CCI"),
    "Materials": ("LIN", "SHW", "ECL", "APD", "FCX", "NEM", "NUE", "MLM", "VMC"),
}
# GICS -> Yahoo sector naming (used for sector peers and as a fallback sector)
GICS_TO_YAHOO = {
    "Information Technology": "Technology",
    "Financials": "Financial Services",
    "Health Care": "Healthcare",
    "Consumer Discretionary": "Consumer Cyclical",
    "Communication Services": "Communication Services",
    "Industrials": "Industrials",
    "Consumer Staples": "Consumer Defensive",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Materials": "Basic Materials",
}
_STATIC_SECTOR_YAHOO = {t: GICS_TO_YAHOO[s] for s, ts in US_UNIVERSE.items() for t in ts}


def interleaved_universe() -> list[str]:
    """Round-robin over sectors (fixed order), so any prefix stays sector-diverse."""
    lists = list(US_UNIVERSE.values())
    out: list[str] = []
    for i in range(max(len(v) for v in lists)):
        out.extend(v[i] for v in lists if i < len(v))
    return list(dict.fromkeys(out))


# =============================================================================== provider


class SecSession(PoliteSession):
    """PoliteSession (retries, per-host spacing) + a GLOBAL <= 8 req/s budget across sec.gov hosts."""

    def __init__(self, user_agent: str):
        super().__init__(user_agent=user_agent, timeout=30.0)
        self.headers.pop("Accept-Language", None)
        self.headers["Accept-Encoding"] = "gzip, deflate"

    def request(self, method, url, *args, **kwargs):  # type: ignore[override]
        _SEC_THROTTLE.wait()
        return super().request(method, url, *args, **kwargs)


@dataclass
class _PlanStep:
    """One candidate filing: selected by (priority, order) until ``limit`` items are built."""

    priority: int
    order: int
    label: str
    build: Callable[[], FilingItem | None]


class USProvider(MarketProvider):
    """US equities: yfinance/Yahoo/Nasdaq prices, SEC EDGAR + Motley Fool text, Yahoo/Naver FX.

    Every public method may raise. Degraded-but-served results are flagged via ``DataFrame.attrs``
    and Korean notes (``data_warnings`` / ``drain_notes``). See the module docstring for the source
    chains.
    """

    market = Market.US
    SECTOR_PEERS_MAX = 12
    SECTOR_PEERS_MIN = 4

    def __init__(self, settings: Settings, cache: DiskCache):
        super().__init__(settings, cache)
        self._notes: dict[str, list[str]] = {}
        self._notes_lock = threading.Lock()
        self._sec_session: SecSession | None = None
        self._sec_lock = threading.Lock()
        self._web = PoliteSession()

    # ------------------------------------------------------------------ cache

    def _cached(
        self,
        namespace: str,
        key: str,
        ttl_hours: float | None,
        fetch: Callable[[], T],
    ) -> T:
        """:func:`cache_get_or_fetch` on this provider's cache (per-key lock, non-fatal write)."""
        return cache_get_or_fetch(self.cache, namespace, key, ttl_hours, fetch)

    # ------------------------------------------------------------------ notes

    def _note(self, ticker: str, message: str) -> None:
        log.warning("%s: %s", ticker, message)
        with self._notes_lock:
            lst = self._notes.setdefault(ticker, [])
            if message not in lst:
                lst.append(message)

    def _collect_notes(self, ticker: str, clear: bool) -> list[str]:
        with self._notes_lock:  # read (and clear) atomically: a concurrent note is never lost
            key = normalise_us_ticker(ticker)
            own = self._notes.pop(key, []) if clear else list(self._notes.get(key, []))
            glob = list(self._notes.get(_GLOBAL_NOTES, []))
        return own + [g for g in glob if g not in own]

    def data_warnings(self, ticker: str) -> list[str]:
        """Korean, report-ready notes on every degraded / fallback path taken for ``ticker``
        (``MarketProvider`` API, merged into ``MarketData.warnings`` by the shared loader).

        Non-destructive. The ticker's own notes come first, then provider-wide ones (an FX source
        switch). Frame-level notes are ALSO in the frame's ``attrs["warnings"]`` with identical text
        (benchmark notes as ``"벤치마크: " + attrs text``), so the loader's exact-match merge keeps one.
        """
        return self._collect_notes(ticker, clear=False)

    def drain_notes(self, ticker: str) -> list[str]:
        """Alias of :meth:`data_warnings` that also clears the ticker's own notes (provider-wide notes
        stay). Kept for callers that poll one provider repeatedly."""
        return self._collect_notes(ticker, clear=True)

    # ------------------------------------------------------------------ resolve

    def resolve(self, ticker: str) -> Instrument:
        sym = normalise_us_ticker(ticker)
        if not sym:
            raise ValueError(_tr(self.settings, "빈 티커", "empty ticker"))
        lang = lang_of(self.settings)
        try:
            inst = instrument_from_info(sym, self._info(sym)["info"], lang=lang)
        except UnsupportedInstrumentError:
            raise
        except Exception as exc:  # noqa: BLE001 - fall back to chart metadata, flagged
            try:
                inst = instrument_from_chart_meta(sym, self._chart_meta(sym), lang=lang)
            except UnsupportedInstrumentError:
                raise
            except Exception as exc2:  # noqa: BLE001
                why = f"info: {type(exc).__name__}: {exc}; chart: {type(exc2).__name__}: {exc2}"
                raise RuntimeError(
                    _tr(self.settings, f"{sym}: 종목 확인 실패 ({why})", f"{sym}: symbol lookup failed ({why})")
                ) from exc2
            self._note(
                sym,
                _tr(
                    self.settings,
                    f"종목정보: Yahoo info 실패({type(exc).__name__}) → 차트 메타데이터로 대체",
                    f"Instrument info: Yahoo info failed ({type(exc).__name__}) → using chart metadata",
                ),
            )
        if inst.currency != "USD":
            self._note(
                sym,
                _tr(
                    self.settings,
                    f"통화가 USD가 아님({inst.currency}) — 환율 변환이 지원되지 않을 수 있음",
                    f"Currency is not USD ({inst.currency}) — FX conversion may not be supported",
                ),
            )
        return inst

    def _info(self, symbol: str) -> dict:
        """Cached ``{'as_of': iso date (ET), 'info': yfinance .info}`` for ``price_ttl_hours``."""

        def fetch() -> dict:
            def call() -> dict:
                _YAHOO_THROTTLE.wait()
                return yf.Ticker(symbol).info

            info = with_backoff(call, what=f"yfinance info {symbol}")
            if not isinstance(info, dict) or not (
                info.get("quoteType") or info.get("longName") or info.get("shortName")
            ):
                raise ValueError(
                    _tr(
                        self.settings,
                        f"{symbol}: Yahoo info 비어 있음 (잘못된 티커?)",
                        f"{symbol}: Yahoo info empty (wrong ticker?)",
                    )
                )
            return {"as_of": today_et().isoformat(), "info": info}

        # keyed on the last completed session like the price cache: a quote fetched intraday (ratios
        # re-priced to the previous close, see session_anchor_price) must not outlive the next close
        # while the price series moves on to that new bar
        key = DiskCache.key("info", _CACHE_VERSION, symbol, last_completed_us_session().isoformat())
        return self._cached("us_info", key, self.settings.data.price_ttl_hours, fetch)

    def _chart_meta(self, symbol: str) -> dict:
        res = fetch_yahoo_chart(symbol, today_et() - pd.Timedelta(10, "D").to_pytimedelta())
        return res.get("meta") or {}

    # ------------------------------------------------------------------ prices

    def prices(self, inst: Instrument, start: date) -> pd.DataFrame:
        """Adjusted daily OHLCV.

        ``attrs``: ``source``, ``adjusted`` (False when only a split-adjusted fallback worked, e.g.
        Nasdaq), ``missing_sessions`` (vs the SPY calendar, AFTER repair), ``repaired_sessions``
        (``{date: source}`` of bars filled by :meth:`_repair_sessions`) and ``warnings`` - the Korean
        notes for this frame, identical to the ones recorded for :meth:`data_warnings`.
        """
        sym = inst.yf_symbol or inst.ticker
        start_d = _as_date(start)
        df = self._cached_prices(sym, start_d, ("stocks", "etf")).copy()
        session_notes: list[str] = []
        if sym != BENCHMARK:
            df, session_notes = self._check_sessions(inst.ticker, sym, df, start_d)
        price = _tr(self.settings, "가격: ", "Price: ")
        notes = [f"{price}{m}" for m in self._frame_issues(df, lang=lang_of(self.settings))] + session_notes
        df.attrs["warnings"] = notes
        for n in notes:
            self._note(inst.ticker, n)
        return df

    def _check_sessions(
        self, ticker: str, symbol: str, df: pd.DataFrame, start: date
    ) -> tuple[pd.DataFrame, list[str]]:
        """Sessions the stock frame lacks vs the SPY calendar: repaired where a second source has the
        bar (:meth:`_repair_sessions`), the rest reported. Sets ``attrs['missing_sessions']`` and
        ``attrs['repaired_sessions']``; returns the (repaired) frame and its Korean notes."""
        try:
            cal = self._cached_prices(BENCHMARK, start, ("etf", "stocks")).index
        except Exception as exc:  # noqa: BLE001 - the loader reports a benchmark failure on its own
            log.warning("%s: session check skipped (SPY calendar unavailable: %s)", ticker, exc)
            return df, []
        miss = missing_sessions(df, cal)
        repaired: dict[str, str] = {}
        unresolved: dict[str, str] = {}
        if miss:
            df, repaired, unresolved = self._repair_sessions(symbol, df, miss)
            miss = missing_sessions(df, cal)
        df.attrs["missing_sessions"] = [d.date().isoformat() for d in miss]
        df.attrs["repaired_sessions"] = dict(repaired)
        notes = []
        if repaired:
            labels = {
                "yahoo_chart": _tr(self.settings, "Yahoo 차트 재조회", "Yahoo chart re-fetch"),
                "nasdaq": _tr(self.settings, "나스닥 종가", "Nasdaq close"),
            }
            items = sorted(repaired.items())
            shown = ", ".join(f"{d} {labels.get(s, s)}" for d, s in items[-5:])
            more = (
                _tr(self.settings, f" 외 {len(items) - 5}일", f" and {len(items) - 5} more day(s)")
                if len(items) > 5
                else ""
            )
            notes.append(
                _tr(
                    self.settings,
                    f"가격: Yahoo 결측 {len(items)}거래일 보정({shown}{more}) — 인접일 수정주가 비율로 환산해 채움",
                    f"Price: repaired {len(items)} session(s) missing from Yahoo ({shown}{more}) — filled by "
                    "rescaling with the adjacent days' adjusted-price ratio",
                )
            )
        if not miss:
            return df, notes
        if len(df) and cal[-1] not in df.index:
            notes.append(
                _tr(
                    self.settings,
                    f"가격: 최근 거래일 {cal[-1].date()} 봉 누락(Yahoo 결측/거래정지) — "
                    f"최신가가 {df.index[-1].date()} 기준",
                    f"Price: bar for the latest session {cal[-1].date()} missing (Yahoo gap/trading halt) — latest "
                    f"price is as of {df.index[-1].date()}",
                )
            )
        shown = ", ".join(d.date().isoformat() for d in miss[-3:])
        why = sorted({unresolved[d.date().isoformat()] for d in miss if d.date().isoformat() in unresolved})
        reason = (
            _tr(
                self.settings,
                f" (재조회 보정 불가: {'; '.join(why)[:160]})",
                f" (not repairable by re-fetch: {'; '.join(why)[:160]})",
            )
            if why
            else ""
        )
        notes.append(
            _tr(
                self.settings,
                f"가격: SPY 거래일 대비 {len(miss)}거래일 결측({shown}) — 결측 구간 수익률은 다일간 수익률{reason}",
                f"Price: {len(miss)} session(s) missing vs the SPY calendar ({shown}) — returns across the gaps are "
                f"multi-day returns{reason}",
            )
        )
        return df, notes

    def _repair_sessions(
        self, symbol: str, df: pd.DataFrame, miss: list[pd.Timestamp]
    ) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
        """Fill sessions a Yahoo series lacks (``close=None`` holes) from a second source.

        Chain: the raw Yahoo v8 chart (full window up to now; a narrow window can return a mislabelled
        bar), then Nasdaq for the sessions still missing. Each bar is scaled onto the frame's price
        basis and validated by :func:`scale_missing_bars`. Only single sessions are filled - the series
        itself is never replaced. The result is cached per (symbol, session, holes) once it is
        definitive (every source answered); a transient failure is retried on the next call.
        Returns ``(frame, {date: source}, {date: reason})`` for repaired / unresolved sessions.
        """
        isos = [d.date().isoformat() for d in miss]
        key = DiskCache.key(
            "repair",
            _CACHE_VERSION,
            symbol,
            df.index[0].date().isoformat() if len(df) else "",
            last_completed_us_session().isoformat(),
            isos,
            df.attrs.get("source"),
        )
        payload = self.cache.get("us_repair", key, self.settings.data.price_ttl_hours)
        if payload is None:
            payload = self._repair_payload(symbol, df, miss)
            if payload["definitive"]:
                try:
                    self.cache.set("us_repair", key, payload)
                except OSError as exc:
                    log.warning("cache write failed (us_repair/%s): %s", key, exc)
        bars = {pd.Timestamp(k): v for k, v in payload["bars"].items()}
        out = apply_repaired_bars(df, bars)
        if payload["bars"]:
            log.warning(
                "%s: repaired %d missing session(s) %s",
                symbol,
                len(payload["bars"]),
                payload["sources"],
            )
        if payload["unresolved"]:
            log.warning(
                "%s: %d missing session(s) not repaired: %s (source errors: %s)",
                symbol,
                len(payload["unresolved"]),
                payload["unresolved"],
                payload["errors"],
            )
        return out, dict(payload["sources"]), dict(payload["unresolved"])

    def _repair_payload(self, symbol: str, df: pd.DataFrame, miss: list[pd.Timestamp]) -> dict:
        todo = sorted(pd.Timestamp(d) for d in miss)
        window_start = (todo[0] - pd.Timedelta(30, "D")).date()
        bars: dict[str, list[float]] = {}
        sources: dict[str, str] = {}
        reasons: dict[str, list[str]] = {}
        errors: list[str] = []
        events: pd.DatetimeIndex | None = None
        steps: list[tuple[str, Callable[[], pd.DataFrame]]] = []

        def chart() -> pd.DataFrame:
            nonlocal events
            res = self._repair_chart(symbol, window_start)
            events = chart_event_dates(res)
            return parse_yahoo_chart(res)

        steps.append(("yahoo_chart", chart))
        steps.append(("nasdaq", lambda: self._repair_nasdaq(symbol, window_start, ("stocks", "etf"))))
        for name, fetch in steps:
            if not todo:
                break
            try:
                cand = fetch()
            except Exception as exc:  # noqa: BLE001 - next source; the hole stays reported if all fail
                errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:120]}")
                for d in todo:
                    reasons.setdefault(d.date().isoformat(), []).append(
                        _tr(self.settings, f"{name} 조회 실패", f"{name} fetch failed")
                    )
                continue
            got, rejected = scale_missing_bars(df, todo, cand, events, lang=lang_of(self.settings))
            for d, bar in got.items():
                bars[d.date().isoformat()], sources[d.date().isoformat()] = bar, name
            for d, why in rejected.items():
                reasons.setdefault(d.date().isoformat(), []).append(f"{name}: {why}")
            todo = [d for d in todo if d not in got]
        return {
            "bars": bars,
            "sources": sources,
            "unresolved": {d.date().isoformat(): "; ".join(reasons.get(d.date().isoformat(), [])) for d in todo},
            "errors": errors,
            "definitive": not errors or not todo,
        }

    def _repair_chart(self, symbol: str, start: date) -> dict:
        """Raw Yahoo v8 chart result (window from ``start`` up to now) for the session repair."""
        return fetch_yahoo_chart(symbol, start)

    def _repair_nasdaq(self, symbol: str, start: date, classes: tuple[str, ...]) -> pd.DataFrame:
        """Nasdaq bars (split-adjusted, unadjusted for dividends) for the session repair."""
        return self._fetch_nasdaq_frame(symbol, start, classes)

    def benchmark(self, inst: Instrument, start: date) -> pd.DataFrame:
        """SPY through the same chain (SPY, unlike ^GSPC, has no missing days).

        ``attrs["warnings"]`` holds the frame's notes WITHOUT a label; the ticker's note is the same text
        prefixed with ``"벤치마크: "``, exactly what the shared loader writes for benchmark frames.
        """
        sym = inst.benchmark or BENCHMARK
        df = self._cached_prices(sym, _as_date(start), ("etf", "stocks")).copy()
        frame_notes = [f"{sym} {m}" for m in self._frame_issues(df, lang=lang_of(self.settings))]
        df.attrs["warnings"] = frame_notes
        bench = _tr(self.settings, "벤치마크: ", "Benchmark: ")
        for n in frame_notes:
            self._note(inst.ticker, f"{bench}{n}")
        return df

    def _cached_prices(self, symbol: str, start: date, nasdaq_classes: tuple[str, ...]) -> pd.DataFrame:
        key = DiskCache.key(
            "px",
            _CACHE_VERSION,
            symbol,
            start.isoformat(),
            last_completed_us_session().isoformat(),
        )
        return self._cached(
            "us_prices",
            key,
            self.settings.data.price_ttl_hours,
            lambda: self._price_chain(symbol, start, nasdaq_classes),
        )

    def _price_chain(self, symbol: str, start: date, nasdaq_classes: tuple[str, ...]) -> pd.DataFrame:
        """yfinance -> Yahoo chart -> Nasdaq. One source per series, never spliced."""
        attempts: list[tuple[str, Callable[[], pd.DataFrame]]] = [
            ("yfinance", lambda: self._fetch_yf_frame(symbol, start)),
            ("yahoo_chart", lambda: self._fetch_chart_frame(symbol, start)),
            ("nasdaq", lambda: self._fetch_nasdaq_frame(symbol, start, nasdaq_classes)),
        ]
        errors: list[str] = []
        for name, fn in attempts:
            try:
                df = fn()
                if df is None or df.empty:
                    raise ValueError("empty frame")
            except Exception as exc:  # noqa: BLE001 - try the next source, report all at the end
                errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:160]}")
                log.warning("%s prices via %s failed: %s", symbol, name, exc)
                continue
            df.attrs["failed_sources"] = list(errors)
            if errors:
                log.warning(
                    "%s prices served by fallback source %s (adjusted=%s)",
                    symbol,
                    name,
                    df.attrs.get("adjusted"),
                )
            return df
        raise RuntimeError(
            _tr(self.settings, f"{symbol}: 모든 가격 소스 실패 - ", f"{symbol}: all price sources failed - ")
            + " | ".join(errors)
        )

    def _fetch_yf_frame(self, symbol: str, start: date) -> pd.DataFrame:
        raw, reg_end = _yf_history(symbol, start)
        return frame_from_yf_history(raw, reg_end)

    def _fetch_chart_frame(self, symbol: str, start: date) -> pd.DataFrame:
        return parse_yahoo_chart(fetch_yahoo_chart(symbol, start))

    def _fetch_nasdaq_frame(self, symbol: str, start: date, classes: tuple[str, ...]) -> pd.DataFrame:
        errors = []
        for cls in classes:
            try:
                return parse_nasdaq_rows(fetch_nasdaq_rows(symbol, start, cls))
            except Exception as exc:  # noqa: BLE001 - wrong asset class answers 'Symbol not exists'
                errors.append(f"{cls}: {exc}")
        raise RuntimeError("; ".join(errors))

    @staticmethod
    def _frame_issues(df: pd.DataFrame, *, lang: str = "ko") -> list[str]:
        """Degraded-frame notes (unlabelled): a split-only-adjusted fallback, a stale last bar."""
        out = []
        if df.attrs.get("adjusted") is False:
            src = df.attrs.get("source")
            head = (
                _tr(lang, "Yahoo 실패로 Nasdaq 대체", "Nasdaq used after Yahoo failed")
                if src == "nasdaq"
                else _tr(lang, f"{src}: 수정주가(adjclose) 없음", f"{src}: no adjusted close (adjclose)")
            )
            out.append(
                _tr(
                    lang,
                    f"{head} — 분할만 반영, 배당 미조정 종가 (장기 수익률 소폭 과소)",
                    f"{head} — split-adjusted only, closes not dividend-adjusted "
                    "(long-run returns slightly understated)",
                )
            )
        if len(df):
            last = df.index[-1].date()
            expected = last_completed_us_session()
            if int(np.busday_count(np.datetime64(last), np.datetime64(expected))) > 5:
                out.append(
                    _tr(
                        lang,
                        f"최근 봉 {last} (기대 {expected}) — 데이터 지연/거래정지 확인 필요",
                        f"Latest bar {last} (expected {expected}) — check for data delay/trading halt",
                    )
                )
        return out

    # ------------------------------------------------------------------ fundamentals

    def fundamentals(self, inst: Instrument) -> Fundamentals:
        sym = inst.yf_symbol or inst.ticker
        try:
            wrapped = self._info(sym)
        except Exception as exc:  # noqa: BLE001 - XBRL-only fallback, flagged (raises if impossible)
            log.warning("%s yfinance info failed: %s", inst.ticker, exc)
            return self._fundamentals_xbrl_only(inst, exc)
        info = wrapped["info"]
        anchor, quote = session_anchor_price(info)
        if anchor != quote:
            log.info(
                "%s: .info ratios priced at live quote %s re-priced to the last completed close %s",
                inst.ticker,
                quote,
                anchor,
            )
        fund = fundamentals_from_info(info, as_of=date.fromisoformat(wrapped["as_of"]), anchor=anchor)
        sources = ["yfinance"]
        missing = [
            f
            for f in (
                "revenue_growth_yoy",
                "earnings_growth_yoy",
                "operating_margin",
                "eps",
                "per",
            )
            if getattr(fund, f) is None
        ]
        if self._sec_ua():
            try:
                snap = xbrl_snapshot(self._companyfacts(inst.ticker), today_et())
                filled = merge_xbrl_snapshot(fund, snap, anchor, label=inst.ticker)
                sources.append(f"SEC XBRL 보완({', '.join(filled)})" if filled else "SEC XBRL 교차검증")
            except Exception as exc:  # noqa: BLE001 - optional cross-check
                log.warning("%s XBRL cross-check failed: %s", inst.ticker, exc)
                if missing:
                    self._note(
                        inst.ticker,
                        _tr(
                            self.settings,
                            f"펀더멘털 누락({', '.join(missing)}) — SEC XBRL 보완 실패: {type(exc).__name__}",
                            f"Fundamentals missing ({', '.join(missing)}) — SEC XBRL fill-in failed: "
                            f"{type(exc).__name__}",
                        ),
                    )
        elif missing:
            self._note(
                inst.ticker,
                _tr(
                    self.settings,
                    f"펀더멘털 누락({', '.join(missing)}) — SEC_USER_AGENT 미설정으로 SEC XBRL 보완 불가",
                    f"Fundamentals missing ({', '.join(missing)}) — SEC XBRL fill-in unavailable "
                    "(SEC_USER_AGENT not set)",
                ),
            )
        try:
            fund.sector_per = self._sector_per(inst.ticker, info.get("sector"))
        except Exception as exc:  # noqa: BLE001 - optional field, but its loss is reported
            self._note(
                inst.ticker,
                _tr(
                    self.settings,
                    f"업종 PER 산출 실패({type(exc).__name__}) — 업종 대비 밸류에이션 비교 생략",
                    f"Sector PER calculation failed ({type(exc).__name__}) — sector-relative valuation skipped",
                ),
            )
        self._fill_earnings_date(inst, fund, info)
        fund.source = " + ".join(sources)
        return fund

    def _fill_earnings_date(self, inst: Instrument, fund: Fundamentals, info: dict | None) -> None:
        """Set ``next_earnings_date`` / ``next_earnings_is_estimate`` (exchange-local date).

        Sources: the quote ``info`` (already fetched) and the Yahoo earnings-calendar page (cached per
        ET day) as a cross-check; see :func:`resolve_next_earnings`. A disagreement is logged and
        flagged as an estimate. When no date can be established because a source FAILED, a note says
        the pre-earnings entry rule cannot be applied (an unscheduled date is normal and not noted).
        """
        if str((info or {}).get("quoteType") or "").upper() == "ETF":
            return  # funds have no earnings releases
        tz = str((info or {}).get("exchangeTimezoneName") or ET)
        sym = inst.yf_symbol or inst.ticker
        calendar: dict[str, list[date]] | None = None
        cal_err = ""
        try:
            calendar = self._earnings_calendar(sym, tz)
        except Exception as exc:  # noqa: BLE001 - cross-check source; handled below
            cal_err = f"{type(exc).__name__}: {str(exc)[:80]}"
            log.warning("%s earnings calendar page failed: %s", inst.ticker, exc)
        nxt, est, detail = resolve_next_earnings(info, calendar, today_et(), tz)
        fund.next_earnings_date, fund.next_earnings_is_estimate = nxt, est
        if "DISAGREE" in detail:
            log.warning(
                "%s next earnings: sources disagree (%s) -> earliest %s, flagged estimate",
                inst.ticker,
                detail,
                nxt,
            )
        else:
            log.info("%s next earnings: %s (%s)", inst.ticker, nxt, detail)
        if nxt is None and cal_err:  # info had no usable date AND the cross-check source failed
            self._note(
                inst.ticker,
                _tr(
                    self.settings,
                    f"실적발표일: Yahoo 실적 캘린더 조회 실패({cal_err}) — 다음 실적발표일 미확인 "
                    "(실적 임박 진입 제한 미적용)",
                    f"Earnings date: Yahoo earnings calendar lookup failed ({cal_err}) — next earnings date "
                    "unknown (pre-earnings entry limit not applied)",
                ),
            )

    def _earnings_calendar(self, symbol: str, tz: str) -> dict[str, list[date]]:
        """Upcoming / reported earnings dates from Yahoo's earnings-calendar page (``get_earnings_dates``,
        exchange-local), cached per ET day. A page without a table (no coverage) gives empty lists."""

        def fetch() -> dict:
            def call() -> pd.DataFrame | None:
                _YAHOO_THROTTLE.wait()
                return yf.Ticker(symbol).get_earnings_dates(limit=4)

            df = with_backoff(call, tries=2, allow_empty=True, what=f"yfinance earnings dates {symbol}")
            parsed = parse_earnings_calendar(df, tz)
            return {k: [d.isoformat() for d in v] for k, v in parsed.items()}

        key = DiskCache.key("earnings_cal", _CACHE_VERSION, symbol, today_et().isoformat())
        payload = self._cached("us_info", key, self.settings.data.price_ttl_hours, fetch)
        return {k: [date.fromisoformat(d) for d in v] for k, v in payload.items()}

    def _fundamentals_xbrl_only(self, inst: Instrument, info_exc: BaseException) -> Fundamentals:
        if not self._sec_ua():
            raise RuntimeError(
                _tr(
                    self.settings,
                    f"Yahoo info 실패({type(info_exc).__name__}: {str(info_exc)[:80]}), "
                    f"SEC_USER_AGENT 미설정으로 XBRL 대체 불가",
                    f"Yahoo info failed ({type(info_exc).__name__}: {str(info_exc)[:80]}), "
                    "XBRL fallback unavailable (SEC_USER_AGENT not set)",
                )
            ) from info_exc
        today = today_et()
        snap = xbrl_snapshot(self._companyfacts(inst.ticker), today)
        px = self._cached_prices(
            inst.yf_symbol or inst.ticker,
            today - pd.Timedelta(30, "D").to_pytimedelta(),
            ("stocks", "etf"),
        )
        fund = Fundamentals(as_of=today, source="SEC XBRL (Yahoo info 실패)")
        merge_xbrl_snapshot(fund, snap, float(px["close"].iloc[-1]), label=inst.ticker)
        self._note(
            inst.ticker,
            _tr(
                self.settings,
                f"펀더멘털: Yahoo info 실패({type(info_exc).__name__}) → SEC XBRL 대체 "
                "(애널리스트 목표가·PBR·ROE 없음)",
                f"Fundamentals: Yahoo info failed ({type(info_exc).__name__}) → SEC XBRL fallback "
                "(no analyst target · PBR · ROE)",
            ),
        )
        self._fill_earnings_date(inst, fund, None)  # calendar page only -> an estimate
        return fund

    def _sector_per(self, ticker: str, sector: str | None) -> float | None:
        """Median trailing P/E of static-universe peers in the same Yahoo sector (the ticker itself
        excluded). Peer ``.info`` is disk-cached and shared with the panel loader."""
        if not sector:
            return None
        peers = [t for t, s in _STATIC_SECTOR_YAHOO.items() if s == sector and t != ticker][: self.SECTOR_PEERS_MAX]
        if len(peers) < self.SECTOR_PEERS_MIN:
            return None

        def pe(t: str) -> float | None:
            try:
                info = self._info(t)["info"]
                v = _num(info.get("trailingPE"))
                anchor, quote = session_anchor_price(info)  # live intraday P/E -> completed close
                eps = _first(_num(info.get("trailingEps")), _num(info.get("epsTrailingTwelveMonths")))
                if v is not None and anchor and eps and eps > 0:
                    v = anchor / eps  # = trailingPE re-priced exactly (see fundamentals_from_info)
                elif v is not None and anchor and quote:
                    v *= anchor / quote
            except Exception as exc:  # noqa: BLE001 - a missing peer just shrinks the sample
                log.info("sector peer %s info failed: %s", t, exc)
                return None
            return v if v is not None and 0 < v < 300 else None

        with ThreadPoolExecutor(max_workers=4) as pool:
            pes = [v for v in pool.map(pe, peers) if v is not None]
        if len(pes) < self.SECTOR_PEERS_MIN:
            log.info("%s sector P/E skipped: only %d/%d peers", ticker, len(pes), len(peers))
            return None
        return float(np.median(pes))

    # ------------------------------------------------------------------ valuation history

    def valuation_history(self, inst: Instrument, start: date) -> pd.DataFrame | None:
        """Point-in-time trailing P/E history (column ``per``; no ``pbr``).

        Uses the split-adjusted raw close (not dividend-adjusted) and XBRL diluted-EPS TTM,
        split-adjusted per fact and available from ``first_filed`` + 1 day. Needs SEC_USER_AGENT
        (RuntimeError otherwise).
        """
        self._require_sec_ua()
        start = _as_date(start)
        facts = self._companyfacts(inst.ticker)
        raw = self._raw_close_and_splits(inst.yf_symbol or inst.ticker, start)
        vh = trailing_pe_history(facts, raw["close"], raw["splits"])
        if vh is not None:
            vh = vh[vh.index >= pd.Timestamp(start)]
        if vh is None or vh.empty:
            self._note(
                inst.ticker,
                _tr(
                    self.settings,
                    "밸류에이션 히스토리: XBRL 희석 EPS(TTM, USD/주) 없음·부족 — PER 히스토리 생략",
                    "Valuation history: XBRL diluted EPS (TTM, USD/share) missing or insufficient — PER history "
                    "skipped",
                ),
            )
            return None
        vh.attrs.update(
            source=f"SEC XBRL EPS TTM + {raw['source']} close",
            metric="trailing P/E",
            warnings=[],
        )
        return vh

    def _raw_close_and_splits(self, symbol: str, start: date) -> dict:
        """Split-adjusted (NOT dividend-adjusted) close since ``start`` plus split events since
        ``start`` - 3y (EPS filed before a later split must be restated)."""
        fetch_start = start - pd.Timedelta(3 * 366, "D").to_pytimedelta()

        def fetch() -> dict:
            try:
                raw, reg_end = _yf_history(symbol, fetch_start, auto_adjust=False, actions=True)
                splits = raw["Stock Splits"] if "Stock Splits" in raw.columns else pd.Series(dtype="float64")
                splits = splits[splits.fillna(0) > 0]
                idx = pd.DatetimeIndex(splits.index)
                splits.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
                close = frame_from_yf_history(raw, reg_end)["close"]
                return {
                    "close": close,
                    "splits": splits.astype("float64"),
                    "source": "yfinance",
                }
            except Exception as exc:  # noqa: BLE001 - same data via the raw chart
                log.warning("%s raw close via yfinance failed (%s) -> Yahoo chart", symbol, exc)
            res = fetch_yahoo_chart(symbol, fetch_start)
            return {
                "close": parse_yahoo_chart(res, adjust=False)["close"],
                "splits": chart_splits(res),
                "source": "yahoo_chart",
            }

        key = DiskCache.key(
            "rawpx",
            _CACHE_VERSION,
            symbol,
            start.isoformat(),
            last_completed_us_session().isoformat(),
        )
        out = self._cached("us_prices", key, self.settings.data.price_ttl_hours, fetch)
        return {**out, "close": out["close"][out["close"].index >= pd.Timestamp(start)]}

    # ------------------------------------------------------------------ news

    def news(self, inst: Instrument, since: datetime, limit: int) -> list[NewsItem]:
        sym = inst.yf_symbol or inst.ticker
        since_utc = _as_utc_datetime(since)
        days = max(
            1,
            math.ceil((datetime.now(UTC) - since_utc).total_seconds() / 86400),
        )
        key = DiskCache.key("news", _CACHE_VERSION, sym, inst.name, days)
        payload = self._cached(
            "us_news",
            key,
            self.settings.data.text_ttl_hours,
            lambda: self._fetch_news_rows(sym, inst.name, days),
        )
        for n in payload["notes"]:
            self._note(inst.ticker, n)
        return merge_news(payload["rows"], since_utc, limit, inst.ticker, inst.name)

    def _fetch_news_rows(self, symbol: str, company: str, days: int) -> dict:
        rows: list[dict] = []
        errors: dict[str, str] = {}
        yf_rows: list[dict] = []
        try:
            yf_rows = parse_yf_news(self._fetch_yf_news(symbol))
        except Exception as exc:  # noqa: BLE001
            errors["yfinance"] = f"{type(exc).__name__}: {exc}"
        if not yf_rows:
            # yfinance also returns [] when Yahoo answers HTTP 502 -> same pool via RSS
            log.info(
                "%s yfinance news empty/failed (%s) -> Yahoo RSS",
                symbol,
                errors.get("yfinance", "empty"),
            )
            try:
                rows += parse_rss(self._fetch_yahoo_rss(symbol), "yahoo_rss")
            except Exception as exc:  # noqa: BLE001
                errors["yahoo_rss"] = f"{type(exc).__name__}: {exc}"
        rows += yf_rows
        try:
            rows += parse_rss(
                self._fetch_google_news(google_news_query(symbol, company, days)),
                "google",
            )
        except Exception as exc:  # noqa: BLE001
            errors["google"] = f"{type(exc).__name__}: {exc}"
        for src, err in errors.items():
            log.warning("%s news source %s failed: %s", symbol, src, err)
        if not rows and errors:
            raise RuntimeError(
                _tr(self.settings, "모든 뉴스 소스 실패: ", "All news sources failed: ")
                + "; ".join(f"{k}: {v[:60]}" for k, v in errors.items())
            )
        notes = []
        yahoo_down = "yahoo_rss" in errors and not yf_rows
        if "google" in errors or yahoo_down:
            down = [
                n
                for n, bad in (
                    ("Google News", "google" in errors),
                    ("Yahoo", yahoo_down),
                )
                if bad
            ]
            notes.append(
                _tr(
                    self.settings,
                    f"뉴스: {'/'.join(down)} 수집 실패 — 일부 소스만 반영",
                    f"News: {'/'.join(down)} fetch failed — partial sources only",
                )
            )
        return {"rows": rows, "notes": notes}

    def _fetch_yf_news(self, symbol: str) -> list[dict]:
        def call() -> list[dict]:
            _YAHOO_THROTTLE.wait()
            return yf.Ticker(symbol).get_news(count=40, tab="news")  # fresh Ticker: news is cached per object

        return with_backoff(call, tries=2, allow_empty=True, what=f"yfinance news {symbol}") or []

    def _fetch_yahoo_rss(self, symbol: str) -> bytes:
        # Yahoo 429s plain requests with a Chrome UA
        from curl_cffi import requests as creq

        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

        def call() -> bytes:
            _YAHOO_THROTTLE.wait()
            r = creq.get(url, impersonate="chrome", timeout=20)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:80]}")
            return r.content

        return with_backoff(call, tries=2, what=f"yahoo rss {symbol}")

    def _fetch_google_news(self, query: str) -> bytes:
        r = self._web.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
        )
        r.raise_for_status()
        return r.content

    # ------------------------------------------------------------------ SEC plumbing

    def _sec_ua(self) -> str:
        return (self.settings.data.sec_user_agent or "").strip()

    def _require_sec_ua(self) -> str:
        ua = self._sec_ua()
        if not ua:
            raise RuntimeError(
                _tr(
                    self.settings,
                    'SEC_USER_AGENT 미설정 — .env에 SEC_USER_AGENT="Org Name contact@domain" '
                    "형식으로 설정하세요 (SEC EDGAR 필수)",
                    'SEC_USER_AGENT not set — set SEC_USER_AGENT="Org Name contact@domain" in .env '
                    "(required by SEC EDGAR)",
                )
            )
        if "@" not in ua:
            log.warning("SEC_USER_AGENT has no contact e-mail; SEC may answer 403")
        return ua

    def _sec(self) -> SecSession:
        ua = self._require_sec_ua()
        with self._sec_lock:
            if self._sec_session is None or self._sec_session.headers.get("User-Agent") != ua:
                self._sec_session = SecSession(ua)
            return self._sec_session

    def _sec_get(self, url: str) -> bytes:
        """One SEC request (global 8 req/s budget, retries on 429/5xx). 403 -> a clear UA hint."""
        r = self._sec().get(url)
        if r.status_code == 403:
            raise RuntimeError(
                _tr(
                    self.settings,
                    f"SEC EDGAR 403 (User-Agent 거부) — SEC_USER_AGENT 형식 확인: {url}",
                    f"SEC EDGAR 403 (User-Agent rejected) — check the SEC_USER_AGENT format: {url}",
                )
            )
        r.raise_for_status()
        return r.content

    def _sec_json(self, url: str) -> dict:
        return json.loads(self._sec_get(url))

    def _sec_text(self, url: str) -> str:
        """Cleaned text of an archived EDGAR document (immutable -> cached for 30 days)."""
        key = DiskCache.key("sec_text", _CACHE_VERSION, url)
        return self._cached(
            "us_sec_text",
            key,
            _SEC_TEXT_TTL_HOURS,
            lambda: edgar_html_to_text(self._sec_get(url)),
        )

    def _filing_index(self, url: str) -> pd.DataFrame:
        key = DiskCache.key("sec_index", _CACHE_VERSION, url)
        recs = self._cached(
            "us_sec_text",
            key,
            _SEC_TEXT_TTL_HOURS,
            lambda: parse_filing_index(self._sec_get(url)).to_dict("records"),
        )
        return pd.DataFrame(recs, columns=["seq", "description", "document", "type", "size", "url"])

    def _cik(self, ticker: str) -> tuple[str, str]:
        key = DiskCache.key("company_tickers", _CACHE_VERSION)
        mapping = self._cached(
            "us_sec",
            key,
            _DAY_TTL_HOURS,
            lambda: parse_company_tickers(self._sec_json("https://www.sec.gov/files/company_tickers.json")),
        )
        t = normalise_us_ticker(ticker)
        if t not in mapping:
            raise RuntimeError(
                _tr(
                    self.settings,
                    f"{t}: SEC company_tickers.json에 없음 (SEC 등록 종목이 아니거나 티커 변경)",
                    f"{t}: not in SEC company_tickers.json (not SEC-registered or ticker changed)",
                )
            )
        return mapping[t]

    def _submissions(self, cik: str) -> tuple[dict, pd.DataFrame]:
        key = DiskCache.key("submissions", _CACHE_VERSION, cik)
        j = self._cached(
            "us_sec",
            key,
            self.settings.data.text_ttl_hours,
            lambda: self._sec_json(f"https://data.sec.gov/submissions/CIK{str(int(cik)).zfill(10)}.json"),
        )
        return parse_submissions(j)

    def _companyfacts(self, ticker: str) -> dict:
        self._require_sec_ua()
        cik, _ = self._cik(ticker)
        key = DiskCache.key("companyfacts", _CACHE_VERSION, cik)
        return self._cached(
            "us_sec",
            key,
            _DAY_TTL_HOURS,
            lambda: trim_companyfacts(self._sec_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")),
        )

    # ------------------------------------------------------------------ filings

    def filings(self, inst: Instrument, since: datetime, limit: int) -> list[FilingItem]:
        """SEC EDGAR + Motley Fool text for the LLM, newest first.

        Selection priority for ``limit``: the latest 8-K Item 2.02 earnings release (EX-99.1 +
        EX-99.2), the latest earnings-call transcript, adverse 8-Ks (severity <= -2), the 10-Q/10-K
        risk factors and MD&A digests, then other 8-Ks. Each text is at most
        ``settings.data.max_filing_chars``, keeping the head and tail. One exception to the ``since``
        window: when the in-window 10-Q says its risk factors have no material changes, the
        risk-factor item comes from the older 10-K it refers to, and the title says so.
        """
        self._require_sec_ua()
        since_utc = _as_utc_datetime(since)
        max_chars = self.settings.data.max_filing_chars
        key = DiskCache.key(
            "filings",
            _CACHE_VERSION,
            inst.ticker,
            since_utc.date().isoformat(),
            limit,
            max_chars,
        )
        payload = self._cached(
            "us_filings",
            key,
            self.settings.data.text_ttl_hours,
            lambda: self._build_filings(inst, since_utc, limit),
        )
        for n in payload["notes"]:
            self._note(inst.ticker, n)
        return [FilingItem.model_validate(d) for d in payload["items"]]

    def _build_filings(self, inst: Instrument, since: datetime, limit: int) -> dict:
        notes: list[str] = []
        max_chars = self.settings.data.max_filing_chars
        cik, _ = self._cik(inst.ticker)
        _, df = self._submissions(cik)
        since_ts = pd.Timestamp(since)
        lookback = max(1, (datetime.now(UTC) - since).days)
        win = df[df["filed_utc"] >= since_ts]
        plan: list[_PlanStep] = []

        eightk = win[win["form"].isin(EIGHT_K_FORMS)]
        is_er = eightk["items"].str.contains(r"\b2\.02\b", regex=True)
        er_row = eightk[is_er].iloc[0] if bool(is_er.any()) else None
        if er_row is not None:
            plan.append(
                _PlanStep(
                    0,
                    0,
                    _tr(self.settings, "실적 보도자료(8-K 2.02)", "Earnings release (8-K 2.02)"),
                    lambda r=er_row: self._earnings_release_item(r, max_chars, notes),
                )
            )
        else:
            notes.append(
                _tr(
                    self.settings,
                    f"최근 {lookback}일 내 실적발표 8-K(Item 2.02) 없음",
                    f"No earnings 8-K (Item 2.02) in the last {lookback} days",
                )
            )
        er_date = er_row["filingDate"].date() if er_row is not None else None
        plan.append(
            _PlanStep(
                1,
                0,
                _tr(self.settings, "컨퍼런스콜 녹취록", "Earnings call transcript"),
                lambda: self._transcript_item(inst, since_ts, er_date, max_chars, notes),
            )
        )

        for i, (_, row) in enumerate(eightk.iterrows()):
            if er_row is not None and row["accessionNumber"] == er_row["accessionNumber"]:
                continue
            sev = item_severity(parse_8k_items(row["items"]))
            plan.append(
                _PlanStep(
                    2 if sev <= -2 else 5,
                    i,
                    f"8-K {row['filingDate'].date()}",
                    lambda r=row: self._other_8k_item(r, max_chars),
                )
            )

        periodic = df[df["form"].isin(PERIODIC_FORMS)]
        if not periodic.empty and periodic.iloc[0]["filed_utc"] >= since_ts:
            memo: dict[str, dict] = {}

            def sections() -> dict:
                if "v" not in memo:
                    memo["v"] = self._periodic_sections(df, periodic.iloc[0])
                return memo["v"]

            plan.append(
                _PlanStep(
                    3,
                    0,
                    _tr(self.settings, "위험요인(Item 1A)", "Risk factors (Item 1A)"),
                    lambda: self._risk_item(sections(), max_chars, notes),
                )
            )
            plan.append(_PlanStep(4, 0, "MD&A", lambda: self._mdna_item(sections(), max_chars, notes)))
        else:
            notes.append(
                _tr(self.settings, f"최근 {lookback}일 내 10-Q/10-K 없음", f"No 10-Q/10-K in the last {lookback} days")
            )

        items: list[FilingItem] = []
        for step in sorted(plan, key=lambda p: (p.priority, p.order)):
            if len(items) >= limit:
                break
            try:
                item = step.build()
            except Exception as exc:  # noqa: BLE001 - one bad document must not drop the rest
                log.warning("%s filings: %s failed: %s", inst.ticker, step.label, exc)
                notes.append(
                    _tr(self.settings, f"{step.label} 수집 실패", f"{step.label} fetch failed")
                    + f": {type(exc).__name__}: {str(exc)[:80]}"
                )
                continue
            if item is not None and item.text.strip():
                items.append(item)
        items.sort(key=lambda f: f.filed_at, reverse=True)
        return {"items": [it.model_dump(mode="json") for it in items], "notes": notes}

    def _earnings_release_item(self, row: pd.Series, max_chars: int, notes: list[str]) -> FilingItem:
        exhibits = exhibits_99(self._filing_index(row["index_url"]))[:2]
        parts = [(str(x["type"]).upper(), self._sec_text(x["url"])) for x in exhibits]
        url = exhibits[0]["url"] if exhibits else row["url"]
        if not parts:
            notes.append(
                _tr(
                    self.settings,
                    "실적 8-K에 EX-99 보도자료 없음 — 8-K 본문으로 대체",
                    "No EX-99 press release in the earnings 8-K — using the 8-K body",
                )
            )
            parts = [("8-K", body_after_items(self._sec_text(row["url"]), max_chars))]
        text = compose_sections(parts, max_chars, [0.62, 0.38][: len(parts)])
        types = "+".join(label for label, _ in parts)
        return FilingItem(
            id=text_id("SEC", row["accessionNumber"], "earnings"),
            title=f"실적 발표 8-K Item 2.02 ({types}) {row['filingDate'].date()}",
            form_type=row["form"],
            filed_at=row["filed_utc"].to_pydatetime(),
            url=url,
            text=text,
            is_earnings=True,
            source="SEC EDGAR",
        )

    def _other_8k_item(self, row: pd.Series, max_chars: int) -> FilingItem:
        items = parse_8k_items(row["items"])
        body = body_after_items(self._sec_text(row["url"]), max_chars)
        parts = [("8-K", body)]
        if len(body) < 1500:  # "see Exhibit 99.1" style body -> attach the press release
            exhibits = exhibits_99(self._filing_index(row["index_url"]))[:1]
            parts += [(str(x["type"]).upper(), self._sec_text(x["url"])) for x in exhibits]
        desc = ", ".join(f"{i} {ITEM_8K_KO.get(i, '')}".strip() for i in items if i != "9.01") or "기타"
        return FilingItem(
            id=text_id("SEC", row["accessionNumber"], "8-K"),
            title=f"8-K ({desc}) {row['filingDate'].date()}",
            form_type=row["form"],
            filed_at=row["filed_utc"].to_pydatetime(),
            url=row["url"],
            text=compose_sections(parts, max_chars, [0.5, 0.5][: len(parts)]),
            is_earnings="2.02" in items,
            source="SEC EDGAR",
        )

    def _periodic_sections(self, df: pd.DataFrame, row: pd.Series) -> dict:
        """Risk factors + MD&A of the latest 10-Q/10-K. When the 10-Q risk factors only say 'no material
        changes' (< 2,000 chars), the latest 10-K risk factors are used and the 10-Q text is kept as
        the update."""
        form = row["form"]
        text = self._sec_text(row["url"])
        rf = extract_section(text, form, "risk_factors", None)
        out = {
            "row": row,
            "mdna": extract_section(text, form, "mdna", None),
            "rf": rf,
            "rf_update": "",
            "rf_row": row,
            "rf_basis": "",
        }
        if form.startswith("10-Q") and len(rf) < 2000:
            k = df[df["form"] == "10-K"]
            if not k.empty:
                k_rf = extract_section(self._sec_text(k.iloc[0]["url"]), "10-K", "risk_factors", None)
                if k_rf:
                    # an empty 10-Q section is NOT a statement of "no material changes" - say which it was
                    basis = (
                        "absent"
                        if not rf
                        else "no_change"
                        if re.search(r"no\s+material\s+changes?", rf, re.I)
                        else "update"
                    )
                    out.update(rf=k_rf, rf_update=rf, rf_row=k.iloc[0], rf_basis=basis)
        return out

    def _risk_item(self, sec: dict, max_chars: int, notes: list[str]) -> FilingItem | None:
        row, src = sec["row"], sec["rf_row"]
        if not sec["rf"]:
            notes.append(
                _tr(
                    self.settings,
                    f"{row['form']} 위험요인(Item 1A) 추출 실패",
                    f"{row['form']} risk factors (Item 1A) extraction failed",
                )
            )
            return None
        heads = risk_factor_headlines(sec["rf"], max_items=40)
        head_txt = truncate_head_tail("\n".join(f"- {h}" for h in heads), int(max_chars * 0.3))
        upd_txt = truncate_head_tail(sec["rf_update"], int(max_chars * 0.15)) if sec["rf_update"] else ""
        budget = max(1000, max_chars - len(head_txt) - len(upd_txt) - 120)
        digest = select_paragraphs(sec["rf"], budget) or truncate_head_tail(sec["rf"], budget)
        text = compose_sections(
            [
                (f"{row['form']} 위험요인 업데이트", upd_txt),
                ("위험요인 헤드라인", head_txt),
                ("위험요인 본문 발췌", digest),
            ],
            max_chars,
        )
        title = f"{src['form']} 위험요인 (Item 1A) {src['filingDate'].date()}"
        if src["accessionNumber"] != row["accessionNumber"]:
            basis = sec.get("rf_basis")
            if basis == "absent":
                title += f" — {row['form']} {row['filingDate'].date()}에 위험요인 항목 없음(최신 10-K 기준)"
                notes.append(
                    _tr(
                        self.settings,
                        f"{row['form']} {row['filingDate'].date()} 위험요인(Item 1A) 미추출 — "
                        f"최신 10-K {src['filingDate'].date()} 위험요인으로 대체",
                        f"{row['form']} {row['filingDate'].date()} risk factors (Item 1A) not extracted — "
                        f"replaced by the latest 10-K {src['filingDate'].date()} risk factors",
                    )
                )
            elif basis == "no_change":
                title += f" — {row['form']} {row['filingDate'].date()} 기준 중요 변경 없음"
            else:  # a short update or a bare cross-reference: the 10-Q text is included verbatim
                title += f" + {row['form']} {row['filingDate'].date()} 위험요인 항목"
        return FilingItem(
            id=text_id("SEC", src["accessionNumber"], "risk_factors"),
            title=title,
            form_type=src["form"],
            filed_at=src["filed_utc"].to_pydatetime(),
            url=src["url"],
            text=text,
            is_earnings=False,
            source="SEC EDGAR",
        )

    def _mdna_item(self, sec: dict, max_chars: int, notes: list[str]) -> FilingItem | None:
        row = sec["row"]
        if not sec["mdna"]:
            notes.append(_tr(self.settings, f"{row['form']} MD&A 추출 실패", f"{row['form']} MD&A extraction failed"))
            return None
        budget = max_chars - 40
        digest = select_paragraphs(sec["mdna"], budget) or truncate_head_tail(sec["mdna"], budget)
        return FilingItem(
            id=text_id("SEC", row["accessionNumber"], "mdna"),
            title=f"{row['form']} 경영진 분석(MD&A) {row['filingDate'].date()}",
            form_type=row["form"],
            filed_at=row["filed_utc"].to_pydatetime(),
            url=row["url"],
            text=compose_sections([("MD&A 발췌", digest)], max_chars),
            is_earnings=False,
            source="SEC EDGAR",
        )

    # ------------------------------------------------------------------ Motley Fool

    def _fool_page(self, url: str) -> str | None:
        """GET a fool.com page (>= 1 s spacing). None on 404/redirect-to-error."""
        _FOOL_THROTTLE.wait()
        r = self._web.get(
            url,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=20,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.text

    def _fool_links(self, inst: Instrument) -> list[dict]:
        fool_t = inst.ticker.replace("-", ".").lower()
        if not _FOOL_TICKER_RE.fullmatch(fool_t):
            return []
        first = "nyse" if "NYSE" in (inst.exchange or "").upper() else "nasdaq"
        for ex in (first, "nasdaq" if first == "nyse" else "nyse"):
            page = self._fool_page(f"https://www.fool.com/quote/{ex}/{fool_t}/")
            if page and "call-transcript" in page:
                links = parse_fool_links(page, inst.ticker)
                if links:
                    return links
        return []

    def _transcript_item(
        self,
        inst: Instrument,
        since: pd.Timestamp,
        er_date: date | None,
        max_chars: int,
        notes: list[str],
    ) -> FilingItem | None:
        try:
            links = self._fool_links(inst)
        except Exception as exc:  # noqa: BLE001 - scraping a public site; press release still covers it
            notes.append(
                _tr(
                    self.settings,
                    f"컨퍼런스콜 녹취록 수집 실패({type(exc).__name__}) — 실적 보도자료로 대체",
                    f"Earnings call transcript fetch failed ({type(exc).__name__}) — using the earnings press release",
                )
            )
            return None
        if not links:
            notes.append(
                _tr(
                    self.settings,
                    "Motley Fool 컨퍼런스콜 녹취록 없음 — 실적 보도자료로 대체",
                    "No Motley Fool earnings call transcript — using the earnings press release",
                )
            )
            return None
        top = links[0]
        published = pd.Timestamp(top["published"]).tz_localize(ET).tz_convert("UTC")
        # Item 2.02 may be furnished up to 4 business days after the release/call, so a transcript
        # published a few days BEFORE the 8-K is still the current quarter's (quarters are ~90 days apart).
        fresh_from = (er_date - pd.Timedelta(_TRANSCRIPT_ER_SLACK_DAYS, "D").to_pytimedelta()) if er_date else None
        if fresh_from is not None and published.tz_convert(ET).date() < fresh_from:
            notes.append(
                _tr(
                    self.settings,
                    f"최신 실적(8-K {er_date}) 컨퍼런스콜 녹취록 미게시(최신 {top['published']}) — "
                    "실적 보도자료로 대체",
                    f"Earnings call transcript for the latest results (8-K {er_date}) not yet published (latest "
                    f"{top['published']}) — using the earnings press release",
                )
            )
            return None
        if published < since:
            notes.append(
                _tr(
                    self.settings,
                    f"기간 내 컨퍼런스콜 녹취록 없음(최신 {top['published']})",
                    f"No earnings call transcript in the window (latest {top['published']})",
                )
            )
            return None
        page = self._fool_page(top["url"])
        if not page:
            notes.append(
                _tr(
                    self.settings,
                    "컨퍼런스콜 녹취록 페이지 없음(404) — 실적 보도자료로 대체",
                    "Earnings call transcript page not found (404) — using the earnings press release",
                )
            )
            return None
        t = parse_fool_transcript(page, top["url"])
        if not t["qa"]:
            log.warning("%s transcript: no Q&A split found (%s)", inst.ticker, top["url"])
        return FilingItem(
            id=text_id("Motley Fool", top["url"]),
            title=f"실적 컨퍼런스콜 녹취록 FY{top['fiscal_year']} Q{top['fiscal_quarter']} (Motley Fool)",
            form_type="Earnings Call Transcript",
            filed_at=published.to_pydatetime(),
            url=top["url"],
            text=transcript_digest(t, max_chars),
            is_earnings=True,
            source="Motley Fool",
        )

    # ------------------------------------------------------------------ FX

    def fx_rate(self, from_ccy: str, to_ccy: str) -> float:
        """USD<->KRW under the definition shared with ``KRProvider`` (:func:`cached_usdkrw`): the last
        completed Yahoo KRW=X bar aligned to the last completed US session; Naver fallback flagged."""
        return usdkrw_fx(from_ccy, to_ccy, self._usdkrw, lang=lang_of(self.settings))

    def _usdkrw(self) -> float:
        payload = cached_usdkrw(
            self.cache,
            self.settings.data.price_ttl_hours,
            lambda: self._fetch_krw_series_yf(),
            lambda: self._fetch_usdkrw_naver(),
            lang=lang_of(self.settings),
        )
        note = usdkrw_fallback_note(payload, lang=lang_of(self.settings))
        if note:
            self._note(_GLOBAL_NOTES, note)
        return float(payload["rate"])

    def _fetch_krw_series_yf(self) -> tuple[pd.Series, str]:
        return fetch_krw_series_yf(lang=lang_of(self.settings))

    def _fetch_usdkrw_naver(self) -> pd.Series:
        r = self._web.get(NAVER_FX_PRICES, params=dict(NAVER_FX_PARAMS))
        r.raise_for_status()
        return parse_naver_fx(r.json())

    # ------------------------------------------------------------------ universe

    def universe(self, n: int) -> list[str]:
        """First ``n`` names of a static, sector-interleaved list of 107 US large caps (11 GICS sectors,
        no ETFs). The same ``n`` always gives the same list."""
        names = interleaved_universe()
        if n > len(names):
            log.warning("US universe has %d names < requested %d", len(names), n)
        return names[: max(0, n)]


def _as_date(value: date | datetime) -> date:
    return value.date() if isinstance(value, datetime) else value


__all__ = [
    "USProvider",
    "UnsupportedInstrumentError",
    "SecSession",
    "US_UNIVERSE",
    "GICS_TO_YAHOO",
    "normalise_us_ticker",
    "last_completed_us_session",
    "drop_unfinished_bar",
    "frame_from_yf_history",
    "parse_yahoo_chart",
    "chart_splits",
    "parse_nasdaq_rows",
    "missing_sessions",
    "chart_event_dates",
    "scale_missing_bars",
    "apply_repaired_bars",
    "fundamentals_from_info",
    "epoch_local_date",
    "parse_earnings_calendar",
    "resolve_next_earnings",
    "recommendation_label",
    "dividend_yield_from_info",
    "instrument_from_info",
    "edgar_html_to_text",
    "extract_section",
    "truncate_head_tail",
    "risk_factor_headlines",
    "select_paragraphs",
    "body_after_items",
    "compose_sections",
    "parse_company_tickers",
    "parse_submissions",
    "parse_filing_index",
    "exhibits_99",
    "quarterly_series",
    "ttm_series",
    "xbrl_snapshot",
    "merge_xbrl_snapshot",
    "trailing_pe_history",
    "split_factor_after",
    "parse_fool_links",
    "parse_fool_transcript",
    "transcript_digest",
    "parse_yf_news",
    "parse_rss",
    "merge_news",
    "NewsRelevance",
    "self_published_matcher",
    "clean_company_name",
    "google_news_query",
    "pick_completed_fx",
    "parse_naver_fx",
    "UnsupportedCurrencyPairError",
    "fetch_krw_series_yf",
    "usdkrw_payload",
    "usdkrw_cache_key",
    "cached_usdkrw",
    "usdkrw_fallback_note",
    "usdkrw_fx",
    "cache_get_or_fetch",
    "interleaved_universe",
    "with_backoff",
]
