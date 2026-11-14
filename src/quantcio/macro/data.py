"""Daily macro time series (global / US / Korea) from keyless sources, cached.

Live-verified on 2026-09-25 from this machine:

- **Yahoo** through :meth:`quantcio.data.us.USProvider.benchmark`, the project's verified US price chain:
  yfinance ``history(auto_adjust=True)`` with ``hide_exceptions=False``, then the raw Yahoo v8 chart. The
  running session's partial bar is dropped in the quote's own exchange timezone; on 2026-09-25 (07:20 ET)
  the live 09-25 bars of CL=F, GC=F, HG=F, DX-Y.NYB, ^VIX and KRW=X were removed. All 17 symbols
  answered with history from 2012: ACWI, EEM, SPY, EWY, HYG, LQD, TLT, HG=F, GC=F, CL=F, DX-Y.NYB, ^VIX,
  ^TNX, ^FVX, ^IRX, KRW=X and ^KS11. Yahoo yields (^TNX/^FVX/^IRX) are in **percent**, e.g. 5.162 = 5.162%,
  with no x10 scaling from 2014 to 2026. ^IRX can be negative (min -0.105).
- **KOSPI** through :meth:`quantcio.data.kr.KRProvider.benchmark`: Naver fchart with at most 3,000 bars
  (back to 2014-07-04), falling back to yfinance ``^KS11``, which has missing days.
- **Korean government bond yields** from the keyless Naver bond API
  ``api.stock.naver.com/marketindex/bond/{KR3YT=RR|KR10YT=RR}/prices``. It serves at most 60 rows per page
  and only about 3 years: the last page (13) ends at 2023-09-26.

Probed and NOT usable:

- FRED is blocked here, and the Bank of Korea ECOS API needs a key.
- ``finance.naver.com/marketindex/interestDailyQuote.naver`` (CD 91일, 국고채) answers HTTP 410
  "이 페이지는 더 이상 제공되지 않습니다".
- ``m.stock.naver.com/front-api/marketIndex/prices`` with ``category=domesticInterest`` returns an empty
  result for IRR_CD91, IRR_GOVT03Y and IRR_CALL, so there is no CD 91 series.
- The Naver ``chart`` endpoints for bonds return 400/404.
- FinanceDataReader ``KR3YT=RR``/``KR10YT=RR`` is routed to Yahoo, which answers 404.

Failures are never silent. A series that cannot be loaded is missing from the result and recorded in
:attr:`MacroSeries.failures` (key -> error), in :attr:`MacroSeries.notes` (Korean) and at WARNING.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from datetime import time as dtime
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from quantcio.config import Settings
from quantcio.data.cache import DiskCache
from quantcio.data.http import PoliteSession
from quantcio.models import Instrument, Market

log = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")
CACHE_NAMESPACE = "macro_series"
_CACHE_VERSION = 2

NAVER_BOND_PRICES = "https://api.stock.naver.com/marketindex/bond/{code}/prices"
NAVER_BOND_PAGE_SIZE = 60  # server maximum ("pageSize: must be less than or equal to 60")
NAVER_BOND_MAX_PAGES = 40  # safety stop (history ended on page 13 on 2026-09-25)
#: A Naver bond row dated today (KST) is the live intraday quote until the close is final (~15:50-16:20 KST).
NAVER_BOND_FINAL_KST = dtime(17, 0)
#: Naver fchart serves at most 3000 bars; asking for more only produces a truncation note.
KOSPI_MAX_YEARS = 12.0  # 3000 KRX sessions reach back ~12.2 years (2014-07-04 on 2026-09-25)

SourceKind = Literal["yahoo", "kr_benchmark", "naver_bond"]
ValueKind = Literal["price", "yield", "level"]


@dataclass(frozen=True)
class SeriesSpec:
    key: str  # canonical key used by the regime module
    source: SourceKind
    symbol: str  # Yahoo symbol / KR benchmark id / Naver reuters code
    label: str  # Korean display label
    kind: ValueKind  # price/level: must be > 0 (dropped otherwise); yield: percent, may be <= 0


SERIES: tuple[SeriesSpec, ...] = (
    SeriesSpec("acwi", "yahoo", "ACWI", "MSCI ACWI ETF", "price"),
    SeriesSpec("eem", "yahoo", "EEM", "MSCI 신흥국 ETF", "price"),
    SeriesSpec("spy", "yahoo", "SPY", "S&P 500 ETF(SPY)", "price"),
    SeriesSpec("ewy", "yahoo", "EWY", "MSCI 한국 ETF(EWY, 달러)", "price"),
    SeriesSpec("hyg", "yahoo", "HYG", "미 하이일드 회사채 ETF", "price"),
    SeriesSpec("lqd", "yahoo", "LQD", "미 투자등급 회사채 ETF", "price"),
    SeriesSpec("tlt", "yahoo", "TLT", "미 장기국채 ETF", "price"),
    SeriesSpec("copper", "yahoo", "HG=F", "구리 선물(연결)", "price"),
    SeriesSpec("gold", "yahoo", "GC=F", "금 선물(연결)", "price"),
    SeriesSpec("wti", "yahoo", "CL=F", "WTI 원유 선물(연결)", "price"),
    SeriesSpec("dxy", "yahoo", "DX-Y.NYB", "달러인덱스(DXY)", "price"),
    SeriesSpec("vix", "yahoo", "^VIX", "VIX", "level"),
    SeriesSpec("vix3m", "yahoo", "^VIX3M", "VIX 3개월(기간 구조)", "level"),
    SeriesSpec("us10y", "yahoo", "^TNX", "미 국채 10년(%)", "yield"),
    SeriesSpec("us5y", "yahoo", "^FVX", "미 국채 5년(%)", "yield"),
    SeriesSpec("us3m", "yahoo", "^IRX", "미 T-bill 3개월(%)", "yield"),
    SeriesSpec("usdkrw", "yahoo", "KRW=X", "USD/KRW", "price"),
    SeriesSpec("kospi", "kr_benchmark", "KOSPI", "KOSPI", "price"),
    SeriesSpec("kr3y", "naver_bond", "KR3YT=RR", "국고채 3년(%)", "yield"),
    SeriesSpec("kr10y", "naver_bond", "KR10YT=RR", "국고채 10년(%)", "yield"),
)
SERIES_BY_KEY = {s.key: s for s in SERIES}


class MacroSeries(dict):
    """``dict[str, pd.Series]`` of daily macro series, plus the provenance of every entry.

    Each series is float64 on a tz-naive, ascending, unique DatetimeIndex of the source's own session
    dates (exchange-local; KRW=X uses London dates). ``series.attrs`` carries ``source``, ``symbol``,
    ``label``, ``kind`` and ``notes``. Keys that failed to load are absent and are listed in ``failures``.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.sources: dict[str, str] = {}
        self.failures: dict[str, str] = {}
        self.notes: list[str] = []


# ----------------------------------------------------------------------------------- pure helpers


def default_start(settings: Settings, today: date | None = None) -> date:
    """History needed for a ``lookback_years`` backtest with a ``z_window_years`` z-window and a 200-day MA."""
    years = settings.macro.lookback_years + settings.macro.z_window_years + 1
    t = pd.Timestamp(today or datetime.now(KST).date())
    return (t - pd.Timedelta(int(round(365.25 * years)), "D")).date()


def clean_series(s: pd.Series, kind: ValueKind) -> tuple[pd.Series, list[str]]:
    """Normalise to the MacroSeries contract. Drops NaNs and, for prices/levels, non-positive values
    (e.g. WTI settled at -37.63 on 2020-04-20), which are reported."""
    notes: list[str] = []
    out = pd.to_numeric(pd.Series(s), errors="coerce").astype("float64")
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[np.isfinite(out.to_numpy())]
    if kind in ("price", "level"):
        bad = out[out <= 0]
        if len(bad):
            shown = ", ".join(d.date().isoformat() for d in bad.index[:3])
            notes.append(f"0 이하 값 {len(bad)}일 제외({shown})")
            out = out[out > 0]
    out.index.name = "date"
    return out, notes


def parse_naver_bond_prices(rows: Any) -> pd.DataFrame:
    """Naver ``marketindex/bond/{code}/prices`` JSON rows -> DataFrame(index KST date, close, traded_at).

    ``localTradedAt`` is ISO-8601 with the market's own offset (+09:00 for Korean bonds), so its date part
    is the local session date. ``closePrice`` is a string in percent (e.g. "3.9980").
    """
    if not isinstance(rows, list):
        raise ValueError(f"naver bond: expected a list of rows, got {type(rows).__name__}")
    if not rows:
        return pd.DataFrame(columns=["close", "traded_at"], index=pd.DatetimeIndex([], name="date"))
    df = pd.DataFrame(rows)
    if "closePrice" not in df.columns or "localTradedAt" not in df.columns:
        raise ValueError(f"naver bond: unexpected columns {list(df.columns)[:8]}")
    stamps = pd.to_datetime(df["localTradedAt"].astype(str), utc=True).dt.tz_convert(KST)
    close = pd.to_numeric(df["closePrice"].astype(str).str.replace(",", ""), errors="coerce")
    out = pd.DataFrame(
        {"close": close.to_numpy(dtype="float64"), "traded_at": stamps.to_numpy()},
        index=pd.DatetimeIndex(stamps.dt.tz_localize(None).dt.normalize(), name="date"),
    )
    out = out[out["close"].notna() & (out["close"] > -5.0) & (out["close"] < 30.0)]
    return out[~out.index.duplicated(keep="first")].sort_index()


def drop_live_bond_row(df: pd.DataFrame, now_kst: datetime) -> pd.DataFrame:
    """Drop today's (KST) row while it can still be the live intraday quote."""
    if df.empty:
        return df
    now = now_kst.astimezone(KST)
    today = pd.Timestamp(now.date())
    if df.index[-1] == today and now.time() < NAVER_BOND_FINAL_KST:
        return df.iloc[:-1]
    return df


# ----------------------------------------------------------------------------------- fetchers


def fetch_naver_bond(
    code: str,
    start: date,
    now_kst: datetime | None = None,
    session: requests.Session | None = None,
) -> tuple[pd.Series, str, list[str]]:
    """All pages of a Naver bond yield series back to ``start`` -> (Series in %, source, notes)."""
    http = session or PoliteSession()
    now = now_kst or datetime.now(KST)
    url = NAVER_BOND_PRICES.format(code=code)
    frames: list[pd.DataFrame] = []
    for page in range(1, NAVER_BOND_MAX_PAGES + 1):
        r = http.get(url, params={"page": page, "pageSize": NAVER_BOND_PAGE_SIZE})
        r.raise_for_status()
        df = parse_naver_bond_prices(r.json())
        if df.empty:
            break
        frames.append(df)
        if df.index.min() <= pd.Timestamp(start):
            break
    if not frames:
        raise ValueError(f"네이버 채권 {code}: 데이터 없음")
    full = pd.concat(frames).sort_index()
    full = drop_live_bond_row(full[~full.index.duplicated(keep="first")], now)
    s = full["close"].loc[pd.Timestamp(start) :]
    notes = []
    if len(s) and s.index[0] > pd.Timestamp(start) + pd.Timedelta(30, "D"):
        notes.append(f"네이버 채권 이력은 {s.index[0].date()}부터만 제공(요청 {start})")
    return s, f"네이버 채권 {code}", notes


def _yahoo_instrument(symbol: str) -> Instrument:
    return Instrument(
        ticker=symbol,
        market=Market.US,
        name=symbol,
        currency="USD",
        yf_symbol=symbol,
        benchmark=symbol,
        benchmark_name=symbol,
    )


def fetch_yahoo(symbol: str, start: date, settings: Settings, cache: DiskCache) -> tuple[pd.Series, str, list[str]]:
    """Close series through the verified US chain (USProvider.benchmark: yfinance -> Yahoo chart)."""
    from quantcio.data.us import USProvider

    df = USProvider(settings, cache).benchmark(_yahoo_instrument(symbol), start)
    if df is None or df.empty or "close" not in df.columns:
        raise ValueError(f"Yahoo {symbol}: 빈 시계열")
    src = str(df.attrs.get("source") or "yahoo")
    return df["close"], f"Yahoo {symbol} ({src})", [str(w) for w in df.attrs.get("warnings", []) or []]


def fetch_kospi(start: date, settings: Settings, cache: DiskCache) -> tuple[pd.Series, str, list[str]]:
    """KOSPI through the KR provider's benchmark chain (Naver fchart -> yfinance ^KS11)."""
    from quantcio.data.kr import KRProvider

    floor = (pd.Timestamp(datetime.now(KST).date()) - pd.Timedelta(int(365.25 * KOSPI_MAX_YEARS), "D")).date()
    inst = Instrument(
        ticker="KOSPI",
        market=Market.KR,
        name="KOSPI",
        currency="KRW",
        benchmark="KOSPI",
        benchmark_name="KOSPI",
    )
    df = KRProvider(settings, cache).benchmark(inst, max(start, floor))
    if df is None or df.empty or "close" not in df.columns:
        raise ValueError("KOSPI: 빈 시계열")
    src = str(df.attrs.get("source") or "KR provider")
    return df["close"], src, [str(w) for w in df.attrs.get("warnings", []) or []]


def _fetch_series(
    spec: SeriesSpec, start: date, settings: Settings, cache: DiskCache
) -> tuple[pd.Series, str, list[str]]:
    """Dispatch one spec to its source. Tests monkeypatch this function."""
    if spec.source == "yahoo":
        return fetch_yahoo(spec.symbol, start, settings, cache)
    if spec.source == "kr_benchmark":
        return fetch_kospi(start, settings, cache)
    if spec.source == "naver_bond":
        return fetch_naver_bond(spec.symbol, start)
    raise ValueError(f"unknown macro source {spec.source!r}")


def _cache_key(spec: SeriesSpec, start: date) -> str:
    from quantcio.data.us import last_completed_us_session

    now = datetime.now(KST)
    # a new US session or a new KST day (and the KR bond close at 17:00) must never be served from an older entry
    return DiskCache.key(
        "macro",
        _CACHE_VERSION,
        spec.key,
        spec.source,
        spec.symbol,
        start.isoformat(),
        KOSPI_MAX_YEARS if spec.source == "kr_benchmark" else None,
        now.date().isoformat(),
        now.time() >= NAVER_BOND_FINAL_KST,
        last_completed_us_session().isoformat(),
    )


def load_macro_series(
    settings: Settings,
    cache: DiskCache,
    *,
    start: date | None = None,
    use_cache: bool = True,
    keys: tuple[str, ...] | None = None,
    max_workers: int = 4,
) -> MacroSeries:
    """Load every macro series in :data:`SERIES` (or ``keys``) from ``start`` (default :func:`default_start`).

    Each series is cached separately for ``settings.macro.ttl_hours`` under ``macro_series``. A failed
    fetch is never cached. With ``use_cache=False`` both this cache and the providers' own caches are
    bypassed. Failures go to ``.failures`` and ``.notes``, and a missing key is never an empty series.
    """
    start_d = pd.Timestamp(start).date() if start is not None else default_start(settings)
    fetch_cache = cache if use_cache else DiskCache(cache.root, enabled=False)
    specs = [SERIES_BY_KEY[k] for k in keys] if keys else list(SERIES)
    out = MacroSeries()

    def one(spec: SeriesSpec) -> dict:
        def fetch() -> dict:
            s, src, notes = _fetch_series(spec, start_d, settings, fetch_cache)
            return {"series": s, "source": src, "notes": list(notes)}

        if not use_cache:
            return fetch()
        return cache.get_or_fetch(CACHE_NAMESPACE, _cache_key(spec, start_d), settings.macro.ttl_hours, fetch)

    results: dict[str, dict | BaseException] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {spec.key: pool.submit(one, spec) for spec in specs}
        for key, fut in futures.items():
            try:
                results[key] = fut.result()
            except Exception as exc:  # noqa: BLE001 - recorded below, never silent
                results[key] = exc

    for spec in specs:
        res = results[spec.key]
        if isinstance(res, BaseException):
            err = f"{type(res).__name__}: {str(res)[:200]}"
            _fail(out, spec, err)
            continue
        s, clean_notes = clean_series(res["series"], spec.kind)
        if s.empty:
            _fail(out, spec, "정제 후 빈 시계열")
            continue
        notes = [*res.get("notes", []), *clean_notes]
        s.name = spec.key
        s.attrs.update(source=res["source"], symbol=spec.symbol, label=spec.label, kind=spec.kind, notes=notes)
        out[spec.key] = s
        out.sources[spec.key] = res["source"]
        out.notes.extend(f"{spec.label}: {n}" for n in notes)
    return out


def _fail(out: MacroSeries, spec: SeriesSpec, err: str) -> None:
    log.warning("macro series %s (%s %s) failed: %s", spec.key, spec.source, spec.symbol, err)
    out.failures[spec.key] = err
    out.notes.append(f"{spec.label}({spec.symbol}) 수집 실패: {err}")


def series_from_mapping(data: dict[str, pd.Series], sources: dict[str, str] | None = None) -> MacroSeries:
    """Wrap pre-built series (tests, replays) in a :class:`MacroSeries` after the same cleaning."""
    out = MacroSeries()
    for key, s in data.items():
        kind: ValueKind = SERIES_BY_KEY[key].kind if key in SERIES_BY_KEY else "level"
        clean, _ = clean_series(s, kind)
        if clean.empty:
            out.failures[key] = "빈 시계열"
            continue
        src = (sources or {}).get(key, "synthetic")
        clean.name = key
        clean.attrs.update(source=src, symbol=key, label=key, kind=kind, notes=[])
        out[key] = clean
        out.sources[key] = src
    return out
