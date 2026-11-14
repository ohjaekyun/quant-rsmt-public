"""Provider interface + the market-agnostic loader that assembles MarketData."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from quantcio.config import Settings
from quantcio.data.cache import DiskCache
from quantcio.i18n import lang_of, tr
from quantcio.models import (
    PRICE_COLUMNS,
    FilingItem,
    Fundamentals,
    Instrument,
    Market,
    MarketData,
    NewsItem,
)

log = logging.getLogger(__name__)

# 6-digit KRX codes, incl. the newer alphanumeric short codes such as 0126Z0 / 0001A0
_KR_CODE = re.compile(r"^(\d{6}|\d{4}[0-9A-Z]\d)(?:\.(KS|KQ))?$", re.IGNORECASE)


def detect_market(ticker: str) -> Market:
    """KRX codes (optionally with .KS/.KQ) are Korean; everything else is treated as US."""
    return Market.KR if _KR_CODE.match(ticker.strip().upper()) else Market.US


def normalise_ticker(ticker: str) -> str:
    """Canonical ticker: KRX code without suffix, or US symbol with share-class '.'/'/' mapped to '-'."""
    t = ticker.strip().upper()
    m = _KR_CODE.match(t)
    if m:
        return m.group(1)
    return t.replace(".", "-").replace("/", "-")


def _frame_warnings(df: pd.DataFrame | None, lang: str = "ko") -> list[str]:
    """Provider notes attached to a price frame (attrs), incl. an unadjusted-price marker."""
    if df is None:
        return []
    notes = [str(w) for w in df.attrs.get("warnings", []) or []]
    if df.attrs.get("adjusted") is False:
        notes.append(
            tr(
                lang,
                "수정주가 미적용 데이터(대체 소스) 사용 - 배당/분할 반영 안 됨",
                "Unadjusted prices from a fallback source - dividends/splits not reflected",
            )
        )
    return notes


def provider_warnings(provider: MarketProvider, ticker: str, lang: str = "ko") -> list[str]:
    """Collect a provider's degraded-path notes for `ticker` (KR: data_warnings, US: drain_notes)."""
    for name in ("data_warnings", "drain_notes"):
        fn = getattr(provider, name, None)
        if callable(fn):
            try:
                return [str(w) for w in fn(ticker) or []]
            except Exception as exc:  # noqa: BLE001
                return [
                    tr(
                        lang,
                        f"데이터 경고 수집 실패: {type(exc).__name__}",
                        f"Data warnings unavailable: {type(exc).__name__}",
                    )
                ]
    return []


def _extend_unique(target: list[str], items: Iterable[str]) -> None:
    for item in items:
        if item and item not in target:
            target.append(item)


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce any provider frame into the PRICE_COLUMNS contract (see models.py)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS, dtype="float64")
    out = df.copy()
    out.columns = [str(c).strip().lower().replace("adj close", "adj_close") for c in out.columns]
    missing = [c for c in PRICE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"price frame missing columns {missing}; got {list(out.columns)}")
    out = out[PRICE_COLUMNS].astype("float64")
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[(out["close"] > 0) & out["close"].notna()]
    return out


class MarketProvider(ABC):
    """One implementation per market (quantcio.data.kr.KRProvider, quantcio.data.us.USProvider).

    Every method may raise; the loader converts failures of optional parts into warnings.
    `prices` and `resolve` are mandatory - their failure aborts the ticker.
    """

    market: Market

    def __init__(self, settings: Settings, cache: DiskCache):
        self.settings = settings
        self.cache = cache

    @abstractmethod
    def resolve(self, ticker: str) -> Instrument: ...

    @abstractmethod
    def prices(self, inst: Instrument, start: date) -> pd.DataFrame:
        """Adjusted daily OHLCV from `start` to latest available session (clean_prices contract)."""

    @abstractmethod
    def benchmark(self, inst: Instrument, start: date) -> pd.DataFrame:
        """Benchmark index OHLCV (same contract). volume may be 0."""

    @abstractmethod
    def fundamentals(self, inst: Instrument) -> Fundamentals: ...

    def valuation_history(self, inst: Instrument, start: date) -> pd.DataFrame | None:
        """Optional daily/monthly PER/PBR history (columns subset of {"per","pbr"})."""
        return None

    @abstractmethod
    def news(self, inst: Instrument, since: datetime, limit: int) -> list[NewsItem]:
        """Company-specific news newer than `since` (UTC), newest first, de-duplicated."""

    @abstractmethod
    def filings(self, inst: Instrument, since: datetime, limit: int) -> list[FilingItem]:
        """Disclosures newer than `since`, newest first; earnings-related ones flagged and prioritised."""

    def fx_rate(self, from_ccy: str, to_ccy: str) -> float:
        """Spot FX to convert amounts from `from_ccy` to `to_ccy` (1.0 when equal)."""
        return 1.0

    def data_warnings(self, ticker: str) -> list[str]:
        """Korean notes on degraded / fallback data paths taken for `ticker` (shown on the report)."""
        return []

    def english_name(self, inst: Instrument) -> str | None:
        """English display name of `inst` for English reports, or None to keep `inst.name` (default: names are
        already English). Implementations must not raise and must not add report warnings."""
        return None

    @abstractmethod
    def universe(self, n: int) -> list[str]:
        """Canonical tickers of a broad, sector-diverse training universe (RSMT-Q pooled panel).

        KR: top-n common stocks by market cap across KOSPI+KOSDAQ (exclude preferred shares, SPACs, ETFs/ETNs,
        REITs). US: a static, sector-diverse large-cap list of >= n names. Deterministic for a given day.
        """


def get_provider(market: Market, settings: Settings, cache: DiskCache) -> MarketProvider:
    if market is Market.KR:
        from quantcio.data.kr import KRProvider

        return KRProvider(settings, cache)
    from quantcio.data.us import USProvider

    return USProvider(settings, cache)


@dataclass
class PricePanel:
    """Prices for the pooled RSMT-Q training universe of ONE market."""

    market: Market
    prices: dict[str, pd.DataFrame]  # ticker -> clean_prices frame
    benchmarks: dict[str, pd.DataFrame]  # benchmark id -> clean_prices frame
    ticker_benchmark: dict[str, str]  # ticker -> benchmark id (KOSPI vs KOSDAQ for KR)
    failed: dict[str, str] = field(default_factory=dict)  # ticker -> error (reported, never silent)
    # PIT membership: rows (date, ticker) = ticker is a universe member from `date` until the next rebalance date.
    # None = every loaded ticker is eligible on every date (the legacy 'current list' universe).
    membership: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)  # coverage / residual-bias notes (shown in diagnostics)


def load_price_panel(
    market: Market,
    settings: Settings,
    cache: DiskCache,
    extra_tickers: Iterable[str] = (),
    max_workers: int = 8,
) -> PricePanel:
    """Load the training universe (+ the requested tickers) for one market, in parallel, cached."""
    provider = get_provider(market, settings, cache)
    n = settings.rsmt.markets[market.value].universe_size
    tickers = list(dict.fromkeys([*provider.universe(n), *(normalise_ticker(t) for t in extra_tickers)]))
    start = date.today() - timedelta(days=int(365.25 * settings.history_years) + 30)

    prices: dict[str, pd.DataFrame] = {}
    ticker_bench: dict[str, str] = {}
    benchmarks: dict[str, pd.DataFrame] = {}
    failed: dict[str, str] = {}

    def _one(t: str):
        inst = provider.resolve(t)
        return inst, clean_prices(provider.prices(inst, start))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_one, t): t for t in tickers}
        for fut in as_completed(futures):
            requested = futures[fut]
            try:
                inst, px = fut.result()
            except Exception as exc:  # noqa: BLE001
                failed[requested] = f"{type(exc).__name__}: {str(exc)[:160]}"
                continue
            t = inst.ticker  # key by the provider's canonical ticker (e.g. BRK.B -> BRK-B)
            if len(px) < 260:
                failed[t] = tr(settings, f"이력 부족({len(px)}행)", f"history too short ({len(px)} rows)")
                continue
            prices[t] = px
            ticker_bench[t] = inst.benchmark
            if inst.benchmark not in benchmarks:
                try:
                    benchmarks[inst.benchmark] = clean_prices(provider.benchmark(inst, start))
                except Exception as exc:  # noqa: BLE001
                    failed[f"benchmark:{inst.benchmark}"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    if failed:
        log.warning(
            "price panel %s: %d/%d failed: %s",
            market.value,
            len(failed),
            len(tickers),
            failed,
        )
    return PricePanel(market, prices, benchmarks, ticker_bench, failed)


def _english_instrument(provider: MarketProvider, inst: Instrument) -> Instrument:
    """`inst` with its English display name (KR: `KRProvider.english_name`); unchanged when there is none.

    A failure keeps the original name and is logged at DEBUG only: the name is cosmetic, never a data problem."""
    try:
        name = provider.english_name(inst)
    except Exception as exc:  # noqa: BLE001 - cosmetic enrichment, the local name stays
        log.debug("%s English name unavailable: %s", inst.ticker, exc)
        return inst
    if not name or name == inst.name:
        return inst
    return inst.model_copy(update={"name": name})


def load_market_data(
    ticker: str,
    settings: Settings,
    cache: DiskCache,
    base_currency: str = "KRW",
    include_text: bool = True,
) -> MarketData:
    market = detect_market(ticker)
    provider = get_provider(market, settings, cache)
    inst = provider.resolve(normalise_ticker(ticker))
    start = date.today() - timedelta(days=int(365.25 * settings.history_years) + 30)

    raw_prices = provider.prices(inst, start)
    lang = lang_of(settings)
    warnings: list[str] = []
    _extend_unique(warnings, _frame_warnings(raw_prices, lang))
    prices = clean_prices(raw_prices)
    if len(prices) < 260:
        raise RuntimeError(f"{inst.ticker}: only {len(prices)} price rows - need >= 260 for analysis")

    def _optional(label: str, fn, default):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider failures are reported, not fatal
            log.warning("%s %s failed: %s", inst.ticker, label, exc)
            warnings.append(
                tr(
                    lang,
                    f"{label} 수집 실패: {type(exc).__name__}: {str(exc)[:120]}",
                    f"{label} unavailable: {type(exc).__name__}: {str(exc)[:120]}",
                )
            )
            return default

    def _bench() -> pd.DataFrame:
        raw = provider.benchmark(inst, start)
        prefix = tr(lang, "벤치마크: ", "Benchmark: ")
        _extend_unique(warnings, [f"{prefix}{w}" for w in _frame_warnings(raw, lang)])
        return clean_prices(raw)

    bench = _optional(tr(lang, "벤치마크", "Benchmark"), _bench, clean_prices(None))
    fund = _optional(
        tr(lang, "펀더멘털", "Fundamentals"),
        lambda: provider.fundamentals(inst),
        Fundamentals(source="unavailable"),
    )
    val_hist = _optional(
        tr(lang, "밸류에이션 히스토리", "Valuation history"), lambda: provider.valuation_history(inst, start), None
    )

    news: list[NewsItem] = []
    filings: list[FilingItem] = []
    if include_text:
        now = datetime.now(UTC)
        news = _optional(
            tr(lang, "뉴스", "News"),
            lambda: provider.news(
                inst,
                now - timedelta(days=settings.data.news_lookback_days),
                settings.data.max_news_items,
            ),
            [],
        )
        filings = _optional(
            tr(lang, "공시", "Filings"),
            lambda: provider.filings(
                inst,
                now - timedelta(days=settings.data.filing_lookback_days),
                settings.data.max_filings,
            ),
            [],
        )
        if not news:
            warnings.append(
                tr(lang, "최근 뉴스 없음 - 감성 점수 신뢰도 낮음", "No recent news - low sentiment reliability")
            )
        if not filings:
            warnings.append(tr(lang, "최근 공시/실적 자료 없음", "No recent filings/earnings"))

    fx = 1.0
    if inst.currency != base_currency:
        fx = _optional(tr(lang, "환율", "FX rate"), lambda: provider.fx_rate(inst.currency, base_currency), 1.0)

    # degraded / fallback paths the provider took must reach the report (never silent)
    _extend_unique(warnings, provider_warnings(provider, inst.ticker, lang))
    if lang == "en":  # display only: every provider call above used the provider's own (Korean) name
        inst = _english_instrument(provider, inst)

    return MarketData(
        instrument=inst,
        prices=prices,
        benchmark=bench,
        fundamentals=fund,
        valuation_history=val_hist,
        news=news,
        filings=filings,
        fx_to_base=fx,
        warnings=warnings,
    )
