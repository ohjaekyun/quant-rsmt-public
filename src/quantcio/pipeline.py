"""Report orchestration (async): data -> quant (RSMT-Q per market) -> LLM -> fusion -> execution -> sizing -> PDF.

Public API
----------
- ``run_report(tickers, settings, capital=None, base_currency="KRW", use_llm=True, out_dir=None, llm_client=None)
  -> ReportArtifacts(json_path, html_path, pdf_path, pages, report, timings, warnings)``
- ``build_report(...) -> RunResult`` - the same run without rendering (used by ``quantcio analyze``).
- ``fit_market(market, settings, cache, requested) -> MarketFit`` - the universe-fitted RSMT-Q market model (cached
  per code / config / market for rsmt.model_max_age_days) served on a fresh panel of today's prices + the requests.
- ``tune_rsmt(panel_data, settings) -> TuneReport`` - the ``quantcio rsmt --tune`` grid (recommendations only).

Flow per run (docs/ARCHITECTURE.md, "pipeline.py / cli.py"):

1. In parallel: ``load_market_data`` per ticker (thread pool; a failing ticker is reported in
   ``PortfolioReport.notes`` and the run continues), and ``fit_market`` for every
   market present (``load_price_panel`` with the requested tickers -> ``build_panel`` -> ``fit_market_model``).
2. ``MarketModel.predict_many`` once per market (hold_days = fusion.target_horizon_days), so the batch health
   checks (served collapse, failed panel share) are final before any forecast is used.
3. ``build_quant`` per ticker (the served ReturnDistribution object is kept for the execution plan).
4. LLM (one ``LLMClient`` per run): news + filings -> preliminary ``fuse`` -> critic; skipped with --no-llm.
5. Final ``fuse`` -> ``build_execution_plan`` (barrier from the SAME path matrix) -> ``apply_barrier_guardrail``.
6. ``size_portfolio`` -> ``portfolio_risk`` -> market regime chips -> model diagnostics -> render (PDF via
   ``asyncio.to_thread(fit_to_pages_isolated)``: a watchdog subprocess with a timeout, one retry and an HTML
   fallback, so a crashed Playwright driver can never hang the run) -> JSON next to the HTML/PDF -> forward ledger
   (``tracking.record_predictions`` when ``holdout.record_reports``; a failure is a run warning, never fatal).

Nothing is dropped silently: every degraded path lands in the ticker's warnings, the report notes or the
model diagnostics, and in the log.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pandas as pd

from quantcio import __version__
from quantcio.config import Settings
from quantcio.data import base as data_base
from quantcio.data.base import PricePanel, detect_market, normalise_ticker
from quantcio.data.cache import DiskCache
from quantcio.fusion.risk import build_execution_plan, portfolio_risk, size_portfolio
from quantcio.fusion.signal import apply_barrier_guardrail, fuse
from quantcio.i18n import lang_of, tr
from quantcio.llm import LLMClient, LLMConfigError, run_llm_analysis
from quantcio.macro.regime import LABELS_EN as MACRO_LABELS_EN
from quantcio.models import (
    ExcessForecast,
    ExecutionPlan,
    ExecutiveSignal,
    FactorModelScore,
    Instrument,
    LLMIntelligence,
    MacroRegime,
    Market,
    MarketData,
    PortfolioReport,
    QuantSignal,
    Signal,
    TickerReport,
)
from quantcio.quant import build_quant

if TYPE_CHECKING:
    from quantcio.quant.distribution import ReturnDistribution
    from quantcio.quant.panel import PanelData
    from quantcio.quant.rsmt import MarketModel

log = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")
MODEL_CACHE_NS = "rsmt_model"
MODEL_CACHE_VERSION = 3  # 3: fit (universe-only market model, reused for rsmt.model_max_age_days) / serve split
LOAD_WORKERS = 8
BENCH_LABELS = {"KOSPI": "KOSPI", "KOSDAQ": "KOSDAQ", "SPY": "S&P 500"}
REGIME_ORDER = ("KOSPI", "KOSDAQ", "SPY")
VIX_BANDS = ((15.0, "저변동"), (25.0, "보통"), (math.inf, "고변동"))
VIX_BANDS_EN = {"저변동": "low vol", "보통": "normal", "고변동": "high vol"}
NO_LLM_NOTE = "LLM 분석 생략(--no-llm): 뉴스·공시 필러 제외, 정량 신호만 반영"
NO_LLM_NOTE_EN = "LLM analysis skipped (--no-llm): news / filings pillar excluded, quantitative signal only"
# modules whose code defines a fitted MarketModel: a change to any of them invalidates the model cache
_MODEL_CODE_MODULES = ("rsmt.py", "panel.py", "distribution.py", "features.py")


class PipelineError(RuntimeError):
    """The run cannot produce a report (no ticker loaded, LLM misconfigured, ...). The message is Korean."""


# =============================================================================== results


@dataclass
class MarketFit:
    """RSMT-Q model of one market (or why there is none)."""

    market: Market
    model: MarketModel | None = None
    benchmarks: dict[str, pd.DataFrame] = field(default_factory=dict)
    error: str | None = None
    cached: bool = False
    seconds: float = 0.0
    n_panel: int = 0
    notes: list[str] = field(default_factory=list)  # panel data-quality lines for the diagnostics box
    pit: bool = False  # trained on the point-in-time universe (settings.pit.universe_mode == "pit")


@dataclass
class TickerRun:
    """Everything computed for one requested ticker."""

    requested: str
    md: MarketData
    quant: QuantSignal | None = None
    dist: ReturnDistribution | None = None
    llm: LLMIntelligence | None = None
    execsig: ExecutiveSignal | None = None
    plan: ExecutionPlan | None = None

    @property
    def ticker(self) -> str:
        return self.md.instrument.ticker

    @property
    def market(self) -> Market:
        return self.md.instrument.market


@dataclass
class RunResult:
    report: PortfolioReport
    runs: list[TickerRun]
    fits: dict[str, MarketFit]
    timings: dict[str, float]
    llm_usage: dict[str, dict[str, float]] = field(default_factory=dict)  # LLMClient.usage_summary()


@dataclass
class ReportArtifacts:
    json_path: Path
    html_path: Path | None
    pdf_path: Path | None  # None when no browser could start (HTML fallback)
    pages: int  # 0 on the HTML fallback
    report: PortfolioReport
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)  # run-level (render) warnings
    llm_usage: dict[str, dict[str, float]] = field(default_factory=dict)  # per task tokens / cost of the run


# =============================================================================== helpers


def parse_tickers(tickers: str | Sequence[str]) -> list[str]:
    """'005930, AAPL nvda' or a list -> requested spellings, de-duplicated by canonical ticker (order kept)."""
    raw = tickers.replace(",", " ").split() if isinstance(tickers, str) else [str(t) for t in tickers]
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        t = t.strip()
        if not t:
            continue
        key = normalise_ticker(t)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _short(exc: BaseException, n: int = 160) -> str:
    text = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
    return " ".join(str(text).split())[:n]


def _warn(md: MarketData, msg: str) -> None:
    log.warning("%s: %s", md.instrument.ticker, msg)
    if msg not in md.warnings:
        md.warnings.append(msg)


class _Timer:
    def __init__(self, timings: dict[str, float], key: str, progress: Callable[[str], None] | None = None):
        self.timings, self.key, self.progress = timings, key, progress

    def __enter__(self) -> _Timer:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        dt = time.perf_counter() - self.t0
        self.timings[self.key] = round(self.timings.get(self.key, 0.0) + dt, 3)
        log.info("단계 %s: %.2fs", self.key, dt)


@functools.lru_cache(maxsize=1)
def _model_code_fingerprint() -> str:
    root = Path(__file__).resolve().parent / "quant"
    h = hashlib.sha1(__version__.encode())
    for name in _MODEL_CODE_MODULES:
        try:
            h.update((root / name).read_bytes())
        except OSError:
            h.update(name.encode())
    return h.hexdigest()[:12]


# =============================================================================== RSMT-Q per market


def requested_price_panel(
    market: Market, settings: Settings, cache: DiskCache, requested: Sequence[str], max_workers: int = 8
) -> PricePanel:
    """Production-provider prices of the requested tickers only (and their benchmarks)."""
    provider = data_base.get_provider(market, settings, cache)
    start = date.today() - timedelta(days=int(365.25 * settings.history_years) + 30)
    prices: dict[str, pd.DataFrame] = {}
    ticker_bench: dict[str, str] = {}
    benchmarks: dict[str, pd.DataFrame] = {}
    failed: dict[str, str] = {}

    def one(t: str):
        inst = provider.resolve(t)
        return inst, data_base.clean_prices(provider.prices(inst, start))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(one, normalise_ticker(t)): t for t in requested}
        for fut, t in futures.items():
            try:
                inst, px = fut.result()
            except Exception as exc:  # noqa: BLE001 - recorded per ticker, never silent
                failed[t] = f"{type(exc).__name__}: {str(exc)[:160]}"
                continue
            if len(px) < 260:
                failed[inst.ticker] = tr(settings, f"이력 부족({len(px)}행)", f"history too short ({len(px)} rows)")
                continue
            prices[inst.ticker] = px
            ticker_bench[inst.ticker] = inst.benchmark
            if inst.benchmark not in benchmarks:
                try:
                    benchmarks[inst.benchmark] = data_base.clean_prices(provider.benchmark(inst, start))
                except Exception as exc:  # noqa: BLE001
                    failed[f"benchmark:{inst.benchmark}"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return PricePanel(market, prices, benchmarks, ticker_bench, failed)


MAX_ALIGN_SESSIONS = 5  # extras ahead of the universe by more than this are left alone (a data problem to see)
ALIGN_MEMBER_SHARE = 0.8  # the universe's "latest common bar": the date at least this share of members reached


def align_extras_to_universe(panel: PricePanel, extras: Sequence[str], settings: Settings) -> list[str]:
    """Cut the requested names' frames (in the RSMT-Q panel only) to the universe's latest common bar when the
    universe lags them by 1..``MAX_ALIGN_SESSIONS`` sessions (in place). Returns the notes.

    Why: the cross-sectional momentum z of a date needs the universe members of that date. A universe one session
    behind the production prices of the requested names left the latest date with only the requested names, so
    every one of them was served in degraded mode (xs_mom_z -> 0; seen 2026-09-27 on all US tickers). One session
    older inputs are better than a zeroed offset.
    """
    mem = panel.membership
    if mem is None or mem.empty or not extras:
        return []
    last_reb = mem["date"].max()
    members = [t for t in mem.loc[mem["date"] == last_reb, "ticker"].astype(str) if t in panel.prices]
    lasts = sorted((panel.prices[t].index[-1] for t in members if len(panel.prices[t])), reverse=True)
    if not lasts:
        return []
    common = lasts[max(0, math.ceil(len(lasts) * ALIGN_MEMBER_SHARE) - 1)]
    cut: list[str] = []
    lag_max = 0
    for t in extras:
        px = panel.prices.get(t)
        if px is None or not len(px) or px.index[-1] <= common:
            continue
        lag = int((px.index > common).sum())
        if lag > MAX_ALIGN_SESSIONS:
            continue
        panel.prices[t] = px.loc[px.index <= common]
        cut.append(t)
        lag_max = max(lag_max, lag)
    if not cut:
        return []
    log.warning(
        "RSMT-Q %s: %d requested names aligned to the universe's last common bar %s",
        panel.market.value,
        len(cut),
        common.date(),
    )
    return [
        tr(
            settings,
            f"[정보] RSMT-Q: 유니버스 가격이 요청 종목보다 최대 {lag_max}세션 늦어, 요청 종목 {len(cut)}개의 분포 예측 "
            f"기준일을 유니버스 공통 최신일 {common.date()}로 맞춤(횡단면 모멘텀 z 계산용)",
            f"[info] RSMT-Q: the universe's prices lag the requested names by up to {lag_max} sessions - the "
            f"distribution of {len(cut)} requested names is predicted as of the universe's last common bar "
            f"{common.date()} (for the cross-sectional momentum z)",
        )
    ]


def rsmt_history_start(market: Market, settings: Settings, today: date | None = None) -> date:
    """First date of the RSMT-Q training universe: rsmt.markets.<M>.history_start, else today - history_years."""
    fixed = settings.rsmt.markets[Market(market).value].history_start
    if fixed is not None:
        return fixed
    return (today or date.today()) - timedelta(days=int(365.25 * settings.history_years) + 30)


def training_panel(
    market: Market, settings: Settings, cache: DiskCache, requested: Sequence[str] = ()
) -> tuple[PricePanel, list[str], set[str] | None]:
    """RSMT-Q training panel: the point-in-time universe (``pit.universe_mode == "pit"``: top
    ``rsmt.markets.<M>.universe_size`` members per month-end, delisted names with their terminal prices, rows
    restricted to members at each date) plus the requested tickers from the production providers; or the
    legacy current-list universe. A PIT load failure falls back to the current list with a loud note.

    Returns (panel, notes, universe tickers or None for the legacy universe).
    """
    notes: list[str] = []
    if settings.pit.enabled and settings.pit.universe_mode == "pit":
        try:
            from quantcio.data.pit import load_pit_universe

            n = int(settings.rsmt.markets[market.value].universe_size)
            start = rsmt_history_start(market, settings)
            uni = load_pit_universe(market, settings, cache, n, start)
            extra = requested_price_panel(market, settings, cache, requested) if requested else None
            panel = uni.to_price_panel(extra)
            if extra is not None:
                notes.extend(align_extras_to_universe(panel, list(extra.prices), settings))
            members = set(panel.membership["ticker"].astype(str)) if panel.membership is not None else set()
            n_ent = len(members)
            notes.append(
                tr(
                    settings,
                    f"[정보] RSMT-Q 학습 유니버스: 시점별(PIT) 상위 {n}종목(월말 구성, 상장폐지 포함 {n_ent}개 엔티티, "
                    f"{start}~) + 요청 종목",
                    f"[info] RSMT-Q training universe: point-in-time top {n} (month-end membership, {n_ent} entities "
                    f"incl. delisted, from {start}) + requested names",
                )
            )
            return panel, notes, members
        except Exception as exc:  # noqa: BLE001 - loud fallback, never silent
            log.exception("RSMT-Q %s: PIT universe failed - falling back to the current list", market.value)
            notes.append(
                tr(
                    settings,
                    f"⚠ PIT 유니버스 로드 실패({type(exc).__name__}: {_short(exc, 100)}) → 현재 목록 유니버스로 대체"
                    "(생존 편향 가능)",
                    f"⚠ PIT universe failed ({type(exc).__name__}: {_short(exc, 100)}) -> current-list universe "
                    "(survivorship bias possible)",
                )
            )
    panel = data_base.load_price_panel(market, settings, cache, extra_tickers=requested)
    return panel, notes, None


def fit_market(
    market: Market,
    settings: Settings,
    cache: DiskCache,
    requested: Sequence[str] = (),
    *,
    use_cache: bool = True,
) -> MarketFit:
    """Price panel (universe + requested) -> served MarketModel; never raises.

    Fit and serve are split (docs/RSMT.md section 8). The market model - tree, offsets, sigma equation, CV skill gate
    and the v centering c_v - is fitted on the UNIVERSE only (the longest history, every session; minutes to tens of
    minutes) and cached (namespace ``rsmt_model``) under the code fingerprint, the config digest, the market and the
    report language for ``rsmt.model_max_age_days``; it never depends on which other tickers a report requests. Every
    call builds a fresh serving panel from today's prices (universe + requested, with the fitted c_v) and serves it
    with the cached model. ``use_cache=False`` refits (and stores the new fit). A failure is returned in
    ``MarketFit.error``.
    """
    from quantcio.quant.panel import build_panel
    from quantcio.quant.rsmt import MarketModel

    t0 = time.perf_counter()
    try:
        panel, notes, universe = training_panel(market, settings, cache, requested)
        t_load = time.perf_counter() - t0
        gap = panel_gap_notes(panel, settings=settings)
        notes.extend(gap)
        for n in gap:
            log.warning("RSMT-Q %s %s", market.value, n)
        fitted, cached = market_model_fit(market, panel, settings, cache, universe=universe, use_cache=use_cache)
        notes.extend(fitted.get("notes") or [])
        serving = build_panel(panel, settings, c_v=float(fitted["c_v"]))
        model = MarketModel(serving, fitted["tree"], fitted["cv"], settings)
        served = pd.Timestamp(serving.as_of) if serving.as_of is not None else pd.Timestamp(fitted["as_of"])
        age = (served - pd.Timestamp(fitted["as_of"])).days
        span, n_fit, max_age = f"{fitted['first_date']}~{fitted['as_of']}", fitted["n_train"], fitted_max_age(settings)
        notes.append(
            tr(
                settings,
                f"[정보] RSMT-Q 시장 모델: 유니버스 {fitted['n_tickers']}종목·{span} 매 거래일 {n_fit:,}행으로 적합"
                f"(데이터 {age}일 전 기준, 최대 {max_age}일 재사용) · 예측은 {served.date()} 가격으로 계산",
                f"[info] RSMT-Q market model: fitted on the {fitted['n_tickers']}-name universe, {span}, "
                f"{n_fit:,} daily rows (data {age} days old, reused up to {max_age} days) · forecasts use "
                f"{served.date()} prices",
            )
        )
        log.info(
            "RSMT-Q %s: %s model (fitted on data to %s), serving panel of %d names built in %.1fs",
            market.value,
            "cached" if cached else "new",
            fitted["as_of"],
            len(panel.prices),
            time.perf_counter() - t0 - t_load,
        )
    except Exception as exc:  # noqa: BLE001 - returned and reported per ticker, never silent
        log.exception("RSMT-Q %s fit failed", market.value)
        return MarketFit(
            market=market,
            error=f"{type(exc).__name__}: {_short(exc)}",
            seconds=round(time.perf_counter() - t0, 3),
        )
    return MarketFit(
        market=market,
        model=model,
        benchmarks=dict(panel.benchmarks),
        cached=cached,
        seconds=round(time.perf_counter() - t0, 3),
        n_panel=len(panel.prices),
        notes=notes,
        pit=universe is not None,
    )


def fitted_max_age(settings: Settings) -> str:
    return f"{float(settings.rsmt.model_max_age_days):g}"


def market_model_key(market: Market, settings: Settings) -> str:
    """Cache key of the universe-fitted market model: code, config digest, market, report language (no data snapshot:
    the model is reused for rsmt.model_max_age_days while the serving panel is rebuilt every run)."""
    return DiskCache.key(
        "rsmt-market-model",
        MODEL_CACHE_VERSION,
        _model_code_fingerprint(),
        Market(market).value,
        settings.digest(),
        lang_of(settings),  # diagnostics text is generated at fit time
    )


def universe_price_panel(
    market: Market, panel: PricePanel, settings: Settings, cache: DiskCache, universe: set[str] | None
) -> PricePanel:
    """The price panel without the requested names outside the market universe (the model is fitted on this)."""
    if universe is None:
        n = settings.rsmt.markets[market.value].universe_size
        universe = {normalise_ticker(t) for t in data_base.get_provider(market, settings, cache).universe(n)}
    else:
        universe = {normalise_ticker(t) for t in universe} | set(universe)
    extras = {t for t in panel.prices if t not in universe and normalise_ticker(t) not in universe}
    if not extras:
        return panel
    return PricePanel(
        market=market,
        prices={t: px for t, px in panel.prices.items() if t not in extras},
        benchmarks=panel.benchmarks,
        ticker_benchmark={t: b for t, b in panel.ticker_benchmark.items() if t not in extras},
        failed={t: e for t, e in panel.failed.items() if normalise_ticker(t) in universe},
        membership=panel.membership,
    )


def market_model_fit(
    market: Market,
    panel: PricePanel,
    settings: Settings,
    cache: DiskCache,
    *,
    universe: set[str] | None = None,
    use_cache: bool = True,
) -> tuple[dict[str, Any], bool]:
    """(fitted market model, from cache). The fit: universe-only panel -> `fit_market_model` (CV skill gate + final
    fit). Stored: tree, cv, c_v, the fit panel's as_of / first date / row and ticker counts and its notes."""
    from quantcio.quant.panel import build_panel
    from quantcio.quant.rsmt import fit_market_model

    key = market_model_key(market, settings)
    ttl = float(settings.rsmt.model_max_age_days) * 24.0
    # a fit on the fallback universe (PIT configured but its load failed -> current list) is served but never cached:
    # it must not stand in for the PIT model for model_max_age_days
    fallback = settings.pit.enabled and settings.pit.universe_mode == "pit" and universe is None
    hit = cache.get(MODEL_CACHE_NS, key, ttl) if use_cache and not fallback else None
    if isinstance(hit, dict) and hit.get("tree") is not None:
        return hit, True
    t1 = time.perf_counter()
    uni = universe_price_panel(market, panel, settings, cache, universe)
    pdata = build_panel(uni, settings)
    model = fit_market_model(pdata, settings)
    tr_ = pdata.train
    fitted = {
        "tree": model.tree,
        "cv": model.cv,
        "c_v": float(pdata.c_v),
        "as_of": str(pd.Timestamp(pdata.as_of).date()) if pdata.as_of is not None else "",
        "first_date": str(pd.to_datetime(tr_["date"]).min().date()),
        "n_train": int(len(tr_)),
        "n_tickers": int(tr_["ticker"].nunique()),
        "fitted_at": datetime.now(KST).isoformat(),
        "seconds": round(time.perf_counter() - t1, 1),
        "notes": list(pdata.notes),
    }
    log.info("RSMT-Q %s: market model fitted in %.1fs (%d rows)", market.value, fitted["seconds"], fitted["n_train"])
    if fallback:
        log.warning("RSMT-Q %s: fitted on the fallback universe - not cached", market.value)
        return fitted, False
    try:
        cache.set(MODEL_CACHE_NS, key, fitted)
    except Exception as exc:  # noqa: BLE001 - a cache write failure only costs a refit next time
        log.warning("RSMT-Q %s model cache write failed: %s", market.value, exc)
    return fitted, False


PANEL_GAP_WINDOW = 60  # sessions checked for bars missing against the benchmark calendar


def panel_gap_notes(
    panel: PricePanel, window: int = PANEL_GAP_WINDOW, *, settings: Settings | None = None
) -> list[str]:
    """Recent bars missing against the benchmark calendar across the training panel (one '[정보]' line).

    The providers flag such gaps per ticker, but the universe names never reach a report ticker's
    warnings; a feed that dropped one session for most of the panel must still be visible.
    """
    missing: dict[str, list[pd.Timestamp]] = {}
    for t, px in panel.prices.items():
        bench = panel.benchmarks.get(panel.ticker_benchmark.get(t, ""))
        if bench is None or bench.empty or px.empty:
            continue
        cal = bench.index[(bench.index >= px.index[0]) & (bench.index <= px.index[-1])][-window:]
        gap = cal.difference(px.index)
        if len(gap):
            missing[t] = list(gap)
    if not missing:
        return []
    counts = pd.Series([d for ds in missing.values() for d in ds]).value_counts()
    top = ", ".join(f"{d.date()}×{n}" for d, n in counts.head(3).items())
    return [
        tr(
            settings,
            f"[정보] 패널 데이터: 최근 {window}거래일 중 벤치마크 대비 결측 봉 {len(missing)}/{len(panel.prices)}종목 "
            f"(최빈 결측일 {top}) — 해당 구간 수익률은 다일간 수익률",
            f"[info] panel data: bars missing vs the benchmark in the last {window} sessions for "
            f"{len(missing)}/{len(panel.prices)} names (most common {top}) - returns there span several days",
        )
    ]


def market_diagnostics(fit: MarketFit, settings: Settings | None = None) -> list[str]:
    """Model diagnostics lines for the report: MarketModel.diagnostics() + health(), or the failure."""
    if fit.model is None:
        return [
            tr(
                settings,
                f"⚠ RSMT-Q {fit.market.value} 모델 적합 실패: {fit.error or '사유 미상'} — 해당 시장은 ML 필러 제외",
                f"⚠ RSMT-Q {fit.market.value} fit failed: {fit.error or 'unknown reason'} - ML pillar dropped for this "
                "market",
            )
        ]
    try:
        return [*fit.model.diagnostics(), *fit.model.health(), *fit.notes]
    except Exception as exc:  # noqa: BLE001
        log.exception("RSMT-Q %s diagnostics failed", fit.market.value)
        return [
            tr(
                settings,
                f"⚠ RSMT-Q {fit.market.value} 진단 출력 오류: {type(exc).__name__}: {_short(exc)}",
                f"⚠ RSMT-Q {fit.market.value} diagnostics error: {type(exc).__name__}: {_short(exc)}",
            )
        ]


# =============================================================================== regime


def _fmt_signed_pct(x: float) -> str:
    return f"{x * 100:+.1f}%".replace("-", "−")


def regime_text(bench: pd.DataFrame, settings: Settings | None = None) -> str | None:
    """'20D +3.1% · MA200 위' from a benchmark frame (None when the history is too short)."""
    if bench is None or bench.empty or "close" not in bench:
        return None
    c = pd.to_numeric(bench["close"], errors="coerce").dropna()
    c = c[c > 0]
    if len(c) < 200:
        return None
    r20 = float(c.iloc[-1] / c.iloc[-21] - 1.0)
    above = float(c.iloc[-1]) > float(c.iloc[-200:].mean())
    pos = tr(settings, "위" if above else "아래", "above" if above else "below")
    return f"20D {_fmt_signed_pct(r20)} · MA200 {pos}"


def vix_text(settings: Settings, cache: DiskCache) -> str:
    """Latest VIX close with a band label, e.g. '15.2 (저변동)'. Raises when unavailable."""
    provider = data_base.get_provider(Market.US, settings, cache)
    inst = Instrument(
        ticker="^VIX",
        market=Market.US,
        name="CBOE VIX",
        currency="USD",
        yf_symbol="^VIX",
        benchmark="^VIX",
        benchmark_name="VIX",
    )
    df = data_base.clean_prices(provider.benchmark(inst, date.today() - timedelta(days=45)))
    if df.empty:
        raise ValueError("VIX 시계열 없음")
    level = float(df["close"].iloc[-1])
    if not (math.isfinite(level) and 3.0 < level < 150.0):
        raise ValueError(f"VIX 값 이상: {level}")
    label = next(lbl for lim, lbl in VIX_BANDS if level < lim)
    return f"{level:.1f} ({tr(settings, label, VIX_BANDS_EN[label])})"


def market_regime(
    runs: Sequence[TickerRun],
    fits: dict[str, MarketFit],
    settings: Settings,
    cache: DiskCache,
    base_currency: str,
    notes: list[str],
) -> dict[str, str]:
    """Regime chips: KOSPI/KOSDAQ (KR present), S&P 500 (US present): 20D return and MA200 position; VIX; FX."""
    markets = {r.market for r in runs}
    frames: dict[str, pd.DataFrame] = {}
    for fit in fits.values():
        frames.update({k: v for k, v in fit.benchmarks.items() if k not in frames})
    for r in runs:  # a failed market fit still has the tickers' own benchmark frames
        b = r.md.instrument.benchmark
        if b not in frames and r.md.benchmark is not None and not r.md.benchmark.empty:
            frames[b] = r.md.benchmark
    wanted = [
        b
        for b in REGIME_ORDER
        if (b in ("KOSPI", "KOSDAQ") and Market.KR in markets) or (b == "SPY" and Market.US in markets)
    ]
    out: dict[str, str] = {}
    for b in wanted:
        text = regime_text(frames.get(b), settings) if b in frames else None
        if text is None:
            log.warning("market regime: %s benchmark unavailable", b)
            notes.append(
                tr(
                    settings,
                    f"시장 레짐: {BENCH_LABELS[b]} 지수 데이터 없음 — 칩 생략",
                    f"Market regime: no {BENCH_LABELS[b]} index data - chip omitted",
                )
            )
            continue
        out[BENCH_LABELS[b]] = text
    if Market.US in markets:
        try:
            out["VIX"] = vix_text(settings, cache)
        except Exception as exc:  # noqa: BLE001 - optional chip, reported
            log.warning("VIX unavailable: %s", exc)
            notes.append(
                tr(
                    settings,
                    f"VIX 수집 실패({type(exc).__name__}) — 시장 레짐 칩 생략",
                    f"VIX unavailable ({type(exc).__name__}) - regime chip omitted",
                )
            )
    if base_currency == "KRW" and Market.US in markets:
        fx = next((r.md.fx_to_base for r in runs if r.md.instrument.currency == "USD" and r.md.fx_to_base != 1.0), None)
        if fx is not None:
            out["USD/KRW"] = f"{fx:,.0f}"
    elif base_currency == "USD" and Market.KR in markets:
        fx = next((r.md.fx_to_base for r in runs if r.md.instrument.currency == "KRW" and r.md.fx_to_base != 1.0), None)
        if fx:
            out["USD/KRW"] = f"{1.0 / fx:,.0f}"
    return out


# =============================================================================== LLM


def make_llm_client(settings: Settings) -> LLMClient:
    """One client per run. A missing credential is a clear Korean error that points at --no-llm."""
    try:
        return LLMClient(settings)
    except LLMConfigError as exc:
        hint = (
            ""
            if "--no-llm" in str(exc)
            else tr(
                settings,
                "\n→ ANTHROPIC_API_KEY를 .env에 설정하거나 --no-llm 옵션으로 실행하세요.",
                "\n→ set ANTHROPIC_API_KEY in .env or run with --no-llm.",
            )
        )
        raise PipelineError(
            tr(settings, f"LLM 초기화 실패: {exc}{hint}", f"LLM initialisation failed: {exc}{hint}")
        ) from exc


async def _llm_for(run: TickerRun, client: Any, settings: Settings) -> LLMIntelligence:
    md, quant = run.md, run.quant
    assert quant is not None

    def prelim(partial: LLMIntelligence) -> ExecutiveSignal:  # critic is None in the partial
        return fuse(md, quant, partial, settings)

    try:
        return await run_llm_analysis(client, md, quant, settings, prelim)
    except Exception as exc:  # noqa: BLE001 - run_llm_analysis never raises for LLM failures; a bug is reported
        log.exception("%s: LLM analysis raised", run.ticker)
        return LLMIntelligence(
            errors=[
                tr(
                    settings,
                    f"LLM 분석 예외: {type(exc).__name__}: {_short(exc)}",
                    f"LLM analysis exception: {type(exc).__name__}: {_short(exc)}",
                )
            ]
        )


# =============================================================================== build


async def build_report(
    tickers: str | Sequence[str],
    settings: Settings,
    capital: float | None = None,
    base_currency: str = "KRW",
    use_llm: bool = True,
    llm_client: Any = None,
    *,
    cache: DiskCache | None = None,
    use_model_cache: bool = True,
    progress: Callable[[str], None] | None = None,
) -> RunResult:
    """Everything except rendering: returns the PortfolioReport plus the per-ticker intermediate objects.

    Raises:
        PipelineError: bad arguments, LLM misconfigured, or no ticker could be loaded.
    """
    requested = parse_tickers(tickers)
    if not requested:
        raise PipelineError(
            tr(
                settings,
                "분석할 종목이 없습니다 (--tickers 005930,AAPL 형식으로 입력)",
                "No tickers to analyse (use --tickers 005930,AAPL)",
            )
        )
    base_currency = base_currency.upper()
    if base_currency not in ("KRW", "USD"):
        raise PipelineError(
            tr(
                settings,
                f"기준 통화는 KRW 또는 USD만 지원합니다: {base_currency}",
                f"Base currency must be KRW or USD: {base_currency}",
            )
        )
    if capital is not None and not (math.isfinite(capital) and capital > 0):
        raise PipelineError(
            tr(settings, f"투자금(capital)은 양수여야 합니다: {capital}", f"Capital must be positive: {capital}")
        )

    say = progress or (lambda _msg: None)
    timings: dict[str, float] = {}
    notes: list[str] = []
    t_start = time.perf_counter()
    cache = cache or DiskCache(settings.cache_path)

    own_client = False
    if use_llm and llm_client is None:
        llm_client = make_llm_client(settings)  # fail fast, before any data is loaded
        own_client = True
    try:
        by_market: dict[Market, list[str]] = {}
        for t in requested:
            by_market.setdefault(detect_market(t), []).append(t)

        loop = asyncio.get_running_loop()
        runs: list[TickerRun] = []
        failed: dict[str, str] = {}
        fits: dict[str, MarketFit] = {}
        with ThreadPoolExecutor(max_workers=LOAD_WORKERS, thread_name_prefix="qc-load") as io_pool:

            async def load(t: str) -> TickerRun | BaseException:
                try:
                    t0 = time.perf_counter()
                    md = await loop.run_in_executor(
                        io_pool,
                        functools.partial(
                            data_base.load_market_data,
                            t,
                            settings,
                            cache,
                            base_currency=base_currency,
                            include_text=use_llm,
                        ),
                    )
                    log.info("%s: data %.1fs", md.instrument.ticker, time.perf_counter() - t0)
                    return TickerRun(requested=t, md=md)
                except Exception as exc:  # noqa: BLE001 - reported in the notes, the run continues
                    log.exception("%s: data load failed", t)
                    return exc

            async def fit(m: Market) -> MarketFit:
                return await loop.run_in_executor(
                    io_pool,
                    functools.partial(fit_market, m, settings, cache, by_market[m], use_cache=use_model_cache),
                )

            say(
                tr(
                    settings,
                    f"데이터 로드 {len(requested)}종목 + RSMT-Q 시장 모델 {len(by_market)}개 적합 중",
                    f"Loading data for {len(requested)} tickers, fitting {len(by_market)} RSMT-Q models",
                )
            )
            with _Timer(timings, "data+fit"):
                fit_markets = list(by_market) if settings.rsmt.enabled else []
                results = await asyncio.gather(
                    *(load(t) for t in requested),
                    *(fit(m) for m in fit_markets),
                )
            for t, res in zip(requested, results[: len(requested)]):
                if isinstance(res, BaseException):
                    failed[t] = f"{type(res).__name__}: {_short(res)}"
                else:
                    runs.append(res)
            for m, res in zip(fit_markets, results[len(requested) :]):
                fits[m.value] = res
                timings[f"fit_{m.value}"] = res.seconds
                if res.cached:
                    say(
                        tr(
                            settings,
                            f"RSMT-Q {m.value}: 동일 데이터 스냅샷의 적합 모델 캐시 재사용",
                            f"RSMT-Q {m.value}: fitted model reused from the cache (same data snapshot)",
                        )
                    )
        for t, err in failed.items():
            notes.append(
                tr(
                    settings,
                    f"{t}: 데이터 로드 실패({err}) — 리포트에서 제외",
                    f"{t}: data load failed ({err}) - excluded",
                )
            )
        if not runs:
            raise PipelineError(
                tr(settings, "모든 종목의 데이터 로드에 실패했습니다: ", "Data load failed for every ticker: ")
                + "; ".join(f"{t} {e}" for t, e in failed.items())
            )
        if not settings.rsmt.enabled:
            notes.append(
                tr(
                    settings,
                    "RSMT-Q 비활성화(rsmt.enabled=false) — ML 필러 제외",
                    "RSMT-Q disabled (rsmt.enabled=false) - the ML pillar is dropped",
                )
            )

        # ---- selected stock-selection model (quantcio modeling select) + PIT factor model, per market
        say(tr(settings, "종목 선택 모델·팩터 모델 점수화 중", "Scoring the stock-selection and factor models"))
        excess: dict[str, ExcessForecast] = {}
        factor_scores: dict[str, FactorModelScore] = {}
        factor_bt: dict[str, Any] = {}
        order_v: dict[str, Any] = {}
        diag_extra: dict[str, list[str]] = {}
        with _Timer(timings, "selection_model"):
            for m in by_market:
                mruns = [r for r in runs if r.market is m]
                if not mruns:
                    continue
                ex, ex_lines = await loop.run_in_executor(None, _excess_for_market, m, mruns, settings, cache, notes)
                excess.update(ex)
                diag_extra.setdefault(m.value, []).extend(ex_lines)
                ov = await loop.run_in_executor(None, _order_for_market, m, mruns, settings, cache)
                if ov:
                    order_v[m.value] = ov
                fm, fb = await loop.run_in_executor(None, _factor_model_for_market, m, mruns, settings, cache, notes)
                factor_scores.update(fm)
                if fb:
                    factor_bt[m.value] = fb

        # ---- RSMT-Q serving: one batch per market, health checks final before anything is used
        say(tr(settings, "RSMT-Q 분포 예측(시장별 일괄) 중", "RSMT-Q distribution forecasts (one batch per market)"))
        served: dict[str, Any] = {}
        with _Timer(timings, "predict"):
            for mkey, fit_res in fits.items():
                mruns = [r for r in runs if r.market.value == mkey]
                if not mruns:
                    continue
                if fit_res.model is None:
                    notes.append(
                        tr(
                            settings,
                            f"RSMT-Q {mkey} 모델 적합 실패({fit_res.error}) — 해당 시장 종목은 ML 필러 제외",
                            f"RSMT-Q {mkey} fit failed ({fit_res.error}) - ML pillar dropped for this market",
                        )
                    )
                    for r in mruns:
                        _warn(
                            r.md,
                            tr(
                                settings,
                                f"RSMT-Q {mkey} 모델 없음(적합 실패: {fit_res.error}) — ML 필러 제외",
                                f"no RSMT-Q {mkey} model (fit failed: {fit_res.error}) - ML pillar dropped",
                            ),
                        )
                        if r.ticker in excess:
                            _warn(
                                r.md,
                                tr(
                                    settings,
                                    "종목 선택 모델 기대치는 분포(RSMT-Q) 없이 표시만 됨",
                                    "stock-selection forecast shown only (no RSMT-Q distribution to serve it)",
                                ),
                            )
                    continue
                overrides = _overrides_for(mruns, excess)
                out = await loop.run_in_executor(
                    None,
                    functools.partial(
                        fit_res.model.predict_many,
                        [r.ticker for r in mruns],
                        hold_days=settings.fusion.target_horizon_days,
                        overrides=overrides,
                    ),
                )
                served.update(out)
                for t, res in out.items():
                    if t in excess and not isinstance(res, BaseException):
                        excess[t].p_beat_benchmark = float(res[1].prob_outperform)

        # ---- quant signal per ticker
        with _Timer(timings, "quant"):
            kept: list[TickerRun] = []
            for r in runs:
                try:
                    res = build_quant(
                        r.md,
                        settings,
                        None,
                        served.get(r.ticker),
                        excess_model=excess.get(r.ticker),
                        factor_model=factor_scores.get(r.ticker),
                    )
                except Exception as exc:  # noqa: BLE001 - e.g. too little history for the technicals
                    log.exception("%s: quant signal failed", r.ticker)
                    notes.append(
                        tr(
                            settings,
                            f"{r.requested}: 퀀트 신호 산출 실패({type(exc).__name__}: {_short(exc)}) — "
                            "리포트에서 제외",
                            f"{r.requested}: quant signal failed ({type(exc).__name__}: {_short(exc)}) - excluded",
                        )
                    )
                    continue
                r.quant, r.dist = res.signal, res.dist
                kept.append(r)
            runs = kept
        if not runs:
            raise PipelineError(
                tr(settings, "퀀트 신호를 산출한 종목이 없습니다: ", "No ticker produced a quant signal: ")
                + " / ".join(notes)
            )

        # ---- LLM
        if use_llm:
            say(
                tr(
                    settings,
                    f"LLM 분석(뉴스·공시·비평) {len(runs)}종목 진행 중",
                    f"LLM analysis (news, filings, critic) for {len(runs)} tickers",
                )
            )
            with _Timer(timings, "llm"):
                llms = await asyncio.gather(*(_llm_for(r, llm_client, settings) for r in runs))
            for r, li in zip(runs, llms):
                r.llm = li
        else:
            notes.append(tr(settings, NO_LLM_NOTE, NO_LLM_NOTE_EN))
            for r in runs:
                r.llm = LLMIntelligence(skipped=True)

        # ---- fusion, execution plan, barrier guardrail
        with _Timer(timings, "fusion"):
            for r in runs:
                assert r.quant is not None and r.llm is not None
                r.execsig = fuse(r.md, r.quant, r.llm, settings)
                r.plan = build_execution_plan(r.md, r.quant, r.execsig, settings, dist=r.dist, llm=r.llm)
                r.execsig = apply_barrier_guardrail(r.execsig, r.plan, settings)

            macro_regs, sizing_settings = await loop.run_in_executor(None, macro_overlay, runs, settings, cache, notes)
            items = [(r.md, r.plan, r.execsig) for r in runs]
            plans, cash = size_portfolio(items, sizing_settings, capital, base_currency)  # type: ignore[arg-type]
            for r, p in zip(runs, plans):
                r.plan = p
            prisk = portfolio_risk(
                [(r.md, r.plan, r.execsig) for r in runs],  # type: ignore[misc]
                horizon_days=settings.horizon_days,
            )

        with _Timer(timings, "regime+diag"):
            regime = await loop.run_in_executor(None, market_regime, runs, fits, settings, cache, base_currency, notes)
            diagnostics = {
                m: market_diagnostics(f, settings) for m, f in fits.items() if any(r.market.value == m for r in runs)
            }
            for m, lines in diag_extra.items():
                if lines and any(r.market.value == m for r in runs):
                    diagnostics.setdefault(m, []).extend(lines)

        tickers_out = [
            TickerReport(
                instrument=r.md.instrument,
                as_of=r.md.as_of,
                last_price=r.md.last_price,
                fundamentals=r.md.fundamentals,
                quant=r.quant,
                llm=r.llm,
                executive=r.execsig,
                execution=r.plan,
                warnings=list(r.md.warnings),
            )
            for r in runs
        ]
        report = PortfolioReport(
            generated_at=datetime.now(KST),
            base_currency=base_currency,
            capital=capital,
            tickers=tickers_out,
            cash_weight=cash,
            market_regime=regime,
            model_diagnostics=diagnostics,
            portfolio_risk=prisk,
            macro=macro_regs,
            factor_backtest=factor_bt,
            order_ventile=order_v,
            llm_cost_usd=round(sum(r.llm.cost_usd for r in runs if r.llm is not None), 6),
            config_digest=settings.digest(),
            notes=notes,
        )
        timings["build_total"] = round(time.perf_counter() - t_start, 3)
        summary = getattr(llm_client, "usage_summary", None) if use_llm else None
        usage = summary() if callable(summary) else {}
        return RunResult(report=report, runs=runs, fits=fits, timings=timings, llm_usage=usage)
    finally:
        if own_client:
            try:
                await llm_client.aclose()
            except Exception as exc:  # noqa: BLE001
                log.warning("LLM client close failed: %s", exc)


# =============================================================================== stock selection / factor model


def _order_for_market(
    market: Market, mruns: Sequence[TickerRun], settings: Settings, cache: DiskCache
) -> list[dict[str, Any]]:
    """The reference ventile views, one per stored model (docs/MODELING.md 93, 94.3; empty without an artifact,
    never raises - it is display only and sets nothing)."""
    try:
        from quantcio.modeling.serve import order_for_report

        return order_for_report(market, [(r.ticker, r.md) for r in mruns], settings, cache)
    except Exception:  # noqa: BLE001 - reference only: the report just goes without it
        log.exception("ordering model %s failed", market.value)
        return []


def _excess_for_market(
    market: Market, mruns: Sequence[TickerRun], settings: Settings, cache: DiskCache, notes: list[str]
) -> tuple[dict[str, ExcessForecast], list[str]]:
    """ExcessForecast per ticker from the cached serving artifact of ``quantcio modeling select`` (never raises;
    no artifact = the legacy RSMT-Q mean with a note) + diagnostics lines for the market."""
    try:
        from quantcio.modeling.serve import score_for_report

        out, mnotes, art = score_for_report(market, [(r.ticker, r.md) for r in mruns], settings, cache)
    except Exception as exc:  # noqa: BLE001 - the legacy path serves, loudly
        log.exception("stock-selection model %s failed", market.value)
        notes.append(
            tr(
                settings,
                f"종목 선택 모델({market.value}) 점수화 실패({type(exc).__name__}: {_short(exc, 80)}) — 기존 RSMT-Q "
                "평균 사용",
                f"Stock-selection model ({market.value}) failed ({type(exc).__name__}: {_short(exc, 80)}) - legacy "
                "RSMT-Q mean used",
            )
        )
        return {}, []
    notes.extend(mnotes)
    served = [out[r.ticker] for r in mruns if out.get(r.ticker) is not None]
    # a note every served ticker of this market carries (e.g. "skill gate failed") is a market-level fact:
    # state it once in the report notes instead of repeating it in every ticker's warnings
    common = set(served[0].notes).intersection(*(fc.notes for fc in served[1:])) if len(served) > 1 else set()
    for n in (x for x in (served[0].notes if served else []) if x in common):
        notes.append(
            tr(settings, f"종목 선택 모델({market.value}): {n}", f"Stock-selection model ({market.value}): {n}")
        )
    for r in mruns:
        fc = out.get(r.ticker)
        if fc is None:
            continue
        for n in fc.notes:
            if n not in common:
                _warn(r.md, tr(settings, f"종목 선택 모델: {n}", f"stock-selection model: {n}"))
    lines: list[str] = []
    if art is not None:
        m = art.metrics
        oa, ha = m.get("outer_auc") or {}, m.get("holdout_auc") or {}
        mm = art.market_model or {}
        verdict = tr(settings, "통과", "passed") if art.skill_ok else tr(settings, "미달", "not passed")
        # modeling protocol v2: the served verdict (incl. the XSH confirmation) and which AUC label it is
        verdict = tr(settings, m.get("gate_verdict_ko", verdict), m.get("gate_verdict_en", verdict))
        auc_ko, auc_en = m.get("auc_label_ko", "날짜 내 AUC"), m.get("auc_label_en", "within-date AUC")

        def f3(x: Any) -> str:
            return "n/a" if x is None else f"{x:.3f}"

        def pa(x: Any) -> str:  # a per-month excess, shown annualised
            return "n/a" if x is None else f"{x * 12:+.1%}"

        op, hp = m.get("outer_primary") or {}, m.get("holdout_primary") or {}
        money_ko = (
            f"상위 20% 베타 조정 순초과 연 {pa(op.get('mean'))} [{pa(op.get('lo'))}, {pa(op.get('hi'))}] · 홀드아웃 "
            f"{pa(hp.get('mean'))} · "
            if op
            else ""
        )
        money_en = (
            f"top-quintile beta-adjusted net excess {pa(op.get('mean'))}/yr [{pa(op.get('lo'))}, {pa(op.get('hi'))}] "
            f"· holdout {pa(hp.get('mean'))} · "
            if op
            else ""
        )
        lines.append(
            tr(
                settings,
                f"종목 선택 모델({art.as_of.date()}): {art.label_ko} [{art.selected}] · {money_ko}"
                f"{auc_ko} {f3(oa.get('mean'))} "
                f"[{f3(oa.get('lo'))}, {f3(oa.get('hi'))}] (워크포워드 표본 외) · 홀드아웃 {f3(ha.get('mean'))} · "
                f"스킬 게이트 {verdict} · 초과수익 배수 {art.multiplier:.2f}",
                f"Stock-selection model ({art.as_of.date()}): {art.label_en} [{art.selected}] · {money_en}{auc_en} "
                f"{f3(oa.get('mean'))} [{f3(oa.get('lo'))}, {f3(oa.get('hi'))}] (walk-forward OOS) · holdout "
                f"{f3(ha.get('mean'))} · skill gate {verdict} · excess multiplier {art.multiplier:.2f}",
            )
        )
        p_up = mm.get("served_p_up")
        lines.append(
            tr(
                settings,
                f"시장 방향 요소: {'로지스틱 모델' if mm.get('use_model') else '기준률(모델 스킬 미확인)'} "
                f"P(시장 상승) {f3(p_up)} — 분포의 시장 다리는 "
                f"{'모델 확률에 맞춤' if mm.get('use_model') else '사전 축소 드리프트 유지'}",
                f"Market component: {'logistic model' if mm.get('use_model') else 'base rate (no model skill)'} "
                f"P(market up) {f3(p_up)} - the distribution's market leg "
                f"{'matches the model probability' if mm.get('use_model') else 'keeps the prior-shrunk drift'}",
            )
        )
    return out, lines


def _overrides_for(mruns: Sequence[TickerRun], excess: dict[str, ExcessForecast]) -> dict[str, Any]:
    from quantcio.modeling.serve import excess_override

    return {r.ticker: excess_override(excess[r.ticker]) for r in mruns if r.ticker in excess}


def _factor_model_for_market(
    market: Market, mruns: Sequence[TickerRun], settings: Settings, cache: DiskCache, notes: list[str]
) -> tuple[dict[str, FactorModelScore], dict[str, Any]]:
    """QuantSignal.factor_model per ticker and PortfolioReport.factor_backtest[market] from the cached factor
    backtest (``quantcio backtest``); never blocks the report (a missing cache is a note)."""
    try:
        from quantcio.backtest import load_cached_backtest, report_summary, score_tickers

        res = load_cached_backtest(market, settings, cache)
        if res is None:
            notes.append(
                tr(
                    settings,
                    f"팩터 백테스트 캐시({market.value}) 없음 — `quantcio backtest --market {market.value}` "
                    "실행 전까지 팩터 모델 점수 생략",
                    f"No cached factor backtest ({market.value}) - run `quantcio backtest --market {market.value}`; "
                    "factor-model scores omitted",
                )
            )
            return {}, {}
        scores = score_tickers(
            market,
            [r.ticker for r in mruns],
            settings,
            cache,
            prices={r.ticker: r.md.prices for r in mruns},
            result=res,
        )
        return scores, report_summary(res)
    except Exception as exc:  # noqa: BLE001 - optional input, reported
        log.exception("factor model %s failed", market.value)
        notes.append(
            tr(
                settings,
                f"팩터 모델({market.value}) 점수화 실패({type(exc).__name__}: {_short(exc, 80)}) — 생략",
                f"Factor model ({market.value}) failed ({type(exc).__name__}: {_short(exc, 80)}) - omitted",
            )
        )
        return {}, {}


# =============================================================================== macro overlay

MACRO_REGION = {"KR": "kr", "US": "us"}


def macro_overlay(
    runs: Sequence[TickerRun], settings: Settings, cache: DiskCache, notes: list[str]
) -> tuple[list[MacroRegime], Settings]:
    """Current global/US/KR regimes and the sizing settings with max_gross scaled by the overlay.

    The multiplier is the mean of the market regimes' ``exposure_mult`` over the Buy/Strong Buy candidates (the only
    names that can get weight). Failures never block the report: they are noted and no overlay is applied.
    """
    if not settings.macro.enabled:
        return [], settings
    try:
        from quantcio.macro import current_regimes

        regimes = current_regimes(settings, cache)
    except Exception as exc:  # noqa: BLE001 - macro is an overlay, the report must still render
        log.warning("macro regimes failed: %s", exc)
        notes.append(
            tr(
                settings,
                f"매크로 레짐 수집 실패({type(exc).__name__}: {_short(exc, 80)}) — 오버레이 미적용",
                f"Macro regimes unavailable ({type(exc).__name__}: {_short(exc, 80)}) - no overlay",
            )
        )
        return [], settings
    for r in regimes:
        for w in r.warnings:
            notes.append(tr(settings, f"매크로({r.region}): {w}", f"Macro ({r.region}): {w}"))
    if not settings.macro.overlay:
        return regimes, settings
    by_region = {r.region: r for r in regimes}
    buyish = [r for r in runs if r.execsig is not None and r.execsig.signal in (Signal.BUY, Signal.STRONG_BUY)]
    mults = [
        by_region[MACRO_REGION[r.market.value]].exposure_mult
        for r in buyish
        if MACRO_REGION.get(r.market.value) in by_region
    ]
    mult = float(sum(mults) / len(mults)) if mults else 1.0
    if mult >= 0.999:
        return regimes, settings
    sized = settings.model_copy(deep=True)
    sized.risk.max_gross = round(settings.risk.max_gross * mult, 4)
    detail = " · ".join(
        f"{tr(settings, MACRO_REGION_LABELS.get(k, k), k.upper())} "
        f"{tr(settings, by_region[k].label, MACRO_LABELS_EN.get(by_region[k].label, by_region[k].label))}"
        f"({by_region[k].score:+.2f})"
        for k in ("kr", "us")
        if k in by_region and by_region[k].score is not None
    )
    notes.append(
        tr(
            settings,
            f"매크로 오버레이: 총 투자 한도 {settings.risk.max_gross:.0%} → {sized.risk.max_gross:.0%} "
            f"(배수 {mult:.2f}; {detail}) — 백테스트상 수익 개선 근거는 없고 리스크 관리용",
            f"Macro overlay: gross cap {settings.risk.max_gross:.0%} → {sized.risk.max_gross:.0%} "
            f"(multiplier {mult:.2f}; {detail}) - risk management only, no backtested return improvement",
        )
    )
    return regimes, sized


MACRO_REGION_LABELS = {"global": "글로벌", "us": "미국", "kr": "한국"}


# =============================================================================== render


def _charts(runs: Sequence[TickerRun]) -> dict[str, str]:
    from quantcio.report import price_chart_svg

    out: dict[str, str] = {}
    for r in runs:
        try:
            out[r.ticker] = price_chart_svg(r.md.prices, r.plan, r.execsig.target_price if r.execsig else None)
        except Exception as exc:  # noqa: BLE001 - the report shows "차트 없음" for a missing chart
            log.warning("%s: price chart failed: %s", r.ticker, exc)
    return out


def report_dir(settings: Settings, out_dir: str | Path | None, when: datetime) -> Path:
    """<out_dir or settings.report.output_dir>/<YYYYMMDD_HHMM>/ (created)."""
    root = Path(out_dir) if out_dir is not None else Path(settings.report.output_dir)
    d = root / when.astimezone(KST).strftime("%Y%m%d_%H%M")
    d.mkdir(parents=True, exist_ok=True)
    return d


async def run_report(
    tickers: str | Sequence[str],
    settings: Settings,
    capital: float | None = None,
    base_currency: str = "KRW",
    use_llm: bool = True,
    out_dir: str | Path | None = None,
    llm_client: Any = None,
    *,
    cache: DiskCache | None = None,
    use_model_cache: bool = True,
    progress: Callable[[str], None] | None = None,
) -> ReportArtifacts:
    """Build the report, render it to an A4 PDF (<= report.max_pages) and write the JSON next to it.

    The PDF is rendered in a watchdog subprocess (`report.worker.fit_to_pages_isolated`): a renderer that hangs or
    crashes is killed after `RENDER_TIMEOUT_S`, retried once, then replaced by the HTML file (pages=0, warning)."""
    from quantcio.report.worker import fit_to_pages_isolated

    result = await build_report(
        tickers,
        settings,
        capital,
        base_currency,
        use_llm,
        llm_client,
        cache=cache,
        use_model_cache=use_model_cache,
        progress=progress,
    )
    report, timings = result.report, result.timings
    folder = report_dir(settings, out_dir, report.generated_at)
    stem = f"quantcio_report_{report.generated_at.astimezone(KST):%Y%m%d_%H%M}"
    (progress or (lambda _m: None))(tr(settings, "PDF 렌더링 중", "Rendering the PDF"))
    t0 = time.perf_counter()
    charts = await asyncio.to_thread(_charts, result.runs)
    fit = await asyncio.to_thread(fit_to_pages_isolated, report, settings, folder / f"{stem}.pdf", charts=charts)
    timings["render"] = round(time.perf_counter() - t0, 3)
    warnings = list(fit.warnings)
    for w in warnings:
        log.warning("render: %s", w)
    json_path = folder / f"{stem}.json"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if settings.holdout.record_reports:
        warnings.extend(_record_forward_ledger(result, settings))
    return ReportArtifacts(
        json_path=json_path,
        html_path=fit.html_path,
        pdf_path=fit.pdf_path,
        pages=fit.pages,
        report=report,
        timings=timings,
        warnings=warnings,
        llm_usage=result.llm_usage,
    )


def _record_forward_ledger(result: RunResult, settings: Settings) -> list[str]:
    """Append the report's predictions to the forward ledger (docs/HOLDOUT.md); never raises.

    Returns Korean run-level warnings: a failed write, or a prediction recorded as not valid for the
    registered forward holdout (config / code changed since the registration).
    """
    try:
        from quantcio import tracking

        models = {m: f.model for m, f in result.fits.items() if f.model is not None}
        rec = tracking.record_predictions(result.report, settings, source="report", models=models)
    except Exception as exc:  # noqa: BLE001 - the ledger must never fail the report
        log.warning("forward ledger write failed: %s", exc, exc_info=True)
        return [
            tr(
                settings,
                f"전진 추적 원장 기록 실패({type(exc).__name__}: {_short(exc)}) — 리포트는 정상 생성",
                f"Forward ledger write failed ({type(exc).__name__}: {_short(exc)}) - the report is complete",
            )
        ]
    return [n for n in rec.notes if "valid=false" in n or "실패" in n]


# =============================================================================== rsmt --tune


@dataclass
class TuneRow:
    param: str
    value: float
    current: bool
    metrics: dict[str, float]
    note: str = ""
    skill_ok: bool = False  # the full CV skill gate (AUC gain, log loss, date-block bound)


@dataclass
class TuneReport:
    market: str
    rows: list[TuneRow]
    recommendations: list[str]


TUNE_GRID: dict[str, tuple[float, ...]] = {
    "nu_min": (3.5, 4.5, 6.0, 8.0),  # tail floor -> judged on CRPS / cover80 (never on direction AUC)
    "alpha": (0.01, 0.05, 0.10),  # split-test level -> judged on log loss
    "min_leaf": (100, 150, 250),  # leaf size -> judged on log loss
}
TUNE_TOL = {"crps": 2e-4, "logloss": 5e-4}  # smaller differences are noise: keep the current value


def tune_rsmt(
    panel_data: PanelData,
    settings: Settings,
    grid: dict[str, Sequence[float]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> TuneReport:
    """One-parameter-at-a-time CV grid; each parameter is judged on the metric it actually moves.

    nu_min on CRPS (tie-break |cover80 - 0.80|), alpha and min_leaf on the CV log loss. Prints
    recommendations only: the config is never changed. A difference within TUNE_TOL keeps the current
    value (it is inside the CV noise).
    """
    from quantcio.quant.rsmt import cross_validate

    market = panel_data.market.value
    grid = {k: tuple(v) for k, v in (grid or TUNE_GRID).items()}
    cur = settings.rsmt.markets[market]
    rows: list[TuneRow] = []
    recs: list[str] = []
    for param, values in grid.items():
        current = float(getattr(cur, param))
        prow: list[TuneRow] = []
        for v in values:
            s = settings.model_copy(deep=True)
            setattr(s.rsmt.markets[market], param, int(v) if param == "min_leaf" else float(v))
            (progress or (lambda _m: None))(
                tr(settings, f"{param}={v:g} 교차검증 중", f"{param}={v:g}: cross-validating")
            )
            cv = cross_validate(panel_data, s)
            m = dict(cv.metrics)
            note = "; ".join(cv.notes)
            prow.append(TuneRow(param, float(v), math.isclose(float(v), current), m, note, bool(cv.skill_ok)))
        rows.extend(prow)
        ok = [r for r in prow if r.metrics]
        if not ok:
            recs.append(tr(settings, f"{param}: CV 불가 — 권장값 없음", f"{param}: CV unavailable - no recommendation"))
            continue
        metric = "crps" if param == "nu_min" else "logloss"
        best = min(ok, key=lambda r: (r.metrics[metric], abs(r.metrics.get("cover80", 0.8) - 0.8)))
        cur_row = next((r for r in ok if r.current), None)
        label = "CRPS" if metric == "crps" else "logloss"
        if cur_row is None:
            recs.append(
                tr(
                    settings,
                    f"{param}: 현재값 {current:g}이 그리드에 없음 → 그리드 최적 {best.value:g} "
                    f"({label} {best.metrics[metric]:.4f})",
                    f"{param}: current value {current:g} not on the grid -> grid optimum {best.value:g} "
                    f"({label} {best.metrics[metric]:.4f})",
                )
            )
        elif best is cur_row or cur_row.metrics[metric] - best.metrics[metric] <= TUNE_TOL[metric]:
            diff = cur_row.metrics[metric] - best.metrics[metric]
            recs.append(
                tr(
                    settings,
                    f"{param}: 현재값 {current:g} 유지 ({label} {cur_row.metrics[metric]:.4f}; "
                    f"최적 {best.value:g}와 차이 {diff:.4f} ≤ 허용오차 {TUNE_TOL[metric]:g})",
                    f"{param}: keep {current:g} ({label} {cur_row.metrics[metric]:.4f}; gap to optimum {best.value:g} "
                    f"{diff:.4f} <= tolerance {TUNE_TOL[metric]:g})",
                )
            )
        else:
            extra = f", cover80 {best.metrics.get('cover80', float('nan')):.2f}" if metric == "crps" else ""
            recs.append(
                tr(
                    settings,
                    f"{param}: {current:g} → {best.value:g} 권장 ({label} {cur_row.metrics[metric]:.4f} → "
                    f"{best.metrics[metric]:.4f}{extra})",
                    f"{param}: {current:g} → {best.value:g} recommended ({label} {cur_row.metrics[metric]:.4f} → "
                    f"{best.metrics[metric]:.4f}{extra})",
                )
            )
    return TuneReport(market=market, rows=rows, recommendations=recs)
