"""QuantCIO command line (typer + rich, Korean output).

Commands
--------
- ``quantcio report --tickers 005930,000660,AAPL [--capital N] [--base KRW|USD] [--no-llm]
  [--config f] [--out d] [--open] [--lang ko|en] [--no-record]``
- ``quantcio analyze TICKER [--json] [--no-llm] [--lang ko|en]``
- ``quantcio rsmt --market KR|US [--tune]``
- ``quantcio evaluate --market KR|US [--universe N] [--years Y] [--folds 3,4,5,6] [--out d] [--no-ml]``
- ``quantcio doctor [--lang ko|en]``
- ``quantcio holdout register --kind forward|xs [--market KR|US] [--allow-dirty]`` / ``holdout status`` /
  ``holdout evaluate --id ID [--force]`` (docs/HOLDOUT.md)
- ``quantcio track snapshot --market KR|US [--n N]`` / ``track score`` / ``track report [--holdout ID] [--json]``

Logs go to ``<cache_dir>/logs/quantcio_<YYYYMMDD>.log`` (INFO); the console shows errors only unless ``-v``.
Every degraded path is still printed: ticker warnings, report notes and render warnings are listed after
each run. ``--lang`` (report / analyze / doctor) overrides ``report.language`` from the config: the PDF, the LLM
outputs and this command's console summary are written in that language (default Korean).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from quantcio.config import PROJECT_ROOT, Settings, load_settings
from quantcio.i18n import LANGS, lang_of, tr
from quantcio.models import Market, PortfolioReport, Signal, TickerReport

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="QuantCIO — 정량 신호 + LLM 비정형 인텔리전스로 만드는 CIO 매매 타이밍 리포트 (KR·US)",
)
console = Console()
err_console = Console(stderr=True)

SIGNAL_STYLE = {
    Signal.STRONG_BUY: ("강력 매수", "bold green"),
    Signal.BUY: ("매수", "green"),
    Signal.HOLD: ("관망", "yellow"),
    Signal.SELL: ("매도", "red"),
}
# fusion/signal.py FACTOR_BLEND_PREFIX / FACTOR_BLEND_PREFIX_EN (not imported: the CLI stays light to import)
BLEND_NOTE_PREFIXES = ("팩터 필러 블렌드:", "Factor pillar blend:", "ML 필러 블렌드:", "ML pillar blend:")
# console language of the running command (set by _settings from --lang / report.language)
_LANG = "ko"
LANG_HELP = "리포트·LLM 출력 언어: ko 또는 en (기본: 설정의 report.language)"


def _t(ko: str, en: str) -> str:
    return tr(_LANG, ko, en)


NOISY_LOGGERS = ("yfinance", "urllib3", "matplotlib", "asyncio", "httpx", "httpcore", "peewee", "filelock")


# ============================================================================== setup


class _ConsoleFormatter(logging.Formatter):
    """Console log line: the message plus a one-line exception summary, no traceback.

    Expected failures (a mistyped ticker, one dead source) are logged with ``log.exception`` so the log
    file keeps the full traceback; on the console that was ~100 lines per bad ticker, burying the run's
    own summary (which already reports each failure). ``-v`` shows the full traceback again.
    """

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        exc = record.exc_info[1] if record.exc_info else None
        if exc is not None:
            msg += f" ({type(exc).__name__}: {' '.join(str(exc).split())[:200]}) — 상세 트레이스백은 로그 파일 참조"
        return msg


def _setup_logging(settings: Settings, verbose: bool) -> Path:
    log_dir = settings.cache_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"quantcio_{datetime.now():%Y%m%d}.log"
    root = logging.getLogger()
    for h in list(root.handlers):  # only our own handlers from an earlier call (never a host's handlers)
        if getattr(h, "_quantcio", False):
            root.removeHandler(h)
            h.close()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"))
    ch = RichHandler(console=err_console, show_path=False, rich_tracebacks=False, markup=False)
    ch.setLevel(logging.INFO if verbose else logging.ERROR)
    if not verbose:
        ch.setFormatter(_ConsoleFormatter())
    for h in (fh, ch):
        h._quantcio = True  # type: ignore[attr-defined]
        root.addHandler(h)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.captureWarnings(True)
    return log_file


def _settings(config: Path | None, lang: str | None = None) -> Settings:
    """Load the config; `lang` ("ko" / "en", from --lang) overrides report.language. Bad input exits with 2."""
    global _LANG
    if lang is not None and lang.strip().lower() not in LANGS:
        err_console.print(
            f"[red]--lang은 {' 또는 '.join(LANGS)}만 가능합니다 (입력: {lang!r}) / --lang must be ko or en[/red]"
        )
        raise typer.Exit(2)
    try:
        settings = load_settings(config)
    except FileNotFoundError as exc:
        err_console.print(f"[red]설정 파일을 찾을 수 없습니다: {exc}[/red]")
        raise typer.Exit(2) from exc
    except Exception as exc:  # noqa: BLE001 - pydantic / yaml errors
        err_console.print(f"[red]설정 오류: {type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(2) from exc
    if lang is not None:
        settings.report.language = lang.strip().lower()  # type: ignore[assignment]
    _LANG = lang_of(settings)
    return settings


# ============================================================================== formatting


def _money(v: float | None, ccy: str) -> str:
    from quantcio.fusion.risk import format_money

    return format_money(v, ccy, _LANG)


def _pct(v: float | None, digits: int = 1, signed: bool = True) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    return f"{v * 100:+.{digits}f}%" if signed else f"{v * 100:.{digits}f}%"


def _signal(sig: Signal) -> str:
    ko, style = SIGNAL_STYLE[sig]
    if _LANG == "en":
        return f"[{style}]{sig.value}[/{style}]"
    return f"[{style}]{sig.value} ({ko})[/{style}]"


def _summary_table(report: PortfolioReport) -> Table:
    tb = Table(title="Executive Signal Matrix", show_lines=False, header_style="bold")
    for col, just in (
        (_t("종목", "Name"), "left"),
        (_t("신호", "Signal"), "left"),
        (_t("신뢰도", "Confidence"), "right"),
        (_t("현재가", "Price"), "right"),
        (_t("목표가(여력)", "Target (upside)"), "right"),
        (_t("P(상승) 20D", "P(up) 20D"), "right"),
        (_t("비중", "Weight"), "right"),
        (_t("손절 / 익절", "Stop / take-profit"), "right"),
        (_t("익절·시간·손절 선도달", "TP · time · SL first"), "right"),
    ):
        tb.add_column(col, justify=just)  # type: ignore[arg-type]
    for t in report.tickers:
        ccy = t.instrument.currency
        ex, pl = t.executive, t.execution
        d = t.quant.distribution
        pup = d.prob_up if d is not None else None
        b = pl.barrier
        bar = (
            f"{b.p_take_profit:.0%} · {b.p_time_exit:.0%} · {b.p_stop_loss:.0%} ({b.horizon_days}{_t('일', 'D')})"
            if b is not None
            else "—"
        )
        tb.add_row(
            f"{t.instrument.name}\n[dim]{t.instrument.ticker}[/dim]",
            _signal(ex.signal),
            f"{ex.confidence}",
            _money(t.last_price, ccy),
            f"{_money(ex.target_price, ccy)}\n[dim]{_pct(ex.upside)}[/dim]",
            "—" if pup is None else f"{pup:.0%}",
            _pct(pl.weight, 1, signed=False),
            f"{_money(pl.stop_loss, ccy)} ({_pct(pl.stop_loss_pct)})\n"
            f"{_money(pl.take_profit, ccy)} ({_pct(pl.take_profit_pct)})",
            bar,
        )
    return tb


def _print_warnings(report: PortfolioReport, extra: list[str] = ()) -> None:
    """Report notes, then every ticker warning, guardrail/adjustment override and LLM error.

    Overrides and LLM errors are printed even when no ticker has a data warning (they used to be nested
    under the warning count, so a run whose only alarms were guardrail downgrades printed none of them).
    In a run-wide --no-llm run the identical 'LLM 분석 미실행' override is printed once, not per ticker.
    """
    run_skipped = bool(report.tickers) and all(t.llm.skipped for t in report.tickers)
    common = (
        {
            o
            for t in report.tickers
            for o in t.executive.overrides
            if o.startswith(("LLM 분석 미실행", "LLM analysis not run"))
        }
        if run_skipped
        else set()
    )
    # the factor-pillar blend note is information (shown in the PDF's factor-model table), not a guardrail
    common |= {o for t in report.tickers for o in t.executive.overrides if o.startswith(BLEND_NOTE_PREFIXES)}
    notes = [
        *report.notes,
        *(f"{_t('전 종목 공통', 'All names')}: {o}" for o in sorted(common) if not o.startswith(BLEND_NOTE_PREFIXES)),
        *extra,
    ]
    if notes:
        console.print(f"[bold]{_t('리포트 알림', 'Report notes')}[/bold]")
        for note in notes:
            console.print(f"  • {note}", markup=False)
    n = sum(len(t.warnings) for t in report.tickers)
    n_ov = sum(1 for t in report.tickers for o in t.executive.overrides if o not in common)
    n_err = sum(len(t.llm.errors) for t in report.tickers)
    if n or n_ov or n_err:
        console.print(
            "[bold yellow]"
            + _t(
                f"종목별 경고 {n}건 · 가드레일/조정 {n_ov}건 · LLM 오류 {n_err}건",
                f"Ticker warnings {n} · guardrails/adjustments {n_ov} · LLM errors {n_err}",
            )
            + "[/bold yellow]"
        )
    for t in report.tickers:
        tk = t.instrument.ticker
        for w in t.warnings:
            console.print(f"  [yellow]{tk}[/yellow] ", end="")
            console.print(w, markup=False)
        for o in t.executive.overrides:
            if o not in common:
                console.print(f"  [magenta]{tk} {_t('가드레일/조정', 'guardrail/adjustment')}[/magenta] ", end="")
                console.print(o, markup=False)
        for e in t.llm.errors:
            console.print(f"  [red]{tk} {_t('LLM 오류', 'LLM error')}[/red] ", end="")
            console.print(e, markup=False)


def _print_llm_usage(usage: dict[str, dict[str, float]]) -> None:
    """One line per LLM task: calls (response-cache hits), input / cache-read / output (thinking) tokens, USD."""
    if not usage:
        return
    names = {"news": _t("뉴스", "news"), "filings": _t("공시", "filings"), "critic": _t("비평", "critic")}
    names["total"] = _t("합계", "total")
    order = [k for k in ("news", "filings", "critic", "web_research") if k in usage]
    order += [k for k in usage if k not in (*order, "total")] + ["total"]
    parts = []
    for k in order:
        u = usage.get(k)
        if not u:
            continue
        parts.append(
            f"{names.get(k, k)} {int(u['calls'])}{_t('회', ' calls')}"
            + (f"({_t('캐시', 'cached')} {int(u['cached'])})" if u.get("cached") else "")
            + f" {_t('입력', 'in')} {int(u['input_tokens']):,}"
            + (
                f"+{_t('캐시읽기', 'cache')} {int(u['cache_read_input_tokens']):,}"
                if u.get("cache_read_input_tokens")
                else ""
            )
            + (
                f"+{_t('캐시쓰기', 'cache write')} {int(u['cache_creation_input_tokens']):,}"
                if u.get("cache_creation_input_tokens")
                else ""
            )
            + f" · {_t('출력', 'out')} {int(u['output_tokens']):,}"
            + (f"({_t('생각', 'thinking')} {int(u['thinking_tokens']):,})" if u.get("thinking_tokens") else "")
            + f" · ${u['cost_usd']:.3f}"
        )
    console.print(_t("LLM 토큰: ", "LLM tokens: ") + " | ".join(parts), markup=False)


def _print_timings(timings: dict[str, float]) -> None:
    parts = " · ".join(f"{k} {v:.1f}s" for k, v in timings.items())
    console.print(f"[dim]{_t('소요 시간', 'Elapsed')}: {parts}[/dim]")


def _progress_printer() -> Any:
    t0 = time.perf_counter()

    def say(msg: str) -> None:
        console.print(f"[dim]{time.perf_counter() - t0:6.1f}s[/dim] {msg}")

    return say


# ============================================================================== commands


# ============================================================================== report: which tickers

WATCHLIST_DEFAULT = PROJECT_ROOT / "config" / "watchlist.txt"
MAX_PICKS_PER_MARKET = 20  # --top / --bottom upper bound per market


class TickerSelectionError(ValueError):
    """No usable ticker list (Korean + English message; exit code 2)."""


def read_watchlist(path: Path) -> list[str]:
    """Tickers of a watchlist file: separated by commas / spaces / new lines; ``#`` starts a comment.

    Raises:
        TickerSelectionError: the file does not exist or cannot be read.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise TickerSelectionError(
            _t(f"관심 종목 파일을 읽을 수 없습니다: {path} ({exc})", f"cannot read the watchlist file: {path} ({exc})")
        ) from exc
    out: list[str] = []
    for line in text.splitlines():
        out.extend(t for t in line.split("#", 1)[0].replace(",", " ").split() if t.strip())
    return out


def model_picks(
    markets: list[Market], top: int, bottom: int, settings: Settings, cache: Any
) -> tuple[list[dict[str, Any]], list[str]]:
    """Top / bottom names of each market by the stock-selection model's latest score (``modeling select``'s cached
    serving artifact). Returns (picks with ticker, name, market, side, percentile, as_of; notes to print).

    Raises:
        TickerSelectionError: a market has no serving artifact (run ``quantcio modeling select`` first).
    """
    from quantcio.data.base import normalise_ticker
    from quantcio.modeling.serve import STALE_DAYS, load_artifact, served_state

    picks: list[dict[str, Any]] = []
    notes: list[str] = []
    for mk in markets:
        art = load_artifact(mk, settings, cache)
        if art is None or art.latest is None or art.latest.empty:
            raise TickerSelectionError(
                _t(
                    f"{mk.value} 종목 선택 모델 산출물이 없습니다 — `quantcio modeling select --market {mk.value}`를 "
                    "먼저 실행하세요",
                    f"no {mk.value} stock-selection artifact - run "
                    f"`quantcio modeling select --market {mk.value}` first",
                )
            )
        lat = art.latest.dropna(subset=["score"]).sort_values(["score", "ticker"], ascending=[False, True])
        chosen = [("top", r) for r in lat.head(top).itertuples(index=False)] if top else []
        if bottom:
            low = lat.iloc[::-1].head(bottom)  # lowest score first
            taken = {r.ticker for _, r in chosen}
            chosen += [("bottom", r) for r in low.itertuples(index=False) if r.ticker not in taken]
        for side, r in chosen:
            picks.append(
                {
                    "ticker": normalise_ticker(str(r.ticker)),
                    "name": str(getattr(r, "name", "") or ""),
                    "market": mk.value,
                    "side": side,
                    "percentile": float(getattr(r, "percentile", float("nan"))),
                    "as_of": str(art.as_of.date()),
                }
            )
        state = served_state(art, settings)
        if not state["skill_ok"]:
            notes.append(
                _t(
                    f"{mk.value}: 종목 선택 모델이 스킬 미확인(서빙 ×{state['multiplier']:.2f}) — 상위·하위 선정은 "
                    "참고용입니다",
                    f"{mk.value}: the stock-selection model has no confirmed skill (served x{state['multiplier']:.2f}) "
                    "- the top/bottom picks are for reference only",
                )
            )
        age = (date.today() - art.as_of.date()).days
        if age > STALE_DAYS:
            notes.append(
                _t(
                    f"{mk.value}: 모델 산출물 기준일 {art.as_of.date()} ({age}일 전) — "
                    "`quantcio modeling select` 재실행 권장",
                    f"{mk.value}: artifact as of {art.as_of.date()} ({age} days old) - "
                    "rerun `quantcio modeling select`",
                )
            )
    return picks, notes


def resolve_report_tickers(
    tickers: str | None,
    top: int,
    bottom: int,
    markets: str,
    watchlist: Path | None,
    settings: Settings,
    cache: Any = None,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """The report's ticker list: ``--tickers`` + ``--watchlist`` + model picks (``--top`` / ``--bottom`` per market
    in ``--markets``), de-duplicated in that order. With none of them the default watchlist
    (``config/watchlist.txt``) is used. Returns (tickers, picks, notes).

    Raises:
        TickerSelectionError: bad options, a missing artifact / file, or an empty list.
    """
    from quantcio.pipeline import parse_tickers

    for label, n in (("--top", top), ("--bottom", bottom)):
        if not 0 <= int(n) <= MAX_PICKS_PER_MARKET:
            raise TickerSelectionError(
                _t(
                    f"{label}는 0~{MAX_PICKS_PER_MARKET} 사이여야 합니다: {n}",
                    f"{label} must be 0-{MAX_PICKS_PER_MARKET}: {n}",
                )
            )
    mks: list[Market] = []
    for m in markets.replace(",", " ").split():
        try:
            mk = Market(m.strip().upper())
        except ValueError as exc:
            raise TickerSelectionError(
                _t(f"--markets 는 KR·US 중에서: {m}", f"--markets takes KR and/or US: {m}")
            ) from exc
        if mk not in mks:
            mks.append(mk)
    notes: list[str] = []
    explicit = parse_tickers(tickers) if tickers else []
    listed = read_watchlist(watchlist) if watchlist is not None else []
    picks: list[dict[str, Any]] = []
    if top or bottom:
        if not mks:
            raise TickerSelectionError(_t("--markets 가 비어 있습니다", "--markets is empty"))
        from quantcio.data.cache import DiskCache

        picks, pnotes = model_picks(mks, int(top), int(bottom), settings, cache or DiskCache(settings.cache_path))
        notes.extend(pnotes)
    if not (explicit or listed or picks or top or bottom or watchlist is not None):
        if not WATCHLIST_DEFAULT.exists():
            raise TickerSelectionError(
                _t(
                    "종목을 지정하세요: -t 005930,AAPL · --top 5 (모델 상위) · --watchlist 파일 · 또는 "
                    f"{WATCHLIST_DEFAULT}에 종목 작성",
                    "choose tickers: -t 005930,AAPL · --top 5 (model top) · --watchlist FILE · or list them in "
                    f"{WATCHLIST_DEFAULT}",
                )
            )
        listed = read_watchlist(WATCHLIST_DEFAULT)
        notes.append(_t(f"관심 종목 파일 사용: {WATCHLIST_DEFAULT}", f"using the watchlist file {WATCHLIST_DEFAULT}"))
    out = parse_tickers([*explicit, *listed, *(p["ticker"] for p in picks)])
    if not out:
        raise TickerSelectionError(
            _t(
                "리포트할 종목이 없습니다(관심 종목 파일이 비었거나 모델 선정 결과 없음)",
                "no ticker to report (empty watchlist or no model pick)",
            )
        )
    cmn = int(settings.report.compact_max_names)
    if len(out) > cmn:
        pages_txt = (
            _t(
                f"내용 생략 없이 쪽수를 늘려 출력(최대 {settings.report.max_pages_extended}쪽, 1분 판단용 "
                f"{settings.report.max_pages}쪽 형식은 {cmn}개 이하)",
                f"printed in full on more pages (up to {settings.report.max_pages_extended}; the "
                f"{settings.report.max_pages}-page one-minute format is for up to {cmn} names)",
            )
            if settings.report.auto_extend
            else _t(
                f"{settings.report.max_pages}쪽에 맞추려 압축됨(차트·뉴스 일부 생략)",
                f"compacted to {settings.report.max_pages} pages (charts / some news omitted)",
            )
        )
        notes.append(
            _t(
                f"종목 {len(out)}개 — {pages_txt}. LLM 비용은 종목 수에 비례합니다(--no-llm으로 먼저 추리는 것을 권장)",
                f"{len(out)} tickers - {pages_txt}. LLM cost scales with the count (screen with --no-llm first)",
            )
        )
    return out, picks, notes


def _print_picks(picks: list[dict[str, Any]]) -> None:
    if not picks:
        return
    tb = Table(title=_t("모델 자동 선정 종목", "Model picks"), header_style="bold")
    for col in (
        _t("시장", "Market"),
        _t("구분", "Side"),
        _t("종목", "Ticker"),
        _t("이름", "Name"),
        _t("백분위", "Pct"),
    ):
        tb.add_column(col)
    for p in picks:
        side = _t("상위", "top") if p["side"] == "top" else _t("하위", "bottom")
        pct = f"{p['percentile']:.0%}" if math.isfinite(p["percentile"]) else "—"
        tb.add_row(p["market"], side, p["ticker"], p["name"], pct)
    console.print(tb)
    console.print(
        f"[dim]{_t('기준일', 'as of')} " + ", ".join(sorted({f"{p['market']} {p['as_of']}" for p in picks})) + "[/dim]"
    )


@app.command()
def report(
    tickers: str | None = typer.Option(
        None, "--tickers", "-t", help="쉼표로 구분한 종목 코드 (예: 005930,000660,AAPL,NVDA)"
    ),
    top: int = typer.Option(0, "--top", help="종목 선택 모델 점수 상위 N종목을 시장별로 추가 (0~20)"),
    bottom: int = typer.Option(0, "--bottom", help="모델 점수 하위 N종목(회피·매도 후보)을 시장별로 추가 (0~20)"),
    markets: str = typer.Option("KR,US", "--markets", help="--top/--bottom 대상 시장 (KR, US 또는 KR,US)"),
    watchlist: Path | None = typer.Option(
        None, "--watchlist", help="관심 종목 파일 (기본: 종목 옵션이 하나도 없으면 config/watchlist.txt)"
    ),
    max_pages: int | None = typer.Option(
        None, "--max-pages", help="최대 쪽수 (기본 report.max_pages=3; 8종목 초과는 자동 확장, 설정 report.auto_extend)"
    ),
    capital: float | None = typer.Option(None, "--capital", help="투자금(기준통화). 주면 종목별 주식 수를 계산"),
    base: str = typer.Option("KRW", "--base", help="기준 통화: KRW 또는 USD"),
    no_llm: bool = typer.Option(False, "--no-llm", help="LLM(뉴스·공시·비평) 없이 정량 신호만 사용"),
    config: Path | None = typer.Option(None, "--config", help="default.yaml 위에 덮어쓸 YAML"),
    out: Path | None = typer.Option(None, "--out", help="출력 폴더 (기본: report.output_dir)"),
    open_pdf: bool = typer.Option(False, "--open", help="생성 후 PDF 열기 (macOS)"),
    no_model_cache: bool = typer.Option(False, "--no-model-cache", help="RSMT-Q 적합 모델 캐시를 쓰지 않고 재적합"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="로그를 콘솔에도 출력"),
    lang: str | None = typer.Option(None, "--lang", help=LANG_HELP),
    no_record: bool = typer.Option(
        False,
        "--no-record",
        help="개발·검증 실행: 전진 추적 원장(holdout/ledger, git 추적)에 기록하지 않음 (QUANTCIO_NO_RECORD=1과 같음)",
    ),
) -> None:
    """A4 2~3쪽 PDF 리포트 생성 (한국어 기본, --lang en 영어; 4개 모듈: 시그널 매트릭스·퀀트/ML·LLM·실행/리스크)."""
    from quantcio.pipeline import PipelineError, run_report
    from quantcio.tracking import NO_RECORD_ENV

    settings = _settings(config, lang)
    if max_pages is not None:
        if max_pages < 1:
            err_console.print(f"[red]--max-pages 는 1 이상이어야 합니다: {max_pages}[/red]")
            raise typer.Exit(2)
        settings.report.max_pages = int(max_pages)
    try:
        chosen, picks, pick_notes = resolve_report_tickers(tickers, top, bottom, markets, watchlist, settings)
    except TickerSelectionError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    _print_picks(picks)
    for n in pick_notes:
        console.print(f"[yellow]•[/yellow] {n}", markup=True)
    console.print(f"{_t('리포트 종목', 'Tickers')} ({len(chosen)}): {', '.join(chosen)}", markup=False)
    if no_record:  # same switch as the environment variable: this process only
        os.environ[NO_RECORD_ENV] = "1"
    log_file = _setup_logging(settings, verbose)
    t0 = time.perf_counter()
    try:
        art = asyncio.run(
            run_report(
                chosen,
                settings,
                capital=capital,
                base_currency=base,
                use_llm=not no_llm,
                out_dir=out,
                use_model_cache=not no_model_cache,
                progress=_progress_printer(),
            )
        )
    except PipelineError as exc:
        err_console.print(f"[bold red]{_t('리포트 생성 실패', 'Report failed')}[/bold red]: {exc}")
        raise typer.Exit(1) from exc
    rep = art.report
    console.print(_summary_table(rep))
    invested = sum(t.execution.weight for t in rep.tickers)
    pr = rep.portfolio_risk
    vol = pr.get("ex_ante_vol", float("nan"))
    h = settings.horizon_days
    console.print(
        _t(
            f"투자 비중 {invested:.1%} · 현금 {rep.cash_weight:.1%} · 사전 변동성 {vol:.1%}"
            f" · VaR95({h}일) {_pct(pr.get('var95'))} · LLM 비용 ${rep.llm_cost_usd:.4f}",
            f"Invested {invested:.1%} · cash {rep.cash_weight:.1%} · ex-ante vol {vol:.1%}"
            f" · VaR95({h}D) {_pct(pr.get('var95'))} · LLM cost ${rep.llm_cost_usd:.4f}",
        )
    )
    if rep.market_regime:
        console.print(
            _t("시장 레짐: ", "Market regime: ") + " · ".join(f"{k} {v}" for k, v in rep.market_regime.items())
        )
    _print_llm_usage(getattr(art, "llm_usage", None) or {})
    _print_warnings(rep, art.warnings)
    if art.pdf_path is None:
        err_console.print(
            "[bold red]"
            + _t(
                f"PDF 생성 불가 — HTML 대체 파일만 저장됨: {art.html_path}",
                f"PDF not generated — HTML fallback saved only: {art.html_path}",
            )
            + "[/bold red]"
        )
    else:
        console.print(f"[bold green]PDF[/bold green] {art.pdf_path} ({art.pages}{_t('쪽', ' pages')})")
    console.print(f"HTML {art.html_path}\nJSON {art.json_path}\n{_t('로그', 'Log')} {log_file}")
    _print_timings({**art.timings, "total": time.perf_counter() - t0})
    if open_pdf and art.pdf_path is not None and sys.platform == "darwin":
        subprocess.run(["open", str(art.pdf_path)], check=False)


@app.command()
def analyze(
    ticker: str = typer.Argument(..., help="종목 코드 (예: 035720, AAPL)"),
    as_json: bool = typer.Option(False, "--json", help="TickerReport JSON 출력"),
    no_llm: bool = typer.Option(False, "--no-llm", help="LLM 없이 정량 신호만 사용"),
    base: str = typer.Option("KRW", "--base", help="기준 통화: KRW 또는 USD"),
    config: Path | None = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    lang: str | None = typer.Option(None, "--lang", help=LANG_HELP),
) -> None:
    """단일 종목 분석 (PDF 없이 콘솔/JSON 출력; --lang en 영어)."""
    from quantcio.pipeline import PipelineError, build_report

    settings = _settings(config, lang)
    log_file = _setup_logging(settings, verbose)
    try:
        res = asyncio.run(
            build_report(
                [ticker],
                settings,
                base_currency=base,
                use_llm=not no_llm,
                progress=None if as_json else _progress_printer(),
            )
        )
    except PipelineError as exc:
        err_console.print(f"[bold red]{_t('분석 실패', 'Analysis failed')}[/bold red]: {exc}")
        raise typer.Exit(1) from exc
    rep = res.report
    if as_json:
        sys.stdout.write(rep.tickers[0].model_dump_json(indent=2) + "\n")
        for n in rep.notes:
            err_console.print(f"[dim]{_t('알림', 'Note')}: {n}[/dim]")
        return
    _print_ticker(rep.tickers[0], settings)
    diag = rep.model_diagnostics.get(rep.tickers[0].instrument.market.value, [])
    if diag:
        console.print(f"[bold]{_t('RSMT-Q 모델 진단', 'RSMT-Q model diagnostics')}[/bold]")
        for ln in diag:
            console.print(f"  {ln}", markup=False)
    _print_warnings(rep)
    console.print(f"[dim]{_t('로그', 'Log')} {log_file}[/dim]")
    _print_timings(res.timings)


def _print_ticker(t: TickerReport, settings: Settings) -> None:
    ccy = t.instrument.currency
    ex, pl, q = t.executive, t.execution, t.quant
    passed, failed = _t("통과", "pass"), _t("미달", "fail")
    console.rule(
        f"{t.instrument.name} ({t.instrument.ticker}) · {t.instrument.exchange or ''} · "
        f"{_t('기준일', 'as of')} {t.as_of}"
    )
    console.print(
        _t(
            f"신호 {_signal(ex.signal)} · 신뢰도 {ex.confidence} · 종합점수 {ex.composite:+.3f} · "
            f"현재가 {_money(t.last_price, ccy)} → 목표가 {_money(ex.target_price, ccy)} "
            f"({_pct(ex.upside)}, {ex.horizon_label})",
            f"Signal {_signal(ex.signal)} · confidence {ex.confidence} · composite {ex.composite:+.3f} · "
            f"price {_money(t.last_price, ccy)} → target {_money(ex.target_price, ccy)} "
            f"({_pct(ex.upside)}, {ex.horizon_label})",
        )
    )
    console.print(_t("필러: ", "Pillars: ") + " · ".join(f"{k} {v:+.3f}" for k, v in ex.pillars.items()))
    console.print(f"{_t('논지', 'Thesis')}: {ex.thesis}", markup=False)
    d = q.distribution
    if d is not None:
        console.print(
            f"RSMT-Q({d.horizon_days}{_t('일', 'D')}): μ {_pct(d.mu)} · σ {_pct(d.sigma, 1, False)} · "
            f"{_t('P(상승)', 'P(up)')} {d.prob_up:.1%} · {_t('P(BM 초과)', 'P(beat BM)')} {d.prob_outperform:.1%} · "
            f"10~90% {_pct(d.q10)} ~ {_pct(d.q90)} · VaR95 {_pct(d.var95)} · "
            f"{_t('스킬', 'skill')} {passed if d.skill_ok else failed}"
            f"{_t(' · 저하', ' · degraded') if d.degraded else ''}"
        )
        console.print(
            f"  {_t('리프', 'leaf')} #{d.leaf_id} (n={d.leaf_n}): "
            f"{' / '.join(d.leaf_path) or _t('루트(분할 없음)', 'root (no split)')} · {_t('블렌드', 'blend')} {d.blend}"
        )
    console.print(
        _t("팩터: ", "Factors: ")
        + " · ".join(f"{f.label} {'—' if f.score is None else f'{f.score:.0f}'}" for f in q.factors)
    )
    fm = q.factor_model
    if fm is not None:
        console.print(
            f"{_t('팩터 모델', 'Factor model')}: {_t('백분위', 'pctile')} {fm.percentile:.0%} · z {fm.score:+.2f} · "
            f"{_t('필러', 'pillar')} {fm.pillar:+.2f} · OOS IC t "
            f"{'—' if fm.oos_ic_t is None else f'{fm.oos_ic_t:+.1f}'} · {_t('스킬', 'skill')} "
            f"{passed if fm.skill_ok else failed}"
        )
    b = pl.barrier
    console.print(
        _t(
            f"실행: 진입 {_money(pl.entry_low, ccy)}~{_money(pl.entry_high, ccy)} · "
            f"손절 {_money(pl.stop_loss, ccy)} ({_pct(pl.stop_loss_pct)})"
            f" · 익절 {_money(pl.take_profit, ccy)} ({_pct(pl.take_profit_pct)}) · "
            f"R:R {pl.reward_risk:.2f} · 비중 {pl.weight:.1%}",
            f"Plan: entry {_money(pl.entry_low, ccy)}~{_money(pl.entry_high, ccy)} · "
            f"stop {_money(pl.stop_loss, ccy)} ({_pct(pl.stop_loss_pct)})"
            f" · take-profit {_money(pl.take_profit, ccy)} ({_pct(pl.take_profit_pct)}) · "
            f"R:R {pl.reward_risk:.2f} · weight {pl.weight:.1%}",
        )
    )
    if b is not None:
        console.print(
            _t(
                f"배리어({b.horizon_days}거래일): 익절 선도달 {b.p_take_profit:.1%} · 시간청산 {b.p_time_exit:.1%} · "
                f"손절 선도달 {b.p_stop_loss:.1%} · 기대 거래수익 {_pct(b.expected_trade_return)} · "
                f"Kelly {b.kelly_optimal:.3f}",
                f"Barrier ({b.horizon_days} trading days): TP first {b.p_take_profit:.1%} · time exit "
                f"{b.p_time_exit:.1%} · SL first {b.p_stop_loss:.1%} · expected trade return "
                f"{_pct(b.expected_trade_return)} · Kelly {b.kelly_optimal:.3f}",
            )
        )
    console.print(f"{_t('행동', 'Action')}: {pl.action}", markup=False)
    for tg in pl.triggers:
        console.print(f"  - {tg}", markup=False)
    if t.llm.skipped:
        console.print(f"[dim]{_t('LLM 분석 생략(--no-llm)', 'LLM analysis skipped (--no-llm)')}[/dim]")
    else:
        n = t.llm.news
        news = "—" if n is None else f"{n.score_100:+d} ({_t('사용', 'used')} {n.n_used}/{n.n_articles})"
        tone = "—" if t.llm.filings is None else f"{t.llm.filings.tone:+.2f}"
        console.print(
            f"LLM: {_t('뉴스 감성', 'news sentiment')} {news} · {_t('공시 톤', 'filings tone')} {tone} · "
            f"{_t('비용', 'cost')} ${t.llm.cost_usd:.4f}"
        )


@app.command()
def rsmt(
    market: str = typer.Option(..., "--market", "-m", help="KR 또는 US"),
    tune: bool = typer.Option(False, "--tune", help="ν_min(CRPS·cover80), alpha·min_leaf(logloss) 그리드 권장값 출력"),
    config: Path | None = typer.Option(None, "--config"),
    no_model_cache: bool = typer.Option(False, "--no-model-cache", help="적합 모델 캐시를 쓰지 않고 재적합"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """RSMT-Q 시장 모델 적합: 트리·진단·CV(기준 모형 대비) 출력, --tune 시 권장값."""
    from quantcio.data.cache import DiskCache
    from quantcio.pipeline import fit_market, market_diagnostics, tune_rsmt

    try:
        mk = Market(market.strip().upper())
    except ValueError as exc:
        err_console.print(f"[red]--market 은 KR 또는 US 여야 합니다: {market}[/red]")
        raise typer.Exit(2) from exc
    settings = _settings(config)
    log_file = _setup_logging(settings, verbose)
    say = _progress_printer()
    say(f"RSMT-Q {mk.value}: 유니버스 {settings.rsmt.markets[mk.value].universe_size}종목 가격 로드·적합 중")
    fit = fit_market(mk, settings, DiskCache(settings.cache_path), (), use_cache=not no_model_cache)
    if fit.model is None:
        err_console.print(f"[bold red]적합 실패[/bold red]: {fit.error}")
        raise typer.Exit(1)
    model = fit.model
    say(f"적합 완료 ({fit.seconds:.1f}s{', 캐시 재사용' if fit.cached else ''}; 패널 {fit.n_panel}종목)")
    console.rule(f"RSMT-Q {mk.value} 진단")
    for ln in market_diagnostics(fit):
        console.print(ln, markup=False)
    m = model.cv.metrics
    if m:
        tb = Table(title="CV (확장 윈도, OOS 풀링) — 모형 vs 기준(루트·오프셋 없음)", header_style="bold")
        for col in ("지표", "RSMT-Q", "기준", "차이", "판정"):
            tb.add_column(col, justify="right" if col != "지표" else "left")
        for key, label, better in (
            ("auc", "AUC P(상승)", "high"),
            ("logloss", "LogLoss", "low"),
            ("brier", "Brier", "low"),
            ("crps", "CRPS", "low"),
            ("cover80", "Cover80 (목표 0.80)", "target"),
            ("ic", "IC Spearman(μ, r)", "high"),
            ("auc_rel", "AUC P(BM 초과)", "high"),
        ):
            v, bv = m.get(key, float("nan")), m.get(f"base_{key}", float("nan"))
            if better == "target":
                good = abs(v - 0.8) < abs(bv - 0.8)
            elif better == "high":
                good = v > bv
            else:
                good = v < bv
            diff = v - bv
            tb.add_row(
                label,
                f"{v:.4f}",
                f"{bv:.4f}" if math.isfinite(bv) else "—",
                f"{diff:+.4f}" if math.isfinite(diff) else "—",
                "[green]개선[/green]" if good else "[red]미개선[/red]",
            )
        console.print(tb)
        console.print(model.gate_line(), markup=False)
    if tune:
        from dataclasses import replace as dc_replace

        from quantcio import holdout

        console.rule("튜닝 그리드 (권장값만 출력 — 설정은 변경하지 않음)")
        try:
            train, dropped = holdout.guard_panel(model.panel.train, settings)
        except holdout.HoldoutError as exc:
            err_console.print(f"[bold red]홀드아웃 레지스트리 확인 실패[/bold red]: {exc}")
            raise typer.Exit(1) from exc
        if any(dropped.values()):
            console.print(
                f"[yellow]홀드아웃 제외[/yellow]: XSH {dropped['xsh_names']}종목 {dropped['xsh_rows']}행 · "
                f"전진 홀드아웃 시작 이후 {dropped['forward_rows']}행 (튜닝에 쓰지 않음)",
            )
        tr = tune_rsmt(dc_replace(model.panel, train=train), settings, progress=say)
        tb = Table(header_style="bold")
        for col in ("파라미터", "값", "AUC", "LogLoss", "CRPS", "Cover80", "스킬"):
            tb.add_column(col, justify="right" if col != "파라미터" else "left")
        for r in tr.rows:
            mm = r.metrics
            tb.add_row(
                f"{r.param}{' (현재)' if r.current else ''}",
                f"{r.value:g}",
                f"{mm.get('auc', float('nan')):.4f}",
                f"{mm.get('logloss', float('nan')):.4f}",
                f"{mm.get('crps', float('nan')):.4f}",
                f"{mm.get('cover80', float('nan')):.3f}",
                "—" if not mm else ("통과" if r.skill_ok else "미달"),
            )
        console.print(tb)
        console.print("[bold]권장[/bold]")
        for rec in tr.recommendations:
            console.print(f"  • {rec}", markup=False)
        console.print(
            "[dim]주의: 같은 CV로 고른 값은 낙관적으로 편향될 수 있습니다. "
            "변경은 --config YAML로 적용하고 다음 기간에 재검증하세요.[/dim]"
        )
    console.print(f"[dim]로그 {log_file}[/dim]")


def _parse_folds(text: str | None) -> list[int] | None:
    """'3,4,5,6' -> [3, 4, 5, 6] (None when not given). Raises ValueError on a non-integer or < 1."""
    if text is None or not text.strip():
        return None
    out = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        k = int(part)
        if k < 1:
            raise ValueError(f"폴드 수는 1 이상이어야 합니다: {k}")
        out.append(k)
    if not out:
        raise ValueError("폴드 목록이 비어 있습니다")
    return sorted(set(out))


@app.command()
def evaluate(
    market: str = typer.Option(..., "--market", "-m", help="KR 또는 US"),
    universe: int | None = typer.Option(None, "--universe", help="평가 유니버스 종목 수 (기본: rsmt.markets.<시장>)"),
    years: int | None = typer.Option(None, "--years", help="가격 이력 연수 (기본: history_years)"),
    folds: str | None = typer.Option(None, "--folds", help="평가할 CV 폴드 배치, 예: 3,4,5,6 (기본: robust_cv_folds)"),
    out: Path | None = typer.Option(None, "--out", help="출력 폴더 (기본: <report.output_dir>/evaluation)"),
    # section F (per-ticker ML ensemble) was removed 2026-10-02; --no-ml is still accepted (ignored) for old scripts
    no_ml: bool = typer.Option(False, "--no-ml", hidden=True, help="(폐지·무시됨) 섹션 F 삭제"),
    no_cache: bool = typer.Option(False, "--no-cache", help="평가용 가격 패널 캐시를 쓰지 않고 다시 로드"),
    universe_mode: str | None = typer.Option(
        None,
        "--universe-mode",
        help="pit(시점별 상위 N·상장폐지 포함, 기본) 또는 current(오늘 목록, 생존 편향)",
    ),
    config: Path | None = typer.Option(None, "--config", help="default.yaml 위에 덮어쓸 YAML"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="로그를 콘솔에도 출력"),
) -> None:
    """엄격한 표본 외(OOS) 신호 평가: RSMT-Q CV·종목 선택 IC·확률 보정·손절/익절·가격 팩터·ML → 한국어 MD + JSON."""
    from quantcio.evaluation import EvaluationError, run_evaluation

    try:
        mk = Market(market.strip().upper())
    except ValueError as exc:
        err_console.print(f"[red]--market 은 KR 또는 US 여야 합니다: {market}[/red]")
        raise typer.Exit(2) from exc
    try:
        fold_list = _parse_folds(folds)
    except ValueError as exc:
        err_console.print(f"[red]--folds 형식 오류(예: 3,4,5,6): {exc}[/red]")
        raise typer.Exit(2) from exc
    settings = _settings(config)
    log_file = _setup_logging(settings, verbose)
    say = _progress_printer()
    t0 = time.perf_counter()
    try:
        art = run_evaluation(
            mk,
            settings,
            universe=universe,
            years=years,
            folds=fold_list,
            out_dir=out,
            use_cache=not no_cache,
            progress=say,
            universe_mode=universe_mode,
        )
    except ValueError as exc:
        err_console.print(f"[red]입력 오류: {exc}[/red]")
        raise typer.Exit(2) from exc
    except EvaluationError as exc:
        err_console.print(f"[bold red]평가 실패[/bold red]: {exc}")
        raise typer.Exit(1) from exc
    _print_evaluation(art.result)
    console.print(f"[bold green]MD[/bold green] {art.md_path}\nJSON {art.json_path}\n로그 {log_file}")
    _print_timings({**art.result.timings, "total": time.perf_counter() - t0})


def _print_evaluation(res: Any) -> None:
    """Key numbers of every section on the console (the full report is the .md file)."""
    s = res.sections
    m = res.meta
    mode = {"pit": " · 시점별(PIT)", "current": " · 현재 목록"}.get(m.get("universe_mode"), "")
    console.rule(f"평가 {m['market']}{mode} · {m['n_tickers_panel']}종목 · OOS {m['oos_start']}~{m['oos_end']}")
    tb = Table(title="A) RSMT-Q vs 기준 쌍둥이 (폴드 배치별)", header_style="bold")
    for col in ("폴드", "AUC 개선 [90% CI]", "로그손실 개선", "CRPS 개선", "80% 적중", "VaR95 초과", "게이트"):
        tb.add_column(col, justify="right" if col != "폴드" else "left")
    for lay in s["A"]["layouts"]:
        if not lay.get("available"):
            tb.add_row(str(lay["folds"]), "CV 불가", "—", "—", "—", "—", "미달")
            continue
        rows = {r["key"]: r for r in lay["rows"]}
        cov = {r["key"]: r for r in lay["coverage"]}
        auc = rows["auc"]
        tb.add_row(
            str(lay["folds"]),
            f"{auc['gain']:+.4f} [{auc['ci90'][0]:+.4f}, {auc['ci90'][1]:+.4f}]",
            f"{rows['logloss']['gain']:+.4f}",
            f"{rows['crps']['gain']:+.5f}",
            f"{cov['cover80']['model']:.1%}",
            f"{cov['exc95']['model']:.1%}",
            "[green]통과[/green]" if lay["gate_pass"] else "[red]미달[/red]",
        )
    console.print(tb)
    prod = s["A"]["production"]
    console.print(
        f"운영 스킬 게이트(폴드 배치 {prod['layouts']} 과반): {'통과' if prod['skill_ok'] else '미달'} "
        f"{prod['pass']}/{prod['n']}"
    )
    tb = Table(title="B) 횡단면 순위 IC (vs 베타조정 초과수익)", header_style="bold")
    for col in ("신호", "평균 IC", "IC t", "IC>0", "5분위 스프레드", "t"):
        tb.add_column(col, justify="right" if col != "신호" else "left")
    for sig in s["B"]["signals"]:
        tb.add_row(
            sig["label"],
            f"{sig['mean_ic']:+.3f}" if math.isfinite(sig["mean_ic"]) else "—",
            f"{sig['t_ic']:+.2f}" if math.isfinite(sig["t_ic"]) else "—",
            f"{sig['share_ic_pos']:.0%}" if math.isfinite(sig["share_ic_pos"]) else "—",
            _pct(sig["spread_mean"], 2),
            f"{sig['spread_t']:+.2f}" if math.isfinite(sig["spread_t"]) else "—",
        )
    console.print(tb)
    pu = s["C"]["p_up"]["model"]
    console.print(
        f"C) P(상승) ECE {pu['ece']:.3f} · 평균 예측 {pu['mean_pred']:.1%} vs 실현 {pu['rate']:.1%} · "
        f"재보정 기울기 {pu['slope']:.2f}"
    )
    d = s["D"]
    if d.get("n"):
        console.print(
            f"D) 익절 먼저 예측 {d['tp']['pred']:.1%} vs 실현 {d['tp']['real']['hat']:.1%} · 손절 먼저 예측 "
            f"{d['sl']['pred']:.1%} vs 실현 {d['sl']['real']['hat']:.1%} · "
            f"Brier 개선 {d['brier3']['gain']['hat']:+.4f} ({d['n']}건, 제외 {sum(d['excluded'].values())})"
        )
    parts = [f"{r['label']} t={r['full']['t_ic']:+.2f}" for r in s["E"]["signals"] if math.isfinite(r["full"]["t_ic"])]
    console.print("E) 가격 팩터 IC t: " + " · ".join(parts))
    console.print("[bold]G) 결론[/bold]")
    for ln in s["G"]["lines"]:
        console.print(f"  • {ln['topic']} — {ln['tier']}", markup=False)
    console.print(
        "권장 필러 가중(설정 미변경): "
        + " · ".join(f"{w['label']} {w['current']:.0%}→{w['recommended']:.0%}" for w in s["G"]["weights"])
    )
    for n in res.notes:
        console.print(f"  알림: {n}", markup=False)


def _anthropic_probe() -> tuple[str, str]:
    """Free live check of the configured Anthropic credentials (models.list), incl. the workspace header."""
    import anthropic

    from quantcio.llm.client import WORKSPACE_ENV, WORKSPACE_HINT, is_workspace_error, with_workspace_header

    try:
        from quantcio.llm.client import WORKSPACE_HINT_EN
    except ImportError:  # older llm.client without the English hint
        WORKSPACE_HINT_EN = WORKSPACE_HINT  # noqa: N806

    try:
        client = anthropic.Anthropic(max_retries=0, timeout=15.0, **with_workspace_header({}))
        client.models.list(limit=1)
    except anthropic.BadRequestError as exc:
        if is_workspace_error(exc):
            return "FAIL", _t(WORKSPACE_HINT, WORKSPACE_HINT_EN)
        return "FAIL", _t(f"잘못된 요청(400): {str(exc)[:160]}", f"invalid request (400): {str(exc)[:160]}")
    except anthropic.AuthenticationError:
        return "FAIL", _t(
            "인증 실패(401) — ANTHROPIC_API_KEY가 유효하지 않거나 폐기된 키입니다",
            "authentication failed (401) — ANTHROPIC_API_KEY is invalid or revoked",
        )
    except anthropic.PermissionDeniedError as exc:
        return "FAIL", _t(f"권한 오류(403): {str(exc)[:160]}", f"permission error (403): {str(exc)[:160]}")
    except anthropic.APIConnectionError as exc:
        return "WARN", _t(f"연결 실패: {type(exc).__name__}", f"connection failed: {type(exc).__name__}")
    except Exception as exc:  # noqa: BLE001 - doctor reports, never crashes
        return "WARN", _t(
            f"확인 실패: {type(exc).__name__}: {str(exc)[:120]}",
            f"check failed: {type(exc).__name__}: {str(exc)[:120]}",
        )
    ws = os.environ.get(WORKSPACE_ENV, "").strip()
    return "OK", _t("모델 목록 조회 성공", "model list OK") + (
        _t(f" (워크스페이스 {ws})", f" (workspace {ws})") if ws else ""
    )


@app.command()
def backtest(
    market: str = typer.Option(..., "--market", "-m", help="KR 또는 US / KR or US"),
    config: Path | None = typer.Option(None, "--config"),
    out: Path | None = typer.Option(None, "--out", help="리포트 폴더 (기본: <report.output_dir>/backtest)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="캐시된 백테스트를 쓰지 않고 다시 계산"),
    no_macro: bool = typer.Option(
        False, "--no-macro", help="매크로 레짐 생략(macro_conditional 후보 제외, rsmt_tree는 시장 변수만)"
    ),
    lang: str | None = typer.Option(None, "--lang", help="출력 언어 ko|en (기본: report.language)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """PIT 팩터 백테스트(PER·PBR·거래량·3M/6M 모멘텀·섹터) + 워크포워드 멀티팩터 모델 후보 비교·선택 → MD + JSON."""
    from quantcio.backtest import report_summary, run_factor_backtest, write_backtest_report
    from quantcio.data.cache import DiskCache

    try:
        mk = Market(market.strip().upper())
    except ValueError as exc:
        err_console.print(f"[red]--market 은 KR 또는 US 여야 합니다 / --market must be KR or US: {market}[/red]")
        raise typer.Exit(2) from exc
    settings = _settings(config, lang)
    log_file = _setup_logging(settings, verbose)
    cache = DiskCache(settings.cache_path)
    say = _progress_printer()
    regime = None
    if not no_macro and settings.macro.enabled:
        try:
            from quantcio.macro import regime_history

            regime = regime_history(settings, cache)
        except Exception as exc:  # noqa: BLE001 - the backtest runs without the conditioned variant
            err_console.print(
                tr(
                    settings,
                    f"[yellow]매크로 레짐 이력 실패({type(exc).__name__}) — 조건부 비교 생략[/yellow]",
                    f"[yellow]Macro regime history failed ({type(exc).__name__})"
                    " — conditioned variant skipped[/yellow]",
                )
            )
    try:
        res = run_factor_backtest(mk, settings, cache, use_cache=not no_cache, progress=say, regime=regime)
    except Exception as exc:  # noqa: BLE001 - report the failure with a clear exit code
        err_console.print(
            f"[bold red]{tr(settings, '백테스트 실패', 'Backtest failed')}[/bold red]: {type(exc).__name__}: {exc}"
        )
        raise typer.Exit(1) from exc
    out_dir = out or Path(settings.report.output_dir) / "backtest"
    md_path, json_path = write_backtest_report(res, out_dir)
    summary = report_summary(res)
    table = Table(title=tr(settings, f"팩터 백테스트 {mk.value}", f"Factor backtest {mk.value}"))
    for col in ("factor", "variant", "IC t", "L/S net", "Sharpe", "BH"):
        table.add_column(col)
    for row in summary.get("factors", []):
        ls = row.get("ls_net") or {}
        table.add_row(
            str(row.get("label") or row.get("factor", "")),
            str(row.get("variant", "")),
            _fmt_opt(row.get("ic_t"), "+.2f"),
            _fmt_opt(ls.get("ann_return"), "+.1%"),
            _fmt_opt(ls.get("sharpe"), "+.2f"),
            ("✓" if row.get("bh_significant") else "") + ("" if row.get("direction_ok", True) else " ↺"),
        )
    console.print(table)
    model = summary.get("model") or {}
    if model.get("candidates"):
        comp = Table(
            title=tr(
                settings,
                f"모델 후보 비교 {mk.value} (홀드아웃 {model.get('holdout_months')}개월은 선택에 미사용)",
                f"Model candidates {mk.value} (the {model.get('holdout_months')}-month holdout is not used to pick)",
            )
        )
        for col in ("candidate", "pre IC", "pre t", "holdout IC", "holdout t", "full t", ""):
            comp.add_column(col)
        for c in model["candidates"]:
            if not c.get("available"):
                comp.add_row(str(c.get("name")), "n/a", "", "", "", "", "")
                continue
            comp.add_row(
                str(c.get("name")),
                _fmt_opt(c.get("pre_ic_mean"), "+.4f"),
                _fmt_opt(c.get("pre_ic_t"), "+.2f"),
                _fmt_opt(c.get("holdout_ic_mean"), "+.4f"),
                _fmt_opt(c.get("holdout_ic_t"), "+.2f"),
                _fmt_opt(c.get("full_ic_t"), "+.2f"),
                "✓" if c.get("name") == model.get("method") else "",
            )
        console.print(comp)
    if model:
        console.print(
            tr(
                settings,
                f"모델({model.get('method')}) OOS IC t {_fmt_opt(model.get('oos_ic_t'), '+.2f')} · "
                f"skill_ok={model.get('skill_ok')} · 롱온리 초과(비용 후) "
                f"{_fmt_opt((model.get('lo_net') or {}).get('ann_return'), '+.1%')}",
                f"Model ({model.get('method')}) OOS IC t {_fmt_opt(model.get('oos_ic_t'), '+.2f')} · "
                f"skill_ok={model.get('skill_ok')} · long-only excess (net) "
                f"{_fmt_opt((model.get('lo_net') or {}).get('ann_return'), '+.1%')}",
            )
        )
    console.print(f"MD {md_path}\nJSON {json_path}\n{tr(settings, '로그', 'Log')} {log_file}")


def _fmt_opt(v: object, fmt: str) -> str:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    return "—" if f != f else format(f, fmt)


@app.command()
def macro(
    config: Path | None = typer.Option(None, "--config"),
    out: Path | None = typer.Option(None, "--out", help="리포트 폴더 (기본: <report.output_dir>/macro)"),
    no_backtest: bool = typer.Option(False, "--no-backtest", help="레짐 조건부·오버레이 백테스트 생략"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """매크로 레짐(글로벌·미국·한국): 현재 지표·점수·노출 배수, 10년 조건부/오버레이 백테스트 → 한국어 MD + JSON."""
    from quantcio.data.cache import DiskCache
    from quantcio.macro import current_regimes, load_macro_series, macro_backtest, write_macro_report

    settings = _settings(config)
    log_file = _setup_logging(settings, verbose)
    cache = DiskCache(settings.cache_path)
    say = _progress_printer()
    say("매크로 시계열 로드 중")
    data = load_macro_series(settings, cache)
    for f in getattr(data, "failures", {}) or {}:
        err_console.print(f"[yellow]수집 실패[/yellow] {f}: {data.failures[f]}")
    regimes = current_regimes(settings, cache, series=data)
    table = Table(title="매크로 레짐", show_lines=False)
    for col in ("지역", "점수", "레짐", "노출 배수", "요약"):
        table.add_column(col)
    names = {"global": "글로벌", "us": "미국", "kr": "한국"}
    for r in regimes:
        score = "—" if r.score is None else f"{r.score:+.2f}"
        table.add_row(names.get(r.region, r.region), score, r.label, f"{r.exposure_mult:.2f}", r.summary)
    console.print(table)
    for r in regimes:
        ind = Table(title=f"{names.get(r.region, r.region)} 지표", show_lines=False)
        for col in ("지표", "값", "3M 변화", "z(+위험선호)", "기준일"):
            ind.add_column(col)
        for i in r.indicators:
            rate_like = i.unit in ("%", "%p")
            unit = i.unit if i.unit in ("%", "%p", "원") else (f" {i.unit}" if i.unit not in ("", "x") else "")
            val = "—" if i.value is None else f"{i.value:,.4g}{unit}"
            if i.change_3m is None:
                chg = "—"
            else:  # rates: absolute %p change; prices / ratios: decimal return shown as %
                chg = f"{i.change_3m:+.2f}%p" if rate_like else f"{i.change_3m:+.1%}"
            z = "—" if i.z is None else f"{i.z:+.2f}"
            ind.add_row(i.label, val, chg, z, str(i.as_of or "—"))
        console.print(ind)
    bt: dict = {}
    if not no_backtest:
        say("레짐 조건부·오버레이 백테스트 중")
        bt = macro_backtest(settings, cache, series=data)
    out_dir = out or Path(settings.report.output_dir) / "macro"
    md_path, json_path = write_macro_report(regimes, bt, out_dir)
    console.print(f"MD {md_path}\nJSON {json_path}\n로그 {log_file}")


@app.command()
def doctor(
    config: Path | None = typer.Option(None, "--config"),
    out: Path | None = typer.Option(None, "--out", help="샘플 PDF 출력 폴더"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    lang: str | None = typer.Option(None, "--lang", help=LANG_HELP),
) -> None:
    """환경 점검: API 키, Chrome 실행, KR·US 데이터 소스 접속, 샘플 PDF 렌더링 (--lang en: 영어 샘플)."""
    settings = _settings(config, lang)
    log_file = _setup_logging(settings, verbose)
    rows: list[tuple[str, str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        rows.append((name, status, detail))

    # 1) environment keys (load_settings already read .env)
    has_llm = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    add(
        "ANTHROPIC_API_KEY",
        "OK" if has_llm else "WARN",
        _t("설정됨", "set")
        if has_llm
        else _t("없음 — LLM 계층 사용 불가(--no-llm으로 실행)", "missing — LLM layer unavailable (run with --no-llm)"),
    )
    if has_llm:
        add(_t("Claude API 연결", "Claude API connection"), *_anthropic_probe())
    add(
        "SEC_USER_AGENT",
        "OK" if settings.data.sec_user_agent.strip() else "WARN",
        settings.data.sec_user_agent
        or _t("없음 — 미국 SEC EDGAR 공시 수집 불가", "missing — US SEC EDGAR filings unavailable"),
    )
    add(
        "DART_API_KEY",
        "OK" if settings.data.dart_api_key else "INFO",
        _t("설정됨", "set")
        if settings.data.dart_api_key
        else _t("없음(선택) — 키 없는 DART 웹 검색으로 대체", "missing (optional) — keyless DART web search used"),
    )

    # 2) Chrome / PDF engine
    try:
        from quantcio.report import PdfRenderer

        t0 = time.perf_counter()
        with PdfRenderer() as r:
            ok = r.active
        add(
            "Chrome(PDF)",
            "OK" if ok else "FAIL",
            f"{_t('Playwright 브라우저 실행', 'Playwright browser launch')} {time.perf_counter() - t0:.1f}s",
        )
    except Exception as exc:  # noqa: BLE001
        add("Chrome(PDF)", "FAIL", f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}")

    # 3) data sources: one ticker per market, live (cache disabled)
    from quantcio.data import base as data_base
    from quantcio.data.cache import DiskCache

    nocache = DiskCache(settings.cache_path / "doctor", enabled=False)
    for mk, tk in ((Market.KR, "005930"), (Market.US, "AAPL")):
        t0 = time.perf_counter()
        try:
            prov = data_base.get_provider(mk, settings, nocache)
            inst = prov.resolve(tk)
            px = data_base.clean_prices(prov.prices(inst, date.today() - timedelta(days=40)))
            if px.empty:
                raise ValueError("가격 0행")
            notes = data_base.provider_warnings(prov, inst.ticker, lang_of(settings))
            close = _money(float(px["close"].iloc[-1]), inst.currency)
            detail = _t(
                f"{inst.name}: {len(px)}행, 최근 {px.index[-1].date()} 종가 {close} ({time.perf_counter() - t0:.1f}s)",
                f"{inst.name}: {len(px)} rows, last {px.index[-1].date()} close {close} "
                f"({time.perf_counter() - t0:.1f}s)",
            )
            add(
                f"{_t('데이터', 'Data')} {mk.value}",
                "WARN" if notes else "OK",
                detail + (f" · {_t('경고', 'warnings')}: {'; '.join(notes)[:160]}" if notes else ""),
            )
        except Exception as exc:  # noqa: BLE001
            add(f"{_t('데이터', 'Data')} {mk.value}", "FAIL", f"{tk}: {type(exc).__name__}: {str(exc)[:160]}")

    # 4) sample PDF
    try:
        from quantcio.report import build_sample_charts, build_sample_report, fit_to_pages_isolated

        folder = out if out is not None else Path(settings.report.output_dir) / "doctor"  # --out is the folder itself
        rep = build_sample_report(6, settings=settings)
        t0 = time.perf_counter()
        # the Korean sample keeps its documented name; the English one gets its own file
        name = "sample_report_en.pdf" if _LANG == "en" else "sample_report.pdf"
        res = fit_to_pages_isolated(rep, settings, folder / name, charts=build_sample_charts(rep))
        label = _t("샘플 PDF", "Sample PDF")
        if res.pdf_path is None:
            add(label, "FAIL", f"{_t('HTML 대체 저장', 'HTML fallback saved')}: {res.html_path}")
        else:
            add(
                label,
                "OK" if res.fits else "WARN",
                _t(
                    f"{res.pdf_path} ({res.pages}쪽, 압축 레벨 {res.compact_level}, {time.perf_counter() - t0:.1f}s)",
                    f"{res.pdf_path} ({res.pages} pages, compaction level {res.compact_level}, "
                    f"{time.perf_counter() - t0:.1f}s)",
                ),
            )
    except Exception as exc:  # noqa: BLE001
        add(_t("샘플 PDF", "Sample PDF"), "FAIL", f"{type(exc).__name__}: {str(exc)[:160]}")

    tb = Table(title="QuantCIO doctor", header_style="bold")
    tb.add_column(_t("항목", "Check"))
    tb.add_column(_t("상태", "Status"))
    tb.add_column(_t("상세", "Detail"))
    style = {"OK": "green", "WARN": "yellow", "INFO": "cyan", "FAIL": "bold red"}
    for name, status, detail in rows:
        tb.add_row(name, f"[{style[status]}]{status}[/{style[status]}]", detail)
    console.print(tb)
    console.print(f"[dim]{_t('로그', 'Log')} {log_file}[/dim]")
    if any(s == "FAIL" for _, s, _ in rows):
        raise typer.Exit(1)


# ============================================================================== holdout / track (docs/HOLDOUT.md)

holdout_app = typer.Typer(no_args_is_help=True, help="홀드아웃 등록·상태·평가 (docs/HOLDOUT.md, 추가 전용 레지스트리)")
track_app = typer.Typer(no_args_is_help=True, help="전진 추적 원장: 스냅샷 기록·채점·주 지표 리포트")
from quantcio.data.alt.cli import alt_app  # noqa: E402
from quantcio.modeling.cli import modeling_app  # noqa: E402 - registered with the other sub-apps

app.add_typer(holdout_app, name="holdout")
app.add_typer(track_app, name="track")
app.add_typer(modeling_app, name="modeling")
app.add_typer(alt_app, name="alt")


def _market_opt(market: str) -> Market:
    try:
        return Market(market.strip().upper())
    except ValueError as exc:
        err_console.print(f"[red]--market 은 KR 또는 US 여야 합니다: {market}[/red]")
        raise typer.Exit(2) from exc


def _evaluation_hook(name: str) -> Any:
    """``quantcio.evaluation.<name>`` or None (provided by the evaluation module; optional here)."""
    try:
        from quantcio import evaluation
    except Exception as exc:  # noqa: BLE001 - reported by the caller as a clear message
        logging.getLogger(__name__).warning("evaluation import failed: %s", exc)
        return None
    fn = getattr(evaluation, name, None)
    return fn if callable(fn) else None


def _json_out(obj: Any) -> None:
    import json

    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")


def _xsh_headline_lines(head: dict[str, Any]) -> list[str]:
    """Console lines of an XSH look (the full report is the .md next to the JSON)."""
    if not head:
        return []

    def f(x: Any, nd: int = 4, signed: bool = False) -> str:
        if x is None:
            return "—"
        return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"

    return [
        f"  판정: {head.get('verdict')} (confirmed={head.get('confirmed')})",
        f"  모델 {head.get('selected')} · 날짜 {head.get('n_dates')}개 · 종목 {head.get('n_names')}개",
        f"  날짜 내 AUC(중앙값 초과) {f(head.get('auc_median'))} [{f(head.get('auc_median_lo'))}, "
        f"{f(head.get('auc_median_hi'))}] · 목표 {f(head.get('target_auc'), 2)} "
        f"{'달성' if head.get('target_met') else '미달'}",
        f"  AUC(y_e>0) {f(head.get('auc_zero'))} · 순위 IC {f(head.get('ic'), 4, True)} "
        f"(t {f(head.get('ic_t'), 2, True)}) · 모멘텀 대비 AUC {f(head.get('vs_momentum'), 4, True)}",
    ]


@holdout_app.command("register")
def holdout_register(
    kind: str = typer.Option(..., "--kind", help="forward(전진 FTH) 또는 xs(횡단면 XSH)"),
    market: str | None = typer.Option(None, "--market", "-m", help="KR 또는 US (xs 필수, forward 생략 시 KR·US)"),
    allow_dirty: bool = typer.Option(False, "--allow-dirty", help="커밋되지 않은 변경이 있어도 등록(기록에 남음)"),
    start: str | None = typer.Option(None, "--start", help="xs 평가 기간 시작 YYYY-MM-DD (기본: history_years 전)"),
    end: str | None = typer.Option(None, "--end", help="xs 평가 기간 끝 YYYY-MM-DD (기본: 오늘)"),
    config: Path | None = typer.Option(None, "--config", help="default.yaml 위에 덮어쓸 YAML"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """홀드아웃 등록: 설정 다이제스트·신호 코드 지문·git 커밋·사전 등록 판정 규칙을 레지스트리에 추가."""
    from quantcio import holdout
    from quantcio.data.cache import DiskCache

    settings = _settings(config)
    _setup_logging(settings, verbose)
    k = kind.strip().lower()
    try:
        if k in ("forward", "fth"):
            markets = [_market_opt(market).value] if market else ["KR", "US"]
            entry = holdout.register_forward(settings, markets, allow_dirty=allow_dirty)
        elif k in ("xs", "cross_sectional", "xsh"):
            if not market:
                err_console.print("[red]--kind xs 에는 --market KR|US 가 필요합니다[/red]")
                raise typer.Exit(2)
            mk = _market_opt(market)
            fn = _evaluation_hook("xsh_candidate_names")
            if fn is None:
                err_console.print(
                    "[bold red]XSH 등록 불가[/bold red]: evaluation.xsh_candidate_names(market, settings, cache)가 "
                    "아직 없습니다 — 평가 모듈이 후보 종목 함수를 제공해야 합니다"
                )
                raise typer.Exit(1)
            cache = DiskCache(settings.cache_path)
            names = list(fn(mk, settings, cache))
            tuned = holdout.tuning_names(mk, settings, cache)
            p1 = date.fromisoformat(end) if end else date.today()
            p0 = date.fromisoformat(start) if start else p1 - timedelta(days=int(365.25 * settings.history_years))
            band = list(settings.holdout.xsh_kr_rank_band) if mk is Market.KR else None
            entry = holdout.register_cross_sectional(
                settings, mk, names, [p0, p1], tuned, allow_dirty=allow_dirty, rank_band=band
            )
        else:
            err_console.print(f"[red]--kind 는 forward 또는 xs 여야 합니다: {kind}[/red]")
            raise typer.Exit(2)
    except holdout.HoldoutError as exc:
        err_console.print(f"[bold red]등록 실패[/bold red]: {exc}")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        err_console.print(f"[red]입력 오류: {exc}[/red]")
        raise typer.Exit(2) from exc
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 - e.g. the candidate / universe source failed: nothing was registered
        logging.getLogger(__name__).exception("holdout register failed")
        err_console.print(f"[bold red]등록 실패[/bold red]: {type(exc).__name__}: {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[bold green]등록 완료[/bold green] {entry['id']} ({entry['kind']})", markup=True)
    console.print(
        f"  설정 {entry['config_digest']} · 코드 지문 {entry['code_fingerprint']} · git {entry['git_commit'] or '없음'}"
        f"{' (dirty)' if entry['git_dirty'] else ''} · h={entry['horizon_days']} · 보유 {entry['hold_days']}일",
        markup=False,
    )
    if entry["kind"] == "cross_sectional":
        console.print(f"  종목 {len(entry['names'])}개 · 기간 {entry['period'][0]}~{entry['period'][1]}", markup=False)
    console.print(f"  레지스트리 {holdout.registry_path(settings)}", markup=False)


@holdout_app.command("status")
def holdout_status(
    as_json: bool = typer.Option(False, "--json", help="JSON 출력"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """등록된 홀드아웃: 현재 설정·코드가 동결값과 같은지, look 횟수, pristine 여부."""
    from quantcio import holdout

    settings = _settings(config)
    try:
        rows = holdout.status(settings)
    except holdout.HoldoutError as exc:
        err_console.print(f"[bold red]레지스트리 오류[/bold red]: {exc}")
        raise typer.Exit(1) from exc
    if as_json:
        _json_out(rows)
        return
    if not rows:
        console.print(f"등록된 홀드아웃 없음 ({holdout.registry_path(settings)})", markup=False)
        return
    tb = Table(title="홀드아웃 레지스트리", header_style="bold")
    for col in ("ID", "종류", "시장", "등록", "동결", "모델 범위", "look(유효)", "pristine"):
        tb.add_column(col)
    for r in rows:
        pristine = "—" if r["pristine"] is None else ("예(미열람)" if r["pristine"] else "아니오")
        tb.add_row(
            r["id"],
            "전진" if r["kind"] == "forward" else f"횡단면 {r['n_names']}종목",
            ",".join(r["markets"]),
            str(r["registered_at"])[:16],
            "[green]일치[/green]" if r["frozen_ok"] else "[red]변경됨[/red]",
            "—"
            if r.get("model_scope_ok") is None
            else ("[green]일치[/green]" if r["model_scope_ok"] else "[red]변경·미기록[/red]"),
            f"{r['n_looks']}({r['n_valid_looks']})",
            pristine,
        )
    console.print(tb)
    for r in rows:
        for why in r["reasons"]:
            console.print(f"  [yellow]{r['id']}[/yellow] ", end="")
            console.print(why, markup=False)


@holdout_app.command("attest-scope")
def holdout_attest_scope(
    entry_id: str = typer.Option(..., "--id", help="모델 범위를 기록할 XSH ID (등록 당시 상태가 유지될 때만)"),
    scope_version: int = typer.Option(2, "--scope-version", help="1: 전 시장 데이터 코드, 2: 시장별(기본)"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """예전 XSH 항목에 종목 선택 모델 범위(코드 지문·설정 다이제스트)를 추가 전용으로 기록.

    LLM·리포트 코드를 바꿔도 종목 선택 모델의 XSH 확인이 무효가 되지 않도록 합니다(docs/HOLDOUT.md).
    """
    from quantcio import holdout

    settings = _settings(config)
    try:
        att = holdout.attest_model_scope(entry_id, settings, version=scope_version)
    except holdout.HoldoutError as exc:
        err_console.print(f"[bold red]증명 실패[/bold red]: {exc}")
        raise typer.Exit(1) from exc
    console.print(
        f"[bold green]기록[/bold green] {entry_id}: 모델 범위 v{att.get('scope_version', 1)} "
        f"지문 {att['model_scope_fingerprint']} · 설정 {att['model_scope_digest']} "
        f"(근거 {att.get('basis', 'full_freeze')})",
        markup=True,
    )


@holdout_app.command("evaluate")
def holdout_evaluate(
    entry_id: str = typer.Option(..., "--id", help="홀드아웃 ID (예: XSH-KR-20260925, FTH-20260925)"),
    force: bool = typer.Option(False, "--force", help="동결값과 달라도 실행(결과는 contaminated로 기록)"),
    config: Path | None = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """홀드아웃 평가 1회 실행 + look 기록 (XSH는 첫 유효 look만 pristine)."""
    from quantcio import holdout, tracking
    from quantcio.data.cache import DiskCache

    settings = _settings(config)
    _setup_logging(settings, verbose)
    try:
        entry = holdout.load_registry(settings).get(entry_id)
    except holdout.HoldoutError as exc:
        err_console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(1) from exc
    ok, reasons = holdout.check_frozen(entry, settings)
    if not ok and not force:
        err_console.print(f"[bold red]{entry_id}: 동결값과 달라 평가를 거부합니다[/bold red]")
        for why in reasons:
            err_console.print(f"  • {why}", markup=False)
        err_console.print("  → 등록 시 설정·코드로 되돌리거나, --force로 실행(결과는 contaminated로 기록)")
        raise typer.Exit(1)
    if entry["kind"] == "cross_sectional" and entry["looks"]:
        console.print(
            f"[yellow]{entry_id}: 이미 {len(entry['looks'])}회 열람 — 이번 결과는 pristine이 아닙니다[/yellow]"
        )
    try:
        if entry["kind"] == "forward":
            res = tracking.metrics(settings, entry_id)
            for ln in tracking.report_lines(res):
                console.print(ln, markup=False)
            result_path, head = None, tracking.headline(res)
        else:
            fn = _evaluation_hook("run_xsh_evaluation")
            if fn is None:
                err_console.print(
                    "[bold red]XSH 평가 불가[/bold red]: evaluation.run_xsh_evaluation(entry, settings, cache)가 "
                    "아직 없습니다 — look은 기록하지 않았습니다"
                )
                raise typer.Exit(1)
            out = fn(entry, settings, DiskCache(settings.cache_path))
            get = out.get if isinstance(out, dict) else (lambda key, _o=out: getattr(_o, key, None))
            result_path = get("result_path") or get("json_path") or get("md_path")
            head = get("headline") or {}
            console.print(f"결과 {result_path}" + (f" · {get('md_path')}" if get("md_path") else ""), markup=False)
            for ln in _xsh_headline_lines(head):
                console.print(ln, markup=False)
    except (holdout.HoldoutError, tracking.TrackingError) as exc:
        err_console.print(f"[bold red]평가 실패[/bold red]: {exc}")
        raise typer.Exit(1) from exc
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 - the evaluation crashed before a result existed: no look recorded
        logging.getLogger(__name__).exception("holdout evaluate failed")
        err_console.print(f"[bold red]평가 실패[/bold red](look 미기록): {type(exc).__name__}: {exc}")
        raise typer.Exit(1) from exc
    look = holdout.record_look(entry_id, settings, result_path=result_path, headline=head, forced=force and not ok)
    tag = "유효" if look["valid"] else "무효(contaminated)"
    pristine = "" if look["pristine"] is None else (" · pristine" if look["pristine"] else " · pristine 아님")
    console.print(f"look 기록: {tag}{pristine} · {look['at']}", markup=False)


@track_app.command("snapshot")
def track_snapshot(
    market: str = typer.Option(..., "--market", "-m", help="KR 또는 US"),
    n: int | None = typer.Option(None, "--n", help="유니버스 종목 수 (기본: rsmt.markets.<시장>.universe_size)"),
    config: Path | None = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """유니버스 전체를 --no-llm으로 산출해 원장에 기록 (주 1회 cron/launchd 권장)."""
    from quantcio import holdout, tracking
    from quantcio.pipeline import PipelineError

    mk = _market_opt(market)
    settings = _settings(config)
    log_file = _setup_logging(settings, verbose)
    try:
        rec = tracking.snapshot(mk, settings, n, progress=_progress_printer())
    except (tracking.TrackingError, holdout.HoldoutError, PipelineError) as exc:
        err_console.print(f"[bold red]스냅샷 실패[/bold red]: {exc}")
        raise typer.Exit(1) from exc
    console.print(
        f"[bold green]기록[/bold green] 신규 {rec.n_new}건 · 기존 {rec.n_existing}건 · 홀드아웃 "
        + ", ".join(f"{m}={h or '없음'}{'' if rec.valid.get(m) else '(무효)'}" for m, h in rec.holdout_ids.items())
    )
    for note in rec.notes:
        console.print(f"  • {note}", markup=False)
    console.print(f"원장 {rec.path}\n로그 {log_file}", markup=False)


@track_app.command("score")
def track_score(
    config: Path | None = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """만기가 된 예측을 새 가격으로 채점 (멱등; 미만기는 건너뜀)."""
    from quantcio import holdout, tracking

    settings = _settings(config)
    _setup_logging(settings, verbose)
    try:
        res = tracking.score(settings)
    except holdout.HoldoutError as exc:
        err_console.print(f"[bold red]채점 실패[/bold red]: {exc}")
        raise typer.Exit(1) from exc
    console.print(
        f"채점: {settings.horizon_days}일 결과 신규 {res.n_h}건 · 배리어 신규 {res.n_barrier}건"
        f" · 완료 {res.n_complete}건 · 미만기 {res.n_immature}건 · 실패 {res.n_failed}건",
        markup=False,
    )
    for note in res.notes:
        console.print(f"  • {note}", markup=False)
    console.print(f"원장 {res.path}", markup=False)


@track_app.command("report")
def track_report(
    holdout_id: str | None = typer.Option(None, "--holdout", help="FTH ID (기본: 최신 전진 홀드아웃)"),
    as_json: bool = typer.Option(False, "--json", help="JSON 출력"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """주 지표 5개(BH 보정)와 참고 지표, 표본 수와 판정 가능 여부."""
    from quantcio import holdout, tracking

    settings = _settings(config)
    try:
        res = tracking.metrics(settings, holdout_id)
    except (holdout.HoldoutError, tracking.TrackingError) as exc:
        err_console.print(f"[bold red]리포트 실패[/bold red]: {exc}")
        raise typer.Exit(1) from exc
    if res.get("holdout_id"):
        try:
            look = holdout.record_look(
                res["holdout_id"], settings, headline=tracking.headline(res), note="track report"
            )
            res["look"] = {"valid": look["valid"], "at": look["at"]}
        except holdout.HoldoutError as exc:
            res.setdefault("notes", []).append(f"look 기록 실패: {exc}")
    if as_json:
        _json_out(res)
        return
    for ln in tracking.report_lines(res):
        console.print(ln, markup=False)


def main() -> None:  # pragma: no cover - console entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    app()
