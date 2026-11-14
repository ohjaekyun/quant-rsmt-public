"""Claude LLM layer: news sentiment, filing intelligence, adversarial critic, and the bounded LLM pillar.

Public API:

- ``LLMClient`` / ``FakeLLMClient`` / ``LLMConfigError`` / ``LLMRecorder`` (``quantcio.llm.client``)
- ``analyze_news`` / ``analyze_filings`` / ``web_research`` / ``critique``
- ``llm_pillar(news, filings) -> float`` in [-1, 1]
- ``run_llm_analysis(client, md, quant, settings, prelim_fn) -> LLMIntelligence``. It never raises for LLM
  failures. Hard failures (a part is missing) go to ``LLMIntelligence.errors``. Degraded-but-served
  paths (fallback model, fallback beta rejected, clamps, dropped ungrounded items) go to
  ``LLMIntelligence.notices``.

Output language: ``settings.report.language`` ("ko" default / "en") drives the prompts, the wire schemas, the
user messages and every error / notice text (``LLMRecorder(lang=...)``). Evidence quotes stay verbatim.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime

from quantcio.config import Settings
from quantcio.i18n import lang_of, tr
from quantcio.llm.client import FakeLLMClient, LLMClient, LLMConfigError, LLMLike, LLMRecorder
from quantcio.llm.critic import critique
from quantcio.llm.filings import WebResearch, analyze_filings, web_research
from quantcio.llm.news import analyze_news
from quantcio.models import (
    ExecutiveSignal,
    FilingIntelligence,
    LLMIntelligence,
    MarketData,
    NewsSentiment,
    QuantSignal,
)

log = logging.getLogger(__name__)

__all__ = [
    "FakeLLMClient",
    "LLMClient",
    "LLMConfigError",
    "LLMLike",
    "LLMRecorder",
    "WebResearch",
    "analyze_filings",
    "analyze_news",
    "critique",
    "llm_pillar",
    "risk_penalty",
    "run_llm_analysis",
    "web_research",
]

NEWS_WEIGHT = 0.65
FILINGS_WEIGHT = 0.35
NEWS_FULL_EVIDENCE = 5  # n_used at which news gets its full weight
RISK_PENALTY = {"high": 0.05, "critical": 0.20}
RISK_PENALTY_CAP = 0.40

PrelimFn = Callable[[LLMIntelligence], ExecutiveSignal | Awaitable[ExecutiveSignal]]


def risk_penalty(filings: FilingIntelligence | None) -> float:
    """min(0.40, 0.05 * #high + 0.20 * #critical risk factors), or 0 without filings (always >= 0)."""
    if filings is None:
        return 0.0
    return min(RISK_PENALTY_CAP, sum(RISK_PENALTY.get(r.severity, 0.0) for r in filings.risk_factors))


def llm_pillar(news: NewsSentiment | None, filings: FilingIntelligence | None) -> float:
    """Bounded LLM pillar in [-1, 1] (ARCHITECTURE.md, llm/).

    The inputs are:

        N = news.score * min(1, n_used / 5)     (news part, scaled by evidence)
        F = filings.tone                        (filings part, one holistic judgement)
        P = risk_penalty(filings)               (min(0.40, 0.05*#high + 0.20*#critical), 0 without filings)

    The pillar for each case is:

        both present      clip(0.65*N + 0.35*F - P)
        news only         clip(N)              renormalised 0.65/0.65, the way fusion renormalises
                                               a missing pillar
        filings only      clip(0.35*F - P)     NOT renormalised, so |pillar| <= 0.35*|tone| + P
        neither           0.0

    Why filings-only is capped instead of renormalised:

    - ``tone`` is a single model judgement over a handful of documents, and it has no evidence count
      that could scale it down. Renormalising 0.35/0.35 would let one filing's tone drive the pillar
      to +-1, and so move 40% of the composite.
    - The cap makes the pillar continuous in the news evidence. Filings-only gives exactly the same
      value as "news present with n_used = 0" (0.65*0 + 0.35*F - P).
    - The risk penalty is kept in full. Downside risk from verified filing risk factors must never be
      diluted by a missing news part.
    """
    if news is None and filings is None:
        return 0.0
    penalty = risk_penalty(filings)
    if news is None:  # filings only: capped at 0.35*|tone| (+ the full risk penalty)
        base = FILINGS_WEIGHT * filings.tone
    else:
        n_part = news.score * min(1.0, news.n_used / NEWS_FULL_EVIDENCE)
        if filings is None:  # news only: renormalised to full weight
            base = n_part
        else:
            base = NEWS_WEIGHT * n_part + FILINGS_WEIGHT * filings.tone
    return float(max(-1.0, min(1.0, base - penalty)))


async def _guarded(label: str, coro: Awaitable, rec: LLMRecorder):
    """Await ``coro``. Any exception is a hard failure: it is recorded as an error (and logged with its
    traceback), never propagated."""
    try:
        return await coro
    except Exception as exc:  # noqa: BLE001 - run_llm_analysis must never raise for LLM failures
        log.exception("LLM %s raised", label)
        detail = f"{type(exc).__name__}: {' '.join(str(exc).split())[:160]}"
        rec.error(tr(rec.lang, f"{label} 예외: {detail}", f"{label} failed with an exception: {detail}"))
        return None


async def run_llm_analysis(
    client: LLMLike,
    md: MarketData,
    quant: QuantSignal,
    settings: Settings,
    prelim_fn: PrelimFn,
    *,
    now: datetime | None = None,
) -> LLMIntelligence:
    """Full LLM pass for one ticker.

    1. News and filings run concurrently (``asyncio.gather``). For a US ticker with ``settings.llm.web_research``,
       the filings branch first adds the cited earnings-call web research as a supplementary document.
    2. Compute ``llm_pillar`` and call ``prelim_fn(partial LLMIntelligence)`` (sync or async) for the
       preliminary signal.
    3. Run the critic on it.

    Cost and messages are accumulated per ticker, and the function never raises for LLM failures. Messages
    go to one of two channels:

    - ``errors``: hard failures, where the affected part (news / filings / critic) is None. Examples: a
      failed or refused call, invalid output, an exception, or a critic skipped because ``prelim_fn``
      failed.
    - ``notices``: degraded-but-served paths, where the part is present. Examples: a server fallback
      model answered, the fallback beta was rejected, values were clamped, ungrounded or unknown items
      were dropped, or duplicates were merged.

    The partial result passed to ``prelim_fn`` carries both channels as they stand at that point.
    """
    lang = lang_of(settings)
    rec = LLMRecorder(lang=lang)
    inst = md.instrument

    async def filings_branch() -> FilingIntelligence | None:
        wr = await _guarded(
            rec.label("web_research"), web_research(client, inst, settings, as_of=md.as_of, recorder=rec), rec
        )
        supplementary = [wr.filing] if wr is not None else []
        fi = await analyze_filings(
            client, inst, md.filings, md.fundamentals, settings, recorder=rec, supplementary=supplementary
        )
        if fi is not None and wr is not None:
            fi = fi.model_copy(update={"sources": [*fi.sources, *wr.source_labels()]})
        return fi

    news, filings = await asyncio.gather(
        _guarded(rec.label("news"), analyze_news(client, inst, md.news, settings, recorder=rec, now=now), rec),
        _guarded(rec.label("filings"), filings_branch(), rec),
    )
    pillar = llm_pillar(news, filings)
    partial = LLMIntelligence(
        news=news,
        filings=filings,
        llm_pillar=pillar,
        cost_usd=rec.cost_usd,
        errors=list(rec.errors),
        notices=list(rec.notices),
    )

    critic = None
    prelim: ExecutiveSignal | None = None
    try:
        out = prelim_fn(partial)
        prelim = await out if inspect.isawaitable(out) else out
    except Exception as exc:  # noqa: BLE001 - the critic is optional; the failure is recorded
        log.exception("prelim_fn failed for %s", inst.ticker)
        detail = f"{type(exc).__name__}: {' '.join(str(exc).split())[:160]}"
        rec.error(
            tr(
                lang,
                f"예비 신호 계산 실패로 비평 생략: {detail}",
                f"preliminary signal failed — critic skipped: {detail}",
            )
        )
    else:
        if not isinstance(prelim, ExecutiveSignal):  # never skip the critic silently
            kind = type(prelim).__name__
            rec.error(
                tr(
                    lang,
                    f"예비 신호가 ExecutiveSignal이 아니어서 비평 생략: {kind}",
                    f"invalid preliminary signal (not an ExecutiveSignal) — critic skipped: {kind}",
                )
            )
            prelim = None
    if prelim is not None:
        critic = await _guarded(
            rec.label("critic"),
            critique(
                client,
                inst,
                quant,
                news,
                filings,
                md.fundamentals,
                prelim,
                news_items=md.news,
                filing_items=md.filings,
                recorder=rec,
                max_filing_tokens=settings.llm.critic_filing_tokens,
                max_news=settings.llm.critic_max_news,
                max_filings=settings.llm.critic_max_filings,
                lang=lang,
            ),
            rec,
        )

    return LLMIntelligence(
        news=news,
        filings=filings,
        critic=critic,
        llm_pillar=pillar,
        cost_usd=round(rec.cost_usd, 6),
        errors=list(rec.errors),
        notices=list(rec.notices),
        skipped=False,
    )
