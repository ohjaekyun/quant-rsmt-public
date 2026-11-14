"""Adversarial critic: reviews the preliminary signal against the quant data and the text evidence.

The critic's influence is bounded in code, whatever the model returns:

- ``confidence_adjustment`` is clamped to [-25, +5]. A positive adjustment is also zeroed when the text
  evidence is thin (no used news and no filing content).
- ``critical_flag`` is honoured only when the model names a critical category AND quotes a document
  verbatim: a contiguous quote (no fuzzy match, no ellipsis) found in the raw news/filing text it was
  shown, and not negated there. Otherwise it is dropped and recorded.
- Numbers in the thesis and in the bull/bear/contradiction lines must appear, within rounding at their
  stated precision, in the JSON context or the documents. An unsupported line is dropped. An
  unsupported thesis becomes "" (never a partially edited sentence), so fusion falls back to its
  deterministic thesis.
- Every correction above is a recorder NOTICE, and the review is still served. A failed call is a
  recorder ERROR, and the review is None.

The context JSON is compact, sorted and rounded (returns as percentages), so the critic can copy numbers
exactly. Everything volatile is in the user message; ``CRITIC_SYSTEM`` stays byte-stable.

Output language: ``critique(..., lang="en")`` (``run_llm_analysis`` passes ``settings.report.language``) selects
``CRITIC_SYSTEM_EN``, the English wire schema, English user-message headers and English notices / critical
reason. The critical evidence is still a verbatim quote in the source document's own language.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from quantcio.i18n import lang_of, tr
from quantcio.llm.client import LLMLike, LLMRecorder, en_count, task_label
from quantcio.llm.filings import order_filings, prepare_filings, source_label
from quantcio.llm.news import article_source_text, instrument_header, select_articles
from quantcio.llm.prompts import (
    Doc,
    build_user_message,
    document_notice,
    system_prompt,
    wrap_block,
    wrap_documents,
)
from quantcio.llm.schemas import (
    NumberIndex,
    clean_lines,
    critical_category_label,
    locate_evidence,
    to_critic_review,
    ungrounded_numbers,
    wire_model,
)
from quantcio.models import (
    CriticReview,
    ExecutiveSignal,
    FilingIntelligence,
    FilingItem,
    Fundamentals,
    Instrument,
    NewsItem,
    NewsSentiment,
    QuantSignal,
)

log = logging.getLogger(__name__)

TASK = "critic"
CRITIC_FILING_TOKENS = 1000  # defaults of llm.critic_filing_tokens / critic_max_news / critic_max_filings
CRITIC_MAX_NEWS = 6
CRITIC_MAX_FILINGS = 3
RISK_PRIORITY = ("critical", "high")  # filings behind these risk factors are re-shown to the critic first
MAX_POINTS = 3


# --------------------------------------------------------------------------- compact context


def _pct(x: float | None) -> float | None:
    return None if x is None else round(float(x) * 100.0, 1)


def _r(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(float(x), nd)


def _price(x: float | None, currency: str) -> float | None:
    return None if x is None else round(float(x), 0 if currency == "KRW" else 2)


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None and v != [] and v != {}}


def compact_json(obj: Any) -> str:
    """Deterministic, compact JSON (sorted keys, UTF-8 kept)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def quant_context(inst: Instrument, quant: QuantSignal, fund: Fundamentals | None) -> dict[str, Any]:
    """Compact, display-rounded quant view (returns as percentages, prices in local currency)."""
    t = quant.technicals
    cur = inst.currency
    ctx: dict[str, Any] = {
        "technicals": _drop_none(
            {
                "last_close": _price(t.last_close, cur),
                "ma50": _price(t.ma50, cur),
                "ma200": _price(t.ma200, cur),
                "rsi14": _r(t.rsi14, 1),
                "ret_1m_pct": _pct(t.ret_1m),
                "ret_3m_pct": _pct(t.ret_3m),
                "ret_12m_ex1m_pct": _pct(t.ret_12m_ex1m),
                "rel_strength_3m_pct": _pct(t.rel_strength_3m),
                "realized_vol_60d_pct": _pct(t.realized_vol_60d),
                "beta_1y": _r(t.beta_1y),
                "max_drawdown_1y_pct": _pct(t.max_drawdown_1y),
                "high_52w": _price(t.high_52w, cur),
                "low_52w": _price(t.low_52w, cur),
                "atr_pct": _pct(t.atr_pct),
            }
        ),
        "factors": [
            _drop_none(
                {
                    "name": f.name,
                    "label": f.label,
                    "score": _r(f.score, 1),
                    "rationale": f.rationale or None,
                }
            )
            for f in quant.factors
        ],
        "factor_composite": _r(quant.factor_composite, 1),
        "factor_pillar": _r(quant.factor_pillar),
        "ml_pillar": _r(quant.ml_pillar),
    }
    if quant.distribution is not None:
        d = quant.distribution
        ctx["rsmt_distribution"] = _drop_none(
            {
                "horizon_days": d.horizon_days,
                "expected_return_pct": _pct(d.expected_return),
                "prob_up_pct": _pct(d.prob_up),
                "prob_outperform_pct": _pct(d.prob_outperform),
                "q10_pct": _pct(d.q10),
                "q50_pct": _pct(d.q50),
                "q90_pct": _pct(d.q90),
                "var95_pct": _pct(d.var95),
                "skill_ok": d.skill_ok,
                "degraded": d.degraded,
                "degraded_reasons": d.degraded_reasons,
                "leaf_path": d.leaf_path,
            }
        )
    if fund is not None:
        ctx["fundamentals"] = _drop_none(
            {
                "per": _r(fund.per, 1),
                "forward_per": _r(fund.forward_per, 1),
                "pbr": _r(fund.pbr, 2),
                "sector_per": _r(fund.sector_per, 1),
                "roe_pct": _pct(fund.roe),
                "operating_margin_pct": _pct(fund.operating_margin),
                "revenue_growth_yoy_pct": _pct(fund.revenue_growth_yoy),
                "earnings_growth_yoy_pct": _pct(fund.earnings_growth_yoy),
                "dividend_yield_pct": _pct(fund.dividend_yield),
                "analyst_target_price": _price(fund.analyst_target_price, cur),
                "analyst_count": fund.analyst_count,
                "analyst_rating": fund.analyst_rating,
            }
        )
    return ctx


def llm_context(news: NewsSentiment | None, filings: FilingIntelligence | None) -> dict[str, Any]:
    ctx: dict[str, Any] = {"news": None, "filings": None}
    if news is not None:
        ctx["news"] = _drop_none(
            {
                "score_100": news.score_100,
                "n_articles": news.n_articles,
                "n_used": news.n_used,
                "trend_7d": _r(news.trend_7d),
                "key_positives": news.key_positives,
                "key_negatives": news.key_negatives,
                "key_events": news.key_events,
                "articles": [
                    {
                        "news_id": a.news_id,
                        "sentiment": _r(a.sentiment),
                        "relevance": _r(a.relevance),
                        "event_type": a.event_type,
                        "grounded": a.grounded,
                    }
                    for a in news.articles
                ],
            }
        )
    if filings is not None:
        ctx["filings"] = _drop_none(
            {
                "summary": filings.summary,
                "earnings_highlights": filings.earnings_highlights,
                "guidance": filings.guidance,
                "tone": _r(filings.tone),
                "risk_factors": [
                    {
                        "title": r.title,
                        "severity": r.severity,
                        "category": r.category,
                        "source_ref": r.source_ref,
                    }
                    for r in filings.risk_factors
                ],
            }
        )
    return ctx


def prelim_context(prelim: ExecutiveSignal, currency: str) -> dict[str, Any]:
    return _drop_none(
        {
            "signal": prelim.signal.value,
            "composite": _r(prelim.composite),
            "confidence": prelim.confidence,
            "target_price": _price(prelim.target_price, currency),
            "upside_pct": _pct(prelim.upside),
            "horizon_days": prelim.horizon_days,
            "horizon_label": prelim.horizon_label,
            "pillars": {k: _r(v) for k, v in sorted(prelim.pillars.items())},
            "overrides": prelim.overrides,
        }
    )


def build_critic_user(
    inst: Instrument,
    quant_json: str,
    llm_json: str,
    prelim_json: str,
    docs: Sequence[Doc],
    *,
    lang: Any = "ko",
) -> str:
    blocks = [
        wrap_block("quant_json", quant_json),
        wrap_block("llm_json", llm_json),
        wrap_block("prelim_signal", prelim_json),
    ]
    if docs:
        blocks.append(wrap_documents(docs))
    header = [
        *instrument_header(inst, lang),
        tr(lang, f"원문 자료 수: {len(docs)}", f"Source documents: {len(docs)}"),
        document_notice(lang),
    ]
    return build_user_message(header, blocks)


# ------------------------------------------------------------------------------- critique


def critic_articles(items: Sequence[NewsItem], news: NewsSentiment | None, limit: int) -> list[NewsItem]:
    """The articles re-shown to the critic: highest relevance x |sentiment| from the news analysis first (unrated
    ones last), then the newest; at most ``limit``. Without a news analysis: the newest."""
    if not items:
        return []
    pool = select_articles(items, max(len(items), 1))  # newest first, unique ids
    rated = {a.news_id: a for a in (news.articles if news is not None else [])}

    def weight(n: NewsItem) -> float:
        a = rated.get(n.id)
        return a.relevance * max(abs(a.sentiment), 0.1) if a is not None else -1.0

    order = sorted(range(len(pool)), key=lambda i: (-weight(pool[i]), i))
    return [pool[i] for i in order[: max(0, int(limit))]]


def critic_filings(items: Sequence[FilingItem], filings: FilingIntelligence | None, limit: int) -> list[FilingItem]:
    """The filings re-shown to the critic: those behind high / critical risk factors first, then the usual order
    (earnings first, newest); at most ``limit``."""
    ordered = order_filings(items, max(len(items), 1))
    risky = [r.source_ref for r in (filings.risk_factors if filings is not None else []) if r.severity in RISK_PRIORITY]
    first = [f for f in ordered if f.id in risky]
    rest = [f for f in ordered if f.id not in risky]
    return [*first, *rest][: max(0, int(limit))]


def _evidence_is_thin(news: NewsSentiment | None, filings: FilingIntelligence | None) -> bool:
    news_thin = news is None or news.n_used == 0
    filings_thin = filings is None or not (filings.summary or filings.earnings_highlights or filings.risk_factors)
    return news_thin and filings_thin


async def critique(
    client: LLMLike,
    inst: Instrument,
    quant: QuantSignal,
    news: NewsSentiment | None,
    filings: FilingIntelligence | None,
    fundamentals: Fundamentals | None,
    prelim: ExecutiveSignal,
    *,
    news_items: Sequence[NewsItem] = (),
    filing_items: Sequence[FilingItem] = (),
    recorder: LLMRecorder | None = None,
    max_filing_tokens: int = CRITIC_FILING_TOKENS,
    max_news: int = CRITIC_MAX_NEWS,
    max_filings: int = CRITIC_MAX_FILINGS,
    lang: Any = None,
) -> CriticReview | None:
    """Adversarial review of ``prelim``. Returns None when the call fails (the failure is recorded).

    ``news_items`` / ``filing_items`` are the raw documents. A ``critical_flag`` can only be honoured
    with a quote grounded in them, so without raw documents it is always dropped.

    ``lang``: output language ("ko" / "en", or a Settings). None uses the recorder's language (Korean
    without a recorder).
    """
    lang = lang_of(lang if lang is not None else (recorder.lang if recorder is not None else None))
    rec = recorder if recorder is not None else LLMRecorder(lang=lang)
    label = task_label(TASK, lang)
    quant_json = compact_json(quant_context(inst, quant, fundamentals))
    llm_json = compact_json(llm_context(news, filings))
    prelim_json = compact_json(prelim_context(prelim, inst.currency))

    arts = critic_articles(news_items, news, max_news)
    prepared = (
        prepare_filings(critic_filings(filing_items, filings, max_filings), max_filing_tokens, lang=lang)
        if filing_items
        else []
    )
    docs = [
        Doc(
            n.id,
            article_source_text(n),
            {
                "kind": "news",
                "published": f"{n.published_at:%Y-%m-%d}",
                "source": n.source or None,
            },
        )
        for n in arts
    ]
    docs += [
        Doc(
            p.item.id,
            p.shown,
            {
                "kind": "filing",
                "form_type": p.item.form_type,
                "filed": f"{p.item.filed_at:%Y-%m-%d}",
            },
        )
        for p in prepared
    ]

    wire, usage, err = await client.call_structured(
        TASK,
        system_prompt(TASK, lang),
        build_critic_user(inst, quant_json, llm_json, prelim_json, docs, lang=lang),
        wire_model(TASK, lang),
    )
    rec.record(TASK, usage, err)
    if wire is None:
        return None

    ground_docs = {n.id: article_source_text(n) for n in arts}
    ground_docs.update({p.item.id: p.grounding for p in prepared})
    news_word, filing_word = tr(lang, "뉴스", "news"), tr(lang, "공시", "filing")
    labels = {n.id: f"{news_word} '{n.title}'" for n in arts}
    labels.update({p.item.id: f"{filing_word} '{source_label(p.item)}'" for p in prepared})
    index = NumberIndex([quant_json, llm_json, prelim_json, *ground_docs.values()])

    def unsupported(line: str) -> list[str]:
        return ungrounded_numbers(line, index, lenient=True, ignore=[inst.ticker])

    dropped = 0

    def keep(lines: Sequence[str]) -> list[str]:
        nonlocal dropped
        out = []
        for line in clean_lines(lines):
            if unsupported(line):
                dropped += 1
            else:
                out.append(line)
        return out[:MAX_POINTS]

    bull, bear, contra = (
        keep(wire.bull_points),
        keep(wire.bear_points),
        keep(wire.contradictions),
    )
    if dropped:
        rec.notice(
            tr(
                lang,
                f"{label}: 입력 자료에 없는 수치를 포함한 문장 {dropped}건 폐기",
                f"{label}: dropped {en_count(dropped, 'line')} with figures not in the inputs",
            )
        )
    thesis = " ".join(wire.thesis.split())
    bad = unsupported(thesis)
    if bad:
        eg = ", ".join(bad[:3])
        rec.notice(
            tr(
                lang,
                f"{label}: 투자 논지에 입력 자료로 확인되지 않는 수치({eg}) → 결정론적 논지로 대체",
                f"{label}: thesis has figures not supported by the inputs ({eg}) → replaced by the deterministic "
                "thesis",
            )
        )
        thesis = ""  # fusion treats an empty critic thesis as absent and builds its deterministic one

    adj = float(wire.confidence_adjustment)
    if adj > 0 and _evidence_is_thin(news, filings):
        rec.notice(
            tr(
                lang,
                f"{label}: 텍스트 근거가 부족해 양(+)의 신뢰도 조정(+{int(adj)})을 0으로 제한",
                f"{label}: positive confidence adjustment (+{int(adj)}) capped at 0 — thin text evidence",
            )
        )
        adj = 0.0

    critical, reason = False, None
    if wire.critical_flag:
        doc_id = None
        if wire.critical_category and (wire.critical_evidence or "").strip():
            # strict: a contiguous verbatim quote, not negated in the source. A fuzzy 3-gram match would
            # accept "There is substantial doubt ..." against "There is no substantial doubt ...".
            doc_id = locate_evidence(
                wire.critical_evidence or "",
                wire.critical_source_ref or "",
                ground_docs,
                strict=True,
            )
        if doc_id is None:
            rec.notice(
                tr(
                    lang,
                    f"{label}: critical_flag의 문서 근거 인용을 원문에서 그대로(부정 표현 없이) 확인할 수 없어 무시",
                    f"{label}: critical_flag ignored — its evidence quote could not be verified verbatim (and "
                    "not negated) in the source",
                )
            )
        else:
            critical = True
            cat_label = critical_category_label(str(wire.critical_category), lang)
            quote = " ".join((wire.critical_evidence or "").split())[:160]
            why = " ".join((wire.critical_reason or "").split())
            basis = tr(lang, "근거", "evidence")
            reason = f'{cat_label}: {why} ({basis}: "{quote}" — {labels[doc_id]})'.replace(":  (", ": (")

    review, clamped = to_critic_review(
        thesis=thesis,
        bull_points=bull,
        bear_points=bear,
        contradictions=contra,
        confidence_adjustment=adj,
        critical_flag=critical,
        critical_reason=reason,
    )
    if clamped:
        rec.notice(
            tr(
                lang,
                f"{label}: 신뢰도 조정값 {wire.confidence_adjustment} → 허용 범위 [-25, +5]로 보정",
                f"{label}: confidence adjustment {wire.confidence_adjustment} clamped to the allowed range [-25, +5]",
            )
        )
    return review
