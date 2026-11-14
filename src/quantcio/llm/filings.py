"""Filings and earnings intelligence: one Claude call per ticker, then code-side grounding.

- Earnings releases come first, then the newest filings, capped at ``settings.data.max_filings``; a filing with the
  same form type and filing date as a kept one and near-identical text is dropped (duplicate). Each text becomes a
  SALIENT EXCERPT of at most ``llm.filing_max_tokens`` estimated tokens (the head for context, then the paragraphs
  with the most figures and key terms, in document order, omissions marked), and all filings of a ticker share
  ``llm.filings_total_tokens``. Budgets are in estimated tokens (``approx_tokens``), so Korean text (about 1.2 tokens
  per character) is not twice as expensive as English (about 0.35).
- The fundamentals block sent with the filings carries only filing-stable fields (no date, price, market cap or
  price ratios), so an unchanged set of filings hits the response cache on later days.
- Every figure in ``earnings_highlights`` and ``summary`` must appear in the text the model was shown,
  after commas, %, $, 조/억 and billion/million are normalised. A line with any unsupported figure is
  dropped, and the count is recorded as a NOTICE.
- Every risk factor needs a ``source_ref`` that resolves to a provided document, and an evidence quote
  grounded in that document. Anything else is dropped and counted (NOTICE).
- A failed call is an ERROR, and the result is None. Drops, merges and clamps are NOTICES, and the
  result is still served (see ``LLMRecorder``).
- ``web_research`` (optional, US only, ``settings.llm.web_research``): server ``web_search`` for recent
  earnings-call coverage. Only the verbatim ``cited_text`` snippets of cited sources become a
  supplementary document. The model's own uncited prose is discarded, so it cannot ground numbers.
- Output language: ``settings.report.language`` selects the system prompts, the wire schema, the user-message
  headers, the truncation marker and the notice texts (Korean by default, English for "en").
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from quantcio.config import Settings
from quantcio.i18n import lang_of, tr
from quantcio.llm.client import LLMLike, LLMRecorder, en_count, task_label
from quantcio.llm.news import instrument_header
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
    locate_evidence,
    normalize_text,
    to_filing_intelligence,
    to_risk_factor,
    ungrounded_numbers,
    wire_model,
)
from quantcio.models import (
    FilingIntelligence,
    FilingItem,
    Fundamentals,
    Instrument,
    Market,
    RiskFactor,
    text_id,
)

log = logging.getLogger(__name__)

TASK = "filings"
WEB_TASK = "web_research"
HEAD_FRACTION = 0.7
EXCERPT_HEAD_FRACTION = 0.25  # salient excerpt: share of the budget kept from the document start (cover, period)
EXCERPT_CHUNK = 600  # a paragraph longer than this is split at sentence ends into pieces of about this size
MIN_DOC_TOKENS = 300  # a filing that would get less than this of the shared budget is left out
EMPTY_RETRY_EFFORT = "high"  # one re-ask when a substantial document set came back with an empty analysis
EMPTY_RETRY_MIN_CHARS = 800
DUP_SIMILARITY = 0.8  # same form + date and this 3-gram Jaccard on the first 3,000 chars = duplicate filing
KEY_TERMS = (
    "매출",
    "영업이익",
    "영업손실",
    "순이익",
    "순손실",
    "실적",
    "전망",
    "가이던스",
    "위험",
    "리스크",
    "소송",
    "감사의견",
    "계속기업",
    "배당",
    "자기주식",
    "유상증자",
    "전환사채",
    "합병",
    "분할",
    "수주",
    "revenue",
    "net income",
    "net loss",
    "operating income",
    "earnings",
    "guidance",
    "outlook",
    "risk",
    "litigation",
    "impairment",
    "dividend",
    "repurchase",
    "margin",
    "cash flow",
    "going concern",
    "material weakness",
    "backlog",
)
STABLE_FUNDAMENTAL_FIELDS = (
    "eps",
    "forward_eps",
    "bps",
    "roe",
    "operating_margin",
    "revenue_growth_yoy",
    "earnings_growth_yoy",
    "next_earnings_date",
    "next_earnings_is_estimate",
)
_NUM_RE = re.compile(r"\d[\d,.]*")
_HANGUL_RE = re.compile(r"[가-힣]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_SENT_END_RE = re.compile(r"(?<=[.!?。다])\s+")
MAX_HIGHLIGHTS = 5
MAX_RISKS = 5
WEB_RESEARCH_FORM = "웹 리서치(실적 콜)"
WEB_RESEARCH_FORM_EN = "Web research (earnings call)"
WEB_RESEARCH_SOURCE = "Claude web_search"
US_EARNINGS_DOMAINS: tuple[str, ...] = (
    "fool.com",
    "reuters.com",
    "cnbc.com",
    "sec.gov",
    "marketwatch.com",
    "investing.com",
    "seekingalpha.com",
    "finance.yahoo.com",
)
_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


# ------------------------------------------------------------------------------- helpers


def truncate_head_tail(
    text: str, max_chars: int, head_fraction: float = HEAD_FRACTION, *, lang: Any = "ko"
) -> tuple[str, str, bool]:
    """Keep the head and the tail of a long text.

    Returns ``(shown_text, grounding_text, truncated)``. ``shown_text`` carries a "[중략]" marker (English:
    "[omitted: N chars]"). ``grounding_text`` is the head and tail WITHOUT the marker, so the marker's own
    count can never ground a number.
    """
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, text, False
    head_n = max(1, int(max_chars * head_fraction))
    tail_n = max(1, max_chars - head_n)
    head, tail = text[:head_n], text[-tail_n:]
    omitted = len(text) - head_n - tail_n
    marker = tr(lang, f"…[중략: {omitted:,}자 생략]…", f"…[omitted: {omitted:,} chars]…")
    shown = f"{head}\n{marker}\n{tail}"
    return shown, f"{head}\n{tail}", True


def approx_tokens(text: str) -> int:
    """Estimated Claude tokens of ``text``: 1.2 per Hangul syllable, 0.3 per Latin letter, 0.6 per other character
    (digits, spaces, punctuation). Calibrated 2026-09-27 with the token-counting API on KR / US filings and news
    (about +-20%)."""
    text = text or ""
    h = len(_HANGUL_RE.findall(text))
    a = len(_LATIN_RE.findall(text))
    return int(1.2 * h + 0.3 * a + 0.6 * (len(text) - h - a)) + 1


def chars_for_tokens(text: str, tokens: int) -> int:
    """How many characters of ``text`` fit ``tokens`` estimated tokens (the text's own characters-per-token)."""
    return int(max(0, tokens) * len(text or "") / max(approx_tokens(text), 1))


def _pieces(text: str) -> list[str]:
    """Paragraphs of ``text`` (blank-line or line breaks); a long paragraph is split at sentence ends."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n|\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= EXCERPT_CHUNK:
            out.append(para)
            continue
        cur = ""
        for sent in _SENT_END_RE.split(para):
            if cur and len(cur) + len(sent) + 1 > EXCERPT_CHUNK:
                out.append(cur)
                cur = sent
            else:
                cur = f"{cur} {sent}".strip()
        if cur:
            out.append(cur)
    return out


def _salience(piece: str) -> float:
    low = piece.lower()
    terms = sum(1 for k in KEY_TERMS if k in low)
    nums = min(len(_NUM_RE.findall(piece)), 8)
    return 2.0 * terms + 0.5 * nums


def salient_excerpt(text: str, max_chars: int, *, lang: Any = "ko") -> tuple[str, str, bool]:
    """At most ``max_chars`` of a long text: the head (``EXCERPT_HEAD_FRACTION`` of the budget) plus the most
    salient paragraphs (figures and ``KEY_TERMS``), in document order.

    Returns ``(shown_text, grounding_text, truncated)`` like ``truncate_head_tail``: omissions carry a "[중략]"
    marker in ``shown_text``; ``grounding_text`` holds the kept pieces WITHOUT markers (a marker's own count can
    never ground a number).
    """
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, text, False
    head_n = max(1, int(max_chars * EXCERPT_HEAD_FRACTION))
    head = text[:head_n]
    rest = _pieces(text[head_n:])
    budget = max_chars - head_n
    ranked = sorted(range(len(rest)), key=lambda i: (-_salience(rest[i]), i))
    chosen: set[int] = set()
    used = 0
    for i in ranked:
        if _salience(rest[i]) <= 0:
            break
        if used + len(rest[i]) + 1 > budget:
            continue
        chosen.add(i)
        used += len(rest[i]) + 1
    if not chosen:  # nothing salient: fall back to the plain head / tail cut
        return truncate_head_tail(text, max_chars, lang=lang)
    shown_parts, ground_parts = [head], [head]
    gap = 0
    for i, piece in enumerate(rest):
        if i in chosen:
            if gap:
                shown_parts.append(tr(lang, f"…[중략: {gap:,}자 생략]…", f"…[omitted: {gap:,} chars]…"))
            shown_parts.append(piece)
            ground_parts.append(piece)
            gap = 0
        else:
            gap += len(piece) + 1
    if gap:
        shown_parts.append(tr(lang, f"…[중략: {gap:,}자 생략]…", f"…[omitted: {gap:,} chars]…"))
    return "\n".join(shown_parts), "\n".join(ground_parts), True


def _trigrams(text: str) -> frozenset[str]:
    t = re.sub(r"\s+", " ", (text or "")[:3000])
    return frozenset(t[i : i + 3] for i in range(len(t) - 2))


def order_filings(filings: Sequence[FilingItem], limit: int) -> list[FilingItem]:
    """Earnings releases first, then the newest. De-duplicated by id and by content (same form type and filing
    date with near-identical text, e.g. an amended copy filed the same day), capped at ``limit``."""

    def ts(f: FilingItem) -> float:
        dt = f.filed_at if f.filed_at.tzinfo else f.filed_at.replace(tzinfo=UTC)
        return dt.timestamp()

    seen: set[str] = set()
    out: list[FilingItem] = []
    grams: list[frozenset[str]] = []
    for f in sorted(filings, key=lambda f: (not f.is_earnings, -ts(f))):
        if f.id in seen:
            continue
        seen.add(f.id)
        g = _trigrams(f.text)
        dup = False
        for k, other in enumerate(out):
            if other.form_type == f.form_type and other.filed_at.date() == f.filed_at.date() and g and grams[k]:
                if len(g & grams[k]) / len(g | grams[k]) >= DUP_SIMILARITY:
                    dup = True
                    break
        if dup:
            log.info("filings: dropped duplicate %s %s (%s)", f.form_type, f.filed_at.date(), f.id)
            continue
        out.append(f)
        grams.append(g)
    return out[: max(1, limit)]


def source_label(f: FilingItem) -> str:
    """Human-readable reference used in ``risk_factors[].source_ref`` and ``sources``."""
    return f"{f.title} ({f.filed_at:%Y-%m-%d})"


def fundamentals_json(fund: Fundamentals | None, *, stable_only: bool = False) -> str:
    """Fundamentals as JSON. ``stable_only`` keeps ``STABLE_FUNDAMENTAL_FIELDS`` (no date, price, market cap, price
    ratios or analyst targets), so the filings message only changes when the filings or the reported figures do."""
    if fund is None:
        return "{}"
    data = fund.model_dump(mode="json", exclude_none=True)
    if stable_only:
        data = {k: v for k, v in data.items() if k in STABLE_FUNDAMENTAL_FIELDS}
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class PreparedFiling:
    item: FilingItem
    shown: str  # title + (truncated) text shown to the model
    grounding: str  # title + head/tail without the truncation marker
    truncated: bool


def prepare_filings(
    filings: Sequence[FilingItem], max_tokens: int, *, lang: Any = "ko", total_tokens: int | None = None
) -> list[PreparedFiling]:
    """Salient excerpts of ``filings`` in order: each at most ``max_tokens`` estimated tokens, all together at most
    ``total_tokens`` (a filing that would get less than ``MIN_DOC_TOKENS`` of what is left is not shown)."""
    out = []
    left = int(total_tokens) if total_tokens else None
    for f in filings:
        cap = int(max_tokens)
        if left is not None:
            if left < MIN_DOC_TOKENS and out:
                log.info("filings: shared budget used up - %s (%s) not shown", f.form_type, f.id)
                continue
            cap = min(cap, max(left, MIN_DOC_TOKENS))
        shown, grounding, truncated = salient_excerpt(f.text, chars_for_tokens(f.text, cap), lang=lang)
        if left is not None:
            left -= approx_tokens(shown)
        out.append(
            PreparedFiling(
                f,
                f"{f.title}\n{shown}".strip(),
                f"{f.title}\n{grounding}".strip(),
                truncated,
            )
        )
    return out


def build_filings_user(
    inst: Instrument, prepared: Sequence[PreparedFiling], fund: Fundamentals | None, *, lang: Any = "ko"
) -> str:
    docs = [
        Doc(
            p.item.id,
            p.shown,
            {
                "form_type": p.item.form_type,
                "filed": f"{p.item.filed_at:%Y-%m-%d}",
                "source": p.item.source or None,
                "is_earnings": p.item.is_earnings,
                "truncated": p.truncated,
            },
        )
        for p in prepared
    ]
    n = len(prepared)
    header = [
        *instrument_header(inst, lang),
        tr(lang, f"통화: {inst.currency}", f"Currency: {inst.currency}"),
        tr(lang, f"자료 수: {n} (실적 발표 자료 우선 정렬)", f"Documents: {n} (earnings releases first)"),
        document_notice(lang),
    ]
    return build_user_message(
        header,
        [wrap_block("fundamentals", fundamentals_json(fund, stable_only=True)), wrap_documents(docs)],
    )


# ----------------------------------------------------------------------------- analysis


async def analyze_filings(
    client: LLMLike,
    inst: Instrument,
    filings: Sequence[FilingItem],
    fundamentals: Fundamentals | None,
    settings: Settings,
    *,
    recorder: LLMRecorder | None = None,
    supplementary: Sequence[FilingItem] = (),
) -> FilingIntelligence | None:
    """Summarise one ticker's filings and earnings. Returns None when there are no filings or the call fails.

    ``supplementary`` documents (e.g. from ``web_research``) are appended after the ``max_filings`` cap.
    """
    lang = lang_of(settings)
    rec = recorder if recorder is not None else LLMRecorder(lang=lang)
    selected = order_filings(filings, settings.data.max_filings) if filings else []
    selected += [s for s in supplementary if s.id not in {f.id for f in selected}]
    if not selected:
        return None
    prepared = prepare_filings(
        selected, settings.llm.filing_max_tokens, lang=lang, total_tokens=settings.llm.filings_total_tokens
    )
    label = task_label(TASK, lang)
    n_trunc = sum(p.truncated for p in prepared)
    if n_trunc:
        log.info("%s %s: %d건 문서를 핵심 문단 발췌로 축약", inst.ticker, label, n_trunc)

    user = build_filings_user(inst, prepared, fundamentals, lang=lang)
    wire, usage, err = await client.call_structured(TASK, system_prompt(TASK, lang), user, wire_model(TASK, lang))
    rec.record(TASK, usage, err)
    if wire is None:
        return None
    shown_chars = sum(len(p.shown) for p in prepared)
    if not (wire.summary or wire.earnings_highlights or wire.risk_factors) and shown_chars >= EMPTY_RETRY_MIN_CHARS:
        # the model occasionally answers a substantial document set with an EMPTY analysis (seen live 2026-09-27 on
        # 005930; the same input gave a full answer when asked again). Re-ask ONCE at a higher effort (a different
        # response-cache key, so a cached empty answer is not served again); keep the empty answer if that fails too
        retry, usage2, err2 = await client.call_structured(
            TASK, system_prompt(TASK, lang), user, wire_model(TASK, lang), effort=EMPTY_RETRY_EFFORT
        )
        rec.record(TASK, usage2, err2)
        filled = retry is not None and (retry.summary or retry.earnings_highlights or retry.risk_factors)
        rec.notice(
            tr(
                lang,
                f"{label}: 빈 분석 응답 → 1회 재요청({'결과 확보' if filled else '여전히 비어 있음'})",
                f"{label}: empty analysis → re-asked once ({'filled' if filled else 'still empty'})",
            )
        )
        if filled:
            wire = retry

    index = NumberIndex(p.grounding for p in prepared)
    ignore = [inst.ticker]

    def keep_grounded(lines: Sequence[str], what: str, what_en: str, limit: int) -> list[str]:
        kept, dropped = [], []
        for line in clean_lines(lines):
            bad = ungrounded_numbers(line, index, ignore=ignore)
            (dropped if bad else kept).append((line, bad))
        if dropped:
            eg = ", ".join(dropped[0][1][:3])
            rec.notice(
                tr(
                    lang,
                    f"{label}: 원문에서 확인되지 않는 수치가 있는 {what} {len(dropped)}건 폐기 (예: {eg})",
                    f"{label}: dropped {en_count(len(dropped), what_en)} with figures not found in the source "
                    f"(e.g. {eg})",
                )
            )
        return [line for line, _ in kept][:limit]

    highlights = keep_grounded(wire.earnings_highlights, "실적 하이라이트", "earnings highlight", MAX_HIGHLIGHTS)
    summary = keep_grounded(wire.summary, "요약 문장", "summary line", 3)

    docs = {p.item.id: p.grounding for p in prepared}
    titles = {p.item.id: p.item.title for p in prepared}
    by_id = {p.item.id: p.item for p in prepared}
    risks: list[RiskFactor] = []
    no_ref = ungrounded = 0
    for rf in wire.risk_factors:
        if not rf.source_ref.strip():
            no_ref += 1
            continue
        doc_id = locate_evidence(rf.evidence, rf.source_ref, docs, titles)
        if doc_id is None:
            ungrounded += 1
            continue
        risks.append(to_risk_factor(rf, source_ref=source_label(by_id[doc_id])))
    if no_ref:
        rec.notice(
            tr(
                lang,
                f"{label}: source_ref가 없는 위험 요인 {no_ref}건 폐기",
                f"{label}: dropped {en_count(no_ref, 'risk factor')} without a source_ref",
            )
        )
    if ungrounded:
        rec.notice(
            tr(
                lang,
                f"{label}: 근거 인용을 원문에서 찾을 수 없는 위험 요인 {ungrounded}건 폐기",
                f"{label}: dropped {en_count(ungrounded, 'risk factor')} whose evidence quote is not in the source",
            )
        )
    # A repeated risk would double its llm_pillar penalty: keep one per title (the most severe).
    unique: dict[str, RiskFactor] = {}
    for r in sorted(risks, key=lambda r: _SEVERITY_RANK[r.severity]):
        unique.setdefault(normalize_text(r.title), r)
    if len(unique) < len(risks):
        n_dup = len(risks) - len(unique)
        rec.notice(
            tr(
                lang,
                f"{label}: 중복 위험 요인 {n_dup}건 병합",
                f"{label}: merged {en_count(n_dup, 'duplicate risk factor')}",
            )
        )
    risks = list(unique.values())[:MAX_RISKS]

    fi, tone_clamped = to_filing_intelligence(
        wire,
        summary=summary,
        earnings_highlights=highlights,
        risk_factors=risks,
        sources=[source_label(p.item) for p in prepared],
    )
    if tone_clamped:
        rec.notice(tr(lang, f"{label}: 범위를 벗어난 tone 보정(clamp)", f"{label}: out-of-range tone clamped"))
    return fi


# ------------------------------------------------------------------------- web research


@dataclass(frozen=True)
class WebResearch:
    """Cited earnings-call coverage from the server web_search tool."""

    filing: FilingItem  # supplementary document: verbatim cited snippets only
    sources: list[dict[str, Any]]  # {url, title, page_age, cited_text: [...]}
    text: str  # the model's own summary (NOT used for grounding)
    lang: str = "ko"  # language of the labels ("웹: ..." / "Web: ...")

    def source_labels(self) -> list[str]:
        prefix = tr(self.lang, "웹", "Web")
        return [f"{prefix}: {s.get('title') or s['url']} <{s['url']}>" for s in self.sources if s.get("cited_text")]


def _web_user(inst: Instrument, as_of: date, lang: Any = "ko") -> str:
    if lang_of(lang) == "en":
        return build_user_message(
            [*instrument_header(inst, lang), f"Reference date: {as_of:%Y-%m-%d}"],
            [
                f"Research the key points of the most recent quarterly earnings release and conference call of "
                f"{inst.name} ({inst.ticker}). Summarise management guidance, key figures such as revenue and "
                "profit, and the main issues raised in the Q&A, citing sources."
            ],
        )
    return build_user_message(
        [*instrument_header(inst), f"기준일: {as_of:%Y-%m-%d}"],
        [
            f"{inst.name}({inst.ticker})의 가장 최근 분기 실적 발표와 컨퍼런스콜의 핵심 내용을 조사하라. "
            "경영진 가이던스, 매출·이익 등 주요 수치, 질의응답의 주요 쟁점을 출처와 함께 정리하라."
        ],
    )


async def web_research(
    client: LLMLike,
    inst: Instrument,
    settings: Settings,
    *,
    as_of: date | None = None,
    recorder: LLMRecorder | None = None,
    allowed_domains: Sequence[str] = US_EARNINGS_DOMAINS,
    max_uses: int = 5,
) -> WebResearch | None:
    """Optional US earnings-call coverage. Returns None when disabled, not a US instrument, or failed.

    Guarded: it only runs when ``settings.llm.web_research`` is true (extra cost, $10 per 1,000 searches plus
    tokens). Nothing is ever raised. A failed call goes to ``recorder.errors``. A result without citations,
    or a client without web search, goes to ``recorder.notices``, because the filings analysis is still
    served without the supplement.
    """
    if not settings.llm.web_research:
        return None
    if inst.market is not Market.US:
        log.debug("web_research: %s is not a US instrument; skipped", inst.ticker)
        return None
    lang = lang_of(settings)
    rec = recorder if recorder is not None else LLMRecorder(lang=lang)
    label = task_label(WEB_TASK, lang)
    search = getattr(client, "web_search", None)
    if search is None:
        rec.notice(
            tr(
                lang,
                f"{label}: 클라이언트가 web_search를 지원하지 않아 건너뜀",
                f"{label}: skipped — the client does not support web_search",
            )
        )
        return None
    as_of = as_of or datetime.now(UTC).date()
    text, sources, usage, err = await search(
        WEB_TASK,
        system_prompt(WEB_TASK, lang),
        _web_user(inst, as_of, lang),
        allowed_domains=list(allowed_domains),
        max_uses=max_uses,
    )
    rec.record(WEB_TASK, usage, err)
    if err is not None:
        return None
    cited = [s for s in sources if s.get("cited_text")]
    if not cited:
        rec.notice(
            tr(
                lang,
                f"{label}: 인용된 출처가 없어 결과를 사용하지 않음",
                f"{label}: no cited sources — result not used",
            )
        )
        return None
    lines = []
    untitled = tr(lang, "제목 없음", "untitled")
    for i, s in enumerate(cited, 1):
        lines.append(f"[{i}] {s.get('title') or untitled}")
        lines.extend(f'"{c}"' for c in s["cited_text"])
    filing = FilingItem(
        id=text_id("web_research", inst.ticker, as_of.isoformat()),
        title=tr(
            lang,
            f"최근 실적 발표 컨퍼런스콜 웹 리서치 (인용 출처 {len(cited)}건)",
            f"Web research on the latest earnings call ({en_count(len(cited), 'cited source')})",
        ),
        form_type=tr(lang, WEB_RESEARCH_FORM, WEB_RESEARCH_FORM_EN),
        filed_at=datetime(as_of.year, as_of.month, as_of.day, tzinfo=UTC),
        url=cited[0]["url"],
        text="\n".join(lines),
        is_earnings=False,
        source=WEB_RESEARCH_SOURCE,
    )
    return WebResearch(filing=filing, sources=cited, text=text, lang=lang)
