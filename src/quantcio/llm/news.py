"""News sentiment: one Claude call per ticker, then deterministic scoring in code.

The LLM only scores articles one by one (sentiment, relevance, event type, verbatim evidence). The
score itself is computed here:

    used     = grounded evidence AND relevance >= settings.llm.min_relevance
    w_i      = relevance_i * 0.5 ** (age_days_i / settings.llm.news_half_life_days)
    score    = sum(w_i * s_i) / sum(w_i)            (0 when nothing is used)
    trend_7d = mean(s | age <= 7d) - mean(s | age > 7d)   over used articles (None if either side is empty)

Articles the model invented (unknown ids), ungrounded quotes, clamped values and key lines that contain
numbers absent from the articles are dropped or corrected. Each such case is counted as a recorder
NOTICE (the sentiment is still served). A failed call is a recorder ERROR, and the result is None.

Output language: ``settings.report.language`` selects the system prompt, the wire schema, the user-message
headers and the notice texts (Korean by default, English for "en"). Evidence stays a verbatim source quote.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from quantcio.config import Settings
from quantcio.i18n import lang_of, tr
from quantcio.llm.client import LLMLike, LLMRecorder, en_count, task_label
from quantcio.llm.prompts import (
    Doc,
    build_user_message,
    document_notice,
    system_prompt,
    wrap_documents,
)
from quantcio.llm.schemas import (
    NumberIndex,
    clean_lines,
    is_grounded,
    to_article_sentiment,
    to_news_sentiment,
    ungrounded_numbers,
    wire_model,
)
from quantcio.models import ArticleSentiment, Instrument, NewsItem, NewsSentiment

log = logging.getLogger(__name__)

TASK = "news"
KEY_LINES_MAX = 3
RECENT_WINDOW_DAYS = 7.0


COVERAGE_RETRY_SHARE = 0.8  # re-ask when fewer than 80% of the articles were rated
COVERAGE_RETRY_MIN = 3  # ... and at least this many are unrated
COVERAGE_RETRY_EFFORT = "medium"  # the retry thinks harder than the default low-effort news call
COVERAGE_RETRY_CHUNK = 10  # unrated articles per retry call (calls run concurrently, one round only)


def _utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


def article_source_text(item: NewsItem) -> str:
    """The exact text the model sees for an article (title line + summary). Evidence is grounded against it."""
    return f"{item.title}\n{item.summary}".strip()


_BRACKETS_RE = re.compile(r"[\[(（【<《][^\])）】>》]{0,20}[\])）】>》]")
_SOURCE_SUFFIX_RE = re.compile(r"\s+[-|–—]\s+[^-|–—]{1,30}$")
_NON_WORD_RE = re.compile(r"[^0-9a-z가-힣]+")
MIN_TITLE_BIGRAMS = 6  # shorter titles are never merged (too little text to call them the same story)


def title_bigrams(title: str) -> frozenset[str]:
    """Character bigrams of a normalised title: lower case, bracketed tags ("[속보]", "(종합)") and a trailing
    " - source" removed, only letters / digits / Hangul kept."""
    t = _SOURCE_SUFFIX_RE.sub("", _BRACKETS_RE.sub(" ", (title or "").lower()))
    t = _NON_WORD_RE.sub("", t)
    return frozenset(t[i : i + 2] for i in range(len(t) - 1))


def same_story(a: frozenset[str], b: frozenset[str], threshold: float) -> bool:
    """Jaccard similarity of two title bigram sets >= ``threshold`` (both titles long enough)."""
    if threshold >= 1.0 or len(a) < MIN_TITLE_BIGRAMS or len(b) < MIN_TITLE_BIGRAMS:
        return False
    return len(a & b) / len(a | b) >= threshold


def select_articles(news: Sequence[NewsItem], limit: int, dedupe_similarity: float | None = None) -> list[NewsItem]:
    """Newest first, de-duplicated by id, near-duplicate stories merged (the newest copy kept) when
    ``dedupe_similarity`` is given, capped at ``limit``."""
    seen: set[str] = set()
    kept: list[tuple[NewsItem, frozenset[str]]] = []
    merged = 0
    for n in sorted(news, key=lambda x: _utc(x.published_at), reverse=True):
        if n.id in seen:
            continue
        seen.add(n.id)
        grams = title_bigrams(n.title)
        if dedupe_similarity is not None and any(same_story(grams, g, dedupe_similarity) for _, g in kept):
            merged += 1
            continue
        kept.append((n, grams))
        if len(kept) >= max(1, limit):
            break
    if merged:
        log.info("news: merged %d near-duplicate articles", merged)
    return [n for n, _ in kept]


def instrument_header(inst: Instrument, lang: Any = "ko") -> list[str]:
    """The company line of a user message ("대상 기업: ..." / "Company: ...")."""
    en = lang_of(lang) == "en"
    extra = ", ".join(
        x
        for x in (
            inst.exchange,
            f"{'market' if en else '시장'} {inst.market.value}",
            inst.sector and f"{'sector' if en else '업종'} {inst.sector}",
        )
        if x
    )
    if en:
        return [f"Company: {inst.name} (ticker {inst.ticker}, {extra})"]
    return [f"대상 기업: {inst.name} (티커 {inst.ticker}, {extra})"]


def build_news_user(inst: Instrument, items: Sequence[NewsItem], now: datetime, *, lang: Any = "ko") -> str:
    """User message: company header, reference date and one ``<document>`` per article. All volatile content goes here.

    Only the reference DATE is included, not the clock time. The message is the response-cache key, so a
    minute-level timestamp would make every rerun a cache miss (and cost). Code computes the exact
    article ages for the time decay.
    """
    docs = [
        Doc(
            n.id,
            article_source_text(n),
            {
                "published": _utc(n.published_at).strftime("%Y-%m-%d %H:%M UTC"),
                "source": n.source or None,
                "lang": n.lang,
            },
        )
        for n in items
    ]
    day = f"{_utc(now):%Y-%m-%d}"
    header = [
        *instrument_header(inst, lang),
        tr(lang, f"기준일: {day} (UTC)", f"Reference date: {day} (UTC)"),
        tr(lang, f"기사 수: {len(items)}", f"Articles: {len(items)}"),
        document_notice(lang),
    ]
    return build_user_message(header, [wrap_documents(docs)])


def time_decay_weight(relevance: float, age_days: float, half_life_days: float) -> float:
    """w = relevance * 0.5 ** (age / half_life). A future-dated article counts as age 0."""
    hl = half_life_days if half_life_days > 0 else 7.0
    return relevance * 0.5 ** (max(0.0, age_days) / hl)


def aggregate_sentiment(
    articles: Sequence[ArticleSentiment],
    ages: dict[str, float],
    *,
    min_relevance: float,
    half_life_days: float,
) -> tuple[float, int, float | None]:
    """(score, n_used, trend_7d) from per-article outputs (see the module docstring for the formulas)."""
    used = [a for a in articles if a.grounded and a.relevance >= min_relevance]
    num = den = 0.0
    for a in used:
        w = time_decay_weight(a.relevance, ages.get(a.news_id, 0.0), half_life_days)
        num += w * a.sentiment
        den += w
    score = num / den if den > 0 else 0.0
    recent = [a.sentiment for a in used if ages.get(a.news_id, 0.0) <= RECENT_WINDOW_DAYS]
    older = [a.sentiment for a in used if ages.get(a.news_id, 0.0) > RECENT_WINDOW_DAYS]
    trend = (sum(recent) / len(recent) - sum(older) / len(older)) if recent and older else None
    return score, len(used), trend


async def analyze_news(
    client: LLMLike,
    inst: Instrument,
    news: Sequence[NewsItem],
    settings: Settings,
    *,
    recorder: LLMRecorder | None = None,
    now: datetime | None = None,
) -> NewsSentiment | None:
    """Score one ticker's news with a single LLM call. Returns None when there is no news or the call fails.

    A failed call goes to ``recorder.errors``. Every drop or correction goes to ``recorder.notices``. When
    ``recorder`` is None a local one is used, and its messages are still logged.
    """
    lang = lang_of(settings)
    rec = recorder if recorder is not None else LLMRecorder(lang=lang)
    if not news:
        return None
    now = _utc(now or datetime.now(UTC))
    limit = min(int(settings.data.max_news_items), int(settings.llm.news_max_articles))
    items = select_articles(news, limit, settings.llm.news_dedupe_similarity)
    by_id = {n.id: n for n in items}

    async def rate(batch: Sequence[NewsItem], effort: str | None = None):
        wire, usage, err = await client.call_structured(
            TASK,
            system_prompt(TASK, lang),
            build_news_user(inst, batch, now, lang=lang),
            wire_model(TASK, lang),
            effort=effort,
        )
        rec.record(TASK, usage, err)
        return wire

    wire = await rate(items)
    if wire is None:
        return None

    articles: list[ArticleSentiment] = []
    unknown = dup = ungrounded = clamped = 0

    def absorb(rated) -> None:
        nonlocal unknown, dup, ungrounded, clamped
        for w in rated:
            item = by_id.get(w.news_id.strip())
            if item is None:
                unknown += 1
                continue
            if any(a.news_id == item.id for a in articles):
                dup += 1
                continue
            grounded = is_grounded(w.evidence, article_source_text(item))
            art, was_clamped = to_article_sentiment(w, grounded=grounded)
            ungrounded += not grounded
            clamped += was_clamped
            articles.append(art)

    absorb(wire.articles)
    label = task_label(TASK, lang)
    # Coverage retry: the model occasionally stops after rating a handful of articles (seen live 2026-09-25: 1 of
    # 30). Re-ask ONCE for the unrated articles only and merge; never loop, never invent ratings.
    rated_ids = {a.news_id for a in articles}
    unrated = [n for n in items if n.id not in rated_ids]
    if len(unrated) >= COVERAGE_RETRY_MIN and len(articles) < COVERAGE_RETRY_SHARE * len(items):
        # same prompt at the same low effort tends to fail the same way (seen live): think harder once
        # small chunks: a long mixed-language batch is where the model skips work (seen live: 1 of 29 re-rated)
        chunks = [unrated[i : i + COVERAGE_RETRY_CHUNK] for i in range(0, len(unrated), COVERAGE_RETRY_CHUNK)]
        retries = await asyncio.gather(*(rate(c, effort=COVERAGE_RETRY_EFFORT) for c in chunks))
        before = len(articles)
        for retry in retries:
            if retry is None:
                continue
            absorb(retry.articles)
            if not (wire.key_positives or wire.key_negatives or wire.key_events):
                wire = retry  # the first answer had no key points at all
        if any(r is not None for r in retries):
            rec.notice(
                tr(
                    lang,
                    f"{label}: 평가 누락 {len(unrated)}건 재요청 → {len(articles) - before}건 추가 평가",
                    f"{label}: re-asked for {en_count(len(unrated), 'unrated article')} → "
                    f"{len(articles) - before} more rated",
                )
            )
    missing = len(items) - len(articles)

    if unknown:
        rec.notice(
            tr(
                lang,
                f"{label}: 입력에 없는 기사 id {unknown}건 폐기",
                f"{label}: dropped {en_count(unknown, 'article id')} not in the input",
            )
        )
    if dup:
        rec.notice(
            tr(
                lang,
                f"{label}: 중복 기사 평가 {dup}건 폐기",
                f"{label}: dropped {en_count(dup, 'duplicate article rating')}",
            )
        )
    if missing:
        rec.notice(
            tr(
                lang,
                f"{label}: 모델이 평가하지 않은 기사 {missing}건(점수에서 제외)",
                f"{label}: {en_count(missing, 'article')} not rated by the model (excluded from the score)",
            )
        )
    if ungrounded:
        rec.notice(
            tr(
                lang,
                f"{label}: 근거 인용이 원문과 일치하지 않은 기사 {ungrounded}건 점수에서 제외",
                f"{label}: {en_count(ungrounded, 'article')} excluded from the score — evidence quote does not match "
                "the source",
            )
        )
    if clamped:
        rec.notice(
            tr(
                lang,
                f"{label}: 범위를 벗어난 점수 {clamped}건 보정(clamp)",
                f"{label}: {en_count(clamped, 'out-of-range score')} clamped",
            )
        )

    ages = {
        a.news_id: max(0.0, (now - _utc(by_id[a.news_id].published_at)).total_seconds() / 86400.0) for a in articles
    }
    score, n_used, trend = aggregate_sentiment(
        articles,
        ages,
        min_relevance=settings.llm.min_relevance,
        half_life_days=settings.llm.news_half_life_days,
    )

    index = NumberIndex(article_source_text(n) for n in items)
    dropped_lines = 0

    def grounded_lines(lines: Sequence[str]) -> list[str]:
        nonlocal dropped_lines
        kept = []
        for line in clean_lines(lines):
            if ungrounded_numbers(line, index, ignore=[inst.ticker]):
                dropped_lines += 1
                continue
            kept.append(line)
        return kept[:KEY_LINES_MAX]

    positives = grounded_lines(wire.key_positives)
    negatives = grounded_lines(wire.key_negatives)
    events = grounded_lines(wire.key_events)
    if dropped_lines:
        rec.notice(
            tr(
                lang,
                f"{label}: 원문에 없는 숫자를 포함한 요약 문장 {dropped_lines}건 폐기",
                f"{label}: dropped {en_count(dropped_lines, 'summary line')} with figures not in the articles",
            )
        )

    ordered = sorted(articles, key=lambda a: ages.get(a.news_id, 0.0))
    return to_news_sentiment(
        score=score,
        n_articles=len(items),
        n_used=n_used,
        trend_7d=trend,
        key_positives=positives,
        key_negatives=negatives,
        key_events=events,
        articles=ordered,
    )
