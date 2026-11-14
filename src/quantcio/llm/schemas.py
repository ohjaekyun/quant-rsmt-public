"""LLM wire schemas, code-side grounding checks, and the wire -> domain mapping.

Everything between the raw JSON the model returns and the shared domain models in
``quantcio.models`` lives here.

Wire-schema rules (from the verified SDK probe, docs/sources/anthropic-sdk.json):

- No ``dict`` fields. ``transform_schema`` turns them into ``{"properties": {}, "additionalProperties": false}``,
  so the model could only return ``{}``.
- No defaults. A field with a default is dropped from ``required``. Every key is required; nullable
  fields use ``Optional[X]`` *without* a default.
- No single-value ``Literal``, no ``pattern``, no ge/le/min_length. The API does not enforce these
  (they only go into the description). Numeric ranges are **clamped in code after validation**, so
  an out-of-range number never costs a re-sample.

Grounding rules (ARCHITECTURE.md, "Design principles"):

- An ``evidence`` quote is grounded when its normalised form (NFKC, case-folded, whitespace, quote and
  invisible format characters removed) is a substring of the normalised source. It is also grounded
  when at least 80% of its character 3-grams occur inside ONE contiguous span of the source about the
  quote's length. Counting 3-grams over the whole document would let a fabricated sentence pass
  against any long filing. A quote with a ``...`` ellipsis is grounded when its parts occur in order,
  each gap at most ``MAX_ELLIPSIS_GAP`` characters.
- ``strict=True`` (used for the critic's ``critical_flag``) accepts only a contiguous verbatim quote,
  and only when at least one occurrence is not negated in the source ("There is no substantial
  doubt ...", "... 해당하지 않습니다").
- A number in a highlight is grounded when it appears in the source text after commas, %, $, 조/억/만
  and trillion/billion/million (tn/bn/mn/T/B/M/K) are normalised, so an English rendering of a Korean amount
  with the same value ("KRW 10.4 trillion" for "10조 4,000억원", "KRW 642 billion" for "6,420억원") is grounded.
  A currency code glued to the digits ("KRW10.4tn", "W642bn") is split off first. A scaled figure may
  differ from the source by rounding at its stated precision, but by no more than ``SCALED_REL_TOL`` (5%).
  A percentage or a part of a compound amount never grounds a scaled amount. Period and ordinal tokens
  (3분기, 2026년, Q3, FY2026, 52주) are not figures and are ignored, as are their English forms (3rd,
  52-week, 12 months, 2 years). Share counts (3,000,000주) are figures.
"""

from __future__ import annotations

import bisect
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel, Field

from quantcio.i18n import lang_of
from quantcio.models import (
    ArticleSentiment,
    CriticReview,
    EventType,
    FilingIntelligence,
    NewsSentiment,
    RiskFactor,
)

# =============================================================================== wire schemas

Severity = Literal["low", "medium", "high", "critical"]
RiskCategory = Literal[
    "financial",
    "legal_regulatory",
    "operational",
    "market_demand",
    "competition",
    "governance",
    "macro_fx",
    "liquidity_dilution",
    "other",
]
Guidance = Literal["raised", "maintained", "lowered", "none", "unknown"]
CriticalCategory = Literal[
    "fraud_embezzlement",
    "audit_opinion",
    "delisting_halt",
    "going_concern",
    "debt_default",
    "dilutive_issuance",
]

CRITICAL_CATEGORY_LABELS: dict[str, str] = {
    "fraud_embezzlement": "횡령·배임·회계부정",
    "audit_opinion": "감사의견 거절·부적정",
    "delisting_halt": "상장폐지·거래정지",
    "going_concern": "계속기업 존속 불확실성",
    "debt_default": "채무불이행·부도·회생",
    "dilutive_issuance": "20% 초과 대규모 희석성 증자",
}
CRITICAL_CATEGORY_LABELS_EN: dict[str, str] = {
    "fraud_embezzlement": "fraud / embezzlement / accounting fraud",
    "audit_opinion": "audit opinion disclaimer / adverse",
    "delisting_halt": "delisting / trading halt",
    "going_concern": "going-concern uncertainty",
    "debt_default": "default / insolvency / rehabilitation",
    "dilutive_issuance": "dilutive issuance > 20%",
}


class ArticleWire(BaseModel):
    """One analysed news article (wire)."""

    news_id: str = Field(description="입력 <document>의 id 속성값을 그대로 복사")
    sentiment: float = Field(description="-1.0(치명적 악재) ~ +1.0(결정적 호재), 0.0 = 중립")
    relevance: float = Field(description="0.0 ~ 1.0, 기사가 이 기업에 관한 정도")
    event_type: EventType
    evidence: str = Field(description="이 기사 제목/요약에서 그대로 복사한 근거 구절")


class NewsWire(BaseModel):
    """Per-ticker news analysis (wire)."""

    articles: list[ArticleWire]
    key_positives: list[str] = Field(description="핵심 긍정 요인, 최대 3개, 한국어 한 줄")
    key_negatives: list[str] = Field(description="핵심 부정 요인, 최대 3개, 한국어 한 줄")
    key_events: list[str] = Field(description="주가에 영향을 줄 구체적 이벤트, 최대 3개, 한국어 한 줄")


class RiskFactorWire(BaseModel):
    """One risk factor taken from a filing (wire)."""

    title: str
    detail: str
    severity: Severity
    category: RiskCategory
    source_ref: str = Field(description="근거 문서의 id 속성값을 그대로 복사")
    evidence: str = Field(description="그 문서 원문에서 그대로 복사한 근거 구절")


class FilingsWire(BaseModel):
    """Per-ticker filings / earnings analysis (wire)."""

    summary: list[str] = Field(description="한국어 불릿 최대 3개")
    earnings_highlights: list[str] = Field(description="실적 핵심 수치, 숫자는 원문 그대로, 최대 5개")
    guidance: Guidance
    tone: float = Field(description="-1.0 ~ +1.0 공시·실적의 종합 방향")
    risk_factors: list[RiskFactorWire]


class CriticWire(BaseModel):
    """Adversarial review of the preliminary signal (wire)."""

    thesis: str = Field(description="1~2문장 한국어 투자 논지")
    bull_points: list[str]
    bear_points: list[str]
    contradictions: list[str]
    confidence_adjustment: int = Field(description="정수 -25 ~ +5, 근거가 빈약하면 0")
    critical_flag: bool
    critical_category: CriticalCategory | None
    critical_reason: str | None
    critical_evidence: str | None = Field(description="critical_flag가 true일 때 문서 원문 그대로의 인용")
    critical_source_ref: str | None = Field(description="critical_evidence가 나온 문서의 id")


# English-output twins (report.language == "en"): subclasses that only override field DESCRIPTIONS, so the JSON
# schema never asks for Korean. Field names, order, types and validation are identical, and every EN instance is
# an instance of its Korean base, so the wire -> domain converters below work unchanged.


class ArticleWireEN(ArticleWire):
    """One analysed news article (wire)."""

    news_id: str = Field(description="copy the id attribute of the input <document> verbatim")
    sentiment: float = Field(description="-1.0 (critical negative) to +1.0 (decisive positive), 0.0 = neutral")
    relevance: float = Field(description="0.0 to 1.0, how much the article is about this company")
    evidence: str = Field(
        description="passage copied verbatim from this article's headline/summary, in its original language"
    )


class NewsWireEN(NewsWire):
    """Per-ticker news analysis (wire)."""

    articles: list[ArticleWireEN]
    key_positives: list[str] = Field(description="key positive drivers, at most 3, one line of English each")
    key_negatives: list[str] = Field(description="key negative drivers, at most 3, one line of English each")
    key_events: list[str] = Field(
        description="specific events that could move the share price, at most 3, one line of English each"
    )


class RiskFactorWireEN(RiskFactorWire):
    """One risk factor taken from a filing (wire)."""

    title: str = Field(description="short English title")
    detail: str = Field(description="one or two English sentences")
    source_ref: str = Field(description="copy the id attribute of the supporting document verbatim")
    evidence: str = Field(description="supporting passage copied verbatim from that document, in its original language")


class FilingsWireEN(FilingsWire):
    """Per-ticker filings / earnings analysis (wire)."""

    summary: list[str] = Field(description="up to 3 English bullets")
    earnings_highlights: list[str] = Field(
        description="key earnings figures in English, at most 5; figures exactly as written in the source or the "
        "same value with an English scale word"
    )
    tone: float = Field(description="-1.0 to +1.0, overall direction of the filings and results")
    risk_factors: list[RiskFactorWireEN]


class CriticWireEN(CriticWire):
    """Adversarial review of the preliminary signal (wire)."""

    thesis: str = Field(description="1-2 sentence investment thesis in English")
    bull_points: list[str] = Field(description="at most 3, one line of English each")
    bear_points: list[str] = Field(description="at most 3, one line of English each")
    contradictions: list[str] = Field(description="conflicts between quant signals and text evidence, in English")
    confidence_adjustment: int = Field(description="integer -25 to +5, 0 when the evidence is thin")
    critical_reason: str | None = Field(description="one English sentence when critical_flag is true")
    critical_evidence: str | None = Field(
        description="when critical_flag is true, a verbatim quote from the document text in its original language"
    )
    critical_source_ref: str | None = Field(description="id of the document critical_evidence comes from")


WIRE_BY_TASK: dict[str, dict[str, type[BaseModel]]] = {
    "news": {"ko": NewsWire, "en": NewsWireEN},
    "filings": {"ko": FilingsWire, "en": FilingsWireEN},
    "critic": {"ko": CriticWire, "en": CriticWireEN},
}


def wire_model(task: str, lang: Any = "ko") -> type[BaseModel]:
    """The wire model for ``task`` ("news" / "filings" / "critic") in the output language (Korean by default)."""
    return WIRE_BY_TASK[task][lang_of(lang)]


def critical_category_label(category: str, lang: Any = "ko") -> str:
    """Display label of a critical category (``CRITICAL_CATEGORY_LABELS`` or its English twin)."""
    labels = CRITICAL_CATEGORY_LABELS_EN if lang_of(lang) == "en" else CRITICAL_CATEGORY_LABELS
    return labels.get(category, category)


# ================================================================================== clamping


def clamp(x: float, lo: float, hi: float, *, default: float = 0.0) -> tuple[float, bool]:
    """Clamp ``x`` to [lo, hi]. A non-finite value becomes ``default``. Returns (value, changed)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default, True
    if not math.isfinite(v):
        return default, True
    if v < lo:
        return lo, True
    if v > hi:
        return hi, True
    return v, False


def clean_lines(items: Iterable[str], limit: int | None = None) -> list[str]:
    """Strip items, drop empty ones and exact duplicates, and keep the first ``limit`` of them."""
    out: list[str] = []
    for s in items:
        t = " ".join(str(s).split())
        if t and t not in out:
            out.append(t)
    return out if limit is None else out[:limit]


# ============================================================================ quote grounding

_QUOTE_CHARS = frozenset("\"'`´“”‘’«»「」『』〈〉《》")
_EDGE_PUNCT = ".·-–—,:;…"
MAX_ELLIPSIS_GAP = 300  # normalised chars allowed between the parts of an elided quote
# Negation right before (English) or right after (Korean) a quote flips its meaning. Normalised text.
_NEG_BEFORE = ("no", "not", "never", "without", "none", "nor")
_NEG_AFTER = ("않", "없", "아니", "아닙", "아님", "못")
_NEG_BEFORE_CHARS = 12
_NEG_AFTER_CHARS = 10


def normalize_text(s: str) -> str:
    """NFKC and case-fold, map '…' to '...', and remove whitespace, quote and invisible format (Cf) characters."""
    s = unicodedata.normalize("NFKC", s or "").casefold().replace("…", "...")
    return "".join(ch for ch in s if not ch.isspace() and ch not in _QUOTE_CHARS and unicodedata.category(ch) != "Cf")


def _ngrams(s: str, n: int = 3) -> set[str]:
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _local_ngram_overlap(q: str, src: str, n: int = 3) -> float:
    """Best share of ``q``'s character n-grams found inside ONE window of ``src`` about ``len(q)`` long.

    The window is ``len(q) + max(8, len(q)//4)`` characters, which tolerates a few inserted or
    dropped words but not n-grams scattered across a long document.
    """
    qg = _ngrams(q, n)
    if not qg or len(src) < n:
        return 0.0
    grams = [src[i : i + n] for i in range(len(src) - n + 1)]
    k = max(1, len(q) + max(8, len(q) // 4) - n + 1)  # n-grams per window
    counts: dict[str, int] = {}
    present = best = 0
    for i, g in enumerate(grams):
        if g in qg:
            c = counts.get(g, 0)
            present += c == 0
            counts[g] = c + 1
        if i >= k:
            old = grams[i - k]
            if old in qg:
                counts[old] -= 1
                present -= counts[old] == 0
        best = max(best, present)
    return best / len(qg)


def _ellipsis_grounded(parts: Sequence[str], src: str) -> bool:
    """Parts of an elided quote occur in order, each gap <= MAX_ELLIPSIS_GAP."""
    start = src.find(parts[0])
    while start >= 0:
        pos, ok = start + len(parts[0]), True
        for p in parts[1:]:
            i = src.find(p, pos)
            if i < 0 or i - pos > MAX_ELLIPSIS_GAP:
                ok = False
                break
            pos = i + len(p)
        if ok:
            return True
        start = src.find(parts[0], start + 1)
    return False


def _affirmed_occurrence(q: str, src: str) -> bool:
    """True when at least one verbatim occurrence of ``q`` in ``src`` is not negated by its context."""
    start = src.find(q)
    while start >= 0:
        before = src[max(0, start - _NEG_BEFORE_CHARS) : start]
        after = src[start + len(q) : start + len(q) + _NEG_AFTER_CHARS]
        if not before.endswith(_NEG_BEFORE) and not any(t in after for t in _NEG_AFTER):
            return True
        start = src.find(q, start + 1)
    return False


def is_grounded(
    quote: str,
    source: str,
    *,
    min_chars: int = 4,
    ngram_threshold: float = 0.8,
    strict: bool = False,
) -> bool:
    """True when ``quote`` is (almost) verbatim in ``source`` (see the module docstring).

    ``strict``: only a contiguous verbatim match with at least one non-negated occurrence counts.
    """
    q = normalize_text(quote).strip(_EDGE_PUNCT)
    if len(q) < min_chars:
        return False
    src = normalize_text(source)
    if not src:
        return False
    if strict:
        return _affirmed_occurrence(q, src)
    if q in src:
        return True
    parts = [p for p in re.split(r"\.{3,}", q) if p]
    if len(parts) > 1:
        return all(len(p) >= 2 for p in parts) and sum(map(len, parts)) >= min_chars and _ellipsis_grounded(parts, src)
    qg = _ngrams(q)
    if not qg or len(qg & _ngrams(src)) / len(qg) < ngram_threshold:  # global share bounds the local one
        return False
    return _local_ngram_overlap(q, src) >= ngram_threshold


def locate_evidence(
    evidence: str,
    source_ref: str,
    docs: Mapping[str, str],
    titles: Mapping[str, str] | None = None,
    *,
    strict: bool = False,
) -> str | None:
    """Return the id of the document that grounds ``evidence``, or None.

    The document named by ``source_ref`` (an exact id, or a case-insensitive title match) is tried first.
    If the quote is found only in another document, that document's id is returned, so a mis-cited but
    genuine quote is re-pointed to the right document instead of being lost.
    """
    ref = (source_ref or "").strip()
    first: str | None = None
    if ref in docs:
        first = ref
    elif ref and titles:
        low = ref.casefold()
        first = next((i for i, t in titles.items() if t.strip().casefold() == low), None)
    if first is None and ref:
        first = next((i for i in docs if i in ref), None)
    order = ([first] if first else []) + [i for i in docs if i != first]
    return next((i for i in order if is_grounded(evidence, docs[i], strict=strict)), None)


# =========================================================================== number grounding

_KR_UNITS: tuple[tuple[str, Decimal], ...] = (  # longest first
    ("천억", Decimal("1e11")),
    ("백억", Decimal("1e10")),
    ("십억", Decimal("1e9")),
    ("천만", Decimal("1e7")),
    ("백만", Decimal("1e6")),
    ("조", Decimal("1e12")),
    ("억", Decimal("1e8")),
    ("만", Decimal("1e4")),
)
_EN_UNITS: dict[str, Decimal] = {
    "trillion": Decimal("1e12"),
    "tn": Decimal("1e12"),
    "billion": Decimal("1e9"),
    "bn": Decimal("1e9"),
    "million": Decimal("1e6"),
    "mn": Decimal("1e6"),
    "mm": Decimal("1e6"),
    "thousand": Decimal("1e3"),
    "b": Decimal("1e9"),
    "m": Decimal("1e6"),
    "k": Decimal("1e3"),
    "t": Decimal("1e12"),  # "KRW 8.2T" (upper-case T only, see _EN_LETTER_UNIT_RE; lower-case t is tonnes)
}
_NUM_RE = re.compile(r"(?<![A-Za-z0-9.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")
_KR_UNIT_RE = re.compile(r"\s*(" + "|".join(u for u, _ in _KR_UNITS) + ")")
_EN_WORD_UNIT_RE = re.compile(r"\s*(trillion|billion|million|thousand|tn|bn|mn|mm)(?![A-Za-z])", re.IGNORECASE)
_EN_LETTER_UNIT_RE = re.compile(r"([bBmMkKT])(?![A-Za-z])")
# period / ordinal / date suffixes: these numbers are labels, not reported figures.
# "주" is a week label ("52주", "3주 연속") only for a short plain integer; "3,000,000주" is a share count.
_PERIOD_SUFFIX_RE = re.compile(r"\s*(분기|반기|개월|월|일|년|주|차|위(?!안)|호|기|시|분|[QqHh](?![A-Za-z]))")
# the English counterparts of those labels (English LLM output renders "3분기" as "3rd quarter", "52주" as "52-week")
_EN_LABEL_SUFFIX_RE = re.compile(
    r"(?:st|nd|rd|th)(?![A-Za-z])"
    r"|[-\s]?(?:quarters?|qtrs?|months?|mos?|weeks?|wks?|years?|yrs?|days?|hours?|hrs?|minutes?|mins?)(?![A-Za-z])",
    re.IGNORECASE,
)
# A currency code glued to its amount ("KRW10.4tn", "W642bn", "USD1.2bn"): split it off, or the letter before the
# digits would hide the figure from _NUM_RE (whose look-behind skips labels such as "Q3" / "FY2026").
_CURRENCY_PREFIX_RE = re.compile(r"(?<![A-Za-z])(KRW|USD|EUR|JPY|CNY|RMB|HKD|GBP|W)(?=\d)")
_PERCENT_RE = re.compile(r"\s*(?:%|퍼센트|percent(?![A-Za-z])|pct(?![A-Za-z]))", re.IGNORECASE)
SCALED_REL_TOL = Decimal("0.05")  # a rounded scaled figure may differ from the source by at most 5%
_DATE_RE = re.compile(
    r"(?:19|20)\d{2}\s*[-./]\s*(?:0?[1-9]|1[0-2])\s*[-./]\s*(?:0?[1-9]|[12]\d|3[01])"
    r"|(?:19|20)\d{2}-(?:0?[1-9]|1[0-2])(?!\d)"
    r"|(?:19|20)\d{2}\s*년\s*\d{1,2}\s*월(?:\s*\d{1,2}\s*일)?"
    r"|\d{1,2}\s*월\s*\d{1,2}\s*일"
)


_NEXT_NUM_RE = re.compile(r"\s*" + _NUM_RE.pattern)


@dataclass(frozen=True)
class NumToken:
    """A figure found in text.

    ``mantissa`` is the first written number, ``value`` the scaled absolute value (None when no scale
    unit follows), and ``tolerance`` half a unit of the last written digit at its scale. For a Korean
    compound ("10조 4,000억"), ``parts`` holds every written number and ``lead_value`` the leading term (1e13).
    """

    raw: str
    mantissa: Decimal
    value: Decimal | None
    tolerance: Decimal
    compound: bool = False
    parts: tuple[Decimal, ...] = ()
    lead_value: Decimal | None = None
    percent: bool = False  # unscaled figure directly followed by % / 퍼센트 / percent


def _dec(s: str) -> Decimal | None:
    try:
        return Decimal(s.replace(",", ""))
    except InvalidOperation:
        return None


def _half_unit(num: str) -> Decimal:
    decimals = len(num.split(".", 1)[1]) if "." in num else 0
    return Decimal(5) * Decimal(10) ** (-(decimals + 1))


def _kr_unit_at(text: str, pos: int) -> tuple[Decimal | None, int]:
    m = _KR_UNIT_RE.match(text, pos)
    if not m:
        return None, pos
    return dict(_KR_UNITS)[m.group(1)], m.end()


def _en_unit_at(text: str, pos: int) -> tuple[Decimal | None, int]:
    m = _EN_WORD_UNIT_RE.match(text, pos) or _EN_LETTER_UNIT_RE.match(text, pos)
    if not m:
        return None, pos
    return _EN_UNITS[m.group(1).lower()], m.end()


def extract_numbers(text: str, *, ignore: Iterable[str] = ()) -> list[NumToken]:
    """Extract figures from ``text``, ignoring dates, period/ordinal labels, bare years and ``ignore`` tokens.

    Korean compounds such as "10조 4,000억" become ONE token whose value is 1.04e13.
    """
    ignore_set = {i.replace(",", "") for i in ignore if i}
    text = _DATE_RE.sub(" ", unicodedata.normalize("NFKC", text or ""))
    text = _CURRENCY_PREFIX_RE.sub(r"\1 ", text)
    out: list[NumToken] = []
    pos = 0
    while True:
        m = _NUM_RE.search(text, pos)
        if not m:
            return out
        num, end = m.group(1), m.end()
        pos = end
        mant = _dec(num)
        if mant is None or num.replace(",", "") in ignore_set:
            continue
        suffix = _PERIOD_SUFFIX_RE.match(text, end)
        if suffix and not (suffix.group(1) == "주" and ("," in num or "." in num or len(num) > 3)):
            continue
        if _EN_LABEL_SUFFIX_RE.match(text, end):
            continue
        mult, after = _kr_unit_at(text, end)
        if mult is not None:
            lead = value = mant * mult
            tol, raw_end, parts = _half_unit(num) * mult, after, [mant]
            while True:  # chained smaller Korean units: "10조 4,000억(원)"
                nxt = _NEXT_NUM_RE.match(text, raw_end)
                if not nxt:
                    break
                sub_mult, sub_after = _kr_unit_at(text, nxt.end())
                sub = _dec(nxt.group(1))
                if sub_mult is None or sub is None or sub_mult >= mult:
                    break
                value += sub * sub_mult
                parts.append(sub)
                tol, mult, raw_end = (
                    _half_unit(nxt.group(1)) * sub_mult,
                    sub_mult,
                    sub_after,
                )
            out.append(
                NumToken(
                    text[m.start() : raw_end].strip(),
                    mant,
                    value,
                    tol,
                    compound=len(parts) > 1,
                    parts=tuple(parts),
                    lead_value=lead,
                )
            )
            pos = raw_end
            continue
        mult, after = _en_unit_at(text, end)
        if mult is not None:
            out.append(
                NumToken(
                    text[m.start() : after].strip(),
                    mant,
                    mant * mult,
                    _half_unit(num) * mult,
                )
            )
            pos = after
            continue
        if "," not in num and "." not in num and 1900 <= mant <= 2100 and len(num) == 4:
            continue  # bare year
        out.append(
            NumToken(
                num,
                mant,
                None,
                _half_unit(num),
                percent=bool(_PERCENT_RE.match(text, end)),
            )
        )


def _significant_digits(d: Decimal) -> int:
    digits = d.normalize().as_tuple().digits if d != 0 else (0,)
    return len(digits)


class NumberIndex:
    """Every figure in a corpus, for membership (exact) and rounding-tolerant (range) lookups.

    ``bare``: every written number (incl. percentages and the parts of compound amounts).
    ``plain``: standalone unscaled, non-percent numbers only, e.g. table cells whose unit sits in a
    header. Only these may ground a scaled figure by its digits alone.
    """

    def __init__(self, texts: Iterable[str]):
        bare: set[Decimal] = set()
        plain: set[Decimal] = set()
        values: set[Decimal] = set()
        for t in texts:
            for tok in extract_numbers(t):
                for p in tok.parts or (tok.mantissa,):
                    bare.add(p)
                    values.add(p)
                if tok.value is not None:
                    # "10조" for "10조 4,000억" is grounded by rounding (within 5%); the leading term is NOT
                    # indexed on its own, or "1조" would be grounded by "1조 6,000억" (-37.5%).
                    values.add(tok.value)
                elif not tok.percent:
                    plain.add(tok.mantissa)
        self.bare = frozenset(bare)
        self.plain = frozenset(plain)
        self._sorted = sorted(values)

    def near(self, value: Decimal, tol: Decimal) -> bool:
        i = bisect.bisect_left(self._sorted, value - tol)
        return i < len(self._sorted) and self._sorted[i] <= value + tol

    def __len__(self) -> int:
        return len(self._sorted)


def number_grounded(tok: NumToken, index: NumberIndex, *, lenient: bool = False) -> bool:
    """Is this figure present in the indexed corpus?

    - Unscaled figure: exact value match. With ``lenient`` it also matches when rounded at its own
      precision (e.g. "57%" for 57.3).
    - Scaled figure (조/억/billion...): the absolute value matches at its stated precision, so
      "10.4조" matches "10조 4,000억" and "$94.9 billion" matches "$94,930 million". The rounding gap
      is also capped at ``SCALED_REL_TOL`` of the value, so "1조" does not match "6,000억". A mantissa
      with at least 3 significant digits also matches a verbatim *plain* number (unscaled, not a
      percentage), which covers tables whose unit sits in a header ("(in millions)"). It never matches
      "12.3%" as "12.3조".
    """
    if tok.value is None:
        return tok.mantissa in index.bare or (lenient and index.near(tok.mantissa, tok.tolerance))
    if index.near(tok.value, min(tok.tolerance, abs(tok.value) * SCALED_REL_TOL)):
        return True
    return not tok.compound and _significant_digits(tok.mantissa) >= 3 and tok.mantissa in index.plain


def ungrounded_numbers(
    text: str, index: NumberIndex, *, lenient: bool = False, ignore: Iterable[str] = ()
) -> list[str]:
    """Figures in ``text`` that the corpus does not support (an empty list means fully grounded)."""
    return [t.raw for t in extract_numbers(text, ignore=ignore) if not number_grounded(t, index, lenient=lenient)]


# ============================================================================ wire -> domain


def to_article_sentiment(w: ArticleWire, *, grounded: bool) -> tuple[ArticleSentiment, bool]:
    """Map one article. Returns (domain object, whether a value had to be clamped)."""
    s, c1 = clamp(w.sentiment, -1.0, 1.0)
    r, c2 = clamp(w.relevance, 0.0, 1.0)
    art = ArticleSentiment(
        news_id=w.news_id.strip(),
        sentiment=s,
        relevance=r,
        event_type=w.event_type,
        evidence=" ".join(w.evidence.split())[:300],
        grounded=grounded,
    )
    return art, (c1 or c2)


def to_news_sentiment(
    *,
    score: float,
    n_articles: int,
    n_used: int,
    trend_7d: float | None,
    key_positives: Sequence[str],
    key_negatives: Sequence[str],
    key_events: Sequence[str],
    articles: Sequence[ArticleSentiment],
) -> NewsSentiment:
    sc, _ = clamp(score, -1.0, 1.0)
    tr = None if trend_7d is None else clamp(trend_7d, -2.0, 2.0)[0]
    return NewsSentiment(
        score=sc,
        score_100=max(-100, min(100, int(round(sc * 100)))),
        n_articles=n_articles,
        n_used=n_used,
        trend_7d=tr,
        key_positives=list(key_positives),
        key_negatives=list(key_negatives),
        key_events=list(key_events),
        articles=list(articles),
    )


def to_risk_factor(w: RiskFactorWire, *, source_ref: str) -> RiskFactor:
    return RiskFactor(
        title=" ".join(w.title.split()),
        detail=" ".join(w.detail.split()),
        severity=w.severity,
        category=w.category,
        source_ref=source_ref,
    )


def to_filing_intelligence(
    w: FilingsWire,
    *,
    summary: Sequence[str],
    earnings_highlights: Sequence[str],
    risk_factors: Sequence[RiskFactor],
    sources: Sequence[str],
) -> tuple[FilingIntelligence, bool]:
    """Map the filings analysis. Returns (domain object, whether tone had to be clamped)."""
    tone, clamped = clamp(w.tone, -1.0, 1.0)
    fi = FilingIntelligence(
        summary=list(summary)[:3],
        earnings_highlights=list(earnings_highlights),
        guidance=w.guidance,
        tone=tone,
        risk_factors=list(risk_factors),
        sources=list(sources),
    )
    return fi, clamped


def to_critic_review(
    *,
    thesis: str,
    bull_points: Sequence[str],
    bear_points: Sequence[str],
    contradictions: Sequence[str],
    confidence_adjustment: float,
    critical_flag: bool,
    critical_reason: str | None,
) -> tuple[CriticReview, bool]:
    """Build the domain review. Returns (review, whether the adjustment had to be clamped into [-25, 5])."""
    adj, clamped = clamp(confidence_adjustment, -25, 5)
    return CriticReview(
        thesis=thesis,
        bull_points=list(bull_points),
        bear_points=list(bear_points),
        contradictions=list(contradictions),
        confidence_adjustment=int(round(adj)),
        critical_flag=critical_flag,
        critical_reason=critical_reason if critical_flag else None,
    ), clamped


WIRE_MODELS: tuple[type[BaseModel], ...] = (NewsWire, FilingsWire, CriticWire)
WIRE_MODELS_EN: tuple[type[BaseModel], ...] = (NewsWireEN, FilingsWireEN, CriticWireEN)
