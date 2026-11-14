"""Filings intelligence: number grounding, head+tail truncation, risk-factor refs, and guarded web research."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, date, datetime, timedelta

import anthropic
import httpx2
import pytest
from conftest import make_filings, make_instrument

from quantcio.llm.client import FakeLLMClient, LLMClient, LLMRecorder
from quantcio.llm.filings import (
    WEB_RESEARCH_FORM,
    WEB_RESEARCH_FORM_EN,
    analyze_filings,
    build_filings_user,
    order_filings,
    prepare_filings,
    truncate_head_tail,
    web_research,
)
from quantcio.llm.prompts import DOCUMENT_NOTICE_EN, FILINGS_SYSTEM, FILINGS_SYSTEM_EN, WEB_RESEARCH_SYSTEM_EN
from quantcio.llm.schemas import FilingsWire, FilingsWireEN, NumberIndex, extract_numbers, ungrounded_numbers
from quantcio.models import FilingItem, Fundamentals, Market, text_id

NOW = datetime(2026, 9, 24, tzinfo=UTC)
RISK_TEXT = (
    "분기보고서 II. 사업의 내용\n회사는 주요 고객사의 발주 감소로 인해 재고자산 평가손실이 발생할 수 있습니다.\n"
    "당사는 환율 변동에 따른 위험에 노출되어 있습니다."
)


def _10q() -> FilingItem:
    return FilingItem(
        id=text_id("분기보고서"),
        title="분기보고서 (2026.06)",
        form_type="분기보고서",
        filed_at=NOW - timedelta(days=2),
        text=RISK_TEXT,
        is_earnings=False,
        source="DART",
    )


# ---------------------------------------------------------------------------- number grounding

KR_TEXT = make_filings(NOW)[0].text
US_TEXT = "Total net sales were $94,930 million, up 6% year over year. Diluted EPS of $1.64. Gross margin 46.5%."


@pytest.mark.parametrize(
    "line,bad",
    [
        ("3분기 매출액 79조 1,000억원(전년동기 대비 +12.3%)", []),  # quarter label ignored, sign ignored
        ("영업이익 10.4조원", []),  # unit-normalised equal value
        ("영업이익 10조원", []),  # leading term of "10조 4,000억"
        ("순이익 8.2조", []),
        ("영업이익 145.2% 증가", []),
        ("매출 80조원", ["80조"]),
        ("영업이익 150% 증가", ["150"]),
        ("순이익 9조원", ["9조"]),
        ("2026년 9월 24일 공시, 삼성전자(005930) 매출 79조 1,000억원", []),  # dates and ticker ignored
    ],
)
def test_kr_number_grounding(line, bad):
    assert ungrounded_numbers(line, NumberIndex([KR_TEXT]), ignore=["005930"]) == bad


@pytest.mark.parametrize(
    "line,bad",
    [
        ("FY2026 Q3 매출 $94,930 million", []),
        ("매출 $94.9 billion (+6% YoY)", []),  # rounded at the stated precision
        ("매출 949억 3,000만 달러", []),  # Korean units, same absolute value
        ("매출 $95.2 billion", ["95.2 billion"]),
        ("EPS $1.64, 총이익률 46.5%", []),
        ("EPS $1.70", ["1.70"]),
    ],
)
def test_us_number_grounding(line, bad):
    assert ungrounded_numbers(line, NumberIndex([US_TEXT])) == bad


KR_TEXT_2 = KR_TEXT + "당기순이익(지배) 1조 6,000억원. 자사주 매입 6,000억원. 보통주 1,200,000주 소각\n"


@pytest.mark.parametrize(
    "line,bad",
    [
        ("매출 12.3조원", ["12.3조"]),  # a growth RATE (12.3%) must not ground an amount
        ("영업이익 145.2조", ["145.2조"]),
        ("매출 1,000억원", ["1,000억"]),  # a compound's minor part is not an amount on its own
        ("순이익 2조원", ["2조"]),  # 1조 6,000억 -> "2조" is 25% off
        ("자사주 1조원 매입", ["1조"]),  # 6,000억 / 1조 6,000억 -> "1조" is not a rounding
        ("보통주 3,000,000주 신규 발행", ["3,000,000"]),  # share counts are figures, not week labels
        ("보통주 1,200,000주 소각", []),
        ("52주 신고가, 3주 연속 상승", []),  # week labels stay ignored
        ("순이익 1.6조, 영업이익 10조원, 매출 79조", []),  # roundings within 5% of the stated value
    ],
)
def test_number_grounding_regressions(line, bad):
    assert ungrounded_numbers(line, NumberIndex([KR_TEXT_2])) == bad


@pytest.mark.parametrize(
    "line,source,bad",
    [
        ("매출 $35.1 billion", "Revenue of $35,082 million", []),
        ("매출 $35.1 billion", "Revenue grew 35.1% to $41.2 billion", ["35.1 billion"]),
        ("매출 $35,082 million", "Revenue  35,082  (in millions)", []),
        ("매출 $36 billion", "Revenue of $35,082 million", ["36 billion"]),
    ],
)
def test_us_scaled_number_regressions(line, source, bad):
    assert ungrounded_numbers(line, NumberIndex([source])) == bad


def test_extract_numbers_compound_and_labels():
    toks = extract_numbers("매출액 79조 1,000억원, 3분기, 2026년, Q3, 1H26, 12.3%")
    assert [t.raw for t in toks] == ["79조 1,000억", "12.3"]
    assert toks[0].compound and toks[0].value == 79_100_000_000_000


def test_unscaled_table_value_does_not_ground_a_rescaled_figure():
    """Scale info that sits only in a table header can't be verified.

    So a rescaled figure is rejected unless the digits match.
    """
    idx = NumberIndex(["Net sales $ 94,930 (in millions)"])
    assert ungrounded_numbers("매출 $94.9 billion", idx) == ["94.9 billion"]
    assert ungrounded_numbers("매출 $94,930 million", idx) == []  # verbatim digits (>= 3 significant)


# --------------------------------------------------------------------------- truncation / order


def test_truncate_head_tail():
    text = "H" * 700 + "M" * 5000 + "T" * 300
    shown, grounding, truncated = truncate_head_tail(text, 1000)
    assert truncated and shown.startswith("H" * 700) and shown.endswith("T" * 300)
    assert "[중략: 5,000자 생략]" in shown and "M" not in shown
    assert "5,000" not in grounding and grounding.count("H") == 700 and grounding.count("T") == 300
    assert truncate_head_tail("short", 1000) == ("short", "short", False)


def test_order_filings_puts_earnings_first():
    earnings = make_filings(NOW)[0]  # filed 10 days ago
    newer = _10q()  # filed 2 days ago, not earnings
    assert [f.id for f in order_filings([newer, earnings, newer], 6)] == [earnings.id, newer.id]
    assert len(order_filings([newer, earnings], 1)) == 1


def test_user_message_contract(settings):
    s = settings.model_copy(deep=True)
    long = _10q().model_copy(update={"text": RISK_TEXT + "\n" + "가" * 30000})
    prepared = prepare_filings(order_filings([long, make_filings(NOW)[0]], 6), s.data.max_filing_chars)
    msg = build_filings_user(make_instrument(), prepared, Fundamentals(per=13.5, roe=0.105))
    assert msg.index('is_earnings="true"') < msg.index('is_earnings="false"')
    assert 'truncated="true"' in msg and "[중략" in msg
    # only filing-stable fundamentals: no date / price ratios, so unchanged filings hit the cache on later days
    assert "<fundamentals>" in msg and '"roe": 0.105' in msg and '"per"' not in msg
    assert "삼성전자 (티커 005930" in msg


# ------------------------------------------------------------------------------ analysis


def _filings_wire(earnings_id: str, q_id: str) -> dict:
    return {
        "summary": ["3분기 영업이익 10조 4,000억원으로 급증", "매출 100조원 돌파 전망", ""],
        "earnings_highlights": [
            "매출액 79조 1,000억원, 전년동기 대비 12.3% 증가",
            "영업이익 10.4조원(+145.2%)",
            "순이익 9조원",
        ],
        "guidance": "none",
        "tone": 1.4,
        "risk_factors": [
            {
                "title": "재고 평가손실",
                "detail": "주요 고객 발주 감소",
                "severity": "high",
                "category": "market_demand",
                "source_ref": q_id,
                "evidence": "주요 고객사의 발주 감소로 인해 재고자산 평가손실이 발생할 수 있습니다",
            },
            {
                "title": "근거 없음",
                "detail": "x",
                "severity": "critical",
                "category": "governance",
                "source_ref": " ",
                "evidence": "주요 고객사의 발주 감소",
            },
            {
                "title": "환율",
                "detail": "환율 변동",
                "severity": "low",
                "category": "macro_fx",
                "source_ref": earnings_id,
                "evidence": "환율 변동에 따른 위험에 노출되어 있습니다",
            },  # mis-cited -> re-pointed
            {
                "title": "날조",
                "detail": "x",
                "severity": "critical",
                "category": "governance",
                "source_ref": q_id,
                "evidence": "대표이사가 횡령 혐의로 구속되었습니다",
            },  # not in any document
        ],
    }


def test_analyze_filings_grounds_numbers_and_risk_refs(settings):
    earnings, q = make_filings(NOW)[0], _10q()
    fake = FakeLLMClient({"filings": _filings_wire(earnings.id, q.id)}, cost_per_call=0.05)
    rec = LLMRecorder()
    fi = asyncio.run(
        analyze_filings(fake, make_instrument(), [q, earnings], Fundamentals(per=13.5), settings, recorder=rec)
    )
    assert fi.earnings_highlights == ["매출액 79조 1,000억원, 전년동기 대비 12.3% 증가", "영업이익 10.4조원(+145.2%)"]
    assert fi.summary == ["3분기 영업이익 10조 4,000억원으로 급증"]
    assert fi.tone == 1.0 and fi.guidance == "none"
    assert [(r.title, r.severity) for r in fi.risk_factors] == [("재고 평가손실", "high"), ("환율", "low")]
    assert fi.risk_factors[0].source_ref == "분기보고서 (2026.06) (2026-09-22)"
    assert fi.risk_factors[1].source_ref.startswith("분기보고서")  # re-pointed to where the quote is
    assert fi.sources[0].startswith("연결재무제표기준영업(잠정)실적")  # earnings first
    notices = " | ".join(rec.notices)
    for needle in (
        "실적 하이라이트 1건 폐기",
        "요약 문장 1건 폐기",
        "source_ref가 없는 위험 요인 1건",
        "근거 인용을 원문에서 찾을 수 없는 위험 요인 1건",
        "tone 보정",
    ):
        assert needle in notices
    assert rec.errors == []  # the analysis is served: drops/clamps are notices
    assert rec.cost_usd == pytest.approx(0.05)
    call = fake.calls[0]
    assert call.system == FILINGS_SYSTEM and call.wire_model is FilingsWire
    assert call.user.index(earnings.id) < call.user.index(q.id)


def test_duplicate_risk_factors_are_merged_so_the_penalty_is_not_doubled(settings):
    q = _10q()
    dup = {
        "title": "재고 평가손실",
        "detail": "d",
        "severity": "high",
        "category": "market_demand",
        "source_ref": q.id,
        "evidence": "재고자산 평가손실이 발생할 수 있습니다",
    }
    wire = {
        "summary": [],
        "earnings_highlights": [],
        "guidance": "none",
        "tone": 0.0,
        "risk_factors": [{**dup, "severity": "medium"}, dup, {**dup, "title": " 재고  평가손실 "}],
    }
    rec = LLMRecorder()
    fi = asyncio.run(
        analyze_filings(FakeLLMClient({"filings": wire}), make_instrument(), [q], None, settings, recorder=rec)
    )
    assert [(r.title, r.severity) for r in fi.risk_factors] == [("재고 평가손실", "high")]  # the most severe is kept
    assert any("중복 위험 요인 2건" in n for n in rec.notices)


def test_analyze_filings_none_without_documents_and_on_failure(settings):
    fake = FakeLLMClient({"filings": "서버 오류(500)"})
    assert asyncio.run(analyze_filings(fake, make_instrument(), [], None, settings)) is None
    assert fake.calls == []
    rec = LLMRecorder()
    assert (
        asyncio.run(analyze_filings(fake, make_instrument(), make_filings(NOW), None, settings, recorder=rec)) is None
    )
    assert rec.errors == ["공시 분석 실패: 서버 오류(500)"] and rec.notices == []


def test_supplementary_documents_bypass_the_cap(settings):
    s = settings.model_copy(deep=True)
    s.data.max_filings = 1
    extra = _10q().model_copy(update={"id": "web1", "title": "웹 리서치", "form_type": WEB_RESEARCH_FORM})
    fake = FakeLLMClient(
        {"filings": {"summary": [], "earnings_highlights": [], "guidance": "unknown", "tone": 0.1, "risk_factors": []}}
    )
    fi = asyncio.run(
        analyze_filings(fake, make_instrument(), [_10q(), make_filings(NOW)[0]], None, s, supplementary=[extra])
    )
    assert len(fi.sources) == 2 and 'id="web1"' in fake.calls[0].user


# --------------------------------------------------------------------------- web research


def _web_settings(settings, enabled=True):
    s = settings.model_copy(deep=True)
    s.llm.web_research = enabled
    s.llm.response_cache = False
    return s


def test_web_research_is_guarded(settings):
    fake = FakeLLMClient({"web_research": ("x", [{"url": "u", "title": "t", "cited_text": ["c"]}])})
    assert asyncio.run(web_research(fake, make_instrument(Market.US), _web_settings(settings, False))) is None
    assert asyncio.run(web_research(fake, make_instrument(Market.KR), _web_settings(settings))) is None
    assert fake.calls == []


def test_web_research_uses_only_cited_snippets(settings):
    sources = [
        {
            "url": "https://www.fool.com/a",
            "title": "Apple Q3 2026 call",
            "page_age": None,
            "cited_text": ["Revenue was $94,930 million, up 6% year over year."],
        },
        {"url": "https://x.com/uncited", "title": "no cite", "page_age": None, "cited_text": []},
    ]
    fake = FakeLLMClient({"web_research": ("모델 요약: 매출 $999 billion", sources)}, cost_per_call=0.03)
    rec = LLMRecorder()
    wr = asyncio.run(
        web_research(fake, make_instrument(Market.US), _web_settings(settings), as_of=date(2026, 9, 24), recorder=rec)
    )
    assert wr is not None and wr.filing.form_type == WEB_RESEARCH_FORM
    assert "$94,930 million" in wr.filing.text and "999" not in wr.filing.text  # model prose is never a source
    assert wr.source_labels() == ["웹: Apple Q3 2026 call <https://www.fool.com/a>"]
    assert rec.cost_usd == pytest.approx(0.03)
    assert "Apple Inc." in fake.calls[0].user and "2026-09-24" in fake.calls[0].user


def test_web_research_without_citations_or_on_error_is_recorded(settings):
    rec = LLMRecorder()
    nocite = FakeLLMClient({"web_research": ("text", [{"url": "u", "title": "t", "cited_text": []}])})
    assert asyncio.run(web_research(nocite, make_instrument(Market.US), _web_settings(settings), recorder=rec)) is None
    failing = FakeLLMClient({"web_research": "요청 한도 초과(429)"})
    assert asyncio.run(web_research(failing, make_instrument(Market.US), _web_settings(settings), recorder=rec)) is None
    assert rec.notices == ["웹 리서치: 인용된 출처가 없어 결과를 사용하지 않음"]  # served without the supplement
    assert rec.errors == ["웹 리서치 실패: 요청 한도 초과(429)"]  # the call itself failed


def test_web_research_through_mock_transport(settings):
    """A real LLMClient over httpx2.MockTransport: server tool shape, citations, cost incl. $10/1k searches."""
    sent = []
    body = {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "Apple earnings call"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "s1",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://www.fool.com/a",
                        "title": "Apple (AAPL) Q3 2026 Call",
                        "encrypted_content": "E",
                        "page_age": "July 31, 2026",
                    }
                ],
            },
            {
                "type": "text",
                "text": "매출은 $94,930 million으로 증가했다.",
                "citations": [
                    {
                        "type": "web_search_result_location",
                        "url": "https://www.fool.com/a",
                        "title": "Apple (AAPL) Q3 2026 Call",
                        "encrypted_index": "I",
                        "cited_text": "Total revenue was $94,930 million.",
                    }
                ],
            },
        ],
        "usage": {
            "input_tokens": 4000,
            "output_tokens": 200,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 0},
        },
    }

    async def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=body, headers={"request-id": "req_w"})

    s = _web_settings(settings)
    ac = anthropic.AsyncAnthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    rec = LLMRecorder()
    wr = asyncio.run(
        web_research(
            LLMClient(s, anthropic_client=ac),
            make_instrument(Market.US),
            s,
            as_of=date(2026, 9, 24),
            recorder=rec,
            max_uses=4,
        )
    )
    tool = sent[0]["tools"][0]
    assert tool["type"] == "web_search_20260209" and tool["name"] == "web_search" and tool["max_uses"] == 4
    assert "fool.com" in tool["allowed_domains"]
    assert sent[0]["output_config"] == {"effort": "medium"} and "format" not in sent[0]["output_config"]
    assert sent[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert wr is not None and "Total revenue was $94,930 million." in wr.filing.text
    assert rec.cost_usd == pytest.approx((4000 * 5 + 200 * 25) / 1e6 + 2 * 0.01)


# ------------------------------------------------------------------------ English output (report.language)

HANGUL = re.compile("[가-힣]")
# a Korean 잠정실적 release plus a buyback / dividend line
KR_RELEASE = KR_TEXT + "자기주식 취득 결정: 취득예정금액 6,420억원\n현금배당 총액 312억원\n"


def _en(settings):
    s = settings.model_copy(deep=True)
    s.report.language = "en"
    return s


@pytest.mark.parametrize(
    "line,bad",
    [
        # copied as written, or the same value with an English scale word at the stated precision
        ("Revenue KRW 79.1 trillion, up 12.3% YoY", []),
        ("Operating profit KRW 10.4 trillion (+145.2% YoY)", []),
        ("Operating profit of 10조 4,000억원, up 145.2% year over year", []),
        ("Net income 8.2 trillion won", []),
        ("Share buyback of KRW 642 billion", []),
        ("Cash dividends totalling KRW 31.2 billion", []),
        ("Revenue ₩79.1tn; operating profit KRW10.4tn; net income W8.2T", []),  # symbols / glued codes / T
        ("Buyback KRW 642bn, dividends KRW 31.2 bn", []),
        ("Revenue KRW 79,100 billion", []),
        ("3rd-quarter operating profit KRW 10.4 trillion, a 52-week high", []),  # labels, not figures
        ("Q3 2026 revenue KRW 79.1 trillion", []),
        # fabricated or computed figures are rejected
        ("Revenue KRW 80 trillion", ["80 trillion"]),
        ("Operating profit up 150% YoY", ["150"]),
        ("Net income KRW 9 trillion", ["9 trillion"]),
        ("Buyback KRW 700 billion", ["700 billion"]),
        ("Revenue KRW99 trillion", ["99 trillion"]),  # a glued currency code does not hide a figure
        ("Buyback W700bn", ["700bn"]),
        ("Revenue KRW 79.1 trillion vs. KRW 70.4 trillion a year earlier", ["70.4 trillion"]),  # computed base
        ("Operating margin 13.1%", ["13.1"]),  # computed ratio
        ("Revenue USD 57.3 billion", ["57.3 billion"]),  # currency conversion
    ],
)
def test_english_renderings_of_korean_figures_are_grounded(line, bad):
    assert ungrounded_numbers(line, NumberIndex([KR_RELEASE]), ignore=["005930"]) == bad


def test_english_scale_words_keep_the_precision_rules():
    idx = NumberIndex([KR_RELEASE])
    assert ungrounded_numbers("Operating profit KRW 10 trillion", idx) == []  # 10조 4,000억 at 1-trillion precision
    assert ungrounded_numbers("Buyback KRW 0.64 trillion", idx) == []  # 6,420억 at 0.01-trillion precision
    assert ungrounded_numbers("Buyback KRW 0.6 trillion", idx) == ["0.6 trillion"]  # 6.5% off: beyond SCALED_REL_TOL
    assert ungrounded_numbers("Revenue KRW 12.3 trillion", idx) == ["12.3 trillion"]  # a rate never grounds an amount
    assert ungrounded_numbers("Revenue KRW 1,000 billion", idx) == ["1,000 billion"]  # a compound's minor part
    toks = extract_numbers("KRW10.4tn, W642bn, 3rd quarter, 12 months, 52-week, 2 years, 5 trading days")
    assert [(t.raw, t.value) for t in toks] == [("10.4tn", 10_400_000_000_000), ("642bn", 642_000_000_000), ("5", None)]


def test_korean_extraction_is_unchanged_by_the_english_rules():
    toks = extract_numbers(KR_RELEASE)
    assert [t.raw for t in toks] == [
        "79조 1,000억",
        "12.3",
        "10조 4,000억",
        "145.2",
        "8조 2,000억",
        "6,420억",
        "312억",
    ]


def test_truncate_head_tail_in_english():
    text = "H" * 700 + "M" * 5000 + "T" * 300
    shown, grounding, truncated = truncate_head_tail(text, 1000, lang="en")
    assert truncated and "\n…[omitted: 5,000 chars]…\n" in shown and "중략" not in shown
    assert "5,000" not in grounding and "omitted" not in grounding  # the marker never grounds a number
    assert grounding == truncate_head_tail(text, 1000)[1]  # same grounding text in both languages
    assert truncate_head_tail("short", 1000, lang="en") == ("short", "short", False)


def test_user_message_contract_in_english(settings):
    s = _en(settings)
    long = _10q().model_copy(update={"text": RISK_TEXT + "\n" + "가" * 30000})
    prepared = prepare_filings(order_filings([long, make_filings(NOW)[0]], 6), s.data.max_filing_chars, lang="en")
    msg = build_filings_user(make_instrument(), prepared, Fundamentals(per=13.5), lang="en")
    assert msg.split("\n\n<fundamentals>", 1)[0].splitlines() == [
        "Company: 삼성전자 (ticker 005930, KOSPI, market KR, sector 반도체)",
        "Currency: KRW",
        "Documents: 2 (earnings releases first)",
        DOCUMENT_NOTICE_EN,
    ]
    assert "[omitted: " in msg and "[중략" not in msg and 'truncated="true"' in msg
    assert all("omitted" not in p.grounding and "중략" not in p.grounding for p in prepared)


def _filings_wire_en(earnings_id: str, q_id: str) -> dict:
    wire = _filings_wire(earnings_id, q_id)
    wire["summary"] = ["Q3 operating profit surged to KRW 10.4 trillion", "Revenue expected to top KRW 100 trillion"]
    wire["earnings_highlights"] = [
        "Revenue KRW 79.1 trillion, up 12.3% YoY",
        "Operating profit KRW 10.4 trillion (+145.2% YoY)",
        "Net income KRW 9 trillion",  # the release says 8조 2,000억원
    ]
    titles = ["Inventory valuation loss", "No reference", "FX exposure", "Fabricated"]
    for rf, title in zip(wire["risk_factors"], titles, strict=True):
        rf["title"], rf["detail"] = title, "English detail."  # evidence stays the verbatim Korean quote
    return wire


def test_analyze_filings_in_english(settings):
    s = _en(settings)
    earnings, q = make_filings(NOW)[0], _10q()
    fake = FakeLLMClient({"filings": _filings_wire_en(earnings.id, q.id)}, settings=s)
    rec = LLMRecorder(lang="en")
    fi = asyncio.run(analyze_filings(fake, make_instrument(), [q, earnings], Fundamentals(per=13.5), s, recorder=rec))

    call = fake.calls[0]
    assert call.system is FILINGS_SYSTEM_EN and call.wire_model is FilingsWireEN
    assert "Currency: KRW" in call.user and "Documents: 2 (earnings releases first)" in call.user
    assert "통화:" not in call.user and "자료 수" not in call.user
    assert fi.earnings_highlights == [
        "Revenue KRW 79.1 trillion, up 12.3% YoY",
        "Operating profit KRW 10.4 trillion (+145.2% YoY)",
    ]
    assert fi.summary == ["Q3 operating profit surged to KRW 10.4 trillion"]
    assert [(r.title, r.severity) for r in fi.risk_factors] == [
        ("Inventory valuation loss", "high"),
        ("FX exposure", "low"),
    ]
    assert rec.notices == [
        "Filings analysis: dropped 1 earnings highlight with figures not found in the source (e.g. 9 trillion)",
        "Filings analysis: dropped 1 summary line with figures not found in the source (e.g. 100 trillion)",
        "Filings analysis: dropped 1 risk factor without a source_ref",
        "Filings analysis: dropped 1 risk factor whose evidence quote is not in the source",
        "Filings analysis: out-of-range tone clamped",
    ]
    assert rec.errors == [] and not any(HANGUL.search(n) for n in rec.notices)


def test_analyze_filings_in_english_merges_and_fails_in_english(settings):
    s = _en(settings)
    q = _10q()
    dup = {
        "title": "Inventory valuation loss",
        "detail": "d",
        "severity": "high",
        "category": "market_demand",
        "source_ref": q.id,
        "evidence": "재고자산 평가손실이 발생할 수 있습니다",
    }
    wire = {"summary": [], "earnings_highlights": [], "guidance": "none", "tone": 0.0, "risk_factors": [dup, dup]}
    rec = LLMRecorder(lang="en")
    asyncio.run(analyze_filings(FakeLLMClient({"filings": wire}), make_instrument(), [q], None, s, recorder=rec))
    assert rec.notices == ["Filings analysis: merged 1 duplicate risk factor"]
    rec = LLMRecorder(lang="en")
    fake = FakeLLMClient({"filings": "server error (500)"}, settings=s)
    assert asyncio.run(analyze_filings(fake, make_instrument(), make_filings(NOW), None, s, recorder=rec)) is None
    assert rec.errors == ["Filings analysis failed: server error (500)"]


def test_web_research_in_english(settings):
    s = _en(_web_settings(settings))
    sources = [
        {
            "url": "https://www.fool.com/a",
            "title": "Apple Q3 2026 call",
            "page_age": None,
            "cited_text": ["Revenue was $94,930 million, up 6% year over year."],
        },
        {"url": "https://x.com/b", "title": None, "page_age": None, "cited_text": ["Services hit a record."]},
    ]
    fake = FakeLLMClient({"web_research": ("summary", sources)}, settings=s)
    rec = LLMRecorder(lang="en")
    wr = asyncio.run(web_research(fake, make_instrument(Market.US), s, as_of=date(2026, 9, 24), recorder=rec))
    call = fake.calls[0]
    assert call.system is WEB_RESEARCH_SYSTEM_EN
    assert call.user.startswith(
        "Company: Apple Inc. (ticker AAPL, NASDAQ, market US, sector Technology)\nReference date: 2026-09-24"
    )
    assert not HANGUL.search(call.user)
    assert wr.filing.form_type == WEB_RESEARCH_FORM_EN
    assert wr.filing.title == "Web research on the latest earnings call (2 cited sources)"
    assert "[2] untitled" in wr.filing.text and not HANGUL.search(wr.filing.text)
    assert wr.source_labels() == [
        "Web: Apple Q3 2026 call <https://www.fool.com/a>",
        "Web: https://x.com/b <https://x.com/b>",
    ]

    rec = LLMRecorder(lang="en")
    nocite = FakeLLMClient({"web_research": ("text", [{"url": "u", "title": "t", "cited_text": []}])}, settings=s)
    assert asyncio.run(web_research(nocite, make_instrument(Market.US), s, recorder=rec)) is None
    failing = FakeLLMClient({"web_research": "rate limit exceeded (429)"}, settings=s)
    assert asyncio.run(web_research(failing, make_instrument(Market.US), s, recorder=rec)) is None
    nosearch = type("NoSearch", (), {})()
    assert asyncio.run(web_research(nosearch, make_instrument(Market.US), s, recorder=rec)) is None
    assert rec.notices == [
        "Web research: no cited sources — result not used",
        "Web research: skipped — the client does not support web_search",
    ]
    assert rec.errors == ["Web research failed: rate limit exceeded (429)"]


# ------------------------------------------------------------------------------ token budgets (2026-09-27)


def test_approx_tokens_counts_hangul_heavier_than_english():
    from quantcio.llm.filings import approx_tokens, chars_for_tokens

    ko, en = "가" * 1000, "a" * 1000
    assert approx_tokens(ko) > 3 * approx_tokens(en)
    assert 800 <= chars_for_tokens(ko, 1000) <= 900 and chars_for_tokens(en, 1000) >= 3000


def test_salient_excerpt_keeps_head_and_paragraphs_with_figures():
    from quantcio.llm.filings import salient_excerpt

    filler = "\n".join(f"일반 현황 설명 문단 {chr(0xAC00 + i)}입니다." * 3 for i in range(60))
    key = "당기 영업이익은 3조 2,000억원으로 전년 대비 15% 증가했으며 매출은 40조원이다."
    text = "반기보고서 표지\n" + filler + "\n" + key + "\n" + filler
    shown, grounding, truncated = salient_excerpt(text, 900)
    assert truncated and shown.startswith("반기보고서 표지") and key in shown and key in grounding
    assert "[중략" in shown and "[중략" not in grounding and len(shown) < 1200
    assert salient_excerpt("짧은 글", 900) == ("짧은 글", "짧은 글", False)
    plain = "가" * 5000  # nothing salient: plain head / tail cut
    assert salient_excerpt(plain, 1000)[2] and len(salient_excerpt(plain, 1000)[1]) <= 1001


def test_prepare_filings_shares_one_token_budget():
    from quantcio.llm.filings import approx_tokens

    docs = [
        _10q().model_copy(update={"id": f"f{i}", "text": RISK_TEXT + "\n" + "매출 1조원 증가. " * 800})
        for i in range(5)
    ]
    one = prepare_filings(docs, 3000, total_tokens=None)
    assert len(one) == 5 and all(approx_tokens(p.shown) <= 3600 for p in one)
    shared = prepare_filings(docs, 3000, total_tokens=7000)
    assert len(shared) < 5 and sum(approx_tokens(p.shown) for p in shared) <= 7000 * 1.25


def test_order_filings_drops_same_day_duplicates_but_keeps_different_filings():
    a = _10q()
    copy = a.model_copy(update={"id": "copy", "text": a.text + " "})  # the same 10-Q filed twice
    other = a.model_copy(update={"id": "other", "text": "완전히 다른 내용의 공시입니다. " * 50})
    assert [f.id for f in order_filings([a, copy, other], 6)] == [a.id, "other"]


def test_filings_message_leaves_out_daily_changing_fundamentals():
    from quantcio.llm.filings import fundamentals_json

    f = Fundamentals(as_of=date(2026, 9, 26), per=13.5, pbr=1.2, market_cap=5e14, roe=0.105, eps=5000.0)
    stable = json.loads(fundamentals_json(f, stable_only=True))
    assert stable == {"eps": 5000.0, "roe": 0.105}
    assert "per" in json.loads(fundamentals_json(f))  # the full form is unchanged


def test_an_empty_analysis_is_re_asked_once_at_a_higher_effort(settings):
    from quantcio.llm.filings import EMPTY_RETRY_EFFORT

    filings = [f.model_copy(update={"text": f.text * 10}) for f in make_filings(NOW)]  # a substantial set
    efforts: list = []
    full = {
        "summary": ["3분기 실적 개선"],
        "earnings_highlights": [],
        "guidance": "none",
        "tone": 0.3,
        "risk_factors": [],
    }
    empty = {"summary": [], "earnings_highlights": [], "guidance": "none", "tone": 0.0, "risk_factors": []}

    class Client(FakeLLMClient):
        async def call_structured(self, task, system, user, wire_model, effort=None, max_tokens=None):
            efforts.append(effort)
            return await super().call_structured(task, system, user, wire_model, effort, max_tokens)

    answers = iter([empty, full])
    rec = LLMRecorder()
    fi = asyncio.run(
        analyze_filings(Client(lambda *a: next(answers)), make_instrument(), filings, None, settings, recorder=rec)
    )
    assert efforts == [None, EMPTY_RETRY_EFFORT] and fi.summary == ["3분기 실적 개선"]
    assert any("재요청" in n and "결과 확보" in n for n in rec.notices)
    # a normal answer is never re-asked
    efforts.clear()
    asyncio.run(analyze_filings(Client(lambda *a: full), make_instrument(), filings, None, settings))
    assert efforts == [None]
