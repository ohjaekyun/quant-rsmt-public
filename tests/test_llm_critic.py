"""Critic bounds and grounding, llm_pillar, and the run_llm_analysis orchestration (offline, FakeLLMClient)."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime, timedelta

import pytest
from conftest import make_filings, make_market_data, make_news

from quantcio.llm import llm_pillar, prompts, risk_penalty, run_llm_analysis
from quantcio.llm.client import FakeLLMClient, LLMRecorder
from quantcio.llm.critic import compact_json, critique, llm_context, prelim_context, quant_context
from quantcio.llm.prompts import CRITIC_SYSTEM, CRITIC_SYSTEM_EN, DOCUMENT_NOTICE_EN
from quantcio.llm.schemas import (
    CRITICAL_CATEGORY_LABELS,
    CRITICAL_CATEGORY_LABELS_EN,
    CriticWire,
    CriticWireEN,
    FilingsWireEN,
    NewsWireEN,
    critical_category_label,
)
from quantcio.models import (
    ExecutiveSignal,
    FactorScore,
    FilingIntelligence,
    FilingItem,
    LLMIntelligence,
    Market,
    NewsSentiment,
    QuantSignal,
    RiskFactor,
    Signal,
    Technicals,
    text_id,
)

NOW = datetime(2026, 9, 24, tzinfo=UTC)
AUDIT_TEXT = "감사인은 계속기업 존속능력에 대한 중요한 불확실성으로 인하여 감사의견을 거절하였습니다."


def make_quant() -> QuantSignal:
    tech = Technicals(
        last_close=60000.0,
        atr=1500.0,
        atr_pct=0.025,
        rsi14=58.3,
        ma50=58000.0,
        ma200=55000.0,
        high_52w=65000.0,
        low_52w=45000.0,
        realized_vol_60d=0.28,
        beta_1y=1.1,
        max_drawdown_1y=-0.18,
        ret_1m=0.042,
        ret_3m=0.11,
        ret_12m_ex1m=0.25,
        rel_strength_3m=0.05,
        support=57000.0,
        resistance=65000.0,
    )
    names = [
        ("momentum", "모멘텀", 71.0),
        ("value", "밸류에이션", 55.0),
        ("volatility", "변동성(안정성)", 48.0),
        ("quality", "퀄리티", 62.0),
        ("trend", "추세", 66.0),
    ]
    factors = [FactorScore(name=n, label=lab, score=sc, rationale="") for n, lab, sc in names]
    return QuantSignal(technicals=tech, factors=factors, factor_composite=61.2, factor_pillar=0.22, ml_pillar=0.1)


def make_prelim() -> ExecutiveSignal:
    return ExecutiveSignal(
        signal=Signal.BUY,
        composite=0.21,
        confidence=62,
        target_price=67200.0,
        upside=0.12,
        horizon_days=60,
        horizon_label="1~3개월",
        pillars={"factor": 0.22, "ml": 0.1, "llm": 0.3},
    )


def news_sentiment(score=0.5, n_used=6) -> NewsSentiment:
    return NewsSentiment(score=score, score_100=round(score * 100), n_articles=8, n_used=n_used)


def filings_intel(tone=0.3, severities=()) -> FilingIntelligence:
    risks = [
        RiskFactor(title=f"r{i}", detail="d", severity=s, category="other", source_ref="x")
        for i, s in enumerate(severities)
    ]
    return FilingIntelligence(summary=["요약"], tone=tone, risk_factors=risks)


def audit_filing() -> FilingItem:
    return FilingItem(
        id=text_id("감사보고서"),
        title="감사보고서 제출",
        form_type="감사보고서",
        filed_at=NOW - timedelta(days=1),
        text=AUDIT_TEXT,
        is_earnings=False,
        source="DART",
    )


def critic_wire(**kw) -> dict:
    base = {
        "thesis": "HBM4 퀄 통과와 RSI 58.3의 추세가 매수 논지를 지지한다.",
        "bull_points": ["모멘텀 점수 71"],
        "bear_points": ["파운드리 수율 부진"],
        "contradictions": [],
        "confidence_adjustment": 3,
        "critical_flag": False,
        "critical_category": None,
        "critical_reason": None,
        "critical_evidence": None,
        "critical_source_ref": None,
    }
    return {**base, **kw}


def run_critique(wire: dict, *, news=None, filings=None, filing_items=(), news_items=(), rec=None):
    md = make_market_data(Market.KR)
    fake = FakeLLMClient({"critic": wire})
    out = asyncio.run(
        critique(
            fake,
            md.instrument,
            make_quant(),
            news,
            filings,
            md.fundamentals,
            make_prelim(),
            news_items=news_items,
            filing_items=filing_items,
            recorder=rec,
        )
    )
    return out, fake


# ---------------------------------------------------------------------------------- critic


def test_critic_happy_path_and_context(settings):
    rec = LLMRecorder()
    news = make_news(8, now=NOW)
    review, fake = run_critique(
        critic_wire(),
        news=news_sentiment(),
        filings=filings_intel(),
        news_items=news,
        filing_items=make_filings(NOW),
        rec=rec,
    )
    assert review.thesis.startswith("HBM4") and review.confidence_adjustment == 3 and not review.critical_flag
    assert review.bull_points == ["모멘텀 점수 71"] and rec.errors == []
    call = fake.calls[0]
    assert call.task == "critic" and call.system == CRITIC_SYSTEM and call.wire_model is CriticWire
    for tag in ("<quant_json>", "<llm_json>", "<prelim_signal>", f'<document id="{news[0].id}"'):
        assert tag in call.user
    assert '"signal":"Buy"' in call.user and '"rsi14":58.3' in call.user and '"ret_1m_pct":4.2' in call.user


def test_confidence_adjustment_is_clamped(settings):
    rec = LLMRecorder()
    low, _ = run_critique(critic_wire(confidence_adjustment=-40), news=news_sentiment(), rec=rec)
    high, _ = run_critique(critic_wire(confidence_adjustment=9), news=news_sentiment(), filings=filings_intel())
    assert low.confidence_adjustment == -25 and high.confidence_adjustment == 5
    assert any("[-25, +5]" in n for n in rec.notices) and rec.errors == []  # clamp = degraded, not failed


def test_positive_adjustment_needs_evidence():
    rec = LLMRecorder()
    review, _ = run_critique(critic_wire(confidence_adjustment=4), news=news_sentiment(n_used=0), filings=None, rec=rec)
    assert review.confidence_adjustment == 0 and any("근거가 부족" in n for n in rec.notices)
    neg, _ = run_critique(critic_wire(confidence_adjustment=-8), news=None, filings=None)
    assert neg.confidence_adjustment == -8  # a negative adjustment is always allowed


def test_critical_flag_honoured_only_with_a_grounded_quote():
    f = audit_filing()
    good, _ = run_critique(
        critic_wire(
            critical_flag=True,
            critical_category="audit_opinion",
            critical_reason="감사의견 거절 공시",
            critical_evidence="감사의견을 거절하였습니다",
            critical_source_ref=f.id,
        ),
        filing_items=[f],
    )
    assert (
        good.critical_flag
        and "감사의견 거절" in good.critical_reason
        and "감사의견을 거절하였습니다" in good.critical_reason
    )
    assert "감사보고서 제출" in good.critical_reason

    rec = LLMRecorder()
    fabricated, _ = run_critique(
        critic_wire(
            critical_flag=True,
            critical_category="fraud_embezzlement",
            critical_reason="횡령",
            critical_evidence="대표이사 횡령 혐의 구속",
            critical_source_ref=f.id,
        ),
        filing_items=[f],
        rec=rec,
    )
    assert not fabricated.critical_flag and fabricated.critical_reason is None
    assert any("critical_flag" in n for n in rec.notices) and rec.errors == []

    no_cat, _ = run_critique(
        critic_wire(
            critical_flag=True,
            critical_category=None,
            critical_reason="x",
            critical_evidence="감사의견을 거절하였습니다",
            critical_source_ref=f.id,
        ),
        filing_items=[f],
    )
    assert not no_cat.critical_flag
    no_docs, _ = run_critique(
        critic_wire(
            critical_flag=True,
            critical_category="audit_opinion",
            critical_reason="x",
            critical_evidence="감사의견을 거절하였습니다",
            critical_source_ref=f.id,
        )
    )
    assert not no_docs.critical_flag  # no raw documents -> nothing to ground against


@pytest.mark.parametrize(
    "category,evidence,text",
    [
        (
            "going_concern",
            "substantial doubt about the Company's ability to continue as a going concern",
            "The auditor concluded there is no substantial doubt about the Company's ability to continue "
            "as a going concern.",
        ),
        (
            "going_concern",
            "There is substantial doubt about the Company's ability to continue as a going concern",
            "There is no substantial doubt about the Company's ability to continue as a going concern.",
        ),  # fuzzy only
        ("delisting_halt", "상장폐지 사유에 해당", "당사는 상장폐지 사유에 해당하지 않습니다."),
        ("audit_opinion", "감사의견 거절", "감사인은 적정의견을 표명하였으며, 감사의견 거절 사유는 없습니다."),
    ],
)
def test_critical_flag_rejects_negated_or_non_verbatim_quotes(category, evidence, text):
    """Regression: the fuzzy 3-gram rule accepted a quote whose source sentence says the opposite, and a
    critical_flag caps the signal at Hold/Sell."""
    f = FilingItem(id=text_id(text), title="감사보고서", form_type="감사보고서", filed_at=NOW, text=text, source="DART")
    rec = LLMRecorder()
    review, _ = run_critique(
        critic_wire(
            critical_flag=True,
            critical_category=category,
            critical_reason="r",
            critical_evidence=evidence,
            critical_source_ref=f.id,
        ),
        filing_items=[f],
        rec=rec,
    )
    assert not review.critical_flag and review.critical_reason is None
    assert any("critical_flag" in n for n in rec.notices)


def test_unsupported_numbers_are_dropped():
    rec = LLMRecorder()
    review, _ = run_critique(
        critic_wire(
            thesis="목표가 90,000원까지 상승 여력이 크다.",
            bull_points=["상승여력 +12%", "상승확률 83%"],
            bear_points=["RSI 58"],
        ),
        news=news_sentiment(),
        rec=rec,
    )
    assert review.thesis == ""  # fusion falls back to its deterministic thesis
    assert review.bull_points == ["상승여력 +12%"]  # 12.0 is in prelim (upside_pct), 83 is not anywhere
    assert review.bear_points == ["RSI 58"]  # lenient rounding of 58.3
    assert any("결정론적 논지로 대체" in n for n in rec.notices) and any("문장 1건 폐기" in n for n in rec.notices)
    assert rec.errors == []  # the review is still served


@pytest.mark.parametrize(
    "thesis,kept",
    [
        ("상승확률 83%로 매수 우위가 뚜렷하다.", False),  # 83 is nowhere in the context
        ("RSI 58.3과 상승여력 12%가 매수 논지를 지지한다.", True),  # both in quant/prelim JSON
        ("RSI 58과 상승여력 +12.0%가 매수 논지를 지지한다.", True),  # rounding at the stated precision
        ("삼성전자(005930)는 HBM4 퀄 통과로 추세가 견조하다.", True),  # the ticker is not a figure
        ("목표가 67,200원, 상승여력 12%이나 상승확률 91%는 과도하다.", False),  # one bad number sinks the thesis
    ],
)
def test_critic_thesis_is_all_or_nothing_on_unsupported_numbers(thesis, kept):
    """An unsupported number blanks the WHOLE thesis ('' -> fusion's deterministic thesis); never a partial edit."""
    rec = LLMRecorder()
    review, _ = run_critique(critic_wire(thesis=thesis), news=news_sentiment(), rec=rec)
    assert review.thesis == (" ".join(thesis.split()) if kept else "")
    assert any("결정론적 논지로 대체" in n for n in rec.notices) is (not kept)


def test_critic_failure_returns_none_and_records():
    rec = LLMRecorder()
    review, _ = run_critique("모델 거절(refusal)", rec=rec)
    assert review is None and rec.errors == ["비평 실패: 모델 거절(refusal)"]


def test_compact_context_is_deterministic():
    md = make_market_data(Market.KR)
    q = quant_context(md.instrument, make_quant(), md.fundamentals)
    assert compact_json(q) == compact_json(quant_context(md.instrument, make_quant(), md.fundamentals))
    assert q["technicals"]["ret_12m_ex1m_pct"] == 25.0 and q["fundamentals"]["roe_pct"] == 10.5
    assert json.loads(compact_json(llm_context(None, None))) == {"news": None, "filings": None}
    assert prelim_context(make_prelim(), "KRW")["upside_pct"] == 12.0


# -------------------------------------------------------------------------------- llm_pillar


def test_llm_pillar_formula():
    assert llm_pillar(None, None) == 0.0
    assert llm_pillar(news_sentiment(0.5, 10), None) == pytest.approx(0.5)  # renormalised news-only
    assert llm_pillar(news_sentiment(0.5, 2), None) == pytest.approx(0.5 * 0.4)  # thin evidence scaled
    assert llm_pillar(None, filings_intel(-0.4)) == pytest.approx(0.35 * -0.4)  # filings-only: NOT renormalised
    assert llm_pillar(news_sentiment(0.5, 10), filings_intel(-0.2)) == pytest.approx(0.65 * 0.5 + 0.35 * -0.2)


def test_llm_pillar_risk_penalty_and_bounds():
    both = llm_pillar(news_sentiment(0.5, 10), filings_intel(-0.2, ["high", "high", "critical", "low"]))
    assert both == pytest.approx(0.255 - 0.30)
    assert llm_pillar(None, filings_intel(0.0, ["critical"] * 5)) == pytest.approx(-0.40)  # capped
    assert llm_pillar(news_sentiment(-1.0, 9), filings_intel(-1.0, ["critical"] * 3)) == -1.0  # clipped
    assert risk_penalty(None) == 0.0 and risk_penalty(filings_intel(0.0, ["medium", "low"])) == 0.0


def test_filings_only_pillar_is_capped_so_one_tone_cannot_drive_it_to_the_bound():
    """Regression: with news missing, 0.35/0.35 renormalisation turned one filing's tone of +-1 into a
    pillar of +-1 (40% of the composite). The magnitude is now capped at 0.35*|tone| + risk penalty."""
    assert llm_pillar(None, filings_intel(1.0)) == pytest.approx(0.35)
    assert llm_pillar(None, filings_intel(-1.0)) == pytest.approx(-0.35)
    # the risk penalty applies in full on top of the capped tone (downside is never diluted)
    assert llm_pillar(None, filings_intel(1.0, ["critical"])) == pytest.approx(0.35 - 0.20)
    assert llm_pillar(None, filings_intel(-1.0, ["critical"] * 3)) == pytest.approx(-0.35 - 0.40)
    for tone in (-1.0, -0.6, -0.2, 0.0, 0.3, 0.8, 1.0):
        for sev in ((), ("high",), ("high", "critical"), ("critical",) * 4):
            f = filings_intel(tone, sev)
            p = llm_pillar(None, f)
            assert abs(p) <= 0.35 * abs(tone) + risk_penalty(f) + 1e-12
            # continuity: filings-only == news present but with no usable evidence
            assert p == pytest.approx(llm_pillar(news_sentiment(0.9, n_used=0), f))
    # news-only keeps full weight (evidence-scaled by n_used); both present still sum to weight 1
    assert llm_pillar(news_sentiment(1.0, 5), None) == pytest.approx(1.0)
    assert llm_pillar(news_sentiment(1.0, 5), filings_intel(1.0)) == pytest.approx(1.0)


# ------------------------------------------------------------------------- run_llm_analysis

NEWS_WIRE = {"articles": [], "key_positives": [], "key_negatives": [], "key_events": []}
FILINGS_WIRE = {
    "summary": ["3분기 영업이익 10조 4,000억원"],
    "earnings_highlights": ["매출액 79조 1,000억원"],
    "guidance": "none",
    "tone": 0.4,
    "risk_factors": [
        {"title": "t", "detail": "d", "severity": "high", "category": "financial", "source_ref": "", "evidence": "x"}
    ],
}


class GatedFake(FakeLLMClient):
    """News waits until the filings call has started. Sequential execution would deadlock, so it proves concurrency."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.filings_started = None

    async def call_structured(self, task, system, user, wire_model, effort=None, max_tokens=None):
        if self.filings_started is None:
            self.filings_started = asyncio.Event()
        if task == "filings":
            self.filings_started.set()
        elif task == "news":
            await self.filings_started.wait()
        return await super().call_structured(task, system, user, wire_model, effort, max_tokens)


def _news_wire_for(md) -> dict:
    return {
        **NEWS_WIRE,
        "articles": [
            {"news_id": n.id, "sentiment": 0.6, "relevance": 1.0, "event_type": "product", "evidence": n.title}
            for n in md.news
        ],
    }


def test_run_llm_analysis_end_to_end(settings):
    md = make_market_data(Market.KR)
    fake = GatedFake({"news": _news_wire_for(md), "filings": FILINGS_WIRE, "critic": critic_wire()}, cost_per_call=0.01)
    seen: list[LLMIntelligence] = []

    def prelim_fn(partial: LLMIntelligence) -> ExecutiveSignal:
        seen.append(partial)
        return make_prelim()

    out = asyncio.run(asyncio.wait_for(run_llm_analysis(fake, md, make_quant(), settings, prelim_fn, now=NOW), 5))
    assert [c.task for c in fake.calls][-1] == "critic" and {c.task for c in fake.calls[:2]} == {"news", "filings"}
    assert out.news.n_used == len(md.news) and out.news.score == pytest.approx(0.6)
    assert out.filings.tone == pytest.approx(0.4) and out.filings.risk_factors == []  # the ref-less risk was dropped
    assert out.critic is not None and out.critic.confidence_adjustment == 3
    assert out.llm_pillar == pytest.approx(llm_pillar(out.news, out.filings)) == pytest.approx(0.65 * 0.6 + 0.35 * 0.4)
    assert out.cost_usd == pytest.approx(0.03) and not out.skipped
    # a dropped ungrounded risk is degraded-but-served: a notice, not an error
    assert any("source_ref가 없는 위험 요인" in n for n in out.notices) and out.errors == []
    assert seen and seen[0].critic is None and seen[0].llm_pillar == out.llm_pillar
    assert seen[0].notices == out.notices[: len(seen[0].notices)]  # the partial carries notices too
    critic_user = fake.calls[-1].user
    assert f'<document id="{md.filings[0].id}"' in critic_user  # raw docs reach the critic


def test_run_llm_analysis_never_raises(settings):
    md = make_market_data(Market.KR)
    fake = FakeLLMClient({"news": RuntimeError("socket closed"), "filings": "서버 오류(500)", "critic": critic_wire()})

    async def prelim_async(partial):
        return make_prelim()

    out = asyncio.run(run_llm_analysis(fake, md, make_quant(), settings, prelim_async, now=NOW))
    assert out.news is None and out.filings is None and out.llm_pillar == 0.0
    assert out.critic is not None  # critic still runs on quant + prelim
    assert any("뉴스 감성 예외: RuntimeError" in e for e in out.errors)  # hard failures stay in errors
    assert any("공시 분석 실패: 서버 오류(500)" in e for e in out.errors)
    assert not any("실패" in n or "예외" in n for n in out.notices)


def test_prelim_failure_skips_critic_but_keeps_results(settings):
    md = make_market_data(Market.KR)
    fake = FakeLLMClient({"news": _news_wire_for(md), "filings": FILINGS_WIRE, "critic": critic_wire()})

    def boom(_):
        raise ValueError("fusion not ready")

    out = asyncio.run(run_llm_analysis(fake, md, make_quant(), settings, boom, now=NOW))
    assert out.news is not None and out.filings is not None and out.critic is None
    assert any("예비 신호 계산 실패" in e for e in out.errors)
    assert "critic" not in [c.task for c in fake.calls]


@pytest.mark.parametrize("bad_prelim", [None, {"signal": "Buy"}])
def test_prelim_without_a_signal_skips_critic_observably(settings, bad_prelim):
    """Regression: prelim_fn returning None skipped the critic with no entry in errors."""
    md = make_market_data(Market.KR)
    fake = FakeLLMClient({"news": _news_wire_for(md), "filings": FILINGS_WIRE, "critic": critic_wire()})
    out = asyncio.run(run_llm_analysis(fake, md, make_quant(), settings, lambda p: bad_prelim, now=NOW))
    assert out.critic is None and out.news is not None
    assert any("비평 생략" in e for e in out.errors)
    assert "critic" not in [c.task for c in fake.calls]


def test_run_llm_analysis_web_research_for_us(settings):
    s = settings.model_copy(deep=True)
    s.llm.web_research = True
    md = make_market_data(Market.US, seed=7)
    web = (
        "요약",
        [
            {
                "url": "https://www.fool.com/a",
                "title": "Apple Q3 call",
                "page_age": None,
                "cited_text": ["Revenue was $94,930 million."],
            }
        ],
    )
    fake = FakeLLMClient(
        {"news": _news_wire_for(md), "filings": FILINGS_WIRE, "critic": critic_wire(), "web_research": web}
    )
    out = asyncio.run(run_llm_analysis(fake, md, make_quant(), s, lambda p: make_prelim(), now=NOW))
    tasks = [c.task for c in fake.calls]
    assert tasks.index("web_research") < tasks.index("filings")
    filings_user = next(c.user for c in fake.calls if c.task == "filings")
    assert "Revenue was $94,930 million." in filings_user
    assert "웹: Apple Q3 call <https://www.fool.com/a>" in out.filings.sources

    s.llm.web_research = False
    fake2 = FakeLLMClient({"news": _news_wire_for(md), "filings": FILINGS_WIRE, "critic": critic_wire()})
    asyncio.run(run_llm_analysis(fake2, md, make_quant(), s, lambda p: make_prelim(), now=NOW))
    assert "web_research" not in [c.task for c in fake2.calls]


# ------------------------------------------------------------------------ English output (report.language)

HANGUL = re.compile("[가-힣]")


def _en(settings):
    s = settings.model_copy(deep=True)
    s.report.language = "en"
    return s


def run_critique_en(wire: dict, *, news=None, filings=None, rec=None, **docs):
    """``critique`` with ``lang="en"`` (``docs``: news_items / filing_items)."""
    md = make_market_data(Market.KR)
    fake = FakeLLMClient({"critic": wire})
    args = (fake, md.instrument, make_quant(), news, filings, md.fundamentals, make_prelim())
    return asyncio.run(critique(*args, recorder=rec, lang="en", **docs)), fake


def test_critic_in_english_uses_english_prompt_schema_and_headers():
    f = audit_filing()
    review, fake = run_critique_en(
        critic_wire(thesis="Momentum is strong with RSI at 58.3.", bull_points=["Momentum score 71"]),
        news=news_sentiment(),
        filing_items=[f],
        news_items=make_news(2, now=NOW),
    )
    call = fake.calls[0]
    assert call.system is CRITIC_SYSTEM_EN and call.wire_model is CriticWireEN
    assert call.user.startswith(
        "Company: 삼성전자 (ticker 005930, KOSPI, market KR, sector 반도체)\nSource documents: 3\n"
    )
    assert DOCUMENT_NOTICE_EN in call.user and "원문 자료 수" not in call.user
    assert review.thesis == "Momentum is strong with RSI at 58.3." and review.bull_points == ["Momentum score 71"]
    # the language follows the recorder when lang is not given
    rec = LLMRecorder(lang="en")
    _, fake2 = run_critique(critic_wire(), rec=rec)
    assert fake2.calls[0].system is CRITIC_SYSTEM_EN
    _, fake3 = run_critique(critic_wire())
    assert fake3.calls[0].system is CRITIC_SYSTEM and fake3.calls[0].wire_model is CriticWire


def test_critic_notices_in_english():
    rec = LLMRecorder(lang="en")
    review, _ = run_critique_en(
        critic_wire(
            thesis="Upside to a KRW 90,000 target.",
            bull_points=["Upside +12%", "Probability of a rise 83%"],
            confidence_adjustment=4,
            critical_flag=True,
            critical_category="fraud_embezzlement",
            critical_reason="Embezzlement.",
            critical_evidence="대표이사 횡령 혐의 구속",
            critical_source_ref="x",
        ),
        news=news_sentiment(n_used=0),
        rec=rec,
    )
    assert review.thesis == "" and review.bull_points == ["Upside +12%"] and review.confidence_adjustment == 0
    assert not review.critical_flag
    assert rec.notices == [
        "Critic: dropped 1 line with figures not in the inputs",
        "Critic: thesis has figures not supported by the inputs (90,000) → replaced by the deterministic thesis",
        "Critic: positive confidence adjustment (+4) capped at 0 — thin text evidence",
        "Critic: critical_flag ignored — its evidence quote could not be verified verbatim (and not negated) in "
        "the source",
    ]
    rec = LLMRecorder(lang="en")
    clamped, _ = run_critique_en(critic_wire(confidence_adjustment=-40), rec=rec)
    assert clamped.confidence_adjustment == -25
    assert rec.notices == ["Critic: confidence adjustment -40 clamped to the allowed range [-25, +5]"]
    rec = LLMRecorder(lang="en")
    assert run_critique_en("model refused (refusal)", rec=rec)[0] is None
    assert rec.errors == ["Critic failed: model refused (refusal)"]


def test_critical_flag_in_english_keeps_the_verbatim_korean_quote():
    f = audit_filing()
    review, _ = run_critique_en(
        critic_wire(
            critical_flag=True,
            critical_category="audit_opinion",
            critical_reason="The auditor disclaimed its opinion.",
            critical_evidence="감사의견을 거절하였습니다",
            critical_source_ref=f.id,
        ),
        filing_items=[f],
    )
    assert review.critical_flag
    assert review.critical_reason == (
        "audit opinion disclaimer / adverse: The auditor disclaimed its opinion. "
        f"(evidence: \"감사의견을 거절하였습니다\" — filing '감사보고서 제출 ({f.filed_at:%Y-%m-%d})')"
    )
    translated, _ = run_critique_en(  # a translated quote cannot be verified against the Korean source
        critic_wire(
            critical_flag=True,
            critical_category="audit_opinion",
            critical_reason="r",
            critical_evidence="the auditor disclaimed its opinion",
            critical_source_ref=f.id,
        ),
        filing_items=[f],
    )
    assert not translated.critical_flag


@pytest.mark.parametrize(
    "evidence,text",
    [
        ("상장폐지 사유에 해당", "당사는 상장폐지 사유에 해당하지 않습니다."),
        (
            "substantial doubt about the Company's ability to continue as a going concern",
            "The auditor concluded there is no substantial doubt about the Company's ability to continue "
            "as a going concern.",
        ),
    ],
)
def test_negated_critical_quotes_are_rejected_in_english_mode(evidence, text):
    f = FilingItem(id=text_id(text), title="Audit report", form_type="10-K", filed_at=NOW, text=text, source="SEC")
    rec = LLMRecorder(lang="en")
    review, _ = run_critique_en(
        critic_wire(
            critical_flag=True,
            critical_category="going_concern",
            critical_reason="r",
            critical_evidence=evidence,
            critical_source_ref=f.id,
        ),
        filing_items=[f],
        rec=rec,
    )
    assert not review.critical_flag and any(n.startswith("Critic: critical_flag ignored") for n in rec.notices)


def test_critical_category_labels_have_english_twins():
    assert set(CRITICAL_CATEGORY_LABELS_EN) == set(CRITICAL_CATEGORY_LABELS)
    assert critical_category_label("going_concern", "en") == "going-concern uncertainty"
    assert critical_category_label("going_concern") == CRITICAL_CATEGORY_LABELS["going_concern"]
    assert critical_category_label("dilutive_issuance", "en") == "dilutive issuance > 20%"
    assert not any(HANGUL.search(v) for v in CRITICAL_CATEGORY_LABELS_EN.values())


def test_run_llm_analysis_in_english_end_to_end(settings):
    s = _en(settings)
    md = make_market_data(Market.KR)
    fake = FakeLLMClient(
        {"news": _news_wire_for(md), "filings": FILINGS_WIRE, "critic": critic_wire(thesis="Trend is firm.")},
        settings=s,
    )
    out = asyncio.run(run_llm_analysis(fake, md, make_quant(), s, lambda p: make_prelim(), now=NOW))
    by_task = {c.task: c for c in fake.calls}
    assert by_task["news"].system is prompts.NEWS_SYSTEM_EN and by_task["news"].wire_model is NewsWireEN
    assert by_task["filings"].system is prompts.FILINGS_SYSTEM_EN and by_task["filings"].wire_model is FilingsWireEN
    assert by_task["critic"].system is CRITIC_SYSTEM_EN and by_task["critic"].wire_model is CriticWireEN
    assert all(c.user.startswith("Company: 삼성전자 (ticker 005930") for c in fake.calls)
    assert out.critic.thesis == "Trend is firm." and out.errors == []
    assert out.notices == ["Filings analysis: dropped 1 risk factor without a source_ref"]


def test_run_llm_analysis_errors_in_english(settings):
    s = _en(settings)
    md = make_market_data(Market.KR)
    fake = FakeLLMClient(
        {"news": RuntimeError("socket closed"), "filings": "server error (500)", "critic": critic_wire()}, settings=s
    )
    out = asyncio.run(run_llm_analysis(fake, md, make_quant(), s, lambda p: make_prelim(), now=NOW))
    assert out.errors == [
        "News sentiment failed with an exception: RuntimeError: socket closed",
        "Filings analysis failed: server error (500)",
    ]

    def boom(_):
        raise ValueError("fusion not ready")

    fake = FakeLLMClient({"news": _news_wire_for(md), "filings": FILINGS_WIRE, "critic": critic_wire()}, settings=s)
    out = asyncio.run(run_llm_analysis(fake, md, make_quant(), s, boom, now=NOW))
    assert out.errors == ["preliminary signal failed — critic skipped: ValueError: fusion not ready"]
    out = asyncio.run(run_llm_analysis(fake, md, make_quant(), s, lambda p: None, now=NOW))
    assert out.errors == ["invalid preliminary signal (not an ExecutiveSignal) — critic skipped: NoneType"]
    fake = FakeLLMClient({"news": _news_wire_for(md), "filings": FILINGS_WIRE}, settings=s)  # no critic answer
    out = asyncio.run(run_llm_analysis(fake, md, make_quant(), s, lambda p: make_prelim(), now=NOW))
    assert out.errors == ["Critic failed: FakeLLMClient error: no canned response for 'critic'"]


def test_run_llm_analysis_web_research_in_english(settings):
    s = _en(settings)
    s.llm.web_research = True
    md = make_market_data(Market.US, seed=7)
    web = (
        "summary",
        [{"url": "https://www.fool.com/a", "title": "Apple Q3 call", "cited_text": ["Revenue was $94,930 million."]}],
    )
    fake = FakeLLMClient(
        {"news": _news_wire_for(md), "filings": FILINGS_WIRE, "critic": critic_wire(), "web_research": web},
        settings=s,
    )
    out = asyncio.run(run_llm_analysis(fake, md, make_quant(), s, lambda p: make_prelim(), now=NOW))
    assert "Web: Apple Q3 call <https://www.fool.com/a>" in out.filings.sources
    web_call = next(c for c in fake.calls if c.task == "web_research")
    assert web_call.system is prompts.WEB_RESEARCH_SYSTEM_EN
    filings_user = next(c.user for c in fake.calls if c.task == "filings")
    assert 'form_type="Web research (earnings call)"' in filings_user
    assert all(not HANGUL.search(m) for m in [*out.errors, *out.notices])


def test_critic_reshows_only_the_most_relevant_documents():
    from quantcio.llm.critic import critic_articles, critic_filings
    from quantcio.models import ArticleSentiment, FilingIntelligence, NewsSentiment, RiskFactor

    news = make_news(8)
    rated = [
        ArticleSentiment(news_id=news[5].id, sentiment=-0.9, relevance=1.0, event_type="other", evidence="x"),
        ArticleSentiment(news_id=news[2].id, sentiment=0.2, relevance=0.5, event_type="other", evidence="x"),
    ]
    ns = NewsSentiment(score=0.0, score_100=0, n_articles=8, n_used=2, articles=rated)
    picked = critic_articles(news, ns, 3)
    assert [n.id for n in picked[:2]] == [news[5].id, news[2].id] and len(picked) == 3
    assert [n.id for n in critic_articles(news, None, 2)] == [n.id for n in news[:2]]  # newest without ratings
    assert critic_articles([], ns, 3) == []

    filings = make_filings(datetime(2026, 9, 24, tzinfo=UTC))
    risky = filings[-1]
    fi = FilingIntelligence(
        risk_factors=[
            RiskFactor(title="t", detail="d", severity="critical", category="legal_regulatory", source_ref=risky.id)
        ]
    )
    chosen = critic_filings(filings, fi, 1)
    assert [f.id for f in chosen] == [risky.id]  # the filing behind a critical risk comes first
