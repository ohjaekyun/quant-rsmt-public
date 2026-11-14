"""Offline tests for quantcio.fusion.signal (fuse, guardrails, confidence, target price, thesis)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import make_market_data

from quantcio.fusion.signal import (
    HORIZON_LABELS,
    PILLARS,
    apply_barrier_guardrail,
    build_thesis,
    fuse,
    pillar_coverage,
    pillar_values,
    signal_from_composite,
    target_price_components,
)
from quantcio.models import (
    BarrierOutcome,
    CriticReview,
    DistributionForecast,
    ExecutionPlan,
    ExecutiveSignal,
    FactorScore,
    FilingIntelligence,
    LLMIntelligence,
    Market,
    MarketData,
    NewsSentiment,
    QuantSignal,
    RiskFactor,
    Signal,
    Technicals,
)

FACTOR_LABELS = {
    "momentum": "모멘텀",
    "value": "밸류에이션",
    "volatility": "변동성(안정성)",
    "quality": "퀄리티",
    "trend": "추세",
}


# ----------------------------------------------------------------------------- builders


def make_tech(close: float = 100.0, ret_1m: float | None = 0.02, atr: float = 2.0) -> Technicals:
    return Technicals(
        last_close=close,
        atr=atr,
        atr_pct=atr / close,
        rsi14=55.0,
        ma50=close * 0.97,
        ma200=close * 0.9,
        high_52w=close * 1.1,
        low_52w=close * 0.7,
        realized_vol_60d=0.28,
        beta_1y=1.0,
        max_drawdown_1y=-0.2,
        ret_1m=ret_1m,
        ret_3m=0.05,
        ret_12m_ex1m=0.2,
        rel_strength_3m=0.01,
        support=close * 0.95,
        resistance=close * 1.08,
    )


def make_factors(trend: float | None = 60.0) -> list[FactorScore]:
    scores = {"momentum": 72.0, "value": 41.0, "volatility": 55.0, "quality": 60.0, "trend": trend}
    return [FactorScore(name=n, label=FACTOR_LABELS[n], score=s) for n, s in scores.items()]


def make_dist(
    mu: float = 0.03,
    h: int = 20,
    degraded: bool = False,
    skill_ok: bool = True,
    cv: dict | None = None,
    reasons: list[str] | None = None,
) -> DistributionForecast:
    return DistributionForecast(
        horizon_days=h,
        mu=mu,
        sigma=0.08,
        nu=6.0,
        beta=1.0,
        mu_excess=0.01,
        sigma_excess=0.06,
        mu_market=0.005,
        sigma_market=0.04,
        rho=0.1,
        prob_up=0.62,
        prob_outperform=0.55,
        expected_return=0.032,
        q10=-0.07,
        q50=0.03,
        q90=0.13,
        var95=-0.10,
        cvar95=-0.14,
        cv_metrics=cv if cv is not None else {"auc": 0.56, "base_auc": 0.53, "logloss": 0.685, "base_logloss": 0.690},
        skill_ok=skill_ok,
        direction_skill_ok=skill_ok,  # these tests model the direction gate (skill_ok's meaning before 2026-09-29)
        degraded=degraded,
        degraded_reasons=reasons or [],
    )


# RSMT-Q CV metrics worth the full ML skill component (AUC gain >= 0.05, log-loss gain >= 0.01) and none of it
_FULL_SKILL_CV = {"auc": 0.60, "base_auc": 0.53, "logloss": 0.670, "base_logloss": 0.690}
_NO_EDGE_CV = {"auc": 0.53, "base_auc": 0.53, "logloss": 0.690, "base_logloss": 0.690}
# default ML source: a skilled distribution with a +2% 20-day log mean (the ML pillar exists unless dist=None)
_DEFAULT_DIST = make_dist(mu=0.02, cv=_FULL_SKILL_CV)


def make_quant(
    factor: float = 0.3,
    ml_p: float = 0.3,
    *,
    dist: DistributionForecast | None = _DEFAULT_DIST,
    trend: float | None = 60.0,
    ret_1m: float | None = 0.02,
    factor_available: bool = True,
) -> QuantSignal:
    return QuantSignal(
        technicals=make_tech(ret_1m=ret_1m),
        factors=make_factors(trend),
        factor_composite=(50 + 50 * factor) if factor_available else None,
        factor_pillar=factor if factor_available else 0.0,
        distribution=dist,
        ml_pillar=ml_p,
    )


def make_news(n_used: int = 8, score: float = 0.3) -> NewsSentiment:
    return NewsSentiment(score=score, score_100=round(score * 100), n_articles=n_used + 2, n_used=n_used)


def make_llm(
    pillar: float = 0.3,
    *,
    news: bool = True,
    filings: bool = True,
    critic: CriticReview | None = None,
    skipped: bool = False,
    n_used: int = 8,
    risks: list[RiskFactor] | None = None,
) -> LLMIntelligence:
    return LLMIntelligence(
        news=make_news(n_used) if news else None,
        filings=FilingIntelligence(summary=["요약"], tone=0.2, risk_factors=risks or []) if filings else None,
        critic=critic,
        llm_pillar=pillar,
        skipped=skipped,
    )


@pytest.fixture
def md() -> MarketData:
    return make_market_data(Market.KR)


# ------------------------------------------------------------------ threshold mapping


@pytest.mark.parametrize(
    ("composite", "confidence", "expected"),
    [
        (0.45, 65, Signal.STRONG_BUY),
        (0.45, 64, Signal.BUY),  # Strong Buy needs confidence >= 65
        (0.449999, 95, Signal.BUY),
        (0.15, 50, Signal.BUY),
        (0.149999, 50, Signal.HOLD),
        (-0.149999, 50, Signal.HOLD),
        (-0.15, 50, Signal.SELL),
        (-1.0, 90, Signal.SELL),
    ],
)
def test_signal_mapping_boundaries(settings, composite, confidence, expected):
    sig, notes = signal_from_composite(composite, confidence, settings)
    assert sig == expected
    if composite >= 0.45 and confidence < 65:
        assert notes and "Strong Buy를 Buy로 하향" in notes[0]


@pytest.mark.parametrize(
    ("value", "expected"), [(0.45, Signal.STRONG_BUY), (0.15, Signal.BUY), (-0.15, Signal.SELL), (0.0, Signal.HOLD)]
)
def test_fuse_exact_threshold_composites(settings, md, value, expected):
    """All three pillars equal to a threshold -> composite hits it exactly (no float drift)."""
    sig = fuse(md, make_quant(value, value), make_llm(value), settings)
    assert sig.composite == pytest.approx(value, abs=0)
    if expected == Signal.STRONG_BUY:
        assert sig.confidence >= settings.fusion.strong_buy_min_confidence
    assert sig.signal == expected
    assert sig.horizon_label == HORIZON_LABELS[expected]


def test_strong_buy_confidence_downgrade(settings, md):
    critic = CriticReview(thesis="근거 부족", confidence_adjustment=-25)
    quant = make_quant(0.6, 0.6, dist=make_dist(mu=0.02, cv=_NO_EDGE_CV))  # direction gate passed, no CV edge
    sig = fuse(md, quant, make_llm(0.6, critic=critic), settings)
    assert sig.composite >= settings.fusion.thresholds.strong_buy
    assert sig.confidence < settings.fusion.strong_buy_min_confidence
    assert sig.signal == Signal.BUY
    assert any("Strong Buy를 Buy로 하향" in o for o in sig.overrides)


# ------------------------------------------------------------------------ guardrails


def test_guardrail_critical_flag_caps_at_hold(settings, md):
    critic = CriticReview(thesis="횡령 의혹", critical_flag=True, critical_reason="대표이사 횡령 혐의 공시")
    sig = fuse(md, make_quant(0.5, 0.5), make_llm(0.5, critic=critic), settings)
    assert sig.composite > 0
    assert sig.signal == Signal.HOLD
    assert any("치명적 리스크 가드레일" in o and "횡령" in o for o in sig.overrides)


def test_guardrail_critical_flag_negative_composite_sells(settings, md):
    critic = CriticReview(thesis="감사의견 거절", critical_flag=True, critical_reason="감사의견 거절")
    sig = fuse(md, make_quant(-0.1, -0.1), make_llm(-0.1, critic=critic), settings)
    assert -0.15 < sig.composite < 0  # would be Hold without the guardrail
    assert sig.signal == Signal.SELL


def test_guardrail_critical_risk_factor(settings, md):
    risk = RiskFactor(
        title="계속기업 불확실성", detail="감사인 지적", severity="critical", category="financial", source_ref="f1"
    )
    sig = fuse(md, make_quant(0.5, 0.5), make_llm(0.5, risks=[risk]), settings)
    assert sig.signal == Signal.HOLD
    assert any("계속기업 불확실성" in o for o in sig.overrides)


def test_guardrail_falling_knife(settings, md):
    quant = make_quant(0.3, 0.3, trend=20.0, ret_1m=-0.25)
    sig = fuse(md, quant, make_llm(0.3), settings)
    assert sig.signal == Signal.HOLD
    assert any("낙하하는 칼" in o for o in sig.overrides)


def test_guardrail_falling_knife_llm_override_and_trend_ok(settings, md):
    # strong text evidence (llm_pillar > 0.4) lets the Buy stand
    sig = fuse(md, make_quant(0.3, 0.3, trend=20.0, ret_1m=-0.25), make_llm(0.5), settings)
    assert sig.signal in (Signal.BUY, Signal.STRONG_BUY)
    assert not any("낙하하는 칼" in o for o in sig.overrides)
    # trend score not weak -> no guardrail
    sig = fuse(md, make_quant(0.3, 0.3, trend=45.0, ret_1m=-0.25), make_llm(0.3), settings)
    assert sig.signal == Signal.BUY
    # ret_1m exactly -20% is not "< -20%"
    sig = fuse(md, make_quant(0.3, 0.3, trend=20.0, ret_1m=-0.20), make_llm(0.3), settings)
    assert sig.signal == Signal.BUY


def test_guardrail_buy_needs_a_positive_target_upside(settings, md):
    """Integration regression (live run 2026-09-24): AAPL was served 'Buy, target $334.20 (-0.8%)'.
    A Buy-side call whose blended target is not above the price is capped at Hold (noted)."""
    price = md.last_price
    md.fundamentals = md.fundamentals.model_copy(
        update={"analyst_target_price": price * 0.85, "per": 30.0, "sector_per": 15.0}
    )
    md.valuation_history = None
    sig = fuse(md, make_quant(0.5, 0.5, dist=make_dist(mu=-0.02)), make_llm(0.5), settings)
    assert sig.composite >= settings.fusion.thresholds.buy and sig.upside <= 0
    assert sig.signal == Signal.HOLD and sig.horizon_label == HORIZON_LABELS[Signal.HOLD]
    assert any(o.startswith("목표가 괴리 가드레일") and o.endswith("→Hold") for o in sig.overrides)
    assert sig.thesis == build_thesis(
        make_quant(0.5, 0.5, dist=make_dist(mu=-0.02)),
        Signal.HOLD,
        sig.composite,
        pillar_values(make_quant(0.5, 0.5, dist=make_dist(mu=-0.02)), make_llm(0.5)),
    )
    # a target above the price keeps the Buy-side call
    md.fundamentals = md.fundamentals.model_copy(update={"analyst_target_price": price * 1.3, "per": 12.0})
    sig2 = fuse(md, make_quant(0.5, 0.5, dist=make_dist(mu=0.02)), make_llm(0.5), settings)
    assert sig2.upside > 0 and sig2.signal in (Signal.BUY, Signal.STRONG_BUY)
    assert not any(o.startswith("목표가 괴리 가드레일") for o in sig2.overrides)
    # Sell / Hold calls are never touched by this rule
    sig3 = fuse(md, make_quant(-0.6, -0.6, dist=make_dist(mu=-0.05)), make_llm(-0.6), settings)
    assert sig3.signal == Signal.SELL


def _ml_only_target(md: MarketData, settings, upside: float) -> DistributionForecast:
    """Leave the RSMT-Q μ as the ONLY target component and tune it so the target upside is `upside`."""
    md.fundamentals = md.fundamentals.model_copy(update={"analyst_target_price": None, "per": None})
    H = settings.fusion.target_horizon_days
    return make_dist(mu=math.log1p(upside) * 20 / H, h=20)


def test_buy_needs_min_upside_not_just_a_positive_one(settings, md):
    """Generalised target-gap guardrail: Buy/Strong Buy needs upside >= fusion.min_buy_upside (2%), so a
    'Buy, target +1%' (positive, but inside noise) is Hold with a Korean note."""
    assert settings.fusion.min_buy_upside == pytest.approx(0.02)
    dist = _ml_only_target(md, settings, 0.01)
    sig = fuse(md, make_quant(0.5, 0.5, dist=dist), make_llm(0.5), settings)
    assert sig.upside == pytest.approx(0.01) and sig.composite >= settings.fusion.thresholds.buy
    assert sig.signal == Signal.HOLD and sig.horizon_label == HORIZON_LABELS[Signal.HOLD]
    note = next(o for o in sig.overrides if o.startswith("목표가 괴리 가드레일"))
    assert "+1.0% < 매수 최소 상승여력 +2.0%" in note and note.endswith("→Hold")
    # Strong Buy is capped the same way (straight to Hold, not one notch)
    sig = fuse(md, make_quant(0.9, 0.9, dist=dist), make_llm(0.9), settings)
    assert sig.signal == Signal.HOLD and any("Strong Buy→Hold" in o for o in sig.overrides)
    # a note never prints two equal numbers around '<': 1.96% vs 2% gets two decimals
    dist = _ml_only_target(md, settings, 0.0196)
    sig = fuse(md, make_quant(0.5, 0.5, dist=dist), make_llm(0.5), settings)
    assert any("+1.96% < 매수 최소 상승여력 +2.00%" in o for o in sig.overrides)


def test_target_gap_guardrail_boundaries(settings, md):
    """Buy survives at upside == min_buy_upside (>=); Sell survives at upside == max_sell_upside (the lift
    needs strictly >). The thresholds are set to the exact upside the fuse computes."""
    dist = _ml_only_target(md, settings, 0.03)
    settings.fusion.min_buy_upside = -1.0
    probe = fuse(md, make_quant(0.3, 0.3, dist=dist), make_llm(0.3), settings)
    u = probe.upside
    assert probe.signal == Signal.BUY and u == pytest.approx(0.03)
    settings.fusion.min_buy_upside = u
    assert fuse(md, make_quant(0.3, 0.3, dist=dist), make_llm(0.3), settings).signal == Signal.BUY
    settings.fusion.min_buy_upside = math.nextafter(u, 1.0)
    assert fuse(md, make_quant(0.3, 0.3, dist=dist), make_llm(0.3), settings).signal == Signal.HOLD

    sell_q = make_quant(-0.5, -0.5, dist=dist)
    settings.fusion.max_sell_upside = u
    sig = fuse(md, sell_q, make_llm(-0.5), settings)
    assert sig.signal == Signal.SELL and not any(o.startswith("목표가 괴리") for o in sig.overrides)
    settings.fusion.max_sell_upside = math.nextafter(u, -1.0)
    assert fuse(md, sell_q, make_llm(-0.5), settings).signal == Signal.HOLD
    # a broken threshold fails loudly
    settings.fusion.max_sell_upside = float("nan")
    with pytest.raises(ValueError, match="max_sell_upside"):
        fuse(md, sell_q, make_llm(-0.5), settings)


def test_sell_with_large_target_upside_is_lifted_to_hold(settings, md):
    """Symmetric guardrail: 'Sell, target +8%' is conflicting evidence -> Hold with a Korean note; the
    deterministic thesis and horizon follow the final call."""
    assert settings.fusion.max_sell_upside == pytest.approx(0.05)
    dist = _ml_only_target(md, settings, 0.08)
    quant = make_quant(-0.5, -0.5, dist=dist)
    sig = fuse(md, quant, LLMIntelligence(skipped=True), settings)
    assert sig.composite <= settings.fusion.thresholds.sell and sig.upside == pytest.approx(0.08)
    assert sig.signal == Signal.HOLD and sig.horizon_label == HORIZON_LABELS[Signal.HOLD]
    note = next(o for o in sig.overrides if o.startswith("목표가 괴리 가드레일"))
    assert "매도 신호이나 목표가 상승여력 +8.0% > 매도 허용 상한 +5.0%" in note
    assert "상충" in note and note.endswith("Sell→Hold")
    assert "관망 판단" in sig.thesis and "매도" not in sig.thesis.split("→")[-1].split(".")[0]
    # below the limit the Sell stands
    sig = fuse(md, make_quant(-0.5, -0.5, dist=_ml_only_target(md, settings, 0.04)), make_llm(-0.5), settings)
    assert sig.signal == Signal.SELL
    # Buy-side and Hold calls are never touched by the Sell-side rule
    sig = fuse(md, make_quant(0.0, 0.0, dist=_ml_only_target(md, settings, 0.3)), make_llm(0.0), settings)
    assert sig.signal == Signal.HOLD and not any(o.startswith("목표가 괴리") for o in sig.overrides)


def test_critical_event_sell_is_never_lifted_by_the_target(settings, md):
    """A consensus target that has not priced fraud / going concern must not turn the critical Sell into
    a Hold."""
    dist = _ml_only_target(md, settings, 0.15)
    critic = CriticReview(thesis="횡령 혐의", critical_flag=True, critical_reason="대표이사 횡령 혐의 공시")
    sig = fuse(md, make_quant(-0.5, -0.5, dist=dist), make_llm(-0.5, critic=critic), settings)
    assert sig.upside > settings.fusion.max_sell_upside
    assert sig.signal == Signal.SELL
    assert not any(o.startswith("목표가 괴리") for o in sig.overrides)
    # a critical risk factor (no critic flag) turning a mildly negative composite into Sell: same rule
    risk = RiskFactor(
        title="계속기업 불확실성", detail="감사인 지적", severity="critical", category="financial", source_ref="f1"
    )
    sig = fuse(md, make_quant(-0.1, -0.1, dist=dist), make_llm(-0.1, risks=[risk]), settings)
    assert -0.15 < sig.composite < 0 and sig.signal == Signal.SELL


def test_guardrail_llm_skipped_and_unskilled_ml(settings, md):
    # direction gate failed (2 of 3 fold layouts passed: the skill component keeps 2/3 of its points)
    gate = {**_FULL_SKILL_CV, "gate_folds_n": 3.0, "gate_folds_pass": 2.0}
    unskilled = make_quant(0.8, 0.8, dist=make_dist(mu=0.02, skill_ok=False, cv=gate))
    sig = fuse(md, unskilled, LLMIntelligence(skipped=True), settings)
    assert sig.composite >= settings.fusion.thresholds.strong_buy
    assert sig.confidence >= settings.fusion.strong_buy_min_confidence
    assert sig.signal == Signal.BUY
    assert any("Strong Buy를 Buy로 제한" in o for o in sig.overrides)

    skilled = make_quant(0.8, 0.8)
    assert fuse(md, skilled, LLMIntelligence(skipped=True), settings).signal == Signal.STRONG_BUY
    # llm=None is treated like skipped
    assert fuse(md, unskilled, None, settings).signal == Signal.BUY


def test_guardrail3_uses_rsmt_skill_gate_when_distribution_present(settings, md):
    quant = make_quant(0.8, 0.8, dist=make_dist(skill_ok=False))
    sig = fuse(md, quant, LLMIntelligence(skipped=True), settings)
    assert sig.signal == Signal.BUY


# ------------------------------------------------------------------ RSMT-Q adjustments


def test_degraded_rsmt_penalty(settings, md):
    base = fuse(md, make_quant(0.2, 0.2, dist=make_dist()), make_llm(0.2, n_used=2), settings)
    deg = fuse(
        md,
        make_quant(0.2, 0.2, dist=make_dist(degraded=True, reasons=["오프셋 입력 누락(z|v)"])),
        make_llm(0.2, n_used=2),
        settings,
    )
    assert 5 < base.confidence - 10 and base.confidence < 95  # both inside the clip range
    assert base.confidence - deg.confidence == 10
    assert deg.confidence_breakdown["rsmt_degraded"] == -10
    assert base.confidence_breakdown["rsmt_degraded"] == 0
    assert "RSMT 저하 모드: 오프셋 입력 누락(z|v)" in deg.overrides


def test_rsmt_ml_skill_component(settings, md):
    # auc edge 0.03/0.05 = 0.6 -> 6 pts; logloss edge 0.005/0.01 = 0.5 -> 5 pts
    quant = make_quant(0.2, 0.2, dist=make_dist())
    sig = fuse(md, quant, make_llm(0.2), settings)
    assert sig.confidence_breakdown["ml_skill"] == pytest.approx(11.0)
    # missing CV metrics -> 0 and an observable note
    sig3 = fuse(md, make_quant(0.2, 0.2, dist=make_dist(cv={})), make_llm(0.2), settings)
    assert sig3.confidence_breakdown["ml_skill"] == 0
    assert any("교차검증 지표 누락" in o for o in sig3.overrides)


def test_rsmt_ml_skill_component_follows_robust_gate(settings, md):
    # the auc/logloss gains are the primary layout's; a model the fold-robust gate rejects earns only the
    # share of layouts that passed (11 pts x 1/3), and nothing without a gate record
    cv = {"auc": 0.53, "base_auc": 0.50, "logloss": 0.690, "base_logloss": 0.695}
    rejected = make_dist(skill_ok=False, cv={**cv, "gate_folds_n": 3.0, "gate_folds_pass": 1.0})
    sig = fuse(md, make_quant(0.2, 0.2, dist=rejected), make_llm(0.2), settings)
    assert sig.confidence_breakdown["ml_skill"] == pytest.approx(11.0 / 3, abs=1e-3)
    no_record = make_dist(skill_ok=False, cv=cv)
    sig2 = fuse(md, make_quant(0.2, 0.2, dist=no_record), make_llm(0.2), settings)
    assert sig2.confidence_breakdown["ml_skill"] == 0
    passed = make_dist(skill_ok=True, cv={**cv, "gate_folds_n": 3.0, "gate_folds_pass": 2.0})
    sig3 = fuse(md, make_quant(0.2, 0.2, dist=passed), make_llm(0.2), settings)
    assert sig3.confidence_breakdown["ml_skill"] == pytest.approx(11.0)


# ------------------------------------------------------------------ barrier guardrail


def _execsig(signal: Signal) -> ExecutiveSignal:
    return ExecutiveSignal(
        signal=signal,
        composite=0.3,
        confidence=60,
        target_price=110.0,
        upside=0.1,
        horizon_days=60,
        horizon_label=HORIZON_LABELS[signal],
        pillars={},
        overrides=["기존"],
    )


def _plan(p_tp: float, p_sl: float, horizon: int = 60) -> ExecutionPlan:
    barrier = BarrierOutcome(
        p_take_profit=p_tp,
        p_stop_loss=p_sl,
        p_time_exit=round(1 - p_tp - p_sl, 10),
        expected_trade_return=0.0,
        kelly_optimal=0.0,
        n_paths=1000,
        horizon_days=horizon,
    )
    return ExecutionPlan(
        entry_price=100,
        entry_low=99,
        entry_high=101,
        stop_loss=94,
        stop_loss_pct=-0.06,
        take_profit=109,
        take_profit_pct=0.09,
        reward_risk=1.5,
        barrier=barrier,
        weight=0,
    )


def test_barrier_guardrail_downgrade(settings):
    out = apply_barrier_guardrail(_execsig(Signal.BUY), _plan(p_tp=0.20, p_sl=0.45), settings)
    assert out.signal == Signal.HOLD
    assert out.horizon_label == HORIZON_LABELS[Signal.HOLD]
    assert "배리어 가드레일" in out.overrides[0] and "45%" in out.overrides[0]
    assert "기존" in out.overrides

    out = apply_barrier_guardrail(_execsig(Signal.STRONG_BUY), _plan(p_tp=0.20, p_sl=0.45), settings)
    assert out.signal == Signal.BUY


def test_barrier_guardrail_keeps_thesis_consistent(settings, md):
    quant = make_quant(0.3, 0.3)
    sig = fuse(md, quant, LLMIntelligence(skipped=True), settings)
    assert sig.signal == Signal.BUY and "매수 판단." in sig.thesis
    out = apply_barrier_guardrail(sig, _plan(p_tp=0.2, p_sl=0.5), settings)
    assert out.signal == Signal.HOLD
    assert "관망 판단(배리어 가드레일로 하향)." in out.thesis and "매수 판단." not in out.thesis

    critic = CriticReview(thesis="수요 회복이 확인되는 구간이다.")
    sig = fuse(md, quant, make_llm(0.3, critic=critic), settings)
    out = apply_barrier_guardrail(sig, _plan(p_tp=0.2, p_sl=0.5), settings)
    assert out.thesis.startswith(critic.thesis) and "배리어 가드레일: 매수→관망 하향" in out.thesis
    assert sig.thesis == critic.thesis  # input not mutated


def test_barrier_guardrail_is_idempotent(settings):
    """Regression: re-applying the guardrail (e.g. after rebuilding the plan for the downgraded call)
    demoted Strong Buy -> Buy -> Hold; the rule is ONE notch."""
    once = apply_barrier_guardrail(_execsig(Signal.STRONG_BUY), _plan(p_tp=0.20, p_sl=0.45), settings)
    twice = apply_barrier_guardrail(once, _plan(p_tp=0.20, p_sl=0.45), settings)
    assert once.signal == twice.signal == Signal.BUY
    assert twice is once
    assert sum(o.startswith("배리어 가드레일:") for o in twice.overrides) == 1


def test_barrier_guardrail_no_trigger(settings):
    buy = _execsig(Signal.BUY)
    # exactly at the margin (0.45 == 0.30 + 0.15) -> not strictly greater
    assert apply_barrier_guardrail(buy, _plan(p_tp=0.30, p_sl=0.45), settings).signal == Signal.BUY
    # Hold/Sell are never touched
    assert apply_barrier_guardrail(_execsig(Signal.HOLD), _plan(0.1, 0.8), settings).signal == Signal.HOLD
    # no barrier -> no-op
    plan = _plan(0.1, 0.8).model_copy(update={"barrier": None})
    assert apply_barrier_guardrail(buy, plan, settings) is buy


def test_barrier_guardrail_margin_comes_from_settings(settings):
    """The margin is settings.fusion.barrier_guard_margin, not a module constant."""
    buy = _execsig(Signal.BUY)
    assert settings.fusion.barrier_guard_margin == pytest.approx(0.15)
    assert apply_barrier_guardrail(buy, _plan(p_tp=0.20, p_sl=0.45), settings).signal == Signal.HOLD
    settings.fusion.barrier_guard_margin = 0.30  # 0.45 > 0.20 + 0.30 is false -> no downgrade
    assert apply_barrier_guardrail(buy, _plan(p_tp=0.20, p_sl=0.45), settings).signal == Signal.BUY
    settings.fusion.barrier_guard_margin = 0.05  # 0.40 > 0.30 + 0.05 -> downgrade
    out = apply_barrier_guardrail(buy, _plan(p_tp=0.30, p_sl=0.40), settings)
    assert out.signal == Signal.HOLD and "+ 5%p" in out.overrides[0]
    # the keyword override wins over the config (what-if checks)
    assert apply_barrier_guardrail(buy, _plan(p_tp=0.30, p_sl=0.40), settings, margin=0.2).signal == Signal.BUY
    # a broken config value or a missing settings object fails loudly, never silently
    settings.fusion.barrier_guard_margin = float("nan")
    with pytest.raises(ValueError):
        apply_barrier_guardrail(buy, _plan(p_tp=0.30, p_sl=0.40), settings)
    with pytest.raises(TypeError):
        apply_barrier_guardrail(buy, _plan(p_tp=0.30, p_sl=0.40), 0.15)  # old positional `margin`


def test_barrier_guardrail_note_states_simulated_horizon(settings):
    out = apply_barrier_guardrail(_execsig(Signal.BUY), _plan(p_tp=0.20, p_sl=0.45, horizon=60), settings)
    assert out.overrides[0].startswith("배리어 가드레일: 60거래일 보유 시뮬레이션 기준 P(손절 선행) 45%")
    # an outcome without a recorded horizon: no invented basis in the note
    out = apply_barrier_guardrail(_execsig(Signal.BUY), _plan(p_tp=0.20, p_sl=0.45, horizon=0), settings)
    assert out.overrides[0].startswith("배리어 가드레일: P(손절 선행) 45%") and "거래일" not in out.overrides[0]


# ---------------------------------------------------------------- pillar renormalisation


def test_pillar_renormalisation_llm_skipped(settings, md):
    f, m = 0.4, 0.1
    sig = fuse(md, make_quant(f, m), LLMIntelligence(skipped=True, llm_pillar=0.9), settings)
    w = settings.fusion.weights
    expected = (w["factor"] * f + w["ml"] * m) / (w["factor"] + w["ml"])
    assert sig.composite == pytest.approx(expected, abs=1e-6)
    assert sig.pillars == {"factor": f, "ml": m}  # unavailable pillar omitted, never 0.0
    assert any("LLM 분석 미실행" in o for o in sig.overrides)
    assert sig.confidence_breakdown["evidence"] == 0


def test_pillar_renormalisation_no_text(settings, md):
    llm = make_llm(0.9, news=False, filings=False)  # ran, but no news/filings
    sig = fuse(md, make_quant(0.4, 0.1), llm, settings)
    w = settings.fusion.weights
    assert sig.composite == pytest.approx((w["factor"] * 0.4 + w["ml"] * 0.1) / (w["factor"] + w["ml"]), abs=1e-6)
    assert any("LLM 텍스트" in o for o in sig.overrides)


def test_pillar_renormalisation_ml_missing(settings, md):
    sig = fuse(md, make_quant(0.4, 0.0, dist=None), make_llm(0.2), settings)
    w = settings.fusion.weights
    assert sig.composite == pytest.approx((w["factor"] * 0.4 + w["llm"] * 0.2) / (w["factor"] + w["llm"]), abs=1e-6)
    assert sig.confidence_breakdown["ml_skill"] == 0
    assert any("ML/RSMT-Q 예측 없음" in o for o in sig.overrides)


# --------------------------------------------------------------------- confidence


def test_confidence_breakdown_components(settings, md):
    critic = CriticReview(thesis="테스트", confidence_adjustment=-5)
    md.warnings.append("뉴스 수집 실패")
    sig = fuse(md, make_quant(0.3, 0.1), make_llm(0.2, critic=critic, n_used=4), settings)
    cb = sig.confidence_breakdown
    for key in (
        "agreement",
        "strength",
        "ml_skill",
        "data_quality",
        "evidence",
        "rsmt_degraded",
        "critic_adjustment",
        "total_unclipped",
    ):
        assert key in cb
    parts = sum(v for k, v in cb.items() if k != "total_unclipped")
    assert cb["total_unclipped"] == pytest.approx(parts, abs=1e-3)
    assert sig.confidence == round(min(95, max(5, cb["total_unclipped"])))
    assert cb["strength"] == pytest.approx(20 * sig.composite / 0.6, abs=1e-3)
    assert cb["data_quality"] == pytest.approx(12.0)  # (15 - 3*1) * completeness 1.0
    assert cb["evidence"] == pytest.approx(5.0)  # 10 * 4/8
    assert cb["critic_adjustment"] == -5
    # all pillars positive -> share 1; std of (0.3, 0.1, 0.2) reduces agreement
    assert cb["agreement"] == pytest.approx(35 * (1 - np.std([0.3, 0.1, 0.2])), abs=1e-3)


def test_agreement_half_counts_neutral_and_conflicts(settings, md):
    # factor +0.5, ml -0.3 (disagrees), llm 0.02 (neutral -> half)
    sig = fuse(md, make_quant(0.5, -0.3), make_llm(0.02), settings)
    w = settings.fusion.weights
    share = w["factor"] * 1 + w["ml"] * 0 + w["llm"] * 0.5
    assert sig.confidence_breakdown["agreement"] == pytest.approx(
        35 * share * (1 - np.std([0.5, -0.3, 0.02])), abs=1e-3
    )


def test_agreement_scaled_by_pillar_coverage(settings, md):
    """Regression: agreement was renormalised over the available pillars, so a single pillar 'agreed with
    itself' (share 1, std 0) and scored the full 35 points - more than three agreeing, dispersed pillars."""
    w = settings.fusion.weights
    assert pillar_coverage(PILLARS, settings) == pytest.approx(1.0)
    assert pillar_coverage(["factor", "ml"], settings) == pytest.approx(w["factor"] + w["ml"])
    assert pillar_coverage([], settings) == 0.0

    # factor only (ML missing, LLM skipped): 35 x coverage 0.35, not 35
    solo = fuse(md, make_quant(0.4, 0.0, dist=None), LLMIntelligence(skipped=True), settings)
    assert solo.pillars == {"factor": 0.4}
    assert solo.confidence_breakdown["agreement"] == pytest.approx(35 * w["factor"], abs=1e-3)
    # LLM skipped: coverage 0.60 and one note says so (folded into the missing-pillar chip)
    two = fuse(md, make_quant(0.3, 0.1), LLMIntelligence(skipped=True), settings)
    cov = w["factor"] + w["ml"]
    assert two.confidence_breakdown["agreement"] == pytest.approx(35 * cov * (1 - np.std([0.3, 0.1])), abs=1e-3)
    notes = [o for o in two.overrides if "LLM 분석 미실행" in o]
    assert len(notes) == 1 and f"가용 필러 비중 {cov:.0%}만 반영" in notes[0]
    # all three pillars: coverage 1, no coverage clause
    full = fuse(md, make_quant(0.3, 0.1), make_llm(0.2), settings)
    assert not any("가용 필러 비중" in o for o in full.overrides)
    # three agreeing but dispersed pillars now beat a lone pillar on agreement
    dispersed = fuse(md, make_quant(0.6, 0.2), make_llm(0.4), settings)
    assert dispersed.confidence_breakdown["agreement"] > solo.confidence_breakdown["agreement"]


def test_confidence_clipped(settings, md):
    critic = CriticReview(thesis="매우 불확실", confidence_adjustment=-25)
    md.warnings.extend(["a", "b", "c", "d", "e"])
    sig = fuse(md, make_quant(0.0, 0.0, dist=None), make_llm(0.0, critic=critic, news=False), settings)
    assert sig.confidence == 5


# ------------------------------------------------------------------------ target price


def test_target_price_components_scaled_to_horizon(settings, md):
    p = md.last_price
    H = settings.fusion.target_horizon_days
    md.fundamentals.analyst_target_price = p * 1.2
    md.fundamentals.per = 10.0  # own-history median PER = 13.5
    comps = target_price_components(md, make_quant(dist=make_dist(mu=0.02, h=20)), settings)
    assert comps["analyst"] == pytest.approx(p * (1 + 0.2 * H / 252))
    assert comps["ml"] == pytest.approx(p * math.exp(0.02 * H / 20))  # mu is a LOG return
    assert comps["valuation"] == pytest.approx(p * (1 + (13.5 / 10.0 - 1) * H / 252))

    sig = fuse(md, make_quant(dist=make_dist(mu=0.02, h=20)), make_llm(), settings)
    b = settings.fusion.target_price_blend
    expected = sum(b[k] * comps[k] for k in comps) / sum(b[k] for k in comps)
    assert sig.target_price == pytest.approx(expected)
    assert sig.upside == pytest.approx(expected / p - 1)
    assert sig.target_components == pytest.approx(comps)
    assert sig.horizon_days == H


def test_target_price_negative_analyst_upside(settings, md):
    p = md.last_price
    md.fundamentals.analyst_target_price = p * 0.8
    comps = target_price_components(md, make_quant(dist=None), settings)
    H = settings.fusion.target_horizon_days
    assert comps["analyst"] == pytest.approx(p * (1 - 0.2 * H / 252))
    assert comps["analyst"] < p
    assert "ml" not in comps


def test_target_price_ml_component_is_log_return(settings, md):
    """Regression: DistributionForecast.mu is an h-day LOG mean. The component must be price·exp(mu·H/h), not
    the compounded simple return (1+mu)^(H/h), and a valid log mean <= -1 must not silently drop the component."""
    p = md.last_price
    settings.fusion.target_horizon_days = 20  # H == h -> exactly price·exp(mu)
    mu = 0.08
    comps = target_price_components(md, make_quant(dist=make_dist(mu=mu, h=20)), settings)
    assert comps["ml"] == pytest.approx(p * math.exp(mu), rel=1e-12)
    assert comps["ml"] != pytest.approx(p * (1 + mu), rel=1e-6)
    settings.fusion.target_horizon_days = 60
    comps = target_price_components(md, make_quant(dist=make_dist(mu=-1.2, h=20)), settings)
    assert comps["ml"] == pytest.approx(p * math.exp(-3.6))  # present, not dropped


def test_finite_accepts_numpy_scalars():
    """Regression: np.float32 / np.int64 are not float/int subclasses; they must not read as 'missing'."""
    from quantcio.fusion import signal as fsig

    assert fsig._finite(np.float32(1.5)) and fsig._finite(np.int64(3))
    assert not fsig._finite(np.float32("nan")) and not fsig._finite(True) and not fsig._finite(None)


def test_target_price_ml_component_uses_distribution(settings, md):
    p = md.last_price
    comps = target_price_components(md, make_quant(dist=make_dist(mu=0.03, h=20)), settings)
    assert comps["ml"] == pytest.approx(p * math.exp(0.03 * settings.fusion.target_horizon_days / 20))


def test_target_price_valuation_rules(settings, md):
    md.fundamentals.per = -3.0  # loss-making -> skip
    assert "valuation" not in target_price_components(md, make_quant(), settings)
    md.fundamentals.per = 12.0
    md.valuation_history = md.valuation_history.iloc[-10:]  # < 1 year -> fall back to sector PER 15
    comps = target_price_components(md, make_quant(), settings)
    p, H = md.last_price, settings.fusion.target_horizon_days
    assert comps["valuation"] == pytest.approx(p * (1 + (15.0 / 12.0 - 1) * H / 252))
    md.fundamentals.sector_per = None
    assert "valuation" not in target_price_components(md, make_quant(), settings)


def test_target_price_fallback_when_no_component(settings, md):
    md.fundamentals.analyst_target_price = None
    md.fundamentals.per = None
    sig = fuse(md, make_quant(0.3, 0.0, dist=None), make_llm(0.3), settings)
    assert sig.target_components == {}
    assert sig.target_price == pytest.approx(md.last_price * (1 + sig.composite * 0.1))
    assert any("목표가 산출 근거" in o for o in sig.overrides)


# ------------------------------------------------------------------------------ thesis


def test_thesis_from_critic_or_deterministic(settings, md):
    critic = CriticReview(thesis="HBM 공급 확대로 실적 개선 가시성이 높다.")
    assert fuse(md, make_quant(), make_llm(critic=critic), settings).thesis == critic.thesis
    sig = fuse(md, make_quant(), LLMIntelligence(skipped=True), settings)
    assert "LLM 미반영" in sig.thesis and "모멘텀(72) 우위" in sig.thesis and "밸류에이션(41) 약점" in sig.thesis
    quant = make_quant(dist=make_dist())
    text = build_thesis(quant, Signal.BUY, 0.3, {"factor": 0.3, "ml": 0.3, "llm": None})
    assert text == build_thesis(quant, Signal.BUY, 0.3, {"factor": 0.3, "ml": 0.3, "llm": None})  # deterministic
    assert "RSMT-Q 20일 상승확률 62%" in text


def test_fuse_does_not_mutate_inputs(settings, md):
    quant, llm = make_quant(), make_llm()
    q0, l0, w0 = quant.model_dump(), llm.model_dump(), list(md.warnings)
    fuse(md, quant, llm, settings)
    assert quant.model_dump() == q0 and llm.model_dump() == l0 and md.warnings == w0


# --------------------------------------------------------------- adversarial regressions


def test_unavailable_pillar_is_omitted_so_the_report_shows_missing(settings, md):
    """Regression: an unavailable pillar was written as 0.0, and report/render.py drew it as a neutral
    "0.00" bar (a missing key renders as "—" with no bar). The critic prompt also read it as neutral."""
    sig = fuse(md, make_quant(0.4, 0.1), LLMIntelligence(skipped=True), settings)
    assert "llm" not in sig.pillars
    sig = fuse(md, make_quant(0.4, 0.0, dist=None), make_llm(0.2, news=False, filings=False), settings)
    assert sig.pillars == {"factor": 0.4}
    # a genuinely neutral LLM pillar is still shown as 0.0
    assert fuse(md, make_quant(0.4, 0.1), make_llm(0.0), settings).pillars["llm"] == 0.0
    render = pytest.importorskip("quantcio.report.render")
    assert render.fmt_num(sig.pillars.get("llm"), 2, signed=True) == render.MISSING
    assert render._signed_share(sig.pillars.get("llm"))["missing"] is True


@pytest.mark.parametrize("H", [252, 504, 756])
def test_target_components_positive_beyond_one_year(settings, md, H):
    """Regression: the linear 12-month scaling went negative for H > 252 (H=504, analyst target 0.3x
    price -> component -0.4x price -> a NEGATIVE target price in the matrix)."""
    p = md.last_price
    settings.fusion.target_horizon_days = H
    md.fundamentals.analyst_target_price = 0.3 * p
    md.fundamentals.per = 60.0  # own-history median 13.5 -> fair 0.225x
    comps = target_price_components(md, make_quant(dist=None), settings)
    assert all(v > 0 for v in comps.values())
    assert comps["analyst"] == pytest.approx(p * 0.3 ** (H / 252))
    sig = fuse(md, make_quant(dist=None), make_llm(), settings)
    assert sig.target_price > 0 and sig.upside > -1
    # continuity at 12 months: linear and compounded forms agree
    if H == 252:
        assert comps["analyst"] == pytest.approx(0.3 * p)


def test_barrier_guardrail_never_edits_critic_words(settings):
    """Regression: the rewrite matched the bare "매수 판단.", a substring of "강력 매수 판단.", so a
    critic thesis on a Buy call became "... 강력 관망 판단(배리어 가드레일로 하향)."."""
    text = "HBM 수요 확대로 강력 매수 판단. 다만 밸류 부담."
    sig = _execsig(Signal.BUY).model_copy(update={"thesis": text})
    out = apply_barrier_guardrail(sig, _plan(p_tp=0.2, p_sl=0.5), settings)
    assert out.thesis.startswith(text) and "강력 관망" not in out.thesis
    assert "배리어 가드레일: 매수→관망 하향" in out.thesis


# -------------------------------------------------------------------- property tests


def test_property_fuse_invariants(settings):
    """Random pillars / models / LLM states / critic -> bounded components, guardrails only downgrade."""
    from quantcio.fusion.signal import BUYISH, SIGNAL_ORDER

    rng = np.random.default_rng(42)
    for _ in range(300):
        md = make_market_data(Market.KR)
        md.warnings.extend(["경고"] * int(rng.integers(0, 4)))
        f, m, lp = (float(x) for x in rng.uniform(-1, 1, 3))
        cv = {
            "auc": float(rng.uniform(0.45, 0.62)),
            "base_auc": 0.5,
            "logloss": float(rng.uniform(0.67, 0.71)),
            "base_logloss": 0.69,
        }
        dist = (
            make_dist(
                mu=float(rng.uniform(-0.2, 0.2)),
                skill_ok=bool(rng.random() < 0.5),
                degraded=bool(rng.random() < 0.2),
                cv=cv if rng.random() < 0.9 else {},
            )
            if rng.random() < 0.7
            else None
        )
        quant = make_quant(
            f,
            m,
            dist=dist,
            trend=float(rng.uniform(0, 100)) if rng.random() < 0.9 else None,
            ret_1m=float(rng.uniform(-0.4, 0.2)),
            factor_available=bool(rng.random() < 0.9),
        )
        critic = CriticReview(
            thesis="" if rng.random() < 0.5 else "비평 논지",
            confidence_adjustment=int(rng.integers(-25, 6)),
            critical_flag=bool(rng.random() < 0.1),
        )
        mode = int(rng.integers(0, 3))
        llm = (
            None
            if mode == 0
            else LLMIntelligence(skipped=True)
            if mode == 1
            else make_llm(
                lp,
                news=bool(rng.random() < 0.8),
                filings=bool(rng.random() < 0.8),
                critic=critic,
                n_used=int(rng.integers(0, 12)),
            )
        )
        sig = fuse(md, quant, llm, settings)
        cb = sig.confidence_breakdown
        assert 5 <= sig.confidence <= 95 and -1 <= sig.composite <= 1
        assert 0 <= cb["agreement"] <= 35 and 0 <= cb["strength"] <= 20 and 0 <= cb["ml_skill"] <= 20
        assert 0 <= cb["data_quality"] <= 15 and 0 <= cb["evidence"] <= 10
        assert cb["rsmt_degraded"] in (0, -10) and -25 <= cb["critic_adjustment"] <= 5
        base, _ = signal_from_composite(sig.composite, sig.confidence, settings)
        critical = llm is not None and llm.critic is not None and llm.critic.critical_flag
        lifted = [o for o in sig.overrides if o.startswith("목표가 괴리 가드레일: 매도 신호이나")]
        if base == Signal.SELL and sig.signal == Signal.HOLD:
            # the ONLY upgrade: a score Sell contradicted by its own target (never a critical-event Sell)
            assert sig.upside > settings.fusion.max_sell_upside and not critical and len(lifted) == 1
        else:
            assert SIGNAL_ORDER.index(sig.signal) <= SIGNAL_ORDER.index(base)  # otherwise never upgrade
            assert not lifted
        # symmetric target-gap guardrail: a surviving call never contradicts its target
        assert sig.signal not in BUYISH or sig.upside >= settings.fusion.min_buy_upside
        assert (
            sig.signal != Signal.SELL
            or sig.upside <= settings.fusion.max_sell_upside
            or critical
            or any("치명적 리스크 가드레일" in o for o in sig.overrides)
        )
        assert sig.signal != Signal.STRONG_BUY or sig.confidence >= settings.fusion.strong_buy_min_confidence
        assert sig.horizon_label == HORIZON_LABELS[sig.signal]
        if critical:
            assert sig.signal not in BUYISH
            assert sig.composite >= 0 or sig.signal == Signal.SELL
        assert sig.target_price > 0 and sig.upside == pytest.approx(sig.target_price / md.last_price - 1)
        assert set(sig.pillars) == {k for k, v in pillar_values(quant, llm).items() if v is not None}
        # agreement never exceeds the configured weight share of the pillars actually present
        assert cb["agreement"] <= 35 * pillar_coverage(sig.pillars, settings) + 1e-3


def _state(
    avail: dict[str, float], *, ml_stats: tuple[float, float], n_used: int, critic: CriticReview | None
) -> tuple[QuantSignal, LLMIntelligence]:
    """Inputs whose available pillars are exactly `avail`; everything else is held fixed by the caller."""
    auc, logloss = ml_stats
    cv = {"auc": auc, "base_auc": 0.5, "logloss": logloss, "base_logloss": 0.69}
    quant = make_quant(
        avail.get("factor", 0.0),
        avail.get("ml", 0.0),
        dist=make_dist(mu=0.02, cv=cv) if "ml" in avail else None,
        factor_available="factor" in avail,
    )
    has_llm = "llm" in avail
    llm = make_llm(avail.get("llm", 0.0), news=has_llm, filings=has_llm, critic=critic, n_used=n_used)
    return quant, llm


def test_property_removing_a_pillar_never_raises_confidence(settings):
    """Removing a pillar never increases confidence, all else equal.

    "All else equal": the remaining pillars keep their values; warnings, fundamentals, the critic and the
    model statistics are unchanged; and the removed pillar did not contradict the call - its value lies
    between the remaining pillars' composite and their most extreme value in the same direction, so it
    confirms the call without being weaker than it. Losing such a pillar may only cost confidence
    (agreement, strength, ML skill, evidence). The unconditional version cannot hold for any
    sign-agreement rule and should not: a present pillar that CONTRADICTS the call must score below
    an absent one, or conflicting evidence would be free. Before the coverage fix a lone pillar kept the
    full 35 agreement points, so dropping a confirming second pillar cost nothing (asserted strictly below).
    """
    rng = np.random.default_rng(20260924)
    W = settings.fusion.weights
    checked_strict = 0
    for _ in range(400):
        md = make_market_data(Market.KR)
        md.warnings.extend(["경고"] * int(rng.integers(0, 3)))
        removed = str(rng.choice(PILLARS))
        rest = {p: float(rng.uniform(-1, 1)) for p in PILLARS if p != removed and rng.random() < 0.75}
        if rest:
            c_rest = sum(W[p] * v for p, v in rest.items()) / sum(W[p] for p in rest)
            extreme = max(rest.values()) if c_rest >= 0 else min(rest.values())
            q = float(c_rest + rng.uniform(0, 1) * (extreme - c_rest))
        else:
            q = float(rng.uniform(-1, 1))
        critic = (
            CriticReview(thesis="비평", confidence_adjustment=int(rng.integers(-25, 6))) if rng.random() < 0.5 else None
        )
        fixed = dict(
            ml_stats=(float(rng.uniform(0.45, 0.62)), float(rng.uniform(0.67, 0.71))),
            n_used=int(rng.integers(0, 12)),
            critic=critic,
        )
        with_q = fuse(md, *_state({**rest, removed: q}, **fixed), settings)
        without = fuse(md, *_state(rest, **fixed), settings)
        assert set(with_q.pillars) == set(rest) | {removed} and set(without.pillars) == set(rest)
        a_with, a_without = with_q.confidence_breakdown, without.confidence_breakdown
        assert a_without["agreement"] <= a_with["agreement"] + 1e-3
        assert a_without["total_unclipped"] <= a_with["total_unclipped"] + 1e-3
        assert without.confidence <= with_q.confidence or (
            abs(a_without["total_unclipped"] - a_with["total_unclipped"]) < 1e-3
        )
        if len(rest) == 1:  # the old bug: a lone pillar kept all 35 agreement points
            assert a_without["agreement"] < a_with["agreement"] - 1.0
            checked_strict += 1
    assert checked_strict > 20


# ------------------------------------------------------------------ English output + factor-model blend

_HANGUL = __import__("re").compile("[가-힣]")


def _en(settings):
    s = settings.model_copy(deep=True)
    s.report.language = "en"
    return s


def _fm(pillar: float, skill_ok: bool = True, weak_side: str | None = None):
    from datetime import date

    from quantcio.models import FactorModelScore

    return FactorModelScore(
        as_of=date(2026, 9, 23), score=0.4, percentile=0.7, pillar=pillar, skill_ok=skill_ok, weak_side=weak_side
    )


def test_fuse_english_notes_and_labels_have_no_hangul(settings, md):
    en = _en(settings)
    for llm in (None, make_llm(skipped=True, news=False, filings=False)):
        sig = fuse(md, make_quant(0.3, 0.3), llm, en)
        assert sig.overrides and not any(_HANGUL.search(o) for o in sig.overrides), sig.overrides
        assert sig.overrides[0].startswith("LLM analysis not run — LLM pillar dropped, factor/ML weights renormalised")
        assert not _HANGUL.search(sig.thesis) and "composite" in sig.thesis
        assert sig.horizon_label in {
            "1–3 months (scale in)",
            "1–3 months",
            "Wait · review in 1 month",
            "Trim within 2 weeks",
        }
    ko = fuse(md, make_quant(0.3, 0.3), None, settings)
    assert ko.overrides[0].startswith("LLM 분석 미실행") and ko.horizon_label in HORIZON_LABELS.values()


def test_english_guardrail_notes(settings, md):
    en = _en(settings)
    knife = fuse(md, make_quant(0.6, 0.6, trend=20.0, ret_1m=-0.3), make_llm(0.1), en)
    assert any(o.startswith("Falling-knife guardrail: 1M return") for o in knife.overrides)
    crit = make_llm(
        0.5,
        risks=[
            RiskFactor(title="Audit disclaimer", detail="d", severity="critical", category="governance", source_ref="x")
        ],
    )
    sig = fuse(md, make_quant(0.5, 0.5), crit, en)
    assert any(o.startswith("Critical-risk guardrail (Audit disclaimer)") for o in sig.overrides)
    assert not any(_HANGUL.search(o) for o in sig.overrides + knife.overrides)


def test_barrier_guardrail_english_rewrites_verdict_and_is_idempotent(settings, md):
    en = _en(settings)
    q = make_quant(0.35, 0.35)
    sig = fuse(md, q, None, en)
    assert sig.signal == Signal.BUY and "Buy call." in sig.thesis
    b = BarrierOutcome(
        p_take_profit=0.2,
        p_stop_loss=0.5,
        p_time_exit=0.3,
        expected_trade_return=-0.01,
        kelly_optimal=0.0,
        n_paths=1000,
        horizon_days=60,
    )
    plan = ExecutionPlan(
        entry_price=100,
        entry_low=98,
        entry_high=101,
        stop_loss=94,
        stop_loss_pct=-0.06,
        take_profit=109,
        take_profit_pct=0.09,
        reward_risk=1.5,
        barrier=b,
        weight=0.0,
    )
    out = apply_barrier_guardrail(sig, plan, en)
    assert out.signal == Signal.HOLD and out.horizon_label == "Wait · review in 1 month"
    assert out.overrides[0].startswith("Barrier guardrail: 60-trading-day holding simulation: P(stop first) 50%")
    assert "Hold call (lowered by the barrier guardrail)." in out.thesis and "Buy call." not in out.thesis
    assert apply_barrier_guardrail(out, plan, en) is out  # idempotent
    critic = sig.model_copy(update={"thesis": "Strong franchise."})
    out2 = apply_barrier_guardrail(critic, plan, en)
    assert out2.thesis == "Strong franchise. (barrier guardrail: Buy→Hold downgrade)"


def test_build_thesis_english():
    q = make_quant(0.3, 0.3)
    t = build_thesis(q, Signal.BUY, 0.3, {"factor": 0.3, "ml": 0.3, "llm": None}, lang="en")
    assert t.startswith("Factor +0.30 · ML +0.30 · LLM n/a → composite +0.30, Buy call.")
    assert "(72) strong" in t and "(41) weak" in t


def test_factor_model_blend_math_and_note(settings, md):
    b = settings.backtest.factor_pillar_blend
    q = make_quant(0.3, 0.3).model_copy(update={"factor_model": _fm(0.7)})
    vals = pillar_values(q, None, settings)
    assert vals["factor"] == pytest.approx((1 - b) * 0.3 + b * 0.7)
    assert pillar_values(q, None)["factor"] == pytest.approx(0.3)  # no settings -> legacy only
    sig = fuse(md, q, None, settings)
    assert sig.pillars["factor"] == pytest.approx((1 - b) * 0.3 + b * 0.7)
    note = next(o for o in sig.overrides if o.startswith("팩터 필러 블렌드:"))
    assert f"기존 합성 +0.30×{1 - b:.0%} + 팩터 모델 +0.70×{b:.0%}" in note
    en_sig = fuse(md, q.model_copy(update={"factor_model": _fm(0.7, skill_ok=False)}), None, _en(settings))
    en_note = next(o for o in en_sig.overrides if o.startswith("Factor pillar blend:"))
    assert "legacy composite +0.30" in en_note and "(model unskilled — model pillar ×0.35 applied)" in en_note
    # confidence breakdown keeps its invariant (no extra keys)
    cb = sig.confidence_breakdown
    assert cb["total_unclipped"] == pytest.approx(sum(v for k, v in cb.items() if k != "total_unclipped"), abs=1e-3)


def test_factor_model_alone_when_legacy_missing(settings, md):
    q = make_quant(0.3, 0.3, factor_available=False).model_copy(update={"factor_model": _fm(-0.4)})
    sig = fuse(md, q, None, settings)
    assert sig.pillars["factor"] == pytest.approx(-0.4)
    assert any(o.startswith("팩터 필러 블렌드: 기존 합성 없음 → 팩터 모델 -0.40 단독") for o in sig.overrides)
    assert not any(o.startswith("팩터 점수 산출 불가") for o in sig.overrides)


def test_no_factor_model_leaves_fusion_unchanged(settings, md):
    q = make_quant(0.3, 0.3)
    sig = fuse(md, q, None, settings)
    assert sig.pillars["factor"] == pytest.approx(0.3)
    assert not any("블렌드" in o for o in sig.overrides)


def test_factor_model_weak_side_is_shrunk_only_on_that_side(settings, md):
    b = settings.backtest.factor_pillar_blend
    neg = make_quant(0.3, 0.3).model_copy(update={"factor_model": _fm(-0.6, weak_side="neg")})
    assert pillar_values(neg, None, settings)["factor"] == pytest.approx((1 - b) * 0.3 + b * -0.6 * 0.35)
    note = next(o for o in fuse(md, neg, None, settings).overrides if o.startswith("팩터 필러 블렌드:"))
    assert "팩터 모델 -0.21×" in note and "(모델 음(−)쪽 근거 약함 — ×0.35)" in note
    # the other side is untouched
    pos = neg.model_copy(update={"factor_model": _fm(0.6, weak_side="neg")})
    assert pillar_values(pos, None, settings)["factor"] == pytest.approx((1 - b) * 0.3 + b * 0.6)
    # pos weak side, English note
    en_note = next(
        o
        for o in fuse(
            md, neg.model_copy(update={"factor_model": _fm(0.6, weak_side="pos")}), None, _en(settings)
        ).overrides
        if o.startswith("Factor pillar blend:")
    )
    assert "(model's positive side is weak — ×0.35)" in en_note
    # symmetric verdict or the switch off: no change
    assert pillar_values(neg.model_copy(update={"factor_model": _fm(-0.6)}), None, settings)["factor"] == pytest.approx(
        (1 - b) * 0.3 + b * -0.6
    )
    off = settings.model_copy(deep=True)
    off.fusion.factor_pillar_asymmetry = "off"
    assert pillar_values(neg, None, off)["factor"] == pytest.approx((1 - b) * 0.3 + b * -0.6)


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_invalid_factor_blend_weight_raises(settings, md, bad):
    s = settings.model_copy(deep=True)
    s.backtest.factor_pillar_blend = bad
    q = make_quant(0.3, 0.3).model_copy(update={"factor_model": _fm(0.2)})
    with pytest.raises(ValueError, match="factor_pillar_blend"):
        fuse(md, q, None, s)


# --------------------------------------------------------------------------- ML pillar: stock-selection blend


def _excess(mu: float = 0.012, p_beat: float | None = 0.6, skill: bool = True):
    from datetime import date

    from quantcio.models import ExcessForecast

    return ExcessForecast(
        model="logit_w",
        as_of=date(2026, 9, 23),
        mu_excess=mu,
        score_z=1.2,
        percentile=0.88,
        p_beat_benchmark=p_beat,
        skill_ok=skill,
        gate={"confirmed": skill},
        oos_auc=0.534,
    )


def test_ml_pillar_blends_the_validated_selection_model():
    from quantcio.config import load_settings
    from quantcio.fusion.signal import SELECTION_BLEND, SELECTION_UNSKILLED, ml_pillar_value, selection_view

    s = load_settings()
    q = make_quant(ml_p=0.1, dist=make_dist())
    assert ml_pillar_value(q, s) == (0.1, "")  # no stock-selection model: the distribution pillar as is
    sel = selection_view(q.model_copy(update={"excess_model": _excess()}))
    expected_sel = 0.5 * math.tanh(2 * 0.012 / 0.06) + 0.5 * (2 * 0.6 - 1)
    assert sel == pytest.approx(expected_sel)
    ok, note = ml_pillar_value(q.model_copy(update={"excess_model": _excess()}), s)
    assert ok == pytest.approx((1 - SELECTION_BLEND) * 0.1 + SELECTION_BLEND * expected_sel)
    assert note.startswith("ML 필러 블렌드:") and "logit_w" in note and "스킬 확인" in note
    weak, note2 = ml_pillar_value(q.model_copy(update={"excess_model": _excess(skill=False)}), s)
    assert weak == pytest.approx((1 - SELECTION_BLEND) * 0.1 + SELECTION_BLEND * SELECTION_UNSKILLED * expected_sel)
    assert "스킬 미확인" in note2
    # the fused composite uses the blended ML pillar and keeps the note out of the warning chips' prefix set
    md = make_market_data()
    ex = fuse(md, q.model_copy(update={"excess_model": _excess()}), make_llm(0.0), s)
    assert ex.pillars["ml"] == pytest.approx(ok) and any(o.startswith("ML 필러 블렌드:") for o in ex.overrides)
    # without a distribution the probability alone carries the view
    q2 = make_quant(ml_p=0.0, dist=None).model_copy(update={"excess_model": _excess(p_beat=0.7)})
    assert selection_view(q2) == pytest.approx(0.4)


def test_ml_skill_follows_the_selection_model_when_it_carries_the_direction():
    from quantcio.fusion.signal import ML_SKILL_MAX, SELECTION_UNSKILLED, _ml_skill_component, ml_skill_ok

    # RSMT-Q gate failed, but a confirmed stock-selection model carries the direction: its skill counts
    q = make_quant(ml_p=0.1, dist=make_dist(skill_ok=False))
    confirmed = q.model_copy(update={"excess_model": _excess(skill=True)})
    unconfirmed = q.model_copy(update={"excess_model": _excess(skill=False)})
    assert not ml_skill_ok(q) and ml_skill_ok(confirmed) and not ml_skill_ok(unconfirmed)
    full, _ = _ml_skill_component(confirmed)
    part, _ = _ml_skill_component(unconfirmed)
    auc_part, ic_part = (0.534 - 0.5) / 0.05, 0.0  # _excess() has oos_auc 0.534 and no IC
    assert full == pytest.approx(ML_SKILL_MAX * 0.5 * auc_part + ML_SKILL_MAX * 0.5 * ic_part)
    assert part == pytest.approx(full * SELECTION_UNSKILLED)
