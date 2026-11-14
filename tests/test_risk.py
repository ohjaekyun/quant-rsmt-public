"""Offline tests for quantcio.fusion.risk (stops, take-profit, barrier/Kelly, sizing, portfolio risk)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from conftest import make_instrument, make_market_data

from quantcio.fusion.risk import (
    build_execution_plan,
    closed_form_kelly,
    conviction_scale,
    conviction_weights,
    earnings_trigger,
    estimate_covariance,
    format_money,
    portfolio_risk,
    size_portfolio,
    stop_loss_level,
    time_exit_trigger,
    trading_days_until,
)
from quantcio.fusion.signal import HORIZON_LABELS
from quantcio.models import (
    BarrierOutcome,
    DistributionForecast,
    ExecutionPlan,
    ExecutiveSignal,
    Fundamentals,
    LLMIntelligence,
    Market,
    MarketData,
    NewsSentiment,
    QuantSignal,
    Signal,
    Technicals,
)

# ----------------------------------------------------------------------------- builders


def make_tech(close: float, atr: float, support: float | None = None, rv: float = 0.28) -> Technicals:
    return Technicals(
        last_close=close,
        atr=atr,
        atr_pct=atr / close,
        rsi14=55.0,
        ma50=close,
        ma200=close * 0.9,
        high_52w=close * 1.1,
        low_52w=close * 0.7,
        realized_vol_60d=rv,
        max_drawdown_1y=-0.2,
        ret_1m=0.01,
        support=support,
    )


def make_quant(
    close: float,
    atr: float,
    support: float | None = None,
    *,
    vol: float | None = 0.25,
    rv: float = 0.28,
    with_dist: bool = False,
) -> QuantSignal:
    """`vol` = QuantSignal.vol_forecast_annual (the GARCH sizing vol; None = not available)."""
    dist = None
    if with_dist:
        dist = DistributionForecast(
            horizon_days=20,
            mu=0.02,
            sigma=0.08,
            nu=6,
            beta=1,
            mu_excess=0.01,
            sigma_excess=0.06,
            mu_market=0.01,
            sigma_market=0.04,
            rho=0.1,
            prob_up=0.7,
            prob_outperform=0.5,
            expected_return=0.02,
            q10=-0.08,
            q50=0.02,
            q90=0.12,
            var95=-0.1,
            cvar95=-0.14,
        )
    return QuantSignal(
        technicals=make_tech(close, atr, support, rv),
        factors=[],
        factor_composite=60.0,
        factor_pillar=0.2,
        distribution=dist,
        ml_pillar=0.2,
        vol_forecast_annual=vol,
    )


def make_sig(
    signal: Signal = Signal.BUY,
    composite: float = 0.3,
    confidence: int = 60,
    target: float = 0.0,
    upside: float = 0.05,
) -> ExecutiveSignal:
    return ExecutiveSignal(
        signal=signal,
        composite=composite,
        confidence=confidence,
        target_price=target,
        upside=upside,
        horizon_days=60,
        horizon_label=HORIZON_LABELS[signal],
        pillars={},
    )


def md_from_close(
    close: np.ndarray,
    market: Market = Market.KR,
    fx: float = 1.0,
    end: str = "2026-09-23",
) -> MarketData:
    idx = pd.bdate_range(end=end, periods=len(close))
    px = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(close), 1e6),
        },
        index=idx,
    ).astype("float64")
    return MarketData(
        instrument=make_instrument(market),
        prices=px,
        benchmark=px.iloc[0:0],
        fundamentals=Fundamentals(source="synthetic"),
        fx_to_base=fx,
    )


def correlated_mds(n: int, rho_load: float = 0.03, idio: float = 0.005, days: int = 400, seed: int = 1):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, days)
    out = []
    for i in range(n):
        r = rho_load * f + idio * rng.normal(0, 1, days)
        out.append(md_from_close(100.0 * np.exp(np.cumsum(r))))
    return out


def independent_mds(n: int, vol: float = 0.03, days: int = 400, seed: int = 1):
    rng = np.random.default_rng(seed)
    return [md_from_close(100.0 * np.exp(np.cumsum(vol * rng.normal(0, 1, days)))) for _ in range(n)]


def sized_plan(raw: float, kelly: float = 0.25, entry: float = 100.0) -> ExecutionPlan:
    return ExecutionPlan(
        entry_price=entry,
        entry_low=entry * 0.98,
        entry_high=entry * 1.01,
        stop_loss=entry * 0.94,
        stop_loss_pct=-0.06,
        take_profit=entry * 1.09,
        take_profit_pct=0.09,
        reward_risk=1.5,
        weight=0.0,
        raw_weight=raw,
        kelly_fraction=kelly,
        action="분할매수",
    )


class FakeDist:
    """Duck-typed stand-in for quant.distribution.ReturnDistribution."""

    def __init__(self, outcome: BarrierOutcome | None = None, fail: bool = False, horizon: int | None = None):
        self.outcome = outcome
        self.fail = fail
        self.calls: list[tuple[float, float]] = []
        if horizon is not None:
            self.horizon = horizon

    def barrier(self, stop_pct: float, take_pct: float) -> BarrierOutcome:
        self.calls.append((stop_pct, take_pct))
        if self.fail:
            raise RuntimeError("path matrix unavailable")
        return self.outcome


@pytest.fixture
def md() -> MarketData:
    return make_market_data(Market.KR)


# ------------------------------------------------------------------------ stop-loss


@pytest.mark.parametrize(
    ("atr", "expected_stop"),
    [
        (1.0, 96.0),  # 2*ATR = 2 < min 4% -> 96
        (3.0, 94.0),  # 2*ATR = 6 within [4, 15]
        (10.0, 85.0),
    ],  # 2*ATR = 20 > max 15% -> 85
)
def test_stop_loss_clipping(settings, atr, expected_stop):
    stop, snapped = stop_loss_level(100.0, atr, None, settings.risk)
    assert stop == pytest.approx(expected_stop)
    assert not snapped


def test_support_snap_rule(settings):
    cfg = settings.risk
    # ATR 3 -> stop_dist 6 -> snap window [92.2, 97]
    stop, snapped = stop_loss_level(100.0, 3.0, 93.0, cfg)
    assert snapped and stop == pytest.approx(93.0 * 0.99)
    stop, snapped = stop_loss_level(100.0, 3.0, 92.2, cfg)  # inclusive far edge
    assert snapped and stop == pytest.approx(92.2 * 0.99)
    stop, snapped = stop_loss_level(100.0, 3.0, 97.0, cfg)  # inclusive near edge: 96.03 < min 4% -> 96
    assert snapped and stop == pytest.approx(96.0)
    for support in (98.0, 90.0, None, float("nan")):  # outside the window / unknown
        stop, snapped = stop_loss_level(100.0, 3.0, support, cfg)
        assert not snapped and stop == pytest.approx(94.0)


def test_support_snap_respects_hard_limits(settings):
    cfg = settings.risk
    # ATR 1 -> dist 4 (min) -> window [94.8, 98]; 97.5*0.99 = 96.525 is tighter than min 4% -> 96
    stop, snapped = stop_loss_level(100.0, 1.0, 97.5, cfg)
    assert snapped and stop == pytest.approx(96.0)
    # ATR 10 -> dist 15 (max) -> window [80.5, 92.5]; 81*0.99 = 80.19 is wider than max 15% -> 85
    stop, snapped = stop_loss_level(100.0, 10.0, 81.0, cfg)
    assert snapped and stop == pytest.approx(85.0)


# ------------------------------------------------------------------- execution plan


def test_plan_levels_and_min_reward_risk(settings, md):
    p = md.last_price
    atr = 0.03 * p  # stop dist 6%
    plan = build_execution_plan(md, make_quant(p, atr), make_sig(target=p * 1.01), settings)
    assert plan.entry_price == pytest.approx(p)
    assert plan.entry_low == pytest.approx(p - 0.5 * atr)
    assert plan.entry_high == pytest.approx(p + 0.25 * atr)
    assert plan.stop_loss == pytest.approx(p * 0.94)
    assert plan.stop_loss_pct == pytest.approx(-0.06)
    # target below entry + 1.5R -> TP = entry + 1.5R and R:R exactly the minimum
    assert plan.take_profit == pytest.approx(p + 1.5 * 0.06 * p)
    assert plan.take_profit_pct == pytest.approx(0.09)
    assert plan.reward_risk == pytest.approx(settings.risk.min_reward_risk)
    # TP2 = max(analyst high 1.45p scaled to H=60: 1 + 0.45*60/252 = 1.107p, entry + 3R = 1.18p)
    assert plan.take_profit_2 == pytest.approx(p * 1.18)
    assert plan.weight == 0.0


def test_plan_tp_uses_target_when_higher(settings, md):
    p = md.last_price
    plan = build_execution_plan(md, make_quant(p, 0.03 * p), make_sig(target=p * 1.25), settings)
    assert plan.take_profit == pytest.approx(p * 1.25)
    assert plan.reward_risk == pytest.approx(0.25 / 0.06)
    assert plan.reward_risk >= settings.risk.min_reward_risk


def test_plan_tp_min_rr_holds_after_support_snap(settings, md):
    p = md.last_price
    atr = 0.03 * p  # window [0.922p, 0.97p]
    plan = build_execution_plan(md, make_quant(p, atr, support=0.93 * p), make_sig(target=0.5 * p), settings)
    assert plan.stop_loss == pytest.approx(0.93 * p * 0.99)
    assert plan.reward_risk == pytest.approx(settings.risk.min_reward_risk)


def test_tp2_none_when_not_above_tp(settings, md):
    p = md.last_price
    md.fundamentals.analyst_target_high = None
    plan = build_execution_plan(md, make_quant(p, 0.03 * p), make_sig(target=p * 1.5), settings)
    assert plan.take_profit_2 is None  # 3R = 1.18p < TP 1.5p


def test_tp2_scales_the_12m_analyst_high_to_the_holding_horizon(settings, md):
    """Integration regression (live run 2026-09-24): an unscaled 12-month analyst high printed
    'TP2 725,000원 (+154%)' on a 60-day plan for 005930. It is scaled like the target-price analyst
    component: entry * (1 + (high/entry - 1) * H/252)."""
    p = md.last_price
    H = settings.fusion.target_horizon_days
    md.fundamentals.analyst_target_high = 2.54 * p  # 12-month high +154%
    plan = build_execution_plan(md, make_quant(p, 0.03 * p), make_sig(target=p * 1.01), settings)
    assert plan.take_profit_2 == pytest.approx(p * (1 + 1.54 * H / 252))  # ~ +36.7% at H = 60
    assert plan.take_profit_2 > plan.take_profit
    md.fundamentals.analyst_target_high = 1.10 * p  # scaled +2.4% < 3R -> 3R rules
    plan = build_execution_plan(md, make_quant(p, 0.03 * p), make_sig(target=p * 1.01), settings)
    assert plan.take_profit_2 == pytest.approx(p * 1.18)


def test_kelly_from_barrier(settings, md):
    p = md.last_price
    outcome = BarrierOutcome(
        p_take_profit=0.4,
        p_stop_loss=0.25,
        p_time_exit=0.35,
        expected_trade_return=0.012,
        kelly_optimal=0.6,
        n_paths=1000,
        horizon_days=settings.fusion.target_horizon_days,
    )
    dist = FakeDist(outcome)
    plan = build_execution_plan(
        md,
        make_quant(p, 0.03 * p, with_dist=True),
        make_sig(target=p * 1.1),
        settings,
        dist=dist,
    )
    assert dist.calls == [(pytest.approx(plan.stop_loss_pct), pytest.approx(plan.take_profit_pct))]
    assert plan.barrier == outcome
    assert plan.kelly_fraction == pytest.approx(0.6 * settings.risk.kelly_cap)
    assert md.warnings == []


def test_barrier_failure_is_observable(settings, md):
    p = md.last_price
    plan = build_execution_plan(
        md,
        make_quant(p, 0.03 * p, with_dist=True),
        make_sig(target=p * 1.1),
        settings,
        dist=FakeDist(fail=True),
    )
    assert plan.barrier is None
    assert plan.kelly_fraction > 0  # closed-form fallback
    assert any("배리어 계산 실패" in w for w in md.warnings)


def test_missing_dist_object_is_observable(settings, md):
    p = md.last_price
    build_execution_plan(md, make_quant(p, 0.03 * p, with_dist=True), make_sig(target=p * 1.1), settings)
    assert any("경로 행렬 미전달" in w for w in md.warnings)


def test_kelly_closed_form_without_distribution(settings, md):
    p = md.last_price
    sig = make_sig(composite=0.3, target=p * 1.1)
    # no RSMT-Q distribution -> p0 = 0.5 + 0.25c
    plan2 = build_execution_plan(md, make_quant(p, 0.03 * p), sig, settings)
    prob2 = 0.5 * (0.5 + 0.25 * 0.3) + 0.5 * (0.5 + 0.5 * 0.3)
    assert plan2.kelly_fraction == pytest.approx(
        min(1, max(0, prob2 - (1 - prob2) / plan2.reward_risk)) * settings.risk.kelly_cap
    )
    assert closed_form_kelly(0.3, 1.5, 0.25) == 0.0  # negative edge -> 0
    assert closed_form_kelly(0.9, 0.0, 0.25) == 0.0  # degenerate payoff -> 0


def test_raw_weight_rules(settings, md):
    p = md.last_price
    q = make_quant(p, 0.03 * p, vol=0.25)
    plan = build_execution_plan(md, q, make_sig(Signal.BUY, composite=0.3, confidence=60), settings)
    assert plan.raw_weight == pytest.approx(0.3 * 0.60 / 0.25)
    low_vol = make_quant(p, 0.03 * p, vol=0.05)
    plan = build_execution_plan(md, low_vol, make_sig(Signal.STRONG_BUY, composite=0.5, confidence=80), settings)
    assert plan.raw_weight == pytest.approx(0.5 * 0.80 / 0.10)  # vol floored at 10%
    no_vol = make_quant(p, 0.03 * p, vol=None, rv=0.4)
    plan = build_execution_plan(md, no_vol, make_sig(Signal.BUY, composite=0.2, confidence=50), settings)
    assert plan.raw_weight == pytest.approx(0.2 * 0.5 / 0.4)  # realized vol fallback
    for s, c in ((Signal.HOLD, 0.1), (Signal.SELL, -0.3)):
        assert build_execution_plan(md, q, make_sig(s, composite=c), settings).raw_weight == 0.0


def test_action_and_triggers_korean(settings, md):
    p = md.last_price
    llm = LLMIntelligence(news=NewsSentiment(score=0.25, score_100=25, n_articles=10, n_used=6))
    md.fundamentals.next_earnings_date = (pd.Timestamp(md.as_of) + pd.Timedelta(10, "D")).date()
    plan = build_execution_plan(
        md,
        make_quant(p, 0.03 * p),
        make_sig(target=p * 1.1),
        settings,
        llm=llm,
    )
    H = settings.fusion.target_horizon_days
    assert "분할매수" in plan.action and "원" in plan.action
    assert f"(보유 {H}거래일)" in plan.action
    assert any(t.startswith("종가 기준 손절가") and "전량 청산" in t for t in plan.triggers)
    assert any("50% 차익실현 후 손절가를 진입가로 상향" in t for t in plan.triggers)
    assert time_exit_trigger(H) in plan.triggers and f"보유 {H}거래일" in plan.triggers[2]
    # 2026-10-03 is 7 weekdays after 2026-09-23 -> a neutral date line, not the D-3 caution
    assert "다음 실적발표 2026-10-03" in plan.triggers
    assert not any("D-3" in t for t in plan.triggers)
    assert any("뉴스 감성 +25점" in t for t in plan.triggers)
    md.fundamentals.next_earnings_date = None
    hold = build_execution_plan(md, make_quant(p, 0.03 * p), make_sig(Signal.HOLD, composite=0.0), settings)
    assert "신규 진입 보류" in hold.action
    assert not any("실적발표" in t for t in hold.triggers)  # no known earnings date
    assert not any("거래일 내 익절" in t for t in hold.triggers)  # no position opened -> no time exit


def test_invalid_atr_is_observable(settings, md):
    p = md.last_price
    q = make_quant(p, 0.03 * p)
    q.technicals.atr = float("nan")
    plan = build_execution_plan(md, q, make_sig(target=p * 1.1), settings)
    assert math.isfinite(plan.stop_loss) and plan.stop_loss < p
    assert any("ATR 산출 불가" in w for w in md.warnings)


# ------------------------------------------------------------------------- sizing


def test_weights_sum_le_max_gross(settings):
    settings.risk.max_weight = 0.5
    settings.risk.target_portfolio_vol = 5.0  # isolate steps 1-3
    mds = independent_mds(3)
    items = [(m, sized_plan(1.0, kelly=1.0), make_sig()) for m in mds]
    plans, cash = size_portfolio(items, settings, None, "KRW")
    w = [pl.weight for pl in plans]
    assert sum(w) == pytest.approx(settings.risk.max_gross)
    assert w == pytest.approx([settings.risk.max_gross / 3] * 3)
    assert cash == pytest.approx(1 - settings.risk.max_gross)


def test_per_name_cap_and_kelly_cap(settings):
    """max_weight caps the proportional weight of the top-conviction name (it sets k = 0.12/4 = 0.03);
    a Kelly cap clips its own name only - the others keep w = raw·k; Kelly 0 -> 0."""
    settings.risk.target_portfolio_vol = 5.0
    mds = independent_mds(5)
    items = [
        (mds[0], sized_plan(4.0), make_sig()),  # top conviction -> exactly max_weight
        (mds[1], sized_plan(2.0, kelly=0.05), make_sig()),  # raw·k = 0.06 > Kelly 0.05 -> Kelly binds
        (mds[2], sized_plan(1.0, kelly=0.0), make_sig()),  # Kelly 0 -> cap 0
        (mds[3], sized_plan(2.0), make_sig()),  # same conviction as [1], no Kelly clip
        (mds[4], sized_plan(1.0), make_sig()),
    ]
    plans, cash = size_portfolio(items, settings, None, "KRW")
    k = settings.risk.max_weight / 4.0
    assert [pl.weight for pl in plans] == pytest.approx([0.12, 0.05, 0.0, 2 * k, k])
    assert all(pl.weight <= settings.risk.max_weight for pl in plans)
    assert sum(pl.weight for pl in plans) <= settings.risk.max_gross
    assert cash == pytest.approx(1 - sum(pl.weight for pl in plans))


def test_min_weight_drop(settings):
    settings.risk.max_weight = 0.5
    settings.risk.target_portfolio_vol = 5.0
    mds = independent_mds(3)
    items = [
        (mds[0], sized_plan(1.0, kelly=1.0), make_sig()),
        (mds[1], sized_plan(1.0, kelly=1.0), make_sig()),
        (mds[2], sized_plan(0.03, kelly=1.0), make_sig()),
    ]  # ~1.4% after scaling -> dropped
    plans, cash = size_portfolio(items, settings, None, "KRW")
    assert plans[2].weight == 0.0
    assert plans[0].weight > settings.risk.min_weight
    assert cash == pytest.approx(1 - plans[0].weight - plans[1].weight)


def test_vol_targeting_scales_down_correlated_book(settings):
    settings.risk.max_weight = 0.3
    target = settings.risk.target_portfolio_vol
    mds = correlated_mds(5)  # ~48% vol each, pairwise corr ~0.97
    items = [(m, sized_plan(1.0, kelly=1.0), make_sig()) for m in mds]
    cov, n_obs, fallback = estimate_covariance(mds)
    assert not fallback and n_obs == 252
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    assert corr[np.triu_indices(5, 1)].min() > 0.9

    pre = np.full(5, 0.2 * settings.risk.max_gross)
    pre_vol = math.sqrt(pre @ cov @ pre)
    assert pre_vol > target
    plans, cash = size_portfolio(items, settings, None, "KRW")
    w = np.array([pl.weight for pl in plans])
    assert w == pytest.approx(pre * target / pre_vol)  # uniform scale-down
    assert math.sqrt(w @ cov @ w) == pytest.approx(target, rel=1e-9)
    assert portfolio_risk(items, w)["ex_ante_vol"] == pytest.approx(target, rel=1e-9)
    assert cash == pytest.approx(1 - w.sum())

    # the same gross book of independent names stays below target -> no scaling
    ind = independent_mds(5, vol=0.015)  # ~24% vol each -> book vol ~0.10 < 0.15
    plans_ind, _ = size_portfolio(
        [(m, sized_plan(1.0, kelly=1.0), make_sig()) for m in ind],
        settings,
        None,
        "KRW",
    )
    assert [pl.weight for pl in plans_ind] == pytest.approx(pre)


def test_hold_and_sell_get_zero_weight(settings):
    mds = independent_mds(3)
    items = [
        (mds[0], sized_plan(1.0), make_sig(Signal.BUY)),
        (
            mds[1],
            sized_plan(1.0),
            make_sig(Signal.HOLD),
        ),  # stale raw weight (post-plan downgrade)
        (mds[2], sized_plan(0.0), make_sig(Signal.SELL, composite=-0.4)),
    ]
    plans, cash = size_portfolio(items, settings, None, "KRW")
    assert plans[1].weight == 0.0 and plans[2].weight == 0.0
    assert plans[1].raw_weight == 0.0 and "신규 진입 보류" in plans[1].action
    # the lone Buy is the top-conviction name -> max_weight (max_gross is a budget, not a multiplier)
    assert plans[0].weight == pytest.approx(settings.risk.max_weight)
    # nothing investable -> all cash
    plans, cash = size_portfolio([(mds[1], sized_plan(1.0), make_sig(Signal.HOLD))], settings, None, "KRW")
    assert plans[0].weight == 0.0 and cash == 1.0


def test_shares_floor_with_fx_conversion(settings, md_us, md_kr):
    capital = 100_000_000.0  # KRW
    items = [
        (md_us, sized_plan(1.0, entry=md_us.last_price), make_sig()),
        (md_kr, sized_plan(1.0, entry=md_kr.last_price), make_sig()),
    ]
    plans, _ = size_portfolio(items, settings, capital, "KRW")
    for (m, _, _), pl in zip(items, plans):
        value = pl.weight * capital
        assert pl.position_value == pytest.approx(value)
        expected = math.floor(value / (m.last_price * m.fx_to_base))
        assert pl.shares == expected
        assert pl.shares * m.last_price * m.fx_to_base <= value
        assert (pl.shares + 1) * m.last_price * m.fx_to_base > value
    assert md_us.fx_to_base == 1380.0 and plans[0].shares < plans[0].position_value / md_us.last_price


def test_missing_fx_blocks_share_count(settings, md_us):
    md_us.fx_to_base = 1.0  # FX fetch failed upstream (loader default)
    plans, _ = size_portfolio(
        [(md_us, sized_plan(1.0, entry=md_us.last_price), make_sig())],
        settings,
        50_000_000.0,
        "KRW",
    )
    assert plans[0].shares is None and plans[0].position_value > 0
    assert any("환율" in w for w in md_us.warnings)


def test_no_capital_leaves_shares_empty(settings):
    mds = independent_mds(2)
    plans, _ = size_portfolio([(m, sized_plan(1.0), make_sig()) for m in mds], settings, None, "KRW")
    assert all(pl.shares is None and pl.position_value is None for pl in plans)
    with pytest.raises(ValueError):
        size_portfolio([(mds[0], sized_plan(1.0), make_sig())], settings, -5.0, "KRW")


# ------------------------------------------------------------ conviction-proportional sizing


def _vol(w: np.ndarray, cov: np.ndarray) -> float:
    return math.sqrt(max(0.0, float(w @ cov @ w)))


def test_few_buys_are_sized_by_conviction_not_flattened_to_the_cap(settings):
    """Regression: raw weights were normalised to sum 1 and THEN capped, so with few Buys every name hit
    max_weight and got the same 11.4% whatever its conviction."""
    settings.risk.target_portfolio_vol = 5.0
    raws = (3.0, 2.0, 1.0)
    old = np.minimum(np.array(raws) / sum(raws), settings.risk.max_weight) * settings.risk.max_gross
    assert np.allclose(old, old[0])  # the old rule: [0.114, 0.114, 0.114]
    items = [(m, sized_plan(r), make_sig()) for m, r in zip(independent_mds(3), raws)]
    plans, cash = size_portfolio(items, settings, None, "KRW")
    w = [pl.weight for pl in plans]
    assert w == pytest.approx([0.12, 0.08, 0.04])  # w = raw·k, k = max_weight / max(raw)
    assert w[0] > w[1] > w[2]
    assert cash == pytest.approx(1 - sum(w))


def test_kelly_capped_name_stays_capped_and_freed_budget_flows_proportionally(settings):
    """12 names: the gross budget binds. A Kelly-capped name stays at its Kelly cap; the budget it frees
    raises the common k for the others, which stay proportional to their conviction."""
    settings.risk.target_portfolio_vol = 5.0
    G = settings.risk.max_gross
    mds = independent_mds(12)
    raws = [1.2] + [1.0] * 10 + [0.5]

    def book(kelly0: float) -> np.ndarray:
        items = [
            (m, sized_plan(r, kelly=kelly0 if i == 0 else 0.25), make_sig()) for i, (m, r) in enumerate(zip(mds, raws))
        ]
        return np.array([pl.weight for pl in size_portfolio(items, settings, None, "KRW")[0]])

    free = book(0.25)  # no Kelly clip: pure proportional, gross-bound
    k = G / sum(raws)
    assert free == pytest.approx(np.array(raws) * k) and free.sum() == pytest.approx(G)
    clipped = book(0.03)  # name 0 clipped at 3%
    k2 = (G - 0.03) / sum(raws[1:])
    assert clipped[0] == pytest.approx(0.03)
    assert clipped[1:] == pytest.approx(np.array(raws[1:]) * k2) and k2 > k
    assert clipped.sum() == pytest.approx(G)
    assert clipped[1] / clipped[-1] == pytest.approx(raws[1] / raws[-1])  # proportionality kept


def test_min_weight_drops_the_smallest_tier_then_resolves(settings):
    """Dropping every sub-minimum name at once would also drop C, which clears min_weight once D's budget
    is re-solved: the rule drops the smallest position(s) and re-solves k."""
    settings.risk.max_weight = 0.5
    settings.risk.target_portfolio_vol = 5.0
    G, mw = settings.risk.max_gross, settings.risk.min_weight
    raws = (1.0, 1.0, 0.0435, 0.04)
    k1 = G / sum(raws)
    assert raws[2] * k1 < mw and raws[3] * k1 < mw  # both below the minimum in round 1
    items = [(m, sized_plan(r, kelly=1.0), make_sig()) for m, r in zip(independent_mds(4), raws)]
    plans, cash = size_portfolio(items, settings, None, "KRW")
    k2 = G / sum(raws[:3])
    assert [pl.weight for pl in plans] == pytest.approx([k2, k2, raws[2] * k2, 0.0])
    assert plans[2].weight >= mw and "분할매수" in plans[2].action
    assert plans[3].weight == 0.0 and "최소 비중" in plans[3].action
    assert "총투자 한도 기준 배율" in plans[3].action  # the constraint that set k in D's round
    assert cash == pytest.approx(1 - G)


def test_kelly_cap_below_min_weight_says_so(settings):
    settings.risk.target_portfolio_vol = 5.0
    mds = independent_mds(2)
    items = [(mds[0], sized_plan(1.0), make_sig()), (mds[1], sized_plan(2.0, kelly=0.01), make_sig())]
    plans, _ = size_portfolio(items, settings, None, "KRW")
    assert plans[1].weight == 0.0 and "켈리 비중 상한 1.0%" in plans[1].action
    assert "최소 비중" in plans[1].action and "분할매수" not in plans[1].action
    # a name that can never be held does not anchor the scale: the other Buy is the top conviction
    assert plans[0].weight == pytest.approx(settings.risk.max_weight)


def test_conviction_scale_takes_the_largest_feasible_k_under_a_hedge():
    """Vol is NOT monotone in k: A caps at 10% (Kelly) and the growing hedge B then LOWERS book vol. The
    largest feasible k lies past a region where vol exceeds the target; a first-crossing (or bisection
    on a monotone assumption) would stop at k ≈ 0.053 with a tiny book."""
    raw = np.array([1.0, 0.2])
    caps = np.array([0.1, 0.5])
    cov = np.array([[0.09, -0.081], [-0.081, 0.09]])  # 30% vols, correlation -0.9
    T = 0.0132
    k, binding = conviction_scale(raw, caps, cov, max_weight=0.5, max_gross=0.95, target_vol=T)
    # second piece (A capped): 0.0036k² − 0.00324k + 0.0009 = T² -> larger root
    a, b, c = 0.0036, -0.00324, 0.0009 - T * T
    root = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    assert binding == "target_vol" and k == pytest.approx(root, rel=1e-9)
    w = np.minimum(raw * k, caps)
    assert _vol(w, cov) == pytest.approx(T, rel=1e-9)
    first_crossing = T / _vol(raw, cov)
    assert first_crossing < 0.1 < k  # the vol path crosses T well before the answer
    for kk in np.linspace(k, 0.5, 300)[1:]:  # nothing larger is feasible
        assert _vol(np.minimum(raw * kk, caps), cov) > T
    # with a looser target the whole path up to the max_weight ceiling (k = 0.5) is admissible
    k, binding = conviction_scale(raw, caps, cov, max_weight=0.5, max_gross=0.95, target_vol=0.02)
    assert (k, binding) == (pytest.approx(0.5), "max_weight")


def test_property_conviction_scale_is_feasible_and_largest():
    """Random books incl. negative correlations: every constraint holds at k, the binding label is
    right, and no larger k (up to the max_weight ceiling) is strictly feasible."""
    rng = np.random.default_rng(99)
    for _ in range(250):
        n = int(rng.integers(1, 7))
        raw = rng.uniform(0.05, 3.0, n)
        mw = float(rng.uniform(0.05, 0.6))
        caps = np.minimum(mw, rng.uniform(0.01, 0.4, n))
        A = rng.normal(0, 0.35, (n, n + 1))
        cov = A @ A.T / (n + 1)
        G, T = float(rng.uniform(0.1, 1.0)), float(rng.uniform(0.01, 0.4))
        k, binding = conviction_scale(raw, caps, cov, max_weight=mw, max_gross=G, target_vol=T)
        w = np.minimum(raw * k, caps)
        assert k >= 0 and np.all(raw * k <= mw * (1 + 1e-12)) and np.all(w <= caps)
        assert w.sum() <= G * (1 + 1e-12) and _vol(w, cov) <= T * (1 + 1e-9)
        k_top = mw / raw.max()
        if binding == "max_weight":
            assert k == pytest.approx(k_top)
        elif binding == "max_gross":
            assert w.sum() == pytest.approx(G, rel=1e-9)
        else:
            assert binding == "target_vol" and _vol(w, cov) == pytest.approx(T, rel=1e-6)
        if k_top > k * (1 + 1e-9):
            for kk in np.linspace(k, k_top, 200)[1:]:
                ww = np.minimum(raw * kk, caps)
                strictly_feasible = ww.sum() < G * (1 - 1e-9) and _vol(ww, cov) < T * (1 - 1e-9)
                assert not strictly_feasible or np.allclose(ww, w)


def test_property_sizing_preserves_conviction_ordering(settings):
    """All else equal (same Kelly cap), a higher-conviction name never gets a lower weight - through the
    vol target, hedges, the gross budget and min-weight drops. Weights are invariant to a common
    rescaling of the raw weights (only relative conviction matters)."""
    rng = np.random.default_rng(31)
    checked = 0
    for _ in range(50):
        settings.risk.max_weight = float(rng.uniform(0.05, 0.6))
        settings.risk.min_weight = float(rng.uniform(0.0, 0.05))
        settings.risk.target_portfolio_vol = float(rng.uniform(0.05, 0.3))
        n = int(rng.integers(2, 8))
        f = rng.normal(0, 1, 300)
        mds = [
            md_from_close(
                100 * np.exp(np.cumsum(rng.uniform(-0.03, 0.03) * f + rng.uniform(0.003, 0.03) * rng.normal(0, 1, 300)))
            )
            for _ in range(n)
        ]
        kelly = rng.choice([0.03, 0.08, 0.25], n)
        raw = rng.uniform(0.01, 3.0, n)
        sigs = [make_sig(Signal.BUY if rng.random() < 0.85 else Signal.HOLD) for _ in range(n)]

        def weights(scale: float) -> np.ndarray:
            items = [
                (m, sized_plan(float(r * scale), kelly=float(kf)), s) for m, r, kf, s in zip(mds, raw, kelly, sigs)
            ]
            return np.array([pl.weight for pl in size_portfolio(items, settings, None, "KRW")[0]])

        w = weights(1.0)
        for i in range(n):
            for j in range(n):
                buy_ij = sigs[i].signal == Signal.BUY and sigs[j].signal == Signal.BUY
                if buy_ij and kelly[i] == kelly[j] and raw[i] > raw[j]:
                    assert w[i] >= w[j] - 1e-12
                    checked += 1
        assert weights(float(rng.uniform(0.1, 10.0))) == pytest.approx(w, abs=1e-12)
    assert checked > 50


def test_conviction_weights_rejects_broken_risk_config(settings):
    settings.risk.target_portfolio_vol = float("nan")
    items = [(m, sized_plan(1.0), make_sig()) for m in independent_mds(2)]
    with pytest.raises(ValueError, match="target_portfolio_vol"):
        size_portfolio(items, settings, None, "KRW")
    settings.risk.target_portfolio_vol = 0.15
    settings.risk.max_gross = -0.1
    with pytest.raises(ValueError, match="max_gross"):
        conviction_weights([1.0], [0.1], lambda idx: np.eye(len(idx)) * 0.04, settings.risk)


def test_zero_vol_budget_leaves_all_cash_and_says_why(settings):
    settings.risk.target_portfolio_vol = 0.0
    items = [(m, sized_plan(1.0), make_sig()) for m in independent_mds(2)]
    plans, cash = size_portfolio(items, settings, None, "KRW")
    assert cash == 1.0 and all(pl.weight == 0.0 for pl in plans)
    assert all("목표 포트폴리오 변동성" in pl.action and "분할매수" not in pl.action for pl in plans)


# ------------------------------------------------------------------ covariance / risk


def test_covariance_fallback_is_conservative_and_observable(settings):
    short = [
        md_from_close(100 * np.exp(np.cumsum(np.full(30, 0.01) * (-1) ** np.arange(30)))),
        md_from_close(100 * np.exp(np.cumsum(np.full(30, 0.02) * (-1) ** np.arange(30)))),
    ]
    cov, n_obs, fallback = estimate_covariance(short)
    assert fallback and n_obs == 29
    corr = cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1])
    assert corr == pytest.approx(1.0)  # rho = 1 assumed
    items = [(m, sized_plan(1.0), make_sig()) for m in short]
    size_portfolio(items, settings, None, "KRW")
    assert all(any("공분산 표본 부족" in w for w in m.warnings) for m in short)


def test_portfolio_risk_normal_approximation():
    mds = independent_mds(2)
    barrier = BarrierOutcome(
        p_take_profit=0.4,
        p_stop_loss=0.3,
        p_time_exit=0.3,
        expected_trade_return=0.01,
        kelly_optimal=0.3,
        n_paths=100,
    )
    items = [
        (mds[0], sized_plan(1.0).model_copy(update={"barrier": barrier}), make_sig()),
        (mds[1], sized_plan(1.0), make_sig(upside=0.12)),
    ]
    w = [0.1, 0.05]
    out = portfolio_risk(items, w, horizon_days=20)
    cov, _, _ = estimate_covariance(mds)
    vol = math.sqrt(np.array(w) @ cov @ np.array(w))
    mu = 0.1 * 0.01 + 0.05 * ((1.12) ** (20 / 60) - 1)
    sigma = vol * math.sqrt(20 / 252)
    assert out["ex_ante_vol"] == pytest.approx(vol)
    assert out["expected_return"] == pytest.approx(mu)
    assert out["var95"] == pytest.approx(mu - 1.6448536269514722 * sigma)
    assert out["var95"] < 0
    # normal expected shortfall: mu - sigma * phi(z95) / 0.05
    phi = math.exp(-0.5 * 1.6448536269514722**2) / math.sqrt(2 * math.pi)
    assert out["cvar95"] == pytest.approx(mu - sigma * phi / 0.05)
    assert out["cvar95"] < out["var95"]
    assert out["p_loss"] == pytest.approx(0.5 * (1 + math.erf(-mu / sigma / math.sqrt(2))))
    assert out["gross"] == pytest.approx(0.15) and out["n_positions"] == 2.0
    # defaults to the plans' own weights; empty book -> zeros
    zero = portfolio_risk(items)
    assert zero["ex_ante_vol"] == 0.0 and zero["gross"] == 0.0 and zero["n_positions"] == 0.0
    with pytest.raises(ValueError):
        portfolio_risk(items, [0.1])


def test_portfolio_risk_keys_match_renderer():
    """Regression: report/render.py (which never imports quantcio.fusion) reads these exact keys for the
    risk tiles. The old *_normal_approx / ex_ante_vol_annual keys rendered the tiles as blanks and
    dumped raw English keys into the Korean risk line."""
    from quantcio.fusion.risk import PORTFOLIO_RISK_LABELS

    render = pytest.importorskip("quantcio.report.render")
    if hasattr(render, "LOCAL_PORTFOLIO_RISK_LABELS"):
        # the report's own fallback labels must cover every key even when fusion cannot be imported
        render_keys = set(render.LOCAL_PORTFOLIO_RISK_LABELS)
        assert set(PORTFOLIO_RISK_LABELS) <= set(render.portfolio_risk_labels())
    else:
        render_keys = set(render.PORTFOLIO_RISK_LABELS)

    mds = independent_mds(2)
    items = [(m, sized_plan(1.0).model_copy(update={"weight": 0.1}), make_sig()) for m in mds]
    out = portfolio_risk(items, horizon_days=20)
    assert set(out) == set(PORTFOLIO_RISK_LABELS)
    assert set(out) <= set(render_keys)  # nothing lands in the raw "extra" list
    assert {"ex_ante_vol", "var95", "cvar95", "p_loss", "expected_return"} <= set(out)


# --------------------------------------------------------------- adversarial regressions


@pytest.mark.parametrize("bad_vol", [float("nan"), 0.0, -0.1])
def test_unusable_vol_is_conservative_and_observable(settings, md, bad_vol):
    """Regression: a NaN/zero realized vol (no GARCH vol) used to size like the 10% floor -
    the LARGEST possible raw weight - with no warning."""
    p = md.last_price
    q = make_quant(p, 0.03 * p, vol=None, rv=bad_vol)
    plan = build_execution_plan(md, q, make_sig(Signal.BUY, composite=0.3, confidence=60), settings)
    assert plan.raw_weight == pytest.approx(0.3 * 0.60 / 0.50)  # UNKNOWN_VOL, not the floor
    assert plan.raw_weight < 0.3 * 0.60 / 0.10
    assert any("변동성 산출 불가" in w for w in md.warnings)
    # an invalid GARCH vol with a valid realized vol falls back silently-by-spec to realized vol
    md.warnings.clear()
    q2 = make_quant(p, 0.03 * p, vol=float("nan"), rv=0.4)
    plan2 = build_execution_plan(md, q2, make_sig(Signal.BUY, composite=0.3, confidence=60), settings)
    assert plan2.raw_weight == pytest.approx(0.3 * 0.60 / 0.4)
    assert not any("변동성 산출 불가" in w for w in md.warnings)


@pytest.mark.parametrize(("atr_frac", "support_frac"), [(0.35, None), (0.20, 0.91), (0.275, 0.91)])
def test_entry_zone_stays_above_stop_under_extreme_atr(settings, md, atr_frac, support_frac):
    """Regression: with ATR > ~15% of price (support snap) or > ~30% (no snap) the entry zone floor
    entry - 0.5·ATR fell BELOW the stop, i.e. the plan said to buy under its own stop-loss."""
    p = md.last_price
    support = support_frac * p if support_frac else None
    plan = build_execution_plan(md, make_quant(p, atr_frac * p, support), make_sig(target=p * 1.3), settings)
    assert plan.stop_loss < plan.entry_low < plan.entry_price < plan.entry_high
    assert plan.entry_low == pytest.approx(plan.stop_loss + 0.5 * (p - plan.stop_loss))
    # normal ATR keeps the spec zone exactly
    plan = build_execution_plan(md, make_quant(p, 0.03 * p), make_sig(target=p * 1.1), settings)
    assert plan.entry_low == pytest.approx(p - 0.5 * 0.03 * p)


def test_vol_target_holds_after_min_weight_drop(settings):
    """Regression: step 5 (min-weight drop) runs after vol targeting. Dropping a small, negatively
    correlated diversifier raised book vol ABOVE the target (and the renderer flags vol > target)."""
    settings.risk.max_weight = 1.0  # cap never binds: isolate steps 4-5
    settings.risk.min_weight = 0.02
    settings.risk.target_portfolio_vol = 0.10
    rng = np.random.default_rng(11)
    f = rng.normal(0, 1, 400)
    a = md_from_close(100 * np.exp(np.cumsum(0.02 * f + 0.004 * rng.normal(0, 1, 400))))
    b = md_from_close(100 * np.exp(np.cumsum(-0.02 * f + 0.004 * rng.normal(0, 1, 400))))  # hedge-like
    items = [
        (a, sized_plan(1.0, kelly=1.0), make_sig()),
        (b, sized_plan(0.05, kelly=1.0), make_sig()),
    ]

    # what the one-pass spec order produces: scale to target on {a, b}, then b falls under min_weight
    w0 = np.array([1.0, 0.05]) / 1.05 * settings.risk.max_gross
    cov_ab, _, _ = estimate_covariance([a, b])
    scaled = w0 * min(1.0, settings.risk.target_portfolio_vol / math.sqrt(w0 @ cov_ab @ w0))
    assert 0 < scaled[1] < settings.risk.min_weight
    cov_a, _, _ = estimate_covariance([a])
    assert math.sqrt(scaled[0] ** 2 * cov_a[0, 0]) > settings.risk.target_portfolio_vol * 1.01  # the old bug

    plans, cash = size_portfolio(items, settings, None, "KRW")
    w = np.array([pl.weight for pl in plans])
    assert w[1] == 0.0 and w[0] > 0
    assert portfolio_risk(items, w)["ex_ante_vol"] <= settings.risk.target_portfolio_vol * (1 + 1e-9)
    assert cash == pytest.approx(1 - w.sum())


def test_action_refreshed_after_strong_buy_downgrade(settings, md):
    """Regression: a Strong Buy plan downgraded to Buy by the barrier guardrail kept '2회 분할매수'."""
    p = md.last_price
    plan = build_execution_plan(
        md,
        make_quant(p, 0.03 * p),
        make_sig(Signal.STRONG_BUY, composite=0.5, confidence=80, target=p * 1.2),
        settings,
    )
    assert "2회 분할매수" in plan.action
    plans, _ = size_portfolio(
        [(md, plan, make_sig(Signal.BUY, composite=0.5, confidence=80))],
        settings,
        None,
        "KRW",
    )
    assert "2회" not in plans[0].action and "분할매수" in plans[0].action
    assert plans[0].weight > 0


def test_numpy_capital_and_fx_are_accepted(settings, md_us):
    """Regression: np.int64 capital raised ValueError and an np.float32 FX was reported as '환율 값 이상'
    with shares=None, because np.float32/np.int64 are not float/int subclasses."""
    md_us.fx_to_base = np.float32(1380.0)
    items = [(md_us, sized_plan(1.0, entry=md_us.last_price), make_sig())]
    plans, _ = size_portfolio(items, settings, np.int64(100_000_000), "KRW")
    assert plans[0].shares == math.floor(plans[0].weight * 1e8 / (md_us.last_price * 1380.0))
    assert not any("환율" in w for w in md_us.warnings)
    with pytest.raises(ValueError):
        size_portfolio(items, settings, np.float64("nan"), "KRW")


def test_kr_name_in_usd_base_book(settings, md_kr):
    """KRW stock in a USD book: fx < 1, shares = floor(value_usd / (price_krw · fx))."""
    md_kr.fx_to_base = 1 / 1380.0
    plans, _ = size_portfolio(
        [(md_kr, sized_plan(1.0, entry=md_kr.last_price), make_sig())],
        settings,
        100_000.0,
        "USD",
    )
    pl = plans[0]
    assert pl.position_value == pytest.approx(pl.weight * 100_000.0)
    assert pl.shares == math.floor(pl.position_value / (md_kr.last_price / 1380.0))
    assert pl.shares * md_kr.last_price / 1380.0 <= pl.position_value


def test_barrier_from_real_return_distribution(settings, md):
    """Integration with quant.distribution.ReturnDistribution (the object is passed in, not rebuilt)."""
    rd_mod = pytest.importorskip("quantcio.quant.distribution")
    dist = rd_mod.ReturnDistribution(0.01, 0.07, 0.005, 0.04, 6.0, 8.0, 1.0, 0.1, horizon=20, n_paths=4000, seed=3)
    # the simulated horizon (BarrierOutcome.horizon_days; older distributions: the path length)
    hb = dist.barrier(-0.05, 0.05).horizon_days or dist.horizon
    settings.fusion.target_horizon_days = hb  # holding horizon == simulated horizon -> no warning
    p = md.last_price
    plan = build_execution_plan(
        md,
        make_quant(p, 0.03 * p, with_dist=True),
        make_sig(target=p * 1.1),
        settings,
        dist=dist,
    )
    b = plan.barrier
    assert b is not None and b.p_take_profit + b.p_stop_loss + b.p_time_exit == 1.0
    raw = dist.barrier(plan.stop_loss_pct, plan.take_profit_pct)  # same path matrix
    assert b.model_dump(exclude={"horizon_days"}) == raw.model_dump(exclude={"horizon_days"})
    assert b.horizon_days == hb > 0
    assert plan.kelly_fraction == pytest.approx(b.kelly_optimal * settings.risk.kelly_cap)
    assert 0 <= plan.kelly_fraction <= settings.risk.kelly_cap
    assert not any("배리어" in w for w in md.warnings)


@pytest.mark.parametrize("kind", ["datetime", "timestamp", "timestamp_tz"])
def test_earnings_date_accepts_datetime_and_timestamp(settings, md, kind):
    """Regression: datetime / pd.Timestamp pass isinstance(x, date) but raise TypeError when compared
    with md.as_of (a date), which crashed build_execution_plan for the whole ticker."""
    import datetime as dt

    p = md.last_price
    d = md.as_of + dt.timedelta(days=5)
    ne = {
        "datetime": dt.datetime.combine(d, dt.time(9, 0)),
        "timestamp": pd.Timestamp(d),
        "timestamp_tz": pd.Timestamp(d, tz="UTC"),
    }[kind]
    md.fundamentals.next_earnings_date = ne
    plan = build_execution_plan(
        md,
        make_quant(p, 0.03 * p),
        make_sig(target=p * 1.1),
        settings,
    )
    # 2026-09-28 is 3 weekdays after 2026-09-23 -> caution trigger
    assert any(
        t.startswith("실적발표 D-3 이내 신규진입 자제") and f"발표 {d:%Y-%m-%d}, 3거래일 후" in t for t in plan.triggers
    )


def test_earnings_date_nat_and_unknown_type(settings, md):
    p = md.last_price
    md.fundamentals.next_earnings_date = pd.NaT
    plan = build_execution_plan(
        md,
        make_quant(p, 0.03 * p),
        make_sig(target=p * 1.1),
        settings,
    )
    assert not any("실적발표" in t for t in plan.triggers) and md.warnings == []
    md.fundamentals.next_earnings_date = "2026-10-01"  # wrong type: observable, no crash
    plan = build_execution_plan(
        md,
        make_quant(p, 0.03 * p),
        make_sig(target=p * 1.1),
        settings,
    )
    assert not any("실적발표" in t for t in plan.triggers)
    assert any("실적발표일 형식 인식 불가" in w for w in md.warnings)


def test_guardrail_hold_and_sell_triggers_name_the_guardrail(settings, md):
    """Regression: a Hold forced by a guardrail with composite +0.50 said "종합점수 +0.15 이상 회복 시
    신규 진입 검토" (already met), and a critical-flag Sell at -0.05 said "-0.15 상향 회복 전까지"."""
    p = md.last_price
    hold = build_execution_plan(md, make_quant(p, 0.03 * p), make_sig(Signal.HOLD, composite=0.5), settings)
    assert "가드레일" in hold.triggers[0] and "+0.15 이상 회복" not in hold.triggers[0]
    sell = build_execution_plan(md, make_quant(p, 0.03 * p), make_sig(Signal.SELL, composite=-0.05), settings)
    assert "치명 리스크" in sell.triggers[0]
    # score-driven Hold / Sell keep the score-level trigger
    hold = build_execution_plan(md, make_quant(p, 0.03 * p), make_sig(Signal.HOLD, composite=0.05), settings)
    assert "+0.15 이상 회복 시 신규 진입 검토" in hold.triggers[0]
    sell = build_execution_plan(md, make_quant(p, 0.03 * p), make_sig(Signal.SELL, composite=-0.4), settings)
    assert "-0.15 상향 회복 전까지" in sell.triggers[0]
    # a Hold at a Sell-level composite is a Sell lifted by the target-gap guardrail: say when it returns
    lifted = build_execution_plan(
        md, make_quant(p, 0.03 * p), make_sig(Signal.HOLD, composite=-0.4, upside=0.12), settings
    )
    assert "목표가 상승여력 상충" in lifted.triggers[0] and "+5.0% 이하로 하향 시" in lifted.triggers[0]
    assert "+0.15 이상 회복" not in lifted.triggers[0] and lifted.raw_weight == 0.0


def test_triggers_refreshed_after_post_plan_downgrade(settings, md):
    """Regression: a Buy downgraded to Hold by the barrier guardrail (after the plan was built) kept the
    Buy's lead triggers, e.g. "익절가 도달 시 50% 차익실현", while the non-signal triggers must survive."""
    p = md.last_price
    llm = LLMIntelligence(news=NewsSentiment(score=0.25, score_100=25, n_articles=10, n_used=6))
    plan = build_execution_plan(
        md,
        make_quant(p, 0.03 * p),
        make_sig(Signal.BUY, composite=0.3, target=p * 1.1),
        settings,
        llm=llm,
    )
    plans, _ = size_portfolio([(md, plan, make_sig(Signal.HOLD, composite=0.3))], settings, None, "KRW")
    tr = plans[0].triggers
    assert "가드레일" in tr[0] and not any("50% 차익실현" in t for t in tr)
    assert not any("거래일 내 익절·손절" in t for t in tr)  # Buy-only time-exit rule dropped
    assert any("뉴스 감성 +25점" in t for t in tr) and len(tr) == len(plan.triggers) - 1
    # an unchanged Buy keeps exactly its triggers
    plans, _ = size_portfolio([(md, plan, make_sig(Signal.BUY, composite=0.3))], settings, None, "KRW")
    assert plans[0].triggers == plan.triggers


def test_buy_sized_to_zero_says_why(settings):
    """Regression: a Buy whose weight ended at 0 (Kelly 0 from the path matrix, or below min_weight)
    kept the action "... 구간 분할매수", an instruction to buy next to a 0% weight."""
    settings.risk.target_portfolio_vol = 5.0
    settings.risk.max_weight = 0.5
    mds = independent_mds(3)
    items = [
        (mds[0], sized_plan(1.0, kelly=1.0), make_sig()),
        (mds[1], sized_plan(1.0, kelly=0.0), make_sig()),  # Kelly 0
        (mds[2], sized_plan(0.01, kelly=1.0), make_sig()),
    ]  # below min_weight
    plans, _ = size_portfolio(items, settings, None, "KRW")
    assert plans[0].weight > 0 and "분할매수" in plans[0].action
    assert plans[1].weight == 0 and "켈리 비중 0" in plans[1].action and "분할매수" not in plans[1].action
    assert plans[2].weight == 0 and "최소 비중" in plans[2].action and "분할매수" not in plans[2].action


def test_same_currency_fx_mismatch_is_corrected(settings, md_us):
    """Regression: a USD name in a USD book with fx_to_base 1380 (md loaded for a KRW base) got
    floor(value / (price·1380)) = 0 shares instead of ~215, with no warning."""
    plans, _ = size_portfolio(
        [(md_us, sized_plan(1.0, entry=md_us.last_price), make_sig())],
        settings,
        100_000.0,
        "USD",
    )
    pl = plans[0]
    assert pl.shares == math.floor(pl.position_value / md_us.last_price) and pl.shares > 0
    assert any("동일 통화" in w for w in md_us.warnings)


# ------------------------------------------------------ money format / earnings / horizon


@pytest.mark.parametrize(
    ("value", "ccy", "expected"),
    [
        (286500, "KRW", "286,500원"),
        (286500.4, "KRW", "286,500원"),  # no decimals for KRW
        (71_200.0, "krw", "71,200원"),
        (-1500, "KRW", "-1,500원"),
        (123.45, "USD", "$123.45"),
        (1234.5, "USD", "$1,234.50"),
        (-12.3, "USD", "-$12.30"),  # sign before the symbol
        (-0.001, "USD", "$0.00"),  # rounds to zero: no sign
        (-0.4, "KRW", "0원"),
        (1234.5, "EUR", "1,234.50 EUR"),
        (float("nan"), "KRW", "—"),
        (None, "USD", "—"),
        (float("inf"), "USD", "—"),
    ],
)
def test_format_money(value, ccy, expected):
    assert format_money(value, ccy) == expected


@pytest.mark.parametrize("market", [Market.KR, Market.US])
def test_every_amount_in_plan_text_uses_format_money(settings, market):
    """KRW '286,500원' / USD '$123.45' everywhere: action, triggers and the zero-weight action."""
    md = make_market_data(market)
    ccy = md.instrument.currency
    p = md.last_price
    plan = build_execution_plan(md, make_quant(p, 0.03 * p), make_sig(target=p * 1.1), settings)
    stop, tp = format_money(plan.stop_loss, ccy), format_money(plan.take_profit, ccy)
    assert stop in plan.action and format_money(plan.entry_low, ccy) in plan.action
    assert stop in plan.triggers[0] and tp in plan.triggers[1]
    text = " ".join([plan.action, *plan.triggers])
    if ccy == "KRW":
        assert "원" in text and "$" not in text and "₩" not in text
        assert "." not in stop  # no decimals
    else:
        assert "$" in text and "원" not in text
        assert stop.startswith("$") and stop.split(".")[-1].isdigit() and len(stop.split(".")[-1]) == 2
    # a Buy sized to zero states the stop in the same format
    plans, _ = size_portfolio(
        [(md, plan.model_copy(update={"kelly_fraction": 0.0}), make_sig())], settings, None, "KRW"
    )
    assert plans[0].weight == 0 and stop in plans[0].action and "켈리 비중 0" in plans[0].action


def test_trading_days_until():
    wed = pd.Timestamp("2026-09-23").date()
    assert trading_days_until(wed, wed) == 0
    assert trading_days_until(wed, pd.Timestamp("2026-09-24").date()) == 1
    assert trading_days_until(wed, pd.Timestamp("2026-09-28").date()) == 3  # skips the weekend
    assert trading_days_until(wed, pd.Timestamp("2026-09-30").date()) == 5  # 7 calendar days
    assert trading_days_until(wed, pd.Timestamp("2026-10-01").date()) == 6


@pytest.mark.parametrize(
    ("days", "estimate", "expected"),
    [
        (0, False, "실적발표 D-3 이내 신규진입 자제 — 발표 2026-09-23, 당일"),
        (5, False, "실적발표 D-3 이내 신규진입 자제 — 발표 2026-09-28, 3거래일 후"),
        (7, True, "실적발표 D-3 이내 신규진입 자제 — 발표 2026-09-30 (예정일 추정), 5거래일 후"),
        (8, False, "다음 실적발표 2026-10-01"),  # 6 weekdays: neutral
        (30, True, "다음 실적발표 2026-10-23 (예정일 추정)"),
        (30, None, "다음 실적발표 2026-10-23"),
    ],
)
def test_earnings_trigger_window_and_estimate_flag(md, days, estimate, expected):
    """'D-3 이내 신규진입 자제' only within the next 5 trading days (~7 calendar days); otherwise a
    neutral date line. An estimated date says '(예정일 추정)'."""
    assert md.as_of == pd.Timestamp("2026-09-23").date()
    md.fundamentals.next_earnings_date = (pd.Timestamp(md.as_of) + pd.Timedelta(days, "D")).date()
    md.fundamentals.next_earnings_is_estimate = estimate
    assert earnings_trigger(md) == expected
    assert md.warnings == []


def test_stale_earnings_date_is_observable(settings, md):
    md.fundamentals.next_earnings_date = (pd.Timestamp(md.as_of) - pd.Timedelta(3, "D")).date()
    p = md.last_price
    plan = build_execution_plan(md, make_quant(p, 0.03 * p), make_sig(target=p * 1.1), settings)
    assert not any("실적발표" in t for t in plan.triggers)
    assert any("실적 일정 미갱신" in w for w in md.warnings)


def test_barrier_horizon_is_checked_against_holding_horizon(settings, md):
    """The barrier must simulate the holding horizon (target_horizon_days): a shorter simulation
    understates P(TP first) for a take-profit set from the 60-day target. Never silent."""
    p = md.last_price
    H = settings.fusion.target_horizon_days
    q = make_quant(p, 0.03 * p, with_dist=True)

    def outcome(h: int) -> BarrierOutcome:
        return BarrierOutcome(
            p_take_profit=0.3,
            p_stop_loss=0.2,
            p_time_exit=0.5,
            expected_trade_return=0.01,
            kelly_optimal=0.4,
            n_paths=100,
            horizon_days=h,
        )

    plan = build_execution_plan(md, q, make_sig(target=p * 1.1), settings, dist=FakeDist(outcome(H)))
    assert plan.barrier.horizon_days == H and md.warnings == []

    plan = build_execution_plan(md, q, make_sig(target=p * 1.1), settings, dist=FakeDist(outcome(20)))
    assert plan.barrier.horizon_days == 20
    assert any(f"20거래일 시뮬레이션 기준 — 보유기간 {H}거래일과 불일치" in w for w in md.warnings)

    # an outcome without the field: the path length of the distribution is recorded ...
    md.warnings.clear()
    plan = build_execution_plan(md, q, make_sig(target=p * 1.1), settings, dist=FakeDist(outcome(0), horizon=H))
    assert plan.barrier.horizon_days == H and md.warnings == []
    # ... and when even that is unknown the plan says so
    plan = build_execution_plan(md, q, make_sig(target=p * 1.1), settings, dist=FakeDist(outcome(0)))
    assert plan.barrier is not None and plan.barrier.horizon_days == 0
    assert any("배리어 시뮬레이션 기간 미기재" in w for w in md.warnings)


def test_barrier_from_real_distribution_covers_holding_horizon(settings, md):
    """quant.distribution.ReturnDistribution(hold_days=60): the barrier walks the 60-day holding horizon
    by default, the plan records it, and nothing is warned; the plan text states the same horizon."""
    rd_mod = pytest.importorskip("quantcio.quant.distribution")
    import inspect

    if "hold_days" not in inspect.signature(rd_mod.ReturnDistribution).parameters:
        pytest.skip("ReturnDistribution without hold_days")
    H = settings.fusion.target_horizon_days
    dist = rd_mod.ReturnDistribution(
        0.01, 0.07, 0.005, 0.04, 6.0, 8.0, 1.0, 0.1, horizon=20, n_paths=4000, seed=3, hold_days=H
    )
    p = md.last_price
    plan = build_execution_plan(
        md, make_quant(p, 0.03 * p, with_dist=True), make_sig(target=p * 1.1), settings, dist=dist
    )
    assert plan.barrier.horizon_days == H and md.warnings == []
    assert plan.barrier == dist.barrier(plan.stop_loss_pct, plan.take_profit_pct)
    assert f"(보유 {H}거래일)" in plan.action and time_exit_trigger(H) in plan.triggers
    # a legacy-style outcome (no horizon recorded) takes the object's hold_days, not its model horizon
    legacy = FakeDist(plan.barrier.model_copy(update={"horizon_days": 0}), horizon=20)
    legacy.hold_days = H
    plan2 = build_execution_plan(
        md, make_quant(p, 0.03 * p, with_dist=True), make_sig(target=p * 1.1), settings, dist=legacy
    )
    assert plan2.barrier.horizon_days == H and md.warnings == []


def test_portfolio_risk_rescales_barrier_return_to_risk_horizon():
    """Regression: the barrier's expected trade return now covers the 60-day hold; adding it unscaled to
    a 20-day sigma overstated the portfolio's expected return (and understated VaR / P(loss)) ~3x."""
    mds = independent_mds(1)
    b60 = BarrierOutcome(
        p_take_profit=0.4,
        p_stop_loss=0.3,
        p_time_exit=0.3,
        expected_trade_return=0.03,
        kelly_optimal=0.3,
        n_paths=100,
        horizon_days=60,
    )
    items = [(mds[0], sized_plan(1.0).model_copy(update={"barrier": b60}), make_sig())]
    out = portfolio_risk(items, [0.1], horizon_days=20)
    assert out["expected_return"] == pytest.approx(0.1 * (1.03 ** (20 / 60) - 1))
    same = b60.model_copy(update={"horizon_days": 20})
    items = [(mds[0], sized_plan(1.0).model_copy(update={"barrier": same}), make_sig())]
    assert portfolio_risk(items, [0.1], horizon_days=20)["expected_return"] == pytest.approx(0.003)


# -------------------------------------------------------------------- property tests


def test_property_execution_plan_invariants(settings):
    """Random prices/ATR/support/targets/signals -> sign conventions and rules always hold."""
    rng = np.random.default_rng(2026)
    md = make_market_data(Market.KR)
    p = md.last_price
    cfg = settings.risk
    for _ in range(400):
        atr = p * float(rng.uniform(0.001, 0.45))
        support = p * float(rng.uniform(0.5, 1.05)) if rng.random() < 0.8 else None
        signal = Signal(rng.choice([s.value for s in Signal]))
        target = p * float(rng.uniform(0.4, 2.2))
        md.fundamentals.analyst_target_high = p * float(rng.uniform(0.5, 2.5)) if rng.random() < 0.7 else None
        md.warnings.clear()
        q = make_quant(
            p,
            atr,
            support,
            vol=float(rng.uniform(0.02, 1.2)) if rng.random() < 0.7 else None,
            rv=float(rng.uniform(0.02, 1.2)),
        )
        sig = make_sig(
            signal,
            composite=float(rng.uniform(-1, 1)),
            confidence=int(rng.integers(5, 96)),
            target=target,
        )
        plan = build_execution_plan(md, q, sig, settings)
        assert plan.stop_loss_pct < 0 < plan.take_profit_pct
        assert -cfg.max_stop_pct - 1e-12 <= plan.stop_loss_pct <= -cfg.min_stop_pct + 1e-12
        assert plan.stop_loss < plan.entry_low < plan.entry_price < plan.entry_high
        assert plan.reward_risk >= cfg.min_reward_risk - 1e-9
        assert plan.reward_risk == pytest.approx((plan.take_profit - p) / (p - plan.stop_loss))
        assert plan.take_profit >= target - 1e-9  # TP = max(target, entry + 1.5R)
        assert plan.take_profit_2 is None or plan.take_profit_2 > plan.take_profit
        assert 0.0 <= plan.kelly_fraction <= cfg.kelly_cap
        assert plan.raw_weight >= 0 and (signal in (Signal.BUY, Signal.STRONG_BUY) or plan.raw_weight == 0)
        assert plan.raw_weight <= max(sig.composite, 0) * sig.confidence / 100 / 0.10 + 1e-12


def test_property_sizing_invariants(settings):
    """Random books (incl. negatively correlated names) -> caps, min weight, gross, cash, vol target."""
    rng = np.random.default_rng(7)
    for _ in range(60):
        settings.risk.max_weight = float(rng.uniform(0.05, 0.6))
        settings.risk.min_weight = float(rng.uniform(0.0, 0.05))
        settings.risk.target_portfolio_vol = float(rng.uniform(0.05, 0.3))
        n = int(rng.integers(1, 7))
        f = rng.normal(0, 1, 300)
        mds = [
            md_from_close(
                100 * np.exp(np.cumsum(rng.uniform(-0.03, 0.03) * f + rng.uniform(0.003, 0.03) * rng.normal(0, 1, 300)))
            )
            for _ in range(n)
        ]
        items = [
            (
                m,
                sized_plan(float(rng.uniform(0, 3)), kelly=float(rng.uniform(0, 0.25))),
                make_sig(Signal(rng.choice([s.value for s in Signal]))),
            )
            for m in mds
        ]
        plans, cash = size_portfolio(items, settings, None, "KRW")
        w = np.array([pl.weight for pl in plans])
        assert w.sum() <= settings.risk.max_gross + 1e-12
        assert cash == pytest.approx(1 - w.sum(), abs=1e-12)
        for (_, pl, sg), out in zip(items, plans):
            assert out.weight <= min(settings.risk.max_weight, pl.kelly_fraction) + 1e-12
            assert out.weight == 0 or out.weight >= settings.risk.min_weight - 1e-12
            assert sg.signal in (Signal.BUY, Signal.STRONG_BUY) or out.weight == 0
            assert out.weight > 0 or "분할매수" not in out.action  # never "buy" next to 0%
        assert portfolio_risk(items, w)["ex_ante_vol"] <= settings.risk.target_portfolio_vol * (1 + 1e-9)


# ------------------------------------------------------------------ English output

_HANGUL = __import__("re").compile("[가-힣]")


def _en(settings):
    s = settings.model_copy(deep=True)
    s.report.language = "en"
    return s


def test_format_money_english():
    assert format_money(286_500, "KRW", "en") == "₩286,500"
    assert format_money(-1_500, "KRW", "en") == "-₩1,500"
    assert format_money(0.2, "KRW", "en") == "₩0"
    assert format_money(123.456, "USD", "en") == "$123.46"
    assert format_money(1234.5, "EUR", "en") == "1,234.50 EUR"
    assert format_money(float("nan"), "KRW", "en") == "—"
    assert format_money(286_500, "KRW") == "286,500원"  # Korean default unchanged


def test_action_and_triggers_english(settings, md):
    en = _en(settings)
    p = md.last_price
    llm = LLMIntelligence(news=NewsSentiment(score=0.25, score_100=25, n_articles=10, n_used=6))
    md.fundamentals.next_earnings_date = (pd.Timestamp(md.as_of) + pd.Timedelta(10, "D")).date()
    plan = build_execution_plan(md, make_quant(p, 0.03 * p), make_sig(target=p * 1.1), en, llm=llm)
    H = en.fusion.target_horizon_days
    assert plan.action.startswith("Scale in within ₩") and f"(hold {H} trading days)" in plan.action
    assert plan.triggers[0].startswith("Exit fully on a close below the stop ₩")
    assert plan.triggers[1].startswith("Take 50% profit at ₩")
    assert plan.triggers[2] == time_exit_trigger(H, "en")
    assert "Next earnings 2026-10-03" in plan.triggers
    assert any("from +25" in t for t in plan.triggers)
    for text in (plan.action, *plan.triggers):
        assert not _HANGUL.search(text), text
    md.fundamentals.next_earnings_date = (pd.Timestamp(md.as_of) + pd.Timedelta(1, "D")).date()
    assert earnings_trigger(md, "en").startswith("Avoid new entries within D-3 of earnings — release")


def test_size_portfolio_english_regenerates_english_triggers(settings):
    en = _en(settings)
    mds = independent_mds(2)
    items = []
    for md_ in mds:
        p = md_.last_price
        plan = build_execution_plan(md_, make_quant(p, 0.02 * p), make_sig(Signal.BUY, target=p * 1.1), en)
        items.append((md_, plan, make_sig(Signal.HOLD, composite=0.0)))
    plans, _ = size_portfolio(items, en, None, "KRW")
    for pl in plans:
        assert pl.weight == 0.0 and pl.action.startswith("No new entry")
        assert pl.triggers[0].startswith("Consider entry if the composite recovers to")
        assert not any(t.startswith("Take 50% profit") for t in pl.triggers)
        assert not any(_HANGUL.search(t) for t in pl.triggers)


def test_portfolio_risk_labels_english_twin_has_same_keys():
    from quantcio.fusion.risk import PORTFOLIO_RISK_LABELS, PORTFOLIO_RISK_LABELS_EN

    assert set(PORTFOLIO_RISK_LABELS) == set(PORTFOLIO_RISK_LABELS_EN)
    assert not any(_HANGUL.search(v) for v in PORTFOLIO_RISK_LABELS_EN.values())
    for k in ("var95", "cvar95", "p_loss"):
        assert "approx" in PORTFOLIO_RISK_LABELS_EN[k]


def test_covariance_fallback_warning_language():
    mds = [md_from_close(100.0 * np.exp(np.cumsum(np.full(20, 0.001)))) for _ in range(2)]
    items = [(m, sized_plan(0.5).model_copy(update={"weight": 0.3}), make_sig()) for m in mds]
    portfolio_risk(items, horizon_days=20, lang="en")
    assert any(w.startswith("Too few days for the portfolio covariance") for w in mds[0].warnings)
