"""Execution & risk: entry zone, stop-loss, take-profit, barrier/Kelly, raw weight and portfolio sizing.

Spec: docs/ARCHITECTURE.md, "fusion/risk.py".

- Stops are placed on daily *closes* ("종가 기준 손절"), matching the RSMT-Q barrier simulation.
- With an RSMT-Q `ReturnDistribution` the barrier probabilities and the Kelly fraction come from the
  SAME path matrix that produced the signal (`dist.barrier(stop_pct, take_pct)`); the object is
  passed in, never reconstructed. `dist` is duck-typed: anything with
  `barrier(stop_pct: float, take_pct: float) -> BarrierOutcome` works.
- Every degraded path (no ATR, barrier failure, missing FX, thin covariance sample, stale plan after
  a guardrail downgrade) is logged AND written to `md.warnings` / the plan text - never silent.
- `portfolio_risk` VaR/CVaR/P(loss) are normal approximations on a sample covariance (its docstring
  and PORTFOLIO_RISK_LABELS say so); its keys are the ones report/render.py reads.
- An unusable volatility sizes with a deliberately high UNKNOWN_VOL and a warning, never with the floor.
- The barrier is simulated over the HOLDING horizon (`BarrierOutcome.horizon_days`, which should equal
  `settings.fusion.target_horizon_days`, the horizon of the target price that sets the take-profit).
  A different or unknown simulated horizon is surfaced as a warning.
- Every amount in the action / trigger strings goes through `format_money` ('286,500원', '$123.45'; English
  reports: '₩286,500'). Every user-facing text follows `settings.report.language` (`quantcio.i18n.tr`).
- Sizing is proportional to conviction (`raw_weight`) with ONE common scale k (see `conviction_weights`):
  a higher-conviction name never gets a lower weight than an otherwise-equal name. The old
  normalise-then-cap rule gave every name of a small book the same capped weight.
"""

from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

from quantcio.config import RiskConfig, Settings
from quantcio.fusion.signal import scale_annual_target
from quantcio.i18n import is_en, tr
from quantcio.models import (
    BarrierOutcome,
    ExecutionPlan,
    ExecutiveSignal,
    LLMIntelligence,
    MarketData,
    QuantSignal,
    Signal,
)

if TYPE_CHECKING:  # written concurrently in quant/distribution.py; duck-typed at runtime
    from quantcio.quant.distribution import ReturnDistribution

log = logging.getLogger(__name__)

TRADING_DAYS = 252
BUYISH = frozenset({Signal.BUY, Signal.STRONG_BUY})

# ---- fixed-rule constants from the spec (not configurable in Settings)
ENTRY_LOW_ATR = 0.5  # entry zone = [entry - 0.5·ATR, entry + 0.25·ATR]
ENTRY_HIGH_ATR = 0.25
SUPPORT_SNAP_FAR = 1.3  # snap window: support in [entry - 1.3·dist, entry - 0.5·dist]
SUPPORT_SNAP_NEAR = 0.5
SUPPORT_BUFFER = 0.99  # stop = support·0.99
TP2_RISK_MULT = 3.0  # stretch target: entry + 3R
MIN_VOL_FOR_SIZING = 0.10  # raw weight divides by max(vol, 0.10)
UNKNOWN_VOL = 0.50  # deliberately high vol when none is usable (never the 10% floor)
COV_LOOKBACK = 252
COV_MIN_OBS = 60  # below this the covariance falls back to rho = 1 (conservative)
Z95 = 1.6448536269514722  # one-sided 95% normal quantile
PHI_Z95 = 0.10313564037537146  # standard normal pdf at Z95 (normal CVaR95 = mu - sigma·PHI_Z95/0.05)
BARRIER_SUM_TOL = 1e-6
NEWS_SHIFT_PT = 30  # news-sentiment trigger threshold (score_100 points)
EARNINGS_CAUTION_TRADING_DAYS = 5  # "D-3 이내 신규진입 자제" when earnings are <= 5 weekdays away (~7 calendar days)
MONEY_MISSING = "—"  # format_money of a missing / non-finite amount (same glyph as report/render.MISSING)

# Keys returned by `portfolio_risk`. They are the keys report/render.py (which deliberately does not
# import quantcio.fusion) reads for the risk tiles, so the dict can be stored in
# PortfolioReport.portfolio_risk as is. VaR / CVaR / P(loss) are normal approximations over the
# `horizon_days` passed in (the renderer labels them with settings.horizon_days).
PORTFOLIO_RISK_LABELS: dict[str, str] = {
    "gross": "투자 비중 합계",
    "n_positions": "편입 종목 수",
    "ex_ante_vol": "사전 변동성(연율, 252일 표본공분산)",
    "expected_return": "기대수익률(기간, 근사)",
    "var95": "VaR95(기간, 정규분포 근사)",
    "cvar95": "CVaR95(기간, 정규분포 근사)",
    "p_loss": "손실확률(기간, 정규분포 근사)",
}
# English twin (same keys; "horizon" is the placeholder the renderer replaces with "20 trading days")
PORTFOLIO_RISK_LABELS_EN: dict[str, str] = {
    "gross": "Gross exposure",
    "n_positions": "Positions",
    "ex_ante_vol": "Ex-ante vol (annualised, 252-day sample covariance)",
    "expected_return": "Expected return (horizon, approx.)",
    "var95": "VaR95 (horizon, normal approx.)",
    "cvar95": "CVaR95 (horizon, normal approx.)",
    "p_loss": "P(loss) (horizon, normal approx.)",
}


class BarrierSource(Protocol):
    """What build_execution_plan needs from quant.distribution.ReturnDistribution.

    `barrier(stop_pct, take_pct)` simulates its default holding horizon (`hold_days`) and reports it in
    `BarrierOutcome.horizon_days`; an older object may only expose `horizon` (its path length).
    """

    def barrier(self, stop_pct: float, take_pct: float) -> BarrierOutcome: ...


# --------------------------------------------------------------------------- helpers


def _finite(x: object) -> bool:
    """True for a finite real number, including numpy scalars (np.float32 / np.int64 are not float/int)."""
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(float(x))


def _warn(md: MarketData, msg: str) -> None:
    """Degraded path: log AND surface on the report via md.warnings (deduplicated)."""
    log.warning("%s: %s", md.instrument.ticker, msg)
    if msg not in md.warnings:
        md.warnings.append(msg)


def format_money(value: object, currency: str | None, lang: object = None) -> str:
    """Report money format, shared by every action / trigger / override string.

    KRW -> '286,500원' (no decimals; English reports `lang="en"`: '₩286,500'); USD -> '$123.45'; any other
    code -> '1,234.50 EUR'.
    Negative amounts put the sign first ('-$12.30', '-1,500원'); an amount that rounds to zero never
    shows a sign. A missing or non-finite amount is '—' (never 'nan'). The report can reuse this.
    """
    if not _finite(value):
        return MONEY_MISSING
    x = float(value)
    ccy = (currency or "").strip().upper()
    if ccy == "KRW":
        body = f"₩{abs(x):,.0f}" if is_en(lang) else f"{abs(x):,.0f}원"
    elif ccy == "USD":
        body = f"${abs(x):,.2f}"
    else:
        body = f"{abs(x):,.2f} {ccy}".rstrip()
    rounds_to_zero = not any(ch in "123456789" for ch in body)
    return f"-{body}" if x < 0 and not rounds_to_zero else body


fmt_price = format_money  # backwards-compatible name (report/render.py documents parity with it)


def stop_loss_level(entry: float, atr: float, support: float | None, cfg: RiskConfig) -> tuple[float, bool]:
    """Close-based stop level. Returns (stop, snapped_to_support).

    stop_dist = clip(stop_atr_mult·ATR, min_stop_pct·entry, max_stop_pct·entry). When `support`
    lies in [entry − 1.3·stop_dist, entry − 0.5·stop_dist] the stop snaps to support·0.99. The
    final distance is re-clipped to [min_stop_pct, max_stop_pct]·entry so a snap can never break
    the configured hard limits.
    """
    lo, hi = cfg.min_stop_pct * entry, cfg.max_stop_pct * entry
    stop_dist = float(np.clip(cfg.stop_atr_mult * atr, lo, hi))
    stop = entry - stop_dist
    snapped = False
    if _finite(support) and entry - SUPPORT_SNAP_FAR * stop_dist <= support <= entry - SUPPORT_SNAP_NEAR * stop_dist:
        stop = float(support) * SUPPORT_BUFFER
        snapped = True
    stop = float(np.clip(stop, entry - hi, entry - lo))
    return stop, snapped


def _usable_atr(md: MarketData, quant: QuantSignal, entry: float, lang: object = None) -> float:
    atr = quant.technicals.atr
    if _finite(atr) and atr > 0:
        return float(atr)
    vol = quant.technicals.realized_vol_60d
    fallback = entry * vol / math.sqrt(TRADING_DAYS) if _finite(vol) and vol > 0 else entry * 0.02
    _warn(
        md,
        tr(
            lang,
            "ATR 산출 불가 — 일간 변동성 기반 대체값으로 손절폭 계산",
            "ATR unavailable — stop distance from a daily-volatility substitute",
        ),
    )
    return fallback


def _sizing_vol(md: MarketData, quant: QuantSignal, lang: object = None) -> float:
    """quant.vol_forecast_annual (GARCH(1,1)-t, EWMA fallback), else realized_vol_60d. If neither is a positive
    finite number, warn and return UNKNOWN_VOL: an unknown vol must never size like the 10% floor (the largest
    weight)."""
    v = quant.vol_forecast_annual
    if v is not None and _finite(v) and v > 0:
        return float(v)
    rv = quant.technicals.realized_vol_60d
    if _finite(rv) and rv > 0:
        return float(rv)
    _warn(
        md,
        tr(
            lang,
            f"변동성 산출 불가({rv}) — 비중 산정에 보수적 변동성 {UNKNOWN_VOL:.0%} 적용",
            f"Volatility unavailable ({rv}) — sizing uses a conservative {UNKNOWN_VOL:.0%} vol",
        ),
    )
    return UNKNOWN_VOL


def _prob_up_estimate(quant: QuantSignal, composite: float) -> float:
    """p for the closed-form Kelly: model P(up) (RSMT-Q) or 0.5+0.25c, blended 50/50 with 0.5+0.5c."""
    if quant.distribution is not None:
        p0 = quant.distribution.prob_up
    else:
        p0 = 0.5 + 0.25 * composite
    return 0.5 * p0 + 0.5 * (0.5 + 0.5 * composite)


def closed_form_kelly(p: float, reward_risk: float, kelly_cap: float) -> float:
    """clip(p − (1−p)/b, 0, 1)·kelly_cap with b = reward_risk."""
    if not (_finite(reward_risk) and reward_risk > 0):
        return 0.0
    f = p - (1.0 - p) / reward_risk
    return float(np.clip(f, 0.0, 1.0)) * kelly_cap


def _coerce_barrier(obj: object) -> BarrierOutcome:
    if isinstance(obj, BarrierOutcome):
        return obj
    return BarrierOutcome.model_validate(obj, from_attributes=True)


def _checked_barrier(md: MarketData, dist: object, raw: object, settings: Settings) -> BarrierOutcome:
    """Coerce `dist.barrier(...)`'s result and check its simulated horizon against the holding horizon.

    - `horizon_days > 0`: used as is.
    - `horizon_days == 0` (a distribution that predates the field): its barrier walked the
      distribution's own path matrix, so the horizon is `dist.hold_days` (else `dist.horizon`),
      which is recorded on the outcome. If even that is unknown, a warning says so.
    - A simulated horizon different from `settings.fusion.target_horizon_days` is warned. The
      take-profit comes from the target price for that holding horizon, so a shorter simulation
      understates P(TP first) and overstates P(time exit), which also tightens the barrier guardrail.
    """
    b = _coerce_barrier(raw)
    hold = int(settings.fusion.target_horizon_days)
    hb = int(b.horizon_days)
    if hb <= 0:
        path_len = getattr(dist, "hold_days", None)
        if not (_finite(path_len) and int(path_len) > 0):
            path_len = getattr(dist, "horizon", None)
        if not (_finite(path_len) and int(path_len) > 0):
            _warn(
                md,
                tr(
                    settings,
                    "배리어 시뮬레이션 기간 미기재 — 손절·익절·시간청산 확률의 보유기간 해석 불가",
                    "Barrier simulation horizon not recorded — stop/take-profit/time-exit probabilities have no "
                    "known holding period",
                ),
            )
            return b
        hb = int(path_len)
        b = b.model_copy(update={"horizon_days": hb})
    if hb != hold:
        _warn(
            md,
            tr(
                settings,
                f"배리어 확률은 {hb}거래일 시뮬레이션 기준 — 보유기간 {hold}거래일과 불일치, 확률 해석 주의",
                f"Barrier probabilities simulated over {hb} trading days — differs from the {hold}-day holding "
                "period, interpret with care",
            ),
        )
    return b


# ------------------------------------------------------------------------- text


def action_text(
    signal: Signal,
    plan: ExecutionPlan,
    currency: str,
    hold_days: int | None = None,
    lang: object = None,
) -> str:
    """Deterministic one-liner for the plan (Korean, or English with `lang="en"` / an English Settings).
    `hold_days` (the holding horizon, settings.fusion.target_horizon_days) is stated on Buy / Strong Buy
    actions when given."""
    zone = f"{format_money(plan.entry_low, currency, lang)}~{format_money(plan.entry_high, currency, lang)}"
    stop = format_money(plan.stop_loss, currency, lang)
    if is_en(lang):
        hold = f" (hold {int(hold_days)} trading days)" if hold_days and hold_days > 0 else ""
        if signal == Signal.STRONG_BUY:
            return f"Scale in twice within {zone}{hold}; exit fully on a close below {stop}"
        if signal == Signal.BUY:
            return f"Scale in within {zone}{hold}; exit fully on a close below {stop}"
        if signal == Signal.HOLD:
            return f"No new entry · keep holdings with a closing stop at {stop}"
        return f"Reduce — sell into rebounds near {format_money(plan.entry_high, currency, lang)}; no new buying"
    hold = f"(보유 {int(hold_days)}거래일)" if hold_days and hold_days > 0 else ""
    if signal == Signal.STRONG_BUY:
        return f"{zone} 구간 2회 분할매수{hold}, 종가 기준 {stop} 이탈 시 전량 청산"
    if signal == Signal.BUY:
        return f"{zone} 구간 분할매수{hold}, 종가 기준 {stop} 이탈 시 전량 청산"
    if signal == Signal.HOLD:
        return f"신규 진입 보류 · 보유분은 종가 기준 손절가 {stop} 유지"
    return f"비중 축소 — 반등 시 {format_money(plan.entry_high, currency)} 부근 분할 매도, 신규 매수 금지"


def time_exit_trigger(hold_days: int, lang: object = None) -> str:
    """The time-exit rule the barrier simulation assumes: a path that touches neither the stop nor the
    take-profit within the holding horizon exits at that horizon's close (P(time exit))."""
    return tr(
        lang,
        f"보유 {int(hold_days)}거래일 내 익절·손절 모두 미도달 시 기간 만료 종가 청산 후 재평가",
        f"If neither take-profit nor stop is hit within {int(hold_days)} trading days, exit at that close and reassess",
    )


def signal_triggers(
    signal: Signal,
    composite: float,
    plan: ExecutionPlan,
    settings: Settings,
    currency: str,
) -> list[str]:
    """The signal-specific triggers, which lead `plan.triggers` (Buy/Strong Buy: 3, Hold/Sell: 2).

    A Buy states the stop, the take-profit and the time exit at the holding horizon
    (settings.fusion.target_horizon_days), so the plan text matches the three barrier outcomes.
    A Hold whose composite is already >= the buy threshold, or a Sell whose composite is above the
    sell threshold, comes from a guardrail (critical event, falling knife, barrier, target gap), not
    from the score. For those calls the trigger names the guardrail instead of a score level that is
    already met ("+0.15 이상 회복 시" on a composite of +0.50 would be wrong).
    A Hold whose composite is <= the sell threshold is a Sell lifted by the target-gap guardrail
    (target upside > fusion.max_sell_upside, conflicting evidence). Its trigger says when the Sell
    comes back: once the target is cut.
    """
    t = settings.fusion.thresholds
    if is_en(settings):
        stop = f"{format_money(plan.stop_loss, currency, 'en')} ({plan.stop_loss_pct:+.1%})"
        tp = f"{format_money(plan.take_profit, currency, 'en')} ({plan.take_profit_pct:+.1%})"
        if signal in BUYISH:
            return [
                f"Exit fully on a close below the stop {stop}",
                f"Take 50% profit at {tp}, then raise the stop to the entry price",
                time_exit_trigger(settings.fusion.target_horizon_days, "en"),
            ]
        if signal == Signal.HOLD:
            if composite >= t.buy:
                first = (
                    f"Reconsider entry once the guardrail reason (see overrides) clears (composite {composite:+.2f})"
                )
            elif composite <= t.sell:
                first = (
                    f"Sell signal (composite {composite:+.2f}) conflicts with target upside — reconsider trimming "
                    f"once target upside falls to {settings.fusion.max_sell_upside:+.1%} or below"
                )
            else:
                first = f"Consider entry if the composite recovers to {t.buy:+.2f} or above"
            return [first, f"Holdings: exit fully on a close below the stop {stop}"]
        first = (
            "No new buying until the critical risk is resolved (guardrail Sell)"
            if composite > t.sell
            else f"No new buying until the composite recovers above {t.sell:+.2f}"
        )
        return [first, f"Holdings: exit the rest on a close below {stop}"]
    stop = f"{format_money(plan.stop_loss, currency)}({plan.stop_loss_pct:+.1%})"
    tp = f"{format_money(plan.take_profit, currency)}({plan.take_profit_pct:+.1%})"
    if signal in BUYISH:
        return [
            f"종가 기준 손절가 {stop} 이탈 시 전량 청산",
            f"익절가 {tp} 도달 시 50% 차익실현 후 손절가를 진입가로 상향",
            time_exit_trigger(settings.fusion.target_horizon_days),
        ]
    if signal == Signal.HOLD:
        if composite >= t.buy:
            first = f"가드레일 사유(오버라이드 참조) 해소 시 신규 진입 재검토 (종합점수 {composite:+.2f})"
        elif composite <= t.sell:
            first = (
                f"매도 신호(종합점수 {composite:+.2f})와 목표가 상승여력 상충 — "
                f"목표가 상승여력 {settings.fusion.max_sell_upside:+.1%} 이하로 하향 시 비중 축소 재검토"
            )
        else:
            first = f"종합점수 {t.buy:+.2f} 이상 회복 시 신규 진입 검토"
        return [first, f"보유분: 종가 기준 손절가 {stop} 이탈 시 전량 청산"]
    first = (
        "치명 리스크 해소 확인 전까지 신규 매수 금지 (가드레일 매도)"
        if composite > t.sell
        else f"종합점수 {t.sell:+.2f} 상향 회복 전까지 신규 매수 금지"
    )
    return [first, f"보유분: 종가 기준 {stop} 이탈 시 잔여 전량 청산"]


def _as_date(md: MarketData, x: object, lang: object = None) -> date | None:
    """Normalise an earnings date. `datetime` / `pd.Timestamp` are `date` subclasses but raise TypeError
    when compared with a `date`, so they are converted (a pydantic model does not re-validate on
    attribute assignment, so a provider can leave one there). NaT and unknown types are ignored with a
    warning."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(x, datetime):  # includes pd.Timestamp
        return x.date()
    if isinstance(x, date):
        return x
    _warn(
        md,
        tr(
            lang,
            f"실적발표일 형식 인식 불가({type(x).__name__}) — 실적 트리거 생략",
            f"Earnings date format not recognised ({type(x).__name__}) — earnings trigger skipped",
        ),
    )
    return None


def trading_days_until(as_of: date, when: date) -> int:
    """Weekdays in (as_of, when]: 0 for the same day, 1 for the next weekday. Exchange holidays are not
    known here, so this is an approximation (5 weekdays == 7 calendar days)."""
    start = np.datetime64(as_of, "D") + np.timedelta64(1, "D")
    end = np.datetime64(when, "D") + np.timedelta64(1, "D")
    return int(np.busday_count(start, end))


def earnings_trigger(md: MarketData, lang: object = None) -> str | None:
    """Trigger text for `md.fundamentals.next_earnings_date` (None when unknown).

    - earnings within the next EARNINGS_CAUTION_TRADING_DAYS weekdays (incl. today):
      "실적발표 D-3 이내 신규진입 자제 — 발표 YYYY-MM-DD, N거래일 후" ("당일" for today);
    - later: the neutral "다음 실적발표 YYYY-MM-DD";
    - "(예정일 추정)" follows the date when `next_earnings_is_estimate` is True;
    - a date before `md.as_of` is stale provider data: warned and skipped.
    English (`lang="en"`): "Avoid new entries within D-3 of earnings — release YYYY-MM-DD, in N trading days"
    ("today"), "Next earnings YYYY-MM-DD", " (date estimated)".
    """
    f = md.fundamentals
    when = _as_date(md, f.next_earnings_date, lang)
    if when is None:
        return None
    as_of = md.as_of
    if when < as_of:
        _warn(
            md,
            tr(
                lang,
                f"다음 실적발표일({when:%Y-%m-%d})이 기준일({as_of:%Y-%m-%d}) 이전 — "
                "실적 일정 미갱신, 실적 트리거 생략",
                f"Next earnings date ({when:%Y-%m-%d}) is before the as-of date ({as_of:%Y-%m-%d}) — earnings "
                "calendar stale, earnings trigger skipped",
            ),
        )
        return None
    n = trading_days_until(as_of, when)
    if is_en(lang):
        est = " (date estimated)" if f.next_earnings_is_estimate else ""
        if n <= EARNINGS_CAUTION_TRADING_DAYS:
            when_txt = "today" if n == 0 else f"in {n} trading days"
            return f"Avoid new entries within D-3 of earnings — release {when:%Y-%m-%d}{est}, {when_txt}"
        return f"Next earnings {when:%Y-%m-%d}{est}"
    est = " (예정일 추정)" if f.next_earnings_is_estimate else ""
    if n <= EARNINGS_CAUTION_TRADING_DAYS:
        # "N거래일 후", not "D-n": the report's earnings chip counts CALENDAR days as D-n
        when_txt = "당일" if n == 0 else f"{n}거래일 후"
        return f"실적발표 D-3 이내 신규진입 자제 — 발표 {when:%Y-%m-%d}{est}, {when_txt}"
    return f"다음 실적발표 {when:%Y-%m-%d}{est}"


def _triggers(
    md: MarketData,
    execsig: ExecutiveSignal,
    plan: ExecutionPlan,
    settings: Settings,
    llm: LLMIntelligence | None,
) -> list[str]:
    out = signal_triggers(execsig.signal, execsig.composite, plan, settings, md.instrument.currency)
    earnings = earnings_trigger(md, settings)
    if earnings is not None:
        out.append(earnings)
    news = llm.news if (llm is not None and not llm.skipped) else None
    if news is not None:
        out.append(
            tr(
                settings,
                f"뉴스 감성 {news.score_100:+d}점에서 ±{NEWS_SHIFT_PT}점 이상 급변 시 재평가",
                f"Reassess if news sentiment moves ±{NEWS_SHIFT_PT} pts or more from {news.score_100:+d}",
            )
        )
    else:
        out.append(
            tr(
                settings,
                f"뉴스 감성 급변(±{NEWS_SHIFT_PT}점) 또는 중대 공시 발생 시 재평가",
                f"Reassess on a sharp news-sentiment swing (±{NEWS_SHIFT_PT} pts) or a material filing",
            )
        )
    return out


# ---------------------------------------------------------------- execution plan


def build_execution_plan(
    md: MarketData,
    quant: QuantSignal,
    execsig: ExecutiveSignal,
    settings: Settings,
    dist: ReturnDistribution | None = None,
    *,
    llm: LLMIntelligence | None = None,
) -> ExecutionPlan:
    """Entry zone, close-based stop, take-profits, barrier/Kelly and the raw (pre-normalisation) weight.

    `weight` is left at 0 - it is filled by `size_portfolio`. `llm` (optional) tailors the news
    trigger; `md.fundamentals.next_earnings_date` drives the earnings trigger (see `earnings_trigger`).
    With `dist`, `dist.barrier(stop_loss_pct, take_profit_pct)` is simulated over its default holding
    horizon; see `_checked_barrier` for the horizon checks.
    """
    cfg = settings.risk
    entry = md.last_price
    atr = _usable_atr(md, quant, entry, settings)

    stop, snapped = stop_loss_level(entry, atr, quant.technicals.support, cfg)
    risk_amt = entry - stop
    target = execsig.target_price if _finite(execsig.target_price) and execsig.target_price > 0 else 0.0
    take_profit = max(target, entry + cfg.min_reward_risk * risk_amt)
    # The analyst high is a 12-month target: scale it to the holding horizon exactly like the
    # target-price analyst component (unscaled, a single bullish 12-month target printed "TP2 +185%" on
    # a 60-day plan).
    analyst_high = md.fundamentals.analyst_target_high
    high_h = (
        scale_annual_target(entry, float(analyst_high), settings.fusion.target_horizon_days)
        if _finite(analyst_high) and analyst_high > 0
        else 0.0
    )
    tp2_raw = max(high_h, entry + TP2_RISK_MULT * risk_amt)
    take_profit_2 = tp2_raw if tp2_raw > take_profit else None  # a stretch target must exceed TP1
    reward_risk = (take_profit - entry) / risk_amt
    stop_pct = stop / entry - 1.0
    take_pct = take_profit / entry - 1.0

    barrier: BarrierOutcome | None = None
    kelly: float
    if dist is not None:
        try:
            barrier = _checked_barrier(md, dist, dist.barrier(stop_pct, take_pct), settings)
            total = barrier.p_take_profit + barrier.p_stop_loss + barrier.p_time_exit
            if abs(total - 1.0) > BARRIER_SUM_TOL:
                _warn(
                    md,
                    tr(
                        settings,
                        f"배리어 확률 합 {total:.4f} ≠ 1 — 경로 행렬 점검 필요",
                        f"Barrier probabilities sum to {total:.4f} ≠ 1 — check the path matrix",
                    ),
                )
            if not _finite(barrier.kelly_optimal):
                raise ValueError(f"kelly_optimal not finite: {barrier.kelly_optimal}")
            kelly = float(np.clip(barrier.kelly_optimal, 0.0, 1.0)) * cfg.kelly_cap
        except Exception as exc:  # noqa: BLE001 - reported, then the closed-form fallback is used
            barrier = None
            _warn(
                md,
                tr(
                    settings,
                    f"RSMT-Q 배리어 계산 실패({type(exc).__name__}) — 근사 켈리 사용",
                    f"RSMT-Q barrier computation failed ({type(exc).__name__}) — approximate Kelly used",
                ),
            )
            kelly = closed_form_kelly(_prob_up_estimate(quant, execsig.composite), reward_risk, cfg.kelly_cap)
    else:
        if quant.distribution is not None:
            _warn(
                md,
                tr(
                    settings,
                    "RSMT-Q 경로 행렬 미전달 — 배리어 확률 없이 근사 켈리 사용",
                    "RSMT-Q path matrix not passed — approximate Kelly without barrier probabilities",
                ),
            )
        kelly = closed_form_kelly(_prob_up_estimate(quant, execsig.composite), reward_risk, cfg.kelly_cap)

    raw_weight = 0.0
    if execsig.signal in BUYISH:
        vol = _sizing_vol(md, quant, settings)
        raw_weight = max(execsig.composite, 0.0) * execsig.confidence / 100.0 / max(vol, MIN_VOL_FOR_SIZING)

    # The entry zone must lie above the stop, otherwise the plan tells you to buy below its own stop.
    # That happens only when ATR is extreme (above ~15% of price with a support snap, ~30% without).
    # In that case the zone floor becomes the midpoint between the stop and the entry.
    entry_low = entry - ENTRY_LOW_ATR * atr
    if entry_low <= stop:
        entry_low = stop + 0.5 * risk_amt
        log.info(
            "%s: entry zone floor raised above the stop (ATR %.1f%% of price)",
            md.instrument.ticker,
            100 * atr / entry,
        )

    plan = ExecutionPlan(
        entry_price=entry,
        entry_low=entry_low,
        entry_high=entry + ENTRY_HIGH_ATR * atr,
        stop_loss=stop,
        stop_loss_pct=stop_pct,
        take_profit=take_profit,
        take_profit_pct=take_pct,
        take_profit_2=take_profit_2,
        reward_risk=reward_risk,
        barrier=barrier,
        weight=0.0,
        raw_weight=float(raw_weight),
        kelly_fraction=float(kelly),
    )
    triggers = _triggers(md, execsig, plan, settings, llm)
    if snapped:
        log.debug("%s: stop snapped to support %.4f", md.instrument.ticker, stop)
    return plan.model_copy(
        update={
            "action": action_text(
                execsig.signal,
                plan,
                md.instrument.currency,
                settings.fusion.target_horizon_days,
                settings,
            ),
            "triggers": triggers,
        }
    )


# ---------------------------------------------------------------------- covariance


def estimate_covariance(
    mds: Sequence[MarketData],
    lookback: int = COV_LOOKBACK,
    min_obs: int = COV_MIN_OBS,
) -> tuple[np.ndarray, int, bool]:
    """Annualised covariance of the last `lookback` aligned daily log returns (local currency).

    Closes are inner-joined on common dates (constant FX: the FX effect is ignored, per spec).
    Returns (cov, n_obs, used_fallback). With fewer than `min_obs` common returns (or a non-finite
    estimate) it falls back to each name's own-calendar vol with correlation 1 - a conservative
    upper bound - and the caller must surface `used_fallback`.
    """
    n = len(mds)
    if n == 0:
        return np.zeros((0, 0)), 0, False
    closes = pd.concat({i: md.prices["close"] for i, md in enumerate(mds)}, axis=1, join="inner").sort_index()
    rets = np.log(closes).diff().dropna().iloc[-lookback:]
    n_obs = int(len(rets))
    if n_obs >= min_obs:
        cov = rets.cov(ddof=1).to_numpy(dtype=float) * TRADING_DAYS
        if np.all(np.isfinite(cov)):
            return cov, n_obs, False
    vols = []
    for md in mds:
        own = np.log(md.prices["close"]).diff().dropna().iloc[-lookback:]
        v = float(own.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(own) > 1 else float("nan")
        vols.append(v if math.isfinite(v) and v > 0 else 0.5)  # unknown vol -> a deliberately high 50%
    s = np.asarray(vols)
    return np.outer(s, s), n_obs, True


def _portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(math.sqrt(max(0.0, float(w @ cov @ w))))


def _rescale_return(ret: float, from_days: int, to_days: int) -> float:
    """Compound a simple return over `from_days` to `to_days`: (1+ret)^(to/from) − 1 (−1 stays −1)."""
    if ret <= -1.0:
        return -1.0
    if from_days == to_days:
        return ret
    return (1.0 + ret) ** (to_days / max(1, from_days)) - 1.0


# ------------------------------------------------------------------------- sizing


Item = tuple[MarketData, ExecutionPlan, ExecutiveSignal]

# Which constraint set the common scale k (logged; also in SizingResult.binding).
SIZING_BINDING_KO: dict[str, str] = {
    "max_weight": "최고 확신 종목 비중 상한",
    "max_gross": "총투자 한도",
    "target_vol": "목표 포트폴리오 변동성",
    "none": "편입 종목 없음",
}
SIZING_BINDING_EN: dict[str, str] = {
    "max_weight": "top-conviction weight cap",
    "max_gross": "gross exposure limit",
    "target_vol": "target portfolio volatility",
    "none": "no positions",
}
_POLISH_WARN = 1e-6  # a final feasibility rescale larger than this means the closed form is off: log it
_BISECT_ITERS = 200


@dataclass(frozen=True)
class SizingResult:
    """Output of `conviction_weights`. `weights` has one entry per input name (0 = not held)."""

    weights: np.ndarray
    scale: float  # the common k of the final round (0 when nothing is held)
    binding: str  # key of SIZING_BINDING_KO: the constraint that set k in the last solve ("none": no solve)
    dropped: tuple[int, ...]  # names removed by the min-weight rule, in drop order
    rounds: int  # solve rounds (1 + number of min-weight drop steps)
    drop_binding: dict[int, str] = field(default_factory=dict)  # dropped name -> binding of ITS round


def _scale_pieces(raw: np.ndarray, caps: np.ndarray, k_hi: float) -> list[tuple[float, float, np.ndarray]]:
    """Split [0, k_hi] at the cap breakpoints b_i = cap_i/raw_i. Returns (lo, hi, capped) per piece, where
    `capped` marks the names at their cap for every k in (lo, hi) (b_i ≤ lo; no breakpoint lies inside)."""
    b = caps / raw
    knots = np.unique(np.concatenate(([0.0, k_hi], b[(b > 0.0) & (b < k_hi)])))
    return [(float(lo), float(hi), b <= lo) for lo, hi in zip(knots[:-1], knots[1:])]


def _largest_feasible_k(qa: float, qb: float, qc: float, t2: float, lo: float, hi: float) -> float | None:
    """Largest k in [lo, hi] with f(k) = qa·k² + 2·qb·k + qc ≤ t2, or None. qa ≥ 0, so f is convex and
    its sublevel set is one interval. The larger root is computed in its cancellation-free form, and a
    bisection covers the rounding case where the root lands a hair outside [lo, hi]."""

    def f(k: float) -> float:
        return (qa * k + 2.0 * qb) * k + qc

    if f(hi) <= t2:
        return hi
    cand: float | None = None
    if qa <= 0.0:  # linear
        if qb > 0.0:
            cand = (t2 - qc) / (2.0 * qb)
    else:
        disc = qb * qb - qa * (qc - t2)
        if disc >= 0.0:
            s = math.sqrt(disc)
            cand = (t2 - qc) / (qb + s) if qb > 0.0 else (s - qb) / qa
    if cand is not None and lo <= cand <= hi and f(cand) <= t2 * (1.0 + 1e-12):
        return float(cand)
    if f(lo) <= t2:  # f crosses t2 inside [lo, hi]; the closed form was lost to rounding
        a, z = lo, hi
        for _ in range(_BISECT_ITERS):
            m = 0.5 * (a + z)
            if f(m) <= t2:
                a = m
            else:
                z = m
        return a
    return None


def conviction_scale(
    raw: Sequence[float],
    caps: Sequence[float],
    cov: np.ndarray,
    *,
    max_weight: float,
    max_gross: float,
    target_vol: float,
) -> tuple[float, str]:
    """Largest common scale k for w_i(k) = min(raw_i·k, cap_i). Returns (k, binding).

    The rule (docs/ARCHITECTURE.md, fusion/risk.py → size_portfolio):
    - max_weight applies to the PROPORTIONAL weight: raw_i·k ≤ max_weight for every name, so
      k ≤ max_weight / max(raw). The highest-conviction name sets the scale.
    - cap_i = min(max_weight, kelly_i) clips each name on its own. A name that reaches its (Kelly)
      cap stays capped while the others keep w ∝ raw. The excess is not redistributed past the
      max_weight ceiling. If it were, every name of a small book would grow to the same cap: that
      was the old flattening.
    - Σw ≤ max_gross and √(wᵀΣw) ≤ target_vol. k is the LARGEST value satisfying all of them.
      Σw is monotone in k, so the gross limit is a closed-form cut of the piecewise-linear sum.
      Vol is NOT monotone when the book holds a hedge (a capped name offset by a growing,
      negatively correlated name). On each piece between cap breakpoints, vol² is a convex
      quadratic in k, so the largest feasible k is found exactly, piece by piece from the top.

    `raw` must be > 0 and finite; `caps` must be in (0, max_weight]. `cov` is the annualised
    covariance of these names.
    """
    raw = np.asarray(raw, dtype=float)
    caps = np.asarray(caps, dtype=float)
    n = raw.size
    if caps.shape != raw.shape:
        raise ValueError(f"raw/caps shape mismatch: {raw.shape} vs {caps.shape}")
    if n == 0:
        return 0.0, "none"
    if not (np.all(np.isfinite(raw)) and np.all(raw > 0)):
        raise ValueError(f"conviction_scale: raw weights must be finite and > 0, got {raw}")
    if not (np.all(np.isfinite(caps)) and np.all(caps > 0)):
        raise ValueError(f"conviction_scale: caps must be finite and > 0, got {caps}")
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (n, n) or not np.all(np.isfinite(cov)):
        raise ValueError(f"conviction_scale: covariance must be a finite {n}x{n} matrix")

    k_hi = max_weight / float(raw.max())
    binding = "max_weight"
    # gross: S(k) = Σ min(raw_i·k, cap_i) is continuous and non-decreasing
    for lo, hi, capped in _scale_pieces(raw, caps, k_hi):
        c_sum, u_sum = float(caps[capped].sum()), float(raw[~capped].sum())
        if c_sum + u_sum * hi <= max_gross:
            continue
        # S(lo) ≤ max_gross < S(hi) on this piece, hence u_sum > 0
        k_hi = float(np.clip((max_gross - c_sum) / u_sum, lo, hi))
        binding = "max_gross"
        break
    if k_hi <= 0.0:
        return 0.0, binding

    t2 = target_vol * target_vol
    for lo, hi, capped in reversed(_scale_pieces(raw, caps, k_hi)):
        c = np.where(capped, caps, 0.0)
        u = np.where(capped, 0.0, raw)
        qa = max(0.0, float(u @ cov @ u))
        qb = float(c @ cov @ u)
        qc = max(0.0, float(c @ cov @ c))
        k = _largest_feasible_k(qa, qb, qc, t2, lo, hi)
        if k is not None:
            return k, (binding if k >= k_hi else "target_vol")
    return 0.0, "target_vol"


def conviction_weights(
    raw: Sequence[float],
    caps: Sequence[float],
    cov_for: Callable[[np.ndarray], np.ndarray],
    cfg: RiskConfig,
) -> SizingResult:
    """Conviction-proportional weights with one common scale (the whole `size_portfolio` weight rule).

    `raw[i]` is the conviction (ExecutionPlan.raw_weight; ≤ 0 means not investable) and `caps[i]` is
    min(max_weight, kelly_i). `cov_for(idx)` returns the annualised covariance of the names `idx`
    (size_portfolio: the 252-day sample covariance). Rule:

    1. Eligible names have raw > 0 and a cap ≥ max(min_weight, tiny). A cap below min_weight can
       never reach min_weight.
    2. Solve the largest common k (`conviction_scale`): w_i = min(raw_i·k, cap_i), with
       raw_i·k ≤ max_weight, Σw ≤ max_gross and ex-ante vol ≤ target_portfolio_vol.
    3. If any w_i < min_weight, drop the SMALLEST such position(s) (ties together, so equal names
       stay equal), re-estimate the covariance of the survivors and go back to 2. Dropping one tier at
       a time never discards a name that clears min_weight once the budget is re-solved. Re-solving
       also re-checks vol: dropping a diversifier can raise it.

    Guarantees: every w_i ≤ cap_i; Σw ≤ max_gross; vol ≤ target; w_i = 0 or w_i ≥ min_weight. With
    equal caps, raw_i ≥ raw_j implies w_i ≥ w_j (one common k; a smaller position is dropped first).
    Weights are invariant to a common rescaling of `raw`.
    """
    raw = np.asarray(raw, dtype=float)
    caps = np.asarray(caps, dtype=float)
    n = raw.size
    for name, v in (
        ("max_weight", cfg.max_weight),
        ("min_weight", cfg.min_weight),
        ("max_gross", cfg.max_gross),
        ("target_portfolio_vol", cfg.target_portfolio_vol),
    ):
        if not (_finite(v) and float(v) >= 0.0):
            raise ValueError(f"risk.{name} must be a finite number >= 0, got {v!r}")
    w = np.zeros(n)
    ok = np.isfinite(raw) & (raw > 0) & np.isfinite(caps) & (caps > 0) & (caps >= cfg.min_weight)
    active = np.flatnonzero(ok)
    dropped: list[int] = []
    drop_binding: dict[int, str] = {}
    k, binding, rounds = 0.0, "none", 0
    while active.size:
        rounds += 1
        cov = np.asarray(cov_for(active), dtype=float)
        k, binding = conviction_scale(
            raw[active],
            caps[active],
            cov,
            max_weight=cfg.max_weight,
            max_gross=cfg.max_gross,
            target_vol=cfg.target_portfolio_vol,
        )
        wa = np.minimum(raw[active] * k, caps[active])
        # numerical polish only: a uniform shrink keeps every cap, the proportions and the ordering
        shrink = 1.0
        gross = float(wa.sum())
        if gross > cfg.max_gross:
            shrink = cfg.max_gross / gross
        vol = _portfolio_vol(wa * shrink, cov)
        if vol > cfg.target_portfolio_vol:
            shrink *= cfg.target_portfolio_vol / vol
        if shrink < 1.0:
            if shrink < 1.0 - _POLISH_WARN:
                log.warning("sizing: feasibility polish shrank the book by %.3g", 1.0 - shrink)
            wa = wa * shrink
        below = wa < cfg.min_weight
        if not below.any():
            w[active] = wa
            break
        smallest = float(wa[below].min())
        tier = below & (wa <= smallest * (1.0 + 1e-12) + 1e-300)
        for i in active[tier]:
            dropped.append(int(i))
            drop_binding[int(i)] = binding
        active = active[~tier]
    if not active.size:
        k = 0.0  # nothing held; `binding` still names the constraint of the last solve (if any)
    return SizingResult(
        weights=w,
        scale=float(k),
        binding=binding,
        dropped=tuple(dropped),
        rounds=rounds,
        drop_binding=drop_binding,
    )


def size_portfolio(
    items: list[Item],
    settings: Settings,
    capital: float | None,
    base_currency: str,
) -> tuple[list[ExecutionPlan], float]:
    """Portfolio weights + optional share counts. Returns (plans, cash_weight).

    Weights follow `conviction_weights`: w_i = min(raw_i·k, min(max_weight, kelly_i)), with ONE common
    scale k. k is the largest value such that the highest-conviction name is at most max_weight,
    Σw ≤ max_gross and the ex-ante vol (252-day sample covariance) ≤ target_portfolio_vol. A name at
    its Kelly cap stays capped while the rest keep w ∝ raw. Positions below min_weight are dropped
    smallest-first and k is re-solved, so the final book meets every constraint, including the vol
    target that `portfolio_risk` reports. cash = 1 − Σw.
    Conviction ordering is preserved: of two otherwise-equal names, the higher raw weight never gets
    the lower weight. The old normalise → cap → ×max_gross rule gave every name of a book with few
    Buys the same capped weight. Hold/Sell get 0; the signal is re-checked here, so a post-plan
    guardrail downgrade is honoured.
    The action text and the signal-specific triggers are regenerated from the final signal, so they
    always match the call after a guardrail downgrade. A Buy/Strong Buy that ends at 0% (Kelly 0,
    below min_weight, or a stale plan) gets an action that states why it is not bought.
    With `capital` (base currency): position_value = w·capital, shares = floor(value / (entry·fx_to_base)).
    A same-currency name must have fx 1; any other value is corrected to 1 with a warning. A foreign
    name with fx exactly 1.0 (the loader's failure default) gets shares=None and a warning.
    """
    cfg = settings.risk
    if capital is not None and not (_finite(capital) and capital > 0):
        raise ValueError(f"capital must be a positive number, got {capital!r}")
    plans: list[ExecutionPlan] = []
    raw = np.zeros(len(items))
    caps = np.zeros(len(items))
    for i, (md, plan, sig) in enumerate(items):
        if sig.signal not in BUYISH and plan.raw_weight > 0:
            # plan was built before a guardrail downgrade: zero it so a Hold/Sell never gets weight
            log.info(
                "%s: plan built for a Buy but final signal is %s - weight zeroed",
                md.instrument.ticker,
                sig.signal.value,
            )
            plan = plan.model_copy(update={"raw_weight": 0.0})
        ccy = md.instrument.currency
        action = action_text(sig.signal, plan, ccy, settings.fusion.target_horizon_days, settings)
        if plan.action != action:
            # e.g. Strong Buy -> Buy after the barrier guardrail: "2회 분할매수" must not survive
            plan = plan.model_copy(update={"action": action})
        # same for the signal-specific triggers (Buy -> Hold must not keep "익절가 도달 시 50% 차익실현" as
        # its lead rule): drop any signal trigger of any call, then prepend the final call's own
        stale = {t for s in Signal for t in signal_triggers(s, sig.composite, plan, settings, ccy)}
        triggers = signal_triggers(sig.signal, sig.composite, plan, settings, ccy) + [
            t for t in plan.triggers if t not in stale
        ]
        if triggers != plan.triggers:
            plan = plan.model_copy(update={"triggers": triggers})
        plans.append(plan)
        if sig.signal in BUYISH and _finite(plan.raw_weight) and plan.raw_weight > 0:
            raw[i] = plan.raw_weight
        kf = plan.kelly_fraction if _finite(plan.kelly_fraction) else 0.0
        caps[i] = max(0.0, min(cfg.max_weight, kf))

    def cov_for(idx: np.ndarray) -> np.ndarray:
        cov, n_obs, fallback = estimate_covariance([items[i][0] for i in idx])
        if fallback:
            for i in idx:
                _warn(items[i][0], _cov_fallback_msg(n_obs, settings))
        return cov

    res = conviction_weights(raw, caps, cov_for, cfg)
    w = res.weights
    if w.any():
        log.info(
            "sizing: common scale k=%.4g set by %s (%d round(s), min-weight drops %s)",
            res.scale,
            res.binding,
            res.rounds,
            [items[i][0].instrument.ticker for i in res.dropped],
        )
    cash = float(1.0 - w.sum())

    out: list[ExecutionPlan] = []
    for i, ((md, _, sig), plan) in enumerate(zip(items, plans)):
        update: dict[str, object] = {"weight": float(w[i])}
        if sig.signal in BUYISH and w[i] <= 0.0:
            # A Buy call that ends at 0% must not keep a "분할매수" action: say why it is not bought.
            en = is_en(settings)
            if raw[i] <= 0.0:
                reason = tr(
                    settings,
                    "실행계획 원시 비중 0(매수 전 신호 기준으로 생성된 계획)",
                    "raw plan weight 0 (plan built for the pre-Buy signal)",
                )
                _warn(
                    md,
                    tr(
                        settings,
                        f"매수 신호이나 {reason} — 실행계획 재생성 필요",
                        f"Buy signal but {reason} — execution plan must be rebuilt",
                    ),
                )
            elif cfg.max_weight <= 0.0:
                reason = tr(settings, "종목당 최대 비중(risk.max_weight) 0 설정", "risk.max_weight set to 0")
            elif caps[i] <= 0.0:
                reason = tr(
                    settings,
                    "켈리 비중 0(손익 분포상 기대 우위 없음)",
                    "Kelly weight 0 (no expected edge in the P&L distribution)",
                )
            elif caps[i] < cfg.min_weight:
                reason = tr(
                    settings,
                    f"켈리 비중 상한 {caps[i]:.1%}이 최소 비중 {cfg.min_weight:.0%} 미만",
                    f"Kelly cap {caps[i]:.1%} below the {cfg.min_weight:.0%} minimum weight",
                )
            else:
                bind = res.drop_binding.get(i, res.binding)
                if en:
                    reason = (
                        f"conviction-scaled weight below the {cfg.min_weight:.0%} minimum "
                        f"(scale set by the {SIZING_BINDING_EN.get(bind, bind)})"
                        if bind != "none"
                        else f"scaled weight below the {cfg.min_weight:.0%} minimum"
                    )
                else:
                    reason = (
                        f"확신도 비례 조정 비중이 최소 비중 {cfg.min_weight:.0%} 미만"
                        f"({SIZING_BINDING_KO.get(bind, bind)} 기준 배율)"
                        if bind != "none"
                        else f"조정 비중이 최소 비중 {cfg.min_weight:.0%} 미만"
                    )
            log.info(
                "%s: %s signal sized to 0 - %s",
                md.instrument.ticker,
                sig.signal.value,
                reason,
            )
            update["action"] = tr(
                settings,
                f"매수 신호이나 {reason} → 신규 편입 보류 · 보유분은 종가 기준 손절가 "
                f"{format_money(plan.stop_loss, md.instrument.currency)} 유지",
                f"Buy signal but {reason} → no new position · keep holdings with a closing stop at "
                f"{format_money(plan.stop_loss, md.instrument.currency, 'en')}",
            )
        if capital is not None:
            value = float(w[i] * capital)
            update["position_value"] = value
            fx = md.fx_to_base
            if md.instrument.currency == base_currency and fx != 1.0:
                # same currency means fx is 1 by definition; a different value means md was loaded
                # for another base currency and would mis-size shares by that factor (e.g. 1380x)
                _warn(
                    md,
                    tr(
                        settings,
                        f"기준통화와 동일 통화({base_currency})인데 환율 {fx} — 1.0으로 보정해 주식 수 산출",
                        f"Same currency as the base ({base_currency}) but FX {fx} — corrected to 1.0 for share count",
                    ),
                )
                fx = 1.0
            if md.instrument.currency != base_currency and fx == 1.0:
                _warn(
                    md,
                    tr(
                        settings,
                        f"환율({md.instrument.currency}→{base_currency}) 미확인 — 주식 수 산출 보류",
                        f"FX ({md.instrument.currency}→{base_currency}) unconfirmed — share count withheld",
                    ),
                )
                update["shares"] = None
            elif not (_finite(fx) and fx > 0):
                _warn(
                    md,
                    tr(
                        settings, f"환율 값 이상({fx}) — 주식 수 산출 보류", f"Invalid FX ({fx}) — share count withheld"
                    ),
                )
                update["shares"] = None
            else:
                update["shares"] = int(math.floor(value / (plan.entry_price * fx) + 1e-9))
        out.append(plan.model_copy(update=update))
    return out, cash


# ------------------------------------------------------------------ portfolio risk


def _cov_fallback_msg(n_obs: int, lang: object) -> str:
    return tr(
        lang,
        f"포트폴리오 공분산 표본 부족({n_obs}일) — 상관계수 1 가정(보수적) 적용",
        f"Too few days for the portfolio covariance ({n_obs}) — correlation 1 assumed (conservative)",
    )


def portfolio_risk(
    items: list[Item],
    weights: Sequence[float] | None = None,
    horizon_days: int = 20,
    lang: object = None,
) -> dict[str, float]:
    """Ex-ante portfolio risk of the sized book. The VaR, CVaR and P(loss) values are normal APPROXIMATIONS.

    Keys (see PORTFOLIO_RISK_LABELS; they match what report/render.py reads):
    - gross: Σw; n_positions: the number of names with w > 0.
    - ex_ante_vol: √(wᵀΣw) with Σ the annualised 252-day sample covariance (see estimate_covariance).
    - expected_return: Σ w_i·μ_i over `horizon_days`. μ_i is the plan's RSMT-Q barrier expected trade
      return rescaled from its simulated horizon Hb = barrier.horizon_days to h:
      (1+R)^(h/Hb) − 1 (Hb = 0, an outcome that predates the field, is taken as h). Without a barrier
      it is the target-implied return rescaled the same way: (1+upside)^(h/H) − 1. Mixing a 60-day
      mean with the h-day sigma below would overstate μ_p about 3x.
    - var95: μ_p − 1.645·σ_p (5% quantile of the horizon return, negative decimal, same sign
      convention as DistributionForecast.var95); σ_p = vol·√(h/252).
    - cvar95: μ_p − σ_p·φ(1.645)/0.05, the normal expected shortfall below var95.
    - p_loss: Φ(−μ_p/σ_p).
    Cash earns 0. `weights` defaults to each plan's `weight`. Pass horizon_days = settings.horizon_days
    (the renderer labels VaR/P(loss) with it). A thin covariance sample adds a warning to md.warnings
    (see estimate_covariance), in the language `lang` (Settings / "ko" / "en"; default Korean).
    """
    w = np.asarray(
        [p.weight for _, p, _ in items] if weights is None else list(weights),
        dtype=float,
    )
    if len(w) != len(items):
        raise ValueError(f"weights length {len(w)} != items length {len(items)}")
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    gross = float(w.sum())
    out: dict[str, float] = {
        "gross": gross,
        "n_positions": float(np.count_nonzero(w)),
        "ex_ante_vol": 0.0,
        "expected_return": 0.0,
        "var95": 0.0,
        "cvar95": 0.0,
        "p_loss": 0.0,
    }
    active = np.flatnonzero(w > 0)
    if not active.size:
        return out
    cov, n_obs, fallback = estimate_covariance([items[i][0] for i in active])
    if fallback:
        for i in active:
            _warn(items[i][0], _cov_fallback_msg(n_obs, lang))
    vol = _portfolio_vol(w[active], cov)
    mu = 0.0
    for i in active:
        _, plan, sig = items[i]
        b = plan.barrier
        if b is not None and _finite(b.expected_trade_return):
            ret = float(b.expected_trade_return)
            H = b.horizon_days if b.horizon_days > 0 else horizon_days
        else:
            ret = float(sig.upside)
            H = max(1, sig.horizon_days)
        mu_i = _rescale_return(ret, H, horizon_days)
        mu += w[i] * mu_i
    sigma_h = vol * math.sqrt(horizon_days / TRADING_DAYS)
    p_loss = 0.5 * (1.0 + math.erf((-mu / sigma_h) / math.sqrt(2.0))) if sigma_h > 0 else float(mu < 0)
    out.update(
        ex_ante_vol=vol,
        expected_return=float(mu),
        var95=float(mu - Z95 * sigma_h),
        cvar95=float(mu - sigma_h * PHI_Z95 / 0.05),
        p_loss=float(p_loss),
    )
    return out
