"""Fusion: pillars -> composite -> confidence -> signal -> guardrails -> target price -> thesis.

Deterministic core, bounded LLM (docs/ARCHITECTURE.md, "fusion/signal.py"):

- The composite is a weighted mean of the *available* pillars (factor, ml, llm), each in [-1, 1],
  with `settings.fusion.weights` renormalised over the available ones. A missing pillar is never
  silently treated as a neutral 0: it is dropped from the mean and an explanatory note is added to
  `ExecutiveSignal.overrides`.
- The LLM can only move the call through the bounded `llm_pillar`, the bounded critic
  `confidence_adjustment` ([-25, +5]) and the `critical_flag`, which acts through a fixed rule.
- Every confidence component is returned in `ExecutiveSignal.confidence_breakdown`.
- The agreement component is scaled by pillar *coverage* (the configured weight share of the
  available pillars), so a missing pillar can never count as agreement: a factor-only call cannot
  reach the full 35 points that three agreeing pillars earn.
- `apply_barrier_guardrail` runs after the execution plan exists (it needs the RSMT-Q barrier
  probabilities, simulated over the holding horizon) and can only downgrade a Buy-or-better call by
  one notch. Its margin is `settings.fusion.barrier_guard_margin`.
- When `quant.factor_model` (the PIT multi-factor model) is present, the factor pillar is
  `(1 - b)·legacy composite pillar + b·factor_model.pillar` with b = `settings.backtest.factor_pillar_blend`
  (`factor_pillar_value`); the blend is stated in `overrides` with the prefix FACTOR_BLEND_PREFIX. With
  `fusion.factor_pillar_asymmetry: auto` the model pillar is ×FACTOR_WEAK_SIDE on the side its backtest side test
  found weak (`FactorModelScore.weak_side`, backtest.model.pillar_side_test; None = symmetric).
- Role separation (2026-09-29): RSMT-Q is the risk / distribution engine; the DIRECTION comes from the stock-selection
  model. When `quant.excess_model` is present the ML pillar is `SELECTION_BLEND·w·selection view` with
  SELECTION_BLEND = 1 (`ml_pillar_value`): the selection view is the model's RELATIVE call
  (0.5·tanh(2·mu_excess/sigma_excess) + 0.5·(2·P(beat benchmark) - 1)) and w = 1 when the model's skill is established
  (serving gate + untouched-name confirmation), else SELECTION_UNSKILLED. The RSMT-Q distribution pillar is only the
  fallback without a selection model. "ML skill" (confidence component, Strong-Buy guardrail) then follows the
  selection model's served skill too (`ml_skill_ok`, `_ml_skill_component`). Stated in `overrides` with
  ML_BLEND_PREFIX.
- Every user-facing note is built in the report language (`settings.report.language`, `quantcio.i18n.tr`).
- The target-gap guardrail is symmetric. A Buy/Strong Buy whose target upside is below
  `fusion.min_buy_upside` becomes Hold. A Sell whose target upside is above `fusion.max_sell_upside`
  becomes Hold, because the score and the target price contradict each other. That lift is the ONLY
  guardrail that can raise a call, only from Sell to Hold, and never for a critical-event Sell.

Constants that are fixed rules in the spec (not in `Settings`) are module-level names below so
they are visible and testable in one place.
"""

from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from quantcio.config import Settings
from quantcio.i18n import is_en, lang_of, tr
from quantcio.models import (
    ExecutionPlan,
    ExecutiveSignal,
    LLMIntelligence,
    MarketData,
    QuantSignal,
    Signal,
)

log = logging.getLogger(__name__)

TRADING_DAYS = 252

PILLARS: tuple[str, ...] = ("factor", "ml", "llm")
PILLAR_LABELS = {"factor": "팩터", "ml": "ML", "llm": "LLM"}
PILLAR_LABELS_EN = {"factor": "Factor", "ml": "ML", "llm": "LLM"}
# English factor names for the deterministic thesis (quant/factors.py FACTOR_LABELS_EN; kept local so fusion
# does not import the quant package): an English report never borrows a Korean FactorScore.label
_FACTOR_NAMES_EN = {
    "momentum": "Momentum",
    "value": "Value",
    "volatility": "Volatility (stability)",
    "quality": "Quality",
    "trend": "Trend",
}

# worst -> best; used for caps and one-notch downgrades
SIGNAL_ORDER: tuple[Signal, ...] = (
    Signal.SELL,
    Signal.HOLD,
    Signal.BUY,
    Signal.STRONG_BUY,
)
BUYISH = frozenset({Signal.BUY, Signal.STRONG_BUY})

HORIZON_LABELS: dict[Signal, str] = {
    Signal.STRONG_BUY: "1~3개월 (분할매수)",
    Signal.BUY: "1~3개월",
    Signal.HOLD: "관망 · 1개월 후 재평가",
    Signal.SELL: "즉시~2주 내 비중 축소",
}
HORIZON_LABELS_EN: dict[Signal, str] = {
    Signal.STRONG_BUY: "1–3 months (scale in)",
    Signal.BUY: "1–3 months",
    Signal.HOLD: "Wait · review in 1 month",
    Signal.SELL: "Trim within 2 weeks",
}
SIGNAL_KO: dict[Signal, str] = {
    Signal.STRONG_BUY: "강력 매수",
    Signal.BUY: "매수",
    Signal.HOLD: "관망",
    Signal.SELL: "매도(비중 축소)",
}
SIGNAL_EN: dict[Signal, str] = {
    Signal.STRONG_BUY: "Strong Buy",
    Signal.BUY: "Buy",
    Signal.HOLD: "Hold",
    Signal.SELL: "Sell",
}

# ---- fixed-rule constants from the spec (not configurable in Settings)
NEUTRAL_PILLAR_BAND = 0.05  # |pillar| below this counts as half-agreeing
STRENGTH_FULL_SCALE = 0.6  # |composite| at which the strength component saturates
AGREEMENT_MAX = 35.0
STRENGTH_MAX = 20.0
ML_SKILL_MAX = 20.0
DATA_QUALITY_MAX = 15.0
DATA_QUALITY_PER_WARNING = 3.0
COMPLETENESS_FLOOR = 0.4
EVIDENCE_MAX = 10.0
EVIDENCE_FULL_ARTICLES = 8
RSMT_DEGRADED_PENALTY = -10.0
CONFIDENCE_FLOOR, CONFIDENCE_CEIL = 5, 95
CRITIC_ADJ_MIN, CRITIC_ADJ_MAX = -25, 5
FALLING_KNIFE_RET_1M = -0.20
FALLING_KNIFE_TREND_MAX = 30.0
FALLING_KNIFE_LLM_OVERRIDE = 0.4
# the barrier-guardrail margin is settings.fusion.barrier_guard_margin (configurable, no module default)
BARRIER_NOTE_PREFIX = "배리어 가드레일:"  # override note prefix; also makes the guardrail idempotent
BARRIER_NOTE_PREFIX_EN = "Barrier guardrail:"
TARGET_NOTE_PREFIX = "목표가 괴리 가드레일:"  # both sides of the symmetric target-gap guardrail
TARGET_NOTE_PREFIX_EN = "Target-gap guardrail:"
# the factor-pillar blend note (an informational override; report/render.py shows it in Module 2, not as a chip)
FACTOR_BLEND_PREFIX = "팩터 필러 블렌드:"
FACTOR_BLEND_PREFIX_EN = "Factor pillar blend:"
# the ML-pillar blend note (informational like the factor blend; not a chip)
ML_BLEND_PREFIX = "ML 필러 블렌드:"
ML_BLEND_PREFIX_EN = "ML pillar blend:"
SELECTION_BLEND = 1.0  # share of the ML pillar taken by the stock-selection model's relative view (role separation)
SELECTION_UNSKILLED = 0.35  # weight of that view when the model's skill is not established (same as elsewhere)
FACTOR_WEAK_SIDE = 0.35  # factor-model pillar on the side the backtest side test finds weak (same shrink)
NO_COMPONENT_TP_SCALE = 0.1  # target = price * (1 + composite * 0.1) when no component exists
_EPS = 1e-12


# --------------------------------------------------------------------------- helpers


def horizon_label(signal: Signal, lang: object = None) -> str:
    """Recommended holding-period label of a call in the report language (`lang`: Settings / "ko" / "en")."""
    return (HORIZON_LABELS_EN if is_en(lang) else HORIZON_LABELS)[signal]


def signal_label(signal: Signal, lang: object = None) -> str:
    return (SIGNAL_EN if is_en(lang) else SIGNAL_KO)[signal]


def _finite(x: object) -> bool:
    """True for a finite real number, including numpy scalars (np.float32 / np.int64 are not float/int)."""
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(float(x))


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def cap_signal(signal: Signal, ceiling: Signal) -> Signal:
    """Return `signal`, but no better than `ceiling`."""
    return signal if SIGNAL_ORDER.index(signal) <= SIGNAL_ORDER.index(ceiling) else ceiling


def downgrade(signal: Signal) -> Signal:
    """One notch worse (Sell stays Sell)."""
    return SIGNAL_ORDER[max(0, SIGNAL_ORDER.index(signal) - 1)]


def target_upside_bounds(settings: Settings) -> tuple[float, float]:
    """(`fusion.min_buy_upside`, `fusion.max_sell_upside`), validated. A non-finite value is a config
    error and raises: it must never silently switch the target-gap guardrail off or on."""
    lo, hi = settings.fusion.min_buy_upside, settings.fusion.max_sell_upside
    if not (_finite(lo) and _finite(hi)):
        raise ValueError(f"fusion.min_buy_upside / fusion.max_sell_upside must be finite, got {lo!r} / {hi!r}")
    return float(lo), float(hi)


def _pct_pair(a: float, b: float) -> tuple[str, str]:
    """Format two signed percentages with the fewest decimals (1-3) at which they differ, so a note
    never reads "+2.0% < +2.0%" for 1.96% vs 2%."""
    for d in (1, 2, 3):
        sa, sb = f"{a:+.{d}%}", f"{b:+.{d}%}"
        if sa != sb:
            return sa, sb
    return sa, sb


def llm_available(llm: LLMIntelligence | None) -> bool:
    """The LLM pillar exists only when the LLM ran and produced news or filing intelligence."""
    return llm is not None and not llm.skipped and (llm.news is not None or llm.filings is not None)


def ml_available(quant: QuantSignal) -> bool:
    """The ML pillar exists only with an RSMT-Q distribution (it also carries the stock-selection model's mean)."""
    return quant.distribution is not None


def ml_skill_ok(quant: QuantSignal) -> bool:
    """Skill of the model that carries the ML pillar's direction: the stock-selection model's served skill (gate +
    untouched-name confirmation) when it is present; else the RSMT-Q CV direction gate (False without either)."""
    if quant.excess_model is not None:
        return bool(quant.excess_model.skill_ok)
    # the DIRECTION verdict: RSMT-Q's skill_ok is its distribution (CRPS / coverage) gate since 2026-09-29
    return bool(quant.distribution is not None and quant.distribution.direction_skill_ok)


def factor_pillar_blend_weight(settings: Settings) -> float:
    """`settings.backtest.factor_pillar_blend`, validated: a non-finite or out-of-range weight is a config error."""
    b = settings.backtest.factor_pillar_blend
    if not (_finite(b) and 0.0 <= float(b) <= 1.0):
        raise ValueError(f"backtest.factor_pillar_blend must be in [0, 1], got {b!r}")
    return float(b)


def factor_pillar_value(quant: QuantSignal, settings: Settings | None = None) -> tuple[float | None, str | None]:
    """(factor pillar, blend note or None).

    Without `quant.factor_model` (or without `settings`) this is the legacy composite pillar
    (`quant.factor_pillar`, None when no factor score exists) and no note. With the PIT multi-factor model
    it is `(1 - b)·legacy + b·factor_model.pillar`, b = `settings.backtest.factor_pillar_blend`, clipped to
    [-1, 1]; when the legacy composite is unavailable the model pillar is used alone. The note (prefix
    FACTOR_BLEND_PREFIX / FACTOR_BLEND_PREFIX_EN) states the arithmetic in the report language.
    """
    legacy = float(quant.factor_pillar) if quant.factor_composite is not None else None
    fm = quant.factor_model
    if fm is None or settings is None:
        return legacy, None
    b = factor_pillar_blend_weight(settings)
    model = float(fm.pillar)
    if not _finite(model):
        log.warning("factor model pillar not finite (%r) - legacy factor pillar only", fm.pillar)
        note = tr(
            settings,
            f"{FACTOR_BLEND_PREFIX} 팩터 모델 필러 값 이상({fm.pillar}) — 기존 합성만 사용",
            f"{FACTOR_BLEND_PREFIX_EN} factor model pillar invalid ({fm.pillar}) — legacy composite only",
        )
        return legacy, note
    skill = (
        ""
        if fm.skill_ok
        else tr(settings, " (모델 스킬 미달 — 모델 필러 ×0.35 반영)", " (model unskilled — model pillar ×0.35 applied)")
    )
    side = "neg" if model < 0 else "pos" if model > 0 else None
    if settings.fusion.factor_pillar_asymmetry == "auto" and side is not None and fm.weak_side == side:
        model *= FACTOR_WEAK_SIDE
        skill += tr(
            settings,
            f" (모델 {'음(−)' if side == 'neg' else '양(+)'}쪽 근거 약함 — ×{FACTOR_WEAK_SIDE:g})",
            f" (model's {'negative' if side == 'neg' else 'positive'} side is weak — ×{FACTOR_WEAK_SIDE:g})",
        )
    if legacy is None:
        value = _clip(model, -1.0, 1.0)
        note = tr(
            settings,
            f"{FACTOR_BLEND_PREFIX} 기존 합성 없음 → 팩터 모델 {_no_neg_zero(model):+.2f} 단독",
            f"{FACTOR_BLEND_PREFIX_EN} legacy composite unavailable → factor model {_no_neg_zero(model):+.2f} only",
        )
        return value, note + skill
    value = _clip((1.0 - b) * legacy + b * model, -1.0, 1.0)
    lg, md_, bl = _no_neg_zero(legacy), _no_neg_zero(model), _no_neg_zero(value)
    note = tr(
        settings,
        f"{FACTOR_BLEND_PREFIX} 기존 합성 {lg:+.2f}×{1 - b:.0%} + 팩터 모델 {md_:+.2f}×{b:.0%} = {bl:+.2f}",
        f"{FACTOR_BLEND_PREFIX_EN} legacy composite {lg:+.2f}×{1 - b:.0%} + factor model {md_:+.2f}×{b:.0%} "
        f"= {bl:+.2f}",
    )
    return value, note + skill


def selection_view(quant: QuantSignal) -> float | None:
    """The stock-selection model's relative call in [-1, 1] (see the module docstring), or None without one."""
    ex = quant.excess_model
    if ex is None:
        return None
    dist = quant.distribution
    parts: list[float] = []
    se = float(dist.sigma_excess) if dist is not None else float("nan")
    mu = float(ex.mu_excess)
    if math.isfinite(mu) and math.isfinite(se) and se > 0:
        parts.append(math.tanh(2.0 * mu / se))
    p = ex.p_beat_benchmark if ex.p_beat_benchmark is not None else (dist.prob_outperform if dist else None)
    if p is not None and math.isfinite(float(p)):
        parts.append(2.0 * float(p) - 1.0)
    if not parts:
        return None
    return _clip(sum(parts) / len(parts), -1.0, 1.0)


def ml_pillar_value(quant: QuantSignal, settings: Settings | None = None) -> tuple[float | None, str]:
    """(ML pillar value, informational note). Without a stock-selection model: the distribution pillar as is."""
    if not ml_available(quant):
        return None, ""
    base = float(quant.ml_pillar)
    sel = selection_view(quant)
    if sel is None:
        return base, ""
    ex = quant.excess_model
    w = 1.0 if ex.skill_ok else SELECTION_UNSKILLED
    b = SELECTION_BLEND
    value = _clip((1.0 - b) * base + b * w * sel, -1.0, 1.0)
    bs, sv, vv = _no_neg_zero(base), _no_neg_zero(sel), _no_neg_zero(value)
    status = tr(settings, "스킬 확인" if ex.skill_ok else "스킬 미확인", "skill ok" if ex.skill_ok else "no skill")
    if b >= 1.0:
        note = tr(
            settings,
            f"{ML_BLEND_PREFIX} 방향 = 종목 선택 모델({ex.model}, {status}) {sv:+.2f}×{w:g} = {vv:+.2f} "
            "(RSMT-Q 분포는 위험·확률 전용)",
            f"{ML_BLEND_PREFIX_EN} direction = stock-selection model ({ex.model}, {status}) {sv:+.2f}×{w:g} = "
            f"{vv:+.2f} (the RSMT-Q distribution is used for risk / probabilities only)",
        )
        return value, note
    note = tr(
        settings,
        f"{ML_BLEND_PREFIX} RSMT-Q 분포 {bs:+.2f}×{1 - b:.0%} + 종목 선택 모델({ex.model}, {status}) "
        f"{sv:+.2f}×{w:g}×{b:.0%} = {vv:+.2f}",
        f"{ML_BLEND_PREFIX_EN} RSMT-Q distribution {bs:+.2f}×{1 - b:.0%} + stock-selection model ({ex.model}, "
        f"{status}) {sv:+.2f}×{w:g}×{b:.0%} = {vv:+.2f}",
    )
    return value, note


def pillar_values(
    quant: QuantSignal, llm: LLMIntelligence | None, settings: Settings | None = None
) -> dict[str, float | None]:
    """Pillar value per name, or None when that pillar is unavailable (never a silent 0).

    With `settings` the factor pillar includes the factor-model blend (`factor_pillar_value`).
    """
    return {
        "factor": factor_pillar_value(quant, settings)[0],
        "ml": ml_pillar_value(quant, settings)[0],
        "llm": float(llm.llm_pillar) if (llm is not None and llm_available(llm)) else None,
    }


def pillar_weights(available: Iterable[str], settings: Settings) -> dict[str, float]:
    """`settings.fusion.weights` renormalised over the available pillars (empty if none has weight)."""
    raw = {p: max(0.0, float(settings.fusion.weights.get(p, 0.0))) for p in available}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {p: w / total for p, w in raw.items() if w > 0}


def pillar_coverage(available: Iterable[str], settings: Settings) -> float:
    """Configured weight share of the available pillars: Σ_{avail} W_p / Σ_{all pillars} W_p, in [0, 1].

    It uses the *configured* `settings.fusion.weights` (not the renormalised ones), so it is 1.0
    only when every pillar is available and e.g. 0.60 with the default weights when the LLM pillar
    (0.40) is missing. Keys in the config that are not pillars are ignored.
    """
    cfg = {p: max(0.0, float(settings.fusion.weights.get(p, 0.0))) for p in PILLARS}
    total = sum(cfg.values())
    if total <= 0:
        return 0.0
    return min(1.0, sum(cfg[p] for p in set(available) if p in cfg) / total)


def composite_score(pillars: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Σ w·pillar over weighted pillars, clipped to [-1, 1] and rounded to 1e-6 (stable threshold compares)."""
    if not weights:
        return 0.0
    c = sum(weights[p] * pillars[p] for p in weights)
    return round(_clip(c, -1.0, 1.0), 6)


def signal_from_composite(composite: float, confidence: int, settings: Settings) -> tuple[Signal, list[str]]:
    """Threshold mapping. Strong Buy requires `strong_buy_min_confidence`, otherwise it becomes Buy (noted)."""
    t = settings.fusion.thresholds
    notes: list[str] = []
    if composite >= t.strong_buy:
        min_conf = settings.fusion.strong_buy_min_confidence
        if confidence >= min_conf:
            return Signal.STRONG_BUY, notes
        notes.append(
            tr(
                settings,
                f"신뢰도 {confidence} < {min_conf} → Strong Buy를 Buy로 하향",
                f"Confidence {confidence} < {min_conf} → Strong Buy lowered to Buy",
            )
        )
        return Signal.BUY, notes
    if composite >= t.buy:
        return Signal.BUY, notes
    if composite <= t.sell:
        return Signal.SELL, notes
    return Signal.HOLD, notes


# ------------------------------------------------------------------------ confidence


def _agreement(
    pillars: Mapping[str, float],
    weights: Mapping[str, float],
    composite: float,
    coverage: float,
) -> float:
    """35 × coverage × (weighted share of pillars agreeing with the composite's sign; |p| < 0.05 counts
    half) × (1 − min(1, population std of the available pillars)).

    `weights` are the weights renormalised over the available pillars, so `coverage × share` equals
    Σ_{avail} W_p·s_p with the CONFIGURED weights W_p: a missing pillar contributes nothing, exactly
    like a present pillar that contradicts the call. Without the coverage factor a single available
    pillar always "agreed with itself" (share 1, std 0) and scored the full 35 points, more than
    three agreeing but dispersed pillars.
    """
    if not weights or coverage <= 0:
        return 0.0
    sign_c = float(np.sign(composite))
    share = 0.0
    for p, w in weights.items():
        v = pillars[p]
        if abs(v) < NEUTRAL_PILLAR_BAND:
            share += 0.5 * w
        elif float(np.sign(v)) == sign_c:
            share += w
    dispersion = float(np.std([pillars[p] for p in weights]))  # population std of available pillars
    return AGREEMENT_MAX * _clip(coverage, 0.0, 1.0) * share * (1.0 - min(1.0, dispersion))


def _robust_gate_share(cv: Mapping[str, float], pass_key: str = "gate_folds_pass") -> float:
    """<pass_key> / gate_folds_n from DistributionForecast.cv_metrics, clipped to [0, 1]; 0 if absent. Older models
    without the direction record fall back to gate_folds_pass."""
    n, k = cv.get("gate_folds_n"), cv.get(pass_key, cv.get("gate_folds_pass"))
    if not (_finite(n) and _finite(k)) or float(n) <= 0:
        return 0.0
    return _clip(float(k) / float(n), 0.0, 1.0)


def _ml_skill_component(quant: QuantSignal, lang: object = None) -> tuple[float, list[str]]:
    """Validated skill of the model behind the ML pillar's direction (0..ML_SKILL_MAX).

    Stock-selection model present: its out-of-sample within-date AUC (0.5 -> 0, 0.55 -> full) and rank IC (0 -> 0,
    0.10 -> full), times SELECTION_UNSKILLED when its skill is not established. Else the RSMT-Q CV skill (AUC /
    log-loss vs the baseline twin); 0 without either."""
    notes: list[str] = []
    ex = quant.excess_model
    if ex is not None:
        auc_part = _clip((float(ex.oos_auc) - 0.5) / 0.05, 0.0, 1.0) if _finite(ex.oos_auc) else 0.0
        ic_part = _clip(float(ex.oos_ic) / 0.1, 0.0, 1.0) if _finite(ex.oos_ic) else 0.0
        gate = 1.0 if ex.skill_ok else SELECTION_UNSKILLED
        return (ML_SKILL_MAX * auc_part * 0.5 + ML_SKILL_MAX * ic_part * 0.5) * gate, notes
    dist = quant.distribution
    if dist is None:
        return 0.0, notes
    cv = dist.cv_metrics or {}
    keys = ("auc", "base_auc", "logloss", "base_logloss")
    if not all(_finite(cv.get(k)) for k in keys):
        notes.append(
            tr(
                lang,
                "RSMT-Q 교차검증 지표 누락 — ML 스킬 점수 0 처리",
                "RSMT-Q cross-validation metrics missing — ML skill score set to 0",
            )
        )
        return 0.0, notes
    auc_part = _clip((cv["auc"] - cv["base_auc"]) / 0.05, 0.0, 1.0)
    ll_part = _clip((cv["base_logloss"] - cv["logloss"]) / 0.01, 0.0, 1.0)
    # auc/logloss are the PRIMARY layout's; direction_skill_ok is the fold-robust DIRECTION verdict (skill_ok is
    # the distribution gate). A rejected model earns only the share of layouts whose direction gate passed.
    gate = 1.0 if dist.direction_skill_ok else _robust_gate_share(cv, "direction_gate_folds_pass")
    return (ML_SKILL_MAX * auc_part * 0.5 + ML_SKILL_MAX * ll_part * 0.5) * gate, notes


def fundamentals_completeness(md: MarketData) -> float:
    """Share of (per, pbr, roe, analyst target) present, floored at COMPLETENESS_FLOOR."""
    f = md.fundamentals
    present = [_finite(v) for v in (f.per, f.pbr, f.roe, f.analyst_target_price)]
    return max(COMPLETENESS_FLOOR, sum(present) / len(present))


def confidence_breakdown(
    md: MarketData,
    quant: QuantSignal,
    llm: LLMIntelligence | None,
    pillars: Mapping[str, float],
    weights: Mapping[str, float],
    composite: float,
    *,
    coverage: float,
    lang: object = None,
) -> tuple[dict[str, float], list[str]]:
    """All confidence components (before clipping) + notes. Keys:

    agreement (0-35, scaled by `coverage` = pillar_coverage(...)), strength (0-20), ml_skill (0-20),
    data_quality (0-15), evidence (0-10), rsmt_degraded (0 or -10), critic_adjustment (-25..+5),
    total_unclipped (the sum of the other keys). `lang` (Settings / "ko" / "en") sets the notes' language.
    """
    ml_skill, notes = _ml_skill_component(quant, lang)
    dq = max(0.0, DATA_QUALITY_MAX - DATA_QUALITY_PER_WARNING * len(md.warnings)) * fundamentals_completeness(md)
    news = llm.news if (llm is not None and llm_available(llm)) else None
    evidence = EVIDENCE_MAX * min(1.0, news.n_used / EVIDENCE_FULL_ARTICLES) if news is not None else 0.0
    degraded = RSMT_DEGRADED_PENALTY if (quant.distribution is not None and quant.distribution.degraded) else 0.0
    critic_adj = 0.0
    if llm is not None and llm.critic is not None:
        critic_adj = float(min(CRITIC_ADJ_MAX, max(CRITIC_ADJ_MIN, llm.critic.confidence_adjustment)))
    parts = {
        "agreement": _agreement(pillars, weights, composite, coverage),
        "strength": STRENGTH_MAX * min(1.0, abs(composite) / STRENGTH_FULL_SCALE),
        "ml_skill": ml_skill,
        "data_quality": dq,
        "evidence": evidence,
        "rsmt_degraded": degraded,
        "critic_adjustment": critic_adj,
    }
    parts = {k: round(float(v), 4) for k, v in parts.items()}
    parts["total_unclipped"] = round(sum(parts.values()), 4)
    return parts, notes


# ---------------------------------------------------------------------- target price


def fair_per(md: MarketData) -> tuple[float | None, str]:
    """Reference PER: median of own-history PER (>= 1 year of positive values), else sector PER."""
    vh = md.valuation_history
    if vh is not None and not vh.empty and "per" in vh.columns:
        per = pd.to_numeric(vh["per"], errors="coerce")
        per = per[np.isfinite(per) & (per > 0)]
        if len(per) >= 2:
            span = pd.Timestamp(per.index.max()) - pd.Timestamp(per.index.min())
            if span >= pd.Timedelta(365, "D"):
                return float(per.median()), "history"
    sector = md.fundamentals.sector_per
    if _finite(sector) and sector > 0:
        return float(sector), "sector"
    return None, ""


def scale_annual_target(price: float, target: float, horizon_days: int) -> float:
    """Scale a 12-month price target to `horizon_days`.

    Spec form for H ≤ 252: `price·(1 + (target/price − 1)·H/252)`. Beyond 12 months that linear
    extrapolation goes NEGATIVE whenever target < price·(1 − 252/H) (e.g. H = 504 and a target below
    half the price). So for H > 252 the ratio is compounded instead, `price·(target/price)^(H/252)`,
    which is always > 0 and equals the linear form at H = 252.
    """
    ratio = target / price
    if horizon_days <= TRADING_DAYS:
        return price * (1.0 + (ratio - 1.0) * horizon_days / TRADING_DAYS)
    return price * ratio ** (horizon_days / TRADING_DAYS)


def target_price_components(md: MarketData, quant: QuantSignal, settings: Settings) -> dict[str, float]:
    """Target price components, each scaled to H = settings.fusion.target_horizon_days.

    - ml: `price·exp(μ·H/h)` with the RSMT-Q μ (an h-day LOG mean) when a distribution exists; absent otherwise.
    - analyst: `price·(1 + (target/price − 1)·H/252)` (12-month target, linearly scaled; may be < price).
    - valuation: fair = price·fairPER/PER, then `price·(1 + (fair/price − 1)·H/252)`; skipped when PER ≤ 0.
    The two 12-month components compound instead of scaling linearly when H > 252
    (see `scale_annual_target`), so no component is ever ≤ 0.
    """
    price = md.last_price
    H = settings.fusion.target_horizon_days
    comps: dict[str, float] = {}

    dist = quant.distribution
    if dist is not None and _finite(dist.mu) and dist.horizon_days > 0:
        comps["ml"] = price * math.exp(dist.mu * H / dist.horizon_days)

    target = md.fundamentals.analyst_target_price
    if _finite(target) and target > 0:
        comps["analyst"] = scale_annual_target(price, float(target), H)

    per = md.fundamentals.per
    ref, _ = fair_per(md)
    if _finite(per) and per > 0 and ref is not None:
        comps["valuation"] = scale_annual_target(price, price * ref / per, H)
    return comps


def blend_target_price(
    price: float, comps: Mapping[str, float], composite: float, settings: Settings
) -> tuple[float, bool]:
    """Blend components with `target_price_blend` renormalised over the present ones.

    Returns (target, used_fallback). The fallback `price·(1 + composite·0.1)` is flagged, never silent.
    """
    blend = settings.fusion.target_price_blend
    w = {k: float(blend.get(k, 0.0)) for k in comps if float(blend.get(k, 0.0)) > 0}
    total = sum(w.values())
    if total <= 0:
        return price * (1.0 + composite * NO_COMPONENT_TP_SCALE), True
    return sum(w[k] * comps[k] for k in w) / total, False


# ---------------------------------------------------------------------------- thesis


def _verdict(signal: Signal, lang: object = None) -> str:
    return f"{SIGNAL_EN[signal]} call" if is_en(lang) else f"{SIGNAL_KO[signal]} 판단"


def _verdict_clause(composite: float, signal: Signal, lang: object = None) -> str:
    """The exact '종합 +0.30, 매수 판단' ('composite +0.30, Buy call') clause of a deterministic thesis. The
    barrier guardrail matches on this whole clause, NOT on the bare verdict: "매수 판단." is a substring of
    "강력 매수 판단." ("Buy call." of "Strong Buy call."), so a bare match would rewrite the wrong verdict."""
    if is_en(lang):
        return f"composite {_no_neg_zero(composite):+.2f}, {_verdict(signal, lang)}"
    return f"종합 {_no_neg_zero(composite):+.2f}, {_verdict(signal)}"


def _no_neg_zero(v: float, digits: int = 2) -> float:
    """Round for display so a tiny negative (-0.004) prints as +0.00, not -0.00."""
    return round(float(v), digits) + 0.0


def build_thesis(
    quant: QuantSignal,
    signal: Signal,
    composite: float,
    pillars: Mapping[str, float | None],
    *,
    lang: object = None,
) -> str:
    """Deterministic 1-2 sentence thesis from the pillars (used when the critic gave none), in the report
    language (`lang`: Settings / "ko" / "en"; default Korean)."""
    en = is_en(lang)
    labels = PILLAR_LABELS_EN if en else PILLAR_LABELS
    parts = []
    for key in PILLARS:
        v = pillars.get(key)
        label = labels[key]
        missing = "n/a" if en else "미반영"
        parts.append(f"{label} {_no_neg_zero(v):+.2f}" if v is not None else f"{label} {missing}")
    first = f"{' · '.join(parts)} → {_verdict_clause(composite, signal, lang)}."

    extras: list[str] = []
    scored = [f for f in quant.factors if f.score is not None]
    if scored:
        best = max(scored, key=lambda f: f.score)
        worst = min(scored, key=lambda f: f.score)
        if en:
            extras.append(f"{_FACTOR_NAMES_EN.get(best.name, best.label)} ({best.score:.0f}) strong")
        else:
            extras.append(f"{best.label}({best.score:.0f}) 우위")
        if worst.name != best.name:
            if en:
                extras.append(f"{_FACTOR_NAMES_EN.get(worst.name, worst.label)} ({worst.score:.0f}) weak")
            else:
                extras.append(f"{worst.label}({worst.score:.0f}) 약점")
    dist = quant.distribution
    if dist is not None:
        extras.append(
            f"RSMT-Q {dist.horizon_days}D P(up) {dist.prob_up:.0%}"
            if en
            else f"RSMT-Q {dist.horizon_days}일 상승확률 {dist.prob_up:.0%}"
        )
    return first if not extras else f"{first} {', '.join(extras)}."


# ------------------------------------------------------------------------------ fuse


def _missing_pillar_note(name: str, llm: LLMIntelligence | None, lang: object = None) -> str:
    if name == "llm":
        if llm is None or llm.skipped:
            return tr(
                lang,
                "LLM 분석 미실행 — LLM 필러 제외 후 팩터·ML 가중치 재정규화",
                "LLM analysis not run — LLM pillar dropped, factor/ML weights renormalised",
            )
        return tr(
            lang,
            "LLM 텍스트(뉴스·공시) 분석 결과 없음 — LLM 필러 제외 후 가중치 재정규화",
            "No LLM text (news/filings) result — LLM pillar dropped, weights renormalised",
        )
    if name == "ml":
        return tr(
            lang,
            "ML/RSMT-Q 예측 없음 — ML 필러 제외 후 가중치 재정규화",
            "No ML/RSMT-Q forecast — ML pillar dropped, weights renormalised",
        )
    return tr(
        lang,
        "팩터 점수 산출 불가 — 팩터 필러 제외 후 가중치 재정규화",
        "Factor scores unavailable — factor pillar dropped, weights renormalised",
    )


def fuse(
    md: MarketData,
    quant: QuantSignal,
    llm: LLMIntelligence | None,
    settings: Settings,
) -> ExecutiveSignal:
    """Fuse the quant signal and the (optional) LLM intelligence into the executive call.

    `pillars` carries ONLY the available pillars. An unavailable pillar is left out of the dict, as it
    is left out of the composite, the agreement term and the weights, and an explanatory Korean note
    is appended to `overrides`. It is never written as 0.0: report/render.py shows a missing key as
    "—" with no bar but a 0.0 as a neutral "0.00" bar, and the critic prompt would read a 0.0 as neutral
    text evidence.
    """
    values = pillar_values(quant, llm, settings)
    _, blend_note = factor_pillar_value(quant, settings)
    _, ml_note = ml_pillar_value(quant, settings)
    avail = {k: v for k, v in values.items() if v is not None}
    weights = pillar_weights(avail.keys(), settings)
    composite = composite_score(avail, weights)
    coverage = pillar_coverage(weights.keys(), settings)

    guard: list[str] = []  # guardrail / risk notes (shown first)
    info: list[str] = []  # degradation notes
    for name in PILLARS:
        if values[name] is None:
            info.append(_missing_pillar_note(name, llm, settings))
    if info and weights and coverage < 1.0:
        # one clause on the last missing-pillar note (every override is a chip in the matrix)
        info[-1] = tr(
            settings,
            f"{info[-1]} · 합의도 점수는 가용 필러 비중 {coverage:.0%}만 반영",
            f"{info[-1]} · agreement score counts only the {coverage:.0%} weight share of available pillars",
        )
    if not weights:
        log.warning("%s: no pillar available - composite forced to 0", md.instrument.ticker)
        info.append(
            tr(
                settings,
                "사용 가능한 필러 없음 — 종합점수 0(중립) 처리",
                "No pillar available — composite set to 0 (neutral)",
            )
        )
    if blend_note:
        info.append(blend_note)
    if ml_note:
        info.append(ml_note)

    breakdown, cb_notes = confidence_breakdown(
        md, quant, llm, avail, weights, composite, coverage=coverage, lang=settings
    )
    info.extend(cb_notes)
    dist = quant.distribution
    if dist is not None and dist.degraded:
        reasons = ", ".join(dist.degraded_reasons) or tr(settings, "사유 미상", "reason unknown")
        guard.append(tr(settings, f"RSMT 저하 모드: {reasons}", f"RSMT degraded mode: {reasons}"))
        log.warning(
            "%s: RSMT-Q degraded (%s) - confidence %+.0f",
            md.instrument.ticker,
            reasons,
            RSMT_DEGRADED_PENALTY,
        )
    confidence = int(round(_clip(breakdown["total_unclipped"], CONFIDENCE_FLOOR, CONFIDENCE_CEIL)))

    signal, notes = signal_from_composite(composite, confidence, settings)
    guard.extend(notes)

    # ---- guardrail 1: critical event (critic flag or a critical risk factor)
    critical_reasons: list[str] = []
    if llm is not None and llm.critic is not None and llm.critic.critical_flag:
        critical_reasons.append(
            llm.critic.critical_reason or tr(settings, "critic 치명 이벤트 플래그", "critic critical-event flag")
        )
    if llm is not None and llm.filings is not None:
        critical_reasons.extend(rf.title for rf in llm.filings.risk_factors if rf.severity == "critical")
    if critical_reasons:
        before = signal
        signal = Signal.SELL if composite < 0 else cap_signal(signal, Signal.HOLD)
        change = (
            f"{before.value}→{signal.value}"
            if signal != before
            else tr(settings, f"{signal.value} 유지", f"{signal.value} kept")
        )
        guard.append(
            tr(
                settings,
                f"치명적 리스크 가드레일({'; '.join(critical_reasons)}): {change}",
                f"Critical-risk guardrail ({'; '.join(critical_reasons)}): {change}",
            )
        )

    # ---- guardrail 2: falling knife
    ret_1m = quant.technicals.ret_1m
    trend = next((f.score for f in quant.factors if f.name == "trend"), None)
    llm_p = values["llm"] if values["llm"] is not None else 0.0
    if signal in BUYISH and _finite(ret_1m) and ret_1m < FALLING_KNIFE_RET_1M:
        # an unknown trend score cannot clear the stock: treated as weak (conservative)
        trend_weak = trend is None or trend < FALLING_KNIFE_TREND_MAX
        if trend_weak and not llm_p > FALLING_KNIFE_LLM_OVERRIDE:
            before = signal
            signal = cap_signal(signal, Signal.HOLD)
            if trend is None:
                trend_txt = tr(settings, "추세 점수 없음", "no trend score")
            else:
                trend_txt = tr(settings, f"추세 점수 {trend:.0f}", f"trend score {trend:.0f}")
            guard.append(
                tr(
                    settings,
                    f"낙하하는 칼 가드레일: 1개월 수익률 {ret_1m:+.0%}, {trend_txt} → {before.value}→{signal.value}",
                    f"Falling-knife guardrail: 1M return {ret_1m:+.0%}, {trend_txt} → {before.value}→{signal.value}",
                )
            )

    # ---- guardrail 3: no independent text confirmation and an unskilled model
    if signal == Signal.STRONG_BUY and not llm_available(llm) and not ml_skill_ok(quant):
        signal = Signal.BUY
        guard.append(
            tr(
                settings,
                "LLM 미반영 + ML 검증 스킬 미달 → Strong Buy를 Buy로 제한",
                "No LLM input + ML skill gate not passed → Strong Buy capped at Buy",
            )
        )

    # ---- target price
    price = md.last_price
    comps = target_price_components(md, quant, settings)
    target, fallback = blend_target_price(price, comps, composite, settings)
    if fallback:
        info.append(
            tr(
                settings,
                "목표가 산출 근거(ML·컨센서스·밸류에이션) 없음 — 종합점수 기반 근사치",
                "No target-price basis (ML/consensus/valuation) — composite-based approximation",
            )
        )
        log.warning("%s: no target price component - composite fallback", md.instrument.ticker)

    # ---- guardrail 4 (symmetric target gap). The composite (factor/ML/LLM pillars) and the target blend
    # (ML/consensus/valuation) can disagree. "Buy, target +0.5%" and "Sell, target +12%" both contradict
    # themselves, so either side falls back to Hold with a note:
    #  - Buy/Strong Buy needs upside >= fusion.min_buy_upside (a NaN upside also fails: conservative);
    #  - Sell with upside > fusion.max_sell_upside is lifted to Hold, EXCEPT a critical-event Sell
    #    (guardrail 1). A consensus target that has not caught up with fraud or a going-concern flag is
    #    no reason to hold.
    upside = float(target / price - 1.0)
    min_buy, max_sell = target_upside_bounds(settings)
    if signal in BUYISH and not upside >= min_buy:
        before = signal
        signal = Signal.HOLD
        up_txt, min_txt = _pct_pair(upside, min_buy)
        guard.append(
            tr(
                settings,
                f"{TARGET_NOTE_PREFIX} 목표가 상승여력 {up_txt} < 매수 최소 상승여력 {min_txt} "
                f"(종합점수 {composite:+.2f}와 불일치) → {before.value}→{signal.value}",
                f"{TARGET_NOTE_PREFIX_EN} target upside {up_txt} < minimum buy upside {min_txt} "
                f"(inconsistent with composite {composite:+.2f}) → {before.value}→{signal.value}",
            )
        )
        log.info(
            "%s: %s with target upside %.4f < min_buy_upside %.4f capped at Hold",
            md.instrument.ticker,
            before.value,
            upside,
            min_buy,
        )
    elif signal == Signal.SELL and not critical_reasons and upside > max_sell:
        signal = Signal.HOLD
        up_txt, max_txt = _pct_pair(upside, max_sell)
        guard.append(
            tr(
                settings,
                f"{TARGET_NOTE_PREFIX} 매도 신호이나 목표가 상승여력 {up_txt} > 매도 허용 상한 {max_txt} "
                f"(종합점수 {composite:+.2f}와 상충 — 근거 불일치) → {Signal.SELL.value}→{signal.value}",
                f"{TARGET_NOTE_PREFIX_EN} Sell signal but target upside {up_txt} > sell ceiling {max_txt} "
                f"(conflicts with composite {composite:+.2f} — evidence disagrees) "
                f"→ {Signal.SELL.value}→{signal.value}",
            )
        )
        log.info(
            "%s: Sell with target upside %.4f > max_sell_upside %.4f lifted to Hold",
            md.instrument.ticker,
            upside,
            max_sell,
        )

    thesis = ""
    if llm is not None and llm.critic is not None and llm.critic.thesis.strip():
        thesis = llm.critic.thesis.strip()
    if not thesis:
        thesis = build_thesis(quant, signal, composite, values, lang=settings)

    return ExecutiveSignal(
        signal=signal,
        composite=composite,
        confidence=confidence,
        target_price=float(target),
        upside=upside,
        horizon_days=settings.fusion.target_horizon_days,
        horizon_label=horizon_label(signal, settings),
        pillars={k: v for k, v in values.items() if v is not None},
        target_components={k: float(v) for k, v in comps.items()},
        confidence_breakdown=breakdown,
        overrides=guard + info,
        thesis=thesis,
    )


# ------------------------------------------------------------------ barrier guardrail


def barrier_guard_margin(settings: Settings) -> float:
    """`settings.fusion.barrier_guard_margin`, validated: a non-finite or out-of-range margin is a config
    error and raises (it must never silently disable or hair-trigger the guardrail)."""
    m = settings.fusion.barrier_guard_margin
    if not (_finite(m) and 0.0 <= float(m) < 1.0):
        raise ValueError(f"fusion.barrier_guard_margin must be in [0, 1), got {m!r}")
    return float(m)


def apply_barrier_guardrail(
    execsig: ExecutiveSignal,
    plan: ExecutionPlan,
    settings: Settings,
    *,
    margin: float | None = None,
) -> ExecutiveSignal:
    """Downgrade a Buy-or-better call by one notch when P(SL first) > P(TP first) + margin.

    `margin` defaults to `settings.fusion.barrier_guard_margin` (the keyword only overrides it for
    what-if checks). The probabilities come from the same RSMT-Q path matrix that produced the signal
    (plan.barrier), simulated over the holding horizon `plan.barrier.horizon_days`, which the Korean
    override note states. Returns a new ExecutiveSignal; the input is not mutated. Without a barrier
    this is a no-op.
    The thesis is kept consistent: a deterministic thesis gets its verdict rewritten, a critic
    thesis (written for the pre-guardrail call) gets a short downgrade clause appended.
    Note: the plan's raw_weight/action were computed for the pre-guardrail signal; `size_portfolio`
    re-checks the final signal and zeroes non-Buy weights.
    Idempotent: a signal that already carries the barrier note is returned unchanged, so rebuilding
    the plan and applying the guardrail again cannot demote the call a second notch.
    """
    if not isinstance(settings, Settings):
        raise TypeError(
            "apply_barrier_guardrail(execsig, plan, settings): settings must be a Settings "
            f"(the margin comes from settings.fusion.barrier_guard_margin), got {type(settings).__name__}"
        )
    if margin is None:
        margin = barrier_guard_margin(settings)
    elif not (_finite(margin) and 0.0 <= float(margin) < 1.0):
        raise ValueError(f"barrier guard margin must be in [0, 1), got {margin!r}")
    b = plan.barrier
    if b is None or execsig.signal not in BUYISH:
        return execsig
    if any(o.startswith((BARRIER_NOTE_PREFIX, BARRIER_NOTE_PREFIX_EN)) for o in execsig.overrides):
        return execsig
    if b.p_stop_loss > b.p_take_profit + margin + _EPS:
        old, new = execsig.signal, downgrade(execsig.signal)
        en = is_en(settings)
        thesis = execsig.thesis
        clause = _verdict_clause(execsig.composite, old, lang_of(settings)) + "."
        if en:
            basis = f"{b.horizon_days}-trading-day holding simulation: " if b.horizon_days > 0 else ""
            note = (
                f"{BARRIER_NOTE_PREFIX_EN} {basis}P(stop first) {b.p_stop_loss:.0%} > P(take-profit first) "
                f"{b.p_take_profit:.0%} + {margin * 100:g}pp (time exit {b.p_time_exit:.0%}) "
                f"→ {old.value}→{new.value}"
            )
            rewritten = f"{_verdict_clause(execsig.composite, new, 'en')} (lowered by the barrier guardrail)."
            appended = f" (barrier guardrail: {SIGNAL_EN[old]}→{SIGNAL_EN[new]} downgrade)"
        else:
            basis = f"{b.horizon_days}거래일 보유 시뮬레이션 기준 " if b.horizon_days > 0 else ""
            note = (
                f"{BARRIER_NOTE_PREFIX} {basis}P(손절 선행) {b.p_stop_loss:.0%} > P(익절 선행) "
                f"{b.p_take_profit:.0%} + {margin * 100:g}%p (시간청산 {b.p_time_exit:.0%}) "
                f"→ {old.value}→{new.value}"
            )
            rewritten = f"{_verdict_clause(execsig.composite, new)}(배리어 가드레일로 하향)."
            appended = f" (배리어 가드레일: {SIGNAL_KO[old]}→{SIGNAL_KO[new]} 하향)"
        if clause in thesis:  # deterministic thesis: rewrite its verdict
            thesis = thesis.replace(clause, rewritten, 1)
        else:  # critic (free-text) thesis: never edit its words, append a clause
            thesis = f"{thesis.rstrip()}{appended}".strip()
        return execsig.model_copy(
            update={
                "signal": new,
                "horizon_label": horizon_label(new, settings),
                "thesis": thesis,
                "overrides": [note, *execsig.overrides],
            }
        )
    return execsig
