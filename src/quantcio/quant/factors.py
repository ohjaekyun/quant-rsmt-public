"""Five 0..100 factor scores (momentum, value, volatility, quality, trend) + composite and pillar.

Scoring contract (ARCHITECTURE.md, section quant/factors.py):
- Each sub-metric maps its raw input to 0..100 with a documented monotone function: a logistic
  ``100 / (1 + exp(-(x - c) / s))`` (s < 0 means lower is better), or a percentile against the
  stock's own history using only data <= t (t = last price date). RSI is the one deliberate
  exception required by the spec: a unimodal piecewise-linear map (non-decreasing up to 50, flat on
  50..65, non-increasing above 65).
- A missing / NaN / non-finite input gives a missing sub-score; the factor score is the mean of the
  available sub-scores and ``None`` when none is available (explicit, never a silent 50).
- ``metrics`` holds every raw input (None when missing) so the report can show what was used.
- ``rationale`` is one deterministic Korean sentence from the strongest and weakest sub-metric.

Calibration (x -> score):

=============  =======================================  ==========================================
factor         sub-metric (metrics key)                 mapping
=============  =======================================  ==========================================
momentum       ret_12m_ex1m                             logistic c=0, s=0.20
               ret_3m                                   logistic c=0, s=0.10
               rel_strength_3m                          logistic c=0, s=0.07
               dist_52w_high = price/high_52w - 1       logistic through (0, 80), (-0.30, 20):
                                                        c=-0.15, s=0.1082
value          per_pct_own (PER pct vs own history)     100 * (1 - pct), history >= 1y, >= 12 obs
               pbr_pct_own (PBR pct vs own history)     100 * (1 - pct), same rules
               per_rel_sector = PER / sector_per        logistic on ln(ratio), c=0, s=-0.3685:
                                                        1.0 -> 50, 0.6 -> 80, 1.5 -> 25
               analyst_upside = target/price - 1        logistic c=0, s=0.15
volatility     realized_vol_60d                         logistic through (0.15, 85), (0.60, 15):
                                                        c=0.375, s=-0.1297
               vol_pct_3y (features.vol_pct_3y)         100 * (1 - pct)
               beta_1y                                  logistic on |beta - 0.8|, c=0.5, s=-0.25
                                                        (penalised low beta reads "시장 연동 낮음")
               max_drawdown_1y                          logistic through (-0.10, 80), (-0.50, 15):
                                                        c=-0.2777, s=0.1282
quality        roe                                      logistic c=0.10, s=0.06
               operating_margin                         logistic c=0.10, s=0.07
               earnings_growth_yoy                      logistic c=0,    s=0.20
               revenue_growth_yoy                       logistic c=0.05, s=0.10
trend          close_vs_ma50 = close/MA50 - 1           logistic c=0, s=0.05
               close_vs_ma200 = close/MA200 - 1         logistic c=0, s=0.10
               ma50_vs_ma200 = MA50/MA200 - 1           logistic c=0, s=0.03 (soft MA50 > MA200)
               ma200_slope_20d = MA200_t/MA200_{t-20}-1 logistic c=0, s=0.015
               rsi14                                    piecewise-linear knots (0,30) (30,45)
                                                        (50,85) (65,85) (75,50) (100,5)
=============  =======================================  ==========================================

A PER that is None or <= 0 is treated as loss-making: both PER metrics are skipped and the rationale
says so. Own-history percentiles use the mid-rank definition
``(#{h < v} + 0.5 * #{h == v}) / N`` over positive history values dated <= t.

Labels, rationales and notes follow ``settings.report.language`` ("ko" default, "en"): every sub-score
function takes an optional ``lang`` keyword; numbers are formatted identically in both languages.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantcio.config import Settings
from quantcio.i18n import is_en, lang_of
from quantcio.models import FactorScore, Fundamentals, MarketData, Technicals
from quantcio.quant.features import regime_features
from quantcio.quant.indicators import MA_LONG, valid_prices

log = logging.getLogger(__name__)

FACTOR_ORDER: tuple[str, ...] = ("momentum", "value", "volatility", "quality", "trend")
FACTOR_LABELS: dict[str, str] = {
    "momentum": "모멘텀",
    "value": "밸류에이션",
    "volatility": "변동성(안정성)",
    "quality": "퀄리티",
    "trend": "추세",
}
FACTOR_LABELS_EN: dict[str, str] = {
    "momentum": "Momentum",
    "value": "Value",
    "volatility": "Volatility (stability)",
    "quality": "Quality",
    "trend": "Trend",
}


def factor_label(name: str, lang: object = None) -> str:
    return (FACTOR_LABELS_EN if is_en(lang) else FACTOR_LABELS)[name]


VALUATION_MIN_SPAN = pd.Timedelta(365, "D")  # own-history percentile needs >= 1 year of data
VALUATION_MIN_OBS = 12  # ... and at least monthly density
MA200_SLOPE_LAG = 20
BETA_TARGET = 0.8

__all__ = [
    "FACTOR_ORDER",
    "FACTOR_LABELS",
    "FACTOR_LABELS_EN",
    "factor_label",
    "Logistic",
    "PiecewiseLinear",
    "SubScore",
    "own_percentile",
    "momentum_subscores",
    "value_subscores",
    "volatility_subscores",
    "quality_subscores",
    "trend_subscores",
    "compute_factors",
    "factor_composite",
    "factor_pillar",
]


# ---------------------------------------------------------------------------- mappings


def _finite(x: object) -> float | None:
    """float(x) when it is a finite number, else None (None/NaN/inf/non-numeric)."""
    if x is None:
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass(frozen=True)
class Logistic:
    """score(x) = 100 / (1 + exp(-(x - c) / s)); increasing when s > 0, decreasing when s < 0.

    Output is in [0, 100] for every finite x (computed in an overflow-safe form); None for a missing
    or non-finite input.
    """

    c: float
    s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.s) or self.s == 0:
            raise ValueError(f"Logistic scale must be finite and non-zero, got {self.s}")

    @classmethod
    def from_anchors(cls, a: tuple[float, float], b: tuple[float, float]) -> Logistic:
        """The logistic passing exactly through two (x, score) anchor points (0 < score < 100)."""
        (x1, y1), (x2, y2) = a, b
        l1, l2 = math.log(y1 / (100.0 - y1)), math.log(y2 / (100.0 - y2))
        s = (x1 - x2) / (l1 - l2)
        return cls(c=x1 - s * l1, s=s)

    def __call__(self, x: object) -> float | None:
        v = _finite(x)
        if v is None:
            return None
        z = (v - self.c) / self.s
        if z >= 0:
            return 100.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return 100.0 * ez / (1.0 + ez)


@dataclass(frozen=True)
class PiecewiseLinear:
    """Linear interpolation through (x, score) knots, flat beyond the end knots; None for missing x."""

    knots: tuple[tuple[float, float], ...]

    def __call__(self, x: object) -> float | None:
        v = _finite(x)
        if v is None:
            return None
        xs, ys = zip(*self.knots)
        return float(np.clip(np.interp(v, xs, ys), 0.0, 100.0))


def own_percentile(value: object, history: Iterable[float]) -> float | None:
    """Mid-rank percentile of `value` within `history` in [0, 1]: (#{h < v} + 0.5 #{h == v}) / N.

    Non-finite history values are ignored; None when `value` is missing or the history is empty.
    """
    v = _finite(value)
    h = np.asarray(list(history), dtype="float64")
    h = h[np.isfinite(h)]
    if v is None or h.size == 0:
        return None
    return float(((h < v).sum() + 0.5 * (h == v).sum()) / h.size)


# momentum
MOM_12_1 = Logistic(c=0.0, s=0.20)
MOM_3M = Logistic(c=0.0, s=0.10)
MOM_REL_3M = Logistic(c=0.0, s=0.07)
MOM_52W_HIGH = Logistic.from_anchors((0.0, 80.0), (-0.30, 20.0))
# value (applied to ln(PER / sector_per))
VAL_PER_REL_SECTOR = Logistic.from_anchors((0.0, 50.0), (math.log(0.6), 80.0))
VAL_ANALYST_UPSIDE = Logistic(c=0.0, s=0.15)
# volatility
VOL_ABS = Logistic.from_anchors((0.15, 85.0), (0.60, 15.0))
VOL_BETA_DEV = Logistic(c=0.5, s=-0.25)  # applied to |beta - 0.8|
VOL_MDD = Logistic.from_anchors((-0.10, 80.0), (-0.50, 15.0))
# quality
QUAL_ROE = Logistic(c=0.10, s=0.06)
QUAL_OPM = Logistic(c=0.10, s=0.07)
QUAL_EARN_G = Logistic(c=0.0, s=0.20)
QUAL_REV_G = Logistic(c=0.05, s=0.10)
# trend
TREND_MA50 = Logistic(c=0.0, s=0.05)
TREND_MA200 = Logistic(c=0.0, s=0.10)
TREND_CROSS = Logistic(c=0.0, s=0.03)
TREND_SLOPE = Logistic(c=0.0, s=0.015)
TREND_RSI = PiecewiseLinear(((0, 30), (30, 45), (50, 85), (65, 85), (75, 50), (100, 5)))


def _pct_to_score(pct: float | None) -> float | None:
    """Lower-is-better percentile (0..1) -> 0..100 score."""
    return None if pct is None else float(np.clip(100.0 * (1.0 - pct), 0.0, 100.0))


# --------------------------------------------------------------------------- formatting


def _fmt_pct(x: float, digits: int = 0, signed: bool = True, unit: str = "%") -> str:
    """Percent text: +34%, -3%, +0.4% (one decimal below 1%), never '-0'."""
    v = x * 100.0
    if digits == 0 and abs(v) < 1.0:
        digits = 1
    v = round(v, digits)
    if v == 0:
        v = 0.0
    return f"{v:+.{digits}f}{unit}" if signed else f"{v:.{digits}f}{unit}"


# final consonant of the Korean reading of the last digit: 0 영/십, 3 삼, 6 육 take '으로'
_DIGIT_PARTICLE = {"0": "으로", "3": "으로", "6": "으로"}


def _ro(text: str) -> str:
    """Instrumental particle 로/으로 for `text` (digits read in Korean; %, 배, ) read as 로)."""
    ch = text.rstrip()[-1:] or "%"
    if ch.isdigit():
        return _DIGIT_PARTICLE.get(ch, "로")
    if "가" <= ch <= "힣":
        jong = (ord(ch) - 0xAC00) % 28
        return "로" if jong in (0, 8) else "으로"  # no final consonant or ㄹ -> 로
    return "로"


@dataclass(frozen=True)
class SubScore:
    """One sub-metric: raw input, 0..100 score (None when missing) and its Korean descriptor."""

    key: str  # metrics key, e.g. "ret_12m_ex1m"
    value: float | None  # raw input (decimal / ratio)
    score: float | None  # 0..100, None when the input is missing
    text: str = ""  # e.g. "12-1M 수익률 +34%" ("" when missing)
    tone: str | None = None  # overrides the factor's generic tone word (used for RSI)


def _sub(
    key: str,
    value: float | None,
    score: float | None,
    text: str,
    tone: str | None = None,
) -> SubScore:
    if value is None or score is None:
        return SubScore(key=key, value=value, score=None, text="", tone=None)
    return SubScore(
        key=key,
        value=value,
        score=float(np.clip(score, 0.0, 100.0)),
        text=text,
        tone=tone,
    )


_TONES: dict[str, tuple[str, str]] = {
    "momentum": ("강세", "약세"),
    "value": ("저평가", "고평가"),
    "volatility": ("안정적", "불안정"),
    "quality": ("우수", "부진"),
    "trend": ("상승 추세", "하락 추세"),
}
_NEUTRAL = "중립"
_TONES_EN: dict[str, tuple[str, str]] = {
    "momentum": ("strong", "weak"),
    "value": ("cheap", "expensive"),
    "volatility": ("stable", "unstable"),
    "quality": ("strong", "weak"),
    "trend": ("uptrend", "downtrend"),
}
_NEUTRAL_EN = "neutral"


def _tone(factor: str, sub: SubScore, lang: object = None) -> str:
    if sub.tone:
        return sub.tone
    pos, neg = (_TONES_EN if is_en(lang) else _TONES)[factor]
    assert sub.score is not None
    if sub.score >= 55.0:
        return pos
    if sub.score <= 45.0:
        return neg
    return _NEUTRAL_EN if is_en(lang) else _NEUTRAL


def _rationale(factor: str, subs: Sequence[SubScore], notes: Sequence[str], lang: object = None) -> str:
    """'<strongest>로 <tone>, <weakest>로 <tone>' (+ notes); tone omitted on the second clause if equal.

    English: '<strongest> (<tone>), <weakest> (<tone>)' (+ notes), both tones always stated.
    """
    en = is_en(lang)
    avail = [s for s in subs if s.score is not None]
    if not avail:
        text = "not computable: no usable inputs" if en else "산출 불가: 사용 가능한 입력 없음"
        return text + (f" ({', '.join(notes)})" if notes else "")
    strong = max(avail, key=lambda s: s.score)  # first maximum in definition order
    weak = min(avail, key=lambda s: s.score)  # first minimum in definition order
    strong_tone = _tone(factor, strong, lang)
    if en:
        sentence = f"{strong.text} ({strong_tone})"
        if weak is not strong:
            sentence += f", {weak.text} ({_tone(factor, weak, lang)})"
    else:
        sentence = f"{strong.text}{_ro(strong.text)} {strong_tone}"
        if weak is not strong:
            weak_tone = _tone(factor, weak)
            sentence += f", {weak.text}" + (f"{_ro(weak.text)} {weak_tone}" if weak_tone != strong_tone else "")
    extra = list(notes)
    if len(avail) < len(subs):
        extra.append(f"{len(avail)}/{len(subs)} inputs used" if en else f"{len(avail)}/{len(subs)} 지표 반영")
    return sentence + (f" ({', '.join(extra)})" if extra else "")


# --------------------------------------------------------------------------- sub-scores


def momentum_subscores(tech: Technicals, *, lang: object = None) -> list[SubScore]:
    """12-1M return, 3M return, 3M relative strength, proximity to the 52-week high."""
    en = is_en(lang)
    r12 = _finite(tech.ret_12m_ex1m)
    r3 = _finite(tech.ret_3m)
    rel = _finite(tech.rel_strength_3m)
    high, last = _finite(tech.high_52w), _finite(tech.last_close)
    prox = last / high - 1.0 if high and high > 0 and last is not None else None
    return [
        _sub(
            "ret_12m_ex1m",
            r12,
            MOM_12_1(r12),
            f"12-1M return {_fmt_pct(r12 or 0.0)}" if en else f"12-1M 수익률 {_fmt_pct(r12 or 0.0)}",
        ),
        _sub(
            "ret_3m",
            r3,
            MOM_3M(r3),
            f"3M return {_fmt_pct(r3 or 0.0)}" if en else f"3개월 수익률 {_fmt_pct(r3 or 0.0)}",
        ),
        _sub(
            "rel_strength_3m",
            rel,
            MOM_REL_3M(rel),
            f"3M vs benchmark {_fmt_pct(rel or 0.0, unit='pp')}"
            if en
            else f"벤치마크 대비 3개월 {_fmt_pct(rel or 0.0, unit='%p')}",
        ),
        _sub(
            "dist_52w_high",
            prox,
            MOM_52W_HIGH(prox),
            f"vs 52-week high {_fmt_pct(prox or 0.0)}" if en else f"52주 고점 대비 {_fmt_pct(prox or 0.0)}",
        ),
    ]


def _own_history(history: pd.DataFrame | None, column: str, as_of: pd.Timestamp | None) -> pd.Series | None:
    """Positive, finite `column` values dated <= as_of; None unless they span >= 1 year and >= 12 obs."""
    if history is None or as_of is None or history.empty or column not in history.columns:
        return None
    values = pd.to_numeric(history[column], errors="coerce").to_numpy(dtype="float64")
    idx = pd.DatetimeIndex(pd.to_datetime(history.index, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(values, index=idx.normalize())
    s = s[s.index.notna() & (s.index <= as_of)]
    s = s[np.isfinite(s.to_numpy()) & (s.to_numpy() > 0)]
    if len(s) < VALUATION_MIN_OBS or (s.index.max() - s.index.min()) < VALUATION_MIN_SPAN:
        return None
    return s


def value_subscores(
    md: MarketData, tech: Technicals, *, lang: object = None
) -> tuple[list[SubScore], list[str], dict[str, float | None]]:
    """PER/PBR own-history percentiles, PER vs sector PER, analyst upside (all: cheaper = higher).

    Returns (sub-scores, rationale notes, extra raw metrics {"per", "pbr"}).
    """
    en = is_en(lang)
    fund: Fundamentals = md.fundamentals
    notes: list[str] = []
    per, pbr = _finite(fund.per), _finite(fund.pbr)
    price = _finite(tech.last_close)
    as_of = _as_of(md)

    loss_making = per is None or per <= 0
    if per is None:
        notes.append("PER missing/loss-making: PER metrics excluded" if en else "PER 미제공/적자: PER 지표 제외")
    elif per <= 0:
        notes.append("loss-making (PER≤0): PER metrics excluded" if en else "적자(PER≤0): PER 지표 제외")
    if pbr is not None and pbr <= 0:
        notes.append("PBR≤0: PBR metric excluded" if en else "PBR≤0: PBR 지표 제외")
    pbr_ok = pbr is not None and pbr > 0

    per_pct = pbr_pct = None
    per_hist = _own_history(md.valuation_history, "per", as_of) if not loss_making else None
    pbr_hist = _own_history(md.valuation_history, "pbr", as_of) if pbr_ok else None
    if per_hist is not None:
        per_pct = own_percentile(per, per_hist)
    if pbr_hist is not None:
        pbr_pct = own_percentile(pbr, pbr_hist)
    if (not loss_making and per_hist is None) or (pbr_ok and pbr_hist is None):
        notes.append("valuation history < 1 year / none" if en else "밸류에이션 과거 이력 1년 미만/없음")

    sector = _finite(fund.sector_per)
    rel = per / sector if not loss_making and sector is not None and sector > 0 else None
    target = _finite(fund.analyst_target_price)
    upside = target / price - 1.0 if target is not None and target > 0 and price else None

    subs = [
        _sub(
            "per_pct_own",
            per_pct,
            _pct_to_score(per_pct),
            f"PER own-history pct {_fmt_pct(per_pct or 0.0, signed=False)}"
            if en
            else f"PER 과거 이력 백분위 {_fmt_pct(per_pct or 0.0, signed=False)}",
        ),
        _sub(
            "pbr_pct_own",
            pbr_pct,
            _pct_to_score(pbr_pct),
            f"PBR own-history pct {_fmt_pct(pbr_pct or 0.0, signed=False)}"
            if en
            else f"PBR 과거 이력 백분위 {_fmt_pct(pbr_pct or 0.0, signed=False)}",
        ),
        _sub(
            "per_rel_sector",
            rel,
            VAL_PER_REL_SECTOR(math.log(rel)) if rel else None,
            f"PER vs sector {rel or 0.0:.2f}x" if en else f"업종 PER 대비 {rel or 0.0:.2f}배",
        ),
        _sub(
            "analyst_upside",
            upside,
            VAL_ANALYST_UPSIDE(upside),
            f"target-price gap {_fmt_pct(upside or 0.0)}" if en else f"목표주가 괴리 {_fmt_pct(upside or 0.0)}",
        ),
    ]
    return subs, notes, {"per": per, "pbr": pbr}


def volatility_subscores(md: MarketData, tech: Technicals, *, lang: object = None) -> list[SubScore]:
    """Absolute 60d vol, 60d vol percentile vs own trailing 3y, |beta - 0.8|, 1y max drawdown."""
    en = is_en(lang)
    rv = _finite(tech.realized_vol_60d)
    vol_pct = _vol_pct_3y(md)
    b = _finite(tech.beta_1y)
    mdd = _finite(tech.max_drawdown_1y)
    beta_score = VOL_BETA_DEV(abs(b - BETA_TARGET)) if b is not None else None
    # the |beta - 0.8| penalty also hits LOW betas; calling those "불안정" would mislead the reader
    beta_tone = (
        ("low market linkage" if en else "시장 연동 낮음")
        if b is not None and b < BETA_TARGET and beta_score is not None and beta_score <= 45.0
        else None
    )
    return [
        _sub(
            "realized_vol_60d",
            rv,
            VOL_ABS(rv),
            f"60D vol (ann.) {_fmt_pct(rv or 0.0, signed=False)}"
            if en
            else f"60일 변동성(연율) {_fmt_pct(rv or 0.0, signed=False)}",
        ),
        _sub(
            "vol_pct_3y",
            vol_pct,
            _pct_to_score(vol_pct),
            f"3Y vol percentile {_fmt_pct(vol_pct or 0.0, signed=False)}"
            if en
            else f"변동성 3년 백분위 {_fmt_pct(vol_pct or 0.0, signed=False)}",
        ),
        _sub("beta_1y", b, beta_score, f"beta {b or 0.0:.2f}" if en else f"베타 {b or 0.0:.2f}", tone=beta_tone),
        _sub(
            "max_drawdown_1y",
            mdd,
            VOL_MDD(mdd),
            f"1Y max drawdown {_fmt_pct(mdd or 0.0)}" if en else f"1년 최대낙폭 {_fmt_pct(mdd or 0.0)}",
        ),
    ]


def quality_subscores(fund: Fundamentals, *, lang: object = None) -> list[SubScore]:
    """ROE, operating margin, earnings growth YoY, revenue growth YoY (all: higher = better)."""
    en = is_en(lang)
    roe = _finite(fund.roe)
    opm = _finite(fund.operating_margin)
    eg = _finite(fund.earnings_growth_yoy)
    rg = _finite(fund.revenue_growth_yoy)
    return [
        _sub("roe", roe, QUAL_ROE(roe), f"ROE {_fmt_pct(roe or 0.0, digits=1)}"),
        _sub(
            "operating_margin",
            opm,
            QUAL_OPM(opm),
            f"operating margin {_fmt_pct(opm or 0.0, digits=1)}"
            if en
            else f"영업이익률 {_fmt_pct(opm or 0.0, digits=1)}",
        ),
        _sub(
            "earnings_growth_yoy",
            eg,
            QUAL_EARN_G(eg),
            f"earnings growth (YoY) {_fmt_pct(eg or 0.0)}" if en else f"이익 성장률(YoY) {_fmt_pct(eg or 0.0)}",
        ),
        _sub(
            "revenue_growth_yoy",
            rg,
            QUAL_REV_G(rg),
            f"revenue growth (YoY) {_fmt_pct(rg or 0.0)}" if en else f"매출 성장률(YoY) {_fmt_pct(rg or 0.0)}",
        ),
    ]


def _rsi_tone(r: float, lang: object = None) -> str:
    en = is_en(lang)
    if r > 75:
        return "overbought" if en else "과매수"
    if r > 65:
        return "overheating risk" if en else "과열 경계"
    if r >= 50:
        return "healthy" if en else "적정"
    if r >= 30:
        return _NEUTRAL_EN if en else _NEUTRAL
    return "oversold" if en else "과매도"


def trend_subscores(md: MarketData, tech: Technicals, *, lang: object = None) -> list[SubScore]:
    """Close vs MA50 / MA200, MA50 vs MA200, 20-day MA200 slope, RSI14."""
    en = is_en(lang)
    last = _finite(tech.last_close)
    ma50, ma200 = _finite(tech.ma50), _finite(tech.ma200)
    d50 = last / ma50 - 1.0 if last is not None and ma50 and ma50 > 0 else None
    d200 = last / ma200 - 1.0 if last is not None and ma200 and ma200 > 0 else None
    cross = ma50 / ma200 - 1.0 if ma50 is not None and ma200 and ma200 > 0 else None
    slope = _ma200_slope(md)
    r = _finite(tech.rsi14)
    cross_text = ""
    if cross is not None:
        if en:
            cross_text = f"MA50 {'above' if cross >= 0 else 'below'} MA200 ({_fmt_pct(cross)})"
        else:
            cross_text = f"50일선이 200일선 {'위' if cross >= 0 else '아래'}({_fmt_pct(cross)})"
    rsi_text = f"RSI {r or 0.0:.0f}"
    # tone from the number actually printed, so "RSI 65" never reads "과열 경계" (65.3 > 65)
    rsi_tone = _rsi_tone(float(f"{r:.0f}"), lang) if r is not None else None
    return [
        _sub(
            "close_vs_ma50",
            d50,
            TREND_MA50(d50),
            f"vs MA50 {_fmt_pct(d50 or 0.0)}" if en else f"50일선 대비 {_fmt_pct(d50 or 0.0)}",
        ),
        _sub(
            "close_vs_ma200",
            d200,
            TREND_MA200(d200),
            f"vs MA200 {_fmt_pct(d200 or 0.0)}" if en else f"200일선 대비 {_fmt_pct(d200 or 0.0)}",
        ),
        _sub("ma50_vs_ma200", cross, TREND_CROSS(cross), cross_text),
        _sub(
            "ma200_slope_20d",
            slope,
            TREND_SLOPE(slope),
            f"MA200 20D slope {_fmt_pct(slope or 0.0)}" if en else f"200일선 20일 기울기 {_fmt_pct(slope or 0.0)}",
        ),
        _sub("rsi14", r, TREND_RSI(r), rsi_text, tone=rsi_tone),
    ]


def _as_of(md: MarketData) -> pd.Timestamp | None:
    px = valid_prices(md.prices)
    return None if px.empty else pd.Timestamp(px.index[-1]).normalize()


def _vol_pct_3y(md: MarketData) -> float | None:
    """Shared features.vol_pct_3y at t: rank of today's 60d vol within the trailing 3y (<= t only)."""
    px = valid_prices(md.prices)
    if px.empty:
        return None
    feats = regime_features(px, md.benchmark if md.benchmark is not None else px.iloc[0:0])
    return _finite(feats["vol_pct_3y"].iloc[-1])


def _ma200_slope(md: MarketData) -> float | None:
    close = valid_prices(md.prices)["close"].astype("float64")
    if len(close) < MA_LONG + MA200_SLOPE_LAG:
        return None
    ma = close.rolling(MA_LONG, min_periods=MA_LONG).mean()
    now, before = _finite(ma.iloc[-1]), _finite(ma.iloc[-1 - MA200_SLOPE_LAG])
    return now / before - 1.0 if now is not None and before else None


# ------------------------------------------------------------------------------ public


def _assemble(
    name: str,
    subs: Sequence[SubScore],
    notes: Sequence[str] = (),
    extra_metrics: Mapping[str, float | None] | None = None,
    lang: object = None,
) -> FactorScore:
    scores = [s.score for s in subs if s.score is not None]
    score = float(np.clip(np.mean(scores), 0.0, 100.0)) if scores else None
    metrics: dict[str, float | None] = dict(extra_metrics or {})
    metrics.update({s.key: s.value for s in subs})
    return FactorScore(
        name=name,  # type: ignore[arg-type]
        label=factor_label(name, lang),
        score=score,
        metrics=metrics,
        rationale=_rationale(name, subs, notes, lang),
    )


def compute_factors(
    md: MarketData,
    tech: Technicals,
    settings: Settings,
    *,
    warnings: list[str] | None = None,
) -> list[FactorScore]:
    """All five factor scores, always in the order momentum, value, volatility, quality, trend.

    Args:
        md: market data (prices for the MA200 slope and the 3y vol percentile, fundamentals,
            valuation_history filtered to dates <= the last price date).
        tech: output of indicators.compute_technicals for the same prices.
        settings: only the report language (``settings.report.language``) is read; calibrations are the
            documented module constants.
        warnings: optional list (normally ``md.warnings``) that receives a note (report language) for every
            factor that could not be computed at all; such factors are also logged at WARNING level.
    """
    lang = lang_of(settings)
    value_subs, value_notes, value_extra = value_subscores(md, tech, lang=lang)
    factors = [
        _assemble("momentum", momentum_subscores(tech, lang=lang), lang=lang),
        _assemble("value", value_subs, value_notes, value_extra, lang=lang),
        _assemble("volatility", volatility_subscores(md, tech, lang=lang), lang=lang),
        _assemble("quality", quality_subscores(md.fundamentals, lang=lang), lang=lang),
        _assemble("trend", trend_subscores(md, tech, lang=lang), lang=lang),
    ]
    for f in factors:
        if f.score is None:
            # rationale starts with "산출 불가: ..." / "not computable: ..."
            msg = f"{f.label} factor {f.rationale}" if is_en(lang) else f"{f.label} 팩터 {f.rationale}"
            log.warning("%s %s", md.instrument.ticker, msg)
            if warnings is not None and msg not in warnings:
                warnings.append(msg)
    return factors


def factor_composite(factors: Iterable[FactorScore], weights: Mapping[str, float]) -> float | None:
    """Weighted mean of the available (non-None) factor scores, weights renormalised over them.

    Factors with a missing, non-positive or non-finite weight are ignored. None when no factor with a
    positive weight has a score.
    """
    num = den = 0.0
    for f in factors:
        score = _finite(f.score)
        w = _finite(weights.get(f.name))
        if score is None or w is None or w <= 0:
            continue
        num += w * score
        den += w
    if den <= 0:
        return None
    return float(np.clip(num / den, 0.0, 100.0))


def factor_pillar(composite: float | None) -> float:
    """(composite - 50) / 50 clipped to [-1, 1]; 0.0 when the composite is None/NaN."""
    c = _finite(composite)
    if c is None:
        return 0.0
    return float(np.clip((c - 50.0) / 50.0, -1.0, 1.0))
