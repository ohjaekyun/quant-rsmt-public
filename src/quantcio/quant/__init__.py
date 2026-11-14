"""Quant layer wiring: indicators -> factors -> RSMT-Q served distribution -> QuantSignal.

Public API (docs/ARCHITECTURE.md, "quant/__init__.py")
------------------------------------------------------
- ``build_quant_signal(md, settings, market_model=None, served=None) -> QuantSignal``
- ``build_quant(...) -> QuantResult`` - the same, plus the served ``ReturnDistribution`` object, which the
  execution plan needs (its barrier must come from the SAME path matrix as the signal; it is never rebuilt).
- ``distribution_ml_pillar(fc) -> float`` - the ML pillar computed from the RSMT-Q distribution.
- ``sizing_vol_forecast(md, settings) -> Optional[float]`` - the GARCH(1,1)-t horizon vol (EWMA fallback) that
  sizes the position (``QuantSignal.vol_forecast_annual``; computed by the ensemble module until 2026-10-02).

The served mean is RSMT-Q's own (100%), or the selected stock-selection model's excess mean when the pipeline
passes it as an ``ExcessOverride`` (``DistributionForecast.blend`` records which; the report prints it). The
per-ticker HGB+Ridge ensemble that used to take 20% of the mean (when it passed its own skill gate) was removed
on 2026-10-02: it passed that gate for 0/15 names in every market.

Nothing degrades silently: a failed or degraded RSMT-Q prediction and the EWMA volatility fallback add a Korean line
to ``md.warnings`` (and the log); without a distribution the ML pillar is 0 and fusion drops it as unavailable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from quantcio.config import Settings
from quantcio.i18n import tr
from quantcio.models import (
    DistributionForecast,
    ExcessForecast,
    FactorModelScore,
    MarketData,
    QuantSignal,
)
from quantcio.quant.factors import compute_factors, factor_composite, factor_pillar
from quantcio.quant.indicators import compute_technicals

if TYPE_CHECKING:  # heavy modules are imported lazily at call time
    from quantcio.quant.distribution import ReturnDistribution
    from quantcio.quant.rsmt import MarketModel

log = logging.getLogger(__name__)

__all__ = [
    "QuantResult",
    "build_quant",
    "build_quant_signal",
    "distribution_ml_pillar",
    "sizing_vol_forecast",
]

SKILL_MULTIPLIER = {True: 1.0, False: 0.35}
DEGRADED_MULTIPLIER = 0.5

Served = tuple["ReturnDistribution", DistributionForecast]


@dataclass
class QuantResult:
    signal: QuantSignal
    dist: ReturnDistribution | None  # the served path matrix (None without an RSMT-Q prediction)


def _warn(md: MarketData, msg: str) -> None:
    log.warning("%s: %s", md.instrument.ticker, msg)
    if msg not in md.warnings:
        md.warnings.append(msg)


def _short(exc: BaseException, n: int = 200) -> str:
    text = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
    return " ".join(str(text).split())[:n]


# ------------------------------------------------------------------------------ pillar


def distribution_ml_pillar(fc: DistributionForecast) -> float:
    """ML pillar from the RSMT-Q distribution (ARCHITECTURE.md, "quant/__init__.py").

    ``z = mu / sigma``, ``raw = 0.5*tanh(2z) + 0.5*(2*prob_up - 1)``, times 1.0 if ``direction_skill_ok`` else
    0.35, times a further 0.5 if ``degraded``; clipped to [-1, 1]. The multiplier reads the DIRECTION verdict (the
    served mean's source: the stock-selection model's served skill, or RSMT-Q's AUC gate), never ``skill_ok``, which
    with rsmt.skill_metric = "distribution" certifies the risk distribution (CRPS / coverage), not a direction.

    Raises:
        ValueError: non-finite mu, non-positive sigma or prob_up outside [0, 1] (contract violation; a NaN
            would otherwise come out as a maximal pillar).
    """
    mu, sigma, p = float(fc.mu), float(fc.sigma), float(fc.prob_up)
    if not (math.isfinite(mu) and math.isfinite(sigma) and sigma > 0 and math.isfinite(p) and 0.0 <= p <= 1.0):
        raise ValueError(f"DistributionForecast needs finite mu, sigma > 0, prob_up in [0, 1] (got {mu}, {sigma}, {p})")
    raw = 0.5 * math.tanh(2.0 * mu / sigma) + 0.5 * (2.0 * p - 1.0)
    mult = SKILL_MULTIPLIER[bool(fc.direction_skill_ok)] * (DEGRADED_MULTIPLIER if fc.degraded else 1.0)
    return float(max(-1.0, min(1.0, raw * mult)))


# ------------------------------------------------------------------------------ sizing vol


def sizing_vol_forecast(md: MarketData, settings: Settings) -> float | None:
    """GARCH(1,1)-t horizon-average annualised vol (EWMA fallback) of the close, for position sizing.

    The EWMA fallback is stated in ``md.warnings``; ``None`` (with a warning) when neither estimate exists - sizing
    then uses the realised 60-day vol (fusion.risk).
    """
    from quantcio.quant.volatility import garch_vol_forecast_ex

    try:
        vf = garch_vol_forecast_ex(md.prices["close"].astype("float64"), int(settings.horizon_days))
    except Exception as exc:  # noqa: BLE001 - VolatilityError or bad input: sizing falls back, never silently
        _warn(
            md,
            tr(
                settings,
                f"변동성 예측 불가({type(exc).__name__}): {_short(exc)} — 비중 산정에 실현변동성 사용",
                f"Volatility forecast unavailable ({type(exc).__name__}): {_short(exc)} - sizing uses realised vol",
            ),
        )
        return None
    if vf.is_fallback:
        _warn(md, vf.note())
    return float(vf.annual_vol)


# ------------------------------------------------------------------------------ RSMT-Q


def _served_distribution(
    md: MarketData,
    market_model: MarketModel | None,
    served: Served | BaseException | None,
    settings: Settings | None = None,
) -> tuple[ReturnDistribution | None, DistributionForecast | None]:
    ticker = md.instrument.ticker
    if served is None and market_model is not None:
        try:
            served = market_model.predict(ticker)
        except Exception as exc:  # noqa: BLE001 - reported below, never silent
            served = exc
    if served is None:
        return None, None
    if isinstance(served, BaseException):
        if not isinstance(served, KeyError | ValueError):
            log.error("%s: RSMT-Q prediction failed", ticker, exc_info=(type(served), served, served.__traceback__))
        _warn(
            md,
            tr(
                settings,
                f"RSMT-Q 예측 실패({type(served).__name__}): {_short(served)} — ML 필러 제외",
                f"RSMT-Q forecast failed ({type(served).__name__}): {_short(served)} - ML pillar dropped",
            ),
        )
        return None, None
    dist, fc = served
    if not isinstance(fc, DistributionForecast):
        raise TypeError(f"served must be (ReturnDistribution, DistributionForecast), got {type(fc).__name__}")
    return dist, fc


# ------------------------------------------------------------------------------ public


def build_quant(
    md: MarketData,
    settings: Settings,
    market_model: MarketModel | None = None,
    served: Served | BaseException | None = None,
    *,
    excess_model: ExcessForecast | None = None,
    factor_model: FactorModelScore | None = None,
) -> QuantResult:
    """Indicators -> factors -> sizing vol -> RSMT-Q distribution -> QuantSignal (+ the served path matrix).

    Args:
        md: market data; every degraded path appends a Korean line to ``md.warnings``.
        settings: effective settings.
        market_model: fitted RSMT-Q model of the ticker's market; used only when ``served`` is None.
        served: the (ReturnDistribution, DistributionForecast) that ``MarketModel.predict_many`` returned
            for this ticker (preferred: the batch health checks are then final), or the exception it
            returned in its place.
        excess_model: the selected stock-selection model's forecast (``quantcio.modeling.serve``); its excess
            mean is already inside ``served`` when the pipeline passed it to ``predict_many`` as an override.
        factor_model: the PIT factor-model score (``backtest.score_tickers``), read by the fusion factor pillar.

    Raises:
        ValueError: the price history is too short for the technical snapshot.
    """
    tech = compute_technicals(md.prices, md.benchmark, settings.risk.atr_period, warnings=md.warnings)
    factors = compute_factors(md, tech, settings, warnings=md.warnings)
    composite = factor_composite(factors, settings.factors.weights)
    vol = sizing_vol_forecast(md, settings)

    dist, fc = _served_distribution(md, market_model, served, settings)
    pillar: float | None = None
    if fc is not None:
        try:
            pillar = distribution_ml_pillar(fc)
        except ValueError as exc:
            _warn(
                md,
                tr(
                    settings,
                    f"RSMT-Q 분포 값 이상 — 분포 미사용, ML 필러 제외: {_short(exc)}",
                    f"RSMT-Q distribution invalid - not used, ML pillar dropped: {_short(exc)}",
                ),
            )
            dist, fc = None, None
    if fc is not None and fc.degraded:
        reasons = "; ".join(fc.degraded_reasons) or tr(settings, "사유 미기재", "reason not given")
        _warn(
            md,
            tr(
                settings,
                f"RSMT-Q 저하 모드: {reasons} — ML 필러 ×{DEGRADED_MULTIPLIER:g}, 신뢰도 감점",
                f"RSMT-Q degraded: {reasons} - ML pillar x{DEGRADED_MULTIPLIER:g}, confidence penalty",
            ),
        )

    signal = QuantSignal(
        technicals=tech,
        factors=factors,
        factor_composite=composite,
        factor_pillar=factor_pillar(composite),
        distribution=fc,
        ml_pillar=0.0 if pillar is None else pillar,
        vol_forecast_annual=vol,
        factor_model=factor_model,
        excess_model=excess_model,
    )
    return QuantResult(signal=signal, dist=dist)


def build_quant_signal(
    md: MarketData,
    settings: Settings,
    market_model: MarketModel | None = None,
    served: Served | BaseException | None = None,
    *,
    excess_model: ExcessForecast | None = None,
    factor_model: FactorModelScore | None = None,
) -> QuantSignal:
    """``build_quant(...).signal`` (see ``build_quant``; use it when the path matrix is needed as well)."""
    return build_quant(md, settings, market_model, served, excess_model=excess_model, factor_model=factor_model).signal
