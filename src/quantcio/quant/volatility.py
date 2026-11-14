"""Horizon volatility forecast: GARCH(1,1) with Student-t errors, EWMA(0.94) fallback.

Public API
----------
- ``garch_vol_forecast(close, horizon_days) -> float``
    The spec signature. Returns the annualised volatility forecast (decimal, e.g. 0.28 == 28%).
    When the GARCH path is not usable it returns the EWMA fallback AND logs a WARNING on this
    module's logger. Callers that must surface the fallback (the ML forecast notes, the report)
    should use ``garch_vol_forecast_ex`` instead.
- ``garch_vol_forecast_ex(close, horizon_days) -> VolForecast``
    Same computation, but returns the value together with the method actually used
    (``"garch"`` or ``"ewma"``), the fallback reason, and the fitted GARCH parameters, so the
    degraded path is never silent (lesson from the RSMT deployment incident).
- ``ewma_vol_forecast(close, lam=0.94) -> float``
    The RiskMetrics fallback on its own (the EWMA variance forecast is flat over the horizon).

Algorithm (ARCHITECTURE.md, quant/volatility.py)
------------------------------------------------
1. r = 100 * daily log returns of ``close``; keep the last <= 1500 observations.
2. Fit GARCH(1,1), constant mean, Student-t innovations (``arch``), no rescaling.
3. Take the mean of the h analytic variance forecasts (daily, in %^2), annualise:
   ``sqrt(mean_var * 252) / 100``.
4. Fall back to EWMA(lambda = 0.94) when: too few observations (< ``MIN_GARCH_OBS``), the
   optimiser raises or does not converge, or the result is non-finite / outside
   ``VOL_BOUNDS`` = [0.03, 2.5].
5. If even the EWMA estimate is unusable (fewer than ``MIN_EWMA_OBS`` returns, zero or
   non-finite variance) ``VolatilityError`` is raised - there is no silent third fallback.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from quantcio.quant.features import EWMA_LAMBDA, TRADING_DAYS

log = logging.getLogger(__name__)

VolMethod = Literal["garch", "ewma"]

MAX_OBS = 1500  # GARCH is fitted on the last <= 1500 daily returns
MIN_GARCH_OBS = 250  # below ~1y of returns the (omega, alpha, beta, nu) fit is unreliable
MIN_EWMA_OBS = 20  # minimum returns for the EWMA fallback to be meaningful
VOL_BOUNDS = (0.03, 2.5)  # annualised; anything outside is treated as a failed fit
_SCALE = 100.0  # returns are fitted in percent (numerically friendlier for arch)


class VolatilityError(ValueError):
    """Raised when no volatility estimate (neither GARCH nor EWMA) can be produced."""


@dataclass(frozen=True)
class VolForecast:
    """Annualised volatility forecast plus provenance.

    Attributes:
        annual_vol: annualised volatility of daily log returns (decimal), averaged over the horizon.
        method: ``"garch"`` when the GARCH(1,1)-t forecast was used, ``"ewma"`` for the fallback.
        horizon_days: forecast horizon in trading days.
        n_obs: number of daily returns used by the method that produced ``annual_vol``.
        fallback_reason: Korean reason why GARCH was not used (None when method == "garch").
        params: fitted GARCH parameters (mu, omega, alpha[1], beta[1], nu) when available.
    """

    annual_vol: float
    method: VolMethod
    horizon_days: int
    n_obs: int
    fallback_reason: str | None = None
    params: dict[str, float] = field(default_factory=dict)

    @property
    def is_fallback(self) -> bool:
        return self.method != "garch"

    def note(self) -> str:
        """One-line Korean description for report notes."""
        if self.method == "garch":
            persistence = self.params.get("alpha[1]", float("nan")) + self.params.get("beta[1]", float("nan"))
            extra = f", 지속성 {persistence:.3f}" if math.isfinite(persistence) else ""
            return (
                f"변동성: GARCH(1,1)-t {self.horizon_days}일 평균 예측 연 {self.annual_vol:.1%} (n={self.n_obs}{extra})"
            )
        return (
            f"변동성: EWMA(λ={EWMA_LAMBDA}) 대체 사용 연 {self.annual_vol:.1%} "
            f"— GARCH 미사용 사유: {self.fallback_reason}"
        )


class _GarchUnusableError(Exception):
    """Internal: GARCH path failed; message is the (Korean) fallback reason."""


def _clean_log_returns(close: pd.Series) -> pd.Series:
    s = pd.to_numeric(pd.Series(close), errors="coerce").astype("float64")
    s = s[np.isfinite(s) & (s > 0)]
    r = np.log(s).diff()
    return r[np.isfinite(r)]


def ewma_vol_forecast(close: pd.Series, lam: float = EWMA_LAMBDA) -> float:
    """RiskMetrics EWMA annualised vol from daily log returns (flat forecast over any horizon).

    Mirrors ``features.ewma_vol`` (squared returns, adjust=False, seeded by the first return) but only
    requires ``MIN_EWMA_OBS`` returns. Raises ``VolatilityError`` when it cannot be estimated.
    """
    r = _clean_log_returns(close)
    if len(r) < MIN_EWMA_OBS:
        raise VolatilityError(f"EWMA 변동성 추정 불가: 수익률 {len(r)}개 (최소 {MIN_EWMA_OBS}개 필요)")
    var = float((r**2).ewm(alpha=1.0 - lam, adjust=False).mean().iloc[-1])
    vol = math.sqrt(var * TRADING_DAYS) if var >= 0 else float("nan")
    if not math.isfinite(vol) or vol <= 0:
        raise VolatilityError(f"EWMA 변동성 추정 불가: 비정상 값 {vol!r} (가격 변동 없음?)")
    return vol


def _garch_fit_forecast(r_pct: pd.Series, horizon_days: int) -> tuple[float, dict[str, float]]:
    """Fit GARCH(1,1)-t on percent returns; return (annualised vol over horizon, params).

    Raises ``_GarchUnusableError`` on non-convergence; any optimiser exception propagates to the caller,
    which converts it into a fallback reason.
    """
    from arch import arch_model  # local import: arch is heavy and only needed here

    model = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    # No warnings.catch_warnings()/simplefilter("ignore") here: the filter list is process-global and
    # catch_warnings is not thread-safe, so concurrent fits (the pipeline runs tickers in a thread pool)
    # leaked a blanket "ignore" filter that silenced every later warning in the process.
    # show_warning=False already mutes arch's ConvergenceWarning, and convergence is checked explicitly below.
    res = model.fit(disp="off", show_warning=False)
    if int(res.convergence_flag) != 0:
        raise _GarchUnusableError(f"최적화 미수렴 (flag={int(res.convergence_flag)})")
    fc = res.forecast(horizon=horizon_days, reindex=False)
    daily_var_pct2 = np.asarray(fc.variance.to_numpy()[-1], dtype="float64")
    mean_var = float(np.mean(daily_var_pct2))
    vol = math.sqrt(mean_var * TRADING_DAYS) / _SCALE if mean_var >= 0 else float("nan")
    params = {str(k): float(v) for k, v in res.params.items()}
    return vol, params


def garch_vol_forecast_ex(close: pd.Series, horizon_days: int) -> VolForecast:
    """GARCH(1,1)-t horizon-average vol forecast with an observable EWMA fallback.

    Args:
        close: daily close prices (ascending index). Non-positive / NaN values are ignored.
        horizon_days: forecast horizon in trading days (>= 1).

    Returns:
        ``VolForecast``; check ``.method`` / ``.is_fallback`` / ``.fallback_reason``.

    Raises:
        ValueError: horizon_days < 1.
        VolatilityError: neither GARCH nor EWMA can produce a finite, positive estimate.
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    r = _clean_log_returns(close).iloc[-MAX_OBS:]
    n = len(r)
    params: dict[str, float] = {}
    try:
        if n < MIN_GARCH_OBS:
            raise _GarchUnusableError(f"표본 부족 (수익률 {n}개 < {MIN_GARCH_OBS}개)")
        vol, params = _garch_fit_forecast(r * _SCALE, horizon_days)
        lo, hi = VOL_BOUNDS
        if not math.isfinite(vol):
            raise _GarchUnusableError(f"비유한 예측값 ({vol!r})")
        if not lo <= vol <= hi:
            raise _GarchUnusableError(f"비정상 예측값 연 {vol:.1%} (허용 범위 {lo:.0%}~{hi:.0%} 밖)")
        return VolForecast(
            annual_vol=vol,
            method="garch",
            horizon_days=horizon_days,
            n_obs=n,
            params=params,
        )
    except _GarchUnusableError as exc:
        reason = str(exc)
    except Exception as exc:  # noqa: BLE001 - any optimiser failure becomes an explicit fallback reason
        reason = f"적합 실패 ({type(exc).__name__}: {str(exc)[:120]})"

    ewma = ewma_vol_forecast(close)  # raises VolatilityError if unusable - never a silent third path
    log.warning(
        "GARCH vol forecast unusable (%s); using EWMA(%.2f) = %.4f",
        reason,
        EWMA_LAMBDA,
        ewma,
    )
    return VolForecast(
        annual_vol=ewma,
        method="ewma",
        horizon_days=horizon_days,
        n_obs=len(_clean_log_returns(close)),
        fallback_reason=reason,
        params=params,
    )


def garch_vol_forecast(close: pd.Series, horizon_days: int) -> float:
    """Annualised horizon-average vol (GARCH(1,1)-t, EWMA fallback). See ``garch_vol_forecast_ex``.

    The fallback is logged at WARNING level; use ``garch_vol_forecast_ex`` to get the method in-band.
    """
    return garch_vol_forecast_ex(close, horizon_days).annual_vol
