"""Tests for quant.volatility: GARCH(1,1)-t horizon forecast and its observable EWMA fallback (offline)."""

from __future__ import annotations

import logging
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from conftest import make_prices

from quantcio.quant import volatility as vol_mod
from quantcio.quant.volatility import (
    MAX_OBS,
    MIN_GARCH_OBS,
    VolatilityError,
    VolForecast,
    ewma_vol_forecast,
    garch_vol_forecast,
    garch_vol_forecast_ex,
)

LOGGER = "quantcio.quant.volatility"


def _garch_close(
    n: int = 2000,
    annual_vol: float = 0.25,
    alpha: float = 0.08,
    beta: float = 0.90,
    nu: float = 8.0,
    seed: int = 3,
) -> pd.Series:
    """Close series whose percent log returns follow a GARCH(1,1) with Student-t innovations."""
    rng = np.random.default_rng(seed)
    uncond_var = (annual_vol / math.sqrt(252) * 100.0) ** 2  # daily, in %^2
    omega = uncond_var * (1.0 - alpha - beta)
    z = rng.standard_t(nu, n) * math.sqrt((nu - 2.0) / nu)
    r = np.empty(n)
    var = uncond_var
    for t in range(n):
        r[t] = math.sqrt(var) * z[t]
        var = omega + alpha * r[t] ** 2 + beta * var
    idx = pd.bdate_range(end="2026-09-23", periods=n)
    return pd.Series(100.0 * np.exp(np.cumsum(r / 100.0)), index=idx, name="close")


# ----------------------------------------------------------------------------- happy path


def test_garch_path_on_garch_data():
    close = _garch_close()
    vf = garch_vol_forecast_ex(close, 20)
    assert isinstance(vf, VolForecast)
    assert vf.method == "garch" and not vf.is_fallback and vf.fallback_reason is None
    assert 0.12 < vf.annual_vol < 0.50
    assert vf.n_obs == MAX_OBS
    assert {"omega", "alpha[1]", "beta[1]", "nu"} <= set(vf.params)
    assert "GARCH" in vf.note()
    assert garch_vol_forecast(close, 20) == pytest.approx(vf.annual_vol)


def test_horizon_average_moves_toward_unconditional():
    """A high current variance decays toward the long-run level, so the 60d average is below the 1d value."""
    close = _garch_close(seed=11)
    # append a violent week to push the conditional variance far above its unconditional level
    shock = close.iloc[-1] * np.exp(np.cumsum([0.08, -0.09, 0.07, -0.08, 0.09]))
    idx = pd.bdate_range(start=close.index[-1] + pd.Timedelta(1, "D"), periods=5)
    close2 = pd.concat([close, pd.Series(shock, index=idx)])
    v1 = garch_vol_forecast_ex(close2, 1)
    v60 = garch_vol_forecast_ex(close2, 60)
    assert v1.method == v60.method == "garch"
    assert v1.params["alpha[1]"] + v1.params["beta[1]"] < 1.0
    assert v60.annual_vol < v1.annual_vol


def test_uses_last_1500_returns_in_percent(monkeypatch):
    seen = {}

    def spy(r_pct, horizon_days):
        seen["n"], seen["std"], seen["h"] = len(r_pct), float(r_pct.std()), horizon_days
        return 0.3, {}

    monkeypatch.setattr(vol_mod, "_garch_fit_forecast", spy)
    close = make_prices(n=2200, vol=0.018, seed=5)["close"]
    vf = garch_vol_forecast_ex(close, 20)
    assert vf.method == "garch" and vf.annual_vol == 0.3
    assert seen["n"] == MAX_OBS
    assert seen["std"] == pytest.approx(1.8, rel=0.1)  # 100 x daily log returns
    assert seen["h"] == 20


def test_ewma_matches_riskmetrics_recursion():
    close = make_prices(n=400, vol=0.02, seed=9)["close"]
    r = np.log(close).diff().dropna().to_numpy()
    var = r[0] ** 2
    for x in r[1:]:
        var = 0.94 * var + 0.06 * x**2
    assert ewma_vol_forecast(close) == pytest.approx(math.sqrt(var * 252))


# ----------------------------------------------------------------------------- fallback is observable


def _assert_fallback(vf: VolForecast, close: pd.Series, reason_part: str) -> None:
    assert vf.method == "ewma" and vf.is_fallback
    assert vf.fallback_reason and reason_part in vf.fallback_reason
    assert vf.annual_vol == pytest.approx(ewma_vol_forecast(close))
    assert "EWMA" in vf.note() and reason_part in vf.note()


def test_fallback_short_history(caplog):
    close = make_prices(n=MIN_GARCH_OBS - 50, seed=1)["close"]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        vf = garch_vol_forecast_ex(close, 20)
    _assert_fallback(vf, close, "표본 부족")
    assert any("EWMA" in rec.getMessage() for rec in caplog.records if rec.name == LOGGER)


def test_fallback_when_fit_raises(monkeypatch, caplog):
    def boom(r_pct, horizon_days):
        raise np.linalg.LinAlgError("singular matrix")

    monkeypatch.setattr(vol_mod, "_garch_fit_forecast", boom)
    close = make_prices(n=800, seed=2)["close"]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        vf = garch_vol_forecast_ex(close, 20)
        value = garch_vol_forecast(close, 20)  # float API: same value, fallback logged
    _assert_fallback(vf, close, "LinAlgError")
    assert value == pytest.approx(vf.annual_vol)
    assert sum("unusable" in rec.getMessage() for rec in caplog.records if rec.name == LOGGER) == 2


@pytest.mark.parametrize(
    "bad_value, reason_part",
    [(float("nan"), "비유한"), (float("inf"), "비유한"), (5.0, "범위"), (0.01, "범위")],
)
def test_fallback_on_non_finite_or_absurd_value(monkeypatch, bad_value, reason_part):
    monkeypatch.setattr(vol_mod, "_garch_fit_forecast", lambda r, h: (bad_value, {"omega": 1.0}))
    close = make_prices(n=800, seed=4)["close"]
    vf = garch_vol_forecast_ex(close, 20)
    _assert_fallback(vf, close, reason_part)
    assert vf.params == {"omega": 1.0}  # the rejected fit's parameters stay inspectable


def test_fallback_on_non_convergence(monkeypatch):
    """Exercise the real convergence check with a fake arch result (convergence_flag != 0)."""
    import arch

    class _FakeModel:
        def fit(self, **kwargs):
            return SimpleNamespace(
                convergence_flag=4,
                params=pd.Series({"omega": 0.1, "alpha[1]": 0.1, "beta[1]": 0.8, "nu": 5.0}),
                forecast=lambda **kw: pytest.fail("forecast must not be used after non-convergence"),
            )

    monkeypatch.setattr(arch, "arch_model", lambda *a, **k: _FakeModel())
    close = make_prices(n=800, seed=6)["close"]
    vf = garch_vol_forecast_ex(close, 20)
    _assert_fallback(vf, close, "미수렴")


def _blanket_ignores() -> int:
    """Number of process-global 'ignore every warning' filters (what simplefilter('ignore') installs)."""
    return sum(1 for f in warnings.filters if f[0] == "ignore" and f[1] is None and f[2] is Warning and f[3] is None)


def test_garch_fit_does_not_mute_warnings_globally(monkeypatch):
    """Regression: the fit ran under catch_warnings()+simplefilter('ignore'). The filter list is process-global,
    so while a fit ran every other thread's warnings were muted (deterministic check below)."""
    import arch

    seen: dict[str, int] = {}

    class _FakeModel:
        def fit(self, **kwargs):
            seen["during_fit"] = _blanket_ignores()
            return SimpleNamespace(convergence_flag=4, params=pd.Series({"omega": 0.1}))

    monkeypatch.setattr(arch, "arch_model", lambda *a, **k: _FakeModel())
    before = _blanket_ignores()
    garch_vol_forecast_ex(make_prices(n=800, seed=6)["close"], 20)
    assert seen["during_fit"] == before


def test_concurrent_fits_do_not_leak_warning_filters():
    """Regression: concurrent fits (tickers run in a thread pool) interleaved catch_warnings save/restore and
    left a permanent blanket 'ignore' filter, silencing every later warning in the process."""
    closes = [make_prices(n=1600, seed=s)["close"] for s in range(8)]
    before = _blanket_ignores()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda c: garch_vol_forecast_ex(c, 20), closes * 3))
    assert all(r.method == "garch" for r in results)
    assert _blanket_ignores() == before


# the flat series is degenerate on purpose: arch divides by its zero variance (RuntimeWarnings inside arch) before our
# code turns the result into an explicit VolatilityError - which is what this test checks. Scoped to this test only
@pytest.mark.filterwarnings("ignore::RuntimeWarning:arch\\..*")
def test_no_silent_third_path():
    flat = pd.Series(100.0, index=pd.bdate_range(end="2026-09-23", periods=400))
    with pytest.raises(VolatilityError):
        garch_vol_forecast_ex(flat, 20)
    with pytest.raises(VolatilityError):
        garch_vol_forecast(make_prices(n=10)["close"], 20)


def test_invalid_horizon():
    with pytest.raises(ValueError):
        garch_vol_forecast_ex(make_prices(n=400)["close"], 0)


def test_ignores_nan_and_non_positive_prices():
    close = make_prices(n=900, seed=8)["close"].copy()
    close.iloc[[100, 200]] = np.nan
    close.iloc[300] = 0.0
    vf = garch_vol_forecast_ex(close, 20)
    assert math.isfinite(vf.annual_vol) and vf.annual_vol > 0
