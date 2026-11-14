"""Tests for quant/indicators.py (synthetic, offline)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from conftest import make_prices

from quantcio.models import PRICE_COLUMNS, Technicals
from quantcio.quant.features import wilder_rsi
from quantcio.quant.indicators import (
    BETA_MIN_OBS,
    atr,
    beta,
    compute_technicals,
    max_drawdown,
    realized_vol,
    rsi,
    sanitize_high_low,
    true_range,
    wilder,
)


def _frame(rows: list[tuple[float, float, float]], start: str = "2026-01-05") -> pd.DataFrame:
    """(high, low, close) rows -> PRICE_COLUMNS frame on business days (open = close, volume = 1)."""
    idx = pd.bdate_range(start=start, periods=len(rows))
    hi, lo, c = (np.array(col, dtype="float64") for col in zip(*rows))
    return pd.DataFrame({"open": c, "high": hi, "low": lo, "close": c, "volume": np.ones(len(rows))}, index=idx)


# Hand-worked example (period 3):
#   TR1 = max(1.0, |10.5-9.5|, |9.5-9.5|) = 1.0
#   TR2 = max(1.1, |11.5-10.2|, |10.4-10.2|) = 1.3
#   TR3 = max(1.2, |11.2-11.0|, |10.0-11.0|) = 1.2
#   TR4 = max(1.8, |10.8-10.3|, |9.0-10.3|) = 1.8
#   TR5 = max(1.0, |12.0-9.4|, |11.0-9.4|) = 2.6      (gap up)
#   ATR3 = (1.0+1.3+1.2)/3 = 7/6 ; ATR4 = (2*7/6 + 1.8)/3 = 62/45 ; ATR5 = (2*62/45 + 2.6)/3 = 241/135
TINY = [
    (10.0, 9.0, 9.5),
    (10.5, 9.5, 10.2),
    (11.5, 10.4, 11.0),
    (11.2, 10.0, 10.3),
    (10.8, 9.0, 9.4),
    (12.0, 11.0, 11.8),
]


def test_true_range_hand_computed():
    tr = true_range(_frame(TINY))
    assert math.isnan(tr.iloc[0])
    np.testing.assert_allclose(tr.iloc[1:].to_numpy(), [1.0, 1.3, 1.2, 1.8, 2.6], rtol=0, atol=1e-12)


def test_wilder_atr_hand_computed():
    a = atr(_frame(TINY), period=3)
    assert a.iloc[:3].isna().all()
    np.testing.assert_allclose(a.iloc[3:].to_numpy(), [7 / 6, 62 / 45, 241 / 135], rtol=0, atol=1e-12)


def test_wilder_seed_recursion_and_gap_carry():
    x = pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0, np.nan, 5.0])
    w = wilder(x, 3)
    # seed = mean(1,2,3) = 2 ; y4 = 2 + (4-2)/3 = 8/3 ; NaN input carries 8/3 ; y6 = 8/3 + (5-8/3)/3 = 31/9
    expected = [np.nan, np.nan, np.nan, 2.0, 8 / 3, 8 / 3, 31 / 9]
    np.testing.assert_allclose(w.to_numpy(), expected, rtol=0, atol=1e-12, equal_nan=True)
    with pytest.raises(ValueError):
        wilder(x, 0)


def test_rsi_hand_computed():
    # n=2, alpha=1/2, seeded with the first change (features.wilder_rsi convention):
    #   gains [1, 0, 1, 0, 1] -> avg 1, .5, .75, .375, .6875 ; losses [0, .5, 0, .5, 0] -> 0, .25, .125, .3125, .15625
    #   RSI = 100 - 100/(1+RS): RS 2, 6, 1.2, 4.4 -> 200/3, 600/7, 600/11, 2200/27
    close = pd.Series([10.0, 11.0, 10.5, 11.5, 11.0, 12.0])
    r = rsi(close, 2)
    np.testing.assert_allclose(
        r.to_numpy(), [np.nan, np.nan, 200 / 3, 600 / 7, 600 / 11, 2200 / 27], rtol=0, atol=1e-10, equal_nan=True
    )
    pd.testing.assert_series_equal(rsi(close, 2), wilder_rsi(close, 2))  # one shared definition
    assert rsi(pd.Series(np.arange(1.0, 40.0)), 14).iloc[-1] == 100.0  # only gains


def test_realized_vol_matches_numpy():
    px = make_prices(n=300, seed=3)
    lr = np.diff(np.log(px["close"].to_numpy()))
    expected = np.std(lr[-60:], ddof=1) * math.sqrt(252)
    rv = realized_vol(px["close"], 60)
    assert rv.iloc[-1] == pytest.approx(expected, rel=1e-12)
    assert rv.iloc[:60].isna().all() and not math.isnan(rv.iloc[60])


def _beta_pair(n: int = 400, k: float = 1.5, seed: int = 1) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-09-23", periods=n)
    b = rng.normal(0.0, 0.01, n)
    bench = pd.Series(3000.0 * np.exp(np.cumsum(b)), index=idx)
    stock = pd.Series(100.0 * np.exp(np.cumsum(k * b)), index=idx)
    return stock, bench


def test_beta_exact_on_linear_log_returns():
    stock, bench = _beta_pair(k=1.5)
    assert beta(stock, bench) == pytest.approx(1.5, abs=1e-9)


def test_beta_uses_common_dates_not_fake_zero_returns():
    stock, bench = _beta_pair(k=1.5, seed=2)
    holes = bench.index[-200::7]  # benchmark missing ~29 recent sessions
    # inner join: multi-day returns on both legs keep the exact 1.5 relation
    assert beta(stock, bench.drop(holes)) == pytest.approx(1.5, abs=1e-9)


def test_beta_none_cases():
    stock, bench = _beta_pair()
    assert beta(stock, None) is None
    assert beta(stock, bench.iloc[0:0]) is None
    assert beta(stock, bench.iloc[-(BETA_MIN_OBS - 10) :]) is None  # too little overlap
    flat = pd.Series(3000.0, index=bench.index)
    assert beta(stock, flat) is None  # zero benchmark variance


def test_max_drawdown_hand_computed():
    close = pd.Series([100.0, 120.0, 90.0, 110.0, 60.0, 80.0])
    assert max_drawdown(close, None) == pytest.approx(-0.5)  # 60 / 120 - 1
    assert max_drawdown(close, 2) == pytest.approx(60.0 / 110.0 - 1)  # last 3 closes: 110, 60, 80
    assert max_drawdown(pd.Series(np.arange(1.0, 50.0))) == 0.0
    assert max_drawdown(pd.Series([], dtype=float)) == 0.0


def test_compute_technicals_definitions(md_kr):
    px, bench = md_kr.prices, md_kr.benchmark
    close, high, low = px["close"], px["high"], px["low"]
    t = compute_technicals(px, bench, 14)
    assert isinstance(t, Technicals)
    assert t.last_close == close.iloc[-1]
    assert t.atr == pytest.approx(atr(px, 14).iloc[-1]) and t.atr_pct == pytest.approx(t.atr / t.last_close)
    assert t.rsi14 == pytest.approx(wilder_rsi(close, 14).iloc[-1])
    assert t.ma50 == pytest.approx(close.iloc[-50:].mean())
    assert t.ma200 == pytest.approx(close.iloc[-200:].mean())
    assert t.high_52w == pytest.approx(high.iloc[-252:].max())
    assert t.low_52w == pytest.approx(low.iloc[-252:].min())
    assert t.ret_1m == pytest.approx(close.iloc[-1] / close.iloc[-22] - 1)
    assert t.ret_3m == pytest.approx(close.iloc[-1] / close.iloc[-64] - 1)
    assert t.ret_12m_ex1m == pytest.approx(close.iloc[-22] / close.iloc[-253] - 1)
    bret = bench["close"].iloc[-1] / bench["close"].iloc[-64] - 1  # same calendar in the fixture
    assert t.rel_strength_3m == pytest.approx(t.ret_3m - bret)
    assert t.support == pytest.approx(low.iloc[-20:].min())
    assert t.resistance == pytest.approx(high.iloc[-60:].max())
    assert t.realized_vol_60d == pytest.approx(realized_vol(close).iloc[-1])
    assert t.beta_1y is not None and -3 < t.beta_1y < 3
    assert t.max_drawdown_1y <= 0
    assert t.high_52w >= t.last_close >= t.low_52w


def test_empty_benchmark_is_explicit():
    px = make_prices(n=400, seed=4)
    notes: list[str] = []
    t = compute_technicals(px, pd.DataFrame(columns=PRICE_COLUMNS, dtype="float64"), warnings=notes)
    assert t.beta_1y is None and t.rel_strength_3m is None
    assert any("벤치마크 데이터 없음" in n for n in notes)
    assert compute_technicals(px, None).beta_1y is None


def test_short_history_degrades_explicitly():
    px = make_prices(n=150, seed=5)
    bench = make_prices(n=150, seed=6, start_price=2500.0)
    notes: list[str] = []
    t = compute_technicals(px, bench, warnings=notes)
    assert t.ma200 is None and t.ma50 is not None
    assert t.ret_12m_ex1m is None and t.ret_3m is not None
    assert t.beta_1y is not None  # 149 common returns >= BETA_MIN_OBS
    assert t.high_52w == pytest.approx(px["high"].max())
    assert any("1년 미만" in n for n in notes)

    very_short = make_prices(n=40, seed=5)
    notes2: list[str] = []
    t2 = compute_technicals(very_short, make_prices(n=40, seed=6), warnings=notes2)
    lr = np.diff(np.log(very_short["close"].to_numpy()))
    assert t2.realized_vol_60d == pytest.approx(np.std(lr, ddof=1) * math.sqrt(252))
    assert t2.beta_1y is None and t2.ret_3m is None and t2.rel_strength_3m is None
    assert any("실현변동성" in n for n in notes2) and any("베타 미산출" in n for n in notes2)

    with pytest.raises(ValueError):
        compute_technicals(make_prices(n=15), None)


def test_bad_rows_are_repaired_and_reported():
    px = make_prices(n=300, seed=8)
    px.iloc[100, px.columns.get_loc("high")] = 0.0  # trading-halt style row
    px.iloc[100, px.columns.get_loc("low")] = 0.0
    px.iloc[150, px.columns.get_loc("high")] = np.nan
    px.iloc[200, px.columns.get_loc("close")] = np.nan  # dropped row
    high, low, n_fixed = sanitize_high_low(px.dropna(subset=["close"]))
    assert n_fixed == 2 and (high >= low).all()
    notes: list[str] = []
    t = compute_technicals(px, make_prices(n=300, seed=9), warnings=notes)
    assert math.isfinite(t.atr) and t.atr > 0
    assert any("1행 제외" in n for n in notes) and any("2행" in n for n in notes)
    # notes are deduplicated when the same list is passed twice
    compute_technicals(px, make_prices(n=300, seed=9), warnings=notes)
    assert len(notes) == len(set(notes))


def _reference_atr(high: list[float], low: list[float], close: list[float], n: int) -> list[float]:
    """Textbook Wilder ATR written independently (plain loops, no pandas): SMA seed of TR_1..TR_n."""
    tr = [math.nan] + [
        max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])) for i in range(1, len(close))
    ]
    out = [math.nan] * len(close)
    out[n] = sum(tr[1 : n + 1]) / n
    for i in range(n + 1, len(close)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


@pytest.mark.parametrize("period", [5, 14, 20])
def test_atr_matches_independent_wilder_loop(period):
    px = make_prices(n=700, seed=21)
    ref = _reference_atr(px["high"].tolist(), px["low"].tolist(), px["close"].tolist(), period)
    np.testing.assert_allclose(atr(px, period).to_numpy(), ref, rtol=1e-12, atol=1e-9, equal_nan=True)
    assert compute_technicals(px, None, period).atr == pytest.approx(ref[-1], rel=1e-12)


def test_rsi_edge_cases():
    up = pd.Series(np.arange(1.0, 41.0))
    assert rsi(up).iloc[-1] == 100.0 and rsi(up[::-1].reset_index(drop=True)).iloc[-1] == 0.0
    # regression: a close that never moved is 0/0 (undefined), not RSI 100 ("과매수")
    flat = pd.Series(np.full(40, 100.0))
    assert rsi(flat).isna().all()
    # once it moves the value is defined again and identical to the shared definition
    moved = pd.Series(np.r_[np.full(30, 100.0), 101.0, 100.5])
    assert rsi(moved).iloc[-1] == pytest.approx(wilder_rsi(moved).iloc[-1]) and 0 < rsi(moved).iloc[-1] < 100

    idx = pd.bdate_range(end="2026-09-23", periods=60)
    c = np.full(60, 50_000.0)
    flat_px = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": np.ones(60)}, index=idx)
    notes: list[str] = []
    t = compute_technicals(flat_px, None, warnings=notes)
    assert t.rsi14 is None and any("RSI 산출 불가" in n for n in notes)
    assert t.atr == 0.0 and t.realized_vol_60d == 0.0 and t.max_drawdown_1y == 0.0


def test_infinite_high_low_are_repaired_and_reported():
    # regression: +inf passed the `> 0` check -> high_52w / resistance = inf and no note
    px = make_prices(n=300, seed=12)
    px.iloc[-5, px.columns.get_loc("high")] = np.inf
    px.iloc[-8, px.columns.get_loc("low")] = -np.inf
    high, low, n_fixed = sanitize_high_low(px)
    assert n_fixed == 2 and np.isfinite(high).all() and np.isfinite(low).all()
    assert high.iloc[-5] == px["close"].iloc[-5]
    notes: list[str] = []
    t = compute_technicals(px, None, warnings=notes)
    assert math.isfinite(t.high_52w) and math.isfinite(t.resistance) and math.isfinite(t.atr)
    assert t.high_52w >= t.last_close
    assert any("고가/저가 이상값 2행" in n for n in notes)
    assert np.isfinite(true_range(px).iloc[1:]).all()


def test_exact_row_offsets_on_arange_close():
    """close at position i equals i + 1, so every lookback reveals exactly which rows it used."""
    n = 400
    c = np.arange(1.0, n + 1.0)
    idx = pd.bdate_range(end="2026-09-23", periods=n)
    px = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0}, index=idx)
    t = compute_technicals(px, None)
    assert t.ret_1m == pytest.approx(n / (n - 21) - 1)  # close[t] / close[t-21]
    assert t.ret_3m == pytest.approx(n / (n - 63) - 1)  # close[t] / close[t-63]
    assert t.ret_12m_ex1m == pytest.approx((n - 21) / (n - 252) - 1)  # close[t-21] / close[t-252]
    assert t.ma50 == pytest.approx(np.mean(c[-50:])) and t.ma200 == pytest.approx(np.mean(c[-200:]))
    assert t.low_52w == pytest.approx((n - 251) * 0.99)  # 252 sessions incl. today
    assert t.support == pytest.approx((n - 19) * 0.99) and t.resistance == pytest.approx(n * 1.01)
    assert t.max_drawdown_1y == 0.0


def test_beta_window_and_extra_benchmark_dates():
    rng = np.random.default_rng(3)
    n = 600
    idx = pd.bdate_range(end="2026-09-23", periods=n)
    b = rng.normal(0.0, 0.01, n)
    k = np.where(np.arange(n) >= n - 252, 1.5, 0.3)  # regime change 252 returns ago
    bench = pd.Series(3000.0 * np.exp(np.cumsum(b)), index=idx)
    stock = pd.Series(100.0 * np.exp(np.cumsum(k * b)), index=idx)
    assert beta(stock, bench) == pytest.approx(1.5, abs=1e-9)  # only the last 252 returns
    # stock rows missing (halt) while the benchmark traded: the extra benchmark dates are skipped and
    # both legs take the same multi-day return, so the relation stays exact
    k2 = np.where(np.arange(n) >= n - 400, 1.5, 0.3)
    halted = pd.Series(100.0 * np.exp(np.cumsum(k2 * b)), index=idx).drop(idx[-150::9])
    assert beta(halted, bench) == pytest.approx(1.5, abs=1e-9)


def test_look_ahead_invariance():
    px = make_prices(n=900, seed=10)
    bench = make_prices(n=900, seed=11, start_price=2500.0)
    t_cut = px.index[700]
    px_t, bench_t = px.loc[:t_cut], bench.loc[:t_cut]

    # future benchmark rows (even wild ones) must not change anything at t
    wild_future = bench.copy()
    wild_future.loc[wild_future.index > t_cut, ["open", "high", "low", "close"]] *= 3.0
    base = compute_technicals(px_t, bench_t)
    assert compute_technicals(px_t, wild_future) == base
    assert compute_technicals(px_t, bench) == base

    # helper series are causal: value at t on the full history == last value on the truncated history
    assert atr(px, 14).loc[t_cut] == pytest.approx(atr(px_t, 14).iloc[-1], rel=1e-12)
    assert realized_vol(px["close"]).loc[t_cut] == pytest.approx(realized_vol(px_t["close"]).iloc[-1], rel=1e-12)
    assert rsi(px["close"]).loc[t_cut] == pytest.approx(rsi(px_t["close"]).iloc[-1], rel=1e-12)
