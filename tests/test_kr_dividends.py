"""KR total-return labels: dividend yields and the label-window sums (docs/MODELING.md 52.1)."""

import numpy as np
import pandas as pd

from quantcio.data.cache import DiskCache
from quantcio.data.pit import kr_dividends as kdiv


def _hist() -> pd.DataFrame:
    idx = pd.bdate_range("2024-12-23", periods=8)
    close = [100.0, 101.0, 102.0, 99.0, 99.5, 100.0, 100.5, 101.0]
    div = [0.0, 0.0, 0.0, 2.04, 0.0, 0.0, 0.0, 0.0]  # ex-date = 4th session, 2% of the previous close (102)
    return pd.DataFrame({"close": close, "div": div}, index=idx)


def test_dividend_yield_uses_the_previous_close():
    y = kdiv.dividend_yields(_hist())
    assert len(y) == 1
    assert y.index[0] == pd.bdate_range("2024-12-23", periods=8)[3]
    assert abs(y.iloc[0] - 2.04 / 102.0) < 1e-12


def test_window_sum_counts_ex_dates_inside_t0_exclusive_t1_inclusive():
    y = kdiv.dividend_yields(_hist())
    ex = y.index[0].to_datetime64()
    one = np.timedelta64(1, "D")
    t0 = np.array([ex - 3 * one, ex, ex - one, ex - 10 * one], dtype="datetime64[ns]")
    t1 = np.array([ex + 3 * one, ex + 3 * one, ex, np.datetime64("NaT", "ns")], dtype="datetime64[ns]")
    got = kdiv.window_dividend_log(y, t0, t1)
    want = np.log1p(y.iloc[0])
    assert np.allclose(got, [want, 0.0, want, 0.0])


def test_load_uses_the_listing_market_first_and_caches(tmp_path):
    cache = DiskCache(tmp_path)
    calls = []

    def fetch(sym):
        calls.append(sym)
        return _hist() if sym.endswith(".KQ") else None

    from quantcio.config import load_settings

    s = load_settings()
    out, cov = kdiv.load_kr_dividends({"123456": "KOSDAQ"}, s, cache, fetch=fetch)
    assert calls == ["123456.KQ"]
    assert cov["with_dividends"] == 1 and "123456" in out
    out2, _ = kdiv.load_kr_dividends({"123456": "KOSDAQ"}, s, cache, fetch=fetch)
    assert calls == ["123456.KQ"]  # second call served from the cache
    assert np.isclose(out2["123456"].iloc[0], out["123456"].iloc[0])


def test_missing_yahoo_symbol_keeps_price_returns(tmp_path):
    from quantcio.config import load_settings

    out, cov = kdiv.load_kr_dividends({"654321": "KOSPI"}, load_settings(), DiskCache(tmp_path), fetch=lambda s: None)
    assert out == {} and cov["with_yahoo"] == 0
