"""Offline tests for quantcio.macro (synthetic series + one saved Naver response; no network)."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantcio.config import load_settings
from quantcio.data.cache import DiskCache
from quantcio.macro import (
    LABEL_NA,
    LABEL_NEUTRAL,
    LABEL_RISK_OFF,
    LABEL_RISK_ON,
    MacroWarning,
    conditional_stats,
    current_regimes,
    exposure_multiplier,
    load_macro_series,
    macro_backtest,
    regime_history,
    regime_label,
    write_macro_report,
)
from quantcio.macro import backtest as mbt
from quantcio.macro import data as mdata
from quantcio.macro import regime as mreg

FIX = Path(__file__).parent / "fixtures" / "macro"
END = "2026-09-24"


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def _walk(idx: pd.DatetimeIndex, start: float, vol: float, seed: int, drift: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(start * np.exp(np.cumsum(rng.normal(drift, vol, len(idx)))), index=idx)


def _level(idx: pd.DatetimeIndex, start: float, vol: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(start + np.cumsum(rng.normal(0.0, vol, len(idx))), index=idx)


def synthetic_series(end: str = END, years: int = 8) -> dict[str, pd.Series]:
    us = pd.bdate_range(end=end, periods=252 * years)
    kr = us[np.arange(len(us)) % 37 != 5]  # a different holiday calendar
    kr_bonds = kr[kr >= kr[-1] - pd.Timedelta(3 * 365, "D")]  # ~3 years only, like Naver
    rng = np.random.default_rng(99)
    vix = pd.Series(15.0 * np.exp(0.3 * np.sin(np.arange(len(us)) / 40.0) + rng.normal(0, 0.05, len(us))), index=us)
    return {
        "acwi": _walk(us, 100.0, 0.009, 1, 0.0003),
        "spy": _walk(us, 300.0, 0.010, 2, 0.0004),
        "ewy": _walk(us, 60.0, 0.013, 3),
        "hyg": _walk(us, 80.0, 0.004, 4),
        "lqd": _walk(us, 110.0, 0.004, 5),
        "copper": _walk(us, 3.0, 0.014, 6),
        "gold": _walk(us, 1500.0, 0.009, 7),
        "wti": _walk(us, 60.0, 0.02, 8),
        "dxy": _walk(us, 95.0, 0.004, 9),
        "vix": vix,
        "us10y": _level(us, 3.0, 0.05, 10),
        "us3m": _level(us, 2.0, 0.03, 11),
        "usdkrw": _walk(us, 1200.0, 0.005, 12),
        "kospi": _walk(kr, 2500.0, 0.012, 13, 0.0002),
        "kr3y": _level(kr_bonds, 3.5, 0.03, 14),
        "kr10y": _level(kr_bonds, 3.8, 0.03, 15),
    }


@pytest.fixture(scope="module")
def synth():
    return mdata.series_from_mapping(synthetic_series())


# --------------------------------------------------------------------------------- indicator table


def test_indicator_table_has_fixed_documented_signs():
    keys = [s.key for s in mreg.INDICATORS]
    assert len(keys) == len(set(keys))
    for spec in mreg.INDICATORS:
        assert spec.sign in (-1, 1)
        assert spec.rationale
        assert set(spec.regions) <= set(mreg.REGIONS)
        assert all(k in mdata.SERIES_BY_KEY for k in spec.inputs)
        assert spec.key in (mreg.__doc__ or "")  # every sign is documented in the module docstring
        if spec.one_sided:
            assert spec.sign == -1
    assert {s.key for s in mreg.INDICATORS if s.one_sided} == {"g_oil_3m", "us_10y_3m"}
    assert [len(mreg.region_indicators(r)) for r in mreg.REGIONS] == [6, 4, 6]
    assert "dxy_3m" in [s.key for s in mreg.region_indicators("kr")]


def test_sign_adjust_one_sided():
    z = pd.Series([-2.0, 0.0, 1.5])
    assert list(mreg.sign_adjust(z, -1, True)) == [0.0, 0.0, -1.5]
    assert list(mreg.sign_adjust(z, -1, False)) == [2.0, 0.0, -1.5]
    assert mreg.sign_adjust(-2.0, -1, True) == 0.0


# --------------------------------------------------------------------------------- point in time


def test_rolling_z_matches_manual_and_ignores_future():
    idx = pd.bdate_range("2018-01-01", periods=1500)
    x = _level(idx, 0.0, 1.0, 3)
    z = mreg.rolling_z(x, 3)
    t = idx[1200]
    win = x[(x.index > t - pd.Timedelta(int(round(365.25 * 3)), "D")) & (x.index <= t)]
    manual = (x[t] - win.mean()) / win.std(ddof=1)
    assert z[t] == pytest.approx(float(np.clip(manual, -3, 3)))
    z_trunc = mreg.rolling_z(x.loc[:t], 3)
    pd.testing.assert_series_equal(z.loc[:t], z_trunc)
    assert z.iloc[:251].isna().all() and z.iloc[251:].notna().all()  # min 252 observations
    assert z.abs().max() <= 3.0


def test_change_3m_is_calendar_based_and_causal():
    idx = pd.to_datetime(["2024-01-02", "2024-02-01", "2024-04-01", "2024-04-02", "2024-07-15"])
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0], index=idx)
    c = mreg.change_3m(x)
    # 2024-04-02 - 91D = 2024-01-02 -> 4 - 1; 2024-07-15 - 91D = 2024-04-15 -> last obs 04-02 -> 10 - 4
    assert np.isnan(c.iloc[0]) and np.isnan(c.iloc[1]) and np.isnan(c.iloc[2])
    assert c.iloc[3] == 3.0 and c.iloc[4] == 6.0


def test_history_is_point_in_time(settings, synth):
    full = regime_history(settings, None, series=synth)
    cut = pd.Timestamp("2025-03-19")
    trunc = {k: s.loc[:cut] for k, s in synth.items()}
    part = regime_history(settings, None, series=mdata.series_from_mapping(trunc))
    m = full.merge(part, on=["date", "region"], suffixes=("_full", "_cut"))
    assert len(m) == len(part) and m["date"].max() <= cut
    np.testing.assert_allclose(m["score_full"].to_numpy(), m["score_cut"].to_numpy(), equal_nan=True)
    assert (m["label_full"] == m["label_cut"]).all()
    recent = full[full["date"] >= "2023-01-01"]
    assert recent["score"].notna().all()  # every region scored once z history exists
    assert set(full.columns) >= {"date", "region", "score", "label"}


def test_current_regimes_consistent_with_history(settings, synth):
    hist = regime_history(settings, None, series=synth)
    d = pd.Timestamp("2026-06-30")
    regs = current_regimes(settings, None, as_of=d, series=synth)
    later = {k: s.copy() for k, s in synth.items()}
    for s in later.values():  # future data must not matter
        s.loc[s.index > d] *= 3.0
    regs_mod = current_regimes(settings, None, as_of=d, series=mdata.series_from_mapping(later))
    for r, r2 in zip(regs, regs_mod, strict=True):
        h = hist[(hist["date"] == d) & (hist["region"] == r.region)].iloc[0]
        assert r.score == pytest.approx(h["score"])
        assert r.label == h["label"]
        assert r2.score == pytest.approx(r.score)
        assert -1.0 <= r.score <= 1.0 and 0.6 <= r.exposure_mult <= 1.0
        assert r.summary and r.as_of <= d.date()
        for ind in r.indicators:
            assert ind.z is None or -3.0 <= ind.z <= 3.0
    kr = next(r for r in regs if r.region == "kr")
    assert {i.key for i in kr.indicators} == {s.key for s in mreg.region_indicators("kr")}


# --------------------------------------------------------------------------------- staleness / missing


def test_stale_indicator_is_flagged_and_excluded(settings, synth):
    as_of = synth["spy"].index[-1]
    data = dict(synth)
    data["vix"] = synth["vix"].loc[: as_of - pd.Timedelta(14, "D")]  # 10 business days old
    regs = current_regimes(settings, None, as_of=as_of, series=mdata.series_from_mapping(data))
    g = next(r for r in regs if r.region == "global")
    assert "g_vix" in g.stale
    vix = next(i for i in g.indicators if i.key == "g_vix")
    assert vix.z is None and vix.value is not None and vix.as_of < as_of.date()
    used = [i.z for i in g.indicators if i.z is not None]
    assert len(used) == 5
    assert g.score == pytest.approx(float(np.mean(np.clip(np.array(used) / 2, -1, 1))))
    # 3 business days old is still fresh
    data["vix"] = synth["vix"].loc[: as_of - pd.Timedelta(3, "D")]
    g2 = next(r for r in current_regimes(settings, None, as_of=as_of, series=data) if r.region == "global")
    assert "g_vix" not in g2.stale


def test_half_missing_gives_nan_score(settings, synth, caplog):
    as_of = synth["spy"].index[-1]
    only_spy = {k: v for k, v in synth.items() if k != "spy"}
    us = next(r for r in current_regimes(settings, None, as_of=as_of, series=only_spy) if r.region == "us")
    assert math.isfinite(us.score)  # 1 of 4 missing -> still scored

    half = {k: v for k, v in synth.items() if k not in ("spy", "us3m")}  # SPY trend + curve missing = 2 of 4
    with caplog.at_level("WARNING"):
        regs = current_regimes(settings, None, as_of=as_of, series=half)
    us = next(r for r in regs if r.region == "us")
    assert us.score is None and any("점수 산출 불가" in w for w in us.warnings)
    assert us.label == LABEL_NA and us.exposure_mult == 1.0
    assert "경고" in us.summary
    assert any("score unavailable" in rec.message for rec in caplog.records)
    missing = [i for i in us.indicators if i.key in ("us_spy_ma200", "us_curve_10y_3m")]
    assert all(i.z is None and i.value is None for i in missing)
    hist = regime_history(settings, None, series=half)
    assert hist[hist["region"] == "us"]["score"].isna().all()


# --------------------------------------------------------------------------------- overlay / labels


def test_exposure_multiplier(settings):
    assert exposure_multiplier(0.5, settings) == 1.0
    assert exposure_multiplier(0.0, settings) == 1.0
    assert exposure_multiplier(-1.0, settings) == pytest.approx(0.6)
    assert exposure_multiplier(-0.5, settings) == pytest.approx(0.8)
    assert exposure_multiplier(-3.0, settings) == pytest.approx(0.6)
    with pytest.warns(MacroWarning):
        assert exposure_multiplier(float("nan"), settings) == 1.0
    off = settings.model_copy(deep=True)
    off.macro.overlay = False
    assert exposure_multiplier(-1.0, off) == 1.0


def test_label_thresholds(settings):
    assert regime_label(-0.25, settings) == LABEL_RISK_OFF
    assert regime_label(-0.2499, settings) == LABEL_NEUTRAL
    assert regime_label(0.0, settings) == LABEL_NEUTRAL
    assert regime_label(0.2499, settings) == LABEL_NEUTRAL
    assert regime_label(0.25, settings) == LABEL_RISK_ON
    assert regime_label(float("nan"), settings) == LABEL_NA
    assert regime_label(None, settings) == LABEL_NA


# --------------------------------------------------------------------------------- backtest arithmetic


def test_conditional_stats_arithmetic():
    idx = pd.date_range("2020-01-31", periods=7, freq="ME")
    ret = pd.Series([0.01, 0.03, -0.02, 0.04, -0.01, 0.02, np.nan], index=idx)
    lab = pd.Series(
        [LABEL_RISK_ON, LABEL_RISK_ON, LABEL_RISK_OFF, LABEL_RISK_ON, LABEL_RISK_OFF, None, "중립"], index=idx
    )
    st = conditional_stats(ret, lab)
    on = np.array([0.01, 0.03, 0.04])
    off = np.array([-0.02, -0.01])
    s_on = st["labels"][LABEL_RISK_ON]
    assert s_on["n"] == 3 and s_on["mean"] == pytest.approx(on.mean())
    assert s_on["t"] == pytest.approx(on.mean() / (on.std(ddof=1) / math.sqrt(3)))
    assert s_on["hit_rate"] == pytest.approx(1.0)
    assert st["labels"][LABEL_RISK_OFF]["n"] == 2
    assert st["labels"][LABEL_NA]["n"] == 1  # a None label counts as 데이터 부족
    assert st["labels"][LABEL_NEUTRAL]["n"] == 0  # its return is NaN -> dropped
    assert st["all"]["n"] == 6
    d = st["on_minus_off"]
    se = math.sqrt(on.var(ddof=1) / 3 + off.var(ddof=1) / 2)
    assert d["diff"] == pytest.approx(on.mean() - off.mean())
    assert d["t"] == pytest.approx((on.mean() - off.mean()) / se)


def test_next_period_returns_timing():
    sessions = pd.bdate_range("2024-01-01", "2024-03-29")
    close = pd.Series(np.arange(1, len(sessions) + 1, dtype=float), index=sessions)
    sig = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-31"])
    ee = mbt.entry_exit(sessions, sig)
    assert ee.loc["2024-01-31", "entry"] == pd.Timestamp("2024-02-01")
    assert ee.loc["2024-01-31", "exit"] == pd.Timestamp("2024-02-29")
    r = mbt.next_period_returns(close, sig)
    assert r["2024-01-31"] == pytest.approx(close["2024-02-29"] / close["2024-02-01"] - 1)
    assert r["2024-02-29"] == pytest.approx(close["2024-03-29"] / close["2024-03-01"] - 1)
    assert np.isnan(r["2024-03-31"])  # no next signal -> no complete month


def test_overlay_returns_and_perf_stats():
    sessions = pd.bdate_range("2024-01-01", "2024-06-28")
    close = _walk(sessions, 100.0, 0.01, 5)
    sig = pd.date_range("2024-01-31", "2024-05-31", freq="ME")
    full = mbt.overlay_returns(close, pd.Series(1.0, index=sig))
    np.testing.assert_allclose(full["overlay"], full["bench"])
    assert full.index[0] == pd.Timestamp("2024-02-02")  # set at the 02-01 close, earns from 02-02
    half = mbt.overlay_returns(close, pd.Series(0.5, index=sig))
    np.testing.assert_allclose(half["overlay"], 0.5 * half["bench"])
    assert mbt.perf_stats(half["overlay"])["vol"] == pytest.approx(0.5 * mbt.perf_stats(half["bench"])["vol"])

    r = pd.Series([0.10, -0.20, 0.05], index=pd.bdate_range("2024-01-01", periods=3))
    st = mbt.perf_stats(r)
    assert st["mdd"] == pytest.approx(-0.20)
    assert st["total_return"] == pytest.approx(1.1 * 0.8 * 1.05 - 1)


def test_verdict_wording():
    bh = {"sharpe": 0.50, "mdd": -0.30}
    assert mbt.verdict(bh, {"sharpe": 0.52, "mdd": -0.29})["text"].startswith(mbt.VERDICT_NO_EDGE)
    v = mbt.verdict(bh, {"sharpe": 0.51, "mdd": -0.22})
    assert v["text"].startswith(mbt.VERDICT_NO_EDGE) and v["mdd_improved"] and not v["sharpe_improved"]
    assert mbt.verdict(bh, {"sharpe": 0.65, "mdd": -0.25})["sharpe_improved"]


def test_macro_backtest_and_report(settings, synth, tmp_path):
    bt = macro_backtest(settings, DiskCache(tmp_path / "c", enabled=False), series=synth, today=pd.Timestamp(END))
    assert set(bt["markets"]) == {"KR", "US"}
    for res in bt["markets"].values():
        cond = next(iter(res["conditional"].values()))
        assert sum(v["n"] for v in cond["labels"].values()) == cond["all"]["n"] == res["period"]["months_with_return"]
        assert 0.6 <= res["overlay"]["avg_exposure"] <= 1.0
        assert res["overlay"]["verdict"]["text"]
    regs = current_regimes(settings, None, as_of=END, series=synth)
    regs[1] = regs[1].model_copy(update={"score": float("nan"), "label": LABEL_NA})  # NaN must serialise as null
    md, js = write_macro_report(regs, bt, tmp_path / "out")
    text = md.read_text(encoding="utf-8")
    assert "# 매크로 레짐 리포트" in text and "오버레이 vs 바이앤홀드" in text and "판정" in text
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["regimes"][1]["score"] is None
    assert {r["region"] for r in payload["regimes"]} == {"global", "us", "kr"}


# --------------------------------------------------------------------------------- data layer


def test_parse_naver_bond_fixture():
    rows = json.loads((FIX / "naver_bond_KR3YT_page1.json").read_text(encoding="utf-8"))
    df = mdata.parse_naver_bond_prices(rows)
    assert len(df) == 10 and df.index.is_monotonic_increasing and df.index.is_unique
    assert df.index[-1] == pd.Timestamp("2026-09-23") and df.index.tz is None
    assert df["close"].iloc[-1] == pytest.approx(3.998)
    assert df["close"].between(2.0, 6.0).all()
    assert mdata.parse_naver_bond_prices([]).empty
    with pytest.raises(ValueError):
        mdata.parse_naver_bond_prices({"error": "x"})


def test_drop_live_bond_row():
    rows = json.loads((FIX / "naver_bond_KR3YT_page1.json").read_text(encoding="utf-8"))
    df = mdata.parse_naver_bond_prices(rows)
    intraday = datetime(2026, 9, 23, 10, 0, tzinfo=mdata.KST)
    evening = datetime(2026, 9, 23, 18, 0, tzinfo=mdata.KST)
    assert mdata.drop_live_bond_row(df, intraday).index[-1] < pd.Timestamp("2026-09-23")
    assert mdata.drop_live_bond_row(df, evening).index[-1] == pd.Timestamp("2026-09-23")


def test_load_macro_series_records_failures_and_caches(settings, monkeypatch, tmp_path):
    synthetic = synthetic_series()
    synthetic["wti"].iloc[100] = -37.63  # the 2020-04-20 case: dropped and reported
    calls: list[str] = []

    def fake(spec, start, settings_, cache_):
        calls.append(spec.key)
        if spec.key == "kr3y":
            raise ConnectionError("naver down")
        if spec.key not in synthetic:
            return synthetic["spy"] * 0.5, f"fake {spec.symbol}", []
        return synthetic[spec.key], f"fake {spec.symbol}", ["note"]

    monkeypatch.setattr(mdata, "_fetch_series", fake)
    cache = DiskCache(tmp_path / "cache")
    out = load_macro_series(settings, cache, max_workers=1)
    assert "kr3y" not in out and "ConnectionError" in out.failures["kr3y"]
    assert any("수집 실패" in n for n in out.notes)
    assert (out["wti"] > 0).all() and any("0 이하 값 1일 제외" in n for n in out.notes)
    assert out["spy"].attrs["source"] == "fake SPY" and out.sources["kospi"] == "fake KOSPI"
    assert all(s.index.tz is None and s.index.is_monotonic_increasing for s in out.values())
    n_first = len(calls)
    out2 = load_macro_series(settings, cache, max_workers=1)
    assert calls[n_first:] == ["kr3y"]  # successes served from cache, the failure retried
    assert set(out2) == set(out)
    load_macro_series(settings, cache, max_workers=1, use_cache=False)
    assert len(calls) == n_first + 1 + len(mdata.SERIES)
