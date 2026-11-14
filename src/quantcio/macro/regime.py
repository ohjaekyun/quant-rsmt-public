"""Macro regime scores for the global, US and Korean markets (docs/BACKTEST_MACRO.md section 4).

Method
------
1. Each indicator turns one or more daily series into a signal ``x_t`` that uses data ``<= t`` only.
   The pieces are 200-session moving averages, 3-month changes and series aligned on a union calendar
   with forward fill.
2. ``z_t = (x_t - mean) / std`` over the trailing ``macro.z_window_years`` window. The window is a time
   window ending at t that includes t, with at least :data:`Z_MIN_OBS` observations; before 3 years of
   data exist the window is expanding. z is clipped to +-3.
3. Sign adjustment uses the fixed a-priori sign in the table below, where + = risk-on. It was chosen
   before looking at any backtest and never fitted. A *one-sided* indicator only counts in its adverse
   direction: ``z_adj = min(sign * z, 0)``. Its opposite move is economically ambiguous, e.g. an oil or
   10Y slump can be disinflation or a demand shock, so it contributes 0.
4. The region score is the mean of ``clip(z_adj / 2, -1, 1)`` over the region's usable indicators.
   An indicator is unusable when an input series is missing, its z has too little history, or it is
   **stale**: its latest input observation is more than :data:`STALE_SESSIONS` business days before
   the evaluation date. When at least half of a region's indicators are unusable, the score is NaN,
   the label is "데이터 부족" and a warning is logged and written into the summary.
5. Labels: score <= ``risk_off_threshold`` -> 위험회피; score >= ``risk_on_threshold`` -> 위험선호;
   otherwise 중립.
6. Overlay: :func:`exposure_multiplier` is 1.0 for score >= 0 and falls linearly to
   ``overlay_min_mult`` at score -1.

Point-in-time: every transform is causal (rolling / shift / ffill on the past), so truncating the data
after t never changes a score at or before t. ``tests/test_macro.py`` checks this. :func:`current_regimes`
and :func:`regime_history` share :func:`evaluate_indicators`, so today's value is the same number the
history would record for today.

Indicators (``x`` = z-scored signal; value/change_3m units are described after the table)
-----------------------------------------------------------------------------------------
============================  ========================================  ======  ===========================
key (regions)                 signal x                                  sign    a-priori reason
============================  ========================================  ======  ===========================
g_acwi_ma200 (global)         ACWI / MA200 - 1                          +       global equity trend
g_copper_gold_3m (global)     3M log change of copper/gold (HG/GC)      +       growth vs safety demand
g_oil_3m (global)             3M log change of WTI (CL=F)               - (1s)  oil spike = inflation/cost
                                                                                shock ("과열 시 −")
g_vix (global)                VIX level                                 -       implied equity risk
g_credit_3m (global)          3M log change of HYG/LQD                  +       HY outperforming IG =
                                                                                credit risk appetite
dxy_3m (global, us, kr)       3M log change of DXY                      -       strong dollar = tighter
                                                                                global USD liquidity
us_curve_10y_3m (us)          ^TNX - ^IRX (%p)                          +       inversion = recession lead
us_10y_3m (us)                3M change of ^TNX (%p)                    - (1s)  yield spike = tighter
                                                                                financial conditions
us_spy_ma200 (us)             SPY / MA200 - 1                           +       US equity trend
kr_usdkrw_3m (kr)             3M log change of USD/KRW                  -       KRW weakness = outflows
kr_ktb3y_3m (kr)              3M change of KTB 3Y (%p)                  -       rising KR rates = tighter
                                                                                domestic conditions
kr_us_spread (kr)             KTB 10Y - UST 10Y (%p)                    +       deeper negative gap =
                                                                                capital-outflow pressure
kr_kospi_rel_acwi_3m (kr)     3M log ret KOSPI - 3M log ret ACWI        +       Korea relative momentum
kr_ewy_ma200 (kr)             EWY / MA200 - 1                           +       USD-based (foreign
                                                                                investor) Korea trend
============================  ========================================  ======  ===========================

(1s) = one-sided (adverse direction only). "3M" = 91 calendar days, looked up as-of so that series
on different calendars are comparable. MA200 = 200 observations of the series' own calendar.

MacroIndicator units: ``value`` is the latest level in ``unit``. ``change_3m`` is in the SAME unit for
"%", "%p" and "pt"; for "원", "달러", "지수" and "비율" it is a decimal return. All MA-gap values are
in % (5.2 = 5.2% above the 200-day average).

Known limits: Korean bond yields have only ~3 years of keyless history (Naver, from 2023-09-26). The
two KR rate indicators therefore start contributing only in 2024 (spread) / 2025 (3Y change), and KR
history before that is scored on the four remaining indicators. Futures are front-month continuous
series, so roll gaps enter the copper/gold and oil changes.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import numpy as np
import pandas as pd

from quantcio.config import Settings
from quantcio.data.cache import DiskCache
from quantcio.i18n import lang_of, tr
from quantcio.macro.data import KST, MacroSeries, load_macro_series
from quantcio.models import MacroIndicator, MacroRegime

log = logging.getLogger(__name__)

Region = Literal["global", "us", "kr"]
REGIONS: tuple[Region, ...] = ("global", "us", "kr")
REGION_LABELS = {"global": "글로벌", "us": "미국", "kr": "한국"}
REGION_LABELS_EN = {"global": "Global", "us": "US", "kr": "Korea"}
LABELS_EN = {"위험선호": "Risk-on", "중립": "Neutral", "위험회피": "Risk-off", "데이터 부족": "No data"}
INDICATOR_LABELS_EN = {
    "g_acwi_ma200": "ACWI vs MA200",
    "g_copper_gold_3m": "Copper/gold 3M",
    "g_oil_3m": "WTI 3M",
    "g_vix": "VIX",
    "g_credit_3m": "Credit HYG/LQD 3M",
    "dxy_3m": "DXY 3M",
    "us_curve_10y_3m": "US 10Y−3M",
    "us_10y_3m": "US 10Y 3M chg",
    "us_spy_ma200": "SPY vs MA200",
    "kr_usdkrw_3m": "USD/KRW 3M",
    "kr_ktb3y_3m": "KTB 3Y 3M chg",
    "kr_us_spread": "KR−US 10Y",
    "kr_kospi_rel_acwi_3m": "KOSPI vs ACWI 3M",
    "kr_ewy_ma200": "EWY vs MA200",
}

LABEL_RISK_ON = "위험선호"
LABEL_NEUTRAL = "중립"
LABEL_RISK_OFF = "위험회피"
LABEL_NA = "데이터 부족"
LABELS = (LABEL_RISK_ON, LABEL_NEUTRAL, LABEL_RISK_OFF, LABEL_NA)

MA_WINDOW = 200  # observations
CHANGE_DAYS = 91  # "3M" in calendar days (~63 sessions)
Z_MIN_OBS = 252  # minimum observations inside the trailing z window
Z_CLIP = 3.0
STALE_SESSIONS = 5  # business days


class MacroWarning(RuntimeWarning):
    """Emitted when a caller uses a NaN macro score (the overlay then falls back to 1.0)."""


# ----------------------------------------------------------------------------------- transforms


def ma_gap_pct(s: pd.Series, window: int = MA_WINDOW) -> pd.Series:
    """100 * (s / trailing ``window``-observation mean - 1)."""
    return 100.0 * (s / s.rolling(window, min_periods=window).mean() - 1.0)


def change_3m(x: pd.Series, days: int = CHANGE_DAYS) -> pd.Series:
    """x_t - x at the last observation on or before t - ``days`` (calendar), NaN before that exists."""
    x = x.dropna()
    if x.empty:
        return x
    prev = x.reindex(x.index - pd.Timedelta(days, "D"), method="ffill").to_numpy()
    return pd.Series(x.to_numpy() - prev, index=x.index)


def aligned(*series: pd.Series) -> pd.DataFrame:
    """Union calendar, each column forward-filled from its own past (rows before every column exists
    are dropped)."""
    df = pd.concat([s.rename(str(i)) for i, s in enumerate(series)], axis=1).sort_index().ffill()
    return df.dropna()


def rolling_z(x: pd.Series, window_years: float, min_obs: int = Z_MIN_OBS) -> pd.Series:
    """Trailing time-window z-score (window ends at and includes t), clipped to +-Z_CLIP."""
    x = x.dropna()
    if x.empty:
        return x
    win = f"{int(round(365.25 * window_years))}D"
    roll = x.rolling(win, min_periods=min_obs)
    mean, std = roll.mean(), roll.std(ddof=1)
    z = (x - mean) / std.where(std > 1e-12)
    return z.clip(-Z_CLIP, Z_CLIP)


def sign_adjust(z: pd.Series | float, sign: int, one_sided: bool) -> pd.Series | float:
    """+ = risk-on. One-sided indicators keep only the adverse (negative after signing) side."""
    adj = z * sign
    if one_sided:
        return adj.clip(upper=0.0) if isinstance(adj, pd.Series) else min(adj, 0.0)
    return adj


# ----------------------------------------------------------------------------------- indicator table

Built = tuple[pd.Series, pd.Series, pd.Series | None]  # (signal x, display value, change_3m)


def _b_ma_gap(key: str) -> Callable[[dict[str, pd.Series]], Built]:
    def build(d: dict[str, pd.Series]) -> Built:
        x = ma_gap_pct(d[key])
        return x, x, change_3m(x)

    return build


def _b_log_change(key: str) -> Callable[[dict[str, pd.Series]], Built]:
    def build(d: dict[str, pd.Series]) -> Built:
        s = d[key]
        x = change_3m(np.log(s))
        return x, s, np.expm1(x)

    return build


def _b_ratio_change(num: str, den: str, scale: float = 1.0) -> Callable[[dict[str, pd.Series]], Built]:
    def build(d: dict[str, pd.Series]) -> Built:
        df = aligned(d[num], d[den])
        ratio = df.iloc[:, 0] / df.iloc[:, 1]
        x = change_3m(np.log(ratio))
        return x, ratio * scale, np.expm1(x)

    return build


def _b_level(key: str) -> Callable[[dict[str, pd.Series]], Built]:
    def build(d: dict[str, pd.Series]) -> Built:
        s = d[key]
        return s, s, change_3m(s)

    return build


def _b_diff_change(key: str) -> Callable[[dict[str, pd.Series]], Built]:
    def build(d: dict[str, pd.Series]) -> Built:
        s = d[key]
        x = change_3m(s)
        return x, s, x

    return build


def _b_spread(a: str, b: str) -> Callable[[dict[str, pd.Series]], Built]:
    def build(d: dict[str, pd.Series]) -> Built:
        df = aligned(d[a], d[b])
        x = df.iloc[:, 0] - df.iloc[:, 1]
        return x, x, change_3m(x)

    return build


def _b_relative(a: str, b: str) -> Callable[[dict[str, pd.Series]], Built]:
    def build(d: dict[str, pd.Series]) -> Built:
        ra = 100.0 * change_3m(np.log(d[a]))  # each on its own calendar first
        rb = 100.0 * change_3m(np.log(d[b]))
        df = aligned(ra, rb)
        x = df.iloc[:, 0] - df.iloc[:, 1]
        return x, x, None

    return build


@dataclass(frozen=True)
class IndicatorSpec:
    key: str
    label: str  # Korean display label
    regions: tuple[Region, ...]
    inputs: tuple[str, ...]  # MacroSeries keys
    unit: str
    sign: int  # a-priori: +1 = higher x is risk-on, -1 = higher x is risk-off
    one_sided: bool
    rationale: str  # a-priori economic reason (Korean)
    build: Callable[[dict[str, pd.Series]], Built]


INDICATORS: tuple[IndicatorSpec, ...] = (
    IndicatorSpec(
        "g_acwi_ma200",
        "ACWI 200일선 대비",
        ("global",),
        ("acwi",),
        "%",
        +1,
        False,
        "글로벌 주식이 200일선 위 = 추세적 위험선호",
        _b_ma_gap("acwi"),
    ),
    IndicatorSpec(
        "g_copper_gold_3m",
        "구리/금 비율 3M 변화",
        ("global",),
        ("copper", "gold"),
        "비율",
        +1,
        False,
        "경기민감 구리가 안전자산 금보다 강함 = 성장 기대",
        _b_ratio_change("copper", "gold", 1000.0),
    ),
    IndicatorSpec(
        "g_oil_3m",
        "WTI 유가 3M 변화",
        ("global",),
        ("wti",),
        "달러",
        -1,
        True,
        "유가 급등 = 인플레·비용 충격(과열 시 −). 하락은 방향이 모호해 0",
        _b_log_change("wti"),
    ),
    IndicatorSpec(
        "g_vix",
        "VIX 수준",
        ("global",),
        ("vix",),
        "pt",
        -1,
        False,
        "내재변동성 상승 = 위험회피",
        _b_level("vix"),
    ),
    IndicatorSpec(
        "g_credit_3m",
        "신용(HYG/LQD) 3M 추세",
        ("global",),
        ("hyg", "lqd"),
        "비율",
        +1,
        False,
        "하이일드가 투자등급보다 강함 = 신용 위험선호",
        _b_ratio_change("hyg", "lqd"),
    ),
    IndicatorSpec(
        "dxy_3m",
        "달러인덱스 3M 변화",
        ("global", "us", "kr"),
        ("dxy",),
        "지수",
        -1,
        False,
        "강달러 = 글로벌 달러 유동성 긴축·신흥국 자금 유출",
        _b_log_change("dxy"),
    ),
    IndicatorSpec(
        "us_curve_10y_3m",
        "미 10Y−3M 금리차",
        ("us",),
        ("us10y", "us3m"),
        "%p",
        +1,
        False,
        "장단기 금리차 역전 = 경기침체 선행",
        _b_spread("us10y", "us3m"),
    ),
    IndicatorSpec(
        "us_10y_3m",
        "미 10년물 3M 변화",
        ("us",),
        ("us10y",),
        "%",
        -1,
        True,
        "10년물 급등 = 할인율 상승·금융여건 긴축(급등 시 −). 하락은 안전자산 선호일 수 있어 0",
        _b_diff_change("us10y"),
    ),
    IndicatorSpec(
        "us_spy_ma200",
        "SPY 200일선 대비",
        ("us",),
        ("spy",),
        "%",
        +1,
        False,
        "미국 주식이 200일선 위 = 추세적 위험선호",
        _b_ma_gap("spy"),
    ),
    IndicatorSpec(
        "kr_usdkrw_3m",
        "USD/KRW 3M 변화",
        ("kr",),
        ("usdkrw",),
        "원",
        -1,
        False,
        "원화 약세 = 외국인 자금 유출",
        _b_log_change("usdkrw"),
    ),
    IndicatorSpec(
        "kr_ktb3y_3m",
        "국고채 3년 3M 변화",
        ("kr",),
        ("kr3y",),
        "%",
        -1,
        False,
        "국내 금리 상승 = 국내 금융여건 긴축",
        _b_diff_change("kr3y"),
    ),
    IndicatorSpec(
        "kr_us_spread",
        "한미 10년 금리차",
        ("kr",),
        ("kr10y", "us10y"),
        "%p",
        +1,
        False,
        "한미 금리차 역전 확대 = 자본 유출·원화 약세 압력",
        _b_spread("kr10y", "us10y"),
    ),
    IndicatorSpec(
        "kr_kospi_rel_acwi_3m",
        "KOSPI vs ACWI 3M 상대강도",
        ("kr",),
        ("kospi", "acwi"),
        "%p",
        +1,
        False,
        "한국 증시의 상대 모멘텀",
        _b_relative("kospi", "acwi"),
    ),
    IndicatorSpec(
        "kr_ewy_ma200",
        "EWY 200일선 대비",
        ("kr",),
        ("ewy",),
        "%",
        +1,
        False,
        "달러 기준 한국 주식 추세 = 외국인 관점 위험선호",
        _b_ma_gap("ewy"),
    ),
)
INDICATORS_BY_KEY = {s.key: s for s in INDICATORS}


def region_indicators(region: str) -> list[IndicatorSpec]:
    return [s for s in INDICATORS if region in s.regions]


# ----------------------------------------------------------------------------------- computation


@dataclass
class IndicatorFrame:
    spec: IndicatorSpec
    x: pd.Series | None = None
    value: pd.Series | None = None
    change: pd.Series | None = None
    z_adj: pd.Series | None = None  # sign-adjusted clipped z on x's (non-NaN) index
    input_index: dict[str, pd.DatetimeIndex] | None = None
    source: str = ""
    error: str = ""  # why the indicator cannot be computed at all


def build_indicator_frames(series: dict[str, pd.Series], settings: Settings) -> dict[str, IndicatorFrame]:
    """Full-history signal / value / change / sign-adjusted z series for every indicator (causal)."""
    out: dict[str, IndicatorFrame] = {}
    sources = getattr(series, "sources", {}) or {}
    for spec in INDICATORS:
        frame = IndicatorFrame(spec)
        missing = [k for k in spec.inputs if k not in series or series[k] is None or len(series[k]) == 0]
        if missing:
            frame.error = f"입력 결측: {', '.join(missing)}"
            out[spec.key] = frame
            continue
        try:
            x, value, change = spec.build({k: series[k] for k in spec.inputs})
            x = x.dropna()
            z = rolling_z(x, settings.macro.z_window_years)
            frame.x = x
            frame.value = value.dropna()
            frame.change = change.dropna() if change is not None else None
            frame.z_adj = sign_adjust(z, spec.sign, spec.one_sided)
            frame.input_index = {k: pd.DatetimeIndex(series[k].dropna().index) for k in spec.inputs}
            srcs = [str(sources.get(k) or series[k].attrs.get("source") or k) for k in spec.inputs]
            frame.source = " / ".join(dict.fromkeys(srcs))
        except Exception as exc:  # noqa: BLE001 - reported as a missing indicator
            log.warning("macro indicator %s failed: %s", spec.key, exc)
            frame.error = f"계산 실패: {type(exc).__name__}: {str(exc)[:120]}"
        out[spec.key] = frame
    return out


def _pos(index: pd.DatetimeIndex, dates: np.ndarray) -> np.ndarray:
    """Position of the last index entry <= each date (-1 when none)."""
    return index.searchsorted(dates, side="right") - 1


def _take(s: pd.Series | None, dates: np.ndarray) -> np.ndarray:
    """Value at the last observation <= each date (NaN when none); NaN values are NOT skipped."""
    if s is None or s.empty:
        return np.full(len(dates), np.nan)
    p = _pos(pd.DatetimeIndex(s.index), dates)
    vals = s.to_numpy(dtype="float64")
    return np.where(p >= 0, vals[np.clip(p, 0, None)], np.nan)


def evaluate_indicators(frames: dict[str, IndicatorFrame], dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Per indicator: DataFrame(index=dates) with z, value, change, obs_date, status (ok/stale/missing/nohist)."""
    d64 = pd.DatetimeIndex(dates).to_numpy(dtype="datetime64[ns]")
    day = d64.astype("datetime64[D]")
    out: dict[str, pd.DataFrame] = {}
    for key, fr in frames.items():
        df = pd.DataFrame(index=pd.DatetimeIndex(dates))
        if fr.error or fr.input_index is None:
            df["z"] = np.nan
            df["value"] = np.nan
            df["change"] = np.nan
            df["obs_date"] = pd.NaT
            df["status"] = "missing"
            out[key] = df
            continue
        last = np.full(len(d64), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        have_all = np.ones(len(d64), dtype=bool)
        for idx in fr.input_index.values():
            if len(idx) == 0:
                have_all[:] = False
                continue
            p = _pos(idx, d64)
            have_all &= p >= 0
            li = idx.to_numpy(dtype="datetime64[ns]")[np.clip(p, 0, None)]
            last = np.where(np.isnat(last), li, np.minimum(last, li))
        last = np.where(have_all, last, np.datetime64("NaT", "ns"))
        z = _take(fr.z_adj, d64)
        df["z"] = z
        df["value"] = _take(fr.value, d64)
        df["change"] = _take(fr.change, d64)
        df["obs_date"] = pd.DatetimeIndex(last)
        lag = np.full(len(d64), 10**6)
        ok_last = ~np.isnat(last)
        lag[ok_last] = np.busday_count(last[ok_last].astype("datetime64[D]"), day[ok_last])
        status = np.where(
            ~have_all, "missing", np.where(lag > STALE_SESSIONS, "stale", np.where(np.isnan(z), "nohist", "ok"))
        )
        df["status"] = status
        out[key] = df
    return out


def region_scores(evals: dict[str, pd.DataFrame], region: str) -> pd.DataFrame:
    """Score / n_used / n_total per date for one region (NaN when >= half the indicators are unusable)."""
    specs = region_indicators(region)
    comps = []
    for spec in specs:
        e = evals[spec.key]
        c = np.clip(e["z"].to_numpy(dtype="float64") / 2.0, -1.0, 1.0)
        comps.append(np.where(e["status"].to_numpy() == "ok", c, np.nan))
    mat = np.vstack(comps) if comps else np.empty((0, 0))
    n_total = len(specs)
    n_used = np.sum(~np.isnan(mat), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(mat, axis=0)
    score = np.where((n_total - n_used) * 2 >= n_total, np.nan, mean)
    index = next(iter(evals.values())).index
    return pd.DataFrame({"score": score, "n_used": n_used, "n_total": n_total}, index=index)


def regime_label(score: float | None, settings: Settings) -> str:
    if score is None or not math.isfinite(score):
        return LABEL_NA
    if score <= settings.macro.risk_off_threshold:
        return LABEL_RISK_OFF
    if score >= settings.macro.risk_on_threshold:
        return LABEL_RISK_ON
    return LABEL_NEUTRAL


def overlay_multiplier(score: float | None, min_mult: float, *, warn: bool = True) -> float:
    """1.0 for score >= 0, linear down to ``min_mult`` at score -1 (scores below -1 clamp). NaN -> 1.0."""
    if score is None or not math.isfinite(float(score)):
        if warn:
            msg = "매크로 점수 NaN(데이터 부족) → 노출 배수 1.0(오버레이 미적용)"
            log.warning(msg)
            warnings.warn(msg, MacroWarning, stacklevel=3)
        return 1.0
    s = max(float(score), -1.0)
    if s >= 0.0:
        return 1.0
    return float(1.0 + s * (1.0 - float(min_mult)))


def exposure_multiplier(score: float, settings: Settings) -> float:
    """Gross-exposure multiplier of the overlay (1.0 when ``macro.enabled`` or ``macro.overlay`` is off).

    NaN / None score -> 1.0 plus a :class:`MacroWarning` and a log warning, so the caller is told.
    """
    if not (settings.macro.enabled and settings.macro.overlay):
        return 1.0
    return overlay_multiplier(score, settings.macro.overlay_min_mult)


# ----------------------------------------------------------------------------------- public API


def _as_ts(as_of: date | datetime | str | pd.Timestamp | None) -> pd.Timestamp:
    if as_of is None:
        return pd.Timestamp(datetime.now(KST).date())
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(KST).tz_localize(None)
    return ts.normalize()


def _truncate(series: dict[str, pd.Series], as_of: pd.Timestamp) -> MacroSeries:
    out = MacroSeries()
    for k, s in series.items():
        t = s.loc[:as_of]
        t.attrs = dict(s.attrs)
        out[k] = t
    out.sources = dict(getattr(series, "sources", {}) or {})
    out.failures = dict(getattr(series, "failures", {}) or {})
    out.notes = list(getattr(series, "notes", []) or [])
    return out


def _fmt_z(z: float) -> str:
    return f"{z:+.1f}"


def _summary_en(
    region: str, score: float, label: str, parts: list[tuple[IndicatorSpec, float]], unusable: list[str]
) -> str:
    name = REGION_LABELS_EN[region]
    lab = LABELS_EN.get(label, label)
    if not math.isfinite(score):
        return (
            f"{name} macro: {len(unusable)} indicators missing or stale ({', '.join(unusable)}) - no score (warning)."
        )
    pos = sorted((p for p in parts if p[1] >= 0.5), key=lambda p: -p[1])[:2]
    neg = sorted((p for p in parts if p[1] <= -0.5), key=lambda p: p[1])[:2]
    bits = []
    if pos:
        bits.append(
            "supportive: " + ", ".join(f"{INDICATOR_LABELS_EN.get(s.key, s.key)} (z {_fmt_z(z)})" for s, z in pos)
        )
    if neg:
        bits.append(
            "headwinds: " + ", ".join(f"{INDICATOR_LABELS_EN.get(s.key, s.key)} (z {_fmt_z(z)})" for s, z in neg)
        )
    text = f"{name} macro is {lab} (score {score:+.2f}) - " + (" / ".join(bits) if bits else "no clear drivers")
    if unusable:
        text += f" (excluded: {', '.join(unusable)})"
    return text + "."


def _summary(
    region: str,
    score: float,
    label: str,
    parts: list[tuple[IndicatorSpec, float]],
    unusable: list[str],
    lang: str = "ko",
) -> str:
    if lang == "en":
        return _summary_en(region, score, label, parts, unusable)
    name = REGION_LABELS[region]
    if not math.isfinite(score):
        return (
            f"{name} 매크로: 지표 {len(unusable)}개가 결측·지연({', '.join(unusable)})이라 "
            "점수를 내지 않았습니다(경고)."
        )
    pos = sorted((p for p in parts if p[1] >= 0.5), key=lambda p: -p[1])[:2]
    neg = sorted((p for p in parts if p[1] <= -0.5), key=lambda p: p[1])[:2]
    text = f"{name} 매크로는 {label}(점수 {score:+.2f})입니다"
    bits = []
    if pos:
        bits.append("우호: " + ", ".join(f"{s.label}(z {_fmt_z(z)})" for s, z in pos))
    if neg:
        bits.append("부담: " + ", ".join(f"{s.label}(z {_fmt_z(z)})" for s, z in neg))
    if not bits:
        bits.append("뚜렷한 우호·부담 요인 없음")
    text += " — " + " / ".join(bits)
    if unusable:
        text += f" (제외: {', '.join(unusable)})"
    return text + "."


def _opt(v: float) -> float | None:
    return float(v) if v is not None and math.isfinite(float(v)) else None


def regimes_at(series: dict[str, pd.Series], settings: Settings, as_of: pd.Timestamp) -> list[MacroRegime]:
    """Regimes from pre-loaded series, using data <= ``as_of`` only."""
    data = _truncate(series, as_of)
    frames = build_indicator_frames(data, settings)
    evals = evaluate_indicators(frames, pd.DatetimeIndex([as_of]))
    regimes: list[MacroRegime] = []
    for region in REGIONS:
        sc = region_scores(evals, region)
        score = float(sc["score"].iloc[0])
        label = regime_label(score, settings)
        indicators: list[MacroIndicator] = []
        stale: list[str] = []
        unusable: list[str] = []
        parts: list[tuple[IndicatorSpec, float]] = []
        obs_dates: list[pd.Timestamp] = []
        for spec in region_indicators(region):
            e = evals[spec.key].iloc[0]
            fr = frames[spec.key]
            status = str(e["status"])
            obs = e["obs_date"]
            usable = status == "ok"
            z = _opt(e["z"]) if usable else None
            if status == "stale":
                stale.append(spec.key)
            if not usable:
                unusable.append(spec.label)
                why = {"stale": "지연", "missing": fr.error or "결측", "nohist": "z 이력 부족"}.get(status, status)
                log.warning("macro %s %s unusable on %s: %s", region, spec.key, as_of.date(), why)
            else:
                parts.append((spec, float(e["z"])))
                obs_dates.append(pd.Timestamp(obs))
            indicators.append(
                MacroIndicator(
                    key=spec.key,
                    label=spec.label,
                    value=_opt(e["value"]),
                    unit=spec.unit,
                    change_3m=_opt(e["change"]),
                    z=z,
                    as_of=pd.Timestamp(obs).date() if not pd.isna(obs) else None,
                    source=fr.source,
                )
            )
        reg_as_of = max(obs_dates).date() if obs_dates else as_of.date()
        summary = _summary(region, score, label, parts, unusable, lang_of(settings))
        base = MacroRegime(
            region=region,
            as_of=reg_as_of,
            score=score if math.isfinite(score) else None,
            label=label,
            indicators=indicators,
            summary=summary,
            exposure_mult=exposure_multiplier(score, settings) if math.isfinite(score) else 1.0,
            stale=stale,
        )
        if not math.isfinite(score):
            log.warning("macro %s: score unavailable (%s)", region, summary)
            base.warnings.append(tr(settings, f"점수 산출 불가: {summary}", f"No score: {summary}"))
        regimes.append(base)
    return regimes


def current_regimes(
    settings: Settings,
    cache: DiskCache,
    as_of: date | datetime | str | pd.Timestamp | None = None,
    *,
    series: dict[str, pd.Series] | None = None,
) -> list[MacroRegime]:
    """Global / US / Korea regimes as of ``as_of`` (default: today, KST), from data <= as_of only.

    ``series`` (optional) replaces :func:`load_macro_series` (tests / replays). Loader failures appear
    as missing indicators; the loader's ``failures`` / ``notes`` are available from the loader itself.
    """
    data = series if series is not None else load_macro_series(settings, cache)
    return regimes_at(data, settings, _as_ts(as_of))


def history_dates(series: dict[str, pd.Series], freq: str = "ME") -> pd.DatetimeIndex:
    """Period-end dates (``freq``) from the first to the last observation across ``series``; an
    unfinished period (end after the last observation) is excluded."""
    firsts = [s.index[0] for s in series.values() if len(s)]
    lasts = [s.index[-1] for s in series.values() if len(s)]
    if not firsts:
        return pd.DatetimeIndex([])
    return pd.date_range(min(firsts), max(lasts), freq=freq)


def regime_history(
    settings: Settings,
    cache: DiskCache | None = None,
    *,
    freq: str = "ME",
    series: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Point-in-time regime scores at each period end (``freq``, default calendar month-end).

    Columns: date, region, score, label, n_used, n_total. The score on ``date`` uses data <= date only;
    the same causal series as :func:`current_regimes`. Rows before any z history exists have NaN score
    and label "데이터 부족". Join to monthly returns with ``date.dt.to_period("M")``.
    """
    if series is None:
        if cache is None:
            raise ValueError("regime_history needs a cache when series is not given")
        series = load_macro_series(settings, cache)
    dates = history_dates(series, freq)
    cols = ["date", "region", "score", "label", "n_used", "n_total"]
    if len(dates) == 0:
        return pd.DataFrame(columns=cols)
    frames = build_indicator_frames(series, settings)
    evals = evaluate_indicators(frames, dates)
    rows = []
    for region in REGIONS:
        sc = region_scores(evals, region)
        df = pd.DataFrame(
            {
                "date": sc.index,
                "region": region,
                "score": sc["score"].to_numpy(),
                "n_used": sc["n_used"].to_numpy(),
                "n_total": sc["n_total"].to_numpy(),
            }
        )
        df["label"] = [regime_label(s, settings) for s in df["score"]]
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)[cols]
    return out.sort_values(["date", "region"], kind="stable").reset_index(drop=True)
