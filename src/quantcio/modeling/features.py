"""Look-ahead-free features of the stock-selection models (docs/MODELING.md, "특징").

Every value at date t uses prices with index <= t, fundamentals with ``available_from <= t`` (point in time),
benchmark / macro data <= t (KR: macro series <= t - 1 calendar day, because the US close of day t is only
known the next Korean morning). Rolling windows use ``min_periods`` so early values are NaN, never biased.

Model inputs:
- ``x_<name>``: CROSS-SECTIONAL features, rank-gauss normalised per date over that date's PIT members
  (rank -> (rank - 0.5)/n -> Phi^-1). NaN stays NaN (tree models route it, linear models use 0 = the median).
- ``sx_<name>``: the same, ranked WITHIN the member's sector group on that date (sector-neutral).
- ``r_<name>``: TIME-SERIES regime / stress features, identical for every member on a date (raw values; they
  can only matter through interactions, e.g. in trees or the RSMT-Q v2 partition).

Event dummies are deliberately absent (``modeling.events``): the stress state below is what is knowable at t.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import special

from quantcio.quant.features import wilder_rsi

log = logging.getLogger(__name__)

TRADING_DAYS = 252
PRICE_MAX_AGE_DAYS = 7  # a close older than this at t counts as missing (halt / feed gap)

# (Korean, English) description of every feature; the raw column has the same name without the prefix.
FEATURE_DOCS: dict[str, tuple[str, str]] = {
    # price / volume (own daily bars <= t)
    "ret_1w": ("1주 수익률 C_t/C_{t-5}−1 (단기 반전)", "1-week return (short-term reversal)"),
    "ret_1m": ("1개월 수익률 C_t/C_{t-21}−1 (1개월 반전)", "1-month return (reversal)"),
    "mom_3m": ("3개월 모멘텀 C_t/C_{t-63}−1", "3-month momentum"),
    "mom_6m": ("6개월 모멘텀 C_t/C_{t-126}−1", "6-month momentum"),
    "mom_12_1": ("12-1개월 모멘텀 C_{t-21}/C_{t-252}−1", "12-1 month momentum"),
    "high52": ("52주 고가 근접도 C_t/max(C_{t-251..t})", "proximity to the 52-week high"),
    "ma50_dist": ("50일선 이격도", "distance to the 50-day MA"),
    "ma200_dist": ("200일선 이격도", "distance to the 200-day MA"),
    "rsi14": ("RSI(14, Wilder)", "RSI(14, Wilder)"),
    "vol20": ("20일 실현 변동성(연율)", "20-day realised volatility (annualised)"),
    "vol60": ("60일 실현 변동성(연율)", "60-day realised volatility (annualised)"),
    "dvol60": ("60일 하방 반편차(연율)", "60-day downside semi-deviation (annualised)"),
    "ivol60": ("60일 고유 변동성(벤치마크 회귀 잔차, 연율)", "60-day idiosyncratic volatility vs benchmark"),
    "beta": ("252일 베타(126일 이상)", "252-day beta vs benchmark (>= 126 obs)"),
    "max_ret_1m": ("최근 21일 최대 일간 수익률(MAX 효과)", "max daily return over 21 days (MAX effect)"),
    "volume_trend": ("거래대금 추세 log(ADV20/ADV120)", "traded-value trend log(ADV20/ADV120)"),
    "turnover": ("회전율 log(ADV20/시가총액)", "turnover log(ADV20 / market cap)"),
    "amihud": ("아미후드 비유동성 log(평균 |r|/거래대금, 21일)", "Amihud illiquidity (21 days, log)"),
    "skew60": ("60일 일간 로그수익 왜도(복권 선호)", "60-day skewness of daily log returns (lottery demand)"),
    "resid_ret_1m": ("1개월 잔차 수익 log r − β·벤치마크(잔차 반전)", "1-month residual return (residual reversal)"),
    "seas": ("수익률 계절성: 1~5년 전 같은 20일 창 수익 평균", "return seasonality: same 20-day window 1-5 years ago"),
    "log_mcap": ("log 시가총액(시점별)", "log market cap (point in time)"),
    # fundamentals (available_from <= t, period end within max_staleness_days)
    "ep": ("E/P = TTM 지배순이익/시가총액(적자 포함, 부호 유지)", "E/P = TTM parent net income / market cap"),
    "bp": ("B/P = 지배지분 자본/시가총액", "B/P = parent equity / market cap"),
    "roe": ("ROE = TTM 순이익/자본(자본>0)", "ROE = TTM net income / equity (equity > 0)"),
    "op_margin": ("영업이익률 = TTM 영업이익/매출", "operating margin = TTM operating income / revenue"),
    "rev_yoy": ("매출 성장률 = TTM 매출/전년 TTM 매출−1", "revenue growth, TTM vs TTM a year earlier"),
    "op_yield": ("영업이익 수익률 = TTM 영업이익/시가총액", "operating earnings yield = TTM operating income / mcap"),
    "sp": ("매출 수익률 = TTM 매출/시가총액", "sales yield = TTM revenue / market cap"),
    "op_yoy": ("영업이익 성장 (TTM−전년)/|전년|", "operating-income growth (TTM - TTM a year earlier) / |prev|"),
    "dep": ("이익 수익률 변화 (TTM 순이익−전년)/시총", "earnings-yield change (TTM NI - NI a year earlier) / mcap"),
    # sector
    "sector_mom_3m": ("섹터 3개월 모멘텀(같은 날 섹터 구성원 평균)", "sector 3-month momentum (members' mean)"),
    # regime / stress (same for all members on a date)
    "mkt_ret_1m": ("시장(주 벤치마크) 1개월 수익률", "market 1-month return (primary benchmark)"),
    "mkt_ret_3m": ("시장 3개월 수익률", "market 3-month return"),
    "mkt_dd": ("시장 52주 고점 대비 낙폭", "market drawdown from its 252-day high"),
    "mkt_vol20": ("시장 20일 변동성(연율)", "market 20-day volatility (annualised)"),
    "mkt_vol_ratio": ("시장 변동성 비율(20/120일)", "market volatility ratio (20/120 days)"),
    "xs_disp": ("횡단면 분산도: 구성원 1개월 수익률 표준편차", "cross-sectional dispersion of 1-month returns"),
    "breadth": ("시장 폭: 200일선 위 구성원 비중", "breadth: share of members above their 200-day MA"),
    "vix_z": ("VIX 수준 z(부호 조정, +=위험선호)", "VIX level z (sign-adjusted, + = risk-on)"),
    "credit_z": ("신용 HYG/LQD 3M z", "credit HYG/LQD 3M z"),
    "curve_z": ("미 10Y−3M 금리차 z", "US 10Y-3M curve z"),
    "us10y_z": ("미 10년물 3M 변화 z(급등=−)", "US 10Y 3M change z (spike = -)"),
    "dxy_z": ("달러인덱스 3M z(강달러=−)", "DXY 3M change z (strong dollar = -)"),
    "oil_z": ("유가 3M z(급등=−)", "oil 3M change z (spike = -)"),
    "cu_au_z": ("구리/금 3M z", "copper/gold 3M z"),
    "usdkrw_z": ("USD/KRW 3M z(원화 약세=−, KR)", "USD/KRW 3M z (weak won = -, KR)"),
    "score_global": ("글로벌 매크로 레짐 점수", "global macro regime score"),
    "score_region": ("시장 지역(KR/US) 매크로 레짐 점수", "home-region macro regime score"),
    # features v3 (docs/MODELING.md 16.3): regime
    "vix_term": ("VIX/VIX3M 기간 구조(1 초과 = 단기 공포)", "VIX / VIX3M term structure (> 1 = near-term stress)"),
    # features v3: per-stock macro exposure (252-session partial beta controlling for the benchmark)
    "beta_fx": ("환율 노출: USD/KRW(KR)·달러인덱스(US) 변화 베타", "FX exposure: beta to USD/KRW (KR) / DXY (US)"),
    "beta_oil": ("유가 노출: WTI 변화에 대한 베타", "oil exposure: beta to WTI changes"),
    "beta_rate": ("금리 노출: 미 10년물 금리 변화에 대한 베타", "rate exposure: beta to US 10Y yield changes"),
    "beta_credit": ("신용 노출: HYG/LQD 변화에 대한 베타", "credit exposure: beta to HYG/LQD changes"),
    # features v3: KR OpenDART
    "dart_filings_90": ("공시 빈도: 주요사항·거래소·기타 공시 수(90일, log)", "disclosure count, 90 days (log)"),
    "dart_buyback_180": ("자사주 취득·소각 결정 수(180일)", "buyback / share-cancellation decisions, 180 days"),
    "dart_treasury_sale_180": ("자사주 처분 결정 수(180일)", "treasury-share disposals, 180 days"),
    "dart_dilution_365": ("유상증자·CB·BW·EB 발행 결정 수(365일)", "rights issues / convertible issuance, 365 days"),
    "dart_supply_90": ("단일판매·공급계약 체결 − 해지(90일)", "supply contracts signed minus cancelled, 90 days"),
    "dart_ma_365": ("합병·분할·양수도·지분 취득/처분 결정 수(365일)", "M&A / asset deals, 365 days"),
    "dart_governance_365": ("최대주주·대표이사 변경 수(365일)", "largest-holder / CEO changes, 365 days"),
    "dart_distress_365": ("불성실공시·관리종목·소송 등 수(365일)", "unfaithful disclosure / admin / lawsuit, 365 days"),
    "dart_inquiry_90": ("조회공시 요구·풍문 해명 수(90일)", "exchange inquiries / rumour clarifications, 90 days"),
    "dart_correction_180": ("정정 공시 수(180일)", "corrected filings, 180 days"),
    "insider_net_90": ("임원·주요주주 순매수 주식/상장주식(90일)", "insider net shares / shares outstanding, 90 days"),
    "insider_breadth_90": ("내부자 매수 보고 − 매도 보고 수(90일)", "insider buy minus sell reports, 90 days"),
    "block_net_90": ("5% 대량보유자 지분 변동 합(%p, 90일)", "5% holders' stake change (%p), 90 days"),
    "emp_growth": ("직원 수 증가율(사업보고서, 전년 대비)", "headcount growth (annual report, YoY)"),
    "emp_tenure": ("평균 근속연수", "average tenure (years)"),
    "emp_pay_growth": ("1인 평균 급여 증가율(전년 대비)", "average pay per employee, YoY"),
    "emp_contract_share": ("기간제 근로자 비중", "share of fixed-term employees"),
    # features v3: KR National Pension Service workplace data
    "nps_growth_3m": ("국민연금 가입자 수 3개월 증가율", "NPS members, 3-month growth"),
    "nps_growth_12m": ("국민연금 가입자 수 12개월 증가율", "NPS members, 12-month growth"),
    "nps_hire_3m": ("국민연금 신규 취득률(3개월)", "NPS new members / members, 3 months"),
    "nps_quit_3m": ("국민연금 상실률(3개월)", "NPS lost members / members, 3 months"),
    # features v3: US SEC
    "sec_8k_90": ("8-K 수(90일, log)", "8-K count, 90 days (log)"),
    "sec_8k_material_90": ("8-K 중요 계약·인수·채무 항목 수(90일)", "8-K items 1.01 / 2.01 / 2.03, 90 days"),
    "sec_8k_negative_365": (
        "8-K 부정 항목(파산·손상·상장폐지 통지·감사인 교체·재작성)·NT 수(365일)",
        "negative 8-K items + NT filings, 365 days",
    ),
    "sec_officer_change_180": ("8-K 임원 변동(5.02) 수(180일)", "8-K officer changes (5.02), 180 days"),
    "sec_form4_90": ("Form 4 수(90일, log)", "Form 4 count, 90 days (log)"),
    "sec_form144_90": ("Form 144 매도 예정 신고 수(90일)", "Form 144 planned-sale notices, 90 days"),
    "sec_activist_365": ("SC 13D 수(365일)", "SC 13D filings, 365 days"),
    "sec_issuance_365": ("424B·S-3 수(365일)", "424B / S-3 filings, 365 days"),
    "insider_buy_180": ("내부자 장내 매수 금액/시가총액(180일)", "insider open-market buys / market cap, 180 days"),
    "insider_sell_180": ("내부자 장내 매도 금액/시가총액(180일)", "insider open-market sales / market cap, 180 days"),
}

PRICE_FEATURES = [
    "ret_1w",
    "ret_1m",
    "mom_3m",
    "mom_6m",
    "mom_12_1",
    "high52",
    "ma50_dist",
    "ma200_dist",
    "rsi14",
    "vol20",
    "vol60",
    "dvol60",
    "ivol60",
    "beta",
    "max_ret_1m",
    "volume_trend",
    "amihud",
    "skew60",
    "resid_ret_1m",
    "seas",
]
SIZE_FEATURES = ["turnover", "log_mcap"]
FUND_FEATURES = ["ep", "bp", "roe", "op_margin", "rev_yoy", "op_yield", "sp", "op_yoy", "dep"]
SECTOR_FEATURES = ["sector_mom_3m"]
XS_FEATURES = [*PRICE_FEATURES, *SIZE_FEATURES, *FUND_FEATURES, *SECTOR_FEATURES]
SECTOR_RELATIVE = [
    "ret_1m",
    "mom_3m",
    "mom_6m",
    "ep",
    "bp",
    "volume_trend",
    "vol60",
    "resid_ret_1m",
    "op_yield",
    "sp",
    "op_yoy",
    "dep",
]
# features v2 (pre-declared in docs/MODELING.md 0.2 before the v2 run); every other raw feature is v1
FEATURES_V2 = ["skew60", "resid_ret_1m", "seas", "op_yield", "sp", "op_yoy", "dep"]
SEAS_YEARS = 5  # return seasonality: lags of 1..5 years (252 sessions each), the ones available
PREV_YEAR_ITEMS = ("revenue_ttm", "op_income_ttm", "ni_parent_ttm")  # "<item>_prev" = the TTM a year earlier
BENCH_REGIME = ["mkt_ret_1m", "mkt_ret_3m", "mkt_dd", "mkt_vol20", "mkt_vol_ratio"]
XS_REGIME = ["xs_disp", "breadth"]
MACRO_INDICATORS = {  # macro.regime indicator key -> feature name
    "g_vix": "vix_z",
    "g_credit_3m": "credit_z",
    "us_curve_10y_3m": "curve_z",
    "us_10y_3m": "us10y_z",
    "dxy_3m": "dxy_z",
    "g_oil_3m": "oil_z",
    "g_copper_gold_3m": "cu_au_z",
    "kr_usdkrw_3m": "usdkrw_z",
}
MACRO_FEATURES = [*MACRO_INDICATORS.values(), "score_global", "score_region"]
EXTRA_REGIME = ["vix_term"]  # features v3 (docs/MODELING.md 16.3)
REGIME_FEATURES = [*BENCH_REGIME, *XS_REGIME, *MACRO_FEATURES]  # the v2 (core) regime set; v3 adds EXTRA_REGIME


def x_col(name: str) -> str:
    return f"x_{name}"


def sx_col(name: str) -> str:
    return f"sx_{name}"


def r_col(name: str) -> str:
    return f"r_{name}"


def feature_label(col: str, lang: str = "ko") -> str:
    """Display label of a model column (x_/sx_/r_ prefix understood)."""
    prefix, _, name = col.partition("_")
    doc = FEATURE_DOCS.get(name)
    if doc is None:
        return col
    text = doc[1] if lang == "en" else doc[0]
    if prefix == "sx":
        return text + (" [within sector]" if lang == "en" else " [섹터 내 순위]")
    return text


# ------------------------------------------------------------------------------------------------ price


def align_close(index: pd.DatetimeIndex, bench: pd.DataFrame | None, limit: int = 3) -> pd.Series:
    """Benchmark close on ``index`` (forward-filled at most ``limit`` sessions, never back-filled)."""
    if bench is None or bench.empty:
        return pd.Series(np.nan, index=index)
    c = bench["close"].astype(float)
    return c.reindex(index.union(c.index)).ffill(limit=limit).reindex(index)


def entity_price_features(px: pd.DataFrame, bench: pd.DataFrame | None) -> pd.DataFrame:
    """Every price/volume feature of one entity on its own daily index (value at t uses bars <= t)."""
    close = px["close"].astype(float)
    volume = px["volume"].astype(float).clip(lower=0.0)
    lr = np.log(close.where(close > 0)).diff()
    simple = close.pct_change(fill_method=None)
    # traded value: the frame's own ``value`` (PIT frames, docs/MODELING.md 84.3) - adjusted close x volume would be
    # re-scaled by later splits (KR back-adjustment) or later dividends (US total-return adjustment)
    dv = px["value"].astype(float).clip(lower=0.0) if "value" in px else close * volume
    b = align_close(px.index, bench)
    blr = np.log(b.where(b > 0)).diff()
    ann = math.sqrt(TRADING_DAYS)
    out = pd.DataFrame(index=px.index)
    out["ret_1w"] = close / close.shift(5) - 1.0
    out["ret_1m"] = close / close.shift(21) - 1.0
    out["mom_3m"] = close / close.shift(63) - 1.0
    out["mom_6m"] = close / close.shift(126) - 1.0
    out["mom_12_1"] = close.shift(21) / close.shift(252) - 1.0
    out["high52"] = close / close.rolling(252, min_periods=126).max()
    out["ma50_dist"] = close / close.rolling(50, min_periods=50).mean() - 1.0
    out["ma200_dist"] = close / close.rolling(200, min_periods=200).mean() - 1.0
    out["rsi14"] = wilder_rsi(close, 14)
    out["vol20"] = lr.rolling(20, min_periods=20).std() * ann
    out["vol60"] = lr.rolling(60, min_periods=60).std() * ann
    neg = lr.clip(upper=0.0)
    out["dvol60"] = np.sqrt((neg * neg).rolling(60, min_periods=60).mean() * TRADING_DAYS)
    var_r = lr.rolling(60, min_periods=60).var()
    var_m = blr.rolling(60, min_periods=60).var()
    cov = lr.rolling(60, min_periods=60).cov(blr)
    resid = (var_r - cov * cov / var_m.where(var_m > 0)).clip(lower=0.0)
    out["ivol60"] = np.sqrt(resid * TRADING_DAYS)
    cov_y = lr.rolling(252, min_periods=126).cov(blr)
    var_y = blr.rolling(252, min_periods=126).var()
    out["beta"] = cov_y / var_y.where(var_y > 0)
    out["max_ret_1m"] = simple.rolling(21, min_periods=21).max()
    # features v2 (docs/MODELING.md 0.2)
    out["skew60"] = lr.rolling(60, min_periods=60).skew()
    lr21 = np.log(close / close.shift(21))
    blr21 = np.log(b / b.shift(21))
    out["resid_ret_1m"] = lr21 - out["beta"].fillna(1.0) * blr21
    lr20 = np.log(close / close.shift(20))  # 20-session log return ending at each bar
    lags = [lr20.shift(TRADING_DAYS * k - 20) for k in range(1, SEAS_YEARS + 1)]  # window [t-252k, t-252k+20]
    out["seas"] = pd.concat(lags, axis=1).mean(axis=1, skipna=True)
    adv20 = dv.rolling(20, min_periods=20).mean()
    adv120 = dv.rolling(120, min_periods=120).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["volume_trend"] = np.log(adv20 / adv120)
        illiq = (simple.abs() / dv.where(dv > 0)).rolling(21, min_periods=15).mean()
        out["amihud"] = np.log(illiq.where(illiq > 0))
    out["adv20"] = adv20
    out["close"] = close
    out["traded"] = volume > 0
    return out.replace([np.inf, -np.inf], np.nan)


def sample_positions(index: pd.DatetimeIndex, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(position of the last bar <= each date, fresh mask: that bar is at most PRICE_MAX_AGE_DAYS old)."""
    idx = index.to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(idx, dates, side="right") - 1
    has = pos >= 0
    fresh = has.copy()
    age = np.timedelta64(PRICE_MAX_AGE_DAYS, "D")
    fresh[has] = (dates[has] - idx[pos[has]]) <= age
    return pos, fresh


# ------------------------------------------------------------------------------------------------ fundamentals


def _asof_item(
    rows: pd.DataFrame, panel: pd.DataFrame, item: str, max_staleness_days: int
) -> tuple[pd.Series, pd.Series]:
    """Latest PIT value of ``item`` per (date, entity) row (available_from <= date, ties -> latest period end;
    dropped when the period ended more than ``max_staleness_days`` before the date)."""
    out = pd.Series(np.nan, index=rows.index, dtype=float)
    pe_out = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns]")
    if panel is None or panel.empty or item not in panel:
        return out, pe_out
    p = panel.dropna(subset=[item, "available_from"])
    p = p[p["entity"].isin(set(rows["entity"]))][["entity", "available_from", "period_end", item]]
    if p.empty:
        return out, pe_out
    p = p.sort_values(["available_from", "period_end"], kind="stable").rename(
        columns={"available_from": "_af", "period_end": "_pe"}
    )
    left = rows[["date", "entity"]].reset_index().sort_values("date", kind="stable")
    j = pd.merge_asof(left, p, left_on="date", right_on="_af", by="entity", direction="backward").set_index("index")
    j = j.reindex(rows.index)
    ok = j["_pe"].notna() & ((j["date"] - j["_pe"]).dt.days <= int(max_staleness_days))
    out = j[item].where(ok).astype(float)
    pe_out = j["_pe"].where(ok)
    return out, pe_out


def add_prev_year(panel: pd.DataFrame, items: tuple[str, ...] = PREV_YEAR_ITEMS) -> pd.DataFrame:
    """``<item>_prev``: the same entity's TTM value of the period ending ~1 year earlier (period end 365 +- 25
    days before; its FIRST filed value), kept only when that value was available no later than the row itself
    (``available_from`` of the prev value <= the row's) - so an as-of lookup of a row never sees a later filing."""
    if panel is None or panel.empty:
        return panel
    p = panel.copy()
    for item in items:
        if item not in p:
            continue
        base = (
            p.dropna(subset=[item, "period_end", "available_from"])
            .sort_values(["available_from"], kind="stable")
            .drop_duplicates(["entity", "period_end"], keep="first")[["entity", "period_end", "available_from", item]]
            .rename(columns={"period_end": "_pe_prev", "available_from": "_af_prev", item: f"{item}_prev"})
            .sort_values("_pe_prev")
        )
        left = p[["entity", "period_end", "available_from"]].reset_index()
        left["_key"] = left["period_end"] - pd.Timedelta(365, "D")
        left = left.dropna(subset=["_key"]).sort_values("_key", kind="stable")
        j = pd.merge_asof(
            left,
            base,
            left_on="_key",
            right_on="_pe_prev",
            by="entity",
            direction="nearest",
            tolerance=pd.Timedelta(25, "D"),
        ).set_index("index")
        known = j["_af_prev"] <= j["available_from"]
        p[f"{item}_prev"] = j[f"{item}_prev"].where(known).reindex(p.index)
    return p


def add_revenue_prev(panel: pd.DataFrame) -> pd.DataFrame:
    """v1 name of ``add_prev_year`` (adds every ``PREV_YEAR_ITEMS`` prev column, incl. ``revenue_ttm_prev``)."""
    return add_prev_year(panel)


def _filing_changes(panel: pd.DataFrame) -> pd.DataFrame:
    """Per FILING row: year-over-year quantities whose current and prev-year values come from that same row
    (``_rev_yoy``, ``_op_yoy``, ``_dni``), so an as-of lookup can never pair values of two different filings."""
    p = panel.copy()
    cols = set(p.columns)
    if {"revenue_ttm", "revenue_ttm_prev"} <= cols:
        prev = p["revenue_ttm_prev"]
        p["_rev_yoy"] = (p["revenue_ttm"] / prev - 1.0).where(prev > 0)
    if {"op_income_ttm", "op_income_ttm_prev"} <= cols:
        prev = p["op_income_ttm_prev"]
        p["_op_yoy"] = ((p["op_income_ttm"] - prev) / prev.abs()).where(prev.abs() > 0)
    if {"ni_parent_ttm", "ni_parent_ttm_prev"} <= cols:
        p["_dni"] = p["ni_parent_ttm"] - p["ni_parent_ttm_prev"]
    return p.replace([np.inf, -np.inf], np.nan)


def fundamental_features(
    rows: pd.DataFrame, panel: pd.DataFrame, market_cap: pd.Series, max_staleness_days: int
) -> pd.DataFrame:
    """Fundamental features per (date, entity) row (see FEATURE_DOCS): E/P, B/P (signed), ROE, operating margin,
    revenue YoY and (v2) operating-earnings yield, sales yield, operating-income growth, earnings-yield change.

    Levels are as-of lookups of each item (``available_from <= date``, the latest filing that has it);
    year-over-year features are computed per filing row first (``_filing_changes``: current and prev-year TTM of
    the SAME filing, the prev value itself known by then - ``add_prev_year``) and then looked up as of the date."""
    if panel is not None and not panel.empty:
        panel = _filing_changes(panel)
    items = {}
    for item in ("ni_parent_ttm", "equity_parent", "revenue_ttm", "op_income_ttm", "_rev_yoy", "_op_yoy", "_dni"):
        items[item], _ = _asof_item(rows, panel, item, max_staleness_days)
    mcap = market_cap.astype(float).where(market_cap > 0)
    ni, eq, rev, op = (items[k] for k in ("ni_parent_ttm", "equity_parent", "revenue_ttm", "op_income_ttm"))
    out = pd.DataFrame(index=rows.index)
    out["ep"] = ni / mcap
    out["bp"] = eq / mcap
    out["roe"] = (ni / eq).where(eq > 0)
    out["op_margin"] = (op / rev).where(rev > 0)
    out["rev_yoy"] = items["_rev_yoy"]
    out["op_yield"] = op / mcap
    out["sp"] = (rev / mcap).where(rev > 0)
    out["op_yoy"] = items["_op_yoy"]
    out["dep"] = items["_dni"] / mcap
    return out.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------------------------------------------ normalisation


def rank_gauss(values: pd.Series, groups: pd.Series | list[pd.Series]) -> pd.Series:
    """Per-group rank-gauss: average rank among non-NaN values -> (rank - 0.5)/n -> Phi^-1 (NaN stays NaN)."""
    r = values.groupby(groups).rank(method="average")
    n = values.notna().groupby(groups).transform("sum")
    u = (r - 0.5) / n
    out = pd.Series(special.ndtri(u.to_numpy(dtype=float)), index=values.index)
    return out.where(values.notna())


def uniform_rank(values: pd.Series, groups: pd.Series | list[pd.Series]) -> pd.Series:
    """Per-group average rank mapped to (0, 1): (rank - 0.5)/n (NaN stays NaN)."""
    r = values.groupby(groups).rank(method="average")
    n = values.notna().groupby(groups).transform("sum")
    return ((r - 0.5) / n).where(values.notna())


# ------------------------------------------------------------------------------------------------ regime


def benchmark_regime(bench: pd.DataFrame | None, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Market-level regime features of the primary benchmark at ``dates`` (bars <= t)."""
    out = pd.DataFrame(index=dates, columns=BENCH_REGIME, dtype=float)
    if bench is None or bench.empty:
        return out
    c = bench["close"].astype(float)
    lr = np.log(c).diff()
    f = pd.DataFrame(index=c.index)
    f["mkt_ret_1m"] = c / c.shift(21) - 1.0
    f["mkt_ret_3m"] = c / c.shift(63) - 1.0
    f["mkt_dd"] = c / c.rolling(252, min_periods=126).max() - 1.0
    v20 = lr.rolling(20, min_periods=20).std()
    f["mkt_vol20"] = v20 * math.sqrt(TRADING_DAYS)
    f["mkt_vol_ratio"] = v20 / lr.rolling(120, min_periods=120).std()
    pos, fresh = sample_positions(pd.DatetimeIndex(c.index), dates.to_numpy(dtype="datetime64[ns]"))
    vals = f.to_numpy(dtype=float)[np.clip(pos, 0, None)]
    vals[~fresh] = np.nan
    return pd.DataFrame(vals, index=dates, columns=BENCH_REGIME)


def macro_features(
    series: dict[str, pd.Series] | None, settings: Any, dates: pd.DatetimeIndex, region: str, lag_days: int
) -> tuple[pd.DataFrame, list[str]]:
    """Sign-adjusted macro indicator z's and regime scores (``quantcio.macro``) as of ``date - lag_days``.

    Returns (frame indexed by ``dates`` with MACRO_FEATURES columns, notes). Indicators whose status is not
    'ok' (stale, missing, no z history) are NaN. ``series`` None -> all NaN plus a note (never silent).
    """
    out = pd.DataFrame(np.nan, index=dates, columns=MACRO_FEATURES, dtype=float)
    notes: list[str] = []
    if not series:
        notes.append("macro series unavailable - macro regime features are NaN")
        return out, notes
    from quantcio.macro.regime import build_indicator_frames, evaluate_indicators, region_scores

    asof = pd.DatetimeIndex(dates) - pd.Timedelta(int(lag_days), "D")
    frames = build_indicator_frames(series, settings)
    evals = evaluate_indicators(frames, asof)
    for key, name in MACRO_INDICATORS.items():
        e = evals.get(key)
        if e is None:
            continue
        z = e["z"].to_numpy(dtype=float)
        ok = e["status"].to_numpy() == "ok"
        out[name] = np.where(ok, z, np.nan)
    g = region_scores(evals, "global")["score"].to_numpy(dtype=float)
    reg = region_scores(evals, region)["score"].to_numpy(dtype=float)
    out["score_global"] = g
    out["score_region"] = reg
    missing = [n for n in MACRO_FEATURES if out[n].isna().all()]
    if missing:
        notes.append(f"macro features entirely missing: {', '.join(missing)}")
    return out, notes


def vix_term(series: dict[str, pd.Series] | None, dates: pd.DatetimeIndex, lag_days: int) -> pd.Series:
    """VIX / VIX3M as of ``date - lag_days`` (each level the last close on or before that day, at most 7 days old)."""
    out = pd.Series(np.nan, index=dates, dtype=float)
    if not series or series.get("vix") is None or series.get("vix3m") is None:
        return out
    asof = (pd.DatetimeIndex(dates) - pd.Timedelta(int(lag_days), "D")).to_numpy(dtype="datetime64[ns]")

    def at(s: pd.Series) -> np.ndarray:
        s = s.sort_index()
        pos, fresh = sample_positions(pd.DatetimeIndex(s.index), asof)
        v = s.to_numpy(dtype=float)[np.clip(pos, 0, None)]
        v[~fresh] = np.nan
        return v

    v, v3 = at(series["vix"]), at(series["vix3m"])
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:] = np.where(v3 > 0, v / v3, np.nan)
    return out


__all__ = [
    "BENCH_REGIME",
    "EXTRA_REGIME",
    "FEATURES_V2",
    "FEATURE_DOCS",
    "FUND_FEATURES",
    "MACRO_FEATURES",
    "PRICE_FEATURES",
    "REGIME_FEATURES",
    "SECTOR_RELATIVE",
    "XS_FEATURES",
    "XS_REGIME",
    "add_prev_year",
    "add_revenue_prev",
    "benchmark_regime",
    "entity_price_features",
    "feature_label",
    "fundamental_features",
    "macro_features",
    "r_col",
    "rank_gauss",
    "sample_positions",
    "sx_col",
    "uniform_rank",
    "vix_term",
    "x_col",
]
