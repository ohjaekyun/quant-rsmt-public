"""US v9 engineered features (docs/MODELING.md 25.1, block E). Every value uses only the row's own point-in-time
inputs or the same entity's PAST values (91 days earlier), so nothing looks ahead. Research code outside src/.

Monotone transforms alone (log, sqrt) do not change a per-date rank-gauss input, so logs are used only inside ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantcio.modeling import features as feat

CHANGE_DAYS = 91


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.where(b.abs() > 1e-12)).replace([np.inf, -np.inf], np.nan)


def _past(f: pd.DataFrame, cols: list[str], days: int = CHANGE_DAYS) -> pd.DataFrame:
    """The same entity's values of ``cols`` at the last row dated <= t - days (within 21 days)."""
    left = f[["entity", "date"]].copy()
    left["_i"] = np.arange(len(f))
    left["key"] = left["date"] - pd.Timedelta(days, "D")
    right = f[["entity", "date", *cols]].rename(columns={"date": "key"})
    m = pd.merge_asof(
        left.sort_values("key"),
        right.sort_values("key"),
        on="key",
        by="entity",
        direction="backward",
        tolerance=pd.Timedelta(21, "D"),
    )
    return m.sort_values("_i")[cols].set_axis(f.index)


def engineer(f: pd.DataFrame) -> tuple[list[str], list[str], dict[str, tuple[str, str]]]:
    """Adds raw e_* columns and their model inputs x_e_* to ``f``. Returns (ranked names, product names, docs)."""
    has = set(f.columns)

    def ok(*cols: str) -> bool:
        return all(c in has for c in cols)

    ranked: dict[str, tuple[pd.Series, str]] = {}
    prod: dict[str, tuple[pd.Series, str]] = {}
    # risk-adjusted ratios
    if ok("mom_12_1", "vol60"):
        ranked["e_mom_sharpe"] = (_safe_div(f["mom_12_1"], f["vol60"]), "12-1 모멘텀 ÷ 60일 변동성")
    if ok("mom_6m", "vol60"):
        ranked["e_mom6_sharpe"] = (_safe_div(f["mom_6m"], f["vol60"]), "6개월 모멘텀 ÷ 60일 변동성")
    if ok("ret_1m", "vol20"):
        ranked["e_ret1m_z"] = (_safe_div(f["ret_1m"], f["vol20"]), "1개월 수익 ÷ 20일 변동성")
    if ok("resid_ret_1m", "ivol60"):
        ranked["e_resid_z"] = (_safe_div(f["resid_ret_1m"], f["ivol60"]), "잔차 수익 ÷ 고유 변동성")
    if ok("vol20", "vol60"):
        ranked["e_vol_ratio"] = (
            np.log(_safe_div(f["vol20"], f["vol60"]).where(lambda s: s > 0)),
            "log(20일 ÷ 60일 변동성)",
        )
    if ok("dvol60", "vol60"):
        ranked["e_down_share"] = (_safe_div(f["dvol60"], f["vol60"]), "하방 변동성 비중")
    if ok("ivol60", "vol60"):
        ranked["e_idio_share"] = (_safe_div(f["ivol60"], f["vol60"]), "고유 변동성 비중")
    if ok("max_ret_1m", "vol20"):
        ranked["e_lottery"] = (_safe_div(f["max_ret_1m"], f["vol20"]), "최대 일간 수익 ÷ 20일 변동성")
    if ok("high52", "vol60"):
        ranked["e_dd_vol"] = (_safe_div(f["high52"] - 1.0, f["vol60"]), "(52주 고가 근접도 − 1) ÷ 변동성")
    # shape / differences
    if ok("mom_3m", "mom_6m"):
        ranked["e_mom_accel"] = (f["mom_3m"] - f["mom_6m"] / 2.0, "3개월 − 6개월/2 (가속)")
    if ok("mom_6m", "mom_12_1"):
        ranked["e_mom_curve"] = (f["mom_6m"] - f["mom_12_1"], "6개월 − 12-1개월")
    if ok("ma50_dist", "ma200_dist"):
        ranked["e_ma_spread"] = (f["ma50_dist"] - f["ma200_dist"], "50일선 − 200일선 이격도")
    # value / quality combinations
    if ok("ep", "bp"):
        ranked["e_ep_bp"] = (f["ep"] - f["bp"], "E/P − B/P")
    if ok("ep", "sp"):
        ranked["e_margin"] = (_safe_div(f["ep"], f["sp"]), "E/P ÷ S/P (순이익률 대용)")
    if ok("op_yield", "ep"):
        ranked["e_op_minus_ep"] = (f["op_yield"] - f["ep"], "영업이익 수익률 − E/P")
    if ok("us_div_yield", "ep"):
        ranked["e_payout"] = (_safe_div(f["us_div_yield"], f["ep"].where(f["ep"] > 0)), "배당수익률 ÷ E/P (배당성향)")
    if ok("roe", "us_leverage"):
        ranked["e_roa"] = (f["roe"] * (1.0 - f["us_leverage"]), "ROE × (1 − 부채비율) (ROA 대용)")
    if ok("insider_buy_180", "insider_sell_180"):
        ranked["e_insider_net"] = (f["insider_buy_180"] - f["insider_sell_180"], "내부자 순매수(매수 − 매도)")
    # 3-month changes (same entity, 91 days earlier)
    base = [c for c in ("ep", "roe", "rev_yoy", "op_margin", "us_sue", "vol60", "turnover", "beta") if c in has]
    if base:
        past = _past(f, base)
        names = {
            "ep": "E/P 3개월 변화",
            "roe": "ROE 3개월 변화",
            "rev_yoy": "매출 성장률 3개월 변화(가속)",
            "op_margin": "영업이익률 3개월 변화",
            "us_sue": "실적 서프라이즈 3개월 변화",
            "turnover": "회전율 3개월 변화",
            "beta": "베타 3개월 변화",
        }
        for c in base:
            if c == "vol60":
                ranked["e_d_vol60"] = (
                    np.log(_safe_div(f["vol60"], past["vol60"]).where(lambda s: s > 0)),
                    "log(60일 변동성 ÷ 3개월 전)",
                )
            else:
                ranked[f"e_d_{c}"] = (f[c] - past[c], names[c])
    # sector-relative differences
    if "sector_group" in has:
        for c, ko in (
            ("ep", "E/P"),
            ("bp", "B/P"),
            ("sp", "S/P"),
            ("roe", "ROE"),
            ("rev_yoy", "매출 성장률"),
            ("mom_12_1", "12-1 모멘텀"),
            ("vol60", "60일 변동성"),
            ("us_gpa", "매출총이익/자산"),
        ):
            if c in has:
                med = f.groupby(["date", "sector_group"])[c].transform("median")
                ranked[f"e_sr_{c}"] = (f[c] - med, f"{ko} − 업종 중앙값")
    # cross-sectional interactions (products of rank-gauss inputs)
    for a, b, name, ko in (
        ("x_ep", "x_roe", "e_i_value_quality", "가치×질 (E/P×ROE)"),
        ("x_us_gpa", "x_bp", "e_i_gpa_bp", "매출총이익/자산×B/P"),
        ("x_rev_yoy", "x_ep", "e_i_growth_value", "매출 성장×E/P"),
        ("x_log_mcap", "x_bp", "e_i_size_value", "규모×B/P"),
        ("x_log_mcap", "x_mom_12_1", "e_i_size_mom", "규모×모멘텀"),
        ("x_amihud", "x_ret_1m", "e_i_illiq_rev", "비유동성×1개월 수익"),
        ("x_us_short_ratio_20", "x_mom_12_1", "e_i_short_mom", "공매도 비율×모멘텀"),
        ("x_us_sue", "x_us_ear", "e_i_sue_ear", "실적 서프라이즈×발표 반응"),
        ("x_us_sue_rev", "x_us_sue", "e_i_sue_rev", "매출 서프라이즈×순이익 서프라이즈"),
    ):
        if ok(a, b):
            prod[name] = (f[a] * f[b], ko)
    # market-state interactions (rank-gauss input × the date's regime value, regime clipped to +-3)
    for a, r, name, ko in (
        ("x_beta", "r_mkt_ret_3m", "e_m_beta_mkt", "베타×시장 3개월 수익"),
        ("x_vol60", "r_vix_z", "e_m_vol_vix", "변동성×VIX z"),
        ("x_mom_12_1", "r_xs_disp", "e_m_mom_disp", "모멘텀×횡단면 분산"),
        ("x_bp", "r_curve_z", "e_m_bp_curve", "B/P×장단기 금리차 z"),
        ("x_log_mcap", "r_credit_z", "e_m_size_credit", "규모×신용 z"),
    ):
        if ok(a, r):
            rv = f[r]
            if r in ("r_mkt_ret_3m", "r_xs_disp"):  # not z-scores: z against the PAST dates only
                d = f.groupby("date")[r].first().sort_index()
                z = (d - d.expanding(24).mean().shift(1)) / d.expanding(24).std().shift(1)
                rv = f["date"].map(z)
            prod[name] = (f[a] * rv.clip(-3, 3), ko)
    # events
    if ok("us_ear", "us_ear_days"):
        ranked["e_ear_recent"] = (f["us_ear"].where(f["us_ear_days"] <= np.log1p(20)), "최근 20일 안 발표 반응")
    if ok("us_rd_mcap", "us_rd_none"):
        ranked["e_rd_active"] = (f["us_rd_mcap"] * (1.0 - f["us_rd_none"].fillna(0)), "R&D/시가총액(보고 기업)")
    if ok("us_emp_growth", "us_rd_growth"):
        prod_src = feat.rank_gauss(f["us_emp_growth"], f["date"]) * feat.rank_gauss(f["us_rd_growth"], f["date"])
        prod["e_i_emp_rd"] = (prod_src, "직원 증가율×R&D 증가율")
    docs = {}
    for name, (s, ko) in ranked.items():
        f[name] = s.astype(float)
        f[feat.x_col(name)] = feat.rank_gauss(f[name], f["date"])
        docs[name] = (ko, name)
    for name, (s, ko) in prod.items():
        f[name] = s.astype(float).clip(-3, 3)
        f[feat.x_col(name)] = f[name]
        docs[name] = (ko, name)
    return list(ranked), list(prod), docs
