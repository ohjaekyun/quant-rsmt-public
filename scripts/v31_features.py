"""Features from full-history rates, prices by category and production by industry (docs/MODELING.md 75.2, declared
2026-10-24 with the sector maps). Research only. Built with the 71 rules (scripts/mixfreq.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mixfreq as mf  # noqa: E402
import v30_features as v30  # noqa: E402
import v31_macro_data as md  # noqa: E402

KR_PPI_MAP = {
    "반도체": ["30911AA"],
    "IT·전자": ["309AA"],
    "2차전지·전기장비": ["310AA"],
    "화학": ["305AA"],
    "에너지": ["304AA"],
    "철강·소재": ["307AA"],
    "자동차": ["3121AA"],
    "산업재": ["311AA", "312AA"],
    "제약·바이오": ["3054AA"],
    "필수소비재": ["301AA"],
    "경기소비재": ["305622AA", "302AA"],
    "커뮤니케이션": ["503AA"],
    "운송": ["501AA"],
    "유틸리티": ["401AA"],
    "건설": ["306AA"],
    "금융": ["504AA"],
}
KR_CPI_MAP = {
    "필수소비재": ["A", "B"],
    "경기소비재": ["C", "E", "K"],
    "커뮤니케이션": ["H", "I"],
    "자동차": ["G"],
    "제약·바이오": ["F"],
    "유틸리티": ["D"],
}
KR_IP_MAP = {
    "반도체": ["I11ACQ"],
    "IT·전자": ["I11ACQ"],
    "2차전지·전기장비": ["I11ACS"],
    "화학": ["I11ACK", "I11ACM"],
    "제약·바이오": ["I11ACL"],
    "에너지": ["I11ACJ"],
    "철강·소재": ["I11ACO", "I11ACN"],
    "자동차": ["I11ACU"],
    "산업재": ["I11ACT", "I11ACV"],
    "필수소비재": ["I11ACA", "I11ACB"],
    "경기소비재": ["I11ACE", "I11ACF", "I11ACW"],
    "유틸리티": ["I11ADA"],
}
US_PPI_MAP = {
    "Information Technology": ["PCU334413334413", "PCU511210511210"],
    "Health Care": ["PCU325412325412"],
    "Materials": ["PCU325325", "PCU331110331110", "PCU327320327320"],
    "Energy": ["PCU211111211111"],
    "Consumer Discretionary": ["PCU336110336110"],
    "Industrials": ["PCU336411336411"],
    "Utilities": ["PCU2211--2211--"],
}
US_CPI_MAP = {
    "Consumer Discretionary": ["CPIAPPSL", "CUSR0000SETA01", "CUSR0000SEFV", "CPIRECSL"],
    "Consumer Staples": ["CPIFABSL"],
    "Health Care": ["CPIMEDSL"],
    "Energy": ["CPIENGSL"],
    "Real Estate": ["CUSR0000SAH1"],
    "Industrials": ["CPITRNSL"],
}
US_IP_MAP = {
    "Information Technology": ["IPG334S", "IPG3344S"],
    "Consumer Discretionary": ["IPG3361T3S"],
    "Materials": ["IPG325S", "IPG331S", "IPG327S", "IPG322S"],
    "Health Care": ["IPG3254S"],
    "Energy": ["IPG211S"],
    "Consumer Staples": ["IPG311A2S"],
    "Industrials": ["IPG3364T9S", "IPG333S", "IPG335S"],
    "Utilities": ["IPUTIL"],
}
DOCS: dict[str, tuple[str, str]] = {
    "sx_ppi_yoy": ("업종 산출물 PPI 전년 대비 z", "sector output PPI yoy z"),
    "sx_ppi_rel": ("업종 PPI 전년 대비 − 전체 PPI 전년 대비(z)", "sector PPI yoy minus total PPI yoy z"),
    "sx_cpi_yoy": ("업종 CPI 품목 전년 대비 z", "sector CPI item yoy z"),
    "sx_ip_yoy": ("업종 생산 전년 대비 z", "sector production yoy z"),
    "sx_ship_yoy": ("업종 출하 전년 대비 z", "sector shipments yoy z"),
    "sx_inv_ship": ("업종 재고÷출하 3개월 변화 z", "sector inventory / shipments 3-month change z"),
    "kr_beta_ktb": ("국고채 3년 월 변화 민감도(36개월)", "beta to KTB 3Y monthly change"),
    "kr_impact_ktb": ("국고채 민감도 × 최근 3개월 국고채 변화 z", "KTB beta x recent KTB change z"),
    "kr_beta_credit": ("신용 스프레드 월 변화 민감도(36개월)", "beta to credit-spread monthly change"),
    "kr_impact_credit": ("신용 스프레드 민감도 × 최근 3개월 스프레드 변화 z", "credit beta x recent spread change z"),
}


def _yoy_z(t: pd.DataFrame) -> pd.DataFrame:
    return mf.zchange(t[["date", "value", "available"]], "M", 12, yoy=True)


def _series_map(monthly: pd.DataFrame, key: str) -> dict[str, pd.DataFrame]:
    return {k: g.sort_values("date") for k, g in monthly.groupby(key)}


def _sector(frame, maps: dict[str, list[str]], z: dict[str, pd.DataFrame]) -> np.ndarray:
    return v30._sector_rows(frame, {s: v30._mean_z([z[i] for i in ids if i in z]) for s, ids in maps.items()}, "M")


def _level_z3(t: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Level z against the trailing three years of the same series (71.3; past only, at least one year)."""
    d = t.sort_values("date").drop_duplicates("date", keep="last")
    win, mn = {"D": 756, "M": 36}[freq], {"D": 252, "M": 12}[freq]
    v = d["value"].astype(float)
    mu, sd = v.rolling(win, min_periods=mn).mean().shift(1), v.rolling(win, min_periods=mn).std().shift(1)
    return d.assign(z=(v - mu) / sd.where(sd > 0))[["date", "available", "z"]].dropna(subset=["z"])


def _rel(own: pd.DataFrame, total: pd.DataFrame) -> pd.DataFrame:
    """Own yoy minus total yoy (log points), z-scored as a level against its own past three years."""
    a = own.set_index("date")["value"].astype(float)
    b = total.set_index("date")["value"].astype(float).reindex(a.index)
    diff = (np.log(a) - np.log(a.shift(12))) - (np.log(b) - np.log(b.shift(12)))
    av = own.set_index("date")["available"].reindex(a.index).to_numpy()
    return _level_z3(pd.DataFrame({"date": a.index, "value": diff.to_numpy(), "available": av}).dropna(), "M")


def _inv_ship(inv: pd.DataFrame, ship: pd.DataFrame) -> pd.DataFrame:
    a = inv.set_index("date")["value"].astype(float)
    b = ship.set_index("date")["value"].astype(float).reindex(a.index)
    av = inv.set_index("date")["available"].reindex(a.index).to_numpy()
    t = pd.DataFrame({"date": a.index, "value": (a / b).to_numpy(), "available": av})
    return mf.zchange(t.dropna(), "M", 3)


def _rate_betas(frame: pd.DataFrame, monthly_change: pd.Series, shock_z: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per entity, 36-month OLS beta of month-end ret_1m on the monthly change (months strictly before t), and beta x
    the latest 3-month change z; month-end values carried to the following rows (<= 35 days)."""
    me = frame[frame["is_me"]][["date", "entity", "ret_1m"]].copy()
    me["month"] = me["date"].dt.to_period("M")
    chg = monthly_change.copy()
    chg.index = chg.index.to_period("M")
    beta_out = np.full(len(frame), np.nan)
    imp_out = np.full(len(frame), np.nan)
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    for e, g in me.groupby("entity"):
        g = g.sort_values("date").drop_duplicates("month", keep="last")
        x = chg.reindex(g["month"]).to_numpy(dtype=float)
        y = g["ret_1m"].to_numpy(dtype=float)
        betas = np.full(len(g), np.nan)
        for i in range(len(g)):
            lo = max(0, i - 36)
            xx, yy = x[lo:i], y[lo:i]
            ok = np.isfinite(xx) & np.isfinite(yy)
            if ok.sum() >= 24 and np.var(xx[ok]) > 0:
                betas[i] = np.cov(xx[ok], yy[ok])[0, 1] / np.var(xx[ok], ddof=1)
        tbl = pd.DataFrame({"date": g["date"].to_numpy(), "value": betas, "available": g["date"].to_numpy()})
        ii = np.where(ents == str(e))[0]
        if not len(ii):
            continue
        b = mf.asof(tbl.rename(columns={"value": "z"}), "z", dates[ii], "M")
        beta_out[ii] = b
        imp_out[ii] = b * mf.asof(shock_z, "z", dates[ii], "D")
    return beta_out, imp_out


def build_kr(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, list[str], dict]:
    d = md.root(settings)
    rates = pd.read_parquet(d / "kr_rates.parquet")
    mon = pd.read_parquet(d / "kr_monthly.parquet")
    out = pd.DataFrame(index=frame.index)
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    ppi = _series_map(mon[mon["table"] == "ppi"], "item")
    pz = {k: _yoy_z(g) for k, g in ppi.items()}
    out["sx_ppi_yoy"] = _sector(frame, KR_PPI_MAP, pz)
    rel = {k: _rel(g, ppi["*AA"]) for k, g in ppi.items() if k != "*AA"}
    out["sx_ppi_rel"] = _sector(frame, KR_PPI_MAP, rel)
    cpi = _series_map(mon[mon["table"] == "cpi"], "item")
    out["sx_cpi_yoy"] = _sector(frame, KR_CPI_MAP, {k: _yoy_z(g) for k, g in cpi.items()})
    ip = mon[mon["table"] == "ip"].copy()
    ip["ind"], ip["typ"] = ip["item"].str.split("/").str[0], ip["item"].str.split("/").str[1]
    prod = {k: _yoy_z(g) for k, g in ip[ip["typ"] == "2"].groupby("ind")}
    ship = {k: _yoy_z(g) for k, g in ip[ip["typ"] == "4"].groupby("ind")}
    invs = {}
    for k in set(ip["ind"]):
        a, b = ip[(ip["ind"] == k) & (ip["typ"] == "6")], ip[(ip["ind"] == k) & (ip["typ"] == "4")]
        if len(a) and len(b):
            invs[k] = _inv_ship(a.sort_values("date"), b.sort_values("date"))
    out["sx_ip_yoy"] = _sector(frame, KR_IP_MAP, prod)
    out["sx_ship_yoy"] = _sector(frame, KR_IP_MAP, ship)
    out["sx_inv_ship"] = _sector(frame, KR_IP_MAP, invs)
    # rates: daily series -> regime z and stock-level sensitivities
    piv = rates.pivot_table(index="date", columns="series", values="value").sort_index()
    avail = piv.index + pd.Timedelta(1, "D")

    def ser(v: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({"date": v.index, "value": v.to_numpy(), "available": avail})

    reg = pd.DataFrame(index=frame.index)
    reg["r_kr_ktb3_chg"] = mf.asof(mf.zchange(ser(piv["ktb3"]), "D", 60, log=False), "z", dates, "D")
    reg["r_kr_term"] = mf.asof(_level_z3(ser(piv["ktb10"] - piv["ktb3"]), "D"), "z", dates, "D")
    spread = piv["aa3"] - piv["ktb3"]
    reg["r_kr_credit"] = mf.asof(_level_z3(ser(spread), "D"), "z", dates, "D")
    reg["r_kr_credit_chg"] = mf.asof(mf.zchange(ser(spread), "D", 60, log=False), "z", dates, "D")
    reg["r_kr_cd_chg"] = mf.asof(mf.zchange(ser(piv["cd91"]), "D", 60, log=False), "z", dates, "D")
    dgs = pd.read_parquet(d / "us_dgs10.parquet").set_index("date")["value"]
    gap = (piv["ktb10"] - dgs.reindex(piv.index).ffill(limit=5)).dropna()
    gap_t = pd.DataFrame({"date": gap.index, "value": gap.to_numpy(), "available": gap.index + pd.Timedelta(1, "D")})
    reg["r_kr_us_gap"] = mf.asof(_level_z3(gap_t, "D"), "z", dates, "D")
    me_rate = piv.resample("ME").last()
    ktb_m, cred_m = me_rate["ktb3"].diff(), (me_rate["aa3"] - me_rate["ktb3"]).diff()
    ktb_shock = mf.zchange(ser(piv["ktb3"]), "D", 63, log=False)
    cred_shock = mf.zchange(ser(spread), "D", 63, log=False)
    out["kr_beta_ktb"], out["kr_impact_ktb"] = _rate_betas(frame, ktb_m, ktb_shock)
    out["kr_beta_credit"], out["kr_impact_credit"] = _rate_betas(frame, cred_m, cred_shock)
    info = {"ppi": len(ppi), "cpi": len(cpi), "ip_industries": len(prod), "rates_from": str(piv.index.min().date())}
    return pd.concat([out, reg], axis=1), list(reg.columns), info


def build_us(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, list[str], dict]:
    t = pd.read_parquet(md.root(settings) / "us_monthly.parquet")
    out = pd.DataFrame(index=frame.index)
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    by = _series_map(t, "series")
    yz = {k: _yoy_z(g) for k, g in by.items()}
    out["sx_ppi_yoy"] = _sector(frame, US_PPI_MAP, yz)
    rel = {k: _rel(g, by["PPIACO"]) for k, g in by.items() if k.startswith("PCU") and "PPIACO" in by}
    out["sx_ppi_rel"] = _sector(frame, US_PPI_MAP, rel)
    out["sx_cpi_yoy"] = _sector(frame, US_CPI_MAP, yz)
    out["sx_ip_yoy"] = _sector(frame, US_IP_MAP, yz)
    reg = pd.DataFrame(index=frame.index)
    for name, sid in (("r_us_cpi_3m", "CPIAUCSL"), ("r_us_core_cpi_3m", "CPILFESL")):
        if sid in by:
            reg[name] = mf.asof(mf.zchange(by[sid][["date", "value", "available"]], "M", 3), "z", dates, "M")
    return pd.concat([out, reg], axis=1), list(reg.columns), {"series": len(by)}


def block(market: str, frame: pd.DataFrame, settings, cache=None) -> tuple[pd.DataFrame, list[str], dict]:
    return build_kr(frame, settings) if market == "KR" else build_us(frame, settings)
