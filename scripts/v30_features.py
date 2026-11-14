"""Features from the 69 / 70 data (docs/MODELING.md 73, declared 2026-10-20 with the sector maps). Research only.

Stock level (company search / Wikipedia top list), sector level (theme search, shopping, topics, retail sales, SOC,
Form D - each stock gets its sector's value) and market regime (credit, loans, deposits) - all through the 71 rules
(scripts/mixfreq.py): known from the release date, staleness caps, frequency-matched z-scores.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import mixfreq as mf  # noqa: E402
import v28_popular_data as v28  # noqa: E402
import v29_credit_vc_data as v29  # noqa: E402

KR_THEME_SECTOR = {
    "반도체": ["반도체", "인공지능"],
    "2차전지·전기장비": ["2차전지", "배터리", "전기차"],
    "IT·전자": ["인공지능"],
    "제약·바이오": ["바이오", "신약"],
    "산업재": ["방산", "조선", "원전", "우주", "로봇", "수소"],
    "경기소비재": ["화장품", "면세점", "여행"],
    "커뮤니케이션": ["게임", "엔터", "아이돌", "드라마", "영화"],
    "운송": ["항공"],
    "건설": ["부동산", "아파트"],
    "금융": ["주식", "금리", "비트코인"],
    "유틸리티": ["원전"],
    "화학": ["태양광", "수소"],
}
KR_SHOP_SECTOR = {
    "경기소비재": ["50000000", "50000001", "50000002", "50000004", "50000007", "50000009", "50000010", "50000005"],
    "필수소비재": ["50000006", "50000008"],
    "IT·전자": ["50000003"],
    "커뮤니케이션": ["50005542"],
}
US_TOPIC_SECTOR = {
    "Information Technology": ["Artificial_intelligence", "Semiconductor", "Cloud_computing", "Data_center"],
    "Consumer Discretionary": ["Electric_vehicle", "Black_Friday_(shopping)", "Cyber_Monday"],
    "Materials": ["Lithium-ion_battery", "Gold"],
    "Financials": ["Cryptocurrency", "Bitcoin"],
    "Energy": ["Oil", "Natural_gas"],
    "Real Estate": ["Housing_bubble"],
    "Health Care": ["Pandemic", "Obesity", "Semaglutide"],
    "Utilities": ["Nuclear_power", "Solar_power"],
    "Industrials": ["Defense_industry"],
    "Communication Services": ["Video_game", "Streaming_media", "Box_office"],
}
US_MACRO_TOPICS = [
    "Recession",
    "Inflation",
    "Interest_rate",
    "Tariff",
    "Stock_market_crash",
    "Federal_Reserve",
    "Unemployment",
]
US_RETAIL_SECTOR = {
    "Consumer Discretionary": ["441", "442", "443", "444", "448", "451", "452", "453", "454", "722"],
    "Consumer Staples": ["445", "446"],
    "Energy": ["447"],
}
US_SOC_SECTOR = {
    "Industrials": "all",
    "Materials": "all",
    "Utilities": ["water", "c_water_sewer", "c_power_comm_line", "power", "sewage_waste", "water_supply"],
}
FORMD_SECTOR = [  # (pattern on the Form D industry group, GICS sector); first match wins
    (r"Computers|Telecommunications|Other Technology", "Information Technology"),
    (r"Biotech|Pharma|Health|Hospital", "Health Care"),
    (r"Electric Utilities", "Utilities"),
    (r"Oil and Gas|Coal|Energy|Environmental", "Energy"),
    (r"Commercial$|Residential|REITS|Real Estate|Construction", "Real Estate"),
    (r"Bank|Insurance|Investing|Financial", "Financials"),
    (r"Retail|Restaurant|Travel|Lodging|Tourism", "Consumer Discretionary"),
    (r"Manufacturing|Business Services|Airlines|Transportation", "Industrials"),
    (r"Agriculture", "Consumer Staples"),
]
DOCS: dict[str, tuple[str, str]] = {
    "cx_search_z": ("회사명 검색량 4주 평균의 52주 대비 z(네이버)", "company search volume z"),
    "cx_search_spike": ("회사명 검색량 최근 5일 최대 ÷ 60일 중앙값(log)", "company search spike"),
    "cx_wiki_days": ("최근 30일 위키백과 상위 1,000 진입 일수", "days in Wikipedia top 1,000, 30 days"),
    "cx_wiki_rank": ("최근 30일 위키백과 최고 순위(−log)", "best Wikipedia rank, 30 days"),
    "sx_theme": ("업종 테마 검색어 z(네이버)", "sector theme search z"),
    "sx_shop": ("업종 쇼핑 클릭 z(네이버 쇼핑인사이트)", "sector shopping clicks z"),
    "sx_topic": ("업종 주제 위키백과 조회 z", "sector topic pageviews z"),
    "sx_retail": ("업종 소매판매 전년 대비 z(MARTS)", "sector retail sales yoy z"),
    "sx_soc": ("업종 SOC 지출 전년 대비 z(USAspending·VIP)", "sector public investment yoy z"),
    "sx_formd": ("업종 사모 자금 조달 13주 z(Form D 비펀드)", "sector private funding z (Form D)"),
}


def daily_z(t: pd.DataFrame, col: str = "ratio") -> pd.DataFrame:
    """Calendar-daily series -> 28-day mean vs the preceding 364 days (z), known the next day (73)."""
    s = t.set_index("date")[col].astype(float).sort_index().asfreq("D")
    m28 = s.rolling(28, min_periods=20).mean()
    mu, sd = s.rolling(364, min_periods=180).mean().shift(28), s.rolling(364, min_periods=180).std().shift(28)
    out = pd.DataFrame({"date": s.index, "z": ((m28 - mu) / sd.where(sd > 0)).to_numpy()})
    out["available"] = out["date"] + pd.Timedelta(1, "D")
    spike = np.log(s.rolling(5, min_periods=3).max() / s.rolling(60, min_periods=30).median().shift(5))
    out["spike"] = spike.replace([np.inf, -np.inf], np.nan).to_numpy()
    return out.dropna(subset=["z"])


def _sector_rows(frame: pd.DataFrame, by_sector: dict[str, pd.DataFrame], freq: str) -> np.ndarray:
    """Each row gets its sector's value. US frames carry Korean GICS names (backtest.sectors.GICS_KO), the maps English
    ones - both are accepted (the first 73 US run missed every sector feature on this, 2026-10-21)."""
    from quantcio.backtest.sectors import GICS_KO

    ko_to_en = {v: k for k, v in GICS_KO.items()}
    out = np.full(len(frame), np.nan)
    sec = (
        np.array([ko_to_en.get(x, x) for x in frame["sector"].astype(str)])
        if any(k in GICS_KO for k in by_sector)
        else frame["sector"].astype(str).to_numpy()
    )
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    for k, z in by_sector.items():
        ii = np.where(sec == k)[0]
        if len(ii) and len(z):
            out[ii] = mf.asof(z, "z", dates[ii], freq)
    return out


def _mean_z(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["date", "available", "z"])
    t = pd.concat(parts).groupby("available", as_index=False).agg(z=("z", "mean"), date=("date", "max"))
    return t[["date", "available", "z"]]


# ================================================================================================ KR


def build_kr(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, list[str], dict]:
    r28 = v28.root(settings)
    out = pd.DataFrame(index=frame.index)
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    comp = pd.read_parquet(r28 / "naver_search_company.parquet")
    cz, cs = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    for e, g in comp.groupby("key"):
        ii = np.where(ents == str(e))[0]
        if len(ii):
            z = daily_z(g)
            cz[ii], cs[ii] = mf.asof(z, "z", dates[ii], "D"), mf.asof(z, "spike", dates[ii], "D")
    out["cx_search_z"], out["cx_search_spike"] = cz, cs
    th = pd.read_parquet(r28 / "naver_search_theme.parquet")
    tz = {k: daily_z(g) for k, g in th.groupby("key")}
    out["sx_theme"] = _sector_rows(
        frame, {s: _mean_z([tz[k] for k in ks if k in tz]) for s, ks in KR_THEME_SECTOR.items()}, "D"
    )
    sh = pd.read_parquet(r28 / "naver_shop.parquet")
    sz = {c: daily_z(g) for c, g in sh.groupby("code")}
    out["sx_shop"] = _sector_rows(
        frame, {s: _mean_z([sz[c] for c in cs_ if c in sz]) for s, cs_ in KR_SHOP_SECTOR.items()}, "D"
    )
    wiki = wiki_top_features(frame, settings, "ko")
    for c in wiki:
        out[c] = wiki[c]
    # market regime (trees only)
    reg = pd.DataFrame(index=frame.index)
    r29 = v29.root(settings)
    cr = pd.read_parquet(r29 / "kofia_credit.parquet")
    fu = pd.read_parquet(r29 / "kofia_funds.parquet")
    for name, t, col in (
        ("r_kr_margin", cr, "margin_total"),
        ("r_kr_collateral", cr, "collateral_loans"),
        ("r_kr_deposits", fu, "deposits"),
    ):
        z = mf.zchange(t.rename(columns={col: "value"})[["date", "value", "available"]], "D", 20)
        reg[name] = mf.asof(z, "z", dates, "D")
    forced = fu[["date", "available", "forced_share_pct"]].rename(columns={"forced_share_pct": "value"})
    reg["r_kr_forced"] = mf.asof(mf.level_z(forced, "D", smooth=20), "z", dates, "D")
    ec = pd.read_parquet(r29 / "ecos_loans.parquet")
    for name, item in (("r_kr_mortgage", "11100A0"), ("r_kr_other_loans", "11100B0")):
        t = ec[(ec["table"] == "household_by_purpose") & (ec["item"] == item)]
        t = t.rename(columns={"month": "date"})[["date", "value", "available"]]
        reg[name] = mf.asof(mf.zchange(t, "M", 3), "z", dates, "M")
    info = {
        "companies": int(comp["key"].nunique()),
        "themes": len(tz),
        "shop": len(sz),
        "wiki_days": wiki.attrs.get("days"),
    }
    return pd.concat([out, reg], axis=1), list(reg.columns), info


# ================================================================================================ US


def wiki_top_features(frame: pd.DataFrame, settings, lang: str) -> pd.DataFrame:
    """cx_wiki_days / cx_wiki_rank from the daily top-1,000 lists (US: company article map of section 27; KR: company
    names as Korean article titles). NaN outside the fetched span (71.5 (a))."""
    fs = sorted(glob.glob(str(v28.root(settings) / "wiki_top" / f"{lang}_*.parquet")))
    out = pd.DataFrame(np.nan, index=frame.index, columns=["cx_wiki_days", "cx_wiki_rank"])
    if not fs:
        out.attrs["days"] = 0
        return out
    top = pd.concat(pd.read_parquet(f) for f in fs)
    if lang == "en":
        arts = json.loads((kd.root(settings, "wiki") / "articles.json").read_text())  # CIK -> article
        import v16_data as v16

        from quantcio.data.cache import DiskCache

        ents = np.unique(frame["entity"].astype(str))
        chains = v16.us_chains(settings, DiskCache(settings.cache_path), pd.DataFrame({"entity": ents}))
        title = {e: arts.get(str(int(c))) for e in ents for c in chains.get(e, set()) if arts.get(str(int(c)))}
    else:
        import v16_data as v16

        from quantcio.data.cache import DiskCache

        uni = v16.universe("KR", settings, DiskCache(settings.cache_path))
        title = {str(e): str(n).replace(" ", "_") for e, n in zip(uni["entity"], uni["name"], strict=True) if n}
    top["article"] = top["article"].astype(str)
    span = (top["date"].min(), top["date"].max())
    full = pd.date_range(*span, freq="D")
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    by_title = {t: g for t, g in top[top["article"].isin(set(title.values()))].groupby("article")}
    for e, t in title.items():
        ii = np.where(ents == e)[0]
        if not len(ii):
            continue
        g = by_title.get(t)
        daily_in = pd.Series(0.0, index=full)
        best = pd.Series(np.nan, index=full)
        if g is not None:
            gg = g.groupby("date")["rank"].min()
            daily_in.loc[gg.index] = 1.0
            best.loc[gg.index] = -np.log(gg.to_numpy(dtype=float))
        days = daily_in.rolling(30, min_periods=20).sum().shift(1)  # known the next day
        rank = best.rolling(30, min_periods=1).max().shift(1).fillna(-np.log(1001.0))
        frame_days = pd.Series(dates[ii]).astype("datetime64[ns]")
        inside = (frame_days >= span[0] + pd.Timedelta(30, "D")) & (frame_days <= span[1] + pd.Timedelta(1, "D"))
        out.iloc[ii, 0] = np.where(inside, days.reindex(frame_days).to_numpy(), np.nan)
        out.iloc[ii, 1] = np.where(inside, rank.reindex(frame_days).to_numpy(), np.nan)
    out.attrs["days"] = len(full)
    return out


def build_us(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, list[str], dict]:
    r28, r29 = v28.root(settings), v29.root(settings)
    out = pd.DataFrame(index=frame.index)
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    wt = pd.read_parquet(r28 / "wiki_topics.parquet")
    en = wt[wt["project"] == "en.wikipedia"].rename(columns={"views": "ratio"})
    tz = {a: daily_z(g) for a, g in en.groupby("article")}
    out["sx_topic"] = _sector_rows(
        frame, {s: _mean_z([tz[a] for a in arts if a in tz]) for s, arts in US_TOPIC_SECTOR.items()}, "D"
    )
    mr = pd.read_parquet(r28 / "census_marts.parquet")
    mr["date"] = pd.to_datetime(mr["time"]) + pd.offsets.MonthEnd(0)
    mz = {c: mf.zchange(g[["date", "value"]], "M", 12, yoy=True) for c, g in mr.groupby("category_code")}
    out["sx_retail"] = _sector_rows(
        frame, {s: _mean_z([mz[c] for c in cs if c in mz]) for s, cs in US_RETAIL_SECTOR.items()}, "M"
    )
    soc = pd.read_parquet(r28 / "us_soc_monthly.parquet").rename(columns={"month": "date", "obligations": "value"})
    soc["available"] = soc["date"] + pd.offsets.MonthEnd(1) + pd.Timedelta(15, "D")  # USAspending: next month 15th
    vip = pd.read_parquet(r28 / "census_vip_public.parquet")
    vip = vip.assign(date=pd.to_datetime(vip["time"]) + pd.offsets.MonthEnd(0))
    vip = vip[["kind", "date", "value"]].assign(
        available=lambda x: x["date"] + pd.offsets.MonthEnd(1) + pd.Timedelta(2, "D")
    )
    socz = {k: mf.zchange(g[["date", "value", "available"]], "M", 12, yoy=True) for k, g in soc.groupby("kind")}
    socz |= {k: mf.zchange(g[["date", "value", "available"]], "M", 12, yoy=True) for k, g in vip.groupby("kind")}
    soc_map = {s: _mean_z([z for k, z in socz.items() if ks == "all" or k in ks]) for s, ks in US_SOC_SECTOR.items()}
    out["sx_soc"] = _sector_rows(frame, soc_map, "M")
    fd = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob(str(r29 / "formd" / "*.parquet"))))
    fd = fd[~fd["fund"]].copy()
    fd["sector"] = fd["industry"].astype(str).map(lambda x: next((s for p, s in FORMD_SECTOR if re.search(p, x)), None))
    fdz = {}
    for s, g in fd.dropna(subset=["sector"]).groupby("sector"):
        w = g.groupby(pd.Grouper(key="date", freq="W-FRI"))["sold"].sum().rename("value").reset_index()
        fdz[s] = mf.zchange(w.assign(available=w["date"] + pd.Timedelta(1, "D")), "W", 13)
    out["sx_formd"] = _sector_rows(frame, fdz, "W")
    wiki = wiki_top_features(frame, settings, "en")
    for c in wiki:
        out[c] = wiki[c]
    reg = pd.DataFrame(index=frame.index)
    fin = pd.read_parquet(r29 / "finra_margin.parquet").rename(columns={"month": "date", "debit": "value"})
    reg["r_us_margin"] = mf.asof(mf.zchange(fin, "M", 12, yoy=True), "z", dates, "M")
    for name in ("h8_loans", "h8_ci", "h8_real_estate", "h8_consumer", "h8_cards"):
        t = pd.read_parquet(r29 / f"fred_{name}.parquet")
        reg[f"r_us_{name}"] = mf.asof(mf.zchange(t, "W", 13), "z", dates, "W")
    g19 = pd.read_parquet(r29 / "fred_g19_total.parquet")
    reg["r_us_g19"] = mf.asof(mf.zchange(g19, "M", 3), "z", dates, "M")
    tot = fd.groupby(pd.Grouper(key="date", freq="W-FRI"))["sold"].sum().rename("value").reset_index()
    reg["r_us_formd"] = mf.asof(
        mf.zchange(tot.assign(available=tot["date"] + pd.Timedelta(1, "D")), "W", 13), "z", dates, "W"
    )
    macro = _mean_z([tz[a] for a in US_MACRO_TOPICS if a in tz])
    reg["r_us_macro_topics"] = mf.asof(macro, "z", dates, "D")
    info = {
        "topics": len(tz),
        "retail": len(mz),
        "soc_series": len(socz),
        "formd_sectors": len(fdz),
        "wiki_days": wiki.attrs.get("days"),
    }
    return pd.concat([out, reg], axis=1), list(reg.columns), info


def block(market: str, frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, list[str], dict]:
    return build_kr(frame, settings) if market == "KR" else build_us(frame, settings)
