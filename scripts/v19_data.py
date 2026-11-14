"""Blocks A (audit), K (bankruptcy / distress), M (M&A), P (disclosures), G (famous investors), Y (firm age),
Q (departures) and N (ETF) - docs/MODELING.md 53, pre-declared 2026-10-13. Research only.

    .venv/bin/python scripts/v19_data.py fetch --what all     # or kr_d001, krx_etf, yahoo_etf, sec13f, wikidata

Sources: cached OpenDART disclosure lists (B/E/I, 2013+) and periodic-report audit items (scripts/v18_data.py
dart_extra), OpenDART D001/D004 ownership lists (fetched here: filer names), OpenDART company records (est_dt), KRX ETF
all-issue daily screen [MDCSTAT04301] sampled on the last session of every week (2015+, delisted ETFs included),
Yahoo ETF prices, SEC submissions (cached), SEC 13F data sets (re-downloaded; only the listed managers kept) and
Wikidata (inception P571 by CIK P5531). Every event counts from the day after its filing date (``v16._windowed``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_data as v13  # noqa: E402
import v16_data as v16  # noqa: E402

say = v13.say
DAY = np.timedelta64(1, "D")
UA = {"User-Agent": "Mozilla/5.0 (compatible; research bot)"}
WIKIDATA = "https://query.wikidata.org/sparql"
KRX_ETF_FROM = pd.Timestamp("2015-01-01")
SUBS_START = pd.Timestamp("2009-01-01")
REGIME = {
    "r_kr_etf_levinv_4w",
    "r_kr_etf_dom_4w",
    "r_kr_etf_ovs_4w",
    "r_kr_ewy_dvol",
    "r_us_etf_efa_rel_1m",
    "r_us_etf_eem_rel_1m",
    "r_us_etf_spy_dvol",
}


def root(settings) -> Path:
    return kd.root(settings, "v19")


# ================================================================================================ KR disclosure lists

_SUBSID = re.compile(r"\((자회사|종속회사)의주요경영사항\)")
KR_EVENTS = {  # feature -> (pattern on the whitespace-free report name, window days)
    "au_late_24m": (r"감사보고서제출지연|반기검토\(감사\)의견부적정|감사의견(부적정|거절|한정)|감사범위제한", 730),
    "k_rehab_24m": (r"회생절차|파산신청|채권은행등의관리절차|경영정상화계획", 730),
    "k_default_24m": (r"부도발생|은행거래정지|대출원리금연체|사채원리금미지급", 730),
    "k_impair_24m": (r"자본잠식|매출액50억원미만", 730),
    "k_delist_24m": (r"관리종목|상장적격성실질심사|상장폐지사유|개선기간|상장폐지관련|투자주의환기", 730),
    "k_capred_24m": (r"감자결정", 730),
    "ma_merger_12m": (r"합병결정|주식교환|포괄적교환", 365),
    "ma_split_12m": (r"회사분할결정|분할\(분할합병\)결정", 365),
    "ma_acq_12m": (r"타법인주식및출자증권(취득|양수)결정|영업양수결정|중요한(영업|자산)양수도결정", 365),
    "ma_disp_12m": (r"타법인주식및출자증권(처분|양도)결정|영업양도결정|유형자산(처분|양도)결정", 365),
    "ma_control_12m": (r"최대주주변경(?!을수반하는주식담보)|경영권변경", 365),
    "dc_fair_90d": (r"공정공시", 90),
    "dc_guidance_12m": (r"영업실적등에대한전망|장래사업ㆍ?경영계획", 365),
    "dc_ir_90d": (r"기업설명회", 90),
    "dc_cb_12m": (r"전환사채권발행결정|신주인수권부사채권발행결정|교환사채권발행결정", 365),
    "dc_third_12m": (r"제3자배정", 365),
    "dc_bonus_12m": (r"무상증자결정", 365),
    "dc_pledge_12m": (r"주식담보제공계약", 365),
    "dc_option_12m": (r"주식매수선택권부여", 365),
    "dc_rumor_12m": (r"풍문", 365),
    "q_ceo_24m": (r"대표이사(\(대표집행임원\))?변경", 730),
    "q_audit_resign_24m": (r"감사ㆍ?감사위원회위원중도퇴임", 730),
    "q_director_24m": (r"독립이사의선임|사외이사의선임", 730),
}
KR_RATIOS = ("dc_voluntary_share_12m", "dc_correction_share_12m")


def kr_list_table(settings, corp: str) -> pd.DataFrame | None:
    """(date, nm, no, corrected) of the company-level filings in the cached B/E/I lists; None = not cached."""
    d = Path(settings.cache_path) / "alt" / "kr_dart" / "list" / corp
    files = list(d.glob("*.json")) if d.exists() else []
    if not files:
        return None
    rows = []
    for f in files:
        for r in json.loads(f.read_text()).get("rows", []):
            raw = re.sub(r"\s+", "", str(r.get("report_nm") or ""))
            if _SUBSID.search(raw):
                continue
            nm = re.sub(r"^\[[^\]]*\]", "", raw)
            rows.append((pd.Timestamp(r["rcept_dt"]), nm, r.get("rcept_no"), raw[:1] == "["))
    t = pd.DataFrame(rows, columns=["date", "nm", "no", "corrected"])
    return t.drop_duplicates("no").reset_index(drop=True)


# ================================================================================================ KR audit opinions

_BIG4 = re.compile(r"삼일|삼정|안진|한영")
_BAD = re.compile(r"한정|부적정|거절|의견없음")
_NONE = {"", "-", "해당사항없음", "없음", "해당없음", "N/A"}


def _clean(s) -> str:
    return re.sub(r"\s+", "", str(s or ""))


def kr_audit_table(settings, code: str) -> pd.DataFrame | None:
    """(avail, au_bad, au_gc, au_emph, au_kam_n, au_change, au_big4) per business year from v18 dart_extra."""
    p = kd.root(settings, "v18") / "dart_extra" / f"{code}.json"
    if not p.exists():
        return None
    years = json.loads(p.read_text()).get("accnutAdtorNmNdAdtOpinion") or {}
    recs = []
    for _y, rows in sorted(years.items()):
        if not rows:
            continue
        cur = next((r for r in rows if "당기" in str(r.get("bsns_year"))), rows[0])
        no = str(cur.get("rcept_no") or "")
        if not re.match(r"^\d{8}", no):
            continue
        op, emph, kam = _clean(cur.get("adt_opinion")), _clean(cur.get("emphs_matter")), str(cur.get("core_adt_matter"))
        special = _clean(cur.get("adt_reprt_spcmnt_matter"))
        kam_c = _clean(kam)
        n_kam = 0.0 if kam_c in _NONE else float(max(1, len(re.findall(r"(?:^|\n)\s*(?:\d+[\.\)]|[-•ㆍ·])", kam))))
        recs.append(
            {
                "avail": pd.Timestamp(no[:8]),
                "auditor": re.sub(r"회계법인|\(주\)|㈜|주식회사", "", _clean(cur.get("adtor"))),
                "au_bad": np.nan if op in _NONE else float(bool(_BAD.search(op))),
                "au_gc": float("계속기업" in emph + special),
                "au_emph": float(emph not in _NONE),
                "au_kam_n": n_kam,
                "au_big4": float(bool(_BIG4.search(_clean(cur.get("adtor"))))),
            }
        )
    if not recs:
        return None
    t = pd.DataFrame(recs).sort_values("avail").drop_duplicates("avail", keep="last").reset_index(drop=True)
    prev = t["auditor"].shift(1)
    t["au_change"] = np.where(prev.isna() | (t["auditor"] == ""), np.nan, (t["auditor"] != prev).astype(float))
    return t


# ================================================================================================ KR D001 / D004

KR_GURUS = {  # matched case-insensitively on the filer name without spaces, dots and commas (DART strips spaces)
    "gu_kr_nps_24m": r"국민연금",
    "gu_kr_value_24m": (
        r"브이아이피|VIP|한국투자밸류|신영자산|트러스톤|에셋플러스|라자드|"
        r"머스트자산|밸류파트너스|플랫폼파트너스|타임폴리오|한국밸류"
    ),
    "gu_kr_activist_24m": (
        r"케이씨지아이|KCGI|얼라인|엘리엇|Elliott|화이트박스|차파트너스|"
        r"시티오브런던|CityofLondon|오비스|Orbis|SC펀더멘털|돌핀트리|Dolphin|팰리서|Palliser"
    ),
    "gu_kr_foreign_24m": (
        r"블랙록|BlackRock|뱅가드|Vanguard|캐피탈리서치|CapitalResearch|CapitalGroup|템플턴|Templeton|"
        r"피델리티|Fidelity|노르웨이|Norges|싱가포르|Singapore|JPMorgan|제이피모간|"
        r"모건스탠리|MorganStanley|골드만|Goldman|매튜스|Matthews|웰링턴|Wellington"
    ),
}


def fetch_kr_d001(settings, cache) -> dict:
    """OpenDART list.json D001 (5% holdings) and D004 (tender offers) of every universe company, 2013+ (filer names)."""
    k = kd.key("DART_API_KEY")
    if not k:
        return {"skipped": "DART_API_KEY 없음"}
    cc = v16.kr_corp_codes(settings)
    codes = sorted(v16.universe("KR", settings, cache)["entity"].astype(str))
    sess, calls = requests.Session(), 0
    today = pd.Timestamp.today().strftime("%Y%m%d")
    for ty in ("D001", "D004"):
        d = root(settings) / ty.lower()
        d.mkdir(exist_ok=True)
        for i, code in enumerate(codes, 1):
            corp = cc.get(code)
            p = d / f"{corp}.json"
            if not corp or p.exists():
                continue
            rows, page = [], 1
            while True:
                params = {"corp_code": corp, "bgn_de": "20130101", "end_de": today, "pblntf_detail_ty": ty}
                js = v16._dart_get(sess, k, "list", {**params, "page_no": page, "page_count": 100})
                calls += 1
                for x in js.get("list") or []:
                    rows.append({c: x.get(c) for c in ("rcept_no", "rcept_dt", "report_nm", "flr_nm")})
                if js.get("status") != "000" or page >= int(js.get("total_page") or 1):
                    break
                page += 1
                time.sleep(0.1)
            p.write_text(json.dumps({"fetched": today, "rows": rows}, ensure_ascii=False))
            time.sleep(0.1)
            if i % 100 == 0:
                say(f"DART {ty} {i}/{len(codes)} ({calls} calls)")
    return {"codes": len(codes), "calls": calls}


def kr_d_rows(settings, ty: str, corp: str) -> pd.DataFrame | None:
    p = root(settings) / ty.lower() / f"{corp}.json"
    if not p.exists():
        return None
    t = pd.DataFrame(json.loads(p.read_text()).get("rows", []), columns=["rcept_no", "rcept_dt", "report_nm", "flr_nm"])
    t = t[~t["report_nm"].astype(str).str.strip().str.startswith("[")]
    t["date"] = pd.to_datetime(t["rcept_dt"], format="%Y%m%d", errors="coerce")
    return t.dropna(subset=["date"]).drop_duplicates("rcept_no")


# ================================================================================================ KRX ETF (weekly)


def fetch_krx_etf(settings) -> dict:
    """All-ETF screen on the last session of every week since 2015 (resumable; one file per date)."""
    if not v13.krx_ready():
        return {"skipped": "KRX 로그인 없음"}
    v13._krx_stock()
    from pykrx.website.krx.krxio import KrxWebIo

    class _Etf(KrxWebIo):
        @property
        def bld(self):
            return "dbms/MDC/STAT/standard/MDCSTAT04301"

    d = root(settings) / "krx_etf"
    d.mkdir(exist_ok=True)
    have = {p.stem for p in d.glob("*.parquet")} | {p.stem for p in d.glob("*.empty")}
    fridays = pd.date_range(KRX_ETF_FROM, pd.Timestamp.today() - pd.Timedelta(1, "D"), freq="W-FRI")
    n_new, empty_run = 0, 0
    for fr in fridays:
        if any((fr - pd.Timedelta(k, "D")).strftime("%Y%m%d") in have for k in range(5)):
            continue
        got = False
        for k in range(5):  # Friday, else the last earlier session of that week
            day = (fr - pd.Timedelta(k, "D")).strftime("%Y%m%d")
            out = []
            for attempt in range(4):
                try:
                    out = _Etf().read(trdDd=day).get("output") or []
                    break
                except ValueError:
                    time.sleep(3.0 * (attempt + 1))
            time.sleep(v13.KRX_PAUSE)
            if out:
                cols = [
                    "ISU_SRT_CD",
                    "ISU_ABBRV",
                    "NAV",
                    "INVSTASST_NETASST_TOTAMT",
                    "LIST_SHRS",
                    "ACC_TRDVAL",
                    "IDX_IND_NM",
                ]
                df = pd.DataFrame(out)[cols]
                for c in ("NAV", "INVSTASST_NETASST_TOTAMT", "LIST_SHRS", "ACC_TRDVAL"):
                    df[c] = df[c].map(v13._num_krx)
                df.insert(0, "date", pd.Timestamp(day))
                df.to_parquet(d / f"{day}.parquet", index=False)
                got, n_new, empty_run = True, n_new + 1, 0
                break
        if not got:
            empty_run += 1
            (d / f"{fr.strftime('%Y%m%d')}.empty").write_text("")
            if empty_run >= 2 and not v13._krx_alive(v13._krx_stock()):
                for p in d.glob("*.empty"):
                    if p.stat().st_mtime > time.time() - 3600:
                        p.unlink()
                raise RuntimeError(f"KRX 세션 만료({n_new}주 새로 저장) — KRX_JSESSIONID를 바꾼 뒤 다시 실행")
        if n_new and n_new % 50 == 0:
            say(f"KRX ETF {n_new} weeks")
    return {"weeks_new": n_new, "weeks": len(list(d.glob("*.parquet")))}


def fetch_basic_etf(settings) -> dict:
    """KRX ETF basic info [MDCSTAT04601] (classification of the ETFs listed today) - validates the name rules."""
    v13._krx_stock()
    from pykrx.website.krx.krxio import KrxWebIo

    class _Info(KrxWebIo):
        @property
        def bld(self):
            return "dbms/MDC/STAT/standard/MDCSTAT04601"

    out = _Info().read(share="1", csvxls_isNo="false").get("output") or []
    df = pd.DataFrame(out)
    df.to_parquet(root(settings) / "krx_etf_info.parquet", index=False)
    return {"etfs": len(df)}


_NON_EQUITY = re.compile(  # case-sensitive on purpose: short Latin tokens (CD, ICE) must not hit ordinary words
    r"채권|국채|국고채|회사채|특수채|은행채|금융채|공사채|통안|금리|CD|KOFR|SOFR|MMF|머니마켓|단기자금|크레딧|"
    r"원유|WTI|브렌트|금선물|금현물|골드|Gold|GOLD|은선물|구리|원자재|농산물|달러|엔선물|엔화|유로선물|환율|"
    r"리츠|부동산|TDF|TRF|자산배분|혼합|멀티에셋"
)
_OVERSEAS = re.compile(
    r"미국|S&P ?500|나스닥|NASDAQ|Nasdaq|NYSE|중국|차이나|CSI|항셍|HSCEI|홍콩|일본|니케이225|닛케이|TOPIX|"
    r"인도|니프티|Nifty|NIFTY|베트남|VN30|유로|유럽|EURO|Euro|STOXX|Stoxx|독일|DAX|글로벌(?!.*코스닥)|"
    r"선진|신흥|이머징|Emerging|라틴|브라질|러시아|대만|필라델피아|FANG|Solactive|Indxx|FactSet|"
    r"MSCI (World|ACWI|EM|EAFE|China|Japan|India|AC)|Bloomberg|ICE |Russell|아시아|Asia|북미|"
    r"싱가포르|인도네시아|호주|캐나다|영국|프랑스|스위스|엔비디아|테슬라|팔란티어|TSMC|구글|Google|애플|"
    r"마이크로소프트|아마존|브로드컴|빅테크"
)
_LEV = re.compile(r"레버리지|2X|울트라", re.I)
_INV = re.compile(r"인버스|숏|SHORT|베어", re.I)
KR_ETF_SECTOR = [  # (sector of backtest.sectors.KSIC_TO_SECTOR, index-name pattern); first match wins
    ("반도체", r"반도체"),
    ("2차전지·전기장비", r"2차전지|이차전지|배터리|전기차|전기장비"),
    ("IT·전자", r"정보기술|IT|전기전자|전자|소프트웨어"),
    ("커뮤니케이션", r"커뮤니케이션|미디어|통신|게임|인터넷|엔터"),
    ("제약·바이오", r"헬스케어|바이오|제약|의료"),
    ("자동차", r"자동차"),
    ("필수소비재", r"필수소비재|생활소비재|음식료"),
    ("경기소비재", r"경기소비재|소비재|유통|화장품|여행|레저"),
    ("건설", r"건설"),
    ("금융", r"은행|증권|보험|금융"),
    ("산업재", r"산업재|기계|조선|방산|중공업"),
    ("화학", r"화학"),
    ("에너지", r"에너지|정유"),
    ("운송", r"운송|해운|항공"),
    ("유틸리티", r"유틸리티|전력"),
    ("철강·소재", r"철강|소재|비철"),
]


def etf_class(idx: str, name: str, krx: tuple[str, str] | None = None) -> tuple[str, str]:
    """(kind, sector): kind in non_equity | overseas | lev | inv | sector | market; sector '' unless kind == sector.
    ``krx`` = KRX's (market, asset class) of an ETF listed today - it decides domestic equity vs not when known;
    the name rules decide for delisted ETFs."""
    idx, name = str(idx or ""), str(name or "")
    if krx is not None:
        if krx[1] != "주식":
            return "non_equity", ""
        if krx[0] != "국내":
            return "overseas", ""
    elif _NON_EQUITY.search(idx) or _NON_EQUITY.search(name):
        return "non_equity", ""
    elif _OVERSEAS.search(idx) or _OVERSEAS.search(name):
        return "overseas", ""
    if _LEV.search(name):
        return "lev", ""
    if _INV.search(name):
        return "inv", ""
    for sec, pat in KR_ETF_SECTOR:
        if re.search(pat, idx):
            return "sector", sec
    return "market", ""


def kr_etf_weekly(settings) -> pd.DataFrame | None:
    """Weekly ETF panel with flows: date, code, kind, sector, aum, flow (KRW, NaN in an ETF's first week)."""
    d = root(settings) / "krx_etf"
    parts = [pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))] if d.exists() else []
    if not parts:
        return None
    t = pd.concat(parts, ignore_index=True).rename(
        columns={"ISU_SRT_CD": "code", "INVSTASST_NETASST_TOTAMT": "aum", "LIST_SHRS": "shares", "NAV": "nav"}
    )
    cls = t.drop_duplicates("code", keep="last").set_index("code")
    ip = root(settings) / "krx_etf_info.parquet"
    krx = {}
    if ip.exists():
        info = pd.read_parquet(ip)
        krx = dict(
            zip(info["ISU_SRT_CD"], zip(info["IDX_MKT_CLSS_NM"], info["IDX_ASST_CLSS_NM"], strict=True), strict=True)
        )
    kinds = {c: etf_class(r.IDX_IND_NM, r.ISU_ABBRV, krx.get(c)) for c, r in cls.iterrows()}
    t["kind"] = t["code"].map(lambda c: kinds[c][0])
    t["sector"] = t["code"].map(lambda c: kinds[c][1])
    t = t.sort_values(["code", "date"])
    prev = t.groupby("code")["shares"].shift(1)
    gap = t.groupby("code")["date"].diff().dt.days
    t["flow"] = np.where(gap <= 10, (t["shares"] - prev) * t["nav"], np.nan)
    t["aum_prev"] = np.where(gap <= 10, t.groupby("code")["aum"].shift(1), np.nan)
    return t


def etf_class_check(settings) -> dict:
    """Agreement of the name rules alone (the fallback for delisted ETFs) with the KRX classification of today's ETFs
    (domestic equity vs not)."""
    p = root(settings) / "krx_etf_info.parquet"
    if not p.exists():
        return {}
    info = pd.read_parquet(p)
    dom_eq = (info["IDX_MKT_CLSS_NM"] == "국내") & (info["IDX_ASST_CLSS_NM"] == "주식")
    rule = info.apply(lambda r: etf_class(r["ETF_OBJ_IDX_NM"], r["ISU_ABBRV"])[0], axis=1)
    rule_dom = rule.isin(["lev", "inv", "sector", "market"])
    return {"etfs": len(info), "agree_share": round(float((dom_eq == rule_dom).mean()), 3)}


# ================================================================================================ Yahoo ETFs

US_SECTOR_ETF = {
    "Materials": "XLB",
    "Communication Services": "XLC",
    "Energy": "XLE",
    "Financials": "XLF",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Consumer Staples": "XLP",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
}
YAHOO_ETFS = [*US_SECTOR_ETF.values(), "SPY", "EFA", "EEM", "EWY"]


def fetch_yahoo_etf(settings) -> dict:
    import yfinance as yf

    d = yf.download(YAHOO_ETFS, period="max", auto_adjust=False, group_by="ticker", progress=False, threads=True)
    rows = []
    for s in YAHOO_ETFS:
        x = d[s].rename(columns=str.lower)[["close", "volume"]].dropna(subset=["close"])
        x.index = pd.DatetimeIndex(x.index).tz_localize(None)
        rows.append(x.assign(sym=s).reset_index(names="date"))
    t = pd.concat(rows, ignore_index=True)
    t.to_parquet(root(settings) / "yahoo_etf.parquet", index=False)
    return {"symbols": t["sym"].nunique(), "rows": len(t)}


def yahoo_etf(settings) -> dict[str, pd.DataFrame]:
    p = root(settings) / "yahoo_etf.parquet"
    if not p.exists():
        return {}
    t = pd.read_parquet(p)
    return {s: g.set_index("date").sort_index()[["close", "volume"]] for s, g in t.groupby("sym")}


def _dvol_shock(px: pd.DataFrame, short: int, long: int) -> pd.Series:
    dv = (px["close"] * px["volume"]).replace(0, np.nan)
    return np.log(dv.rolling(short, min_periods=short // 2).mean() / dv.rolling(long, min_periods=long // 2).mean())


# ================================================================================================ SEC 13F managers

US_GURUS = {
    "berkshire": r"^BERKSHIRE HATHAWAY INC",
    "baupost": r"BAUPOST GROUP",
    "pershing": r"PERSHING SQUARE CAPITAL",
    "greenlight": r"GREENLIGHT CAPITAL",
    "appaloosa": r"APPALOOSA",
    "third_point": r"THIRD POINT LLC",
    "icahn": r"ICAHN (CARL|ENTERPRISES|CAPITAL)",
    "scion": r"SCION ASSET",
    "oaktree": r"OAKTREE CAPITAL MANAGEMENT",
    "tiger_global": r"TIGER GLOBAL MANAGEMENT",
    "viking": r"VIKING GLOBAL INVESTORS",
    "lone_pine": r"LONE PINE CAPITAL",
    "coatue": r"COATUE MANAGEMENT",
    "duquesne": r"DUQUESNE FAMILY OFFICE",
    "himalaya": r"HIMALAYA CAPITAL",
    "pabrai": r"DALAL STREET",
    "akre": r"AKRE CAPITAL",
    "markel": r"^MARKEL (CORP|GROUP)",
    "elliott": r"ELLIOTT (INVESTMENT )?MANAGEMENT",
    "trian": r"TRIAN FUND MANAGEMENT",
    "valueact": r"VALUEACT",
    "starboard": r"STARBOARD VALUE",
    "soros": r"SOROS FUND MANAGEMENT",
    "ruane": r"RUANE,? CUNNIFF",
    "gardner_russo": r"GARDNER RUSSO",
    "davis": r"DAVIS SELECTED ADVISERS",
    "tweedy": r"TWEEDY,? BROWNE",
    "first_eagle": r"FIRST EAGLE INVESTMENT",
    "southeastern": r"SOUTHEASTERN ASSET MANAGEMENT",
    "dodge_cox": r"DODGE & COX",
    "fundsmith": r"FUNDSMITH",
    "glenview": r"GLENVIEW CAPITAL",
    "jana": r"JANA PARTNERS",
    "fairholme": r"FAIRHOLME CAPITAL",
    "maverick": r"MAVERICK CAPITAL",
    "eminence": r"EMINENCE CAPITAL",
    "sachem": r"SACHEM HEAD",
    "corvex": r"CORVEX",
    "wedgewood": r"WEDGEWOOD PARTNERS",
    "yacktman": r"YACKTMAN",
    "chou": r"CHOU ASSOCIATES",
    "polen": r"POLEN CAPITAL",
    "giverny": r"GIVERNY CAPITAL",
    "abrams": r"ABRAMS BISON",
}
US_PASSIVE = {
    "blackrock": r"^BLACKROCK,? INC",
    "vanguard": r"^VANGUARD GROUP",
    "state_street": r"^STATE STREET CORP",
    "geode": r"^GEODE CAPITAL MANAGEMENT",
}


def _mgr_key(name: str) -> str | None:
    n = str(name or "").upper().strip()
    for k, pat in US_GURUS.items():
        if re.search(pat, n):
            return f"guru:{k}"
    for k, pat in US_PASSIVE.items():
        if re.search(pat, n):
            return f"passive:{k}"
    return None


def _f13_managers(zpath: Path) -> pd.DataFrame:
    """Original 13F-HR filings (filed by the deadline) of the listed managers: holdings rows."""
    z = zipfile.ZipFile(zpath)
    names = {Path(n).name.upper(): n for n in z.namelist()}
    sub = pd.read_csv(
        z.open(names["SUBMISSION.TSV"]),
        sep="\t",
        dtype=str,
        usecols=["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"],
    )
    cover = pd.read_csv(
        z.open(names["COVERPAGE.TSV"]), sep="\t", dtype=str, usecols=["ACCESSION_NUMBER", "FILINGMANAGER_NAME"]
    )
    sub = sub[sub["SUBMISSIONTYPE"].str.strip() == "13F-HR"].merge(cover, on="ACCESSION_NUMBER", how="inner")
    sub["mgr"] = sub["FILINGMANAGER_NAME"].map(_mgr_key)
    sub = sub[sub["mgr"].notna()].copy()
    sub["filed"] = pd.to_datetime(sub["FILING_DATE"], format="mixed", errors="coerce")
    sub["period"] = pd.to_datetime(sub["PERIODOFREPORT"], format="mixed", errors="coerce")
    sub = sub.dropna(subset=["filed", "period"])
    sub = sub[sub["filed"] <= sub["period"] + pd.Timedelta(v13.F13_DEADLINE, "D")]
    cols = ["period", "filed", "cik", "mgr", "name", "cusip9", "shares", "value"]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    info = pd.read_csv(
        z.open(names["INFOTABLE.TSV"]),
        sep="\t",
        dtype=str,
        usecols=["ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"],
    )
    info = info[info["ACCESSION_NUMBER"].isin(set(sub["ACCESSION_NUMBER"]))]
    info = info[(info["SSHPRNAMTTYPE"].str.strip() == "SH") & info["PUTCALL"].isna()]
    info = info.merge(
        sub[["ACCESSION_NUMBER", "period", "filed", "CIK", "mgr", "FILINGMANAGER_NAME"]], on="ACCESSION_NUMBER"
    )
    return pd.DataFrame(
        {
            "period": info["period"],
            "filed": info["filed"],
            "cik": info["CIK"].astype(int),
            "mgr": info["mgr"],
            "name": info["FILINGMANAGER_NAME"],
            "cusip9": info["CUSIP"].str.strip().str.upper(),
            "shares": pd.to_numeric(info["SSHPRNAMT"], errors="coerce"),
            "value": pd.to_numeric(info["VALUE"], errors="coerce"),
        }
    )


def fetch_sec13f(settings) -> dict:
    """Re-download the SEC 13F data sets and keep the listed managers' holdings (zips deleted after reading)."""
    d = root(settings) / "sec13f"
    d.mkdir(exist_ok=True)
    hdr = v13._sec_headers()
    page = requests.get(v13.F13_PAGE, headers=hdr, timeout=120).text
    links = sorted(set(re.findall(r'href="([^"]+form13f[^"]*\.zip)"', page)))
    done = 0
    for link in links:
        name = link.split("/")[-1]
        out = d / name.replace(".zip", "_mgr.parquet")
        if out.exists():
            continue
        zp = d / name
        if not zp.exists():
            v13._download("https://www.sec.gov" + link if link.startswith("/") else link, zp, hdr)
            time.sleep(0.5)
        _f13_managers(zp).to_parquet(out, index=False)
        zp.unlink(missing_ok=True)
        done += 1
        say(f"13F managers {name} ({done} new)")
    return {"files": len(links), "new": done}


def f13_managers(settings) -> pd.DataFrame:
    d = root(settings) / "sec13f"
    parts = [pd.read_parquet(p) for p in sorted(d.glob("*_mgr.parquet"))] if d.exists() else []
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame()
    t = pd.concat(parts, ignore_index=True)
    # one filing per manager CIK and period (the latest filed: a period can span two data-set files); a manager
    # key matched by two CIKs in one period (e.g. a re-registered parent) keeps the CIK with the larger book
    last = t.groupby(["mgr", "cik", "period"])["filed"].transform("max")
    t = t[t["filed"] == last]
    unit = np.where(t["period"] >= v13.F13_DOLLAR_PERIODS_FROM, 1.0, 1000.0)
    t = t.assign(value_usd=t["value"] * unit, cusip6=t["cusip9"].str[:6])
    book = t.groupby(["mgr", "period", "cik"])["value_usd"].sum().reset_index()
    top = book.sort_values("value_usd").drop_duplicates(["mgr", "period"], keep="last")[["mgr", "period", "cik"]]
    return t.merge(top, on=["mgr", "period", "cik"], how="inner")


# ================================================================================================ Wikidata (inception)


def fetch_wikidata(settings) -> dict:
    q = "SELECT ?cik ?inc WHERE { ?item wdt:P5531 ?cik . ?item wdt:P571 ?inc . }"
    r = requests.get(WIKIDATA, params={"query": q, "format": "json"}, headers=UA, timeout=300)
    r.raise_for_status()
    rows = [
        (b["cik"]["value"], b["inc"]["value"]) for b in r.json()["results"]["bindings"] if "cik" in b and "inc" in b
    ]
    t = pd.DataFrame(rows, columns=["cik", "inception"])
    t["cik"] = pd.to_numeric(t["cik"].str.lstrip("0"), errors="coerce")
    t["year"] = pd.to_numeric(t["inception"].str.extract(r"^([+-]?\d{1,4})-")[0], errors="coerce")
    t = t.dropna(subset=["cik", "year"])
    t = t[(t["year"] > 1600) & (t["year"] <= pd.Timestamp.today().year)]
    t = t.groupby("cik", as_index=False)["year"].min()
    t.to_parquet(root(settings) / "wikidata_inception.parquet", index=False)
    return {"ciks": len(t)}


# ================================================================================================ US submissions events


def _us_form_kind(form: str) -> str | None:
    f = str(form or "").strip().upper()
    if f in ("NT 10-K", "NT 10-Q"):
        return "late"
    if f in ("10-K/A", "10-Q/A"):
        return "amend"
    if f in ("SC 14D9", "DEFM14A", "PREM14A"):
        return "target"
    if f == "SC TO-T":
        return "tender_subject"
    if f in ("S-4", "425"):
        return "s4"
    if f == "SC TO-I":
        return "selftender"
    if f in ("SC 13G", "SC 13G/A", "SCHEDULE 13G", "SCHEDULE 13G/A"):
        return "13g"
    if f == "S-8":
        return "s8"
    if f == "8-K":
        return "8k"
    return None


US_ITEM_FEATURES = {  # feature -> (8-K item, window days)
    "au_auditor_chg_24m": ("4.01", 730),
    "au_nonreliance_24m": ("4.02", 730),
    "k_bankrupt_24m": ("1.03", 730),
    "k_accel_24m": ("2.04", 730),
    "k_restruct_24m": ("2.05", 730),
    "k_impair_24m": ("2.06", 730),
    "k_delist_24m": ("3.01", 730),
    "ma_done_12m": ("2.01", 365),
    "ma_agree_6m": ("1.01", 182),
    "dc_regfd_90d": ("7.01", 90),
    "dc_other_90d": ("8.01", 90),
    "q_502_12m": ("5.02", 365),
}
US_FORM_FEATURES = {  # feature -> (form kind, window days)
    "au_late_24m": ("late", 730),
    "au_amend_12m": ("amend", 365),
    "ma_target_12m": ("target", 365),
    "ma_tender_target_12m": ("tender_subject", 365),
    "ma_s4_12m": ("s4", 365),
    "ma_selftender_12m": ("selftender", 365),
    "dc_13g_12m": ("13g", 365),
    "dc_s8_12m": ("s8", 365),
}


def us_cik_events(subs, cik: int) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """(published, kind, item) events of one CIK since SUBS_START and the CIK's first EDGAR filing date."""
    from quantcio.data.alt import us_sec

    blocks = us_sec.submission_blocks(subs, cik, SUBS_START)
    sub = subs.get(cik) or {}
    first = [f.get("filingFrom") for f in (sub.get("filings") or {}).get("files", []) or [] if f.get("filingFrom")]
    frames = [us_sec._block_frame(b) for b in blocks if b]
    if not frames:
        return pd.DataFrame(columns=["published", "kind", "item"]), None
    df = pd.concat(frames, ignore_index=True)
    df = df[~(df["accessionNumber"].notna() & df["accessionNumber"].duplicated())]
    fd = pd.to_datetime(df["filingDate"], errors="coerce")
    first_filed = min([pd.Timestamp(x) for x in first] + ([fd.min()] if fd.notna().any() else []), default=None)
    df = df.assign(kind=df["form"].map(_us_form_kind))
    df = df[df["kind"].notna()]
    subject = df["kind"].isin(["13g", "tender_subject"])
    filer_role = subject & df["fileNumber"].map(lambda v: isinstance(v, str) and not v.strip())
    df = df[~filer_role].copy()
    df["published"] = us_sec.published_dates(df["acceptanceDateTime"], df["filingDate"]).to_numpy()
    df = df[df["published"].notna()]
    k8 = df[df["kind"] == "8k"].assign(
        item=lambda x: x["items"].map(lambda s: [i.strip() for i in str(s or "").split(",")])
    )
    k8 = k8.explode("item").drop_duplicates(["accessionNumber", "item"])
    other = df[df["kind"] != "8k"].assign(item="")
    out = pd.concat([k8[["published", "kind", "item"]], other[["published", "kind", "item"]]], ignore_index=True)
    return out, first_filed


# ================================================================================================ blocks

DOCS = {
    "au_bad": ("감사의견 비적정(한정·부적정·의견거절)", "non-clean audit opinion"),
    "au_gc": ("감사보고서 강조사항에 계속기업 불확실성", "going-concern emphasis"),
    "au_emph": ("감사보고서 강조사항 있음", "emphasis of matter present"),
    "au_kam_n": ("핵심감사사항 개수", "number of key audit matters"),
    "au_change": ("감사인 교체(직전 사업연도와 다름)", "auditor changed vs previous year"),
    "au_big4": ("4대 회계법인 감사", "Big-4 auditor"),
    "au_late_24m": ("감사 지연·부적정 관련 공시(24개월)", "late / adverse audit filings, 24m"),
    "au_auditor_chg_24m": ("8-K 4.01 감사인 교체(24개월)", "8-K 4.01 auditor change, 24m"),
    "au_nonreliance_24m": ("8-K 4.02 재무제표 신뢰 불가(24개월)", "8-K 4.02 non-reliance, 24m"),
    "au_amend_12m": ("10-K/A·10-Q/A 정정(12개월)", "10-K/A, 10-Q/A amendments, 12m"),
    "k_rehab_24m": ("회생·파산·채권단 관리(24개월)", "rehabilitation / bankruptcy / creditor workout, 24m"),
    "k_default_24m": ("부도·거래정지·원리금 연체(24개월)", "default / bank suspension / missed payments, 24m"),
    "k_impair_24m": ("자본잠식·매출 미달 / 8-K 2.06 손상(24개월)", "capital impairment / 8-K 2.06 impairment, 24m"),
    "k_delist_24m": ("관리종목·실질심사·상장폐지 사유 / 8-K 3.01(24개월)", "delisting risk / 8-K 3.01, 24m"),
    "k_capred_24m": ("감자결정(24개월)", "capital reduction, 24m"),
    "k_bankrupt_24m": ("8-K 1.03 파산(24개월)", "8-K 1.03 bankruptcy, 24m"),
    "k_accel_24m": ("8-K 2.04 채무 가속 사유(24개월)", "8-K 2.04 triggering events, 24m"),
    "k_restruct_24m": ("8-K 2.05 구조조정 비용(24개월)", "8-K 2.05 exit costs, 24m"),
    "ma_merger_12m": ("합병·주식교환 결정(12개월)", "merger / share exchange decisions, 12m"),
    "ma_split_12m": ("분할 결정(12개월)", "spin-off decisions, 12m"),
    "ma_acq_12m": ("타법인 주식·영업·자산 양수(12개월)", "acquisitions of stakes / businesses, 12m"),
    "ma_disp_12m": ("타법인 주식·영업·자산 양도(12개월)", "disposals, 12m"),
    "ma_control_12m": ("최대주주·경영권 변경(12개월)", "change of control, 12m"),
    "ma_tender_12m": ("공개매수 신고(12개월)", "tender offers, 12m"),
    "ma_done_12m": ("8-K 2.01 인수·처분 완료(12개월)", "8-K 2.01 completed acquisition / disposition, 12m"),
    "ma_agree_6m": ("8-K 1.01 중요 계약(6개월)", "8-K 1.01 material agreement, 6m"),
    "ma_target_12m": ("피인수 측 서류(SC 14D9·합병 위임장, 12개월)", "target-side filings, 12m"),
    "ma_tender_target_12m": ("공개매수 대상(SC TO-T, 12개월)", "tender-offer target, 12m"),
    "ma_s4_12m": ("합병 등록·소통(S-4·425, 12개월)", "merger registration / communications, 12m"),
    "ma_selftender_12m": ("자기 공개매수(SC TO-I, 12개월)", "issuer tender offer, 12m"),
    "dc_fair_90d": ("공정공시(90일)", "fair-disclosure filings, 90d"),
    "dc_guidance_12m": ("실적 전망·사업계획 공시(12개월)", "guidance filings, 12m"),
    "dc_ir_90d": ("IR 개최(90일)", "IR events, 90d"),
    "dc_cb_12m": ("CB·BW·EB 발행 결정(12개월)", "convertible / warrant / exchangeable issues, 12m"),
    "dc_third_12m": ("제3자배정 유상증자(12개월)", "third-party allotments, 12m"),
    "dc_bonus_12m": ("무상증자 결정(12개월)", "bonus issues, 12m"),
    "dc_pledge_12m": ("주식담보 제공 계약(12개월)", "share pledges, 12m"),
    "dc_option_12m": ("주식매수선택권 부여(12개월)", "stock-option grants, 12m"),
    "dc_rumor_12m": ("풍문·보도 해명(12개월)", "rumour clarifications, 12m"),
    "dc_voluntary_share_12m": ("자율공시 비중(12개월)", "voluntary-disclosure share, 12m"),
    "dc_correction_share_12m": ("정정 공시 비중(12개월)", "correction share, 12m"),
    "dc_regfd_90d": ("8-K 7.01 공정공시(90일)", "8-K 7.01 Reg FD, 90d"),
    "dc_other_90d": ("8-K 8.01 기타 사건(90일)", "8-K 8.01 other events, 90d"),
    "dc_13g_12m": ("SC 13G 5% 단순 보유 신고(12개월)", "SC 13G passive 5% filings, 12m"),
    "dc_s8_12m": ("S-8 임직원 주식보상 등록(12개월)", "S-8 registrations, 12m"),
    "gu_kr_nps_24m": ("국민연금 5% 보고(24개월)", "NPS 5% reports, 24m"),
    "gu_kr_value_24m": ("가치 운용사 5% 보고(24개월)", "value managers' 5% reports, 24m"),
    "gu_kr_activist_24m": ("행동주의 5% 보고(24개월)", "activists' 5% reports, 24m"),
    "gu_kr_foreign_24m": ("해외 대형 운용사 5% 보고(24개월)", "global managers' 5% reports, 24m"),
    "gu_kr_new_12m": ("거장 신규 5% 진입(12개월)", "new 5% entries by listed investors, 12m"),
    "gu_n": ("13F 거장 보유 수", "number of listed 13F investors holding"),
    "gu_new": ("거장 신규 편입 수(분기)", "new positions by listed investors"),
    "gu_add": ("거장 비중 확대 수(분기)", "listed investors adding"),
    "gu_trim": ("거장 비중 축소 수(분기)", "listed investors trimming"),
    "gu_exit": ("거장 전량 매도 수(분기)", "listed investors exiting"),
    "gu_wt_max": ("거장 포트폴리오 내 최대 비중", "max weight in a listed investor's portfolio"),
    "age_found": ("설립 이후 햇수(log)", "log years since founding"),
    "age_edgar": ("EDGAR 첫 제출 이후 햇수(log)", "log years since the first EDGAR filing"),
    "q_ceo_24m": ("대표이사 변경(24개월)", "CEO changes, 24m"),
    "q_audit_resign_24m": ("감사·감사위원 중도퇴임(24개월)", "audit-committee resignations, 24m"),
    "q_director_24m": ("사외이사 선임·해임·중도퇴임(24개월)", "outside-director changes, 24m"),
    "q_502_12m": ("8-K 5.02 임원 변동(12개월)", "8-K 5.02 officer / director changes, 12m"),
    "q_502_cluster": ("90일 안 5.02 두 번 이상", "two or more 5.02 filings within 90 days"),
    "etf_sec_flow_4w": ("섹터 ETF 4주 자금 흐름 ÷ 순자산", "sector ETF 4-week flows / AUM"),
    "etf_sec_flow_13w": ("섹터 ETF 13주 자금 흐름 ÷ 순자산", "sector ETF 13-week flows / AUM"),
    "etf_sec_aum_share": ("섹터 ETF 순자산 ÷ 섹터 시가총액", "sector ETF AUM / sector market cap"),
    "etf_sec_dvol_20_250": ("섹터 ETF 거래대금 충격(20일÷250일)", "sector ETF dollar-volume shock 20d/250d"),
    "etf_sec_dvol_5_60": ("섹터 ETF 거래대금 충격(5일÷60일)", "sector ETF dollar-volume shock 5d/60d"),
    "etf_own": ("국내 주식 ETF 상위 30개 보유액 ÷ 시가총액(분기말)", "holdings of the 30 largest domestic ETFs / mcap"),
    "etf_own_chg_4q": ("ETF 보유 비율 4분기 변화", "4-quarter change of ETF ownership"),
    "etf_passive_own": ("13F 패시브 4사 보유 ÷ 시가총액", "13F passive giants' holdings / market cap"),
    "etf_passive_chg_4q": ("패시브 보유 비율 4분기 변화", "4-quarter change of passive ownership"),
    "r_kr_etf_levinv_4w": ("레버리지 − 인버스 ETF 4주 흐름", "leveraged minus inverse ETF flows, 4w"),
    "r_kr_etf_dom_4w": ("국내 주식 ETF 4주 흐름", "domestic equity ETF flows, 4w"),
    "r_kr_etf_ovs_4w": ("해외 주식 ETF 4주 흐름(국내 대비)", "overseas equity ETF flows vs domestic AUM, 4w"),
    "r_kr_ewy_dvol": ("EWY 거래대금 충격", "EWY dollar-volume shock"),
    "r_us_etf_efa_rel_1m": ("EFA − SPY 1개월 수익", "EFA minus SPY 1-month return"),
    "r_us_etf_eem_rel_1m": ("EEM − SPY 1개월 수익", "EEM minus SPY 1-month return"),
    "r_us_etf_spy_dvol": ("SPY 거래대금 충격", "SPY dollar-volume shock"),
}
BLOCK_COLS = {
    "KR": {
        "A": ["au_bad", "au_gc", "au_emph", "au_kam_n", "au_change", "au_big4", "au_late_24m"],
        "K": ["k_rehab_24m", "k_default_24m", "k_impair_24m", "k_delist_24m", "k_capred_24m"],
        "M": ["ma_merger_12m", "ma_split_12m", "ma_acq_12m", "ma_disp_12m", "ma_control_12m", "ma_tender_12m"],
        "P": [c for c in KR_EVENTS if c.startswith("dc_")] + list(KR_RATIOS),
        "G": [*KR_GURUS, "gu_kr_new_12m"],
        "Y": ["age_found"],
        "Q": ["q_ceo_24m", "q_audit_resign_24m", "q_director_24m"],
        "N": [
            "etf_sec_flow_4w",
            "etf_sec_flow_13w",
            "etf_sec_aum_share",
            "etf_own",
            "etf_own_chg_4q",
            "r_kr_etf_levinv_4w",
            "r_kr_etf_dom_4w",
            "r_kr_etf_ovs_4w",
            "r_kr_ewy_dvol",
        ],
    },
    "US": {
        "A": ["au_auditor_chg_24m", "au_nonreliance_24m", "au_late_24m", "au_amend_12m"],
        "K": ["k_bankrupt_24m", "k_accel_24m", "k_restruct_24m", "k_impair_24m", "k_delist_24m"],
        "M": ["ma_done_12m", "ma_agree_6m", "ma_target_12m", "ma_tender_target_12m", "ma_s4_12m", "ma_selftender_12m"],
        "P": ["dc_regfd_90d", "dc_other_90d", "dc_13g_12m", "dc_s8_12m"],
        "G": ["gu_n", "gu_new", "gu_add", "gu_trim", "gu_exit", "gu_wt_max"],
        "Y": ["age_found", "age_edgar"],
        "Q": ["q_502_12m", "q_502_cluster"],
        "N": [
            "etf_sec_dvol_20_250",
            "etf_sec_dvol_5_60",
            "etf_passive_own",
            "etf_passive_chg_4q",
            "r_us_etf_efa_rel_1m",
            "r_us_etf_eem_rel_1m",
            "r_us_etf_spy_dvol",
        ],
    },
}


def _cnt(ev_dates, dates, days) -> np.ndarray:
    c, _ = v16._windowed(np.asarray(ev_dates, dtype="datetime64[ns]"), np.ones(len(ev_dates)), dates, days)
    return np.log1p(c)


def _put(out: pd.DataFrame, ii: np.ndarray, col: str, vals) -> None:
    out.iloc[ii, out.columns.get_loc(col)] = vals


def _years(dates: np.ndarray, since: pd.Timestamp) -> np.ndarray:
    y = (dates - np.datetime64(since, "ns")) / np.timedelta64(1, "D") / 365.25
    return np.log(np.clip(y, 0.1, None))


def build_kr(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, dict]:
    cols = [c for b in BLOCK_COLS["KR"].values() for c in b]
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    cc = v16.kr_corp_codes(settings)
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    info = {"events": dict.fromkeys(KR_EVENTS, 0), "lists": 0, "audit": 0, "d001": 0, "est_dt": 0}
    audit_cols = BLOCK_COLS["KR"]["A"][:6]
    for e in np.unique(ents):
        corp = cc.get(e)
        ii = np.where(ents == e)[0]
        d = dates[ii]
        if corp:
            t = kr_list_table(settings, corp)
            if t is not None:
                info["lists"] += 1
                orig = t[~t["corrected"]]
                for k, (pat, days) in KR_EVENTS.items():
                    sub = orig[orig["nm"].str.contains(pat, regex=True)]
                    info["events"][k] += len(sub)
                    _put(out, ii, k, _cnt(sub["date"].to_numpy(), d, days))
                td = t["date"].to_numpy()
                tot, _ = v16._windowed(td, np.ones(len(t)), d, 365)
                _, vs = v16._windowed(td, t["nm"].str.contains("자율공시").to_numpy(float), d, 365)
                _, cs = v16._windowed(td, t["corrected"].to_numpy(float), d, 365)
                _put(out, ii, "dc_voluntary_share_12m", np.where(tot > 0, vs / np.maximum(tot, 1), np.nan))
                _put(out, ii, "dc_correction_share_12m", np.where(tot > 0, cs / np.maximum(tot, 1), np.nan))
            d1 = kr_d_rows(settings, "D001", corp)
            if d1 is not None:
                info["d001"] += 1
                any_guru = np.zeros(len(d1), dtype=bool)
                for k, pat in KR_GURUS.items():
                    nm = d1["flr_nm"].astype(str).str.replace(r"[\s\.,]", "", regex=True)
                    m = nm.str.contains(pat, regex=True, case=False).to_numpy()
                    any_guru |= m
                    _put(out, ii, k, _cnt(d1.loc[m, "date"].to_numpy(), d, 730))
                g = d1[any_guru].sort_values("date")
                new = []
                for flr, gg in g.groupby("flr_nm"):
                    ds_ = gg["date"].to_numpy(dtype="datetime64[ns]")
                    gaps = np.diff(ds_, prepend=np.datetime64("1900-01-01")) / np.timedelta64(1, "D")
                    new += [x for x, gp in zip(ds_, gaps, strict=True) if gp > 730 and x >= np.datetime64("2015-01-01")]
                vals = _cnt(np.array(new, dtype="datetime64[ns]"), d, 365)
                _put(out, ii, "gu_kr_new_12m", np.where(d >= np.datetime64("2016-01-01"), vals, np.nan))
            d4 = kr_d_rows(settings, "D004", corp)
            if d4 is not None:
                _put(out, ii, "ma_tender_12m", _cnt(d4["date"].to_numpy(), d, 365))
            est = str(v16.kr_company(settings, corp).get("est_dt") or "")
            if re.match(r"^\d{8}$", est):
                info["est_dt"] += 1
                _put(out, ii, "age_found", _years(d, pd.Timestamp(est)))
        a = kr_audit_table(settings, e)
        if a is not None:
            info["audit"] += 1
            out.iloc[ii, [out.columns.get_loc(c) for c in audit_cols]] = v16._asof(a, audit_cols, "avail", d, 550)
    _kr_etf(out, frame, settings, info)
    return out, info


def _kr_etf(out: pd.DataFrame, frame: pd.DataFrame, settings, info: dict) -> None:
    t = kr_etf_weekly(settings)
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    if t is not None:
        info["etf_weeks"] = int(t["date"].nunique())
        info["etf_codes"] = int(t["code"].nunique())
        info["etf_rule_check"] = etf_class_check(settings)
        dom = t[t["kind"].isin(["lev", "inv", "sector", "market"])]
        wk = sorted(t["date"].unique())

        def flows(sub: pd.DataFrame) -> pd.DataFrame:
            g = sub.groupby("date").agg(flow=("flow", "sum"), aum_prev=("aum_prev", "sum"), aum=("aum", "sum"))
            return g.reindex(wk).fillna({"flow": 0.0, "aum_prev": 0.0, "aum": 0.0})

        dom_g, lev, inv = flows(dom), flows(t[t["kind"] == "lev"]), flows(t[t["kind"] == "inv"])
        ovs, plain = flows(t[t["kind"] == "overseas"]), flows(t[t["kind"].isin(["sector", "market"])])
        base = dom_g["aum"].shift(4).replace(0, np.nan)
        reg = pd.DataFrame(
            {
                "r_kr_etf_levinv_4w": (lev["flow"] - inv["flow"]).rolling(4).sum() / base,
                "r_kr_etf_dom_4w": plain["flow"].rolling(4).sum() / plain["aum"].shift(4).replace(0, np.nan),
                "r_kr_etf_ovs_4w": ovs["flow"].rolling(4).sum() / base,
            }
        )
        reg.index = pd.DatetimeIndex(reg.index) + pd.Timedelta(1, "D")  # a weekly close is known the next day
        for c in reg.columns:
            out[c] = v13._asof_rows(reg[[c]], dates, 10)[:, 0]
        sec = t[t["kind"] == "sector"] if "sector" in frame else t.iloc[:0]
        mcap = np.exp(frame["log_mcap"].astype(float))
        sec_mcap = mcap.groupby([frame["date"], frame["sector"]]).transform("sum").to_numpy()
        for s, g in sec.groupby("sector"):
            f = flows(g)
            a4, a13 = f["aum"].shift(4), f["aum"].shift(13)
            tab = pd.DataFrame(
                {
                    "f4": f["flow"].rolling(4).sum() / a4.where(a4 > 1e10),
                    "f13": f["flow"].rolling(13).sum() / a13.where(a13 > 1e10),
                    "aum": f["aum"].where(f["aum"] > 0),
                }
            )
            tab.index = pd.DatetimeIndex(tab.index) + pd.Timedelta(1, "D")
            ii = np.where(frame["sector"].astype(str).to_numpy() == s)[0]
            if not len(ii):
                continue
            v = v13._asof_rows(tab, dates[ii], 10)
            _put(out, ii, "etf_sec_flow_4w", v[:, 0])
            _put(out, ii, "etf_sec_flow_13w", v[:, 1])
            _put(out, ii, "etf_sec_aum_share", v[:, 2] / sec_mcap[ii])
    _kr_etf_own(out, frame, settings, info)
    ya = yahoo_etf(settings)
    if "EWY" in ya:
        s = _dvol_shock(ya["EWY"], 20, 250).to_frame("x")
        s.index = s.index + pd.Timedelta(1, "D")  # the US session of d-1 is known in Seoul on d
        out["r_kr_ewy_dvol"] = v13._asof_rows(s, dates, 7)[:, 0]


def _kr_etf_own(out: pd.DataFrame, frame: pd.DataFrame, settings, info: dict) -> None:
    """etf_own = sum over the quarter-end PDFs of (weight x ETF net assets) / market cap; known the next day."""
    d = root(settings) / "krx_etf_pdf"
    parts = [pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))] if d.exists() else []
    if not parts:
        info["etf_pdf"] = "no data"
        return
    pdf = pd.concat(parts, ignore_index=True)
    pdf["held"] = pdf["weight"].fillna(0.0) / 100.0 * pdf["etf_aum"]
    held = pdf.groupby(["code", "date"])["held"].sum()
    pdates = pd.DatetimeIndex(sorted(pdf["date"].unique()))
    info["etf_pdf_dates"] = [str(pdates[0].date()), str(pdates[-1].date()), len(pdates)]
    mcap = np.exp(frame["log_mcap"].astype(float))
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    ents = frame["entity"].astype(str).to_numpy()
    for e in np.unique(ents):
        ii = np.where(ents == e)[0]
        gm = pd.Series(mcap.to_numpy()[ii], index=pd.DatetimeIndex(dates[ii])).sort_index()
        gm = gm[~gm.index.duplicated()].dropna()
        if gm.empty:
            continue
        h = held.xs(e, level=0) if e in held.index.get_level_values(0) else pd.Series(dtype=float)
        h = h.reindex(pdates).fillna(0.0)  # in no top ETF's PDF on that date: 0
        mc = gm.reindex(pdates, method="ffill", tolerance=pd.Timedelta(14, "D"))
        own = (h / mc).clip(upper=1.0)
        tab = pd.DataFrame({"etf_own": own, "etf_own_chg_4q": own - own.shift(4)})
        tab = tab[pdates >= gm.index[0]]
        tab.index = tab.index + pd.Timedelta(1, "D")
        out.iloc[ii, [out.columns.get_loc("etf_own"), out.columns.get_loc("etf_own_chg_4q")]] = v13._asof_rows(
            tab, dates[ii], 200
        )


def build_us(frame: pd.DataFrame, ds, settings, cache) -> tuple[pd.DataFrame, dict]:
    from quantcio.data.pit.us_cik import Submissions
    from quantcio.data.pit.us_net import PitNet

    cols = [c for b in BLOCK_COLS["US"].values() for c in b]
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    uni = pd.DataFrame({"entity": np.unique(ents)})
    chains = v16.us_chains(settings, cache, uni)
    net = PitNet(settings)
    wiki = root(settings) / "wikidata_inception.parquet"
    inc = pd.read_parquet(wiki).set_index("cik")["year"].to_dict() if wiki.exists() else {}
    info = {"entities": len(uni), "with_subs": 0, "wikidata": 0, "events": {}}
    for e in uni["entity"]:
        ii = np.where(ents == e)[0]
        d = dates[ii]
        evs, firsts, years = [], [], []
        for c in chains.get(e, set()):
            subs = Submissions(net)
            try:
                ev, first = us_cik_events(subs, int(c))
            except Exception as exc:  # noqa: BLE001 - one CIK failing leaves the others
                say(f"US {e} CIK {c}: {type(exc).__name__}")
                continue
            evs.append(ev)
            if first is not None:
                firsts.append(first)
            if int(c) in inc:
                years.append(inc[int(c)])
        if years:
            info["wikidata"] += 1
            _put(out, ii, "age_found", _years(d, pd.Timestamp(f"{int(min(years))}-07-01")))
        if firsts:
            _put(out, ii, "age_edgar", _years(d, min(firsts)))
        if not evs:
            continue
        ev = pd.concat(evs, ignore_index=True)
        info["with_subs"] += 1
        pub = ev["published"].to_numpy(dtype="datetime64[ns]")
        for k, (item, days) in US_ITEM_FEATURES.items():
            m = ((ev["kind"] == "8k") & (ev["item"] == item)).to_numpy()
            info["events"][k] = info["events"].get(k, 0) + int(m.sum())
            _put(out, ii, k, _cnt(pub[m], d, days))
        for k, (kind, days) in US_FORM_FEATURES.items():
            m = (ev["kind"] == kind).to_numpy()
            info["events"][k] = info["events"].get(k, 0) + int(m.sum())
            _put(out, ii, k, _cnt(pub[m], d, days))
        m = ((ev["kind"] == "8k") & (ev["item"] == "5.02")).to_numpy()
        c90, _ = v16._windowed(pub[m], np.ones(int(m.sum())), d, 90)
        _put(out, ii, "q_502_cluster", (c90 >= 2).astype(float))
    _us_13f(out, frame, settings, info)
    _us_etf(out, frame, settings, info)
    return out, info


def _us_13f(out: pd.DataFrame, frame: pd.DataFrame, settings, info: dict) -> None:
    from quantcio.data.pit.us_cik import entity_cik

    t = f13_managers(settings)
    if t.empty:
        info["13f"] = "no data"
        return
    m = pd.read_csv(kd.root(settings, "sec13f") / "cik_cusip.csv", dtype=str)
    m["cik"] = pd.to_numeric(m["cik"], errors="coerce")
    m = m.dropna(subset=["cik", "cusip6"])
    by_cik = m.groupby(m["cik"].astype(int))["cusip6"].agg(lambda s: sorted({x.strip().upper() for x in s}))
    filed_mgr = t.groupby("period")["mgr"].agg(lambda s: set(s))
    periods = sorted(filed_mgr.index)
    guru = t[t["mgr"].str.startswith("guru:")].copy()
    port = guru.groupby(["mgr", "period"])["value_usd"].transform("sum")
    guru["wt"] = guru["value_usd"] / port.replace(0, np.nan)
    pas = t[t["mgr"].str.startswith("passive:")]
    info["13f_managers"] = sorted(t["mgr"].unique())
    info["13f_periods"] = [str(periods[0].date()), str(periods[-1].date())]
    mcap = np.exp(frame["log_mcap"].astype(float))
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    ents = frame["entity"].astype(str).to_numpy()
    matched = 0
    g_by = guru.groupby("cusip6")
    p_by = pas.groupby("cusip6")
    for e in np.unique(ents):
        try:
            cus = by_cik.get(int(entity_cik(e)))
        except (TypeError, ValueError):
            cus = None
        if not cus:
            continue
        ii = np.where(ents == e)[0]
        gm = pd.Series(mcap.to_numpy()[ii], index=pd.DatetimeIndex(dates[ii])).sort_index()
        gm = gm[~gm.index.duplicated()]
        gp = (
            pd.concat([g_by.get_group(c) for c in cus if c in g_by.groups])
            if any(c in g_by.groups for c in cus)
            else None
        )
        pp = (
            pd.concat([p_by.get_group(c) for c in cus if c in p_by.groups])
            if any(c in p_by.groups for c in cus)
            else None
        )
        matched += 1
        rows = []
        prev_hold: dict[str, float] = {}
        for i, per in enumerate(periods):
            filers = filed_mgr[per]
            h = (
                gp[gp["period"] == per].groupby("mgr").agg(sh=("shares", "sum"), wt=("wt", "sum"))
                if gp is not None
                else None
            )
            hold = h["sh"].to_dict() if h is not None and len(h) else {}
            gurus_filed = {x for x in filers if x.startswith("guru:")}
            prev_filers = filed_mgr[periods[i - 1]] if i else set()
            both = gurus_filed & prev_filers
            new = sum(1 for k in hold if k in both and k not in prev_hold)
            add = sum(1 for k in hold if k in prev_hold and hold[k] > 1.1 * prev_hold[k])
            trim = sum(1 for k in hold if k in prev_hold and hold[k] < 0.9 * prev_hold[k])
            exit_ = sum(1 for k in prev_hold if k in both and k not in hold)
            mc = gm.reindex([per], method="ffill", tolerance=pd.Timedelta(14, "D")).iloc[0] if len(gm) else np.nan
            pv = pp.loc[pp["period"] == per, "value_usd"].sum() if pp is not None else np.nan
            rows.append(
                {
                    "avail": per + pd.Timedelta(v13.F13_DEADLINE + 1, "D"),
                    "gu_n": np.log1p(len(hold)),
                    "gu_new": float(new),
                    "gu_add": float(add),
                    "gu_trim": float(trim),
                    "gu_exit": float(exit_),
                    "gu_wt_max": float(h["wt"].max()) if h is not None and len(h) else 0.0,
                    "etf_passive_own": min(pv / mc, 1.0) if mc and np.isfinite(mc) and mc > 0 else np.nan,
                }
            )
            prev_hold = hold
        tab = pd.DataFrame(rows)
        tab["etf_passive_chg_4q"] = tab["etf_passive_own"] - tab["etf_passive_own"].shift(4)
        c = ["gu_n", "gu_new", "gu_add", "gu_trim", "gu_exit", "gu_wt_max", "etf_passive_own", "etf_passive_chg_4q"]
        out.iloc[ii, [out.columns.get_loc(x) for x in c]] = v16._asof(tab, c, "avail", dates[ii], 200)
    info["13f_entities"] = matched


def _us_etf(out: pd.DataFrame, frame: pd.DataFrame, settings, info: dict) -> None:
    ya = yahoo_etf(settings)
    if not ya:
        info["etf"] = "no data"
        return
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    sec = frame["sector"].astype(str).to_numpy() if "sector" in frame else np.array([""] * len(frame))
    info["us_sectors_seen"] = sorted(set(sec))
    for s, sym in US_SECTOR_ETF.items():
        ii = np.where(sec == s)[0]
        if not len(ii) or sym not in ya:
            continue
        tab = pd.DataFrame({"a": _dvol_shock(ya[sym], 20, 250), "b": _dvol_shock(ya[sym], 5, 60)})
        tab.index = tab.index + pd.Timedelta(1, "D")
        v = v13._asof_rows(tab, dates[ii], 7)
        _put(out, ii, "etf_sec_dvol_20_250", v[:, 0])
        _put(out, ii, "etf_sec_dvol_5_60", v[:, 1])
    if {"SPY", "EFA", "EEM"} <= set(ya):
        lr = {k: np.log(ya[k]["close"]).diff(21) for k in ("SPY", "EFA", "EEM")}
        reg = pd.DataFrame(
            {
                "r_us_etf_efa_rel_1m": lr["EFA"] - lr["SPY"],
                "r_us_etf_eem_rel_1m": lr["EEM"] - lr["SPY"],
                "r_us_etf_spy_dvol": _dvol_shock(ya["SPY"], 20, 250),
            }
        ).dropna(how="all")
        reg.index = reg.index + pd.Timedelta(1, "D")
        for c in reg.columns:
            out[c] = v13._asof_rows(reg[[c]], dates, 7)[:, 0]


def block(market: str, frame: pd.DataFrame, ds, settings, cache) -> tuple[pd.DataFrame, dict]:
    """Every block of the market at once (one pass over the sources)."""
    return build_kr(frame, settings) if market == "KR" else build_us(frame, ds, settings, cache)


# ================================================================================================ main


def main() -> None:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--what", default="all", choices=["all", "kr_d001", "krx_etf", "yahoo_etf", "sec13f", "wikidata"])
    a = ap.parse_args()
    s = load_settings()
    cache = DiskCache(s.cache_path)
    steps = {
        "yahoo_etf": lambda: fetch_yahoo_etf(s),
        "wikidata": lambda: fetch_wikidata(s),
        "kr_d001": lambda: fetch_kr_d001(s, cache),
        "krx_etf": lambda: {**fetch_basic_etf(s), **fetch_krx_etf(s)},
        "sec13f": lambda: fetch_sec13f(s),
    }
    for name, fn in steps.items():
        if a.what in ("all", name):
            try:
                say(f"{name}: {fn()}")
            except Exception as exc:  # noqa: BLE001 - the other sources still load
                say(f"{name}: FAILED {type(exc).__name__}: {kd._redact(str(exc), kd.key('DART_API_KEY'))}")


if __name__ == "__main__":
    main()
