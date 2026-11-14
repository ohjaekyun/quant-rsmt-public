"""KR delisting master (keyless): FDR GitHub cache of KRX MDCSTAT23801 + KIND 상장폐지현황, with reason categories."""
import json, re, time
from io import StringIO
import pandas as pd
import requests

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
FDR_CACHE_API = "https://api.github.com/repos/FinanceData/fdr_krx_data_cache/contents/data/listing/delisting"
FDR_CACHE_RAW = "https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache/refs/heads/master/data/listing/delisting/{}"
KIND_DEL = "https://kind.krx.co.kr/investwarn/delcompany.do"

_REASON_RULES = [  # order matters: first match wins
    ("transfer",        r"이전상장|유가증권시장 상장|코스닥시장 상장"),
    ("spac_merger",     r"스팩소멸합병"),
    ("spac_expiry",     r"상장예비심사\s*청구서\s*미제출|상장예비심사신청서 미제출|합병상장예비심사|SPAC 상장예비심사|상장예비심사청구서 미제출"),
    ("bankruptcy",      r"파산|부도|어음 또는 수표"),
    ("merger",          r"피흡수합병|해산\s*사유\s*발생|해산사유 발생"),
    ("squeeze_out",     r"완전자회사|주식교환"),
    ("voluntary",       r"신청에 의한 상장폐지|상장폐지\s*신청"),
    ("audit_opinion",   r"감사의견|감사범위|한정의견|의견거절"),
    ("capital_impair",  r"자본전액잠식|자본잠식"),
    ("governance",      r"기업의 계속성|경영의 투명성"),
    ("filing_failure",  r"사업보고서|반기보고서|분기보고서|공시서류 미제출"),
    ("konex_adviser",   r"지정자문인"),
    ("pref_thresholds", r"종류주식"),
    ("thresholds",      r"시가총액|분산요건|주식분포|거래량|영업손실|매출액"),
    ("term_expiry",     r"존속기간 만료"),
]

def delist_category(reason: str) -> str:
    s = str(reason or "")
    for cat, rx in _REASON_RULES:
        if re.search(rx, s):
            return cat
    return "other"

def fdr_delisting_latest(session=None) -> tuple[pd.DataFrame, str]:
    """Latest FDR GitHub-cached KRX delisting table (all securities since 1956).
    fdr.StockListing('KRX-DELISTING') silently returns EMPTY when the cache lags KRX max_work_dt,
    so list the directory and take the newest file instead."""
    s = session or requests.Session()
    names = sorted(x["name"] for x in s.get(FDR_CACHE_API, timeout=30).json() if x["name"].endswith(".csv"))
    name = names[-1]
    df = pd.read_csv(FDR_CACHE_RAW.format(name), index_col=0, dtype={"Symbol": str, "ToSymbol": str})
    for c in ["ListingDate", "DelistingDate", "ArrantEnforceDate", "ArrantEndDate"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df.reset_index(drop=True), name.removesuffix(".csv")

def kind_delisted(from_date: str = "2014-01-01", to_date: str | None = None, market_type: str = "", session=None) -> pd.DataFrame:
    """KIND 상장폐지현황 Excel export. market_type: ''=all, '1'=KOSPI, '2'=KOSDAQ, '6'=KONEX.
    Issuers only (no preferreds); ALSO lists market transfers (KONEX->KOSDAQ, KOSDAQ->KOSPI) as '폐지'.
    GOTCHA: with an empty toDate KIND ignores fromDate too (returns full history since 1960 or a
    3-month default), so always send both dates."""
    s = session or requests.Session()
    to_date = to_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    data = dict(method="searchDelCompanySub", forward="delcompany_down", currentPageSize="3000", pageIndex="1",
                orderMode="2", orderStat="D", marketType=market_type, fromDate=from_date, toDate=to_date, tabType="1")
    r = s.post(KIND_DEL, data=data, timeout=60,
               headers={"User-Agent": UA, "Referer": KIND_DEL + "?method=searchDelCompanyMain"})
    r.raise_for_status()
    t = pd.read_html(StringIO(r.content.decode("euc-kr", errors="replace")), converters={"종목코드": str})[0]
    t = t.rename(columns={"회사명": "name", "종목코드": "code", "폐지일자": "delist_date", "폐지사유": "reason", "비고": "note"})
    t["code"] = t["code"].str.zfill(6)
    t["delist_date"] = pd.to_datetime(t["delist_date"])
    return t[["code", "name", "delist_date", "reason", "note"]]

def kind_delisted_by_market(from_date="2014-01-01", to_date=None, pause=0.5, session=None) -> pd.DataFrame:
    out = []
    for mt, lab in [("1", "KOSPI"), ("2", "KOSDAQ"), ("6", "KONEX")]:
        t = kind_delisted(from_date, to_date, mt, session)
        t["market"] = lab
        out.append(t)
        time.sleep(pause)
    return pd.concat(out, ignore_index=True)

def delisting_master(since: str = "2014-01-01") -> pd.DataFrame:
    """Union of FDR/KRX (주권 incl. preferreds) and KIND (issuers, incl. transfers & fresher than FDR cache).
    Columns: code,name,market,kind,listing_date,delist_date,reason,category,src."""
    s = requests.Session()
    f_all, asof = fdr_delisting_latest(s)
    f_all = f_all[f_all.DelistingDate >= since]
    key = set(zip(f_all.Symbol, f_all.DelistingDate))   # de-dup against ALL SecuGroups (ship funds, REITs, 외국주권)
    f = f_all[f_all.SecuGroup == "주권"]
    a = pd.DataFrame({"code": f.Symbol, "name": f.Name, "market": f.Market, "kind": f.Kind,
                      "listing_date": f.ListingDate, "delist_date": f.DelistingDate, "reason": f.Reason,
                      "src": "fdr_krx"})
    k = kind_delisted_by_market(since, None, session=s)
    k = k[k.delist_date >= since]
    k = k[[(c, d) not in key for c, d in zip(k.code, k.delist_date)]]
    b = pd.DataFrame({"code": k.code, "name": k.name, "market": k.market, "kind": "보통주",
                      "listing_date": pd.NaT, "delist_date": k.delist_date, "reason": k.reason, "src": "kind_only"})
    m = pd.concat([a, b], ignore_index=True)
    m["category"] = m.reason.map(delist_category)
    m["is_spac"] = m.name.str.contains(r"스팩|기업인수목적|SPAC", regex=True, na=False)
    m.attrs["fdr_asof"] = asof
    return m.sort_values("delist_date").reset_index(drop=True)
