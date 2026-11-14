"""Point-in-time (PIT) fundamentals for Korean listed companies - keyless.

Sources (all verified live 2026-09-25, no API key):
  * OpenDART "재무정보 일괄다운로드" bulk zips: every KOSPI/KOSDAQ XBRL filer, BS/PL/CF/CE, FY2015 (FY only),
    2016Q1 .. latest quarter; includes later-delisted companies. Values are the LATEST corrected version.
  * DART web detailSearch.ax (per stock code, works for delisted codes): receipt date (rcept_dt) of every
    periodic report (original + [기재정정]) -> the publication date of each fiscal period.
  * DART web per-filing original XBRL ("IFRS(원문XBRL)" download): fills bulk gaps and gives AS-FIRST-REPORTED
    values for later-restated periods.
  * OpenDART API (DART_API_KEY): fnlttSinglAcntAll / fnlttXbrl / stockTotqySttus / list.json - optional.

PIT rule: a fiscal period's values are usable from the first KRX session AFTER the original periodic report's
rcept_dt (filings land up to 18:00 KST, after the 15:30 close). Unknown rcept_dt -> period_end + 60 days
(quarter/half) or + 120 days (annual), rolled to a KRX session, then the next session.
"""

from __future__ import annotations

import csv
import io
import re
import time
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import requests
from lxml import etree

from quantcio.data.http import PoliteSession  # per-host throttle: dart 0.5s, opendart 0.2s
from quantcio.data.kr import parse_dart_search, parse_dart_viewer_nodes

OPENDART_WEB = "https://opendart.fss.or.kr"
OPENDART_API = "https://opendart.fss.or.kr/api"
DART_WEB = "https://dart.fss.or.kr"

# ----------------------------------------------------------------------------------------------- bulk

BULK_PERIOD = {"FQ": "1분기보고서", "HY": "반기보고서", "TQ": "3분기보고서", "FY": "사업보고서"}
_META = ["재무제표종류", "종목코드", "회사명", "시장구분", "업종", "업종명", "결산월", "결산기준일", "보고서종류",
         "통화", "항목코드", "항목명"]


def bulk_file_list(session: requests.Session) -> pd.DataFrame:
    """All bulk zips -> [bsn_yr:int, period:FQ|HY|TQ|FY, stmt:BS|PL|CF|CE, fl_nm, built:Timestamp].
    172 files on 2026-09-25 (2015 FY + 2016Q1..2026HY, x4 statements). One POST, ~90 KB HTML."""
    r = session.post(f"{OPENDART_WEB}/disclosureinfo/fnltt/dwld/list.do", data={},
                     headers={"X-Requested-With": "XMLHttpRequest",
                              "Referer": f"{OPENDART_WEB}/disclosureinfo/fnltt/dwld/main.do"}, timeout=30)
    r.raise_for_status()
    rows = re.findall(r"download_ext002\('(\d{4})',\s*'(\w+)',\s*'(\w+)',\s*'([^']+\.zip)'\)", r.text)
    if not rows:
        raise ValueError("OpenDART bulk list: no download_ext002(...) entries (page changed?)")
    df = pd.DataFrame(rows, columns=["bsn_yr", "period", "stmt", "fl_nm"])
    df["bsn_yr"] = df["bsn_yr"].astype(int)
    df["built"] = pd.to_datetime(df["fl_nm"].str.extract(r"_(\d{14})\.zip$")[0], format="%Y%m%d%H%M%S")
    return df


def fetch_bulk_zip(session: requests.Session, fl_nm: str) -> bytes:
    """GET /cmm/downloadFnlttZip.do?fl_nm=... (no key, no cookie). 1-8 MB, ~0.5-3.5 s."""
    r = session.get(f"{OPENDART_WEB}/cmm/downloadFnlttZip.do", params={"fl_nm": fl_nm}, timeout=300)
    r.raise_for_status()
    if r.content[:2] != b"PK":
        raise ValueError(f"bulk {fl_nm}: not a zip ({r.headers.get('Content-Type')}, {len(r.content)} B)")
    return r.content


def _num(s: str) -> float | None:
    s = (s or "").strip().replace(",", "")
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_bulk_zip(content: bytes) -> pd.DataFrame:
    """Bulk zip -> long frame [member, 재무제표종류, 회사명, 시장구분, 결산월, 보고서종류, 통화, 항목코드, 항목명,
    col, value, stock_code, period_end, consolidated].

    Gotchas handled: member names are cp949 without the zip UTF-8 flag; text is cp949 TSV; the header interleaves
    EMPTY separator columns ('', '당기', '', '', '전기', '전전기', '') so values must be read by header POSITION."""
    out = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for info in z.infolist():
            name = info.filename if info.flag_bits & 0x800 else info.filename.encode("cp437").decode("cp949")
            rdr = csv.reader(io.StringIO(z.read(info).decode("cp949", errors="replace")), delimiter="\t",
                             quoting=csv.QUOTE_NONE)
            header = [c.strip() for c in next(rdr)]
            vcols = [(k, c) for k, c in enumerate(header) if k >= 12 and c]
            recs = []
            for row in rdr:
                if len(row) < 12:
                    continue
                meta = [c.strip() for c in row[:12]]
                for k, c in vcols:
                    v = _num(row[k]) if k < len(row) else None
                    if v is not None:
                        recs.append((*meta, c, v))
            if recs:
                df = pd.DataFrame(recs, columns=[*_META, "col", "value"])
                df.insert(0, "member", name)
                out.append(df)
    df = pd.concat(out, ignore_index=True)
    df["stock_code"] = df["종목코드"].str.strip("[]")
    df["period_end"] = pd.to_datetime(df["결산기준일"], errors="coerce")
    df["consolidated"] = df["member"].str.contains("_연결_")
    return df.drop(columns=["종목코드", "결산기준일", "업종", "업종명"])


# item -> ordered list of standard concept local names (prefix ifrs_/ifrs-full_/dart_ stripped)
KEY_ITEMS = {
    "revenue": ["Revenue"],
    "op_income": ["OperatingIncomeLoss"],
    "ni": ["ProfitLoss"],
    "ni_parent": ["ProfitLossAttributableToOwnersOfParent"],
    "eps": ["BasicEarningsLossPerShare"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["Equity"],
    "equity_parent": ["EquityAttributableToOwnersOfParent"],
    "nci_equity": ["NoncontrollingInterests"],
    "cfo": ["CashFlowsFromUsedInOperatingActivities"],
}
_FLOW = {"revenue", "op_income", "ni", "ni_parent", "eps", "cfo"}
# used only when no standard concept exists for the item (company-extension codes such as
# 'entity00872984_udf_CF_..._StatementOfCashFlowsAbstract' = 영업활동으로부터의 순현금유입, 이마트 2018-2021)
LABEL_FALLBACK = {
    "revenue": r"^(매출액|수익\(매출액\)|영업수익|매출)$",
    "op_income": r"^영업이익(\(손실\))?$",
    "cfo": r"^영업활동(으로인한|으로부터의|으로부터|에서의|의)?(순)?현금(흐름|유입|유출|유입\(유출\)|흐름합계)?$",
}
_STD = re.compile(r"^(?:ifrs-full|ifrs|dart)_(.+)$")


def extract_key_items(long: pd.DataFrame) -> pd.DataFrame:
    """Long bulk rows -> one row per (stock_code, period_end, 보고서종류) with KEY_ITEMS as-reported.

    Columns: <item> (FY: full year / BS: period end / Q reports: YTD cumulative), <item>_3m (Q/H reports only:
    the 3-month quarter), fs ('CFS' consolidated if the company filed consolidated statements for that report,
    else 'OFS'). Fallbacks: ni_parent = sum of company-extension lines '*_ProfitLossAttributableToOwnersOfParent'
    (e.g. NAVER 2021 continuing+discontinued) else ni when no NCI line; equity_parent = equity when no NCI."""
    df = long[long["col"].str.startswith("당기")].copy()
    df["std"] = df["항목코드"].str.extract(_STD)[0]
    df["is_3m"] = df["col"].str.endswith("3개월")
    # value-column names vary by file/year: '당기', '당기 반기 누적', '당기 반기 3개월', '당기 반기말', '당기반기',
    # '당기 3분기', '당기3분기', '당기 3분기말' -> flows prefer YTD ('누적' / bare period), stocks prefer '...말'
    is_end = df["col"].str.endswith("말")
    is_flow_item = df["std"].isin([c for i in _FLOW for c in KEY_ITEMS[i]])
    df["crank"] = np.where(is_flow_item, np.where(is_end, 2, 0), np.where(is_end | df["col"].eq("당기"), 0, 2))
    has_cfs = df.groupby(["stock_code", "period_end", "보고서종류"])["consolidated"].transform("any")
    df = df[df["consolidated"] == has_cfs]
    # PL items may appear in both 02_손익계산서 and 03_포괄손익계산서: prefer 손익계산서 member
    df["mrank"] = np.where(df["member"].str.contains("_02_"), 0, 1)
    df = df.sort_values(["crank", "mrank"], kind="stable")
    keys = ["stock_code", "period_end", "보고서종류"]
    out = df.groupby(keys).agg(fs=("consolidated", lambda s: "CFS" if s.iloc[0] else "OFS"),
                               name=("회사명", "first"), market=("시장구분", "first"),
                               # [VERIFIER FIX] reporting currency: 두산밥캣 (USD), 900xxx (CNY/USD), JTC (JPY)... file
                               # in non-KRW; values/EPS are in that currency. Convert (or exclude) before mixing.
                               currency=("통화", "first"), fy_month=("결산월", "first"))
    for item, concepts in KEY_ITEMS.items():
        x = df[df["std"].isin(concepts)]
        for suffix, part in [("", x[~x["is_3m"]]), ("_3m", x[x["is_3m"]])]:
            if suffix and item not in _FLOW:
                continue
            out[item + suffix] = part.groupby(keys)["value"].first()
    # fallbacks
    label = df["항목명"].str.replace(r"\s+", "", regex=True)
    for item, pat in LABEL_FALLBACK.items():
        hit = df[df["std"].isna() & label.str.match(pat) & ~df["is_3m"]]
        out[item] = out[item].fillna(hit.groupby(keys)["value"].first())
    eps_parts = df[df["std"].isin(["BasicEarningsLossPerShareFromContinuingOperations",
                                   "BasicEarningsLossPerShareFromDiscontinuedOperations"]) & ~df["is_3m"]]
    out["eps"] = out["eps"].fillna(eps_parts.groupby(keys + ["std"])["value"].first().groupby(level=keys).sum())
    ext = df[df["항목코드"].str.endswith("_ProfitLossAttributableToOwnersOfParent") & df["std"].isna()]
    ext_sum = ext[~ext["is_3m"]].groupby(keys)["value"].sum()
    out["ni_parent"] = out["ni_parent"].fillna(ext_sum)
    # [VERIFIER FIX] company-extension parent NI / common EPS (e.g. CJ제일제당, 롯데케미칼, NC 2017-2022:
    # 'entity..._udf_IS_..._ProfitLoss' labelled 지배기업의 소유주지분; LG전자/CJ/삼성전기 EPS as
    # 'entity..._BasicEarningsLossPerShare' / '..._EarningsPerShareAbstract' labelled 보통주기본(및희석)주당이익).
    # Code suffix AND label must both match: the same label is reused for the comprehensive-income attribution.
    ext_np = df[df["std"].isna() & ~df["is_3m"]
                & df["항목코드"].str.contains(r"_(?:ProfitLoss|ProfitLossAttributableToAbstract|"
                                              r"ProfitLossAttributableToOwnersOfParentOfProfitLoss)$")
                & label.str.contains(r"지배") & ~label.str.contains(r"비지배|포괄|계속|중단")]
    ext_np = ext_np.sort_index()  # original statement order: the NI attribution precedes the CI attribution
    cand = ext_np.groupby(keys)["value"].first().reindex(out.index)
    cand = cand.where(out["ni"].isna() | (cand.abs() <= 3 * out["ni"].abs() + 1e8))  # reject implausible picks
    out["ni_parent"] = out["ni_parent"].fillna(cand)
    eps_lab = label.str.replace(r"[.\-·]", "", regex=True)
    ext_eps = df[df["std"].isna() & ~df["is_3m"]
                 & eps_lab.str.match(r"^(?:보통주)?(?:지배기업(?:의)?소유주(?:지분)?)?(?:귀속)?기본(?:및희석|/희석)?주당"
                                     r"(?:순)?(?:이익|손익|손실|순손익)(?:\(손실\))?$")].sort_index()
    out["eps_ext"] = ext_eps.groupby(keys)["value"].first()  # applied after ni_parent is final (see below)
    nci_flow = df[df["std"].eq("ProfitLossAttributableToNoncontrollingInterests")].groupby(keys).size()
    no_nci = ~out.index.isin(nci_flow.index) & out["nci_equity"].isna()
    out.loc[no_nci, "ni_parent"] = out.loc[no_nci, "ni_parent"].fillna(out.loc[no_nci, "ni"])
    out.loc[no_nci, "ni_parent_3m"] = out.loc[no_nci, "ni_parent_3m"].fillna(out.loc[no_nci, "ni_3m"])
    out.loc[no_nci, "equity_parent"] = out.loc[no_nci, "equity_parent"].fillna(out.loc[no_nci, "equity"])
    # [VERIFIER FIX] filer XBRL scale errors exist (삼양사 2016 EPS 9.0e9 for 9,000; 현대로템 2016 257,000 for 257;
    # 대산F&B 2023 net loss -8.08조 for -61.6억). Guard with the implied share count ni_parent / eps.
    def _shares_ok(eps):
        sh = out["ni_parent"] / eps.where(eps != 0)
        return sh.between(3e5, 1.5e10) | out["ni_parent"].abs().lt(1e8)  # tiny NI: implied count meaningless
    out["eps_source"] = np.where(out["eps"].notna(), "std", None)
    ext_ok = out["eps_ext"].where(_shares_ok(out["eps_ext"]))
    fill = out["eps"].isna() & ext_ok.notna()
    out.loc[fill, "eps"] = ext_ok[fill]
    out.loc[fill, "eps_source"] = "ext"
    out["eps_suspect"] = out["eps"].notna() & out["ni_parent"].notna() & ~_shares_ok(out["eps"])
    return out.drop(columns="eps_ext").reset_index()


# ------------------------------------------------------------------------------ publication dates (DART web)

_PERIODIC_TYPES = ["A001", "A002", "A003"]  # 사업 / 반기 / 분기보고서
_PERIOD_RE = re.compile(r"(사업보고서|반기보고서|분기보고서)\s*\((\d{4})\.(\d{2})\)")
BULK_TO_DART_KIND = {"1분기보고서": "분기보고서", "3분기보고서": "분기보고서", "반기보고서": "반기보고서",
                     "사업보고서": "사업보고서"}


def _post_retry(session: requests.Session, url: str, *, data: dict, headers: dict, tries: int = 4) -> requests.Response:
    """Retry transient POST failures (PoliteSession's urllib3 Retry covers GET/HEAD only). NOTE: after ~700 DART
    web requests in ~34 min (0.5 s spacing, 2026-09-25 07:08 KST) dart.fss.or.kr blocked this IP: TLS completes,
    then every request gets RemoteDisconnected (curl: HTTP 000). Retries do not help - stop, persist progress,
    resume later, and keep the sustained rate far lower (<= 10 requests/min)."""
    for k in range(tries):
        try:
            r = session.post(url, data=data, headers=headers, timeout=30)
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout):
            if k == tries - 1:
                raise
            time.sleep(2.0 * 2**k)
    raise AssertionError("unreachable")


def _windows(start: date, end: date, years: int = 5) -> list[tuple[date, date]]:
    """DART web search silently returns '조회 결과가 없습니다' when endDate - startDate > 10 years
    (20150101-20250101 OK, 20150101-20250102 empty) -> query in <= 5-year windows."""
    out, s = [], start
    while s <= end:
        e = min(end, (pd.Timestamp(s) + pd.DateOffset(years=years) - pd.Timedelta(days=1)).date())  # Feb-29 safe
        out.append((s, e))
        s = e + pd.Timedelta(days=1).to_pytimedelta()
    return out


def dart_periodic_filings(session: requests.Session, stock_code: str, start: date, end: date,
                          types: list[str] | None = None, max_pages: int = 20) -> list[dict]:
    """Every periodic report (originals AND corrections) filed by `stock_code` in [start, end], keyless.
    Works for delisted codes. 5-year windows, ~20-35 rows each -> 1 request per window (0.3-0.9 s)."""
    rows: list[dict] = []
    for ws, we in _windows(start, end):
        page, n_pages = 1, 1
        while page <= min(n_pages, max_pages):
            form = {"currentPage": str(page), "maxResults": "100", "maxLinks": "10", "sort": "date",
                    "series": "desc", "startDate": ws.strftime("%Y%m%d"), "endDate": we.strftime("%Y%m%d"),
                    "option": "corp", "textCrpNm": stock_code, "textCrpCik": "",
                    "publicType": types or _PERIODIC_TYPES}
            r = _post_retry(session, f"{DART_WEB}/dsab007/detailSearch.ax", data=form,
                            headers={"Referer": f"{DART_WEB}/dsab007/main.do",
                                     "X-Requested-With": "XMLHttpRequest"})
            part, n_pages = parse_dart_search(r.text)
            if page == 1 and not part and "조회 결과가 없습니다" not in r.text:
                raise ValueError(f"DART search {stock_code}: unparsable result page")
            rows += part
            page += 1
        if n_pages > max_pages:
            raise RuntimeError(f"DART search {stock_code} {ws}..{we}: truncated at {max_pages}/{n_pages} pages")
    return list({r["rcept_no"]: r for r in rows}.values())


def publication_table(rows: list[dict]) -> pd.DataFrame:
    """DART rows -> one row per (kind, period_end): first_rcept_dt/no (earliest filing = the original, or the
    earliest correction if the original is not listed), orig_rcept_dt, last_rcept_dt/no (latest version, which
    is what the bulk file carries), n_filings."""
    recs = []
    for r in rows:
        m = _PERIOD_RE.search(r["report_nm"] or "")
        if not m:
            continue
        recs.append({"kind": m.group(1),
                     "period_end": pd.Timestamp(int(m.group(2)), int(m.group(3)), 1) + pd.offsets.MonthEnd(0),
                     "rcept_dt": pd.Timestamp(r["rcept_dt"]), "rcept_no": r["rcept_no"],
                     "is_correction": r["report_nm"].lstrip().startswith("[")})
    cols = ["kind", "period_end", "first_rcept_dt", "first_rcept_no", "orig_rcept_dt", "last_rcept_dt",
            "last_rcept_no", "n_filings"]
    if not recs:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(recs).sort_values(["rcept_dt", "rcept_no"])
    g = df.groupby(["kind", "period_end"])
    out = g.agg(first_rcept_dt=("rcept_dt", "first"), first_rcept_no=("rcept_no", "first"),
                last_rcept_dt=("rcept_dt", "last"), last_rcept_no=("rcept_no", "last"),
                n_filings=("rcept_no", "size"))
    out["orig_rcept_dt"] = df[~df["is_correction"]].groupby(["kind", "period_end"])["rcept_dt"].first()
    return out.reset_index()[cols]


# ------------------------------------------------------------------------------------------- lag rule


def next_session_after(d: pd.Timestamp, sessions: pd.DatetimeIndex) -> pd.Timestamp:
    """First KRX session strictly after calendar date d."""
    if d is None or pd.isna(d):  # [VERIFIER FIX] NaT used to raise AttributeError
        return pd.NaT
    d = pd.Timestamp(d).normalize()
    if len(sessions) == 0 or d < sessions[0] - pd.Timedelta(days=7):  # [VERIFIER FIX] calendar does not cover d:
        raise ValueError(f"session calendar starts {sessions[0] if len(sessions) else None}, cannot place {d.date()}")
    i = sessions.searchsorted(d, side="right")
    return sessions[i] if i < len(sessions) else pd.NaT


def available_from(period_end: pd.Timestamp, kind: str, rcept_dt: pd.Timestamp | None,
                   sessions: pd.DatetimeIndex, fallback_days: tuple[int, int] = (60, 120)) -> tuple[pd.Timestamp, str]:
    """-> (first session the period's values may be used, 'rcept'|'lag_rule').
    kind: '사업보고서' | '반기보고서' | '분기보고서' (bulk 보고서종류 is mapped via BULK_TO_DART_KIND)."""
    if rcept_dt is not None and not pd.isna(rcept_dt):
        return next_session_after(rcept_dt, sessions), "rcept"
    days = fallback_days[1] if kind == "사업보고서" else fallback_days[0]
    d = pd.Timestamp(period_end) + pd.Timedelta(days=days)
    i = sessions.searchsorted(d)  # roll a weekend/holiday deadline forward ...
    d = sessions[i] if i < len(sessions) else d
    return next_session_after(d, sessions), "lag_rule"  # ... and use it the session after


# ------------------------------------------------------------------------- per-filing XBRL (keyless)

_XBRLI = "http://www.xbrl.org/2003/instance"
_XBRLDI = "http://xbrl.org/2006/xbrldi"


def dart_main_dcm_no(session: requests.Session, rcept_no: str) -> str:
    r = session.get(f"{DART_WEB}/dsaf001/main.do", params={"rcpNo": rcept_no}, timeout=30)
    r.raise_for_status()
    nodes = parse_dart_viewer_nodes(r.text)
    if not nodes:
        raise ValueError(f"DART main.do {rcept_no}: no document nodes")
    return nodes[0][1]


def dart_filing_xbrl(session: requests.Session, rcept_no: str, dcm_no: str | None = None) -> bytes | None:
    """Original XBRL zip of one filing, keyless (0.1-1.3 MB). Requires ONLY Referer = the download-menu URL
    (without it: HTTP 200 + 0 bytes). None when the filing has no XBRL (financial companies before 2023Q3,
    some small filers: the menu offers 'pdf' only)."""
    dcm_no = dcm_no or dart_main_dcm_no(session, rcept_no)
    ref = f"{DART_WEB}/pdf/download/main.do?rcp_no={rcept_no}&dcm_no={dcm_no}"
    r = session.get(f"{DART_WEB}/pdf/download/ifrs.do", params={"rcp_no": rcept_no, "dcm_no": dcm_no, "lang": "ko"},
                    headers={"Referer": ref}, timeout=60)
    r.raise_for_status()
    return r.content if r.content[:2] == b"PK" else None


def xbrl_key_items(zip_bytes: bytes, period_end: pd.Timestamp) -> dict[str, float]:
    """XBRL instance -> KEY_ITEMS for the current period (consolidated if present, else separate; flows = the
    longest duration ending at period_end, i.e. FY or YTD)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        inst = [n for n in z.namelist() if n.lower().endswith(".xbrl")]
        # recover=True: some filers declare invalid namespace URIs (e.g. xmlns:XBRLConsultingFirm='자체작성')
        root = etree.fromstring(z.read(inst[0]), parser=etree.XMLParser(recover=True, huge_tree=True))
    ctx = {}
    for c in root.iter(f"{{{_XBRLI}}}context"):
        per = c.find(f"{{{_XBRLI}}}period")
        dims = [(m.text or "").strip() for m in c.iter(f"{{{_XBRLDI}}}explicitMember")]
        dims += ["typed:" + (m.get("dimension") or "") for m in c.iter(f"{{{_XBRLDI}}}typedMember")]  # [VERIFIER FIX]
        ctx[c.get("id")] = (per.findtext(f"{{{_XBRLI}}}startDate"), per.findtext(f"{{{_XBRLI}}}endDate"),
                            per.findtext(f"{{{_XBRLI}}}instant"), dims)
    pe = pd.Timestamp(period_end)
    best: dict[str, tuple] = {}
    want = {c: item for item, cs in KEY_ITEMS.items() for c in cs}
    for el in root:
        if not isinstance(el.tag, str) or el.get("contextRef") is None:
            continue
        item = want.get(etree.QName(el).localname)
        if item is None:
            continue
        s, e, i, dims = ctx.get(el.get("contextRef"), (None, None, None, []))
        if len(dims) > 1 or (dims and not re.search(r"(Consolidated|Separate)Member$", dims[0])):
            continue
        end = pd.Timestamp(i or e) if (i or e) else None
        if end != pe:
            continue
        try:
            v = float(el.text)
        except (TypeError, ValueError):
            continue
        cons = 1 if (not dims or dims[0].endswith("ConsolidatedMember")) else 0
        dur = (pd.Timestamp(e) - pd.Timestamp(s)).days if s and e else 0
        rank = (cons, dur)
        if item not in best or rank > best[item][0]:
            best[item] = (rank, v)
    # [VERIFIER FIX] never mix statements: a consolidated filer that tags one line only under an extension concept
    # (현대차 FY2018 consolidated OP; 이마트 2019Q1 consolidated EPS) used to get the SEPARATE value for that item.
    filing_cons = any(rank[0] == 1 for rank, _ in best.values())
    return {k: v for k, (rank, v) in best.items() if rank[0] == 1 or not filing_cons}


# ------------------------------------------------------------------------- OpenDART API (DART_API_KEY)

REPRT_CODE = {"1분기보고서": "11013", "반기보고서": "11012", "3분기보고서": "11014", "사업보고서": "11011"}


def opendart_json(session: requests.Session, key: str, api: str, **params) -> dict:
    """GET https://opendart.fss.or.kr/api/<api>.json. Errors (verified 2026-09-25): no crtfc_key -> HTTP 200
    text/html 4694-byte 'OpenDART시스템' page (no charset); bad key -> {"status":"010","message":"등록되지 않은
    인증키입니다."}; 013 = no data; 020 = daily limit (~20,000 calls); 021 = >100 companies (fnlttMultiAcnt)."""
    r = session.get(f"{OPENDART_API}/{api}.json", params={"crtfc_key": key, **params}, timeout=30)
    r.raise_for_status()
    if "json" not in r.headers.get("Content-Type", ""):
        raise RuntimeError(f"OpenDART {api}: non-JSON response (missing/invalid crtfc_key or path)")
    j = r.json()
    if j.get("status") not in ("000", "013"):
        raise RuntimeError(f"OpenDART {api}: {j.get('status')} {j.get('message')}")
    return j


def opendart_fnltt_all(session, key: str, corp_code: str, bsns_year: int, reprt_code: str,
                       fs_div: str = "CFS") -> pd.DataFrame:
    """fnlttSinglAcntAll: every XBRL line of one periodic report (2015+). fs_div CFS|OFS. list fields: rcept_no,
    reprt_code, bsns_year, corp_code, sj_div (BS|IS|CIS|CF|SCE), sj_nm, account_id (XBRL id, e.g.
    'ifrs-full_Revenue', or '-표준계정코드 미사용-'), account_nm, account_detail (SCE only), thstrm_nm,
    thstrm_amount (Q reports' IS/CIS: 3 months), thstrm_add_amount (YTD), frmtrm_nm, frmtrm_amount,
    frmtrm_q_nm, frmtrm_q_amount, frmtrm_add_amount, bfefrmtrm_nm, bfefrmtrm_amount (annual only), ord,
    currency. Returns the latest (corrected) version - rcept_no tells which filing."""
    j = opendart_json(session, key, "fnlttSinglAcntAll", corp_code=corp_code, bsns_year=str(bsns_year),
                      reprt_code=reprt_code, fs_div=fs_div)
    return pd.DataFrame(j.get("list") or [])


def opendart_xbrl(session, key: str, rcept_no: str, reprt_code: str) -> bytes:
    """fnlttXbrl.xml: original XBRL zip of ONE filing (use the ORIGINAL rcept_no for as-first-reported values)."""
    r = session.get(f"{OPENDART_API}/fnlttXbrl.xml", params={"crtfc_key": key, "rcept_no": rcept_no,
                                                              "reprt_code": reprt_code}, timeout=60)
    r.raise_for_status()
    if r.content[:2] != b"PK":
        m = re.search(rb"<status>(\d+)</status>\s*<message>(.*?)</message>", r.content, re.S)
        raise RuntimeError(f"fnlttXbrl {rcept_no}: {m.group(1).decode()} {m.group(2).decode()}" if m
                           else f"fnlttXbrl {rcept_no}: not a zip")
    return r.content


def opendart_share_counts(session, key: str, corp_code: str, bsns_year: int, reprt_code: str) -> pd.DataFrame:
    """stockTotqySttus (2015+): se (보통주/우선주/합계), istc_totqy (발행주식총수), tesstk_co (자기주식수),
    distb_stock_co (유통주식수), stlm_dt, rcept_no, corp_cls (Y/K/N/E). Not in any bulk file."""
    j = opendart_json(session, key, "stockTotqySttus", corp_code=corp_code, bsns_year=str(bsns_year),
                      reprt_code=reprt_code)
    return pd.DataFrame(j.get("list") or [])


# ------------------------------------------------------------------------------------ assemble PIT panel


def quarterly_standalone(items: pd.DataFrame) -> pd.DataFrame:
    """Per company, standalone quarter flows: Q1/Q2/Q3 = the '3개월' column, Q4 = FY - 3Q YTD (fiscal order by
    period_end). Stocks (assets, equity...) are copied as of each period end."""
    order = {"1분기보고서": 1, "반기보고서": 2, "3분기보고서": 3, "사업보고서": 4}
    df = items.copy()
    df["fq"] = df["보고서종류"].map(order)
    df = df.sort_values(["stock_code", "period_end"])
    for f in ["revenue", "op_income", "ni", "ni_parent", "cfo"]:
        q = df[f + "_3m"] if f + "_3m" in df else pd.Series(np.nan, index=df.index)
        prev_ytd = df.groupby("stock_code")[f].shift(1)
        prev_is_q3 = df.groupby("stock_code")["fq"].shift(1).eq(3)
        gap_ok = (df["period_end"] - df.groupby("stock_code")["period_end"].shift(1)).dt.days.between(80, 100)
        q4 = (df[f] - prev_ytd).where(df["fq"].eq(4) & prev_is_q3 & gap_ok)
        # [VERIFIER FIX] Q1: YTD == 3 months when the '3개월' column is absent; Q2/Q3: YTD difference
        prev_fq = df.groupby("stock_code")["fq"].shift(1)
        q = q.fillna(df[f].where(df["fq"].eq(1)))
        q = q.fillna((df[f] - prev_ytd).where(df["fq"].isin([2, 3]) & prev_fq.eq(df["fq"] - 1) & gap_ok))
        df[f + "_q"] = q.where(df["fq"].lt(4), q4)
        if f == "cfo":  # CF is YTD-only in quarterly reports: diff consecutive YTDs within the fiscal year
            prev = df.groupby("stock_code")[f].shift(1)
            df["cfo_q"] = np.where(df["fq"].eq(1), df[f], (df[f] - prev).where(gap_ok))
    return df


def build_pit_panel(items: pd.DataFrame, pubs: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """items (extract_key_items / quarterly_standalone) + pubs (publication_table rows with a 'stock_code'
    column) -> items with available_from, date_source, first/last rcept, restated_later flag."""
    df = items.copy()
    df["kind"] = df["보고서종류"].map(BULK_TO_DART_KIND)
    df = df.merge(pubs, on=["stock_code", "kind", "period_end"], how="left")
    res = [available_from(p, k, r, sessions) for p, k, r in zip(df["period_end"], df["kind"], df["first_rcept_dt"])]
    df["available_from"] = [a for a, _ in res]
    df["date_source"] = [s for _, s in res]
    # bulk carries the LATEST version: if a correction came later, values may differ from what was first public
    df["restated_later"] = df["last_rcept_dt"].gt(df["first_rcept_dt"])
    df["latest_version_from"] = [next_session_after(d, sessions) if not pd.isna(d) else pd.NaT
                                 for d in df["last_rcept_dt"]]
    return df


def asof(panel: pd.DataFrame, when: pd.Timestamp, strict_restatements: bool = True) -> pd.DataFrame:
    """Latest usable report per company as of session `when`. strict_restatements=True uses a period's (latest,
    bulk) values only from latest_version_from when it was corrected later (no restatement look-ahead)."""
    when = pd.Timestamp(when)
    usable = panel["available_from"].le(when)
    if strict_restatements:
        usable &= ~panel["restated_later"].fillna(False) | panel["latest_version_from"].le(when)
    x = panel[usable].sort_values(["stock_code", "period_end", "available_from"])
    return x.groupby("stock_code").tail(1)


def kind_delisted(session: requests.Session, start: date, end: date) -> pd.DataFrame:
    """KIND 상장폐지현황 (KOSPI+KOSDAQ+KONEX, no market column) -> [회사명, 종목코드, 폐지일자, 폐지사유, 비고].
    POST with forward=delcompany_down returns an EUC-KR HTML table served as .xls (759 rows 2014-01..2026-09)."""
    form = {"method": "searchDelCompanySub", "forward": "delcompany_down", "currentPageSize": "3000",
            "pageIndex": "1", "isurCd": "", "repIsuSrtCd": "", "searchCorpName": "", "marketType": "",
            "fromDate": start.isoformat(), "toDate": end.isoformat(), "orderMode": "", "orderStat": ""}
    r = session.post("https://kind.krx.co.kr/investwarn/delcompany.do", data=form, timeout=60,
                     headers={"Referer": "https://kind.krx.co.kr/investwarn/delcompany.do?method=searchDelCompanyMain"})
    r.raise_for_status()
    df = pd.read_html(io.StringIO(r.content.decode("euc-kr", errors="replace")), converters={"종목코드": str})[0]
    df["종목코드"] = df["종목코드"].str.zfill(6)
    df["폐지일자"] = pd.to_datetime(df["폐지일자"])
    return df.drop(columns=["번호"], errors="ignore")


def new_session() -> requests.Session:
    return PoliteSession()
