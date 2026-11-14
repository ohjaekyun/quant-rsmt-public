"""OpenDART bulk financial statements (재무정보 일괄다운로드, keyless): FY2015, then every quarter from 2016Q1.

Ported from ``docs/sources/pit/reference/bulk.py`` and the verifier-fixed ``kr_pit_fundamentals.py``
(``docs/sources/pit/kr-fundamentals.json``). Every KOSPI/KOSDAQ XBRL filer incl. later-delisted companies; the
values are the LATEST corrected version (restatements are handled in kr_fund.py with DART receipt dates).

Gotchas handled:
- member names are cp949 without the zip UTF-8 flag; the TSV text is cp949;
- the header interleaves EMPTY separator columns ('', '당기', '', '', '전기', ...): values are read by POSITION;
- value-column names vary ('당기', '당기 반기 누적', '당기 반기 3개월', '당기 반기말', '당기3분기' ...): flows prefer
  YTD, balance-sheet items prefer '...말';
- consolidated statements when the company filed ANY consolidated statement for that report, else separate;
- the reporting currency (통화) is kept: 919 company-periods are CNY/USD/JPY/HKD/GBP (e.g. 두산밥캣 USD);
- extension-coded parent NI / EPS (CJ제일제당, 롯데케미칼, NC, LG전자, 삼성전기 ...): code suffix AND label must match;
- the no-NCI fallback (ni_parent = ni) applies only when neither an NCI profit line nor NCI equity exists;
- filer XBRL scale errors in EPS are flagged (eps_suspect). EPS is as-reported and NOT split-adjusted: it is never
  used for valuation (callers use market cap / aggregate NI or equity).

Only the lines the extraction can use are materialised while parsing (key concepts + extension candidates), which
keeps a period's parse at a few MB instead of ~1.5M long rows; ``has_cfs`` is computed from ALL rows.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from typing import Any

import numpy as np
import pandas as pd

OPENDART_WEB = "https://opendart.fss.or.kr"
BULK_PERIOD = {"FQ": "1분기보고서", "HY": "반기보고서", "TQ": "3분기보고서", "FY": "사업보고서"}
REPORT_FQ = {"1분기보고서": 1, "반기보고서": 2, "3분기보고서": 3, "사업보고서": 4}
META_COLS = [
    "재무제표종류",
    "종목코드",
    "회사명",
    "시장구분",
    "업종",
    "업종명",
    "결산월",
    "결산기준일",
    "보고서종류",
    "통화",
    "항목코드",
    "항목명",
]
STATEMENTS = ("BS", "PL", "CF")  # CE (자본변동표) is not needed

# item -> standard concept local names (prefix ifrs_/ifrs-full_/dart_ stripped)
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
FLOW_ITEMS = {"revenue", "op_income", "ni", "ni_parent", "eps", "cfo"}
BS_ITEMS = ("assets", "liabilities", "equity", "equity_parent", "nci_equity")
_EPS_PARTS = (
    "BasicEarningsLossPerShareFromContinuingOperations",
    "BasicEarningsLossPerShareFromDiscontinuedOperations",
)
_NCI_FLOW = "ProfitLossAttributableToNoncontrollingInterests"
WANTED_STD = frozenset({c for cs in KEY_ITEMS.values() for c in cs} | set(_EPS_PARTS) | {_NCI_FLOW})
# used only when no standard concept exists for the item (company-extension codes such as
# 'entity00872984_udf_CF_..._StatementOfCashFlowsAbstract' = 영업활동으로부터의 순현금유입, 이마트 2018-2021)
LABEL_FALLBACK = {
    "revenue": r"^(매출액|수익\(매출액\)|영업수익|매출)$",
    "op_income": r"^영업이익(\(손실\))?$",
    "cfo": r"^영업활동(으로인한|으로부터의|으로부터|에서의|의)?(순)?현금(흐름|유입|유출|유입\(유출\)|흐름합계)?$",
}
_STD = re.compile(r"^(?:ifrs-full|ifrs|dart)_(.+)$")
_LABEL_FALLBACK_RE = {k: re.compile(v) for k, v in LABEL_FALLBACK.items()}
_EXT_NP_CODE = re.compile(
    r"_(?:ProfitLoss|ProfitLossAttributableToAbstract|ProfitLossAttributableToOwnersOfParentOfProfitLoss)$"
)
_EXT_NP_EXCL = re.compile(r"비지배|포괄|계속|중단")
_EPS_LABEL = re.compile(
    r"^(?:보통주)?(?:지배기업(?:의)?소유주(?:지분)?)?(?:귀속)?기본(?:및희석|/희석)?주당(?:순)?(?:이익|손익|손실|순손익)(?:\(손실\))?$"
)
_WS = re.compile(r"\s+")
_EPS_PUNCT = re.compile(r"[.\-·]")


# ----------------------------------------------------------------------------------------------- network


def parse_bulk_list(html: str) -> pd.DataFrame:
    """list.do HTML -> [bsn_yr:int, period:FQ|HY|TQ|FY, stmt:BS|PL|CF|CE, fl_nm, built:Timestamp]."""
    rows = re.findall(r"download_ext002\('(\d{4})',\s*'(\w+)',\s*'(\w+)',\s*'([^']+\.zip)'\)", html or "")
    if not rows:
        raise ValueError("OpenDART 일괄다운로드 목록: download_ext002(...) 항목 없음 (페이지 구조 변경?)")
    df = pd.DataFrame(rows, columns=["bsn_yr", "period", "stmt", "fl_nm"])
    df["bsn_yr"] = df["bsn_yr"].astype(int)
    df["built"] = pd.to_datetime(df["fl_nm"].str.extract(r"_(\d{14})\.zip$")[0], format="%Y%m%d%H%M%S")
    return df


def fetch_bulk_list(session: Any) -> pd.DataFrame:
    """All bulk zips (one POST, ~90 KB HTML; 172 files on 2026-09-25). File names change on every rebuild."""
    r = session.post(
        f"{OPENDART_WEB}/disclosureinfo/fnltt/dwld/list.do",
        data={},
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"{OPENDART_WEB}/disclosureinfo/fnltt/dwld/main.do",
        },
        timeout=30,
    )
    r.raise_for_status()
    return parse_bulk_list(r.text)


def fetch_bulk_zip(session: Any, fl_nm: str) -> bytes:
    """GET /cmm/downloadFnlttZip.do?fl_nm=... (no key, no cookie; 1-8 MB). A bad name returns HTML -> ValueError."""
    r = session.get(f"{OPENDART_WEB}/cmm/downloadFnlttZip.do", params={"fl_nm": fl_nm}, timeout=300)
    r.raise_for_status()
    if r.content[:2] != b"PK":
        raise ValueError(f"bulk {fl_nm}: zip 아님 ({r.headers.get('Content-Type')}, {len(r.content)} B)")
    return r.content


def latest_files(listing: pd.DataFrame) -> pd.DataFrame:
    """Newest build per (bsn_yr, period, stmt), BS/PL/CF only, one row per (bsn_yr, period) with the 3 names."""
    x = listing[listing["stmt"].isin(STATEMENTS)].sort_values("built")
    x = x.groupby(["bsn_yr", "period", "stmt"]).tail(1)
    wide = x.pivot(index=["bsn_yr", "period"], columns="stmt", values="fl_nm").reset_index()
    return wide.sort_values(["bsn_yr", "period"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------- parse


def _num(s: str) -> float | None:
    s = (s or "").strip().replace(",", "")
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _keep_line(code: str, label: str) -> bool:
    """True when extract_key_items can use this line (standard key concept or an extension-code candidate)."""
    m = _STD.match(code)
    if m:
        return m.group(1) in WANTED_STD
    lab = _WS.sub("", label)
    if any(p.match(lab) for p in _LABEL_FALLBACK_RE.values()):
        return True
    if code.endswith("_ProfitLossAttributableToOwnersOfParent"):
        return True
    if _EXT_NP_CODE.search(code) and "지배" in lab and not _EXT_NP_EXCL.search(lab):
        return True
    return bool(_EPS_LABEL.match(_EPS_PUNCT.sub("", lab)))


def read_bulk_zip(content: bytes) -> tuple[pd.DataFrame, set[tuple[str, str, str]]]:
    """Bulk zip -> (long frame of usable lines, consolidated report keys).

    Long columns: member, 재무제표종류, 회사명, 시장구분, 결산월, 보고서종류, 통화, 항목코드, 항목명, col (value
    header, '당기...' only), value, stock_code, period_end, consolidated. The key set holds (stock_code, 결산기준일,
    보고서종류) of every report with at least one numeric current-period value in a CONSOLIDATED member (all lines,
    not only the kept ones) - the consolidated-else-separate choice needs it.
    """
    recs: list[tuple] = []
    cons_keys: set[tuple[str, str, str]] = set()
    decisions: dict[tuple[str, str], bool] = {}
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for info in z.infolist():
            name = info.filename if info.flag_bits & 0x800 else info.filename.encode("cp437").decode("cp949", "replace")
            consolidated = "_연결_" in name
            rdr = csv.reader(
                io.StringIO(z.read(info).decode("cp949", errors="replace")), delimiter="\t", quoting=csv.QUOTE_NONE
            )
            header = [c.strip() for c in next(rdr, [])]
            vcols = [(k, c) for k, c in enumerate(header) if k >= 12 and c.startswith("당기")]
            if not vcols:
                continue
            for row in rdr:
                if len(row) < 12:
                    continue
                dk = (row[10].strip(), row[11].strip())
                keep = decisions.get(dk)
                if keep is None:
                    keep = decisions[dk] = _keep_line(*dk)
                key = None
                if consolidated:
                    key = (row[1].strip().strip("[]"), row[7].strip(), row[8].strip())
                    if key in cons_keys:
                        key = None
                if not keep and key is None:
                    continue
                vals = [(c, v) for k, c in vcols if k < len(row) and (v := _num(row[k])) is not None]
                if not vals:
                    continue
                if key is not None:
                    cons_keys.add(key)
                if keep:
                    meta = tuple(c.strip() for c in row[:12])
                    for c, v in vals:
                        recs.append((name, *meta, c, v))
    df = pd.DataFrame(recs, columns=["member", *META_COLS, "col", "value"])
    df["stock_code"] = df["종목코드"].str.strip("[]")
    df["period_end"] = pd.to_datetime(df["결산기준일"], errors="coerce")
    df["consolidated"] = df["member"].str.contains("_연결_")
    return df.drop(columns=["종목코드", "결산기준일", "업종", "업종명"]), cons_keys


def extract_key_items(long: pd.DataFrame, cons_keys: set[tuple[str, str, str]]) -> pd.DataFrame:
    """Long bulk rows -> one row per (stock_code, period_end, 보고서종류) with KEY_ITEMS as reported.

    Columns: <item> (FY: full year / BS: period end / Q reports: YTD), <item>_3m (Q/H reports: the 3-month
    quarter), fs ('CFS' when the company filed consolidated statements for that report, else 'OFS'), name, market
    (PIT 시장구분 as filed), currency (reporting currency), fy_month, eps_source, eps_suspect.
    """
    keys = ["stock_code", "period_end", "보고서종류"]
    df = long[long["col"].str.startswith("당기")].copy()
    if df.empty:
        return pd.DataFrame(columns=[*keys, "fs", "name", "market", "currency", "fy_month", *KEY_ITEMS])
    df["std"] = df["항목코드"].str.extract(_STD)[0]
    df["is_3m"] = df["col"].str.endswith("3개월")
    is_end = df["col"].str.endswith("말")
    is_flow_item = df["std"].isin([c for i in FLOW_ITEMS for c in KEY_ITEMS[i]])
    df["crank"] = np.where(is_flow_item, np.where(is_end, 2, 0), np.where(is_end | df["col"].eq("당기"), 0, 2))
    ck = pd.DataFrame(list(cons_keys), columns=["stock_code", "pe_raw", "보고서종류"])
    ck["period_end"] = pd.to_datetime(ck["pe_raw"], errors="coerce")
    ck = ck.drop(columns="pe_raw").drop_duplicates(keys).assign(has_cfs=True)
    df = df.merge(ck, on=keys, how="left")
    df["has_cfs"] = df["has_cfs"].eq(True)
    df = df[df["consolidated"] == df["has_cfs"]]
    # PL items may appear in both 02_손익계산서 and 03_포괄손익계산서: prefer the 손익계산서 member
    df["mrank"] = np.where(df["member"].str.contains("_02_"), 0, 1)
    df = df.sort_values(["crank", "mrank"], kind="stable")
    out = df.groupby(keys).agg(
        fs=("consolidated", lambda s: "CFS" if s.iloc[0] else "OFS"),
        name=("회사명", "first"),
        market=("시장구분", "first"),
        currency=("통화", "first"),
        fy_month=("결산월", "first"),
    )
    for item, concepts in KEY_ITEMS.items():
        x = df[df["std"].isin(concepts)]
        for suffix, part in (("", x[~x["is_3m"]]), ("_3m", x[x["is_3m"]])):
            if suffix and item not in FLOW_ITEMS:
                continue
            out[item + suffix] = part.groupby(keys)["value"].first()
    label = df["항목명"].str.replace(r"\s+", "", regex=True)
    for item, pat in LABEL_FALLBACK.items():
        hit = df[df["std"].isna() & label.str.match(pat) & ~df["is_3m"]]
        out[item] = out[item].fillna(hit.groupby(keys)["value"].first())
    eps_parts = df[df["std"].isin(list(_EPS_PARTS)) & ~df["is_3m"]]
    out["eps"] = out["eps"].fillna(eps_parts.groupby([*keys, "std"])["value"].first().groupby(level=keys).sum())
    ext = df[df["항목코드"].str.endswith("_ProfitLossAttributableToOwnersOfParent") & df["std"].isna()]
    out["ni_parent"] = out["ni_parent"].fillna(ext[~ext["is_3m"]].groupby(keys)["value"].sum())
    # extension-coded parent NI: code suffix AND label (the label alone is reused for the CI attribution)
    ext_np = df[
        df["std"].isna()
        & ~df["is_3m"]
        & df["항목코드"].str.contains(_EXT_NP_CODE.pattern)
        & label.str.contains(r"지배")
        & ~label.str.contains(_EXT_NP_EXCL.pattern)
    ].sort_index()  # original statement order: the NI attribution precedes the CI attribution
    cand = ext_np.groupby(keys)["value"].first().reindex(out.index)
    cand = cand.where(out["ni"].isna() | (cand.abs() <= 3 * out["ni"].abs() + 1e8))
    out["ni_parent"] = out["ni_parent"].fillna(cand)
    eps_lab = label.str.replace(r"[.\-·]", "", regex=True)
    ext_eps = df[df["std"].isna() & ~df["is_3m"] & eps_lab.str.match(_EPS_LABEL.pattern)].sort_index()
    out["eps_ext"] = ext_eps.groupby(keys)["value"].first()
    nci_flow = df[df["std"].eq(_NCI_FLOW)].groupby(keys).size()
    no_nci = ~out.index.isin(nci_flow.index) & out["nci_equity"].isna()
    out.loc[no_nci, "ni_parent"] = out.loc[no_nci, "ni_parent"].fillna(out.loc[no_nci, "ni"])
    out.loc[no_nci, "ni_parent_3m"] = out.loc[no_nci, "ni_parent_3m"].fillna(out.loc[no_nci, "ni_3m"])
    out.loc[no_nci, "equity_parent"] = out.loc[no_nci, "equity_parent"].fillna(out.loc[no_nci, "equity"])

    def _shares_ok(eps: pd.Series) -> pd.Series:  # filer scale errors: the implied share count must be plausible
        sh = out["ni_parent"] / eps.where(eps != 0)
        return sh.between(3e5, 1.5e10) | out["ni_parent"].abs().lt(1e8)

    out["eps_source"] = np.where(out["eps"].notna(), "std", None)
    ext_ok = out["eps_ext"].where(_shares_ok(out["eps_ext"]))
    fill = out["eps"].isna() & ext_ok.notna()
    out.loc[fill, "eps"] = ext_ok[fill]
    out.loc[fill, "eps_source"] = "ext"
    out["eps_suspect"] = out["eps"].notna() & out["ni_parent"].notna() & ~_shares_ok(out["eps"])
    return out.drop(columns="eps_ext").reset_index()


def quarterly_standalone(items: pd.DataFrame) -> pd.DataFrame:
    """Per company standalone quarter flows <item>_q: Q1 = YTD, Q2/Q3 = the '3개월' column (else the YTD difference
    with the previous quarter), Q4 = FY - 3Q YTD. Consecutive reports must be 80-100 days apart. A negative derived
    Q4 revenue (FY reclassifies discontinued operations, the 3Q YTD does not) is set to NaN."""
    df = items.copy()
    df["fq"] = df["보고서종류"].map(REPORT_FQ)
    df = df.sort_values(["stock_code", "period_end", "fq"])
    g = df.groupby("stock_code")
    prev_fq = g["fq"].shift(1)
    gap_ok = (df["period_end"] - g["period_end"].shift(1)).dt.days.between(80, 100)
    for f in ("revenue", "op_income", "ni", "ni_parent", "cfo"):
        q = df[f + "_3m"] if f + "_3m" in df else pd.Series(np.nan, index=df.index)
        prev_ytd = g[f].shift(1)
        q4 = (df[f] - prev_ytd).where(df["fq"].eq(4) & prev_fq.eq(3) & gap_ok)
        q = q.fillna(df[f].where(df["fq"].eq(1)))
        q = q.fillna((df[f] - prev_ytd).where(df["fq"].isin([2, 3]) & prev_fq.eq(df["fq"] - 1) & gap_ok))
        df[f + "_q"] = q.where(df["fq"].lt(4), q4)
        if f == "cfo":  # CF is YTD-only in quarterly reports: diff consecutive YTDs within the fiscal year
            df["cfo_q"] = np.where(df["fq"].eq(1), df[f], (df[f] - prev_ytd).where(gap_ok & prev_fq.eq(df["fq"] - 1)))
    df.loc[df["fq"].eq(4) & (df["revenue_q"] < 0), "revenue_q"] = np.nan
    return df
