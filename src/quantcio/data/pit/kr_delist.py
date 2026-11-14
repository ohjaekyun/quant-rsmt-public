"""KR delisting master (keyless): FDR GitHub cache of KRX MDCSTAT23801 + KIND 상장폐지현황, with reason categories.

Ported from ``docs/sources/pit/reference/kr_delist.py`` with the verifier's fixes
(``docs/sources/pit/kr-universe.json``):
- ``fdr.StockListing('KRX-DELISTING')`` silently returns an EMPTY frame when its GitHub cache lags KRX, so the newest
  CSV is read directly from the GitHub directory listing.
- The FDR table covers ALL securities (수익증권, 선박투자회사, 부동산투자회사, 투자회사, 외국주권, 주식예탁증권 ...):
  its non-주권 6-character symbols are the ``nonstock_codes`` excluded from the universe, and KIND rows are
  de-duplicated against ALL groups (otherwise ship funds / REITs leak in as commons).
- KIND lists market transfers (KOSDAQ->KOSPI, KONEX->KOSDAQ) as delistings (category 'transfer': not an exit) and
  keeps the first delisting of relisted issuers that FDR drops. Always send both fromDate and toDate.
- is_spac also catches SPACs whose names lack '스팩' ('\\d+호$', category spac_expiry).
- Delisting date / reason are look-ahead information: they are metadata only. The only PIT-safe field is the
  정리매매 window (ArrantEnforceDate..ArrantEndDate), used as an eligibility filter from its start date.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from io import StringIO
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

FDR_CACHE_API = "https://api.github.com/repos/FinanceData/fdr_krx_data_cache/contents/data/listing/delisting"
FDR_CACHE_RAW = (
    "https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache/refs/heads/master/data/listing/delisting/{}"
)
KIND_DEL = "https://kind.krx.co.kr/investwarn/delcompany.do"
KIND_MARKETS = (("1", "KOSPI"), ("2", "KOSDAQ"), ("6", "KONEX"))

_REASON_RULES = [  # order matters: first match wins
    ("transfer", r"이전상장|유가증권시장 상장|코스닥시장 상장"),
    ("spac_merger", r"스팩소멸합병"),
    (
        "spac_expiry",
        r"상장예비심사\s*청구서\s*미제출|상장예비심사신청서 미제출|합병상장예비심사|SPAC 상장예비심사"
        r"|상장예비심사청구서 미제출",
    ),
    ("bankruptcy", r"파산|부도|어음 또는 수표"),
    ("merger", r"피흡수합병|해산\s*사유\s*발생|해산사유 발생"),
    ("squeeze_out", r"완전자회사|주식교환"),
    ("voluntary", r"신청에 의한 상장폐지|상장폐지\s*신청"),
    ("audit_opinion", r"감사의견|감사범위|한정의견|의견거절"),
    ("capital_impair", r"자본전액잠식|자본잠식"),
    ("governance", r"기업의 계속성|경영의 투명성"),
    ("filing_failure", r"사업보고서|반기보고서|분기보고서|공시서류 미제출"),
    ("konex_adviser", r"지정자문인"),
    ("pref_thresholds", r"종류주식"),
    ("thresholds", r"시가총액|분산요건|주식분포|거래량|영업손실|매출액"),
    ("term_expiry", r"존속기간 만료"),
]
CATEGORY_LABELS = {
    "transfer": "시장 이전",
    "spac_merger": "스팩 소멸합병",
    "spac_expiry": "스팩 기한 만료",
    "bankruptcy": "부도·파산",
    "merger": "피흡수합병·해산",
    "squeeze_out": "완전자회사화·주식교환",
    "voluntary": "자진 상장폐지",
    "audit_opinion": "감사의견 거절·한정",
    "capital_impair": "자본잠식",
    "governance": "상장적격성 실질심사",
    "filing_failure": "공시서류 미제출",
    "konex_adviser": "지정자문인 계약 해지",
    "pref_thresholds": "종류주식 요건 미달",
    "thresholds": "시가총액·분산 등 요건 미달",
    "term_expiry": "존속기간 만료",
    "other": "기타",
}
MASTER_COLUMNS = [
    "code",
    "name",
    "market",
    "kind",
    "listing_date",
    "delist_date",
    "reason",
    "src",
    "category",
    "is_spac",
]
_SPAC_NAME_RE = r"스팩|기업인수목적|SPAC|\d+호$"


def delist_category(reason: Any) -> str:
    s = str(reason or "")
    for cat, rx in _REASON_RULES:
        if re.search(rx, s):
            return cat
    return "other"


def fetch_fdr_delisting(session: Any) -> tuple[pd.DataFrame, str]:
    """Newest FDR GitHub-cached KRX delisting table (all securities since 1956) -> (frame, 'YYYY-MM-DD' as-of)."""
    r = session.get(FDR_CACHE_API, timeout=30, headers={"Accept": "application/vnd.github+json"})
    r.raise_for_status()
    names = sorted(x["name"] for x in r.json() if str(x.get("name", "")).endswith(".csv"))
    if not names:
        raise ValueError("FDR 상장폐지 캐시 디렉터리에 CSV 없음")
    name = names[-1]
    r2 = session.get(FDR_CACHE_RAW.format(name), timeout=60)
    r2.raise_for_status()
    return parse_fdr_delisting(r2.content.decode("utf-8-sig", errors="replace")), name.removesuffix(".csv")


def parse_fdr_delisting(text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(text), index_col=0, dtype={"Symbol": str, "ToSymbol": str})
    for c in ("ListingDate", "DelistingDate", "ArrantEnforceDate", "ArrantEndDate"):
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df.reset_index(drop=True)


def fetch_kind_delisted(session: Any, from_date: date, to_date: date, market_type: str = "") -> pd.DataFrame:
    """KIND 상장폐지현황 export (EUC-KR HTML served as .xls). market_type ''=all, '1' KOSPI, '2' KOSDAQ, '6' KONEX."""
    data = {
        "method": "searchDelCompanySub",
        "forward": "delcompany_down",
        "currentPageSize": "3000",
        "pageIndex": "1",
        "orderMode": "2",
        "orderStat": "D",
        "marketType": market_type,
        "fromDate": from_date.isoformat(),
        "toDate": to_date.isoformat(),  # an empty toDate makes KIND ignore fromDate too
        "tabType": "1",
    }
    r = session.post(KIND_DEL, data=data, timeout=60, headers={"Referer": KIND_DEL + "?method=searchDelCompanyMain"})
    r.raise_for_status()
    return parse_kind_delisted(r.content)


def parse_kind_delisted(content: bytes) -> pd.DataFrame:
    text = content.decode("euc-kr", errors="replace")
    if "<table" not in text.lower():
        return pd.DataFrame(columns=["code", "name", "delist_date", "reason", "note"])
    t = pd.read_html(StringIO(text), converters={"종목코드": str})[0]
    t = t.rename(
        columns={"회사명": "name", "종목코드": "code", "폐지일자": "delist_date", "폐지사유": "reason", "비고": "note"}
    )
    t["code"] = t["code"].astype(str).str.zfill(6)
    t["delist_date"] = pd.to_datetime(t["delist_date"], errors="coerce")
    for c in ("reason", "note"):
        if c not in t:
            t[c] = ""
    return t[["code", "name", "delist_date", "reason", "note"]]


def build_master(fdr_all: pd.DataFrame | None, kind: pd.DataFrame | None, since: date) -> pd.DataFrame:
    """Union of FDR/KRX 주권 delistings (incl. preferreds) and KIND-only issuer rows (transfers, first delistings of
    relisted codes, rows newer than the FDR cache) -> MASTER_COLUMNS."""
    parts = []
    key: set[tuple[str, pd.Timestamp]] = set()
    if fdr_all is not None and len(fdr_all):
        f_all = fdr_all[fdr_all["DelistingDate"] >= pd.Timestamp(since)]
        key = set(zip(f_all["Symbol"], f_all["DelistingDate"], strict=True))
        f = f_all[f_all["SecuGroup"] == "주권"]
        parts.append(
            pd.DataFrame(
                {
                    "code": f["Symbol"],
                    "name": f["Name"],
                    "market": f["Market"],
                    "kind": f["Kind"],
                    "listing_date": f["ListingDate"],
                    "delist_date": f["DelistingDate"],
                    "reason": f["Reason"],
                    "src": "fdr_krx",
                }
            )
        )
    if kind is not None and len(kind):
        k = kind[kind["delist_date"] >= pd.Timestamp(since)]
        k = k[[(c, d) not in key for c, d in zip(k["code"], k["delist_date"], strict=True)]]
        parts.append(
            pd.DataFrame(
                {
                    "code": k["code"],
                    "name": k["name"],
                    "market": k.get("market", pd.Series("", index=k.index)),
                    "kind": "보통주",
                    "listing_date": pd.NaT,
                    "delist_date": k["delist_date"],
                    "reason": k["reason"],
                    "src": "kind_only",
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=MASTER_COLUMNS)
    m = pd.concat(parts, ignore_index=True)
    m["category"] = m["reason"].map(delist_category)
    m["is_spac"] = m["name"].astype(str).str.contains(_SPAC_NAME_RE, regex=True, na=False) | m["category"].eq(
        "spac_expiry"
    )
    return m.sort_values(["delist_date", "code"]).reset_index(drop=True)[MASTER_COLUMNS]


@dataclass
class DelistInfo:
    master: pd.DataFrame  # MASTER_COLUMNS
    nonstock_codes: frozenset[str]  # 6-char symbols of non-주권 securities (ship funds, REITs, 외국주권, KDR ...)
    arrant: pd.DataFrame  # code, start, end: 정리매매 windows (PIT-safe from their start)
    version: str  # cache/identity key: FDR as-of + KIND fetch date
    notes: list[str] = field(default_factory=list)
    fetched_at: str = ""


def empty_delist_info(note: str) -> DelistInfo:
    return DelistInfo(
        master=pd.DataFrame(columns=MASTER_COLUMNS),
        nonstock_codes=frozenset(),
        arrant=pd.DataFrame(columns=["code", "start", "end"]),
        version="none",
        notes=[note],
    )


def delist_info_from_parts(
    fdr_all: pd.DataFrame | None,
    fdr_asof: str,
    kind: pd.DataFrame | None,
    since: date,
    fetched: date,
    notes: list[str] | None = None,
) -> DelistInfo:
    nonstock: frozenset[str] = frozenset()
    arrant = pd.DataFrame(columns=["code", "start", "end"])
    if fdr_all is not None and len(fdr_all):
        sym = fdr_all["Symbol"].astype(str)
        nonstock = frozenset(sym[(fdr_all["SecuGroup"] != "주권") & (sym.str.len() == 6)])
        a = fdr_all[fdr_all["ArrantEnforceDate"].notna()]
        arrant = pd.DataFrame(
            {"code": a["Symbol"].astype(str), "start": a["ArrantEnforceDate"], "end": a["ArrantEndDate"]}
        )
    master = build_master(fdr_all, kind, since)
    return DelistInfo(
        master=master,
        nonstock_codes=nonstock,
        arrant=arrant.reset_index(drop=True),
        version=f"fdr:{fdr_asof or 'none'}|kind:{fetched.isoformat() if kind is not None else 'none'}",
        notes=list(notes or []),
        fetched_at=fetched.isoformat(),
    )


def fetch_delist_info(session: Any, since: date, today: date, pause: float = 0.5) -> DelistInfo:
    """Live fetch (1 GitHub API call, 1 CSV, 3 KIND POSTs). Partial failures are noted, never raised."""
    notes: list[str] = []
    fdr_all, asof = None, ""
    try:
        fdr_all, asof = fetch_fdr_delisting(session)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"FDR 상장폐지 목록 수집 실패({type(exc).__name__}) → 비주권 코드 제외·정리매매 필터 없이 진행")
        log.warning("FDR delisting fetch failed: %s", exc)
    kinds = []
    for mt, label in KIND_MARKETS:
        try:
            t = fetch_kind_delisted(session, since, today, mt)
            kinds.append(t.assign(market=label))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"KIND 상장폐지현황({label}) 수집 실패({type(exc).__name__})")
            log.warning("KIND delisted %s failed: %s", label, exc)
        if pause:
            time.sleep(pause)
    kind = pd.concat(kinds, ignore_index=True) if kinds else None
    if fdr_all is None and kind is None:
        raise RuntimeError("상장폐지 목록(FDR·KIND) 모두 수집 실패")
    return delist_info_from_parts(fdr_all, asof, kind, since, today, notes)


def match_delisting(master: pd.DataFrame, code: str, last_row: pd.Timestamp, window_days: int = 45) -> dict | None:
    """Master row for a spell of `code` whose last marcap row is `last_row` (KRX delists the session after the last
    row; KIND/FDR dates are calendar dates). Transfers are not exits. None when nothing matches."""
    if master is None or master.empty:
        return None
    m = master[(master["code"] == code) & (master["category"] != "transfer")]
    if m.empty:
        return None
    lo, hi = pd.Timestamp(last_row), pd.Timestamp(last_row) + pd.Timedelta(window_days, "D")
    m = m[(m["delist_date"] > lo) & (m["delist_date"] <= hi)]
    if m.empty:
        return None
    return m.sort_values("delist_date").iloc[0].to_dict()
