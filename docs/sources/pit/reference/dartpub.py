"""Keyless DART web search -> publication dates of periodic reports and earnings releases (probe version)."""
from __future__ import annotations

import re
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, "src")
from quantcio.data.kr import normalise_report_name, parse_dart_search  # noqa: E402  (read-only reuse)

from pcache import post  # noqa: E402

DART_WEB = "https://dart.fss.or.kr"
PERIODIC_TYPES = ["A001", "A002", "A003"]           # 사업 / 반기 / 분기보고서
RELEASE_TYPES = ["I001", "I002"]                     # 수시공시 (손익구조변동, 감사보고서제출) / 공정공시 (영업(잠정)실적)
_PERIOD_RE = re.compile(r"(사업보고서|반기보고서|분기보고서)\s*\((\d{4})\.(\d{2})\)")
_RELEASE_RE = re.compile(r"영업\(잠정\)실적|매출액또는손익구조")
_AUDIT_RE = re.compile(r"감사보고서제출")


def dart_web_rows(code: str, start: date, end: date, types: list[str], max_pages: int = 20) -> list[dict]:
    """All DART web filings of a stock code in [start, end] (both FSS and KRX), paginated, deduped by rcept_no."""
    rows, page, n_pages = [], 1, 1
    while page <= min(n_pages, max_pages):
        form = {"currentPage": str(page), "maxResults": "100", "maxLinks": "10", "sort": "date", "series": "desc",
                "startDate": start.strftime("%Y%m%d"), "endDate": end.strftime("%Y%m%d"), "option": "corp",
                "textCrpNm": code, "textCrpCik": "", "publicType": types}
        st, h, b = post(f"{DART_WEB}/dsab007/detailSearch.ax", data=form,
                        headers={"Referer": f"{DART_WEB}/dsab007/main.do", "X-Requested-With": "XMLHttpRequest"})
        text = b.decode("utf-8", "replace")
        if st != 200:
            raise RuntimeError(f"DART search {code} p{page}: HTTP {st}")
        part, n_pages = parse_dart_search(text)
        if page == 1 and not part and "조회 결과가 없습니다" not in text:
            raise ValueError(f"DART search {code}: unparsable page")
        rows += part
        page += 1
    if n_pages > max_pages:
        raise RuntimeError(f"DART search {code}: truncated {max_pages}/{n_pages} pages")
    return list({r["rcept_no"]: r for r in rows}.values())


def fiscal_period_of(report_nm: str) -> tuple[str, pd.Timestamp] | None:
    """'[기재정정]분기보고서 (2024.03)' -> ('분기보고서', 2024-03-31)."""
    m = _PERIOD_RE.search(report_nm or "")
    if not m:
        return None
    kind, y, mth = m.group(1), int(m.group(2)), int(m.group(3))
    return kind, pd.Timestamp(y, mth, 1) + pd.offsets.MonthEnd(0)


def periodic_publication_table(rows: list[dict]) -> pd.DataFrame:
    """Periodic reports -> one row per (kind, period_end): first (original) filing date, last correction date."""
    recs = []
    for r in rows:
        fp = fiscal_period_of(r["report_nm"])
        if not fp:
            continue
        recs.append({"kind": fp[0], "period_end": fp[1], "rcept_dt": pd.Timestamp(r["rcept_dt"]),
                     "rcept_no": r["rcept_no"], "is_correction": r["report_nm"].lstrip().startswith("["),
                     "report_nm": r["report_nm"]})
    if not recs:
        return pd.DataFrame(columns=["kind", "period_end", "first_rcept_dt", "first_rcept_no", "n_corrections",
                                     "last_correction_dt"])
    df = pd.DataFrame(recs)
    g = df.sort_values(["rcept_dt", "rcept_no"]).groupby(["kind", "period_end"])
    out = g.agg(first_rcept_dt=("rcept_dt", "first"), first_rcept_no=("rcept_no", "first"),
                n_filings=("rcept_no", "size"))
    orig = df[~df.is_correction].sort_values("rcept_dt").groupby(["kind", "period_end"])["rcept_dt"].first()
    corr = df[df.is_correction].groupby(["kind", "period_end"])["rcept_dt"].max()
    out["orig_rcept_dt"] = orig
    out["last_correction_dt"] = corr
    return out.reset_index()
