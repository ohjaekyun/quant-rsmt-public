"""Keyless OpenDART bulk financial statements (재무정보 일괄다운로드), 2015FY..latest, all listed XBRL filers.

Probe version: uses pcache (disk cache + throttle). The production version is in the StructuredOutput.
"""
from __future__ import annotations

import csv
import io
import re
import zipfile

import pandas as pd

from pcache import get, post

OPENDART_WEB = "https://opendart.fss.or.kr"
PERIOD_Q = {"FQ": 1, "HY": 2, "TQ": 3, "FY": 4}
META_COLS = ["재무제표종류", "종목코드", "회사명", "시장구분", "업종", "업종명", "결산월", "결산기준일", "보고서종류",
             "통화", "항목코드", "항목명"]


def bulk_file_list() -> pd.DataFrame:
    """-> DataFrame[bsn_yr:int, period:'FQ|HY|TQ|FY', stmt:'BS|PL|CF|CE', fl_nm:str, built:Timestamp]."""
    st, h, b = post(f"{OPENDART_WEB}/disclosureinfo/fnltt/dwld/list.do", data={},
                    headers={"X-Requested-With": "XMLHttpRequest",
                             "Referer": f"{OPENDART_WEB}/disclosureinfo/fnltt/dwld/main.do"}, ttl_h=24)
    rows = re.findall(r"download_ext002\('(\d{4})',\s*'(\w+)',\s*'(\w+)',\s*'([^']+\.zip)'\)", b.decode("utf-8"))
    df = pd.DataFrame(rows, columns=["bsn_yr", "period", "stmt", "fl_nm"])
    df["bsn_yr"] = df["bsn_yr"].astype(int)
    df["built"] = pd.to_datetime(df["fl_nm"].str.extract(r"_(\d{14})\.zip$")[0], format="%Y%m%d%H%M%S")
    return df


def _num(s: str) -> float | None:
    s = (s or "").strip().replace(",", "")
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_bulk_zip(fl_nm: str) -> pd.DataFrame:
    """One bulk zip -> long frame [member, 재무제표종류, stock_code, 회사명, 시장구분, 결산월, period_end, 보고서종류,
    통화, 항목코드, 항목명, col, value]. Member names are cp949 (zip utf-8 flag unset); text is cp949 TSV whose
    value-column headers differ by report ('당기 1분기 3개월', '당기 1분기 누적', '당기', '전기', ...)."""
    st, h, b = get(f"{OPENDART_WEB}/cmm/downloadFnlttZip.do", params={"fl_nm": fl_nm}, timeout=300)
    if st != 200 or b[:2] != b"PK":
        raise RuntimeError(f"bulk {fl_nm}: HTTP {st} {h.get('Content-Type')} {b[:80]!r}")
    out = []
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        for info in z.infolist():
            name = info.filename if info.flag_bits & 0x800 else info.filename.encode("cp437").decode("cp949")
            text = z.read(info).decode("cp949", errors="replace")
            rdr = csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE)
            header = [c.strip() for c in next(rdr)]
            # value headers are interleaved with EMPTY separator columns ('', '당기', '', '', '전기', '전전기', ''):
            # keep their positions, never compact the list
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
            df = pd.DataFrame(recs, columns=[*META_COLS, "col", "value"])
            df.insert(0, "member", name)
            out.append(df)
    df = pd.concat(out, ignore_index=True)
    df["stock_code"] = df["종목코드"].str.strip("[]")
    df["period_end"] = pd.to_datetime(df["결산기준일"], errors="coerce")
    df["consolidated"] = df["member"].str.contains("_연결_")
    return df.drop(columns=["종목코드", "결산기준일", "업종", "업종명"])
