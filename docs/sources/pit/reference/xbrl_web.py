"""Keyless per-filing XBRL (DART web 'IFRS(원문XBRL)' download) -> tidy facts (probe version)."""
from __future__ import annotations

import io
import re
import sys
import zipfile

import pandas as pd
from lxml import etree

sys.path.insert(0, "src")
from quantcio.data.kr import parse_dart_viewer_nodes  # noqa: E402  (read-only reuse)

from pcache import get  # noqa: E402

DART_WEB = "https://dart.fss.or.kr"
XBRLI = "http://www.xbrl.org/2003/instance"
XBRLDI = "http://xbrl.org/2006/xbrldi"


def main_dcm_no(rcept_no: str) -> str:
    """Main document dcmNo of a filing (first TOC node of dsaf001/main.do)."""
    st, h, b = get(f"{DART_WEB}/dsaf001/main.do", params={"rcpNo": rcept_no})
    nodes = parse_dart_viewer_nodes(b.decode("utf-8", "replace"))
    if not nodes:
        raise ValueError(f"DART main.do {rcept_no}: no document nodes")
    return nodes[0][1]


def download_menu(rcept_no: str, dcm_no: str) -> list[str]:
    """Download kinds offered for a filing: subset of ['excel','ifrs','pdf'] (financial cos pre-2023: pdf only)."""
    st, h, b = get(f"{DART_WEB}/pdf/download/main.do", params={"rcp_no": rcept_no, "dcm_no": dcm_no})
    return sorted(set(re.findall(r"/pdf/download/([a-z]+)\.do", b.decode("utf-8", "replace"))))


def download_xbrl_zip(rcept_no: str, dcm_no: str) -> bytes:
    """Original XBRL zip. The ONLY requirement is Referer = the download-menu URL (else HTTP 200 with 0 bytes)."""
    ref = f"{DART_WEB}/pdf/download/main.do?rcp_no={rcept_no}&dcm_no={dcm_no}"
    st, h, b = get(f"{DART_WEB}/pdf/download/ifrs.do", params={"rcp_no": rcept_no, "dcm_no": dcm_no, "lang": "ko"},
                   headers={"Referer": ref}, timeout=60)
    if st != 200 or b[:2] != b"PK":
        raise ValueError(f"XBRL {rcept_no}/{dcm_no}: HTTP {st}, {len(b)} bytes (no XBRL for this filing?)")
    return b


def xbrl_facts(zip_bytes: bytes) -> pd.DataFrame:
    """XBRL instance -> [concept, prefix, value, unit, decimals, period_start, period_end, instant, dims, context]."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        inst = [n for n in z.namelist() if n.lower().endswith(".xbrl")]
        if not inst:
            raise ValueError("XBRL zip without .xbrl instance")
        root = etree.fromstring(z.read(inst[0]))
    ns_prefix = {v: k for k, v in root.nsmap.items() if k}
    ctx = {}
    for c in root.iter(f"{{{XBRLI}}}context"):
        per = c.find(f"{{{XBRLI}}}period")
        dims = tuple(sorted((m.get("dimension"), (m.text or "").strip()) for m in c.iter(f"{{{XBRLDI}}}explicitMember")))
        ctx[c.get("id")] = (per.findtext(f"{{{XBRLI}}}startDate"), per.findtext(f"{{{XBRLI}}}endDate"),
                            per.findtext(f"{{{XBRLI}}}instant"), dims)
    recs = []
    for el in root:
        if not isinstance(el.tag, str) or el.get("contextRef") is None:
            continue
        q = etree.QName(el)
        try:
            val = float((el.text or "").strip())
        except ValueError:
            continue
        s, e, i, dims = ctx.get(el.get("contextRef"), (None, None, None, ()))
        recs.append((q.localname, ns_prefix.get(q.namespace, q.namespace), val, el.get("unitRef"), el.get("decimals"),
                     s, e, i, dims, el.get("contextRef")))
    df = pd.DataFrame(recs, columns=["concept", "prefix", "value", "unit", "decimals", "period_start", "period_end",
                                     "instant", "dims", "context"])
    for c in ("period_start", "period_end", "instant"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df
