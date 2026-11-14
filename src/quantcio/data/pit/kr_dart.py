"""DART receipt dates of periodic reports and as-first-reported XBRL values (keyless DART web, or OpenDART API).

Ported from ``docs/sources/pit/reference/dartpub.py``, ``xbrl_web.py`` and the verifier-fixed
``kr_pit_fundamentals.py`` (``docs/sources/pit/kr-fundamentals.json``):
- DART web ``detailSearch.ax`` per stock code (works for delisted codes). Ranges over 10 years silently return
  '조회 결과가 없습니다' -> fixed 5-year windows from 2014-01-01. Closed windows are cached forever, the open window
  once per day.
- dart.fss.or.kr blocked this IP for 45+ minutes after ~700 requests in 34 minutes at 0.5 s spacing (and the block
  also disables the production keyless disclosure path). Every DART web request therefore goes through a
  ``DartGate``: >= 6 s spacing (10/min), a per-build request budget, and a circuit breaker that stops all DART web
  traffic of the process for 45 minutes at the first connection error. Anything not fetched falls back to the
  conservative lag rule and is counted.
- With ``DART_API_KEY`` the receipt dates come from OpenDART ``list.json`` (pblntf_ty=A, last_reprt_at=N) and the
  original XBRL from ``fnlttXbrl.xml`` (opendart.fss.or.kr was never blocked).
- ``xbrl_key_items`` never mixes separate values into a consolidated filing (Hyundai FY2018 OP -593억 separate vs
  2.42조 consolidated) and skips typed-dimension (segment) contexts.
- ``next_session_after`` is NaT-safe and refuses dates before the session calendar (no silent misplacement).
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quantcio.data.kr import parse_corp_code_zip, parse_dart_search, parse_dart_viewer_nodes
from quantcio.data.pit.kr_bulk import KEY_ITEMS

log = logging.getLogger(__name__)

DART_WEB = "https://dart.fss.or.kr"
OPENDART_API = "https://opendart.fss.or.kr/api"
PERIODIC_TYPES = ["A001", "A002", "A003"]  # 사업 / 반기 / 분기보고서
WINDOW_ORIGIN = date(2014, 1, 1)
WINDOW_YEARS = 5
DART_WEB_MIN_INTERVAL_S = 6.0  # <= 10 requests / minute (the IP block came at ~20 / minute sustained)
DART_WEB_BLOCK_PAUSE_S = 45 * 60  # measured block duration: 43+ minutes
_PERIOD_RE = re.compile(r"(사업보고서|반기보고서|분기보고서)\s*\((\d{4})\.(\d{2})\)")
BULK_TO_DART_KIND = {
    "1분기보고서": "분기보고서",
    "3분기보고서": "분기보고서",
    "반기보고서": "반기보고서",
    "사업보고서": "사업보고서",
}
REPRT_CODE = {"1분기보고서": "11013", "반기보고서": "11012", "3분기보고서": "11014", "사업보고서": "11011"}
_XBRLI = "http://www.xbrl.org/2003/instance"
_XBRLDI = "http://xbrl.org/2006/xbrldi"

_blocked_until = 0.0  # process-wide circuit breaker for dart.fss.or.kr (monotonic clock)


@dataclass
class DartGate:
    """Throttle + budget + circuit breaker for dart.fss.or.kr requests."""

    max_requests: int
    min_interval: float = DART_WEB_MIN_INTERVAL_S
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    used: int = 0
    refused: int = 0
    blocked_error: str = ""
    _last: float = field(default=-1e18, repr=False)

    @property
    def blocked(self) -> bool:
        return bool(self.blocked_error) or self.clock() < _blocked_until

    def acquire(self) -> bool:
        if self.blocked or self.used >= self.max_requests:
            self.refused += 1
            return False
        wait = self._last + self.min_interval - self.clock()
        if wait > 0:
            self.sleep(wait)
        self._last = self.clock()
        self.used += 1
        return True

    def trip(self, exc: BaseException) -> None:
        global _blocked_until
        self.blocked_error = f"{type(exc).__name__}: {str(exc)[:120]}"
        _blocked_until = self.clock() + DART_WEB_BLOCK_PAUSE_S
        log.warning("DART web circuit breaker tripped (%s) - no DART web requests for 45 min", self.blocked_error)


def reset_circuit_breaker() -> None:
    global _blocked_until
    _blocked_until = 0.0


# ------------------------------------------------------------------------------------------ search windows


def search_windows(
    today: date, origin: date = WINDOW_ORIGIN, years: int = WINDOW_YEARS
) -> list[tuple[date, date, bool]]:
    """Fixed <= 5-year windows [start, end] from `origin` (Feb-29 safe); the last one ends today (open)."""
    out = []
    s = origin
    while s <= today:
        block_end = (pd.Timestamp(s) + pd.DateOffset(years=years) - pd.Timedelta(1, "D")).date()
        e = min(block_end, today)
        out.append((s, e, block_end < today))
        s = (pd.Timestamp(block_end) + pd.Timedelta(1, "D")).date()
    return out


def _is_conn_error(exc: BaseException) -> bool:
    return isinstance(exc, requests.ConnectionError | requests.Timeout | ConnectionError)


def dart_web_periodic(
    session: Any, gate: DartGate, code: str, ws: date, we: date, max_pages: int = 20
) -> list[dict[str, Any]] | None:
    """Every periodic report (originals AND corrections) filed by `code` in [ws, we] via DART web search.
    None when the gate refuses (budget exhausted or blocked) - the caller must treat the dates as unknown."""
    rows: list[dict[str, Any]] = []
    page, n_pages = 1, 1
    while page <= min(n_pages, max_pages):
        if not gate.acquire():
            return None
        form = {
            "currentPage": str(page),
            "maxResults": "100",
            "maxLinks": "10",
            "sort": "date",
            "series": "desc",
            "startDate": ws.strftime("%Y%m%d"),
            "endDate": we.strftime("%Y%m%d"),
            "option": "corp",
            "textCrpNm": code,
            "textCrpCik": "",
            "publicType": PERIODIC_TYPES,
        }
        try:
            r = session.post(
                f"{DART_WEB}/dsab007/detailSearch.ax",
                data=form,
                headers={"Referer": f"{DART_WEB}/dsab007/main.do", "X-Requested-With": "XMLHttpRequest"},
                timeout=30,
            )
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            if _is_conn_error(exc):
                gate.trip(exc)
                return None
            raise
        part, n_pages = parse_dart_search(r.text)
        if page == 1 and not part and "조회 결과가 없습니다" not in r.text:
            raise ValueError(f"DART 검색 {code}: 결과 페이지 파싱 실패")
        rows += part
        page += 1
    if n_pages > max_pages:
        raise RuntimeError(f"DART 검색 {code} {ws}~{we}: {max_pages}/{n_pages}페이지에서 잘림")
    return list({r["rcept_no"]: r for r in rows}.values())


def opendart_json(session: Any, key: str, api: str, **params: Any) -> dict[str, Any]:
    """GET https://opendart.fss.or.kr/api/<api>.json. No key -> HTML page; bad key -> status 010; 013 = no data."""
    r = session.get(f"{OPENDART_API}/{api}.json", params={"crtfc_key": key, **params}, timeout=30)
    r.raise_for_status()
    if "json" not in r.headers.get("Content-Type", ""):
        raise RuntimeError(f"OpenDART {api}: JSON 아님 (crtfc_key 누락/오류)")
    j = r.json()
    if j.get("status") not in ("000", "013"):
        raise RuntimeError(f"OpenDART {api}: {j.get('status')} {j.get('message')}")
    return j


def opendart_corp_codes(session: Any, key: str) -> dict[str, str]:
    r = session.get(f"{OPENDART_API}/corpCode.xml", params={"crtfc_key": key}, timeout=60)
    r.raise_for_status()
    if r.content[:2] != b"PK":
        raise RuntimeError("OpenDART corpCode.xml: zip 아님 (crtfc_key 확인)")
    return parse_corp_code_zip(r.content)


def opendart_periodic(session: Any, key: str, corp_code: str, start: date, end: date) -> list[dict[str, Any]]:
    """Periodic reports (pblntf_ty=A, originals and corrections) of one company via OpenDART list.json."""
    rows: dict[str, dict[str, Any]] = {}
    page = 1
    while True:
        j = opendart_json(
            session,
            key,
            "list",
            corp_code=corp_code,
            bgn_de=start.strftime("%Y%m%d"),
            end_de=end.strftime("%Y%m%d"),
            last_reprt_at="N",
            pblntf_ty="A",
            page_no=page,
            page_count=100,
        )
        for x in j.get("list") or []:
            rcp = str(x.get("rcept_no") or "")
            try:
                dt = datetime.strptime(str(x.get("rcept_dt")), "%Y%m%d").date()
            except ValueError:
                continue
            rows[rcp] = {
                "rcept_dt": dt,
                "rcept_no": rcp,
                "report_nm": re.sub(r"\s+", " ", str(x.get("report_nm") or "")),
            }
        if page >= int(j.get("total_page") or 1):
            break
        page += 1
    return list(rows.values())


# ------------------------------------------------------------------------------------- publication table


def is_numeric_correction(report_nm: str) -> bool:
    """'[기재정정]' changes the filed statements; '[첨부정정]' / '[첨부추가]' only replace attachments."""
    m = re.match(r"\s*\[([^\]]*)\]", report_nm or "")
    return bool(m and "기재정정" in m.group(1))


def publication_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """DART rows -> one row per (kind, period_end): first_rcept_dt / first_rcept_no (earliest filing: the original,
    or the earliest correction when the original is outside the search), orig_rcept_dt (first non-correction),
    last_rcept_dt / last_rcept_no (latest version with changed statements = what the bulk file carries),
    n_filings, n_numeric_corrections."""
    cols = [
        "kind",
        "period_end",
        "first_rcept_dt",
        "first_rcept_no",
        "orig_rcept_dt",
        "last_rcept_dt",
        "last_rcept_no",
        "n_filings",
        "n_numeric_corrections",
    ]
    recs = []
    for r in rows:
        m = _PERIOD_RE.search(r.get("report_nm") or "")
        if not m:
            continue
        nm = r["report_nm"]
        recs.append(
            {
                "kind": m.group(1),
                "period_end": pd.Timestamp(int(m.group(2)), int(m.group(3)), 1) + pd.offsets.MonthEnd(0),
                "rcept_dt": pd.Timestamp(r["rcept_dt"]),
                "rcept_no": str(r["rcept_no"]),
                "is_correction": nm.lstrip().startswith("["),
                "numeric": (not nm.lstrip().startswith("[")) or is_numeric_correction(nm),
            }
        )
    if not recs:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(recs).sort_values(["rcept_dt", "rcept_no"])
    g = df.groupby(["kind", "period_end"])
    out = g.agg(
        first_rcept_dt=("rcept_dt", "first"),
        first_rcept_no=("rcept_no", "first"),
        n_filings=("rcept_no", "size"),
    )
    num = df[df["numeric"]].groupby(["kind", "period_end"])
    out["last_rcept_dt"] = num["rcept_dt"].last()
    out["last_rcept_no"] = num["rcept_no"].last()
    out["last_rcept_dt"] = out["last_rcept_dt"].fillna(out["first_rcept_dt"])
    out["last_rcept_no"] = out["last_rcept_no"].fillna(out["first_rcept_no"])
    out["orig_rcept_dt"] = df[~df["is_correction"]].groupby(["kind", "period_end"])["rcept_dt"].first()
    out["n_numeric_corrections"] = df[df["is_correction"] & df["numeric"]].groupby(["kind", "period_end"]).size()
    out["n_numeric_corrections"] = out["n_numeric_corrections"].fillna(0).astype(int)
    return out.reset_index()[cols]


# ------------------------------------------------------------------------------------------------ lag rule


def next_session_after(d: Any, sessions: pd.DatetimeIndex) -> pd.Timestamp:
    """First KRX session strictly after calendar date d (filings land up to 18:00, after the 15:30 close);
    NaT if d is missing or after the calendar. Raises for dates before the calendar (never misplaced silently)."""
    if d is None or pd.isna(d):
        return pd.NaT
    d = pd.Timestamp(d).normalize()
    if len(sessions) == 0 or d < sessions[0] - pd.Timedelta(7, "D"):
        start = sessions[0].date() if len(sessions) else None
        raise ValueError(f"거래일 달력 시작({start}) 이전 날짜 {d.date()}는 배치할 수 없음")
    i = sessions.searchsorted(d, side="right")
    return sessions[i] if i < len(sessions) else pd.NaT


def extend_sessions(sessions: pd.DatetimeIndex, until: pd.Timestamp) -> tuple[pd.DatetimeIndex, pd.Timestamp]:
    """Append weekdays after the last known session up to `until` (filings after the marcap data). A weekday
    estimate is never LATER than the true next session, and the gap days are not sessions, so no session t sees a
    value earlier than it was public. Returns (sessions, last real session)."""
    last = sessions[-1]
    extra = pd.bdate_range(last + pd.Timedelta(1, "D"), max(until, last + pd.Timedelta(1, "D")))
    return sessions.append(extra), last


def lag_rule_available_from(
    period_end: pd.Timestamp, kind: str, sessions: pd.DatetimeIndex, quarter_days: int, annual_days: int
) -> pd.Timestamp:
    """Unknown receipt date: period_end + 60 (quarter / half) or + 120 (annual) days, rolled forward to a session,
    then the NEXT session (0 of 2,460 periods too early in 2014-2026; +45/+90 was early 1.8-3.6%)."""
    days = annual_days if kind == "사업보고서" else quarter_days
    d = pd.Timestamp(period_end) + pd.Timedelta(days, "D")
    i = sessions.searchsorted(d)
    d = sessions[i] if i < len(sessions) else d
    return next_session_after(d, sessions)


# ------------------------------------------------------------------------------ per-filing XBRL (keyless)


def dart_filing_xbrl(session: Any, gate: DartGate, rcept_no: str) -> bytes | None | str:
    """Original XBRL zip of one filing via DART web (2 requests: main.do for dcm_no, then ifrs.do with the
    download-menu Referer - without it DART returns HTTP 200 + 0 bytes). Returns None when the filing has no XBRL
    (financials before 2023Q3, PDF-only filers) and 'refused' when the gate refuses."""
    if not gate.acquire():
        return "refused"
    try:
        r = session.get(f"{DART_WEB}/dsaf001/main.do", params={"rcpNo": rcept_no}, timeout=30)
        r.raise_for_status()
        nodes = parse_dart_viewer_nodes(r.text)
        if not nodes:
            raise ValueError(f"DART main.do {rcept_no}: 문서 노드 없음")
        dcm_no = nodes[0][1]
        if not gate.acquire():
            return "refused"
        ref = f"{DART_WEB}/pdf/download/main.do?rcp_no={rcept_no}&dcm_no={dcm_no}"
        r = session.get(
            f"{DART_WEB}/pdf/download/ifrs.do",
            params={"rcp_no": rcept_no, "dcm_no": dcm_no, "lang": "ko"},
            headers={"Referer": ref},
            timeout=60,
        )
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        if _is_conn_error(exc):
            gate.trip(exc)
            return "refused"
        raise
    return r.content if r.content[:2] == b"PK" else None


def opendart_xbrl(session: Any, key: str, rcept_no: str, reprt_code: str) -> bytes | None:
    """fnlttXbrl.xml: the XBRL zip of ONE filing (use the ORIGINAL rcept_no for as-first-reported values)."""
    r = session.get(
        f"{OPENDART_API}/fnlttXbrl.xml",
        params={"crtfc_key": key, "rcept_no": rcept_no, "reprt_code": reprt_code},
        timeout=60,
    )
    r.raise_for_status()
    if r.content[:2] == b"PK":
        return r.content
    m = re.search(rb"<status>(\d+)</status>\s*<message>(.*?)</message>", r.content, re.S)
    if m and m.group(1) in (b"013", b"014"):
        return None
    raise RuntimeError(
        f"fnlttXbrl {rcept_no}: {m.group(1).decode()} {m.group(2).decode('utf-8', 'replace')}" if m else "zip 아님"
    )


def xbrl_key_items(zip_bytes: bytes, period_end: Any) -> dict[str, Any]:
    """XBRL instance -> KEY_ITEMS for the current period: flows = the longest duration ending at period_end (FY or
    YTD), stocks at the period-end instant. Consolidated facts only when the filing has any; separate facts are
    used only by separate-only filers ('_fs' = 'CFS' / 'OFS'). '_currency' = ISO code of the monetary unit."""
    from lxml import etree

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        inst = [n for n in z.namelist() if n.lower().endswith(".xbrl")]
        if not inst:
            raise ValueError("XBRL zip에 .xbrl 인스턴스 없음")
        # recover=True: some filers declare invalid namespace URIs (xmlns:XBRLConsultingFirm='자체작성')
        root = etree.fromstring(z.read(inst[0]), parser=etree.XMLParser(recover=True, huge_tree=True))
    ctx: dict[str, tuple] = {}
    for c in root.iter(f"{{{_XBRLI}}}context"):
        per = c.find(f"{{{_XBRLI}}}period")
        if per is None:
            continue
        dims = [(m.text or "").strip() for m in c.iter(f"{{{_XBRLDI}}}explicitMember")]
        dims += ["typed:" + (m.get("dimension") or "") for m in c.iter(f"{{{_XBRLDI}}}typedMember")]
        ctx[c.get("id")] = (
            per.findtext(f"{{{_XBRLI}}}startDate"),
            per.findtext(f"{{{_XBRLI}}}endDate"),
            per.findtext(f"{{{_XBRLI}}}instant"),
            dims,
        )
    units: dict[str, str] = {}
    for u in root.iter(f"{{{_XBRLI}}}unit"):
        meas = u.findtext(f"{{{_XBRLI}}}measure") or ""
        units[u.get("id")] = meas.split(":")[-1].strip().upper()
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
            best[item] = (rank, v, units.get(el.get("unitRef") or "", ""))
    filing_cons = any(rank[0] == 1 for rank, _, _ in best.values())
    out: dict[str, Any] = {k: v for k, (rank, v, _) in best.items() if rank[0] == 1 or not filing_cons}
    out["_fs"] = "CFS" if filing_cons else "OFS"
    for k in ("revenue", "equity", "assets", "ni"):
        if k in best and best[k][2]:
            out["_currency"] = best[k][2]
            break
    return out


# -------------------------------------------------------------------------------------------- disk cache


def _json_read(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{time.monotonic_ns()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _rows_to_json(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"rcept_dt": str(r["rcept_dt"]), "rcept_no": r["rcept_no"], "report_nm": r.get("report_nm", "")} for r in rows
    ]


def _rows_from_json(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**r, "rcept_dt": date.fromisoformat(str(r["rcept_dt"])[:10])} for r in rows]


def cached_web_filings(
    root: Path, session: Any, gate: DartGate, code: str, today: date
) -> tuple[list[dict[str, Any]] | None, int]:
    """All periodic filings of `code` since 2014 via DART web, window by window through the disk cache
    (root/<code>/<start>_<end>.json). Returns (rows or None when any window is unavailable, network windows)."""
    rows: list[dict[str, Any]] = []
    fetched = 0
    for ws, we, closed in search_windows(today):
        path = root / code / f"{ws:%Y%m%d}_{we:%Y%m%d}.json"
        hit = _json_read(path)
        if hit is None:
            part = dart_web_periodic(session, gate, code, ws, we)
            if part is None:
                return None, fetched
            fetched += 1
            _json_write(path, _rows_to_json(part))
            if not closed:  # older copies of the open window
                for old in path.parent.glob(f"{ws:%Y%m%d}_*.json"):
                    if old != path:
                        old.unlink(missing_ok=True)
            hit = _rows_to_json(part)
        rows += _rows_from_json(hit)
    return rows, fetched
