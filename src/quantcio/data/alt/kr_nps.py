"""National Pension Service workplace file (국민연금 가입 사업장 내역, data.go.kr dataset 15083277) as KR measures.

Source (verified 2026-10-02, no key needed):
- The dataset page ``https://www.data.go.kr/data/15083277/fileData.do`` carries the CURRENT monthly CSV
  (hidden input ``publicDataDetailPk``). Its tab "주기성 과거 데이터" is filled by JavaScript from
  ``POST /tcs/dss/selectHistAndCsvData.do`` (publicDataPk, publicDataDetailPk): one table row per past file with the
  title and the registration date (등록일), and a ``data-public-pk`` / ``data-public-detail-sn`` per file. The
  popup ``POST /tcs/dss/selectDpkDetailInfo.do`` (publicDataDetailPk, publicDataHistSn) of a file gives its
  ``atchFileId`` (``fn_fileDataDown(..., 'FILE_...', '1', 'csv')``), 등록일 and 수정일.
  Download: ``GET /cmm/cmm/fileDownload.do?atchFileId=..&fileDetailSn=1&insertDataPrcus=N`` (cp949 CSV, ~116 MB
  in 2026, ignores HTTP Range -> a broken download restarts from zero). The browser first asks
  ``POST /cmm/cmm/check-limit.json`` and shows a CAPTCHA when ``needCaptcha`` is true; the bulk loader asks the
  same and stops cleanly (resumable) instead of solving anything.
- 2026-10-02: 129 past files + the current one (130 files, ~80-116 MB each). The oldest file (titled '2015년 12월')
  holds data month (자료생성년월) 2015-11: titles name the release month, the data month is the month before.
  Every file titled 2015-12 .. 2019-07 was registered on 2019-08-05/06 in one batch (portal migration), so their
  registration date is NOT their original release date. Since 2019-08 a file is registered on the 17th-31st of
  the month after its data month (data month 2026-07 -> 2026-08-25, 2026-08 -> 2026-09-23); a few were modified
  later (수정일 up to ~3 weeks after 등록일). The 2015-11 file has 사업장형태구분코드 1 on every row (the filter
  removes nothing there; the 6-digit prefix already carries the 법인 type code 81/86/87/88).
- Columns: 자료생성년월 (data month), 사업장명, 사업자등록번호 (first 6 digits only), 사업장가입상태코드 (1 등록 /
  2 탈퇴), ..., 사업장형태구분코드 (1 법인 / 2 개인), 사업장업종코드, ..., 가입자수, 당월고지금액, 신규취득자수,
  상실가입자수. One row per NPS workplace unit; a corporation usually has one unit (삼성전자(주): 125,720 members in
  2026-07), sometimes several with the same registration-number prefix and name (summed here).

Point in time (``quantcio.data.alt`` rule: a row is usable at grid date t only when ``published < t``):
- ``period_end`` = last calendar day of the data month (자료생성년월 read from the file itself, never the title).
- ``published`` = the file's release date on data.go.kr = max(등록일, 수정일): the downloaded bytes may be the
  modified version, so the later date is the safe one.
- Backfilled files (registered more than ``RELEASE_MAX_LAG_DAYS`` after the end of their data month: the
  2019-08-05 batch) have no known original release date. Default ``backfill_rule="registration"`` keeps the
  registration date (the only date that is proven; the 2015-12 .. 2019-06 months are then history, never "latest",
  before 2019-08). ``backfill_rule="month_after"`` instead ASSUMES a timely original release and uses the
  conservative unknown-release date below; a note says so.
- Release date unknown (a file ingested without portal metadata): the last calendar day of the month AFTER the
  data month (data month 2026-08 -> 2026-09-30); recent files were registered on the 17th-31st of that month.

Matching listed companies (6-digit stock codes): stock code -> DART corp_code (``opendart_corp_codes``) ->
OpenDART ``company.json`` (bizr_no, corp_name, stock_name; cached on disk). An NPS row belongs to the entity when
its 6-digit 사업자등록번호 equals ``bizr_no[:6]`` AND the normalised names agree (``norm_company_name``): exact
first; only when no row of the prefix is an exact match, rows whose name contains the DART name or vice versa
(``contained``). Other rows of the same prefix (the prefix is tax office + type + 1 serial digit, so dozens of
unrelated companies share it) are excluded and counted. company.json carries the CURRENT names: months before a
rename only match through ``aliases`` and are otherwise counted as ``none``.

Storage: ``<cache>/alt/kr_nps/<YYYYMM>.parquet`` (법인 rows only, compact columns) + ``<YYYYMM>.json`` (file
metadata), ``_files.json`` (atchFileId -> data month, so a file is never downloaded twice), ``_listing.json``
(portal listing, refreshed after ``LISTING_TTL_HOURS``). Downloads are sequential with a politeness gap.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from quantcio.config import Settings
from quantcio.data.alt import AltData, empty_measures, normalise_measures
from quantcio.data.http import PoliteSession

log = logging.getLogger(__name__)

SOURCE = "kr_nps"
ITEMS = ("nps_members", "nps_new", "nps_lost")
DATASET_PK = "15083277"
PORTAL = "https://www.data.go.kr"
PAGE_URL = f"{PORTAL}/data/{DATASET_PK}/fileData.do"
HIST_URL = f"{PORTAL}/tcs/dss/selectHistAndCsvData.do"
DETAIL_URL = f"{PORTAL}/tcs/dss/selectDpkDetailInfo.do"
LIMIT_URL = f"{PORTAL}/cmm/cmm/check-limit.json"
DOWNLOAD_URL = f"{PORTAL}/cmm/cmm/fileDownload.do"
PORTAL_MIN_INTERVAL_S = 1.0  # data.go.kr is not in quantcio.data.http's table: own politeness gap
RELEASE_MAX_LAG_DAYS = 75  # registered later than data-month end + 75 days -> backfill, not the release
LISTING_TTL_HOURS = 24.0
COMPANY_TTL_DAYS = 90.0
CORP_CODES_TTL_DAYS = 7.0
MIN_CONTAINED_LEN = 2  # a normalised name shorter than this never matches by containment
BackfillRule = Literal["registration", "month_after"]
Progress = Callable[[int, int, str], None]

# source header (first token, descriptive suffixes such as "1 법인 2 개인" dropped) -> compact column
NPS_COLUMNS = {
    "자료생성년월": "data_month",
    "사업장명": "name",
    "사업자등록번호": "bizr6",
    "사업장가입상태코드": "status",
    "사업장형태구분코드": "form",
    "사업장업종코드": "industry",
    "가입자수": "members",
    "당월고지금액": "billed",
    "신규취득자수": "new",
    "상실가입자수": "lost",
}
COMPACT_COLUMNS = ["data_month", "name", "bizr6", "status", "industry", "members", "billed", "new", "lost"]
ITEM_COLUMN = {"nps_members": "members", "nps_new": "new", "nps_lost": "lost"}


class NpsDownloadLimitError(RuntimeError):
    """data.go.kr asks for a CAPTCHA (per-client download limit): stop, resume later."""


# --------------------------------------------------------------------------------------------- listing


@dataclass(frozen=True)
class NpsFile:
    """One monthly file offered by the portal. ``data_month`` is the month read from the file once ingested,
    else an estimate from the title / registration date (``month_estimated``)."""

    detail_pk: str
    hist_sn: str
    title: str
    registered: str  # YYYY-MM-DD (등록일)
    modified: str = ""  # YYYY-MM-DD (수정일), "" if unknown
    atch_file_id: str = ""
    file_sn: str = "1"  # fileDetailSn of the CSV
    data_month: str = ""  # YYYY-MM
    month_estimated: bool = True
    current: bool = False

    @property
    def url(self) -> str:
        return f"{DOWNLOAD_URL}?atchFileId={self.atch_file_id}&fileDetailSn={self.file_sn}&insertDataPrcus=N"

    @property
    def release_date(self) -> date | None:
        """max(등록일, 수정일) = the date the downloadable bytes were public (see module docstring)."""
        ds = [d for d in (self.registered, self.modified) if d]
        return max(date.fromisoformat(d) for d in ds) if ds else None

    @property
    def backfilled(self) -> bool:
        return _is_backfill(self.data_month, self.registered)


_TITLE_YM = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월")
_DETAIL_PK = re.compile(r'id="publicDataDetailPk"[^>]*value="([^"]+)"')
_DOWN_CALL = re.compile(
    r"fn_fileDataDown\(\s*'[^']*'\s*,\s*'([^']+)'\s*,\s*'(FILE_[0-9A-Za-z_]+)'\s*,\s*'(\d+)'\s*,\s*'csv'"
)


def month_end(data_month: str) -> pd.Timestamp:
    """'YYYY-MM' -> last calendar day of that month."""
    return pd.Period(data_month, "M").to_timestamp(how="end").normalize()


def conservative_published(data_month: str) -> pd.Timestamp:
    """Release date unknown: the last calendar day of the month AFTER the data month (2026-08 -> 2026-09-30)."""
    return (pd.Period(data_month, "M") + 1).to_timestamp(how="end").normalize()


def _is_backfill(data_month: str, registered: str) -> bool:
    if not data_month or not registered:
        return False
    return pd.Timestamp(registered) > month_end(data_month) + pd.Timedelta(RELEASE_MAX_LAG_DAYS, "D")


def estimate_data_month(title: str, registered: str) -> str:
    """Data month guess before download: the month BEFORE 'YYYY년 M월' in the title, else the month before
    registration (2026-08-25 -> 2026-07). Verified: the file titled '2015년 12월' holds 자료생성년월 2015-11, the one
    registered 2026-08-25 holds 2026-07. Only used to select files; the ingested month is read from the CSV."""
    m = _TITLE_YM.search(title or "")
    if m:
        return str(pd.Period(f"{int(m.group(1)):04d}-{int(m.group(2)):02d}", "M") - 1)
    return str(pd.Period(registered, "M") - 1)


def parse_detail_pk(page_html: str) -> str:
    """Current file's ``publicDataDetailPk`` from the dataset page."""
    m = _DETAIL_PK.search(page_html)
    if not m:
        raise ValueError("data.go.kr 15083277: publicDataDetailPk 없음 (페이지 구조 변경?)")
    return m.group(1)


def parse_hist_listing(html: str) -> list[dict[str, str]]:
    """'주기성 과거 데이터' tab (selectHistAndCsvData.do) -> [{detail_pk, hist_sn, title, registered}], newest first."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    tab = soup.find(id="tab-layer-file-05")
    if tab is None:
        raise ValueError("data.go.kr: 주기성 과거 데이터 탭(tab-layer-file-05) 없음")
    out: list[dict[str, str]] = []
    for tr in tab.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        registered = tds[-1].get_text(strip=True)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", registered):
            continue
        for a in tds[0].find_all("a", attrs={"data-public-pk": True}):
            out.append(
                {
                    "detail_pk": str(a["data-public-pk"]),
                    "hist_sn": str(a.get("data-public-detail-sn") or "1"),
                    "title": re.sub(r"\s+", " ", a.get_text(" ", strip=True)),
                    "registered": registered,
                }
            )
    return out


def parse_detail_info(html: str) -> dict[str, str]:
    """File popup (selectDpkDetailInfo.do) -> {atch_file_id, file_sn, registered, modified, next_due}."""
    from bs4 import BeautifulSoup

    m = _DOWN_CALL.search(html)
    if not m:
        raise ValueError("data.go.kr 상세: CSV 다운로드 정보(fn_fileDataDown ... 'csv') 없음")
    kv: dict[str, str] = {}
    for li in BeautifulSoup(html, "lxml").find_all("li"):
        k, v = li.find("strong", class_="key"), li.find(class_="value")
        if k is not None and v is not None:
            kv[k.get_text(strip=True)] = v.get_text(" ", strip=True)

    def _d(label: str) -> str:
        s = kv.get(label, "")
        return s[:10] if re.match(r"\d{4}-\d{2}-\d{2}", s) else ""

    return {
        "detail_pk": m.group(1),
        "atch_file_id": m.group(2),
        "file_sn": m.group(3),
        "registered": _d("등록일"),
        "modified": _d("수정일"),
        "next_due": _d("차기 등록 예정일"),
    }


class _Gate:
    """Process-wide politeness gap for data.go.kr (sequential use; the lock only keeps threads honest)."""

    def __init__(self, gap: float):
        self.gap = gap
        self._last = -1e18
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            pause = self._last + self.gap - time.monotonic()
            if pause > 0:
                time.sleep(pause)
            self._last = time.monotonic()


_gate = _Gate(PORTAL_MIN_INTERVAL_S)


def _post(session: Any, url: str, data: Mapping[str, str], timeout: float = 30.0) -> Any:
    _gate.wait()
    r = session.post(
        url,
        data=dict(data),
        headers={"Referer": PAGE_URL, "X-Requested-With": "XMLHttpRequest"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r


def fetch_listing(session: Any, progress: Progress | None = None) -> list[NpsFile]:
    """Current file + every past file of the dataset with its atchFileId and dates (1 + 1 + N portal requests)."""
    _gate.wait()
    r = session.get(PAGE_URL, timeout=30)
    r.raise_for_status()
    current_pk = parse_detail_pk(r.text)
    hist = parse_hist_listing(
        _post(session, HIST_URL, {"publicDataPk": DATASET_PK, "publicDataDetailPk": current_pk}).text
    )
    entries = [{"detail_pk": current_pk, "hist_sn": "1", "title": "(current)", "registered": "", "current": True}]
    entries += [{**h, "current": False} for h in hist if h["detail_pk"] != current_pk]
    out: list[NpsFile] = []
    for i, e in enumerate(entries, 1):
        info = parse_detail_info(
            _post(session, DETAIL_URL, {"publicDataDetailPk": e["detail_pk"], "publicDataHistSn": e["hist_sn"]}).text
        )
        registered = e["registered"] or info["registered"]
        title = e["title"] if not e["current"] else f"(current) {info['registered']}"
        out.append(
            NpsFile(
                detail_pk=e["detail_pk"],
                hist_sn=e["hist_sn"],
                title=title,
                registered=registered,
                modified=info["modified"],
                atch_file_id=info["atch_file_id"],
                file_sn=info["file_sn"],
                data_month=estimate_data_month(e["title"] if not e["current"] else "", registered),
                month_estimated=True,
                current=bool(e["current"]),
            )
        )
        if progress is not None:
            progress(i, len(entries), f"listing {registered} {info['atch_file_id']}")
    return out


# ---------------------------------------------------------------------------------------------- storage


def nps_root(settings: Settings) -> Path:
    return settings.cache_path / "alt" / SOURCE


def _json_read(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("kr_nps: %s 읽기 실패 (%s) - 없는 것으로 처리", path.name, exc)
        return None


def _json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def _month_key(data_month: str) -> str:
    return data_month.replace("-", "")


def cached_months(root: Path) -> dict[str, dict[str, Any]]:
    """{'YYYY-MM': sidecar metadata} of every ingested month (parquet + json both present)."""
    out: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return out
    for meta_path in sorted(root.glob("[0-9]" * 6 + ".json")):
        meta = _json_read(meta_path)
        if meta and meta_path.with_suffix(".parquet").exists():
            out[str(meta["data_month"])] = meta
    return out


def _files_index(root: Path) -> dict[str, str]:
    return _json_read(root / "_files.json") or {}


def nps_available_months(
    settings: Settings,
    *,
    refresh: bool = False,
    offline: bool = False,
    session: Any | None = None,
    progress: Progress | None = None,
) -> list[NpsFile]:
    """What the portal offers (newest first): one ``NpsFile`` per file with its data month (actual once ingested,
    else estimated), registration / modification date and download URL. The listing costs ~131 portal requests
    and is cached for ``LISTING_TTL_HOURS``; ``offline=True`` returns the cached listing of any age ([] if none).
    The earliest data month is ``earliest_data_month(files)`` (verified once that file is ingested)."""
    root = nps_root(settings)
    path = root / "_listing.json"
    hit = None if refresh else _json_read(path)
    fresh = hit is not None and (
        offline
        or (datetime.now(UTC) - datetime.fromisoformat(hit["fetched_at"])).total_seconds() < LISTING_TTL_HOURS * 3600
    )
    if fresh:
        files = [NpsFile(**f) for f in hit["files"]]
    elif offline:
        return []
    else:
        files = fetch_listing(session or PoliteSession(timeout=60), progress)
        _json_write(path, {"fetched_at": datetime.now(UTC).isoformat(), "files": [asdict(f) for f in files]})
    index = _files_index(root)
    out = []
    for f in files:
        actual = index.get(f.atch_file_id)
        if actual:
            f = NpsFile(**{**asdict(f), "data_month": actual, "month_estimated": False})
        else:  # re-estimated on every read: the rule may have improved since the listing was cached
            est = estimate_data_month("" if f.current else f.title, f.registered)
            f = NpsFile(**{**asdict(f), "data_month": est, "month_estimated": True})
        out.append(f)
    return sorted(out, key=lambda f: (f.registered, f.data_month), reverse=True)


def earliest_data_month(files: Sequence[NpsFile]) -> tuple[str, bool]:
    """(earliest data month 'YYYY-MM' the portal offers, verified). ``verified`` is True when that month was read
    from an ingested file (``_files.json``); otherwise it is the estimate from the title ('2015년 12월') or the
    registration date. ('', False) for an empty listing."""
    if not files:
        return "", False
    first = min(files, key=lambda f: (f.data_month, f.month_estimated))
    return first.data_month, not first.month_estimated


# ---------------------------------------------------------------------------------------------- parsing


def _sniff_encoding(head: bytes) -> str:
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        head.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError as exc:
        if exc.start >= len(head) - 3:  # a multi-byte char cut by the sample boundary
            return "utf-8"
        return "cp949"


def _header_key(col: str) -> str:
    return re.split(r"[\s(:]", str(col).strip().lstrip("\ufeff"), maxsplit=1)[0]


def _norm_month(s: Any) -> str:
    d = re.sub(r"\D", "", str(s))
    return f"{d[:4]}-{d[4:6]}" if len(d) >= 6 else ""


def parse_nps_csv(path: Path | str) -> pd.DataFrame:
    """One monthly CSV (cp949, or utf-8) -> 법인 rows (사업장형태구분코드 == 1) with ``COMPACT_COLUMNS``:
    data_month 'YYYY-MM', name, bizr6 (6 digits), status (1 등록 / 2 탈퇴), industry, members, billed, new, lost."""
    path = Path(path)
    with path.open("rb") as fh:
        head = fh.read(65536)
    enc = _sniff_encoding(head)
    header = pd.read_csv(path, encoding=enc, nrows=0, encoding_errors="replace").columns
    rename = {}
    for c in header:
        key = _header_key(c)
        if key in NPS_COLUMNS and NPS_COLUMNS[key] not in rename.values():
            rename[c] = NPS_COLUMNS[key]
    missing = set(NPS_COLUMNS.values()) - set(rename.values())
    if missing:
        raise ValueError(f"NPS CSV {path.name}: 컬럼 없음 {sorted(missing)} (헤더 {list(header)[:6]}...)")
    df = pd.read_csv(path, encoding=enc, usecols=list(rename), dtype=str, encoding_errors="replace")
    df = df.rename(columns=rename)
    df = df[df["form"].str.strip() == "1"].copy()
    out = pd.DataFrame(
        {
            "data_month": df["data_month"].map(_norm_month),
            "name": df["name"].fillna("").str.strip(),
            "bizr6": df["bizr6"].fillna("").str.replace(r"\D", "", regex=True).str[:6].str.zfill(6),
            "status": pd.to_numeric(df["status"], errors="coerce").fillna(0).astype("int8"),
            "industry": df["industry"].fillna("").str.strip(),
        }
    )
    for c in ("members", "billed", "new", "lost"):
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return out[COMPACT_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------------------------- download


def download_csv(session: Any, f: NpsFile, tmp_dir: Path, *, check_limit: bool = True) -> Path:
    """Stream one monthly file to ``tmp_dir/<atchFileId>.csv`` (via ``.part``; the portal ignores Range, so a
    broken download restarts). A complete CSV left by an interrupted run is reused."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    final = tmp_dir / f"{f.atch_file_id}.csv"
    if final.exists() and final.stat().st_size > 0:
        return final
    if check_limit:
        r = _post(session, LIMIT_URL, {"atchFileId": f.atch_file_id, "fileDetailSn": f.file_sn})
        if r.json().get("needCaptcha"):
            raise NpsDownloadLimitError("data.go.kr 다운로드 제한(CAPTCHA 요구) - 나중에 재개")
    part = tmp_dir / f"{f.atch_file_id}.csv.part"
    part.unlink(missing_ok=True)
    _gate.wait()
    try:
        with session.get(
            DOWNLOAD_URL,
            params={"atchFileId": f.atch_file_id, "fileDetailSn": f.file_sn, "insertDataPrcus": "N"},
            headers={"Referer": PAGE_URL},
            stream=True,
            timeout=(15, 120),
        ) as r:
            if r.status_code == 403:
                raise NpsDownloadLimitError(f"data.go.kr 403 ({f.atch_file_id}) - 다운로드 제한")
            r.raise_for_status()
            if "text/html" in r.headers.get("Content-Type", ""):
                raise RuntimeError(f"data.go.kr {f.atch_file_id}: CSV 대신 HTML 응답")
            expected = int(r.headers.get("Content-Length") or 0)
            got = 0
            with part.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if got == 0 and chunk.lstrip()[:1] == b"<":
                        raise RuntimeError(f"data.go.kr {f.atch_file_id}: CSV 대신 HTML 응답")
                    fh.write(chunk)
                    got += len(chunk)
    except BaseException:  # a partial file is useless (no Range support): never leave it behind
        part.unlink(missing_ok=True)
        raise
    if expected and got != expected:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"data.go.kr {f.atch_file_id}: {got}/{expected} bytes (중단된 다운로드)")
    part.replace(final)
    return final


def ingest_csv(
    root: Path,
    csv_path: Path,
    *,
    registered: str = "",
    modified: str = "",
    atch_file_id: str = "",
    title: str = "",
    delete_csv: bool = True,
) -> dict[str, Any]:
    """Parse one monthly CSV into ``root/<YYYYMM>.parquet`` + sidecar json, record the file in ``_files.json`` and
    delete the CSV. When the month already exists from another file, the copy with the EARLIER release date is
    kept (what the market saw first) and the other one is recorded as a duplicate."""
    df = parse_nps_csv(csv_path)
    months = df["data_month"].value_counts()
    if months.empty:
        raise ValueError(f"{csv_path.name}: 법인 행 없음")
    data_month = str(months.index[0])
    if len(months) > 1:
        log.warning("kr_nps %s: 자료생성년월 여러 개 %s - %s 만 사용", csv_path.name, dict(months), data_month)
        df = df[df["data_month"] == data_month]
    meta: dict[str, Any] = {
        "data_month": data_month,
        "atch_file_id": atch_file_id,
        "title": title,
        "registered": registered,
        "modified": modified,
        "rows_corp": len(df),
        "csv_bytes": csv_path.stat().st_size,
        "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "duplicates": [],
    }
    root.mkdir(parents=True, exist_ok=True)
    key = _month_key(data_month)
    old = _json_read(root / f"{key}.json")
    pq_path = root / f"{key}.parquet"
    keep_new = True
    if old and pq_path.exists() and old.get("atch_file_id") != atch_file_id:
        old_rel = max([d for d in (old.get("registered"), old.get("modified")) if d] or ["9999-12-31"])
        new_rel = max([d for d in (registered, modified) if d] or ["9999-12-31"])
        keep_new = new_rel < old_rel
        loser = {k: (old if keep_new else meta).get(k) for k in ("atch_file_id", "title", "registered", "modified")}
        meta["duplicates"] = [*old.get("duplicates", []), loser] if keep_new else []
        if not keep_new:
            old["duplicates"] = [*old.get("duplicates", []), loser]
            _json_write(root / f"{key}.json", old)
        log.warning("kr_nps %s: 같은 자료월의 파일 2개 - 이른 공개본 유지 (%s)", data_month, loser)
    if keep_new:
        tmp = pq_path.with_name(f"{pq_path.name}.{os.getpid()}.tmp")
        df.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(pq_path)
        _json_write(root / f"{key}.json", meta)
    if atch_file_id:
        index = _files_index(root)
        index[atch_file_id] = data_month
        _json_write(root / "_files.json", index)
    if delete_csv:
        csv_path.unlink(missing_ok=True)
    return meta if keep_new else old  # type: ignore[return-value]


def load_month(root: Path, data_month: str, prefixes: Iterable[str] | None = None) -> pd.DataFrame:
    """Compact rows of one ingested month, optionally only the given 6-digit registration-number prefixes."""
    path = root / f"{_month_key(data_month)}.parquet"
    filters = None
    if prefixes is not None:
        filters = [("bizr6", "in", sorted(set(prefixes)))]
    return pd.read_parquet(path, filters=filters)


@dataclass
class DownloadSummary:
    selected: int = 0
    skipped_cached: int = 0
    downloaded: int = 0
    failed: list[str] = field(default_factory=list)
    stopped: str = ""
    seconds: float = 0.0
    bytes: int = 0


def download_kr_nps_history(
    settings: Settings,
    start: date | None = None,
    end: date | None = None,
    *,
    session: Any | None = None,
    progress: Progress | None = None,
    max_files: int | None = None,
    files: Sequence[NpsFile] | None = None,
) -> DownloadSummary:
    """Download + ingest, one file at a time (newest first), every listed file whose (estimated) data month is in
    [start - 1 month, end + 1 month] and whose atchFileId was not ingested before. Resumable: stop at any point
    and call again. Stops at the first download-limit answer (CAPTCHA); other failures are counted and skipped.
    The full history (130 files x ~100 MB) is a long job: run it in the background with ``max_files``."""
    root = nps_root(settings)
    session = session or PoliteSession(timeout=60)
    files = list(files) if files is not None else nps_available_months(settings, session=session)
    lo = str(pd.Period(start, "M") - 1) if start else "0000-00"
    hi = str(pd.Period(end, "M") + 1) if end else "9999-99"
    index = _files_index(root)
    todo = []
    out = DownloadSummary()
    for f in files:
        if not (lo <= f.data_month <= hi) or not f.atch_file_id:
            continue
        out.selected += 1
        if f.atch_file_id in index:
            out.skipped_cached += 1
            continue
        todo.append(f)
    if max_files is not None:
        todo = todo[:max_files]
    t_all = time.monotonic()
    for i, f in enumerate(todo, 1):
        t0 = time.monotonic()
        try:
            csv_path = download_csv(session, f, root / "_tmp")
            size = csv_path.stat().st_size
            meta = ingest_csv(
                root,
                csv_path,
                registered=f.registered,
                modified=f.modified,
                atch_file_id=f.atch_file_id,
                title=f.title,
            )
        except NpsDownloadLimitError as exc:
            out.stopped = str(exc)
            log.warning("kr_nps: %s", exc)
            if progress is not None:
                progress(i, len(todo), f"{f.data_month} stopped: {exc}")
            break
        except Exception as exc:  # noqa: BLE001 - one bad file never stops the rest; counted in failed
            out.failed.append(f"{f.atch_file_id} ({f.data_month}): {type(exc).__name__}: {str(exc)[:160]}")
            log.warning("kr_nps %s 실패: %s", f.atch_file_id, exc)
            if progress is not None:
                progress(i, len(todo), f"{f.data_month} failed: {exc}")
            continue
        out.downloaded += 1
        out.bytes += size
        if progress is not None:
            progress(
                i,
                len(todo),
                f"{meta['data_month']} {f.atch_file_id} {size / 1e6:.0f} MB {time.monotonic() - t0:.0f} s",
            )
    out.seconds = time.monotonic() - t_all
    return out


# ------------------------------------------------------------------------------------------------- matching

_LEGAL_FORMS = re.compile(
    r"주식회사|유한회사|유한책임회사|\(주\)|\(유\)|㈜|\(株\)|株式會社|\bco\b\.?,?\s*ltd\b\.?|\binc\b\.?|\bcorp(?:oration)?\b\.?"
)
_NON_WORD = re.compile(r"[\W_]+")


def norm_company_name(s: Any) -> str:
    """Name key for matching: legal-form tokens ((주), 주식회사, ㈜, ...), spaces and punctuation removed, lower case.
    '에스케이하이닉스 주식회사' / '에스케이하이닉스(주)' -> '에스케이하이닉스'."""
    t = str(s or "").lower()
    t = _LEGAL_FORMS.sub("", t)
    return _NON_WORD.sub("", t)


@dataclass(frozen=True)
class Company:
    entity: str
    corp_code: str
    corp_name: str
    stock_name: str
    bizr_no: str
    adres: str = ""
    induty_code: str = ""

    @property
    def prefix(self) -> str:
        d = re.sub(r"\D", "", self.bizr_no)
        return d[:6] if len(d) >= 6 else ""

    def names(self, aliases: Sequence[str] = ()) -> list[str]:
        return [n for n in (self.corp_name, self.stock_name, *aliases) if n]


def match_rows(rows: pd.DataFrame, prefix: str, names: Sequence[str]) -> tuple[pd.DataFrame, str, int]:
    """Rows of ``prefix`` belonging to a company with ``names``: (matched rows, quality, excluded rows).
    quality 'exact' (normalised names equal; preferred), 'contained' (one name contains the other, only when no
    exact row exists), 'none'. Rows of the prefix with other names are excluded and counted."""
    cand = rows[rows["bizr6"] == prefix]
    if cand.empty:
        return cand, "none", 0
    keys = {k for k in (norm_company_name(n) for n in names) if k}
    nk = cand["name"].map(norm_company_name)
    exact = nk.isin(keys)
    if exact.any():
        return cand[exact], "exact", int((~exact).sum())

    def _contained(x: str) -> bool:
        return len(x) >= MIN_CONTAINED_LEN and any(len(k) >= MIN_CONTAINED_LEN and (k in x or x in k) for k in keys)

    cont = nk.map(_contained)
    if cont.any():
        return cand[cont], "contained", int((~cont).sum())
    return cand.iloc[0:0], "none", len(cand)


# ------------------------------------------------------------------------------------------ DART companies


def _corp_codes(settings: Settings, session: Any, key: str) -> dict[str, str]:
    """{stock_code: corp_code} (OpenDART corpCode.xml), cached for ``CORP_CODES_TTL_DAYS``; a stale copy is used
    when there is no key or the refresh fails (logged)."""
    from quantcio.data.pit.kr_dart import opendart_corp_codes

    path = settings.cache_path / "alt" / "kr_dart_company" / "_corp_codes.json"
    hit = _json_read(path)
    if hit and time.time() - path.stat().st_mtime < CORP_CODES_TTL_DAYS * 86400:
        return hit
    if not key:
        return hit or {}
    try:
        codes = opendart_corp_codes(session, key)
    except Exception as exc:  # noqa: BLE001 - fall back to the stale map when there is one; logged
        if not hit:
            raise
        log.warning("kr_nps: corpCode 갱신 실패 (%s) - 이전 캐시 사용", exc)
        return hit
    _json_write(path, codes)
    return codes


def dart_companies(
    settings: Settings,
    entities: Iterable[str],
    *,
    session: Any | None = None,
    key: str | None = None,
) -> tuple[dict[str, Company], Counter[str], list[str]]:
    """{entity: Company} from OpenDART corpCode + company.json (answers cached under
    ``<cache>/alt/kr_dart_company/<corp_code>.json`` for ``COMPANY_TTL_DAYS``). Without a key only cached answers
    are used. Returns (companies, counters, notes)."""
    from quantcio.data.pit.kr_dart import opendart_json

    key = settings.data.dart_api_key if key is None else key
    session = session or PoliteSession()
    root = settings.cache_path / "alt" / "kr_dart_company"
    cnt: Counter[str] = Counter()
    notes: list[str] = []
    try:
        codes = _corp_codes(settings, session, key)
    except Exception as exc:  # noqa: BLE001 - no corp-code map -> nothing matchable; counted
        codes = {}
        notes.append(f"kr_nps: OpenDART corpCode 실패 ({type(exc).__name__}: {str(exc)[:120]})")
    out: dict[str, Company] = {}
    for ent in entities:
        cc = codes.get(ent)
        if not cc:
            cnt["no_corp_code"] += 1
            continue
        path = root / f"{cc}.json"
        j = _json_read(path)
        stale = j is None or time.time() - path.stat().st_mtime > COMPANY_TTL_DAYS * 86400
        if stale and key:
            try:
                j = opendart_json(session, key, "company", corp_code=cc)
                _json_write(path, j)
                cnt["company_fetched"] += 1
            except Exception as exc:  # noqa: BLE001 - one company failing never stops the rest; counted
                cnt["company_failed"] += 1
                log.warning("kr_nps: company.json %s (%s) 실패: %s", ent, cc, exc)
        if not j or j.get("status") != "000":
            cnt["no_company_info"] += 1
            continue
        c = Company(
            entity=ent,
            corp_code=cc,
            corp_name=str(j.get("corp_name") or ""),
            stock_name=str(j.get("stock_name") or ""),
            bizr_no=str(j.get("bizr_no") or ""),
            adres=str(j.get("adres") or ""),
            induty_code=str(j.get("induty_code") or ""),
        )
        if not c.prefix:
            cnt["no_bizr_no"] += 1
            continue
        out[ent] = c
    if cnt["no_corp_code"]:
        notes.append(f"kr_nps: DART corp_code 없음 {cnt['no_corp_code']}개 종목 (상장폐지/코드 변경) - 매칭 불가")
    return out, cnt, notes


# --------------------------------------------------------------------------------------------- published


def release_published(
    data_month: str, registered: str = "", modified: str = "", backfill_rule: BackfillRule = "registration"
) -> tuple[pd.Timestamp, str]:
    """(published, rule) of one ingested month; rules 'release', 'backfill_registration', 'backfill_month_after',
    'unknown_month_after' (see module docstring)."""
    dates = [d for d in (registered, modified) if d]
    if not dates:
        return conservative_published(data_month), "unknown_month_after"
    release = pd.Timestamp(max(dates)).normalize()
    if _is_backfill(data_month, registered or max(dates)):
        if backfill_rule == "month_after":
            return conservative_published(data_month), "backfill_month_after"
        return release, "backfill_registration"
    return release, "release"


# ------------------------------------------------------------------------------------------------- loader


def load_kr_nps(
    settings: Settings,
    entities: Iterable[str],
    start: date,
    end: date | None = None,
    *,
    download: bool = True,
    data_go_kr_key: str = "",
    progress: Progress | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
    backfill_rule: BackfillRule = "registration",
    session: Any | None = None,
) -> AltData:
    """NPS measures ``ITEMS`` (members, newly insured, lost) per entity and data month in [start, end] (market KR).

    ``download=False`` uses only months already ingested. ``data_go_kr_key`` is not needed: the monthly files
    (current and history) download without a key; the parameter is accepted for interface stability.
    ``aliases``: extra names per entity (former names) for months before a rename.

    ``covered["kr_nps"]`` = the matched entities when every month the portal offers inside [start, end] is ingested
    (else empty: an incomplete range never claims "no row = nothing"); unmatched entities are never covered.
    """
    end = end or date.today()
    ents = sorted({str(e).strip().zfill(6) for e in entities if str(e).strip()})
    root = nps_root(settings)
    notes: list[str] = []
    cov: dict[str, Any] = {"entities_asked": len(ents)}
    session = session or PoliteSession(timeout=60)

    listing: list[NpsFile] = []
    if download:
        try:
            listing = nps_available_months(settings, session=session)
            summary = download_kr_nps_history(settings, start, end, session=session, progress=progress, files=listing)
            cov["download"] = asdict(summary)
            if summary.stopped:
                notes.append(f"kr_nps: 다운로드 중단 - {summary.stopped} (다음 실행에서 재개)")
            if summary.failed:
                notes.append(f"kr_nps: {len(summary.failed)}개 파일 실패: {summary.failed[:3]}")
        except Exception as exc:  # noqa: BLE001 - the portal being down must not stop cached months; counted
            notes.append(f"kr_nps: 목록/다운로드 실패 ({type(exc).__name__}: {str(exc)[:160]}) - 캐시만 사용")
            cov["listing_failed"] = 1
    else:
        listing = nps_available_months(settings, offline=True)
    if listing:
        earliest, verified = earliest_data_month(listing)
        cov["months_available"] = len({f.data_month for f in listing})
        cov["files_available"] = len(listing)
        cov["earliest_month_available"] = earliest
        cov["earliest_month_verified"] = verified

    months = cached_months(root)
    lo, hi = str(pd.Period(start, "M")), str(pd.Period(end, "M"))
    in_range = {m: meta for m, meta in months.items() if lo <= m <= hi}
    cov["months_cached"] = len(months)
    cov["months_in_range"] = len(in_range)
    cov["earliest_month_cached"] = min(months) if months else ""
    complete = bool(listing)
    if listing:
        want = {f.data_month for f in listing if lo <= f.data_month <= hi}
        cov["months_missing"] = sorted(want - set(in_range))
        complete = not cov["months_missing"]
        if cov["months_missing"]:
            notes.append(
                f"kr_nps: 요청 구간의 {len(cov['months_missing'])}개 월 미수집 {cov['months_missing'][:6]} - "
                "covered 비움 (다운로드 재개 후 완전)"
            )
    else:
        notes.append("kr_nps: 포털 목록 없음 - 구간 완전성 확인 불가, covered 비움")

    companies, ccnt, cnotes = dart_companies(settings, ents, session=session)
    notes += cnotes
    cov["company_info"] = dict(ccnt) | {"with_bizr_no": len(companies)}
    if not in_range or not companies:
        cov["entities_matched"] = 0
        cov["entities_unmatched"] = len(ents)
        notes.append(f"kr_nps: 사용 가능한 월 {len(in_range)}개, 사업자번호 확인 종목 {len(companies)}개 - 측정치 없음")
        return AltData("KR", measures=empty_measures(), coverage={SOURCE: cov}, notes=notes, covered={SOURCE: []})

    prefixes = {c.prefix for c in companies.values()}
    aliases = aliases or {}
    recs: list[dict[str, Any]] = []
    quality: dict[str, Counter[str]] = {e: Counter() for e in companies}
    excluded = 0
    rules: Counter[str] = Counter()
    for m, meta in sorted(in_range.items()):
        published, rule = release_published(m, meta.get("registered", ""), meta.get("modified", ""), backfill_rule)
        rules[rule] += 1
        rows = load_month(root, m, prefixes)
        for ent, c in companies.items():
            got, q, n_ex = match_rows(rows, c.prefix, c.names(aliases.get(ent, ())))
            quality[ent][q] += 1
            excluded += n_ex
            if got.empty:
                continue
            for item, col in ITEM_COLUMN.items():
                recs.append(
                    {
                        "entity": ent,
                        "published": published,
                        "period_end": month_end(m),
                        "item": item,
                        "value": float(got[col].sum(min_count=1)),
                        "source": SOURCE,
                    }
                )
    best = {e: ("exact" if q["exact"] else "contained" if q["contained"] else "none") for e, q in quality.items()}
    matched = sorted(e for e, b in best.items() if b != "none")
    cov |= {
        "entities_matched": len(matched),
        "entities_unmatched": len(ents) - len(matched),
        "unmatched": sorted(set(ents) - set(matched)),
        "match_quality": dict(Counter(best.values())),
        "entity_months": {q: sum(c[q] for c in quality.values()) for q in ("exact", "contained", "none")},
        "prefix_rows_excluded": excluded,
        "published_rules": dict(rules),
        "rows": len(recs),
    }
    if rules["backfill_registration"]:
        notes.append(
            f"kr_nps: {rules['backfill_registration']}개 월은 2019-08 일괄 등록분 - published = 등록일 "
            "(원 공개일 미상; backfill_rule='month_after' 는 적시 공개를 가정)"
        )
    if rules["backfill_month_after"]:
        notes.append(
            f"kr_nps: {rules['backfill_month_after']}개 월은 공개일 미상 - 자료월 다음 달 말일을 공개일로 가정"
        )
    if rules["unknown_month_after"]:
        notes.append(f"kr_nps: {rules['unknown_month_after']}개 월은 공개일 미상 - 자료월 다음 달 말일 사용")
    if cov["entity_months"]["contained"]:
        notes.append(f"kr_nps: 이름 부분일치 매칭 {cov['entity_months']['contained']}건 (정확일치 없음)")
    cov["covered"] = len(matched) if complete else 0
    measures = normalise_measures(pd.DataFrame(recs)) if recs else empty_measures()
    return AltData(
        "KR", measures=measures, coverage={SOURCE: cov}, notes=notes, covered={SOURCE: matched if complete else []}
    )


__all__ = [
    "ITEMS",
    "SOURCE",
    "Company",
    "DownloadSummary",
    "NpsDownloadLimitError",
    "NpsFile",
    "cached_months",
    "conservative_published",
    "dart_companies",
    "download_csv",
    "download_kr_nps_history",
    "earliest_data_month",
    "estimate_data_month",
    "fetch_listing",
    "ingest_csv",
    "load_kr_nps",
    "load_month",
    "match_rows",
    "month_end",
    "norm_company_name",
    "nps_available_months",
    "parse_detail_info",
    "parse_detail_pk",
    "parse_hist_listing",
    "parse_nps_csv",
    "release_published",
]
