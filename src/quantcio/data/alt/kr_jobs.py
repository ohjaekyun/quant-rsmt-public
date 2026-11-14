"""Job postings of listed KR companies - FORWARD RECORDING ONLY (no historical archive exists).

Neither source keeps closed postings queryable, so a past date can never be reconstructed: this module only takes
snapshots from today on and appends them to an append-only JSONL ledger (default ``alt_ledger/kr_jobs.jsonl``,
committed by the user). Use the measures in models only after >= 12 months of snapshots have accumulated.

Sources (docs checked 2026-10-02; both answer without a key with an error message):
- Saramin Open API ``GET https://oapi.saramin.co.kr/job-search`` (header ``Accept: application/json``), key
  parameter ``access-key`` (apply at https://oapi.saramin.co.kr/ -> 이용신청; approval required). 500 calls per
  key per day (error code 4 beyond), at most 110 postings per call (``count``), ``start`` is the 0-based page.
  Only ACTIVE postings are served. There is no company-id filter: ``keywords`` searches company name, title,
  keywords and job text at once, so results contain other companies; ``stock=kospi,kosdaq,konex`` limits to
  listed companies. Posting fields used: ``company.detail.name`` and ``company.detail.href`` (``csn=`` is the
  business registration number for most companies), ``active``, ``posting-timestamp``, ``expiration-timestamp``
  (unix seconds), ``position.job-mid-code.name`` (category, comma separated), ``position.experience-level.code``
  (1 신입, 2 경력, 3 신입/경력, 0 무관). Errors: ``{"code": 1|2|3|4|99, "message": ...}``. Requests need a browser
  User-Agent (a curl UA gets HTTP 307 to an error page).
  Modes: ``company`` = one keyword search per company (1+ calls each); ``sweep`` = every active posting of listed
  companies (``stock`` filter, no keyword; ~total/110 calls) matched locally - the only way to cover ~2,500
  companies within 500 calls/day. ``auto`` picks sweep above ``SWEEP_MIN_COMPANIES`` companies.
- 고용24 (Work24, formerly 워크넷) 채용정보 목록 OPEN-API ``GET https://www.work24.go.kr/cm/openApi/call/wk/
  callOpenApiSvcInfo210L01.do`` with ``authKey``, ``callTp=L``, ``returnType=XML``, ``startPage`` (1-based),
  ``display`` (<= 100) and ``coNm`` (company name filter). Rows ``<wanted>``: wantedAuthNo, company, busino,
  title, career, regDt, closeDt, jobsCd, smodifyDtm, ...; errors ``<GO24><error>..</error></GO24>`` (legacy
  openapi.work.go.kr: ``<messageCd>``/``<message>``). The key is issued at https://www.work24.go.kr (OPEN-API
  menu) to COMPANY members only - a personal-member key answers "개인회원은 사용할 수 없는 OPEN-API입니다". The
  response layout above is from the legacy WorkNet docs and public clients: re-check with the first real key.

Company matching is strict: a posting belongs to an entity only when its normalised company name
(``quantcio.data.alt.kr_nps.norm_company_name``) equals one of the entity's normalised names, or its
registration number equals the entity's ``bizr_no`` when given. Everything else the search returned is excluded
and counted (``excluded_other``); a normalised name shared by two entities is ambiguous and never matched.

Point in time: a snapshot taken on date d is published on d (``published`` = snapshot date, ``period_end`` = d);
the shared rule (``published < t``) uses it from the next grid date on. Snapshots that did not complete (quota,
page cap, errors) are recorded with ``complete = false`` and never become measures.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree

import pandas as pd

from quantcio.config import PROJECT_ROOT
from quantcio.data.alt import AltData, empty_measures, normalise_measures
from quantcio.data.alt.kr_nps import norm_company_name
from quantcio.data.http import PoliteSession

log = logging.getLogger(__name__)

SARAMIN_URL = "https://oapi.saramin.co.kr/job-search"
WORK24_URL = "https://www.work24.go.kr/cm/openApi/call/wk/callOpenApiSvcInfo210L01.do"
SARAMIN_DAILY_LIMIT = 500
SARAMIN_PAGE = 110
SARAMIN_STOCKS = "kospi,kosdaq,konex"
WORK24_PAGE = 100
SWEEP_MIN_COMPANIES = 40
MIN_GAP_S = {"saramin": 0.5, "work24": 0.5}
NEW_DAYS = 7
DEFAULT_LEDGER = PROJECT_ROOT / "alt_ledger" / "kr_jobs.jsonl"  # nothing is created at import
ITEMS = ("job_postings_active", "job_postings_new_7d")
ITEM_COLUMN = {"job_postings_active": "active_postings", "job_postings_new_7d": "new_7d"}
SOURCES = ("saramin", "work24")
SNAPSHOT_COLUMNS = [
    "date",
    "entity",
    "source",
    "active_postings",
    "new_7d",
    "postings_matched",
    "categories",
    "experience",
    "first_posted",
    "last_posted",
    "last_expiration",
    "query",
    "query_hits",
    "excluded_other",
    "complete",
    "api_calls",
    "fetched_at",
]
SaraminMode = Literal["auto", "company", "sweep"]
_EXPERIENCE = {"0": "any", "1": "new", "2": "experienced", "3": "new_or_experienced"}
_LEGAL = re.compile(r"주식회사|\(주\)|㈜|\(유\)|유한회사")


class JobsApiError(RuntimeError):
    """Error answer of a job-posting API (bad / missing key, bad parameter, system error)."""


class JobsQuotaError(JobsApiError):
    """Daily call limit reached (Saramin code 4) or the caller's call budget is used up."""


# ------------------------------------------------------------------------------------------------ parsing


@dataclass(frozen=True)
class Posting:
    """One posting reduced to what the snapshot needs (source-independent)."""

    source: str
    posting_id: str
    company: str
    bizr_no: str  # digits only, "" if the source does not say
    active: bool
    posted: datetime | None
    expires: datetime | None
    categories: tuple[str, ...] = ()
    experience: str = ""


def _ts(x: Any) -> datetime | None:
    try:
        v = int(str(x).strip())
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(v, UTC) if v > 0 else None


def parse_saramin(payload: Mapping[str, Any]) -> tuple[int, list[Posting]]:
    """Saramin job-search JSON -> (total hits, postings). Raises ``JobsQuotaError`` (code 4) / ``JobsApiError``."""
    if "jobs" not in payload:
        code = str(payload.get("code", "?"))
        msg = f"Saramin {code}: {payload.get('message', '')}".strip()
        raise JobsQuotaError(msg) if code == "4" else JobsApiError(msg)
    jobs = payload["jobs"] or {}
    total = int(str(jobs.get("total") or 0).replace(",", "") or 0)
    rows = jobs.get("job") or []
    if isinstance(rows, Mapping):
        rows = [rows]
    out = []
    for j in rows:
        comp = (j.get("company") or {}).get("detail") or {}
        href = str(comp.get("href") or "")
        m = re.search(r"[?&]csn=(\d+)", href)
        pos = j.get("position") or {}
        cat = str((pos.get("job-mid-code") or {}).get("name") or "")
        exp = str((pos.get("experience-level") or {}).get("code", ""))
        out.append(
            Posting(
                source="saramin",
                posting_id=str(j.get("id") or ""),
                company=str(comp.get("name") or "").strip(),
                bizr_no=m.group(1) if m else "",
                active=str(j.get("active", "1")) == "1",
                posted=_ts(j.get("posting-timestamp")),
                expires=_ts(j.get("expiration-timestamp")),
                categories=tuple(c.strip() for c in cat.split(",") if c.strip()),
                experience=_EXPERIENCE.get(exp, exp),
            )
        )
    return total, out


_DATE = re.compile(r"(\d{2,4})[-./](\d{1,2})[-./](\d{1,2})")


def _kdate(s: Any) -> datetime | None:
    """Work24 dates ('24-10-02', '2024-10-02', '20241002', '채용시까지 24-10-31') -> 00:00 KST as UTC-aware."""
    t = str(s or "")
    m = _DATE.search(t)
    if m:
        y, mo, d = (int(g) for g in m.groups())
    else:
        m8 = re.search(r"(\d{8})", t)
        if not m8:
            return None
        y, mo, d = int(m8.group(1)[:4]), int(m8.group(1)[4:6]), int(m8.group(1)[6:])
    y = y + 2000 if y < 100 else y
    try:
        return datetime(y, mo, d, tzinfo=UTC) - timedelta(hours=9)
    except ValueError:
        return None


def parse_work24(xml_text: str | bytes) -> tuple[int, list[Posting]]:
    """Work24 채용정보 목록 XML -> (total hits, postings). Raises ``JobsApiError`` for an error answer."""
    root = ElementTree.fromstring(xml_text)
    err = root.findtext(".//error") or ""
    if err.strip():
        raise JobsApiError(f"Work24: {err.strip()}")
    code = (root.findtext(".//messageCd") or "").strip()
    if code and code != "000":
        raise JobsApiError(f"Work24 {code}: {(root.findtext('.//message') or '').strip()}")
    total = int(re.sub(r"\D", "", root.findtext("total") or "0") or 0)
    out = []
    for w in root.iter("wanted"):

        def tx(tag: str, w: ElementTree.Element = w) -> str:
            return (w.findtext(tag) or "").strip()

        close = tx("closeDt")
        expires = _kdate(close)
        out.append(
            Posting(
                source="work24",
                posting_id=tx("wantedAuthNo"),
                company=tx("company"),
                bizr_no=re.sub(r"\D", "", tx("busino")),
                active=True,  # the list serves open postings; closeDt is checked against the snapshot date
                posted=_kdate(tx("regDt")),
                expires=expires + timedelta(days=1) if expires else None,  # closeDt = last day open
                categories=(tx("jobsCd")[:3],) if tx("jobsCd") else (),
                experience={"N": "new", "E": "experienced", "Z": "any"}.get(tx("career")[:1], tx("career")),
            )
        )
    return total, out


# ----------------------------------------------------------------------------------------------- matching


def build_name_index(companies: Mapping[str, Sequence[str]]) -> tuple[dict[str, str], set[str]]:
    """{normalised name: entity} over every entity's names, and the ambiguous names (shared by 2+ entities),
    which are left out of the index."""
    owners: dict[str, set[str]] = {}
    for ent, names in companies.items():
        for n in names:
            k = norm_company_name(n)
            if k:
                owners.setdefault(k, set()).add(ent)
    index = {k: next(iter(v)) for k, v in owners.items() if len(v) == 1}
    return index, {k for k, v in owners.items() if len(v) > 1}


def match_posting(p: Posting, index: Mapping[str, str], bizr: Mapping[str, str]) -> str | None:
    """Entity of a posting: registration number first (when both sides have one), else the exact normalised name."""
    if p.bizr_no and p.bizr_no in bizr:
        return bizr[p.bizr_no]
    return index.get(norm_company_name(p.company))


def query_name(names: Sequence[str]) -> str:
    """Search keyword for a company: its first name without legal-form tokens ('삼성전자(주)' -> '삼성전자')."""
    for n in names:
        q = re.sub(r"\s+", " ", _LEGAL.sub(" ", str(n))).strip()
        if q:
            return q
    return ""


@dataclass
class _Acc:
    """Matched postings of one (entity, source) plus query bookkeeping."""

    postings: dict[str, Posting] = field(default_factory=dict)
    query: str = ""
    hits: int = 0
    excluded: int = 0
    calls: int = 0
    complete: bool = True


def _row(day: date, ent: str, source: str, acc: _Acc, when_utc: datetime) -> dict[str, Any]:
    cut_new = when_utc - timedelta(days=NEW_DAYS)
    active = [p for p in acc.postings.values() if p.active and (p.expires is None or p.expires > when_utc)]
    posted = [p.posted for p in active if p.posted is not None]
    expires = [p.expires for p in active if p.expires is not None]
    cats = Counter(c for p in active for c in p.categories)
    exp = Counter(p.experience for p in active if p.experience)
    return {
        "date": day.isoformat(),
        "entity": ent,
        "source": source,
        "active_postings": len(active),
        "new_7d": sum(1 for p in active if p.posted is not None and p.posted >= cut_new),
        "postings_matched": len(acc.postings),
        "categories": json.dumps(dict(cats.most_common()), ensure_ascii=False),
        "experience": json.dumps(dict(exp.most_common()), ensure_ascii=False),
        "first_posted": min(posted).isoformat() if posted else "",
        "last_posted": max(posted).isoformat() if posted else "",
        "last_expiration": max(expires).isoformat() if expires else "",
        "query": acc.query,
        "query_hits": acc.hits,
        "excluded_other": acc.excluded,
        "complete": acc.complete,
        "api_calls": acc.calls,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------------------------- fetching


class _Budget:
    def __init__(self, calls: int):
        self.left = calls
        self._last: dict[str, float] = {}

    def take(self, source: str) -> None:
        if self.left <= 0:
            raise JobsQuotaError(f"{source}: call budget used up")
        self.left -= 1
        gap = MIN_GAP_S.get(source, 0.0) + self._last.get(source, -1e18) - time.monotonic()
        if gap > 0:
            time.sleep(gap)
        self._last[source] = time.monotonic()


def _saramin_page(session: Any, key: str, budget: _Budget, **params: Any) -> tuple[int, list[Posting]]:
    budget.take("saramin")
    r = session.get(
        SARAMIN_URL,
        params={"access-key": key, "count": SARAMIN_PAGE, "sort": "pd", **params},
        headers={"Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    if "json" not in r.headers.get("Content-Type", ""):
        raise JobsApiError(f"Saramin: JSON 아님 (HTTP {r.status_code}, {r.headers.get('Content-Type')})")
    return parse_saramin(r.json())


def _work24_page(session: Any, key: str, budget: _Budget, **params: Any) -> tuple[int, list[Posting]]:
    budget.take("work24")
    r = session.get(
        WORK24_URL,
        params={"authKey": key, "callTp": "L", "returnType": "XML", "display": WORK24_PAGE, **params},
        timeout=30,
    )
    r.raise_for_status()
    return parse_work24(r.content)


def _paged(
    fetch: Callable[..., tuple[int, list[Posting]]], page_param: str, first_page: int, max_pages: int, **params: Any
) -> tuple[int, list[Posting], int, bool]:
    """All pages of one query: (total, postings, calls, complete). A quota stop after the first page returns what
    was fetched as incomplete; before it, the quota error propagates (the caller stops querying)."""
    total, out, calls, page = 0, [], 0, first_page
    while True:
        try:
            total, rows = fetch(**{page_param: page}, **params)
        except JobsQuotaError:
            if calls == 0:
                raise
            return total, out, calls, False
        calls += 1
        out += rows
        if len(out) >= total or not rows:
            return total, out, calls, len(out) >= total
        if calls >= max_pages:
            return total, out, calls, False
        page += 1


def snapshot_job_postings(
    companies: Mapping[str, Sequence[str]],
    when: date,
    *,
    saramin_key: str = "",
    work24_key: str = "",
    session: Any | None = None,
    bizr_nos: Mapping[str, str] | None = None,
    saramin_mode: SaraminMode = "auto",
    max_calls: int = 450,
    max_pages_per_company: int = 5,
) -> pd.DataFrame:
    """Snapshot of the active postings of ``companies`` ({entity: [names]}) on ``when`` (today: the APIs only know
    the present). One row per (entity, source) whose query ran (``SNAPSHOT_COLUMNS``); sources without a key are
    skipped and the reason is in ``df.attrs['notes']``. ``bizr_nos`` ({entity: 10-digit 사업자등록번호}) adds
    exact registration-number matching. ``max_calls`` caps the API calls per source (Saramin: 500/day/key)."""
    notes: list[str] = []
    rows: list[dict[str, Any]] = []
    if when != date.today():
        notes.append(f"kr_jobs: 스냅샷 날짜 {when} != 오늘 - API는 현재 공고만 제공 (과거 재구성 불가)")
    comp = {str(e): [str(n) for n in ns if str(n).strip()] for e, ns in companies.items()}
    index, ambiguous = build_name_index(comp)
    if ambiguous:
        notes.append(f"kr_jobs: 여러 종목이 공유하는 이름 {len(ambiguous)}개는 매칭 제외: {sorted(ambiguous)[:5]}")
    bizr = {
        re.sub(r"\D", "", str(v)): str(e)
        for e, v in (bizr_nos or {}).items()
        if str(e) in comp and re.sub(r"\D", "", str(v))
    }
    session = session or PoliteSession(timeout=30)
    # "active" / "new in 7 days" are judged at the snapshot moment (today) or at the end of ``when`` (KST)
    now = datetime.now(UTC)
    when_utc = now if when == date.today() else datetime.combine(when, datetime.min.time(), UTC) + timedelta(hours=15)

    if not saramin_key:
        notes.append("kr_jobs: saramin 건너뜀 - access-key 없음 (SARAMIN_API_KEY, https://oapi.saramin.co.kr/)")
    else:
        rows += _snapshot_saramin(
            comp,
            when,
            when_utc,
            index,
            bizr,
            saramin_key,
            session,
            saramin_mode,
            max_calls,
            max_pages_per_company,
            notes,
        )
    if not work24_key:
        notes.append("kr_jobs: work24 건너뜀 - authKey 없음 (WORK24_API_KEY, 고용24 기업회원 OPEN-API)")
    else:
        rows += _snapshot_work24(
            comp, when, when_utc, index, bizr, work24_key, session, max_calls, max_pages_per_company, notes
        )
    df = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    df.attrs["notes"] = notes
    for n in notes:
        log.info(n)
    return df


def _snapshot_saramin(
    comp: Mapping[str, Sequence[str]],
    when: date,
    when_utc: datetime,
    index: Mapping[str, str],
    bizr: Mapping[str, str],
    key: str,
    session: Any,
    mode: SaraminMode,
    max_calls: int,
    max_pages: int,
    notes: list[str],
) -> list[dict[str, Any]]:
    budget = _Budget(min(max_calls, SARAMIN_DAILY_LIMIT))
    fetch = partial(_saramin_page, session, key, budget)
    sweep = mode == "sweep" or (mode == "auto" and len(comp) > SWEEP_MIN_COMPANIES)
    if sweep:
        acc = {e: _Acc(query=f"sweep stock={SARAMIN_STOCKS}") for e in comp}
        excluded = 0
        try:
            total, posts, calls, complete = _paged(fetch, "start", 0, budget.left, stock=SARAMIN_STOCKS)
        except JobsApiError as exc:
            notes.append(f"kr_jobs: saramin sweep 실패 - {exc}")
            return []
        for p in posts:
            ent = match_posting(p, index, bizr)
            if ent is None:
                excluded += 1
            else:
                acc[ent].postings[p.posting_id] = p
        if not complete:
            notes.append(f"kr_jobs: saramin sweep 미완료 ({len(posts)}/{total}건, {calls}회) - complete=false")
        for a in acc.values():
            a.hits, a.excluded, a.calls, a.complete = total, excluded, calls, complete
        return [_row(when, e, "saramin", a, when_utc) for e, a in acc.items()]
    out = []
    failed = 0
    for ent, names in comp.items():
        q = query_name(names)
        if not q:
            continue
        a = _Acc(query=f"keywords={q} stock={SARAMIN_STOCKS}")
        try:
            a.hits, posts, a.calls, a.complete = _paged(fetch, "start", 0, max_pages, keywords=q, stock=SARAMIN_STOCKS)
        except JobsQuotaError as exc:
            notes.append(f"kr_jobs: saramin 한도 도달 ({exc}) - 남은 {len(comp) - len(out)}개 종목 미조회")
            break
        except Exception as exc:  # noqa: BLE001 - one company failing never stops the rest; counted
            failed += 1
            log.warning("kr_jobs saramin %s 실패: %s", ent, exc)
            continue
        for p in posts:
            if match_posting(p, index, bizr) == ent:
                a.postings[p.posting_id] = p
            else:
                a.excluded += 1
        out.append(_row(when, ent, "saramin", a, when_utc))
    if failed:
        notes.append(f"kr_jobs: saramin {failed}개 종목 조회 실패")
    return out


def _snapshot_work24(
    comp: Mapping[str, Sequence[str]],
    when: date,
    when_utc: datetime,
    index: Mapping[str, str],
    bizr: Mapping[str, str],
    key: str,
    session: Any,
    max_calls: int,
    max_pages: int,
    notes: list[str],
) -> list[dict[str, Any]]:
    budget = _Budget(max_calls)
    fetch = partial(_work24_page, session, key, budget)
    out = []
    failed = 0
    for ent, names in comp.items():
        q = query_name(names)
        if not q:
            continue
        a = _Acc(query=f"coNm={q}")
        try:
            a.hits, posts, a.calls, a.complete = _paged(fetch, "startPage", 1, max_pages, coNm=q)
        except JobsQuotaError as exc:
            notes.append(f"kr_jobs: work24 호출 한도 ({exc}) - 남은 {len(comp) - len(out)}개 종목 미조회")
            break
        except JobsApiError as exc:
            notes.append(f"kr_jobs: work24 오류 - {exc} (중단)")
            break
        except Exception as exc:  # noqa: BLE001 - one company failing never stops the rest; counted
            failed += 1
            log.warning("kr_jobs work24 %s 실패: %s", ent, exc)
            continue
        for p in posts:
            if match_posting(p, index, bizr) == ent:
                a.postings[p.posting_id] = p
            else:
                a.excluded += 1
        out.append(_row(when, ent, "work24", a, when_utc))
    if failed:
        notes.append(f"kr_jobs: work24 {failed}개 종목 조회 실패")
    return out


# ------------------------------------------------------------------------------------------------ ledger


def _ledger_keys(path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{n}: JSON 아님 ({exc}) - 원장 수동 확인 필요") from exc
            keys.add((str(r["date"]), str(r["entity"]), str(r["source"])))
    return keys


def _jsonable(v: Any) -> Any:
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):  # numpy scalars
        return v.item()
    return v


def append_snapshot(df: pd.DataFrame, path: Path = DEFAULT_LEDGER) -> int:
    """Append the rows of ``df`` whose (date, entity, source) is not in the ledger yet; returns rows written.
    The file is only ever appended to (never rewritten), so the committed history cannot change."""
    if df is None or df.empty:
        return 0
    path = Path(path)
    have = _ledger_keys(path)
    lines = []
    for rec in df[SNAPSHOT_COLUMNS].to_dict("records"):
        k = (str(rec["date"]), str(rec["entity"]), str(rec["source"]))
        if k in have:
            continue
        have.add(k)
        lines.append(json.dumps({c: _jsonable(rec[c]) for c in SNAPSHOT_COLUMNS}, ensure_ascii=False, sort_keys=False))
    if not lines:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return len(lines)


def read_ledger(path: Path = DEFAULT_LEDGER) -> pd.DataFrame:
    """The whole ledger as a frame (``SNAPSHOT_COLUMNS``); empty when the file does not exist."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(recs, columns=SNAPSHOT_COLUMNS)


def jobs_measures(path: Path = DEFAULT_LEDGER, sources: Iterable[str] = SOURCES) -> AltData:
    """Ledger -> KR measures ``job_postings_active`` / ``job_postings_new_7d`` (source ``kr_jobs_<api>``),
    ``published`` = ``period_end`` = snapshot date. Incomplete snapshots are dropped and counted.
    ``covered['kr_jobs_<api>']`` = entities with a complete row on every snapshot date of that API."""
    df = read_ledger(path)
    sources = tuple(sources)
    notes: list[str] = []
    cov: dict[str, Any] = {}
    covered: dict[str, list[str]] = {}
    if df.empty:
        return AltData(
            "KR", measures=empty_measures(), coverage={"kr_jobs": {"rows": 0}}, notes=["kr_jobs: 원장 없음/빈 원장"]
        )
    df = df[df["source"].isin(sources)]
    complete = df["complete"].astype(bool)
    if (~complete).any():
        notes.append(f"kr_jobs: 미완료 스냅샷 {int((~complete).sum())}행 제외")
    ok = df[complete]
    recs = []
    for item, col in ITEM_COLUMN.items():
        part = ok[["entity", "date", "source", col]].rename(columns={col: "value"})
        recs.append(
            part.assign(
                published=pd.to_datetime(part["date"]),
                period_end=pd.to_datetime(part["date"]),
                item=item,
                source="kr_jobs_" + part["source"],
            )
        )
    measures = normalise_measures(pd.concat(recs, ignore_index=True)) if recs else empty_measures()
    for src in sources:
        s = ok[ok["source"] == src]
        days = sorted(set(df.loc[df["source"] == src, "date"]))
        n_days = s.groupby("entity")["date"].nunique()
        covered[f"kr_jobs_{src}"] = sorted(n_days[n_days == len(days)].index.astype(str)) if days else []
        cov[f"kr_jobs_{src}"] = {
            "snapshot_days": len(days),
            "first_day": days[0] if days else "",
            "last_day": days[-1] if days else "",
            "entities": int(s["entity"].nunique()),
            "rows": len(s),
            "incomplete_rows": int(((df["source"] == src) & ~complete).sum()),
        }
        if days and len(days) < 250:
            notes.append(f"kr_jobs_{src}: 스냅샷 {len(days)}일 - 12개월 누적 전에는 모델에 쓰지 말 것")
    return AltData("KR", measures=measures, coverage=cov, notes=notes, covered=covered)


__all__ = [
    "DEFAULT_LEDGER",
    "ITEMS",
    "SARAMIN_URL",
    "SNAPSHOT_COLUMNS",
    "WORK24_URL",
    "JobsApiError",
    "JobsQuotaError",
    "Posting",
    "append_snapshot",
    "build_name_index",
    "jobs_measures",
    "match_posting",
    "parse_saramin",
    "parse_work24",
    "query_name",
    "read_ledger",
    "snapshot_job_postings",
]
