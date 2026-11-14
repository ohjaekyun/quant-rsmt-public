"""KR alternative data from OpenDART (docs/MODELING.md section 16): disclosure events, insider and 5% holder
ownership changes, employee status. Requires ``DART_API_KEY`` (settings.data.dart_api_key).

Sources (all point in time by the DART receipt date ``rcept_dt``; the shared rule of ``quantcio.data.alt`` then
uses a row only on grid dates STRICTLY after that date):

- ``list.json`` per company for the publication types B (주요사항보고), I (거래소공시) and E (기타공시), 2013-01-01
  onwards. Type D (지분공시: insider / 5% reports - thousands per large company) is skipped here because the two
  ownership APIs below carry the same filings with their numbers; type A (periodic reports) is already the
  fundamentals' publication table. Report names are classified into the pre-declared ``LIST_KINDS`` by
  ``classify_report`` (prefixes such as [기재정정] are stripped first; a correction also counts as ``correction``).
- ``elestock.json`` (임원ㆍ주요주주 특정증권등 소유상황보고서): every report of the company in one call. Events
  ``insider_net_shares`` (signed share change), ``insider_buy_n`` / ``insider_sell_n`` (one per report with a
  positive / negative change). Grants and option exercises cannot be told apart from market trades here.
- ``majorstock.json`` (주식등의 대량보유상황보고서, 5% rule): ``block_pct_change`` (signed change of the
  holder's stake in %p), ``block_up_n`` / ``block_down_n``.
- ``empSttus.json`` (직원 현황) of every annual report (reprt_code 11011) for business years ``EMP_FIRST_YEAR``
  onwards: measures ``headcount``, ``contract_workers``, ``avg_tenure_years`` (headcount-weighted),
  ``salary_total`` (KRW). ``published`` = the receipt date in ``rcept_no`` (first 8 digits), ``period_end`` =
  ``stlm_dt``.

API budget: OpenDART allows 20,000 calls per key per day (status 020 beyond). Every answer is cached on disk under
``<cache>/alt/kr_dart`` and the loader fetches in priority order (ownership -> lists -> employee years), so a run
that hits the quota stops cleanly, reports what is missing (``coverage``/``notes``) and the next run resumes.
Closed history is never refetched; open ranges are refreshed when older than ``refresh_days``.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quantcio.config import Settings
from quantcio.data.alt import AltData, empty_events, empty_measures, normalise_events, normalise_measures
from quantcio.data.http import PoliteSession

log = logging.getLogger(__name__)

OPENDART_API = "https://opendart.fss.or.kr/api"
LIST_START = date(2013, 1, 1)
LIST_TYPES = ("B", "I", "E")  # 주요사항보고, 거래소공시, 기타공시
EMP_FIRST_YEAR = 2014  # business year of the first annual report used (YoY needs the year before 2016)
EMP_REPORT_CODE = "11011"  # annual report
EMP_DEADLINE = (6, 30)  # an annual report of year Y missing after Y+1-06-30 is final (no data)
SOURCE = "kr_dart"

# Pre-declared event kinds of list.json (docs/MODELING.md section 16.2). Order matters: the first match wins.
_LIST_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("supply_contract_cancel", re.compile(r"단일판매ㆍ?공급계약해지")),
    ("supply_contract", re.compile(r"단일판매ㆍ?공급계약체결")),
    ("buyback", re.compile(r"자기주식취득결정|자기주식취득신탁계약체결결정")),
    ("treasury_disposal", re.compile(r"자기주식처분결정|자기주식취득신탁계약해지결정")),
    ("share_cancel", re.compile(r"주식소각결정")),
    ("rights_issue", re.compile(r"유상증자결정|유무상증자결정")),
    ("bonus_issue", re.compile(r"무상증자결정")),
    ("convertible_issue", re.compile(r"전환사채권발행결정|신주인수권부사채권발행결정|교환사채권발행결정")),
    ("capital_reduction", re.compile(r"감자결정")),
    (
        "ma_deal",
        re.compile(
            r"회사합병결정|회사분할결정|회사분할합병결정|영업양수결정|타법인주식및출자증권취득결정|"
            r"유형자산양수결정|주식교환ㆍ?이전결정|주식의포괄적교환ㆍ?이전결정"
        ),
    ),
    ("asset_sale", re.compile(r"영업양도결정|타법인주식및출자증권처분결정|유형자산양도결정")),
    ("lawsuit", re.compile(r"소송등의제기|소송등의판결")),
    ("major_holder_change", re.compile(r"최대주주변경")),
    ("ceo_change", re.compile(r"대표이사변경")),
    ("unfaithful", re.compile(r"불성실공시법인")),
    ("inquiry", re.compile(r"조회공시요구|풍문또는보도에대한해명")),
    (
        "admin_issue",
        re.compile(
            r"관리종목|상장적격성|매매거래정지|투자주의환기|상장폐지|감사의견|횡령ㆍ?배임|회생절차|부도발생|해산사유"
        ),
    ),
    ("earnings_prelim", re.compile(r"영업\(?잠정\)?실적|잠정\)?실적")),
    ("profit_change", re.compile(r"매출액또는손익구조")),
    ("dividend", re.compile(r"현금ㆍ?현물배당결정")),
    ("ir", re.compile(r"기업설명회")),
    ("new_facility", re.compile(r"신규시설투자")),
    ("debt_guarantee", re.compile(r"채무보증결정")),
    ("borrowing", re.compile(r"단기차입금증가결정|사채발행결정")),
    ("stock_option", re.compile(r"주식매수선택권부여")),
]
LIST_KINDS = (
    *(k for k, _ in _LIST_RULES),
    "correction",  # a report whose name carries a correction prefix ([기재정정], [첨부정정], ...)
    "filing",  # every B / I / E filing (disclosure intensity)
)
OWNERSHIP_KINDS = (
    "insider_net_shares",
    "insider_buy_n",
    "insider_sell_n",
    "block_pct_change",
    "block_up_n",
    "block_down_n",
)
KINDS = (*LIST_KINDS, *OWNERSHIP_KINDS)
ITEMS = ("headcount", "contract_workers", "avg_tenure_years", "salary_total")

_PREFIX = re.compile(r"^\s*(\[[^\]]*\]\s*)+")
_CORRECTION = re.compile(r"정정")


class DartQuotaError(RuntimeError):
    """OpenDART answered status 020 (daily call limit): stop fetching, keep what is cached."""


class DartNetworkError(RuntimeError):
    """OpenDART unreachable for longer than ``NETWORK_PATIENCE_S`` (e.g. the machine slept): stop, keep the cache."""


NETWORK_PATIENCE_S = 15 * 60  # wait this long (backoff 30 s .. 5 min) for the network before giving up


# ================================================================================================ parsing (pure)


def _num(x: Any) -> float:
    """'1,234' / '-100' / '-' / '' -> float (NaN when not a number)."""
    s = str(x if x is not None else "").replace(",", "").strip()
    if s in ("", "-", "--"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _date8(s: Any) -> pd.Timestamp | None:
    t = re.sub(r"[^0-9]", "", str(s or ""))[:8]
    if len(t) != 8:
        return None
    try:
        return pd.Timestamp(datetime.strptime(t, "%Y%m%d"))
    except ValueError:
        return None


def classify_report(report_nm: str) -> tuple[str | None, bool]:
    """(kind of ``LIST_KINDS`` or None, is_correction) of one list.json report name."""
    nm = re.sub(r"\s+", "", str(report_nm or ""))
    m = _PREFIX.match(nm)
    corr = bool(m and _CORRECTION.search(m.group(0)))
    body = nm[m.end() :] if m else nm
    for kind, rx in _LIST_RULES:
        if rx.search(body):
            return kind, corr
    return None, corr


def list_events(entity: str, rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """list.json rows (rcept_no, rcept_dt, report_nm) of one company -> events (deduplicated by rcept_no)."""
    out: list[tuple[str, pd.Timestamp, str, float, str]] = []
    seen: set[str] = set()
    for r in rows:
        rno = str(r.get("rcept_no") or "")
        if rno and rno in seen:
            continue
        seen.add(rno)
        d = _date8(r.get("rcept_dt"))
        if d is None:
            continue
        kind, corr = classify_report(str(r.get("report_nm") or ""))
        out.append((entity, d, "filing", 1.0, f"{SOURCE}_list"))
        if kind is not None:
            out.append((entity, d, kind, 1.0, f"{SOURCE}_list"))
        if corr:
            out.append((entity, d, "correction", 1.0, f"{SOURCE}_list"))
    if not out:
        return empty_events()
    return pd.DataFrame(out, columns=["entity", "published", "kind", "value", "source"])


def insider_events(entity: str, rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """elestock.json rows -> insider_net_shares / insider_buy_n / insider_sell_n events (zero changes skipped)."""
    out = []
    for r in rows:
        d = _date8(r.get("rcept_dt"))
        chg = _num(r.get("sp_stock_lmp_irds_cnt"))
        if d is None or not (chg == chg) or chg == 0:  # NaN or no change
            continue
        out.append((entity, d, "insider_net_shares", chg, f"{SOURCE}_insider"))
        out.append((entity, d, "insider_buy_n" if chg > 0 else "insider_sell_n", 1.0, f"{SOURCE}_insider"))
    if not out:
        return empty_events()
    return pd.DataFrame(out, columns=["entity", "published", "kind", "value", "source"])


def block_events(entity: str, rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """majorstock.json rows -> block_pct_change / block_up_n / block_down_n events (zero changes skipped)."""
    out = []
    for r in rows:
        d = _date8(r.get("rcept_dt"))
        chg = _num(r.get("stkrt_irds"))
        if d is None or not (chg == chg) or chg == 0:
            continue
        out.append((entity, d, "block_pct_change", chg, f"{SOURCE}_block"))
        out.append((entity, d, "block_up_n" if chg > 0 else "block_down_n", 1.0, f"{SOURCE}_block"))
    if not out:
        return empty_events()
    return pd.DataFrame(out, columns=["entity", "published", "kind", "value", "source"])


_TOTAL_ROW = re.compile(r"(합계|총계|^계$|^소계$|^전체$|^total$)", re.IGNORECASE)


def _is_total(v: Any) -> bool:
    return bool(_TOTAL_ROW.search(str(v or "").strip()))


def employee_measures(entity: str, rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """empSttus.json rows of ONE report -> headcount / contract_workers / avg_tenure_years / salary_total.

    The API returns one row per (division ``fo_bbm``, sex ``sexdstn``); most companies add total rows such as
    ``fo_bbm = 성별합계`` per sex, and some a grand total. Rule: a row that is a total in both fields is the grand
    total and is used alone; else the division-total rows (one per sex) are summed; with no total rows the detail
    rows are summed. Tenure is the headcount-weighted mean of the rows used.
    """
    rows = list(rows)
    if not rows:
        return empty_measures()
    rno = str(rows[0].get("rcept_no") or "")
    pub = _date8(rno[:8])
    pe = _date8(rows[0].get("stlm_dt"))
    if pub is None or pe is None:
        return empty_measures()
    div_tot = [r for r in rows if _is_total(r.get("fo_bbm"))]
    sex_tot = [r for r in rows if _is_total(r.get("sexdstn"))]
    grand = [r for r in div_tot if _is_total(r.get("sexdstn"))]
    if grand:
        use = grand[:1]
    elif div_tot:  # one total per sex (fo_bbm = 성별합계)
        use = div_tot
    elif sex_tot:  # one total per division (sexdstn = 합계)
        use = sex_tot
    else:
        use = rows
    hc = sum(v for v in (_num(r.get("sm")) for r in use) if v == v)
    if not hc or hc <= 0:
        return empty_measures()
    cont = sum(v for v in (_num(r.get("cnttk_co")) for r in use) if v == v)
    sal = [v for v in (_num(r.get("fyer_salary_totamt")) for r in use) if v == v]
    ten_w = [(_num(r.get("avrg_cnwk_sdytrn")), _num(r.get("sm"))) for r in use]
    ten_w = [(t, w) for t, w in ten_w if t == t and w == w and w > 0]
    vals = {"headcount": hc, "contract_workers": cont}
    if ten_w:
        vals["avg_tenure_years"] = sum(t * w for t, w in ten_w) / sum(w for _, w in ten_w)
    if sal:
        vals["salary_total"] = float(sum(sal))
    out = [(entity, pub, pe, k, float(v), f"{SOURCE}_emp") for k, v in vals.items()]
    return pd.DataFrame(out, columns=["entity", "published", "period_end", "item", "value", "source"])


def parse_corp_list(content: bytes) -> pd.DataFrame:
    """OpenDART corpCode.xml zip -> every corp (corp_code, corp_name, stock_code, modify_date), listed or not."""
    from xml.etree import ElementTree

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        root = ElementTree.fromstring(z.read(z.namelist()[0]))
    rows = [
        (
            (e.findtext("corp_code") or "").strip(),
            (e.findtext("corp_name") or "").strip(),
            (e.findtext("stock_code") or "").strip(),
            (e.findtext("modify_date") or "").strip(),
        )
        for e in root.iter("list")
    ]
    return pd.DataFrame(rows, columns=["corp_code", "corp_name", "stock_code", "modify_date"])


def norm_corp_name(s: Any) -> str:
    t = re.sub(r"\(주\)|㈜|주식회사|\(株\)|co\.?,?\s*ltd\.?|inc\.?|corp\.?", "", str(s or ""), flags=re.IGNORECASE)
    return re.sub(r"[\s\.\,\-\(\)\[\]·ㆍ&']", "", t).lower()


def map_corp_codes(
    entities: Iterable[str], corps: pd.DataFrame, names: Mapping[str, Iterable[str]] | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    """({entity: corp_code}, {entity: how}) - by stock code, else by a UNIQUE normalised-name match among corps
    without a stock code (delisted companies lose it), else unmapped (counted by the caller)."""
    by_code = {r.stock_code: r.corp_code for r in corps.itertuples(index=False) if r.stock_code}
    unlisted = corps[corps["stock_code"] == ""]
    by_name: dict[str, list[str]] = {}
    for r in unlisted.itertuples(index=False):
        by_name.setdefault(norm_corp_name(r.corp_name), []).append(r.corp_code)
    out: dict[str, str] = {}
    how: dict[str, str] = {}
    for e in entities:
        e = str(e)
        if e in by_code:
            out[e], how[e] = by_code[e], "stock_code"
            continue
        cands = set()
        for nm in (names or {}).get(e, []) or []:
            cands.update(by_name.get(norm_corp_name(nm), []))
        if len(cands) == 1:
            out[e], how[e] = cands.pop(), "name"
        else:
            how[e] = "ambiguous_name" if cands else "unmapped"
    return out, how


# ================================================================================================ client + cache


class DartAltClient:
    """OpenDART JSON calls with a call counter, the daily-quota stop and a JSON file cache."""

    def __init__(self, settings: Settings, session: Any = None, root: Path | None = None, today: date | None = None):
        self.key = settings.data.dart_api_key
        if not self.key:
            raise RuntimeError("DART_API_KEY가 없습니다 (OpenDART 대안 데이터)")
        self.session = session or PoliteSession()
        self.root = Path(root) if root is not None else settings.cache_path / "alt" / "kr_dart"
        self.today = today or date.today()
        self.calls = 0
        self.quota_hit = False
        self.network_down = False
        self.sleep = time.sleep  # injectable (tests)

    def redact(self, text: Any) -> str:
        """``text`` with the API key masked (exceptions of requests carry the full URL)."""
        return str(text).replace(self.key, "***") if self.key else str(text)

    def _http(self, url: str, params: dict[str, Any], timeout: float) -> Any:
        """GET with patience for connection errors / timeouts; DartNetworkError after NETWORK_PATIENCE_S."""
        waited, pause = 0.0, 30.0
        while True:
            try:
                return self.session.get(url, params=params, timeout=timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if waited >= NETWORK_PATIENCE_S:
                    self.network_down = True
                    raise DartNetworkError(f"OpenDART unreachable for {waited:.0f}s: {self.redact(exc)}") from None
                log.warning("OpenDART unreachable (%s) - retrying in %.0fs", type(exc).__name__, pause)
                self.sleep(pause)
                waited += pause
                pause = min(pause * 2, 300.0)

    # ------------------------------------------------------------------ io
    def get(self, api: str, **params: Any) -> dict[str, Any]:
        """GET <api>.json -> JSON with status 000 or 013 (no data). 020 -> DartQuotaError."""
        if self.quota_hit:
            raise DartQuotaError("OpenDART daily quota already hit in this run")
        if self.network_down:
            raise DartNetworkError("OpenDART unreachable earlier in this run")
        r = self._http(f"{OPENDART_API}/{api}.json", {"crtfc_key": self.key, **params}, 30)
        self.calls += 1
        r.raise_for_status()
        if "json" not in r.headers.get("Content-Type", ""):
            raise RuntimeError(f"OpenDART {api}: not JSON (key?)")
        j = r.json()
        st = str(j.get("status"))
        if st == "020":
            self.quota_hit = True
            raise DartQuotaError(f"OpenDART {api}: 020 {j.get('message')}")
        if st not in ("000", "013"):
            raise RuntimeError(f"OpenDART {api}: {st} {j.get('message')}")
        return j

    def corp_list(self, max_age_days: int = 7) -> pd.DataFrame:
        path = self.root / "corp_list.parquet"
        if path.exists() and (datetime.now().timestamp() - path.stat().st_mtime) < max_age_days * 86400:
            return pd.read_parquet(path)
        r = self._http(f"{OPENDART_API}/corpCode.xml", {"crtfc_key": self.key}, 60)
        self.calls += 1
        r.raise_for_status()
        if r.content[:2] != b"PK":
            raise RuntimeError("OpenDART corpCode.xml: not a zip (key?)")
        df = parse_corp_list(r.content)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return df

    def _read(self, rel: str) -> dict[str, Any] | None:
        p = self.root / rel
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write(self, rel: str, obj: dict[str, Any]) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    def _stale(self, obj: Mapping[str, Any] | None, refresh_days: int | None) -> bool:
        if obj is None:
            return True
        if refresh_days is None:
            return False
        fetched = date.fromisoformat(str(obj.get("fetched")))
        return (self.today - fetched).days >= refresh_days

    # ------------------------------------------------------------------ sources
    def disclosures(self, corp: str, ty: str, refresh_days: int | None) -> list[dict[str, Any]]:
        """list.json rows of one company and publication type since LIST_START (incremental refresh)."""
        rel = f"list/{corp}/{ty}.json"
        obj = self._read(rel)
        if not self._stale(obj, refresh_days):
            return obj["rows"]  # type: ignore[index]
        rows = {r["rcept_no"]: r for r in (obj or {}).get("rows", [])}
        start = LIST_START if obj is None else date.fromisoformat(obj["fetched"]) - timedelta(days=7)
        page = 1
        while True:
            j = self.get(
                "list",
                corp_code=corp,
                pblntf_ty=ty,
                bgn_de=start.strftime("%Y%m%d"),
                end_de=self.today.strftime("%Y%m%d"),
                page_no=page,
                page_count=100,
            )
            for x in j.get("list") or []:
                rows[str(x.get("rcept_no"))] = {
                    "rcept_no": str(x.get("rcept_no")),
                    "rcept_dt": str(x.get("rcept_dt")),
                    "report_nm": str(x.get("report_nm") or ""),
                }
            if page >= int(j.get("total_page") or 1):
                break
            page += 1
        out = sorted(rows.values(), key=lambda r: r["rcept_no"])
        self._write(rel, {"fetched": self.today.isoformat(), "rows": out})
        return out

    def whole_history(self, api: str, corp: str, refresh_days: int | None) -> list[dict[str, Any]]:
        """elestock / majorstock: the company's reports in one call. The API returns only a rolling ~2-year window
        (first answers start 2024-10-04), so every answer is MERGED into the cache (keyed by rcept_no + reporter):
        history accumulates from the first fetch on (docs/MODELING.md 16.8)."""
        rel = f"{api}/{corp}.json"
        obj = self._read(rel)
        if not self._stale(obj, refresh_days):
            return obj["rows"]  # type: ignore[index]
        j = self.get(api, corp_code=corp)

        def key(r: Mapping[str, Any]) -> str:
            return "|".join(str(r.get(k) or "") for k in ("rcept_no", "repror", "isu_exctv_ofcps", "stkqy_irds"))

        merged = {key(r): r for r in (obj or {}).get("rows", [])}
        merged.update({key(r): r for r in (j.get("list") or [])})
        rows = sorted(merged.values(), key=lambda r: (str(r.get("rcept_dt") or ""), str(r.get("rcept_no") or "")))
        first = (obj or {}).get("first_fetched") or self.today.isoformat()
        self._write(rel, {"fetched": self.today.isoformat(), "first_fetched": first, "rows": rows})
        return rows

    def employees(self, corp: str, year: int) -> list[dict[str, Any]] | None:
        """empSttus rows of the annual report of ``year`` ([] = final no-data, None = not yet available)."""
        rel = f"emp/{corp}/{year}.json"
        obj = self._read(rel)
        deadline = date(year + 1, *EMP_DEADLINE)
        if obj is not None and (obj.get("rows") or date.fromisoformat(obj["fetched"]) > deadline):
            return obj.get("rows") or []
        if self.today <= date(year + 1, 1, 1):
            return None
        j = self.get("empSttus", corp_code=corp, bsns_year=str(year), reprt_code=EMP_REPORT_CODE)
        rows = list(j.get("list") or [])
        self._write(rel, {"fetched": self.today.isoformat(), "status": j.get("status"), "rows": rows})
        if not rows and self.today <= deadline:
            return None
        return rows


# ================================================================================================ loader


def load_kr_dart_alt(
    settings: Settings,
    entities: Iterable[str],
    start: date,
    end: date | None = None,
    *,
    names: Mapping[str, Iterable[str]] | None = None,
    refresh_days: int | None = 7,
    employees: bool = True,
    client: DartAltClient | None = None,
    progress: Callable[[str], None] | None = None,
) -> AltData:
    """OpenDART alternative data of ``entities`` (6-digit stock codes) published in [start, end].

    ``names``: {entity: [company names]} used to map delisted entities whose stock code DART no longer lists.
    ``refresh_days``: cached open ranges older than this are refreshed (None = use the cache as is; a run that
    must not spend API calls). Fetch order: ownership (2 calls/company) -> lists (3 types) -> employee years.
    """
    say = progress or (lambda _m: None)
    end = end or date.today()
    ents = sorted({str(e) for e in entities})
    cov: dict[str, Any] = {"entities": len(ents)}
    notes: list[str] = []
    try:
        cl = client or DartAltClient(settings)
    except RuntimeError as exc:
        notes.append(f"DART 대안 데이터 없음: {exc}")
        return AltData("KR", coverage={SOURCE: {**cov, "error": str(exc)}}, notes=notes)
    try:
        corps = cl.corp_list()
    except Exception as exc:  # noqa: BLE001 - nothing can be mapped; recorded
        notes.append(f"DART corpCode 로드 실패: {type(exc).__name__}: {cl.redact(exc)}")
        return AltData("KR", coverage={SOURCE: {**cov, "error": str(exc)}}, notes=notes)
    cmap, how = map_corp_codes(ents, corps, names)
    cov["mapped"] = len(cmap)
    cov["mapped_by"] = pd.Series(how).value_counts().to_dict() if how else {}
    unmapped = [e for e in ents if e not in cmap]
    if unmapped:
        notes.append(f"DART corp_code 미매칭 {len(unmapped)}개 엔티티(대안 데이터 결측): {', '.join(unmapped[:12])}")
    ev_parts: list[pd.DataFrame] = []
    me_parts: list[pd.DataFrame] = []
    failures: dict[str, int] = {}
    missing: dict[str, int] = {}

    def fail(stage: str, exc: Exception) -> None:
        failures[stage] = failures.get(stage, 0) + 1
        log.warning("kr dart alt %s: %s", stage, cl.redact(exc))

    stages: list[tuple[str, Callable[[str, str], None]]] = []
    covered: dict[str, list[str]] = {f"{SOURCE}_{k}": [] for k in ("insider", "block", "list", "emp")}

    def own(e: str, corp: str) -> None:
        ins = insider_events(e, cl.whole_history("elestock", corp, refresh_days))
        blk = block_events(e, cl.whole_history("majorstock", corp, refresh_days))
        ev_parts.extend([ins, blk])
        covered[f"{SOURCE}_insider"].append(e)
        covered[f"{SOURCE}_block"].append(e)

    def lists(e: str, corp: str) -> None:
        parts = [list_events(e, cl.disclosures(corp, ty, refresh_days)) for ty in LIST_TYPES]
        ev_parts.extend(parts)  # only after all three types were read: a partial company is not "covered"
        covered[f"{SOURCE}_list"].append(e)

    def emp(e: str, corp: str) -> None:
        last = end.year - 1
        parts = []
        for y in range(EMP_FIRST_YEAR, last + 1):
            rows = cl.employees(corp, y)
            if rows:
                parts.append(employee_measures(e, rows))
        me_parts.extend(parts)
        covered[f"{SOURCE}_emp"].append(e)

    stages.append(("ownership", own))
    stages.append(("lists", lists))
    if employees:
        stages.append(("employees", emp))
    for stage, fn in stages:
        say(f"DART {stage}: {len(cmap)} companies (calls so far {cl.calls})")
        for i, (e, corp) in enumerate(sorted(cmap.items()), start=1):
            if cl.quota_hit or cl.network_down:
                missing[stage] = missing.get(stage, 0) + 1
                continue
            try:
                fn(e, corp)
            except (DartQuotaError, DartNetworkError):
                missing[stage] = missing.get(stage, 0) + 1
            except Exception as exc:  # noqa: BLE001 - one company failing never stops the rest; counted
                fail(stage, exc)
            if i % 100 == 0:
                say(f"DART {stage}: {i}/{len(cmap)} (calls {cl.calls})")
    events = normalise_events(pd.concat(ev_parts, ignore_index=True) if ev_parts else None)
    measures = normalise_measures(pd.concat(me_parts, ignore_index=True) if me_parts else None)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    events = events[(events["published"] >= lo) & (events["published"] <= hi)].reset_index(drop=True)
    measures = measures[(measures["published"] <= hi)].reset_index(drop=True)  # older reports feed YoY
    cov.update(
        {
            "calls": cl.calls,
            "quota_hit": cl.quota_hit,
            "network_down": cl.network_down,
            "failures": failures,
            "missing_due_to_quota": missing,
            "event_rows": len(events),
            "measure_rows": len(measures),
            "entities_with_events": int(events["entity"].nunique()) if len(events) else 0,
            "entities_with_headcount": int(measures.loc[measures["item"] == "headcount", "entity"].nunique())
            if len(measures)
            else 0,
        }
    )
    if cl.quota_hit:
        notes.append(
            "OpenDART 일일 호출 한도(020)에 도달 — 일부 회사의 대안 데이터가 비어 있습니다. "
            "내일 다시 실행하면 캐시에서 이어 받습니다"
            f" ({', '.join(f'{k} {v}' for k, v in missing.items())})"
        )
    if cl.network_down:
        notes.append(
            "OpenDART 연결 불가(네트워크) — 일부 회사의 대안 데이터가 비어 있습니다. "
            "다시 실행하면 캐시에서 이어 받습니다"
            f" ({', '.join(f'{k} {v}' for k, v in missing.items())})"
        )
    if failures:
        notes.append(f"DART 대안 데이터 일부 실패: {', '.join(f'{k} {v}' for k, v in failures.items())}")
    cov["covered"] = {k: len(v) for k, v in covered.items()}
    return AltData("KR", events, measures, {SOURCE: cov}, notes, covered)


__all__ = [
    "EMP_FIRST_YEAR",
    "ITEMS",
    "KINDS",
    "LIST_KINDS",
    "LIST_TYPES",
    "OWNERSHIP_KINDS",
    "DartAltClient",
    "DartNetworkError",
    "DartQuotaError",
    "block_events",
    "classify_report",
    "employee_measures",
    "insider_events",
    "list_events",
    "load_kr_dart_alt",
    "map_corp_codes",
    "norm_corp_name",
    "parse_corp_list",
]
