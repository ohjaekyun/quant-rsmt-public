"""KR point-in-time fundamentals: OpenDART bulk values + DART receipt dates + as-first-reported XBRL.

Design (probe + verifier, docs/sources/pit/kr-fundamentals.json):
1. VALUES - OpenDART bulk zips (BS/PL/CF, FY2015 and 2016Q1..latest), consolidated-else-separate, key items
   extracted as filed (kr_bulk.py). Non-KRW filers (두산밥캣 USD, 900xxx CNY, JTC JPY ...) are converted to KRW with
   Yahoo FX (period-end rate for balance-sheet items, period-average rate for flows) or excluded with a note.
2. TTM - FY rows: the fiscal-year value; quarter rows: YTD + previous FY - previous year's same-quarter YTD (all
   three reports on the same statement basis). Standalone quarters (for YoY) come from quarterly_standalone.
3. PUBLICATION DATE - ``available_from`` = first KRX session strictly AFTER the ORIGINAL report's receipt date
   (DART web search keyless, or OpenDART list.json with DART_API_KEY). Unknown date: period_end + 60 days (quarter
   / half) or + 120 days (annual), rolled to a session, then the next session.
4. RESTATEMENTS - the bulk carries the LATEST corrected values. For a period corrected ('[기재정정]') more than 7
   days after its original, the original filing's XBRL is fetched and used until the correction's next session;
   a second row with the corrected values is emitted when it becomes available before the next period. Short
   corrections (<= 7 days) simply delay available_from to the correction. Components of TTM / YoY follow the same
   rule as of each row's available_from. Values that could not be verified are flagged (lookahead_risk) and
   counted, never silently mixed.
5. UNIVERSE LINK - bulk rows are keyed by stock code; each row is mapped to the listing spell (entity) that is
   listed at (or next listed after) its available_from. Rows available only after the entity's delisting are
   dropped and counted (the bulk keeps post-delisting 기타법인 filings).
Never use EPS / BPS for valuation (as-reported, not split-adjusted; scale errors): callers compute E/P, B/P from
market cap (pit_market_cap_kr) and ni_parent_ttm / equity_parent.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantcio.config import Settings
from quantcio.data.cache import DiskCache
from quantcio.data.pit import FUNDAMENTAL_COLUMNS, PITFundamentals
from quantcio.data.pit import kr_bulk as kb
from quantcio.data.pit import kr_dart as kdart
from quantcio.data.pit import kr_marcap as km
from quantcio.data.pit.kr import KRArtefacts, _now_kst, _read_pickle, _write_pickle, kr_artefacts, kr_root
from quantcio.models import Market

log = logging.getLogger(__name__)

BULK_PARSER_VERSION = 1
DEFAULT_WEB_BUDGET = 150  # DART web requests per call (keyless; >= 6 s apart -> <= 15 minutes)
SHORT_CORRECTION_DAYS = 7
TTM_ITEMS = ("revenue", "op_income", "ni_parent")
Q_ITEMS = ("revenue", "op_income", "ni_parent")
EXTRA_COLUMNS = [
    "code",
    "name",
    "market",
    "report",
    "fs",
    "reporting_currency",
    "fx_bs",
    "fx_flow",
    "revenue_q",
    "op_income_q",
    "ni_parent_q",
    "revenue_q_prev_year",
    "ni_parent_q_prev_year",
    "date_source",
    "rcept_dt",
    "rcept_no",
    "last_rcept_dt",
    "values_version",
    "lookahead_risk",
    "is_restatement_row",
]
PANEL_COLUMNS = [*FUNDAMENTAL_COLUMNS, *EXTRA_COLUMNS]


# ----------------------------------------------------------------------------------------------- bulk values


def _bulk_listing(root: Path, session: Any, now: datetime, notes: list[str], cov: dict[str, Any]) -> pd.DataFrame:
    """Bulk file list: live (at most every 12 h), else the cached list (noted), else local zips (noted)."""
    path = root / "list.pkl"
    cached = _read_pickle(path)
    if cached is not None and (now.replace(tzinfo=None) - cached[0]).total_seconds() < 12 * 3600:
        cov["bulk_list_source"] = "cache"
        return cached[1]
    try:
        listing = kb.fetch_bulk_list(session)
        _write_pickle(path, (now.replace(tzinfo=None), listing))
        cov["bulk_list_source"] = "live"
        return listing
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {str(exc)[:100]}"
        if cached is not None:
            notes.append(f"OpenDART 일괄다운로드 목록 갱신 실패({err}) → {cached[0]:%Y-%m-%d %H:%M} 목록 사용")
            cov["bulk_list_source"] = "stale_cache"
            return cached[1]
        local = sorted(p.name for p in (root / "zips").glob("*.zip"))
        if not local:
            raise RuntimeError(f"OpenDART 일괄다운로드 목록 수집 실패: {err}") from exc
        notes.append(f"OpenDART 일괄다운로드 목록 수집 실패({err}) → 로컬 보관 파일 {len(local)}개만 사용")
        cov["bulk_list_source"] = "local_zips"
        rows = []
        for n in local:
            parts = n.split("_")
            period = {"1Q": "FQ", "2Q": "HY", "3Q": "TQ", "4Q": "FY"}.get(parts[1], "")
            rows.append({"bsn_yr": int(parts[0]), "period": period, "stmt": parts[2], "fl_nm": n})
        df = pd.DataFrame(rows)
        df["built"] = pd.to_datetime(df["fl_nm"].str.extract(r"_(\d{14})\.zip$")[0], format="%Y%m%d%H%M%S")
        return df


def _zip_bytes(root: Path, session: Any, fl_nm: str) -> tuple[bytes, bool]:
    path = root / "zips" / fl_nm
    if path.exists():
        return path.read_bytes(), False
    content = kb.fetch_bulk_zip(session, fl_nm)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{fl_nm}.{time.monotonic_ns()}.tmp")
    tmp.write_bytes(content)
    tmp.replace(path)
    return content, True


def load_bulk_items(
    settings: Settings,
    cache: DiskCache,
    session: Any,
    now: datetime,
    notes: list[str],
    cov: dict[str, Any],
) -> pd.DataFrame:
    """Key items of every bulk period (all companies), parsed once per bulk build and cached."""
    root = kr_root(settings) / "dart_bulk"
    files = kb.latest_files(_bulk_listing(root, session, now, notes, cov))
    parts, downloaded, failed = [], 0, []
    for row in files.itertuples(index=False):
        names = [getattr(row, s, None) for s in kb.STATEMENTS]
        names = [n for n in names if isinstance(n, str)]
        tag = f"{row.bsn_yr}_{row.period}"
        if len(names) < len(kb.STATEMENTS):
            notes.append(f"OpenDART 일괄파일 {tag}: 재무제표 일부 없음({len(names)}/3) → 해당 항목 결측 가능")
        digest = hashlib.sha1("|".join(sorted(names)).encode()).hexdigest()[:12]
        ipath = root / "items" / f"{tag}_{digest}_v{BULK_PARSER_VERSION}.pkl"
        items = _read_pickle(ipath) if cache.enabled else None
        if items is None:
            try:
                longs, cons = [], set()
                for n in names:
                    content, fresh = _zip_bytes(root, session, n)
                    downloaded += int(fresh)
                    long, ck = kb.read_bulk_zip(content)
                    longs.append(long)
                    cons |= ck
                items = kb.extract_key_items(pd.concat(longs, ignore_index=True), cons)
                items["bulk_period"] = tag
                if cache.enabled:
                    _write_pickle(ipath, items)
                    for old in ipath.parent.glob(f"{tag}_*.pkl"):
                        if old != ipath:
                            old.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001 - one broken period must not sink the panel
                failed.append(f"{tag}: {type(exc).__name__}: {str(exc)[:80]}")
                log.warning("bulk period %s failed: %s", tag, exc)
                continue
        parts.append(items)
    cov["bulk_periods"] = len(files)
    cov["bulk_zips_downloaded"] = downloaded
    cov["bulk_period_failures"] = failed
    if failed:
        notes.append(f"OpenDART 일괄파일 {len(failed)}개 기간 처리 실패 → 해당 기간 결측: {', '.join(failed[:3])}")
    if not parts:
        raise RuntimeError("OpenDART 일괄 재무데이터를 하나도 읽지 못함")
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------------------------------------------ FX


def fx_history_yahoo(ccy: str, root: Path, now: datetime) -> pd.Series | None:
    """Daily KRW per 1 unit of `ccy` (Yahoo '<CCY>KRW=X' closes), refreshed at most once per day."""
    path = root / "fx" / f"{ccy}KRW.pkl"
    cached = _read_pickle(path)
    if cached is not None and cached[0] == now.date():
        return cached[1]
    try:
        import yfinance as yf

        h = yf.Ticker(f"{ccy}KRW=X").history(start="2013-01-01", auto_adjust=False)
        s = h["Close"].astype(float)
        idx = pd.DatetimeIndex(s.index)
        s.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        s = s[s > 0].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        if len(s) < 250:
            raise ValueError(f"{ccy}KRW=X: {len(s)}개 관측치")
        _write_pickle(path, (now.date(), s))
        return s
    except Exception as exc:  # noqa: BLE001
        log.warning("FX %sKRW history failed: %s", ccy, exc)
        return cached[1] if cached is not None else None


def _period_start(pe: pd.Timestamp, months: int) -> pd.Timestamp:
    return (pe.to_period("M") - months + 1).to_timestamp(how="start")


def convert_currency(items: pd.DataFrame, fx_history: Any, notes: list[str], cov: dict[str, Any]) -> pd.DataFrame:
    """Non-KRW rows -> KRW (balance sheet at the period-end rate, YTD flows at the YTD average, 3-month flows at the
    quarter average); rows whose currency has no FX history are dropped and counted."""
    df = items.reset_index(drop=True)
    if "currency" not in df:
        df["currency"] = "KRW"
    ccy = df["currency"].fillna("").astype(str).str.upper().str.strip().replace({"": "KRW", "NAN": "KRW"})
    df["reporting_currency"] = ccy
    df["fx_bs"] = 1.0
    df["fx_flow"] = 1.0
    converted: dict[str, int] = {}
    excluded: dict[str, int] = {}
    drop = pd.Series(False, index=df.index)
    for cur in sorted(set(ccy) - {"KRW"}):
        m = ccy == cur
        s = fx_history(cur)
        if s is None or len(s) == 0:
            excluded[cur] = int(m.sum())
            drop |= m
            continue
        for i in df.index[m]:
            pe = pd.Timestamp(df.at[i, "period_end"])
            fq = int(kb.REPORT_FQ.get(df.at[i, "보고서종류"], 4))
            bs = s.asof(pe)
            ytd = s[_period_start(pe, 3 * fq) : pe].mean()
            q3 = s[_period_start(pe, 3) : pe].mean()
            if pd.isna(bs) or pd.isna(ytd):
                drop[i] = True
                excluded[cur] = excluded.get(cur, 0) + 1
                continue
            for c in kb.BS_ITEMS:
                df.at[i, c] = df.at[i, c] * bs
            for c in kb.FLOW_ITEMS:
                df.at[i, c] = df.at[i, c] * ytd
                if c + "_3m" in df:
                    df.at[i, c + "_3m"] = df.at[i, c + "_3m"] * (q3 if not pd.isna(q3) else ytd)
            df.at[i, "fx_bs"] = float(bs)
            df.at[i, "fx_flow"] = float(ytd)
            converted[cur] = converted.get(cur, 0) + 1
    df.loc[~drop, "currency"] = "KRW"
    cov["fx_converted_rows"] = converted
    cov["fx_excluded_rows"] = excluded
    if converted:
        detail = ", ".join(f"{k} {v}" for k, v in converted.items())
        notes.append(
            f"외화 보고 재무제표 {sum(converted.values())}건을 원화로 환산({detail}; "
            "재무상태표=기말 환율, 손익=기간 평균)."
        )
    if excluded:
        detail = ", ".join(f"{k} {v}" for k, v in excluded.items())
        notes.append(f"환율 데이터가 없어 외화 보고 재무제표 {sum(excluded.values())}건 제외({detail}).")
    return df[~drop]


# --------------------------------------------------------------------------------------- publication dates


def _publications(
    settings: Settings,
    codes: list[str],
    session: Any,
    now: datetime,
    gate: kdart.DartGate,
    notes: list[str],
    cov: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    root = kr_root(settings)
    key = settings.data.dart_api_key
    today = now.date()
    pubs: dict[str, pd.DataFrame] = {}
    source: dict[str, str] = {}
    corp_map: dict[str, str] | None = None
    errors: dict[str, str] = {}
    windows = 0
    if key:
        try:
            cpath = root / "opendart" / "corp_codes.pkl"
            cached = _read_pickle(cpath)
            if cached is not None and (today - cached[0]).days < 7:
                corp_map = cached[1]
            else:
                corp_map = kdart.opendart_corp_codes(session, key)
                _write_pickle(cpath, (today, corp_map))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"OpenDART 고유번호 목록 실패({type(exc).__name__}) → 키 없는 DART 웹 검색으로 대체")
            corp_map = None
    for code in codes:
        rows = None
        try:
            if corp_map is not None and code in corp_map:
                opath = root / "opendart" / "list" / f"{code}.pkl"
                cached = _read_pickle(opath)
                if cached is not None and cached[0] == today:
                    rows = cached[1]
                else:
                    rows = kdart.opendart_periodic(session, key, corp_map[code], kdart.WINDOW_ORIGIN, today)
                    _write_pickle(opath, (today, rows))
                source[code] = "opendart_list"
            else:
                rows, fetched = kdart.cached_web_filings(root / "dart_pub", session, gate, code, today)
                windows += fetched
                if rows is not None:
                    source[code] = "dart_web"
        except Exception as exc:  # noqa: BLE001 - dates unknown -> lag rule, counted
            errors[code] = f"{type(exc).__name__}: {str(exc)[:80]}"
            rows = None
        if rows is not None:
            pubs[code] = kdart.publication_table(rows)
    missing = [c for c in codes if c not in pubs]
    cov["dart_dates"] = {
        "codes": len(codes),
        "with_dates": len(pubs),
        "without_dates": len(missing),
        "without_dates_codes": missing[:50],
        "errors": dict(list(errors.items())[:20]),
        "web_requests": gate.used,
        "web_refused": gate.refused,
        "web_windows_fetched": windows,
        "web_blocked": gate.blocked_error or None,
    }
    if missing:
        why = "DART 웹 차단 의심" if gate.blocked_error else ("요청 한도 소진" if gate.refused else "조회 실패")
        notes.append(
            f"공시 접수일 미확인 {len(missing)}/{len(codes)}개 종목({why}) → 보수적 지연 규칙(분기·반기 +"
            f"{settings.pit.fundamentals.kr_fallback_lag_days_quarter}일, 사업 +"
            f"{settings.pit.fundamentals.kr_fallback_lag_days_annual}일 후 다음 거래일) 적용, "
            "정정 여부 미확인으로 표시."
        )
    return pubs, source


def _originals(
    settings: Settings,
    todo: list[tuple[str, str, pd.Timestamp, str]],
    session: Any,
    gate: kdart.DartGate,
    cov: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Parsed original-filing XBRL items per rcept_no (cached on disk; 'no_xbrl' results are cached too)."""
    root = kr_root(settings) / "dart_xbrl"
    key = settings.data.dart_api_key
    out: dict[str, dict[str, Any]] = {}
    stats = {"requested": len(todo), "cached": 0, "fetched": 0, "no_xbrl": 0, "refused": 0, "errors": 0}
    for _code, rcept_no, period_end, report in todo:
        path = root / f"{rcept_no}.json"
        hit = kdart._json_read(path)
        if hit is not None:
            stats["cached"] += 1
        else:
            try:
                if key:
                    content = kdart.opendart_xbrl(session, key, rcept_no, kdart.REPRT_CODE.get(report, "11011"))
                else:
                    content = kdart.dart_filing_xbrl(session, gate, rcept_no)
                if isinstance(content, str):
                    stats["refused"] += 1
                    continue
                if content is None:
                    hit = {"status": "no_xbrl"}
                else:
                    hit = {"status": "ok", "items": kdart.xbrl_key_items(content, period_end)}
                kdart._json_write(path, hit)
                stats["fetched"] += 1
            except Exception as exc:  # noqa: BLE001
                stats["errors"] += 1
                log.warning("original XBRL %s failed: %s", rcept_no, exc)
                continue
        if hit.get("status") == "ok":
            out[rcept_no] = hit["items"]
        else:
            stats["no_xbrl"] += 1
    cov["original_xbrl"] = stats
    return out


# ------------------------------------------------------------------------------------------ versioned rows


def _pe_shift(pe: pd.Timestamp, months: int) -> pd.Timestamp:
    return (pe.to_period("M") - months).to_timestamp(how="end").normalize()


def _convert_original(orig: dict[str, Any], row: pd.Series) -> dict[str, Any]:
    """Original XBRL items in the row's reporting currency -> KRW with the row's rates (same basis as the bulk)."""
    out = {}
    fx_bs = float(row.get("fx_bs", 1.0) or 1.0)
    fx_flow = float(row.get("fx_flow", 1.0) or 1.0)
    for k, v in orig.items():
        if k.startswith("_") or v is None:
            continue
        out[k] = float(v) * (fx_bs if k in kb.BS_ITEMS else fx_flow)
    return out


def build_code_rows(
    items: pd.DataFrame,
    pubs: pd.DataFrame | None,
    originals: dict[str, dict[str, Any]],
    sessions: pd.DatetimeIndex,
    lag_days: tuple[int, int],
    stats: dict[str, int],
    date_source: str = "dart_web",
) -> list[dict[str, Any]]:
    """Versioned PIT rows of ONE company (items after quarterly_standalone and currency conversion)."""
    items = items.sort_values(["period_end", "fq"]).drop_duplicates("period_end", keep="last")
    by_pe = {pd.Timestamp(r["period_end"]): r for _, r in items.iterrows()}
    pub_map: dict[tuple[str, pd.Timestamp], Any] = {}
    if pubs is not None and len(pubs):
        for _, p in pubs.iterrows():
            pub_map[(p["kind"], pd.Timestamp(p["period_end"]))] = p

    info: dict[pd.Timestamp, dict[str, Any]] = {}
    for pe, r in by_pe.items():
        kind = kdart.BULK_TO_DART_KIND.get(r["보고서종류"], "사업보고서")
        p = pub_map.get((kind, pe))
        d: dict[str, Any] = {"kind": kind, "orig": None}
        if p is not None:
            first, last = pd.Timestamp(p["first_rcept_dt"]), pd.Timestamp(p["last_rcept_dt"])
            gap = (last - first).days
            d.update(source=date_source, rcept_dt=first, rcept_no=p["first_rcept_no"], last_rcept_dt=last)
            if gap > SHORT_CORRECTION_DAYS:
                d.update(af=kdart.next_session_after(first, sessions), restated=True)
                d["latest_from"] = kdart.next_session_after(last, sessions)
                o = originals.get(str(p["first_rcept_no"]))
                if o is not None:
                    o_fs = o.get("_fs")
                    if o_fs and o_fs != r["fs"]:
                        stats["original_fs_mismatch"] += 1
                    else:
                        d["orig"] = _convert_original(o, r)
            else:
                if gap > 0:
                    stats["short_corrections_waited"] += 1
                d.update(af=kdart.next_session_after(last, sessions), restated=False, latest_from=None)
        else:
            d.update(
                source="lag_rule",
                rcept_dt=pd.NaT,
                rcept_no=None,
                last_rcept_dt=pd.NaT,
                restated=None,
                latest_from=None,
                af=kdart.lag_rule_available_from(pe, kind, sessions, lag_days[0], lag_days[1]),
            )
        info[pe] = d

    def value(pe: pd.Timestamp, item: str, asof: pd.Timestamp, flags: set[str]) -> float:
        r = by_pe[pe]
        v = r.get(item, np.nan)
        d = info[pe]
        if d["restated"] is None:
            flags.add("unverified")
            return v
        if not d["restated"] or (not pd.isna(d["latest_from"]) and d["latest_from"] <= asof):
            return v
        o = d["orig"]
        if o is not None and item in o and not pd.isna(o[item]):
            flags.add("original")
            return o[item]
        flags.add("lookahead")
        return v

    def row_at(pe: pd.Timestamp, fq: int, item_pe: pd.Timestamp | None) -> pd.Timestamp | None:
        if item_pe is None or item_pe not in by_pe:
            return None
        return item_pe if int(by_pe[item_pe]["fq"]) == fq else None

    def ttm(pe: pd.Timestamp, item: str, asof: pd.Timestamp, flags: set[str]) -> float:
        r = by_pe[pe]
        fq = int(r["fq"])
        if fq == 4:
            return value(pe, item, asof, flags)
        fy = row_at(pe, 4, _pe_shift(pe, 3 * fq))
        py = row_at(pe, fq, _pe_shift(pe, 12))
        if fy is None or py is None:
            stats[f"ttm_missing_{item}"] += 1
            return np.nan
        if not (by_pe[fy]["fs"] == r["fs"] == by_pe[py]["fs"]):
            stats["ttm_fs_mismatch"] += 1
            return np.nan
        return value(pe, item, asof, flags) + value(fy, item, asof, flags) - value(py, item, asof, flags)

    def quarter(pe: pd.Timestamp, item: str, asof: pd.Timestamp, flags: set[str]) -> float:
        r = by_pe[pe]
        fq = int(r["fq"])
        sub: set[str] = set()
        ytd = value(pe, item, asof, sub)
        prev = row_at(pe, fq - 1, _pe_shift(pe, 3)) if fq > 1 else None
        if fq == 1:
            q = ytd
        elif prev is not None:
            q = ytd - value(prev, item, asof, sub)
        else:
            q = np.nan
        if not (sub & {"original", "lookahead"}) and not pd.isna(r.get(item + "_q", np.nan)):
            q = r[item + "_q"]  # bulk basis: prefer the as-filed 3-month column (quarterly_standalone)
        if fq == 4 and item == "revenue" and not pd.isna(q) and q < 0:
            q = np.nan  # FY reclassifies discontinued operations, the 3Q YTD does not
        flags |= sub
        return q

    out: list[dict[str, Any]] = []
    pes = sorted(by_pe)
    afs = {pe: info[pe]["af"] for pe in pes}
    for k, pe in enumerate(pes):
        r = by_pe[pe]
        d = info[pe]
        later_afs = [afs[x] for x in pes[k + 1 :] if not pd.isna(afs[x])]
        next_af = min(later_afs) if later_afs else pd.NaT
        versions = [(d["af"], False)]
        if d["restated"] and not pd.isna(d["latest_from"]):
            if pd.isna(next_af) or d["latest_from"] < next_af:
                versions.append((d["latest_from"], True))
                stats["restatement_rows"] += 1
            else:
                stats["restatement_rows_superseded"] += 1
        for asof, is_restatement in versions:
            if pd.isna(asof):
                stats["no_available_from"] += 1
                continue
            flags: set[str] = set()
            rec: dict[str, Any] = {
                "period_end": pe,
                "period_type": "FY" if int(r["fq"]) == 4 else "Q",
                "available_from": asof,
                "currency": "KRW",
                "equity_parent": value(pe, "equity_parent", asof, flags),
                "shares_out": np.nan,
                "code": r["stock_code"],
                "name": r.get("name"),
                "market": r.get("market"),
                "report": r["보고서종류"],
                "fs": r["fs"],
                "reporting_currency": r.get("reporting_currency", "KRW"),
                "fx_bs": r.get("fx_bs", 1.0),
                "fx_flow": r.get("fx_flow", 1.0),
                "date_source": d["source"],
                "rcept_dt": d["rcept_dt"],
                "rcept_no": d["rcept_no"],
                "last_rcept_dt": d["last_rcept_dt"],
                "is_restatement_row": is_restatement,
            }
            for item in TTM_ITEMS:
                rec[f"{item}_ttm"] = ttm(pe, item, asof, flags)
            for item in Q_ITEMS:
                rec[f"{item}_q"] = quarter(pe, item, asof, flags)
            prev_year = row_at(pe, int(r["fq"]), _pe_shift(pe, 12))
            for item in ("revenue", "ni_parent"):
                rec[f"{item}_q_prev_year"] = quarter(prev_year, item, asof, flags) if prev_year is not None else np.nan
            own_original = d["orig"] is not None and d["restated"] and not is_restatement
            if is_restatement:
                version = "restatement"
            elif d["restated"] is None:
                version = "unverified"
            elif d["restated"]:
                version = "original_xbrl" if own_original else "restated_bulk"
            else:
                version = "as_filed"
            rec["values_version"] = version
            rec["restated_later"] = bool(d["restated"] is None or (d["restated"] and not is_restatement))
            rec["lookahead_risk"] = bool(flags & {"lookahead", "unverified"})
            rec["source"] = "opendart_bulk+dart_xbrl" if "original" in flags else "opendart_bulk"
            out.append(rec)
    return out


# --------------------------------------------------------------------------------------------- entity link


def assign_entities(panel: pd.DataFrame, spells: pd.DataFrame, stats: dict[str, int]) -> pd.DataFrame:
    """Map each row to the listing spell listed at (or next listed after) its available_from; rows available only
    after the code's last spell ended are dropped (post-delisting filings)."""
    if panel.empty:
        return panel.assign(entity=pd.Series(dtype=object))
    sp = spells.sort_values(["Code", "spell"])
    by_code: dict[str, list[tuple[bool, pd.Timestamp, str]]] = {}
    for code, active, last, entity in zip(sp["Code"], sp["active"], sp["last_row_date"], sp["entity"], strict=True):
        by_code.setdefault(code, []).append((bool(active), pd.Timestamp(last), entity))
    ent = []
    for code, af in zip(panel["code"], panel["available_from"], strict=True):
        chosen = None
        for active, last, entity in by_code.get(code, []):
            if active or af <= last:
                chosen = entity
                break
        ent.append(chosen)
    out = panel.assign(entity=ent)
    stats["post_delisting_rows_dropped"] += int(out["entity"].isna().sum())
    return out[out["entity"].notna()]


def _shares_out(panel: pd.DataFrame, month_end: pd.DataFrame) -> pd.Series:
    """Listed common shares (marcap Stocks) on the last session of the period-end month, same listing spell."""
    if panel.empty or month_end.empty:
        return pd.Series(np.nan, index=panel.index)
    me = month_end.assign(ym=month_end["Date"].dt.to_period("M"))
    me = me.assign(entity=[km.entity_id(c, s) for c, s in zip(me["Code"], me["spell"], strict=True)])
    look = me.set_index(["entity", "ym"])["Stocks"]
    look = look[~look.index.duplicated(keep="last")]
    key = pd.MultiIndex.from_arrays([panel["entity"], pd.to_datetime(panel["period_end"]).dt.to_period("M")])
    return pd.Series(look.reindex(key).to_numpy(), index=panel.index)


# ---------------------------------------------------------------------------------------------------- main


def build_kr_pit_fundamentals(
    settings: Settings,
    cache: DiskCache,
    entities: list[str] | None = None,
    *,
    session: Any = None,
    now: datetime | None = None,
    max_dart_requests: int | None = None,
    dart_gate: kdart.DartGate | None = None,
    fx_history: Any = None,
    artefacts: KRArtefacts | None = None,
    items: pd.DataFrame | None = None,
) -> PITFundamentals:
    """PIT fundamentals panel for `entities` ('<code>' / '<code>#<spell>'; None = every KOSPI/KOSDAQ common listed
    since settings.pit.kr.start_year). ``items`` (pre-extracted bulk key items), ``session``, ``now``,
    ``dart_gate``, ``fx_history`` (ccy -> KRW rate Series) and ``artefacts`` are injection points for tests.

    ``max_dart_requests`` caps DART web requests of this call (keyless default 150; >= 6 s apart); dates or
    originals beyond the budget fall back to the lag rule / flags and are counted - later calls continue from the
    disk cache.
    """
    t0 = time.monotonic()
    now = _now_kst(now)
    notes: list[str] = []
    cov: dict[str, Any] = {}
    stats: dict[str, int] = {
        k: 0
        for k in (
            "short_corrections_waited",
            "original_fs_mismatch",
            "restatement_rows",
            "restatement_rows_superseded",
            "no_available_from",
            "ttm_fs_mismatch",
            "ttm_missing_revenue",
            "ttm_missing_op_income",
            "ttm_missing_ni_parent",
            "post_delisting_rows_dropped",
            "duplicate_period_rows",
        )
    }
    if session is None:
        from quantcio.data.http import PoliteSession

        session = PoliteSession(timeout=60)
    art = artefacts or kr_artefacts(settings, cache, now=now, session=session)
    spells = art.spells
    known = set(spells["entity"])
    if entities is None:
        wanted = sorted(known)
    else:
        wanted = list(dict.fromkeys(entities))
    unknown = [e for e in wanted if e not in known]
    wanted = [e for e in wanted if e in known]
    if unknown:
        notes.append(f"marcap에 없는 엔티티 {len(unknown)}개 제외: {', '.join(unknown[:5])}")
    codes = list(dict.fromkeys(km.parse_entity(e)[0] for e in wanted))

    if items is None:
        items = load_bulk_items(settings, cache, session, now, notes, cov)
    items = items[items["stock_code"].isin(codes) & items["period_end"].notna()].copy()
    fx = fx_history or (lambda ccy: fx_history_yahoo(ccy, kr_root(settings), now))
    items = convert_currency(items, fx, notes, cov)
    items = kb.quarterly_standalone(items)
    dup = items.duplicated(["stock_code", "period_end"], keep=False)
    stats["duplicate_period_rows"] = int(dup.sum())

    key_mode = bool(settings.data.dart_api_key)
    budget = max_dart_requests if max_dart_requests is not None else DEFAULT_WEB_BUDGET
    gate = dart_gate or kdart.DartGate(max_requests=int(budget))
    item_codes = [c for c in codes if c in set(items["stock_code"])]
    pubs, date_src = _publications(settings, item_codes, session, now, gate, notes, cov)

    # originals of periods corrected more than 7 days after their first filing
    todo = []
    for code, pt in pubs.items():
        if pt.empty:
            continue
        long_gap = (pd.to_datetime(pt["last_rcept_dt"]) - pd.to_datetime(pt["first_rcept_dt"])).dt.days
        own = items[items["stock_code"] == code]
        reports = {
            (kdart.BULK_TO_DART_KIND.get(k, ""), pd.Timestamp(pe)): k
            for k, pe in zip(own["보고서종류"], own["period_end"], strict=True)
        }
        for _, p in pt[long_gap > SHORT_CORRECTION_DAYS].iterrows():
            report = reports.get((p["kind"], pd.Timestamp(p["period_end"])))
            if report is None:
                continue
            todo.append((code, str(p["first_rcept_no"]), pd.Timestamp(p["period_end"]), report))
    originals = _originals(settings, todo, session, gate, cov) if todo else {}
    if not todo:
        cov["original_xbrl"] = {"requested": 0}

    sessions, last_real = kdart.extend_sessions(art.sessions, pd.Timestamp(now.date()) + pd.Timedelta(400, "D"))
    lag = (
        int(settings.pit.fundamentals.kr_fallback_lag_days_quarter),
        int(settings.pit.fundamentals.kr_fallback_lag_days_annual),
    )
    rows: list[dict[str, Any]] = []
    for code, g in items.groupby("stock_code", sort=False):
        rows += build_code_rows(g, pubs.get(code), originals, sessions, lag, stats, date_src.get(code, "dart_web"))
    panel = pd.DataFrame(rows)
    if panel.empty:
        panel = pd.DataFrame(columns=[*PANEL_COLUMNS, "entity"])
    else:
        panel = assign_entities(panel, spells, stats)
        panel = panel[panel["entity"].isin(set(wanted))].copy()
        panel["shares_out"] = _shares_out(panel, art.month_end)
    for c in PANEL_COLUMNS:
        if c not in panel:
            panel[c] = np.nan
    panel = panel[PANEL_COLUMNS].sort_values(["entity", "available_from", "period_end"]).reset_index(drop=True)
    for c in ("period_end", "available_from", "rcept_dt", "last_rcept_dt"):
        panel[c] = pd.to_datetime(panel[c])

    # coverage + notes
    with_rows = set(panel["entity"])
    missing_entities = [e for e in wanted if e not in with_rows]
    vv = panel["values_version"].value_counts().to_dict() if len(panel) else {}
    ds = panel["date_source"].value_counts().to_dict() if len(panel) else {}
    first_pe = panel.groupby("entity")["period_end"].min() if len(panel) else pd.Series(dtype="datetime64[ns]")
    sp_first = spells.set_index("entity")["first_date"]
    late = [
        e
        for e, pe in first_pe.items()
        if pe >= pd.Timestamp("2023-01-01") and pd.Timestamp(sp_first.get(e, pd.NaT)) < pd.Timestamp("2022-01-01")
    ]
    after_last = int((panel["available_from"] > last_real).sum()) if len(panel) else 0
    cov.update(
        {
            "entities_requested": len(wanted) + len(unknown),
            "entities_unknown": unknown[:50],
            "entities_with_rows": len(with_rows),
            "entities_without_rows": len(missing_entities),
            "entities_without_rows_list": missing_entities[:100],
            "entities_first_period_2023_plus": late[:100],
            "rows": len(panel),
            "rows_by_year": {
                str(k): int(v) for k, v in panel["available_from"].dt.year.value_counts().sort_index().items()
            }
            if len(panel)
            else {},
            "date_source_rows": {str(k): int(v) for k, v in ds.items()},
            "values_version_rows": {str(k): int(v) for k, v in vv.items()},
            "lookahead_risk_rows": int(panel["lookahead_risk"].fillna(False).astype(bool).sum()) if len(panel) else 0,
            "restated_later_rows": int(panel["restated_later"].fillna(False).astype(bool).sum()) if len(panel) else 0,
            "available_from_after_last_session": after_last,
            "item_coverage": {c: round(float(panel[c].notna().mean()), 4) for c in FUNDAMENTAL_COLUMNS}
            if len(panel)
            else {},
            "key_mode": "opendart_api" if key_mode else "keyless",
            "dart_web_requests_total": gate.used,
            "dart_web_refused_total": gate.refused,
            **stats,
            "seconds": round(time.monotonic() - t0, 2),
        }
    )
    notes.insert(
        0,
        f"KR 시점별 재무: OpenDART 일괄 재무제표(연결 우선, 없으면 별도) {len(with_rows)}/{len(wanted)}개 종목, "
        f"{len(panel)}행. 사용 가능일 = 원 공시 접수일 다음 거래일(미확인 시 보수적 지연 규칙).",
    )
    if missing_entities:
        notes.append(
            f"재무 데이터 없는 종목 {len(missing_entities)}개"
            "(금융업 2023년 3분기 이전·PDF 전용 공시·코넥스 등 일괄파일 미수록)."
        )
    if late:
        notes.append(f"2023년 이후부터 재무가 있는 종목 {len(late)}개(주로 금융업: 일괄파일은 2023년 3분기부터 수록).")
    restated = vv.get("original_xbrl", 0) + vv.get("restated_bulk", 0)
    if restated:
        notes.append(
            f"사후 정정된 기간 {restated}건: 원 공시 XBRL 값 사용 {vv.get('original_xbrl', 0)}건, "
            f"확보 실패로 정정값 사용 {vv.get('restated_bulk', 0)}건(lookahead_risk 표시)."
        )
    if stats["post_delisting_rows_dropped"]:
        notes.append(f"상장폐지 이후에만 공개된 재무 {stats['post_delisting_rows_dropped']}행 제외.")
    notes.append(
        "EPS·BPS는 수정되지 않은 보고값이라 밸류에이션에 쓰지 않음 — PER·PBR은 시가총액 ÷ 지배순이익·지배자본으로 계산."
    )
    return PITFundamentals(market=Market.KR, panel=panel, coverage=cov, notes=notes)
