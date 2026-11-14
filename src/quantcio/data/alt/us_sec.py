"""US alternative-data events from SEC EDGAR (keyless): forms / 8-K items per issuer and Form 4 insider trades.

Both sources are reduced to the ``events`` table of :mod:`quantcio.data.alt` (entity ``CIK##########``, rule
``published < t``). Verified live 2026-10-02.

``sec_forms`` - data.sec.gov submissions JSON of each CIK. The cache of ``us_cik.Submissions`` is reused (the main
    JSON is refreshed daily by ``data.alt.load``; an older ``files`` page on disk is reused only while it matches its
    ``files`` entry - EDGAR re-cuts those pages as the recent block rolls over, and a stale page dropped the filings
    between it and the recent block, docs/MODELING.md 84.5; a page fetched just now that still differs from its entry
    makes the main JSON be fetched again, so all blocks are of one EDGAR cut, docs/MODELING.md 92; older pages are
    read only until ``start`` is covered), but parsed here for the fields it ignores. With the data vintages on, the
    events of the CIKs read completely are frozen as first seen (``freeze_form_events``, docs/MODELING.md 92): EDGAR
    re-cutting its pages no longer changes the past. One event (value 1.0) per filing of a pre-declared form
    family (``FORM_KINDS``) and one per listed 8-K item (``8k_<item>``, ``ITEMS_8K``; pre-2004 item numbers such as
    "5" produce no item event).
    ``published`` = America/New_York calendar date of ``acceptanceDateTime``. EDGAR stores it in UTC (a Form 4
    accepted 20:30 ET reads ``...T00:30:00Z`` of the next UTC day with the same ``filingDate``). The acceptance time
    is the dissemination time: an 8-K accepted after 17:30 ET is public that evening although its ``filingDate`` is
    the next business day, and a filing accepted days after an adjusted ``filingDate`` is dated by the later
    acceptance. ``filingDate`` is used only when the acceptance time is missing (counted).
    Role filter: SC 13D/G and Form 144 are filed ABOUT a subject company by someone else, and the submissions list of
    a CIK also holds the ones it filed as the FILER about other issuers (BlackRock Finance: 946 SC 13G/A in two
    years). EDGAR gives a subject-role filing the subject's file number ("005-..." for 13D/G, the issuer's "001-..."
    for 144) and a filer-role filing an empty ``fileNumber`` (checked on EDGAR index pages), so these three kinds are
    kept only with a non-empty ``fileNumber``; skipped ones are counted. Forms 4 carry no file number either way: a
    company that is itself a reporting owner of another issuer (Goldman, Fluor) adds those Forms 4 to its own
    ``form4`` count (15 of 11,579 Forms 4 over 769 cached CIKs in 2026Q2, ~0.1%).
    The submissions JSON of a CIK is cached without expiry by ``Submissions``: filings after the fetch date are
    missing for that CIK; CIKs whose cache predates ``end`` are counted and noted.

``sec_form345`` - SEC Form 3/4/5 insider-transaction data sets (quarterly zips 2006q1.., 8-17 MB each). Only Table I
    (non-derivative) open-market purchases (TRANS_CODE P, acquired) and sales (S, disposed) of Forms 4 and 4/A count.
    Per (issuer, FILING_DATE): ``insider_buy_value`` / ``insider_sell_value`` = sum of shares x price (USD, positive)
    and ``insider_buy_n`` / ``insider_sell_n`` = distinct reporting-owner CIKs with such a trade (every co-filer of a
    joint filing counts). ``published`` = FILING_DATE: the data sets have no acceptance time, and a Form 4 accepted up
    to 22:00 ET keeps that day's filing date, so FILING_DATE is never earlier than dissemination.
    Dropped and counted: missing / non-positive shares or price, a code contradicting its acquired/disposed flag
    (P+D, S+A), a 4/A row repeating a transaction (issuer, owners, date, code, shares, price) of an original Form 4
    filed no later (amendments usually restate the whole form), an accession posted in two quarters (the later copy).
    Values are as filed (no winsorising: a mistyped price gives an extreme value). The data sets are posted after
    each quarter ends, so trades filed after the last listed quarter are missing (noted); the ``form4`` filing counts
    of ``sec_forms`` stay current. The 10b5-1 plan flag (AFF10B5ONE, filled from 2023) is kept in the cache only.
    Cache: ``settings.cache_path/alt/us_sec/form345/<yyyy>q<n>.parquet`` holds the P/S rows of ALL issuers (a wider
    CIK set needs no download); the zip is deleted after extraction. Closed quarters are cached forever, the latest
    listed quarter is re-downloaded when its parquet is older than 7 days. Quarter links are scraped from the index
    page (the newest may live under a different path).
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import time
import warnings
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from quantcio.config import Settings
from quantcio.data.alt import AltData, empty_events, normalise_events
from quantcio.data.pit.us_cik import Submissions, cik_entity
from quantcio.data.pit.us_net import PitNet, SecUserAgentError, atomic_write
from quantcio.data.vintage import KST, VintageStore

log = logging.getLogger(__name__)

SOURCE_FORMS = "sec_forms"
FORMS_VINTAGE = "us_sec_forms_v1"  # bump when the events of a filing change (FORM_KINDS, parsing): a new vintage
SOURCE_FORM345 = "sec_form345"
NY_TZ = "America/New_York"

FORM_KINDS = ("form4", "form144", "sc13d", "sc13g", "nt_late", "prospectus_424b", "s3", "8k")
ITEMS_8K = (
    "1.01",
    "1.02",
    "1.03",
    "2.01",
    "2.02",
    "2.03",
    "2.05",
    "2.06",
    "3.01",
    "3.02",
    "4.01",
    "4.02",
    "5.02",
    "5.07",
    "7.01",
    "8.01",
)
ITEM_KINDS = tuple(f"8k_{i}" for i in ITEMS_8K)
INSIDER_KINDS = ("insider_buy_value", "insider_sell_value", "insider_buy_n", "insider_sell_n")
KINDS: tuple[str, ...] = FORM_KINDS + ITEM_KINDS + INSIDER_KINDS
SUBJECT_ROLE_KINDS = frozenset({"form144", "sc13d", "sc13g"})  # filed ABOUT the subject; empty fileNumber = filer

FORM345_INDEX = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
FORM345_FIRST = "2006q1"
FORM4_DOCS = frozenset({"4", "4/A"})
OPEN_MARKET = {"P": "A", "S": "D"}  # trans code -> the acquired/disposed flag it must carry
CURRENT_QUARTER_MAX_AGE_D = 7.0
INDEX_MAX_AGE_H = 24.0
SUB_COLS = ("ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK")
TRANS_COLS = (
    "ACCESSION_NUMBER",
    "TRANS_DATE",
    "TRANS_CODE",
    "TRANS_SHARES",
    "TRANS_PRICEPERSHARE",
    "TRANS_ACQUIRED_DISP_CD",
)
OWNER_COLS = ("ACCESSION_NUMBER", "RPTOWNERCIK")
ROW_COLUMNS = [
    "accession",
    "issuer_cik",
    "filing_date",
    "doc_type",
    "trans_date",
    "code",
    "ad",
    "shares",
    "price",
    "aff10b5one",
    "owners",
]
_SUB_FIELDS = ("accessionNumber", "filingDate", "acceptanceDateTime", "form", "items", "fileNumber")
_QKEY = re.compile(r"^(\d{4})q([1-4])$")


# ============================================================================ submissions (pure)


def form_kind(form: Any) -> str | None:
    """Pre-declared event kind of an EDGAR form type (None = not an event)."""
    f = str(form or "").strip().upper()
    base = f.removesuffix("/A")
    if base == "4":
        return "form4"
    if base == "144":
        return "form144"
    if base in ("SC 13D", "SCHEDULE 13D"):
        return "sc13d"
    if base in ("SC 13G", "SCHEDULE 13G"):
        return "sc13g"
    if base in ("NT 10-K", "NT 10-Q"):
        return "nt_late"
    if f.startswith("424B"):
        return "prospectus_424b"
    if f in ("S-3", "S-3ASR", "S-3/A"):
        return "s3"
    if base == "8-K":
        return "8k"
    return None


def acceptance_ny_dates(acceptance: Iterable[Any]) -> pd.Series:
    """UTC ``acceptanceDateTime`` strings -> America/New_York calendar dates (NaT where missing / unparsable)."""
    txt = pd.Series(list(acceptance), dtype=object).map(
        lambda v: v.strip() if isinstance(v, str) and v.strip() else None
    )
    ts = pd.to_datetime(txt, utc=True, errors="coerce", format="ISO8601")
    return ts.dt.tz_convert(NY_TZ).dt.tz_localize(None).dt.normalize()


def published_dates(acceptance: Iterable[Any], filing_date: Iterable[Any]) -> pd.Series:
    """``published`` of a filing: NY date of the acceptance time, else ``filingDate`` (see module docstring)."""
    ny = acceptance_ny_dates(acceptance)
    fd = pd.to_datetime(pd.Series(list(filing_date), dtype=object), errors="coerce").dt.normalize()
    return ny.fillna(fd)


def _block_frame(block: dict) -> pd.DataFrame:
    """One submissions block (``filings.recent`` or an older ``files`` page) -> columns of ``_SUB_FIELDS``."""
    n = len(block.get("form") or [])
    cols = {}
    for k in _SUB_FIELDS:
        v = block.get(k)
        cols[k] = list(v) if isinstance(v, list) and len(v) == n else [None] * n
    return pd.DataFrame(cols, columns=list(_SUB_FIELDS))


def parse_submission_blocks(
    blocks: list[dict], entity: str, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, Counter]:
    """Submissions blocks of one CIK -> ``sec_forms`` events published in [start, end] and counters
    (``filings`` matched to a kind, ``filer_role_skipped``, ``acceptance_fallback``, ``no_date``)."""
    st: Counter = Counter()
    frames = [_block_frame(b) for b in blocks if b]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(_SUB_FIELDS))
    if df.empty:
        return empty_events(), st
    acc = df["accessionNumber"]
    df = df[~(acc.notna() & acc.duplicated())]  # a filing listed in two blocks counts once
    df = df.assign(kind=df["form"].map(form_kind))
    df = df[df["kind"].notna()].reset_index(drop=True)
    filer = df["kind"].isin(SUBJECT_ROLE_KINDS) & df["fileNumber"].map(lambda v: isinstance(v, str) and not v.strip())
    st["filer_role_skipped"] = int(filer.sum())
    df = df[~filer].reset_index(drop=True)
    ny = acceptance_ny_dates(df["acceptanceDateTime"])
    df["published"] = published_dates(df["acceptanceDateTime"], df["filingDate"])
    st["acceptance_fallback"] = int((ny.isna() & df["published"].notna()).sum())
    st["no_date"] = int(df["published"].isna().sum())
    df = df[df["published"].notna() & (df["published"] >= start) & (df["published"] <= end)]
    st["filings"] = len(df)
    if df.empty:
        return empty_events(), st
    base = pd.DataFrame({"published": df["published"], "kind": df["kind"], "value": 1.0})
    k8 = df[df["kind"] == "8k"]
    items = (
        k8.assign(item=k8["items"].map(lambda s: [x.strip() for x in str(s or "").split(",")]))
        .explode("item")
        .loc[lambda x: x["item"].isin(ITEMS_8K)]
        .drop_duplicates(["accessionNumber", "item"])
    )
    item_ev = pd.DataFrame({"published": items["published"], "kind": "8k_" + items["item"].astype(str), "value": 1.0})
    ev = pd.concat([base, item_ev], ignore_index=True) if len(item_ev) else base
    ev["entity"] = entity
    ev["source"] = SOURCE_FORMS
    return normalise_events(ev), st


# ============================================================================ submissions (io)


def submission_blocks(subs: Submissions, cik: int, start: pd.Timestamp) -> list[dict]:
    """``filings.recent`` + older ``files`` pages of the CIK, the pages loaded (cache first) only while the oldest
    filing seen is still after ``start``. [] when the CIK has no submissions JSON (404).

    One cut of EDGAR: a page fetched just now that still differs from its ``files`` entry means EDGAR re-cut the
    pages after the main JSON was fetched (the cached main JSON can be a day old); a page of the new cut next to a
    cached page of the old one dropped the filings between them (docs/MODELING.md 84.6 check 3, 92). The main JSON
    is then fetched again and the pages checked against it once more; pages that still differ stay in
    ``subs.mismatched`` (the caller does not mark the CIK complete)."""
    sub = subs.get(cik)
    blocks: list[dict] = []
    for attempt in range(2):
        if not sub:
            return []
        subs.mismatched.clear()
        filings = sub.get("filings") or {}
        blocks = [filings.get("recent") or {}]
        for f in filings.get("files", []) or []:
            oldest = min((d for b in blocks for d in (b.get("filingDate") or []) if d), default=None)
            if oldest is not None and pd.Timestamp(oldest) <= start:
                break
            blocks.append(subs.page(f["name"], f))
        if not subs.mismatched or attempt:
            break
        sub = subs.refresh(cik)
    return blocks


def _event_counts(events: pd.DataFrame) -> dict[str, Any]:
    return {
        "ciks_with_events": int(events["entity"].nunique()),
        "rows": len(events),
        "rows_by_kind": {k: int(v) for k, v in events["kind"].value_counts().sort_index().items()},
        "latest_published": None if events.empty else str(events["published"].max().date()),
    }


def _load_forms(
    net: PitNet, ciks: list[int], start: pd.Timestamp, end: pd.Timestamp, say: Callable[[str], None]
) -> tuple[pd.DataFrame, dict[str, Any], list[str], list[str], dict[str, pd.Timestamp]]:
    """Events, coverage, notes, the entities read completely (main JSON + every older page back to start, all of
    one EDGAR cut) and the New York date each entity's main JSON was fetched."""
    st: Counter = Counter()
    frames: list[pd.DataFrame] = []
    failed: list[int] = []
    missing: list[int] = []
    gaps: list[int] = []
    recut: list[int] = []
    complete: list[str] = []
    stale: list[tuple[int, pd.Timestamp]] = []
    fetched_on: dict[str, pd.Timestamp] = {}
    for i, cik in enumerate(ciks, 1):
        subs = Submissions(net)  # fresh per CIK: Submissions keeps every JSON it read in memory
        calls = net.stats["http_sec"]
        try:
            blocks = submission_blocks(subs, cik, start)
        except SecUserAgentError:
            raise
        except Exception as exc:  # noqa: BLE001 - one CIK is counted and noted, the rest continues
            log.warning("sec_forms: CIK %s failed (%s: %s)", cik, type(exc).__name__, exc)
            failed.append(cik)
            continue
        st["http_requests"] += net.stats["http_sec"] - calls
        st["main_refetched"] += len(subs.refreshed)
        if not blocks:
            missing.append(cik)
            continue
        st["pages_read"] += len(blocks) - 1
        if any(not b for b in blocks[1:]):  # a listed older page answered 404 / empty: history incomplete
            gaps.append(cik)
        elif subs.mismatched:  # the pages still differ from the main JSON fetched again: no single cut
            recut.append(cik)
        else:
            complete.append(cik_entity(cik))
        p = net.path("sec", f"subs/{cik}.json.gz")
        if p.exists():
            fetched = pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC").tz_convert(NY_TZ).tz_localize(None)
            fetched_on[cik_entity(cik)] = fetched.normalize()
            if fetched.normalize() < end:
                stale.append((cik, fetched.normalize()))
        ev, s = parse_submission_blocks(blocks, cik_entity(cik), start, end)
        st.update(s)
        frames.append(ev)
        if i % 50 == 0 or i == len(ciks):
            say(f"SEC submissions {i}/{len(ciks)} CIKs")
    events = normalise_events(pd.concat(frames, ignore_index=True)) if frames else empty_events()
    cov: dict[str, Any] = {
        "ciks_asked": len(ciks),
        "ciks_no_submissions": len(missing),
        "ciks_failed": len(failed),
        "ciks_page_missing": len(gaps),
        "ciks_recut_unresolved": len(recut),
        "ciks_complete": len(complete),
        **_event_counts(events),
        "filings": int(st["filings"]),
        "older_pages_read": int(st["pages_read"]),
        "http_requests": int(st["http_requests"]),
        "main_refetched": int(st["main_refetched"]),
        "filer_role_skipped": int(st["filer_role_skipped"]),
        "acceptance_fallback": int(st["acceptance_fallback"]),
        "no_date": int(st["no_date"]),
        "ciks_cache_before_end": len(stale),
    }
    notes: list[str] = []
    if recut:
        notes.append(
            f"sec_forms: the older submissions pages of {len(recut)} CIKs still differed from their main JSON after "
            f"it was fetched again (EDGAR re-cut them meanwhile); their events are kept but they are not marked "
            f"covered (e.g. {', '.join(map(str, recut[:5]))})"
        )
    if failed:
        notes.append(
            f"sec_forms: submissions of {len(failed)} of {len(ciks)} CIKs could not be loaded and have no form "
            f"events (e.g. {', '.join(map(str, failed[:5]))})"
        )
    if missing:
        notes.append(
            f"sec_forms: {len(missing)} CIKs have no SEC submissions JSON (404) and no form events "
            f"(e.g. {', '.join(map(str, missing[:5]))})"
        )
    if gaps:
        notes.append(
            f"sec_forms: an older submissions page of {len(gaps)} CIKs was missing (404 / empty); their events are "
            f"kept but they are not marked covered (e.g. {', '.join(map(str, gaps[:5]))})"
        )
    if stale:
        oldest = min(d for _, d in stale)
        notes.append(
            f"sec_forms: the cached submissions JSON of {len(stale)} CIKs was fetched before {end.date()} (oldest "
            f"{oldest.date()}); their filings after the fetch date are missing (delete "
            f"{net.path('sec', 'subs')}/<cik>.json.gz to refresh)"
        )
    if st["no_date"]:
        notes.append(f"sec_forms: {st['no_date']} filings had neither an acceptance time nor a filing date (dropped)")
    return events, cov, notes, complete, fetched_on


# ============================================================================ submissions (vintage)


def forms_vintage_ns(start: pd.Timestamp) -> str:
    """Vintage namespace of the ``sec_forms`` events loaded from Jan 1 of ``start``'s year."""
    return f"{FORMS_VINTAGE}_{pd.Timestamp(start).year}"


def freeze_form_events(
    store: VintageStore,
    events: pd.DataFrame,
    *,
    start: pd.Timestamp,
    asked: Iterable[str],
    eligible: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """``sec_forms`` events through their first-seen vintage (docs/MODELING.md 92; the rule of ``data.vintage``).

    Per entity, the store keeps the events of its first complete read up to ``through`` (the vintage cutoff of that
    read): an entity in the store gets its stored events dated <= ``through`` plus the fresh events after it; an
    entity seen for the first time is stored only when ``eligible`` (read completely, of one EDGAR cut, main JSON
    fetched after the cutoff), so a failed or partial read is never frozen as "no filings". An eligible stored
    entity whose ``through`` is before the store's cutoff takes the fresh events up to the cutoff (a new month).
    Filings EDGAR adds or drops later for dates already stored are counted (eligible entities only), not used.
    ``events``: the fresh events loaded from ``start`` (Jan 1 of the namespace year). Returns (events of the
    ``asked`` entities, coverage dict, notes)."""
    ns = forms_vintage_ns(start)
    data, meta, restarted = store.read(ns)
    cut = store.cut
    cols = ["entity", "published", "kind", "value"]
    through = {str(k): pd.Timestamp(v) for k, v in (meta.get("through") or {}).items()}
    stored = (data if data is not None else pd.DataFrame(columns=cols)).reindex(columns=cols)
    stored["entity"] = stored["entity"].astype(str)
    stored["published"] = pd.to_datetime(stored["published"]).astype("datetime64[ns]")
    stored["value"] = stored["value"].astype(float)
    ev = events.reindex(columns=cols).copy()
    ev["entity"] = ev["entity"].astype(str)
    ev["published"] = pd.to_datetime(ev["published"]).astype("datetime64[ns]")
    ask = {str(a) for a in asked}
    ok = {str(e) for e in eligible} & ask
    ev = ev[ev["entity"].isin(ask)].reset_index(drop=True)
    th = pd.to_datetime(ev["entity"].map(through)).astype("datetime64[ns]")  # NaT: not stored yet
    known = th.notna().to_numpy()
    after = (ev["published"] > th).to_numpy()  # False where th is NaT
    v = stored[stored["entity"].isin(ask)]
    keep = [p for p in (v, ev[~known | after]) if len(p)]
    out = pd.concat(keep, ignore_index=True) if keep else ev.iloc[:0]
    # revisions of stored dates, counted on the entities read completely now
    both = sorted(set(through) & ok)
    a = ev[known & ~after & ev["entity"].isin(both).to_numpy()].groupby(cols).size()
    b = v[v["entity"].isin(both)].groupby(cols).size()
    d = a.sub(b, fill_value=0)
    added, dropped = int(d.clip(lower=0).sum()), int((-d).clip(lower=0).sum())
    upto = (ev["published"] <= cut).to_numpy()
    take = ev["entity"].isin(ok).to_numpy() & upto & (~known | after)
    new_keys = sorted(ok - set(through))
    extended = sorted(e for e in ok & set(through) if through[e] < cut)
    for e in (*new_keys, *extended):
        through[e] = cut
    take &= ev["entity"].isin([*new_keys, *extended]).to_numpy()
    add = ev[take]
    not_frozen = sorted(ask - set(through))
    if new_keys or extended:
        parts = [p for p in (stored, add) if len(p)]
        long = pd.concat(parts, ignore_index=True) if parts else add
        long = long.sort_values(["entity", "published", "kind"], kind="stable").reset_index(drop=True)
        now = datetime.now(KST).isoformat(timespec="seconds")
        event = {"at": now, "cutoff": str(cut.date()), "new_keys": len(new_keys), "extended": len(extended)}
        hist = [*meta.get("history", []), {**event, "rows_added": len(add)}][-500:]
        store.write(
            ns,
            long,
            {
                "cutoff": str(max(through.values()).date()),
                "start": str(pd.Timestamp(start).date()),
                "keys": sorted(through),
                "through": {k: str(t.date()) for k, t in sorted(through.items())},
                "created": meta.get("created", now),
                "updated": now,
                "history": hist,
            },
        )
    info = {
        "namespace": ns,
        "cutoff": str(cut.date()),
        "created": data is None and bool(new_keys),
        "restarted": restarted,
        "entities_stored": len(through),
        "new_keys": len(new_keys),
        "extended": len(extended),
        "rows_added": len(add) if new_keys or extended else 0,
        "store_only": len({e for e in ask & set(through)} - set(ev["entity"])),
        "not_frozen": len(not_frozen),
        "added": added,
        "dropped": dropped,
    }
    notes: list[str] = []
    if added or dropped:
        notes.append(
            f"sec_forms vintage: EDGAR now lists {added} filings on dates already frozen that the first-seen copy "
            f"lacks (ignored) and lacks {dropped} that it has (kept)"
        )
    if not_frozen:
        notes.append(
            f"sec_forms vintage: {len(not_frozen)} CIKs were not read completely after the cutoff {cut.date()}; "
            f"their fresh events are used and not frozen (e.g. {', '.join(not_frozen[:5])})"
        )
    if restarted:
        notes.append(f"sec_forms vintage {ns} was unreadable: moved aside, a new vintage starts")
    out = normalise_events(out.assign(source=SOURCE_FORMS)) if len(out) else empty_events()
    return out, info, notes


# ============================================================================ form 3/4/5 data sets (pure)


def _require(df: pd.DataFrame, cols: Iterable[str], table: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{table}.tsv lacks columns {miss} (has {list(df.columns)[:20]})")


def _dd_mon_yyyy(s: pd.Series) -> pd.Series:
    """'31-JAN-2013' (the data sets' documented format) -> Timestamp; ISO dates accepted as a fallback."""
    d = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    other = d.isna() & s.astype(str).str.strip().ne("")
    if other.any():
        d[other] = pd.to_datetime(s[other], format="ISO8601", errors="coerce")
    return d


def parse_form345_tables(sub: pd.DataFrame, trans: pd.DataFrame, owners: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    """SUBMISSION / NONDERIV_TRANS / REPORTINGOWNER tables (all str) -> one row per non-derivative P or S
    transaction of ANY issuer and document type (``ROW_COLUMNS``; ``owners`` = '|'-joined owner CIKs).
    Counter: ``no_submission`` (transaction rows whose accession or issuer / filing date is unusable, dropped)."""
    _require(sub, SUB_COLS, "SUBMISSION")
    _require(trans, TRANS_COLS, "NONDERIV_TRANS")
    _require(owners, OWNER_COLS, "REPORTINGOWNER")
    st: Counter = Counter()
    code = trans["TRANS_CODE"].astype(str).str.strip().str.upper()
    t = trans.loc[code.isin(OPEN_MARKET.keys()), list(TRANS_COLS)].assign(TRANS_CODE=code)
    keep = [*SUB_COLS, *(["AFF10B5ONE"] if "AFF10B5ONE" in sub.columns else [])]
    s = sub[keep].drop_duplicates("ACCESSION_NUMBER")
    m = t.merge(s, on="ACCESSION_NUMBER", how="left")
    issuer = pd.to_numeric(m["ISSUERCIK"], errors="coerce")
    fdate = _dd_mon_yyyy(m["FILING_DATE"].fillna(""))
    ok = (issuer.notna() & fdate.notna()).to_numpy()
    st["no_submission"] = int((~ok).sum())
    o = (
        owners[list(OWNER_COLS)]
        .assign(cik=pd.to_numeric(owners["RPTOWNERCIK"], errors="coerce"))
        .dropna(subset=["cik"])
    )
    joined = o.groupby("ACCESSION_NUMBER")["cik"].agg(lambda c: "|".join(str(int(x)) for x in sorted(set(c))))
    aff = (
        m["AFF10B5ONE"].astype(str).str.strip().str.upper().map({"1": 1.0, "TRUE": 1.0, "0": 0.0, "FALSE": 0.0})
        if "AFF10B5ONE" in m.columns
        else pd.Series(np.nan, index=m.index)
    )
    out = pd.DataFrame(
        {
            "accession": m["ACCESSION_NUMBER"].astype(str).str.strip(),
            "issuer_cik": issuer,
            "filing_date": fdate,
            "doc_type": m["DOCUMENT_TYPE"].astype(str).str.strip().str.upper(),
            "trans_date": _dd_mon_yyyy(m["TRANS_DATE"].fillna("")),
            "code": m["TRANS_CODE"],
            "ad": m["TRANS_ACQUIRED_DISP_CD"].astype(str).str.strip().str.upper(),
            "shares": pd.to_numeric(m["TRANS_SHARES"], errors="coerce"),
            "price": pd.to_numeric(m["TRANS_PRICEPERSHARE"], errors="coerce"),
            "aff10b5one": aff.astype(float),
            "owners": m["ACCESSION_NUMBER"].map(joined).fillna(""),
        }
    )[ok].reset_index(drop=True)
    out["issuer_cik"] = out["issuer_cik"].astype("int64")
    return out[ROW_COLUMNS], st


def _read_tsv(z: zipfile.ZipFile, name: str, cols: Iterable[str], st: Counter) -> pd.DataFrame:
    member = next((n for n in z.namelist() if n.rsplit("/", 1)[-1].upper() == name.upper()), None)
    if member is None:
        raise ValueError(f"{name} not in the zip ({z.namelist()})")
    want = set(cols)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", pd.errors.ParserWarning)
        with z.open(member) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                dtype=str,
                quoting=csv.QUOTE_NONE,
                keep_default_na=False,
                usecols=lambda c: c in want,
                encoding="utf-8",
                encoding_errors="replace",
                on_bad_lines="warn",
            )
    st[f"bad_lines_{name}"] += sum(str(w.message).count("Skipping line") for w in caught)
    return df


def read_form345_zip(source: bytes | str | Path) -> tuple[pd.DataFrame, Counter]:
    """A quarterly ``<yyyy>q<n>_form345.zip`` (path or bytes) -> :func:`parse_form345_tables` rows and counters
    (malformed TSV lines are skipped and counted as ``bad_lines_<table>``)."""
    st: Counter = Counter()
    raw = io.BytesIO(source) if isinstance(source, bytes) else source
    with zipfile.ZipFile(raw) as z:
        sub = _read_tsv(z, "SUBMISSION.tsv", [*SUB_COLS, "AFF10B5ONE"], st)
        trans = _read_tsv(z, "NONDERIV_TRANS.tsv", TRANS_COLS, st)
        owners = _read_tsv(z, "REPORTINGOWNER.tsv", OWNER_COLS, st)
    rows, s = parse_form345_tables(sub, trans, owners)
    st.update(s)
    return rows, st


def form345_events(
    rows: pd.DataFrame, ciks: Iterable[int], start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, Counter]:
    """Cached P/S rows (optionally with a ``quarter`` column) -> ``sec_form345`` events of the CIKs with
    FILING_DATE in [start, end], one row per (entity, published, kind). Counters of everything dropped."""
    st: Counter = Counter()
    want = {int(c) for c in ciks}
    r = rows[rows["issuer_cik"].isin(want)].reset_index(drop=True)
    # parquet round trips may return dictionary (categorical) columns: a categorical groupby key would build the
    # full cartesian product of categories (observed=False) - plain objects keep the groupbys row-sized
    for c in r.columns:
        if isinstance(r[c].dtype, pd.CategoricalDtype):
            r[c] = r[c].astype(object)
    if "quarter" in r.columns and not r.empty:  # an accession posted in two quarters: keep the first posting
        first = r.groupby("accession")["quarter"].transform("min")
        st["duplicate_rows"] = int((r["quarter"] != first).sum())
        r = r[r["quarter"] == first]
    form4 = r["doc_type"].astype(str).isin(FORM4_DOCS)
    st["not_form4"] = int((~form4).sum())
    r = r[form4]
    ad_ok = r["ad"].astype(str) == r["code"].astype(str).map(OPEN_MARKET)
    st["code_ad_mismatch"] = int((~ad_ok).sum())
    r = r[ad_ok]
    good = (r["shares"] > 0) & (r["price"] > 0)  # NaN compares False
    st["bad_shares_or_price"] = int((~good).sum())
    r = r[good].reset_index(drop=True)
    key = ["issuer_cik", "owners", "trans_date", "code", "shares", "price"]
    is_am = r["doc_type"].astype(str) == "4/A"
    if is_am.any() and (~is_am).any():
        orig = r.loc[~is_am, [*key, "filing_date"]].groupby(key, dropna=False)["filing_date"].min()
        am = r.loc[is_am, [*key, "filing_date"]].reset_index()
        am = am.merge(orig.rename("orig_date").reset_index(), on=key, how="left")
        restated = am.loc[am["orig_date"].notna() & (am["orig_date"] <= am["filing_date"]), "index"]
        st["amendment_restated"] = len(restated)
        r = r.drop(index=restated.to_numpy())
    r = r[(r["filing_date"] >= start) & (r["filing_date"] <= end)]
    st["transactions"] = len(r)
    if r.empty:
        return empty_events(), st
    r = r.assign(
        value=r["shares"] * r["price"],
        side=np.where(r["code"].astype(str) == "P", "buy", "sell"),
        owner=r["owners"].where(r["owners"] != "", "acc:" + r["accession"]).str.split("|"),
    )
    st["no_owner"] = int((r["owners"] == "").sum())
    g = ["issuer_cik", "filing_date", "side"]
    val = r.groupby(g)["value"].sum().reset_index()
    val["kind"] = "insider_" + val["side"] + "_value"
    cnt = r.explode("owner").groupby(g)["owner"].nunique().rename("value").reset_index()
    cnt["kind"] = "insider_" + cnt["side"] + "_n"
    ev = pd.concat([val, cnt], ignore_index=True)
    ev["entity"] = ev["issuer_cik"].map(cik_entity)
    ev["published"] = ev["filing_date"]
    ev["source"] = SOURCE_FORM345
    return normalise_events(ev), st


def parse_form345_index(html: str, base: str = FORM345_INDEX) -> dict[str, str]:
    """Index page -> {'2026q1': absolute zip URL} of every quarterly data set linked on it."""
    out: dict[str, str] = {}
    for m in re.finditer(r"""href=["']([^"']*?/(\d{4}q[1-4])_form345\.zip)["']""", html, re.IGNORECASE):
        out.setdefault(m.group(2).lower(), urljoin(base, m.group(1)))
    return dict(sorted(out.items()))


def quarter_keys(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """'2013q1'..'2026q2' of every calendar quarter touching [start, end]."""
    return [f"{p.year}q{p.quarter}" for p in pd.period_range(start, end, freq="Q")]


def quarter_end(key: str) -> pd.Timestamp:
    m = _QKEY.match(key)
    if not m:
        raise ValueError(f"bad quarter key {key!r}")
    return pd.Period(f"{m.group(1)}Q{m.group(2)}", freq="Q").end_time.normalize()


def needs_download(cached_mtime: float | None, is_latest: bool, now: float, max_age_days: float = 7.0) -> bool:
    """Refresh rule of a quarter: download when nothing is cached; a closed quarter's cache is kept forever, the
    latest listed quarter's cache only while younger than ``max_age_days``."""
    if cached_mtime is None:
        return True
    if not is_latest:
        return False
    return now - cached_mtime > max_age_days * 86400


# ============================================================================ form 3/4/5 data sets (io)


def form345_cache_dir(settings: Settings) -> Path:
    return settings.cache_path / "alt" / "us_sec" / "form345"


def _mtime(p: Path) -> float | None:
    return p.stat().st_mtime if p.exists() else None


def _write_parquet(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    try:
        c = df[ROW_COLUMNS].copy()
        for col in ("doc_type", "code", "ad"):
            c[col] = c[col].astype("category")
        c.to_parquet(tmp, compression="zstd", index=False)
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def _index_html(net: PitNet, folder: Path, notes: list[str]) -> str:
    p = folder / "index.html"
    if p.exists() and time.time() - p.stat().st_mtime < INDEX_MAX_AGE_H * 3600:
        return p.read_text(encoding="utf-8", errors="replace")
    try:
        body = net.http_get(FORM345_INDEX, "sec").content
    except SecUserAgentError:
        raise
    except Exception as exc:  # noqa: BLE001 - a stale list is better than none, and it is noted
        if not p.exists():
            raise
        age_d = (time.time() - p.stat().st_mtime) / 86400
        notes.append(
            f"sec_form345: index page refresh failed ({type(exc).__name__}); using the {age_d:.1f}-day-old copy"
        )
        return p.read_text(encoding="utf-8", errors="replace")
    atomic_write(p, body)
    return body.decode("utf-8", errors="replace")


def _quarter_rows(
    net: PitNet, folder: Path, key: str, url: str, is_latest: bool, st: Counter, notes: list[str]
) -> pd.DataFrame | None:
    """Cached parquet of one quarter, downloaded + extracted when the refresh rule says so (None = failed)."""
    pq = folder / f"{key}.parquet"
    now = time.time()
    if not needs_download(_mtime(pq), is_latest, now, CURRENT_QUARTER_MAX_AGE_D):
        try:
            df = pd.read_parquet(pq)
            st["quarters_cached"] += 1
            return df
        except Exception as exc:  # noqa: BLE001 - a corrupt cache file is rebuilt
            log.warning("sec_form345: cache %s unreadable (%s) - downloading again", pq, exc)
            pq.unlink(missing_ok=True)
    zp = folder / f"{key}_form345.zip"
    try:
        body = net.http_get(url, "sec").content
        atomic_write(zp, body)
        st["quarters_downloaded"] += 1
        st["zip_bytes"] += len(body)
        rows, s = read_form345_zip(zp)
        st.update(s)
        _write_parquet(rows, pq)
        return rows
    except SecUserAgentError:
        raise
    except Exception as exc:  # noqa: BLE001 - one quarter is counted and noted, the rest continues
        log.warning("sec_form345: quarter %s failed (%s: %s)", key, type(exc).__name__, exc)
        if pq.exists():
            notes.append(f"sec_form345: refresh of {key} failed ({type(exc).__name__}); using the older cached copy")
            return pd.read_parquet(pq)
        st["quarters_failed"] += 1
        st[f"failed:{key}"] += 1
        return None
    finally:
        zp.unlink(missing_ok=True)  # only the compact parquet is kept


def _load_form345(
    net: PitNet,
    settings: Settings,
    ciks: list[int],
    start: pd.Timestamp,
    end: pd.Timestamp,
    say: Callable[[str], None],
) -> tuple[pd.DataFrame, dict[str, Any], list[str], list[str]]:
    """Events, coverage, notes and the covered entities (every asked CIK once any needed quarter loaded: the data
    sets list all issuers, so a CIK without a row had no open-market trade; failed quarters are noted)."""
    notes: list[str] = []
    st: Counter = Counter()
    folder = form345_cache_dir(settings)
    need = quarter_keys(start, end)
    cov: dict[str, Any] = {"ciks_asked": len(ciks), "quarters_needed": len(need)}
    try:
        links = parse_form345_index(_index_html(net, folder, notes))
        if not links:
            raise ValueError("no <yyyy>q<n>_form345.zip link on the index page")
    except SecUserAgentError:
        raise
    except Exception as exc:  # noqa: BLE001 - the whole source is skipped, loudly
        notes.append(
            f"sec_form345: the data-set index could not be read ({type(exc).__name__}: {exc}); no insider data"
        )
        cov.update(rows=0, quarters_failed=len(need), error=f"{type(exc).__name__}: {exc}")
        return empty_events(), cov, notes, []
    latest = max(links)
    listed = [k for k in need if k in links]
    unlisted = [k for k in need if k not in links]
    frames: list[pd.DataFrame] = []
    loaded = 0
    for i, key in enumerate(listed, 1):
        say(f"SEC Form 345 data set {key} ({i}/{len(listed)})")
        df = _quarter_rows(net, folder, key, links[key], key == latest, st, notes)
        if df is not None:
            loaded += 1
            frames.append(df[df["issuer_cik"].isin(set(ciks))].assign(quarter=key))
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*ROW_COLUMNS, "quarter"])
    events, es = form345_events(rows, ciks, start, end) if not rows.empty else (empty_events(), Counter())
    failed = sorted(k.split(":", 1)[1] for k in st if k.startswith("failed:"))
    through = quarter_end(latest)
    cov.update(
        {
            "ciks_with_trades": int(events["entity"].nunique()),
            "rows": len(events),
            "rows_by_kind": {k: int(v) for k, v in events["kind"].value_counts().sort_index().items()},
            "quarters_listed": len(listed),
            "quarters_cached": int(st["quarters_cached"]),
            "quarters_downloaded": int(st["quarters_downloaded"]),
            "quarters_failed": len(failed),
            "failed_quarters": failed,
            "unlisted_quarters": unlisted,
            "zip_mb_downloaded": round(st["zip_bytes"] / 1e6, 1),
            "covered_through": str(through.date()),
            "transactions": int(es["transactions"]),
            "dropped_bad_shares_or_price": int(es["bad_shares_or_price"]),
            "dropped_code_ad_mismatch": int(es["code_ad_mismatch"]),
            "dropped_amendment_restated": int(es["amendment_restated"]),
            "dropped_duplicate_rows": int(es["duplicate_rows"]),
            "skipped_not_form4": int(es["not_form4"]),
            "no_owner": int(es["no_owner"]),
            "bad_tsv_lines": int(sum(v for k, v in st.items() if k.startswith("bad_lines_"))),
            "no_submission_rows": int(st["no_submission"]),
        }
    )
    covered = [cik_entity(c) for c in ciks] if loaded else []
    cov["ciks_covered"] = len(covered)
    if failed:
        notes.append(
            f"sec_form345: {len(failed)} quarterly data sets failed and have no insider events ({', '.join(failed)})"
            + ("; the CIKs are still marked covered, so those quarters read as 'no trade'" if covered else "")
        )
    if not loaded and listed:
        notes.append("sec_form345: no quarterly data set could be loaded; no CIK is marked covered")
    early = [k for k in unlisted if k < FORM345_FIRST]
    late = [k for k in unlisted if k > latest]
    if early:
        notes.append(
            f"sec_form345: data sets start {FORM345_FIRST}; {len(early)} earlier quarters have no insider data"
        )
    if late or end > through:
        notes.append(
            f"sec_form345: insider trades are covered through {through.date()} (latest data set {latest}); trades "
            f"filed after that are missing (SEC posts the data sets quarterly; sec_forms 'form4' counts are current)"
        )
    if es["bad_shares_or_price"]:
        notes.append(
            f"sec_form345: {es['bad_shares_or_price']} P/S transactions without a positive share count and price "
            "were dropped"
        )
    if st["no_submission"] or cov["bad_tsv_lines"]:
        notes.append(
            f"sec_form345: {cov['bad_tsv_lines']} malformed TSV lines and {st['no_submission']} transactions without "
            "a usable SUBMISSION row were skipped (all issuers of the downloaded quarters)"
        )
    return events, cov, notes, covered


# ============================================================================ entry point


def _as_cik(c: Any) -> int:
    return int(str(c).strip().upper().removeprefix("CIK"))


def load_us_sec_alt(
    settings: Settings,
    ciks: Iterable[int],
    start: date,
    end: date | None = None,
    *,
    net: PitNet | None = None,
    form345: bool = True,
    progress: Callable[[str], None] | None = None,
) -> AltData:
    """US alternative-data events of ``ciks`` published in [start, end] (``end`` None = today): ``sec_forms``
    (submissions forms / 8-K items) and, with ``form345``, ``sec_form345`` (insider open-market trades).

    A CIK or quarter that fails is counted in ``coverage`` and noted; the rest continues. An empty
    ``settings.data.sec_user_agent`` raises ``SecUserAgentError`` as soon as anything must be downloaded.

    With ``settings.modeling.vintage`` the ``sec_forms`` events go through their first-seen vintage
    (``freeze_form_events``, store ``settings.cache_path/vintage``, one namespace per start year: the submissions
    are read from Jan 1 of ``start``'s year and the events cut to [start, end] afterwards); ``coverage["sec_forms"]
    ["vintage"]`` counts what it did. ``sec_form345`` is not frozen: its latest quarter is posted late, so a frozen
    copy would keep the missing trades.
    """
    net = net if net is not None else PitNet(settings)

    def say(msg: str) -> None:
        log.info("us_sec_alt: %s", msg)
        if progress is not None:
            progress(msg)

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end if end is not None else date.today()).normalize()
    if end_ts < start_ts:
        raise ValueError(f"end {end_ts.date()} is before start {start_ts.date()}")
    cik_list = sorted({_as_cik(c) for c in ciks})
    n_net_notes = len(net.notes)
    store = (
        VintageStore(settings.cache_path / "vintage", settle_days=int(settings.modeling.vintage_settle_days))
        if settings.modeling.vintage
        else None
    )
    load_start = pd.Timestamp(start_ts.year, 1, 1) if store is not None else start_ts
    ev_forms, cov_forms, notes, cov_ent, fetched_on = _load_forms(net, cik_list, load_start, end_ts, say)
    if store is not None:
        cut = store.cut
        complete = set(cov_ent) if end_ts >= cut else set()  # a window ending before the cutoff is never stored
        eligible = [e for e in complete if e in fetched_on and fetched_on[e] > cut]
        asked = [cik_entity(c) for c in cik_list]
        ev_forms, cov_forms["vintage"], v_notes = freeze_form_events(
            store, ev_forms, start=load_start, asked=asked, eligible=eligible
        )
        ev_forms = ev_forms[(ev_forms["published"] >= start_ts) & (ev_forms["published"] <= end_ts)]
        cov_forms.update(_event_counts(ev_forms))
        notes += v_notes
    frames = [ev_forms]
    coverage: dict[str, Any] = {SOURCE_FORMS: cov_forms}
    covered: dict[str, list[str]] = {SOURCE_FORMS: cov_ent}
    if form345:
        ev_345, cov_345, notes_345, cov_ent_345 = _load_form345(net, settings, cik_list, start_ts, end_ts, say)
        frames.append(ev_345)
        coverage[SOURCE_FORM345] = cov_345
        covered[SOURCE_FORM345] = cov_ent_345
        notes += notes_345
    else:
        coverage[SOURCE_FORM345] = {"ciks_asked": len(cik_list), "rows": 0, "skipped": True}
        covered[SOURCE_FORM345] = []
        notes.append("sec_form345: not requested (form345=False); no insider buy/sell events")
    notes += [f"pit-us net: {n}" for n in net.notes[n_net_notes:]]
    events = normalise_events(pd.concat(frames, ignore_index=True))
    unknown = sorted(set(events["kind"]) - set(KINDS))
    if unknown:  # programming error: every kind must be pre-declared
        raise RuntimeError(f"undeclared event kinds {unknown}")
    return AltData("US", events, coverage=coverage, notes=notes, covered=covered)


__all__ = [
    "FORM_KINDS",
    "INSIDER_KINDS",
    "ITEMS_8K",
    "ITEM_KINDS",
    "KINDS",
    "SOURCE_FORM345",
    "SOURCE_FORMS",
    "acceptance_ny_dates",
    "form345_cache_dir",
    "form345_events",
    "form_kind",
    "forms_vintage_ns",
    "freeze_form_events",
    "load_us_sec_alt",
    "needs_download",
    "parse_form345_index",
    "parse_form345_tables",
    "parse_submission_blocks",
    "published_dates",
    "quarter_keys",
    "read_form345_zip",
    "submission_blocks",
]
