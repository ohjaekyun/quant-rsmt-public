"""US mid-cap universe for the cross-sectional holdout (XSH): S&P MidCap 400 members that were never S&P 500 members.

Every US name used for tuning so far is a point-in-time S&P 500 member since ``settings.backtest.us_start``
(``quantcio.data.pit.us``). The untouched US set is therefore the S&P 400 minus every entity (SEC CIK) that was an
S&P 500 member at any time since then. This module builds it in the shared ``PITUniverse`` contract
(``quantcio.data.pit``), reusing the S&P 500 pipeline (``us._Builder``) for CIKs, SEC-FTD-validated Yahoo prices,
SEC XBRL shares and PIT market-cap ranks. Nothing here relates a feature to a future return (holdout hygiene).

Membership (verified live 2026-09-27, docs/sources/pit/us-midcap.json)
  1. Wikipedia 'List of S&P 400 companies': current constituents (Symbol, Security, GICS Sector, GICS Sub-Industry,
     Headquarters Location, SEC filings; the SEC link carries a numeric CIK for ~30% of rows, the rest map through
     SEC company_tickers.json) and 'Selected past and announced changes' (Date, Added Ticker/Security, Removed
     Ticker/Security, Reason; 623 rows from 2012-01-13, sparse before 2016). Month-end membership is reconstructed
     BACKWARD from today's set by undoing every dated change, keyed by SEC CIK (``us_cik.cik_for_event`` for removed
     names) so that ticker renames (PSTG -> P, GPS -> GAP, CHK -> EXE) and reused tickers (COHR, HR, VAL, CZR) do not
     break the walk; a renamed spell is split at the rename (last SEC FTD print of the old ticker's CUSIP).
  2. The month-end revisions of the same page (the table as editors kept it then; PIT information) fill what the
     change table lacks, all counted in ``coverage['membership_checks']``: the end of an addition whose removal row is
     missing, the start of a re-addition without a row, the start of issuers open "since before 2012" without an
     addition row (spin-offs, new-CIK successors: first month-end of their final run in up-to-date revisions after
     >= 3 not listing them) and issuers the table never mentions but the revisions list (removed losers such as Akorn
     and Office Depot, old CIKs of Crane Co. / Aaron's; only when the ticker maps to a CIK via the dated SEC ticker
     files).
  3. Validation: every month-end is compared with the complete, up-to-date page revision (it reflects >= 80% of the
     changes dated 7-90 days before it; the page went unedited for months in 2019-2020), keyed by issuer.
     ``coverage['pit_from']`` is the first month-end from which they agree (Jaccard >= ``PIT_MIN_AGREEMENT``, at most
     one miss in any ``PIT_WINDOW`` validating month-ends, never below ``PIT_FLOOR``). Earlier month-ends use the same
     reconstruction but are SURVIVORSHIP-BIASED (missing change rows: today's survivors appear too early, removed
     names are missing): ``coverage['survivorship_biased']`` / ``coverage['biased_dates']``.
  4. Exclusion: every entity of the PIT S&P 500 build since ``backtest.us_start`` (and the predecessor CIKs of its
     chains), every issuer that carried a ticker an S&P 500 member carried (reused tickers, unmapped S&P 500 spells:
     ``holdout.tuning_names`` compares tickers too), every issuer in today's S&P 500 table (the next tuning build will
     contain it) and the caller's ``exclude`` (tickers or entity ids), all counted.
     NOTE: excluding names that were promoted to the S&P 500 LATER uses future information (their mid-cap months are
     removed too): the XSH cross-section under-represents future large-cap winners.

Prices / ranks / fundamentals: exactly the S&P 500 path (``us._Builder``): Yahoo series validated against SEC
fails-to-deliver prices of the ticker traded then (``attrs['adjusted']``), FTD month-end fills and terminal marks,
rank = XBRL shares (filed <= t) x price, or 60-session dollar volume when shares cover < 80% of a date's members.
Benchmark: SPY for every entity (the frozen model was trained on SPY-beta-adjusted excess returns). No RSP tracking
check (it would compute returns of holdout names; the universe is not an index either).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import date
from typing import Any
from urllib.parse import unquote

import numpy as np
import pandas as pd

from quantcio.config import Settings
from quantcio.data.base import normalise_ticker
from quantcio.data.cache import DiskCache
from quantcio.data.pit import MEMBERSHIP_COLUMNS, PITUniverse, us_cik, us_membership, us_prices
from quantcio.data.pit import us as pit_us
from quantcio.data.pit.us_calendar import month_end_sessions, next_sessions, nyse_sessions
from quantcio.data.pit.us_membership import MembershipError, dash, ftd_symbol
from quantcio.data.pit.us_net import PitNet, require_sec_user_agent
from quantcio.models import Market

log = logging.getLogger(__name__)

MIDCAP_VERSION = 2
INDEX_SIZE = 400
WIKI_400 = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_TITLE = "List of S&P 400 companies"
RAW_DIR = "sp400"  # raw files live under the PitNet 'sp500' source directory (<cache>/pit/us/sp500_v1/sp400/)
CURRENT_COLUMNS = ["symbol", "security", "sector", "sub_industry", "hq", "sec_filings"]
CHANGE_COLUMNS = ["date", "add_t", "add_n", "add_link", "rem_t", "rem_n", "rem_link", "reason"]
COUNT_RANGE = (390, 410)  # month-end member counts outside this are noted
COUNT_HARD_RANGE = (300, 500)  # outside this the sources are broken -> raise
MAX_CURRENT_ROWS = (380, 420)  # today's table outside this -> raise (parser/layout break)
MIN_REVISION_ROWS = 300  # a page revision's constituents table needs this many rows to be parsed at all
REVISION_COMPLETE = (388, 412)  # rows of a usable (complete) revision table; partial tables (2013-08: 376) are ignored
FRESH_MIN = 0.8  # a revision is up to date when it reflects >= this share of the changes dated 7-90 days before it
FRESH_WINDOW = (7, 90)
PIT_MIN_AGREEMENT = 0.95  # CIK-keyed Jaccard(reconstruction, page revision) of a month-end counted as agreeing
PIT_WINDOW = 10  # at most one validating month-end below PIT_MIN_AGREEMENT in any this many consecutive ones
PIT_FLOOR = 0.85  # no month-end from pit_from on may agree less than this
START_UNKNOWN = pd.Timestamp("1990-01-01")  # spells already open before the first dated change
TRIM_MIN_ABSENT = 3  # month-end revisions not listing an issuer right before its final run -> a later addition
VALIDATION_MAX_LAG = 2  # month-ends a page revision may lag the index (editors update weeks after the change)
FAR = pd.Timestamp("2200-01-01")  # 'still open' in date arithmetic (Timestamp.max overflows)
BUILD_MAX_AGE_DAYS = 3
_CAT_FAILED = r"bankrupt|chapter 11|receivership"
_CAT_ACQ = r"acqui|merg|purchas|privat|bought|buyout|take-private"


# ============================================================================ public API


def load_us_midcap_universe(
    settings: Settings,
    cache: DiskCache,
    start: date,
    end: date | None = None,
    exclude: Iterable[str] = (),
    *,
    include_sp500: bool = False,
) -> PITUniverse:
    """S&P MidCap 400 members that were never PIT S&P 500 members since ``settings.backtest.us_start``, month-end
    rebalance dates in [start, end] (see the module docstring). ``exclude``: extra tickers or entity ids
    ("CIK##########") dropped on every date (e.g. ``holdout.tuning_names``); counted in ``coverage['excluded']``.
    ``include_sp500=True`` skips the S&P 500 exclusion: every PIT S&P 400 member of each date, including issuers
    that were S&P 500 members at other times (no future-information condition; NOT an untouched holdout - used to
    train / evaluate a mid-cap model, docs/MODELING.md 19). Cached separately.

    Read ``coverage['pit_from']``: month-ends before it are SURVIVORSHIP-BIASED (``coverage['survivorship_biased']``).
    Needs ``settings.data.sec_user_agent``. Cold cost (2026-09-27): ~1,500 Wikipedia/SEC/Yahoo requests plus the PIT
    S&P 500 build it loads for the exclusion; warm: served from ``<cache>/pit/us/build_v1/midcap_*.pkl``.
    Scripts calling it must guard with ``if __name__ == "__main__":`` (the XBRL step uses a process pool).
    """
    require_sec_user_agent(settings)
    net = PitNet(settings)
    full = midcap_build(settings, cache, net, pd.Timestamp(start).normalize(), _ts(end), include_sp500=include_sp500)
    return to_universe(full, exclude)


# ============================================================================ Wikipedia parsing


def _cell(x: Any) -> tuple[str | None, str | None]:
    """``read_html(extract_links='body')`` cell -> (text, link)."""
    if isinstance(x, tuple):
        text, link = x
    else:
        text, link = x, None
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None, link
    text = re.sub(r"\[[^\]]*\]", "", str(text)).strip()
    return text or None, link


def _wiki_title(link: str | None) -> str | None:
    if not link or "/wiki/" not in link or "redlink" in link:
        return None
    return unquote(link.split("/wiki/")[-1].split("#")[0]) or None


def _tickers(text: str | None) -> list[str]:
    """'UA/UAA' (two share classes in one row) -> ['UA', 'UAA']; share classes spelled with '-' ('MOG.A' -> 'MOG-A')."""
    if not text:
        return []
    parts = [re.sub(r"[^A-Za-z0-9.\-]", "", p) for p in re.split(r"[/,]", str(text))]
    return [dash(p) for p in parts if p]


def parse_wiki400(html: bytes | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(current constituents, dated changes) of the Wikipedia 'List of S&P 400 companies' page.

    current: symbol, sym (dash spelling), security, sector, sub_industry, cik (Int64 from the SEC link), link.
    changes: CHANGE_COLUMNS (tickers raw; ``_tickers`` splits multi-class cells), newest first.
    Raises MembershipError when either table's layout changed (never a silent column shift).
    """
    text = html.decode() if isinstance(html, bytes) else html
    try:
        tables = pd.read_html(io.StringIO(text), extract_links="body", flavor="lxml")
    except ValueError as exc:
        raise MembershipError(f"Wikipedia S&P 400 페이지에서 표를 읽지 못함: {exc}") from exc
    cur = chg = None
    for t in tables:
        cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        low = [c.lower() for c in cols]
        if cur is None and low and low[0] == "symbol" and "security" in low:
            if len(cols) != len(CURRENT_COLUMNS):
                raise MembershipError(f"Wikipedia S&P 400 구성종목 표 형식 변경: {cols}")
            cur = t
        elif chg is None and isinstance(t.columns, pd.MultiIndex) and len(cols) == 6 and "added ticker" in low:
            if low != [
                "date date",
                "added ticker",
                "added security",
                "removed ticker",
                "removed security",
                "reason reason",
            ]:
                raise MembershipError(f"Wikipedia S&P 400 변경 표 형식 변경: {cols}")
            chg = t
    if cur is None or chg is None:
        raise MembershipError("Wikipedia S&P 400 페이지에 구성종목/변경 표 없음 (형식 변경?)")
    rows = []
    for vals in cur.itertuples(index=False):
        sym, _ = _cell(vals[0])
        sec, link = _cell(vals[1])
        _, sec_link = _cell(vals[5])
        m = re.search(r"CIK=(\d+)", sec_link or "")
        if not sym:
            continue
        rows.append(
            dict(
                symbol=sym,
                sym=dash(sym),
                security=sec,
                sector=_cell(vals[2])[0],
                sub_industry=_cell(vals[3])[0],
                cik=int(m.group(1)) if m else None,
                link=_wiki_title(link),
            )
        )
    current = pd.DataFrame(rows)
    current["cik"] = pd.array(current["cik"], dtype="Int64")
    out = []
    for vals in chg.itertuples(index=False):
        d_txt, _ = _cell(vals[0])
        try:
            d = pd.to_datetime(d_txt, format="%B %d, %Y")
        except (ValueError, TypeError):
            continue
        add_t, _ = _cell(vals[1])
        add_n, add_l = _cell(vals[2])
        rem_t, _ = _cell(vals[3])
        rem_n, rem_l = _cell(vals[4])
        reason, _ = _cell(vals[5])
        if not add_t and not rem_t:
            continue
        out.append(
            dict(
                date=d,
                add_t=add_t,
                add_n=add_n,
                add_link=_wiki_title(add_l),
                rem_t=rem_t,
                rem_n=rem_n,
                rem_link=_wiki_title(rem_l),
                reason=reason,
            )
        )
    if not out:
        raise MembershipError("Wikipedia S&P 400 변경 표에서 행을 읽지 못함 (형식 변경?)")
    changes = pd.DataFrame(out, columns=CHANGE_COLUMNS).sort_values("date", ascending=False, kind="stable")
    return current, changes.reset_index(drop=True)


def parse_revision(html: str) -> dict[str, str] | None:
    """{ticker: security name} of the constituents table in an old revision of the page (column names changed over
    the years: 'Ticker Symbol' / 'Ticker symbol' / 'Symbol', 'Company' / 'Security'); None when no such table with
    >= MIN_REVISION_ROWS rows exists."""
    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except ValueError:
        return None
    for t in tables:
        cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        k = [i for i, c in enumerate(cols) if re.fullmatch(r"(ticker )?symbol", c.strip(), re.I)]
        n = [i for i, c in enumerate(cols) if re.fullmatch(r"company|security", c.strip(), re.I)]
        if k and len(t) >= MIN_REVISION_ROWS:
            out: dict[str, str] = {}
            for _, row in t.iterrows():
                name = str(row.iloc[n[0]]) if n and pd.notna(row.iloc[n[0]]) else ""
                for tk in _tickers(str(row.iloc[k[0]])) if pd.notna(row.iloc[k[0]]) else []:
                    out[tk] = name
            return out
    return None


def parse_revision_tickers(html: str) -> set[str] | None:
    rev = parse_revision(html)
    return None if rev is None else set(rev)


def classify_reason(reason: str | None) -> str:
    """Informational label of a removal (failed / acquired / demoted = moved to another index or market cap)."""
    s = str(reason or "").lower()
    if re.search(_CAT_FAILED, s):
        return "failed"
    if re.search(_CAT_ACQ, s):
        return "acquired"
    return "demoted" if s else "unknown"


# ============================================================================ backward reconstruction


def reconstruct_spells(
    current: pd.DataFrame,
    changes: pd.DataFrame,
    today: pd.Timestamp,
    resolve: Callable[[str, str | None, str | None, pd.Timestamp, str], tuple[int | None, str]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Membership spells [start, end) (end = first day out) by undoing the dated changes backward from today's set.

    ``current`` needs sym, security, link and cik (<NA> = resolve via ``resolve(ticker, name, link, date, side)``).
    Undo an addition of X at d: the open spell with ticker X; else the open spell whose CIK is X's CIK at d (a later
    ticker rename: alias X recorded, split later); else a unique strict name match; else X's removal is missing from
    the table (``end_unknown``). Undo a removal of Y at d: open a spell ending at d, unless the same issuer is open
    later (re-added without an addition row: that later spell's start is unknown). Returns (spells, anomalies)."""
    today = pd.Timestamp(today).normalize()
    active: list[dict[str, Any]] = []
    done: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []

    def new(ticker, cik, how, name, link, end=pd.NaT, reason=None):
        return dict(
            ticker=ticker,
            cik=cik,
            cik_method=how,
            name=name,
            link=link,
            start=pd.NaT,
            end=end,
            reason=reason,
            alias=None,
            alias_name=None,
            start_unknown=False,
            end_unknown=False,
            supplement=False,
        )

    for r in current.itertuples(index=False):
        cik = int(r.cik) if pd.notna(r.cik) else None
        how = "wiki_current"
        if cik is None:
            cik, how = resolve(r.sym, r.security, r.link, today, "current")
        active.append(new(r.sym, cik, how, r.security, r.link))

    def later(s: dict, d: pd.Timestamp) -> bool:  # open after d (not a spell opened by a removal on d itself)
        return pd.isna(s["end"]) or s["end"] > d

    todo = changes[changes["date"] <= today].sort_values("date", ascending=False, kind="stable")
    for d, g in todo.groupby("date", sort=False):
        for r in g.itertuples(
            index=False
        ):  # undo additions first: a same-day add+remove of one issuer stays contiguous
            for x in _tickers(r.add_t):
                sp, how = _match_addition(active, x, r, d, resolve)
                if sp is not None:
                    active.remove(sp)
                    sp["start"] = d
                    if how != "ticker":
                        sp.update(alias=x, alias_name=r.add_n)
                        anomalies.append(dict(kind=f"rename_{how}", date=d, ticker=x, now=sp["ticker"]))
                    done.append(sp)
                    continue
                c, m = resolve(x, r.add_n, r.add_link, d, "add")
                sp = new(x, c, m, r.add_n, r.add_link)
                sp.update(start=d, end_unknown=True)
                done.append(sp)
                anomalies.append(dict(kind="missing_removal", date=d, ticker=x, now=None))
        for r in g.itertuples(index=False):
            for y in _tickers(r.rem_t):
                c, m = resolve(y, r.rem_n, r.rem_link, d, "remove")
                again = [
                    s
                    for s in active
                    if later(s, d)
                    and ((c is not None and s["cik"] == c) or (s["ticker"] == y and (c is None or s["cik"] is None)))
                ]
                for s in again:  # the issuer is a member again later without an addition row
                    active.remove(s)
                    s.update(start=d, start_unknown=True)
                    done.append(s)
                    anomalies.append(dict(kind="missing_addition", date=d, ticker=y, now=s["ticker"]))
                active.append(new(y, c, m, r.rem_n, r.rem_link, end=d, reason=r.reason))
    for s in active:
        s["start"] = START_UNKNOWN
    out = pd.DataFrame(done + active)
    out["start"] = pd.to_datetime(out["start"])
    out["end"] = pd.to_datetime(out["end"])
    return out.sort_values(["ticker", "start"], ignore_index=True), anomalies


def _match_addition(active: list[dict], x: str, r: Any, d: pd.Timestamp, resolve: Callable) -> tuple[dict | None, str]:
    """The open spell an addition of ticker ``x`` on ``d`` started: same ticker (the nearest in time when a reused
    ticker is open twice, unless the CIK tells them apart), else the same CIK (renamed later), else one exact name."""
    same_t = [s for s in active if s["ticker"] == x]
    if len(same_t) == 1:
        return same_t[0], "ticker"
    c, _ = resolve(x, r.add_n, r.add_link, d, "add")
    if same_t:
        by_cik = [s for s in same_t if c is not None and s["cik"] == c]
        pick = by_cik or sorted(same_t, key=lambda s: s["end"] if pd.notna(s["end"]) else FAR)
        return pick[0], "ticker"
    same_c = sorted((s for s in active if c is not None and s["cik"] == c), key=lambda s: s["ticker"])
    if same_c:
        return same_c[0], "cik"
    same_n = [s for s in active if r.add_n and s["name"] and _same_name(r.add_n, s["name"])]
    if len(same_n) == 1:
        return same_n[0], "name"
    return None, ""


def _same_name(a: str, b: str) -> bool:
    sa, sb = us_cik.norm_name(a).replace(" ", ""), us_cik.norm_name(b).replace(" ", "")
    return bool(sa) and sa == sb


class RevisionKeys:
    """Issuer key ('CIK<n>', else 'T:<ticker>') of a page-revision row (ticker, security name, month-end): the
    reconstruction's own spell with that ticker or alias around the date, else the SEC ticker file captured nearest
    the date (<= 45 days, 2017-08 on), else a unique exact name match with a spell within a year, else today's SEC
    ticker file."""

    def __init__(self, spells: pd.DataFrame, pairs: pd.DataFrame, sec_cur: dict[str, int]):
        self.by_t: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, int]]] = {}
        self.by_n: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, int]]] = {}
        for s in spells.itertuples(index=False):
            if s.cik is None or pd.isna(s.cik):
                continue
            hi = s.end if pd.notna(s.end) else FAR
            for t in {s.ticker, s.alias} - {None}:
                self.by_t.setdefault(t, []).append((s.start, hi, int(s.cik)))
            for nm in {s.name, getattr(s, "alias_name", None)} - {None}:
                if isinstance(nm, str) and _squash(nm):
                    self.by_n.setdefault(_squash(nm), []).append((s.start, hi, int(s.cik)))
        self.pairs: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, int]]] = {}
        for r in pairs.itertuples(index=False):
            self.pairs.setdefault(dash(r.ticker), []).append((r.first_seen, r.last_seen, int(r.cik)))
        self.sec_cur = sec_cur
        self.memo: dict[tuple, tuple[int | None, str]] = {}

    def __call__(self, t: str, name: str | None, d: pd.Timestamp) -> str:
        return _key(self.cik_how(t, name, d)[0], t)

    def cik_how(self, t: str, name: str | None, d: pd.Timestamp) -> tuple[int | None, str]:
        k = (t, name, d)
        if k not in self.memo:
            self.memo[k] = self._cik(t, name, d)
        return self.memo[k]

    def _cik(self, t: str, name: str | None, d: pd.Timestamp) -> tuple[int | None, str]:
        for lo, hi, c in self.by_t.get(t, []):
            if lo - pd.Timedelta(92, "D") <= d < hi + pd.Timedelta(92, "D"):
                return c, "spell"
        near = sorted((_gap_days(lo, hi, d), c) for lo, hi, c in self.pairs.get(t, []))
        if near and near[0][0] <= 45:
            return near[0][1], "sec_ticker_file"
        if name and _squash(name):
            hits = {c for lo, hi, c in self.by_n.get(_squash(name), []) if _gap_days(lo, hi, d) <= 366}
            if len(hits) == 1:
                return hits.pop(), "name"
        c = self.sec_cur.get(t)
        return (c, "sec_current") if c else (None, "none")


def _squash(name: str) -> str:
    return us_cik.norm_name(name).replace(" ", "")


def revision_keys(snapshots: dict[pd.Timestamp, dict[str, str]], keys: Callable) -> dict[pd.Timestamp, set[str]]:
    return {d: {keys(t, nm, d) for t, nm in rev.items()} for d, rev in snapshots.items()}


def _spell_key(s: Any) -> str:
    return _key(s["cik"], s["ticker"])


def dated_events(spells: pd.DataFrame) -> list[tuple[pd.Timestamp, str, str]]:
    """(date, issuer key, 'add'|'remove') of every membership change the change table dates (not the bridged ones)."""
    ev = []
    for _, s in spells.iterrows():
        k = _spell_key(s)
        if s["start"] != START_UNKNOWN and not s["start_unknown"]:
            ev.append((s["start"], k, "add"))
        if pd.notna(s["end"]) and not s["end_unknown"]:
            ev.append((s["end"], k, "remove"))
    return ev


def revision_freshness(
    events: list[tuple[pd.Timestamp, str, str]], rev_keys: dict[pd.Timestamp, set[str]]
) -> dict[pd.Timestamp, float]:
    """Per revision: the share of the changes dated FRESH_WINDOW days before it that it already reflects (an addition
    listed, a removal gone); NaN when nothing changed then. Stale revisions (the page went unedited for months in
    2019-2020 while the index changed) cannot validate a month-end."""
    out = {}
    lo, hi = pd.Timedelta(FRESH_WINDOW[1], "D"), pd.Timedelta(FRESH_WINDOW[0], "D")
    for d, keys in rev_keys.items():
        recent = [(k, kind) for t, k, kind in events if d - lo < t <= d - hi]
        ok = [(k in keys) == (kind == "add") for k, kind in recent]
        out[d] = float(np.mean(ok)) if ok else float("nan")
    return out


def usable_revisions(rev_keys: dict[pd.Timestamp, set[str]]) -> dict[pd.Timestamp, set[str]]:
    """Complete revision tables only (a partial table would read as dozens of removals)."""
    return {d: k for d, k in rev_keys.items() if REVISION_COMPLETE[0] <= len(k) <= REVISION_COMPLETE[1]}


def bridge_unknowns(spells: pd.DataFrame, rev_keys: dict[pd.Timestamp, set[str]]) -> tuple[pd.DataFrame, dict]:
    """Dates the change table does not give, from the month-end page revisions (issuer keys): a missing removal ends
    at the first month-end after the revisions first list the issuer on which they no longer do; a missing addition
    starts at the first month-end whose revision lists it. Without revision evidence the spell covers no month-end."""
    sp = spells.copy()
    dates = sorted(rev_keys)
    stats: Counter = Counter()
    for i, s in sp[sp["end_unknown"]].iterrows():
        k = _spell_key(s)
        after = [d for d in dates if d > s["start"]]
        present = [d for d in after if k in rev_keys[d]]
        if not present:
            stats["removal_unresolved"] += 1
            sp.loc[i, "end"] = after[0] if after else s["start"] + pd.Timedelta(1, "D")
            continue
        end = next((d for d in after if d > present[0] and k not in rev_keys[d]), None)
        stats["removal_bridged" if end is not None else "removal_listed_in_last_revision"] += 1
        sp.loc[i, "end"] = end if end is not None else after[-1] + pd.Timedelta(1, "D")
    for i, s in sp[sp["start_unknown"]].iterrows():
        k = _spell_key(s)
        hi = s["end"] if pd.notna(s["end"]) else FAR
        first = next((d for d in dates if s["start"] < d < hi and k in rev_keys[d]), None)
        if first is None:
            stats["addition_unresolved"] += 1
            sp.loc[i, "start"] = hi if pd.notna(s["end"]) else (dates[-1] if dates else s["start"])
        else:
            stats["addition_bridged"] += 1
            sp.loc[i, "start"] = first
    return sp, dict(stats)


def trim_open_starts(
    spells: pd.DataFrame, rev_keys: dict[pd.Timestamp, set[str]], min_absent: int = TRIM_MIN_ABSENT
) -> tuple[pd.DataFrame, dict]:
    """Spells open since before the first dated change (no addition row) start at the first month-end of the
    revisions' final unbroken run of the issuer when >= ``min_absent`` month-end revisions right before that run do
    not list it (a later addition the change table missed: a spin-off, a new-CIK successor). Otherwise unchanged."""
    sp = spells.copy()
    dates = sorted(rev_keys)
    stats: Counter = Counter()
    for i, s in sp[sp["start"] == START_UNKNOWN].iterrows():
        k = _spell_key(s)
        hi = s["end"] if pd.notna(s["end"]) else FAR
        seq = [(d, k in rev_keys[d]) for d in dates if d < hi]
        if not seq or not seq[-1][1]:
            stats["no_revision_evidence" if not any(p for _, p in seq) else "kept"] += 1
            continue
        j = len(seq) - 1
        while j > 0 and seq[j - 1][1]:
            j -= 1
        absent = 0
        while j - absent - 1 >= 0 and not seq[j - absent - 1][1]:
            absent += 1
        if j > 0 and absent >= min_absent:
            sp.loc[i, "start"] = seq[j][0]
            stats["trimmed"] += 1
        else:
            stats["kept"] += 1
    return sp, dict(stats)


def revision_supplements(
    spells: pd.DataFrame, snapshots: dict[pd.Timestamp, dict[str, str]], keys: RevisionKeys
) -> tuple[pd.DataFrame, dict]:
    """Issuers the complete page revisions list for >= 2 consecutive month-ends but the change table never mentions
    (removals it missed: Akorn, Office Depot; issuers of an old CIK: Crane Co., Aaron's Inc.). Added only when the
    ticker maps to a CIK through the dated SEC ticker files during the run and that CIK has no spell at all; the spell
    covers the run (member from its first listed month-end, out from the first month-end no longer listing it)."""
    dates = sorted(snapshots)
    known = {int(c) for c in spells["cik"].dropna()}
    runs: dict[str, list[list[pd.Timestamp]]] = {}
    for i, d in enumerate(dates):
        for t in snapshots[d]:
            r = runs.setdefault(t, [])
            if i and r and r[-1][-1] == dates[i - 1]:
                r[-1].append(d)
            else:
                r.append([d])
    rows, stats = [], Counter()
    for t, rr in runs.items():
        for run in rr:
            if len(run) < 2:
                continue
            got = Counter()
            for d in run:
                c, how = keys.cik_how(t, snapshots[d].get(t), d)
                if c is not None and how == "sec_ticker_file":
                    got[c] += 1
            if not got:
                continue
            c = got.most_common(1)[0][0]
            if c in known:
                continue
            after = [d for d in dates if d > run[-1]]
            if not after:
                continue
            stats["spells"] += 1
            rows.append(
                dict(
                    ticker=t,
                    cik=c,
                    cik_method="wiki_revision_sec_ticker_file",
                    name=snapshots[run[-1]].get(t) or t,
                    link=None,
                    start=START_UNKNOWN if run[0] == dates[0] else run[0],
                    end=after[0],
                    reason=None,
                    alias=None,
                    alias_name=None,
                    start_unknown=False,
                    end_unknown=False,
                    supplement=True,
                )
            )
    out = pd.DataFrame(rows, columns=list(spells.columns) + (["supplement"] if "supplement" not in spells else []))
    stats["issuers"] = int(out["cik"].nunique()) if len(out) else 0
    return out, dict(stats)


def members_at(spells: pd.DataFrame, when: pd.Timestamp) -> pd.DataFrame:
    return us_membership.members_at(spells, when)


def snapshot_agreement(
    spells: pd.DataFrame, rev_keys: dict[pd.Timestamp, set[str]], lags: int = VALIDATION_MAX_LAG
) -> pd.DataFrame:
    """Per month-end: members of the reconstruction vs the page revision in force then, both keyed by issuer (a ticker
    rename is not a disagreement): date, n_rec, n_rev, only_rec, only_rev, agreement (Jaccard) and agreement_lag (the
    best Jaccard against the revisions of that and the next ``lags`` month-ends: editors lag the index by weeks)."""
    dates = sorted(rev_keys)
    rows = []
    for i, d in enumerate(dates):
        m = members_at(spells, d)
        rec = {_key(c, t) for c, t in zip(m["cik"], m["ticker"], strict=False)}
        rev = rev_keys[d]
        js = [_jaccard(rec, rev_keys[x]) for x in dates[i : i + lags + 1]]
        rows.append(
            dict(
                date=d,
                n_rec=len(rec),
                n_rev=len(rev),
                only_rec=len(rec - rev),
                only_rev=len(rev - rec),
                agreement=js[0],
                agreement_lag=max(js),
            )
        )
    return pd.DataFrame(rows, columns=["date", "n_rec", "n_rev", "only_rec", "only_rev", "agreement", "agreement_lag"])


def _jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return round(len(a & b) / u, 4) if u else float("nan")


def _by_year(agree: pd.DataFrame, col: str) -> dict[int, float]:
    if agree.empty:
        return {}
    return {int(y): round(float(g[col].median()), 4) for y, g in agree.groupby(agree["date"].dt.year)}


def _gap_days(lo: pd.Timestamp, hi: pd.Timestamp, d: pd.Timestamp) -> int:
    """Days from ``d`` to the interval [lo, hi] (0 inside)."""
    return 0 if lo <= d <= hi else int(min(abs((lo - d).days), abs((hi - d).days)))


def _key(cik: Any, ticker: str) -> str:
    return f"CIK{int(cik)}" if cik is not None and pd.notna(cik) else f"T:{ticker}"


def pit_from_date(agreement: pd.DataFrame, col: str = "agreement_lag") -> pd.Timestamp | None:
    """First month-end from which every validating month-end agrees >= PIT_FLOOR and at most one in any PIT_WINDOW
    consecutive ones falls below PIT_MIN_AGREEMENT (a late-edited revision around a big rebalance is tolerated, a
    drift is not); the first one itself must agree >= PIT_MIN_AGREEMENT. Validating = ``fresh`` (an up-to-date
    revision) when that column exists: stale revisions neither pass nor fail a month-end."""
    a = agreement.dropna(subset=[col]).sort_values("date")
    if "fresh" in a:
        a = a[a["fresh"].astype(bool)]
    if a.empty:
        return None
    vals = a[col].to_numpy()
    best = None
    last_miss = None  # index of the nearest miss after i
    for i in range(len(a) - 1, -1, -1):
        if vals[i] < PIT_FLOOR:
            break
        if vals[i] < PIT_MIN_AGREEMENT:
            if last_miss is not None and last_miss - i < PIT_WINDOW:
                break
            last_miss = i
            continue
        best = a["date"].iloc[i]
    return pd.Timestamp(best) if best is not None else None


# ============================================================================ network loaders


def load_wiki400(net: PitNet, max_age_h: float | None = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    return parse_wiki400(net.cached_bytes("sp500", f"{RAW_DIR}/wiki_sp400.html", WIKI_400, "wiki", max_age_h))


def revision_list(net: PitNet, max_age_h: float | None = 24 * 7) -> pd.DataFrame:
    """revid, timestamp (tz-naive UTC) of every revision of the page (MediaWiki API, 500 per call)."""
    rel = f"{RAW_DIR}/revisions.json"
    p = net.path("sp500", rel)
    hit = net.read_json("sp500", rel)
    if hit is not None and (max_age_h is None or time.time() - p.stat().st_mtime < max_age_h * 3600):
        revs = hit
    else:
        revs, cont = [], {}
        try:
            while True:
                params = dict(
                    action="query",
                    prop="revisions",
                    titles=WIKI_TITLE,
                    rvlimit=500,
                    rvprop="ids|timestamp",
                    rvdir="newer",
                    format="json",
                    **cont,
                )
                j = net.http_get(WIKI_API, "wiki", params=params).json()
                for pg in (j.get("query") or {}).get("pages", {}).values():
                    revs += [{"revid": r["revid"], "timestamp": r["timestamp"]} for r in pg.get("revisions", [])]
                if "continue" not in j:
                    break
                cont = {k: v for k, v in j["continue"].items() if k != "continue"}
            net.write_json("sp500", rel, revs)
        except Exception as exc:  # noqa: BLE001 - a stale list only misses the newest revisions; reported
            if hit is None:
                raise
            net.note(f"S&P 400 Wikipedia 수정 이력 갱신 실패({type(exc).__name__}) — 캐시 사용")
            revs = hit
    df = pd.DataFrame(revs, columns=["revid", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    return df.sort_values("timestamp", ignore_index=True)


def month_end_snapshots(net: PitNet, dates: Iterable[pd.Timestamp]) -> tuple[dict[pd.Timestamp, dict[str, str]], dict]:
    """{ticker: name} of the page revision in force at the end of each date (US/Eastern close ~ 21:00 UTC; revisions
    are immutable -> cached forever). Returns (snapshots, info)."""
    revs = revision_list(net)
    out: dict[pd.Timestamp, dict[str, str]] = {}
    info: Counter = Counter()
    for d in dates:
        cut = pd.Timestamp(d).normalize() + pd.Timedelta(21, "h")
        r = revs[revs["timestamp"] <= cut]
        if r.empty:
            info["before_first_revision"] += 1
            continue
        rid = int(r["revid"].iloc[-1])
        rel = f"{RAW_DIR}/rev/{rid}.json.gz"
        try:
            raw = net.cached_bytes(
                "sp500",
                rel,
                WIKI_API,
                "wiki",
                None,
                params=dict(action="parse", oldid=rid, prop="text", format="json", formatversion=2),
            )
            tick = parse_revision(json.loads(raw)["parse"]["text"])
        except Exception as exc:  # noqa: BLE001 - a missing revision only weakens the validation; counted
            info["failed"] += 1
            net.note(f"S&P 400 Wikipedia 수정본 {rid} 실패({type(exc).__name__}) — 해당 월 검증 생략")
            continue
        if tick is None:
            info["no_table"] += 1
            continue
        out[pd.Timestamp(d).normalize()] = tick
        info["ok"] += 1
    return out, dict(info)


# ============================================================================ build


def _ts(d: date | pd.Timestamp | str | None) -> pd.Timestamp | None:
    return None if d is None else pd.Timestamp(d).normalize()


def build_key(settings: Settings, start: pd.Timestamp, end: pd.Timestamp | None, today: pd.Timestamp) -> str:
    """Cache key of a mid-cap build: every input that shapes it + both code versions (the caller's ``exclude`` is
    applied after the cache, so it is not part of the key)."""
    blob = json.dumps(
        [
            "midcap",
            MIDCAP_VERSION,
            pit_us.BUILD_VERSION,
            str(start.date()),
            str(end.date()) if end is not None else None,
            str(today.date()),
            settings.backtest.us_start,
            settings.pit.model_dump(),
            [PIT_MIN_AGREEMENT, PIT_WINDOW, PIT_FLOOR, FRESH_MIN, FRESH_WINDOW, VALIDATION_MAX_LAG, TRIM_MIN_ABSENT],
            [REVISION_COMPLETE, pit_us.MIN_SHARE_COVERAGE],
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def midcap_build(
    settings: Settings,
    cache: DiskCache,
    net: PitNet,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    *,
    today: pd.Timestamp | None = None,
    spy_loader: Any = None,
    sp500_loader: Callable[[], PITUniverse] | None = None,
    include_sp500: bool = False,
) -> pit_us.FullBuild:
    """Full mid-cap build (every retained member per date, before the caller's exclusion), cached on disk.
    ``include_sp500``: skip the S&P 500 exclusion (``load_us_midcap_universe``)."""
    today = today or pit_us._today()
    key = build_key(settings, start, end, today)
    use_cache = getattr(cache, "enabled", True)
    name = f"midcap_{'all_' if include_sp500 else ''}{key}.pkl"
    if use_cache:
        hit = net.read_pickle("build", name)
        if isinstance(hit, pit_us.FullBuild):
            return hit
    if sp500_loader is None:

        def sp500_loader() -> PITUniverse:
            return pit_us.load_us_pit_universe(settings, cache, 500, pd.Timestamp(settings.backtest.us_start).date())

    fb = _MidcapBuilder(settings, cache, net, start, end, today, spy_loader, sp500_loader, include_sp500).run()
    if use_cache:
        net.write_pickle("build", name, fb)
        cutoff = time.time() - BUILD_MAX_AGE_DAYS * 86400
        for p in net.path("build").glob("midcap_*.pkl"):  # both variants age out the same way
            if p.name != name and p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
    return fb


def sp500_exclusion(uni: PITUniverse) -> dict[str, Any]:
    """Entities, CIKs (incl. predecessor chains) and tickers of the PIT S&P 500 tuning universe, plus each entity's
    first/last member date (to count names promoted later vs demoted earlier)."""
    mem = uni.membership
    ents = set(mem["entity"].astype(str))
    ciks = {us_cik.entity_cik(e) for e in ents}
    for succ, preds in pit_us.MANUAL_PREDECESSORS.items():
        if succ in ciks:
            ciks.update(preds)
    chains = ((uni.coverage or {}).get("fundamentals") or {}).get("cik_chains") or {}
    for e, ch in chains.items():
        if e in ents:
            ciks.update(int(c) for c in ch)
    tickers = {normalise_ticker(str(t)) for t in mem["ticker"].dropna() if str(t).strip()}
    for s in (uni.coverage or {}).get("unmapped_spells") or []:
        if str(s).split():
            tickers.add(normalise_ticker(str(s).split()[0]))
    meta = uni.meta if uni.meta is not None else pd.DataFrame()
    if "yahoo_symbol" in meta:
        tickers |= {normalise_ticker(str(t)) for t in meta["yahoo_symbol"].dropna()}
    span = mem.groupby("entity")["date"].agg(["min", "max"])
    return {
        "entities": ents,
        "ciks": ciks,
        "tickers": tickers,
        "first": span["min"].to_dict(),
        "last": span["max"].to_dict(),
    }


class _MidcapBuilder(pit_us._Builder):
    """The S&P 500 builder with S&P 400 membership, the S&P 500 exclusion and no RSP tracking check."""

    def __init__(self, settings, cache, net, start, end, today, spy_loader, sp500_loader, include_sp500=False):
        super().__init__(settings, cache, net, start, end, today, spy_loader)
        self.sp500_loader = sp500_loader
        self.include_sp500 = bool(include_sp500)

    # ------------------------------------------------------------------ membership

    def _membership(self) -> None:
        net = self.net
        cur, ch = load_wiki400(net)
        if not (MAX_CURRENT_ROWS[0] <= len(cur) <= MAX_CURRENT_ROWS[1]):
            raise MembershipError(f"Wikipedia S&P 400 구성종목 {len(cur)}개 — 형식 변경 가능성, 빌드 중단")
        future = ch[ch["date"] > self.today]
        if len(future):
            self.notes.append(f"S&P 400 예정 변경 {len(future)}건(발효일 {future['date'].min().date()}~)은 아직 미반영")
        sec_cur: dict[str, int] = {}
        for t, c, _ in us_cik.current_company_tickers(net):
            sec_cur.setdefault(dash(t), c)
        pairs = us_cik.sec_ticker_history(net, self.today)
        lookup = us_cik.cik_lookup_index(net)
        links = [x for x in pd.concat([ch["rem_link"], ch["add_link"]]).dropna().unique()]
        wd = us_cik.wikidata_ciks(net, links)
        resolve = self._resolver(pairs, lookup, wd, sec_cur)
        spells, anomalies = reconstruct_spells(cur, ch, self.today, resolve)
        # month-end page revisions: bridge the dates the change table lacks, then validate every month-end
        me_all = month_end_sessions(nyse_sessions(min(self.start, pd.Timestamp("2012-01-01")), self.today))
        snaps, snap_info = month_end_snapshots(net, [d for d in me_all if d <= self.today])
        keys = RevisionKeys(spells, pairs, sec_cur)
        rev_keys = usable_revisions(revision_keys(snaps, keys))
        fresh = revision_freshness(dated_events(spells), rev_keys)
        fresh_keys = {d: k for d, k in rev_keys.items() if fresh[d] >= FRESH_MIN}  # trimming evidence: proven fresh
        snap_info = snap_info | {"complete": len(rev_keys), "fresh": len(fresh_keys)}
        raw_agree = snapshot_agreement(spells[~spells["end_unknown"]], rev_keys)
        spells, bridged = bridge_unknowns(spells, rev_keys)
        spells, trimmed = trim_open_starts(spells, fresh_keys)
        agree = snapshot_agreement(spells, rev_keys)
        agree["freshness"] = agree["date"].map(fresh)
        agree["fresh"] = ~(agree["freshness"] < FRESH_MIN)
        pit_from = pit_from_date(agree)
        # issuers the change table never mentions but the revisions list (after the validation: not self-confirming)
        supp, supp_stats = revision_supplements(spells, {d: snaps[d] for d in rev_keys}, keys)
        if len(supp):
            spells = pd.concat([spells, supp], ignore_index=True)
        final = snapshot_agreement(spells, rev_keys)
        final["fresh"] = agree["fresh"].to_numpy()
        counts = pd.Series([len(members_at(spells, d)) for d in self.dates], index=self.dates, dtype="float64")
        if len(counts) and ((counts < COUNT_HARD_RANGE[0]) | (counts > COUNT_HARD_RANGE[1])).any():
            bad = counts[(counts < COUNT_HARD_RANGE[0]) | (counts > COUNT_HARD_RANGE[1])]
            raise MembershipError(f"S&P 400 월말 구성종목 수 비정상 {bad.head(5).to_dict()} — 빌드 중단")
        off = counts[(counts < COUNT_RANGE[0]) | (counts > COUNT_RANGE[1])]
        self.cov["membership_checks"] = self._membership_checks(
            cur,
            ch,
            spells,
            anomalies,
            bridged | {f"open_start_{k}": v for k, v in trimmed.items()},
            snap_info,
            agree,
            pit_from,
            counts,
            off,
        )
        chk = self.cov["membership_checks"]
        chk["agreement_raw_median_by_year"] = _by_year(raw_agree, "agreement")
        chk["revision_supplements"] = supp_stats
        chk["agreement_with_supplements_median_by_year"] = _by_year(final[final["fresh"]], "agreement_lag")
        self.pit_from = pit_from
        self.agreement = agree
        # the S&P 500 exclusion (before any price / XBRL work) - skipped for the all-members variant
        if self.include_sp500:
            self.cov["sp500_exclusion"] = "skipped (include_sp500: every PIT S&P 400 member)"
        else:
            spells = self._exclude_sp500(spells)
        self.win, self.events = self._spells_to_window(spells)
        self.current = cur
        unm = self.win[self.win["cik"].isna()]
        self.cov["cik_methods"] = dict(Counter(self.win["cik_method"]))
        self.cov["unmapped_spells"] = [
            f"{r['ticker']} {r['start'].date()}~{r['end'].date() if pd.notna(r['end']) else '현재'}"
            for _, r in unm.iterrows()
        ]

    def _resolver(self, pairs, lookup, wd, sec_cur):
        subs = self.subs
        today = self.today
        memo: dict[tuple, tuple[int | None, str]] = {}

        def resolve(ticker: str, name: str | None, link: str | None, d: pd.Timestamp, side: str):
            k = (ticker, name, link, d, side)
            if k in memo:
                return memo[k]
            if side == "current":
                c = sec_cur.get(dash(ticker))
                out = (c, "sec_current") if c else (None, "unmapped")
            else:
                ev = pd.Series(dict(date=d, ticker=dash(ticker), fja_ticker=ticker, name=name, link=link))
                c, m = us_cik.cik_for_event(ev, pairs, lookup, wd, subs)
                if c is None and side == "add":  # a spin-off / IPO added on listing has no periodic filing yet
                    ev["date"] = min(d + pd.Timedelta(365, "D"), today)
                    c, m = us_cik.cik_for_event(ev, pairs, lookup, wd, subs)
                    m = f"{m}_after_add" if c is not None else m
                if c is None and dash(ticker) in sec_cur and (today - d).days <= 400:
                    c, m = sec_cur[dash(ticker)], "sec_current_recent"
                out = (c, m)
            memo[k] = out
            return out

        return resolve

    def _membership_checks(self, cur, ch, spells, anomalies, bridged, snap_info, agree, pit_from, counts, off):
        kinds = Counter(a["kind"] for a in anomalies)
        by_year = _by_year(agree[agree["fresh"]], "agreement_lag") if "fresh" in agree else {}
        dated = ch[ch["date"] <= self.today]
        info = {
            "source": "Wikipedia 'List of S&P 400 companies' (현재 표 + 변경 표) 역산 + 월말 수정본 검증",
            "current_count": len(cur),
            "current_cik_from_wikipedia": int(cur["cik"].notna().sum()),
            "change_rows": len(dated),
            "change_rows_by_year": {int(y): int(n) for y, n in dated.groupby(dated["date"].dt.year).size().items()},
            "first_change": str(dated["date"].min().date()) if len(dated) else None,
            "spells": len(spells),
            "anomalies": dict(kinds),
            "anomaly_examples": [
                f"{a['kind']} {pd.Timestamp(a['date']).date()} {a['ticker']}->{a['now']}" for a in anomalies[:40]
            ],
            "bridged_by_revisions": bridged,
            "revisions": snap_info,
            "agreement_median_by_year": by_year,
            "fresh_revisions_by_year": (
                {int(y): int(g["fresh"].sum()) for y, g in agree.groupby(agree["date"].dt.year)} if len(agree) else {}
            ),
            "agreement_min_since_pit_from": (
                round(float(agree.loc[(agree["date"] >= pit_from) & agree["fresh"], "agreement_lag"].min()), 4)
                if pit_from is not None
                else None
            ),
            "pit_from": str(pit_from.date()) if pit_from is not None else None,
            "pit_rule": (
                f"발행사(CIK) 기준 Jaccard(역산, 해당·이후 {VALIDATION_MAX_LAG}개월 수정본 중 최고) "
                f">= {PIT_MIN_AGREEMENT}, 연속 {PIT_WINDOW}개 검증 월말 중 미달 최대 1개, 최저 >= {PIT_FLOOR} "
                "(최신 상태의 수정본만 검증에 사용)"
            ),
            "month_end_count_min": int(counts.min()) if len(counts) else None,
            "month_end_count_max": int(counts.max()) if len(counts) else None,
            "month_end_count_out_of_range": int(len(off)),
        }
        if len(off):
            self.notes.append(
                f"S&P 400 역산 월말 구성종목 수 {len(off)}개월이 {COUNT_RANGE[0]}-{COUNT_RANGE[1]} 범위 밖 "
                f"(예: {', '.join(f'{d.date()}={int(v)}' for d, v in off.head(4).items())})"
            )
        return info

    def _exclude_sp500(self, spells: pd.DataFrame) -> pd.DataFrame:
        """Drop every issuer of the PIT S&P 500 tuning universe (entity / chain CIK / carried ticker) and of today's
        S&P 500 table (promoted after the tuning build's last rebalance date)."""
        uni = self.sp500_loader()
        x = sp500_exclusion(uni)
        # today's S&P 500 (the next tuning build will contain names promoted after its last rebalance date)
        cur500 = us_membership.parse_wiki_current(
            self.net.cached_bytes("sp500", "wiki_sp500.html", us_membership.WIKI_CURRENT, "wiki", 24)
        )
        cur_ciks = {int(c) for c in cur500["cik"].dropna()}
        cur_ticks = {normalise_ticker(t) for t in cur500["sym"]}
        sp = spells.copy()
        sp["entity"] = [us_cik.cik_entity(c) if c is not None and pd.notna(c) else None for c in sp["cik"]]
        in_win = sp[(sp["start"] <= self.W1) & (sp["end"].isna() | (sp["end"] > self.W0))]
        reason: dict[str, str] = {}
        for key, g in in_win.groupby(in_win["entity"].fillna("T:" + in_win["ticker"])):
            ciks = {int(c) for c in g["cik"].dropna()}
            ticks = {normalise_ticker(t) for t in pd.concat([g["ticker"], g["alias"]]).dropna()}
            now = {normalise_ticker(t) for t in g.loc[g["end"].isna(), "ticker"]}
            if key in x["entities"] or ciks & x["ciks"]:
                reason[key] = "sp500_entity"
            elif ticks & x["tickers"]:
                reason[key] = "sp500_ticker"
            elif ciks & cur_ciks or now & cur_ticks:
                reason[key] = "sp500_current"
        promoted = demoted = 0
        for key, why in reason.items():
            if why != "sp500_entity" or key not in x["first"]:
                continue
            g = in_win[in_win["entity"] == key]
            first_mid = max(g["start"].min(), self.W0)
            promoted += int(x["first"][key] > first_mid)
            demoted += int(x["first"][key] <= first_mid)
        keys = in_win["entity"].fillna("T:" + in_win["ticker"])
        keep = ~keys.isin(list(reason))
        kept = in_win[keep]
        ex_by = Counter(reason.values())
        tick_ex = sorted(
            {t for k, why in reason.items() if why == "sp500_ticker" for t in in_win.loc[keys == k, "ticker"]}
        )
        self.cov["exclusion_sp500"] = {
            "sp500_entities": len(x["entities"]),
            "sp500_tickers": len(x["tickers"]),
            "midcap_issuers_in_window": int(keys.nunique()),
            "excluded_issuers": len(reason),
            "excluded_by_reason": dict(ex_by),
            "excluded_sp500_entity_promoted_later": promoted,
            "excluded_sp500_entity_demoted_or_member_before": demoted,
            "excluded_ticker_only": tick_ex,
            "current_members_excluded": int(len({k for k in reason if (in_win.loc[keys == k, "end"].isna()).any()})),
            "retained_issuers": int(kept["entity"].fillna("T:" + kept["ticker"]).nunique()),
        }
        self.notes.append(
            f"S&P 500 제외: 창 안 S&P 400 발행사 {int(keys.nunique())}개 중 {len(reason)}개 제외"
            f"(S&P 500 엔터티 {ex_by.get('sp500_entity', 0)}, 티커 공유 {ex_by.get('sp500_ticker', 0)}, "
            f"현재 S&P 500 {ex_by.get('sp500_current', 0)}; "
            f"그중 나중에 S&P 500 편입 {promoted}개 — 미래 정보로 제외되어 향후 대형주 승자가 빠짐)"
        )
        return kept.drop(columns=["entity"])

    def _spells_to_window(self, spells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Spells -> (win, events) in the columns ``us._Builder`` expects. Spells with a pending ticker alias are split
        later (``_ftd``) once the rename date is known."""
        rows, ev_rows = [], []
        for s in spells.itertuples(index=False):
            cat = "current" if pd.isna(s.end) else classify_reason(s.reason)
            ev = -1
            if pd.notna(s.end):
                ev = len(ev_rows)
                ev_rows.append(
                    dict(
                        date=s.end,
                        start=s.start,
                        fja_ticker=s.ticker,
                        ticker=s.ticker,
                        name=s.name,
                        link=s.link,
                        reason=s.reason,
                        rename_to=None,
                        cat=cat,
                        cik=float(s.cik) if s.cik is not None and pd.notna(s.cik) else np.nan,
                        cik_method=s.cik_method,
                        patch="",
                    )
                )
            rows.append(
                dict(
                    ticker=s.ticker,
                    start=s.start,
                    end=s.end,
                    cik=float(s.cik) if s.cik is not None and pd.notna(s.cik) else np.nan,
                    cik_method=s.cik_method,
                    name=s.name,
                    yahoo_hint=None,
                    patch="",
                    disp=s.ticker,
                    ftd_sym=ftd_symbol(s.ticker),
                    sname=s.name,
                    ev=ev,
                    cat=cat,
                    reason=s.reason,
                    alias=s.alias,
                    alias_name=s.alias_name,
                    start_unknown=bool(s.start_unknown),
                    end_unknown=bool(s.end_unknown),
                    supplement=bool(s.supplement),
                )
            )
        win = pd.DataFrame(rows).reset_index(drop=True)
        events = pd.DataFrame(ev_rows, columns=us_membership.EVENT_COLUMNS)
        return win, events

    # ------------------------------------------------------------------ FTD + rename split

    def _ftd(self) -> None:
        win = self.win
        aliases = {ftd_symbol(a) for a in win["alias"].dropna()}
        syms = set(win["ftd_sym"]) | aliases
        lo = self.W0 - pd.Timedelta(60, "D")
        hi = min(self.today, self.W1 + pd.Timedelta(75, "D"))
        panel, missing = us_prices.ftd_panel(self.net, syms, lo, hi)
        self.ftd_missing = missing
        self.ftd_by = {k: g.reset_index(drop=True) for k, g in panel.groupby("key")} if len(panel) else {}
        self.cov["ftd"] = {
            "rows": len(panel),
            "symbols_with_rows": len(self.ftd_by),
            "symbols_needed": len(syms),
            "files_missing": [k for k, _ in missing],
            "last_settle": str(panel["settle"].max().date()) if len(panel) else None,
        }
        if missing:
            self.notes.append(f"SEC FTD 파일 {len(missing)}개 수집 실패: {[k for k, _ in missing][:6]}")
        self._split_renames()

    def _split_renames(self) -> None:
        """A spell reached through a rename (alias X, now Y) becomes [start, r) as X and [r, end) as Y: r from the last
        SEC FTD print of X's CUSIP, else the first SEC ticker-file capture of Y, else the addition date (counted)."""
        win = self.win
        pairs = us_cik.sec_ticker_history(self.net, self.today)
        new_rows, new_ev, how_c = [], [], Counter()
        for i, s in win[win["alias"].notna()].iterrows():
            x, y = str(s["alias"]), str(s["ticker"])
            end = s["end"] if pd.notna(s["end"]) else self.today + pd.Timedelta(1, "D")
            r, how = self._rename_date(x, y, int(s["cik"]) if pd.notna(s["cik"]) else None, s["start"], end, pairs)
            how_c[how] = how_c.get(how, 0) + 1
            if r is None or r <= s["start"]:
                continue
            old = s.to_dict()
            ev = len(self.events) + len(new_ev)
            new_ev.append(
                dict(
                    date=r,
                    start=s["start"],
                    fja_ticker=x,
                    ticker=x,
                    name=s["alias_name"] or s["name"],
                    link=None,
                    reason=f"ticker change {x} -> {y}",
                    rename_to=y,
                    cat="renamed",
                    cik=s["cik"],
                    cik_method=s["cik_method"],
                    patch="",
                )
            )
            old.update(
                ticker=x,
                end=r,
                disp=x,
                ftd_sym=ftd_symbol(x),
                name=s["alias_name"] or s["name"],
                sname=s["alias_name"] or s["name"],
                ev=ev,
                cat="renamed",
                reason=f"ticker change {x} -> {y}",
                alias=None,
                end_unknown=False,
            )
            new_rows.append(old)
            win.loc[i, "start"] = r
            win.loc[i, "start_unknown"] = False
        if new_rows:
            win = pd.concat([win, pd.DataFrame(new_rows)], ignore_index=True)
            self.events = pd.concat([self.events, pd.DataFrame(new_ev, columns=self.events.columns)], ignore_index=True)
        self.win = win.sort_values(["ticker", "start"], ignore_index=True)
        self.cov["membership_checks"]["rename_dates"] = dict(how_c)

    def _rename_date(self, x, y, cik, start, end, pairs) -> tuple[pd.Timestamp | None, str]:
        f = self.ftd_by.get(ftd_symbol(x))
        if f is not None and len(f):
            own = f[(f["symbol"] == f["key"]) & (f["px_date"] >= start - pd.Timedelta(30, "D")) & (f["px_date"] < end)]
            if len(own):
                near = own[own["px_date"] <= start + pd.Timedelta(92, "D")]
                cus = (near if len(near) else own)["cusip"].mode().iloc[0]
                last = own.loc[own["cusip"] == cus, "px_date"].max()
                if start < last < end - pd.Timedelta(20, "D"):
                    return pd.Timestamp(next_sessions(pd.DatetimeIndex([last]))[0]), "ftd_last_print"
        if cik is not None and len(pairs):
            p = pairs[(pairs["cik"] == cik) & (pairs["ticker"] == y)]
            if len(p):
                fs = p["first_seen"].min()
                if start < fs < end and fs > us_cik.WAYBACK_FIRST:
                    return pd.Timestamp(fs), "sec_ticker_file"
        return None, "unknown"

    # ------------------------------------------------------------------ no tracking check, no S&P 500 side effects

    def _rsp(self) -> pd.Series | None:
        return None

    def _tracking(self, rsp: pd.Series | None) -> None:
        self.cov["tracking_check"] = {
            "ok": None,
            "reason": "홀드아웃 위생: 보유 종목 수익률 집계 금지 + S&P 500 제외로 지수와 다른 구성 — 추적 점검 생략",
        }

    def _persist_known(self) -> None:
        """The S&P 500 build's known_entities.json / entity_splits.pkl must not be overwritten by the mid-cap set."""

    def run(self) -> pit_us.FullBuild:
        self.pit_from = None
        fb = super().run()
        ms = fb.month_src.copy()
        ms["ret"] = np.nan  # holdout hygiene: no member-month returns are kept (only the price source)
        fb.month_src = ms
        pit_from = self.pit_from
        biased = [d for d in fb.dates if pit_from is None or d < pit_from]
        fb.coverage.update(
            index="S&P MidCap 400 (전체 구성원)" if self.include_sp500 else "S&P MidCap 400 (S&P 500 이력 제외)",
            membership_method="wikipedia_changes_backward_validated_by_revisions",
            pit_from=str(pit_from.date()) if pit_from is not None else None,
            survivorship_biased=bool(biased),
            biased_dates=len(biased),
            biased_range=[str(biased[0].date()), str(biased[-1].date())] if biased else None,
        )
        fb.diagnostics["agreement"] = self.agreement
        return fb


# ============================================================================ universe


def _exclude_keys(exclude: Iterable[str]) -> tuple[set[str], set[str]]:
    ents, ticks = set(), set()
    for x in exclude or ():
        s = str(x).strip()
        if not s:
            continue
        if re.fullmatch(r"CIK\d{1,10}", s.upper()):
            ents.add(us_cik.cik_entity(us_cik.entity_cik(s)))
        else:
            ticks.add(normalise_ticker(s.split("#", 1)[0]))
    return ents, ticks


def to_universe(full: pit_us.FullBuild, exclude: Iterable[str] = ()) -> PITUniverse:
    """PITUniverse of every retained member per date minus the caller's ``exclude`` (tickers or entity ids)."""
    mem = full.members
    ex_ents, ex_ticks = _exclude_keys(exclude)
    meta = full.meta
    carried = mem.groupby("entity")["ticker"].agg(lambda s: {normalise_ticker(str(t)) for t in s.dropna()}).to_dict()
    ysym = dict(zip(meta["entity"], meta["yahoo_symbol"], strict=False)) if len(meta) else {}
    hit: dict[str, str] = {}
    for e in mem["entity"].unique():
        if e in ex_ents:
            hit[e] = "entity"
            continue
        names = set(carried.get(e, set()))
        if isinstance(ysym.get(e), str):
            names.add(normalise_ticker(ysym[e]))
        if names & ex_ticks:
            hit[e] = "ticker"
    sel = mem[~mem["entity"].isin(hit)].copy()
    sel = sel.sort_values(["date", "rank_value", "entity"], ascending=[True, False, True], na_position="last")
    sel["rank"] = sel.groupby("date").cumcount() + 1
    membership = sel[MEMBERSHIP_COLUMNS].sort_values(["date", "rank"]).reset_index(drop=True)
    ents = set(membership["entity"])
    prices = {e: full.frames[e] for e in sorted(ents) if e in full.frames}
    ms = full.month_src.merge(membership[["date", "entity"]], on=["date", "entity"], how="inner")
    cov = dict(full.coverage)
    cov.update(pit_us._month_coverage(ms))
    cov["n"] = INDEX_SIZE
    cov["no_price_entities"] = sorted(ents - set(prices))
    cov["entities"] = len(ents)
    cov["entities_with_prices"] = len(prices)
    per_date = membership.groupby("date").size()
    cov["members_per_date"] = {
        "min": int(per_date.min()) if len(per_date) else 0,
        "median": float(per_date.median()) if len(per_date) else 0.0,
        "max": int(per_date.max()) if len(per_date) else 0,
    }
    cov["excluded"] = {
        "caller_names": len(ex_ents) + len(ex_ticks),
        "caller_entities_dropped": len(hit),
        "by": dict(Counter(hit.values())),
        "entities": sorted(hit)[:50],
        "sp500": cov.get("exclusion_sp500"),
    }
    out_meta = meta[meta["entity"].isin(ents)].reset_index(drop=True) if len(meta) else meta
    notes = _summary_notes(membership, cov) + list(full.notes)
    return PITUniverse(
        market=Market.US,
        n=INDEX_SIZE,
        membership=membership,
        prices=prices,
        benchmarks=dict(full.benchmarks),
        entity_benchmark={e: pit_us.BENCHMARK for e in prices},
        meta=out_meta,
        coverage=cov,
        notes=notes,
    )


def _summary_notes(membership: pd.DataFrame, cov: dict) -> list[str]:
    dates = membership["date"]
    tot = max(cov["member_months"], 1)
    y, f, m = cov["member_months_with_prices"], cov["member_months_ftd_only"], cov["member_months_missing"]
    pit_from = cov.get("pit_from")
    head = (
        "미국 S&P MidCap 400 시점별 전체 구성원(S&P 500 이력 종목 포함)"
        if str(cov.get("sp500_exclusion", "")).startswith("skipped")
        else "미국 XSH 유니버스: S&P MidCap 400 시점별 구성 중 S&P 500 편입 이력(백테스트 시작 이후) 없는 종목"
    )
    lines = [
        f"{head}, CIK 단위 "
        f"{dates.min().date() if len(dates) else '-'}~{dates.max().date() if len(dates) else '-'} 월말 "
        f"{dates.nunique()}회, 날짜당 {cov['members_per_date']['min']}~{cov['members_per_date']['max']}개",
        f"회원-월 {cov['member_months']}개: 일별 가격 {y}({pit_us._pct(y / tot)}), FTD 월말가만 {f}"
        f"({pit_us._pct(f / tot)}), 결측 {m}({pit_us._pct(m / tot)})",
    ]
    if cov.get("survivorship_biased"):
        lines.append(
            f"경고 — 생존편향(SURVIVORSHIP-BIASED): {cov.get('biased_range')} 월말 {cov.get('biased_dates')}회는 "
            f"Wikipedia 변경 표가 불완전해 시점 구성이 검증되지 않음(오늘 생존 종목이 너무 일찍 등장, 제외 종목 누락). "
            f"시점(PIT) 검증 구간은 {pit_from or '없음'}부터 — 평가는 이 날짜 이후로 제한하거나 편향을 명시할 것"
        )
    else:
        lines.append(f"시점 구성 검증: 전 구간이 Wikipedia 월말 수정본과 일치(PIT 시작 {pit_from})")
    ex = cov.get("exclusion_sp500") or {}
    if ex:
        lines.append(
            f"S&P 500 튜닝 종목 제외 {ex.get('excluded_issuers')}개(나중에 S&P 500 편입 "
            f"{ex.get('excluded_sp500_entity_promoted_later')}개 포함: 미래 정보 기반 제외 — 향후 승자 과소대표)"
        )
    if cov["excluded"]["caller_entities_dropped"]:
        lines.append(f"호출자 제외 목록으로 {cov['excluded']['caller_entities_dropped']}개 종목 추가 제외")
    if cov.get("rejected_series") or cov.get("masked_rows"):
        lines.append(
            f"SEC FTD 가격 대조로 다른 발행사 Yahoo 시계열 {cov.get('rejected_series', 0)}건 거부, "
            f"부분 불일치 {len(cov.get('masked_entities', []))}개 종목 {cov.get('masked_rows', 0)}행 제거"
        )
    if cov.get("no_price_entities"):
        lines.append(f"일별 가격 없는 종목 {len(cov['no_price_entities'])}개 — RSMT-Q/평가에서 제외")
    if cov.get("unmapped_spells"):
        lines.append(f"CIK 미확정 구간 {len(cov['unmapped_spells'])}개 제외: {cov['unmapped_spells'][:4]}")
    lines.append(
        "잔여 편향: S&P 위원회 편입 기준(순수 시총 규칙 아님), Wikipedia 변경 표 누락분은 월말 수정본으로 보정, "
        "FTD 월말가는 배당 미포함·최대 7일 지연"
    )
    return lines
