"""Keyless point-in-time S&P 500 membership (verified 2026-09-25).

Sources
- fja05680/sp500 (GitHub raw): 'S&P 500 Historical Components & Changes (Updated).csv'
  rows = (date, comma-joined tickers) snapshots on every change date since 1996-01-02.
  Tickers are those in use ON that date (FB before 2022-06-09, META after). '.' for classes (BRK.B).
- Wikipedia 'List of S&P 500 companies' (current table only since 2026-08-11) and
  'Historical components of the S&P 500' (the old 'Selected changes' table was MOVED there
  on 2026-08-11, revid 1368903137). One row per (added, removed) pair; no symbol renames.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)
WIKI_UA = "QuantCIO Research research@example.com"
FJA_RAW = "https://raw.githubusercontent.com/fja05680/sp500/master/"
FJA_API = "https://api.github.com/repos/fja05680/sp500/contents/"


def _get_cached(url: str, name: str, max_age_h: float = 24.0, headers: dict | None = None) -> bytes:
    p = CACHE / name
    if p.exists() and (time.time() - p.stat().st_mtime) < max_age_h * 3600:
        return p.read_bytes()
    r = requests.get(url, headers=headers or {"User-Agent": WIKI_UA}, timeout=60)
    r.raise_for_status()
    p.write_bytes(r.content)
    return r.content


# ----------------------------------------------------------------------------- fja05680
def fja_history_filename() -> str:
    """Discover the current 'Updated' CSV name (it used to carry a date suffix,
    e.g. '...(10-14-2023).csv'; on 2026-09-25 it is '...(Updated).csv')."""
    items = json.loads(_get_cached(FJA_API, "gh_contents.json", 6))
    names = [e["name"] for e in items if e["name"].startswith("S&P 500 Historical Components") and e["name"].endswith(".csv")]
    upd = [n for n in names if n != "S&P 500 Historical Components & Changes.csv"]
    return sorted(upd)[-1] if upd else names[0]


def fja_snapshots() -> pd.DataFrame:
    """DataFrame index=date (Timestamp), column 'tickers' = frozenset[str] (fja spelling, '.' classes)."""
    fn = fja_history_filename()
    raw = _get_cached(FJA_RAW + quote(fn), "fja_hist.csv", 24)
    df = pd.read_csv(io.BytesIO(raw))
    df["date"] = pd.to_datetime(df["date"])
    df["tickers"] = df["tickers"].map(lambda s: frozenset(t.strip() for t in s.split(",") if t.strip()))
    return df.set_index("date").sort_index()


def fja_members_at(snaps: pd.DataFrame, when) -> frozenset[str]:
    """Members in force at the close of `when` (last snapshot dated <= when)."""
    when = pd.Timestamp(when)
    i = snaps.index.searchsorted(when, side="right") - 1
    if i < 0:
        raise ValueError(f"{when.date()} before first snapshot {snaps.index[0].date()}")
    return snaps["tickers"].iloc[i]


def fja_intervals(snaps: pd.DataFrame) -> pd.DataFrame:
    """Membership spells: ticker, start (first snapshot date in index), end (first snapshot
    date NOT in index = removal effective date; NaT if still in). One row per spell."""
    rows, open_ = [], {}
    prev: frozenset[str] = frozenset()
    for d, cur in snaps["tickers"].items():
        for t in cur - prev:
            open_[t] = d
        for t in prev - cur:
            rows.append((t, open_.pop(t), d))
        prev = cur
    rows += [(t, s, pd.NaT) for t, s in open_.items()]
    return pd.DataFrame(rows, columns=["ticker", "start", "end"]).sort_values(["ticker", "start"], ignore_index=True)


# ----------------------------------------------------------------------------- Wikipedia
def wiki_current() -> pd.DataFrame:
    h = _get_cached("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "wiki_sp500.html", 24)
    t = pd.read_html(io.StringIO(h.decode()))[0]
    t.columns = ["symbol", "security", "sector", "sub_industry", "hq", "date_added", "cik", "founded"]
    t["cik"] = t["cik"].astype(int)
    return t


def wiki_changes() -> pd.DataFrame:
    """date, add_t, add_n, rem_t, rem_n, reason (NaN where one side is empty)."""
    h = _get_cached("https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500", "wiki_hist.html", 24)
    t = pd.read_html(io.StringIO(h.decode()))[0]
    t.columns = ["date", "add_t", "add_n", "rem_t", "rem_n", "reason", "refs"]
    t["date"] = pd.to_datetime(t["date"], format="%B %d, %Y")
    return t.drop(columns="refs").sort_values("date", ascending=False, ignore_index=True)


def wiki_members_at(current: set[str], changes: pd.DataFrame, when) -> set[str]:
    """Backward roll from today's list: undo every change effective AFTER `when`.
    Tickers stay in CURRENT spelling for renamed companies (e.g. META for 2020)."""
    when = pd.Timestamp(when)
    mem = set(current)
    for _, r in changes[changes["date"] > when].iterrows():  # newest first
        if isinstance(r["add_t"], str):
            mem.discard(r["add_t"])
        if isinstance(r["rem_t"], str):
            mem.add(r["rem_t"])
    return mem
