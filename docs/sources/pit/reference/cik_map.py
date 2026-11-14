"""Historical ticker -> SEC CIK mapping for removed S&P 500 members (keyless). Verified 2026-09-25.

Methods (in priority order):
  M1 wayback : union of monthly Wayback snapshots of sec.gov/files/company_tickers.json (2017-08..now),
               date-aware (the (ticker,cik) pair seen closest to the removal date).
  M2 wikidata: Wikipedia article (from the changes table) -> Wikidata QID -> P5531 (CIK).
  M3 name    : exact normalised-name match in sec.gov/Archives/edgar/cik-lookup-data.txt (all names ever
               used incl. former names), disambiguated by 10-K/10-Q activity around the removal date.
Validation : data.sec.gov/submissions/CIK##########.json -> the CIK filed a 10-K/10-Q/20-F/40-F in the
             540 days before removal (active issuer then) + name/formerNames token overlap.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parent / "cache"
SEC_UA = {"User-Agent": "QuantCIO Research research@example.com"}
_STOP = {"inc", "corp", "corporation", "co", "company", "the", "group", "holdings", "holding", "plc", "ltd",
         "limited", "n", "v", "nv", "de", "new", "sa", "lp", "llc", "incorporated", "international", "intl", "cos",
         "companies", "trust", "and", "of", "&"}


def norm_name(s: str) -> str:
    s = re.sub(r"/[a-z]{2}/?$", "", str(s).lower().strip())          # 'CELGENE CORP /DE/' -> 'celgene corp'
    s = re.sub(r"[^a-z0-9 ]", " ", s.replace("&", " and "))
    return " ".join(w for w in s.split() if w not in _STOP)


def name_tokens(s: str) -> set[str]:
    return set(norm_name(s).split())


def names_match(a: str, b: str) -> bool:
    """Loose issuer-name match: shared token, or one squashed name contains / prefixes the other
    ('TSYS' ~ 'Total System Services' via initials, 'Kohl's' ~ 'KOHLS', 'VF' ~ 'V F', 'QuintilesIMS' ~ 'Quintiles IMS')."""
    ta, tb = name_tokens(a), name_tokens(b)
    if ta & tb:
        return True
    sa, sb = "".join(norm_name(a).split()), "".join(norm_name(b).split())
    if not sa or not sb:
        return False
    if sa in sb or sb in sa or sa[:6] == sb[:6]:
        return True
    ini = lambda s: "".join(w[0] for w in norm_name(s).split())  # noqa: E731
    return sa == ini(b) or sb == ini(a) or sa.startswith(ini(b)) and len(ini(b)) >= 3


# ---------------------------------------------------------------- M1 Wayback company_tickers.json history
def wayback_pairs() -> pd.DataFrame:
    return pd.read_pickle(CACHE / "wayback_ticker_cik.pkl")  # ticker, cik, title, first_seen, last_seen, n


def m1_wayback(pairs: pd.DataFrame, ticker: str, when: pd.Timestamp) -> tuple[int | None, str]:
    t = ticker.upper().replace(".", "-")
    c = pairs[pairs.ticker == t]
    if c.empty:
        return None, "no_ticker"
    # distance (days) from removal date to the pair's [first_seen, last_seen] window
    dist = c.apply(lambda r: 0 if r.first_seen <= when <= r.last_seen else
                   min(abs((r.first_seen - when).days), abs((r.last_seen - when).days)), axis=1)
    best = c.loc[dist.idxmin()]
    if dist.min() > 120 and not (when < pd.Timestamp("2017-08-28") and best.first_seen == pd.Timestamp("2017-08-28")):
        return None, f"nearest_pair_{int(dist.min())}d_away"
    return int(best.cik), "in_window" if dist.min() == 0 else f"{int(dist.min())}d"


# ---------------------------------------------------------------- M3 cik-lookup-data name match
def load_cik_lookup() -> dict[str, set[int]]:
    idx: dict[str, set[int]] = {}
    with open(CACHE / "cik-lookup-data.txt", encoding="latin-1") as f:
        for line in f:
            m = re.match(r"^(.*):(\d{10}):\s*$", line)
            if m and m.group(1):
                idx.setdefault(norm_name(m.group(1)), set()).add(int(m.group(2)))
    return idx


# ---------------------------------------------------------------- SEC submissions (validation)
_sess = requests.Session()
_sess.headers.update(SEC_UA)
_last = [0.0]


def submissions(cik: int) -> dict | None:
    p = CACHE / "subs" / f"{cik}.json"
    p.parent.mkdir(exist_ok=True)
    if p.exists():
        return json.loads(p.read_text()) or None
    w = _last[0] + 0.15 - time.monotonic()      # <= ~7 req/s
    if w > 0:
        time.sleep(w)
    _last[0] = time.monotonic()
    r = _sess.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", timeout=30)
    j = r.json() if r.status_code == 200 else {}
    p.write_text(json.dumps(j))
    return j or None


def periodic_filing_dates(sub: dict, need_before: pd.Timestamp | None = None) -> list[pd.Timestamp]:
    """10-K/10-Q/20-F/40-F filing dates from 'recent'; older pages (filings.files[]) are fetched only
    while the oldest date seen is still after `need_before`."""
    out = []
    blocks = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files", []):
        oldest = min((pd.Timestamp(d) for b in blocks for d in b.get("filingDate", [])), default=None)
        if need_before is not None and oldest is not None and oldest <= need_before:
            break
        p = CACHE / "subs" / f["name"]
        if not p.exists():
            w = _last[0] + 0.15 - time.monotonic()
            if w > 0:
                time.sleep(w)
            _last[0] = time.monotonic()
            r = _sess.get(f"https://data.sec.gov/submissions/{f['name']}", timeout=30)
            p.write_text(r.text if r.status_code == 200 else "{}")
        blocks.append(json.loads(p.read_text()) or {"form": [], "filingDate": []})
    for b in blocks:
        for form, d in zip(b.get("form", []), b.get("filingDate", [])):
            if form in ("10-K", "10-Q", "20-F", "40-F", "10-K405", "10-KT"):
                out.append(pd.Timestamp(d))
    return out


def validate(cik: int, name: str | None, when: pd.Timestamp) -> dict:
    sub = submissions(cik)
    if not sub:
        return dict(valid=False, why="no_submissions")
    dates = periodic_filing_dates(sub, when - pd.Timedelta(540, unit="D"))
    active = any(when - pd.Timedelta(540, unit="D") <= d <= when + pd.Timedelta(10, unit="D") for d in dates)
    names = [sub.get("name", "")] + [f.get("name", "") for f in sub.get("formerNames", [])]
    overlap = any(names_match(name, n) for n in names) if name else None
    return dict(valid=active and (overlap is None or overlap), active=active, overlap=overlap,
                sec_name=sub.get("name"), sec_tickers=",".join(sub.get("tickers", [])),
                last_periodic=max(dates).date() if dates else None)


def m1_wayback_validated(pairs: pd.DataFrame, ticker: str, when: pd.Timestamp, name: str | None = None,
                         max_days: int = 400) -> tuple[int | None, str]:
    """Like m1_wayback but walks ALL (ticker, cik) pairs by distance to `when` and returns the first CIK
    that was an active 10-K/10-Q filer around `when` (and matches `name` if given).
    Needed because the 2017-08..2019-09 SEC ticker files carry stale pairs from long-dead issuers
    (ANET -> Authorize.Net, KEYS -> Keystone Automotive, DOW -> Dow Chemical while Dow Inc traded)."""
    t = ticker.upper().replace(".", "-")
    c = pairs[pairs.ticker == t]
    if c.empty:
        return None, "no_ticker"
    dist = c.apply(lambda r: 0 if r.first_seen <= when <= r.last_seen else
                   min(abs((r.first_seen - when).days), abs((r.last_seen - when).days)), axis=1)
    early = when < pd.Timestamp("2017-08-28")
    for i in dist.sort_values(kind="stable").index:
        if dist[i] > max_days and not (early and c.loc[i, "first_seen"] == pd.Timestamp("2017-08-28")):
            continue
        v = validate(int(c.loc[i, "cik"]), name, when)
        if v.get("valid"):
            return int(c.loc[i, "cik"]), f"validated_{int(dist[i])}d"
    return None, "no_valid_pair"
