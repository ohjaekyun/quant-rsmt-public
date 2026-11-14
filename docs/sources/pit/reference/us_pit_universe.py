"""Keyless point-in-time S&P 500 universe for QuantCIO (US, 2014-2026). Verified live 2026-09-25.

Pipeline
  1. membership   : fja05680/sp500 daily-change snapshots, patched with Wikipedia changes after fja's last row
  2. events       : every ended spell -> Wikipedia name/reason (+-10d), fja same-day-add rename detection,
                    alias of fja back-filled tickers (BTUUQ->BTU, AABA->YHOO, DXC->CSC, WYND->WYN, KDP->DPS)
  3. CIK          : M1 Wayback history of sec.gov company_tickers.json (validated by 10-K/10-Q activity via
                    data.sec.gov submissions) -> M2 Wikidata P5531 -> M3 cik-lookup-data.txt name match
  4. prices       : Yahoo v8 chart for [removal ticker, fja ticker, rename target, SEC current tickers of CIK];
                    accept only a series whose firstTradeDate precedes the spell end and that trades at the end
  5. gaps         : SEC fails-to-deliver PRICE (prior-day close, unadjusted) for members Yahoo no longer has
Everything is cached under CACHE; SEC/GitHub 5 req/s with a descriptive UA, Yahoo 4 req/s, Wikimedia with 429 backoff.
Verifier fixes (2026-09-25): FTD issuer check + partial-coverage label in resolve_yahoo, ftd_ratio/ftd_consistency/
ftd_bad_months, date-aware fja file pick, Wikipedia layout guard, negative-cache TTL, split events kept.
"""
from __future__ import annotations

import io
import json
import pickle
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

CACHE = Path(__file__).resolve().parent / "cache"
SEC_UA = "QuantCIO Research research@example.com"
WIKI_UA = "QuantCIO-PIT/0.1 (research@example.com) python-requests"
YAHOO_UA = "Mozilla/5.0 AppleWebKit/537.36"   # a full Chrome UA gets HTTP 429 from query2 (TLS fingerprint)
_GAP = {"sec": 0.2, "yahoo": 0.25, "wiki": 0.3, "archive": 1.0, "github": 0.2}   # every host <= 5 req/s
_last: dict[str, float] = {}


def _get(url: str, kind: str, headers: dict | None = None, params: dict | None = None, tries: int = 5,
         ok=(200,)) -> requests.Response:
    for i in range(tries):
        w = _last.get(kind, 0) + _GAP[kind] - time.monotonic()
        if w > 0:
            time.sleep(w)
        _last[kind] = time.monotonic()
        try:
            r = requests.get(url, headers=headers or {}, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** i)
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(float(r.headers.get("retry-after") or 0) or 3 * 2 ** i)
            continue
        if r.status_code not in ok:
            r.raise_for_status()
        return r
    raise RuntimeError(f"gave up after {tries} tries: {url}")


def _cached_bytes(name: str, url: str, kind: str, headers: dict, max_age_h: float | None = 24) -> bytes:
    p = CACHE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and (max_age_h is None or time.time() - p.stat().st_mtime < max_age_h * 3600):
        return p.read_bytes()
    b = _get(url, kind, headers).content
    p.write_bytes(b)
    return b


# ============================================================================ 1. membership
def fja_snapshots(max_age_h: float | None = 24) -> pd.DataFrame:
    """index=date, 'tickers'=frozenset. fja spells tickers with '.' (BRK.B). Pre-2019 rows carry the ticker
    in use in 2019 (ANTM in 2014 although it was WLP; BHGE for Baker Hughes Inc; AABA for Yahoo)."""
    items = json.loads(_cached_bytes("gh_contents.json", "https://api.github.com/repos/fja05680/sp500/contents/",
                                     "github", {"User-Agent": WIKI_UA}, max_age_h))
    names = [e["name"] for e in items if e["name"].startswith("S&P 500 Historical Components") and e["name"].endswith(".csv")]

    def key(n: str):   # '(Updated)' > newest '(MM-DD-YYYY)' > bare name; lexical sort mis-orders dates across years
        m = re.search(r"\((\d{2})-(\d{2})-(\d{4})\)", n)
        return (2, pd.Timestamp.min) if "Updated" in n else (1, pd.Timestamp(f"{m[3]}-{m[1]}-{m[2]}")) if m else (0, pd.Timestamp.min)
    fn = max(names, key=key)
    raw = _cached_bytes("fja_hist.csv", "https://raw.githubusercontent.com/fja05680/sp500/master/" + quote(fn),
                        "github", {"User-Agent": WIKI_UA}, max_age_h)
    df = pd.read_csv(io.BytesIO(raw))
    df["date"] = pd.to_datetime(df["date"])
    df["tickers"] = df["tickers"].map(lambda s: frozenset(t.strip() for t in s.split(",") if t.strip()))
    df = df.set_index("date").sort_index()
    if (pd.Timestamp.today() - df.index[-1]).days > 45:
        print(f"WARNING fja last row {df.index[-1].date()} is stale; patched_spells() fills the gap from Wikipedia (no renames there)")
    return df


def spells_from_snapshots(snaps: pd.DataFrame) -> pd.DataFrame:
    """ticker, start (first day in), end (first day out = removal effective date; NaT = still in)."""
    rows, open_, prev = [], {}, frozenset()
    for d, cur in snaps["tickers"].items():
        for t in cur - prev:
            open_[t] = d
        for t in prev - cur:
            rows.append((t, open_.pop(t), d))
        prev = cur
    rows += [(t, s, pd.NaT) for t, s in open_.items()]
    return pd.DataFrame(rows, columns=["ticker", "start", "end"]).sort_values(["ticker", "start"], ignore_index=True)


def wiki_current(max_age_h: float | None = 24) -> pd.DataFrame:
    h = _cached_bytes("wiki_sp500.html", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "wiki",
                      {"User-Agent": WIKI_UA}, max_age_h)
    t = pd.read_html(io.StringIO(h.decode()))[0]
    if [str(c).lower() for c in t.columns][:1] != ["symbol"] or "CIK" not in list(t.columns) or t.shape[1] != 8:
        raise ValueError(f"Wikipedia constituents table layout changed: {list(t.columns)}")   # no silent column shift
    t.columns = ["symbol", "security", "sector", "sub_industry", "hq", "date_added", "cik", "founded"]
    t["cik"] = t["cik"].astype(int)
    t["sym"] = t.symbol.str.replace(".", "-", regex=False)
    return t


def wiki_changes(max_age_h: float | None = 24) -> pd.DataFrame:
    """The 'Selected changes' table MOVED on 2026-08-11 to 'Historical components of the S&P 500'.
    One row per (added, removed) pair with article links; symbol renames are NOT listed."""
    h = _cached_bytes("wiki_hist.html", "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500",
                      "wiki", {"User-Agent": WIKI_UA}, max_age_h).decode()
    tbl = BeautifulSoup(h, "lxml").select("table.wikitable")[0]

    def cell(td):
        a = td.find("a", href=True)
        link = unquote(a["href"].split("/wiki/")[-1]) if a and "/wiki/" in a["href"] else None
        return td.get_text(" ", strip=True) or None, link

    rows = []
    for tr in tbl.select("tbody tr"):
        tds = tr.find_all("td")
        if len(tds) < 6:
            continue
        add_t, _ = cell(tds[1]); add_n, add_l = cell(tds[2]); rem_t, _ = cell(tds[3]); rem_n, rem_l = cell(tds[4])
        add_t, rem_t = [re.sub(r"[^A-Za-z0-9.\-]", "", t) or None if t else None for t in (add_t, rem_t)]
        rows.append(dict(date=pd.to_datetime(tds[0].get_text(" ", strip=True), format="%B %d, %Y"),
                         add_t=add_t, add_n=add_n, add_link=add_l, rem_t=rem_t, rem_n=rem_n, rem_link=rem_l,
                         reason=tds[5].get_text(" ", strip=True)))
    return pd.DataFrame(rows).sort_values("date", ascending=False, ignore_index=True)


def patched_spells(snaps: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """fja spells + Wikipedia rows dated after fja's last snapshot (fja lags; 2026-09-21 BE/P/ILMN for TAP/TTD/BLDR)."""
    sp = spells_from_snapshots(snaps)
    last = snaps.index[-1]
    for _, r in changes[changes.date > last].sort_values("date").iterrows():
        if isinstance(r.rem_t, str):
            hit = (sp.ticker.str.replace(".", "-", regex=False) == r.rem_t.replace(".", "-")) & sp.end.isna()
            if not hit.any():
                print(f"WARNING patched_spells: {r.date.date()} removal {r.rem_t} matches no open fja spell (renamed after fja's last row?)")
            sp.loc[hit, "end"] = r.date
        if isinstance(r.add_t, str):
            sp = pd.concat([sp, pd.DataFrame([dict(ticker=r.add_t, start=r.date, end=pd.NaT)])], ignore_index=True)
    return sp


def members_at(spells: pd.DataFrame, when) -> pd.DataFrame:
    when = pd.Timestamp(when)
    return spells[(spells.start <= when) & (spells.end.isna() | (spells.end > when))]


# ============================================================================ 2. removal events
_RENAME_RX = r"changed its ticker|ticker symbol from|changed its name|renamed|changed to|retained the"
_ACQ_RX = r"acqui|merg|purchas|privati|taken private|bought|buyout"


def removal_events(spells: pd.DataFrame, changes: pd.DataFrame, w0="2014-01-01") -> pd.DataFrame:
    """One row per ended spell in the window: date, fja_ticker, ticker (spelling traded then), name, link, reason,
    rename_to (fja same-day add when Wikipedia has no such change), cat (regex; informational)."""
    ch = changes.copy()
    for c in ("add_t", "rem_t"):
        ch[c] = ch[c].str.replace(".", "-", regex=False)
    ended = spells[spells.end.notna() & (spells.end >= pd.Timestamp(w0))].copy()
    ended["tn"] = ended.ticker.str.replace(".", "-", regex=False)
    starts = spells.assign(tn=spells.ticker.str.replace(".", "-", regex=False))
    rem = ch[ch.rem_t.notna() & (ch.date >= pd.Timestamp(w0) - pd.Timedelta(15, unit="D"))]
    used = set()
    out = []
    for _, s in ended.iterrows():
        m = rem[(rem.rem_t == s.tn) & ((rem.date - s.end).abs() <= pd.Timedelta(10, unit="D"))]
        row = dict(date=s.end, start=s.start, fja_ticker=s.ticker, ticker=s.tn, name=None, link=None, reason=None, rename_to=None)
        if len(m):
            used.add(m.index[0]); row.update(name=m.rem_n.iloc[0], link=m.rem_link.iloc[0], reason=m.reason.iloc[0])
        else:
            adds = starts[starts.start == s.end].tn.tolist()
            wadds = set(ch[(ch.date - s.end).abs() <= pd.Timedelta(10, unit="D")].add_t.dropna())
            cand = [a for a in adds if a not in wadds]
            if len(cand) == 1:
                row["rename_to"] = cand[0]
        out.append(row)
    ev = pd.DataFrame(out)
    # fja back-filled ticker <-> Wikipedia row with the ticker traded then (same date +-3d, one-to-one)
    left = rem[~rem.index.isin(used) & (rem.date >= pd.Timestamp(w0))]
    for i, r in ev[ev.name.isna() & ev.rename_to.isna()].iterrows():
        m = left[(left.date - r.date).abs() <= pd.Timedelta(3, unit="D")]
        m = m[~m.rem_t.isin(ev.ticker)]
        if len(m) == 1:
            ev.loc[i, ["ticker", "name", "link", "reason"]] = [m.rem_t.iloc[0], m.rem_n.iloc[0], m.rem_link.iloc[0], m.reason.iloc[0]]
            left = left.drop(index=m.index)

    def cat(r):
        s = str(r.reason).lower()
        if r.rename_to or re.search(_RENAME_RX, s):
            return "renamed"
        if re.search(r"bankrupt|receivership|chapter 11", s):
            return "failed"
        first = re.sub(r"[^a-z0-9 ]", "", str(r["name"]).lower()).split()[:1]
        if re.search(_ACQ_RX, s) and ((first and first[0] in s) or r.ticker.lower() in s):
            return "acquired"
        return "demoted" if r.reason else "unknown"
    ev["cat"] = ev.apply(cat, axis=1)
    return ev.sort_values("date", ignore_index=True)


# ============================================================================ 3. ticker -> CIK
_STOP = {"inc", "corp", "corporation", "co", "company", "the", "group", "holdings", "holding", "plc", "ltd", "limited",
         "n", "v", "nv", "de", "new", "sa", "lp", "llc", "incorporated", "international", "intl", "cos", "companies",
         "trust", "and", "of"}


def norm_name(s: str) -> str:
    s = re.sub(r"/[a-z]{2}/?$", "", str(s).lower().strip())
    s = re.sub(r"[^a-z0-9 ]", " ", s.replace("&", " and "))
    return " ".join(w for w in s.split() if w not in _STOP)


def names_match(a: str, b: str) -> bool:
    ta, tb = set(norm_name(a).split()), set(norm_name(b).split())
    if ta & tb:
        return True
    sa, sb = norm_name(a).replace(" ", ""), norm_name(b).replace(" ", "")
    if not sa or not sb:
        return False
    if sa in sb or sb in sa or sa[:6] == sb[:6]:
        return True
    ini = lambda s: "".join(w[0] for w in norm_name(s).split())  # noqa: E731
    return sa == ini(b) or sb == ini(a) or (len(ini(b)) >= 3 and sa.startswith(ini(b)))


def sec_ticker_history() -> pd.DataFrame:
    """All monthly Wayback captures of sec.gov/files/company_tickers.json (2017-08-28..) + today's file ->
    ticker, cik, title, first_seen, last_seen. ~95 files, ~77 MB raw, ~7-8 min at 1 req/s the first time."""
    cdx = json.loads(_cached_bytes(
        "cdx_sec.gov_files_company_tickers.json.json",
        "https://web.archive.org/cdx/search/cdx?url=sec.gov/files/company_tickers.json&output=json"
        "&filter=statuscode:200&collapse=timestamp:6&fl=timestamp,digest,length", "archive", {"User-Agent": SEC_UA}, 24 * 7))[1:]
    rows = []
    for ts, _, _ in cdx:
        b = _cached_bytes(f"wb_ct/{ts}.json", f"https://web.archive.org/web/{ts}id_/https://www.sec.gov/files/company_tickers.json",
                          "archive", {"User-Agent": SEC_UA}, None)
        for v in json.loads(b).values():
            rows.append((v["ticker"].upper(), int(v["cik_str"]), v["title"], pd.Timestamp(ts[:8])))
    cur = json.loads(_cached_bytes("company_tickers.json", "https://www.sec.gov/files/company_tickers.json", "sec", {"User-Agent": SEC_UA}, 24))
    rows += [(v["ticker"].upper(), int(v["cik_str"]), v["title"], pd.Timestamp.today().normalize()) for v in cur.values()]
    df = pd.DataFrame(rows, columns=["ticker", "cik", "title", "seen"])
    return df.groupby(["ticker", "cik"]).agg(title=("title", "last"), first_seen=("seen", "min"),
                                             last_seen=("seen", "max")).reset_index()


def sec_current_tickers_by_cik() -> dict[int, list[str]]:
    cur = json.loads(_cached_bytes("company_tickers.json", "https://www.sec.gov/files/company_tickers.json", "sec", {"User-Agent": SEC_UA}, 24))
    out: dict[int, list[str]] = {}
    for v in cur.values():
        out.setdefault(int(v["cik_str"]), []).append(v["ticker"])
    return out


def cik_lookup_index() -> dict[str, set[int]]:
    """sec.gov/Archives/edgar/cik-lookup-data.txt ('NAME:0000000000:'), ~1.06M lines, 39 MB."""
    _cached_bytes("cik-lookup-data.txt", "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt", "sec", {"User-Agent": SEC_UA}, 24 * 30)
    idx: dict[str, set[int]] = {}
    with open(CACHE / "cik-lookup-data.txt", encoding="latin-1") as f:
        for line in f:
            m = re.match(r"^(.*):(\d{10}):\s*$", line)
            if m and m.group(1):
                idx.setdefault(norm_name(m.group(1)), set()).add(int(m.group(2)))
    return idx


def submissions(cik: int) -> dict | None:
    """data.sec.gov submissions JSON (cached forever; '{}' for unknown CIKs)."""
    p = CACHE / "subs" / f"{cik}.json"
    if not p.exists():
        r = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", "sec", {"User-Agent": SEC_UA}, ok=(200, 404))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(r.text if r.status_code == 200 else "{}")
    try:
        return json.loads(p.read_text()) or None
    except json.JSONDecodeError:
        return None


def periodic_dates(sub: dict, need_before: pd.Timestamp) -> list[pd.Timestamp]:
    """10-K/10-Q/20-F/40-F filing dates; older filings.files[] pages fetched only while still needed."""
    blocks = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files", []):
        oldest = min((pd.Timestamp(d) for b in blocks for d in b.get("filingDate", [])), default=None)
        if oldest is not None and oldest <= need_before:
            break
        blocks.append(json.loads(_cached_bytes(f"subs/{f['name']}", f"https://data.sec.gov/submissions/{f['name']}", "sec",
                                               {"User-Agent": SEC_UA}, None) or b"{}") or {})
    return [pd.Timestamp(d) for b in blocks for form, d in zip(b.get("form", []), b.get("filingDate", []))
            if form in ("10-K", "10-Q", "20-F", "40-F", "10-K405", "10-KT")]


def validate_cik(cik: int, name: str | None, when: pd.Timestamp, days: int = 540) -> bool:
    """CIK filed a periodic report in [when-days, when+10d] and (if name given) one of its names matches."""
    sub = submissions(cik)
    if not sub:
        return False
    lo = when - pd.Timedelta(days, unit="D")
    if not any(lo <= d <= when + pd.Timedelta(10, unit="D") for d in periodic_dates(sub, lo)):
        return False
    if not name:
        return True
    names = [sub.get("name", "")] + [f.get("name", "") for f in sub.get("formerNames", [])]
    return any(names_match(name, n) for n in names)


def wikidata_ciks(titles: list[str]) -> dict[str, list[int]]:
    """enwiki article title -> Wikidata P5531 CIKs (article = brand/entity, can differ from the issuer then)."""
    p = CACHE / "wikidata_cache.json"
    cache = json.loads(p.read_text()) if p.exists() else {}
    todo = sorted({t for t in titles if t and t not in cache})
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        j = _get("https://en.wikipedia.org/w/api.php", "wiki", {"User-Agent": WIKI_UA},
                 dict(action="query", prop="pageprops", ppprop="wikibase_item", redirects=1, titles="|".join(batch), format="json")).json()["query"]
        norm = {n["from"]: n["to"] for n in j.get("normalized", [])}
        redir = {n["from"]: n["to"] for n in j.get("redirects", [])}
        byt = {pg["title"]: pg.get("pageprops", {}).get("wikibase_item") for pg in j["pages"].values()}
        qid = {}
        for t in batch:
            k = t.replace("_", " "); k = norm.get(k, k); k = redir.get(k, k)
            qid[t] = byt.get(k)
        qs = sorted({q for q in qid.values() if q})
        claims = {}
        if qs:
            ents = _get("https://www.wikidata.org/w/api.php", "wiki", {"User-Agent": WIKI_UA},
                        dict(action="wbgetentities", ids="|".join(qs), props="claims", format="json")).json()["entities"]
            claims = {q: sorted({int(c["mainsnak"]["datavalue"]["value"]) for c in e.get("claims", {}).get("P5531", [])
                                 if c["mainsnak"].get("datavalue")}) for q, e in ents.items()}
        for t in batch:
            cache[t] = claims.get(qid[t], []) if qid[t] else []
    p.write_text(json.dumps(cache))
    return {t: cache.get(t, []) for t in titles if t}


def cik_for_event(ev_row, pairs: pd.DataFrame, lookup: dict, wd: dict) -> tuple[int | None, str]:
    """M1 validated Wayback pair -> M2 Wikidata -> M3 name match -> M1 unvalidated (bank/FDIC filers, fresh spin-offs)."""
    when, name = pd.Timestamp(ev_row.date), ev_row["name"]
    raw = None
    for tk in dict.fromkeys([ev_row.ticker, ev_row.fja_ticker.replace(".", "-")]):
        c = pairs[pairs.ticker == tk]
        if c.empty:
            continue
        dist = c.apply(lambda r: 0 if r.first_seen <= when <= r.last_seen else
                       min(abs((r.first_seen - when).days), abs((r.last_seen - when).days)), axis=1)
        if raw is None and dist.min() <= 30:
            raw = int(c.loc[dist.idxmin(), "cik"])
        for i in dist.sort_values(kind="stable").index:
            early_ok = when < pd.Timestamp("2017-08-28") and c.loc[i, "first_seen"] == pd.Timestamp("2017-08-28")
            if dist[i] <= 400 or early_ok:
                nm = name or c.loc[i, "title"]
                if validate_cik(int(c.loc[i, "cik"]), nm, when):
                    return int(c.loc[i, "cik"]), "m1_wayback"
    ciks = wd.get(ev_row.link, []) if isinstance(ev_row.link, str) else []
    if len(ciks) == 1 and validate_cik(ciks[0], name, when):
        return ciks[0], "m2_wikidata"
    if name:
        toks = norm_name(name).split()
        for key in [" ".join(toks[:k]) for k in range(len(toks), 0, -1)] + [ev_row.ticker.lower()]:
            # the ticker-as-name key ('beam' -> 'BEAM INC') must still match the event name
            # (without it 'tss' -> 'TSS, Inc.' (TSSI) was accepted for TSYS)
            act = [c for c in sorted(lookup.get(key, set()))[:15] if validate_cik(c, name, when)]
            if len(act) == 1:
                return act[0], "m3_name"
            if act:
                break
    if raw is not None:
        return raw, "m1_unvalidated"
    return None, "unmapped"


# ============================================================================ 4. Yahoo prices
def yahoo_chart(symbol: str, start: str = "2013-01-01", neg_max_age_h: float = 24 * 7) -> dict:
    """Raw v8 chart (same backend as yfinance; 304/304 identical availability in the probe).
    df: open high low close(split-adj) adjclose(split+div) volume; meta has firstTradeDate/longName; 'splits' and
    'dividends' keep Yahoo's events so raw price LEVELS can be rebuilt (close * later split factors) without look-ahead.
    Failed lookups (404 / no rows) are cached for neg_max_age_h only, successes forever."""
    p = CACHE / "px" / f"chart__{symbol}_{start}.pkl"
    if p.exists():
        out = pickle.loads(p.read_bytes())
        if out["ok"] or time.time() - p.stat().st_mtime < neg_max_age_h * 3600:
            return out
    r = _get(f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}", "yahoo", {"User-Agent": YAHOO_UA},
             dict(period1=int(pd.Timestamp(start, tz="UTC").timestamp()), period2=int(time.time()), interval="1d",
                  includeAdjustedClose="true", events="div,splits"), ok=(200, 404))
    out = dict(ok=False, status=r.status_code, err=None, meta=None, df=None)
    if r.status_code != 200:
        out["err"] = (r.json().get("chart", {}).get("error") or {}).get("description") if r.headers.get("content-type", "").startswith("application/json") else r.text[:100]
    else:
        res = r.json()["chart"]["result"][0]
        ts = res.get("timestamp") or []
        out["meta"] = res["meta"]
        out["splits"] = {pd.Timestamp(int(v["date"]), unit="s").normalize(): v["numerator"] / v["denominator"]
                         for v in (res.get("events") or {}).get("splits", {}).values()}
        out["dividends"] = {pd.Timestamp(int(v["date"]), unit="s").normalize(): v["amount"]
                            for v in (res.get("events") or {}).get("dividends", {}).values()}
        if ts:
            idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(res["meta"].get("exchangeTimezoneName", "America/New_York")).tz_localize(None).normalize()
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame({k: q[k] for k in ["open", "high", "low", "close", "volume"]}, index=idx, dtype="float64")
            df["adjclose"] = pd.Series(res["indicators"]["adjclose"][0]["adjclose"], index=idx, dtype="float64")
            out.update(ok=True, df=df[~df.index.duplicated(keep="last")].dropna(subset=["close"]))
        else:
            out["err"] = "no timestamps"
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(pickle.dumps(out))
    return out


def ftd_ratio(df: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    """FTD PRICE (unadjusted close of the day before settle) / Yahoo split-adjusted close on that day, indexed by
    the Yahoo trading day. Same issuer -> piecewise constant (jumps only at later splits); another issuer -> drifts."""
    if df is None or f is None or not len(f):
        return pd.Series(dtype=float)
    f = f.dropna(subset=["price"]).drop_duplicates("settle").sort_values("settle")
    f = f[f.price > 0]
    pos = df.index.searchsorted(f.settle.values, side="left") - 1     # last Yahoo day strictly before settle
    ok = pos >= 0
    days = df.index[pos[ok]]
    fresh = (f.settle.values[ok] - days.values) <= pd.Timedelta(7, unit="D").to_timedelta64()
    r = pd.Series(f.price.values[ok][fresh] / df.close.values[pos[ok]][fresh], index=days[fresh])
    return r[~r.index.duplicated(keep="last")].replace(np.inf, np.nan).dropna()


def ftd_consistency(r: pd.Series, tol: float = 0.002, min_len: int = 5) -> tuple[int, float, pd.Series]:
    """(n, share of observations on a 'plateau', per-observation flags). A plateau is a run of >= min_len consecutive
    observations whose FTD/Yahoo ratio moves < tol (log) step to step. FTD PRICE equals Yahoo's raw close to ~5e-8, so
    the same issuer gives a few long plateaus (breaks only at splits/spin-offs), another issuer gives none, even when
    highly correlated (Chubb Corp vs ACE). 2026-09-25: 631 correct spells share >= 0.872 (p1 0.997); wrong issuers
    BBT .011, DD(2017) .05, DISCK->WBD .08, MWV .02, WRK/PARA/VIAC 0; partly wrong CB .75, IR .71, JCI .71."""
    if len(r) < min_len:
        return len(r), float("nan"), pd.Series(True, index=r.index)
    d = np.log(r).diff().abs()
    seg = (d.isna() | (d >= tol)).cumsum()
    okf = seg.map(seg.value_counts()) >= min_len
    return len(r), float(okf.mean()), okf


def ftd_bad_months(df: pd.DataFrame, f: pd.DataFrame, lo, hi) -> set:
    """Business-month-ends in [lo, hi] where >= 2 FTD observations exist and most are off-plateau (see ftd_consistency)."""
    n, _, okf = ftd_consistency(ftd_ratio(df, f[(f.settle > lo) & (f.settle <= hi)]) if f is not None else pd.Series(dtype=float))
    if n < 5:
        return set()
    g = okf.groupby(okf.index.to_period("M")).agg(["size", "mean"])
    return {pd.offsets.BMonthEnd().rollforward(p.to_timestamp()) for p, r in g.iterrows() if r["size"] >= 2 and r["mean"] < 0.5}


def resolve_yahoo(spell, ev_row, cik: int | None, cur_by_cik: dict, rename_map: dict | None = None,
                  ftd_rows: pd.DataFrame | None = None, w0="2014-01-01", min_consistent: float = 0.5) -> tuple[str | None, str]:
    """Pick the Yahoo symbol whose series is THIS issuer during the spell. Rejects reused tickers
    (firstTradeDate after the spell end: LIFE->Ethos 2026, SE->Sea Ltd, FB->ETF, Q->Qnity, DOW->Dow Inc 2019).
    FIX (verifier): firstTradeDate cannot catch a ticker taken over by an issuer that ALREADY traded under another
    symbol (Yahoo re-keys the old history): BBT->Beacon Financial, PARA/VIAC->Banzai International, WRK/MWV->SW
    (Smurfit Kappa history), DD(2017)->Dow lineage. With ftd_rows (SEC FTD rows of the traded-then symbol) a candidate
    whose FTD/close ratio is mostly unstable (>= 5 obs, < min_consistent) is rejected; partly wrong series (IR 2017-20,
    JCI/CB pre-2016) pass and must be masked month by month with ftd_bad_months(). A series that does not reach back to
    the in-window spell start (EQR/AVB: 1-4 rows) is only used as a last resort and labelled '<how>_partial'."""
    end = spell.end if pd.notna(spell.end) else pd.Timestamp.today().normalize() - pd.Timedelta(1, unit="D")
    lo = max(pd.Timestamp(spell.start), pd.Timestamp(w0))
    if ftd_rows is not None and len(ftd_rows):
        ftd_rows = ftd_rows[(ftd_rows.settle > lo) & (ftd_rows.settle <= end + pd.Timedelta(3, unit="D"))]
    partial = None
    cands = [(spell.ticker.replace(".", "-"), "spell_ticker")]
    if ev_row is not None:
        cands.append((ev_row.ticker, "event_ticker"))
        t, hops = ev_row.rename_to, 0          # follow chains CBS->VIAC->PARA->PSKY, HCP->PEAK->DOC
        while isinstance(t, str) and hops < 6:
            cands.append((t, "rename_to"))
            t, hops = (rename_map or {}).get(t), hops + 1
    if cik is not None:
        cands += [(t, "cik_current") for t in cur_by_cik.get(int(cik), []) if "-P" not in t and not t.endswith(("-WT", "WS"))][:3]
    seen = set()
    for s, label in cands:
        if s in seen:
            continue
        seen.add(s)
        c = yahoo_chart(s)
        if not c["ok"]:
            continue
        ft = c["meta"].get("firstTradeDate")
        first = pd.Timestamp(ft, unit="s") if isinstance(ft, (int, float)) else c["df"].index[0]
        near = c["df"].loc[(c["df"].index >= end - pd.Timedelta(12, unit="D")) & (c["df"].index <= end)]
        if first <= end - pd.Timedelta(30, unit="D") and len(near):
            if label == "cik_current":             # the CIK route must also agree on the issuer name
                sub = submissions(int(cik)) or {}
                sec_names = [sub.get("name", "")] + [f.get("name", "") for f in sub.get("formerNames", [])]
                yn = c["meta"].get("longName") or c["meta"].get("shortName") or ""
                if not any(names_match(yn, n) for n in sec_names):
                    continue
            if ftd_rows is not None and len(ftd_rows):
                n, share, _ = ftd_consistency(ftd_ratio(c["df"], ftd_rows))
                if n >= 5 and share < min_consistent:
                    continue                           # another issuer's prices under this symbol
            if c["df"].index[0] <= lo + pd.Timedelta(10, unit="D"):
                return s, label
            partial = partial or (s, label + "_partial")
    return partial or (None, "none")


# ============================================================================ 5. SEC fails-to-deliver prices
def ftd_urls() -> dict[str, str]:
    """'YYYYMMa'/'YYYYMMb' -> URL. Files before 2017-07 sit under /files/data/frequently-requested-foia-document-
    fails-deliver-data/, later ones under /files/data/fails-deliver-data/ (guessing one path 404s)."""
    html = _cached_bytes("ftd_index.html", "https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data",
                         "sec", {"User-Agent": SEC_UA}, 24).decode()
    return {k: (h if h.startswith("http") else "https://www.sec.gov" + h)
            for h, k in re.findall(r'href="([^"]*cnsfails(\d{6}[ab])\.zip)"', html)}


def ftd_file(key: str, urls: dict) -> pd.DataFrame:
    """settle, cusip, symbol, qty, desc, price. PRICE = close of the business day before settle (unadjusted)."""
    p = CACHE / "ftd" / f"cnsfails{key}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    raw = zipfile.ZipFile(io.BytesIO(_get(urls[key], "sec", {"User-Agent": SEC_UA}).content))
    txt = raw.read(raw.namelist()[0]).decode("latin-1")
    rows = [ln.split("|") for ln in txt.splitlines()[1:]]
    rows = [r[:4] + ["|".join(r[4:-1]), r[-1]] for r in rows if len(r) >= 6]   # DESCRIPTION may contain '|'
    df = pd.DataFrame(rows, columns=["settle", "cusip", "symbol", "qty", "desc", "price"])
    df = df[df.settle.str.fullmatch(r"\d{8}")]
    df["settle"] = pd.to_datetime(df.settle, format="%Y%m%d")
    df["price"] = pd.to_numeric(df.price, errors="coerce")
    df["qty"] = pd.to_numeric(df.qty, errors="coerce")
    p.parent.mkdir(exist_ok=True)
    df.to_pickle(p)
    return df


def ftd_panel(symbols: set[str], start="2013-12", end="2026-08") -> pd.DataFrame:
    urls = ftd_urls()
    keys = sorted(k for k in urls if start.replace("-", "") <= k[:6] <= end.replace("-", ""))
    if not keys:
        raise ValueError(f"no FTD files between {start} and {end} (index lists {min(urls)}..{max(urls)})")
    out = pd.concat([f[f.symbol.isin(symbols)] for f in (ftd_file(k, urls) for k in keys)], ignore_index=True)
    out["px_date"] = out.settle - pd.offsets.BDay(1)
    return out.dropna(subset=["price"]).sort_values("px_date", ignore_index=True)
