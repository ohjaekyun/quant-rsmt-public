"""Historical ticker -> SEC CIK for S&P 500 spells (keyless; verified 2026-09-25).

Priority for a removed member (``cik_for_event``):
  M1 Wayback history of sec.gov/files/company_tickers.json (2017-08..today), every (ticker, CIK) pair walked by
     distance to the removal date and accepted only if the CIK filed a 10-K/10-Q in [date-540 d, date+10 d] and one
     of its (former) names matches -> stale pairs (ANET->Authorize.Net, KEYS->Keystone, DOW->Dow Chemical) rejected;
  M2 Wikidata P5531 of the Wikipedia article (the article's entity, so validated the same way);
  M3 exact normalised-name match in cik-lookup-data.txt, validated;
  M1 unvalidated pair in force on the date (FDIC bank filers SBNY/FRC, spin-offs removed before their first 10-K).
Current members take the CIK of the Wikipedia constituents table.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

import pandas as pd

from quantcio.data.pit.us_net import PitNet

SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_LOOKUP = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
SUBMISSIONS = "https://data.sec.gov/submissions/"
WAYBACK_CDX = (
    "https://web.archive.org/cdx/search/cdx?url=sec.gov/files/company_tickers.json&output=json"
    "&filter=statuscode:200&collapse=timestamp:6&fl=timestamp,digest,length"
)
WAYBACK_SNAP = "https://web.archive.org/web/{ts}id_/https://www.sec.gov/files/company_tickers.json"
PERIODIC = ("10-K", "10-Q", "20-F", "40-F", "10-K405", "10-KT")
WAYBACK_FIRST = pd.Timestamp("2017-08-28")
# EDGAR's filingTo of an older page is often one day before the page's last filingDate: 293 of 1,674 cached pages,
# fetched together with their main JSON, differed by exactly one day (docs/MODELING.md 84.6, check 3)
FILING_TO_SLACK_DAYS = 1
_LOOKUP_TTL_H = 24 * 30

_STOP = {
    "inc",
    "corp",
    "corporation",
    "co",
    "company",
    "the",
    "group",
    "holdings",
    "holding",
    "plc",
    "ltd",
    "limited",
    "n",
    "v",
    "nv",
    "de",
    "new",
    "sa",
    "lp",
    "llc",
    "incorporated",
    "international",
    "intl",
    "cos",
    "companies",
    "trust",
    "and",
    "of",
}


def norm_name(s: Any) -> str:
    s = re.sub(r"/[a-z]{2}/?$", "", str(s or "").lower().strip())  # 'CELGENE CORP /DE/' -> 'celgene corp'
    s = re.sub(r"[^a-z0-9 ]", " ", s.replace("&", " and "))
    return " ".join(w for w in s.split() if w not in _STOP)


def _initials(s: str) -> str:
    return "".join(w[0] for w in norm_name(s).split())


def names_match(a: Any, b: Any) -> bool:
    """Loose issuer-name match: a shared token, containment/prefix of the squashed names, or initials
    ('TSYS' ~ 'Total System Services', 'Kohl's' ~ 'KOHLS', 'VF' ~ 'V F')."""
    ta, tb = set(norm_name(a).split()), set(norm_name(b).split())
    if ta & tb:
        return True
    sa, sb = norm_name(a).replace(" ", ""), norm_name(b).replace(" ", "")
    if not sa or not sb:
        return False
    if sa in sb or sb in sa or sa[:6] == sb[:6]:
        return True
    ia, ib = _initials(a), _initials(b)
    return sa == ib or sb == ia or (len(ib) >= 3 and sa.startswith(ib))


def cik_entity(cik: int | float) -> str:
    return f"CIK{int(cik):010d}"


def entity_cik(entity: str) -> int:
    return int(str(entity).upper().removeprefix("CIK"))


# ============================================================================ SEC ticker files


def parse_company_tickers(j: dict) -> list[tuple[str, int, str]]:
    return [(str(v["ticker"]).upper(), int(v["cik_str"]), str(v.get("title", ""))) for v in j.values()]


def current_company_tickers(net: PitNet) -> list[tuple[str, int, str]]:
    raw = net.cached_bytes("sec", "company_tickers.json", SEC_TICKERS, "sec", 24)
    return parse_company_tickers(json.loads(raw))


def sec_current_tickers_by_cik(net: PitNet) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for t, c, _ in current_company_tickers(net):
        out.setdefault(c, []).append(t)
    return out


def pairs_from_snapshots(snaps: list[tuple[pd.Timestamp, list[tuple[str, int, str]]]]) -> pd.DataFrame:
    """[(seen date, [(ticker, cik, title)])] -> ticker, cik, title, first_seen, last_seen."""
    rows = [(t, c, title, seen) for seen, lst in snaps for t, c, title in lst]
    df = pd.DataFrame(rows, columns=["ticker", "cik", "title", "seen"])
    if df.empty:
        return pd.DataFrame(columns=["ticker", "cik", "title", "first_seen", "last_seen"])
    df = df.sort_values("seen")
    return (
        df.groupby(["ticker", "cik"])
        .agg(title=("title", "last"), first_seen=("seen", "min"), last_seen=("seen", "max"))
        .reset_index()
    )


def sec_ticker_history(net: PitNet, today: pd.Timestamp) -> pd.DataFrame:
    """Monthly Wayback captures of company_tickers.json (2017-08-28..) + today's file -> dated (ticker, CIK) pairs.
    ~95 files (~77 MB raw, cached gzip forever); cold ~7-8 min at 1 req/s."""
    cdx = json.loads(net.cached_bytes("sec", "wayback_cdx.json", WAYBACK_CDX, "archive", 24 * 7))[1:]
    stamps = sorted({row[0] for row in cdx if row and str(row[0]).isdigit()})
    key = hashlib.sha1(("|".join(stamps) + today.strftime("%Y%m%d")).encode()).hexdigest()[:12]
    hit = net.read_pickle("sec", f"pairs_{key}.pkl")
    if hit is not None:
        return hit
    snaps = []
    for ts in stamps:
        try:
            b = net.cached_bytes("sec", f"wb_ct/{ts}.json.gz", WAYBACK_SNAP.format(ts=ts), "archive", None)
            snaps.append((pd.Timestamp(ts[:8]), parse_company_tickers(json.loads(b))))
        except Exception as exc:  # noqa: BLE001 - one missing month is reported, not fatal
            net.note(f"Wayback SEC 티커 파일 {ts} 실패({type(exc).__name__}) — 해당 월 제외")
    snaps.append((today.normalize(), current_company_tickers(net)))
    pairs = pairs_from_snapshots(snaps)
    net.write_pickle("sec", f"pairs_{key}.pkl", pairs)
    return pairs


def parse_cik_lookup(text: str) -> dict[str, set[int]]:
    idx: dict[str, set[int]] = {}
    for line in text.splitlines():
        m = re.match(r"^(.*):(\d{10}):\s*$", line)
        if m and m.group(1):
            idx.setdefault(norm_name(m.group(1)), set()).add(int(m.group(2)))
    return idx


def cik_lookup_index(net: PitNet) -> dict[str, set[int]]:
    """sec.gov/Archives/edgar/cik-lookup-data.txt ('NAME:0000000000:', ~1.06M lines, 40 MB) -> norm name -> CIKs."""
    raw_p = net.path("sec", "cik-lookup-data.txt")
    idx_p = net.path("sec", "cik_lookup_idx.pkl")
    raw_fresh = raw_p.exists() and time.time() - raw_p.stat().st_mtime < _LOOKUP_TTL_H * 3600
    if raw_fresh and idx_p.exists() and idx_p.stat().st_mtime >= raw_p.stat().st_mtime:
        hit = net.read_pickle("sec", "cik_lookup_idx.pkl")
        if hit is not None:
            return hit
    raw = net.cached_bytes("sec", "cik-lookup-data.txt", SEC_LOOKUP, "sec", _LOOKUP_TTL_H)
    idx = parse_cik_lookup(raw.decode("latin-1"))
    net.write_pickle("sec", "cik_lookup_idx.pkl", idx)
    return idx


# ============================================================================ submissions


def page_matches(page: dict, meta: dict) -> bool:
    """A cached older submissions page still is the page its ``files`` entry describes: same ``filingCount``, first
    filing on ``filingFrom`` and none after ``filingTo`` + ``FILING_TO_SLACK_DAYS`` (EDGAR's filingTo is the range
    boundary: often a few days after the page's last filing, often one day before it). Keys missing from ``meta``
    are not checked."""
    dates = [d for d in page.get("filingDate") or [] if d]
    if "filingCount" in meta and len(page.get("accessionNumber") or []) != int(meta["filingCount"]):
        return False
    if not dates:
        return not int(meta.get("filingCount", 0) or 0)
    if meta.get("filingFrom") and min(dates) != str(meta["filingFrom"]):
        return False
    if not meta.get("filingTo"):
        return True
    try:
        last = (pd.Timestamp(str(meta["filingTo"])) + pd.Timedelta(FILING_TO_SLACK_DAYS, "D")).strftime("%Y-%m-%d")
    except ValueError:
        last = str(meta["filingTo"])
    return max(dates) <= last


def newest_periodic(sub: dict | None, forms: tuple[str, ...] = ("10-K", "10-Q", "10-KT", "10-QT")) -> str | None:
    """Filing date (ISO) of the newest original periodic report in a submissions JSON's recent block, or None."""
    rec = ((sub or {}).get("filings") or {}).get("recent") or {}
    dates = [d for f, d in zip(rec.get("form") or [], rec.get("filingDate") or [], strict=False) if f in forms and d]
    return max(dates) if dates else None


class Submissions:
    """data.sec.gov submissions JSON per CIK (cached gzip; older filing pages fetched only while needed)."""

    def __init__(self, net: PitNet):
        self.net = net
        self._mem: dict[int, dict | None] = {}
        self.mismatched: set[str] = set()  # pages fetched just now that still differ from their ``files`` entry
        self.refreshed: set[int] = set()  # CIKs whose main JSON ``refresh`` fetched again

    def get(self, cik: int) -> dict | None:
        cik = int(cik)
        if cik in self._mem:
            return self._mem[cik]
        j = self.net.read_json("sec", f"subs/{cik}.json.gz")
        if j is None:
            return self._fetch(cik)
        self._mem[cik] = j or None
        return self._mem[cik]

    def refresh(self, cik: int) -> dict | None:
        """Fetch the main JSON of the CIK again (replacing the cached copy); None on 404."""
        self.refreshed.add(int(cik))
        return self._fetch(cik)

    def _fetch(self, cik: int) -> dict | None:
        cik = int(cik)
        r = self.net.http_get(f"{SUBMISSIONS}CIK{cik:010d}.json", "sec", ok=(200, 404))
        j = r.json() if r.status_code == 200 else {}
        self.net.write_json("sec", f"subs/{cik}.json.gz", j)
        self._mem[cik] = j or None
        return self._mem[cik]

    def page(self, name: str, meta: dict | None = None) -> dict:
        """An older filings page. EDGAR re-cuts these pages whenever ``filings.recent`` rolls over (a page keeps its
        name but covers a later range), so a cached page is used only while it matches ``meta`` - its ``files`` entry
        in the main JSON - and is fetched again otherwise; a stale page left a gap of filings between it and the
        recent block (docs/MODELING.md 84.5). A page fetched just now that still differs from ``meta`` is added to
        ``mismatched``: the main JSON predates EDGAR's latest re-cut (docs/MODELING.md 92)."""
        rel = f"subs/{name}.gz"
        j = self.net.read_json("sec", rel)
        if j is not None and meta is not None and not page_matches(j, meta):
            j = None
        if j is None:
            r = self.net.http_get(f"{SUBMISSIONS}{name}", "sec", ok=(200, 404))
            j = r.json() if r.status_code == 200 else {}
            self.net.write_json("sec", rel, j)
            if meta is not None and not page_matches(j, meta):
                self.mismatched.add(name)
        return j or {}

    def names(self, cik: int) -> list[str]:
        sub = self.get(cik) or {}
        return [sub.get("name", "")] + [f.get("name", "") for f in sub.get("formerNames", []) or []]

    def filings(self, cik: int, need_before: pd.Timestamp | None = None) -> pd.DataFrame:
        """form, filingDate (Timestamp), accessionNumber, reportDate of the CIK; older pages loaded while the oldest
        filing seen is still after ``need_before`` (None = all pages)."""
        sub = self.get(cik)
        if not sub:
            return pd.DataFrame(columns=["form", "filingDate", "accessionNumber"])
        blocks = [(sub.get("filings") or {}).get("recent") or {}]
        for f in (sub.get("filings") or {}).get("files", []) or []:
            oldest = min((pd.Timestamp(d) for b in blocks for d in b.get("filingDate", [])), default=None)
            if need_before is not None and oldest is not None and oldest <= need_before:
                break
            blocks.append(self.page(f["name"], f))
        rows = [
            (form, pd.Timestamp(d), acc)
            for b in blocks
            for form, d, acc in zip(
                b.get("form", []), b.get("filingDate", []), b.get("accessionNumber", []), strict=False
            )
        ]
        return pd.DataFrame(rows, columns=["form", "filingDate", "accessionNumber"])

    def periodic_dates(self, cik: int, need_before: pd.Timestamp | None = None) -> list[pd.Timestamp]:
        f = self.filings(cik, need_before)
        return f[f["form"].isin(PERIODIC)]["filingDate"].tolist()

    def validate(self, cik: int, name: str | None, when: pd.Timestamp, days: int = 540) -> bool:
        """CIK filed a periodic report in [when-days, when+10d] and (if ``name``) one of its names matches."""
        if not self.get(cik):
            return False
        lo = when - pd.Timedelta(days, "D")
        hi = when + pd.Timedelta(10, "D")
        if not any(lo <= d <= hi for d in self.periodic_dates(cik, lo)):
            return False
        return True if not name else any(names_match(name, n) for n in self.names(cik))


# ============================================================================ Wikidata


def wikidata_ciks(net: PitNet, titles: list[str]) -> dict[str, list[int]]:
    """enwiki article title -> Wikidata P5531 CIKs (the article's entity; can differ from the issuer at the time)."""
    cache = net.read_json("sec", "wikidata_p5531.json") or {}
    todo = sorted({t for t in titles if t and t not in cache})
    for i in range(0, len(todo), 50):
        batch = todo[i : i + 50]
        try:
            j = net.http_get(
                "https://en.wikipedia.org/w/api.php",
                "wiki",
                params=dict(
                    action="query",
                    prop="pageprops",
                    ppprop="wikibase_item",
                    redirects=1,
                    titles="|".join(batch),
                    format="json",
                ),
            ).json()["query"]
        except Exception as exc:  # noqa: BLE001 - Wikidata is a fallback route only; reported
            net.note(f"Wikidata 조회 실패({type(exc).__name__}) — {len(batch)}개 제목은 M2 경로 없이 진행")
            continue
        norm = {n["from"]: n["to"] for n in j.get("normalized", [])}
        redir = {n["from"]: n["to"] for n in j.get("redirects", [])}
        byt = {pg.get("title"): (pg.get("pageprops") or {}).get("wikibase_item") for pg in j.get("pages", {}).values()}
        qid = {}
        for t in batch:
            k = t.replace("_", " ")
            k = norm.get(k, k)
            k = redir.get(k, k)
            qid[t] = byt.get(k)
        qs = sorted({q for q in qid.values() if q})
        claims: dict[str, list[int]] = {}
        if qs:
            try:
                ents = net.http_get(
                    "https://www.wikidata.org/w/api.php",
                    "wiki",
                    params=dict(action="wbgetentities", ids="|".join(qs), props="claims", format="json"),
                ).json()["entities"]
            except Exception as exc:  # noqa: BLE001
                net.note(f"Wikidata 엔티티 조회 실패({type(exc).__name__}) — M2 경로 일부 생략")
                continue
            for q, e in ents.items():
                vals = set()
                for c in (e.get("claims") or {}).get("P5531", []):
                    dv = (c.get("mainsnak") or {}).get("datavalue")
                    if dv:
                        try:
                            vals.add(int(dv["value"]))
                        except (TypeError, ValueError):
                            continue
                claims[q] = sorted(vals)
        for t in batch:
            cache[t] = claims.get(qid[t], []) if qid[t] else []
    net.write_json("sec", "wikidata_p5531.json", cache)
    return {t: cache.get(t, []) for t in titles if t}


# ============================================================================ event -> CIK


def _pair_distance(r: pd.Series, when: pd.Timestamp) -> int:
    if r["first_seen"] <= when <= r["last_seen"]:
        return 0
    return int(min(abs((r["first_seen"] - when).days), abs((r["last_seen"] - when).days)))


def cik_for_event(
    ev_row: pd.Series,
    pairs: pd.DataFrame,
    lookup: dict[str, set[int]],
    wd: dict[str, list[int]],
    subs: Submissions,
) -> tuple[int | None, str]:
    """M1 validated Wayback pair -> M2 Wikidata -> M3 name match -> M1 unvalidated (see module docstring)."""
    when, name = pd.Timestamp(ev_row["date"]), ev_row["name"]
    name = name if isinstance(name, str) and name else None
    raw = None
    for tk in dict.fromkeys([ev_row["ticker"], str(ev_row["fja_ticker"]).replace(".", "-")]):
        c = pairs[pairs["ticker"] == tk]
        if c.empty:
            continue
        dist = c.apply(lambda r: _pair_distance(r, when), axis=1)
        if raw is None and dist.min() <= 30:
            raw = int(c.loc[dist.idxmin(), "cik"])
        for i in dist.sort_values(kind="stable").index:
            early_ok = when < WAYBACK_FIRST and c.loc[i, "first_seen"] == WAYBACK_FIRST
            if dist[i] <= 400 or early_ok:
                nm = name or c.loc[i, "title"]
                if subs.validate(int(c.loc[i, "cik"]), nm, when):
                    return int(c.loc[i, "cik"]), "m1_wayback"
    link = ev_row.get("link")
    ciks = wd.get(link, []) if isinstance(link, str) else []
    if len(ciks) == 1 and subs.validate(ciks[0], name, when):
        return ciks[0], "m2_wikidata"
    if name:
        toks = norm_name(name).split()
        for key in [" ".join(toks[:k]) for k in range(len(toks), 0, -1)] + [str(ev_row["ticker"]).lower()]:
            # the ticker-as-name key must still match the event name ('tss' -> TSS Inc. was wrong for TSYS)
            act = [c for c in sorted(lookup.get(key, set()))[:15] if subs.validate(c, name, when)]
            if len(act) == 1:
                return act[0], "m3_name"
            if act:
                break
    if raw is not None:
        return raw, "m1_unvalidated"
    return None, "unmapped"
