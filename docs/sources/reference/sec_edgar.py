"""Keyless SEC EDGAR client for quantcio (US disclosures & earnings).

Verified live 2026-09-24 on AAPL / NVDA / MSFT.  Deps: requests, pandas, lxml,
edgar_text.py (HTML cleaner + section extractor).
"""
from __future__ import annotations

import re
import threading
import time
from datetime import date, timedelta

import pandas as pd
import requests
from lxml import html as LH

from edgar_text import (edgar_html_to_text, extract_section, risk_factor_headlines,
                        select_paragraphs, truncate_smart)

UA = "QuantCIO Research research@example.com"   # SEC: "Company Name contact@domain"


class _SecSession:
    """requests.Session + global <=8 req/s throttle + retry on 429/5xx."""

    def __init__(self, ua: str = UA, max_rps: float = 8.0):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self.min_dt = 1.0 / max_rps
        self._lock = threading.Lock()
        self._last = 0.0

    def get(self, url: str, timeout: float = 30, retries: int = 3, **kw) -> requests.Response:
        # FIX(verify): retry also on ConnectionError/Timeout, honour Retry-After, and do not
        # sleep after the last attempt (SEC answers some bad Archive paths with 503 -> the
        # original wasted 1+2+4 s before raising).
        last_exc = None
        for i in range(retries):
            with self._lock:
                wait = self._last + self.min_dt - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
            try:
                r = self.s.get(url, timeout=timeout, **kw)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                if i < retries - 1:
                    time.sleep(2 ** i)
                continue
            if r.status_code in (429, 500, 502, 503, 504) and i < retries - 1:
                ra = r.headers.get("Retry-After", "")
                time.sleep(min(float(ra), 30) if ra.isdigit() else 2 ** i)
                continue
            r.raise_for_status()
            return r
        raise last_exc  # only reached when every attempt was a network error


SEC = _SecSession()
_TICKER_CACHE: dict = {}

ITEM_8K = {
    "1.01": "Entry into Material Definitive Agreement", "1.02": "Termination of Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership", "1.04": "Mine Safety", "1.05": "Material Cybersecurity Incident",
    "2.01": "Completion of Acquisition or Disposition of Assets", "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation", "2.04": "Triggering Events Accelerating an Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities", "2.06": "Material Impairments",
    "3.01": "Notice of Delisting / Failure to Satisfy Listing Rule", "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders", "4.01": "Change in Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements", "5.01": "Change in Control of Registrant",
    "5.02": "Departure/Election of Directors or Officers; Compensation", "5.03": "Amendments to Articles/Bylaws; Fiscal Year Change",
    "5.04": "Suspension of Trading Under Employee Benefit Plans", "5.05": "Amendment to Code of Ethics",
    "5.07": "Submission of Matters to a Vote of Security Holders", "5.08": "Shareholder Director Nominations",
    "7.01": "Regulation FD Disclosure", "8.01": "Other Events", "9.01": "Financial Statements and Exhibits",
}
# rough event-risk weight for the signal layer (negative = adverse tail risk)
ITEM_8K_SEVERITY = {"1.03": -3, "4.02": -3, "3.01": -2, "1.05": -2, "2.06": -2, "4.01": -1, "2.04": -2,
                    "2.05": -1, "5.02": -1, "1.02": -1, "2.01": 0, "1.01": 0, "2.03": 0, "3.02": 0,
                    "2.02": 0, "7.01": 0, "8.01": 0, "5.07": 0, "9.01": 0}


# ------------------------------------------------------------------ ticker -> CIK
def ticker_to_cik(ticker: str) -> tuple[str, str]:
    """'AAPL' -> ('0000320193', 'Apple Inc.').  'BRK.B' is normalised to 'BRK-B'."""
    if not _TICKER_CACHE:
        d = SEC.get("https://www.sec.gov/files/company_tickers.json").json()
        for v in d.values():
            _TICKER_CACHE[v["ticker"].upper()] = (str(v["cik_str"]).zfill(10), v["title"])
    t = ticker.upper().replace(".", "-").strip()
    if t not in _TICKER_CACHE:
        raise KeyError(f"ticker {ticker!r} not in SEC company_tickers.json")
    return _TICKER_CACHE[t]


def is_sec_ticker(ticker: str) -> bool:
    """Router guard: KRX codes ('005930', '247540') are never in EDGAR -> send them to DART.
    No network call for 6-digit KRX codes."""
    if re.fullmatch(r"\d{6}", ticker.strip()):
        return False
    try:
        ticker_to_cik(ticker)
        return True
    except KeyError:
        return False


# ------------------------------------------------------------------ filings list
def get_filings(cik: str, include_older: bool = False) -> tuple[dict, pd.DataFrame]:
    """Return (company_meta, DataFrame of filings newest-first).

    columns: accessionNumber, filingDate (tz-naive ET calendar date), reportDate, acceptanceDateTime (str UTC),
             accepted_utc / accepted_et (tz-aware), form, items, primaryDocument, primaryDocDescription,
             isXBRL, isInlineXBRL, size, url, index_url
    'recent' holds the latest 1000 filings (measured 2026-09-24: AAPL back to 2015-07, NVDA to 2020-09).
    include_older=True pages filings.files[] (NVDA: 2,487 rows back to 1998, no duplicate accessions).
    """
    cik10 = str(int(cik)).zfill(10)
    j = SEC.get(f"https://data.sec.gov/submissions/CIK{cik10}.json").json()
    frames = [pd.DataFrame(j["filings"]["recent"])]
    if include_older:
        for f in j["filings"].get("files", []):
            older = pd.DataFrame(SEC.get(f"https://data.sec.gov/submissions/{f['name']}").json())
            frames.append(older.dropna(axis=1, how="all"))  # FIX: silences pandas FutureWarning on concat
    df = pd.concat(frames, ignore_index=True).drop_duplicates("accessionNumber")
    for c in ("items", "reportDate", "primaryDocument"):
        df[c] = df[c].fillna("")
    # FIX(verify): acceptanceDateTime is TRUE UTC ('...20:30:35.000Z' == index.htm 'Accepted 16:30:35' ET).
    # Expose tz-aware columns so the signal layer can map an 8-K to the first tradable US session.
    df["accepted_utc"] = pd.to_datetime(df["acceptanceDateTime"], utc=True, errors="coerce")
    df["accepted_et"] = df["accepted_utc"].dt.tz_convert("America/New_York")
    ncik = int(cik10)
    acc = df["accessionNumber"].str.replace("-", "", regex=False)
    df["url"] = "https://www.sec.gov/Archives/edgar/data/" + str(ncik) + "/" + acc + "/" + df["primaryDocument"]
    df["index_url"] = ("https://www.sec.gov/Archives/edgar/data/" + str(ncik) + "/" + acc + "/"
                       + df["accessionNumber"] + "-index.htm")
    df["filingDate"] = pd.to_datetime(df["filingDate"])
    meta = {k: j.get(k) for k in ["cik", "name", "tickers", "exchanges", "sic", "sicDescription", "fiscalYearEnd", "category"]}
    return meta, df.sort_values("filingDate", ascending=False, kind="stable").reset_index(drop=True)


def filing_index(index_url: str) -> pd.DataFrame:
    """Parse '<acc>-index.htm' -> DataFrame[seq, description, document, type, size, url]
    (the only keyless place that gives the exhibit TYPE such as EX-99.1)."""
    doc = LH.fromstring(SEC.get(index_url).content)
    rows = []
    for tbl in doc.xpath('//table[contains(@class,"tableFile")]'):
        for tr in tbl.xpath(".//tr")[1:]:
            tds = tr.xpath("./td")
            if len(tds) < 5:
                continue
            a = tds[2].xpath(".//a")
            href = a[0].get("href") if a else ""
            href = href.replace("/ix?doc=", "")  # inline-XBRL viewer -> raw document
            rows.append({"seq": tds[0].text_content().strip(), "description": tds[1].text_content().strip(),
                         "document": tds[2].text_content().strip().split()[0] if tds[2].text_content().strip() else "",
                         "type": tds[3].text_content().strip(), "size": tds[4].text_content().strip(),
                         "url": ("https://www.sec.gov" + href) if href.startswith("/") else href})
    return pd.DataFrame(rows)


def fetch_text(url: str) -> str:
    return edgar_html_to_text(SEC.get(url, timeout=60).content)


def _body_after_items(text: str, max_chars: int = 6000) -> str:
    """8-K main doc: skip cover page, start at the first 'Item x.xx'."""
    m = re.search(r"^item\s*\d\.\d\d", text, re.I | re.M)
    body = text[m.start():] if m else text
    body = re.split(r"^signatures?\s*$", body, flags=re.I | re.M)[0]
    return body[:max_chars].strip()


# ------------------------------------------------------------------ (a) earnings 8-K
def latest_earnings_release(ticker: str, max_chars: int = 40_000) -> dict:
    """Latest 8-K with Item 2.02 + its EX-99.1 (press release) text (+ EX-99.2 if any, e.g. NVDA CFO commentary)."""
    cik, name = ticker_to_cik(ticker)
    _, df = get_filings(cik)
    e = df[df["form"].isin(["8-K", "8-K/A"]) & df["items"].fillna("").str.contains(r"\b2\.02\b")]
    if e.empty:
        return {}
    f = e.iloc[0]
    idx = filing_index(f["index_url"])
    # FIX(verify): empty/unknown index layout -> KeyError on idx["type"] in the original
    ex = idx[idx["type"].str.upper().str.startswith("EX-99")] if "type" in idx else idx
    acc_et = f["accepted_et"]
    out = {"ticker": ticker.upper(), "company": name, "form": f["form"], "filing_date": f["filingDate"].date().isoformat(),
           "accepted": f["acceptanceDateTime"],
           "accepted_et": acc_et.isoformat() if pd.notna(acc_et) else None,
           # released at/after 16:00 ET (or before 09:30) -> first price reaction is the next session open
           "outside_rth": bool(pd.notna(acc_et) and (acc_et.hour >= 16 or (acc_et.hour, acc_et.minute) < (9, 30))),
           "report_date": f["reportDate"], "items": f["items"],
           "accession": f["accessionNumber"], "index_url": f["index_url"], "exhibits": []}
    for _, x in ex.iterrows():
        if not x["url"].lower().endswith((".htm", ".html", ".txt")):
            continue
        txt = fetch_text(x["url"])
        out["exhibits"].append({"type": x["type"], "document": x["document"], "url": x["url"],
                                "chars": len(txt), "text": txt[:max_chars]})
    return out


# ------------------------------------------------------------------ (c) recent 8-Ks
def recent_8k(ticker: str, days: int = 30, as_of: date | None = None, with_text: bool = True) -> list[dict]:
    """All 8-K / 8-K/A in the last `days` with item codes decoded + short body text."""
    cik, _ = ticker_to_cik(ticker)
    _, df = get_filings(cik)
    # FIX(verify): default 'today' in US/Eastern, not the host clock (KST is 13-14 h ahead of ET)
    as_of = as_of or pd.Timestamp.now(tz="America/New_York").date()
    since = pd.Timestamp(as_of - timedelta(days=days))
    e = df[df["form"].isin(["8-K", "8-K/A"]) & (df["filingDate"] >= since)]
    out = []
    for _, f in e.iterrows():
        items = [i.strip() for i in str(f["items"]).split(",") if i.strip()]
        rec = {"filing_date": f["filingDate"].date().isoformat(), "form": f["form"], "items": items,
               "accepted_et": f["accepted_et"].isoformat() if pd.notna(f["accepted_et"]) else None,
               "item_desc": [ITEM_8K.get(i, i) for i in items],
               "severity": min([ITEM_8K_SEVERITY.get(i, 0) for i in items] or [0]),
               "accession": f["accessionNumber"], "url": f["url"]}
        if with_text:
            rec["text"] = _body_after_items(fetch_text(f["url"]))
        out.append(rec)
    return out


# ------------------------------------------------------------------ (b) 10-Q / 10-K sections
def latest_periodic_sections(ticker: str, forms=("10-Q", "10-K"), max_chars: int = 60_000,
                             digest_chars: int = 12_000) -> dict:
    """Latest 10-Q or 10-K (whichever is newer): Item 1A risk factors + MD&A.
    If a 10-Q only says 'no material changes' (< 2,000 chars), risk factors fall back to the
    latest 10-K (risk_source tells which).  *_digest = keyword-budgeted paragraphs for the LLM."""
    cik, name = ticker_to_cik(ticker)
    _, df = get_filings(cik)
    cand = df[df["form"].isin(forms)]
    if cand.empty:  # FIX(verify): foreign private issuers (20-F/6-K: TSM, KB ADR) -> IndexError in original
        return {}
    f = cand.iloc[0]
    t0 = time.time()
    text = fetch_text(f["url"])
    rf_full = extract_section(text, f["form"], "risk_factors", max_chars=None)
    md_full = extract_section(text, f["form"], "mdna", max_chars=None)
    risk_source = f"{f['form']} {f['filingDate'].date()}"
    rf_update = ""
    if f["form"].startswith("10-Q") and len(rf_full) < 2000:
        k = df[df["form"] == "10-K"]
        if len(k):
            rf_update = rf_full
            rf_full = extract_section(fetch_text(k.iloc[0]["url"]), "10-K", "risk_factors", max_chars=None)
            risk_source = f"10-K {k.iloc[0]['filingDate'].date()} (10-Q: no material changes)"
    return {"ticker": ticker.upper(), "company": name, "form": f["form"],
            "filing_date": f["filingDate"].date().isoformat(), "period": f["reportDate"], "url": f["url"],
            "doc_chars": len(text), "fetch_parse_sec": round(time.time() - t0, 2),
            "risk_source": risk_source, "risk_update_10q": rf_update,
            "risk_factors": truncate_smart(rf_full, max_chars) if len(rf_full) > max_chars else rf_full,
            "risk_headlines": risk_factor_headlines(rf_full),
            "mdna": truncate_smart(md_full, max_chars) if len(md_full) > max_chars else md_full,
            "mdna_digest": select_paragraphs(md_full, digest_chars),
            "risk_digest": select_paragraphs(rf_full, digest_chars)}


# ------------------------------------------------------------------ full-text search (efts)
def full_text_search(q: str, forms: str = "8-K", start: str | None = None, end: str | None = None,
                     ciks: list[str] | None = None, max_pages: int = 3) -> pd.DataFrame:
    """EDGAR FTS (keyless). Hits are PER DOCUMENT (main 8-K and its EX-99.1 are separate hits),
    100 per page -> paginate with 'from' and keep one row per (accession, file_type).
    columns: adsh, file_date, form, file_type ('8-K'|'EX-99.1'..), items, display_names, document, url."""
    params = {"q": q, "forms": forms}
    if start or end:
        params.update({"dateRange": "custom", "startdt": start or "2001-01-01", "enddt": end or date.today().isoformat()})
    if ciks:
        params["ciks"] = ",".join(str(int(c)).zfill(10) for c in ciks)
    rows, frm = [], 0
    for _ in range(max_pages):
        j = SEC.get("https://efts.sec.gov/LATEST/search-index", params={**params, "from": frm}, timeout=30).json()
        hits = j.get("hits", {}).get("hits", [])
        for h in hits:
            s = h["_source"]
            doc = h["_id"].split(":", 1)[1] if ":" in h["_id"] else ""
            cik = int(s["ciks"][0]) if s.get("ciks") else 0
            rows.append({"adsh": s.get("adsh"), "file_date": s.get("file_date"), "form": s.get("form"),
                         "file_type": s.get("file_type"), "items": ",".join(s.get("items") or []),
                         "display_names": "; ".join(s.get("display_names") or []), "document": doc,
                         "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{s.get('adsh', '').replace('-', '')}/{doc}"})
        total = j.get("hits", {}).get("total", {}).get("value", 0)
        frm += len(hits)
        if not hits or frm >= total:
            break
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["adsh", "file_type", "document"]).reset_index(drop=True) if len(df) else df


# ------------------------------------------------------------------ new-filing trigger (RPA)
def filings_atom(cik: str, form: str = "8-K", count: int = 10) -> list[dict]:
    """Per-company EDGAR Atom feed (cheap poll to trigger a re-run when a new 8-K lands)."""
    import feedparser
    url = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form}"
           f"&dateb=&owner=include&count={count}&output=atom")
    fp = feedparser.parse(SEC.get(url).content)
    return [{"updated": e.get("updated"), "form": e.get("filing-type"), "accession": e.get("accession-number"),
             "filing_date": e.get("filing-date"), "index_url": e.get("link")} for e in fp.entries]


if __name__ == "__main__":
    from datetime import date as _d
    for t in ["AAPL", "NVDA", "005930", "247540"]:
        if not is_sec_ticker(t):
            print(t, "-> not an SEC registrant (route KR codes to DART)")
            continue
        cik, name = ticker_to_cik(t)
        er = latest_earnings_release(t, max_chars=30_000)
        r8 = recent_8k(t, days=30, as_of=_d(2026, 9, 24))
        ps = latest_periodic_sections(t)
        print(t, cik, name, "| 2.02", er["filing_date"], er["accepted_et"], "outside_rth", er["outside_rth"],
              [(x["type"], x["chars"]) for x in er["exhibits"]],
              "| 8-Ks", [(r["filing_date"], r["items"], r["severity"]) for r in r8],
              "| periodic", ps["form"], ps["filing_date"], len(ps["risk_factors"]), len(ps["mdna"]),
              len(ps["risk_headlines"]))
    fts = full_text_search('"Results of Operations"', "8-K", "2026-06-24", "2026-09-24", ciks=["320193", "1045810"])
    print(fts[["adsh", "file_date", "file_type", "items", "document"]].to_string())
