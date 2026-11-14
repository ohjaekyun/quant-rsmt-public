"""Keyless earnings-call transcripts via Motley Fool (fool.com) quote page -> transcript page.

Verified live 2026-09-24 for AAPL / NVDA / MSFT.  HTML scraping of a public site:
low volume (1 quote page + 1 transcript per ticker per run), browser UA, >=1s spacing.
"""
from __future__ import annotations

import re
import time

import requests
from bs4 import BeautifulSoup

BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/128.0.0.0 Safari/537.36")
_S = requests.Session()
_S.headers.update({"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.9",
                   "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
_LINK_RE = re.compile(r"/earnings/call-transcripts/(\d{4})/(\d\d)/(\d\d)/([a-z0-9\-]+?-q([1-4])-(\d{4})-earnings-call-transcript)/")


def fool_transcript_links(ticker: str, exchange: str = "nasdaq") -> list[dict]:
    """List transcript URLs linked from https://www.fool.com/quote/{exchange}/{ticker}/ (newest first).
    exchange: 'nasdaq' | 'nyse' (wrong exchange -> redirect/404, so try both)."""
    # FIX(verify): KRX codes ('005930', '247540') cost 2 requests + 1 s sleep for nothing
    if not re.fullmatch(r"[A-Za-z][A-Za-z.\-]{0,5}", ticker.strip()):
        return []
    for ex in [exchange, "nyse" if exchange == "nasdaq" else "nasdaq"]:
        r = _S.get(f"https://www.fool.com/quote/{ex}/{ticker.lower()}/", timeout=20)
        if r.ok and "call-transcript" in r.text:
            break
        time.sleep(1)
    else:
        return []
    seen, out = set(), []
    for m in _LINK_RE.finditer(r.text):
        slug = m.group(4)
        if slug in seen:
            continue
        seen.add(slug)
        out.append({"url": "https://www.fool.com" + m.group(0), "published": f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
                    "fiscal_quarter": int(m.group(5)), "fiscal_year": int(m.group(6))})
    # newest fiscal period first (publish date alone is unreliable: re-published old calls exist)
    return sorted(out, key=lambda d: (d["fiscal_year"], d["fiscal_quarter"], d["published"]), reverse=True)


_CHIP_RE = re.compile(r"\(\s*(?:NASDAQ|NYSE)?:?\s*[A-Z.]{1,6}\s*[-+]?\d+(?:\.\d+)?%\s*\)")


def fool_transcript_text(url: str) -> dict:
    """Fetch a fool.com transcript page ->
    {'title','sections':{h2 -> text}, 'takeaways','risks','summary','transcript','prepared','qa','chars'}.
    Page layout (2026): h2 DATE / CALL PARTICIPANTS / TAKEAWAYS / RISKS / SUMMARY /
    INDUSTRY GLOSSARY / Full Conference Call Transcript."""
    r = _S.get(url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    body = soup.select_one("div.article-body") or soup.select_one("article")
    if body is None:  # FIX(verify): layout change / consent wall -> AttributeError in original
        raise ValueError(f"fool.com layout changed: no article body in {url}")
    for bad in body.select("script, style, aside, figure, img"):
        bad.decompose()
    sections, cur = {}, "_intro"
    for el in body.find_all(["h2", "h3", "p", "li"]):
        if el.name == "li" and el.find_parent("li"):
            continue
        txt = _CHIP_RE.sub("", el.get_text(" ", strip=True))
        txt = re.sub(r"\s+([,.;:%)])", r"\1", re.sub(r"\s{2,}", " ", txt)).strip()
        if not txt:
            continue
        if el.name == "h2":
            cur = txt
            continue
        sections.setdefault(cur, []).append(("- " if el.name == "li" else "") + txt)
    sections = {k: "\n".join(v) for k, v in sections.items()}
    tkey = next((k for k in sections if "transcript" in k.lower()), None)
    transcript = sections.get(tkey, "") if tkey else "\n".join(sections.values())
    # Q&A starts at the first operator line that opens the question queue
    # FIX(verify): the opening Operator line often says "...there will be a question-and-answer
    # session"; re.search stopped there (pos < 2000) and the whole Q&A was lost. Take the first
    # matching Operator line AFTER the prepared remarks instead.
    m = next((x for x in re.finditer(r"^Operator:.*?(question|first caller|Q&A|queue)", transcript, re.I | re.M)
              if x.start() > 2000), None)
    prepared, qa = (transcript[:m.start()], transcript[m.start():]) if m else (transcript, "")
    get = lambda name: next((v for k, v in sections.items() if k.upper().startswith(name)), "")
    return {"url": url, "title": soup.title.get_text(strip=True) if soup.title else "",
            "sections": sections, "takeaways": get("TAKEAWAYS"), "risks": get("RISKS"),
            "summary": get("SUMMARY"), "transcript": transcript, "prepared": prepared, "qa": qa,
            "chars": len(transcript)}


def latest_transcript(ticker: str, exchange: str = "nasdaq", min_published: str | None = None) -> dict:
    """Newest transcript; pass min_published = filing date of the latest 8-K Item 2.02
    (e.g. '2026-07-30').  If the newest Fool transcript is older -> returns {'stale': True, ...}
    without fetching, so the caller falls back to EX-99.1 (+ optional Claude web_search)."""
    links = fool_transcript_links(ticker, exchange)
    if not links:
        return {}
    top = links[0]
    if min_published and top["published"] < min_published:
        return {**top, "stale": True}
    time.sleep(1.0)
    return {**top, **fool_transcript_text(top["url"]), "stale": False}


if __name__ == "__main__":
    for t, min_pub in [("AAPL", "2026-07-30"), ("NVDA", "2026-08-26"), ("005930", None), ("247540", None)]:
        d = latest_transcript(t, min_published=min_pub)
        print(t, {k: d.get(k) for k in ("url", "published", "fiscal_quarter", "fiscal_year", "stale", "chars")},
              "prepared", len(d.get("prepared", "")), "qa", len(d.get("qa", "")))
        time.sleep(1.0)
