"""
kr_text.py - Korean unstructured-text collectors for LLM sentiment/NLP.
Independently re-verified 2026-09-24 (verify-kr-text). Changes vs the claimed kr-text/kr_text.py are tagged "FIX:".

Sources (all keyless unless noted):
  NEWS
    naver_stock_news()        m.stock.naver.com/api/news/stock/{code}          (stock-tagged, but see relevance note)
    naver_article_body()      n.news.naver.com/mnews/article/{oid}/{aid}      (full body, utf-8)
    google_news_rss()         news.google.com/rss/search                       (depth to 30d; also works for US with hl=en, gl=US)
    decode_google_news_url()  Google News redirect -> publisher URL (2 requests, undocumented)
  DISCLOSURES (공시)
    naver_disclosures()/naver_disclosure_text()   KRX(KOSCOM) 공시 list + full HTML body (date only)
    daum_disclosures()/daum_disclosure_text()     same KRX feed, REAL KST timestamps (Referer required)
    dart_web_search()/dart_web_document_text()    dart.fss.or.kr web (all FSS+KRX filings, keyless)
    dart_company_rss()                            dart.fss.or.kr companyRSS (latest N filings, needs corp_code)
    DartOpenAPI                                   opendart.fss.or.kr (needs DART_API_KEY)
  RESEARCH
    naver_research()/naver_research_detail()      broker reports: opinion, goalPrice, summary, PDF url
"""
from __future__ import annotations

import io
import json
import os
import re
import time
import urllib.parse
import warnings
import zipfile
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)  # FIX: short titles parsed with bs4

KST = ZoneInfo("Asia/Seoul")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
_S = requests.Session()
_S.headers.update({"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"})

EARNINGS_RE = re.compile(r"영업\(잠정\)실적|잠정실적|매출액또는손익구조|손익구조|분기보고서|반기보고서|사업보고서|실적")

# FIX: every list function returns a DataFrame with a fixed schema even when empty
# (the claimed code returned pd.DataFrame([]) with NO columns -> KeyError downstream, e.g. 005930 Daum/Naver 30d, AAPL).
NEWS_COLS = ["published_at", "title", "snippet", "source", "url", "office_id", "article_id", "cluster_size"]
GNEWS_COLS = ["published_at", "title", "source", "source_url", "gnews_url"]
NAVER_DISC_COLS = ["date", "disclosure_id", "title", "author", "is_earnings", "url"]
DAUM_DISC_COLS = ["published_at", "disclosure_id", "title", "author", "is_earnings"]
DART_COLS = ["rcept_dt", "rcept_no", "corp_code", "corp_name", "market", "report_nm", "flr_nm", "rm",
             "is_krx", "is_earnings", "url"]
RESEARCH_COLS = ["date", "broker", "title", "research_id", "preview", "opinion_raw", "opinion", "goal_price",
                 "price_at_write", "implied_upside", "summary", "pdf_url"]
RSS_COLS = ["published_at", "title", "filer", "rcept_no", "url"]


def _frame(rows: list[dict], cols: list[str]) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows, columns=cols)
    return pd.DataFrame({c: pd.Series(dtype="datetime64[ns, Asia/Seoul]" if c == "published_at" else "object")
                         for c in cols})


def _get(url, *, params=None, headers=None, timeout=15, sleep=0.3, **kw):
    r = _S.get(url, params=params, headers=headers, timeout=timeout, **kw)
    time.sleep(sleep)  # politeness
    r.raise_for_status()
    return r


# ----------------------------------------------------------------------------------------------
# Shared HTML -> LLM-friendly plain text (tables flattened to "a | b | c" rows)
# ----------------------------------------------------------------------------------------------
def html_to_text(html: str, max_chars: int | None = None) -> str:
    soup = BeautifulSoup(html, "lxml")
    for x in soup(["script", "style", "head"]):
        x.decompose()
    for tr in soup.find_all(["tr"]):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th", "te", "tu"])]
        tr.replace_with(soup.new_string(" | ".join(c for c in cells if c) + "\n"))
    txt = soup.get_text("\n")
    txt = re.sub(r"[ \t\xa0　]+", " ", txt)
    txt = re.sub(r"\n\s*\n+", "\n", txt).strip()
    return txt[:max_chars] if max_chars else txt


# ----------------------------------------------------------------------------------------------
# 1) NAVER stock news (mobile JSON API). Desktop news_news.naver -> HTTP 410.
# ----------------------------------------------------------------------------------------------
def naver_stock_news(code: str, days: int = 14, page_size: int = 50, max_pages: int = 20) -> pd.DataFrame:
    """Stock-tagged news clusters (representative article per cluster), newest first.
    datetime is KST 'YYYYMMDDHHMM'. pageSize<=60 (100 -> HTTP 400). Depth cap ~40 pages x 50.
    WARNING (verified): for KOSDAQ names the feed is mostly generic market wrap-ups
    (247540: only 3 of 147 items mention 에코프로 in title/snippet). Filter by relevance before LLM use.
    Non-KR symbols (AAPL) return an empty frame."""
    cutoff = datetime.now(KST) - timedelta(days=days)
    rows, seen = [], set()
    for page in range(1, max_pages + 1):
        r = _get(f"https://m.stock.naver.com/api/news/stock/{code}",
                 params={"pageSize": page_size, "page": page},
                 headers={"Referer": f"https://m.stock.naver.com/domestic/stock/{code}/news"})
        clusters = r.json()
        if not clusters:
            break
        oldest = None
        for cl in clusters:
            for it in cl.get("items", []):
                if it["id"] in seen:
                    continue
                seen.add(it["id"])
                ts = datetime.strptime(it["datetime"], "%Y%m%d%H%M").replace(tzinfo=KST)
                oldest = ts if oldest is None or ts < oldest else oldest
                if ts < cutoff:
                    continue
                rows.append({
                    "published_at": ts,
                    "title": BeautifulSoup(it.get("titleFull") or it["title"], "lxml").get_text(),
                    "snippet": BeautifulSoup(it.get("body") or "", "lxml").get_text().strip(),  # FIX: body may be null
                    "source": it.get("officeName"),
                    "url": it.get("mobileNewsUrl") or f"https://n.news.naver.com/mnews/article/{it['officeId']}/{it['articleId']}",
                    "office_id": it["officeId"], "article_id": it["articleId"],
                    "cluster_size": cl.get("total", 1),
                })
        if oldest is not None and oldest < cutoff:
            break
    df = _frame(rows, NEWS_COLS)
    return df.sort_values("published_at", ascending=False).reset_index(drop=True) if len(df) else df


def naver_article_body(url: str, max_chars: int = 6000) -> dict:
    """Full article text from n.news.naver.com (utf-8). Works for mnews/article/{oid}/{aid} URLs."""
    r = _get(url, headers={"Referer": "https://m.stock.naver.com/"})
    soup = BeautifulSoup(r.text, "lxml")
    area = soup.select_one("#dic_area") or soup.select_one("#newsct_article") or soup.select_one("article")
    body = ""
    if area:
        for x in area.select("em.img_desc, span.end_photo_org, div.ab_photo, script, style, .media_end_summary"):
            x.decompose()
        body = re.sub(r"\n\s*\n+", "\n", area.get_text("\n", strip=True))[:max_chars]
    ts = soup.select_one("span._ARTICLE_DATE_TIME")
    mod = soup.select_one("span._ARTICLE_MODIFY_DATE_TIME")
    press = soup.select_one("a.media_end_head_top_logo img")
    title = soup.select_one("#title_area") or soup.select_one("h2.media_end_head_headline")
    return {
        "url": r.url,
        "title": title.get_text(strip=True) if title else None,
        "published_at": datetime.strptime(ts["data-date-time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST) if ts else None,
        "modified_at": datetime.strptime(mod["data-modify-date-time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST) if mod else None,
        "press": press.get("alt") if press else None,
        "body": body,
    }


# ----------------------------------------------------------------------------------------------
# 2) GOOGLE NEWS RSS. Titles only; max ~100 items; relevance order (sorted here by time).
# ----------------------------------------------------------------------------------------------
def google_news_rss(query: str, days: int = 14, hl: str = "ko", gl: str = "KR", tz: ZoneInfo = KST) -> pd.DataFrame:
    """KR: google_news_rss('"삼성전자"'); US: google_news_rss('NVDA Nvidia stock', hl='en', gl='US',
    tz=ZoneInfo('America/New_York')). FIX: tz is a parameter (claimed code hard-coded KST)."""
    q = f"{query} when:{days}d"
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": q, "hl": hl, "gl": gl, "ceid": f"{gl}:{hl}"})
    feed = feedparser.parse(_get(url).content)
    rows = []
    for e in feed.entries:
        src = e.get("source", {}) or {}
        title = e.title
        if src.get("title") and title.endswith(" - " + src["title"]):
            title = title[: -(len(src["title"]) + 3)]
        pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc).astimezone(tz) if e.get("published_parsed") else None
        rows.append({"published_at": pub, "title": title, "source": src.get("title"),
                     "source_url": src.get("href"), "gnews_url": e.link})
    df = _frame(rows, GNEWS_COLS)
    if len(df):  # same story is often syndicated twice; keep newest per title
        df = df.sort_values("published_at", ascending=False).drop_duplicates("title").reset_index(drop=True)
    return df


def decode_google_news_url(gnews_url: str) -> str | None:
    """Resolve news.google.com/rss/articles/<id> to the publisher URL (2 requests, ~1.2s). Undocumented endpoint."""
    art_id = urllib.parse.urlparse(gnews_url).path.split("/")[-1]
    r = _get(f"https://news.google.com/rss/articles/{art_id}")
    div = BeautifulSoup(r.text, "lxml").select_one("[data-n-a-sg]")
    if not div:
        return None
    sg, ts = div["data-n-a-sg"], div["data-n-a-ts"]
    inner = ('["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",null,1,null,null,null,null,null,0,1],'
             f'"X","X",1,[1,1,1],1,1,null,0,0,null,0],"{art_id}",{ts},"{sg}"]')
    r2 = _S.post("https://news.google.com/_/DotsSplashUi/data/batchexecute",
                 data={"f.req": json.dumps([[["Fbv4je", inner]]])},
                 headers={"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"}, timeout=15)
    r2.raise_for_status()
    return json.loads(json.loads(r2.text.split("\n\n", 1)[1])[0][2])[1]


# ----------------------------------------------------------------------------------------------
# 3) NAVER disclosures (KRX/KOSCOM feed). list 'datetime' time-of-day is bogus (~06:5x); date is OK.
#    NOTE (verified): not a complete KRX mirror - 005930's two 최대주주등소유주식변동신고서 (rcpNo 8xxxxx,
#    2026-09-08/09-18) are on DART but absent from Naver and Daum.
# ----------------------------------------------------------------------------------------------
def naver_disclosures(code: str, days: int = 30, page_size: int = 20, max_pages: int = 5) -> pd.DataFrame:
    cutoff = (datetime.now(KST) - timedelta(days=days)).date()
    rows = []
    for page in range(1, max_pages + 1):
        data = _get(f"https://m.stock.naver.com/api/stock/{code}/disclosure",
                    params={"pageSize": page_size, "page": page},
                    headers={"Referer": f"https://m.stock.naver.com/domestic/stock/{code}/notice"}).json()
        if not data:
            break
        for d in data:
            dt = datetime.fromisoformat(d["datetime"]).date()
            if dt < cutoff:
                continue
            rows.append({"date": dt, "disclosure_id": d["disclosureId"], "title": d["title"],
                         "author": d.get("author"), "is_earnings": bool(EARNINGS_RE.search(d["title"])),
                         "url": f"https://m.stock.naver.com/domestic/stock/{code}/notice/{d['disclosureId']}"})
        if datetime.fromisoformat(data[-1]["datetime"]).date() < cutoff:
            break
    return _frame(rows, NAVER_DISC_COLS)


def naver_disclosure_text(code: str, disclosure_id: int | str, max_chars: int = 8000) -> dict:
    d = _get(f"https://m.stock.naver.com/api/stock/{code}/disclosure/{disclosure_id}",
             headers={"Referer": "https://m.stock.naver.com/"}).json()["disclosure"]
    return {"title": d["title"], "category": d.get("comment"), "cause_code": d.get("causeCode"),
            "biz_date_seq": d.get("bizDateSeq"), "text": html_to_text(d["contents"] or "", max_chars)}


# ----------------------------------------------------------------------------------------------
# 4) DAUM disclosures (same KRX feed, correct KST createdAt). Referer REQUIRED (403 without).
# ----------------------------------------------------------------------------------------------
def daum_disclosures(code: str, days: int = 30, per_page: int = 30, max_pages: int = 5) -> pd.DataFrame:
    cutoff = datetime.now(KST) - timedelta(days=days)
    h = {"Referer": f"https://finance.daum.net/quotes/A{code}"}
    rows = []
    for page in range(1, max_pages + 1):
        j = _get("https://finance.daum.net/api/disclosures",
                 params={"symbolCode": f"A{code}", "page": page, "perPage": per_page}, headers=h).json()
        data = j.get("data") or []
        for d in data:
            ts = datetime.strptime(d["createdAt"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
            if ts < cutoff:
                continue
            rows.append({"published_at": ts, "disclosure_id": d["disclosureId"], "title": d["title"],
                         "author": d.get("author"), "is_earnings": bool(EARNINGS_RE.search(d["title"]))})
        if not data or page >= j.get("totalPages", 1) or \
                datetime.strptime(data[-1]["createdAt"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST) < cutoff:
            break
    return _frame(rows, DAUM_DISC_COLS)


def daum_disclosure_text(disclosure_id: str, max_chars: int = 8000) -> str:
    j = _get(f"https://finance.daum.net/api/disclosures/{disclosure_id}",
             headers={"Referer": "https://finance.daum.net/"}).json()
    return html_to_text(j["data"]["content"] or "", max_chars)


# ----------------------------------------------------------------------------------------------
# 5) DART web (keyless). detailSearch.ax accepts a 6-digit stock code in textCrpNm.
# ----------------------------------------------------------------------------------------------
_PAGEINFO_RE = re.compile(r"\[(\d+)/(\d+)\]\s*\[총\s*([\d,]+)건\]")


def _is_krx(rcp: str, rm: str) -> bool:
    # FIX: claimed code used rcp[8] == "8", which is only 유가증권(KOSPI). KOSDAQ KRX filings are 9xxxxx
    # (verified: 247540 영업(잠정)실적 20260731900486, rm='코' was flagged is_krx=False). 1xxxxx = FSS notices
    # (효력발생안내) and 0xxxxx = FSS filings. Also trust the rm marker 유/코/넥.
    return rcp[8] in ("8", "9") or bool(re.search(r"[유코넥]", rm or ""))


def dart_web_search(stock_code: str, days: int = 30, max_results: int = 100,
                    public_types: list[str] | None = None, max_pages: int = 5) -> pd.DataFrame:
    """public_types (web form 'publicType'): A001 사업 A002 반기 A003 분기 | B001 주요사항보고서 |
    C001 증권신고(지분) | D001 대량보유 D002 임원ㆍ주요주주 | I001 거래소 수시공시 I002 공정공시(영업(잠정)실적)
    I003 시장조치 I004 지분공시. FIX: follows '[p/N] [총 X건]' pagination (claimed code read page 1 only)."""
    end = datetime.now(KST).date()
    start = end - timedelta(days=days)
    rows, page, n_pages = [], 1, 1
    while page <= min(n_pages, max_pages):
        form = {"currentPage": str(page), "maxResults": str(max_results), "maxLinks": "10", "sort": "date",
                "series": "desc", "startDate": start.strftime("%Y%m%d"), "endDate": end.strftime("%Y%m%d"),
                "finalReport": "recent", "option": "corp", "textCrpNm": stock_code, "textCrpCik": ""}
        if public_types:
            form["publicType"] = public_types
        r = _S.post("https://dart.fss.or.kr/dsab007/detailSearch.ax", data=form, timeout=30,
                    headers={"Referer": "https://dart.fss.or.kr/dsab007/main.do", "X-Requested-With": "XMLHttpRequest"})
        r.raise_for_status()
        time.sleep(0.5)
        m = _PAGEINFO_RE.search(r.text)
        n_pages = int(m.group(2)) if m else 1
        soup = BeautifulSoup(r.text, "lxml")
        for tr in soup.select("table tbody tr"):
            tds = tr.find_all("td")
            a = tr.select_one("a[href*='rcpNo=']")
            if len(tds) < 5 or not a:
                continue
            corp = tr.select_one("a[href*='openCorpInfoNew']")
            mkt = tds[1].select_one("span[class^=tagCom]")
            rcp = re.search(r"rcpNo=(\d{14})", a["href"]).group(1)
            rm = tds[5].get_text(" ", strip=True) if len(tds) > 5 else ""
            rows.append({
                "rcept_dt": datetime.strptime(tds[4].get_text(strip=True), "%Y.%m.%d").date(),
                "rcept_no": rcp,
                "corp_code": re.search(r"openCorpInfoNew\('(\d{8})'", corp["href"]).group(1) if corp else None,
                "corp_name": corp.get_text(strip=True) if corp else None,
                "market": mkt.get("title") if mkt else None,          # 유가증권시장 / 코스닥시장 / ...
                "report_nm": re.sub(r"\s+", " ", a.get_text(" ", strip=True)),
                "flr_nm": tds[3].get_text(strip=True),
                "rm": rm,
                "is_krx": _is_krx(rcp, rm),
                "is_earnings": bool(EARNINGS_RE.search(a.get_text())),
                "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcp}",
            })
        page += 1
    df = _frame(rows, DART_COLS)
    return df.drop_duplicates("rcept_no").reset_index(drop=True) if len(df) else df


def latest_earnings_disclosure(stock_code: str, lookback_days: int = 200, max_chars: int = 6000) -> dict | None:
    """Most recent 영업(잠정)실적 (공정공시, I002) + its plain text. Falls back to I001 (손익구조 변동)."""
    for pts in (["I002"], ["I001"]):
        df = dart_web_search(stock_code, days=lookback_days, public_types=pts)
        if len(df):
            hit = df[df.report_nm.str.contains(r"잠정\)?실적|손익구조", regex=True)]
            if len(hit):
                # newest date first; on the same date prefer the original over '[기재정정]'
                hit = hit.assign(_corr=hit.report_nm.str.startswith("[")).sort_values(
                    ["rcept_dt", "_corr"], ascending=[False, True])
                row = hit.iloc[0].drop(labels="_corr").to_dict()
                row["text"] = dart_web_document_text(row["rcept_no"], max_chars=max_chars)
                return row
    return None


_NODE_RE = re.compile(r"(\w+)\['text'\] = \"([^\"]*)\";(.*?)\1\['dtd'\] = \"([^\"]+)\";", re.S)


def _node_field(block: str, key: str) -> str:
    m = re.search(rf"\['{key}'\] = \"(\d+)\"", block)
    return m.group(1) if m else "0"


def dart_web_document_text(rcept_no: str, max_sections: int = 25, max_chars: int = 20000,
                           section_re: str | None = None) -> str:
    """Plain text of any DART filing via the viewer iframe (/report/viewer.do).
    KRX single docs (8xxxxx KOSPI, 9xxxxx KOSDAQ; dtd=HTML) are served as MS949 -> decoded cp949.
    FSS docs (dart4.xsd) are utf-8 and must be fetched per TOC node.
    FIX: section_re (e.g. r"사업의 내용|경영진단|재무에 관한") selects TOC sections by title - without it a
    반기보고서 hits max_chars on the cover page/회사의 개요 and never reaches MD&A. Sections are prefixed '## title'."""
    main = _get(f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}").text
    nodes = []
    for var, title, block, dtd in _NODE_RE.findall(main):
        dcm = _node_field(block, "dcmNo")
        if dcm == "0":
            continue
        nodes.append((title, dcm, _node_field(block, "eleId"), _node_field(block, "offset"),
                      _node_field(block, "length"), dtd))
    if nodes and section_re:
        sel = [n for n in nodes if re.search(section_re, n[0])]
        nodes = sel or nodes
    if not nodes:
        m = re.search(r'viewDoc\("(\d+)", "(\d+)", "(\d+)", "(\d+)", "(\d+)", "([^"]+)"', main)
        if not m:
            return ""
        nodes = [(None, *m.groups()[1:])]
    parts = []
    for title, dcm, ele, off, ln, dtd in nodes[:max_sections]:
        r = _get("https://dart.fss.or.kr/report/viewer.do",
                 params={"rcpNo": rcept_no, "dcmNo": dcm, "eleId": ele, "offset": off, "length": ln, "dtd": dtd},
                 headers={"Referer": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"})
        enc = (r.encoding or "utf-8").lower()
        html = r.content.decode("cp949" if enc in ("ms949", "euc-kr", "cp949") else enc, errors="replace")
        txt = html_to_text(html)
        parts.append(f"## {title}\n{txt}" if title and len(nodes) > 1 else txt)
        if sum(map(len, parts)) > max_chars:
            break
    return "\n".join(parts)[:max_chars]


def dart_company_rss(corp_code: str) -> pd.DataFrame:
    """Latest filings for one company (needs the 8-digit corp_code; a stock code returns 0 items).
    Verified item count varies: 5 for 00126380 (삼성전자), 20 for 01160363 (에코프로비엠)."""
    f = feedparser.parse(_get(f"https://dart.fss.or.kr/api/companyRSS.xml?crpCd={corp_code}").content)
    rows = [{"published_at": datetime(*e.published_parsed[:6], tzinfo=timezone.utc).astimezone(KST),
             "title": e.title, "filer": e.get("author"),
             "rcept_no": e.link.split("rcpNo=")[-1], "url": e.link} for e in f.entries]
    return _frame(rows, RSS_COLS)


# ----------------------------------------------------------------------------------------------
# 6) DART OpenAPI (needs DART_API_KEY, 40 chars; https://opendart.fss.or.kr).
# ----------------------------------------------------------------------------------------------
class DartError(RuntimeError):
    pass


class DartOpenAPI:
    BASE = "https://opendart.fss.or.kr/api"
    # status codes: 000 OK, 010 unregistered key, 011 disabled key, 012 IP not allowed, 013 no data,
    # 014 file not found, 020 request limit exceeded (~20,000/day), 021 corp count >100, 100 bad field,
    # 101 bad access, 800 maintenance, 900 undefined, 901 key expired
    def __init__(self, api_key: str | None = None, cache_dir: str | None = None):
        self.key = api_key or os.environ.get("DART_API_KEY")
        if not self.key:
            raise DartError("DART_API_KEY not set (without crtfc_key OpenDART returns an HTML 'URL not found' page)")
        self.cache_dir = cache_dir
        self._corp_map: dict[str, dict] | None = None

    def _call(self, path: str, **params) -> requests.Response:
        r = _get(f"{self.BASE}/{path}", params={"crtfc_key": self.key, **params}, timeout=30)
        ctype = r.headers.get("content-type", "")
        if r.content[:2] == b"PK":                     # zip payload (corpCode.xml, document.xml)
            return r
        if "json" in ctype:
            j = r.json()
            if j.get("status") not in ("000", None):
                raise DartError(f"{j.get('status')}: {j.get('message')}")
        elif "xml" in ctype:
            m = re.search(rb"<status>(\d+)</status>\s*<message>(.*?)</message>", r.content, re.S)
            if m and m.group(1) != b"000":
                raise DartError(f"{m.group(1).decode()}: {m.group(2).decode('utf-8', 'replace')}")
        elif "html" in ctype:
            raise DartError("OpenDART returned HTML (missing crtfc_key or bad path)")
        return r

    @staticmethod
    def parse_corp_code_zip(content: bytes) -> dict[str, dict]:
        # FIX: ElementTree instead of BeautifulSoup(lxml-xml) - the real CORPCODE.xml has ~100k <list> entries.
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            root = ET.fromstring(z.read(z.namelist()[0]))
        out = {}
        for e in root.iter("list"):
            sc = (e.findtext("stock_code") or "").strip()
            if sc:
                out[sc] = {"corp_code": (e.findtext("corp_code") or "").strip(),
                           "corp_name": (e.findtext("corp_name") or "").strip(),
                           "modify_date": (e.findtext("modify_date") or "").strip() or None}
        return out

    def corp_code(self, stock_code: str) -> str:
        if self._corp_map is None:
            cache = os.path.join(self.cache_dir, "CORPCODE.zip") if self.cache_dir else None
            if cache and os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 7 * 86400:
                content = open(cache, "rb").read()
            else:
                content = self._call("corpCode.xml").content    # ~3MB zip, refresh weekly
                if cache:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    open(cache, "wb").write(content)
            self._corp_map = self.parse_corp_code_zip(content)
        return self._corp_map[stock_code]["corp_code"]

    def list_filings(self, stock_code: str, days: int = 30, pblntf_ty: str | None = None,
                     last_reprt_at: str = "Y") -> pd.DataFrame:
        """pblntf_ty: A 정기공시, B 주요사항보고, C 발행공시, D 지분공시, E 기타공시, F 외부감사, G 펀드,
        H 자산유동화, I 거래소공시(영업(잠정)실적 등), J 공정위공시."""
        end = datetime.now(KST).date(); start = end - timedelta(days=days)
        params = {"corp_code": self.corp_code(stock_code), "bgn_de": start.strftime("%Y%m%d"),
                  "end_de": end.strftime("%Y%m%d"), "last_reprt_at": last_reprt_at,
                  "sort": "date", "sort_mth": "desc", "page_count": 100}
        if pblntf_ty:
            params["pblntf_ty"] = pblntf_ty
        rows, page = [], 1
        while True:
            try:
                j = self._call("list.json", page_no=page, **params).json()
            except DartError as e:
                if str(e).startswith("013"):   # no data
                    break
                raise
            rows += j.get("list", [])
            if page >= int(j.get("total_page", 1)):
                break
            page += 1
        df = pd.DataFrame(rows)  # corp_code corp_name stock_code corp_cls report_nm rcept_no flr_nm rcept_dt rm
        if len(df):
            df["rcept_dt"] = pd.to_datetime(df["rcept_dt"], format="%Y%m%d").dt.date
            df["is_earnings"] = df["report_nm"].str.contains(EARNINGS_RE)
            df["is_krx"] = [_is_krx(r, m) for r, m in zip(df["rcept_no"], df.get("rm", pd.Series([""] * len(df))))]
            df["url"] = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + df["rcept_no"]
        return df

    @staticmethod
    def document_zip_to_text(content: bytes, max_chars: int = 20000) -> str:
        texts = []
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            for name in sorted(z.namelist()):
                raw = z.read(name)
                m = re.search(rb'encoding="([^"]+)"', raw[:200])
                enc = (m.group(1).decode() if m else "utf-8").lower()
                enc = "cp949" if enc in ("euc-kr", "ms949", "ksc5601") else enc
                try:
                    s = raw.decode(enc)
                except UnicodeDecodeError:
                    s = raw.decode("cp949", errors="replace")
                texts.append(html_to_text(s))   # lxml HTML parser is lenient with DART's malformed XML
        return "\n".join(texts)[:max_chars]

    def document_text(self, rcept_no: str, max_chars: int = 20000) -> str:
        return self.document_zip_to_text(self._call("document.xml", rcept_no=rcept_no).content, max_chars)


# ----------------------------------------------------------------------------------------------
# 7) NAVER broker research (opinion / target price / summary text / PDF)
# ----------------------------------------------------------------------------------------------
def naver_research(code: str, days: int = 60, page_size: int = 20, with_detail: bool = True,
                   max_pages: int = 5) -> pd.DataFrame:
    """FIX: pages until the date cutoff (claimed code read page 1 only -> silently truncated when a name has
    more than page_size reports in the window; 005930 page 1 at pageSize=20 reached back only to 2026-07-08)."""
    cutoff = (datetime.now(KST) - timedelta(days=days)).date()
    rows, seen = [], set()
    for page in range(1, max_pages + 1):
        data = _get(f"https://m.stock.naver.com/api/research/stock/{code}",
                    params={"pageSize": page_size, "page": page}, headers={"Referer": "https://m.stock.naver.com/"}).json()
        if not data:
            break
        for d in data:
            wd = datetime.strptime(d["writeDate"], "%Y-%m-%d").date()
            if wd < cutoff or d["researchId"] in seen:
                continue
            seen.add(d["researchId"])
            row = {"date": wd, "broker": d["brokerName"], "title": d["title"], "research_id": d["researchId"],
                   "preview": html_to_text(d.get("previewContent") or "")}
            if with_detail:
                row.update(naver_research_detail(d["researchId"]))
            rows.append(row)
        if datetime.strptime(data[-1]["writeDate"], "%Y-%m-%d").date() < cutoff:
            break
    return _frame(rows, RESEARCH_COLS if with_detail else RESEARCH_COLS[:5])


# FIX: keys are lower-cased with ALL whitespace removed ('StrongBuy', 'Strong Buy', 'Trading  Buy' all match);
# '없음'/'NR' (not rated, seen for 247540 IBK and 한국IR협의회) -> None instead of passing '없음' through.
_OPINION_MAP = {"매수": "Buy", "buy": "Buy", "outperform": "Buy", "강력매수": "Strong Buy", "적극매수": "Strong Buy",
                "strongbuy": "Strong Buy", "tradingbuy": "Trading Buy", "단기매수": "Trading Buy",
                "중립": "Hold", "보유": "Hold", "hold": "Hold", "neutral": "Hold", "marketperform": "Hold",
                "매도": "Sell", "sell": "Sell", "underperform": "Sell",
                "비중확대": "Overweight", "overweight": "Overweight", "비중축소": "Underweight", "underweight": "Underweight",
                "없음": None, "nr": None, "notrated": None, "": None}


def normalize_opinion(op: str | None) -> str | None:
    k = re.sub(r"\s+", "", (op or "")).lower()
    return _OPINION_MAP.get(k, op.strip() if op else None)


def naver_research_detail(research_id: int) -> dict:
    """'prevGoalPrice' is NOT the previous target (equals priceAtWriteDate) -> ignored.
    price_at_write is RAW (unadjusted) and is usually the PREVIOUS trading day's close
    (verified: 005930 7/31 reports -> 207,000 = 7/30 close; 247540 8/3 reports -> 103,500 = 7/31 close),
    but not always (005930 9/23 report: 276,500 matches no close). Recompute upside vs today's raw price."""
    rc = _get(f"https://m.stock.naver.com/api/research/company/{research_id}",
              headers={"Referer": "https://m.stock.naver.com/"}).json()["researchContent"]

    def num(v):
        try:
            x = float(str(v).replace(",", "")) if v not in (None, "") else None
        except ValueError:
            return None
        return x if x else None

    goal, px = num(rc.get("goalPrice")), num(rc.get("priceAtWriteDate"))
    op = (rc.get("opinion") or "").strip()
    return {"opinion_raw": op or None, "opinion": normalize_opinion(op),
            "goal_price": goal, "price_at_write": px,
            "implied_upside": round(goal / px - 1, 4) if goal and px else None,
            "summary": html_to_text(rc.get("content") or ""), "pdf_url": rc.get("attachUrl")}


# ----------------------------------------------------------------------------------------------
# 8) Recommended chain: one call -> LLM-ready bundle, with per-source fallbacks + error log
# ----------------------------------------------------------------------------------------------
def _norm_title(t: str) -> str:
    return re.sub(r"[\W_]+", "", (t or "")).lower()[:40]


def _relevance(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    pat = "|".join(re.escape(a) for a in aliases if a)
    t = df["title"].fillna("").str.contains(pat, regex=True)
    s = df["snippet"].fillna("").str.contains(pat, regex=True) if "snippet" in df else False
    return t.astype(int) * 2 + pd.Series(s, index=df.index).astype(int)


def collect_kr_text(code: str, name: str, days: int = 14, n_bodies: int = 5, disc_days: int = 30,
                    aliases: list[str] | None = None) -> dict:
    """FIX vs claimed chain:
    - adds news['relevance'] (2 = name/alias in title, 1 = in snippet, 0 = none) and picks article bodies by
      relevance -> cluster_size -> recency. Claimed chain took the 5 most recent Naver items, which for 247540
      were 5 generic '코스피 마감' wrap-ups (0 of 5 about 에코프로비엠).
    - news_bodies / news always present; disclosure chain falls through to the next source on an EMPTY result
      (a silently broken DART HTML parse would otherwise return 0 rows with no fallback)."""
    aliases = aliases or [name]
    errors, out = [], {"code": code, "name": name, "news_bodies": []}
    frames = []
    try:
        nv = naver_stock_news(code, days=days, max_pages=6)
        if len(nv):
            frames.append(nv.assign(provider="naver"))
    except Exception as e:
        errors.append(f"naver_news: {type(e).__name__}: {e}")
    try:
        g = google_news_rss(f'"{name}"', days=days)
        if len(g):
            frames.append(g.rename(columns={"gnews_url": "url"}).assign(snippet="", provider="google"))
    except Exception as e:
        errors.append(f"google_news: {type(e).__name__}: {e}")
    news = pd.concat(frames, ignore_index=True) if frames else _frame([], NEWS_COLS + ["provider"])
    if len(news):
        news["_k"] = news["title"].map(_norm_title)
        news = (news.sort_values(["provider", "published_at"], ascending=[False, False])  # naver first on dup
                    .drop_duplicates("_k").drop(columns="_k")
                    .sort_values("published_at", ascending=False).reset_index(drop=True))
        news["relevance"] = _relevance(news, aliases)
        cand = (news[news.provider == "naver"]
                .sort_values(["relevance", "cluster_size", "published_at"], ascending=False))
        for u in cand.url.head(n_bodies):
            try:
                b = naver_article_body(u)
                b["relevance"] = int(_relevance(pd.DataFrame([{"title": b["title"] or "", "snippet": b["body"]}]), aliases).iloc[0])
                out["news_bodies"].append(b)
            except Exception as e:
                errors.append(f"naver_body {u}: {e}")
    out["news"] = news
    out["n_relevant_news"] = int((news["relevance"] > 0).sum()) if "relevance" in news else 0
    disc, first_empty = None, None
    for label, fn in [("dart_web", lambda: dart_web_search(code, days=disc_days)),
                      ("daum", lambda: daum_disclosures(code, days=disc_days)),
                      ("naver", lambda: naver_disclosures(code, days=disc_days))]:
        try:
            d = fn()
        except Exception as e:
            errors.append(f"{label}_disclosures: {type(e).__name__}: {e}")
            continue
        if len(d):
            disc = d; out["disclosure_source"] = label
            break
        first_empty = first_empty or (label, d)
    if disc is None:
        out["disclosure_source"], disc = first_empty if first_empty else (None, _frame([], DART_COLS))
    out["disclosures"] = disc
    try:
        out["earnings"] = latest_earnings_disclosure(code)
    except Exception as e:
        errors.append(f"earnings: {type(e).__name__}: {e}")
        out["earnings"] = None
    try:
        out["research"] = naver_research(code, days=60)
    except Exception as e:
        errors.append(f"research: {type(e).__name__}: {e}")
        out["research"] = _frame([], RESEARCH_COLS)
    out["errors"] = errors
    return out
