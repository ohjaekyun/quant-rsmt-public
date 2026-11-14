"""Investor-attention data (docs/MODELING.md section 27, docs/ALTDATA.md section 5): fetchers + point-in-time features.
Research code outside ``src/quantcio``; keys only from ``.env`` (never printed).

History (usable in backtests):
- Wikipedia pageviews (US): daily user views of each company's English article (Wikimedia REST API, 2015-07 ..), the
  article found through Wikidata's SEC CIK property (P5531). No key.
- Naver search trend (KR): weekly relative search volume of each company name (NAVER API HUB, 2016 ..); needs
  NAVER_CLIENT_ID / NAVER_CLIENT_SECRET. Each request is scaled to its own maximum: only within-series ratios used.
Forward ledgers only (no history available for free):
- ApeWisdom: Reddit / 4chan ticker mention counts and ranks (no key) -> alt_ledger/us_apewisdom.jsonl.
- Reddit: posts of r/wallstreetbets, r/stocks, r/investing: ticker mentions + a simple bullish / bearish word count
  (app-only OAuth; REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET) -> alt_ledger/us_reddit.jsonl.
StockTwits is not used: its API is closed to new users and automated access is blocked.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402

UA = {"User-Agent": "quantcio-research/1.0 (academic research script)"}
WIKIDATA = "https://query.wikidata.org/sparql"
PAGEVIEWS = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{a}/daily/{s}/{e}"
)
WIKI_START = "20150701"
NAVER_TREND = "https://naverapihub.apigw.ntruss.com/search-trend/v1/search"
NAVER_START = "2016-01-01"
APEWISDOM = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/{p}"
REDDIT_TOKEN = "https://www.reddit.com/api/v1/access_token"
REDDIT_SUBS = ("wallstreetbets", "stocks", "investing")
BULL = {"call", "calls", "moon", "buy", "buying", "long", "bull", "bullish", "rocket", "squeeze", "undervalued", "yolo"}
BEAR = {
    "put",
    "puts",
    "short",
    "shorting",
    "sell",
    "selling",
    "bear",
    "bearish",
    "dump",
    "overvalued",
    "crash",
    "bagholder",
}
TICKER_STOP = {
    "A",
    "I",
    "IT",
    "ON",
    "ALL",
    "ARE",
    "FOR",
    "CAN",
    "DD",
    "CEO",
    "USA",
    "NOW",
    "ONE",
    "GO",
    "SO",
    "AM",
    "BE",
    "OR",
    "AN",
}


def say_nothing(_m: str) -> None:
    return None


# ================================================================================================ Wikipedia (US)


def wiki_articles(settings, ciks: list[int], say=say_nothing) -> dict[int, str]:
    """CIK -> English Wikipedia article title via Wikidata P5531 (cached JSON; re-queried for unseen CIKs)."""
    p = kd.root(settings, "wiki") / "articles.json"
    have = {int(k): v for k, v in json.loads(p.read_text()).items()} if p.exists() else {}
    todo = [c for c in ciks if c not in have]
    for i in range(0, len(todo), 150):
        part = todo[i : i + 150]
        vals = " ".join(f'"{c:010d}" "{c}"' for c in part)
        q = (
            "SELECT ?cik ?article WHERE { ?item wdt:P5531 ?cik . ?article schema:about ?item ; "
            f"schema:isPartOf <https://en.wikipedia.org/> . VALUES ?cik {{{vals}}} }}"
        )
        r = requests.get(
            WIKIDATA, params={"query": q}, headers={**UA, "Accept": "application/sparql-results+json"}, timeout=120
        )
        r.raise_for_status()
        for b in r.json()["results"]["bindings"]:
            cik = int(b["cik"]["value"])
            have.setdefault(cik, b["article"]["value"].rsplit("/wiki/", 1)[-1])
        for c in part:
            have.setdefault(c, "")
        time.sleep(1.0)
        say(f"Wikidata {min(i + 150, len(todo))}/{len(todo)} CIK")
    p.write_text(json.dumps({str(k): v for k, v in have.items()}, ensure_ascii=False))
    return {k: v for k, v in have.items() if v}


def wiki_fetch(settings, articles: list[str], say=say_nothing) -> dict:
    """Daily user pageviews per article (one request per article; cached parquet, refreshed after 5 days)."""
    d = kd.root(settings, "wiki") / "views"
    d.mkdir(exist_ok=True)
    end = (date.today() - timedelta(1)).strftime("%Y%m%d")
    n_new = 0
    for i, a in enumerate(sorted(set(articles)), 1):
        p = d / f"{quote(a, safe='')[:180]}.parquet"
        if p.exists() and time.time() - p.stat().st_mtime < 5 * 86400:
            continue
        url = PAGEVIEWS.format(a=quote(a, safe=""), s=WIKI_START, e=end)
        for attempt in range(8):  # 429: wait (Retry-After or growing pause) and retry
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code != 429:
                break
            time.sleep(float(r.headers.get("Retry-After") or 0) or 15.0 * (attempt + 1))
        if r.status_code == 404:
            pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "views": pd.Series(dtype=float)}).to_parquet(p)
            continue
        r.raise_for_status()
        items = r.json().get("items", [])
        pd.DataFrame(
            {"date": pd.to_datetime([x["timestamp"][:8] for x in items]), "views": [float(x["views"]) for x in items]}
        ).to_parquet(p, index=False)
        n_new += 1
        time.sleep(0.4)
        if i % 100 == 0:
            say(f"Wikipedia pageviews {i}/{len(set(articles))}")
    return {"articles": len(set(articles)), "downloaded": n_new}


def wiki_series(settings, article: str) -> pd.Series:
    p = kd.root(settings, "wiki") / "views" / f"{quote(article, safe='')[:180]}.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    return (
        pd.Series(df["views"].to_numpy(float), index=pd.DatetimeIndex(df["date"])).sort_index()
        if len(df)
        else pd.Series(dtype=float)
    )


def wiki_features(views: pd.Series) -> pd.DataFrame:
    """Daily (date = the day the value is first usable: views of day d are known on d + 1):
    wiki_abn = log((mean views of the last 7 days + 1) / (median daily views of the 56 days before + 1)),
    wiki_trend = log((mean of 28 days + 1) / (mean of the 365 days before + 1)),
    wiki_level = log(mean of 28 days + 1)."""
    if views.empty:
        return pd.DataFrame(columns=["wiki_abn", "wiki_trend", "wiki_level"])
    v = views.asfreq("D")
    m7 = v.rolling(7, min_periods=5).mean()
    base = v.shift(7).rolling(56, min_periods=40).median()
    m28 = v.rolling(28, min_periods=20).mean()
    m365 = v.shift(28).rolling(365, min_periods=250).mean()
    out = pd.DataFrame(
        {
            "wiki_abn": np.log((m7 + 1) / (base + 1)),
            "wiki_trend": np.log((m28 + 1) / (m365 + 1)),
            "wiki_level": np.log(m28 + 1),
        }
    )
    out.index = out.index + pd.Timedelta(1, "D")  # published the next day
    return out


# ============================================================================================ Naver search trend (KR)


def naver_trend_fetch(settings, names: dict[str, str], say=say_nothing) -> dict:
    """Weekly relative search volume per company name, 5 names per request (2016 ..); cached per entity."""
    cid, sec = kd.key("NAVER_CLIENT_ID"), kd.key("NAVER_CLIENT_SECRET")
    if not cid or not sec:
        return {"skipped": "NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 없음"}
    d = kd.root(settings, "naver_trend")
    p = d / "weekly.parquet"
    have = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["entity", "week", "ratio"])
    fresh = p.exists() and time.time() - p.stat().st_mtime < 5 * 86400
    todo = [(e, n) for e, n in names.items() if n and (not fresh or e not in set(have["entity"]))]
    rows = [] if not fresh else [have]
    end = date.today().strftime("%Y-%m-%d")
    h = {"X-NCP-APIGW-API-KEY-ID": cid, "X-NCP-APIGW-API-KEY": sec, "Content-Type": "application/json"}
    for i in range(0, len(todo), 5):
        part = todo[i : i + 5]
        body = {
            "startDate": NAVER_START,
            "endDate": end,
            "timeUnit": "week",
            "keywordGroups": [{"groupName": e, "keywords": [n]} for e, n in part],
        }
        try:
            r = requests.post(NAVER_TREND, json=body, headers=h, timeout=60)
        except requests.RequestException as exc:
            say(kd._redact(f"Naver trend: {type(exc).__name__}", cid, sec))
            continue
        if r.status_code != 200:
            raise RuntimeError(kd._redact(f"Naver trend HTTP {r.status_code} {r.text[:120]}", cid, sec))
        for g in r.json().get("results", []):
            for x in g["data"]:
                rows.append(
                    pd.DataFrame(
                        {"entity": [g["title"]], "week": [pd.Timestamp(x["period"])], "ratio": [float(x["ratio"])]}
                    )
                )
        time.sleep(0.15)
        if (i // 5) % 20 == 0:
            say(f"Naver trend {min(i + 5, len(todo))}/{len(todo)}")
    out = pd.concat(rows, ignore_index=True) if rows else have
    out.to_parquet(p, index=False)
    return {"entities": int(out["entity"].nunique()), "weeks": int(out["week"].nunique())}


def naver_trend_features(settings) -> dict[str, pd.DataFrame]:
    """entity -> daily features usable from (week start + 8 days):
    nsv_abn = log((ratio of the week + e) / (median of the 8 weeks before + e)), nsv_trend = log(mean 4 weeks / mean of
    the 52 weeks before); e = 1% of the series' own median (scale-invariant)."""
    p = kd.root(settings, "naver_trend") / "weekly.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    out = {}
    for e, g in df.groupby("entity"):
        s = g.set_index("week")["ratio"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        eps = max(float(s[s > 0].median()) * 0.01, 1e-6) if (s > 0).any() else 1e-6
        abn = np.log((s + eps) / (s.shift(1).rolling(8, min_periods=6).median() + eps))
        trend = np.log(
            (s.rolling(4, min_periods=4).mean() + eps) / (s.shift(4).rolling(52, min_periods=40).mean() + eps)
        )
        f = pd.DataFrame({"nsv_abn": abn, "nsv_trend": trend})
        f.index = f.index + pd.Timedelta(8, "D")
        out[str(e)] = f
    return out


# ================================================================================================ forward ledgers


def apewisdom_ledger(say=say_nothing) -> dict:
    """Today's ApeWisdom mention counts for every listed ticker (all pages)."""
    today = str(date.today())
    out = kd.ledger("us_apewisdom.jsonl")
    if out.exists() and f'"date": "{today}"' in out.read_text()[-5000:]:
        return {"skipped": "오늘 이미 기록"}
    recs, page, pages = [], 1, 1
    while page <= pages:
        r = requests.get(APEWISDOM.format(p=page), headers=UA, timeout=60)
        r.raise_for_status()
        js = r.json()
        pages = int(js.get("pages") or 1)
        for x in js.get("results", []):
            recs.append(
                {
                    "date": today,
                    **{
                        k: x.get(k)
                        for k in ("ticker", "rank", "mentions", "upvotes", "rank_24h_ago", "mentions_24h_ago")
                    },
                }
            )
        page += 1
        time.sleep(0.5)
    with open(out, "a") as fh:
        for rec in recs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    say(f"ApeWisdom 기록 {len(recs)}종목")
    return {"recorded": len(recs)}


def reddit_ledger(tickers: list[str], say=say_nothing) -> dict:
    """Ticker mentions + bullish / bearish word counts in the newest posts (last 24 h) of REDDIT_SUBS."""
    cid, sec = kd.key("REDDIT_CLIENT_ID"), kd.key("REDDIT_CLIENT_SECRET")
    if not cid or not sec:
        return {"skipped": "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET 없음"}
    ua = {"User-Agent": "quantcio-research/1.0 (by quantcio research script)"}
    tok = requests.post(
        REDDIT_TOKEN, auth=(cid, sec), data={"grant_type": "client_credentials"}, headers=ua, timeout=60
    )
    if tok.status_code != 200:
        raise RuntimeError(kd._redact(f"Reddit 인증 실패 HTTP {tok.status_code} {tok.text[:120]}", cid, sec))
    h = {**ua, "Authorization": f"bearer {tok.json()['access_token']}"}
    universe = {t.upper().replace(".", "") for t in tickers} - TICKER_STOP
    cutoff = time.time() - 86400
    mentions, bull, bear = Counter(), Counter(), Counter()
    posts = 0
    for sub in REDDIT_SUBS:
        after = None
        for _ in range(10):  # up to 1000 newest posts per subreddit
            params = {"limit": 100, **({"after": after} if after else {})}
            r = requests.get(f"https://oauth.reddit.com/r/{sub}/new", params=params, headers=h, timeout=60)
            if r.status_code != 200:
                break
            data = r.json().get("data", {})
            children = data.get("children", [])
            stop = False
            for c in children:
                p = c.get("data", {})
                if float(p.get("created_utc") or 0) < cutoff:
                    stop = True
                    break
                posts += 1
                text = f"{p.get('title', '')} {p.get('selftext', '')}"
                words = set(re.findall(r"[a-z]+", text.lower()))
                found = {m.lstrip("$") for m in re.findall(r"\$?\b[A-Z]{1,5}\b", text)} & universe
                for t in found:
                    mentions[t] += 1
                    bull[t] += len(words & BULL)
                    bear[t] += len(words & BEAR)
            after = data.get("after")
            if stop or not after:
                break
            time.sleep(0.7)  # 100 requests / minute
    today = str(date.today())
    out = kd.ledger("us_reddit.jsonl")
    with open(out, "a") as fh:
        for t, n in mentions.items():
            fh.write(
                json.dumps({"date": today, "ticker": t, "mentions": n, "bull_words": bull[t], "bear_words": bear[t]})
                + "\n"
            )
        fh.write(json.dumps({"date": today, "ticker": "_ALL_", "posts": posts}) + "\n")
    say(f"Reddit 게시물 {posts}건, 언급 종목 {len(mentions)}개")
    return {"posts": posts, "tickers": len(mentions)}
