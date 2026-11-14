"""News and press-release text for the NLP classifiers (docs/MODELING.md 64, pre-declared 2026-10-16). Research only.

    .venv/bin/python scripts/v25_news_data.py fetch --what pr        # US non-earnings 8-K press releases, 2012+
    .venv/bin/python scripts/v25_news_data.py fetch --what finnhub   # US company-news headlines, last year (free tier)

Text stays in the git-ignored cache (<cache>/alt/keyed/v25/). Naver finance / m.stock / Google News RSS / Hankyung
consensus disallow crawling in robots.txt (checked 2026-10-16), so KR headlines come only from the official Naver search
API, forward from today: the daily `api_keys.py ledger` run keeps the headline text it already fetches (Naver + Finnhub)
under <cache>/alt/keyed/v25/ledger/ (keyed_data.save_news_text) - no extra calls.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_data as v13  # noqa: E402
import v16_data as v16  # noqa: E402
import v22_tone as tone  # noqa: E402

say = v13.say
START = pd.Timestamp("2012-01-01")
PR_ITEMS = r"\b(?:1\.01|1\.02|2\.01|2\.03|2\.05|2\.06|3\.01|4\.02|5\.02|7\.01|8\.01)\b"
KEEP_CHARS = 6000  # only the title and lead are classified
FINNHUB_CAP = 200  # a window returning this many headlines is split (the endpoint truncates around 250)


def root(settings) -> Path:
    return kd.root(settings, "v25")


def _append(p: Path, recs: list[dict]) -> None:
    with gzip.open(p, "at") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ================================================================================================ US press releases


def press_releases(subs, cik: int) -> pd.DataFrame:
    """8-K filings without item 2.02 but with a news-type item, since START."""
    from quantcio.data.alt import us_sec

    blocks = us_sec.submission_blocks(subs, cik, START - pd.Timedelta(200, "D"))
    frames = [us_sec._block_frame(b) for b in blocks if b]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[~(df["accessionNumber"].notna() & df["accessionNumber"].duplicated())]
    items = df["items"].astype(str)
    df = df[(df["form"] == "8-K") & ~items.str.contains(r"\b2\.02\b") & items.str.contains(PR_ITEMS)].copy()
    df["published"] = us_sec.published_dates(df["acceptanceDateTime"], df["filingDate"]).to_numpy()
    return df[df["published"] >= START].reset_index(drop=True)


def fetch_pr(settings, cache) -> dict:
    from quantcio.data.pit.us_cik import Submissions
    from quantcio.data.pit.us_net import PitNet
    from quantcio.data.us import edgar_html_to_text, exhibits_99, parse_filing_index

    d = root(settings) / "pr"
    d.mkdir(parents=True, exist_ok=True)
    uni = v16.universe("US", settings, cache)
    chains = v16.us_chains(settings, cache, uni)
    ciks = sorted({int(c) for cs in chains.values() for c in cs})
    net = PitNet(settings)
    n_docs, n_req, t0 = 0, 0, time.time()
    for i, cik in enumerate(ciks, 1):
        p = d / f"{cik}.jsonl.gz"
        have = {r["acc"] for r in tone.read_records(p)} if p.exists() else set()
        rel = press_releases(Submissions(net), cik)
        todo = rel[~rel["accessionNumber"].isin(have)] if not rel.empty else rel
        for r in todo.itertuples(index=False):
            acc = str(r.accessionNumber)
            base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/"
            rec = {"acc": acc, "published": str(pd.Timestamp(r.published).date()), "items": str(r.items), "text": ""}
            try:
                idx = parse_filing_index(net.http_get(base + f"{acc}-index.htm", "sec").content)
                n_req += 1
                ex = exhibits_99(idx)
                if ex:
                    rec["doc"], rec["type"] = ex[0]["document"], ex[0]["type"]
                    rec["text"] = edgar_html_to_text(net.http_get(ex[0]["url"], "sec").content)[:KEEP_CHARS]
                    n_req += 1
            except Exception as exc:  # noqa: BLE001 - one filing failing is skipped (retried on the next run)
                say(f"CIK {cik} {acc}: {type(exc).__name__}")
                continue
            _append(p, [rec])
            n_docs += 1
        if i % 25 == 0:
            rate = n_req / max(time.time() - t0, 1)
            say(f"8-K press releases {i}/{len(ciks)} CIKs, {n_docs} filings, {rate:.1f} req/s")
    return {"ciks": len(ciks), "filings_new": n_docs, "requests": n_req}


# ================================================================================================ Finnhub headlines


def _finnhub(sess: requests.Session, k: str, ticker: str, a: date, b: date) -> list[dict]:
    for attempt in range(5):
        try:
            r = sess.get(
                f"{kd.FINNHUB}/company-news",
                params={"symbol": ticker, "from": str(a), "to": str(b)},
                headers={"X-Finnhub-Token": k},  # header, so the key never appears in a URL / error message
                timeout=60,
            )
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
            continue
        time.sleep(1.4)  # ~43 calls / minute (free tier: 60)
        if r.status_code == 429:
            time.sleep(60)
            continue
        js = r.json() if r.ok else []
        return js if isinstance(js, list) else []
    return []


def _headline(x: dict, ticker: str) -> dict:
    return {
        "id": x.get("id"),
        "ticker": ticker,
        "datetime": int(x.get("datetime") or 0),
        "headline": str(x.get("headline") or "")[:400],
        "summary": str(x.get("summary") or "")[:400],
        "source": str(x.get("source") or ""),
        "related": str(x.get("related") or ""),
    }


def _window(sess, k, ticker, a: date, b: date) -> list[dict]:
    js = _finnhub(sess, k, ticker, a, b)
    if len(js) >= FINNHUB_CAP and (b - a).days >= 1:
        mid = a + (b - a) // 2
        return _window(sess, k, ticker, a, mid) + _window(sess, k, ticker, mid + timedelta(1), b)
    return js


def us_recent_members(settings, cache, since: str) -> dict[str, str]:
    from quantcio.data.pit import load_pit_universe
    from quantcio.modeling.dataset import universe_params
    from quantcio.models import Market

    n, start = universe_params(Market.US, settings)
    m = load_pit_universe(Market.US, settings, cache, n, start).membership
    m = m[m["date"] >= pd.Timestamp(since)].sort_values("date").drop_duplicates("entity", keep="last")
    return dict(zip(m["ticker"].astype(str), m["entity"].astype(str), strict=True))


def fetch_finnhub(settings, cache) -> dict:
    k = kd.key("FINNHUB_API_KEY")
    if not k:
        return {"skipped": "FINNHUB_API_KEY 없음"}
    d = root(settings) / "finnhub"
    d.mkdir(parents=True, exist_ok=True)
    today = date.today()
    first = today - timedelta(365)
    members = us_recent_members(settings, cache, str(first))
    sess, n = requests.Session(), 0
    for i, (ticker, ent) in enumerate(sorted(members.items()), 1):
        p = d / f"{ticker}.jsonl.gz"
        done = {r.get("window") for r in tone.read_records(p) if r.get("window")} if p.exists() else set()
        a = first
        while a <= today:
            b = min(a + timedelta(30), today)
            w = f"{a}_{b}"
            if w not in done and b < today:
                recs = [{**_headline(x, ticker), "entity": ent} for x in _window(sess, k, ticker, a, b)]
                _append(p, [*recs, {"window": w, "n": len(recs)}])
                n += len(recs)
            a = b + timedelta(1)
        if i % 25 == 0:
            say(f"Finnhub {i}/{len(members)} tickers, {n:,} headlines")
    return {"tickers": len(members), "headlines_new": n, "from": str(first)}


def main() -> None:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--what", default="all", choices=["all", "pr", "finnhub"])
    a = ap.parse_args()
    s = load_settings()
    cache = DiskCache(s.cache_path)
    secrets = [kd.key(x) for x in ("FINNHUB_API_KEY", "NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET")]
    steps = {"pr": lambda: fetch_pr(s, cache), "finnhub": lambda: fetch_finnhub(s, cache)}
    for name, fn in steps.items():
        if a.what in ("all", name):
            try:
                say(f"{name}: {fn()}")
            except Exception as exc:  # noqa: BLE001 - the other sources still load
                say(f"{name}: FAILED {type(exc).__name__}: {kd._redact(str(exc), *secrets)}")


if __name__ == "__main__":
    main()
