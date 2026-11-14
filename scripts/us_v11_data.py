"""US v11 block N (docs/MODELING.md section 29): 8-K filing-time anomalies, 10-K text drift (Item 1A / Item 7 cosine
similarity), naive distance-to-default, Wikipedia pageview spikes, Tranco domain ranks, and a GitHub forward ledger.
Research code outside ``src/quantcio``. SEC acceptanceDateTime values are UTC (checked on Apple's 16:30 ET 8-Ks).
"""

from __future__ import annotations

import gzip
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import us_v8_data as v8  # noqa: E402

UA = {"User-Agent": "quantcio-research/1.0 (academic research script)"}
TRANCO_DATE = "https://tranco-list.eu/api/lists/date/{d}"
TRANCO_TOP = 100_000
TRANCO_START = date(2019, 8, 5)
WIKIDATA = "https://query.wikidata.org/sparql"
FEATURES = [
    "fil_dump_90",
    "fil_dump_10",
    "fil_after_90",
    "txt_sim_1a",
    "txt_sim_7",
    "txt_len_1a",
    "dd",
    "dd_chg",
    "dd_mom_gap",
    "wiki_spike_1",
    "wiki_spike_3",
    "web_rank_4w",
    "web_rank_12w",
]
DOCS = {
    "fil_dump_90": ("8-K 숨기기 시간대 제출 비율(90일)", "8-K dump-hour share (90d)"),
    "fil_dump_10": ("최근 10일 숨기기 시간대 8-K 여부", "dump-hour 8-K in the last 10 days"),
    "fil_after_90": ("8-K 장 마감 후 제출 비율(90일)", "8-K after-close share (90d)"),
    "txt_sim_1a": ("10-K 위험요인 문장 유사도(직전 대비)", "10-K Item 1A similarity"),
    "txt_sim_7": ("10-K 경영진 분석 문장 유사도(직전 대비)", "10-K Item 7 similarity"),
    "txt_len_1a": ("10-K 위험요인 길이 변화(log)", "10-K Item 1A length change"),
    "dd": ("부도거리(단순 Merton)", "naive distance to default"),
    "dd_chg": ("부도거리 4주 변화", "distance-to-default change (4w)"),
    "dd_mom_gap": ("부도거리 변화 순위 − 1개월 수익률 순위", "credit-equity gap"),
    "wiki_spike_1": ("위키백과 1일 조회수 급증(÷ 직전 20일)", "Wikipedia 1-day spike"),
    "wiki_spike_3": ("위키백과 3일 조회수 급증(÷ 직전 20일)", "Wikipedia 3-day spike"),
    "web_rank_4w": ("웹 도메인 순위 개선(4주, log)", "Tranco rank change (4w)"),
    "web_rank_12w": ("웹 도메인 순위 개선(12주, log)", "Tranco rank change (12w)"),
}


def _say(_m: str) -> None:
    return None


# ================================================================================================ 8-K filing times


def filing_times(subs, cik: int) -> pd.DataFrame:
    """Every 8-K of the CIK with its acceptance time in New York (UTC values converted)."""
    sub = subs.get(int(cik)) or {}
    blocks = [(sub.get("filings") or {}).get("recent") or {}]
    blocks += [subs.page(f["name"]) for f in (sub.get("filings") or {}).get("files", []) or []]
    rows = []
    for b in blocks:
        for form, acc in zip(b.get("form", []) or [], b.get("acceptanceDateTime", []) or [], strict=False):
            if str(form).startswith("8-K") and acc:
                ts = pd.to_datetime(acc, utc=True, errors="coerce")
                if pd.notna(ts):
                    rows.append(ts.tz_convert("America/New_York"))
    if not rows:
        return pd.DataFrame(columns=["when", "dump", "after"])
    ny = pd.DatetimeIndex(sorted(rows))
    dump = (ny.dayofweek >= 5) | ((ny.dayofweek == 4) & (ny.hour >= 16)) | (ny.hour >= 20) | (ny.hour < 6)
    after = ny.hour >= 16
    return pd.DataFrame(
        {"when": ny.tz_convert("UTC").tz_localize(None), "dump": dump.astype(float), "after": after.astype(float)}
    )


def filing_features(dates: np.ndarray, fl: pd.DataFrame) -> np.ndarray:
    out = np.full((len(dates), 3), np.nan)
    if fl.empty:
        out[:, :] = [0.0, 0.0, 0.0]
        return out
    w = fl["when"].to_numpy(dtype="datetime64[ns]")
    cd, ca = np.concatenate([[0.0], np.cumsum(fl["dump"])]), np.concatenate([[0.0], np.cumsum(fl["after"])])
    for i, d in enumerate(dates):
        hi = int(np.searchsorted(w, d, "left"))
        lo90 = int(np.searchsorted(w, d - np.timedelta64(90, "D"), "left"))
        lo10 = int(np.searchsorted(w, d - np.timedelta64(10, "D"), "left"))
        n = hi - lo90
        out[i, 0] = (cd[hi] - cd[lo90]) / n if n else 0.0
        out[i, 1] = 1.0 if cd[hi] - cd[lo10] > 0 else 0.0
        out[i, 2] = (ca[hi] - ca[lo90]) / n if n else 0.0
    return out


# ================================================================================================ 10-K text drift

_ITEM_1A = re.compile(r"item\s*1a\.?\s*[-:–—]?\s*risk\s+factors", re.I)
_ITEM_1B = re.compile(r"item\s*(1b|1c|2)\.?\s*[-:–—]?\s*(unresolved|cybersecurity|properties)", re.I)
_ITEM_7 = re.compile(r"item\s*7\.?\s*[-:–—]?\s*management.{0,5}s\s+discussion", re.I)
_ITEM_7A = re.compile(r"item\s*(7a|8)\.?\s*[-:–—]?\s*(quantitative|financial\s+statements)", re.I)
_WORD = re.compile(r"[a-z]{3,}")


def _section(text: str, start: re.Pattern, end: re.Pattern) -> str:
    """The longest span from a start heading to the next end heading (skips the table of contents)."""
    best = ""
    for m in start.finditer(text):
        e = end.search(text, m.end())
        span = text[m.end() : e.start()] if e else ""
        if len(span) > len(best):
            best = span
    return best if len(best) > 2000 else ""


def fetch_sections(net, subs, ciks, root: Path, say=_say) -> dict:
    """Item 1A / Item 7 text of every 10-K since 2010 (gzip JSON per CIK; resumable)."""
    d = root / "tenk_sections"
    d.mkdir(exist_ok=True)
    n_doc = n_new = 0
    t0 = time.time()
    for i, cik in enumerate(sorted(set(ciks)), 1):
        p = d / f"{cik}.json.gz"
        have = json.loads(gzip.decompress(p.read_bytes())) if p.exists() else {}
        changed = False
        for f in v8.ten_k_list(subs, cik):
            n_doc += 1
            if f["acc"] in have:
                continue
            url = v8.ARCHIVE.format(cik=cik, acc=f["acc"].replace("-", ""), doc=f["doc"])
            try:
                r = net.http_get(url, "sec", ok=(200, 404))
            except Exception:  # noqa: BLE001 - retried on the next run
                continue
            text = v8.html_text(r.content) if r.status_code == 200 else ""
            have[f["acc"]] = {
                "filed": f["filed"],
                "s1a": _section(text, _ITEM_1A, _ITEM_1B)[:400_000],
                "s7": _section(text, _ITEM_7, _ITEM_7A)[:400_000],
            }
            changed = True
            n_new += 1
        if changed:
            p.write_bytes(gzip.compress(json.dumps(have).encode()))
        if i % 25 == 0:
            say(f"10-K sections {i}/{len(set(ciks))} ({time.time() - t0:.0f}s, new {n_new})")
    return {"docs": n_doc, "new": n_new}


def _cos(a: str, b: str) -> float:
    if not a or not b:
        return float("nan")
    ca, cb = Counter(_WORD.findall(a.lower())), Counter(_WORD.findall(b.lower()))
    num = sum(v * cb.get(k, 0) for k, v in ca.items())
    den = math.sqrt(sum(v * v for v in ca.values())) * math.sqrt(sum(v * v for v in cb.values()))
    return num / den if den > 0 else float("nan")


def text_table(root: Path, ciks) -> pd.DataFrame:
    """(avail, sim_1a, sim_7, len_1a) per 10-K of the CIK chain vs the previous 10-K."""
    recs = []
    for cik in ciks:
        p = root / "tenk_sections" / f"{cik}.json.gz"
        if p.exists():
            recs += list(json.loads(gzip.decompress(p.read_bytes())).values())
    recs = sorted(recs, key=lambda r: r["filed"])
    out = []
    for prev, cur in zip(recs, recs[1:], strict=False):
        l1, l0 = len(cur["s1a"]), len(prev["s1a"])
        out.append(
            {
                "avail": pd.Timestamp(cur["filed"]) + pd.Timedelta(1, "D"),
                "sim_1a": _cos(cur["s1a"], prev["s1a"]),
                "sim_7": _cos(cur["s7"], prev["s7"]),
                "len_1a": math.log(l1 / l0) if l1 > 0 and l0 > 0 else float("nan"),
            }
        )
    return pd.DataFrame(out)


# ================================================================================================ distance to default


def naive_dd(mcap: np.ndarray, debt: np.ndarray, vol: np.ndarray, ret_1y: np.ndarray) -> np.ndarray:
    """Bharath-Shumway (2008) naive distance to default, T = 1 year."""
    e, f = mcap, np.where(debt > 0, debt, np.nan)
    sig_d = 0.05 + 0.25 * vol
    v = e + f
    sig_v = (e / v) * vol + (f / v) * sig_d
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.log(v / f) + (ret_1y - 0.5 * sig_v**2)) / sig_v


# ================================================================================================ Wikipedia spikes


def wiki_spikes(views: pd.Series) -> pd.DataFrame:
    if views.empty:
        return pd.DataFrame(columns=["wiki_spike_1", "wiki_spike_3"])
    v = views.asfreq("D")
    base = v.shift(3).rolling(20, min_periods=15).mean()
    out = pd.DataFrame(
        {
            "wiki_spike_1": np.log((v + 1) / (base + 1)),
            "wiki_spike_3": np.log((v.rolling(3, min_periods=3).mean() + 1) / (base + 1)),
        }
    )
    out.index = out.index + pd.Timedelta(1, "D")
    return out


# ================================================================================================ Tranco ranks


def company_links(settings, ciks: list[int], say=_say) -> dict[int, dict]:
    """CIK -> {"domain": registrable domain of the official website, "github": org} via Wikidata (cached)."""
    p = kd.root(settings, "wiki") / "links.json"
    have = {int(k): v for k, v in json.loads(p.read_text()).items()} if p.exists() else {}
    todo = [c for c in ciks if c not in have]
    for i in range(0, len(todo), 150):
        part = todo[i : i + 150]
        vals = " ".join(f'"{c:010d}" "{c}"' for c in part)
        q = (
            "SELECT ?cik ?web ?gh WHERE { ?item wdt:P5531 ?cik . OPTIONAL { ?item wdt:P856 ?web } "
            f"OPTIONAL {{ ?item wdt:P2037 ?gh }} VALUES ?cik {{{vals}}} }}"
        )
        r = requests.get(
            WIKIDATA, params={"query": q}, headers={**UA, "Accept": "application/sparql-results+json"}, timeout=120
        )
        r.raise_for_status()
        webs: dict[int, list[str]] = {}
        ghs: dict[int, str] = {}
        for b in r.json()["results"]["bindings"]:
            cik = int(b["cik"]["value"])
            if "web" in b:
                webs.setdefault(cik, []).append(b["web"]["value"])
            if "gh" in b:
                ghs.setdefault(cik, b["gh"]["value"])
        for c in part:
            doms = sorted(
                {_domain(w) for w in webs.get(c, []) if _domain(w)}, key=lambda x: (not x.endswith(".com"), len(x))
            )
            have[c] = {"domain": doms[0] if doms else "", "github": ghs.get(c, "")}
        time.sleep(1.0)
        say(f"Wikidata links {min(i + 150, len(todo))}/{len(todo)}")
    p.write_text(json.dumps({str(k): v for k, v in have.items()}))
    return have


def _domain(url: str) -> str:
    h = re.sub(r"^https?://", "", str(url).lower()).split("/")[0].split(":")[0]
    h = h[4:] if h.startswith("www.") else h
    parts = h.split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "com", "net", "org", "ac") and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else ""


def fetch_tranco(settings, domains: set[str], say=_say) -> dict:
    """Weekly (Monday) Tranco top-100k ranks of ``domains`` since 2019-08 (compact parquet, resumable)."""
    d = kd.root(settings, "tranco")
    p = d / "ranks.parquet"
    have = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["date", "domain", "rank"])
    done = set(pd.to_datetime(have["date"]).dt.date) if len(have) else set()
    rows = [have] if len(have) else []
    day, n = TRANCO_START, 0
    while day <= date.today() - timedelta(2):
        if day not in done:
            try:
                meta = requests.get(TRANCO_DATE.format(d=day), headers=UA, timeout=60).json()
                if meta.get("available"):
                    url = meta["download"].rsplit("/", 1)[0] + f"/{TRANCO_TOP}"
                    txt = requests.get(url, headers=UA, timeout=180).text
                    hit = []
                    for ln in txt.splitlines():
                        rk, _, dom = ln.partition(",")
                        if dom.strip() in domains:
                            hit.append((pd.Timestamp(day), dom.strip(), int(rk)))
                    rows.append(pd.DataFrame(hit or [(pd.Timestamp(day), "", 0)], columns=["date", "domain", "rank"]))
                    n += 1
            except (requests.RequestException, ValueError):
                pass
            time.sleep(0.5)
            if n and n % 25 == 0:
                say(f"Tranco {day} ({n} lists)")
        day += timedelta(7)
    out = pd.concat(rows, ignore_index=True) if rows else have
    out.to_parquet(p, index=False)
    return {"lists": int(pd.to_datetime(out["date"]).nunique()) if len(out) else 0, "new": n}


def tranco_features(settings, domain: str) -> pd.DataFrame:
    p = kd.root(settings, "tranco") / "ranks.parquet"
    if not domain or not p.exists():
        return pd.DataFrame(columns=["web_rank_4w", "web_rank_12w"])
    df = pd.read_parquet(p)
    all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(df["date"]).unique()))
    s = df[df["domain"] == domain].set_index("date")["rank"].astype(float).reindex(all_dates)
    s = s.fillna(TRANCO_TOP * 1.5) if s.notna().any() else s  # listed before but not in the top 100k: below it
    out = pd.DataFrame({"web_rank_4w": np.log(s.shift(4) / s), "web_rank_12w": np.log(s.shift(12) / s)})
    out.index = out.index + pd.Timedelta(1, "D")
    return out


# ================================================================================================ GitHub ledger


def github_ledger(orgs: dict[str, str], say=_say) -> dict:
    """Forward ledger (alt_ledger/us_github.jsonl): org public repos / followers and the top-30 repositories' stars,
    forks and open issues (GITHUB_TOKEN)."""
    tok = kd.key("GITHUB_TOKEN")
    if not tok:
        return {"skipped": "GITHUB_TOKEN 없음"}
    h = {**UA, "Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    today = str(date.today())
    out = kd.ledger("us_github.jsonl")
    n = 0
    with open(out, "a") as fh:
        for ticker, org in orgs.items():
            try:
                o = requests.get(f"https://api.github.com/orgs/{org}", headers=h, timeout=60)
                s = requests.get(
                    "https://api.github.com/search/repositories",
                    params={"q": f"org:{org}", "sort": "stars", "per_page": 30},
                    headers=h,
                    timeout=60,
                )
            except requests.RequestException:
                continue
            if o.status_code != 200:
                continue
            items = s.json().get("items", []) if s.status_code == 200 else []
            rec = {
                "date": today,
                "ticker": ticker,
                "org": org,
                "public_repos": o.json().get("public_repos"),
                "followers": o.json().get("followers"),
                "stars_top30": sum(int(x.get("stargazers_count") or 0) for x in items),
                "forks_top30": sum(int(x.get("forks_count") or 0) for x in items),
                "open_issues_top30": sum(int(x.get("open_issues_count") or 0) for x in items),
                "pushed_30d": sum(
                    1 for x in items if str(x.get("pushed_at", ""))[:10] >= str(date.today() - timedelta(30))
                ),
            }
            fh.write(json.dumps(rec) + "\n")
            n += 1
            time.sleep(2.2)  # search API: 30 requests / minute
    say(f"GitHub 기록 {n}개 기관")
    return {"recorded": n}
