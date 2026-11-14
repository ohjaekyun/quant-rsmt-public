"""Data loading for later research (docs/MODELING.md 51, 2026-10-12): strikes (KR production-halt filing reasons,
KOSIS labour-dispute statistics, BLS major work stoppages), US analyst estimates (Yahoo earnings dates with the
consensus at the time; a DAILY forward ledger of the EPS trend / revisions / price targets) and more OpenDART periodic-
report items. Collection only - every test is pre-declared separately. Research only.

    .venv/bin/python scripts/v18_data.py fetch --what all       # or kr_halts, kosis, bls, earnings, dart_extra
    .venv/bin/python scripts/v18_data.py ledger                 # daily: US analyst estimates (alt_ledger/us_estimates/)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_data as v13  # noqa: E402
import v16_data as v16  # noqa: E402

from quantcio.config import PROJECT_ROOT  # noqa: E402

say = v13.say
BLS_LISTING = "https://www.bls.gov/web/wkstp/monthly-listing.xlsx"
BLS_UA = {"User-Agent": "Mozilla/5.0 (compatible; research bot)"}
KOSIS_DATA = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
KOSIS_TABLES = {"strikes_total": "DT_11826_N001", "strikes_cause": "DT_11826_N003", "strikes_industry": "DT_11826_N005"}
DART_EXTRA = ("hyslrChgSttus", "tesstkAcqsDspsSttus", "accnutAdtorNmNdAdtOpinion", "hmvAuditAllSttus", "mrhlSttus")
STRIKE = re.compile(r"파업|노조|노동조합|쟁의|노사|태업|직장폐쇄")
ACCIDENT = re.compile(r"화재|폭발|사고|정전|설비|고장|누출|재해|붕괴|침수|가스")
EST_LEDGER = PROJECT_ROOT / "alt_ledger" / "us_estimates"
DART_WEB_GAP = 6.0  # seconds between DART web filings (the site blocked an IP after ~700 requests in 34 minutes)


def root(settings) -> Path:
    return kd.root(settings, "v18")


def _kr_codes(settings, cache) -> list[str]:
    return sorted(v16.universe("KR", settings, cache)["entity"].astype(str))


# ================================================================================================ strikes


def halt_filings(settings, codes: list[str]) -> pd.DataFrame:
    """Original (non-correction) '생산중단' filings of the codes in the cached DART lists."""
    cc = v16.kr_corp_codes(settings)
    rows = []
    for code in codes:
        corp = cc.get(code)
        if not corp:
            continue
        for f in (Path(settings.cache_path) / "alt" / "kr_dart" / "list" / corp).glob("*.json"):
            for r in json.loads(f.read_text()).get("rows", []):
                nm = re.sub(r"\s+", "", str(r.get("report_nm") or ""))
                if "생산중단" in nm and not nm.startswith("["):
                    rows.append({"code": code, "rcept_no": r["rcept_no"], "date": r["rcept_dt"], "report_nm": nm})
    return pd.DataFrame(rows).drop_duplicates("rcept_no") if rows else pd.DataFrame()


def classify(text: str) -> tuple[str, str]:
    t = re.sub(r"\s+", " ", text)
    s, a = STRIKE.search(t), ACCIDENT.search(t)
    if s and (not a or s.start() <= a.start()):
        m, reason = s, "strike"
    elif a:
        m, reason = a, "accident"
    else:
        return "other", t[:200]
    return reason, t[max(0, m.start() - 120) : m.end() + 180]


def fetch_kr_halts(settings, cache) -> dict:
    from quantcio.data.kr import KRProvider

    out = root(settings) / "kr_halts.jsonl"
    done = {json.loads(x)["rcept_no"] for x in out.read_text().splitlines()} if out.exists() else set()
    hf = halt_filings(settings, _kr_codes(settings, cache))
    todo = hf[~hf["rcept_no"].isin(done)] if len(hf) else hf
    say(f"KR halt filings: {len(hf)} ({len(done)} already read)")
    prov = KRProvider(settings, cache)
    fails, n = 0, 0
    for r in todo.itertuples(index=False):
        try:
            text = prov._dart_web_document_text(r.rcept_no, 8000)
            fails = 0
        except Exception as exc:  # noqa: BLE001 - a blocked / failed viewer: stop after three in a row
            fails += 1
            say(f"{r.rcept_no}: {type(exc).__name__}")
            if fails >= 3:
                say("DART 웹 실패 3회 연속 — 차단 의심, 중단(다음 실행이 이어 받음)")
                break
            time.sleep(60)
            continue
        reason, snip = classify(text)
        with out.open("a") as fh:
            fh.write(json.dumps({**r._asdict(), "reason": reason, "snippet": snip}, ensure_ascii=False) + "\n")
        n += 1
        if n % 25 == 0:
            say(f"KR halts {n}/{len(todo)}")
        time.sleep(DART_WEB_GAP)
    return {"filings": len(hf), "read_now": n}


def fetch_kosis(settings) -> dict:
    k = kd.key("KOSIS_API_KEY")
    if not k:
        return {"skipped": "KOSIS_API_KEY 없음"}
    stats = {}
    for name, tbl in KOSIS_TABLES.items():
        params = {
            "method": "getList",
            "apiKey": k,
            "orgId": "118",
            "tblId": tbl,
            "itmId": "ALL",
            "objL1": "ALL",
            "objL2": "ALL",
            "objL3": "",
            "prdSe": "Y",
            "startPrdDe": "1996",
            "endPrdDe": str(pd.Timestamp.today().year),
            "format": "json",
            "jsonVD": "Y",
        }
        r = requests.get(KOSIS_DATA, params=params, timeout=120)
        try:
            js = r.json()
        except ValueError:
            stats[name] = kd._redact(r.text[:120], k)
            continue
        if isinstance(js, dict):  # error object
            params["objL2"] = ""
            js = requests.get(KOSIS_DATA, params=params, timeout=120).json()
        if isinstance(js, list):
            pd.DataFrame(js).to_parquet(root(settings) / f"kosis_{name}.parquet", index=False)
            stats[name] = len(js)
        else:
            stats[name] = kd._redact(str(js)[:120], k)
        time.sleep(0.5)
    return stats


def fetch_bls(settings) -> dict:
    import us_v7_data as v7

    p = root(settings) / "bls_stoppages.xlsx"
    r = requests.get(BLS_LISTING, headers=BLS_UA, timeout=120)
    r.raise_for_status()
    p.write_bytes(r.content)
    sheets = v7.read_xlsx(p)
    rows = next(iter(sheets.values()))
    hi = next(i for i, x in enumerate(rows) if x and str(x[0]).startswith("Organizations"))
    head = [str(h or f"col{j}").split("[")[0].strip() for j, h in enumerate(rows[hi])]
    body = [x + [None] * (len(head) - len(x)) for x in rows[hi + 1 :] if x and x[0]]
    df = pd.DataFrame([b[: len(head)] for b in body], columns=head)
    for c in df.columns:  # mixed cells (numbers and text with non-breaking spaces) -> text, dates parsed below
        df[c] = df[c].map(lambda v: v if v is None else str(v).replace("\xa0", " ").strip())
    for c in [c for c in df.columns if "date" in c.lower()]:
        num = pd.to_numeric(df[c], errors="coerce")
        df[c] = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    df.to_parquet(root(settings) / "bls_stoppages.parquet", index=False)
    return {"rows": int(len(df)), "columns": list(df.columns)[:12]}


# ================================================================================================ US analyst data


def fetch_earnings(settings, cache) -> dict:
    """Yahoo earnings dates (consensus EPS estimate at the time, reported EPS, surprise) of every PIT S&P 500 ticker."""
    import yfinance as yf

    d = root(settings) / "earnings"
    d.mkdir(exist_ok=True)
    tick = sorted(v16.universe("US", settings, cache)["ticker"].astype(str).str.upper().str.replace(".", "-").unique())
    got, empty = 0, 0
    for i, t in enumerate(tick, 1):
        p = d / f"{t}.parquet"
        if p.exists():
            got += 1
            continue
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=100)
        except Exception:  # noqa: BLE001 - delisted / renamed tickers have no page
            e = None
        if e is None or e.empty:
            empty += 1
            pd.DataFrame(columns=["date", "eps_est", "eps_rep", "surprise_pct"]).to_parquet(p, index=False)
        else:
            e = e.reset_index()
            e.columns = ["date", "eps_est", "eps_rep", "surprise_pct"][: len(e.columns)]
            e["date"] = pd.to_datetime(e["date"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
            e.to_parquet(p, index=False)
            got += 1
        time.sleep(0.4)
        if i % 100 == 0:
            say(f"Yahoo earnings {i}/{len(tick)}")
    return {"tickers": len(tick), "with_data": got, "empty": empty}


def estimates_ledger(settings, cache) -> dict:
    """Daily snapshot of the current S&P 500 members' EPS trend, revisions and price targets."""
    import yfinance as yf

    from quantcio.data.pit import load_pit_universe
    from quantcio.modeling.dataset import universe_params
    from quantcio.models import Market

    n, start = universe_params(Market.US, settings)
    m = load_pit_universe(Market.US, settings, cache, n, start).membership
    tick = sorted(m.loc[m["date"] == m["date"].max(), "ticker"].astype(str).str.upper().str.replace(".", "-").unique())
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    p = EST_LEDGER / f"{today:%Y-%m}.parquet"
    have = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    done = set(have.loc[pd.to_datetime(have["date"]) == today, "ticker"]) if len(have) else set()
    rows, fails = [], 0
    for t in tick:
        if t in done:
            continue
        try:
            tk = yf.Ticker(t)
            tr, rv, pt = tk.eps_trend, tk.eps_revisions, tk.analyst_price_targets or {}
        except Exception:  # noqa: BLE001 - Yahoo throttling / missing modules
            fails += 1
            time.sleep(2.0)
            continue
        rec = {"date": today, "ticker": t}
        for per in ("0q", "+1q", "0y", "+1y"):
            if tr is not None and per in getattr(tr, "index", []):
                for c in ("current", "7daysAgo", "30daysAgo", "60daysAgo", "90daysAgo"):
                    rec[f"eps_{per}_{c}"] = float(tr.loc[per, c]) if c in tr.columns else np.nan
            if rv is not None and per in getattr(rv, "index", []):
                for c in rv.columns:
                    if c != "currency":
                        rec[f"rev_{per}_{c}"] = float(rv.loc[per, c])
        for c in ("current", "high", "low", "mean", "median"):
            rec[f"pt_{c}"] = float(pt.get(c)) if pt.get(c) is not None else np.nan
        rows.append(rec)
        time.sleep(0.25)
    if rows:
        EST_LEDGER.mkdir(parents=True, exist_ok=True)
        new = pd.DataFrame(rows)
        allr = pd.concat([have, new], ignore_index=True) if len(have) else new
        allr.drop_duplicates(["date", "ticker"], keep="last").sort_values(["date", "ticker"]).to_parquet(p, index=False)
    return {"tickers": len(tick), "recorded": len(rows), "failed": fails, "date": str(today.date())}


# ================================================================================================ OpenDART extra


def fetch_dart_extra(settings, cache, first_year: int = 2015) -> dict:
    """Annual-report items (DART_EXTRA) per code and year; stops cleanly at the daily quota (resumable)."""
    k = kd.key("DART_API_KEY")
    if not k:
        return {"skipped": "DART_API_KEY 없음"}
    d = root(settings) / "dart_extra"
    d.mkdir(exist_ok=True)
    cc = v16.kr_corp_codes(settings)
    sess = requests.Session()
    last = pd.Timestamp.today().year - 1
    calls = 0
    for i, code in enumerate(_kr_codes(settings, cache), 1):
        corp = cc.get(code)
        if not corp:
            continue
        p = d / f"{code}.json"
        have = json.loads(p.read_text()) if p.exists() else {}
        try:
            for api in DART_EXTRA:
                got = have.setdefault(api, {})
                for y in range(first_year, last + 1):
                    if str(y) in got and (got[str(y)] or y < last):
                        continue
                    js = v16._dart_get(sess, k, api, {"corp_code": corp, "bsns_year": str(y), "reprt_code": "11011"})
                    got[str(y)] = js.get("list", []) if js.get("status") == "000" else []
                    calls += 1
                    time.sleep(0.1)
        except RuntimeError as exc:  # daily quota (020): keep what we have, the next run resumes
            p.write_text(json.dumps(have, ensure_ascii=False))
            say(f"DART extra stopped at {code}: {exc}")
            return {"calls": calls, "stopped_at": code}
        p.write_text(json.dumps(have, ensure_ascii=False))
        if i % 50 == 0:
            say(f"DART extra {i} codes, {calls} calls")
    return {"calls": calls, "complete": True}


def main() -> None:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--what", default="all", choices=["all", "kr_halts", "kosis", "bls", "earnings", "dart_extra"])
    sub.add_parser("ledger")
    a = ap.parse_args()
    s = load_settings()
    cache = DiskCache(s.cache_path)
    if a.cmd == "ledger":
        say(f"US estimates ledger: {estimates_ledger(s, cache)}")
        return
    w = a.what
    if w in ("all", "kosis"):
        say(f"KOSIS labour disputes: {fetch_kosis(s)}")
    if w in ("all", "bls"):
        say(f"BLS work stoppages: {fetch_bls(s)}")
    if w in ("all", "earnings"):
        say(f"Yahoo earnings dates: {fetch_earnings(s, cache)}")
    if w in ("all", "dart_extra"):
        say(f"DART extra: {fetch_dart_extra(s, cache)}")
    if w in ("all", "kr_halts"):
        say(f"KR halts: {fetch_kr_halts(s, cache)}")


if __name__ == "__main__":
    main()
