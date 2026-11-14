"""Data for the v17 research runs (docs/MODELING.md 43, pre-declared 2026-10-12): block V - legal risk, labour,
cyber incidents and web traffic. KR: DART filing names (production halts, serious industrial accidents, criminal
punishment for them) and the Tranco rank of the company homepage; US: federal corporate prosecutions (Corporate
Prosecution Registry), California data-breach notices and 8-K Item 1.05 (material cybersecurity incidents).
Research only - nothing here is served.

    .venv/bin/python scripts/v17_data.py fetch
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_data as v13  # noqa: E402
import v16_data as v16  # noqa: E402

say = v13.say
BUA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0"
}
CA_BREACH = "https://oag.ca.gov/privacy/databreach/list-export"
CPR = "https://corporate-prosecution-registry.com/media/corp-crime.csv"
TRANCO_DATE = "https://tranco-list.eu/api/lists/date/{d}"
TRANCO_TOP = 100_000
TRANCO_START = date(2019, 8, 5)

DOCS: dict[str, tuple[str, str]] = {
    "v_halt_12m": ("생산중단 공시 수(12개월, log)", "production halt filings (12m)"),
    "v_accident_24m": ("중대재해·재해발생 공시 수(24개월, log)", "industrial accident filings (24m)"),
    "v_punish_24m": ("중대재해 형사처벌 확인 공시(24개월, log)", "criminal punishment for accidents (24m)"),
    "v_web_level": ("홈페이지 Tranco 순위(−log)", "homepage Tranco rank (-log)"),
    "v_web_4w": ("홈페이지 순위 4주 변화", "homepage rank change (4w)"),
    "v_web_12w": ("홈페이지 순위 12주 변화", "homepage rank change (12w)"),
    "v_crime_24m": ("연방 기업 형사사건 수(24개월, log)", "federal corporate prosecutions (24m)"),
    "v_crime_pay_24m": ("형사사건 납부액(24개월) ÷ 시가총액", "prosecution payments (24m) / market cap"),
    "v_breach_12m": ("캘리포니아 정보유출 신고 수(12개월, log)", "California data-breach notices (12m)"),
    "v_cyber_8k_12m": ("8-K 1.05 사이버 사고 공시 수(12개월, log)", "8-K Item 1.05 cyber incidents (12m)"),
}
KR_V = {
    "v_halt_12m": (re.compile(r"생산중단"), 365),
    "v_accident_24m": (re.compile(r"중대재해발생|재해발생"), 730),
    "v_punish_24m": (re.compile(r"중대재해관련형사처벌"), 730),
}


def root(settings) -> Path:
    return kd.root(settings, "v17")


# ================================================================================================ fetch


def fetch_csv(url: str, path: Path) -> dict:
    r = requests.get(url, headers=BUA, timeout=300)
    r.raise_for_status()
    path.write_bytes(r.content)
    return {"bytes": len(r.content)}


def kr_domains(settings, codes: list[str]) -> dict[str, str]:
    """stock code -> registrable homepage domain (DART company.json hm_url)."""
    cc = v16.kr_corp_codes(settings)
    out = {}
    for code in codes:
        url = str(v16.kr_company(settings, cc.get(code, "")).get("hm_url") or "").strip().lower()
        url = re.sub(r"^[a-z]+://", "", url).split("/")[0].split(":")[0]
        url = re.sub(r"^www\d*\.", "", url)
        parts = [p for p in url.split(".") if p]
        if len(parts) < 2:
            continue
        n = 3 if len(parts) >= 3 and parts[-1] == "kr" and parts[-2] in ("co", "or", "ne", "go", "re", "pe") else 2
        out[code] = ".".join(parts[-n:])
    return out


def fetch_tranco_kr(settings, domains: set[str]) -> dict:
    """Weekly (Monday) Tranco top-100k ranks of the KR homepage domains since 2019-08 (resumable)."""
    p = root(settings) / "tranco_kr.parquet"
    have = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["date", "domain", "rank"])
    done = set(pd.to_datetime(have["date"]).dt.date) if len(have) else set()
    rows = [have] if len(have) else []
    day, n = TRANCO_START, 0
    while day <= date.today() - timedelta(2):
        if day not in done:
            try:
                meta = requests.get(TRANCO_DATE.format(d=day), headers=BUA, timeout=60).json()
                if meta.get("available"):
                    url = meta["download"].rsplit("/", 1)[0] + f"/{TRANCO_TOP}"
                    txt = requests.get(url, headers=BUA, timeout=180).text
                    hit = []
                    for ln in txt.splitlines():
                        rk, _, dom = ln.partition(",")
                        if dom.strip() in domains:
                            hit.append((pd.Timestamp(day), dom.strip(), int(rk)))
                    rows.append(pd.DataFrame(hit or [(pd.Timestamp(day), "", 0)], columns=["date", "domain", "rank"]))
                    n += 1
            except (requests.RequestException, ValueError):
                pass
            time.sleep(0.4)
            if n and n % 50 == 0:
                say(f"Tranco KR {day} ({n} lists)")
                pd.concat(rows, ignore_index=True).to_parquet(p, index=False)
        day += timedelta(7)
    out = pd.concat(rows, ignore_index=True) if rows else have
    out.to_parquet(p, index=False)
    return {"lists": int(pd.to_datetime(out["date"]).nunique()) if len(out) else 0, "new": n}


# ================================================================================================ events


def _match(names: dict[str, set[tuple[str, str]]], ev: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """entity -> events whose normalised name equals a company name or starts with a 2+-word company name."""
    by_norm = {n: g for n, g in ev.groupby("norm") if str(n).strip()}  # empty names (blank company field) skipped
    by_first: dict[str, list[str]] = {}
    for n in by_norm:
        by_first.setdefault(n.split()[0], []).append(n)
    out = {}
    for e, keys in names.items():
        hit = set()
        for _, n in keys:
            if n in by_norm:
                hit.add(n)
            if len(n.split()) >= 2:
                hit.update(x for x in by_first.get(n.split()[0], []) if x.startswith(n + " "))
        if hit:
            out[e] = pd.concat([by_norm[h] for h in hit]).drop_duplicates("eid")
    return out


def ca_breaches(settings) -> pd.DataFrame:
    p = root(settings) / "ca_breach.csv"
    d = pd.read_csv(p, dtype=str)
    d.columns = [re.sub(r"\s+", " ", c).strip() for c in d.columns]
    org = d["Organization Name"].fillna("")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(d["Reported Date"], format="%m/%d/%Y", errors="coerce"),
            "norm": org.map(v16.norm_en),
            "eid": np.arange(len(d)),
        }
    ).dropna(subset=["date"])
    return out[out["norm"].str.len() >= 3]


def prosecutions(settings) -> pd.DataFrame:
    p = root(settings) / "cpr.csv"
    d = pd.read_csv(io.BytesIO(p.read_bytes()), encoding="latin-1", low_memory=False)
    dt = pd.to_datetime(d["DATE"], errors="coerce")
    for c in ("JUDGMENT_DATE", "PLEA_DATE"):
        dt = dt.fillna(pd.to_datetime(d[c], errors="coerce"))
    pay = pd.to_numeric(d["TOTAL_PAYMENT"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
    return pd.DataFrame(
        {
            "date": dt,
            "norm": d["COMPANY"].fillna("").map(v16.norm_en),
            "ticker": d["TICKER"].fillna("").astype(str).str.upper().str.strip(),
            "payment": pay.fillna(0.0),
            "eid": np.arange(len(d)),
        }
    ).dropna(subset=["date"])


def cyber_8k(subs, cik: int) -> list[pd.Timestamp]:
    """Filing dates of the CIK's 8-K filings with Item 1.05."""
    sub = subs.get(int(cik))
    if not sub:
        return []
    blocks = [(sub.get("filings") or {}).get("recent") or {}]
    for f in (sub.get("filings") or {}).get("files", []) or []:
        oldest = min((pd.Timestamp(x) for b in blocks for x in b.get("filingDate", []) or []), default=None)
        if oldest is not None and oldest <= pd.Timestamp("2023-12-01"):
            break
        blocks.append(subs.page(f["name"]))
    out = set()
    for b in blocks:
        forms = b.get("form", []) or []
        items = b.get("items", []) or [""] * len(forms)
        fdates = b.get("filingDate", []) or [""] * len(forms)
        for form, it, fd in zip(forms, items, fdates, strict=False):
            if str(form).startswith("8-K") and "1.05" in str(it) and fd:
                out.add(pd.Timestamp(fd))
    return sorted(out)


# ================================================================================================ blocks


def _put_counts(out, col, ii, ev_dates, dates, days, vals=None) -> np.ndarray:
    w = np.ones(len(ev_dates)) if vals is None else np.asarray(vals, float)
    cnt, s = v16._windowed(np.asarray(ev_dates, dtype="datetime64[ns]"), w, dates, days)
    out.iloc[ii, out.columns.get_loc(col)] = np.log1p(cnt)
    return s


def block_kr(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, dict]:
    cols = [*KR_V, "v_web_level", "v_web_4w", "v_web_12w"]
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    cc = v16.kr_corp_codes(settings)
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    n_ev = {k: 0 for k in KR_V}
    for e in np.unique(ents):
        corp = cc.get(e)
        if not corp:
            continue
        rows = []
        for f in (Path(settings.cache_path) / "alt" / "kr_dart" / "list" / corp).glob("*.json"):
            for r in json.loads(f.read_text()).get("rows", []):
                nm = str(r.get("report_nm") or "").strip()
                if nm.startswith("["):
                    continue
                rows.append((pd.Timestamp(r["rcept_dt"]), re.sub(r"\s+", "", nm), r.get("rcept_no")))
        t = pd.DataFrame(rows, columns=["date", "nm", "no"]).drop_duplicates("no")
        ii = np.where(ents == e)[0]
        for k, (pat, days) in KR_V.items():
            sub = t[t["nm"].str.contains(pat)] if len(t) else t
            n_ev[k] += len(sub)
            _put_counts(out, k, ii, sub["date"].to_numpy() if len(sub) else [], dates[ii], days)
    # web: Tranco ranks of the homepage domains
    dom = kr_domains(settings, sorted(set(ents)))
    p = root(settings) / "tranco_kr.parquet"
    n_web = 0
    if p.exists():
        tr = pd.read_parquet(p)
        all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(tr["date"]).unique()))
        for e in np.unique(ents):
            d = dom.get(e)
            s = tr[tr["domain"] == d].set_index("date")["rank"].astype(float) if d else pd.Series(dtype=float)
            if s.empty:
                continue
            n_web += 1
            s = s.reindex(all_dates).fillna(TRANCO_TOP * 1.5)  # listed at some point: below the top 100k otherwise
            f = pd.DataFrame({"lvl": -np.log(s), "w4": np.log(s.shift(4) / s), "w12": np.log(s.shift(12) / s)})
            f.index = f.index + pd.Timedelta(1, "D")
            ii = np.where(ents == e)[0]
            out.iloc[ii, [out.columns.get_loc(c) for c in ("v_web_level", "v_web_4w", "v_web_12w")]] = v13._asof_rows(
                f, dates[ii], 14
            )
    return out, {"kr_events": n_ev, "kr_domains": len(dom), "kr_web_listed": n_web}


def block_us(frame: pd.DataFrame, ds, settings, cache) -> tuple[pd.DataFrame, dict]:
    from quantcio.data.pit.us_cik import Submissions
    from quantcio.data.pit.us_net import PitNet

    cols = ["v_crime_24m", "v_crime_pay_24m", "v_breach_12m", "v_cyber_8k_12m"]
    out = pd.DataFrame(0.0, index=frame.index, columns=cols)
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    mcap = np.exp(frame["log_mcap"].astype(float)).to_numpy()
    chains = v16.us_chains(settings, cache, pd.DataFrame({"entity": np.unique(ents)}))
    uni = v16.universe("US", settings, cache)
    names = v16.us_names(settings, cache, uni, chains)
    tickers = (
        frame.groupby("entity")["ticker"].agg(lambda s: set(s.astype(str).str.upper())).to_dict()
        if "ticker" in frame
        else {}
    )
    cpr = prosecutions(settings)
    by_name = _match(names, cpr)
    br = _match(names, ca_breaches(settings))
    subs = Submissions(PitNet(settings))
    info = {"crime_entities": 0, "breach_entities": 0, "cyber_entities": 0}
    for e in np.unique(ents):
        ii = np.where(ents == e)[0]
        d = dates[ii]
        cr = cpr[cpr["ticker"].isin(tickers.get(e, set()) - {""})]
        if e in by_name:
            cr = pd.concat([cr, by_name[e]]).drop_duplicates("eid")
        if len(cr):
            info["crime_entities"] += 1
            ev = (cr["date"] + pd.Timedelta(1, "D")).to_numpy()
            pay = _put_counts(out, "v_crime_24m", ii, ev, d, 730, cr["payment"].to_numpy())
            out.iloc[ii, out.columns.get_loc("v_crime_pay_24m")] = pay / mcap[ii]
        b = br.get(e)
        if b is not None and len(b):
            info["breach_entities"] += 1
            _put_counts(out, "v_breach_12m", ii, (b["date"] + pd.Timedelta(1, "D")).to_numpy(), d, 365)
        cy = set()
        for c in chains.get(e, set()):
            try:
                cy.update(cyber_8k(subs, c))
            except Exception:  # noqa: BLE001 - a missing submissions file leaves the entity without 8-K items
                continue
        if cy:
            info["cyber_entities"] += 1
            _put_counts(out, "v_cyber_8k_12m", ii, [x + pd.Timedelta(1, "D") for x in sorted(cy)], d, 365)
    return out, info


def block(name: str, market: str, frame: pd.DataFrame, ds, settings, cache) -> tuple[pd.DataFrame, dict]:
    if name != "V":
        raise ValueError(name)
    return block_kr(frame, settings) if market == "KR" else block_us(frame, ds, settings, cache)


def main() -> None:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--what", default="all", choices=["all", "ca", "cpr", "tranco"])
    a = ap.parse_args()
    s = load_settings()
    w = a.what
    if w in ("all", "ca"):
        say(f"CA breach list: {fetch_csv(CA_BREACH, root(s) / 'ca_breach.csv')}")
    if w in ("all", "cpr"):
        say(f"Corporate Prosecution Registry: {fetch_csv(CPR, root(s) / 'cpr.csv')}")
    if w in ("all", "tranco"):
        kr = v16.universe("KR", s, DiskCache(s.cache_path))
        dom = kr_domains(s, sorted(kr["entity"].astype(str)))
        say(f"KR homepage domains: {len(dom)} of {len(kr)}")
        say(f"Tranco KR: {fetch_tranco_kr(s, set(dom.values()))}")


if __name__ == "__main__":
    main()
