"""Data that needs a (free) API key: fetchers + point-in-time features (docs/MODELING.md section 23, docs/ALTDATA.md
section 6). Research code outside ``src/quantcio``. Keys are read from ``.env`` only and never printed; a source whose
key is missing is skipped. ``scripts/api_keys.py`` checks the keys and runs the fetchers.

US: USPTO (patents: PatentsView tables on the USPTO Open Data Portal), FRED (macro, first-release values),
BLS (JOLTS v2), EIA (natural-gas storage), Finnhub (analyst recommendations / news counts - forward ledger).
KR: ECOS (Bank of Korea macro + industry business-survey outlook), KOSIS (사업체노동력조사 industry hires /
separations), KIPRIS Plus (Korean patent registrations per applicant), Naver search (news counts - forward ledger).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import zipfile
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "Mozilla/5.0 (research; quantcio)"}
Say = Callable[[str], None]


def _nosay(_m: str) -> None:
    return None


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT / ".env")
    except ImportError:  # pragma: no cover - python-dotenv is a project dependency
        pass


def key(name: str) -> str:
    load_env()
    return (os.environ.get(name) or "").strip()


def root(settings, name: str) -> Path:
    p = Path(settings.cache_path) / "alt" / "keyed" / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def ledger(name: str) -> Path:
    p = PROJECT / "alt_ledger"
    p.mkdir(exist_ok=True)
    return p / name


def save_news_text(kind: str, recs: list[dict]) -> None:
    """Headline text the daily ledgers already fetch, kept for the NLP study (docs/MODELING.md 64) in the git-ignored
    cache (<cache>/alt/keyed/v25/ledger/<kind>_<date>.jsonl.gz) - the committed ledgers stay counts only."""
    if not recs:
        return
    import gzip

    from quantcio.config import load_settings

    d = root(load_settings(), "v25") / "ledger"
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / f"{kind}_{date.today()}.jsonl.gz", "at") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _redact(msg: str, *secrets: str) -> str:
    for s in secrets:
        if s:
            msg = msg.replace(s, "***")
    return msg


def _daily_available(values: pd.Series, available_from: pd.Series, max_age_days: int) -> pd.Series:
    """Daily series: on day d the latest value whose available_from <= d (at most max_age_days old)."""
    df = pd.DataFrame({"v": values.to_numpy(float), "a": pd.to_datetime(available_from).to_numpy()}).dropna()
    df = df.sort_values("a").drop_duplicates("a", keep="last")
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(df["v"].to_numpy(), index=pd.DatetimeIndex(df["a"]))
    end = max(pd.Timestamp.today().normalize(), s.index.max())
    return s.reindex(pd.date_range(s.index.min(), end, freq="D")).ffill(limit=max_age_days)


def _z_past(s: pd.Series, window: int, minp: int) -> pd.Series:
    m, sd = s.rolling(window, min_periods=minp).mean(), s.rolling(window, min_periods=minp).std()
    return (s - m) / sd.where(sd > 0)


def _asof(series: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    if series is None or not len(series):
        return np.full(len(dates), np.nan)
    return series.reindex(dates, method="ffill").to_numpy(float)


# ================================================================================================ US: USPTO patents

ODP = "https://api.uspto.gov/api/v1/datasets/products"
PV_FILES = ("g_patent.tsv.zip", "g_assignee_disambiguated.tsv.zip")


def _walk(obj, key_name: str, out: list) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key_name:
                out.append(obj if key_name == "fileName" else v)
            if isinstance(v, dict | list):
                _walk(v, key_name, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, key_name, out)


def uspto_check(k: str) -> str:
    r = requests.get(f"{ODP}/search", params={"q": "PatentsView"}, headers={**UA, "x-api-key": k}, timeout=60)
    return "ok" if r.status_code == 200 else f"HTTP {r.status_code}"


def _read_tsv_zip(p: Path, cols: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(p) as z:
        name = next(n for n in z.namelist() if n.endswith(".tsv"))
        with z.open(name) as fh:
            return pd.read_csv(fh, sep="\t", usecols=cols, dtype=str, quoting=3)


def uspto_fetch(settings, say: Say = _nosay) -> dict:
    """Download the PatentsView granted-patent and assignee tables once (ODP: 20 downloads of a file per key per
    year) and compact them to (patent_id, date, org) parquet; date = grant date (public that day)."""
    k = key("USPTO_API_KEY")
    if not k:
        return {"skipped": "USPTO_API_KEY 없음"}
    d = root(settings, "uspto")
    out = d / "patents.parquet"
    if out.exists():
        return {"cached": str(out)}
    h = {**UA, "x-api-key": k}
    r = requests.get(f"{ODP}/search", params={"q": "PatentsView"}, headers=h, timeout=120)
    r.raise_for_status()
    files: list[dict] = []
    ids: list[str] = []
    _walk(r.json(), "fileName", files)
    _walk(r.json(), "productIdentifier", ids)
    for pid in sorted({i for i in ids if isinstance(i, str)}):  # product metadata lists the files
        m = requests.get(f"{ODP}/{pid}", headers=h, timeout=120)
        if m.status_code == 200:
            _walk(m.json(), "fileName", files)
    want: dict[str, dict] = {}
    for f in files:
        n = str(f.get("fileName") or "")
        if n in PV_FILES and n not in want:
            want[n] = f
    missing = [n for n in PV_FILES if n not in want]
    if missing:
        raise RuntimeError(f"USPTO ODP: PatentsView 파일을 찾지 못했습니다 {missing} (상품 {sorted(set(ids))[:8]})")
    for n, f in want.items():
        p = d / n
        if p.exists():
            continue
        url = f.get("fileDownloadURI") or f"{ODP}/files/{f.get('productIdentifier', '')}/{n}"
        say(f"USPTO 다운로드 {n}")
        with requests.get(url, headers=h, timeout=7200, stream=True) as s:
            s.raise_for_status()
            tmp = p.with_suffix(".part")
            with open(tmp, "wb") as fh:
                for chunk in s.iter_content(1 << 20):
                    fh.write(chunk)
            tmp.replace(p)
    say("USPTO 표 압축(특허번호·등록일·출원인 기관)")
    from us_v8_data import norm_name

    pat = _read_tsv_zip(d / "g_patent.tsv.zip", ["patent_id", "patent_date"])
    pat = pat[pat["patent_date"] >= "2005-01-01"]
    asg = _read_tsv_zip(d / "g_assignee_disambiguated.tsv.zip", ["patent_id", "disambig_assignee_organization"])
    m = pat.merge(asg.dropna(subset=["disambig_assignee_organization"]), on="patent_id", how="inner")
    m["org"] = m["disambig_assignee_organization"].map(norm_name)
    m["date"] = pd.to_datetime(m["patent_date"], errors="coerce")
    m[["patent_id", "date", "org"]].dropna().to_parquet(out, index=False)
    return {"rows": int(len(m)), "file": str(out)}


def uspto_monthly_by_company(settings, companies: dict[int, list[str]]) -> pd.DataFrame:
    """(cik, month, grants): monthly patent grants of each CIK, matched on normalised assignee names (exact, or a
    longer name that starts with the company's name - same rule as the H-1B match)."""
    from us_v8_data import GENERIC_NAMES, norm_name

    p = root(settings, "uspto") / "patents.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["cik", "month", "grants"])
    pt = pd.read_parquet(p)
    pt["month"] = pt["date"].dt.to_period("M").dt.to_timestamp()
    by_org = pt.groupby(["org", "month"]).size().rename("n").reset_index()
    orgs = {o: g.set_index("month")["n"] for o, g in by_org.groupby("org")}
    by_first: dict[str, list[str]] = {}
    for o in orgs:
        if o:
            by_first.setdefault(o.split()[0], []).append(o)
    rows = []
    for cik, names in companies.items():
        hit: set[str] = set()
        for n in {norm_name(x) for x in names if x} - {""}:
            if n in orgs:
                hit.add(n)
            toks = n.split()
            if len(toks) >= 2 or (len(n) >= 6 and n not in GENERIC_NAMES):
                hit.update(o for o in by_first.get(toks[0], []) if o.startswith(n + " "))
        if hit:
            s = pd.concat([orgs[o] for o in hit]).groupby(level=0).sum()
            rows.append(pd.DataFrame({"cik": cik, "month": s.index, "grants": s.to_numpy()}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["cik", "month", "grants"])


# ================================================================================================ US: FRED

FRED = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES = {  # key -> (series id, "first" = first-release values (ALFRED) | "daily" = market data, not revised)
    "icsa": ("ICSA", "first"),
    "nfci": ("NFCI", "first"),
    "umcsent": ("UMCSENT", "first"),
    "dgs2": ("DGS2", "daily"),
    "t10y2y": ("T10Y2Y", "daily"),
}


def fred_check(k: str) -> str:
    r = requests.get(FRED, params={"series_id": "DGS2", "api_key": k, "file_type": "json", "limit": 1}, timeout=60)
    return "ok" if r.status_code == 200 else f"HTTP {r.status_code}"


def fred_fetch(settings, say: Say = _nosay) -> dict:
    k = key("FRED_API_KEY")
    if not k:
        return {"skipped": "FRED_API_KEY 없음"}
    d = root(settings, "fred")
    stats = {}
    for name, (sid, kind) in FRED_SERIES.items():
        params = {"series_id": sid, "api_key": k, "file_type": "json", "observation_start": "2005-01-01"}
        if kind == "first":
            params.update({"output_type": 4, "realtime_start": "1776-07-04", "realtime_end": "9999-12-31"})
        r = requests.get(FRED, params=params, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(_redact(f"FRED {sid}: HTTP {r.status_code} {r.text[:200]}", k))
        obs = pd.DataFrame(r.json().get("observations", []))
        obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
        obs["date"] = pd.to_datetime(obs["date"])
        if kind == "first":
            obs["available"] = pd.to_datetime(obs["realtime_start"]) + pd.Timedelta(1, "D")
        else:
            obs["available"] = obs["date"] + pd.Timedelta(1, "D")
        obs[["date", "value", "available"]].dropna().to_parquet(d / f"{name}.parquet", index=False)
        stats[name] = int(obs["value"].notna().sum())
        time.sleep(0.6)
    say(f"FRED {stats}")
    return stats


def fred_series(settings, name: str) -> pd.DataFrame | None:
    p = root(settings, "fred") / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ================================================================================================ US: BLS (JOLTS v2)

BLS_V2 = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


def bls_check(k: str) -> str:
    body = {"seriesid": ["JTS000000000000000HIR"], "startyear": "2024", "endyear": "2024", "registrationkey": k}
    js = requests.post(BLS_V2, json=body, timeout=60).json()
    return "ok" if js.get("status") == "REQUEST_SUCCEEDED" else str(js.get("message"))[:120]


def bls_fetch(settings, say: Say = _nosay) -> dict:
    """JOLTS hires / quits / layoffs / openings rates of the v8 industries in one v2 call per 20 years; written to the
    v8 cache (scripts/us_v8_data.py reads it)."""
    k = key("BLS_API_KEY")
    if not k:
        return {"skipped": "BLS_API_KEY 없음"}
    from us_v8_data import JOLTS_INDUSTRY
    from us_v8_data import root_dir as v8_root

    ids = [f"JTS{i}000000000{e}R" for i in sorted(set(JOLTS_INDUSTRY.values())) for e in ("HI", "QU", "LD", "JO")]
    raw = []
    for k0 in range(0, len(ids), 50):
        body = {
            "seriesid": ids[k0 : k0 + 50],
            "startyear": "2007",
            "endyear": str(date.today().year),
            "registrationkey": k,
        }
        js = requests.post(BLS_V2, json=body, timeout=180).json()
        if js.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS v2: {js.get('status')} {js.get('message')}")
        raw += js["Results"]["series"]
    (v8_root(settings) / "jolts.json").write_text(json.dumps(raw))
    say(f"BLS JOLTS {len(raw)} series")
    return {"series": len(raw)}


# ================================================================================================ US: EIA

EIA = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"


def eia_check(k: str) -> str:
    r = requests.get(EIA, params={"api_key": k, "frequency": "weekly", "data[0]": "value", "length": 1}, timeout=60)
    return "ok" if r.status_code == 200 else f"HTTP {r.status_code}"


def eia_fetch(settings, say: Say = _nosay) -> dict:
    """Lower-48 working gas in storage (weekly, Bcf); released on the Thursday after the week ending Friday."""
    k = key("EIA_API_KEY")
    if not k:
        return {"skipped": "EIA_API_KEY 없음"}
    params = {
        "api_key": k,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": "NW2_EPG0_SWO_R48_BCF",
        "start": "2005-01-01",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    r = requests.get(EIA, params=params, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(_redact(f"EIA: HTTP {r.status_code} {r.text[:200]}", k))
    df = pd.DataFrame(r.json()["response"]["data"])
    df["date"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["available"] = df["date"] + pd.Timedelta(7, "D")
    df[["date", "value", "available"]].dropna().to_parquet(root(settings, "eia") / "ng_storage.parquet", index=False)
    say(f"EIA 천연가스 저장량 {len(df)}주")
    return {"weeks": int(len(df))}


def eia_surprise(settings) -> pd.Series:
    """Daily (available) z of the weekly storage change minus the mean change of the same week in the past 5 years."""
    p = root(settings, "eia") / "ng_storage.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(p).sort_values("date").reset_index(drop=True)
    df["chg"] = df["value"].diff()
    df["woy"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year
    norm = []
    for r in df.itertuples():
        past = df[(df["woy"] == r.woy) & (df["year"] < r.year) & (df["year"] >= r.year - 5)]["chg"]
        norm.append(past.mean() if len(past) >= 3 else np.nan)
    df["surprise"] = df["chg"] - np.array(norm)
    df["z"] = _z_past(df["surprise"], 156, 52)
    return _daily_available(df["z"], df["available"], 14)


# ================================================================================================ US: Finnhub ledger

FINNHUB = "https://finnhub.io/api/v1"


def finnhub_check(k: str) -> str:
    r = requests.get(
        f"{FINNHUB}/stock/recommendation", params={"symbol": "AAPL"}, headers={"X-Finnhub-Token": k}, timeout=60
    )
    return "ok" if r.status_code == 200 else f"HTTP {r.status_code}"


def finnhub_ledger(settings, tickers: list[str], say: Say = _nosay) -> dict:
    """Forward ledger (alt_ledger/us_finnhub.jsonl): today's analyst recommendation counts and 7-day news count."""
    k = key("FINNHUB_API_KEY")
    if not k:
        return {"skipped": "FINNHUB_API_KEY 없음"}
    today = date.today()
    out = ledger("us_finnhub.jsonl")
    done = set()
    if out.exists():
        for ln in out.read_text().splitlines():
            r = json.loads(ln)
            if r.get("date") == str(today):
                done.add(r["ticker"])
    n = 0
    with open(out, "a") as fh:
        for t in tickers:
            if t in done:
                continue
            h = {"X-Finnhub-Token": k}  # header, so the key never appears in a URL / error message
            try:
                rec = requests.get(f"{FINNHUB}/stock/recommendation", params={"symbol": t}, headers=h, timeout=60)
                news = requests.get(
                    f"{FINNHUB}/company-news",
                    params={"symbol": t, "from": str(today - timedelta(7)), "to": str(today)},
                    headers=h,
                    timeout=60,
                )
            except requests.RequestException as exc:  # network blip: skip this ticker, keep going
                say(_redact(f"Finnhub {t}: {type(exc).__name__}", k))
                time.sleep(5)
                continue
            row = {
                "date": str(today),
                "ticker": t,
                "recommendation": rec.json() if rec.status_code == 200 else None,
                "news_7d": len(news.json()) if news.status_code == 200 else None,
            }
            if news.status_code == 200 and isinstance(news.json(), list):
                save_news_text(
                    "us",
                    [
                        {
                            "ticker": t,
                            "id": x.get("id"),
                            "datetime": int(x.get("datetime") or 0),
                            "headline": str(x.get("headline") or "")[:400],
                            "summary": str(x.get("summary") or "")[:400],
                            "source": str(x.get("source") or ""),
                        }
                        for x in news.json()
                    ],
                )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            time.sleep(2.1)  # free tier: 60 calls / minute
    say(f"Finnhub 기록 {n}종목")
    return {"recorded": n}


# ================================================================================================ KR: ECOS

ECOS = "https://ecos.bok.or.kr/api"
ECOS_SERIES = {  # key -> (stat code, item code or a name to look for, available months after the reference month)
    "base_rate": ("722Y001", "0101000", 1),
    "cpi": ("901Y009", "0", 1),
    "esi": ("513Y001", "E1000", 1),
    "ip": ("901Y033", "A00", 2),
    "csi": ("511Y002", "소비자심리지수", 1),
    "bsi_outlook": ("512Y014", None, 1),  # every industry
}
KR_BSI_INDUSTRY = {  # KR sector group -> ECOS BSI industry name (spaces / dots ignored); fallback 전산업
    "반도체": "전자영상통신장비",
    "IT·전자": "전자영상통신장비",
    "자동차": "자동차",
    "화학": "화학물질",
    "철강·소재": "1차금속",
    "2차전지·전기장비": "전기장비",
    "제약·바이오": "의료",
    "건설": "건설",
    "운송": "운수창고",
    "커뮤니케이션": "정보통신",
    "경기소비재": "도소매",
    "필수소비재": "도소매",
    "산업재": "기타기계장비",
}


def ecos_check(k: str) -> str:
    js = requests.get(f"{ECOS}/StatisticSearch/{k}/json/kr/1/1/722Y001/M/202401/202401/0101000", timeout=60).json()
    return "ok" if "StatisticSearch" in js else str(js.get("RESULT", js))[:120]


def _ecos_rows(k: str, code: str, item: str | None, start: str, end: str) -> list[dict]:
    rows, s = [], 1
    while True:
        tail = f"/{item}" if item else ""
        url = f"{ECOS}/StatisticSearch/{k}/json/kr/{s}/{s + 999}/{code}/M/{start}/{end}{tail}"
        js = requests.get(url, timeout=120).json()
        if "StatisticSearch" not in js:
            res = js.get("RESULT") or {}
            if str(res.get("CODE")) == "INFO-200" and not rows:  # no data in the period
                return rows
            raise RuntimeError(_redact(f"ECOS {code}: {res.get('CODE')} {res.get('MESSAGE')}", k))
        part = (js.get("StatisticSearch") or {}).get("row") or []
        rows += part
        total = int((js.get("StatisticSearch") or {}).get("list_total_count") or 0)
        s += 1000
        if not part or s > total:
            return rows
        time.sleep(0.3)


def ecos_fetch(settings, say: Say = _nosay) -> dict:
    k = key("ECOS_API_KEY")
    if not k:
        return {"skipped": "ECOS_API_KEY 없음"}
    d = root(settings, "ecos")
    end = date.today().strftime("%Y%m")
    stats = {}
    for name, (code, item, _lag) in ECOS_SERIES.items():
        exact = item if item and re.fullmatch(r"[A-Z0-9]+", item) else None
        rows = _ecos_rows(k, code, exact, "200501", end)
        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError(f"ECOS {code}: 빈 응답")
        if item and not exact:  # pick by item name
            df = df[df["ITEM_NAME1"].str.replace(" ", "").str.contains(item.replace(" ", ""), na=False)]
        df.to_parquet(d / f"{name}.parquet", index=False)
        stats[name] = int(len(df))
        time.sleep(0.3)
    say(f"ECOS {stats}")
    return stats


def _ecos_monthly(settings, name: str) -> pd.DataFrame | None:
    p = root(settings, "ecos") / f"{name}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["month"] = pd.to_datetime(df["TIME"], format="%Y%m")
    df["value"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    return df


def ecos_regime(settings) -> dict[str, pd.Series]:
    """Daily PIT KR regime series: value of month m from the 1st of month m + lag."""
    out = {}
    for name, (_c, _i, lag) in ECOS_SERIES.items():
        df = _ecos_monthly(settings, name)
        if df is None or name == "bsi_outlook":
            continue
        s = df.groupby("month")["value"].mean().sort_index()
        if name in ("cpi", "ip"):
            s = np.log(s).diff(12)
        elif name == "base_rate":
            s = s.diff(12)
        else:
            s = _z_past(s, 36, 24)
        avail = s.index + pd.offsets.MonthBegin(lag)
        out[f"r_kr_{name}"] = _daily_available(s, pd.Series(avail), 124)
    bsi = _ecos_monthly(settings, "bsi_outlook")
    if bsi is not None:
        allind = bsi[bsi["ITEM_CODE1"] == "99988"]
        if "ITEM_NAME2" in allind and allind["ITEM_NAME2"].notna().any():
            allind = allind[allind["ITEM_NAME2"].str.contains("업황", na=False)]
        s = _z_past(allind.groupby("month")["value"].mean().sort_index(), 36, 24)
        out["r_kr_bsi"] = _daily_available(s, pd.Series(s.index + pd.offsets.MonthBegin(1)), 124)
    return out


def ecos_sector_bsi(settings) -> dict[str, pd.Series]:
    """KR sector group -> daily PIT z (vs 36 months) of its industry's BSI outlook (fallback: all industries)."""
    bsi = _ecos_monthly(settings, "bsi_outlook")
    if bsi is None:
        return {}
    if "ITEM_NAME2" in bsi and bsi["ITEM_NAME2"].notna().any():
        bsi = bsi[bsi["ITEM_NAME2"].str.contains("업황", na=False)]
    bsi["ind"] = bsi["ITEM_NAME1"].str.replace(r"[\s·.,]", "", regex=True)
    by_ind = {i: g.groupby("month")["value"].mean().sort_index() for i, g in bsi.groupby("ind")}
    total = by_ind.get("전산업")
    out = {}
    for sec in [*KR_BSI_INDUSTRY, "금융", "기타"]:
        want = KR_BSI_INDUSTRY.get(sec)
        s = next((v for i, v in by_ind.items() if want and want in i), None) if want else None
        s = total if s is None else s
        if s is None:
            continue
        z = _z_past(s, 36, 24)
        out[sec] = _daily_available(z, pd.Series(z.index + pd.offsets.MonthBegin(1)), 124)
    return out


# ================================================================================================ KR: KOSIS

KOSIS = "https://kosis.kr/openapi"
KOSIS_TABLES = (  # 사업체노동력조사 산업/규모별 고용 (고용노동부, orgId 118): one table per KSIC period
    ("DT_118N_MON036", 2010, 2017),
    ("DT_118N_MON056", 2018, 2023),
    ("DT_118N_MON066", 2024, 2100),
)
KOSIS_ITEMS = {"16118MP_4": "hire", "16118MP_13": "sep"}  # 입직률_전체, 이직률_전체
KOSIS_SIZE_ALL = "13102732820SIZES.0"  # 규모 전체
KOSIS_INDUSTRY = {  # KR sector group -> KSIC industry name fragment (spaces / punctuation removed); 'section' = 대분류
    "반도체": "전자부품컴퓨터영상음향및통신장비제조업",
    "IT·전자": "전자부품컴퓨터영상음향및통신장비제조업",
    "자동차": "자동차및트레일러제조업",
    "화학": "화학물질및화학제품제조업",
    "제약·바이오": "의료용물질및의약품제조업",
    "철강·소재": "1차금속제조업",
    "2차전지·전기장비": "전기장비제조업",
    "산업재": "기타기계및장비제조업",
    "필수소비재": "식료품제조업",
    "건설": "F건설업",
    "운송": "H운수",
    "커뮤니케이션": "J",  # J.정보통신업 (10th KSIC) / J.출판영상방송통신및정보서비스업 (9th)
    "금융": "K금융",
    "경기소비재": "G도매및소매업",
    "기타": "전체",
}


def kosis_check(k: str) -> str:
    r = requests.get(
        f"{KOSIS}/statisticsSearch.do",
        params={"method": "getList", "apiKey": k, "searchNm": "입직", "format": "json", "jsonVD": "Y"},
        timeout=60,
    )
    t = r.text
    return "ok" if r.status_code == 200 and "err" not in t[:40] else t[:120]


def _kosis_json(r: requests.Response):
    t = r.text.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:  # KOSIS sometimes answers unquoted keys: {err:"11",...}
        return json.loads(re.sub(r"([{,])\s*([A-Za-z_]+)\s*:", r'\1"\2":', t))


def kosis_fetch(settings, say: Say = _nosay) -> dict:
    """입직률 / 이직률 (all sizes) per industry and month from the three 사업체노동력조사 tables (2010 ..)."""
    k = key("KOSIS_API_KEY")
    if not k:
        return {"skipped": "KOSIS_API_KEY 없음"}
    d = root(settings, "kosis")
    rows = []
    for tbl, y0, y1 in KOSIS_TABLES:
        for y in range(y0, min(y1, date.today().year) + 1):
            params = {
                "method": "getList",
                "apiKey": k,
                "orgId": "118",
                "tblId": tbl,
                "itmId": "+".join(KOSIS_ITEMS) + "+",
                "objL1": "ALL",
                "objL2": KOSIS_SIZE_ALL,
                "format": "json",
                "jsonVD": "Y",
                "prdSe": "M",
                "startPrdDe": f"{y}01",
                "endPrdDe": f"{y}12",
            }
            js = _kosis_json(requests.get(f"{KOSIS}/Param/statisticsParameterData.do", params=params, timeout=180))
            if isinstance(js, dict):
                if str(js.get("err")) == "30":  # no data (e.g. months not yet published)
                    continue
                raise RuntimeError(f"KOSIS {tbl} {y}: {js}")
            rows += [{**r, "TABLE": tbl} for r in js]
            time.sleep(0.3)
        say(f"KOSIS {tbl}: {len(rows)} rows so far")
    df = pd.DataFrame(rows)
    df.to_parquet(d / "flows.parquet", index=False)
    return {"rows": int(len(df)), "months": int(df["PRD_DE"].nunique()) if len(df) else 0}


def _norm_ind(s: str) -> str:
    return re.sub(r"[\s·ㆍ.,;()~0-9]", "", str(s))


def kosis_sector_flows(settings) -> dict[str, dict[str, pd.Series]]:
    """KR sector group -> {"hire": series, "sep": series}: rate 3-month mean - 24-month mean, month m usable from the
    1st of month m + 2 (사업체노동력조사 is published about a month after the reference month)."""
    p = root(settings, "kosis") / "flows.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    df["month"] = pd.to_datetime(df["PRD_DE"].astype(str), format="%Y%m", errors="coerce")
    df["value"] = pd.to_numeric(df["DT"], errors="coerce")
    df["kind"] = df["ITM_ID"].map(KOSIS_ITEMS)
    df["ind"] = df["C1_NM"].map(_norm_ind)
    out: dict[str, dict[str, pd.Series]] = {}
    for sec, frag in KOSIS_INDUSTRY.items():
        f = _norm_ind(frag)
        if frag == "J":
            m = df["C1_NM"].astype(str).str.match(r"\s*J\s*\.")
        elif frag == "전체":
            m = df["ind"] == "전체"
        else:
            m = (
                df["ind"].str.startswith(f)
                if f[:1].isalpha() and f[:1].isascii()
                else df["ind"].str.contains(f, regex=False)
            )
        out[sec] = {}
        for kind in ("hire", "sep"):
            s = df[m & (df["kind"] == kind)].groupby("month")["value"].mean().sort_index()
            if len(s) < 30:
                continue
            x = (s.rolling(3, min_periods=3).mean() - s.rolling(24, min_periods=18).mean()).dropna()
            out[sec][kind] = _daily_available(x, pd.Series(x.index + pd.offsets.MonthBegin(2)), 93)
    return out


# ================================================================================================ KR: KIPRIS Plus

KIPRIS = "http://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getAdvancedSearch"
KIPRIS_YEARS = range(2012, date.today().year)


def _kipris_count(k: str, applicant: str, year: int) -> int | None:
    params = {
        "applicant": applicant,
        "registerDate": f"{year}0101~{year}1231",
        "patent": "true",
        "utility": "false",
        "numOfRows": 1,
        "pageNo": 1,
        "ServiceKey": k,
    }
    r = requests.get(KIPRIS, params=params, timeout=60)
    x = ElementTree.fromstring(r.content)
    code = (x.findtext(".//resultCode") or "").strip()
    if code not in ("", "00"):
        raise RuntimeError(f"KIPRIS {code} {x.findtext('.//resultMsg')}")
    tc = x.findtext(".//totalCount")
    return int(tc) if tc and tc.strip().isdigit() else None


def kipris_check(k: str) -> str:
    try:
        n = _kipris_count(k, "삼성전자", 2020)
        return "ok" if n is not None else "응답에 totalCount 없음"
    except Exception as exc:  # noqa: BLE001 - shown to the user
        return _redact(str(exc), k)[:120]


def kipris_name(name: str) -> str | None:
    n = re.sub(r"\(주\)|㈜|주식회사|\s+", "", str(name or ""))
    return n if len(n) >= 3 else None  # 2-letter names (e.g. "LG") would match every affiliate


def kipris_fetch(settings, companies: dict[str, str], say: Say = _nosay, max_calls: int = 5000) -> dict:
    """Patent registrations per applicant name and calendar year (cached; resumable; stops after max_calls)."""
    k = key("KIPRIS_API_KEY")
    if not k:
        return {"skipped": "KIPRIS_API_KEY 없음"}
    p = root(settings, "kipris") / "counts.json"
    have = json.loads(p.read_text()) if p.exists() else {}
    calls = 0
    for ent, name in companies.items():
        q = kipris_name(name)
        if not q:
            continue
        rec = have.setdefault(ent, {"name": q, "years": {}})
        for y in KIPRIS_YEARS:
            if str(y) in rec["years"] or calls >= max_calls:
                continue
            rec["years"][str(y)] = _kipris_count(k, q, y)
            calls += 1
            if calls % 200 == 0:
                p.write_text(json.dumps(have, ensure_ascii=False))
                say(f"KIPRIS {calls}회 호출")
            time.sleep(1.05)  # 60 requests / minute
    p.write_text(json.dumps(have, ensure_ascii=False))
    left = sum(
        1
        for e, n in companies.items()
        if kipris_name(n)
        for y in KIPRIS_YEARS
        if str(y) not in have.get(e, {}).get("years", {})
    )
    return {"calls": calls, "remaining": left}


def kipris_counts(settings) -> dict[str, dict[int, int]]:
    p = root(settings, "kipris") / "counts.json"
    if not p.exists():
        return {}
    return {
        e: {int(y): int(v) for y, v in r["years"].items() if v is not None}
        for e, r in json.loads(p.read_text()).items()
    }


# ================================================================================================ KR: Naver news ledger

NAVER_HUB = "https://naverapihub.apigw.ntruss.com/search/v1/news"  # NAVER Cloud Platform API HUB (2025-)
NAVER = "https://openapi.naver.com/v1/search/news.json"  # former developers.naver.com endpoint


def _naver_get(cid: str, secret: str, params: dict) -> requests.Response:
    """API HUB first (X-NCP-APIGW-* headers), then the former developers.naver.com endpoint."""
    r = requests.get(
        NAVER_HUB,
        params={**params, "format": "json"},
        headers={"X-NCP-APIGW-API-KEY-ID": cid, "X-NCP-APIGW-API-KEY": secret},
        timeout=60,
    )
    if r.status_code == 200:
        return r
    r2 = requests.get(
        NAVER, params=params, headers={"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": secret}, timeout=60
    )
    return r2 if r2.status_code == 200 else r


def naver_check(cid: str, secret: str) -> str:
    r = _naver_get(cid, secret, {"query": "삼성전자", "display": 1})
    return "ok" if r.status_code == 200 else f"HTTP {r.status_code} {r.text[:120]}"


def _plain(s) -> str:
    import html

    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", str(s or "")))).strip()


def naver_ledger(companies: dict[str, str], say: Say = _nosay) -> dict:
    """Forward ledger (alt_ledger/kr_news_naver.jsonl): articles in the last 24 hours per company name (up to 100)."""
    cid, secret = key("NAVER_CLIENT_ID"), key("NAVER_CLIENT_SECRET")
    if not cid or not secret:
        return {"skipped": "NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 없음"}
    from email.utils import parsedate_to_datetime

    today = date.today()
    out = ledger("kr_news_naver.jsonl")
    done = set()
    if out.exists():
        done = {json.loads(ln)["entity"] for ln in out.read_text().splitlines() if f'"date": "{today}"' in ln}
    n = 0
    with open(out, "a") as fh:
        for ent, name in companies.items():
            if ent in done:
                continue
            try:
                r = _naver_get(cid, secret, {"query": name, "display": 100, "sort": "date"})
            except requests.RequestException as exc:
                say(_redact(f"Naver {name}: {type(exc).__name__}", cid, secret))
                continue
            if r.status_code != 200:
                continue
            items = r.json().get("items", [])
            save_news_text(
                "kr",
                [
                    {
                        "entity": ent,
                        "name": name,
                        "published": str(it.get("pubDate")),
                        "title": _plain(it.get("title")),
                        "desc": _plain(it.get("description"))[:400],
                        "link": it.get("originallink") or it.get("link") or "",
                    }
                    for it in items
                ],
            )
            now = pd.Timestamp.now(tz="Asia/Seoul")
            recent = sum(
                1 for it in items if (now - pd.Timestamp(parsedate_to_datetime(it["pubDate"]))) <= pd.Timedelta(1, "D")
            )
            fh.write(
                json.dumps({"date": str(today), "entity": ent, "name": name, "n_24h": recent}, ensure_ascii=False)
                + "\n"
            )
            n += 1
            time.sleep(0.12)
    say(f"네이버 뉴스 기록 {n}종목")
    return {"recorded": n}
