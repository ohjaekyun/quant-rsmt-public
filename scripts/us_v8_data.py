"""US v8 data (docs/MODELING.md section 22): R&D (XBRL), employee counts (10-K text), H-1B initial approvals (USCIS
employer data hub) and BLS JOLTS industry hires / quits / layoffs. Research code outside ``src/quantcio``.

    .venv/bin/python scripts/us_v8_data.py fetch    # downloads (resumable; 10-K text ~1-2 h)
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import us_v4_data as v4  # noqa: E402

from quantcio.data.pit.us_net import PitNet  # noqa: E402

COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
H1B = "https://www.uscis.gov/sites/default/files/document/data/h1b_datahubexport-{y}.csv"
H1B_YEARS = range(2009, 2023)  # FY2023's file is partial (about half the rows): not used
H1B_AVAILABLE_MONTH = 4  # FY y (Oct y-1 .. Sep y) usable from April 1 of y + 1, for one year
BLS_API = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
RD_CONCEPTS = [
    "ResearchAndDevelopmentExpense",
    "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
]
JOLTS_INDUSTRY = {  # GICS sector (Korean label) -> JOLTS industry code (docs/MODELING.md 22.2)
    "에너지": "110099",
    "소재": "300000",
    "산업재": "320000",
    "유틸리티": "480099",
    "금융": "520000",
    "부동산": "530000",
    "헬스케어": "620000",
    "정보기술": "540099",
    "커뮤니케이션": "510000",
    "경기소비재": "440000",
    "필수소비재": "340000",
}
JOLTS_ELEMENTS = {"hires": "HI", "quits": "QU", "layoffs": "LD"}
JOLTS_WINDOWS = [(2009, 2018), (2019, 2026)]
EMP_MAX_LOG_CHANGE = 1.1
FEATURES = [
    "us_rd_sales",
    "us_rd_mcap",
    "us_rd_growth",
    "us_rd_none",
    "us_emp_growth",
    "us_emp_none",
    "us_h1b_init",
    "us_h1b_growth",
    "us_sec_hires",
    "us_sec_quits",
    "us_sec_layoffs",
]
DOCS = {
    "us_rd_sales": ("R&D/매출(최근 회계연도)", "R&D / sales (latest fiscal year)"),
    "us_rd_mcap": ("R&D/시가총액", "R&D / market cap"),
    "us_rd_growth": ("R&D 증가율(전년 대비 log)", "R&D growth (log YoY)"),
    "us_rd_none": ("R&D 보고 없음", "no R&D reported"),
    "us_emp_growth": ("직원 수 증가율(10-K, log)", "employee growth (10-K, log)"),
    "us_emp_none": ("직원 수 추출 실패", "employee count not found"),
    "us_h1b_init": ("H-1B 신규 승인 수(학사 이상 전문직, log)", "H-1B initial approvals (log)"),
    "us_h1b_growth": ("H-1B 신규 승인 증가(log 차이)", "H-1B initial approval growth"),
    "us_sec_hires": ("업종 채용률 변화(JOLTS, 3개월−24개월)", "sector hires-rate change (JOLTS)"),
    "us_sec_quits": ("업종 자발적 퇴사율 변화(JOLTS)", "sector quits-rate change (JOLTS)"),
    "us_sec_layoffs": ("업종 해고율 변화(JOLTS)", "sector layoffs-rate change (JOLTS)"),
}
UA = {"User-Agent": "Mozilla/5.0 (research; quantcio)"}


def root_dir(settings) -> Path:
    p = Path(settings.cache_path) / "alt" / "us_v8"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------------------------------------ R&D (XBRL)


def _trim_rd(j: dict) -> dict:
    g = (j.get("facts") or {}).get("us-gaap") or {}
    out = {}
    for c in [*RD_CONCEPTS, *v4.REVENUE]:
        usd = ((g.get(c) or {}).get("units") or {}).get("USD")
        if usd:
            out[c] = [{k: f.get(k) for k in ("start", "end", "val", "form", "filed")} for f in usd]
    return {"cik": j.get("cik"), "facts": out}


def fetch_rd(net: PitNet, ciks: Iterable[int], progress: Callable[[str], None] | None = None) -> dict[int, dict]:
    out = {}
    ciks = sorted({int(c) for c in ciks})
    for i, cik in enumerate(ciks, 1):
        rel = f"facts_rd/{cik}.json.gz"
        j = net.read_json("sec", rel)
        if j is None:
            r = net.http_get(COMPANYFACTS.format(cik=cik), "sec", ok=(200, 404))
            j = _trim_rd(r.json()) if r.status_code == 200 else {"cik": cik, "facts": {}}
            net.write_json("sec", rel, j)
        out[cik] = j
        if progress and i % 100 == 0:
            progress(f"R&D companyfacts {i}/{len(ciks)}")
    return out


def rd_table(facts: dict) -> pd.DataFrame:
    """Annual (10-K, ~1 year) R&D and revenue per fiscal year end with the filed dates."""
    v4.CONCEPTS.setdefault("rd", RD_CONCEPTS)
    rd = v4.first_filed(facts, "rd", annual=True)
    rev = v4.first_filed(facts, "revenue", annual=True)
    ends = sorted(set(rd["end"]).union(rev["end"]))
    recs = []
    for e in ends:
        a, b = v4._match(rd, e), v4._match(rev, e)
        recs.append(
            {
                "end": e,
                "rd": float(a["value"]) if a is not None else np.nan,
                "rd_filed": a["filed"] if a is not None else pd.NaT,
                "rev": float(b["value"]) if b is not None else np.nan,
                "rev_filed": b["filed"] if b is not None else pd.NaT,
            }
        )
    t = pd.DataFrame(recs)
    if t.empty:
        return t
    t["filed"] = t[["rd_filed", "rev_filed"]].min(axis=1)
    return t.drop_duplicates("end").sort_values("end").reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ employees (10-K)

_NUM = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(million|thousand)?"
_EMP_WORDS = (
    r"(?:(?:regular|permanent|active|total)\s+)?(?:full[- ]time(?:\s+(?:and|or)\s+part[- ]time)?\s+)?"
    r"(?:equivalent\s+)?(?:employees|associates|team members|colleagues)"
)
EMP_PATTERNS = [
    re.compile(
        r"(?:approximately|about|over|more than|nearly|roughly|a total of|had|employed|employ|employs)\s+"
        + _NUM
        + r"\s+"
        + _EMP_WORDS,
        re.I,
    ),
    re.compile(
        r"employ(?:ed|s)?\s+(?:approximately\s+|about\s+|over\s+)?"
        + _NUM
        + r"\s+(?:full[- ]time\s+)?(?:people|persons|individuals)",
        re.I,
    ),
    re.compile(
        r"number of (?:regular |full[- ]time |total )?(?:employees|associates)\s+(?:was|were|totaled|totalled|is)\s+"
        r"(?:approximately\s+|about\s+)?" + _NUM,
        re.I,
    ),
    re.compile(_NUM + r"\s+" + _EMP_WORDS + r"\s+(?:worldwide|globally|as of|at)", re.I),
    re.compile(_NUM + r"\s+(?:full[- ]time\s+)?(?:persons|people|individuals)\s+(?:were\s+)?employed", re.I),
    re.compile(r"(?:workforce|headcount)\s+(?:of|was|totaled|totalled)\s+(?:approximately\s+|about\s+)?" + _NUM, re.I),
]


def html_text(raw: bytes) -> str:
    t = raw.decode("utf-8", "replace")
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", html.unescape(t))


def employee_count(text: str) -> tuple[float, str]:
    """Largest 'N employees'-style headcount in a 10-K text (NaN when none); returns (count, snippet)."""
    best, snip = np.nan, ""
    for pat in EMP_PATTERNS:
        for m in pat.finditer(text):
            num, scale = m.group(1), (m.group(2) or "").lower()
            try:
                v = float(num.replace(",", ""))
            except ValueError:
                continue
            v *= 1e6 if scale == "million" else 1e3 if scale == "thousand" else 1.0
            if 10 <= v <= 5e6 and (not np.isfinite(best) or v > best):
                best, snip = v, text[max(0, m.start() - 80) : m.end() + 40]
    return best, snip


def ten_k_list(subs, cik: int, since: str = "2010-01-01") -> list[dict]:
    sub = subs.get(cik) or {}
    blocks = [(sub.get("filings") or {}).get("recent") or {}]
    blocks += [subs.page(f["name"]) for f in (sub.get("filings") or {}).get("files", []) or []]
    out = []
    for b in blocks:
        for form, d, acc, doc, rep in zip(
            b.get("form", []),
            b.get("filingDate", []),
            b.get("accessionNumber", []),
            b.get("primaryDocument", []),
            b.get("reportDate", []),
            strict=False,
        ):
            if form in ("10-K", "10-KT") and d >= since and doc:
                out.append({"filed": d, "acc": acc, "doc": doc, "report": rep})
    return sorted(out, key=lambda x: x["filed"])


def fetch_employees(net: PitNet, subs, ciks: Iterable[int], root: Path, progress=None) -> dict[int, list[dict]]:
    d = root / "emp"
    d.mkdir(exist_ok=True)
    out = {}
    ciks = sorted({int(c) for c in ciks})
    t0 = time.time()
    for i, cik in enumerate(ciks, 1):
        p = d / f"{cik}.json"
        have = {r["acc"]: r for r in json.loads(p.read_text())} if p.exists() else {}
        recs = []
        changed = False
        for f in ten_k_list(subs, cik):
            if f["acc"] in have:
                recs.append(have[f["acc"]])
                continue
            url = ARCHIVE.format(cik=cik, acc=f["acc"].replace("-", ""), doc=f["doc"])
            try:
                r = net.http_get(url, "sec", ok=(200, 404))
                emp, snip = employee_count(html_text(r.content)) if r.status_code == 200 else (np.nan, "")
            except Exception as exc:  # noqa: BLE001 - recorded as missing; retried on the next run
                if progress:
                    progress(f"10-K {cik} {f['acc']}: {type(exc).__name__}")
                continue
            recs.append({**f, "emp": None if not np.isfinite(emp) else emp, "snippet": snip})
            changed = True
        if changed or not p.exists():
            p.write_text(json.dumps(recs))
        out[cik] = recs
        if progress and i % 25 == 0:
            progress(f"10-K employees {i}/{len(ciks)} ({time.time() - t0:.0f}s)")
    return out


# ------------------------------------------------------------------------------------------------ H-1B

_SUFFIX = {
    "INC",
    "INCORPORATED",
    "CORP",
    "CORPORATION",
    "CO",
    "COMPANY",
    "LLC",
    "LP",
    "LLP",
    "LTD",
    "LIMITED",
    "PLC",
    "HOLDINGS",
    "HOLDING",
    "GROUP",
    "THE",
    "NA",
    "DE",
    "DEL",
}
GENERIC_NAMES = {"AMERICAN", "GENERAL", "UNITED", "NATIONAL", "FIRST", "INTERNATIONAL", "GLOBAL", "ADVANCED", "ACADEMY"}


def norm_name(s: str) -> str:
    s = re.sub(r"\bD/?B/?A\b.*$", " ", str(s or "").upper())
    s = re.sub(r"&", " AND ", s)
    toks = [t for t in re.sub(r"[^A-Z0-9 ]", " ", s).split() if t not in _SUFFIX]
    return " ".join(toks)


def fetch_h1b(root: Path, progress=None) -> pd.DataFrame:
    d = root / "h1b"
    d.mkdir(exist_ok=True)
    parts = []
    for y in H1B_YEARS:
        p = d / f"{y}.csv"
        if not p.exists():
            r = requests.get(H1B.format(y=y), timeout=180, headers=UA)
            r.raise_for_status()
            p.write_bytes(r.content)
            time.sleep(0.5)
        df = pd.read_csv(p, dtype=str)
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        df = df.rename(columns=lambda c: re.sub(r"(Approval|Denial)s$", r"\1", c))
        num = lambda c: pd.to_numeric(df[c].str.replace(",", ""), errors="coerce").fillna(0.0)  # noqa: E731
        parts.append(
            pd.DataFrame(
                {
                    "fy": y,
                    "employer": df["Employer"].fillna(""),
                    "tax4": df["Tax ID"].fillna("").str.strip().str.zfill(4),
                    "init": num("Initial Approval"),
                    "cont": num("Continuing Approval"),
                }
            )
        )
    h = pd.concat(parts, ignore_index=True)
    h["name"] = h["employer"].map(norm_name)
    if progress:
        progress(f"H-1B rows {len(h)} ({H1B_YEARS.start}-{H1B_YEARS.stop - 1})")
    return h.groupby(["fy", "name", "tax4"], as_index=False)[["init", "cont"]].sum()


def match_h1b(h: pd.DataFrame, companies: dict[int, dict]) -> pd.DataFrame:
    """companies: cik -> {"names": [...], "ein": "123456789"}. Returns (cik, fy, init, cont)."""
    by_name = {n: g for n, g in h.groupby("name")}
    by_tax = {t: g for t, g in h.groupby("tax4")}
    by_first: dict[str, list[str]] = {}
    for n in by_name:
        if n:
            by_first.setdefault(n.split()[0], []).append(n)
    rows = []
    for cik, info in companies.items():
        names = {norm_name(n) for n in info.get("names", []) if n}
        names.discard("")
        firsts = {n.split()[0] for n in names}
        ein = re.sub(r"\D", "", str(info.get("ein") or ""))
        hit = []
        for n in names:
            if n in by_name:
                hit.append(by_name[n])
            toks = n.split()
            if len(toks) >= 2 or (len(n) >= 6 and n not in GENERIC_NAMES):  # subsidiaries: "AMAZON COM SERVICES"
                hit += [by_name[x] for x in by_first.get(toks[0], []) if x.startswith(n + " ")]
        if len(ein) >= 4 and ein[-4:] in by_tax:
            g = by_tax[ein[-4:]]
            hit.append(g[g["name"].map(lambda s: bool(s) and s.split()[0] in firsts)])
        if hit:
            m = pd.concat(hit).drop_duplicates(["fy", "name", "tax4"])
            agg = m.groupby("fy")[["init", "cont"]].sum().reset_index()
            agg.insert(0, "cik", cik)
            rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["cik", "fy", "init", "cont"])


# ------------------------------------------------------------------------------------------------ JOLTS


def fetch_jolts(root: Path, progress=None) -> pd.DataFrame:
    """Monthly seasonally adjusted rates: columns (sector, element), index = month start."""
    p = root / "jolts.json"
    if p.exists() and time.time() - p.stat().st_mtime < 20 * 86400:
        raw = json.loads(p.read_text())
    else:
        try:
            raw = _fetch_jolts_raw()
        except (requests.RequestException, RuntimeError) as exc:
            if not p.exists():
                raise
            # BLS outages (e.g. a 503 for a whole weekend, 2026-10-25): the older cache serves - monthly data,
            # the newest month or two are simply missing (as-of rules handle it)
            if progress:
                progress(f"JOLTS refresh failed ({type(exc).__name__}) - using the cache from {_mtime(p)}")
            raw = json.loads(p.read_text())
        else:
            p.write_text(json.dumps(raw))
    return _jolts_frame(raw, progress)


def _mtime(p: Path) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(p.stat().st_mtime))


def _fetch_jolts_raw() -> list:
    ids = [f"JTS{ind}000000000{el}R" for ind in sorted(set(JOLTS_INDUSTRY.values())) for el in JOLTS_ELEMENTS.values()]
    raw = []
    for y0, y1 in JOLTS_WINDOWS:
        for k in range(0, len(ids), 25):
            body = {"seriesid": ids[k : k + 25], "startyear": str(y0), "endyear": str(y1)}
            r = requests.post(BLS_API, json=body, timeout=120)
            r.raise_for_status()
            js = r.json()
            if js.get("status") != "REQUEST_SUCCEEDED":
                raise RuntimeError(f"BLS: {js.get('status')} {js.get('message')}")
            raw += js["Results"]["series"]
            time.sleep(1.0)
    return raw


def _jolts_frame(raw: list, progress=None) -> pd.DataFrame:
    recs = []
    for s in raw:
        sid = s["seriesID"]
        ind, el = sid[3:9], sid[-3:-1]
        for d in s.get("data", []):
            if d["period"].startswith("M") and d["period"] != "M13":
                v = pd.to_numeric(d["value"], errors="coerce")
                recs.append((pd.Timestamp(int(d["year"]), int(d["period"][1:]), 1), ind, el, v))
    df = pd.DataFrame(recs, columns=["month", "ind", "el", "value"]).drop_duplicates(["month", "ind", "el"])
    if progress:
        progress(
            f"JOLTS series {df.groupby(['ind', 'el']).ngroups}, "
            f"months {df['month'].min().date()}..{df['month'].max().date()}"
        )
    return df.pivot_table(index="month", columns=["ind", "el"], values="value").sort_index()


def jolts_sector(jolts: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    """sector -> element -> daily series of (3-month mean - 24-month mean), month m usable from the 15th of m+2."""
    out: dict[str, dict[str, pd.Series]] = {}
    for sec, ind in JOLTS_INDUSTRY.items():
        out[sec] = {}
        for name, el in JOLTS_ELEMENTS.items():
            if (ind, el) not in jolts:
                continue
            s = jolts[(ind, el)].dropna()
            x = s.rolling(3, min_periods=3).mean() - s.rolling(24, min_periods=18).mean()
            x = x.dropna()
            x.index = x.index + pd.offsets.MonthBegin(2) + pd.Timedelta(14, "D")
            idx = pd.date_range(x.index.min(), max(pd.Timestamp.today().normalize(), x.index.max()), freq="D")
            out[sec][name] = x.reindex(idx).ffill(limit=62)
    return out


# ------------------------------------------------------------------------------------------------ features


def _asof_rows(filed: np.ndarray, d: np.datetime64) -> int:
    return int(np.searchsorted(filed, d, "left")) - 1  # last record filed strictly before d


def stock_features(
    rows: pd.DataFrame,
    cik_chains: dict[str, set[int]],
    rd_facts: dict[int, dict],
    emp: dict[int, list[dict]],
    h1b: pd.DataFrame,
    jolts_by_sector: dict,
) -> pd.DataFrame:
    """FEATURES for (date, entity, mcap, sector_group) rows."""
    out = pd.DataFrame(np.nan, index=rows.index, columns=FEATURES)
    ents = rows["entity"].astype(str).to_numpy()
    dates = rows["date"].to_numpy(dtype="datetime64[ns]")
    mcap = rows["mcap"].to_numpy(dtype=float)
    res = {c: out[c].to_numpy(copy=True) for c in FEATURES}
    for e in pd.unique(ents):
        ix = np.flatnonzero(ents == e)
        chain = cik_chains.get(e, set())
        # R&D: merge the chain's facts
        merged: dict[str, list] = {}
        for c in chain:
            for k, v in ((rd_facts.get(c) or {}).get("facts") or {}).items():
                merged.setdefault(k, []).extend(v)
        t = rd_table(merged) if merged else pd.DataFrame()
        if len(t):
            t = t[t["filed"].notna()].sort_values("filed").reset_index(drop=True)
            f_filed = t["filed"].to_numpy(dtype="datetime64[ns]")
            f_end = t["end"].to_numpy(dtype="datetime64[ns]")
            for k in ix:
                d = dates[k]
                j = v4._latest_known(f_filed, f_end, d, v4.ANNUAL_MAX_AGE)
                if j < 0:
                    continue
                rev = t.at[j, "rev"] if pd.notna(t.at[j, "rev_filed"]) and t.at[j, "rev_filed"] < d else np.nan
                known = t[(t["filed"] < d) & (t["end"] >= t.at[j, "end"] - pd.Timedelta(800, "D"))]
                rd_now = t.at[j, "rd"] if pd.notna(t.at[j, "rd_filed"]) and t.at[j, "rd_filed"] < d else np.nan
                if not np.isfinite(rd_now) and known["rd"].notna().sum() == 0 and np.isfinite(rev):
                    rd_now, none = 0.0, 1.0  # files financials but no R&D for two years: none
                else:
                    none = 0.0
                if np.isfinite(rd_now):
                    res["us_rd_none"][k] = none
                    if np.isfinite(rev) and rev > 0:
                        res["us_rd_sales"][k] = rd_now / rev
                    if np.isfinite(mcap[k]) and mcap[k] > 0:
                        res["us_rd_mcap"][k] = rd_now / mcap[k]
                    prev = v4._match(known[known["rd"].notna()], t.at[j, "end"] - pd.Timedelta(365, "D"), tol=45)
                    if prev is not None and prev["rd"] > 0 and rd_now > 0:
                        res["us_rd_growth"][k] = float(np.log(rd_now / prev["rd"]))
        # employees: consecutive 10-Ks of the chain
        er = sorted((r for c in chain for r in emp.get(c, [])), key=lambda r: r["filed"])
        if er:
            e_filed = np.array([np.datetime64(r["filed"], "ns") for r in er])
            e_val = np.array([r["emp"] if r.get("emp") else np.nan for r in er], dtype=float)
            for k in ix:
                j = _asof_rows(e_filed, dates[k] - np.timedelta64(0, "D"))
                if j < 0 or (dates[k] - e_filed[j]) > np.timedelta64(500, "D"):
                    continue
                if not np.isfinite(e_val[j]):
                    res["us_emp_none"][k] = 1.0
                    continue
                res["us_emp_none"][k] = 0.0
                if j >= 1 and np.isfinite(e_val[j - 1]) and e_val[j - 1] > 0:
                    g = float(np.log(e_val[j] / e_val[j - 1]))
                    res["us_emp_growth"][k] = g if abs(g) <= EMP_MAX_LOG_CHANGE else np.nan
        # H-1B
        hh = h1b[h1b["cik"].isin(chain)].groupby("fy")["init"].sum() if len(h1b) else pd.Series(dtype=float)
        for k in ix if len(hh) else []:  # never matched in any year: unknown, not zero
            d = pd.Timestamp(dates[k])
            fy = d.year - 1 if d.month >= H1B_AVAILABLE_MONTH else d.year - 2  # latest FY usable at d
            if fy < H1B_YEARS.start or fy > H1B_YEARS.stop - 1:
                continue
            cur = float(hh.get(fy, 0.0))
            res["us_h1b_init"][k] = np.log1p(cur)
            if fy - 1 >= H1B_YEARS.start:
                res["us_h1b_growth"][k] = np.log1p(cur) - np.log1p(float(hh.get(fy - 1, 0.0)))
    sec = rows["sector_group"].astype(str).to_numpy()
    dts = pd.DatetimeIndex(dates)
    for name in JOLTS_ELEMENTS:
        col = res[f"us_sec_{name}"]
        for s, series in jolts_by_sector.items():
            m = sec == s
            if m.any() and name in series:
                col[m] = series[name].reindex(dts[m], method="ffill").to_numpy(float)
    for c in FEATURES:
        out[c] = res[c]
    return out


# ------------------------------------------------------------------------------------------------ fetch phase


def cik_chains(ds, settings, cache) -> dict[str, set[int]]:
    """entity -> every CIK of its predecessor chain (as in us_v4_study.add_v4_features)."""
    from quantcio.data.pit import load_pit_fundamentals
    from quantcio.data.pit.us_cik import entity_cik
    from quantcio.models import Market

    ents = sorted(ds.frame["entity"].astype(str).unique())
    fund = load_pit_fundamentals(Market.US, settings, cache, ents)
    chains: dict[str, set[int]] = {e: {entity_cik(e)} for e in ents}
    if fund.panel is not None and "cik_chain" in fund.panel:
        for e, ch in fund.panel[["entity", "cik_chain"]].dropna().itertuples(index=False):
            if e in chains:
                chains[e].update(int(x) for x in str(ch).split(",") if x.strip().isdigit())
    return chains


def load_all(ds, settings, cache, progress=None, fetch_text: bool = True) -> dict:
    from quantcio.data.pit.us_cik import Submissions

    say = progress or (lambda m: None)
    root = root_dir(settings)
    net = PitNet(settings)
    chains = cik_chains(ds, settings, cache)
    ciks = sorted({c for cs in chains.values() for c in cs})
    say(f"v8: {len(chains)} entities, {len(ciks)} CIKs")
    rd = fetch_rd(net, ciks, say)
    subs = Submissions(net)
    companies = {c: {"names": subs.names(c), "ein": (subs.get(c) or {}).get("ein")} for c in ciks}
    h = fetch_h1b(root, say)
    h1b = match_h1b(h, companies)
    say(f"v8: H-1B matched CIKs {h1b['cik'].nunique()} / {len(ciks)}")
    jolts = jolts_sector(fetch_jolts(root, say))
    emp = fetch_employees(net, subs, ciks, root, say) if fetch_text else {}
    return {"chains": chains, "rd": rd, "h1b": h1b, "jolts": jolts, "emp": emp}


def main() -> None:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache
    from quantcio.modeling.dataset import build_dataset
    from quantcio.models import Market

    def say(m: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    ds = build_dataset(Market.US, settings, cache, progress=say)
    d = load_all(ds, settings, cache, say)
    n_doc = sum(len(v) for v in d["emp"].values())
    n_ok = sum(1 for v in d["emp"].values() for r in v if r.get("emp"))
    say(f"v8 fetch done: 10-K docs {n_doc}, employee counts found {n_ok}")


if __name__ == "__main__":
    main()
