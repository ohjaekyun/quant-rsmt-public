"""Credit / collateral lending and venture-investment data (docs/MODELING.md 70, pre-declared 2026-10-20).
Research only.

    .venv/bin/python scripts/v29_credit_vc_data.py fetch --what all   # or kofia, ecos, fred, finra, formd, kr_vc

KR: KOFIA FreeSIS daily credit balances (margin loans by market, short-sale stock loans, securities-collateral loans)
and stock-market funds (investor deposits, RP, unpaid trades, forced liquidations); BOK ECOS monthly household loans
by purpose / sector and bank loans by type; the venture-investment portal's year-to-date table by industry (forward
ledger - no monthly history online). US: FINRA margin statistics (monthly), Fed H.8 weekly bank loans and G.19 monthly
consumer credit as FIRST releases with their publication dates (ALFRED), SEC Form D data sets (2008Q1-) aggregated per
filing date x industry group. Files under <cache>/alt/keyed/v29/.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_data as v13  # noqa: E402

say = v13.say
UA = {"User-Agent": "Mozilla/5.0 (compatible; research bot)"}
FREESIS = "https://freesis.kofia.or.kr/meta/getMetaDataList.do"
KOFIA = {  # FreeSIS statistic -> column names (TMPV2..), values in KRW million; checked against press figures
    "credit": (
        "STATSCU0100000070BO",
        [
            "margin_total",
            "margin_kospi",
            "margin_kosdaq",
            "stockloan_total",
            "stockloan_kospi",
            "stockloan_kosdaq",
            "other",
            "collateral_loans",
        ],
    ),
    "funds": (
        "STATSCU0100000060BO",
        ["deposits", "deriv_deposits", "rp_sold", "unpaid", "forced_sales", "forced_share_pct"],
    ),
}
ECOS_TABLES = {"151Y005": "household_by_purpose", "151Y002": "household_by_sector", "104Y016": "bank_loans"}
FRED_CREDIT = {  # H.8 weekly (all commercial banks, SA) and G.19 monthly consumer credit (SA)
    "h8_loans": "TOTLL",
    "h8_ci": "TOTCI",
    "h8_real_estate": "RELACBW027SBOG",
    "h8_consumer": "CLSACBW027SBOG",
    "h8_cards": "CCLACBW027SBOG",
    "g19_total": "TOTALSL",
    "g19_revolving": "REVOLSL",
    "g19_nonrevolving": "NONREVSL",
}
FORMD_PAGE = "https://www.sec.gov/data-research/sec-markets-data/form-d-data-sets"
FINRA_PAGE = "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics"
VCS = "https://www.vcs.go.kr/web/portal/statistics/list"


def root(settings) -> Path:
    return kd.root(settings, "v29")


# ================================================================================================ KR


def fetch_kofia(settings) -> dict:
    """Daily FreeSIS statistics, one request per calendar year (3 s apart), 2005-."""
    sess, out = requests.Session(), {}
    hdr = {**UA, "Content-Type": "application/json; charset=UTF-8"}
    for name, (obj, cols) in KOFIA.items():
        rows = []
        for y in range(2005, date.today().year + 1):
            body = {
                "dmSearch": {
                    "tmpV40": "1000000",
                    "tmpV41": "1",
                    "tmpV1": "D",
                    "tmpV45": f"{y}0101",
                    "tmpV46": f"{y}1231",
                    "OBJ_NM": obj,
                }
            }
            for attempt in range(4):
                try:
                    r = sess.post(FREESIS, data=json.dumps(body), headers=hdr, timeout=60)
                    break
                except requests.RequestException:
                    time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError(f"FreeSIS {name} {y}: 응답 없음")
            time.sleep(3.0)
            for x in (r.json() if r.ok else {}).get("ds1") or []:
                rec = {"date": pd.Timestamp(str(x["TMPV1"]))}
                rec.update({c: x.get(f"TMPV{i + 2}") for i, c in enumerate(cols)})
                rows.append(rec)
        t = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
        for c in cols:
            t[c] = pd.to_numeric(t[c], errors="coerce")
        t["available"] = t["date"] + pd.Timedelta(1, "D")  # published after the close of the day
        t.to_parquet(root(settings) / f"kofia_{name}.parquet", index=False)
        out[name] = {"days": len(t), "from": str(t["date"].min().date())}
    return out


def fetch_ecos(settings) -> dict:
    k = kd.key("ECOS_API_KEY")
    if not k:
        return {"skipped": "ECOS_API_KEY 없음"}
    rows, end = [], f"{date.today():%Y%m}"
    for code, table in ECOS_TABLES.items():
        js = requests.get(f"{kd.ECOS}/StatisticItemList/{k}/json/kr/1/200/{code}", timeout=60).json()
        items = [x for x in (js.get("StatisticItemList") or {}).get("row") or [] if x.get("CYCLE") == "M"]
        for it in items:
            for x in kd._ecos_rows(k, code, it["ITEM_CODE"], "200001", end):
                rows.append(
                    {
                        "table": table,
                        "item": it["ITEM_CODE"],
                        "name": it["ITEM_NAME"],
                        "period": x["TIME"],
                        "value": pd.to_numeric(x.get("DATA_VALUE"), errors="coerce"),
                    }
                )
            time.sleep(0.3)
    t = pd.DataFrame(rows)
    t["month"] = pd.to_datetime(t["period"], format="%Y%m")
    # release lag unknown per vintage: the end of the following month (71.1)
    t["available"] = t["month"] + pd.offsets.MonthEnd(2) + pd.Timedelta(1, "D")
    t.to_parquet(root(settings) / "ecos_loans.parquet", index=False)
    return {"series": int(t.groupby(["table", "item"]).ngroups), "months": int(t["month"].nunique())}


def kr_vc_ledger(settings) -> dict:
    """The portal's year-to-date investment by industry (KRW 100 million) as of today -> forward ledger."""
    r = requests.get(VCS, headers=UA, timeout=60)
    r.raise_for_status()
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", r.text, flags=re.S)
    t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t))
    m = re.search(r"구분 ((?:\d{4} )+)(.*?)합계 ([\d, ]+)", t)
    if not m:
        return {"skipped": "표 형식이 바뀜"}
    years = m.group(1).split()
    rows = []
    row = re.compile(r"([가-힣A-Za-z/]+) ((?:[\d,]+ ){" + str(len(years)) + "})")
    for name, nums in row.findall(m.group(2) + " "):
        vals = [float(v.replace(",", "")) for v in nums.split()]
        rows += [
            {"asof": str(date.today()), "industry": name, "year": int(y), "krw_100m": v}
            for y, v in zip(years, vals, strict=True)
        ]
    d = root(settings) / "kr_vc"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / f"{date.today()}.parquet", index=False)
    return {"rows": len(rows), "industries": len({x["industry"] for x in rows})}


# ================================================================================================ US


def fetch_fred(settings) -> dict:
    k = kd.key("FRED_API_KEY")
    if not k:
        return {"skipped": "FRED_API_KEY 없음"}
    out = {}
    for name, sid in FRED_CREDIT.items():
        params = {
            "series_id": sid,
            "api_key": k,
            "file_type": "json",
            "observation_start": "2005-01-01",
            "output_type": 4,
            "realtime_start": "1776-07-04",
            "realtime_end": "9999-12-31",
        }
        r = requests.get(kd.FRED, params=params, timeout=120)
        kind = "first"
        if r.status_code != 200:  # no vintages: the latest values with a conservative lag
            params = {"series_id": sid, "api_key": k, "file_type": "json", "observation_start": "2005-01-01"}
            r, kind = requests.get(kd.FRED, params=params, timeout=120), "latest"
        if r.status_code != 200:
            raise RuntimeError(kd._redact(f"FRED {sid}: HTTP {r.status_code} {r.text[:160]}", k))
        obs = pd.DataFrame(r.json().get("observations", []))
        obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
        obs["date"] = pd.to_datetime(obs["date"])
        if kind == "first":
            obs["available"] = pd.to_datetime(obs["realtime_start"]) + pd.Timedelta(1, "D")
        else:
            lag = 9 if name.startswith("h8") else 40
            obs["available"] = obs["date"] + pd.Timedelta(lag, "D")
        obs = obs.dropna(subset=["value"])[["date", "value", "available"]].assign(series=name, vintage=kind)
        obs.to_parquet(root(settings) / f"fred_{name}.parquet", index=False)
        out[name] = (kind, len(obs))
        time.sleep(0.6)
    return out


def fetch_finra(settings) -> dict:
    page = requests.get(FINRA_PAGE, headers=UA, timeout=60)
    page.raise_for_status()
    links = re.findall(r'href="([^"]+\.xlsx?)"', page.text)
    link = next((x for x in links if "margin" in x.lower()), None)
    if not link:
        return {"skipped": "엑셀 링크를 찾지 못함"}
    url = link if link.startswith("http") else "https://www.finra.org" + link
    x = pd.read_excel(io.BytesIO(requests.get(url, headers=UA, timeout=120).content), header=None)
    hdr = next(i for i in range(min(len(x), 20)) if x.iloc[i].astype(str).str.contains("Debit", case=False).any())
    t = pd.DataFrame(x.iloc[hdr + 1 :].to_numpy(), columns=[str(c).strip() for c in x.iloc[hdr]])
    first = t.columns[0]
    t["month"] = pd.to_datetime(t[first], errors="coerce") + pd.offsets.MonthEnd(0)
    t = t.dropna(subset=["month"])
    debit = next(c for c in t.columns if "debit" in c.lower())
    t = t.assign(debit=pd.to_numeric(t[debit], errors="coerce"))
    t["available"] = t["month"] + pd.Timedelta(25, "D")  # published around the third week of the next month
    t[["month", "debit", "available"]].dropna().to_parquet(root(settings) / "finra_margin.parquet", index=False)
    return {"months": int(t["debit"].notna().sum()), "from": str(t["month"].min().date())}


def fetch_formd(settings) -> dict:
    """Every quarterly Form D data set -> per (filing date, fund / non-fund, industry group): new filings, amount
    sold."""
    ua = {"User-Agent": kd.key("SEC_USER_AGENT") or UA["User-Agent"]}
    d = root(settings) / "formd"
    d.mkdir(parents=True, exist_ok=True)
    page = requests.get(FORMD_PAGE, headers=ua, timeout=60)
    page.raise_for_status()
    zips = sorted(set(re.findall(r'href="([^"]*form-d-data-sets/[^"]+\.zip)"', page.text)))
    n_new = 0
    for z in zips:
        tag = re.search(r"(\d{4}q\d)", z).group(1)
        p = d / f"{tag}.parquet"
        if p.exists():
            continue
        raw = requests.get("https://www.sec.gov" + z, headers=ua, timeout=300).content
        time.sleep(0.5)
        zf = zipfile.ZipFile(io.BytesIO(raw))

        def tsv(name, zf=zf):
            f = next(n for n in zf.namelist() if n.upper().endswith(name))
            return pd.read_csv(zf.open(f), sep="\t", dtype=str, low_memory=False, on_bad_lines="skip")

        sub, off = tsv("FORMDSUBMISSION.TSV"), tsv("OFFERING.TSV")
        sub.columns, off.columns = sub.columns.str.upper(), off.columns.str.upper()
        m = sub[["ACCESSIONNUMBER", "FILING_DATE", "SUBMISSIONTYPE"]].merge(
            off[["ACCESSIONNUMBER", "INDUSTRYGROUPTYPE", "INVESTMENTFUNDTYPE", "TOTALAMOUNTSOLD"]], on="ACCESSIONNUMBER"
        )
        m["date"] = pd.to_datetime(m["FILING_DATE"], errors="coerce", format="mixed")
        m["sold"] = pd.to_numeric(m["TOTALAMOUNTSOLD"], errors="coerce")
        m["new"] = m["SUBMISSIONTYPE"].str.upper().eq("D")
        m["fund"] = m["INDUSTRYGROUPTYPE"].str.contains("Pooled Investment Fund", case=False, na=False)
        agg = (
            m[m["new"]]
            .groupby(["date", "fund", "INDUSTRYGROUPTYPE"], dropna=False)
            .agg(filings=("ACCESSIONNUMBER", "count"), sold=("sold", "sum"))
            .reset_index()
        )
        agg.rename(columns={"INDUSTRYGROUPTYPE": "industry"}).assign(quarter=tag).to_parquet(p, index=False)
        n_new += 1
    return {"quarters": len(zips), "new": n_new}


def main() -> None:
    from quantcio.config import load_settings

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    names = ["kofia", "ecos", "fred", "finra", "formd", "kr_vc"]
    f.add_argument("--what", default="all", choices=["all", *names])
    a = ap.parse_args()
    s = load_settings()
    steps = {
        "kofia": fetch_kofia,
        "ecos": fetch_ecos,
        "fred": fetch_fred,
        "finra": fetch_finra,
        "formd": fetch_formd,
        "kr_vc": kr_vc_ledger,
    }
    secrets = [kd.key(x) for x in ("ECOS_API_KEY", "FRED_API_KEY")]
    for name in names:
        if a.what in ("all", name):
            try:
                say(f"{name}: {steps[name](s)}")
            except Exception as exc:  # noqa: BLE001 - the other sources still load
                say(f"{name}: FAILED {type(exc).__name__}: {kd._redact(str(exc), *secrets)}")


if __name__ == "__main__":
    main()
