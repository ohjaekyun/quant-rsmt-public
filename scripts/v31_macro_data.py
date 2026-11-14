"""Full-history rates, prices by category and production by industry (docs/MODELING.md 75, pre-declared 2026-10-24).
Research only.

    .venv/bin/python scripts/v31_macro_data.py fetch

KR (ECOS): daily market rates (KTB 3Y / 10Y, AA- corporate 3Y, CD 91D), monthly CPI by purpose, PPI by product,
production / shipments / inventory by industry (seasonally adjusted). US (FRED / ALFRED first releases): CPI items,
PPI by industry, industrial production by industry, headline / core CPI; DGS10 daily (market data, not revised).
Files under <cache>/alt/keyed/v31/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_data as v13  # noqa: E402

say = v13.say
KR_RATES = {"ktb3": "010200000", "ktb10": "010210000", "aa3": "010300000", "cd91": "010502000"}
KR_CPI = list("0ABCDEFGHIJKL")
KR_PPI = [
    "*AA",
    "301AA",
    "302AA",
    "304AA",
    "305AA",
    "3054AA",
    "305622AA",
    "306AA",
    "307AA",
    "309AA",
    "30911AA",
    "310AA",
    "311AA",
    "312AA",
    "3121AA",
    "401AA",
    "501AA",
    "503AA",
    "504AA",
]
KR_IP = [
    "I11ACA",
    "I11ACB",
    "I11ACE",
    "I11ACF",
    "I11ACJ",
    "I11ACK",
    "I11ACL",
    "I11ACM",
    "I11ACN",
    "I11ACO",
    "I11ACQ",
    "I11ACS",
    "I11ACT",
    "I11ACU",
    "I11ACV",
    "I11ACW",
    "I11ADA",
]
KR_IP_TYPES = {"2": "prod", "4": "ship", "6": "inv"}  # seasonally adjusted production / shipments / inventory
US_CPI = [
    "CPIAPPSL",
    "CPIMEDSL",
    "CPIFABSL",
    "CPIENGSL",
    "CUSR0000SAH1",
    "CPITRNSL",
    "CUSR0000SETA01",
    "CPIRECSL",
    "CUSR0000SEFV",
    "CPIAUCSL",
    "CPILFESL",
]
US_PPI = [
    "PCU334413334413",
    "PCU211111211111",
    "PCU325412325412",
    "PCU325325",
    "PCU331110331110",
    "PCU336110336110",
    "PCU336411336411",
    "PCU511210511210",
    "PCU327320327320",
    "PCU2211--2211--",
    "PPIACO",
]
US_IP = [
    "IPG334S",
    "IPG3344S",
    "IPG3361T3S",
    "IPG325S",
    "IPG3254S",
    "IPG331S",
    "IPG211S",
    "IPG311A2S",
    "IPG3364T9S",
    "IPG333S",
    "IPG335S",
    "IPG327S",
    "IPG322S",
    "IPUTIL",
]


def root(settings) -> Path:
    return kd.root(settings, "v31")


def _ecos(k: str, code: str, cycle: str, item: str, start: str, end: str) -> list[dict]:
    rows, s = [], 1
    while True:
        url = f"{kd.ECOS}/StatisticSearch/{k}/json/kr/{s}/{s + 999}/{code}/{cycle}/{start}/{end}/{item}"
        js = requests.get(url, timeout=120).json()
        if "StatisticSearch" not in js:
            return rows
        part = js["StatisticSearch"].get("row") or []
        rows += part
        total = int(js["StatisticSearch"].get("list_total_count") or 0)
        s += 1000
        time.sleep(0.25)
        if not part or s > total:
            return rows


def fetch_kr(settings) -> dict:
    k = kd.key("ECOS_API_KEY")
    if not k:
        return {"skipped": "ECOS_API_KEY 없음"}
    d = root(settings)
    today = pd.Timestamp.today()
    rates = []
    for name, item in KR_RATES.items():
        for x in _ecos(k, "817Y002", "D", item, "19950101", f"{today:%Y%m%d}"):
            rates.append(
                {
                    "series": name,
                    "date": pd.Timestamp(x["TIME"]),
                    "value": pd.to_numeric(x["DATA_VALUE"], errors="coerce"),
                }
            )
    r = pd.DataFrame(rates).dropna(subset=["value"])
    r["available"] = r["date"] + pd.Timedelta(1, "D")
    r.to_parquet(d / "kr_rates.parquet", index=False)
    monthly = []
    for table, items in (
        ("cpi", [("901Y009", i) for i in KR_CPI]),
        ("ppi", [("404Y014", i) for i in KR_PPI]),
        ("ip", [("901Y032", f"{i}/{t}") for i in KR_IP for t in KR_IP_TYPES]),
    ):
        for code, item in items:
            for x in _ecos(k, code, "M", item, "200001", f"{today:%Y%m}"):
                monthly.append(
                    {
                        "table": table,
                        "item": item,
                        "period": x["TIME"],
                        "value": pd.to_numeric(x["DATA_VALUE"], errors="coerce"),
                    }
                )
    m = pd.DataFrame(monthly).dropna(subset=["value"])
    m["date"] = pd.to_datetime(m["period"], format="%Y%m") + pd.offsets.MonthEnd(0)
    m["available"] = m["date"] + pd.offsets.MonthEnd(1) + pd.Timedelta(1, "D")  # 71.1: end of the next month
    m.to_parquet(d / "kr_monthly.parquet", index=False)
    return {
        "rates": r.groupby("series")["date"].min().dt.date.astype(str).to_dict(),
        "monthly_series": int(m.groupby(["table", "item"]).ngroups),
    }


def _fred(k: str, sid: str, first: bool) -> pd.DataFrame | None:
    params = {"series_id": sid, "api_key": k, "file_type": "json", "observation_start": "2000-01-01"}
    if first:
        params.update({"output_type": 4, "realtime_start": "1776-07-04", "realtime_end": "9999-12-31"})
    r = requests.get(kd.FRED, params=params, timeout=120)
    time.sleep(0.6)
    if r.status_code != 200:
        return None
    o = pd.DataFrame(r.json().get("observations", []))
    if o.empty:
        return None
    o["value"] = pd.to_numeric(o["value"], errors="coerce")
    o["date"] = pd.to_datetime(o["date"])
    o["available"] = (pd.to_datetime(o["realtime_start"]) if first else o["date"]) + pd.Timedelta(1, "D")
    return o.dropna(subset=["value"])[["date", "value", "available"]]


def fetch_us(settings) -> dict:
    k = kd.key("FRED_API_KEY")
    if not k:
        return {"skipped": "FRED_API_KEY 없음"}
    parts, kinds, missing = [], {}, []
    for group, ids in (("cpi", US_CPI), ("ppi", US_PPI), ("ip", US_IP)):
        for sid in ids:
            o = _fred(k, sid, True)
            kind = "first"
            if o is None:  # no vintages: latest values, available a conservative 45 days after the month
                o, kind = _fred(k, sid, False), "latest"
                if o is not None:
                    o["available"] = o["date"] + pd.offsets.MonthEnd(0) + pd.Timedelta(45, "D")
            if o is None:
                missing.append(sid)
                continue
            parts.append(o.assign(group=group, series=sid, vintage=kind))
            kinds[sid] = kind
    t = pd.concat(parts, ignore_index=True)
    t.to_parquet(root(settings) / "us_monthly.parquet", index=False)
    dgs = _fred(k, "DGS10", False)
    if dgs is not None:
        dgs.to_parquet(root(settings) / "us_dgs10.parquet", index=False)
    return {"series": len(kinds), "latest_only": [s for s, v in kinds.items() if v == "latest"], "missing": missing}


def main() -> None:
    from quantcio.config import load_settings

    s = load_settings()
    secrets = [kd.key("ECOS_API_KEY"), kd.key("FRED_API_KEY")]
    for name, fn in (("kr", fetch_kr), ("us", fetch_us)):
        try:
            say(f"{name}: {fn(s)}")
        except Exception as exc:  # noqa: BLE001 - the other market still loads
            say(f"{name}: FAILED {type(exc).__name__}: {kd._redact(str(exc), *secrets)}")


if __name__ == "__main__":
    main()
