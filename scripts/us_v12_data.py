"""US v12 block P (docs/MODELING.md section 30): power grid (EIA-930 Dominion / PJM daily load, Virginia commercial
power price), real estate (FRED CRE price index, Case-Shiller, construction spending - first releases; Zillow Loudoun
County ZHVI), data centres (PeeringDB facility presences of hyperscalers, hyperscaler capex from SEC XBRL).
Every series is returned as a daily, point-in-time regime value (value usable from its 'available' date).
Research code outside ``src/quantcio``.
"""

from __future__ import annotations

import io
import json
import time
from datetime import date

import keyed_data as kd
import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "quantcio-research/1.0 (academic research script)"}
EIA = "https://api.eia.gov/v2/"
ZILLOW = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)
PEERINGDB = "https://www.peeringdb.com/api/"
HYPER_ASN = {
    "AWS": [16509, 14618],
    "Microsoft": [8075],
    "Google": [15169, 396982],
    "Meta": [32934],
    "Oracle": [31898],
    "Apple": [714],
}
HYPER_CIK = {
    "Microsoft": [789019],
    "Amazon": [1018724],
    "Alphabet": [1652044, 1288776],
    "Meta": [1326801],
    "Oracle": [1341439],
}
CAPEX = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"]
FRED_P = {"cre": "BOGZ1FL075035503Q", "hpi": "CSUSHPINSA", "cons": "TTLCONS"}
REGIME = [
    "r_dom_load_yoy",
    "r_pjm_load_yoy",
    "r_ercot_load_yoy",
    "r_dom_share_chg",
    "r_va_price_yoy",
    "r_cre_yoy",
    "r_hpi_yoy",
    "r_cons_yoy",
    "r_loudoun_yoy",
    "r_dc_buildout",
    "r_hyper_capex_yoy",
]
AI_INFRA_SIC = {"3674", "3612", "3613", "3621", "3629", "3690", "3585", "1731", "1623"}
DC_REITS = {"EQIX", "DLR", "IRM"}


def _say(_m: str) -> None:
    return None


def _daily(values: pd.Series, avail: pd.Series, max_age: int) -> pd.Series:
    return kd._daily_available(values.reset_index(drop=True), pd.Series(avail).reset_index(drop=True), max_age)


# ================================================================================================ EIA


def _eia_all(route: str, params: dict) -> list[dict]:
    k = kd.key("EIA_API_KEY")
    out, off = [], 0
    while True:
        r = requests.get(EIA + route, params={"api_key": k, **params, "offset": off, "length": 5000}, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(kd._redact(f"EIA {route}: HTTP {r.status_code} {r.text[:150]}", k))
        part = r.json()["response"]["data"]
        out += part
        if len(part) < 5000:
            return out
        off += 5000
        time.sleep(0.3)


def fetch_power(settings, say=_say) -> dict:
    d = kd.root(settings, "power")
    sort = {"sort[0][column]": "period", "sort[0][direction]": "asc"}
    dom = _eia_all(
        "electricity/rto/daily-region-sub-ba-data/data/",
        {"frequency": "daily", "data[0]": "value", "facets[subba][]": "DOM", "facets[timezone][]": "Eastern", **sort},
    )
    pjm = _eia_all(
        "electricity/rto/daily-region-data/data/",
        {
            "frequency": "daily",
            "data[0]": "value",
            "facets[respondent][]": "PJM",
            "facets[type][]": "D",
            "facets[timezone][]": "Eastern",
            **sort,
        },
    )
    ercot = _eia_all(
        "electricity/rto/daily-region-data/data/",
        {
            "frequency": "daily",
            "data[0]": "value",
            "facets[respondent][]": "ERCO",
            "facets[type][]": "D",
            "facets[timezone][]": "Central",
            **sort,
        },
    )
    price = _eia_all(
        "electricity/retail-sales/data/",
        {"frequency": "monthly", "data[0]": "price", "facets[stateid][]": "VA", "facets[sectorid][]": "COM", **sort},
    )
    for name, rows, col in (
        ("dom", dom, "value"),
        ("pjm", pjm, "value"),
        ("ercot", ercot, "value"),
        ("va_price", price, "price"),
    ):
        df = pd.DataFrame(rows)
        pd.DataFrame({"period": df["period"], "value": pd.to_numeric(df[col], errors="coerce")}).to_parquet(
            d / f"{name}.parquet", index=False
        )
    say(f"EIA power: Dominion {len(dom)}, PJM {len(pjm)}, VA price {len(price)}")
    return {"dom": len(dom), "pjm": len(pjm), "va_price": len(price)}


def power_regime(settings) -> dict[str, pd.Series]:
    d = kd.root(settings, "power")
    out = {}
    if not (d / "dom.parquet").exists():
        return out
    dom = (
        pd.read_parquet(d / "dom.parquet")
        .assign(t=lambda x: pd.to_datetime(x["period"]))
        .set_index("t")["value"]
        .asfreq("D")
    )
    pjm = (
        pd.read_parquet(d / "pjm.parquet")
        .assign(t=lambda x: pd.to_datetime(x["period"]))
        .set_index("t")["value"]
        .asfreq("D")
    )
    d28, p28 = dom.rolling(28, min_periods=21).mean(), pjm.rolling(28, min_periods=21).mean()
    pairs = [("r_dom_load_yoy", np.log(d28 / d28.shift(364))), ("r_pjm_load_yoy", np.log(p28 / p28.shift(364)))]
    if (d / "ercot.parquet").exists():
        er = (
            pd.read_parquet(d / "ercot.parquet").assign(t=lambda x: pd.to_datetime(x["period"])).set_index("t")["value"]
        )
        e28 = er[~er.index.duplicated()].asfreq("D").rolling(28, min_periods=21).mean()
        pairs.append(("r_ercot_load_yoy", np.log(e28 / e28.shift(364))))
    for name, s in pairs:
        out[name] = _daily(s.dropna(), s.dropna().index + pd.Timedelta(2, "D"), 10)
    share = d28 / p28
    sc = (share - share.shift(364)).dropna()
    out["r_dom_share_chg"] = _daily(sc, sc.index + pd.Timedelta(2, "D"), 10)
    pr = pd.read_parquet(d / "va_price.parquet").assign(t=lambda x: pd.to_datetime(x["period"])).set_index("t")["value"]
    py = np.log(pr / pr.shift(12)).dropna()
    out["r_va_price_yoy"] = _daily(py, py.index + pd.offsets.MonthBegin(2), 70)
    return out


# ================================================================================================ FRED / Zillow


def fetch_realestate(settings, say=_say) -> dict:
    k = kd.key("FRED_API_KEY")
    d = kd.root(settings, "realestate")
    for name, sid in FRED_P.items():
        params = {
            "series_id": sid,
            "api_key": k,
            "file_type": "json",
            "observation_start": "2000-01-01",
            "output_type": 4,
            "realtime_start": "1776-07-04",
            "realtime_end": "9999-12-31",
        }
        r = requests.get(kd.FRED, params=params, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(kd._redact(f"FRED {sid}: HTTP {r.status_code}", k))
        obs = pd.DataFrame(r.json().get("observations", []))
        obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
        obs[["date", "realtime_start", "value"]].to_parquet(d / f"{name}.parquet", index=False)
        time.sleep(0.6)
    z = requests.get(ZILLOW, headers=UA, timeout=180)
    z.raise_for_status()
    zz = pd.read_csv(io.BytesIO(z.content))
    row = zz[(zz["RegionName"] == "Loudoun County") & (zz["State"] == "VA")]
    s = row.drop(columns=[c for c in zz.columns if not c[:2].isdigit()]).T.iloc[:, 0]
    pd.DataFrame({"month": pd.to_datetime(s.index), "value": s.to_numpy(float)}).to_parquet(
        d / "loudoun.parquet", index=False
    )
    say(f"FRED real estate + Zillow Loudoun ({len(s)} months)")
    return {"fred": list(FRED_P), "zillow_months": len(s)}


def realestate_regime(settings) -> dict[str, pd.Series]:
    d = kd.root(settings, "realestate")
    out = {}
    lags = {"cre": 4, "hpi": 12, "cons": 12}
    for name, n in lags.items():
        p = d / f"{name}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).dropna(subset=["value"])
        df["date"], df["avail"] = (
            pd.to_datetime(df["date"]),
            pd.to_datetime(df["realtime_start"]) + pd.Timedelta(1, "D"),
        )
        df = df.sort_values("date").reset_index(drop=True)
        yoy = np.log(df["value"] / df["value"].shift(n))
        ok = yoy.notna()
        out[f"r_{name}_yoy"] = _daily(yoy[ok], df.loc[ok, "avail"], 400 if name == "cre" else 70)
    p = d / "loudoun.parquet"
    if p.exists():
        z = pd.read_parquet(p).set_index("month")["value"].sort_index()
        y = np.log(z / z.shift(12)).dropna()
        avail = (y.index.to_period("M").to_timestamp() + pd.offsets.MonthBegin(2)) + pd.Timedelta(14, "D")
        out["r_loudoun_yoy"] = _daily(y, avail, 70)
    return out


# ================================================================================================ data centres


def fetch_peeringdb(settings, say=_say) -> dict:
    d = kd.root(settings, "peeringdb")
    rows = []
    for co, asns in HYPER_ASN.items():
        for asn in asns:
            net = requests.get(PEERINGDB + "net", params={"asn": asn}, headers=UA, timeout=60).json().get("data", [])
            time.sleep(3.5)
            for nt in net:
                fac = (
                    requests.get(PEERINGDB + "netfac", params={"net_id": nt["id"]}, headers=UA, timeout=60)
                    .json()
                    .get("data", [])
                )
                rows += [
                    {"company": co, "asn": asn, "fac_id": f.get("fac_id"), "created": f.get("created")} for f in fac
                ]
                time.sleep(3.5)
    df = pd.DataFrame(rows)
    df.to_parquet(d / "netfac.parquet", index=False)
    say(f"PeeringDB hyperscaler facility presences {len(df)}")
    return {"presences": len(df)}


def dc_buildout_regime(settings) -> dict[str, pd.Series]:
    p = kd.root(settings, "peeringdb") / "netfac.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    c = pd.to_datetime(df["created"], utc=True, errors="coerce").dt.tz_localize(None).dropna()
    daily = c.dt.normalize().value_counts().sort_index().asfreq("D", fill_value=0)
    daily = daily.reindex(pd.date_range(daily.index.min(), pd.Timestamp.today().normalize(), freq="D"), fill_value=0)
    n12 = daily.rolling(365, min_periods=365).sum()
    s = (np.log1p(n12) - np.log1p(n12.shift(365))).dropna()
    return {"r_dc_buildout": _daily(s, s.index + pd.Timedelta(1, "D"), 5)}


def _quarterly_capex(facts: list[dict]) -> pd.DataFrame:
    """Quarterly capex flows (end, filed, value) from YTD facts: Q = YTD(k) - YTD(k-1) within a fiscal year."""
    rows = []
    for f in facts:
        if f.get("start") and f.get("end") and f.get("val") is not None and f.get("filed"):
            rows.append((pd.Timestamp(f["start"]), pd.Timestamp(f["end"]), pd.Timestamp(f["filed"]), float(f["val"])))
    if not rows:
        return pd.DataFrame(columns=["end", "filed", "value"])
    df = (
        pd.DataFrame(rows, columns=["start", "end", "filed", "val"])
        .sort_values("filed")
        .drop_duplicates(["start", "end"])
    )
    out = []
    for _s, g in df.groupby("start"):
        g = g.sort_values("end")
        prev = 0.0
        prev_end = None
        for r in g.itertuples(index=False):
            dur = (r.end - r.start).days
            if dur < 80 or dur > 380:
                continue
            q = r.val - prev if prev_end is not None else r.val
            if 80 <= (r.end - (prev_end or r.start)).days <= 100:
                out.append((r.end, r.filed, q))
            prev, prev_end = r.val, r.end
    return pd.DataFrame(out, columns=["end", "filed", "value"]).drop_duplicates("end")


def fetch_hyper_capex(settings, say=_say) -> dict:
    from quantcio.data.pit.us_net import PitNet

    net = PitNet(settings)
    d = kd.root(settings, "hyper_capex")
    n = 0
    for co, ciks in HYPER_CIK.items():
        facts = []
        for cik in ciks:
            r = net.http_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", "sec", ok=(200, 404))
            if r.status_code != 200:
                continue
            g = (r.json().get("facts") or {}).get("us-gaap") or {}
            for c in CAPEX:
                facts += ((g.get(c) or {}).get("units") or {}).get("USD", [])
        q = _quarterly_capex(facts)
        q.to_parquet(d / f"{co}.parquet", index=False)
        n += len(q)
    say(f"hyperscaler capex quarters {n}")
    return {"quarters": n}


def hyper_capex_regime(settings) -> dict[str, pd.Series]:
    d = kd.root(settings, "hyper_capex")
    tabs = {co: pd.read_parquet(d / f"{co}.parquet") for co in HYPER_CIK if (d / f"{co}.parquet").exists()}
    if len(tabs) < len(HYPER_CIK):
        return {}
    days = pd.date_range("2012-01-01", pd.Timestamp.today().normalize(), freq="7D")

    def ttm(t: pd.Timestamp, asof: pd.Timestamp) -> float:
        tot = 0.0
        for q in tabs.values():
            k = q[(q["filed"] < asof)].sort_values("end").tail(4)
            if len(k) < 4 or (asof - k["end"].iloc[-1]).days > 200:
                return float("nan")
            tot += float(k["value"].sum())
        return tot

    vals = pd.Series({t: ttm(t, t) for t in days})
    yoy = np.log(vals / vals.shift(52)).dropna()
    return {"r_hyper_capex_yoy": _daily(yoy, yoy.index, 10)}


def all_regime(settings) -> dict[str, pd.Series]:
    out = {}
    for fn in (power_regime, realestate_regime, dc_buildout_regime, hyper_capex_regime):
        out.update(fn(settings))
    return out


def fetch_all(settings, say=_say) -> dict:
    res = {}
    for name, fn in (
        ("power", fetch_power),
        ("realestate", fetch_realestate),
        ("peeringdb", fetch_peeringdb),
        ("capex", fetch_hyper_capex),
    ):
        try:
            res[name] = fn(settings, say)
        except Exception as exc:  # noqa: BLE001 - one source failing does not stop the others
            res[name] = f"실패 {type(exc).__name__}: {str(exc)[:160]}"
            say(f"{name}: {res[name]}")
    (kd.root(settings, "power") / "_fetch_status.json").write_text(
        json.dumps({"date": str(date.today()), **{k: str(v) for k, v in res.items()}})
    )
    return res
