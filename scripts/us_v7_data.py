"""US v7 data (docs/MODELING.md section 21): weather, news sentiment, policy uncertainty, regulation and extra macro
drivers, and the per-stock features built from them. Research code outside ``src/quantcio``.

Every driver is point in time: a value observed on day d is used only from d + lag (21.1). Downloads are cached under
``<cache>/alt/us_v7`` (refreshed after REFRESH_DAYS).
"""

from __future__ import annotations

import json
import re
import time
import zipfile
from collections.abc import Callable
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests

REFRESH_DAYS = 3
UA = {"User-Agent": "Mozilla/5.0 (research; quantcio)"}
URL = {
    "epu": "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv",
    "cepu": "https://www.policyuncertainty.com/media/Categorical_EPU_Data.xlsx",
    "tpu": "https://www.matteoiacoviello.com/tpu_files/tpu_web_latest.xlsx",
    "gpr": "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.dta",
    "nsi": "https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx",
    "oni": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "cpc": "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/{y}/Population.{kind}.txt",
    "meteo": "https://archive-api.open-meteo.com/v1/archive",
    "fedreg": "https://www.federalregister.gov/api/v1/documents/facets/agency",
}
LAG_DAYS = {"epu": 2, "gpr": 8, "nsi": 10, "hdd": 2, "cdd": 2, "natgas": 0, "ags": 0, "bei": 0}
BETA_DRIVERS = ("epu", "gpr", "nsi", "hdd", "cdd", "natgas", "ags", "bei")
YAHOO = {"natgas": ["NG=F"], "ags": ["ZC=F", "ZW=F", "ZS=F"], "tip": ["TIP"], "ief": ["IEF"]}
AG_REGIONS = {  # (lat, lon): US corn belt / wheat / Gulf, Brazil soy, Argentina, Ukraine, China NE, central India
    "iowa": (41.59, -93.62),
    "kansas": (37.69, -97.34),
    "texas": (29.76, -95.37),
    "mato_grosso": (-15.60, -56.10),
    "argentina": (-32.95, -60.65),
    "ukraine": (49.99, 36.23),
    "heilongjiang": (45.75, 126.65),
    "india": (21.15, 79.09),
}
METEO_START = "2000-01-01"
METEO_LAG_DAYS = 7
NORMAL_YEARS = 10
NORMAL_HALF_WINDOW = 7
SHOCK_SESSIONS = 20
MAX_MONTHLY_AGE_DAYS = 124  # a monthly value older than ~4 months counts as missing
SHOCK_SD_SESSIONS = 756
REGIME = ["r_epu_z", "r_tpu_z", "r_gpr_z", "r_nsi", "r_oni", "r_agwx_t", "r_agwx_p", "r_us_dd"]
SECTOR_AGENCIES = {
    "헬스케어": [
        "health-and-human-services-department",
        "centers-for-medicare-medicaid-services",
        "food-and-drug-administration",
    ],
    "금융": [
        "securities-and-exchange-commission",
        "consumer-financial-protection-bureau",
        "federal-reserve-system",
        "treasury-department",
        "comptroller-of-the-currency",
        "federal-deposit-insurance-corporation",
        "commodity-futures-trading-commission",
    ],
    "에너지": [
        "energy-department",
        "federal-energy-regulatory-commission",
        "interior-department",
        "environmental-protection-agency",
    ],
    "유틸리티": [
        "federal-energy-regulatory-commission",
        "energy-department",
        "environmental-protection-agency",
        "nuclear-regulatory-commission",
    ],
    "소재": ["environmental-protection-agency", "interior-department"],
    "산업재": [
        "transportation-department",
        "federal-aviation-administration",
        "defense-department",
        "labor-department",
        "occupational-safety-and-health-administration",
    ],
    "커뮤니케이션": ["federal-communications-commission"],
    "정보기술": ["federal-communications-commission", "industry-and-security-bureau"],
    "경기소비재": [
        "transportation-department",
        "national-highway-traffic-safety-administration",
        "consumer-product-safety-commission",
        "federal-trade-commission",
    ],
    "필수소비재": ["agriculture-department", "food-and-drug-administration", "federal-trade-commission"],
    "부동산": ["housing-and-urban-development-department"],
}
SECTOR_EPU = {
    "헬스케어": "health",
    "금융": "finreg",
    "산업재": "tpu",
    "소재": "tpu",
    "정보기술": "tpu",
    "경기소비재": "tpu",
}
FEATURES = [
    *[f"us_beta_{k}" for k in BETA_DRIVERS],
    *[f"us_impact_{k}" for k in BETA_DRIVERS],
    "us_sec_reg",
    "us_sec_epu",
]
_NAMES_KO = {
    "epu": "경제정책 불확실성",
    "gpr": "지정학 위험",
    "nsi": "경제 뉴스 심리",
    "hdd": "난방 도일 이상(미국)",
    "cdd": "냉방 도일 이상(미국)",
    "natgas": "천연가스",
    "ags": "곡물(옥수수·밀·대두)",
    "bei": "기대인플레이션(TIP/IEF)",
}
DOCS = {
    **{
        f"us_beta_{k}": (f"{v} 민감도(252일, SPY 통제)", f"{k} sensitivity (252d, SPY-controlled)")
        for k, v in _NAMES_KO.items()
    },
    **{
        f"us_impact_{k}": (f"{v} 충격 전달(민감도 × 최근 20일 변화)", f"{k} shock pass-through")
        for k, v in _NAMES_KO.items()
    },
    "us_sec_reg": (
        "업종 규제 강도(연방 관보 중요 규제, 3개월 vs 36개월)",
        "sector regulatory intensity (Federal Register)",
    ),
    "us_sec_epu": ("업종 정책 불확실성(분야별 EPU·TPU z)", "sector policy uncertainty (categorical EPU / TPU z)"),
}


def root_dir(settings) -> Path:
    p = Path(settings.cache_path) / "alt" / "us_v7"
    p.mkdir(parents=True, exist_ok=True)
    return p


def download(url: str, path: Path, refresh_days: float = REFRESH_DAYS, params: dict | None = None) -> Path:
    if path.exists() and (time.time() - path.stat().st_mtime) < refresh_days * 86400:
        return path
    for attempt in range(6):  # 429 (rate limit): wait and retry
        r = requests.get(url, params=params, timeout=180, headers=UA)
        if r.status_code != 429:
            break
        time.sleep(65 * (attempt + 1))
    r.raise_for_status()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(r.content)
    tmp.replace(path)
    return path


# ------------------------------------------------------------------------------------------------ xlsx (no openpyxl)

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_T = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
_ROW = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"
_RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_xlsx(path: Path) -> dict[str, list[list]]:
    """Minimal .xlsx reader (shared strings, numbers, inline strings) -> {sheet: rows}."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ElementTree.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
            shared.append("".join(t.text or "" for t in si.iter(_T)))
    wb = ElementTree.fromstring(z.read("xl/workbook.xml"))
    rels = {r.get("Id"): r.get("Target") for r in ElementTree.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    out: dict[str, list[list]] = {}
    for sh in wb.find("m:sheets", _NS):
        target = rels[sh.get(_RID)].lstrip("/")
        target = target if target.startswith("xl/") else "xl/" + target
        rows = []
        for row in ElementTree.fromstring(z.read(target)).iter(_ROW):
            vals = {}
            for c in row.findall("m:c", _NS):
                v = c.find("m:v", _NS)
                if v is None:
                    inl = c.find("m:is", _NS)
                    val = "".join(x.text or "" for x in inl.iter(_T)) if inl is not None else None
                elif c.get("t") == "s":
                    val = shared[int(v.text)]
                elif c.get("t") in ("str", "b", "e"):
                    val = v.text
                else:
                    val = float(v.text) if v.text else None  # an empty <v/> (the 2026-10 TPU file has some)
                vals[_col(c.get("r"))] = val
            if vals:
                rows.append([vals.get(i) for i in range(max(vals) + 1)])
        out[sh.get("name")] = rows
    return out


def _excel_date(x: float) -> pd.Timestamp:
    return pd.Timestamp("1899-12-30") + pd.Timedelta(float(x), "D")


# ------------------------------------------------------------------------------------------------ raw series


def epu_daily(root: Path) -> pd.Series:
    d = pd.read_csv(download(URL["epu"], root / "epu_daily.csv"))
    idx = pd.to_datetime(dict(year=d["year"], month=d["month"], day=d["day"]))
    return pd.Series(d["daily_policy_index"].to_numpy(float), index=idx).sort_index()


def gpr_daily(root: Path) -> pd.Series:
    d = pd.read_stata(download(URL["gpr"], root / "gpr_daily.dta"))
    return pd.Series(d["GPRD"].to_numpy(float), index=pd.to_datetime(d["date"])).sort_index()


def nsi_daily(root: Path) -> pd.Series:
    rows = read_xlsx(download(URL["nsi"], root / "nsi.xlsx"))["Data"]
    vals = [(r[0], r[1]) for r in rows[1:] if isinstance(r[0], float) and isinstance(r[1], float)]
    return pd.Series([v for _, v in vals], index=[_excel_date(d) for d, _ in vals]).sort_index()


def tpu_monthly(root: Path) -> pd.Series:
    rows = read_xlsx(download(URL["tpu"], root / "tpu.xlsx"))["TPU_MONTHLY"]
    head = rows[0]
    i_d, i_v = head.index("DATE"), head.index("TPU")
    vals = [
        (r[i_d], r[i_v]) for r in rows[1:] if len(r) > i_v and isinstance(r[i_d], float) and isinstance(r[i_v], float)
    ]
    idx = [_excel_date(d).to_period("M").to_timestamp() for d, _ in vals]
    return pd.Series([v for _, v in vals], index=idx).sort_index()


def cepu_monthly(root: Path) -> pd.DataFrame:
    rows = read_xlsx(download(URL["cepu"], root / "cepu.xlsx"))["Indices"]
    head = [str(h or "") for h in rows[0]]

    def find(word: str) -> int:
        return next(i for i, h in enumerate(head) if word.lower() in h.lower())

    cols = {"health": find("Health care"), "finreg": find("Financial Regulation"), "natsec": find("National security")}
    try:
        cols["trade"] = find("Trade policy")
    except StopIteration:
        pass
    recs = []
    for r in rows[1:]:
        if len(r) < 2 or not isinstance(r[0], float) or not isinstance(r[1], float):
            continue
        rec = {"month": pd.Timestamp(int(r[0]), int(r[1]), 1)}
        for k, i in cols.items():
            rec[k] = r[i] if i < len(r) and isinstance(r[i], float) else np.nan
        recs.append(rec)
    return pd.DataFrame(recs).set_index("month").sort_index()


def oni_monthly(root: Path) -> pd.Series:
    """ONI anomaly indexed by the season's LAST month (DJF 1950 -> 1950-02; NDJ 1950 -> 1951-01)."""
    end = {
        "DJF": 2,
        "JFM": 3,
        "FMA": 4,
        "MAM": 5,
        "AMJ": 6,
        "MJJ": 7,
        "JJA": 8,
        "JAS": 9,
        "ASO": 10,
        "SON": 11,
        "OND": 12,
    }
    out = {}
    for ln in download(URL["oni"], root / "oni.txt").read_text().splitlines()[1:]:
        p = ln.split()
        if len(p) < 4:
            continue
        seas, yr, anom = p[0], int(p[1]), float(p[3])
        m = pd.Timestamp(yr + 1, 1, 1) if seas == "NDJ" else pd.Timestamp(yr, end[seas], 1)
        out[m] = anom
    return pd.Series(out).sort_index()


def cpc_degree_days(root: Path, years: range) -> pd.DataFrame:
    """CONUS population-weighted daily HDD / CDD (NOAA CPC)."""
    cols = {}
    for kind, key in (("Heating", "hdd"), ("Cooling", "cdd")):
        parts = {}
        for y in years:
            refresh = 1.0 if y >= date.today().year else 3650.0
            try:
                txt = download(URL["cpc"].format(y=y, kind=kind), root / f"cpc_{key}_{y}.txt", refresh).read_text()
            except requests.HTTPError:
                continue
            lines = [ln for ln in txt.splitlines() if "|" in ln]
            head = lines[0].split("|")
            conus = next((ln.split("|") for ln in lines if ln.startswith("CONUS|")), None)
            if conus is None:
                continue
            for d, v in zip(head[1:], conus[1:], strict=False):
                if re.fullmatch(r"\d{8}", d) and v.strip() not in ("", "-999"):
                    parts[pd.Timestamp(d)] = float(v)
        cols[key] = pd.Series(parts).sort_index()
    return pd.DataFrame(cols)


def meteo(root: Path, end: date) -> dict[str, pd.DataFrame]:
    out = {}
    for name, (lat, lon) in AG_REGIONS.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": METEO_START,
            "end_date": str(end),
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "UTC",
        }
        p = download(URL["meteo"], root / f"meteo_{name}.json", REFRESH_DAYS, params)
        d = json.loads(p.read_text())["daily"]
        out[name] = pd.DataFrame(
            {"t": d["temperature_2m_mean"], "p": d["precipitation_sum"]}, index=pd.to_datetime(d["time"]), dtype=float
        )
        time.sleep(1.0)
    return out


def fedreg_monthly(root: Path, start: str = "2009-01", end: str | None = None) -> pd.DataFrame:
    """Significant final rules per agency and month (Federal Register facets API), month x agency slug."""
    months = pd.period_range(start, end or pd.Timestamp.today().strftime("%Y-%m"), freq="M")
    d = root / "fedreg"
    d.mkdir(exist_ok=True)
    recs = {}
    cur = pd.Timestamp.today().to_period("M")
    for m in months:
        p = d / f"{m}.json"
        if not p.exists() or m >= cur - 1:
            params = {
                "conditions[publication_date][gte]": str(m.start_time.date()),
                "conditions[publication_date][lte]": str(m.end_time.date()),
                "conditions[type][]": "RULE",
                "conditions[significant]": "1",
            }
            r = requests.get(URL["fedreg"], params=params, timeout=120, headers=UA)
            r.raise_for_status()
            p.write_text(r.text)
            time.sleep(0.3)
        js = json.loads(p.read_text())
        recs[m.to_timestamp()] = {k: v.get("count", 0) for k, v in js.items()}
    return pd.DataFrame.from_dict(recs, orient="index").fillna(0.0).sort_index()


def yahoo_closes(settings, cache) -> dict[str, pd.Series]:
    from quantcio.macro.data import fetch_yahoo

    out = {}
    for key, syms in YAHOO.items():
        ser = []
        for s in syms:
            c, _, _ = fetch_yahoo(s, date(2009, 1, 1), settings, cache)
            ser.append(np.log(c.where(c > 0)).rename(s))
        out[key] = pd.concat(ser, axis=1).mean(axis=1) if len(ser) > 1 else ser[0]
    return out


# ------------------------------------------------------------------------------------------------ transforms


def _daily(s: pd.Series) -> pd.Series:
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.asfreq("D")


def normal_anomaly(s: pd.Series) -> pd.Series:
    """s - mean of s on the same calendar day +-7 days in the previous NORMAL_YEARS years (daily; past only)."""
    s = _daily(s)
    shifted = [
        s.shift(365 * k + j)
        for k in range(1, NORMAL_YEARS + 1)
        for j in range(-NORMAL_HALF_WINDOW, NORMAL_HALF_WINDOW + 1)
    ]
    normal = pd.concat(shifted, axis=1).mean(axis=1, skipna=True)
    n = pd.concat(shifted, axis=1).notna().sum(axis=1)
    return (s - normal).where(n >= 5 * (2 * NORMAL_HALF_WINDOW + 1))


def _to_today(s: pd.Series) -> pd.Series:
    """Extend a daily series to today (NaN) so a forward fill cannot carry a stale value past its limit."""
    end = max(pd.Timestamp.today().normalize(), s.index.max()) if len(s) else pd.Timestamp.today().normalize()
    return s.reindex(pd.date_range(s.index.min(), end, freq="D")) if len(s) else s


def lagged(s: pd.Series, lag_days: int, max_gap_days: int = 10) -> pd.Series:
    """Daily-calendar series whose value at d is the last observation dated <= d - lag (at most max_gap_days old)."""
    s = _to_today(_daily(s)).ffill(limit=max_gap_days)
    return s.shift(lag_days)


def monthly_available(s: pd.Series, months_after: int) -> pd.Series:
    """Monthly series (index = month start) -> daily, value of month m usable from the 1st of month m + months_after."""
    s = s.dropna().sort_index()
    if not len(s):
        return s
    s.index = s.index + pd.offsets.MonthBegin(months_after)
    return _to_today(s.asfreq("D")).ffill(limit=MAX_MONTHLY_AGE_DAYS)


def zscore_past(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    m = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return (s - m) / sd.where(sd > 0)


def build_drivers(settings, cache, progress: Callable[[str], None] | None = None) -> dict:
    """Daily PIT driver levels (for shocks), daily factor 'changes' (for betas, unlagged - the betas are shifted by
    the lag), regime series and the monthly sector inputs."""
    say = progress or (lambda m: None)
    root = root_dir(settings)
    say("v7: policy / news / geopolitics")
    epu, gpr, nsi = epu_daily(root), gpr_daily(root), nsi_daily(root)
    tpu, cepu, oni = tpu_monthly(root), cepu_monthly(root), oni_monthly(root)
    say("v7: weather (NOAA CPC degree days, Open-Meteo ERA5)")
    dd = cpc_degree_days(root, range(2000, date.today().year + 1))
    wx = meteo(root, date.today() - pd.Timedelta(METEO_LAG_DAYS, "D").to_pytimedelta())
    say("v7: Federal Register significant rules")
    fr = fedreg_monthly(root)
    say("v7: extra macro (Yahoo)")
    yh = yahoo_closes(settings, cache)
    m7 = {
        "epu": np.log(_daily(epu).interpolate(limit=3).rolling(7, min_periods=4).mean()),
        "gpr": np.log(_daily(gpr).interpolate(limit=3).rolling(7, min_periods=4).mean()),
        "nsi": _daily(nsi).interpolate(limit=3).rolling(7, min_periods=4).mean(),
    }
    hdd_a, cdd_a = normal_anomaly(dd["hdd"]), normal_anomaly(dd["cdd"])
    levels = {  # level used for the 20-session shock (observation-dated; lag applied below)
        **m7,
        "hdd": hdd_a.rolling(14, min_periods=10).sum(),
        "cdd": cdd_a.rolling(14, min_periods=10).sum(),
        "natgas": yh["natgas"],
        "ags": yh["ags"],
        "bei": yh["tip"] - yh["ief"],
    }
    changes = {  # contemporaneous daily 'factor change' for the betas (observation-dated)
        "epu": m7["epu"].diff(),
        "gpr": m7["gpr"].diff(),
        "nsi": m7["nsi"].diff(),
        "hdd": hdd_a,
        "cdd": cdd_a,
    }
    # global ag-region weather: 30-day mean temperature / precipitation-sum anomalies, averaged over regions
    t_an, p_an = [], []
    for g in wx.values():
        t30 = _daily(g["t"]).rolling(30, min_periods=25).mean()
        p30 = _daily(g["p"]).rolling(30, min_periods=25).sum()
        t_an.append(normal_anomaly(t30))
        pa = normal_anomaly(p30)
        p_an.append(pa / (p30 - pa).abs().clip(lower=5.0))  # share of normal (normal >= 5 mm)
    regime_daily = {
        "r_epu_z": lagged(zscore_past(_daily(m7["epu"]), 1095, 365), LAG_DAYS["epu"]),
        "r_gpr_z": lagged(zscore_past(_daily(m7["gpr"]), 1095, 365), LAG_DAYS["gpr"]),
        "r_nsi": lagged(m7["nsi"], LAG_DAYS["nsi"]),
        "r_tpu_z": monthly_available(zscore_past(np.log(tpu), 36, 24), 2),
        "r_oni": monthly_available(oni, 2),
        "r_agwx_t": lagged(pd.concat(t_an, axis=1).mean(axis=1), METEO_LAG_DAYS),
        "r_agwx_p": lagged(pd.concat(p_an, axis=1).mean(axis=1), METEO_LAG_DAYS),
        "r_us_dd": lagged(levels["hdd"].fillna(0) + levels["cdd"].fillna(0), LAG_DAYS["hdd"]),
    }
    # sector inputs (monthly): regulatory intensity and policy uncertainty
    sec_reg = {}
    for sec, ags_ in SECTOR_AGENCIES.items():
        cnt = (
            fr[[a for a in ags_ if a in fr]].sum(axis=1)
            if any(a in fr for a in ags_)
            else pd.Series(0.0, index=fr.index)
        )
        x = np.log1p(cnt.rolling(3, min_periods=3).sum())
        sec_reg[sec] = monthly_available(x - x.rolling(36, min_periods=24).mean(), 1)
    epu_src = {"health": cepu.get("health"), "finreg": cepu.get("finreg"), "tpu": tpu}
    sec_epu = {}
    for sec, key in SECTOR_EPU.items():
        s = epu_src.get(key)
        if s is not None:
            sec_epu[sec] = monthly_available(zscore_past(np.log(s.where(s > 0)), 36, 24), 2)
    cover = {
        "epu": [str(epu.index.min().date()), str(epu.index.max().date())],
        "gpr": [str(gpr.index.min().date()), str(gpr.index.max().date())],
        "nsi": [str(nsi.index.min().date()), str(nsi.index.max().date())],
        "tpu": [str(tpu.index.min().date()), str(tpu.index.max().date())],
        "cepu": [str(cepu.index.min().date()), str(cepu.index.max().date())],
        "oni": [str(oni.index.min().date()), str(oni.index.max().date())],
        "cpc_dd": [str(dd.index.min().date()), str(dd.index.max().date())],
        "meteo_regions": len(wx),
        "fedreg_months": len(fr),
    }
    return {
        "levels": levels,
        "changes": changes,
        "regime": regime_daily,
        "sec_reg": sec_reg,
        "sec_epu": sec_epu,
        "cover": cover,
    }


# ------------------------------------------------------------------------------------------------ stock features


def _sessions_lag(lag_days: int) -> int:
    return 0 if lag_days <= 0 else int(np.ceil(lag_days * 5 / 7)) + 1


def partial_betas(close: pd.Series, bench: pd.Series, factors: pd.DataFrame, window: int = 252, minp: int = 126):
    """Rolling partial regression coefficients of the stock's daily log return on each factor, controlling for the
    benchmark (same formula as ``modeling.altfeatures.macro_betas``)."""
    idx = close.index
    r = np.log(close.where(close > 0)).diff()
    b = bench.reindex(idx.union(bench.index)).ffill(limit=3).reindex(idx)
    m = np.log(b.where(b > 0)).diff()
    out = {}
    for c in factors.columns:
        f = factors[c].reindex(idx)
        ok = r.notna() & m.notna() & f.notna()
        rr, mm, ff = r.where(ok), m.where(ok), f.where(ok)
        c_rf = rr.rolling(window, min_periods=minp).cov(ff)
        c_rm = rr.rolling(window, min_periods=minp).cov(mm)
        c_mf = mm.rolling(window, min_periods=minp).cov(ff)
        v_m = mm.rolling(window, min_periods=minp).var()
        v_f = ff.rolling(window, min_periods=minp).var()
        den = v_f * v_m - c_mf * c_mf
        out[c] = ((c_rf * v_m - c_rm * c_mf) / den.where(den > 1e-18)).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(out, index=idx)


def stock_features(
    rows: pd.DataFrame,
    prices: dict,
    bench: pd.Series,
    drv: dict,
    rank_gauss: Callable,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """FEATURES + REGIME for (date, entity, sector_group) rows."""
    say = progress or (lambda m: None)
    sessions = pd.DatetimeIndex(bench.index)
    # session-indexed factor changes for the betas: market drivers = daily log change; others as observed
    fac = {}
    for k in BETA_DRIVERS:
        if k in drv["changes"]:
            fac[k] = drv["changes"][k].reindex(sessions)
        else:
            fac[k] = drv["levels"][k].reindex(sessions).diff()
    fac = pd.DataFrame(fac)
    # shocks: PIT level change over SHOCK_SESSIONS, z by the trailing sd of such changes, clipped, / 3
    shock = {}
    for k in BETA_DRIVERS:
        lv = lagged(drv["levels"][k], LAG_DAYS[k]).reindex(sessions)
        ch = lv if k in ("hdd", "cdd") else lv - lv.shift(SHOCK_SESSIONS)
        sd = ch.rolling(SHOCK_SD_SESSIONS, min_periods=252).std()
        shock[k] = (ch / sd.where(sd > 0)).clip(-3, 3) / 3.0
    shock = pd.DataFrame(shock)
    out = pd.DataFrame(index=rows.index, columns=[*FEATURES, *REGIME], dtype=float)
    d_all = rows["date"].to_numpy(dtype="datetime64[ns]")
    done = 0
    for ent, g in rows.groupby("entity", sort=False):
        px = prices.get(ent)
        done += 1
        if px is None or px.empty or "close" not in px:
            continue
        betas = partial_betas(px["close"].astype(float), bench, fac.reindex(px.index))
        for k in BETA_DRIVERS:  # the factor value of day d is known only at d + lag: shift the estimate
            betas[k] = betas[k].shift(_sessions_lag(LAG_DAYS[k]))
        bi = pd.DatetimeIndex(betas.index).to_numpy()
        gd = g["date"].to_numpy(dtype="datetime64[ns]")
        pos = np.searchsorted(bi, gd, "right") - 1
        fresh = (pos >= 0) & ((gd - bi[np.clip(pos, 0, None)]) <= np.timedelta64(7, "D"))
        vals = betas[list(BETA_DRIVERS)].to_numpy(float)[np.clip(pos, 0, None)]
        vals[~fresh] = np.nan
        out.loc[g.index, [f"us_beta_{k}" for k in BETA_DRIVERS]] = vals
        if progress and done % 100 == 0:
            say(f"v7 betas {done} entities")
    dates = pd.DatetimeIndex(d_all)
    for k in BETA_DRIVERS:
        rg = rank_gauss(out[f"us_beta_{k}"], rows["date"])
        s = shock[k].reindex(sessions).ffill(limit=5)
        sv = s.reindex(dates, method="ffill").to_numpy(float)
        out[f"us_impact_{k}"] = rg.to_numpy(float) * sv
    for name, s in drv["regime"].items():
        out[name] = s.reindex(dates, method="ffill").to_numpy(float) if len(s) else np.nan
    sec = rows["sector_group"].astype(str).to_numpy()
    reg = np.full(len(rows), np.nan)
    pol = np.zeros(len(rows))
    for sname, s in drv["sec_reg"].items():
        m = sec == sname
        if m.any() and len(s):
            reg[m] = s.reindex(dates[m], method="ffill").to_numpy(float)
    for sname, s in drv["sec_epu"].items():
        m = sec == sname
        if m.any() and len(s):
            pol[m] = np.nan_to_num(s.reindex(dates[m], method="ffill").to_numpy(float), nan=0.0)
    out["us_sec_reg"] = reg
    out["us_sec_epu"] = pol
    return out


__all__ = ["DOCS", "FEATURES", "REGIME", "build_drivers", "stock_features"]
