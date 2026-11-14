"""Company headquarters location (docs/MODELING.md 38, pre-declared 2026-10-10): point-in-time HQ history and the
location features of block L. Research only.

US: the business address of every XBRL filing in the SEC Financial Statement Data Sets (``sub.txt``, 2009Q1+; only
that member is read from each quarterly zip with HTTP range requests) -> per-CIK address history known from each filing
date; ZIP -> coordinates from the Census ZCTA gazetteer; state unemployment / coincident index from FRED.
KR: the current address (OpenDART company.json, cached) is valid from the latest '본점소재지변경' filing (cached DART
lists) on; before it the address is unknown (NaN) - the filings' old addresses sit in KRX KIND documents that the
OpenDART document API does not serve. Relocation events are dated by the filing.

    .venv/bin/python scripts/v15_location.py fetch
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_data as v13  # noqa: E402

say = v13.say
FSDS_PAGE = "https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets"
ZCTA = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_zcta_national.zip"
NYC = (40.7128, -74.0060)
SEOUL = (37.5665, 126.9780)
PEER_KM = 100.0
STATES = (
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR "
    "PA RI SC SD TN TX UT VT VA WA WV WI WY"
).split()
# 시도 centroids (approximate city-hall coordinates) - distances only need ~10 km accuracy
KR_REGION = {
    "서울": (37.5665, 126.9780),
    "부산": (35.1796, 129.0756),
    "대구": (35.8714, 128.6014),
    "인천": (37.4563, 126.7052),
    "광주": (35.1595, 126.8526),
    "대전": (36.3504, 127.3845),
    "울산": (35.5384, 129.3114),
    "세종": (36.4800, 127.2890),
    "경기": (37.2752, 127.0095),
    "강원": (37.8813, 127.7298),
    "충북": (36.6424, 127.4890),
    "충남": (36.6588, 126.6728),
    "전북": (35.8242, 127.1480),
    "전남": (34.8161, 126.4629),
    "경북": (36.5760, 128.5056),
    "경남": (35.2383, 128.6924),
    "제주": (33.4996, 126.5312),
}
KR_PREFIX = [
    ("서울", "서울"),
    ("부산", "부산"),
    ("대구", "대구"),
    ("인천", "인천"),
    ("광주", "광주"),
    ("대전", "대전"),
    ("울산", "울산"),
    ("세종", "세종"),
    ("경기", "경기"),
    ("강원", "강원"),
    ("충청북", "충북"),
    ("충북", "충북"),
    ("충청남", "충남"),
    ("충남", "충남"),
    ("전라북", "전북"),
    ("전북", "전북"),
    ("전라남", "전남"),
    ("전남", "전남"),
    ("경상북", "경북"),
    ("경북", "경북"),
    ("경상남", "경남"),
    ("경남", "경남"),
    ("제주", "제주"),
]
GANGNAM3 = ("강남구", "서초구", "송파구")


def haversine_km(a: tuple[float, float], b: np.ndarray) -> np.ndarray:
    lat1, lon1 = np.radians(a[0]), np.radians(a[1])
    lat2, lon2 = np.radians(b[:, 0]), np.radians(b[:, 1])
    d = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(d))


# ================================================================================================ US fetch


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests (reads one zip member only)."""

    def __init__(self, url: str, headers: dict):
        self.url, self.h, self.pos = url, dict(headers), 0
        self.s = requests.Session()
        r = self.s.head(url, headers=self.h, timeout=60, allow_redirects=True)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        for attempt in range(5):
            try:
                r = self.s.get(self.url, headers={**self.h, "Range": f"bytes={self.pos}-{end}"}, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException:
                time.sleep(5.0 * (attempt + 1))
        else:
            raise RuntimeError(f"range read failed: {self.url}")
        data = r.content
        b[: len(data)] = data
        self.pos += len(data)
        time.sleep(0.15)
        return len(data)


def us_fetch(settings, say=say) -> dict:
    d = kd.root(settings, "hq_us")
    hdr = v13._sec_headers()
    page = requests.get(FSDS_PAGE, headers=hdr, timeout=120).text
    links = sorted(set(re.findall(r'href="([^"]+/(\d{4}q\d)\.zip)"', page)))
    new = 0
    for link, q in links:
        out = d / f"sub_{q}.parquet"
        if out.exists():
            continue
        url = "https://www.sec.gov" + link if link.startswith("/") else link
        z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url, hdr), buffer_size=4 << 20))
        sub = pd.read_csv(
            z.open("sub.txt"),
            sep="\t",
            dtype=str,
            usecols=lambda c: c in ("cik", "form", "filed", "countryba", "stprba", "cityba", "zipba"),
        )
        sub.to_parquet(out, index=False)
        new += 1
        say(f"FSDS {q}: {len(sub)} filings")
    zp = d / "zcta.parquet"
    if not zp.exists():
        r = requests.get(ZCTA, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        zz = zipfile.ZipFile(io.BytesIO(r.content))
        g = pd.read_csv(zz.open(zz.namelist()[0]), sep="\t", dtype={"GEOID": str})
        g.columns = [c.strip() for c in g.columns]
        g[["GEOID", "INTPTLAT", "INTPTLONG"]].rename(
            columns={"GEOID": "zip", "INTPTLAT": "lat", "INTPTLONG": "lon"}
        ).to_parquet(zp, index=False)
    fred = {}
    k = kd.key("FRED_API_KEY")
    fp = d / "state_fred.json"
    if k and not fp.exists():
        for st in STATES:
            for suffix in ("UR", "PHCI"):
                sid = f"{st}{suffix}"
                r = requests.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={"series_id": sid, "api_key": k, "file_type": "json"},
                    timeout=60,
                )
                if r.status_code == 200:
                    fred[sid] = [(o["date"], o["value"]) for o in r.json().get("observations", [])]
                time.sleep(0.2)
        fp.write_text(json.dumps(fred))
    return {"quarters": len(links), "new": new, "fred_series": len(fred) if fred else "cached"}


def us_history(settings) -> pd.DataFrame:
    d = kd.root(settings, "hq_us")
    parts = [pd.read_parquet(p) for p in sorted(d.glob("sub_*.parquet"))]
    h = pd.concat(parts, ignore_index=True)
    h["cik"] = pd.to_numeric(h["cik"], errors="coerce")
    h["filed"] = pd.to_datetime(h["filed"], format="%Y%m%d", errors="coerce")
    h = h.dropna(subset=["cik", "filed"])
    h["zip5"] = h["zipba"].astype(str).str.extract(r"(\d{5})")[0]
    z = pd.read_parquet(d / "zcta.parquet").set_index("zip")
    h = h.join(z, on="zip5")
    h["city"] = h["cityba"].astype(str).str.upper().str.strip()
    h["state"] = h["stprba"].astype(str).str.upper().str.strip()
    h["country"] = h["countryba"].astype(str).str.upper().str.strip()
    return h.sort_values(["cik", "filed"]).drop_duplicates(["cik", "filed"], keep="last")[
        ["cik", "filed", "state", "city", "zip5", "country", "lat", "lon"]
    ]


# ================================================================================================ KR fetch (caches)


def kr_region(addr: str) -> tuple[str | None, str | None]:
    a = str(addr or "").strip()
    reg = next((r for p, r in KR_PREFIX if a.startswith(p)), None)
    parts = a.split()
    district = parts[1] if len(parts) > 1 else None
    return reg, district


def kr_history(settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(addresses: stock_code, valid_from, region, district) from the current address, (moves: stock_code, date)."""
    comp = {}
    for p in glob.glob(str(Path(settings.cache_path) / "alt" / "kr_dart_company" / "*.json")):
        try:
            js = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if js.get("stock_code"):
            comp[Path(p).stem] = (str(js["stock_code"]).strip(), js.get("adres"))
    moves = []
    for f in glob.glob(str(Path(settings.cache_path) / "alt" / "kr_dart" / "list" / "*" / "*.json")):
        corp = f.split(os.sep)[-2]
        if corp not in comp:
            continue
        for r in json.loads(Path(f).read_text(encoding="utf-8")).get("rows", []):
            nm = str(r.get("report_nm", ""))
            if "본점소재지변경" in nm and "정정" not in nm:
                moves.append({"stock_code": comp[corp][0], "date": pd.Timestamp(str(r["rcept_dt"]))})
    mv = pd.DataFrame(moves).drop_duplicates() if moves else pd.DataFrame(columns=["stock_code", "date"])
    last = mv.groupby("stock_code")["date"].max() if len(mv) else pd.Series(dtype="datetime64[ns]")
    rows = []
    for _corp, (code, addr) in comp.items():
        reg, dist = kr_region(addr)
        start = last.get(code, pd.Timestamp("1990-01-01"))
        rows.append({"stock_code": code, "valid_from": start + pd.Timedelta(1, "D"), "region": reg, "district": dist})
    return pd.DataFrame(rows), mv


# ================================================================================================ features

L_DOCS = {
    "loc_geo_peer_1m": (
        "본사 근처(미국 100km·한국 같은 시도) 다른 업종 회사들의 1개월 수익(시총 가중)",
        "local peers' 1-month return",
    ),
    "loc_geo_peer_gap": ("근처 회사 1개월 수익 − 자기 1개월 수익", "local peers' 1-month return minus own"),
    "loc_n_peers": ("근처 유니버스 회사 수 log(1 + n)", "log number of local universe peers"),
    "loc_dist_center": ("금융 중심지까지 거리 log(1 + km) (뉴욕·서울)", "log distance to the financial centre"),
    "loc_moved_24m": (
        "최근 24개월 본사 이전(미국 도시·주 변경, 한국 본점소재지변경 공시)",
        "HQ relocation in 24 months",
    ),
    "loc_moved_far_24m": ("최근 24개월 다른 주로 이전(미국)", "HQ moved to another state in 24 months"),
    "loc_foreign": ("본사가 미국 밖(미국)", "HQ outside the US"),
    "loc_state_ur": ("본사 주 실업률 12개월 변화(미국)", "HQ state unemployment rate, 12-month change"),
    "loc_state_ci": ("본사 주 동행지수 3개월 log 변화(미국)", "HQ state coincident index, 3-month change"),
    "loc_capital": ("수도권(서울·경기·인천) 본사(한국)", "HQ in the capital area (KR)"),
    "loc_gangnam": ("서울 강남·서초·송파 본사(한국)", "HQ in Seoul Gangnam 3 districts (KR)"),
}


def _peer_returns(frame: pd.DataFrame, lat: np.ndarray, lon: np.ndarray, group: np.ndarray | None) -> tuple:
    """Cap-weighted 1-month return of OTHER-sector members within PEER_KM (by coordinates) or in the same group."""
    ret = frame["ret_1m"].to_numpy(float) if "ret_1m" in frame else np.full(len(frame), np.nan)
    mcap = np.exp(frame["log_mcap"].to_numpy(float))
    sec = frame["sector_group"].astype(str).to_numpy()
    peer = np.full(len(frame), np.nan)
    npeer = np.full(len(frame), np.nan)
    for _, idx in frame.groupby("date").indices.items():
        idx = np.asarray(idx)
        ok = np.isfinite(ret[idx]) & np.isfinite(mcap[idx])
        if group is None:
            have = np.isfinite(lat[idx]) & np.isfinite(lon[idx])
        else:
            have = pd.notna(group[idx])
        for j, i in enumerate(idx):
            if not have[j]:
                continue
            if group is None:
                dist = haversine_km((lat[i], lon[i]), np.column_stack([lat[idx], lon[idx]]))
                near = (dist <= PEER_KM) & have
            else:
                near = (group[idx] == group[i]) & have
            near &= idx != i
            npeer[i] = float(near.sum())
            m = near & ok & (sec[idx] != sec[i])
            if m.sum() >= 3:
                w = mcap[idx][m]
                peer[i] = float(np.sum(w * ret[idx][m]) / np.sum(w))
    return peer, npeer, ret


def block_l_us(frame: pd.DataFrame, ds, settings, cache) -> tuple[pd.DataFrame, dict]:
    from us_v8_data import cik_chains

    from quantcio.data.pit.us_cik import entity_cik

    h = us_history(settings)
    by_cik = {c: g for c, g in h.groupby("cik")}
    chains = cik_chains(ds, settings, cache)
    n = len(frame)
    lat, lon = np.full(n, np.nan), np.full(n, np.nan)
    state = np.full(n, None, dtype=object)
    moved, moved_far, foreign = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    pos = {ix: i for i, ix in enumerate(frame.index)}
    matched = 0
    for ent, g in frame.groupby("entity", sort=False):
        ciks = chains.get(str(ent)) or {entity_cik(str(ent))}
        hh = (
            pd.concat([by_cik[c] for c in ciks if c in by_cik], ignore_index=True)
            if any(c in by_cik for c in ciks)
            else None
        )
        if hh is None or hh.empty:
            continue
        matched += 1
        hh = hh.sort_values("filed").drop_duplicates("filed", keep="last").reset_index(drop=True)
        # a move = the HQ coordinates jump > 50 km between consecutive filings, or the state changes (city spellings
        # vary between filings, so names are not compared)
        prev = np.column_stack([hh["lat"].shift(1).to_numpy(float), hh["lon"].shift(1).to_numpy(float)])
        cur = hh[["lat", "lon"]].to_numpy(float)
        jump = np.array(
            [
                haversine_km((p[0], p[1]), c[None, :])[0] > 50
                if np.isfinite(p).all() and np.isfinite(c).all()
                else False
                for p, c in zip(prev, cur, strict=True)
            ]
        )
        state_chg = (hh["state"] != hh["state"].shift(1)) & hh["state"].shift(1).notna()
        city_chg = pd.Series(jump, index=hh.index) | state_chg
        city_chg.iloc[0] = state_chg.iloc[0] = False
        avail = (hh["filed"] + pd.Timedelta(1, "D")).to_numpy(dtype="datetime64[ns]")
        dts = g["date"].to_numpy(dtype="datetime64[ns]")
        p = np.searchsorted(avail, dts, "right") - 1
        ok = p >= 0
        rows = np.array([pos[i] for i in g.index])
        pp = np.clip(p, 0, None)
        lat[rows] = np.where(ok, hh["lat"].to_numpy(float)[pp], np.nan)
        lon[rows] = np.where(ok, hh["lon"].to_numpy(float)[pp], np.nan)
        state[rows] = np.where(ok, hh["state"].to_numpy(object)[pp], None)
        foreign[rows] = np.where(ok, (hh["country"].to_numpy(object)[pp] != "US").astype(float), np.nan)
        for arr, chg in ((moved, city_chg), (moved_far, state_chg)):
            ch_dates = avail[chg.to_numpy()]
            out = np.array([float(((ch_dates <= d) & (ch_dates > d - np.timedelta64(730, "D"))).any()) for d in dts])
            arr[rows] = np.where(ok, out, np.nan)
    peer, npeer, ret = _peer_returns(frame, lat, lon, None)
    out = pd.DataFrame(index=frame.index)
    out["loc_geo_peer_1m"] = peer
    out["loc_geo_peer_gap"] = peer - ret
    out["loc_n_peers"] = np.log1p(npeer)
    coords = np.column_stack([lat, lon])
    out["loc_dist_center"] = np.where(np.isfinite(lat), np.log1p(haversine_km(NYC, np.nan_to_num(coords))), np.nan)
    out["loc_moved_24m"] = moved
    out["loc_moved_far_24m"] = moved_far
    out["loc_foreign"] = foreign
    # state macro (as of the row date; FRED month values available month end + 50 / + 60 days)
    fred = json.loads((kd.root(settings, "hq_us") / "state_fred.json").read_text())
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    ur, ci = np.full(n, np.nan), np.full(n, np.nan)
    st_series = pd.Series(state)
    for st, idx in st_series.groupby(st_series).groups.items():
        ii = np.asarray(list(idx))
        for key, arr, fn, lag in (
            (f"{st}UR", ur, lambda s: s - s.shift(12), 50),
            (f"{st}PHCI", ci, lambda s: np.log(s / s.shift(3)), 60),
        ):
            obs = fred.get(key)
            if not obs:
                continue
            s = pd.Series({pd.Timestamp(a): pd.to_numeric(b, errors="coerce") for a, b in obs}).sort_index()
            ch = fn(s).dropna()
            ser = kd._daily_available(
                ch.reset_index(drop=True), pd.Series(ch.index + pd.offsets.MonthEnd(0) + pd.Timedelta(lag, "D")), 70
            )
            arr[ii] = kd._asof(ser, pd.DatetimeIndex(dates[ii]))
    out["loc_state_ur"] = ur
    out["loc_state_ci"] = ci
    return out, {"hq_entities_matched": matched, "fsds_filings": int(len(h))}


def block_l_kr(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, dict]:
    addr, mv = kr_history(settings)
    a = addr.set_index("stock_code")
    n = len(frame)
    region = np.full(n, None, dtype=object)
    district = np.full(n, None, dtype=object)
    moved = np.zeros(n)
    codes = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    mv_by = {c: g["date"].to_numpy(dtype="datetime64[ns]") for c, g in mv.groupby("stock_code")} if len(mv) else {}
    for i, (c, d) in enumerate(zip(codes, dates, strict=True)):
        if c in a.index:
            row = a.loc[c]
            if d >= np.datetime64(row["valid_from"]):
                region[i], district[i] = row["region"], row["district"]
        ev = mv_by.get(c)
        if ev is not None:
            moved[i] = float(((ev < d) & (ev >= d - np.timedelta64(730, "D"))).any())
    lat = np.array([KR_REGION[r][0] if r in KR_REGION else np.nan for r in region])
    lon = np.array([KR_REGION[r][1] if r in KR_REGION else np.nan for r in region])
    peer, npeer, ret = _peer_returns(frame, lat, lon, region)
    out = pd.DataFrame(index=frame.index)
    out["loc_geo_peer_1m"] = peer
    out["loc_geo_peer_gap"] = peer - ret
    out["loc_n_peers"] = np.log1p(npeer)
    known = pd.notna(region)
    out["loc_dist_center"] = np.where(
        known, np.log1p(haversine_km(SEOUL, np.column_stack([np.nan_to_num(lat), np.nan_to_num(lon)]))), np.nan
    )
    out["loc_moved_24m"] = moved
    out["loc_capital"] = np.where(known, np.isin(region, ["서울", "경기", "인천"]).astype(float), np.nan)
    out["loc_gangnam"] = np.where(known, ((region == "서울") & np.isin(district, GANGNAM3)).astype(float), np.nan)
    info = {
        "kr_companies_with_address": int(len(a)),
        "kr_moves": int(len(mv)),
        "kr_rows_location_known": round(float(known.mean()), 3),
    }
    return out, info


def main() -> None:
    from quantcio.config import load_settings

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    ap.parse_args()
    s = load_settings()
    say(f"US HQ: {us_fetch(s)}")
    addr, mv = kr_history(s)
    say(
        f"KR HQ: {len(addr)} addresses, {len(mv)} relocation filings, "
        f"regions {addr['region'].value_counts().head(6).to_dict()}"
    )


if __name__ == "__main__":
    main()
