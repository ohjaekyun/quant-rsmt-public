"""Data for the v14 research runs (docs/MODELING.md 32, pre-declared 2026-10-08): block C (precious metals, crypto and
agricultural futures - prices and volumes), block D (dividends: KR DART, US SEC XBRL + Yahoo ex-dividend dates) and
block H (market holidays). Research only - nothing here is served.

    .venv/bin/python scripts/v14_data.py fetch --what cmdty|dart_div|all

Point-in-time rules are in docs/MODELING.md 32 (commodity / crypto values lagged one day, DART reports from the day
after receipt, XBRL from the day after filing, only ex-dividend dates <= t, holidays known in advance).
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_data as v13  # noqa: E402

say = v13.say

# ================================================================================================ block C

CMDTY = {
    "gold": "GC=F",
    "silver": "SI=F",
    "platinum": "PL=F",
    "palladium": "PA=F",
    "btc": "BTC-USD",
    "eth": "ETH-USD",
    "corn": "ZC=F",
    "wheat": "ZW=F",
    "soy": "ZS=F",
    "coffee": "KC=F",
    "sugar": "SB=F",
    "cocoa": "CC=F",
    "cotton": "CT=F",
    "cattle": "LE=F",
    "hogs": "HE=F",
}
AGRI = ["corn", "wheat", "soy", "coffee", "sugar", "cocoa", "cotton", "cattle", "hogs"]


def cmdty_fetch(settings, say=say) -> dict:
    import yfinance as yf

    d = kd.root(settings, "v14_cmdty")
    df = yf.download(
        list(CMDTY.values()), start="2004-01-01", progress=False, auto_adjust=False, group_by="ticker", threads=True
    )
    out = {}
    for name, sym in CMDTY.items():
        q = df[sym][["Close", "Volume"]].dropna(subset=["Close"]).rename(columns=str.lower)
        q.index = pd.DatetimeIndex(q.index).tz_localize(None).normalize()
        q.to_parquet(d / f"{name}.parquet")
        out[name] = (str(q.index.min().date()), str(q.index.max().date()), len(q))
    say(f"commodities/crypto: {out}")
    return out


def cmdty_load(settings) -> dict[str, pd.DataFrame]:
    d = kd.root(settings, "v14_cmdty")
    return {n: pd.read_parquet(d / f"{n}.parquet") for n in CMDTY if (d / f"{n}.parquet").exists()}


def _z3y(s: pd.Series) -> pd.Series:
    m = s.rolling(756, min_periods=252).mean()
    sd = s.rolling(756, min_periods=252).std()
    return (s - m) / sd.where(sd > 0)


def _daily_lagged(s: pd.Series, days: int = 1, max_age: int = 7) -> pd.Series:
    """Own-date series -> calendar-daily series whose value on day d uses observations <= d - ``days``."""
    s = s.dropna()
    if s.empty:
        return s
    end = max(pd.Timestamp.today().normalize(), s.index.max()) + pd.Timedelta(days, "D")
    daily = s.reindex(pd.date_range(s.index.min(), end, freq="D")).ffill(limit=max_age)
    return daily.shift(days)


def _vol_trend(v: pd.Series) -> pd.Series:
    v = v.where(v > 0)
    return np.log(v.rolling(20, min_periods=15).mean() / v.rolling(120, min_periods=90).mean())


def agri_index(cm: dict[str, pd.DataFrame]) -> pd.Series:
    """Equal-weight agricultural index from the mean daily log change of the available contracts."""
    closes = pd.DataFrame({n: cm[n]["close"] for n in AGRI if n in cm})
    closes = closes.ffill(limit=3)
    r = np.log(closes.where(closes > 0)).diff()
    return r.mean(axis=1, skipna=True).fillna(0.0).cumsum()


C_DOCS = {
    "r_gold_mom": ("금 63일 변화 z", "gold 63-day change z"),
    "r_silver_mom": ("은 63일 변화 z", "silver 63-day change z"),
    "r_pgm_mom": ("백금·팔라듐 63일 변화 z", "platinum/palladium 63-day change z"),
    "r_btc_mom": ("비트코인 63일 변화 z", "bitcoin 63-day change z"),
    "r_agri_mom": ("농산물 지수 63일 변화 z", "agricultural index 63-day change z"),
    "r_gold_silver": ("log(금 ÷ 은) z", "gold/silver ratio z"),
    "r_gold_vol": ("금 선물 거래량 추세", "gold futures volume trend"),
    "r_btc_vol": ("비트코인 거래량 추세", "bitcoin volume trend"),
    "r_agri_vol": ("농산물 선물 거래량 추세(평균)", "agricultural futures volume trend"),
    "r_btc_rv": ("비트코인 20일 실현 변동성(log)", "bitcoin 20-day realised volatility (log)"),
    "r_eth_btc": ("log(이더리움 ÷ 비트코인) 63일 변화", "ETH/BTC 63-day change"),
    "c_beta_gold": ("금 민감도(252일 부분 베타)", "gold beta (252-day partial)"),
    "c_beta_btc": ("비트코인 민감도", "bitcoin beta"),
    "c_beta_agri": ("농산물 지수 민감도", "agricultural index beta"),
    "c_impact_gold": ("금 민감도 순위 × 최근 20일 충격", "gold beta rank x 20-day shock"),
    "c_impact_btc": ("비트코인 민감도 순위 × 최근 20일 충격", "bitcoin beta rank x 20-day shock"),
    "c_impact_agri": ("농산물 민감도 순위 × 최근 20일 충격", "agri beta rank x 20-day shock"),
}


def cmdty_regime(settings) -> dict[str, pd.Series]:
    cm = cmdty_load(settings)
    out: dict[str, pd.Series] = {}

    def mom(s):
        return np.log(s.where(s > 0) / s.shift(63))

    for name, col in (("gold", "r_gold_mom"), ("silver", "r_silver_mom"), ("btc", "r_btc_mom")):
        if name in cm:
            out[col] = _z3y(mom(cm[name]["close"]))
    pg = [mom(cm[n]["close"]) for n in ("platinum", "palladium") if n in cm]
    if pg:
        out["r_pgm_mom"] = _z3y(pd.concat(pg, axis=1).mean(axis=1))
    ai = agri_index(cm)
    out["r_agri_mom"] = _z3y(ai - ai.shift(63))
    if "gold" in cm and "silver" in cm:
        gs = pd.concat([cm["gold"]["close"], cm["silver"]["close"]], axis=1, keys=["g", "s"]).dropna()
        out["r_gold_silver"] = _z3y(np.log(gs["g"] / gs["s"]))
    if "gold" in cm:
        out["r_gold_vol"] = _vol_trend(cm["gold"]["volume"])
    if "btc" in cm:
        out["r_btc_vol"] = _vol_trend(cm["btc"]["volume"])
        lr = np.log(cm["btc"]["close"]).diff()
        out["r_btc_rv"] = np.log(lr.rolling(20, min_periods=15).std() * np.sqrt(365))
    out["r_agri_vol"] = pd.concat([_vol_trend(cm[n]["volume"]) for n in AGRI if n in cm], axis=1).mean(axis=1)
    if "btc" in cm and "eth" in cm:
        eb = pd.concat([cm["eth"]["close"], cm["btc"]["close"]], axis=1, keys=["e", "b"]).dropna()
        out["r_eth_btc"] = np.log(eb["e"] / eb["b"]).diff(63)
    return {k: _daily_lagged(v.replace([np.inf, -np.inf], np.nan)) for k, v in out.items()}


def block_c(frame: pd.DataFrame, prices: dict, bench: pd.Series, market: str, settings) -> pd.DataFrame:
    """Per-stock commodity / crypto betas and beta-rank x shock (all from data <= t - 1)."""
    import us_v7_data as v7

    from quantcio.modeling import features as feat

    cm = cmdty_load(settings)
    levels = {"gold": cm["gold"]["close"], "btc": cm["btc"]["close"], "agri": np.exp(agri_index(cm))}
    # KR closes before the US day of the same date: the KR return of session d meets the commodity move of day d-1.
    # US: same-day move (gold settles before the US close; crypto's UTC day ends after it, so betas are lagged a
    # session below anyway).
    lag_days = 1 if market == "KR" else 0
    daily = {k: _daily_lagged(v, lag_days) for k, v in levels.items()}
    cols = ["c_beta_gold", "c_beta_btc", "c_beta_agri"]
    out = np.full((len(frame), 3), np.nan)
    pos = {ix: i for i, ix in enumerate(frame.index)}
    for ent, g in frame.groupby("entity", sort=False):
        px = prices.get(ent)
        if px is None or len(px) < 150 or "close" not in px:
            continue
        close = px["close"].astype(float).dropna()
        idx = close.index
        fac = pd.DataFrame({k: np.log(s.reindex(idx)).diff() for k, s in daily.items()}, index=idx)
        b = v7.partial_betas(close, bench, fac).shift(1)  # a beta at t uses returns <= t - 1
        rows = np.array([pos[i] for i in g.index])
        out[rows] = v13._asof_rows(b[["gold", "btc", "agri"]], g["date"].to_numpy(dtype="datetime64[ns]"), 7)
    res = pd.DataFrame(out, index=frame.index, columns=cols)
    dates = pd.DatetimeIndex(frame["date"])
    for k in ("gold", "btc", "agri"):
        lv = _daily_lagged(levels[k], 1)
        ch = np.log(lv / lv.shift(28))  # ~20 sessions
        sd = ch.rolling(1095, min_periods=365).std()
        shock = (ch / sd.where(sd > 0)).clip(-3, 3) / 3.0
        rk = feat.rank_gauss(res[f"c_beta_{k}"], frame["date"]).to_numpy(float)
        res[f"c_impact_{k}"] = rk * kd._asof(shock, dates)
    return res


# ================================================================================================ block D: KR (DART)

D_DOCS = {
    "kr_div_yield": ("현금배당 총액 ÷ 시가총액(DART)", "cash dividends / market cap (DART)"),
    "kr_div_payout": ("현금배당성향", "cash payout ratio"),
    "kr_div_growth": ("현금배당 총액 증가(log, 당기 vs 전기)", "cash dividend growth (log)"),
    "kr_div_years": ("최근 3개 사업연도 중 현금배당 연수", "dividend-paying years of the last 3"),
    "kr_div_cut": ("배당 삭감(당기 < 전기 × 0.9)", "dividend cut"),
    "kr_div_init": ("배당 개시(전기 0 → 당기 > 0)", "dividend initiation"),
    "us_div_growth": ("배당 지급액 증가(log, XBRL)", "dividends paid growth (log, XBRL)"),
    "us_div_payout": ("배당 지급액 ÷ 순이익", "dividends paid / net income"),
    "us_div_streak": ("연속 배당 회계연도 수 ÷ 10", "consecutive dividend years / 10"),
    "us_div_cut": ("배당 삭감", "dividend cut"),
    "us_div_init": ("배당 개시", "dividend initiation"),
    "us_exdiv_soon": ("다음 배당락 예상일이 28일 안", "next expected ex-date within 28 days"),
    "us_exdiv_recent": ("최근 28일 안에 배당락", "ex-dividend date in the last 28 days"),
}


def _dart_num(x) -> float:
    s = str(x or "").replace(",", "").strip()
    if s in ("", "-"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def dart_div_fetch(settings, codes: list[str], say=say, first_year: int = 2015) -> dict:
    """DART 'alotMatter' (배당에 관한 사항, annual report) per company and business year -> one json per code."""
    from quantcio.data.pit.kr_dart import OPENDART_API, opendart_corp_codes

    k = kd.key("DART_API_KEY")
    if not k:
        return {"skipped": "DART_API_KEY 없음"}
    d = kd.root(settings, "dart_div")
    sess = requests.Session()
    cpath = Path(settings.cache_path) / "pit" / "kr" / "opendart" / "corp_codes.pkl"  # kept by data/pit/kr_fund.py
    if cpath.exists():
        with open(cpath, "rb") as fh:
            cc = pickle.load(fh)[1]
    else:
        cc = opendart_corp_codes(sess, k)
    last_year = pd.Timestamp.today().year - 1
    n_calls, n_codes, missing = 0, 0, []

    def call(corp: str, y: int) -> dict:
        for attempt in range(40):
            try:
                r = sess.get(
                    f"{OPENDART_API}/alotMatter.json",
                    params={"crtfc_key": k, "corp_code": corp, "bsns_year": str(y), "reprt_code": "11011"},
                    timeout=60,
                )
                js = r.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt >= 5:
                    raise RuntimeError(kd._redact(f"DART alotMatter {corp} {y}: {exc}", k)) from None
                time.sleep(3.0 * (attempt + 1))
                continue
            if js.get("status") == "800":  # 시스템 점검: wait it out (up to ~6.5 hours)
                say("DART 시스템 점검(800) — 10분 뒤 다시 시도")
                time.sleep(600)
                continue
            return js
        raise RuntimeError("DART 점검이 끝나지 않음")

    for i, code in enumerate(sorted(codes), 1):
        corp = cc.get(code)
        if not corp:
            missing.append(code)
            continue
        p = d / f"{code}.json"
        have = json.loads(p.read_text()) if p.exists() else {}
        for y in range(first_year, last_year + 1):
            if str(y) in have and (have[str(y)] or y < last_year):
                continue
            js = call(corp, y)
            n_calls += 1
            st = js.get("status")
            if st == "020":
                raise RuntimeError("DART 일일 한도(020) 도달 — 내일 다시 실행")
            have[str(y)] = js.get("list", []) if st == "000" else []
            time.sleep(0.12)
        p.write_text(json.dumps(have, ensure_ascii=False))
        n_codes += 1
        if i % 50 == 0:
            say(f"DART dividends {i}/{len(codes)} codes, {n_calls} calls")
    say(f"DART dividends done: {n_codes} codes, {n_calls} calls, no corp code {len(missing)}")
    return {"codes": n_codes, "calls": n_calls, "no_corp_code": missing}


def dart_div_table(settings, code: str) -> pd.DataFrame:
    """Rows (avail, total_th, total_fr, total_lw, payout) per annual report of ``code``."""
    p = kd.root(settings, "dart_div") / f"{code}.json"
    if not p.exists():
        return pd.DataFrame()
    recs = []
    for _y, rows in json.loads(p.read_text()).items():
        if not rows:
            continue
        rc = str(rows[0].get("rcept_no") or "")
        if len(rc) < 8:
            continue
        avail = pd.Timestamp(rc[:8]) + pd.Timedelta(1, "D")
        tot = next((r for r in rows if str(r.get("se", "")).startswith("현금배당금총액")), None)
        pay = next((r for r in rows if "현금배당성향" in str(r.get("se", ""))), None)
        if tot is None:
            continue
        recs.append(
            {
                "avail": avail,
                "th": _dart_num(tot.get("thstrm")),
                "fr": _dart_num(tot.get("frmtrm")),
                "lw": _dart_num(tot.get("lwfr")),
                "payout": _dart_num(pay.get("thstrm")) / 100.0 if pay is not None else np.nan,
            }
        )
    if not recs:
        return pd.DataFrame()
    return pd.DataFrame(recs).sort_values("avail").drop_duplicates("avail", keep="last").reset_index(drop=True)


def block_d_kr(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, dict]:
    mcap = np.exp(frame["log_mcap"].astype(float))
    cols = list(c for c in D_DOCS if c.startswith("kr_"))
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    n = 0
    for ent, g in frame.groupby("entity", sort=False):
        t = dart_div_table(settings, str(ent))
        if t.empty:
            continue
        n += 1
        f = pd.DataFrame(
            {
                "total": t["th"] * 1e6,
                "payout": t["payout"].clip(0, 3),
                "growth": np.log1p(t["th"].clip(lower=0)) - np.log1p(t["fr"].clip(lower=0)),
                "years": (t[["th", "fr", "lw"]] > 0).sum(axis=1).astype(float),
                "cut": ((t["fr"] > 0) & (t["th"] < 0.9 * t["fr"])).astype(float),
                "init": ((t["fr"] <= 0) & (t["th"] > 0)).astype(float),
            }
        )
        f.index = pd.DatetimeIndex(t["avail"])
        v = v13._asof_rows(f, g["date"].to_numpy(dtype="datetime64[ns]"), 500)
        out.loc[g.index, "kr_div_yield"] = v[:, 0] / mcap[g.index].to_numpy()
        for j, c in enumerate(["kr_div_payout", "kr_div_growth", "kr_div_years", "kr_div_cut", "kr_div_init"], 1):
            out.loc[g.index, c] = v[:, j]
    return out, {"dart_div_codes": n}


# ================================================================================================ block D: US


def _yahoo_dividends() -> dict[str, pd.DatetimeIndex | None]:
    """Yahoo symbol -> ex-dividend dates from the PIT price cache (longest cached chart per symbol); None when the
    cached chart failed."""
    from quantcio.config import load_settings

    root = Path(load_settings().cache_path) / "pit" / "us" / "yahoo_v1"
    best: dict[str, tuple[str, str]] = {}
    for p in glob.glob(str(root / "*__*.pkl")):
        name = Path(p).name[:-4]
        sym, start = name.rsplit("__", 1)
        if sym not in best or start < best[sym][1]:
            best[sym] = (p, start)
    out: dict[str, pd.DatetimeIndex | None] = {}
    for sym, (p, _s) in best.items():
        try:
            with open(p, "rb") as fh:
                h = pickle.load(fh)
        except Exception:  # noqa: BLE001 - unreadable cache entry
            continue
        if not h.get("ok"):
            out[sym] = None
            continue
        dv = h.get("dividends")
        out[sym] = pd.DatetimeIndex(dv.index).sort_values() if dv is not None and len(dv) else pd.DatetimeIndex([])
    return out


def _exdiv_features(ex: pd.DatetimeIndex, dates: np.ndarray) -> np.ndarray:
    """(soon, recent) per date from ex-dates <= date only."""
    out = np.zeros((len(dates), 2))
    arr = ex.to_numpy(dtype="datetime64[ns]")
    for i, d in enumerate(dates):
        k = np.searchsorted(arr, d, "right")
        if k == 0:
            continue
        last = arr[k - 1]
        out[i, 1] = float((d - last) < np.timedelta64(28, "D"))
        if k >= 2:
            gaps = np.diff(arr[max(0, k - 5) : k]).astype("timedelta64[D]").astype(float)
            gap = float(np.median(gaps[-4:]))
            if gap <= 400:
                nxt = last + np.timedelta64(int(round(gap)), "D")
                out[i, 0] = float(d < nxt <= d + np.timedelta64(28, "D"))
    return out


def block_d_us(frame: pd.DataFrame, ds, settings, cache) -> tuple[pd.DataFrame, dict]:
    import us_v10_data as m
    from us_v8_data import cik_chains

    from quantcio.data.pit.us_cik import entity_cik
    from quantcio.data.pit.us_net import PitNet

    cols = [c for c in D_DOCS if c.startswith("us_")]
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    chains = cik_chains(ds, settings, cache)
    net = PitNet(settings)
    ciks = sorted({c for cs in chains.values() for c in cs})
    facts = m.fetch_facts_m(net, ciks)
    n_x = 0
    for ent, g in frame.groupby("entity", sort=False):
        merged: dict[str, list] = {}
        for c in chains.get(str(ent), {entity_cik(str(ent))}):
            for k, v in ((facts.get(c) or {}).get("facts") or {}).items():
                merged.setdefault(k, []).extend(v)
        t = m.annual_table(merged) if merged else pd.DataFrame()
        if t.empty or "dividends" not in t:
            continue
        n_x += 1
        div = t["dividends"].fillna(0.0).clip(lower=0)
        prev = div.shift(1)
        paid = div > 0
        streak = paid.astype(int).groupby((~paid).cumsum()).cumsum().clip(upper=10)
        f = pd.DataFrame(
            {
                "growth": (np.log1p(div / 1e6) - np.log1p(prev / 1e6)),
                "payout": (div / t["net_income"].where(t["net_income"] > 0)).clip(0, 3),
                "streak": streak / 10.0,
                "cut": ((prev > 0) & (div < 0.9 * prev)).astype(float).where(prev.notna()),
                "init": ((prev <= 0) & (div > 0)).astype(float).where(prev.notna()),
            }
        )
        f.index = pd.DatetimeIndex(t["filed"] + pd.Timedelta(1, "D"))
        f = f[~f.index.duplicated(keep="last")].sort_index()
        v = v13._asof_rows(f, g["date"].to_numpy(dtype="datetime64[ns]"), 500)
        for j, c in enumerate(["us_div_growth", "us_div_payout", "us_div_streak", "us_div_cut", "us_div_init"]):
            out.loc[g.index, c] = v[:, j]
    yd = _yahoo_dividends()
    sym = frame["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    n_y = 0
    for s, idx in sym.groupby(sym).groups.items():
        ex = yd.get(s)
        if ex is None:
            continue
        n_y += 1
        ii = frame.index.get_indexer(idx)
        v = _exdiv_features(ex, dates[ii])
        out.iloc[ii, out.columns.get_loc("us_exdiv_soon")] = v[:, 0]
        out.iloc[ii, out.columns.get_loc("us_exdiv_recent")] = v[:, 1]
    return out, {"xbrl_entities": n_x, "yahoo_symbols": n_y}


# ================================================================================================ block H

ADHOC = {  # closure -> announcement date (unknown before it)
    "US": {
        "2012-10-29": "2012-10-28",
        "2012-10-30": "2012-10-29",
        "2018-12-05": "2018-12-01",
        "2025-01-09": "2024-12-30",
    },
    "KR": {
        "2015-08-14": "2015-08-04",
        "2016-05-06": "2016-04-26",
        "2017-10-02": "2017-09-05",
        "2020-08-17": "2020-07-21",
        "2023-10-02": "2023-09-05",
        "2024-10-01": "2024-09-03",
        "2025-01-27": "2025-01-08",
    },
}
H_DOCS = {
    "r_hol_n28": ("앞으로 28일 안 평일 휴장 수", "weekday market holidays in the next 28 days"),
    "r_hol_long": ("앞으로 28일 안 가장 긴 연속 휴장(주말 포함)", "longest closed streak in the next 28 days"),
    "r_hol_to": ("다음 평일 휴장까지 일수 ÷ 60", "days to the next weekday holiday / 60"),
    "r_hol_past": ("지난 28일 안 평일 휴장 수", "weekday holidays in the last 28 days"),
    "h_cons_hol": ("소비재 업종 × 앞으로 28일 휴장 수", "consumer sectors x holidays ahead"),
    "h_trav_hol": ("운송 업종 × 앞으로 28일 휴장 수", "transport x holidays ahead"),
    "h_cons_long": ("소비재 업종 × 가장 긴 연속 휴장", "consumer sectors x longest closure ahead"),
}
CONSUMER = {"경기소비재", "필수소비재"}
TRANSPORT = {"운송"}


def holiday_regime(calendar: pd.DatetimeIndex, market: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """r_hol_* for each date in ``dates`` from the realised session calendar (ad-hoc closures only after their
    announcement)."""
    cal = pd.DatetimeIndex(calendar).normalize().unique().sort_values()
    lo, hi = cal.min(), cal.max()
    days = pd.date_range(lo, hi, freq="D")
    closed = ~days.isin(cal)
    weekday_hol = closed & (days.weekday < 5)
    hol_days = days[weekday_hol]
    adhoc = {pd.Timestamp(k): pd.Timestamp(v) for k, v in ADHOC.get(market, {}).items()}
    closed_s = pd.Series(closed, index=days)
    out = pd.DataFrame(index=pd.DatetimeIndex(dates).unique().sort_values(), columns=list(H_DOCS)[:4], dtype=float)
    for t in out.index:
        if t < lo or t > hi:
            continue
        end28, end60 = t + pd.Timedelta(28, "D"), t + pd.Timedelta(60, "D")
        known = [h for h in hol_days if t < h <= end60 and (h not in adhoc or adhoc[h] <= t)]
        if end28 <= hi:
            n28 = [h for h in known if h <= end28]
            out.loc[t, "r_hol_n28"] = len(n28)
            win = closed_s.loc[t + pd.Timedelta(1, "D") : end28].copy()
            for h, a in adhoc.items():  # an unannounced closure still looks like a session at t
                if h in win.index and a > t:
                    win.loc[h] = False
            run, best = 0, 0
            for v in win.to_numpy():
                run = run + 1 if v else 0
                best = max(best, run)
            out.loc[t, "r_hol_long"] = best
        if known:
            out.loc[t, "r_hol_to"] = min((known[0] - t).days / 60.0, 1.0)
        elif end60 <= hi:
            out.loc[t, "r_hol_to"] = 1.0
        out.loc[t, "r_hol_past"] = int(((hol_days > t - pd.Timedelta(28, "D")) & (hol_days <= t)).sum())
    return out


def block_h(frame: pd.DataFrame, calendar: pd.DatetimeIndex, market: str) -> pd.DataFrame:
    reg = holiday_regime(calendar, market, pd.DatetimeIndex(frame["date"]))
    out = reg.reindex(pd.DatetimeIndex(frame["date"])).set_axis(frame.index)
    sec = frame["sector_group"].astype(str)
    cons, trav = sec.isin(CONSUMER).astype(float), sec.isin(TRANSPORT).astype(float)
    out["h_cons_hol"] = cons * out["r_hol_n28"]
    out["h_trav_hol"] = trav * out["r_hol_n28"]
    out["h_cons_long"] = cons * out["r_hol_long"]
    return out


# ================================================================================================ CLI


def main() -> None:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--what", choices=["cmdty", "dart_div", "all"], default="all")
    a = ap.parse_args()
    settings = load_settings()
    if a.what in ("cmdty", "all"):
        cmdty_fetch(settings)
    if a.what in ("dart_div", "all"):
        spans, _ = v13._spans("KR", settings, DiskCache(settings.cache_path))
        say(f"DART dividends: {dart_div_fetch(settings, sorted(spans))}")


if __name__ == "__main__":
    main()
