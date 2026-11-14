"""Data for the v13 research runs (docs/MODELING.md 31, pre-declared 2026-10-07): block T (chart / trend indicators),
block F (investor flows: KR foreign / institution / individual / pension, US 13F institutions, SEC MIDAS algorithmic
trading metrics, Fed balance sheet) and block S (social statistics). Research only - nothing here is served.

    .venv/bin/python scripts/v13_data.py fetch --what naver|krx|f13|midas|social|all

Every value is point in time: a feature at date t only uses data published by t (rules in docs/MODELING.md 31).
Downloads are cached under <cache>/alt/keyed/<source>; big SEC zips are reduced to small parquet files and deleted.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402

Say = Callable[[str], None]
FRESH_DAYS = 7  # a daily value older than this at the row date counts as missing (halt / feed gap)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _asof_rows(series_by_date: pd.DataFrame, dates: np.ndarray, max_age_days: int) -> np.ndarray:
    """Rows of ``series_by_date`` (DatetimeIndex, sorted) as of each date, NaN when older than max_age_days."""
    idx = series_by_date.index.to_numpy(dtype="datetime64[ns]")
    vals = series_by_date.to_numpy(float)
    out = np.full((len(dates), vals.shape[1]), np.nan)
    if not len(idx):
        return out
    p = np.searchsorted(idx, dates, "right") - 1
    ok = (p >= 0) & ((dates - idx[np.clip(p, 0, None)]) <= np.timedelta64(max_age_days, "D"))
    out[ok] = vals[p[ok]]
    return out


# ================================================================================================ block T

T_DOCS: dict[str, tuple[str, str]] = {
    "t_bb_pctb": ("볼린저 %B(20일, 2σ)", "Bollinger %B (20, 2 sigma)"),
    "t_bb_width": ("볼린저 밴드 폭 log(4σ ÷ 20일선)", "Bollinger band width (log)"),
    "t_bb_squeeze": ("밴드 폭 ÷ 과거 126일 중앙값(log, 수축)", "band width vs its 126-day median (squeeze)"),
    "t_ma20_dist": ("20일선 이격도", "distance to the 20-day MA"),
    "t_ma_cross": ("50일선 ÷ 200일선 − 1 (골든·데드크로스 상태)", "50-day MA / 200-day MA - 1"),
    "t_cross_age": ("마지막 50/200일선 교차 이후 경과(골든 +, 데드 −)", "age of the last 50/200 cross (signed)"),
    "t_ma_align": ("정배열 점수(C·5·20·60·120일선, −4~+4)", "MA alignment score (-4..+4)"),
    "t_ma60_slope": ("60일선 20거래일 기울기", "20-session slope of the 60-day MA"),
    "t_macd": ("MACD(12,26) ÷ 종가", "MACD(12,26) / close"),
    "t_macd_hist": ("MACD 히스토그램(시그널 9) ÷ 종가", "MACD histogram / close"),
    "t_stoch_k": ("스토캐스틱 %K(14)", "stochastic %K (14)"),
    "t_adx": ("ADX(14) ÷ 100 (추세 강도)", "ADX(14) / 100"),
    "t_di_diff": ("(+DI − −DI) ÷ 100", "(+DI - -DI) / 100"),
    "t_atr_pct": ("ATR(14) ÷ 종가", "ATR(14) / close"),
    "t_donchian": ("55일 채널 안 위치", "position in the 55-day channel"),
    "t_breakout": ("종가 ÷ 직전 55일 최고가 − 1", "close / prior 55-day high - 1"),
    "t_low52": ("종가 ÷ 252일 최저가 − 1", "close / 252-day low - 1"),
    "t_obv_slope": ("OBV 20일 변화 ÷ 20일 거래량", "OBV 20-day change / 20-day volume"),
    "t_mfi": ("MFI(14) ÷ 100", "money flow index (14) / 100"),
    "t_cci": ("CCI(20) ÷ 100", "CCI(20) / 100"),
    "t_vwap_dist": ("20일 VWAP 이격도", "distance to the 20-day VWAP"),
    "t_cloud": ("일목 구름 상단 대비 위치(선행스팬은 26일 전 계산값)", "position vs the Ichimoku cloud top"),
}
T_COLS = list(T_DOCS)


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _mean_abs_dev(s: pd.Series, n: int) -> pd.Series:
    a = s.to_numpy(float)
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        w = np.lib.stride_tricks.sliding_window_view(a, n)
        out[n - 1 :] = np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1)
    return pd.Series(out, index=s.index)


def tech_daily(px: pd.DataFrame) -> pd.DataFrame:
    """Daily block-T indicators of one stock (bars <= t only; rolling windows need full history)."""
    q = px.dropna(subset=["close"]).sort_index()
    c = q["close"].astype(float)
    h = q["high"].astype(float).fillna(c) if "high" in q else c
    lo = q["low"].astype(float).fillna(c) if "low" in q else c
    v = q["volume"].astype(float).fillna(0.0) if "volume" in q else pd.Series(0.0, index=c.index)
    ma = {n: c.rolling(n, min_periods=n).mean() for n in (5, 20, 50, 60, 120, 200)}
    sd = c.rolling(20, min_periods=20).std(ddof=0).where(lambda s: s > 0)
    width = 4 * sd / ma[20]
    out = pd.DataFrame(index=c.index)
    out["t_bb_pctb"] = (c - (ma[20] - 2 * sd)) / (4 * sd)
    out["t_bb_width"] = np.log(width)
    out["t_bb_squeeze"] = np.log(width / width.rolling(126, min_periods=63).median())
    out["t_ma20_dist"] = c / ma[20] - 1
    out["t_ma_cross"] = ma[50] / ma[200] - 1
    state = np.sign(ma[50] - ma[200])
    run = state.groupby((state != state.shift()).cumsum()).cumcount() + 1
    out["t_cross_age"] = (state * np.minimum(run / 252.0, 1.0)).where(ma[200].notna())
    align = np.sign(c - ma[5]) + np.sign(ma[5] - ma[20]) + np.sign(ma[20] - ma[60]) + np.sign(ma[60] - ma[120])
    out["t_ma_align"] = align.where(ma[120].notna())
    out["t_ma60_slope"] = ma[60] / ma[60].shift(20) - 1
    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    out["t_macd"] = macd / c
    out["t_macd_hist"] = (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / c
    hh14, ll14 = h.rolling(14, min_periods=14).max(), lo.rolling(14, min_periods=14).min()
    out["t_stoch_k"] = (c - ll14) / (hh14 - ll14).where(lambda s: s > 0)
    cp = c.shift(1)
    tr = pd.concat([h - lo, (h - cp).abs(), (lo - cp).abs()], axis=1).max(axis=1)
    up, dn = h.diff(), -lo.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    atr = _wilder(tr, 14)
    pdi, ndi = 100 * _wilder(pdm, 14) / atr, 100 * _wilder(ndm, 14) / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).where(lambda s: s > 0)
    out["t_adx"] = _wilder(dx, 14) / 100
    out["t_di_diff"] = (pdi - ndi) / 100
    out["t_atr_pct"] = atr / c
    hh55, ll55 = h.rolling(55, min_periods=55).max(), lo.rolling(55, min_periods=55).min()
    out["t_donchian"] = (c - ll55) / (hh55 - ll55).where(lambda s: s > 0)
    out["t_breakout"] = c / h.shift(1).rolling(55, min_periods=55).max() - 1
    out["t_low52"] = c / lo.rolling(252, min_periods=200).min() - 1
    obv = (np.sign(c.diff()).fillna(0.0) * v).cumsum()
    out["t_obv_slope"] = (obv - obv.shift(20)) / v.rolling(20, min_periods=20).sum().where(lambda s: s > 0)
    tp = (h + lo + c) / 3
    mf = tp * v
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(14, min_periods=14).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(14, min_periods=14).sum()
    tot = (pos + neg).where(lambda s: s > 0)
    out["t_mfi"] = pos / tot
    out["t_cci"] = (
        (tp - tp.rolling(20, min_periods=20).mean()) / (0.015 * _mean_abs_dev(tp, 20)).where(lambda s: s > 0) / 100
    )
    out["t_vwap_dist"] = c / (mf.rolling(20, min_periods=20).sum() / v.rolling(20, min_periods=20).sum()) - 1
    conv = (h.rolling(9, min_periods=9).max() + lo.rolling(9, min_periods=9).min()) / 2
    base = (h.rolling(26, min_periods=26).max() + lo.rolling(26, min_periods=26).min()) / 2
    span_a = ((conv + base) / 2).shift(26)
    span_b = ((h.rolling(52, min_periods=52).max() + lo.rolling(52, min_periods=52).min()) / 2).shift(26)
    out["t_cloud"] = (c - np.maximum(span_a, span_b)) / c
    return out.replace([np.inf, -np.inf], np.nan)[T_COLS]


def block_t(frame: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """Block T at every (date, entity) row of ``frame``: the latest bar <= date, at most FRESH_DAYS old."""
    out = np.full((len(frame), len(T_COLS)), np.nan)
    pos = {ix: i for i, ix in enumerate(frame.index)}
    for ent, g in frame.groupby("entity", sort=False):
        px = prices.get(ent)
        if px is None or len(px) < 30 or "close" not in px:
            continue
        td = tech_daily(px)
        rows = np.array([pos[i] for i in g.index])
        out[rows] = _asof_rows(td, g["date"].to_numpy(dtype="datetime64[ns]"), FRESH_DAYS)
    return pd.DataFrame(out, index=frame.index, columns=T_COLS)


# ================================================================================================ block F: KR (Naver)

NAVER_TREND = "https://m.stock.naver.com/api/stock/{code}/trend"
NAVER_UA = {"User-Agent": "Mozilla/5.0"}
NAVER_PAGE = 60  # the endpoint's maximum page size


def _num(x) -> float:
    s = str(x).replace(",", "").replace("%", "").replace("+", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _naver_page(code: str, before: pd.Timestamp, session: requests.Session) -> list[dict]:
    """Up to 60 daily rows strictly before ``before`` (newest first)."""
    for attempt in range(6):
        try:
            r = session.get(
                NAVER_TREND.format(code=code),
                params={"pageSize": NAVER_PAGE, "bizdate": before.strftime("%Y%m%d")},
                headers=NAVER_UA,
                timeout=30,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code in (404, 400):
                return []
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"naver trend {code} {before.date()}: repeated failures")


def _naver_rows(js: list[dict]) -> pd.DataFrame:
    rows = []
    for d in js:
        vol = _num(d.get("accumulatedTradingVolume"))
        rows.append(
            {
                "date": pd.Timestamp(d["bizdate"]),
                "frgn": _num(d.get("foreignerPureBuyQuant")),
                "inst": _num(d.get("organPureBuyQuant")),
                "indiv": _num(d.get("individualPureBuyQuant")),
                "hold": _num(d.get("foreignerHoldRatio")),
                "close": _num(d.get("closePrice")),
                "volume": vol,
            }
        )
    return pd.DataFrame(rows)


def naver_fetch(settings, spans: dict[str, tuple[pd.Timestamp, pd.Timestamp]], say: Say = say, rate: float = 4.0):
    """Cache Naver investor flows per code for [lo, hi] (only the part not cached yet)."""
    d = kd.root(settings, "naver_flows")
    sess = requests.Session()
    n_req, t0 = 0, time.monotonic()
    for k, (code, (lo, hi)) in enumerate(sorted(spans.items())):
        p = d / f"{code}.parquet"
        have = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["date"])
        segs = []
        if have.empty:
            segs.append((lo, hi))
        else:
            hmin, hmax = pd.Timestamp(have["date"].min()), pd.Timestamp(have["date"].max())
            if lo < hmin - pd.Timedelta(5, "D"):
                segs.append((lo, hmin))
            if hi > hmax + pd.Timedelta(5, "D"):
                segs.append((hmax, hi))
        parts = [have] if not have.empty else []
        for a, b in segs:
            before = b + pd.Timedelta(1, "D")
            while before > a:
                js = _naver_page(code, before, sess)
                n_req += 1
                time.sleep(max(0.0, n_req / rate - (time.monotonic() - t0)))
                if not js:
                    break
                df = _naver_rows(js)
                parts.append(df)
                nxt = df["date"].min()
                if nxt >= before:
                    break
                before = nxt
        if segs:
            allr = pd.concat(parts, ignore_index=True).drop_duplicates("date").sort_values("date")
            allr.to_parquet(p, index=False)
        if (k + 1) % 25 == 0:
            say(f"naver flows {k + 1}/{len(spans)} codes, {n_req} requests")
    say(f"naver flows done: {len(spans)} codes, {n_req} requests")


def _naver_daily(settings, code: str) -> pd.DataFrame | None:
    p = kd.root(settings, "naver_flows") / f"{code}.parquet"
    if not p.exists():
        return None
    q = pd.read_parquet(p).sort_values("date").set_index("date")
    halt = q["volume"].isna() | (q["volume"] <= 0)
    q.loc[halt, ["frgn", "inst", "indiv"]] = 0.0  # no trading = no net buying
    q.loc[halt | (q["hold"] <= 0), "hold"] = np.nan
    return q


KR_F_DOCS = {
    "kr_frgn_net_20": ("외국인 순매수 금액 20일 합 ÷ 시가총액", "foreign net buying, 20 days / market cap"),
    "kr_frgn_net_60": ("외국인 순매수 금액 60일 합 ÷ 시가총액", "foreign net buying, 60 days / market cap"),
    "kr_inst_net_20": ("기관 순매수 금액 20일 합 ÷ 시가총액", "institutional net buying, 20 days / market cap"),
    "kr_inst_net_60": ("기관 순매수 금액 60일 합 ÷ 시가총액", "institutional net buying, 60 days / market cap"),
    "kr_indiv_net_20": ("개인 순매수 금액 20일 합 ÷ 시가총액", "retail net buying, 20 days / market cap"),
    "kr_smart_net_20": ("외국인 + 기관 순매수 20일 ÷ 시가총액", "foreign + institutional net buying, 20 days"),
    "kr_frgn_hold": ("외국인 보유율(%)", "foreign ownership (%)"),
    "kr_frgn_hold_chg": ("외국인 보유율 60거래일 변화(%p)", "foreign ownership change, 60 sessions (pp)"),
    "kr_frgn_persist": ("최근 20거래일 중 외국인 순매수일 비율", "share of foreign net-buy days, 20 sessions"),
    "kr_pension_net_20": ("연기금 등 순매수 20일 합 ÷ 시가총액(KRX)", "pension funds & government net buying, 20 days"),
    "kr_pension_net_60": ("연기금 등 순매수 60일 합 ÷ 시가총액(KRX)", "pension funds & government net buying, 60 days"),
}


def kr_flow_daily(q: pd.DataFrame) -> pd.DataFrame:
    """Daily flow sums in KRW (shifted one session: the value at day t uses flows up to t-1)."""
    out = pd.DataFrame(index=q.index)
    for col in ("frgn", "inst", "indiv"):
        val = q[col] * q["close"]
        out[f"{col}_20"] = val.rolling(20, min_periods=15).sum()
        out[f"{col}_60"] = val.rolling(60, min_periods=45).sum()
    out["hold"] = q["hold"]
    out["hold_chg"] = q["hold"] - q["hold"].shift(60)
    out["persist"] = (q["frgn"] > 0).astype(float).rolling(20, min_periods=15).mean()
    return out.shift(1)


def block_f_kr(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, dict]:
    mcap = np.exp(frame["log_mcap"].astype(float)).to_numpy()
    cols = ["frgn_20", "frgn_60", "inst_20", "inst_60", "indiv_20", "hold", "hold_chg", "persist"]
    raw = np.full((len(frame), len(cols)), np.nan)
    pos = {ix: i for i, ix in enumerate(frame.index)}
    n_codes = 0
    for ent, g in frame.groupby("entity", sort=False):
        q = _naver_daily(settings, str(ent))
        if q is None or q.empty:
            continue
        n_codes += 1
        daily = kr_flow_daily(q)[cols]
        rows = np.array([pos[i] for i in g.index])
        raw[rows] = _asof_rows(daily, g["date"].to_numpy(dtype="datetime64[ns]"), FRESH_DAYS)
    r = pd.DataFrame(raw, index=frame.index, columns=cols)
    out = pd.DataFrame(index=frame.index)
    for k in ("frgn_20", "frgn_60", "inst_20", "inst_60", "indiv_20"):
        name = {"frgn": "kr_frgn_net", "inst": "kr_inst_net", "indiv": "kr_indiv_net"}[k.split("_")[0]]
        out[f"{name}_{k.split('_')[1]}"] = r[k] / mcap
    out["kr_smart_net_20"] = (r["frgn_20"] + r["inst_20"]) / mcap
    out["kr_frgn_hold"] = r["hold"]
    out["kr_frgn_hold_chg"] = r["hold_chg"]
    out["kr_frgn_persist"] = r["persist"]
    # KRX pension / government flows (only with KRX_ID / KRX_PW)
    pen = np.full((len(frame), 2), np.nan)
    n_pen = 0
    for ent, g in frame.groupby("entity", sort=False):
        p = kd.root(settings, "krx_flows") / f"{ent}.parquet"
        if not p.exists():
            continue
        q = pd.read_parquet(p).sort_values("date").set_index("date")
        if "pension" not in q:
            continue
        n_pen += 1
        dd = pd.DataFrame(
            {
                "p20": q["pension"].rolling(20, min_periods=15).sum(),
                "p60": q["pension"].rolling(60, min_periods=45).sum(),
            }
        ).shift(1)
        rows = np.array([pos[i] for i in g.index])
        pen[rows] = _asof_rows(dd, g["date"].to_numpy(dtype="datetime64[ns]"), FRESH_DAYS)
    if n_pen:
        out["kr_pension_net_20"] = pen[:, 0] / mcap
        out["kr_pension_net_60"] = pen[:, 1] / mcap
    # regime: value-weighted universe flows (members on the date)
    reg = pd.DataFrame(index=frame.index)
    for col, src in (("r_kr_frgn_flow", "kr_frgn_net_20"), ("r_kr_inst_flow", "kr_inst_net_20")):
        num = (out[src] * mcap).groupby(frame["date"]).transform("sum")
        den = (
            pd.Series(np.where(out[src].notna(), mcap, 0.0), index=frame.index).groupby(frame["date"]).transform("sum")
        )
        reg[col] = num / den.where(den > 0)
    prog = krx_program_regime(settings)
    if prog is not None:
        for c in prog.columns:
            reg[c] = _asof_rows(prog[[c]], frame["date"].to_numpy(dtype="datetime64[ns]"), FRESH_DAYS)[:, 0]
    info = {"naver_codes": n_codes, "krx_pension_codes": n_pen, "krx_program": prog is not None}
    return pd.concat([out, reg], axis=1), info


# ======================================================================================= block F: KR (KRX login)


KRX_PAUSE = 2.0  # seconds between KRX requests (a faster pace got the IP 'Access Temporarily Restricted')


def krx_ready() -> bool:
    return bool(kd.key("KRX_JSESSIONID") or (kd.key("KRX_ID") and kd.key("KRX_PW")))


def _krx_stock():
    """pykrx ``stock`` with a KRX session: the browser's JSESSIONID (KRX_JSESSIONID) when set - accounts whose script
    login fails (CD005 for an ID that exists, e.g. one first created with a Google sign-up) - else KRX_ID / KRX_PW."""
    kd.load_env()  # pykrx reads KRX_ID / KRX_PW from the environment when it is imported
    cookie = kd.key("KRX_JSESSIONID")
    saved = {}
    if cookie:  # keep pykrx from trying (and printing) an ID/PW login at import
        saved = {k: os.environ.pop(k) for k in ("KRX_ID", "KRX_PW") if k in os.environ}
    from pykrx import stock
    from pykrx.website.comm import auth

    if cookie:
        s = auth.KRXSession(is_authenticated=True, expiry_time=time.time() + 30 * 86400)
        s.cookies = {
            "JSESSIONID": {"value": cookie, "domain": "data.krx.co.kr", "path": "/", "secure": True, "expires": 0}
        }
        auth.set_auth_session(s)
        os.environ.update(saved)
    return stock


def _krx_alive(stock) -> bool:
    """A known-good request (Samsung Electronics, January 2024) returns rows only with a valid session."""
    df = stock.get_market_trading_value_by_date("20240102", "20240131", "005930", detail=True)
    return df is not None and len(df) > 0


def krx_fetch(settings, spans: dict[str, tuple[pd.Timestamp, pd.Timestamp]], say: Say = say) -> dict:
    """Per-stock investor-type net buying (KRW) incl. '연기금' (pension funds & government) from KRX [12009]."""
    if not krx_ready():
        return {"skipped": "KRX_ID / KRX_PW 또는 KRX_JSESSIONID 없음 (data.krx.co.kr 무료 회원가입 후 .env에 입력)"}
    stock = _krx_stock()
    if not _krx_alive(stock):
        raise RuntimeError("KRX 로그인 세션이 동작하지 않음 — KRX_JSESSIONID(브라우저 쿠키) 또는 KRX_ID / KRX_PW 확인")
    d = kd.root(settings, "krx_flows")
    done = 0
    for code, (lo, hi) in sorted(spans.items()):
        p = d / f"{code}.parquet"
        if p.exists():
            continue
        for attempt in range(4):  # one request returns the whole span (checked: 2015-07..2026-10 in ~7 s)
            try:
                df = stock.get_market_trading_value_by_date(
                    lo.strftime("%Y%m%d"), hi.strftime("%Y%m%d"), code, detail=True
                )
                break
            except Exception as exc:  # noqa: BLE001 - KRX answers HTML / empty bodies on throttling
                if attempt == 3:
                    raise RuntimeError(f"KRX {code}: {exc}") from exc
                time.sleep(3.0 * (attempt + 1))
        time.sleep(KRX_PAUSE)
        if df is None or not len(df):
            if not _krx_alive(stock):  # an expired session answers empty: stop, do not store a gap
                raise RuntimeError(
                    f"KRX 세션 만료({done}개 종목 새로 저장) — 브라우저에서 다시 로그인해 KRX_JSESSIONID를 바꾼 뒤 "
                    "같은 명령을 다시 실행하면 이어서 받습니다"
                )
            continue
        pcol = next((c for c in df.columns if str(c).startswith("연기금")), None)
        out = pd.DataFrame({"date": pd.DatetimeIndex(df.index)})
        out["pension"] = df[pcol].to_numpy(float) if pcol else np.nan
        for c in df.columns:
            out[f"v_{c}"] = df[c].to_numpy(float)
        out.to_parquet(p, index=False)
        done += 1
        if done % 25 == 0:
            say(f"KRX flows {done} codes")
    prog = krx_program_fetch(settings, min(lo for lo, _ in spans.values()), max(hi for _, hi in spans.values()), say)
    return {"codes": done, "program": prog}


def _num_krx(x) -> float:
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return np.nan


def krx_program_fetch(settings, lo: pd.Timestamp, hi: pd.Timestamp, say: Say = say) -> dict:
    """KOSPI program trading [12012] (bld MDCSTAT02601): the screen returns period sums (차익 / 비차익 / 전체 rows), so
    it is queried one calendar week (Mon..Sun) at a time (~590 requests, resumable; one session per day would need
    ~2,800 and outlive a ~27-minute KRX login) - plus the KOSPI index's daily traded value as the denominator."""
    if not krx_ready():
        return {"skipped": "KRX 로그인 없음"}
    stock = _krx_stock()
    from pykrx.website.krx.krxio import KrxWebIo

    class _Program(KrxWebIo):
        @property
        def bld(self):
            return "dbms/MDC/STAT/standard/MDCSTAT02601"

    d = kd.root(settings, "krx_flows")
    ix = stock.get_index_ohlcv(lo.strftime("%Y%m%d"), hi.strftime("%Y%m%d"), "1001")  # 코스피
    if ix is None or not len(ix):
        return {"error": "코스피 지수 시세 빈 응답(세션 확인)"}
    pd.DataFrame({"date": pd.DatetimeIndex(ix.index), "mkt_val": ix["거래대금"].to_numpy(float)}).to_parquet(
        d / "_kospi_value.parquet", index=False
    )
    p = d / "_program_kospi_weekly.parquet"
    have = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["date"])
    done = set(pd.DatetimeIndex(have["date"]))
    rows = [have] if len(have) else []
    sessions = pd.DatetimeIndex(ix.index)
    week_end = sessions + pd.to_timedelta(6 - sessions.weekday, "D")  # the Sunday closing each session's week
    todo = [w for w in pd.DatetimeIndex(week_end.unique()).sort_values() if w not in done and w < pd.Timestamp.today()]
    for i, t in enumerate(todo, 1):
        a, b = (t - pd.Timedelta(6, "D")).strftime("%Y%m%d"), t.strftime("%Y%m%d")
        out = []
        for attempt in range(4):  # KRX sometimes answers an HTML page instead of JSON - retry, then check the session
            try:
                out = _Program().read(strtDd=a, endDd=b, mktId="STK").get("output") or []
                break
            except ValueError:
                time.sleep(3.0 * (attempt + 1))
        rec = {"date": t}
        for r in out:
            k = {"차익": "arb", "비차익": "nonarb", "전체": "tot"}.get(str(r.get("ITM_TP_NM")).strip())
            if k:
                rec[f"{k}_buy"] = _num_krx(r.get("BID_TRDVAL"))
                rec[f"{k}_sell"] = _num_krx(r.get("ASK_TRDVAL"))
                rec[f"{k}_net"] = _num_krx(r.get("NETBID_TRDVAL"))
        if "tot_buy" not in rec:
            if not _krx_alive(stock):
                if rows:
                    pd.concat(rows, ignore_index=True).to_parquet(p, index=False)
                raise RuntimeError(
                    "KRX 세션 만료(프로그램매매) — KRX_JSESSIONID를 바꾼 뒤 다시 실행하면 이어서 받습니다"
                )
            continue
        rows.append(pd.DataFrame([rec]))
        if i % 100 == 0:
            pd.concat(rows, ignore_index=True).to_parquet(p, index=False)
            say(f"KRX program trading {i}/{len(todo)} weeks")
        time.sleep(KRX_PAUSE)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if df.empty:
        return {"error": "빈 응답"}
    df.to_parquet(p, index=False)
    say(f"KRX program trading: {len(df)} weeks")
    return {"weeks": len(df)}


def krx_program_regime(settings) -> pd.DataFrame | None:
    """r_kr_prog_share = last-4-weeks (program buy + sell) / (2 x KOSPI traded value) and r_kr_prog_net = last-4-weeks
    non-arbitrage net buying / KOSPI traded value; a week counts from the day after its Sunday (daily series, as of
    t); None without the cached tables."""
    d = kd.root(settings, "krx_flows")
    p, q = d / "_program_kospi_weekly.parquet", d / "_kospi_value.parquet"
    if not (p.exists() and q.exists()):
        return None
    pr = pd.read_parquet(p).set_index("date").sort_index()
    mv = pd.read_parquet(q).set_index("date")["mkt_val"]
    if pr.index.max() < mv.index.max() - pd.Timedelta(21, "D"):
        return None  # an incomplete download (e.g. stopped by a KRX access restriction) is not used
    mv = mv.groupby(mv.index + pd.to_timedelta(6 - mv.index.weekday, "D")).sum().reindex(pr.index)
    gap_ok = pd.Series(pr.index, index=pr.index).diff(3).dt.days.eq(21)  # four consecutive weeks
    den = mv.rolling(4).sum()
    weekly = pd.DataFrame(
        {
            "r_kr_prog_share": ((pr["tot_buy"] + pr["tot_sell"]).rolling(4).sum() / (2 * den)).where(gap_ok),
            "r_kr_prog_net": (pr["nonarb_net"].rolling(4).sum() / den).where(gap_ok),
        }
    )
    weekly.index = weekly.index + pd.Timedelta(1, "D")  # known from the Monday after the week
    end = max(pd.Timestamp.today().normalize(), weekly.index.max())
    return weekly.reindex(pd.date_range(weekly.index.min(), end, freq="D")).ffill(limit=10)


# ================================================================================================ block F: US 13F

F13_PAGE = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
CIK_CUSIP = "https://raw.githubusercontent.com/leoliu0/cik-cusip-mapping/master/cik-cusip-maps.csv"
F13_DEADLINE = 45  # days after the quarter end
# VALUE is reported in dollars for filings after 2023-01-03 (periods from 2022-12-31 on, given the deadline) and in
# thousands before - but a minority of filers already used dollars earlier, which inflates a VALUE x 1000 sum by up to
# 10x. Holdings are therefore valued as shares x the MEDIAN implied price (VALUE / shares) of all filers of that CUSIP
# in that period, which those few rows cannot move.
F13_DOLLAR_PERIODS_FROM = pd.Timestamp("2022-12-31")
F13_REDUCE_VERSION = 2


def _sec_headers() -> dict:
    kd.load_env()
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError("SEC_USER_AGENT가 .env에 없습니다")
    return {"User-Agent": ua}


def _download(url: str, path: Path, headers: dict) -> Path:
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(5):
        try:
            with requests.get(url, headers=headers, timeout=900, stream=True) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for ch in r.iter_content(1 << 20):
                        fh.write(ch)
            tmp.rename(path)
            return path
        except requests.RequestException:
            time.sleep(10.0 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


def _f13_reduce(zpath: Path) -> pd.DataFrame:
    """One 13F data-set zip -> rows (period, cusip9, shares, n_rows, px_raw, n_filers) from original 13F-HR filings
    filed by the deadline (px_raw = median VALUE / shares, in the period's reporting unit); plus pseudo rows
    cusip9='__ALL__' with the number of filers per period."""
    z = zipfile.ZipFile(zpath)
    names = {Path(n).name.upper(): n for n in z.namelist()}
    sub = pd.read_csv(
        z.open(names["SUBMISSION.TSV"]),
        sep="\t",
        dtype=str,
        usecols=["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"],
    )
    sub = sub[sub["SUBMISSIONTYPE"].str.strip() == "13F-HR"].copy()
    sub["filed"] = pd.to_datetime(sub["FILING_DATE"], format="mixed", dayfirst=False, errors="coerce")
    sub["period"] = pd.to_datetime(sub["PERIODOFREPORT"], format="mixed", dayfirst=False, errors="coerce")
    sub = sub.dropna(subset=["filed", "period"])
    sub = sub[sub["filed"] <= sub["period"] + pd.Timedelta(F13_DEADLINE, "D")]
    cols = ["period", "cusip9", "shares", "n_rows", "px_raw", "n_filers"]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    info = pd.read_csv(
        z.open(names["INFOTABLE.TSV"]),
        sep="\t",
        dtype=str,
        usecols=["ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"],
    )
    info = info[(info["SSHPRNAMTTYPE"].str.strip() == "SH") & info["PUTCALL"].isna()]
    info = info.merge(sub[["ACCESSION_NUMBER", "CIK", "period"]], on="ACCESSION_NUMBER", how="inner")
    info["shares"] = pd.to_numeric(info["SSHPRNAMT"], errors="coerce")
    info["px"] = pd.to_numeric(info["VALUE"], errors="coerce") / info["shares"].where(info["shares"] > 0)
    info["cusip9"] = info["CUSIP"].str.strip().str.upper()
    agg = info.groupby(["period", "cusip9"]).agg(
        shares=("shares", "sum"), n_rows=("shares", "size"), px_raw=("px", "median"), n_filers=("CIK", "nunique")
    )
    agg = agg.reset_index()
    tot = sub.groupby("period")["CIK"].nunique().rename("n_filers").reset_index()
    tot["cusip9"], tot["shares"], tot["n_rows"], tot["px_raw"] = "__ALL__", np.nan, 0, np.nan
    return pd.concat([agg, tot[cols]], ignore_index=True)


def f13_fetch(settings, say: Say = say) -> dict:
    d = kd.root(settings, "sec13f")
    hdr = _sec_headers()
    page = requests.get(F13_PAGE, headers=hdr, timeout=120).text
    links = sorted(set(re.findall(r'href="([^"]+form13f[^"]*\.zip)"', page)))
    done = 0
    for link in links:
        name = link.split("/")[-1]
        out = d / name.replace(".zip", f"_v{F13_REDUCE_VERSION}.parquet")
        if out.exists():
            continue
        zp = d / name
        if not zp.exists():
            _download("https://www.sec.gov" + link if link.startswith("/") else link, zp, hdr)
            time.sleep(0.5)
        _f13_reduce(zp).to_parquet(out, index=False)
        zp.unlink(missing_ok=True)
        done += 1
        say(f"13F {name}: reduced ({done} new)")
    m = d / "cik_cusip.csv"
    if not m.exists():
        r = requests.get(CIK_CUSIP, timeout=120)
        r.raise_for_status()
        m.write_bytes(r.content)
    return {"files": len(links), "new": done}


def _f13_table(settings) -> pd.DataFrame:
    """(period, cusip6, value_usd, n_filers) + cusip6='__ALL__' filer counts, from the reduced data-set files."""
    d = kd.root(settings, "sec13f")
    parts = [pd.read_parquet(p) for p in sorted(d.glob(f"*_form13f_v{F13_REDUCE_VERSION}.parquet"))]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame()
    t = pd.concat(parts, ignore_index=True)
    allf = t[t["cusip9"] == "__ALL__"].groupby("period", as_index=False)["n_filers"].sum()
    h = t[t["cusip9"] != "__ALL__"].sort_values("n_rows")
    # a period's filings can span two data-set files: sum shares and filers, keep the price of the larger part
    h = h.groupby(["period", "cusip9"], as_index=False).agg(
        shares=("shares", "sum"), n_filers=("n_filers", "sum"), px_raw=("px_raw", "last")
    )
    unit = np.where(h["period"] >= F13_DOLLAR_PERIODS_FROM, 1.0, 1000.0)
    h["value_usd"] = h["shares"] * h["px_raw"] * unit
    h["cusip6"] = h["cusip9"].str[:6]
    out = h.groupby(["period", "cusip6"], as_index=False).agg(
        value_usd=("value_usd", "sum"), n_filers=("n_filers", "sum")
    )
    allf["cusip6"], allf["value_usd"] = "__ALL__", np.nan
    return pd.concat([out, allf[out.columns]], ignore_index=True)


US_F_DOCS = {
    "us_inst_own": ("13F 기관 보유 평가액 ÷ 분기말 시가총액", "13F institutional holdings / market cap"),
    "us_inst_own_chg": ("13F 기관 보유 비율의 분기 변화", "quarterly change of 13F ownership"),
    "us_inst_breadth_chg": ("보유 기관 수 비중의 분기 변화(Chen·Hong·Stein)", "change in breadth of 13F ownership"),
    "us_algo_c2t": ("log(주문 취소 ÷ 체결 건수) — 알고리즘 매매 강도", "log cancel-to-trade ratio (MIDAS)"),
    "us_algo_oddlot": ("단주 거래량 비중", "odd-lot volume share (MIDAS)"),
    "us_algo_hidden": ("숨은 주문 체결 비중", "hidden-order volume share (MIDAS)"),
    "us_algo_t2o": ("log(체결 거래량 ÷ 주문 거래량)", "log trade-to-order volume ratio (MIDAS)"),
    "us_algo_c2t_chg": ("주문 취소 비율의 분기 변화", "quarterly change of the cancel-to-trade ratio"),
}


def block_f_13f(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, dict]:
    from quantcio.data.pit.us_cik import entity_cik

    t = _f13_table(settings)
    out = pd.DataFrame(
        index=frame.index, columns=["us_inst_own", "us_inst_own_chg", "us_inst_breadth_chg"], dtype=float
    )
    if t.empty:
        return out, {"13f": "no data"}
    m = pd.read_csv(kd.root(settings, "sec13f") / "cik_cusip.csv", dtype=str)
    m["cik"] = pd.to_numeric(m["cik"], errors="coerce")
    m = m.dropna(subset=["cik", "cusip6"])
    by_cik = m.groupby(m["cik"].astype(int))["cusip6"].agg(lambda s: sorted({x.strip().upper() for x in s}))
    total = t[t["cusip6"] == "__ALL__"].set_index("period")["n_filers"]
    hold = t[t["cusip6"] != "__ALL__"].set_index(["cusip6", "period"])
    mcap = np.exp(frame["log_mcap"].astype(float))
    pos = {ix: i for i, ix in enumerate(frame.index)}
    vals = np.full((len(frame), 3), np.nan)
    matched = 0
    for ent, g in frame.groupby("entity", sort=False):
        try:
            cik = int(entity_cik(str(ent)))
        except (TypeError, ValueError):
            continue
        cus = by_cik.get(cik)
        if not cus:
            continue
        sub = hold[hold.index.get_level_values(0).isin(cus)]
        if sub.empty:
            continue
        matched += 1
        q = sub.groupby(level=1).agg(value_usd=("value_usd", "sum"), n=("n_filers", "sum")).sort_index()
        gm = pd.Series(mcap[g.index].to_numpy(), index=pd.DatetimeIndex(g["date"])).sort_index()
        gm = gm[~gm.index.duplicated()]
        mc = gm.reindex(q.index, method="ffill", tolerance=pd.Timedelta(14, "D"))
        own = (q["value_usd"] / mc).clip(0, 1.5)
        breadth = q["n"] / total.reindex(q.index)
        prev_ok = pd.Series(q.index, index=q.index).diff().dt.days.between(80, 100)
        feat_q = pd.DataFrame(
            {
                "own": own,
                "own_chg": (own - own.shift(1)).where(prev_ok),
                "breadth_chg": (breadth - breadth.shift(1)).where(prev_ok),
            }
        )
        feat_q.index = feat_q.index + pd.Timedelta(F13_DEADLINE + 1, "D")  # available from
        rows = np.array([pos[i] for i in g.index])
        vals[rows] = _asof_rows(feat_q, g["date"].to_numpy(dtype="datetime64[ns]"), 200)
    out[:] = vals
    return out, {"13f_entities_matched": matched, "13f_periods": int(total.size)}


# ================================================================================================ block F: US MIDAS

MIDAS_PAGE = "https://www.sec.gov/data-research/sec-markets-data/market-structure-data-security-exchange"
MIDAS_LAG = 100  # days after the quarter end
MIDAS_SUM = [
    "Cancels",
    "Trades",
    "LitTrades",
    "OddLots",
    "Hidden",
    "TradesForHidden",
    "OrderVol('000)",
    "TradeVol('000)",
    "LitVol('000)",
    "OddLotVol('000)",
    "HiddenVol('000)",
    "TradeVolForHidden('000)",
]


def _norm_ticker(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def _midas_reduce(zpath: Path) -> pd.DataFrame:
    z = zipfile.ZipFile(zpath)
    frames = []

    def csvs(zz: zipfile.ZipFile):
        for n in zz.namelist():
            if n.lower().endswith(".zip"):
                yield from csvs(zipfile.ZipFile(io.BytesIO(zz.read(n))))
            elif n.lower().endswith(".csv"):
                yield zz, n

    for zz, n in csvs(z):
        df = pd.read_csv(zz.open(n), usecols=lambda c: c in ("Security", "Ticker", *MIDAS_SUM), low_memory=False)
        df = df[df["Security"].astype(str).str.strip().str.lower() == "stock"]
        for c in MIDAS_SUM:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        frames.append(df.groupby("Ticker")[MIDAS_SUM].sum(min_count=1))
    if not frames:
        return pd.DataFrame()
    agg = pd.concat(frames).groupby(level=0).sum(min_count=1).reset_index()
    agg["ticker"] = _norm_ticker(agg["Ticker"])
    return agg.drop(columns=["Ticker"])


def midas_fetch(settings, say: Say = say) -> dict:
    d = kd.root(settings, "midas")
    hdr = _sec_headers()
    page = requests.get(MIDAS_PAGE, headers=hdr, timeout=120).text
    links = sorted(set(re.findall(r'href="([^"]+individual_security_exchange[^"]*\.zip)"', page)))
    done = 0
    for link in links:
        name = link.split("/")[-1]
        mt = re.search(r"(\d{4})_q(\d)", name)
        if not mt:
            continue
        out = d / f"midas_{mt.group(1)}_q{mt.group(2)}.parquet"
        if out.exists():
            continue
        zp = d / name
        if not zp.exists():
            _download("https://www.sec.gov" + link if link.startswith("/") else link, zp, hdr)
            time.sleep(0.5)
        red = _midas_reduce(zp)
        red["year"], red["q"] = int(mt.group(1)), int(mt.group(2))
        red.to_parquet(out, index=False)
        zp.unlink(missing_ok=True)
        done += 1
        say(f"MIDAS {mt.group(1)} Q{mt.group(2)}: {len(red)} tickers ({done} new)")
    return {"files": len(links), "new": done}


def block_f_midas(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, dict]:
    d = kd.root(settings, "midas")
    parts = [pd.read_parquet(p) for p in sorted(d.glob("midas_*.parquet"))]
    cols = ["us_algo_c2t", "us_algo_oddlot", "us_algo_hidden", "us_algo_t2o", "us_algo_c2t_chg"]
    out = pd.DataFrame(index=frame.index, columns=cols, dtype=float)
    if not parts:
        return out, {"midas": "no data"}
    m = pd.concat(parts, ignore_index=True)
    m["qend"] = (
        pd.PeriodIndex([f"{y}Q{q}" for y, q in zip(m["year"], m["q"], strict=True)], freq="Q")
        .to_timestamp(how="end")
        .normalize()
    )
    m = m.groupby(["ticker", "qend"], as_index=False)[MIDAS_SUM].sum(min_count=1)

    def ratio(a, b):
        return m[a] / m[b].where(m[b] > 0)

    m["c2t"] = np.log(ratio("Cancels", "LitTrades").where(lambda s: s > 0))
    m["oddlot"] = ratio("OddLotVol('000)", "TradeVol('000)")
    m["hidden"] = ratio("HiddenVol('000)", "TradeVolForHidden('000)")
    m["t2o"] = np.log(ratio("LitVol('000)", "OrderVol('000)").where(lambda s: s > 0))
    m = m.sort_values(["ticker", "qend"])
    gap = m.groupby("ticker")["qend"].diff().dt.days.between(80, 100)
    m["c2t_chg"] = (m["c2t"] - m.groupby("ticker")["c2t"].shift(1)).where(gap)
    m["avail"] = m["qend"] + pd.Timedelta(MIDAS_LAG, "D")
    by_t = {t: g.set_index("avail")[["c2t", "oddlot", "hidden", "t2o", "c2t_chg"]] for t, g in m.groupby("ticker")}
    tick = _norm_ticker(frame["ticker"])
    vals = np.full((len(frame), 5), np.nan)
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    for t, idx in tick.groupby(tick).groups.items():
        s = by_t.get(t)
        if s is None:
            continue
        ii = frame.index.get_indexer(idx)
        vals[ii] = _asof_rows(s.sort_index(), dates[ii], 200)
    out[:] = vals
    return out, {"midas_quarters": int(m["qend"].nunique()), "midas_tickers": int(m["ticker"].nunique())}


def fed_balance_sheet(settings) -> pd.Series:
    """r_fed_bs: 13-week log change of WALCL, available two days after its Wednesday date."""
    k = kd.key("FRED_API_KEY")
    p = kd.root(settings, "fred") / "walcl.json"
    if not p.exists() or time.time() - p.stat().st_mtime > 86400:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": "WALCL", "api_key": k, "file_type": "json"},
            timeout=120,
        )
        r.raise_for_status()
        p.write_text(json.dumps(r.json()["observations"]))
    obs = pd.DataFrame(json.loads(p.read_text()))
    s = pd.Series(pd.to_numeric(obs["value"], errors="coerce").to_numpy(), index=pd.to_datetime(obs["date"])).dropna()
    ch = np.log(s / s.shift(13))
    return kd._daily_available(ch, pd.Series(ch.index + pd.Timedelta(2, "D")), 14)


# ================================================================================================ block S


def _kosis(params: dict, objs: list[str]) -> list[dict]:
    """KOSIS rows; ``objs`` are the classification selections in some order - KOSIS wants the table's own order
    (error 21 otherwise), so every permutation is tried."""
    from itertools import permutations

    k = kd.key("KOSIS_API_KEY")
    base = {"method": "getList", "apiKey": k, "format": "json", "jsonVD": "Y", "orgId": "101"}
    last = None
    for perm in permutations(objs):
        prm = {**base, **params, **{f"objL{i + 1}": o for i, o in enumerate(perm)}}
        for attempt in range(4):
            try:
                r = requests.get(f"{kd.KOSIS}/Param/statisticsParameterData.do", params=prm, timeout=180)
                js = kd._kosis_json(r)
                break
            except requests.RequestException:
                time.sleep(5.0 * (attempt + 1))
        else:
            raise RuntimeError(f"KOSIS {params.get('tblId')}: repeated failures")
        if isinstance(js, list):
            return js
        last = js
        if str(js.get("err")) != "21":
            break
    raise RuntimeError(f"KOSIS {params.get('tblId')}: {last}")


def _kosis_code_col(df: pd.DataFrame, codes: set[str]) -> str:
    for c in ("C1", "C2", "C3", "C4"):
        if c in df and set(df[c].astype(str)) & codes:
            return c
    raise KeyError(f"KOSIS: no classification column holds {sorted(codes)}")


def social_fetch(settings, say: Say = say) -> dict:
    """KOSIS (KR) + World Bank + Census ACS (US) -> <cache>/alt/keyed/social/*.json."""
    d = kd.root(settings, "social")
    res = {}
    if kd.key("KOSIS_API_KEY"):
        this_year = pd.Timestamp.today().year
        annual = {"prdSe": "Y", "startPrdDe": "1995", "endPrdDe": str(this_year)}
        jobs = {
            "kr_monthly": (
                {
                    "tblId": "DT_1B8000G",
                    "itmId": "T1",
                    "prdSe": "M",
                    "startPrdDe": "199701",
                    "endPrdDe": pd.Timestamp.today().strftime("%Y%m"),
                },
                ["00", "10+15+20+30+"],
            ),
            "kr_tfr": ({"tblId": "DT_1B8000F", "itmId": "T1", **annual}, ["30"]),
            "kr_life": ({"tblId": "DT_1B41", "itmId": "T6", **annual}, ["01"]),
            "kr_suicide": ({"tblId": "DT_1B34E01", "itmId": "T5", **annual}, ["J06", "0", "00"]),
            "kr_marry_age": ({"tblId": "DT_1B83A09", "itmId": "T10+T20+", **annual}, ["00"]),
        }
        for name, (prm, objs) in jobs.items():
            rows = _kosis(prm, objs)
            (d / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False))
            res[name] = len(rows)
            say(f"KOSIS {name}: {len(rows)} rows")
            time.sleep(0.5)
    else:
        res["kosis"] = "KOSIS_API_KEY 없음"
    wb = {}
    for ind in ("SP.DYN.TFRT.IN", "SP.DYN.LE00.IN", "SH.STA.SUIC.P5"):
        r = requests.get(
            f"https://api.worldbank.org/v2/country/USA/indicator/{ind}",
            params={"format": "json", "per_page": 200, "date": "1990:2030"},
            timeout=120,
        )
        r.raise_for_status()
        wb[ind] = {x["date"]: x["value"] for x in r.json()[1] if x["value"] is not None}
    (d / "us_worldbank.json").write_text(json.dumps(wb))
    res["worldbank"] = {k: len(v) for k, v in wb.items()}
    acs = {}
    ck = kd.key("CENSUS_API_KEY")  # api.census.gov now redirects key-less requests to missing_key.html
    for y in range(2005, pd.Timestamp.today().year) if ck else ():
        try:
            r = requests.get(
                f"https://api.census.gov/data/{y}/acs/acs1",
                params={"get": "B12007_001E,B12007_002E", "for": "us:1", "key": ck},
                timeout=60,
                allow_redirects=False,
            )
            if r.status_code == 200:
                hdr, row = r.json()[0], r.json()[1]
                acs[y] = (float(row[hdr.index("B12007_001E")]) + float(row[hdr.index("B12007_002E")])) / 2
        except (requests.RequestException, ValueError, IndexError):
            continue
        time.sleep(0.3)
    if acs:
        (d / "us_acs_marry_age.json").write_text(json.dumps(acs))
    res["acs_years"] = sorted(acs) if ck else "CENSUS_API_KEY 없음 (api.census.gov/data/key_signup.html 무료)"
    return res


S_DOCS = {
    "r_kr_births": ("출생아 수 12개월 합 전년 대비 log 변화", "births, 12-month sum, YoY log change"),
    "r_kr_marriages": ("혼인 건수 12개월 합 전년 대비", "marriages, 12-month sum, YoY"),
    "r_kr_deaths": ("사망자 수 12개월 합 전년 대비", "deaths, 12-month sum, YoY"),
    "r_kr_divorces": ("이혼 건수 12개월 합 전년 대비", "divorces, 12-month sum, YoY"),
    "r_kr_tfr": ("합계출산율 전년 대비 변화", "total fertility rate, change"),
    "r_kr_life_exp": ("기대수명(0세) 전년 대비 변화", "life expectancy at birth, change"),
    "r_kr_suicide": ("자살 사망률(10만 명당) 전년 대비 변화", "suicide rate per 100k, change"),
    "r_kr_marry_age": ("평균 초혼 연령(남녀 평균) 전년 대비 변화", "mean age at first marriage, change"),
    "r_us_tfr": ("미국 합계출산율 전년 대비 변화", "US total fertility rate, change"),
    "r_us_life_exp": ("미국 기대수명 전년 대비 변화", "US life expectancy, change"),
    "r_us_suicide": ("미국 자살률 전년 대비 변화", "US suicide rate, change"),
    "r_us_marry_age": ("미국 초혼 연령 중앙값 전년 대비 변화", "US median age at first marriage, change"),
}


def _annual_change(values: dict, avail_month_day: tuple[int, int], years_after: int = 1) -> pd.Series:
    """{year: value} -> daily series of the year-over-year change, available from (year + years_after)-MM-DD."""
    s = pd.Series({int(k): float(v) for k, v in values.items() if v is not None}).sort_index()
    ch = s.diff().dropna()
    if ch.empty:
        return pd.Series(dtype=float)
    avail = pd.Series(
        [pd.Timestamp(y + years_after, avail_month_day[0], avail_month_day[1]) for y in ch.index], index=ch.index
    )
    return kd._daily_available(ch.reset_index(drop=True), avail.reset_index(drop=True), 800)


def social_regime(settings, market: str) -> dict[str, pd.Series]:
    d = kd.root(settings, "social")
    out: dict[str, pd.Series] = {}

    def load(name):
        p = d / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    if market == "KR":
        rows = load("kr_monthly")
        if rows:
            df = pd.DataFrame(rows)
            df["v"] = pd.to_numeric(df["DT"], errors="coerce")
            df["m"] = pd.to_datetime(df["PRD_DE"], format="%Y%m")
            cc = _kosis_code_col(df, {"10", "15", "20", "30"})
            for code, name in (
                ("10", "r_kr_births"),
                ("20", "r_kr_marriages"),
                ("15", "r_kr_deaths"),
                ("30", "r_kr_divorces"),
            ):
                s = df[df[cc].astype(str) == code].set_index("m")["v"].sort_index()
                s12 = s.rolling(12, min_periods=12).sum()
                ch = np.log(s12 / s12.shift(12)).dropna()
                avail = pd.Series(ch.index + pd.offsets.MonthEnd(2) + pd.Timedelta(1, "D"))
                out[name] = kd._daily_available(ch.reset_index(drop=True), avail, 120)
        for name, file, md, ya in (
            ("r_kr_tfr", "kr_tfr", (3, 1), 1),
            ("r_kr_life_exp", "kr_life", (1, 1), 2),
            ("r_kr_suicide", "kr_suicide", (10, 1), 1),
        ):
            rows = load(file)
            if rows:
                vals = {r["PRD_DE"]: pd.to_numeric(r["DT"], errors="coerce") for r in rows}
                out[name] = _annual_change(vals, md, ya)
        rows = load("kr_marry_age")
        if rows:
            df = pd.DataFrame(rows)
            df["v"] = pd.to_numeric(df["DT"], errors="coerce")
            vals = df.groupby("PRD_DE")["v"].mean().to_dict()
            out["r_kr_marry_age"] = _annual_change(vals, (4, 1), 1)
    else:
        wb = load("us_worldbank") or {}
        for ind, name in (
            ("SP.DYN.TFRT.IN", "r_us_tfr"),
            ("SP.DYN.LE00.IN", "r_us_life_exp"),
            ("SH.STA.SUIC.P5", "r_us_suicide"),
        ):
            if wb.get(ind):
                out[name] = _annual_change(wb[ind], (12, 31), 1)
        acs = load("us_acs_marry_age")
        if acs:
            out["r_us_marry_age"] = _annual_change(acs, (10, 1), 1)
    return {k: v for k, v in out.items() if v is not None and len(v)}


# ================================================================================================ CLI


def _spans(market: str, settings, cache) -> tuple[dict, object]:
    from quantcio.data.pit import load_pit_universe
    from quantcio.modeling.dataset import universe_params
    from quantcio.models import Market

    n, start = universe_params(Market(market), settings)
    uni = load_pit_universe(Market(market), settings, cache, n, start)
    m = uni.membership
    spans = {}
    for e, g in m.groupby("entity"):
        lo = pd.Timestamp(g["date"].min()) - pd.Timedelta(150, "D")
        px = uni.prices.get(e)
        hi = min(pd.Timestamp(g["date"].max()) + pd.Timedelta(40, "D"), pd.Timestamp.today().normalize())
        if px is not None and len(px):
            hi = min(hi, pd.Timestamp(px.index.max()))
        spans[str(e)] = (lo, hi)
    return spans, uni


def main() -> None:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--what", choices=["naver", "krx", "f13", "midas", "social", "all"], default="all")
    a = ap.parse_args()
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    w = a.what
    if w in ("naver", "krx", "all"):
        spans, _ = _spans("KR", settings, cache)
        if w in ("naver", "all"):
            naver_fetch(settings, spans)
        if w in ("krx", "all"):
            say(f"KRX: {krx_fetch(settings, spans)}")
    if w in ("f13", "all"):
        say(f"13F: {f13_fetch(settings)}")
    if w in ("midas", "all"):
        say(f"MIDAS: {midas_fetch(settings)}")
    if w in ("social", "all"):
        say(f"social: {social_fetch(settings)}")


if __name__ == "__main__":
    main()
