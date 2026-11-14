"""Keyless daily price probes for (possibly delisted) US tickers. Verified 2026-09-25.

yfinance / raw Yahoo v8 chart / Nasdaq API. All results cached to disk (pickle) so reruns are free.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

yf.config.debug.hide_exceptions = False   # else errors come back as empty frames
yf.config.network.retries = 1

CACHE = Path(__file__).resolve().parent / "cache" / "px"
CACHE.mkdir(parents=True, exist_ok=True)
UA_YAHOO_SHORT = "Mozilla/5.0 AppleWebKit/537.36"   # full Chrome UA gets 429 from query2 (fingerprint)
UA_BROWSER = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/140.0 Safari/537.36")
_last: dict[str, float] = {}


def _polite(host: str, gap: float) -> None:
    w = _last.get(host, 0) + gap - time.monotonic()
    if w > 0:
        time.sleep(w)
    _last[host] = time.monotonic()


def _cached(kind: str, key: str, fn):
    p = CACHE / f"{kind}__{key.replace('/', '_')}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    out = fn()
    p.write_bytes(pickle.dumps(out))
    return out


def yahoo_chart(symbol: str, start: str = "2013-01-01") -> dict:
    """Raw v8 chart. Returns {'ok', 'status', 'err', 'meta', 'df'} with df columns
    open high low close adjclose volume (close = split-adjusted, adjclose = split+div)."""
    def fetch():
        _polite("yahoo", 0.25)
        p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
        r = requests.get(f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
                         params=dict(period1=p1, period2=int(time.time()), interval="1d",
                                     includeAdjustedClose="true", events="div,splits"),
                         headers={"User-Agent": UA_YAHOO_SHORT}, timeout=30)
        if r.status_code != 200:
            try:
                err = r.json()["chart"]["error"]["description"]
            except Exception:  # noqa: BLE001
                err = r.text[:120]
            return dict(ok=False, status=r.status_code, err=err, meta=None, df=None)
        res = r.json()["chart"]["result"][0]
        meta = res["meta"]
        ts = res.get("timestamp") or []
        if not ts:
            return dict(ok=False, status=200, err="no timestamps", meta=meta, df=None)
        tz = meta.get("exchangeTimezoneName", "America/New_York")
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame({k: q[k] for k in ["open", "high", "low", "close", "volume"]}, index=idx, dtype="float64")
        df["adjclose"] = pd.Series(res["indicators"]["adjclose"][0]["adjclose"], index=idx, dtype="float64")
        ev = res.get("events", {})
        df.attrs["splits"] = {pd.Timestamp(int(k), unit="s").date(): v["numerator"] / v["denominator"]
                              for k, v in ev.get("splits", {}).items()}
        df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
        return dict(ok=True, status=200, err=None, meta=meta, df=df)
    return _cached("chart", f"{symbol}_{start}", fetch)


def yf_history(symbol: str, start: str = "2013-01-01") -> dict:
    def fetch():
        _polite("yahoo", 0.25)
        tk = yf.Ticker(symbol)
        try:
            h = tk.history(start=start, auto_adjust=False, actions=True)
        except Exception as e:  # noqa: BLE001
            return dict(ok=False, err=f"{type(e).__name__}: {str(e)[:160]}", df=None, meta=None)
        if h is None or h.empty:
            return dict(ok=False, err="empty", df=None, meta=None)
        h.index = h.index.tz_localize(None).normalize()
        try:  # keep only plain scalars (history_metadata holds unpicklable objects)
            meta = {k: v for k, v in (tk.history_metadata or {}).items() if isinstance(v, (str, int, float, bool))}
        except Exception:  # noqa: BLE001
            meta = None
        return dict(ok=True, err=None, df=h, meta=meta)
    return _cached("yf", f"{symbol}_{start}", fetch)


def nasdaq_history(symbol: str, start: str = "2014-01-01", assetclass: str = "stocks") -> dict:
    """api.nasdaq.com: ~10y max, split-adjusted but NOT dividend-adjusted; active symbols only."""
    def fetch():
        _polite("nasdaq", 0.4)
        try:
            r = requests.get(f"https://api.nasdaq.com/api/quote/{symbol}/historical",
                             params=dict(assetclass=assetclass, fromdate=start,
                                         todate=pd.Timestamp.today().strftime("%Y-%m-%d"), limit=9999),
                             headers={"User-Agent": UA_BROWSER, "Accept": "application/json"}, timeout=40)
            j = r.json()
        except Exception as e:  # noqa: BLE001
            return dict(ok=False, err=f"{type(e).__name__}: {str(e)[:120]}", df=None)
        if not j.get("data") or not j["data"].get("tradesTable") or not j["data"]["tradesTable"].get("rows"):
            return dict(ok=False, err=str(j.get("status", {}).get("bCodeMessage"))[:160], df=None)
        df = pd.DataFrame(j["data"]["tradesTable"]["rows"])
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
        return dict(ok=True, err=None, df=df.set_index("date").sort_index())
    return _cached("nasdaq", f"{symbol}_{start}_{assetclass}", fetch)
