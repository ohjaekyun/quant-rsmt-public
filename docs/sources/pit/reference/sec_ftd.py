"""SEC fails-to-deliver files as a sparse, keyless, UNADJUSTED price source for delisted US stocks.

https://www.sec.gov/files/data/fails-deliver-data/cnsfailsYYYYMM{a|b}.zip  (a = 1st half, b = 2nd half)
Pipe-delimited: SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE
PRICE = closing price of the security on the business day BEFORE the settlement date.
A row exists only on days the security had fails outstanding -> sparse, not a daily series.
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parent / "cache" / "ftd"
CACHE.mkdir(parents=True, exist_ok=True)
_S = requests.Session()
_S.headers["User-Agent"] = "QuantCIO Research research@example.com"
_last = [0.0]
_INDEX = "https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data"
_URLS: dict[str, str] = {}


def ftd_urls() -> dict[str, str]:
    """'YYYYMMa'/'YYYYMMb' -> absolute URL. Files before 2017-07 live under
    /files/data/frequently-requested-foia-document-fails-deliver-data/, later ones under
    /files/data/fails-deliver-data/ (guessing the path 404s)."""
    if not _URLS:
        import re
        p = CACHE.parent / "ftd_index.html"
        html = p.read_text() if p.exists() else _S.get(_INDEX, timeout=30).text
        p.write_text(html)
        for href, key in re.findall(r'href="([^"]*cnsfails(\d{6}[ab])\.zip)"', html):
            _URLS[key] = href if href.startswith("http") else "https://www.sec.gov" + href
    return _URLS


def ftd_file(yyyymm: str, half: str) -> pd.DataFrame:
    p = CACHE / f"cnsfails{yyyymm}{half}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    w = _last[0] + 0.25 - time.monotonic()
    if w > 0:
        time.sleep(w)
    _last[0] = time.monotonic()
    url = ftd_urls().get(f"{yyyymm}{half}")
    if url is None:
        raise requests.HTTPError(f"no FTD file listed for {yyyymm}{half}")
    r = _S.get(url, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    for ln in raw.splitlines()[1:]:
        parts = ln.split("|")
        if len(parts) >= 6:  # DESCRIPTION may itself contain '|': take 4 fields left, price right
            rows.append(parts[:4] + ["|".join(parts[4:-1]), parts[-1]])
    df = pd.DataFrame(rows, columns=["settle", "cusip", "symbol", "qty", "desc", "price"])
    df = df[df.settle.str.fullmatch(r"\d{8}")]
    df["settle"] = pd.to_datetime(df.settle, format="%Y%m%d")
    df["price"] = pd.to_numeric(df.price, errors="coerce")
    df["qty"] = pd.to_numeric(df.qty, errors="coerce")
    df = df[["settle", "cusip", "symbol", "qty", "desc", "price"]]
    df.to_pickle(p)
    return df


def ftd_prices(symbol: str, start, end) -> pd.DataFrame:
    """Rows for `symbol` with settle in [start, end]. Column px_date = settle - 1 business day (price date)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    for m in pd.period_range(start, end, freq="M"):
        for half in ("a", "b"):
            if (half == "a" and m.start_time.replace(day=15) < start) or (half == "b" and m.start_time.replace(day=16) > end):
                continue
            try:
                f = ftd_file(m.strftime("%Y%m"), half)
            except requests.HTTPError:
                continue
            out.append(f[f.symbol == symbol])
    if not out:
        return pd.DataFrame(columns=["settle", "cusip", "symbol", "qty", "desc", "price", "px_date"])
    d = pd.concat(out)
    d = d[(d.settle >= start) & (d.settle <= end)].sort_values("settle")
    d["px_date"] = d.settle - pd.offsets.BDay(1)
    return d.reset_index(drop=True)
