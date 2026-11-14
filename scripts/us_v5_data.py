"""US v5 short-selling features (docs/MODELING.md section 19.1). Research code outside ``src/quantcio``.

- FINRA Reg SHO daily short-sale volume (off-exchange trades reported to FINRA TRFs, consolidated "CNMS" files,
  https://cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt; the CDN serves 2018-10 onwards). A trade date's
  file is used from the NEXT session (date < t).
- SEC fails-to-deliver (half-month zips, 2009 onwards): the fail QUANTITY per settlement date (the PIT price cache
  keeps only prices). A settlement date is used only 35+ days later (conservative publication lag).
Symbols are matched on a punctuation-free upper-case ticker (BRK.B / BRK-B -> BRKB).
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from quantcio.data.pit.us_net import PitNet
from quantcio.data.pit.us_prices import ftd_urls

FINRA = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"
FINRA_START = pd.Timestamp("2018-10-01")
FTD_LAG_DAYS = 35
FTD_WINDOW_DAYS = 30
FEATURES = ["us_short_ratio_20", "us_short_ratio_chg", "us_ftd"]
DOCS = {
    "us_short_ratio_20": ("공매도 거래 비중(20거래일, FINRA 장외분)", "short-sale volume share, 20 sessions (FINRA)"),
    "us_short_ratio_chg": ("공매도 거래 비중 변화(20일 − 120일)", "short-sale volume share change (20 - 120 sessions)"),
    "us_ftd": ("결제 불이행 주식 수 / 발행주식수(35일 지연)", "fails-to-deliver / shares outstanding (35-day lag)"),
}


def norm_symbol(s: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


# ------------------------------------------------------------------------------------------------ FINRA


def finra_dir(root: Path) -> Path:
    return root / "alt" / "finra_shvol"


def fetch_finra(root: Path, sessions: Iterable[pd.Timestamp], progress: Callable[[str], None] | None = None) -> dict:
    """Download + compact (symbol, short, total) every session's file since FINRA_START (cached; a missing file is
    remembered so it is not retried)."""
    d = finra_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    miss_path = d / "_missing.txt"
    missing = set(miss_path.read_text().split()) if miss_path.exists() else set()
    sess = [pd.Timestamp(s) for s in sessions if pd.Timestamp(s) >= FINRA_START]
    stats = {"sessions": len(sess), "downloaded": 0, "cached": 0, "missing": 0}
    http = requests.Session()
    http.headers["User-Agent"] = "Mozilla/5.0 (research; quantcio)"
    for i, s in enumerate(sess, 1):
        key = s.strftime("%Y%m%d")
        p = d / f"{key}.parquet"
        if p.exists():
            stats["cached"] += 1
            continue
        if key in missing and s < pd.Timestamp.today().normalize() - pd.Timedelta(7, "D"):
            stats["missing"] += 1
            continue
        r = http.get(FINRA.format(d=key), timeout=30)
        if r.status_code != 200 or not r.text.startswith("Date|"):
            missing.add(key)
            stats["missing"] += 1
            continue
        df = pd.read_csv(io.StringIO(r.text), sep="|", usecols=["Symbol", "ShortVolume", "TotalVolume"], dtype=str)
        df = df[df["Symbol"].notna()]
        out = (
            pd.DataFrame(
                {
                    "symbol": df["Symbol"].map(norm_symbol),
                    "short": pd.to_numeric(df["ShortVolume"], errors="coerce"),
                    "total": pd.to_numeric(df["TotalVolume"], errors="coerce"),
                }
            )
            .groupby("symbol", as_index=False)
            .sum()
        )
        out.to_parquet(p, index=False)
        stats["downloaded"] += 1
        if progress and i % 250 == 0:
            progress(f"FINRA {i}/{len(sess)}")
    miss_path.write_text("\n".join(sorted(missing)))
    return stats


def finra_panel(root: Path, symbols: set[str]) -> pd.DataFrame:
    """Long (date, symbol, short, total) for ``symbols`` from the cache."""
    parts = []
    for p in sorted(finra_dir(root).glob("*.parquet")):
        df = pd.read_parquet(p)
        df = df[df["symbol"].isin(symbols)]
        if len(df):
            df.insert(0, "date", pd.Timestamp(p.stem))
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "symbol", "short", "total"])


# ------------------------------------------------------------------------------------------------ SEC FTD


def ftd_dir(root: Path) -> Path:
    return root / "alt" / "ftd_qty"


def parse_ftd_qty(content: bytes) -> pd.DataFrame:
    """SEC cnsfails zip -> (settle, symbol, qty). DESCRIPTION may contain '|' (fields from the left)."""
    z = zipfile.ZipFile(io.BytesIO(content))
    txt = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    for ln in txt.splitlines()[1:]:
        f = ln.split("|")
        if len(f) < 6 or not re.fullmatch(r"\d{8}", f[0].strip()):
            continue
        rows.append((f[0].strip(), norm_symbol(f[2]), f[3].strip()))
    df = pd.DataFrame(rows, columns=["settle", "symbol", "qty"])
    df["settle"] = pd.to_datetime(df["settle"], format="%Y%m%d")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    return df.dropna(subset=["qty"]).groupby(["settle", "symbol"], as_index=False)["qty"].sum()


def fetch_ftd(net: PitNet, root: Path, progress: Callable[[str], None] | None = None) -> dict:
    d = ftd_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    urls = ftd_urls(net)
    stats = {"files": len(urls), "downloaded": 0, "cached": 0, "failed": 0}
    for i, (key, url) in enumerate(sorted(urls.items()), 1):
        p = d / f"{key}.parquet"
        if p.exists():
            stats["cached"] += 1
            continue
        try:
            r = net.http_get(url, "sec", ok=(200,))
            parse_ftd_qty(r.content).to_parquet(p, index=False)
            stats["downloaded"] += 1
        except Exception as exc:  # noqa: BLE001 - counted; the feature is NaN where no file covers the window
            stats["failed"] += 1
            if progress:
                progress(f"FTD {key} failed: {type(exc).__name__}")
        if progress and i % 50 == 0:
            progress(f"FTD {i}/{len(urls)}")
    return stats


def ftd_panel(root: Path, symbols: set[str]) -> tuple[pd.DataFrame, tuple[pd.Timestamp, pd.Timestamp] | None]:
    """(settle, symbol, qty) rows of ``symbols`` + the settlement-date range the files cover (a covered symbol
    without rows had no fails: 0, not missing)."""
    parts = [pd.read_parquet(p) for p in sorted(ftd_dir(root).glob("*.parquet"))]
    if not parts:
        return pd.DataFrame(columns=["settle", "symbol", "qty"]), None
    df = pd.concat(parts, ignore_index=True)
    cover = (pd.Timestamp(df["settle"].min()), pd.Timestamp(df["settle"].max()))
    return df[df["symbol"].isin(symbols)], cover


# ------------------------------------------------------------------------------------------------ features


def short_features(
    rows: pd.DataFrame,
    finra: pd.DataFrame,
    ftd: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    ftd_cover: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """us_short_ratio_20, us_short_ratio_chg, us_ftd for (date, ticker, mcap, close) rows."""
    out = np.full((len(rows), 3), np.nan)
    sym = rows["ticker"].map(norm_symbol).to_numpy()
    dates = rows["date"].to_numpy(dtype="datetime64[ns]")
    mcap = rows["mcap"].to_numpy(dtype=float)
    close = rows["close"].to_numpy(dtype=float)
    sess = sessions[sessions >= FINRA_START].to_numpy(dtype="datetime64[ns]")
    pos = {d: i for i, d in enumerate(sess)}
    fin = {s: g for s, g in finra.groupby("symbol")} if len(finra) else {}
    ftdg = {s: g.sort_values("settle") for s, g in ftd.groupby("symbol")} if len(ftd) else {}
    for s in pd.unique(sym):
        ix = np.flatnonzero(sym == s)
        g = fin.get(s)
        if g is not None and len(sess):
            sh = np.zeros(len(sess))
            tot = np.zeros(len(sess))
            has = np.zeros(len(sess))
            for d, a, b in g[["date", "short", "total"]].itertuples(index=False):
                i = pos.get(np.datetime64(d, "ns"))
                if i is not None:
                    sh[i], tot[i], has[i] = a, b, 1.0
            cs, ct, ch = (np.concatenate([[0.0], np.cumsum(x)]) for x in (sh, tot, has))
            for k in ix:
                hi = int(np.searchsorted(sess, dates[k], "left"))  # sessions strictly before t

                def ratio(n: int, need: int, _hi=hi) -> float:
                    lo = max(_hi - n, 0)
                    if ch[_hi] - ch[lo] < need:
                        return np.nan
                    t_ = ct[_hi] - ct[lo]
                    return (cs[_hi] - cs[lo]) / t_ if t_ > 0 else np.nan

                r20 = ratio(20, 10)
                out[k, 0] = r20
                r120 = ratio(120, 60)
                out[k, 1] = r20 - r120 if np.isfinite(r20) and np.isfinite(r120) else np.nan
        f = ftdg.get(s)
        if f is None:
            f = pd.DataFrame({"settle": pd.Series(dtype="datetime64[ns]"), "qty": pd.Series(dtype=float)})
        if ftd_cover is not None:
            st = f["settle"].to_numpy(dtype="datetime64[ns]")
            q = np.concatenate([[0.0], np.cumsum(f["qty"].to_numpy(dtype=float))])
            c0, c1 = (np.datetime64(x, "ns") for x in ftd_cover)
            for k in ix:
                hi_d = dates[k] - np.timedelta64(FTD_LAG_DAYS, "D")
                lo_d = hi_d - np.timedelta64(FTD_WINDOW_DAYS, "D")
                if lo_d < c0 or hi_d > c1:  # window not fully covered by the downloaded files
                    continue
                hi = int(np.searchsorted(st, hi_d, "right"))
                lo = int(np.searchsorted(st, lo_d, "left"))
                shares = mcap[k] / close[k] if close[k] > 0 and mcap[k] > 0 else np.nan
                n_sess = max(int(np.busday_count(lo_d.astype("datetime64[D]"), hi_d.astype("datetime64[D]"))), 1)
                if np.isfinite(shares) and shares > 0:
                    out[k, 2] = (q[hi] - q[lo]) / n_sess / shares
    return pd.DataFrame(out, index=rows.index, columns=FEATURES)
