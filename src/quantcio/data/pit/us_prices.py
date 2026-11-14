"""Prices for PIT S&P 500 spells: validated Yahoo daily series + SEC fails-to-deliver (FTD) marks.

- Yahoo v8 chart per candidate symbol (spell ticker, ticker traded at removal, rename chain, CIK's current SEC
  tickers with a name check). ``close`` is split-adjusted, ``adjclose`` also dividend/merger-cash adjusted.
- Every candidate is cross-checked against the SEC FTD PRICE of the ticker traded at the time (the unadjusted close of
  the session before settlement; equals Yahoo's raw close to ~5e-8). For the same issuer the FTD/close ratio is
  piecewise constant (breaks only at splits); another issuer's series drifts. Candidates with a plateau share < 0.5
  are rejected (BBT->Beacon Financial, PARA/VIAC->Banzai, WRK/MWV->Smurfit Kappa, DD 2017->Dow lineage); partly
  wrong series (IR 2017-20, JCI/CB pre-2016, UA 2016) are masked month by month.
- Member-months without a Yahoo return are filled from FTD prices (price-only, sparse, ~2-week publication lag,
  rows persist after delisting -> always capped strictly before the removal date, the first day out; a symbol reused
  from that day on, AA -> Alcoa Corp 2016-11-01, never leaks in). Terminal (delisting) marks come from FTD, tracked by
  CUSIP (FRC -> FRCB at $0.33 OTC).
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from collections.abc import Callable
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from quantcio.data.pit.us_calendar import prev_sessions
from quantcio.data.pit.us_cik import names_match
from quantcio.data.pit.us_net import PitNet
from quantcio.models import PRICE_COLUMNS

CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
FTD_INDEX = "https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data"
NEG_CHART_TTL_H = 24 * 7  # failed chart lookups are retried after a week; successes are kept
REFRESH_MIN_AGE_H = 6  # a cached series behind ``refresh_before`` is refetched only when at least this old
# a bar fetched before its session closed is a partial (intraday) bar: refetched once the close has passed.
# 16:00 New York is 20:00 UTC (EDT) or 21:00 UTC (EST); 21:30 UTC covers both plus Yahoo's settle delay
SESSION_FINAL_UTC = pd.Timedelta(21 * 60 + 30, unit="min")  # value + unit: no NumPy 2.5 "generic unit" warning
FTD_COLUMNS = ["settle", "cusip", "symbol", "price"]
FTD_MOVE_RANGE = (-0.6, 1.0)  # monthly FTD moves outside this are unusable (unadjusted splits, bad prints)
MIN_PARTIAL_BARS = 21  # a series starting inside the spell needs >= 1 month of bars in it (AVB 2026: 1-bar stub)
_EXCHANGES = {
    "NYQ": "NYSE",
    "NYSE": "NYSE",
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NASDAQ": "NASDAQ",
    "NasdaqGS": "NASDAQ",
    "NasdaqGM": "NASDAQ",
    "NasdaqCM": "NASDAQ",
    "ASE": "NYSE American",
    "PCX": "NYSE Arca",
    "BTS": "Cboe BZX",
    "PNK": "OTC",
    "OQB": "OTC",
    "OQX": "OTC",
}


# ============================================================================ Yahoo chart


def parse_chart_result(res: dict) -> dict:
    """Yahoo v8 ``result[0]`` -> {meta, df[open high low close adjclose volume], splits, dividends}."""
    meta = res.get("meta") or {}
    keep = (
        "symbol",
        "currency",
        "exchangeName",
        "fullExchangeName",
        "instrumentType",
        "firstTradeDate",
        "longName",
        "shortName",
        "exchangeTimezoneName",
    )
    out: dict[str, Any] = {"meta": {k: meta.get(k) for k in keep if k in meta}}
    tz = meta.get("exchangeTimezoneName") or "America/New_York"
    events = res.get("events") or {}

    def ev_series(kind: str, fn: Callable[[dict], float | None]) -> pd.Series:
        rows = {}
        for v in (events.get(kind) or {}).values():
            try:
                d = pd.Timestamp(int(v["date"]), unit="s", tz="UTC").tz_convert(tz).tz_localize(None).normalize()
                x = fn(v)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            if x is not None and np.isfinite(x) and x > 0:
                rows[d] = float(x)
        return pd.Series(rows, dtype="float64").sort_index()

    out["splits"] = ev_series("splits", lambda v: float(v["numerator"]) / float(v["denominator"]))
    out["dividends"] = ev_series("dividends", lambda v: float(v["amount"]))
    ts = res.get("timestamp") or []
    if not ts:
        out.update(ok=False, err="no timestamps", df=None)
        return out
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    q = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    df = pd.DataFrame({k: q.get(k) for k in ["open", "high", "low", "close", "volume"]}, index=idx, dtype="float64")
    adj = ((res.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
    df["adjclose"] = pd.Series(adj, index=idx, dtype="float64") if adj is not None and len(adj) == len(df) else np.nan
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
    df = df[df["close"] > 0]
    out.update(ok=bool(len(df)), err=None if len(df) else "no bars", df=df)
    return out


def load_chart(
    net: PitNet,
    symbol: str,
    start: date,
    *,
    refresh_before: pd.Timestamp | None = None,
    need_events: bool = False,
) -> dict:
    """Cached Yahoo chart for ``symbol`` from ``start``: {ok, status, err, meta, df, splits, dividends, fetched}.

    ``refresh_before``: refetch a cached success whose last bar is older (open spells need current bars; at least
    ``REFRESH_MIN_AGE_H`` old), or whose last bar was fetched before that session closed (a partial intraday bar
    that is final now).
    ``need_events``: refetch a cached entry stored without split/dividend events (seeded caches).
    """
    rel = f"{symbol}__{pd.Timestamp(start).date().isoformat()}.pkl"
    hit = net.read_pickle("yahoo", rel)
    if hit is not None:
        age_h = (time.time() - float(hit.get("fetched") or 0)) / 3600
        if hit.get("ok"):
            stale = refresh_before is not None and (
                (hit["df"].index[-1] < refresh_before and age_h > REFRESH_MIN_AGE_H)
                or partial_last_bar(hit["df"].index[-1], float(hit.get("fetched") or 0))
            )
            missing_ev = need_events and hit.get("splits") is None
            if not stale and not missing_ev:
                return hit
        elif age_h < NEG_CHART_TTL_H:
            return hit
    params = {
        "period1": int(pd.Timestamp(start).tz_localize("UTC").timestamp()),
        "period2": int(time.time()),
        "interval": "1d",
        "includeAdjustedClose": "true",
        "events": "div,splits",
    }
    try:
        r = net.http_get(CHART_URL.format(symbol=symbol), "yahoo", params=params, ok=(200, 404))
    except Exception as exc:  # noqa: BLE001 - transient failure: keep a stale success, else report 'error'
        if hit is not None and hit.get("ok"):
            net.note(f"Yahoo {symbol} 갱신 실패({type(exc).__name__}) — 캐시 사용")
            return hit
        return dict(ok=False, status=None, err=f"{type(exc).__name__}: {str(exc)[:120]}", meta=None, df=None)
    out: dict[str, Any] = dict(ok=False, status=r.status_code, err=None, meta=None, df=None, splits=None)
    if r.status_code != 200:
        try:
            out["err"] = ((r.json().get("chart") or {}).get("error") or {}).get("description")
        except ValueError:
            out["err"] = r.text[:100]
    else:
        try:
            result = ((r.json().get("chart") or {}).get("result") or [None])[0]
        except ValueError:
            result = None
        if not result:
            out["err"] = "empty result"
        else:
            out.update(parse_chart_result(result))
            out["status"] = 200
    out["fetched"] = time.time()
    net.write_pickle("yahoo", rel, out)
    return out


def partial_last_bar(last: pd.Timestamp, fetched: float, now: float | None = None) -> bool:
    """True when the last bar was fetched before its session's close (``SESSION_FINAL_UTC``) and that close has
    passed by ``now``: the cached bar is a partial intraday bar that can be replaced by the final one."""
    final = (pd.Timestamp(last).normalize() + SESSION_FINAL_UTC).timestamp()
    return fetched < final <= (time.time() if now is None else now)


def chart_first_trade(chart: dict) -> pd.Timestamp:
    ft = (chart.get("meta") or {}).get("firstTradeDate")
    if isinstance(ft, int | float):
        return pd.Timestamp(int(ft), unit="s").normalize()
    return chart["df"].index[0]


def chart_exchange(chart: dict | None) -> str:
    meta = (chart or {}).get("meta") or {}
    for k in ("exchangeName", "fullExchangeName"):
        v = meta.get(k)
        if v in _EXCHANGES:
            return _EXCHANGES[v]
    return str(meta.get("fullExchangeName") or "US")


def price_frame(df: pd.DataFrame, adjusted: bool = True) -> pd.DataFrame:
    """Chart df -> PRICE_COLUMNS frame + ``value``; OHLC scaled by adjclose/close (total return) when ``adjusted``.
    ``value`` = split-adjusted close x split-adjusted volume, the day's own traded value - the dividend-adjusted
    close x volume would shrink it for every LATER dividend (docs/MODELING.md 84.3)."""
    out = df[["open", "high", "low", "close", "volume"]].astype("float64").copy()
    value = (out["close"] * out["volume"].fillna(0.0)).to_numpy()
    ok_adj = adjusted and "adjclose" in df and df["adjclose"].notna().all() and len(df)
    if ok_adj:
        ratio = df["adjclose"] / df["close"]
        for c in ("open", "high", "low", "close"):
            out[c] = out[c] * ratio
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0.0)
    out = out[PRICE_COLUMNS].copy()
    out["value"] = value
    out.attrs["adjusted"] = bool(ok_adj)
    return out


# ============================================================================ SEC fails-to-deliver


def parse_ftd_index(html: str) -> dict[str, str]:
    """'YYYYMMa'/'YYYYMMb' -> URL. Files before 2017-07 sit under /files/data/frequently-requested-foia-document-
    fails-deliver-data/, later ones under /files/data/fails-deliver-data/ (guessing one path 404s)."""
    return {
        k: (h if h.startswith("http") else "https://www.sec.gov" + h)
        for h, k in re.findall(r'href="([^"]*cnsfails(\d{6}[ab])\.zip)"', html)
    }


def ftd_urls(net: PitNet) -> dict[str, str]:
    return parse_ftd_index(net.cached_bytes("ftd", "ftd_index.html", FTD_INDEX, "sec", 24).decode("latin-1"))


def parse_ftd_text(txt: str) -> pd.DataFrame:
    """Pipe file 'SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE' -> settle, cusip, symbol, price.
    DESCRIPTION may itself contain '|' (4 fields from the left, price from the right)."""
    rows = [ln.split("|") for ln in txt.splitlines()[1:]]
    rows = [r[:3] + [r[-1]] for r in rows if len(r) >= 6]
    df = pd.DataFrame(rows, columns=FTD_COLUMNS)
    df = df[df["settle"].str.fullmatch(r"\d{8}")]
    df["settle"] = pd.to_datetime(df["settle"], format="%Y%m%d")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["cusip"] = df["cusip"].astype(str).str.strip()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    return df.reset_index(drop=True)


def parse_ftd_zip(content: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(content))
    return parse_ftd_text(z.read(z.namelist()[0]).decode("latin-1"))


def store_ftd(net: PitNet, key: str, df: pd.DataFrame) -> None:
    """Compact cache (settle, cusip, symbol, price; ~0.3 MB per half-month file instead of ~4 MB raw)."""
    c = df[FTD_COLUMNS].copy()
    c["cusip"] = c["cusip"].astype("category")
    c["symbol"] = c["symbol"].astype("category")
    p = net.path("ftd", f"cnsfails{key}.parquet")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    c.to_parquet(tmp, compression="zstd", index=False)
    tmp.replace(p)


def ftd_file(net: PitNet, key: str, urls: dict[str, str]) -> pd.DataFrame:
    p = net.path("ftd", f"cnsfails{key}.parquet")
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:  # noqa: BLE001 - corrupt cache -> re-download
            p.unlink(missing_ok=True)
    df = parse_ftd_zip(net.http_get(urls[key], "sec").content)
    store_ftd(net, key, df)
    return df


def ftd_panel(net: PitNet, symbols: set[str], start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list]:
    """FTD rows of ``symbols`` + every row sharing their CUSIPs (OTC successor symbols, FRC -> FRCB), settle in
    [start, end]. Adds ``px_date`` = the NYSE session before settlement and ``key`` = the tracked symbol.
    Returns (panel, missing file keys)."""
    urls = ftd_urls(net)
    lo, hi = pd.Timestamp(start).strftime("%Y%m"), pd.Timestamp(end).strftime("%Y%m")
    keys = sorted(k for k in urls if lo <= k[:6] <= hi)
    if not keys:
        raise ValueError(f"FTD 파일 없음 {lo}..{hi} (목록 {min(urls, default='-')}..{max(urls, default='-')})")
    own_frames, ok_keys, missing = [], [], []
    for k in keys:
        try:
            f = ftd_file(net, k, urls)
        except Exception as exc:  # noqa: BLE001 - reported in coverage
            missing.append((k, f"{type(exc).__name__}: {str(exc)[:80]}"))
            continue
        ok_keys.append(k)
        own_frames.append(_plain(f[f["symbol"].isin(symbols)]))
    if not own_frames:
        return pd.DataFrame(columns=[*FTD_COLUMNS, "px_date", "key"]), missing
    own = pd.concat(own_frames, ignore_index=True)
    own["key"] = own["symbol"]
    # CUSIP successors (OTC suffix after a failure): rows of a tracked CUSIP under another symbol, keyed by the
    # tracked symbol that used the CUSIP last
    cus = own.sort_values("settle").groupby("cusip")["symbol"].last().to_dict()
    other_frames = []
    for k in ok_keys:
        f = ftd_file(net, k, urls)
        o = f[f["cusip"].isin(list(cus)) & ~f["symbol"].isin(symbols)]
        if len(o):
            other_frames.append(_plain(o))
    out = own
    if other_frames:
        other = pd.concat(other_frames, ignore_index=True)
        other["key"] = other["cusip"].map(cus)
        out = pd.concat([own, other], ignore_index=True)
    out = out.dropna(subset=["price"])
    out = out[(out["price"] > 0) & (out["settle"] >= start) & (out["settle"] <= end)].copy()
    out["px_date"] = prev_sessions(out["settle"]) if len(out) else pd.Series(dtype="datetime64[ns]")
    return out.sort_values(["key", "px_date"], ignore_index=True), missing


def _plain(f: pd.DataFrame) -> pd.DataFrame:
    out = f[FTD_COLUMNS].copy()
    out["cusip"] = out["cusip"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    return out


# ============================================================================ issuer cross-check


def ftd_ratio(df: pd.DataFrame | None, f: pd.DataFrame | None) -> pd.Series:
    """FTD PRICE / Yahoo split-adjusted close on the Yahoo session before settlement (<= 7 days), indexed by that
    session. Same issuer -> piecewise constant (jumps only at later splits); another issuer -> drifts."""
    if df is None or f is None or not len(f) or not len(df):
        return pd.Series(dtype=float)
    f = f.dropna(subset=["price"]).drop_duplicates("settle").sort_values("settle")
    f = f[f["price"] > 0]
    pos = df.index.searchsorted(f["settle"].values, side="left") - 1
    ok = pos >= 0
    days = df.index[pos[ok]]
    fresh = (f["settle"].values[ok] - days.values) <= pd.Timedelta(7, "D").to_timedelta64()
    r = pd.Series(f["price"].values[ok][fresh] / df["close"].values[pos[ok]][fresh], index=days[fresh])
    return r[~r.index.duplicated(keep="last")].replace(np.inf, np.nan).dropna()


def ftd_consistency(r: pd.Series, tol: float = 0.002, min_len: int = 5) -> tuple[int, float, pd.Series]:
    """(n, plateau share, per-observation flags). A plateau is a run of >= ``min_len`` consecutive observations whose
    FTD/Yahoo ratio moves < ``tol`` (log) step to step. 2026-09-25: 631 correct spells >= 0.872 (p1 0.997); wrong
    issuers BBT .011, DD(2017) .05, DISCK->WBD .08, MWV .02, WRK/PARA/VIAC 0; partly wrong CB .75, IR .71, JCI .71."""
    if len(r) < min_len:
        return len(r), float("nan"), pd.Series(True, index=r.index)
    d = np.log(r).diff().abs()
    seg = (d.isna() | (d >= tol)).cumsum()
    okf = seg.map(seg.value_counts()) >= min_len
    return len(r), float(okf.mean()), okf


def ftd_bad_months(df: pd.DataFrame | None, f: pd.DataFrame | None, lo, hi) -> set[pd.Timestamp]:
    """Month-end dates (calendar month end) in [lo, hi] with >= 2 FTD observations mostly off-plateau."""
    if f is None or df is None:
        return set()
    n, _, okf = ftd_consistency(ftd_ratio(df, f[(f["settle"] > lo) & (f["settle"] <= hi)]))
    if n < 5:
        return set()
    g = okf.groupby(okf.index.to_period("M")).agg(["size", "mean"])
    return {p.to_timestamp(how="end").normalize() for p, r in g.iterrows() if r["size"] >= 2 and r["mean"] < 0.5}


def resolve_yahoo(
    spell: pd.Series,
    ev_row: pd.Series | None,
    cik: int | None,
    cur_by_cik: dict[int, list[str]],
    rename_map: dict[str, str],
    ftd_rows: pd.DataFrame | None,
    chart_fn: Callable[[str], dict],
    sec_names_fn: Callable[[int], list[str]],
    today: pd.Timestamp,
    w0: pd.Timestamp,
    min_consistent: float = 0.5,
) -> tuple[str | None, str, list[tuple[str, str]]]:
    """Pick the Yahoo symbol whose series is THIS issuer during the spell.

    Order: patch hint, spell ticker, ticker traded at removal, rename chain (CBS->VIAC->PARA->PSKY), the CIK's
    current SEC tickers (name-checked; TSYS must not become TSS Inc). A candidate needs firstTradeDate before the spell
    end, a bar within 12 days of the end, and (with FTD rows) a plateau share >= ``min_consistent``. A series not
    reaching back to the in-window spell start is a last resort labelled '<how>_partial' if it has at least
    ``MIN_PARTIAL_BARS`` bars in the spell (Yahoo keeps a 1-4 bar stub of some acquired names: 'stub_series').
    Returns (symbol, how, rejected [(symbol, reason)]).
    """
    end = spell["end"] if pd.notna(spell["end"]) else today
    lo = max(pd.Timestamp(spell["start"]), pd.Timestamp(w0))
    if ftd_rows is not None and len(ftd_rows):
        ftd_rows = ftd_rows[(ftd_rows["settle"] > lo) & (ftd_rows["settle"] <= end + pd.Timedelta(3, "D"))]
    partial: tuple[str, str] | None = None
    rejected: list[tuple[str, str]] = []
    cands: list[tuple[str, str]] = []
    hint = spell.get("yahoo_hint")
    if isinstance(hint, str) and hint:
        cands.append((hint, "patch_hint"))
    cands.append((str(spell["ticker"]).replace(".", "-"), "spell_ticker"))
    if ev_row is not None:
        cands.append((str(ev_row["ticker"]), "event_ticker"))
        t, hops = ev_row.get("rename_to"), 0
        while isinstance(t, str) and hops < 6:
            cands.append((t, "rename_to"))
            t, hops = rename_map.get(t), hops + 1
    if cik is not None:
        cur = [t for t in cur_by_cik.get(int(cik), []) if "-P" not in t and not t.endswith(("-WT", "WS"))][:3]
        cands += [(t, "cik_current") for t in cur]
    seen: set[str] = set()
    for s, label in cands:
        if s in seen:
            continue
        seen.add(s)
        c = chart_fn(s)
        if not c.get("ok"):
            continue
        df = c["df"]
        first = chart_first_trade(c)
        near = df.loc[(df.index >= end - pd.Timedelta(12, "D")) & (df.index <= end)]
        if not (first <= end - pd.Timedelta(30, "D") and len(near)):
            if first > end - pd.Timedelta(30, "D"):
                rejected.append((s, "first_trade_after_spell"))
            continue
        if label == "cik_current":
            yn = (c.get("meta") or {}).get("longName") or (c.get("meta") or {}).get("shortName") or ""
            if not any(names_match(yn, n) for n in sec_names_fn(int(cik))):
                rejected.append((s, "name_mismatch"))
                continue
        if ftd_rows is not None and len(ftd_rows):
            n, share, _ = ftd_consistency(ftd_ratio(df, ftd_rows))
            if n >= 5 and share < min_consistent:
                rejected.append((s, f"ftd_issuer_mismatch({share:.2f})"))
                continue
        if df.index[0] <= lo + pd.Timedelta(10, "D"):
            return s, label, rejected
        bars = int(((df.index >= lo) & (df.index <= end)).sum())
        if bars < MIN_PARTIAL_BARS:
            rejected.append((s, f"stub_series({bars})"))
            continue
        partial = partial or (s, label + "_partial")
    if partial:
        return partial[0], partial[1], rejected
    return None, "none", rejected


# ============================================================================ FTD fills and terminal marks


def ftd_price_at(f: pd.DataFrame, when: pd.Timestamp, max_age_days: int = 7) -> tuple[pd.Timestamp, float] | None:
    """Last FTD price dated in (when - max_age_days, when]."""
    if f is None or not len(f):
        return None
    m = f[(f["px_date"] > when - pd.Timedelta(max_age_days, "D")) & (f["px_date"] <= when)]
    if not len(m):
        return None
    r = m.iloc[-1]
    return pd.Timestamp(r["px_date"]), float(r["price"])


def ftd_month_return(
    f: pd.DataFrame | None,
    prev: pd.Timestamp,
    m: pd.Timestamp,
    spell_end: pd.Timestamp | None,
    max_age_days: int = 7,
    cap: bool = True,
) -> float | None:
    """Price-only FTD return over (prev, m]: last price within ``max_age_days`` of each date; when the spell ends inside
    the month, the last price before the end (the first day out). Moves outside (-60%, +100%) are treated as unusable
    (split-like). A halted failing firm keeps its last print (callers pass a longer ``max_age_days``)."""
    if f is None or not len(f):
        return None
    p0 = ftd_price_at(f, prev, max_age_days)
    if p0 is None:
        return None
    if spell_end is not None and pd.notna(spell_end) and spell_end <= m:
        t = f[(f["px_date"] < spell_end) & (f["px_date"] > prev)]
        p1 = float(t["price"].iloc[-1]) if len(t) else None
    else:
        x = ftd_price_at(f, m, max_age_days)
        p1 = x[1] if x else None
    if p1 is None:
        return None
    r = p1 / p0[1] - 1
    return r if not cap or FTD_MOVE_RANGE[0] < r < FTD_MOVE_RANGE[1] else None


def split_like(r: float, tol: float = 0.01) -> bool:
    """A monthly FTD move within 1% of a common split ratio (1/2, 1/3, 1/4, 2, 3): reported, not dropped (every such
    hit in the verification was a real crash/rally)."""
    return bool(np.abs(np.array([1 / 2, 1 / 3, 1 / 4, 2.0, 3.0]) - (1 + r)).min() < tol)


def terminal_mark(
    f: pd.DataFrame | None, removal: pd.Timestamp, failed: bool
) -> tuple[pd.Timestamp, float, str] | None:
    """Delisting mark of a removed member from FTD prices: the last print before the removal date, the first day out
    (within 92 days; later prints of the same symbol are ignored — rows persist after delisting, and the symbol may
    pass to another issuer on the removal date). For a failed firm the first OTC print of the same CUSIP after its
    last exchange print (within 60 days of removal; FRC -> FRCB at $0.33), else 0.0 (-100%: SIVB/SBNY equity was
    wiped out in receivership and has no later print)."""
    own = f[f["symbol"] == f["key"]] if f is not None and "key" in f else f
    last = None
    if own is not None and len(own):
        before = own[(own["px_date"] < removal) & (own["px_date"] > removal - pd.Timedelta(92, "D"))]
        if len(before):
            last = pd.Timestamp(before["px_date"].iloc[-1]), float(before["price"].iloc[-1])
    if failed:
        if f is not None and len(f) and "key" in f:
            lo = last[0] if last else removal - pd.Timedelta(92, "D")
            otc = f[(f["symbol"] != f["key"]) & (f["px_date"] > lo) & (f["px_date"] <= removal + pd.Timedelta(60, "D"))]
            if len(otc):
                return pd.Timestamp(otc["px_date"].iloc[0]), float(otc["price"].iloc[0]), "ftd_otc"
        return removal, 0.0, "failed_assumed_zero"
    if last is None:
        return None
    return last[0], last[1], "ftd"
