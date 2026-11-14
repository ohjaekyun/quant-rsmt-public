"""Daily forward record of US option chains (docs/MODELING.md 42, started 2026-10-12). Free Yahoo option chains
(no key, unofficial) for the latest S&P 500 members; research data only - nothing here is served.

Outside US trading hours Yahoo blanks bid / ask and its implied volatility (~1e-5), so the implied volatility is
recomputed from the LAST TRADE of the latest session (Black-Scholes, no dividends, risk-free = ^IRX); only contracts
that traded in that session get an IV. Run once a day, any time (e.g. 07:00 KST, Tuesday to Saturday):

    .venv/bin/python scripts/us_options_record.py            # resumable: tickers already recorded are skipped
    .venv/bin/python scripts/us_options_record.py --limit 5  # quick check

Writes .cache/quantcio/alt/us_options/raw/<as_of>.parquet (contracts; git-ignored - back it up) and
alt_ledger/us_options/<YYYY-MM>.parquet (one row per ticker and session; commit it).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v13_data as v13  # noqa: E402

from quantcio.config import PROJECT_ROOT  # noqa: E402

say = v13.say
NY = "America/New_York"
LEDGER = PROJECT_ROOT / "alt_ledger" / "us_options"
MONEYNESS = (0.8, 1.2)
TARGET_DTE = 30
MIN_DTE = 7
PAUSE = 0.25  # seconds between Yahoo calls
SUMMARY_COLS = [
    "as_of",
    "ticker",
    "spot",
    "rate",
    "atm_iv_30",
    "cp_spread",
    "skew",
    "pc_vol",
    "pc_oi",
    "opt_volume",
    "opt_oi",
    "n_traded",
    "quoted_share",
    "expiries",
    "recorded_at",
]


# ------------------------------------------------------------------------------------------------ Black-Scholes


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(s: float, k: float, t: float, r: float, sigma: float, call: bool) -> float:
    if sigma <= 0 or t <= 0:
        return max(0.0, (s - k * math.exp(-r * t)) if call else (k * math.exp(-r * t) - s))
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if call:
        return s * _ncdf(d1) - k * math.exp(-r * t) * _ncdf(d2)
    return k * math.exp(-r * t) * _ncdf(-d2) - s * _ncdf(-d1)


def implied_vol(price: float, s: float, k: float, t: float, r: float, call: bool) -> float:
    """Bisection on [1e-4, 5]; NaN outside the no-arbitrage bounds."""
    if not (np.isfinite(price) and price > 0 and s > 0 and k > 0 and t > 0):
        return np.nan
    lo_b = max(0.0, (s - k * math.exp(-r * t)) if call else (k * math.exp(-r * t) - s))
    hi_b = s if call else k * math.exp(-r * t)
    if not (lo_b < price < hi_b):
        return np.nan
    lo, hi = 1e-4, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_price(s, k, t, r, mid, call) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-6:
            break
    v = 0.5 * (lo + hi)
    return v if 1e-3 < v < 4.99 else np.nan


# ------------------------------------------------------------------------------------------------ fetch


def members() -> list[str]:
    """Yahoo symbols of the latest PIT S&P 500 members."""
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache
    from quantcio.data.pit import load_pit_universe
    from quantcio.modeling.dataset import universe_params
    from quantcio.models import Market

    s = load_settings()
    n, start = universe_params(Market.US, s)
    m = load_pit_universe(Market.US, s, DiskCache(s.cache_path), n, start).membership
    last = m[m["date"] == m["date"].max()]
    return sorted(last["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False).unique())


def _retry(fn, *a, tries: int = 4):
    for i in range(tries):
        try:
            return fn(*a)
        except Exception as exc:  # noqa: BLE001 - Yahoo throttling / transient errors: back off, then give up
            if i == tries - 1:
                raise
            time.sleep(3.0 * (i + 1) + (10.0 if "429" in str(exc) or "Too Many" in str(exc) else 0.0))
    return None


def risk_free() -> float:
    import yfinance as yf

    h = _retry(lambda: yf.Ticker("^IRX").history(period="10d"))
    v = float(h["Close"].dropna().iloc[-1]) / 100.0 if h is not None and len(h) else 0.04
    return v if 0 <= v < 0.2 else 0.04


def pick_expiries(expiries: list[str], as_of: pd.Timestamp) -> list[str]:
    dte = {e: (pd.Timestamp(e) - as_of).days for e in expiries}
    ok = [e for e in expiries if dte[e] >= MIN_DTE]
    if not ok:
        return []
    out = {ok[0]}
    below = [e for e in ok if dte[e] <= TARGET_DTE]
    above = [e for e in ok if dte[e] >= TARGET_DTE]
    if below:
        out.add(below[-1])
    if above:
        out.add(above[0])
    return sorted(out)


def ticker_rows(sym: str, rate: float) -> tuple[pd.DataFrame, pd.Timestamp | None, float]:
    """Filtered contracts of the chosen expiries with own IV; (rows, as_of session, spot)."""
    import yfinance as yf

    tk = yf.Ticker(sym)
    expiries = list(_retry(lambda: tk.options) or [])
    time.sleep(PAUSE)
    if not expiries:
        return pd.DataFrame(), None, np.nan
    first = _retry(tk.option_chain, expiries[0])
    time.sleep(PAUSE)
    und = getattr(first, "underlying", None) or {}
    spot = float(und.get("regularMarketPrice") or np.nan)
    mt = und.get("regularMarketTime")
    as_of = pd.Timestamp(mt, unit="s", tz="UTC").tz_convert(NY).normalize().tz_localize(None) if mt else None
    if as_of is None or not np.isfinite(spot) or spot <= 0:
        return pd.DataFrame(), None, np.nan
    parts = []
    for e in pick_expiries(expiries, as_of):
        ch = first if e == expiries[0] else _retry(tk.option_chain, e)
        if e != expiries[0]:
            time.sleep(PAUSE)
        for kind, df in (("C", ch.calls), ("P", ch.puts)):
            if df is None or df.empty:
                continue
            d = df.copy()
            d["type"], d["expiry"] = kind, pd.Timestamp(e)
            parts.append(d)
    if not parts:
        return pd.DataFrame(), as_of, spot
    d = pd.concat(parts, ignore_index=True)
    d["moneyness"] = d["strike"] / spot
    d = d[d["moneyness"].between(*MONEYNESS)].copy()
    ltd = pd.to_datetime(d["lastTradeDate"], utc=True, errors="coerce").dt.tz_convert(NY).dt.normalize()
    d["traded"] = (ltd.dt.tz_localize(None) == as_of).to_numpy()
    d["dte"] = (d["expiry"] - as_of).dt.days
    bid, ask = d["bid"].fillna(0.0), d["ask"].fillna(0.0)
    d["px"] = np.where((bid > 0) & (ask > 0), 0.5 * (bid + ask), np.where(d["traded"], d["lastPrice"], np.nan))
    d["iv"] = [
        implied_vol(p, spot, k, max(t, 1) / 365.0, rate, ty == "C")
        for p, k, t, ty in d[["px", "strike", "dte", "type"]].itertuples(index=False)
    ]
    d.insert(0, "ticker", sym)
    d.insert(1, "as_of", as_of)
    d["spot"] = spot
    keep = [
        "ticker", "as_of", "spot", "type", "expiry", "dte", "strike", "moneyness", "lastPrice", "bid", "ask",
        "volume", "openInterest", "impliedVolatility", "traded", "px", "iv", "lastTradeDate",
    ]  # fmt: skip
    return d[keep].reset_index(drop=True), as_of, spot


def _atm_iv(g: pd.DataFrame) -> float:
    """ATM IV of one expiry: the mean call / put IV, linear in moneyness around 1."""
    vals = []
    for ty in ("C", "P"):
        x = g[(g["type"] == ty) & g["iv"].notna()].sort_values("moneyness")
        if len(x) == 0:
            continue
        lo, hi = x[x["moneyness"] <= 1.0], x[x["moneyness"] >= 1.0]
        if len(lo) and len(hi):
            a, b = lo.iloc[-1], hi.iloc[0]
            w = 0.0 if b["moneyness"] == a["moneyness"] else (1.0 - a["moneyness"]) / (b["moneyness"] - a["moneyness"])
            vals.append((1 - w) * a["iv"] + w * b["iv"])
        elif abs(x["moneyness"] - 1.0).min() <= 0.03:
            vals.append(float(x.loc[(x["moneyness"] - 1.0).abs().idxmin(), "iv"]))
    return float(np.mean(vals)) if vals else np.nan


def summary(d: pd.DataFrame, rate: float) -> dict:
    """One ticker-session row (docs/MODELING.md 42)."""
    quoted = (d["bid"].fillna(0) > 0) & (d["ask"].fillna(0) > 0)  # live quotes (US session) vs last trades only
    out = {
        "rate": rate,
        "n_traded": int(d["traded"].sum()),
        "quoted_share": round(float(quoted.mean()), 3),
        "expiries": ",".join(sorted({str(e.date()) for e in d["expiry"]})),
    }
    by_exp = {e: g for e, g in d.groupby("dte")}
    atm = {t: _atm_iv(g) for t, g in by_exp.items()}
    atm = {t: v for t, v in atm.items() if np.isfinite(v)}
    if atm:
        ts = sorted(atm)
        lo = [t for t in ts if t <= TARGET_DTE]
        hi = [t for t in ts if t >= TARGET_DTE]
        if lo and hi and lo[-1] != hi[0]:
            t1, t2 = lo[-1], hi[0]
            v1, v2 = atm[t1] ** 2 * t1, atm[t2] ** 2 * t2  # interpolate total variance
            w = (TARGET_DTE - t1) / (t2 - t1)
            out["atm_iv_30"] = float(math.sqrt(max(((1 - w) * v1 + w * v2) / TARGET_DTE, 0.0)))
        else:
            t = min(ts, key=lambda x: abs(x - TARGET_DTE))
            out["atm_iv_30"] = float(atm[t])
    near = min(by_exp, key=lambda x: abs(x - TARGET_DTE)) if by_exp else None
    if near is not None:
        g = by_exp[near]
        c = g[(g["type"] == "C") & g["iv"].notna()].set_index("strike")["iv"]
        p = g[(g["type"] == "P") & g["iv"].notna()].set_index("strike")["iv"]
        both = c.index.intersection(p.index)
        both = [k for k in both if 0.95 <= k / g["spot"].iloc[0] <= 1.05]
        out["cp_spread"] = float((c[both] - p[both]).mean()) if both else np.nan
        pu = g[(g["type"] == "P") & g["iv"].notna()]
        ca = g[(g["type"] == "C") & g["iv"].notna()]
        if len(pu) and len(ca):
            otm = pu.loc[(pu["moneyness"] - 0.90).abs().idxmin()]
            atmc = ca.loc[(ca["moneyness"] - 1.0).abs().idxmin()]
            ok = abs(otm["moneyness"] - 0.90) <= 0.03 and abs(atmc["moneyness"] - 1.0) <= 0.03
            out["skew"] = float(otm["iv"] - atmc["iv"]) if ok else np.nan
    vol = d["volume"].fillna(0.0)
    oi = d["openInterest"].fillna(0.0)
    cv, pv = float(vol[d["type"] == "C"].sum()), float(vol[d["type"] == "P"].sum())
    co, po = float(oi[d["type"] == "C"].sum()), float(oi[d["type"] == "P"].sum())
    out.update(
        {
            "pc_vol": pv / cv if cv > 0 else np.nan,
            "pc_oi": po / co if co > 0 and po > 0 else np.nan,  # Yahoo zeroes open interest outside trading hours
            "opt_volume": cv + pv,
            "opt_oi": co + po if co + po > 0 else np.nan,
        }
    )
    return out


def _ledger_path(as_of: pd.Timestamp) -> Path:
    return LEDGER / f"{as_of:%Y-%m}.parquet"


def recorded(as_of: pd.Timestamp) -> set[str]:
    p = _ledger_path(as_of)
    if not p.exists():
        return set()
    s = pd.read_parquet(p, columns=["as_of", "ticker"])
    return set(s.loc[pd.to_datetime(s["as_of"]) == as_of, "ticker"])


def append(rows: list[dict], raw: list[pd.DataFrame], as_of: pd.Timestamp, raw_dir: Path) -> None:
    LEDGER.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    p = _ledger_path(as_of)
    new = pd.DataFrame(rows).reindex(columns=SUMMARY_COLS)
    allr = pd.concat([pd.read_parquet(p), new], ignore_index=True) if p.exists() else new
    allr["as_of"] = pd.to_datetime(allr["as_of"])
    allr = allr.drop_duplicates(["as_of", "ticker"], keep="last").sort_values(["as_of", "ticker"])
    allr.to_parquet(p, index=False)
    if raw:
        rp = raw_dir / f"{as_of:%Y-%m-%d}.parquet"
        rr = pd.concat(raw, ignore_index=True)
        if rp.exists():
            rr = pd.concat([pd.read_parquet(rp), rr], ignore_index=True)
        rr.drop_duplicates(["ticker", "type", "expiry", "strike"], keep="last").to_parquet(rp, index=False)


def main() -> None:
    from quantcio.config import load_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N tickers only (check)")
    a = ap.parse_args()
    raw_dir = Path(load_settings().cache_path) / "alt" / "us_options" / "raw"
    syms = members()
    if a.limit:
        syms = syms[: a.limit]
    rate = risk_free()
    say(f"US options: {len(syms)} tickers, risk-free {rate:.4f}")
    done: dict[pd.Timestamp, set[str]] = {}
    rows, raw, failed, n = [], [], [], 0
    as_of_run = None
    for i, sym in enumerate(syms, 1):
        try:
            d, as_of, spot = ticker_rows(sym, rate)
        except Exception as exc:  # noqa: BLE001 - one ticker failing must not stop the record
            failed.append(f"{sym}: {type(exc).__name__}")
            continue
        if as_of is None or d.empty:
            failed.append(f"{sym}: no chain")
            continue
        if as_of not in done:
            done[as_of] = recorded(as_of)
        if sym in done[as_of]:
            continue
        as_of_run = as_of
        rows.append(
            {"as_of": as_of, "ticker": sym, "spot": spot, **summary(d, rate), "recorded_at": pd.Timestamp.now(tz=NY)}
        )
        raw.append(d)
        n += 1
        if len(rows) >= 50:
            append(rows, raw, as_of, raw_dir)
            rows, raw = [], []
            say(f"US options {i}/{len(syms)} ({n} recorded, {len(failed)} failed)")
    if rows:
        append(rows, raw, as_of_run, raw_dir)
    say(f"US options done: {n} recorded for {sorted(str(x.date()) for x in done)}, failed {len(failed)} {failed[:8]}")


if __name__ == "__main__":
    main()
