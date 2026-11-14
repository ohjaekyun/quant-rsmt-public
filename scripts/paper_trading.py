"""Paper trading of the served signals (docs/MODELING.md 50, started 2026-10-12). No orders are sent anywhere.

Portfolios per market (as in docs/MODELING.md 48): P1 top-quintile equal weight, P2 cap weight without the bottom
quintile, benchmarks B1 universe equal weight and B2 universe cap weight (no costs); P1B (docs/MODELING.md 80.8,
added 2026-10-25) = P1 with a turnover buffer: a held name stays while its score is in the top 40%, the book is
filled back to the top-quintile count with the best-scored new names. Scores = the serving artifact's
latest cross-section; a new artifact date (monthly) triggers a rebalance at the latest Yahoo close. Notional capital:
KR 1,000,000,000 KRW, US 10,000,000 USD; costs = scripts/costs.py at that size.

    .venv/bin/python scripts/paper_trading.py run       # mark to market, rebalance when the scores changed
    .venv/bin/python scripts/paper_trading.py report    # realised vs backtest-expected excess (z, 95% band)

Ledgers (git-tracked, append-only): alt_ledger/paper/{trades,nav}.jsonl; state: alt_ledger/paper/state.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import costs as cm  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402

LEDGER = PROJECT_ROOT / "alt_ledger" / "paper"
EXPECT = PROJECT_ROOT / "reports" / "strategy"
CAPITAL = {"KR": 1e9, "US": 1e7}
KINDS = ("P1", "P2", "B1", "B2", "P1B")
BENCH = {"P1": "B1", "P2": "B2", "P1B": "B1"}
BUFFER_KEEP = 0.6  # P1B: keep a held name while its score percentile >= this


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _append(name: str, rows: list[dict]) -> None:
    LEDGER.mkdir(parents=True, exist_ok=True)
    with (LEDGER / name).open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def load_state() -> dict:
    p = LEDGER / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(st: dict) -> None:
    LEDGER.mkdir(parents=True, exist_ok=True)
    (LEDGER / "state.json").write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str))


def latest_scores(market: str, settings, cache) -> tuple[str, pd.DataFrame]:
    from quantcio.modeling.serve import load_artifact
    from quantcio.models import Market

    art = load_artifact(Market(market), settings, cache)
    if art is None:
        raise SystemExit(f"{market}: 서빙 산출물 없음")
    lt = art.latest[["entity", "ticker", "log_mcap", "score"]].copy()
    return str(pd.Timestamp(art.as_of).date()), lt


def yahoo_symbols(market: str, lt: pd.DataFrame, settings, cache) -> dict[str, str]:
    if market == "US":
        return {t: str(t).upper().replace(".", "-") for t in lt["ticker"]}
    from quantcio.data.pit import load_pit_universe
    from quantcio.modeling.dataset import universe_params
    from quantcio.models import Market

    n, start = universe_params(Market.KR, settings)
    m = load_pit_universe(Market.KR, settings, cache, n, start).membership
    mk = m.sort_values("date").groupby("ticker")["market"].last().to_dict()
    return {t: f"{t}{'.KQ' if mk.get(t) == 'KOSDAQ' else '.KS'}" for t in lt["ticker"]}


def prices(symbols: list[str], days: int = 90) -> dict[str, pd.DataFrame]:
    """Yahoo daily OHLCV (not dividend-adjusted closes) of the symbols."""
    import yfinance as yf

    out = {}
    for i in range(0, len(symbols), 100):
        chunk = symbols[i : i + 100]
        d = yf.download(chunk, period=f"{days}d", auto_adjust=False, group_by="ticker", progress=False, threads=True)
        for s in chunk:
            try:
                x = d[s] if len(chunk) > 1 else d
            except KeyError:
                continue
            x = x.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
            if len(x):
                x.index = pd.DatetimeIndex(x.index).tz_localize(None)
                out[s] = x
        time.sleep(1.0)
    return out


def targets(lt: pd.DataFrame, kind: str, held: set[str] | None = None) -> pd.Series:
    """Target weights by ticker (``held``: the current P1B names, for its buffer)."""
    g = lt.dropna(subset=["score"])
    q = g["score"].rank(pct=True)
    if kind == "P1B":
        k = int((q > 0.8).sum())
        keep = g[(q >= BUFFER_KEEP).to_numpy() & g["ticker"].isin(held or set()).to_numpy()].nlargest(k, "score")
        add = g[~g["ticker"].isin(set(keep["ticker"]))].nlargest(max(k - len(keep), 0), "score")
        names = [*keep["ticker"], *add["ticker"]]
        return pd.Series(1.0 / len(names), index=names) if names else pd.Series(dtype=float)
    if kind == "P1":
        s = g[q > 0.8]
        return pd.Series(1.0 / len(s), index=s["ticker"])
    if kind == "B1":
        return pd.Series(1.0 / len(g), index=g["ticker"])
    s = g[q > 0.2] if kind == "P2" else g
    m = np.exp(s.set_index("ticker")["log_mcap"].astype(float))
    m = m[np.isfinite(m) & (m > 0)]  # a few names lack a market cap in the artifact: left out
    return m / m.sum()


def run() -> None:
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    st = load_state()
    today = str(pd.Timestamp.now(tz="Asia/Seoul").date())
    for market in ("KR", "US"):
        as_of, lt = latest_scores(market, settings, cache)
        sym = yahoo_symbols(market, lt, settings, cache)
        ms = st.setdefault(market, {})
        held = {t for k in KINDS for t in (ms.get(k, {}).get("shares") or {})}
        tickers = sorted(set(lt["ticker"]) | held)
        px = prices(sorted({sym.get(t, t) for t in tickers}))
        close = {t: float(px[sym.get(t, t)]["close"].iloc[-1]) for t in tickers if sym.get(t, t) in px}
        pdate = max((str(px[s].index[-1].date()) for s in px), default=today)
        ci = {t: cm.cost_inputs(px[sym.get(t, t)]).iloc[-1] for t in tickers if sym.get(t, t) in px}
        rebalance = ms.get("as_of") != as_of
        trades, navs = [], []
        for kind in KINDS:
            pf = ms.setdefault(kind, {"cash": CAPITAL[market], "shares": {}, "start": pdate})
            shares = {t: float(v) for t, v in pf["shares"].items()}
            nav = pf["cash"] + sum(v * close.get(t, 0.0) for t, v in shares.items())
            if rebalance:
                tw = targets(lt[lt["ticker"].isin(close)], kind, set(shares))
                tw = tw / tw.sum()
                names = sorted(set(tw.index) | set(shares))
                cur = np.array([shares.get(t, 0.0) * close[t] if t in close else 0.0 for t in names])
                tgt = np.array([tw.get(t, 0.0) * nav for t in names])
                trade = np.abs(tgt - cur)
                cost = 0.0
                if kind in BENCH:  # portfolios pay costs, benchmarks are indices
                    frac = cm.side_cost(
                        trade,
                        market,
                        np.array([ci[t]["half_spread"] if t in ci else np.nan for t in names]),
                        np.array([ci[t]["sigma_d"] if t in ci else np.nan for t in names]),
                        np.array([ci[t]["adv"] if t in ci else np.nan for t in names]),
                    )
                    cost = float((trade * frac).sum())
                nav_after = nav - cost
                new = {
                    t: float(tw.get(t, 0.0) * nav_after / close[t]) for t in names if t in close and tw.get(t, 0) > 0
                }
                for t, tv, cv in zip(names, tgt, cur, strict=True):
                    if abs(tv - cv) > 1e-9 * max(nav, 1.0):
                        trades.append(
                            {"date": pdate, "market": market, "portfolio": kind, "ticker": t,
                             "value": float(tv - cv), "price": close.get(t), "scores_as_of": as_of}
                        )  # fmt: skip
                pf.update({"shares": new, "cash": 0.0, "last_rebalance": pdate, "last_cost": cost})
                nav = nav_after
            pf["nav"] = nav
            navs.append({"date": pdate, "market": market, "portfolio": kind, "nav": nav, "n": len(pf["shares"])})
        if rebalance:
            ms["as_of"] = as_of
            say(f"{market}: rebalanced on the scores of {as_of} at {pdate} closes ({len(trades)} trades)")
        _append("trades.jsonl", trades)
        _append("nav.jsonl", [dict(r, recorded=today) for r in navs])
        parts = ", ".join("{} {:,.0f}".format(r["portfolio"], r["nav"]) for r in navs)
        say(f"{market}: NAV {parts} ({pdate})")
    save_state(st)


def report() -> str:
    p = LEDGER / "nav.jsonl"
    if not p.exists():
        return "기록 없음"
    nav = pd.DataFrame([json.loads(x) for x in p.read_text().splitlines()])
    nav = nav.sort_values("recorded").drop_duplicates(["date", "market", "portfolio"], keep="last")
    lines = ["# 가상 매매 추적 (docs/MODELING.md 50)", ""]
    for market, g in nav.groupby("market"):
        ep = EXPECT / f"portfolio_expect_{market}.json"
        ex = json.loads(ep.read_text()) if ep.exists() else {}
        w = g.pivot(index="date", columns="portfolio", values="nav").sort_index()
        w.index = pd.to_datetime(w.index)
        m = w.resample("ME").last().dropna(how="all")
        start = w.iloc[0]
        lines.append(f"## {market} ({w.index[0].date()} ~ {w.index[-1].date()}, 기록 {len(w)}일)")
        for pfn, bn in BENCH.items():
            if pfn not in w or bn not in w:
                continue
            cum = float(w[pfn].iloc[-1] / start[pfn] - w[bn].iloc[-1] / start[bn])
            mr = (m[pfn].pct_change() - m[bn].pct_change()).dropna()
            e = ex.get(pfn, {})
            mu, sd = e.get("monthly_excess_mean"), e.get("monthly_excess_sd")
            n = len(mr)
            line = f"- {pfn} − {bn}: 누적 초과 {cum:+.2%}, 완결 월 {n}개"
            if mu is not None and sd and n:
                z = (mr.mean() - mu) / (sd / np.sqrt(n))
                band = (mu * n - 1.96 * sd * np.sqrt(n), mu * n + 1.96 * sd * np.sqrt(n))
                streak = int((((mr - mu) / sd).abs() > 2).astype(int).iloc[-3:].sum()) if n >= 3 else 0
                warn = " ⚠ 기대 범위 밖" if not band[0] <= mr.sum() <= band[1] or streak == 3 else ""
                line += (
                    f", 월 초과 평균 {mr.mean():+.2%} vs 백테스트 {mu:+.2%} (z {z:+.2f}), "
                    f"누적 기대 95% 범위 [{band[0]:+.2%}, {band[1]:+.2%}]{warn}"
                )
            lines.append(line)
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "run":
        run()
    else:
        print(report())


if __name__ == "__main__":
    main()
