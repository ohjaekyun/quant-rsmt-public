"""Event-risk exposure cap (proposal D) - pre-declared market-level test (docs/RSMT.md section 7.1). Research only.

Usage:
    .venv/bin/python scripts/event_cap.py [--out reports/rsmt]

Benchmark per market (KR: KOSPI incl. the marcap replica splice from `data.pit.kr`, US: SPY), daily closes from
1996 (price index; cash earns 0 - a slight bias AGAINST de-risking). Gross exposure is set at the close of t and earns
the return of t -> t+1, 10 bp per unit of exposure change.
- D0: volatility target at market level (the risk module's rule): min(1, 15% / EWMA vol, lambda 0.94);
- D1: D0 x 0.6 while ev_vix (VIX of the last US session known at t's close) is above its trailing 3-year 90th
  percentile; D2: the same for ev_gpr (geopolitical risk, 8-day publication lag) above its 95th percentile;
  D3: either. Percentiles use data <= t only.
Rule: adopt a trigger only if its Sharpe difference vs D0 has a 21-session block-bootstrap 90% lower bound > 0 in BOTH
markets before the holdout (the last 24 months); drawdown-only improvements are reported as a preference choice.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rsmt_ts as T

log = logging.getLogger("event_cap")

START = pd.Timestamp("1996-01-01")
HOLDOUT_MONTHS = 24
TARGET_VOL = 0.15
CAP = 0.6
COST = 0.0010
WINDOW = 756  # 3 years of sessions
Q_VIX, Q_GPR = 0.90, 0.95
BLOCK = 21
N_BOOT = 2000
EPISODES = {
    "1997-98 Asian crisis": ("1997-07-01", "1998-12-31"),
    "2000-02 dot-com": ("2000-03-01", "2002-10-31"),
    "2008 GFC": ("2008-09-01", "2009-03-31"),
    "2011 euro / US downgrade": ("2011-07-01", "2011-10-31"),
    "2018 trade war": ("2018-01-26", "2018-12-31"),
    "2020 COVID": ("2020-02-15", "2020-04-30"),
    "2022 rate shock / Ukraine": ("2022-01-01", "2022-10-31"),
    "2024-25 KR martial law / US tariffs": ("2024-12-01", "2025-05-31"),
}


def ewma_vol(r: pd.Series, lam: float = 0.94) -> pd.Series:
    return np.sqrt(252.0 * (r * r).ewm(alpha=1 - lam, adjust=False).mean())


def exposures(close: pd.Series, trig: dict[str, pd.Series]) -> pd.DataFrame:
    """Exposure decided at each close (data <= t)."""
    r = np.log(close).diff()
    d0 = np.minimum(1.0, TARGET_VOL / ewma_vol(r)).fillna(0.0)
    out = pd.DataFrame({"D0": d0})
    on_vix = trig["vix"].astype(bool)
    on_gpr = trig["gpr"].astype(bool)
    out["D1"] = d0 * np.where(on_vix, CAP, 1.0)
    out["D2"] = d0 * np.where(on_gpr, CAP, 1.0)
    out["D3"] = d0 * np.where(on_vix | on_gpr, CAP, 1.0)
    out["BH"] = 1.0
    return out


def strategy_returns(close: pd.Series, expo: pd.DataFrame) -> pd.DataFrame:
    """Simple returns t -> t+1 earned by the exposure set at t, net of costs on exposure changes."""
    ret = close.pct_change().shift(-1)
    out = {}
    for c in expo:
        e = expo[c]
        out[c] = e * ret - COST * e.diff().abs().fillna(e.abs())
    return pd.DataFrame(out).dropna()


def perf(r: pd.Series) -> dict[str, float]:
    if len(r) < 20:
        return {}
    wealth = (1 + r).cumprod()
    years = len(r) / 252.0
    ann = float(wealth.iloc[-1] ** (1 / years) - 1)
    vol = float(r.std(ddof=1) * math.sqrt(252))
    mdd = float((wealth / wealth.cummax() - 1).min())
    return {
        "ann_return": ann,
        "vol": vol,
        "sharpe": float(r.mean() / r.std(ddof=1) * math.sqrt(252)) if r.std() > 0 else float("nan"),
        "mdd": mdd,
        "calmar": ann / abs(mdd) if mdd < 0 else float("nan"),
        "days": int(len(r)),
    }


def sharpe_diff_ci(a: pd.Series, b: pd.Series, seed: int = 0) -> tuple[float, float, float]:
    """Sharpe(a) - Sharpe(b) with a moving-block bootstrap (BLOCK sessions) 90% interval."""
    x = np.column_stack([a.to_numpy(), b.to_numpy()])
    n = len(x)

    def sh(m: np.ndarray) -> np.ndarray:
        return m.mean(axis=0) / m.std(axis=0, ddof=1) * math.sqrt(252)

    point = float(np.subtract(*sh(x)))
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / BLOCK))
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK + 1, nb)
        idx = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n]
        diffs[i] = float(np.subtract(*sh(x[idx])))
    return point, float(np.quantile(diffs, 0.05)), float(np.quantile(diffs, 0.95))


def triggers(dates: pd.DatetimeIndex, market: str, events: pd.DataFrame) -> dict[str, pd.Series]:
    ev = T.event_columns(pd.Series(dates, index=dates), market, events)

    def above(s: pd.Series, q: float) -> pd.Series:
        thr = s.rolling(WINDOW, min_periods=252).quantile(q)
        return (s > thr).fillna(False)

    return {"vix": above(ev["ev_vix"], Q_VIX), "gpr": above(ev["ev_gpr"], Q_GPR)}


def benchmark(market: str, settings: Any, cache: Any) -> pd.Series:
    if market == "US":
        from quantcio.macro.data import fetch_yahoo

        close, _, _ = fetch_yahoo("SPY", date(1993, 1, 1), settings, cache)
        return close.sort_index()
    from quantcio.models import Market
    from quantcio.pipeline import training_panel

    s = settings.model_copy(deep=True)
    s.pit.kr.start_year = 1995
    s.history_years = 31
    price, _, _ = training_panel(Market.KR, s, cache)
    return price.benchmarks["KOSPI"]["close"].sort_index()


def run(out_dir: Path) -> dict[str, Any]:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache

    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    events = T.load_events(settings, cache)
    res: dict[str, Any] = {"generated_at": datetime.now().isoformat(timespec="seconds"), "markets": {}}
    for mk in ("KR", "US"):
        close = benchmark(mk, settings, cache)
        close = close[close.index >= START - pd.DateOffset(years=4)].dropna()
        trig = triggers(close.index, mk, events)
        expo = exposures(close, trig)
        rets = strategy_returns(close, expo)
        rets = rets[rets.index >= START]
        hold_start = rets.index[-1] - pd.DateOffset(months=HOLDOUT_MONTHS)
        periods = {
            "pre": rets[rets.index < hold_start],
            "holdout": rets[rets.index >= hold_start],
            "full": rets,
        }
        m: dict[str, Any] = {
            "first": str(rets.index[0].date()),
            "holdout_start": str(hold_start.date()),
            "trigger_share": {k: float(v[v.index >= START].mean()) for k, v in trig.items()},
            "mean_exposure": {c: float(expo.loc[expo.index >= START, c].mean()) for c in expo},
            "periods": {},
            "episodes": {},
        }
        for pname, pr in periods.items():
            row = {c: perf(pr[c]) for c in pr}
            for c in ("D1", "D2", "D3"):
                pt, lo, hi = sharpe_diff_ci(pr[c], pr["D0"])
                row[c]["sharpe_diff_vs_d0"] = {"point": pt, "lo90": lo, "hi90": hi}
            m["periods"][pname] = row
        for ename, (a, b) in EPISODES.items():
            pr = rets[(rets.index >= a) & (rets.index <= b)]
            if len(pr) > 10:
                m["episodes"][ename] = {c: float((1 + pr[c]).prod() - 1) for c in pr}
        res["markets"][mk] = m
        log.info("%s done: %s", mk, {c: m["periods"]["pre"][c].get("sharpe") for c in ("BH", "D0", "D1", "D2", "D3")})
    verdict = {}
    for c in ("D1", "D2", "D3"):
        ok = all(res["markets"][mk]["periods"]["pre"][c]["sharpe_diff_vs_d0"]["lo90"] > 0 for mk in ("KR", "US"))
        verdict[c] = "adopt" if ok else "not adopted"
    res["verdict"] = verdict
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "event_cap.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (out_dir / "event_cap.md").write_text(render(res), encoding="utf-8")
    return res


def render(res: dict[str, Any]) -> str:
    md = [f"# Event-risk exposure cap (pre-declared, docs/RSMT.md §7.1) — {res['generated_at']}", ""]
    for mk, m in res["markets"].items():
        md += [
            f"## {mk} (from {m['first']}, holdout from {m['holdout_start']})",
            "",
            f"- trigger share of sessions: {', '.join(f'{k} {v:.1%}' for k, v in m['trigger_share'].items())}; mean "
            f"exposure {', '.join(f'{k} {v:.2f}' for k, v in m['mean_exposure'].items())}",
            "",
        ]
        for pname, row in m["periods"].items():
            md += [
                f"**{pname}**",
                "",
                "| rule | ann. return | vol | Sharpe | Sharpe − D0 [90%] | MDD | Calmar |",
                "|---|---|---|---|---|---|---|",
            ]
            for c, p in row.items():
                if not p:
                    continue
                sd = p.get("sharpe_diff_vs_d0")
                sd_txt = f"{sd['point']:+.3f} [{sd['lo90']:+.3f}, {sd['hi90']:+.3f}]" if sd else "—"
                md.append(
                    f"| {c} | {p['ann_return']:+.2%} | {p['vol']:.1%} | {p['sharpe']:.3f} | {sd_txt} | "
                    f"{p['mdd']:.1%} | {p['calmar']:.2f} |"
                )
            md.append("")
        md += ["| episode | " + " | ".join(("BH", "D0", "D1", "D2", "D3")) + " |", "|---|---|---|---|---|---|"]
        for e, v in m["episodes"].items():
            md.append(f"| {e} | " + " | ".join(f"{v[c]:+.1%}" for c in ("BH", "D0", "D1", "D2", "D3")) + " |")
        md.append("")
    md += ["**Verdict (rule):** " + ", ".join(f"{k}: {v}" for k, v in res["verdict"].items()), ""]
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="reports/rsmt")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    res = run(Path(args.out))
    print("verdict", res["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
