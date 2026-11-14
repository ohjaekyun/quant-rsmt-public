"""Long-short / market-neutral books of the served models (docs/MODELING.md 58.1, pre-declared 2026-10-14).
Research only.

    .venv/bin/python scripts/longshort_study.py --market KR

LS10 = top-10% long / bottom-10% short (equal weight, equal dollars); KR also LS10_kr (no position during the full
short-sale bans, short leg limited to KOSPI200 / KOSDAQ150 members during the partial ban); HEDGE = top-20% long + an
index short of the book's 60-session beta. Daily paths, the 47-section cost model on both legs, borrow 0.5% (US) / 2%
(KR) a year on the short notional, index hedge 0.1% a year. Writes reports/strategy/longshort_<M>.{md,json}.
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
import portfolio_backtest as pb  # noqa: E402
import v19_data as v19  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "strategy"
BORROW = {"US": 0.005, "KR": 0.02}
HEDGE_COST = 0.001
BENCH = {"KR": "KOSPI", "US": "SPY"}
KR_FULL_BANS = [("2020-03-16", "2021-05-02"), ("2023-11-06", "2025-03-30")]
KR_PARTIAL = ("2021-05-03", "2023-11-05")


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _in(day: pd.Timestamp, a: str, b: str) -> bool:
    return pd.Timestamp(a) <= day <= pd.Timestamp(b)


def kr_index_members(settings) -> dict[pd.Timestamp, set[str]]:
    d = v19.root(settings) / "krx_index"
    out = {}
    for p in sorted(d.glob("*.parquet")) if d.exists() else []:
        t = pd.read_parquet(p)
        out[pd.Timestamp(p.stem)] = set(t["code"].astype(str))
    return out


def legs(g: pd.DataFrame, kind: str, day: pd.Timestamp, members: dict) -> tuple[pd.Series, pd.Series]:
    """(long weights, short weights) as fractions of NAV for one rebalance cross-section."""
    g = g.dropna(subset=["score"])
    q = g["score"].rank(pct=True)
    if kind == "HEDGE":
        top = g[q > 0.8]
        return pd.Series(1.0 / len(top), index=top["entity"]) if len(top) else pd.Series(dtype=float), pd.Series(
            dtype=float
        )
    top, bot = g[q > 0.9], g[q <= 0.1]
    if kind == "LS10_kr":
        if any(_in(day, a, b) for a, b in KR_FULL_BANS):
            return pd.Series(dtype=float), pd.Series(dtype=float)
        if _in(day, *KR_PARTIAL):
            keys = [k for k in members if k <= day]
            allowed = members[max(keys)] if keys else set()
            bot = g[g["entity"].astype(str).isin(allowed)].nsmallest(len(top), "score")  # same count as the long leg
    if top.empty or bot.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return pd.Series(1.0 / len(top), index=top["entity"]), pd.Series(1.0 / len(bot), index=bot["entity"])


def simulate(
    market: str, sc: pd.DataFrame, w: dict, kind: str, aum: float, mret: pd.Series, members: dict, with_cost: bool
):
    cal = w["close"].index
    dates = sorted(sc["date"].unique())
    execs = [cal[min(cal.searchsorted(pd.Timestamp(d), "right"), len(cal) - 1)] for d in dates]
    by_date = {d: g for d, g in sc.groupby("date")}
    nav = float(aum)
    lv, sv = pd.Series(dtype=float), pd.Series(dtype=float)
    hedge = 0.0
    path, months = [], []

    def tcost(trade: pd.Series, ex) -> float:
        if not with_cost or trade.sum() <= 0:
            return 0.0
        frac = cm.side_cost(
            trade.to_numpy(),
            market,
            w["hs"].loc[ex].reindex(trade.index).to_numpy(dtype=float),
            w["sig"].loc[ex].reindex(trade.index).to_numpy(dtype=float),
            w["adv"].loc[ex].reindex(trade.index).to_numpy(dtype=float),
        )
        return float((trade.to_numpy() * frac).sum())

    for i, (d, ex) in enumerate(zip(dates, execs, strict=True)):
        g = by_date[d]
        ok = w["close"].loc[ex].reindex(g["entity"]).notna().to_numpy()
        lw, sw = legs(g[ok], kind, pd.Timestamp(d), members)
        tl, ts = lw * nav, sw * nav
        trade = pd.concat([tl.sub(lv, fill_value=0).abs(), ts.sub(sv, fill_value=0).abs()])
        trade = trade.groupby(level=0).sum()
        nav -= tcost(trade, ex)
        lv, sv = lw * nav, sw * nav
        if kind == "HEDGE" and len(lv):
            hist = w["ret"].loc[:ex, lv.index].iloc[-61:-1].mean(axis=1)
            mk = mret.reindex(hist.index)
            ok2 = hist.notna() & mk.notna()
            beta = float(np.polyfit(mk[ok2], hist[ok2], 1)[0]) if ok2.sum() > 30 else 1.0
            hedge = beta * nav
        else:
            hedge = 0.0
        end = execs[i + 1] if i + 1 < len(execs) else cal[min(cal.searchsorted(ex) + 21, len(cal) - 1)]
        span = cal[(cal > ex) & (cal <= end)]
        start = nav
        for day in span:
            pnl = 0.0
            if len(lv):
                r = w["ret"].loc[day, lv.index].fillna(0.0)
                pnl += float((lv * r).sum())
                lv = lv * (1 + r)
            if len(sv):
                r = w["ret"].loc[day, sv.index].fillna(0.0)
                pnl -= float((sv * r).sum())
                sv = sv * (1 + r)
                if with_cost:
                    pnl -= BORROW[market] / 252 * float(sv.sum())
            if hedge:
                rm = float(mret.get(day, 0.0)) if np.isfinite(mret.get(day, np.nan)) else 0.0
                pnl -= hedge * rm
                if with_cost:
                    pnl -= HEDGE_COST / 252 * abs(hedge)
                hedge *= 1 + rm
            nav += pnl
            path.append((day, nav))
        months.append(
            {
                "date": d,
                "part": g["part"].iloc[0],
                "ret": nav / start - 1 if start > 0 else 0.0,
                "active": bool(len(lv) or len(sv)),
            }
        )
    return pd.Series(dict(path)).sort_index(), pd.DataFrame(months)


def stats(nav: pd.Series, per: pd.DataFrame, mret: pd.Series) -> dict:
    r = nav.pct_change().dropna()
    yrs = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    mk = mret.reindex(r.index)
    ok = r.notna() & mk.notna() & (r != 0)
    beta = float(np.polyfit(mk[ok], r[ok], 1)[0]) if ok.sum() > 30 else float("nan")
    m = per.loc[per["active"], "ret"]
    out = {
        "cagr": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1),
        "vol": float(r.std() * np.sqrt(252)),
        "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"),
        "mdd": pb.mdd(nav),
        "beta": beta,
        "corr": float(r[ok].corr(mk[ok])) if ok.sum() > 30 else float("nan"),
        "month_mean": float(m.mean()),
        "month_t": float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))) if len(m) > 2 else float("nan"),
        "active_months": int(len(m)),
    }
    for part in ("outer", "holdout"):
        x = per.loc[(per["part"] == part) & per["active"], "ret"]
        out[f"{part}_ann"] = float((1 + x).prod() ** (12 / len(x)) - 1) if len(x) else float("nan")
    return out


def run(market: str) -> dict:
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    say(f"{market}: out-of-sample scores of the served model")
    sc, uni, ds = pb.oos_scores(market, settings, cache)
    cal = pd.DatetimeIndex(ds.calendar)
    cal = cal[cal >= pd.Timestamp(sc["date"].min())]
    w = pb.wide(uni.prices, sorted(sc["entity"].unique()), cal)
    bench = uni.benchmarks[BENCH[market]]["close"].astype(float).sort_index()
    mret = bench.pct_change().reindex(cal)
    members = kr_index_members(settings) if market == "KR" else {}
    mid = cm.AUM[market][1]
    kinds = ["LS10", "HEDGE"] + (["LS10_kr"] if market == "KR" else [])
    res: dict = {"market": market, "model": pb.SERVED[market], "books": {}, "index_months": len(members)}
    for kind in kinds:
        for tag, wc in (("gross", False), ("net", True)):
            nav, per = simulate(market, sc, w, kind, mid, mret, members, wc)
            st = stats(nav, per, mret)
            st["stress"] = {k: pb.window(nav, (1 + mret.fillna(0)).cumprod(), a, b) for k, (a, b) in pb.STRESS.items()}
            res["books"][f"{kind}|{tag}"] = st
        n = res["books"][f"{kind}|net"]
        res["books"][f"{kind}|net"]["alpha"] = bool((n["month_t"] or 0) >= 2 and (n["holdout_ann"] or -1) >= 0)
        say(f"{market} {kind} done")
    return res


def _f(x, spec="+.1%") -> str:
    return "–" if x is None or not np.isfinite(x) else format(x, spec)


def write_md(res: dict) -> str:
    m = res["market"]
    lines = [f"# {m} 롱숏·시장 중립 (docs/MODELING.md 58.1) — 모델 `{res['model']}`", ""]
    lines += [
        "| 포트폴리오 | 연 수익 | 변동성 | 샤프 | 최대 낙폭 | 시장 베타 | 시장 상관 | 월 수익 t | 외부 연 "
        "| 홀드아웃 연 | 판정 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, x in res["books"].items():
        verdict = ("알파 있음" if x.get("alpha") else "알파 없음") if k.endswith("net") else ""
        lines.append(
            f"| {k} | {_f(x['cagr'])} | {_f(x['vol'])} | {_f(x['sharpe'], '.2f')} | {_f(x['mdd'])} | "
            f"{_f(x['beta'], '+.2f')} | {_f(x['corr'], '+.2f')} | {_f(x['month_t'], '+.2f')} | "
            f"{_f(x['outer_ann'])} | {_f(x['holdout_ann'])} | {verdict} |"
        )
    text = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"longshort_{m}.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    res = run(ap.parse_args().market)
    (OUT / f"longshort_{res['market']}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(write_md(res), flush=True)


if __name__ == "__main__":
    main()
