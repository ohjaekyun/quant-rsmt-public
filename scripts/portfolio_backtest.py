"""Signal-portfolio backtest with the cost model and crisis windows (docs/MODELING.md 47-48, pre-declared 2026-10-12).

Scores = the served model's out-of-sample predictions (KR et on the v3 inputs, US cat_rank on the v9 inputs; outer folds
+ the 24-month holdout of protocol v3, recomputed here). Portfolios: P1 top-quintile equal weight, P2 cap weight
without the bottom quintile; benchmarks B1 universe equal weight, B2 universe cap weight. Month-end scores are traded at
the NEXT session's close and held to the next rebalance; daily NAV from daily closes. Research only.

    .venv/bin/python scripts/portfolio_backtest.py --market KR

Writes reports/strategy/portfolio_<M>.{md,json}, the per-period table portfolio_periods_<M>.parquet and the paper-
trading expectation reports/strategy/portfolio_expect_<M>.json.
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
import v13_study as v13s  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "strategy"
SERVED = {"KR": "et", "US": "cat_rank"}
PORTFOLIOS = ("P1", "P2")
BENCH = {"P1": "B1", "P2": "B2"}
STRESS = {
    "2018Q4 급락": ("2018-10-01", "2018-12-24"),
    "코로나 급락": ("2020-02-19", "2020-03-23"),
    "코로나 급락~반등": ("2020-02-19", "2020-06-30"),
    "2022 긴축 약세장": ("2022-01-03", "2022-10-14"),
    "2024-12 계엄(한국)": ("2024-12-03", "2024-12-31"),
    "2025-04 관세 충격": ("2025-04-01", "2025-04-30"),
}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ scores


def oos_scores(market: str, settings, cache) -> tuple[pd.DataFrame, object, object]:
    """(date, entity, score, part, mcap) of the served model's OOS predictions, the universe and the dataset."""
    uni, ds = v13s.load_inputs(market, settings, cache)
    every = {c.name: c for c in cmod.default_candidates()}
    name = SERVED[market]
    sel.COMPOSITE_CANDIDATES, sel.COMPOSITES, sel.DIAGNOSTIC_PAIRS = (), (), ()
    ds.snapshot = f"{ds.snapshot}|p48"
    res = sel.run_selection(ds, settings, cache, candidates=[every["mom_12_1"], every[name]], threads=8, progress=say)
    parts = [res.oos[["date", "entity", name]].assign(part="outer")]
    if res.holdout_preds is not None:
        parts.append(res.holdout_preds[["date", "entity", name]].assign(part="holdout"))
    sc = pd.concat(parts, ignore_index=True).rename(columns={name: "score"})
    mc = ds.frame[["date", "entity", "log_mcap"]].drop_duplicates(["date", "entity"])
    sc = sc.merge(mc, on=["date", "entity"], how="left")
    sc["mcap"] = np.exp(sc["log_mcap"].astype(float))
    return sc.drop(columns="log_mcap"), uni, ds


# ------------------------------------------------------------------------------------------------ simulation


def wide(prices: dict, ents: list[str], cal: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    closes, hs, sig, adv = {}, {}, {}, {}
    for e in ents:
        px = prices.get(e)
        if px is None or "close" not in px or len(px) < 30:
            continue
        px = px.sort_index()
        closes[e] = px["close"].astype(float)
        if {"high", "low", "volume"} <= set(px.columns):
            ci = cm.cost_inputs(px)
            hs[e], sig[e], adv[e] = ci["half_spread"], ci["sigma_d"], ci["adv"]
    c = pd.DataFrame(closes).reindex(cal)
    c = c.ffill()  # halts carry the last price; after delisting the position is cash at the last price
    return {
        "close": c,
        "ret": c.pct_change().fillna(0.0),
        "hs": pd.DataFrame(hs).reindex(cal).ffill(),
        "sig": pd.DataFrame(sig).reindex(cal).ffill(),
        "adv": pd.DataFrame(adv).reindex(cal).ffill(),
    }


def targets(g: pd.DataFrame, kind: str) -> pd.Series:
    g = g.dropna(subset=["score"])
    if g.empty:
        return pd.Series(dtype=float)
    q = g["score"].rank(pct=True)
    if kind == "P1":
        s = g[q > 0.8]
        return pd.Series(1.0 / len(s), index=s["entity"]) if len(s) else pd.Series(dtype=float)
    if kind == "P2":
        s = g[q > 0.2]
    elif kind == "B1":
        return pd.Series(1.0 / len(g), index=g["entity"])
    else:
        s = g
    m = s.set_index("entity")["mcap"].astype(float)
    m = m[np.isfinite(m) & (m > 0)]
    return m / m.sum() if len(m) else pd.Series(dtype=float)


def simulate(market: str, sc: pd.DataFrame, w: dict, kind: str, aum: float, scale: float, with_cost: bool):
    """Daily NAV and per-period records of one portfolio."""
    cal = w["close"].index
    dates = sorted(sc["date"].unique())
    execs = [cal[min(cal.searchsorted(pd.Timestamp(d), "right"), len(cal) - 1)] for d in dates]
    by_date = {d: g for d, g in sc.groupby("date")}
    hold = pd.Series(dtype=float)  # currency value per entity
    nav_path, periods = [], []
    nav = float(aum)
    for i, (d, ex) in enumerate(zip(dates, execs, strict=True)):
        g = by_date[d]
        price_ok = w["close"].loc[ex].reindex(g["entity"]).notna().to_numpy()
        tw = targets(g[price_ok], kind)
        tw = tw / tw.sum() if tw.sum() > 0 else tw
        nav = float(hold.sum()) if len(hold) else nav
        tgt = tw * nav
        names = tgt.index.union(hold.index)
        trade = (tgt.reindex(names).fillna(0.0) - hold.reindex(names).fillna(0.0)).abs()
        cost = 0.0
        if with_cost and trade.sum() > 0:
            frac = cm.side_cost(
                trade.to_numpy(),
                market,
                w["hs"].loc[ex].reindex(names).to_numpy(dtype=float),
                w["sig"].loc[ex].reindex(names).to_numpy(dtype=float),
                w["adv"].loc[ex].reindex(names).to_numpy(dtype=float),
                scale,
            )
            cost = float((trade.to_numpy() * frac).sum())
        nav_after = nav - cost
        hold = tw * nav_after
        end = execs[i + 1] if i + 1 < len(execs) else cal[min(cal.searchsorted(ex) + 21, len(cal) - 1)]
        span = cal[(cal > ex) & (cal <= end)]
        if len(hold) and len(span):
            r = w["ret"].loc[span, hold.index].fillna(0.0)
            path = (1.0 + r).cumprod() * hold.to_numpy()
            nav_series = path.sum(axis=1)
            hold = path.iloc[-1]
        else:
            nav_series = pd.Series(nav_after, index=span)
        nav_path.append(pd.Series([nav_after], index=[ex]))
        nav_path.append(nav_series)
        periods.append(
            {
                "date": d,
                "exec": ex,
                "part": g["part"].iloc[0],
                "n": int(len(tw)),
                "turnover": float(trade.sum() / (2 * nav)) if nav > 0 else 0.0,
                "cost": cost / nav if nav > 0 else 0.0,
                "ret": float(nav_series.iloc[-1] / nav - 1.0) if len(nav_series) and nav > 0 else 0.0,
            }
        )
    navs = pd.concat(nav_path)
    navs = navs[~navs.index.duplicated(keep="last")].sort_index()
    return navs, pd.DataFrame(periods)


# ------------------------------------------------------------------------------------------------ metrics


def mdd(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min()) if len(nav) else float("nan")


def stats(nav: pd.Series, bench: pd.Series, per: pd.DataFrame, bper: pd.DataFrame) -> dict:
    r = nav.pct_change().dropna()
    yrs = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    ex = per["ret"].to_numpy() - bper["ret"].to_numpy()
    n = len(ex)
    te = float(np.std(ex, ddof=1) * np.sqrt(12)) if n > 1 else float("nan")
    out = {
        "cagr": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1),
        "vol": float(r.std() * np.sqrt(252)),
        "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"),
        "mdd": mdd(nav),
        "bench_cagr": float((bench.iloc[-1] / bench.iloc[0]) ** (1 / yrs) - 1),
        "bench_mdd": mdd(bench),
        "excess_ann": float(np.mean(ex) * 12) if n else float("nan"),
        "te": te,
        "ir": float(np.mean(ex) * 12 / te) if te and te > 0 else float("nan"),
        "excess_t": float(np.mean(ex) / (np.std(ex, ddof=1) / np.sqrt(n))) if n > 2 else float("nan"),
        "turnover_yr": float(per["turnover"].mean() * 12),
        "cost_yr": float(per["cost"].mean() * 12),
        "months": n,
    }
    return out


def window(nav: pd.Series, bench: pd.Series, a: str, b: str) -> dict | None:
    sl = nav[(nav.index >= a) & (nav.index <= b)]
    bl = bench[(bench.index >= a) & (bench.index <= b)]
    if len(sl) < 3:
        return None
    base = nav[nav.index < a]
    bbase = bench[bench.index < a]
    p0 = base.iloc[-1] if len(base) else sl.iloc[0]
    b0 = bbase.iloc[-1] if len(bbase) else bl.iloc[0]
    return {
        "ret": float(sl.iloc[-1] / p0 - 1),
        "mdd": mdd(pd.concat([pd.Series([p0]), sl.reset_index(drop=True)])),
        "bench_ret": float(bl.iloc[-1] / b0 - 1),
        "bench_mdd": mdd(pd.concat([pd.Series([b0]), bl.reset_index(drop=True)])),
    }


# ------------------------------------------------------------------------------------------------ report


def pct(x: float | None, d: int = 1) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def run(market: str) -> dict:
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    say(f"{market}: out-of-sample scores of {SERVED[market]}")
    sc, uni, ds = oos_scores(market, settings, cache)
    OUT.mkdir(parents=True, exist_ok=True)
    sc.to_parquet(OUT / f"portfolio_scores_{market}.parquet", index=False)
    cal = pd.DatetimeIndex(ds.calendar)
    cal = cal[(cal >= pd.Timestamp(sc["date"].min())) & (cal <= cal[-1])]
    say(f"{market}: price matrix ({sc['entity'].nunique()} entities, {len(cal)} sessions)")
    w = wide(uni.prices, sorted(sc["entity"].unique()), cal)
    small, mid, large = cm.AUM[market]
    scen = {
        "gross": (mid, 1.0, False),
        f"cost@{small:,.0f}": (small, 1.0, True),
        f"cost@{mid:,.0f}": (mid, 1.0, True),
        f"cost@{large:,.0f}": (large, 1.0, True),
        f"cost2x@{mid:,.0f}": (mid, 2.0, True),
    }
    bench = {k: simulate(market, sc, w, k, mid, 1.0, False) for k in ("B1", "B2")}
    res: dict = {"market": market, "model": SERVED[market], "scenarios": {}, "stress": {}, "parts": {}}
    keep_periods = []
    for p in PORTFOLIOS:
        bn, bp = bench[BENCH[p]]
        for sname, (aum, scale, wc) in scen.items():
            nav, per = simulate(market, sc, w, p, aum, scale, wc)
            res["scenarios"][f"{p}|{sname}"] = stats(nav, bn, per, bp)
            if sname == f"cost@{mid:,.0f}":
                for part in ("outer", "holdout"):
                    m = per["part"] == part
                    t0 = per.loc[m, "exec"].min()
                    t1 = per.loc[m, "exec"].max() + pd.Timedelta(35, "D")
                    sl = (nav.index >= t0) & (nav.index <= t1)
                    bsl = (bn.index >= t0) & (bn.index <= t1)
                    res["parts"][f"{p}|{part}"] = stats(nav[sl], bn[bsl], per[m], bp[m])
                res["stress"][p] = {k: window(nav, bn, a, b) for k, (a, b) in STRESS.items()}
                keep_periods.append(per.assign(portfolio=p, bench_ret=bp["ret"].to_numpy()))
                ex = per["ret"].to_numpy() - bp["ret"].to_numpy()
                res.setdefault("expect", {})[p] = {
                    "monthly_excess_mean": float(np.mean(ex)),
                    "monthly_excess_sd": float(np.std(ex, ddof=1)),
                    "aum": mid,
                    "bench": BENCH[p],
                }
        say(f"{market} {p} done")
    pd.concat(keep_periods).to_parquet(OUT / f"portfolio_periods_{market}.parquet", index=False)
    s1, s2 = res["parts"].get("P1|outer", {}), res["parts"].get("P2|outer", {})
    h1, h2 = res["parts"].get("P1|holdout", {}), res["parts"].get("P2|holdout", {})
    ok = any((o.get("excess_t") or 0) >= 2 and (h.get("excess_ann") or -1) >= 0 for o, h in ((s1, h1), (s2, h2)))
    res["verdict"] = {"makes_money": bool(ok)}
    (OUT / f"portfolio_{market}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    (OUT / f"portfolio_expect_{market}.json").write_text(json.dumps(res["expect"], indent=1))
    md = write_md(res)
    (OUT / f"portfolio_{market}.md").write_text(md, encoding="utf-8")
    return res


def _row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def write_md(res: dict) -> str:
    m = res["market"]
    head = [
        "포트폴리오",
        "비용 시나리오",
        "연 수익",
        "변동성",
        "샤프",
        "MDD",
        "기준 연 수익",
        "연 초과",
        "추적오차",
        "IR",
    ]
    head += ["월 초과 t", "연 회전율", "연 비용"]
    lines = [
        f"# {m} 신호 포트폴리오 백테스트 — {res['model']} 표본 외 점수 (docs/MODELING.md 47·48)",
        "",
        "P1 = 점수 상위 20% 동일가중, P2 = 점수 하위 20%를 뺀 시가총액가중. 기준: P1 ↔ 유니버스 동일가중(B1),",
        "P2 ↔ 유니버스 시가총액가중(B2). 월말 점수 → 다음 거래일 종가 매매.",
        "",
        "## 전체 표본 외 기간",
        "",
        _row(*head),
        _row(*["---"] * len(head)),
    ]
    for k, s in res["scenarios"].items():
        p, sc = k.split("|")
        lines.append(
            _row(
                p,
                sc,
                pct(s["cagr"]),
                pct(s["vol"]),
                f"{s['sharpe']:.2f}",
                pct(s["mdd"]),
                pct(s["bench_cagr"]),
                pct(s["excess_ann"]),
                pct(s["te"]),
                f"{s['ir']:.2f}",
                f"{s['excess_t']:+.2f}",
                f"{s['turnover_yr']:.1f}",
                pct(s["cost_yr"], 2),
            )  # fmt: skip
        )
    lines += ["", "## 외부 vs 홀드아웃 (중간 규모, 기본 비용)", ""]
    lines += [_row("포트폴리오·구간", "연 수익", "MDD", "연 초과", "IR", "월 초과 t"), _row(*["---"] * 6)]
    for k, s in res["parts"].items():
        lines.append(
            _row(k, pct(s["cagr"]), pct(s["mdd"]), pct(s["excess_ann"]), f"{s['ir']:.2f}", f"{s['excess_t']:+.2f}")
        )
    lines += ["", "## 위기 구간 (중간 규모, 기본 비용)", ""]
    lines += [_row("구간", "P1 수익 / MDD", "B1 수익 / MDD", "P2 수익 / MDD", "B2 수익 / MDD"), _row(*["---"] * 5)]
    for k in STRESS:
        a, b = res["stress"]["P1"].get(k), res["stress"]["P2"].get(k)
        if a is None or b is None:
            lines.append(_row(k, "표본 밖", "", "", ""))
            continue
        lines.append(
            _row(
                k,
                f"{pct(a['ret'])} / {pct(a['mdd'])}",
                f"{pct(a['bench_ret'])} / {pct(a['bench_mdd'])}",
                f"{pct(b['ret'])} / {pct(b['mdd'])}",
                f"{pct(b['bench_ret'])} / {pct(b['bench_mdd'])}",
            )  # fmt: skip
        )
    ok = res["verdict"]["makes_money"]
    verdict = "돈이 됨 — 외부 월 초과 t ≥ 2 이고 홀드아웃 초과 ≥ 0" if ok else "돈이 된다고 말할 근거 없음"
    lines += [
        "",
        f"**판정(48): {verdict}**",
        "",
        "- 2008년은 종목 선택 자료가 시작되기 전이라 볼 수 없습니다(위험 모형의 위기 점검은 49절).",
    ]
    if m == "KR":
        lines.append(
            "- 한국 가격은 현금배당 미포함이라 P·B 모두 수익이 조금 낮게 잡힙니다"
            "(같은 방향이라 초과수익 비교는 대체로 유지)."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    a = ap.parse_args()
    run(a.market)
    print((OUT / f"portfolio_{a.market}.md").read_text())


if __name__ == "__main__":
    main()
