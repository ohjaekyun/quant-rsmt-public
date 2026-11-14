"""O'Shaughnessy-style value strategies on the PIT universes (docs/STRATEGY_VALUE.md - pre-declared 2026-10-03).

Strategies (rules fixed in docs/STRATEGY_VALUE.md sections 2-6, never tuned):
- VC (value composite): top N by the mean percentile of earnings / book / sales / operating-income yield and net
  buyback (negative earnings or book -> 0, missing -> 50, >= 3 metrics required).
- TV (trending value): inside the top 30% by VC, the N names with the highest 6-month return.
Universes: KR ranks 1-200, KR ranks 201-600 (20-day traded value >= 500M KRW), US S&P 500. N = 10% of the universe.
Quarterly rebalancing (Mar / Jun / Sep / Dec month-ends), equal weight, buy-and-hold inside the quarter, costs =
traded weight x half the round-trip cost per side. Monthly returns = ``backtest.factors.compute_factor_panel``'s
``fwd_ret`` (next-session close -> next month-end close, delistings closed at the terminal price).

Run (long: loads the PIT universes):  .venv/bin/python scripts/oshaughnessy_backtest.py [--market KR|US]
Writes reports/strategy/oshaughnessy_<M>.json and reports/strategy/oshaughnessy.md (Korean).

``--study lowvol`` (docs/STRATEGY_VALUE.md section 8, pre-declared 2026-10-03): VLV = inside the top 30% by VC, the
N names with the LOWEST 60-session volatility; LV = the N lowest-volatility names; VC for comparison. Universes add
KR ranks 601-1000 (the untouched confirmation sample; the verdict is VLV's excess t >= 2 there only). Writes
reports/strategy/value_lowvol_<M>.json, value_lowvol.md and the forward picks in reports/strategy/forward/.

Lives outside ``src/quantcio`` on purpose: the stock-selection model's holdout freeze fingerprints that tree.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from quantcio.backtest.factors import compute_factor_panel, price_signals, us_market_caps
from quantcio.config import PROJECT_ROOT, load_settings
from quantcio.data.cache import DiskCache
from quantcio.data.pit import load_pit_fundamentals, load_pit_universe
from quantcio.models import Market

OUT = PROJECT_ROOT / "reports" / "strategy"
REBAL_MONTHS = (3, 6, 9, 12)
TV_POOL = 0.30
MIN_METRICS = 3
KR_SMALL_MIN_ADV = 5e8  # 20-day average traded value, KRW
BOOT = 5000
BLOCK = 3
SEED = 20261003
UNIVERSES = {
    "KR": [
        {"key": "KR_large", "label": "KR 대형(1~200위)", "lo": 1, "hi": 200, "n_hold": 20, "cost": 0.0030},
        {"key": "KR_small", "label": "KR 중소형(201~600위)", "lo": 201, "hi": 600, "n_hold": 40, "cost": 0.0060},
    ],
    "US": [{"key": "US_large", "label": "US 대형(S&P 500)", "lo": 1, "hi": 10_000, "n_hold": 50, "cost": 0.0010}],
}
UNIVERSES_LOWVOL = {
    "KR": [
        *UNIVERSES["KR"],
        {
            "key": "KR_micro",
            "label": "KR 소형(601~1000위, 확인 표본)",
            "lo": 601,
            "hi": 1000,
            "n_hold": 40,
            "cost": 0.0080,
        },
    ],
    "US": UNIVERSES["US"],
}
START = {"KR": date(2016, 1, 1), "US": date(2012, 1, 1)}
N_LOAD = {"KR": 600, "US": 500}
N_LOAD_LOWVOL = {"KR": 1000, "US": 500}
VOL_WINDOW, VOL_MIN = 60, 40
LIQUID_KEYS = ("KR_small", "KR_micro")
INDEX = {"KR": "KOSPI", "US": "SPY"}
METRICS = ["ey", "by", "sy", "oy", "buyback"]


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ data


def asof_item(rows: pd.DataFrame, fpanel: pd.DataFrame, item: str, stale_days: int) -> pd.Series:
    """Latest PIT value of ``item`` per (date, entity) row (available_from <= date, period end within stale days)."""
    out = pd.Series(np.nan, index=rows.index, dtype=float)
    if fpanel is None or fpanel.empty or item not in fpanel:
        return out
    p = fpanel.dropna(subset=[item, "available_from"])[["entity", "available_from", "period_end", item]]
    p = p.sort_values(["available_from", "period_end"], kind="stable").rename(
        columns={"available_from": "_af", "period_end": "_pe"}
    )
    left = rows[["date", "entity"]].reset_index().sort_values("date", kind="stable")
    j = pd.merge_asof(left, p, left_on="date", right_on="_af", by="entity", direction="backward").set_index("index")
    j = j.reindex(rows.index)
    ok = j["_pe"].notna() & ((j["date"] - j["_pe"]).dt.days <= stale_days)
    return j[item].where(ok).astype(float)


def last_close(px: pd.DataFrame | None, when: pd.Timestamp, max_age_days: int = 7) -> float:
    if px is None or px.empty:
        return float("nan")
    s = px["close"].loc[:when]
    if s.empty or (when - s.index[-1]).days > max_age_days:
        return float("nan")
    return float(s.iloc[-1])


def adv20(px: pd.DataFrame | None, when: pd.Timestamp) -> float:
    if px is None or px.empty or "volume" not in px:
        return float("nan")
    g = px.loc[:when].tail(20)
    if len(g) < 15:
        return float("nan")
    return float((g["close"].astype(float) * g["volume"].astype(float)).mean())


def vol60(px: pd.DataFrame | None, dates: pd.DatetimeIndex) -> np.ndarray:
    """60-session annualised volatility of daily log returns (>= 40 obs) at each date (last bar <= date)."""
    out = np.full(len(dates), np.nan)
    if px is None or px.empty:
        return out
    c = px["close"].astype(float)
    lr = np.log(c.where(c > 0)).diff()
    v = lr.rolling(VOL_WINDOW, min_periods=VOL_MIN).std() * math.sqrt(252)
    idx = v.index.to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(idx, dates.to_numpy(dtype="datetime64[ns]"), "right") - 1
    ok = pos >= 0
    vals = v.to_numpy(dtype=float)
    out[ok] = vals[pos[ok]]
    age = np.full(len(dates), np.inf)
    age[ok] = (dates.to_numpy(dtype="datetime64[ns]")[ok] - idx[pos[ok]]) / np.timedelta64(1, "D")
    out[age > 7] = np.nan
    return out


def build(
    market: Market, settings, cache, n_load: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """Monthly panel of every member of the loaded universe, the monthly return of EVERY entity with prices
    (date x entity; a held name keeps its return after it leaves the universe ranking) and the index's monthly
    returns on the same windows."""
    mk = market.value
    n_load = n_load or N_LOAD[mk]
    say(f"{mk}: PIT universe (top {n_load})")
    uni = load_pit_universe(market, settings, cache, n_load, START[mk])
    ents = sorted(uni.membership["entity"].astype(str).unique())
    say(f"{mk}: PIT fundamentals ({len(ents)} entities)")
    fund = load_pit_fundamentals(market, settings, cache, ents)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(uni.membership["date"].unique())))
    say(f"{mk}: factor panel ({len(dates)} month-ends)")
    panel = compute_factor_panel(uni, fund, settings, dates, cache=cache)
    rank = uni.membership[["date", "entity", "rank"]].copy()
    rank["date"] = pd.to_datetime(rank["date"]).dt.normalize()
    panel = panel.merge(rank, on=["date", "entity"], how="left")
    stale = int(settings.pit.fundamentals.max_staleness_days)
    fp = fund.panel
    mcap = panel["market_cap"].astype(float).where(panel["market_cap"] > 0)
    rev = asof_item(panel, fp, "revenue_ttm", stale)
    op = asof_item(panel, fp, "op_income_ttm", stale)
    panel["ey"] = panel["ep"]  # NaN when <= 0 (flag ep_negative)
    panel["by"] = panel["bp"]
    panel["sy"] = (rev / mcap).where(rev > 0)
    panel["oy"] = op / mcap
    panel["oy_negative"] = panel["oy"] <= 0
    # net buyback: change of the split-neutral share count proxy mcap / adjusted close over 12 months
    say(f"{mk}: share-count proxy (market cap / adjusted close)")
    if market is Market.KR:
        from quantcio.data.pit.kr import pit_market_cap_kr

        wide = pit_market_cap_kr(settings, cache, ents, list(dates))
    else:
        wide = us_market_caps(uni, fund, settings)
    wide = wide.reindex(index=dates)
    closes = pd.DataFrame({e: [last_close(uni.prices.get(e), d) for d in dates] for e in wide.columns}, index=dates)
    shares = (wide / closes).where((wide > 0) & (closes > 0))
    chg = shares / shares.shift(12) - 1.0
    chg_long = chg.stack(future_stack=True).rename("share_chg").reset_index()
    chg_long.columns = ["date", "entity", "share_chg"]
    panel = panel.merge(chg_long, on=["date", "entity"], how="left")
    panel["buyback"] = -panel["share_chg"]
    say(f"{mk}: 60-session volatility")
    panel["vol60"] = np.nan
    for e, g in panel.groupby("entity", sort=False):
        panel.loc[g.index, "vol60"] = vol60(uni.prices.get(e), pd.DatetimeIndex(g["date"]))
    if market is Market.KR:
        say(f"{mk}: 20-day traded value")
        panel["adv20"] = [adv20(uni.prices.get(e), d) for d, e in zip(panel["date"], panel["entity"], strict=True)]
    # index returns on the same windows as fwd_ret: next session close after d -> last close <= next month-end
    b = uni.benchmarks.get(INDEX[mk])
    idx_ret = pd.Series(np.nan, index=dates)
    if b is not None and len(b):
        c = b["close"].astype(float)
        for d0, d1 in zip(dates[:-1], dates[1:], strict=True):
            after = c.loc[c.index > d0]
            upto = c.loc[:d1]
            if len(after) and len(upto) and upto.index[-1] > after.index[0]:
                idx_ret[d0] = upto.iloc[-1] / after.iloc[0] - 1.0
    # every entity's monthly return, membership or not (positions are held to the next rebalance)
    say(f"{mk}: monthly returns of every entity")
    allp = pd.DataFrame([(d, e) for d in dates for e in uni.prices], columns=["date", "entity"])
    nxt = dict(zip(dates[:-1], dates[1:], strict=False))
    sig = price_signals(uni, allp, settings, nxt)
    allp = pd.concat([allp, sig[["fwd_ret", "fwd_status"]]], axis=1)
    rets = allp.pivot_table(index="date", columns="entity", values="fwd_ret", aggfunc="first")
    status = allp.pivot_table(index="date", columns="entity", values="fwd_status", aggfunc="first")
    rets.attrs["status"] = status
    info = {
        "entities": len(ents),
        "dates": [str(dates[0].date()), str(dates[-1].date())],
        "share_chg_abs_gt_50pct": int((panel["share_chg"].abs() > 0.5).sum()),
        "share_chg_rows": int(panel["share_chg"].notna().sum()),
    }
    return panel, rets, idx_ret, info


# ------------------------------------------------------------------------------------------------ strategy


def pct(s: pd.Series) -> pd.Series:
    """Percentile 0..100 (higher = cheaper) among non-NaN values."""
    r = s.rank(method="average", pct=True)
    return 100.0 * r


NEGATIVE_FLAG = {"ey": "ep_negative", "by": "bp_negative", "oy": "oy_negative"}  # a loss / deficit scores 0


def value_scores(u: pd.DataFrame) -> pd.Series:
    """Value composite of one date's universe rows (docs/STRATEGY_VALUE.md section 3): mean of the metric
    percentiles (loss / deficit -> 0, missing -> 50); NaN when fewer than MIN_METRICS metrics are known."""
    cols, avail = {}, {}
    for m in METRICS:
        known = u[m].notna()
        p = pct(u[m].where(~u[NEGATIVE_FLAG[m]].fillna(False).astype(bool)) if m in NEGATIVE_FLAG else u[m])
        if m in NEGATIVE_FLAG:
            neg = u[NEGATIVE_FLAG[m]].fillna(False).astype(bool)
            p = p.where(~neg, 0.0)
            known = known | neg
        cols[m], avail[m] = p, known
    score = pd.DataFrame(cols).fillna(50.0).mean(axis=1)
    return score.where(pd.DataFrame(avail).sum(axis=1) >= MIN_METRICS)


def universe_rows(panel: pd.DataFrame, d: pd.Timestamp, spec: dict, market: str) -> pd.DataFrame:
    u = panel[(panel["date"] == d) & (panel["rank"] >= spec["lo"]) & (panel["rank"] <= spec["hi"])]
    # investable = a fresh daily close at t (known at t; the factor backtest drops the same no-daily-price rows,
    # e.g. US names that only have SEC FTD month-end marks after an acquisition)
    u = u[u["price_status"] == "ok"]
    if spec["key"] in LIQUID_KEYS:
        u = u[u["adv20"] >= KR_SMALL_MIN_ADV]
    return u


def select(u: pd.DataFrame, strategy: str, n: int) -> list[str]:
    if strategy == "LV":  # the N lowest-volatility names (docs/STRATEGY_VALUE.md 8.1)
        v = u["vol60"].dropna().sort_values(ascending=True)
        return u.loc[v.index[:n], "entity"].tolist()
    sc = value_scores(u).dropna()
    if sc.empty:
        return []
    if strategy == "VC":
        return u.loc[sc.sort_values(ascending=False).index[:n], "entity"].tolist()
    pool = sc.sort_values(ascending=False).index[: max(n, int(math.ceil(TV_POOL * len(u))))]
    if strategy == "VLV":  # value top 30%, then the lowest volatility
        v = u.loc[pool, "vol60"].dropna().sort_values(ascending=True)
        return u.loc[v.index[:n], "entity"].tolist()
    mom = u.loc[pool, "mom_6m"].dropna().sort_values(ascending=False)
    return u.loc[mom.index[:n], "entity"].tolist()


def simulate(panel: pd.DataFrame, rets: pd.DataFrame, spec: dict, market: str, strategy: str) -> pd.DataFrame:
    """Monthly returns (gross, cost, net) of one strategy; buy-and-hold inside each quarter. ``rets`` = every
    entity's monthly return (a holding that leaves the universe ranking keeps earning its own return)."""
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    status = rets.attrs.get("status")
    rebal = [d for d in dates if d.month in REBAL_MONTHS]
    # start at the first rebalance from which EVERY rebalance fills the book (data coverage, not results: early
    # month-ends lack trailing-4-quarter fundamentals and the 12-month share change; docs/STRATEGY_VALUE.md 7)
    chosen = {d: select(universe_rows(panel, d, spec, market), strategy, spec["n_hold"]) for d in rebal}
    full = [len(chosen[d]) >= spec["n_hold"] for d in rebal]
    k = next((i for i in range(len(rebal)) if all(full[i:])), len(rebal) - 1)
    skipped = [str(d.date()) for d in rebal[:k]]
    rebal = rebal[k:]
    first = rebal[0]
    rows = []
    w: dict[str, float] = {}
    n_missing = 0
    missing_status: dict[str, int] = {}
    picks: list[int] = []
    for d in dates[dates >= first]:
        if d not in rets.index or rets.loc[d].notna().sum() == 0:
            break  # the last month-end has no forward return
        cost = 0.0
        hold_n = len(w)
        if d in rebal:
            names = chosen[d]
            picks.append(len(names))
            new = {e: 1.0 / len(names) for e in names} if names else {}
            traded = sum(abs(new.get(e, 0.0) - w.get(e, 0.0)) for e in set(new) | set(w))
            cost = traded * spec["cost"] / 2.0
            w = new
            hold_n = len(w)
        r = rets.loc[d]
        gross = 0.0
        new_w = {}
        for e, wi in w.items():
            ri = r.get(e, np.nan)
            if not np.isfinite(ri):
                n_missing += 1
                st = str(status.at[d, e]) if status is not None and e in status.columns and d in status.index else "?"
                missing_status[st] = missing_status.get(st, 0) + 1
                ri = 0.0
            gross += wi * ri
            new_w[e] = wi * (1.0 + ri)
        tot = sum(new_w.values())
        w = {e: v / tot for e, v in new_w.items()} if tot > 0 else {}
        rows.append({"date": d, "gross": gross, "cost": cost, "net": (1 + gross) * (1 - cost) - 1, "n": hold_n})
    out = pd.DataFrame(rows).set_index("date")
    out.attrs["missing_returns"] = n_missing
    out.attrs["missing_status"] = missing_status
    out.attrs["picks"] = picks
    out.attrs["skipped_rebalances"] = skipped
    return out


def universe_ew(panel: pd.DataFrame, spec: dict, market: str) -> pd.Series:
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    out = {}
    for d in dates:
        u = universe_rows(panel, d, spec, market)
        r = u["fwd_ret"].dropna()
        if len(r):
            out[d] = float(r.mean())
    return pd.Series(out)


# ------------------------------------------------------------------------------------------------ statistics


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    v = (1 + r).cumprod()
    n = len(r)
    cagr = float(v.iloc[-1] ** (12.0 / n) - 1.0) if n else float("nan")
    vol = float(r.std(ddof=1) * math.sqrt(12))
    roll12 = (1 + r).rolling(12).apply(np.prod, raw=True) - 1
    return {
        "months": n,
        "cagr": cagr,
        "vol": vol,
        "sharpe": float(r.mean() * 12 / vol) if vol > 0 else float("nan"),
        "mdd": float((v / v.cummax() - 1).min()),
        "worst_12m": float(roll12.min()) if n >= 12 else float("nan"),
        "best_12m": float(roll12.max()) if n >= 12 else float("nan"),
        "total": float(v.iloc[-1] - 1.0),
    }


def excess_stats(x: pd.Series) -> dict:
    x = x.dropna().to_numpy(dtype=float)
    n = len(x)
    m = float(x.mean())
    sd = float(x.std(ddof=1))
    t = m / (sd / math.sqrt(n)) if sd > 0 else float("nan")
    rng = np.random.default_rng(SEED)
    nb = int(math.ceil(n / BLOCK))
    means = np.empty(BOOT)
    for b in range(BOOT):
        starts = rng.integers(0, max(n - BLOCK + 1, 1), nb)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        means[b] = x[idx].mean()
    return {
        "mean_monthly": m,
        "annualised": float((1 + m) ** 12 - 1),
        "t": t,
        "p_one_sided": float(1 - stats.t.cdf(t, n - 1)) if np.isfinite(t) else float("nan"),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "hit": float((x > 0).mean()),
        "months": n,
    }


def bh(pvals: list[float], q: float = 0.10) -> list[bool]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, dtype=bool)
    out[order[:k]] = True
    return out.tolist()


# ------------------------------------------------------------------------------------------------ main


def run_market(mk: str, settings, cache, study: str = "oshaughnessy") -> dict:
    market = Market(mk)
    lowvol = study == "lowvol"
    strategies = ("VLV", "LV", "VC") if lowvol else ("VC", "TV")
    specs = UNIVERSES_LOWVOL[mk] if lowvol else UNIVERSES[mk]
    panel, rets, idx_ret, info = build(market, settings, cache, N_LOAD_LOWVOL[mk] if lowvol else None)
    res = {"market": mk, "study": study, "info": info, "universes": {}, "forward": {}}
    for spec in specs:
        say(f"{spec['key']}: strategies")
        ew = universe_ew(panel, spec, mk)
        u_out = {"label": spec["label"], "n_hold": spec["n_hold"], "cost_round_trip": spec["cost"], "strategies": {}}
        sims = {strat: simulate(panel, rets, spec, mk, strat) for strat in strategies}
        common = sims[strategies[0]].index
        for sim in sims.values():
            common = common.intersection(sim.index)
        series = {}
        for strat, sim in sims.items():
            sim = sim.loc[common]
            net = sim["net"]
            ewa = ew.reindex(net.index)
            u_out["strategies"][strat] = {
                "perf": perf(net),
                "perf_gross": perf(sim["gross"]),
                "excess_vs_universe": excess_stats(net - ewa),
                "turnover_cost_per_year": float(sim["cost"].sum() / (len(sim) / 12)),
                "avg_names": float(sim["n"].mean()),
                "missing_returns": int(sims[strat].attrs.get("missing_returns", 0)),
                "missing_status": sims[strat].attrs.get("missing_status", {}),
                "picks_min": int(min(sims[strat].attrs.get("picks") or [0])),
                "picks_first": (sims[strat].attrs.get("picks") or [None])[:4],
                "skipped_rebalances": sims[strat].attrs.get("skipped_rebalances", []),
                "by_year": {
                    str(y): {
                        "strategy": float((1 + g).prod() - 1),
                        "universe": float((1 + ewa.loc[g.index]).prod() - 1),
                    }
                    for y, g in net.groupby(net.index.year)
                },
            }
            series[strat] = net
        u_out["universe_ew"] = perf(ew.reindex(common))
        u_out["index"] = perf(idx_ret.reindex(common)) if idx_ret.reindex(common).notna().sum() > 12 else None
        if lowvol:
            u_out["vlv_minus_vc"] = excess_stats(series["VLV"] - series["VC"])
            u_out["vlv_minus_lv"] = excess_stats(series["VLV"] - series["LV"])
            last = max(d for d in panel["date"].unique() if len(universe_rows(panel, d, spec, mk)))
            u = universe_rows(panel, last, spec, mk)
            picks = select(u, "VLV", spec["n_hold"])
            sub = u.set_index("entity").loc[picks]
            res["forward"][spec["key"]] = {
                "rule": "VLV (docs/STRATEGY_VALUE.md 8.1)",
                "data_as_of": str(pd.Timestamp(last).date()),
                "names": [
                    {
                        "entity": e,
                        "ticker": str(r.get("ticker")),
                        "name": str(r.get("name")),
                        "vol60": float(r["vol60"]),
                    }
                    for e, r in sub.iterrows()
                ],
            }
        else:
            u_out["tv_minus_vc"] = excess_stats(series["TV"] - series["VC"])
        u_out["period"] = [str(common[0].date()), str(common[-1].date())]
        res["universes"][spec["key"]] = u_out
    return res


def fmt_pct(x: float | None, nd: int = 1) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:+.{nd}f}%"


def plain_pct(x: float | None, nd: int = 1) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:.{nd}f}%"


STRAT_KO = {"VC": "가치 종합", "TV": "추세 가치"}


def _row(u: dict, s: str, v: dict, ok: bool) -> str:
    p, e = v["perf"], v["excess_vs_universe"]
    lo, hi = (1 + e["lo"]) ** 12 - 1, (1 + e["hi"]) ** 12 - 1
    period = f"{u['period'][0][:7]}~{u['period'][1][:7]}"
    mark = ("의미 있음" if ok else "아님") + (" · 연 20% 이상" if p["cagr"] >= 0.20 else "")
    cells = [
        u["label"],
        STRAT_KO[s],
        period,
        plain_pct(p["cagr"]),
        f"{p['sharpe']:.2f}",
        fmt_pct(p["mdd"], 0),
        fmt_pct(p["worst_12m"], 0),
        plain_pct(u["universe_ew"]["cagr"]),
        f"{fmt_pct(e['annualised'])} [{fmt_pct(lo)}, {fmt_pct(hi)}]",
        f"{e['t']:+.2f}",
        mark,
    ]
    return "| " + " | ".join(cells) + " |"


def render(results: dict[str, dict]) -> str:
    tests = []
    for mk, r in results.items():
        for uk, u in r["universes"].items():
            for s, v in u["strategies"].items():
                tests.append((mk, uk, s, v["excess_vs_universe"]["p_one_sided"], v["excess_vs_universe"]["t"]))
    passed = bh([t[3] for t in tests]) if tests else []
    verdict = {(t[0], t[1], t[2]): (p and t[4] >= 2) for t, p in zip(tests, passed, strict=True)}
    head = [
        "유니버스",
        "전략",
        "기간",
        "연평균 수익률",
        "위험 대비 수익(샤프)",
        "최대 낙폭",
        "최악 12개월",
        "유니버스 동일 비중 연수익",
        "초과수익(연) [95% 구간]",
        "t",
        "판정",
    ]
    lines = ["# 오쇼너시식 가치 전략 백테스트 결과", ""]
    lines.append(f"- 생성 {time.strftime('%Y-%m-%d %H:%M')} · 규칙: docs/STRATEGY_VALUE.md")
    lines += ["", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for mk, r in results.items():
        for uk, u in r["universes"].items():
            for s, v in u["strategies"].items():
                lines.append(_row(u, s, v, verdict[(mk, uk, s)]))
    lines += ["", "## 비교 기준", "", "| 유니버스 | 유니버스 동일 비중 | 시장 지수 |", "|---|---|---|"]
    for r in results.values():
        for u in r["universes"].values():
            ew, ix = u["universe_ew"], u["index"]
            ix_txt = f"연 {plain_pct(ix['cagr'])}, 최대 낙폭 {fmt_pct(ix['mdd'], 0)}" if ix else "–"
            ew_txt = f"연 {plain_pct(ew['cagr'])}, 최대 낙폭 {fmt_pct(ew['mdd'], 0)}"
            lines.append(f"| {u['label']} | {ew_txt} | {ix_txt} |")
    lines += ["", "## 추세 가치 − 가치 종합 (모멘텀을 더한 효과, 월 차이)", ""]
    for r in results.values():
        for u in r["universes"].values():
            e = u["tv_minus_vc"]
            lines.append(f"- {u['label']}: 연 {fmt_pct(e['annualised'])} (t {e['t']:+.2f})")
    lines += ["", "## 연도별 수익률 (전략 / 유니버스 동일 비중)", ""]
    for r in results.values():
        for u in r["universes"].values():
            for s, v in u["strategies"].items():
                yrs = " · ".join(
                    f"{y} {plain_pct(x['strategy'], 0)}/{plain_pct(x['universe'], 0)}" for y, x in v["by_year"].items()
                )
                lines.append(f"- {u['label']} {STRAT_KO[s]}: {yrs}")
    lines += ["", "## 데이터 메모", ""]
    for mk, r in results.items():
        lines.append(f"- {mk}: {json.dumps(r['info'], ensure_ascii=False)}")
        for u in r["universes"].values():
            for s, v in u["strategies"].items():
                cost = plain_pct(v["turnover_cost_per_year"], 2)
                lines.append(
                    f"  - {u['label']} {s}: 평균 보유 {v['avg_names']:.1f}종목, 연 거래비용 {cost}, "
                    f"수익률 없는 보유 월 {v['missing_returns']}건(0으로 처리: {v.get('missing_status')}), "
                    f"리밸런싱당 최소 선택 {v.get('picks_min')}종목"
                )
    return "\n".join(lines) + "\n"


LOWVOL_KO = {"VLV": "가치 + 저변동성", "LV": "저변동성만", "VC": "가치 종합"}


def render_lowvol(results: dict[str, dict]) -> str:
    head = [
        "유니버스",
        "전략",
        "기간",
        "연평균 수익률",
        "샤프",
        "최대 낙폭",
        "최악 12개월",
        "유니버스 동일 비중",
        "초과수익(연) [95% 구간]",
        "t",
        "판정",
    ]
    lines = [
        "# 가치 + 저변동성 백테스트 결과",
        "",
        f"- 생성 {time.strftime('%Y-%m-%d %H:%M')} · 규칙: docs/STRATEGY_VALUE.md 8절",
    ]
    lines += [
        "- 판정은 확인 표본(KR 601~1000위)의 '가치 + 저변동성'에만 합니다(t ≥ 2).",
        "- 나머지는 이미 본 데이터라 참고용입니다.",
        "",
    ]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results.values():
        for uk, u in r["universes"].items():
            for s, v in u["strategies"].items():
                p, e = v["perf"], v["excess_vs_universe"]
                lo, hi = (1 + e["lo"]) ** 12 - 1, (1 + e["hi"]) ** 12 - 1
                if uk == "KR_micro" and s == "VLV":
                    mark = "**확인**" if e["t"] >= 2 else "확인 안 됨"
                elif uk == "KR_micro":
                    mark = "비교용"
                else:
                    mark = "참고(사후)"
                cells = [
                    u["label"],
                    LOWVOL_KO[s],
                    f"{u['period'][0][:7]}~{u['period'][1][:7]}",
                    plain_pct(p["cagr"]),
                    f"{p['sharpe']:.2f}",
                    fmt_pct(p["mdd"], 0),
                    fmt_pct(p["worst_12m"], 0),
                    plain_pct(u["universe_ew"]["cagr"]),
                    f"{fmt_pct(e['annualised'])} [{fmt_pct(lo)}, {fmt_pct(hi)}]",
                    f"{e['t']:+.2f}",
                    mark,
                ]
                lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## 무엇이 기여했나 (월 수익률 차이)", ""]
    for r in results.values():
        for u in r["universes"].values():
            a, b = u["vlv_minus_vc"], u["vlv_minus_lv"]
            lines.append(
                f"- {u['label']}: 저변동성 조건의 효과(가치+저변동성 − 가치 종합) "
                f"연 {fmt_pct(a['annualised'])} (t {a['t']:+.2f}) · 가치 조건의 효과(가치+저변동성 − 저변동성만) "
                f"연 {fmt_pct(b['annualised'])} (t {b['t']:+.2f})"
            )
    lines += ["", "## 비교 기준", "", "| 유니버스 | 유니버스 동일 비중 | 시장 지수 |", "|---|---|---|"]
    for r in results.values():
        for u in r["universes"].values():
            ew, ix = u["universe_ew"], u["index"]
            ix_txt = f"연 {plain_pct(ix['cagr'])}, 최대 낙폭 {fmt_pct(ix['mdd'], 0)}" if ix else "–"
            lines.append(f"| {u['label']} | 연 {plain_pct(ew['cagr'])}, 최대 낙폭 {fmt_pct(ew['mdd'], 0)} | {ix_txt} |")
    lines += ["", "## 연도별 수익률 (전략 / 유니버스 동일 비중)", ""]
    for r in results.values():
        for u in r["universes"].values():
            v = u["strategies"]["VLV"]
            yrs = " · ".join(
                f"{y} {plain_pct(x['strategy'], 0)}/{plain_pct(x['universe'], 0)}" for y, x in v["by_year"].items()
            )
            lines.append(f"- {u['label']} 가치 + 저변동성: {yrs}")
    lines += ["", "## 데이터 메모", ""]
    for mk, r in results.items():
        lines.append(f"- {mk}: {json.dumps(r['info'], ensure_ascii=False)}")
        for u in r["universes"].values():
            for s, v in u["strategies"].items():
                cost = plain_pct(v["turnover_cost_per_year"], 2)
                skipped = len(v.get("skipped_rebalances") or [])
                lines.append(
                    f"  - {u['label']} {s}: 평균 보유 {v['avg_names']:.1f}종목, 연 거래비용 {cost}, "
                    f"수익률 없는 보유 월 {v['missing_returns']}건({v.get('missing_status')}), "
                    f"시작 전 건너뛴 리밸런싱 {skipped}회"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US", "ALL"], default="ALL")
    ap.add_argument("--study", choices=["oshaughnessy", "lowvol"], default="oshaughnessy")
    args = ap.parse_args()
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    OUT.mkdir(parents=True, exist_ok=True)
    mks = ["KR", "US"] if args.market == "ALL" else [args.market]
    prefix = "oshaughnessy" if args.study == "oshaughnessy" else "value_lowvol"
    for mk in mks:
        path = OUT / f"{prefix}_{mk}.json"
        res = run_market(mk, settings, cache, args.study)
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        say(f"{mk}: wrote {path}")
        if res.get("forward"):
            fdir = OUT / "forward"
            fdir.mkdir(parents=True, exist_ok=True)
            fpath = fdir / f"value_lowvol_{mk}_{time.strftime('%Y%m%d')}.json"
            if not fpath.exists():  # append-only: a day's record is never overwritten
                rec = {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **res["forward"]}
                fpath.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
                say(f"{mk}: forward picks -> {fpath}")
    every = {}
    for mk in ("KR", "US"):
        p = OUT / f"{prefix}_{mk}.json"
        if p.exists():
            every[mk] = json.loads(p.read_text(encoding="utf-8"))
    md = render(every) if args.study == "oshaughnessy" else render_lowvol(every)
    (OUT / f"{prefix}.md").write_text(md, encoding="utf-8")
    say(f"wrote {OUT / f'{prefix}.md'}")


if __name__ == "__main__":
    main()
