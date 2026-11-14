"""Trading-rule tests: take-profit / stop-loss grid (KR, US) and a KR retail-crowd avoidance filter
(docs/MODELING.md 68, pre-declared 2026-10-18). Research only.

    .venv/bin/python scripts/rules68_study.py --market KR      # then --market US (one heavy run at a time)

Served model's outer + holdout scores (portfolio_backtest.oos_scores), top 20% equal weight, monthly rebalance,
next-session execution, the 47 cost model, daily NAV. Writes reports/modeling/<M>_rules68.{json,md}.
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
import hybrid_study as hs  # noqa: E402
import keyed_data as kd  # noqa: E402
import portfolio_backtest as pb  # noqa: E402
import risk_overlap_study as ro  # noqa: E402
import v13_data as v13  # noqa: E402
import v16_data as v16  # noqa: E402
import v26_features as vf  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
TP = (None, 0.10, 0.20, 0.30)
SL = (None, 0.10, 0.15, 0.20)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def simulate(market: str, sc: pd.DataFrame, w: dict, aum: float, tp: float | None, sl: float | None):
    """hybrid_study.simulate_book (R0) with take-profit / stop-loss exits against the execution price."""
    cal = w["close"].index
    dates = sorted(sc["date"].unique())
    execs = [cal[min(cal.searchsorted(pd.Timestamp(d), "right"), len(cal) - 1)] for d in dates]
    by_date = {d: g for d, g in sc.groupby("date")}
    hold = pd.Series(dtype=float)
    cash = float(aum)
    path, periods, n_tp, n_sl = [], [], 0, 0

    def cost_of(trade: pd.Series, ex) -> float:
        if trade.sum() <= 0:
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
        tw = hs.book_weights(g[ok], "R0", {})
        nav = cash + float(hold.sum())
        tgt = tw * nav
        names = tgt.index.union(hold.index)
        trade = (tgt.reindex(names).fillna(0.0) - hold.reindex(names).fillna(0.0)).abs()
        c = cost_of(trade, ex)
        nav_after = nav - c
        hold = tw * nav_after
        cash = nav_after - float(hold.sum())
        entry = w["close"].loc[ex].reindex(hold.index)
        end = execs[i + 1] if i + 1 < len(execs) else cal[min(cal.searchsorted(ex) + 21, len(cal) - 1)]
        for day in cal[(cal > ex) & (cal <= end)]:
            if len(hold):
                hold = hold * (1.0 + w["ret"].loc[day, hold.index].fillna(0.0))
                if tp is not None or sl is not None:
                    r = w["close"].loc[day, hold.index] / entry.reindex(hold.index) - 1.0
                    up = r >= tp if tp is not None else r > np.inf
                    dn = r <= -sl if sl is not None else r < -np.inf
                    out = (up | dn) & (hold > 0)
                    if out.any():
                        val = hold[out[out].index]
                        cash += float(val.sum()) - cost_of(val.abs(), day)
                        hold = hold.drop(val.index)
                        n_tp, n_sl = n_tp + int((up & out).sum()), n_sl + int((dn & out & ~up).sum())
            path.append((day, cash + float(hold.sum())))
        periods.append({"date": d, "exec": ex, "part": g["part"].iloc[0], "turnover": float(trade.sum() / (2 * nav))})
    per = pd.DataFrame(periods)
    per.attrs.update(tp=n_tp, sl=n_sl)
    return pd.Series(dict(path)).sort_index(), per


def monthly(nav: pd.Series) -> pd.Series:
    return nav.resample("ME").last().pct_change().dropna()


def split(nav: pd.Series, bench: pd.Series, per: pd.DataFrame, base: pd.Series | None) -> dict:
    hold0 = per.loc[per["part"] == "holdout", "exec"].min()
    mo = monthly(nav)
    bm = monthly(bench).reindex(mo.index).fillna(0.0)
    out = {}
    for part, mask in (("outer", mo.index < hold0), ("holdout", mo.index >= hold0)):
        st = ro._book_stats(mo[mask], bm[mask])
        sub = nav[(nav.index < hold0) if part == "outer" else (nav.index >= hold0)]
        st["mdd_daily"] = pb.mdd(sub / sub.iloc[0]) if len(sub) else float("nan")
        dr = sub.pct_change().dropna()
        st["sharpe_daily"] = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else float("nan")
        if base is not None:
            st["vs_base"] = mt.summarize((mo - base.reindex(mo.index)).to_numpy()[mask])
        out[part] = st
    out["turnover_yr"] = float(per["turnover"].mean() * 12)
    out["exits"] = {"tp": per.attrs.get("tp", 0), "sl": per.attrs.get("sl", 0)}
    return out


# ================================================================================================ KR crowd filter


def crowd_flags(sc: pd.DataFrame, ds, settings, cache) -> pd.DataFrame:
    """F1 warnings (60 days), F2 retail net buying top decile, F3 1-week return top 5% - per (date, entity) of sc."""
    names = vf.name_map(settings, cache)
    t = vf.warnings_table(settings, names)
    out = sc[["date", "entity", "mcap"]].copy()
    out["entity"] = out["entity"].astype(str)
    f1 = np.zeros(len(out), dtype=bool)
    ret1w = ds.frame[["date", "entity", "ret_1w"]].assign(entity=lambda x: x["entity"].astype(str))
    out = out.merge(ret1w.drop_duplicates(["date", "entity"]), on=["date", "entity"], how="left")
    indiv = np.full(len(out), np.nan)
    for e, ii in out.groupby("entity").indices.items():
        d = out["date"].to_numpy(dtype="datetime64[ns]")[ii]
        g = t[t["entity"] == e] if len(t) else t
        if len(g):
            w = g.loc[g["kind"] == "warn", "ann"].to_numpy(dtype="datetime64[ns]")
            c = g.loc[g["kind"] == "caution", "ann"].to_numpy(dtype="datetime64[ns]")
            nw, _ = v16._windowed(w, np.ones(len(w)), d, 60)
            nc, _ = v16._windowed(c, np.ones(len(c)), d, 60)
            f1[ii] = (nw >= 1) | (nc >= 2)
        p = kd.root(settings, "krx_flows") / f"{e}.parquet"
        if p.exists():
            q = pd.read_parquet(p).sort_values("date").set_index("date")
            if "v_개인" in q:
                s20 = q["v_개인"].rolling(20, min_periods=15).sum().shift(1).to_frame()
                indiv[ii] = v13._asof_rows(s20, d, v13.FRESH_DAYS)[:, 0]
    out["F1"] = f1
    out["indiv"] = indiv / out["mcap"].to_numpy()
    out["F2"] = out.groupby("date")["indiv"].rank(pct=True) > 0.9
    out["F3"] = out.groupby("date")["ret_1w"].rank(pct=True) > 0.95
    out["F_all"] = out[["F1", "F2", "F3"]].any(axis=1)
    return out


# ================================================================================================ main


def fmt(x, spec):
    return "–" if x is None or not np.isfinite(x) else format(x, spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    m = ap.parse_args().market
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    sc, uni, ds = pb.oos_scores(m, settings, cache)
    cal = pd.DatetimeIndex(ds.calendar)
    cal = cal[cal >= pd.Timestamp(sc["date"].min())]
    w = pb.wide(uni.prices, sorted(sc["entity"].unique()), cal)
    mid = cm.AUM[m][1]
    bench, _ = pb.simulate(m, sc, w, "B1", mid, 1.0, False)
    say(f"{m}: {sc['date'].nunique()} rebalances, simulating {len(TP) * len(SL)} exit rules")
    base_nav, base_per = simulate(m, sc, w, mid, None, None)
    base_mo = monthly(base_nav)
    r = {"market": m, "grid": {}, "filters": {}}
    for tp in TP:
        for sl in SL:
            key = f"tp{int((tp or 0) * 100)}_sl{int((sl or 0) * 100)}"
            nav, per = (base_nav, base_per) if tp is None and sl is None else simulate(m, sc, w, mid, tp, sl)
            r["grid"][key] = split(nav, bench, per, None if key == "tp0_sl0" else base_mo)
            say(
                f"{key}: outer ann {r['grid'][key]['outer']['ann_ret']:+.3f}, "
                f"holdout {r['grid'][key]['holdout']['ann_ret']:+.3f}"
            )
    b = r["grid"]["tp0_sl0"]
    for key, x in r["grid"].items():
        if key == "tp0_sl0":
            continue
        o, h = x["outer"], x["holdout"]
        x["return_better"] = bool((o["vs_base"]["t"] or 0) >= 2 and (h["vs_base"]["mean"] or -1) >= 0)
        x["risk_tool"] = bool(
            o["mdd"] > b["outer"]["mdd"]
            and o["cvar5"] > b["outer"]["cvar5"]
            and o["ann_ret"] >= b["outer"]["ann_ret"] - 0.01
        )
    if m == "KR":
        fl = crowd_flags(sc, ds, settings, cache)
        r["filter_share"] = {f: float(fl[f].mean()) for f in ("F1", "F2", "F3", "F_all")}
        sc2 = sc.assign(entity=sc["entity"].astype(str)).merge(
            fl[["date", "entity", "F1", "F2", "F3", "F_all"]], on=["date", "entity"], how="left"
        )
        for f in ("F1", "F2", "F3", "F_all"):
            nav, per = simulate(m, sc2[~sc2[f].fillna(False)], w, mid, None, None)
            x = split(nav, bench, per, base_mo)
            o, h = x["outer"]["vs_base"], x["holdout"]["vs_base"]
            x["verdict"] = (
                "도움"
                if (o["mean"] or 0) > 0 and (o["t"] or 0) >= 2 and (h["mean"] or -1) >= 0
                else ("도움(약함)" if (o["mean"] or 0) > 0 and (h["mean"] or -1) >= 0 else "도움 없음")
            )
            r["filters"][f] = x
            say(
                f"filter {f}: outer vs base {o['mean']:+.4f}/mo (t {o['t']:+.2f}), "
                f"holdout {h['mean']:+.4f} -> {x['verdict']}"
            )
    lines = [
        f"# {m} 매매 규칙 시험 (docs/MODELING.md 68)",
        "",
        "## 68.1 익절·손절 격자 (상위 20% 동일가중, 비용 포함)",
        "",
    ]
    lines += [
        "| 규칙 | 외부 연 수익 | 외부 최대 낙폭(일별) | 외부 월 CVaR | 외부 기준 대비 월 차이 (t) "
        "| 홀드아웃 연 수익 | 홀드아웃 기준 대비 | 익절/손절 횟수 | 판정 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, x in r["grid"].items():
        o, h = x["outer"], x["holdout"]
        vo, vh = o.get("vs_base") or {}, h.get("vs_base") or {}
        verdict = (
            "기준"
            if key == "tp0_sl0"
            else ("수익 개선" if x.get("return_better") else ("위험 관리용" if x.get("risk_tool") else "—"))
        )
        lines.append(
            f"| {key} | {fmt(o['ann_ret'], '+.1%')} | {fmt(o['mdd_daily'], '+.1%')} | {fmt(o['cvar5'], '+.1%')} | "
            f"{fmt(vo.get('mean'), '+.2%')} ({fmt(vo.get('t'), '+.2f')}) | {fmt(h['ann_ret'], '+.1%')} | "
            f"{fmt(vh.get('mean'), '+.2%')} | "
            f"{x['exits']['tp']}/{x['exits']['sl']} | {verdict} |"
        )
    if r["filters"]:
        lines += [
            "",
            "## 68.2 한국 개인 쏠림 회피 필터 (et 상위 20%에서 제외)",
            "",
            f"- 걸린 비율(점수 행 기준): {r['filter_share']}",
            "",
        ]
        lines += [
            "| 필터 | 외부 연 수익 | 외부 기준 대비 월 차이 (t) | 홀드아웃 연 수익 | 홀드아웃 기준 대비 | 판정 |",
            "|---|---|---|---|---|---|",
        ]
        for f, x in r["filters"].items():
            o, h = x["outer"], x["holdout"]
            lines.append(
                f"| {f} | {fmt(o['ann_ret'], '+.1%')} | {fmt(o['vs_base']['mean'], '+.2%')} "
                f"({fmt(o['vs_base']['t'], '+.2f')}) | "
                f"{fmt(h['ann_ret'], '+.1%')} | {fmt(h['vs_base']['mean'], '+.2%')} | {x['verdict']} |"
            )
    text = "\n".join(lines) + "\n"
    (OUT / f"{m}_rules68.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    (OUT / f"{m}_rules68.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
