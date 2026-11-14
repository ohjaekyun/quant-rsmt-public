"""Hybrid: beam-search feature subspace + extra-trees alpha + RSMT-Q blend + risk / sizing layer
(docs/MODELING.md 57, KR, pre-declared 2026-10-13), extended to both markets with the soft decision forest alpha
(docs/MODELING.md 65, pre-declared 2026-10-17). Research only.

    .venv/bin/python scripts/hybrid_study.py --market US

Comparisons are against the served model (KR et, US cat_rank). Writes reports/modeling/<M>_hybrid65.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import costs as cm  # noqa: E402
import fs_select_study as fss  # noqa: E402
import model_zoo as zoo  # noqa: E402
import portfolio_backtest as pb  # noqa: E402
import risk_overlap_study as ro  # noqa: E402
import v13_study as v13s  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
BEAM = {"width": 5, "max_k": 15, "cand": 25, "val_months": 12, "tol": 0.001, "patience": 2, "trees": 60}
BEAM_LOG: dict = {}
BEAM_STORE: Path | None = None
_f = fss._f


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================================================ alpha


def _et(train: pd.DataFrame, cols: list[str], params: dict, scheme: str, ctx, trees: int = 400):
    from sklearn.ensemble import ExtraTreesRegressor

    y = cmod._rg_target(train, scheme)
    ok = np.isfinite(y)
    md = params.get("max_depth")
    m = ExtraTreesRegressor(
        n_estimators=trees,
        max_features=float(params.get("max_features", 0.33)),
        max_depth=None if md in (None, 0) else int(md),
        min_samples_leaf=cmod._frac_leaf(int(ok.sum()), float(params["leaf_frac"])),
        bootstrap=False,
        n_jobs=int(ctx.threads),
        random_state=int(ctx.seed),
    ).fit(train[cols].to_numpy(dtype=np.float64)[ok], y[ok], sample_weight=cmod.row_weights(train, scheme, ctx)[ok])
    return m


def _fit_et_wide(c, train, params, scheme, ctx):
    cols = cmod._tree_cols(ctx, train)
    return cmod.Fitted(c.name, params, scheme, _et(train, cols, params, scheme, ctx), cols, "sk_tree")


def _date_ic(dates: np.ndarray, pred: np.ndarray, y: np.ndarray) -> float:
    d = pd.DataFrame({"d": dates, "p": pred, "y": y})
    r = d.groupby("d")[["p", "y"]].rank()
    r["d"] = d["d"]
    ics = r.groupby("d").apply(lambda g: g["p"].corr(g["y"]) if len(g) >= 20 else np.nan, include_groups=False)
    return float(np.nanmean(ics)) if ics.notna().any() else -1.0


def beam_select(train: pd.DataFrame, ctx) -> list[str]:
    """57.1 beam search on the fit's own month-end rows (memoised by the window); returns the chosen xs columns."""
    from sklearn.ensemble import ExtraTreesRegressor

    me = train[train["is_me"] & train["labelled"]]
    me = me[np.isfinite(me["y_e"].to_numpy(dtype=float))]
    xs = [c for c in ctx.xs_cols if c in me]
    key = f"{pd.Timestamp(me['date'].max()).date()}|{len(me)}|{len(xs)}"
    if key in BEAM_LOG:
        return BEAM_LOG[key]["xs"]
    dates = pd.DatetimeIndex(sorted(me["date"].unique()))
    val_d = dates[-BEAM["val_months"] :]
    cal = pd.DatetimeIndex(ctx.calendar)
    p0 = cal.searchsorted(val_d[0]) - 2 * int(ctx.horizon)
    fit_end = cal[max(p0, 0)]
    fit = me[me["date"] <= fit_end]
    val = me[me["date"].isin(val_d)]
    yf = feat.rank_gauss(fit["y_e"], fit["date"]).to_numpy(dtype=float)
    ok = np.isfinite(yf)
    t = fss._ic_t(fit, xs).abs().sort_values(ascending=False)
    pool = [c for c in t.index[: BEAM["cand"]] if c in xs]
    leaf = cmod._frac_leaf(int(ok.sum()), 0.02)
    xf_all, xv_all = fit[pool].to_numpy(dtype=np.float64)[ok], val[pool].to_numpy(dtype=np.float64)
    vd, vy = val["date"].to_numpy(), val["y_e"].to_numpy(dtype=float)
    idx = {c: i for i, c in enumerate(pool)}
    seen: dict[frozenset, float] = {}

    def score(sub: tuple[str, ...]) -> float:
        k = frozenset(sub)
        if k not in seen:
            j = [idx[c] for c in sub]
            m = ExtraTreesRegressor(
                n_estimators=BEAM["trees"],
                max_features=0.5,
                min_samples_leaf=leaf,
                bootstrap=False,
                n_jobs=int(ctx.threads),
                random_state=int(ctx.seed),
            ).fit(xf_all[:, j], yf[ok])
            seen[k] = _date_ic(vd, m.predict(xv_all[:, j]), vy)
        return seen[k]

    beams: list[tuple[tuple[str, ...], float]] = [((), -1.0)]
    best, best_s, stale = (), -1.0, 0
    for _depth in range(BEAM["max_k"]):
        nxt = {}
        for sub, _s in beams:
            for c in pool:
                if c not in sub:
                    cand = tuple(sorted((*sub, c)))
                    if cand not in nxt:
                        nxt[cand] = score(cand)
        beams = sorted(nxt.items(), key=lambda kv: -kv[1])[: BEAM["width"]]
        top_sub, top_s = beams[0]
        if top_s > best_s + BEAM["tol"]:
            best, best_s, stale = top_sub, top_s, 0
        else:
            stale += 1
            if top_s > best_s:
                best, best_s = top_sub, top_s
            if stale >= BEAM["patience"]:
                break
    out = list(best)
    BEAM_LOG[key] = {
        "xs": out,
        "score": best_s,
        "evals": len(seen),
        "train_end": str(pd.Timestamp(me["date"].max()).date()),
    }
    if BEAM_STORE is not None:
        BEAM_STORE.write_text(json.dumps(BEAM_LOG, ensure_ascii=False))
    say(f"beam {BEAM_LOG[key]['train_end']}: {len(out)} features, val IC {best_s:+.4f}, {len(seen)} subsets")
    return out


def _fit_beam_et(c, train, params, scheme, ctx):
    from dataclasses import replace

    sub = beam_select(train, ctx)
    ctx2 = replace(ctx, xs_cols=sub)
    cols = cmod._tree_cols(ctx2, train)
    return cmod.Fitted(c.name, params, scheme, _et(train, cols, params, scheme, ctx2), cols, "sk_tree")


def candidates(market: str) -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    served = [] if fss.SERVED[market] == "et" else [every[fss.SERVED[market]]]
    grid = [
        {"leaf_frac": lf, "max_features": mf, "max_depth": md}
        for lf in (0.005, 0.01, 0.02)
        for mf in (0.33, 0.6)
        for md in (0, 8)
    ]
    cmod._FITTERS["et_wide"] = _fit_et_wide
    cmod._FITTERS["beam_et"] = _fit_beam_et
    return [
        every["mom_12_1"],
        *served,
        every["et"],
        cmod.Candidate(
            "et_wide",
            "bagging",
            "엑스트라 트리 넓은 격자(400그루, 깊이·특징 비율 튜닝)",
            "extra trees, wide grid",
            grid=grid,
            schemes=cmod._RANK_SCHEMES,
            fitter="et_wide",
        ),
        cmod.Candidate(
            "beam_et",
            "bagging",
            "빔 서치 특징 부분공간 + 엑스트라 트리",
            "beam-search subspace + extra trees",
            grid=grid,
            schemes=cmod._RANK_SCHEMES,
            fitter="beam_et",
        ),
        every["rsmt_v3"],
        zoo.soft_forest_candidate(),
    ]


def _blend(frame: pd.DataFrame, base: str = "et") -> pd.Series:
    return sel.rank_mean(frame, [base, "rsmt_v3"])


# ================================================================================================ risk layer


def _atr14(px: pd.DataFrame) -> pd.Series:
    c = px["close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([px["high"] - px["low"], (px["high"] - pc).abs(), (px["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=10).mean()


def _cvar5(px: pd.DataFrame, at: pd.Timestamp) -> float:
    r = px["close"].astype(float).pct_change().loc[:at].dropna().iloc[-252:]
    if len(r) < 120:
        return np.nan
    cut = r.quantile(0.05)
    return float(-r[r <= cut].mean())


def _cap_scale(w: pd.Series, gross: float, cap: float = 0.12) -> pd.Series:
    """Scale proportional weights to `gross`, capping single names at `cap` (capped names fixed, rest re-scaled)."""
    w = w.clip(lower=0)
    if w.sum() <= 0:
        return w
    out = pd.Series(0.0, index=w.index)
    free = w.copy()
    budget = gross
    for _ in range(50):
        s = free / free.sum() * budget
        over = s > cap
        if not over.any():
            out.loc[s.index] = s
            break
        out.loc[s.index[over]] = cap
        budget -= cap * over.sum()
        free = free[~over]
        if free.empty or budget <= 0:
            break
    return out


def book_weights(g: pd.DataFrame, kind: str, px: dict, mult: float = 1.0) -> pd.Series:
    """Target weights (sum <= 1, rest cash) of one rebalance cross-section g (date, entity, score, ewma, cvar)."""
    g = g.dropna(subset=["score"])
    z = (g["score"] - g["score"].mean()) / (g["score"].std(ddof=0) or 1.0)
    top = g[g["score"].rank(pct=True) > 0.8].assign(z=z)
    top = top.set_index("entity")
    if top.empty:
        return pd.Series(dtype=float)
    if kind in ("R0", "R3", "R4"):
        w = pd.Series(1.0 / len(top), index=top.index)
        return w * (mult if kind == "R4" else 1.0)
    if kind == "R1":
        sig = top["ewma"].clip(lower=0.10)
        return _cap_scale(top["z"].clip(lower=1e-6) / sig**2, 0.95)
    if kind == "R2":
        cv = top["cvar"].where(top["cvar"] > 0)
        return _cap_scale(top["z"].clip(lower=1e-6) / cv.fillna(cv.median()), 1.0)
    raise ValueError(kind)


def simulate_book(
    market: str,
    sc: pd.DataFrame,
    w: dict,
    kind: str,
    aum: float,
    px: dict,
    atr: pd.DataFrame,
    mult: pd.Series | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Daily NAV with cash, the 47-section cost model, next-session execution, optional ATR stop (R3)."""
    cal = w["close"].index
    dates = sorted(sc["date"].unique())
    execs = [cal[min(cal.searchsorted(pd.Timestamp(d), "right"), len(cal) - 1)] for d in dates]
    by_date = {d: g for d, g in sc.groupby("date")}
    hold = pd.Series(dtype=float)
    cash = float(aum)
    path, periods, n_stops = [], [], 0

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
        m = float(mult.asof(pd.Timestamp(d))) if mult is not None and len(mult) else 1.0
        tw = book_weights(g[ok], kind, px, m if np.isfinite(m) else 1.0)
        nav = cash + float(hold.sum())
        tgt = tw * nav
        names = tgt.index.union(hold.index)
        trade = (tgt.reindex(names).fillna(0.0) - hold.reindex(names).fillna(0.0)).abs()
        c = cost_of(trade, ex)
        nav_after = nav - c
        hold = tw * nav_after
        cash = nav_after - float(hold.sum())
        stop_lvl = None
        if kind == "R3" and len(hold):
            entry = w["close"].loc[ex].reindex(hold.index)
            stop_lvl = entry - 2.0 * atr.loc[ex].reindex(hold.index).fillna(np.inf)
        end = execs[i + 1] if i + 1 < len(execs) else cal[min(cal.searchsorted(ex) + 21, len(cal) - 1)]
        span = cal[(cal > ex) & (cal <= end)]
        start_nav = nav_after
        for day in span:
            if len(hold):
                hold = hold * (1.0 + w["ret"].loc[day, hold.index].fillna(0.0))
                if stop_lvl is not None:
                    hit = w["close"].loc[day, hold.index] < stop_lvl.reindex(hold.index)
                    hit = hit[hit & (hold > 0)]
                    if len(hit):
                        val = hold[hit.index]
                        cash += float(val.sum()) - cost_of(val.abs(), day)
                        hold = hold.drop(hit.index)
                        n_stops += len(hit)
            path.append((day, cash + float(hold.sum())))
        periods.append(
            {
                "date": d,
                "exec": ex,
                "part": g["part"].iloc[0],
                "turnover": float(trade.sum() / (2 * nav)),
                "cost": c / nav,
                "ret": (cash + float(hold.sum())) / nav - 1.0 if nav > 0 else 0.0,
                "start": start_nav,
            }
        )
    nav_s = pd.Series(dict(path)).sort_index()
    per = pd.DataFrame(periods)
    per.attrs["stops"] = n_stops
    return nav_s, per


def risk_layer(market: str, sc: pd.DataFrame, uni, ds, settings, cache) -> dict:
    cal = pd.DatetimeIndex(ds.calendar)
    cal = cal[cal >= pd.Timestamp(sc["date"].min())]
    ents = sorted(sc["entity"].unique())
    w = pb.wide(uni.prices, ents, cal)
    px = {e: uni.prices[e].sort_index() for e in ents if e in uni.prices}
    atr = pd.DataFrame({e: _atr14(p) for e, p in px.items() if {"high", "low"} <= set(p.columns)}).reindex(cal).ffill()
    sc = sc.copy()
    sc["ewma"] = [
        float(ro.ewma_vol(px[e]).asof(pd.Timestamp(d))) if e in px else np.nan for e, d in zip(sc["entity"], sc["date"])
    ]
    sc["cvar"] = [_cvar5(px[e], pd.Timestamp(d)) if e in px else np.nan for e, d in zip(sc["entity"], sc["date"])]
    from quantcio.macro import load_macro_series
    from quantcio.macro.regime import overlay_multiplier, regime_history

    hist = regime_history(settings, cache, series=load_macro_series(settings, cache))
    h = hist[hist["region"].astype(str).str.upper() == market].set_index("date")["score"].sort_index()
    mult = h.map(
        lambda x: overlay_multiplier(None if pd.isna(x) else float(x), settings.macro.overlay_min_mult, warn=False)
    )
    mid = cm.AUM[market][1]
    bn, _ = pb.simulate(market, sc, w, "B1", mid, 1.0, False)
    out = {}
    for kind in ("R0", "R1", "R2", "R3", "R4"):
        nav, per = simulate_book(market, sc, w, kind, mid, px, atr, mult if kind == "R4" else None)
        mo = nav.resample("ME").last().pct_change().dropna()
        bm = bn.resample("ME").last().pct_change().reindex(mo.index).fillna(0.0)
        st = ro._book_stats(mo, bm)
        dr = nav.pct_change().dropna()
        st["sharpe_daily"] = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else float("nan")
        st["mdd_daily"] = pb.mdd(nav)  # ro._book_stats' mdd is on month-end NAV
        st["turnover_yr"] = float(per["turnover"].mean() * 12)
        st["stops"] = int(per.attrs.get("stops", 0))
        st["stress"] = {k: pb.window(nav, bn, a, b) for k, (a, b) in pb.STRESS.items()}
        out[kind] = st
    b0 = out["R0"]
    for k in ("R1", "R2", "R3", "R4"):
        o = out[k]
        out[k]["helps"] = bool(
            o["mdd"] > b0["mdd"] and o["cvar5"] > b0["cvar5"] and o["ann_ret"] >= b0["ann_ret"] - 0.01
        )
    return out


# ================================================================================================ report


def _stack_bases(d: dict) -> dict:
    rs = next((c for c in d["candidates"] if c["name"] == "rsmt_v3"), None)
    return {k: (v or {}).get("stack_base") for k, v in ((rs or {}).get("configs") or {}).items()}


def write_md(m: str, r: dict) -> str:
    sv = r["served"]
    lines = [
        f"# {m} 하이브리드 — 빔 서치 + et + RSMT-Q + 소프트 결정 숲 (docs/MODELING.md 57·65)",
        "",
        "## 알파 후보",
        "",
    ]
    lines += ["| 후보 | 외부 AUC [95% CI] | 외부 IC (t) | 홀드아웃 AUC |", "|---|---|---|---|"]
    for n, x in r["table"].items():
        o, h, ic = x["outer"]["auc_median"], (x.get("holdout") or {}).get("auc_median") or {}, x["outer"]["ic"]
        star = " **(1위)**" if n == r["winner"] else ""
        lines.append(
            f"| {n}{star} | {_f(o['mean'], '.4f')} [{_f(o['lo'], '.4f')}, {_f(o['hi'], '.4f')}] | "
            f"{_f(ic['mean'])} ({_f(ic['t'], '+.2f')}) | {_f(h.get('mean'), '.4f')} |"
        )
    for k, x in r["vs_served"].items():
        o, h = x["outer"], x["holdout"]
        lines.append(
            f"- {k} − {sv}: 외부 {_f(o['mean'])} [{_f(o['lo'])}, {_f(o['hi'])}] (t {_f(o['t'], '+.2f')}) · "
            f"홀드아웃 {_f(h.get('mean'))}"
        )
    lines += ["", f"→ 서빙 교체 후보: **{r['switch'] or f'없음 — {sv} 유지'}**", ""]
    b = r["beam"]
    lines += [
        f"- 빔 서치 {b['fits']}회: 고른 특징 수 중앙값 {b['k_median']} (범위 {b['k_range']}), "
        f"검증 IC 중앙값 {b['ic_median']:+.4f}",
        f"- 자주 뽑힌 특징: {', '.join(b['top'])}",
        "",
    ]
    for alpha, rl in r["risk"].items():
        aum = "10억 원" if m == "KR" else "$1,000만"
        lines += [f"## 위험·사이징 층 — 알파 `{alpha}` (일별, 47절 비용, {aum})", ""]
        lines += [
            "| 방식 | 연 수익 | 샤프 | 최대 낙폭 | 월 CVaR 5% | 기준 대비 연 초과 (t) | 회전율/년 | 손절 "
            "| 코로나 | 2022 긴축 | 판정 |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for k, ko in (
            ("R0", "상위 20% 동일가중"),
            ("R1", "분수 켈리형"),
            ("R2", "꼬리 위험(CVaR)"),
            ("R3", "ATR 손절"),
            ("R4", "매크로 오버레이"),
        ):
            x = rl[k]
            c19 = (x["stress"].get("코로나 급락") or {}).get("ret")
            c22 = (x["stress"].get("2022 긴축 약세장") or {}).get("ret")
            verdict = "기준" if k == "R0" else ("도움" if x.get("helps") else "도움 없음")
            lines.append(
                f"| {ko} | {_f(x['ann_ret'], '+.1%')} | {_f(x['sharpe_daily'], '.2f')} | {_f(x['mdd'], '+.1%')} | "
                f"{_f(x['cvar5'], '+.1%')} | {_f(x['excess_ann'], '+.1%')} ({_f(x['excess_t'], '+.2f')}) | "
                f"{_f(x['turnover_yr'], '.1f')} | {x['stops']} | {_f(c19, '+.1%')} | {_f(c22, '+.1%')} | {verdict} |"
            )
        lines.append("")
    text = "\n".join(lines) + "\n"
    (OUT / f"{m}_hybrid65.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    global BEAM_STORE
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    m = ap.parse_args().market
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    uni, ds = v13s.load_inputs(m, settings, cache)
    ds.core_xs_cols, ds.core_regime_cols = list(ds.xs_cols), list(ds.regime_cols)
    tag = ",".join(f"{k}={v}" for k, v in BEAM.items())
    ds.snapshot = f"{ds.snapshot}|hybrid65|{tag}"
    BEAM_STORE = settings.cache_path / "fs52" / f"{m}_beam_{DiskCache.key('beam', ds.snapshot)}.json"
    if BEAM_STORE.exists():
        BEAM_LOG.update(json.loads(BEAM_STORE.read_text()))
    sel.COMPOSITE_CANDIDATES, sel.COMPOSITES, sel.DIAGNOSTIC_PAIRS = (), (), ()
    served = fss.SERVED[m]
    res = sel.run_selection(ds, settings, cache, candidates=candidates(m), threads=8, progress=say)
    d = res.to_dict()
    table = {c["name"]: {"outer": c["outer"], "holdout": c["holdout"]} for c in d["candidates"]}
    oos, hold = res.oos.copy(), res.holdout_preds.copy()
    for base in dict.fromkeys(["et", served]):
        if "rsmt_v3" in oos and base in oos:
            name = f"blend_{base}_rsmt"
            oos[name], hold[name] = _blend(oos, base), _blend(hold, base)
            table[name] = {
                "outer": mt.evaluate(oos, ds.cost_bps, name),
                "holdout": mt.evaluate(hold, ds.cost_bps, name),
            }
    pool = {n: t["outer"] for n, t in table.items() if n != "mom_12_1"}
    winner = sel.select_best(pool)
    vs = {}
    for n in pool:
        if n != served:
            vs[n] = {
                "outer": mt.paired_difference(mt.per_date_primary(oos, n), mt.per_date_primary(oos, served)),
                "holdout": mt.paired_difference(mt.per_date_primary(hold, n), mt.per_date_primary(hold, served)),
            }
    sw = vs.get(winner)
    switch = winner if sw and (sw["outer"].get("lo") or -1) > 0 and (sw["holdout"].get("mean") or -1) >= 0 else None
    ks = [len(v["xs"]) for v in BEAM_LOG.values()]
    cnt = Counter(c for v in BEAM_LOG.values() for c in v["xs"])
    r = {
        "market": m,
        "served": served,
        "table": table,
        "winner": winner,
        "vs_served": vs,
        "switch": switch,
        "gate": d["gate"],
        "stack_base": _stack_bases(d),
        "beam": {
            "fits": len(BEAM_LOG),
            "k_median": int(np.median(ks)) if ks else 0,
            "k_range": [min(ks), max(ks)] if ks else [],
            "ic_median": float(np.median([v["score"] for v in BEAM_LOG.values()])) if ks else float("nan"),
            "top": [c for c, _ in cnt.most_common(15)],
            "log": BEAM_LOG,
        },
        "risk": {},
    }
    mc = ds.frame[["date", "entity", "log_mcap"]].drop_duplicates(["date", "entity"])
    for alpha in dict.fromkeys([winner, served]):
        sc = pd.concat(
            [
                oos[["date", "entity", alpha]].assign(part="outer"),
                hold[["date", "entity", alpha]].assign(part="holdout"),
            ]
        )
        sc = sc.rename(columns={alpha: "score"}).merge(mc, on=["date", "entity"], how="left")
        sc["mcap"] = np.exp(sc["log_mcap"].astype(float))
        try:
            r["risk"][alpha] = risk_layer(m, sc, uni, ds, settings, cache)
        except Exception as exc:  # noqa: BLE001 - the alpha result stands without the risk table
            say(f"risk layer {alpha} skipped: {type(exc).__name__}: {exc}")
    (OUT / f"{m}_hybrid65.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    print(write_md(m, r), flush=True)


if __name__ == "__main__":
    main()
