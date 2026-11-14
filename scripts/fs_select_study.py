"""Feature selection -> model selection (docs/MODELING.md 52, pre-declared 2026-10-12). Research only.

1. Feature pool: the served inputs (KR v3, US v9) + every research block (v13-v17; US also block W).
2. Feature selection INSIDE every fit: the selected set is computed from that fit's own training rows (outer folds,
   inner validation folds, holdout and final fits alike), so no fold ever selects features with data it is
   evaluated on (52.3): IC screen (t >= 1.5) U LightGBM gain top 40, redundancy |rho| > 0.85 removed, 15..80.
3. Models on the selected features (fs_*), tuned by the protocol's inner CV, plus the served configuration on its
   own inputs (*_core), baselines and ensembles; protocol v3 picks the best model.

    .venv/bin/python scripts/fs_select_study.py --market KR

Writes reports/modeling/<M>_fs52.{json,md} and reports/modeling/<M>_fs52_summary.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import location_study as v15s  # noqa: E402
import portfolio_backtest as pb  # noqa: E402
import us_weekly_study as wk  # noqa: E402
import v13_study as v13s  # noqa: E402
import v14_study as v14s  # noqa: E402
import v16_study as v16s  # noqa: E402
import v17_study as v17s  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
FS_MODELS = ("logit_w", "enet_logit", "ridge_rg", "lgbm_clf", "lgbm_rank", "xgb_clf", "cat_rank", "et", "rf")
SERVED = {"KR": "et", "US": "cat_rank"}
IC_T = 1.5
GAIN_TOP = 40
RHO_MAX = 0.85
K_MIN, K_MAX = 15, 80
WINDOW_MONTHS = 60
FS_LOG: dict = {}  # "train_end|rows|xs|regime" -> the selection (persisted: checkpointed contexts do not refit)
FS_STORE: Path | None = None


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ pool


def build_pool(market: str, settings, cache):
    uni, ds = v13s.load_inputs(market, settings, cache)
    base_xs, base_reg = list(ds.xs_cols), list(ds.regime_cols)
    add_x, add_r, info = [], [], {}

    def take(tag, raw, reg, inf):
        add_x.extend(feat.x_col(c) for c in raw)
        add_r.extend(reg)
        info[tag] = {"features": len(raw), "regime": len(reg)}
        say(f"pool {tag}: +{len(raw)} features, +{len(reg)} regime")

    for b in ("T", "F", "S"):
        take(b, *v13s.add_block(b, market, ds, uni, settings))
    for b in ("C", "D", "H"):
        take(b, *v14s.add_block(b, market, ds, uni, settings, cache))
    take("L", *v15s.add_block(market, ds, settings, cache))
    for b in ("X", "J", "B", "R"):
        take(b, *v16s.add_block(b, market, ds, settings, cache))
    take("V", *v17s.add_block("V", market, ds, settings, cache))
    if market == "US":
        w, _ = wk.block_w(ds, uni, settings, cache)
        f = ds.frame
        raw = []
        for c in w.columns:
            vals = w[c].astype(float)
            if vals.notna().sum() and vals.nunique() > 1:
                f[c] = vals.to_numpy()
                f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
                feat.FEATURE_DOCS[c] = wk.W_DOCS.get(c, (c, c))
                raw.append(c)
        take("W", raw, [], {})
    f = ds.frame
    add_x = [c for c in dict.fromkeys(add_x) if c in f and c not in base_xs]
    add_r = [c for c in dict.fromkeys(add_r) if c in f and c not in base_reg]
    ds.core_xs_cols, ds.core_regime_cols = base_xs, base_reg
    ds.xs_cols, ds.regime_cols = [*base_xs, *add_x], [*base_reg, *add_r]
    fs = f"{IC_T},{GAIN_TOP},{RHO_MAX},{K_MIN},{K_MAX},{WINDOW_MONTHS},spearman"
    ds.snapshot = f"{ds.snapshot}|fs52|{fs}|{len(ds.xs_cols)}|{len(ds.regime_cols)}"
    info["pool"] = {"xs": len(ds.xs_cols), "regime": len(ds.regime_cols), "core_xs": len(base_xs)}
    return uni, ds, info


# ------------------------------------------------------------------------------------------------ selection


def _ic_t(me: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Per-column t of the mean per-date Spearman IC (feature rank vs label rank, NaN rows left out)."""
    out = {c: [] for c in cols}
    for _, g in me.groupby("date"):
        y = g["y_e"].rank().to_numpy(dtype=float)
        if len(y) < 20:
            continue
        y = (y - y.mean()) / (y.std() or 1.0)
        xm = g[cols].rank().to_numpy(dtype=float)
        ok = np.isfinite(xm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns on early dates
            mu = np.nanmean(xm, axis=0)
            sd = np.nanstd(xm, axis=0)
        z = np.where(ok, (xm - mu) / np.where(sd > 0, sd, np.nan), 0.0)
        n = ok.sum(axis=0)
        r = (z * y[:, None]).sum(axis=0) / np.maximum(n, 1)
        for j, c in enumerate(cols):
            if n[j] >= 20 and np.isfinite(r[j]):
                out[c].append(r[j])
    t = {}
    for c, v in out.items():
        v = np.asarray(v)
        t[c] = float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 5 and v.std(ddof=1) > 0 else 0.0
    return pd.Series(t)


def select_features(train: pd.DataFrame, xs: list[str], reg: list[str], seed: int = 0) -> tuple[list[str], list[str]]:
    import lightgbm as lgb

    me = train[train["is_me"] & train["labelled"]] if "labelled" in train else train[train["is_me"]]
    me = me[np.isfinite(me["y_e"].to_numpy(dtype=float))]
    dates = pd.DatetimeIndex(sorted(me["date"].unique()))
    if len(dates) > WINDOW_MONTHS:
        me = me[me["date"] >= dates[-WINDOW_MONTHS]]
    key = f"{pd.Timestamp(me['date'].max()).date()}|{len(me)}|{len(xs)}|{len(reg)}"
    if key in FS_LOG:
        return FS_LOG[key]["xs"], FS_LOG[key]["reg"]
    xs = [c for c in xs if c in me]
    reg = [c for c in reg if c in me]
    t = _ic_t(me, xs)
    uni_set = [c for c in xs if abs(t.get(c, 0.0)) >= IC_T]
    y = (me["y_e"] > me.groupby("date")["y_e"].transform("median")).astype(int).to_numpy()
    X = me[[*xs, *reg]].to_numpy(dtype=np.float32)  # noqa: N806
    m = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, importance_type="gain", random_state=seed, verbose=-1,
    ).fit(X, y)  # fmt: skip
    gain = pd.Series(m.feature_importances_, index=[*xs, *reg]).sort_values(ascending=False)
    top = [c for c in gain.index[:GAIN_TOP] if gain[c] > 0]
    reg_sel = [c for c in top if c in reg]
    cand = list(dict.fromkeys([*uni_set, *[c for c in top if c in xs]]))
    rank_t = t.abs().rank(ascending=False)
    rank_g = gain.reindex(xs).rank(ascending=False)
    score = pd.concat([rank_t, rank_g], axis=1).min(axis=1).reindex(cand).sort_values()
    rest = [c for c in t.abs().sort_values(ascending=False).index if c not in cand]  # the floor's fill order
    sample = me.sample(min(len(me), 20000), random_state=seed)[xs]
    corr = sample.rank().corr().abs()
    kept: list[str] = []
    for i, c in enumerate([*score.index, *rest]):
        if i >= len(score) and len(kept) >= K_MIN:
            break
        if all(not (corr.loc[c, k] > RHO_MAX) for k in kept):
            kept.append(c)
        if len(kept) >= K_MAX:
            break
    FS_LOG[key] = {
        "xs": kept,
        "reg": reg_sel,
        "n_ic": len(uni_set),
        "n_union": len(cand),
        "train_end": str(pd.Timestamp(me["date"].max()).date()),
    }
    if FS_STORE is not None:
        FS_STORE.write_text(json.dumps(FS_LOG, ensure_ascii=False))
    return kept, reg_sel


def fs_candidate(base: cmod.Candidate) -> cmod.Candidate:
    name = f"fs_{base.name}"
    base_fit = cmod._FITTERS[base.fitter or base.name]

    def fit(c, train, params, scheme, ctx, base_fit=base_fit):
        xs, reg = select_features(train, list(ctx.xs_cols), list(ctx.regime_cols or []), int(ctx.seed))
        return base_fit(c, train, params, scheme, replace(ctx, xs_cols=xs, regime_cols=reg))

    cmod._FITTERS[name] = fit
    d = copy.copy(base)
    d.name, d.fitter, d.feature_set = name, name, "all"
    d.label_ko = base.label_ko + " · 특징 선택"
    d.label_en = base.label_en + " · feature selection"
    return d


def candidates(market: str) -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    out = [every["mom_12_1"], every["factor_ic"]]
    out += [fs_candidate(every[n]) for n in FS_MODELS]
    core = copy.copy(every[SERVED[market]])
    core.name, core.feature_set = f"{SERVED[market]}_core", "core"
    core.fitter = every[SERVED[market]].fitter or every[SERVED[market]].name
    core.label_ko = every[SERVED[market]].label_ko + " · 서빙 구성(기존 입력)"
    out.append(core)
    return out


# ------------------------------------------------------------------------------------------------ report


def portfolio(market: str, res, uni, ds, name: str) -> dict:
    parts = [res.oos[["date", "entity", name]].assign(part="outer")]
    if res.holdout_preds is not None and name in res.holdout_preds:
        parts.append(res.holdout_preds[["date", "entity", name]].assign(part="holdout"))
    sc = pd.concat(parts, ignore_index=True).rename(columns={name: "score"})
    mc = ds.frame[["date", "entity", "log_mcap"]].drop_duplicates(["date", "entity"])
    sc = sc.merge(mc, on=["date", "entity"], how="left")
    sc["mcap"] = np.exp(sc["log_mcap"].astype(float))
    cal = pd.DatetimeIndex(ds.calendar)
    cal = cal[cal >= pd.Timestamp(sc["date"].min())]
    w = pb.wide(uni.prices, sorted(sc["entity"].unique()), cal)
    mid = pb.cm.AUM[market][1]
    out = {}
    for p, bk in pb.BENCH.items():
        bn, bp = pb.simulate(market, sc, w, bk, mid, 1.0, False)
        for tag, wc in (("gross", False), ("net", True)):
            nav, per = pb.simulate(market, sc, w, p, mid, 1.0, wc)
            out[f"{p}|{tag}"] = pb.stats(nav, bn, per, bp)
    return out


def _f(x, spec: str = "+.4f") -> str:
    return "–" if x is None or not np.isfinite(x) else format(x, spec)


def write_summary(market: str, d: dict, extra: dict) -> str:
    cand = {c["name"]: c for c in d["candidates"]}
    sel_name = d["selected"]
    core = f"{SERVED[market]}_core"
    lines = [
        f"# {market} 특징 선택 → 모델 선정 (docs/MODELING.md 52)",
        "",
        f"- 특징 풀: {extra['pool']}",
        f"- 학습마다 고른 특징 수: 중앙값 {extra['fs_n_median']}, 범위 {extra['fs_n_range']} "
        f"(선택 {extra['fs_fits']}회)",
        f"- 가장 자주 뽑힌 특징(상위 20): {', '.join(extra['top_features'])}",
        "",
        "| 후보 | 외부 AUC [95% CI] | 외부 IC (t) | 홀드아웃 AUC | 비용 후 상위 5분위(월, 외부) |",
        "|---|---|---|---|---|",
    ]
    for n, c in cand.items():
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        tn = c["outer"].get("top_net") or {}
        ic = c["outer"]["ic"]
        star = " **(선정)**" if n == sel_name else ""
        lines.append(
            f"| {n}{star} | {_f(o['mean'], '.4f')} [{_f(o['lo'], '.4f')}, {_f(o['hi'], '.4f')}] | "
            f"{_f(ic['mean'])} ({_f(ic['t'], '+.2f')}) | {_f(h.get('mean'), '.4f')} | "
            f"{_f(tn.get('mean'), '+.3%')} ({_f(tn.get('t'), '+.2f')}) |"
        )
    g = d["gate"]
    lines += ["", f"**가장 괜찮은 모델: `{sel_name}`** · 서빙 게이트 {'통과' if g['passed'] else '미달'}", ""]
    for k, v in extra.get("vs_core", {}).items():
        o, h = v["outer"], v.get("holdout") or {}
        lines.append(
            f"- {k}: 외부 {_f(o['mean'])} [{_f(o['lo'])}, {_f(o['hi'])}] (t {_f(o['t'], '+.2f')}) · "
            f"홀드아웃 {_f(h.get('mean'))} [{_f(h.get('lo'))}, {_f(h.get('hi'))}]"
        )
    lines += ["", "## 선정 모델의 신호 포트폴리오 (중간 규모, 47절 비용)", ""]
    lines += ["| 포트폴리오 | 연 수익 | MDD | 기준 대비 연 초과 | 월 초과 t |", "|---|---|---|---|---|"]
    for k, s in extra.get("portfolio", {}).items():
        lines.append(
            f"| {k} | {_f(s['cagr'], '+.1%')} | {_f(s['mdd'], '+.1%')} | {_f(s['excess_ann'], '+.1%')} | "
            f"{_f(s['excess_t'], '+.2f')} |"
        )
    lines += ["", f"(비교: 서빙 구성 `{core}`)"]
    text = "\n".join(lines) + "\n"
    (OUT / f"{market}_fs52_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    a = ap.parse_args()
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    uni, ds, info = build_pool(a.market, settings, cache)
    say(f"pool: {info['pool']}")
    global FS_STORE
    FS_STORE = settings.cache_path / "fs52" / f"{a.market}_{DiskCache.key('fs52', ds.snapshot)}.json"
    FS_STORE.parent.mkdir(parents=True, exist_ok=True)
    if FS_STORE.exists():
        FS_LOG.update(json.loads(FS_STORE.read_text()))
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE, cmod.ENSEMBLE_W)
    sel.COMPOSITES = (cmod.ENSEMBLE.name, cmod.ENSEMBLE_W.name)
    sel.DIAGNOSTIC_PAIRS = ()
    res = sel.run_selection(ds, settings, cache, candidates=candidates(a.market), threads=8, progress=say)
    d = res.to_dict()
    core = f"{SERVED[a.market]}_core"
    best = d["selected"]
    vs = {}
    for n in dict.fromkeys([best, *[c for c in res.oos.columns if c.startswith("fs_")]]):
        if n in res.oos and core in res.oos and n != core:
            vs[f"{n} − {core}"] = {
                "outer": mt.paired_difference(mt.per_date_primary(res.oos, n), mt.per_date_primary(res.oos, core)),
                "holdout": (
                    mt.paired_difference(
                        mt.per_date_primary(res.holdout_preds, n), mt.per_date_primary(res.holdout_preds, core)
                    )
                    if res.holdout_preds is not None and n in res.holdout_preds
                    else None
                ),
            }
    ns = [len(v["xs"]) for v in FS_LOG.values()]
    cnt = Counter(c for v in FS_LOG.values() for c in v["xs"])
    extra = {
        "pool": info["pool"],
        "blocks": {k: v for k, v in info.items() if k != "pool"},
        "fs_fits": len(FS_LOG),
        "fs_n_median": int(np.median(ns)) if ns else 0,
        "fs_n_range": [int(min(ns)), int(max(ns))] if ns else [0, 0],
        "top_features": [c for c, _ in cnt.most_common(20)],
        "fs_log": FS_LOG,
        "vs_core": vs,
    }
    try:
        extra["portfolio"] = portfolio(a.market, res, uni, ds, best if best in res.oos else core)
    except Exception as exc:  # noqa: BLE001 - the selection result stands without the portfolio table
        say(f"portfolio skipped: {type(exc).__name__}: {exc}")
    d["protocol"]["fs52"] = {k: v for k, v in extra.items() if k != "fs_log"}
    text = json.dumps({**d, "fs_log": FS_LOG}, ensure_ascii=False, indent=1, default=str)
    (OUT / f"{a.market}_fs52.json").write_text(text)
    print(write_summary(a.market, d, extra), flush=True)


if __name__ == "__main__":
    main()
