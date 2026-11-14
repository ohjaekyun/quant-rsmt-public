"""Shadow-feature (Boruta-style) selection for every model family (docs/MODELING.md 72, pre-declared 2026-10-20).
Research only.

    .venv/bin/python scripts/shadow_select.py --market KR       # one heavy run at a time

Per training window (outer and inner folds, the 60-month window of section 52): every feature gets a shadow copy whose
rows are shuffled within each date; R = 50 times a 70% subsample of the window's months is drawn, the shadows are
reshuffled and a LightGBM classifier is fitted on real + shadow columns; a feature scores a hit when its gain beats
the best shadow. Hits ~ Binomial(R, 0.5) under "no better than the best noise"; two-sided exact tests with Holm
(alpha 0.05) confirm / reject; undecided features are kept when their median gain beats the median best shadow.
Candidates `bs_<model>` use the confirmed set; the study compares them with section 52's `fs_<model>` on the 55 pool.
Writes reports/modeling/<M>_shadow72.{json,md}.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_select_study as fss  # noqa: E402

from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

R_RUNS, SUBSAMPLE, ALPHA, FALLBACK_K = 50, 0.7, 0.05, 5
BS_LOG: dict = {}
BS_STORE: Path | None = None


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def holm(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Boolean rejections of Holm's step-down procedure."""
    order = np.argsort(pvals)
    m = len(pvals)
    out = np.zeros(m, dtype=bool)
    for i, j in enumerate(order):
        if pvals[j] <= alpha / (m - i):
            out[j] = True
        else:
            break
    return out


def shadow_select(train: pd.DataFrame, xs: list[str], reg: list[str], seed: int = 0) -> tuple[list[str], list[str]]:
    import lightgbm as lgb

    me = train[train["is_me"] & train["labelled"]] if "labelled" in train else train[train["is_me"]]
    me = me[np.isfinite(me["y_e"].to_numpy(dtype=float))]
    dates = pd.DatetimeIndex(sorted(me["date"].unique()))
    if len(dates) > fss.WINDOW_MONTHS:
        me = me[me["date"] >= dates[-fss.WINDOW_MONTHS]]
    key = f"{pd.Timestamp(me['date'].max()).date()}|{len(me)}|{len(xs)}|{len(reg)}"
    if key in BS_LOG:
        return BS_LOG[key]["xs"], BS_LOG[key]["reg"]
    cols = [c for c in [*xs, *reg] if c in me]
    p = len(cols)
    X = me[cols].to_numpy(dtype=np.float32)  # noqa: N806
    y = (me["y_e"] > me.groupby("date")["y_e"].transform("median")).astype(int).to_numpy()
    codes, uniq = pd.factorize(me["date"])
    groups = [np.where(codes == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    hits = np.zeros(p)
    imp_runs, best_shadow = [], []
    for r in range(R_RUNS):
        pick = rng.choice(len(groups), max(1, int(round(SUBSAMPLE * len(groups)))), replace=False)
        blocks = [groups[k] for k in pick]
        rows = np.concatenate(blocks)
        S = np.empty((len(rows), p), dtype=np.float32)  # noqa: N806
        at = 0
        for b in blocks:  # shuffle each column independently within the date
            perm = np.argsort(rng.random((len(b), p)), axis=0)
            S[at : at + len(b)] = np.take_along_axis(X[b], perm, axis=0)
            at += len(b)
        m = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            importance_type="gain",
            random_state=seed * 1000 + r,
            n_jobs=2,
            verbose=-1,
        ).fit(np.hstack([X[rows], S]), y[rows])
        g = m.feature_importances_
        real, shadow = g[:p], g[p:]
        t = shadow.max()
        hits += real > t
        imp_runs.append(real)
        best_shadow.append(t)
    p_hi = binom.sf(hits - 1, R_RUNS, 0.5)  # P(H >= h)
    p_lo = binom.cdf(hits, R_RUNS, 0.5)  # P(H <= h)
    confirmed = holm(np.minimum(1.0, 2 * p_hi), ALPHA) & (hits > R_RUNS / 2)
    rejected = holm(np.minimum(1.0, 2 * p_lo), ALPHA) & (hits < R_RUNS / 2)
    med = np.median(np.vstack(imp_runs), axis=0)
    tentative = ~confirmed & ~rejected
    keep = confirmed | (tentative & (med > np.median(best_shadow)))
    if not keep.any():
        keep[np.argsort(-hits)[:FALLBACK_K]] = True
    kept = [c for c, k in zip(cols, keep, strict=True) if k]
    BS_LOG[key] = {
        "xs": [c for c in kept if c in xs],
        "reg": [c for c in kept if c in reg],
        "n_confirmed": int(confirmed.sum()),
        "n_rejected": int(rejected.sum()),
        "n_tentative_kept": int((tentative & keep).sum()),
        "p": p,
        "train_end": str(pd.Timestamp(me["date"].max()).date()),
        "hits": {c: int(h) for c, h in zip(cols, hits, strict=True)},
    }
    if BS_STORE is not None:
        BS_STORE.write_text(json.dumps(BS_LOG, ensure_ascii=False))
    return BS_LOG[key]["xs"], BS_LOG[key]["reg"]


def bs_candidate(base: cmod.Candidate) -> cmod.Candidate:
    name = f"bs_{base.name}"
    base_fit = cmod._FITTERS[base.fitter or base.name]

    def fit(c, train, params, scheme, ctx, base_fit=base_fit):
        xs, reg = shadow_select(train, list(ctx.xs_cols), list(ctx.regime_cols or []), int(ctx.seed))
        return base_fit(c, train, params, scheme, replace(ctx, xs_cols=xs, regime_cols=reg))

    cmod._FITTERS[name] = fit
    d = copy.copy(base)
    d.name, d.fitter, d.feature_set = name, name, "all"
    d.label_ko = base.label_ko + " · 그림자 변수 선택"
    d.label_en = base.label_en + " · shadow-feature selection"
    return d


def main() -> None:
    global BS_STORE
    import v21_study as v21s

    from quantcio.config import PROJECT_ROOT, load_settings
    from quantcio.data.cache import DiskCache

    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    m = ap.parse_args().market
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    st = v21s.run_through_55(m, settings, cache)
    ds, res55 = st["ds"], st["res55"]
    ds.snapshot = f"{ds.snapshot}|shadow72|R{R_RUNS}|s{SUBSAMPLE}|a{ALPHA}"
    BS_STORE = Path(settings.cache_path) / "fs52" / f"{m}_shadow_{DiskCache.key('bs', ds.snapshot)}.json"
    if BS_STORE.exists():
        BS_LOG.update(json.loads(BS_STORE.read_text()))
    every = {c.name: c for c in cmod.default_candidates()}
    cands = [every["mom_12_1"], *(bs_candidate(every[n]) for n in fss.FS_MODELS), every["rsmt_v3"]]
    sel.COMPOSITE_CANDIDATES, sel.COMPOSITES, sel.DIAGNOSTIC_PAIRS = (), (), ()
    say(f"{m}: {len(ds.xs_cols)} pool features, candidates {[c.name for c in cands]}")
    res = sel.run_selection(ds, settings, cache, candidates=cands, threads=8, progress=say)
    per = {}
    for n in fss.FS_MODELS:
        a, b = f"bs_{n}", f"fs_{n}"
        if a in res.oos and b in res55.oos:
            per[n] = {
                "outer": mt.paired_difference(mt.per_date_primary(res.oos, a), mt.per_date_primary(res55.oos, b)),
                "holdout": mt.paired_difference(
                    mt.per_date_primary(res.holdout_preds, a), mt.per_date_primary(res55.holdout_preds, b)
                ),
            }

    def mean_diff(part: str) -> dict:
        frames = [
            mt.per_date_primary(res.oos if part == "outer" else res.holdout_preds, f"bs_{n}")
            - mt.per_date_primary(res55.oos if part == "outer" else res55.holdout_preds, f"fs_{n}")
            for n in per
        ]
        return mt.summarize(pd.concat(frames, axis=1).mean(axis=1).dropna().to_numpy())

    o, h = mean_diff("outer"), mean_diff("holdout")
    better = (o.get("lo") or -1) > 0 and (h.get("mean") or -1) >= 0
    logs = list(BS_LOG.values())
    fs_logs = {x["train_end"]: set(x["xs"]) for x in fss.FS_LOG.values() if "train_end" in x}
    overlap = [
        len(set(x["xs"]) & fs_logs[x["train_end"]]) / max(len(set(x["xs"]) | fs_logs[x["train_end"]]), 1)
        for x in logs
        if x["train_end"] in fs_logs
    ]
    sets = [set(x["xs"]) for x in sorted(logs, key=lambda x: x["train_end"])]
    jac = [len(a & b) / max(len(a | b), 1) for a, b in zip(sets, sets[1:], strict=False)]
    d = res.to_dict()
    r = {
        "market": m,
        "design": {
            "R": R_RUNS,
            "subsample": SUBSAMPLE,
            "alpha": ALPHA,
            "shadows": "one per feature, shuffled within date",
        },
        "selection": d,
        "per_model": per,
        "mean_outer": o,
        "mean_holdout": h,
        "better": better,
        "windows": len(logs),
        "kept_median": float(np.median([len(x["xs"]) + len(x["reg"]) for x in logs])) if logs else None,
        "kept_range": [min(len(x["xs"]) for x in logs), max(len(x["xs"]) for x in logs)] if logs else None,
        "confirmed_median": float(np.median([x["n_confirmed"] for x in logs])) if logs else None,
        "jaccard_vs_fs52_median": float(np.median(overlap)) if overlap else None,
        "jaccard_consecutive_median": float(np.median(jac)) if jac else None,
    }
    f = fss._f
    lines = [
        f"# {m} 랜덤 그림자 변수 선택 (Boruta 방식, docs/MODELING.md 72)",
        "",
        f"- 설계: 반복 {R_RUNS}회, 월 단위 {SUBSAMPLE:.0%} 표본, 특징마다 날짜 안에서 섞은 그림자 1개, "
        f"이항 검정 + Holm α = {ALPHA}",
        f"- 학습 창 {r['windows']}개 · 남긴 특징 수 중앙값 {r['kept_median']} (범위 {r['kept_range']}) · "
        f"채택(검정 통과) 중앙값 {r['confirmed_median']}",
        f"- 52절 선택과의 겹침(Jaccard) 중앙값 {f(r['jaccard_vs_fs52_median'], '.2f')} · "
        f"이웃 창 사이 안정성 {f(r['jaccard_consecutive_median'], '.2f')}",
        f"- **그림자 선택 − 52절 선택**(같은 모형 9개 평균): 외부 {f(o['mean'])} [{f(o['lo'])}, {f(o['hi'])}] · "
        f"홀드아웃 {f(h['mean'])} "
        f"→ **{'더 나음' if better else '나아지지 않음'}**",
        "",
        "| 모형 | 외부 차이 [95% CI] | 홀드아웃 차이 |",
        "|---|---|---|",
    ]
    for n, x in per.items():
        lines.append(
            f"| {n} | {f(x['outer']['mean'])} [{f(x['outer']['lo'])}, {f(x['outer']['hi'])}] | "
            f"{f(x['holdout'].get('mean'))} |"
        )
    g = d["gate"]
    lines += [
        "",
        f"- 그림자 선택 후보 중 1위 `{d['selected']}` · 외부 {f(g['auc'], '.4f')} (하한 {f(g['auc_lo'], '.4f')}) · "
        f"게이트 {'통과' if g['passed'] else '미달'}",
    ]
    out = PROJECT_ROOT / "reports" / "modeling"
    text = "\n".join(lines) + "\n"
    (out / f"{m}_shadow72.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    (out / f"{m}_shadow72.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
