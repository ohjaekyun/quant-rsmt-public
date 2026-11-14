"""Joint model / variable-set / hyper-parameter selection (docs/MODELING.md section 24, pre-declared 2026-10-04).

The variable set is a grid dimension ``fs`` chosen by the inner CV like any hyper-parameter (nested: outer test dates
never choose it): ``core`` (the served v3 inputs), ``all`` (every research block + the API-key block), ``ic30`` (the 30
inputs of ``all`` with the largest |t| of the per-date rank IC on the fit's own training month-ends).
Research only: the served models and ``src/quantcio`` are not changed.

    .venv/bin/python scripts/full_opt_study.py --market US
    .venv/bin/python scripts/full_opt_study.py --market KR

Writes reports/modeling/<MKT>_fullopt.{md,json} and reports/modeling/<MKT>_fullopt_summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_study as ks  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.dataset import build_dataset  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
FS = ("core", "all", "ic30")
IC_TOP = 30
SETS: dict[str, list[str]] = {}  # "core_xs", "core_reg", "all_xs", "all_reg"
_IC_MEMO: dict[tuple, list[str]] = {}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ blocks


def _x(f: pd.DataFrame, c: str, raw: bool) -> str:
    f[feat.x_col(c)] = f[c].clip(-3, 3) if raw else feat.rank_gauss(f[c], f["date"])
    return feat.x_col(c)


def us_blocks(ds, settings, cache, universe=None, fundamentals=None) -> tuple[list[str], list[str], dict]:
    import us_v4_study as v4s
    import us_v5_data as v5
    import us_v7_data as v7
    import us_v8_data as v8

    from quantcio.data.pit import load_pit_universe

    f = ds.frame
    xs, reg, info = [], [], {}
    say("block v4 (earnings events / quality)")
    v4s.add_v4_features(ds, settings, cache, universe=universe, fundamentals=fundamentals)
    xs += [feat.x_col(c) for c in v4s.v4.FEATURES]
    say("block v5 (short selling)")
    syms = set(f["ticker"].map(v5.norm_symbol))
    ftd, cover = v5.ftd_panel(settings.cache_path, syms)
    uni = universe or load_pit_universe(Market.US, settings, cache, 500, date.fromisoformat(settings.backtest.us_start))
    sessions = pd.DatetimeIndex(uni.benchmarks["SPY"].index)
    sf = v5.short_features(
        f[["date", "ticker", "mcap", "close"]], v5.finra_panel(settings.cache_path, syms), ftd, sessions, cover
    )
    for c in v5.FEATURES:
        f[c] = sf[c].to_numpy(float)
        xs.append(_x(f, c, False))
        feat.FEATURE_DOCS[c] = v5.DOCS[c]
    say("block v7 (weather / policy / news sentiment / extra macro)")
    drv = v7.build_drivers(settings, cache, say)
    bench = uni.benchmarks["SPY"]["close"].astype(float)
    sf = v7.stock_features(f[["date", "entity", "sector_group"]], uni.prices, bench, drv, feat.rank_gauss)
    for c in v7.FEATURES:
        f[c] = sf[c].to_numpy(float)
        xs.append(_x(f, c, c.startswith(("us_impact_", "us_sec_"))))
        feat.FEATURE_DOCS[c] = v7.DOCS[c]
    for r in v7.REGIME:
        f[r] = sf[r].to_numpy(float)
        reg.append(r)
    say("block v8 (R&D / employees / H-1B / JOLTS)")
    d = v8.load_all(ds, settings, cache, say)
    sf = v8.stock_features(
        f[["date", "entity", "mcap", "sector_group"]], d["chains"], d["rd"], d["emp"], d["h1b"], d["jolts"]
    )
    raw8 = {"us_rd_none", "us_emp_none", "us_sec_hires", "us_sec_quits", "us_sec_layoffs"}
    for c in v8.FEATURES:
        f[c] = sf[c].to_numpy(float)
        xs.append(_x(f, c, c in raw8))
        feat.FEATURE_DOCS[c] = v8.DOCS[c]
    say("block K (API keys)")
    kx, kreg, kinfo = ks.us_block(ds, settings, cache, universe=universe)
    for c in kx:
        xs.append(_x(f, c, c in ks.RAW))
        feat.FEATURE_DOCS[c] = ks.DOCS.get(c, (c, c))
    reg += kreg
    info["keyed"] = {"xs": kx, "regime": kreg, **kinfo}
    return xs, reg, info


def kr_blocks(ds, settings, cache) -> tuple[list[str], list[str], dict]:
    f = ds.frame
    say("block K (API keys: ECOS / KOSIS / KIPRIS)")
    kx, kreg, kinfo = ks.kr_block(ds, settings, cache)
    xs = []
    for c in kx:
        xs.append(_x(f, c, c in ks.RAW))
        feat.FEATURE_DOCS[c] = ks.DOCS.get(c, (c, c))
    return xs, kreg, {"keyed": {"xs": kx, "regime": kreg, **kinfo}}


# ------------------------------------------------------------------------------------------------ variable sets


def ic_top(train: pd.DataFrame) -> list[str]:
    """IC_TOP inputs of ``all`` by |t| of the per-date rank IC on the training month-ends (memoised per window)."""
    me = train[train["is_me"]] if "is_me" in train else train
    key = (str(me["date"].min()), str(me["date"].max()), len(me))
    if key in _IC_MEMO:
        return _IC_MEMO[key]
    cols = [c for c in SETS["all_xs"] if c in me]
    y = feat.rank_gauss(me["y_e"], me["date"])
    stats = {}
    for c in cols:
        x = me[c]
        ok = x.notna() & y.notna()
        if ok.sum() < 1000:
            continue
        g = pd.DataFrame({"d": me["date"][ok], "x": x[ok], "y": y[ok]}).groupby("d")
        ic = g.apply(lambda t: t["x"].corr(t["y"]) if len(t) >= 20 and t["x"].std() > 0 else np.nan).dropna()
        if len(ic) >= 12 and ic.std() > 0:
            stats[c] = abs(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))))
    top = [c for c, _ in sorted(stats.items(), key=lambda kv: -kv[1])[:IC_TOP]]
    _IC_MEMO[key] = top
    return top


def _ctx_for(ctx, train: pd.DataFrame, fs: str):
    if fs == "core":
        return replace(ctx, xs_cols=SETS["core_xs"], regime_cols=SETS["core_reg"])
    if fs == "all":
        return replace(ctx, xs_cols=SETS["all_xs"], regime_cols=SETS["all_reg"])
    return replace(ctx, xs_cols=ic_top(train), regime_cols=SETS["all_reg"])


def enet_fit(c, train, params, scheme, ctx):
    from sklearn.linear_model import LogisticRegression

    cx = _ctx_for(ctx, train, params["fs"])
    cols = [x for x in cx.xs_cols if x in train]
    X = np.nan_to_num(train[cols].to_numpy(dtype=np.float64), nan=0.0)
    y = train["y_cls"].to_numpy(dtype=float)
    ok = np.isfinite(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=float(params["l1"]),
            C=float(params["C"]),
            max_iter=300,
            tol=1e-3,
            random_state=0,
        ).fit(X[ok], y[ok].astype(int), sample_weight=cmod.row_weights(train, scheme, cx)[ok])
    return cmod.Fitted(c.name, params, scheme, m, cols, "logit")


def lgbm_fit(c, train, params, scheme, ctx):
    return cmod._FITTERS["lgbm_clf"](c, train, params, scheme, _ctx_for(ctx, train, params["fs"]))


def et_fit(c, train, params, scheme, ctx):
    return cmod._FITTERS["et"](c, train, {**params, "kind": "et"}, scheme, _ctx_for(ctx, train, params["fs"]))


def candidates() -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    schemes = every["enet_logit"].schemes
    grids = {
        "enet": [{"C": cc, "l1": l1, "fs": fs} for fs in FS for cc in (0.002, 0.01, 0.05, 0.2) for l1 in (0.2, 0.8)],
        "lgbm": [{"leaves": lv, "leaf_frac": lf, "fs": fs} for fs in FS for lv in (7, 15, 31) for lf in (0.005, 0.02)],
        "et": [
            {"leaf_frac": lf, "max_features": mf, "fs": fs} for fs in FS for lf in (0.002, 0.01) for mf in (0.2, 0.5)
        ],
    }
    labels = {
        "enet": ("탄력망 로지스틱(변수 집합·C·L1 비율 선택)", "elastic-net logistic (variable set, C, l1 selected)"),
        "lgbm": ("LightGBM 분류(변수 집합·잎 수·최소 잎 선택)", "LightGBM classifier (variable set, leaves, min leaf)"),
        "et": ("엑스트라 트리(변수 집합·최소 잎·분할 후보 선택)", "extra trees (variable set, min leaf, max features)"),
    }
    fams = {"enet": "linear", "lgbm": "boosting", "et": "bagging"}
    fits = {"enet": enet_fit, "lgbm": lgbm_fit, "et": et_fit}
    out = [every["mom_12_1"], every["factor_ic"]]
    for name in ("enet", "lgbm", "et"):
        key = f"fullopt__{name}"
        cmod._FITTERS[key] = fits[name]
        out.append(cmod.Candidate(name, fams[name], *labels[name], grid=grids[name], schemes=schemes, fitter=key))
    return out


# ------------------------------------------------------------------------------------------------ run


def write_summary(mkt: str) -> str:
    """reports/modeling/<MKT>_fullopt_summary.md from the saved JSON."""
    d = json.loads((OUT / f"{mkt}_fullopt.json").read_text())
    info = d["protocol"]["fullopt"]["info"]
    lines = [
        f"# {mkt} 전체 최적화 요약 (docs/MODELING.md 24)",
        "",
        f"- 입력 수: {info['inputs']}",
        f"- 선정: `{d['selected']}` · 게이트 {'통과' if d['gate']['passed'] else '미달'}",
        "",
        "| 후보 | 외부 AUC [95% CI] | 외부 IC (t) | 홀드아웃 AUC | 문맥별 변수 집합 선택 | 전체 자료로 고른 설정 |",
        "|---|---|---|---|---|---|",
    ]
    for c in d["candidates"]:
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        hm = f"{h['mean']:.4f}" if h.get("mean") is not None else "–"
        cfgs = c.get("configs") or {}
        fsel = [((v or {}).get("params") or {}).get("fs") for v in cfgs.values()]
        fs_count = ", ".join(f"{k} {fsel.count(k)}" for k in FS if fsel.count(k)) or "–"
        fin = cfgs.get("final") or {}
        fin_s = f"{fin.get('params')} · {fin.get('scheme')}" if fin.get("params") else "–"
        ic = c["outer"]["ic"]
        lines.append(
            f"| {c['name']} | {o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] | {ic['mean']:+.4f} ({ic['t']:+.2f}) | "
            f"{hm} | {fs_count} | {fin_s} |"
        )
    lines += ["", "## 선정 모델 중요도 상위(0이 아닌 것)", ""]
    lines += [f"- {k}: {v:.4f}" for k, v in list(d["importances"].items())[:15] if v > 0]
    text = "\n".join(lines) + "\n"
    (OUT / f"{mkt}_fullopt_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    from quantcio.modeling.report import render_markdown

    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["US", "KR"], required=True)
    ap.add_argument("--summary", action="store_true", help="only rewrite the summary from the saved JSON")
    a = ap.parse_args()
    mkt = a.market
    if a.summary:
        print(write_summary(mkt))
        return
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    sel.DIAGNOSTIC_PAIRS = ()
    say(f"{mkt} modeling dataset")
    ds = build_dataset(Market(mkt), settings, cache, progress=say)
    SETS["core_xs"], SETS["core_reg"] = list(ds.xs_cols), list(ds.regime_cols)
    xs, reg, info = (us_blocks if mkt == "US" else kr_blocks)(ds, settings, cache)
    SETS["all_xs"] = list(dict.fromkeys([*SETS["core_xs"], *xs]))
    SETS["all_reg"] = list(dict.fromkeys([*SETS["core_reg"], *reg]))
    ds.xs_cols, ds.regime_cols = SETS["all_xs"], SETS["all_reg"]  # _Data keeps every column
    ds.snapshot = f"{ds.snapshot}|fullopt|{len(SETS['all_xs'])}|{len(SETS['all_reg'])}"
    f = ds.frame
    lab = f[f["labelled"] & f["is_me"]]
    info["inputs"] = {k: len(v) for k, v in SETS.items()}
    info["new_missing_share"] = {c: round(float(lab[c].isna().mean()), 3) for c in [*xs, *reg]}
    say(f"inputs {info['inputs']}")
    res = sel.run_selection(ds, settings, cache, candidates=candidates(), threads=8, progress=say)
    chosen = {
        c["name"]: {ctx: (v or {}).get("params") for ctx, v in (c.get("configs") or {}).items()} for c in res.candidates
    }
    res.protocol["fullopt"] = {"declared": "docs/MODELING.md 24", "info": info, "chosen": chosen}
    stem = OUT / f"{mkt}_fullopt"
    stem.with_suffix(".json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        f"# 종목 선택 모델 선정 — {mkt}", f"# {mkt} 전체 최적화 연구 (서빙 아님) — {mkt}", 1
    )
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    print(write_summary(mkt))
    say(f"selected {res.selected}, gate passed {res.gate['passed']}")


if __name__ == "__main__":
    main()
