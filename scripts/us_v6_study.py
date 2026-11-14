"""US v6 research run (docs/MODELING.md section 20, pre-declared 2026-10-04): the untested parts of an external
improvement proposal - size/sector residual target, sqrt(market-cap) training weights, a missing-operating-income
flag, factor-composite inputs and the proposed LightGBM rank-loss recipe - against the served v3 ``enet_logit``.
Research only: the served model and ``src/quantcio`` are not changed.

    .venv/bin/python scripts/us_v6_study.py

Writes reports/modeling/US_v6.{md,json} and reports/modeling/US_v6_summary.md.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from quantcio.config import PROJECT_ROOT, load_settings
from quantcio.data.cache import DiskCache
from quantcio.modeling import candidates as cmod
from quantcio.modeling import features as feat
from quantcio.modeling import metrics as mt
from quantcio.modeling import selection as sel
from quantcio.modeling.dataset import build_dataset
from quantcio.models import Market

OUT = PROJECT_ROOT / "reports" / "modeling"
FLAG = "x_op_missing"
EXTRA = ("y_cls_res", "y_q10", "y_q10_res", "y_rp_res", "w_cap", "x_log_mcap_raw")
# factor composites: (sign, column) - mean of the available per-date rank-gauss inputs, then rank-gauss again
COMPOSITES = {
    "c_mom": ["mom_3m", "mom_6m", "mom_12_1", "high52", "ma200_dist", "sector_mom_3m"],
    "c_short": ["ret_1w", "ret_1m", "resid_ret_1m", "rsi14", "ma50_dist", "max_ret_1m"],
    "c_value": ["ep", "bp", "sp", "op_yield"],
    "c_quality": ["roe", "op_margin"],
    "c_growth": ["rev_yoy", "op_yoy", "dep"],
    "c_risk": ["vol20", "vol60", "dvol60", "ivol60", "beta"],
    "c_size": ["log_mcap", "-amihud"],
    "c_attention": ["turnover", "volume_trend"],
    "c_filings": ["sec_8k_90", "sec_8k_material_90", "sec_form4_90", "sec_issuance_365"],
    "c_bad_news": ["sec_8k_negative_365", "sec_officer_change_180", "sec_activist_365", "sec_form144_90"],
    "c_insider": ["insider_buy_180", "-insider_sell_180", "insider_breadth_90"],
}
SINGLES = ["x_skew60", "x_seas", "x_beta_fx", "x_beta_oil", "x_beta_rate", "x_beta_credit"]
COMPOSITE_DOCS = {
    "c_mom": ("모멘텀 묶음", "momentum composite"),
    "c_short": ("단기 강세 묶음", "short-term strength composite"),
    "c_value": ("가치 묶음", "value composite"),
    "c_quality": ("질 묶음", "quality composite"),
    "c_growth": ("성장 묶음", "growth composite"),
    "c_risk": ("위험(변동성·베타)묶음", "risk composite"),
    "c_size": ("규모·유동성 묶음", "size / liquidity composite"),
    "c_attention": ("거래 관심 묶음", "trading-attention composite"),
    "c_filings": ("공시 활동 묶음", "filing-activity composite"),
    "c_bad_news": ("악재 공시 묶음", "bad-news filing composite"),
    "c_insider": ("내부자 순매수 묶음", "insider net-buying composite"),
    "op_missing": ("영업이익 결측 표시(금융사 등)", "operating income missing flag"),
}
TREE = {"n_estimators": 300, "learning_rate": 0.03, "colsample_bytree": 0.5, "reg_lambda": 10.0, "subsample": 0.8}
DEPTH_GRID = [{"depth": 3}, {"depth": 4}]
CHANGES = ("enet_flag", "enet_cmp", "enet_res", "enet_cw", "lgbm_rk", "lgbm_rk_res", "lgbm_rk_res_cw", "lgbm_rp_res")


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ data


def residual_target(f: pd.DataFrame) -> pd.Series:
    """Per-date residual of y_e on [1, rank-gauss log market cap, sector dummies] over the labelled rows."""
    out = pd.Series(np.nan, index=f.index)
    lab = f[f["labelled"] & f["y_e"].notna()]
    for _, g in lab.groupby("date"):
        if len(g) < 30:
            continue
        size = g["x_log_mcap"].fillna(0.0).to_numpy(float)
        sec = pd.get_dummies(g["sector_group"].astype(str)).to_numpy(float)
        X = np.column_stack([size, sec])  # sector dummies span the intercept
        y = g["y_e"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.loc[g.index] = y - X @ beta
    return out


def deciles(y: pd.Series, dates: pd.Series) -> pd.Series:
    u = y.groupby(dates).rank(pct=True)
    return np.floor(u * 10 - 1e-9).clip(0, 9).where(y.notna())


def prepare(settings, cache):
    say("US modeling dataset (v3 inputs)")
    ds = build_dataset(Market.US, settings, cache, progress=say)
    f = ds.frame
    ref = list(ds.xs_cols)
    say("v6 labels, weights and inputs")
    f[FLAG] = f["op_margin"].isna().astype(float) if "op_margin" in f else 0.0
    res = residual_target(f)
    lab = f["labelled"].to_numpy()
    med = res.groupby(f["date"]).transform("median")
    f["y_cls_res"] = (res > med).astype(float).where(lab & res.notna())
    f["y_q10"] = deciles(f["y_e"].where(lab), f["date"])
    f["y_q10_res"] = deciles(res, f["date"])
    f["y_rp_res"] = res.groupby(f["date"]).rank(pct=True)
    mc = f["mcap"].where(f["mcap"] > 0)
    w = np.sqrt(mc)
    f["w_cap"] = (w / w.groupby(f["date"]).transform("mean")).fillna(1.0)
    f["x_log_mcap_raw"] = f["x_log_mcap"]
    for name, parts in COMPOSITES.items():
        cols = []
        for p in parts:
            sign, base = (-1.0, p[1:]) if p.startswith("-") else (1.0, p)
            if feat.x_col(base) in f:
                cols.append(sign * f[feat.x_col(base)])
        raw = pd.concat(cols, axis=1).mean(axis=1, skipna=True) if cols else pd.Series(np.nan, index=f.index)
        f[name] = feat.rank_gauss(raw, f["date"])
    cmp_cols = [*COMPOSITES, *[c for c in SINGLES if c in f], FLAG]
    for k, v in COMPOSITE_DOCS.items():
        feat.FEATURE_DOCS[k] = v
    ds.xs_cols = list(dict.fromkeys([*ref, FLAG, *COMPOSITES]))
    ds.snapshot = f"{ds.snapshot}|us_v6"
    info = {
        "ref_inputs": len(ref),
        "op_missing_share": round(float(f.loc[f["labelled"] & f["is_me"], FLAG].mean()), 3),
        "residual_known": round(float(res.notna().mean()), 3),
        "cmp_inputs": cmp_cols,
    }
    return ds, ref, cmp_cols, info


class DataV6(sel._Data):
    """``selection._Data`` that also keeps the v6 labels / weights."""

    def __init__(self, ds):
        super().__init__(ds)
        for c in EXTRA:
            if c in ds.frame:
                self.f[c] = ds.frame.loc[self.f.index, c].to_numpy()


# ------------------------------------------------------------------------------------------------ candidates


def _weights(train, scheme, ctx, cap: bool) -> np.ndarray:
    w = cmod.row_weights(train, scheme, ctx)
    if cap:
        w = w * train["w_cap"].fillna(1.0).to_numpy(float)
        w = w / w.mean() if w.mean() > 0 else np.ones(len(train))
    return w


def enet_fitter(cols: list[str], label: str = "y_cls", cap: bool = False):
    def fit(c, train, params, scheme, ctx):
        from sklearn.linear_model import LogisticRegression

        X = np.nan_to_num(train[cols].to_numpy(dtype=np.float64), nan=0.0)
        y = train[label].to_numpy(dtype=float)
        ok = np.isfinite(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5, C=float(params["C"]), max_iter=300, tol=1e-3
            ).fit(X[ok], y[ok].astype(int), sample_weight=_weights(train, scheme, ctx, cap)[ok])
        return cmod.Fitted(c.name, params, scheme, m, list(cols), "logit")

    return fit


def _tree_kw(params, n, ctx) -> dict:
    d = int(params["depth"])
    return {
        **TREE,
        "max_depth": d,
        "num_leaves": 2**d - 1,
        "min_child_samples": max(50, int(0.01 * n)),
        "subsample_freq": 1,
        "n_jobs": int(ctx.threads),
        "random_state": int(ctx.seed),
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }


def rank_fitter(cols: list[str], rel: str, cap: bool = False):
    def fit(c, train, params, scheme, ctx):
        import lightgbm as lgb

        use = [x for x in [*cols, *ctx.regime_cols] if train[x].notna().any()]
        d = train[np.isfinite(train[rel].to_numpy(dtype=float))].sort_values("date", kind="stable")
        groups = d.groupby("date", sort=True).size().to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = lgb.LGBMRanker(
                objective="lambdarank", lambdarank_truncation_level=10_000, **_tree_kw(params, len(d), ctx)
            )
            m.fit(
                d[use].to_numpy(dtype=np.float64),
                d[rel].to_numpy(dtype=int),
                group=groups,
                sample_weight=_weights(d, scheme, ctx, cap),
            )
        return cmod.Fitted(c.name, params, scheme, m, use, "lgbm")

    return fit


def rp_fitter(cols: list[str], target: str):
    def fit(c, train, params, scheme, ctx):
        import lightgbm as lgb

        use = [x for x in [*cols, *ctx.regime_cols] if train[x].notna().any()]
        y = train[target].to_numpy(dtype=float)
        ok = np.isfinite(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = lgb.LGBMRegressor(objective="regression", **_tree_kw(params, int(ok.sum()), ctx)).fit(
                train[use].to_numpy(dtype=np.float64)[ok], y[ok], sample_weight=_weights(train, scheme, ctx, False)[ok]
            )
        return cmod.Fitted(c.name, params, scheme, m, use, "lgbm")

    return fit


def v6_candidates(ref: list[str], cmp_cols: list[str]) -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    grid_c = every["enet_logit"].grid
    schemes = every["enet_logit"].schemes
    flag = [*ref, FLAG]
    spec = [
        (
            "enet_ref",
            "linear",
            "기준: 서빙 중인 v3 탄력망 로지스틱(입력 59)",
            "reference: served v3 enet_logit",
            grid_c,
            enet_fitter(ref),
        ),
        (
            "enet_flag",
            "linear",
            "탄력망 + 영업이익 결측 표시",
            "enet + op-income missing flag",
            grid_c,
            enet_fitter(flag),
        ),
        (
            "enet_cmp",
            "linear",
            "탄력망 · 팩터 묶음 입력 18개",
            "enet on 18 factor-composite inputs",
            grid_c,
            enet_fitter(cmp_cols),
        ),
        (
            "enet_res",
            "linear",
            "탄력망 · 시가총액·업종 잔차 목표",
            "enet, size/sector residual target",
            grid_c,
            enet_fitter(flag, "y_cls_res"),
        ),
        ("enet_cw", "linear", "탄력망 · √시가총액 가중", "enet, sqrt-cap weights", grid_c, enet_fitter(flag, cap=True)),
        (
            "lgbm_rk",
            "rank",
            "LightGBM 순위(깊이 3~4, 열 0.5, L2 10, 10분위)",
            "LightGBM lambdarank (depth 3-4)",
            DEPTH_GRID,
            rank_fitter(flag, "y_q10"),
        ),
        (
            "lgbm_rk_res",
            "rank",
            "LightGBM 순위 · 잔차 10분위",
            "LightGBM lambdarank, residual deciles",
            DEPTH_GRID,
            rank_fitter(flag, "y_q10_res"),
        ),
        (
            "lgbm_rk_res_cw",
            "rank",
            "제안 조합: 잔차 순위 + LightGBM 순위 손실 + √시가총액 가중",
            "proposed recipe: residual rank + lambdarank + sqrt-cap",
            DEPTH_GRID,
            rank_fitter(flag, "y_q10_res", cap=True),
        ),
        (
            "lgbm_rp_res",
            "tree",
            "LightGBM 회귀 · 잔차 순위 백분위",
            "LightGBM regression on residual rank pct",
            DEPTH_GRID,
            rp_fitter(flag, "y_rp_res"),
        ),
    ]
    out = [every["mom_12_1"], every["factor_ic"]]
    for name, fam, ko, en, grid, fitter in spec:
        key = f"v6__{name}"
        cmod._FITTERS[key] = fitter
        out.append(cmod.Candidate(name, fam, ko, en, grid=list(grid), schemes=schemes, fitter=key))
    return out


def patch_protocol() -> None:
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    sel._Data = DataV6
    sel.DIAGNOSTIC_PAIRS = ()


# ------------------------------------------------------------------------------------------------ analysis


def exposures(oos: pd.DataFrame, frame: pd.DataFrame, names: list[str]) -> dict:
    key = frame.set_index(["date", "entity"])[["x_log_mcap", "x_beta"]]
    o = oos.join(key, on=["date", "entity"])
    o["mrank"] = o.groupby("date")["x_log_mcap"].rank(ascending=False)
    out = {}
    for n in names:
        if n not in o:
            continue
        sz = o.groupby("date").apply(lambda g, n=n: g[n].rank().corr(g["x_log_mcap"].rank())).mean()
        bt = o.groupby("date").apply(lambda g, n=n: g[n].rank().corr(g["x_beta"].rank())).mean()
        top = mt.per_date_primary(o[o["mrank"] <= 100], n).mean()
        out[n] = {"size": round(float(sz), 3), "beta": round(float(bt), 3), "auc_top100": round(float(top), 4)}
    return out


def comparisons(oos: pd.DataFrame, hold: pd.DataFrame) -> dict:
    base_o, base_h = mt.per_date_primary(oos, "enet_ref"), mt.per_date_primary(hold, "enet_ref")
    rows = {}
    for n in CHANGES:
        if n not in oos:
            continue
        a = mt.per_date_primary(oos, n)
        common = a.index.intersection(base_o.index)
        d = (a.reindex(common) - base_o.reindex(common)).dropna().to_numpy(float)
        draws = mt.block_bootstrap_means(d)
        hd = mt.paired_difference(mt.per_date_primary(hold, n), base_h) if n in hold else {}
        rows[n] = {
            "outer": mt.summarize(d, 0.0),
            "p_one_sided": float((draws <= 0).mean()),
            "holdout": hd,
        }
    ps = np.array([r["p_one_sided"] for r in rows.values()])
    order = np.argsort(ps)
    m = len(ps)
    ok = ps[order] <= 0.10 * np.arange(1, m + 1) / m
    passed = set()
    if ok.any():
        passed = {list(rows)[i] for i in order[: int(np.max(np.nonzero(ok)[0])) + 1]}
    for n in rows:
        rows[n]["bh_q10"] = n in passed
    return rows


def main() -> None:
    from quantcio.modeling.report import render_markdown

    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    ds, ref, cmp_cols, info = prepare(settings, cache)
    say(f"v6 inputs: ref {len(ref)}, compact {len(cmp_cols)}, op missing {info['op_missing_share']}")
    patch_protocol()
    cands = v6_candidates(ref, cmp_cols)
    res = sel.run_selection(ds, settings, cache, candidates=cands, threads=8, progress=say)
    names = [c["name"] for c in res.candidates]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmp = comparisons(res.oos, res.holdout_preds)
        expo = {"outer": exposures(res.oos, ds.frame, names), "holdout": exposures(res.holdout_preds, ds.frame, names)}
    res.protocol["us_v6"] = {"declared": "docs/MODELING.md 20", "info": info, "comparisons": cmp, "exposures": expo}
    (OUT / "US_v6.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        "# 종목 선택 모델 선정 — US", "# 미국 v6 연구 — 외부 개선안 시험 (서빙 아님) — US", 1
    )
    (OUT / "US_v6.md").write_text(md, encoding="utf-8")
    lines = [
        "# 미국 v6 요약 (docs/MODELING.md 20)",
        "",
        f"- 선정: `{res.selected}` · 게이트 {'통과' if res.gate['passed'] else '미달'}",
        "",
        "| 후보 | 외부 AUC [95% CI] | 홀드아웃 AUC | 기준 대비 외부 차이 [95% CI] | 단측 p | BH "
        "| 시총 노출(외부/홀드) | 베타 노출(외부/홀드) | 상위100 AUC(외부/홀드) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in res.candidates:
        n = c["name"]
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        cp = cmp.get(n)
        diff = f"{cp['outer']['mean']:+.4f} [{cp['outer']['lo']:+.4f}, {cp['outer']['hi']:+.4f}]" if cp else "–"
        pv = f"{cp['p_one_sided']:.3f}" if cp else "–"
        bh = ("유의" if cp["bh_q10"] else "아님") if cp else "–"
        eo, eh = expo["outer"].get(n, {}), expo["holdout"].get(n, {})
        hm = f"{h['mean']:.4f}" if h.get("mean") is not None else "–"
        lines.append(
            f"| {n} | {o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] | {hm} | {diff} | {pv} | {bh} | "
            f"{eo.get('size', '–')} / {eh.get('size', '–')} | {eo.get('beta', '–')} / {eh.get('beta', '–')} | "
            f"{eo.get('auc_top100', '–')} / {eh.get('auc_top100', '–')} |"
        )
    (OUT / "US_v6_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    say(f"selected {res.selected}, gate passed {res.gate['passed']}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
