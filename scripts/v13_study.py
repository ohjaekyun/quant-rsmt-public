"""v13 research runs (docs/MODELING.md 31, pre-declared 2026-10-07): one market x one block per run -
T (chart / trend indicators), F (investor flows, institutions, algorithmic trading, pension / government buying),
S (social statistics as regime inputs). Diagnostic pairs drop only that block. Research only.

    .venv/bin/python scripts/v13_data.py fetch --what all      # data for blocks F and S first
    .venv/bin/python scripts/v13_study.py --market KR --block T

Writes reports/modeling/<MKT>_v13_<B>.{md,json} and reports/modeling/<MKT>_v13_<B>_summary.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_data as v  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
CANDS = {  # market -> (base candidates, diagnostic pairs)
    "KR": (("logit_w", "lgbm_clf", "et"), ("et", "logit_w", "lgbm_clf")),
    "US": (("cat_rank", "ridge", "logit_w", "lgbm_clf", "et"), ("cat_rank", "logit_w", "lgbm_clf")),
}
SERVED = {"KR": "et", "US": "cat_rank"}
BLOCK_KO = {"T": "차트·추세 지표", "F": "투자자 수급·기관·알고리즘 매매", "S": "사회 지표"}
LEAK_AUC = 0.02  # |within-date AUC of the missing indicator - 0.5| above this marks a possible leak (31.2)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_inputs(market: str, settings, cache):
    from quantcio.data.pit import load_pit_universe
    from quantcio.modeling.dataset import build_dataset, universe_params

    n, start = universe_params(Market(market), settings)
    uni = load_pit_universe(Market(market), settings, cache, n, start)
    if market == "KR":
        ds = build_dataset(Market.KR, settings, cache, universe=uni, progress=say)
    else:
        import us_v9_study as v9

        ds, _ = v9.build_inputs(settings, cache, universe=uni)
    return uni, ds


def missing_auc(f: pd.DataFrame, col: str) -> float | None:
    """Within-date AUC of 'feature is missing' for the above-median label (labelled month-end rows)."""
    from sklearn.metrics import roc_auc_score

    lab = f.loc[f["labelled"] & f["is_me"], ["date", "y_e", col]].dropna(subset=["y_e"])
    acc, wsum = 0.0, 0
    for _, g in lab.groupby("date"):
        y = (g["y_e"] > g["y_e"].median()).astype(int)
        m = g[col].isna().astype(int)
        if y.nunique() < 2 or m.nunique() < 2:
            continue
        acc += roc_auc_score(y, m) * len(g)
        wsum += len(g)
    return round(acc / wsum, 4) if wsum else None


def add_block(block: str, market: str, ds, uni, settings) -> tuple[list[str], list[str], dict]:
    f = ds.frame
    dates = pd.DatetimeIndex(f["date"])
    info: dict = {}
    reg_vals: dict[str, np.ndarray] = {}
    if block == "T":
        df, docs = v.block_t(f, uni.prices), v.T_DOCS
    elif block == "F" and market == "KR":
        df, info = v.block_f_kr(f, settings)
        docs = v.KR_F_DOCS
    elif block == "F":
        a, i1 = v.block_f_13f(f, settings)
        b, i2 = v.block_f_midas(f, settings)
        df, info, docs = pd.concat([a, b], axis=1), {**i1, **i2}, v.US_F_DOCS
        fed = v.fed_balance_sheet(settings)
        reg_vals["r_fed_bs"] = kd._asof(fed, dates)
        docs = {**docs, "r_fed_bs": ("연준 총자산 13주 log 변화", "Fed total assets, 13-week log change")}
    elif block == "S":
        df, docs = pd.DataFrame(index=f.index), v.S_DOCS
        for name, s in v.social_regime(settings, market).items():
            reg_vals[name] = kd._asof(s, dates)
    else:
        raise ValueError(block)
    raw, reg = [], []
    for c in df.columns:
        vals = df[c].astype(float)
        if vals.notna().sum() == 0:
            info.setdefault("empty", []).append(c)
            continue
        f[c] = vals.to_numpy()
        if c.startswith("r_"):
            reg.append(c)
        else:
            f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
            raw.append(c)
        feat.FEATURE_DOCS[c] = docs.get(c, (c, c))
    for c, vals in reg_vals.items():
        if np.isfinite(vals).any():
            f[c] = vals
            reg.append(c)
            feat.FEATURE_DOCS[c] = docs.get(c, (c, c))
    lab = f[f["labelled"] & f["is_me"]]
    info["missing_share"] = {c: round(float(lab[c].isna().mean()), 3) for c in [*raw, *reg]}
    info["known_from"] = {
        c: str(lab.loc[lab[c].notna(), "date"].min().date()) if lab[c].notna().any() else None for c in [*raw, *reg]
    }
    if block == "F":
        aucs = {c: missing_auc(f, c) for c in raw}
        info["missing_auc"] = aucs
        info["leak_suspect"] = sorted(c for c, a in aucs.items() if a is not None and abs(a - 0.5) > LEAK_AUC)
    return raw, reg, info


def candidates(market: str, block: str) -> list[cmod.Candidate]:
    base, pairs = CANDS[market]
    every = {c.name: c for c in cmod.default_candidates()}
    out = [every["mom_12_1"], every["factor_ic"], *(every[x] for x in base)]
    for name in pairs:
        d = copy.copy(every[name])
        d.name, d.diagnostic, d.feature_set = f"{name}_no{block}", True, "core"
        d.fitter = every[name].fitter or every[name].name
        d.label_ko = every[name].label_ko + f" · 진단: 묶음 {block} 제외"
        out.append(d)
    return out


def write_summary(market: str, block: str) -> str:
    stem = OUT / f"{market}_v13_{block}"
    d = json.loads(stem.with_suffix(".json").read_text())
    info = d["protocol"]["v13"]
    served = SERVED[market]
    lines = [
        f"# {market} v13 묶음 {block} — {BLOCK_KO[block]} (docs/MODELING.md 31)",
        "",
        f"- 특징 {len(info['features'])}개 · 국면 {len(info['regime'])}개 · 선정 `{d['selected']}` · "
        f"게이트 {'통과' if d['gate']['passed'] else '미달'}",
    ]
    if info.get("empty"):
        lines.append(f"- 자료가 없어 빠진 항목: {', '.join(info['empty'])}")
    if block == "F":
        lines.append(f"- 결측 누수 점검(|AUC−0.5| > {LEAK_AUC}): {', '.join(info.get('leak_suspect') or []) or '없음'}")
    lines += ["", "| 후보 | 외부 AUC [95% CI] | 외부 IC (t) | 홀드아웃 AUC |", "|---|---|---|---|"]
    for c in d["candidates"]:
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        hm = f"{h['mean']:.4f}" if h.get("mean") is not None else "–"
        lines.append(
            f"| {c['name']} | {o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] | "
            f"{c['outer']['ic']['mean']:+.4f} ({c['outer']['ic']['t']:+.2f}) | {hm} |"
        )
    lines += ["", f"## 묶음 {block}의 기여 (같은 모델에서 묶음 {block}만 뺀 쌍과의 날짜별 AUC 차이)", ""]
    verdict = "기여 없음"

    def fmt(x, spec: str = "+.4f") -> str:
        return "–" if x is None else format(x, spec)

    for k, val in (d["protocol"].get("alt_value") or {}).items():
        o, h = val["outer"], val.get("holdout") or {}
        hh = (
            f" · 홀드아웃 {fmt(h['mean'])} [{fmt(h.get('lo'))}, {fmt(h.get('hi'))}]"
            if h.get("mean") is not None
            else ""
        )
        # t is None when the two models score identically (e.g. a linear model ignores regime-only inputs)
        lines.append(f"- {k}: 외부 {fmt(o['mean'])} [{fmt(o['lo'])}, {fmt(o['hi'])}] (t {fmt(o.get('t'), '+.2f')}){hh}")
        if k.split()[0] == served:  # keys read "<model> - <model>_no<B>"
            if o["lo"] > 0 and (h.get("mean") is None or h["mean"] >= 0):
                verdict = "기여 있음(31.4 기준)"
    if block == "F" and info.get("leak_suspect"):
        verdict += " — 누수 의심 항목이 있어 채택 판단에서 제외"
    lines += ["", f"**판정(서빙 모델 `{served}` 쌍 기준): {verdict}**"]
    text = "\n".join(lines) + "\n"
    (OUT / f"{market}_v13_{block}_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    from quantcio.modeling.report import render_markdown

    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    ap.add_argument("--block", choices=["T", "F", "S"], required=True)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        print(write_summary(a.market, a.block))
        return
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    uni, ds = load_inputs(a.market, settings, cache)
    base_xs, base_reg = list(ds.xs_cols), list(ds.regime_cols)
    say(f"block {a.block}")
    raw, reg, info = add_block(a.block, a.market, ds, uni, settings)
    if not raw and not reg:
        raise SystemExit(f"묶음 {a.block}: 자료 없음 — scripts/v13_data.py fetch 먼저")
    ds.core_xs_cols, ds.core_regime_cols = base_xs, base_reg
    ds.xs_cols = [*base_xs, *[feat.x_col(c) for c in raw]]
    ds.regime_cols = [*base_reg, *reg]
    ds.snapshot = f"{ds.snapshot}|v13|{a.block}|{len(raw)}|{len(reg)}"
    ds.alt_raw = [*list(ds.alt_raw or []), *raw]
    say(f"block {a.block}: {len(raw)} features {raw} + regime {reg}; {json.dumps(info, ensure_ascii=False)[:1500]}")
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    sel.DIAGNOSTIC_PAIRS = tuple((x, f"{x}_no{a.block}") for x in CANDS[a.market][1])
    res = sel.run_selection(ds, settings, cache, candidates=candidates(a.market, a.block), threads=8, progress=say)
    res.protocol["v13"] = {"declared": "docs/MODELING.md 31", "block": a.block, "features": raw, "regime": reg, **info}
    stem = OUT / f"{a.market}_v13_{a.block}"
    stem.with_suffix(".json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        f"# 종목 선택 모델 선정 — {a.market}",
        f"# v13 연구 — 묶음 {a.block} {BLOCK_KO[a.block]} (서빙 아님) — {a.market}",
        1,
    )
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    print(write_summary(a.market, a.block))


if __name__ == "__main__":
    main()
