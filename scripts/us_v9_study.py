"""US v9 research run (docs/MODELING.md section 25, pre-declared 2026-10-05): every candidate model of the v3
protocol (linear, boosting, ranking, bagging, CatBoost, MLP, RSMT-Q v3 stacked, five composites) plus variable-set
search candidates, on the v3 inputs + every research block + the engineered block E (``us_v9_features``).
Research only: the served model and ``src/quantcio`` are not changed.

    .venv/bin/python scripts/us_v9_study.py            # full run
    .venv/bin/python scripts/us_v9_study.py --summary  # rewrite the summary from the saved JSON

Writes reports/modeling/US_v9.{md,json} and reports/modeling/US_v9_summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_opt_study as fo  # noqa: E402
import us_v9_features as v9f  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.dataset import build_dataset  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
FS = ("core", "core_eng", "all", "ic40")
IC_TOP = 40
SETS: dict[str, list[str]] = {}
_IC: dict[tuple, list[str]] = {}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ic_top(train: pd.DataFrame) -> list[str]:
    me = train[train["is_me"]]
    key = (str(me["date"].min()), str(me["date"].max()), len(me))
    if key not in _IC:
        fo.SETS["all_xs"] = SETS["all_xs"]
        old = fo.IC_TOP
        fo.IC_TOP = IC_TOP
        fo._IC_MEMO.clear()
        _IC[key] = fo.ic_top(train)
        fo.IC_TOP = old
    return _IC[key]


def _ctx(ctx, train, fs: str):
    if fs == "core":
        return replace(ctx, xs_cols=SETS["core_xs"], regime_cols=SETS["core_reg"])
    if fs == "core_eng":
        return replace(ctx, xs_cols=SETS["core_eng_xs"], regime_cols=SETS["core_reg"])
    if fs == "all":
        return replace(ctx, xs_cols=SETS["all_xs"], regime_cols=SETS["all_reg"])
    return replace(ctx, xs_cols=ic_top(train), regime_cols=SETS["all_reg"])


def fs_enet(c, train, params, scheme, ctx):
    from sklearn.linear_model import LogisticRegression

    cx = _ctx(ctx, train, params["fs"])
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


def fs_lgbm(c, train, params, scheme, ctx):
    return cmod._FITTERS["lgbm_clf"](c, train, params, scheme, _ctx(ctx, train, params["fs"]))


def fs_et(c, train, params, scheme, ctx):
    return cmod._FITTERS["et"](c, train, {**params, "kind": "et"}, scheme, _ctx(ctx, train, params["fs"]))


def candidates() -> list[cmod.Candidate]:
    every = cmod.default_candidates()
    schemes = next(c for c in every if c.name == "enet_logit").schemes
    extra = [
        (
            "fs_enet",
            "linear",
            "탄력망(변수 집합·C·L1 선택)",
            "elastic net (variable set, C, l1)",
            [{"C": cc, "l1": l1, "fs": fs} for fs in FS for cc in (0.002, 0.01, 0.05) for l1 in (0.2, 0.8)],
            fs_enet,
        ),
        (
            "fs_lgbm",
            "boosting",
            "LightGBM 분류(변수 집합·잎 선택)",
            "LightGBM classifier (variable set, leaves)",
            [{"leaves": lv, "leaf_frac": lf, "fs": fs} for fs in FS for lv in (7, 31) for lf in (0.005, 0.02)],
            fs_lgbm,
        ),
        (
            "fs_et",
            "bagging",
            "엑스트라 트리(변수 집합·잎·분할 선택)",
            "extra trees (variable set, leaf, max features)",
            [{"leaf_frac": lf, "max_features": mf, "fs": fs} for fs in FS for lf in (0.002, 0.01) for mf in (0.2, 0.5)],
            fs_et,
        ),
    ]
    out = list(every)
    for name, fam, ko, en, grid, fit in extra:
        cmod._FITTERS[f"v9__{name}"] = fit
        out.append(cmod.Candidate(name, fam, ko, en, grid=grid, schemes=schemes, fitter=f"v9__{name}"))
    for d in out:
        if d.name in ("logit_w_core", "lgbm_clf_core"):
            d.label_ko = d.label_ko.replace("v2 특징만", "v3 입력만(새 입력 전체 제외)")
    return out


def write_summary() -> str:
    d = json.loads((OUT / "US_v9.json").read_text())
    info = d["protocol"]["us_v9"]
    rows = []
    for c in d["candidates"]:
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        fin = (c.get("configs") or {}).get("final") or {}
        rows.append((o["mean"], c, h.get("mean"), fin))
    rows.sort(key=lambda r: -r[0])
    lines = [
        "# 미국 v9 요약 (docs/MODELING.md 25)",
        "",
        f"- 입력: {info['inputs']} · 묶음 E {len(info['engineered'])}개",
        f"- 후보 {len(rows)}개 · 선정 `{d['selected']}` · 게이트 {'통과' if d['gate']['passed'] else '미달'}",
        "",
        "| 순위 | 후보 | 계열 | 외부 AUC [95% CI] | 외부 IC (t) | 홀드아웃 AUC | 전체 자료로 고른 설정 |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, (m, c, hm, fin) in enumerate(rows, 1):
        o, ic = c["outer"]["auc_median"], c["outer"]["ic"]
        hs = f"{hm:.4f}" if hm is not None else "–"
        fs = (
            f"{fin.get('params')} · {fin.get('scheme')}"
            if fin.get("params")
            else ("–" if not fin else str(fin.get("members") or "–"))
        )
        lines.append(
            f"| {i} | {c['name']}{' (진단)' if c.get('diagnostic') else ''} | {c['family']} | "
            f"{m:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] | "
            f"{ic['mean']:+.4f} ({ic['t']:+.2f}) | {hs} | {fs} |"
        )
    lines += ["", "## 새 입력 전체(연구 묶음 + 묶음 E)의 기여 — 같은 모델에서 v3 입력만 쓴 쌍과의 날짜별 AUC 차이", ""]
    for k, v in (d["protocol"].get("alt_value") or {}).items():
        o, h = v["outer"], v.get("holdout") or {}
        hh = f" · 홀드아웃 {h['mean']:+.4f}" if h.get("mean") is not None else ""
        lines.append(f"- {k}: 외부 {o['mean']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}] (t {o['t']:+.2f}){hh}")
    lines += ["", "## 선정 모델 중요도 상위 20", ""]
    lines += [f"- {k}: {v:.4f}" for k, v in list(d["importances"].items())[:20]]
    text = "\n".join(lines) + "\n"
    (OUT / "US_v9_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    from quantcio.modeling.report import render_markdown

    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    if ap.parse_args().summary:
        print(write_summary())
        return
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    ds, info = build_inputs(settings, cache)
    res = sel.run_selection(ds, settings, cache, candidates=candidates(), threads=8, progress=say)
    res.protocol["us_v9"] = info
    (OUT / "US_v9.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        "# 종목 선택 모델 선정 — US", "# 미국 v9 연구 — 모델 전체 + 특징 공학 (서빙 아님) — US", 1
    )
    (OUT / "US_v9.md").write_text(md, encoding="utf-8")
    print(write_summary())
    say(f"selected {res.selected}, gate passed {res.gate['passed']}")


def build_inputs(settings, cache, universe=None):
    """The v9 modeling dataset: v3 inputs + every research block + block E (SETS filled). Returns (ds, info)."""
    say("US modeling dataset")
    ds = build_dataset(
        Market.US, settings, cache, progress=say, **({"universe": universe} if universe is not None else {})
    )
    SETS["core_xs"], SETS["core_reg"] = list(ds.xs_cols), list(ds.regime_cols)
    bx, breg, binfo = fo.us_blocks(ds, settings, cache)
    say("block E (feature engineering)")
    ranked, prod, docs = v9f.engineer(ds.frame)
    for k, v in docs.items():
        feat.FEATURE_DOCS[k] = v
    ex = [feat.x_col(c) for c in [*ranked, *prod]]
    SETS["core_eng_xs"] = list(dict.fromkeys([*SETS["core_xs"], *ex]))
    SETS["all_xs"] = list(dict.fromkeys([*SETS["core_xs"], *bx, *ex]))
    SETS["all_reg"] = list(dict.fromkeys([*SETS["core_reg"], *breg]))
    ds.xs_cols, ds.regime_cols = SETS["all_xs"], SETS["all_reg"]
    ds.core_xs_cols, ds.core_regime_cols = SETS["core_xs"], SETS["core_reg"]
    ds.snapshot = f"{ds.snapshot}|us_v9|{len(SETS['all_xs'])}"
    f = ds.frame
    lab = f[f["labelled"] & f["is_me"]]
    info = {
        "inputs": {k: len(v) for k, v in SETS.items()},
        "engineered": [*ranked, *prod],
        "engineered_missing": {c: round(float(lab[c].isna().mean()), 3) for c in [*ranked, *prod]},
        "blocks": binfo,
    }
    say(f"inputs {info['inputs']}; engineered {len(info['engineered'])}")
    return ds, info


if __name__ == "__main__":
    main()
