"""Subspace-ensemble test (docs/MODELING.md 33, pre-declared 2026-10-09): does averaging three LightGBM regressors,
each trained on a FIXED feature subspace of the US v9 inputs, beat the same learner on all inputs? Research only.

Target: per-date rank percentile (0..1) of the 20-day excess return's residual on size and sector (as us_v6_study's
``y_rp_res``), squared loss. Evaluation: the protocol's within-date AUC on the ORIGINAL excess return.

    .venv/bin/python scripts/subspace_study.py

Writes reports/modeling/US_subspace.{md,json} and reports/modeling/US_subspace_summary.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import us_v5_data as v5  # noqa: E402
import us_v6_study as v6  # noqa: E402
import us_v7_data as v7  # noqa: E402
import us_v8_data as v8  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
TARGET = "y_rp_res"
SUBS = ("sub_A", "sub_B", "sub_C")
# B = trading activity / liquidity, short selling, insider trading, macro sensitivities
B_BASE = {
    "turnover",
    "volume_trend",
    "amihud",
    "insider_buy_180",
    "insider_sell_180",
    "insider_breadth_90",
    "insider_net_90",
    "sec_form4_90",
    "sec_form144_90",
    "beta_fx",
    "beta_oil",
    "beta_rate",
    "beta_credit",
    *v5.FEATURES,
}
B_ENGINEERED = {"e_insider_net", "e_i_short_mom", "e_i_illiq_rev", "e_d_turnover"}
# C = alternative data and filing metadata
C_BASE = {*v7.FEATURES, *v8.FEATURES}
C_PREFIX = ("sec_",)  # SEC filing counts (8-K, material, negative, officer change, activist, issuance)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def group_of(col: str) -> str:
    base = col
    for p in ("x_", "sx_"):
        if base.startswith(p):
            base = base[len(p) :]
            break
    if base in B_BASE or base in B_ENGINEERED:
        return "B"
    if base in C_BASE or base.startswith(C_PREFIX):
        return "C"
    return "A"


class DataSub(sel._Data):
    """``selection._Data`` that also keeps the residual rank-percentile target."""

    def __init__(self, ds):
        super().__init__(ds)
        self.f[TARGET] = ds.frame.loc[self.f.index, TARGET].to_numpy()


SUB_EQ = cmod.Candidate(
    "sub_eq", "ensemble", "서브스페이스 A·B·C 균등 평균", "subspace A/B/C equal-weight rank mean", tuned=False
)
SUB_IC = cmod.Candidate(
    "sub_ic",
    "ensemble",
    "서브스페이스 A·B·C, 내부 CV 순위 IC 비례 가중",
    "subspace A/B/C weighted by inner-CV rank IC",
    tuned=False,
)
_orig_rule = sel.composite_rule


def composite_rule(name, inner, cands):
    if name in (SUB_EQ.name, SUB_IC.name):
        if not all(s in inner for s in SUBS):
            return [], []
        if name == SUB_EQ.name:
            return list(SUBS), [1.0, 1.0, 1.0]
        w = [max(float(np.nan_to_num(inner[s].get("inner_ic", 0.0))), 0.0) for s in SUBS]
        return (list(SUBS), w) if any(w) else (list(SUBS), [1.0, 1.0, 1.0])
    return _orig_rule(name, inner, cands)


def candidates(groups: dict[str, list[str]], all_xs: list[str]) -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    out = [every["mom_12_1"], every["factor_ic"], every["cat_rank"]]
    spec = [("single_all", all_xs, "LightGBM 회귀 · 입력 157개 전체 (단일 기준)", "LightGBM regression, all inputs")]
    for g in ("A", "B", "C"):
        spec.append((f"sub_{g}", groups[g], f"LightGBM 회귀 · 서브스페이스 {g}", f"LightGBM regression, subspace {g}"))
    for name, cols, ko, en in spec:
        key = f"subspace__{name}"
        cmod._FITTERS[key] = v6.rp_fitter(cols, TARGET)
        out.append(cmod.Candidate(name, "tree", ko, en, grid=list(v6.DEPTH_GRID), schemes=("none",), fitter=key))
    return out


def write_summary() -> str:
    d = json.loads((OUT / "US_subspace.json").read_text())
    info = d["protocol"]["subspace"]
    by = {c["name"]: c for c in d["candidates"]}

    def f(x, spec="+.4f"):
        return "–" if x is None else format(x, spec)

    lines = [
        "# 미국 서브스페이스 앙상블 시험 (docs/MODELING.md 33)",
        "",
        f"- 묶음 크기: A {len(info['groups']['A'])} · B {len(info['groups']['B'])} · C {len(info['groups']['C'])} "
        f"(국면 {info['n_regime']}개는 모든 모델 공통) · 선정 `{d['selected']}` · "
        f"게이트 {'통과' if d['gate']['passed'] else '미달'}",
        "",
        "| 후보 | 외부 AUC [95% CI] | 외부 IC (t) | 홀드아웃 AUC |",
        "|---|---|---|---|",
    ]
    for c in d["candidates"]:
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        hm = f"{h['mean']:.4f}" if h.get("mean") is not None else "–"
        lines.append(
            f"| {c['name']} | {o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] | "
            f"{c['outer']['ic']['mean']:+.4f} ({c['outer']['ic']['t']:+.2f}) | {hm} |"
        )
    lines += ["", "## 질문 1: 섞은 것이 단일 모델보다 나은가 (날짜별 AUC 차이)", ""]
    q1 = False
    for k, v in (d["protocol"].get("alt_value") or {}).items():
        o, h = v["outer"], v.get("holdout") or {}
        hh = f" · 홀드아웃 {f(h.get('mean'))} [{f(h.get('lo'))}, {f(h.get('hi'))}]" if h.get("mean") is not None else ""
        lines.append(f"- {k}: 외부 {f(o['mean'])} [{f(o['lo'])}, {f(o['hi'])}] (t {f(o.get('t'), '+.2f')}){hh}")
        if k.split()[0] in (SUB_EQ.name, SUB_IC.name) and k.endswith("single_all"):
            if o["lo"] is not None and o["lo"] > 0 and (h.get("mean") is None or h["mean"] >= 0):
                q1 = True
    lines += [
        "",
        f"**질문 1 판정: {'유효(앙상블 > 단일)' if q1 else '유효하지 않음'}**",
        "",
        "## 질문 2: B·C 단독 실력",
        "",
    ]
    random_level = []
    for s in ("sub_B", "sub_C"):
        o, ic = by[s]["outer"]["auc_median"], by[s]["outer"]["ic"]
        rnd = o["lo"] <= 0.5 and ic["t"] < 2.0
        random_level.append(rnd)
        lines.append(
            f"- {s}: 외부 AUC {o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}], IC t {ic['t']:+.2f} → "
            f"{'무작위 수준' if rnd else '무작위보다 나음'}"
        )
    discard = all(random_level)
    verdict = "B·C 모두 무작위 수준 → 서브스페이스 앙상블 폐기, 단일 모델 정제로 회귀" if discard else "폐기 조건 아님"
    lines += ["", f"**질문 2 판정: {verdict}**"]
    text = "\n".join(lines) + "\n"
    (OUT / "US_subspace_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    import us_v9_study as v9

    from quantcio.modeling.report import render_markdown

    if "--summary" in sys.argv:
        print(write_summary())
        return
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    ds, _ = v9.build_inputs(settings, cache)
    f = ds.frame
    res = v6.residual_target(f)
    f[TARGET] = res.groupby(f["date"]).rank(pct=True)
    all_xs = list(ds.xs_cols)
    groups = {g: [c for c in all_xs if group_of(c) == g] for g in ("A", "B", "C")}
    say(f"groups A {len(groups['A'])} B {len(groups['B'])} C {len(groups['C'])}; regime {len(ds.regime_cols)}")
    sel._Data = DataSub
    sel.composite_rule = composite_rule
    sel.COMPOSITE_CANDIDATES = (SUB_EQ, SUB_IC)
    sel.COMPOSITES = (SUB_EQ.name, SUB_IC.name)
    sel.DIAGNOSTIC_PAIRS = (("sub_eq", "single_all"), ("sub_ic", "single_all"), ("single_all", "cat_rank"))
    ds.snapshot = f"{ds.snapshot}|subspace|{TARGET}"
    out = sel.run_selection(ds, settings, cache, candidates=candidates(groups, all_xs), threads=8, progress=say)
    out.protocol["subspace"] = {
        "declared": "docs/MODELING.md 33",
        "target": TARGET,
        "groups": groups,
        "n_regime": len(ds.regime_cols),
        "residual_known": round(float(res.notna().mean()), 3),
    }
    (OUT / "US_subspace.json").write_text(json.dumps(out.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(out, settings).replace(
        "# 종목 선택 모델 선정 — US", "# 미국 서브스페이스 앙상블 시험 (서빙 아님) — US", 1
    )
    (OUT / "US_subspace.md").write_text(md, encoding="utf-8")
    print(write_summary())


if __name__ == "__main__":
    main()
