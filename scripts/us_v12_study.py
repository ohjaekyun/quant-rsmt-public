"""US v12 research run A (docs/MODELING.md 30.2, pre-declared 2026-10-05): the v9 inputs + block P (power grid, real
estate and data-centre regime values; sector x regime interactions), with diagnostic pairs that drop only block P.
Research only.

    .venv/bin/python scripts/us_v12_study.py

Writes reports/modeling/US_v12.{md,json} and reports/modeling/US_v12_summary.md.
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
import us_v12_data as p  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
BASE = ("cat_rank", "ridge", "logit_w", "lgbm_clf", "et")
PAIRS = ("cat_rank", "logit_w", "lgbm_clf")
INTER = {
    "p_ai_capex": ("ai", "r_hyper_capex_yoy", "AI 인프라 × 하이퍼스케일러 설비투자"),
    "p_ai_load": ("ai", "r_dom_load_yoy", "AI 인프라 × 북버지니아 전력 부하"),
    "p_ai_dc": ("ai", "r_dc_buildout", "AI 인프라 × 데이터센터 입주 증가"),
    "p_util_price": ("util", "r_va_price_yoy", "유틸리티 × 버지니아 상업용 전기요금"),
    "p_re_cre": ("re", "r_cre_yoy", "부동산 × 상업용 부동산 가격"),
    "p_build_cons": ("build", "r_cons_yoy", "건설·소재 × 건설 지출"),
}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _z_daily(s: pd.Series) -> pd.Series:
    m = s.rolling(1095, min_periods=365).mean().shift(1)
    sd = s.rolling(1095, min_periods=365).std().shift(1)
    return ((s - m) / sd.where(sd > 0)).clip(-3, 3)


def add_block_p(ds, settings, cache) -> tuple[list[str], list[str], dict]:
    from quantcio.data.pit.us_cik import Submissions, entity_cik
    from quantcio.data.pit.us_net import PitNet

    f = ds.frame
    dates = pd.DatetimeIndex(f["date"])
    reg = p.all_regime(settings)
    for name in p.REGIME:
        s = reg.get(name)
        f[name] = s.reindex(dates, method="ffill").to_numpy(float) if s is not None and len(s) else np.nan
    subs = Submissions(PitNet(settings))
    sic = {e: str((subs.get(entity_cik(str(e))) or {}).get("sic") or "") for e in f["entity"].unique()}
    s4 = f["entity"].map(sic).fillna("")
    groups = {
        "ai": s4.isin(p.AI_INFRA_SIC) | f["ticker"].astype(str).isin(p.DC_REITS),
        "util": s4.str.startswith("49"),
        "re": f["sector_group"].astype(str).eq("부동산"),
        "build": s4.str[:2].isin(["15", "16", "17", "32"]),
    }
    xs = []
    for name, (grp, r, ko) in INTER.items():
        s = reg.get(r)
        z = (
            _z_daily(s).reindex(dates, method="ffill").to_numpy(float)
            if s is not None and len(s)
            else np.full(len(f), np.nan)
        )
        f[name] = np.where(groups[grp].to_numpy(), np.nan_to_num(z, nan=0.0), 0.0)
        f[feat.x_col(name)] = f[name]
        feat.FEATURE_DOCS[name] = (ko, name)
        xs.append(feat.x_col(name))
    lab = f[f["labelled"] & f["is_me"]]
    info = {
        "group_share": {k: round(float(v[f["is_me"]].mean()), 3) for k, v in groups.items()},
        "regime_missing": {c: round(float(lab[c].isna().mean()), 3) for c in p.REGIME},
        "regime_known_from": {
            c: str(lab.loc[lab[c].notna(), "date"].min().date()) if lab[c].notna().any() else None for c in p.REGIME
        },
    }
    return xs, list(p.REGIME), info


def candidates() -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    out = [every["mom_12_1"], every["factor_ic"], *(every[x] for x in BASE)]
    for name in PAIRS:
        d = copy.copy(every[name])
        d.name, d.diagnostic, d.feature_set = f"{name}_noP", True, "core"
        d.fitter = every[name].fitter or every[name].name
        d.label_ko = every[name].label_ko + " · 진단: 묶음 P 제외"
        out.append(d)
    return out


def write_summary() -> str:
    d = json.loads((OUT / "US_v12.json").read_text())
    info = d["protocol"]["us_v12"]
    lines = [
        "# 미국 v12 요약 (docs/MODELING.md 30.2)",
        "",
        f"- 업종 비중: {info['group_share']} · 국면 시작일: {info['regime_known_from']}",
        f"- 선정 `{d['selected']}` · 게이트 {'통과' if d['gate']['passed'] else '미달'}",
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
    lines += ["", "## 묶음 P의 기여 (같은 모델에서 묶음 P만 뺀 쌍과의 날짜별 AUC 차이)", ""]
    for k, v in (d["protocol"].get("alt_value") or {}).items():
        o, h = v["outer"], v.get("holdout") or {}
        hh = f" · 홀드아웃 {h['mean']:+.4f} [{h['lo']:+.4f}, {h['hi']:+.4f}]" if h.get("mean") is not None else ""
        lines.append(f"- {k}: 외부 {o['mean']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}] (t {o['t']:+.2f}){hh}")
    text = "\n".join(lines) + "\n"
    (OUT / "US_v12_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    from quantcio.modeling.report import render_markdown

    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    if ap.parse_args().summary:
        print(write_summary())
        return
    import us_v9_study as v9

    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    ds, _ = v9.build_inputs(settings, cache)
    base_xs, base_reg = list(ds.xs_cols), list(ds.regime_cols)
    px, preg, info = add_block_p(ds, settings, cache)
    ds.core_xs_cols, ds.core_regime_cols = base_xs, base_reg
    ds.xs_cols, ds.regime_cols = [*base_xs, *px], [*base_reg, *preg]
    ds.snapshot = f"{ds.snapshot}|us_v12|{len(px)}|{len(preg)}"
    say(f"block P {info}")
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    sel.DIAGNOSTIC_PAIRS = tuple((x, f"{x}_noP") for x in PAIRS)
    res = sel.run_selection(ds, settings, cache, candidates=candidates(), threads=8, progress=say)
    res.protocol["us_v12"] = {"declared": "docs/MODELING.md 30.2", **info}
    (OUT / "US_v12.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        "# 종목 선택 모델 선정 — US", "# 미국 v12 연구 — 전력망·부동산·데이터센터 (서빙 아님) — US", 1
    )
    (OUT / "US_v12.md").write_text(md, encoding="utf-8")
    print(write_summary())


if __name__ == "__main__":
    main()
