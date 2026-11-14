"""US v8 research run (docs/MODELING.md section 22, pre-declared 2026-10-04): R&D, employee growth (10-K text),
H-1B initial approvals and JOLTS sector labour flows (block J, ``us_v8_data``) in the S&P 500 stock-selection model.
Research only: the served model and ``src/quantcio`` are not changed. Run ``us_v8_data.py`` first (downloads).

    .venv/bin/python scripts/us_v8_study.py

Writes reports/modeling/US_v8.{md,json} and reports/modeling/US_v8_summary.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import us_v8_data as v8  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.dataset import build_dataset  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
BASE = ("enet_logit", "logit_w", "lgbm_clf", "et")
RAW = ("us_rd_none", "us_emp_none", "us_sec_hires", "us_sec_quits", "us_sec_layoffs")  # flags / sector rates: as is


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build(settings, cache):
    say("US modeling dataset (v3 inputs)")
    ds = build_dataset(Market.US, settings, cache, progress=say)
    v3_xs, v3_reg = list(ds.xs_cols), list(ds.regime_cols)
    d = v8.load_all(ds, settings, cache, say)
    f = ds.frame
    say("v8 stock features")
    sf = v8.stock_features(
        f[["date", "entity", "mcap", "sector_group"]], d["chains"], d["rd"], d["emp"], d["h1b"], d["jolts"]
    )
    for c in v8.FEATURES:
        f[c] = sf[c].to_numpy(float)
        f[feat.x_col(c)] = f[c].clip(-3, 3) if c in RAW else feat.rank_gauss(f[c], f["date"])
        feat.FEATURE_DOCS[c] = v8.DOCS[c]
    ds.xs_cols = [*v3_xs, *[feat.x_col(c) for c in v8.FEATURES]]
    ds.core_xs_cols, ds.core_regime_cols = v3_xs, v3_reg  # diagnostics: v3 inputs only (no block J)
    ds.snapshot = f"{ds.snapshot}|us_v8"
    ds.alt_raw = [*list(ds.alt_raw or []), *v8.FEATURES]
    lab = f[f["labelled"] & f["is_me"]]
    docs = [r for v in d["emp"].values() for r in v]
    cov = {
        "missing_share": {c: round(float(lab[c].isna().mean()), 3) for c in v8.FEATURES},
        "known_by_year": lab.groupby(lab["date"].dt.year)[v8.FEATURES]
        .apply(lambda g: g.notna().mean().round(2))
        .to_dict(orient="index"),
        "rd_none_share": round(float(lab["us_rd_none"].mean()), 3),
        "ten_k_docs": len(docs),
        "ten_k_employee_found": round(sum(1 for r in docs if r.get("emp")) / max(len(docs), 1), 3),
        "h1b_matched_ciks": int(d["h1b"]["cik"].nunique()) if len(d["h1b"]) else 0,
        "ciks": len({c for cs in d["chains"].values() for c in cs}),
    }
    return ds, cov


def candidates() -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    out = [every["mom_12_1"], every["factor_ic"], *(every[n] for n in BASE)]
    for name in ("logit_w_core", "lgbm_clf_core"):
        dg = every[name]
        dg.label_ko = dg.label_ko.replace("v2 특징만", "R&D·인력·채용 묶음 제외")
        dg.label_en = dg.label_en.replace("v2 inputs only", "without the R&D / workforce block")
        out.append(dg)
    return out


def patch_protocol() -> None:
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    sel.DIAGNOSTIC_PAIRS = (("logit_w", "logit_w_core"), ("lgbm_clf", "lgbm_clf_core"))


def main() -> None:
    from quantcio.modeling.report import render_markdown

    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    patch_protocol()
    ds, cov = build(settings, cache)
    say(f"v8 coverage: {json.dumps({k: v for k, v in cov.items() if k != 'known_by_year'})}")
    res = sel.run_selection(ds, settings, cache, candidates=candidates(), threads=8, progress=say)
    res.protocol["us_v8"] = {"declared": "docs/MODELING.md 22", "features": v8.FEATURES, **cov}
    (OUT / "US_v8.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        "# 종목 선택 모델 선정 — US", "# 미국 v8 연구 — R&D·인력·전문직 채용·업종 고용 흐름 (서빙 아님) — US", 1
    )
    (OUT / "US_v8.md").write_text(md, encoding="utf-8")
    lines = [
        "# 미국 v8 요약 (docs/MODELING.md 22)",
        "",
        f"- 선정: `{res.selected}` · 게이트 {'통과' if res.gate['passed'] else '미달'}",
        f"- 10-K {cov['ten_k_docs']}건 중 직원 수 추출 {cov['ten_k_employee_found']:.0%}, "
        f"H-1B 매칭 CIK {cov['h1b_matched_ciks']}/{cov['ciks']}, R&D 없음 비율 {cov['rd_none_share']:.0%}",
        "",
        "| 후보 | 외부 AUC [95% CI] | 외부 IC (t) | 홀드아웃 AUC |",
        "|---|---|---|---|",
    ]
    for c in res.candidates:
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        hm = f"{h['mean']:.4f}" if h.get("mean") is not None else "–"
        ic = c["outer"]["ic"]
        lines.append(
            f"| {c['name']} | {o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] | "
            f"{ic['mean']:+.4f} ({ic['t']:+.2f}) | {hm} |"
        )
    lines += ["", "## 묶음 J의 기여 (같은 모델에서 묶음 J만 뺀 쌍과의 날짜별 AUC 차이)", ""]
    for k, v in (res.protocol.get("alt_value") or {}).items():
        o, h = v["outer"], v.get("holdout") or {}
        hh = f" · 홀드아웃 {h['mean']:+.4f} [{h['lo']:+.4f}, {h['hi']:+.4f}]" if h.get("mean") is not None else ""
        lines.append(f"- {k}: 외부 {o['mean']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}] (t {o['t']:+.2f}){hh}")
    imp = {k: v for k, v in res.importances.items() if k.startswith("x_us_")}
    lines += ["", "## 선정 모델의 묶음 J 중요도(상위 20 안에 든 것)", ""]
    lines += [f"- {k}: {v:.4f}" for k, v in imp.items()] or ["- 없음"]
    (OUT / "US_v8_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    say(f"selected {res.selected}, gate passed {res.gate['passed']}")


if __name__ == "__main__":
    main()
