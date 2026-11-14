"""US v7 research run (docs/MODELING.md section 21, pre-declared 2026-10-04): weather, news sentiment, policy
uncertainty, regulation and extra macro drivers (block H, ``us_v7_data``) in the S&P 500 stock-selection model.
Research only: the served model and ``src/quantcio`` are not changed.

    .venv/bin/python scripts/us_v7_study.py

Writes reports/modeling/US_v7.{md,json} and reports/modeling/US_v7_summary.md.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import us_v7_data as v7  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.data.pit import load_pit_universe  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.dataset import build_dataset  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
BASE = ("enet_logit", "logit_w", "lgbm_clf", "et")
RAW_SCALED = ("us_impact_", "us_sec_")  # already scaled: used as is (clipped), not rank-gauss


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build(settings, cache):
    say("US modeling dataset (v3 inputs)")
    ds = build_dataset(Market.US, settings, cache, progress=say)
    v3_xs, v3_reg = list(ds.xs_cols), list(ds.regime_cols)
    drv = v7.build_drivers(settings, cache, say)
    uni = load_pit_universe(Market.US, settings, cache, 500, date.fromisoformat(settings.backtest.us_start))
    bench = uni.benchmarks["SPY"]["close"].astype(float)
    f = ds.frame
    say("v7 stock features")
    sf = v7.stock_features(f[["date", "entity", "sector_group"]], uni.prices, bench, drv, feat.rank_gauss, say)
    for c in v7.FEATURES:
        f[c] = sf[c].to_numpy(float)
        if c.startswith(RAW_SCALED):
            f[feat.x_col(c)] = f[c].clip(-3, 3)
        else:
            f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
        feat.FEATURE_DOCS[c] = v7.DOCS[c]
    for r in v7.REGIME:
        f[r] = sf[r].to_numpy(float)
    ds.xs_cols = [*v3_xs, *[feat.x_col(c) for c in v7.FEATURES]]
    ds.regime_cols = [*v3_reg, *v7.REGIME]
    ds.core_xs_cols, ds.core_regime_cols = v3_xs, v3_reg  # diagnostics: v3 inputs only (no block H)
    ds.snapshot = f"{ds.snapshot}|us_v7"
    ds.alt_raw = [*list(ds.alt_raw or []), *v7.FEATURES]
    lab = f[f["labelled"] & f["is_me"]]
    cov = {
        "drivers": drv["cover"],
        "missing_share": {c: round(float(lab[c].isna().mean()), 3) for c in [*v7.FEATURES, *v7.REGIME]},
        "known_by_year": lab.groupby(lab["date"].dt.year)[[*v7.FEATURES[:8], "us_sec_reg", "r_agwx_t"]]
        .apply(lambda g: g.notna().mean().round(2))
        .to_dict(orient="index"),
    }
    return ds, cov


def candidates() -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    out = [every["mom_12_1"], every["factor_ic"], *(every[n] for n in BASE)]
    for name in ("logit_w_core", "lgbm_clf_core"):
        d = every[name]
        d.label_ko = d.label_ko.replace("v2 특징만", "날씨·뉴스·정책 묶음 제외")
        d.label_en = d.label_en.replace("v2 inputs only", "without the weather / news / policy block")
        out.append(d)
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
    say(f"v7 missing: {json.dumps(cov['missing_share'])}")
    res = sel.run_selection(ds, settings, cache, candidates=candidates(), threads=8, progress=say)
    res.protocol["us_v7"] = {"declared": "docs/MODELING.md 21", "features": v7.FEATURES, "regime": v7.REGIME, **cov}
    (OUT / "US_v7.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        "# 종목 선택 모델 선정 — US", "# 미국 v7 연구 — 날씨·뉴스·정책·매크로 (서빙 아님) — US", 1
    )
    (OUT / "US_v7.md").write_text(md, encoding="utf-8")
    lines = [
        "# 미국 v7 요약 (docs/MODELING.md 21)",
        "",
        f"- 선정: `{res.selected}` · 게이트 {'통과' if res.gate['passed'] else '미달'}",
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
    lines += ["", "## 묶음 H의 기여 (같은 모델에서 묶음 H만 뺀 쌍과의 날짜별 AUC 차이)", ""]
    for k, v in (res.protocol.get("alt_value") or {}).items():
        o, h = v["outer"], v.get("holdout") or {}
        hh = f" · 홀드아웃 {h['mean']:+.4f} [{h['lo']:+.4f}, {h['hi']:+.4f}]" if h.get("mean") is not None else ""
        lines.append(f"- {k}: 외부 {o['mean']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}] (t {o['t']:+.2f}){hh}")
    block = ("x_us_", *v7.REGIME)
    imp = {k: v for k, v in res.importances.items() if k.startswith(block)}
    lines += ["", "## 선정 모델의 묶음 H 중요도(상위 20 안에 든 것)", ""]
    lines += [f"- {k}: {v:.4f}" for k, v in imp.items()] or ["- 없음"]
    (OUT / "US_v7_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    say(f"selected {res.selected}, gate passed {res.gate['passed']}")


if __name__ == "__main__":
    main()
