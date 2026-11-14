"""v15 research run (docs/MODELING.md 38, pre-declared 2026-10-10): block L = company headquarters location features
(local peers' returns, distance to the financial centre, relocations, US state economy, KR capital area). Diagnostic
pairs drop only block L; the KR / US served models' pairs decide. Research only.

    .venv/bin/python scripts/v15_location.py fetch
    .venv/bin/python scripts/location_study.py --market KR

Writes reports/modeling/<MKT>_v15_L.{md,json} and reports/modeling/<MKT>_v15_L_summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v13_study as v13s  # noqa: E402
import v14_study as v14s  # noqa: E402
import v15_location as v  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
BLOCK = "L"


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_block(market: str, ds, settings, cache) -> tuple[list[str], list[str], dict]:
    f = ds.frame
    df, info = v.block_l_kr(f, settings) if market == "KR" else v.block_l_us(f, ds, settings, cache)
    raw = []
    for c in df.columns:
        vals = df[c].astype(float)
        if vals.notna().sum() == 0 or vals.nunique() < 2:
            info.setdefault("empty", []).append(c)
            continue
        f[c] = vals.to_numpy()
        f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
        feat.FEATURE_DOCS[c] = v.L_DOCS.get(c, (c, c))
        raw.append(c)
    lab = f[f["labelled"] & f["is_me"]]
    info["missing_share"] = {c: round(float(lab[c].isna().mean()), 3) for c in raw}
    aucs = {c: v13s.missing_auc(f, c) for c in raw}
    info["missing_auc"] = aucs
    info["leak_suspect"] = sorted(c for c, a in aucs.items() if a is not None and abs(a - 0.5) > v13s.LEAK_AUC)
    return raw, [], info


def write_summary(market: str) -> str:
    stem = OUT / f"{market}_v15_{BLOCK}"
    d = json.loads(stem.with_suffix(".json").read_text())
    info = d["protocol"]["v15"]
    served = v13s.SERVED[market]
    lines = [
        f"# {market} v15 묶음 L — 본사 위치 (docs/MODELING.md 38)",
        "",
        f"- 특징 {len(info['features'])}개 · 선정 `{d['selected']}` · "
        f"게이트 {'통과' if d['gate']['passed'] else '미달'}",
        f"- 결측 누수 점검(|AUC−0.5| > {v13s.LEAK_AUC}): {', '.join(info.get('leak_suspect') or []) or '없음'}",
        f"- 결측률: {info.get('missing_share')}",
    ]
    if info.get("empty"):
        lines.append(f"- 자료가 없거나 값이 하나뿐이라 빠진 항목: {', '.join(info['empty'])}")
    lines += ["", "| 후보 | 외부 AUC [95% CI] | 외부 IC (t) | 홀드아웃 AUC |", "|---|---|---|---|"]
    for c in d["candidates"]:
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        hm = f"{h['mean']:.4f}" if h.get("mean") is not None else "–"
        lines.append(
            f"| {c['name']} | {o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] | "
            f"{c['outer']['ic']['mean']:+.4f} ({c['outer']['ic']['t']:+.2f}) | {hm} |"
        )
    lines += ["", "## 묶음 L의 기여 (같은 모델에서 묶음 L만 뺀 쌍과의 날짜별 AUC 차이)", ""]
    dl, ok = v14s._diff_lines(d["protocol"].get("alt_value"), served)
    lines += dl
    verdict = "기여 있음(38절 기준)" if ok else "기여 없음"
    if info.get("leak_suspect"):
        verdict += " — 누수 의심 항목이 있어 채택 판단에서 제외"
    lines += ["", f"**판정(서빙 모델 `{served}` 쌍 기준): {verdict}**"]
    text = "\n".join(lines) + "\n"
    (OUT / f"{market}_v15_{BLOCK}_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    from quantcio.modeling.report import render_markdown

    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        print(write_summary(a.market))
        return
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    _, ds = v13s.load_inputs(a.market, settings, cache)
    base_xs, base_reg = list(ds.xs_cols), list(ds.regime_cols)
    say("block L (HQ location)")
    raw, _, info = add_block(a.market, ds, settings, cache)
    ds.core_xs_cols, ds.core_regime_cols = base_xs, base_reg
    ds.xs_cols = [*base_xs, *[feat.x_col(c) for c in raw]]
    ds.snapshot = f"{ds.snapshot}|v15|L|{len(raw)}"
    ds.alt_raw = [*list(ds.alt_raw or []), *raw]
    say(f"block L: {raw}; {json.dumps(info, ensure_ascii=False)[:1500]}")
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    sel.DIAGNOSTIC_PAIRS = tuple((x, f"{x}_no{BLOCK}") for x in v13s.CANDS[a.market][1])
    res = sel.run_selection(ds, settings, cache, candidates=v13s.candidates(a.market, BLOCK), threads=8, progress=say)
    res.protocol["v15"] = {"declared": "docs/MODELING.md 38", "block": BLOCK, "features": raw, **info}
    stem = OUT / f"{a.market}_v15_{BLOCK}"
    stem.with_suffix(".json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        f"# 종목 선택 모델 선정 — {a.market}", f"# v15 연구 — 묶음 L 본사 위치 (서빙 아님) — {a.market}", 1
    )
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    print(write_summary(a.market))


if __name__ == "__main__":
    main()
