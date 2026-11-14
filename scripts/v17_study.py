"""v17 research runs (docs/MODELING.md 43, pre-declared 2026-10-12): block V (legal risk, labour, cyber incidents, web
traffic) per market. Diagnostic pairs drop only that block; the served models' pairs decide. Research only.

    .venv/bin/python scripts/v17_data.py fetch
    .venv/bin/python scripts/v17_study.py --market KR --block V

Writes reports/modeling/<MKT>_v17_V.{md,json} and reports/modeling/<MKT>_v17_V_summary.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v13_study as v13s  # noqa: E402
import v14_study as v14s  # noqa: E402
import v17_data as v  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
BLOCK_KO = {"V": "법적 리스크·노동·해킹·웹 접속"}
REGIME: set[str] = set()


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_block(block: str, market: str, ds, settings, cache) -> tuple[list[str], list[str], dict]:
    f = ds.frame
    df, info = v.block(block, market, f, ds, settings, cache)
    raw, reg = [], []
    for c in df.columns:
        vals = df[c].astype(float)
        if vals.notna().sum() == 0 or vals.nunique() < 2:
            info.setdefault("empty", []).append(c)
            continue
        f[c] = vals.to_numpy()
        if c in REGIME:
            reg.append(c)
        else:
            f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
            raw.append(c)
        feat.FEATURE_DOCS[c] = v.DOCS.get(c, (c, c))
    lab = f[f["labelled"] & f["is_me"]]
    info["missing_share"] = {c: round(float(lab[c].isna().mean()), 3) for c in [*raw, *reg]}
    info["known_from"] = {
        c: str(lab.loc[lab[c].notna(), "date"].min().date()) if lab[c].notna().any() else None for c in [*raw, *reg]
    }
    aucs = {c: v13s.missing_auc(f, c) for c in raw}
    info["missing_auc"] = aucs
    info["leak_suspect"] = sorted(c for c, a in aucs.items() if a is not None and abs(a - 0.5) > v13s.LEAK_AUC)
    return raw, reg, info


def _link_info(info: dict) -> dict:
    keys = ("kr_", "crime", "breach", "cyber")
    return {k: val for k, val in info.items() if k.startswith(keys)}


def write_summary(market: str, block: str) -> str:
    stem = OUT / f"{market}_v17_{block}"
    d = json.loads(stem.with_suffix(".json").read_text())
    info = d["protocol"]["v17"]
    served = v13s.SERVED[market]
    lines = [
        f"# {market} v17 묶음 {block} — {BLOCK_KO[block]} (docs/MODELING.md 43)",
        "",
        f"- 특징 {len(info['features'])}개 · 국면 {len(info['regime'])}개 · 선정 `{d['selected']}` · "
        f"게이트 {'통과' if d['gate']['passed'] else '미달'}",
        f"- 결측 누수 점검(|AUC−0.5| > {v13s.LEAK_AUC}): {', '.join(info.get('leak_suspect') or []) or '없음'}",
        f"- 결측률: {info.get('missing_share')}",
        f"- 연결·수집 정보: {_link_info(info)}",
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
    lines += ["", f"## 묶음 {block}의 기여 (같은 모델에서 묶음 {block}만 뺀 쌍과의 날짜별 AUC 차이)", ""]
    dl, ok = v14s._diff_lines(d["protocol"].get("alt_value"), served)
    lines += dl
    verdict = "기여 있음(43 기준)" if ok else "기여 없음"
    if info.get("leak_suspect"):
        verdict += " — 누수 의심 항목이 있어 채택 판단에서 제외"
    lines += ["", f"**판정(서빙 모델 `{served}` 쌍 기준): {verdict}**"]
    text = "\n".join(lines) + "\n"
    (OUT / f"{market}_v17_{block}_summary.md").write_text(text, encoding="utf-8")
    return text


def run_block(block: str, market: str, base, settings, cache, features_only: bool = False) -> None:
    from quantcio.modeling.report import render_markdown

    ds = copy.copy(base)  # one dataset build serves every block of the market; each block gets a fresh frame
    ds.frame = base.frame.copy()
    base_xs, base_reg = list(base.xs_cols), list(base.regime_cols)
    say(f"block {block} ({BLOCK_KO[block]})")
    raw, reg, info = add_block(block, market, ds, settings, cache)
    say(f"block {block}: {raw} + regime {reg}; {json.dumps(info, ensure_ascii=False, default=str)[:2500]}")
    if features_only:
        return
    if not raw and not reg:
        raise SystemExit(f"묶음 {block}: 자료 없음 — scripts/v17_data.py fetch 먼저")
    ds.core_xs_cols, ds.core_regime_cols = base_xs, base_reg
    ds.xs_cols = [*base_xs, *[feat.x_col(c) for c in raw]]
    ds.regime_cols = [*base_reg, *reg]
    ds.snapshot = f"{base.snapshot}|v17|{block}|{len(raw)}|{len(reg)}"
    ds.alt_raw = [*list(base.alt_raw or []), *raw]
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    sel.DIAGNOSTIC_PAIRS = tuple((x, f"{x}_no{block}") for x in v13s.CANDS[market][1])
    res = sel.run_selection(ds, settings, cache, candidates=v13s.candidates(market, block), threads=8, progress=say)
    res.protocol["v17"] = {"declared": "docs/MODELING.md 43", "block": block, "features": raw, "regime": reg, **info}
    stem = OUT / f"{market}_v17_{block}"
    stem.with_suffix(".json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        f"# 종목 선택 모델 선정 — {market}",
        f"# v17 연구 — 묶음 {block} {BLOCK_KO[block]} (서빙 아님) — {market}",
        1,
    )
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    print(write_summary(market, block), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    ap.add_argument("--block", required=True, help="one block or a comma list, e.g. X,J,B,R")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--features-only", action="store_true", help="build the block and print its info, no selection")
    a = ap.parse_args()
    blocks = a.block.split(",")
    if any(b not in BLOCK_KO for b in blocks):
        raise SystemExit(f"--block: {list(BLOCK_KO)}")
    if a.summary:
        for b in blocks:
            print(write_summary(a.market, b))
        return
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    _, base = v13s.load_inputs(a.market, settings, cache)
    for b in blocks:
        run_block(b, a.market, base, settings, cache, a.features_only)


if __name__ == "__main__":
    main()
