"""v14 research runs (docs/MODELING.md 32, pre-declared 2026-10-08): one market x one block per run - C (precious
metals, crypto, agricultural futures: prices and volumes), D (dividends), H (holidays). Diagnostic pairs drop only that
block. Research only.

    .venv/bin/python scripts/v14_data.py fetch --what all
    .venv/bin/python scripts/v14_study.py --market KR --block D

Writes reports/modeling/<MKT>_v14_<B>.{md,json} and reports/modeling/<MKT>_v14_<B>_summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_study as v13s  # noqa: E402
import v14_data as v  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
SERVED = v13s.SERVED
CANDS = v13s.CANDS
BLOCK_KO = {"C": "귀금속·코인·농산물", "D": "배당", "H": "휴일"}
LEAK_AUC = v13s.LEAK_AUC
GUARD_MONTHS = (11, 12)  # KR ex-dividend season inside the 20-session label window (32.2)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_block(block: str, market: str, ds, uni, settings, cache) -> tuple[list[str], list[str], dict]:
    f = ds.frame
    dates = pd.DatetimeIndex(f["date"])
    info: dict = {}
    if block == "C":
        bench_name = "KOSPI" if market == "KR" else "SPY"
        bench = uni.benchmarks[bench_name]["close"].astype(float)
        df = v.block_c(f, uni.prices, bench, market, settings)
        for name, s in v.cmdty_regime(settings).items():
            df[name] = kd._asof(s, dates)
        docs = v.C_DOCS
    elif block == "D" and market == "KR":
        df, info = v.block_d_kr(f, settings)
        docs = v.D_DOCS
    elif block == "D":
        df, info = v.block_d_us(f, ds, settings, cache)
        docs = v.D_DOCS
    elif block == "H":
        df, docs = v.block_h(f, ds.calendar, market), v.H_DOCS
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
    lab = f[f["labelled"] & f["is_me"]]
    info["missing_share"] = {c: round(float(lab[c].isna().mean()), 3) for c in [*raw, *reg]}
    info["known_from"] = {
        c: str(lab.loc[lab[c].notna(), "date"].min().date()) if lab[c].notna().any() else None for c in [*raw, *reg]
    }
    if block in ("C", "D"):
        aucs = {c: v13s.missing_auc(f, c) for c in raw}
        info["missing_auc"] = aucs
        info["leak_suspect"] = sorted(c for c, a in aucs.items() if a is not None and abs(a - 0.5) > LEAK_AUC)
    return raw, reg, info


def guard(res, market: str, block: str) -> dict | None:
    """KR block D: the pair differences on month-ends outside November / December (32.2)."""
    if not (market == "KR" and block == "D") or res.oos is None:
        return None
    out = {}
    for full in CANDS[market][1]:
        core = f"{full}_no{block}"
        o = res.oos[~pd.DatetimeIndex(res.oos["date"]).month.isin(GUARD_MONTHS)]
        h = res.holdout_preds
        h = h[~pd.DatetimeIndex(h["date"]).month.isin(GUARD_MONTHS)] if h is not None else None
        if full in o and core in o:
            out[f"{full} - {core}"] = {
                "outer": mt.paired_difference(mt.per_date_primary(o, full), mt.per_date_primary(o, core)),
                "holdout": (
                    mt.paired_difference(mt.per_date_primary(h, full), mt.per_date_primary(h, core))
                    if h is not None and full in h and core in h
                    else None
                ),
            }
    return out


def _fmt(x, spec: str = "+.4f") -> str:
    return "–" if x is None else format(x, spec)


def _diff_lines(av: dict, served: str) -> tuple[list[str], bool]:
    lines, ok = [], False
    for k, val in (av or {}).items():
        o, h = val["outer"], val.get("holdout") or {}
        hh = (
            f" · 홀드아웃 {_fmt(h.get('mean'))} [{_fmt(h.get('lo'))}, {_fmt(h.get('hi'))}]"
            if h.get("mean") is not None
            else ""
        )
        lines.append(
            f"- {k}: 외부 {_fmt(o['mean'])} [{_fmt(o['lo'])}, {_fmt(o['hi'])}] (t {_fmt(o.get('t'), '+.2f')}){hh}"
        )
        if k.split()[0] == served and o["lo"] is not None and o["lo"] > 0 and (h.get("mean") is None or h["mean"] >= 0):
            ok = True
    return lines, ok


def write_summary(market: str, block: str) -> str:
    stem = OUT / f"{market}_v14_{block}"
    d = json.loads(stem.with_suffix(".json").read_text())
    info = d["protocol"]["v14"]
    served = SERVED[market]
    lines = [
        f"# {market} v14 묶음 {block} — {BLOCK_KO[block]} (docs/MODELING.md 32)",
        "",
        f"- 특징 {len(info['features'])}개 · 국면 {len(info['regime'])}개 · 선정 `{d['selected']}` · "
        f"게이트 {'통과' if d['gate']['passed'] else '미달'}",
    ]
    if info.get("empty"):
        lines.append(f"- 자료가 없어 빠진 항목: {', '.join(info['empty'])}")
    if block in ("C", "D"):
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
    dl, ok = _diff_lines(d["protocol"].get("alt_value"), served)
    lines += dl
    if info.get("guard"):
        lines += ["", "## 점검: 11·12월 말 행 제외(한국 배당락 시즌 — 정답이 배당 미포함 가격 수익률)", ""]
        gl, gok = _diff_lines(info["guard"], served)
        lines += gl
        ok = ok and gok
    verdict = "기여 있음(32.4 기준)" if ok else "기여 없음"
    if info.get("leak_suspect"):
        verdict += " — 누수 의심 항목이 있어 채택 판단에서 제외"
    lines += ["", f"**판정(서빙 모델 `{served}` 쌍 기준): {verdict}**"]
    text = "\n".join(lines) + "\n"
    (OUT / f"{market}_v14_{block}_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    from quantcio.modeling.report import render_markdown

    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    ap.add_argument("--block", choices=["C", "D", "H"], required=True)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        print(write_summary(a.market, a.block))
        return
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    uni, ds = v13s.load_inputs(a.market, settings, cache)
    base_xs, base_reg = list(ds.xs_cols), list(ds.regime_cols)
    say(f"block {a.block}")
    raw, reg, info = add_block(a.block, a.market, ds, uni, settings, cache)
    if not raw and not reg:
        raise SystemExit(f"묶음 {a.block}: 자료 없음 — scripts/v14_data.py fetch 먼저")
    ds.core_xs_cols, ds.core_regime_cols = base_xs, base_reg
    ds.xs_cols = [*base_xs, *[feat.x_col(c) for c in raw]]
    ds.regime_cols = [*base_reg, *reg]
    ds.snapshot = f"{ds.snapshot}|v14|{a.block}|{len(raw)}|{len(reg)}"
    ds.alt_raw = [*list(ds.alt_raw or []), *raw]
    say(f"block {a.block}: {len(raw)} features {raw} + regime {reg}; {json.dumps(info, ensure_ascii=False)[:1500]}")
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    sel.DIAGNOSTIC_PAIRS = tuple((x, f"{x}_no{a.block}") for x in CANDS[a.market][1])
    cands = v13s.candidates(a.market, a.block)
    res = sel.run_selection(ds, settings, cache, candidates=cands, threads=8, progress=say)
    g = guard(res, a.market, a.block)
    res.protocol["v14"] = {
        "declared": "docs/MODELING.md 32",
        "block": a.block,
        "features": raw,
        "regime": reg,
        **info,
        **({"guard": g} if g else {}),
    }
    stem = OUT / f"{a.market}_v14_{a.block}"
    stem.with_suffix(".json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        f"# 종목 선택 모델 선정 — {a.market}",
        f"# v14 연구 — 묶음 {a.block} {BLOCK_KO[a.block]} (서빙 아님) — {a.market}",
        1,
    )
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    print(write_summary(a.market, a.block))


if __name__ == "__main__":
    main()
