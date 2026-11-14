"""Investor-attention research run (docs/MODELING.md section 27, pre-declared 2026-10-05): Wikipedia pageviews (US) and
Naver search volume (KR) added to the stock-selection inputs, with diagnostic pairs that drop only the attention block.
Research only: the served models and ``src/quantcio`` are not changed.

    .venv/bin/python scripts/attention_study.py --market US [--fetch-only]
    .venv/bin/python scripts/attention_study.py --market KR [--fetch-only]

Writes reports/modeling/<MKT>_attention.{md,json} and reports/modeling/<MKT>_attention_summary.md.
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
import attention_data as att  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.dataset import build_dataset  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
DOCS = {
    "wiki_abn": ("위키백과 비정상 조회수(7일 ÷ 그 전 56일 중앙값, log)", "abnormal Wikipedia views"),
    "wiki_trend": ("위키백과 조회 추세(28일 ÷ 그 전 365일, log)", "Wikipedia views trend"),
    "wiki_size": ("규모 대비 위키백과 관심(log 조회수 − log 시가총액)", "Wikipedia attention per size"),
    "nsv_abn": ("네이버 비정상 검색량(그 주 ÷ 그 전 8주 중앙값, log)", "abnormal Naver search volume"),
    "nsv_trend": ("네이버 검색 추세(4주 ÷ 그 전 52주, log)", "Naver search trend"),
}
SPEC = {
    "US": {"served": "cat_rank", "base": ("cat_rank", "ridge", "logit_w", "lgbm_clf", "et")},
    "KR": {"served": "et", "base": ("et", "enet_logit", "logit_w", "lgbm_clf")},
}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _asof_frame(feats: pd.DataFrame, dates: np.ndarray, max_age_days: int = 10) -> np.ndarray:
    if feats is None or feats.empty:
        return np.full((len(dates), 0), np.nan)
    idx = feats.index.to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(idx, dates, "right") - 1
    ok = (pos >= 0) & ((dates - idx[np.clip(pos, 0, None)]) <= np.timedelta64(max_age_days, "D"))
    vals = feats.to_numpy(float)[np.clip(pos, 0, None)]
    vals[~ok] = np.nan
    return vals


def us_attention(ds, settings, cache, fetch_only: bool) -> list[str]:
    import us_v8_data as v8

    f = ds.frame
    chains = v8.cik_chains(ds, settings, cache)
    arts = att.wiki_articles(settings, sorted({c for cs in chains.values() for c in cs}), say)
    say(f"Wikipedia: {len(arts)} articles for {len(chains)} entities")
    att.wiki_fetch(settings, list(arts.values()), say)
    if fetch_only:
        return []
    cols = ["wiki_abn", "wiki_trend", "wiki_level"]
    vals = np.full((len(f), 3), np.nan)
    for e, g in f.groupby("entity", sort=False):
        from quantcio.data.pit.us_cik import entity_cik

        own = entity_cik(str(e))
        cands = [own, *sorted(chains.get(str(e), set()) - {own})]
        art = next((arts[c] for c in cands if c in arts), None)
        if not art:
            continue
        wf = att.wiki_features(att.wiki_series(settings, art))
        if wf.empty:
            continue
        ix = f.index.get_indexer(g.index)
        vals[ix] = _asof_frame(wf[cols], g["date"].to_numpy(dtype="datetime64[ns]"))
    f["wiki_abn"], f["wiki_trend"] = vals[:, 0], vals[:, 1]
    f["wiki_size"] = vals[:, 2] - np.log(f["mcap"].where(f["mcap"] > 0))
    return ["wiki_abn", "wiki_trend", "wiki_size"]


def kr_attention(ds, settings, cache, fetch_only: bool) -> list[str]:
    from api_keys import _kr_companies

    f = ds.frame
    names = _kr_companies(settings, cache)
    say(f"Naver trend: {att.naver_trend_fetch(settings, names, say)}")
    if fetch_only:
        return []
    feats = att.naver_trend_features(settings)
    vals = np.full((len(f), 2), np.nan)
    for e, g in f.groupby("entity", sort=False):
        nf = feats.get(str(e))
        if nf is None or nf.empty:
            continue
        ix = f.index.get_indexer(g.index)
        vals[ix] = _asof_frame(nf[["nsv_abn", "nsv_trend"]], g["date"].to_numpy(dtype="datetime64[ns]"))
    f["nsv_abn"], f["nsv_trend"] = vals[:, 0], vals[:, 1]
    return ["nsv_abn", "nsv_trend"]


def candidates(mkt: str) -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    out = [every["mom_12_1"], every["factor_ic"], *(every[n] for n in SPEC[mkt]["base"])]
    for name in sorted({SPEC[mkt]["served"], "logit_w", "lgbm_clf"}):
        d = copy.copy(every[name])
        d.name, d.diagnostic, d.feature_set = f"{name}_noattn", True, "core"
        d.fitter = every[name].fitter or every[name].name
        d.label_ko = every[name].label_ko + " · 진단: 관심도 제외"
        out.append(d)
    return out


def write_summary(mkt: str) -> str:
    d = json.loads((OUT / f"{mkt}_attention.json").read_text())
    info = d["protocol"]["attention"]
    lines = [
        f"# {mkt} 투자자 관심도 연구 요약 (docs/MODELING.md 27)",
        "",
        f"- 관심도 특징: {info['features']} · 결측률 {info['missing_share']}",
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
    lines += ["", "## 관심도의 기여 (같은 모델에서 관심도만 뺀 쌍과의 날짜별 AUC 차이)", ""]
    for k, v in (d["protocol"].get("alt_value") or {}).items():
        o, h = v["outer"], v.get("holdout") or {}
        hh = f" · 홀드아웃 {h['mean']:+.4f} [{h['lo']:+.4f}, {h['hi']:+.4f}]" if h.get("mean") is not None else ""
        lines.append(f"- {k}: 외부 {o['mean']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}] (t {o['t']:+.2f}){hh}")
    text = "\n".join(lines) + "\n"
    (OUT / f"{mkt}_attention_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    from quantcio.modeling.report import render_markdown

    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["US", "KR"], required=True)
    ap.add_argument("--fetch-only", action="store_true")
    a = ap.parse_args()
    mkt = a.market
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    if mkt == "US" and not a.fetch_only:
        import us_v9_study as v9

        ds, _ = v9.build_inputs(settings, cache)
    else:
        ds = build_dataset(Market(mkt), settings, cache, progress=say)
    raw = (us_attention if mkt == "US" else kr_attention)(ds, settings, cache, a.fetch_only)
    if a.fetch_only:
        return
    f = ds.frame
    for c in raw:
        f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
        feat.FEATURE_DOCS[c] = DOCS[c]
    base_xs = list(ds.xs_cols)
    ds.core_xs_cols, ds.core_regime_cols = base_xs, list(ds.regime_cols)  # diagnostics: everything but attention
    ds.xs_cols = [*base_xs, *[feat.x_col(c) for c in raw]]
    ds.snapshot = f"{ds.snapshot}|attention|{'+'.join(raw)}"
    lab = f[f["labelled"] & f["is_me"]]
    info = {"features": raw, "missing_share": {c: round(float(lab[c].isna().mean()), 3) for c in raw}}
    say(f"attention {info}")
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    served = SPEC[mkt]["served"]
    sel.DIAGNOSTIC_PAIRS = tuple((n, f"{n}_noattn") for n in sorted({served, "logit_w", "lgbm_clf"}))
    res = sel.run_selection(ds, settings, cache, candidates=candidates(mkt), threads=8, progress=say)
    res.protocol["attention"] = {"declared": "docs/MODELING.md 27", **info}
    (OUT / f"{mkt}_attention.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        f"# 종목 선택 모델 선정 — {mkt}", f"# {mkt} 투자자 관심도 연구 (서빙 아님) — {mkt}", 1
    )
    (OUT / f"{mkt}_attention.md").write_text(md, encoding="utf-8")
    print(write_summary(mkt))


if __name__ == "__main__":
    main()
