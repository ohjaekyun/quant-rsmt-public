"""Hugging Face sentiment block on top of the 55 pool (docs/MODELING.md 64, pre-declared 2026-10-16). Research only.

    .venv/bin/python scripts/v25_study.py --market US

Contribution = per-date AUC of the same nine fs_* models on (55 pool + NLP block) minus the 55 pool, model-averaged.
US also reports (descriptive, not part of the verdict) FinBERT vs the LM word list on the same releases, and the
one-year Finnhub headline IC. Writes reports/modeling/<M>_v25.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_select_study as fss  # noqa: E402
import nlp_models as nm  # noqa: E402
import v19_study as v19s  # noqa: E402
import v20_study as v20s  # noqa: E402
import v21_study as v21s  # noqa: E402
import v22_tone as tone  # noqa: E402
import v25_nlp_features as v  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
_f = fss._f
_pd = v19s._pd
TITLE = {"KR": "한국 사업보고서 KR-FinBERT 어조", "US": "미국 실적 발표문·보도자료 FinBERT 어조"}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ic_summary(me: pd.DataFrame, col: str) -> dict:
    """Mean per-date Spearman IC of a raw column with y_e (dates with >= 20 non-missing names)."""
    ics = []
    for _, g in me[["date", col, "y_e"]].dropna().groupby("date"):
        if len(g) >= 20:
            ics.append(g[col].rank().corr(g["y_e"].rank()))
    return mt.summarize(np.array(ics, dtype=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    ap.add_argument("--en", default=nm.DEFAULT["en"])
    ap.add_argument("--ko", default=nm.DEFAULT["ko"])
    a = ap.parse_args()
    m = a.market
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    st = v21s.run_through_55(m, settings, cache)
    ds, res55 = st["ds"], st["res55"]
    xs55, snap55 = list(ds.xs_cols), ds.snapshot
    f = ds.frame
    df, info = v.build_us(f, settings, cache, a.en) if m == "US" else v.build_kr(f, settings, a.ko)
    new, cover = [], {}
    for c in df.columns:
        vals = df[c].astype(float)
        if vals.notna().sum() == 0 or vals.nunique() < 2:
            continue
        f[c] = vals.to_numpy()
        f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
        feat.FEATURE_DOCS[c] = v.DOCS[c]
        new.append(feat.x_col(c))
        cover[feat.x_col(c)] = float(vals[f["is_me"].to_numpy()].notna().mean())
    aucs, sus = v20s.leak_screen(ds, new)
    new = [x for x in new if x not in set(sus)]
    ds.xs_cols = [*xs55, *new]
    ds.snapshot = f"{snap55}|v25|{info['model']}|{len(new)}"
    say(f"NLP pool: +{len(new)} features ({info}; dropped as leak suspects: {sus})")
    v19s._store(settings, m, ds.snapshot)
    res = sel.run_selection(ds, settings, cache, candidates=st["cands55"], threads=8, progress=say)
    d, d55 = res.to_dict(), res55.to_dict()
    core = f"{fss.SERVED[m]}_core"
    best = d["selected"]
    vs = {}
    for ref, rres, tag in ((core, res, "64"), (d55["selected"], res55, "55")):
        if ref == best and tag == "64":
            continue
        vs[f"{best}(64) − {ref}({tag})"] = {
            "outer": mt.paired_difference(_pd(res, best, "outer"), _pd(rres, ref, "outer")),
            "holdout": mt.paired_difference(_pd(res, best, "holdout"), _pd(rres, ref, "holdout")),
        }
    n = len(ds.xs_cols)
    fits = [x for k, x in fss.FS_LOG.items() if k.split("|")[2] == str(n)]
    cnt = Counter(c for x in fits for c in x["xs"])
    hold0 = pd.Timestamp(d["protocol"]["holdout"][0])
    me = f[f["is_me"] & f["labelled"] & (f["date"] < hold0)]
    r = {
        "market": m,
        "model": nm.MODELS[a.en if m == "US" else a.ko].__dict__,
        "block": info,
        "coverage_me": cover,
        "missing_auc": aucs,
        "leak_suspect": sus,
        "new_xs": new,
        "ic_t": fss._ic_t(me, new).round(2).to_dict(),
        "selected_share": {c: cnt.get(c, 0) / max(len(fits), 1) for c in new},
        "selection": d,
        "selection55": d55,
        "vs": vs,
        "contribution": v19s.contribution(res, res55),
    }
    extra = []
    if m == "US":
        lm, _ = tone.block_tx(f, settings, cache, "lm")
        f["tx_tone_lm"] = lm["tx_tone"].to_numpy()
        me = f[f["is_me"] & f["labelled"] & (f["date"] < hold0)]
        both = me.dropna(subset=["nb_tone", "tx_tone_lm"])
        r["finbert_vs_lm"] = {"nb_tone": ic_summary(both, "nb_tone"), "tx_tone_lm": ic_summary(both, "tx_tone_lm")}
        r["finbert_vs_lm"]["rank_corr"] = float(both["nb_tone"].rank().corr(both["tx_tone_lm"].rank()))
        c3, i3 = v.build_us_c(f, settings, a.en)
        for c in c3.columns:
            f[c] = c3[c].to_numpy()
        allme = f[f["is_me"] & f["labelled"]]
        r["finnhub"] = {"block": i3, **{c: ic_summary(allme, c) for c in c3.columns}}
        fb = r["finbert_vs_lm"]
        extra = [
            "",
            "보조(판정 미사용) — 같은 실적 발표문에서 FinBERT 어조 vs LM 사전 어조, 홀드아웃 전 월말 IC:",
            f"- FinBERT `nb_tone`: 평균 {_f(fb['nb_tone']['mean'], '+.4f')} (t {_f(fb['nb_tone']['t'], '+.2f')}) · "
            f"LM `tx_tone`: 평균 {_f(fb['tx_tone_lm']['mean'], '+.4f')} (t {_f(fb['tx_tone_lm']['t'], '+.2f')}) · "
            f"두 점수 순위상관 {fb['rank_corr']:.2f}",
            f"- Finnhub 뉴스 제목 {i3['headlines']:,}건({i3['first'][:10]}~, 1년 — 전부 홀드아웃 기간 안, "
            "기술적 참고만):",
        ]
        for c in c3.columns:
            x = r["finnhub"][c]
            extra.append(f"  - `{c}`: 월말 IC 평균 {_f(x['mean'], '+.4f')} (t {_f(x['t'], '+.2f')}, {x['n']}개 월말)")
    cb = r["contribution"]
    o, h = cb["mean_outer"], cb["mean_holdout"]
    g = d["gate"]
    spec = nm.MODELS[a.en if m == "US" else a.ko]
    lines = [
        f"# {m} {TITLE[m]} (docs/MODELING.md 64)",
        "",
        f"- 모델 `{spec.repo}` ({spec.revision[:8]}, {spec.licence}) · 자료: {info}",
        f"- 새 특징 {len(new)}개 · 누출 의심(빼고 시험) {sus or '없음'}",
        f"- 선정 `{best}` · 외부 {_f(g['auc'], '.4f')} (하한 {_f(g['auc_lo'], '.4f')}) · "
        f"게이트 {'통과' if g['passed'] else '미달'} · 55절 선정 `{d55['selected']}`",
        f"- **자연어 분류 기여**(같은 fs 9개, 모델 평균): 외부 {_f(o['mean'])} [{_f(o['lo'])}, {_f(o['hi'])}] · "
        f"홀드아웃 {_f(h['mean'])} → **{'도움' if cb['helps'] else '도움 없음'}**",
        "",
        "| 특징 | 설명 | 월말 채움 | IC t(홀드아웃 전) | 선택 비율 |",
        "|---|---|---|---|---|",
    ]
    for c in new:
        raw = c.removeprefix("x_")
        lines.append(
            f"| {c} | {v.DOCS[raw][0]} | {cover[c]:.0%} | {_f(r['ic_t'].get(c), '+.2f')} | "
            f"{r['selected_share'][c]:.0%} |"
        )
    for k, x in vs.items():
        lines.append(f"- {k}: 외부 {_f(x['outer']['mean'])} · 홀드아웃 {_f(x['holdout'].get('mean'))}")
    text = "\n".join(lines + extra) + "\n"
    (OUT / f"{m}_v25.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    (OUT / f"{m}_v25.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
