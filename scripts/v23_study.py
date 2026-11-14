"""KR short interest + index membership on top of the 55 pool (docs/MODELING.md 61, pre-declared 2026-10-14).
Research only.

    .venv/bin/python scripts/v23_study.py

Writes reports/modeling/KR_v23.{json,md}.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_select_study as fss  # noqa: E402
import v19_study as v19s  # noqa: E402
import v20_study as v20s  # noqa: E402
import v21_study as v21s  # noqa: E402
import v23_short_index as v  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
_f = fss._f
_pd = v19s._pd


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def regime_ic(f: pd.DataFrame, col: str) -> dict:
    """Per-date Spearman IC of a raw feature with y_e (labelled month-ends), split by the ban regime."""
    me = f[f["is_me"] & f["labelled"]][["date", col, "y_e", "r_kr_short_ban"]].dropna()
    out = {}
    for lvl, g in me.groupby("r_kr_short_ban"):
        ics = (
            g.groupby("date")
            .apply(lambda x: x[col].rank().corr(x["y_e"].rank()) if len(x) > 20 else np.nan, include_groups=False)
            .dropna()
        )
        out[str(lvl)] = mt.summarize(ics.to_numpy())
    return out


def main() -> None:
    m = "KR"
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    st = v21s.run_through_55(m, settings, cache)
    ds, res55 = st["ds"], st["res55"]
    xs55, reg55, snap55 = list(ds.xs_cols), list(ds.regime_cols), ds.snapshot
    f = ds.frame
    df, info = v.build_kr(f, settings)
    new_x, new_r = [], []
    for c in df.columns:
        vals = df[c].astype(float)
        if vals.notna().sum() == 0 or vals.nunique() < 2:
            continue
        f[c] = vals.to_numpy()
        feat.FEATURE_DOCS[c] = v.DOCS[c]
        if c in v.REGIME:
            new_r.append(c)
        else:
            f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
            new_x.append(feat.x_col(c))
    aucs, sus = v20s.leak_screen(ds, new_x)
    new_x = [x for x in new_x if x not in set(sus)]
    ds.xs_cols, ds.regime_cols = [*xs55, *new_x], [*reg55, *new_r]
    ds.snapshot = f"{snap55}|v23|{len(new_x)}|{len(new_r)}"
    say(f"SI/IX pool: +{len(new_x)} features, +{len(new_r)} regime (dropped as leak suspects: {sus})")
    v19s._store(settings, m, ds.snapshot)
    res = sel.run_selection(ds, settings, cache, candidates=st["cands55"], threads=8, progress=say)
    d, d55 = res.to_dict(), res55.to_dict()
    core = f"{fss.SERVED[m]}_core"
    best = d["selected"]
    vs = {}
    for ref, rres, tag in ((core, res, "61"), (d55["selected"], res55, "55")):
        if ref == best and tag == "61":
            continue
        vs[f"{best}(61) − {ref}({tag})"] = {
            "outer": mt.paired_difference(_pd(res, best, "outer"), _pd(rres, ref, "outer")),
            "holdout": mt.paired_difference(_pd(res, best, "holdout"), _pd(rres, ref, "holdout")),
        }
    n = len(ds.xs_cols)
    fits = [x for k, x in fss.FS_LOG.items() if k.split("|")[2] == str(n)]
    cnt = Counter(c for x in fits for c in x["xs"])
    hold0 = pd.Timestamp(d["protocol"]["holdout"][0])
    me = f[f["is_me"] & f["labelled"] & (f["date"] < hold0)]
    r = {
        "block": info,
        "missing_auc": aucs,
        "leak_suspect": sus,
        "new_xs": new_x,
        "new_regime": new_r,
        "ic_t": fss._ic_t(me, new_x).round(2).to_dict(),
        "si_ratio_ic_by_regime": regime_ic(f, "si_ratio"),
        "selected_share": {c: cnt.get(c, 0) / max(len(fits), 1) for c in new_x},
        "selection": d,
        "selection55": d55,
        "vs": vs,
        "contribution": v19s.contribution(res, res55),
    }
    cb = r["contribution"]
    o, h = cb["mean_outer"], cb["mean_holdout"]
    g = d["gate"]
    lines = [
        "# KR 공매도 잔고·지수 편입 (docs/MODELING.md 61)",
        "",
        f"- 자료: {info}",
        f"- 새 특징 {len(new_x)}개 + 국면 {len(new_r)}개 · 누출 의심 {sus or '없음'}",
        f"- 선정 `{best}` · 외부 {_f(g['auc'], '.4f')} (하한 {_f(g['auc_lo'], '.4f')}) · "
        f"게이트 {'통과' if g['passed'] else '미달'} · "
        f"55절 선정 `{d55['selected']}`",
        f"- **SI·IX 기여**(같은 fs 9개, 모델 평균): 외부 {_f(o['mean'])} [{_f(o['lo'])}, {_f(o['hi'])}] · "
        f"홀드아웃 {_f(h['mean'])} "
        f"→ **{'도움' if cb['helps'] else '도움 없음'}**",
        "",
        "| 특징 | IC t(홀드아웃 전) | 선택 비율 |",
        "|---|---|---|",
    ]
    for c in new_x:
        lines.append(f"| {c} | {_f(r['ic_t'].get(c), '+.2f')} | {r['selected_share'][c]:.0%} |")
    lines += ["", "`si_ratio` 날짜별 IC (국면별, 판정 미사용):"]
    for lvl, x in r["si_ratio_ic_by_regime"].items():
        name = {"0.0": "허용", "0.5": "부분 허용", "1.0": "전면 금지"}.get(lvl, lvl)
        lines.append(f"- {name}: 평균 {_f(x['mean'], '+.4f')} (t {_f(x['t'], '+.2f')}, {x['n']}개 날짜)")
    for k, x in vs.items():
        lines.append(f"- {k}: 외부 {_f(x['outer']['mean'])} · 홀드아웃 {_f(x['holdout'].get('mean'))}")
    text = "\n".join(lines) + "\n"
    (OUT / "KR_v23.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    (OUT / "KR_v23.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
