"""US: the 62/65 model families on the 67.2 pool (55 pool + text-peer block) vs the served configuration
(docs/MODELING.md 67.3, amendment declared 2026-10-17 before any 67 result). Research only.

    .venv/bin/python scripts/v27b_study.py --part trees    # then --part zoo (own process, 67 amendment 2), merge

Writes reports/modeling/US_v27b.{json,md}.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_select_study as fss  # noqa: E402
import model_zoo as zoo  # noqa: E402
import v20_study as v20s  # noqa: E402
import v21_study as v21s  # noqa: E402
import v27_peers as v  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
_f = fss._f
SKIP = ("ft_transformer",)  # ~45 min per fit on ~340 inputs (67 amendment)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["trees", "zoo", "merge"], required=True)
    part = ap.parse_args().part
    m = "US"
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    st = v21s.run_through_55(m, settings, cache)
    ds = st["ds"]
    xs55, snap55 = list(ds.xs_cols), ds.snapshot
    f = ds.frame
    df, info = v.build(f, settings, cache)
    new = []
    for c in df.columns:
        vals = df[c].astype(float)
        if vals.notna().sum() == 0 or vals.nunique() < 2:
            continue
        f[c] = vals.to_numpy()
        f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
        feat.FEATURE_DOCS[c] = v.DOCS[c]
        new.append(feat.x_col(c))
    _, sus = v20s.leak_screen(ds, new)
    new = [x for x in new if x not in set(sus)]
    ds.xs_cols = [*xs55, *new]
    ds.snapshot = f"{snap55}|v27b|{len(new)}"
    sel.COMPOSITE_CANDIDATES, sel.COMPOSITES, sel.DIAGNOSTIC_PAIRS = (), (), ()
    every = {c.name: c for c in cmod.default_candidates()}
    served = f"{fss.SERVED[m]}_core"
    core = next(c for c in st["cands55"] if c.name == served)
    zoo_c = [*(c for c in zoo.candidates() if c.name not in SKIP), zoo.soft_forest_candidate()]
    parts = {"trees": [every["mom_12_1"], core, every["et"], every["rsmt_v3"]], "zoo": [every["mom_12_1"], *zoo_c]}
    d = Path(settings.cache_path) / "v27b"
    d.mkdir(parents=True, exist_ok=True)
    if part != "merge":
        cands = parts[part]
        if part == "zoo":
            zoo.warm_up()
        say(f"67.3 [{part}]: {len(ds.xs_cols)} inputs, candidates {[c.name for c in cands]}")
        res = sel.run_selection(ds, settings, cache, candidates=cands, threads=8, progress=say)
        keep = ["date", "entity", "r", "y_e", *[c.name for c in cands]]
        res.oos[[c for c in keep if c in res.oos]].to_parquet(d / f"{part}_oos.parquet", index=False)
        res.holdout_preds[[c for c in keep if c in res.holdout_preds]].to_parquet(
            d / f"{part}_hold.parquet", index=False
        )
        say(f"[{part}] saved")
        return
    oos, hold = (
        pd.read_parquet(d / f"trees_{k}.parquet").merge(
            pd.read_parquet(d / f"zoo_{k}.parquet").drop(columns=["r", "y_e", "mom_12_1"], errors="ignore"),
            on=["date", "entity"],
            how="left",
        )
        for k in ("oos", "hold")
    )
    cands = [*parts["trees"], *zoo_c]
    if "rsmt_v3" in oos and "et" in oos:
        oos["blend_et_rsmt"], hold["blend_et_rsmt"] = (
            sel.rank_mean(oos, ["et", "rsmt_v3"]),
            sel.rank_mean(hold, ["et", "rsmt_v3"]),
        )
    cost = ds.cost_bps
    names = [c.name for c in cands if c.name in oos] + (["blend_et_rsmt"] if "blend_et_rsmt" in oos else [])
    table = {n: {"outer": mt.evaluate(oos, cost, n), "holdout": mt.evaluate(hold, cost, n)} for n in names}
    vs = {
        n: {
            "outer": mt.paired_difference(mt.per_date_primary(oos, n), mt.per_date_primary(oos, served)),
            "holdout": mt.paired_difference(mt.per_date_primary(hold, n), mt.per_date_primary(hold, served)),
        }
        for n in names
        if n not in (served, "mom_12_1")
    }
    switch = [n for n, x in vs.items() if (x["outer"].get("lo") or -1) > 0 and (x["holdout"].get("mean") or -1) >= 0]
    r = {
        "block": info,
        "leak_suspect": sus,
        "new_xs": new,
        "table": table,
        "vs_served": vs,
        "switch": switch,
    }
    lines = [
        f"# US 새 모델 계열 — 55절 풀 + 동종 기업 묶음 (docs/MODELING.md 67.3), 입력 {len(ds.xs_cols)}개",
        "",
        "| 후보 | 외부 AUC [95% CI] | 외부 IC (t) | 홀드아웃 AUC | − 서빙(외부 [CI] · 홀드아웃) |",
        "|---|---|---|---|---|",
    ]
    for n, x in table.items():
        o, h, ic = x["outer"]["auc_median"], x["holdout"]["auc_median"], x["outer"]["ic"]
        d = vs.get(n)
        diff = (
            f"{_f(d['outer']['mean'])} [{_f(d['outer']['lo'])}, {_f(d['outer']['hi'])}] · "
            f"{_f(d['holdout'].get('mean'))}"
            if d
            else "—"
        )
        lines.append(
            f"| {n} | {_f(o['mean'], '.4f')} [{_f(o['lo'], '.4f')}, {_f(o['hi'], '.4f')}] | {_f(ic['mean'])} "
            f"({_f(ic['t'], '+.2f')}) | {_f(h['mean'], '.4f')} | {diff} |"
        )
    lines += [
        "",
        f"- 서빙 교체 후보(외부 하한 > 0, 홀드아웃 ≥ 0): **{', '.join(switch) or '없음'}**",
        f"- FT-Transformer는 비용 때문에 제외(67 보충) · 동종 기업 묶음 {len(new)}개 특징 · 자료 {info}",
    ]
    text = "\n".join(lines) + "\n"
    (OUT / "US_v27b.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    (OUT / "US_v27b.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
