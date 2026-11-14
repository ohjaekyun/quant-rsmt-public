"""Why is the stock-selection AUC low? In-sample vs out-of-sample AUC per walk-forward fold (diagnostic, 2026-10-25).
Research only.

    .venv/bin/python scripts/auc_diag.py --market KR

For representative candidates (linear logit, extra trees, LightGBM / XGBoost classifiers) and every protocol
context (6-10 outer folds + the holdout): fit on the purged training window exactly as the selection does, then
the within-date above-median AUC (the v2/v3 primary) on (a) the training month-ends (in-sample), (b) the last 12
training month-ends (recent in-sample), (c) the test block (out of sample). Control: the same fit on labels
shuffled within each date (pure memorisation). KR uses the top-200 universe of the v3 / v4 runs.
Writes reports/modeling/<M>_auc_diag.md.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from quantcio.config import PROJECT_ROOT, load_settings
from quantcio.data.cache import DiskCache
from quantcio.modeling import candidates as cmod
from quantcio.modeling import metrics as mt
from quantcio.modeling import selection as sel
from quantcio.modeling.cli import guard_xsh
from quantcio.modeling.dataset import add_date_targets, build_dataset
from quantcio.models import Market

OUT = PROJECT_ROOT / "reports" / "modeling"
MODELS = {  # candidate -> params (middle of its grid)
    "logit": {"C": 0.005},
    "et": {"kind": "et", "leaf_frac": 0.005},
    "lgbm_clf": {"leaves": 31, "leaf_frac": 0.01},
    "xgb_clf": {"depth": 3},
}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def auc(frame: pd.DataFrame, score: np.ndarray) -> float:
    return float(mt.per_date_auc_median(frame.assign(score=score)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    s = load_settings().model_copy(deep=True)
    if a.market == "KR":
        s.modeling.kr_universe_n, s.modeling.kr_cost_bps = 200, 30.0
    cache = DiskCache(s.cache_path)
    ds = guard_xsh(build_dataset(Market(a.market), s, cache, use_cache=True, progress=say), s, say)
    data = sel._Data(ds)
    outer, hold, _, _ = sel.make_contexts(data)
    by = {c.name: c for c in cmod.default_candidates()}
    rows = []
    rng = np.random.default_rng(0)
    for ctx in [*outer, hold]:
        for name, params in MODELS.items():
            c = by[name]
            tr = data.train(ctx.train_end, c.rows)
            te = data.eval_rows(ctx.test_dates)
            fc = sel._ctx(data, ctx.test_dates[0], s, a.threads)
            f = c.fit(tr, params, "none", fc)
            tr_me = tr[tr["is_me"]]
            recent = tr_me[tr_me["date"] >= sorted(tr_me["date"].unique())[-12]]
            row = {
                "ctx": ctx.name,
                "model": name,
                "train": auc(tr_me, f.predict(tr_me)),
                "recent": auc(recent, f.predict(recent)),
                "test": auc(te, f.predict(te)),
                "n_train_dates": int(tr_me["date"].nunique()),
            }
            if ctx is hold and name in ("et", "lgbm_clf"):
                sh = tr.copy()
                sh["y_e"] = sh.groupby("date")["y_e"].transform(lambda x: rng.permutation(x.to_numpy()))
                sh = add_date_targets(sh)
                g = c.fit(sh, params, "none", fc)
                sh_me = sh[sh["is_me"]]
                row["shuffled_train"] = auc(sh_me, g.predict(sh_me))
                row["shuffled_test"] = auc(te, g.predict(te))
            rows.append(row)
            say(f"{ctx.name} {name}: train {row['train']:.3f} recent {row['recent']:.3f} test {row['test']:.3f}")
    t = pd.DataFrame(rows)
    summ = t.groupby("model")[["train", "recent", "test"]].mean()
    lines = [
        f"# {a.market} AUC 진단 — 학습(표본 내) vs 표본 외 (날짜 내 AUC, 중앙값 초과)",
        "",
        "| 모델 | 학습 구간 전체(표본 내) | 학습 마지막 12개월(표본 내) | 다음 블록(표본 외) | 차이(학습 − 표본 외) |",
        "|---|---|---|---|---|",
    ]
    for m, r in summ.iterrows():
        lines.append(
            f"| {m} | {r['train']:.3f} | {r['recent']:.3f} | {r['test']:.3f} | {r['train'] - r['test']:+.3f} |"
        )
    lines += ["", "## 폴드별 표본 외 AUC", "", "| 폴드 | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for ctx, g in t.groupby("ctx", sort=False):
        lines.append(
            f"| {ctx} | " + " | ".join(f"{v:.3f}" for v in g.set_index("model").reindex(MODELS)["test"]) + " |"
        )
    sh = t.dropna(subset=["shuffled_train"]) if "shuffled_train" in t else pd.DataFrame()
    if len(sh):
        lines += ["", "## 대조: 날짜 안에서 라벨을 섞어 학습(순수 암기)", ""]
        for _, r in sh.iterrows():
            lines.append(
                f"- {r['model']}: 섞은 라벨 학습 AUC {r['shuffled_train']:.3f} · 표본 외 {r['shuffled_test']:.3f} "
                f"(실제 라벨: 학습 {r['train']:.3f} · 표본 외 {r['test']:.3f})"
            )
    text = "\n".join(lines) + "\n"
    (OUT / f"{a.market}_auc_diag.md").write_text(text, encoding="utf-8")
    t.to_csv(OUT / f"{a.market}_auc_diag.csv", index=False)
    print(text, flush=True)


if __name__ == "__main__":
    sys.exit(main())
