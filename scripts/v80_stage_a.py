"""Stage A of the return-model improvement (docs/MODELING.md 80.1, pre-declared 2026-10-25). Research only.

    .venv/bin/python scripts/v80_stage_a.py --market KR

No refitting: the protocol-v4 selection run is reloaded from its checkpoints, and its out-of-sample (outer folds)
+ holdout predictions are turned into score variants (EMA smoothing, momentum blend) and books (top quintile,
buffer, sector-neutral). Each variant is judged by the v4 gate on the v4 primary (top-quintile beta-adjusted net
excess): mean above mom_12_1 with the same book on the same dates AND 95% lower bound > 0.
Writes reports/modeling/<M>_v80a.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from quantcio.config import PROJECT_ROOT, load_settings
from quantcio.data.cache import DiskCache
from quantcio.modeling import metrics as mt
from quantcio.modeling import selection as sel
from quantcio.modeling.cli import guard_xsh
from quantcio.modeling.dataset import build_dataset
from quantcio.models import Market

OUT = PROJECT_ROOT / "reports" / "modeling"
HALF_LIFE = 2.0  # months
BUFFER_KEEP = 0.6  # keep a held name while its rank percentile >= this
BOOKS = ("top", "buffer", "sector")
BOOK_KO = {"top": "상위 20%", "buffer": "버퍼(20% 진입·40% 이탈)", "sector": "업종 중립 상위 20%"}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pct_rank(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("date")[col].rank(pct=True)


def ema(frame: pd.DataFrame, col: str, half_life: float = HALF_LIFE) -> pd.Series:
    """Per entity EMA (over its own month-ends, in date order) of the per-date rank percentile of ``col``."""
    alpha = 1.0 - 0.5 ** (1.0 / half_life)
    u = pct_rank(frame, col)
    d = frame[["date", "entity"]].assign(u=u.to_numpy()).sort_values(["entity", "date"])
    sm = d.groupby("entity")["u"].transform(lambda x: x.ewm(alpha=alpha, adjust=False, ignore_na=True).mean())
    return sm.reindex(frame.index)


def book_series(frame: pd.DataFrame, score: str, kind: str, cost_bps: float) -> pd.DataFrame:
    """Per date: book mean y_e - universe mean y_e - one-way turnover x cost (the v4 primary for kind 'top')."""
    d = frame[["date", "entity", score, "y_e", "sector"]].dropna(subset=[score, "y_e"])
    rows, prev = [], set()
    for t, g in d.groupby("date", sort=True):
        if len(g) < mt.MIN_NAMES:
            continue
        k = max(1, int(round(len(g) / 5)))
        if kind == "top":
            book = set(g.nlargest(k, score)["entity"])
        elif kind == "buffer":
            pct = g[score].rank(pct=True)
            keep = g[(pct >= BUFFER_KEEP).to_numpy() & g["entity"].isin(prev).to_numpy()]
            keep = keep.nlargest(k, score)
            add = g[~g["entity"].isin(set(keep["entity"]))].nlargest(max(k - len(keep), 0), score)
            book = set(keep["entity"]) | set(add["entity"])
        else:  # sector-neutral: the top fifth of every sector
            book = set()
            for _, s in g.groupby(g["sector"].fillna("기타")):
                ks = int(round(len(s) / 5))
                if ks:
                    book |= set(s.nlargest(ks, score)["entity"])
        if not book:
            continue
        turnover = 1.0 if not prev else len(book - prev) / len(book)
        gross = float(g.loc[g["entity"].isin(book), "y_e"].mean() - g["y_e"].mean())
        rows.append({"date": t, "net": gross - turnover * cost_bps / 1e4, "gross": gross, "turnover": turnover})
        prev = book
    return pd.DataFrame(rows).set_index("date")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    m = ap.parse_args().market
    s = load_settings()
    cache = DiskCache(s.cache_path)
    mk = Market(m)
    ds = build_dataset(mk, s, cache, use_cache=True, progress=say)
    ds = guard_xsh(ds, s, say)
    res = sel.run_selection(ds, s, cache, threads=8, use_cache=True, progress=say)
    fam = {c["name"]: c for c in res.candidates}
    ml = [
        n
        for n, c in fam.items()
        if not c.get("diagnostic") and c["family"] not in ("baseline", "factor") and n != sel.BASELINE
    ]
    b = max(ml, key=lambda n: (fam[n]["outer"].get(mt.PRIMARY_AUC) or {}).get("mean") or -1)
    say(f"B = {b} (outer AUC {fam[b]['outer'][mt.PRIMARY_AUC]['mean']:.4f}), M = {sel.BASELINE}")
    sec = ds.frame[ds.frame["is_me"]][["date", "entity", "sector"]].drop_duplicates(["date", "entity"])
    cols = ["date", "entity", "y_e", sel.BASELINE, b]
    f = pd.concat(
        [
            _labelled(res.oos, ds, cols).assign(part="outer"),
            _labelled(res.holdout_preds, ds, cols).assign(part="holdout"),
        ],
        ignore_index=True,
    ).merge(sec, on=["date", "entity"], how="left")
    f = f.sort_values(["date", "entity"]).reset_index(drop=True)
    f["M"], f["B"] = pct_rank(f, sel.BASELINE), pct_rank(f, b)
    f["MB"] = (f["M"] + f["B"]) / 2
    for c in ("M", "B", "MB"):
        f[f"{c}_ema2"] = ema(f, c)
    scores = ["M", "B", "MB", "M_ema2", "B_ema2", "MB_ema2"]
    cost = float(ds.cost_bps)
    check = book_series(f[f["part"] == "outer"], "M", "top", cost)["net"]
    ref = mt.per_date_top_ye_net(f[f["part"] == "outer"], "M", cost)
    assert np.allclose(check.reindex(ref.index).to_numpy(), ref.to_numpy()), "book 'top' must equal the v4 primary"
    out: dict = {"market": m, "B": b, "cost_bps": cost, "variants": {}}
    for kind in BOOKS:
        base = {p: book_series(f[f["part"] == p], "M", kind, cost) for p in ("outer", "holdout")}
        for sc in scores:
            ser = (
                base if sc == "M" else {p: book_series(f[f["part"] == p], sc, kind, cost) for p in ("outer", "holdout")}
            )
            o, h = ser["outer"], ser["holdout"]
            so, sh = mt.summarize(o["net"]), mt.summarize(h["net"])
            vs = mt.paired_difference(o["net"], base["outer"]["net"]) if sc != "M" else None
            mb = (mt.summarize(base["outer"]["net"])["mean"] or 0.0) if sc != "M" else None
            passed = bool(sc != "M" and so["mean"] > mb and (so["lo"] or -1) > 0)
            out["variants"][f"{sc}|{kind}"] = {
                "outer": so,
                "holdout": sh,
                "turnover": float(o["turnover"].mean()),
                "gross_outer": mt.summarize(o["gross"])["mean"],
                "vs_M": vs,
                "passed": passed,
            }
            say(
                f"{sc:8s} {kind:7s} outer {so['mean'] * 12:+.2%}/yr lo {so['lo'] * 12:+.2%} "
                f"holdout {sh['mean'] * 12:+.2%}"
            )
    lines = [
        f"# {m} 수익 모델 개선 A단계 — 비용·조합·책 구성 (docs/MODELING.md 80.1)",
        "",
        f"- B = `{b}`(외부 AUC 1위 ML), M = `{sel.BASELINE}` · 비용 {cost:g}bp × 편도 회전율 · "
        "값 = 연환산(월 평균 × 12)",
        "- 판정: 외부 평균 > 같은 책의 M **그리고** 95% 하한 > 0 (79의 게이트와 같음)",
        "",
        "| 점수 | 책 | 외부 순초과(연) [95% CI] | 비용 전(연) | 월 회전율 | M 대비(연, t) | 홀드아웃(연) [CI] | 통과 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def a(x):
        return "–" if x is None else f"{x * 12:+.1%}"

    for key, v in out["variants"].items():
        sc, kind = key.split("|")
        o, h, vs = v["outer"], v["holdout"], v["vs_M"]
        vs_txt = "–" if not vs else f"{a(vs['mean'])} ({vs['t']:+.2f})"
        lines.append(
            f"| {sc} | {BOOK_KO[kind]} | {a(o['mean'])} [{a(o['lo'])}, {a(o['hi'])}] | {a(v['gross_outer'])} | "
            f"{v['turnover']:.0%} | {vs_txt} | {a(h['mean'])} [{a(h['lo'])}, {a(h['hi'])}] | "
            f"{'통과' if v['passed'] else '–'} |"
        )
    text = "\n".join(lines) + "\n"
    (OUT / f"{m}_v80a.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    (OUT / f"{m}_v80a.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def _labelled(pred: pd.DataFrame, ds, cols: list[str]) -> pd.DataFrame:
    """Predictions with the label y_e (labelled month-end rows only)."""
    f = ds.frame
    lab = f[f["is_me"] & f["labelled"]][["date", "entity", "y_e"]].drop_duplicates(["date", "entity"])
    p = pred.drop(columns=[c for c in ("y_e", "r") if c in pred]).merge(lab, on=["date", "entity"])
    return p[cols]


if __name__ == "__main__":
    sys.exit(main())
