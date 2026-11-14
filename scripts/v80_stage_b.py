"""Stage B of the return-model improvement (docs/MODELING.md 80.4, pre-declared 2026-10-25). Research only.

    .venv/bin/python scripts/v80_stage_b.py --market KR --variant sector

One protocol-v4 selection run per variant (sector-relative label / 60-session label / stronger regularisation /
KR top-400 universe). Every variant's selected model is then judged on the SAME yardstick: its out-of-sample and
holdout predictions against the standard 20-session simple label y_e, top-quintile / sector-neutral / smoothed
books, against mom_12_1 of the same run (v4 gate: mean above M and 95% lower bound > 0).
Writes reports/modeling/<M>_v80b_<variant>.{md,json}.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v80_stage_a as va  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.cli import guard_xsh  # noqa: E402
from quantcio.modeling.dataset import add_date_targets, build_dataset  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
VARIANTS = ("sector", "h60", "reg", "wide")
VARIANT_KO = {
    "sector": "업종 상대 라벨",
    "h60": "60거래일 라벨",
    "reg": "정규화 강화 + 내부 CV 3블록",
    "wide": "한국 상위 400 · 비용 50bp",
}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def regularised(c: cmod.Candidate) -> cmod.Candidate:
    if c.name in cmod.STACKED:  # RSMT-Q's "alpha" is its split-test level, not a penalty: unchanged
        return c

    def tighten(p: dict) -> dict:
        q = dict(p)
        if "leaf_frac" in q:
            q["leaf_frac"] = min(0.1, float(q["leaf_frac"]) * 3)
        if "depth" in q:
            q["depth"] = max(2, int(q["depth"]) - 1)
        if "leaves" in q:
            q["leaves"] = max(7, int(q["leaves"]) // 2)
        if "a" in q:
            q["a"] = float(q["a"]) * 10
        if "C" in q:
            q["C"] = float(q["C"]) / 10
        if "alpha" in q and c.name == "mlp":
            q["alpha"] = float(q["alpha"]) * 10
        return q

    return dataclasses.replace(c, grid=[tighten(p) for p in c.grid])


def sector_relative(ds):
    """A copy of the dataset whose label is y_e minus its (date, sector) mean over labelled rows."""
    d = copy.copy(ds)
    f = ds.frame.copy()
    lab = f["labelled"] & f["y_e"].notna()
    key = [f["date"], f["sector"].fillna("기타")]
    mean = f["y_e"].where(lab).groupby(key).transform("mean")
    f["y_e"] = (f["y_e"] - mean).where(lab)
    d.frame = add_date_targets(f)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    a = ap.parse_args()
    m, v = a.market, a.variant
    if v == "wide" and m != "KR":
        raise SystemExit("wide is a KR variant")
    s = load_settings()
    cache = DiskCache(s.cache_path)
    mk = Market(m)
    if v == "wide":
        s = s.model_copy(deep=True)
        s.backtest.kr_universe_n = 400
    ds20 = guard_xsh(build_dataset(mk, s, cache, use_cache=True, progress=say), s, say)
    if v == "wide":
        ds20.cost_bps = 50.0
    cands = cmod.default_candidates()
    sv = s
    if v == "sector":
        dsv = sector_relative(ds20)
    elif v == "h60":
        sv = s.model_copy(deep=True)
        sv.horizon_days = 60
        dsv = guard_xsh(build_dataset(mk, sv, cache, use_cache=True, progress=say), sv, say)
    else:
        dsv = copy.copy(ds20)
    if v == "reg":
        cands = [regularised(c) for c in cands]
        sel.INNER_BLOCKS = 3
    dsv.snapshot = f"{dsv.snapshot}|v80b_{v}"
    res = sel.run_selection(dsv, sv, cache, candidates=cands, threads=8, use_cache=True, progress=say)
    chosen = res.selected
    names = [chosen] if chosen == sel.BASELINE else [chosen, sel.BASELINE]
    cols = ["date", "entity", "y_e", *names]
    sec = ds20.frame[ds20.frame["is_me"]][["date", "entity", "sector"]].drop_duplicates(["date", "entity"])
    f = pd.concat(
        [
            va._labelled(res.oos, ds20, cols).assign(part="outer"),
            va._labelled(res.holdout_preds, ds20, cols).assign(part="holdout"),
        ],
        ignore_index=True,
    ).merge(sec, on=["date", "entity"], how="left")
    f = f.sort_values(["date", "entity"]).reset_index(drop=True)
    f["M"], f["S"] = va.pct_rank(f, sel.BASELINE), va.pct_rank(f, chosen)
    f["S_ema2"] = va.ema(f, "S")
    cost = float(ds20.cost_bps)
    out: dict = {
        "market": m,
        "variant": v,
        "selected": chosen,
        "variant_gate": res.gate,
        "cost_bps": cost,
        "rows": {},
    }
    for sc, kind in (("S", "top"), ("S", "sector"), ("S_ema2", "top"), ("M", "top"), ("M", "sector")):
        o = va.book_series(f[f["part"] == "outer"], sc, kind, cost)
        h = va.book_series(f[f["part"] == "holdout"], sc, kind, cost)
        out["rows"][f"{sc}|{kind}"] = {
            "outer": mt.summarize(o["net"]),
            "holdout": mt.summarize(h["net"]),
            "turnover": float(o["turnover"].mean()),
            "series": o["net"],
        }
    base = out["rows"]["M|top"]
    for key, r in out["rows"].items():
        if key.startswith("M|"):
            r["vs_M"], r["passed"] = None, None
            continue
        bk = "M|sector" if key.endswith("sector") else "M|top"
        r["vs_M"] = mt.paired_difference(r["series"], out["rows"][bk]["series"])
        r["passed"] = bool(r["outer"]["mean"] > out["rows"][bk]["outer"]["mean"] and (r["outer"]["lo"] or -1) > 0)
    for r in out["rows"].values():
        r.pop("series")

    def ann(x):
        return "–" if x is None else f"{x * 12:+.1%}"

    def vs_txt(vs):
        if not vs or vs.get("t") is None:  # the variant selected mom_12_1 itself: no difference to test
            return "–"
        return f"{ann(vs['mean'])} ({vs['t']:+.2f})"

    lines = [
        f"# {m} 수익 모델 개선 B단계 — {VARIANT_KO[v]} (docs/MODELING.md 80.4)",
        "",
        f"- 변형 안의 v4 선정: `{chosen}` · 변형 지표 외부 {res.gate.get('value')} (게이트 {res.gate.get('passed')})",
        f"- 평가는 표준 20일 단순 라벨 · 비용 {cost:g}bp × 편도 회전율 · 값 = 연환산 · "
        "판정 = 평균 > 같은 책 M, 하한 > 0",
        f"- 기준(M = mom_12_1, 상위 20%) 외부 {ann(base['outer']['mean'])} [{ann(base['outer']['lo'])}, "
        f"{ann(base['outer']['hi'])}]",
        "",
        "| 점수 | 책 | 외부(연) [95% CI] | 월 회전율 | M 대비(연, t) | 홀드아웃(연) [CI] | 통과 |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, r in out["rows"].items():
        sc, kind = key.split("|")
        o, h, vs = r["outer"], r["holdout"], r["vs_M"]
        lines.append(
            f"| {chosen if sc.startswith('S') else 'mom_12_1'}{' 평활' if sc.endswith('ema2') else ''} | "
            f"{va.BOOK_KO[kind]} | {ann(o['mean'])} [{ann(o['lo'])}, {ann(o['hi'])}] | {r['turnover']:.0%} | "
            f"{vs_txt(vs)} | "
            f"{ann(h['mean'])} [{ann(h['lo'])}, {ann(h['hi'])}] | "
            f"{'–' if r['passed'] is None else '통과' if r['passed'] else '미달'} |"
        )
    text = "\n".join(lines) + "\n"
    (OUT / f"{m}_v80b_{v}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    (OUT / f"{m}_v80b_{v}.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    sys.exit(main())
