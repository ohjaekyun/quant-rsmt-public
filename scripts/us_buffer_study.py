"""US one-week signal with turnover control (docs/MODELING.md 67.1, pre-declared 2026-10-17). Research only.

    .venv/bin/python scripts/us_buffer_study.py --part trees     # then --part zoo, --part ft, --part merge

Reruns the 40 weekly selection (same inputs and protocol, fewer candidates) and books its outer / holdout scores with
an entry-at-top-20% / exit-below-top-q_out buffer and an optional EMA of the per-date score ranks; the 40 metric
(equal-weight book minus the universe, one-way turnover x 10 bp). Writes reports/modeling/US_buffer67.{json,md}.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_zoo as zoo  # noqa: E402
import us_weekly_study as ws  # noqa: E402
import v13_study as v13s  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
KEEP = ("mom_12_1", "et", "cat_rank", "cat_rank_noW")
NEW = (
    "nn_gkx",
    "ft_transformer",
    "xs_attention",
    "ipca",
    "gam_spline",
    "knn",
    "soft_forest",
    "rsmt_v3",
)  # 67 amendment
Q_OUT = (0.2, 0.3, 0.4, 0.5)
HALF_LIFE = (None, 2, 4)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def book(frame: pd.DataFrame, score: str, q_out: float, hl: int | None, cost_bps: float) -> pd.DataFrame:
    """Per week: buffered equal-weight book minus the universe (gross, one-way turnover, net, book size)."""
    d = frame[["date", "entity", score, "r"]].dropna(subset=[score, "r"]).copy()
    d["R"] = np.expm1(d["r"])
    d["p"] = d.groupby("date")[score].rank(pct=True)
    if hl:
        alpha = 1.0 - 0.5 ** (1.0 / hl)
        d = d.sort_values(["entity", "date"])
        d["p"] = d.groupby("entity")["p"].transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
        d["p"] = d.groupby("date")["p"].rank(pct=True)
    held: set[str] = set()
    rows = []
    for t, g in d.groupby("date", sort=True):
        if len(g) < mt.MIN_NAMES:
            continue
        order = g.sort_values("p", ascending=False)["entity"].to_numpy()
        k = max(1, int(round(len(g) / 5)))
        top = set(order[:k])
        stay = set(order[: max(k, int(round(len(g) * q_out)))])
        names = top | (held & stay)
        turnover = 1.0 if not held else len(names - held) / len(names)
        gross = float(g.loc[g["entity"].isin(names), "R"].mean() - g["R"].mean())
        rows.append({"date": t, "gross": gross, "turnover": turnover, "n": len(names)})
        held = names
    out = pd.DataFrame(rows).set_index("date")
    out["net"] = out["gross"] - out["turnover"] * float(cost_bps) / 1e4
    return out


def stats(b: pd.DataFrame) -> dict:
    return {
        "net": mt.summarize(b["net"].to_numpy()),
        "gross": mt.summarize(b["gross"].to_numpy()),
        "turnover": float(b["turnover"].mean()),
        "book": float(b["n"].mean()),
    }


PARTS = {  # one process per group: torch next to LightGBM / CatBoost segfaulted (67 amendment 2)
    "trees": ("mom_12_1", "et", "cat_rank", "cat_rank_noW", "rsmt_v3"),
    "zoo": ("mom_12_1", "nn_gkx", "xs_attention", "ipca", "gam_spline", "knn", "soft_forest"),
    "ft": ("mom_12_1", "ft_transformer"),
}


def part_dir(settings) -> Path:
    d = Path(settings.cache_path) / "buffer67"
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare():
    settings = load_settings(horizon_days=ws.H)
    cache = DiskCache(settings.cache_path)
    uni, ds = v13s.load_inputs("US", settings, cache)
    f = ds.frame
    base_xs, base_reg = list(ds.xs_cols), list(ds.regime_cols)
    w, info = ws.block_w(ds, uni, settings, cache)
    raw = []
    for c in w.columns:
        vals = w[c].astype(float)
        if vals.notna().sum() == 0 or vals.nunique() < 2:
            continue
        f[c] = vals.to_numpy()
        f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
        feat.FEATURE_DOCS[c] = ws.W_DOCS.get(c, (c, c))
        raw.append(c)
    f["is_me"] = ws.weekly_eval_rows(f)
    ds.core_xs_cols, ds.core_regime_cols = base_xs, base_reg
    ds.xs_cols = [*base_xs, *[feat.x_col(c) for c in raw]]
    ds.snapshot = f"{ds.snapshot}|weekly40|{len(raw)}"
    for k, v in ws.WEEKLY.items():
        setattr(sel, k, v)
    sel.COMPOSITE_CANDIDATES, sel.COMPOSITES, sel.DIAGNOSTIC_PAIRS = (), (), ()
    return settings, cache, ds


def run_part(part: str) -> None:
    settings, cache, ds = prepare()
    every = {c.name: c for c in cmod.default_candidates()}
    pool = {c.name: c for c in ws.candidates()}
    pool.update({c.name: c for c in zoo.candidates()})
    pool["soft_forest"], pool["rsmt_v3"] = zoo.soft_forest_candidate(), every["rsmt_v3"]
    cands = [pool[n] for n in PARTS[part]]
    if part in ("zoo", "ft"):
        zoo.warm_up()
    say(f"weekly rerun [{part}]: {[c.name for c in cands]}, {len(ds.xs_cols)} inputs")
    res = sel.run_selection(ds, settings, cache, candidates=cands, threads=8, progress=say)
    d = part_dir(settings)
    keep = ["date", "entity", "r", "y_e", *[n for n in PARTS[part] if n in res.oos]]
    res.oos[[c for c in keep if c in res.oos]].to_parquet(d / f"{part}_oos.parquet", index=False)
    res.holdout_preds[[c for c in keep if c in res.holdout_preds]].to_parquet(d / f"{part}_hold.parquet", index=False)
    (d / f"{part}.json").write_text(json.dumps({"snapshot": ds.snapshot, "cost_bps": float(ds.cost_bps)}))
    say(f"[{part}] saved {len(res.oos):,} outer / {len(res.holdout_preds):,} holdout rows")


def merged(settings) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    d = part_dir(settings)
    out = []
    for kind in ("oos", "hold"):
        base = pd.read_parquet(d / f"trees_{kind}.parquet")
        for part in ("zoo", "ft"):
            p = d / f"{part}_{kind}.parquet"
            if p.exists():
                x = pd.read_parquet(p)
                cols = [c for c in x.columns if c not in ("r", "y_e", "mom_12_1")]
                base = base.merge(x[cols], on=["date", "entity"], how="left")
        out.append(base)
    cost = json.loads((d / "trees.json").read_text())["cost_bps"]
    return out[0], out[1], cost


def evaluate(oos: pd.DataFrame, hold: pd.DataFrame, cost: float) -> None:
    from types import SimpleNamespace

    res = SimpleNamespace(oos=oos, holdout_preds=hold)
    r = {"cost_bps": cost, "models": {}}
    if "rsmt_v3" in res.oos and "et" in res.oos:
        res.oos["blend_et_rsmt"] = sel.rank_mean(res.oos, ["et", "rsmt_v3"])
        res.holdout_preds["blend_et_rsmt"] = sel.rank_mean(res.holdout_preds, ["et", "rsmt_v3"])
    r["auc"] = {
        m: {
            "outer": mt.evaluate(res.oos, cost, m)["auc_median"],
            "holdout": mt.evaluate(res.holdout_preds, cost, m)["auc_median"],
        }
        for m in (*KEEP, *NEW, "blend_et_rsmt")
        if m in res.oos
    }
    for m in (*KEEP, *NEW, "blend_et_rsmt"):
        if m not in res.oos:
            continue
        base = mt.top_quintile(res.oos, cost, m)["net"].mean()
        grid = {}
        for q in Q_OUT:
            for hl in HALF_LIFE:
                key = f"q{int(q * 100)}_hl{hl or 0}"
                grid[key] = {
                    "outer": stats(book(res.oos, m, q, hl, cost)),
                    "holdout": stats(book(res.holdout_preds, m, q, hl, cost)),
                }
        pick = max(grid, key=lambda k: grid[k]["outer"]["net"]["mean"] or -1)
        g = grid[pick]
        cand = (g["outer"]["net"]["t"] or 0) >= 2.0 and (g["holdout"]["net"]["mean"] or -1) >= 0
        r["models"][m] = {"check_q20_hl0_vs_metric": [grid["q20_hl0"]["outer"]["net"]["mean"], float(base)],
                          "grid": grid, "pick": pick, "candidate": cand}  # fmt: skip
        say(f"{m}: pick {pick} outer net {g['outer']['net']['mean']:+.5f} (t {g['outer']['net']['t']:+.2f}) "
            f"holdout {g['holdout']['net']['mean']:+.5f}")  # fmt: skip
    lines = [
        "# US 주간 신호 + 회전율 통제 (docs/MODELING.md 67.1)",
        "",
        f"- 비용 {cost:.0f}bp(한 방향 회전율 기준), 상위 20%에서 사고 상위 q_out 밖에서 팜, "
        "점수 순위의 지수 이동평균(반감기 주)",
        "",
        "| 모델 | 변형 | 외부 주당 순초과 (t) | 외부 회전율 | 평균 보유 수 | 홀드아웃 주당 순초과 (t) "
        "| 홀드아웃 회전율 |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, x in r["models"].items():
        for key in ("q20_hl0", x["pick"]):
            g = x["grid"][key]
            o, h = g["outer"], g["holdout"]
            tag = " **(고름)**" if key == x["pick"] else " (버퍼 없음)"
            lines.append(
                f"| {m} | {key}{tag} | {o['net']['mean'] * 100:+.3f}% ({o['net']['t']:+.2f}) | {o['turnover']:.0%} | "
                f"{o['book']:.0f} | {h['net']['mean'] * 100:+.3f}% ({h['net']['t']:+.2f}) | {h['turnover']:.0%} |"
            )
    lines += [
        "",
        "운용 후보(외부 t ≥ 2, 홀드아웃 ≥ 0): "
        + (", ".join(m for m, x in r["models"].items() if x["candidate"]) or "없음"),
    ]
    lines += [
        "",
        "모델별 주간 AUC(날짜 내, 중앙값 초과):",
        "",
        "| 모델 | 외부 AUC [95% CI] | 홀드아웃 AUC |",
        "|---|---|---|",
    ]
    for m, a in r["auc"].items():
        o, h = a["outer"], a["holdout"]
        lines.append(f"| {m} | {o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] | {h['mean']:.4f} |")
    text = "\n".join(lines) + "\n"
    (OUT / "US_buffer67.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    (OUT / "US_buffer67.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=[*PARTS, "merge"], required=True)
    part = ap.parse_args().part
    mt.PERIODS_PER_YEAR = 52
    mt.block_bootstrap_means.__defaults__ = (13, mt.BOOT_SIMS, 0)
    if part == "merge":
        settings = load_settings(horizon_days=ws.H)
        evaluate(*merged(settings))
    else:
        run_part(part)


if __name__ == "__main__":
    main()
