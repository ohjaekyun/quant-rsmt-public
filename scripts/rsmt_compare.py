"""Pre-declared RSMT-Q variant comparison on the real point-in-time panel of one market (docs/RSMT.md, section 1).

Usage:
    .venv/bin/python scripts/rsmt_compare.py --market KR [--out reports/rsmt] [--paths 1000]

Protocol (declared in docs/RSMT.md before the first run):
- data: `pipeline.training_panel` (PIT universe) with the registered holdouts removed (`holdout.guard_panel`);
- final holdout: the last 24 months of sample dates, never used for the choice (fit on every earlier row whose label
  ended before it, evaluated once - for the winner and V0 only);
- choice: expanding-window purged CV (`quant.rsmt.cross_validate_layouts`) on the pre-holdout rows, layouts 3/4/5,
  every variant evaluated on the same non-overlapping test rows against the same baseline twin;
- rule: lowest mean CRPS over the layouts among variants whose mean cover80 is in [0.75, 0.85] and mean exc95 in
  [0.02, 0.08]; a relative CRPS difference < 0.1% is a tie -> fewer switches on, then the lower V-number.

Writes reports/rsmt/compare_<M>.json and .md. Never touches the holdout registry / ledger.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantcio import holdout
from quantcio.config import Settings, load_settings
from quantcio.data.cache import DiskCache
from quantcio.models import Market
from quantcio.pipeline import training_panel
from quantcio.quant.panel import RSMT_VARIANT_KEYS, PanelData, build_panel
from quantcio.quant.rsmt import CV_PATHS, RSMTree, cross_validate_layouts, evaluate_split

log = logging.getLogger("rsmt_compare")

LAYOUTS = (3, 4, 5)
HOLDOUT_MONTHS = 24
TIE_REL = 0.001  # relative CRPS difference below which two variants tie
COVER80_BAND = (0.75, 0.85)
EXC95_BAND = (0.02, 0.08)
OVERLAP_STEP = 5
PCA_K = 3
GAMMA_K = 50.0
# V0 = the original model; every other variant turns switches on (docs/RSMT.md table 1.1)
BASE = {
    "overlap_step": 0,
    "multiple_testing": "bonferroni",
    "pca_components": PCA_K,
    "target_scale": "none",
    "gamma_mode": "global",
    "gamma_shrink_k": GAMMA_K,
}
VARIANTS: dict[str, dict[str, Any]] = {
    "V0": {},
    "V1": {"overlap_step": OVERLAP_STEP},
    "V2": {"multiple_testing": "simes"},
    "V3": {"multiple_testing": "pca"},
    "V4": {"target_scale": "ewma"},
    "V5": {"gamma_mode": "leaf"},
    "V6": {"overlap_step": OVERLAP_STEP, "multiple_testing": "simes", "target_scale": "ewma", "gamma_mode": "leaf"},
    "V7": {"overlap_step": OVERLAP_STEP, "multiple_testing": "pca", "target_scale": "ewma", "gamma_mode": "leaf"},
}
LABELS = {
    "V0": "baseline (current)",
    "V1": "+overlap (step 5, block clusters)",
    "V2": "+simes",
    "V3": "+pca (K=3)",
    "V4": "+standardised target",
    "V5": "+leaf gamma (k=50)",
    "V6": "overlap+simes+standardised+leaf gamma",
    "V7": "overlap+pca+standardised+leaf gamma",
}


def n_switches(name: str) -> int:
    """Switches on (overlap, multiple testing, target scale, gamma mode) - the rule's simplicity order."""
    return len({"overlap_step", "multiple_testing", "target_scale", "gamma_mode"} & set(VARIANTS[name]))


def variant_settings(base: Settings, name: str) -> Settings:
    """`base` with every model switch at V0 (global and per market) and then the variant's switches on."""
    s = base.model_copy(deep=True)
    for k, v in {**BASE, **VARIANTS[name]}.items():
        setattr(s.rsmt, k, v)
    for mcfg in s.rsmt.markets.values():
        for k in RSMT_VARIANT_KEYS:
            setattr(mcfg, k, None)
    return s


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def select(rows: dict[str, dict[str, Any]]) -> tuple[str, list[str], str]:
    """The pre-declared rule: (winner, tied variants, explanation)."""
    eligible = [
        n
        for n, r in rows.items()
        if r["crps"] is not None
        and r["cover80"] is not None
        and r["exc95"] is not None
        and COVER80_BAND[0] <= r["cover80"] <= COVER80_BAND[1]
        and EXC95_BAND[0] <= r["exc95"] <= EXC95_BAND[1]
    ]
    if not eligible:
        return "V0", [], "no variant inside both coverage bands -> V0 stays"
    best = min(rows[n]["crps"] for n in eligible)
    tied = [n for n in eligible if (rows[n]["crps"] - best) / best < TIE_REL]
    tied.sort(key=lambda n: (n_switches(n), int(n[1:])))
    lowest = min(eligible, key=lambda n: rows[n]["crps"])
    why = (
        f"eligible {eligible}; lowest mean CRPS {lowest} ({best:.6f}); tied within {TIE_REL:.1%}: {tied}; "
        f"simplest tied variant -> {tied[0]}"
    )
    return tied[0], tied, why


def run(market: Market, out_dir: Path, n_paths: int) -> dict[str, Any]:
    t_all = time.perf_counter()
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    log.info("%s: loading the PIT training panel", market.value)
    price, notes, universe = training_panel(market, settings, cache)
    price, dropped = holdout.guard_panel(price, settings)
    log.info("%s: guard_panel dropped %s", market.value, dropped)

    panels: dict[int, PanelData] = {}
    for name in VARIANTS:
        step = int(variant_settings(settings, name).rsmt.overlap_step)
        if step not in panels:
            t = time.perf_counter()
            panels[step] = build_panel(price, variant_settings(settings, name))
            log.info(
                "%s: panel overlap_step=%d built in %.1fs (%d rows)",
                market.value,
                step,
                time.perf_counter() - t,
                len(panels[step].train),
            )
    coarse = panels[0]
    grid_dates = pd.DatetimeIndex(sorted(coarse.train.loc[coarse.train["on_grid"], "date"].unique()))
    cutoff = grid_dates[-1] - pd.DateOffset(months=HOLDOUT_MONTHS)
    hold_dates = grid_dates[grid_dates > cutoff]
    first_hold = hold_dates[0]

    def pre(p: PanelData) -> PanelData:
        tr = p.train
        keep = (pd.to_datetime(tr["t_end"]) < first_hold).to_numpy()
        return replace(p, train=tr[keep].reset_index(drop=True))

    pre_panels = {k: pre(p) for k, p in panels.items()}
    info = {
        "market": market.value,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_digest": settings.digest(),
        "as_of": str(coarse.as_of.date()) if coarse.as_of is not None else None,
        "universe_notes": notes,
        "holdout_guard": dropped,
        "n_train_coarse": int(len(coarse.train)),
        "n_train_overlap": int(len(panels[OVERLAP_STEP].train)) if OVERLAP_STEP in panels else None,
        "n_grid_dates": int(len(grid_dates)),
        "first_date": str(grid_dates[0].date()),
        "holdout_start": str(first_hold.date()),
        "holdout_dates": int(len(hold_dates)),
        "pre_holdout_rows_coarse": int(len(pre_panels[0].train)),
        "layouts": list(LAYOUTS),
        "n_paths": n_paths,
        "alpha": settings.rsmt.markets[market.value].alpha,
        "rule": (
            "lowest mean CRPS over layouts 3/4/5 among variants with mean cover80 in [0.75, 0.85] and mean exc95 in "
            "[0.02, 0.08]; relative CRPS difference < 0.1% = tie -> fewer switches, then lower V-number"
        ),
    }
    rows: dict[str, dict[str, Any]] = {}
    for name in VARIANTS:
        s = variant_settings(settings, name)
        pdata = pre_panels[int(s.rsmt.overlap_step)]
        t = time.perf_counter()
        runs = cross_validate_layouts(pdata, s, list(LAYOUTS), n_paths)
        cv_secs = time.perf_counter() - t
        per: dict[str, Any] = {}
        for k, res in runs.items():
            m = res.metrics
            per[str(k)] = {
                "available": res.available,
                "crps": _f(m.get("crps")),
                "base_crps": _f(m.get("base_crps")),
                "crps_gain": _f(m.get("crps_gain")),
                "crps_gain_p05": _f(m.get("crps_gain_p05")),
                "cover80": _f(m.get("cover80")),
                "exc95": _f(m.get("exc95")),
                "auc": _f(m.get("auc")),
                "base_auc": _f(m.get("base_auc")),
                "logloss": _f(m.get("logloss")),
                "base_logloss": _f(m.get("base_logloss")),
                "auc_gain_p05": _f(m.get("auc_gain_p05")),
                "n_oos": int(m.get("n_oos", 0)),
                "n_oos_dates": int(m.get("n_oos_dates", 0)),
                "splits_by_fold": [int(f.get("n_splits", 0)) for f in res.folds if "skipped" not in f],
                "distribution_gate": bool(res.distribution_ok),
                "direction_gate": bool(res.direction_ok),
                "seconds": float(res.seconds),
                "notes": list(res.notes),
            }

        def mean(key: str) -> float | None:
            vals = [per[str(k)][key] for k in LAYOUTS if per[str(k)][key] is not None]
            return float(np.mean(vals)) if len(vals) == len(LAYOUTS) else None

        t = time.perf_counter()
        tree = RSMTree.from_settings(s, market).fit_frame(pdata.train)
        fit_secs = time.perf_counter() - t
        n_dist = sum(per[str(k)]["distribution_gate"] for k in LAYOUTS)
        n_dir = sum(per[str(k)]["direction_gate"] for k in LAYOUTS)
        rows[name] = {
            "label": LABELS[name],
            "switches": {**BASE, **VARIANTS[name]},
            "n_switches": n_switches(name),
            "crps": mean("crps"),
            "base_crps": mean("base_crps"),
            "cover80": mean("cover80"),
            "exc95": mean("exc95"),
            "auc": mean("auc"),
            "logloss": mean("logloss"),
            "crps_gain_p05_min": min((per[str(k)]["crps_gain_p05"] or -np.inf) for k in LAYOUTS),
            "distribution_gate_layouts": int(n_dist),
            "direction_gate_layouts": int(n_dir),
            "splits_final": int(tree.n_splits),
            "tree": tree.describe("en"),
            "gamma": float(tree.gamma),
            "eta_equivalent": float(tree.eta_equivalent),
            "fit_seconds": round(fit_secs, 2),
            "cv_seconds": round(cv_secs, 2),
            "layouts": per,
        }
        r = rows[name]
        log.info(
            "%s %s: CRPS %.6f (base %.6f) cover80 %.3f exc95 %.3f AUC %.4f splits %d fit %.1fs cv %.1fs",
            market.value,
            name,
            r["crps"] or float("nan"),
            r["base_crps"] or float("nan"),
            r["cover80"] or float("nan"),
            r["exc95"] or float("nan"),
            r["auc"] or float("nan"),
            r["splits_final"],
            fit_secs,
            cv_secs,
        )
    winner, tied, why = select(rows)
    log.info("%s: winner %s (%s)", market.value, winner, why)

    hold: dict[str, Any] = {}
    for name in dict.fromkeys(["V0", winner]):
        s = variant_settings(settings, name)
        full = panels[int(s.rsmt.overlap_step)]
        tr = full.train
        train_mask = (pd.to_datetime(tr["t_end"]) < first_hold).to_numpy()
        test_mask = (pd.to_datetime(tr["date"]) >= first_hold).to_numpy()
        t = time.perf_counter()
        m, model, oos = evaluate_split(full, s, train_mask, test_mask, n_paths)
        hold[name] = {
            **{
                k: _f(m.get(k))
                for k in (
                    "crps",
                    "base_crps",
                    "crps_gain",
                    "crps_gain_p05",
                    "cover80",
                    "base_cover80",
                    "exc95",
                    "base_exc95",
                    "auc",
                    "base_auc",
                    "logloss",
                    "base_logloss",
                )
            },
            "n_oos": int(m.get("n_oos", 0)),
            "n_oos_dates": int(m.get("n_oos_dates", 0)),
            "splits": int(model.n_splits),
            "distribution_gate": bool(m.get("distribution_gate")),
            "direction_gate": bool(m.get("direction_gate")),
            "seconds": round(time.perf_counter() - t, 2),
        }
        log.info("%s holdout %s: %s", market.value, name, hold[name])
    result = {
        "info": info,
        "variants": rows,
        "winner": winner,
        "tied": tied,
        "why": why,
        "holdout": hold,
        "seconds": round(time.perf_counter() - t_all, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"compare_{market.value}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (out_dir / f"compare_{market.value}.md").write_text(render(result), encoding="utf-8")
    return result


def _n(x: Any, d: int = 4, signed: bool = False) -> str:
    v = _f(x)
    if v is None:
        return "—"
    return f"{v:+.{d}f}" if signed else f"{v:.{d}f}"


def render(res: dict[str, Any]) -> str:
    info, rows, hold = res["info"], res["variants"], res["holdout"]
    md = [
        f"# RSMT-Q variant comparison — {info['market']} (pre-declared rule, docs/RSMT.md §1)",
        "",
        f"- generated {info['generated_at']} · data as of {info['as_of']} · config digest `{info['config_digest']}`",
        f"- panel: {info['n_train_coarse']:,} non-overlapping rows ({info['n_grid_dates']} sample dates from "
        f"{info['first_date']}), {info['n_train_overlap']:,} rows with overlap step {OVERLAP_STEP}; holdout guard "
        f"{info['holdout_guard']}",
        f"- final holdout: {info['holdout_dates']} sample dates from {info['holdout_start']} (never used for the "
        f"choice); CV on {info['pre_holdout_rows_coarse']:,} pre-holdout non-overlapping rows, layouts "
        f"{info['layouts']}, {info['n_paths']} paths per row, alpha {info['alpha']}",
        f"- rule: {info['rule']}",
        "",
        "## CV (pre-holdout; means over layouts 3/4/5)",
        "",
        "| variant | switches | CRPS | base CRPS | rel. gain | cover80 | exc95 | AUC | logloss | eligible | "
        "dist. gate layouts | dir. gate layouts | splits (final / per fold) | fit s | CV s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        elig = (
            r["cover80"] is not None
            and COVER80_BAND[0] <= r["cover80"] <= COVER80_BAND[1]
            and r["exc95"] is not None
            and EXC95_BAND[0] <= r["exc95"] <= EXC95_BAND[1]
        )
        rel = (r["base_crps"] - r["crps"]) / r["base_crps"] if r["crps"] is not None and r["base_crps"] else None
        folds = "/".join(",".join(map(str, r["layouts"][str(k)]["splits_by_fold"])) for k in LAYOUTS)
        md.append(
            f"| **{name}** {r['label']} | {r['n_switches']} | {_n(r['crps'], 6)} | {_n(r['base_crps'], 6)} | "
            f"{'—' if rel is None else f'{rel:+.3%}'} | {_n(r['cover80'], 3)} | {_n(r['exc95'], 3)} | "
            f"{_n(r['auc'], 4)} | {_n(r['logloss'], 4)} | {'yes' if elig else 'no'} | "
            f"{r['distribution_gate_layouts']}/3 | {r['direction_gate_layouts']}/3 | {r['splits_final']} / {folds} | "
            f"{r['fit_seconds']:.1f} | {r['cv_seconds']:.1f} |"
        )
    md += [
        "",
        f"**Winner: {res['winner']}** — {res['why']}",
        "",
        "## Per layout",
        "",
        "| variant | layout | CRPS | CRPS gain bound (5%) | cover80 | exc95 | AUC | AUC gain bound (5%) | dist. gate | "
        "dir. gate |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        for k in LAYOUTS:
            lay = r["layouts"][str(k)]
            md.append(
                f"| {name} | {k} | {_n(lay['crps'], 6)} | {_n(lay['crps_gain_p05'], 6, True)} | "
                f"{_n(lay['cover80'], 3)} | {_n(lay['exc95'], 3)} | {_n(lay['auc'], 4)} | "
                f"{_n(lay['auc_gain_p05'], 4, True)} | {'pass' if lay['distribution_gate'] else 'fail'} | "
                f"{'pass' if lay['direction_gate'] else 'fail'} |"
            )
    md += [
        "",
        "## Final holdout (evaluated once; not used for the choice)",
        "",
        "| variant | CRPS | base CRPS | CRPS gain bound (5%) | cover80 | exc95 | AUC | base AUC | logloss | splits | "
        "rows / dates |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, h in hold.items():
        md.append(
            f"| {name} | {_n(h['crps'], 6)} | {_n(h['base_crps'], 6)} | {_n(h['crps_gain_p05'], 6, True)} | "
            f"{_n(h['cover80'], 3)} | {_n(h['exc95'], 3)} | {_n(h['auc'], 4)} | {_n(h['base_auc'], 4)} | "
            f"{_n(h['logloss'], 4)} | {h['splits']} | {h['n_oos']:,} / {h['n_oos_dates']} |"
        )
    md += [
        "",
        "Winner's curse: 8 variants were compared on the same CV rows, so the winner's CV CRPS is optimistically "
        "biased; the holdout row above is the unbiased check against V0.",
        "",
        "## Trees (final pre-holdout fit)",
        "",
    ]
    for name, r in rows.items():
        md.append(f"**{name}** (γ {r['gamma']:+.4f}, η≡{r['eta_equivalent']:.3f})")
        md.append("")
        md += [f"    {ln}" for ln in r["tree"]]
        md.append("")
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--market", required=True, choices=["KR", "US"])
    ap.add_argument("--out", default="reports/rsmt")
    ap.add_argument("--paths", type=int, default=CV_PATHS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    res = run(Market(args.market), Path(args.out), args.paths)
    print(f"{args.market}: winner {res['winner']} ({res['why']}) in {res['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
