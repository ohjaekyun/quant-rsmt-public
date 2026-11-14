"""Head-to-head audit of the RSMT-Q configuration adopted 2026-10-01 against the one served before it (docs/RSMT.md 8).

Usage:
    .venv/bin/python scripts/rsmt_adopted_check.py --market KR [--out reports/rsmt] [--workers 6]

OLD = the configuration served 2026-09-29..30 (7-year PIT universe, a row every 5 sessions, half-life 504, no sigma
terms, split-test clusters of one h-block); NEW = config/default.yaml as adopted (longest history, every session,
half-life 1260, per-market sigma terms, 3-horizon clusters). Both are scored on the SAME rows:
- CV: every session of the OLD panel's CV folds (layouts 3/4/5) before the last 24 months, rows present in both panels;
- holdout: every session of the last 24 months (fit on rows whose label ended before it). The holdout was already
  looked at in sections 6-7, so it is an out-of-time check, not an untouched sample.
Settings change in memory only. Fold tasks run in worker processes (`rsmt_history._task`).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import rsmt_history as H

import quantcio.quant.rsmt as R
from quantcio.quant.panel import market_config

log = logging.getLogger("rsmt_adopted_check")


def old_settings(base: Any, market: str) -> Any:
    """The pre-adoption served model, built as a DAILY panel (it trains on its every-5th-session subset)."""
    s = base.model_copy(deep=True)
    m = s.rsmt.markets[market]
    m.history_start = None
    m.half_life_days = 504.0
    m.sigma_terms = []
    s.rsmt.sigma_terms = []
    s.rsmt.boot_block_horizons = 1
    s.rsmt.overlap_step = 1
    m.overlap_step = None
    return s


def run(market: str, out_dir: Path, n_paths: int, workers: int) -> dict[str, Any]:
    from quantcio import holdout
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache
    from quantcio.models import Market
    from quantcio.pipeline import training_panel
    from quantcio.quant.panel import build_panel

    t_all = time.perf_counter()
    mk = Market(market)
    base = load_settings()
    cache = DiskCache(base.cache_path)
    s_old_daily = old_settings(base, market)
    s_old = s_old_daily.model_copy(deep=True)
    s_old.rsmt.overlap_step = H.L0_STEP  # as served: row multiplicity / min_leaf of step-5 rows
    s_new = base
    panels = {}
    for name, s in (("OLD", s_old_daily), ("NEW", s_new)):
        price, _, _ = training_panel(mk, s, cache)
        price, _ = holdout.guard_panel(price, s)
        panels[name] = build_panel(price, s)
    ref = panels["OLD"].train
    grid = pd.DatetimeIndex(sorted(ref.loc[ref["on_grid"], "date"].unique()))
    first_hold = grid[grid > grid[-1] - pd.DateOffset(months=H.HOLDOUT_MONTHS)][0]
    last_date = pd.Timestamp(min(ref["date"].max(), panels["NEW"].train["date"].max()))
    both = ref[["ticker", "date"]].merge(panels["NEW"].train[["ticker", "date"]], on=["ticker", "date"])
    ref_pre = ref[(pd.to_datetime(ref["t_end"]) < first_hold).to_numpy()].reset_index(drop=True)
    min_train = max(2 * market_config(s_old, market).min_leaf, 30)
    tasks: list[dict[str, Any]] = []

    def keys_in(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        d = pd.to_datetime(both["date"])
        return both[((d >= start) & (d <= end)).to_numpy()]

    for k in H.LAYOUTS:
        for fold in R.cv_splits(ref_pre, k):
            if int(ref_pre.iloc[fold.train_idx]["on_grid"].sum()) < min_train:
                continue
            keys = keys_in(fold.test_start, fold.test_end)
            for name in ("OLD", "NEW"):
                tasks.append(
                    {
                        "key": f"cv|{name}|{k}|{fold.fold}",
                        "kind": "eval",
                        "panel": name,
                        "settings": name,
                        "step5": name == "OLD",
                        "start": fold.test_start,
                        "end": fold.test_end,
                        "keys": keys,
                        "fold": fold.fold,
                    }
                )
    hkeys = keys_in(first_hold, last_date)
    for name in ("OLD", "NEW"):
        tasks.append(
            {
                "key": f"hold|{name}|0",
                "kind": "eval",
                "panel": name,
                "settings": name,
                "step5": name == "OLD",
                "start": first_hold,
                "end": last_date,
                "keys": hkeys,
                "describe": True,
            }
        )
    settings = {"OLD": s_old, "NEW": s_new}
    work = {
        "panels": panels,
        "settings": settings,
        "step5": {"OLD": H.step_mask(ref, H.L0_STEP)},
        "n_paths": n_paths,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, other = H._run_pool(
        tasks, work, out_dir / f".adopted_{market}.pkl", workers, lambda g: settings[g.split("|")[1]], n_paths
    )
    res: dict[str, Any] = {
        "market": market,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "holdout_start": str(first_hold.date()),
        "cv": {},
        "holdout": {},
    }
    for name in ("OLD", "NEW"):
        res["cv"][name] = H._summarise({str(k): metrics[f"cv|{name}|{k}"] for k in H.LAYOUTS})
        res["holdout"][name] = {**metrics[f"hold|{name}"], "tree": other[f"hold|{name}|0"]["tree"]}
    res["seconds"] = round(time.perf_counter() - t_all, 1)
    (out_dir / f"adopted_check_{market}.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    log.info("%s done: %s", market, {n: res["cv"][n]["crps"] for n in ("OLD", "NEW")})
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--market", required=True, choices=["KR", "US"])
    ap.add_argument("--out", default="reports/rsmt")
    ap.add_argument("--paths", type=int, default=R.CV_PATHS)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    res = run(args.market, Path(args.out), args.paths, args.workers)
    print(json.dumps({k: res[k] for k in ("cv", "holdout")}, default=str)[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())
