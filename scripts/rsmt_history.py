"""RSMT-Q with the longest history on every business day - pre-declared comparison (docs/RSMT.md sections 7 / 7.1).

Usage:
    .venv/bin/python scripts/rsmt_history.py --market KR [--out reports/rsmt] [--paths 1000] [--workers 8]

Research only: settings change in memory (the FTH-20260930 freeze covers the config values and the signal code).
- maximum history: KR `pit.kr.start_year = 1995`, `history_years = 31`; US `history_years = 17` (validated prices and
  XBRL share counts start mid-2009); every session is a training row (`overlap_step = 1`) for these configs;
- validation on EVERY session of each test block; confidence bounds resample h-session blocks;
- stage 1: L0 (the served model: 7 years, a row every 5 sessions) vs L1-L3 (maximum history, daily rows, half-life
  504 / 1260 / none) on the same test rows = every session of the L0 CV folds before the final holdout;
- stage 2: W0-W5, W7 (section 6) and E1-E3 (event risk, section 7.1) on the maximum-history daily panel with the best
  half-life of L1-L3;
- holdout (last 24 months of sample dates, every session): once for L0 and the stage-1 winner, W0 and the stage-2
  winner. Fold fits and evaluations run in worker processes (`--workers`).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import pickle
import sys
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rsmt_ts as T

import quantcio.quant.rsmt as R
from quantcio.quant.panel import PanelData, market_config

log = logging.getLogger("rsmt_history")

LAYOUTS = (3, 4, 5)
HOLDOUT_MONTHS = 24
TIE_REL = 0.001
L0_STEP = 5  # the served model's training rows: every 5th session (rsmt.overlap_step)
EQUAL_WEIGHTS_HALF_LIFE = 1e9  # "none": 0.5 ** (age / 1e9) = 1 to 1e-5
MAX_HISTORY = {"KR": {"start_year": 1995, "history_years": 31}, "US": {"history_years": 17}}
CONFIGS: dict[str, dict[str, Any]] = {
    "L0": {"long": False, "half_life": None, "changes": 0, "label": "7 years, step 5, half-life 504 (served)"},
    "L1": {"long": True, "half_life": None, "changes": 2, "label": "maximum history, daily, half-life 504"},
    "L2": {"long": True, "half_life": 1260.0, "changes": 3, "label": "maximum history, daily, half-life 1260"},
    "L3": {"long": True, "half_life": EQUAL_WEIGHTS_HALF_LIFE, "changes": 3, "label": "maximum history, daily, equal"},
}
STAGE2 = ("W0", "W1", "W2", "W3", "W4", "W5", "W7", "E1", "E2", "E3")
METRIC_KEYS = ("crps", "base_crps", "crps_gain_p05", "cover80", "exc95", "auc", "base_auc", "ic")
# the pooled OOS arrays `quant.rsmt._pooled_metrics` reads (the rest is dropped in the worker: memory)
POOLED_KEYS = (
    "r",
    "m",
    "p_up",
    "base_p_up",
    "p_out",
    "base_p_out",
    "crps",
    "base_crps",
    "in80",
    "base_in80",
    "exc95",
    "base_exc95",
    "mu",
    "base_mu",
    "clamped",
    "base_clamped",
)
SMOKE = {"layouts": (3,), "stage2": ("W0", "E3"), "paths": 200}


def config_settings(base: Any, market: str, name: str) -> Any:
    """Settings of a stage-1 config; every config builds a DAILY panel (L0 trains on its every-5th-session subset)."""
    s = base.model_copy(deep=True)
    spec = CONFIGS[name]
    if spec["long"]:
        mx = MAX_HISTORY[market]
        s.history_years = int(mx["history_years"])
        if "start_year" in mx:
            s.pit.kr.start_year = int(mx["start_year"])
    if spec["half_life"] is not None:
        s.rsmt.markets[market].half_life_days = float(spec["half_life"])
    s.rsmt.overlap_step = 1
    s.rsmt.markets[market].overlap_step = None
    return s


def served_settings(base: Any, market: str) -> Any:
    """The L0 model as served (overlap step 5: row multiplicity / min_leaf counted for step-5 rows)."""
    s = config_settings(base, market, "L0")
    s.rsmt.overlap_step = L0_STEP
    return s


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _eligible(r: dict[str, Any]) -> bool:
    return (
        r.get("crps") is not None
        and r.get("cover80") is not None
        and r.get("exc95") is not None
        and T.COVER80_BAND[0] <= r["cover80"] <= T.COVER80_BAND[1]
        and T.EXC95_BAND[0] <= r["exc95"] <= T.EXC95_BAND[1]
    )


def select(rows: dict[str, dict[str, Any]], order_key: Any) -> tuple[str | None, list[str], str]:
    eligible = [n for n, r in rows.items() if _eligible(r)]
    if not eligible:
        return None, [], "no eligible config"
    best = min(rows[n]["crps"] for n in eligible)
    tied = sorted((n for n in eligible if (rows[n]["crps"] - best) / best < TIE_REL), key=order_key)
    lowest = min(eligible, key=lambda n: rows[n]["crps"])
    return tied[0], tied, f"eligible {eligible}; lowest {lowest} ({best:.6f}); tied within {TIE_REL:.1%}: {tied}"


def step_mask(train: pd.DataFrame, step: int) -> np.ndarray:
    """Rows on the every-`step`-session grid counted back from the latest labelled session (the grid of
    `build_panel(overlap_step=step)`: h is a multiple of step, so it contains the on_grid rows)."""
    cal = pd.DatetimeIndex(sorted(pd.unique(train["date"])))
    back = pd.Series(np.arange(len(cal))[::-1], index=cal)
    m = (back.reindex(pd.DatetimeIndex(train["date"])).to_numpy() % step) == 0
    if not m[train["on_grid"].to_numpy(dtype=bool)].all():
        raise RuntimeError("step grid does not contain the on_grid rows - calendar mismatch")
    return m


# ------------------------------------------------------------------------------------------ worker side

_WORK: dict[str, Any] = {}


def _init_worker(path: str) -> None:
    logging.basicConfig(level=logging.WARNING)
    with open(path, "rb") as fh:
        _WORK.update(pickle.load(fh))


@contextlib.contextmanager
def _variant(name: str | None) -> Iterator[None]:
    if name is None:
        yield
        return
    spec = T.VARIANTS[name]
    with T.installed(spec.get("sigma", ()), spec.get("groups", ())):
        yield


def _task(t: dict[str, Any]) -> dict[str, Any]:
    """One fold (or holdout / final fit): fit on rows with date < start and t_end < start (optionally the step-5
    subset), evaluate every session in [start, end] (optionally restricted to `keys`)."""
    p: PanelData = _WORK["panels"][t["panel"]]
    s = _WORK["settings"][t["settings"]]
    tr = p.train
    d = pd.to_datetime(tr["date"])
    start = pd.Timestamp(t["start"])
    trn_mask = ((d < start) & (pd.to_datetime(tr["t_end"]) < start)).to_numpy()
    if t.get("step5"):
        trn_mask &= _WORK["step5"][t["panel"]]
    trn = tr[trn_mask]
    t0 = time.perf_counter()
    with _variant(t.get("variant")):
        if t["kind"] == "fit":
            model = R.RSMTree.from_settings(s, p.market).fit_frame(trn)
            return {
                "key": t["key"],
                "splits": int(model.n_splits),
                "tree": model.describe("en"),
                "sigma_equation": model.sigma_equation()
                if isinstance(model, T.RSMTreeX) and model.extended
                else {"eta_v": float(model.eta)},
                "gamma": float(model.gamma),
                "seconds": round(time.perf_counter() - t0, 1),
            }
        te = tr[((d >= start) & (d <= pd.Timestamp(t["end"]))).to_numpy()]
        if t.get("keys") is not None:
            te = te.merge(t["keys"], on=["ticker", "date"], how="inner")
        ev, evb, prm, prm_b, model = R._fold_forecasts(p, s, trn, te, _WORK["n_paths"])
    pooled: dict[str, list[np.ndarray]] = {}
    R._pool_rows(pooled, ev, evb, prm, prm_b, te, int(t.get("fold", 0)))
    out = {k: np.concatenate(pooled[k]) for k in POOLED_KEYS}
    out["block"] = te["block"].to_numpy()
    return {
        "key": t["key"],
        "pooled": out,
        "splits": int(model.n_splits),
        "n_train": int(len(trn)),
        "n_test": int(len(te)),
        "train_from": str(pd.to_datetime(trn["date"]).min().date()),
        "tree": model.describe("en") if t.get("describe") else None,
        "seconds": round(time.perf_counter() - t0, 1),
    }


# ------------------------------------------------------------------------------------------ main side


def _metrics(parts: list[dict[str, np.ndarray]], s: Any, n_paths: int) -> dict[str, Any]:
    """Pooled metrics; the bootstrap clusters are h-session BLOCKS (daily test rows of one block share a window)."""
    P = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}  # noqa: N806 - the production name
    P["date"] = P["block"]
    m, _ = R._pooled_metrics(P, s, n_paths)
    ok, _ = R.distribution_gate(m, s.rsmt.skill_cover80_band, s.rsmt.skill_exc95_band)
    return {**{k: _f(m.get(k)) for k in METRIC_KEYS}, "n_oos": int(m.get("n_oos", 0)), "dist_gate": bool(ok)}


def _summarise(per_layout: dict[str, dict[str, Any]], layouts: tuple[int, ...] = LAYOUTS) -> dict[str, Any]:
    row: dict[str, Any] = {"layouts": per_layout}
    for key in ("crps", "base_crps", "cover80", "exc95", "auc", "base_auc", "ic"):
        vals = [per_layout[str(k)][key] for k in layouts if per_layout.get(str(k), {}).get(key) is not None]
        row[key] = float(np.mean(vals)) if len(vals) == len(layouts) else None
    row["dist_gate_layouts"] = sum(bool(per_layout[str(k)]["dist_gate"]) for k in layouts if str(k) in per_layout)
    return row


def _run_pool(
    tasks: list[dict[str, Any]], work: dict[str, Any], path: Path, workers: int, settings_of: Any, n_paths: int
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Run the tasks in worker processes (`work` pickled to `path` for their initializer). Evaluation results are
    pooled per group ('<stage>|<name>|<layout>') and reduced to metrics as soon as a group is complete.
    Returns (metrics by group, the other per-task results without their arrays)."""
    with open(path, "wb") as fh:
        pickle.dump(work, fh, protocol=pickle.HIGHEST_PROTOCOL)
    expected = Counter(t["key"].rsplit("|", 1)[0] for t in tasks if t["kind"] == "eval" and t.get("group", True))
    buffers: dict[str, list[dict[str, np.ndarray]]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    other: dict[str, dict[str, Any]] = {}
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(path),)) as pool:
            futs = [pool.submit(_task, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), start=1):
                res = fut.result()
                key = res["key"]
                log.info(
                    "task %d/%d %s: %ss, splits %s, train %s rows from %s, test %s",
                    i,
                    len(tasks),
                    key,
                    res["seconds"],
                    res["splits"],
                    res.get("n_train"),
                    res.get("train_from"),
                    res.get("n_test"),
                )
                group = key.rsplit("|", 1)[0]
                if "pooled" in res and group in expected:
                    buffers.setdefault(group, []).append(res.pop("pooled"))
                    other[key] = res
                    if len(buffers[group]) == expected[group]:
                        metrics[group] = _metrics(buffers.pop(group), settings_of(group), n_paths)
                else:
                    other[key] = res
    finally:
        path.unlink(missing_ok=True)
    return metrics, other


def run(market_code: str, out_dir: Path, n_paths: int, workers: int, smoke: bool = False) -> dict[str, Any]:
    from quantcio import holdout
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache
    from quantcio.models import Market
    from quantcio.pipeline import training_panel
    from quantcio.quant.panel import build_panel

    t_all = time.perf_counter()
    market = Market(market_code)
    mk = market.value
    layouts = SMOKE["layouts"] if smoke else LAYOUTS
    stage2_names = SMOKE["stage2"] if smoke else STAGE2
    base = load_settings()
    cache = DiskCache(base.cache_path)
    settings = {name: config_settings(base, mk, name) for name in CONFIGS}
    settings["L0_served"] = served_settings(base, mk)
    prices: dict[bool, Any] = {}
    guards: dict[str, Any] = {}
    for long in (False, True):
        s = settings["L1" if long else "L0"]
        price, _, _ = training_panel(market, s, cache)
        prices[long], guards[str(long)] = holdout.guard_panel(price, s)
    panels = {name: build_panel(prices[CONFIGS[name]["long"]], settings[name]) for name in CONFIGS}
    ref = panels["L0"]
    grid = pd.DatetimeIndex(sorted(ref.train.loc[ref.train["on_grid"], "date"].unique()))
    first_hold = grid[grid > grid[-1] - pd.DateOffset(months=HOLDOUT_MONTHS)][0]
    last_date = pd.Timestamp(ref.train["date"].max())
    info: dict[str, Any] = {
        "market": mk,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_digest": base.digest(),
        "holdout_start": str(first_hold.date()),
        "n_paths": n_paths,
        "workers": workers,
        "smoke": smoke,
        "holdout_guard": guards,
        "panels": {},
    }
    for name, p in panels.items():
        tr = p.train
        info["panels"][name] = {
            "rows": int(len(tr)),
            "sessions": int(tr["date"].nunique()),
            "grid_dates": int(tr.loc[tr["on_grid"], "date"].nunique()),
            "first_date": str(pd.to_datetime(tr["date"]).min().date()),
            "tickers": int(tr["ticker"].nunique()),
            "kish_dates": _kish_dates(tr),
        }
    log.info("%s panels: %s", mk, info["panels"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def s1_settings(group: str) -> Any:
        name = group.split("|")[1]
        return settings["L0_served" if name == "L0" else name]

    # ------------------------------------------------------------------- stage 1: common all-session test rows
    ref_pre = ref.train[(pd.to_datetime(ref.train["t_end"]) < first_hold).to_numpy()].reset_index(drop=True)
    step5 = {"L0": step_mask(ref.train, L0_STEP)}
    min_train = max(2 * market_config(settings["L0"], mk).min_leaf, 30)
    tasks: list[dict[str, Any]] = []
    for k in layouts:
        for fold in R.cv_splits(ref_pre, k):
            if int(ref_pre.iloc[fold.train_idx]["on_grid"].sum()) < min_train:
                continue
            d = pd.to_datetime(ref_pre["date"])
            keys = ref_pre.loc[((d >= fold.test_start) & (d <= fold.test_end)).to_numpy(), ["ticker", "date"]]
            for name in CONFIGS:
                tasks.append(
                    {
                        "key": f"s1|{name}|{k}|{fold.fold}",
                        "kind": "eval",
                        "panel": name,
                        "settings": "L0_served" if name == "L0" else name,
                        "step5": name == "L0",
                        "start": fold.test_start,
                        "end": fold.test_end,
                        "keys": keys,
                        "fold": fold.fold,
                    }
                )
    work = {"panels": panels, "settings": settings, "step5": step5, "n_paths": n_paths}
    m1, o1 = _run_pool(tasks, work, out_dir / f".history_{mk}_s1.pkl", workers, s1_settings, n_paths)
    n_test = {}
    for key, r in o1.items():
        n_test.setdefault(key.split("|", 2)[2], set()).add(r["n_test"])
    mismatch = {f: sorted(v) for f, v in n_test.items() if len(v) > 1}
    if mismatch:
        log.warning("%s stage1: test-row counts differ between configs: %s", mk, mismatch)
    info["stage1_test_rows_equal"] = not mismatch
    stage1: dict[str, dict[str, Any]] = {}
    for name in CONFIGS:
        per = {str(k): m1[f"s1|{name}|{k}"] for k in layouts}
        stage1[name] = {"label": CONFIGS[name]["label"], **_summarise(per, layouts)}
        log.info("%s stage1 %s: %s", mk, name, {k: stage1[name][k] for k in ("crps", "cover80", "exc95", "auc")})
    order = {n: (CONFIGS[n]["changes"], int(n[1:])) for n in CONFIGS}
    w1, tied1, why1 = select(stage1, lambda n: order[n])
    best_long, _, why_long = select({n: stage1[n] for n in ("L1", "L2", "L3")}, lambda n: order[n])
    best_long = best_long or "L1"
    log.info("%s stage1 winner %s (%s); best long %s (%s)", mk, w1, why1, best_long, why_long)
    hold1: dict[str, Any] = {}
    if not smoke:
        tasks = [
            {
                "key": f"h1|{name}|0",
                "kind": "eval",
                "panel": name,
                "settings": "L0_served" if name == "L0" else name,
                "step5": name == "L0",
                "start": first_hold,
                "end": last_date,
                "describe": True,
            }
            for name in dict.fromkeys(["L0", w1 or "L0"])
        ]
        hm, ho = _run_pool(tasks, work, out_dir / f".history_{mk}_h1.pkl", len(tasks), s1_settings, n_paths)
        for name in dict.fromkeys(["L0", w1 or "L0"]):
            r = ho[f"h1|{name}|0"]
            hold1[name] = {**hm[f"h1|{name}"], "splits": r["splits"], "tree": r["tree"]}
    del work

    # ------------------------------------------------------------------- stage 2: variants on the long daily panel
    events = T.load_events(base, cache)
    long_panel, counts = T.augment(panels[best_long], prices[True], events, mk)
    del panels
    ltr = long_panel.train
    long_pre = replace(
        long_panel, train=ltr[(pd.to_datetime(ltr["t_end"]) < first_hold).to_numpy()].reset_index(drop=True)
    )
    s2 = {name: T.variant_settings(settings[best_long], mk, name) for name in stage2_names}
    tasks = []
    for k in layouts:
        for fold in R.cv_splits(long_pre.train, k):
            if int(long_pre.train.iloc[fold.train_idx]["on_grid"].sum()) < min_train:
                continue
            for name in stage2_names:
                tasks.append(
                    {
                        "key": f"s2|{name}|{k}|{fold.fold}",
                        "kind": "eval",
                        "panel": "pre",
                        "settings": name,
                        "variant": name,
                        "start": fold.test_start,
                        "end": fold.test_end,
                        "fold": fold.fold,
                    }
                )
    for name in stage2_names:  # final pre-holdout fit: tree / sigma equation (start after every pre-holdout row)
        tasks.append(
            {
                "key": f"f2|{name}|0",
                "kind": "fit",
                "panel": "pre",
                "settings": name,
                "variant": name,
                "start": last_date,
            }
        )
    work = {"panels": {"pre": long_pre}, "settings": s2, "step5": {}, "n_paths": n_paths}
    m2, o2 = _run_pool(tasks, work, out_dir / f".history_{mk}_s2.pkl", workers, lambda g: s2[g.split("|")[1]], n_paths)
    stage2: dict[str, dict[str, Any]] = {}
    for name in stage2_names:
        per = {str(k): m2[f"s2|{name}|{k}"] for k in layouts}
        fit = o2[f"f2|{name}|0"]
        stage2[name] = {
            "label": T.VARIANTS[name]["label"],
            "components": T.VARIANTS[name]["components"],
            **_summarise(per, layouts),
            "splits_final": fit["splits"],
            "tree": fit["tree"],
            "sigma_equation": fit["sigma_equation"],
            "gamma": fit["gamma"],
        }
        log.info("%s stage2 %s: %s", mk, name, {k: stage2[name][k] for k in ("crps", "cover80", "exc95", "auc")})
    w2, tied2, why2 = select(stage2, lambda n: (T.VARIANTS[n]["components"], T.ORDER.index(n)))
    w2 = w2 or "W0"
    log.info("%s stage2 winner %s (%s)", mk, w2, why2)
    hold2: dict[str, Any] = {}
    if not smoke:
        names = list(dict.fromkeys(["W0", w2]))
        tasks = [
            {
                "key": f"h2|{name}|0",
                "kind": "eval",
                "panel": "full",
                "settings": name,
                "variant": name,
                "start": first_hold,
                "end": last_date,
                "describe": True,
            }
            for name in names
        ]
        work = {"panels": {"full": long_panel}, "settings": s2, "step5": {}, "n_paths": n_paths}
        hm, ho = _run_pool(
            tasks, work, out_dir / f".history_{mk}_h2.pkl", len(tasks), lambda g: s2[g.split("|")[1]], n_paths
        )
        for name in names:
            r = ho[f"h2|{name}|0"]
            hold2[name] = {**hm[f"h2|{name}"], "splits": r["splits"], "tree": r["tree"]}
    res = {
        "info": {**info, "augment": counts, "stage2_half_life_config": best_long},
        "stage1": stage1,
        "stage1_winner": w1,
        "stage1_tied": tied1,
        "stage1_why": why1,
        "stage1_best_long": best_long,
        "stage1_best_long_why": why_long,
        "holdout1": hold1,
        "stage2": stage2,
        "stage2_winner": w2,
        "stage2_tied": tied2,
        "stage2_why": why2,
        "holdout2": hold2,
        "seconds": round(time.perf_counter() - t_all, 1),
    }
    stem = f"history_{mk}" + ("_smoke" if smoke else "")
    (out_dir / f"{stem}.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(render(res), encoding="utf-8")
    return res


def _kish_dates(tr: pd.DataFrame) -> float:
    """Kish effective number of sample dates of the time-decay weights (one weight per on-grid date)."""
    g = tr[tr["on_grid"]].groupby("date")["w"].first()
    return round(float(g.sum() ** 2 / (g * g).sum()), 1) if len(g) else 0.0


def _n(x: Any, d: int = 4) -> str:
    v = _f(x)
    return "—" if v is None else f"{v:.{d}f}"


def render(res: dict[str, Any]) -> str:
    info = res["info"]
    md = [
        f"# RSMT-Q longest history, every session — {info['market']} (pre-declared, docs/RSMT.md §7 / §7.1)",
        "",
        f"- generated {info['generated_at']} · config digest `{info['config_digest']}` · holdout from "
        f"{info['holdout_start']} · {info['n_paths']} paths · guard {info['holdout_guard']}",
        "",
        "| panel | rows | sessions | 20-session dates | first date | tickers | Kish effective dates |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, p in info["panels"].items():
        md.append(
            f"| {name} | {p['rows']:,} | {p['sessions']:,} | {p['grid_dates']} | {p['first_date']} | {p['tickers']} | "
            f"{p['kish_dates']} |"
        )
    base = res["stage1"]["L0"]["crps"]
    md += [
        "",
        "## Stage 1 — same test rows (every session of the L0 CV folds)",
        "",
        "| config | CRPS | vs L0 | twin CRPS | cover80 | exc95 | AUC | μ IC | dist. gate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in res["stage1"].items():
        rel = "—" if r["crps"] is None or base is None else f"{(r['crps'] - base) / base:+.2%}"
        md.append(
            f"| **{name}** {r['label']} | {_n(r['crps'], 6)} | {rel} | {_n(r['base_crps'], 6)} | "
            f"{_n(r['cover80'], 3)} | {_n(r['exc95'], 3)} | {_n(r['auc'])} | {_n(r['ic'])} | "
            f"{r['dist_gate_layouts']}/{len(r['layouts'])} |"
        )
    md += [
        "",
        f"**Stage 1 winner: {res['stage1_winner']}** — {res['stage1_why']}; best long config "
        f"{res['stage1_best_long']} ({res['stage1_best_long_why']})",
        "",
    ]
    base2 = res["stage2"]["W0"]["crps"]
    md += [
        f"## Stage 2 — variants on the long daily panel ({info['stage2_half_life_config']}), every-session CV",
        "",
        "| variant | CRPS | vs W0 | twin CRPS | cover80 | exc95 | AUC | μ IC | dist. gate | splits | sigma equation |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in res["stage2"].items():
        rel = "—" if r["crps"] is None or base2 is None else f"{(r['crps'] - base2) / base2:+.2%}"
        eq = ", ".join(f"{k} {v:+.3f}" for k, v in r["sigma_equation"].items() if v is not None)
        md.append(
            f"| **{name}** {r['label']} | {_n(r['crps'], 6)} | {rel} | {_n(r['base_crps'], 6)} | "
            f"{_n(r['cover80'], 3)} | {_n(r['exc95'], 3)} | {_n(r['auc'])} | {_n(r['ic'])} | "
            f"{r['dist_gate_layouts']}/{len(r['layouts'])} | {r['splits_final']} | {eq} |"
        )
    md += ["", f"**Stage 2 winner: {res['stage2_winner']}** — {res['stage2_why']}", "", "## Final holdout (once)", ""]
    md += [
        "| run | CRPS | twin CRPS | gain bound | cover80 | exc95 | AUC | μ IC | splits |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tag, hold in (("stage 1", res["holdout1"]), ("stage 2", res["holdout2"])):
        for name, h in hold.items():
            md.append(
                f"| {tag} {name} | {_n(h['crps'], 6)} | {_n(h['base_crps'], 6)} | {_n(h['crps_gain_p05'], 6)} | "
                f"{_n(h['cover80'], 3)} | {_n(h['exc95'], 3)} | {_n(h['auc'])} | {_n(h['ic'])} | {h['splits']} |"
            )
    md += ["", "## Trees", ""]
    for name, r in res["stage2"].items():
        md += [f"**{name}**", "", *[f"    {ln}" for ln in r["tree"]], ""]
    for tag, hold in (("holdout stage 1", res["holdout1"]), ("holdout stage 2", res["holdout2"])):
        for name, h in hold.items():
            md += [f"**{tag} {name}**", "", *[f"    {ln}" for ln in (h["tree"] or [])], ""]
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--market", required=True, choices=["KR", "US"])
    ap.add_argument("--out", default="reports/rsmt")
    ap.add_argument("--paths", type=int, default=R.CV_PATHS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--smoke", action="store_true", help="layout 3, W0 / E3, no holdout (crash check, not the comparison)"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    paths = SMOKE["paths"] if args.smoke else args.paths
    res = run(args.market, Path(args.out), paths, args.workers, args.smoke)
    print(f"{args.market}: stage1 {res['stage1_winner']}, stage2 {res['stage2_winner']} in {res['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
