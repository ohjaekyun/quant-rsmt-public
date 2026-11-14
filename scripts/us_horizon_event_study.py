"""US horizon extension and earnings-event target (docs/MODELING.md 44, pre-declared 2026-10-12). Research only.

    .venv/bin/python scripts/us_horizon_event_study.py --part h60      # v9 inputs, h = 60, month-end evaluation
    .venv/bin/python scripts/us_horizon_event_study.py --part h120
    .venv/bin/python scripts/us_horizon_event_study.py --part event    # rows expecting earnings in 5 sessions, h = 10

Writes reports/modeling/US_44_<part>.json and reports/modeling/US_44_<part>_summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import us_weekly_study as wk  # noqa: E402
import v13_study as v13s  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
MODELS = ("cat_rank", "ridge", "logit_w", "lgbm_clf", "et")
MIN_EVENT_NAMES = 30
EVENT_H = 10


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def candidates() -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    return [every["mom_12_1"], every["factor_ic"], *(every[x] for x in MODELS)]


def judge(d: dict) -> tuple[list[str], bool]:
    cand = {c["name"]: c for c in d["candidates"]}
    s = cand[d["selected"]]
    so, sh = s["outer"]["auc_median"], (s["holdout"] or {}).get("auc_median") or {}
    mom = cand["mom_12_1"]["outer"]["auc_median"]
    ok1 = so["lo"] > 0.5 and so["mean"] > mom["mean"] and (sh.get("mean") or 0) >= 0.5
    tn, htn = s["outer"].get("top_net") or {}, (s["holdout"] or {}).get("top_net") or {}
    ok2 = (tn.get("mean") or 0) > 0 and (tn.get("t") or 0) >= 2 and (htn.get("mean") or 0) >= 0
    nan = float("nan")
    lines = [
        f"1. 실력(`{d['selected']}`): 외부 {so['mean']:.4f} [{so['lo']:.4f}, {so['hi']:.4f}]"
        f", 모멘텀 {mom['mean']:.4f}, 홀드아웃 {sh.get('mean', nan):.4f} → **{'통과' if ok1 else '미달'}**",
        f"2. 비용 후 상위 5분위 순초과(평가 기간당): 외부 {(tn.get('mean') or nan):+.3%}"
        f" (t {(tn.get('t') or nan):+.2f}, "
        f"95% [{(tn.get('lo') or nan):+.3%}, {(tn.get('hi') or nan):+.3%}]), 홀드아웃 {(htn.get('mean') or nan):+.3%} "
        f"→ **{'통과' if ok2 else '미달'}**",
    ]
    return lines, ok1 and ok2


def write_summary(part: str) -> str:
    d = json.loads((OUT / f"US_44_{part}.json").read_text())
    info = d["protocol"]["p44"]
    lines = [f"# US 44절 — {part}", "", f"- 설정: {json.dumps(info, ensure_ascii=False, default=str)[:600]}", ""]
    lines += [
        "| 후보 | 외부 AUC [95% CI] | 외부 IC (t) | 홀드아웃 AUC | 순초과/기간 (t) | 회전율 |",
        "|---|---|---|---|---|---|",
    ]
    for c in d["candidates"]:
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        tn = c["outer"].get("top_net") or {}
        hm = f"{h['mean']:.4f}" if h.get("mean") is not None else "–"
        tt = f"{tn['mean']:+.3%} ({tn['t']:+.2f})" if tn.get("mean") is not None and tn.get("t") is not None else "–"
        lines.append(
            f"| {c['name']} | {o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] | {c['outer']['ic']['mean']:+.4f} "
            f"({c['outer']['ic']['t']:+.2f}) | {hm} | {tt} | {c['outer'].get('turnover') or 0:.2f} |"
        )
    jl, ok = judge(d)
    lines += ["", "## 판정 (44)", "", *jl, "", f"**종합: {'후보 — 전진 기록부터' if ok else '후보 아님'}**"]
    if part != "event":
        lines.append(
            "- 주의: 월말 정답이 서로 겹쳐(h > 20) t는 과대할 수 있으므로"
            " 부트스트랩 구간(블록 = h/20개월)을 함께 봅니다."
        )
    text = "\n".join(lines) + "\n"
    (OUT / f"US_44_{part}_summary.md").write_text(text, encoding="utf-8")
    return text


def run_horizon(h: int) -> None:
    settings = load_settings(horizon_days=h)
    cache = DiskCache(settings.cache_path)
    _, ds = v13s.load_inputs("US", settings, cache)
    ds.snapshot = f"{ds.snapshot}|p44|h{h}"
    mt.block_bootstrap_means.__defaults__ = (max(3, h // 20), mt.BOOT_SIMS, 0)
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    sel.DIAGNOSTIC_PAIRS = ()
    res = sel.run_selection(ds, settings, cache, candidates=candidates(), threads=8, progress=say)
    res.protocol["p44"] = {"declared": "docs/MODELING.md 44", "horizon": h, "boot_block_months": max(3, h // 20)}
    part = f"h{h}"
    (OUT / f"US_44_{part}.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    print(write_summary(part), flush=True)


def run_event() -> None:
    settings = load_settings(horizon_days=EVENT_H)
    cache = DiskCache(settings.cache_path)
    uni, ds = v13s.load_inputs("US", settings, cache)
    f = ds.frame
    base_xs = list(ds.xs_cols)
    say("block W (short horizon + earnings timing)")
    w, info = wk.block_w(ds, uni, settings, cache)
    raw = []
    for c in w.columns:
        vals = w[c].astype(float)
        if vals.notna().sum() == 0 or vals.nunique() < 2:
            continue
        f[c] = vals.to_numpy()
        f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
        raw.append(c)
    week_end = wk.weekly_eval_rows(f)
    event = (f["w_earn_next"] == 1).to_numpy()
    keep = event & week_end
    ds.frame = f = f[keep].reset_index(drop=True)
    from quantcio.modeling.dataset import add_date_targets

    for c in raw:  # re-rank within the event cross-section
        f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
    add_date_targets(f)
    per_week = f[f["labelled"]].groupby("date").size()
    season = set(per_week[per_week >= MIN_EVENT_NAMES].index)
    f["is_me"] = f["date"].isin(season).to_numpy()
    years = (max(season) - min(season)).days / 365.25 if season else 1.0
    c = len(season) / max(years, 1.0)  # evaluation weeks per year
    spans = {
        "HOLDOUT_MONTHS": int(round(2 * c)),
        "OUTER_MIN_TRAIN_MONTHS": int(round(3 * c)),
        "OUTER_BLOCK_MONTHS": int(round(c)),
        "INNER_BLOCK_MONTHS": int(round(c)),
        "INNER_MIN_TRAIN_MONTHS": int(round(c)),
        "STACK_MIN_TRAIN_MONTHS": int(round(2 * c)),
    }
    for k, v in spans.items():
        setattr(sel, k, v)
    mt.PERIODS_PER_YEAR = c
    mt.block_bootstrap_means.__defaults__ = (max(2, int(round(c / 4))), mt.BOOT_SIMS, 0)
    ds.xs_cols = [*base_xs, *[feat.x_col(x) for x in raw]]
    ds.core_xs_cols = base_xs
    ds.snapshot = f"{ds.snapshot}|p44|event|{len(raw)}"
    info.update(
        {
            "rows": int(len(f)),
            "season_weeks": len(season),
            "weeks_per_year": round(c, 1),
            "median_names": float(per_week[per_week >= MIN_EVENT_NAMES].median()) if season else 0.0,
            "spans": spans,
        }
    )
    say(f"event rows {json.dumps(info, default=str)}")
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    sel.DIAGNOSTIC_PAIRS = ()
    res = sel.run_selection(ds, settings, cache, candidates=candidates(), threads=8, progress=say)
    res.protocol["p44"] = {"declared": "docs/MODELING.md 44", "horizon": EVENT_H, **info}
    (OUT / "US_44_event.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    print(write_summary("event"), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["h60", "h120", "event"], required=True)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        print(write_summary(a.part))
    elif a.part == "event":
        run_event()
    else:
        run_horizon(int(a.part[1:]))


if __name__ == "__main__":
    np.seterr(all="ignore")
    pd.set_option("mode.chained_assignment", None)
    main()
