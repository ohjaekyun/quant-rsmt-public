"""Serve the US v9 model (docs/MODELING.md section 26, user decision 2026-10-05): CatBoost YetiRank (``cat_rank``) on
the v9 inputs, refitted with the unchanged protocol and saved as the US serving artifact (the cache reports read).
Also writes the forward picks of the latest cross-section. Re-run this (not ``quantcio modeling select --market US``,
which would rebuild the v3 model) to refresh the US model, e.g. monthly.

    .venv/bin/python scripts/us_v9_serve.py

Writes reports/modeling/US_v9_serving.json and reports/strategy/forward/us_v9_picks_<as_of>.csv.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import us_v9_study as v9  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.serve import build_artifact, load_artifact, save_artifact, served_state  # noqa: E402

MODEL = "cat_rank"
OUT = PROJECT_ROOT / "reports" / "modeling"
FWD = PROJECT_ROOT / "reports" / "strategy" / "forward"


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    ds, info = v9.build_inputs(settings, cache)
    every = {c.name: c for c in cmod.default_candidates()}
    sel.COMPOSITE_CANDIDATES = ()
    sel.COMPOSITES = ()
    sel.DIAGNOSTIC_PAIRS = ()
    res = sel.run_selection(ds, settings, cache, candidates=[every["mom_12_1"], every[MODEL]], threads=8, progress=say)
    if res.selected != MODEL:
        raise SystemExit(f"선택 결과가 {res.selected} — {MODEL}이 기준 모델보다 낫지 않아 서빙을 바꾸지 않습니다")
    prev = load_artifact("US", settings, cache)
    art = build_artifact(res, ds, settings)
    art.notes = [
        {
            "ko": "미국 v9 모델(CatBoost 순위, v9 입력 157개) — 사용자 결정으로 서빙(docs/MODELING.md 26). "
            "후보 33개 탐색의 최댓값이라 실력은 전진 기록으로 확인 중입니다.",
            "en": "US v9 model (CatBoost YetiRank, 157 v9 inputs) served by user decision (docs/MODELING.md 26); "
            "it is the best of a 33-candidate search - skill is being confirmed forward.",
        }
    ]
    save_artifact(art, settings, cache)
    state = served_state(art, settings)
    cand = res.candidate(MODEL)
    o, h = cand["outer"]["auc_median"], (cand["holdout"] or {}).get("auc_median") or {}
    summary = {
        "served_at": art.created_at,
        "as_of": str(art.as_of.date()),
        "model": MODEL,
        "previous_model": getattr(prev, "selected", None),
        "previous_as_of": str(getattr(prev, "as_of", "")),
        "outer_auc": o,
        "holdout_auc": h,
        "gate": res.gate,
        "calibration": res.calibration,
        "served_state": state,
        "inputs": info["inputs"],
        "final_config": (cand.get("configs") or {}).get("final"),
        "importances_top20": dict(list(res.importances.items())[:20]),
    }
    (OUT / "US_v9_serving.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    # forward picks of the latest cross-section (append-only record; scored later by us_v9_forward.py)
    lat = art.latest.copy()
    lat["quintile"] = pd.qcut(lat["percentile"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    FWD.mkdir(parents=True, exist_ok=True)
    cols = ["date", "entity", "ticker", "name", "sector_group", "score", "percentile", "quintile", "mu_excess"]
    lat[[c for c in cols if c in lat]].sort_values("percentile", ascending=False).to_csv(
        FWD / f"us_v9_picks_{art.as_of.date()}.csv", index=False
    )
    say(
        f"served {MODEL} as_of {art.as_of.date()}: outer {o['mean']:.4f} [{o['lo']:.4f}], holdout {h.get('mean')}, "
        f"state {state['status']}, multiplier {state['multiplier']:.4f}; previous {summary['previous_model']}"
    )


if __name__ == "__main__":
    main()
