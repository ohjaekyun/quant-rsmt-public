"""Serve a market's stock-selection model by a user rule instead of the protocol's outer-AUC rule
(docs/MODELING.md 74, user decision 2026-10-22). Research / operations script - changes no file under src/quantcio.

    .venv/bin/python scripts/serve_override.py --market US --rule holdout

Runs the same selection as `quantcio modeling select --market <M>`, then serves the eligible (non-diagnostic)
candidate with the best HOLDOUT primary AUC: its own serving gate, calibration and importances are recomputed, its
label says it is a user override, and the artifact + selection summary are cached exactly where `modeling select`
puts them. Using the holdout to choose spends it: from now on only the forward holdout tests this choice.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import pandas as pd

from quantcio.config import load_settings
from quantcio.data.cache import DiskCache
from quantcio.modeling import selection as sel
from quantcio.modeling.cli import guard_xsh, serving_summary
from quantcio.modeling.dataset import build_dataset
from quantcio.modeling.report import summary_lines, write_report
from quantcio.modeling.serve import build_artifact, save_artifact
from quantcio.models import Market


def holdout_best(res: sel.SelectionResult) -> str:
    key = sel.PRIMARY  # protocol v4: the money metric (docs/MODELING.md 79)
    rows = [c for c in res.candidates if not c.get("diagnostic") and c["name"] != sel.BASELINE]
    scored = [(((c.get("holdout") or {}).get(key) or {}).get("mean"), c["name"]) for c in rows]
    scored = [(v, n) for v, n in scored if v is not None]
    return max(scored)[1]


def override(res: sel.SelectionResult, name: str, rule: str) -> sel.SelectionResult:
    prev = res.selected
    res.selected = name
    res.gate = sel.serving_gate(res.candidate(name)["outer"], res.candidate(sel.BASELINE)["outer"])
    res.gate["override"] = {"rule": rule, "protocol_choice": prev, "decided": "user, 2026-10-22 (docs/MODELING.md 74)"}
    allo = pd.concat([res.oos, res.holdout_preds], ignore_index=True)
    res.calibration = sel.calibration(allo, name)
    imp: dict[str, float] = {}
    members = sel.composite_members(name, res.final.configs)
    for m in members:
        f = res.final.fitted.get(m)
        if f is None:
            continue
        raw = f.importance()
        tot = sum(abs(v) for v in raw.values()) or 1.0
        for k, v in raw.items():
            imp[k] = imp.get(k, 0.0) + abs(v) / tot / max(len(members), 1)
    res.importances = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:20])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    ap.add_argument("--rule", choices=["holdout"], default="holdout")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    s = load_settings()
    cache = DiskCache(s.cache_path)
    mk = Market(a.market)
    t0 = time.monotonic()

    def say(msg: str) -> None:
        print(f"[{time.monotonic() - t0:7.1f}s] {msg}", flush=True)

    ds = build_dataset(mk, s, cache, use_cache=True, progress=say)
    ds = guard_xsh(ds, s, say)
    res = sel.run_selection(ds, s, cache, threads=a.threads, use_cache=True, progress=say)
    choice = holdout_best(res)
    say(f"protocol choice {res.selected}; user rule '{a.rule}' -> {choice}")
    res = override(res, choice, a.rule)
    art = build_artifact(res, ds, s)
    art.label_ko = (
        f"{art.label_ko} · 사용자 지정(홀드아웃 1위, 사전 규칙 선택은 {res.gate['override']['protocol_choice']})"
    )
    art.label_en = (
        f"{art.label_en} · user override (best holdout; protocol pick {res.gate['override']['protocol_choice']})"
    )
    art.notes = [*art.notes, "served by the user rule 'best holdout AUC' (docs/MODELING.md 74), not the protocol rule"]
    serving = serving_summary(art)
    md, js = write_report(res, s, None, serving)
    art.report_path = str(md)
    save_artifact(art, s, cache)
    light = res.to_dict()
    light["summary_ko"], light["summary_en"] = summary_lines(res)
    light["serving"] = serving
    light["report"] = {"md": str(md), "json": str(js)}
    cache.set(sel.RESULT_NS, mk.value, light)
    o, h = res.candidate(choice)["outer"], res.candidate(choice)["holdout"] or {}
    k = sel.PRIMARY
    print(
        f"served {mk.value}: {choice} — outer {k} {o[k]['mean']:.5f}/month (lo {o[k]['lo']:.5f}), holdout "
        f"{(h.get(k) or {}).get('mean')}, AUC {o[sel.PRIMARY_AUC]['mean']:.4f}, gate passed {res.gate['passed']}, "
        f"multiplier {art.multiplier:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
