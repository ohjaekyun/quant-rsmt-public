"""New model families on the served inputs (docs/MODELING.md 62, pre-declared 2026-10-16). Research only.

    .venv/bin/python scripts/model_zoo_study.py --market KR

Writes reports/modeling/<M>_zoo62.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_select_study as fss  # noqa: E402
import model_zoo as zoo  # noqa: E402
import v13_study as v13s  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
_f = fss._f


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    m = ap.parse_args().market
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    uni, ds = v13s.load_inputs(m, settings, cache)
    ds.core_xs_cols, ds.core_regime_cols = list(ds.xs_cols), list(ds.regime_cols)
    ds.snapshot = f"{ds.snapshot}|zoo62"
    sel.COMPOSITE_CANDIDATES, sel.COMPOSITES, sel.DIAGNOSTIC_PAIRS = (), (), ()
    every = {c.name: c for c in cmod.default_candidates()}
    served = fss.SERVED[m]
    cands = [every["mom_12_1"], every[served], *zoo.candidates()]
    say(f"{m}: {len(ds.xs_cols)} inputs, candidates {[c.name for c in cands]}")
    res = sel.run_selection(ds, settings, cache, candidates=cands, threads=8, progress=say)
    d = res.to_dict()
    best = d["selected"]
    vs = {}
    for c in cands[2:]:
        if c.name in res.oos:
            vs[c.name] = {
                "outer": mt.paired_difference(
                    mt.per_date_primary(res.oos, c.name), mt.per_date_primary(res.oos, served)
                ),
                "holdout": mt.paired_difference(
                    mt.per_date_primary(res.holdout_preds, c.name), mt.per_date_primary(res.holdout_preds, served)
                ),
            }
    sw = vs.get(best)
    switch = best if sw and (sw["outer"].get("lo") or -1) > 0 and (sw["holdout"].get("mean") or -1) >= 0 else None
    r = {"market": m, "selection": d, "vs_served": vs, "switch": switch}
    try:
        r["portfolio"] = fss.portfolio(m, res, uni, ds, best)
    except Exception as exc:  # noqa: BLE001 - the selection result stands without the portfolio table
        say(f"portfolio skipped: {type(exc).__name__}: {exc}")
    lines = [
        f"# {m} 새 모델 계열 (docs/MODELING.md 62) — 입력 {len(ds.xs_cols)}개",
        "",
        "| 후보 | 외부 AUC [95% CI] | 외부 IC (t) | 홀드아웃 AUC | − 서빙 구성 (외부 [CI] · 홀드아웃) |",
        "|---|---|---|---|---|",
    ]
    for c in d["candidates"]:
        o, h, ic = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}, c["outer"]["ic"]
        x = vs.get(c["name"])
        diff = (
            f"{_f(x['outer']['mean'])} [{_f(x['outer']['lo'])}, {_f(x['outer']['hi'])}] · "
            f"{_f(x['holdout'].get('mean'))}"
            if x
            else "—"
        )
        star = " **(1위)**" if c["name"] == best else ""
        lines.append(
            f"| {c['name']}{star} | {_f(o['mean'], '.4f')} [{_f(o['lo'], '.4f')}, {_f(o['hi'], '.4f')}] | "
            f"{_f(ic['mean'])} ({_f(ic['t'], '+.2f')}) | {_f(h.get('mean'), '.4f')} | {diff} |"
        )
    g = d["gate"]
    lines += [
        "",
        f"- 1위 `{best}` · 게이트 {'통과' if g['passed'] else '미달'} · 서빙 교체 후보: **{switch or '없음'}**",
    ]
    for k, s in (r.get("portfolio") or {}).items():
        lines.append(
            f"- 포트폴리오 {k}: 연 {_f(s['cagr'], '+.1%')}, 기준 대비 {_f(s['excess_ann'], '+.1%')} "
            f"(t {_f(s['excess_t'], '+.2f')})"
        )
    text = "\n".join(lines) + "\n"
    (OUT / f"{m}_zoo62.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    (OUT / f"{m}_zoo62.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
