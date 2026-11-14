"""Second look at the S&P 400 (docs/MODELING.md 26.3): the v9 ``cat_rank`` refitted per context on S&P 500 rows only
(the context's inner-CV configuration from the v9 run) scores every S&P 400 point-in-time member row of that context's
test month-ends (all members, from the verified PIT start). The S&P 400 sample was already used (v4 18.5, v5 R4/R5),
so this is NOT a pristine confirmation. Research only.

    .venv/bin/python scripts/us_v9_sp400.py

Writes reports/modeling/US_v9_sp400.{md,json}.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_opt_study as fo  # noqa: E402
import us_v9_features as v9f  # noqa: E402
import us_v9_study as v9  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.data.pit import load_pit_fundamentals  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.dataset import build_dataset  # noqa: E402
from quantcio.models import Market  # noqa: E402

MODEL = "cat_rank"
OUT = PROJECT_ROOT / "reports" / "modeling"


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    from quantcio.data.alt.load import load_alt_data
    from quantcio.data.pit.us_midcap import load_us_midcap_universe

    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    ds, _info = v9.build_inputs(settings, cache)
    data = sel._Data(ds)
    outer, hold, _final, _ = sel.make_contexts(data)
    result = json.loads((OUT / "US_v9.json").read_text())
    cfgs = next(c for c in result["candidates"] if c["name"] == MODEL)["configs"]
    say("S&P 400 universe (all PIT members)")
    uni = load_us_midcap_universe(settings, cache, date(2012, 1, 1), None, include_sp500=True)
    pf = pd.Timestamp((uni.coverage or {}).get("pit_from") or "2017-07-31")
    uni.membership = uni.membership[uni.membership["date"] >= pf].reset_index(drop=True)
    ents = sorted(uni.membership["entity"].astype(str).unique())
    alt = load_alt_data(Market.US, settings, ents, date(2011, 1, 1), progress=say)
    xds = build_dataset(Market.US, settings, cache, universe=uni, alt=alt, progress=say)
    fund = load_pit_fundamentals(Market.US, settings, cache, ents)
    fo.us_blocks(xds, settings, cache, universe=uni, fundamentals=fund)
    xf = xds.frame
    for r in v9.SETS["all_reg"]:  # date-level regime values of the S&P 500 dataset (as XSH does)
        xf[r] = xf["date"].map(ds.frame.groupby("date")[r].first())
    v9f.engineer(xf)
    missing = [c for c in v9.SETS["all_xs"] if c not in xf]
    if missing:
        say(f"inputs missing in the S&P 400 frame (NaN): {missing[:10]}{'…' if len(missing) > 10 else ''}")
        for c in missing:
            xf[c] = float("nan")
    cand = {c.name: c for c in cmod.default_candidates()}[MODEL]
    scored = []
    for ctx in [*outer, hold]:
        rows = xf[xf["labelled"] & xf["is_me"] & xf["date"].isin(ctx.test_dates) & (xf["date"] >= pf)]
        if rows.empty:
            continue
        mc = cfgs.get(ctx.name) or {}
        tr = data.train(ctx.train_end, cand.rows)
        fitted = cand.fit(
            tr,
            dict(mc.get("params") or cand.grid[0]),
            str(mc.get("scheme") or "none"),
            sel._ctx(data, ctx.test_dates[0], settings, 8),
        )
        scored.append(
            rows[["date", "entity", "y_e", "r", "x_mom_12_1"]].assign(score=fitted.predict(rows), part=ctx.name)
        )
        say(f"{ctx.name}: {len(rows)} S&P 400 rows")
    frame = pd.concat(scored, ignore_index=True)
    res = {}
    for part, sub in (("outer", frame[frame["part"] != "holdout"]), ("holdout", frame[frame["part"] == "holdout"])):
        if sub.empty:
            continue
        m = mt.evaluate(sub, 10.0, "score")
        b = mt.evaluate(sub.assign(mom=sub["x_mom_12_1"]), 10.0, "mom")
        res[part] = {
            "model": m,
            "momentum": b,
            "dates": int(sub["date"].nunique()),
            "names": int(sub["entity"].nunique()),
        }
    (OUT / "US_v9_sp400.json").write_text(json.dumps(res, default=str, indent=1))
    lines = [
        "# 미국 v9 모델 — S&P 400 두 번째 평가 (docs/MODELING.md 26.3)",
        "",
        f"- 모델 `{MODEL}`(문맥마다 S&P 500 행으로만 다시 학습), S&P 400 전체 시점별 구성원 {pf.date()} 이후",
        "- 주의: S&P 400 표본은 v4·v5에서 이미 썼으므로 미열람 확인이 아닙니다.",
        "",
        "| 구간 | 날짜 | 종목 | AUC(중앙값 초과) [95% CI] | 순위 IC (t) | 12-1 모멘텀 AUC | 상위 20% 순초과(월) |",
        "|---|---|---|---|---|---|---|",
    ]
    for part, r in res.items():
        a, ic, tn = r["model"][mt.PRIMARY_AUC], r["model"]["ic"], r["model"]["top_net"]
        lines.append(
            f"| {part} | {r['dates']} | {r['names']} | {a['mean']:.4f} [{a['lo']:.4f}, {a['hi']:.4f}] | "
            f"{ic['mean']:+.4f} ({(ic['t'] or 0):+.2f}) | {r['momentum'][mt.PRIMARY_AUC]['mean']:.4f} | "
            f"{(tn['mean'] or 0) * 100:+.2f}% (t {(tn['t'] or 0):+.2f}) |"
        )
    (OUT / "US_v9_sp400.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
