"""KR RSMT-Q tail diagnosis (docs/MODELING.md 36, 2026-10-10 - a post-hoc diagnostic: it was triggered by the 3.5%
VaR95 exceedance of the last 24 months found in section 34). The adopted model (P of scripts/rsmt_np_study.py) is
scored on layout 5 of RSMT-Q's own CV and on the last 24 months; per test row: lower / upper tail exceedance
(r < q0.05, r > q0.95), 80% cover, and the standardised squared error z^2 with z = (r - q0.5) / ((q0.9 - q0.1) / 2.5631)
(1 = calibrated width).

    .venv/bin/python scripts/rsmt_tail_check.py --market KR

Writes reports/rsmt/tail_check_<M>.{md,json} and the per-row parquet.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import rsmt_np_study as np_study

OUT = Path(__file__).resolve().parents[1] / "reports" / "rsmt"
Z80 = 2.5631  # (q0.9 - q0.1) of a standard normal


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rows_of(model, panel, s, trn, te) -> pd.DataFrame:
    p = np_study.model_quantiles(model, panel, s, trn, te)
    q = p["grid"]
    r = te["r"].to_numpy(dtype="float64")
    lv = list(np_study.LEVELS)
    q05, q10, q50, q90, q95 = (q[:, lv.index(x)] for x in (0.05, 0.10, 0.50, 0.90, 0.95))
    z = (r - q50) / ((q90 - q10) / Z80)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(te["date"]).to_numpy(),
            "ticker": te["ticker"].to_numpy(),
            "r": r,
            "m": te["m"].to_numpy(dtype="float64"),
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "lo_exc": (r < q05).astype(float),
            "hi_exc": (r > q95).astype(float),
            "in80": ((q10 <= r) & (r <= q90)).astype(float),
            "z2": z**2,
        }
    )


def run(market: str) -> dict:
    import quantcio.quant.rsmt as R
    from quantcio import holdout
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache
    from quantcio.models import Market
    from quantcio.pipeline import training_panel
    from quantcio.quant.panel import build_panel

    s = load_settings()
    cache = DiskCache(s.cache_path)
    price, _, _ = training_panel(Market(market), s, cache)
    price, _ = holdout.guard_panel(price, s)
    panel = build_panel(price, s)
    tr = panel.train
    grid = pd.DatetimeIndex(sorted(tr.loc[tr["on_grid"], "date"].unique()))
    first_hold = grid[grid > grid[-1] - pd.DateOffset(months=np_study.HOLDOUT_MONTHS)][0]
    pre = tr[(pd.to_datetime(tr["t_end"]) < first_hold).to_numpy()].reset_index(drop=True)
    parts = []
    for fold in R.cv_splits(pre, 5):
        trn, te = pre.iloc[fold.train_idx], pre.iloc[fold.test_idx]
        if len(te) == 0 or trn["on_grid"].sum() < 60:
            continue
        model = R.RSMTree.from_settings(s, panel.market).fit_frame(trn)
        parts.append(rows_of(model, panel, s, trn, te).assign(part="cv"))
        say(f"{market} fold {fold.fold}: {len(te)} rows")
    d = pd.to_datetime(tr["date"])
    trn = tr[((d < first_hold) & (pd.to_datetime(tr["t_end"]) < first_hold)).to_numpy()]
    te = tr[((d >= first_hold) & tr["on_grid"].to_numpy(dtype=bool)).to_numpy() & tr["r"].notna().to_numpy()]
    model = R.RSMTree.from_settings(s, panel.market).fit_frame(trn)
    parts.append(rows_of(model, panel, s, trn, te).assign(part="holdout"))
    rows = pd.concat(parts, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUT / f"tail_check_{market}.parquet", index=False)
    # rolling 24-month windows over the CV test history (step 1 month) -> the natural range of the lower-tail rate
    cv = rows[rows["part"] == "cv"]
    months = pd.DatetimeIndex(sorted(cv["date"].dt.to_period("M").dt.to_timestamp().unique()))
    roll = []
    for i in range(len(months) - 23):
        w = cv[(cv["date"] >= months[i]) & (cv["date"] < months[i + 23] + pd.offsets.MonthEnd(1))]
        if len(w) >= 500:
            roll.append(
                {
                    "start": str(months[i].date()),
                    "lo": float(w["lo_exc"].mean()),
                    "hi": float(w["hi_exc"].mean()),
                    "mkt": float(w.groupby("date")["m"].mean().mean()),
                }
            )
    rl = pd.DataFrame(roll)
    hold = rows[rows["part"] == "holdout"]

    def stat(x: pd.DataFrame) -> dict:
        return {
            "n": int(len(x)),
            "lo_exc": float(x["lo_exc"].mean()),
            "hi_exc": float(x["hi_exc"].mean()),
            "cover80": float(x["in80"].mean()),
            "z2": float(x["z2"].mean()),
            "mkt_mean_m": float(x.groupby("date")["m"].mean().mean()),
        }

    by_year = {str(y): stat(g) for y, g in rows.groupby(rows["date"].dt.year)}
    res = {
        "market": market,
        "holdout_start": str(first_hold.date()),
        "cv": stat(cv),
        "holdout": stat(hold),
        "by_year": by_year,
        "rolling24": {
            "n_windows": int(len(rl)),
            "lo_p05": float(rl["lo"].quantile(0.05)) if len(rl) else None,
            "lo_p95": float(rl["lo"].quantile(0.95)) if len(rl) else None,
            "hi_p05": float(rl["hi"].quantile(0.05)) if len(rl) else None,
            "hi_p95": float(rl["hi"].quantile(0.95)) if len(rl) else None,
            "corr_lo_mkt": float(rl[["lo", "mkt"]].corr().iloc[0, 1]) if len(rl) > 2 else None,
        },
    }
    h, rr = res["holdout"], res["rolling24"]
    res["verdict"] = {
        "lo_within_range": bool(rr["lo_p05"] is not None and rr["lo_p05"] <= h["lo_exc"] <= rr["lo_p95"]),
        "location_shift": bool(h["lo_exc"] < 0.05 < h["hi_exc"]),
        "too_wide": bool(h["z2"] < 0.9 and h["lo_exc"] < 0.05 and h["hi_exc"] < 0.05),
    }
    return res


def write_md(res: dict) -> str:
    h, c, rr, v = res["holdout"], res["cv"], res["rolling24"], res["verdict"]
    lines = [
        f"# RSMT-Q 꼬리 점검 — {res['market']} (docs/MODELING.md 36, 사후 진단)",
        "",
        "| 구간 | 행 | 하방 이탈 (r < q5%) | 상방 이탈 (r > q95%) | 80% 구간 | 표준화 제곱오차 z² "
        "| 시장 20일 평균 log 수익 |",
        "|---|---|---|---|---|---|---|",
        f"| CV(레이아웃 5) | {c['n']} | {c['lo_exc']:.3f} | {c['hi_exc']:.3f} | {c['cover80']:.3f} | {c['z2']:.2f} | "
        f"{c['mkt_mean_m']:+.4f} |",
        f"| 최근 24개월 | {h['n']} | {h['lo_exc']:.3f} | {h['hi_exc']:.3f} | {h['cover80']:.3f} | {h['z2']:.2f} | "
        f"{h['mkt_mean_m']:+.4f} |",
        "",
        "## 연도별",
        "",
        "| 연도 | 행 | 하방 | 상방 | 80% | z² | 시장 |",
        "|---|---|---|---|---|---|---|",
    ]
    for y, s in res["by_year"].items():
        lines.append(
            f"| {y} | {s['n']} | {s['lo_exc']:.3f} | {s['hi_exc']:.3f} | {s['cover80']:.3f} | {s['z2']:.2f} | "
            f"{s['mkt_mean_m']:+.4f} |"
        )
    lines += [
        "",
        f"- 과거 24개월 이동 구간 {rr['n_windows']}개의 하방 이탈 5~95% 범위: {rr['lo_p05']:.3f} ~ {rr['lo_p95']:.3f} "
        f"(상방 {rr['hi_p05']:.3f} ~ {rr['hi_p95']:.3f}); 하방 이탈과 시장 평균 수익의 상관 {rr['corr_lo_mkt']:+.2f}",
        f"- 최근 24개월 하방 이탈이 과거 범위 안: {'예' if v['lo_within_range'] else '아니오'} · "
        f"위치 어긋남(하방 < 5% < 상방): {'예' if v['location_shift'] else '아니오'} · "
        f"폭이 넓음(z² < 0.9, 양쪽 < 5%): {'예' if v['too_wide'] else '아니오'}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], default="KR")
    a = ap.parse_args()
    res = run(a.market)
    (OUT / f"tail_check_{a.market}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    md = write_md(res)
    (OUT / f"tail_check_{a.market}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
