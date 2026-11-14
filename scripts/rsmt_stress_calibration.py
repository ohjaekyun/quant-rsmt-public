"""RSMT-Q crisis widening test (docs/MODELING.md 49, pre-declared 2026-10-12): post-hoc rule q' = q50 + s (q - q50),
s = max(1, v)^b with v = market 20-day / 252-day realised volatility at t (Yahoo ^KS11 / SPY daily closes <= t). b is
fitted on the training years only (KR <= 2007, US <= 2015) by the mean pinball loss of q05 / q95 and tested on the
later rows of scripts/rsmt_tail_check.py. Research only.

    .venv/bin/python scripts/rsmt_tail_check.py --market US         # once, the US rows
    .venv/bin/python scripts/rsmt_stress_calibration.py --market KR

Writes reports/rsmt/stress_calibration_<M>.{md,json}.
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

from quantcio.modeling import metrics as mt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "rsmt"
INDEX = {"KR": "^KS11", "US": "SPY"}
TRAIN_END = {"KR": "2007-12-31", "US": "2015-12-31"}
GRID = np.round(np.arange(0.0, 3.0001, 0.05), 2)
CRISIS = ("2008", "2020")


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def vol_ratio(market: str) -> pd.Series:
    import yfinance as yf

    h = yf.Ticker(INDEX[market]).history(period="max", auto_adjust=True)
    c = h["Close"].astype(float)
    c.index = pd.DatetimeIndex(c.index).tz_localize(None).normalize()
    lr = np.log(c).diff()
    return (lr.rolling(20, min_periods=15).std() / lr.rolling(252, min_periods=200).std()).dropna()


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    d = y - q
    return np.maximum(tau * d, (tau - 1.0) * d)


def widen(d: pd.DataFrame, b: float) -> tuple[np.ndarray, np.ndarray]:
    s = np.maximum(1.0, d["v"].to_numpy()) ** b
    q50 = d["q50"].to_numpy()
    return q50 - s * (q50 - d["q05"].to_numpy()), q50 + s * (d["q95"].to_numpy() - q50)


def loss(d: pd.DataFrame, b: float) -> np.ndarray:
    lo, hi = widen(d, b)
    y = d["r"].to_numpy()
    return 0.5 * (pinball(y, lo, 0.05) + pinball(y, hi, 0.95))


def exceed(d: pd.DataFrame, b: float) -> dict:
    lo, hi = widen(d, b)
    y = d["r"].to_numpy()
    return {"lo": float((y < lo).mean()), "hi": float((y > hi).mean()), "n": int(len(d))}


def run(market: str) -> dict:
    p = OUT / f"tail_check_{market}.parquet"
    rows = pd.read_parquet(p)
    rows["date"] = pd.to_datetime(rows["date"])
    v = vol_ratio(market)
    rows["v"] = v.reindex(rows["date"].dt.normalize(), method="ffill").to_numpy()
    rows = rows.dropna(subset=["v", "q05", "q50", "q95", "r"])
    train = rows[rows["date"] <= TRAIN_END[market]]
    test = rows[rows["date"] > TRAIN_END[market]]
    fit = {float(b): float(loss(train, b).mean()) for b in GRID}
    b_hat = min(fit, key=fit.get)
    say(f"{market}: train rows {len(train)}, b = {b_hat}")
    base_l, adj_l = loss(test, 0.0), loss(test, b_hat)
    diff = pd.Series(base_l - adj_l).groupby(test["date"].to_numpy()).mean()
    gain = mt.summarize(diff.sort_index().to_numpy(), 0.0)
    years = {}
    for y, g in test.groupby(test["date"].dt.year):
        years[str(y)] = {"base": exceed(g, 0.0), "adj": exceed(g, b_hat), "v_mean": float(g["v"].mean())}
    hold = test[test["part"] == "holdout"]
    normal = test[~test["date"].dt.year.astype(str).isin(CRISIS)]
    res = {
        "market": market,
        "train_end": TRAIN_END[market],
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "b": b_hat,
        "fit_curve": fit,
        "pinball_gain": gain,
        "pinball_gain_rel": float((base_l.mean() - adj_l.mean()) / base_l.mean()),
        "by_year": years,
        "holdout": {"base": exceed(hold, 0.0), "adj": exceed(hold, b_hat)} if len(hold) else None,
        "normal_years": {"base": exceed(normal, 0.0), "adj": exceed(normal, b_hat)},
    }
    crisis_ok = all(
        abs(years[y]["adj"]["lo"] - 0.05) < abs(years[y]["base"]["lo"] - 0.05) for y in CRISIS if y in years
    )
    res["verdict"] = {
        "pinball_lo_above_0": bool(gain["lo"] is not None and gain["lo"] > 0),
        "crisis_closer_to_5pct": bool(crisis_ok),
        "normal_lo_not_below_2pct": bool(res["normal_years"]["adj"]["lo"] >= 0.02),
    }
    res["verdict"]["adoption_worthy"] = all(res["verdict"].values())
    return res


def write_md(res: dict) -> str:
    m = res["market"]
    g = res["pinball_gain"]
    vd = res["verdict"]
    yes = {True: "예", False: "아니오"}
    lines = [
        f"# RSMT-Q 위기 구간 보정 시험 — {m} (docs/MODELING.md 49)",
        "",
        "- 규칙 s = max(1, 20일/252일 시장 변동성)^b",
        f"- 학습(~{res['train_end']}) {res['train_rows']}행에서 b = **{res['b']}**",
        f"- 시험 {res['test_rows']}행: 핀볼 손실 개선 {res['pinball_gain_rel']:+.2%}, "
        f"날짜별 개선 평균 {g['mean']:+.6f} [{g['lo']:+.6f}, {g['hi']:+.6f}]",
        "",
        "| 연도 | 하방 초과(현재) | 하방 초과(보정) | 상방 초과(현재) | 상방 초과(보정) | 평균 변동성 비율 |",
        "|---|---|---|---|---|---|",
    ]
    for y, v in res["by_year"].items():
        b, a = v["base"], v["adj"]
        lines.append(f"| {y} | {b['lo']:.3f} | {a['lo']:.3f} | {b['hi']:.3f} | {a['hi']:.3f} | {v['v_mean']:.2f} |")
    if res.get("holdout"):
        hb, ha = res["holdout"]["base"], res["holdout"]["adj"]
        lines.append(f"| 최근 24개월 | {hb['lo']:.3f} | {ha['lo']:.3f} | {hb['hi']:.3f} | {ha['hi']:.3f} | |")
    lines += [
        "",
        f"- ① 핀볼 개선 하한 > 0: {yes[vd['pinball_lo_above_0']]}",
        f"- ② 위기 연도(2008·2020) 하방 초과가 5%에 가까워짐: {yes[vd['crisis_closer_to_5pct']]}",
        f"- ③ 평시 하방 초과 ≥ 2%: {yes[vd['normal_lo_not_below_2pct']]}",
        "",
        f"**판정: {'채택 가치 있음(서빙 반영은 사용자 결정)' if vd['adoption_worthy'] else '채택하지 않음'}**",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    a = ap.parse_args()
    res = run(a.market)
    (OUT / f"stress_calibration_{a.market}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    md = write_md(res)
    (OUT / f"stress_calibration_{a.market}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
