"""Score the recorded US v9 forward picks (reports/strategy/forward/us_v9_picks_<as_of>.csv, written by
``us_v9_serve.py``) once their 20-session labels exist (docs/MODELING.md 26.2). Uses the modeling dataset's own
labels (beta-adjusted 20-session excess return y_e), so the scoring matches the model's target exactly.

    .venv/bin/python scripts/us_v9_forward.py

Writes reports/strategy/forward/us_v9_forward_report.md.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quantcio.config import PROJECT_ROOT, load_settings
from quantcio.data.cache import DiskCache
from quantcio.modeling.dataset import build_dataset
from quantcio.models import Market

FWD = PROJECT_ROOT / "reports" / "strategy" / "forward"
MIN_DATES = 12  # the verdict needs at least 12 non-overlapping dates (as the forward holdout)


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    lab = (y > np.median(y)).astype(int)
    if lab.min() == lab.max():
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    n1 = lab.sum()
    return float((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(lab) - n1)))


def main() -> None:
    files = sorted(FWD.glob("us_v9_picks_*.csv"))
    if not files:
        sys.exit("기록된 예측이 없습니다 — scripts/us_v9_serve.py를 먼저 실행하세요")
    settings = load_settings()
    ds = build_dataset(Market.US, settings, DiskCache(settings.cache_path))
    lab = ds.frame[ds.frame["labelled"]][["date", "entity", "y_e"]]
    rows, pending = [], []
    for p in files:
        pk = pd.read_csv(p, parse_dates=["date"])
        m = pk.merge(lab, on=["date", "entity"], how="inner").dropna(subset=["y_e"])
        d = str(pk["date"].iloc[0].date())
        if len(m) < 50:
            pending.append(d)
            continue
        top = m[m["quintile"] == 5]["y_e"].mean() - m["y_e"].mean()
        rows.append(
            {
                "date": d,
                "names": len(m),
                "auc_median": _auc(m["score"].to_numpy(float), m["y_e"].to_numpy(float)),
                "ic": float(m["score"].rank().corr(m["y_e"].rank())),
                "top_minus_all": float(top),
            }
        )
    lines = ["# 미국 v9 모델 전진 기록 채점", ""]
    lines.append(f"- 기록 {len(files)}건 · 채점 {len(rows)}건 · 대기(20거래일 미경과) {len(pending)}건 {pending}")
    if rows:
        t = pd.DataFrame(rows)
        lines += [
            "",
            "| 기록일 | 종목 | AUC(중앙값 초과) | 순위 IC | 상위 20% − 전체(20거래일 초과수익) |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {r.date} | {r.names} | {r.auc_median:.4f} | {r.ic:+.4f} | {r.top_minus_all * 100:+.2f}% |"
            for r in t.itertuples()
        ]

        def tstat(x: pd.Series) -> float:
            return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 and x.std() > 0 else float("nan")

        lines += [
            "",
            f"- 평균 AUC {t['auc_median'].mean():.4f} (t vs 0.5: {tstat(t['auc_median'] - 0.5):+.2f}) · "
            f"평균 IC {t['ic'].mean():+.4f} (t {tstat(t['ic']):+.2f}) · "
            f"상위 20% 초과 {t['top_minus_all'].mean() * 100:+.2f}%",
            "- 판정: "
            + (
                "가능"
                if len(t) >= MIN_DATES
                else f"아직 불가 — 겹치지 않는 날짜 {MIN_DATES}개 이상 필요(현재 {len(t)}개)"
            ),
        ]
    (FWD / "us_v9_forward_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
