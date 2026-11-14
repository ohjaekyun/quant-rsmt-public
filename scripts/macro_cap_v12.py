"""US v12 run B (docs/MODELING.md 30.3, pre-declared 2026-10-05): market-level exposure caps triggered by commercial
real-estate weakness or a Virginia power-price spike, tested exactly like event_cap.py (RSMT.md 7.1). Research only.

    .venv/bin/python scripts/macro_cap_v12.py

Writes reports/rsmt/macro_cap_v12.{md,json}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import event_cap as ec  # noqa: E402
import us_v12_data as p  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "rsmt"
Q_LOW, Q_HIGH = 0.10, 0.90


def main() -> None:
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    close = ec.benchmark("US", settings, cache)
    dates = pd.DatetimeIndex(close.index)
    reg = p.all_regime(settings)
    cre = reg["r_cre_yoy"].reindex(dates, method="ffill")
    price = reg["r_va_price_yoy"].reindex(dates, method="ffill")
    low = (cre < cre.rolling(ec.WINDOW, min_periods=252).quantile(Q_LOW)).fillna(False)
    high = (price > price.rolling(ec.WINDOW, min_periods=252).quantile(Q_HIGH)).fillna(False)
    r = np.log(close).diff()
    d0 = np.minimum(1.0, ec.TARGET_VOL / ec.ewma_vol(r)).fillna(0.0)
    expo = pd.DataFrame(
        {
            "D0": d0,
            "P1": d0 * np.where(low, ec.CAP, 1.0),
            "P2": d0 * np.where(high, ec.CAP, 1.0),
            "P3": d0 * np.where(low | high, ec.CAP, 1.0),
            "BH": 1.0,
        }
    )
    start = max(cre.first_valid_index(), price.first_valid_index()) + pd.Timedelta(400, "D")  # trigger history needed
    rets = ec.strategy_returns(close, expo)
    rets = rets[rets.index >= start]
    cut = rets.index.max() - pd.DateOffset(months=ec.HOLDOUT_MONTHS)
    pre, hold = rets[rets.index <= cut], rets[rets.index > cut]
    res = {
        "start": str(start.date()),
        "holdout_from": str(cut.date()),
        "on_share": {"P1": float(low[low.index >= start].mean()), "P2": float(high[high.index >= start].mean())},
    }
    for name in ("P1", "P2", "P3"):
        pt, lo, hi = ec.sharpe_diff_ci(pre[name], pre["D0"])
        res[name] = {
            "pre": ec.perf(pre[name]),
            "hold": ec.perf(hold[name]),
            "sharpe_diff_pre": [pt, lo, hi],
            "adopt": bool(lo > 0),
        }
    res["D0"] = {"pre": ec.perf(pre["D0"]), "hold": ec.perf(hold["D0"])}
    res["BH"] = {"pre": ec.perf(pre["BH"]), "hold": ec.perf(hold["BH"])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "macro_cap_v12.json").write_text(json.dumps(res, indent=1, default=str))
    lines = [
        "# 미국 v12 B — 부동산·전력요금 노출 상한 (docs/MODELING.md 30.3)",
        "",
        f"- 기간 {res['start']} ~ (홀드아웃 {res['holdout_from']} 이후) · 상한 작동 비율 "
        f"CRE {res['on_share']['P1']:.0%}, 전기요금 {res['on_share']['P2']:.0%}",
        "",
        "| 규칙 | 홀드아웃 전 연수익 | 샤프 | MDD | D0 대비 샤프 차이 [90% 구간] | 채택 |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("D0", "P1", "P2", "P3", "BH"):
        pr = res[name]["pre"]
        if name in ("P1", "P2", "P3"):
            pt, lo, hi = res[name]["sharpe_diff_pre"]
            diff = f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            ad = "예" if res[name]["adopt"] else "아니오"
        else:
            diff, ad = "–", "–"
        lines.append(
            f"| {name} | {pr['ann_return'] * 100:+.2f}% | {pr['sharpe']:.3f} | {pr['mdd'] * 100:.1f}% | {diff} | {ad} |"
        )
    (OUT / "macro_cap_v12.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
