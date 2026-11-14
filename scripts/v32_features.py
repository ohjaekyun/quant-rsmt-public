"""Sector export features from Korean customs data (docs/MODELING.md 76, declared 2026-10-24 with the HS -> sector map
in scripts/v32_trade_data.py). Research only. Built with the 71 rules (scripts/mixfreq.py); KR only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mixfreq as mf  # noqa: E402
import v30_features as v30  # noqa: E402
import v31_features as v31  # noqa: E402
import v32_trade_data as td  # noqa: E402

DOCS: dict[str, tuple[str, str]] = {
    "sx_exp_yoy": ("업종 주력 품목 수출액(3개월 합) 전년 대비 z", "sector product exports (3-month sum) yoy z"),
    "sx_exp_mom": ("업종 수출액 3개월 합 직전 3개월 대비 z", "sector exports 3-month sum vs previous 3 months z"),
    "sx_exp_price": ("업종 수출 단가(달러/kg) 전년 대비 z(금액 가중)", "sector export unit value yoy z, value-wtd"),
    "sx_trade_bal": ("업종 무역수지÷수출 3개월 변화 z", "sector trade balance / exports 3-month change z"),
}


def _frame(date: pd.Index, value: pd.Series) -> pd.DataFrame:
    av = td.available(pd.Series(date)).to_numpy()
    return pd.DataFrame({"date": date, "value": value.to_numpy(), "available": av})


def monthly(trade: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per HS: month x (exp_usd, exp_kg, imp_usd, imp_kg) on a full month grid; a month with no rows up to the last
    fetched month means no trade (zero, 71: counts that mean none)."""
    t = trade.copy()
    t["date"] = pd.to_datetime(t["date"])
    grid = pd.date_range(t["date"].min(), t["date"].max(), freq="ME")
    cols = list(td.FIELDS.values())
    return {h: g.set_index("date")[cols].reindex(grid).fillna(0.0) for h, g in t.groupby("hs")}


def sector_series(by_hs: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.DataFrame]]:
    out: dict[str, dict[str, pd.DataFrame]] = {k: {} for k in DOCS}
    for sec, codes in td.HS_SECTOR.items():
        parts = [by_hs[h] for h in codes if h in by_hs]
        if not parts:
            continue
        s3 = sum(parts).rolling(3).sum()
        exp3 = s3["exp_usd"].where(s3["exp_usd"] > 0)
        out["sx_exp_yoy"][sec] = mf.zchange(_frame(s3.index, exp3), "M", 12, yoy=True)
        out["sx_exp_mom"][sec] = mf.zchange(_frame(s3.index, exp3), "M", 3)
        bal = (s3["exp_usd"] - s3["imp_usd"]) / exp3
        out["sx_trade_bal"][sec] = mf.zchange(_frame(s3.index, bal).dropna(), "M", 3, log=False)
        # unit value: per HS log yoy change of (3-month USD / 3-month kg), averaged with the HS's 3-month USD as weight
        chg, wts = [], []
        for p in parts:
            p3 = p.rolling(3).sum()
            uv = np.log((p3["exp_usd"] / p3["exp_kg"]).where((p3["exp_kg"] > 0) & (p3["exp_usd"] > 0)))
            chg.append(uv - uv.shift(12))
            wts.append(p3["exp_usd"])
        keys = list(range(len(parts)))  # same column labels, so c * w aligns by HS
        c, w = pd.concat(chg, axis=1, keys=keys), pd.concat(wts, axis=1, keys=keys)
        w = w.where(c.notna())
        price = (c * w).sum(axis=1, min_count=1) / w.sum(axis=1).where(w.sum(axis=1) > 0)
        out["sx_exp_price"][sec] = v31._level_z3(_frame(price.index, price).dropna(), "M")
    return out


def build_kr(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, list[str], dict]:
    d = td.root(settings)
    trade = pd.read_parquet(d / "kr_trade.parquet")
    by_hs = monthly(trade)
    out = pd.DataFrame(index=frame.index)
    for name, by_sec in sector_series(by_hs).items():
        out[name] = v30._sector_rows(frame, by_sec, "M")
    reg = pd.DataFrame(index=frame.index)
    tot = pd.read_parquet(d / "kr_trade_total.parquet")
    exp = tot[tot["series"] == "exp_usd"].sort_values("date").set_index("date")["value"]
    exp3 = exp.rolling(3).sum()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    reg["r_kr_exports_yoy"] = mf.asof(mf.zchange(_frame(exp3.index, exp3), "M", 12, yoy=True), "z", dates, "M")
    info = {
        "hs_codes": len(by_hs),
        "from": str(min(g.index.min() for g in by_hs.values()).date()),
        "to": str(max(g.index.max() for g in by_hs.values()).date()),
    }
    return pd.concat([out, reg], axis=1), list(reg.columns), info


def block(market: str, frame: pd.DataFrame, settings, cache=None) -> tuple[pd.DataFrame, list[str], dict]:
    if market != "KR":
        raise ValueError("76 is a KR block")
    return build_kr(frame, settings)
