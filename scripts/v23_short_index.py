"""KR short interest (SI) and index membership (IX) features from the KRX collection (docs/MODELING.md 61,
pre-declared 2026-10-14). Research only.

Short balances: weekly all-stock snapshots (scripts/v19_krx.py short_bal), usable from the snapshot date + 5 days
(KRX publishes the balance of day T on T+2 business days). Ban periods: the LEVEL is always used; changes and days to
cover only when the stock could be newly shorted at both ends of the window. Index lists: monthly KOSPI200 / KOSDAQ150
members (scripts/v19_krx.py index), usable from the next day.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v13_data as v13  # noqa: E402
import v19_data as v19  # noqa: E402

SI_LAG = pd.Timedelta(5, "D")
FULL_BANS = [
    (pd.Timestamp("2020-03-16"), pd.Timestamp("2021-05-02")),
    (pd.Timestamp("2023-11-06"), pd.Timestamp("2025-03-30")),
]
PARTIAL = (pd.Timestamp("2021-05-03"), pd.Timestamp("2023-11-05"))
REGIME = {"r_kr_short_ban"}
DOCS = {
    "si_ratio": ("공매도 잔고 ÷ 상장주식수(%)", "short balance / listed shares (%)"),
    "si_chg_4w": ("공매도 잔고 비율 4주 변화(%p, 새 공매도 가능할 때만)", "4-week change of the short ratio"),
    "si_chg_12w": ("공매도 잔고 비율 12주 변화(%p, 새 공매도 가능할 때만)", "12-week change of the short ratio"),
    "si_dtc": ("상환일수 = 잔고 비율 ÷ 일평균 회전율", "days to cover"),
    "si_shortable": ("그날 새 공매도 가능(1/0)", "newly shortable that day"),
    "r_kr_short_ban": ("공매도 금지 국면(허용 0, 부분 0.5, 전면 1)", "short-sale ban regime"),
    "ix_k200": ("KOSPI200 구성 종목", "KOSPI200 member"),
    "ix_k150": ("KOSDAQ150 구성 종목", "KOSDAQ150 member"),
    "ix_added_6m": ("6개월 안 지수 신규 편입", "added to KOSPI200 / KOSDAQ150 within 6 months"),
    "ix_dropped_6m": ("6개월 안 지수 편출", "dropped within 6 months"),
    "ix_weight": ("지수 안 비중(log)", "log weight inside its index"),
}
COLS = [c for c in DOCS]


def ban_level(day: pd.Timestamp) -> float:
    if any(a <= day <= b for a, b in FULL_BANS):
        return 1.0
    return 0.5 if PARTIAL[0] <= day <= PARTIAL[1] else 0.0


def short_table(settings) -> pd.DataFrame:
    files = sorted(glob.glob(str(v19.root(settings) / "krx_short_bal" / "*.parquet")))
    t = pd.concat([pd.read_parquet(f, columns=["date", "ISU_CD", "BAL_RTO"]) for f in files], ignore_index=True)
    t = t.rename(columns={"ISU_CD": "code", "BAL_RTO": "ratio"})
    t["code"] = t["code"].astype(str)
    return t.dropna(subset=["ratio"]).drop_duplicates(["date", "code"])


def index_lists(settings) -> tuple[list[pd.Timestamp], dict[pd.Timestamp, pd.DataFrame]]:
    d = Path(v19.root(settings)) / "krx_index"
    snaps = {pd.Timestamp(p.stem): pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))}
    return sorted(snaps), snaps


def build_kr(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame(np.nan, index=frame.index, columns=COLS)
    ents = frame["entity"].astype(str).to_numpy()
    dates = pd.DatetimeIndex(frame["date"])
    d64 = dates.to_numpy(dtype="datetime64[ns]")
    # ---- index membership per frame row (latest list known the day before)
    snap_dates, snaps = index_lists(settings)
    sd = pd.DatetimeIndex(snap_dates)
    members = {s: {k: set(g["code"].astype(str)) for k, g in snaps[s].groupby("index")} for s in snap_dates}
    weights = {}
    for s in snap_dates:
        g = snaps[s].copy()
        g["w"] = g["mcap"] / g.groupby("index")["mcap"].transform("sum")
        weights[s] = dict(zip(g["code"].astype(str), np.log(g["w"].clip(lower=1e-6)), strict=True))
    pos = sd.searchsorted(dates - pd.Timedelta(1, "D"), "right") - 1
    k200, k150, added, dropped, wgt = (np.full(len(frame), np.nan) for _ in range(5))
    for i, (e, p) in enumerate(zip(ents, pos, strict=True)):
        if p < 0:
            continue
        s = snap_dates[p]
        m = members[s]
        in200, in150 = e in m.get("kospi200", set()), e in m.get("kosdaq150", set())
        k200[i], k150[i] = float(in200), float(in150)
        wgt[i] = weights[s].get(e, np.nan) if (in200 or in150) else np.nan
        q = sd.searchsorted(s - pd.DateOffset(months=6), "right") - 1
        if q >= 0:
            prev = members[snap_dates[q]]
            was = e in prev.get("kospi200", set()) or e in prev.get("kosdaq150", set())
            added[i], dropped[i] = float((in200 or in150) and not was), float(was and not (in200 or in150))
    out["ix_k200"], out["ix_k150"], out["ix_added_6m"], out["ix_dropped_6m"], out["ix_weight"] = (
        k200,
        k150,
        added,
        dropped,
        wgt,
    )

    def shortable(code: str, day: pd.Timestamp) -> bool:
        lvl = ban_level(day)
        if lvl == 1.0:
            return False
        if lvl == 0.0:
            return True
        p = sd.searchsorted(day, "right") - 1
        m = members[snap_dates[p]] if p >= 0 else {}
        return code in m.get("kospi200", set()) or code in m.get("kosdaq150", set())

    out["r_kr_short_ban"] = [ban_level(x) for x in dates]
    out["si_shortable"] = [float(shortable(e, x)) for e, x in zip(ents, dates, strict=True)]
    # ---- short balances
    t = short_table(settings)
    by = {c: g.sort_values("date") for c, g in t.groupby("code")}
    n_matched = 0
    for e in np.unique(ents):
        g = by.get(e)
        if g is None or g.empty:
            continue
        n_matched += 1
        g = g.set_index("date")["ratio"].astype(float)
        sh = pd.Series([shortable(e, x) for x in g.index], index=g.index)
        tab = pd.DataFrame({"si_ratio": g})
        for w, col in ((4, "si_chg_4w"), (12, "si_chg_12w")):
            ok = sh & sh.shift(w, fill_value=False)
            tab[col] = (g - g.shift(w)).where(ok)
        tab.index = tab.index + SI_LAG
        ii = np.where(ents == e)[0]
        v = v13._asof_rows(tab[["si_ratio", "si_chg_4w", "si_chg_12w"]], d64[ii], 21)
        out.iloc[ii, out.columns.get_loc("si_ratio")] = v[:, 0]
        out.iloc[ii, out.columns.get_loc("si_chg_4w")] = v[:, 1]
        out.iloc[ii, out.columns.get_loc("si_chg_12w")] = v[:, 2]
    if "turnover" in frame:  # dataset turnover = log(ADV20 / market cap): daily traded fraction = exp(turnover)
        to = np.exp(frame["turnover"].astype(float))
        dtc = (out["si_ratio"] / 100.0) / to.where(to > 0)
        out["si_dtc"] = dtc.where(out["si_shortable"] > 0)
    info = {
        "short_weeks": int(t["date"].nunique()),
        "short_codes_matched": n_matched,
        "index_months": len(snap_dates),
        "ban_share_rows": {
            str(k): round(float(v), 3) for k, v in out["r_kr_short_ban"].value_counts(normalize=True).items()
        },
    }
    return out, info
