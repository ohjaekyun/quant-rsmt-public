"""QA cross-checks for the marcap backbone (keyless; never used to BUILD the universe).

Ported from ``docs/sources/pit/reference/kr_crosscheck.py``. The probe/verifier QA assertions kept here:
- the marcap calendar equals Naver's KOSPI calendar on the overlap (3,000 of 3,000 sessions);
- the base-price chain matches Naver's adjusted close within 1% for large caps (Naver closes are NXT-integrated
  from ~2026-09-14, so only earlier days are compared; Naver stitches relisting spells - compare within one spell);
- marcap Stocks equals Daum listedSharesCount (Daum needs a Referer; codes delisted before ~2024-09 return 400).
"""

from __future__ import annotations

import re
import time
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd

NAVER_FCHART = "https://fchart.stock.naver.com/sise.nhn"
DAUM_DAYS = "https://finance.daum.net/api/quote/A{code}/days"
NXT_CUTOFF = pd.Timestamp("2026-09-14")  # Naver / Daum closes are NXT-integrated from here on


def naver_fchart_spells(session: Any, code: str, gap_days: int = 20, count: int = 3000) -> pd.DataFrame:
    """Naver fchart daily bars (<= 3,000; adjusted, but not always correctly for small caps) with a 'spell' column:
    Naver STITCHES relisting spells, so returns must never be computed across them."""
    r = session.get(
        NAVER_FCHART, params={"timeframe": "day", "count": count, "requestType": 0, "symbol": code}, timeout=20
    )
    r.raise_for_status()
    rows = re.findall(r'<item data="(.*?)"\s*/>', r.text)
    if not rows:
        raise ValueError(f"fchart: {code} 데이터 없음")
    df = pd.read_csv(
        StringIO("\n".join(rows)),
        sep="|",
        header=None,
        dtype={"d": str},
        names=["d", "open", "high", "low", "close", "volume"],
    )
    df.index = pd.to_datetime(df.pop("d"), format="%Y%m%d")
    df["spell"] = (df.index.to_series().diff().dt.days > gap_days).cumsum().to_numpy()
    df["ret"] = df.groupby("spell")["close"].pct_change()
    return df


def daum_shares_history(session: Any, code: str, per_page: int = 1000, pause: float = 0.25) -> pd.DataFrame:
    """Daily listedSharesCount (+ close/volume/value) for listed codes (and codes delisted since ~2024-09)."""
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        r = session.get(
            DAUM_DAYS.format(code=code),
            params={"symbolCode": f"A{code}", "page": page, "perPage": per_page, "pagination": "true"},
            headers={"Referer": f"https://finance.daum.net/quotes/A{code}"},
            timeout=30,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Daum quote/days {code}: HTTP {r.status_code}")
        j = r.json()
        rows += j["data"]
        if page >= int(j.get("totalPages") or 1):
            break
        page += 1
        time.sleep(pause)
    d = pd.DataFrame(rows)
    d.index = pd.to_datetime(d["date"].str[:10])
    return d.sort_index()[["tradePrice", "accTradeVolume", "accTradePrice", "listedSharesCount"]]


def calendar_diff(sessions: pd.DatetimeIndex, naver_index: pd.DatetimeIndex) -> dict[str, int]:
    """Session calendars compared on their overlap: counts of dates only in one of them."""
    lo, hi = max(sessions[0], naver_index[0]), min(sessions[-1], naver_index[-1])
    a = set(sessions[(sessions >= lo) & (sessions <= hi)])
    b = set(naver_index[(naver_index >= lo) & (naver_index <= hi)])
    return {"overlap": len(a & b), "only_marcap": len(a - b), "only_naver": len(b - a)}


def adjusted_close_agreement(frame: pd.DataFrame, naver: pd.DataFrame, tol: float = 0.01) -> dict[str, float]:
    """Share of common days (before the NXT switch, last Naver spell only) where the chain-adjusted close is within
    `tol` of Naver's adjusted close after aligning both on their last common day."""
    nv = naver[naver["spell"] == naver["spell"].max()]
    common = frame.index.intersection(nv.index)
    common = common[common < NXT_CUTOFF]
    if len(common) < 20:
        return {"days": float(len(common)), "within_tol": float("nan")}
    a = frame.loc[common, "close"].to_numpy(dtype=float)
    b = nv.loc[common, "close"].to_numpy(dtype=float)
    a = a / a[-1]
    b = b / b[-1]
    ok = np.abs(a / b - 1.0) <= tol
    return {"days": float(len(common)), "within_tol": float(ok.mean())}
