"""SEC XBRL companyfacts -> clean quarterly series (Revenue / EPS / NI ...) with YoY.

Verified 2026-09-24 on AAPL (FY ends Sep), NVDA (FY ends Jan).
FIXES vs probe version (verifier):
  * any missing quarter is derived from consecutive YTD facts (6M-3M, 9M-6M, FY-9M), not only Q4
    -> cash-flow metrics (reported YTD-only in 10-Qs) no longer have holes every Q2/Q3
  * qoq is NaN when the previous row is not the adjacent quarter (no more cross-gap qoq)
  * point-in-time support: `first_filed` column (first date the number was public) and
    vintage="first" (as-reported values; NOT split-adjusted, see quarterly_series docstring)
  * retry on 429/5xx/network errors
"""
from __future__ import annotations

import time

import pandas as pd
import requests

UA = "QuantCIO Research research@example.com"   # SEC requires 'Company contact@domain'
_S = requests.Session()
_S.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})

CONCEPTS = {
    # metric: ordered us-gaap concept candidates (first one wins per period)
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "RevenuesNetOfInterestExpense"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "op_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
}

_PERIODIC_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-QT", "20-F", "20-F/A", "40-F", "6-K"}
# duration buckets (days): 0 = quarter, 1 = 6M YTD, 2 = 9M YTD, 3 = fiscal year
# pd.Timedelta(days=n) warns under pandas 2.3.3 + numpy 2.5 ("generic unit ... will raise") -> use (n, "D")
_D = lambda n: pd.Timedelta(n, "D")  # noqa: E731
_BUCKETS = [(80, 100), (170, 200), (260, 285), (350, 380)]


def get_companyfacts(cik: str | int, retries: int = 3) -> dict:
    cik10 = str(int(cik)).zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    for i in range(retries):
        try:
            r = _S.get(url, timeout=30)
        except (requests.ConnectionError, requests.Timeout):
            if i == retries - 1:
                raise
            time.sleep(2 ** i)
            continue
        if r.status_code in (429, 500, 502, 503, 504) and i < retries - 1:
            time.sleep(2 ** i)
            continue
        r.raise_for_status()
        return r.json()


def _bucket(days: int):
    for i, (lo, hi) in enumerate(_BUCKETS):
        if lo <= days <= hi:
            return i
    return None


def _facts_df(facts: dict, concepts: list[str], vintage: str = "latest") -> pd.DataFrame:
    rows = []
    g = facts.get("facts", {}).get("us-gaap", {})
    for prio, c in enumerate(concepts):
        for unit, arr in g.get(c, {}).get("units", {}).items():
            for f in arr:
                if "start" not in f:  # instant facts (balance sheet) not handled here
                    continue
                rows.append({**f, "concept": c, "unit": unit, "prio": prio})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df["form"].isin(_PERIODIC_FORMS)].copy()  # drop DEF 14A / 8-K / S-4 copies of facts
    if df.empty:
        return df
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df["days"] = (df["end"] - df["start"]).dt.days
    # first date this period's number was public (for point-in-time / no look-ahead features)
    df["first_filed"] = df.groupby(["start", "end"])["filed"].transform("min")
    if vintage == "first":   # as originally reported (ignores later restatements)
        df = df.sort_values(["start", "end", "filed", "prio"])
    else:                    # latest restated value, best-priority concept
        df = df.sort_values(["start", "end", "prio", "filed"], ascending=[True, True, True, False])
    df = df.drop_duplicates(["start", "end"], keep="first")
    df["bucket"] = df["days"].map(_bucket)
    return df[df["bucket"].notna()]


def _derived_row(a, start, val, first_filed) -> dict:
    return {"start": start, "end": a.end, "val": val, "concept": a.concept, "form": a.form, "accn": a.accn,
            "filed": a.filed, "first_filed": first_filed, "unit": a.unit, "prio": a.prio, "derived": True}


def quarterly_series(facts: dict, metric: str = "revenue", derive: bool = True,
                     vintage: str = "latest") -> pd.DataFrame:
    """Return DataFrame[start, end, value, yoy, qoq, derived, concept, unit, form, accn, filed, first_filed].

    - quarterly = 80..100-day durations. Missing quarters are derived from consecutive YTD
      facts with the same fiscal-year start (Q2 = 6M - Q1, Q3 = 9M - 6M, Q4 = FY - 9M, or
      FY - Q1 - Q2 - Q3) and flagged derived=True. Derived EPS ignores share-count drift (estimate).
    - yoy = value / value of the quarter ending 364 +/- 16 days earlier - 1 (NaN on <= 0 base).
    - qoq is only filled when the previous row is the adjacent quarter (80..100 days apart).
    - filed = filing the kept value came from (for 'latest' this is usually a LATER comparative:
      66/73 AAPL rows); first_filed = first public date of that period -> use first_filed, never
      'end' or 'filed', as the availability date when joining to prices in backtests.
    - vintage='latest' (default): restated + split-adjusted values (AAPL 2019-12 EPS 1.25).
      vintage='first': as originally reported (AAPL 2019-12 EPS 4.99, pre 4:1 split) -> EPS yoy is
      WRONG across splits in this mode; use it for revenue/NI only, or keep 'latest' + first_filed.
    """
    df = _facts_df(facts, CONCEPTS[metric], vintage)
    if df.empty:
        return df
    q = df[df.bucket == 0].copy()
    q["derived"] = False
    if derive:
        add: list = []
        for b in (1, 2, 3):
            known = pd.concat([q, pd.DataFrame(add)], ignore_index=True) if add else q
            for _, a in df[df.bucket == b].iterrows():
                if ((known.end - a.end).abs().dt.days <= 3).any():
                    continue  # quarter already reported (or derived)
                prev = df[(df.bucket == b - 1) & ((df.start - a.start).abs().dt.days <= 3)
                          & (df.end < a.end - _D(60))]
                if len(prev):
                    p = prev.sort_values("end").iloc[-1]
                    add.append(_derived_row(a, p.end + _D(1), a.val - p.val,
                                            max(a.first_filed, p.first_filed)))
                    continue
                qs = known[(known.start >= a.start - _D(3))
                           & (known.end < a.end - _D(20))]
                if len(qs) == b:  # sum of the b earlier quarters of this fiscal year
                    add.append(_derived_row(a, qs.end.max() + _D(1), a.val - qs.val.sum(),
                                            max([a.first_filed] + list(qs.first_filed))))
        if add:
            q = pd.concat([q, pd.DataFrame(add)], ignore_index=True)
    q = q.sort_values(["end", "prio", "derived"]).drop_duplicates("end", keep="first").reset_index(drop=True)
    q = q.rename(columns={"val": "value"})
    q["value"] = q["value"].astype(float).round(4)  # YTD subtraction float noise (4.8099999 -> 4.81)
    # YoY: match quarter ending ~1y earlier
    prev = q[["end", "value"]].rename(columns={"end": "end_prev", "value": "value_prev"})
    q["_key"] = q["end"] - _D(364)
    q = pd.merge_asof(q.sort_values("_key"), prev.sort_values("end_prev"), left_on="_key",
                      right_on="end_prev", direction="nearest", tolerance=_D(16))
    q["yoy"] = q["value"] / q["value_prev"] - 1
    q.loc[q["value_prev"].fillna(0) <= 0, "yoy"] = float("nan")  # meaningless on negative base
    q = q.sort_values("end").reset_index(drop=True)
    gap = q["end"].diff().dt.days
    q["qoq"] = (q["value"] / q["value"].shift(1) - 1).where(gap.between(80, 100))
    q.loc[q["value"].shift(1).fillna(0) <= 0, "qoq"] = float("nan")
    return q[["start", "end", "value", "yoy", "qoq", "derived", "concept", "unit", "form", "accn",
              "filed", "first_filed"]]


def fundamentals_snapshot(cik: str | int, n_quarters: int = 8, vintage: str = "latest") -> dict:
    """One call -> dict of last-N quarterly series for revenue / EPS / NI / GP / OpInc / OCF + gross margin."""
    facts = get_companyfacts(cik)
    out = {"entity": facts.get("entityName"), "cik": facts.get("cik")}
    for m in ["revenue", "eps_diluted", "net_income", "gross_profit", "operating_income", "op_cash_flow"]:
        s = quarterly_series(facts, m, vintage=vintage)
        out[m] = s.tail(n_quarters).reset_index(drop=True) if len(s) else s
    rev = out["revenue"]
    if len(rev) and len(out["gross_profit"]):
        gm = pd.merge(rev[["end", "value"]], out["gross_profit"][["end", "value"]], on="end", suffixes=("_rev", "_gp"))
        out["gross_margin"] = gm.assign(gross_margin=gm.value_gp / gm.value_rev)[["end", "gross_margin"]]
    return out


if __name__ == "__main__":
    for t, cik in {"AAPL": "320193", "NVDA": "1045810"}.items():
        snap = fundamentals_snapshot(cik, 4)
        print(t, snap["entity"])
        for m in ["revenue", "eps_diluted", "op_cash_flow"]:
            print(snap[m][["start", "end", "value", "yoy", "qoq", "derived", "first_filed"]].to_string())
