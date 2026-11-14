"""US v4 research features (docs/MODELING.md section 18.2): earnings-announcement reactions, earnings surprises,
quality / investment from SEC XBRL. Research code: lives outside ``src/quantcio`` so the KR holdout freeze stays.

Point in time:
- XBRL facts: each period takes the FIRST filed value (the original, never a later restatement); usable at grid
  dates t with ``filed < t``. Annual flows only from 10-K-family forms with a 330-400 day duration.
- 8-K item 2.02 (results of operations): the reaction session is the acceptance session (NY time < 16:00) or the
  next session (>= 16:00); the two-session reaction [d0, d0+1] is usable from the session after d0+1.
- Earnings surprise from the PIT fundamentals panel (``available_from <= t``, the same rule as the modeling panel).
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from quantcio.data.pit.us_cik import Submissions
from quantcio.data.pit.us_fund import REVENUE
from quantcio.data.pit.us_net import PitNet

COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
CONCEPTS: dict[str, list[str]] = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "liab_equity": ["LiabilitiesAndStockholdersEquity"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "gross_profit": ["GrossProfit"],
    "revenue": list(REVENUE),
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
    ],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "dividends": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
}
INSTANT = {"assets", "liabilities", "liab_equity", "equity"}
ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A"}
PERIODIC_FORMS = ANNUAL_FORMS | {"10-Q", "10-Q/A", "10-QT", "10-QT/A"}
ANNUAL_MAX_AGE = 500  # days from fiscal year end
INSTANT_MAX_AGE = 200
EAR_MAX_SESSIONS = 63
ANN_HORIZON = 20
FEATURES = [
    "us_ear",
    "us_ear_days",
    "us_ann_soon",
    "us_sue",
    "us_sue_rev",
    "us_gpa",
    "us_asset_growth",
    "us_accruals",
    "us_leverage",
    "us_div_yield",
    "us_capex",
]
DOCS = {
    "us_ear": ("실적 발표 반응(2거래일, SPY 대비)", "earnings-announcement reaction (2 sessions vs SPY)"),
    "us_ear_days": ("실적 발표 후 경과일(log)", "sessions since the earnings reaction (log)"),
    "us_ann_soon": ("20거래일 안 실적 발표 예상", "earnings announcement expected within 20 sessions"),
    "us_sue": ("실적 서프라이즈(순이익, 표준화)", "standardised unexpected earnings (net income)"),
    "us_sue_rev": ("매출 서프라이즈(표준화)", "standardised unexpected revenue"),
    "us_gpa": ("매출총이익/총자산", "gross profit / total assets"),
    "us_asset_growth": ("총자산 증가율(전년 대비)", "total asset growth (YoY)"),
    "us_accruals": ("발생액((순이익−영업현금흐름)/평균 자산)", "accruals ((NI - CFO) / average assets)"),
    "us_leverage": ("부채비율(총부채/총자산)", "leverage (liabilities / assets)"),
    "us_div_yield": ("배당수익률(연간 배당 지급/시가총액)", "dividend yield (dividends paid / market cap)"),
    "us_capex": ("설비투자/총자산", "capital expenditure / total assets"),
}


# ================================================================================================ XBRL


def _trim(j: dict) -> dict:
    g = (j.get("facts") or {}).get("us-gaap") or {}
    want = {c for cs in CONCEPTS.values() for c in cs}
    out = {}
    for c in want:
        if c in g:
            usd = (g[c].get("units") or {}).get("USD")
            if usd:
                out[c] = [{k: f.get(k) for k in ("start", "end", "val", "form", "filed")} for f in usd]
    return {"cik": j.get("cik"), "facts": out}


def fetch_facts(net: PitNet, ciks: Iterable[int], progress=None) -> dict[int, dict]:
    """Trimmed companyfacts per CIK (cached gz JSON under the PIT US sec cache; 404 -> empty, cached too)."""
    out: dict[int, dict] = {}
    ciks = sorted({int(c) for c in ciks})
    for i, cik in enumerate(ciks, 1):
        rel = f"facts_quality/{cik}.json.gz"
        j = net.read_json("sec", rel)
        if j is None:
            r = net.http_get(COMPANYFACTS.format(cik=cik), "sec", ok=(200, 404))
            j = _trim(r.json()) if r.status_code == 200 else {"cik": cik, "facts": {}}
            net.write_json("sec", rel, j)
        out[cik] = j
        if progress and i % 100 == 0:
            progress(f"companyfacts {i}/{len(ciks)}")
    return out


def _days(a: str | None, b: str | None) -> float:
    if not a or not b:
        return float("nan")
    return (pd.Timestamp(b) - pd.Timestamp(a)).days


def first_filed(facts: dict, item: str, annual: bool) -> pd.DataFrame:
    """(end, filed, value) per period end: the first filed value over the item's concepts (priority order)."""
    rows = []
    for prio, c in enumerate(CONCEPTS[item]):
        for f in facts.get(c, []) or []:
            form = str(f.get("form") or "")
            if item in INSTANT:
                if form not in (ANNUAL_FORMS if annual else PERIODIC_FORMS):
                    continue
            else:
                if form not in ANNUAL_FORMS:
                    continue
                dur = _days(f.get("start"), f.get("end"))
                if not (330 <= dur <= 400):
                    continue
            if f.get("val") is None or not f.get("end") or not f.get("filed"):
                continue
            rows.append((pd.Timestamp(f["end"]), pd.Timestamp(f["filed"]), prio, float(f["val"])))
    if not rows:
        return pd.DataFrame(columns=["end", "filed", "value"])
    df = pd.DataFrame(rows, columns=["end", "filed", "prio", "value"]).sort_values(["end", "filed", "prio"])
    return df.drop_duplicates("end", keep="first")[["end", "filed", "value"]].reset_index(drop=True)


def quality_tables(facts: dict) -> dict[str, pd.DataFrame]:
    """Per item: first-filed annual (or instant) values of one company."""
    t = {k: first_filed(facts, k, annual=True) for k in CONCEPTS}
    t["assets_q"] = first_filed(facts, "assets", annual=False)
    t["liabilities_q"] = first_filed(facts, "liabilities", annual=False)
    t["liab_equity_q"] = first_filed(facts, "liab_equity", annual=False)
    t["equity_q"] = first_filed(facts, "equity", annual=False)
    return t


def _match(df: pd.DataFrame, end: pd.Timestamp, tol: int = 10) -> pd.Series | None:
    if df.empty:
        return None
    d = (df["end"] - end).abs().dt.days
    i = d.idxmin()
    return df.loc[i] if d.loc[i] <= tol else None


def _annual_records(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One record per fiscal year end (anchored on annual total assets) with every component and its filed date."""
    recs = []
    a = t["assets"]
    for r in a.itertuples(index=False):
        rec = {"end": r.end, "filed": r.filed, "assets": r.value}
        for k in ("gross_profit", "revenue", "cost_of_revenue", "cfo", "net_income", "dividends", "capex"):
            m = _match(t[k], r.end)
            rec[k] = float(m["value"]) if m is not None else np.nan
            rec[f"{k}_filed"] = m["filed"] if m is not None else pd.NaT
        prev = _match(a, r.end - pd.Timedelta(365, "D"), tol=45)
        rec["assets_prev"] = float(prev["value"]) if prev is not None else np.nan
        recs.append(rec)
    return pd.DataFrame(recs).sort_values("filed").reset_index(drop=True) if recs else pd.DataFrame()


def _latest_known(filed: np.ndarray, end: np.ndarray, d: np.datetime64, max_age: int) -> int:
    """Index of the record with the latest period end among those filed strictly before d (-1 if none / stale)."""
    n = int(np.searchsorted(filed, d, "left"))  # records [0, n) were filed before d
    if n == 0:
        return -1
    j = int(np.argmax(end[:n]))
    return j if (d - end[j]) / np.timedelta64(1, "D") <= max_age else -1


def quality_features(rows: pd.DataFrame, tables_by_entity: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """us_gpa, us_asset_growth, us_accruals, us_leverage, us_div_yield, us_capex for (date, entity, mcap) rows.

    Dividend yield: a company that files a cash-flow statement (CFO known) but tags no dividend payments for that
    fiscal year is taken as paying none (0); without a cash-flow statement it is unknown (NaN)."""
    cols = ["us_gpa", "us_asset_growth", "us_accruals", "us_leverage", "us_div_yield", "us_capex"]
    out = np.full((len(rows), len(cols)), np.nan)
    ents = rows["entity"].astype(str).to_numpy()
    dates = rows["date"].to_numpy(dtype="datetime64[ns]")
    mcap = rows["mcap"].to_numpy(dtype=float)
    for e in pd.unique(ents):
        t = tables_by_entity.get(e)
        if t is None:
            continue
        ix = np.flatnonzero(ents == e)
        fy = _annual_records(t)
        if len(fy):
            f_filed = fy["filed"].to_numpy(dtype="datetime64[ns]")
            f_end = fy["end"].to_numpy(dtype="datetime64[ns]")
            comp = {
                k: (fy[k].to_numpy(dtype=float), fy[f"{k}_filed"].to_numpy(dtype="datetime64[ns]"))
                for k in ("gross_profit", "revenue", "cost_of_revenue", "cfo", "net_income", "dividends", "capex")
            }
            assets = fy["assets"].to_numpy(dtype=float)
            aprev = fy["assets_prev"].to_numpy(dtype=float)
        q = t["assets_q"].sort_values("filed")
        q_filed = q["filed"].to_numpy(dtype="datetime64[ns]")
        q_end = q["end"].to_numpy(dtype="datetime64[ns]")
        q_val = q["value"].to_numpy(dtype=float)
        liab = {k: t[k].sort_values("filed") for k in ("liabilities_q", "liab_equity_q", "equity_q")}
        for k in ix:
            d = dates[k]
            if len(fy):
                j = _latest_known(f_filed, f_end, d, ANNUAL_MAX_AGE)
                if j >= 0 and assets[j] > 0:

                    def val(name: str, _j=j, _d=d) -> float:
                        v, f = comp[name][0][_j], comp[name][1][_j]
                        return float(v) if not np.isnat(f) and f < _d and np.isfinite(v) else np.nan

                    gp = val("gross_profit")
                    if not np.isfinite(gp):
                        rev, cogs = val("revenue"), val("cost_of_revenue")
                        gp = rev - cogs if np.isfinite(rev) and np.isfinite(cogs) else np.nan
                    out[k, 0] = gp / assets[j]
                    if np.isfinite(aprev[j]) and aprev[j] > 0:
                        out[k, 1] = assets[j] / aprev[j] - 1.0
                        cfo, ni = val("cfo"), val("net_income")
                        if np.isfinite(cfo) and np.isfinite(ni):
                            out[k, 2] = (ni - cfo) / ((assets[j] + aprev[j]) / 2.0)
                    cap = val("capex")
                    out[k, 5] = cap / assets[j] if np.isfinite(cap) else np.nan
                    div = val("dividends")
                    if not np.isfinite(div) and np.isfinite(val("cfo")):
                        div = 0.0
                    if np.isfinite(div) and np.isfinite(mcap[k]) and mcap[k] > 0:
                        out[k, 4] = div / mcap[k]
            if len(q):
                jq = _latest_known(q_filed, q_end, d, INSTANT_MAX_AGE)
                if jq >= 0 and q_val[jq] > 0:
                    end_q = pd.Timestamp(q_end[jq])

                    def inst(name: str, _d=d, _end=end_q) -> float:
                        tt = liab[name]
                        tt = tt[tt["filed"].to_numpy(dtype="datetime64[ns]") < _d]
                        m = _match(tt, _end, 3)
                        return float(m["value"]) if m is not None else np.nan

                    lv = inst("liabilities_q")
                    if not np.isfinite(lv):
                        le, eq = inst("liab_equity_q"), inst("equity_q")
                        lv = le - eq if np.isfinite(le) and np.isfinite(eq) else np.nan
                    if np.isfinite(lv):
                        out[k, 3] = lv / q_val[jq]
    return pd.DataFrame(out, index=rows.index, columns=cols)


# ================================================================================================ earnings events


def reaction_days(subs: Submissions, cik: int, sessions: pd.DatetimeIndex, start: pd.Timestamp) -> list[pd.Timestamp]:
    """Reaction sessions of the company's 8-K item 2.02 filings (NY acceptance time < 16:00 -> that session, else
    the next one), sorted, one per session."""
    sub = subs.get(int(cik))
    if not sub:
        return []
    blocks = [(sub.get("filings") or {}).get("recent") or {}]
    for f in (sub.get("filings") or {}).get("files", []) or []:
        oldest = min((pd.Timestamp(d) for b in blocks for d in b.get("filingDate", []) or []), default=None)
        if oldest is not None and oldest <= start:
            break
        blocks.append(subs.page(f["name"]))
    out = set()
    sess = sessions.to_numpy(dtype="datetime64[ns]")
    for b in blocks:
        forms = b.get("form", []) or []
        items = b.get("items", []) or [""] * len(forms)
        acc = b.get("acceptanceDateTime", []) or [""] * len(forms)
        fdates = b.get("filingDate", []) or [""] * len(forms)
        for form, it, a, fd in zip(forms, items, acc, fdates, strict=False):
            if not str(form).startswith("8-K") or "2.02" not in str(it):
                continue
            ts = pd.to_datetime(a, utc=True, errors="coerce") if a else pd.NaT
            if pd.notna(ts):
                ny = ts.tz_convert("America/New_York")
                day = pd.Timestamp(ny.date())
                after_close = ny.hour >= 16
            else:
                day = pd.Timestamp(fd) if fd else pd.NaT
                after_close = True  # unknown time: assume after the close (conservative)
            if pd.isna(day) or day < start:
                continue
            p = np.searchsorted(sess, np.datetime64(day, "ns"), "left")  # first session >= day
            if p < len(sess) and after_close and sess[p] == np.datetime64(day, "ns"):
                p += 1
            if p < len(sess):
                out.add(pd.Timestamp(sess[p]))
    return sorted(out)


def event_features(
    rows: pd.DataFrame,
    reactions: dict[str, list[pd.Timestamp]],
    prices: dict[str, pd.DataFrame],
    bench: pd.Series,
    sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    """us_ear, us_ear_days, us_ann_soon for (date, entity) rows."""
    out = np.full((len(rows), 3), np.nan)
    sess = sessions.to_numpy(dtype="datetime64[ns]")
    pos_of = {d: i for i, d in enumerate(sess)}
    b = bench.reindex(sessions).ffill()
    ents = rows["entity"].astype(str).to_numpy()
    dates = rows["date"].to_numpy(dtype="datetime64[ns]")
    for e in pd.unique(ents):
        rd = reactions.get(e) or []
        ix = np.flatnonzero(ents == e)
        if not rd:
            continue
        px = prices.get(e)
        c = px["close"].astype(float).reindex(sessions).ffill(limit=3) if px is not None and len(px) else None
        r_pos = np.array([pos_of.get(np.datetime64(d, "ns"), -1) for d in rd])
        r_pos = r_pos[r_pos > 0]
        ear = np.full(len(r_pos), np.nan)
        if c is not None:
            for j, p in enumerate(r_pos):
                if p + 1 < len(sess):
                    s0, s1 = c.iloc[p - 1], c.iloc[p + 1]
                    m0, m1 = b.iloc[p - 1], b.iloc[p + 1]
                    if s0 > 0 and m0 > 0 and np.isfinite(s1) and np.isfinite(m1):
                        ear[j] = (s1 / s0) - (m1 / m0)
        gaps = np.diff(r_pos)
        for k in ix:
            tp = np.searchsorted(sess, dates[k], "right") - 1  # session position of the grid date
            usable = np.flatnonzero(r_pos + 1 < tp)  # the reaction's second session strictly before t
            known = np.flatnonzero(r_pos < tp)
            if len(usable):
                j = usable[-1]
                since = tp - r_pos[j]
                if since <= EAR_MAX_SESSIONS:
                    out[k, 0] = ear[j]
            if len(known):
                out[k, 1] = math.log1p(tp - r_pos[known[-1]])
                g = gaps[: max(len(known) - 1, 0)]
                if len(g) >= 4:
                    nxt = r_pos[known[-1]] + float(np.median(g))
                    out[k, 2] = 1.0 if tp < nxt <= tp + ANN_HORIZON else 0.0
    return pd.DataFrame(out, index=rows.index, columns=["us_ear", "us_ear_days", "us_ann_soon"])


# ================================================================================================ surprises


def surprise_table(fpanel: pd.DataFrame) -> pd.DataFrame:
    """Per entity filing row: standardised change of TTM net income / revenue vs the previous quarter (first
    available version per period end; sd over the previous 8 changes, >= 4)."""
    p = fpanel.dropna(subset=["available_from", "period_end"]).sort_values(["entity", "available_from"])
    p = p.drop_duplicates(["entity", "period_end"], keep="first").sort_values(["entity", "period_end"])
    parts = []
    for _e, g in p.groupby("entity", sort=False):
        g = g.copy()
        for src, dst in (("ni_parent_ttm", "us_sue"), ("revenue_ttm", "us_sue_rev")):
            gap = g["period_end"].diff().dt.days
            d = g[src].diff().where((gap >= 70) & (gap <= 110))
            sd = d.shift(1).rolling(8, min_periods=4).std()
            g[dst] = (d / sd.where(sd > 0)).replace([np.inf, -np.inf], np.nan)
        parts.append(g[["entity", "period_end", "available_from", "us_sue", "us_sue_rev"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def surprise_features(rows: pd.DataFrame, table: pd.DataFrame, max_age_days: int = 200) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=rows.index, columns=["us_sue", "us_sue_rev"])
    if table.empty:
        return out
    t = table.sort_values(["available_from", "period_end"]).rename(
        columns={"available_from": "_af", "period_end": "_pe"}
    )
    left = rows[["date", "entity"]].reset_index().sort_values("date")
    j = pd.merge_asof(left, t, left_on="date", right_on="_af", by="entity", direction="backward").set_index("index")
    j = j.reindex(rows.index)
    ok = j["_pe"].notna() & ((j["date"] - j["_pe"]).dt.days <= max_age_days)
    for c in ("us_sue", "us_sue_rev"):
        out[c] = j[c].where(ok)
    return out


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(obj, default=str).encode()))
