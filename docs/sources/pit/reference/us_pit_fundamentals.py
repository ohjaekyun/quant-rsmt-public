"""Point-in-time (PIT) US fundamentals from SEC XBRL companyfacts.

Reuses the verified quarter logic in quantcio.data.us (quarterly_series: YTD -> quarter derivation,
unit filtering, form filtering, per-fact split adjustment, YoY; ttm_series: 4 consecutive quarters)
and adds:

- multi-CIK merge (predecessor + successor filer: Google Inc -> Alphabet, XOM 2026 holdco),
- per-FILING concept coalescing before quarterly_series():
    revenue = max over total-revenue concepts tagged in that filing for that period
              (Revenues > RevenueFromContract... at XOM/WMT/CELG; SalesRevenueNet > Revenues at PG
              FY12-14), else SalesRevenueGoodsNet + SalesRevenueServicesNet (KO/JNJ pre-2018);
    NI / EPS  = first concept present in priority order (EPS falls back to the continuing-ops EPS
              that PG's FY2014 10-Qs carry instead of EarningsPerShareDiluted),
- as-of snapshots: at filing date d only facts with filed <= d are visible and each period takes
  the LATEST value filed <= d (the original until an amendment/recast is filed - never a later one),
- instants (equity, share counts) with the same as-of rule,
- event table (one row per filing date) and a month-end panel: TTM EPS, P/E, P/B, ROE, operating
  margin, revenue YoY, usable from filed + 1 day.

Deps: numpy, pandas, quantcio (read-only import).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantcio.data.us import XBRL_CONCEPTS, quarterly_series, split_factor_after, ttm_series

PERIODIC_FORMS = frozenset({"10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-QT", "10-KT/A", "10-QT/A"})
REVENUE = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "RevenuesNetOfInterestExpense",
]
REVENUE_PARTS = ["SalesRevenueGoodsNet", "SalesRevenueServicesNet"]  # summed only if no total is tagged
NET_INCOME = ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]
OP_INCOME = ["OperatingIncomeLoss"]
EPS_DILUTED = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "IncomeLossFromContinuingOperationsPerDilutedShare"]
EQUITY = ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
NCI = ["MinorityInterest"]
SHARES_BS = ["CommonStockSharesOutstanding"]
SHARES_WAVG = ["WeightedAverageNumberOfDilutedSharesOutstanding"]
EPS_BASIC = ["EarningsPerShareBasic"]  # FIX: only to sanity-check share counts when diluted EPS is absent
DURATION = {  # metric -> (concepts, coalesce mode); metric names are quantcio's XBRL_CONCEPTS keys
    "revenue": (REVENUE, "max"),
    "net_income": (NET_INCOME, "prio"),
    "operating_income": (OP_INCOME, "prio"),
    "eps_diluted": (EPS_DILUTED, "prio"),
}
_WANT_GAAP = set(REVENUE + REVENUE_PARTS + NET_INCOME + OP_INCOME + EPS_DILUTED + EQUITY + NCI + SHARES_BS + SHARES_WAVG + EPS_BASIC)
_WANT_DEI = {"EntityCommonStockSharesOutstanding"}
_WINDOW_DAYS = 1200  # a snapshot needs ~9 quarters back (TTM, YoY, avg equity, Q4 = FY - 9M)
_TOL = pd.Timedelta(3, "D")
_FAR = "2099-12-31"


def _D(n: int) -> pd.Timedelta:
    return pd.Timedelta(n, "D")


def _coalesce(gaap: dict, concepts: list[str], mode: str, parts: list[str] | None = None) -> dict[str, list[dict]]:
    """unit -> one duration fact per (accn, start, end). mode 'prio': first concept present in the
    filing; 'max': largest value. ``parts`` are summed when no main concept exists for that key."""
    best: dict[tuple, dict] = {}
    for prio, c in enumerate(concepts):
        for unit, arr in (gaap.get(c) or {}).items():
            for f in arr:
                if "start" not in f:
                    continue
                k = (unit, f["accn"], f["start"], f["end"])
                cur = best.get(k)
                if cur is None or (mode == "max" and f["val"] > cur["val"]):
                    best[k] = {**f, "src_concept": c, "src_prio": prio}
    extra: dict[tuple, dict] = {}
    for c in parts or []:
        for unit, arr in (gaap.get(c) or {}).items():
            for f in arr:
                k = (unit, f["accn"], f.get("start"), f["end"])
                if "start" in f and k not in best:
                    e = extra.setdefault(k, {**f, "val": 0.0, "src_concept": "+".join(parts), "src_prio": len(concepts)})
                    e["val"] += f["val"]
    out: dict[str, list[dict]] = {}
    for (unit, *_), f in list(best.items()) + list(extra.items()):
        out.setdefault(unit, []).append(f)
    return out


@dataclass
class CompanyFacts:
    """Periodic-form (10-K/10-Q and amendments) facts of one company, possibly from several CIKs."""

    ciks: list[int]
    gaap: dict[str, dict[str, list[dict]]] = field(default_factory=dict)  # concept -> unit -> facts
    dei: dict[str, dict[str, list[dict]]] = field(default_factory=dict)
    duration: dict[str, dict[str, list[dict]]] = field(default_factory=dict)  # metric -> unit -> coalesced facts
    events: list[str] = field(default_factory=list)  # distinct filed dates (ISO)

    @classmethod
    def from_json(cls, facts_by_cik: dict[int, dict]) -> CompanyFacts:
        """``facts_by_cik``: {cik: companyfacts JSON}. Pass predecessor CIKs too (same company)."""
        cf = cls(ciks=list(facts_by_cik))
        for cik, j in facts_by_cik.items():
            for ns, want, dst in (("us-gaap", _WANT_GAAP, cf.gaap), ("dei", _WANT_DEI, cf.dei)):
                g = (j.get("facts") or {}).get(ns) or {}
                for c in want & set(g):
                    for unit, arr in (g[c].get("units") or {}).items():
                        keep = [{**f, "cik": cik} for f in arr if f.get("form") in PERIODIC_FORMS]
                        if keep:
                            dst.setdefault(c, {}).setdefault(unit, []).extend(keep)
        for m, (concepts, mode) in DURATION.items():
            cf.duration[m] = _coalesce(cf.gaap, concepts, mode, REVENUE_PARTS if m == "revenue" else None)
        filed = {f["filed"] for src in (cf.gaap, cf.dei) for c in src.values() for arr in c.values() for f in arr}
        cf.events = sorted(filed)
        return cf

    def _snap(self, metric: str, asof: str, lo: str) -> dict:
        """companyfacts-shaped dict (facts filed <= asof, ending >= lo) keyed so quarterly_series(metric) reads it."""
        units = {}
        for u, arr in self.duration.get(metric, {}).items():
            sel = [f for f in arr if f["filed"] <= asof and f["end"] >= lo]
            if sel:
                units[u] = sel
        return {"facts": {"us-gaap": {XBRL_CONCEPTS[metric][0]: {"units": units}}}} if units else {"facts": {}}


def pit_quarterly(cf: CompanyFacts, metric: str, asof: str = _FAR, splits: pd.Series | None = None, lo: str = "1990-01-01") -> pd.DataFrame:
    """Quarterly series of ``metric`` exactly as known after the filings of day ``asof``.
    Output = quantcio quarterly_series columns [start, end, value, yoy, qoq, derived, concept, unit,
    form, accn, filed, first_filed]; ``concept`` is quantcio's key, the source concept is lost."""
    return quarterly_series(cf._snap(metric, asof, lo), metric, splits=splits if metric == "eps_diluted" else None)


def pit_instant(cf: CompanyFacts, concepts: list[str], asof: str = _FAR, lo: str = "1990-01-01", ns: str = "us-gaap") -> pd.DataFrame:
    """Instant facts known at ``asof``: DataFrame[end, val, filed, prio, concept, unit, accn], one row per
    ``end`` (best-priority concept, then latest filing <= asof)."""
    src = cf.gaap if ns == "us-gaap" else cf.dei
    rows = []
    for prio, c in enumerate(concepts):
        for u, arr in (src.get(c) or {}).items():
            for f in arr:
                if "start" not in f and f["filed"] <= asof and f["end"] >= lo:
                    rows.append((f["end"], f["val"], f["filed"], prio, c, u, f["accn"]))
    cols = ["end", "val", "filed", "prio", "concept", "unit", "accn"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols).sort_values(["end", "prio", "filed"], ascending=[True, True, False])
    df = df.drop_duplicates("end").copy()
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df.reset_index(drop=True)


def _at(df: pd.DataFrame, when: pd.Timestamp, col: str, tol: pd.Timedelta = _TOL) -> float:
    if df is None or df.empty:
        return np.nan
    m = df[(df["end"] - when).abs() <= tol]
    return float(m[col].iloc[-1]) if len(m) else np.nan


def _split_mult(when: pd.Timestamp, splits: pd.Series | None) -> float:
    """Product of split ratios dated strictly after ``when`` (count -> today's share basis)."""
    if splits is None or not len(splits):
        return 1.0
    return float(split_factor_after(pd.Series([pd.Timestamp(when)]), splits).iloc[0])


def _fy_eps(cf: CompanyFacts, q_end: pd.Timestamp, asof: str, splits: pd.Series | None) -> float | None:
    """Reported fiscal-year diluted EPS ending at ``q_end`` (latest filing <= asof), today's share basis."""
    best = None
    for f in cf.duration["eps_diluted"].get("USD/shares", []):
        if f["filed"] > asof or abs((pd.Timestamp(f["end"]) - q_end).days) > 3:
            continue
        if 350 <= (pd.Timestamp(f["end"]) - pd.Timestamp(f["start"])).days <= 380 and (best is None or f["filed"] > best["filed"]):
            best = f
    return None if best is None else float(best["val"]) / _split_mult(pd.Timestamp(best["filed"]), splits)


def _period_implied(cf: CompanyFacts, asof: str, lo: str, splits: pd.Series | None, eps_concepts: list[str]) -> float:
    """Implied share count (today's basis) from the latest period (filed <= asof) that has both NI and an
    ``eps_concepts`` EPS with |EPS as filed| >= 0.05. NI available to common is preferred over NI
    (TARP-era preferred dividends: JPM/BAC 2009 NI/EPS was 2-2.5x the share count)."""
    def latest(arrs):
        best = {}
        for f in arrs:
            if "start" in f and f["filed"] <= asof and f["end"] >= lo:
                k = (f["start"], f["end"])
                if k not in best or f["filed"] > best[k]["filed"]:
                    best[k] = f
        return best
    ni = latest(cf.duration["net_income"].get("USD", []))
    ni.update(latest((cf.gaap.get("NetIncomeLossAvailableToCommonStockholdersBasic") or {}).get("USD", [])))
    eps = latest(f for c in eps_concepts for f in (cf.gaap.get(c) or {}).get("USD/shares", []))
    for k in sorted(set(ni) & set(eps), key=lambda k: (k[1], k[0]), reverse=True):
        e = eps[k]
        if abs(e["val"]) >= 0.05:
            return float(ni[k]["val"]) / (float(e["val"]) / _split_mult(pd.Timestamp(e["filed"]), splits))
    return np.nan


def snapshot_features(cf: CompanyFacts, asof: str, splits: pd.Series | None = None) -> dict:
    """Fundamentals known after the filings of day ``asof`` (ISO date). Per-share values are on TODAY's
    share basis when ``splits`` (Series date -> ratio, every split up to today) is given."""
    d = pd.Timestamp(asof)
    lo = (d - _D(_WINDOW_DAYS)).strftime("%Y-%m-%d")
    rev, ni, op, eps = (pit_quarterly(cf, m, asof, splits, lo) for m in ("revenue", "net_income", "operating_income", "eps_diluted"))
    if not eps.empty and set(eps["unit"].str.upper()) != {"USD/SHARES"}:
        eps = eps.iloc[0:0]
    ends = [s["end"].iloc[-1] for s in (rev, ni, eps) if not s.empty]
    out: dict = {"asof": d}
    if not ends:
        return out
    q_end = max(ends)
    out["q_end"] = q_end
    t_rev, t_ni, t_op, t_eps = (ttm_series(s) for s in (rev, ni, op, eps))
    out["rev_q"] = _at(rev, q_end, "value")
    out["rev_yoy_q"] = _at(rev, q_end, "yoy")
    out["ttm_rev"] = _at(t_rev, q_end, "ttm")
    out["ttm_ni"] = _at(t_ni, q_end, "ttm")
    out["ttm_op"] = _at(t_op, q_end, "ttm")
    out["ttm_eps"] = _at(t_eps, q_end, "ttm")
    out["eps_src"] = "xbrl_eps" if not np.isnan(out["ttm_eps"]) else None
    fy = _fy_eps(cf, q_end, asof, splits)  # at a fiscal year end the reported FY EPS beats Q1+Q2+Q3+(FY-9M)
    if fy is not None:
        out["ttm_eps"], out["eps_src"] = fy, "xbrl_fy_eps"
    rev_1y = _at(t_rev, q_end - _D(364), "ttm", _D(16))
    out["rev_yoy_ttm"] = out["ttm_rev"] / rev_1y - 1 if rev_1y > 0 else np.nan
    out["op_margin"] = out["ttm_op"] / out["ttm_rev"] if out["ttm_rev"] > 0 else np.nan
    # parent equity: StockholdersEquity, else total equity - MinorityInterest of the same date
    eq = pit_instant(cf, EQUITY, asof, lo)
    nci = pit_instant(cf, NCI, asof, lo)
    if not eq.empty and eq["prio"].gt(0).any() and not nci.empty:
        eq = eq.merge(nci[["end", "val"]].rename(columns={"val": "nci"}), on="end", how="left")
        adj = eq["prio"].gt(0)
        eq.loc[adj, "val"] = eq.loc[adj, "val"] - eq.loc[adj, "nci"].fillna(0.0)
    e0 = _at(eq, q_end, "val")
    e1 = _at(eq, q_end - _D(364), "val", _D(16))
    out["equity"] = e0
    out["equity_src"] = _at(eq, q_end, "prio")  # 0 = StockholdersEquity, 1 = total - NCI
    avg = (e0 + e1) / 2 if not (np.isnan(e0) or np.isnan(e1)) else np.nan
    out["roe"] = out["ttm_ni"] / avg if avg > 0 else np.nan
    # shares on today's basis: dei cover count -> balance-sheet count -> diluted weighted average,
    # each accepted only within [0.67, 1.5] x the NI/EPS-implied diluted count (catches one-class-only
    # cover counts of multi-class filers) and at most 200 days old
    # FIX: the 0.05 guard is meant for EPS as filed (2-decimal rounding); ttm_eps is on TODAY's basis, so a
    # 70x-split name (NFLX) skipped the check for years and let a 7x-wrong cover count through
    eps_as_filed = out["ttm_eps"] * _split_mult(d, splits)
    implied = out["ttm_ni"] / out["ttm_eps"] if abs(eps_as_filed) >= 0.05 else np.nan
    has_dil = any(f["filed"] <= asof and f["end"] >= lo for f in cf.duration["eps_diluted"].get("USD/shares", []))
    if not has_dil:  # FIX: no diluted EPS at all (BRK, V: per-class dimensional EPS) -> check against basic EPS
        implied = _period_implied(cf, asof, lo, splits, EPS_BASIC)
    no_check = not has_dil and np.isnan(implied)  # nothing to validate a share count against
    cands = []
    cov = pit_instant(cf, ["EntityCommonStockSharesOutstanding"], asof, lo, ns="dei")
    if not cov.empty:  # a cover count is AS OF its own date -> splits after `end` ...
        r = cov.iloc[-1]
        m_end, m_filed = _split_mult(r["end"], splits), _split_mult(r["filed"], splits)
        if m_end == m_filed:
            cands.append(("dei_cover", r["val"] * m_end, r["end"]))
        elif not np.isnan(implied) and implied > 0:
            # FIX ... but some filers tag it with the PERIOD-END context and a post-split value (NFLX 10-Q
            # 2015-07-17: end 2015-06-30, split 2015-07-15). A split inside (end, filed] -> pick the basis
            # closer to the NI/EPS-implied count; with nothing to compare against, drop the candidate
            v = min((r["val"] * m for m in (m_end, m_filed)), key=lambda x: abs(np.log(x / implied)))
            cands.append(("dei_cover", v, r["end"]))
    bs = pit_instant(cf, SHARES_BS, asof, lo)
    if not bs.empty:  # statement values are restated for splits before filing -> splits after `filed`
        r = bs.iloc[-1]
        cands.append(("bs_shares", r["val"] * _split_mult(r["filed"], splits), r["end"]))
    wa = [f for arr in (cf.gaap.get(SHARES_WAVG[0]) or {}).values() for f in arr
          if "start" in f and f["filed"] <= asof and f["end"] >= lo]
    if wa:
        f = max(wa, key=lambda x: (x["end"], x["filed"]))
        cands.append(("wavg_diluted", f["val"] * _split_mult(pd.Timestamp(f["filed"]), splits), pd.Timestamp(f["end"])))
    out["shares"], out["shares_src"] = np.nan, None
    # FIX: drop zero / mis-scaled counts (KO 2009 cover count 0; AGN 2013-10-31 balance-sheet count 7.0)
    fresh = [(s_, v_, e_) for s_, v_, e_ in cands if (d - e_).days <= 200 and v_ >= 1e4]
    for src, val, end in fresh:
        if no_check:  # FIX: no EPS at all -> require a second, independent source within [0.67, 1.5]x
            ok = any(o != src and 0.67 <= val / ov <= 1.5 for o, ov, _ in fresh)  # (BRK 2011: class-A-only count)
        else:
            ok = np.isnan(implied) or implied <= 0 or 0.67 <= val / implied <= 1.5
        if ok:
            out["shares"], out["shares_src"] = float(val), src
            break
    out["shares_implied"] = implied
    out["bvps"] = e0 / out["shares"] if out["shares"] > 0 else np.nan
    if np.isnan(out["ttm_eps"]) and out["shares"] > 0 and not np.isnan(out["ttm_ni"]):
        out["ttm_eps"], out["eps_src"] = out["ttm_ni"] / out["shares"], "ni/shares"  # GOOGL 2015-18: per-class EPS only
    out["q_derived"] = bool(eps["derived"].iloc[-1]) if not eps.empty else None
    return out


def pit_feature_events(cf: CompanyFacts, splits: pd.Series | None = None, since: str = "2009-06-01") -> pd.DataFrame:
    """One row per filing date: fundamentals known after that day's filings.
    ``available`` = filed + 1 day (10-Q/10-K often land after the close)."""
    ev = pd.DataFrame([snapshot_features(cf, e, splits) for e in cf.events if e >= since])
    if not ev.empty:
        ev["available"] = ev["asof"] + _D(1)
    return ev


FEATURES = ["ttm_eps", "ttm_ni", "ttm_rev", "ttm_op", "rev_q", "rev_yoy_q", "rev_yoy_ttm", "op_margin", "roe", "equity", "shares", "bvps"]


def month_end_panel(ev: pd.DataFrame, close: pd.Series | None, start: str = "2011-01-31", end: str = "2026-08-31", max_age_days: int = 200) -> pd.DataFrame:
    """Month-end PIT features (+ pe, ey, pb, bm when ``close`` is given). ``close`` must be split-adjusted,
    NOT dividend-adjusted, on the same share basis as the ``splits`` given to the events. Features whose
    latest quarter ended more than ``max_age_days`` before the month-end are NaN (stopped filing)."""
    grid = pd.DataFrame({"date": pd.date_range(start, end, freq="ME")})
    if ev.empty:
        return grid
    p = pd.merge_asof(grid, ev.sort_values("available"), left_on="date", right_on="available", direction="backward")
    stale = (p["date"] - p["q_end"]).dt.days > max_age_days
    p.loc[stale | p["q_end"].isna(), FEATURES] = np.nan
    if close is not None and len(close):
        px = close.sort_index()
        px = px[~px.index.duplicated()].rename("close").rename_axis("d").reset_index()
        p["close"] = pd.merge_asof(grid, px, left_on="date", right_on="d", direction="backward", tolerance=_D(7))["close"]
        p["pe"] = (p["close"] / p["ttm_eps"]).where(p["ttm_eps"] > 0)
        p["ey"] = p["ttm_eps"] / p["close"]  # defined for losses too (better ML input than P/E)
        p["pb"] = (p["close"] / p["bvps"]).where(p["bvps"] > 0)
        p["bm"] = p["bvps"] / p["close"]
    return p


# ------------------------------------------------------------------ XBRL-only split detection
_RATIOS = [2, 3, 4, 5, 7, 8, 10, 15, 20, 25, 30, 50, 1.5, 1.25, 2.5, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 8,
           1 / 10, 1 / 15, 1 / 20, 1 / 25, 1 / 30, 1 / 50, 2 / 3, 0.4]


def detect_splits_xbrl(cf: CompanyFacts, min_evidence: int = 2, tol: float = 0.03) -> pd.DataFrame:
    """Splits inferred from re-filed diluted EPS: the same period in a later filing = old / ratio.
    DataFrame[window_lo, window_hi, ratio, n]: the split happened in (window_lo, window_hi];
    ``window_hi`` is the first filing on the new basis. For delisted names (no Yahoo split events)."""
    arr = [f for c in EPS_DILUTED[:2] for f in (cf.gaap.get(c) or {}).get("USD/shares", []) if "start" in f]
    cols = ["window_lo", "window_hi", "ratio", "n"]  # + share_count_confirms
    if not arr:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(arr)[["start", "end", "val", "filed"]]
    ev = []
    for _, g in df.sort_values("filed").groupby(["start", "end"]):
        vs = g.drop_duplicates("filed", keep="last")[["val", "filed"]].to_numpy()
        for (v0, f0), (v1, f1) in zip(vs[:-1], vs[1:]):
            if abs(v0) < 0.05 or abs(v1) < 0.01 or np.sign(v0) != np.sign(v1):
                continue
            k = min(_RATIOS, key=lambda x: abs(v0 / v1 / x - 1))
            if abs(v0 / v1 / k - 1) < tol:
                ev.append((f0, f1, k))
    out: list[list] = []
    for lo_, hi_, k in sorted(ev, key=lambda x: (x[1], x[0])):  # FIX: merge with ANY overlapping same-ratio window
        for c in out:
            if c[2] == k and lo_ < c[1] and hi_ > c[0]:
                c[0], c[1], c[3] = max(c[0], lo_), min(c[1], hi_), c[3] + 1
                break
        else:
            out.append([lo_, hi_, k, 1])
    res = pd.DataFrame(out, columns=cols)
    res = res[res["n"] >= min_evidence].reset_index(drop=True)
    # confirm with the dei cover share count: the count must jump by ~ratio across the window
    cov = sorted((f["filed"], f["val"]) for arr in (cf.dei.get("EntityCommonStockSharesOutstanding") or {}).values() for f in arr)
    conf = []
    for r in res.itertuples():
        before = [v for fd, v in cov if fd <= r.window_lo]
        after = [v for fd, v in cov if fd >= r.window_hi]
        ok = before and after and before[-1] > 0
        conf.append(bool(0.85 <= after[0] / before[-1] / r.ratio <= 1.15) if ok else None)  # None = no cover counts
    res["share_count_confirms"] = conf
    return res


def splits_series(det: pd.DataFrame) -> pd.Series:
    """detect_splits_xbrl() -> Series(date -> ratio) usable as ``splits`` (date = first new-basis filing)."""
    if det.empty:
        return pd.Series(dtype="float64")
    return pd.Series(det["ratio"].to_numpy(dtype=float), index=pd.to_datetime(det["window_hi"])).sort_index()
