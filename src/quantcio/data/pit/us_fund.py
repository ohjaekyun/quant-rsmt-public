"""Point-in-time US fundamentals from SEC XBRL companyfacts (keyless; verified + fixed 2026-09-25).

As-of rule: a snapshot at filing date d sees only facts with filed <= d and each period takes the LATEST value filed
<= d (the original until an amendment/recast is filed, never a later one). ``available_from`` = d + 1 NYSE session.

- Concepts coalesced per filing: revenue = max over the total-revenue concepts tagged in that filing for the period
  (Revenues vs RevenueFromContract at XOM/WMT/CELG; PG FY12-14 'Revenues' 28.4B vs SalesRevenueNet 82.0B), else
  goods + services; NI / EPS = first concept present (ProfitLoss minus the NCI share when NetIncomeLoss is absent).
- Per-share values and share counts are on the basis of the snapshot date (only splits known by d are applied), so no
  future split leaks into a level. Share counts (dei cover -> balance sheet -> diluted weighted average) are accepted
  only within [0.67, 1.5]x the NI/EPS-implied count on the AS-FILED basis; filers without diluted EPS are checked
  against basic EPS, and with no EPS at all two independent sources must agree (multi-class BRK/V -> NaN, not a
  class-A-only count). A cover count tagged at the period end with a post-split value (NFLX 2015) takes the basis
  closer to the implied count. Zero / mis-scaled counts (KO 2009 = 0, AGN 2013 = 7.0) are dropped.
- Several CIKs per entity (predecessor chain: Google Inc -> Alphabet, XOM 34088 -> 2115436); a predecessor's facts
  are used only up to the successor's first periodic filing + 120 days.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantcio.data.pit import FUNDAMENTAL_COLUMNS
from quantcio.data.pit.us_calendar import next_sessions
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
NI_NCI = "NetIncomeLossAttributableToNoncontrollingInterest"
OP_INCOME = ["OperatingIncomeLoss"]
EPS_DILUTED = [
    "EarningsPerShareDiluted",
    "EarningsPerShareBasicAndDiluted",
    "IncomeLossFromContinuingOperationsPerDilutedShare",
]
EQUITY = ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
NCI = ["MinorityInterest"]
SHARES_BS = ["CommonStockSharesOutstanding"]
SHARES_WAVG = ["WeightedAverageNumberOfDilutedSharesOutstanding"]
EPS_BASIC = ["EarningsPerShareBasic"]  # only to sanity-check share counts when diluted EPS is absent
DURATION = {  # metric -> (concepts, coalesce mode); metric names are quantcio's XBRL_CONCEPTS keys
    "revenue": (REVENUE, "max"),
    "net_income": (NET_INCOME, "prio"),
    "operating_income": (OP_INCOME, "prio"),
    "eps_diluted": (EPS_DILUTED, "prio"),
}
WANT_GAAP = frozenset(
    REVENUE
    + REVENUE_PARTS
    + NET_INCOME
    + [NI_NCI]
    + OP_INCOME
    + EPS_DILUTED
    + EQUITY
    + NCI
    + SHARES_BS
    + SHARES_WAVG
    + EPS_BASIC
)
WANT_DEI = frozenset({"EntityCommonStockSharesOutstanding"})
WINDOW_DAYS = 1200  # a snapshot needs ~9 quarters back (TTM, YoY, avg equity, Q4 = FY - 9M)
PREDECESSOR_OVERLAP_DAYS = 120
RESTATE_TOL = 0.005
_TOL = pd.Timedelta(3, "D")
_FAR = "2099-12-31"


def _d(n: int) -> pd.Timedelta:
    return pd.Timedelta(n, "D")


def trim_companyfacts(j: dict) -> dict:
    """Keep only the concepts used here (the full JSON is 3-8 MB; trimmed ~0.3 MB)."""
    facts = j.get("facts") or {}
    g = facts.get("us-gaap") or {}
    dei = facts.get("dei") or {}
    return {
        "cik": j.get("cik"),
        "entityName": j.get("entityName"),
        "facts": {
            "us-gaap": {c: g[c] for c in WANT_GAAP if c in g},
            "dei": {c: dei[c] for c in WANT_DEI if c in dei},
        },
    }


def first_periodic_filed(j: dict) -> str | None:
    """Earliest ``filed`` of a periodic-form fact in a companyfacts JSON (ISO date) or None."""
    best = None
    for ns in ("us-gaap", "dei"):
        for c in ((j.get("facts") or {}).get(ns) or {}).values():
            for arr in (c.get("units") or {}).values():
                for f in arr:
                    if f.get("form") in PERIODIC_FORMS and (best is None or f["filed"] < best):
                        best = f["filed"]
    return best


def _coalesce(gaap: dict, concepts: list[str], mode: str, parts: list[str] | None = None) -> dict[str, list[dict]]:
    """unit -> one duration fact per (accn, start, end). 'prio': first concept present in the filing; 'max': the
    largest value. ``parts`` are summed when no main concept exists for that key."""
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
                    e = extra.setdefault(
                        k, {**f, "val": 0.0, "src_concept": "+".join(parts), "src_prio": len(concepts)}
                    )
                    e["val"] += f["val"]
    out: dict[str, list[dict]] = {}
    for (unit, *_), f in list(best.items()) + list(extra.items()):
        out.setdefault(unit, []).append(f)
    return out


def _net_income_parent(gaap: dict, coalesced: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """ProfitLoss includes the non-controlling interests: subtract the same filing's NCI share when available."""
    nci = {
        (u, f["accn"], f.get("start"), f["end"]): f["val"]
        for u, arr in (gaap.get(NI_NCI) or {}).items()
        for f in arr
        if "start" in f
    }
    if not nci:
        return coalesced
    out: dict[str, list[dict]] = {}
    for u, arr in coalesced.items():
        new = []
        for f in arr:
            k = (u, f["accn"], f["start"], f["end"])
            if f.get("src_concept") == "ProfitLoss" and k in nci:
                f = {**f, "val": f["val"] - nci[k], "src_concept": "ProfitLoss-NCI"}
            new.append(f)
        out[u] = new
    return out


@dataclass
class CompanyFacts:
    """Periodic-form (10-K/10-Q and amendments) facts of one entity, possibly from several CIKs."""

    ciks: list[int]
    gaap: dict[str, dict[str, list[dict]]] = field(default_factory=dict)  # concept -> unit -> facts
    dei: dict[str, dict[str, list[dict]]] = field(default_factory=dict)
    duration: dict[str, dict[str, list[dict]]] = field(default_factory=dict)  # metric -> unit -> coalesced facts
    events: list[str] = field(default_factory=list)  # distinct filed dates (ISO)
    names: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_json(cls, facts_by_cik: dict[int, dict], cutoffs: dict[int, str] | None = None) -> CompanyFacts:
        """``facts_by_cik``: {cik: companyfacts JSON}; ``cutoffs``: {cik: last usable filed date} (predecessors)."""
        cf = cls(ciks=list(facts_by_cik))
        for cik, j in facts_by_cik.items():
            cf.names[cik] = str(j.get("entityName") or "")
            cut = (cutoffs or {}).get(cik) or _FAR
            for ns, want, dst in (("us-gaap", WANT_GAAP, cf.gaap), ("dei", WANT_DEI, cf.dei)):
                g = (j.get("facts") or {}).get(ns) or {}
                for c in want & set(g):
                    for unit, arr in (g[c].get("units") or {}).items():
                        keep = [{**f, "cik": cik} for f in arr if f.get("form") in PERIODIC_FORMS and f["filed"] <= cut]
                        if keep:
                            dst.setdefault(c, {}).setdefault(unit, []).extend(keep)
        for m, (concepts, mode) in DURATION.items():
            cf.duration[m] = _coalesce(cf.gaap, concepts, mode, REVENUE_PARTS if m == "revenue" else None)
        cf.duration["net_income"] = _net_income_parent(cf.gaap, cf.duration["net_income"])
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


def pit_quarterly(
    cf: CompanyFacts, metric: str, asof: str = _FAR, splits: pd.Series | None = None, lo: str = "1990-01-01"
) -> pd.DataFrame:
    """Quarterly series of ``metric`` exactly as known after the filings of day ``asof`` (quantcio quarterly_series
    columns [start, end, value, yoy, qoq, derived, concept, unit, form, accn, filed, first_filed])."""
    return quarterly_series(cf._snap(metric, asof, lo), metric, splits=splits if metric == "eps_diluted" else None)


def pit_instant(
    cf: CompanyFacts, concepts: list[str], asof: str = _FAR, lo: str = "1990-01-01", ns: str = "us-gaap"
) -> pd.DataFrame:
    """Instant facts known at ``asof``: one row per ``end`` (best-priority concept, then latest filing <= asof)."""
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
    """Product of split ratios dated strictly after ``when``."""
    if splits is None or not len(splits):
        return 1.0
    return float(split_factor_after(pd.Series([pd.Timestamp(when)]), splits).iloc[0])


def _fy_eps(cf: CompanyFacts, q_end: pd.Timestamp, asof: str, splits: pd.Series | None) -> float | None:
    """Reported fiscal-year diluted EPS ending at ``q_end`` (latest filing <= asof)."""
    best = None
    for f in cf.duration["eps_diluted"].get("USD/shares", []):
        if f["filed"] > asof or abs((pd.Timestamp(f["end"]) - q_end).days) > 3:
            continue
        span = (pd.Timestamp(f["end"]) - pd.Timestamp(f["start"])).days
        if 350 <= span <= 380 and (best is None or f["filed"] > best["filed"]):
            best = f
    return None if best is None else float(best["val"]) / _split_mult(pd.Timestamp(best["filed"]), splits)


def _period_implied(cf: CompanyFacts, asof: str, lo: str, splits: pd.Series | None, eps_concepts: list[str]) -> float:
    """Implied share count from the latest period (filed <= asof) with both NI and an ``eps_concepts`` EPS with
    |EPS as filed| >= 0.05 (NI available to common preferred: TARP-era preferred dividends)."""

    def latest(arrs) -> dict:
        best: dict = {}
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


def snapshot_features(
    cf: CompanyFacts, asof: str, splits: pd.Series | None = None, max_share_age_days: int = 200
) -> dict[str, Any]:
    """Fundamentals known after the filings of day ``asof`` (ISO date). Per-share values and share counts are on the
    share basis implied by ``splits`` (pass only the splits known by ``asof`` for a PIT basis)."""
    d = pd.Timestamp(asof)
    lo = (d - _d(WINDOW_DAYS)).strftime("%Y-%m-%d")
    rev, ni, op, eps = (
        pit_quarterly(cf, m, asof, splits, lo) for m in ("revenue", "net_income", "operating_income", "eps_diluted")
    )
    if not eps.empty and set(eps["unit"].str.upper()) != {"USD/SHARES"}:
        eps = eps.iloc[0:0]
    ends = [s["end"].iloc[-1] for s in (rev, ni, eps) if not s.empty]
    out: dict[str, Any] = {"asof": d}
    if not ends:
        return out
    q_end = max(ends)
    out["q_end"] = q_end
    out["currency"] = _currency(rev if not rev.empty else ni)
    t_rev, t_ni, t_op, t_eps = (ttm_series(s) for s in (rev, ni, op, eps))
    out["rev_q"] = _at(rev, q_end, "value")
    out["ttm_rev"] = _at(t_rev, q_end, "ttm")
    out["ttm_ni"] = _at(t_ni, q_end, "ttm")
    out["ttm_op"] = _at(t_op, q_end, "ttm")
    out["ttm_eps"] = _at(t_eps, q_end, "ttm")
    out["eps_src"] = "xbrl_eps" if not np.isnan(out["ttm_eps"]) else None
    fy = _fy_eps(cf, q_end, asof, splits)  # at a fiscal year end the reported FY EPS beats Q1+Q2+Q3+(FY-9M)
    if fy is not None:
        out["ttm_eps"], out["eps_src"] = fy, "xbrl_fy_eps"
    out["period_type"] = "FY" if _has_annual(cf, q_end, asof) else "Q"
    # parent equity: StockholdersEquity, else total equity - MinorityInterest of the same date
    eq = pit_instant(cf, EQUITY, asof, lo)
    nci = pit_instant(cf, NCI, asof, lo)
    if not eq.empty and eq["prio"].gt(0).any() and not nci.empty:
        eq = eq.merge(nci[["end", "val"]].rename(columns={"val": "nci"}), on="end", how="left")
        adj = eq["prio"].gt(0)
        eq.loc[adj, "val"] = eq.loc[adj, "val"] - eq.loc[adj, "nci"].fillna(0.0)
    out["equity"] = _at(eq, q_end, "val")
    out["equity_src"] = _at(eq, q_end, "prio")  # 0 = StockholdersEquity, 1 = total - NCI
    _shares(cf, out, d, asof, lo, splits, max_share_age_days)
    if np.isnan(out["ttm_eps"]) and out["shares"] > 0 and not np.isnan(out["ttm_ni"]):
        out["ttm_eps"], out["eps_src"] = out["ttm_ni"] / out["shares"], "ni/shares"  # GOOGL 2015-18: per-class EPS
    return out


def _currency(q: pd.DataFrame) -> str:
    if q is None or q.empty:
        return "USD"
    u = str(q["unit"].iloc[-1]).upper()
    return u.split("/")[0] if u else "USD"


def _has_annual(cf: CompanyFacts, q_end: pd.Timestamp, asof: str) -> bool:
    for m in ("revenue", "net_income"):
        for arr in cf.duration.get(m, {}).values():
            for f in arr:
                if f["filed"] <= asof and abs((pd.Timestamp(f["end"]) - q_end).days) <= 3:
                    if 350 <= (pd.Timestamp(f["end"]) - pd.Timestamp(f["start"])).days <= 380:
                        return True
    return False


def _shares(
    cf: CompanyFacts,
    out: dict[str, Any],
    d: pd.Timestamp,
    asof: str,
    lo: str,
    splits: pd.Series | None,
    max_age: int,
) -> None:
    """Validated share count (see module docstring) -> out['shares'], out['shares_src'], out['shares_implied']."""
    eps_as_filed = out["ttm_eps"] * _split_mult(d, splits)
    implied = out["ttm_ni"] / out["ttm_eps"] if abs(eps_as_filed) >= 0.05 else np.nan
    has_dil = any(f["filed"] <= asof and f["end"] >= lo for f in cf.duration["eps_diluted"].get("USD/shares", []))
    if not has_dil:  # no diluted EPS at all (BRK, V: per-class dimensional EPS) -> check against basic EPS
        implied = _period_implied(cf, asof, lo, splits, EPS_BASIC)
    no_check = not has_dil and np.isnan(implied)
    cands = []
    cov = pit_instant(cf, ["EntityCommonStockSharesOutstanding"], asof, lo, ns="dei")
    if not cov.empty:  # a cover count is AS OF its own date -> splits after `end` ...
        r = cov.iloc[-1]
        m_end, m_filed = _split_mult(r["end"], splits), _split_mult(r["filed"], splits)
        if m_end == m_filed:
            cands.append(("dei_cover", r["val"] * m_end, r["end"]))
        elif not np.isnan(implied) and implied > 0 and r["val"] > 0:
            # ... but some filers tag it with the PERIOD-END context and a post-split value (NFLX 10-Q 2015-07-17)
            v = min((r["val"] * m for m in (m_end, m_filed)), key=lambda x: abs(np.log(x / implied)))
            cands.append(("dei_cover", v, r["end"]))
    bs = pit_instant(cf, SHARES_BS, asof, lo)
    if not bs.empty:  # statement values are restated for splits before filing -> splits after `filed`
        r = bs.iloc[-1]
        cands.append(("bs_shares", r["val"] * _split_mult(r["filed"], splits), r["end"]))
    wa = [
        f
        for arr in (cf.gaap.get(SHARES_WAVG[0]) or {}).values()
        for f in arr
        if "start" in f and f["filed"] <= asof and f["end"] >= lo
    ]
    if wa:
        f = max(wa, key=lambda x: (x["end"], x["filed"]))
        cands.append(("wavg_diluted", f["val"] * _split_mult(pd.Timestamp(f["filed"]), splits), pd.Timestamp(f["end"])))
    out["shares"], out["shares_src"] = np.nan, None
    fresh = [(s_, v_, e_) for s_, v_, e_ in cands if (d - e_).days <= max_age and v_ >= 1e4]
    for src, val, _end in fresh:
        if no_check:  # no EPS at all -> a second, independent source within [0.67, 1.5]x (BRK: class-A-only count)
            ok = any(o != src and 0.67 <= val / ov <= 1.5 for o, ov, _ in fresh)
        else:
            ok = np.isnan(implied) or implied <= 0 or 0.67 <= val / implied <= 1.5
        if ok:
            out["shares"], out["shares_src"] = float(val), src
            break
    out["shares_implied"] = implied


# ------------------------------------------------------------------ restatement flag


def _versions(cf: CompanyFacts) -> dict[tuple, list[tuple[str, float]]]:
    """(metric, unit, start, end) -> [(filed, val)] sorted by filed (revenue, NI, equity versions)."""
    out: dict[tuple, list[tuple[str, float]]] = {}
    for m in ("revenue", "net_income"):
        for u, arr in cf.duration.get(m, {}).items():
            for f in arr:
                out.setdefault((m, u, f["start"], f["end"]), []).append((f["filed"], float(f["val"])))
    for c in EQUITY[:1]:
        for u, arr in (cf.gaap.get(c) or {}).items():
            for f in arr:
                if "start" not in f:
                    out.setdefault(("equity", u, None, f["end"]), []).append((f["filed"], float(f["val"])))
    for k in out:
        out[k].sort()
    return out


def restated_after(versions: dict[tuple, list[tuple[str, float]]], asof: str, q_end: pd.Timestamp) -> bool:
    """True if a value used at ``asof`` (periods ending in the TTM window of ``q_end``) was changed by > 0.5% in a
    filing made after ``asof``."""
    lo = (q_end - _d(370)).strftime("%Y-%m-%d")
    hi = (q_end + _TOL).strftime("%Y-%m-%d")
    for (_m, _u, _s, end), vs in versions.items():
        if not (lo < end <= hi):
            continue
        known = [v for f, v in vs if f <= asof]
        later = [v for f, v in vs if f > asof]
        if not known or not later:
            continue
        base = known[-1]
        if any(abs(v - base) > RESTATE_TOL * max(abs(base), 1.0) for v in later):
            return True
    return False


# ------------------------------------------------------------------ XBRL-only split detection

_RATIOS = [
    2,
    3,
    4,
    5,
    7,
    8,
    10,
    15,
    20,
    25,
    30,
    50,
    1.5,
    1.25,
    2.5,
    1 / 2,
    1 / 3,
    1 / 4,
    1 / 5,
    1 / 8,
    1 / 10,
    1 / 15,
    1 / 20,
    1 / 25,
    1 / 30,
    1 / 50,
    2 / 3,
    0.4,
]


def detect_splits_xbrl(cf: CompanyFacts, min_evidence: int = 2, tol: float = 0.03) -> pd.DataFrame:
    """Splits inferred from re-filed diluted EPS (the same period in a later filing = old / ratio).
    DataFrame[window_lo, window_hi, ratio, n, share_count_confirms]: the split happened in (window_lo, window_hi];
    ``window_hi`` is the first filing on the new basis. Windows are merged with ANY overlapping same-ratio window
    (a merge with only the last one emitted TSLA 2020 twice)."""
    arr = [f for c in EPS_DILUTED[:2] for f in (cf.gaap.get(c) or {}).get("USD/shares", []) if "start" in f]
    cols = ["window_lo", "window_hi", "ratio", "n", "share_count_confirms"]
    if not arr:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(arr)[["start", "end", "val", "filed"]]
    ev = []
    for _, g in df.sort_values("filed").groupby(["start", "end"]):
        vs = g.drop_duplicates("filed", keep="last")[["val", "filed"]].to_numpy()
        for (v0, f0), (v1, f1) in zip(vs[:-1], vs[1:], strict=False):
            if abs(v0) < 0.05 or abs(v1) < 0.01 or np.sign(v0) != np.sign(v1):
                continue
            k = min(_RATIOS, key=lambda x: abs(v0 / v1 / x - 1))
            if abs(v0 / v1 / k - 1) < tol:
                ev.append((f0, f1, k))
    out: list[list] = []
    for lo_, hi_, k in sorted(ev, key=lambda x: (x[1], x[0])):
        for c in out:
            if c[2] == k and lo_ < c[1] and hi_ > c[0]:
                c[0], c[1], c[3] = max(c[0], lo_), min(c[1], hi_), c[3] + 1
                break
        else:
            out.append([lo_, hi_, k, 1])
    res = pd.DataFrame(out, columns=cols[:4])
    res = res[res["n"] >= min_evidence].reset_index(drop=True)
    cov = sorted(
        (f["filed"], f["val"]) for arr in (cf.dei.get("EntityCommonStockSharesOutstanding") or {}).values() for f in arr
    )
    conf = []
    for r in res.itertuples():
        before = [v for fd, v in cov if fd <= r.window_lo]
        after = [v for fd, v in cov if fd >= r.window_hi]
        ok = before and after and before[-1] > 0
        conf.append(bool(0.85 <= after[0] / before[-1] / r.ratio <= 1.15) if ok else None)
    res["share_count_confirms"] = conf
    return res


def splits_series(det: pd.DataFrame) -> pd.Series:
    """detect_splits_xbrl() -> Series(date -> ratio) usable as ``splits`` (date = first new-basis filing)."""
    if det.empty:
        return pd.Series(dtype="float64")
    return pd.Series(det["ratio"].to_numpy(dtype=float), index=pd.to_datetime(det["window_hi"])).sort_index()


def combine_splits(yahoo: pd.Series | None, xbrl: pd.Series, yahoo_from: pd.Timestamp | None) -> pd.Series:
    """Yahoo split dates (exact) where the chart covers them; XBRL-detected splits before the chart window."""
    if yahoo is None or yahoo_from is None:
        return xbrl.sort_index()
    early = xbrl[xbrl.index < yahoo_from] if len(xbrl) else xbrl
    parts = [p for p in (early, yahoo.astype("float64")) if len(p)]
    if not parts:
        return pd.Series(dtype="float64")
    out = pd.concat(parts) if len(parts) > 1 else parts[0]
    return out[~out.index.duplicated(keep="last")].sort_index()


# ------------------------------------------------------------------ entity panel

PANEL_EXTRA = ["filed", "eps_ttm", "eps_source", "shares_source", "cik_chain"]


def event_fingerprint(facts_by_cik: dict[int, dict], cutoffs: dict[int, str] | None = None) -> dict[str, int]:
    """filed date -> number of periodic facts filed that day (used concepts only). Under the as-of rule a snapshot at
    d depends only on facts filed <= d, so a cached row stays valid while the fingerprint up to d is unchanged."""
    fp: dict[str, int] = {}
    for cik, j in facts_by_cik.items():
        cut = (cutoffs or {}).get(cik) or _FAR
        for ns, want in (("us-gaap", WANT_GAAP), ("dei", WANT_DEI)):
            g = (j.get("facts") or {}).get(ns) or {}
            for c in want & set(g):
                for arr in (g[c].get("units") or {}).values():
                    for f in arr:
                        if f.get("form") in PERIODIC_FORMS and f["filed"] <= cut:
                            fp[f["filed"]] = fp.get(f["filed"], 0) + 1
    return fp


def restate_flags(df: pd.DataFrame, facts_by_cik: dict[int, dict], cutoffs: dict[int, str] | None = None) -> pd.Series:
    """``restated_later`` for every row (it looks forward, so it is refreshed whenever new filings arrive)."""
    if df.empty:
        return pd.Series(dtype=bool)
    versions = _versions(CompanyFacts.from_json(facts_by_cik, cutoffs))
    vals = [
        restated_after(versions, pd.Timestamp(f).strftime("%Y-%m-%d"), pd.Timestamp(q))
        for f, q in zip(df["filed"], df["period_end"], strict=False)
    ]
    return pd.Series(vals, index=df.index, dtype=bool)


def drop_unchanged(df: pd.DataFrame) -> pd.DataFrame:
    """A filing that changes nothing (same period and values as the previous row) adds no row."""
    if len(df) < 2:
        return df.reset_index(drop=True)
    vals = ["period_end", "revenue_ttm", "op_income_ttm", "ni_parent_ttm", "equity_parent", "shares_out"]
    prev = df[vals].shift()
    same = prev.eq(df[vals]) | (prev.isna() & df[vals].isna())
    keep = ~same.all(axis=1)
    keep.iloc[0] = True
    return df[keep].reset_index(drop=True)


def entity_panel(
    entity: str,
    facts_by_cik: dict[int, dict],
    cutoffs: dict[int, str] | None = None,
    yahoo_splits: pd.Series | None = None,
    yahoo_from: pd.Timestamp | None = None,
    since: str = "2009-06-01",
    max_share_age_days: int = 200,
    events_from: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One row per filing date d (>= ``since`` and >= ``events_from``): fundamentals known after d, available from
    d + 1 NYSE session. Returns (rows in FUNDAMENTAL_COLUMNS + PANEL_EXTRA, stats)."""
    cols = FUNDAMENTAL_COLUMNS + PANEL_EXTRA
    cf = CompanyFacts.from_json(facts_by_cik, cutoffs)
    xs = splits_series(detect_splits_xbrl(cf))
    splits = combine_splits(yahoo_splits, xs, yahoo_from)
    versions = _versions(cf)
    chain = ",".join(str(c) for c in cf.ciks)
    stats: dict[str, Any] = {"events": 0, "rows": 0, "no_period": 0, "xbrl_splits": len(xs), "splits": len(splits)}
    rows = []
    first = max(since, events_from or since)
    for e in cf.events:
        if e < first:
            continue
        stats["events"] += 1
        known = splits[splits.index <= pd.Timestamp(e)] if len(splits) else None
        s = snapshot_features(cf, e, known, max_share_age_days)
        if "q_end" not in s:
            stats["no_period"] += 1
            continue
        rev, op, ni = s["ttm_rev"], s["ttm_op"], s["ttm_ni"]
        if rev > 1e8:  # mis-scaled first filings (AGN 2013Q3 op income 970 vs 155.7M) -> NaN, not a wrong value
            op = np.nan if 0 < abs(op) < 1e4 else op
            ni = np.nan if 0 < abs(ni) < 1e4 else ni
        rows.append(
            {
                "entity": entity,
                "period_end": s["q_end"],
                "period_type": s["period_type"],
                "available_from": pd.Timestamp(e),  # shifted to the next session below (vectorised)
                "currency": s["currency"],
                "revenue_ttm": rev,
                "op_income_ttm": op,
                "ni_parent_ttm": ni,
                "equity_parent": s["equity"],
                "shares_out": s["shares"],
                "source": "SEC XBRL companyfacts (as-of)",
                "restated_later": restated_after(versions, e, s["q_end"]),
                "filed": pd.Timestamp(e),
                "eps_ttm": s["ttm_eps"],
                "eps_source": s["eps_src"],
                "shares_source": s["shares_src"],
                "cik_chain": chain,
            }
        )
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df["available_from"] = next_sessions(df["filed"])
        df["restated_later"] = df["restated_later"].astype(bool)
    stats["rows"] = len(df)
    return df, stats
