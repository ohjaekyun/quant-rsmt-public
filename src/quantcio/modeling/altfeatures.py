"""Features v3 (docs/MODELING.md 16.3): alternative-data window features and per-stock macro exposures.

Point in time (one rule, ``quantcio.data.alt``): an event / measurement is used at grid date t only when its
``published`` date is STRICTLY before t. Window features sum events with ``t - window <= published < t``.

Coverage: an entity whose source was not fetched completely (``AltData.covered_entities``) gets NaN for that
source's features (trees route it, linear models use the median 0 after rank-gauss); a covered entity without
events gets 0 - "no buyback filed" is information, "not fetched" is not. A source complete only up to a date
(``AltData.covered_through``, e.g. the quarterly SEC insider data sets) gives NaN for grid dates whose window
would reach past it.

Macro exposures (``macro_betas``): the coefficient of a 252-session (>= 126) OLS of the stock's daily log return
on [its benchmark's log return, the factor's daily change], computed from rolling covariances. Factor levels are
taken as of ``date - lag_days`` (KR: 1 day, US closes are known the next Korean morning) on the stock's own
calendar, so a change never uses a later US close.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantcio.data.alt import AltData

TRADING_DAYS = 252
BETA_MIN_OBS = 126
EMP_MAX_AGE_DAYS = 550  # an annual employee report whose fiscal year ended earlier than this is stale
EMP_YOY_TOL_DAYS = 45  # the previous report's period end must be 365 +- this many days earlier
NPS_MAX_AGE_DAYS = 120  # the latest usable NPS data month must end within this many days of t


@dataclass(frozen=True)
class EventSpec:
    """One window feature: sum over ``plus`` kinds minus ``minus`` kinds of ``value`` (counts carry value 1)."""

    name: str
    source: str  # the AltData ``source`` whose coverage decides NaN vs 0
    plus: tuple[str, ...]
    window: int  # calendar days
    minus: tuple[str, ...] = ()
    log1p: bool = False
    scale: str = ""  # "" | "shares" (divide by shares outstanding = mcap / close) | "mcap" (divide by market cap)


KR_EVENT_SPECS: tuple[EventSpec, ...] = (
    EventSpec("dart_filings_90", "kr_dart_list", ("filing",), 90, log1p=True),
    EventSpec("dart_buyback_180", "kr_dart_list", ("buyback", "share_cancel"), 180),
    EventSpec("dart_treasury_sale_180", "kr_dart_list", ("treasury_disposal",), 180),
    EventSpec("dart_dilution_365", "kr_dart_list", ("rights_issue", "convertible_issue"), 365),
    EventSpec("dart_supply_90", "kr_dart_list", ("supply_contract",), 90, minus=("supply_contract_cancel",)),
    EventSpec("dart_ma_365", "kr_dart_list", ("ma_deal", "asset_sale"), 365),
    EventSpec("dart_governance_365", "kr_dart_list", ("major_holder_change", "ceo_change"), 365),
    EventSpec("dart_distress_365", "kr_dart_list", ("unfaithful", "admin_issue", "lawsuit"), 365),
    EventSpec("dart_inquiry_90", "kr_dart_list", ("inquiry",), 90),
    EventSpec("dart_correction_180", "kr_dart_list", ("correction",), 180),
)
# FORWARD-ONLY (docs/MODELING.md 16.8, amended before the v3 run): OpenDART elestock / majorstock return only the
# last ~2 years (first answer 2024-10-04), so 2016-2024 would read as "no insider activity". Not model inputs until
# the accumulating cache (``DartAltClient.whole_history`` merges every answer) covers a backtestable span.
KR_FORWARD_ONLY_SPECS: tuple[EventSpec, ...] = (
    EventSpec("insider_net_90", "kr_dart_insider", ("insider_net_shares",), 90, scale="shares"),
    EventSpec("insider_breadth_90", "kr_dart_insider", ("insider_buy_n",), 90, minus=("insider_sell_n",)),
    EventSpec("block_net_90", "kr_dart_block", ("block_pct_change",), 90),
)
KR_EMP_FEATURES = ("emp_growth", "emp_tenure", "emp_pay_growth", "emp_contract_share")
KR_EMP_SOURCE = "kr_dart_emp"
KR_NPS_FEATURES = ("nps_growth_3m", "nps_growth_12m", "nps_hire_3m", "nps_quit_3m")
KR_NPS_SOURCE = "kr_nps"
US_EVENT_SPECS: tuple[EventSpec, ...] = (
    EventSpec("sec_8k_90", "sec_forms", ("8k",), 90, log1p=True),
    EventSpec("sec_8k_material_90", "sec_forms", ("8k_1.01", "8k_2.01", "8k_2.03"), 90),
    EventSpec(
        "sec_8k_negative_365", "sec_forms", ("8k_1.03", "8k_2.06", "8k_3.01", "8k_4.01", "8k_4.02", "nt_late"), 365
    ),
    EventSpec("sec_officer_change_180", "sec_forms", ("8k_5.02",), 180),
    EventSpec("sec_form4_90", "sec_forms", ("form4",), 90, log1p=True),
    EventSpec("sec_form144_90", "sec_forms", ("form144",), 90),
    EventSpec("sec_activist_365", "sec_forms", ("sc13d",), 365),
    EventSpec("sec_issuance_365", "sec_forms", ("prospectus_424b", "s3"), 365),
    EventSpec("insider_buy_180", "sec_form345", ("insider_buy_value",), 180, scale="mcap"),
    EventSpec("insider_sell_180", "sec_form345", ("insider_sell_value",), 180, scale="mcap"),
    EventSpec("insider_breadth_90", "sec_form345", ("insider_buy_n",), 90, minus=("insider_sell_n",)),
)
MACRO_BETA_FEATURES = ("beta_fx", "beta_oil", "beta_rate", "beta_credit")


def alt_feature_names(market: str, nps: bool = False) -> list[str]:
    """Raw alternative-data feature names of ``market`` (KR: DART events + employees [+ NPS]; US: SEC)."""
    if str(market).upper() == "KR":
        return [*(s.name for s in KR_EVENT_SPECS), *KR_EMP_FEATURES, *(KR_NPS_FEATURES if nps else ())]
    return [s.name for s in US_EVENT_SPECS]


# ================================================================================================ events


def _window_sums(pub: np.ndarray, val: np.ndarray, t: np.ndarray, window_days: int) -> np.ndarray:
    """Sum of ``val`` with ``t - window <= pub < t`` for each t (``pub`` sorted ascending)."""
    cs = np.concatenate([[0.0], np.cumsum(val)])
    hi = np.searchsorted(pub, t, side="left")  # strictly before t
    lo = np.searchsorted(pub, t - np.timedelta64(int(window_days), "D"), side="left")
    return cs[hi] - cs[lo]


def event_features(rows: pd.DataFrame, alt: AltData, specs: Iterable[EventSpec]) -> pd.DataFrame:
    """Window features of ``specs`` for every (date, entity) row; needs ``mcap`` / ``close`` for scaled specs."""
    specs = list(specs)
    out = pd.DataFrame(np.nan, index=rows.index, columns=[s.name for s in specs], dtype=float)
    if rows.empty or not specs:
        return out
    ev = alt.events
    dates = rows["date"].to_numpy(dtype="datetime64[ns]")
    ents = rows["entity"].astype(str).to_numpy()
    cov_cache: dict[str, set[str]] = {}
    groups = {e: np.flatnonzero(ents == e) for e in pd.unique(ents)}
    ev_by = {e: g for e, g in ev.groupby(ev["entity"].astype(str))} if len(ev) else {}
    for s in specs:
        cov = cov_cache.setdefault(s.source, alt.covered_entities(s.source))
        col = np.full(len(rows), np.nan)
        for e, ix in groups.items():
            if e not in cov:
                continue
            g = ev_by.get(e)
            total = np.zeros(len(ix))
            if g is not None:
                g = g[g["source"] == s.source]
                for kinds, sign in ((s.plus, 1.0), (s.minus, -1.0)):
                    if not kinds:
                        continue
                    h = g[g["kind"].isin(kinds)]
                    if h.empty:
                        continue
                    h = h.sort_values("published", kind="stable")
                    total += sign * _window_sums(
                        h["published"].to_numpy(dtype="datetime64[ns]"),
                        h["value"].to_numpy(dtype=float),
                        dates[ix],
                        s.window,
                    )
            col[ix] = total
        until = alt.covered_through(s.source)
        if until is not None:  # a window reaching past the source's last complete day would undercount: NaN
            col[dates > np.datetime64(until + pd.Timedelta(1, "D"), "ns")] = np.nan
        if s.scale:
            mcap = rows["mcap"].to_numpy(dtype=float)
            if s.scale == "shares":
                shares = mcap / rows["close"].to_numpy(dtype=float)
                col = col / np.where(shares > 0, shares, np.nan)
            else:
                col = col / np.where(mcap > 0, mcap, np.nan)
        if s.log1p:
            col = np.log1p(np.clip(col, 0.0, None))
        out[s.name] = col
    return out


# ================================================================================================ measurements


def employee_features(rows: pd.DataFrame, alt: AltData) -> pd.DataFrame:
    """``KR_EMP_FEATURES`` from the annual employee measures (latest report published before t and not stale;
    the year-over-year partner is the same entity's report for the fiscal year one year earlier)."""
    out = pd.DataFrame(np.nan, index=rows.index, columns=list(KR_EMP_FEATURES), dtype=float)
    me = alt.measures
    me = me[me["source"] == KR_EMP_SOURCE] if len(me) else me
    if rows.empty or me.empty:
        return out
    cov = alt.covered_entities(KR_EMP_SOURCE)
    wide = me.pivot_table(index=["entity", "period_end", "published"], columns="item", values="value", aggfunc="first")
    wide = wide.reset_index().sort_values(["entity", "published"], kind="stable")
    # one row per (entity, fiscal period): the FIRST published version (conservative)
    wide = wide.drop_duplicates(["entity", "period_end"], keep="first")
    for c in ("headcount", "contract_workers", "avg_tenure_years", "salary_total"):
        if c not in wide:
            wide[c] = np.nan
    by = {e: g.reset_index(drop=True) for e, g in wide.groupby("entity")}
    ents = rows["entity"].astype(str).to_numpy()
    dates = rows["date"].to_numpy(dtype="datetime64[ns]")
    res = np.full((len(rows), len(KR_EMP_FEATURES)), np.nan)
    for e in pd.unique(ents):
        ix = np.flatnonzero(ents == e)
        if e not in cov:
            continue
        g = by.get(e)
        if g is None or g.empty:
            continue
        pub = g["published"].to_numpy(dtype="datetime64[ns]")
        pe = g["period_end"].to_numpy(dtype="datetime64[ns]")
        hc = g["headcount"].to_numpy(dtype=float)
        cw = g["contract_workers"].to_numpy(dtype=float)
        ten = g["avg_tenure_years"].to_numpy(dtype=float)
        pay = g["salary_total"].to_numpy(dtype=float) / np.where(hc > 0, hc, np.nan)
        for k, t in zip(ix, dates[ix], strict=True):
            known = np.flatnonzero(pub < t)
            if not len(known):
                continue
            j = known[np.argmax(pe[known])]  # latest fiscal period among the reports known at t
            if (t - pe[j]) > np.timedelta64(EMP_MAX_AGE_DAYS, "D"):
                continue
            row = [np.nan] * len(KR_EMP_FEATURES)
            prev = [i for i in known if abs(((pe[j] - pe[i]) / np.timedelta64(1, "D")) - 365) <= EMP_YOY_TOL_DAYS]
            if prev and hc[prev[-1]] > 0 and math.isfinite(hc[j]):
                row[0] = hc[j] / hc[prev[-1]] - 1.0
            row[1] = ten[j]
            if prev and np.isfinite(pay[prev[-1]]) and pay[prev[-1]] > 0 and np.isfinite(pay[j]):
                row[2] = pay[j] / pay[prev[-1]] - 1.0
            if hc[j] > 0:
                row[3] = (cw[j] if np.isfinite(cw[j]) else 0.0) / hc[j]
            res[k] = row
    out.loc[:, :] = res
    return out


def nps_features(rows: pd.DataFrame, alt: AltData) -> pd.DataFrame:
    """``KR_NPS_FEATURES`` from monthly NPS measures (nps_members / nps_new / nps_lost per data month)."""
    out = pd.DataFrame(np.nan, index=rows.index, columns=list(KR_NPS_FEATURES), dtype=float)
    me = alt.measures
    me = me[me["source"] == KR_NPS_SOURCE] if len(me) else me
    if rows.empty or me.empty:
        return out
    cov = alt.covered_entities(KR_NPS_SOURCE)
    wide = me.pivot_table(index=["entity", "period_end"], columns="item", values="value", aggfunc="sum")
    pubs = me.groupby(["entity", "period_end"])["published"].max()
    wide = wide.join(pubs).reset_index().sort_values(["entity", "period_end"], kind="stable")
    for c in ("nps_members", "nps_new", "nps_lost"):
        if c not in wide:
            wide[c] = np.nan
    by = {e: g.reset_index(drop=True) for e, g in wide.groupby("entity")}
    ents = rows["entity"].astype(str).to_numpy()
    dates = rows["date"].to_numpy(dtype="datetime64[ns]")
    res = np.full((len(rows), len(KR_NPS_FEATURES)), np.nan)
    for e in pd.unique(ents):
        if e not in cov or e not in by:
            continue
        g = by[e]
        month = g["period_end"].dt.to_period("M")
        pos = {m: i for i, m in enumerate(month)}
        pub = g["published"].to_numpy(dtype="datetime64[ns]")
        mem = g["nps_members"].to_numpy(dtype=float)
        new = g["nps_new"].to_numpy(dtype=float)
        lost = g["nps_lost"].to_numpy(dtype=float)
        pe = g["period_end"].to_numpy(dtype="datetime64[ns]")
        for k in np.flatnonzero(ents == e):
            t = dates[k]
            known = np.flatnonzero(pub < t)
            if not len(known):
                continue
            j = known[-1]
            if (t - pe[j]) > np.timedelta64(NPS_MAX_AGE_DAYS, "D"):
                continue
            m = month.iloc[j]

            def at(offset: int, _m=m, _known=set(known.tolist())) -> int | None:
                i = pos.get(_m - offset)
                return i if i is not None and i in _known else None

            row = [np.nan] * 4
            for c, off in ((0, 3), (1, 12)):
                i = at(off)
                if i is not None and mem[i] > 0:
                    row[c] = mem[j] / mem[i] - 1.0
            last3 = [i for i in (at(0), at(1), at(2)) if i is not None]
            if len(last3) == 3:
                avg = np.nanmean(mem[last3])
                if avg > 0:
                    row[2] = np.nansum(new[last3]) / avg
                    row[3] = np.nansum(lost[last3]) / avg
            res[k] = row
    out.loc[:, :] = res
    return out


def alt_features(rows: pd.DataFrame, alt: AltData | None, market: str, nps: bool = False) -> pd.DataFrame:
    """Every raw alternative-data feature of ``market`` for ``rows`` (columns = ``alt_feature_names``)."""
    names = alt_feature_names(market, nps)
    if alt is None:
        return pd.DataFrame(np.nan, index=rows.index, columns=names, dtype=float)
    if str(market).upper() == "KR":
        parts = [event_features(rows, alt, KR_EVENT_SPECS), employee_features(rows, alt)]
        if nps:
            parts.append(nps_features(rows, alt))
    else:
        parts = [event_features(rows, alt, US_EVENT_SPECS)]
    return pd.concat(parts, axis=1)[names]


# ================================================================================================ macro betas


def factor_changes(
    series: Mapping[str, pd.Series] | None, market: str, index: pd.DatetimeIndex, lag_days: int
) -> pd.DataFrame:
    """Daily factor changes on ``index`` (a market calendar) from levels taken as of ``date - lag_days``:
    fx = log change of USD/KRW (KR) or the dollar index (US), oil = log change of WTI, rate = change of the US 10Y
    yield (percentage points), credit = log change of HYG/LQD."""
    out = pd.DataFrame(np.nan, index=index, columns=list(MACRO_BETA_FEATURES), dtype=float)
    if not series:
        return out
    asof = index - pd.Timedelta(int(lag_days), "D")

    def level(key: str) -> pd.Series | None:
        s = series.get(key)
        if s is None or len(s) == 0:
            return None
        s = s.sort_index()
        pos = np.searchsorted(s.index.to_numpy(dtype="datetime64[ns]"), asof.to_numpy(dtype="datetime64[ns]"), "right")
        vals = np.where(pos > 0, s.to_numpy(dtype=float)[np.clip(pos - 1, 0, None)], np.nan)
        return pd.Series(vals, index=index)

    fx = level("usdkrw" if str(market).upper() == "KR" else "dxy")
    oil = level("wti")
    rate = level("us10y")
    hyg, lqd = level("hyg"), level("lqd")
    with np.errstate(divide="ignore", invalid="ignore"):
        if fx is not None:
            out["beta_fx"] = np.log(fx.where(fx > 0)).diff()
        if oil is not None:
            out["beta_oil"] = np.log(oil.where(oil > 0)).diff()
        if rate is not None:
            out["beta_rate"] = rate.diff()
        if hyg is not None and lqd is not None:
            out["beta_credit"] = np.log((hyg / lqd).where((hyg > 0) & (lqd > 0))).diff()
    return out.replace([np.inf, -np.inf], np.nan)


def macro_betas(close: pd.Series, bench_close: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """Rolling (252, >= 126) partial regression coefficients of the stock's daily log return on each factor,
    controlling for its benchmark: beta_f = (c_rf v_m - c_rm c_mf) / (v_f v_m - c_mf^2)."""
    idx = close.index
    r = np.log(close.where(close > 0)).diff()
    b = bench_close.reindex(idx.union(bench_close.index)).ffill(limit=3).reindex(idx)
    m = np.log(b.where(b > 0)).diff()
    f_all = factors.reindex(idx)
    out = pd.DataFrame(index=idx, columns=list(MACRO_BETA_FEATURES), dtype=float)
    w, mp = TRADING_DAYS, BETA_MIN_OBS
    for c in MACRO_BETA_FEATURES:
        f = f_all[c] if c in f_all else pd.Series(np.nan, index=idx)
        ok = r.notna() & m.notna() & f.notna()
        rr, mm, ff = r.where(ok), m.where(ok), f.where(ok)
        c_rf = rr.rolling(w, min_periods=mp).cov(ff)
        c_rm = rr.rolling(w, min_periods=mp).cov(mm)
        c_mf = mm.rolling(w, min_periods=mp).cov(ff)
        v_m = mm.rolling(w, min_periods=mp).var()
        v_f = ff.rolling(w, min_periods=mp).var()
        den = v_f * v_m - c_mf * c_mf
        out[c] = ((c_rf * v_m - c_rm * c_mf) / den.where(den > 1e-18)).replace([np.inf, -np.inf], np.nan)
    return out


__all__ = [
    "KR_EMP_FEATURES",
    "KR_EVENT_SPECS",
    "KR_FORWARD_ONLY_SPECS",
    "KR_NPS_FEATURES",
    "MACRO_BETA_FEATURES",
    "US_EVENT_SPECS",
    "EventSpec",
    "alt_feature_names",
    "alt_features",
    "employee_features",
    "event_features",
    "factor_changes",
    "macro_betas",
    "nps_features",
]
