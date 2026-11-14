"""PIT factor panel: PER/PBR (as E/P, B/P), traded-value trend, 3M/6M momentum and sector momentum at month-ends.

See docs/BACKTEST_MACRO.md section 1. Every signal on rebalance date t uses prices <= t and fundamentals with
``available_from <= t`` only; value uses E/P and B/P from the PIT market cap (never EPS / BPS). ``fwd_ret`` is the
return from the close of the first session after t through the next rebalance date, ending at the delisting
terminal price when the name stops trading while held. Nothing is dropped silently: every excluded (date, entity) is
labelled in ``price_status`` / ``fwd_status`` and counted by :func:`panel_coverage`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from quantcio.backtest.sectors import OTHER
from quantcio.config import Settings
from quantcio.data.cache import DiskCache
from quantcio.data.pit import PITFundamentals, PITUniverse
from quantcio.models import Market

log = logging.getLogger(__name__)

FACTORS = ["value_ep", "value_bp", "volume_trend", "mom_3m", "mom_6m", "sector_mom_3m"]
FACTOR_LABELS = {
    "value_ep": "E/P (PER 역수)",
    "value_bp": "B/P (PBR 역수)",
    "volume_trend": "거래대금 추세 log(ADV20/ADV120)",
    "mom_3m": "3개월 모멘텀",
    "mom_6m": "6개월 모멘텀",
    "sector_mom_3m": "섹터 3개월 모멘텀",
}
EXPECTED_SIGN = {"value_ep": 1, "value_bp": 1, "volume_trend": 0, "mom_3m": 1, "mom_6m": 1, "sector_mom_3m": 1}
RAW_COLUMN = {
    "value_ep": "ep",
    "value_bp": "bp",
    "volume_trend": "volume_trend",
    "mom_3m": "mom_3m",
    "mom_6m": "mom_6m",
    "sector_mom_3m": "sector_mom_3m",
}
# sector momentum is constant inside a sector: its "sector-neutral" rank is degenerate, so it is always ranked
# across the whole universe (documented in the report).
UNIVERSE_RANKED = {"sector_mom_3m"}
PRICE_MAX_AGE_DAYS = 7  # a close older than this at t (or at a momentum anchor) is treated as missing
MIN_SHARE_COVERAGE_US = 0.8  # mirrors data/pit/us.MIN_SHARE_COVERAGE (dates below it were ranked by dollar volume)

PANEL_COLUMNS = [
    "date",
    "entity",
    "ticker",
    "name",
    "sector",
    "sector_group",
    "sector_source",
    "market_cap",
    "per",
    "pbr",
    "ep",
    "bp",
    "ep_negative",
    "bp_negative",
    "volume_trend",
    "mom_3m",
    "mom_6m",
    "sector_mom_3m",
    "close",
    *[f"z_{f}" for f in FACTORS],
    *[f"zr_{f}" for f in FACTORS],
    "price_status",
    "fwd_ret",
    "fwd_status",
]


# ------------------------------------------------------------------------------------------------ ranking


def uniform_rank(values: pd.Series) -> pd.Series:
    """Average ranks mapped to (-1, 1): z = 2 * (rank - 0.5) / n - 1 (NaN stays NaN; -inf ranks last)."""
    ok = values.notna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    n = int(ok.sum())
    if n == 0:
        return out
    r = values[ok].rank(method="average")
    out[ok] = 2.0 * (r - 0.5) / n - 1.0
    return out


def rank_input(panel: pd.DataFrame, factor: str) -> pd.Series:
    """Ranking variable: raw value; E/P / B/P of loss makers / negative book (flag set, value NaN) rank last."""
    col = RAW_COLUMN[factor]
    x = panel[col].astype(float).copy()
    if factor == "value_ep":
        x[panel["ep_negative"].fillna(False).astype(bool)] = -np.inf
    elif factor == "value_bp":
        x[panel["bp_negative"].fillna(False).astype(bool)] = -np.inf
    return x


def neutral_groups(panel: pd.DataFrame, min_names: int) -> pd.Series:
    """Sector used for neutralisation: sectors with fewer than ``min_names`` members on a date merge into "기타"."""
    size = panel.groupby(["date", "sector"])["entity"].transform("size")
    return panel["sector"].where(size >= int(min_names), OTHER)


def add_zscores(panel: pd.DataFrame, sector_neutral: bool, min_names: int) -> pd.DataFrame:
    """z_<f>: rank within sector group (or the universe when ``sector_neutral`` is False); zr_<f>: universe rank."""
    panel = panel.copy()
    panel["sector_group"] = neutral_groups(panel, min_names)
    for f in FACTORS:
        x = rank_input(panel, f)
        zr = x.groupby(panel["date"]).transform(uniform_rank)
        panel[f"zr_{f}"] = zr
        if sector_neutral and f not in UNIVERSE_RANKED:
            panel[f"z_{f}"] = x.groupby([panel["date"], panel["sector_group"]]).transform(uniform_rank)
        else:
            panel[f"z_{f}"] = zr
    return panel


# ------------------------------------------------------------------------------------------------ prices


def market_calendar(universe: PITUniverse) -> pd.DatetimeIndex:
    """Sessions of the market: union of the benchmark and every entity frame index."""
    parts = [df.index.values for df in universe.benchmarks.values() if df is not None and len(df)]
    parts += [px.index.values for px in universe.prices.values() if px is not None and len(px)]
    if not parts:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(np.unique(np.concatenate(parts)))


def _terminal_marks(universe: PITUniverse) -> dict[str, tuple[pd.Timestamp, float]]:
    """entity -> (terminal date, price on the frame basis) for US delisted names (``data.pit.us.terminal_price``)."""
    m = universe.meta
    if m is None or m.empty or "terminal_date" not in m:
        return {}
    out = {}
    for _, r in m.iterrows():
        if pd.isna(r.get("terminal_date")) or pd.isna(r.get("terminal_price")):
            continue
        if str(r.get("terminal_basis") or "frame") != "frame":
            continue
        out[str(r["entity"])] = (pd.Timestamp(r["terminal_date"]), float(r["terminal_price"]))
    return out


def price_signals(
    universe: PITUniverse,
    members: pd.DataFrame,
    settings: Settings,
    next_date: dict[pd.Timestamp, pd.Timestamp],
    data_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per (date, entity): close, mom_3m, mom_6m, volume_trend, fwd_ret and status labels (vectorised per entity)."""
    cal = market_calendar(universe)
    w3, w6 = (int(x) for x in settings.backtest.momentum_windows[:2])
    v_short, v_long = (int(x) for x in settings.backtest.volume_windows[:2])
    delisted = set()
    if universe.meta is not None and len(universe.meta) and "delisted" in universe.meta:
        delisted = set(universe.meta.loc[universe.meta["delisted"].astype(bool), "entity"].astype(str))
    terminal = _terminal_marks(universe)
    halts = universe.market is Market.KR  # KRX keeps halted sessions as zero-volume rows
    data_end = data_end if data_end is not None else (cal[-1] if len(cal) else None)
    max_age = np.timedelta64(PRICE_MAX_AGE_DAYS, "D")
    frames = []
    for e, g in members.groupby("entity", sort=False):
        dates = pd.DatetimeIndex(g["date"])
        res = pd.DataFrame(index=g.index)
        res["close"] = np.nan
        for c in ("mom_3m", "mom_6m", "volume_trend", "fwd_ret"):
            res[c] = np.nan
        res["price_status"] = "no_price"
        res["fwd_status"] = "no_price"
        px = universe.prices.get(e)
        if px is None or px.empty:
            frames.append(res)
            continue
        idx = px.index.values
        close = px["close"].to_numpy(dtype=float)
        dv = close * px["volume"].to_numpy(dtype=float)
        dv = np.where(np.isfinite(dv), dv, 0.0)
        cs = np.concatenate([[0.0], np.cumsum(dv)])
        d = dates.values
        i = np.searchsorted(idx, d, side="right") - 1
        has = i >= 0
        fresh = has.copy()
        fresh[has] = (d[has] - idx[i[has]]) <= max_age
        res.loc[has & ~fresh, "price_status"] = "stale"
        res.loc[~has, "price_status"] = "no_bar_yet"
        ok = fresh
        res.loc[ok, "price_status"] = "ok"
        ci = np.where(ok, close[np.clip(i, 0, len(close) - 1)], np.nan)
        res["close"] = ci
        # momentum anchors on the market calendar
        pos = np.searchsorted(cal.values, d, side="right") - 1
        for w, col in ((w3, "mom_3m"), (w6, "mom_6m")):
            lp = pos - w
            valid = ok & (lp >= 0)
            anchor = cal.values[np.clip(lp, 0, len(cal) - 1)]
            j = np.searchsorted(idx, anchor, side="right") - 1
            valid &= j >= 0
            valid[valid] &= (anchor[valid] - idx[j[valid]]) <= max_age
            m = np.full(len(d), np.nan)
            m[valid] = ci[valid] / close[j[valid]] - 1.0
            res[col] = m
        # traded-value trend
        valid = ok & (i + 1 >= v_long)
        ii = np.clip(i, 0, len(close) - 1) + 1
        a_s = (cs[ii] - cs[np.clip(ii - v_short, 0, None)]) / v_short
        a_l = (cs[ii] - cs[np.clip(ii - v_long, 0, None)]) / v_long
        with np.errstate(divide="ignore", invalid="ignore"):
            vt = np.log(a_s / a_l)
        vt[~valid | ~np.isfinite(vt)] = np.nan
        res["volume_trend"] = vt
        # forward return: close of the first TRADED session after t -> close at the next rebalance date.
        # KR frames keep halted sessions as zero-volume rows with the carried close: a position cannot be bought on
        # them, and a position halted at the period end is valued at its next traded close (resumption or 정리매매),
        # so a halt that ends in a delisting is charged to the month in which the name was held.
        if halts:
            traded = px["volume"].to_numpy(dtype=float) > 0
        else:
            traded = np.ones(len(idx), dtype=bool)
        tr_pos = np.flatnonzero(traded)
        fwd = np.full(len(d), np.nan)
        status = np.array(["last_period"] * len(d), dtype=object)
        is_delisted = e in delisted
        term = terminal.get(e)
        for n, t in enumerate(dates):
            t1 = next_date.get(t)
            if t1 is None:
                continue
            if not ok[n]:
                status[n] = "no_price_t"
                continue
            t1_ = np.datetime64(t1)
            k0 = int(np.searchsorted(idx, np.datetime64(t), side="right"))
            q = int(np.searchsorted(tr_pos, k0))
            k = int(tr_pos[q]) if q < len(tr_pos) else len(idx)
            if k >= len(idx) or idx[k] > t1_:
                if k0 >= len(idx) and is_delisted:
                    status[n] = "delisted_before_entry"
                elif k0 < len(idx) and idx[k0] <= t1_:
                    status[n] = "halted_no_entry"
                else:
                    status[n] = "no_entry"
                continue
            entry = close[k]
            m_ = int(np.searchsorted(idx, t1_, side="right")) - 1
            exit_px, st = close[m_], "ok"
            gap = (t1_ - idx[m_]) > max_age
            ended = m_ == len(idx) - 1 and idx[m_] < t1_
            if not traded[m_] and m_ > k:
                # halted at the period end: next traded close after t1, else the last traded close if delisted
                q2 = int(np.searchsorted(tr_pos, m_ + 1))
                if q2 < len(tr_pos):
                    exit_px, st = close[tr_pos[q2]], "halted_exit_next_trade"
                elif is_delisted:
                    exit_px, st = close[tr_pos[max(int(np.searchsorted(tr_pos, m_)) - 1, 0)]], "delisted"
                else:
                    status[n] = "halted_unresolved"
                    continue
            elif ended and is_delisted:
                # stopped trading while held: last traded close (KR incl. 정리매매) or the terminal mark (US)
                st = "delisted"
                if term is not None and np.datetime64(term[0]) > idx[m_] and term[0] <= t1:
                    exit_px, st = term[1], "delisted_terminal"
            elif ended and data_end is not None and t1 > data_end:
                status[n] = "incomplete"
                continue
            elif gap:
                status[n] = "price_gap"
                continue
            if not (np.isfinite(entry) and entry > 0 and np.isfinite(exit_px)):
                status[n] = "bad_price"
                continue
            fwd[n] = exit_px / entry - 1.0
            status[n] = st
        res["fwd_ret"] = fwd
        res["fwd_status"] = status
        frames.append(res)
    if not frames:
        return pd.DataFrame(
            columns=["close", "mom_3m", "mom_6m", "volume_trend", "fwd_ret", "price_status", "fwd_status"]
        )
    return pd.concat(frames).reindex(members.index)


# ------------------------------------------------------------------------------------------------ market caps


def us_market_caps(universe: PITUniverse, fundamentals: PITFundamentals, settings: Settings) -> pd.DataFrame:
    """Wide PIT market caps (dates x entities) for the US universe from ``membership.rank_value``.

    The US builder ranks by PIT market cap = XBRL shares (filed <= t) x split-adjusted close; members without a PIT
    share count get a dollar-volume imputation, and dates where < 80% of members have shares are ranked by dollar
    volume. Neither imputation is a market cap, so they are replaced by NaN here: the share-count rule is replayed on
    the same PIT fundamentals (available_from <= t, period end within max_staleness_days). Needs n >= 500 (every
    member) for the date-level rule to be exact; with a smaller n the rule is evaluated on the top-n only.
    """
    mem = universe.membership[["date", "entity", "rank_value"]].copy()
    p = fundamentals.panel
    stale_days = int(settings.pit.fundamentals.max_staleness_days)
    if p is None or p.empty or "shares_out" not in p:
        has = pd.Series(False, index=mem.index)
    else:
        sh = p.dropna(subset=["shares_out", "available_from"])[["entity", "available_from", "period_end"]]
        sh = sh.sort_values("available_from")
        left = mem.reset_index().sort_values("date")
        j = pd.merge_asof(
            left, sh, left_on="date", right_on="available_from", by="entity", direction="backward"
        ).set_index("index")
        j = j.reindex(mem.index)
        has = j["period_end"].notna() & ((j["date"] - j["period_end"]).dt.days <= stale_days)
    cover = has.groupby(mem["date"]).transform("mean")
    mem["mcap"] = mem["rank_value"].where(has & (cover >= MIN_SHARE_COVERAGE_US))
    wide = mem.pivot_table(index="date", columns="entity", values="mcap", aggfunc="first", dropna=False)
    wide = wide.reindex(columns=sorted(mem["entity"].unique()))
    wide.attrs["no_shares_member_months"] = int((~has).sum())
    wide.attrs["dollar_volume_ranked_dates"] = sorted(
        str(d.date()) for d, c in cover.groupby(mem["date"]).first().items() if c < MIN_SHARE_COVERAGE_US
    )
    return wide


def valuation_frame(fundamentals: PITFundamentals, market_caps: pd.DataFrame, max_staleness_days: int) -> pd.DataFrame:
    """(date, entity) -> market_cap, ep, bp, per, pbr, ni_period_end, eq_period_end.

    Each item is taken from the latest PIT row that carries it (available_from <= date, ties -> latest period end)
    and dropped when its period end is more than ``max_staleness_days`` before the date. A row with an unknown TTM
    (e.g. a quarter whose year-ago YTD is missing) therefore does not hide the previous usable TTM. E/P = TTM
    parent NI / market cap, B/P = parent equity / market cap; PER / PBR only when the numerator is > 0.
    """
    long = market_caps.copy()
    long.index = pd.DatetimeIndex(long.index).rename("date")
    long.columns = pd.Index([str(c) for c in long.columns], name="entity")
    long = long.stack(future_stack=True).rename("market_cap").reset_index().dropna(subset=["market_cap"])
    cols = ["date", "entity", "market_cap", "ep", "bp", "per", "pbr", "ni_period_end", "eq_period_end"]
    if long.empty:
        return pd.DataFrame(columns=cols)
    long = long.sort_values("date").reset_index(drop=True)
    panel = fundamentals.panel
    for item, key in (("ni_parent_ttm", "ni"), ("equity_parent", "eq")):
        long[key] = np.nan
        long[f"{key}_period_end"] = pd.NaT
        if panel is None or panel.empty or item not in panel:
            continue
        p = panel.dropna(subset=[item, "available_from"])
        p = p[p["entity"].isin(set(long["entity"]))][["entity", "available_from", "period_end", item]]
        if p.empty:
            continue
        p = p.sort_values(["available_from", "period_end"], kind="stable").rename(
            columns={"available_from": "af", "period_end": "pe"}
        )
        j = pd.merge_asof(
            long[["date", "entity"]].reset_index(), p, left_on="date", right_on="af", by="entity", direction="backward"
        ).set_index("index")
        ok = j["pe"].notna() & ((j["date"] - j["pe"]).dt.days <= int(max_staleness_days))
        long[key] = j[item].where(ok).astype(float)
        long[f"{key}_period_end"] = j["pe"].where(ok)
    mcap = long["market_cap"].astype(float)
    pos = mcap > 0
    long["ep"] = (long["ni"] / mcap).where(pos)
    long["bp"] = (long["eq"] / mcap).where(pos)
    long["per"] = (mcap / long["ni"]).where(pos & (long["ni"] > 0))
    long["pbr"] = (mcap / long["eq"]).where(pos & (long["eq"] > 0))
    return long[cols]


# ------------------------------------------------------------------------------------------------ panel


def _members_for(universe: PITUniverse, dates: Any) -> pd.DataFrame:
    mem = universe.membership
    if dates is not None:
        ds = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
        mem = mem[mem["date"].isin(ds)]
    return mem[["date", "entity", "ticker", "name"]].drop_duplicates(["date", "entity"]).reset_index(drop=True)


def _sectors_for(
    universe: PITUniverse, members: pd.DataFrame, settings: Settings, cache: DiskCache | None, sectors: Any
) -> tuple[pd.DataFrame, list[str]]:
    """date, entity, sector, sector_source for every member (explicit table / dict, meta column, or sectors.py)."""
    if isinstance(sectors, pd.DataFrame):
        s = sectors.copy()
        if "date" in s:
            out = members[["date", "entity"]].merge(s, on=["date", "entity"], how="left")
        else:
            out = members[["date", "entity"]].merge(s, on="entity", how="left")
        if "sector_source" not in out:
            out["sector_source"] = np.where(out["sector"].notna(), "given", "none")
        out["sector"] = out["sector"].fillna(OTHER)
        return out, []
    if isinstance(sectors, dict):
        out = members[["date", "entity"]].copy()
        out["sector"] = out["entity"].map(sectors)
        out["sector_source"] = np.where(out["sector"].notna(), "given", "none")
        out["sector"] = out["sector"].fillna(OTHER)
        return out, []
    meta = universe.meta
    if meta is not None and len(meta) and "sector" in meta:
        m = meta.drop_duplicates("entity").set_index("entity")["sector"]
        out = members[["date", "entity"]].copy()
        out["sector"] = out["entity"].map(m)
        out["sector_source"] = np.where(out["sector"].notna(), "meta", "none")
        out["sector"] = out["sector"].fillna(OTHER)
        return out, []
    from quantcio.backtest.sectors import sector_panel

    cache = cache or DiskCache(settings.cache_path)
    return sector_panel(universe.market, members[["date", "entity"]], settings, cache, meta)


def compute_factor_panel(
    universe: PITUniverse,
    fundamentals: PITFundamentals,
    settings: Settings,
    dates: Any = None,
    *,
    market_caps: pd.DataFrame | None = None,
    sectors: pd.DataFrame | dict[str, str] | None = None,
    cache: DiskCache | None = None,
    members: pd.DataFrame | None = None,
    next_dates: dict[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Month-end factor panel (PANEL_COLUMNS) for the PIT universe members on ``dates`` (None = every rebalance).

    ``market_caps`` (wide, index dates x entity columns, local currency) defaults to the PIT market cap of the market
    (KR: ``pit_market_cap_kr`` incl. preferred lines; US: :func:`us_market_caps`). ``sectors`` (DataFrame with
    entity[, date], sector or dict entity -> sector) defaults to ``universe.meta['sector']`` when present, else
    ``sectors.sector_panel``. ``members`` (date, entity, ticker, name) overrides the membership rows (serving
    snapshots between rebalances); ``next_dates`` overrides the holding-period end of each date (default: the next
    date of the panel; the last date gets NaN ``fwd_ret``).
    """
    mem = members.copy() if members is not None else _members_for(universe, dates)
    for c in ("ticker", "name"):
        if c not in mem:
            mem[c] = None
    mem["date"] = pd.to_datetime(mem["date"]).dt.normalize()
    mem = mem.sort_values(["date", "entity"]).reset_index(drop=True)
    if mem.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    uniq = pd.DatetimeIndex(sorted(mem["date"].unique()))
    if next_dates is None:
        next_dates = dict(zip(uniq[:-1], uniq[1:], strict=False))
    notes: list[str] = []

    # prices
    ps = price_signals(universe, mem, settings, next_dates)
    panel = pd.concat([mem, ps], axis=1)

    # value (PIT market cap + PIT fundamentals)
    stale = int(settings.pit.fundamentals.max_staleness_days)
    if market_caps is None:
        ents = sorted(mem["entity"].unique())
        if universe.market is Market.KR:
            from quantcio.data.pit.kr import pit_market_cap_kr

            market_caps = pit_market_cap_kr(settings, cache or DiskCache(settings.cache_path), ents, list(uniq))
        else:
            market_caps = us_market_caps(universe, fundamentals, settings)
    mc = market_caps.reindex(index=uniq)
    mc = mc.loc[:, [c for c in mc.columns if c in set(mem["entity"])]]
    val = valuation_frame(fundamentals, mc, stale)
    panel = panel.merge(val, on=["date", "entity"], how="left")
    if "market_cap" not in panel:
        panel["market_cap"] = np.nan
    # the market cap exists but no fundamentals row was usable: keep the cap, value stays NaN
    cap_long = mc.stack(future_stack=True).rename("mc_all").reset_index()
    cap_long.columns = ["date", "entity", "mc_all"]
    panel = panel.merge(cap_long, on=["date", "entity"], how="left")
    panel["market_cap"] = panel["market_cap"].fillna(panel["mc_all"])
    panel = panel.drop(columns=["mc_all"])
    panel["ep_negative"] = panel["ep"].notna() & (panel["ep"] <= 0)
    panel["bp_negative"] = panel["bp"].notna() & (panel["bp"] <= 0)
    panel.loc[panel["ep_negative"], "ep"] = np.nan
    panel.loc[panel["bp_negative"], "bp"] = np.nan

    # sectors
    sec, snotes = _sectors_for(universe, mem, settings, cache, sectors)
    notes += snotes
    panel = panel.merge(sec[["date", "entity", "sector", "sector_source"]], on=["date", "entity"], how="left")
    panel["sector"] = panel["sector"].fillna(OTHER)
    panel["sector_source"] = panel["sector_source"].fillna("none")
    real = panel["sector_source"] != "none"
    sm = panel[real].groupby(["date", "sector"])["mom_3m"].transform("mean")
    panel["sector_mom_3m"] = np.nan
    panel.loc[real, "sector_mom_3m"] = sm

    panel = add_zscores(panel, bool(settings.backtest.sector_neutral), int(settings.backtest.min_names_per_sector))
    panel = panel[PANEL_COLUMNS].sort_values(["date", "entity"]).reset_index(drop=True)
    panel.attrs["notes"] = notes
    if hasattr(market_caps, "attrs"):
        panel.attrs["market_cap_info"] = dict(market_caps.attrs)
    return panel


def panel_coverage(panel: pd.DataFrame) -> dict[str, Any]:
    """Counts of every exclusion / missing value in a factor panel (overall and by year)."""
    if panel.empty:
        return {"member_months": 0}
    out: dict[str, Any] = {
        "member_months": len(panel),
        "dates": int(panel["date"].nunique()),
        "entities": int(panel["entity"].nunique()),
        "price_status": {str(k): int(v) for k, v in panel["price_status"].value_counts().items()},
        "fwd_status": {str(k): int(v) for k, v in panel["fwd_status"].value_counts().items()},
        "market_cap_missing": int(panel["market_cap"].isna().sum()),
        "ep_negative": int(panel["ep_negative"].sum()),
        "bp_negative": int(panel["bp_negative"].sum()),
        "value_missing": int((panel["ep"].isna() & ~panel["ep_negative"]).sum()),
        "factor_available": {f: int(panel[f"z_{f}"].notna().sum()) for f in FACTORS},
        "sector_unclassified": int((panel["sector_source"] == "none").sum()),
        "sector_merged_into_other": int(((panel["sector_group"] == OTHER) & (panel["sector"] != OTHER)).sum()),
    }
    by_year: dict[str, dict[str, int]] = {}
    for y, g in panel.groupby(panel["date"].dt.year):
        by_year[str(int(y))] = {
            "member_months": len(g),
            "fwd_ret_ok": int(g["fwd_ret"].notna().sum()),
            "delisted_exits": int(g["fwd_status"].isin(["delisted", "delisted_terminal"]).sum()),
            "halted_exits": int((g["fwd_status"] == "halted_exit_next_trade").sum()),
            "price_missing": int((g["price_status"] != "ok").sum()),
            "value_ep_available": int(g["z_value_ep"].notna().sum()),
            "value_bp_available": int(g["z_value_bp"].notna().sum()),
            "sector_unclassified": int((g["sector_source"] == "none").sum()),
            "sector_merged_into_other": int(((g["sector_group"] == OTHER) & (g["sector"] != OTHER)).sum()),
        }
    out["by_year"] = by_year
    return out
