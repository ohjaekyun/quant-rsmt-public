"""Features from the extra KRX data (docs/MODELING.md 66, pre-declared 2026-10-17). Research only.

66.1 institution-type net buying (KRX [12009] cache), 66.2 KIND market-warning designations, 66.3 single-stock futures.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_data as v13  # noqa: E402
import v16_data as v16  # noqa: E402
import v26_krx_data as kx  # noqa: E402

FLOW_TYPES = {
    "invtrust": (["v_투신"], "투신(공모 펀드)"),
    "private": (["v_사모"], "사모 펀드"),
    "fininv": (["v_금융투자"], "금융투자(증권사 자기매매)"),
    "insur": (["v_보험"], "보험"),
    "bank": (["v_은행", "v_기타금융"], "은행·기타금융"),
    "corp": (["v_기타법인"], "기타법인(일반 회사)"),
}
SSF_AGE = 10
NEVER = float(np.log(3650.0))
DOCS: dict[str, tuple[str, str]] = {
    **{
        f"kr_inv_{k}_{w}": (f"{ko} 순매수 {w}일 합 ÷ 시가총액(KRX)", f"{k} net buying, {w} days / market cap")
        for k, (_, ko) in FLOW_TYPES.items()
        for w in (20, 60)
    },
    "kr_active_net_20": ("투신 + 사모 순매수 20일 합 ÷ 시가총액", "mutual + private fund net buying, 20 days"),
    "mw_caution_60": ("60일 투자주의 지정 수(KIND)", "investment-caution designations, 60 days"),
    "mw_caution_250": ("250일 투자주의 지정 수", "investment-caution designations, 250 days"),
    "mw_warn_250": ("250일 투자경고·위험 지정 수", "warning / risk designations, 250 days"),
    "mw_warn_active": ("그날 투자경고·위험 지정 중", "under a warning / risk designation"),
    "mw_since_warn": ("마지막 투자경고·위험 지정 후 경과일(log)", "log days since the last warning"),
    "ssf_basis": ("주식선물 근월물 베이시스(연율)", "single-stock futures basis, annualised"),
    "ssf_oi": ("주식선물 미결제약정 ÷ 시가총액", "single-stock futures open interest / market cap"),
    "ssf_oi_chg": ("주식선물 미결제약정 4주 로그 변화", "open interest 4-week log change"),
    "ssf_turn": ("주식선물 주간 거래대금 ÷ 시가총액", "futures traded value / market cap"),
}


def _norm(name: str) -> str:
    return re.sub(r"\(주\)|㈜|주식회사|\s", "", str(name or "")).upper()


def name_map(settings, cache) -> dict[str, str]:
    """Normalised KRX short name -> entity over every (entity, name) the KR universe ever had; names shared by two
    entities are dropped."""
    from quantcio.data.pit import load_pit_universe
    from quantcio.modeling.dataset import universe_params
    from quantcio.models import Market

    n, start = universe_params(Market.KR, settings)
    m = load_pit_universe(Market.KR, settings, cache, n, start).membership[["entity", "name"]].drop_duplicates()
    m["k"] = m["name"].map(_norm)
    g = m.groupby("k")["entity"].nunique()
    m = m[m["k"].isin(g[g == 1].index)]
    return dict(zip(m["k"], m["entity"].astype(str), strict=True))


# ================================================================================================ 66.1 flows


def build_flows(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, dict]:
    mcap = np.exp(frame["log_mcap"].astype(float)).to_numpy()
    keys = [(k, w) for k in FLOW_TYPES for w in (20, 60)]
    raw = np.full((len(frame), len(keys)), np.nan)
    pos = {ix: i for i, ix in enumerate(frame.index)}
    n = 0
    for ent, g in frame.groupby("entity", sort=False):
        p = kd.root(settings, "krx_flows") / f"{ent}.parquet"
        if not p.exists():
            continue
        q = pd.read_parquet(p).sort_values("date").set_index("date")
        dd = pd.DataFrame(index=q.index)
        for k, w in keys:
            cols = [c for c in FLOW_TYPES[k][0] if c in q]
            if not cols:
                continue
            v = q[cols].sum(axis=1, min_count=1)
            dd[f"{k}_{w}"] = v.rolling(w, min_periods=int(w * 0.75)).sum()
        dd = dd.reindex(columns=[f"{k}_{w}" for k, w in keys]).shift(1)  # flows up to the previous session
        n += 1
        rows = np.array([pos[i] for i in g.index])
        raw[rows] = v13._asof_rows(dd, g["date"].to_numpy(dtype="datetime64[ns]"), v13.FRESH_DAYS)
    out = pd.DataFrame(raw / mcap[:, None], index=frame.index, columns=[f"kr_inv_{k}_{w}" for k, w in keys])
    out["kr_active_net_20"] = out["kr_inv_invtrust_20"] + out["kr_inv_private_20"]
    return out, {"flow_codes": n}


# ================================================================================================ 66.2 warnings


def _d(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, errors="coerce")


def warnings_table(settings, names: dict[str, str]) -> pd.DataFrame:
    rows = []
    for p in sorted((kx.root(settings) / "kind").glob("*.json")):
        kind = p.stem.split("_")[0]
        for c in json.loads(p.read_text()):
            ent = names.get(_norm(c[1]))
            if ent is None:
                continue
            if kind == "caution":  # no, name, reason, announced, designated
                rows.append({"entity": ent, "kind": kind, "ann": _d(c[3]), "start": _d(c[4]), "end": pd.NaT})
            else:  # no, name, announced, designated, released
                rows.append({"entity": ent, "kind": "warn", "ann": _d(c[2]), "start": _d(c[3]), "end": _d(c[4])})
    t = pd.DataFrame(rows)
    return t.dropna(subset=["ann"]).drop_duplicates() if len(t) else t


def build_warnings(frame: pd.DataFrame, settings, names: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    cols = ["mw_caution_60", "mw_caution_250", "mw_warn_250", "mw_warn_active", "mw_since_warn"]
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    t = warnings_table(settings, names)
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    first = t["ann"].min() if len(t) else pd.Timestamp.max
    by = {e: g for e, g in t.groupby("entity")} if len(t) else {}
    for e in np.unique(ents):
        ii = np.where(ents == e)[0]
        d = dates[ii]
        g = by.get(e)
        if g is None:
            out.iloc[ii, :] = [0.0, 0.0, 0.0, 0.0, NEVER]
            continue
        ca = g.loc[g["kind"] == "caution", "ann"].to_numpy(dtype="datetime64[ns]")
        wa = g[g["kind"] == "warn"]
        wann = wa["ann"].to_numpy(dtype="datetime64[ns]")
        out.iloc[ii, 0] = v16._windowed(ca, np.ones(len(ca)), d, 60)[0]
        out.iloc[ii, 1] = v16._windowed(ca, np.ones(len(ca)), d, 250)[0]
        out.iloc[ii, 2] = v16._windowed(wann, np.ones(len(wann)), d, 250)[0]
        act = np.zeros(len(d))
        for s0, e0, a0 in zip(wa["start"], wa["end"], wa["ann"], strict=True):
            lo = np.datetime64(max(s0, a0 + pd.Timedelta(1, "D")) if pd.notna(s0) else a0 + pd.Timedelta(1, "D"))
            hi = np.datetime64(e0) if pd.notna(e0) else np.datetime64("2100-01-01")
            act = np.maximum(act, ((d >= lo) & (d < hi)).astype(float))
        out.iloc[ii, 3] = act
        srt = np.sort(wann)
        if len(srt):
            k = np.searchsorted(srt, d, "left") - 1  # announcements strictly before the row date
            since = np.where(k >= 0, (d - srt[np.clip(k, 0, None)]) / np.timedelta64(1, "D"), np.nan)
            out.iloc[ii, 4] = np.where(np.isnan(since), NEVER, np.log1p(np.minimum(since, 3650)))
        else:
            out.iloc[ii, 4] = NEVER
    out.loc[frame["date"].to_numpy() < np.datetime64(first + pd.Timedelta(250, "D")), :] = np.nan  # no history yet
    n_c = int((t["kind"] == "caution").sum()) if len(t) else 0
    return out, {"caution_rows": n_c, "warn_rows": len(t) - n_c, "entities_warned": len(by)}


# ================================================================================================ 66.3 futures


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").replace({"-": np.nan, "": np.nan}), errors="coerce")


def _second_thursday(ym: str) -> pd.Timestamp:
    first = pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01")
    return first + pd.Timedelta(days=(3 - first.weekday()) % 7 + 7)


def ssf_table(settings, names: dict[str, str]) -> pd.DataFrame:
    recs = []
    for p in sorted((kx.root(settings) / "ssf").glob("*.parquet")):
        day = pd.Timestamp(p.stem)
        df = pd.read_parquet(p)
        if df.empty or "ISU_NM" not in df:
            continue
        m = df["ISU_NM"].str.extract(r"^(.*?)\s+F\s+(\d{6})")
        df = df.assign(und=m[0].str.strip(), ym=m[1]).dropna(subset=["und", "ym"])
        if df.empty:
            continue
        df["close"] = _num(df["TDD_CLSPRC"]).fillna(_num(df["SETL_PRC"]))
        df["spot"], df["oi"], df["val"] = _num(df["SPOT_PRC"]), _num(df["ACC_OPNINT_QTY"]), _num(df["ACC_TRDVAL"])
        df["mat"] = df["ym"].map(_second_thursday)
        df["days"] = (df["mat"] - day).dt.days
        for und, g in df.groupby("und"):
            ent = names.get(_norm(und))
            if ent is None:
                continue
            live = g[(g["oi"] > 0) & (g["days"] >= 14) & (g["close"] > 0) & (g["spot"] > 0)].sort_values("days")
            basis = np.nan
            if len(live):
                f = live.iloc[0]
                basis = (f["close"] / f["spot"] - 1.0) * 365.0 / f["days"]
            spot = g["spot"].dropna().iloc[0] if g["spot"].notna().any() else np.nan
            recs.append(
                {
                    "entity": ent,
                    "date": day,
                    "basis": basis,
                    "oi_val": g["oi"].sum() * kx.SSF_MULT * spot,
                    "oi": g["oi"].sum(),
                    "val": g["val"].sum(),
                }
            )
    t = pd.DataFrame(recs)
    if t.empty:
        return t
    t = t.sort_values(["entity", "date"])
    t["oi_chg"] = t.groupby("entity")["oi"].transform(lambda s: np.log1p(s) - np.log1p(s.shift(4)))
    return t


def build_ssf(frame: pd.DataFrame, settings, names: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    t = ssf_table(settings, names)
    cols = ["ssf_basis", "ssf_oi", "ssf_oi_chg", "ssf_turn"]
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    if t.empty:
        return out, {"ssf_entities": 0}
    mcap = np.exp(frame["log_mcap"].astype(float)).to_numpy()
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    for e, g in t.groupby("entity"):
        ii = np.where(ents == e)[0]
        if not len(ii):
            continue
        s = g.set_index(pd.DatetimeIndex(g["date"]) + pd.Timedelta(1, "D"))[["basis", "oi_val", "oi_chg", "val"]]
        v = v13._asof_rows(s, dates[ii], SSF_AGE)
        out.iloc[ii, 0] = v[:, 0]
        out.iloc[ii, 1] = v[:, 1] / mcap[ii]
        out.iloc[ii, 2] = v[:, 2]
        out.iloc[ii, 3] = v[:, 3] / mcap[ii]
    return out, {"ssf_entities": int(t["entity"].nunique()), "ssf_weeks": int(t["date"].nunique())}


def block(frame: pd.DataFrame, settings, cache) -> tuple[pd.DataFrame, dict]:
    names = name_map(settings, cache)
    a, ia = build_flows(frame, settings)
    b, ib = build_warnings(frame, settings, names)
    c, ic = build_ssf(frame, settings, names)
    return pd.concat([a, b, c], axis=1), {**ia, **ib, **ic, "names": len(names)}
