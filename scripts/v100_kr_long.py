"""KR from 2002: longer training history and a fresh 2005-2015 out-of-sample check (docs/MODELING.md 100, pre-declared
2026-11-08). Research only: src/quantcio and the served models are not changed.

    .venv/bin/python scripts/v100_kr_long.py build --h 1     # 2002-start dataset (20 sessions), feature rule
    .venv/bin/python scripts/v100_kr_long.py build --h 12    # 2002-start dataset (240 sessions)
    .venv/bin/python scripts/v100_kr_long.py study --arm A   # A/B (h=1: 2002 / 2016 start), C/D (h=12)
    .venv/bin/python scripts/v100_kr_long.py analyse         # re-read the four cached runs (no fit), apply the rules

Arms: A = 2002 start, 20 sessions; B = A's data cut to 2016-01 onward; C = 2002 start, 240 sessions; D = C cut to 2016.
Features (100): month-end raw non-missing share >= 90% in 2003-2014 and in 2016+, the same list for every arm; the
x_dart_* / x_emp_* / x_nps_* families are excluded by name; sx_X is judged on raw X. Candidates rf_bo, rf, mom_12_1.
Books as in 98 / 99 (50 bp, 100 bp sensitivity), monthly returns from A's 20-session labels.
Writes reports/modeling/KR_v100_long.{json,md}; state in reports/modeling/KR_v100_state.json.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v99_horizons as v99  # noqa: E402  (pops KRX_ID / KRX_PW before pykrx can be imported)

from quantcio.config import load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.cli import guard_xsh  # noqa: E402
from quantcio.modeling.dataset import build_dataset  # noqa: E402
from quantcio.models import Market  # noqa: E402

V100_VERSION = 1
START = "2002-01-01"
CUT = "2016-01-01"
SESS = {1: 20, 12: 240}
ARMS = {"A": (1, None), "B": (1, CUT), "C": (12, None), "D": (12, CUT)}
EXCLUDE_PREFIX = ("x_dart_", "x_emp_", "x_nps_")
MIN_COVER = 0.90
OLD = ("2003-01-01", "2014-12-31")
FRESH = ("2005-01-01", "2015-12-31")
CRISIS = ("2007-11-01", "2009-02-28")
# 100 supplement 2: the 99 datasets' prices start in 2013, so x_seas (1-5 years back) differs before this date
SEAS_FROM = "2018-07-01"
# 100 supplement 3: the frozen-window digest record of the broken first 20-session build (2026-11-09 01:05)
BROKEN_CUT = "2026-10-31"
BROKEN_FEATURES = "d5bb1db1993bd9b0"
BROKEN_LABELS = "5692013b44d7239b"
NAMES = v99.NAMES
COSTS = v99.COSTS
K = v99.K
OUT = v99.OUT
STATE = OUT / "KR_v100_state.json"
say = v99.say


def settings(h: int):
    # 100 supplement 2: the marcap artefacts (market cap, fundamentals entity list) are built from pit.kr.start_year's
    # files; start them in 2002 too, so the listing-spell numbering matches the 2002 universe.
    over = {"backtest": {"kr_start": START}, "pit": {"kr": {"start_year": int(START[:4])}}}
    if SESS[h] != 20:
        over["horizon_days"] = SESS[h]
    return load_settings(**over)


def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def raw_of(col: str, frame: pd.DataFrame) -> str:
    for p in ("sx_", "x_"):
        if col.startswith(p) and col[len(p) :] in frame.columns:
            return col[len(p) :]
    return col


def coverage(frame: pd.DataFrame, cols: list[str]) -> dict:
    me = frame[frame["is_me"].astype(bool)]
    old = me[(me["date"] >= OLD[0]) & (me["date"] <= OLD[1])]
    new = me[me["date"] >= CUT]
    return {c: (float(old[raw_of(c, frame)].notna().mean()), float(new[raw_of(c, frame)].notna().mean())) for c in cols}


def broken_first_record(ds, dig: dict, h: int) -> bool:
    """100 supplement 3: the digest identity has no `pit` settings, so the broken first 20-session build (01:05, market
    cap missing before 2013) was recorded as "first". Accept a mismatch only when it is that record's features: the
    one differing part is BROKEN_CUT features, the record's features digest is BROKEN_FEATURES, and the labels digest
    of this dataset equals the record's (BROKEN_LABELS)."""
    from quantcio.data import vintage as vt
    from quantcio.modeling.dataset import LABEL_DIGEST_COLS

    if h != 1 or dig.get("status") != "mismatch" or dig.get("mismatch") != [f"{BROKEN_CUT}:features"]:
        return False
    s = settings(h)
    book = json.loads((Path(s.cache_path) / "vintage" / vt.DIGESTS).read_text(encoding="utf-8"))
    recs = [v[BROKEN_CUT] for v in book.values() if v.get(BROKEN_CUT, {}).get("features") == BROKEN_FEATURES]
    f = ds.frame
    cut = np.datetime64(pd.Timestamp(BROKEN_CUT), "ns")
    lab = f["labelled"].to_numpy(dtype=bool)
    te = pd.to_datetime(f["t_end"]).to_numpy(dtype="datetime64[ns]")
    labels, _ = vt.frame_digest(f, LABEL_DIGEST_COLS, lab & ~np.isnat(te) & (te <= cut))
    ok = len(recs) == 1 and recs[0].get("labels") == labels == BROKEN_LABELS
    say(f"digest mismatch vs the broken first record: records {len(recs)}, labels {labels} -> accepted {ok}")
    return ok


def build(h: int) -> None:
    s = settings(h)
    cache = DiskCache(s.cache_path)
    ds = build_dataset(Market.KR, s, cache, use_cache=True, progress=say)
    dig = (ds.coverage.get("vintage") or {}).get("digest") or {}
    f = ds.frame
    me = f[f["is_me"].astype(bool)]
    by_year = me.groupby(me["date"].dt.year)["labelled"].agg(["size", "mean"])
    say(f"h={h}m: dataset {ds.snapshot} as_of {ds.as_of.date()} rows {len(f):,}; digest {dig.get('status')}")
    say(
        "labelled month-end share by year: "
        + " ".join(f"{y}:{r['mean']:.2f}/{int(r['size'])}" for y, r in by_year.iterrows())
    )
    if dig.get("status") not in ("match", "first") and not broken_first_record(ds, dig, h):
        raise SystemExit(f"frozen-window digest: {dig}")
    old_lab = me[(me["date"] >= OLD[0]) & (me["date"] <= OLD[1])]["labelled"].mean()
    if not old_lab >= 0.80:
        raise SystemExit(f"h={h}: only {old_lab:.2f} of the 2003-2014 month-end rows are labelled (< 0.80)")
    mc_miss = me.groupby(me["date"].dt.year)["mcap"].apply(lambda v: float(v.isna().mean()))
    say("month-end market cap missing share by year: " + " ".join(f"{y}:{v:.4f}" for y, v in mc_miss.items()))
    bad = {int(y): v for y, v in mc_miss.items() if y >= 2003 and v > 0.02}
    if bad:
        raise SystemExit(f"h={h}: month-end market cap missing > 2% in {bad}")
    spells = sorted(e for e in f["entity"].astype(str).unique() if "#" in e)
    say(f"relisted-spell entities ({len(spells)}): {' '.join(spells)}")
    st = state()
    ref_snap = (v99.state().get(str(h)) or {}).get("snapshot")
    ref = cache.get("modeling_dataset", ref_snap, None) if ref_snap else None
    if ref is None or list(ref.xs_cols) != list(ds.xs_cols):
        raise SystemExit(f"h={h}: the 99 dataset ({ref_snap}) is missing or has different inputs")

    def by_code(fr: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        x = fr[["date", "entity", *cols]].copy()
        x["code"] = x["entity"].astype(str).str.split("#").str[0]
        return x.drop(columns="entity")

    k = ["date", "code"]
    a = by_code(ref.frame, list(ref.xs_cols)).merge(
        by_code(f.loc[f["date"] >= CUT], list(ds.xs_cols)), on=k, suffixes=("", "_l")
    )
    if a.duplicated(k).any():
        raise SystemExit(f"h={h}: (date, code) is not unique after dropping the spell suffix")

    def eq(b: pd.DataFrame, cols: list[str]) -> float:
        x, y = b[cols].to_numpy(float), b[[f"{c}_l" for c in cols]].to_numpy(float)
        return float(np.isclose(x, y, rtol=1e-6, atol=1e-9, equal_nan=True).mean())

    seas = "x_seas"
    # 100 supplement 4: the name-excluded families are never model inputs (alt data has no relisted-spell rows)
    rest = [c for c in ref.xs_cols if c != seas and not c.startswith(EXCLUDE_PREFIX)]
    fam = [c for c in ref.xs_cols if c.startswith(EXCLUDE_PREFIX)]
    same = eq(a, rest)
    per = {c: eq(a, [c]) for c in ref.xs_cols}
    late = a["date"] >= SEAS_FROM
    seas_same = eq(a[late], [seas]) if seas in ref.xs_cols else 1.0
    fam_same = eq(a, fam) if fam else 1.0
    n_ref = int((ref.frame["date"] >= CUT).sum())
    say(
        f"h={h}m: 2016+ features vs the 99 dataset (by code): {len(a):,} of {n_ref:,} rows shared; "
        f"used candidates (no {seas}, no excluded families) {100 * same:.4f}% equal; "
        f"{seas} from {SEAS_FROM} {100 * seas_same:.4f}%; excluded families {100 * fam_same:.4f}% (reported)"
    )
    low = sorted(per.items(), key=lambda t: t[1])[:8]
    say("lowest per-feature equality: " + " ".join(f"{c}:{v:.4f}" for c, v in low))
    if seas in ref.xs_cols:
        yr = {int(y): eq(a[a["date"].dt.year == y], [seas]) for y in sorted(a["date"].dt.year.unique())}
        txt = " ".join(f"{y}:{v:.3f}" for y, v in yr.items())
        say(f"{seas} equality by year (99 prices start 2013, reported): {txt}")
    rg = [c for c in ds.regime_cols if c in ref.dates_frame.columns and c in ds.dates_frame.columns]
    j = ref.dates_frame[rg].join(ds.dates_frame[rg], rsuffix="_l", how="inner")
    rsame = float(np.isclose(j[rg].to_numpy(float), j[[f"{c}_l" for c in rg]].to_numpy(float), equal_nan=True).mean())
    say(f"h={h}m: regime columns equal on {len(j)} shared dates: {100 * rsame:.2f}% (reported, not a stop rule)")
    if len(a) < 0.95 * n_ref or same < 0.999 or seas_same < 0.999:
        raise SystemExit(f"h={h}: 2016+ features differ from the 99 dataset ({same:.6f}, {seas} {seas_same:.6f})")
    del ref, a, j
    gc.collect()
    entry = {
        "snapshot": ds.snapshot,
        "as_of": str(ds.as_of.date()),
        "sessions": SESS[h],
        "digest": dig,
        "labelled_by_year": {int(y): float(r["mean"]) for y, r in by_year.iterrows()},
        "features_equal_to_99": same,
        "seas_equal_to_99_late": seas_same,
        "excluded_families_equal_to_99": fam_same,
        "features_equal_to_99_by_col": per,
        "mcap_missing_by_year": {int(y): v for y, v in mc_miss.items()},
        "spell_entities": spells,
        "regime_equal_to_99": rsame,
        "version": V100_VERSION,
    }
    if h == 1:
        xs = [c for c in ds.xs_cols if not c.startswith(EXCLUDE_PREFIX)]
        cx = coverage(f, xs)
        rf_ = ds.dates_frame
        old = rf_[(rf_.index >= OLD[0]) & (rf_.index <= OLD[1])]
        new = rf_[rf_.index >= CUT]
        cr = {c: (float(old[c].notna().mean()), float(new[c].notna().mean())) for c in ds.regime_cols}
        keep_x = [c for c, (o, n) in cx.items() if o >= MIN_COVER and n >= MIN_COVER]
        keep_r = [c for c, (o, n) in cr.items() if o >= MIN_COVER and n >= MIN_COVER]
        say(f"features kept {len(keep_x)}/{len(ds.xs_cols)}: {' '.join(keep_x)}")
        say(f"dropped: {' '.join(f'{c}({o:.2f}/{n:.2f})' for c, (o, n) in cx.items() if c not in keep_x)}")
        say(f"regime kept {len(keep_r)}/{len(ds.regime_cols)}: {' '.join(keep_r)}")
        say(f"regime dropped: {' '.join(f'{c}({o:.2f}/{n:.2f})' for c, (o, n) in cr.items() if c not in keep_r)}")
        if len(keep_x) < 10:
            raise SystemExit(f"only {len(keep_x)} features pass the coverage rule")
        entry.update({"keep_xs": keep_x, "keep_regime": keep_r, "coverage_xs": cx, "coverage_regime": cr})
    st[str(h)] = entry
    STATE.write_text(json.dumps(st, indent=1, default=str))


def prepared(arm: str, s, cache):
    h, cut = ARMS[arm]
    st = state()
    if str(h) not in st or "1" not in st:
        raise SystemExit(f"arm {arm}: run build --h 1 and --h {h} first")
    ds = cache.get("modeling_dataset", st[str(h)]["snapshot"], None)
    if ds is None or int(ds.horizon) != SESS[h]:
        raise SystemExit(f"arm {arm}: dataset missing or wrong horizon")
    ds = guard_xsh(ds, s, say)
    kx, kr = st["1"]["keep_xs"], st["1"]["keep_regime"]
    ds.xs_cols = [c for c in ds.xs_cols if c in kx]
    ds.regime_cols = [c for c in ds.regime_cols if c in kr]
    ds.core_xs_cols = [c for c in ds.core_xs_cols if c in kx]
    ds.core_regime_cols = [c for c in ds.core_regime_cols if c in kr]
    if cut is not None:
        ds.frame = ds.frame[ds.frame["date"] >= cut].reset_index(drop=True)
        ds.month_ends = ds.month_ends[ds.month_ends >= cut]
        ds.dates_frame = ds.dates_frame[ds.dates_frame.index >= cut]
    tag = hashlib.sha1(json.dumps([kx, kr]).encode()).hexdigest()[:10]
    return ds, f"{ds.snapshot}|v100|{V100_VERSION}|{arm}|{tag}"


def run(arm: str, no_fit: bool):
    h, _ = ARMS[arm]
    s = settings(h)
    cache = DiskCache(s.cache_path)
    ds, tag = prepared(arm, s, cache)

    def guard(msg: str) -> None:
        if not no_fit:
            say(f"[{arm}] {msg}")
        elif "train <=" in msg:
            raise SystemExit(f"analyse: arm {arm} context is not cached ({msg})")

    snap0 = ds.snapshot
    ds.snapshot = tag
    try:
        res = sel.run_selection(ds, s, cache, candidates=v99.candidates(), threads=8, use_cache=True, progress=guard)
    finally:
        ds.snapshot = snap0
    return ds, res


def study(arm: str) -> None:
    _, res = run(arm, no_fit=False)
    say(f"arm {arm} done: protocol selected {res.selected}")


# ------------------------------------------------------------------------------------------------ analysis


def window(r: pd.Series, a: str, b: str) -> pd.Series:
    return r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]


def analyse() -> None:
    preds, info, latest = {}, {}, {}
    ret = uni = None
    for arm in ARMS:
        ds, res = run(arm, no_fit=True)
        f = pd.concat([res.oos, res.holdout_preds], ignore_index=True)
        f = f[f.groupby("date")["y_e"].transform("size") >= mt.MIN_NAMES]
        preds[arm] = f[["date", "entity", *NAMES]].copy()
        info[arm] = {
            "oos": [str(f["date"].min().date()), str(f["date"].max().date())],
            "months": int(f["date"].nunique()),
            "holdout": [str(res.holdout_preds["date"].min().date()), str(res.holdout_preds["date"].max().date())],
            "protocol_selected": res.selected,
            "ic_oos": {n: float(mt.per_date_ic(f, n).mean()) for n in NAMES},
            "features": len(ds.xs_cols),
            "regime": len(ds.regime_cols),
        }
        if res.final is not None and arm in ("A", "C"):
            z = res.final.predictions.dropna(subset=["rf_bo"]).sort_values("rf_bo", ascending=False)
            latest[arm] = [str(e) for e in z["entity"].head(int(np.ceil(len(z) / K)))]
        if arm == "A":
            lab = f.dropna(subset=["r"])
            R = np.expm1(lab["r"].to_numpy(float))
            ret = pd.Series(R, index=pd.MultiIndex.from_frame(lab[["date", "entity"]]))
            ret = ret[~ret.index.duplicated()]
            uni = lab.assign(R=R).groupby("date")["R"].mean()
        say(f"arm {arm}: OOS {info[arm]['oos']} ({info[arm]['months']} month-ends)")
        del ds, res, f
        gc.collect()

    def books(arm: str, n: str, grid: list, hold: int):
        bk = v99.Books(grid, ret, uni)
        top, top20 = v99.ventile_sets(preds[arm][preds[arm]["date"].isin(grid)], n)
        r, tc = bk.hold(top, hold) if hold else bk.buffer(top, top20)[:2]
        return {c: r - c * tc for c in COSTS} | {"gross": r, "tc": tc}

    out = {"version": V100_VERSION, "start": START, "cut": CUT, "info": info, "rule": {}, "cells": {}}
    # rule 1: A vs B, hold 1, B's OOS months
    g_b = sorted(set(preds["B"]["date"].unique()) & set(preds["A"]["date"].unique()) & set(uni.index))
    a1, b1 = books("A", "rf_bo", g_b, 1), books("B", "rf_bo", g_b, 1)
    p = v99.paired(a1[COSTS[0]], b1[COSTS[0]])
    out["rule"]["1_long_history_1m"] = {
        **p,
        "helps": bool(p["t"] >= 2 and p["last24_sum"] >= 0),
        "window": [str(g_b[0].date()), str(g_b[-1].date())],
    }
    # rule 2: A fresh 2005-2015, hold 1
    g_a = sorted(set(preds["A"]["date"].unique()) & set(uni.index))
    fresh = {}
    for n in NAMES:
        x = books("A", n, g_a, 1)
        fr = window(x[COSTS[0]], *FRESH)
        fresh[n] = v99.stats(fr, uni.reindex(fr.index))
    out["rule"]["2_fresh_2005_2015_1m"] = {
        **fresh["rf_bo"],
        "confirmed": bool(fresh["rf_bo"]["t_vs_u"] >= 2),
        "others": {n: fresh[n] for n in NAMES if n != "rf_bo"},
    }
    # rule 3: C vs D at hold 12; C fresh at hold 12
    g_cd = sorted(set(preds["C"]["date"].unique()) & set(preds["D"]["date"].unique()) & set(uni.index))
    c12, d12 = books("C", "rf_bo", g_cd, 12), books("D", "rf_bo", g_cd, 12)
    w = g_cd[11:]
    p = v99.paired(c12[COSTS[0]].loc[w], d12[COSTS[0]].loc[w])
    out["rule"]["3_long_history_12m"] = {
        **p,
        "helps": bool(p["t"] >= 2 and p["last24_sum"] >= 0),
        "window": [str(w[0].date()), str(w[-1].date())],
    }
    g_c = sorted(set(preds["C"]["date"].unique()) & set(uni.index))
    c12a = books("C", "rf_bo", g_c, 12)
    fr = window(c12a[COSTS[0]].loc[g_c[11:]], *FRESH)
    st = v99.stats(fr, uni.reindex(fr.index))
    out["rule"]["3_fresh_2005_2015_12m"] = {**st, "confirmed": bool(st["t_vs_u"] >= 2)}
    # 100 addendum (2026-11-09, before any 100 result): the 99 post hoc candidate rf @ 12 months held 12 months
    c12rf = books("C", "rf", g_c, 12)
    fr = window(c12rf[COSTS[0]].loc[g_c[11:]], *FRESH)
    st = v99.stats(fr, uni.reindex(fr.index))
    out["rule"]["4_fresh_2005_2015_rf_12m"] = {**st, "confirmed": bool(st["t_vs_u"] >= 2)}
    # descriptive cells on each arm's own grid (from its 12th month)
    grids = {"A": g_a, "B": g_b, "C": g_c, "D": sorted(set(preds["D"]["date"].unique()) & set(uni.index))}
    for arm, grid in grids.items():
        ww = grid[11:]
        for n in NAMES:
            for hold in (1, 3, 6, 12, 0):
                x = books(arm, n, grid, hold)
                key = f"{n}@{arm}/{'buffer' if hold == 0 else f'hold{hold}'}"
                out["cells"][key] = {
                    "turnover": float(x["tc"].loc[ww].mean()),
                    "gross": v99.stats(x["gross"].loc[ww], uni.reindex(ww)),
                    **{f"net{int(c * 1e4)}": v99.stats(x[c].loc[ww], uni.reindex(ww)) for c in COSTS},
                }
    x = books("A", "rf_bo", g_a, 1)
    yt = (1 + x[COSTS[0]].fillna(0)).groupby(x[COSTS[0]].index.year).prod() - 1
    yu = (1 + uni.reindex(g_a).fillna(0)).groupby(pd.DatetimeIndex(g_a).year).prod() - 1
    out["yearly_A_rf_bo_net50"] = {int(y): [float(yt[y]), float(yu[y])] for y in yt.index}
    cr = window(x[COSTS[0]], *CRISIS)
    out["crisis_A_rf_bo"] = {
        "D20_cum": float((1 + cr.fillna(0)).prod() - 1),
        "universe_cum": float((1 + uni.reindex(cr.index).fillna(0)).prod() - 1),
    }
    out["universe_ann_A"] = v99.stats(uni.reindex(g_a[11:]), uni.reindex(g_a[11:]))["ann"]
    out["latest_rf_bo_D20"] = latest
    p99 = OUT / "KR_v99_horizons.json"
    if p99.exists():
        c99 = json.loads(p99.read_text())["cells"].get("rf_bo@1m/hold1")
        out["ref_99_full_features_1m"] = c99
    (OUT / "KR_v100_long.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_md(out)
    say(f"wrote {OUT / 'KR_v100_long.json'}")


def write_md(o: dict) -> None:
    def pc(x: float) -> str:
        return f"{100 * x:+.1f}%"

    r = o["rule"]
    lines = [f"# KR 100: 2002년부터 학습·평가 (시작 {o['start']}, 자르기 {o['cut']})", ""]
    for arm, i in o["info"].items():
        lines.append(
            f"- {arm}: OOS {i['oos'][0]}~{i['oos'][1]} ({i['months']}개월), 특징 {i['features']}·국면 {i['regime']}"
        )
    lines += ["", "## 판정 (100, 사전 고정)", ""]
    a = r["1_long_history_1m"]
    lines.append(
        f"1. 긴 학습(1개월) A−B: 월 {100 * a['mean_month']:+.2f}%p, t {a['t']:+.2f} → "
        f"{'도움' if a['helps'] else '도움 없음'}"
    )
    b = r["2_fresh_2005_2015_1m"]
    lines.append(
        f"2. 2005~2015 `rf_bo` D20 비용 후 연 {pc(b['ann'])}, 전체 대비 t {b['t_vs_u']:+.2f} → "
        f"{'확인' if b['confirmed'] else '확인 안 됨'}"
    )
    c = r["3_long_history_12m"]
    lines.append(
        f"3. 긴 학습(12개월) C−D: 월 {100 * c['mean_month']:+.2f}%p, t {c['t']:+.2f} → "
        f"{'도움' if c['helps'] else '도움 없음'}"
    )
    d = r["3_fresh_2005_2015_12m"]
    lines.append(
        f"   2005~2015 12개월 보유 비용 후 연 {pc(d['ann'])}, t {d['t_vs_u']:+.2f} → "
        f"{'확인' if d['confirmed'] else '확인 안 됨'}"
    )
    e = r["4_fresh_2005_2015_rf_12m"]
    lines.append(
        f"4. 2005~2015 `rf` 12개월 모델·12개월 보유 비용 후 연 {pc(e['ann'])}, t {e['t_vs_u']:+.2f} → "
        f"{'확인' if e['confirmed'] else '확인 안 됨'}"
    )
    lines += [
        "",
        "| 칸 | 연(비용 전) | 연(50bp) | 연(100bp) | t(50bp) | 최대 낙폭 | 월 교체 |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, v in o["cells"].items():
        lines.append(
            f"| `{k}` | {pc(v['gross']['ann'])} | {pc(v['net50']['ann'])} | {pc(v['net100']['ann'])} | "
            f"{v['net50']['t_vs_u']:+.2f} | {100 * v['gross']['mdd']:.0f}% | {100 * v['turnover']:.0f}% |"
        )
    lines += [
        "",
        "연도별 A `rf_bo` D20(50bp) / 전체: "
        + " ".join(
            f"{y % 100:02d}:{100 * a_:+.0f}/{100 * b_:+.0f}" for y, (a_, b_) in o["yearly_A_rf_bo_net50"].items()
        ),
    ]
    cr = o["crisis_A_rf_bo"]
    lines.append(f"2007-11~2009-02: D20 {pc(cr['D20_cum'])} / 전체 {pc(cr['universe_cum'])}")
    (OUT / "KR_v100_long.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "study", "analyse"))
    ap.add_argument("--h", type=int, choices=sorted(SESS))
    ap.add_argument("--arm", choices=sorted(ARMS))
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.h)
    elif a.cmd == "study":
        study(a.arm)
    else:
        analyse()


if __name__ == "__main__":
    main()
