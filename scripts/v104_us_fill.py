"""US: fill the S&P 500 members without a price history from Tiingo (research data only) and re-evaluate
(docs/MODELING.md 104, pre-declared 2026-11-14). src/quantcio, the config and the served models are not changed.

    .venv/bin/python scripts/v104_us_fill.py base      # today's universe (frozen vintage) + baseline dataset
    .venv/bin/python scripts/v104_us_fill.py lists     # the rule 2 sample (30 names that already have a frame)
    .venv/bin/python scripts/v104_us_fill.py check     # rule 1 (SEC FTD vs Tiingo raw close), rule 2 (Tiingo vs Yahoo)
    .venv/bin/python scripts/v104_us_fill.py build     # the filled dataset (research cache only)
    .venv/bin/python scripts/v104_us_fill.py books     # rules 3-5: coverage, small-cap book, the 103 signal books
    .venv/bin/python scripts/v104_us_fill.py study     # rule 6: the 95 protocol and display rule on the filled data
    .venv/bin/python scripts/v104_us_fill.py analyse   # rule 6 again from the cached run (no fit)

Raw Tiingo files live in .cache/quantcio/research/tiingo/ (downloaded by .cache/quantcio/research/v104_download.py;
free-tier data, never committed). Writes reports/modeling/US_v104_{fill,books,study}.{json,md}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path

for _k in ("KRX_ID", "KRX_PW"):
    os.environ.pop(_k, None)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v86_full_es as v86  # noqa: E402

from quantcio.config import load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = v86.OUT
say = v86.say
RAW = Path(".cache/quantcio/research/tiingo")
WORK = Path(".cache/quantcio/research/v104")
STATE = OUT / "US_v104_state.json"
COVER = OUT / "US_v103_tiingo_cover.csv"
NS = "research_v104"
V104_VERSION = 1
FRAME_FROM = pd.Timestamp("2011-01-01")
FTD_MIN, FTD_TOL, FTD_SHARE = 5, 0.01, 0.80  # rule 1
N_CHECK, RET_TOL, RET_SHARE = 30, 0.01, 0.95  # rule 2
DELIST_PAD = pd.Timedelta(5, "D")


def base() -> None:
    """Today's PIT universe through the vintages (as ``build_dataset`` does), its month sources, and the baseline
    dataset (frozen-window digest must match)."""
    from quantcio.data import vintage as vt
    from quantcio.data.pit import load_pit_universe
    from quantcio.data.pit import us as pit_us
    from quantcio.data.pit.us_net import PitNet
    from quantcio.modeling.dataset import _vintage_tag, build_dataset, universe_params

    s = load_settings()
    cache = DiskCache(s.cache_path)
    n, start = universe_params(Market.US, s)
    say(f"US universe n={n} start={start}")
    uni = load_pit_universe(Market.US, s, cache, n, start)
    full = pit_us._full_build(s, cache, PitNet(s), pd.Timestamp(start), None)
    vstore = vt.VintageStore(s.cache_path / "vintage", settle_days=int(s.modeling.vintage_settle_days))
    frozen, infos = vt.freeze_universe(vstore, uni, tag=_vintage_tag(Market.US, n, start))
    say(f"frozen universe: {len(frozen.prices)} frames, cutoff {infos['prices'].cutoff}")
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "universe.pkl", "wb") as fh:
        pickle.dump({"universe": frozen, "month_src": full.month_src, "members": full.members}, fh)
    ds = build_dataset(Market.US, s, cache, use_cache=True, progress=say)
    dig = (ds.coverage.get("vintage") or {}).get("digest") or {}
    say(f"baseline dataset {ds.snapshot} as_of {ds.as_of.date()} rows {len(ds.frame):,}; frozen digest {dig}")
    if dig.get("status") != "match":
        raise SystemExit(f"frozen-window digest is not a match: {dig}")
    STATE.write_text(
        json.dumps(
            {"baseline": ds.snapshot, "as_of": str(ds.as_of.date()), "digest": dig, "version": V104_VERSION},
            indent=1,
            default=str,
        )
    )


# ------------------------------------------------------------------------------------------------ Tiingo frames


def load_work() -> dict:
    with open(WORK / "universe.pkl", "rb") as fh:
        return pickle.load(fh)  # noqa: S301 - our own file


def tiingo_chart(sym: str) -> pd.DataFrame | None:
    """Tiingo daily rows -> a Yahoo-chart-like frame: split-adjusted OHLC / volume, ``adjclose`` (total return),
    ``raw_close`` and ``split`` (the day's split factor). Zero-volume rows (frozen after a delisting) are dropped."""
    path = RAW / f"{sym.upper()}.json"
    if not path.exists():
        return None
    j = json.loads(path.read_text())
    if not j:
        return None
    d = pd.DataFrame(j)
    d.index = pd.DatetimeIndex(pd.to_datetime(d["date"]).dt.tz_localize(None)).normalize()
    d = d[~d.index.duplicated(keep="last")].sort_index()
    sf = d["splitFactor"].astype(float).replace(0.0, 1.0).fillna(1.0)
    later = sf[::-1].cumprod()[::-1].shift(-1).fillna(1.0)  # product of the split factors AFTER each day
    out = pd.DataFrame(index=d.index)
    for c in ("open", "high", "low", "close"):
        out[c] = d[c].astype(float) / later
    out["volume"] = d["volume"].astype(float) * later
    out["adjclose"] = d["adjClose"].astype(float)
    out["raw_close"] = d["close"].astype(float)
    out["split"] = sf
    out = out[(d["volume"].astype(float) > 0) & (out["close"] > 0) & out["adjclose"].notna()]
    return out


def missing_table(u) -> pd.DataFrame:
    """Entities of the universe without a frame, with their tickers, membership span and the Tiingo ticker of 103.5."""
    mem = u.membership
    miss = sorted(set(mem["entity"]) - {e for e, f in u.prices.items() if f is not None and len(f)})
    span = mem.groupby("entity")["date"].agg(["min", "max", "size"])
    tk = mem.sort_values("date").groupby("entity")["ticker"].agg(lambda x: list(dict.fromkeys(str(v) for v in x)))
    cov = pd.read_csv(COVER).set_index("entity")
    rows = []
    for e in miss:
        tg = cov["tiingo"].get(e)
        rows.append(
            {
                "entity": e,
                "tickers": tk.loc[e],
                "first": span.loc[e, "min"],
                "last": span.loc[e, "max"],
                "member_months": int(span.loc[e, "size"]),
                "tiingo": str(tg).split("@")[0] if isinstance(tg, str) and tg else None,
            }
        )
    return pd.DataFrame(rows)


def cut_frame(df: pd.DataFrame, delist_date) -> pd.DataFrame:
    df = df[df.index >= FRAME_FROM]
    if delist_date is not None and pd.notna(delist_date):
        df = df[df.index <= pd.Timestamp(delist_date) + DELIST_PAD]
    return df


# ------------------------------------------------------------------------------------------------ rules 1, 2


def check_names(u) -> list[str]:
    """Rule 2 sample: the entities WITH a frame whose sha1(entity id) is smallest."""
    have = [e for e, f in u.prices.items() if f is not None and len(f)]
    return sorted(have, key=lambda e: hashlib.sha1(e.encode()).hexdigest())[:N_CHECK]  # noqa: S324


def check_symbol(u, e: str) -> str:
    return str(u.prices[e].attrs.get("symbol") or u.meta.set_index("entity").loc[e, "yahoo_symbol"])


def lists() -> None:
    u = load_work()["universe"]
    syms = [check_symbol(u, e) for e in check_names(u)]
    (RAW / "list_check.txt").write_text("\n".join(syms) + "\n")
    say(f"rule 2 sample ({len(syms)}): {' '.join(syms)}")


def month_returns(close: pd.Series, dates: pd.DatetimeIndex, nxt: dict) -> pd.Series:
    """Return between consecutive rebalance dates (last bar <= date, at most 7 days old) for ``dates``."""
    idx = close.index
    out = {}
    for d in dates:
        d1 = nxt.get(d)
        if d1 is None:
            continue
        i0, i1 = idx.searchsorted(d, side="right") - 1, idx.searchsorted(d1, side="right") - 1
        if i0 < 0 or i1 <= i0 or (d - idx[i0]).days > 7 or (d1 - idx[i1]).days > 7:
            continue
        out[d] = float(close.iloc[i1] / close.iloc[i0] - 1)
    return pd.Series(out, dtype=float)


def check() -> None:
    from quantcio.data.pit import us_prices
    from quantcio.data.pit.us_net import PitNet

    s = load_settings()
    w = load_work()
    u = w["universe"]
    meta = u.meta.set_index("entity")
    tab = missing_table(u)
    total_mm = int(tab["member_months"].sum())
    all_dates = pd.DatetimeIndex(sorted(u.membership["date"].unique()))
    nxt = dict(zip(all_dates[:-1], all_dates[1:], strict=False))
    # ---- rule 1: SEC FTD raw prices vs Tiingo raw close
    syms = {t for ts in tab["tickers"] for t in ts}
    lo, hi = all_dates[0] - pd.Timedelta(60, "D"), all_dates[-1] + pd.Timedelta(45, "D")
    say(f"FTD panel for {len(syms)} symbols {lo.date()}..{hi.date()}")
    panel, ftd_missing = us_prices.ftd_panel(PitNet(s), syms, lo, hi)
    panel = panel[panel["symbol"] == panel["key"]]
    by = {k: g for k, g in panel.groupby("key")}
    rows = []
    for r in tab.itertuples(index=False):
        o = {
            "entity": r.entity,
            "tickers": "/".join(r.tickers),
            "member_months": r.member_months,
            "first": str(r.first.date()),
            "last": str(r.last.date()),
            "tiingo": r.tiingo,
            "accepted": False,
        }
        df = tiingo_chart(r.tiingo) if r.tiingo else None
        if df is None:
            o["reason"] = "not_listed" if not r.tiingo else "no_file"
            rows.append(o)
            continue
        a, b = r.first - pd.Timedelta(31, "D"), r.last + pd.Timedelta(31, "D")
        parts = [by[t] for t in r.tickers if t in by]
        f = pd.concat(parts) if parts else pd.DataFrame(columns=["px_date", "price"])
        f = f[(f["px_date"] >= a) & (f["px_date"] <= b)].drop_duplicates("px_date")
        c = df["raw_close"].reindex(pd.DatetimeIndex(f["px_date"])).to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = np.isfinite(c) & (np.abs(f["price"].to_numpy(dtype=float) / c - 1) <= FTD_TOL)
        n, k = len(f), int(ok.sum())
        dd = meta.loc[r.entity, "delist_date"] if r.entity in meta.index else None
        fr = cut_frame(df, dd)
        o.update(
            ftd_prints=n,
            ftd_match=k,
            ftd_share=round(k / n, 4) if n else None,
            bars=len(fr),
            first_bar=str(fr.index[0].date()) if len(fr) else None,
            last_bar=str(fr.index[-1].date()) if len(fr) else None,
        )
        if n < FTD_MIN:
            o["reason"] = "few_ftd_prints"
        elif k / n < FTD_SHARE:
            o["reason"] = "ftd_mismatch"
        elif not len(fr):
            o["reason"] = "no_bars_in_window"
        else:
            o["accepted"] = True
            mm = u.membership.loc[u.membership["entity"] == r.entity, "date"]
            pos = fr.index.searchsorted(mm.to_numpy(), side="right") - 1
            okp = pos >= 0
            fresh = np.zeros(len(mm), dtype=bool)
            fresh[okp] = (mm.to_numpy()[okp] - fr.index.to_numpy()[pos[okp]]) <= np.timedelta64(7, "D")
            o["member_months_priced"] = int(fresh.sum())
        rows.append(o)
    d = pd.DataFrame(rows)
    acc = d[d["accepted"]]
    priced = int(acc["member_months_priced"].sum()) if len(acc) else 0
    out = {
        "version": V104_VERSION,
        "missing_entities": len(d),
        "missing_member_months": total_mm,
        "all_member_months": len(u.membership),
        "downloaded": int(d["ftd_prints"].notna().sum()) if "ftd_prints" in d else 0,
        "accepted": len(acc),
        "accepted_member_months_priced": priced,
        "reasons": {str(k): int(v) for k, v in d.loc[~d["accepted"], "reason"].value_counts().items()},
        "ftd_files_missing": [k for k, _ in ftd_missing],
        "rule1": {"min_prints": FTD_MIN, "tolerance": FTD_TOL, "share": FTD_SHARE},
        "entities": rows,
    }
    say(
        f"rule 1: {len(acc)} of {len(d)} accepted ({priced:,} of {total_mm:,} missing member-months priced); "
        f"rejected {out['reasons']}"
    )
    # ---- rule 2: Tiingo vs Yahoo monthly returns on the sample
    pairs = []
    per = {}
    absent = []
    for e in check_names(u):
        sym = check_symbol(u, e)
        df = tiingo_chart(sym)
        if df is None:
            absent.append(sym)
            continue
        yf = u.prices[e]
        adjusted = bool(yf.attrs.get("adjusted", True))
        tc = df["adjclose"] if adjusted else df["close"]
        mm = pd.DatetimeIndex(sorted(u.membership.loc[u.membership["entity"] == e, "date"]))
        ry, rt = month_returns(yf["close"], mm, nxt), month_returns(tc, mm, nxt)
        j = pd.concat([ry.rename("y"), rt.rename("t")], axis=1).dropna()
        okm = (j["y"] - j["t"]).abs() <= RET_TOL
        per[sym] = {
            "entity": e,
            "months": len(j),
            "yahoo_months": len(ry),
            "ok": int(okm.sum()),
            "adjusted": adjusted,
            "max_abs_diff": float((j["y"] - j["t"]).abs().max()) if len(j) else None,
        }
        pairs.append(okm)
    allp = pd.concat(pairs) if pairs else pd.Series(dtype=bool)
    share = float(allp.mean()) if len(allp) else float("nan")
    out["rule2"] = {
        "sample": N_CHECK,
        "compared": len(per),
        "not_downloaded": absent,
        "entity_months": len(allp),
        "share_within_tol": share,
        "tolerance": RET_TOL,
        "threshold": RET_SHARE,
        "complete": not absent,
        "pass": bool(not absent and share >= RET_SHARE),
        "per_symbol": per,
    }
    say(f"rule 2: {len(per)} symbols, {len(allp):,} entity-months, within {RET_TOL:.0%}p {share:.2%}; missing {absent}")
    (OUT / "US_v104_fill.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    lines = [
        "# US 104.1: Tiingo로 메운 종목의 검증",
        "",
        f"- 가격 없는 구성 종목 {len(d)}개(구성원-월 {total_mm:,}) 중 채택 {len(acc)}개, "
        f"가격이 생긴 구성원-월 {priced:,}",
        f"- 제외 사유: {out['reasons']}",
        f"- 규칙 2: {len(per)}종목 {len(allp):,}종목-월 중 {share:.2%}가 {RET_TOL:.0%}p 이내 → "
        f"{'통과' if out['rule2']['pass'] else ('미완료' if absent else '미달')}",
        "",
        "| 종목 | Tiingo | 구성원-월 | FTD 기록 | 일치 | 일치율 | 판정 |",
        "|---|---|---|---|---|---|---|",
    ]
    for o in sorted(rows, key=lambda x: -x["member_months"]):
        sh = "—" if o.get("ftd_share") is None else f"{100 * o['ftd_share']:.0f}%"
        lines.append(
            f"| {o['tickers']} | {o['tiingo'] or '—'} | {o['member_months']} | {o.get('ftd_prints', '—')} | "
            f"{o.get('ftd_match', '—')} | {sh} | {'채택' if o['accepted'] else o.get('reason')} |"
        )
    (OUT / "US_v104_fill.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ filled dataset


def build(dry: bool = False) -> None:
    """``dry``: a code test on whatever is downloaded so far - no rule 2 gate, own cache key, no state change."""
    from quantcio.data.pit import load_pit_fundamentals, us_prices
    from quantcio.modeling.dataset import build_dataset

    s = load_settings()
    cache = DiskCache(s.cache_path)
    fill = json.loads((OUT / "US_v104_fill.json").read_text())
    if not fill["rule2"]["pass"] and not dry:
        raise SystemExit(
            f"rule 2 is not passed: {fill['rule2']['share_within_tol']} (complete {fill['rule2']['complete']})"
        )
    u = load_work()["universe"]
    meta = u.meta.copy().set_index("entity", drop=False)
    bench = next(iter(u.benchmarks))
    today = max(f.index[-1] for f in u.prices.values() if f is not None and len(f))
    ents = sorted(u.membership["entity"].unique())
    fund = load_pit_fundamentals(Market.US, s, cache, ents)
    sh = fund.panel.dropna(subset=["shares_out", "available_from"])
    sh = sh[["entity", "available_from", "period_end", "shares_out", "filed"]].sort_values("available_from")
    stale = int(s.pit.fundamentals.max_staleness_days)
    prices, eb, mem = dict(u.prices), dict(u.entity_benchmark), u.membership.copy()
    n_caps = n_rows = newly_delisted = 0
    for r in fill["entities"]:
        if not r["accepted"]:
            continue
        e = r["entity"]
        df = cut_frame(tiingo_chart(r["tiingo"]), meta.loc[e, "delist_date"])
        fr = us_prices.price_frame(df, adjusted=True)
        fr.attrs.update(source="tiingo", symbol=r["tiingo"], split_adjusted=True, validated=True)
        prices[e], eb[e] = fr, bench
        if fr.index[-1] < today - pd.Timedelta(10, "D") and not bool(meta.loc[e, "delisted"]):
            meta.loc[e, "delisted"] = True
            meta.loc[e, "delist_date"] = fr.index[-1]
            newly_delisted += 1
        meta.loc[e, "first_date"], meta.loc[e, "last_date"] = fr.index[0], fr.index[-1]
        meta.loc[e, "price_source"] = "tiingo"
        # PIT market cap: raw close x shares filed by the date x splits between the filing and the date
        rows = mem.index[mem["entity"] == e]
        left = mem.loc[rows, ["date"]].reset_index().sort_values("date")
        j = pd.merge_asof(left, sh[sh["entity"] == e], left_on="date", right_on="available_from", direction="backward")
        idx = df.index
        for row in j.itertuples(index=False):
            n_rows += 1
            k = idx.searchsorted(row.date, side="right") - 1
            if k < 0 or (row.date - idx[k]).days > 7 or pd.isna(row.shares_out):
                continue
            if (row.date - row.period_end).days > stale:
                continue
            filed = pd.Timestamp(row.filed) if pd.notna(row.filed) else pd.Timestamp(row.available_from)
            mult = float(df["split"][(idx > filed) & (idx <= row.date)].prod())
            mem.loc[row.index, "rank_value"] = float(df["raw_close"].iloc[k]) * float(row.shares_out) * mult
            n_caps += 1
    say(
        f"filled frames {len(prices) - len(u.prices)}, market caps set {n_caps:,} of {n_rows:,} member-months, "
        f"newly marked delisted {newly_delisted}"
    )
    u2 = replace(u, membership=mem, prices=prices, entity_benchmark=eb, meta=meta.reset_index(drop=True))
    ds = build_dataset(Market.US, s, cache, universe=u2, use_cache=False, progress=say)
    cache.set(NS, "dry" if dry else f"filled_v{V104_VERSION}", ds)
    if dry:
        say(f"dry run: {len(ds.frame):,} rows, labels {ds.coverage['label_status']}")
        return
    st = json.loads(STATE.read_text())
    st.update(filled=ds.snapshot, filled_rows=len(ds.frame), filled_as_of=str(ds.as_of.date()), caps_set=n_caps)
    STATE.write_text(json.dumps(st, indent=1, default=str))
    cov = ds.coverage
    say(
        f"filled dataset {ds.snapshot} as_of {ds.as_of.date()} rows {len(ds.frame):,} "
        f"(member rows {cov['member_rows']:,}, without price {cov['no_price_rows']:,}); labels {cov['label_status']}"
    )


def datasets(s, cache, dry: bool = False) -> dict:
    from quantcio.modeling.cli import guard_xsh

    st = json.loads(STATE.read_text())
    base = cache.get("modeling_dataset", st["baseline"], None)
    filled = cache.get(NS, "dry" if dry else f"filled_v{V104_VERSION}", None)
    if base is None or filled is None:
        raise SystemExit("baseline or filled dataset is not cached (run base / build)")
    return {"baseline": guard_xsh(base, s, say), "filled": guard_xsh(filled, s, say)}


# ------------------------------------------------------------------------------------------------ rules 3-5


def books(dry: bool = False) -> None:
    import v103_us_prior as v103

    from quantcio.modeling import metrics as mt

    s = load_settings()
    cache = DiskCache(s.cache_path)
    u = load_work()["universe"]
    dss = datasets(s, cache, dry)
    out: dict = {"version": V104_VERSION, "data": {}, "small": {}, "large": {}, "signals": {}, "avoid": {}}
    for tag, ds in dss.items():
        f = ds.frame
        d = f[f["is_me"].astype(bool) & f["labelled"].astype(bool)].dropna(subset=["r"]).copy()
        d = d[d.groupby("date")["entity"].transform("size") >= mt.MIN_NAMES]
        d["R"] = np.expm1(d["r"].to_numpy(float))
        cost = float(ds.cost_bps) / 1e4
        uni = d.groupby("date")["R"].mean()
        uni.index = pd.DatetimeIndex(uni.index)
        months = len(uni)
        me = f[f["is_me"].astype(bool)]
        mem = u.membership[u.membership["date"].isin(pd.DatetimeIndex(me["date"].unique()))]
        priced = me.groupby(me["date"].dt.year).size()
        member = mem.groupby(mem["date"].dt.year).size()
        out["data"][tag] = {
            "snapshot": ds.snapshot,
            "months": months,
            "span": [str(uni.index[0].date()), str(uni.index[-1].date())],
            "median_names": int(d.groupby("date").size().median()),
            "universe_ann": float((1 + uni).prod() ** (12 / months) - 1),
            "member_months": len(mem),
            "priced_member_months": len(me),
            "missing_share": float(1 - len(me) / len(mem)),
            "missing_share_by_year": {str(int(y)): float(1 - priced.get(y, 0) / n) for y, n in member.items()},
            "label_status": ds.coverage.get("label_status"),
        }
        say(
            f"{tag}: {months} months, median {out['data'][tag]['median_names']} names, universe "
            f"{100 * out['data'][tag]['universe_ann']:+.2f}%/yr, missing member-months "
            f"{100 * out['data'][tag]['missing_share']:.1f}%"
        )
        dm = d[d["mcap"].notna()]
        for key, (lo, hi) in {"small": (0.0, 0.05), "large": (0.95, 1.01)}.items():
            b = v103.book(dm, "mcap", lo, hi, cost)
            out[key][tag] = v103.describe(b["net"], uni, b["turn"])
            say(
                f"{tag} {key} 5%: net {100 * out[key][tag]['ann']:+.1f}%/yr beta {out[key][tag]['beta']:.2f} "
                f"alpha {100 * out[key][tag]['alpha_ann']:+.1f}% t {out[key][tag]['t']:+.2f}"
            )
        sig: dict = {}
        for name, (_file, _fav, _group, col, sign) in v103.SIGNALS.items():
            if col is None or col not in d:
                continue
            d["_s"] = sign * d[col].astype(float).fillna(0.0)
            row = {}
            for bk, (lo, hi) in {"Q5": (0.80, 1.01), "Q1": (0.0, 0.20)}.items():
                b = v103.book(d, "_s", lo, hi, cost)
                row[bk] = v103.describe(b["net"], uni, b["turn"])
            sig[name] = row
            say(
                f"{tag} {name:14s} Q5 alpha {100 * row['Q5']['alpha_ann']:+5.1f}% t {row['Q5']['t']:+.2f} | "
                f"Q1 alpha {100 * row['Q1']['alpha_ann']:+5.1f}% t {row['Q1']['t']:+.2f}"
            )
        out["signals"][tag] = sig
        out["avoid"][tag] = sig["momentum"]["Q1"]
    gap = out["small"]["filled"]["alpha_ann"] - out["small"]["baseline"]["alpha_ann"]
    av = out["avoid"]["filled"]
    out["rule4"] = {
        "alpha_change": gap,
        "threshold": 0.02,
        "verdict": "partly_the_hole" if gap >= 0.02 else "not_the_hole",
    }
    out["rule5"] = {
        "alpha": av["alpha_ann"],
        "t": av["t"],
        "verdict": "kept" if av["alpha_ann"] < 0 and av["t"] <= -2 else ("weaker" if av["alpha_ann"] < 0 else "drop"),
    }
    say(
        f"rule 4: small-cap alpha change {100 * gap:+.2f}%p -> {out['rule4']['verdict']}; rule 5: avoid side alpha "
        f"{100 * av['alpha_ann']:+.1f}% t {av['t']:+.2f} -> {out['rule5']['verdict']}"
    )
    if dry:
        say("dry run of books finished (nothing written)")
        return
    (OUT / "US_v104_books.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    b, fl = out["data"]["baseline"], out["data"]["filled"]
    lines = [
        "# US 104.3: 기준판과 메운판 비교 (규칙 3~5)",
        "",
        "| | 기준판 | 메운판 |",
        "|---|---|---|",
        f"| 가격 없는 구성원-월 | {100 * b['missing_share']:.1f}% | {100 * fl['missing_share']:.1f}% |",
        f"| 날짜당 종목 수(중앙값) | {b['median_names']} | {fl['median_names']} |",
        f"| 전체 동일가중 연 수익 | {100 * b['universe_ann']:+.2f}% | {100 * fl['universe_ann']:+.2f}% |",
        f"| 시가총액 하위 5% 알파(연)·t | {100 * out['small']['baseline']['alpha_ann']:+.1f}% "
        f"({out['small']['baseline']['t']:+.2f}) | {100 * out['small']['filled']['alpha_ann']:+.1f}% "
        f"({out['small']['filled']['t']:+.2f}) |",
        f"| 시가총액 상위 5% 알파(연)·t | {100 * out['large']['baseline']['alpha_ann']:+.1f}% "
        f"({out['large']['baseline']['t']:+.2f}) | {100 * out['large']['filled']['alpha_ann']:+.1f}% "
        f"({out['large']['filled']['t']:+.2f}) |",
        "",
        f"규칙 4: {out['rule4']['verdict']} (알파 변화 {100 * gap:+.2f}%p) · 규칙 5: {out['rule5']['verdict']}",
        "",
        "| 신호 | 상위 20% 알파·t 기준판 | 메운판 | 하위 20% 알파·t 기준판 | 메운판 |",
        "|---|---|---|---|---|",
    ]
    for name in out["signals"]["baseline"]:
        x, y = out["signals"]["baseline"][name], out["signals"]["filled"][name]
        lines.append(
            f"| {name} | {100 * x['Q5']['alpha_ann']:+.1f}% ({x['Q5']['t']:+.2f}) | {100 * y['Q5']['alpha_ann']:+.1f}% "
            f"({y['Q5']['t']:+.2f}) | {100 * x['Q1']['alpha_ann']:+.1f}% ({x['Q1']['t']:+.2f}) | "
            f"{100 * y['Q1']['alpha_ann']:+.1f}% ({y['Q1']['t']:+.2f}) |"
        )
    (OUT / "US_v104_books.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ rule 6


def study(no_fit: bool) -> None:
    import v87_validation as v87
    import v88_ventile as v88
    import v89_order_loss as v89
    import v90_order_tree as v90
    import v91_order_weekly as v91
    import v95_us_multi as v95

    from quantcio.modeling import selection as sel

    s = load_settings()
    cache = DiskCache(s.cache_path)
    fill = json.loads((OUT / "US_v104_fill.json").read_text())
    if not fill["rule2"]["pass"]:
        raise SystemExit("rule 2 is not passed: rule 6 is not run")
    rd = s.cache_path / "research" / "v104" / "US"
    rd.mkdir(parents=True, exist_ok=True)
    for mod in (v87, v88, v89, v90, v91):
        mod.LOG[:] = [rd / "choice_log.jsonl"]
    v86.ES_LOG[:] = [rd / "es_log.jsonl"]
    v86.ARM[0] = "v104"
    cands = v95.candidates()
    ds = datasets(s, cache)["filled"]
    say(f"US filled {ds.snapshot[:12]}: {len(ds.xs_cols)} features, {len(cands)} candidates")

    def guard(msg: str) -> None:
        say(msg)
        if no_fit and "train <=" in msg:
            raise SystemExit(f"analyse: a context is not cached ({msg})")

    snap0 = ds.snapshot
    ds.snapshot = f"{snap0}|v104|{V104_VERSION}"
    try:
        res = sel.run_selection(ds, s, cache, candidates=cands, threads=8, use_cache=True, progress=guard)
    finally:
        ds.snapshot = snap0
    o = v95.analyse(res, ds, cands)
    ref_path = OUT / "US_v95.json"
    ref = json.loads(ref_path.read_text())["models"] if ref_path.exists() else {}
    o["baseline_95"] = {
        n: {k: r.get(k) for k in ("t_outer", "t_holdout", "ex_outer", "ex_holdout")} for n, r in ref.items()
    }
    (OUT / "US_v104_study.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    rows = o["models"]
    k = v95.K
    lines = [
        f"# US 104 규칙 6: 메운 자료로 다시 학습 — 분위 장부 ({o['as_of']}, 비용 {o['cost_bps']:.0f}bp)",
        "",
        f"- 바깥 {o['outer'][0]}~{o['outer'][1]}, 확인 {o['holdout'][0]}~{o['holdout'][1]}",
        f"- 프로토콜 선택 `{o['protocol_selected']}` (서빙은 그대로)",
        f"- 표시 규칙(바깥 t ≥ {v95.T_MIN}, 확인 초과 > 0) 통과: {o['passing'] or '없음'} → "
        f"표시 {o['shown'] or '없음'}",
        "",
        "| 모델 | 군 | D1 | D20 | 전체 | D20−전체 t 바깥/확인 | 95절(기준 자료) t 바깥/확인 | 최고 10개월 뺀 D20 "
        "| 최대 낙폭 | IC 학습→OOS |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in sorted(rows, key=lambda n: -rows[n]["t_outer"]):
        r = rows[n]
        a = r["ann"]["all"]
        ici = "—" if r["ic_in"] is None else f"{r['ic_in']:+.3f}"
        b = ref.get(n)
        old = "—" if b is None else f"{b['t_outer']:+.1f} / {b['t_holdout']:+.1f}"
        lines.append(
            f"| `{n}` | {r['family']} | {100 * a.get('1', np.nan):+.1f}% | {100 * a.get(str(k), np.nan):+.1f}% | "
            f"{100 * a['0']:+.1f}% | {r['t_outer']:+.1f} / {r['t_holdout']:+.1f} | {old} | "
            f"{100 * r['D20_drop10']:+.1f}% | {100 * r['D20']['mdd']:.0f}% | {ici} → {r['ic_outer']:+.3f} |"
        )
    (OUT / "US_v104_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"wrote US_v104_study.json/.md; passing {o['passing']}; shown {o['shown']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["base", "lists", "check", "build", "books", "study", "analyse"])
    ap.add_argument("--dry", action="store_true", help="build / books: code test on a partial download")
    a = ap.parse_args()
    if a.cmd in ("study", "analyse"):
        study(no_fit=a.cmd == "analyse")
    else:
        if a.cmd in ("build", "books"):
            {"build": build, "books": books}[a.cmd](a.dry)
        else:
            {"base": base, "lists": lists, "check": check}[a.cmd]()


if __name__ == "__main__":
    main()
