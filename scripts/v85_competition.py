"""Competition structure block: number of rivals, market shares, concentration (docs/MODELING.md 85, pre-declared
2026-11-02). Research only: src/quantcio and the served models are not changed.

    .venv/bin/python scripts/v85_competition.py fetch-us          # SEC Financial Statement Data Sets -> revenue rows
    .venv/bin/python scripts/v85_competition.py study --market KR

Peers: KR = every KOSPI/KOSDAQ XBRL filer in the cached OpenDART bulk files (KSIC 3-digit of the report);
US = every 10-K/10-Q XBRL filer in the SEC Financial Statement Data Sets (SIC 3-digit of the filing).
Two arms on the same frozen dataset snapshot, same six grid candidates: A = current inputs, B = A + 7 x_cm_*.
Contribution = model-averaged per-date primary metric (top_ye_net) of B minus A.
Writes reports/modeling/<M>_v85.{json,md}.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v13_study as v13s  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.data.pit import kr_bulk as kb  # noqa: E402
from quantcio.data.pit import kr_fund as kf  # noqa: E402
from quantcio.data.pit import us_fund  # noqa: E402
from quantcio.data.pit.kr import kr_root  # noqa: E402
from quantcio.data.pit.us_net import PitNet  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.cli import guard_xsh  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
SNAP = {"KR": "d0afe28c7c6013bbcc292bab9343f637", "US": "b9b116b96c483c8bd34b42706d3c2e16"}  # 85.3: the 84.6 data
CANDS = ("logit_w", "et", "lgbm_rank", "lgbm_clf", "xgb_rank", "cat_clf")
MIN_REV = {"KR": 1e10, "US": 1e7}  # a rival needs TTM revenue >= KRW 10bn / USD 10m
STALE_DAYS = 456  # a firm whose latest report ends more than ~15 months before t is not counted
LAG_YEAR = pd.Timedelta(days=365)
KR_MARKETS = ("유가증권시장상장법인", "코스닥시장상장법인")
KR_AVAIL_DAYS = {1: 60, 2: 60, 3: 60, 4: 100}  # quarterly/half-year reports +60 days, annual +100 (85.1)
US_FORMS = ("10-K", "10-Q", "10-KT", "10-QT")
US_QUARTERS = [f"{y}q{q}" for y in range(2010, 2027) for q in range(1, 5)][: (2026 - 2010) * 4 + 3]  # ..2026q3
US_BLANK_CHECK = "677"  # SIC 6770
# 85.4 (before any result): utilities tag their total as RegulatedAndUnregulatedOperatingRevenue; banks tag no total -
# their revenue is total interest and dividend income + noninterest income (both needed), used after the other rules
US_TOTALS = [*us_fund.REVENUE, "RegulatedAndUnregulatedOperatingRevenue"]
US_BANK = ["InterestAndDividendIncomeOperating", "NoninterestIncome"]
FEATURES = ("cm_rivals", "cm_share", "cm_rival_top", "cm_hhi", "cm_share_chg", "cm_rivals_chg", "cm_ind_growth")
DOCS = {
    "cm_rivals": ("같은 업종 경쟁사 수(log 1+n)", "listed rivals in the same industry (log 1+n)"),
    "cm_share": ("업종 안 매출 점유율", "revenue share within the industry"),
    "cm_rival_top": ("가장 큰 경쟁사의 점유율", "largest rival's revenue share"),
    "cm_hhi": ("업종 집중도(허핀달 지수)", "industry concentration (Herfindahl)"),
    "cm_share_chg": ("점유율 12개월 변화", "12-month change in revenue share"),
    "cm_rivals_chg": ("경쟁사 수 12개월 변화(진입·퇴출)", "12-month change in rival count (entry/exit)"),
    "cm_ind_growth": (
        "경쟁사 매출 성장(같은 회사들, 12개월, 로그)",
        "rivals' revenue growth (balanced, 12 months, log)",
    ),
}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def research_dir(settings) -> Path:
    p = Path(settings.cache_path) / "research" / "v85"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------------------------------------ peer records
# One row per (firm, report): firm, avail (first usable date), period_end, ind (industry code), ttm (revenue).


def _fx(ccy: str, root: Path, now: datetime) -> pd.Series | None:
    """KRW per unit of ``ccy``; Yahoo has no usable CNYKRW=X history, so fall back to the USD cross (KRW=X / CNY=X)."""
    s = kf.fx_history_yahoo(ccy, root, now)
    if s is not None and len(s) >= 250:
        return s
    import yfinance as yf

    def close(sym: str) -> pd.Series:
        h = yf.Ticker(sym).history(start="2013-01-01", auto_adjust=False)["Close"].astype(float)
        idx = pd.DatetimeIndex(h.index)
        h.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        return h[h > 0][~h.index.duplicated(keep="last")]

    cross = (close("KRW=X") / close(f"{ccy}=X")).dropna().sort_index()
    say(f"FX {ccy}KRW from the USD cross: {len(cross)} days")
    return cross if len(cross) >= 250 else None


def kr_records(settings) -> pd.DataFrame:
    root = kr_root(settings)
    paths = sorted((root / "dart_bulk" / "items").glob("*.pkl"))
    items = pd.concat([pd.read_pickle(p) for p in paths], ignore_index=True)
    items = items[items["market"].isin(KR_MARKETS) & items["period_end"].notna()]
    items = items.sort_values(["stock_code", "period_end", "bulk_period"])
    items = items.drop_duplicates(["stock_code", "period_end"], keep="last")
    notes: list[str] = []
    cov: dict = {}
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    items = kf.convert_currency(items, lambda ccy: _fx(ccy, root, now), notes, cov)
    q = kb.quarterly_standalone(items).sort_values(["stock_code", "period_end"]).reset_index(drop=True)
    g = q.groupby("stock_code", sort=False)
    ttm, ok = q["revenue_q"].copy(), q["revenue_q"].notna()
    for k in (1, 2, 3):
        prev = g["revenue_q"].shift(k)
        gap = (q["period_end"] - g["period_end"].shift(k)).dt.days
        ok &= prev.notna() & gap.between(80 * k, 100 * k)
        ttm = ttm + prev.fillna(0.0)
    q["ttm"] = ttm.where(ok)
    from quantcio.backtest.sectors import kr_bulk_industries

    ind = kr_bulk_industries(settings, DiskCache(settings.cache_path)).rename(columns={"code": "stock_code"})
    ind = ind[ind["ksic"].str.len() == 3].sort_values("period_end")
    q = pd.merge_asof(
        q.sort_values("period_end"), ind[["stock_code", "period_end", "ksic"]], on="period_end", by="stock_code"
    )
    q["avail"] = q["period_end"] + pd.to_timedelta(q["fq"].map(KR_AVAIL_DAYS), unit="D")
    rec = q.rename(columns={"stock_code": "firm", "ksic": "ind"})[["firm", "avail", "period_end", "ind", "ttm"]]
    say(f"KR peer records {len(rec):,} ({rec['firm'].nunique():,} firms, TTM {rec['ttm'].notna().mean():.0%}); {notes}")
    return rec


def _parse_fsds(body: bytes) -> pd.DataFrame:
    """Revenue rows (no dimensions, no co-registrant, USD, us-gaap) of every 10-K/10-Q in one quarterly data set."""
    z = zipfile.ZipFile(io.BytesIO(body))
    sub = pd.read_csv(
        z.open("sub.txt"),
        sep="\t",
        dtype=str,
        usecols=["adsh", "cik", "sic", "form", "period", "fy", "fp", "filed"],
        quoting=csv.QUOTE_NONE,
        encoding_errors="replace",
    )
    sub = sub[sub["form"].isin(US_FORMS)]
    keep = set(sub["adsh"])
    tags = set(US_TOTALS) | set(us_fund.REVENUE_PARTS) | set(US_BANK)
    rows = []
    with z.open("num.txt") as fh:
        ix = {c: i for i, c in enumerate(fh.readline().decode().rstrip("\r\n").split("\t"))}
        a, t, v, d, qt, u, co, val = (
            ix[c] for c in ("adsh", "tag", "version", "ddate", "qtrs", "uom", "coreg", "value")
        )
        seg = ix.get("segments")
        for line in fh:
            if (
                b"Revenue" not in line
                and b"InterestAndDividendIncomeOperating" not in line
                and b"NoninterestIncome" not in line
            ):
                continue
            p = line.decode("utf-8", "replace").rstrip("\r\n").split("\t")
            if p[t] not in tags or p[a] not in keep or p[u] != "USD" or p[co] or not p[v].startswith("us-gaap"):
                continue
            if (seg is not None and p[seg]) or not p[val]:
                continue
            rows.append((p[a], p[t], p[d], int(p[qt]), float(p[val])))
    num = pd.DataFrame(rows, columns=["adsh", "tag", "ddate", "qtrs", "value"])
    return num.merge(sub, on="adsh", how="inner")


def fetch_us(settings) -> None:
    net = PitNet(settings)
    out = research_dir(settings)
    for i, qn in enumerate(US_QUARTERS, 1):
        p = out / f"fsds_{qn}.pkl"
        if p.exists():
            continue
        t0 = time.time()
        body = net.cached_bytes(
            "sec",
            f"fsds/{qn}.zip",
            f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{qn}.zip",
            "sec",
            None,
        )
        df = _parse_fsds(body)
        df.to_pickle(p)
        say(
            f"{qn} ({i}/{len(US_QUARTERS)}): {len(body) / 1e6:.0f} MB, {len(df):,} revenue rows, "
            f"{df['adsh'].nunique():,} filings ({time.time() - t0:.0f}s)"
        )


def us_records(settings) -> pd.DataFrame:
    d = research_dir(settings)
    missing = [q for q in US_QUARTERS if not (d / f"fsds_{q}.pkl").exists()]
    if missing:
        raise SystemExit(f"run fetch-us first (missing {missing[:3]} ...)")
    r = pd.concat([pd.read_pickle(d / f"fsds_{q}.pkl") for q in US_QUARTERS], ignore_index=True)
    r = r.drop_duplicates(["adsh", "tag", "ddate", "qtrs", "value"])
    key = ["adsh", "ddate", "qtrs"]
    per_tag = r.groupby([*key, "tag"])["value"].max().reset_index()
    tot = per_tag[per_tag["tag"].isin(US_TOTALS)].groupby(key)["value"].max()
    parts = per_tag[per_tag["tag"].isin(us_fund.REVENUE_PARTS)].groupby(key)["value"].sum()
    bank = per_tag[per_tag["tag"].isin(US_BANK)].groupby(key)["value"].agg(["sum", "size"])
    bank = bank.loc[bank["size"].eq(len(US_BANK)), "sum"]
    rev = tot.combine_first(parts).combine_first(bank).rename("rev").reset_index()
    meta = r.drop_duplicates("adsh").set_index("adsh")[["cik", "sic", "form", "period", "fp", "filed"]]
    rev = rev.join(meta, on="adsh")
    for c in ("ddate", "period", "filed"):
        rev[c] = pd.to_datetime(rev[c], format="%Y%m%d", errors="coerce")
    rev["cik"] = rev["cik"].astype(int)
    lookup = dict(zip(zip(rev["adsh"], rev["ddate"], rev["qtrs"]), rev["rev"], strict=True))
    # annual values first reported in a 10-K (PIT: first filing date wins)
    ann = rev[rev["form"].isin(("10-K", "10-KT")) & rev["qtrs"].eq(4)].sort_values("filed")
    ann = ann.drop_duplicates(["cik", "ddate"], keep="first")
    annual = dict(zip(zip(ann["cik"], ann["ddate"]), zip(ann["filed"], ann["rev"]), strict=True))
    fil = rev.drop_duplicates("adsh")[["adsh", "cik", "sic", "form", "period", "fp", "filed"]].reset_index(drop=True)
    me = pd.offsets.MonthEnd(0)
    ttm = np.full(len(fil), np.nan)
    for i, f in enumerate(fil.itertuples(index=False)):
        if pd.isna(f.period):
            continue
        if f.form in ("10-K", "10-KT"):
            ttm[i] = lookup.get((f.adsh, f.period, 4), np.nan)
            continue
        qn = {"Q1": 1, "Q2": 2, "Q3": 3}.get(str(f.fp))
        if qn is None:
            continue
        cur = lookup.get((f.adsh, f.period, qn), np.nan)
        prior = lookup.get((f.adsh, f.period - pd.DateOffset(months=12) + me, qn), np.nan)
        fy = annual.get((f.cik, f.period - pd.DateOffset(months=3 * qn) + me))
        if fy is not None and fy[0] <= f.filed:
            ttm[i] = fy[1] + cur - prior
    fil["ttm"] = ttm
    fil.loc[fil["ttm"] <= 0, "ttm"] = np.nan
    sic = pd.to_numeric(fil["sic"], errors="coerce")
    fil["ind"] = sic.map(lambda x: f"{int(x):04d}"[:3] if pd.notna(x) and x > 0 else None)
    fil = fil[fil["ind"].ne(US_BLANK_CHECK)]
    fil["firm"] = fil["cik"].map(lambda c: f"CIK{int(c):010d}")
    fil["avail"] = fil["filed"] + pd.Timedelta(days=1)
    rec = fil.rename(columns={"period": "period_end"})[["firm", "avail", "period_end", "ind", "ttm"]]
    say(f"US peer records {len(rec):,} ({rec['firm'].nunique():,} firms, TTM {rec['ttm'].notna().mean():.0%})")
    return rec


# ------------------------------------------------------------------------------------------------ features


def snapshot(rec: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Latest valid report per firm usable at t: index firm, columns ind, ttm."""
    r = rec[rec["avail"] <= t]
    r = r.groupby("firm", sort=False).tail(1)
    r = r[r["period_end"] > t - pd.Timedelta(days=STALE_DAYS)]
    return r.set_index("firm")[["ind", "ttm"]]


def structure(snap: pd.DataFrame, min_rev: float) -> pd.DataFrame:
    """Per firm in ``snap``: rivals n, share, largest rival share, HHI (rivals = others with ttm >= min_rev)."""
    s = snap.copy()
    s["big"] = s["ttm"] >= min_rev
    b = s[s["big"]].assign(sq=lambda x: x["ttm"] ** 2)
    agg = b.groupby("ind").agg(n=("ttm", "size"), tot=("ttm", "sum"), sq=("sq", "sum"))
    ranked = b.reset_index().sort_values(["ind", "ttm"], ascending=[True, False])
    pos = ranked.groupby("ind").cumcount()
    first = ranked[pos.eq(0)].set_index("ind")
    agg["top1"] = first["ttm"]
    agg["top2"] = ranked[pos.eq(1)].set_index("ind")["ttm"]
    top1_firm = first["firm"]
    j = s.join(agg, on="ind")
    for c in ("n", "tot", "sq", "top1", "top2"):
        j[c] = j[c].fillna(0.0)
    own_big = j["big"].astype(float)
    tot = j["tot"] + (1 - own_big) * j["ttm"]
    sq = j["sq"] + (1 - own_big) * j["ttm"] ** 2
    is_top = j.index.to_series().eq(j["ind"].map(top1_firm)).to_numpy()
    rival_top = np.where(is_top, j["top2"], j["top1"])
    return pd.DataFrame(
        {
            "ind": j["ind"],
            "ttm": j["ttm"],
            "n": j["n"] - own_big,
            "share": j["ttm"] / tot,
            "rival_top": rival_top / tot,
            "hhi": sq / tot**2,
        },
        index=j.index,
    )


def growth(snap_t: pd.DataFrame, snap_p: pd.DataFrame, min_rev: float) -> pd.Series:
    """Per firm in snap_t: log growth of the summed revenue of its rivals at t that also have a report at t-1y."""
    s = snap_t.copy()
    s["prev"] = snap_p["ttm"].reindex(s.index)
    s["use"] = (s["ttm"] >= min_rev) & s["prev"].notna()
    u = s[s["use"]]
    agg = u.groupby("ind")[["ttm", "prev"]].sum()
    j = s.join(agg, on="ind", rsuffix="_ind")
    own = j["use"].astype(float)
    num = j["ttm_ind"].fillna(0) - own * j["ttm"]
    den = j["prev_ind"].fillna(0) - own * j["prev"].fillna(0)
    return pd.Series(
        np.where((num > 0) & (den > 0), np.log(num.clip(lower=1e-9) / den.clip(lower=1e-9)), np.nan), s.index
    )


def block(frame: pd.DataFrame, rec: pd.DataFrame, market: str) -> tuple[pd.DataFrame, dict]:
    """The seven cm_* columns aligned to ``frame`` rows."""
    min_rev = MIN_REV[market]
    rec = rec.dropna(subset=["avail", "period_end", "ind", "ttm"]).sort_values(["avail", "period_end"])
    firm_of = frame["entity"].astype(str).str.split("#").str[0]
    cache: dict[pd.Timestamp, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def at(t: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
        if t not in cache:
            sn = snapshot(rec, t)
            cache[t] = (sn, structure(sn, min_rev))
        return cache[t]

    parts = []
    dates = sorted(frame["date"].unique())
    for k, t in enumerate(dates):
        t = pd.Timestamp(t)
        sn, st = at(t)
        snp, stp = at(t - LAG_YEAR)
        firms = firm_of[frame["date"].eq(t)]
        cur = st.reindex(firms.to_numpy())
        prv = stp.reindex(firms.to_numpy())
        same = (cur["ind"] == prv["ind"]).to_numpy()
        g = growth(sn, snp, min_rev).reindex(firms.to_numpy()).to_numpy()
        parts.append(
            pd.DataFrame(
                {
                    "cm_rivals": np.log1p(cur["n"].to_numpy(float)),
                    "cm_share": cur["share"].to_numpy(float),
                    "cm_rival_top": cur["rival_top"].to_numpy(float),
                    "cm_hhi": cur["hhi"].to_numpy(float),
                    "cm_share_chg": np.where(same, cur["share"].to_numpy(float) - prv["share"].to_numpy(float), np.nan),
                    "cm_rivals_chg": np.where(
                        same, np.log1p(cur["n"].to_numpy(float)) - np.log1p(prv["n"].to_numpy(float)), np.nan
                    ),
                    "cm_ind_growth": g,
                },
                index=firms.index,
            )
        )
        if k % 100 == 0:
            say(f"  competition block {k + 1}/{len(dates)} dates")
    out = pd.concat(parts).reindex(frame.index)
    me = frame["is_me"].to_numpy()
    info = {
        "peer_firms": int(rec["firm"].nunique()),
        "coverage_me": {c: float(out.loc[me, c].notna().mean()) for c in FEATURES},
        "median_rivals_me": float(np.expm1(out.loc[me, "cm_rivals"]).median()),
        "median_share_me": float(out.loc[me, "cm_share"].median()),
        "median_hhi_me": float(out.loc[me, "cm_hhi"].median()),
    }
    return out, info


# ------------------------------------------------------------------------------------------------ study


def _per_date(res, name: str, part: str, cost: float) -> pd.Series:
    frame = res.oos if part == "outer" else res.holdout_preds
    return mt.per_date_primary(frame, name, cost) if frame is not None and name in frame else pd.Series(dtype=float)


def contribution(res_b, res_a, cost: float) -> dict:
    out: dict = {"per_model": {}}
    for part in ("outer", "holdout"):
        diffs = []
        for n in CANDS:
            a, b = _per_date(res_b, n, part, cost), _per_date(res_a, n, part, cost)
            common = a.index.intersection(b.index)
            if len(common):
                diffs.append((a.reindex(common) - b.reindex(common)).rename(n))
                out["per_model"].setdefault(n, {})[part] = mt.paired_difference(a, b)
        avg = pd.concat(diffs, axis=1).mean(axis=1) if diffs else pd.Series(dtype=float)
        out[f"mean_{part}"] = mt.summarize(avg.to_numpy(), 0.0)
    mo, mh = out["mean_outer"], out["mean_holdout"]
    out["helps"] = bool((mo["lo"] or -1) > 0 and (mh["mean"] is not None and mh["mean"] >= 0))
    return out


def _f(x, spec: str = "+.4f") -> str:
    return "–" if x is None or not np.isfinite(x) else format(x, spec)


def write_md(m: str, r: dict) -> Path:
    c, ca, cb = r["contribution"], r["arm_a"], r["arm_b"]
    lines = [
        f"# {m} 경쟁 구조(경쟁사 수·점유율·집중도) 시험 (docs/MODELING.md 85)",
        "",
        f"- 데이터셋 `{r['snapshot']}` (84.6 고정 자료), 비용 {r['cost_bps']:.0f}bp, 후보 {', '.join(CANDS)}",
        f"- 경쟁사 자료: {r['block']['peer_firms']:,}개 회사, 월말 채움 "
        + ", ".join(f"{k} {v:.0%}" for k, v in r["block"]["coverage_me"].items()),
        f"- 월말 중앙값: 경쟁사 {r['block']['median_rivals_me']:.0f}개, 점유율 {r['block']['median_share_me']:.1%}, "
        f"HHI {r['block']['median_hhi_me']:.3f}",
        f"- 결측 누출 의심으로 뺀 특징: {', '.join(r['leak_suspect']) or '없음'}",
        "",
        "## 기여 (B − A, 6개 모델 평균, 월 상위 20% 베타 조정 순초과)",
        "",
        f"- 외부 {_f(c['mean_outer']['mean'])} [{_f(c['mean_outer']['lo'])}, {_f(c['mean_outer']['hi'])}] "
        f"(t {_f(c['mean_outer']['t'], '.2f')}), 홀드아웃 {_f(c['mean_holdout']['mean'])} "
        f"[{_f(c['mean_holdout']['lo'])}, {_f(c['mean_holdout']['hi'])}]",
        f"- 판정: **{'도움' if c['helps'] else '도움 없음'}** (외부 하한 > 0 그리고 홀드아웃 ≥ 0)",
        "",
        "| 모델 | A 외부 | B 외부 | B − A 외부 [95%] | B − A 홀드아웃 |",
        "|---|---|---|---|---|",
    ]
    for n in CANDS:
        pm = c["per_model"].get(n, {})
        o, h = pm.get("outer", {}), pm.get("holdout", {})
        lines.append(
            f"| `{n}` | {_f(ca[n])} | {_f(cb[n])} | {_f(o.get('mean'))} [{_f(o.get('lo'))}, {_f(o.get('hi'))}] "
            f"| {_f(h.get('mean'))} |"
        )
    lines += [
        "",
        f"- 각 갈래의 선정(6개 + 모멘텀 안에서, 참고용): A `{r['selected_a']}`, B `{r['selected_b']}`",
        "",
        "## 특징별 단변량 (홀드아웃 전 월말, 순위 IC)",
        "",
        "| 특징 | 설명 | IC 평균 | t | 결측 AUC |",
        "|---|---|---|---|---|",
    ]
    for k in FEATURES:
        u = r["univariate"].get(k, {})
        lines.append(
            f"| `{k}` | {DOCS[k][0]} | {_f(u.get('mean'))} | {_f(u.get('t'), '.2f')} | "
            f"{_f(r['missing_auc'].get(feat.x_col(k)), '.4f')} |"
        )
    p = OUT / f"{m}_v85.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def study(m: str) -> None:
    s = load_settings()
    cache = DiskCache(s.cache_path)
    ds = cache.get("modeling_dataset", SNAP[m], None)
    if ds is None:
        raise SystemExit(f"dataset snapshot {SNAP[m]} is not cached")
    ds = guard_xsh(ds, s, say)
    rec = kr_records(s) if m == "KR" else us_records(s)
    blk, info = block(ds.frame, rec, m)
    say(f"block: {info}")
    cands = [c for c in cmod.default_candidates() if c.name in CANDS]
    snap0, xs0 = ds.snapshot, list(ds.xs_cols)
    ds.snapshot = f"{snap0}|v85A"
    res_a = sel.run_selection(ds, s, cache, candidates=cands, threads=8, use_cache=True, progress=say)
    f = ds.frame
    new = []
    for c in FEATURES:
        f[c] = blk[c].to_numpy(float)
        f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
        feat.FEATURE_DOCS[c] = DOCS[c]
        new.append(feat.x_col(c))
    aucs = {x: v13s.missing_auc(f, x[2:]) for x in new}
    sus = sorted(x for x, a in aucs.items() if a is not None and abs(a - 0.5) > v13s.LEAK_AUC)
    new = [x for x in new if x not in set(sus)]
    ds.xs_cols = [*xs0, *new]
    ds.snapshot = f"{snap0}|v85B|{len(new)}"
    say(f"arm B: +{len(new)} features (leak suspects dropped: {sus})")
    res_b = sel.run_selection(ds, s, cache, candidates=cands, threads=8, use_cache=True, progress=say)
    cost = float(ds.cost_bps)
    da, db = res_a.to_dict(), res_b.to_dict()
    hold0 = pd.Timestamp(db["protocol"]["holdout"][0])
    me = f[f["is_me"] & f["labelled"] & (f["date"] < hold0)]
    uni = {c: mt.summarize(mt.per_date_ic(me, c)) for c in FEATURES}
    prim = sel.PRIMARY
    r = {
        "market": m,
        "snapshot": snap0,
        "cost_bps": cost,
        "block": info,
        "missing_auc": aucs,
        "leak_suspect": sus,
        "univariate": uni,
        "contribution": contribution(res_b, res_a, cost),
        "arm_a": {c["name"]: c["outer"][prim]["mean"] for c in da["candidates"]},
        "arm_b": {c["name"]: c["outer"][prim]["mean"] for c in db["candidates"]},
        "selected_a": da["selected"],
        "selected_b": db["selected"],
        "gate_b": db["gate"],
    }
    js = OUT / f"{m}_v85.json"
    js.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    say(f"wrote {write_md(m, r)} and {js}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch-us", "study"])
    ap.add_argument("--market", choices=["KR", "US"])
    a = ap.parse_args()
    if a.cmd == "fetch-us":
        fetch_us(load_settings())
    else:
        if not a.market:
            raise SystemExit("--market is required")
        study(a.market)


if __name__ == "__main__":
    main()
