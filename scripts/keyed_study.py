"""Research run with the API-key data (docs/MODELING.md section 23, pre-declared 2026-10-04): block K = whatever the
set keys made available (``scripts/api_keys.py fetch`` first). Research only: the served models and ``src/quantcio``
are not changed.

    .venv/bin/python scripts/keyed_study.py --market US
    .venv/bin/python scripts/keyed_study.py --market KR

Writes reports/modeling/<MKT>_keyed.{md,json} and reports/modeling/<MKT>_keyed_summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.dataset import build_dataset  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
BASE = ("enet_logit", "logit_w", "lgbm_clf", "et")
DOCS = {
    "us_pat_12m": ("특허 등록 수(최근 12개월, log)", "patent grants, last 12 months (log)"),
    "us_pat_growth": ("특허 등록 증가(12개월 vs 직전 12개월)", "patent grant growth"),
    "us_beta_dgs2": ("2년 금리 민감도(252일, SPY 통제)", "2-year yield sensitivity"),
    "us_impact_dgs2": ("2년 금리 충격 전달(민감도 × 최근 20일 변화)", "2-year yield shock pass-through"),
    "us_impact_ngstor": (
        "천연가스 저장량 서프라이즈 전달(천연가스 민감도 × 서프라이즈)",
        "gas-storage surprise pass-through",
    ),
    "us_sec_openings": ("업종 구인율 변화(JOLTS, 3개월−24개월)", "sector job-openings-rate change"),
    "kr_pat_1y": ("특허 등록 수(전년, log)", "patent registrations, last year (log)"),
    "kr_pat_growth": ("특허 등록 증가(전년 vs 전전년)", "patent registration growth"),
    "kr_sec_bsi": ("업종 기업경기전망(BSI, 36개월 대비 z)", "sector business-survey outlook z"),
    "kr_sec_hire": ("업종 입직률 변화(사업체노동력조사)", "sector hiring-rate change"),
    "kr_sec_sep": ("업종 이직률 변화(사업체노동력조사)", "sector separation-rate change"),
}
RAW = {"us_impact_dgs2", "us_impact_ngstor", "us_sec_openings", "kr_sec_bsi", "kr_sec_hire", "kr_sec_sep"}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ helpers


def _betas_at(rows: pd.DataFrame, prices: dict, bench: pd.Series, factors: pd.DataFrame, lag_sessions: dict) -> dict:
    """Per-row partial betas of each factor (us_v7_data.partial_betas), shifted by the factor's publication lag."""
    import us_v7_data as v7

    out = {c: np.full(len(rows), np.nan) for c in factors.columns}
    pos_in = {ix: i for i, ix in enumerate(rows.index)}
    for ent, g in rows.groupby("entity", sort=False):
        px = prices.get(ent)
        if px is None or px.empty or "close" not in px:
            continue
        b = v7.partial_betas(px["close"].astype(float), bench, factors.reindex(px.index))
        bi = pd.DatetimeIndex(b.index).to_numpy()
        gd = g["date"].to_numpy(dtype="datetime64[ns]")
        p = np.searchsorted(bi, gd, "right") - 1
        fresh = (p >= 0) & ((gd - bi[np.clip(p, 0, None)]) <= np.timedelta64(7, "D"))
        rows_ix = np.array([pos_in[i] for i in g.index])
        for c in factors.columns:
            v = b[c].shift(lag_sessions.get(c, 0)).to_numpy(float)[np.clip(p, 0, None)]
            v[~fresh] = np.nan
            out[c][rows_ix] = v
    return out


def _shock(level: pd.Series, sessions: pd.DatetimeIndex, n: int = 20) -> pd.Series:
    lv = level.reindex(sessions, method="ffill")
    ch = lv - lv.shift(n)
    sd = ch.rolling(756, min_periods=252).std()
    return (ch / sd.where(sd > 0)).clip(-3, 3) / 3.0


def _monthly_window(counts: pd.Series, dates: pd.DatetimeIndex, months: int, offset: int) -> np.ndarray:
    """Sum of monthly counts over complete months [m - offset - months, m - offset) before each date's month m."""
    if counts.empty:
        return np.full(len(dates), np.nan)
    idx = pd.period_range(counts.index.min().to_period("M"), max(dates.max(), counts.index.max()).to_period("M"))
    c = counts.groupby(counts.index.to_period("M")).sum().reindex(idx, fill_value=0.0)
    cs = np.concatenate([[0.0], np.cumsum(c.to_numpy(float))])
    pos = {p: i for i, p in enumerate(idx)}
    out = np.full(len(dates), np.nan)
    for i, d in enumerate(dates):
        m = pos.get(d.to_period("M"))
        if m is None:
            continue
        hi, lo = m - offset, m - offset - months
        if lo >= 0:
            out[i] = cs[hi] - cs[lo]
    return out


# ------------------------------------------------------------------------------------------------ blocks


def us_block(ds, settings, cache, universe=None) -> tuple[list[str], list[str], dict]:
    from datetime import date as _date

    from quantcio.data.pit import load_pit_universe

    f = ds.frame
    dates = pd.DatetimeIndex(f["date"])
    xs, reg, info = [], [], {}
    uni = universe or load_pit_universe(
        Market.US, settings, cache, 500, _date.fromisoformat(settings.backtest.us_start)
    )
    bench = uni.benchmarks["SPY"]["close"].astype(float)
    sessions = pd.DatetimeIndex(bench.index)
    # patents (USPTO)
    if (kd.root(settings, "uspto") / "patents.parquet").exists():
        from us_v8_data import cik_chains

        from quantcio.data.pit.us_cik import Submissions
        from quantcio.data.pit.us_net import PitNet

        chains = cik_chains(ds, settings, cache)
        subs = Submissions(PitNet(settings))
        ciks = sorted({c for cs in chains.values() for c in cs})
        monthly = kd.uspto_monthly_by_company(settings, {c: subs.names(c) for c in ciks})
        p12, prev = np.full(len(f), np.nan), np.full(len(f), np.nan)
        for e, g in f.groupby("entity", sort=False):
            m = monthly[monthly["cik"].isin(chains.get(str(e), set()))]
            if m.empty:
                continue
            s = m.groupby("month")["grants"].sum()
            s.index = pd.DatetimeIndex(s.index)
            ix = f.index.get_indexer(g.index)
            dd = pd.DatetimeIndex(g["date"])
            p12[ix] = _monthly_window(s, dd, 12, 0)
            prev[ix] = _monthly_window(s, dd, 12, 12)
        f["us_pat_12m"] = np.log1p(p12)
        f["us_pat_growth"] = np.log1p(p12) - np.log1p(prev)
        xs += ["us_pat_12m", "us_pat_growth"]
        info["patent_matched_ciks"] = int(monthly["cik"].nunique())
    # FRED
    dgs2 = kd.fred_series(settings, "dgs2")
    if dgs2 is not None:
        lvl = dgs2.set_index("date")["value"].sort_index()
        fac = pd.DataFrame({"dgs2": lvl.reindex(sessions).diff()})
        b = _betas_at(f[["date", "entity"]], uni.prices, bench, fac, {"dgs2": 2})
        f["us_beta_dgs2"] = b["dgs2"]
        sh = _shock(kd._daily_available(lvl, dgs2.set_index("date")["available"], 10), sessions)
        f["us_impact_dgs2"] = feat.rank_gauss(f["us_beta_dgs2"], f["date"]).to_numpy(float) * _asof(sh, dates)
        xs += ["us_beta_dgs2", "us_impact_dgs2"]
    for name, col, fn in (
        ("icsa", "r_icsa", lambda s: np.log(s.rolling(4, min_periods=4).mean()).diff(52)),
        ("nfci", "r_nfci", lambda s: s),
        ("umcsent", "r_umcsent", lambda s: kd._z_past(s, 36, 24)),
        ("t10y2y", "r_t10y2y", lambda s: s),
    ):
        df = kd.fred_series(settings, name)
        if df is not None:
            df = df.sort_values("date")
            f[col] = _asof(
                kd._daily_available(fn(df["value"].reset_index(drop=True)), df["available"].reset_index(drop=True), 45),
                dates,
            )
            reg.append(col)
    # EIA
    z = kd.eia_surprise(settings)
    if len(z):
        import us_v7_data as v7

        ng = v7.yahoo_closes(settings, cache)["natgas"]
        b = _betas_at(f[["date", "entity"]], uni.prices, bench, pd.DataFrame({"ng": ng.reindex(sessions).diff()}), {})
        f["r_ng_stor"] = _asof(z, dates)
        f["us_impact_ngstor"] = (
            feat.rank_gauss(pd.Series(b["ng"], index=f.index), f["date"]).to_numpy(float)
            * np.clip(f["r_ng_stor"], -3, 3)
            / 3
        )
        xs.append("us_impact_ngstor")
        reg.append("r_ng_stor")
    # BLS job openings (v2 cache contains "JO")
    from us_v8_data import JOLTS_INDUSTRY
    from us_v8_data import root_dir as v8_root

    jp = v8_root(settings) / "jolts.json"
    if jp.exists() and any(str(s["seriesID"]).endswith("JOR") for s in json.loads(jp.read_text())):
        recs = {}
        for s in json.loads(jp.read_text()):
            sid = s["seriesID"]
            if sid.endswith("JOR"):
                vals = {
                    pd.Timestamp(int(d["year"]), int(d["period"][1:]), 1): pd.to_numeric(d["value"], errors="coerce")
                    for d in s["data"]
                    if d["period"].startswith("M") and d["period"] != "M13"
                }
                recs[sid[3:9]] = pd.Series(vals).sort_index()
        col = np.full(len(f), np.nan)
        sec = f["sector_group"].astype(str).to_numpy()
        for sname, ind in JOLTS_INDUSTRY.items():
            s = recs.get(ind)
            if s is None:
                continue
            x = (s.rolling(3, min_periods=3).mean() - s.rolling(24, min_periods=18).mean()).dropna()
            ser = kd._daily_available(x, pd.Series(x.index + pd.offsets.MonthBegin(2) + pd.Timedelta(14, "D")), 62)
            m = sec == sname
            col[m] = _asof(ser, dates[m])
        f["us_sec_openings"] = col
        xs.append("us_sec_openings")
    return xs, reg, info


def kr_block(ds, settings, cache) -> tuple[list[str], list[str], dict]:
    f = ds.frame
    dates = pd.DatetimeIndex(f["date"])
    sec = f["sector_group"].astype(str).to_numpy()
    xs, reg, info = [], [], {}
    for name, s in kd.ecos_regime(settings).items():
        f[name] = _asof(s, dates)
        reg.append(name)
    for col, data in (("kr_sec_bsi", kd.ecos_sector_bsi(settings)),):
        if data:
            v = np.full(len(f), np.nan)
            for sname, s in data.items():
                m = sec == sname
                v[m] = _asof(s, dates[m])
            f[col] = v
            xs.append(col)
    flows = kd.kosis_sector_flows(settings)
    if flows:
        for kind in ("hire", "sep"):
            v = np.full(len(f), np.nan)
            for sname, d in flows.items():
                m = sec == sname
                if kind in d and m.any():
                    v[m] = _asof(d[kind], dates[m])
            f[f"kr_sec_{kind}"] = v
            xs.append(f"kr_sec_{kind}")
    counts = kd.kipris_counts(settings)
    if counts:
        ents = f["entity"].astype(str).to_numpy()
        yrs = dates.year.to_numpy()
        c1, c2 = np.full(len(f), np.nan), np.full(len(f), np.nan)
        for i, (e, y) in enumerate(zip(ents, yrs, strict=True)):
            c = counts.get(e)
            if c and (y - 1) in c:
                c1[i] = c[y - 1]
                if (y - 2) in c:
                    c2[i] = c[y - 2]
        f["kr_pat_1y"] = np.log1p(c1)
        f["kr_pat_growth"] = np.log1p(c1) - np.log1p(c2)
        xs += ["kr_pat_1y", "kr_pat_growth"]
        info["kipris_entities"] = len(counts)
    return xs, reg, info


def _asof(series: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    return kd._asof(series, dates)


# ------------------------------------------------------------------------------------------------ run


def candidates() -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    out = [every["mom_12_1"], every["factor_ic"], *(every[n] for n in BASE)]
    for name in ("logit_w_core", "lgbm_clf_core"):
        d = every[name]
        d.label_ko = d.label_ko.replace("v2 특징만", "키 데이터 묶음 제외")
        d.label_en = d.label_en.replace("v2 inputs only", "without the API-key block")
        out.append(d)
    return out


def main() -> None:
    from quantcio.modeling.report import render_markdown

    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["US", "KR"], required=True)
    mkt = ap.parse_args().market
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    sel.DIAGNOSTIC_PAIRS = (("logit_w", "logit_w_core"), ("lgbm_clf", "lgbm_clf_core"))
    say(f"{mkt} modeling dataset")
    ds = build_dataset(Market(mkt), settings, cache, progress=say)
    core_xs, core_reg = list(ds.xs_cols), list(ds.regime_cols)
    raw, reg, info = (us_block if mkt == "US" else kr_block)(ds, settings, cache)
    if not raw and not reg:
        raise SystemExit("키 데이터가 없습니다: .env에 키를 넣고 `scripts/api_keys.py fetch`를 먼저 실행하세요")
    f = ds.frame
    for c in raw:
        f[feat.x_col(c)] = f[c].clip(-3, 3) if c in RAW else feat.rank_gauss(f[c], f["date"])
        feat.FEATURE_DOCS[c] = DOCS.get(c, (c, c))
    ds.xs_cols = [*core_xs, *[feat.x_col(c) for c in raw]]
    ds.regime_cols = [*core_reg, *reg]
    ds.core_xs_cols, ds.core_regime_cols = core_xs, core_reg
    ds.snapshot = f"{ds.snapshot}|keyed|{'+'.join(sorted([*raw, *reg]))}"
    ds.alt_raw = [*list(ds.alt_raw or []), *raw]
    lab = f[f["labelled"] & f["is_me"]]
    info["missing_share"] = {c: round(float(lab[c].isna().mean()), 3) for c in [*raw, *reg]}
    say(f"block K: {raw} + regime {reg}; {json.dumps(info, ensure_ascii=False)}")
    res = sel.run_selection(ds, settings, cache, candidates=candidates(), threads=8, progress=say)
    res.protocol["keyed"] = {"declared": "docs/MODELING.md 23", "features": raw, "regime": reg, **info}
    stem = OUT / f"{mkt}_keyed"
    stem.with_suffix(".json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        f"# 종목 선택 모델 선정 — {mkt}", f"# {mkt} 키 데이터 연구 (서빙 아님) — {mkt}", 1
    )
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    lines = [
        f"# {mkt} 키 데이터 연구 요약 (docs/MODELING.md 23)",
        "",
        f"- 묶음 K: {', '.join(raw) or '없음'} · 국면: {', '.join(reg) or '없음'}",
        f"- 선정: `{res.selected}` · 게이트 {'통과' if res.gate['passed'] else '미달'}",
        "",
        "| 후보 | 외부 AUC [95% CI] | 홀드아웃 AUC |",
        "|---|---|---|",
    ]
    for c in res.candidates:
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        hm = f"{h['mean']:.4f}" if h.get("mean") is not None else "–"
        lines.append(f"| {c['name']} | {o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] | {hm} |")
    lines += ["", "## 묶음 K의 기여 (같은 모델에서 묶음 K만 뺀 쌍과의 날짜별 AUC 차이)", ""]
    for k, v in (res.protocol.get("alt_value") or {}).items():
        o = v["outer"]
        lines.append(f"- {k}: 외부 {o['mean']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}] (t {o['t']:+.2f})")
    (OUT / f"{mkt}_keyed_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
