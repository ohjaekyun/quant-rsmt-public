"""US v10 research run (docs/MODELING.md section 28, pre-declared 2026-10-05): the v9 inputs + block M (analyst
changes, FINRA short interest, off-exchange share, shareholder yield, F-score, operating accruals, peer lead-lag) and
the CBOE equity put/call regime value, with diagnostic pairs that drop only block M. Research only.

    .venv/bin/python scripts/us_v10_study.py --fetch-only   # downloads (analysts, short interest, XBRL, CBOE)
    .venv/bin/python scripts/us_v10_study.py                # the run

Writes reports/modeling/US_v10.{md,json} and reports/modeling/US_v10_summary.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import us_v10_data as m  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.data.pit import load_pit_universe  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.dataset import build_dataset  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
BASE = ("cat_rank", "ridge", "logit_w", "lgbm_clf", "et")
PAIRS = ("cat_rank", "logit_w", "lgbm_clf")


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch(ds, settings, cache) -> dict:
    import us_v8_data as v8

    from quantcio.data.pit.us_net import PitNet

    tickers = sorted(set(ds.frame["ticker"].astype(str)))
    out = {"analyst": m.fetch_analyst(settings, tickers, say), "si": m.fetch_short_interest(settings, tickers, say)}
    chains = v8.cik_chains(ds, settings, cache)
    m.fetch_facts_m(PitNet(settings), sorted({c for cs in chains.values() for c in cs}), say)
    m.fetch_cboe(settings, say)
    return out


def add_block_m(ds, settings, cache, universe=None) -> tuple[list[str], list[str], dict]:
    import us_v5_data as v5
    import us_v8_data as v8

    from quantcio.data.pit.us_cik import Submissions, entity_cik
    from quantcio.data.pit.us_net import PitNet

    f = ds.frame
    uni = universe or load_pit_universe(Market.US, settings, cache, 500, date.fromisoformat(settings.backtest.us_start))
    res = {c: np.full(len(f), np.nan) for c in m.FEATURES}
    ent_tickers = f.groupby("entity")["ticker"].agg(lambda s: set(s.astype(str))).to_dict()
    chains = v8.cik_chains(ds, settings, cache)
    facts = m.fetch_facts_m(PitNet(settings), sorted({c for cs in chains.values() for c in cs}))
    subs = Submissions(PitNet(settings))
    say("block M: analysts / short interest / fundamentals")
    for e, g in f.groupby("entity", sort=False):
        ix = f.index.get_indexer(g.index)
        d = g["date"].to_numpy(dtype="datetime64[ns]")
        tk = ent_tickers.get(e, set())
        an = m.analyst_features(d, m.analyst_events(settings, tk))
        for j, c in enumerate(("an_rev_30", "an_rev_90", "an_pt_30", "an_pt_90", "an_activity")):
            res[c][ix] = an[:, j]
        si = m.short_interest(settings, tk)
        if len(si):
            a = si["avail"].to_numpy(dtype="datetime64[ns]")
            pos = np.searchsorted(a, d, "right") - 1
            ok = (pos >= 0) & ((d - a[np.clip(pos, 0, None)]) <= np.timedelta64(45, "D"))
            p = np.clip(pos, 0, None)
            shares = (g["mcap"] / g["close"]).to_numpy(float)
            res["si_dtc"][ix] = np.where(ok, si["dtc"].to_numpy(float)[p], np.nan)
            res["si_chg"][ix] = np.where(
                ok, np.log(si["si"].to_numpy(float)[p] / si["si_prev"].replace(0, np.nan).to_numpy(float)[p]), np.nan
            )
            res["si_float"][ix] = np.where(ok, si["si"].to_numpy(float)[p] / shares, np.nan)
        merged: dict[str, list] = {}
        for c in chains.get(str(e), {entity_cik(str(e))}):
            for k, v in ((facts.get(c) or {}).get("facts") or {}).items():
                merged.setdefault(k, []).extend(v)
        ff = m.fundamental_features(m.annual_table(merged)) if merged else pd.DataFrame()
        if len(ff):
            ff = ff.sort_values("avail").reset_index(drop=True)
            a = ff["avail"].to_numpy(dtype="datetime64[ns]")
            pos = np.searchsorted(a, d, "right") - 1
            ok = (pos >= 0) & ((d - a[np.clip(pos, 0, None)]) <= np.timedelta64(500, "D"))
            p = np.clip(pos, 0, None)
            res["sh_yield"][ix] = np.where(ok, ff["payout"].to_numpy(float)[p] / g["mcap"].to_numpy(float), np.nan)
            res["f_score"][ix] = np.where(ok, ff["f_score"].to_numpy(float)[p], np.nan)
            res["op_accruals"][ix] = np.where(ok, ff["op_accruals"].to_numpy(float)[p], np.nan)
    say("block M: off-exchange share")
    syms = set(f["ticker"].map(v5.norm_symbol))
    fin = v5.finra_panel(settings.cache_path, syms)
    fin_by = {s: g.set_index("date")["total"] for s, g in fin.groupby("symbol")}
    for e, g in f.groupby("entity", sort=False):
        px = uni.prices.get(e)
        if px is None or px.empty:
            continue
        vol = px["volume"].astype(float)
        ix = f.index.get_indexer(g.index)
        tot = pd.Series(0.0, index=vol.index)
        for t in {v5.norm_symbol(x) for x in ent_tickers.get(e, set())}:
            s = fin_by.get(t)
            if s is not None:
                tot = tot.add(s.reindex(vol.index).fillna(0.0), fill_value=0.0)
        tot = tot.where(tot.index >= v5.FINRA_START)
        sh20 = tot.rolling(20, min_periods=10).sum() / vol.rolling(20, min_periods=10).sum()
        sh120 = tot.rolling(120, min_periods=60).sum() / vol.rolling(120, min_periods=60).sum()
        sh = pd.DataFrame({"a": sh20.shift(1), "b": (sh20 - sh120).shift(1)})  # sessions strictly before t
        d = g["date"].to_numpy(dtype="datetime64[ns]")
        idx = sh.index.to_numpy(dtype="datetime64[ns]")
        pos = np.searchsorted(idx, d, "right") - 1
        p = np.clip(pos, 0, None)
        ok = pos >= 0
        res["offex_20"][ix] = np.where(ok, sh["a"].to_numpy(float)[p], np.nan)
        res["offex_chg"][ix] = np.where(ok, sh["b"].to_numpy(float)[p], np.nan)
    say("block M: peer lead-lag")
    sic = {}
    for e in f["entity"].unique():
        sub = subs.get(entity_cik(str(e))) or {}
        s3 = str(sub.get("sic") or "")[:3]
        sic[e] = s3 if s3.isdigit() else None
    rets = {}
    for e in f["entity"].unique():
        px = uni.prices.get(e)
        if px is not None and not px.empty:
            c = px["close"].astype(float)
            rets[e] = pd.DataFrame({"r5": c / c.shift(5) - 1, "r10": c / c.shift(10) - 1})
    rr = np.full((len(f), 2), np.nan)
    for e, g in f.groupby("entity", sort=False):
        r = rets.get(e)
        if r is None:
            continue
        idx = r.index.to_numpy(dtype="datetime64[ns]")
        d = g["date"].to_numpy(dtype="datetime64[ns]")
        pos = np.searchsorted(idx, d, "right") - 1
        ok = (pos >= 0) & ((d - idx[np.clip(pos, 0, None)]) <= np.timedelta64(7, "D"))
        v = r.to_numpy(float)[np.clip(pos, 0, None)]
        v[~ok] = np.nan
        rr[f.index.get_indexer(g.index)] = v
    f["_r5"], f["_r10"], f["_sic"] = rr[:, 0], rr[:, 1], f["entity"].map(sic)
    for _d, g in f[f["_sic"].notna()].groupby(["date", "_sic"]):
        if len(g) < 2:
            continue
        top = g.sort_values("mcap", ascending=False).head(4)
        r5, r10 = top["_r5"], top["_r10"]
        for i in g.index:
            keep = top.index != i
            lead5, lead10 = r5[keep].head(3), r10[keep].head(3)
            if len(lead10):
                k = f.index.get_loc(i)
                res["peer_lead_5"][k] = lead5.mean()
                res["peer_lead_10"][k] = lead10.mean()
                own = f.at[i, "_r10"]
                res["peer_gap_10"][k] = own - lead10.mean() if np.isfinite(own) else np.nan
    f.drop(columns=["_r5", "_r10", "_sic"], inplace=True)
    for c in m.FEATURES:
        f[c] = res[c]
        f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
        feat.FEATURE_DOCS[c] = m.DOCS[c]
    z = m.cboe_regime(m.fetch_cboe(settings, say))
    f["r_cboe_pc"] = z.reindex(pd.DatetimeIndex(f["date"]), method="ffill").to_numpy(float)
    lab = f[f["labelled"] & f["is_me"]]
    info = {
        "missing_share": {c: round(float(lab[c].isna().mean()), 3) for c in [*m.FEATURES, *m.REGIME]},
        "known_by_year": lab.groupby(lab["date"].dt.year)[
            ["an_rev_90", "si_dtc", "offex_20", "f_score", "peer_lead_10"]
        ]
        .apply(lambda g: g.notna().mean().round(2))
        .to_dict(orient="index"),
    }
    return [feat.x_col(c) for c in m.FEATURES], list(m.REGIME), info


def candidates() -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    out = [every["mom_12_1"], every["factor_ic"], *(every[n] for n in BASE)]
    for name in PAIRS:
        d = copy.copy(every[name])
        d.name, d.diagnostic, d.feature_set = f"{name}_noM", True, "core"
        d.fitter = every[name].fitter or every[name].name
        d.label_ko = every[name].label_ko + " · 진단: 묶음 M 제외"
        out.append(d)
    return out


def write_summary() -> str:
    d = json.loads((OUT / "US_v10.json").read_text())
    info = d["protocol"]["us_v10"]
    lines = [
        "# 미국 v10 요약 (docs/MODELING.md 28)",
        "",
        f"- 묶음 M 결측률: {info['missing_share']}",
        f"- 선정 `{d['selected']}` · 게이트 {'통과' if d['gate']['passed'] else '미달'}",
        "",
        "| 후보 | 외부 AUC [95% CI] | 외부 IC (t) | 홀드아웃 AUC |",
        "|---|---|---|---|",
    ]
    for c in d["candidates"]:
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        hm = f"{h['mean']:.4f}" if h.get("mean") is not None else "–"
        lines.append(
            f"| {c['name']} | {o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] | "
            f"{c['outer']['ic']['mean']:+.4f} ({c['outer']['ic']['t']:+.2f}) | {hm} |"
        )
    lines += ["", "## 묶음 M의 기여 (같은 모델에서 묶음 M만 뺀 쌍과의 날짜별 AUC 차이)", ""]
    for k, v in (d["protocol"].get("alt_value") or {}).items():
        o, h = v["outer"], v.get("holdout") or {}
        hh = f" · 홀드아웃 {h['mean']:+.4f} [{h['lo']:+.4f}, {h['hi']:+.4f}]" if h.get("mean") is not None else ""
        lines.append(f"- {k}: 외부 {o['mean']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}] (t {o['t']:+.2f}){hh}")
    lines += ["", "## 선정 모델 중요도 상위 20", ""]
    lines += [f"- {k}: {v:.4f}" for k, v in list(d["importances"].items())[:20]]
    text = "\n".join(lines) + "\n"
    (OUT / "US_v10_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    from quantcio.modeling.report import render_markdown

    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        print(write_summary())
        return
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    if a.fetch_only:
        ds = build_dataset(Market.US, settings, cache, progress=say)
        say(f"fetch {fetch(ds, settings, cache)}")
        return
    import us_v9_study as v9

    ds, _ = v9.build_inputs(settings, cache)
    base_xs, base_reg = list(ds.xs_cols), list(ds.regime_cols)
    mx, mreg, info = add_block_m(ds, settings, cache)
    ds.core_xs_cols, ds.core_regime_cols = base_xs, base_reg
    ds.xs_cols, ds.regime_cols = [*base_xs, *mx], [*base_reg, *mreg]
    ds.snapshot = f"{ds.snapshot}|us_v10|{len(mx)}"
    say(f"block M missing {info['missing_share']}")
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    sel.DIAGNOSTIC_PAIRS = tuple((n, f"{n}_noM") for n in PAIRS)
    res = sel.run_selection(ds, settings, cache, candidates=candidates(), threads=8, progress=say)
    res.protocol["us_v10"] = {"declared": "docs/MODELING.md 28", **info}
    (OUT / "US_v10.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        "# 종목 선택 모델 선정 — US", "# 미국 v10 연구 — 애널리스트·공매도 잔고·장외 거래 등 (서빙 아님) — US", 1
    )
    (OUT / "US_v10.md").write_text(md, encoding="utf-8")
    print(write_summary())


if __name__ == "__main__":
    main()
