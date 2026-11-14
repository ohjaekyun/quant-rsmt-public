"""US v11 research run (docs/MODELING.md section 29, pre-declared 2026-10-05): the v9 inputs + block N (8-K filing
times, 10-K text drift, naive distance to default, Wikipedia spikes, Tranco domain ranks), with diagnostic pairs that
drop only block N. Research only.

    .venv/bin/python scripts/us_v11_study.py --fetch-only   # Wikidata links, Tranco ranks, 10-K sections (SEC)
    .venv/bin/python scripts/us_v11_study.py                # the run

Writes reports/modeling/US_v11.{md,json} and reports/modeling/US_v11_summary.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention_data as att  # noqa: E402
import keyed_data as kd  # noqa: E402
import us_v10_data as v10  # noqa: E402
import us_v11_data as n  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.dataset import build_dataset  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
BASE = ("cat_rank", "ridge", "logit_w", "lgbm_clf", "et")
PAIRS = ("cat_rank", "logit_w", "lgbm_clf")
RAW = {"fil_dump_10"}  # 0/1 flag: used as is


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _asof(feats: pd.DataFrame, cols: list[str], d: np.ndarray, max_age_days: int) -> np.ndarray:
    if feats is None or feats.empty:
        return np.full((len(d), len(cols)), np.nan)
    idx = feats.index.to_numpy(dtype="datetime64[ns]") if not isinstance(feats.index, pd.RangeIndex) else None
    if idx is None:
        idx = feats["avail"].to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(idx, d, "right") - 1
    ok = (pos >= 0) & ((d - idx[np.clip(pos, 0, None)]) <= np.timedelta64(max_age_days, "D"))
    v = feats[cols].to_numpy(float)[np.clip(pos, 0, None)]
    v[~ok] = np.nan
    return v


def fetch(ds, settings, cache) -> dict:
    import us_v8_data as v8

    from quantcio.data.pit.us_cik import Submissions
    from quantcio.data.pit.us_net import PitNet

    chains = v8.cik_chains(ds, settings, cache)
    ciks = sorted({c for cs in chains.values() for c in cs})
    links = n.company_links(settings, ciks, say)
    doms = {v["domain"] for v in links.values() if v.get("domain")}
    out = {"links": len(links), "domains": len(doms), "tranco": n.fetch_tranco(settings, doms, say)}
    net = PitNet(settings)
    out["sections"] = n.fetch_sections(net, Submissions(net), ciks, kd.root(settings, "v11"), say)
    return out


def add_block_n(ds, settings, cache) -> tuple[list[str], dict]:
    import us_v8_data as v8
    import us_v9_features as v9f

    from quantcio.data.pit.us_cik import Submissions, entity_cik
    from quantcio.data.pit.us_net import PitNet

    f = ds.frame
    chains = v8.cik_chains(ds, settings, cache)
    net = PitNet(settings)
    subs = Submissions(net)
    links = n.company_links(settings, sorted({c for cs in chains.values() for c in cs}))
    arts = att.wiki_articles(settings, sorted({c for cs in chains.values() for c in cs}))
    facts = v10.fetch_facts_m(net, sorted({c for cs in chains.values() for c in cs}))
    root = kd.root(settings, "v11")
    res = {c: np.full(len(f), np.nan) for c in n.FEATURES}
    debt = np.full(len(f), np.nan)
    say("block N: filings / text / web / debt")
    for e, g in f.groupby("entity", sort=False):
        ix = f.index.get_indexer(g.index)
        d = g["date"].to_numpy(dtype="datetime64[ns]")
        chain = chains.get(str(e), {entity_cik(str(e))})
        fl = pd.concat([n.filing_times(subs, c) for c in chain], ignore_index=True).sort_values("when")
        fv = n.filing_features(d, fl)
        res["fil_dump_90"][ix], res["fil_dump_10"][ix], res["fil_after_90"][ix] = fv[:, 0], fv[:, 1], fv[:, 2]
        tt = n.text_table(root, chain)
        if len(tt):
            tv = _asof(tt, ["sim_1a", "sim_7", "len_1a"], d, 500)
            res["txt_sim_1a"][ix], res["txt_sim_7"][ix], res["txt_len_1a"][ix] = tv[:, 0], tv[:, 1], tv[:, 2]
        own = entity_cik(str(e))
        cands = [own, *sorted(chain - {own})]
        art = next((arts[c] for c in cands if c in arts), None)
        if art:
            ws = n.wiki_spikes(att.wiki_series(settings, art))
            wv = _asof(ws, ["wiki_spike_1", "wiki_spike_3"], d, 5)
            res["wiki_spike_1"][ix], res["wiki_spike_3"][ix] = wv[:, 0], wv[:, 1]
        dom = next((links[c]["domain"] for c in cands if links.get(c, {}).get("domain")), "")
        if dom:
            tr = n.tranco_features(settings, dom)
            rv = _asof(tr, ["web_rank_4w", "web_rank_12w"], d, 14)
            res["web_rank_4w"][ix], res["web_rank_12w"][ix] = rv[:, 0], rv[:, 1]
        merged: dict[str, list] = {}
        for c in chain:
            for k, v in ((facts.get(c) or {}).get("facts") or {}).items():
                merged.setdefault(k, []).extend(v)
        at = v10.annual_table(merged) if merged else pd.DataFrame()
        if len(at):
            at = at.assign(
                avail=at["filed"] + pd.Timedelta(1, "D"), debt=at["liab_current"] + 0.5 * at["lt_debt"].fillna(0)
            )
            at = at.sort_values("avail").reset_index(drop=True)
            debt[ix] = _asof(at.set_index("avail"), ["debt"], d, 500)[:, 0]
    ret_1y = (1 + f["mom_12_1"]) * (1 + f["ret_1m"]) - 1 if "mom_12_1" in f else pd.Series(np.nan, index=f.index)
    f["dd"] = n.naive_dd(f["mcap"].to_numpy(float), debt, f["vol60"].to_numpy(float), ret_1y.to_numpy(float))
    past = v9f._past(f, ["dd"], 28)
    f["dd_chg"] = f["dd"] - past["dd"]
    gap = feat.rank_gauss(f["dd_chg"], f["date"]) - feat.rank_gauss(f["ret_1m"], f["date"])
    res["dd"], res["dd_chg"], res["dd_mom_gap"] = (
        f["dd"].to_numpy(float),
        f["dd_chg"].to_numpy(float),
        gap.to_numpy(float),
    )
    for c in n.FEATURES:
        f[c] = res[c]
        f[feat.x_col(c)] = f[c] if c in RAW else feat.rank_gauss(f[c], f["date"])
        feat.FEATURE_DOCS[c] = n.DOCS[c]
    lab = f[f["labelled"] & f["is_me"]]
    info = {
        "missing_share": {c: round(float(lab[c].isna().mean()), 3) for c in n.FEATURES},
        "known_by_year": lab.groupby(lab["date"].dt.year)[["txt_sim_1a", "dd", "wiki_spike_1", "web_rank_4w"]]
        .apply(lambda g: g.notna().mean().round(2))
        .to_dict(orient="index"),
    }
    return [feat.x_col(c) for c in n.FEATURES], info


def candidates() -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    out = [every["mom_12_1"], every["factor_ic"], *(every[x] for x in BASE)]
    for name in PAIRS:
        d = copy.copy(every[name])
        d.name, d.diagnostic, d.feature_set = f"{name}_noN", True, "core"
        d.fitter = every[name].fitter or every[name].name
        d.label_ko = every[name].label_ko + " · 진단: 묶음 N 제외"
        out.append(d)
    return out


def write_summary() -> str:
    d = json.loads((OUT / "US_v11.json").read_text())
    info = d["protocol"]["us_v11"]
    lines = [
        "# 미국 v11 요약 (docs/MODELING.md 29)",
        "",
        f"- 묶음 N 결측률: {info['missing_share']}",
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
    lines += ["", "## 묶음 N의 기여 (같은 모델에서 묶음 N만 뺀 쌍과의 날짜별 AUC 차이)", ""]
    for k, v in (d["protocol"].get("alt_value") or {}).items():
        o, h = v["outer"], v.get("holdout") or {}
        hh = f" · 홀드아웃 {h['mean']:+.4f} [{h['lo']:+.4f}, {h['hi']:+.4f}]" if h.get("mean") is not None else ""
        lines.append(f"- {k}: 외부 {o['mean']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}] (t {o['t']:+.2f}){hh}")
    lines += ["", "## 선정 모델 중요도 상위 20", ""]
    lines += [f"- {k}: {v:.4f}" for k, v in list(d["importances"].items())[:20]]
    text = "\n".join(lines) + "\n"
    (OUT / "US_v11_summary.md").write_text(text, encoding="utf-8")
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
    nx, info = add_block_n(ds, settings, cache)
    ds.core_xs_cols, ds.core_regime_cols = base_xs, base_reg
    ds.xs_cols = [*base_xs, *nx]
    ds.snapshot = f"{ds.snapshot}|us_v11|{len(nx)}"
    say(f"block N missing {info['missing_share']}")
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    sel.DIAGNOSTIC_PAIRS = tuple((x, f"{x}_noN") for x in PAIRS)
    res = sel.run_selection(ds, settings, cache, candidates=candidates(), threads=8, progress=say)
    res.protocol["us_v11"] = {"declared": "docs/MODELING.md 29", **info}
    (OUT / "US_v11.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        "# 종목 선택 모델 선정 — US", "# 미국 v11 연구 — 공시 행태·문장 변화·신용위험 대용 등 (서빙 아님) — US", 1
    )
    (OUT / "US_v11.md").write_text(md, encoding="utf-8")
    print(write_summary())


if __name__ == "__main__":
    main()
