"""US v4 stock-selection research (docs/MODELING.md section 18, pre-declared 2026-10-04).

Phase ``select``: the cached US v3 modeling dataset + the v4 features (``us_v4_data``) + sector-neutral training
labels; the pre-declared candidates (logit_w, enet_logit, lgbm_clf, et, cat_rank x {standard, sector-neutral label},
ensemble top 3, baselines, two v3-input diagnostics) through the unchanged protocol (``modeling.selection``).
Writes reports/modeling/US_v4_<as_of>.{md,json}.

Phase ``sp400`` (run once, after ``select``): the selected model, refitted per context on S&P 500 rows only, scores
every S&P 400 PIT row (a SECOND look at that sample - docs/MODELING.md 18.5). Writes reports/modeling/US_v4_sp400.md.

Research only: nothing here changes the served model (``modeling_serving`` cache) or ``src/quantcio``.
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
import us_v4_data as v4  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.data.pit import load_pit_fundamentals, load_pit_universe  # noqa: E402
from quantcio.data.pit.us_cik import Submissions, entity_cik  # noqa: E402
from quantcio.data.pit.us_net import PitNet  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.dataset import build_dataset  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
BASE = ("logit_w", "enet_logit", "lgbm_clf", "et", "cat_rank")
SEC_LABELS = ("y_e_sec", "y_dm_sec", "y_cls_sec", "y_q_sec")
MIN_SECTOR = 5
STATE = PROJECT_ROOT / ".cache" / "quantcio" / "research" / "us_v4_state.json"


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ features


def add_v4_features(ds, settings, cache, universe=None, fundamentals=None):
    """Raw + rank-gauss v4 features on ``ds.frame`` (in place). Returns coverage info."""
    f = ds.frame
    uni = universe or load_pit_universe(Market.US, settings, cache, 500, date.fromisoformat(settings.backtest.us_start))
    ents = sorted(f["entity"].astype(str).unique())
    fund = fundamentals or load_pit_fundamentals(Market.US, settings, cache, ents)
    net = PitNet(settings)
    # every CIK of the entity's predecessor chain (e.g. Exxon 34088 -> 2115436): facts and 8-Ks of all of them
    chains: dict[str, set[int]] = {e: {entity_cik(e)} for e in ents}
    if fund.panel is not None and "cik_chain" in fund.panel:
        for e, ch in fund.panel[["entity", "cik_chain"]].dropna().itertuples(index=False):
            if e in chains:
                chains[e].update(int(x) for x in str(ch).split(",") if x.strip().isdigit())
    all_ciks = sorted({c for cs in chains.values() for c in cs})
    say(f"companyfacts for {len(all_ciks)} CIKs ({len(ents)} entities)")
    facts = v4.fetch_facts(net, all_ciks, progress=say)

    def merged(e: str) -> dict:
        out: dict[str, list] = {}
        for c in chains[e]:
            for concept, lst in ((facts.get(c) or {}).get("facts") or {}).items():
                out.setdefault(concept, []).extend(lst)
        return out

    tables = {e: v4.quality_tables(merged(e)) for e in ents}
    say("8-K item 2.02 reaction days")
    bench = uni.benchmarks["SPY"]["close"].astype(float)
    sessions = pd.DatetimeIndex(bench.index)
    subs = Submissions(net)
    start = pd.Timestamp(f["date"].min()) - pd.Timedelta(800, "D")
    reactions = {e: sorted({d for c in chains[e] for d in v4.reaction_days(subs, c, sessions, start)}) for e in ents}
    rows = f[["date", "entity", "mcap"]]
    say("event features")
    ev = v4.event_features(rows, reactions, uni.prices, bench, sessions)
    say("surprise features")
    su = v4.surprise_features(rows, v4.surprise_table(fund.panel))
    say("quality features")
    qu = v4.quality_features(rows, tables)
    raw = pd.concat([ev, su, qu], axis=1)[v4.FEATURES]
    for c in v4.FEATURES:
        f[c] = raw[c].to_numpy()
        f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
        feat.FEATURE_DOCS[c] = v4.DOCS[c]
    lab = f[f["labelled"] & f["is_me"]]
    cov = {
        "missing_share": {c: round(float(f[c].isna().mean()), 3) for c in v4.FEATURES},
        "by_year_share_known": lab.groupby(lab["date"].dt.year)[v4.FEATURES]
        .apply(lambda g: g.notna().mean().round(2))
        .to_dict(orient="index"),
        "entities_without_facts": sum(1 for e in ents if not merged(e)),
        "ciks_without_reactions": sum(1 for e in ents if not reactions.get(e)),
    }
    return cov


def add_sector_labels(f: pd.DataFrame) -> None:
    """Sector-neutral training labels (docs/MODELING.md 18.3); a sector with < MIN_SECTOR labelled names on a date
    falls back to the whole date."""
    lab = f["labelled"].to_numpy()
    y = f["y_e"].where(lab)
    key = [f["date"], f["sector_group"]]
    n_sec = y.notna().groupby(key).transform("sum")
    sec_mean = y.groupby(key).transform("mean")
    sec_med = y.groupby(key).transform("median")
    all_mean = y.groupby(f["date"]).transform("mean")
    all_med = y.groupby(f["date"]).transform("median")
    use = n_sec >= MIN_SECTOR
    mean = sec_mean.where(use, all_mean)
    med = sec_med.where(use, all_med)
    f["y_e_sec"] = (y - mean).where(lab)
    f["y_dm_sec"] = f["y_e_sec"]
    f["y_cls_sec"] = (y > med).astype(float).where(lab)
    u_sec = y.groupby(key).rank(pct=True)
    u_all = y.groupby(f["date"]).rank(pct=True)
    u = u_sec.where(use, u_all)
    f["y_q_sec"] = np.floor(u * 5 - 1e-9).clip(0, 4).where(lab)


# ------------------------------------------------------------------------------------------------ candidates


class DataV4(sel._Data):
    """``selection._Data`` that also keeps the sector-neutral label columns."""

    def __init__(self, ds):
        super().__init__(ds)
        for c in SEC_LABELS:
            if c in ds.frame:
                self.f[c] = ds.frame.loc[self.f.index, c].to_numpy()


def _sector_fitter(base_fitter):
    def fit(cand, train, params, scheme, ctx):
        tr = train.assign(y_e=train["y_e_sec"], y_dm=train["y_dm_sec"], y_cls=train["y_cls_sec"], y_q=train["y_q_sec"])
        return base_fitter(cand, tr, params, scheme, ctx)

    return fit


def v4_candidates() -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    out = [every["mom_12_1"], every["factor_ic"]]
    for name in BASE:
        c = every[name]
        out.append(c)
        s = copy.copy(c)
        s.name = f"{name}_sec"
        s.fitter = f"{name}__sec"
        s.label_ko = c.label_ko + " · 섹터 중립 레이블"
        s.label_en = c.label_en + " · sector-neutral label"
        cmod._FITTERS[s.fitter] = _sector_fitter(cmod._FITTERS[c.fitter or c.name])
        out.append(s)
    out += [every["logit_w_core"], every["lgbm_clf_core"]]  # diagnostics: v3 inputs (core = the v3 xs columns)
    for d in out[-2:]:
        d.label_ko = d.label_ko.replace("v2 특징만", "v3 특징만")
        d.label_en = d.label_en.replace("v2 inputs only", "v3 inputs only")
    return out


def patch_protocol() -> None:
    """Only the pre-declared composite (ensemble = inner top 3) and the v4 data view."""
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    sel._Data = DataV4
    sel.DIAGNOSTIC_PAIRS = (("logit_w", "logit_w_core"), ("lgbm_clf", "lgbm_clf_core"))


# ------------------------------------------------------------------------------------------------ phases


def prepare(settings, cache):
    say("US modeling dataset (v3)")
    ds = build_dataset(Market.US, settings, cache, progress=say)
    v3_xs, v3_reg = list(ds.xs_cols), list(ds.regime_cols)
    cov = add_v4_features(ds, settings, cache)
    add_sector_labels(ds.frame)
    ds.xs_cols = [*v3_xs, *[feat.x_col(c) for c in v4.FEATURES]]
    ds.core_xs_cols, ds.core_regime_cols = v3_xs, v3_reg
    ds.snapshot = f"{ds.snapshot}|us_v4|{len(v4.FEATURES)}"
    ds.alt_raw = [*list(ds.alt_raw or []), *v4.FEATURES]
    return ds, cov


def phase_select(settings, cache) -> None:
    from quantcio.modeling.report import render_markdown

    ds, cov = prepare(settings, cache)
    say(f"v4 coverage: {json.dumps(cov['missing_share'])}")
    patch_protocol()
    res = sel.run_selection(ds, settings, cache, candidates=v4_candidates(), threads=8, progress=say)
    res.protocol["us_v4"] = {"declared": "docs/MODELING.md 18", "features": v4.FEATURES, "coverage": cov}
    stem = OUT / f"US_v4_{res.as_of}"
    stem.with_suffix(".json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        "# 종목 선택 모델 선정 — US", "# 종목 선택 모델 연구 US v4 (서빙 아님) — US", 1
    )
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"selected": res.selected, "as_of": res.as_of, "json": str(stem.with_suffix(".json"))}))
    say(f"selected {res.selected}: wrote {stem}.md")


def phase_sp400(settings, cache) -> None:
    """Second look at the S&P 400 rows (docs/MODELING.md 18.5) with the selected v4 model."""
    from quantcio.data.alt.load import load_alt_data
    from quantcio.data.pit import load_pit_xsh_universe
    from quantcio.xsh import _align_regime

    state = json.loads(STATE.read_text())
    result = json.loads(Path(state["json"]).read_text())
    selected = state["selected"]
    patch_protocol()
    ds, _ = prepare(settings, cache)
    data = sel._Data(ds)
    outer, hold, _final, _ = sel.make_contexts(data)
    say("S&P 400 universe")
    uni = load_pit_xsh_universe(Market.US, settings, cache, date.fromisoformat(settings.backtest.us_start), None)
    ents = sorted(uni.membership["entity"].astype(str).unique())
    alt = load_alt_data(Market.US, settings, ents, date(2011, 1, 1), progress=say)
    xds = build_dataset(Market.US, settings, cache, universe=uni, weekly=False, alt=alt, progress=say)
    fund = load_pit_fundamentals(Market.US, settings, cache, ents)
    add_v4_features(xds, settings, cache, universe=uni, fundamentals=fund)
    xf, _ = _align_regime(xds, ds)
    pf = (uni.coverage or {}).get("pit_from")
    if pf:  # month-ends before pit_from are survivorship-biased (data/pit/us_midcap.py)
        xf = xf[xf["date"] >= pd.Timestamp(pf)]
    cands = {c.name: c for c in v4_candidates()}
    cand_rows = {c["name"]: c for c in result["candidates"]}
    scored = []
    for ctx in [*outer, hold]:
        rows = xf[xf["labelled"] & xf["is_me"] & xf["date"].isin(ctx.test_dates)]
        if rows.empty:
            continue
        cfg = (cand_rows[selected]["configs"] or {}).get(ctx.name) or {}
        members = cfg.get("members") or [selected]
        preds = rows[["date"]].copy()
        for m in members:
            mc = (cand_rows[m]["configs"] or {}).get(ctx.name) or {}
            cand = cands[m]
            tr = data.train(ctx.train_end, cand.rows)
            fitted = cand.fit(
                tr,
                dict(mc.get("params") or cand.grid[0]),
                str(mc.get("scheme") or "none"),
                sel._ctx(data, ctx.test_dates[0], settings, 8),
            )
            preds[m] = fitted.predict(rows)
        score = preds[members[0]] if len(members) == 1 else sel.rank_mean(preds, members)
        scored.append(rows[["date", "entity", "y_e", "r", feat.x_col("mom_12_1")]].assign(score=score.to_numpy()))
        say(f"sp400 {ctx.name}: {len(rows)} rows")
    frame = pd.concat(scored, ignore_index=True)
    m_model = mt.evaluate(frame, 10.0, "score")
    m_mom = mt.evaluate(frame.assign(mom=frame[feat.x_col("mom_12_1")]), 10.0, "mom")
    auc = m_model[mt.PRIMARY_AUC]
    ok = bool(auc["lo"] > 0.5 and (m_model["ic"]["t"] or 0) >= 2)
    lines = [
        "# US v4 — S&P 400 두 번째 평가 (docs/MODELING.md 18.5)",
        "",
        f"- 선정 모델 `{selected}` (S&P 500 행으로만 학습, 문맥별 재적합) · 행 {len(frame):,} · "
        f"날짜 {frame['date'].nunique()} · 종목 {frame['entity'].nunique()}",
        f"- 날짜 내 AUC(중앙값 초과) {auc['mean']:.4f} [{auc['lo']:.4f}, {auc['hi']:.4f}] · "
        f"순위 IC {m_model['ic']['mean']:+.4f} (t {m_model['ic']['t']:+.2f}) · "
        f"12-1 모멘텀 AUC {m_mom[mt.PRIMARY_AUC]['mean']:.4f}",
        f"- 판정: {'중형주에서 재현' if ok else '재현되지 않음'} (기준: AUC 95% 하한 > 0.5 그리고 IC t ≥ 2)",
        "- 주의: 이 표본은 2026-09-27 XSH-US에서 v2 모델로 이미 한 번 봤습니다(완전히 미열람은 아님).",
        "- 표본은 'S&P 500에 한 번도 들지 않은 S&P 400 종목'이라 나중에 승격된(크게 오른) 종목이 빠집니다"
        "(미래 정보 조건).",
        f"- 평가 시작: {pf or '전체'} (시점별 구성이 온전한 달부터)",
    ]
    (OUT / "US_v4_sp400.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "US_v4_sp400.json").write_text(
        json.dumps({"selected": selected, "model": m_model, "momentum": m_mom, "reproduced": ok}, default=str, indent=1)
    )
    say("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["select", "sp400"])
    args = ap.parse_args()
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    if args.phase == "select":
        phase_select(settings, cache)
    else:
        phase_sp400(settings, cache)


if __name__ == "__main__":
    main()
