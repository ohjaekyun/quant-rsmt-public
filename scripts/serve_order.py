"""Build a REFERENCE ventile artifact for the report (docs/MODELING.md 93, 94.3; user requests 2026-11-06 / 11-07).
Research / operations script: the served stock-selection model, the signal and the sizing are not changed.

    .venv/bin/python scripts/serve_order.py --market KR --model lgbm_reg_w_ord   # 91's ordering model
    .venv/bin/python scripts/serve_order.py --market KR --model rf               # 94's random forest
    .venv/bin/python scripts/serve_order.py --market KR --model rf_bo            # 94.2's tuned random forest
    .venv/bin/python scripts/serve_order.py --market KR --model rf_bo_3m         # 99's 3-month rf_bo, held 3 months
    .venv/bin/python scripts/serve_order.py --market KR --drop rf                # stop showing a model

``rf_bo_3m`` (docs/MODELING.md 102): the cached 99 run on 60-session labels; its final fit scores that dataset's latest
month-end; the book statistics hold each month-end D20 cohort 3 months (overlapping thirds, 99's ``Books``) on the
1-month 99 run's 20-session returns.

Re-uses a cached selection run on the 84.6 KR data (``use_cache``, no refit; the script stops before any fit): the
model's final-context fit (every row whose label is known at the latest session) and its scores on the latest PIT
cross-section become a ``ServingArtifact`` under ``quantcio.modeling.serve.ORDER_NS`` (one per model, the index
keeps the display order); the report reads them with ``order_for_report``. Its metrics carry the post hoc book
statistics of that run's OOS predictions (outer + holdout month-ends, 91.5's rules) printed next to the list.
Writes reports/modeling/<M>_order_latest.{md,csv} for every stored model (D20 = top 5%).

The models were chosen after their results were seen and are not confirmed: each artifact says so in its label.
A newer cross-section needs a new dataset snapshot and new runs (the frozen 84.6 data ends 2026-10-30).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v86_full_es as v86  # noqa: E402
import v88_ventile as v88  # noqa: E402
import v91_order_weekly as v91  # noqa: E402
import v99_horizons as v99  # noqa: E402

from quantcio.config import load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.cli import guard_xsh  # noqa: E402
from quantcio.modeling.serve import (  # noqa: E402
    ORDER_K,
    build_artifact,
    load_order_artifact,
    order_models,
    remove_order_artifact,
    save_order_artifact,
    ventile_of,
)

MODELS = {
    "lgbm_reg_w_ord": {
        "source": "v91",
        "doc": "91.5, 93",
        "label_ko": "20분위 순서 모델(주간 회귀 나무 + 순서 손실로 나무 수 선택) · 참고용, 사후 선택·미확인",
        "label_en": "20-quantile ordering model (weekly regression tree, tree count by ordering loss) · reference, "
        "post hoc",
        "short_ko": "LightGBM 순서 모델",
        "short_en": "LightGBM ordering model",
        "abbr_ko": "순서",
        "abbr_en": "Ord",
        "caveat_ko": "추천이 아닙니다: 결과를 본 뒤 고른 미확인 모델이고, 성과가 소수의 급등 달에 몰려 있으며 D20은 "
        "고변동·고베타 종목이 모여 손실 폭이 큽니다(docs/MODELING.md 91.5·93)",
        "caveat_en": "Not a recommendation: the model was chosen after its results were seen and is unconfirmed; the "
        "gains sit in a few surge months and D20 gathers high-volatility, high-beta names with deep drawdowns "
        "(docs/MODELING.md 91.5, 93)",
    },
    "rf": {
        "source": "v86A",
        "doc": "94",
        "label_ko": "랜덤포레스트(rank-gauss 목표) · 참고용, 사후 관찰·미확인",
        "label_en": "random forest (rank-gauss target) · reference, post hoc",
        "short_ko": "랜덤포레스트(기본)",
        "short_en": "Random forest (base)",
        "abbr_ko": "기본 RF",
        "abbr_en": "RF base",
        "caveat_ko": "추천이 아닙니다: 29개 모델 중 결과를 본 뒤 고른 관찰 대상이고, 앞 {n_outer}개월의 초과수익은 "
        "유의하지 않으며(t {t_outer:+.1f}) 성과 대부분이 확인 구간({h0}~{h1}, t {t_holdout:+.1f})에 몰려 있습니다. "
        "D20은 소형·거래가 적은 종목이라 실제 비용이 더 클 수 있습니다(docs/MODELING.md 94)",
        "caveat_en": "Not a recommendation: picked for watching after the results of 29 models were seen; its excess "
        "over the first {n_outer} months is not significant (t {t_outer:+.1f}) and most of the gain sits in the "
        "holdout {h0}..{h1} (t {t_holdout:+.1f}); D20 gathers small, thinly traded names whose real costs can exceed "
        "the book's (docs/MODELING.md 94)",
    },
    "rf_bo": {
        "source": "served1029",
        "rescore": True,  # the 10-29 run's final fit scores the frozen 84.6 data's month-end (10-30) cross-section
        "doc": "94.2, 96",
        "label_ko": "랜덤포레스트 + 정규화 튜닝(베이즈 최적화, 1-SE) · 참고용, 사후 관찰·미확인",
        "label_en": "random forest + regularisation tuning (Bayesian, 1-SE) · reference, post hoc",
        "short_ko": "랜덤포레스트(튜닝) · 1개월 모델",
        "short_en": "Random forest (tuned) · 1-month model",
        "abbr_ko": "1개월 RF",
        "abbr_en": "RF 1m",
        "caveat_ko": "추천이 아닙니다: 약 45개 모델의 결과를 본 뒤 고른 관찰 대상이고"
        "(앞 {n_outer}개월 t {t_outer:+.1f}, 확인 구간 {h0}~{h1} t {t_holdout:+.1f}) 앞으로의 성과로 확인되지 "
        "않았습니다. 최고 10개월을 빼면 연 {drop10}, 최대 낙폭 {mdd}입니다(docs/MODELING.md 94.2·96)",
        "caveat_en": "Not a recommendation: picked for watching after the results of about 45 models were seen (first "
        "{n_outer} months t {t_outer:+.1f}, holdout {h0}..{h1} t {t_holdout:+.1f}) and not confirmed by later months; "
        "without its best 10 months {drop10}/yr, max drawdown {mdd} (docs/MODELING.md 94.2, 96)",
    },
    "rf_bo_3m": {
        "source": "v99_3m",
        "candidate": "rf_bo",  # the run's candidate; stored under this artifact name
        "month_end": True,  # score the dataset's latest month-end cross-section (the books' formation date)
        "hold": 3,
        "doc": "99, 101, 102",
        "label_ko": "3개월 예측 랜덤포레스트 + 정규화 튜닝 · 3개월 보유 · 참고용, 사후 선택",
        "label_en": "3-month random forest + regularisation tuning · held 3 months · reference, post hoc pick",
        "short_ko": "랜덤포레스트(튜닝) · 3개월 모델, 3개월 보유",
        "short_en": "Random forest (tuned) · 3-month model, held 3 months",
        "abbr_ko": "3개월 RF",
        "abbr_en": "RF 3m",
        "caveat_ko": "추천이 아닙니다: 1·3·6·12개월 예측 모델 중 비용 후 성과로 고른 참고 모델입니다(사후 선택). "
        "매달 말 D20을 사서 3개월 보유하므로(매달 1/3씩 교체) 한 달 교체는 약 {turn}입니다. 처음 보는 2005~2015년에서 "
        "사전 판정으로 확인됐지만(연 {fresh}, t {t_fresh:+.1f}), 같은 기간 보유한 1개월 모델보다 낫다는 근거는 "
        "없고 장점은 낮은 거래비용입니다. 최고 10개월을 빼면 연 {drop10}, 최대 낙폭 {mdd}입니다"
        "(docs/MODELING.md 99·101·102)",
        "caveat_en": "Not a recommendation: picked among the 1/3/6/12-month models by after-cost results (post hoc). "
        "Each month-end D20 is held 3 months (a third replaced monthly), so about {turn} of the book turns over a "
        "month. Confirmed by a pre-declared rule on the unseen 2005-2015 window ({fresh}/yr, t {t_fresh:+.1f}), but "
        "not shown to beat the 1-month model held as long: the gain is lower trading cost. Without its best 10 months "
        "{drop10}/yr, max drawdown {mdd} (docs/MODELING.md 99, 101, 102)",
    },
}
SERVED_1029 = "755b5321fa66a3ffeb95200ae6c0e632"  # the served KR run's dataset (docs 94.2)
FRESH_101 = v86.OUT / "KR_v101_ef.json"  # 101.3's fresh-window rule for the 3-month model


def load_run(m: str, source: str, s, cache):
    """The cached selection run of ``source`` on the frozen 84.6 data (99's 60-session data for ``v99_3m``); raises
    before any fit."""
    if source == "v99_3m":
        return v99.run(3, no_fit=True)  # stops if a context is not cached
    ds = v86.load_frozen(m, s, cache)
    snap0 = ds.snapshot
    if source == "v91":
        v88.patch_selection()
        cands = v91.candidates()
        v91.LOG[:] = [v91.research_dir(s, m) / "choice_serve.jsonl"]
        tag = f"{snap0}|v91|{v91.V91_VERSION}"
    elif source == "v86A":
        v86.patch_selection()
        cands = v86.candidates()
        v86.ES_LOG[:] = [v86.research_dir(s, m) / "es_log_serve.jsonl"]
        v86.ARM[0] = "A"
        tag = f"{snap0}|v86A|es{v86.ES_VERSION}"
    elif source == "served1029":
        ds = guard_xsh(cache.get("modeling_dataset", SERVED_1029, None), s, v86.say)
        snap0 = ds.snapshot
        cands = cmod.default_candidates()
        tag = snap0
        _check_served(ds, s, cache, cands)
    else:
        raise SystemExit(f"unknown source {source}")

    def guard(msg: str) -> None:
        if "train <=" in msg or ("rsmt" in msg.lower() and "reused" not in msg):
            raise SystemExit(f"{source}: a context is not cached ({msg}) - not refitting here")

    ds.snapshot = tag
    try:
        res = sel.run_selection(ds, s, cache, candidates=cands, threads=4, use_cache=True, progress=guard)
    finally:
        ds.snapshot = snap0
    return ds, res


def _check_served(ds, s, cache, cands) -> None:
    """Every context (and RSMT) checkpoint of the served run is on disk - else stop before run_selection fits."""
    names = [c.name for c in cands]
    base = [n for n in names if n not in sel.STACKED]
    data = sel._Data(ds)
    outer, hold, final, _ = sel.make_contexts(data)
    fp, fpr = sel._fingerprint(), sel._fingerprint(rsmt=True)
    for ctx in [*outer, hold, final]:
        ident = [ds.snapshot, ctx.name, str(ctx.train_end), [str(x) for x in ctx.test_dates]]
        r = cache.get(sel.CTX_NS, cache.key(sel.CTX_NS, fp, *ident, base), None)
        if r is None:
            raise SystemExit(f"served run: no {ctx.name} checkpoint")
        for sn in (n for n in sel.STACKED if n in names):
            sb = r.stack_base
            rk = cache.key(sel.RSMT_NS, fpr, *ident, base, sb, r.configs.get(sb) if sb else None, sn)
            if not cache._path(sel.RSMT_NS, rk).exists():
                raise SystemExit(f"served run: no {ctx.name} {sn} checkpoint")


def rescore(res, name: str, ds_run, ds_new) -> None:
    """Score ``ds_new``'s latest cross-section with the run's final fit of ``name`` (in place of the run's own
    latest session); first re-predict the run's latest rows - they must equal its stored final predictions."""
    fit = res.final.fitted[name]
    if list(ds_new.xs_cols) != list(ds_run.xs_cols):
        raise SystemExit("rescore: the two datasets have different inputs")
    old = sel._Data(ds_run).latest_rows()
    ref = res.final.predictions.set_index("entity")[name].reindex(old["entity"]).to_numpy(dtype=float)
    if not np.allclose(fit.predict(old), ref, rtol=1e-9, atol=1e-12, equal_nan=True):
        raise SystemExit("rescore: re-predicting the run's latest rows does not reproduce its final predictions")
    new = sel._Data(ds_new).latest_rows()
    res.final.predictions = pd.DataFrame({"entity": new["entity"].to_numpy(), name: fit.predict(new)})
    print(f"rescored {name}: {ds_run.as_of.date()} -> {ds_new.as_of.date()} ({len(new)} rows)")


def _series(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Per month-end: the top / bottom ventile and the universe mean simple return R = e^r - 1 (equal weight), the
    top book's one-way turnover (91.5's ``bin_series``)."""
    d = frame[["date", "entity", name, "y_e", "r"]].dropna(subset=[name, "y_e", "r"])
    d = d[d.groupby("date")["y_e"].transform("size").to_numpy() >= mt.MIN_NAMES]
    d = d[d.groupby("date")[name].transform("nunique").to_numpy() > 1]
    g = d.groupby("date")
    u = (g[name].rank(method="average") - 0.5) / g[name].transform("size")
    d = d.assign(bin=np.clip(np.floor(u * ORDER_K), 0, ORDER_K - 1).astype(int) + 1, R=np.expm1(d["r"].to_numpy(float)))
    dates = sorted(d["date"].unique())
    top, bot = d[d["bin"] == ORDER_K], d[d["bin"] == 1]
    books = top.groupby("date")["entity"].apply(set).reindex(dates)
    turn, prev = [], None
    for x in books:
        if not isinstance(x, set) or not x:
            turn.append(np.nan)
            prev = None
            continue
        turn.append(1.0 if prev is None else len(x - prev) / len(x))
        prev = x
    return pd.DataFrame(
        {
            "top": top.groupby("date")["R"].mean().reindex(dates).to_numpy(),
            "bot": bot.groupby("date")["R"].mean().reindex(dates).to_numpy(),
            "uni": d.groupby("date")["R"].mean().reindex(dates).to_numpy(),
            "turn": turn,
        },
        index=pd.DatetimeIndex(dates),
    )


def _comp(r: pd.Series) -> dict:
    x = r.fillna(0.0).to_numpy()
    nav = np.cumprod(1 + x)
    peak = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    return {
        "ann": float(nav[-1] ** (12 / len(x)) - 1),
        "cum": float(nav[-1] - 1),
        "mdd": float((nav / peak - 1).min()),
        "worst": float(x.min()),
    }


def book_stats(res, name: str, cost: float) -> dict:
    """The model's compounded month-end books (post hoc) over the outer + holdout OOS month-ends."""
    w = {"outer": _series(res.oos, name), "holdout": _series(res.holdout_preds, name)}
    a = pd.concat([w["outer"], w["holdout"]]).sort_index()
    assert not a.index.duplicated().any()
    mom = pd.concat([_series(res.oos, sel.BASELINE), _series(res.holdout_preds, sel.BASELINE)]).sort_index()
    r = a["top"].fillna(0.0).to_numpy()
    keep = np.ones(len(r), bool)
    keep[np.argsort(r)[-10:]] = False

    def t(x: pd.DataFrame) -> float:
        e = (x["top"] - x["uni"]).fillna(0.0)
        return float(e.mean() / e.std(ddof=1) * np.sqrt(len(e)))

    top, net = _comp(a["top"]), _comp(a["top"] - a["turn"] * cost)
    return {
        "post_hoc": True,
        "first": str(a.index[0].date())[:7],
        "last": str(a.index[-1].date())[:7],
        "months": int(len(a)),
        "months_outer": int(len(w["outer"])),
        "holdout_first": str(w["holdout"].index[0].date())[:7],
        "holdout_last": str(w["holdout"].index[-1].date())[:7],
        "cost_bps": float(cost * 1e4),
        "top_ann": top["ann"],
        "top_ann_net": net["ann"],
        "top_cum": top["cum"],
        "top_cum_net": net["cum"],
        "top_mdd": top["mdd"],
        "top_worst": top["worst"],
        "top_ann_outer": _comp(w["outer"]["top"])["ann"],
        "top_ann_holdout": _comp(w["holdout"]["top"])["ann"],
        "top_t_outer": t(w["outer"]),
        "top_t_holdout": t(w["holdout"]),
        "top_ann_drop10": float(np.prod(1 + r[keep]) ** (12 / len(r)) - 1),
        "top_beat_share": float((a["top"] > a["uni"]).mean()),
        "top_turnover": float(a["turn"].mean()),
        "universe_ann": _comp(a["uni"])["ann"],
        "universe_mdd": _comp(a["uni"])["mdd"],
        "bottom_ann": _comp(a["bot"])["ann"],
        "mom_top_ann": _comp(mom["top"])["ann"],
    }


def month_end_view(ds):
    """A shallow copy of ``ds`` whose latest session is its last month-end (the books' formation date)."""
    me = ds.frame.loc[ds.frame["is_me"].astype(bool) & (ds.frame["date"] <= ds.as_of), "date"]
    out = copy.copy(ds)
    out.as_of = pd.Timestamp(me.max())
    return out


def returns_1m() -> tuple[pd.Series, pd.Series]:
    """Monthly simple returns R[(date, entity)] and universe means U[date] from the 1-month 99 run's 20-session
    labels (99's analysis)."""
    _, res = v99.run(1, no_fit=True)
    f = pd.concat([res.oos, res.holdout_preds], ignore_index=True)
    f = f[f.groupby("date")["y_e"].transform("size") >= mt.MIN_NAMES].dropna(subset=["r"])
    R = np.expm1(f["r"].to_numpy(float))
    ret = pd.Series(R, index=pd.MultiIndex.from_frame(f[["date", "entity"]]))
    return ret[~ret.index.duplicated()], f.assign(R=R).groupby("date")["R"].mean()


def hold_stats(res, name: str, hold: int, cost: float) -> dict:
    """Post hoc books of a longer-horizon model (99's rules): each month-end D20 cohort held ``hold`` months in
    overlapping 1/hold slices, monthly 20-session returns (missing -> universe mean), one-way cost x traded share;
    from the first month with every slice live. Same keys as ``book_stats``."""
    ret, uni = returns_1m()
    f = pd.concat([res.oos, res.holdout_preds], ignore_index=True)
    f = f[f.groupby("date")["y_e"].transform("size") >= mt.MIN_NAMES]
    grid = sorted(set(f["date"].unique()) & set(uni.index))

    def book(n: str) -> tuple[pd.Series, pd.Series]:
        top, _ = v99.ventile_sets(f[f["date"].isin(grid)], n)
        return v99.Books(grid, ret, uni).hold(top, hold)

    r, tc = book(name)
    mom, _ = book(sel.BASELINE)
    w = grid[hold - 1 :]
    r, tc, mom, u = r.loc[w], tc.loc[w], mom.loc[w], uni.reindex(w)
    h0 = pd.Timestamp(res.holdout_preds["date"].min())
    outer, held = [d for d in w if d < h0], [d for d in w if d >= h0]
    x = r.fillna(0.0).to_numpy()
    keep = np.ones(len(x), bool)
    keep[np.argsort(x)[-10:]] = False

    def t(ix: list) -> float:
        e = (r.loc[ix] - u.loc[ix]).fillna(0.0)
        return float(e.mean() / e.std(ddof=1) * np.sqrt(len(e)))

    top, net = _comp(r), _comp(r - tc * cost)
    return {
        "post_hoc": True,
        "hold_months": int(hold),
        "first": str(w[0].date())[:7],
        "last": str(w[-1].date())[:7],
        "months": int(len(w)),
        "months_outer": int(len(outer)),
        "holdout_first": str(held[0].date())[:7],
        "holdout_last": str(held[-1].date())[:7],
        "cost_bps": float(cost * 1e4),
        "top_ann": top["ann"],
        "top_ann_net": net["ann"],
        "top_cum": top["cum"],
        "top_cum_net": net["cum"],
        "top_mdd": top["mdd"],
        "top_worst": top["worst"],
        "top_ann_outer": _comp(r.loc[outer])["ann"],
        "top_ann_holdout": _comp(r.loc[held])["ann"],
        "top_t_outer": t(outer),
        "top_t_holdout": t(held),
        "top_ann_drop10": float(np.prod(1 + x[keep]) ** (12 / len(x)) - 1),
        "top_beat_share": float((r > u).mean()),
        "top_turnover": float(tc.mean()),
        "universe_ann": _comp(u)["ann"],
        "universe_mdd": _comp(u)["mdd"],
        "mom_top_ann": _comp(mom)["ann"],
    }


def build(m: str, name: str, s, cache) -> None:
    spec = MODELS[name]
    cand = spec.get("candidate", name)  # the run's candidate (rf_bo_3m is the 3-month run's rf_bo)
    ds, res = load_run(m, spec["source"], s, cache)
    if res.final is None or cand not in res.final.fitted:
        raise SystemExit(f"{name}: no final-context fit of {cand} in the cached {spec['source']} run")
    protocol_pick = res.selected
    res.selected = cand
    res.gate = sel.serving_gate(res.candidate(cand)["outer"], res.candidate(sel.BASELINE)["outer"])
    res.gate["reference_only"] = {"protocol_choice": protocol_pick, "decided": f"user request (docs {spec['doc']})"}
    res.calibration = sel.calibration(pd.concat([res.oos, res.holdout_preds], ignore_index=True), cand)
    raw = res.final.fitted[cand].importance()
    tot = sum(abs(v) for v in raw.values()) or 1.0
    res.importances = dict(sorted(((k, abs(v) / tot) for k, v in raw.items()), key=lambda kv: -kv[1])[:20])
    books_ds = ds
    if spec.get("rescore"):
        ds = v86.load_frozen(m, s, cache)
        rescore(res, cand, books_ds, ds)
    elif spec.get("month_end"):
        ds = month_end_view(books_ds)
        rescore(res, cand, books_ds, ds)
    art = build_artifact(res, ds, s)
    if cand != name:  # store under the artifact name (the index key and the report's model id)
        art.selected, art.members = name, [name]
        art.fitted = {name: art.fitted[cand]}
        art.member_scores = art.member_scores.rename(columns={cand: name})
    art.label_ko, art.label_en = spec["label_ko"], spec["label_en"]
    cost = float(books_ds.cost_bps) / 1e4
    hold = int(spec.get("hold", 1))
    st = book_stats(res, cand, cost) if hold == 1 else hold_stats(res, cand, hold, cost)
    fill = {
        "n_outer": st["months_outer"],
        "t_outer": st["top_t_outer"],
        "t_holdout": st["top_t_holdout"],
        "h0": st["holdout_first"],
        "h1": st["holdout_last"],
        "drop10": f"{100 * st['top_ann_drop10']:+.1f}%",
        "mdd": f"{100 * st['top_mdd']:.0f}%",
        "turn": f"{100 * st['top_turnover']:.0f}%",
    }
    if hold == 3:  # 101.3 rule 5 (pre-declared): the 3-month model held 3 months on 2005-2015
        r5 = json.loads(FRESH_101.read_text())["rule"]["5_fresh_E_rf_bo_hold3"]
        fill.update({"fresh": f"{100 * r5['ann']:+.1f}%", "t_fresh": r5["t_vs_u"]})
        st["fresh_2005_2015"] = {"ann_net50": r5["ann"], "t_vs_universe": r5["t_vs_u"], "months": r5["months"]}
    st.update({k: spec[k] for k in ("short_ko", "short_en", "abbr_ko", "abbr_en")})
    st.update({k: spec[k].format(**fill) for k in ("caveat_ko", "caveat_en")})
    st["source"] = f"docs/MODELING.md {spec['doc']} ({spec['source']} cached run)"
    art.metrics = {**art.metrics, "ventile": st, "snapshot": ds.snapshot, "books_snapshot": books_ds.snapshot}
    art.notes = [
        *art.notes,
        {"ko": f"참고용: 신호·비중·기대수익에 쓰지 않음 (docs/MODELING.md {spec['doc']})", "en": "reference only"},
    ]
    save_order_artifact(art, s, cache)
    back = load_order_artifact(m, s, cache, name)
    assert back is not None and back.selected == name and len(back.latest) == len(art.latest)
    print(
        f"saved the {m} reference artifact {name} (as of {art.as_of.date()}, {len(art.latest)} names): "
        f"D20 {100 * st['top_ann']:+.1f}%/yr (net {100 * st['top_ann_net']:+.1f}%), universe "
        f"{100 * st['universe_ann']:+.1f}%, mdd {100 * st['top_mdd']:.0f}%, drop10 {100 * st['top_ann_drop10']:+.1f}%, "
        f"t {st['top_t_outer']:+.2f}/{st['top_t_holdout']:+.2f}"
    )


def write_lists(m: str, s, cache) -> None:
    """reports/modeling/<M>_order_latest.{csv,md}: every stored model's ventiles and D20 lists."""
    names = order_models(m, s, cache)
    arts = [a for a in (load_order_artifact(m, s, cache, n) for n in names) if a is not None]
    if not arts:
        print("no reference model stored")
        return
    base, tops = None, {}
    for a in arts:
        lat = a.latest[[c for c in ("entity", "ticker", "name", "sector", "percentile", "score") if c in a.latest]]
        lat = lat.assign(ventile=lat["percentile"].map(ventile_of))
        tops[a.selected] = lat[lat["ventile"] == ORDER_K].sort_values("score", ascending=False)
        part = lat[["entity", "ventile", "percentile"]].rename(
            columns={"ventile": f"ventile_{a.selected}", "percentile": f"percentile_{a.selected}"}
        )
        if base is None:
            base = lat[[c for c in ("entity", "ticker", "name", "sector") if c in lat]].merge(part, on="entity")
        else:
            base = base.merge(part, on="entity", how="outer")
    first = arts[0].selected
    csv = v86.OUT / f"{m}_order_latest.csv"
    base.sort_values(f"percentile_{first}", ascending=False).to_csv(csv, index=False, encoding="utf-8-sig")
    lines = [f"# {m} 20분위 참고 모델 — 상위 5%(D20) 목록 (docs/MODELING.md 93, 94.3, 96)", ""]
    common = set.intersection(*(set(t["ticker"]) for t in tops.values())) if len(tops) > 1 else set()
    for a in arts:
        st = a.metrics.get("ventile") or {}
        top = tops[a.selected]
        lines += [
            f"## {st.get('short_ko', a.selected)} (`{a.selected}`)",
            "",
            f"- 기준일 {a.as_of.date()} · 횡단면 {len(a.latest)}종목(확인용으로 떼어 둔 종목 제외)"
            f" · D20 {len(top)}종목",
            f"- {a.label_ko}. 신호·비중·기대수익에 쓰지 않습니다.",
            f"- 사후 장부({st.get('first')}~{st.get('last')}, {st.get('months')}개월, 매달 말 같은 비중,"
            f" {int(a.horizon)}거래일 보유, 한 달 교체 {100 * st.get('top_turnover', np.nan):.0f}%): "
            f"D20 연 {100 * st.get('top_ann', np.nan):+.1f}% (비용 {st.get('cost_bps', 0):.0f}bp 후 "
            f"{100 * st.get('top_ann_net', np.nan):+.1f}%), 전체 평균 {100 * st.get('universe_ann', np.nan):+.1f}%, "
            f"D20 최대 낙폭 {100 * st.get('top_mdd', np.nan):.0f}%, 최고 10개월 제외 시 "
            f"{100 * st.get('top_ann_drop10', np.nan):+.1f}%",
            f"- {st.get('caveat_ko', '')}",
            "",
            "| 순위 | 종목 | 코드 | 업종 | 분위 내 위치 | 공통 |",
            "|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(top.itertuples(index=False), 1):
            both = "○" if r.ticker in common else ""
            lines.append(
                f"| {i} | {r.name} | {r.ticker} | {r.sector} | 상위 {100 * (1 - r.percentile):.1f}% | {both} |"
            )
        lines.append("")
    if len(tops) > 1:
        lines.append(f"- 모든 모델의 D20에 든 종목: {len(common)}개 (표의 ○)")
    md = v86.OUT / f"{m}_order_latest.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md} and {csv} ({', '.join(a.selected for a in arts)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR"], required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", choices=sorted(MODELS))
    g.add_argument("--drop")
    a = ap.parse_args()
    s = load_settings()
    cache = DiskCache(s.cache_path)
    if a.drop:
        print(f"removed {a.drop}" if remove_order_artifact(a.market, a.drop, s, cache) else f"{a.drop} not stored")
    else:
        build(a.market, a.model, s, cache)
    write_lists(a.market, s, cache)


if __name__ == "__main__":
    main()
