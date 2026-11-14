"""Cross-sectional holdout (XSH): never-tuned names and the one-time evaluation of the frozen stock-selection model.

``quantcio holdout register --kind xs --market KR|US`` calls ``xsh_candidate_names``; ``quantcio holdout evaluate
--id XSH-...`` calls ``run_xsh_evaluation`` once and records the look (docs/HOLDOUT.md "횡단면 홀드아웃(XSH)").

Names
-----
- KR: entities whose point-in-time market-cap rank was inside ``holdout.xsh_kr_rank_band`` (201-400) on some
  month-end since ``backtest.kr_start`` and that were NEVER a tuning name (``holdout.tuning_names``: every PIT
  top-200 member since kr_start with every ticker it carried, today's evaluate lists, doc tickers).
- US: S&P MidCap 400 names that were never S&P 500 members (``data.pit.load_pit_xsh_universe``).

Evaluation (one look)
---------------------
1. Tuning dataset = ``modeling.dataset.build_dataset`` (the PIT modeling universe) -> ``selection.make_contexts``
   (the same outer walk-forward blocks + final holdout block, purge and embargo as ``modeling select``).
2. XSH dataset = the same builder on the XSH universe (every month-end member of the band; features are ranked
   within the XSH cross-section). Date-level regime columns (``r_*``) are taken from the tuning dataset, so the
   model sees the regime it was trained on.
3. For every context: the SELECTED candidate of the cached ``modeling select`` result (same code fingerprint and
   config digest, else refused before any XSH row is scored) is tuned by inner CV on TUNING rows only (the
   selection checkpoint's configuration is reused when it matches), refitted on tuning rows dated <= the purged
   train end and scores the XSH rows of the context's test month-ends inside the registered period. Ensemble:
   the members recorded by the selection checkpoint (else ``candidates.composite_rule``), per-date (weighted) rank
   mean. RSMT-Q v3: stacked on the best inner candidate.
4. Metrics per date, then over dates (``modeling.metrics``): primary within-date AUC vs 1{y_e > date median};
   also AUC vs 1{y_e > 0}, rank IC, top-quintile net excess and the 12-1 momentum baseline. The verdict follows the
   rules stored in the registry entry (``holdout.xsh_primary_metrics_block``): BH over three one-sided tests,
   ``confirmed`` = the AUC test survives BH, mean IC > 0 and enough dates; ``target_met`` = mean AUC >= the user's
   target (``holdout.target_auc``; a goal, not a test).
5. Sensitivity (never part of the verdict): the same metrics on EVERY band row, including names that were in the
   top 200 at other dates. Registering only never-tuned names conditions on the future (a name that later grew into
   the top 200 is excluded), so the two samples bracket that selection effect.

Outputs: ``<report.output_dir>/holdout/<id>/xsh_<id>.md`` (Korean) and ``.json``. Progress messages never contain a
metric, so an interrupted run reveals nothing (the CLI records the look only after this returns).
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from quantcio import holdout
from quantcio.config import Settings
from quantcio.data.cache import DiskCache
from quantcio.data.pit import PITUniverse
from quantcio.models import Market

log = logging.getLogger(__name__)

OUT_SUBDIR = "holdout"
BOOT_DRAWS = holdout.BOOT_DRAWS
BOOT_BLOCK = holdout.XSH_BOOT_BLOCK_MONTHS
MIN_NAMES = holdout.XSH_MIN_NAMES_PER_DATE
MOMENTUM_COL = "x_mom_12_1"


class XSHError(holdout.HoldoutError):
    """The XSH evaluation cannot run (Korean message); nothing was scored, so no look is recorded."""


# ================================================================================================ universe / names


def xsh_universe(
    market: Market | str,
    settings: Settings,
    cache: DiskCache,
    *,
    exclude: Sequence[str] = (),
    band: Sequence[int] | None = None,
) -> Any:
    """The XSH universe of `market`: KR every PIT member of the rank band per month-end (all names, the
    registered ones are a subset; ``band`` = the entry's registered band, default ``holdout.xsh_kr_rank_band``);
    US the never-S&P 500 mid caps (``exclude`` = tuning names to drop)."""
    mk = Market(str(getattr(market, "value", market)).upper())
    _, start = holdout.pit_tuning_universe(mk, settings)
    if mk is Market.KR:
        from quantcio.data.pit import load_pit_universe

        lo, hi = (int(x) for x in (band or settings.holdout.xsh_kr_rank_band))
        if lo <= int(settings.backtest.kr_universe_n) or hi < lo:
            raise XSHError(
                f"XSH 순위 구간 {lo}~{hi}가 튜닝 유니버스(상위 {settings.backtest.kr_universe_n})와 겹칩니다"
            )
        full = load_pit_universe(Market.KR, settings, cache, hi, start)
        return band_universe(full, lo, hi)
    from quantcio.data import pit as pit_mod

    fn = getattr(pit_mod, "load_pit_xsh_universe", None)
    if fn is None:
        raise XSHError("US XSH 데이터 소스(data.pit.load_pit_xsh_universe)가 없습니다")
    return fn(mk, settings, cache, start, None, exclude=tuple(exclude))


def band_universe(full: PITUniverse, lo: int, hi: int) -> PITUniverse:
    """Rows of `full` whose PIT rank is in [lo, hi], with prices / meta / benchmarks of those entities only."""
    mem = full.membership
    band = mem[(mem["rank"] >= lo) & (mem["rank"] <= hi)].reset_index(drop=True)
    ents = set(band["entity"].astype(str))
    meta = full.meta
    if meta is not None and len(meta) and "entity" in meta:
        meta = meta[meta["entity"].astype(str).isin(ents)].reset_index(drop=True)
    cov = dict(full.coverage)
    cov["xsh_band"] = [int(lo), int(hi)]
    cov["xsh_band_rows"] = len(band)
    cov["xsh_band_entities"] = len(ents)
    return replace(
        full,
        n=int(hi),
        membership=band,
        prices={e: px for e, px in full.prices.items() if e in ents},
        entity_benchmark={e: b for e, b in full.entity_benchmark.items() if e in ents},
        meta=meta,
        coverage=cov,
        notes=[f"XSH 순위 구간 {lo}~{hi}위(시점별 시가총액) 구성: {len(band):,}행 · {len(ents)}개 엔티티", *full.notes],
    )


def xsh_candidate_names(market: Market | str, settings: Settings, cache: DiskCache) -> list[str]:
    """Never-tuned names of the XSH universe (entity ids), sorted. The CLI registers them as the XSH entry.

    Raises:
        holdout.HoldoutError: the tuning names cannot be established, or no candidate is left.
    """
    mk = Market(str(getattr(market, "value", market)).upper())
    tuned = set(holdout.tuning_names(mk, settings, cache))
    uni = xsh_universe(mk, settings, cache, exclude=sorted(tuned))
    ents = sorted({str(e) for e in uni.membership["entity"].unique()})
    names = [e for e in ents if holdout._name_key(e) not in tuned]
    tickers = uni.membership.groupby("entity")["ticker"].agg(lambda s: {holdout._name_key(t) for t in s.dropna()})
    names = [e for e in names if not (tickers.get(e, set()) & tuned)]
    log.info(
        "xsh candidates %s: %d of %d universe entities (tuning names %d)", mk.value, len(names), len(ents), len(tuned)
    )
    if not names:
        raise holdout.HoldoutError(f"{mk.value} XSH 후보가 없습니다(유니버스 {len(ents)}개가 모두 튜닝 종목)")
    return names


# ================================================================================================ evaluation


def _p_boot(values: np.ndarray, null: float, seed: int) -> float:
    """One-sided block-bootstrap p: share of bootstrap means <= null (+1 smoothing)."""
    from quantcio.modeling import metrics as mt

    draws = mt.block_bootstrap_means(np.asarray(values, dtype=float), block=BOOT_BLOCK, n_boot=BOOT_DRAWS, seed=seed)
    if draws.size == 0:
        return float("nan")
    return float((1 + np.sum(draws <= null)) / (1 + draws.size))


def _p_t(values: np.ndarray) -> float:
    """One-sided t-test p of mean > 0 (df = n - 1)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3 or float(x.std(ddof=1)) == 0.0:
        return float("nan")
    t = float(x.mean() / (x.std(ddof=1) / math.sqrt(x.size)))
    return float(stats.t.sf(t, x.size - 1))


def _selection_result(market: Market, settings: Settings, cache: DiskCache, given: Any) -> dict[str, Any]:
    from quantcio.modeling import selection as sel

    res = given if given is not None else cache.get(sel.RESULT_NS, market.value, None)
    if hasattr(res, "to_dict"):
        res = res.to_dict()
    if not isinstance(res, dict) or not res.get("selected"):
        raise XSHError(
            f"{market.value} 종목 선택 결과가 없습니다 — `quantcio modeling select --market {market.value}`를 먼저 실행"
        )
    fp_now = sel._fingerprint()
    fp_res = (res.get("protocol") or {}).get("fingerprint")
    if fp_res != fp_now:
        raise XSHError(f"선택 결과의 코드 지문({fp_res})이 현재({fp_now})와 다릅니다 — modeling select 재실행 후 평가")
    if res.get("settings_digest") != settings.digest():
        raise XSHError(
            f"선택 결과의 설정 다이제스트({res.get('settings_digest')})가 현재({settings.digest()})와 다릅니다 — "
            "modeling select 재실행 후 평가"
        )
    return res


def _align_regime(xds: Any, ds: Any) -> tuple[pd.DataFrame, int]:
    """XSH frame with the tuning dataset's date-level regime columns (rows whose date the tuning set lacks keep
    their own values; counted)."""
    xf = xds.frame.copy()
    cols = [c for c in ds.regime_cols if c in xf]
    ref = ds.dates_frame.reindex(columns=cols)
    have = xf["date"].isin(ref.index).to_numpy()
    for c in cols:
        own = xf[c].to_numpy(dtype=float)
        xf[c] = np.where(have, ref[c].reindex(xf["date"]).to_numpy(dtype=float), own)
    return xf, int((~have).sum())


def _coverage_pit_from(xds: Any) -> pd.Timestamp | None:
    """``pit_from`` of an injected XSH dataset (``coverage['pit_from']``; None = PIT over the whole period)."""
    pf = (getattr(xds, "coverage", None) or {}).get("pit_from")
    return pd.Timestamp(pf) if pf else None


def _checkpoint_configs(data: Any, ctx: Any, cache: DiskCache) -> dict[str, dict[str, Any]]:
    """Per-candidate configs of the ``modeling select`` checkpoint for this context ({} when absent)."""
    from quantcio.modeling import selection as sel
    from quantcio.modeling.candidates import STACKED, default_candidates

    try:
        base_names = [c.name for c in default_candidates() if c.name not in STACKED]  # = selection's key
        ident = [data.ds.snapshot, ctx.name, str(ctx.train_end), [str(d) for d in ctx.test_dates]]
        key = DiskCache.key(sel.CTX_NS, sel._fingerprint(), *ident, base_names)
        hit = cache.get(sel.CTX_NS, key, None)
        return dict(getattr(hit, "configs", {}) or {})
    except Exception as exc:  # noqa: BLE001 - only costs a re-tune
        log.warning("xsh: checkpoint lookup failed for %s: %s", ctx.name, exc)
        return {}


def _score_context(
    data: Any,
    ctx: Any,
    rows: pd.DataFrame,
    selected: str,
    settings: Settings,
    cache: DiskCache,
    threads: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Scores of the selected model for `rows` (XSH rows of this context's test dates) and the configs used."""
    from quantcio.modeling import selection as sel
    from quantcio.modeling.candidates import COMPOSITES, STACKED, composite_rule, default_candidates
    from quantcio.modeling.features import rank_gauss

    cand_list = default_candidates()
    cands = {c.name: c for c in cand_list}
    if selected not in cands and selected not in COMPOSITES:
        raise XSHError(f"선택 모델 {selected!r}을(를) 후보 목록에서 찾을 수 없습니다")
    eligible = [c for c in cand_list if c.tuned and not c.diagnostic and c.name not in STACKED]
    ck = _checkpoint_configs(data, ctx, cache)
    ck_members = list((ck.get(selected) or {}).get("members") or []) if selected in COMPOSITES else []
    if selected in cands and selected not in STACKED:
        need = [cands[selected]]
    elif ck_members:  # a composite whose members the selection checkpoint recorded for this context
        need = [cands[m] for m in ck_members if m in cands]
    else:  # composite without a checkpoint / RSMT-Q v3 (stacked on the best inner candidate): every eligible one
        need = eligible
    cfgs: dict[str, sel.Config] = {}
    for cand in need:
        c = ck.get(cand.name)
        if isinstance(c, dict) and "params" in c and "scheme" in c:
            cfgs[cand.name] = sel.Config(
                dict(c["params"]), str(c["scheme"]), c.get("inner_auc", np.nan), c.get("inner_ic", np.nan)
            )
        else:
            cfgs[cand.name] = sel.tune(data, cand, ctx.train_end, settings, threads)
    info: dict[str, Any] = {
        "configs": {k: {"params": v.params, "scheme": v.scheme, "inner_auc": v.inner_auc} for k, v in cfgs.items()},
        "checkpoint": bool(ck),
    }

    def fit_predict(name: str, frame: pd.DataFrame) -> tuple[np.ndarray, Any]:
        cand = cands[name]
        tr = data.train(ctx.train_end, cand.rows)
        fitted = cand.fit(
            tr, cfgs[name].params, cfgs[name].scheme, sel._ctx(data, ctx.test_dates[0], settings, threads)
        )
        return np.asarray(fitted.predict(frame), dtype=float), fitted

    def rank_mean(members: list[str], weights: list[float] | None) -> np.ndarray:
        info["members"] = list(members)
        info["weights"] = list(weights) if weights else None
        preds = rows[["date"]].copy()
        for m in members:
            preds[m] = fit_predict(m, rows)[0]
        w = weights if weights and any(x != 1.0 for x in weights) else None
        return sel.rank_mean(preds, list(members), w).to_numpy(dtype=float)

    if selected in cands and selected not in STACKED:
        score, _ = fit_predict(selected, rows)
        return score, info
    inner = {n: {"inner_auc": c.inner_auc, "inner_ic": c.inner_ic} for n, c in cfgs.items()}
    if selected in COMPOSITES:
        if ck_members:
            weights = (ck.get(selected) or {}).get("weights")
            return rank_mean(ck_members, list(weights) if weights else None), info
        members, weights = composite_rule(selected, inner, cand_list)
        if not members:
            raise XSHError(f"{ctx.name}: {selected} 구성원을 정할 수 없습니다")
        return rank_mean(members, weights), info
    ranked = sorted(
        [c.name for c in eligible if c.name in cfgs and np.isfinite(cfgs[c.name].inner_auc)],
        key=lambda n: (-cfgs[n].inner_auc, -np.nan_to_num(cfgs[n].inner_ic)),
    )
    if not ranked:
        raise XSHError(f"{ctx.name}: 내부 CV 점수가 있는 후보가 없습니다")
    # RSMT-Q v3 (STACKED) stacked on the best inner candidate
    base = ranked[0]
    info["stack_base"] = base
    base_score, _ = fit_predict(base, rows)
    sz = sel.stack_scores(data, cands[base], cfgs[base], ctx.train_end, settings, threads)
    rs = cands[selected]
    tr = data.train(ctx.train_end, rs.rows).copy()
    tr["stack_z"] = sz.reindex(tr.index)
    te = rows.copy()
    te["stack_z"] = rank_gauss(pd.Series(base_score, index=te.index), te["date"])
    fitted = rs.fit(tr, dict(rs.grid[0]), "none", sel._ctx(data, ctx.test_dates[0], settings, threads))
    return np.asarray(fitted.predict(te), dtype=float), info


def _metrics(frame: pd.DataFrame, cost_bps: float, seed: int = 0) -> dict[str, Any]:
    from quantcio.modeling import metrics as mt

    out = {"model": mt.evaluate(frame, cost_bps, "score", seed), "momentum": mt.evaluate(frame, cost_bps, "mom", seed)}
    a = mt.per_date_auc_median(frame, "score")
    b = mt.per_date_auc_median(frame, "mom")
    out["vs_momentum"] = mt.paired_difference(a, b, seed)
    out["_series"] = {"auc_median": a, "auc_median_mom": b, "ic": mt.per_date_ic(frame, "score")}
    return out


def verdict(m: Mapping[str, Any], settings: Settings) -> dict[str, Any]:
    """Pre-registered XSH verdict (``holdout.xsh_primary_metrics_block``) from ``_metrics`` output."""
    ser = m["_series"]
    a, b, ic = ser["auc_median"], ser["auc_median_mom"], ser["ic"]
    common = a.index.intersection(b.index)
    diff = (a.reindex(common) - b.reindex(common)).to_numpy(dtype=float)
    p = {
        holdout.XSH_METRIC_AUC: _p_boot(a.to_numpy(dtype=float), 0.5, 11),
        holdout.XSH_METRIC_IC: _p_t(ic.to_numpy(dtype=float)),
        holdout.XSH_METRIC_VS_MOM: _p_boot(diff, 0.0, 13),
    }
    keys = list(holdout.XSH_PRIMARY_KEYS)
    from quantcio.tracking import benjamini_hochberg

    rej = dict(zip(keys, benjamini_hochberg([p[k] for k in keys], float(settings.holdout.bh_q)), strict=True))
    mm = m["model"]
    n_dates = int(len(a))
    enough = n_dates >= int(settings.holdout.min_dates_for_verdict)
    auc = mm["auc_median"]
    ic_s = mm["ic"]
    target = float(settings.holdout.target_auc)
    lo = auc.get("lo")
    lo_ok = lo is not None and math.isfinite(lo) and lo > holdout.XSH_CONFIRM_MIN_LO
    confirmed = bool(enough and rej[holdout.XSH_METRIC_AUC] and lo_ok and (ic_s["mean"] or 0.0) > 0)
    target_met = bool(auc["mean"] is not None and auc["mean"] >= target)
    if not enough:
        text = f"판정 불가: 날짜 {n_dates}개 < {settings.holdout.min_dates_for_verdict}"
    elif confirmed:
        text = "확인: 미열람 종목에서 날짜 내 AUC(중앙값 초과) > 0.5가 BH 보정 후 유의하고 95% 하한 > 0.5"
    elif rej[holdout.XSH_METRIC_AUC] and not lo_ok:
        text = "미확인: BH 보정 후 유의하나 AUC 95% 하한이 0.5 이하(약한 근거)"
    else:
        text = "미확인: 미열람 종목에서 날짜 내 AUC(중앙값 초과) > 0.5를 입증하지 못함"
    return {
        "p": p,
        "bh_reject": rej,
        "n_dates": n_dates,
        "enough_dates": enough,
        "confirmed": confirmed,
        "target_auc": target,
        "target_met": target_met,
        "text": text,
    }


def run_xsh_evaluation(
    entry: Mapping[str, Any],
    settings: Settings,
    cache: DiskCache,
    *,
    threads: int = 8,
    progress: Callable[[str], None] | None = None,
    out_dir: Path | None = None,
    selection: Any = None,
    tuning_ds: Any = None,
    xsh_ds: Any = None,
) -> dict[str, Any]:
    """One XSH look (see the module docstring). Returns {"result_path", "md_path", "headline", "result"}.

    ``selection`` (a ``SelectionResult`` or its dict), ``tuning_ds`` and ``xsh_ds`` (``ModelingDataset``) are
    injection points (tests); by default they come from the cache / builders.

    Raises:
        XSHError: missing / stale selection result, a registered name inside the tuning dataset, no scorable date.
    """
    from quantcio.modeling import selection as sel
    from quantcio.modeling.dataset import build_dataset

    say = progress or (lambda msg: log.info("xsh: %s", msg))
    t0 = time.monotonic()
    if entry.get("kind") != "cross_sectional":
        raise XSHError(f"{entry.get('id')}: 횡단면 홀드아웃이 아닙니다")
    mk = Market(str(entry["market"]).upper())
    names = {str(n) for n in entry["names"]}
    name_keys = {holdout._name_key(n) for n in names}
    p0, p1 = (pd.Timestamp(x) for x in entry["period"])
    res = _selection_result(mk, settings, cache, selection)
    selected = str(res["selected"])
    say(f"선택 모델 {selected} · 튜닝 데이터셋 준비")
    ds = tuning_ds if tuning_ds is not None else build_dataset(mk, settings, cache, progress=say)
    tuned_keys = {holdout._name_key(e) for e in ds.frame["entity"].astype(str).unique()}
    tuned_keys |= {holdout._name_key(t) for t in ds.frame["ticker"].dropna().astype(str).unique()}
    overlap = sorted(name_keys & tuned_keys)
    if overlap:
        raise XSHError(f"XSH 종목 {len(overlap)}개가 튜닝 데이터셋에 있습니다: {', '.join(overlap[:8])}")
    data = sel._Data(ds)
    outer, hold_ctx, _final, info = sel.make_contexts(data)
    contexts = [*outer, hold_ctx]
    say("XSH 데이터셋 준비")
    pit_from = None
    if xsh_ds is None:
        uni = xsh_universe(mk, settings, cache, exclude=sorted(tuned_keys), band=entry.get("rank_band"))
        pf = (uni.coverage or {}).get("pit_from")
        pit_from = pd.Timestamp(pf) if pf else None
        alt = None
        if settings.alt.enabled:  # alternative data of the REGISTERED names only (the sensitivity rows get NaN)
            from quantcio.data.alt.load import load_alt_data

            reg_ents = sorted(e for e in uni.membership["entity"].astype(str).unique() if e in names)
            alt = load_alt_data(mk, settings, reg_ents, date(pd.Timestamp(p0).year - 1, 1, 1), progress=say)
        xsh_ds = build_dataset(mk, settings, cache, universe=uni, weekly=False, alt=alt, progress=say)
    xf, regime_own = _align_regime(xsh_ds, ds)
    pit_from = pit_from if pit_from is not None else _coverage_pit_from(xsh_ds)
    start = max(p0, pit_from) if pit_from is not None and pit_from > p0 else p0
    in_period = xf["labelled"] & xf["is_me"] & (xf["date"] >= p0) & (xf["date"] <= p1)
    before_pit = in_period & (xf["date"] < start)
    ev = xf[in_period & ~before_pit].copy()
    ev["registered"] = ev["entity"].astype(str).isin(names)
    ev["mom"] = ev[MOMENTUM_COL].astype(float) if MOMENTUM_COL in ev else np.nan
    ev["score"] = np.nan
    ev["context"] = None
    ctx_rows: list[dict[str, Any]] = []
    for ctx in contexts:
        m = ev["date"].isin(ctx.test_dates)
        if not m.any():
            continue
        say(f"{ctx.name}: 학습 ≤ {ctx.train_end.date()} · XSH {int(m.sum())}행 점수화")
        tc = time.monotonic()
        score, cinfo = _score_context(data, ctx, ev.loc[m], selected, settings, cache, threads)
        ev.loc[m, "score"] = score
        ev.loc[m, "context"] = ctx.name
        ctx_rows.append(
            {
                "name": ctx.name,
                "train_end": str(ctx.train_end.date()),
                "dates": int(ev.loc[m, "date"].nunique()),
                "rows": int(m.sum()),
                "registered_rows": int((m & ev["registered"]).sum()),
                "seconds": round(time.monotonic() - tc, 1),
                **cinfo,
            }
        )
    no_ctx = ev["context"].isna()
    n_no_ctx_dates = int(ev.loc[no_ctx, "date"].nunique())
    scored = ev[~no_ctx & np.isfinite(ev["score"].to_numpy(dtype=float))]
    prim = scored[scored["registered"]]
    if prim.empty:
        raise XSHError("점수화된 XSH 행이 없습니다(기간·컨텍스트 확인)")
    cost = float(getattr(xsh_ds, "cost_bps", 0.0) or getattr(ds, "cost_bps", 0.0))
    m_prim = _metrics(prim, cost)
    v = verdict(m_prim, settings)
    m_all = _metrics(scored, cost, seed=100)
    by_year = _by_year(m_prim["_series"]["auc_median"], m_prim["_series"]["ic"])
    by_ctx = _by_context(prim)
    ref = next((c for c in res.get("candidates", []) if c.get("name") == selected), {})
    mp = m_prim["model"]
    headline = {
        "selected": selected,
        "primary": "auc_median",
        "auc_median": mp["auc_median"]["mean"],
        "auc_median_lo": mp["auc_median"]["lo"],
        "auc_median_hi": mp["auc_median"]["hi"],
        "auc_zero": mp["auc"]["mean"],
        "ic": mp["ic"]["mean"],
        "ic_t": mp["ic"]["t"],
        "top_net_ann": mp.get("top_net_ann"),
        "momentum_auc_median": m_prim["momentum"]["auc_median"]["mean"],
        "vs_momentum": m_prim["vs_momentum"]["mean"],
        "vs_momentum_lo": m_prim["vs_momentum"]["lo"],
        "p": v["p"],
        "bh_reject": v["bh_reject"],
        "n_dates": v["n_dates"],
        "n_names": int(prim["entity"].nunique()),
        "n_rows": len(prim),
        "target_auc": v["target_auc"],
        "target_met": v["target_met"],
        "confirmed": v["confirmed"],
        "verdict": v["text"],
        "sensitivity_auc_median": m_all["model"]["auc_median"]["mean"],
        "sensitivity_auc_median_lo": m_all["model"]["auc_median"]["lo"],
        "sensitivity_ic": m_all["model"]["ic"]["mean"],
    }
    result = {
        "id": entry["id"],
        "market": mk.value,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "period": [str(p0.date()), str(p1.date())],
        "selected": selected,
        "selection": {
            "created_at": res.get("created_at"),
            "as_of": res.get("as_of"),
            "fingerprint": (res.get("protocol") or {}).get("fingerprint"),
            "settings_digest": res.get("settings_digest"),
            "tuning_outer": ref.get("outer"),
            "tuning_holdout": ref.get("holdout"),
        },
        "protocol": {**info, "rules": entry.get("primary_metrics")},
        "headline": headline,
        "primary": {k: val for k, val in m_prim.items() if k != "_series"},
        "sensitivity_all_band_rows": {
            **{k: val for k, val in m_all.items() if k != "_series"},
            "n_names": int(scored["entity"].nunique()),
            "n_rows": len(scored),
        },
        "by_year": by_year,
        "by_context": by_ctx,
        "contexts": ctx_rows,
        "coverage": {
            "registered_names": len(names),
            "registered_names_scored": int(prim["entity"].nunique()),
            "xsh_rows_in_period": len(ev),
            "rows_without_context": int(no_ctx.sum()),
            "dates_without_context": n_no_ctx_dates,
            "rows_regime_from_xsh": regime_own,
            "pit_from": str(pit_from.date()) if pit_from is not None else None,
            "dates_before_pit_from": int(xf.loc[before_pit, "date"].nunique()),
            "xsh_dataset": {k: val for k, val in (xsh_ds.coverage or {}).items() if not isinstance(val, dict | list)},
        },
        "notes": [n.get("ko", str(n)) if isinstance(n, dict) else str(n) for n in (xsh_ds.notes or [])][:6],
        "seconds": round(time.monotonic() - t0, 1),
    }
    root = Path(out_dir) if out_dir is not None else Path(settings.report.output_dir) / OUT_SUBDIR / str(entry["id"])
    root.mkdir(parents=True, exist_ok=True)
    js = root / f"xsh_{entry['id']}.json"
    md = root / f"xsh_{entry['id']}.md"
    js.write_text(json.dumps(holdout._jsonable(result), ensure_ascii=False, indent=1), encoding="utf-8")
    md.write_text(render_markdown(result), encoding="utf-8")
    return {"result_path": str(js), "md_path": str(md), "headline": headline, "result": result}


def _by_year(auc: pd.Series, ic: pd.Series) -> list[dict[str, Any]]:
    out = []
    years = sorted({d.year for d in auc.index} | {d.year for d in ic.index})
    for y in years:
        a = auc[[d.year == y for d in auc.index]] if len(auc) else auc
        i = ic[[d.year == y for d in ic.index]] if len(ic) else ic
        out.append(
            {
                "year": int(y),
                "dates": int(len(a)),
                "auc_median": float(a.mean()) if len(a) else None,
                "ic": float(i.mean()) if len(i) else None,
            }
        )
    return out


def _by_context(frame: pd.DataFrame) -> list[dict[str, Any]]:
    from quantcio.modeling import metrics as mt

    out = []
    for name, g in frame.groupby("context", sort=False):
        a = mt.per_date_auc_median(g, "score")
        i = mt.per_date_ic(g, "score")
        out.append(
            {
                "context": str(name),
                "first": str(pd.Timestamp(g["date"].min()).date()),
                "last": str(pd.Timestamp(g["date"].max()).date()),
                "dates": int(len(a)),
                "auc_median": float(a.mean()) if len(a) else None,
                "ic": float(i.mean()) if len(i) else None,
            }
        )
    return out


# ================================================================================================ markdown


def _f(x: Any, nd: int = 4, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def _ci(s: Mapping[str, Any] | None, nd: int = 4) -> str:
    s = s or {}
    return f"{_f(s.get('mean'), nd)} [{_f(s.get('lo'), nd)}, {_f(s.get('hi'), nd)}]"


def render_markdown(r: Mapping[str, Any]) -> str:
    h = r["headline"]
    pm = r["primary"]
    sm = r["sensitivity_all_band_rows"]
    sel = r["selection"]
    lines = [
        f"# 횡단면 홀드아웃(XSH) 평가 — {r['id']}",
        "",
        f"- 시장 {r['market']} · 기간 {r['period'][0]} ~ {r['period'][1]} · 생성 {r['created_at']}",
        f"- 평가 모델: `{r['selected']}` (modeling select {sel.get('as_of')} · 코드 지문 {sel.get('fingerprint')} · "
        f"설정 {sel.get('settings_digest')})",
        f"- 등록 종목 {r['coverage']['registered_names']}개 중 점수화 {h['n_names']}개 · {h['n_rows']:,}행 · "
        f"날짜 {h['n_dates']}개",
        "",
        "## 판정 (사전 등록 규칙)",
        "",
        f"**{h['verdict']}** · confirmed = {h['confirmed']} · 목표 AUC {h['target_auc']:.2f} "
        f"{'달성' if h['target_met'] else '미달'}",
        "",
        "| 주 지표 | 값 | 단측 p | BH 기각 |",
        "|---|---|---|---|",
        f"| 날짜 내 AUC(중앙값 초과) > 0.5 | {_ci(pm['model']['auc_median'])} | "
        f"{_f(h['p'].get(holdout.XSH_METRIC_AUC), 4)} | {h['bh_reject'].get(holdout.XSH_METRIC_AUC)} |",
        f"| 순위 IC > 0 | {_f(h['ic'], 4, True)} (t {_f(h['ic_t'], 2, True)}) | "
        f"{_f(h['p'].get(holdout.XSH_METRIC_IC), 4)} | {h['bh_reject'].get(holdout.XSH_METRIC_IC)} |",
        f"| AUC(중앙값) − 12-1 모멘텀 | {_ci(pm['vs_momentum'])} | {_f(h['p'].get(holdout.XSH_METRIC_VS_MOM), 4)} | "
        f"{h['bh_reject'].get(holdout.XSH_METRIC_VS_MOM)} |",
        "",
        "## 상세 지표 (등록 종목)",
        "",
        "| 지표 | 선택 모델 | 12-1 모멘텀 |",
        "|---|---|---|",
        f"| AUC(중앙값 초과) [95% CI] | {_ci(pm['model']['auc_median'])} | {_ci(pm['momentum']['auc_median'])} |",
        f"| AUC(y_e > 0) [95% CI] | {_ci(pm['model']['auc'])} | {_ci(pm['momentum']['auc'])} |",
        f"| 순위 IC [95% CI] | {_ci(pm['model']['ic'])} | {_ci(pm['momentum']['ic'])} |",
        f"| 상위 5분위 순초과(월) | {_ci(pm['model']['top_net'])} | {_ci(pm['momentum']['top_net'])} |",
        f"| 상위 5분위 순초과(연환산) | {_f(pm['model'].get('top_net_ann'), 4, True)} | "
        f"{_f(pm['momentum'].get('top_net_ann'), 4, True)} |",
        "",
        "## 튜닝 유니버스 성적과 비교 (같은 모델)",
        "",
        "| 표본 | AUC(중앙값 초과) | AUC(y_e > 0) | 순위 IC |",
        "|---|---|---|---|",
    ]
    for label, block in (
        ("튜닝 유니버스 outer OOS", sel.get("tuning_outer")),
        ("튜닝 유니버스 24개월 홀드아웃", sel.get("tuning_holdout")),
        ("XSH 등록 종목", pm["model"]),
    ):
        b = block or {}
        lines.append(
            f"| {label} | {_f((b.get('auc_median') or {}).get('mean'))} | {_f((b.get('auc') or {}).get('mean'))} | "
            f"{_f((b.get('ic') or {}).get('mean'), 4, True)} |"
        )
    lines += [
        "",
        "## 민감도 (판정 미사용): 순위 구간의 모든 시점별 행",
        "",
        "등록 종목은 '기간 중 한 번도 튜닝 유니버스에 들지 않은 종목'이라 미래 정보로 표본이 정해집니다(나중에 상위로 "
        "성장한 종목 제외). 같은 구간의 모든 행(튜닝 이력 종목 포함)으로 계산한 값을 함께 봅니다.",
        "",
        f"- {sm['n_names']}개 종목 · {sm['n_rows']:,}행: AUC(중앙값 초과) {_ci(sm['model']['auc_median'])} · "
        f"순위 IC {_ci(sm['model']['ic'])} · 모멘텀 AUC {_ci(sm['momentum']['auc_median'])}",
        "",
        "## 연도별 (등록 종목)",
        "",
        "| 연도 | 날짜 | AUC(중앙값 초과) | 순위 IC |",
        "|---|---|---|---|",
    ]
    for y in r["by_year"]:
        lines.append(f"| {y['year']} | {y['dates']} | {_f(y['auc_median'])} | {_f(y['ic'], 4, True)} |")
    lines += ["", "## 컨텍스트별 (학습 구간)", "", "| 컨텍스트 | 기간 | 날짜 | AUC(중앙값 초과) | 순위 IC | 학습 끝 |"]
    lines.append("|---|---|---|---|---|---|")
    train_end = {c["name"]: c["train_end"] for c in r["contexts"]}
    for c in r["by_context"]:
        lines.append(
            f"| {c['context']} | {c['first']}~{c['last']} | {c['dates']} | {_f(c['auc_median'])} | "
            f"{_f(c['ic'], 4, True)} | {train_end.get(c['context'], '—')} |"
        )
    cov = r["coverage"]
    lines += [
        "",
        "## 방법과 한계",
        "",
        "- 모델은 튜닝 유니버스 행으로만 학습(컨텍스트마다 내부 CV·정제·엠바고 동일), XSH 종목은 점수화에만 사용.",
        "- 특징은 XSH 횡단면 안에서 순위 정규화, 날짜 단위 레짐 특징은 튜닝 데이터셋 값 사용.",
        "- 주 라벨 1{y_e > 날짜 중앙값}은 v1 결과를 본 뒤 정한 것(사후) — 그래서 이 미열람 표본의 확인이 필요합니다.",
        (
            f"- 멤버십이 시점별(PIT)인 날짜는 {cov['pit_from']}부터 — 그 이전 월말 {cov['dates_before_pit_from']}개는 "
            "생존 편향이 있어 평가에서 제외."
            if cov.get("pit_from")
            else "- 멤버십은 전 기간 시점별(PIT)."
        ),
        f"- 컨텍스트 없는 날짜 {cov['dates_without_context']}개(학습 이력 부족 구간) 제외 · 레짐 값을 XSH 자체로 쓴 행 "
        f"{cov['rows_regime_from_xsh']}개.",
        f"- 소요 {r['seconds']}초.",
    ]
    for n in r.get("notes", []):
        lines.append(f"- [데이터] {n}")
    return "\n".join(lines) + "\n"


__all__ = [
    "XSHError",
    "band_universe",
    "render_markdown",
    "run_xsh_evaluation",
    "verdict",
    "xsh_candidate_names",
    "xsh_universe",
]
