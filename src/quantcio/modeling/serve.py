"""Serving the selected stock-selection model (docs/MODELING.md, "서빙").

``build_artifact`` (run by ``quantcio modeling select``) turns the FINAL context of the selection run - every
candidate fitted by the same inner-CV procedure on all rows whose labels are known at the latest session
(date <= latest - h sessions, i.e. t_end <= latest) - into a cached ``ServingArtifact``: the latest PIT
cross-section with score, per-date rank-gauss z, percentile, the calibrated expected excess return and the top
feature contributions of every member.

Expected excess return (h-day log, beta-adjusted): ``mu_excess = shrink * slope * z`` where ``slope`` is the
date-clustered OLS slope of the demeaned excess return on z over ALL out-of-sample rows of the selected model
(pre-holdout outer folds + holdout), ``shrink = t^2 / (1 + t^2)`` of that slope (0 when t <= 0), and a further
x ``selection.UNSKILLED_SHRINK`` unless the skill is ESTABLISHED (protocol v2, ``served_state``):
- gate failed (the model did not beat the momentum baseline on the PRIMARY AUC - the within-date AUC against the
  above-median label - or its 95% lower bound is not above 0.5): x0.35, flagged unskilled;
- gate passed but the primary label is post hoc (chosen after the v1 results) and NOT confirmed on untouched
  names: x0.35, flagged "gate passed (post-hoc label) - x0.35 until confirmed on untouched names (XSH)";
- gate passed and confirmed: the full multiplier. Confirmation is looked up AT SCORING TIME (never baked into the
  cached artifact) via ``quantcio.holdout.xsh_confirmation(market, settings) -> dict | None``; a missing
  function, None, an explicit ``confirmed`` / ``passed`` False or any exception = not confirmed.
The stored ``skill_ok`` / ``multiplier`` / ``latest["mu_excess"]`` are the values WITHOUT confirmation;
``score_for_report`` returns the artifact with ``skill_ok`` / ``multiplier`` set to the SERVED values.

``score_for_report`` (report time, no heavy compute): loads the artifact, maps requested tickers to PIT entities
(KR: 6-digit code; US: the display ticker on the latest rebalance date) and returns ``ExcessForecast`` objects
plus the per-ticker ``quant.rsmt.ExcessOverride`` that makes the served ``ReturnDistribution`` use this excess
mean (market component: base rate / prior-shrunk drift unless the market-direction model passed its rule).
Names outside the universe are scored from their own prices (+ current fundamentals) placed into the latest
cross-section's distributions; the note says so.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import special

from quantcio.config import Settings
from quantcio.data.base import normalise_ticker
from quantcio.data.cache import DiskCache
from quantcio.i18n import tr
from quantcio.modeling import features as feat
from quantcio.modeling.candidates import COMPOSITES, STACKED, Fitted
from quantcio.modeling.dataset import ModelingDataset
from quantcio.modeling.metrics import PRIMARY, PRIMARY_AUC, SECONDARY_AUC, auc_label, metric_label
from quantcio.modeling.selection import UNSKILLED_SHRINK, SelectionResult, composite_members
from quantcio.models import ExcessForecast, Market, MarketData

log = logging.getLogger(__name__)

SERVE_NS = "modeling_serving"
ARTIFACT_VERSION = 3  # v3: candidates / features v3, weighted composites; v2: protocol v2 gate + XSH confirmation
TOP_CONTRIB = 5
STALE_DAYS = 10  # an artifact older than this (calendar days) is flagged stale in the notes


@dataclass
class ServingArtifact:
    market: str
    as_of: pd.Timestamp
    created_at: str
    selected: str
    label_ko: str
    label_en: str
    members: list[str]
    fitted: dict[str, Fitted]
    stack_base: str | None
    gate: dict[str, Any]
    calibration: dict[str, Any]
    skill_ok: bool  # skill established WITHOUT confirmation (gate passed and not post hoc); see served_state
    multiplier: float  # applied to slope * z (calibration shrink x unskilled shrink), before any confirmation
    latest: pd.DataFrame  # the latest cross-section (see build_artifact)
    member_scores: pd.DataFrame  # latest raw score of every member (+ stack base) per row of ``latest``
    xs_cols: list[str]
    regime_cols: list[str]
    regime_latest: dict[str, float]
    raw_cols: list[str]
    metrics: dict[str, Any]
    market_model: dict[str, Any]
    importances: dict[str, float]
    horizon: int
    notes: list[dict[str, str]] = field(default_factory=list)
    report_path: str = ""
    gate_passed: bool = False  # the serving gate on the primary AUC
    post_hoc: bool = False  # the primary label was chosen after earlier results were seen: needs confirmation
    primary_auc: str = PRIMARY_AUC  # metric key of metrics["outer_auc"] / ["holdout_auc"]
    member_weights: list[float] = field(default_factory=list)  # composite weights (empty = equal)
    alt_raw: list[str] = field(default_factory=list)  # raw v3 feature names (alternative data + macro betas)


def _key(market: Market | str, settings: Settings) -> str:
    return DiskCache.key(SERVE_NS, ARTIFACT_VERSION, Market(market).value, int(settings.horizon_days))


def _percentile(values: np.ndarray, x: float) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0 or not math.isfinite(x):
        return 0.5
    return float(((v < x).sum() + 0.5 * (v == x).sum() + 0.5) / (v.size + 1))


def _z(p: float) -> float:
    return float(special.ndtri(min(max(p, 1e-4), 1 - 1e-4)))


def _perturb_contrib(f: Fitted, rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Score change when one cross-sectional input is set to its median (0): f(x) - f(x_j := 0)."""
    base = f.predict(rows)
    out = {}
    for c in cols:
        if c not in rows:
            continue
        alt = rows.copy()
        alt[c] = 0.0
        out[c] = base - f.predict(alt)
    return pd.DataFrame(out, index=rows.index)


def _contributions(
    fitted: dict[str, Fitted], members: list[str], rows: pd.DataFrame, xs_cols: list[str]
) -> pd.DataFrame:
    """Per-row feature contributions of the served score (ensembles: each member scaled by its score sd)."""
    parts = []
    for m in members:
        f = fitted.get(m)
        if f is None:
            continue
        c = f.contributions(rows)
        if c is None:
            c = _perturb_contrib(f, rows, [x for x in f.features if x in xs_cols])
        sd = float(np.nanstd(f.predict(rows))) if len(rows) > 1 else 1.0
        parts.append(c / (sd if sd > 0 else 1.0))
    if not parts:
        return pd.DataFrame(index=rows.index)
    tot = pd.concat(parts).groupby(level=0).sum() / len(parts)
    return tot.reindex(rows.index).fillna(0.0)


def _top(contrib: pd.Series, lang: str) -> list[dict[str, Any]]:
    s = contrib[contrib.abs() > 0].sort_values(key=lambda x: -x.abs()).head(TOP_CONTRIB)
    return [{"feature": k, "label": feat.feature_label(k, lang), "value": float(v)} for k, v in s.items()]


def build_artifact(result: SelectionResult, ds: ModelingDataset, settings: Settings) -> ServingArtifact:
    """Serving artifact from the selection result's final context (see module docstring)."""
    fin = result.final
    if fin is None:
        raise ValueError("selection result has no final context")
    sel = result.selected
    members = composite_members(sel, fin.configs)
    latest_all = ds.frame[ds.frame["date"] == ds.as_of].copy()
    preds = fin.predictions.set_index("entity")
    latest_all["score"] = preds[sel].reindex(latest_all["entity"]).to_numpy()
    latest = latest_all[np.isfinite(latest_all["score"].to_numpy(dtype=float))].copy()
    latest["z"] = feat.rank_gauss(latest["score"], latest["date"])
    latest["percentile"] = feat.uniform_rank(latest["score"], latest["date"])
    cal = result.calibration
    gate_passed = bool(result.gate.get("passed"))
    post_hoc = bool(result.gate.get("post_hoc"))
    skill_ok = gate_passed and not post_hoc  # a post-hoc pass needs confirmation (scoring time, served_state)
    mult = float(cal.get("shrink") or 0.0) * (1.0 if skill_ok else UNSKILLED_SHRINK)
    latest["mu_excess_raw"] = float(cal.get("shrink") or 0.0) * float(cal.get("slope") or 0.0) * latest["z"]
    latest["mu_excess"] = mult * float(cal.get("slope") or 0.0) * latest["z"]
    stack_base = fin.stack_base
    stacked = any(m in STACKED for m in members)  # rsmt_v3 reads its stack base's score (stack_z)
    need = sorted({*members, *([stack_base] if stacked and stack_base else [])})
    member_scores = pd.DataFrame({m: preds[m].reindex(latest["entity"]).to_numpy() for m in need}, index=latest.index)
    rows = latest.copy()
    if stacked and stack_base:
        rows["stack_z"] = feat.rank_gauss(member_scores[stack_base], rows["date"])
    contrib = _contributions(fin.fitted, members, rows, list(ds.xs_cols))
    lang = settings.report.language
    latest["top_contrib"] = [_top(contrib.loc[i], lang) if i in contrib.index else [] for i in latest.index]
    cand = result.candidate(sel)
    raw_cols = [*feat.XS_FEATURES]
    keep = [
        "date",
        "entity",
        "ticker",
        "name",
        "sector",
        "sector_group",
        "score",
        "z",
        "percentile",
        "mu_excess",
        "mu_excess_raw",
        "top_contrib",
        "beta_s",
        *raw_cols,
        *(getattr(ds, "alt_raw", None) or []),  # v3 raw alternative-data / macro-exposure values (display)
        *ds.xs_cols,
        *(["stack_z"] if "stack_z" in rows else []),
    ]
    if "stack_z" in rows:
        latest["stack_z"] = rows["stack_z"]
    latest = latest[[c for c in keep if c in latest]].reset_index(drop=True)
    member_scores = member_scores.reset_index(drop=True)
    regime_latest = {c: float(v) for c, v in ds.frame.loc[ds.frame["date"] == ds.as_of, ds.regime_cols].iloc[0].items()}
    o, h = cand["outer"], cand["holdout"] or {}
    metrics = {
        # protocol v4 PRIMARY (per month): top-quintile beta-adjusted net excess (docs/MODELING.md 79)
        "primary": PRIMARY,
        "primary_label_ko": metric_label(PRIMARY, "ko"),
        "primary_label_en": metric_label(PRIMARY, "en"),
        "outer_primary": o.get(PRIMARY),
        "outer_primary_ann": o.get(f"{PRIMARY}_ann"),
        "holdout_primary": h.get(PRIMARY),
        "holdout_primary_ann": h.get(f"{PRIMARY}_ann"),
        # "outer_auc" / "holdout_auc" = the reported AUC (``PRIMARY_AUC``; the v2/v3 primary, secondary since v4)
        "primary_auc": PRIMARY_AUC,
        "auc_label_ko": auc_label(PRIMARY_AUC, "ko"),
        "auc_label_en": auc_label(PRIMARY_AUC, "en"),
        "outer_auc": o.get(PRIMARY_AUC),
        "outer_auc_secondary": o.get(SECONDARY_AUC),
        "outer_ic": o["ic"],
        "outer_top_net_ann": o.get("top_net_ann"),
        "holdout_auc": h.get(PRIMARY_AUC),
        "holdout_auc_secondary": h.get(SECONDARY_AUC),
        "holdout_ic": h.get("ic"),
        "holdout_top_net_ann": h.get("top_net_ann"),
        "outer_first": o.get("first_date"),
        "outer_last": o.get("last_date"),
    }
    return ServingArtifact(
        market=ds.market.value,
        as_of=pd.Timestamp(ds.as_of),
        created_at=datetime.now().isoformat(timespec="seconds"),
        selected=sel,
        label_ko=cand["label_ko"],
        label_en=cand["label_en"],
        members=members,
        fitted={m: fin.fitted[m] for m in need if m in fin.fitted},
        stack_base=stack_base,
        gate=dict(result.gate),
        calibration=dict(cal),
        skill_ok=skill_ok,
        multiplier=mult,
        latest=latest,
        member_scores=member_scores,
        xs_cols=list(ds.xs_cols),
        regime_cols=list(ds.regime_cols),
        regime_latest=regime_latest,
        raw_cols=raw_cols,
        metrics=metrics,
        market_model=dict(result.market_model),
        importances=dict(result.importances),
        horizon=ds.horizon,
        notes=[],
        gate_passed=gate_passed,
        post_hoc=post_hoc,
        primary_auc=PRIMARY_AUC,
        member_weights=[float(x) for x in ((fin.configs.get(sel) or {}).get("weights") or [])],
        alt_raw=list(getattr(ds, "alt_raw", None) or []),
    )


def save_artifact(art: ServingArtifact, settings: Settings, cache: DiskCache) -> None:
    cache.set(SERVE_NS, _key(art.market, settings), art)


def load_artifact(market: Market | str, settings: Settings, cache: DiskCache) -> ServingArtifact | None:
    hit = cache.get(SERVE_NS, _key(market, settings), None)
    return hit if isinstance(hit, ServingArtifact) else None


# ------------------------------------------------------------------------------------------------ report time


def _match(art: ServingArtifact, ticker: str) -> int | None:
    t = normalise_ticker(ticker)
    lat = art.latest
    if Market(art.market) is Market.KR:
        ent = lat["entity"].astype(str)
        hit = lat.index[(ent == t) | ent.str.startswith(f"{t}#")]
    else:
        tick = lat["ticker"].astype(str).map(normalise_ticker)
        hit = lat.index[tick == t]
    return int(hit[-1]) if len(hit) else None


def _outside_alt(art: ServingArtifact, md: MarketData, settings: Settings, cache: DiskCache) -> dict[str, float]:
    """Raw v3 features (alternative data of this one name + its macro betas) at the artifact date, computed with
    the dataset's own functions (same point-in-time rule: published < as_of). {} when nothing applies."""
    from quantcio.data.alt.load import load_alt_data, nps_included
    from quantcio.modeling import altfeatures as altf
    from quantcio.modeling.dataset import MACRO_LAG_DAYS

    out: dict[str, float] = {}
    if not art.alt_raw:
        return out
    mk = Market(art.market)
    px = md.prices.loc[: art.as_of]
    if px.empty:
        return out
    close = float(px["close"].iloc[-1])
    alt_names = [c for c in art.alt_raw if c not in altf.MACRO_BETA_FEATURES]
    if alt_names and settings.alt.enabled:
        if mk is Market.KR:
            entity = normalise_ticker(md.instrument.ticker)
        else:
            from quantcio.data.pit.us_cik import cik_entity, current_company_tickers
            from quantcio.data.pit.us_net import PitNet

            hit = [c for t, c, _ in current_company_tickers(PitNet(settings)) if t == md.instrument.ticker.upper()]
            entity = cik_entity(hit[0]) if hit else ""
        if entity:
            start = (art.as_of - pd.Timedelta(800, "D")).date()
            alt = load_alt_data(mk, settings, [entity], start, names={entity: [md.instrument.name]})
            rows = pd.DataFrame(
                {"date": [art.as_of], "entity": [entity], "mcap": [md.fundamentals.market_cap], "close": [close]}
            )
            vals = altf.alt_features(rows, alt, mk.value, nps_included(alt))
            out.update({c: float(vals[c].iloc[0]) for c in alt_names if c in vals})
    betas = [c for c in art.alt_raw if c in altf.MACRO_BETA_FEATURES]
    if betas and md.benchmark is not None and len(md.benchmark):
        from quantcio.macro import load_macro_series

        series = load_macro_series(settings, cache)
        fac = altf.factor_changes(series, mk.value, pd.DatetimeIndex(px.index), MACRO_LAG_DAYS[mk])
        mb = altf.macro_betas(px["close"].astype(float), md.benchmark["close"].astype(float), fac)
        out.update({c: float(mb[c].iloc[-1]) for c in betas})
    return out


def _outside_row(
    art: ServingArtifact, md: MarketData, extra: dict[str, float] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Model inputs of a name outside the PIT universe: its own price features at the artifact date (+ current
    fundamentals, + ``extra`` raw v3 values), each placed into the latest cross-section's distribution."""
    missing: list[str] = []
    px = md.prices.loc[: art.as_of]
    raw: dict[str, float] = dict.fromkeys(art.raw_cols, np.nan)
    if len(px) > 20:
        f = feat.entity_price_features(px, md.benchmark)
        last = f.iloc[-1]
        for c in feat.PRICE_FEATURES:
            raw[c] = float(last.get(c, np.nan))
        adv20 = float(last.get("adv20", np.nan))
    else:
        adv20 = float("nan")
    fu = md.fundamentals
    mcap = fu.market_cap
    if mcap and mcap > 0:
        raw["log_mcap"] = math.log(mcap)
        if math.isfinite(adv20) and adv20 > 0:
            raw["turnover"] = math.log(adv20 / mcap)
    if fu.per:
        raw["ep"] = 1.0 / fu.per
    if fu.pbr:
        raw["bp"] = 1.0 / fu.pbr
    raw["roe"] = fu.roe if fu.roe is not None else np.nan
    raw["op_margin"] = fu.operating_margin if fu.operating_margin is not None else np.nan
    raw["rev_yoy"] = fu.revenue_growth_yoy if fu.revenue_growth_yoy is not None else np.nan
    row: dict[str, float] = {}
    for c in art.raw_cols:
        col = feat.x_col(c)
        v = raw.get(c, np.nan)
        if col in art.xs_cols:
            if v is None or not math.isfinite(float(v)) or c not in art.latest:
                row[col] = np.nan
                missing.append(c)
            else:
                row[col] = _z(_percentile(art.latest[c].to_numpy(dtype=float), float(v)))
    for c in art.alt_raw:
        col = feat.x_col(c)
        if col not in art.xs_cols:
            continue
        v = (extra or {}).get(c, np.nan)
        if v is None or not math.isfinite(float(v)) or c not in art.latest:
            row[col] = np.nan
            missing.append(c)
        else:
            row[col] = _z(_percentile(art.latest[c].to_numpy(dtype=float), float(v)))
    for c in art.xs_cols:
        row.setdefault(c, np.nan)  # sector-relative ranks: sector unknown outside the universe
    row.update(art.regime_latest)
    return pd.DataFrame([row]), missing


def _score_outside(art: ServingArtifact, row: pd.DataFrame) -> tuple[float, float]:
    """(percentile, z) of an outside name within the latest cross-section of the served score."""
    lat = art.latest
    if art.selected in COMPOSITES:
        pcts = []
        for m in art.members:
            s = float(art.fitted[m].predict(row)[0])
            pcts.append(_percentile(art.member_scores[m].to_numpy(dtype=float), s))
        # the composite score is the (weighted) mean member percentile; place it among the latest composite scores
        w = np.asarray(art.member_weights, dtype=float) if len(art.member_weights) == len(pcts) else None
        mean_pct = float(np.average(pcts, weights=w)) if w is not None and w.sum() > 0 else float(np.mean(pcts))
        pct = _percentile(lat["score"].to_numpy(dtype=float), mean_pct)
    else:
        r = row.copy()
        if art.selected in STACKED and art.stack_base:
            sb = float(art.fitted[art.stack_base].predict(row)[0])
            r["stack_z"] = _z(_percentile(art.member_scores[art.stack_base].to_numpy(dtype=float), sb))
        s = float(art.fitted[art.selected].predict(r)[0])
        pct = _percentile(lat["score"].to_numpy(dtype=float), s)
    return pct, _z(pct)


def lookup_confirmation(market: Market | str, settings: Settings) -> tuple[bool, dict[str, Any] | None, str]:
    """(confirmed, confirmation dict, reason) from ``quantcio.holdout.xsh_confirmation(market, settings)``.

    Looked up lazily at scoring time. Not confirmed when the function does not exist, returns None / a non-dict,
    a dict whose ``confirmed`` or ``passed`` is False, or raises (the reason says which)."""
    try:
        from quantcio import holdout as holdout_module

        fn = getattr(holdout_module, "xsh_confirmation", None)
        if fn is None:
            return False, None, "no xsh_confirmation"
        conf = fn(Market(market), settings)
    except Exception as exc:  # noqa: BLE001 - any failure = not confirmed (full serving needs a positive answer)
        log.warning("modeling: xsh_confirmation failed: %s", exc)
        return False, None, f"xsh_confirmation failed ({type(exc).__name__})"
    if not isinstance(conf, dict) or not conf:
        return False, None, "no confirmed XSH look"
    if conf.get("confirmed") is False or conf.get("passed") is False:
        return False, conf, "XSH look did not confirm"
    return True, conf, "confirmed"


def served_state(art: ServingArtifact, settings: Settings) -> dict[str, Any]:
    """The SERVED skill flag and multiplier of an artifact (module docstring): full only when the gate passed
    and (the label is pre-declared, or the result is confirmed on untouched names at scoring time)."""
    shrink = float(art.calibration.get("shrink") or 0.0)
    gate_passed = bool(getattr(art, "gate_passed", art.skill_ok))
    post_hoc = bool(getattr(art, "post_hoc", False))
    confirmed, conf, reason = (False, None, "gate failed")
    if gate_passed and post_hoc:
        confirmed, conf, reason = lookup_confirmation(art.market, settings)
    if not gate_passed:
        status = "gate_failed"
    elif not post_hoc:
        status = "passed"
    else:
        status = "confirmed" if confirmed else "unconfirmed"
    ok = status in ("passed", "confirmed")
    return {
        "status": status,
        "skill_ok": ok,
        "multiplier": shrink * (1.0 if ok else UNSKILLED_SHRINK),
        "gate_passed": gate_passed,
        "post_hoc": post_hoc,
        "confirmed": bool(confirmed),
        "confirmation": conf,
        "reason": reason,
    }


def gate_verdict(state: dict[str, Any]) -> tuple[str, str]:
    """(Korean, English) one-word-ish verdict of a served state."""
    return {
        "gate_failed": ("미달", "not passed"),
        "passed": ("통과", "passed"),
        "confirmed": ("통과(사후 지표, 미열람 자료로 확인됨)", "passed (post-hoc metric, confirmed on untouched data)"),
        "unconfirmed": (
            "통과(사후 지표) — 미열람 자료 확인 전 ×0.35",
            "passed (post-hoc metric) - x0.35 until confirmed on untouched data",
        ),
    }[state["status"]]


def _gate_note(art: ServingArtifact, state: dict[str, Any], settings: Settings) -> str | None:
    """The ticker note of a served state that is not the full multiplier (None when the skill is established)."""
    if state["status"] == "unconfirmed":
        return tr(
            settings,
            f"게이트 통과(사후 지표) — 미열람 자료(전진 홀드아웃) 확인 전까지 ×{UNSKILLED_SHRINK:g}",
            f"gate passed (post-hoc metric) - x{UNSKILLED_SHRINK:g} until confirmed on untouched data "
            "(forward holdout)",
        )
    if state["status"] == "gate_failed":
        g = art.gate
        key = str(g.get("metric") or getattr(art, "primary_auc", "auc"))
        null = g.get("null", 0.5)
        ci_ok = g.get("lo95_above_null", g.get("auc_lo95_above_half"))
        return tr(
            settings,
            f"스킬 게이트 미달(모멘텀 기준 초과 {g.get('beats_momentum')}, {metric_label(key, 'ko')} 하한>{null:g} "
            f"{ci_ok}) → 초과수익 기대치 ×{UNSKILLED_SHRINK:g}",
            f"skill gate failed (beats momentum {g.get('beats_momentum')}, {metric_label(key, 'en')} lower "
            f"bound > {null:g} {ci_ok}) -> expected excess x{UNSKILLED_SHRINK:g}",
        )
    return None


def score_for_report(
    market: Market,
    runs: list[tuple[str, MarketData]],
    settings: Settings,
    cache: DiskCache,
) -> tuple[dict[str, ExcessForecast], list[str], ServingArtifact | None]:
    """ExcessForecast per requested ticker of ``market`` from the cached artifact (+ run-level notes).

    Never raises for a missing artifact or an unmatched ticker: the note says what happened (Korean / English
    per ``settings.report.language``).
    """
    notes: list[str] = []
    art = load_artifact(market, settings, cache)
    if art is None:
        notes.append(
            tr(
                settings,
                f"종목 선택 모델({market.value}) 서빙 산출물 없음 — `quantcio modeling select --market {market.value}` "
                "실행 전까지 RSMT-Q 평균 사용",
                f"No stock-selection model artifact for {market.value} - run `quantcio modeling select --market "
                f"{market.value}`; the RSMT-Q mean is used until then",
            )
        )
        return {}, notes, None
    age = (pd.Timestamp(date.today()) - art.as_of).days
    if age > STALE_DAYS:
        notes.append(
            tr(
                settings,
                f"종목 선택 모델({market.value}) 기준일 {art.as_of.date()} — {age}일 경과(재선택·갱신 권장)",
                f"Stock-selection model ({market.value}) as of {art.as_of.date()} - {age} days old (re-run advised)",
            )
        )
    lang = settings.report.language
    state = served_state(art, settings)
    # the returned artifact carries the SERVED flag / multiplier (a fresh object: the cache is not changed)
    art.skill_ok = bool(state["skill_ok"])
    art.multiplier = float(state["multiplier"])
    vko, ven = gate_verdict(state)
    art.metrics = {**art.metrics, "gate_verdict_ko": vko, "gate_verdict_en": ven, "served_status": state["status"]}
    gate_note = _gate_note(art, state, settings)
    m = art.metrics
    mm = art.market_model or {}
    out: dict[str, ExcessForecast] = {}
    for ticker, md in runs:
        i = _match(art, ticker)
        tnotes: list[str] = []
        if i is not None:
            r = art.latest.loc[i]
            pct, z = float(r["percentile"]), float(r["z"])
            top = list(r["top_contrib"])
            inside = True
        else:
            try:
                extra: dict[str, float] = {}
                try:
                    extra = _outside_alt(art, md, settings, cache)
                except Exception as exc:  # noqa: BLE001 - the v3 inputs stay at the median, said in the note
                    log.warning("modeling %s: alternative data for %s failed: %s", market.value, ticker, exc)
                row, missing = _outside_row(art, md, extra)
                pct, z = _score_outside(art, row)
                top = []
                inside = False
                tnotes.append(
                    tr(
                        settings,
                        f"PIT 유니버스 밖 종목: 자체 가격·현재 재무를 {art.as_of.date()} 횡단면 분포에 배치해 점수화"
                        + (f"(결측 {len(missing)}개 특징은 중앙값)" if missing else ""),
                        f"outside the PIT universe: own prices / current fundamentals placed into the "
                        f"{art.as_of.date()} cross-section"
                        + (f" ({len(missing)} missing features at the median)" if missing else ""),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - reported, the legacy path serves this ticker
                log.warning("modeling %s: outside-universe scoring failed for %s: %s", market.value, ticker, exc)
                notes.append(
                    tr(
                        settings,
                        f"{ticker}: 종목 선택 모델 점수화 실패({type(exc).__name__}) — 기존 RSMT-Q 평균 사용",
                        f"{ticker}: stock-selection scoring failed ({type(exc).__name__}) - legacy RSMT-Q mean used",
                    )
                )
                continue
        slope = float(art.calibration.get("slope") or 0.0)
        shrink = float(art.calibration.get("shrink") or 0.0)
        mu_raw = shrink * slope * z
        mu = art.multiplier * slope * z
        if gate_note:
            tnotes.append(gate_note)
        out[ticker] = ExcessForecast(
            model=art.selected,
            model_label=art.label_en if lang == "en" else art.label_ko,
            as_of=art.as_of.date(),
            horizon_days=art.horizon,
            in_universe=inside,
            mu_excess=float(mu),
            mu_excess_raw=float(mu_raw),
            score_z=float(z),
            percentile=float(min(max(pct, 0.0), 1.0)),
            skill_ok=art.skill_ok,
            gate={
                **{k: v for k, v in art.gate.items() if isinstance(v, bool | int | float) or v is None},
                "gate_passed": state["gate_passed"],
                "confirmed": state["confirmed"],
            },
            oos_primary=(m.get("outer_primary") or {}).get("mean"),
            oos_primary_lo=(m.get("outer_primary") or {}).get("lo"),
            holdout_primary=(m.get("holdout_primary") or {}).get("mean"),
            oos_auc=(m.get("outer_auc") or {}).get("mean"),
            oos_auc_lo=(m.get("outer_auc") or {}).get("lo"),
            oos_ic=(m.get("outer_ic") or {}).get("mean"),
            holdout_auc=(m.get("holdout_auc") or {}).get("mean"),
            holdout_ic=(m.get("holdout_ic") or {}).get("mean"),
            market_p_up=mm.get("served_p_up"),
            market_model_used=bool(mm.get("use_model")),
            top_contributions=top,
            notes=tnotes,
        )
    return out, notes, art


def excess_override(fc: ExcessForecast) -> Any:
    """The ``quant.rsmt.ExcessOverride`` of one forecast (market leg: RSMT-Q prior-shrunk drift unless the
    market-direction model passed its rule - then P(market up) is matched by the leg's mean)."""
    from quantcio.quant.rsmt import ExcessOverride

    p = fc.market_p_up if fc.market_model_used else None
    return ExcessOverride(
        mu_excess=float(fc.mu_excess),
        skill_ok=bool(fc.skill_ok),
        model=fc.model,
        market_p_up=p,
        metrics={
            "excess_oos_primary": fc.oos_primary,
            "excess_holdout_primary": fc.holdout_primary,
            "excess_oos_auc": fc.oos_auc,
            "excess_oos_auc_lo": fc.oos_auc_lo,
            "excess_oos_ic": fc.oos_ic,
            "excess_holdout_auc": fc.holdout_auc,
            "excess_percentile": fc.percentile,
        },
    )


# ------------------------------------------------------------------------------------------------ ordering ventile
# docs/MODELING.md 93 (user request 2026-11-06): the 20-quantile ordering model of 91 (`lgbm_reg_w_ord`, a research
# candidate chosen post hoc, unconfirmed) is shown in the report as a REFERENCE: each name's score ventile within
# the artifact's PIT cross-section and the cross-section's top ventile (D20). It sets nothing - not the ML
# direction, the served excess mean, the signal or the sizing. Its artifact is a ServingArtifact built from that
# candidate's final fit (scripts/serve_order.py), cached under its own namespace next to the served one.
# docs/MODELING.md 94.3 (user request 2026-11-07): several reference models per market (one artifact per model,
# an index keeps their display order), e.g. the random forest of 94 next to the ordering model.

ORDER_NS = "modeling_order"
ORDER_K = 20


def _order_key(market: Market | str, settings: Settings, *parts: str) -> str:
    return DiskCache.key(ORDER_NS, ARTIFACT_VERSION, Market(market).value, int(settings.horizon_days), *parts)


def ventile_of(p: float, k: int = ORDER_K) -> int:
    """Score quantile 1..k of a within-date percentile p = (rank - 0.5)/n: floor(k p) + 1, clipped (88.2's bins)."""
    return int(min(max(math.floor(float(p) * k), 0), k - 1)) + 1


def order_models(market: Market | str, settings: Settings, cache: DiskCache) -> list[str]:
    """The reference models of ``market`` in display order (the first saved first)."""
    hit = cache.get(ORDER_NS, _order_key(market, settings, "index"), None)
    return [str(x) for x in hit] if isinstance(hit, list) else []


def save_order_artifact(art: ServingArtifact, settings: Settings, cache: DiskCache) -> None:
    """Store ``art`` under its model name (``art.selected``) and append the name to the market's index."""
    cache.set(ORDER_NS, _order_key(art.market, settings, "model", art.selected), art)
    names = order_models(art.market, settings, cache)
    if art.selected not in names:
        cache.set(ORDER_NS, _order_key(art.market, settings, "index"), [*names, art.selected])


def remove_order_artifact(market: Market | str, model: str, settings: Settings, cache: DiskCache) -> bool:
    """Take ``model`` out of the market's index (the report stops showing it); False if it was not there."""
    names = order_models(market, settings, cache)
    if model not in names:
        return False
    cache.set(ORDER_NS, _order_key(market, settings, "index"), [n for n in names if n != model])
    return True


def load_order_artifact(
    market: Market | str, settings: Settings, cache: DiskCache, model: str | None = None
) -> ServingArtifact | None:
    """The artifact of ``model`` (default: the first indexed model); None if it is not indexed or not stored."""
    names = order_models(market, settings, cache)
    name = model if model is not None else (names[0] if names else None)
    if name is None or name not in names:
        return None
    hit = cache.get(ORDER_NS, _order_key(market, settings, "model", name), None)
    return hit if isinstance(hit, ServingArtifact) else None


def order_for_report(
    market: Market,
    runs: list[tuple[str, MarketData]],
    settings: Settings,
    cache: DiskCache,
) -> list[dict[str, Any]]:
    """The reference views of ``market``, one per indexed model in display order (empty without any artifact).
    A model whose artifact cannot be read or used is left out with a warning."""
    out = []
    for name in order_models(market, settings, cache):
        try:
            art = load_order_artifact(market, settings, cache, name)
            if art is not None:
                out.append(_order_view(market, art, runs, settings, cache))
        except Exception as exc:  # noqa: BLE001 - reference only: the model is left out
            log.warning("ordering model %s/%s failed: %s", market.value, name, exc)
    return out


def _order_view(
    market: Market,
    art: ServingArtifact,
    runs: list[tuple[str, MarketData]],
    settings: Settings,
    cache: DiskCache,
) -> dict[str, Any]:
    """One reference model's view: per requested ticker the score ventile (1 = bottom 5%, 20 = top 5%) within the
    artifact's cross-section - names outside the PIT universe are placed into it from their own prices like
    ``score_for_report`` - and the cross-section's top-ventile names. Never raises for one ticker (an unscorable
    name is left out)."""
    lat = art.latest
    tick: dict[str, dict[str, Any]] = {}
    for ticker, md in runs:
        i = _match(art, ticker)
        if i is not None:
            pct, inside = float(lat.loc[i, "percentile"]), True
        else:
            try:
                extra: dict[str, float] = {}
                try:
                    extra = _outside_alt(art, md, settings, cache)
                except Exception as exc:  # noqa: BLE001 - the v3 inputs stay at the median
                    log.warning(
                        "ordering model %s/%s: alternative data for %s failed: %s",
                        market.value,
                        art.selected,
                        ticker,
                        exc,
                    )
                row, _missing = _outside_row(art, md, extra)
                pct = _score_outside(art, row)[0]
                inside = False
            except Exception as exc:  # noqa: BLE001 - reference only: the name is left out
                log.warning(
                    "ordering model %s/%s: outside-universe scoring failed for %s: %s",
                    market.value,
                    art.selected,
                    ticker,
                    exc,
                )
                continue
        pct = float(min(max(pct, 0.0), 1.0))
        tick[ticker] = {"ventile": ventile_of(pct), "percentile": pct, "in_universe": inside}
    bins = lat["percentile"].map(ventile_of)
    top = lat[bins == ORDER_K].sort_values("score", ascending=False)

    def txt(v: Any) -> str:
        return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)

    def code(r: Any) -> str:
        return txt(getattr(r, "ticker", None)) or str(r.entity).split("#")[0]

    return {
        "market": market.value,
        "model": art.selected,
        "label_ko": art.label_ko,
        "label_en": art.label_en,
        "as_of": str(art.as_of.date()),
        "created_at": art.created_at,
        "age_days": int((pd.Timestamp(date.today()) - art.as_of).days),
        "n": int(len(lat)),
        "k": ORDER_K,
        "horizon": int(art.horizon),
        "stats": dict(art.metrics.get("ventile") or {}),
        "tickers": tick,
        "top": [
            {
                "ticker": code(r),
                "name": txt(getattr(r, "name", None)),
                "sector": txt(getattr(r, "sector", None)),
                "percentile": float(r.percentile),
            }
            for r in top.itertuples(index=False)
        ],
    }


__all__ = [
    "ORDER_K",
    "ServingArtifact",
    "build_artifact",
    "excess_override",
    "gate_verdict",
    "load_artifact",
    "load_order_artifact",
    "order_for_report",
    "order_models",
    "remove_order_artifact",
    "save_artifact",
    "save_order_artifact",
    "score_for_report",
    "served_state",
    "lookup_confirmation",
    "ventile_of",
]
