"""Forward-tracking ledger (docs/HOLDOUT.md "전진 추적 원장"): record predictions, score them, judge them.

Public API
----------
- ``record_predictions(report, settings, source='report'|'snapshot', holdout_id=None, *, models=None)
  -> RecordResult`` - one ``predictions.jsonl`` line per ticker (``id = sha1(run_id|ticker)``; a line whose id
  exists is skipped, so recording the same report twice is a no-op). ``holdout_id`` defaults to the latest
  forward registration covering the ticker's market; ``valid`` = the current config digest and signal-code
  fingerprint equal that registration's (and the run is not older than the registration). ``models``
  (market -> fitted ``MarketModel``) adds ``dist.base_prob_up``, the root-only / no-offset twin's P(up) that
  the direction metric compares against (the same twin as the CV skill gate).
- ``snapshot(market, settings, n=None) -> RecordResult`` - serves the whole market universe WITHOUT text/LLM
  through ``pipeline.build_report`` (fit_market -> predict_many -> build_quant -> fuse -> execution plan ->
  barrier guardrail -> sizing; nothing re-implemented) and records it with ``source='snapshot'``.
- ``score(settings, now=None) -> ScoreResult`` - fetches fresh prices from the production providers and
  appends ``outcomes.jsonl`` lines: stage ``h`` once ``horizon_days`` sessions after ``as_of`` exist (r_h, m_h,
  y_e, up, beat, in80, exc95), stage ``barrier`` once ``hold_days`` sessions exist (tp/sl/time first touch on
  daily closes, the rule of ``ReturnDistribution.barrier_trades``), or one ``final`` line when both are due.
  Immature rows are skipped; (id, stage) is written once (idempotent).
- ``metrics(settings, holdout_id=None) -> dict`` - the five pre-registered primary metrics with
  Benjamini-Hochberg over them, the non-overlap rule and '판정 불가(표본 부족)' below ``min_dates_for_verdict``
  non-overlapping dates; ``report_lines(result)`` renders it in Korean.

Realised quantities use the FRESH series at both ends (the close at ``as_of`` is re-read), so a split or
dividend adjustment made after the prediction does not create a fake return; a gap to the recorded
``last_price`` above 2% is kept in the outcome (``entry_adjust``) and counted in the score notes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from quantcio import holdout
from quantcio.config import Settings
from quantcio.data import base as data_base
from quantcio.data.cache import DiskCache
from quantcio.models import Market, PortfolioReport, Signal, TickerReport

if TYPE_CHECKING:
    from quantcio.quant.rsmt import MarketModel

log = logging.getLogger(__name__)

PREDICTIONS_FILE = "predictions.jsonl"
OUTCOMES_FILE = "outcomes.jsonl"
SOURCES = ("report", "snapshot")
BUY_SIGNALS = {Signal.BUY.value, Signal.STRONG_BUY.value}
ADJUST_TOL = 0.02  # |fresh close at as_of / recorded last_price - 1| above this is reported
PRICE_LOOKBACK_DAYS = 14
VERDICT_NA = "판정 불가(표본 부족)"
VERDICT_NO_DATA = "판정 불가(데이터 없음)"
VERDICT_EVIDENCE = "근거 있음"
VERDICT_BH_FAIL = "기준 충족·다중검정 후 비유의"
VERDICT_NONE = "근거 없음"


class TrackingError(RuntimeError):
    """Ledger failure (Korean message)."""


# =============================================================================== ledger io


def ledger_paths(settings: Settings) -> tuple[Path, Path]:
    d = holdout.ledger_dir(settings)
    return d / PREDICTIONS_FILE, d / OUTCOMES_FILE


def prediction_id(run_id: str, ticker: str) -> str:
    return hashlib.sha1(f"{run_id}|{ticker}".encode()).hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """(rows, number of unreadable lines); a missing file is empty. Bad lines are logged, never dropped silently."""
    if not path.exists():
        return [], 0
    rows: list[dict[str, Any]] = []
    bad = 0
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                log.warning("ledger %s line %d unreadable", path.name, i)
                continue
            if isinstance(obj, dict) and "id" in obj:
                rows.append(obj)
            else:
                bad += 1
    return rows, bad


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    """Append rows (one JSON object per line, fsync'ed); returns the number written."""
    if not rows:
        return 0
    holdout.ensure_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(holdout._jsonable(dict(r)), ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return len(rows)


def load_predictions(settings: Settings) -> tuple[list[dict[str, Any]], int]:
    """Predictions in file order, first line per id (a duplicate id would be a bug; it is counted as bad)."""
    rows, bad = read_jsonl(ledger_paths(settings)[0])
    seen: set[str] = set()
    out = []
    for r in rows:
        if r["id"] in seen:
            bad += 1
            continue
        seen.add(r["id"])
        out.append(r)
    return out, bad


def load_outcomes(settings: Settings) -> tuple[dict[str, dict[str, Any]], int]:
    """id -> merged outcome (stage lines applied in file order; ``stages`` lists the ones present)."""
    rows, bad = read_jsonl(ledger_paths(settings)[1])
    merged: dict[str, dict[str, Any]] = {}
    for r in rows:
        cur = merged.setdefault(r["id"], {"id": r["id"], "stages": []})
        stage = r.get("stage", "final")
        if stage in cur["stages"]:
            continue
        cur["stages"].append(stage)
        cur.update({k: v for k, v in r.items() if k not in ("id", "stage")})
    return merged, bad


def _has_h(o: Mapping[str, Any] | None) -> bool:
    return bool(o) and any(s in o["stages"] for s in ("h", "final"))


def _has_barrier(o: Mapping[str, Any] | None) -> bool:
    return bool(o) and any(s in o["stages"] for s in ("barrier", "final"))


# =============================================================================== record


NO_RECORD_ENV = "QUANTCIO_NO_RECORD"  # =1 for dev/verification runs: skip the forward ledger


@dataclass
class RecordResult:
    path: Path
    run_id: str
    n_new: int = 0
    n_existing: int = 0
    holdout_ids: dict[str, str | None] = field(default_factory=dict)  # market -> id
    valid: dict[str, bool] = field(default_factory=dict)  # market -> valid
    notes: list[str] = field(default_factory=list)  # Korean


def _fin(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _not_before(when: Any, ref: Any) -> bool:
    """Timestamp `when` >= `ref` (ISO strings with offsets compared as instants; unparseable -> False)."""
    try:
        a, b = pd.Timestamp(when), pd.Timestamp(ref)
        if (a.tzinfo is None) != (b.tzinfo is None):
            a, b = (x.tz_localize(holdout.KST) if x.tzinfo is None else x for x in (a, b))
        return bool(a >= b)
    except (TypeError, ValueError):
        return False


def run_id_for(report: PortfolioReport, source: str) -> str:
    return f"{source}:{report.generated_at.isoformat()}:{report.config_digest}"


def root_twin_prob_up(model: MarketModel, tickers: Sequence[str], settings: Settings) -> dict[str, float]:
    """P(up) of the root-only, no-offset twin (the CV skill gate's baseline) for served tickers.

    Same market leg (``MarketModel._market_leg``), beta, path seed and plausibility clamp as the served
    forecast; only the tree (a depth-0 fit on the same training rows) and the offsets differ, exactly like
    the CV twin (``rsmt._cross_validate_layout``). Tickers missing from the prediction panel are left out.
    """
    from quantcio.quant import rsmt
    from quantcio.quant.distribution import path_seed
    from quantcio.quant.panel import on_grid_rows

    panel = model.panel
    pred = panel.pred.set_index("ticker", drop=False)
    keep = [t for t in dict.fromkeys(tickers) if t in pred.index]
    if not keep:
        return {}
    # the CV's baseline twin exactly (root only, no offsets, raw target, non-overlapping rows; docs/RSMT.md)
    base = rsmt.RSMTree.from_settings(settings, panel.market, **rsmt.BASE_TWIN).fit_frame(on_grid_rows(panel.train))
    rows = pred.loc[keep]
    prm = base.predict_params(rows)
    legs = [model._market_leg(rows.iloc[i], t) for i, t in enumerate(keep)]
    params = {
        "mu_e": np.asarray(prm["mu_e"], dtype="float64"),
        "sigma_e": np.asarray(prm["sigma_e"], dtype="float64"),
        "nu": np.asarray(prm["nu"], dtype="float64"),
        "rho": np.asarray(prm["rho"], dtype="float64"),
        "beta": rows["beta"].to_numpy(dtype="float64"),
        "mu_m": np.array([lg[0] for lg in legs]),
        "nu_m": np.array([lg[1] for lg in legs]),
        "sigma_m": np.array([lg[2] for lg in legs]),
    }
    seeds = [path_seed(t, pd.Timestamp(d).date().isoformat()) for t, d in zip(keep, rows["date"])]
    ev = rsmt._evaluate(params, seeds, np.zeros(len(keep)), int(model.horizon), rsmt.CV_PATHS)
    return {t: float(p) for t, p in zip(keep, ev["p_up"])}


def _prediction_row(
    t: TickerReport,
    report: PortfolioReport,
    settings: Settings,
    *,
    run_id: str,
    source: str,
    holdout_id: str | None,
    valid: bool,
    digest: str,
    fingerprint: str,
    base_prob_up: float | None,
) -> dict[str, Any]:
    d = t.quant.distribution
    pl = t.execution
    b = pl.barrier
    ex = t.executive
    li = t.llm
    dist = None
    if d is not None:
        dist = {
            "mu": d.mu,
            "sigma": d.sigma,
            "nu": d.nu,
            "beta": d.beta,
            "mu_excess": d.mu_excess,
            "prob_up": d.prob_up,
            "prob_outperform": d.prob_outperform,
            "q10": d.q10,
            "q50": d.q50,
            "q90": d.q90,
            "var95": d.var95,
            "skill_ok": d.skill_ok,
            "degraded": d.degraded,
            "horizon_days": d.horizon_days,
            "base_prob_up": base_prob_up,
        }
    return {
        "id": prediction_id(run_id, t.instrument.ticker),
        "run_id": run_id,
        "generated_at": report.generated_at.isoformat(),
        "as_of": t.as_of.isoformat(),
        "holdout_id": holdout_id,
        "valid": bool(valid),
        "source": source,
        "ticker": t.instrument.ticker,
        "market": t.instrument.market.value,
        "last_price": t.last_price,
        "benchmark_id": t.instrument.benchmark,
        "signal": ex.signal.value,
        "composite": ex.composite,
        "confidence": ex.confidence,
        "pillars": dict(ex.pillars),
        "horizon_days": int(d.horizon_days) if d is not None else int(settings.horizon_days),
        "dist": dist,
        "plan": {
            "entry": pl.entry_price,
            "stop": pl.stop_loss,
            "take": pl.take_profit,
            "stop_pct": pl.stop_loss_pct,
            "take_pct": pl.take_profit_pct,
            "hold_days": int(b.horizon_days)
            if b is not None and b.horizon_days
            else settings.fusion.target_horizon_days,
            "p_tp": b.p_take_profit if b is not None else None,
            "p_sl": b.p_stop_loss if b is not None else None,
            "p_time": b.p_time_exit if b is not None else None,
            "weight": pl.weight,
        },
        "llm": {
            "news_score": li.news.score if li.news is not None else None,
            "filings_tone": li.filings.tone if li.filings is not None else None,
            "critic_adj": li.critic.confidence_adjustment if li.critic is not None else None,
            "llm_pillar": li.llm_pillar,
            "skipped": li.skipped,
        },
        "config_digest": digest,
        "code_fingerprint": fingerprint,
    }


def record_predictions(
    report: PortfolioReport,
    settings: Settings,
    source: str = "report",
    holdout_id: str | None = None,
    *,
    models: Mapping[str, MarketModel | None] | None = None,
    run_id: str | None = None,
) -> RecordResult:
    """Append one prediction line per ticker of `report` (idempotent by id); see the module docstring.

    Raises:
        TrackingError: unknown source / holdout id; HoldoutError: registry unreadable or write refused.
    """
    if source not in SOURCES:
        raise TrackingError(f"source는 {SOURCES} 중 하나여야 합니다: {source}")
    if os.environ.get(NO_RECORD_ENV, "").strip() not in ("", "0", "false", "False"):
        # development / verification runs must not enter the git-tracked forward ledger
        log.info("forward ledger: %s set - not recording %s", NO_RECORD_ENV, report.generated_at)
        return RecordResult(
            path=ledger_paths(settings)[0],
            run_id=run_id or "",
            notes=[f"{NO_RECORD_ENV} 설정 — 전진 추적 원장에 기록하지 않음(개발·검증 실행)"],
        )
    reg = holdout.load_registry(settings)
    explicit = None
    if holdout_id is not None:
        explicit = reg.get(holdout_id)
        if explicit["kind"] != "forward":
            raise TrackingError(f"{holdout_id}: 전진 홀드아웃(FTH)이 아닙니다")
    digest = settings.digest()
    fingerprint = holdout.code_fingerprint()
    run_id = run_id or run_id_for(report, source)
    pred_path = ledger_paths(settings)[0]
    holdout.ensure_writable(pred_path)  # fail before any work when the write would be refused
    res = RecordResult(path=pred_path, run_id=run_id)

    existing = {r["id"] for r in load_predictions(settings)[0]}
    base: dict[str, float] = {}
    for mkey, model in (models or {}).items():
        if model is None:
            continue
        tickers = [t.instrument.ticker for t in report.tickers if t.instrument.market.value == mkey]
        try:
            base.update(root_twin_prob_up(model, tickers, settings))
        except Exception as exc:  # noqa: BLE001 - the twin is optional; the direction metric skips such rows
            log.warning("tracking: root twin P(up) for %s failed: %s", mkey, exc)
            res.notes.append(f"전진 추적: {mkey} 루트 쌍둥이 P(상승) 계산 실패({type(exc).__name__}) — 방향 지표 제외")

    generated = report.generated_at.isoformat()
    rows = []
    for t in report.tickers:
        mkey = t.instrument.market.value
        if mkey not in res.holdout_ids:
            entry = explicit if explicit is not None else reg.active_forward(mkey)
            if entry is not None and mkey not in entry.get("markets", []):
                res.notes.append(f"전진 추적: {entry['id']}에 {mkey} 시장이 등록되지 않아 holdout_id=null로 기록")
                entry = None
            ok = False
            if entry is not None:
                why = []
                if entry["config_digest"] != digest:
                    why.append(f"설정 다이제스트 {entry['config_digest']}→{digest}")
                if entry["code_fingerprint"] != fingerprint:
                    why.append(f"코드 지문 {entry['code_fingerprint']}→{fingerprint}")
                if not _not_before(generated, entry["registered_at"]):
                    why.append("등록 이전 실행")
                ok = not why
                if why:
                    res.notes.append(f"전진 추적: {entry['id']} 등록값과 달라 valid=false로 기록({'; '.join(why)})")
            res.holdout_ids[mkey] = entry["id"] if entry is not None else None
            res.valid[mkey] = ok
        row = _prediction_row(
            t,
            report,
            settings,
            run_id=run_id,
            source=source,
            holdout_id=res.holdout_ids[mkey],
            valid=res.valid[mkey],
            digest=digest,
            fingerprint=fingerprint,
            base_prob_up=base.get(t.instrument.ticker),
        )
        if row["id"] in existing:
            res.n_existing += 1
            continue
        existing.add(row["id"])
        rows.append(row)
    res.n_new = append_jsonl(pred_path, rows)
    if not any(res.holdout_ids.values()):
        res.notes.append("전진 추적: 등록된 전진 홀드아웃(FTH) 없음 — holdout_id=null로 기록(판정 표본 아님)")
    log.info("tracking: %s recorded %d new / %d existing predictions", run_id, res.n_new, res.n_existing)
    return res


# =============================================================================== snapshot


def snapshot(
    market: str | Market,
    settings: Settings,
    n: int | None = None,
    *,
    cache: DiskCache | None = None,
    progress: Any = None,
) -> RecordResult:
    """Serve the market universe (top ``n``, default rsmt.markets.<M>.universe_size) with --no-llm and record it.

    Raises:
        TrackingError: the universe cannot be loaded; pipeline.PipelineError: nothing could be served.
    """
    from quantcio import pipeline

    try:
        mk = Market(str(getattr(market, "value", market)).strip().upper())
    except ValueError as exc:
        raise TrackingError(f"시장은 KR 또는 US여야 합니다: {market}") from exc
    cache = cache or DiskCache(settings.cache_path)
    size = int(n) if n else settings.rsmt.markets[mk.value].universe_size
    try:
        names = list(data_base.get_provider(mk, settings, cache).universe(size))
    except Exception as exc:  # noqa: BLE001
        raise TrackingError(f"{mk.value} 유니버스 로드 실패({type(exc).__name__}: {str(exc)[:160]})") from exc
    if not names:
        raise TrackingError(f"{mk.value} 유니버스가 비어 있습니다")
    result = asyncio.run(
        pipeline.build_report(
            names,
            settings,
            base_currency="KRW" if mk is Market.KR else "USD",
            use_llm=False,
            cache=cache,
            progress=progress,
        )
    )
    models = {k: f.model for k, f in result.fits.items() if f.model is not None}
    rec = record_predictions(result.report, settings, source="snapshot", models=models)
    served = len(result.report.tickers)
    if served < len(names):
        rec.notes.append(f"스냅샷: 유니버스 {len(names)}종목 중 {served}종목만 산출 — 나머지는 아래 리포트 알림 참조")
        rec.notes.extend(n for n in result.report.notes if n != pipeline.NO_LLM_NOTE)
    return rec


# =============================================================================== score


@dataclass
class ScoreResult:
    path: Path
    n_h: int = 0  # new horizon outcomes
    n_barrier: int = 0  # new barrier outcomes
    n_complete: int = 0  # already fully scored (skipped)
    n_immature: int = 0
    n_stale: int = 0  # past due by far but still without enough bars (halt / delisting / feed gap)
    n_failed: int = 0
    notes: list[str] = field(default_factory=list)


def _stale_days(sessions: int) -> int:
    return int(sessions * 1.6) + 20


def barrier_path(closes: np.ndarray, entry: float, stop_pct: float, take_pct: float) -> tuple[str, float, int]:
    """First touch on DAILY CLOSES (``ReturnDistribution.barrier_trades`` / ``evaluation.realised_barrier``):
    ('tp'|'sl'|'time', simple return at the exit close, bars used up to and including the exit)."""
    c = np.asarray(closes, dtype="float64")
    if c.size == 0 or not np.isfinite(c).all() or not entry > 0:
        raise ValueError("barrier_path needs finite closes and entry > 0")
    simple = c / float(entry) - 1.0
    hit_sl = simple <= stop_pct
    hit_tp = simple >= take_pct
    d = simple.size
    first_sl = int(hit_sl.argmax()) if hit_sl.any() else d
    first_tp = int(hit_tp.argmax()) if hit_tp.any() else d
    if first_tp < first_sl:
        return "tp", float(simple[first_tp]), first_tp + 1
    if first_sl < first_tp:
        return "sl", float(simple[first_sl]), first_sl + 1
    return "time", float(simple[-1]), d


def realise(
    pred: Mapping[str, Any], px: pd.DataFrame, bench: pd.DataFrame | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """(horizon fields or None if immature, barrier fields or None if immature, error or None).

    `px` / `bench` are clean price frames already cut at the scoring date.
    """
    as_of = pd.Timestamp(pred["as_of"])
    h = int(pred.get("horizon_days") or 0)
    plan = pred.get("plan") or {}
    hold = int(plan.get("hold_days") or 0)
    close = px["close"] if px is not None and len(px) else pd.Series(dtype="float64")
    before = close[close.index <= as_of]
    if before.empty:
        return None, None, f"{as_of.date()} 이전 가격 없음"
    entry = float(before.iloc[-1])
    after = close[close.index > as_of]
    h_out = None
    if h > 0 and len(after) >= h:
        end_date = after.index[h - 1]
        r_h = math.log(float(after.iloc[h - 1]) / entry)
        m_h = None
        if bench is not None and len(bench):
            bc = bench["close"]
            b0 = bc[bc.index <= as_of]
            b1 = bc[bc.index <= end_date]
            if len(b0) and len(b1) and b1.index[-1] > b0.index[-1]:
                m_h = math.log(float(b1.iloc[-1]) / float(b0.iloc[-1]))
        d = pred.get("dist") or {}
        beta = _fin(d.get("beta"))
        simple = math.expm1(r_h)
        q10, q90, var95 = _fin(d.get("q10")), _fin(d.get("q90")), _fin(d.get("var95"))
        last = _fin(pred.get("last_price"))
        h_out = {
            "realised_at": end_date.date().isoformat(),
            "h_end": end_date.date().isoformat(),
            "entry_close": entry,
            "entry_adjust": (entry / last - 1.0) if last else None,
            "r_h": r_h,
            "m_h": m_h,
            "y_e": (r_h - beta * m_h) if (beta is not None and m_h is not None) else None,
            "up": r_h > 0,
            "beat": (r_h > m_h) if m_h is not None else None,
            "in80": (q10 <= simple <= q90) if (q10 is not None and q90 is not None) else None,
            "exc95": (simple < var95) if var95 is not None else None,
        }
    b_out = None
    if hold > 0 and len(after) >= hold:
        stop_pct, take_pct = _fin(plan.get("stop_pct")), _fin(plan.get("take_pct"))
        if stop_pct is not None and take_pct is not None and stop_pct < 0 < take_pct:
            label, ret, bars = barrier_path(after.iloc[:hold].to_numpy(), entry, stop_pct, take_pct)
            b_out = {
                "barrier_outcome": label,
                "barrier_return": ret,
                "bars_used": bars,
                "barrier_end": after.index[hold - 1].date().isoformat(),
            }
        else:
            b_out = {"barrier_outcome": None, "barrier_return": None, "bars_used": 0, "barrier_note": "손절/익절 없음"}
    return h_out, b_out, None


def _fetch(market: str, ticker: str, start: date, settings: Settings, cache: DiskCache):
    provider = data_base.get_provider(Market(market), settings, cache)
    inst = provider.resolve(ticker)
    px = data_base.clean_prices(provider.prices(inst, start))
    try:
        bench = data_base.clean_prices(provider.benchmark(inst, start))
    except Exception as exc:  # noqa: BLE001 - m_h / y_e / beat become null and are counted
        log.warning("tracking: benchmark for %s failed: %s", ticker, exc)
        bench = None
    return inst, px, bench


def score(settings: Settings, now: date | datetime | None = None, *, cache: DiskCache | None = None) -> ScoreResult:
    """Score every matured prediction (see module docstring); idempotent.

    Only bars dated on or before `now` are used; the default is YESTERDAY in KST, so a still-forming bar of
    the current session (KR intraday) can never enter an outcome.
    """
    if now is None:
        cut = pd.Timestamp(datetime.now(holdout.KST).date()) - pd.Timedelta(1, "D")
    else:
        cut = pd.Timestamp(now.date() if isinstance(now, datetime) else now)
    preds, bad_p = load_predictions(settings)
    outs, bad_o = load_outcomes(settings)
    out_path = ledger_paths(settings)[1]
    res = ScoreResult(path=out_path)
    if bad_p or bad_o:
        res.notes.append(f"원장 읽기 불가 줄: 예측 {bad_p} · 결과 {bad_o} (건너뜀)")
    todo: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for p in preds:
        o = outs.get(p["id"])
        if _has_h(o) and _has_barrier(o):
            res.n_complete += 1
            continue
        todo.setdefault((p["market"], p["ticker"]), []).append(p)
    cache = cache or DiskCache(settings.cache_path)
    new_rows: list[dict[str, Any]] = []
    stamp = datetime.now(holdout.KST).replace(microsecond=0).isoformat()
    adjusted = 0
    failures: list[str] = []
    for (mkey, ticker), plist in sorted(todo.items()):
        start = (min(pd.Timestamp(p["as_of"]) for p in plist) - pd.Timedelta(PRICE_LOOKBACK_DAYS, "D")).date()
        try:
            inst, px, bench = _fetch(mkey, ticker, start, settings, cache)
        except Exception as exc:  # noqa: BLE001 - counted, retried at the next run
            res.n_failed += len(plist)
            failures.append(f"{ticker}({type(exc).__name__})")
            log.warning("tracking: prices for %s failed: %s", ticker, exc)
            continue
        px = px[px.index <= cut]
        if bench is None:
            failures.append(f"{ticker} 벤치마크 없음")
        else:
            bench = bench[bench.index <= cut]
        for p in plist:
            o = outs.get(p["id"])
            h_out, b_out, err = realise(p, px, bench)
            if err:
                res.n_failed += 1
                failures.append(f"{ticker} {err}")
                continue
            need_h, need_b = not _has_h(o), not _has_barrier(o)
            age = (cut - pd.Timestamp(p["as_of"])).days
            base = {"id": p["id"], "ticker": ticker, "market": mkey, "as_of": p["as_of"], "scored_at": stamp}
            if need_h and h_out is not None and need_b and b_out is not None:
                new_rows.append({**base, "stage": "final", **h_out, **b_out})
                res.n_h += 1
                res.n_barrier += 1
            elif need_h and h_out is not None:
                new_rows.append({**base, "stage": "h", **h_out})
                res.n_h += 1
            elif not need_h and need_b and b_out is not None:
                new_rows.append({**base, "stage": "barrier", **b_out})
                res.n_barrier += 1
            else:
                res.n_immature += 1
                h = int(p.get("horizon_days") or 0)
                hold = int((p.get("plan") or {}).get("hold_days") or 0)
                if (need_h and age > _stale_days(h)) or (not need_h and age > _stale_days(hold)):
                    res.n_stale += 1
            if h_out is not None and need_h and h_out["entry_adjust"] is not None:
                adjusted += abs(h_out["entry_adjust"]) > ADJUST_TOL
            if h_out is not None and need_h and h_out["m_h"] is None:
                failures.append(f"{ticker} {p['as_of']} 벤치마크 수익률 없음(m_h·y_e·beat=null)")
    append_jsonl(out_path, new_rows)
    if res.n_stale:
        res.notes.append(
            f"만기가 한참 지났지만 가격 봉이 부족한 예측 {res.n_stale}건"
            "(거래정지·상장폐지·데이터 결측 가능) — 계속 대기"
        )
    if adjusted:
        res.notes.append(
            f"기준일 종가가 기록 가격과 {ADJUST_TOL:.0%} 넘게 다른 예측 {adjusted}건(수정주가 반영)"
            " — 새 시계열 기준으로 채점"
        )
    if failures:
        uniq = list(dict.fromkeys(failures))
        res.notes.append(f"채점 실패·결측 {len(uniq)}건: {', '.join(uniq[:10])}{' …' if len(uniq) > 10 else ''}")
    log.info("tracking score: %s", res)
    return res


# =============================================================================== metrics


def _t_stat(x: np.ndarray) -> tuple[float, int]:
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 3:
        return float("nan"), n
    sd = float(x.std(ddof=1))
    if sd <= 0:
        return (math.copysign(math.inf, float(x.mean())) if x.mean() != 0 else float("nan")), n
    return float(x.mean() / (sd / math.sqrt(n))), n


def _p_one_sided(t: float, n: int) -> float:
    if not math.isfinite(t) and not math.isinf(t):
        return float("nan")
    if n < 2:
        return float("nan")
    return float(stats.t.sf(t, n - 1))


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    s = np.asarray(score, dtype="float64")
    y = np.asarray(label, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = stats.rankdata(s)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def benjamini_hochberg(pvalues: Sequence[float], q: float) -> list[bool]:
    """BH step-up over all m values (a NaN p counts as 1: an unavailable metric uses its slot)."""
    p = np.array([v if (v is not None and math.isfinite(v)) else 1.0 for v in pvalues], dtype="float64")
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")
    ok = p[order] <= (np.arange(1, m + 1) / m) * q
    reject = np.zeros(m, dtype=bool)
    if ok.any():
        k = int(np.nonzero(ok)[0].max())
        reject[order[: k + 1]] = True
    return reject.tolist()


def _frame(preds: Iterable[Mapping[str, Any]], outs: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    recs = []
    for p in preds:
        o = outs.get(p["id"]) or {}
        d = p.get("dist") or {}
        pl = p.get("plan") or {}
        li = p.get("llm") or {}
        recs.append(
            {
                "id": p["id"],
                "market": p["market"],
                "ticker": p["ticker"],
                "source": p.get("source"),
                "as_of": pd.Timestamp(p["as_of"]),
                "generated_at": str(p.get("generated_at")),
                "signal": p.get("signal"),
                "mu_excess": _fin(d.get("mu_excess")),
                "prob_up": _fin(d.get("prob_up")),
                "base_prob_up": _fin(d.get("base_prob_up")),
                "prob_outperform": _fin(d.get("prob_outperform")),
                "llm_pillar": _fin(li.get("llm_pillar")),
                "llm_skipped": bool(li.get("skipped", True)),
                "p_tp": _fin(pl.get("p_tp")),
                "p_sl": _fin(pl.get("p_sl")),
                "p_time": _fin(pl.get("p_time")),
                "scored": _has_h(o) if o else False,
                "h_end": pd.Timestamp(o["h_end"]) if o.get("h_end") else pd.NaT,
                "y_e": _fin(o.get("y_e")),
                "r_h": _fin(o.get("r_h")),
                "up": o.get("up"),
                "beat": o.get("beat"),
                "in80": o.get("in80"),
                "exc95": o.get("exc95"),
                "barrier_outcome": o.get("barrier_outcome"),
            }
        )
    return pd.DataFrame.from_records(recs)


def non_overlapping(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: per (market, ticker), the earliest row, then the next row whose as_of is on/after the kept
    row's realised window end (h_end) - the verdict sample; the rest are reference rows."""
    keep = pd.Series(False, index=df.index)
    for _, g in df.sort_values(["as_of", "generated_at"]).groupby(["market", "ticker"], sort=False):
        last_end = None
        for idx, row in g.iterrows():
            if last_end is None or row["as_of"] >= last_end:
                keep[idx] = True
                last_end = row["h_end"]
    return keep


def count_nonoverlapping_dates(df: pd.DataFrame) -> int:
    """Greedy count of snapshot dates whose outcome windows do not overlap (next date >= previous window end)."""
    if df.empty:
        return 0
    ends = df.groupby("as_of")["h_end"].max().sort_index()
    n, last_end = 0, None
    for d, end in ends.items():
        if last_end is None or d >= last_end:
            n += 1
            last_end = end
    return n


def _per_date_ic(df: pd.DataFrame, col: str) -> np.ndarray:
    out = []
    for _, g in df.groupby(["market", "as_of"]):
        g = g[[col, "y_e"]].dropna()
        if len(g) < holdout.MIN_NAMES_PER_DATE or g[col].nunique() < 2 or g["y_e"].nunique() < 2:
            continue
        out.append(float(stats.spearmanr(g[col], g["y_e"]).statistic))
    return np.array(out, dtype="float64")


def _ic_metric(df: pd.DataFrame, col: str) -> dict[str, Any]:
    ics = _per_date_ic(df, col)
    t, n = _t_stat(ics)
    return {
        "mean_ic": float(ics.mean()) if ics.size else None,
        "t": t if math.isfinite(t) else None,
        "n_dates": n,
        "n_rows": int(df[[col, "y_e"]].dropna().shape[0]),
        "p": _p_one_sided(t, n) if n >= 3 else None,
        "passed": bool(math.isfinite(t) and t >= holdout.T_EVIDENCE) if n >= 3 else None,
    }


def _direction(df: pd.DataFrame, n_boot: int, seed: int) -> dict[str, Any]:
    d = df.dropna(subset=["prob_up", "base_prob_up"])
    d = d[d["up"].notna()]
    out: dict[str, Any] = {"n_rows": int(len(d)), "n_dates": int(d.groupby(["market", "as_of"]).ngroups)}
    if len(d) == 0 or out["n_dates"] < 2:
        return {**out, "auc": None, "base_auc": None, "gain": None, "lower05": None, "p": None, "passed": None}
    y = d["up"].astype(bool).to_numpy()
    p, b = d["prob_up"].to_numpy(), d["base_prob_up"].to_numpy()
    auc, base_auc = _auc(p, y), _auc(b, y)
    gain = auc - base_auc
    if not math.isfinite(gain):
        return {**out, "auc": None, "base_auc": None, "gain": None, "lower05": None, "p": None, "passed": None}
    codes = d.groupby(["market", "as_of"]).ngroup().to_numpy()
    groups = [np.nonzero(codes == g)[0] for g in range(out["n_dates"])]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xF0A7]))
    draws = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        g = _auc(p[idx], y[idx]) - _auc(b[idx], y[idx])
        if math.isfinite(g):
            draws.append(g)
    arr = np.array(draws)
    lower = float(np.quantile(arr, 0.05)) if arr.size else float("nan")
    pval = float((1 + (arr <= 0).sum()) / (arr.size + 1)) if arr.size else float("nan")
    return {
        **out,
        "auc": auc,
        "base_auc": base_auc,
        "gain": gain,
        "lower05": lower if math.isfinite(lower) else None,
        "p": pval if math.isfinite(pval) else None,
        "passed": bool(math.isfinite(lower) and lower > 0),
    }


def _cluster_rate(df: pd.DataFrame, col: str) -> tuple[float, float, int]:
    """(pooled rate, date-cluster robust standard error, number of dates)."""
    d = df[df[col].notna()]
    if d.empty:
        return float("nan"), float("nan"), 0
    x = d[col].astype(float)
    grp = d.groupby(["market", "as_of"])
    s, n = grp[col].apply(lambda v: float(v.astype(float).sum())), grp.size()
    big_n, g = float(n.sum()), int(len(n))
    est = float(x.mean())
    if g < 2:
        return est, float("nan"), g
    resid = s.to_numpy() - est * n.to_numpy()
    se = math.sqrt(g / (g - 1.0) * float((resid**2).sum())) / big_n
    return est, se, g


def _tost_p(est: float, se: float, g: int, lo: float, hi: float) -> float:
    if not math.isfinite(est):
        return float("nan")
    if not (math.isfinite(se) and se > 0) or g < 2:
        return 0.0 if lo <= est <= hi and g >= 2 else 1.0
    p_lo = float(stats.t.sf((est - lo) / se, g - 1))
    p_hi = float(stats.t.sf((hi - est) / se, g - 1))
    return max(p_lo, p_hi)


def _risk(df: pd.DataFrame) -> dict[str, Any]:
    c, c_se, g1 = _cluster_rate(df, "in80")
    e, e_se, g2 = _cluster_rate(df, "exc95")
    if not (math.isfinite(c) and math.isfinite(e)):
        return {"cover80": None, "exc95": None, "n_rows": 0, "n_dates": 0, "p": None, "passed": None}
    lo_c, hi_c = holdout.COVER80_BAND
    lo_e, hi_e = holdout.EXC95_BAND
    p = max(_tost_p(c, c_se, g1, lo_c, hi_c), _tost_p(e, e_se, g2, lo_e, hi_e))
    return {
        "cover80": c,
        "cover80_se": c_se if math.isfinite(c_se) else None,
        "exc95": e,
        "exc95_se": e_se if math.isfinite(e_se) else None,
        "n_rows": int(df["in80"].notna().sum()),
        "n_dates": min(g1, g2),
        "p": p if math.isfinite(p) else None,
        "passed": bool(lo_c <= c <= hi_c and lo_e <= e <= hi_e),
    }


def _grade(df: pd.DataFrame) -> dict[str, Any]:
    diffs, n_buy, n_rest = [], 0, 0
    for _, g in df.dropna(subset=["y_e"]).groupby(["market", "as_of"]):
        buy = g["signal"].isin(BUY_SIGNALS)
        if buy.any() and (~buy).any():
            diffs.append(float(g.loc[buy, "y_e"].mean() - g.loc[~buy, "y_e"].mean()))
            n_buy += int(buy.sum())
            n_rest += int((~buy).sum())
    arr = np.array(diffs)
    t, n = _t_stat(arr)
    return {
        "mean_diff": float(arr.mean()) if arr.size else None,
        "t": t if math.isfinite(t) else None,
        "n_dates": n,
        "n_buy": n_buy,
        "n_hold_sell": n_rest,
        "p": _p_one_sided(t, n) if n >= 3 else None,
        "passed": bool(math.isfinite(t) and t >= holdout.T_EVIDENCE) if n >= 3 else None,
    }


def _secondary(sample: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    d = sample.dropna(subset=["prob_up"])
    d = d[d["up"].notna()]
    if len(d):
        out["prob_up_mean"] = float(d["prob_up"].mean())
        out["up_rate"] = float(d["up"].astype(float).mean())
        out["auc_up"] = _auc(d["prob_up"].to_numpy(), d["up"].astype(bool).to_numpy())
    d = sample.dropna(subset=["prob_outperform"])
    d = d[d["beat"].notna()]
    if len(d):
        out["prob_outperform_mean"] = float(d["prob_outperform"].mean())
        out["beat_rate"] = float(d["beat"].astype(float).mean())
        out["auc_beat"] = _auc(d["prob_outperform"].to_numpy(), d["beat"].astype(bool).to_numpy())
    b = sample[sample["barrier_outcome"].isin(["tp", "sl", "time"])].dropna(subset=["p_tp", "p_sl", "p_time"])
    if len(b):
        out["barrier"] = {
            "n": int(len(b)),
            "pred": {"tp": float(b["p_tp"].mean()), "sl": float(b["p_sl"].mean()), "time": float(b["p_time"].mean())},
            "real": {k: float((b["barrier_outcome"] == k).mean()) for k in ("tp", "sl", "time")},
        }
    if len(reference):
        ref = _ic_metric(pd.concat([sample, reference]), "mu_excess")
        out["all_rows_selection_ic"] = {k: ref[k] for k in ("mean_ic", "t", "n_dates", "n_rows")}
    return out


def metrics(
    settings: Settings, holdout_id: str | None = None, *, n_boot: int | None = None, seed: int = 0
) -> dict[str, Any]:
    """Primary metrics of a forward holdout (default: the latest registered FTH; none -> every row, 참고용).

    Raises:
        TrackingError: `holdout_id` is not a forward holdout.
    """
    reg = holdout.load_registry(settings)
    entry = reg.get(holdout_id) if holdout_id else reg.active_forward()
    if entry is not None and entry["kind"] != "forward":
        raise TrackingError(f"{entry['id']}: 전진 홀드아웃(FTH)이 아닙니다 — XSH는 'holdout evaluate'로 봅니다")
    pm = entry["primary_metrics"] if entry is not None else holdout.primary_metrics_block(settings)
    min_dates = int(pm.get("min_dates_for_verdict", settings.holdout.min_dates_for_verdict))
    q = float(pm.get("bh_q", settings.holdout.bh_q))
    n_boot = int(n_boot if n_boot is not None else pm.get("bootstrap_draws", holdout.BOOT_DRAWS))

    preds, bad_p = load_predictions(settings)
    outs, bad_o = load_outcomes(settings)
    notes: list[str] = []
    if bad_p or bad_o:
        notes.append(f"원장 읽기 불가 줄: 예측 {bad_p} · 결과 {bad_o} (제외)")
    counts: dict[str, Any] = {"predictions": len(preds)}
    if entry is not None:
        mine = [p for p in preds if p.get("holdout_id") == entry["id"]]
        counts["other_holdout"] = len(preds) - len(mine)
        after = [p for p in mine if _not_before(p.get("generated_at"), entry["registered_at"])]
        counts["before_registration"] = len(mine) - len(after)
        rows = [p for p in after if p.get("valid")]
        counts["invalid"] = len(after) - len(rows)
        if counts["invalid"]:
            notes.append(f"지문·설정 불일치(valid=false) {counts['invalid']}건 제외 — 등록 이후 설정/코드가 바뀐 실행")
    else:
        rows = preds
        notes.append("등록된 전진 홀드아웃(FTH) 없음 — 모든 기록을 참고용으로 집계(판정 아님)")
    df = _frame(rows, outs)
    counts["eligible"] = len(df)
    scored = df[df["scored"]] if len(df) else df
    counts["immature"] = int(len(df) - len(scored))
    counts["scored"] = int(len(scored))
    if len(scored):
        mask = non_overlapping(scored)
        sample, reference = scored[mask], scored[~mask]
    else:
        sample, reference = scored, scored
    counts["sample"] = int(len(sample))
    counts["reference_overlap"] = int(len(reference))
    counts["by_source"] = dict(Counter(sample["source"])) if len(sample) else {}
    n_dates = count_nonoverlapping_dates(sample) if len(sample) else 0
    verdict_ok = n_dates >= min_dates
    if not verdict_ok:
        notes.append(f"비중첩 날짜 {n_dates}/{min_dates}개 — {VERDICT_NA}: 아래 수치는 참고용")

    empty = sample.iloc[0:0] if len(sample) else sample
    llm_rows = sample[~sample["llm_skipped"]] if len(sample) else empty
    values = {
        holdout.METRIC_SELECTION: _ic_metric(sample, "mu_excess") if len(sample) else {"p": None, "passed": None},
        holdout.METRIC_DIRECTION: _direction(sample, n_boot, seed) if len(sample) else {"p": None, "passed": None},
        holdout.METRIC_RISK: _risk(sample) if len(sample) else {"p": None, "passed": None},
        holdout.METRIC_GRADE: _grade(sample) if len(sample) else {"p": None, "passed": None},
        holdout.METRIC_LLM: _ic_metric(llm_rows, "llm_pillar") if len(llm_rows) else {"p": None, "passed": None},
    }
    if len(sample) and values[holdout.METRIC_DIRECTION].get("n_rows", 0) == 0:
        notes.append("루트 쌍둥이 P(상승)(base_prob_up)이 있는 표본 없음 — 방향 지표 판정 불가")
    keys = list(holdout.PRIMARY_KEYS)
    rejects = benjamini_hochberg([values[k].get("p") for k in keys], q)
    specs = {m["key"]: m for m in pm.get("metrics", [])}
    primary = []
    for k, rej in zip(keys, rejects):
        v = values[k]
        passed = v.get("passed")
        if not verdict_ok:
            verdict = VERDICT_NA
        elif passed is None or v.get("p") is None:
            verdict = VERDICT_NO_DATA
        elif passed and rej:
            verdict = VERDICT_EVIDENCE
        elif passed:
            verdict = VERDICT_BH_FAIL
        else:
            verdict = VERDICT_NONE
        primary.append(
            {
                "key": k,
                "label": specs.get(k, {}).get("label", k),
                "rule": specs.get(k, {}).get("rule", ""),
                "values": v,
                "p": v.get("p"),
                "bh_reject": bool(rej) if v.get("p") is not None else None,
                "passed": passed,
                "verdict": verdict,
            }
        )
    return holdout._jsonable(
        {
            "holdout_id": entry["id"] if entry is not None else None,
            "registered_at": entry["registered_at"] if entry is not None else None,
            "computed_at": datetime.now(holdout.KST).replace(microsecond=0).isoformat(),
            "counts": counts,
            "n_dates": n_dates,
            "min_dates_for_verdict": min_dates,
            "verdict_available": verdict_ok,
            "bh_q": q,
            "primary": primary,
            "secondary": _secondary(sample, reference) if len(sample) else {},
            "notes": notes,
        }
    )


def headline(result: Mapping[str, Any]) -> dict[str, Any]:
    """Short summary stored with a registry look."""
    return {
        "n_dates": result.get("n_dates"),
        "verdict_available": result.get("verdict_available"),
        "sample": (result.get("counts") or {}).get("sample"),
        "verdicts": {m["key"]: m["verdict"] for m in result.get("primary", [])},
    }


def _num(x: Any, d: int = 3, signed: bool = True) -> str:
    v = _fin(x)
    if v is None:
        return "—"
    return f"{v:+.{d}f}" if signed else f"{v:.{d}f}"


def report_lines(result: Mapping[str, Any]) -> list[str]:
    """Korean console lines of a metrics() result."""
    c = result.get("counts") or {}
    lines = [
        f"홀드아웃 {result.get('holdout_id') or '(등록 없음)'} · 예측 {c.get('predictions', 0)}건 "
        f"· 대상 {c.get('eligible', 0)}건 · 채점 {c.get('scored', 0)}건 · 미만기 {c.get('immature', 0)}건 "
        f"· 판정 표본 {c.get('sample', 0)}건 "
        f"(겹침 참고 {c.get('reference_overlap', 0)}건)",
        f"비중첩 날짜 {result.get('n_dates', 0)}/{result.get('min_dates_for_verdict')}개 · "
        f"{'판정 가능' if result.get('verdict_available') else VERDICT_NA} · BH q={result.get('bh_q')}",
    ]
    for m in result.get("primary", []):
        v = m.get("values") or {}
        k = m["key"]
        if k in (holdout.METRIC_SELECTION, holdout.METRIC_LLM):
            detail = f"평균 IC {_num(v.get('mean_ic'))} · t {_num(v.get('t'), 2)} · 날짜 {v.get('n_dates', 0)}"
        elif k == holdout.METRIC_DIRECTION:
            detail = (
                f"AUC {_num(v.get('auc'), 3, False)} vs 기준 {_num(v.get('base_auc'), 3, False)} · 개선 "
                f"{_num(v.get('gain'), 4)} · 5% 하한 {_num(v.get('lower05'), 4)}"
            )
        elif k == holdout.METRIC_RISK:
            detail = f"80% 적중 {_num(v.get('cover80'), 3, False)} · VaR95 초과 {_num(v.get('exc95'), 3, False)}"
        else:
            detail = (
                f"Buy+ − Hold/Sell {_num(v.get('mean_diff'), 4)} · t {_num(v.get('t'), 2)} · 날짜 {v.get('n_dates', 0)}"
            )
        lines.append(f"  • {m['label']}: {detail} · p {_num(m.get('p'), 3, False)} → {m['verdict']}")
    sec = result.get("secondary") or {}
    if sec.get("prob_up_mean") is not None:
        lines.append(
            f"  참고: 평균 P(상승) {sec['prob_up_mean']:.1%} vs 실현 {sec.get('up_rate', float('nan')):.1%}"
            + (
                f" · 평균 P(BM 초과) {sec['prob_outperform_mean']:.1%} vs 실현 {sec.get('beat_rate', float('nan')):.1%}"
                if sec.get("prob_outperform_mean") is not None
                else ""
            )
        )
    if sec.get("barrier"):
        b = sec["barrier"]
        lines.append(
            f"  참고: 배리어 {b['n']}건 익절·손절·시간 예측 {b['pred']['tp']:.0%}·{b['pred']['sl']:.0%}·"
            f"{b['pred']['time']:.0%} vs 실현 {b['real']['tp']:.0%}·{b['real']['sl']:.0%}·{b['real']['time']:.0%}"
        )
    lines.extend(f"  알림: {n}" for n in result.get("notes", []))
    return lines
