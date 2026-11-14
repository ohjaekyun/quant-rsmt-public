"""Forward-tracking ledger (docs/HOLDOUT.md): record / snapshot / score / metrics / pipeline hook / CLI.

Offline: stub providers serve synthetic prices; the registry and the ledger live in a temporary folder.
"""

from __future__ import annotations

import asyncio
import json
import math
import zlib
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from conftest import make_prices
from typer.testing import CliRunner

from quantcio import holdout, tracking
from quantcio.config import Settings, load_settings
from quantcio.data import base as data_base
from quantcio.data.base import MarketProvider, normalise_ticker
from quantcio.models import Fundamentals, Instrument, Market
from quantcio.report.sample import build_sample_report

REG_NOW = datetime(2026, 1, 2, 9, 0, tzinfo=holdout.KST)
FP = "c" * 16


def _settings(tmp_path, cache_dir=None) -> Settings:
    return load_settings(
        data={"cache_dir": str(cache_dir or (tmp_path / "cache"))},
        report={"output_dir": str(tmp_path / "reports")},
        holdout={
            "registry_path": str(tmp_path / "hold" / "registry.json"),
            "ledger_dir": str(tmp_path / "hold" / "led"),
        },
    )


@pytest.fixture(autouse=True)
def _recording_on(monkeypatch):
    """These tests exercise the recorder itself: a developer shell with QUANTCIO_NO_RECORD=1 (the dev-run switch)
    must not turn them into no-ops. Ledgers point at tmp_path and the repository ledger refuses test writes."""
    monkeypatch.delenv(tracking.NO_RECORD_ENV, raising=False)


@pytest.fixture
def ts(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setattr(holdout, "git_state", lambda cwd=None: holdout.GitState("abc123", False))
    # other work may edit src/quantcio while the suite runs: pin the fingerprint (tested in test_holdout.py)
    monkeypatch.setattr(holdout, "code_fingerprint", lambda root=None: FP)
    return _settings(tmp_path)


def _lines(path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# =============================================================================== record


def test_record_predictions_fields_and_idempotency(ts):
    rep = build_sample_report(4, settings=ts)
    res = tracking.record_predictions(rep, ts)
    assert res.n_new == 4 and res.n_existing == 0
    assert res.holdout_ids == {"KR": None, "US": None} and not any(res.valid.values())
    assert any("등록된 전진 홀드아웃(FTH) 없음" in n for n in res.notes)
    pred_path, _ = tracking.ledger_paths(ts)
    rows = _lines(pred_path)
    assert [r["ticker"] for r in rows] == [t.instrument.ticker for t in rep.tickers]
    r = rows[0]
    t = rep.tickers[0]
    assert r["id"] == tracking.prediction_id(res.run_id, t.instrument.ticker) and len(r["id"]) == 40
    assert r["source"] == "report" and r["holdout_id"] is None and r["valid"] is False
    assert r["as_of"] == t.as_of.isoformat() and r["generated_at"] == rep.generated_at.isoformat()
    assert r["config_digest"] == ts.digest() and r["code_fingerprint"] == FP
    assert set(r["dist"]) >= {
        "mu",
        "sigma",
        "nu",
        "beta",
        "mu_excess",
        "prob_up",
        "prob_outperform",
        "q10",
        "q50",
        "q90",
        "var95",
        "skill_ok",
        "degraded",
    }
    assert set(r["plan"]) >= {"entry", "stop", "take", "hold_days", "p_tp", "p_sl", "p_time", "weight"}
    assert set(r["llm"]) == {"news_score", "filings_tone", "critic_adj", "llm_pillar", "skipped"}
    assert r["plan"]["hold_days"] == 60 and r["horizon_days"] == 20
    assert r["plan"]["p_tp"] + r["plan"]["p_sl"] + r["plan"]["p_time"] == pytest.approx(1.0)
    assert r["dist"]["mu_excess"] == t.quant.distribution.mu_excess and r["pillars"] == t.executive.pillars
    assert {"ticker", "market", "last_price", "benchmark_id", "signal", "composite", "confidence"} <= set(r)

    again = tracking.record_predictions(rep, ts)  # same run -> same ids -> nothing appended
    assert again.n_new == 0 and again.n_existing == 4 and len(_lines(pred_path)) == 4
    snap = tracking.record_predictions(rep, ts, source="snapshot")  # another run id
    assert snap.n_new == 4 and len(_lines(pred_path)) == 8
    with pytest.raises(tracking.TrackingError):
        tracking.record_predictions(rep, ts, source="cron")


def test_record_predictions_validity_against_the_forward_registration(ts, monkeypatch):
    monkeypatch.setattr(holdout, "_now", lambda: datetime(2026, 9, 1, tzinfo=holdout.KST))
    fth = holdout.register_forward(ts, ["KR"])
    rep = build_sample_report(2, settings=ts)  # 005930 (KR), AAPL (US); generated 2026-09-23
    res = tracking.record_predictions(rep, ts)
    assert res.holdout_ids == {"KR": fth["id"], "US": None} and res.valid == {"KR": True, "US": False}
    rows = {r["ticker"]: r for r in _lines(tracking.ledger_paths(ts)[0])}
    assert rows["005930"]["holdout_id"] == fth["id"] and rows["005930"]["valid"] is True
    assert rows["AAPL"]["holdout_id"] is None and rows["AAPL"]["valid"] is False

    changed = ts.model_copy(deep=True)
    changed.fusion.min_buy_upside = 0.05
    rep2 = build_sample_report(1, settings=ts).model_copy(
        update={"generated_at": datetime(2026, 9, 24, tzinfo=holdout.KST)}
    )
    res2 = tracking.record_predictions(rep2, changed)
    assert res2.valid == {"KR": False} and any("valid=false" in n and "설정 다이제스트" in n for n in res2.notes)
    # a run older than the registration is never valid
    old = rep2.model_copy(update={"generated_at": datetime(2026, 8, 1, tzinfo=holdout.KST)})
    assert tracking.record_predictions(old, ts).valid == {"KR": False}

    xsh = holdout.register_cross_sectional(ts, "KR", ["111111"], ["2018-01-01", "2026-08-01"], [])
    with pytest.raises(tracking.TrackingError, match="FTH"):
        tracking.record_predictions(rep, ts, holdout_id=xsh["id"])


def test_repository_ledger_is_never_written_by_tests(monkeypatch):
    monkeypatch.delenv(holdout.HOLDOUT_ROOT_ENV, raising=False)
    s = load_settings()
    with pytest.raises(holdout.HoldoutError, match="테스트 실행 중"):
        tracking.record_predictions(build_sample_report(1, settings=s), s)


# =============================================================================== score


AS_OF = pd.Timestamp("2026-01-05")
PRE = pd.bdate_range(end=AS_OF, periods=30)
POST = pd.bdate_range(start=AS_OF + pd.Timedelta(1, "D"), periods=80)


def _post_path(ticker: str) -> np.ndarray:
    k = np.arange(1, 81, dtype="float64")
    if ticker == "TP01":  # +5% at bar 20, take-profit (+8%) first touched at bar 30 (108.5)
        c = np.where(k <= 20, 100 + 0.25 * k, 105 + 0.3 * (k - 20))
        c[29] = 108.5
        c[30:] = 108.5
        return c
    if ticker == "SL01":  # stop (-5%) first touched at bar 5 (-5.5%), 97 at bar 20
        c = np.full(80, 97.0)
        c[:4] = 99.0
        c[4] = 94.5
        return c
    return np.full(80, 101.0)  # TM01: neither barrier -> time exit at bar 60, +1%


class PathProvider(MarketProvider):
    """Synthetic post-prediction paths; `limit` = bars available after AS_OF."""

    limit = 80

    def __init__(self, market, settings, cache):
        super().__init__(settings, cache)
        self.market = market

    def resolve(self, ticker):
        if ticker == "GONE":
            raise LookupError("상장폐지(스텁)")
        return Instrument(
            ticker=ticker,
            market=Market.US,
            name=ticker,
            currency="USD",
            benchmark="SPY",
            benchmark_name="SPY",
        )

    def _frame(self, closes):
        idx = PRE.append(POST[: self.limit])
        c = np.asarray(closes[: len(idx)], dtype="float64")
        return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)

    def prices(self, inst, start):
        return self._frame(np.concatenate([np.full(30, 100.0), _post_path(inst.ticker)]))

    def benchmark(self, inst, start):
        return self._frame(np.concatenate([np.full(30, 1000.0), 1000 + 0.5 * np.arange(1, 81)]))

    def fundamentals(self, inst):
        return Fundamentals()

    def news(self, inst, since, limit):
        return []

    def filings(self, inst, since, limit):
        return []

    def universe(self, n):
        return []


def _pred(ticker: str, as_of: pd.Timestamp = AS_OF, last_price: float = 100.0, **kw) -> dict:
    row = {
        "id": tracking.prediction_id("t", f"{ticker}{as_of.date()}"),
        "run_id": "t",
        "generated_at": f"{as_of.date()}T16:00:00+09:00",
        "as_of": as_of.date().isoformat(),
        "holdout_id": None,
        "valid": False,
        "source": "snapshot",
        "ticker": ticker,
        "market": "US",
        "last_price": last_price,
        "benchmark_id": "SPY",
        "signal": "Buy",
        "horizon_days": 20,
        "dist": {"beta": 1.2, "q10": -0.05, "q90": 0.04, "var95": -0.08, "mu_excess": 0.01, "prob_up": 0.6},
        "plan": {"stop_pct": -0.05, "take_pct": 0.08, "hold_days": 60, "p_tp": 0.3, "p_sl": 0.3, "p_time": 0.4},
        "llm": {"skipped": True, "llm_pillar": 0.0},
    }
    row.update(kw)
    return row


def test_score_labels_and_barriers_exact_and_idempotent(ts, monkeypatch):
    monkeypatch.setattr(data_base, "get_provider", lambda m, s, c: PathProvider(m, s, c))
    pred_path, out_path = tracking.ledger_paths(ts)
    late = POST[70]  # only 9 bars after it even at the full limit -> immature
    tracking.append_jsonl(
        pred_path,
        [_pred("TP01"), _pred("SL01"), _pred("TM01", last_price=90.0), _pred("TP01", as_of=late), _pred("GONE")],
    )
    PathProvider.limit = 40
    try:
        # nothing is due before the 20th bar
        early = tracking.score(ts, now=POST[10].date())
        assert (early.n_h, early.n_barrier) == (0, 0) and early.n_immature == 4 and early.n_failed == 1
        res = tracking.score(ts, now=date(2026, 12, 31))
        assert (res.n_h, res.n_barrier, res.n_immature, res.n_failed) == (3, 0, 1, 1)
        assert any("GONE" in n for n in res.notes) and any("수정주가" in n for n in res.notes)
        outs, _ = tracking.load_outcomes(ts)
        tp = outs[_pred("TP01")["id"]]
        assert tp["stages"] == ["h"] and "barrier_outcome" not in tp
        assert tp["r_h"] == pytest.approx(math.log(1.05)) and tp["m_h"] == pytest.approx(math.log(1.01))
        assert tp["y_e"] == pytest.approx(math.log(1.05) - 1.2 * math.log(1.01))
        assert tp["up"] is True and tp["beat"] is True
        assert tp["in80"] is False and tp["exc95"] is False  # +5% > q90 (+4%); not below VaR95
        assert tp["h_end"] == POST[19].date().isoformat() and tp["entry_close"] == 100.0
        sl = outs[_pred("SL01")["id"]]
        assert sl["r_h"] == pytest.approx(math.log(0.97)) and sl["up"] is False and sl["in80"] is True
        tm = outs[_pred("TM01", last_price=90.0)["id"]]
        assert tm["beat"] is False and tm["entry_adjust"] == pytest.approx(100 / 90 - 1)  # r_h == m_h -> not beat

        PathProvider.limit = 80
        res2 = tracking.score(ts, now=date(2026, 12, 31))
        assert (res2.n_h, res2.n_barrier, res2.n_complete) == (0, 3, 0)
        outs, _ = tracking.load_outcomes(ts)
        tp = outs[_pred("TP01")["id"]]
        assert tp["stages"] == ["h", "barrier"]
        assert (tp["barrier_outcome"], tp["bars_used"]) == ("tp", 30) and tp["barrier_return"] == pytest.approx(0.085)
        sl = outs[_pred("SL01")["id"]]
        assert (sl["barrier_outcome"], sl["bars_used"]) == ("sl", 5) and sl["barrier_return"] == pytest.approx(-0.055)
        tm = outs[_pred("TM01", last_price=90.0)["id"]]
        assert (tm["barrier_outcome"], tm["bars_used"]) == ("time", 60) and tm["barrier_return"] == pytest.approx(0.01)
        assert tm["barrier_end"] == POST[59].date().isoformat()

        n_lines = len(_lines(out_path))
        res3 = tracking.score(ts, now=date(2026, 12, 31))  # idempotent
        assert (res3.n_h, res3.n_barrier, res3.n_complete) == (0, 0, 3) and len(_lines(out_path)) == n_lines

        # a fresh prediction whose h and barrier are both due is written as ONE final line
        tracking.append_jsonl(pred_path, [_pred("SL01", run_id="u", id="fresh")])
        res4 = tracking.score(ts, now=date(2026, 12, 31))
        assert (res4.n_h, res4.n_barrier) == (1, 1)
        assert _lines(out_path)[-1]["stage"] == "final" and _lines(out_path)[-1]["barrier_outcome"] == "sl"
    finally:
        PathProvider.limit = 80


def test_barrier_path_matches_the_evaluation_replay():
    from quantcio.evaluation import realised_barrier

    rng = np.random.default_rng(3)
    codes = {"tp": 1, "sl": -1, "time": 0}
    for _ in range(300):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))
        label, ret, bars = tracking.barrier_path(c, 100.0, -0.06, 0.09)
        o, r = realised_barrier(c, 100.0, -0.06, 0.09)
        assert codes[label] == o and ret == pytest.approx(r)
        assert 1 <= bars <= 60 and (label == "time") == (bars == 60 and o == 0)


# =============================================================================== metrics


def _write_ledger(s: Settings, fth_id: str | None, n_dates: int, planted: bool, seed: int = 0, base: bool = True):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-02-02", periods=20 * (n_dates + 1))
    preds, outs = [], []
    for k in range(n_dates):
        as_of, h_end = days[20 * k], days[20 * k + 20]
        mu = rng.normal(0, 0.02, 30)
        y = (mu if planted else 0.0) + rng.normal(0, 0.01 if planted else 0.02, 30)
        in80 = np.zeros(30, dtype=bool)
        in80[rng.permutation(30)[:24]] = True
        exc = np.zeros(30, dtype=bool)
        exc[rng.permutation(30)[: 1 + k % 2]] = True
        order = np.argsort(-mu)
        signal = np.array(["Hold"] * 30, dtype=object)
        signal[order[:10]] = "Buy"
        signal[order[:3]] = "Strong Buy"
        signal[order[-3:]] = "Sell"
        for i in range(30):
            tk = f"T{i:02d}"
            pid = tracking.prediction_id(f"snap{k}", tk)
            preds.append(
                {
                    "id": pid,
                    "run_id": f"snap{k}",
                    "generated_at": f"{as_of.date()}T17:00:00+09:00",
                    "as_of": as_of.date().isoformat(),
                    "holdout_id": fth_id,
                    "valid": True,
                    "source": "snapshot",
                    "ticker": tk,
                    "market": "KR",
                    "signal": signal[i],
                    "horizon_days": 20,
                    "dist": {
                        "mu_excess": float(mu[i]),
                        "prob_up": float(1 / (1 + math.exp(-80 * mu[i]))),
                        "base_prob_up": float(0.5 + rng.normal(0, 0.01)) if base else None,
                        "prob_outperform": 0.5,
                    },
                    "plan": {"p_tp": 0.3, "p_sl": 0.3, "p_time": 0.4, "hold_days": 60},
                    "llm": {"skipped": False, "llm_pillar": float(np.clip(20 * mu[i] + rng.normal(0, 0.05), -1, 1))},
                }
            )
            outs.append(
                {
                    "id": pid,
                    "stage": "final",
                    "as_of": as_of.date().isoformat(),
                    "h_end": h_end.date().isoformat(),
                    "r_h": float(y[i]),
                    "m_h": 0.0,
                    "y_e": float(y[i]),
                    "up": bool(y[i] > 0),
                    "beat": bool(y[i] > 0),
                    "in80": bool(in80[i]),
                    "exc95": bool(exc[i]),
                    "barrier_outcome": ["tp", "sl", "time"][i % 3],
                }
            )
    pred_path, out_path = tracking.ledger_paths(s)
    tracking.append_jsonl(pred_path, preds)
    tracking.append_jsonl(out_path, outs)
    return preds, days


def _by_key(res: dict) -> dict:
    return {m["key"]: m for m in res["primary"]}


def test_metrics_planted_signal_gives_a_verdict(ts, monkeypatch):
    monkeypatch.setattr(holdout, "_now", lambda: REG_NOW)
    fth = holdout.register_forward(ts)
    preds, days = _write_ledger(ts, fth["id"], 15, planted=True)
    pred_path, out_path = tracking.ledger_paths(ts)
    # rows that must NOT enter the verdict sample: overlap (5 bdays after a kept row), invalid, other holdout,
    # immature (no outcome)
    extra = []
    for i, tag in enumerate(("overlap", "invalid", "other", "immature")):
        p = dict(preds[i], id=f"x{i}", run_id=f"x{i}")
        if tag == "overlap":
            p["as_of"] = days[5].date().isoformat()
            tracking.append_jsonl(
                out_path, [{"id": "x0", "stage": "final", "h_end": days[25].date().isoformat(), "y_e": 0.1, "up": True}]
            )
        elif tag == "invalid":
            p["valid"] = False
        elif tag == "other":
            p["holdout_id"] = "FTH-OTHER"
        extra.append(p)
    tracking.append_jsonl(pred_path, extra)

    res = tracking.metrics(ts, n_boot=400)
    c = res["counts"]
    assert res["holdout_id"] == fth["id"] and res["verdict_available"] and res["n_dates"] == 15
    assert (c["sample"], c["reference_overlap"], c["invalid"], c["other_holdout"], c["immature"]) == (450, 1, 1, 1, 1)
    m = _by_key(res)
    for key in holdout.PRIMARY_KEYS:
        assert m[key]["verdict"] == tracking.VERDICT_EVIDENCE, (key, m[key])
        assert m[key]["bh_reject"] is True
    assert m[holdout.METRIC_SELECTION]["values"]["t"] > 5 and m[holdout.METRIC_SELECTION]["values"]["n_dates"] == 15
    assert m[holdout.METRIC_DIRECTION]["values"]["lower05"] > 0
    assert m[holdout.METRIC_RISK]["values"]["cover80"] == pytest.approx(0.8)
    assert m[holdout.METRIC_RISK]["values"]["exc95"] == pytest.approx(22 / 450)
    assert res["secondary"]["barrier"]["n"] == 450
    lines = tracking.report_lines(res)
    assert any("근거 있음" in ln for ln in lines) and "비중첩 날짜 15/12개" in lines[1]
    json.dumps(res)  # JSON-able (no NaN / numpy scalars)


def test_metrics_small_sample_is_not_judged(ts, monkeypatch):
    monkeypatch.setattr(holdout, "_now", lambda: REG_NOW)
    fth = holdout.register_forward(ts)
    _write_ledger(ts, fth["id"], 5, planted=True)
    res = tracking.metrics(ts, n_boot=200)
    assert not res["verdict_available"] and res["n_dates"] == 5
    assert all(m["verdict"] == tracking.VERDICT_NA for m in res["primary"])
    assert _by_key(res)[holdout.METRIC_SELECTION]["values"]["t"] > 2  # still computed, shown as 참고
    assert any("판정 불가(표본 부족)" in n for n in res["notes"])


def test_metrics_null_ledger_and_missing_twin(ts, monkeypatch):
    monkeypatch.setattr(holdout, "_now", lambda: REG_NOW)
    fth = holdout.register_forward(ts)
    _write_ledger(ts, fth["id"], 14, planted=False, seed=5, base=False)
    res = tracking.metrics(ts, fth["id"], n_boot=200)
    m = _by_key(res)
    assert res["verdict_available"]
    assert m[holdout.METRIC_SELECTION]["verdict"] in (tracking.VERDICT_NONE, tracking.VERDICT_BH_FAIL)
    assert m[holdout.METRIC_DIRECTION]["verdict"] == tracking.VERDICT_NO_DATA
    assert any("base_prob_up" in n for n in res["notes"])


def test_metrics_without_registration_and_empty_ledger(ts):
    res = tracking.metrics(ts)
    assert res["holdout_id"] is None and res["counts"]["predictions"] == 0 and not res["verdict_available"]
    assert all(m["verdict"] == tracking.VERDICT_NA for m in res["primary"])
    assert any("FTH) 없음" in n for n in res["notes"])
    tracking.report_lines(res)


def test_benjamini_hochberg_and_non_overlap_rule():
    assert tracking.benjamini_hochberg([0.01, 0.04, 0.03, 0.2, float("nan")], 0.10) == [True, True, True, False, False]
    assert tracking.benjamini_hochberg([0.2, 0.3], 0.10) == [False, False]
    d = pd.Timestamp
    df = pd.DataFrame(
        {
            "market": ["KR"] * 4,
            "ticker": ["A", "A", "A", "B"],
            "as_of": [d("2026-01-05"), d("2026-01-12"), d("2026-02-02"), d("2026-01-12")],
            "generated_at": ["1", "2", "3", "2"],
            "h_end": [d("2026-02-02"), d("2026-02-09"), d("2026-03-02"), d("2026-02-09")],
        }
    )
    assert tracking.non_overlapping(df).tolist() == [True, False, True, True]
    assert tracking.count_nonoverlapping_dates(df[tracking.non_overlapping(df)]) == 2


# =============================================================================== snapshot / pipeline hook

N_DAYS = 1600
UNIVERSE = 24


def _seed(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode()) % (2**31)


class StubProvider(MarketProvider):
    """Deterministic offline provider (a trimmed copy of tests/test_pipeline.StubProvider)."""

    def __init__(self, market, settings, cache):
        super().__init__(settings, cache)
        self.market = market

    def resolve(self, ticker):
        t = normalise_ticker(ticker)
        if self.market is Market.KR:
            return Instrument(
                ticker=t,
                market=Market.KR,
                name=f"테스트{t}",
                exchange="KOSPI",
                currency="KRW",
                yf_symbol=f"{t}.KS",
                benchmark="KOSPI",
                benchmark_name="KOSPI",
            )
        return Instrument(
            ticker=t, market=Market.US, name=t, currency="USD", yf_symbol=t, benchmark="SPY", benchmark_name="SPY"
        )

    def _bench(self, bench):
        if bench == "^VIX":
            px = make_prices(n=60, start_price=16.0, drift=0.0, vol=0.03, seed=77)
            px["volume"] = 0.0
            return px
        return make_prices(n=N_DAYS, start_price=2500.0, drift=0.0002, vol=0.011, seed=_seed("bench", bench))

    def prices(self, inst, start):
        bench = self._bench(inst.benchmark)
        seed = _seed("px", inst.ticker)
        base = make_prices(n=N_DAYS, start_price=50_000.0, drift=0.0001, vol=0.017, seed=seed)
        blr = np.log(bench["close"]).diff().fillna(0.0).to_numpy()
        beta = 0.7 + 0.6 * np.random.default_rng(seed).random()
        out = base.copy()
        out[["open", "high", "low", "close"]] = (
            out[["open", "high", "low", "close"]].to_numpy() * np.exp(np.cumsum(beta * blr))[:, None]
        )
        return out

    def benchmark(self, inst, start):
        return self._bench(inst.benchmark)

    def fundamentals(self, inst):
        return Fundamentals(source="stub", per=14.0, pbr=1.3, roe=0.11, sector_per=15.0)

    def news(self, inst, since, limit):
        return []

    def filings(self, inst, since, limit):
        return []

    def fx_rate(self, from_ccy, to_ccy):
        return 1.0 if from_ccy == to_ccy else (1380.0 if from_ccy == "USD" else 1 / 1380.0)

    def universe(self, n):
        return [f"{900000 + i}" for i in range(n)] if self.market is Market.KR else [f"U{i:03d}" for i in range(n)]


@pytest.fixture(scope="module")
def shared_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("trk_cache")


@pytest.fixture
def stub_ts(tmp_path, monkeypatch, shared_cache) -> Settings:
    s = _settings(tmp_path, cache_dir=shared_cache)
    for m in s.rsmt.markets.values():
        m.universe_size = UNIVERSE
    monkeypatch.setattr(
        data_base, "get_provider", lambda market, settings, cache: StubProvider(market, settings, cache)
    )
    monkeypatch.setattr(holdout, "git_state", lambda cwd=None: holdout.GitState("abc123", False))
    monkeypatch.setattr(holdout, "code_fingerprint", lambda root=None: FP)
    return s


def test_snapshot_serves_the_universe_without_llm_and_records_it(stub_ts):
    fth = holdout.register_forward(stub_ts, ["KR"])
    rec = tracking.snapshot("KR", stub_ts, n=6)
    assert rec.n_new == 6 and rec.holdout_ids == {"KR": fth["id"]} and rec.valid == {"KR": True}
    rows = _lines(tracking.ledger_paths(stub_ts)[0])
    assert [r["ticker"] for r in rows] == [f"{900000 + i}" for i in range(6)]
    for r in rows:
        assert r["source"] == "snapshot" and r["llm"]["skipped"] is True and r["valid"] is True
        assert r["dist"] is not None and 0.0 < r["dist"]["base_prob_up"] < 1.0
        assert r["plan"]["hold_days"] == stub_ts.fusion.target_horizon_days
    with pytest.raises(tracking.TrackingError):
        tracking.snapshot("JP", stub_ts)


def test_run_report_hook_records_and_never_fails_the_report(stub_ts, monkeypatch):
    from quantcio import pipeline

    art = asyncio.run(pipeline.run_report(["900001", "900002"], stub_ts, use_llm=False))
    rows = _lines(tracking.ledger_paths(stub_ts)[0])
    assert [r["ticker"] for r in rows] == ["900001", "900002"] and all(r["source"] == "report" for r in rows)
    assert all(r["holdout_id"] is None for r in rows)
    assert not any("원장" in w for w in art.warnings)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(tracking, "record_predictions", boom)
    res = pipeline.RunResult(report=art.report, runs=[], fits={}, timings={})
    warns = pipeline._record_forward_ledger(res, stub_ts)
    assert len(warns) == 1 and warns[0].startswith("전진 추적 원장 기록 실패(OSError: disk full)")

    off = stub_ts.model_copy(deep=True)
    off.holdout.record_reports = False
    art2 = asyncio.run(pipeline.run_report(["900001"], off, use_llm=False))  # hook disabled: boom never called
    assert not any("원장" in w for w in art2.warnings)


# =============================================================================== CLI


@pytest.fixture
def cli_env(ts, monkeypatch):
    import quantcio.cli as cli

    monkeypatch.setattr(cli, "load_settings", lambda config=None: ts)
    monkeypatch.setattr(cli, "_setup_logging", lambda settings, verbose: ts.cache_path / "test.log")
    monkeypatch.setattr(data_base, "get_provider", lambda m, s, c: PathProvider(m, s, c))
    return cli, CliRunner(), ts


def test_cli_holdout_register_status_evaluate(cli_env, monkeypatch):
    cli, runner, s = cli_env
    res = runner.invoke(cli.app, ["holdout", "status"])
    assert res.exit_code == 0 and "등록된 홀드아웃 없음" in res.stdout
    res = runner.invoke(cli.app, ["holdout", "register", "--kind", "forward"])
    assert res.exit_code == 0, res.output
    assert "등록 완료" in res.stdout and "FTH-" in res.stdout
    res = runner.invoke(cli.app, ["holdout", "register", "--kind", "bogus"])
    assert res.exit_code == 2
    res = runner.invoke(cli.app, ["holdout", "register", "--kind", "xs"])
    assert res.exit_code == 2  # --market required

    monkeypatch.setattr(cli, "_evaluation_hook", lambda name: None)
    res = runner.invoke(cli.app, ["holdout", "register", "--kind", "xs", "--market", "KR"])
    assert res.exit_code == 1 and "xsh_candidate_names" in res.output

    calls = []

    def hook(name):
        if name == "xsh_candidate_names":
            return lambda market, settings, cache: ["111111", "222222"]
        return lambda entry, settings, cache: (
            calls.append(entry["id"])
            or {
                "result_path": "xsh.json",
                "headline": {"ic_t": 1.5},
            }
        )

    monkeypatch.setattr(cli, "_evaluation_hook", hook)
    # the real tuning list loads the PIT universe (disabled offline): the XSH names are checked against this one
    monkeypatch.setattr(holdout, "tuning_names", lambda market, settings, cache: ["005930", "000660"])
    res = runner.invoke(cli.app, ["holdout", "register", "--kind", "xs", "--market", "KR", "--start", "2018-01-01"])
    assert res.exit_code == 0, res.output
    xsh = next(e for e in holdout.load_registry(s).entries if e["kind"] == "cross_sectional")
    assert xsh["names"] == ["111111", "222222"] and xsh["period"][0] == "2018-01-01"

    res = runner.invoke(cli.app, ["holdout", "evaluate", "--id", xsh["id"]])
    assert res.exit_code == 0, res.output
    assert calls == [xsh["id"]] and "pristine" in res.stdout
    res = runner.invoke(cli.app, ["holdout", "evaluate", "--id", xsh["id"]])
    assert res.exit_code == 0 and "pristine 아님" in res.stdout
    looks = holdout.load_registry(s).get(xsh["id"])["looks"]
    assert [lk["pristine"] for lk in looks] == [True, False] and looks[0]["headline"] == {"ic_t": 1.5}

    monkeypatch.setattr(holdout, "code_fingerprint", lambda root=None: "9" * 16)
    res = runner.invoke(cli.app, ["holdout", "evaluate", "--id", xsh["id"]])
    assert res.exit_code == 1 and "거부" in res.output and len(calls) == 2
    res = runner.invoke(cli.app, ["holdout", "evaluate", "--id", xsh["id"], "--force"])
    assert res.exit_code == 0 and "contaminated" in res.stdout
    assert holdout.load_registry(s).get(xsh["id"])["looks"][-1]["contaminated"] is True

    res = runner.invoke(cli.app, ["holdout", "status", "--json"])
    assert res.exit_code == 0
    rows = json.loads(res.stdout)
    assert {r["kind"] for r in rows} == {"forward", "cross_sectional"} and all(not r["frozen_ok"] for r in rows)
    res = runner.invoke(cli.app, ["holdout", "evaluate", "--id", "NOPE"])
    assert res.exit_code == 1


def test_cli_track_commands(cli_env, monkeypatch):
    cli, runner, s = cli_env
    res = runner.invoke(cli.app, ["holdout", "register", "--kind", "forward", "--market", "US"])
    assert res.exit_code == 0, res.output
    tracking.append_jsonl(tracking.ledger_paths(s)[0], [_pred("TP01")])
    res = runner.invoke(cli.app, ["track", "score"])
    assert res.exit_code == 0, res.output
    assert "배리어 신규 1건" in res.stdout and len(_lines(tracking.ledger_paths(s)[1])) == 1
    res = runner.invoke(cli.app, ["track", "report", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["verdict_available"] is False and payload["look"]["valid"] is True
    assert len(holdout.load_registry(s).active_forward()["looks"]) == 1
    res = runner.invoke(cli.app, ["track", "report"])
    assert res.exit_code == 0 and "판정 불가(표본 부족)" in res.stdout
    res = runner.invoke(cli.app, ["track", "report", "--holdout", "NOPE"])
    assert res.exit_code == 1

    fake = tracking.RecordResult(path=tracking.ledger_paths(s)[0], run_id="r", n_new=3, holdout_ids={"KR": None})
    monkeypatch.setattr(tracking, "snapshot", lambda market, settings, n, progress=None: fake)
    res = runner.invoke(cli.app, ["track", "snapshot", "--market", "KR", "--n", "3"])
    assert res.exit_code == 0 and "신규 3건" in res.stdout
    res = runner.invoke(cli.app, ["track", "snapshot", "--market", "JP"])
    assert res.exit_code == 2


def test_no_record_env_skips_the_ledger(tmp_path, monkeypatch):
    """QUANTCIO_NO_RECORD=1 (dev/verification runs) never appends to the git-tracked forward ledger."""
    from quantcio import tracking
    from quantcio.config import load_settings
    from quantcio.report.sample import build_sample_report

    s = load_settings(holdout={"ledger_dir": str(tmp_path / "ledger"), "registry_path": str(tmp_path / "reg.json")})
    monkeypatch.setenv(tracking.NO_RECORD_ENV, "1")
    res = tracking.record_predictions(build_sample_report(2, settings=s), s)
    assert res.n_new == 0 and not (tmp_path / "ledger").exists() and tracking.NO_RECORD_ENV in res.notes[0]
