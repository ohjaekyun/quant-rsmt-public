"""Offline end-to-end tests of the orchestration (quant/__init__.py, pipeline.py, cli.py).

Stub providers replace the network: every market has a 30-name synthetic universe plus the requested
tickers (conftest.make_prices random walks on one business-day calendar), so the whole path runs for real:
load_market_data -> load_price_panel -> build_panel -> fit_market_model -> predict_many ->
build_quant -> (FakeLLMClient | --no-llm) -> fuse -> execution plan -> barrier guardrail -> sizing ->
portfolio risk -> regime chips -> diagnostics -> HTML/PDF + JSON.
"""

from __future__ import annotations

import asyncio
import json
import math
import zlib
from datetime import date, datetime, timedelta
from html import escape as html_escape

import numpy as np
import pandas as pd
import pytest
from conftest import make_filings, make_news, make_prices
from typer.testing import CliRunner

from quantcio.config import Settings, load_settings
from quantcio.data import base as data_base
from quantcio.data.base import MarketProvider, normalise_ticker
from quantcio.llm.client import FakeLLMClient, LLMConfigError
from quantcio.models import (
    DistributionForecast,
    FilingItem,
    Fundamentals,
    Instrument,
    Market,
    MarketData,
    NewsItem,
    PortfolioReport,
)
from quantcio.quant import (
    build_quant,
    build_quant_signal,
    distribution_ml_pillar,
)
from quantcio.report import MODULE_HEADINGS, render_html

N_DAYS = 1600
UNIVERSE = 30
YOUNG = "450080"  # a recent listing: 300 sessions -> vol_pct_3y missing -> RSMT-Q degraded
BROKEN = "999990"  # resolve() raises -> reported in the notes, the run continues
KR_REQ = ["005930", "000660", "247540"]
US_REQ = ["AAPL", "NVDA", "MSFT"]
NAMES = {"005930": "삼성전자", "000660": "SK하이닉스", "247540": "에코프로비엠", YOUNG: "신규상장"}


def _seed(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode()) % (2**31)


class StubProvider(MarketProvider):
    """Deterministic offline provider (same interface as KRProvider / USProvider)."""

    def __init__(self, market: Market, settings: Settings, cache) -> None:
        super().__init__(settings, cache)
        self.market = market
        self.calls: list[str] = []

    # ---- identity
    def resolve(self, ticker: str) -> Instrument:
        t = normalise_ticker(ticker)
        if t == BROKEN:
            raise LookupError(f"{t}: 종목을 찾을 수 없음(스텁)")
        if self.market is Market.KR:
            kosdaq = t == "247540" or (t.startswith("9") and int(t) % 2 == 1)
            bench = "KOSDAQ" if kosdaq else "KOSPI"
            return Instrument(
                ticker=t,
                market=Market.KR,
                name=NAMES.get(t, f"테스트{t}"),
                exchange=bench,
                currency="KRW",
                sector="테스트",
                yf_symbol=f"{t}.KS",
                benchmark=bench,
                benchmark_name=bench,
            )
        return Instrument(
            ticker=t,
            market=Market.US,
            name=f"{t} Inc.",
            exchange="NASDAQ",
            currency="USD",
            sector="Technology",
            yf_symbol=t,
            benchmark="SPY",
            benchmark_name="S&P 500 (SPY)",
        )

    # ---- prices
    def _bench_frame(self, bench: str) -> pd.DataFrame:
        if bench == "^VIX":
            px = make_prices(n=60, start_price=16.0, drift=0.0, vol=0.03, seed=77)
            px["volume"] = 0.0
            return px
        start = {"KOSPI": 2500.0, "KOSDAQ": 800.0, "SPY": 450.0}.get(bench, 1000.0)
        return make_prices(n=N_DAYS, start_price=start, drift=0.0002, vol=0.011, seed=_seed("bench", bench))

    def prices(self, inst: Instrument, start: date) -> pd.DataFrame:
        self.calls.append(f"prices:{inst.ticker}")
        n = 300 if inst.ticker == YOUNG else N_DAYS
        bench = self._bench_frame(inst.benchmark)
        seed = _seed("px", inst.ticker)
        rng = np.random.default_rng(seed)
        base = make_prices(
            n=n, start_price=50_000.0 if self.market is Market.KR else 150.0, drift=0.0001, vol=0.017, seed=seed
        )
        blr = np.log(bench["close"]).diff().fillna(0.0).to_numpy()[-n:]
        beta = 0.7 + 0.6 * rng.random()
        ratio = np.exp(np.cumsum(beta * blr))
        out = base.copy()
        out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]].to_numpy() * ratio[:, None]
        return out

    def benchmark(self, inst: Instrument, start: date) -> pd.DataFrame:
        return self._bench_frame(inst.benchmark)

    # ---- the rest
    def fundamentals(self, inst: Instrument) -> Fundamentals:
        px = float(self.prices(inst, date.today())["close"].iloc[-1])
        return Fundamentals(
            source="stub",
            per=14.0,
            forward_per=12.0,
            pbr=1.3,
            roe=0.11,
            operating_margin=0.12,
            revenue_growth_yoy=0.08,
            earnings_growth_yoy=0.15,
            sector_per=15.0,
            analyst_target_price=px * 1.15,
            analyst_target_high=px * 1.35,
            analyst_count=12,
        )

    def news(self, inst: Instrument, since: datetime, limit: int) -> list[NewsItem]:
        return make_news(8)

    def filings(self, inst: Instrument, since: datetime, limit: int) -> list[FilingItem]:
        return make_filings()

    def fx_rate(self, from_ccy: str, to_ccy: str) -> float:
        if from_ccy == to_ccy:
            return 1.0
        return 1380.0 if from_ccy == "USD" else 1.0 / 1380.0

    def universe(self, n: int) -> list[str]:
        if self.market is Market.KR:
            return [f"{900000 + i}" for i in range(n)]
        return [f"U{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(n)]


@pytest.fixture
def stub_settings(tmp_path, monkeypatch) -> Settings:
    s = load_settings(
        data={"cache_dir": str(tmp_path / "cache")},
        report={"output_dir": str(tmp_path / "reports")},
        # never touch the git-tracked forward ledger / registry, never fetch live macro data offline
        holdout={"ledger_dir": str(tmp_path / "ledger"), "registry_path": str(tmp_path / "registry.json")},
        macro={"enabled": False},
        # the stub universe is today's list; the PIT universe path has its own test (monkeypatched loader)
        pit={"universe_mode": "current"},
    )
    for m in s.rsmt.markets.values():
        m.universe_size = UNIVERSE
    monkeypatch.setattr(
        data_base, "get_provider", lambda market, settings, cache: StubProvider(market, settings, cache)
    )
    return s


# --------------------------------------------------------------------------------- fake LLM


def _news_wire() -> dict:
    return {
        "articles": [
            {"news_id": n.id, "sentiment": 0.5, "relevance": 0.9, "event_type": "product", "evidence": n.title}
            for n in make_news(8)
        ],
        "key_positives": ["HBM4 퀄 테스트 통과"],
        "key_negatives": ["파운드리 수율 부진"],
        "key_events": ["자사주 추가 매입"],
    }


FILINGS_WIRE = {
    "summary": ["3분기 영업이익 10조 4,000억원으로 전년 대비 증가"],
    "earnings_highlights": ["매출액 79조 1,000억원"],
    "guidance": "none",
    "tone": 0.3,
    "risk_factors": [],
}
CRITIC_WIRE = {
    "thesis": "모멘텀과 실적 개선이 매수 논지를 지지하나 변동성에 유의.",
    "bull_points": ["실적 개선"],
    "bear_points": ["변동성"],
    "contradictions": [],
    "confidence_adjustment": -2,
    "critical_flag": False,
    "critical_category": None,
    "critical_reason": None,
    "critical_evidence": None,
    "critical_source_ref": None,
}


def fake_llm(settings: Settings) -> FakeLLMClient:
    def respond(task, system, user, wire_model):
        return {"news": _news_wire(), "filings": FILINGS_WIRE, "critic": CRITIC_WIRE}.get(task)

    return FakeLLMClient(respond, settings=settings, cost_per_call=0.01)


# ------------------------------------------------------------------------------- assertions


def _check_report_invariants(rep: PortfolioReport, settings: Settings) -> None:
    risk = settings.risk
    assert rep.generated_at.tzinfo is not None
    weights = [t.execution.weight for t in rep.tickers]
    assert all(0.0 <= w <= risk.max_weight + 1e-12 for w in weights)
    assert sum(weights) <= risk.max_gross + 1e-9
    assert rep.cash_weight == pytest.approx(1.0 - sum(weights))
    for t in rep.tickers:
        b = t.execution.barrier
        d = t.quant.distribution
        if d is not None:
            assert 0.0 <= d.prob_up <= 1.0 and d.q10 <= d.q50 <= d.q90
            assert d.horizon_days == settings.horizon_days
            assert b is not None, t.instrument.ticker
        if b is not None:
            assert b.p_take_profit + b.p_stop_loss + b.p_time_exit == 1.0  # exact, not approximately
            assert b.horizon_days == settings.fusion.target_horizon_days
        assert -1.0 <= t.quant.ml_pillar <= 1.0
        assert set(t.executive.pillars) <= {"factor", "ml", "llm"}


# ------------------------------------------------------------------------------------ tests


def test_run_report_no_llm_end_to_end(stub_settings, tmp_path):
    from quantcio.pipeline import NO_LLM_NOTE, run_report

    tickers = [*KR_REQ, *US_REQ, YOUNG, BROKEN]
    art = asyncio.run(run_report(tickers, stub_settings, capital=100_000_000, use_llm=False))
    rep = art.report
    got = [t.instrument.ticker for t in rep.tickers]
    assert got == [*KR_REQ, *US_REQ, YOUNG]  # input order, the broken ticker dropped
    assert any(BROKEN in n and "데이터 로드 실패" in n for n in rep.notes)
    assert NO_LLM_NOTE in rep.notes
    _check_report_invariants(rep, stub_settings)
    assert all(t.llm.skipped for t in rep.tickers) and rep.llm_cost_usd == 0.0

    # RSMT-Q served from ONE batch per market with the holding-horizon barrier
    by = {t.instrument.ticker: t for t in rep.tickers}
    for t in [*KR_REQ, *US_REQ]:
        d = by[t].quant.distribution
        assert d is not None and d.cv_metrics and "auc_gain_p05" in d.cv_metrics
    # capital -> shares: KRW names at fx 1, USD names at fx 1380
    for t in rep.tickers:
        if t.execution.weight > 0:
            fx = 1.0 if t.instrument.currency == "KRW" else 1380.0
            assert t.execution.shares == math.floor(t.execution.position_value / (t.execution.entry_price * fx) + 1e-9)

    # every name carries the GARCH(1,1)-t sizing vol (EWMA fallback), the young listing included
    assert all(t.quant.vol_forecast_annual is not None and t.quant.vol_forecast_annual > 0 for t in rep.tickers)
    # degraded paths propagate: RSMT-Q degraded row for the young listing
    young = by[YOUNG]
    d = young.quant.distribution
    assert d is not None and d.degraded and any("vol_pct_3y" in r for r in d.degraded_reasons)
    assert any(w.startswith("RSMT-Q 저하 모드") for w in young.warnings)
    assert any(o.startswith("RSMT 저하 모드") for o in young.executive.overrides)
    # the degraded pillar is halved relative to the undegraded formula
    undegraded = distribution_ml_pillar(d.model_copy(update={"degraded": False}))
    assert young.quant.ml_pillar == pytest.approx(0.5 * undegraded)

    # regime chips, diagnostics, digest
    assert {"KOSPI", "KOSDAQ", "S&P 500", "VIX", "USD/KRW"} <= set(rep.market_regime)
    assert rep.market_regime["USD/KRW"] == "1,380"
    assert set(rep.model_diagnostics) == {"KR", "US"}
    for lines in rep.model_diagnostics.values():
        assert any(ln.startswith("RSMT-Q ") for ln in lines)
        assert any(ln.startswith("스킬 게이트") for ln in lines)
        assert any(ln.startswith("[정상]") or ln.startswith("[저하]") for ln in lines)
    assert rep.config_digest == stub_settings.digest()
    assert set(rep.portfolio_risk) >= {"ex_ante_vol", "var95", "p_loss"}

    # artifacts: JSON round-trips, HTML has the 4 modules and the degraded warning
    assert art.json_path.parent.parent == tmp_path / "reports"
    again = PortfolioReport.model_validate_json(art.json_path.read_text(encoding="utf-8"))
    assert [t.instrument.ticker for t in again.tickers] == got
    html = art.html_path.read_text(encoding="utf-8")
    for heading in MODULE_HEADINGS.values():
        assert html_escape(heading) in html
    assert "vol_pct_3y" in html
    if art.pdf_path is not None:  # a browser is available
        assert art.pdf_path.exists() and 1 <= art.pages <= stub_settings.report.max_pages
    else:
        assert art.pages == 0 and art.warnings
    assert {"data+fit", "predict", "quant", "fusion", "render"} <= set(art.timings)


def test_build_report_with_fake_llm_and_model_cache(stub_settings):
    from quantcio.pipeline import build_report

    tickers = ["005930", "AAPL", "NVDA"]
    fake = fake_llm(stub_settings)
    res = asyncio.run(build_report(tickers, stub_settings, capital=None, use_llm=True, llm_client=fake))
    rep = res.report
    _check_report_invariants(rep, stub_settings)
    assert not any(f.cached for f in res.fits.values())
    for t in rep.tickers:
        assert not t.llm.skipped and t.llm.errors == []
        assert t.llm.news is not None and t.llm.news.n_used == 8
        assert t.llm.filings is not None and t.llm.critic is not None
        assert "llm" in t.executive.pillars
        assert t.executive.pillars["llm"] == pytest.approx(t.llm.llm_pillar)
        assert t.executive.thesis.startswith("모멘텀과 실적 개선")
        assert t.execution.shares is None and t.execution.position_value is None
    tasks = [c.task for c in fake.calls]
    assert tasks.count("news") == tasks.count("filings") == tasks.count("critic") == len(tickers)
    assert rep.llm_cost_usd == pytest.approx(0.01 * len(fake.calls))
    html = render_html(rep, stub_settings)
    for heading in MODULE_HEADINGS.values():
        assert html_escape(heading) in html

    # same data snapshot -> the fitted models are reused from the disk cache, identical diagnostics
    res2 = asyncio.run(build_report(tickers, stub_settings, use_llm=False))
    assert all(f.cached for f in res2.fits.values())
    assert res2.report.model_diagnostics == rep.model_diagnostics
    for a, b in zip(rep.tickers, res2.report.tickers):
        assert a.quant.distribution.mu == b.quant.distribution.mu
        assert a.quant.distribution.prob_up == b.quant.distribution.prob_up


def test_llm_config_error_is_a_clear_korean_error(stub_settings, monkeypatch):
    import quantcio.pipeline as pipeline

    def no_key(settings):
        raise LLMConfigError("Anthropic 자격 증명이 없습니다")

    monkeypatch.setattr(pipeline, "LLMClient", no_key)
    with pytest.raises(pipeline.PipelineError, match="--no-llm"):
        asyncio.run(pipeline.build_report(["005930"], stub_settings, use_llm=True))


def test_pipeline_rejects_bad_arguments_and_all_failed(stub_settings):
    from quantcio.pipeline import PipelineError, build_report, parse_tickers

    assert parse_tickers("005930, aapl 005930.KS  BRK.B") == ["005930", "aapl", "BRK.B"]
    with pytest.raises(PipelineError):
        asyncio.run(build_report("", stub_settings, use_llm=False))
    with pytest.raises(PipelineError, match="KRW 또는 USD"):
        asyncio.run(build_report("005930", stub_settings, base_currency="EUR", use_llm=False))
    with pytest.raises(PipelineError, match="데이터 로드에 실패"):
        asyncio.run(build_report([BROKEN], stub_settings, use_llm=False))


def test_market_fit_failure_is_reported_per_ticker(stub_settings, monkeypatch):
    import quantcio.pipeline as pipeline

    def boom(*a, **k):
        raise RuntimeError("panel down")

    monkeypatch.setattr(data_base, "load_price_panel", boom)
    res = asyncio.run(pipeline.build_report(["005930"], stub_settings, use_llm=False))
    t = res.report.tickers[0]
    assert t.quant.distribution is None and t.execution.barrier is None
    assert any("RSMT-Q KR 모델 없음" in w and "panel down" in w for w in t.warnings)
    assert any("모델 적합 실패" in n for n in res.report.notes)
    assert res.report.model_diagnostics["KR"][0].startswith("⚠ RSMT-Q KR 모델 적합 실패")
    assert "KOSPI" in res.report.market_regime  # the ticker's own benchmark still gives the chip


# ------------------------------------------------------------------------ quant/__init__.py


def _dist_fc(**kw) -> DistributionForecast:
    base = dict(
        horizon_days=20,
        mu=0.02,
        sigma=0.08,
        nu=6.0,
        beta=1.0,
        mu_excess=0.01,
        sigma_excess=0.07,
        mu_market=0.01,
        sigma_market=0.04,
        rho=0.0,
        prob_up=0.6,
        prob_outperform=0.55,
        expected_return=0.023,
        q10=-0.08,
        q50=0.02,
        q90=0.12,
        var95=-0.11,
        cvar95=-0.15,
    )
    base.update(kw)
    return DistributionForecast(**base)


def test_distribution_ml_pillar_multipliers():
    raw = 0.5 * math.tanh(2 * 0.02 / 0.08) + 0.5 * (2 * 0.6 - 1)
    # the multiplier reads the DIRECTION verdict (docs/RSMT.md): a distribution-skill pass alone never un-shrinks it
    assert distribution_ml_pillar(_dist_fc(direction_skill_ok=True)) == pytest.approx(raw)
    assert distribution_ml_pillar(_dist_fc(direction_skill_ok=False)) == pytest.approx(0.35 * raw)
    assert distribution_ml_pillar(_dist_fc(skill_ok=True, direction_skill_ok=False)) == pytest.approx(0.35 * raw)
    assert distribution_ml_pillar(_dist_fc(direction_skill_ok=True, degraded=True)) == pytest.approx(0.5 * raw)
    assert distribution_ml_pillar(_dist_fc(direction_skill_ok=False, degraded=True)) == pytest.approx(0.175 * raw)
    with pytest.raises(ValueError):
        distribution_ml_pillar(_dist_fc(sigma=0.0))
    with pytest.raises(ValueError):
        distribution_ml_pillar(_dist_fc(mu=float("nan")))


def test_build_quant_signal_reports_every_degraded_path(settings, md_kr, monkeypatch):
    from quantcio.quant import volatility
    from quantcio.quant.volatility import VolatilityError, VolForecast

    # 1) no RSMT-Q model -> no distribution, pillar 0; the GARCH sizing vol is still computed
    q = build_quant_signal(md_kr, settings)
    assert q.distribution is None and q.ml_pillar == 0.0
    assert q.vol_forecast_annual is not None and q.vol_forecast_annual > 0

    # 2) the EWMA volatility fallback is a data warning; no estimate at all -> None + warning (sizing uses rv)
    ewma = VolForecast(annual_vol=0.31, method="ewma", horizon_days=20, n_obs=100, fallback_reason="표본 부족")
    monkeypatch.setattr(volatility, "garch_vol_forecast_ex", lambda close, h: ewma)
    md2 = MarketData(**{**md_kr.__dict__, "warnings": []})
    assert build_quant_signal(md2, settings).vol_forecast_annual == pytest.approx(0.31)
    assert ewma.note() in md2.warnings and ewma.note().startswith("변동성: EWMA")

    def fail(close, h):
        raise VolatilityError("EWMA 변동성 추정 불가: 수익률 5개")

    monkeypatch.setattr(volatility, "garch_vol_forecast_ex", fail)
    md2b = MarketData(**{**md_kr.__dict__, "warnings": []})
    assert build_quant_signal(md2b, settings).vol_forecast_annual is None
    assert any(w.startswith("변동성 예측 불가(VolatilityError)") for w in md2b.warnings)
    monkeypatch.undo()

    # 3) a served exception (predict_many returns it in place) is a warning, never silent
    md3 = MarketData(**{**md_kr.__dict__, "warnings": []})
    res = build_quant(md3, settings, served=KeyError("005930: RSMT-Q KR 모델에 예측 행 없음 (패널에 없음)"))
    assert res.dist is None and res.signal.distribution is None and res.signal.ml_pillar == 0.0
    assert any(w.startswith("RSMT-Q 예측 실패(KeyError): 005930") for w in md3.warnings)


def test_old_config_with_the_removed_ensemble_keys_still_loads(tmp_path):
    """The per-ticker ensemble was removed 2026-10-02: an old override YAML with its `ml:` block and
    rsmt.serving_blend still loads (the keys are ignored) and leaves the effective config unchanged."""
    from quantcio.config import load_settings

    old = tmp_path / "old.yaml"
    old.write_text(
        "ml:\n  horizon_days: 20\n  embargo_days: 20\n  skill_min_ic: 0.03\n"
        "rsmt:\n  serving_blend:\n    rsmt: 0.8\n    ensemble: 0.2\n",
        encoding="utf-8",
    )
    s = load_settings(old)
    assert not hasattr(s, "ml") and not hasattr(s.rsmt, "serving_blend")
    assert s.digest() == load_settings().digest()


# ------------------------------------------------------------------------------------ CLI


def test_cli_analyze_json_and_rsmt(stub_settings, monkeypatch):
    import quantcio.cli as cli

    monkeypatch.setattr(cli, "load_settings", lambda config=None: stub_settings)
    monkeypatch.setattr(cli, "_setup_logging", lambda settings, verbose: stub_settings.cache_path / "test.log")
    runner = CliRunner()
    res = runner.invoke(cli.app, ["analyze", "005930", "--json", "--no-llm"])
    assert res.exit_code == 0, res.output
    payload, _ = json.JSONDecoder().raw_decode(res.stdout[res.stdout.index("{") :])
    assert payload["instrument"]["ticker"] == "005930"
    assert payload["execution"]["barrier"]["horizon_days"] == stub_settings.fusion.target_horizon_days

    res = runner.invoke(cli.app, ["rsmt", "--market", "KR"])
    assert res.exit_code == 0, res.output
    assert "RSMT-Q KR" in res.stdout and "스킬 게이트" in res.stdout and "CRPS" in res.stdout

    res = runner.invoke(cli.app, ["rsmt", "--market", "JP"])
    assert res.exit_code == 2


def test_tune_rsmt_recommends_per_metric(stub_settings):
    from quantcio.data.cache import DiskCache
    from quantcio.pipeline import fit_market, tune_rsmt

    fit = fit_market(Market.US, stub_settings, DiskCache(stub_settings.cache_path), ())
    assert fit.model is not None, fit.error
    tr = tune_rsmt(fit.model.panel, stub_settings, grid={"nu_min": (4.5, 8.0), "alpha": (0.05, 0.10)})
    assert [r.param for r in tr.rows] == ["nu_min", "nu_min", "alpha", "alpha"]
    assert sum(r.current for r in tr.rows) == 2  # nu_min 4.5 and alpha 0.05 are the US defaults
    assert all({"crps", "cover80", "logloss"} <= set(r.metrics) for r in tr.rows)
    assert all(isinstance(r.skill_ok, bool) for r in tr.rows)  # the full CV gate, not the AUC gain alone
    assert len(tr.recommendations) == 2
    assert tr.recommendations[0].startswith("nu_min") and tr.recommendations[1].startswith("alpha")
    assert stub_settings.rsmt.markets["US"].nu_min == 4.5  # recommendations only: config untouched


def test_panel_gap_notes_flag_bars_missing_against_the_benchmark():
    """Live finding 2026-09-24: yfinance served 44/60 US universe names without the 2026-09-22 bar. The
    provider notes for universe names only reach the log, so the diagnostics box must say it."""
    from quantcio.data.base import PricePanel
    from quantcio.pipeline import panel_gap_notes

    bench = make_prices(n=300, seed=1)
    full = make_prices(n=300, seed=2)
    gap_day = bench.index[-2]
    holed = full.drop(index=gap_day)
    panel = PricePanel(
        Market.US,
        {"AAA": full, "BBB": holed, "CCC": holed.copy()},
        {"SPY": bench},
        {"AAA": "SPY", "BBB": "SPY", "CCC": "SPY"},
    )
    notes = panel_gap_notes(panel)
    assert len(notes) == 1 and notes[0].startswith("[정보] 패널 데이터")
    assert "2/3종목" in notes[0] and f"{gap_day.date()}×2" in notes[0]
    assert panel_gap_notes(PricePanel(Market.US, {"AAA": full}, {"SPY": bench}, {"AAA": "SPY"})) == []


def test_cli_prints_overrides_and_llm_errors_without_ticker_warnings(monkeypatch):
    """QA regression: overrides / LLM errors were printed only when some ticker ALSO had a data warning,
    so a run whose only alarms were guardrail downgrades printed none of them."""
    from rich.console import Console

    import quantcio.cli as cli
    from quantcio.config import load_settings
    from quantcio.report.sample import build_sample_report

    rep = build_sample_report(2, settings=load_settings())
    for t in rep.tickers:
        t.warnings = []
        t.llm.errors = []
    rep.tickers[0].executive.overrides = ["배리어 가드레일: P(손절 선행) 55% > P(익절 선행) 36% + 15%p → Buy→Hold"]
    rep.tickers[1].llm.errors = ["news: [refusal] stop_reason"]
    rep.notes = []
    rec = Console(record=True, width=200)
    monkeypatch.setattr(cli, "console", rec)
    cli._print_warnings(rep)
    out = rec.export_text()
    assert "배리어 가드레일" in out and "news: [refusal] stop_reason" in out  # markup-looking text kept verbatim

    # run-wide --no-llm: the identical 'LLM 분석 미실행' override is printed once as a common note
    skip = "LLM 분석 미실행 — LLM 필러 제외 후 팩터·ML 가중치 재정규화"
    for t in rep.tickers:
        t.llm.skipped, t.llm.errors = True, []
        t.executive.overrides = [skip]
    rec = Console(record=True, width=200)
    monkeypatch.setattr(cli, "console", rec)
    cli._print_warnings(rep)
    out = rec.export_text()
    assert out.count("LLM 분석 미실행") == 1 and "전 종목 공통" in out


def test_cli_console_log_has_no_traceback_but_the_file_does(tmp_path):
    """QA regression: a mistyped ticker printed a ~100-line traceback per ticker on the console."""
    import logging
    import sys

    import quantcio.cli as cli

    s = load_settings(data={"cache_dir": str(tmp_path / "cache")})
    log_file = cli._setup_logging(s, verbose=False)
    try:
        try:
            raise ValueError("KR 종목 999999 식별 실패")
        except ValueError:
            rec = logging.getLogger("quantcio.pipeline").makeRecord(
                "quantcio.pipeline", logging.ERROR, __file__, 1, "%s: data load failed", ("999999",), sys.exc_info()
            )
        console_h = next(
            h
            for h in logging.getLogger().handlers
            if getattr(h, "_quantcio", False) and h is not None and not isinstance(h, logging.FileHandler)
        )
        line = console_h.format(rec)
        assert "\n" not in line and "Traceback" not in line
        assert "999999: data load failed (ValueError: KR 종목 999999 식별 실패)" in line
        file_h = next(
            h
            for h in logging.getLogger().handlers
            if isinstance(h, logging.FileHandler) and getattr(h, "_quantcio", False)
        )
        assert "Traceback" in file_h.format(rec)
    finally:
        for h in list(logging.getLogger().handlers):
            if getattr(h, "_quantcio", False):
                logging.getLogger().removeHandler(h)
                h.close()
    assert log_file.parent.exists()


def test_market_model_does_not_depend_on_the_other_requested_tickers(stub_settings):
    """QA regression (live 2026-09-24): the KR gate's date-bootstrap bound was +0.0006 on the universe and
    -0.0047 with 247540 added, so 005930/000660 flipped Buy (9.5%) <-> Hold (0%) on the same data depending
    only on whether 247540 was in the same request. Since 2026-10-01 the whole market model (tree, offsets, sigma
    equation, CV gate, v centering) is fitted on the universe only and reused; requested names are only served."""
    from quantcio.data.cache import DiskCache
    from quantcio.pipeline import fit_market

    cache = DiskCache(stub_settings.cache_path)
    base = fit_market(Market.KR, stub_settings, cache, ())
    one = fit_market(Market.KR, stub_settings, cache, KR_REQ[:1])
    more = fit_market(Market.KR, stub_settings, cache, [*KR_REQ, YOUNG])
    for f in (base, one, more):
        assert f.model is not None, f.error
    assert one.cached and more.cached  # fitted once, then reused whatever else is requested
    assert one.model.cv.metrics == base.model.cv.metrics == more.model.cv.metrics
    assert one.model.skill_ok == base.model.skill_ok == more.model.skill_ok
    assert one.model.tree.describe() == base.model.tree.describe() == more.model.tree.describe()
    assert one.model.tree.n_train == base.model.tree.n_train == more.model.tree.n_train
    # the requested names are served on the fitted v scale, without entering the fit
    assert more.model.panel.pred["ticker"].nunique() > base.model.panel.pred["ticker"].nunique()
    assert more.model.panel.c_v == base.model.panel.c_v
    note = [n for n in one.notes if n.startswith("[정보] RSMT-Q 시장 모델")]
    assert len(note) == 1 and "매 거래일" in note[0]
    refit = fit_market(Market.KR, stub_settings, cache, KR_REQ[:1], use_cache=False)
    assert refit.model is not None and not refit.cached
    assert refit.model.tree.describe() == base.model.tree.describe()  # deterministic


def test_macro_overlay_scales_gross_for_buy_candidates(stub_settings, monkeypatch):
    """Risk-off regime of a Buy candidate's market scales max_gross; the reason is noted; failures never block."""
    import quantcio.macro as macro_mod
    import quantcio.pipeline as pipeline
    from quantcio.models import MacroRegime, Signal

    s = stub_settings.model_copy(deep=True)
    s.macro.enabled = True
    today = date(2026, 9, 24)
    regs = [
        MacroRegime(region="global", as_of=today, score=0.1, label="중립", exposure_mult=1.0),
        MacroRegime(region="us", as_of=today, score=-0.5, label="위험회피", exposure_mult=0.8),
        MacroRegime(region="kr", as_of=today, score=0.3, label="위험선호", exposure_mult=1.0),
    ]
    monkeypatch.setattr(macro_mod, "current_regimes", lambda settings, cache: regs)

    class _Run:
        def __init__(self, market, sig):
            self.market, self.execsig = market, type("E", (), {"signal": sig})()

    runs = [_Run(Market.US, Signal.BUY), _Run(Market.KR, Signal.HOLD)]
    notes: list[str] = []
    out_regs, sized = pipeline.macro_overlay(runs, s, None, notes)
    assert out_regs == regs
    assert sized.risk.max_gross == pytest.approx(s.risk.max_gross * 0.8)
    assert s.risk.max_gross == stub_settings.risk.max_gross  # caller's settings untouched
    assert any("매크로 오버레이" in n for n in notes)

    def boom(settings, cache):
        raise RuntimeError("offline")

    monkeypatch.setattr(macro_mod, "current_regimes", boom)
    notes2: list[str] = []
    regs2, sized2 = pipeline.macro_overlay(runs, s, None, notes2)
    assert regs2 == [] and sized2 is s and any("매크로 레짐 수집 실패" in n for n in notes2)


def test_cli_lang_option_english_report_and_analyze(stub_settings, monkeypatch, tmp_path):
    """--lang en: English PDF/HTML report and English console output; a bad value exits with 2."""
    import re

    import quantcio.cli as cli

    monkeypatch.setattr(cli, "_LANG", "ko")  # restored after the test (the CLI keeps the console language global)
    monkeypatch.setattr(cli, "load_settings", lambda config=None: stub_settings.model_copy(deep=True))
    monkeypatch.setattr(cli, "_setup_logging", lambda settings, verbose: stub_settings.cache_path / "test.log")
    runner = CliRunner()

    res = runner.invoke(cli.app, ["analyze", "005930", "--no-llm", "--lang", "xx"])
    assert res.exit_code == 2

    res = runner.invoke(cli.app, ["analyze", "005930", "--no-llm", "--lang", "en"])
    assert res.exit_code == 0, res.output
    assert "Signal" in res.stdout and "Plan: entry" in res.stdout and "Action:" in res.stdout
    assert "₩" in res.stdout  # English money format for a KRW name

    from quantcio.tracking import NO_RECORD_ENV, ledger_paths

    monkeypatch.setenv(NO_RECORD_ENV, "0")  # --no-record sets it for the process; restored after the test
    out = tmp_path / "en_report"
    args = ["report", "-t", "005930,AAPL", "--no-llm", "--lang", "en", "--out", str(out), "--no-record"]
    res = runner.invoke(cli.app, args)
    assert res.exit_code == 0, res.output
    assert not ledger_paths(stub_settings)[0].exists()  # --no-record: nothing entered the forward ledger
    html = next(out.rglob("quantcio_report_*.html")).read_text(encoding="utf-8")
    assert '<html lang="en">' in html and "Quant · AI Trade Timing Report" in html
    assert "Executive Signal Matrix" in html
    body = re.sub(r"<style.*?</style>|<svg.*?</svg>", "", html, flags=re.S)
    labels = ("Final signal", "Portfolio summary", "Entry zone", "Methodology")
    assert all(lbl in body for lbl in labels)
    assert "Report notes" in res.stdout or "Ticker warnings" in res.stdout or "Invested" in res.stdout

    res = runner.invoke(cli.app, ["analyze", "005930", "--no-llm"])  # default stays Korean
    assert res.exit_code == 0 and "신호" in res.stdout


def test_selected_model_serves_the_excess_mean_and_factor_notes(stub_settings):
    """With a cached stock-selection artifact (quantcio modeling select) the served distribution's excess mean
    comes from the selected model, QuantSignal.excess_model is populated and the diagnostics say so; without a
    factor-backtest cache the report notes it and still renders."""
    from test_modeling import synthetic_dataset

    from quantcio.data.cache import DiskCache
    from quantcio.modeling import selection as sel
    from quantcio.modeling import serve
    from quantcio.modeling.candidates import default_candidates
    from quantcio.pipeline import build_report

    ds = synthetic_dataset(n_names=30, years=7)
    cache = DiskCache(stub_settings.cache_path)
    cands = [c for c in default_candidates() if c.name in ("mom_12_1", "ridge", "logit")]
    res = sel.run_selection(ds, stub_settings, cache, candidates=cands, threads=2, use_cache=False)
    art = serve.build_artifact(res, ds, stub_settings)
    art.latest.loc[0, ["entity", "ticker"]] = ["005930", "005930"]
    serve.save_artifact(art, stub_settings, cache)

    run = asyncio.run(build_report(["005930", "000660"], stub_settings, use_llm=False, cache=cache))
    by = {t.instrument.ticker: t for t in run.report.tickers}
    em = by["005930"].quant.excess_model
    assert em is not None and em.in_universe and em.model == art.selected
    d = by["005930"].quant.distribution
    assert d is not None and d.blend == {art.selected: 1.0}
    assert d.mu_excess == pytest.approx(em.mu_excess, abs=1e-9)
    assert em.p_beat_benchmark == pytest.approx(d.prob_outperform)
    assert d.skill_ok == art.skill_ok
    outside = by["000660"].quant.excess_model
    assert outside is not None and not outside.in_universe
    diag = " ".join(run.report.model_diagnostics.get("KR", []))
    assert "종목 선택 모델" in diag and "시장 방향 요소" in diag
    assert any("팩터 백테스트 캐시" in n for n in run.report.notes)
    assert by["005930"].quant.factor_model is None and run.report.factor_backtest == {}


def test_requested_names_are_aligned_to_a_lagging_universe():
    """The universe one session behind the requested names' production prices: the requested frames are cut to the
    universe's last common bar (RSMT-Q panel only), so the latest cross-section has its members (no zeroed z)."""
    import pandas as pd

    from quantcio.config import load_settings
    from quantcio.data.base import PricePanel
    from quantcio.models import Market
    from quantcio.pipeline import MAX_ALIGN_SESSIONS, align_extras_to_universe

    s = load_settings()
    idx = pd.bdate_range("2026-06-01", "2026-09-25")
    full = pd.DataFrame({"close": 100.0, "volume": 1.0}, index=idx)
    members = {f"CIK{i:010d}": full.iloc[:-1] for i in range(10)}  # last bar 09-24
    extras = {"AAPL": full, "FAR": full}  # production prices up to 09-25
    mem = pd.DataFrame([(pd.Timestamp("2026-08-31"), t) for t in members], columns=["date", "ticker"])
    panel = PricePanel(Market.US, {**members, **extras}, {}, {}, {}, membership=mem)
    notes = align_extras_to_universe(panel, ["AAPL"], s)
    assert panel.prices["AAPL"].index[-1] == pd.Timestamp("2026-09-24") and len(notes) == 1 and "2026-09-24" in notes[0]
    assert panel.prices["FAR"].index[-1] == pd.Timestamp("2026-09-25")  # not a requested name: untouched
    # a universe far behind is a data problem to see, not to hide
    old = {t: full.iloc[: -(MAX_ALIGN_SESSIONS + 2)] for t in members}
    panel2 = PricePanel(Market.US, {**old, "AAPL": full}, {}, {}, {}, membership=mem)
    assert align_extras_to_universe(panel2, ["AAPL"], s) == [] and panel2.prices["AAPL"].index[-1] == idx[-1]


def test_rsmt_history_start_and_kr_marcap_years(monkeypatch, tmp_path):
    from datetime import date as _date

    import quantcio.data.pit.kr as kr
    from quantcio.config import load_settings
    from quantcio.pipeline import rsmt_history_start

    s = load_settings(data={"cache_dir": str(tmp_path)})
    assert rsmt_history_start(Market.KR, s) == _date(1995, 9, 1)
    assert rsmt_history_start(Market.US, s) == _date(2009, 8, 31)
    s.rsmt.markets["US"].history_start = None
    assert rsmt_history_start(Market.US, s, today=_date(2026, 10, 1)) == _date(2026, 10, 1) - timedelta(
        days=int(365.25 * s.history_years) + 30
    )
    seen = {}
    monkeypatch.setattr(
        kr.km, "ensure_marcap_files", lambda root, years, now, session=None: seen.setdefault("y", years)
    )
    kr.marcap_files(s, datetime(2026, 10, 1))
    assert seen.pop("y")[0] == s.pit.kr.start_year  # other consumers keep their file set
    kr.marcap_files(s, datetime(2026, 10, 1), first_year=1995)
    assert seen["y"][0] == 1995 and seen["y"][-1] == 2026


def test_fallback_universe_fit_is_served_but_never_cached(tmp_path, monkeypatch):
    """PIT configured but its load failed (universe None -> current list): the fit must not be cached as the market
    model (live 2026-10-01: a failed 1994 marcap download would otherwise have served a survivorship-biased model for
    rsmt.model_max_age_days)."""
    from types import SimpleNamespace

    import quantcio.pipeline as pipeline_mod
    import quantcio.quant.panel as panel_mod
    import quantcio.quant.rsmt as rsmt_mod
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache
    from quantcio.pipeline import MODEL_CACHE_NS, market_model_fit, market_model_key

    s = load_settings(data={"cache_dir": str(tmp_path)})
    s.pit.enabled, s.pit.universe_mode = True, "pit"
    cache = DiskCache(s.cache_path)
    train = pd.DataFrame({"date": pd.to_datetime(["2026-09-01"]), "ticker": ["A"]})
    fake_panel = SimpleNamespace(train=train, c_v=0.1, as_of=pd.Timestamp("2026-09-30"), notes=[])
    monkeypatch.setattr(panel_mod, "build_panel", lambda price, settings: fake_panel)
    monkeypatch.setattr(rsmt_mod, "fit_market_model", lambda pdata, settings: SimpleNamespace(tree="T", cv="C"))
    monkeypatch.setattr(pipeline_mod, "universe_price_panel", lambda market, price, settings, cache, universe: price)
    price = SimpleNamespace(prices={"A": None})
    fitted, cached = market_model_fit(Market.KR, price, s, cache, universe=None)
    assert fitted["tree"] == "T" and not cached
    assert cache.get(MODEL_CACHE_NS, market_model_key(Market.KR, s), None) is None
    fitted, cached = market_model_fit(Market.KR, price, s, cache, universe={"A"})
    assert cache.get(MODEL_CACHE_NS, market_model_key(Market.KR, s), None)["tree"] == "T"
    assert market_model_fit(Market.KR, price, s, cache, universe={"A"})[1] is True  # reused
