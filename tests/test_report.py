"""Tests for quantcio.report: sample report consistency, charts, HTML view/template, PDF rendering.

PDF tests need a Chromium-family browser (the installed Google Chrome on the dev Mac); they are skipped,
not failed, when none can be launched. Nothing here touches the network.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import make_prices

from quantcio.models import (
    FilingIntelligence,
    PortfolioReport,
    RiskFactor,
    Signal,
)
from quantcio.report import charts as charts_mod
from quantcio.report import render as render_mod
from quantcio.report.charts import fig_to_svg, price_chart_svg, sparkline_svg
from quantcio.report.render import (
    DISCLAIMER,
    MISSING,
    MODULE_HEADINGS,
    REPORT_TITLE,
    NoBrowserError,
    PdfRenderer,
    build_view,
    count_pdf_pages,
    fit_to_pages,
    fmt_money_compact,
    fmt_num,
    fmt_pct,
    fmt_price,
    fmt_prob,
    footer_template,
    render_html,
    render_pdf,
)
from quantcio.report.sample import (
    SAMPLE_NOTE,
    SAMPLE_TICKERS,
    build_sample_charts,
    build_sample_report,
    sample_price_history,
)

# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def module_settings():
    from quantcio.config import load_settings

    return load_settings()


@pytest.fixture(scope="module")
def sample6(module_settings) -> PortfolioReport:
    return build_sample_report(6, settings=module_settings)


@pytest.fixture(scope="module")
def sample12(module_settings) -> PortfolioReport:
    return build_sample_report(12, settings=module_settings)


@pytest.fixture(scope="module")
def pdf_renderer():
    """One Chrome process for all PDF tests in this module; skip when no browser can start."""
    try:
        r = PdfRenderer().__enter__()
    except NoBrowserError as exc:
        pytest.skip(f"no Chromium-family browser available: {str(exc).splitlines()[0]}")
    yield r
    r.__exit__(None, None, None)


def _pdf_text(path: Path) -> list[str]:
    from pypdf import PdfReader

    # Chrome 156 (auto-update 2026-10-27) writes the space glyph so that pypdf extracts it as U+0001; the page looks
    # the same, so the text checks treat it as the space it is
    return [(p.extract_text() or "").replace("\x01", " ") for p in PdfReader(str(path)).pages]


def _visible_text(html: str) -> str:
    body = re.sub(r"<style.*?</style>", "", html, flags=re.S)
    body = re.sub(r"<svg.*?</svg>", "", body, flags=re.S)
    return re.sub(r"<[^>]+>", " ", body)


# --------------------------------------------------------------------------- sample report


@pytest.mark.parametrize("n", [1, 4, 6, 8, 12])
def test_sample_report_is_internally_consistent(n, module_settings):
    rep = build_sample_report(n, settings=module_settings)
    th = module_settings.fusion.thresholds
    assert len(rep.tickers) == n
    assert rep.notes and rep.notes[0] == SAMPLE_NOTE
    assert [t.instrument.ticker for t in rep.tickers] == list(SAMPLE_TICKERS[:n])
    assert set(rep.model_diagnostics) == {t.instrument.market.value for t in rep.tickers}
    assert rep.config_digest == module_settings.digest()

    total_w = 0.0
    for t in rep.tickers:
        ex, plan, d = t.executive, t.execution, t.quant.distribution
        assert d is not None
        # one distribution -> ordered quantiles and tails
        assert d.q10 <= d.q50 <= d.q90
        assert d.cvar95 <= d.var95 <= d.q10
        assert 0 <= d.prob_up <= 1 and 0 <= d.prob_outperform <= 1
        # barrier outcomes are counts / n_paths and sum to exactly 1
        b = plan.barrier
        assert b is not None
        assert b.p_take_profit + b.p_stop_loss + b.p_time_exit == pytest.approx(1.0, abs=1e-12)
        assert b.horizon_days == module_settings.fusion.target_horizon_days  # holding horizon, not h
        assert 0.0 <= b.kelly_optimal <= 1.0
        assert plan.kelly_fraction == pytest.approx(b.kelly_optimal * module_settings.risk.kelly_cap)
        # plan geometry
        assert plan.stop_loss < plan.entry_price < plan.take_profit
        assert plan.entry_low < plan.entry_price < plan.entry_high
        assert plan.stop_loss_pct == pytest.approx(plan.stop_loss / plan.entry_price - 1)
        assert plan.reward_risk == pytest.approx(
            (plan.take_profit - plan.entry_price) / (plan.entry_price - plan.stop_loss)
        )
        assert plan.reward_risk >= module_settings.risk.min_reward_risk - 1e-9
        assert ex.upside == pytest.approx(ex.target_price / t.last_price - 1)
        # signal consistent with composite thresholds (downgrades must be explained by an override)
        if ex.signal is Signal.STRONG_BUY:
            assert ex.composite >= th.strong_buy and ex.confidence >= module_settings.fusion.strong_buy_min_confidence
        elif ex.signal is Signal.BUY:
            assert ex.composite >= th.buy
        elif ex.signal is Signal.SELL:
            assert ex.composite <= th.sell
        else:
            assert th.sell < ex.composite < th.buy or ex.overrides
        if ex.signal in (Signal.HOLD, Signal.SELL):
            assert plan.weight == 0.0 and plan.shares is None
        assert plan.weight <= module_settings.risk.max_weight + 1e-12
        total_w += plan.weight
        # degraded state is never silent
        if d.degraded:
            assert any("저하" in w for w in t.warnings) and any("RSMT" in o for o in ex.overrides)
    assert total_w + rep.cash_weight == pytest.approx(1.0, abs=1e-9)
    pr = rep.portfolio_risk
    assert pr["var95"] < 0 < pr["ex_ante_vol"]
    # normal approximation, exactly like fusion.risk.portfolio_risk (the report labels it "정규근사")
    sigma_h = pr["ex_ante_vol"] * (module_settings.horizon_days / 252) ** 0.5
    assert pr["var95"] == pytest.approx(pr["expected_return"] - 1.6448536269514722 * sigma_h)
    assert pr["cvar95"] < pr["var95"] and 0 < pr["p_loss"] < 1
    assert pr["gross"] == pytest.approx(total_w)


def test_sample_report_is_deterministic(module_settings):
    a = build_sample_report(4, settings=module_settings).model_dump()
    b = build_sample_report(4, settings=module_settings).model_dump()
    assert a == b


@pytest.mark.parametrize("n", [0, 13])
def test_sample_report_rejects_bad_sizes(n, module_settings):
    with pytest.raises(ValueError):
        build_sample_report(n, settings=module_settings)


def test_sample_exercises_every_visible_state(sample12):
    by = {t.instrument.ticker: t for t in sample12.tickers}
    signals = {t.executive.signal for t in sample12.tickers}
    assert signals == set(Signal)
    assert by["247540"].quant.distribution.degraded
    assert any(w.startswith("변동성: EWMA") for w in by["207940"].warnings)  # the GARCH sizing-vol fallback
    assert all(t.quant.vol_forecast_annual and t.quant.vol_forecast_annual > 0 for t in sample12.tickers)
    assert by["035420"].llm.filings is None and by["035420"].llm.errors
    assert by["AAPL"].llm.notices and by["000660"].llm.notices and not by["AAPL"].llm.errors
    f = by["005930"].fundamentals
    assert f.next_earnings_date is not None and f.next_earnings_is_estimate is True
    # the fusion-style earnings trigger is derived from the same date
    assert any(f"실적발표({f.next_earnings_date:%Y-%m-%d}) D-3" in tg for tg in by["005930"].execution.triggers)
    assert not any("₩" in tg for t in sample12.tickers for tg in t.execution.triggers)


def test_sample_price_history_contract():
    px = sample_price_history("005930")
    assert list(px.columns) == ["open", "high", "low", "close", "volume"]
    assert px.index.is_monotonic_increasing and px.index.is_unique and px.index.tz is None
    assert px["close"].iloc[-1] == pytest.approx(286_500)
    assert (px["high"] >= px[["open", "close"]].max(axis=1) - 1e-9).all()
    with pytest.raises(KeyError):
        sample_price_history("XXXX")


# --------------------------------------------------------------------------- charts


def test_price_chart_svg_basic_and_ids_prefixed(sample6):
    t = sample6.tickers[0]
    px = make_prices(n=600, start_price=280_000, seed=3)
    a = price_chart_svg(px, t.execution, t.executive.target_price)
    b = price_chart_svg(px, t.execution, t.executive.target_price)
    for svg in (a, b):
        assert svg.startswith("<svg") and 'width="100%"' in svg and "viewBox" in svg
        assert "<?xml" not in svg and "<!DOCTYPE" not in svg
    ids_a = set(re.findall(r'\sid="([^"]+)"', a))
    ids_b = set(re.findall(r'\sid="([^"]+)"', b))
    assert ids_a and not (ids_a & ids_b), "two charts in one page must never share ids"
    # every internal reference points at a prefixed id of the same chart
    for ref in re.findall(r'(?:href="#|url\(#)([^")]+)', a):
        assert ref in ids_a


def test_price_chart_svg_degrades_gracefully():
    short = make_prices(n=30, start_price=150.0, seed=1)  # no MA50 / MA200 possible
    svg = price_chart_svg(short, None, None)
    assert svg.startswith("<svg")
    tz = make_prices(n=300, start_price=150.0, seed=2)
    tz.index = tz.index.tz_localize("America/New_York")  # yfinance-style tz-aware index
    assert price_chart_svg(tz, None, 170.0).startswith("<svg")
    with pytest.raises(ValueError):
        price_chart_svg(pd.DataFrame(columns=["close"]), None, None)
    with pytest.raises(ValueError):
        price_chart_svg(make_prices(n=1), None, None)
    with pytest.raises(ValueError):
        price_chart_svg(None, None, None)  # type: ignore[arg-type]


def test_price_chart_does_not_touch_global_matplotlib_state():
    import matplotlib

    before = dict(matplotlib.rcParams)
    price_chart_svg(make_prices(n=300, seed=5), None, None)
    after = dict(matplotlib.rcParams)
    assert before == after


def test_fig_to_svg_strips_fixed_size():
    from matplotlib.figure import Figure

    fig = Figure(figsize=(2, 1))
    fig.add_subplot().plot([0, 1], [1, 0])
    svg = fig_to_svg(fig, css_width="50%")
    assert svg.startswith("<svg") and 'width="50%"' in svg and 'height="' not in svg.split(">", 1)[0]


def test_sparkline_svg():
    svg = sparkline_svg([1.0, float("nan"), 2.0, 1.5, 3.0])
    assert svg.startswith("<svg") and "<polyline" in svg and "<circle" in svg
    assert ' id="' not in svg
    assert "<circle" not in sparkline_svg(pd.Series([1.0, 2.0]), end_dot=False)
    with pytest.raises(ValueError):
        sparkline_svg([1.0, float("nan")])


def test_price_chart_skips_invalid_plan_levels(sample6, caplog):
    """Regression: take_profit == 0 raised ZeroDivisionError; NaN levels/entry zone must be skipped, not drawn."""
    plan = sample6.tickers[0].execution.model_copy()
    plan.take_profit = 0.0
    plan.stop_loss = float("nan")
    plan.entry_low = float("nan")
    px = sample_price_history("005930")
    with caplog.at_level(logging.WARNING, logger="quantcio.report.charts"):
        svg = price_chart_svg(px, plan, 300_000.0)
    assert svg.startswith("<svg")
    assert "invalid TP level" in caplog.text and "invalid SL level" in caplog.text
    assert price_chart_svg(px, plan, float("nan")).startswith("<svg")


def test_korean_font_detected_on_this_host():
    # Apple SD Gothic Neo / AppleGothic ship with macOS; elsewhere a warning is logged instead.
    fam = charts_mod.korean_font_family()
    assert fam is None or isinstance(fam, str)


# --------------------------------------------------------------------------- formatting


def test_formatters_render_missing_as_dash():
    for v in (None, float("nan"), np.float32("nan"), float("inf"), "x"):
        assert fmt_pct(v) == MISSING
        assert fmt_prob(v) == MISSING
        assert fmt_num(v) == MISSING
        assert fmt_price(v, "KRW") == MISSING
        assert fmt_money_compact(v, "KRW") == MISSING
    assert fmt_pct(-0.0523) == "−5.2%"
    assert fmt_pct(0.0523) == "+5.2%"
    assert fmt_pct(-0.00001) == "0.0%"  # no "-0.0%"
    assert fmt_prob(0.543) == "54%"
    assert fmt_price(286500, "KRW") == "286,500원"  # no ₩ symbol, no decimals (same as fusion/risk)
    assert fmt_price(286500.6, "KRW") == "286,501원"
    assert fmt_price(337.02, "USD") == "$337.02"
    assert fmt_money_compact(1e9, "KRW") == "10억원"
    assert fmt_money_compact(3.25e8, "KRW") == "3.2억원"
    assert fmt_money_compact(4_500.0, "KRW") == "4,500원"
    assert fmt_money_compact(3.2e6, "USD") == "$3.2M"


def test_price_format_matches_fusion_risk():
    """Trigger/action strings are written by fusion/risk.py; the report tables must use the same format."""
    risk = pytest.importorskip("quantcio.fusion.risk")
    for v, ccy in (
        (286_500, "KRW"),
        (71_200.4, "KRW"),
        (1_124_000, "KRW"),
        (189.5, "USD"),
        (337.02, "USD"),
    ):
        assert fmt_price(v, ccy) == risk.fmt_price(v, ccy)


def test_sample_report_has_no_won_symbol(sample12, module_settings):
    html = render_html(sample12, module_settings, 0, None)
    assert "₩" not in _visible_text(html)
    text = _visible_text(html)
    assert "286,500원" in text and "$337.02" in text and "10억원" in text


@pytest.mark.parametrize(
    "probs",
    [[0.333, 0.333, 0.334], [0.125, 0.66, 0.215], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
)
def test_round_to_100(probs):
    out = render_mod._round_to_100(probs)
    assert sum(out) == (100 if sum(probs) > 0 else 0)
    assert all(isinstance(x, int) and x >= 0 for x in out)


# --------------------------------------------------------------------------- HTML


def test_render_html_contains_all_modules_and_required_text(sample6, module_settings):
    charts = build_sample_charts(sample6)
    html = render_html(sample6, module_settings, 0, charts)
    for heading in MODULE_HEADINGS.values():
        assert heading.replace("&", "&amp;") in html
    assert REPORT_TITLE in html and DISCLAIMER in html and SAMPLE_NOTE in html
    assert module_settings.digest() in html
    assert html.count("<figure>") == len(sample6.tickers)
    text = _visible_text(html)
    assert not re.search(r"\bnan\b|\bNone\b|\binf\b", text), "missing values must render as '—'"
    # degraded state, guardrail override and data warnings are visible on the page
    assert "오프셋 입력 누락(z)" in text and "저하" in text
    assert "503 Service Unavailable" in text
    assert "경고 · 가드레일 · 저하 상태" in text


def test_render_html_missing_parts_are_visible(sample12, module_settings):
    text = _visible_text(render_html(sample12, module_settings, 0, None))
    assert "공시 분석 결과 없음" in text  # NAVER filings missing
    assert "LLM 오류: filings" in text  # and the LLM error is shown
    assert "변동성: EWMA(λ=0.94) 대체 사용" in text  # 207940's volatility fallback warning


def test_compact_levels_trim_content_but_never_warnings(sample12, module_settings):
    charts = {t.instrument.ticker: "<svg viewBox='0 0 1 1'></svg>" for t in sample12.tickers}
    v0 = build_view(sample12, module_settings, 0, charts)
    v1 = build_view(sample12, module_settings, 1, charts)
    v2 = build_view(sample12, module_settings, 2, charts)
    v3 = build_view(sample12, module_settings, 3, charts)
    assert v0["show_charts"] and not v1["show_charts"]
    assert v0["page_break_m2"] and not v1["page_break_m2"]
    assert max(len(r["filings"]["bullets"]) for r in v0["rows"] if r["filings"]) == 3
    assert max(len(r["filings"]["bullets"]) for r in v2["rows"] if r["filings"]) <= 2
    assert max(len(r["risks"]) for r in v2["rows"]) <= 2
    assert any(r["news"]["lines"] for r in v2["rows"] if r["news"])
    assert not any(r["news"]["lines"] for r in v3["rows"] if r["news"])
    for v in (v1, v2, v3):
        assert v["n_alerts"] == v0["n_alerts"] > 0
    assert 'class="chart-grid"' in render_html(sample12, module_settings, 0, charts)
    assert 'class="chart-grid"' not in render_html(sample12, module_settings, 1, charts)


def test_critical_risk_survives_compaction_and_is_flagged(sample6, module_settings):
    rep = sample6.model_copy(deep=True)
    t = rep.tickers[0]
    extra = [
        RiskFactor(
            title=f"리스크{i}",
            detail="d",
            severity="low",
            category="other",
            source_ref="x",
        )
        for i in range(3)
    ]
    crit = RiskFactor(
        title="감사의견 거절",
        detail="d",
        severity="critical",
        category="governance",
        source_ref="감사보고서",
    )
    t.llm.filings = FilingIntelligence(summary=["a", "b", "c"], risk_factors=extra + [crit], tone=0.0)
    view = build_view(rep, module_settings, 3, None)
    row = view["rows"][0]
    assert any(r["title"] == "감사의견 거절" and r["sev"] == "critical" for r in row["risks"])
    assert row["has_critical"] and any(a["kind"] == "critical" for a in row["alerts"])


def test_html_is_autoescaped(sample6, module_settings):
    rep = sample6.model_copy(deep=True)
    rep.tickers[0].executive.thesis = "<script>alert(1)</script> & <b>x</b>"
    html = render_html(rep, module_settings)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; &lt;b&gt;x&lt;/b&gt;" in html


def test_missing_chart_placeholder_and_skipped_llm(sample6, module_settings):
    rep = sample6.model_copy(deep=True)
    for t in rep.tickers:
        t.llm.skipped = True
        t.llm.news = t.llm.filings = t.llm.critic = None
    rep.portfolio_risk = {}
    text = _visible_text(render_html(rep, module_settings, 0, {rep.tickers[0].instrument.ticker: "<svg></svg>"}))
    assert "차트 없음" in text
    flat = re.sub(r"\s+", " ", text)
    assert "LLM 분석 생략" in flat and "LLM 생략(--no-llm)" in flat  # header chip: 'LLM <b>생략(--no-llm)</b>'
    assert "포트폴리오 리스크 미산출" in text


def test_missing_distribution_and_allocation_gap_are_visible(sample6, module_settings):
    rep = sample6.model_copy(deep=True)
    rep.tickers[1].quant.distribution = None
    rep.tickers[1].execution.barrier = None
    rep.cash_weight = 0.9  # weights + cash != 1
    text = _visible_text(render_html(rep, module_settings))
    assert "분포 없음" in text and "RSMT-Q 분포 예측 없음" in text
    assert "경로 없음" in text
    assert "합이 100%가 아닙니다" in text


def test_render_html_rejects_empty_report(sample6, module_settings):
    rep = sample6.model_copy(deep=True)
    rep.tickers = []
    with pytest.raises(ValueError):
        render_html(rep, module_settings)


def test_diagnostics_lines_capped_but_alert_lines_kept(sample6, module_settings):
    rep = sample6.model_copy(deep=True)
    rep.model_diagnostics["KR"] = [f"line {i}" for i in range(20)] + ["⚠ 붕괴 감지: 고유 (μ,σ) 비율 0.25 — 저하"]
    diag = {d["market"]: d for d in build_view(rep, module_settings, 0, None)["diagnostics"]}["KR"]
    texts = [ln["text"] for ln in diag["lines"]]
    assert texts[-1].startswith("⚠ 붕괴") and diag["lines"][-1]["alert"]
    assert diag["hidden"] == 20 - 7
    ok = {d["market"]: d for d in build_view(sample6, module_settings, 0, None)["diagnostics"]}
    passed = [ln for ln in ok["US"]["lines"] if "붕괴 검사 통과" in ln["text"]]
    assert passed and not passed[0]["alert"]  # "collapse check passed" is not an alarm


# quant/rsmt.py's real layout: describe() tree FIRST, then the summary line, offsets, CV, dropped/failed rows,
# followed by health() lines ("[정상]/[저하]/[정보] ...") - the sample puts its summary first, reality does not
REAL_RSMT_KR = [
    "vol_ratio_20_120 ≤ 1.12 (p=3.1e-04, n=5210)",
    "├─ mkt_ret_60 ≤ 0.031 (p=2.2e-03, n=3405)",
    "│  ├─ rsi14 ≤ 61.2 (p=9.1e-03, n=1812)",
    "│  │  ├─ 리프#1 n=903 μe +0.1% σe 7.0% ν 6.0",
    "│  │  └─ 리프#2 n=909 μe +0.4% σe 7.3% ν 6.4",
    "│  └─ 리프#3 n=1593 μe +0.9% σe 6.4% ν 8.0",
    "└─ dd_250 ≤ −0.18 (p=8.7e-03, n=1805)",
    "   ├─ 리프#4 n=812 μe −1.1% σe 9.8% ν 5.1",
    "   └─ 리프#5 n=993 μe +0.4% σe 8.3% ν 7.4",
    "RSMT-Q KR v1 · h=20일 · n=5210 · 리프 5 · root p=3.1e-04",
    "오프셋 γ=+0.0071(모멘텀 z) · η=0.930(log σ) · δ=+0.060 · δ_rel=+0.030",
    "CV(OOS 1200행, 4폴드) AUC 0.547 vs 기준 0.500 · logloss 0.6871 vs 0.6927 · Brier 0.2468 vs 0.2497",
    "CRPS 0.0412 vs 0.0421 · cover80 0.79 vs 0.80 · IC +0.071 → 스킬 통과",
    "제외 행 312/5522 (nan_feature 300, nan_offset 12)",
    "로드 실패 0/61",
    "[정상] 오프셋 적용 4/4행(서빙) (γ=+0.0071, η=0.930)",
    "[정보] 오프셋 수렴 미완 (50회)",
    "[정보] CV 스킬 게이트 미달 — ML 필러 가중 축소",
]


@pytest.mark.parametrize("level,n_tree", [(0, 6), (2, 4)])
def test_diagnostics_real_rsmt_layout_keeps_summary_and_health(level, n_tree, sample6, module_settings):
    """Regression: with the tree printed first, the n/leaves/root-p summary and the [정보] health lines were cut."""
    rep = sample6.model_copy(deep=True)
    rep.model_diagnostics["KR"] = list(REAL_RSMT_KR)
    kr = {d["market"]: d for d in build_view(rep, module_settings, level, None)["diagnostics"]}["KR"]
    texts = [ln["text"] for ln in kr["lines"]]
    assert texts[0].startswith("RSMT-Q KR v1") and "리프 5" in texts[0] and "root p=" in texts[0]
    expected_tree = [render_mod._sci_minus(ln) for ln in REAL_RSMT_KR[:n_tree]]
    assert texts[1 : 1 + n_tree] == expected_tree  # the first tree lines, in order
    for must in (
        "[정보] 오프셋 수렴 미완 (50회)",
        "[정보] CV 스킬 게이트 미달 — ML 필러 가중 축소",
        "제외 행 312/5522 (nan_feature 300, nan_offset 12)",
        "로드 실패 0/61",
    ):
        assert must in texts
    assert not any(ln.startswith("[정상]") for ln in texts)
    assert kr["hidden"] == len(REAL_RSMT_KR) - len(texts)
    assert not any(ln["alert"] for ln in kr["lines"])  # nothing here is an alarm
    html = render_html(rep, module_settings, level)
    # exponent minus is typographic (U+2212): Chrome broke lines after "e-" ("p=6.2e-" / "01")
    assert "RSMT-Q KR v1 · h=20일 · n=5210 · 리프 5 · root p=3.1e−04" in html
    assert not re.search(r"\de-\d", _visible_text(html))


def test_degraded_one_letter_ticker_does_not_hide_market_alarm(sample6, module_settings):
    """Regression: substring matching let a degraded ticker 'T' drop every alarm containing a 'T' (e.g. RSMT-Q)."""
    rep = sample6.model_copy(deep=True)
    t = rep.tickers[1]
    t.instrument.ticker = "T"
    t.quant.distribution.degraded = True
    t.quant.distribution.degraded_reasons = ["오프셋 입력 누락(z)"]
    collapse = "[저하] 예측 붕괴(RSMT-Q 서빙 종목): 고유 (μ,σ) 1/4 = 0.25 < 0.50 — 오프셋/분할 미작동 의심"
    per_ticker = "[저하] 오프셋 미적용 1/3행(서빙): T"
    rep.model_diagnostics["US"] = list(rep.model_diagnostics["US"]) + [
        collapse,
        per_ticker,
    ]
    view = build_view(rep, module_settings, 0, None)
    us = [a["text"] for g in view["alert_groups"] if g["ticker"] == "US" for a in g["entries"]]
    assert collapse in us  # market-level alarm reaches page 1
    assert per_ticker not in us  # the per-ticker line is still de-duplicated (row says it)
    assert any("오프셋 입력 누락(z)" in a["text"] for a in view["rows"][1]["alerts"])


def test_every_degraded_reason_reaches_page1(sample6, module_settings):
    """Regression: an upstream warning naming only one reason used to hide the other reasons."""
    rep = sample6.model_copy(deep=True)
    t = rep.tickers[0]
    t.quant.distribution.degraded = True
    t.quant.distribution.degraded_reasons = [
        "오프셋 입력 누락(z)",
        "예측 붕괴(서빙 종목): 고유 (μ,σ) 1/4 = 0.25",
    ]
    t.warnings = ["RSMT-Q 저하: 오프셋 입력 누락(z) — 신뢰도 −10"]
    t.executive.overrides = []
    alerts = " | ".join(a["text"] for a in build_view(rep, module_settings, 0, None)["rows"][0]["alerts"])
    assert "오프셋 입력 누락(z)" in alerts and "예측 붕괴(서빙 종목)" in alerts
    assert alerts.count("오프셋 입력 누락(z)") == 1  # the already-reported reason is not repeated


def test_negative_cash_is_flagged_with_typographic_minus(sample6, module_settings):
    rep = sample6.model_copy(deep=True)
    for t in rep.tickers:
        t.execution.weight = 0.175
    rep.cash_weight = 1.0 - 0.175 * len(rep.tickers)  # -0.05: over-invested, weights + cash == 1
    view = build_view(rep, module_settings, 0, None)
    assert view["portfolio"]["cash_txt"] == "−5.0%"
    texts = [a["text"] for g in view["alert_groups"] for a in g["entries"]]
    assert any("현금 비중 −5.0%" in x and "레버리지" in x for x in texts)
    assert fmt_prob(-0.25) == "−25%"


def test_portfolio_risk_label_does_not_claim_rsmt_paths(sample6, module_settings):
    """fusion/risk.py computes portfolio VaR/P(loss) as a normal approximation - the heading must not say RSMT-Q."""
    html = render_html(sample6, module_settings)
    assert "사전 리스크(20거래일) · LLM 비용" in html
    assert "사전 리스크(20거래일, RSMT-Q 경로)" not in html


def test_module4_shows_barrier_horizon(sample6, module_settings):
    """The TP/SL/time-exit probabilities are simulated over the holding horizon (60D), not the 20D forecast."""
    H = module_settings.fusion.target_horizon_days
    view = build_view(sample6, module_settings, 0, None)
    assert view["barrier_h_txt"] == f"{H}거래일"
    assert all(r["barrier"]["h_note"] == "" for r in view["rows"])  # common horizon -> header only
    html = render_html(sample6, module_settings)
    assert "P(익절 선도달)</span>" in html and f"· {H}거래일</span>" in html
    assert f"보유 {H}거래일" in html
    assert "청산 확률 (RSMT-Q 경로, 20거래일)" not in html  # the old, wrong horizon label


def test_barrier_horizon_missing_or_mixed_is_visible(sample6, module_settings):
    rep = sample6.model_copy(deep=True)
    rep.tickers[0].execution.barrier.horizon_days = 0  # legacy producer: not recorded
    rep.tickers[1].execution.barrier.horizon_days = 20
    view = build_view(rep, module_settings, 0, None)
    assert view["barrier_h_txt"] == "종목별 기간 상이"
    notes = [r["barrier"]["h_note"] for r in view["rows"]]
    assert notes[0] == "⚠ 기간 미기재" and notes[1] == "20거래일" and notes[2] == "60거래일"
    texts = [a["text"] for g in view["alert_groups"] for a in g["entries"]]
    assert any("시뮬레이션 기간 미기재 1종목" in x and rep.tickers[0].instrument.ticker in x for x in texts)
    for t in rep.tickers:
        t.execution.barrier.horizon_days = 0
    view = build_view(rep, module_settings, 0, None)
    assert view["barrier_h_txt"] == "기간 미기재"
    assert "기간 미기재" in _visible_text(render_html(rep, module_settings))


def test_llm_notices_amber_and_errors_red(sample12, module_settings):
    by = {t.instrument.ticker: t for t in sample12.tickers}
    view = build_view(sample12, module_settings, 0, None)
    rows = {r["ticker"]: r for r in view["rows"]}
    notice = by["AAPL"].llm.notices[0]
    assert rows["AAPL"]["llm_notices"] == [notice]
    kinds = {a["kind"] for a in rows["AAPL"]["alerts"]}
    assert "notice" in kinds and "llm" not in kinds  # a notice is not an error
    assert any(a["kind"] == "llm" and "LLM 오류" in a["text"] for a in rows["035420"]["alerts"])
    counts = {c["kind"]: c for c in view["alert_counts"]}
    assert counts["notice"]["label"] == "LLM 알림" and counts["llm"]["label"] == "LLM 오류"
    html = render_html(sample12, module_settings)
    assert f'<div class="chip warn notice">ⓘ LLM 알림: {notice}</div>' in html
    assert '<div class="chip crit">⚠ LLM 오류: filings' in html
    rules = _css_rules((render_mod.TEMPLATE_DIR / "report.css").read_text(encoding="utf-8"))
    assert "crit" in rules[".chip.k-critical"] and "crit" in rules[".chip.k-llm"]  # errors red
    assert "warn" in rules[".chip.k-model"] and "warn" in rules[".chip.k-notice"]  # notices amber


def _css_rules(css: str) -> dict[str, str]:
    """selector -> concatenated declarations of every top-level rule naming it (formatting-independent:
    comments, line breaks and spacing inside selector lists / bodies do not matter)."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out: dict[str, str] = {}
    for sel, body in re.findall(r"([^{}@]+)\{([^{}]*)\}", css):
        for s in sel.split(","):
            key = " ".join(s.split())
            if key:
                out[key] = out.get(key, "") + " ".join(body.split())
    return out


def test_css_rule_parser_ignores_formatting():
    compact = ".a, .b { color: var(--crit-ink); }"
    spread = "/* c */\n.a,\n.b\n{\n   color:  var(--crit-ink);\n}\n"
    assert _css_rules(compact)[".b"] == _css_rules(spread)[".b"] == "color: var(--crit-ink);"


def test_multiple_notices_are_summarised_on_page1_and_listed_in_full(sample6, module_settings):
    rep = sample6.model_copy(deep=True)
    rep.tickers[0].llm.notices = [
        "news: 폴백 모델 응답",
        "filings: 수치 미확인 하이라이트 1건 제외",
        "  ",
    ]
    rep.tickers[0].llm.skipped = True
    view = build_view(rep, module_settings, 0, None)
    row = view["rows"][0]
    assert row["llm_notices"] == [
        "news: 폴백 모델 응답",
        "filings: 수치 미확인 하이라이트 1건 제외",
    ]
    page1 = [a for a in row["alerts"] if a["kind"] == "notice"]
    assert any("LLM 알림 2건: news: 폴백 모델 응답 외 1건" in a["text"] for a in page1)
    assert any("LLM 분석 생략" in a["text"] for a in page1)  # skipped = amber, not red


@pytest.mark.parametrize(
    "days,estimate,expect",
    [
        (0, False, "실적 D-day"),
        (2, None, "실적 D-2"),
        (13, True, "실적 D-13"),
        (14, False, "실적 D-14"),
        (15, False, None),
        (-1, False, None),
    ],
)
def test_earnings_chip_window(days, estimate, expect, sample6, module_settings):
    from datetime import timedelta

    rep = sample6.model_copy(deep=True)
    t = rep.tickers[1]  # AAPL (US, as_of 2026-09-23)
    gen_local = rep.generated_at.astimezone(render_mod._MARKET_TZ["US"]).date()
    ref = max(t.as_of, gen_local)
    t.fundamentals.next_earnings_date = ref + timedelta(days=days)
    t.fundamentals.next_earnings_is_estimate = estimate
    ern = build_view(rep, module_settings, 0, None)["rows"][1]["earnings"]
    if expect is None:
        assert ern is None
        return
    assert ern["txt"] == expect and ern["hot"] == (days <= render_mod.EARNINGS_HOT_DAYS)
    assert ern["date"].endswith("추정") == bool(estimate)
    html = render_html(rep, module_settings)
    assert html.count(expect) >= 2  # Module 1 thesis row + Module 4 card


def test_sample_earnings_chip_uses_generation_date(sample6, module_settings):
    """005930: 3Q 잠정실적 2026-10-07 (estimated); the report is generated 2026-09-24 KST -> D-13."""
    row = build_view(sample6, module_settings, 0, None)["rows"][0]
    assert row["earnings"] == {
        "txt": "실적 D-13",
        "date": "10/7 추정",
        "days": 13,
        "hot": False,
        "estimate": True,
    }
    assert all(r["earnings"] is None for r in build_view(sample6, module_settings, 0, None)["rows"][1:])


def test_portfolio_risk_labels_come_from_fusion_and_say_normal_approx(sample6, module_settings):
    risk = pytest.importorskip("quantcio.fusion.risk")
    render_mod.portfolio_risk_labels.cache_clear()
    labels = render_mod.portfolio_risk_labels()
    for k, v in risk.PORTFOLIO_RISK_LABELS.items():
        assert labels[k] == v
    for k in ("var95", "cvar95", "p_loss"):
        assert "근사" in labels[k]
    view = build_view(sample6, module_settings, 0, None)
    tiles = {t["label"]: t for t in view["portfolio"]["tiles"]}
    assert "정규근사" in tiles["VaR95 (20D)"]["sub"] and "정규근사" in tiles["P(손실, 20D)"]["sub"]
    line = " | ".join(view["portfolio"]["risk_line"])
    # "기간" in the upstream labels is made concrete, and the shared parenthetical is printed once
    assert "기간" not in line.replace("보유기간", "")
    assert line.count("(20거래일, 정규분포 근사)") == 1 and "VaR95 −" in line and "CVaR95 −" in line


def test_portfolio_risk_labels_fallback_when_fusion_unavailable(monkeypatch, caplog, sample6, module_settings):
    import sys

    monkeypatch.setitem(sys.modules, "quantcio.fusion.risk", None)  # force ImportError
    render_mod.portfolio_risk_labels.cache_clear()
    try:
        with caplog.at_level(logging.WARNING, logger="quantcio.report.render"):
            labels = render_mod.portfolio_risk_labels()
        assert labels == render_mod.LOCAL_PORTFOLIO_RISK_LABELS
        assert "기본 라벨" in caplog.text  # the fallback is logged, not silent
        line = " | ".join(build_view(sample6, module_settings, 0, None)["portfolio"]["risk_line"])
        assert "(20거래일, 정규근사)" in line
    finally:
        render_mod.portfolio_risk_labels.cache_clear()


def test_portfolio_risk_legacy_keys_still_render(sample6, module_settings):
    rep = sample6.model_copy(deep=True)
    rep.portfolio_risk = {
        "vol": 0.2,
        "var_95": -0.07,
        "prob_loss": 0.41,
        "tracking": 0.031,
    }
    view = build_view(rep, module_settings, 0, None)
    tiles = {t["label"]: t for t in view["portfolio"]["tiles"]}
    assert tiles["사전 변동성(연)"]["value"] == "20.0%" and tiles["사전 변동성(연)"]["warn"]
    assert tiles["VaR95 (20D)"]["value"] == "−7.0%" and tiles["P(손실, 20D)"]["value"] == "41%"
    assert any("tracking 0.031" in x for x in view["portfolio"]["risk_line"])  # unknown keys are still shown


def test_footer_template_escapes(sample6):
    rep = sample6.model_copy(deep=True)
    rep.config_digest = "<b>x</b>"
    f = footer_template(rep)
    assert "&lt;b&gt;x&lt;/b&gt;" in f and 'class="pageNumber"' in f and 'class="totalPages"' in f


# --------------------------------------------------------------------------- PDF helpers / fallback


def test_count_pdf_pages_regex_fallback(monkeypatch, caplog):
    import sys

    monkeypatch.setitem(sys.modules, "pypdf", None)  # force ImportError
    data = b"%PDF-1.4 << /Type /Pages /Kids [] >> << /Type /Page >> << /Type/Page >> %%EOF"
    with caplog.at_level(logging.WARNING, logger="quantcio.report.render"):
        assert count_pdf_pages(data) == 2
    assert "regex fallback" in caplog.text


def _force_no_browser(monkeypatch):
    def boom(_p):
        raise NoBrowserError("forced for test: no browser")

    monkeypatch.setattr(render_mod, "_launch", boom)


def test_render_pdf_html_fallback_is_observable(monkeypatch, tmp_path, caplog, sample6, module_settings):
    _force_no_browser(monkeypatch)
    html = render_html(sample6, module_settings)
    out = tmp_path / "r.pdf"
    with caplog.at_level(logging.WARNING, logger="quantcio.report.render"):
        pages = render_pdf(html, out)
    assert pages == 0
    assert not out.exists() and out.with_suffix(".html").read_text(encoding="utf-8") == html
    assert any(r.levelno == logging.WARNING and "HTML" in r.getMessage() for r in caplog.records)
    with pytest.raises(NoBrowserError):
        render_pdf(html, tmp_path / "strict.pdf", html_fallback=False)


def test_fit_to_pages_html_fallback_result(monkeypatch, tmp_path, caplog, sample6, module_settings):
    _force_no_browser(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="quantcio.report.render"):
        res = fit_to_pages(sample6, module_settings, tmp_path / "r.pdf")
    assert res.fallback_html and res.pages == 0 and not res.fits and res.pdf_path is None
    assert res.html_path is not None and res.html_path.exists()
    assert res.warnings and "HTML" in res.warnings[0]
    assert caplog.records and caplog.records[-1].levelno == logging.WARNING


class _FakeRenderer:
    """Offline PdfRenderer stand-in: 'renders' (level, scale) to bytes whose page count comes from a table."""

    active = True

    def __init__(self, pages_by_attempt: dict[tuple[int, float], int]) -> None:
        self.pages = pages_by_attempt

    def pdf_bytes(self, html: str, *, scale: float = 1.0, **_: object) -> bytes:
        level = int(re.search(r'<body class="c(\d)"', html).group(1))
        return f"%PDF-fake {level} {scale} {self.pages[(level, scale)]}".encode()


def test_fit_to_pages_keeps_fewest_pages_when_nothing_fits(monkeypatch, tmp_path, sample6, module_settings, caplog):
    """Regression: when nothing fit, the LAST attempt was written even if an earlier one had fewer pages."""
    monkeypatch.setattr(render_mod, "count_pdf_pages", lambda data: int(bytes(data).split()[-1]))
    pages = {
        (0, 1.0): 6,
        (0, 0.9): 5,
        (1, 1.0): 4,
        (1, 0.9): 4,
        (2, 1.0): 5,
        (2, 0.9): 5,
        (3, 1.0): 5,
        (3, 0.9): 5,
        (3, 0.8): 5,
    }
    with caplog.at_level(logging.WARNING, logger="quantcio.report.render"):
        res = fit_to_pages(
            sample6,
            module_settings,
            tmp_path / "f.pdf",
            renderer=_FakeRenderer(pages),
            scales=(1.0, 0.9),
            last_resort_scales=(0.8,),
        )
    assert len(res.attempts) == len(pages) and not res.fits
    assert (res.compact_level, res.scale, res.pages) == (
        1,
        1.0,
        4,
    )  # fewest pages, earliest on the tie
    assert res.pdf_path.read_bytes() == b"%PDF-fake 1 1.0 4"
    written = res.html_path.read_text(encoding="utf-8")
    assert 'class="c1"' in written  # HTML matches the written PDF
    # the document itself says it overflowed (not only FitResult.warnings)
    assert "종목 수가 많아 3페이지 제한 초과" in written and "4페이지" in written
    assert 'class="chip over"' in written
    assert res.warnings and "한도 초과" in res.warnings[0] and "최소 4쪽" in res.warnings[0]
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    # the first fitting attempt still wins immediately, without any overflow note
    ok = fit_to_pages(
        sample6,
        module_settings,
        tmp_path / "g.pdf",
        renderer=_FakeRenderer({**pages, (0, 0.9): 3}),
        scales=(1.0, 0.9),
        last_resort_scales=(0.8,),
    )
    assert ok.fits and (ok.compact_level, ok.scale, ok.pages) == (0, 0.9, 3) and len(ok.attempts) == 2
    assert "제한 초과" not in ok.html_path.read_text(encoding="utf-8")


class _GrowingRenderer(_FakeRenderer):
    """The overflow note itself pushes the document onto one more page."""

    def pdf_bytes(self, html: str, *, scale: float = 1.0, **_: object) -> bytes:
        level = int(re.search(r'<body class="c(\d)"', html).group(1))
        n = self.pages[(level, scale)] + (1 if "제한 초과" in html else 0)
        return f"%PDF-fake {level} {scale} {n}".encode()


def test_overflow_note_states_the_final_page_count(monkeypatch, tmp_path, sample6, module_settings):
    monkeypatch.setattr(render_mod, "count_pdf_pages", lambda data: int(bytes(data).split()[-1]))
    pages = {(lv, 1.0): 4 for lv in range(4)}
    res = fit_to_pages(
        sample6,
        module_settings,
        tmp_path / "grow.pdf",
        renderer=_GrowingRenderer(pages),
        scales=(1.0,),
        last_resort_scales=(),
    )
    assert not res.fits and res.pages == 5 and res.pdf_path.read_bytes().endswith(b" 5")
    written = res.html_path.read_text(encoding="utf-8")
    assert "5페이지" in written and "에서도 4페이지" not in written
    assert "최소 5쪽" in res.warnings[0]


def test_overflow_note_is_rendered_first_on_page1(sample6, module_settings):
    html = render_html(
        sample6,
        module_settings,
        2,
        None,
        overflow_note="종목 수가 많아 3페이지 제한 초과 — 테스트",
    )
    v = build_view(sample6, module_settings, 2, None, overflow_note="X 제한 초과")
    assert v["notes"][0] == {"text": "X 제한 초과", "cls": "over"}
    assert html.index("제한 초과 — 테스트") < html.index('id="m1"')


def test_fit_to_pages_validates_arguments(sample6, module_settings, tmp_path):
    with pytest.raises(ValueError):
        fit_to_pages(sample6, module_settings, tmp_path / "x.pdf", scales=())
    with pytest.raises(ValueError):
        fit_to_pages(sample6, module_settings, tmp_path / "x.pdf", max_pages=0)


# --------------------------------------------------------------------------- PDF with a real browser


@pytest.mark.parametrize(("n", "with_models"), [(4, True), (6, True), (8, True), (6, False)])
def test_pdf_fits_three_pages_with_clean_korean_text(n, with_models, pdf_renderer, tmp_path, module_settings):
    rep = build_sample_report(n, settings=module_settings, with_models=with_models)
    res = fit_to_pages(
        rep,
        module_settings,
        tmp_path / f"s{n}.pdf",
        charts=build_sample_charts(rep),
        renderer=pdf_renderer,
    )
    assert not res.fallback_html and res.fits
    assert 1 <= res.pages <= module_settings.report.max_pages
    assert res.pdf_path.exists() and res.html_path.exists()
    assert count_pdf_pages(res.pdf_path) == res.pages
    pages = _pdf_text(res.pdf_path)
    full = "".join(pages)
    assert full.count("�") == 0
    squashed = re.sub(r"\s+", "", full)
    for heading in MODULE_HEADINGS.values():
        assert re.sub(r"\s+", "", heading) in squashed, heading
    assert re.sub(r"\s+", "", REPORT_TITLE) in squashed
    assert re.sub(r"\s+", "", DISCLAIMER) in squashed
    for t in rep.tickers:
        assert re.sub(r"\s+", "", t.instrument.name) in squashed
    # page 1 is the decision page: title + Module 1; Module 4 is on the last page
    assert "ExecutiveSignalMatrix" in re.sub(r"\s+", "", pages[0])
    assert "Execution&RiskManagement" in re.sub(r"\s+", "", pages[-1])
    # we prefer full content: small universes must not need compaction. With the macro strip, the factor-model
    # table and the backtest summary six names still keep every block (charts included) at a slightly smaller
    # print scale; without those blocks they fit at full scale.
    if n <= 6:
        assert res.compact_level == 0
        assert res.scale == 1.0 if not with_models or n <= 4 else res.scale >= 0.9


def test_fit_to_pages_reports_overflow(pdf_renderer, tmp_path, module_settings, caplog):
    rep = build_sample_report(8, settings=module_settings)
    with caplog.at_level(logging.WARNING, logger="quantcio.report.render"):
        res = fit_to_pages(
            rep,
            module_settings,
            tmp_path / "tight.pdf",
            max_pages=1,
            renderer=pdf_renderer,
            scales=(1.0,),
            last_resort_scales=(),
        )
    assert not res.fits and res.pages > 1 and res.pdf_path.exists()
    assert len(res.attempts) == 4 and [a[0] for a in res.attempts] == [0, 1, 2, 3]
    fewest = min(p for *_, p in res.attempts)
    # the written PDF is the fewest-pages attempt re-rendered with the overflow note (which may add a page)
    assert res.pages == count_pdf_pages(res.pdf_path) and fewest <= res.pages <= fewest + 1
    assert (res.compact_level, res.scale) == next((lv, s) for lv, s, p in res.attempts if p == fewest)
    assert res.warnings and "한도" in res.warnings[0]
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    # the PDF itself carries the overflow note on page 1, with the page count of that very PDF
    page1 = re.sub(r"\s+", "", _pdf_text(res.pdf_path)[0])
    assert "종목수가많아1페이지제한초과" in page1 and f"{res.pages}페이지" in page1


def test_large_universes_are_printed_in_full_on_more_pages(pdf_renderer, tmp_path, module_settings):
    """More than report.compact_max_names names: no compaction (charts, news, risk detail kept), as many pages as
    needed up to max_pages_extended, and page 1 says so. With auto_extend off the old compaction path is used."""
    s = module_settings.model_copy(deep=True)
    s.report.compact_max_names = 6
    rep = build_sample_report(12, settings=s)
    res = fit_to_pages(rep, s, tmp_path / "wide.pdf", charts=build_sample_charts(rep), renderer=pdf_renderer)
    assert res.extended and res.fits and res.compact_level == 0 and res.pages > s.report.max_pages
    assert res.pages == count_pdf_pages(res.pdf_path) and res.pages <= s.report.max_pages_extended
    assert all(level == 0 for level, *_ in res.attempts)  # never compacted
    page1 = re.sub(r"\s+", "", _pdf_text(res.pdf_path)[0])
    assert "내용생략없이" in page1 and f"{res.pages}쪽으로출력" in page1
    assert any("[정보]" in w for w in res.warnings)
    s.report.auto_extend = False
    old = fit_to_pages(rep, s, tmp_path / "squeezed.pdf", charts=build_sample_charts(rep), renderer=pdf_renderer)
    assert not old.extended and (old.compact_level > 0 or old.scale < 1.0)


def test_render_pdf_one_shot_and_inside_event_loop(pdf_renderer, tmp_path, sample6, module_settings):
    html = render_html(sample6, module_settings, 0, build_sample_charts(sample6))
    n_reused = render_pdf(html, tmp_path / "reused.pdf", renderer=pdf_renderer)
    # uncompacted level-0 HTML at scale 1.0: the page count depends on the platform's Korean font metrics (6 names:
    # 4 pages with Apple SD Gothic Neo on macOS, 5 with NanumGothic on the Ubuntu CI runner). This test is about
    # renderer reuse vs the event-loop offload giving the SAME document; fit_to_pages enforces the page limit and
    # is tested on its own. The bound only catches a runaway layout.
    assert 1 <= n_reused <= 8
    assert count_pdf_pages(tmp_path / "reused.pdf") == n_reused

    async def inside_loop() -> int:
        return render_pdf(html, tmp_path / "loop.pdf")  # auto-offloaded to a worker thread

    # run the coroutine on a fresh thread: the module's sync-Playwright renderer owns this thread's loop
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(asyncio.run, inside_loop()).result() == n_reused
    assert (tmp_path / "loop.pdf").read_bytes()[:5] == b"%PDF-"


def test_long_names_do_not_distort_grid_layout(pdf_renderer, sample6, module_settings):
    """Regression: a long company name widened its chart-grid column (1fr min-content) and, through the SVG
    aspect ratio, made that chart much taller; card headers must not clip the signal badge either."""
    rep = sample6.model_copy(deep=True)
    rep.tickers[1].instrument.name = "Berkshire Hathaway Inc. Class B Common Stock"
    rep.tickers[2].instrument.name = "삼성바이오로직스우선주신형전환상장예정"
    html = render_html(rep, module_settings, 0, build_sample_charts(rep))
    page = pdf_renderer._browser.new_page(viewport={"width": 703, "height": 1050})  # A4 content box (186 mm)
    try:
        page.emulate_media(media="print")
        page.set_content(html, wait_until="load")
        figs = page.eval_on_selector_all(
            ".chart-grid figure",
            "els => els.map(e => { const r = e.getBoundingClientRect(); const b = e.querySelector('.badge')"
            ".getBoundingClientRect(); return [r.width, r.height, r.right, b.right]; })",
        )
        cards = page.eval_on_selector_all(
            ".cards .card",
            "els => els.map(e => { const r = e.getBoundingClientRect(); const b = e.querySelector('.card-h .badge')"
            ".getBoundingClientRect(); return [r.width, r.right, b.width, b.right]; })",
        )
    finally:
        page.close()
    widths, heights = [f[0] for f in figs], [f[1] for f in figs]
    assert max(widths) - min(widths) < 1.0, widths
    assert max(heights) - min(heights) < 0.12 * min(heights), heights  # charts share one aspect ratio
    assert all(badge_r <= fig_r + 0.5 for _, _, fig_r, badge_r in figs)  # caption badge inside its figure
    assert max(c[0] for c in cards) - min(c[0] for c in cards) < 1.0
    assert all(bw > 5 and br <= cr + 0.5 for _, cr, bw, br in cards)  # card badge visible, not clipped


_OVERFLOW_JS = """() => { const out = [];
  for (const e of document.querySelectorAll('body *')) {
    const c = e.parentElement &&
      e.parentElement.closest('.card, td, th, figure, .tile, .diag, .trig, .alerts, .risk-line');
    if (!c || getComputedStyle(c).overflow !== 'visible') continue;
    const r = e.getBoundingClientRect(), cr = c.getBoundingClientRect();
    if (r.width && r.right > cr.right + 1)
      out.push(e.tagName + '.' + e.className + ' +' + Math.round(r.right - cr.right));
  } return [document.documentElement.scrollWidth, out]; }"""


def test_long_llm_strings_never_overflow_page_or_card(pdf_renderer, sample6, module_settings):
    """Regression: a long/spaceless LLM risk title (nowrap) overlapped the next card or widened the document,
    which makes Chrome silently shrink the entire PDF. Every nowrap slot gets a long string here."""
    long_ko = "미국 상무부의 대중국 첨단 반도체 장비 및 고대역폭 메모리 수출 통제 확대에 따른 중국 매출 급감 가능성"
    long_en = (
        "MaterialAdverseEffect_https://www.sec.gov/Archives/edgar/data/320193/000032019326000070/aapl-20260627.htm"
    )
    rep = sample6.model_copy(deep=True)
    t0, t1 = rep.tickers[0], rep.tickers[1]
    t0.llm.filings.risk_factors[0].title = long_ko
    t1.llm.filings.risk_factors[0].title = long_en
    t0.instrument.name, t0.instrument.exchange = long_en, long_en
    t0.executive.thesis = long_en
    t0.executive.overrides = [long_en]
    t0.warnings = [long_en]
    t0.llm.news.key_positives = [long_en]
    t0.llm.errors = [long_en]
    t0.llm.notices = [long_en, long_ko]
    t0.fundamentals.next_earnings_date = t0.as_of  # '실적 D-day' chip in the name/trigger rows
    t0.llm.critic.contradictions = [long_en]
    t0.execution.triggers = [long_en]
    t0.execution.action = long_en
    t0.quant.distribution.leaf_path = [long_en]
    rep.market_regime["KOSPI"] = long_ko
    rep.model_diagnostics["KR"] = [long_en * 2]
    for level in (0, 3):
        html = render_html(rep, module_settings, level, build_sample_charts(rep))
        page = pdf_renderer._browser.new_page(viewport={"width": 703, "height": 1050})  # A4 content box
        try:
            page.emulate_media(media="print")
            page.set_content(html, wait_until="load")
            scroll_w, spilled = page.evaluate(_OVERFLOW_JS)
        finally:
            page.close()
        assert scroll_w <= 703, f"level {level}: document wider than the page ({scroll_w}px)"
        assert not spilled, f"level {level}: {spilled}"


def test_renderer_recovers_after_forced_launch_failure(monkeypatch, pdf_renderer, tmp_path, sample6, module_settings):
    html = render_html(sample6, module_settings, 1)
    real_launch = render_mod._launch
    _force_no_browser(monkeypatch)
    assert render_pdf(html, tmp_path / "a.pdf") == 0
    monkeypatch.setattr(render_mod, "_launch", real_launch)
    assert render_pdf(html, tmp_path / "b.pdf") >= 1  # the driver was stopped cleanly


_DRIVER_KILL_SCRIPT = """
import os, signal, time
from playwright.sync_api import Page
from quantcio.report.render import PdfRenderer, count_pdf_pages

groups = set()

def kill_driver(r):
    assert r._browser_groups, "browser process group not recorded"
    groups.update(r._browser_groups)
    os.kill(r._pw._impl_obj._connection._transport._proc.pid, signal.SIGKILL)

def alive(g):
    try:
        os.killpg(g, 0)
        return True
    except ProcessLookupError:
        return False

with PdfRenderer() as r:
    print("first", count_pdf_pages(r.pdf_bytes("<p>a</p>")), flush=True)
    kill_driver(r)  # between renders
    print("between", count_pdf_pages(r.pdf_bytes("<p>b</p>")), flush=True)
    real_pdf = Page.pdf

    def pdf_after_kill(self, **kw):  # the driver dies while page.pdf is in flight
        Page.pdf = real_pdf
        kill_driver(r)
        return real_pdf(self, **kw)

    Page.pdf = pdf_after_kill
    print("during", count_pdf_pages(r.pdf_bytes("<p>c</p>")), flush=True)
print("closed", flush=True)
deadline = time.monotonic() + 10
while any(alive(g) for g in groups) and time.monotonic() < deadline:
    time.sleep(0.2)
print("orphaned browsers", sum(alive(g) for g in groups), flush=True)
"""


def test_renderer_survives_a_dead_playwright_driver(pdf_renderer, tmp_path):
    """Regression (local full-suite hang, 2026-10-06): when the Playwright driver process died, the next sync call
    (page.close in pdf_bytes' finally, browser.close in __exit__) spun forever at 100% CPU. The renderer must notice
    the dead connection, kill the browser the dead driver left behind, restart, retry the render and exit cleanly.
    Runs in a subprocess with a timeout so a regression fails here instead of hanging the suite."""
    src = str(Path(render_mod.__file__).resolve().parents[2])
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER_KILL_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": src},
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert proc.stdout.splitlines() == ["first 1", "between 1", "during 1", "closed", "orphaned browsers 0"], out
    assert out.count("드라이버 연결이 끊김") == 2, out


# ------------------------------------------------- QA regression: run-wide --no-llm noise on page 1


LLM_SKIP_OVERRIDE = (
    "LLM 분석 미실행 — LLM 필러 제외 후 팩터·ML 가중치 재정규화 · 합의도 점수는 가용 필러 비중 60%만 반영"
)


def _no_llm_report(sample6):
    rep = sample6.model_copy(deep=True)
    for t in rep.tickers:
        t.llm.skipped = True
        t.llm.news = t.llm.filings = t.llm.critic = None
        t.llm.errors, t.llm.notices = [], []
        t.executive.overrides = [LLM_SKIP_OVERRIDE]
        t.warnings = []
    rep.tickers[0].warnings = ["가격: SPY 거래일 대비 1거래일 결측(2026-09-22)"]
    rep.tickers[1].executive.overrides = [
        "배리어 가드레일: P(손절 선행) 55% > P(익절 선행) 36% + 15%p → Buy→Hold",
        LLM_SKIP_OVERRIDE,
    ]
    return rep


def test_run_wide_no_llm_is_stated_once_not_per_ticker(sample6, module_settings):
    """Live run (6 names, --no-llm): 12 of 14 page-1 alerts and every '⚠ 2' badge were the same LLM-skipped
    text, which buried the one real data warning. The run-wide skip is stated once; real items stay."""
    rep = _no_llm_report(sample6)
    view = build_view(rep, module_settings, 0, None)
    texts = [a["text"] for g in view["alert_groups"] for a in g["entries"]]
    assert not any("LLM 분석 미실행" in x or "LLM 분석 생략" in x for x in texts)
    assert any("1거래일 결측" in x for x in texts) and any(x.startswith("배리어 가드레일") for x in texts)
    rows = view["rows"]
    assert rows[0]["n_alerts"] == 1 and rows[1]["n_alerts"] == 1
    # the badge counts real items only (the sample's degraded RSMT-Q row keeps its alert)
    assert all(r["n_alerts"] == sum("저하" in a["text"] for a in r["alerts"]) for r in rows[2:])
    assert any("저하" in a["text"] for r in rows for a in r["alerts"])
    assert rows[1]["overrides"] == [rep.tickers[1].executive.overrides[0]]  # the guardrail chip stays
    assert all(LLM_SKIP_OVERRIDE not in r["overrides"] for r in rows)
    assert [n["text"] for n in view["notes"]].count(f"전 종목 공통: {LLM_SKIP_OVERRIDE}") == 1
    text = re.sub(r"\s+", " ", _visible_text(render_html(rep, module_settings, 0, None)))
    assert text.count("LLM 분석 미실행") == 1  # the report-level note only
    assert "미산출(LLM 생략)" in text and "뉴스 없음" not in text  # Module 3 does not claim there was no news


def test_partial_llm_skip_still_flags_each_skipped_ticker(sample6, module_settings):
    rep = _no_llm_report(sample6)
    rep.tickers[2].llm.skipped = False  # not run-wide: the per-ticker items must stay visible
    view = build_view(rep, module_settings, 0, None)
    row0 = view["rows"][0]
    assert any("LLM 분석 생략" in a["text"] for a in row0["alerts"])
    assert LLM_SKIP_OVERRIDE in row0["overrides"]
    assert not any(n["text"].startswith("전 종목 공통") for n in view["notes"])


def test_ml_columns_without_a_selection_model_follow_the_distribution(sample6, module_settings):
    """No stock-selection model: the ML columns show the RSMT-Q distribution that carries the ML pillar (E[R],
    P(up), CV AUC · IC, direction gate); a ticker without a distribution has no ML pillar -> the 'No ML' cells."""
    rep = sample6.model_copy(deep=True)
    d = rep.tickers[0].quant.distribution
    rep.tickers[1].quant.distribution = None
    rows = build_view(rep, module_settings, 0, None)["rows"]
    ml = rows[0]["ml"]
    assert ml["er"] == fmt_pct(d.expected_return) and ml["pup"] == fmt_prob(d.prob_up)
    assert ml["auc"] == fmt_num(d.cv_metrics["auc"], 3) and ml["ic"] == fmt_num(d.cv_metrics["ic"], 3, signed=True)
    assert ml["skill"] is d.direction_skill_ok
    assert rows[1]["ml"] is None
    text = _visible_text(render_html(rep, module_settings, 0, None))
    assert "CV AUC · IC" in text and "ML 없음" in text
    assert "RSMT-Q 분포 예측 없음 — ML 필러 제외" in text


# ------------------------------------------- QA regression: right-edge clipping, Kelly wording, robust gate

# rows whose text is right-aligned to (or can run into) the page's right edge
_EDGE_ROWS = [
    "header.top",
    "dl.meta",
    ".chips-row",
    "h3.sub-h",
    ".chart-legend",
    ".alloc-head",
    ".alloc-legend",
    ".key",
    "section.mod > h2",
    "section.mod > .keep > h2",
    ".trust",
    ".trig .h b",  # a real-length name is shown whole (only a name wider than half the card is ellipsised)
]
_EDGE_JS = """(sels) => {
  const W = document.documentElement.clientWidth, rows = [], rg = document.createRange();
  for (const s of sels) for (const e of document.querySelectorAll(s)) {
    let right = e.getBoundingClientRect().right;
    for (const c of e.querySelectorAll('*')) {
      const r = c.getBoundingClientRect(); if (r.width && r.height) right = Math.max(right, r.right);
    }
    const tw = document.createTreeWalker(e, NodeFilter.SHOW_TEXT);
    while (tw.nextNode()) {
      rg.selectNodeContents(tw.currentNode); const r = rg.getBoundingClientRect();
      if (r.width) right = Math.max(right, r.right);
    }
    rows.push([s, e.scrollWidth, e.clientWidth, Math.round(right * 100) / 100]);
  }
  return [document.documentElement.scrollWidth, W, rows]; }"""
# grows the header meta, the chips and the chart legend far beyond their real size (a longer universe /
# digest / as-of text, extra regime chips, extra legend items) to prove they wrap instead of overflowing
_GROW_JS = """() => {
  const dds = document.querySelectorAll('dl.meta dd');
  dds[0].textContent += ' · KR 2026-09-23 · US 2026-09-22 (시장별 기준일 상이)';
  dds[dds.length - 1].textContent += ' (렌더 시점 설정 · override config/override_long_name.yaml)';
  const row = document.querySelector('.chips-row');
  for (let i = 0; i < 6; i++) { const c = row.querySelector('.chip').cloneNode(true); row.appendChild(c); }
  const lg = document.querySelector('.chart-legend');
  for (const s of [...lg.children]) lg.appendChild(s.cloneNode(true));
  return true; }"""
EDGE_GAP_PX = 2  # nothing may end within this distance of the clip edge (anti-aliased glyphs were cut)


@pytest.mark.parametrize("grow", [False, True], ids=["as-rendered", "grown-content"])
def test_right_edge_rows_never_overflow_or_touch_the_page_edge(grow, pdf_renderer, sample6, module_settings):
    """QA regression (real report 2026-09-24): the page-1 header meta and the chart legend ended exactly at
    the @page content edge, and Chrome's print clip cut the last glyph ("목표가" read "목표기"). Every
    right-aligned row must fit its own box (scrollWidth <= clientWidth) and stay clear of the edge."""
    rep = sample6.model_copy(deep=True)
    rep.config_digest = ""  # longest digest form: "<hash> (렌더 시점 설정)"
    rep.market_regime = {
        "KOSPI": "20D +4.0% · MA200 위",
        "KOSDAQ": "20D +2.1% · MA200 아래",
        "S&P 500": "20D +0.5% · MA200 위",
        "VIX": "15.2 (보통)",
        "USD/KRW": "1,350",
    }
    rep.tickers[1].as_of = rep.tickers[0].as_of - pd.Timedelta(1, "D")  # two as-of dates in the header
    rep.tickers[5].instrument.name = "Microsoft Corporation"  # wrapped around its badge, then got ellipsised
    html = render_html(rep, module_settings, 0, build_sample_charts(rep))
    page = pdf_renderer._browser.new_page(viewport={"width": 703, "height": 1050})  # A4 content box (186 mm)
    try:
        page.emulate_media(media="print")
        page.set_content(html, wait_until="load")
        if grow:
            page.evaluate(_GROW_JS)
        scroll_w, width, rows = page.evaluate(_EDGE_JS, _EDGE_ROWS)
    finally:
        page.close()
    assert width == 703 and scroll_w <= width, f"document wider than the page ({scroll_w}px)"
    seen = {r[0] for r in rows}
    assert {"header.top", "dl.meta", ".chips-row", ".chart-legend", ".key", ".trust"} <= seen, seen
    overflow = [r for r in rows if r[1] > r[2]]
    assert not overflow, f"row content wider than its box: {overflow}"
    touching = [r for r in rows if r[3] > width - EDGE_GAP_PX]
    assert not touching, f"ends within {EDGE_GAP_PX}px of the clipped page edge: {touching}"


def test_kelly_cell_shows_capped_fraction_and_never_f_star_100(sample6, module_settings):
    """QA: 5 of 6 live rows printed '25.0% / f* 100%' - f* at the grid bound read as an all-in recommendation.
    The capped fraction is primary; f* only while it rounds below 100%, otherwise '한도 적용'."""
    rep = sample6.model_copy(deep=True)
    cap = module_settings.risk.kelly_cap
    cases = {0: 1.0, 1: 0.996, 2: 0.43, 3: 0.0}
    for i, f in cases.items():
        plan = rep.tickers[i].execution
        plan.barrier = plan.barrier.model_copy(update={"kelly_optimal": f})
        plan.kelly_fraction = f * cap
    rep.tickers[4].execution.barrier = None
    rep.tickers[4].execution.kelly_fraction = 0.05
    rows = build_view(rep, module_settings, 0, None)["rows"]
    assert rows[0]["kelly"] == {"main": fmt_prob(cap, 1), "sub": render_mod.KELLY_LIMIT_TXT, "at_limit": True}
    assert rows[1]["kelly"]["sub"] == render_mod.KELLY_LIMIT_TXT  # 99.6% would print as "100%"
    assert rows[2]["kelly"] == {"main": "10.8%", "sub": "f* 43%", "at_limit": False}
    assert rows[3]["kelly"]["main"] == "0.0%" and rows[3]["kelly"]["sub"] == "f* 0%"
    assert rows[4]["kelly"] == {"main": "5.0%", "sub": f"f* {MISSING}", "at_limit": False}
    html = render_html(rep, module_settings, 0, None)
    text = re.sub(r"\s+", " ", _visible_text(html))
    assert "f* 100%" not in text and "한도 적용" in text and "f* 43%" in text
    assert "전액 투자 권고가 아닙니다" in text  # the methodology says what the wording means


def _set_cv(rep: PortfolioReport, market: str, update: dict, skill_ok: bool | None = None) -> None:
    for t in rep.tickers:
        d = t.quant.distribution
        if t.instrument.market.value == market and d is not None:
            cv = {k: v for k, v in d.cv_metrics.items() if not k.startswith("gate_folds")}
            cv.update(update)
            d.cv_metrics = cv
            if skill_ok is not None:
                d.skill_ok = skill_ok


@pytest.mark.parametrize(
    "gate,skill_ok,text,detail,cls",
    [
        (
            {
                "gate_folds_3": 0.0,
                "gate_folds_4": 1.0,
                "gate_folds_5": 0.0,
                "gate_folds_n": 3.0,
                "gate_folds_pass": 1.0,
            },
            False,
            "3개 폴드 구성 중 1개만 통과 → 스킬 미확인",
            "3폴드 ✗ · 4폴드 ✓ · 5폴드 ✗",
            "no",
        ),
        (
            {"gate_folds_3": 1.0, "gate_folds_4": 1.0, "gate_folds_5": 1.0},  # counts derived from the flags
            True,
            "3개 폴드 구성 모두 통과 → 스킬 확인",
            "3폴드 ✓ · 4폴드 ✓ · 5폴드 ✓",
            "ok",
        ),
        (
            {"gate_folds_n": 3.0, "gate_folds_pass": 2.0},  # aggregates only
            True,
            "3개 폴드 구성 중 2개 통과 → 스킬 확인",
            "",
            "ok",
        ),
        (
            {"gate_folds_3": 0.0, "gate_folds_4": 0.0, "gate_folds_5": float("nan")},  # 5-fold layout not run
            False,
            "3개 폴드 구성 모두 미통과 → 스킬 미확인",
            "3폴드 ✗ · 4폴드 ✗ · 5폴드 미실행",
            "no",
        ),
        (
            {"gate_folds_n": 5.0, "gate_folds_pass": 3.0, "gate_folds_need": 4.0},  # stricter than a majority
            False,
            "5개 폴드 구성 중 3개만 통과 → 스킬 미확인 (필요 4개)",
            "",
            "no",
        ),
    ],
)
def test_model_trust_line_states_the_robust_gate_in_plain_korean(
    gate, skill_ok, text, detail, cls, sample6, module_settings
):
    rep = sample6.model_copy(deep=True)
    _set_cv(rep, "KR", gate, skill_ok)
    kr = {d["market"]: d for d in build_view(rep, module_settings, 0, None)["diagnostics"]}["KR"]
    assert kr["trust"] == {"text": text, "detail": detail, "cls": cls}
    for level in range(4):  # a verdict, not detail: never trimmed by compaction
        visible = _visible_text(render_html(rep, module_settings, level, None))
        assert visible.count("위험 엔진 RSMT-Q") == 2 and text in visible  # one line per market (KR + US)


def test_sample_trust_lines_match_the_skill_badges(sample6, module_settings):
    diags = {d["market"]: d for d in build_view(sample6, module_settings, 0, None)["diagnostics"]}
    assert diags["KR"]["trust"]["text"] == "3개 폴드 구성 모두 통과 → 스킬 확인"
    assert diags["US"]["trust"]["text"] == "3개 폴드 구성 중 2개 통과 → 스킬 확인"
    assert diags["US"]["trust"]["detail"] == "3폴드 ✓ · 4폴드 ✗ · 5폴드 ✓"
    assert all(d["skill"] is True and d["trust"]["cls"] == "ok" for d in diags.values())


def test_model_trust_line_without_gate_keys_or_with_conflicts_is_explicit(sample6, module_settings, caplog):
    rep = sample6.model_copy(deep=True)
    _set_cv(rep, "KR", {}, skill_ok=False)  # older JSON / rsmt without the robust gate
    _set_cv(rep, "US", {}, skill_ok=True)
    diags = {d["market"]: d for d in build_view(rep, module_settings, 0, None)["diagnostics"]}
    assert diags["KR"]["trust"]["text"] == "단일 4폴드 CV 미달 → 스킬 미확인"
    assert diags["US"]["trust"]["text"] == "단일 4폴드 CV만 통과 → 견고성 미검증"
    assert all(d["trust"]["detail"] == "폴드 구성별 게이트 기록 없음" for d in diags.values())
    # the two verdicts disagree -> both are printed and logged, never silently reconciled
    _set_cv(rep, "KR", {"gate_folds_n": 3.0, "gate_folds_pass": 1.0}, skill_ok=True)
    # an unparseable gate record is shown as such
    _set_cv(rep, "US", {"gate_folds_score": 0.4})
    with caplog.at_level(logging.WARNING, logger="quantcio.report.render"):
        diags = {d["market"]: d for d in build_view(rep, module_settings, 0, None)["diagnostics"]}
    assert diags["KR"]["trust"]["text"] == "3개 폴드 구성 중 1개만 통과 → 스킬 미확인 · 최종 스킬 판정 통과"
    assert diags["KR"]["trust"]["cls"] == "no"
    assert "해석 불가" in diags["US"]["trust"]["text"] and "gate_folds_score" in diags["US"]["trust"]["text"]
    msgs = " | ".join(r.getMessage() for r in caplog.records)
    assert "불일치" in msgs and "해석 불가" in msgs
    rep.tickers = [t for t in rep.tickers if t.instrument.market.value == "KR"]
    for t in rep.tickers:
        t.quant.distribution = None
    kr = build_view(rep, module_settings, 0, None)["diagnostics"][0]
    assert kr["trust"]["cls"] == "na" and "검증 정보 없음" in kr["trust"]["text"]


# --------------------------------------------------------------------------- English output + new data contracts

_HANGUL = re.compile("[가-힣]")


@pytest.fixture(scope="module")
def en_settings(module_settings):
    s = module_settings.model_copy(deep=True)
    s.report.language = "en"
    return s


@pytest.fixture(scope="module")
def sample12_en(en_settings) -> PortfolioReport:
    return build_sample_report(12, settings=en_settings)


def test_text_table_has_both_languages():
    for key, (ko, en) in render_mod._TEXT.items():
        assert ko and en, key
        assert not _HANGUL.search(en), (key, en)


def test_sample_en_text_parity_with_korean_specs():
    from quantcio.report.sample import _SPECS
    from quantcio.report.sample_en import SAMPLE_TEXT_EN

    assert set(SAMPLE_TEXT_EN) == {s.ticker for s in _SPECS}
    for spec in _SPECS:
        e = SAMPLE_TEXT_EN[spec.ticker]
        assert (e["news"] is None) == (spec.news is None) and (e["filings"] is None) == (spec.filings is None)
        if spec.news:
            for k in ("key_positives", "key_negatives", "key_events"):
                assert len(e["news"][k]) == len(spec.news[k]), (spec.ticker, k)
        if spec.filings:
            for k in ("summary", "earnings_highlights", "risk_factors", "sources"):
                assert len(e["filings"][k]) == len(spec.filings[k]), (spec.ticker, k)
        for k in ("contradictions", "llm_errors", "warnings", "degraded", "extra_triggers", "llm_notices"):
            assert len(e[k]) == len(getattr(spec, k)), (spec.ticker, k)
        assert not _HANGUL.search(repr(e)), spec.ticker


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_english_report_has_no_hangul_and_english_formats(level, sample12_en, en_settings):
    html = render_html(sample12_en, en_settings, level, build_sample_charts(sample12_en) if level == 0 else None)
    text = _visible_text(html)
    assert not _HANGUL.search(text), sorted(set(re.findall(r"\S*[가-힣]+\S*", text)))[:20]
    assert '<html lang="en">' in html and 'class="c' in html and "lang-en" in html
    assert render_mod.REPORT_TITLE_EN in text and render_mod.DISCLAIMER_EN in text
    assert "₩286,500" in text and "$337.02" in text and "원" not in text
    assert "Executive Signal Matrix" in text and "Warnings · guardrails · degraded states" in text
    assert not re.search(r"\bnan\b|\bNone\b|\binf\b", text)


def test_english_footer_and_fit_messages(sample6, en_settings):
    foot = footer_template(sample6, en_settings)
    assert "For reference only · not investment advice" in foot and not _HANGUL.search(foot)
    assert "투자 권유 아님" in footer_template(sample6)  # Korean default unchanged


def test_price_format_english_matches_fusion_risk():
    risk = pytest.importorskip("quantcio.fusion.risk")
    for v, ccy in ((286_500, "KRW"), (71_200.4, "KRW"), (-1_500, "KRW"), (337.02, "USD")):
        assert fmt_price(v, ccy, "en").replace("−", "-") == risk.format_money(v, ccy, "en")
    assert fmt_money_compact(1e9, "KRW", "en") == "₩1.0B" and fmt_money_compact(-2.5e6, "KRW", "en") == "−₩2.5M"
    assert fmt_money_compact(1e9, "KRW") == "10억원"


def test_macro_strip_factor_model_and_backtest_render(sample12, module_settings):
    view = build_view(sample12, module_settings, 0, None)
    cells = {c["name"]: c for c in view["macro"]["cells"]}
    assert list(cells) == ["글로벌", "미국", "한국"]
    assert cells["한국"]["cls"] == "off" and cells["한국"]["mult"] == "×0.86" and cells["한국"]["n_stale"] == 1
    assert cells["미국"]["cls"] == "on" and not cells["미국"]["mult"]
    assert all(1 <= len(c["inds"]) <= 3 for c in cells.values())
    rows = {r["ticker"]: r for r in view["factor_model"]}
    assert set(rows) == {t.instrument.ticker for t in sample12.tickers}
    fm = rows["005930"]["fm"]
    assert fm["skill"] and fm["per"] == "14.2" and "→" not in fm["blended"]
    assert rows["AAPL"]["fm"]["skill"] is False
    assert rows["MSFT"]["fm"]["notes"]  # outside-universe note shown as a footnote
    markets = {b["market"]: b for b in view["backtest"]}
    assert set(markets) == {"KR", "US"} and len(markets["KR"]["rows"]) == 6
    assert markets["KR"]["model"]["skill"] and not markets["US"]["model"]["skill"]
    assert any(r["dir_bad"] for r in markets["US"]["rows"])
    html = render_html(sample12, module_settings, 0, None)
    assert 'class="macro-strip"' in html and 'class="fmodel"' in html and 'class="bt-grid"' in html
    assert "매크로 오버레이 ×" in _visible_text(html)  # Module 4 states the overlay-scaled gross limit
    # the factor-pillar blend is information, not a guardrail: never a warning chip or a page-1 alert
    for r in view["rows"]:
        assert not any(o.startswith("팩터 필러 블렌드") for o in r["overrides"])
        assert not any(a["text"].startswith("팩터 필러 블렌드") for a in r["alerts"])
    # the blended pillar shown in Module 2 is the one the signal used
    t = sample12.tickers[0]
    assert fm["blended"] == fmt_num(t.executive.pillars["factor"], 2, signed=True)


def test_new_contracts_empty_render_nothing_and_korean_css_is_unchanged(module_settings):
    rep = build_sample_report(4, settings=module_settings, with_models=False)
    assert not rep.macro and not rep.factor_backtest and all(t.quant.factor_model is None for t in rep.tickers)
    html = render_html(rep, module_settings, 0, None)
    markers = ('class="macro-strip"', 'class="fmodel"', 'class="bt-grid"', ".macro-strip {", "lang-en", 'class="ord"')
    for marker in markers:
        assert marker not in html
    css = (render_mod.TEMPLATE_DIR / "report.css").read_text(encoding="utf-8")
    assert f"<style>{css}</style>" in html  # exactly the stylesheet a Korean report always inlined


def _order_payload(rep: PortfolioReport, model: str = "lgbm_reg_w_ord", **stats) -> dict:
    kr = [t.instrument.ticker for t in rep.tickers if t.instrument.market.value == "KR"]
    return {
        "market": "KR",
        "model": model,
        "as_of": "2026-10-30",
        "age_days": 7,
        "n": 328,
        "k": 20,
        "horizon": 20,
        "stats": {
            "first": "2019-01",
            "last": "2026-08",
            "months": 92,
            "cost_bps": 50.0,
            "top_ann": 0.487,
            "top_ann_net": 0.439,
            "universe_ann": 0.131,
            "top_mdd": -0.48,
            "top_ann_drop10": 0.077,
            "mom_top_ann": 0.252,
            **stats,
        },
        "tickers": {
            kr[0]: {"ventile": 20, "percentile": 0.97, "in_universe": True},
            kr[1]: {"ventile": 7, "percentile": 0.33, "in_universe": False},
        },
        "top": [
            {"ticker": kr[0], "name": "첫 종목", "sector": "반도체", "percentile": 0.97},
            {"ticker": "036540", "name": "SFA반도체", "sector": "반도체", "percentile": 0.998},
        ],
    }


def _rf_payload(rep: PortfolioReport) -> dict:
    kr = [t.instrument.ticker for t in rep.tickers if t.instrument.market.value == "KR"]
    d = _order_payload(
        rep,
        "rf",
        top_ann=0.339,
        short_ko="랜덤포레스트",
        short_en="Random forest",
        abbr_ko="RF",
        abbr_en="RF",
        caveat_ko="추천이 아닙니다: RF 고유 주의",
        caveat_en="Not a recommendation: RF caveat",
    )
    d["tickers"] = {kr[0]: {"ventile": 12, "percentile": 0.6, "in_universe": True}}
    d["top"] = [
        {"ticker": "036540", "name": "SFA반도체", "sector": "반도체", "percentile": 0.99},
        {"ticker": "000210", "name": "DL", "sector": "화학", "percentile": 0.98},
    ]
    return d


def test_order_ventile_block_is_reference_only(sample12, sample12_en, module_settings, en_settings):
    """docs/MODELING.md 93, 94.3: the reference models' ventiles and D20 lists render as a caveated block; markets
    of the report without a model are named once; nothing else in the report changes."""
    rep = sample12.model_copy(deep=True)
    rep.order_ventile = {"KR": [_order_payload(rep)]}
    view = build_view(rep, module_settings, 0, None)
    o = view["order"]
    assert [b["market"] for b in o["blocks"]] == ["KR"] and "US" in o["absent"] and "88~90" in o["absent"]
    b = o["blocks"][0]
    kr = [t.instrument.ticker for t in rep.tickers if t.instrument.market.value == "KR"]
    assert [(r["ticker"], [c["v"] for c in r["cells"]], r["top"], r["outside"]) for r in b["rows"]] == [
        (kr[0], ["D20"], True, False),
        (kr[1], ["D7"], False, True),
    ]
    (lst,) = b["lists"]
    assert [n["mine"] for n in lst["top"]] == [True, False] and lst["label"] == "lgbm_reg_w_ord D20 2종목"
    assert "연 +48.7%" in lst["stats"] and "최고 10개월 제외 시 연 +7.7%" in lst["stats"] and b["has_outside"]
    assert lst["caveat"].startswith("추천이 아닙니다") and "91.5" in lst["caveat"] and not lst["stale"]
    assert not b["legend"] and not b["common"]  # one model: no abbreviations, no common list
    html = render_html(rep, module_settings, 0, None)
    text = _visible_text(html)
    assert 'class="ord"' in html and "20분위 참고 모델" in text and "신호·비중에 미반영" in text
    assert ".ord-row {" in html  # the extended stylesheet carries the block's rules
    plain = build_view(sample12, module_settings, 0, None)["rows"]
    assert [{k: v for k, v in r.items() if k != "ref"} for r in view["rows"]] == [
        {k: v for k, v in r.items() if k != "ref"} for r in plain
    ]  # the tables are unchanged apart from the Module 1 reference tag
    assert all(r["ref"] is None for r in plain)
    tag = {r["ticker"]: r["ref"] for r in view["rows"] if r["ref"]}
    assert tag == {
        kr[0]: {"txt": "참고 lgbm_reg_w_ord D20", "top": True},
        kr[1]: {"txt": "참고 lgbm_reg_w_ord D7", "top": False},
    }
    assert 'class="chip ref top"' in html and 'class="refm keep"' in html
    p = b["primary"]
    assert [[(n["rank"], n["ticker"], n["pos"], n["mine"]) for n in h] for h in p["halves"]] == [
        [(1, kr[0], "상위 3.0%", True)],
        [(2, "036540", "상위 0.2%", False)],
    ]
    assert [x["value"] for x in p["tiles"]] == ["+48.7%", "+43.9%", "+13.1%", "−48%", "+7.7%"]
    assert p["period"].startswith("사후 장부 2019-01~2026-08 · 92개월") and not p["other_label"] and not b["others"]
    old = rep.model_copy(deep=True)
    old.order_ventile["KR"][0]["age_days"] = 40
    assert "40일 경과" in build_view(old, module_settings, 0, None)["order"]["blocks"][0]["lists"][0]["stale"]
    # a report written before 94.3 holds one view (a dict) per market
    legacy = rep.model_copy(deep=True)
    legacy.order_ventile = {"KR": _order_payload(rep)}
    assert build_view(legacy, module_settings, 0, None)["order"] == o
    en = sample12_en.model_copy(deep=True)
    en.order_ventile = {"KR": [_order_payload(en)]}
    ben = build_view(en, en_settings, 0, None)["order"]
    assert ben["blocks"][0]["lists"][0]["caveat"].startswith("Not a recommendation") and "not shown" in ben["absent"]
    assert "20-quantile reference models" in _visible_text(render_html(en, en_settings, 0, None))


def test_order_ventile_block_shows_every_model(sample12, module_settings):
    """94.3: two reference models - each report name's ventile under both (abbreviated), one D20 list each with its
    own label, statistics and caveat, and the names in both D20 lists marked."""
    rep = sample12.model_copy(deep=True)
    rep.order_ventile = {"KR": [_order_payload(rep, short_ko="LightGBM 순서 모델", abbr_ko="순서"), _rf_payload(rep)]}
    b = build_view(rep, module_settings, 0, None)["order"]["blocks"][0]
    kr = [t.instrument.ticker for t in rep.tickers if t.instrument.market.value == "KR"]
    assert [(r["ticker"], [(c["abbr"], c["v"], c["top"]) for c in r["cells"]]) for r in b["rows"]] == [
        (kr[0], [("순서", "D20", True), ("RF", "D12", False)]),
        (kr[1], [("순서", "D7", False)]),
    ]
    assert b["legend"] == "순서 = LightGBM 순서 모델 · RF = 랜덤포레스트"
    assert [lst["label"] for lst in b["lists"]] == ["LightGBM 순서 모델 D20 2종목", "랜덤포레스트 D20 2종목"]
    assert b["lists"][1]["caveat"] == "추천이 아닙니다: RF 고유 주의" and "연 +33.9%" in b["lists"][1]["stats"]
    assert b["common"] == "SFA반도체 036540"
    assert [n["common"] for n in b["lists"][0]["top"]] == [False, True]
    assert [n["mine"] for n in b["lists"][1]["top"]] == [False, False]
    p = b["primary"]
    assert p["other_label"] == "RF D20" and [b["others"][0]["model"]] == ["rf"]
    assert [n["other"] for h in p["halves"] for n in h] == [False, True]
    view = build_view(rep, module_settings, 0, None)
    assert {r["ticker"]: r["ref"]["txt"] for r in view["rows"] if r["ref"]} == {
        kr[0]: "참고 순서 D20",
        kr[1]: "참고 순서 D7",
    }
    text = _visible_text(render_html(rep, module_settings, 0, None))
    assert "RF D12" in text and "공통 D20: SFA반도체 036540" in text and "RF 고유 주의" in text
    assert "참고 순서 D20" in text and "랜덤포레스트 D20" in text and "RF D20" in text


def test_us_avoid_flag_is_reference_only(sample12, sample12_en, module_settings, en_settings, monkeypatch, tmp_path):
    """docs/MODELING.md 103.4, 104: the US avoid-side flag (bottom 20% of 12-1 momentum, file written by
    scripts/serve_avoid.py) renders as a caveated block and a Module 1 tag; without the file, with an unusable file or
    without US names nothing is shown, and no other part of the report changes."""
    import json

    from quantcio.report import render

    us = [t.instrument.ticker for t in sample12.tickers if t.instrument.market.value == "US"]
    assert len(us) >= 2
    path = tmp_path / "avoid_US.json"
    monkeypatch.setattr(render, "_avoid_file", lambda _settings: path)
    plain = build_view(sample12, module_settings, 0, None)
    assert plain["avoid"] is None and all(r["avoid"] is None for r in plain["rows"])  # no file
    payload = {
        "as_of": sample12.generated_at.date().isoformat(),
        "q": 0.2,
        "n": 499,
        "n_avoid": 100,
        "tickers": {
            us[0].upper().replace(".", "-"): {"name": "x", "mom": -0.31, "pct": 0.07, "avoid": True},
            us[1].upper().replace(".", "-"): {"name": "y", "mom": 0.22, "pct": 0.64, "avoid": False},
        },
        "evidence": [
            {"ko": "프렌치 대형주 1963~2011", "en": "French big caps 1963-2011", "alpha": -0.0532, "t": -3.32},
            {"ko": "우리 자료 2012~2026", "en": "our data 2012-2026", "alpha": -0.0596, "t": -2.6},
        ],
        "doc": "103.4",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    view = build_view(sample12, module_settings, 0, None)
    a = view["avoid"]
    assert [(r["ticker"], r["pos"]) for r in a["rows"]] == [(us[0], "하위 7%")]
    assert "499종목" in a["sub"] and "하위 20%(100종목)" in a["sub"] and "신호·비중에 미반영" in a["sub"]
    assert "프렌치 대형주 1963~2011 −5.3% (t −3.3)" in a["stats"] and "우리 자료 2012~2026 −6.0% (t −2.6)" in a["stats"]
    assert a["caveat"].startswith("매도 권고가 아닙니다") and "103.4" in a["caveat"] and not a["stale"]
    tags = {r["ticker"]: r["avoid"] for r in view["rows"] if r["avoid"]}
    assert tags == {us[0]: {"txt": "참고 피할 쪽 · 12-1 하위 7%"}}
    assert [{k: v for k, v in r.items() if k != "avoid"} for r in view["rows"]] == [
        {k: v for k, v in r.items() if k != "avoid"} for r in plain["rows"]
    ]  # the tables are unchanged apart from the Module 1 tag
    html = render_html(sample12, module_settings, 0, None)
    text = _visible_text(html)
    assert "미국 피할 쪽 참고(사후 관찰)" in text and "참고 피할 쪽 · 12-1 하위 7%" in text
    assert 'class="chip ref avoid"' in html and ".chip.ref.avoid" in html
    en = build_view(sample12_en, en_settings, 0, None)["avoid"]
    assert en["h"].startswith("US avoid-side") and en["rows"][0]["pos"] == "bottom 7%"
    assert "French big caps 1963-2011" in en["stats"] and en["caveat"].startswith("Not a sell recommendation")
    payload["tickers"][us[0].upper().replace(".", "-")]["avoid"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    none = build_view(sample12, module_settings, 0, None)["avoid"]
    assert none["rows"] == [] and none["none"] == "이 리포트의 미국 종목 중 해당 없음"
    assert "해당 없음" in _visible_text(render_html(sample12, module_settings, 0, None))
    payload["as_of"] = "2020-01-31"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert "갱신 필요" in build_view(sample12, module_settings, 0, None)["avoid"]["stale"]
    path.write_text("{not json", encoding="utf-8")
    assert build_view(sample12, module_settings, 0, None)["avoid"] is None  # unusable file: no block, no error
    kr_only = sample12.model_copy(deep=True)
    kr_only.tickers = [t for t in kr_only.tickers if t.instrument.market.value == "KR"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert render._avoid_view(kr_only, module_settings, False) is None  # no US names: nothing to flag


def test_partial_factor_model_and_odd_backtest_payloads_are_visible(sample6, module_settings, caplog):
    rep = sample6.model_copy(deep=True)
    rep.tickers[1].quant.factor_model = None
    rep.factor_backtest = {"KR": rep.factor_backtest["KR"], "XX": "not a dict"}
    with caplog.at_level(logging.WARNING, logger="quantcio.report.render"):
        view = build_view(rep, module_settings, 0, None)
    assert view["factor_model"][1]["fm"] is None
    assert "팩터 모델 점수 없음" in _visible_text(render_html(rep, module_settings, 0, None))
    assert [b["market"] for b in view["backtest"]] == ["KR"]
    assert any("factor_backtest[XX]" in r.getMessage() for r in caplog.records)


def test_english_regime_chips_translate_pipeline_tokens(sample6, en_settings):
    rep = sample6.model_copy(deep=True)
    rep.market_regime = {"KOSPI": "20D +1.0% · MA200 위", "SPY": "20D -1.0% · MA200 아래", "VIX": "15.2 (저변동)"}
    vals = [r["v"] for r in build_view(rep, en_settings, 0, None)["regime"]]
    assert vals == ["20D +1.0% · above MA200", "20D -1.0% · below MA200", "15.2 (low vol)"]


def test_english_alerts_and_llm_skip_run_note(en_settings):
    rep = build_sample_report(6, settings=en_settings)
    skip = "LLM analysis not run — LLM pillar dropped, factor/ML weights renormalised"
    for t in rep.tickers:
        t.llm.skipped = True
        t.llm.news = t.llm.filings = t.llm.critic = None
        t.executive.overrides = [skip]
    view = build_view(rep, en_settings, 0, None)
    assert [n["text"] for n in view["notes"]].count(f"All names: {skip}") == 1
    assert not any(skip == o for r in view["rows"] for o in r["overrides"])
    text = _visible_text(render_html(rep, en_settings, 0, None))
    assert "n/a (LLM skipped)" in text and not _HANGUL.search(text)


def test_pdf_english_sample_fits_three_pages(pdf_renderer, tmp_path, en_settings):
    rep = build_sample_report(6, settings=en_settings)
    res = fit_to_pages(rep, en_settings, tmp_path / "en.pdf", charts=build_sample_charts(rep), renderer=pdf_renderer)
    assert res.fits and not res.fallback_html and res.compact_level <= 1  # charts are dropped first
    pages = _pdf_text(res.pdf_path)
    full = re.sub(r"\s+", "", "".join(pages))
    assert "�" not in full and not _HANGUL.search(full)
    for heading in MODULE_HEADINGS.values():
        assert re.sub(r"\s+", "", heading) in full
    assert re.sub(r"\s+", "", render_mod.DISCLAIMER_EN) in full
    assert "PITmulti-factormodel" in full and "Factorbacktestsummary" in full and "Macroregime" in full


# ------------------------------------------------------------------ PDF watchdog (report/worker.py)

_HANG_WORKER = r"""
import json, os, pickle, subprocess, sys, time
payload, result = sys.argv[1], sys.argv[2]
tag = os.environ["QC_TEST_TAG"]
kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True)  # ~Chrome
with open(tag + ".pids", "a") as f:
    f.write(f"{os.getpid()} {kid.pid}\n")
p = pickle.loads(open(payload, "rb").read())
with open(p["attempts_path"], "a") as f:  # the worker had found a fitting level before the driver hung
    f.write(json.dumps([1, 1.0, p["max_pages"]]) + "\n")
time.sleep(300)
"""

_CRASH_WORKER = r"""
import os, sys
with open(os.environ["QC_TEST_TAG"] + ".pids", "a") as f:
    f.write(f"{os.getpid()}\n")
print("SyntaxError: Unterminated string in JSON at position 65536", file=sys.stderr)
sys.exit(3)
"""

_FLAKY_WORKER = r"""
import os, pickle, sys, time
payload, result = sys.argv[1], sys.argv[2]
log = os.environ["QC_TEST_TAG"] + ".pids"
n = sum(1 for _ in open(log)) if os.path.exists(log) else 0
with open(log, "a") as f:
    f.write(f"{os.getpid()}\n")
if n == 0:
    sys.exit(1)  # the first run crashes
out = pickle.loads(open(payload, "rb").read())["out_path"]
open(out, "wb").write(b"%PDF-1.4 fake")
fit = {"pdf_path": out, "html_path": None, "pages": 2, "compact_level": 1, "scale": 0.95, "fits": True,
       "fallback_html": False, "attempts": [[0, 1.0, 4], [1, 0.95, 2]], "warnings": ["w1"]}
open(result + ".tmp", "wb").write(pickle.dumps({"ok": True, "fit": fit}))
os.replace(result + ".tmp", result)
if os.environ.get("QC_TEST_HANG_AFTER_RESULT"):
    time.sleep(300)  # e.g. Chrome never closes after the PDF was written
"""


def _worker_script(tmp_path: Path, body: str, monkeypatch) -> tuple[list[str], Path]:
    import sys

    script = tmp_path / "fake_worker.py"
    script.write_text(body, encoding="utf-8")
    tag = tmp_path / "run"
    monkeypatch.setenv("QC_TEST_TAG", str(tag))
    return [sys.executable, str(script)], tag.with_suffix(".pids")


def _gone(pid: int, wait: float = 10.0) -> bool:
    import os
    import time

    t0 = time.monotonic()
    while time.monotonic() - t0 < wait:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_isolated_render_hang_is_killed_retried_once_then_html(tmp_path, monkeypatch, sample6, module_settings):
    """Incident 2026-09-25: a hung renderer must never hang the report. A worker that never finishes is killed
    (with its detached grandchild, like Playwright's Chrome) after the timeout, retried ONCE, then the HTML is
    written at the level the worker had chosen, with a page-1 note and a warning."""
    import time

    from quantcio.report.worker import fit_to_pages_isolated

    cmd, pids_file = _worker_script(tmp_path, _HANG_WORKER, monkeypatch)
    t0 = time.monotonic()
    res = fit_to_pages_isolated(sample6, module_settings, tmp_path / "r.pdf", timeout=2.0, worker_cmd=cmd)
    assert time.monotonic() - t0 < 30.0
    runs = [line.split() for line in pids_file.read_text().splitlines()]
    assert len(runs) == 2  # the first try + exactly one retry
    assert all(_gone(int(pid)) for run in runs for pid in run)  # worker AND its detached child are dead
    assert res.fallback_html and res.pdf_path is None and res.pages == 0 and not res.fits
    assert res.compact_level == 1  # the worker's chosen level, not blindly the maximum
    assert res.html_path == tmp_path / "r.html" and res.html_path.exists()
    assert not (tmp_path / "r.pdf").exists()
    html = res.html_path.read_text(encoding="utf-8")
    assert "PDF 생성 실패(렌더러 2회: 시간 초과 2초; 시간 초과 2초)" in html
    assert len(res.warnings) == 1 and "PDF 렌더러 2회 실패" in res.warnings[0] and str(res.html_path) in res.warnings[0]


@pytest.mark.parametrize("lang", ["ko", "en"])
def test_isolated_render_crash_is_retried_once(tmp_path, monkeypatch, sample6, module_settings, lang):
    from quantcio.report.render import MAX_COMPACT_LEVEL
    from quantcio.report.worker import fit_to_pages_isolated

    settings = module_settings.model_copy(deep=True)
    settings.report.language = lang
    cmd, pids_file = _worker_script(tmp_path, _CRASH_WORKER, monkeypatch)
    res = fit_to_pages_isolated(sample6, settings, tmp_path / "r.pdf", worker_cmd=cmd)
    assert len(pids_file.read_text().splitlines()) == 2
    assert res.fallback_html and res.pages == 0 and res.compact_level == MAX_COMPACT_LEVEL
    assert "Unterminated string in JSON" in res.warnings[0]
    if lang == "en":
        assert "PDF renderer failed 2x (crashed (exit code 3)" in res.warnings[0]
        assert not re.search(r"[가-힣]", res.warnings[0])
    else:
        assert "PDF 렌더러 2회 실패(비정상 종료(코드 3)" in res.warnings[0]


@pytest.mark.parametrize("hang_after_result", [False, True])
def test_isolated_render_retry_success_and_result_without_exit(
    tmp_path, monkeypatch, sample6, module_settings, hang_after_result
):
    from quantcio.report import worker

    monkeypatch.setattr(worker, "EXIT_GRACE_S", 0.5)
    if hang_after_result:
        monkeypatch.setenv("QC_TEST_HANG_AFTER_RESULT", "1")
    cmd, pids_file = _worker_script(tmp_path, _FLAKY_WORKER, monkeypatch)
    res = worker.fit_to_pages_isolated(sample6, module_settings, tmp_path / "r.pdf", timeout=60.0, worker_cmd=cmd)
    pids = [int(x) for x in pids_file.read_text().split()]
    assert len(pids) == 2 and all(_gone(p) for p in pids)
    assert res.pdf_path == tmp_path / "r.pdf" and res.pages == 2 and res.fits and not res.fallback_html
    assert res.attempts == [(0, 1.0, 4), (1, 0.95, 2)] and res.compact_level == 1
    assert (
        res.warnings[0].startswith("PDF 렌더러 실패 1회(비정상 종료(코드 1)")
        and "재시도 2회차에 성공" in res.warnings[0]
    )
    assert res.warnings[1:] == ["w1"]


def test_isolated_render_real_worker(pdf_renderer, tmp_path, sample6, module_settings):
    """The real `python -m quantcio.report.worker` round trip (pickled report + settings, real Chrome)."""
    from quantcio.report.worker import fit_to_pages_isolated

    res = fit_to_pages_isolated(sample6, module_settings, tmp_path / "w.pdf", charts=build_sample_charts(sample6))
    assert res.pdf_path == tmp_path / "w.pdf" and res.pdf_path.exists() and not res.fallback_html
    assert res.pages == count_pdf_pages(res.pdf_path.read_bytes())
    assert 1 <= res.pages <= module_settings.report.max_pages and res.fits
    assert res.attempts and res.html_path is not None and res.html_path.exists()


def test_isolated_render_validates_before_spawning(tmp_path, sample6, module_settings):
    from quantcio.report.worker import fit_to_pages_isolated

    with pytest.raises(ValueError, match="max_pages"):
        fit_to_pages_isolated(sample6, module_settings, tmp_path / "x.pdf", max_pages=0, worker_cmd=["false"])
    with pytest.raises(ValueError, match="attempts"):
        fit_to_pages_isolated(sample6, module_settings, tmp_path / "x.pdf", attempts=0, worker_cmd=["false"])


def test_quant_table_shows_the_stock_selection_model_when_served(module_settings):
    from datetime import date

    from quantcio.models import ExcessForecast

    rep = build_sample_report(4, settings=module_settings)
    plain = render_html(rep, module_settings, 0, None)
    assert "ML 기대수익" in plain and "선택 모델 백분위" not in plain  # no selection model: the RSMT-Q ML columns
    assert "앙상블" not in plain
    tickers = []
    for i, t in enumerate(rep.tickers):
        fc = ExcessForecast(
            model="logit_w",
            as_of=date(2026, 9, 23),
            mu_excess=0.004 * (i - 1),
            percentile=0.2 + 0.2 * i,
            skill_ok=i % 2 == 0,
            gate={"confirmed": i % 2 == 0},
            oos_auc=0.534,
            holdout_auc=0.521,
        )
        tickers.append(t.model_copy(update={"quant": t.quant.model_copy(update={"excess_model": fc})}))
    html = render_html(rep.model_copy(update={"tickers": tickers}), module_settings, 0, None)
    assert "선택 모델 백분위" in html and "AUC 표본 외 / 홀드아웃" in html and "0.534 / 0.521 · XSH 확인" in html
    # the direction model actually used is named with its performance; RSMT-Q is labelled risk / probability only
    assert "ML 방향 = 종목 선택 모델(" in html and "[logit_w] AUC 0.534 · 홀드아웃 0.521" in html
    assert "위험·확률 전용" in html and "ML(logit_w)" in html and "앙상블" not in html
    en = module_settings.model_copy(deep=True)
    en.report.language = "en"
    html_en = render_html(rep.model_copy(update={"tickers": tickers}), en, 0, None)
    assert "Selection pct." in html_en and "ML ensemble" not in html_en
    assert "[logit_w] AUC 0.534 · holdout 0.521" in html_en and "risk and probabilities only" in html_en


def test_diagnostics_lead_with_the_served_selection_model(sample6, module_settings):
    """Each market's diagnostics card first names the stock-selection model that sets the ML direction (the served
    one, with its OOS / holdout AUC); RSMT-Q below it is labelled the risk engine; the methodology says the same."""
    from datetime import date

    from quantcio.models import ExcessForecast

    kr_fc = ExcessForecast(
        model="et",
        model_label="엑스트라 트리(rank-gauss 목표)",
        as_of=date(2026, 10, 14),
        mu_excess=0.01,
        skill_ok=True,
        gate={"baseline_auc": 0.49, "confirmed": True},
        oos_auc=0.5432,
        oos_auc_lo=0.5217,
        holdout_auc=0.5430,
    )
    us_fc = ExcessForecast(
        model="xgb_clf",
        model_label="XGBoost 분류(날짜 중앙값 초과) · 사용자 지정(홀드아웃 1위, 사전 규칙 선택은 lgbm_reg_w)",
        as_of=date(2026, 10, 21),
        mu_excess=0.0,
        skill_ok=False,
        gate={"baseline_auc": 0.4975},
        oos_auc=0.5027,
        oos_auc_lo=0.4878,
        holdout_auc=0.5181,
    )
    tickers = []
    for t in sample6.tickers:
        fc = kr_fc if t.instrument.market.value == "KR" else us_fc
        tickers.append(t.model_copy(update={"quant": t.quant.model_copy(update={"excess_model": fc})}))
    rep = sample6.model_copy(update={"tickers": tickers})
    view = build_view(rep, module_settings, 0, None)
    d = {x["market"]: x for x in view["diagnostics"]}
    assert d["KR"]["sel"] == {
        "text": "엑스트라 트리 [et]",
        "detail": "표본 외 AUC 0.543(하한 0.522 · 모멘텀 0.490) · 홀드아웃 0.543 · 게이트 통과 · XSH 확인",
        "cls": "ok",
    }
    assert d["US"]["sel"]["text"] == "XGBoost 분류 [xgb_clf] · 사용자 지정(홀드아웃 1위)"
    assert d["US"]["sel"]["cls"] == "no" and "게이트 미달" in d["US"]["sel"]["detail"]
    assert "[et]" in view["methodology"]["sel"][0] and "[xgb_clf]" in view["methodology"]["sel"][0]
    visible = _visible_text(render_html(rep, module_settings, 0, None))
    assert visible.index("방향 모델") < visible.index("위험 엔진 RSMT-Q")
    assert "종목 선택 모델" in visible and "RSMT-Q(위험 엔진)" in visible
    assert all(x["sel"] is None for x in build_view(sample6, module_settings, 0, None)["diagnostics"])  # none served


def test_v4_forecasts_show_the_money_metric_first(sample6, module_settings):
    """Protocol v4 (docs/MODELING.md 79): the selection column, the subtitle and the diagnostics line lead with the
    annualised top-quintile net excess; the AUC is kept as the secondary number."""
    from datetime import date

    from quantcio.models import ExcessForecast

    fc = ExcessForecast(
        model="lgbm_reg",
        model_label="LightGBM 회귀",
        as_of=date(2026, 10, 25),
        mu_excess=0.002,
        skill_ok=False,
        gate={"baseline": 0.0005, "value": 0.0021, "lo": -0.0003},
        oos_primary=0.0021,
        oos_primary_lo=-0.0003,
        holdout_primary=0.0010,
        oos_auc=0.521,
        holdout_auc=0.509,
    )
    tickers = [t.model_copy(update={"quant": t.quant.model_copy(update={"excess_model": fc})}) for t in sample6.tickers]
    rep = sample6.model_copy(update={"tickers": tickers})
    view = build_view(rep, module_settings, 0, None)
    assert view["sel_money"] is True
    row = next(r for r in view["rows"] if r.get("sel"))
    assert row["sel"]["auc"] == "+2.5%" and row["sel"]["hold"] == "+1.2%"
    d = view["diagnostics"][0]["sel"]
    assert d["detail"].startswith("표본 외 순초과(연) +2.5%(하한 −0.4% · 모멘텀 +0.6%) · 홀드아웃 +1.2% · AUC 0.521")
    html = render_html(rep, module_settings, 0, None)
    assert (
        "상위 20% 순초과(연) 표본 외 / 홀드아웃" in html
        and "[lgbm_reg] 순초과 연 +2.5% · 홀드아웃 +1.2% · AUC 0.521" in html
    )
