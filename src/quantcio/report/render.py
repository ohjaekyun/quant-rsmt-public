"""A4 CIO report (Korean, or English with settings.report.language == "en"): PortfolioReport -> HTML (Jinja2) ->
PDF (Playwright + installed Chrome).

Public API
----------
- `render_html(report, settings, compact_level=0, charts=None) -> str`
- `render_pdf(html, out_path, *, renderer=None, scale=1.0, footer_template=None, html_fallback=True) -> int`
  (page count; 0 when no browser could start and the HTML fallback was written - logged at WARNING)
- `fit_to_pages(report, settings, out_path, *, charts=None, ...) -> FitResult`
  (compact levels 0..3 x print-scale steps until pages <= settings.report.max_pages; when nothing fits,
  the written PDF/HTML carries a page-1 overflow note)
- `portfolio_risk_labels() -> dict[str, str]` (fusion.risk.PORTFOLIO_RISK_LABELS, local fallback)
- `PdfRenderer` - reusable context manager (one Chrome process for many renders)
- `count_pdf_pages(pdf) -> int`, `save_html_fallback(html, out_path) -> Path`, `NoBrowserError`

Compact levels (content is dropped in this order, warnings never are):
  0 = everything, 1 = hide price charts, 2 = 2 filing bullets / 2 risks / fewer diagnostics lines,
  3 = also drop the article-level news lines and critic contradictions, 2 triggers per name.

Rendering rules: every missing value is shown as "—" (never "nan"/"None"); warnings, guardrail overrides,
RSMT-Q degraded states, LLM errors (red) and LLM notices / skipped layers (amber) are always visible
(chips + page-1 box). Prices: KRW '286,500원', USD '$337.02' (same as fusion/risk.py). Module 4 prints the
barrier's own simulated horizon (BarrierOutcome.horizon_days, "기간 미기재" when 0); a next earnings date
0..14 days ahead gets a '실적 D-n' chip (Module 1 thesis row + Module 4 trigger card).
Language: every static label, heading, chip, table header, footer, methodology line and disclaimer comes from
the `_TEXT` table (Korean / English side by side) and number, currency and date formatting follows the language
(KRW '286,500원' vs '₩286,500'). Korean output is byte-identical to the Korean-only renderer it replaced.
New data contracts (docs/BACKTEST_MACRO.md) are rendered when present and omitted when empty: the page-1 macro
strip (PortfolioReport.macro), the Module 2 factor-model table (QuantSignal.factor_model) and the backtest summary
(PortfolioReport.factor_backtest).
The browser chain follows the verified reference (docs/sources/reference/render_pdf.py): Chrome channel
(env QUANTCIO_BROWSER_CHANNEL) -> executable path (env QUANTCIO_CHROME or the macOS default) -> Playwright's
bundled chromium -> `NoBrowserError` -> HTML fallback. The sync Playwright API cannot run inside a running
asyncio loop; `render_pdf`/`fit_to_pages` detect that and run in a worker thread (prefer
`await asyncio.to_thread(...)` in async code).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import functools
import html as _html
import io
import json
import logging
import math
import os
import re
import signal
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from markupsafe import Markup

from quantcio.backtest.sectors import sector_label
from quantcio.config import Settings
from quantcio.i18n import is_en
from quantcio.macro.regime import INDICATOR_LABELS_EN as _MACRO_INDICATOR_LABELS_EN
from quantcio.macro.regime import LABELS_EN as _MACRO_LABELS_EN
from quantcio.models import (
    DistributionForecast,
    ExecutionPlan,
    FactorModelScore,
    FilingIntelligence,
    MacroRegime,
    NewsSentiment,
    PortfolioReport,
    RiskFactor,
    Signal,
    TickerReport,
)

log = logging.getLogger(__name__)

__all__ = [
    "MISSING",
    "MAX_COMPACT_LEVEL",
    "REPORT_TITLE",
    "REPORT_TITLE_EN",
    "DISCLAIMER",
    "DISCLAIMER_EN",
    "MODULE_HEADINGS",
    "NoBrowserError",
    "PdfRenderer",
    "FitResult",
    "build_view",
    "portfolio_risk_labels",
    "render_html",
    "render_pdf",
    "fit_to_pages",
    "count_pdf_pages",
    "save_html_fallback",
    "footer_template",
]

TEMPLATE_DIR = Path(__file__).with_name("templates")
MISSING = "—"
MINUS = "−"
MAX_COMPACT_LEVEL = 3
BAND_AXIS_MAX = 0.5  # the shared 10-90% band axis never exceeds ±50% (outliers are clipped with a marker)
# what each compaction level removes - printed on page 1 so a compacted report is never silently thinner
COMPACT_OMISSIONS = {
    1: "가격 차트 생략",
    2: "공시 요약·리스크 종목당 2개, 모델 진단 5줄로 축약",
    3: "기사 단위 뉴스·상충 신호 생략, 트리거 종목당 2개",
}
COMPACT_OMISSIONS_EN = {
    1: "price charts omitted",
    2: "filing summary / risks capped at 2 per name, model diagnostics cut to 5 lines",
    3: "article-level news and contradictions omitted, 2 triggers per name",
}
KST = ZoneInfo("Asia/Seoul")
# BarrierOutcome.kelly_optimal is an argmax on the grid f in [0, 1]: a value that rounds to 100% means the
# search hit its upper bound (the true optimum is >= 100%, i.e. leverage), not a recommendation to bet it all
KELLY_LIMIT_TXT = "한도 적용"
KELLY_LIMIT_TXT_EN = "cap applied"
REPORT_TITLE = "퀀트·AI 매매 타이밍 리포트"
REPORT_TITLE_EN = "Quant · AI Trade Timing Report"
DISCLAIMER = "본 자료는 투자 참고용이며 투자 판단의 최종 책임은 투자자 본인에게 있습니다."
DISCLAIMER_EN = (
    "This report is for reference only and is not investment advice; investors bear sole responsibility for "
    "their investment decisions."
)
MODULE_HEADINGS = {
    1: "Executive Signal Matrix",
    2: "Quantitative & ML Factor Signal",
    3: "LLM Unstructured Intelligence",
    4: "Execution & Risk Management",
}

# signal -> (css class, Korean label)
SIGNAL_META: dict[Signal, tuple[str, str]] = {
    Signal.STRONG_BUY: ("sb", "강력 매수"),
    Signal.BUY: ("buy", "매수"),
    Signal.HOLD: ("hold", "보유·관망"),
    Signal.SELL: ("sell", "매도"),
}
SEVERITY_META = {
    "critical": ("치명", 0),
    "high": ("높음", 1),
    "medium": ("중간", 2),
    "low": ("낮음", 3),
}
SEVERITY_EN = {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low"}
GUIDANCE_KO = {
    "raised": "상향",
    "maintained": "유지",
    "lowered": "하향",
    "none": "미제시",
    "unknown": "확인불가",
}
GUIDANCE_EN = {
    "raised": "raised",
    "maintained": "maintained",
    "lowered": "lowered",
    "none": "none given",
    "unknown": "unclear",
}
# categorical palette (fixed order, validated default) for the allocation bar
SERIES_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
FACTOR_ORDER = ["momentum", "value", "volatility", "quality", "trend"]
FACTOR_LABELS = {
    "momentum": "모멘텀",
    "value": "밸류에이션",
    "volatility": "변동성(안정성)",
    "quality": "퀄리티",
    "trend": "추세",
}
FACTOR_LABELS_EN = {
    "momentum": "Momentum",
    "value": "Value",
    "volatility": "Volatility (stability)",
    "quality": "Quality",
    "trend": "Trend",
}
# portfolio_risk labels: quantcio.fusion.risk.PORTFOLIO_RISK_LABELS is the source of truth (it computes the
# numbers); these local labels are the fallback used when that module cannot be imported, so the report (and
# `doctor`'s sample PDF) still renders when fusion is broken. VaR / CVaR / P(loss) are normal approximations
# in fusion.risk.portfolio_risk, and every label for them says so ("정규근사").
LOCAL_PORTFOLIO_RISK_LABELS: dict[str, str] = {
    "gross": "투자 비중 합계",
    "n_positions": "편입 종목 수",
    "ex_ante_vol": "사전 변동성(연율)",
    "expected_return": "기대수익률(기간, 근사)",
    "var95": "VaR95(기간, 정규근사)",
    "cvar95": "CVaR95(기간, 정규근사)",
    "p_loss": "손실확률(기간, 정규근사)",
}
LOCAL_PORTFOLIO_RISK_LABELS_EN: dict[str, str] = {
    "gross": "Gross exposure",
    "n_positions": "Positions",
    "ex_ante_vol": "Ex-ante vol (annualised)",
    "expected_return": "Expected return (horizon, approx.)",
    "var95": "VaR95 (horizon, normal approx.)",
    "cvar95": "CVaR95 (horizon, normal approx.)",
    "p_loss": "P(loss) (horizon, normal approx.)",
}
# legacy / alternative key spellings -> canonical key
_RISK_ALIASES = {
    "vol": "ex_ante_vol",
    "vol_annual": "ex_ante_vol",
    "var_95": "var95",
    "prob_loss": "p_loss",
}
# canonical key -> display kind (pct = unsigned share, spct = signed return, prob, int)
_RISK_KINDS = {
    "gross": "pct",
    "n_positions": "int",
    "ex_ante_vol": "pct",
    "expected_return": "spct",
    "var95": "spct",
    "cvar95": "spct",
    "p_loss": "prob",
}
_NORMAL_APPROX_KEYS = ("var95", "cvar95", "p_loss")
NORMAL_APPROX = "정규근사"
NORMAL_APPROX_EN = "normal approx."


# ============================================================================ formatting helpers


def _missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        return not math.isfinite(float(v))
    except (TypeError, ValueError):
        return True


def _minus(s: str) -> str:
    return s.replace("-", MINUS)


def fmt_pct(v: Any, digits: int = 1, signed: bool = True) -> str:
    """0.0523 -> '+5.2%' (typographic minus); None/NaN -> '—'; a rounded zero is unsigned."""
    if _missing(v):
        return MISSING
    x = float(v) * 100
    if round(x, digits) == 0:
        return f"{0:.{digits}f}%"
    return _minus(f"{x:+.{digits}f}%" if signed else f"{x:.{digits}f}%")


def fmt_prob(v: Any, digits: int = 0) -> str:
    """Unsigned share/probability: 0.543 -> '54%'; a negative input (e.g. cash < 0) keeps a typographic minus."""
    return MISSING if _missing(v) else _minus(f"{float(v) * 100:.{digits}f}%")


def fmt_num(v: Any, digits: int = 2, signed: bool = False) -> str:
    if _missing(v):
        return MISSING
    x = float(v)
    if round(x, digits) == 0:
        return f"{0:.{digits}f}"
    return _minus(f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}")


def fmt_int(v: Any) -> str:
    return MISSING if _missing(v) else _minus(f"{float(v):,.0f}")


def fmt_price(v: Any, currency: str, lang: Any = None) -> str:
    """KRW -> '286,500원' (no symbol, no decimals; English `lang="en"`: '₩286,500'); USD -> '$337.02';
    other -> '1,234.50 EUR'.

    Same format as `quantcio.fusion.risk.fmt_price`, so the prices inside upstream trigger/action strings
    and the prices in the report tables read identically.
    """
    if _missing(v):
        return MISSING
    x = float(v)
    if currency == "KRW" and is_en(lang):  # sign first, like fusion.risk.format_money: '−₩1,500'
        body = f"₩{abs(x):,.0f}"
        return (MINUS if x < 0 and any(c in "123456789" for c in body) else "") + body
    if currency == "KRW":
        return _minus(f"{x:,.0f}원")
    if currency == "USD":
        return _minus(f"${x:,.2f}")
    return _minus(f"{x:,.2f} {currency}")


def fmt_money_compact(v: Any, currency: str, lang: Any = None) -> str:
    """1e9 KRW -> '10억원'; 3.25e8 KRW -> '3.2억원'; 3.2e6 USD -> '$3.2M'. English: 1e9 KRW -> '₩1.0B'."""
    if _missing(v):
        return MISSING
    x = float(v)
    if currency == "KRW" and is_en(lang):
        sign = MINUS if x < 0 else ""
        for unit, name in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(x) >= unit:
                return f"{sign}₩{abs(x) / unit:,.1f}{name}"
        return f"{sign}₩{abs(x):,.0f}" if round(abs(x)) else "₩0"
    if currency == "KRW":
        for unit, name in ((1e12, "조"), (1e8, "억"), (1e4, "만")):
            if abs(x) >= unit:
                val = x / unit
                txt = f"{val:,.1f}".rstrip("0").rstrip(".") if abs(val) < 100 else f"{val:,.0f}"
                return _minus(f"{txt}{name}원")
        return _minus(f"{x:,.0f}원")
    sym = "$" if currency == "USD" else ""
    for unit, name in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= unit:
            return _minus(f"{sym}{x / unit:,.1f}{name}")
    return _minus(f"{sym}{x:,.0f}")


def _signed_share(v: float | None, lim: float = 1.0) -> dict[str, float]:
    """Diverging bar geometry around a centre line for a value in [-lim, lim] (percent of width)."""
    if _missing(v):
        return {"left": 50.0, "width": 0.0, "neg": False, "missing": True}
    x = max(-lim, min(lim, float(v))) / lim
    return {
        "left": 50.0 if x >= 0 else 50.0 + 50.0 * x,
        "width": abs(x) * 50.0,
        "neg": x < 0,
        "missing": False,
    }


def _round_to_100(probs: list[float]) -> list[int]:
    """Largest-remainder rounding so displayed integer percentages sum to exactly 100."""
    raw = [max(0.0, p) * 100 for p in probs]
    total = sum(raw)
    if total <= 0:
        return [0] * len(raw)
    raw = [r * 100 / total for r in raw]
    floors = [int(math.floor(r)) for r in raw]
    rem = 100 - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[:rem]:
        floors[i] += 1
    return floors


def _circled(i: int) -> str:
    return "①②③④⑤⑥⑦⑧⑨⑩"[i] if 0 <= i < 10 else f"({i + 1})"


def _leaf_text(path: Iterable[str]) -> str:
    parts = [p.replace("<=", "≤").replace(">=", "≥").replace(" -", f" {MINUS}") for p in path]
    return " › ".join(parts)


@functools.lru_cache(maxsize=2)
def portfolio_risk_labels(lang: str = "ko") -> dict[str, str]:
    """portfolio_risk key -> label, from `quantcio.fusion.risk.PORTFOLIO_RISK_LABELS` (English:
    `PORTFOLIO_RISK_LABELS_EN`) when importable.

    Falls back to LOCAL_PORTFOLIO_RISK_LABELS(_EN) (logged at WARNING, once) when fusion cannot be imported or
    its table has an unexpected shape. The labels of the normal-approximation keys (VaR / CVaR / P(loss))
    always carry "근사" ("approx"): a fusion label that ever drops it gets "(정규근사)" ("(normal approx.)")
    appended.
    """
    en = lang == "en"
    labels = dict(LOCAL_PORTFOLIO_RISK_LABELS_EN if en else LOCAL_PORTFOLIO_RISK_LABELS)
    try:
        if en:
            from quantcio.fusion.risk import PORTFOLIO_RISK_LABELS_EN as UPSTREAM_LABELS
        else:
            from quantcio.fusion.risk import PORTFOLIO_RISK_LABELS as UPSTREAM_LABELS
    except Exception as exc:  # noqa: BLE001 - the report must render even when fusion is broken
        log.warning(
            "quantcio.fusion.risk.PORTFOLIO_RISK_LABELS를 불러오지 못해 리포트 기본 라벨을 사용합니다: %s: %s",
            type(exc).__name__,
            exc,
        )
        return labels
    if not isinstance(UPSTREAM_LABELS, dict):
        log.warning(
            "fusion.risk.PORTFOLIO_RISK_LABELS 형식 이상(%s) — 리포트 기본 라벨 사용",
            type(UPSTREAM_LABELS).__name__,
        )
        return labels
    for key, val in UPSTREAM_LABELS.items():
        label = val[0] if isinstance(val, tuple | list) and val else val
        if isinstance(label, str) and label.strip():
            labels[str(key)] = label.strip()
        else:
            log.warning("fusion.risk.PORTFOLIO_RISK_LABELS[%r] 형식 이상 — 기본 라벨 사용", key)
    for key in _NORMAL_APPROX_KEYS:
        if en and "approx" not in labels[key]:
            labels[key] = f"{labels[key]} ({NORMAL_APPROX_EN})"
        elif not en and "근사" not in labels[key]:
            labels[key] = f"{labels[key]}({NORMAL_APPROX})"
    return labels


_LABEL_PAREN = re.compile(r"^(?P<base>.*?)\s*\((?P<paren>[^()]*)\)\s*$")


def _risk_label(key: str, horizon: int, en: bool = False) -> tuple[str, str]:
    """(base label, parenthetical) with the upstream placeholder "기간" ("horizon") made concrete ("20거래일",
    "20 trading days")."""
    label = portfolio_risk_labels("en" if en else "ko").get(key, key)
    m = _LABEL_PAREN.match(label)
    if not m:
        return label, ""
    if en:
        paren = ", ".join(
            f"{horizon} trading days" if part.strip() == "horizon" else part.strip()
            for part in m.group("paren").split(",")
        )
    else:
        paren = ", ".join(
            f"{horizon}거래일" if part.strip() == "기간" else part.strip() for part in m.group("paren").split(",")
        )
    return m.group("base").strip(), paren


# next-earnings chip: shown when the (exchange-local) earnings date is within this many calendar days
EARNINGS_CHIP_DAYS = 14
EARNINGS_HOT_DAYS = 3  # fusion/risk.py's "실적발표 D-3 이내 신규진입 자제" window -> emphasised chip
_MARKET_TZ = {"KR": ZoneInfo("Asia/Seoul"), "US": ZoneInfo("America/New_York")}


def _earnings_view(t: TickerReport, generated_at: datetime, en: bool = False) -> dict[str, Any] | None:
    """'실적 D-n' chip data when Fundamentals.next_earnings_date is 0..EARNINGS_CHIP_DAYS days ahead.

    D-n counts calendar days from the later of the price date (as_of) and the report's generation date in
    the market's own timezone, i.e. what the reader sees when opening the report.
    """
    d = t.fundamentals.next_earnings_date
    if d is None:
        return None
    if isinstance(d, datetime):
        d = d.date()
    if not isinstance(d, date):
        return None
    tz = _MARKET_TZ.get(t.instrument.market.value, KST)
    ref = max(t.as_of, generated_at.astimezone(tz).date())
    n = (d - ref).days
    if not 0 <= n <= EARNINGS_CHIP_DAYS:
        return None
    est = bool(t.fundamentals.next_earnings_is_estimate)
    word = "Earnings" if en else "실적"
    return {
        "txt": f"{word} D-day" if n == 0 else f"{word} D-{n}",
        "date": f"{d.month}/{d.day}" + ((" est." if en else " 추정") if est else ""),
        "days": n,
        "hot": n <= EARNINGS_HOT_DAYS,
        "estimate": est,
    }


# diagnostics lines: HEAD = the model summary line (n / leaves / root p) - quant/rsmt.py prints it AFTER the
# tree, so it is pinned to the top instead of being cut by the line cap; KEEP = always shown even beyond the
# cap (health lines "[저하] ..."/"[정보] ...", failures, dropped rows, missing CV); HOT = highlighted as a warning
_HEAD_LINE = re.compile(r"^\s*RSMT-Q\b")
_KEEP_LINE = re.compile(
    r"⚠|저하|degrad|경고|warn|실패|fail|붕괴|collapse|누락|missing|오류|error|제외|drop|^\s*\[정보\]|결과 없음|미실행"
    r"|^\s*\[info\]|no result|not run|excluded",
    re.I,
)
_HOT_LINE = re.compile(
    r"⚠|저하|degrad|경고|warn|오류|error|실패\s*[1-9]|fail\w*\s*[1-9]|붕괴(?!\s*검사\s*통과)|collapse(?!\s*check\s*pass)",
    re.I,
)


# ============================================================================ view model


def _signal_view(sig: Signal) -> dict[str, str]:
    cls, ko = SIGNAL_META[sig]
    return {"value": sig.value, "cls": cls, "ko": ko}


_MODEL_WARNING = re.compile(r"RSMT|저하|degrad|붕괴|collapse|\bML\b", re.I)
_LLM_WARNING = re.compile(r"^\s*LLM\b")


_LLM_FAILURE = re.compile(r"오류|실패|거부|누락|error|fail|refus|max_tokens|timeout|invalid", re.I)


def _warning_kind(text: str) -> str:
    """Classify an upstream data warning for the page-1 alert chips (model / llm error / llm notice / data)."""
    if _LLM_WARNING.search(text):
        return "llm" if _LLM_FAILURE.search(text) else "notice"
    if _MODEL_WARNING.search(text):
        return "model"
    return "warning"


# fusion/signal.py's missing-LLM-pillar override when the LLM did not run ("LLM 분석 미실행 — LLM 필러 제외 후 ...",
# English "LLM analysis not run — ...")
_LLM_SKIP_OVERRIDE = re.compile(r"^\s*(?:LLM 분석 미실행|LLM analysis not run)")
# fusion/signal.py's factor-pillar blend note: informational (a normal operating mode, not a guardrail), shown in
# the Module 2 factor-model table instead of a warning chip
_FACTOR_BLEND_NOTE = re.compile(r"^\s*(?:팩터 필러 블렌드:|Factor pillar blend:)")
# fusion/signal.py's ML-pillar blend note (RSMT-Q distribution + stock-selection model): informational too
_ML_BLEND_NOTE = re.compile(r"^\s*(?:ML 필러 블렌드:|ML pillar blend:)")


def _is_blend_note(text: str) -> bool:
    """Any informational pillar-blend note (never a warning chip)."""
    return bool(_FACTOR_BLEND_NOTE.match(text) or _ML_BLEND_NOTE.match(text))


def _is_factor_blend_note(text: str) -> bool:
    return bool(_FACTOR_BLEND_NOTE.match(text))


def _selection_view(fc: Any, en: bool) -> dict[str, Any] | None:
    """The stock-selection model's cells of the Module 2 quant table (None without a forecast)."""
    if fc is None:
        return None
    confirmed = bool((fc.gate or {}).get("confirmed"))
    money = getattr(fc, "oos_primary", None) is not None  # protocol v4: the money metric leads (docs/MODELING.md 79)
    if money:
        auc = fmt_pct(fc.oos_primary * 12, 1)
        hold = MISSING if fc.holdout_primary is None else fmt_pct(fc.holdout_primary * 12, 1)
    else:
        auc = MISSING if fc.oos_auc is None else f"{fc.oos_auc:.3f}"
        hold = MISSING if fc.holdout_auc is None else f"{fc.holdout_auc:.3f}"
    return {
        "hold": hold,
        "money": money,
        "pct": f"{fc.percentile:.0%}",
        "pct_w": round(100.0 * float(fc.percentile), 1),
        "mu": fmt_pct(fc.mu_excess, 2),
        "mu_neg": fc.mu_excess < 0,
        "auc": auc,
        "confirmed": confirmed,
        "skill": bool(fc.skill_ok),
        "model": fc.model,
        "outside": not fc.in_universe,
    }


def _direction_models(report: PortfolioReport, en: bool) -> str:
    """'KR 엑스트라 트리 [et] AUC 0.543 · 홀드아웃 0.543 · 스킬 확인 / US ...' - the stock-selection model each market's
    ML pillar direction actually used, with its walk-forward and holdout performance (from the tickers' forecasts)."""
    seen: dict[str, Any] = {}
    for t in report.tickers:
        fc = getattr(t.quant, "excess_model", None)
        if fc is not None and t.instrument.market.value not in seen:
            seen[t.instrument.market.value] = fc
    parts = []
    for mk in sorted(seen, key=lambda m: (m != "KR", m)):
        fc = seen[mk]
        label = (fc.model_label or fc.model).split("(")[0].strip()
        auc = "n/a" if fc.oos_auc is None else f"{fc.oos_auc:.3f}"
        hold = "n/a" if fc.holdout_auc is None else f"{fc.holdout_auc:.3f}"
        op = getattr(fc, "oos_primary", None)
        if op is not None:  # protocol v4: net excess of the top quintile (annualised) leads, the AUC follows
            mh = MISSING if fc.holdout_primary is None else fmt_pct(fc.holdout_primary * 12, 1)
            money = (
                f"net excess {fmt_pct(op * 12, 1)}/yr · holdout {mh} · "
                if en
                else f"순초과 연 {fmt_pct(op * 12, 1)} · 홀드아웃 {mh} · "
            )
        else:
            money = ""
        if en:
            skill = "skill confirmed" if fc.skill_ok else "skill not confirmed (shrunk)"
            hold_txt = "" if money else f" · holdout {hold}"
            parts.append(f"{mk} {label} [{fc.model}] {money}AUC {auc}{hold_txt} · {skill}")
        else:
            skill = "스킬 확인" if fc.skill_ok else "스킬 미확인(축소 반영)"
            hold_txt = "" if money else f" · 홀드아웃 {hold}"
            parts.append(f"{mk} {label} [{fc.model}] {money}AUC {auc}{hold_txt} · {skill}")
    return " / ".join(parts)


def _factor_model_names(report: PortfolioReport, en: bool) -> str:
    """' · selected: KR ic_weighted_raw (auto) · US rsmt_tree (fixed)' from the cached backtest summaries; a tag
    shared by every market is printed once (' · selected (auto): KR ic_weighted · US ic_weighted') so the subtitle
    stays on one line with wider fonts too (NanumGothic on Linux)."""
    parts: list[tuple[str, str]] = []
    for market in sorted(report.factor_backtest or {}, key=lambda m: (m != "KR", m)):
        bt = (report.factor_backtest or {}).get(market)
        model = bt.get("model") if isinstance(bt, dict) else None
        if not isinstance(model, dict) or not model.get("method"):
            continue
        mode = model.get("selection_mode")
        tag = "" if not mode else ("auto" if en else "자동 선정") if mode == "auto" else ("fixed" if en else "고정")
        name = f"{market} {model['method']}"
        if model["method"] == "rsmt_tree":  # an IC-weight tree over market regimes - not the RSMT-Q engine
            name += " (market-regime IC tree, not RSMT-Q)" if en else "(시장 국면별 IC 가중 트리 — RSMT-Q와 별개)"
        parts.append((name, tag))
    if not parts:
        return ""
    tags = {tag for _, tag in parts}
    if len(tags) == 1 and (tag := tags.pop()):
        head = f" · selected ({tag}): " if en else f" · {tag} 모델: "
        return head + " · ".join(name for name, _ in parts)
    names = [f"{name} ({tag})" if tag else name for name, tag in parts]
    return (" · selected: " if en else " · 선정 모델: ") + " · ".join(names)


def _run_skip_override(text: str, run_llm_skipped: bool) -> bool:
    """True for the per-ticker 'LLM not run' override of a run-wide --no-llm report (stated once at report level)."""
    return run_llm_skipped and bool(_LLM_SKIP_OVERRIDE.match(text))


def _alerts_for(t: TickerReport, run_llm_skipped: bool = False, en: bool = False) -> list[dict[str, str]]:
    """All visible warning items for one ticker (overrides, data warnings, degraded model, LLM issues).

    `run_llm_skipped` (every ticker ran with --no-llm): the skip is a run-level choice that the header chip
    and one report note state; repeating it per ticker (an override chip + an amber chip each) buried the
    real warnings (live run: 12 of 14 page-1 alerts were the same LLM-skipped text). Only those two
    identical items are left out; every other override, warning and LLM issue is still listed.
    """
    items: list[dict[str, str]] = []

    def add(kind: str, text: str) -> None:
        # upstream strings are only de-duplicated on exact match - nothing reported upstream is dropped;
        # the structural items synthesised below check for an existing mention before calling add()
        text = text.strip()
        if text and not any(text == it["text"] for it in items):
            items.append({"kind": kind, "text": text})

    crit_risks = [r for r in (t.llm.filings.risk_factors if t.llm.filings else []) if r.severity == "critical"]
    no_reason = "reason not given" if en else "사유 미기재"
    if t.llm.critic and t.llm.critic.critical_flag:
        head = "Critical flag" if en else "치명 플래그"
        add("critical", f"{head}: {t.llm.critic.critical_reason or no_reason}")
    for r in crit_risks:
        add("critical", f"{'Critical risk' if en else '치명 리스크'}: {r.title}")
    for o in t.executive.overrides:
        if not _run_skip_override(o, run_llm_skipped) and not _is_blend_note(o):
            add("override", o)
    for w in t.warnings:
        add(_warning_kind(w), w)
    d = t.quant.distribution
    if d is None:
        add("model", "No RSMT-Q distribution — ML pillar dropped" if en else "RSMT-Q 분포 예측 없음 — ML 필러 제외")
    elif d.degraded:
        # every reason must be visible: an upstream warning that names only some of them does not cover the rest
        unmentioned = [r for r in d.degraded_reasons if not any(r in it["text"] for it in items)]
        if unmentioned or not d.degraded_reasons:
            head = "RSMT-Q degraded mode" if en else "RSMT-Q 저하 모드"
            add("model", f"{head}: {', '.join(unmentioned) or no_reason}")
    if t.llm.skipped and not run_llm_skipped:  # a user choice, not a failure: amber, like the notices
        add(
            "notice",
            "LLM analysis skipped (--no-llm) — quant signals only"
            if en
            else "LLM 분석 생략(--no-llm) — 정량 신호만 반영",
        )
    for e in t.llm.errors:  # hard failures: red
        if not any(e in it["text"] for it in items):
            add("llm", f"{'LLM error' if en else 'LLM 오류'}: {e}")
    # degraded-but-served LLM paths (fallback model, clamps, dropped ungrounded items): amber, summarised on
    # page 1 (every notice is listed in full on its Module 3 card)
    notices = [n.strip() for n in t.llm.notices if n and n.strip()]
    if len(notices) == 1:
        add("notice", f"{'LLM notice' if en else 'LLM 알림'}: {notices[0]}")
    elif notices:
        add(
            "notice",
            f"{len(notices)} LLM notices: {notices[0]} + {len(notices) - 1} more (see Module 3)"
            if en
            else f"LLM 알림 {len(notices)}건: {notices[0]} 외 {len(notices) - 1}건(모듈 3 참조)",
        )
    return items


def _factor_cells(t: TickerReport, en: bool = False) -> list[dict[str, Any]]:
    by_name = {f.name: f for f in t.quant.factors}
    cells = []
    for name in FACTOR_ORDER:
        f = by_name.get(name)
        score = None if f is None else f.score
        cells.append(
            {
                # English: always the English name (an upstream Korean FactorScore.label must not leak in)
                "label": FACTOR_LABELS_EN[name] if en else (f.label if f else FACTOR_LABELS[name]),
                "txt": MISSING if _missing(score) else f"{float(score):.0f}",
                "w": 0.0 if _missing(score) else max(0.0, min(100.0, float(score))),
                "missing": _missing(score),
                "rationale": f.rationale if f else "",
            }
        )
    return cells


def _news_view(n: NewsSentiment | None, level: int, en: bool = False) -> dict[str, Any] | None:
    if n is None:
        return None
    trend = None
    if not _missing(n.trend_7d):
        tv = float(n.trend_7d) * 100
        arrow = "▲" if tv > 0.5 else "▼" if tv < -0.5 else "▬"
        trend = {
            "txt": f"{arrow} {fmt_num(tv, 0, signed=True)}",
            "neg": tv < -0.5,
            "pos": tv > 0.5,
        }
    lines: list[dict[str, str]] = []
    if level < 3:  # article-level lines (dropped at the highest compaction)
        if n.key_positives:
            lines.append({"sign": "+", "cls": "pos", "text": " · ".join(n.key_positives[:2])})
        if n.key_negatives:
            lines.append({"sign": MINUS, "cls": "neg", "text": " · ".join(n.key_negatives[:2])})
        if n.key_events and level < 1:
            lines.append({"sign": "•", "cls": "evt", "text": " · ".join(n.key_events[:2])})
    return {
        "score": n.score_100,
        "score_txt": fmt_num(n.score_100, 0, signed=True),
        "gauge": _signed_share(n.score_100 / 100.0),
        "counts": f"{n.n_used}/{n.n_articles}" + ("" if en else "건"),
        "trend": trend,
        "lines": lines,
    }


def _filings_view(f: FilingIntelligence | None, level: int, en: bool = False) -> dict[str, Any] | None:
    if f is None:
        return None
    n_bullets = 3 if level < 2 else 2
    return {
        "guidance": (GUIDANCE_EN if en else GUIDANCE_KO).get(f.guidance, f.guidance),
        "tone": fmt_num(f.tone, 2, signed=True),
        "bullets": f.summary[:n_bullets],
        "highlights": " · ".join(f.earnings_highlights[:2]) if level < 2 else "",
        "n_sources": len(f.sources),
    }


def _risks_view(risks: list[RiskFactor], level: int, en: bool = False) -> list[dict[str, str]]:
    ordered = sorted(risks, key=lambda r: SEVERITY_META.get(r.severity, ("", 9))[1])
    cap = 3 if level < 2 else 2
    shown = ordered[:cap]
    # a critical risk is a warning: never trimmed away by compaction
    shown += [r for r in ordered[cap:] if r.severity == "critical"]
    return [
        {
            "sev": r.severity,
            "sev_ko": SEVERITY_EN.get(r.severity, r.severity)
            if en
            else SEVERITY_META.get(r.severity, (r.severity, 9))[0],
            "title": r.title,
            "detail": r.detail,
        }
        for r in shown
    ]


def _market_label(m: str, en: bool = False) -> str:
    if en:
        return {"KR": "Korea (KR)", "US": "United States (US)"}.get(m, m)
    return {"KR": "한국 (KR)", "US": "미국 (US)"}.get(m, m)


# Robust skill gate (rsmt.robust_cv_folds): the CV skill gate is re-run for several fold layouts and a MAJORITY
# must pass. Read from DistributionForecast.cv_metrics:
#   gate_folds_n     layouts evaluated            gate_folds_pass  layouts whose skill gate passed
#   gate_folds_need  passes required (optional, default = majority n//2+1)
#   gate_folds_<k>   1.0 pass / 0.0 fail / NaN not run, per k-fold layout (optional, e.g. gate_folds_3)
# (a few alias spellings are accepted; anything unparseable is shown as such, never dropped)
_GATE_PREFIX = "gate_folds"
_GATE_LAYOUT = re.compile(r"^gate_folds_k?(\d+)(?:_(?:pass|passed|ok))?$")
_GATE_N_KEYS = ("gate_folds_n", "gate_folds_total", "gate_folds_tested", "gate_folds_layouts")
_GATE_PASS_KEYS = ("gate_folds_pass", "gate_folds_passed", "gate_folds_ok")
_GATE_NEED_KEYS = ("gate_folds_need", "gate_folds_required", "gate_folds_min")


# "p=6.2e-01": Chrome breaks a line after the hyphen-minus of an exponent ("p=6.2e-" / "01" on the real
# report); the typographic minus used for every other number in the report is not a break opportunity
_SCI_EXP_MINUS = re.compile(r"(?<=\de)-(?=\d)")


def _sci_minus(line: str) -> str:
    return _SCI_EXP_MINUS.sub(MINUS, line)


def _first_metric(cv: dict[str, float], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k in cv and not _missing(cv[k]):
            return float(cv[k])
    return None


def _trust_view(market: str, d0: DistributionForecast | None, en: bool = False) -> dict[str, Any]:
    """Module 2 '모델 신뢰도' line: the robust (multi-layout) CV skill gate in plain Korean.

    e.g. "3개 폴드 구성 중 1개만 통과 → 스킬 미확인" with per-layout marks "3폴드 ✗ · 4폴드 ✓ · 5폴드 ✗".
    cls: ok (skill confirmed) / no (not confirmed) / na (no information). Without gate_folds_* keys the line
    still says what is known (single-layout CV result) and that the robustness check is missing.
    """
    if d0 is None:
        txt = "No RSMT-Q distribution — no validation info" if en else "RSMT-Q 분포 없음 — 검증 정보 없음"
        return {"text": txt, "detail": "", "cls": "na"}
    cv = d0.cv_metrics or {}
    if not cv:
        return {
            "text": "No CV result → skill unconfirmed" if en else "CV 결과 없음 → 스킬 미확인",
            "detail": "",
            "cls": "no",
        }
    if not any(k.startswith(_GATE_PREFIX) for k in cv):
        n_folds = cv.get("n_folds")
        if en:
            single = f"single {int(n_folds)}-fold CV" if not _missing(n_folds) else "single fold-layout CV"
            missing_gate = "no per-layout gate record"
            if d0.skill_ok:
                return {"text": f"passed {single} only → robustness unverified", "detail": missing_gate, "cls": "no"}
            return {"text": f"{single} failed → skill unconfirmed", "detail": missing_gate, "cls": "no"}
        single = f"단일 {int(n_folds)}폴드 CV" if not _missing(n_folds) else "단일 폴드 구성 CV"
        missing_gate = "폴드 구성별 게이트 기록 없음"
        if d0.skill_ok:  # passed one layout only: not contradicted, but not shown to be robust either
            return {"text": f"{single}만 통과 → 견고성 미검증", "detail": missing_gate, "cls": "no"}
        return {"text": f"{single} 미달 → 스킬 미확인", "detail": missing_gate, "cls": "no"}

    layouts: dict[int, bool | None] = {}
    for k, v in cv.items():
        m = _GATE_LAYOUT.match(k)
        if m:
            layouts[int(m.group(1))] = None if _missing(v) else float(v) > 0.5
    n = _first_metric(cv, _GATE_N_KEYS)
    passed = _first_metric(cv, _GATE_PASS_KEYS)
    if n is None and layouts:
        n = float(len(layouts))
    counted = sum(1 for ok in layouts.values() if ok)
    if passed is None and layouts:
        passed = float(counted)
    elif passed is not None and layouts and int(passed) != counted:
        log.warning(
            "%s 견고성 게이트: gate_folds 통과 수(%s)와 폴드 구성별 결과(%d)가 다름 — 통과 수 기준으로 표시",
            market,
            passed,
            counted,
        )
    if en:
        detail = " · ".join(
            f"{k}-fold {'✓' if ok else '✗' if ok is False else 'not run'}" for k, ok in sorted(layouts.items())
        )
    else:
        detail = " · ".join(
            f"{k}폴드 {'✓' if ok else '✗' if ok is False else '미실행'}" for k, ok in sorted(layouts.items())
        )
    if n is None or passed is None or n < 1:
        keys = ", ".join(sorted(k for k in cv if k.startswith(_GATE_PREFIX)))
        log.warning("%s 견고성 게이트 지표 해석 불가: %s", market, keys)
        text = (
            f"fold-layout robustness result unreadable ({keys}) → skill unconfirmed"
            if en
            else f"폴드 구성 견고성 결과 해석 불가({keys}) → 스킬 미확인"
        )
        return {"text": text, "detail": detail, "cls": "no"}
    n_i, p_i = int(round(n)), int(round(passed))
    need = _first_metric(cv, _GATE_NEED_KEYS)
    need_i = int(round(need)) if need is not None else n_i // 2 + 1
    confirmed = p_i >= need_i
    if en:
        if p_i >= n_i:
            text = f"all {n_i} fold layouts passed → skill confirmed"
        elif confirmed:
            text = f"{p_i} of {n_i} fold layouts passed → skill confirmed"
        elif p_i <= 0:
            text = f"none of {n_i} fold layouts passed → skill unconfirmed"
        else:
            text = f"only {p_i} of {n_i} fold layouts passed → skill unconfirmed"
        if need is not None and need_i != n_i // 2 + 1:
            text += f" ({need_i} required)"
    elif p_i >= n_i:
        text = f"{n_i}개 폴드 구성 모두 통과 → 스킬 확인"
    elif confirmed:
        text = f"{n_i}개 폴드 구성 중 {p_i}개 통과 → 스킬 확인"
    elif p_i <= 0:
        text = f"{n_i}개 폴드 구성 모두 미통과 → 스킬 미확인"
    else:
        text = f"{n_i}개 폴드 구성 중 {p_i}개만 통과 → 스킬 미확인"
    if not en and need is not None and need_i != n_i // 2 + 1:
        text += f" (필요 {need_i}개)"
    if confirmed != bool(d0.skill_ok):  # never let the two verdicts contradict silently
        if en:
            text += f" · final skill verdict {'pass' if d0.skill_ok else 'fail'}"
        else:
            text += f" · 최종 스킬 판정 {'통과' if d0.skill_ok else '미달'}"
        log.warning("%s 견고성 게이트(%d/%d)와 skill_ok=%s 불일치", market, p_i, n_i, d0.skill_ok)
    return {"text": text, "detail": detail, "cls": "ok" if confirmed and d0.skill_ok else "no"}


def _selection_line(fc: Any, en: bool) -> dict[str, str]:
    """Diagnostics card's first line: the stock-selection model that sets the ML direction (the served, best model
    by the market's selection rule) with its walk-forward / holdout AUC - RSMT-Q below it is the risk engine only."""
    raw = fc.model_label or fc.model
    name = raw.split("·")[0].split("(")[0].strip()
    override = "사용자 지정" in raw or "user override" in raw
    g = fc.gate or {}
    auc, lo = fmt_num(fc.oos_auc, 3), fmt_num(fc.oos_auc_lo, 3)
    base, hold = fmt_num(g.get("baseline_auc"), 3), fmt_num(fc.holdout_auc, 3)
    op = getattr(fc, "oos_primary", None)
    if op is not None:  # protocol v4: the money metric (annualised net excess of the top quintile) leads

        def ann(x: Any) -> str:
            return MISSING if x is None else fmt_pct(float(x) * 12, 1)

        auc = ann(op)
        lo = ann(fc.oos_primary_lo)
        base = ann(g.get("baseline"))
        hold = f"{ann(fc.holdout_primary)} · AUC {fmt_num(fc.oos_auc, 3)}"
    if en:
        text = f"{name} [{fc.model}]" + (" · user override (best holdout)" if override else "")
        verdict = (
            ("gate passed · XSH confirmed" if g.get("confirmed") else "gate passed · awaiting confirmation")
            if fc.skill_ok
            else "gate failed → expected excess shrunk"
        )
        what = "net excess/yr" if op is not None else "AUC"
        detail = f"OOS {what} {auc} (lower {lo} · momentum {base}) · holdout {hold} · {verdict}"
    else:
        text = f"{name} [{fc.model}]" + (" · 사용자 지정(홀드아웃 1위)" if override else "")
        verdict = (
            ("게이트 통과 · XSH 확인" if g.get("confirmed") else "게이트 통과 · 확인 대기")
            if fc.skill_ok
            else "게이트 미달 → 초과수익 기대치 축소"
        )
        what = "순초과(연)" if op is not None else "AUC"
        detail = f"표본 외 {what} {auc}(하한 {lo} · 모멘텀 {base}) · 홀드아웃 {hold} · {verdict}"
    return {"text": text, "detail": detail, "cls": "ok" if fc.skill_ok else "no"}


def _diagnostics_view(report: PortfolioReport, level: int, en: bool = False) -> list[dict[str, Any]]:
    markets: list[str] = []
    for t in report.tickers:
        if t.instrument.market.value not in markets:
            markets.append(t.instrument.market.value)
    for m in report.model_diagnostics:
        if m not in markets:
            markets.append(m)
    max_lines = 7 if level < 2 else 5  # summary line + first tree lines; alert/health lines always kept
    out = []
    for m in markets:
        dists = [
            t.quant.distribution
            for t in report.tickers
            if t.instrument.market.value == m and t.quant.distribution is not None
        ]
        fcs = [
            t.quant.excess_model
            for t in report.tickers
            if t.instrument.market.value == m and getattr(t.quant, "excess_model", None) is not None
        ]
        sel_line = _selection_line(fcs[0], en) if fcs else None
        d0 = dists[0] if dists else None
        cv = d0.cv_metrics if d0 else {}
        metrics = []
        for label, key, digits, better in (
            ("AUC", "auc", 3, "high"),
            ("LogLoss", "logloss", 4, "low"),
            ("CRPS", "crps", 4, "low"),
            ("Cover80", "cover80", 2, "target"),
        ):
            v, b = cv.get(key), cv.get(f"base_{key}")
            good = None
            if not _missing(v) and not _missing(b):
                good = (v > b) if better == "high" else (v < b) if better == "low" else (abs(v - 0.8) <= abs(b - 0.8))
            metrics.append(
                {
                    "label": label,
                    "v": fmt_num(v, digits),
                    "b": fmt_num(b, digits),
                    "good": good,
                }
            )
        offs = d0.offsets if d0 else {}
        params = (
            f"γ {fmt_num(offs.get('gamma'), 4, signed=True)} · η {fmt_num(offs.get('eta'), 2)} · "
            f"δ {fmt_num(d0.direction_edge if d0 else None, 2, signed=True)} · "
            f"δrel {fmt_num(d0.relative_edge if d0 else None, 2, signed=True)}"
        )
        lines = [_sci_minus(ln) for ln in report.model_diagnostics.get(m, [])]
        head = [ln for ln in lines if _HEAD_LINE.match(ln)]
        body = [ln for ln in lines if not _HEAD_LINE.match(ln)]
        budget = max_lines - min(1, len(head)) - (1 if sel_line else 0)  # the selection line takes a tree line's room
        picked = head + body[:budget] + [ln for ln in body[budget:] if _KEEP_LINE.search(ln)]
        shown = [{"text": ln, "alert": bool(_HOT_LINE.search(ln))} for ln in picked]
        out.append(
            {
                "market": m,
                "label": _market_label(m, en),
                "metrics": metrics,
                "params": params,
                "skill": None if d0 is None else d0.skill_ok,
                "trust": _trust_view(m, d0, en),
                "sel": sel_line,
                "n_degraded": sum(1 for d in dists if d.degraded),
                "n_served": len(dists),
                "lines": shown,
                "hidden": len(lines) - len(shown),
                "has_lines": bool(lines),
                "alarms": [ln for ln in lines if _HOT_LINE.search(ln)],  # every alarm line, capped or not
            }
        )
    return out


def _methodology_en(settings: Settings) -> dict[str, list[str]]:
    fw, th, r = settings.fusion.weights, settings.fusion.thresholds, settings.risk
    H = settings.fusion.target_horizon_days
    core = [
        f"Final signal = weighted sum of the Factor {fw.get('factor', 0):.0%} · ML {fw.get('ml', 0):.0%} · LLM "
        f"{fw.get('llm', 0):.0%} pillars (renormalised over the available pillars) → Strong Buy ≥ "
        f"{fmt_num(th.strong_buy, 2, True)} (confidence ≥ {settings.fusion.strong_buy_min_confidence}) · Buy ≥ "
        f"{fmt_num(th.buy, 2, True)} · Sell ≤ {fmt_num(th.sell, 2, True)}. Confidence = pillar agreement, strength, "
        "model skill, data quality and evidence, then the LLM critic adjustment (−25 to +5). The LLM only scores, "
        "summarises and critiques text with a fixed, bounded influence; quotes and figures are used only after "
        "they pass verification against the source text.",
        f"Stop = entry − {r.stop_atr_mult:g}×ATR({r.atr_period}) ({r.min_stop_pct:.0%}–{r.max_stop_pct:.0%} range, "
        f"support-adjusted, on closes) · take-profit = max(target, entry + {r.min_reward_risk:g}R) · target = "
        f"weighted mean of ML, consensus and valuation scaled to {H} trading days. Weight = one common scale "
        f"proportional to conviction (composite × confidence / volatility) — top-conviction name capped at "
        f"{r.max_weight:.0%}, names at the Kelly×{r.kelly_cap:g} cap stay capped · gross {r.max_gross:.0%} · the "
        f"largest scale within {r.target_portfolio_vol:.0%} portfolio volatility · names below {r.min_weight:.0%} "
        f"dropped smallest-first and re-solved. A Kelly f* at the search bound (100%) is shown as "
        f"'{KELLY_LIMIT_TXT_EN}' (max {r.kelly_cap:.0%}); it is not an all-in recommendation.",
    ]
    rsmt = [
        f"A model-based tree (MOB) split by sup-LM stability tests (Bonferroni) over "
        f"{len(settings.rsmt.partition_features)} regime variables; each leaf estimates the "
        f"{settings.horizon_days}-trading-day beta-adjusted excess-return distribution (μ, σ, ν) with momentum (γ) "
        "and volatility (η) offsets.",
        f"P(up), P(beat BM), bands, VaR, exit probabilities (holding {H} trading days) and Kelly all come from "
        "simulated paths of the same distribution, so they cannot contradict each other; degraded modes are shown "
        f"and penalised, never hidden. Portfolio VaR, CVaR and P(loss) are {NORMAL_APPROX_EN} on the sample "
        "covariance.",
    ]
    return {"core": core, "rsmt": rsmt}


def _methodology_extra(report: PortfolioReport, settings: Settings, en: bool) -> list[str]:
    """One line on the factor-model blend and the macro overlay, only when those contracts are populated."""
    has_fm = any(t.quant.factor_model is not None for t in report.tickers) or bool(report.factor_backtest)
    out: list[str] = []
    if has_fm:
        b = settings.backtest.factor_pillar_blend
        t_min = settings.backtest.model.skill_min_ic_t
        out.append(
            f"Factor pillar = {1 - b:.0%} legacy composite + {b:.0%} PIT multi-factor model (PER, PBR, traded-value "
            f"trend, 3M/6M momentum, sector; sector-neutral, walk-forward weights; model pillar ×0.35 while OOS IC "
            f"t < {t_min:g})."
            if en
            else f"팩터 필러 = 기존 합성 {1 - b:.0%} + PIT 멀티팩터 모델 {b:.0%}"
            f"(PER·PBR·거래대금 추세·3M·6M 모멘텀·섹터, 섹터 중립·워크포워드 가중, "
            f"OOS IC t < {t_min:g}이면 모델 필러 ×0.35)."
        )
    if report.macro:
        m = settings.macro
        out.append(
            f"Macro regime (global · US · Korea) = mean of the indicators' trailing z-scores (+ = risk-on); the "
            f"overlay scales the gross limit down to ×{m.overlay_min_mult:.2f} at the most risk-off score (risk "
            "control only — no backtested return gain)."
            if en
            else "매크로 레짐(글로벌·미국·한국) = 지표별 과거 z(+ = 위험선호)의 평균이며, "
            f"오버레이는 가장 위험회피일 때 총 투자 한도를 ×{m.overlay_min_mult:.2f}까지 줄입니다"
            "(리스크 관리용 — 백테스트상 수익 개선 근거 없음)."
        )
    return out


def _methodology_sel(report: PortfolioReport, en: bool) -> list[str]:
    """Which model sets the ML direction (the served stock-selection model per market), stated before RSMT-Q."""
    models = _direction_models(report, en)
    if not models:
        return []
    if en:
        return [
            "The ML pillar's direction (which names beat which) comes from each market's stock-selection model: "
            "candidates are compared by nested, purged walk-forward on within-date AUC (above-median label); the "
            "expected excess is used in full only when it beats 12-1 momentum and its AUC 95% lower bound is above "
            f"0.5 (else shrunk). Now: {models}."
        ]
    return [
        "ML 필러의 방향(종목 간 우열)은 시장별 종목 선택 모델이 정합니다 — 후보 모델들을 중첩·정화 워크포워드로 "
        "비교해 날짜 내 AUC(중앙값 초과 여부)로 고르고, 12-1 모멘텀보다 낫고 AUC 95% 하한이 0.5를 넘을 때만 "
        f"초과수익 기대치를 그대로 씁니다(미달이면 축소). 현재: {models}."
    ]


def _methodology(settings: Settings) -> dict[str, list[str]]:
    if is_en(settings):
        return _methodology_en(settings)
    fw, th, r = settings.fusion.weights, settings.fusion.thresholds, settings.risk
    core = [
        f"최종 시그널 = 팩터 {fw.get('factor', 0):.0%} · ML {fw.get('ml', 0):.0%} · LLM {fw.get('llm', 0):.0%} 필라 "
        f"가중합"
        f"(가용 필라 재정규화) → Strong Buy ≥ {fmt_num(th.strong_buy, 2, True)}(신뢰도 "
        f"{settings.fusion.strong_buy_min_confidence} 이상)"
        f" · Buy ≥ {fmt_num(th.buy, 2, True)} · Sell ≤ {fmt_num(th.sell, 2, True)}. 신뢰도 = 필라 일치도·강도·모형 "
        f"스킬·데이터 품질·근거량 합산 후"
        " LLM 비평 조정(−25~+5). LLM은 텍스트 채점·요약·비평만 담당하며 영향 범위가 고정되어 있고, 근거 인용·수치는 "
        "원문 대조 검증을 통과한 것만 사용합니다.",
        f"손절 = 진입가 − {r.stop_atr_mult:g}×ATR({r.atr_period}) ({r.min_stop_pct:.0%}~{r.max_stop_pct:.0%} 범위, "
        f"지지선 보정, 종가 기준)"
        f" · 익절 = max(목표가, 진입가 + {r.min_reward_risk:g}R) · 목표가 = ML·컨센서스·밸류에이션의 "
        f"{settings.fusion.target_horizon_days}거래일 환산 가중평균."
        f" 비중 = 확신도(종합점수×신뢰도/변동성)에 비례하는 단일 배율 — 최고 확신 종목 {r.max_weight:.0%} 상한,"
        f" Kelly×{r.kelly_cap:g} 상한 도달 종목은 상한 고정 · 총 {r.max_gross:.0%}"
        f" · 포트폴리오 변동성 {r.target_portfolio_vol:.0%} 이내 최대 배율"
        f" · {r.min_weight:.0%} 미만은 작은 순으로 제외 후 재계산."
        f" Kelly f*가 탐색 상한(100%)에 닿은 종목은 '{KELLY_LIMIT_TXT}'(최대 {r.kelly_cap:.0%})으로 표기하며,"
        " 전액 투자 권고가 아닙니다.",
    ]
    rsmt = [
        f"레짐 변수 {len(settings.rsmt.partition_features)}개의 sup-LM 안정성 검정(Bonferroni)으로 분할하는 모형 "
        f"트리(MOB)이며,"
        f" 각 리프가 {settings.horizon_days}거래일 베타조정 초과수익 분포(μ, σ, ν)를 추정하고 모멘텀(γ)·변동성(η) "
        f"오프셋을 적용합니다.",
        f"P(상승)·P(BM 초과)·밴드·VaR·청산 확률(보유 {settings.fusion.target_horizon_days}거래일)·Kelly는 같은 "
        f"분포의 시뮬레이션 경로에서"
        " 산출되어 서로 모순되지 않으며, 저하 모드는 숨기지 않고 표시·감점합니다."
        f" 포트폴리오 VaR·CVaR·P(손실)은 표본공분산 기반 {NORMAL_APPROX}입니다.",
    ]
    return {"core": core, "rsmt": rsmt}


def _portfolio_view(report: PortfolioReport, settings: Settings) -> dict[str, Any]:
    en = is_en(settings)
    approx = NORMAL_APPROX_EN if en else NORMAL_APPROX
    invested = [t for t in report.tickers if t.execution.weight > 0]
    invested.sort(key=lambda t: t.execution.weight, reverse=True)
    segs = []
    for i, t in enumerate(invested):
        segs.append(
            {
                "name": t.instrument.name,
                "w": t.execution.weight * 100,
                "txt": fmt_prob(t.execution.weight, 1),
                "color": SERIES_COLORS[i] if i < len(SERIES_COLORS) else None,
            }
        )
    if len(segs) > len(SERIES_COLORS):  # never cycle hues: fold the tail into "기타"
        head, tail = segs[: len(SERIES_COLORS) - 1], segs[len(SERIES_COLORS) - 1 :]
        tw = sum(s["w"] for s in tail)
        segs = head + [
            {
                "name": f"Other {len(tail)}" if en else f"기타 {len(tail)}종목",
                "w": tw,
                "txt": fmt_prob(tw / 100, 1),
                "color": "#8a8985",
            }
        ]
    gross = sum(t.execution.weight for t in report.tickers)
    cash = report.cash_weight
    alloc_gap = abs(gross + cash - 1.0) > 0.005

    # canonical keys (legacy spellings folded in; the canonical spelling wins when both exist)
    pr: dict[str, Any] = {}
    for k, v in report.portfolio_risk.items():
        ck = _RISK_ALIASES.get(k, k)
        if ck not in pr or ck == k:
            pr[ck] = v
    h = settings.horizon_days
    tiles: list[dict[str, str]] = []

    def tile(label: str, value: str, sub: str = "", warn: bool = False) -> None:
        tiles.append({"label": label, "value": value, "sub": sub, "warn": warn})

    def fmt_risk(key: str, v: Any) -> str:
        kind = _RISK_KINDS.get(key, "num")
        return {
            "pct": lambda: fmt_prob(v, 1),
            "spct": lambda: fmt_pct(v),
            "prob": lambda: fmt_prob(v),
            "int": lambda: fmt_int(v),
        }.get(kind, lambda: fmt_num(v, 3))()

    vol = pr.get("ex_ante_vol")
    target_word = "target" if en else "목표"
    tile(
        "Ex-ante vol (ann.)" if en else "사전 변동성(연)",
        fmt_prob(vol, 1),
        f"{target_word} {settings.risk.target_portfolio_vol:.0%}",
        warn=(not _missing(vol) and float(vol) > settings.risk.target_portfolio_vol + 1e-9),
    )
    # VaR / CVaR / P(loss) come from fusion.risk.portfolio_risk as NORMAL approximations: say so on the tile
    tile(
        f"VaR95 ({h}D)",
        fmt_pct(pr.get("var95")),
        f"CVaR {fmt_pct(pr.get('cvar95'))} · {approx}",
    )
    tile(
        f"P(loss, {h}D)" if en else f"P(손실, {h}D)",
        fmt_prob(pr.get("p_loss")),
        f"{'exp.' if en else '기대'} {fmt_pct(pr.get('expected_return'))} · {approx}",
    )
    n_pos = sum(1 for t in report.tickers if t.execution.weight > 0)
    tile(
        "Positions" if en else "편입 종목",
        f"{n_pos} / {len(report.tickers)}",
        f"{'gross' if en else '총 투자'} {fmt_prob(gross, 1)}",
    )
    llm_skipped = bool(report.tickers) and all(t.llm.skipped for t in report.tickers)
    cost = report.llm_cost_usd
    tile(
        "LLM cost" if en else "LLM 비용",
        ("skipped" if en else "생략") if llm_skipped else (MISSING if _missing(cost) else f"${float(cost):,.2f}"),
        settings.llm.model if not llm_skipped else "--no-llm",
        warn=llm_skipped,
    )

    counts = []
    for sig in (Signal.STRONG_BUY, Signal.BUY, Signal.HOLD, Signal.SELL):
        n = sum(1 for t in report.tickers if t.executive.signal is sig)
        counts.append({**_signal_view(sig), "n": n})

    # Module 4 risk line: the upstream labels, with a shared parenthetical (e.g. "20거래일, 정규분포 근사")
    # printed once after a run of items that share it
    entries: list[tuple[str, str]] = []  # (text, parenthetical)
    base, paren = _risk_label("gross", h, en)
    entries.append((f"{base} {fmt_prob(gross, 1)}", paren))
    entries.append((f"{'Cash' if en else '현금'} {fmt_prob(cash, 1)}", ""))
    order = ["ex_ante_vol", "var95", "cvar95", "p_loss", "expected_return"]
    extra_keys = [k for k in pr if k not in order and k not in ("gross", "n_positions")]
    extra = []
    for k in order + extra_keys:
        if k not in pr and k not in ("var95", "p_loss", "ex_ante_vol"):
            continue  # the three headline figures show "—" when missing; optional ones are skipped
        base, paren = _risk_label(k, h, en)
        txt = f"{base} {fmt_risk(k, pr.get(k))}"
        if k == "ex_ante_vol":
            target = f"{target_word} {settings.risk.target_portfolio_vol:.0%}"
            paren = f"{paren}, {target}" if paren else target
        entries.append((txt, paren))
        if k in extra_keys:
            extra.append(f"{base} {fmt_risk(k, pr.get(k))}")
    risk_line: list[str] = []
    i = 0
    while i < len(entries):
        j = i
        while j + 1 < len(entries) and entries[j + 1][1] == entries[i][1]:
            j += 1
        group = " · ".join(e[0] for e in entries[i : j + 1])
        risk_line.append(group + (f" ({entries[i][1]})" if entries[i][1] else ""))
        i = j + 1
    overlay = _overlay_text(report, settings, en)
    if overlay:
        risk_line.append(overlay)
    return {
        "segments": segs,
        "invested_txt": fmt_prob(gross, 1),
        "cash_txt": fmt_prob(cash, 1),
        "cash_w": max(0.0, cash * 100),
        "tiles": tiles,
        "extra": extra,
        "counts": counts,
        "risk_line": risk_line,
        "alloc_gap": alloc_gap,
        "has_risk": bool(pr),
    }


def _kelly_view(plan: ExecutionPlan, en: bool = False) -> dict[str, Any]:
    """Module 4 Kelly cell: the capped fraction (f* x kelly_cap) first; f* underneath only while < 100%.

    Printing "f* 100%" under the capped figure read as an all-in recommendation (live run: 5 of 6 rows);
    at the grid bound the capped figure is the kelly_cap limit itself, and the sub line says so.
    """
    b = plan.barrier
    main = fmt_prob(plan.kelly_fraction, 1)
    if b is None:
        return {"main": main, "sub": f"f* {MISSING}", "at_limit": False}
    f_star = b.kelly_optimal
    if _missing(f_star):
        return {"main": main, "sub": f"f* {MISSING}", "at_limit": False}
    if round(float(f_star) * 100) >= 100:
        return {"main": main, "sub": KELLY_LIMIT_TXT_EN if en else KELLY_LIMIT_TXT, "at_limit": True}
    return {"main": main, "sub": f"f* {fmt_prob(f_star, 0)}", "at_limit": False}


def _barrier_horizons(tickers: list[TickerReport], en: bool = False) -> tuple[int | None, str]:
    """(common barrier horizon or None, header text) over the rows that have a BarrierOutcome.

    BarrierOutcome.horizon_days is the holding horizon the TP/SL/time-exit paths were simulated over
    (fusion.target_horizon_days), which differs from the forecast horizon; 0 means "not recorded".
    """
    hs = sorted({t.execution.barrier.horizon_days for t in tickers if t.execution.barrier})
    if not hs:
        return None, ""
    if len(hs) == 1:
        if en:
            return (hs[0], f"{hs[0]} trading days") if hs[0] > 0 else (0, BARRIER_H_MISSING_EN)
        return (hs[0], f"{hs[0]}거래일") if hs[0] > 0 else (0, BARRIER_H_MISSING)
    return None, "horizons differ by name" if en else "종목별 기간 상이"


BARRIER_H_MISSING = "기간 미기재"
BARRIER_H_MISSING_EN = "horizon not recorded"


# Static template text: key -> (Korean, English). Kept side by side so a label cannot silently lack a translation;
# the Korean strings are exactly the literals of the Korean-only template (byte-identical Korean output).
_TEXT: dict[str, tuple[str, str]] = {
    "html_lang": ("ko", "en"),
    "m_asof": ("기준일", "As of"),
    "m_gen": ("생성", "Generated"),
    "m_univ": ("유니버스", "Universe"),
    "m_ccy": ("기준통화", "Base currency"),
    "m_capital": ("운용", "capital"),
    "m_horizon": ("예측기간", "Horizon"),
    "m_digest": ("설정 digest", "Config digest"),
    "regime": ("시장 레짐", "Market regime"),
    "m1_sub": ("최종 시그널 · 신뢰도 · 목표가 · 권장 보유기간", "Final signal · confidence · target · holding period"),
    "th_name": ("종목", "Name"),
    "th_signal": ("최종 시그널", "Final signal"),
    "th_conf": ("신뢰도", "Confidence"),
    "th_pillars": ("신호 구성", "Signal mix"),
    "th_pillars_sub": ("팩터 · ML · LLM → 종합", "Factor · ML · LLM → composite"),
    "th_price": ("현재가", "Price"),
    "th_target": ("목표가", "Target"),
    "th_upside": ("상승여력", "Upside"),
    "th_horizon": ("권장 기간", "Horizon"),
    "th_weight": ("비중", "Weight"),
    "composite": ("종합", "comp."),
    "summary_h": ("포트폴리오 요약", "Portfolio summary"),
    "invested": ("투자", "Invested"),
    "cash": ("현금", "Cash"),
    "alloc_gap": (
        "⚠ 비중 합계와 현금 비중의 합이 100%가 아닙니다 — 사이징 결과를 확인하세요",
        "⚠ Position weights and cash do not add up to 100% — check the sizing",
    ),
    "sigdist": ("시그널 분포", "Signal mix"),
    "alerts_none": ("✓ 경고 없음 — 데이터 소스·모델·LLM 정상", "✓ No warnings — data sources, models and LLM OK"),
    "charts_h": ("1년 가격 추이 · 매매 레벨", "1-year price · trade levels"),
    "lg_close": ("종가", "Close"),
    "lg_entry": ("진입 구간", "Entry zone"),
    "lg_tp": ("익절 TP", "Take-profit TP"),
    "lg_sl": ("손절 SL", "Stop SL"),
    "lg_target": ("목표가", "Target"),
    "nochart": ("⚠ 차트 없음 (가격 이력 미제공)", "⚠ No chart (no price history)"),
    "th_fcomp": ("팩터 종합", "Factor comp."),
    "th_ml_er": ("ML 기대수익", "ML exp. ret."),
    "th_ml_pup": ("ML P(상승)", "ML P(up)"),
    "th_oos": ("CV AUC · IC", "CV AUC · IC"),
    "th_ml_skill": ("ML 스킬", "ML skill"),
    "th_sel_pct": ("선택 모델 백분위", "Selection pct."),
    "th_sel_mu": ("초과수익 기대", "Exp. excess"),
    "th_sel_auc": ("AUC 표본 외 / 홀드아웃", "AUC OOS / holdout"),
    "th_sel_money": ("상위 20% 순초과(연) 표본 외 / 홀드아웃", "Top-20% net excess/yr OOS / holdout"),
    "sel_confirmed": ("XSH 확인", "XSH ✓"),
    "skill_ok": ("스킬 OK", "Skill OK"),
    "skill_no": ("스킬 미달", "No skill"),
    "ml_none": ("ML 없음", "No ML"),
    "rsmt_h": ("RSMT-Q 분포 예측", "RSMT-Q distribution forecast"),
    "th_er": ("기대수익 E[R]", "Exp. return E[R]"),
    "th_er_sub": ("μ(로그) · σ", "μ(log) · σ"),
    "th_pup": ("P(상승)", "P(up)"),
    "th_pout": ("P(BM 초과)", "P(>BM)"),
    "th_band": ("10–90% 밴드", "10–90% band"),
    "th_band_sub": ("▮ 중앙값 · | 0%", "▮ median · | 0%"),
    "th_leaf": ("리프 경로", "Leaf path"),
    "th_leaf_sub": ("리프 · 표본 · 블렌드", "leaf · n · blend"),
    "th_state": ("상태", "Status"),
    "degraded": ("⚠ 저하", "⚠ Degraded"),
    "nodist": ("⚠ 분포 없음", "⚠ No distribution"),
    "diag_h": ("모델 진단", "Model diagnostics"),
    "diag_na": ("검증 정보 없음", "No validation info"),
    "cv_ok": ("RSMT-Q CV 스킬 OK", "RSMT-Q CV skill OK"),
    "cv_no": ("RSMT-Q CV 스킬 미달", "RSMT-Q CV no skill"),
    "trust": ("위험 엔진 RSMT-Q", "Risk engine RSMT-Q"),
    "diag_sel": ("방향 모델", "Direction model"),
    "base": ("기준", "base"),
    "params": ("오프셋·엣지", "Offsets · edges"),
    "nodiag": ("⚠ 모델 진단 기록 없음", "⚠ No model diagnostics recorded"),
    "m3_sub": (
        "뉴스 감성(−100~+100) · 공시·실적 NLP 요약 · 핵심 리스크",
        "News sentiment (−100~+100) · filings & earnings NLP summary · key risks",
    ),
    "senti": ("감성", "Sentiment"),
    "d7": ("7일", "7D"),
    "senti_skipped": ("미산출(LLM 생략)", "n/a (LLM skipped)"),
    "no_news": ("뉴스 없음", "No news"),
    "llm_skipped": (
        "⚠ LLM 분석 생략(--no-llm) — 정량 신호만 반영",
        "⚠ LLM analysis skipped (--no-llm) — quant signals only",
    ),
    "crit_flag": ("⚠ 치명 플래그: ", "⚠ Critical flag: "),
    "fil_h": ("공시·실적 요약", "Filings & earnings"),
    "guidance": ("가이던스", "Guidance"),
    "tone": ("톤", "tone"),
    "no_summary": ("요약 없음", "No summary"),
    "earnings": ("실적", "Results"),
    "no_filings": ("— 공시 분석 결과 없음", "— No filings analysis"),
    "risks": ("리스크", "Risks"),
    "contra": ("⚡ 상충: ", "⚡ Conflict: "),
    "llm_err": ("⚠ LLM 오류: ", "⚠ LLM error: "),
    "llm_notice": ("ⓘ LLM 알림: ", "ⓘ LLM notice: "),
    "m4_sub": ("편입 비중 · 진입 · 손절/익절 · 청산 확률", "Weight · entry · stop/take-profit · exit probabilities"),
    "key_tp": ("P(익절 선도달)", "P(take-profit first)"),
    "key_time": ("P(기간 만료)", "P(time exit)"),
    "key_sl": ("P(손절 선도달)", "P(stop first)"),
    "th_signal2": ("시그널", "Signal"),
    "th_shares": ("수량", "Shares"),
    "th_shares_sub": ("평가액", "value"),
    "th_entry": ("진입 구간", "Entry zone"),
    "th_sl": ("손절 SL", "Stop SL"),
    "th_tp": ("익절 TP", "Take-profit TP"),
    "th_exit": ("청산 확률", "Exit probability"),
    "th_exit_sum": ("합 100%", "sum 100%"),
    "th_exp": ("기대", "Exp."),
    "th_exp_sub": ("거래수익", "trade ret."),
    "no_path": ("— 경로 없음", "— no paths"),
    "risk_h": ("포트폴리오 리스크", "Portfolio risk"),
    "risk_na": ("⚠ 포트폴리오 리스크 미산출", "⚠ Portfolio risk not computed"),
    "trig_h": ("실행 계획 · 모니터링 트리거", "Execution plan · monitoring triggers"),
    "trig_sub": ("종가 기준 · 조건 충족 시 즉시 재평가", "On closes · reassess as soon as a condition is met"),
    "method": ("방법론", "Methodology"),
    "method_fm": ("팩터·매크로", "Factor · macro"),
    "method_sel": ("종목 선택 모델", "Stock-selection model"),
    "method_rsmt": ("RSMT-Q(위험 엔진)", "RSMT-Q (risk engine)"),
    "macro": ("매크로 레짐", "Macro regime"),
    "macro_exposure": ("노출", "exposure"),
    "macro_stale": ("지연", "stale"),
    "macro_warn": ("경고", "warnings"),
    "fm_h": ("PIT 멀티팩터 모델", "PIT multi-factor model"),
    "fm_pct": ("점수 백분위", "Score pctile"),
    "fm_z": ("점수 z", "Score z"),
    "fm_sec_m3": ("섹터 3M", "Sector 3M"),
    "fm_model": ("모델 필러", "Model pillar"),
    "fm_vol": ("거래대금 추세", "Volume trend"),
    "fm_vol_sub": ("log(ADV20/120)", "log(ADV20/120)"),
    "fm_sector": ("섹터", "Sector"),
    "fm_sector_sub": ("섹터 3M", "sector 3M"),
    "fm_pillar": ("팩터 필러", "Factor pillar"),
    "fm_pillar_sub": ("기존 → 블렌드", "legacy → blended"),
    "fm_model_sub": ("모델", "model"),
    "fm_skill": ("모델 스킬", "Model skill"),
    "fm_none": ("팩터 모델 점수 없음", "no factor-model score"),
    "bt_h": ("팩터 백테스트 요약", "Factor backtest summary"),
    "bt_factor": ("팩터", "Factor"),
    "bt_ic_t": ("IC t", "IC t"),
    "bt_ls": ("L/S 연수익", "L/S ann."),
    "bt_ls_sub": ("비용 후", "after costs"),
    "bt_sharpe": ("샤프", "Sharpe"),
    "bt_bh": ("BH", "BH"),
    "bt_model": ("모델 OOS", "Model OOS"),
    "bt_dir_bad": ("역방향", "wrong sign"),
    "ord_h": ("20분위 참고 모델", "20-quantile reference models"),
    "ord_names": ("이 리포트 종목", "This report's names"),
    "ord_common": ("공통 D20", "In every D20"),
    "ord_rank": ("#", "#"),
    "ord_pos": ("분위 내 위치", "Position"),
    "ord_ref": ("참고", "ref."),
    "ord_outside": (
        "* PIT 유니버스 밖 종목은 자체 가격으로 같은 횡단면에 배치",
        "* names outside the PIT universe are placed into the same cross-section from their own prices",
    ),
}


def _texts(en: bool) -> dict[str, str]:
    return {k: (v[1] if en else v[0]) for k, v in _TEXT.items()}


# ============================================================================ macro / factor model / backtest

MACRO_REGION_ORDER = ("global", "us", "kr")
MACRO_REGION_LABELS = {"global": ("글로벌", "Global"), "us": ("미국", "US"), "kr": ("한국", "Korea")}
# English labels live in macro/regime.py (single source; Korean reports use MacroRegime/MacroIndicator labels)
MACRO_LABEL_EN = _MACRO_LABELS_EN
MACRO_INDICATOR_EN = _MACRO_INDICATOR_LABELS_EN
MACRO_MARKET_REGION = {"KR": "kr", "US": "us"}
# backtest/factors.py FACTORS -> short display labels (ko, en)
BACKTEST_FACTOR_LABELS = {
    "value_ep": ("E/P(PER 역수)", "E/P (1/PER)"),
    "value_bp": ("B/P(PBR 역수)", "B/P (1/PBR)"),
    "volume_trend": ("거래대금 추세", "Volume trend"),
    "mom_3m": ("3M 모멘텀", "3M momentum"),
    "mom_6m": ("6M 모멘텀", "6M momentum"),
    "sector_mom_3m": ("섹터 3M 모멘텀", "Sector 3M mom."),
}


# pipeline.py regime-chip values ("20D +3.1% · MA200 위", "15.2 (저변동)") are built in Korean upstream; an English
# report translates their fixed tokens here so no Hangul reaches the page
_REGIME_TOKENS_EN = (
    ("MA200 위", "above MA200"),
    ("MA200 아래", "below MA200"),
    ("(저변동)", "(low vol)"),
    ("(보통)", "(normal)"),
    ("(고변동)", "(high vol)"),
)


def _regime_value_en(v: str) -> str:
    for ko, en in _REGIME_TOKENS_EN:
        v = v.replace(ko, en)
    return v


def _macro_cls(r: MacroRegime, settings: Settings) -> str:
    if r.score is None or _missing(r.score):
        return "na"
    if r.score <= settings.macro.risk_off_threshold:
        return "off"
    if r.score >= settings.macro.risk_on_threshold:
        return "on"
    return "neu"


# level indicators whose key ends in "_3m" only because of the 3-month bill (10Y−3M spread)
MACRO_LEVEL_KEYS = frozenset({"us_curve_10y_3m"})


def _macro_value_txt(key: str, unit: str, value: float | None, change: float | None, en: bool) -> str:
    """The number shown for an indicator: 3M indicators show their 3M change, the others their level."""
    pp = "pp" if en else "%p"
    if key.endswith("_3m") and key not in MACRO_LEVEL_KEYS and not _missing(change):
        if unit in ("%", "%p"):  # yields / spreads: absolute change in percentage points
            return _minus(f"{float(change):+.2f}{pp}")
        return fmt_pct(change, 1)  # prices / ratios: decimal return
    if _missing(value):
        return MISSING
    v = float(value)
    if unit == "%":
        return _minus(f"{v:+.1f}%")
    if unit == "%p":
        return _minus(f"{v:+.2f}{pp}")
    return _minus(f"{v:,.4g}")


def _macro_view(report: PortfolioReport, settings: Settings, en: bool, level: int) -> dict[str, Any] | None:
    """Page-1 macro strip (global / US / Korea): label, score, 2-3 strongest indicators, overlay multiplier."""
    if not report.macro:
        return None
    by_region = {r.region: r for r in report.macro}
    n_ind = 3 if level < 3 else 2
    cells = []
    for region in [*MACRO_REGION_ORDER, *(k for k in by_region if k not in MACRO_REGION_ORDER)]:
        r = by_region.get(region)
        if r is None:
            continue
        usable = [i for i in r.indicators if not _missing(i.z)]
        usable.sort(key=lambda i: -abs(float(i.z)))
        inds = []
        for i in usable[:n_ind]:
            label = MACRO_INDICATOR_EN.get(i.key, i.key) if en else i.label
            inds.append(
                {
                    "label": label,
                    "value": _macro_value_txt(i.key, i.unit, i.value, i.change_3m, en),
                    "z": fmt_num(i.z, 1, signed=True),
                    "neg": float(i.z) < 0,
                }
            )
        ko_name, en_name = MACRO_REGION_LABELS.get(region, (region, region))
        label = MACRO_LABEL_EN.get(r.label, r.label) if en else r.label
        mult = r.exposure_mult
        cells.append(
            {
                "name": en_name if en else ko_name,
                "label": label,
                "cls": _macro_cls(r, settings),
                "score": fmt_num(r.score, 2, signed=True),
                "inds": inds,
                "mult": f"×{mult:.2f}" if not _missing(mult) and mult < 0.995 else "",
                "n_stale": len(r.stale),
                "n_warn": len(r.warnings),
                "as_of": f"{r.as_of:%m-%d}",
            }
        )
    if not cells:
        return None
    as_of = max(r.as_of for r in report.macro)
    return {"cells": cells, "as_of": f"{as_of:%Y-%m-%d}"}


def _overlay_mult(report: PortfolioReport, settings: Settings) -> float | None:
    """Gross-exposure multiplier the macro overlay applies (pipeline.macro_overlay's rule: mean exposure_mult of
    the Buy/Strong Buy names' market regimes); None when no overlay applies."""
    if not report.macro or not (settings.macro.enabled and settings.macro.overlay):
        return None
    by_region = {r.region: r for r in report.macro}
    mults = [
        by_region[MACRO_MARKET_REGION[t.instrument.market.value]].exposure_mult
        for t in report.tickers
        if t.executive.signal in (Signal.BUY, Signal.STRONG_BUY)
        and MACRO_MARKET_REGION.get(t.instrument.market.value) in by_region
    ]
    if not mults:
        return None
    mult = float(sum(mults) / len(mults))
    return mult if mult < 0.999 else None


def _overlay_text(report: PortfolioReport, settings: Settings, en: bool) -> str:
    mult = _overlay_mult(report, settings)
    if mult is None:
        return ""
    g = settings.risk.max_gross
    if en:
        return f"Macro overlay ×{mult:.2f} → gross limit {g:.0%}→{g * mult:.0%} (risk control)"
    return f"매크로 오버레이 ×{mult:.2f} → 총 투자 한도 {g:.0%}→{g * mult:.0%}(리스크 관리용)"


def _factor_model_view(report: PortfolioReport, settings: Settings, en: bool) -> list[dict[str, Any]] | None:
    """Module 2 PIT multi-factor model rows (None when no ticker has QuantSignal.factor_model)."""
    if not any(t.quant.factor_model is not None for t in report.tickers):
        return None
    rows = []
    for t in report.tickers:
        fm: FactorModelScore | None = t.quant.factor_model
        row: dict[str, Any] = {"name": t.instrument.name, "ticker": t.instrument.ticker, "fm": None}
        if fm is not None:
            ex = fm.exposures or {}
            legacy = t.quant.factor_pillar if t.quant.factor_composite is not None else None
            blended = t.executive.pillars.get("factor")
            note = next((o for o in t.executive.overrides if _is_factor_blend_note(o)), "")
            sector = sector_label(fm.sector or t.instrument.sector, "en" if en else "ko") or MISSING
            row["fm"] = {
                "pct": fmt_prob(fm.percentile, 0),
                "pct_w": max(0.0, min(100.0, float(fm.percentile) * 100.0)),
                "score": fmt_num(fm.score, 2, signed=True),
                "per": MISSING if _missing(ex.get("per")) else f"{float(ex['per']):.1f}",
                "pbr": MISSING if _missing(ex.get("pbr")) else f"{float(ex['pbr']):.2f}",
                "vol": fmt_num(ex.get("volume_trend"), 2, signed=True),
                "m3": fmt_pct(ex.get("mom_3m"), 1),
                "m6": fmt_pct(ex.get("mom_6m"), 1),
                "m3_neg": not _missing(ex.get("mom_3m")) and float(ex["mom_3m"]) < 0,
                "m6_neg": not _missing(ex.get("mom_6m")) and float(ex["mom_6m"]) < 0,
                "sector": sector,
                "sec_m3": fmt_pct(ex.get("sector_mom_3m"), 1),
                "pillar": fmt_num(fm.pillar, 2, signed=True),
                "legacy": fmt_num(legacy, 2, signed=True),
                "blended": fmt_num(blended, 2, signed=True),
                "skill": bool(fm.skill_ok),
                "ic_t": fmt_num(fm.oos_ic_t, 1, signed=True),
                "note": note,
                "notes": [n for n in fm.notes if n and n.strip()],
            }
        rows.append(row)
    return rows


ORDER_STALE_DAYS = 35  # a month-end cross-section older than this misses a newer month-end


_ORDER_CAVEAT = (  # the 91 ordering model's caveat; a model's own text comes with its artifact (stats caveat_ko/en)
    "추천이 아닙니다: 결과를 본 뒤 고른 미확인 모델이고, 성과가 소수의 급등 달에 몰려 있으며 D20은 고변동·고베타 "
    "종목이 모여 손실 폭이 큽니다(docs/MODELING.md 91.5·93)",
    "Not a recommendation: the model was chosen after its results were seen and is unconfirmed; the gains sit in a "
    "few surge months and D20 gathers high-volatility, high-beta names with deep drawdowns (docs/MODELING.md 91.5, 93)",
)


def _order_stats(d: dict[str, Any], k: int, en: bool) -> str:
    st = d.get("stats") or {}
    if not st:
        return ""
    p = {key: fmt_pct(st.get(key), 1) for key in ("top_ann", "top_ann_net", "universe_ann", "top_ann_drop10")}
    p["mom"], p["mdd"] = fmt_pct(st.get("mom_top_ann"), 1), fmt_pct(st.get("top_mdd"), 0)
    h, bp = int(d.get("horizon") or 20), float(st.get("cost_bps") or 50)
    a, b, n = st.get("first"), st.get("last"), st.get("months")
    if en:
        return (
            f"Post hoc books {a}–{b} ({n} months, equal weight at each month-end, held {h} trading days): "
            f"D{k} {p['top_ann']}/yr ({p['top_ann_net']} after {bp:.0f} bp), universe {p['universe_ann']}/yr · "
            f"D{k} max drawdown {p['mdd']} · without its best 10 months {p['top_ann_drop10']}/yr · "
            f"12-1 momentum D{k} {p['mom']}/yr"
        )
    return (
        f"사후 장부 {a}~{b}({n}개월, 매달 말 같은 비중·{h}거래일 보유): D{k} 연 {p['top_ann']}"
        f"(비용 {bp:.0f}bp 후 {p['top_ann_net']}) · 전체 평균 연 {p['universe_ann']} · D{k} 최대 낙폭 "
        f"{p['mdd']} · 최고 10개월 제외 시 연 {p['top_ann_drop10']} · 12-1 모멘텀 D{k} 연 {p['mom']}"
    )


def _ord_other_label(en: bool, k: int) -> str:
    return f"other models' D{k}" if en else f"다른 모델 D{k}"


def _order_view(report: PortfolioReport, en: bool) -> dict[str, Any] | None:
    """Module 2 sub-block: the 20-quantile REFERENCE models (PortfolioReport.order_ventile, docs/MODELING.md 93,
    94.3) - per market each report name's score ventile under every stored model (D1 = bottom 5% … D20 = top 5% of
    the model's PIT cross-section), each model's D20 names (names in every model's D20 marked), its post hoc book
    statistics and why it is not a recommendation. Markets of the report without a model are named once. None
    without any data. A market's value is a list of views (one per model) or, in reports before 94.3, one view."""
    ov = report.order_ventile or {}
    if not ov:
        return None
    blocks, subs = [], []
    for m in sorted(ov, key=lambda x: (x != "KR", x)):
        raw = ov[m] or []
        views = [v for v in (raw if isinstance(raw, list) else [raw]) if isinstance(v, dict) and v]
        if not views:
            continue
        k = int(views[0].get("k") or 20)
        mk = _market_label(m, en)

        def lab(d: dict[str, Any], key: str) -> str:
            st = d.get("stats") or {}
            return str(st.get(f"{key}_{'en' if en else 'ko'}") or d.get("model") or "?")

        tops = [{n.get("ticker") for n in d.get("top") or []} for d in views]
        common = set.intersection(*tops) if len(tops) > 1 else set()
        rows = []
        for t in report.tickers:
            if t.instrument.market.value != m:
                continue
            tk = t.instrument.ticker
            cells, outside = [], False
            for d in views:
                v = (d.get("tickers") or {}).get(tk)
                if v is not None:
                    outside = outside or not bool(v.get("in_universe", True))
                cells.append(
                    None
                    if v is None
                    else {"abbr": lab(d, "abbr"), "v": f"D{int(v['ventile'])}", "top": int(v["ventile"]) == k}
                )
            if any(c is not None for c in cells):
                rows.append(
                    {
                        "name": t.instrument.name,
                        "ticker": tk,
                        "cells": [c for c in cells if c is not None],
                        "top": any(c is not None and c["top"] for c in cells),
                        "outside": outside,
                    }
                )
        mine = {r["ticker"] for r in rows}
        lists = []
        for d in views:
            top = [
                {
                    "name": n.get("name") or n.get("ticker"),
                    "ticker": n.get("ticker"),
                    "mine": n.get("ticker") in mine,
                    "common": n.get("ticker") in common,
                }
                for n in d.get("top") or []
            ]
            st = d.get("stats") or {}
            age = int(d.get("age_days") or 0)
            short = lab(d, "short")
            lists.append(
                {
                    "model": d.get("model"),
                    "top": top,
                    "label": f"{short} D{k} · {len(top)} names" if en else f"{short} D{k} {len(top)}종목",
                    "stats": _order_stats(d, k, en),
                    "caveat": str(st.get("caveat_en" if en else "caveat_ko") or _ORDER_CAVEAT[1 if en else 0]),
                    "stale": (
                        f"cross-section {d.get('as_of')} is {age} days old - refresh at the next month-end"
                        if en
                        else f"기준일 {d.get('as_of')}로부터 {age}일 경과 — 새 월말 자료로 갱신 필요"
                    )
                    if age > ORDER_STALE_DAYS
                    else "",
                }
            )
        names = {n["ticker"]: n["name"] for b in lists for n in b["top"]}
        lead, st0 = views[0], views[0].get("stats") or {}
        in_other = set().union(*tops[1:]) if len(tops) > 1 else set()
        ranked = [
            {
                "rank": i,
                "name": n.get("name") or n.get("ticker"),
                "ticker": n.get("ticker"),
                "sector": n.get("sector") or "",
                "pos": (
                    f"top {100 * (1 - float(n['percentile'])):.1f}%"
                    if en
                    else f"상위 {100 * (1 - float(n['percentile'])):.1f}%"
                )
                if n.get("percentile") is not None
                else "",
                "mine": n.get("ticker") in mine,
                "other": n.get("ticker") in in_other,
            }
            for i, n in enumerate(lead.get("top") or [], 1)
        ]
        half = (len(ranked) + 1) // 2
        tiles = []
        if st0:
            bp = float(st0.get("cost_bps") or 50)
            for key, ko, en_l, dig in (
                ("top_ann", f"D{k} 연 수익(비용 전)", f"D{k} per year (gross)", 1),
                ("top_ann_net", f"비용 {bp:.0f}bp 후", f"after {bp:.0f} bp", 1),
                ("universe_ann", "전체 평균", "universe", 1),
                ("top_mdd", f"D{k} 최대 낙폭", f"D{k} max drawdown", 0),
                ("top_ann_drop10", "최고 10개월 제외", "without best 10 months", 1),
            ):
                if st0.get(key) is not None:
                    tiles.append({"label": en_l if en else ko, "value": fmt_pct(st0.get(key), dig)})
        others = views[1:]
        primary = {
            **lists[0],
            "halves": [h for h in (ranked[:half], ranked[half:]) if h],
            "tiles": tiles,
            "period": (
                f"post hoc books {st0.get('first')}–{st0.get('last')} · {st0.get('months')} months · equal weight at "
                f"each month-end, held {int(lead.get('horizon') or 20)} trading days"
                if en
                else f"사후 장부 {st0.get('first')}~{st0.get('last')} · {st0.get('months')}개월 · 매달 말 같은 비중, "
                f"{int(lead.get('horizon') or 20)}거래일 보유"
            )
            if st0
            else "",
            "other_label": (f"{lab(others[0], 'abbr')} D{k}" if len(others) == 1 else _ord_other_label(en, k))
            if others
            else "",
        }
        blocks.append(
            {
                "market": m,
                "legend": " · ".join(f"{lab(d, 'abbr')} = {lab(d, 'short')}" for d in views) if len(views) > 1 else "",
                "rows": rows,
                "lists": lists,
                "primary": primary,
                "others": lists[1:],
                "common": ", ".join(f"{names[c]} {c}" for c in sorted(common, key=lambda c: names[c])),
                "has_outside": any(r["outside"] for r in rows),
                "ref": {
                    r["ticker"]: {
                        "txt": f"{_texts(en)['ord_ref']} {r['cells'][0]['abbr']} {r['cells'][0]['v']}",
                        "top": r["cells"][0]["top"],
                    }
                    for r in rows
                    if (lead.get("tickers") or {}).get(r["ticker"]) is not None
                },
            }
        )
        seen = sorted({(str(d.get("as_of")), int(d.get("n") or 0)) for d in views})
        for a, n in seen:
            subs.append(
                f"{mk} · as of {a} · {n}-name PIT cross-section" if en else f"{mk} · 기준일 {a} · {n}종목 횡단면"
            )
    if not blocks:
        return None
    sub = " / ".join(subs) + (
        " · score ventile D1 (bottom 5%) … D20 (top 5%) · not used by the signal or the sizing"
        if en
        else " · 점수 20분위 D1(하위 5%) … D20(상위 5%) · 신호·비중에 미반영"
    )
    absent = sorted({t.instrument.market.value for t in report.tickers} - {b["market"] for b in blocks})
    return {
        "sub": sub,
        "blocks": blocks,
        "absent": (
            " · ".join(
                f"{_market_label(m, en)}: no model with a D20 edge (docs/MODELING.md 88-90, 95) - not shown"
                if en
                else f"{_market_label(m, en)}: D20 우위가 있는 모델 없음(docs/MODELING.md 88~90·95) — 표시하지 않음"
                for m in absent
            )
        ),
    }


AVOID_STALE_DAYS = 45  # the momentum cross-section moves slowly, but a flag older than this is marked as stale


def _avoid_file(settings: Settings) -> Path:
    return Path(settings.cache_path) / "reference" / "avoid_US.json"


def _avoid_view(report: PortfolioReport, settings: Settings, en: bool) -> dict[str, Any] | None:
    """Module 2 sub-block: the US "avoid" REFERENCE flag (docs/MODELING.md 103.4, 104) - the report's US names whose
    12-1 momentum sits in the bottom quantile of the S&P 500 PIT cross-section, from the file written by
    scripts/serve_avoid.py (``<cache>/reference/avoid_US.json``). Display only: nothing in the signal or the sizing
    reads it. None without US names, without the file or when the file cannot be used."""
    us = [t for t in report.tickers if t.instrument.market.value == "US"]
    if not us:
        return None
    try:
        d = json.loads(_avoid_file(settings).read_text(encoding="utf-8"))
        tick = d["tickers"]
        as_of = date.fromisoformat(str(d["as_of"]))
        q, n, n_low = float(d["q"]), int(d["n"]), int(d["n_avoid"])
        parts = []
        for e in d.get("evidence") or []:
            t_txt = _minus(f"{float(e['t']):+.1f}")
            parts.append(f"{e['en' if en else 'ko']} {fmt_pct(e['alpha'], 1)} (t {t_txt})")
        ev = " · ".join(parts)
    except Exception:  # noqa: BLE001 - reference only: the report just goes without it
        return None
    rows, ref = [], {}
    for t in us:
        v = tick.get(t.instrument.ticker.upper().replace(".", "-"))
        if not v or not v.get("avoid"):
            continue
        pos = f"bottom {100 * float(v['pct']):.0f}%" if en else f"하위 {100 * float(v['pct']):.0f}%"
        rows.append({"name": t.instrument.name, "ticker": t.instrument.ticker, "pos": pos})
        ref[t.instrument.ticker] = {"txt": f"ref. avoid side · 12-1 {pos}" if en else f"참고 피할 쪽 · 12-1 {pos}"}
    gen = report.generated_at if report.generated_at.tzinfo else report.generated_at.replace(tzinfo=UTC)
    age = (gen.astimezone(ZoneInfo("Asia/Seoul")).date() - as_of).days
    qt = f"{100 * q:.0f}%"
    doc = str(d.get("doc") or "103.4")
    if en:
        return {
            "h": "US avoid-side reference (post hoc)",
            "sub": f"as of {as_of} · bottom {qt} by 12-1 momentum among {n} S&P 500 names ({n_low}) · "
            "not used by the signal or the sizing",
            "rows": rows,
            "none": "none of this report's US names",
            "stats": f"Market-risk-adjusted excess return of the bottom {qt} book, per year: {ev}" if ev else "",
            "caveat": "Not a sell recommendation: an observation made after the results were seen (not a "
            f"pre-declared test); bottom-{qt} names do rally in some years (docs/MODELING.md {doc})",
            "stale": f"cross-section is {age} days old - refresh with scripts/serve_avoid.py"
            if age > AVOID_STALE_DAYS
            else "",
            "ref": ref,
        }
    return {
        "h": "미국 피할 쪽 참고(사후 관찰)",
        "sub": f"기준일 {as_of} · S&P 500 {n}종목 중 12-1 모멘텀 하위 {qt}({n_low}종목) · 신호·비중에 미반영",
        "rows": rows,
        "none": "이 리포트의 미국 종목 중 해당 없음",
        "stats": f"하위 {qt} 장부의 시장 위험 감안 초과수익(연): {ev}" if ev else "",
        "caveat": "매도 권고가 아닙니다: 결과를 본 뒤 찾은 관찰이고(사전 판정 대상 아님), 하위 "
        f"{qt} 종목이 크게 오르는 해도 있습니다(docs/MODELING.md {doc})",
        "stale": f"기준일로부터 {age}일 경과 — scripts/serve_avoid.py로 갱신 필요" if age > AVOID_STALE_DAYS else "",
        "ref": ref,
    }


def _period_txt(period: Any) -> str:
    if isinstance(period, list | tuple) and len(period) == 2:
        return f"{str(period[0])[:7]}~{str(period[1])[:7]}"
    if isinstance(period, dict):
        a, b = period.get("start"), period.get("end")
        return f"{str(a)[:7]}~{str(b)[:7]}" if a and b else MISSING
    return str(period) if period else MISSING


def _backtest_view(report: PortfolioReport, settings: Settings, en: bool) -> list[dict[str, Any]] | None:
    """Compact per-market backtest summary (PortfolioReport.factor_backtest[market], backtest.report_summary)."""
    if not report.factor_backtest:
        return None
    variant = "sector_neutral" if settings.backtest.sector_neutral else "raw"
    order = list(BACKTEST_FACTOR_LABELS)
    out = []
    for market in sorted(report.factor_backtest, key=lambda m: (m != "KR", m)):
        bt = report.factor_backtest.get(market) or {}
        if not isinstance(bt, dict):
            log.warning("factor_backtest[%s] is not a dict (%s) - skipped on the report", market, type(bt).__name__)
            continue
        by_factor: dict[str, dict[str, Any]] = {}
        for r in bt.get("factors") or []:
            f = r.get("factor")
            if f is None:
                continue
            if f not in by_factor or r.get("variant") == variant:
                by_factor[f] = r
        rows = []
        for f in sorted(by_factor, key=lambda k: order.index(k) if k in order else len(order)):
            r = by_factor[f]
            ls = r.get("ls_net") or {}
            ko_l, en_l = BACKTEST_FACTOR_LABELS.get(f, (r.get("label") or f, f))
            rows.append(
                {
                    "label": en_l if en else ko_l,
                    "ic_t": fmt_num(r.get("ic_t"), 1, signed=True),
                    "ic_neg": not _missing(r.get("ic_t")) and float(r["ic_t"]) < 0,
                    "ls": fmt_pct(ls.get("ann_return"), 1),
                    "ls_neg": not _missing(ls.get("ann_return")) and float(ls["ann_return"]) < 0,
                    "sharpe": fmt_num(ls.get("sharpe"), 2, signed=True),
                    "bh": bool(r.get("bh_significant")),
                    "dir_bad": r.get("direction_ok") is False,
                }
            )
        m = bt.get("model") or {}
        mls = m.get("ls_net") or {}
        cost = bt.get("cost_bps")
        n_m = bt.get("n_months")
        if en:
            universe = f"PIT top {settings.backtest.kr_universe_n}" if market == "KR" else "PIT S&P 500"
            meta = " · ".join(
                x
                for x in (
                    _period_txt(bt.get("period")),
                    f"{n_m} months" if n_m else "",
                    universe,
                    f"cost {cost:g}bp" if not _missing(cost) else "",
                )
                if x
            )
        else:
            universe = bt.get("universe") or (
                f"PIT 상위 {settings.backtest.kr_universe_n}" if market == "KR" else "PIT S&P 500"
            )
            meta = " · ".join(
                x
                for x in (
                    _period_txt(bt.get("period")),
                    f"{n_m}개월" if n_m else "",
                    str(universe),
                    f"비용 {cost:g}bp" if not _missing(cost) else "",
                )
                if x
            )
        out.append(
            {
                "market": market,
                "label": _market_label(market, en),
                "meta": meta,
                "rows": rows,
                "model": {
                    "ic_t": fmt_num(m.get("oos_ic_t"), 1, signed=True),
                    "ls": fmt_pct(mls.get("ann_return"), 1),
                    "ls_neg": not _missing(mls.get("ann_return")) and float(mls["ann_return"]) < 0,
                    "sharpe": fmt_num(mls.get("sharpe"), 2, signed=True),
                    "skill": bool(m.get("skill_ok")),
                    "months": m.get("oos_months"),
                    "min_t": fmt_num(m.get("skill_min_ic_t") or settings.backtest.model.skill_min_ic_t, 1),
                    "present": bool(m),
                },
                "bh_q": fmt_num(bt.get("bh_q"), 2),
            }
        )
    return out or None


def build_view(
    report: PortfolioReport,
    settings: Settings,
    compact_level: int = 0,
    charts: dict[str, str] | None = None,
    *,
    overflow_note: str | None = None,
    info_note: str | None = None,
) -> dict[str, Any]:
    """Pre-formatted, template-ready view model (every number already a display string).

    `overflow_note` (set by `fit_to_pages` when even the highest compaction exceeds max_pages) is printed
    as the first page-1 note, so an over-long report says so inside the document itself.
    """
    level = max(0, min(MAX_COMPACT_LEVEL, int(compact_level)))
    en = is_en(settings)
    tickers = report.tickers
    gen = report.generated_at
    if gen.tzinfo is None:
        gen = gen.replace(tzinfo=UTC)
    gen_kst = gen.astimezone(KST)
    as_of_by_market: dict[str, str] = {}
    for t in tickers:
        m = t.instrument.market.value
        d = t.as_of.isoformat()
        as_of_by_market[m] = max(as_of_by_market.get(m, d), d)
    as_of_dates = sorted(set(as_of_by_market.values()))
    if len(as_of_dates) == 1:
        as_of_txt = f"{as_of_dates[0]} {'close' if en else '종가'}"
    elif as_of_dates:
        as_of_txt = " · ".join(f"{m} {d}" for m, d in as_of_by_market.items())
    else:
        as_of_txt = MISSING
    n_by_market = {m: sum(1 for t in tickers if t.instrument.market.value == m) for m in as_of_by_market}
    universe = (f"{len(tickers)} names" if en else f"{len(tickers)}종목") + (
        " (" + " · ".join(f"{m} {n}" for m, n in n_by_market.items()) + ")" if n_by_market else ""
    )
    all_skipped = bool(tickers) and all(t.llm.skipped for t in tickers)
    run_skip_notes: list[str] = []  # the per-ticker 'LLM 분석 미실행' override of a --no-llm run, stated once

    # shared 10-90% band axis: rounded up to 5%, non-finite quantiles ignored, capped so one extreme name
    # cannot squash every other row (clipped bars get an edge marker; the exact numbers are printed below)
    band_lim = 0.0
    for t in tickers:
        d = t.quant.distribution
        if d is not None:
            for q in (d.q10, d.q90):
                if not _missing(q):
                    band_lim = max(band_lim, abs(float(q)))
    band_lim = min(BAND_AXIS_MAX, max(0.05, math.ceil(band_lim * 20) / 20))
    barrier_h, barrier_h_txt = _barrier_horizons(tickers, en)

    def to_x(v):
        return (v + band_lim) / (2 * band_lim) * 100

    rows = []
    for t in tickers:
        inst, ex, plan, q = t.instrument, t.executive, t.execution, t.quant
        ccy = inst.currency
        alerts = _alerts_for(t, run_llm_skipped=all_skipped, en=en)
        row_overrides = [o for o in ex.overrides if not _run_skip_override(o, all_skipped) and not _is_blend_note(o)]
        for o in ex.overrides:
            if _run_skip_override(o, all_skipped) and o not in run_skip_notes:
                run_skip_notes.append(o)
        sub = " · ".join(x for x in (inst.ticker, inst.exchange or inst.market.value) if x)
        pillars = []
        ml_label = f"ML({q.excess_model.model})" if getattr(q, "excess_model", None) is not None else "ML"
        for key, label in (("factor", "Factor" if en else "팩터"), ("ml", ml_label), ("llm", "LLM")):
            v = ex.pillars.get(key)
            pillars.append({"label": label, "txt": fmt_num(v, 2, signed=True), **_signed_share(v)})
        d = q.distribution
        dist_view = None
        if d is not None:
            band = None
            if not any(_missing(q) for q in (d.q10, d.q50, d.q90)):
                lo = max(-band_lim, min(band_lim, d.q10))
                hi = max(-band_lim, min(band_lim, d.q90))
                mid = max(-band_lim, min(band_lim, d.q50))
                band = {
                    "left": to_x(lo),
                    "width": max(0.8, to_x(hi) - to_x(lo)),
                    "mid": to_x(mid),
                    "zero": to_x(0.0),
                    "clip_lo": d.q10 < -band_lim,
                    "clip_hi": d.q90 > band_lim,
                }
            dist_view = {
                "er": fmt_pct(d.expected_return),
                "er_neg": d.expected_return < 0,
                "mu": fmt_pct(d.mu, 1),
                "sigma": fmt_prob(d.sigma, 1),
                "pup": fmt_prob(d.prob_up),
                "pout": fmt_prob(d.prob_outperform),
                "band_txt": f"{fmt_pct(d.q10, 1)} ~ {fmt_pct(d.q90, 1)}",
                "band": band,
                "var95": fmt_pct(d.var95),
                "cvar95": fmt_pct(d.cvar95),
                "leaf": (f"leaf #{d.leaf_id}" if en else f"리프#{d.leaf_id}") if d.leaf_id >= 0 else MISSING,
                "leaf_n": f"n={d.leaf_n:,}" if d.leaf_n else "",
                "leaf_path": _leaf_text(d.leaf_path) or ("root (no split)" if en else "루트(분할 없음)"),
                "blend": " · ".join(
                    f"{'RSMT' if k == 'rsmt' else k} {v * 100:.0f}" for k, v in d.blend.items() if v > 0
                )
                or MISSING,
                "skill": d.skill_ok,
                "degraded": d.degraded,
                "degraded_reasons": ", ".join(d.degraded_reasons) or ("reason not given" if en else "사유 미기재"),
            }
        # ML columns without a stock-selection model: the RSMT-Q distribution carries the ML pillar (its direction
        # gate = the pillar's skill); no distribution = no ML pillar ("No ML" placeholder)
        ml_view = None
        if d is not None:
            cv = d.cv_metrics or {}
            ml_view = {
                "er": fmt_pct(d.expected_return),
                "er_neg": (not _missing(d.expected_return)) and d.expected_return < 0,
                "pup": fmt_prob(d.prob_up),
                "auc": fmt_num(cv.get("auc"), 3),
                "ic": fmt_num(cv.get("ic"), 3, signed=True),
                "skill": d.direction_skill_ok,
            }
        sel_view = _selection_view(q.excess_model, en)
        b = plan.barrier
        barrier = None
        if b is not None:
            p_tp, p_time, p_sl = _round_to_100([b.p_take_profit, b.p_time_exit, b.p_stop_loss])
            barrier = {
                "tp": p_tp,
                "time": p_time,
                "sl": p_sl,
                "w_tp": b.p_take_profit * 100,
                "w_time": b.p_time_exit * 100,
                "w_sl": b.p_stop_loss * 100,
                "exp": fmt_pct(b.expected_trade_return),
                "exp_neg": b.expected_trade_return < 0,
                # per-row horizon only when it differs from the one printed in the column header
                "h_note": ""
                if b.horizon_days == barrier_h
                else (
                    (f"{b.horizon_days}D" if en else f"{b.horizon_days}거래일")
                    if b.horizon_days > 0
                    else ("⚠ horizon n/a" if en else "⚠ 기간 미기재")
                ),
                "h_missing": b.horizon_days <= 0,
            }
        llm = t.llm
        risks = _risks_view(llm.filings.risk_factors if llm.filings else [], level, en)
        contradiction = ""
        if level < 3 and llm.critic and llm.critic.contradictions:
            contradiction = llm.critic.contradictions[0]
        chart = None
        if level < 1 and charts is not None:
            svg = charts.get(inst.ticker)
            chart = {"svg": Markup(svg) if svg else None}
        rows.append(
            {
                "ticker": inst.ticker,
                "name": inst.name,
                "sub": sub,
                "market": inst.market.value,
                "ccy": ccy,
                "signal": _signal_view(ex.signal),
                "confidence": int(ex.confidence),
                "conf_txt": f"{int(ex.confidence)}",
                "composite": fmt_num(ex.composite, 2, signed=True),
                "composite_bar": _signed_share(ex.composite),
                "pillars": pillars,
                "price": fmt_price(t.last_price, ccy, settings),
                "target": fmt_price(ex.target_price, ccy, settings),
                "upside": fmt_pct(ex.upside),
                "upside_neg": (not _missing(ex.upside)) and ex.upside < 0,
                "horizon": ex.horizon_label
                or (
                    (f"{ex.horizon_days} trading days" if en else f"{ex.horizon_days}거래일")
                    if ex.horizon_days
                    else MISSING
                ),
                "thesis": ex.thesis or MISSING,
                "overrides": row_overrides,
                "weight": fmt_prob(plan.weight, 1) if plan.weight > 0 else "0%",
                "weight_zero": plan.weight <= 0,
                "alerts": alerts,
                "n_alerts": len(alerts),
                "has_critical": any(a["kind"] == "critical" for a in alerts),
                "factors": _factor_cells(t, en),
                "factor_comp": fmt_num(q.factor_composite, 0),
                "ml": ml_view,
                "sel": sel_view,
                "dist": dist_view,
                "llm_skipped": llm.skipped,
                "llm_errors": llm.errors,
                "llm_notices": [n.strip() for n in llm.notices if n and n.strip()],
                "earnings": _earnings_view(t, gen, en),
                "news": _news_view(llm.news, level, en),
                "filings": _filings_view(llm.filings, level, en),
                "risks": risks,
                "critical_flag": (llm.critic.critical_reason or ("reason not given" if en else "사유 미기재"))
                if (llm.critic and llm.critic.critical_flag)
                else "",
                "contradiction": contradiction,
                "entry_low": fmt_price(plan.entry_low, ccy, settings),
                "entry_high": fmt_price(plan.entry_high, ccy, settings),
                "sl": fmt_price(plan.stop_loss, ccy, settings),
                "sl_pct": fmt_pct(plan.stop_loss_pct),
                "tp": fmt_price(plan.take_profit, ccy, settings),
                "tp_pct": fmt_pct(plan.take_profit_pct),
                "tp2": fmt_price(plan.take_profit_2, ccy, settings),
                "tp2_pct": fmt_pct(plan.take_profit_2 / plan.entry_price - 1)
                if plan.take_profit_2 and plan.entry_price
                else MISSING,
                "rr": fmt_num(plan.reward_risk, 2),
                "shares": (
                    (f"{plan.shares:,} sh" if en else f"{plan.shares:,}주") if plan.shares is not None else MISSING
                ),
                "position_value": fmt_money_compact(plan.position_value, report.base_currency, settings),
                "kelly": _kelly_view(plan, en),
                "barrier": barrier,
                "action": plan.action or MISSING,
                "triggers": [
                    {"no": _circled(i), "text": tg} for i, tg in enumerate(plan.triggers[: (2 if level >= 3 else 5)])
                ],
                "triggers_hidden": max(0, len(plan.triggers) - (2 if level >= 3 else 5)),
                "triggers_more": (
                    f" +{max(0, len(plan.triggers) - (2 if level >= 3 else 5))} more"
                    if en
                    else f" 외 {max(0, len(plan.triggers) - (2 if level >= 3 else 5))}건"
                ),
                "chart": chart,
            }
        )

    alert_groups = [{"name": r["name"], "ticker": r["ticker"], "entries": r["alerts"]} for r in rows if r["alerts"]]
    # market- and portfolio-level problems belong in the page-1 box too: "✓ 경고 없음" must never be shown while
    # page 2/3 carry an alarm (e.g. a universe load failure, a collapse check, missing portfolio risk)
    diagnostics = _diagnostics_view(report, level, en)
    degraded_codes = [t.instrument.ticker for t in tickers if t.quant.distribution and t.quant.distribution.degraded]
    # whole-token match: a plain substring test let a degraded 1-letter ticker ("T", "C", "V") swallow any
    # market-level alarm containing that letter (e.g. "RSMT-Q ... collapse") from the page-1 box
    code_re = (
        re.compile(r"(?<![0-9A-Za-z])(?:" + "|".join(re.escape(c) for c in degraded_codes) + r")(?![0-9A-Za-z])")
        if degraded_codes
        else None
    )
    for dg in diagnostics:
        entries = [
            {"kind": "model", "text": ln.strip()} for ln in dg["alarms"] if code_re is None or not code_re.search(ln)
        ]  # per-ticker rows already say so
        if dg["n_served"] and not dg["has_lines"]:
            entries.append({"kind": "model", "text": "No model diagnostics recorded" if en else "모델 진단 기록 없음"})
        if entries:
            alert_groups.append(
                {
                    "name": f"RSMT-Q {dg['label']}",
                    "ticker": dg["market"],
                    "entries": entries,
                }
            )
    portfolio = _portfolio_view(report, settings)
    port_entries = []
    if portfolio["alloc_gap"]:
        port_entries.append(
            {
                "kind": "warning",
                "text": "Position weights + cash ≠ 100% — check the sizing"
                if en
                else "종목 비중 합계 + 현금 ≠ 100% — 사이징 결과 확인 필요",
            }
        )
    if report.cash_weight < -0.005:
        port_entries.append(
            {
                "kind": "warning",
                "text": f"Cash {fmt_prob(report.cash_weight, 1)} — gross above 100% (leverage), check the sizing"
                if en
                else f"현금 비중 {fmt_prob(report.cash_weight, 1)} — 총 투자비중 100% 초과(레버리지), 사이징 "
                f"결과 확인 필요",
            }
        )
    if not portfolio["has_risk"]:
        port_entries.append(
            {"kind": "warning", "text": "Portfolio risk not computed" if en else "포트폴리오 리스크 미산출"}
        )
    no_h = [
        t.instrument.ticker
        for t in tickers
        if t.execution.barrier is not None and t.execution.barrier.horizon_days <= 0
    ]
    if no_h:
        port_entries.append(
            {
                "kind": "model",
                "text": f"Exit-probability simulation horizon not recorded for {len(no_h)} names ({', '.join(no_h)}) "
                "— interpret the holding period with care"
                if en
                else f"청산 확률 시뮬레이션 기간 미기재 {len(no_h)}종목({', '.join(no_h)}) — 보유기간 해석 주의",
            }
        )
    if port_entries:
        alert_groups.append({"name": "Portfolio" if en else "포트폴리오", "ticker": "", "entries": port_entries})
    if en:
        kinds = {
            "critical": "Critical",
            "override": "Guardrail",
            "warning": "Data warning",
            "model": "Model/diagnostics",
            "llm": "LLM error",
            "notice": "LLM notice",
        }
    else:
        kinds = {
            "critical": "치명",
            "override": "가드레일",
            "warning": "데이터 경고",
            "model": "모델·진단",
            "llm": "LLM 오류",
            "notice": "LLM 알림",
        }
    alert_counts = []
    for k, label in kinds.items():
        n = sum(1 for g in alert_groups for a in g["entries"] if a["kind"] == k)
        if n:
            alert_counts.append({"kind": k, "label": label, "n": n})
    notes = [
        {
            "text": n,
            "cls": "sample" if ("샘플" in n or "sample" in n.lower()) else "note",
        }
        for n in report.notes
    ]
    notes += [{"text": f"{'All names' if en else '전 종목 공통'}: {o}", "cls": "note"} for o in run_skip_notes]
    if overflow_note:
        notes.insert(0, {"text": overflow_note, "cls": "over"})
    if info_note:
        notes.insert(0, {"text": info_note, "cls": "note"})
    compact_note = ""
    if level >= 1:
        omissions = COMPACT_OMISSIONS_EN if en else COMPACT_OMISSIONS
        omitted = [omissions[lv] for lv in range(1, level + 1) if not (lv == 1 and charts is None)]
        if en:
            compact_note = (
                f"Compacted to fit the page limit (level {level})"
                + (": " + " · ".join(omitted) if omitted else "")
                + " (warnings, guardrails and critical risks are never dropped)"
            )
        else:
            compact_note = (
                f"지면 한도 맞춤 압축 레벨 {level}"
                + (": " + " · ".join(omitted) if omitted else "")
                + " (경고·가드레일·치명 리스크는 생략하지 않음)"
            )
        notes.append({"text": compact_note, "cls": "warn"})

    h, H = settings.horizon_days, settings.fusion.target_horizon_days
    band_txt = fmt_pct(band_lim, 0, signed=False)
    n_alerts = sum(len(g["entries"]) for g in alert_groups)
    fm_rows = _factor_model_view(report, settings, en)
    b = settings.backtest.factor_pillar_blend
    order_v = _order_view(report, en)  # docs 93/94.3/96: reference only, also tagged on each Module 1 row
    ref = {(b["market"], tk): v for b in (order_v or {}).get("blocks", []) for tk, v in b["ref"].items()}
    avoid_v = _avoid_view(report, settings, en)  # docs 103.4/104: US reference flag, display only
    for r in rows:
        r["ref"] = ref.get((r["market"], r["ticker"]))
        r["avoid"] = (avoid_v or {}).get("ref", {}).get(r["ticker"]) if r["market"] == "US" else None
    return {
        "t": _texts(en),
        # protocol v4 forecasts carry the money metric: the selection-model column shows it instead of the AUC
        "sel_money": any(
            getattr(t.quant.excess_model, "oos_primary", None) is not None
            for t in report.tickers
            if t.quant.excess_model is not None
        ),
        "en": en,
        "title": REPORT_TITLE_EN if en else REPORT_TITLE,
        "subtitle": "Quant/ML signals + LLM text sentiment → final trade timing"
        if en
        else "정형 퀀트·ML 시그널 + 비정형 LLM 센티먼트 → 최종 매매 타이밍",
        "level": level,
        "as_of": as_of_txt,
        "generated": gen_kst.strftime("%Y-%m-%d %H:%M KST"),
        "universe": universe,
        "base_currency": report.base_currency,
        "capital": fmt_money_compact(report.capital, report.base_currency, settings) if report.capital else "",
        "digest": report.config_digest
        or f"{settings.digest()} ({'config at render time' if en else '렌더 시점 설정'})",
        "llm_model": ("skipped (--no-llm)" if en else "생략(--no-llm)")
        if all_skipped
        else settings.llm.model,  # the chip already says "LLM"
        "horizon": settings.horizon_days,
        "horizon_meta": f"{h} trading days · target {H} trading days" if en else f"{h}거래일 · 목표가 {H}거래일",
        "summary_sub": f"Invested vs cash · ex-ante risk ({h} trading days) · LLM cost"
        if en
        else f"투자 비중 vs 현금 · 사전 리스크({h}거래일) · LLM 비용",
        "alerts_title": f"⚠ Warnings · guardrails · degraded states: {n_alerts}"
        if en
        else f"⚠ 경고 · 가드레일 · 저하 상태 {n_alerts}건",
        "m2_sub": (
            "Factor scores (0–100, 50 = neutral) · ML direction = stock-selection model "
            f"({_direction_models(report, en)}) · RSMT-Q distribution ({h} trading days, risk and probabilities only)"
            if en
            else f"팩터 스코어(0–100, 50=중립) · ML 방향 = 종목 선택 모델({_direction_models(report, en)})"
            f" · RSMT-Q 분포 예측({h}거래일, 위험·확률 전용)"
        )
        if any(r["sel"] for r in rows)
        else (
            f"Factor scores (0–100, 50 = neutral) · ML (RSMT-Q direction) · RSMT-Q distribution ({h} trading days)"
            if en
            else f"팩터 스코어(0–100, 50=중립) · ML(RSMT-Q 방향) · RSMT-Q 분포 예측 ({h}거래일)"
        ),
        "sel_cols": any(r["sel"] for r in rows),
        "rsmt_sub": f"One path matrix → return · probabilities · band · VaR (served mean) · band axis ±{band_txt}"
        if en
        else f"하나의 경로 행렬 → 기대수익·확률·밴드·VaR (서빙 평균 기준) · 밴드 축 ±{band_txt}",
        "diag_more": "… +{n} more lines (CV / offset summary in the tiles above; full text in the JSON result)"
        if en
        else "… 외 {n}줄 (CV·오프셋 요약은 위 지표 참조, 전체는 JSON 결과)",
        "m4_sub_h": (f" (RSMT-Q paths · holding {barrier_h_txt})" if en else f" (RSMT-Q 경로 · 보유 {barrier_h_txt})")
        if barrier_h_txt
        else "",
        "barrier_h_missing": barrier_h_txt in (BARRIER_H_MISSING, BARRIER_H_MISSING_EN),
        "barrier_h_short": (f"{barrier_h}D" if en and barrier_h else barrier_h_txt),
        "macro": _macro_view(report, settings, en, level),
        "factor_model": fm_rows,
        "fm_sub": (
            (
                "sector-neutral PER · PBR · volume trend · 3M/6M momentum · sector, walk-forward weights · "
                f"factor pillar = {1 - b:.0%} legacy + {b:.0%} model"
                if en
                else "섹터 중립 PER·PBR·거래대금 추세·3M·6M 모멘텀·섹터, 워크포워드 가중 · "
                f"팩터 필러 = 기존 {1 - b:.0%} + 모델 {b:.0%}"
            )
            + _factor_model_names(report, en)
        )
        if fm_rows
        else "",
        "backtest": _backtest_view(report, settings, en),
        "order": order_v,
        "avoid": avoid_v,
        "kelly_cap": f"{settings.risk.kelly_cap:g}",
        "page_break_m2": level < 1,  # compaction lets Module 2 use page-1 space once charts are hidden
        "target_horizon": settings.fusion.target_horizon_days,
        "barrier_h_txt": barrier_h_txt,
        "regime": [{"k": k, "v": _regime_value_en(v) if en else v} for k, v in report.market_regime.items()],
        "notes": notes,
        "compact_note": compact_note,
        "rows": rows,
        "portfolio": portfolio,
        "alert_groups": alert_groups,
        "alert_counts": alert_counts,
        "n_alerts": n_alerts,
        "show_charts": level < 1 and charts is not None,
        "diagnostics": diagnostics,
        "band_lim": band_txt,
        "methodology": {
            **_methodology(settings),
            "sel": _methodology_sel(report, en),
            "extra": _methodology_extra(report, settings, en),
        },
        "disclaimer": DISCLAIMER_EN if en else DISCLAIMER,
        "headings": MODULE_HEADINGS,
        "css": Markup(
            _stylesheet(
                en or bool(report.macro) or bool(fm_rows) or bool(report.factor_backtest) or bool(report.order_ventile)
            )
        ),
    }


# ============================================================================ HTML


def _stylesheet(extended: bool) -> str:
    """report.css, plus report_ext.css (English, macro strip, factor model, backtest) only when needed, so a
    Korean report without those blocks inlines exactly the stylesheet it always did."""
    css = (TEMPLATE_DIR / "report.css").read_text(encoding="utf-8")
    if extended:
        css += "\n" + (TEMPLATE_DIR / "report_ext.css").read_text(encoding="utf-8")
    return css


def _jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = fmt_pct
    env.filters["prob"] = fmt_prob
    env.filters["num"] = fmt_num
    env.filters["w"] = lambda v: f"{max(0.0, min(100.0, float(v))):.2f}"  # CSS width percentages
    return env


def render_html(
    report: PortfolioReport,
    settings: Settings,
    compact_level: int = 0,
    charts: dict[str, str] | None = None,
    *,
    overflow_note: str | None = None,
    info_note: str | None = None,
) -> str:
    """Render the self-contained report HTML (inline CSS + inline SVG charts, no external assets).

    `charts` maps ticker -> inline SVG (see `charts.price_chart_svg`); tickers without a chart show a
    visible "차트 없음" placeholder. `compact_level` 0..3 trims content (see module docstring).
    `overflow_note` is printed on page 1 (used by `fit_to_pages` when the page limit cannot be met).
    """
    if not report.tickers:
        raise ValueError("render_html: report has no tickers")
    view = build_view(report, settings, compact_level, charts, overflow_note=overflow_note, info_note=info_note)
    return _jinja_env().get_template("report.html.j2").render(**view)


# ============================================================================ PDF

_PAGE_RE = re.compile(rb"/Type\s*/Page(?![a-zA-Z])")
MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


class NoBrowserError(RuntimeError):
    """No Chromium-family browser could be launched (message lists every attempt)."""


def count_pdf_pages(pdf: Path | bytes) -> int:
    """Page count via pypdf; regex over raw bytes as a fallback (Chrome/Skia PDFs have no object streams)."""
    data = bytes(pdf) if isinstance(pdf, bytes | bytearray) else Path(pdf).read_bytes()
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf not installed - counting PDF pages with a regex fallback")
        return len(_PAGE_RE.findall(data))
    return len(PdfReader(io.BytesIO(data)).pages)


def _launch(p: Any) -> Any:
    """Launch chain: Chrome channel -> executable path -> Playwright bundled chromium -> NoBrowserError."""
    from playwright.sync_api import Error as PlaywrightError

    channel = os.environ.get("QUANTCIO_BROWSER_CHANNEL", "chrome")
    attempts: list[tuple[str, dict[str, Any]]] = [(f"channel={channel}", {"channel": channel})]
    exe = os.environ.get("QUANTCIO_CHROME") or MAC_CHROME
    if Path(exe).exists():
        attempts.append((f"executable_path={exe}", {"executable_path": exe}))
    attempts.append(("bundled chromium-headless-shell", {}))
    errors = []
    for label, kw in attempts:
        try:
            browser = p.chromium.launch(headless=True, **kw)
            log.debug("PDF browser launched via %s", label)
            return browser
        except PlaywrightError as exc:
            errors.append(f"{label}: {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}")
    raise NoBrowserError(
        "No Chromium-family browser could be launched. Install Google Chrome, or run "
        "`python -m playwright install --only-shell chromium`.\n  " + "\n  ".join(errors)
    )


_FOOTER_STYLE = (
    "width:100%;font-size:7px;color:#8a8a8a;padding:0 12mm;"
    "font-family:'Apple SD Gothic Neo','Noto Sans KR','Noto Sans CJK KR','NanumGothic',sans-serif;"
    "display:flex;justify-content:space-between;"
)


def footer_template(report: PortfolioReport | None = None, lang: Any = None) -> str:
    """Chrome footer (page x / y) with the report date and config digest (`lang`: Settings / "ko" / "en")."""
    tail = " · For reference only · not investment advice" if is_en(lang) else " · 투자 참고용 · 투자 권유 아님"
    left = "QuantCIO" + tail
    if report is not None:
        gen = report.generated_at if report.generated_at.tzinfo else report.generated_at.replace(tzinfo=UTC)
        left = f"QuantCIO · {gen.astimezone(KST):%Y-%m-%d} · cfg {report.config_digest or MISSING}" + tail
    return (
        f'<div style="{_FOOTER_STYLE}"><span>{_html.escape(left)}</span>'
        '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>'
    )


def _driver_alive(pw: Any) -> bool:
    """False once the Playwright driver connection is gone (the driver process exited or was killed).

    After that a sync-API call can spin forever at 100% CPU (playwright 1.63 `_sync`: `while not task.done():
    dispatcher.switch()` on a dispatcher greenlet that has already finished), so the renderer must not touch that
    browser again. Unknown Playwright internals -> True (the previous behaviour)."""
    try:
        conn = pw._impl_obj._connection
        return not conn._dispatcher_fiber.dead and not conn._transport.on_error_future.done()
    except AttributeError:
        return True


def _browser_process_groups(pw: Any) -> set[int]:
    """Process groups of the browser this Playwright driver started (Chrome runs detached, in a group of its own),
    so a browser orphaned by a dead driver can be killed. Empty when unknown (or on Windows)."""
    if os.name == "nt":  # pragma: no cover - POSIX is the supported platform
        return set()
    try:
        from quantcio.report.worker import _process_table, descendants

        table = _process_table()
        own = os.getpgid(0)
        kids = descendants(pw._impl_obj._connection._transport._proc.pid, table)
        return {table[p][1] for p in kids if p in table and table[p][1] != own}
    except Exception:  # noqa: BLE001 - best effort; the render itself does not depend on it
        return set()


class PdfRenderer:
    """Reusable renderer: one Chrome process for many PDFs (~0.3 s per report when reused).

    Use as a context manager; `__enter__` raises `NoBrowserError` when no browser can start (and stops the
    Playwright driver first, so later renders in the same process still work). A driver that dies between or
    during renders is detected (`_driver_alive`): the dead connection is dropped without further calls (they
    would hang), a new browser is started and the render is retried once.
    """

    def __init__(self) -> None:
        self._pw: Any = None
        self._browser: Any = None
        self._browser_groups: set[int] = set()

    def __enter__(self) -> PdfRenderer:
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        try:
            self._browser = _launch(self._pw)
        except BaseException:
            self._pw.stop()  # __exit__ is not called when __enter__ raises
            self._pw = None
            raise
        self._browser_groups = _browser_process_groups(self._pw)
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            if self._browser is not None and _driver_alive(self._pw):
                self._browser.close()
        finally:
            self._browser = None
            if self._pw is not None:
                self._pw.stop()  # safe on a dead connection: it only closes the pipe and runs no protocol call
                self._pw = None

    @property
    def active(self) -> bool:
        return self._browser is not None and _driver_alive(self._pw)

    def _restart(self, when: str) -> None:
        """Replace a renderer whose driver died: kill the browser it left behind, drop the dead connection, start
        a new driver and browser."""
        log.warning("PDF 렌더러: Playwright 드라이버 연결이 끊김(%s) - 브라우저를 새로 시작", when)
        for g in sorted(self._browser_groups):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(g, signal.SIGKILL)
        self._browser_groups = set()
        self.__exit__(None, None, None)
        self.__enter__()

    def pdf_bytes(
        self,
        html: str,
        *,
        scale: float = 1.0,
        footer: bool = True,
        footer_html: str | None = None,
        base_dir: Path | None = None,
        timeout_ms: int = 30_000,
    ) -> bytes:
        if self._browser is None:
            raise RuntimeError("PdfRenderer is not started; use `with PdfRenderer() as r:`")
        kw = {
            "scale": scale,
            "footer": footer,
            "footer_html": footer_html,
            "base_dir": base_dir,
            "timeout_ms": timeout_ms,
        }
        if not _driver_alive(self._pw):
            self._restart("렌더 전")
        try:
            return self._pdf_once(html, **kw)
        except Exception:
            if _driver_alive(self._pw):
                raise
            self._restart("렌더 중")
            return self._pdf_once(html, **kw)

    def _pdf_once(
        self,
        html: str,
        *,
        scale: float = 1.0,
        footer: bool = True,
        footer_html: str | None = None,
        base_dir: Path | None = None,
        timeout_ms: int = 30_000,
    ) -> bytes:
        page = self._browser.new_page()
        try:
            page.set_default_timeout(timeout_ms)
            page.emulate_media(media="print")
            if base_dir is None:
                page.set_content(html, wait_until="load")
            else:
                with tempfile.NamedTemporaryFile(
                    "w", suffix=".html", dir=base_dir, delete=False, encoding="utf-8"
                ) as f:
                    f.write(html)
                    tmp = Path(f.name)
                try:
                    page.goto(tmp.as_uri(), wait_until="load")
                finally:
                    tmp.unlink(missing_ok=True)
            page.evaluate("document.fonts.ready.then(() => true)")
            return page.pdf(
                format="A4",
                print_background=True,
                prefer_css_page_size=True,  # @page { size: A4; margin } in report.css is the source of truth
                margin={
                    "top": "12mm",
                    "bottom": "14mm",
                    "left": "12mm",
                    "right": "12mm",
                },
                scale=scale,
                display_header_footer=footer,
                header_template="<span></span>",
                footer_template=(footer_html or footer_template()) if footer else "<span></span>",
                outline=True,
                tagged=True,
            )
        finally:
            if _driver_alive(self._pw):  # on a dead connection close() would never return
                page.close()


def save_html_fallback(html: str, out_path: Path) -> Path:
    """Write `out_path.with_suffix('.html')` (open in a browser -> Print -> Save as PDF keeps A4)."""
    p = Path(out_path).with_suffix(".html")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p


def _outside_event_loop[T](fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run `fn` directly, or in a worker thread when called from inside a running asyncio loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn(*args, **kwargs)
    # an asyncio loop (or an active sync-Playwright session) owns this thread; the sync API cannot run here.
    # Output is unaffected - only the caller's loop blocks while we wait (prefer asyncio.to_thread).
    log.info("running event loop detected in this thread - rendering the PDF in a worker thread")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn, *args, **kwargs).result()


def _render_pdf_impl(
    html: str,
    out_path: Path,
    renderer: PdfRenderer | None,
    scale: float,
    footer_html: str | None,
    html_fallback: bool,
) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if renderer is not None and renderer.active:
            data = renderer.pdf_bytes(html, scale=scale, footer_html=footer_html)
        else:
            with PdfRenderer() as r:
                data = r.pdf_bytes(html, scale=scale, footer_html=footer_html)
    except NoBrowserError as exc:
        if not html_fallback:
            raise
        p = save_html_fallback(html, out_path)
        log.warning(
            "PDF 생성 불가 - HTML 대체 파일을 저장했습니다 (%s): %s",
            p,
            str(exc).splitlines()[0],
        )
        return 0
    out_path.write_bytes(data)
    return count_pdf_pages(data)


def render_pdf(
    html: str,
    out_path: Path | str,
    *,
    renderer: PdfRenderer | None = None,
    scale: float = 1.0,
    footer_template: str | None = None,
    html_fallback: bool = True,
) -> int:
    """Render HTML to an A4 PDF at `out_path` and return its page count.

    When no browser can be launched and `html_fallback` is true, the HTML is written to
    `out_path.with_suffix('.html')`, a WARNING is logged and 0 is returned (callers must treat 0 as
    "no PDF"). Pass an active `PdfRenderer` to reuse one Chrome process across renders.
    """
    if renderer is not None and renderer.active:  # a live renderer is bound to the calling thread
        return _render_pdf_impl(html, Path(out_path), renderer, scale, footer_template, html_fallback)
    return _outside_event_loop(
        _render_pdf_impl,
        html,
        Path(out_path),
        None,
        scale,
        footer_template,
        html_fallback,
    )


@dataclass
class FitResult:
    """Outcome of `fit_to_pages` - every degraded path is explicit here and logged."""

    pdf_path: Path | None  # None when the HTML fallback was used
    html_path: Path | None  # final HTML (always written on fallback; optional otherwise)
    pages: int  # 0 on HTML fallback
    compact_level: int
    scale: float
    fits: bool  # pages <= max_pages (False on fallback); in the extended format: pages <= max_pages_extended
    fallback_html: bool
    attempts: list[tuple[int, float, int]] = field(default_factory=list)  # (level, scale, pages)
    warnings: list[str] = field(default_factory=list)
    extended: bool = False  # large universe printed in full on more pages (report.auto_extend)


DEFAULT_SCALES = (1.0, 0.95, 0.9)
LAST_RESORT_SCALES = (0.85, 0.8)


def _fit_impl(
    report: PortfolioReport,
    settings: Settings,
    out_path: Path,
    charts: dict[str, str] | None,
    max_pages: int,
    renderer: PdfRenderer | None,
    scales: tuple[float, ...],
    last_resort_scales: tuple[float, ...],
    html_fallback: bool,
    write_html: bool,
    on_attempt: Callable[[int, float, int], None] | None = None,
) -> FitResult:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    footer_html = footer_template(report, settings)
    en = is_en(settings)
    plan = [(lvl, s) for lvl in range(MAX_COMPACT_LEVEL + 1) for s in scales]
    plan += [(MAX_COMPACT_LEVEL, s) for s in last_resort_scales]
    html_cache: dict[int, str] = {}

    def html_for(level: int) -> str:
        if level not in html_cache:
            html_cache[level] = render_html(report, settings, level, charts)
        return html_cache[level]

    attempts: list[tuple[int, float, int]] = []
    warnings: list[str] = []

    def overflow_text(pages: int, level: int, scale: float) -> str:
        if en:
            return (
                f"Too many names for the {max_pages}-page limit — {len(report.tickers)} names are still {pages} "
                f"pages at maximum compaction (level {level}, scale {scale:.2f}). Reduce the number of names or raise "
                "report.max_pages (warnings and guardrails are never dropped)"
            )
        return (
            f"종목 수가 많아 {max_pages}페이지 제한 초과 — {len(report.tickers)}종목, 최대 압축(레벨 {level}·배율 "
            f"{scale:.2f})"
            f"에서도 {pages}페이지. 종목 수를 줄이거나 report.max_pages를 늘리세요 (경고·가드레일은 생략하지 않음)"
        )

    n_names = len(report.tickers)
    rep_cfg = settings.report
    extended_budget = max(int(max_pages), int(getattr(rep_cfg, "max_pages_extended", max_pages)))
    extend = bool(getattr(rep_cfg, "auto_extend", False)) and n_names > int(getattr(rep_cfg, "compact_max_names", 99))
    extended_used = False

    def extend_text(pages: int) -> str:
        cmn = int(rep_cfg.compact_max_names)
        if en:
            return (
                f"{n_names} names — printed in full on {pages} pages instead of the {max_pages}-page one-minute format "
                f"(up to {cmn} names; report.auto_extend). Nothing is omitted: start with the Signal Matrix on page 1"
            )
        return (
            f"종목 {n_names}개 — 1분 판단용 {max_pages}쪽 형식(종목 {cmn}개 이하) 대신 "
            f"내용 생략 없이 {pages}쪽으로 출력(report.auto_extend). 1쪽 시그널 매트릭스부터 보세요"
        )

    def run_extended(r: PdfRenderer) -> tuple[bytes, int, int, float, str] | None:
        """Full content (level 0) at the print scale with the fewest pages (ties -> the larger scale); None when
        even that exceeds the extended page cap (the compaction plan then takes over)."""
        best: tuple[bytes, int, float] | None = None
        for scale in scales:
            data = r.pdf_bytes(html_for(0), scale=scale, footer_html=footer_html)
            n = count_pdf_pages(data)
            attempts.append((0, scale, n))
            if on_attempt is not None:
                on_attempt(0, scale, n)
            if best is None or n < best[1]:
                best = (data, n, scale)
        assert best is not None
        _, n, scale = best
        if n > extended_budget:
            return None
        stated = n
        for _ in range(3):  # the note adds a line: re-render until the page count it states is the PDF's
            html = render_html(report, settings, 0, charts, info_note=extend_text(stated))
            data = r.pdf_bytes(html, scale=scale, footer_html=footer_html)
            n = count_pdf_pages(data)
            if n == stated:
                break
            stated = n
        return data, n, 0, scale, html

    def run(r: PdfRenderer) -> tuple[bytes, int, int, float, str]:
        nonlocal extended_used
        if extend:
            done = run_extended(r)
            if done is not None:
                extended_used = True
                return done
        # first attempt that fits wins; if none fits keep the one with the FEWEST pages (ties -> the earliest,
        # i.e. the fullest content at the largest print scale) - not simply the last attempt
        best: tuple[bytes, int, int, float] | None = None
        for level, scale in plan:
            data = r.pdf_bytes(html_for(level), scale=scale, footer_html=footer_html)
            n = count_pdf_pages(data)
            attempts.append((level, scale, n))
            if on_attempt is not None:
                on_attempt(level, scale, n)
            if best is None or n < best[1]:
                best = (data, n, level, scale)
            if n <= max_pages:
                break
        assert best is not None
        data, n, level, scale = best
        if n <= max_pages:
            return data, n, level, scale, html_for(level)
        # nothing fits: the document itself must say so (page 1 note), not only FitResult.warnings. The note
        # adds a line, so re-render until the page count printed in it matches the PDF it sits in.
        stated = n
        for _ in range(3):
            html = render_html(
                report,
                settings,
                level,
                charts,
                overflow_note=overflow_text(stated, level, scale),
            )
            data = r.pdf_bytes(html, scale=scale, footer_html=footer_html)
            n = count_pdf_pages(data)
            if n == stated:
                break
            stated = n
        else:
            log.warning(
                "초과 안내 문구의 쪽수(%d)와 실제 PDF 쪽수(%d)가 수렴하지 않음",
                stated,
                n,
            )
        return data, n, level, scale, html

    try:
        if renderer is not None and renderer.active:
            data, pages, level, scale, final_html = run(renderer)
        else:
            with PdfRenderer() as r:
                data, pages, level, scale, final_html = run(r)
    except NoBrowserError as exc:
        if not html_fallback:
            raise
        p = save_html_fallback(html_for(0), out_path)
        msg = (
            f"PDF not generated (browser failed to start) - HTML fallback saved: {p} ({str(exc).splitlines()[0]})"
            if en
            else f"PDF 생성 불가(브라우저 실행 실패) - HTML 대체 파일 저장: {p} ({str(exc).splitlines()[0]})"
        )
        log.warning(msg)
        return FitResult(
            pdf_path=None,
            html_path=p,
            pages=0,
            compact_level=0,
            scale=0.0,
            fits=False,
            fallback_html=True,
            attempts=attempts,
            warnings=[msg],
        )

    out_path.write_bytes(data)
    html_path = None
    if write_html:
        html_path = out_path.with_suffix(".html")
        html_path.write_text(final_html, encoding="utf-8")
    fits = pages <= (extended_budget if extended_used else max_pages)
    if extended_used:
        msg = (
            f"[info] {n_names} names printed in full on {pages} pages (extended format, report.auto_extend; the "
            f"{max_pages}-page format is for up to {settings.report.compact_max_names} names)"
            if en
            else f"[정보] 종목 {n_names}개 — 내용 생략 없이 {pages}쪽으로 출력(확장 형식, report.auto_extend; "
            f"{max_pages}쪽 형식은 {settings.report.compact_max_names}개 이하)"
        )
        log.info(msg)
        warnings.append(msg)
    elif not fits:
        msg = (
            f"{max_pages}-page limit exceeded at every compaction step - saved at {pages} pages (level {level}, "
            f"scale {scale:.2f}); reduce the number of names (the report states this on page 1)"
            if en
            else f"모든 압축 단계에서 {max_pages}쪽 한도 초과 - "
            f"최소 {pages}쪽(레벨 {level}, 배율 {scale:.2f})으로 저장, "
            "종목 수를 줄이세요 (리포트 1쪽에 초과 안내 표시)"
        )
        log.warning(msg)
        warnings.append(msg)
    elif level > 0 or scale < 1.0:
        msg = (
            f"compaction level {level} and print scale {scale:.2f} applied to fit the {max_pages}-page limit"
            if en
            else f"{max_pages}쪽 한도를 맞추기 위해 압축 레벨 {level}, 인쇄 배율 {scale:.2f} 적용"
        )
        log.info(msg)
        warnings.append(msg)
    return FitResult(
        pdf_path=out_path,
        html_path=html_path,
        pages=pages,
        compact_level=level,
        scale=scale,
        fits=fits,
        fallback_html=False,
        attempts=attempts,
        warnings=warnings,
        extended=extended_used,
    )


def fit_to_pages(
    report: PortfolioReport,
    settings: Settings,
    out_path: Path | str,
    *,
    charts: dict[str, str] | None = None,
    max_pages: int | None = None,
    renderer: PdfRenderer | None = None,
    scales: tuple[float, ...] = DEFAULT_SCALES,
    last_resort_scales: tuple[float, ...] = LAST_RESORT_SCALES,
    html_fallback: bool = True,
    write_html: bool = True,
    on_attempt: Callable[[int, float, int], None] | None = None,
) -> FitResult:
    """Render the report as a PDF of at most `max_pages` (default settings.report.max_pages).

    In-process: a crashed Playwright driver can hang the calling process. Production callers use
    `quantcio.report.worker.fit_to_pages_isolated` (this function in a watchdog subprocess); `on_attempt(level,
    scale, pages)` is called after every compaction attempt (the worker streams them to its parent).

    Tries compact levels 0..3, each at print scales `scales` (full content at a slightly smaller scale is
    preferred over dropping content), then `last_resort_scales` at level 3. The first attempt that fits
    is written. If nothing fits, the attempt with the fewest pages (earliest on ties) is re-rendered with
    an explicit page-1 note ("종목 수가 많아 N페이지 제한 초과 — ...") and written; `fits=False` plus a
    warning report it as well. Without a browser the level-0 HTML is written instead
    (`fallback_html=True`, pages=0, warning).
    """
    if not scales:
        raise ValueError("scales must not be empty")
    limit = int(max_pages if max_pages is not None else settings.report.max_pages)
    if limit < 1:
        raise ValueError(f"max_pages must be >= 1, got {limit}")
    args = (report, settings, Path(out_path), charts, limit)
    if renderer is not None and renderer.active:
        return _fit_impl(
            *args,
            renderer,
            tuple(scales),
            tuple(last_resort_scales),
            html_fallback,
            write_html,
            on_attempt,
        )
    return _outside_event_loop(
        _fit_impl,
        *args,
        None,
        tuple(scales),
        tuple(last_resort_scales),
        html_fallback,
        write_html,
        on_attempt,
    )
