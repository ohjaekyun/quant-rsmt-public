"""Deterministic SAMPLE PortfolioReport - fake but internally consistent - for tests and `quantcio doctor`.

Nothing here touches the network and nothing here is market data: every price path, score and text is
synthetic and the report carries `SAMPLE_NOTE` in `PortfolioReport.notes` (rendered on page 1).

Consistency is by construction rather than by hand-typed numbers:

- A synthetic benchmark + stock price path is generated per ticker (seeded by ticker). Technicals
  (MA, RSI, ATR, vol, beta, drawdown, returns) and factor scores are computed from that path with the
  documented monotone mappings (logistic / thresholds), so e.g. a high momentum score comes with a high
  12-1M return in the rationale.
- ONE simulated daily path matrix per ticker (Student-t daily steps, r = beta*m + e, shared market draws
  per market) of max(horizon_days, fusion.target_horizon_days) days: column horizon_days-1 yields P(up),
  P(beat benchmark), quantiles and VaR/CVaR; the first target_horizon_days columns yield the barrier
  outcome (TP-first / time / SL-first summing to exactly 1, `horizon_days` recorded) and Kelly.
- Portfolio VaR / CVaR / P(loss) are normal approximations (like fusion.risk.portfolio_risk, which the
  report labels "정규근사"), on the covariance of the simulated horizon returns.
- Pillars, composite, confidence, signal thresholds, guardrails (incl. fusion.barrier_guard_margin),
  target price, execution plan, triggers (incl. the "실적발표 D-3" one) and sizing follow the formulas in
  docs/ARCHITECTURE.md (simplified where a module-level detail is irrelevant for rendering). The sample
  itself does not import `quantcio.quant` / `quantcio.fusion`, so it works even when those modules are
  broken - which is exactly when `doctor` needs it.
- Prices in trigger text use the fusion/report format: KRW '286,500원', USD '$337.02' (English: '₩286,500').
- Language: with settings.report.language == "en" every text (names, news, filings, risks, thesis, notes,
  overrides, actions, triggers, diagnostics) comes from report/sample_en.py and the English templates below; the
  numbers are identical in both languages.
- `with_models=True` (default) also fills the new data contracts (docs/BACKTEST_MACRO.md): a PIT multi-factor
  model score per ticker (blended into the factor pillar exactly like fusion/signal.py), global/US/Korea macro
  regimes with the overlay applied to sizing, and a KR/US factor-backtest summary. `with_models=False` builds the
  sample exactly as before those contracts existed.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd

from quantcio.config import Settings, load_settings
from quantcio.i18n import is_en
from quantcio.models import (
    BarrierOutcome,
    CriticReview,
    DistributionForecast,
    ExecutionPlan,
    ExecutiveSignal,
    FactorModelScore,
    FactorScore,
    FilingIntelligence,
    Fundamentals,
    Instrument,
    LLMIntelligence,
    MacroIndicator,
    MacroRegime,
    Market,
    NewsSentiment,
    PortfolioReport,
    QuantSignal,
    RiskFactor,
    Signal,
    Technicals,
    TickerReport,
)
from quantcio.report.sample_en import SAMPLE_TEXT_EN

__all__ = [
    "SAMPLE_NOTE",
    "SAMPLE_NOTE_EN",
    "SAMPLE_TICKERS",
    "build_sample_report",
    "build_sample_charts",
    "sample_price_history",
]

SAMPLE_NOTE = "샘플 데이터 — 렌더링·배포 점검용 가상 수치이며 실제 시세나 투자 권유가 아닙니다."
SAMPLE_NOTE_EN = (
    "Sample data — synthetic figures for rendering and deployment checks; not real quotes or investment advice."
)

AS_OF = date(2026, 9, 23)
GENERATED_AT = datetime(2026, 9, 23, 22, 40, tzinfo=UTC)  # 2026-09-24 07:40 KST
HISTORY_DAYS = 520  # enough for MA200 over the whole 1y chart window
HORIZON = 20  # trading days (settings.horizon_days default)
N_PATHS = 4000
USDKRW = 1385.0
CAPITAL_KRW = 1_000_000_000.0
TRADING_DAYS = 252


# ----------------------------------------------------------------------------- static sample universe


@dataclass(frozen=True)
class _Bench:
    id: str
    name: str
    vol: float  # annualised
    segments: tuple[tuple[int, float], ...]  # recent (days, annual log drift); the rest uses base
    base: float
    seed: int


_BENCHES: dict[str, _Bench] = {
    "KOSPI": _Bench("KOSPI", "KOSPI", 0.16, ((120, 0.22), (20, 0.40)), 0.08, 11),
    "KOSDAQ": _Bench("KOSDAQ", "KOSDAQ", 0.22, ((120, 0.10), (20, -0.25)), 0.03, 12),
    "SPY": _Bench("SPY", "S&P 500 (SPY)", 0.15, ((120, 0.14), (20, 0.15)), 0.10, 13),
}

# Per-market RSMT-Q leaves: id -> (path, n, leaf mu_e (20D log), nu)
_LEAVES: dict[Market, dict[int, tuple[tuple[str, ...], int, float, float]]] = {
    Market.KR: {
        1: (("vol_ratio_20_120 <= 1.12", "mkt_ret_60 <= 0.031"), 1812, 0.002, 6.2),
        2: (("vol_ratio_20_120 <= 1.12", "mkt_ret_60 > 0.031"), 1593, 0.009, 8.0),
        3: (("vol_ratio_20_120 > 1.12", "dd_250 <= -0.18"), 812, -0.011, 5.1),
        4: (("vol_ratio_20_120 > 1.12", "dd_250 > -0.18"), 993, 0.004, 7.4),
    },
    Market.US: {
        1: (("mkt_vol_60 <= 0.162", "rsi14 <= 68.5"), 2140, 0.004, 7.8),
        2: (("mkt_vol_60 <= 0.162", "rsi14 > 68.5"), 1188, -0.002, 6.6),
        3: (("mkt_vol_60 > 0.162",), 1422, 0.001, 5.4),
    },
}

_MARKET_MODEL: dict[Market, dict[str, Any]] = {
    Market.KR: {
        "gamma": 0.0071,
        "eta": 0.93,
        "delta": 0.06,
        "delta_rel": 0.03,
        "mu_m": 0.004,
        "nu_m": 7.0,
        "cv": {
            "auc": 0.547,
            "base_auc": 0.500,
            "logloss": 0.6871,
            "base_logloss": 0.6927,
            "brier": 0.2468,
            "base_brier": 0.2497,
            "crps": 0.0412,
            "base_crps": 0.0421,
            "cover80": 0.79,
            "base_cover80": 0.80,
            "ic": 0.071,
            "base_ic": 0.0,
            "n_folds": 4.0,
            # robust skill gate (rsmt.robust_cv_folds): pass flag per fold layout + counts
            "gate_folds_3": 1.0,
            "gate_folds_4": 1.0,
            "gate_folds_5": 1.0,
            "gate_folds_n": 3.0,
            "gate_folds_pass": 3.0,
            "gate_folds_need": 2.0,
        },
    },
    Market.US: {
        "gamma": 0.0048,
        "eta": 0.88,
        "delta": 0.04,
        "delta_rel": 0.02,
        "mu_m": 0.005,
        "nu_m": 6.0,
        "cv": {
            "auc": 0.531,
            "base_auc": 0.500,
            "logloss": 0.6902,
            "base_logloss": 0.6921,
            "brier": 0.2486,
            "base_brier": 0.2494,
            "crps": 0.0368,
            "base_crps": 0.0372,
            "cover80": 0.81,
            "base_cover80": 0.80,
            "ic": 0.048,
            "base_ic": 0.0,
            "n_folds": 4.0,
            "gate_folds_3": 1.0,
            "gate_folds_4": 0.0,  # one layout misses: still a majority (2 of 3)
            "gate_folds_5": 1.0,
            "gate_folds_n": 3.0,
            "gate_folds_pass": 2.0,
            "gate_folds_need": 2.0,
        },
    },
}


@dataclass(frozen=True)
class _Spec:
    ticker: str
    name: str
    market: Market
    exchange: str
    sector: str
    benchmark: str
    price: float
    vol: float
    beta: float
    base_drift: float
    segments: tuple[tuple[int, float], ...]
    fund: dict[str, Any]
    leaf: int
    z_mom: float
    news: dict[str, Any] | None
    filings: dict[str, Any] | None
    thesis: str
    critic_adj: int = 0
    contradictions: tuple[str, ...] = ()
    llm_errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    degraded: tuple[str, ...] = ()
    extra_triggers: tuple[str, ...] = ()
    next_earnings: date | None = None  # Fundamentals.next_earnings_date
    earnings_estimate: bool | None = None  # Fundamentals.next_earnings_is_estimate
    llm_notices: tuple[str, ...] = ()  # LLMIntelligence.notices (degraded-but-served paths)


def _risk(title: str, detail: str, severity: str, category: str, ref: str) -> dict[str, str]:
    return {
        "title": title,
        "detail": detail,
        "severity": severity,
        "category": category,
        "source_ref": ref,
    }


_SPECS: list[_Spec] = [
    _Spec(
        "005930",
        "삼성전자",
        Market.KR,
        "KOSPI",
        "반도체",
        "KOSPI",
        286_500,
        0.34,
        1.15,
        0.12,
        ((150, 0.55), (60, 0.35), (21, 0.30)),
        {
            "per": 14.2,
            "forward_per": 10.8,
            "pbr": 2.1,
            "roe": 0.148,
            "operating_margin": 0.17,
            "revenue_growth_yoy": 0.123,
            "earnings_growth_yoy": 1.452,
            "dividend_yield": 0.012,
            "sector_per": 16.5,
            "analyst_target_price": 340_000,
            "analyst_target_high": 400_000,
            "analyst_count": 32,
            "analyst_rating": "4.05 매수",
        },
        leaf=2,
        z_mom=1.1,
        news={
            "score": 0.42,
            "n_articles": 24,
            "n_used": 17,
            "trend_7d": 0.12,
            "key_positives": [
                "HBM4 12단 엔비디아 품질 테스트 통과, 4분기 공급 본격화",
                "3분기 영업이익 10조원대 회복 전망",
            ],
            "key_negatives": ["파운드리 2나노 수율 부진 지속"],
            "key_events": ["자사주 3조원 추가 매입 결정"],
        },
        filings={
            "summary": [
                "3분기 잠정 영업이익 10.4조원, 전년동기 대비 145.2% 증가",
                "메모리 가격 상승과 HBM 비중 확대가 이익 회복을 견인",
                "자사주 3조원 추가 매입으로 주주환원 확대",
            ],
            "earnings_highlights": [
                "매출액 79조 1,000억원 (+12.3% YoY)",
                "영업이익 10조 4,000억원 (+145.2% YoY)",
            ],
            "guidance": "none",
            "tone": 0.45,
            "risk_factors": [
                _risk(
                    "美 반도체 수출규제 강화",
                    "대중 장비·HBM 수출 추가 제한 검토 보도",
                    "high",
                    "legal_regulatory",
                    "뉴스: 美 반도체 수출 규제 강화 우려",
                ),
                _risk(
                    "파운드리 수율·수주 지연",
                    "2나노 수율 부진으로 대형 고객 수주 지연 가능성",
                    "medium",
                    "competition",
                    "뉴스: 파운드리 수율 부진",
                ),
                _risk(
                    "노조 파업",
                    "임금협상 결렬로 파업 예고",
                    "low",
                    "operational",
                    "뉴스: 노조 파업 예고",
                ),
            ],
            "sources": ["연결재무제표기준영업(잠정)실적(공정공시)", "자기주식취득결정"],
        },
        thesis="HBM4 공급 본격화와 메모리 가격 반등으로 이익 모멘텀이 강하고 퀀트·텍스트 신호가 같은 방향이다. "
        "수출규제 뉴스에 따른 단기 변동성은 분할 진입으로 대응한다.",
        next_earnings=date(2026, 10, 7),
        earnings_estimate=True,  # 3Q 잠정실적 (early October, estimated)
    ),
    _Spec(
        "AAPL",
        "Apple",
        Market.US,
        "NASDAQ",
        "Technology",
        "SPY",
        337.02,
        0.24,
        1.05,
        0.14,
        ((120, 0.10), (60, 0.05), (21, 0.04)),
        {
            "per": 38.5,
            "forward_per": 33.1,
            "pbr": 55.0,
            "roe": 1.45,
            "operating_margin": 0.31,
            "revenue_growth_yoy": 0.06,
            "earnings_growth_yoy": 0.09,
            "dividend_yield": 0.0045,
            "sector_per": 29.0,
            "analyst_target_price": 350.0,
            "analyst_target_high": 400.0,
            "analyst_count": 45,
            "analyst_rating": "Buy",
        },
        leaf=2,
        z_mom=0.1,
        news={
            "score": 0.12,
            "n_articles": 31,
            "n_used": 14,
            "trend_7d": -0.05,
            "key_positives": ["아이폰 18 초기 판매량 전작 대비 증가"],
            "key_negatives": [
                "중국 스마트폰 점유율 하락",
                "앱스토어 반독점 항소심 판결 임박",
            ],
            "key_events": ["AI 기능 순차 출시"],
        },
        filings={
            "summary": [
                "서비스 매출 사상 최대, 총이익률 47.1%로 개선",
                "관세 관련 비용 분기 11억 달러 반영",
                "자사주 매입 250억 달러 집행, 배당 4% 인상",
            ],
            "earnings_highlights": [
                "Revenue $94.9 billion (+6% YoY)",
                "EPS $1.64 (+12% YoY)",
            ],
            "guidance": "maintained",
            "tone": 0.15,
            "risk_factors": [
                _risk(
                    "앱스토어 반독점 소송",
                    "항소심 패소 시 수수료 구조 변경 불가피",
                    "high",
                    "legal_regulatory",
                    "10-Q Item 1A",
                ),
                _risk(
                    "중국 수요 둔화",
                    "현지 브랜드 경쟁 심화로 중화권 매출 감소",
                    "medium",
                    "market_demand",
                    "10-Q MD&A",
                ),
                _risk(
                    "관세 비용",
                    "대중 관세로 원가 부담 지속",
                    "medium",
                    "macro_fx",
                    "8-K EX-99.1",
                ),
            ],
            "sources": ["10-Q (2026-08-01)", "8-K Item 2.02 EX-99.1"],
        },
        thesis="퀄리티는 최상위이나 밸류에이션 부담과 ML 스킬 미달로 방향성 우위가 약하다. 반독점 판결 이후 "
        "변동성을 확인하고 재평가한다.",
        next_earnings=date(2026, 10, 29),
        earnings_estimate=False,
        llm_notices=("critic: 서버 폴백 경로가 사용됨(usage.iterations 참조)",),
    ),
    _Spec(
        "000660",
        "SK하이닉스",
        Market.KR,
        "KOSPI",
        "반도체",
        "KOSPI",
        612_000,
        0.46,
        1.35,
        0.20,
        ((150, 0.75), (60, 0.60), (21, 0.55)),
        {
            "per": 9.8,
            "forward_per": 7.2,
            "pbr": 3.2,
            "roe": 0.36,
            "operating_margin": 0.46,
            "revenue_growth_yoy": 0.58,
            "earnings_growth_yoy": 0.92,
            "dividend_yield": 0.004,
            "sector_per": 16.5,
            "analyst_target_price": 760_000,
            "analyst_target_high": 900_000,
            "analyst_count": 30,
            "analyst_rating": "4.20 매수",
        },
        leaf=2,
        z_mom=1.8,
        news={
            "score": 0.58,
            "n_articles": 22,
            "n_used": 16,
            "trend_7d": 0.08,
            "key_positives": [
                "HBM4 양산 출하 개시, 점유율 50% 이상 유지 전망",
                "3분기 영업이익 컨센서스 상회 예상",
            ],
            "key_negatives": ["메모리 업황 고점 논란 재점화"],
            "key_events": ["청주 M15X 팹 조기 가동"],
        },
        filings={
            "summary": [
                "2분기 영업이익 9.2조원, 영업이익률 46%로 사상 최대",
                "HBM 매출 비중 40% 돌파, 고부가 제품 믹스 개선",
                "M15X 신규 팹 투자 확대로 CAPEX 계획 상향",
            ],
            "earnings_highlights": [
                "매출액 20조 1,000억원 (+58% YoY)",
                "영업이익 9조 2,000억원 (+92% YoY)",
            ],
            "guidance": "raised",
            "tone": 0.6,
            "risk_factors": [
                _risk(
                    "메모리 사이클 고점 우려",
                    "하반기 범용 D램 가격 상승폭 둔화 가능성",
                    "medium",
                    "market_demand",
                    "증권사 리서치 요약",
                ),
                _risk(
                    "고객 집중도",
                    "HBM 매출의 상당 부분이 단일 고객에 집중",
                    "medium",
                    "operational",
                    "반기보고서",
                ),
                _risk(
                    "CAPEX 부담",
                    "신규 팹 투자로 잉여현금흐름 변동성 확대",
                    "low",
                    "financial",
                    "IR 자료",
                ),
            ],
            "sources": ["영업(잠정)실적(공정공시)", "반기보고서"],
        },
        thesis="HBM 리더십과 실적 가속이 확인되어 모멘텀·퀄리티·텍스트 신호가 모두 강하다. 업황 고점 논란에 대비해 "
        "익절가 도달 시 50% 차익실현을 권고한다.",
        critic_adj=2,
        llm_notices=("news: 근거 인용이 원문과 불일치한 기사 2건 점수에서 제외",),
    ),
    _Spec(
        "NVDA",
        "NVIDIA",
        Market.US,
        "NASDAQ",
        "Semiconductors",
        "SPY",
        225.51,
        0.48,
        1.60,
        0.30,
        ((150, 0.45), (60, 0.40), (21, 0.35)),
        {
            "per": 48.0,
            "forward_per": 31.0,
            "pbr": 38.0,
            "roe": 0.92,
            "operating_margin": 0.61,
            "revenue_growth_yoy": 0.56,
            "earnings_growth_yoy": 0.61,
            "dividend_yield": 0.0002,
            "sector_per": 32.0,
            "analyst_target_price": 270.0,
            "analyst_target_high": 330.0,
            "analyst_count": 58,
            "analyst_rating": "Strong Buy",
        },
        leaf=1,
        z_mom=1.5,
        news={
            "score": 0.35,
            "n_articles": 40,
            "n_used": 22,
            "trend_7d": -0.06,
            "key_positives": [
                "차세대 루빈 GPU 조기 출하 확인",
                "데이터센터 매출 가이던스 상향",
            ],
            "key_negatives": [
                "중국향 H20 판매 제한 장기화",
                "고객사 자체 AI 칩 경쟁 심화",
            ],
            "key_events": ["분기 실적 발표 후 시간외 3% 상승"],
        },
        filings={
            "summary": [
                "데이터센터 매출 전년 대비 56% 증가, 전체 매출의 88%",
                "차분기 매출 가이던스 시장 예상 상회",
                "중국향 매출 제외 전제로 가이던스 제시",
            ],
            "earnings_highlights": [
                "Revenue $46.7 billion (+56% YoY)",
                "Data Center revenue $41.1 billion",
            ],
            "guidance": "raised",
            "tone": 0.5,
            "risk_factors": [
                _risk(
                    "수출 규제",
                    "중국향 첨단 GPU 판매 제한 지속",
                    "high",
                    "legal_regulatory",
                    "10-Q Item 1A",
                ),
                _risk(
                    "고객 집중",
                    "상위 고객 매출 비중 확대",
                    "medium",
                    "market_demand",
                    "10-Q MD&A",
                ),
                _risk(
                    "공급망 제약",
                    "CoWoS 패키징 캐파 제약",
                    "medium",
                    "operational",
                    "8-K EX-99.1",
                ),
            ],
            "sources": ["8-K Item 2.02 EX-99.1", "10-Q (2026-08-27)"],
        },
        thesis="실적·가이던스가 강하고 퀀트·텍스트 신호가 함께 매수를 가리킨다. 높은 변동성과 수출규제 뉴스를 "
        "감안해 비중은 Kelly 상한 이내로 제한한다.",
        critic_adj=-6,
        contradictions=("ML은 상승을 가리키나 7일 뉴스 감성은 하락 추세",),
    ),
    _Spec(
        "247540",
        "에코프로비엠",
        Market.KR,
        "KOSDAQ",
        "2차전지 소재",
        "KOSDAQ",
        105_500,
        0.58,
        1.40,
        -0.10,
        ((150, -0.45), (60, -0.60), (21, -0.70)),
        {
            "per": None,
            "forward_per": 85.0,
            "pbr": 4.1,
            "roe": -0.03,
            "operating_margin": -0.012,
            "revenue_growth_yoy": -0.281,
            "earnings_growth_yoy": None,
            "dividend_yield": 0.0,
            "sector_per": 22.0,
            "analyst_target_price": 96_000,
            "analyst_target_high": 140_000,
            "analyst_count": 14,
            "analyst_rating": "2.80 중립",
        },
        leaf=3,
        z_mom=-1.6,
        news={
            "score": -0.38,
            "n_articles": 18,
            "n_used": 11,
            "trend_7d": -0.15,
            "key_positives": ["북미 양극재 공장 착공"],
            "key_negatives": [
                "리튬 가격 하락에 따른 재고평가손실 확대",
                "주요 고객사 EV 판매 둔화로 출하량 감소",
            ],
            "key_events": ["유상증자 검토설 부인 공시"],
        },
        filings={
            "summary": [
                "2분기 영업손실 312억원으로 적자 지속",
                "양극재 판매량 전분기 대비 14% 감소",
                "재고평가손실 반영으로 수익성 악화",
            ],
            "earnings_highlights": [
                "매출액 6,420억원 (-28.1% YoY)",
                "영업손실 312억원 (적자지속)",
            ],
            "guidance": "none",
            "tone": -0.5,
            "risk_factors": [
                _risk(
                    "재고평가손실 확대",
                    "리튬 가격 약세 지속 시 추가 손실 인식",
                    "high",
                    "financial",
                    "영업(잠정)실적(공정공시)",
                ),
                _risk(
                    "고객 수요 둔화",
                    "EV 판매 둔화로 출하량 감소",
                    "high",
                    "market_demand",
                    "뉴스: 출하량 감소",
                ),
                _risk(
                    "재무구조 악화",
                    "적자 지속 시 차입 확대 가능성",
                    "medium",
                    "liquidity_dilution",
                    "반기보고서",
                ),
            ],
            "sources": ["영업(잠정)실적(공정공시)", "반기보고서"],
        },
        thesis="적자 지속과 고객 수요 둔화로 펀더멘털·추세·텍스트 신호가 모두 약하다. RSMT 저하 모드(오프셋 입력 "
        "누락)로 신뢰도를 감점했다.",
        warnings=("밸류에이션 히스토리 수집 실패: HTTPError: 503 Service Unavailable",),
        degraded=("오프셋 입력 누락(z)",),
    ),
    _Spec(
        "MSFT",
        "Microsoft",
        Market.US,
        "NASDAQ",
        "Software",
        "SPY",
        562.40,
        0.22,
        0.95,
        0.16,
        ((120, 0.22), (60, 0.18), (21, 0.12)),
        {
            "per": 36.5,
            "forward_per": 31.2,
            "pbr": 11.0,
            "roe": 0.33,
            "operating_margin": 0.45,
            "revenue_growth_yoy": 0.15,
            "earnings_growth_yoy": 0.13,
            "dividend_yield": 0.0065,
            "sector_per": 32.0,
            "analyst_target_price": 640.0,
            "analyst_target_high": 720.0,
            "analyst_count": 52,
            "analyst_rating": "Strong Buy",
        },
        leaf=1,
        z_mom=0.6,
        news={
            "score": 0.28,
            "n_articles": 26,
            "n_used": 15,
            "trend_7d": 0.03,
            "key_positives": ["Azure 성장률 재가속 (+39%)", "Copilot 유료 좌석 증가"],
            "key_negatives": ["AI 인프라 CAPEX 부담 확대"],
            "key_events": ["OpenAI 계약 구조 재조정"],
        },
        filings={
            "summary": [
                "Azure 등 클라우드 매출 39% 성장, 시장 기대 상회",
                "분기 CAPEX 300억 달러로 증가, 감가상각 부담 확대",
                "영업이익률 45% 유지",
            ],
            "earnings_highlights": [
                "Revenue $76.4 billion (+18% YoY)",
                "Operating income $34.3 billion",
            ],
            "guidance": "maintained",
            "tone": 0.35,
            "risk_factors": [
                _risk(
                    "CAPEX 부담",
                    "AI 데이터센터 투자로 FCF 둔화",
                    "medium",
                    "financial",
                    "10-K MD&A",
                ),
                _risk(
                    "규제",
                    "EU 클라우드 반독점 조사",
                    "medium",
                    "legal_regulatory",
                    "10-K Item 1A",
                ),
            ],
            "sources": ["10-K (2026-07-30)", "8-K Item 2.02 EX-99.1"],
        },
        thesis="클라우드 재가속과 안정적 이익률로 저변동 매수 후보다. CAPEX 증가가 마진에 미치는 영향은 다음 분기 "
        "확인이 필요하다.",
    ),
    _Spec(
        "035420",
        "NAVER",
        Market.KR,
        "KOSPI",
        "인터넷",
        "KOSPI",
        231_500,
        0.31,
        0.90,
        0.02,
        ((120, 0.08), (60, -0.05), (21, 0.02)),
        {
            "per": 17.8,
            "forward_per": 15.9,
            "pbr": 1.3,
            "roe": 0.078,
            "operating_margin": 0.19,
            "revenue_growth_yoy": 0.09,
            "earnings_growth_yoy": 0.05,
            "dividend_yield": 0.005,
            "sector_per": 19.5,
            "analyst_target_price": 265_000,
            "analyst_target_high": 320_000,
            "analyst_count": 28,
            "analyst_rating": "4.00 매수",
        },
        leaf=1,
        z_mom=-0.2,
        news={
            "score": 0.05,
            "n_articles": 15,
            "n_used": 9,
            "trend_7d": 0.02,
            "key_positives": ["하이퍼클로바X 기업 고객 확대"],
            "key_negatives": ["커머스 경쟁 심화로 거래액 성장 둔화"],
            "key_events": [],
        },
        filings=None,
        thesis="AI 서비스 확장 기대에도 커머스 성장 둔화로 신호가 중립이다. 공시 요약이 누락되어 신뢰도를 낮췄다.",
        llm_errors=("filings: stop_reason=max_tokens — 공시 요약 누락",),
    ),
    _Spec(
        "TSLA",
        "Tesla",
        Market.US,
        "NASDAQ",
        "Automobiles",
        "SPY",
        318.60,
        0.62,
        1.90,
        0.25,
        ((120, 0.40), (60, 0.10), (21, -2.6)),
        {
            "per": 165.0,
            "forward_per": 120.0,
            "pbr": 13.5,
            "roe": 0.08,
            "operating_margin": 0.07,
            "revenue_growth_yoy": -0.02,
            "earnings_growth_yoy": -0.35,
            "dividend_yield": None,
            "sector_per": 18.0,
            "analyst_target_price": 300.0,
            "analyst_target_high": 500.0,
            "analyst_count": 44,
            "analyst_rating": "Hold",
        },
        leaf=3,
        z_mom=0.3,
        news={
            "score": -0.12,
            "n_articles": 38,
            "n_used": 20,
            "trend_7d": -0.22,
            "key_positives": ["로보택시 서비스 지역 확대"],
            "key_negatives": [
                "3분기 인도량 컨센서스 하회 전망",
                "CEO 관련 불확실성 재부각",
            ],
            "key_events": ["1개월간 주가 20%대 하락"],
        },
        filings={
            "summary": [
                "자동차 매출 전년 대비 감소, 에너지 저장 매출은 성장",
                "평균 판매가격 인하로 자동차 총이익률 하락",
                "로보택시·휴머노이드 투자 확대",
            ],
            "earnings_highlights": [
                "Revenue $22.5 billion (-12% YoY)",
                "Energy storage deployments 9.6 GWh",
            ],
            "guidance": "none",
            "tone": -0.1,
            "risk_factors": [
                _risk(
                    "수요 둔화",
                    "EV 가격 경쟁으로 판매량·마진 동반 하락",
                    "high",
                    "market_demand",
                    "10-Q MD&A",
                ),
                _risk(
                    "핵심 인물 리스크",
                    "CEO 의존도 및 지배구조 논란",
                    "medium",
                    "governance",
                    "10-Q Item 1A",
                ),
                _risk(
                    "규제·리콜",
                    "자율주행 관련 NHTSA 조사",
                    "medium",
                    "legal_regulatory",
                    "8-K",
                ),
            ],
            "sources": ["10-Q (2026-07-24)", "8-K"],
        },
        thesis="단기 급락과 추세 훼손으로 팩터·텍스트 신호가 약세이며, ML의 반등 신호는 스킬 미달로 가중치가 낮다. "
        "인도량 발표 확인 전까지 비중 축소를 권고한다.",
        critic_adj=-8,
        contradictions=("ML 신호는 반등(+1.5%)을 예측하나 뉴스 감성 추세는 악화",),
        extra_triggers=("3분기 인도량 발표(10/2) 확인 전 신규진입 보류",),
        next_earnings=date(2026, 10, 21),
        earnings_estimate=True,
    ),
    _Spec(
        "051910",
        "LG화학",
        Market.KR,
        "KOSPI",
        "화학",
        "KOSPI",
        318_000,
        0.42,
        1.20,
        -0.05,
        ((150, -0.30), (60, -0.20), (21, -0.25)),
        {
            "per": 42.0,
            "forward_per": 25.0,
            "pbr": 0.8,
            "roe": 0.02,
            "operating_margin": 0.028,
            "revenue_growth_yoy": -0.08,
            "earnings_growth_yoy": -0.55,
            "dividend_yield": 0.011,
            "sector_per": 14.0,
            "analyst_target_price": 380_000,
            "analyst_target_high": 520_000,
            "analyst_count": 22,
            "analyst_rating": "3.60 매수",
        },
        leaf=3,
        z_mom=-1.2,
        news={
            "score": -0.25,
            "n_articles": 16,
            "n_used": 10,
            "trend_7d": -0.04,
            "key_positives": ["첨단소재 부문 흑자 전환"],
            "key_negatives": [
                "석유화학 업황 부진 장기화",
                "LG에너지솔루션 지분가치 하락",
            ],
            "key_events": ["NCC 설비 매각 검토"],
        },
        filings={
            "summary": [
                "석유화학 부문 영업손실 지속",
                "첨단소재 매출 증가로 적자폭 축소",
                "배당 축소 가능성 언급",
            ],
            "earnings_highlights": ["영업이익 1,210억원 (-72.5% YoY)"],
            "guidance": "none",
            "tone": -0.35,
            "risk_factors": [
                _risk(
                    "석유화학 공급과잉",
                    "중국 증설로 스프레드 약세 지속",
                    "high",
                    "market_demand",
                    "반기보고서",
                ),
                _risk(
                    "자회사 가치 하락",
                    "LG에너지솔루션 주가 하락이 지분가치 훼손",
                    "medium",
                    "financial",
                    "뉴스",
                ),
                _risk("재무 부담", "차입금 증가", "medium", "financial", "반기보고서"),
            ],
            "sources": ["반기보고서", "영업(잠정)실적(공정공시)"],
        },
        thesis="석유화학 업황 부진과 이익 급감으로 퀀트·텍스트 신호가 모두 약세다. 반등 시 비중 축소를 권고한다.",
    ),
    _Spec(
        "AMZN",
        "Amazon",
        Market.US,
        "NASDAQ",
        "Consumer / Cloud",
        "SPY",
        241.85,
        0.30,
        1.20,
        0.12,
        ((120, 0.28), (60, 0.24), (21, 0.15)),
        {
            "per": 39.0,
            "forward_per": 32.0,
            "pbr": 7.1,
            "roe": 0.21,
            "operating_margin": 0.11,
            "revenue_growth_yoy": 0.11,
            "earnings_growth_yoy": 0.28,
            "dividend_yield": None,
            "sector_per": 30.0,
            "analyst_target_price": 285.0,
            "analyst_target_high": 330.0,
            "analyst_count": 61,
            "analyst_rating": "Strong Buy",
        },
        leaf=1,
        z_mom=0.8,
        news={
            "score": 0.30,
            "n_articles": 34,
            "n_used": 18,
            "trend_7d": 0.06,
            "key_positives": ["AWS 성장률 20%대 회복", "광고 매출 고성장 지속"],
            "key_negatives": ["FTC 반독점 소송 일정 확정"],
            "key_events": [],
        },
        filings={
            "summary": [
                "AWS 영업이익률 37%로 개선",
                "북미 리테일 마진 사상 최고",
                "프라임데이 매출 역대 최대",
            ],
            "earnings_highlights": [
                "Net sales $167.7 billion (+11% YoY)",
                "AWS segment sales $30.9 billion (+19% YoY)",
            ],
            "guidance": "maintained",
            "tone": 0.4,
            "risk_factors": [
                _risk(
                    "반독점 소송",
                    "FTC 소송 결과에 따른 사업 구조 변경 가능성",
                    "medium",
                    "legal_regulatory",
                    "10-Q Item 1A",
                ),
                _risk(
                    "CAPEX 부담",
                    "AI 인프라 투자 확대",
                    "medium",
                    "financial",
                    "10-Q MD&A",
                ),
            ],
            "sources": ["10-Q (2026-08-01)"],
        },
        thesis="AWS 재가속과 리테일 마진 개선으로 이익 추정치가 상향 중이다. 반독점 소송은 장기 변수로 단기 신호 "
        "영향은 제한적이다.",
    ),
    _Spec(
        "207940",
        "삼성바이오로직스",
        Market.KR,
        "KOSPI",
        "바이오",
        "KOSPI",
        1_124_000,
        0.27,
        0.70,
        0.08,
        ((120, 0.12), (60, 0.02), (21, 0.05)),
        {
            "per": 58.0,
            "forward_per": 49.0,
            "pbr": 6.8,
            "roe": 0.12,
            "operating_margin": 0.33,
            "revenue_growth_yoy": 0.18,
            "earnings_growth_yoy": 0.21,
            "dividend_yield": None,
            "sector_per": 40.0,
            "analyst_target_price": 1_250_000,
            "analyst_target_high": 1_450_000,
            "analyst_count": 19,
            "analyst_rating": "4.00 매수",
        },
        leaf=1,
        z_mom=0.0,
        news={
            "score": 0.06,
            "n_articles": 12,
            "n_used": 7,
            "trend_7d": 0.0,
            "key_positives": ["미국 대형 제약사와 CMO 계약 체결"],
            "key_negatives": ["美 생물보안법 관련 불확실성"],
            "key_events": [],
        },
        filings={
            "summary": [
                "2분기 매출 1조 2,000억원, 4공장 가동률 상승",
                "5공장 증설 순항",
                "인적분할 후 사업 재편 진행",
            ],
            "earnings_highlights": [
                "매출액 1조 2,000억원 (+18% YoY)",
                "영업이익 4,000억원 (+21% YoY)",
            ],
            "guidance": "none",
            "tone": 0.10,
            "risk_factors": [
                _risk(
                    "밸류에이션 부담",
                    "PER 50배 이상으로 기대치 선반영",
                    "medium",
                    "other",
                    "증권사 리서치 요약",
                ),
                _risk(
                    "정책 불확실성",
                    "美 생물보안법 시행 세칙 미확정",
                    "low",
                    "legal_regulatory",
                    "뉴스",
                ),
            ],
            "sources": ["반기보고서", "단일판매·공급계약체결"],
        },
        thesis="안정적 수주·실적 성장에도 밸류에이션 부담으로 종합 점수가 중립 구간이다. 신규 수주 공시 시 재평가한다.",
        warnings=("변동성: EWMA(λ=0.94) 대체 사용 연 24.6% — GARCH 미사용 사유: 최적화 미수렴 (flag=1)",),
    ),
    _Spec(
        "JPM",
        "JPMorgan Chase",
        Market.US,
        "NYSE",
        "Banks",
        "SPY",
        312.75,
        0.21,
        1.05,
        0.14,
        ((120, 0.20), (60, 0.16), (21, 0.12)),
        {
            "per": 15.8,
            "forward_per": 14.6,
            "pbr": 2.4,
            "roe": 0.17,
            "operating_margin": 0.38,
            "revenue_growth_yoy": 0.07,
            "earnings_growth_yoy": 0.11,
            "dividend_yield": 0.018,
            "sector_per": 14.0,
            "analyst_target_price": 330.0,
            "analyst_target_high": 360.0,
            "analyst_count": 24,
            "analyst_rating": "Buy",
        },
        leaf=1,
        z_mom=0.5,
        news={
            "score": 0.22,
            "n_articles": 20,
            "n_used": 12,
            "trend_7d": 0.04,
            "key_positives": [
                "투자은행 수수료 전년 대비 증가",
                "순이자이익 가이던스 상향",
            ],
            "key_negatives": ["상업용 부동산 대손 우려"],
            "key_events": [],
        },
        filings={
            "summary": [
                "순이자이익 가이던스 상향",
                "투자은행 부문 수수료 회복",
                "CET1 비율 15.3%로 자본 여력 충분",
            ],
            "earnings_highlights": ["Net revenue $45.7 billion (+8% YoY)", "EPS $5.24"],
            "guidance": "raised",
            "tone": 0.35,
            "risk_factors": [
                _risk(
                    "신용 리스크",
                    "상업용 부동산 대출 부실 가능성",
                    "medium",
                    "financial",
                    "10-Q",
                ),
                _risk(
                    "규제 자본",
                    "바젤 III 최종안 적용 시 자본 부담",
                    "low",
                    "legal_regulatory",
                    "10-Q Item 1A",
                ),
            ],
            "sources": ["10-Q (2026-08-05)"],
        },
        thesis="이익 안정성과 자본 여력이 우수하고 변동성이 낮아 방어적 매수 후보다. 금리 경로 변화가 순이자이익에 "
        "미치는 영향을 점검한다.",
        next_earnings=date(2026, 10, 13),
        earnings_estimate=False,
    ),
]

SAMPLE_TICKERS: tuple[str, ...] = tuple(s.ticker for s in _SPECS)
_SPEC_BY_TICKER = {s.ticker: s for s in _SPECS}


def _spec_text(spec: _Spec, en: bool) -> dict[str, Any]:
    """Every text field of a spec in the report language (English from report/sample_en.py, same shapes)."""
    if not en:
        return {
            "name": spec.name,
            "sector": spec.sector,
            "analyst_rating": spec.fund.get("analyst_rating"),
            "news": spec.news,
            "filings": spec.filings,
            "thesis": spec.thesis,
            "contradictions": spec.contradictions,
            "llm_errors": spec.llm_errors,
            "warnings": spec.warnings,
            "degraded": spec.degraded,
            "extra_triggers": spec.extra_triggers,
            "llm_notices": spec.llm_notices,
        }
    e = SAMPLE_TEXT_EN[spec.ticker]
    news = None if spec.news is None else {**spec.news, **e["news"]}
    filings = None
    if spec.filings is not None:
        ef = e["filings"]
        filings = {
            **spec.filings,
            "summary": ef["summary"],
            "earnings_highlights": ef["earnings_highlights"],
            "sources": ef["sources"],
            "risk_factors": [
                _risk(title, detail, ko["severity"], ko["category"], ref)
                for (title, detail, ref), ko in zip(ef["risk_factors"], spec.filings["risk_factors"], strict=True)
            ],
        }
    return {
        **{k: e[k] for k in ("name", "sector", "analyst_rating", "thesis")},
        "news": news,
        "filings": filings,
        **{
            k: tuple(e[k])
            for k in (
                "contradictions",
                "llm_errors",
                "warnings",
                "degraded",
                "extra_triggers",
                "llm_notices",
            )
        },
    }


# ----------------------------------------------------------------------------- new data contracts (with_models)

# PIT multi-factor model per market: OOS skill (KR passes the IC-t gate, US does not: both badges are exercised)
_FACTOR_MODEL: dict[Market, dict[str, Any]] = {
    Market.KR: {"skill_ok": True, "oos_ic_mean": 0.034, "oos_ic_t": 2.41},
    Market.US: {"skill_ok": False, "oos_ic_mean": 0.012, "oos_ic_t": 1.28},
}
# deterministic model-vs-legacy disagreement per ticker (percentile points), so the blend visibly moves the pillar
_FM_OFFSET = {
    "005930": 0.04,
    "AAPL": -0.06,
    "000660": 0.02,
    "NVDA": 0.05,
    "247540": -0.03,
    "MSFT": 0.03,
    "035420": -0.04,
    "TSLA": 0.06,
    "051910": 0.02,
    "AMZN": -0.02,
    "207940": 0.05,
    "JPM": 0.04,
}
_FM_WEIGHTS = {
    "value_ep": 0.19,
    "value_bp": 0.14,
    "volume_trend": -0.08,
    "mom_3m": 0.17,
    "mom_6m": 0.24,
    "sector_mom_3m": 0.18,
}
_FM_OUTSIDE_UNIVERSE = ("MSFT",)  # exercises FactorModelScore.notes on the report

# Macro regimes (macro/regime.py shapes: Korean labels / summaries as that module writes them; keys as its table)
_MACRO: tuple[dict[str, Any], ...] = (
    {
        "region": "global",
        "score": 0.08,
        "label": "중립",
        "indicators": (
            ("g_acwi_ma200", "ACWI 200일선 대비", 4.1, "%", 0.9, 0.62),
            ("g_copper_gold_3m", "구리/금 비율 3M 변화", 5.9, "비율", -0.041, -0.48),
            ("g_oil_3m", "WTI 유가 3M 변화", 64.2, "달러", -0.052, 0.0),
            ("g_vix", "VIX 수준", 15.2, "pt", -1.3, 0.35),
            ("g_credit_3m", "신용(HYG/LQD) 3M 추세", 0.71, "비율", 0.006, 0.21),
            ("dxy_3m", "달러인덱스 3M 변화", 98.6, "지수", 0.018, -0.29),
        ),
    },
    {
        "region": "us",
        "score": 0.31,
        "label": "위험선호",
        "indicators": (
            ("us_curve_10y_3m", "미 10Y−3M 금리차", 0.42, "%p", 0.35, 0.74),
            ("us_10y_3m", "미 10년물 3M 변화", 4.12, "%", -0.18, 0.0),
            ("us_spy_ma200", "SPY 200일선 대비", 7.8, "%", 2.2, 1.05),
            ("dxy_3m", "달러인덱스 3M 변화", 98.6, "지수", 0.018, -0.29),
        ),
    },
    {
        "region": "kr",
        "score": -0.34,
        "label": "위험회피",
        "indicators": (
            ("kr_usdkrw_3m", "USD/KRW 3M 변화", 1385.0, "원", 0.043, -1.12),
            ("kr_ktb3y_3m", "국고채 3년 3M 변화", 2.61, "%", 0.21, -0.86),
            ("kr_us_spread", "한미 10년 금리차", -1.39, "%p", -0.12, -0.41),
            ("kr_kospi_rel_acwi_3m", "KOSPI vs ACWI 3M 상대강도", 1.8, "%p", None, 0.22),
            ("kr_ewy_ma200", "EWY 200일선 대비", -2.4, "%", -3.1, -0.37),
            ("dxy_3m", "달러인덱스 3M 변화", 98.6, "지수", 0.018, -0.29),
        ),
        "stale": ("kr_ktb3y_3m",),
    },
)
_MACRO_REGION_KO = {"global": "글로벌", "us": "미국", "kr": "한국"}
_MACRO_REGION_EN = {"global": "Global", "us": "US", "kr": "Korea"}
_MACRO_LABEL_EN = {"위험선호": "risk-on", "중립": "neutral", "위험회피": "risk-off"}

# Factor backtest summaries (backtest.report_summary shape): per factor raw / sector-neutral rows + model OOS
_BT_FACTORS: dict[Market, tuple[tuple[str, str, float, float, float, bool, bool], ...]] = {
    # (factor, Korean label, IC t, L/S net annual return, L/S net Sharpe, direction_ok, BH significant)
    Market.KR: (
        ("value_ep", "E/P (PER 역수)", 2.6, 0.061, 0.71, True, True),
        ("value_bp", "B/P (PBR 역수)", 1.9, 0.038, 0.44, True, False),
        ("volume_trend", "거래대금 추세 log(ADV20/ADV120)", -1.4, -0.022, -0.25, True, False),
        ("mom_3m", "3개월 모멘텀", 0.8, 0.012, 0.11, True, False),
        ("mom_6m", "6개월 모멘텀", 2.2, 0.049, 0.52, True, True),
        ("sector_mom_3m", "섹터 3개월 모멘텀", 1.6, 0.027, 0.33, True, False),
    ),
    Market.US: (
        ("value_ep", "E/P (PER 역수)", 1.1, 0.014, 0.18, True, False),
        ("value_bp", "B/P (PBR 역수)", -0.6, -0.009, -0.08, False, False),
        ("volume_trend", "거래대금 추세 log(ADV20/ADV120)", 0.9, 0.008, 0.10, True, False),
        ("mom_3m", "3개월 모멘텀", 1.3, 0.021, 0.24, True, False),
        ("mom_6m", "6개월 모멘텀", 2.3, 0.044, 0.49, True, True),
        ("sector_mom_3m", "섹터 3개월 모멘텀", 1.7, 0.025, 0.29, True, False),
    ),
}
_BT_MODEL: dict[Market, dict[str, Any]] = {
    Market.KR: {"oos_months": 91, "ann": 0.072, "sharpe": 0.83},
    Market.US: {"oos_months": 115, "ann": 0.024, "sharpe": 0.31},
}


# ----------------------------------------------------------------------------- synthetic price paths


def _seed(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode("utf-8"))


def _drift_vector(n: int, base: float, segments: tuple[tuple[int, float], ...]) -> np.ndarray:
    """Per-day annual log drift: the recent `segments` at the end, `base` for the remainder."""
    recent = sum(d for d, _ in segments)
    if recent >= n:
        raise ValueError("drift segments longer than the history")
    parts = [np.full(n - recent, base)] + [np.full(d, a) for d, a in segments]
    return np.concatenate(parts)


def _segment_ids(n: int, segments: tuple[tuple[int, float], ...]) -> np.ndarray:
    ids = [np.zeros(n - sum(d for d, _ in segments), dtype=int)]
    ids += [np.full(d, i + 1, dtype=int) for i, (d, _) in enumerate(segments)]
    return np.concatenate(ids)


def _demean_by_segment(x: np.ndarray, seg: np.ndarray) -> np.ndarray:
    out = x.copy()
    for s in np.unique(seg):
        m = seg == s
        out[m] -= out[m].mean()
    return out


def _ohlcv(log_ret: np.ndarray, end_price: float, vol: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    n = len(log_ret)
    idx = pd.bdate_range(end=pd.Timestamp(AS_OF), periods=n)
    close = np.exp(np.cumsum(log_ret))
    close = close / close[-1] * end_price
    vd = vol / math.sqrt(TRADING_DAYS)
    prev = np.concatenate([[close[0]], close[:-1]])
    open_ = prev * np.exp(rng.normal(0.0, vd / 3, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, vd / 2, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, vd / 2, n)))
    volume = rng.lognormal(mean=14.0, sigma=0.35, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    ).astype("float64")


def _bench_log_returns(bench_id: str, n: int = HISTORY_DAYS) -> np.ndarray:
    b = _BENCHES[bench_id]
    rng = np.random.default_rng(b.seed)
    seg = _segment_ids(n, b.segments)
    noise = _demean_by_segment(rng.normal(0.0, b.vol / math.sqrt(TRADING_DAYS), n), seg)
    return _drift_vector(n, b.base, b.segments) / TRADING_DAYS + noise


def sample_benchmark_history(bench_id: str) -> pd.DataFrame:
    """Synthetic benchmark OHLCV (PRICE_COLUMNS contract) for "KOSPI", "KOSDAQ" or "SPY"."""
    start_level = {"KOSPI": 3400.0, "KOSDAQ": 880.0, "SPY": 690.0}[bench_id]
    return _ohlcv(
        _bench_log_returns(bench_id),
        start_level,
        _BENCHES[bench_id].vol,
        _BENCHES[bench_id].seed,
    )


def sample_price_history(ticker: str) -> pd.DataFrame:
    """Synthetic OHLCV for one sample ticker (ends exactly at the sample last price on AS_OF).

    Stock log return = beta * benchmark return + idiosyncratic part, whose drift is set per segment so the
    realised segment returns match the narrative (e.g. TSLA's one-month drop) exactly.
    """
    spec = _SPEC_BY_TICKER.get(ticker)
    if spec is None:
        raise KeyError(f"unknown sample ticker {ticker!r}; choose from {SAMPLE_TICKERS}")
    n = HISTORY_DAYS
    b = _bench_log_returns(spec.benchmark, n)
    bvol = _BENCHES[spec.benchmark].vol
    idio_vol = math.sqrt(max(spec.vol**2 - (spec.beta * bvol) ** 2, (0.3 * spec.vol) ** 2))
    seg = _segment_ids(n, spec.segments)
    rng = np.random.default_rng(_seed("px", ticker))
    e = _demean_by_segment(rng.normal(0.0, idio_vol / math.sqrt(TRADING_DAYS), n), seg)
    target = _drift_vector(n, spec.base_drift, spec.segments) / TRADING_DAYS
    b_seg_mean = np.zeros(n)
    for s in np.unique(seg):
        m = seg == s
        b_seg_mean[m] = b[m].mean()
    log_ret = spec.beta * b + (target - spec.beta * b_seg_mean) + e
    return _ohlcv(log_ret, spec.price, spec.vol, _seed("ohlc", ticker))


# ----------------------------------------------------------------------------- technicals & factors


def _wilder_rsi(close: pd.Series, n: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    g, l_ = float(gain.iloc[-1]), float(loss.iloc[-1])
    return 100.0 if l_ == 0 else 100.0 - 100.0 / (1.0 + g / l_)


def _technicals(px: pd.DataFrame, bench: pd.DataFrame) -> Technicals:
    close, high, low = px["close"], px["high"], px["low"]
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean().iloc[-1])
    last = float(close.iloc[-1])
    lr = np.log(close).diff()
    blr = np.log(bench["close"]).diff()
    cov = lr.iloc[-252:].cov(blr.iloc[-252:])
    beta = float(cov / blr.iloc[-252:].var())
    y1 = close.iloc[-252:]
    ret3 = last / float(close.iloc[-64]) - 1.0
    bret3 = float(bench["close"].iloc[-1]) / float(bench["close"].iloc[-64]) - 1.0
    return Technicals(
        last_close=last,
        atr=atr,
        atr_pct=atr / last,
        rsi14=_wilder_rsi(close),
        ma50=float(close.iloc[-50:].mean()),
        ma200=float(close.iloc[-200:].mean()),
        high_52w=float(high.iloc[-252:].max()),
        low_52w=float(low.iloc[-252:].min()),
        realized_vol_60d=float(lr.iloc[-60:].std() * math.sqrt(TRADING_DAYS)),
        beta_1y=beta,
        max_drawdown_1y=float((y1 / y1.cummax() - 1.0).min()),
        ret_1m=last / float(close.iloc[-22]) - 1.0,
        ret_3m=ret3,
        ret_12m_ex1m=float(close.iloc[-22]) / float(close.iloc[-253]) - 1.0,
        rel_strength_3m=ret3 - bret3,
        support=float(low.iloc[-20:].min()),
        resistance=float(high.iloc[-60:].max()),
    )


def _logistic(x: float | None, c: float, s: float) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return 100.0 / (1.0 + math.exp(-(x - c) / s))


def _mean(xs: list[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return float(np.mean(v)) if v else None


def _pct(x: float) -> str:
    return f"{x * 100:+.0f}%".replace("-", "−")


def _factors(spec: _Spec, t: Technicals, fund: Fundamentals, en: bool = False) -> list[FactorScore]:
    prox = t.last_close / t.high_52w - 1.0
    mom = _mean(
        [
            _logistic(t.ret_12m_ex1m, 0.0, 0.20),
            _logistic(t.ret_3m, 0.0, 0.10),
            _logistic(t.rel_strength_3m, 0.0, 0.07),
            _logistic(prox, -0.15, 0.108),
        ]
    )
    per_ok = fund.per is not None and fund.per > 0
    rel_per = fund.per / fund.sector_per if per_ok and fund.sector_per else None
    upside = (fund.analyst_target_price / t.last_close - 1.0) if fund.analyst_target_price else None
    val = _mean(
        [
            _logistic(-math.log(rel_per), 0.0, 0.368) if rel_per else None,
            _logistic(upside, 0.0, 0.15),
        ]
    )
    beta_score = max(0.0, min(100.0, 100.0 - 60.0 * abs((t.beta_1y or 1.0) - 0.8)))
    vol = _mean(
        [
            _logistic(-t.realized_vol_60d, -0.375, 0.13),
            beta_score,
            _logistic(t.max_drawdown_1y, -0.29, 0.137),
        ]
    )
    qual = _mean(
        [
            _logistic(fund.roe, 0.10, 0.06),
            _logistic(fund.operating_margin, 0.10, 0.07),
            _logistic(fund.earnings_growth_yoy, 0.0, 0.20),
            _logistic(fund.revenue_growth_yoy, 0.05, 0.10),
        ]
    )
    rsi = t.rsi14 or 50.0
    rsi_score = 80.0 if 50 <= rsi <= 65 else 30.0 if rsi > 75 else 40.0 if rsi < 30 else 60.0
    trend = _mean(
        [
            70.0 if t.last_close > (t.ma50 or 0) else 30.0,
            70.0 if t.last_close > (t.ma200 or 0) else 30.0,
            70.0 if (t.ma50 or 0) > (t.ma200 or 0) else 30.0,
            rsi_score,
        ]
    )
    cross = ">" if (t.ma50 or 0) > (t.ma200 or 0) else "<"
    if en:
        per_txt = f"PER {fund.per:.1f}x (sector {fund.sector_per:.1f}x)" if per_ok else "loss-making — PER excluded"
        return [
            FactorScore(
                name="momentum",
                label="Momentum",
                score=mom,
                metrics={
                    "ret_12m_ex1m": t.ret_12m_ex1m,
                    "ret_3m": t.ret_3m,
                    "rel_strength_3m": t.rel_strength_3m,
                    "prox_52w_high": prox,
                },
                rationale=f"12-1M return {_pct(t.ret_12m_ex1m or 0)}, vs 52-week high {_pct(prox)}",
            ),
            FactorScore(
                name="value",
                label="Value",
                score=val,
                metrics={"per": fund.per, "sector_per": fund.sector_per, "analyst_upside": upside},
                rationale=per_txt + (f", consensus upside {_pct(upside)}" if upside is not None else ""),
            ),
            FactorScore(
                name="volatility",
                label="Volatility (stability)",
                score=vol,
                metrics={
                    "realized_vol_60d": t.realized_vol_60d,
                    "beta_1y": t.beta_1y,
                    "max_drawdown_1y": t.max_drawdown_1y,
                },
                rationale=f"60D vol {t.realized_vol_60d * 100:.0f}%, 1Y max drawdown {_pct(t.max_drawdown_1y)}",
            ),
            FactorScore(
                name="quality",
                label="Quality",
                score=qual,
                metrics={
                    "roe": fund.roe,
                    "operating_margin": fund.operating_margin,
                    "earnings_growth_yoy": fund.earnings_growth_yoy,
                    "revenue_growth_yoy": fund.revenue_growth_yoy,
                },
                rationale=f"ROE {(fund.roe or 0) * 100:.0f}%, "
                f"operating margin {(fund.operating_margin or 0) * 100:.0f}%",
            ),
            FactorScore(
                name="trend",
                label="Trend",
                score=trend,
                metrics={"rsi14": t.rsi14, "ma50": t.ma50, "ma200": t.ma200},
                rationale=f"RSI {rsi:.0f}, MA50 {cross} MA200",
            ),
        ]
    per_txt = f"PER {fund.per:.1f}배 (업종 {fund.sector_per:.1f}배)" if per_ok else "적자 — PER 지표 제외"
    return [
        FactorScore(
            name="momentum",
            label="모멘텀",
            score=mom,
            metrics={
                "ret_12m_ex1m": t.ret_12m_ex1m,
                "ret_3m": t.ret_3m,
                "rel_strength_3m": t.rel_strength_3m,
                "prox_52w_high": prox,
            },
            rationale=f"12-1M 수익률 {_pct(t.ret_12m_ex1m or 0)}, 52주 고점 대비 {_pct(prox)}",
        ),
        FactorScore(
            name="value",
            label="밸류에이션",
            score=val,
            metrics={
                "per": fund.per,
                "sector_per": fund.sector_per,
                "analyst_upside": upside,
            },
            rationale=per_txt + (f", 컨센서스 상승여력 {_pct(upside)}" if upside is not None else ""),
        ),
        FactorScore(
            name="volatility",
            label="변동성(안정성)",
            score=vol,
            metrics={
                "realized_vol_60d": t.realized_vol_60d,
                "beta_1y": t.beta_1y,
                "max_drawdown_1y": t.max_drawdown_1y,
            },
            rationale=f"60일 변동성 {t.realized_vol_60d * 100:.0f}%, 1년 최대낙폭 {_pct(t.max_drawdown_1y)}",
        ),
        FactorScore(
            name="quality",
            label="퀄리티",
            score=qual,
            metrics={
                "roe": fund.roe,
                "operating_margin": fund.operating_margin,
                "earnings_growth_yoy": fund.earnings_growth_yoy,
                "revenue_growth_yoy": fund.revenue_growth_yoy,
            },
            rationale=f"ROE {(fund.roe or 0) * 100:.0f}%, 영업이익률 {(fund.operating_margin or 0) * 100:.0f}%",
        ),
        FactorScore(
            name="trend",
            label="추세",
            score=trend,
            metrics={"rsi14": t.rsi14, "ma50": t.ma50, "ma200": t.ma200},
            rationale=f"RSI {rsi:.0f}, MA50 {cross} MA200",
        ),
    ]


# ----------------------------------------------------------------------------- one path matrix


def _t_innovations(z: np.ndarray, nu: float, h: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance Student-t daily innovations from correlated normals `z` (chi-square scale mixing)."""
    nu_d = max(4.05, 4.0 + (nu - 4.0) / h)
    chi = rng.chisquare(nu_d, size=z.shape)
    return z / np.sqrt(chi / nu_d) * math.sqrt((nu_d - 2.0) / nu_d)


def _market_normals(n_paths: int = N_PATHS, n_days: int = HORIZON) -> dict[Market, np.ndarray]:
    rng = np.random.default_rng(_seed("market-normals"))
    z_kr = rng.standard_normal((n_paths, n_days))
    z_us = 0.35 * z_kr + math.sqrt(1 - 0.35**2) * rng.standard_normal((n_paths, n_days))
    return {Market.KR: z_kr, Market.US: z_us}


@dataclass
class _Paths:
    r: np.ndarray  # (n_paths, n_days) cumulative stock log return on each close
    m: np.ndarray  # (n_paths, n_days) cumulative market log return


def _simulate(
    spec: _Spec,
    mu_e: float,
    sigma_e: float,
    nu: float,
    mu_m: float,
    sigma_m: float,
    nu_m: float,
    beta: float,
    rho: float,
    zm: np.ndarray,
    h: int,
) -> _Paths:
    """ONE daily path matrix of zm.shape[1] days; mu/sigma/nu are per `h`-day horizon (daily = mu/h, sigma/sqrt(h)).

    Column h-1 is the forecast-horizon distribution; the barrier reads the first `target_horizon_days`
    columns of the same matrix, so the headline forecast and the TP/SL probabilities share every draw.
    """
    rng = np.random.default_rng(_seed("paths", spec.ticker, AS_OF.isoformat()))
    ze = rho * zm + math.sqrt(1.0 - rho**2) * rng.standard_normal(zm.shape)
    m_d = mu_m / h + sigma_m / math.sqrt(h) * _t_innovations(zm, nu_m, h, rng)
    e_d = mu_e / h + sigma_e / math.sqrt(h) * _t_innovations(ze, nu, h, rng)
    return _Paths(r=np.cumsum(beta * m_d + e_d, axis=1), m=np.cumsum(m_d, axis=1))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _barrier(paths: _Paths, stop_pct: float, take_pct: float, horizon: int) -> BarrierOutcome:
    """First touch on daily closes over the first `horizon` days (the holding horizon) of the path matrix."""
    simple = np.expm1(paths.r[:, :horizon])
    n, h = simple.shape
    tp_hit, sl_hit = simple >= take_pct, simple <= stop_pct
    first_tp = np.where(tp_hit.any(axis=1), tp_hit.argmax(axis=1), h)
    first_sl = np.where(sl_hit.any(axis=1), sl_hit.argmax(axis=1), h)
    is_tp = first_tp < first_sl
    is_sl = first_sl < first_tp
    exit_idx = np.where(is_tp, first_tp, np.where(is_sl, first_sl, h - 1))
    trade = simple[np.arange(n), exit_idx]
    n_tp, n_sl = int(is_tp.sum()), int(is_sl.sum())
    grid = np.arange(0.0, 1.0 + 1e-9, 0.005)
    growth = np.log1p(np.outer(grid, trade)).mean(axis=1)
    return BarrierOutcome(
        p_take_profit=n_tp / n,
        p_stop_loss=n_sl / n,
        p_time_exit=(n - n_tp - n_sl) / n,
        expected_trade_return=float(trade.mean()),
        kelly_optimal=float(grid[int(growth.argmax())]),
        n_paths=n,
        horizon_days=h,
    )


# ----------------------------------------------------------------------------- fusion (sample-local)


_HORIZON_LABEL = {
    Signal.STRONG_BUY: "1~3개월 (분할매수)",
    Signal.BUY: "1~3개월",
    Signal.HOLD: "관망 · 1개월 후 재평가",
    Signal.SELL: "즉시~2주 내 비중 축소",
}
_HORIZON_LABEL_EN = {  # same as fusion/signal.py HORIZON_LABELS_EN
    Signal.STRONG_BUY: "1–3 months (scale in)",
    Signal.BUY: "1–3 months",
    Signal.HOLD: "Wait · review in 1 month",
    Signal.SELL: "Trim within 2 weeks",
}
_ORDER = [Signal.SELL, Signal.HOLD, Signal.BUY, Signal.STRONG_BUY]


def _llm_pillar(news: NewsSentiment | None, filings: FilingIntelligence | None) -> float | None:
    parts: list[tuple[float, float]] = []
    if news is not None:
        parts.append((0.65, news.score * min(1.0, news.n_used / 5)))
    if filings is not None:
        parts.append((0.35, filings.tone))
    if not parts:
        return None
    raw = sum(w * x for w, x in parts) / sum(w for w, _ in parts)
    risks = filings.risk_factors if filings else []
    penalty = min(
        0.4,
        0.05 * sum(r.severity == "high" for r in risks) + 0.20 * sum(r.severity == "critical" for r in risks),
    )
    return float(np.clip(raw - penalty, -1.0, 1.0))


def _confidence(
    settings: Settings,
    pillars: dict[str, float],
    composite: float,
    cv: dict[str, float],
    n_warn: int,
    fund: Fundamentals,
    news: NewsSentiment | None,
    critic_adj: int,
    degraded: bool,
) -> tuple[int, dict[str, float]]:
    w = {k: settings.fusion.weights[k] for k in pillars}
    tot = sum(w.values())
    sign = np.sign(composite)
    share = sum(w[k] * (0.5 if abs(p) < 0.05 else float(np.sign(p) == sign)) for k, p in pillars.items()) / tot
    agreement = 35.0 * share * (1.0 - min(1.0, float(np.std(list(pillars.values())))))
    strength = 20.0 * min(1.0, abs(composite) / 0.6)
    ml_skill = (
        20.0 * float(np.clip((cv["auc"] - cv["base_auc"]) / 0.05, 0, 1)) * 0.5
        + 20.0 * float(np.clip((cv["base_logloss"] - cv["logloss"]) / 0.01, 0, 1)) * 0.5
    )
    present = [fund.per, fund.pbr, fund.roe, fund.analyst_target_price]
    completeness = max(0.4, sum(x is not None for x in present) / 4)
    quality = max(0.0, 15.0 - 3.0 * n_warn) * completeness
    evidence = 10.0 * min(1.0, news.n_used / 8) if news else 0.0
    breakdown = {
        "agreement": agreement,
        "strength": strength,
        "ml_skill": ml_skill,
        "data_quality": quality,
        "evidence": evidence,
        "critic": float(critic_adj),
        "degraded": -10.0 if degraded else 0.0,
    }
    conf = int(round(float(np.clip(sum(breakdown.values()), 5, 95))))
    return conf, {k: round(v, 2) for k, v in breakdown.items()}


def _raw_signal(settings: Settings, composite: float, conf: int, overrides: list[str]) -> Signal:
    th = settings.fusion.thresholds
    if composite >= th.strong_buy:
        if conf >= settings.fusion.strong_buy_min_confidence:
            return Signal.STRONG_BUY
        mc = settings.fusion.strong_buy_min_confidence
        overrides.append(
            f"Confidence {conf} < {mc} → Strong Buy lowered to Buy"
            if is_en(settings)
            else f"신뢰도 {conf} < {mc} → Strong Buy를 Buy로 조정"
        )
        return Signal.BUY
    if composite >= th.buy:
        return Signal.BUY
    if composite <= th.sell:
        return Signal.SELL
    return Signal.HOLD


# ----------------------------------------------------------------------------- builder


def _fmt_money(v: float, ccy: str, en: bool = False) -> str:
    """Same format as fusion/risk.py (and the report tables): KRW '286,500원' ('₩286,500'), USD '$337.02'."""
    if ccy == "KRW":
        return f"₩{v:,.0f}" if en else f"{v:,.0f}원"
    return f"${v:,.2f}"


def _instrument(spec: _Spec, txt: dict[str, Any] | None = None) -> Instrument:
    ccy = "KRW" if spec.market is Market.KR else "USD"
    yf = f"{spec.ticker}.{'KQ' if spec.exchange == 'KOSDAQ' else 'KS'}" if spec.market is Market.KR else spec.ticker
    return Instrument(
        ticker=spec.ticker,
        market=spec.market,
        name=(txt or {}).get("name", spec.name),
        exchange=spec.exchange,
        currency=ccy,
        sector=(txt or {}).get("sector", spec.sector),
        yf_symbol=yf,
        benchmark=spec.benchmark,
        benchmark_name=_BENCHES[spec.benchmark].name,
    )


def _regime(bench_id: str, en: bool = False) -> str:
    c = sample_benchmark_history(bench_id)["close"]
    r20 = float(c.iloc[-1] / c.iloc[-21] - 1.0)
    above = float(c.iloc[-1]) > float(c.iloc[-200:].mean())
    if en:
        return f"20D {_pct_1(r20)} · {'above' if above else 'below'} MA200"
    return f"20D {_pct_1(r20)} · MA200 {'위' if above else '아래'}"


def _pct_1(x: float) -> str:
    return f"{x * 100:+.1f}%".replace("-", "−")


def _diagnostics(market: Market, served: list[TickerReport], en: bool = False) -> list[str]:
    if en:
        return _diagnostics_en(market, served)
    mm = _MARKET_MODEL[market]
    cv = mm["cv"]
    leaves = _LEAVES[market]
    n = sum(v[1] for v in leaves.values())
    if market is Market.KR:
        tree = [
            f"vol_ratio_20_120 ≤ 1.12 (p=3.1e-04, n={n})",
            "├─ mkt_ret_60 ≤ 0.031 (p=2.2e-03, n=3405)",
            "│  ├─ 리프#1 n=1812 μe +0.2% σe 7.1% ν 6.2",
            "│  └─ 리프#2 n=1593 μe +0.9% σe 6.4% ν 8.0",
            "└─ dd_250 ≤ −0.18 (p=8.7e-03, n=1805)",
            "   ├─ 리프#3 n=812 μe −1.1% σe 9.8% ν 5.1",
            "   └─ 리프#4 n=993 μe +0.4% σe 8.3% ν 7.4",
        ]
        tail = ["제외 행: 결측 312 · 적재 실패 1/61 (035720: too short)"]
    else:
        tree = [
            f"mkt_vol_60 ≤ 0.162 (p=1.4e-03, n={n})",
            "├─ rsi14 ≤ 68.5 (p=1.9e-02, n=3328)",
            "│  ├─ 리프#1 n=2140 μe +0.4% σe 6.2% ν 7.8",
            "│  └─ 리프#2 n=1188 μe −0.2% σe 6.9% ν 6.6",
            "└─ 리프#3 n=1422 μe +0.1% σe 8.8% ν 5.4",
        ]
        tail = ["제외 행: 결측 188 · 적재 실패 0/62"]
    head = [
        f"RSMT-Q {market.value} v1 (샘플) · n={n:,} · 리프 {len(leaves)} · 루트 "
        f"p={'3.1e-04' if market is Market.KR else '1.4e-03'}"
    ]
    params = [f"γ={mm['gamma']:+.4f} η={mm['eta']:.2f} δ={mm['delta']:+.2f} δrel={mm['delta_rel']:+.2f}"]
    cvl = [
        f"CV(4폴드) AUC {cv['auc']:.3f} vs {cv['base_auc']:.3f} · LogLoss {cv['logloss']:.4f} vs "
        f"{cv['base_logloss']:.4f}"
        f" · CRPS {cv['crps']:.4f} vs {cv['base_crps']:.4f} · cover80 {cv['cover80']:.2f} vs {cv['base_cover80']:.2f}"
    ]
    bad = [t for t in served if t.quant.distribution is not None and t.quant.distribution.degraded]
    health = [f"헬스체크: 오프셋 적용 {len(served) - len(bad)}/{len(served)} · 붕괴 검사 통과"]
    for t in bad:
        health.append(f"⚠ {t.instrument.ticker} 저하: {', '.join(t.quant.distribution.degraded_reasons)}")
    return head + tree + params + cvl + tail + health


def _diagnostics_en(market: Market, served: list[TickerReport]) -> list[str]:
    """English twin of `_diagnostics` (same numbers, same line structure)."""
    mm = _MARKET_MODEL[market]
    cv = mm["cv"]
    leaves = _LEAVES[market]
    n = sum(v[1] for v in leaves.values())
    if market is Market.KR:
        tree = [
            f"vol_ratio_20_120 ≤ 1.12 (p=3.1e-04, n={n})",
            "├─ mkt_ret_60 ≤ 0.031 (p=2.2e-03, n=3405)",
            "│  ├─ leaf#1 n=1812 μe +0.2% σe 7.1% ν 6.2",
            "│  └─ leaf#2 n=1593 μe +0.9% σe 6.4% ν 8.0",
            "└─ dd_250 ≤ −0.18 (p=8.7e-03, n=1805)",
            "   ├─ leaf#3 n=812 μe −1.1% σe 9.8% ν 5.1",
            "   └─ leaf#4 n=993 μe +0.4% σe 8.3% ν 7.4",
        ]
        tail = ["Excluded rows: missing 312 · load failures 1/61 (035720: too short)"]
    else:
        tree = [
            f"mkt_vol_60 ≤ 0.162 (p=1.4e-03, n={n})",
            "├─ rsi14 ≤ 68.5 (p=1.9e-02, n=3328)",
            "│  ├─ leaf#1 n=2140 μe +0.4% σe 6.2% ν 7.8",
            "│  └─ leaf#2 n=1188 μe −0.2% σe 6.9% ν 6.6",
            "└─ leaf#3 n=1422 μe +0.1% σe 8.8% ν 5.4",
        ]
        tail = ["Excluded rows: missing 188 · load failures 0/62"]
    head = [
        f"RSMT-Q {market.value} v1 (sample) · n={n:,} · leaves {len(leaves)} · root "
        f"p={'3.1e-04' if market is Market.KR else '1.4e-03'}"
    ]
    params = [f"γ={mm['gamma']:+.4f} η={mm['eta']:.2f} δ={mm['delta']:+.2f} δrel={mm['delta_rel']:+.2f}"]
    cvl = [
        f"CV(4-fold) AUC {cv['auc']:.3f} vs {cv['base_auc']:.3f} · LogLoss {cv['logloss']:.4f} vs "
        f"{cv['base_logloss']:.4f}"
        f" · CRPS {cv['crps']:.4f} vs {cv['base_crps']:.4f} · cover80 {cv['cover80']:.2f} vs {cv['base_cover80']:.2f}"
    ]
    bad = [t for t in served if t.quant.distribution is not None and t.quant.distribution.degraded]
    health = [f"Health: offsets applied {len(served) - len(bad)}/{len(served)} · collapse check passed"]
    for t in bad:
        health.append(f"⚠ {t.instrument.ticker} degraded: {', '.join(t.quant.distribution.degraded_reasons)}")
    return head + tree + params + cvl + tail + health


def _factor_model_score(
    spec: _Spec, px: pd.DataFrame, fund: Fundamentals, legacy_pillar: float, sector: str, en: bool
) -> FactorModelScore:
    """Deterministic PIT multi-factor model score (FactorModelScore shape of backtest.score_tickers)."""
    mk = _FACTOR_MODEL[spec.market]
    close = px["close"]
    traded = close * px["volume"]
    exposures: dict[str, float | None] = {
        "per": fund.per if fund.per and fund.per > 0 else None,
        "pbr": fund.pbr if fund.pbr and fund.pbr > 0 else None,
        "ep": (1.0 / fund.per) if fund.per and fund.per > 0 else None,
        "bp": (1.0 / fund.pbr) if fund.pbr and fund.pbr > 0 else None,
        "volume_trend": float(np.log(traded.iloc[-20:].mean() / traded.iloc[-120:].mean())),
        "mom_3m": float(close.iloc[-1] / close.iloc[-64] - 1.0),
        "mom_6m": float(close.iloc[-1] / close.iloc[-127] - 1.0),
    }
    peers = [s for s in _SPECS if s.sector == spec.sector]
    peer_m3 = [sample_price_history(s.ticker)["close"] for s in peers]
    exposures["sector_mom_3m"] = float(np.mean([float(c.iloc[-1] / c.iloc[-64] - 1.0) for c in peer_m3]))
    pct = float(np.clip(0.5 + 0.5 * legacy_pillar + _FM_OFFSET.get(spec.ticker, 0.0), 0.02, 0.98))
    zs = {k: float(np.clip((pct - 0.5) * 3.0 + 0.4 * (i - 2.5) / 2.5, -3, 3)) for i, k in enumerate(_FM_WEIGHTS)}
    contributions = {k: round(_FM_WEIGHTS[k] * zs[k], 4) for k in _FM_WEIGHTS}
    notes: list[str] = []
    if spec.ticker in _FM_OUTSIDE_UNIVERSE:
        notes.append(
            "outside the PIT universe — percentile against the universe distribution"
            if en
            else "PIT 유니버스 밖 종목 — 유니버스 분포 기준 백분위"
        )
    return FactorModelScore(
        as_of=AS_OF,
        score=round(sum(contributions.values()), 4),
        percentile=pct,
        pillar=float(np.clip(2.0 * (pct - 0.5), -1, 1)) * (1.0 if mk["skill_ok"] else 0.35),
        sector=sector,
        exposures=exposures,
        zscores=zs,
        contributions=contributions,
        weights=dict(_FM_WEIGHTS),
        skill_ok=mk["skill_ok"],
        oos_ic_mean=mk["oos_ic_mean"],
        oos_ic_t=mk["oos_ic_t"],
        notes=notes,
    )


def _blend_note(legacy: float, model: float, blended: float, b: float, skill_ok: bool, en: bool) -> str:
    """fusion/signal.py's factor-pillar blend note (same format; the sample does not import fusion.signal)."""
    lg, md_, bl = (round(float(v), 2) + 0.0 for v in (legacy, model, blended))
    if en:
        note = (
            f"Factor pillar blend: legacy composite {lg:+.2f}×{1 - b:.0%} + factor model {md_:+.2f}×{b:.0%} = {bl:+.2f}"
        )
        return note + ("" if skill_ok else " (model unskilled — model pillar ×0.35 applied)")
    note = f"팩터 필러 블렌드: 기존 합성 {lg:+.2f}×{1 - b:.0%} + 팩터 모델 {md_:+.2f}×{b:.0%} = {bl:+.2f}"
    return note + ("" if skill_ok else " (모델 스킬 미달 — 모델 필러 ×0.35 반영)")


def _macro_regimes(settings: Settings) -> list[MacroRegime]:
    """Global / US / Korea regimes in macro/regime.py's shape (exposure multiplier from the config's overlay)."""
    out = []
    lo = settings.macro.overlay_min_mult
    for m in _MACRO:
        score = m["score"]
        mult = 1.0 if score >= 0 or not settings.macro.overlay else 1.0 - (1.0 - lo) * min(1.0, -score)
        inds = [
            MacroIndicator(key=k, label=lbl, value=v, unit=u, change_3m=c, z=z, as_of=AS_OF, source="sample")
            for k, lbl, v, u, c, z in m["indicators"]
        ]
        region_ko = _MACRO_REGION_KO[m["region"]]
        out.append(
            MacroRegime(
                region=m["region"],
                as_of=AS_OF,
                score=score,
                label=m["label"],
                indicators=inds,
                summary=f"{region_ko} 매크로는 {m['label']}(점수 {score:+.2f})입니다 (샘플).",
                exposure_mult=round(mult, 4),
                stale=list(m.get("stale", ())),
            )
        )
    return out


def _factor_backtest(settings: Settings, markets: list[Market]) -> dict[str, Any]:
    """KR / US summaries in backtest.report_summary's shape (factor x variant rows + model OOS stats)."""
    out: dict[str, Any] = {}
    for mk in markets:
        cost = settings.backtest.cost_bps_kr if mk is Market.KR else settings.backtest.cost_bps_us
        rows = []
        for f, label, ic_t, ann, sharpe, dir_ok, bh in _BT_FACTORS[mk]:
            variants = [("raw", 0.8)] + ([] if f == "sector_mom_3m" else [("sector_neutral", 1.0)])
            for variant, k in variants:
                rows.append(
                    {
                        "factor": f,
                        "label": label,
                        "variant": variant,
                        "ic_mean": round(0.012 * ic_t * k, 4),
                        "ic_t": round(ic_t * k, 2),
                        "direction_ok": dir_ok,
                        "bh_significant": bh and variant == "sector_neutral" or (bh and f == "sector_mom_3m"),
                        "ls_net": {"ann_return": round(ann * k, 4), "sharpe": round(sharpe * k, 3), "mdd": -0.21},
                        "lo_net": {"ann_return": round(ann * k * 0.5, 4), "sharpe": round(sharpe * k * 0.6, 3)},
                    }
                )
        fm, bm = _FACTOR_MODEL[mk], _BT_MODEL[mk]
        out[mk.value] = {
            "market": mk.value,
            "as_of": "2026-08-31",
            "period": ["2016-01-29" if mk is Market.KR else "2012-01-31", "2026-08-31"],
            "n_months": 127 if mk is Market.KR else 175,
            "universe": "PIT 시가총액 상위 200" if mk is Market.KR else "PIT S&P 500",
            "cost_bps": cost,
            "bh_q": 0.10,
            "factors": rows,
            "model": {
                # the served model is the rule's pick among the candidates (backtest.model.method "auto")
                "method": "ic_weighted" if settings.backtest.model.method == "auto" else settings.backtest.model.method,
                "selection_mode": "auto" if settings.backtest.model.method == "auto" else "fixed",
                "oos_months": bm["oos_months"],
                "oos_ic_mean": fm["oos_ic_mean"],
                "oos_ic_t": fm["oos_ic_t"],
                "skill_ok": fm["skill_ok"],
                "skill_min_ic_t": settings.backtest.model.skill_min_ic_t,
                "ls_net": {"ann_return": bm["ann"], "sharpe": bm["sharpe"], "mdd": -0.18},
                "lo_net": {"ann_return": bm["ann"] * 0.5, "sharpe": bm["sharpe"] * 0.7, "mdd": -0.12},
                "weights": dict(_FM_WEIGHTS),
                "macro_used": False,
            },
            "live_as_of": "2026-09-23",
            "settings_digest": "sample",
        }
    return out


def build_sample_report(
    n_tickers: int = 6,
    *,
    settings: Settings | None = None,
    capital: float | None = CAPITAL_KRW,
    base_currency: str = "KRW",
    with_models: bool = True,
) -> PortfolioReport:
    """Build a deterministic, clearly-marked SAMPLE `PortfolioReport` with 1..12 KR+US tickers.

    Tickers alternate KR/US (005930, AAPL, 000660, NVDA, 247540, MSFT, 035420, TSLA, 051910, AMZN, 207940,
    JPM). The set deliberately exercises every visible state: Strong Buy/Buy/Hold/Sell, an RSMT-Q
    degraded row with its guardrail override (247540), an EWMA volatility fallback (207940), missing filings with
    an LLM error (035420), LLM notices (AAPL, 000660), an earnings date inside the 14-day chip window
    (005930, estimated) and outside it (AAPL, TSLA, JPM), skill badges and data warnings.
    The language follows `settings.report.language`. `with_models` adds the factor model (blended into the factor
    pillar), the macro regimes (overlay applied to sizing) and the factor-backtest summary.
    """
    if not 1 <= n_tickers <= len(_SPECS):
        raise ValueError(f"n_tickers must be in 1..{len(_SPECS)}, got {n_tickers}")
    if base_currency not in ("KRW", "USD"):
        raise ValueError(f"base_currency must be KRW or USD, got {base_currency!r}")
    settings = settings or load_settings()
    en = is_en(settings)
    specs = _SPECS[:n_tickers]
    blend_b = settings.backtest.factor_pillar_blend
    if with_models and not (math.isfinite(blend_b) and 0.0 <= blend_b <= 1.0):
        raise ValueError(f"backtest.factor_pillar_blend must be in [0, 1], got {blend_b!r}")
    h = settings.horizon_days if settings.horizon_days else HORIZON
    H = settings.fusion.target_horizon_days
    n_days = max(h, H)  # one matrix: forecast at column h-1, barrier over the first H days
    znorm = _market_normals(N_PATHS, n_days)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        txt = _spec_text(spec, en)
        inst = _instrument(spec, txt)
        px = sample_price_history(spec.ticker)
        bench = sample_benchmark_history(spec.benchmark)
        tech = _technicals(px, bench)
        fund = Fundamentals(
            as_of=AS_OF,
            source="sample",
            next_earnings_date=spec.next_earnings,
            next_earnings_is_estimate=spec.earnings_estimate,
            **{**spec.fund, "analyst_rating": txt["analyst_rating"]},
        )
        factors = _factors(spec, tech, fund, en)
        fw = settings.factors.weights
        avail = [(fw[f.name], f.score) for f in factors if f.score is not None]
        fcomp = sum(w * s for w, s in avail) / sum(w for w, _ in avail) if avail else None
        factor_pillar = float(np.clip((fcomp - 50.0) / 50.0, -1, 1)) if fcomp is not None else 0.0
        # PIT multi-factor model, blended into the fusion factor pillar exactly like fusion/signal.py
        fmodel = _factor_model_score(spec, px, fund, factor_pillar, inst.sector or "", en) if with_models else None
        fusion_factor = factor_pillar
        blend_note = ""
        if fmodel is not None:
            fusion_factor = float(np.clip((1 - blend_b) * factor_pillar + blend_b * fmodel.pillar, -1, 1))
            blend_note = _blend_note(factor_pillar, fmodel.pillar, fusion_factor, blend_b, fmodel.skill_ok, en)

        # --- RSMT-Q distribution (served mean = RSMT-Q's own) from ONE path matrix
        mm = _MARKET_MODEL[spec.market]
        leaf_path, leaf_n, leaf_mu, nu = _LEAVES[spec.market][spec.leaf]
        sigma_m = _BENCHES[spec.benchmark].vol * math.sqrt(h / TRADING_DAYS)
        beta = (1 - settings.rsmt.beta_shrink) * tech.beta_1y + settings.rsmt.beta_shrink
        sigma_tot = tech.realized_vol_60d * math.sqrt(h / TRADING_DAYS)
        sigma_e = math.sqrt(max(sigma_tot**2 - (beta * sigma_m) ** 2, (0.35 * sigma_tot) ** 2))
        rho = 0.08 if spec.market is Market.KR else 0.05
        degraded = bool(spec.degraded)
        z = 0.0 if degraded else spec.z_mom  # missing offset input -> explicit 0 + degraded flag
        mu_e_rsmt = leaf_mu + mm["gamma"] * z
        mu_rsmt_total = beta * mm["mu_m"] + mu_e_rsmt
        blend = {"rsmt": 1.0}
        mu_total = mu_rsmt_total
        mu_e = mu_total - beta * mm["mu_m"]
        paths = _simulate(
            spec,
            mu_e,
            sigma_e,
            nu,
            mm["mu_m"],
            sigma_m,
            mm["nu_m"],
            beta,
            rho,
            znorm[spec.market],
            h,
        )
        r_term, m_term = paths.r[:, h - 1], paths.m[:, h - 1]
        simple_term = np.expm1(r_term)
        q05 = float(np.quantile(simple_term, 0.05))
        cv = dict(mm["cv"])
        rsmt_skill = (
            (cv["auc"] - cv["base_auc"] >= 0.01)
            and (cv["logloss"] < cv["base_logloss"])
            and cv["gate_folds_pass"] >= cv["gate_folds_need"]  # majority of the fold layouts
        )
        dist = DistributionForecast(
            horizon_days=h,
            mu=mu_total,
            sigma=float(math.sqrt(beta**2 * sigma_m**2 + sigma_e**2 + 2 * beta * rho * sigma_m * sigma_e)),
            nu=nu,
            beta=beta,
            mu_excess=mu_e,
            sigma_excess=sigma_e,
            mu_market=mm["mu_m"],
            sigma_market=sigma_m,
            rho=rho,
            blend=blend,
            prob_up=_sigmoid(_logit(float((r_term > 0).mean())) + mm["delta"]),
            prob_outperform=_sigmoid(_logit(float((r_term > m_term).mean())) + mm["delta_rel"]),
            expected_return=float(simple_term.mean()),
            q10=float(np.quantile(simple_term, 0.10)),
            q50=float(np.quantile(simple_term, 0.50)),
            q90=float(np.quantile(simple_term, 0.90)),
            var95=q05,
            cvar95=float(simple_term[simple_term <= q05].mean()),
            direction_edge=mm["delta"],
            relative_edge=mm["delta_rel"],
            leaf_id=spec.leaf,
            leaf_path=list(leaf_path),
            leaf_n=leaf_n,
            offsets={
                "gamma": mm["gamma"],
                "eta": mm["eta"],
                "z_mu": z,
                "v_sigma": round(math.log(tech.realized_vol_60d / 0.30), 3),
            },
            cv_metrics=cv,
            skill_ok=rsmt_skill,
            direction_skill_ok=rsmt_skill,  # the sample's gate is the direction gate (AUC / log-loss / folds)
            degraded=degraded,
            degraded_reasons=list(txt["degraded"]),
        )
        zz = dist.mu / dist.sigma
        ml_pillar = 0.5 * math.tanh(2 * zz) + 0.5 * (2 * dist.prob_up - 1)
        ml_pillar *= (1.0 if dist.direction_skill_ok else 0.35) * (0.5 if degraded else 1.0)
        ml_pillar = float(np.clip(ml_pillar, -1, 1))

        quant = QuantSignal(
            technicals=tech,
            factors=factors,
            factor_composite=fcomp,
            factor_pillar=factor_pillar,
            distribution=dist,
            ml_pillar=ml_pillar,
            vol_forecast_annual=tech.realized_vol_60d,  # the sample's stand-in for the GARCH sizing vol
            factor_model=fmodel,
        )

        # --- LLM layer
        news = None
        if txt["news"] is not None:
            nd = txt["news"]
            news = NewsSentiment(
                score=nd["score"],
                score_100=int(round(nd["score"] * 100)),
                n_articles=nd["n_articles"],
                n_used=nd["n_used"],
                trend_7d=nd["trend_7d"],
                key_positives=nd["key_positives"],
                key_negatives=nd["key_negatives"],
                key_events=nd["key_events"],
            )
        filings = None
        if txt["filings"] is not None:
            fd = txt["filings"]
            filings = FilingIntelligence(
                summary=fd["summary"],
                earnings_highlights=fd["earnings_highlights"],
                guidance=fd["guidance"],
                tone=fd["tone"],
                risk_factors=[RiskFactor(**r) for r in fd["risk_factors"]],
                sources=fd["sources"],
            )
        llm_p = _llm_pillar(news, filings)
        critic = CriticReview(
            thesis=txt["thesis"],
            bull_points=(news.key_positives if news else [])[:2],
            bear_points=(news.key_negatives if news else [])[:2],
            contradictions=list(txt["contradictions"]),
            confidence_adjustment=spec.critic_adj,
        )
        cost = round(
            0.045 + 0.0035 * (news.n_articles if news else 0) + (0.06 if filings else 0.02) + 0.035,
            4,
        )
        llm = LLMIntelligence(
            news=news,
            filings=filings,
            critic=critic,
            llm_pillar=llm_p or 0.0,
            cost_usd=cost,
            errors=list(txt["llm_errors"]),
            notices=list(txt["llm_notices"]),
        )

        # --- fusion
        pillars = {"factor": fusion_factor, "ml": ml_pillar}
        if llm_p is not None:
            pillars["llm"] = llm_p
        wsum = sum(settings.fusion.weights[k] for k in pillars)
        composite = float(
            np.clip(
                sum(settings.fusion.weights[k] * p for k, p in pillars.items()) / wsum,
                -1,
                1,
            )
        )
        err_word = "LLM error" if en else "LLM 오류"
        warnings = list(txt["warnings"]) + [f"{err_word}: {e}" for e in txt["llm_errors"]]
        if degraded:
            reasons = ", ".join(txt["degraded"])
            warnings.append(
                f"RSMT-Q degraded mode: {reasons} — confidence −10"
                if en
                else f"RSMT-Q 저하 모드: {reasons} — 신뢰도 −10"
            )
        conf, breakdown = _confidence(
            settings,
            pillars,
            composite,
            cv,
            len(spec.warnings),
            fund,
            news,
            spec.critic_adj,
            degraded,
        )
        overrides: list[str] = []
        if degraded:
            overrides.append(f"{'RSMT degraded mode' if en else 'RSMT 저하 모드'}: {', '.join(txt['degraded'])}")
        signal = _raw_signal(settings, composite, conf, overrides)

        # --- target price (ml / analyst / valuation components scaled to H)
        comps: dict[str, float] = {"ml": spec.price * math.exp(dist.mu * H / h)}
        if fund.analyst_target_price:
            comps["analyst"] = spec.price * (1 + (fund.analyst_target_price / spec.price - 1) * H / TRADING_DAYS)
        if fund.per and fund.per > 0 and fund.sector_per:
            fair = spec.price * fund.sector_per / fund.per
            comps["valuation"] = spec.price * (1 + (fair / spec.price - 1) * H / TRADING_DAYS)
        tb = settings.fusion.target_price_blend
        target = sum(tb[k] * v for k, v in comps.items()) / sum(tb[k] for k in comps)

        # --- execution plan + barrier on the SAME path matrix
        entry, atr = spec.price, tech.atr
        stop_dist = float(
            np.clip(
                settings.risk.stop_atr_mult * atr,
                settings.risk.min_stop_pct * entry,
                settings.risk.max_stop_pct * entry,
            )
        )
        stop = entry - stop_dist
        if tech.support and entry - 1.3 * stop_dist <= tech.support <= entry - 0.5 * stop_dist:
            stop = tech.support * 0.99
        tp = max(target, entry + settings.risk.min_reward_risk * (entry - stop))
        # same rule as fusion/risk.py: the 12-month analyst high is scaled to the holding horizon H
        high = fund.analyst_target_high
        high_h = entry * (1 + (high / entry - 1) * H / TRADING_DAYS) if high else 0.0
        tp2 = max(high_h, entry + 3 * (entry - stop))
        tp2 = tp2 if tp2 > tp else None
        barrier = _barrier(paths, stop / entry - 1, tp / entry - 1, H)
        margin = settings.fusion.barrier_guard_margin
        if _ORDER.index(signal) >= _ORDER.index(Signal.BUY) and barrier.p_stop_loss > barrier.p_take_profit + margin:
            new = _ORDER[_ORDER.index(signal) - 1]
            overrides.append(
                f"Barrier guardrail ({H} trading days): P(stop first) {barrier.p_stop_loss:.0%} > "
                f"P(take-profit) {barrier.p_take_profit:.0%}+{margin * 100:.0f}pp → {new.value}"
                if en
                else f"배리어 가드레일({H}거래일): P(손절 선도달) {barrier.p_stop_loss:.0%} > "
                f"P(익절) {barrier.p_take_profit:.0%}+{margin * 100:.0f}%p → {new.value}"
            )
            signal = new
        if blend_note:
            overrides.append(blend_note)
        vol_for_w = quant.vol_forecast_annual or tech.realized_vol_60d
        raw_w = (
            (max(composite, 0) * conf / 100) / max(vol_for_w, 0.10)
            if signal in (Signal.BUY, Signal.STRONG_BUY)
            else 0.0
        )
        execsig = ExecutiveSignal(
            signal=signal,
            composite=composite,
            confidence=conf,
            target_price=target,
            upside=target / spec.price - 1,
            horizon_days=H if signal in (Signal.BUY, Signal.STRONG_BUY) else (h if signal is Signal.HOLD else 10),
            horizon_label=(_HORIZON_LABEL_EN if en else _HORIZON_LABEL)[signal],
            pillars=pillars,
            target_components=comps,
            confidence_breakdown=breakdown,
            overrides=overrides,
            thesis=txt["thesis"],
        )
        rows.append(
            {
                "spec": spec,
                "inst": inst,
                "tech": tech,
                "fund": fund,
                "quant": quant,
                "llm": llm,
                "exec": execsig,
                "paths": paths,
                "entry": entry,
                "atr": atr,
                "stop": stop,
                "tp": tp,
                "tp2": tp2,
                "barrier": barrier,
                "raw_w": raw_w,
                "warnings": warnings,
                "txt": txt,
            }
        )

    # --- sizing: the production rule (fusion.risk.conviction_weights - one common conviction scale,
    # caps min(max_weight, kelly), max_gross, vol target, min-weight drop-and-resolve). Imported lazily so
    # the sample still renders when fusion is broken (doctor); that fallback is noted on the report.
    risk = settings.risk
    macro = _macro_regimes(settings) if with_models else []
    overlay_notes: list[str] = []
    if macro and settings.macro.enabled and settings.macro.overlay:
        # pipeline.macro_overlay's rule: max_gross x mean exposure multiplier of the Buy/Strong Buy names' markets
        by_region = {m.region: m for m in macro}
        region = {Market.KR: "kr", Market.US: "us"}
        mults = [
            by_region[region[r["spec"].market]].exposure_mult
            for r in rows
            if r["exec"].signal in (Signal.BUY, Signal.STRONG_BUY)
        ]
        mult = float(sum(mults) / len(mults)) if mults else 1.0
        if mult < 0.999:
            sized = round(risk.max_gross * mult, 4)
            names = _MACRO_REGION_EN if en else _MACRO_REGION_KO
            detail = " · ".join(
                f"{names[k]} {(_MACRO_LABEL_EN if en else {}).get(by_region[k].label, by_region[k].label)}"
                f"({by_region[k].score:+.2f})"
                for k in ("kr", "us")
                if k in by_region and by_region[k].score is not None
            )
            overlay_notes.append(
                f"Macro overlay: gross limit {risk.max_gross:.0%} → {sized:.0%} (×{mult:.2f}; {detail}) — risk "
                "control only, no backtested return gain"
                if en
                else f"매크로 오버레이: 총 투자 한도 {risk.max_gross:.0%} → {sized:.0%} (배수 {mult:.2f}; {detail}) — "
                "백테스트상 수익 개선 근거는 없고 리스크 관리용"
            )
            risk = risk.model_copy(update={"max_gross": sized})
    raw = np.array([r["raw_w"] for r in rows])
    kelly = np.array([r["barrier"].kelly_optimal * risk.kelly_cap for r in rows])
    caps = np.minimum(risk.max_weight, np.maximum(kelly, 0.0))
    # covariance of the simulated h-day simple returns (stands in for fusion's 252-day sample covariance)
    R = np.column_stack([np.expm1(r["paths"].r[:, h - 1]) for r in rows])  # (n_paths, n)
    cov_h = np.atleast_2d(np.cov(R, rowvar=False))

    def _vol_ann(weights: np.ndarray) -> float:
        return float(math.sqrt(max(float(weights @ cov_h @ weights), 0.0)) * math.sqrt(TRADING_DAYS / h))

    sizing_notes: list[str] = []
    try:
        from quantcio.fusion.risk import conviction_weights
    except Exception as exc:  # noqa: BLE001 - doctor must still get a sample report
        norm = raw / raw.sum() if raw.sum() > 0 else raw
        w = np.minimum(norm, caps) * risk.max_gross
        vol_ann = _vol_ann(w)
        if vol_ann > risk.target_portfolio_vol and vol_ann > 0:
            w *= risk.target_portfolio_vol / vol_ann
        w = np.where(w < risk.min_weight, 0.0, w)
        sizing_notes.append(
            f"Sample weights: fusion.risk failed to load ({type(exc).__name__}) — simple normalisation used "
            "(differs from the production rule)"
            if en
            else f"샘플 비중: fusion.risk 로드 실패({type(exc).__name__}) — 단순 정규화 규칙으로 대체(운영 규칙과 다름)"
        )
    else:
        ann = TRADING_DAYS / h
        w = conviction_weights(raw, caps, lambda idx: cov_h[np.ix_(idx, idx)] * ann, risk).weights
    # VaR / CVaR / P(loss) as NORMAL approximations, exactly like fusion.risk.portfolio_risk (the report
    # labels them "정규근사"); the mean is each name's h-day expected return from its distribution
    vol_ann = _vol_ann(w)
    mu_p = float(sum(wi * r["quant"].distribution.expected_return for wi, r in zip(w, rows)))
    sigma_h = vol_ann * math.sqrt(h / TRADING_DAYS)
    z95, phi_z95 = 1.6448536269514722, 0.10313564037537146
    portfolio_risk = {
        "gross": float(w.sum()),
        "n_positions": float((w > 0).sum()),
        "ex_ante_vol": vol_ann,
        "expected_return": mu_p,
        "var95": mu_p - z95 * sigma_h,
        "cvar95": mu_p - sigma_h * phi_z95 / 0.05,
        "p_loss": (0.5 * (1.0 + math.erf((-mu_p / sigma_h) / math.sqrt(2.0))) if sigma_h > 0 else float(mu_p < 0)),
    }

    tickers: list[TickerReport] = []
    for r, wi in zip(rows, w):
        spec: _Spec = r["spec"]
        txt = r["txt"]
        inst: Instrument = r["inst"]
        entry, stop, tp = r["entry"], r["stop"], r["tp"]
        fx = 1.0 if inst.currency == base_currency else (USDKRW if base_currency == "KRW" else 1 / USDKRW)
        shares = pv = None
        if capital and wi > 0:
            pv = float(wi * capital)
            shares = int(pv // (entry * fx))
        sig: Signal = r["exec"].signal
        ccy = inst.currency
        if en:
            action, triggers = _plan_text_en(sig, stop, tp, r["tech"].ma50, ccy, spec.next_earnings)
        elif sig in (Signal.BUY, Signal.STRONG_BUY):
            action = (
                "2회 분할매수 (진입구간 하단 우선) · 종가 기준 손절"
                if sig is Signal.STRONG_BUY
                else "진입구간 내 분할매수 · 종가 기준 손절"
            )
            triggers = [
                f"종가 기준 손절가 {_fmt_money(stop, ccy)} 이탈 시 전량 청산",
                f"익절가 {_fmt_money(tp, ccy)} 도달 시 50% 차익실현 후 손절가를 진입가로 상향",
                "뉴스 감성 −30 이하로 급락 시 비중 재검토",
            ]
        elif sig is Signal.HOLD:
            action = "신규 진입 보류 · 보유분 유지, 1개월 후 재평가"
            triggers = [
                "종합점수 +0.15 상향 돌파 시 매수 전환 검토",
                f"종가 {_fmt_money(stop, ccy)} 이탈 시 보유분 축소",
            ]
        else:
            action = "보유 시 반등 구간에서 비중 축소 · 신규 매수 금지"
            triggers = [
                f"반등 시 {_fmt_money(r['tech'].ma50, ccy)}(MA50) 부근 분할 매도",
                "종합점수 −0.15 상향 이탈 시 매도 의견 해제",
            ]
        if not en and spec.next_earnings is not None and spec.next_earnings >= AS_OF:  # fusion/risk.py rule
            triggers.append(f"실적발표({spec.next_earnings:%Y-%m-%d}) D-3 이내 신규진입 자제")
        triggers += list(txt["extra_triggers"])
        plan = ExecutionPlan(
            entry_price=entry,
            entry_low=entry - 0.5 * r["atr"],
            entry_high=entry + 0.25 * r["atr"],
            stop_loss=stop,
            stop_loss_pct=stop / entry - 1,
            take_profit=tp,
            take_profit_pct=tp / entry - 1,
            take_profit_2=r["tp2"],
            reward_risk=(tp - entry) / (entry - stop),
            barrier=r["barrier"],
            weight=float(wi),
            raw_weight=float(r["raw_w"]),
            kelly_fraction=r["barrier"].kelly_optimal * risk.kelly_cap,
            shares=shares,
            position_value=pv,
            action=action,
            triggers=triggers,
        )
        tickers.append(
            TickerReport(
                instrument=inst,
                as_of=AS_OF,
                last_price=entry,
                fundamentals=r["fund"],
                quant=r["quant"],
                llm=r["llm"],
                executive=r["exec"],
                execution=plan,
                warnings=r["warnings"],
            )
        )

    markets = sorted({t.instrument.market for t in tickers}, key=lambda m: m.value)
    regime: dict[str, str] = {}
    for bench_id, label in (
        ("KOSPI", "KOSPI"),
        ("KOSDAQ", "KOSDAQ"),
        ("SPY", "S&P 500"),
    ):
        if any(t.instrument.benchmark == bench_id for t in tickers) or (bench_id == "KOSPI" and Market.KR in markets):
            regime[label] = _regime(bench_id, en)
    if Market.US in markets:
        regime["VIX"] = "15.2 (low vol)" if en else "15.2 (저변동)"
    if base_currency == "KRW" and Market.US in markets:
        regime["USD/KRW"] = f"{USDKRW:,.0f}"

    diagnostics = {m.value: _diagnostics(m, [t for t in tickers if t.instrument.market is m], en) for m in markets}
    return PortfolioReport(
        generated_at=GENERATED_AT,
        base_currency=base_currency,
        capital=capital,
        tickers=tickers,
        cash_weight=float(1.0 - w.sum()),
        market_regime=regime,
        model_diagnostics=diagnostics,
        portfolio_risk=portfolio_risk,
        macro=macro,
        factor_backtest=_factor_backtest(settings, markets) if with_models else {},
        llm_cost_usd=round(sum(t.llm.cost_usd for t in tickers), 4),
        config_digest=settings.digest(),
        notes=[SAMPLE_NOTE_EN if en else SAMPLE_NOTE, *overlay_notes, *sizing_notes],
    )


def _plan_text_en(
    sig: Signal, stop: float, tp: float, ma50: float | None, ccy: str, earnings: date | None
) -> tuple[str, list[str]]:
    """English action + triggers of the sample plan (same rules as the Korean text in build_sample_report)."""
    if sig in (Signal.BUY, Signal.STRONG_BUY):
        action = (
            "Scale in twice (lower half of the entry zone first) · closing stop"
            if sig is Signal.STRONG_BUY
            else "Scale in within the entry zone · closing stop"
        )
        triggers = [
            f"Exit fully on a close below the stop {_fmt_money(stop, ccy, True)}",
            f"Take 50% profit at {_fmt_money(tp, ccy, True)}, then raise the stop to the entry price",
            "Review the weight if news sentiment drops to −30 or below",
        ]
    elif sig is Signal.HOLD:
        action = "No new entry · keep holdings, reassess in 1 month"
        triggers = [
            "Consider switching to Buy if the composite rises above +0.15",
            f"Trim holdings on a close below {_fmt_money(stop, ccy, True)}",
        ]
    else:
        action = "Trim into rebounds · no new buying"
        triggers = [
            f"Sell in tranches near {_fmt_money(ma50 or 0.0, ccy, True)} (MA50) on a rebound",
            "Lift the Sell call once the composite recovers above −0.15",
        ]
    if earnings is not None and earnings >= AS_OF:  # same rule as fusion/risk.py
        triggers.append(f"Avoid new entries within D-3 of earnings ({earnings:%Y-%m-%d})")
    return action, triggers


def build_sample_charts(report: PortfolioReport) -> dict[str, str]:
    """Inline-SVG price charts (1y close, MA50/200, entry zone, SL/TP/target) for the sample tickers."""
    from quantcio.report.charts import price_chart_svg

    out: dict[str, str] = {}
    for t in report.tickers:
        if t.instrument.ticker in _SPEC_BY_TICKER:
            out[t.instrument.ticker] = price_chart_svg(
                sample_price_history(t.instrument.ticker),
                t.execution,
                t.executive.target_price,
            )
    return out
