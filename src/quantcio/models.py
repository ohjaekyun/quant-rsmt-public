"""Shared domain contracts for every QuantCIO module.

Conventions (all modules MUST follow):
- Prices are split/dividend adjusted, in the instrument's local currency.
- Price DataFrames: tz-naive DatetimeIndex of exchange-local trading dates, ascending,
  unique, columns exactly ["open", "high", "low", "close", "volume"] (float64).
- Returns / percentages are decimals (0.05 == 5%) unless a field name ends in `_pct`.
- Datetimes on text items are tz-aware UTC.
- Scores: factor scores 0..100 (50 = neutral); pillar scores -1..+1 (0 = neutral).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


class Market(StrEnum):
    KR = "KR"
    US = "US"


class Signal(StrEnum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# --------------------------------------------------------------------------- data


class Instrument(BaseModel):
    ticker: str  # canonical: "005930" (KR) or "AAPL" (US)
    market: Market
    name: str  # display name (Korean for KR)
    exchange: str | None = None  # KOSPI / KOSDAQ / NASDAQ / NYSE ...
    currency: str  # "KRW" / "USD"
    sector: str | None = None
    yf_symbol: str | None = None  # e.g. "005930.KS", "247540.KQ", "AAPL"
    benchmark: str  # benchmark id understood by the market provider (e.g. "KOSPI", "KOSDAQ", "SPY")
    benchmark_name: str  # "KOSPI" / "KOSDAQ" / "S&P 500 (SPY)"


class Fundamentals(BaseModel):
    as_of: date | None = None
    source: str = ""
    market_cap: float | None = None  # local currency
    per: float | None = None  # trailing P/E
    forward_per: float | None = None
    pbr: float | None = None
    eps: float | None = None  # trailing EPS, local currency
    forward_eps: float | None = None
    bps: float | None = None
    roe: float | None = None  # decimal
    operating_margin: float | None = None  # decimal
    dividend_yield: float | None = None  # decimal
    ev_ebitda: float | None = None
    revenue_growth_yoy: float | None = None  # decimal
    earnings_growth_yoy: float | None = None  # decimal
    sector_per: float | None = None  # industry/sector average P/E when available
    analyst_target_price: float | None = None
    analyst_target_high: float | None = None
    analyst_target_low: float | None = None
    analyst_count: int | None = None
    analyst_rating: str | None = None  # e.g. "Buy", "4.00 매수"
    next_earnings_date: date | None = None  # next scheduled earnings release (exchange-local date)
    next_earnings_is_estimate: bool | None = None  # True when the date is an estimate, not confirmed


def text_id(*parts: str) -> str:
    return hashlib.sha1("|".join(p or "" for p in parts).encode("utf-8")).hexdigest()[:12]


class NewsItem(BaseModel):
    id: str  # text_id(url or title+published_at)
    title: str
    summary: str = ""  # snippet / lead / body excerpt (plain text)
    url: str = ""
    source: str = ""  # publisher
    published_at: datetime  # tz-aware UTC
    lang: Literal["ko", "en"] = "ko"


class FilingItem(BaseModel):
    id: str
    title: str
    form_type: str  # "8-K", "10-Q", "10-K", "영업(잠정)실적(공정공시)", "주요사항보고서" ...
    filed_at: datetime  # tz-aware UTC
    url: str = ""
    text: str = ""  # cleaned plain text (may be truncated, see data.max_filing_chars)
    is_earnings: bool = False  # earnings release / 잠정실적 / 8-K Item 2.02
    source: str = ""  # "DART", "SEC EDGAR", "Naver"


@dataclass
class MarketData:
    """Everything the analytics layers need for one instrument. Produced by data providers."""

    instrument: Instrument
    prices: pd.DataFrame  # PRICE_COLUMNS, see module docstring
    benchmark: pd.DataFrame  # same format, the instrument's benchmark index
    fundamentals: Fundamentals
    valuation_history: pd.DataFrame | None = None  # index date, optional columns "per", "pbr"
    news: list[NewsItem] = field(default_factory=list)
    filings: list[FilingItem] = field(default_factory=list)
    fx_to_base: float = 1.0  # multiply local-currency amounts to get base currency
    warnings: list[str] = field(default_factory=list)

    @property
    def last_price(self) -> float:
        return float(self.prices["close"].iloc[-1])

    @property
    def as_of(self) -> date:
        return self.prices.index[-1].date()


# ------------------------------------------------------------------------- quant


class FactorScore(BaseModel):
    name: Literal["momentum", "value", "volatility", "quality", "trend"]
    label: str  # Korean display label, e.g. "모멘텀"
    score: float | None = Field(default=None, ge=0, le=100)  # None when no inputs available
    metrics: dict[str, float | None] = {}  # raw inputs used (decimals / ratios)
    rationale: str = ""  # one short deterministic Korean sentence


class Technicals(BaseModel):
    last_close: float
    atr: float  # ATR(risk.atr_period), price units
    atr_pct: float  # atr / last_close
    rsi14: float | None = None
    ma50: float | None = None
    ma200: float | None = None
    high_52w: float
    low_52w: float
    realized_vol_60d: float  # annualised, decimal
    beta_1y: float | None = None
    max_drawdown_1y: float  # negative decimal
    ret_1m: float | None = None
    ret_3m: float | None = None
    ret_12m_ex1m: float | None = None
    rel_strength_3m: float | None = None  # 3m return minus benchmark 3m return
    support: float | None = None  # recent swing low (e.g. 20d low)
    resistance: float | None = None  # recent swing high (e.g. 60d high)


class DistributionForecast(BaseModel):
    """RSMT-Q headline forecast: ONE return distribution from which every derived quantity comes.

    All probabilities / quantiles / VaR below are computed from the same simulated path matrix
    (quant.distribution.ReturnDistribution), so they cannot contradict each other.
    Offsets (gamma, eta) ARE included in mu/sigma; the direction edge (delta) is applied ONLY to
    prob_up / prob_outperform (mirrors RSMT: predict_theta has offsets but not home_edge).
    """

    model: str = "RSMT-Q"
    feature_set: str = "v1"
    horizon_days: int
    mu: float  # expected total LOG return over horizon (after offsets / the selection model's excess mean)
    sigma: float  # stdev of total log return over horizon
    nu: float  # Student-t tail parameter at the horizon (bounded [nu_min, nu_max])
    beta: float  # shrunk trailing beta used in r = beta*m + e
    mu_excess: float  # leaf mu of the beta-adjusted excess return (+ gamma*z)
    sigma_excess: float
    mu_market: float
    sigma_market: float
    rho: float  # residual dependence between e and m (Dixon-Coles analogue)
    blend: dict[str, float] = {}  # mean source actually served: {"rsmt": 1.0} or {<selection model id>: 1.0}
    prob_up: float = Field(ge=0, le=1)  # P(r>0), incl. direction edge
    prob_outperform: float = Field(ge=0, le=1)  # P(r > benchmark), incl. relative edge
    expected_return: float  # E[exp(r)-1] (simple, decimal) - no edge applied
    q10: float  # simple-return quantiles (decimal)
    q50: float
    q90: float
    var95: float  # 5% quantile of simple return (negative decimal)
    cvar95: float  # mean simple return below var95
    direction_edge: float = 0.0  # delta (logit shift) applied to prob_up
    relative_edge: float = 0.0  # delta applied to prob_outperform
    leaf_id: int = -1
    leaf_path: list[str] = []  # e.g. ["vol_ratio_20_120 <= 1.12", "mkt_ret_60 > 0.031"]
    leaf_n: int = 0
    offsets: dict[str, float] = {}  # {"gamma":..,"eta":..,"z_mu":..,"v_sigma":..}
    # expanding-window CV: auc, logloss, brier, crps, cover80, ic + base_* twins
    # + robust gate gate_folds_<k>/gate_folds_n/gate_folds_pass/gate_folds_need (read by report '모델 신뢰도')
    cv_metrics: dict[str, float] = {}
    # skill_ok: RSMT-Q's own CV verdict of rsmt.skill_metric (default "distribution": CRPS / coverage of the risk
    # engine). direction_skill_ok: the skill behind the served mean's DIRECTION - the stock-selection model's served
    # skill when its excess mean is served (ExcessOverride), else RSMT-Q's direction (AUC) gate; the directional ML
    # pillar multiplier (quant.distribution_ml_pillar) reads it, never skill_ok
    skill_ok: bool = False
    direction_skill_ok: bool = False
    degraded: bool = False  # NEVER silently serve a reduced model - set this + reasons instead
    degraded_reasons: list[str] = []


class BarrierOutcome(BaseModel):
    """Trade outcome probabilities under the stop-loss / take-profit plan (sum to exactly 1)."""

    p_take_profit: float = Field(ge=0, le=1)  # TP touched first within horizon
    p_stop_loss: float = Field(ge=0, le=1)  # SL touched first within horizon
    p_time_exit: float = Field(ge=0, le=1)  # neither touched -> exit at horizon close
    expected_trade_return: float  # mean simple return of the managed trade
    # argmax_f E[log(1+f*R)] over simulated trade returns (before cap); 1.0 = grid upper bound hit (true optimum >= 1)
    kelly_optimal: float
    n_paths: int
    horizon_days: int = 0  # trading days simulated for the barrier (holding horizon = fusion.target_horizon_days)


class FactorModelScore(BaseModel):
    """PIT multi-factor model (PER/PBR/volume/3M·6M momentum/sector), walk-forward fitted.

    See docs/BACKTEST_MACRO.md.
    """

    model: str = "PIT multi-factor"
    as_of: date
    score: float  # cross-sectional composite z-score at as_of (higher = more attractive)
    percentile: float = Field(ge=0, le=1)  # rank of `score` within the PIT universe on as_of
    pillar: float = Field(ge=-1, le=1)  # mapped to the fusion factor-pillar scale (skill-shrunk)
    sector: str | None = None
    exposures: dict[str, float | None] = {}  # raw values: per, pbr, ep, bp, volume_trend, mom_3m, mom_6m, sector_mom_3m
    zscores: dict[str, float | None] = {}  # sector-neutral z of each factor
    contributions: dict[str, float] = {}  # weight * z per factor (sums to score)
    weights: dict[str, float] = {}  # model weights used on as_of (fitted on data before as_of only)
    skill_ok: bool = False  # OOS IC t-stat >= backtest.model.skill_min_ic_t over the walk-forward backtest
    oos_ic_mean: float | None = None
    oos_ic_t: float | None = None
    # pillar side test (backtest.model.pillar_side_test): 'neg' / 'pos' = that side of the pillar carries
    # significantly less information than the other (fusion shrinks it when fusion.factor_pillar_asymmetry = auto)
    weak_side: Literal["pos", "neg"] | None = None
    side_test: dict[str, Any] = {}
    notes: list[str] = []


class MacroIndicator(BaseModel):
    key: str  # e.g. "us_curve_10y_3m"
    label: str  # Korean display label
    value: float | None = None  # latest level (units in `unit`)
    unit: str = ""
    change_3m: float | None = None  # change over ~63 sessions (same unit, or decimal return for prices)
    z: float | None = None  # standardised signal (trailing window, data <= as_of only), sign-adjusted: + = risk-on
    as_of: date | None = None
    source: str = ""


class MacroRegime(BaseModel):
    region: Literal["global", "us", "kr"]
    as_of: date
    score: float | None = Field(default=None, ge=-1, le=1)  # + risk-on / - risk-off; None = 데이터 부족
    label: str  # "위험선호" / "중립" / "위험회피"
    indicators: list[MacroIndicator] = []
    summary: str = ""  # one Korean sentence
    exposure_mult: float = Field(default=1.0, ge=0, le=1)  # gross-exposure multiplier applied by the overlay
    stale: list[str] = []  # indicators whose latest value is older than 5 sessions (never silently used as fresh)
    warnings: list[str] = []  # loader failures / why the score is missing (shown on the report)


class QuantSignal(BaseModel):
    technicals: Technicals
    factors: list[FactorScore]
    factor_composite: float | None = Field(default=None, ge=0, le=100)
    factor_pillar: float = Field(ge=-1, le=1)  # (factor_composite - 50) / 50, 0 if unavailable
    distribution: DistributionForecast | None = None  # RSMT-Q headline distribution
    ml_pillar: float = Field(ge=-1, le=1)  # from the RSMT-Q distribution; 0 if none
    # GARCH(1,1)-t horizon-average annualised vol (EWMA fallback; quant.volatility) - the position-sizing vol
    vol_forecast_annual: float | None = None
    factor_model: FactorModelScore | None = None  # PIT multi-factor model score (backtested)
    excess_model: ExcessForecast | None = None  # selected stock-selection model (docs/MODELING.md)


class ExcessForecast(BaseModel):
    """Selected stock-selection model's view of one ticker (docs/MODELING.md, quantcio.modeling.serve).

    The served ReturnDistribution uses ``mu_excess`` as its beta-adjusted excess mean (market leg separate), so
    P(beat benchmark) / P(up) come from the same path matrix as everything else (DistributionForecast).
    """

    model: str  # candidate id, e.g. "lgbm_rank"
    model_label: str = ""
    as_of: date  # date of the scored cross-section
    horizon_days: int = 20
    in_universe: bool = True  # False: scored from own prices placed into the PIT cross-section
    mu_excess: float  # served h-day log excess mean (calibrated, skill-shrunk)
    mu_excess_raw: float = 0.0  # calibrated, before the unskilled shrink
    score_z: float = 0.0  # per-date rank-gauss of the model score
    percentile: float = Field(default=0.5, ge=0, le=1)  # within the PIT universe on as_of
    p_beat_benchmark: float | None = None  # P(r > benchmark) from the served distribution (filled after serving)
    skill_ok: bool = False  # serving gate: beats 12-1 momentum on outer OOS AND within-date AUC lower bound > 0.5
    gate: dict[str, Any] = {}
    # protocol v4 primary (per month): top-quintile beta-adjusted excess net of cost, outer OOS / its 95% lower bound
    oos_primary: float | None = None
    oos_primary_lo: float | None = None
    holdout_primary: float | None = None
    oos_auc: float | None = None  # mean within-date AUC, pre-holdout outer OOS (secondary since protocol v4)
    oos_auc_lo: float | None = None  # its 95% date-block bootstrap lower bound
    oos_ic: float | None = None
    holdout_auc: float | None = None  # untouched final holdout (not used for selection)
    holdout_ic: float | None = None
    # P(market up): the direction model's probability when it passed its rule (then also used for the market leg),
    # else the historical base rate (reference only: the market leg keeps the prior-shrunk drift)
    market_p_up: float | None = None
    market_model_used: bool = False
    top_contributions: list[dict[str, Any]] = []  # {"feature", "label", "value"} (signed, score units)
    notes: list[str] = []


QuantSignal.model_rebuild()  # excess_model is declared after QuantSignal


# --------------------------------------------------------------------------- llm

EventType = Literal[
    "earnings",
    "guidance",
    "product",
    "contract_deal",
    "mna",
    "regulation_legal",
    "management",
    "capital_return",
    "financing_dilution",
    "macro_sector",
    "analyst_rating",
    "supply_chain",
    "accident_esg",
    "other",
]


class ArticleSentiment(BaseModel):
    news_id: str
    sentiment: float = Field(ge=-1, le=1)
    relevance: float = Field(ge=0, le=1)  # how specifically the article is about this company
    event_type: EventType
    evidence: str  # short verbatim quote from title/summary
    grounded: bool = True  # set by code: evidence found in source text


class NewsSentiment(BaseModel):
    score: float = Field(ge=-1, le=1)  # time-decayed, relevance-weighted mean (computed in code)
    score_100: int = Field(ge=-100, le=100)  # display: round(score*100)
    n_articles: int
    n_used: int  # grounded + relevant articles that entered the score
    trend_7d: float | None = None  # mean sentiment last 7d minus prior window
    key_positives: list[str] = []  # Korean one-liners
    key_negatives: list[str] = []
    key_events: list[str] = []
    articles: list[ArticleSentiment] = []


class RiskFactor(BaseModel):
    title: str
    detail: str
    severity: Literal["low", "medium", "high", "critical"]
    category: Literal[
        "financial",
        "legal_regulatory",
        "operational",
        "market_demand",
        "competition",
        "governance",
        "macro_fx",
        "liquidity_dilution",
        "other",
    ]
    source_ref: str  # filing id/title or news id it came from


class FilingIntelligence(BaseModel):
    summary: list[str] = []  # <= 3 Korean bullets
    earnings_highlights: list[str] = []  # revenue / OP / EPS YoY, surprises (numbers must come from text)
    guidance: Literal["raised", "maintained", "lowered", "none", "unknown"] = "unknown"
    tone: float = Field(default=0.0, ge=-1, le=1)
    risk_factors: list[RiskFactor] = []
    sources: list[str] = []  # filing titles/urls used


class CriticReview(BaseModel):
    thesis: str  # 1-2 sentence Korean investment thesis for the matrix
    bull_points: list[str] = []
    bear_points: list[str] = []
    contradictions: list[str] = []  # conflicts between quant and text evidence
    confidence_adjustment: int = Field(default=0, ge=-25, le=5)
    critical_flag: bool = False  # e.g. fraud, audit opinion refusal, delisting risk, going concern
    critical_reason: str | None = None


class LLMIntelligence(BaseModel):
    news: NewsSentiment | None = None
    filings: FilingIntelligence | None = None
    critic: CriticReview | None = None
    llm_pillar: float = Field(default=0.0, ge=-1, le=1)
    cost_usd: float = 0.0
    errors: list[str] = []  # hard failures (call failed, refusal, invalid output)
    notices: list[str] = []  # degraded-but-served paths (fallback model, clamps, dropped ungrounded items)
    skipped: bool = False  # True when run with --no-llm


# --------------------------------------------------------------- fusion / risk


class ExecutiveSignal(BaseModel):
    signal: Signal
    composite: float = Field(ge=-1, le=1)
    confidence: int = Field(ge=0, le=100)
    target_price: float
    upside: float  # decimal vs last price
    horizon_days: int
    horizon_label: str  # e.g. "1~3개월"
    pillars: dict[str, float]  # {"factor": .., "ml": .., "llm": ..}
    target_components: dict[str, float] = {}  # {"ml": price, "analyst": price, "valuation": price}
    confidence_breakdown: dict[str, float] = {}
    overrides: list[str] = []  # guardrail notes (Korean)
    thesis: str = ""


class ExecutionPlan(BaseModel):
    entry_price: float
    entry_low: float
    entry_high: float
    stop_loss: float
    stop_loss_pct: float  # negative decimal vs entry
    take_profit: float
    take_profit_pct: float  # positive decimal vs entry
    take_profit_2: float | None = None  # stretch target (e.g. analyst high / 3R)
    reward_risk: float
    barrier: BarrierOutcome | None = None  # from the RSMT-Q path matrix (same distribution as the signal)
    weight: float = Field(ge=0, le=1)  # recommended portfolio weight (decimal)
    raw_weight: float = 0.0  # pre-normalisation weight
    kelly_fraction: float = 0.0
    shares: int | None = None  # when capital is provided
    position_value: float | None = None  # base currency
    action: str = ""  # Korean one-liner, e.g. "분할매수 2회, 손절 이탈 시 전량 청산"
    triggers: list[str] = []  # monitoring triggers (Korean)


class TickerReport(BaseModel):
    instrument: Instrument
    as_of: date
    last_price: float
    fundamentals: Fundamentals
    quant: QuantSignal
    llm: LLMIntelligence
    executive: ExecutiveSignal
    execution: ExecutionPlan
    warnings: list[str] = []


class PortfolioReport(BaseModel):
    generated_at: datetime
    base_currency: str = "KRW"
    capital: float | None = None
    tickers: list[TickerReport]
    cash_weight: float
    market_regime: dict[str, str] = {}  # e.g. {"KOSPI": "20D +3.1%, MA200 위", "VIX": "15.2"}
    model_diagnostics: dict[str, list[str]] = {}  # per market: RSMT tree print, CV vs baseline, health checks
    portfolio_risk: dict[str, float] = {}  # ex-ante vol, VaR95, P(loss) of the sized portfolio
    macro: list[MacroRegime] = []  # global / us / kr regimes (docs/BACKTEST_MACRO.md)
    factor_backtest: dict[str, Any] = {}  # per market: factor stats table, model OOS stats, curves
    # per market: the REFERENCE ventile views, one per stored model (score ventile per ticker + the top-ventile list;
    # modeling.serve.order_for_report, docs/MODELING.md 93, 94.3) - sets nothing in the signal or the sizing.
    # Reports written before 94.3 hold one view (a dict) per market; the renderer reads both.
    order_ventile: dict[str, Any] = {}
    llm_cost_usd: float = 0.0
    config_digest: str = ""
    notes: list[str] = []
