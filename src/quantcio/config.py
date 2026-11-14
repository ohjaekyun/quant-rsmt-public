"""Typed configuration loaded from config/default.yaml (+ optional override file + .env)."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"

Effort = Literal["low", "medium", "high", "xhigh", "max"]


def _normalise(weights: dict[str, float]) -> dict[str, float]:
    total = sum(v for v in weights.values() if v > 0)
    if total <= 0:
        raise ValueError(f"weights must have a positive sum: {weights}")
    return {k: max(v, 0.0) / total for k, v in weights.items()}


# credentials / contact strings: never part of the config digest (they do not change a signal)
SECRET_DATA_KEYS = frozenset({"sec_user_agent", "dart_api_key", "data_go_kr_key", "saramin_api_key", "work24_api_key"})


class DataConfig(BaseModel):
    cache_dir: str = ".cache/quantcio"
    price_ttl_hours: float = 12
    text_ttl_hours: float = 6
    news_lookback_days: int = 21
    filing_lookback_days: int = 120
    max_news_items: int = 30
    max_filings: int = 6
    max_filing_chars: int = 18000
    sec_user_agent: str = ""
    dart_api_key: str = ""
    data_go_kr_key: str = ""  # DATA_GO_KR_KEY (공공데이터포털; NPS OpenAPI path)
    saramin_api_key: str = ""  # SARAMIN_API_KEY (job postings, forward recording)
    work24_api_key: str = ""  # WORK24_API_KEY (고용24 job postings, forward recording)


class FactorConfig(BaseModel):
    weights: dict[str, float]

    @model_validator(mode="after")
    def _norm(self) -> FactorConfig:
        self.weights = _normalise(self.weights)
        return self


class RSMTMarketConfig(BaseModel):
    universe_size: int = 60
    max_depth: int = 4
    min_leaf: int = 150
    alpha: float = 0.05
    half_life_days: float = 504
    nu_min: float = 4.5
    nu_max: float = 30
    # per-market overrides of the rsmt model switches (None = the global rsmt.<key>; quant.panel.rsmt_option)
    overlap_step: int | None = None
    multiple_testing: Literal["bonferroni", "simes", "pca"] | None = None
    pca_components: int | None = None
    target_scale: Literal["none", "ewma"] | None = None
    gamma_mode: Literal["global", "leaf"] | None = None
    gamma_shrink_k: float | None = None
    sigma_terms: list[str] | None = None
    # first date of the RSMT-Q training universe (docs/RSMT.md section 8); None = today - history_years
    history_start: date | None = None


# extra log-scale regressors of the RSMT-Q sigma equation (quant.panel.SIGMA_TERM_COLUMNS computes them)
SIGMA_TERM_NAMES = ("har_5", "har_66", "har_252", "cal_s1", "cal_c1", "cal_s4", "cal_c4")


class RSMTConfig(BaseModel):
    enabled: bool = True
    feature_set: str = "v1"
    partition_features: list[str]
    offset_mu_feature: str = "xs_mom_z"
    offset_sigma_feature: str = "ewma_vol"
    sample_step: int = 20
    trim: float = 0.10
    null_sims: int = 20000
    beta_shrink: float = 0.33
    mu_shrink_k: float = 200
    n_paths: int = 20000
    cv_folds: int = 4
    collapse_min_unique_ratio: float = 0.5
    min_leaf_dates: int = 12  # each child must span >= this many distinct sample dates (date-clustered data)
    boot_sims: int = 999  # date-cluster multiplier bootstrap draws for the split test
    skill_min_auc_gain: float = 0.01  # CV AUC gain over the baseline twin
    robust_cv_folds: list[int] = [3, 4, 5]  # majority of these fold layouts must pass the skill gate
    market_mu_prior_annual: float = 0.06  # long-run equity drift prior for mu_m
    market_mu_shrink: float = 0.8  # weight on the prior when estimating mu_m
    skill_require_date_ci: bool = True  # also require the date-block bootstrap 5% bound of the AUC gain > 0
    # model switches (docs/RSMT.md); every default is the pre-2026-09-29 behaviour, config/default.yaml sets the winners
    overlap_step: int = 0  # 0: grid every sample_step sessions; k>0: every k sessions, overlapping labels, h-blocks
    multiple_testing: Literal["bonferroni", "simes", "pca"] = "bonferroni"  # split test over the partition variables
    pca_components: int = 3  # multiple_testing = pca: K principal components of the partition variables
    target_scale: Literal["none", "ewma"] = "none"  # ewma: split on e / (EWMA vol * sqrt(h/252)), served in returns
    gamma_mode: Literal["global", "leaf"] = "global"  # leaf: per-leaf momentum gamma shrunk toward the parent node's
    gamma_shrink_k: float = 50.0  # prior strength of the leaf gamma, in independent blocks (sample dates)
    # skill gate: distribution = CRPS gain bound > 0 + coverage bands (RSMT-Q is the risk engine); direction = AUC gate
    skill_metric: Literal["direction", "distribution"] = "distribution"
    skill_cover80_band: tuple[float, float] = (0.75, 0.85)
    skill_exc95_band: tuple[float, float] = (0.02, 0.08)
    # sigma equation: log sigma_e = log sigma_leaf + eta*v + sum_j theta_j * term_j (SIGMA_TERM_NAMES; [] = eta*v only)
    sigma_terms: list[str] = []
    # the universe-fitted market model is reused for this many days (serving refreshes the features every run)
    model_max_age_days: float = 7.0
    # split-test bootstrap clusters span this many horizons of overlapping rows (1 = one h-session block): labels of
    # neighbouring blocks overlap, so h-blocks capture only ~2/3 of the score variance (docs/RSMT.md section 8)
    boot_block_horizons: int = 1
    markets: dict[str, RSMTMarketConfig]

    @model_validator(mode="after")
    def _check(self) -> RSMTConfig:
        overlap = {self.offset_mu_feature, self.offset_sigma_feature} & set(self.partition_features)
        if overlap:
            raise ValueError(f"offset features must not be partition variables: {overlap}")
        if not 0 < self.trim < 0.5:
            raise ValueError("rsmt.trim must be in (0, 0.5)")
        for scope, cfg in [("rsmt", self), *((f"rsmt.markets.{k}", v) for k, v in self.markets.items())]:
            step, k_pca, k_gamma = cfg.overlap_step, cfg.pca_components, cfg.gamma_shrink_k
            if step is not None and step < 0:
                raise ValueError(f"{scope}.overlap_step must be >= 0 (0 = off), got {step}")
            if k_pca is not None and k_pca < 1:
                raise ValueError(f"{scope}.pca_components must be >= 1, got {k_pca}")
            if k_gamma is not None and not k_gamma > 0:
                raise ValueError(f"{scope}.gamma_shrink_k must be > 0, got {k_gamma}")
        for scope, terms in [
            ("rsmt", self.sigma_terms),
            *((f"rsmt.markets.{k}", v.sigma_terms) for k, v in self.markets.items()),
        ]:
            unknown = sorted(set(terms or ()) - set(SIGMA_TERM_NAMES))
            if unknown or len(set(terms or ())) != len(terms or ()):
                raise ValueError(f"{scope}.sigma_terms must be distinct names of {SIGMA_TERM_NAMES}, got {terms}")
        if self.boot_block_horizons < 1:
            raise ValueError(f"rsmt.boot_block_horizons must be >= 1, got {self.boot_block_horizons}")
        if not self.model_max_age_days > 0:
            raise ValueError(f"rsmt.model_max_age_days must be > 0, got {self.model_max_age_days}")
        for name in ("skill_cover80_band", "skill_exc95_band"):
            lo, hi = getattr(self, name)
            if not 0.0 <= lo <= hi <= 1.0:
                raise ValueError(f"rsmt.{name} must satisfy 0 <= lo <= hi <= 1, got {(lo, hi)}")
        return self


class FusionThresholds(BaseModel):
    strong_buy: float = 0.45
    buy: float = 0.15
    sell: float = -0.15


class FusionConfig(BaseModel):
    weights: dict[str, float]
    thresholds: FusionThresholds = FusionThresholds()
    strong_buy_min_confidence: int = 65
    target_horizon_days: int = 60
    min_buy_upside: float = 0.02  # Buy+ needs target upside >= this, else Hold
    max_sell_upside: float = 0.05  # Sell with target upside above this is lifted to Hold
    barrier_guard_margin: float = 0.15  # downgrade Buy+ when P(SL first) > P(TP first) + margin
    target_price_blend: dict[str, float]
    # auto: shrink the side of the factor-model pillar the backtest's side test finds weak (FactorModelScore.weak_side)
    factor_pillar_asymmetry: Literal["auto", "off"] = "auto"

    @model_validator(mode="after")
    def _norm(self) -> FusionConfig:
        self.weights = _normalise(self.weights)
        self.target_price_blend = _normalise(self.target_price_blend)
        t = self.thresholds
        if not (t.sell < t.buy < t.strong_buy):
            raise ValueError("fusion thresholds must satisfy sell < buy < strong_buy")
        return self


class RiskConfig(BaseModel):
    atr_period: int = 14
    stop_atr_mult: float = 2.0
    min_stop_pct: float = 0.04
    max_stop_pct: float = 0.15
    min_reward_risk: float = 1.5
    target_portfolio_vol: float = 0.15
    max_weight: float = 0.12
    min_weight: float = 0.02
    max_gross: float = 0.95
    kelly_cap: float = 0.25


class LLMConfig(BaseModel):
    model: str = "claude-opus-5"
    effort: dict[str, Effort] = {"news": "low", "filings": "medium", "critic": "high"}
    max_tokens: dict[str, int] = {"news": 8000, "filings": 12000, "critic": 8000}
    concurrency: int = 4
    news_half_life_days: float = 7
    min_relevance: float = 0.3
    server_fallbacks: bool = True
    web_research: bool = False
    response_cache: bool = True
    # input budgets: what is SENT to the model (data.max_news_items / max_filings / max_filing_chars stay the fetch
    # limits). Measured 2026-09-27 on 4 tickers: news 14%, filings 47%, critic 39% of ~53k input tokens per ticker
    news_max_articles: int = 20  # after merging near-duplicate stories (the same story from several outlets)
    news_dedupe_similarity: float = 0.55  # title bigram Jaccard at/above which two articles are one story (>= 1: off)
    # filing budgets are in ESTIMATED tokens (llm.filings.approx_tokens: Hangul ~1.2, Latin letters ~0.3, other ~0.6
    # per char), so a Korean and an English filing cost about the same
    filing_max_tokens: int = 3000  # per filing: head + the paragraphs with figures / key terms (salient excerpt)
    filings_total_tokens: int = 10000  # all filings of one ticker together
    critic_max_news: int = 6  # articles re-shown to the critic: highest relevance x |sentiment| first
    critic_max_filings: int = 3  # filings re-shown to the critic: those behind high / critical risks first
    critic_filing_tokens: int = 1000


class ReportConfig(BaseModel):
    language: Literal["ko", "en"] = "ko"  # report / LLM output language (CLI --lang)
    max_pages: int = 3
    output_dir: str = "reports"
    # large universes: with more than `compact_max_names` names the report is NOT squeezed into max_pages (charts,
    # news and risk detail would be dropped and it would still overflow - seen 2026-09-27: 20 names at maximum
    # compaction were 8 pages). Full content is printed on as many pages as needed, up to max_pages_extended
    auto_extend: bool = True
    compact_max_names: int = 8
    max_pages_extended: int = 16


class PITKRConfig(BaseModel):
    start_year: int = 2013  # first marcap year to load (one warm-up year before the evaluation window)
    rank_by: Literal["marcap", "marcap_mean60"] = "marcap"  # ranking variable on each rebalance date
    exclude_admin: bool = True  # drop KOSDAQ 관리종목 / 투자주의환기 on the rebalance date
    min_sessions: int = 120  # a stock must have traded this many sessions in its current listing spell


class PITUSConfig(BaseModel):
    index: Literal["sp500"] = "sp500"  # PIT membership source (fja05680 snapshots + Wikipedia patch)
    ftd_fill: bool = True  # SEC fails-to-deliver prices for terminal/delisting marks when Yahoo has no series


class PITFundamentalsConfig(BaseModel):
    kr_fallback_lag_days_quarter: int = 60  # when the DART receipt date is unknown (never +45; measured early)
    kr_fallback_lag_days_annual: int = 120
    max_staleness_days: int = 200  # values older than this (period end -> t) are treated as missing


class PITConfig(BaseModel):
    enabled: bool = True
    universe_mode: Literal["pit", "current"] = "pit"  # RSMT-Q training panel: PIT members per date, or today's list
    rebalance: Literal["month_end"] = "month_end"
    kr: PITKRConfig = PITKRConfig()
    us: PITUSConfig = PITUSConfig()
    fundamentals: PITFundamentalsConfig = PITFundamentalsConfig()


class HoldoutConfig(BaseModel):
    registry_path: str = "holdout/registry.json"  # git-tracked, append-only
    ledger_dir: str = "holdout/ledger"  # git-tracked, append-only predictions/outcomes
    min_dates_for_verdict: int = 12  # non-overlapping snapshot dates required before any verdict
    bh_q: float = 0.10  # Benjamini-Hochberg level over the pre-registered primary metrics
    record_reports: bool = True  # append every `quantcio report` run to the forward ledger
    # the user's target for the stock-selection model: within-date AUC vs 1{excess > date median}. Reported next
    # to the pre-registered significance tests (docs/HOLDOUT.md "XSH"); it is a goal, not a test
    target_auc: float = 0.55
    # KR cross-sectional holdout: point-in-time market-cap ranks never used for tuning (the modeling / factor
    # backtest universe is the PIT top 200 = backtest.kr_universe_n)
    xsh_kr_rank_band: tuple[int, int] = (201, 400)


class BacktestModelConfig(BaseModel):
    # auto = run the pre-declared candidate comparison and serve the rule's choice (docs/BACKTEST_MACRO.md 3.1);
    # a candidate name serves that candidate (the comparison is still computed and reported)
    method: Literal[
        "auto",
        "equal_weight",
        "ic_weighted",
        "ic_weighted_60",
        "ic_weighted_raw",
        "fama_macbeth",
        "ridge",
        "macro_conditional",
        "rsmt_tree",
    ] = "auto"
    lookback_months: int = 36  # ic_weighted / macro_conditional / ic_weighted_raw trailing window of months
    min_months: int = 24  # no model output before this many past months exist
    shrink_to_equal: float = 0.5  # weights = (1-s)*fitted + s*equal (sign-constrained to the a-priori direction)
    skill_min_ic_t: float = 2.0  # OOS IC t-stat needed for skill_ok (full pillar weight); else x0.35
    holdout_months: int = 24  # last OOS months reported but never used to pick the candidate (< 0 counts as 0)


class BacktestConfig(BaseModel):
    kr_universe_n: int = 200  # PIT top-N by market cap (KR)
    us_universe_n: int = 500  # PIT S&P 500 members (500 = all)
    kr_start: str = "2016-01-01"  # KR PIT fundamentals start 2015 (TTM needs 4 quarters)
    us_start: str = "2012-01-01"
    quantiles: int = 5
    cost_bps_kr: float = 30.0  # round-trip cost per 100% turnover incl. KR securities transaction tax
    cost_bps_us: float = 10.0
    sector_neutral: bool = True  # rank factors within sector before combining
    min_names_per_sector: int = 5  # smaller sectors merge into "기타" for neutralisation
    momentum_windows: list[int] = [63, 126]  # 3M, 6M in sessions
    volume_windows: list[int] = [20, 120]  # volume trend = log(ADV20 / ADV120) of traded value
    factor_pillar_blend: float = (
        0.5  # weight of the model pillar inside the fusion factor pillar (rest: legacy composite)
    )
    model: BacktestModelConfig = BacktestModelConfig()


class ModelingConfig(BaseModel):
    """Stock-selection model universe / cost (docs/MODELING.md 80.8, user decision 2026-10-25): KR top 400 by market
    cap (ranks 201-400 carried most of the signal, 80.6-80.7) at a 50bp round trip for the smaller names. The PIT
    factor backtest keeps ``backtest.kr_universe_n``; US uses the backtest settings."""

    kr_universe_n: int = 400
    kr_cost_bps: float = 50.0
    # frozen first-seen copies of the revised inputs (docs/MODELING.md 84.2, user decision 2026-10-29)
    vintage: bool = True
    vintage_settle_days: int = 5  # a month is frozen once its last day is at least this many days old


class AltConfig(BaseModel):
    """Alternative data for the stock-selection features v3 (docs/MODELING.md 16)."""

    enabled: bool = True
    kr_dart: bool = True  # OpenDART disclosures / ownership / employees (DART_API_KEY)
    kr_nps: str = "auto"  # auto: only when past monthly files reach back to <= nps_min_month (16.3 rule) | on | off
    nps_min_month: str = "2015-12"
    us_sec: bool = True  # SEC submissions forms / 8-K items + Form 3/4/5 insider data sets (SEC_USER_AGENT)
    refresh_days: int = 7  # cached open ranges older than this are refetched
    macro_betas: bool = True  # per-stock macro exposure features


class MacroConfig(BaseModel):
    enabled: bool = True
    lookback_years: int = 10  # history for z-scores and the regime backtest
    z_window_years: int = 3  # trailing window for standardising indicators (data <= t only)
    ttl_hours: float = 12
    overlay: bool = True  # scale max_gross by the market's regime exposure multiplier
    overlay_min_mult: float = 0.6  # multiplier at the most risk-off score (-1); 1.0 at score >= 0
    risk_off_threshold: float = -0.25  # score <= this -> "위험회피"
    risk_on_threshold: float = 0.25  # score >= this -> "위험선호"


class Settings(BaseModel):
    horizon_days: int = 20
    history_years: int = 7
    data: DataConfig = DataConfig()
    factors: FactorConfig
    rsmt: RSMTConfig
    fusion: FusionConfig
    risk: RiskConfig = RiskConfig()
    llm: LLMConfig = LLMConfig()
    report: ReportConfig = ReportConfig()
    pit: PITConfig = PITConfig()
    holdout: HoldoutConfig = HoldoutConfig()
    backtest: BacktestConfig = BacktestConfig()
    modeling: ModelingConfig = ModelingConfig()
    macro: MacroConfig = MacroConfig()
    alt: AltConfig = AltConfig()

    def digest(self) -> str:
        """Short hash of the effective config (printed on the report for reproducibility)."""
        # presentation / infrastructure settings (output language, page limit, folders, holdout paths) never change a
        # signal, so they must not change the digest that keys model caches and the holdout freeze
        payload = self.model_dump(
            exclude={
                "data": {"sec_user_agent", "dart_api_key", "data_go_kr_key", "saramin_api_key", "work24_api_key"},
                "report": True,
                "holdout": True,
            }
        )
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]

    @property
    def cache_path(self) -> Path:
        p = Path(self.data.cache_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(override_path: Path | None = None, **overrides: Any) -> Settings:
    """Load default.yaml, deep-merge an optional override YAML and keyword overrides, expand ${ENV}."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) or {}
    if override_path:
        raw = _deep_merge(raw, yaml.safe_load(Path(override_path).read_text(encoding="utf-8")) or {})
    if overrides:
        raw = _deep_merge(raw, overrides)
    raw = _expand_env(raw)
    raw.setdefault("data", {})["dart_api_key"] = os.environ.get("DART_API_KEY", "")
    for key, env in (
        ("data_go_kr_key", "DATA_GO_KR_KEY"),
        ("saramin_api_key", "SARAMIN_API_KEY"),
        ("work24_api_key", "WORK24_API_KEY"),
    ):
        raw["data"][key] = os.environ.get(env, "")
    settings = Settings.model_validate(raw)
    # a single horizon drives factors, RSMT-Q and the report; the RSMT-Q sample grid must cover label overlap
    settings.rsmt.sample_step = max(settings.rsmt.sample_step, settings.horizon_days)
    return settings
