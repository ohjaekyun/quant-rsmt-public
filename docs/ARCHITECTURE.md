# QuantCIO — Architecture & Module Contracts

**Goal:** `quantcio report --tickers 005930,000660,AAPL,NVDA` → an A4, 2–3 page Korean PDF that lets a CIO
decide in about a minute. The concept is **[structured quant signal] + [unstructured LLM sentiment] = [final timing call]**.

The report has four required modules:

1. **Executive Signal Matrix**: ticker, final signal (Strong Buy/Buy/Hold/Sell), confidence score, target price, recommended holding period
2. **Quantitative & ML Factor Signal**: time-series ML forecast, factor scores (momentum, valuation, volatility, quality, trend)
3. **LLM Unstructured Intelligence**: news sentiment score, NLP summary of earnings releases and filings, key risk factors
4. **Execution & Risk Management**: recommended portfolio weight, stop-loss, take-profit

## Design principles (accuracy first)

- **Deterministic core, bounded LLM.** Code computes the signal, target price, stop-loss, take-profit and weights. The LLM only
  (a) scores text, (b) summarises filings, and (c) critiques the call. Its influence has hard limits: `llm_pillar` is in [-1, 1],
  `confidence_adjustment` is in [-25, +5], and a `critical_flag` can only cap the signal through a fixed rule.
- **Grounding checks.** Every LLM `evidence` quote must appear in the source text. Every number in `earnings_highlights` must appear
  in the filing text. Anything that fails is dropped and counted. This defends against hallucination.
- **No look-ahead.** The ML uses walk-forward validation with an embargo of at least `horizon_days`. Features at t use data ≤ t only.
  Labels are forward returns. The out-of-sample (OOS) skill gates how much the ML pillar counts.
- **Graceful degradation.** Any optional source can fail. The failure goes into `warnings`, and missing pillars have their weights
  renormalised. Confidence drops with data quality.
- **Untrusted text.** News and filing text are wrapped in `<document>` tags. The system prompts state that document content is data
  and never instructions. This defends against prompt injection.
- **Reproducible.** Data and LLM responses are cached by input hash. The config digest is printed on the report.

## Package layout (module owner = the file's only writer)

```
src/quantcio/
  models.py            SHARED CONTRACT (read-only for module agents)
  config.py            SHARED CONTRACT (read-only for module agents)
  data/base.py         SHARED: MarketProvider ABC, clean_prices, load_market_data
  data/cache.py        SHARED: DiskCache
  data/http.py         SHARED: PoliteSession (retries, throttle, UA)
  data/kr.py           KRProvider(MarketProvider)   - prices/benchmark/fundamentals/valuation_history/news/filings/fx (shared USD/KRW definition from data/us.py)
  data/us.py           USProvider(MarketProvider)   - same, incl. SEC EDGAR filings
  quant/features.py    SHARED (written): regime_features(), forward_log_return(), ewma_vol(), wilder_rsi(), shrink_beta()
  quant/indicators.py  compute_technicals(...) -> Technicals (+ atr, rsi, realized_vol, beta, max_drawdown helpers)
  quant/factors.py     compute_factors(...) -> list[FactorScore]; factor_composite(...)
  quant/volatility.py  garch_vol_forecast(...) -> float (annualised), EWMA fallback (the position-sizing vol)
  quant/panel.py       build_panel(PricePanel, settings) -> PanelData  (RSMT-Q pooled training set)
  quant/rsmt.py        RSMTree (MOB score tree), sup-LM null cache, offsets, CV, fit_market_model(), health checks
  quant/distribution.py ReturnDistribution: ONE path matrix -> prob_up, quantiles, VaR, barrier TP/SL/time, Kelly
  quant/__init__.py    build_quant_signal(md, settings, market_model) -> QuantSignal; sizing_vol_forecast(md, settings)
  llm/client.py        LLMClient (async, Anthropic SDK), FakeLLMClient for tests, cost accounting, response cache
  llm/prompts.py       system prompts + document wrapping
  llm/news.py          analyze_news(...) -> NewsSentiment
  llm/filings.py       analyze_filings(...) -> FilingIntelligence
  llm/critic.py        critique(...) -> CriticReview
  llm/__init__.py      run_llm_analysis(...) -> LLMIntelligence; llm_pillar(...)
  fusion/signal.py     fuse(...) -> ExecutiveSignal
  fusion/risk.py       build_execution_plan(...) -> ExecutionPlan; size_portfolio(...)
  report/charts.py     price_chart_svg(...)
  report/render.py     render_html(PortfolioReport, settings) -> str; render_pdf(html, path) -> int pages
  report/worker.py     fit_to_pages_isolated(...) -> FitResult: fit_to_pages in a watchdog subprocess
  report/templates/    report.html.j2, report.css
  pipeline.py          run_report(...) orchestration (async)
  evaluation.py        run_evaluation(...) strictly out-of-sample evaluation of every signal (quantcio evaluate)
  cli.py               typer app: report / analyze / rsmt / evaluate / doctor
```

## Public APIs and algorithms

### quant/indicators.py
`compute_technicals(prices: DataFrame, benchmark: DataFrame, atr_period: int = 14) -> Technicals`
- ATR is Wilder's ATR (RMA of the true range). RSI14 is Wilder's RSI. MA50 and MA200 are simple averages; set them to None when
  history is too short.
- `realized_vol_60d` = std of daily log returns over 60 days × √252. `beta_1y` = cov/var of daily returns over 252 days, aligned
  with the benchmark on common dates. It is None if the benchmark is empty.
- `max_drawdown_1y` is ≤ 0. `ret_12m_ex1m` = close[t-21]/close[t-252] - 1. `rel_strength_3m` = ret_3m − benchmark ret_3m.
- `support` = min(low, 20 days). `resistance` = max(high, 60 days).

### quant/factors.py
`compute_factors(md: MarketData, tech: Technicals, settings: Settings) -> list[FactorScore]`. It always returns all 5 names in
the order momentum, value, volatility, quality, trend. Labels: 모멘텀, 밸류에이션, 변동성(안정성), 퀄리티, 추세.
- Each sub-metric maps to 0..100 with a documented monotone function: logistic `100/(1+exp(-(x-c)/s))`, or a percentile against
  the stock's own history when history exists. The factor score is the mean of the available sub-scores. It is `None` when no
  sub-score is available.
- **momentum**: ret_12m_ex1m (c=0, s=0.20), ret_3m (s=0.10), rel_strength_3m (s=0.07), and proximity to the 52-week high
  (price/high−1: 0 → ~80, −0.30 → ~20).
- **value**: lower is better for each of the following. (1) PER percentile vs own `valuation_history` (≥ 1 year of data).
  (2) PBR percentile vs own history. (3) PER / sector_per (1.0 → 50, 0.6 → ~80, 1.5 → ~25). (4) Analyst upside
  target/price−1 (s=0.15, higher is better). A PER that is ≤ 0 or None means loss-making: skip the PER metrics and note it in
  the rationale.
- **volatility** (high score = stable): realized_vol_60d absolute (0.15 → ~85, 0.60 → ~15). The percentile of current 60-day vol
  vs its own rolling 3-year distribution (inverse). beta_1y (|β−0.8| penalty). max_drawdown_1y (−0.10 → ~80, −0.50 → ~15).
- **quality**: roe (c=0.10, s=0.06), operating_margin (c=0.10, s=0.07), earnings_growth_yoy (c=0, s=0.20),
  revenue_growth_yoy (c=0.05, s=0.10).
- **trend**: close vs MA50 and MA200, whether MA50 > MA200, MA200 slope over 20 days, and RSI14. RSI 50–65 is best; RSI > 75
  counts as overbought and is penalised; RSI < 30 counts as oversold and is mildly penalised.
- `rationale`: one deterministic Korean sentence built from the strongest and weakest sub-metric,
  e.g. "12-1M 수익률 +34%로 강세, 52주 고점 대비 -3%".

`factor_composite(factors, weights) -> Optional[float]`: a weighted mean over the non-None factors, with weights renormalised.
`factor_pillar = (composite − 50) / 50`, clipped to [-1, 1]. It is 0 when the composite is None.

### quant/volatility.py
`garch_vol_forecast(close: Series, horizon_days: int) -> float`: fit GARCH(1,1) with Student-t errors on 100 × daily log
returns, using the last ≤ 1500 observations. Take the mean forecast variance over the horizon, then annualise it. If the fit
fails or returns a non-finite or absurd value (outside 0.03–2.5), fall back to EWMA(λ = 0.94).

Position sizing uses it: `quant.sizing_vol_forecast(md, settings)` calls `garch_vol_forecast_ex(close, horizon_days)` and
stores the result in `QuantSignal.vol_forecast_annual`. The EWMA fallback adds its note to `md.warnings`; no estimate at all
gives `None` plus a warning, and sizing then uses `realized_vol_60d`. Until 2026-10-02 the same call ran inside the
per-ticker ensemble (`MLForecast.vol_forecast_annual`), so the numbers are unchanged.

### quant/ml.py (removed 2026-10-02)
2026-10-02: the per-ticker HGB+Ridge walk-forward ensemble was deleted — it passed its skill gate for 0/15 names in every
evaluation, so its 20% share of the serving blend never applied. `MLForecast`, `QuantSignal.ml`, `ml_pillar`, the `ml:`
config block and `rsmt.serving_blend` went with it (old YAML keys are ignored). Its specification is in the git history.

### RSMT-Q — Return-Score Model Tree (quant/panel.py, quant/rsmt.py, quant/distribution.py)

This adapts the user's RSMT sports model to quant. It is a Zeileis–Hothorn–Hornik MOB tree whose **leaves hold the parameters
of a forward-return distribution**. It never classifies up/down directly. Every downstream quantity is derived from ONE
distribution, so none of them can contradict another.

| RSMT (sports) | RSMT-Q (this system) |
|---|---|
| leaf = Poisson λ (home, away) | leaf = (μ, σ²) of the h-day **beta-adjusted excess log return** e = r − β·m, plus the tail ν |
| win/draw/loss sum exactly to 1; O/U, BTTS and exact score come from one matrix | take-profit-first / time-exit / stop-loss-first sum exactly to 1; P(up), P(beat benchmark), quantiles, VaR/CVaR and Kelly all come from ONE simulated path matrix |
| sup-LM fluctuation test per variable, then split scan only inside the winner; Bonferroni | identical; partition set v1 = 9 regime variables (`rsmt.partition_features`) |
| λ offset γ (bypasses Bonferroni) | μ-offset γ·z (z = cross-sectional 12-1 momentum z) and log σ-offset η·v (v = log EWMA vol − c); neither is a split variable |
| overdispersion cap (tune on O/U, not on the 1X2 log loss) | tail floor ν_min: tune on CRPS / 80% coverage, never on direction AUC |
| home_edge δ: moves only the win/loss split, not the expected total | direction edge δ: a logit shift applied only to prob_up / prob_outperform; the mean and quantiles don't include it |
| Dixon–Coles ρ | ρ = residual dependence between e and the market m, used in the joint simulation |
| 3 seasons of baseball data is the bottleneck | pooled panel: the point-in-time market universe (60 names, delisted included) over the longest history (KR 1995-, US 2009-), a row every session (overlapping labels; split test on 60-session block clusters); the market model is fitted on the universe only and serves the requested tickers on a fresh panel (docs/RSMT.md section 8) |
| serving RSMT 80% + rules 20% | serving mean = 100% RSMT-Q (or the selected stock-selection model's excess mean, `ExcessOverride`), then derive everything from that one distribution (the 20% ensemble blend was removed 2026-10-02) |
| a silent offset failure produced collapsed predictions | never fall back silently: `degraded` + reasons, a collapse check (unique (μ, σ) ratio), and a warning on the report with a confidence penalty |

#### quant/panel.py — `build_panel(panel: PricePanel, settings, as_of=None) -> PanelData`
- For each ticker, take `regime_features(prices, bench)`, where the benchmark is `panel.benchmarks[panel.ticker_benchmark[t]]`.
  - Label `r = forward_log_return(close, h)`.
  - Market label `m = forward_log_return(bench close aligned to the stock calendar, h)`.
  - `beta = shrink_beta(beta_252, rsmt.beta_shrink)` as known at t.
  - `y_e = r − beta·m`.
- **Sample grid**: per market, a common calendar (the union of stock dates). Take every `sample_step`-th date counting
  **backwards from the latest date**, so the most recent date is always on the grid. Training rows have a known label
  (t + h ≤ last date). Prediction rows are the latest date for each ticker.
- `xs_mom_z`: on each grid date, the z-score of `mom_12_1` across the tickers present (at least 10 of them, otherwise NaN),
  winsorised at ±3.
- `v_sig = log(ewma_vol) − c`, where c = the weighted mean of log(ewma_vol) over the training rows. Store c so prediction uses
  the same value.
- Weights: `w = 0.5 ** (age_days / half_life_days)`, with age measured from the latest grid date in calendar days.
- Drop training rows that have NaN in any partition feature, offset, beta, y_e or m. **Count the dropped rows and report them.**
  For prediction rows, a NaN in an offset input does NOT drop the row. The row is kept, marked `degraded` with the reason
  "오프셋 입력 누락(z|v)", and the offset input is set explicitly to 0.
- `PanelData` fields: `train: DataFrame` (ticker, date, features…, z, v, beta, y_e, m, r, w), `pred: DataFrame` (the same
  columns without the labels), `c_v`, `market_mu_hist`, `market_nu`, `dropped: dict[str, int]`, and `benchmark_ewma` (the
  benchmark EWMA vol per ticker on the last date).
- `market_mu_hist` is the SERVED market drift μ_m = (1−s)·(time-decayed mean of the benchmark h-returns) +
  s·log(1 + `rsmt.market_mu_prior_annual`)·h/252, with s = `rsmt.market_mu_shrink` (defaults 6%/yr and 0.8), per benchmark
  in `market_mu_by_bench`. Audit fields: `market_mu_raw(_by_bench)`, `market_mu_prior`, `market_mu_shrink`.
  `market_moments(...)` requires the keywords `mu_prior` and `shrink`; the same rule (`moments_from_settings`) is used in every
  CV fold. *(Superseded: "time-decayed mean shrunk 50% toward 0" — that extrapolated recent rallies.)*

#### quant/rsmt.py — the MOB tree
- **Node model.** Gaussian QMLE on the offset-adjusted target. Define ỹ = y_e − γz and s = exp(η·v). Then
  μ = Σ(w·ỹ/s²)/Σ(w/s²) and σ² = Σw(ỹ−μ)²/s² / Σw.
- **Scores (N×2).** `S_i = w_i·[(ỹ_i−μ)/s_i², (ỹ_i−μ)²/s_i² − σ²]`. They sum to 0 exactly at the node MLE. Assert this in tests.
  The meat matrix is `B = SᵀS/N`. Standardise with B^{-1/2} (the symmetric eigen root), NOT the sandwich A⁻¹BA⁻¹. This is the
  same deliberate choice RSMT makes.
- **Step 1 — variable test (no split point is tried).** For each partition variable, stable-sort the rows by x. Then:
  - `W_k = N^{-1/2}·Σ_{i≤k} B^{-1/2}S_(i)` and `LM_k = ‖W_k‖²/(t_k(1−t_k))`, where t_k = k/N.
  - Evaluate only at **tie-group boundaries** inside [trim·N, (1−trim)·N].
  - `supLM = max LM_k`. The p-value comes from the cached null: (1 + #{null ≥ stat})/(n_sims + 1).
  - Adjust with Bonferroni: `p_adj = min(1, m·p)`, where m is the number of partition variables.
  - If min p_adj ≥ alpha, the node becomes a leaf.
- **Step 2 — split scan only in the selected variable.** `Q(k) = N/(N_L·N_R)·S_Lᵀ B⁻¹ S_L`, using cumulative sums in
  O(N log N) with no MLE refit. S_R = −S_L because ΣS = 0. Only tie boundaries with N_L, N_R ≥ min_leaf are admissible. The
  split value is the midpoint between the two adjacent distinct values. Recursion stops at max_depth or when n < 2·min_leaf.
- **Null distribution.** `sup_lm_null(k=2, trim, n_sims, n_grid=1000, seed=0)`: a Monte Carlo sample of
  sup_{t∈[trim,1−trim]} ‖BB(t)‖²/(t(1−t)), where BB is a k-dimensional Brownian bridge. Generate it in batches. Cache it as a
  .npy under `settings.cache_path/rsmt_null/` with **k, trim, n_grid, n_sims and seed all in the file name**, because the null
  depends on trim.
- **Leaves.**
  - μ is shrunk toward the parent: `(n·μ + k·μ_parent)/(n + k)`, with k = mu_shrink_k.
  - σ² is left as is.
  - ν comes from the weighted excess kurtosis κ of the standardised residuals: ν = 4 + 6/κ if κ > 0, else ν_max. Clip it to
    [ν_min, ν_max] for the market.
  - ρ is the weighted correlation between the standardised residual and the standardised market label, shrunk as
    ρ·n/(n + k_shrink).
- **Offsets (global γ, η; like RSMT's single γ).** Alternate until convergence (≤ 50 iterations, tol 1e-8):
  - leaf (μ, σ²) in closed form;
  - γ by weighted least squares: `γ = Σ(w/(σ_L² s²))·z·(y−μ_L) / Σ(w/(σ_L² s²))·z²`;
  - η by a Newton step on `ℓ(η) = Σw[−ηv − r²/(2σ_L²e^{2ηv})]`, bounded to [−0.5, 2.0].

  The outer procedure has 2 rounds: grow the tree with the offsets fixed (initially γ=0, η=1), then refit the offsets with the
  leaves fixed, and repeat. `fit()` **raises** `RSMTOffsetError` when z or v has no finite values. It never drops the offsets
  silently.
- **`describe() -> list[str]`** prints the tree in the user's RSMT style, e.g.
  `vol_ratio_20_120 ≤ 1.12 (p=3.1e-04, n=5210)` / `├─ 리프#1 n=812 μe +0.6% σe 7.1% ν 6.2`.
- **Expanding-window CV (`cross_validate`).**
  - Split the grid dates chronologically into `cv_folds + 1` blocks. Fold f trains on blocks ≤ f, keeping only rows whose
    t + h < the test block start (purge), and tests on block f + 1.
  - For each test row, build a `ReturnDistribution` with `n_paths = 1000`, using the same class as serving.
  - Metrics:
    - AUC of P(up) against 1{r > 0};
    - log loss and Brier of P(up);
    - CRPS (sample-based on the terminal draws);
    - `cover80` = the share of r inside [q10, q90];
    - IC = Spearman(μ, r).
  - Baseline twins `base_*` come from a root-only model with no offsets (the unconditional-distribution analogue of comparing
    to market odds).
  - The direction edge δ = argmin log loss of sigmoid(logit(p) + δ) over the pooled OOS rows, clipped to ±0.5. Do the same for
    δ_rel on 1{r > m}.
  - `CVResult.oos` holds every pooled OOS row (`rsmt.OOS_COLUMNS`): date, ticker, fold, r, m, the model's and the baseline
    twin's P(up) / P(outperform) / μ, and the parameters of the row's simulated distribution (μ_e after the plausibility clamp,
    μ_leaf, σ_e, ν, ρ, β, μ_m, σ_m, ν_m, leaf id; `base_*` twins), plus crps, in80 and exc95 (r below the 5% log quantile).
    `CVResult.folds[i]["leaf_paths"]` names each fold's leaves. `cross_validate_layouts(panel, settings, layouts)` returns the
    full single-layout CVResult of each fold layout (no robust verdict; combine with `layout_gate` + `apply_robust_gate`).
  - **Skill gate (fold-robust).** Single-layout gate = `auc − base_auc ≥ rsmt.skill_min_auc_gain` AND `logloss <
    base_logloss` AND (`skill_require_date_ci` ⇒ date-bootstrap `auc_gain_p05 > 0`). `skill_ok` = the primary CV is available
    AND the gate passes for a strict majority of the layouts `rsmt.robust_cv_folds` (default [3, 4, 5]; each a full
    expanding-window CV; the `cv_folds` run is reused; layouts run in 3 threads). Metrics, δ and the OOS frame are those of
    `cv_folds`. `cv_metrics` adds `gate_folds_<k>` (1/0), `gate_folds_n/pass/need` (need = n//2+1), `auc_gain_folds_<k>`,
    `auc_gain_p05_folds_<k>`, `skill_ok_primary`, `skill_ok_robust` and `n_mu_clamped`. `CVResult` adds `skill_ok_primary`,
    `layouts`, `n_folds_primary` and `seconds`; `cross_validate(..., robust_folds=None)`.
    *(Superseded: `skill_ok = (auc − base_auc ≥ 0.01) and (logloss < base_logloss)` on one layout — a single-layout pass is
    noise: live, 1 of 12 layout gates passed across four evaluations.)*
    **Since 2026-09-29 this is the DIRECTION gate; the served `skill_ok` follows `rsmt.skill_metric` (default
    `distribution`: date-bootstrap 5% bound of the CRPS gain over the twin > 0 AND cover80 ∈ [0.75, 0.85] AND exc95 ∈
    [0.02, 0.08], same fold-layout majority).** Both gates are always computed (`CVResult.direction_ok` /
    `distribution_ok` / `dist_gate`; cv_metrics `direction_gate`, `distribution_gate`, `*_gate_folds_<k>`,
    `crps_gain(_p05)`, `exc95`); `gate_folds_<k>` follow `skill_metric`. See "RSMT-Q v1.1" below and docs/RSMT.md.
- **`fit_market_model(panel_data, settings) -> MarketModel`** runs CV (metrics and δ) and then the final fit on all training
  rows. `MarketModel.predict(ticker, *, hold_days=None, override=None) -> (ReturnDistribution, DistributionForecast)` (and
  `predict_many(tickers, *, hold_days=None, overrides=None)` for a market batch) does the following:
  - Take the leaf params, add the offsets, and use β, the market params (mu_m = market_mu_hist, σ_m = benchmark EWMA vol ×
    √(h/252), ν_m) and ρ.
  - **Served mean**: RSMT-Q's own total mean (`blend = {"rsmt": 1.0}`), or, with an `ExcessOverride`, β·μ_m + the selected
    model's μ_excess (`blend = {model_id: 1.0}`), applied by shifting μ_e. (Until 2026-10-02 a skilled per-ticker ensemble
    took 20% here; it never passed its gate.)
  - **Plausibility clamp.** The served total mean is clamped to |μ| ≤ `MU_PLAUSIBLE_K`·σ_h (`MU_PLAUSIBLE_K = 0.5`, a
    module constant in rsmt.py — a candidate for `rsmt.mu_plausible_k`; σ_h = `distribution.total_log_sd`, the exact sd of the
    simulated h-day log return). `offsets` gains `mu_total_unclamped`, `mu_cap` and `mu_clamp_shift`. A binding clamp is an
    `[정보]` health line, not degraded. The CV applies the same clamp (`n_mu_clamped`).
  - Apply δ.
  - `MarketModel.health() -> list[str]`. Health checks:
    - offsets applied on every served row;
    - collapse: unique (round(μ, 5), round(σ, 5)) / n_served < collapse_min_unique_ratio when n_served ≥ 4, in which case
      every row is flagged degraded;
    - more than 30% of the panel failed to load, which also means degraded;
    - a tree with 0 splits is reported as info, not degraded ("분할 검정 미통과 — 표본 확대 필요").
  - `MarketModel.diagnostics() -> list[str]`: the tree print, n, leaves, root p, γ/η/δ, CV metrics vs the baseline, dropped
    rows and failed tickers, plus the robust gate line ("폴드 강건성 게이트 …: 3폴드 … · 4폴드 … · 5폴드 … → k/n"),
    "시장 드리프트 μ_m(20일) = 0.20×과거 + 0.80×사전 …" and "μ 타당성 상한 …".

#### RSMT-Q v1.1 — risk engine, model switches (docs/RSMT.md)
- **Role separation.** RSMT-Q serves the risk distribution; with a stock-selection serving artifact the excess mean is
  100% the selected model's (`ExcessOverride`: `blend = {model: 1.0}`, `relative_edge = 0`); RSMT-Q's own leaf mean
  (`blend = {"rsmt": 1.0}`) is the fallback without an artifact. `DistributionForecast.skill_ok` = RSMT-Q's
  `skill_metric` verdict (also under an override); `DistributionForecast.direction_skill_ok` = the override's skill or
  RSMT-Q's direction gate — the only verdict the directional ML-pillar multiplier reads.
- **Switches** (`rsmt.<key>`, per market `rsmt.markets.<M>.<key>`, `quant.panel.rsmt_option`; originals in brackets):
  `overlap_step` [0] — panel rows every k sessions, `block` = (calendar position − latest)//h clusters for the split
  test (Bartlett / Driscoll–Kraay HAC equivalent), `on_grid` rows for CV folds, CV test rows and market moments, min_leaf
  and leaf shrinkage in non-overlapping units; `multiple_testing` [bonferroni] | simes | pca (`pca_components` [3],
  loadings fitted on the training rows, stored in `RSMTree.pca`, `split_features` = PC1..PCK); `target_scale` [none] |
  ewma (split on e / `scale_h`, `scale_h` = EWMA vol × √(h/252); residual η, `eta_equivalent` = 1 + η); `gamma_mode`
  [global] | leaf (`TreeNode.gamma`, shrunk toward the parent with `gamma_shrink_k` [50] blocks of prior weight).
- **Comparison.** `scripts/rsmt_compare.py --market KR|US` runs the pre-declared variant comparison (docs/RSMT.md §1:
  CV on pre-holdout rows, 24-month final holdout, `quant.rsmt.evaluate_split`) → `reports/rsmt/compare_<M>.md|json`.
- **`RSMTRanker` v3** (`rsmt_v3`): the same switches on the stacked stock-selection ranker (Simes, target y_dm /
  (vol60·√(h/252)) ranked by the standardised mean, leaf γ), stacked / checkpointed like v2 (`candidates.STACKED`).

#### quant/distribution.py — `ReturnDistribution` (one matrix → every derived quantity)
- **Constructor:** `ReturnDistribution(mu_e, sigma_e, mu_m, sigma_m, nu, nu_m, beta, rho, horizon, n_paths, seed,
  direction_edge=0, relative_edge=0)`. All μ and σ values are over the horizon, in log-return terms.
- **Daily simulation.** Draw h steps.
  - Daily mean = μ/h and daily sd = σ/√h.
  - Innovations are Student-t with `ν_d = max(4.05, 4 + (ν − 4)/h)`, so the excess kurtosis of the h-day sum equals that of a
    t_ν. They are scaled to unit variance by √((ν_d−2)/ν_d).
  - Correlation ρ is applied to the underlying normals before the χ² scale mixing (separate χ² draws per component).
  - The stock path is `r_t = β·m_t + e_t` (cumulative). The seed is deterministic, from sha1(ticker|as_of).
- **Derived quantities.** Everything comes from the path matrix, and everything except the two edge-adjusted probabilities
  excludes δ:
  - `prob_up_raw` = mean(r_T > 0); `prob_up` = sigmoid(logit(raw) + δ);
  - `prob_outperform` = mean(r_T > m_T) with δ_rel;
  - `expected_return` = mean(exp(r_T) − 1);
  - `quantile(q)` of exp(r_T) − 1; `var95` = q05; `cvar95` = the mean below q05.
- **`barrier(stop_pct < 0, take_pct > 0) -> BarrierOutcome`.** Check first touch on daily closes (consistent with the rule
  "종가 기준 손절").
  - The exit return is the close on the breach day. This keeps gap risk, so a stop-loss exit can be worse than stop_pct.
  - Paths that touch neither barrier exit at the time limit with exp(r_T) − 1.
  - p_tp + p_sl + p_time == 1 exactly (they are counts / n_paths).
  - `kelly_optimal` = argmax_f mean(log(1 + f·R)) on a grid f ∈ [0, 1], step 0.005.
- **Other methods.** `shift_mu(delta)` returns a new instance with the same seed. `summary(...) -> DistributionForecast`.

### modeling/ — stock-selection model (docs/MODELING.md)

The old headline AUC scored P(20-day TOTAL return > 0), which is dominated by the common market move of a date.
The problem is now split: (1) STOCK SELECTION on the beta-adjusted excess return y_e = r − β·m, evaluated WITHIN
dates; (2) a separate MARKET-DIRECTION component (base rate unless a small macro/stress model passes its rule).

- `dataset.build_dataset(market, settings, cache) -> ModelingDataset`: PIT universe (`data.pit`, the backtest
  universe: KR top-N from `backtest.kr_start`, US S&P 500 from `backtest.us_start`), weekly + month-end grid,
  features (`modeling/features.py`, every value at t uses data ≤ t / `available_from ≤ t`), labels with delisting
  terminal prices, per-date targets, event flags. Cached (`modeling_dataset`, AST fingerprint of the code).
  KR labels are TOTAL returns since dataset v4 (2026-10-12, docs/MODELING.md 52.1): `r` adds log(1 + D / C_prev)
  for every ex-dividend session in (t, t_end] (`data/pit/kr_dividends.py`, Yahoo); the price return stays in
  `r_price`, the added part in `div_log`; codes Yahoo lacks keep price returns (counted in `coverage`).
- `events.py`: named event calendar + detected index shocks — used ONLY for robustness tables and the candidate
  training-weight schemes (`none` / `event_down` / `winsor`, chosen by inner CV). Never a model input.
- `candidates.py`: baselines (12-1 momentum, the backtest `ic_weighted` model), Fama–MacBeth (signs learned,
  shrunk to 0), ridge, L2 logistic, HGB, LightGBM regressor / classifier / weekly regressor / LambdaRank,
  RSMT-Q v2 (`quant.rsmt.RSMTRanker`), rank-average ensemble.
- `selection.run_selection(ds, settings, cache) -> SelectionResult`: nested purged expanding walk-forward
  (12-month outer blocks after 36 months, 2 inner blocks, purge + h-session embargo), untouched 24-month holdout,
  the pre-declared rule `select_best` (mean within-date AUC of 1{y_e>0}; ties < 0.0005 → rank IC → top-quintile
  net) and `serving_gate` (beats 12-1 momentum AND AUC 95% bootstrap lower bound > 0.5). Contexts are checkpointed
  (`modeling_ctx`, `modeling_rsmt`).
- `market.market_direction(...)`: walk-forward logistic vs base rate on month-end dates; used only when the
  pre-holdout log-loss gain lower bound > 0 AND the holdout gain > 0.
- `serve.py`: `build_artifact` (final context fitted on all labelled rows, calibrated
  `mu_excess = shrink·slope·z`, ×0.35 when the gate fails) cached under `modeling_serving`;
  `score_for_report` → `models.ExcessForecast` per ticker (outside-universe names placed into the latest
  cross-section), `excess_override` → `quant.rsmt.ExcessOverride`.
- `cli.py`: `modeling_app` (`select --market`, `show --market`), registered by the main CLI.

Serving (pipeline.build_report): `MarketModel.predict_many(..., overrides={ticker: ExcessOverride})` keeps every
RSMT-Q v1 risk parameter (σ_e with η, ν, ρ, β, the market leg's σ / ν) but uses the selected model's excess mean
(not the RSMT-Q leaf mean, `blend = {model_id: 1.0}`, `relative_edge = 0`, `skill_ok` = the modeling gate, extra
`cv_metrics` keys `excess_*`). The market leg keeps the prior-shrunk drift unless the market model is used (then
μ_m = σ_m·Φ⁻¹(p)). `QuantSignal.excess_model` carries the ExcessForecast (P(beat benchmark) from the same path
matrix); `QuantSignal.factor_model` / `PortfolioReport.factor_backtest` come from the cached factor backtest. No
artifact → the RSMT-Q leaf mean with a note.

RSMT-Q training panel (`pipeline.training_panel`): with `pit.universe_mode: pit` the PIT top-`universe_size`
universe (`PITUniverse.to_price_panel`, requested names added from the production providers); `quant/panel.py`
honours `PricePanel.membership` (training rows = members at each date, cross-sectional z standardised against
that date's members, prediction rows = latest members + requested + former members still trading; delisted
entities are not served). A PIT failure falls back to the current list with a loud note.

### quant/__init__.py
`build_quant_signal(md: MarketData, settings: Settings, market_model: Optional[MarketModel]) -> QuantSignal` wires
indicators → factors → sizing vol (`sizing_vol_forecast`, see quant/volatility.py) → RSMT-Q prediction.
- `ml_pillar` uses the distribution when it is available: `z = μ/σ`, `raw = 0.5·tanh(2z) + 0.5·(2·prob_up − 1)`. The
  multiplier is 1.0 if `direction_skill_ok` (the fold-robust DIRECTION verdict above, or the stock-selection model's served
  skill under an ExcessOverride — never the distribution verdict `skill_ok`), else 0.35, and a further ×0.5 if
  `degraded`. Without a distribution `ml_pillar` is 0 and fusion drops the ML pillar as unavailable (weights renormalised,
  "ML/RSMT-Q 예측 없음" note); a failed RSMT-Q prediction adds "RSMT-Q 예측 실패(…) — ML 필러 제외" to `md.warnings`.
- A degraded distribution adds a Korean warning to `md.warnings`. It is never silent.

### llm/ (Claude API: anthropic SDK 1.8, model from settings.llm.model = "claude-opus-5")
- **`LLMClient`**: async wrapper around `AsyncAnthropic`.
  - It uses `client.messages.parse(..., output_format=<PydanticModel>)` for structured output, with adaptive thinking and
    `output_config={"effort": ...}`.
  - It caches the system block with `cache_control: {type: "ephemeral"}`.
  - When `server_fallbacks` is on, it sends the header `anthropic-beta: server-side-fallback-2026-07-01` and the body field
    `"fallbacks": "default"` via `extra_headers`/`extra_body`.
  - It checks `stop_reason` before reading content. It handles `refusal` and `max_tokens` (returns None plus an error string).
  - It catches typed errors in order: RateLimitError, APIStatusError, APIConnectionError.
  - It enforces a concurrency semaphore and accumulates cost from `usage`. Pricing for claude-opus-5 is $5/M input, $25/M output,
    cache write ×1.25 and cache read ×0.10.
  - It keeps a disk response cache keyed by the hash of (model, system, user, schema name, effort).
  - `FakeLLMClient` has the same interface and returns canned pydantic objects, for tests.
- **`analyze_news(client, inst, news, settings) -> Optional[NewsSentiment]`**:
  - One call per ticker. Its input is the list of articles (id, date, source, title, summary), each in its own `<document id=..>`.
  - The LLM returns, per article: sentiment, relevance, event_type, and an evidence quote. It also returns key_positives,
    key_negatives and key_events in Korean.
  - Code then sets `grounded` by searching for the normalised evidence quote in the normalised title+summary (whitespace,
    quotes and case folded; also accept a quote whose 80% character 3-gram overlap is present).
  - The score is `Σ w·s / Σ w` over articles that are grounded and have relevance ≥ min_relevance. The weight is
    `w = relevance·0.5^(age_days/half_life)`.
  - `trend_7d` = the mean of the last 7 days minus the mean of the earlier days (None if either side is empty).
  - Returns None when there is no news.
- **`analyze_filings(client, inst, filings, fundamentals, settings) -> Optional[FilingIntelligence]`**:
  - One call per ticker. Earnings releases come first; each filing is truncated so the head and tail are kept.
  - Output is in Korean. `earnings_highlights` must quote figures exactly as they appear in the text.
  - Code check: extract numeric tokens from each highlight. Drop the highlight if any token (after normalising commas, %, 조/억,
    and $) is not found in the concatenated filing text.
  - Every `risk_factor` needs a `source_ref`.
- **`critique(client, inst, quant, news, filings, fundamentals, prelim: ExecutiveSignal) -> Optional[CriticReview]`**:
  - This is the adversarial reviewer. It sees the compact quant JSON, the LLM outputs and the preliminary signal.
  - It writes a 1–2 sentence Korean thesis, bull and bear points, and contradictions, plus a bounded confidence_adjustment.
  - It sets `critical_flag` only for a verifiable critical event in the provided documents: fraud or embezzlement (횡령/배임),
    audit opinion refusal (감사의견 거절), delisting or trading halt, going concern, default, or a large dilutive issuance > 20%.
  - The prompt instructs it to default toward no adjustment when evidence is thin.
- **`llm_pillar(news, filings) -> float`**:
  - Start with `0.65·news.score·min(1, n_used/5) + 0.35·filings.tone`, using whichever parts are available.
  - Then apply the risk penalty: −0.05 per `high` risk factor and −0.20 per `critical`, with the total penalty capped at −0.4.
  - Clip to [-1, 1].
- **`run_llm_analysis(client, md, quant, settings, prelim_fn) -> LLMIntelligence`**: runs news and filings concurrently. It then
  computes the pillar and calls `prelim_fn(llm_partial)` to get the preliminary ExecutiveSignal, runs the critic, and accumulates
  the cost and errors.

### fusion/signal.py
`fuse(md, quant, llm: Optional[LLMIntelligence], settings) -> ExecutiveSignal`

**Composite.** `composite = Σ w_p·pillar_p` over the available pillars, with weights renormalised. The LLM pillar is unavailable
when `llm.skipped` is true or when both news and filings are None.

**RSMT-Q adjustments.** When `quant.distribution` is present (and no stock-selection model is served), the ML-skill component is
`[20·clip((auc−base_auc)/0.05, 0, 1)·0.5 + 20·clip((base_logloss−logloss)/0.01, 0, 1)·0.5] × g`, where the gains are the
primary layout's and g = 1 if `skill_ok`, else `gate_folds_pass/gate_folds_n` (0 without a gate record) — a model the
robust gate rejects no longer earns full confidence points (live KR 2026-09-25: 7.0 → 2.3 points). `distribution.degraded` adds
−10 to the confidence and an override note "RSMT 저하 모드: <reasons>". The ML target-price component uses the distribution:
`price·exp(μ·H/h)`.

**Barrier guardrail** (`apply_barrier_guardrail(execsig, plan) -> ExecutiveSignal`, called after the execution plan is built):
if the signal is Buy or better and `p_stop_loss > p_take_profit + fusion.barrier_guard_margin` (0.15), downgrade it by one
notch and note the barrier probabilities. (Evaluation D, 2026-09-25: the barrier probabilities are right on average but have no
per-name ranking power, AUC 0.49–0.52 — see docs/EVALUATION.md.)

**Confidence (0–100).** Sum these components, then apply the critic adjustment and clip to [5, 95]:
- **Agreement** (0–35): 35 × (the weighted share of available pillars whose sign matches the composite's sign, where a pillar
  with |p| < 0.05 counts as half) × (1 − min(1, std(pillars))).
- **Strength** (0–20): 20·min(1, |composite|/0.6).
- **ML skill** (0–20): as in "RSMT-Q adjustments" above, or from the stock-selection model's AUC / IC when it is served; 0
  without either. (The per-ticker ensemble's hit-rate / IC formula was removed with it on 2026-10-02.)
- **Data quality** (0–15): 15 minus 3 per warning (minimum 0) × fundamentals completeness (the share of per/pbr/roe/analyst
  target that are present, with a floor of 0.4).
- **Evidence** (0–10): `10·min(1, n_used/8)` when news is present. Add 0 when filings are missing.
- Then add `critic.confidence_adjustment`.

**Signal.** Map with the thresholds: composite ≥ strong_buy → Strong Buy (requires confidence ≥ strong_buy_min_confidence,
otherwise Buy); ≥ buy → Buy; ≤ sell → Sell; anything else → Hold.

**Guardrails.** These are appended to `overrides` in Korean:
1. If `critic.critical_flag` is set, or any `critical` risk factor exists, cap the signal at Hold. If the composite is < 0, the
   signal becomes Sell.
2. **Falling knife**: if ret_1m < −0.20, the trend score is < 30 and the signal is Buy or better, cap at Hold unless
   llm_pillar > 0.4.
3. If the LLM is skipped and the ML is not `skill_ok`, cap Strong Buy at Buy.
4. **Target gap (symmetric)**, evaluated after the target price:
   - Buy/Strong Buy with target upside < `fusion.min_buy_upside` (0.02) becomes Hold. Note: "목표가 괴리 가드레일: 목표가
     상승여력 +1.0% < 매수 최소 상승여력 +2.0% (종합점수 +0.50와 불일치) → Buy→Hold".
   - Sell with target upside > `fusion.max_sell_upside` (0.05) becomes Hold. Note: "목표가 괴리 가드레일: 매도 신호이나 목표가
     상승여력 +8.0% > 매도 허용 상한 +5.0% (종합점수 −0.40와 상충 — 근거 불일치) → Sell→Hold".
   - Boundaries: upside == min_buy_upside keeps the Buy; upside == max_sell_upside keeps the Sell.
   - A critical-event Sell (guardrail 1) is never lifted. This is the only guardrail that can raise a call, and only Sell→Hold.
   - Non-finite thresholds raise ValueError.

**Target price.** Use H = settings.fusion.target_horizon_days. Each component is scaled to H:
- **ml**: `price·exp(μ·H/h)` with the RSMT-Q served log mean; absent without a distribution
- **analyst**: `price·(1 + (analyst_target/price − 1)·H/252)`, if present
- **valuation**: fair PER = the median of own-history PER, or else sector_per. Then `fair = price·fairPER/PER`, and the component
  is `price·(1 + (fair/price−1)·H/252)`. Skip it when PER ≤ 0 or no reference exists.

Blend the components with `target_price_blend`, renormalised over the ones present. If none are present, use
`price·(1+composite·0.1)`.

**Horizon label.** Strong Buy → "1~3개월 (분할매수)". Buy → "1~3개월". Hold → "관망 · 1개월 후 재평가". Sell → "즉시~2주 내 비중 축소".
`thesis` comes from `critic.thesis` when present; otherwise it is a deterministic Korean sentence built from the pillars.

### fusion/risk.py
`build_execution_plan(md, quant, execsig, settings) -> ExecutionPlan`. The weight is filled in later.
- entry = last close. Entry zone = [entry − 0.5·ATR, entry + 0.25·ATR].
- stop_dist = clip(stop_atr_mult·ATR, min_stop_pct·entry, max_stop_pct·entry). If support lies in
  [entry − 1.3·stop_dist, entry − 0.5·stop_dist], use stop = support·0.99 instead. stop_loss_pct = stop/entry − 1.
- take_profit = max(target_price, entry + min_reward_risk·stop_dist). take_profit_2 = max(analyst_target_high or 0,
  entry + 3·stop_dist). reward_risk = (tp − entry)/(entry − stop).
- **Barrier and Kelly**:
  - **With a distribution:** `plan.barrier = dist.barrier(stop_loss_pct, take_profit_pct)` and
    `kelly_fraction = barrier.kelly_optimal·kelly_cap`. This means fractional Kelly on the same path matrix that produced the
    signal. The ReturnDistribution object is passed in; it is not reconstructed.
  - **Otherwise:**
    - `p` = the RSMT-Q prob_up if a distribution exists, otherwise `0.5 + 0.25·composite`, then blended 50/50 with
      `0.5 + 0.5·composite`.
    - `b` = reward_risk. `f = p − (1−p)/b`.
    - kelly_fraction = clip(f, 0, 1)·kelly_cap.
- **raw_weight**: for Buy or Strong Buy, `(max(composite,0)·confidence/100) / max(vol, 0.10)`, where vol is
  `QuantSignal.vol_forecast_annual` (GARCH, quant/volatility.py), otherwise realized_vol_60d. Hold and Sell get 0.
- **action** and **triggers** are deterministic Korean strings, for example:
  - "종가 기준 손절가 이탈 시 전량 청산"
  - "익절가 도달 시 50% 차익실현 후 손절가를 진입가로 상향"
  - "실적발표 D-3 이내 신규진입 자제" (when a known upcoming earnings date exists)
  - a trigger on a change in news sentiment

`size_portfolio(items: list[tuple[MarketData, ExecutionPlan, ExecutiveSignal]], settings, capital: Optional[float],
base_currency: str) -> tuple[list[ExecutionPlan], float cash_weight]`:
(`conviction_weights(raw, caps, cov_for, risk) -> SizingResult`; covariance = the last 252 aligned daily returns, converted to
the base currency with a constant FX)
- w_i = min(raw_i·k, cap_i), where cap_i = min(max_weight, kelly_i) and k is ONE common scale.
- k is the largest value with raw_i·k ≤ max_weight for all i (the top-conviction name sets the ceiling), Σw ≤ max_gross and
  ex-ante vol ≤ target_portfolio_vol.
- A name at its Kelly cap stays capped and the others keep w ∝ raw. Freed budget is used through the gross/vol limits but never
  past the max_weight ceiling.
- Solved in closed form: the gross limit is a piecewise-linear cut; vol² is a convex quadratic on each piece between the
  breakpoints b_i = cap_i/raw_i, so take the largest feasible root, searching pieces from the top. Vol is not monotone in k for
  hedged books.
- Names with cap < min_weight are excluded. While some w < min_weight: drop the smallest such tier, re-estimate the covariance
  and re-solve.
- cash = 1 − Σw.
- Guarantee: of two otherwise-equal names (same Kelly cap), the higher conviction never gets the lower weight. Weights are
  invariant to a common rescaling of raw.
- `signal_triggers`: a Hold at a Sell-level composite is a lifted Sell; its trigger is the target-cut condition.

*(Superseded: normalise → cap → ×max_gross → vol scale-down → zero below min_weight. Normalising first let a low-conviction
name outweigh a capped high-conviction one and left budget unused.)* `report/sample.py` calls the same `conviction_weights`
(lazy import; if fusion cannot be imported, the sample falls back to the old rule and says so in `notes`).

When `capital` is given: position_value = w·capital, and shares = floor(position_value / (entry·fx_to_base)).

### report/
- `render_html(report: PortfolioReport, settings, compact_level: int = 0) -> str` uses Jinja2 with `report.html.j2` and
  `report.css`. It is A4 portrait with 12mm margins and Korean fonts ('Apple SD Gothic Neo', 'Noto Sans KR', sans-serif).
  `report.css` keeps a 3px inset (`--edge`) inside the @page content box on both sides, because Chrome clips print output at
  that box and right-aligned text there lost its last glyph pixels. @page stays the single source of truth for margins.
  Header meta, sub-heading legends, chip rows and the Module 4 key wrap instead of overflowing.
- **Page 1**: the header (date, universe, config digest, market regime chips), then the **Executive Signal Matrix**, then the
  portfolio summary (invested weight vs cash, LLM cost). The matrix has one row per ticker: name/code, signal badge (color
  coded), a confidence bar and number, current price, target price and upside, horizon, and a one-line thesis.
- **Page 2**:
  - **Quant & ML Factor Signal** table: five factor score mini-bars and the factor composite, then the model behind the ML
    pillar — the stock-selection model (percentile, excess mean, validated AUC, skill) when it is served, otherwise the
    RSMT-Q distribution (E[R], P(up), CV AUC · IC, direction-gate skill; "ML 없음" without a distribution).
  - **LLM Unstructured Intelligence** cards per ticker: sentiment score (−100..+100) with n_used/n_articles and trend, the
    3 bullets from the filing/earnings summary, and the top risks with severity chips.
- **RSMT-Q in Module 2**: one row per ticker with μ as the expected return, P(up), P(beat benchmark), the 10–90% band,
  VaR95, a short leaf path, and skill/degraded badges. Below the rows, a compact **model diagnostics** box per market: the
  first 6 lines of the tree print, n, leaves, root p, γ/η/δ, and CV AUC/logloss/CRPS/cover80 vs the baseline.
  Each market's box starts with a **'모델 신뢰도'** line: the robust CV skill gate from `DistributionForecast.cv_metrics`
  (`gate_folds_<k>` 1/0 per layout, `gate_folds_n`, `gate_folds_pass`, `gate_folds_need` = n//2+1), e.g. "3개 폴드 구성 중
  1개만 통과 → 스킬 미확인 (3폴드 ✗ · 4폴드 ✓ · 5폴드 ✗)". Without those keys it prints the single-layout verdict and
  "폴드 구성별 게이트 기록 없음". A disagreement with `skill_ok` is printed and logged.
- **Page 3**: the **Execution & Risk Management** table (weight, entry zone, stop-loss (−%), take-profit (+%), TP2, R:R, shares),
  plus **P(TP first) / P(time exit) / P(SL first)** as a stacked bar that sums to 100%, the expected trade return and Kelly.
  Then the portfolio risk line (ex-ante vol, VaR95), per-ticker triggers, a methodology footnote (including RSMT-Q; the sizing
  sentence describes the conviction-scale rule of `size_portfolio`), and the disclaimer.
- **Kelly column**: the capped fraction (f*×kelly_cap) is the main value. f* is printed underneath only while it rounds below
  100%; at the grid bound the sub line reads "한도 적용", because `kelly_optimal = 1.0` means the optimum is 100% or more, not a
  bet-it-all recommendation.
- If rendering exceeds `max_pages`, re-render with a higher `compact_level`: 1 = hide charts, 2 = 2 bullets and 2 risks,
  3 = drop the article-level lines.
- `render_pdf(html, out_path) -> int` uses Playwright with channel "chrome", falling back to bundled chromium. It returns the
  page count (via pypdf).
- Production rendering (`run_report`, `doctor`) goes through `report/worker.py::fit_to_pages_isolated`: `fit_to_pages` runs in
  `python -m quantcio.report.worker <payload.pkl> <result.pkl>` (own session). Timeout `RENDER_TIMEOUT_S` = 180 s per attempt;
  on timeout / crash / worker error the whole process tree (worker, Playwright driver, Chrome's detached process group) is
  SIGKILLed and the render retried once; after two failures the HTML is written in-process at the worker's chosen (else the
  maximum) compaction level with a page-1 note, `pages=0` and a warning in `ReportArtifacts.warnings`. Reason: live
  2026-09-25 the Playwright node driver crashed mid-transfer and the in-process sync API hung the report for 20+ minutes.

### pipeline.py / cli.py
- `run_report(tickers, settings, capital=None, base_currency="KRW", use_llm=True, out_dir=None) -> ReportArtifacts(json_path,
  html_path, pdf_path, pages, report)`.
  - Data loading and quant run in a thread pool; the LLM runs as async tasks.
  - The order per ticker is: data → quant → LLM (news+filings) → preliminary fuse → critic → final fuse → execution plan.
  - Then `size_portfolio`, the market regime, and rendering.
- CLI commands: `quantcio report --tickers ... [--capital N] [--base KRW|USD] [--no-llm] [--config file] [--out dir]`,
  `quantcio analyze TICKER [--json]`, `quantcio rsmt --market KR|US [--tune]` (fits the tree, prints it with CV vs baseline; `--tune` grid-searches ν_min by CRPS/cover80 and alpha/min_leaf by logloss — each parameter tuned on the metric it actually moves), `quantcio evaluate --market KR|US` (strictly OOS evaluation, see evaluation.py below), and `quantcio doctor` (checks env keys, data-source reachability, Chrome, and writes
  a sample PDF).

### evaluation.py — `quantcio evaluate --market KR|US [--universe N] [--years Y] [--folds 3,4,5,6] [--out d]`
Strictly out-of-sample evaluation through the production code paths. Methodology, live results (2026-09-25, four runs),
the leakage audit and model-improvement recommendations: **docs/EVALUATION.md**. Writes `evaluation_<M>_<N>u_<Y>y.md` (Korean) and
`.json` into `--out` (default `<report.output_dir>/evaluation`). The price panel (`load_price_panel`) is cached for the day
(DiskCache namespace `eval_panel`, `data.price_ttl_hours`). Live 2026-09-25, KR 60 names × 7 years: about 45 s.
- **A** `rsmt.cross_validate_layouts` for every layout in `--folds` (default `rsmt.robust_cv_folds`; the primary `cv_folds`
  and the production robust layouts always run): AUC, log loss, Brier, CRPS, cover80 and the VaR95 exceedance rate for the model
  and the root-only / no-offset twin. The 90% CIs of the gains come from a date-block bootstrap (`rsmt.boot_sims` draws). The
  percentile distance is widened by c_G = sqrt(G/(G−1))·t_{0.95,G−1}/z_{0.95}. The served verdict is recomputed exactly
  (`layout_gate` + `apply_robust_gate`), next to the majority over `--folds`.
- **B** On each OOS date of the primary layout, the Spearman IC of μ_excess (μ_e as simulated), P(outperform) and the leaf μ vs
  y_e = r − β·m. Baselines: `xs_mom_z`, −ret_1m and a seeded random score. Reported: mean IC and its t across dates (grid step =
  h, so the windows do not overlap), share of IC > 0, the top − bottom quintile spread (top and bottom buckets both
  `quantile_bucket_size(n)` = round(n/5) names, seeded tie-break, NaN for a constant score), top quintile − date mean, a
  500-permutation random null band for the IC t, the spread t and the top − all t (the '기준: 무작위' row is ONE draw from it), and
  the per-date IC difference vs momentum. A row computed on < 80% of the dates (leaf μ) is flagged as a different sample.
- **C** Reliability deciles, ECE and the logistic recalibration slope and intercept of the raw P(up) / P(outperform), plus the
  served values after δ (in-sample, 1 parameter). cover80 and exc95 are shown by test block, by market-vol tercile, by market-trend
  sign and by (fold, leaf). `survivorship_diagnostic`: per-date mean y_e of TODAY's universe over the training dates (annualised,
  t) and its OLS slope on years-before-as_of (a positive slope = today's winners led more the further back: selection on the
  outcome), plus the OOS rows' root-twin drift, model μ_e and realised y_e. Audit simulation 2026-09-25 (null market, no alpha,
  no momentum): choosing today's top 60 of 600 (7y) / 150 of 1500 (12y) by final cap alone gives +13%/+12% a year of excess
  drift, biases the 12-1 momentum IC DOWN (t ≈ −0.6 over 47 dates, −1.0 to −1.3 over 86–138) and leaves the 1-month reversal IC
  unbiased; the report states this direction (it used to say survivorship makes momentum look good).
- **D** entry = close_t. stop distance = clip(stop_atr_mult·ATR_t, min_stop_pct, max_stop_pct), with ATR from bars ≤ t.
  take-profit = min_reward_risk × stop distance. Predicted P(TP first) / P(SL first) / P(time exit) come from the row's OOS
  ReturnDistribution (1000 paths, hold = fusion.target_horizon_days, no edge). Realised = the first touch on the next `hold`
  daily closes (rows without them are excluded and counted). Reported: 3-class Brier (CI from a bootstrap of ⌈hold/step⌉-date
  blocks), AUCs, reliability quintiles and E[trade], each against the root-only twin. Verdict (`barrier_verdict`): "보정 양호"
  only when both mean levels are within 3pp AND both logistic recalibration slopes lie in [0.7, 1.3]; "평균만 보정" when only the
  levels match (live 2026-09-25: levels within 3pp, AUC ≈ 0.49–0.52, slopes −0.34…0.72 — right on average, no per-name
  information, so not a basis for the per-name barrier guard); "보정 미흡" otherwise.
- **E** Momentum, volatility and trend scores at every grid date, from `compute_technicals` + the `quant.factors` sub-score
  functions on the last 830 bars ≤ t (identical to the full history). The price-only composite uses the factor weights. Same
  statistics as B. Value and quality are excluded because their inputs are a fundamentals snapshot.
- **F** Removed 2026-10-02 with the per-ticker ensemble (it was `quant.ml.forecast` on the first 15 names: 0/15 passed in
  every run). The other section letters are kept; `--no-ml` is still accepted and ignored.
- **G** A rule-based Korean verdict per topic: evidence tiers for skill, and "보정 양호/미흡" for calibration. The recommended
  pillar weights never change the config:
  - An untestable pillar (LLM) keeps its weight.
  - A weak or no-evidence pillar is cut by ×0.6 or ×0.3.
  - The freed weight goes only to pillars with evidence (t ≥ 2 or robust gate passed). If there are none, the weights are
    unchanged and the report recommends stricter thresholds.
  - `multiple_testing_summary`: how many ranking t-statistics (B, E) the report prints, how many reach |t| ≥ 2 and the ~5% of
    them expected by chance; only the pre-specified primary metrics drive the tiers.
  - The tree-split sentence (`split_summary_line`) covers EVERY evaluated fold layout: "모든 폴드 배치의 모든 CV 폴드에서 분할
    없음" only when no layout split; otherwise it names the layouts/folds that did.
  - The distribution line states the CRPS gain relative to the base CRPS (0.3–1.3% live) and that the root-only twin has no
    volatility conditioning (audit re-simulation: the whole gain comes from η·v; the μ/momentum offset adds +0.05% KR, −0.30% US).
- `run_evaluation`: a provider list shorter than `--universe` (the static US list has 107 names) is a Korean note and
  `meta.universe_listed` (title "유니버스 요청 N → 제공 목록 M종목"), never a silent cap; `timings.total` includes the price load.

## Verified data sources & gotchas (live-probed 2026-09-24; details in docs/sources/*.json, code in docs/sources/reference/)

Treat the reference modules as **verified starting points**: port their logic into the provider classes, don't import them.

**KR prices**
- Use Naver fchart (`naver_ohlcv`). It is adjusted for splits and rights issues but NOT for dividends, and it is capped at 3000
  bars.
- **CRITICAL:** since 2026-09-14, recent closes on Naver and Daum are the integrated KRX+NXT 20:00 last trade. Apply
  `official_close_fix`, which uses the Daum base price and, for the last day, the Daum `regularTradePrice`.
- Fall back to yfinance .KS/.KQ, but **never splice** Naver and Yahoo series.
- Trading-halt rows have O/H/L = 0; fix them. Index volume is in thousands. Use `kr_market_of` for KS vs KQ.
- pykrx needs a KRX login and returns empty frames silently, so avoid it.

**KR text**
- News: Naver m.stock news JSON plus Google News RSS (strip " - 출처", dedupe). Filter by relevance to the company name,
  because generic market wraps get tagged.
- Filings: OpenDART when `DART_API_KEY` is set, otherwise the keyless `dart_web_search` plus `dart_web_document_text`. Pages
  are cp949 for KRX single-page documents. `rcpNo` digit 8 or 9 means KRX.
- Earnings: the latest 영업(잠정)실적. No KR transcripts exist, so use 잠정실적, IR disclosures and Naver research summaries.
- Naver disclosure times are bogus; use the dates only.

**US prices**
- Chain: yfinance `history(auto_adjust=True)` → raw Yahoo v8 chart (short UA) → Nasdaq API (unadjusted; flag it).
- Set `yf.config.debug.hide_exceptions=False`.
- **Drop the live partial bar for today.**
- Yahoo `close=None` session holes (live 2026-09-24: 43/60 universe names lacked 2026-09-22) are repaired one bar at a time.
  First the raw v8 chart, always a wide window up to now, because a narrow window returns the running session's partial OHLV
  stamped with an earlier date. Then Nasdaq. Each bar is scaled onto the served series using the close ratio on the
  neighbouring sessions. The chart's ex-dividend and split dates decide which side applies; if they cannot, the bar is not
  repaired. Each bar is validated (OHLC consistency, at most 0.4 log move against its neighbours), listed in
  `attrs["repaired_sessions"]` (kept by `clean_prices`) and noted as "가격: Yahoo 결측 N거래일 보정(...)". An unrepairable hole
  keeps the "결측" warning with the reason. The series is never replaced by another source.
- Benchmark is **SPY**, because ^GSPC has missing days. ^VIX is used as a forward-filled level only.
- **FX** has ONE definition for both providers (`quantcio.data.us.cached_usdkrw` / `usdkrw_fx`; `KRProvider` delegates to it):
  the last COMPLETED KRW=X daily bar (London date before today in London) at or before the last completed US session, never the
  live bar. The fallback is Naver `FX_USDKRW` /prices (Hana Bank first-notice rate, dated today KST or earlier), and its use is
  always flagged. Both providers share one cache entry (namespace `us_fx`, keyed by the US session and the London date), so KR
  and US tickers convert with the identical rate, and KRW→USD is exactly 1/(USD→KRW). Unsupported pairs raise
  `UnsupportedCurrencyPair` (both NotImplementedError and ValueError). *(Superseded: "KRW=X aligned to the stock calendar, with
  Naver as the fallback".)*
- `.info` units are mixed: dividendYield and debtToEquity are in PERCENT, and recommendationMean uses 1 = Strong Buy.

**US text**
- SEC EDGAR (User-Agent required): the 8-K Item 2.02 → EX-99.1 (+99.2) earnings release, 10-Q/10-K Item 1A risk factors and
  MD&A (`edgar_text.extract_section`), and XBRL companyfacts for YoY.
- Earnings-call transcripts come from Motley Fool (`fool_transcripts.latest_transcript`, keyless).
- News: yfinance `get_news(count=40)` (use a fresh Ticker; relevance is loose) plus Google News RSS; Yahoo RSS is the fallback.

**Everywhere**
- Use `pd.Timedelta(n, "D")`, NEVER `pd.Timedelta(days=n)`: it is deprecated under numpy 2.5 and will become an error.

**Claude SDK** (`llm_client_reference.py`)
- Call `client.beta.messages.create(model, max_tokens, system=[cached block], thinking={"type":"adaptive"},
  output_config={"effort":…, "format":{"type":"json_schema","schema":transform_schema(Model)}},
  betas=["server-side-fallback-2026-07-01"], fallbacks="default")`. Then validate the last text block yourself.
- Constraints on the LLM wire schemas:
  - **no dict fields** (they are forced to `{}`);
  - **no defaults** (a field with a default is dropped from `required`);
  - ge/le and minItems ≥ 2 are enforced only client-side, so clamp, or re-sample once.
- If a 400 says the beta was rejected, retry without fallbacks.
- A missing credential raises TypeError at the first request, so run a preflight check.
- APITimeoutError is a subclass of APIConnectionError, so catch it first.

**PDF** (`render_pdf.py`)
- Use `PdfRenderer` (reused) with the Chrome channel first. `@page` CSS is the single source of truth for margins.
- `fit_to_pages` scales 1.0 down to 0.8.
- `fig_to_svg` prefixes SVG ids. Use `-webkit-print-color-adjust: exact`.
- Sync Playwright cannot run inside an event loop; use `asyncio.to_thread`.

## Environment
- Python 3.12 venv at `.venv`. Run tests with `.venv/bin/python -m pytest -m "not live"`.
- `ANTHROPIC_API_KEY` is required for the LLM layer; without it, use `--no-llm`.
- `DART_API_KEY` is optional (KR filings; there is a keyless Naver/DART-web fallback).
- `SEC_USER_AGENT` is required for EDGAR (e.g. "Your Org you@yourdomain.com").
