# RSMT-Q v1.1 — risk / distribution engine: sample efficiency, multiple testing, target scale, leaf γ, distribution gate

Status: section 1 (the pre-declared comparison) was written on 2026-09-29 BEFORE any comparison was run.
Sections 2-4 describe the implementation (written while the runs were in progress; they change no rule); section 5
holds the results, appended after the runs.

## 1. Pre-declared comparison (written before running)

### 1.1 Variants (per market, KR and US)

Every variant keeps everything else of the current configuration (partition variables, per-market alpha / min_leaf /
max_depth / half-life / ν bounds, the date-cluster multiplier bootstrap, the market leg, the plausibility clamp).

| id | overlap_step | multiple_testing | target_scale | gamma_mode | switches on |
|---|---|---|---|---|---|
| V0 | 0 (every 20 sessions, non-overlapping) | bonferroni | none | global | 0 |
| V1 | **5** (weekly, overlapping labels, 20-session block clusters) | bonferroni | none | global | 1 |
| V2 | 0 | **simes** | none | global | 1 |
| V3 | 0 | **pca** (K = 3, Bonferroni m = 3) | none | global | 1 |
| V4 | 0 | bonferroni | **ewma** | global | 1 |
| V5 | 0 | bonferroni | none | **leaf** (k = 50 blocks) | 1 |
| V6 | **5** | **simes** | **ewma** | **leaf** | 4 |
| V7 | **5** | **pca** (K = 3) | **ewma** | **leaf** | 4 |

Fixed, pre-declared constants: overlap step 5 sessions, PCA K = 3, γ shrink k = 50 independent blocks, PCA fitted on
the training rows of each fit only.

### 1.2 Protocol

- Data: the real point-in-time panel the pipeline uses (`pipeline.training_panel` → `quant.panel.build_panel`, PIT
  top-`universe_size` universe with delisted names), with the registered holdouts removed first
  (`holdout.guard_panel` on the PricePanel: XSH names dropped, prices cut before the forward-holdout start), exactly as
  `quantcio rsmt --tune` does.
- **Final holdout**: the last 24 months of sample dates (coarse 20-session grid dates d > last date − 24 months) are
  never used for the choice. Pre-holdout rows are those whose label window ends before the first holdout date.
- **Choice data**: the existing expanding-window purged CV (`cross_validate_layouts`) on the pre-holdout rows, for each
  fold layout 3, 4 and 5. Fold blocks are always cut on the coarse 20-session grid and every variant is evaluated on
  the SAME coarse test rows (overlapping variants only add training rows), so the CRPS of every variant is measured on
  identical (date, name) rows. The baseline twin (root only, no offsets, raw target) is the same for every variant.
- Metrics per variant and layout (pooled OOS rows of the layout): mean CRPS of the 20-day log return (PRIMARY, lower
  is better), cover80 (share of r inside [q10, q90]), exc95 (share below the 5% quantile), P(up) AUC and log loss
  (reported only), number of splits of the final pre-holdout fit (and per CV fold), fit seconds.
- **Selection rule**:
  1. CRPS̄ = mean over the 3 layouts of the layout's mean OOS CRPS; cover80̄ and exc95̄ = the same means of the
     layouts' pooled cover80 / exc95.
  2. Eligible = 0.75 ≤ cover80̄ ≤ 0.85 AND 0.02 ≤ exc95̄ ≤ 0.08.
  3. Winner = the eligible variant with the lowest CRPS̄. Any eligible variant whose CRPS̄ is within 0.1% (relative)
     of the lowest is TIED with it; among tied variants the simpler one wins: fewer switches on (table above), then
     the lower V-number.
  4. No eligible variant → V0 stays.
- **Holdout (once)**: the winner and V0 are each fitted on all pre-holdout rows and evaluated once on the holdout rows
  (CRPS, cover80, exc95, AUC, log loss, and the baseline twin). The holdout numbers are reported, never used to
  change the choice.
- **Default**: the winner becomes the default in `config/default.yaml` (`rsmt` block; per-market overrides under
  `rsmt.markets.<M>` when the winners differ by market).
- **Winner's curse**: 8 variants compete on the same CV rows, so the winner's CV CRPS is optimistically biased; the
  one-time holdout comparison against V0 is the unbiased check. A winner that does not beat V0 on the holdout is still
  the rule's winner (the rule is not changed after seeing the holdout), but the report says so.
- Reproducible: `scripts/rsmt_compare.py --market KR|US` writes `reports/rsmt/compare_<M>.md` and `.json`.

## 2. Role separation and the distribution skill gate

- **Role.** RSMT-Q is the RISK / DISTRIBUTION engine: σ_e (with the η offset), ν, ρ, β, the market leg's σ / ν, the
  quantiles, VaR / CVaR and the stop-loss / take-profit first-touch probabilities, all from one path matrix. The
  beta-adjusted excess MEAN comes 100% from the selected stock-selection model whenever its serving artifact exists
  (`quant.rsmt.ExcessOverride`, built by `modeling.serve.excess_override`): `MarketModel._prepare` then sets the
  served total mean to β·μ_m + μ_excess(model) and records `blend = {model_id: 1.0}`, and `relative_edge = 0`.
  RSMT-Q's own leaf mean + γ·z (`blend = {"rsmt": 1.0}`) is used only when there is no artifact. (2026-10-02: the
  per-ticker HGB+Ridge ensemble and its 20% `serving_blend` were deleted — 0/15 names passed its skill gate in every
  evaluation.) The plausibility clamp |μ| ≤ 0.5·σ_h still applies to the served mean (it only binds on
  pathological values).
- **Skill gate** (`rsmt.skill_metric`, default `distribution`), per fold layout (3/4/5, strict majority):
  - distribution: date-block bootstrap 5% bound of the CRPS gain over the baseline twin > 0 (whole CV test dates
    resampled, small-cluster corrected like the AUC bound) AND cover80 ∈ `skill_cover80_band` [0.75, 0.85] AND
    exc95 (VaR95 exceedance rate) ∈ `skill_exc95_band` [0.02, 0.08];
  - direction (the old gate, `skill_metric: direction`): AUC gain ≥ 0.01, log loss < baseline, AUC-gain bound > 0.
  Both verdicts are always computed: `CVResult.direction_ok` / `distribution_ok`, cv_metrics `direction_gate`,
  `distribution_gate`, `direction_gate_folds_<k>`, `distribution_gate_folds_<k>`; `gate_folds_<k>` (read by the
  report) follow `skill_metric`.
- **Served verdicts.** `DistributionForecast.skill_ok` = RSMT-Q's own verdict of `skill_metric` (it certifies the
  distribution, also when the mean is overridden). `DistributionForecast.direction_skill_ok` = the skill behind the
  served mean's direction: the stock-selection model's served skill under an ExcessOverride, else RSMT-Q's direction
  gate. The directional ML-pillar multiplier (`quant.distribution_ml_pillar`, ×1.0 / ×0.35) reads
  `direction_skill_ok` only, so a distribution pass never un-shrinks a direction.

## 3. Model switches (`rsmt` block; per market in `rsmt.markets.<M>`, `null` = inherit)

| key | values (original) | what it does |
|---|---|---|
| `overlap_step` | 0 | k > 0: training rows every k sessions (k must divide `sample_step`), overlapping h-day labels. The split test clusters on h-session blocks; CV folds, CV test rows and the market leg stay on the non-overlapping grid (`on_grid`); `min_leaf` counts non-overlapping rows (×sample_step/k training rows) and the leaf-mean / ρ shrinkage uses n_eff = n·k/sample_step. (`sample_step` itself is forced ≥ h by `load_settings`, hence a separate key.) |
| `multiple_testing` | bonferroni | `simes`: node p = Simes' global p = min_j m·p_(j)/j (per-variable BH-adjusted p shown). `pca`: the partition variables → `pca_components` principal components (standardised, correlation eigenvectors, fitted on each fit's training rows; the stored mean / scale / loadings route new rows; a missing input is imputed at its mean and the row flagged); Bonferroni over K. |
| `pca_components` | 3 | K for `pca`. |
| `target_scale` | none | `ewma`: the tree splits on ε = e / σ_v,h with σ_v,h = EWMA vol at t × √(h/252) (panel column `scale_h`), so high-vol regimes do not dominate the scores; leaves hold (μ, σ) of ε; served μ_e = σ_v,h·(μ_leaf + γ·z), σ_e = σ_v,h·σ_leaf·exp(η·v). **η in this mode**: kept as the RESIDUAL volatility elasticity (start 0, bounds [−1.5, 1.0] = raw [−0.5, 2.0]); the raw-model equivalent η = 1 + η is what diagnostics / offsets report. γ is then in σ_v units (the momentum effect scales with vol). |
| `gamma_mode` | global | `leaf`: after the tree and the global (γ, η) are fitted, every node's own WLS slope γ̂ (within-node intercept, weights w/s²) is shrunk top-down: γ_node = (n_eff·γ̂ + k·γ_parent)/(n_eff + k), n_eff = the node's distinct independent sample dates / blocks, k = `gamma_shrink_k`; the root keeps the global γ (a tree without splits is unchanged); leaf μ / σ are refitted with the leaf γ. The split test still uses the global γ offset. |
| `gamma_shrink_k` | 50 | prior weight of the parent γ, in independent blocks. |
| `skill_metric` | distribution | `direction` restores the old AUC gate as the served `skill_ok`. |
| `skill_cover80_band` / `skill_exc95_band` | [0.75, 0.85] / [0.02, 0.08] | distribution-gate bands. |

**Dependence and HAC equivalence.** With overlapping labels, two rows whose windows start d < h sessions apart
share (h−d)/h of their daily returns, so their score covariance follows the Bartlett kernel with bandwidth h. The
multiplier bootstrap draws ONE multiplier per h-session block; two rows d sessions apart fall into the same block
with probability (h−d)/h, so the bootstrap variance of the partial score sums is, on average over block alignment,
the Bartlett-kernel HAC variance of the cross-sectionally summed scores — i.e. a Driscoll–Kraay estimator with
bandwidth h (cross-sectional dependence within a date is kept because all names of a date share the multiplier).
On the non-overlapping grid every block holds one sample date, so this is exactly the previous date-cluster
bootstrap.

## 4. RSMT-Q v3 stock-selection candidate (`rsmt_v3`, `modeling/candidates.py`)

`RSMTRanker` with the same switches, pre-declared as ONE configuration (no tuning): Simes / BH over the 8 partition
variables (4 cross-sectional ranks + 4 regime variables), target y_dm / (vol60 × √(h/252)) with the residual η,
leaf-local stacked-score γ (k = 50 blocks), 20-session block clusters (as v2), α = 0.05, depth 3, half-life 756 d.
It ranks by the STANDARDISED mean (μ_leaf + γ_leaf·z): P(y beats the date median) ≈ Φ(μ_e/σ_e) does not depend on
the scale, whereas the return-unit mean would re-rank names by volatility (`predict` still returns μ_e in returns).
It is stacked and checkpointed exactly like `rsmt_v2` (`candidates.STACKED`, `selection.run_rsmt(name=…)`, one
checkpoint per stacked candidate) and competes under the unchanged selection protocol v2.

## 5. Results of the pre-declared comparison (run 2026-09-29 after section 1 was written; sections 2-4 only document the implementation)

Reproduce: `.venv/bin/python scripts/rsmt_compare.py --market KR` (and `US`); full tables, per-layout numbers and the
trees are in `reports/rsmt/compare_<M>.md|json`. Panels: KR 4,989 non-overlapping rows (85 sample dates from
2019-10-22; 19,885 rows with overlap step 5), US 5,220 (87 dates from 2019-10-23; 20,880). Holdout guard: no XSH /
forward-holdout rows in these universes. CV on the pre-holdout rows (KR 3,434 / US 3,600 non-overlapping rows),
final holdout = 25 (KR, from 2024-08-30) / 26 (US, from 2024-08-29) sample dates. CRPS of the 20-day log return,
means over layouts 3/4/5; the baseline twin's CRPS is identical for every variant (same test rows): KR 0.056001, US
0.044767.

### KR (alpha 0.10)

| variant | CRPS | vs V0 | cover80 | exc95 | AUC | logloss | splits final (max per fold) | dist. gate layouts | fit s | CV s |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 0.055413 | — | 0.839 | 0.037 | 0.472 | 0.6984 | 0 (0) | 2/3 | 0.2 | 8.7 |
| V1 overlap | 0.055326 | −0.16% | 0.835 | 0.038 | 0.472 | 0.6974 | 3 (2) | 3/3 | 1.0 | 10.0 |
| V2 simes | 0.055407 | −0.01% | 0.840 | 0.037 | 0.471 | 0.6985 | 0 (1) | 2/3 | 0.2 | 6.3 |
| V3 pca | 0.055398 | −0.03% | 0.839 | 0.035 | 0.477 | 0.6983 | 0 (2) | 2/3 | 0.1 | 6.6 |
| V4 std. target | 0.055166 | −0.44% | 0.843 | 0.037 | 0.479 | 0.6953 | 1 (2) | 3/3 | 0.1 | 5.7 |
| V5 leaf γ | 0.055413 | 0.00% | 0.839 | 0.037 | 0.472 | 0.6984 | 0 (0) | 2/3 | 0.1 | 5.5 |
| **V6** | **0.055064** | **−0.63%** | 0.841 | 0.037 | 0.486 | 0.6945 | 2 (2) | 3/3 | 0.8 | 7.3 |
| V7 | 0.055080 | −0.60% | 0.838 | 0.037 | 0.481 | 0.6948 | 2 (2) | 3/3 | 0.4 | 6.2 |

Every variant is eligible. Winner **V6** (V7 tied within 0.1%; both have 4 switches → lower V-number). Final holdout
(once): V0 CRPS 0.083232 (twin 0.084106, gain bound +0.00035, cover80 0.798, exc95 0.033, AUC 0.508) vs **V6
0.083168** (gain bound +0.00017, cover80 0.802, exc95 0.027, AUC 0.518): V6 is 0.08% better than V0 on the holdout,
inside the tie tolerance — the CV gain (0.63%) did not carry over in size. V6 tree (pre-holdout fit):
`vol_ratio_20_120 ≤ 0.85` → `dd_250 ≤ −0.227`, leaf γ −0.034 / −0.007 / +0.014 (momentum turns into reversal in
calm-but-drawn-down regimes, consistent with the KR reversal finding of the factor work).

### US (alpha 0.05)

| variant | CRPS | vs V0 | cover80 | exc95 | AUC | logloss | splits final (max per fold) | dist. gate layouts | fit s | CV s |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 0.044532 | — | 0.842 | 0.041 | 0.489 | 0.6929 | 0 (0) | 3/3 | 0.1 | 6.2 |
| **V1 overlap** | **0.044173** | **−0.81%** | 0.817 | 0.048 | 0.488 | 0.6925 | 2 (1) | 3/3 | 1.9 | 13.7 |
| V2 simes | 0.044532 | 0.00% | 0.842 | 0.041 | 0.489 | 0.6929 | 0 (0) | 3/3 | 0.2 | 11.8 |
| V3 pca | 0.044532 | 0.00% | 0.842 | 0.041 | 0.489 | 0.6929 | 0 (0) | 3/3 | 0.1 | 11.5 |
| V4 std. target | 0.044437 | −0.21% | 0.842 | 0.041 | 0.503 | 0.6911 | 0 (0) | 3/3 | 0.2 | 5.7 |
| V5 leaf γ | 0.044532 | 0.00% | 0.842 | 0.041 | 0.489 | 0.6929 | 0 (0) | 3/3 | 0.2 | 9.2 |
| V6 | 0.044138 | −0.89% | 0.817 | 0.047 | 0.495 | 0.6915 | 2 (1) | 3/3 | 2.1 | 15.7 |
| V7 | 0.044170 | −0.81% | 0.817 | 0.046 | 0.495 | 0.6916 | 1 (1) | 3/3 | 0.5 | 7.2 |

Every variant is eligible. V6 has the lowest CRPS but V1 and V7 are within 0.1% → tie → **V1** (1 switch). Final
holdout (once): V0 0.050310 (twin 0.050893, gain bound +0.00026, cover80 0.806, exc95 0.046, AUC 0.513) vs **V1
0.050271** (gain bound +0.00024, cover80 0.797, exc95 0.054, AUC 0.499): 0.08% better, again inside the tie
tolerance. V1 tree: `vol_pct_3y ≤ 0.165` / `vol_ratio_20_120 ≤ 1.11` (three leaves).

### Defaults and reading

- `config/default.yaml`: `overlap_step: 5` for both markets (the shared winning switch), KR additionally
  `multiple_testing: simes`, `target_scale: ewma`, `gamma_mode: leaf` (`rsmt.markets.KR`); US keeps Bonferroni / raw
  target / global γ. `skill_metric: distribution`.
- What moved CRPS: overlapping samples (both markets; they are also what makes the tree split at all - 0 → 2-3
  splits) and the standardised target (both markets). Simes, PCA and leaf γ alone do nothing without splits (a
  root-only tree is unchanged by construction) and add little on top.
- Direction: no variant has direction skill (CV AUC 0.47-0.50, direction gate 0/3 layouts everywhere) - RSMT-Q is a
  risk engine, not a stock picker.
- **Winner's curse**: 8 variants on the same CV rows; the holdout improvements over V0 (0.08% in both markets) are
  far smaller than the CV ones (0.63% / 0.81%) and inside the pre-declared tie tolerance. The honest reading: the
  switches do not make the distribution worse, fit the calibration bands at least as well (US cover80 0.842 → 0.817,
  exc95 0.041 → 0.048, closer to the 0.80 / 0.05 targets), and let the tree find regime splits - but the out-of-sample
  CRPS gain over V0 is small and not established. A forward check (next 6-12 months of served forecasts, CRPS vs V0)
  is the proper confirmation.

### The served gate with the new defaults (full panel incl. the holdout period, as `pipeline.fit_market` runs it)

| market | tree (final fit) | CRPS vs twin (4-fold CV) | CRPS-gain bound | cover80 | exc95 | distribution gate | direction gate (AUC vs twin) |
|---|---|---|---|---|---|---|---|
| KR (V6) | root only, root p 0.16 (Simes), γ +0.016 σ_v units, η≡0.58 | 0.06442 vs 0.06538 | +0.00064 | 0.832 | 0.030 | **pass 3/3 layouts** | fail 0/3 (0.517 vs 0.522) |
| US (V1) | `vol_ratio_20_120 ≤ 1.11` / `≤ 1.47`, 3 leaves | 0.04654 vs 0.04706 | +0.00034 | 0.812 | 0.051 | **pass 3/3 layouts** | fail 0/3 (0.515 vs 0.506) |

So `DistributionForecast.skill_ok` is now True in both markets (the risk distribution beats the unconditional twin
with coverage inside the bands), while `direction_skill_ok` stays False: without a stock-selection artifact the
RSMT-Q directional ML pillar is still ×0.35.

## 6. Time-series extensions — research fork (pre-declared 2026-09-30, written before any run)

**Why a fork.** Forward holdout `FTH-20260930` freezes the signal code (`src/quantcio/` minus `report/`, `cli.py`,
`evaluation.py`, `holdout.py`, `tracking.py`, `xsh.py`) and every config value. The extensions therefore live in
`scripts/rsmt_ts.py`: it subclasses `quant.rsmt.RSMTree` at runtime (`RSMTreeX`) and adds columns to the panel after
`build_panel`; the served model, the code fingerprint and the config digest are untouched. A variant moves into
`src/` only if it wins this comparison, and adopting it means registering a new FTH (the user decides).

**Reading of the proposals (user, 2026-09-30).**
- *Differencing*: the target is already a first difference (20-day log return); differencing it again would discard
  the level information of returns (over-differencing). Differencing the REGIME variables is meaningful: 20-session
  changes `d20_vol_ratio` (vol_ratio_20_120), `d20_mkt_vol` (mkt_vol_60), `d20_dist_ma200` = "is volatility / trend
  rising or falling", which the levels do not say.
- *Time information*: `t_years` (calendar time) as a partition candidate is a structural-break test (the sup-LM test
  was designed for breaks in time); a split sends every future row to the latest segment. Half-life weighting
  (504 d) already does a soft version, so this checks whether a hard break is supported.
- *Cycles (sin / cos)*: day-of-year harmonics, annual (k = 1) and quarterly (k = 4). Prior: the quarterly earnings
  season raises idiosyncratic volatility (US: late Jan-Feb, Apr-May, Jul-Aug, Oct-Nov; KR: preliminary results in
  Jan/Apr/Jul/Oct, filings Mar/May/Aug/Nov), so the harmonics belong mainly in the SIGMA equation. A seasonal MEAN
  needs many years per calendar month (the panel has ~6 years) and is not expected to be detectable.
- *Several variables in one split*: an oblique split c'x ≤ s. With a harmonic pair it is exactly a phase window of
  the year (a·sin + b·cos = A·sin(θ + φ)), which a single sin or cos cannot express.
- *Added*: the multi-horizon (HAR) volatility equation, the standard time-series volatility forecaster
  (Corsi 2009), and the leverage effect (volatility rises after losses).

**Extensions (switches of `RSMTreeX`).**
1. *Sigma equation* `sigma_terms`: log σ_e = log σ_leaf + η·v + Σ_j θ_j V_j, all coefficients fitted jointly with
   the leaves inside the offset loop (vector Newton, `fit_offsets_multi`) on each fold's training rows only.
   HAR terms `har_5`, `har_66`, `har_252` = log(realised vol over 5 / 66 / 252 sessions ÷ EWMA vol); leverage
   `lev = min(ret_1m, 0)`; seasonal `cal_s1, cal_c1, cal_s4, cal_c4` = sin / cos(2π·k·(day of year − 1)/365.25).
2. *Time-series partition candidates*: `t_years`, `d20_vol_ratio`, `d20_mkt_vol`, `d20_dist_ma200` (univariate).
3. *Oblique groups*: sup-LM of the projection c'x̃ (x̃ standardised on the training rows) over a FIXED direction grid
   (pairs: 8 angles 0°, 22.5°, …, 157.5°; triples: the 13 cubic-lattice directions); group statistic = the maximum
   over the directions; its null = the maximum over the SAME directions under the SAME cluster multipliers (joint
   multiplier bootstrap) - the direction search is paid for in the p-value; one test per group in the Simes / BH
   adjustment. Groups: quarterly phase (`cal_s4`, `cal_c4`), annual phase (`cal_s1`, `cal_c1`), market state
   (`mkt_vol_60`, `mkt_ret_60`), stock state (`vol_ratio_20_120`, `dd_250`, `ret_1m`). The split is on the best
   direction; new rows are routed with the stored standardisation.

**Variants** (W0 = the served configuration of each market: KR V6, US V1; every other variant adds to it):

| variant | adds | components |
|---|---|---|
| W0 | — (current) | 0 |
| W1 | HAR sigma terms | 1 |
| W2 | HAR + leverage | 2 |
| W3 | HAR + leverage + seasonal sigma terms | 3 |
| W4 | time-series partition candidates (Simes in both markets) | 1 |
| W5 | W4 + the 4 oblique groups | 2 |
| W6 | W3 + W5 | 5 |
| W7 | seasonal sigma terms only | 1 |

**Protocol** (as section 1): the market's PIT training panel with the registered holdouts removed, overlap step 5;
extra columns merged by (ticker, date) and rows with a missing extra dropped for ALL variants (same rows everywhere,
count reported); final holdout = the last 24 months of sample dates, never used for the choice; CV layouts 3/4/5 on
the pre-holdout rows against the unchanged baseline twin. **Rule**: lowest mean CRPS over the layouts among variants
with mean cover80 in [0.75, 0.85] and mean exc95 in [0.02, 0.08]; a relative CRPS difference < 0.1% is a tie →
fewer components, then the lower W-number. Reported but not used: direction AUC / μ rank IC (the "time-series
prediction" of the direction), the fitted θ, splits. The holdout is evaluated once, for W0 and the winner.
**Adoption bar** (in addition to winning): the winner's holdout CRPS must beat W0's by more than the 0.1% tie
tolerance; otherwise nothing moves into `src/`.

**Protocol deviation (recorded before the comparison finished).** A smoke run of W6 alone (KR, 200 paths, to catch
crashes before the long run) also evaluated the final holdout of W0 and W6 once, because the script did not yet skip
it in smoke mode (fixed: `--only` runs never touch the holdout). The choice rule uses the CV only and was not
changed; the KR holdout has nevertheless been looked at once more than declared, which weakens its role as an
unbiased check for W6 slightly. US was not affected.

### 6.1 Results (run 2026-09-30, rule unchanged; `reports/rsmt/ts_compare_<M>.md|json`)

Panels: KR 19,941 training rows (4,989 on the grid, 85 sample dates from 2019-10-23; 1 row dropped for a missing
extra), US 20,880 (5,220, 87 dates from 2019-10-24; none dropped). CV on the pre-holdout rows (KR 13,901, US 14,820),
layouts 3/4/5, 1000 paths; the twin's CRPS is the same for every variant (KR 0.056136, US 0.043758).

| variant | KR CRPS | vs W0 | KR splits | US CRPS | vs W0 | US splits | US dist. gate |
|---|---|---|---|---|---|---|---|
| W0 current | 0.055120 | — | 2 | 0.043241 | — | 2 | 3/3 |
| W1 HAR σ | 0.055140 | +0.04% | 0 | 0.043236 | −0.01% | 1 | 3/3 |
| W2 HAR + leverage | 0.055131 | +0.02% | 0 | 0.043235 | −0.01% | 1 | 3/3 |
| W3 HAR + leverage + seasonal σ | 0.055304 | **+0.33%** | 0 | 0.043533 | **+0.67%** | 1 | **0/3** |
| W4 time-series split candidates | 0.055113 | −0.01% | 0 | 0.043241 | 0.00% | 2 | 3/3 |
| W5 W4 + oblique groups | 0.055113 | −0.01% | 0 | 0.043239 | −0.00% | 1 | 3/3 |
| W6 everything | 0.055324 | +0.37% | 0 | 0.043528 | +0.66% | 1 | 0/3 |
| W7 seasonal σ only | 0.055116 | −0.01% | 2 | 0.043280 | +0.09% | 2 | 3/3 |

**Winner: W0 in both markets** (every variant eligible; the lowest CRPS - KR W5, US W2 - is within 0.1% of W0 → tie
→ fewer components). Nothing moves into `src/`; FTH-20260930 stays valid. Final holdout (once, W0 only): KR CRPS
0.085907 vs twin 0.087001, US 0.048456 vs 0.049039.

Reading:
- **HAR volatility equation**: the coefficients are real and sensible - the 3-month realised vol carries weight beyond
  the EWMA (θ_har66 +0.36 KR, +0.70 US; the EWMA elasticity drops accordingly: volatility mean-reverts toward its
  longer-horizon level) - but the CRPS moves by ≤ 0.04%. The 20-day return CRPS is dominated by the unpredictable
  location (market + idiosyncratic shock); a better σ within ±10% barely moves it, and the coverage was already in the
  bands. Leverage has opposite signs (US −0.61 = volatility after losses, the classical effect; KR +0.80 = lower
  volatility after a 1-month loss, consistent with the KR reversal finding) and adds nothing either.
- **Seasonal σ terms** are harmful combined with HAR (+0.33% / +0.67%, the US distribution gate fails in all three
  layouts) and neutral alone. All four harmonics come out at a similar +0.05…0.07, i.e. they fit date-level
  volatility shocks of the ~70 pre-holdout dates rather than an earnings-season cycle: a date-level regressor fitted
  on 60 rows per date treats one market shock as 60 observations. With ~6 years the quarterly cycle has ~24
  repetitions, too few to separate from shocks.
- **Time-series split candidates and oblique splits**: no test passed. Adding candidates makes the Simes / Bonferroni
  bar higher, so KR even lost its two V6 splits (W4/W5: 0 splits). The oblique groups - including the harmonic phase
  pairs - never reached significance under the joint (search-aware) null. The tree is sample-limited: 85 dates.
- **Direction ("time-series prediction")**: CV AUC 0.478-0.485 in every variant (twin 0.49), μ rank IC −0.03 to
  −0.05. No extension gives the RSMT-Q mean any directional information; the direction stays with the stock-selection
  model (section "Role separation").
- **What would help**: the binding constraint is the number of independent dates, not the model form. The panel
  starts in 2019-10 (history window of the 60-name training universe); a longer PIT history (2012+ for US, 2016+ for
  KR, as the modeling datasets use) would roughly double the dates and is the only lever likely to make the seasonal
  and oblique tests informative. It is a config change (`history_years` / the training universe), so it would be
  evaluated with this fork first and adopted only together with a new FTH registration.

## 7. Longer history (research fork, pre-declared 2026-09-30, written before any run)

**Why.** Section 6 found nothing, and the binding constraint is the number of independent dates: the RSMT-Q panel
starts in 2019-10 only because `history_years: 7` (the project's initial default for price loading) sets the start of
the PIT training universe (`pipeline.training_panel`: today − 7 years). It is not a data limit.

**Maximum validated history.**
- KR: FinanceData/marcap daily KRX snapshots exist from 1995 (the pipeline used `pit.kr.start_year: 2013`). Research
  runs set `pit.kr.start_year = 1995`, `history_years = 31` (universe from 1995-08; members need 120 sessions, the
  features 252 sessions of history).
- US: the price validation (Yahoo series cross-checked against SEC fails-to-deliver prices) and the XBRL share counts
  used for the market-cap rank both start in mid-2009 (the SEC FTD listing begins 2009-07). Before that a series
  would be unvalidated (wrong-issuer risk for reused tickers). Research runs set `history_years = 17` (universe from
  2009-09; `history_years` is an integer).

**Time decay caps the gain.** Training rows are weighted 0.5^(age / half_life), half_life = 504 sessions (~2 years). The
Kish effective sample of an infinite exponentially weighted history is 2·half_life / ln 2 ≈ 1,450 sessions ≈ 73
twenty-day dates, only somewhat above the 7-year panel. A longer history therefore only helps together with a longer
half-life, so the half-life is part of the comparison.

**Stage 1 - data window and half-life** (current model W0 of each market):

| config | history | half_life_days | changes |
|---|---|---|---|
| L0 | 7 years (current) | 504 | 0 |
| L1 | maximum | 504 | 1 |
| L2 | maximum | 1,260 (5 years) | 2 |
| L3 | maximum | none (equal weights) | 2 |

All four are scored on the SAME test rows: the CV folds (layouts 3/4/5) of the L0 panel before the final holdout.
For each fold every config trains on ALL of its own rows with t_end < the fold's test start (the `cv_splits` purge),
so a config differs only in how much (and how weighted) past it learns from; test rows are the (ticker, date) rows of
the L0 fold present in every panel. Rule: lowest mean CRPS over the layouts among configs with mean cover80 in
[0.75, 0.85] and exc95 in [0.02, 0.08]; < 0.1% relative difference = tie → fewer changes, then the lower number.

**Stage 2 - the section 6 variants on the long panel.** W0-W7 (section 6, unchanged) on the maximum-history panel with
the half-life of the best of L1-L3 (the stage-1 rule restricted to L1-L3), standard CV (`cross_validate_layouts`,
layouts 3/4/5) on its pre-holdout rows, same rule and tie tolerance.

**Holdout** (last 24 months of sample dates, as before): evaluated once for L0 and the stage-1 winner, and once for W0
and the stage-2 winner. **Adoption bar**: a winner must also beat its baseline on the holdout by more than 0.1%;
adoption means a config change (an RSMT-specific history setting) and a new FTH registration - the user decides.

### 7.1 Revision before any result (2026-09-30): every business day, event-risk variables, exposure cap

The user asked to train, validate and infer on EVERY business day of the longest history. The first section-7 run
(KR, overlap step 5, grid test rows) was stopped before any of its output was read. One timing fit (KR, maximum
history, daily rows, pre-holdout, half-life 1260: 389,493 rows, 212 s, 4 splits) was run to size the job; its CRPS was
not computed. Checks made while sizing: KR KOSPI/KOSDAQ from Naver reach back only 3,000 sessions (2014-07); before
that the benchmark is the marcap cap-weighted replica index spliced in by `data.pit.kr` when it tracks the index; the
holdout guard drops the 6 registered KR XSH names (36,485 rows) that were top-60 members in earlier decades.

**All business days.**
- Training: `overlap_step = 1` (a row every session) for the maximum-history configs; clustering on h-session blocks
  (the tree's multiplier bootstrap) keeps the tests valid. Information: overlapping 20-day labels share 19/20 of
  their window, so daily rows add little over the 5-session step (the variance of an overlapping-window mean is ~2/3
  of the non-overlapping one already at step 5); the gain is mainly that no sample date is thrown away.
- Validation: the test rows of a fold are ALL sessions of its test block (not only the 20-session grid); the
  date-bootstrap confidence bounds resample h-session blocks (daily rows of one block share their label window).
- Folds: `cv_splits` boundaries (grid blocks), purge t_end < test start, as before.

**Stage 1** (unchanged configs, all-day test rows): L0 = the served model (7 years, step 5); L1-L3 = maximum history,
daily rows, half-life 504 / 1260 / equal. Common test rows = every session in the L0 folds before the holdout. Change
counts: L0 0, L1 2 (history + daily), L2 / L3 3.

**Stage 2** on the maximum-history daily panel with the best long half-life, all-day CV: W0, W1, W2, W3, W4, W5, W7
(section 6; W6 is dropped for compute - its parts are W3 and W5) plus three EVENT-RISK variants (proposal A of the
user's note, implemented with observable, historically calibratable data instead of an LLM severity score):

| variant | adds | components |
|---|---|---|
| E1 | event-risk terms in the sigma equation | 1 |
| E2 | event-risk variables as split candidates (Simes) | 1 |
| E3 | E1 + E2 | 2 |

Event-risk variables (date-level, point in time; `scripts/rsmt_ts.py` `event_columns`):
- `ev_vix` = log VIX close (Yahoo ^VIX) of the last US session ≤ t (US rows) or < t's date (KR rows: the US session of
  that date closes after Seoul);
- `ev_vrp` = log(VIX/100) − log(21-session realised vol of SPY), same session: implied minus realised volatility - the
  market-priced risk of events that have not moved prices yet;
- `ev_gpr` = log(7-day mean of the Caldara-Iacoviello daily Geopolitical Risk index ending 8 calendar days before t) −
  log(its trailing 365-day median) (the file is updated weekly, so an 8-day lag is always published);
- `ev_epu` = the same for the Baker-Bloom-Davis daily US Economic Policy Uncertainty index with a 2-day lag (updated
  daily). Korea's EPU (monthly, last update 2021) is not usable.
Limits: GPR / EPU are newspaper-count indices whose archives can be revised; no vintages exist, so a small look-ahead
through revisions cannot be excluded.

Why not an LLM severity score for the model: scoring past news with today's LLM leaks the outcome (the model has read
what happened next), so its history cannot be backtested honestly; it can only be recorded forward and calibrated
later. The data-driven variables above can be tested on 1996-2026 (KR) / 2009-2026 (US) now.

Rule, holdout and adoption bar: as section 7 (lowest mean CRPS within the bands, 0.1% tie → fewer components, then
the lower name in the order W0, W1, W2, W3, W4, W5, W7, E1, E2, E3; holdout once for the baseline and the winner).

**Exposure cap (proposal D), separate pre-declared test** (`scripts/event_cap.py`): the market benchmark (KR: KOSPI
with the replica splice, 1996-; US: SPY, 1996-), daily, gross exposure set at the close of t and applied from t+1,
10 bp per unit of exposure change.
- D0: volatility target (the risk module's rule at market level): exposure = min(1, 15% / EWMA vol).
- D1: D0 × 0.6 while `ev_vix` is above its trailing 3-year 90th percentile.
- D2: D0 × 0.6 while `ev_gpr` is above its trailing 3-year 95th percentile.
- D3: D0 × 0.6 while either holds.
Metrics: Sharpe, annual return, volatility, maximum drawdown, Calmar; the Sharpe difference vs D0 with a 21-session
block-bootstrap 90% interval; the last 24 months reported separately. **Adoption**: a trigger is adopted only if its
Sharpe difference vs D0 has a lower bound > 0 in BOTH markets before the holdout; a trigger that only lowers the
drawdown at a Sharpe cost is reported as a risk-preference choice for the user, not adopted by rule.

**Protocol note (before the full run).** A smoke run (`--smoke`: KR, layout 3 only, W0 / E3 only, 200 paths, no
holdout) checked the pipeline end to end and printed its layout-3 CV numbers; the rule above is unchanged. It
confirmed identical test rows across L0-L3 in every fold (16,189 / 15,629 / 15,633 rows) and the task times used to
size the run (KR long fits 3-11 minutes each).

### 7.2 Exposure cap result (proposal D; `reports/rsmt/event_cap.md|json`, run 2026-09-30, rule unchanged)

Market level, 1996-2026 daily, before the holdout (from 2024-09); Sharpe difference vs D0 with the 21-session
block-bootstrap 90% interval. Triggers were on for 15% (VIX) / 6% (GPR) of the sessions.

| market | rule | ann. return | vol | Sharpe | Sharpe − D0 [90%] | MDD |
|---|---|---|---|---|---|---|
| KR | buy & hold | +4.18% | 25.2% | 0.289 | — | −69.3% |
| KR | D0 vol target | +1.93% | 14.4% | 0.204 | — | −52.5% |
| KR | D1 VIX cap | +1.73% | 13.5% | 0.195 | −0.010 [−0.065, +0.048] | −47.7% |
| KR | D2 GPR cap | +1.31% | 14.2% | 0.163 | **−0.041 [−0.070, −0.013]** | −50.8% |
| KR | D3 either | +1.47% | 13.3% | 0.176 | −0.028 [−0.085, +0.031] | −45.0% |
| US | buy & hold | +9.97% | 19.3% | 0.588 | — | −55.2% |
| US | D0 vol target | +8.19% | 13.4% | 0.653 | — | −44.0% |
| US | D1 VIX cap | +6.93% | 12.3% | 0.606 | −0.047 [−0.098, +0.007] | −47.2% |
| US | D2 GPR cap | +7.34% | 13.2% | 0.603 | **−0.051 [−0.078, −0.024]** | −46.0% |
| US | D3 either | +6.34% | 12.1% | 0.568 | **−0.085 [−0.142, −0.026]** | −47.2% |

**Verdict: no trigger is adopted** (no Sharpe gain anywhere; the GPR trigger is significantly worse in both markets).
The caps act AFTER the shock is visible (VIX / GPR are already high), so they mostly de-risk into the rebound
(US 1997-98 +40.6% buy & hold vs +12.4% with D3; KR 2024-25 +10.0% vs +2.5%). The only benefit is a shallower KR
drawdown with D1 / D3 (−47.7% / −45.0% vs −52.5%) at a lower Sharpe - a risk-preference choice, not an edge; in the US
the drawdown is even deeper. Holdout (2024-09 onward) says the same: every cap has a lower Sharpe than D0 in both
markets. Side finding: the market-level volatility target itself helps the US (Sharpe 0.653 vs 0.588) but not KR
(0.204 vs 0.289) before the holdout. The existing macro overlay and the risk module's portfolio volatility target stay
as they are.

### 7.3 Results (run 2026-09-30/10-01, rules unchanged; `reports/rsmt/history_<M>.md|json`)

Panels: KR 420,711 daily rows from 1996-05-22 (7,250 sessions, 333 entities; L0: 99,474 rows from 2019-09-30), US
251,260 from 2009-08-31 (4,276 sessions, 146 entities; L0: 104,340). Kish effective dates: L0 45.7 / 46.9, L1 48.6 /
49.9, L2 121.5 / 117.2, L3 364 / 214 (KR / US). Every stage-1 fold scored identical test rows in all four configs;
the holdout rows are identical in every holdout run (KR 28,758, US 28,860 rows, every session from 2024-09).

**Stage 1** (CV CRPS, every session of the L0 folds; holdout once):

| market | config | CV CRPS | vs L0 | cover80 | exc95 | CV AUC / μ IC | holdout CRPS | vs L0 |
|---|---|---|---|---|---|---|---|---|
| KR | L0 served | 0.056298 | — | 0.824 | 0.045 | 0.487 / −0.027 | 0.080104 | — |
| KR | L1 max, hl 504 | 0.055977 | −0.57% | 0.819 | 0.045 | 0.516 / +0.047 | | |
| KR | **L2 max, hl 1260** | 0.055909 | **−0.69%** | 0.814 | 0.048 | 0.514 / +0.045 | 0.080103 | **0.00%** |
| KR | L3 max, equal | 0.055917 | −0.68% | 0.820 | 0.046 | 0.518 / +0.074 | | |
| US | L0 served | 0.043232 | — | 0.819 | 0.043 | 0.480 / −0.046 | 0.051651 | — |
| US | L1 | 0.043064 | −0.39% | 0.807 | 0.048 | 0.489 / −0.026 | | |
| US | **L2** | 0.042967 | **−0.61%** | 0.803 | 0.052 | 0.500 / +0.002 | 0.051726 | **+0.15%** |
| US | L3 | 0.042934 | −0.69% | 0.779 | 0.060 | 0.504 / +0.009 | | |

Winner L2 in both markets (L3 tied within 0.1% and has the same change count → lower number). **Adoption bar not met**:
on the holdout L2 equals L0 in KR and is 0.15% worse in US (80% coverage 0.775 → 0.760).

**Stage 2** (maximum-history daily panel, half-life 1260, standard folds, every-session CV):

| variant | KR CV vs W0 | US CV vs W0 |
|---|---|---|
| W1 HAR σ | **−0.13%** (winner) | −0.08% |
| W2 HAR + leverage | −0.14% | −0.13% |
| W3 HAR + leverage + seasonal σ | −0.13% | −0.21% |
| W4 time / change split candidates | −0.01% | −0.03% |
| W5 W4 + oblique groups | −0.01% | −0.03% |
| W7 seasonal σ only | +0.04% | **−0.15%** (winner) |
| E1 event-risk σ terms | −0.02% | +0.08% |
| E2 event-risk split candidates | +0.01% | −0.00% |
| E3 E1 + E2 | +0.02% | +0.09% |

Holdout (once): KR W0 0.080315 → W1 0.079869 (−0.56%); US W0 0.051726 → W7 0.051584 (−0.27%). Both pass the 0.1% bar
against W0.

**Noise level (found in the results, not declared in advance).** KR stage-2 W0 is the same model as stage-1 L2; it
lost 123 of 390,753 training rows to the augmentation (rows without the 252-session HAR window), grew a different tree
(4 vs 6 splits) and its holdout CRPS moved by 0.26% (0.080103 → 0.080315). Differences of a few tenths of a percent
are therefore within the model's own instability; the 0.1% tie tolerance / adoption bar is looser than that noise.
Against the served L0 on the same holdout rows: KR L2 + W1 −0.29%, US L2 + W7 −0.13%.

Reading:
- **Longest history, every session**: consistent CV gains in both markets (every long config beats L0, −0.4 to −0.7%,
  on five years of common test rows) and a small directional side effect (KR μ rank IC −0.03 → +0.05; RSMT-Q's
  direction is not served while a stock-selection model exists), but the 24-month holdout does not confirm it.
  Time decay must be relaxed for the history to count: half-life 504 gains little (L1), as the Kish numbers predicted.
- **σ equation**: HAR (KR, 1-year realised vol carries the weight: θ_har252 +0.38) and the seasonal terms (US: quarterly
  peak around the 12th day of Jan / Apr / Jul / Oct = the US earnings season, the declared prior) give 0.1-0.2% in CV;
  with 17-30 years the seasonal terms no longer hurt, unlike the 6-year panel of section 6.
- **Event-risk variables**: no gain in either market. Signs are sensible (implied-minus-realised +0.09 / +0.10,
  geopolitical risk +0.03 / +0.06 on log σ) but the information is already in realised volatility.
- **Time / change / oblique split candidates**: still nothing, even with 364 / 214 dates.
- **Direction**: CV AUC 0.49-0.52 everywhere; no variant is a direction model.

## 8. Adopted 2026-10-01 (user decision): longest history, every session, per-market sigma equation

The user chose option 2 of section 7.3 knowingly: stage 1 (the longer history) did not meet its holdout bar, and the
stage-2 gains are of the size of the model's own instability. The configuration below is therefore a principled
choice ("use all business days of the longest history") backed by CV, not an established out-of-sample improvement;
the forward holdout registered after this change is the check.

**Configuration** (`config/default.yaml`):
- `rsmt.overlap_step: 1` (a training row every session), `rsmt.markets.<M>.half_life_days: 1260`;
- `rsmt.markets.KR.history_start: 1995-09-01` (marcap daily files from 1995; `data.pit.kr.marcap_files` loads the
  earlier years only for a universe that starts before `pit.kr.start_year`), `rsmt.markets.US.history_start:
  2009-08-31` (validated prices / XBRL shares); `pipeline.rsmt_history_start` (None = today − `history_years`);
- sigma equation `rsmt.markets.KR.sigma_terms: [har_5, har_66, har_252]` (section 7.3 KR winner W1),
  `rsmt.markets.US.sigma_terms: [cal_s1, cal_c1, cal_s4, cal_c4]` (US winner W7). `quant.panel` computes the
  columns (HAR = log(realised vol over 5 / 66 / 252 sessions ÷ EWMA vol) per ticker; day-of-year harmonics);
  a training row missing a configured term is dropped and counted, a serving row gets 0 (neutral) and is flagged
  degraded ("변동성 식 입력 누락"). `quant.rsmt.RSMTree(sigma_terms=…)` fits log σ_e = log σ_leaf + η·v + θ'V jointly
  with the leaves (`fit_offsets_multi`, a vector version of `fit_offsets`; one column = the original fit);
  `describe()` prints the equation; the baseline twin never uses it.

**Split-test calibration fix (found while porting; applies to every overlapping panel).** The null-panel regression
test failed after the half-life change, and a 24- / 30-seed check showed the cause: with overlapping labels the
multiplier bootstrap drew one multiplier per h-session block, but labels of NEIGHBOURING blocks overlap too, so the
bootstrap kept only a Bartlett window of h - about 2/3 of the variance of MA(h−1) scores (share kept by window L:
h 0.67, 2h 0.83, 3h 0.89, 4h 0.92). Measured false-split rate of a 5% test on synthetic null panels (half-life 1260):

| rows | 1 horizon (h-blocks, before) | 2 horizons | 3 horizons | 4 horizons |
|---|---|---|---|---|
| every 5 sessions (30 seeds) | 3/30 | 1/30 | 0/30 | 0/30 |
| every session (30 seeds) | 3/30 (5/30 at 10%) | 0/30 (3/30 at 10%) | 0/30 (0/30 at 10%) | 0/30 (0/30 at 10%) |

(and 3/24 at half-life 504, 6/24 at 1260 with h-blocks on another seed set). New setting
`rsmt.boot_block_horizons: 3`: with overlapping rows the split-test bootstrap clusters are runs of 3 consecutive
h-blocks (60 sessions); `min_leaf_dates` and the leaf-γ n_eff still count h-blocks. The served KR V6 / US V1 trees since
2026-09-29 were grown with the anti-conservative test; this fix applies from now on. The rsmt_v2 / rsmt_v3
stock-selection candidates (weekly rows, h-block clusters) have the same issue in principle; they never split and are
not served.

**Fit / serve split** (`pipeline.fit_market`): a daily 30-year fit takes minutes and its CV gate tens of minutes, so
the market model (tree, offsets, sigma equation, CV skill gate, v centering c_v) is fitted on the UNIVERSE only and
cached under code × config × market × language for `rsmt.model_max_age_days` (7). Every run builds a fresh serving
panel from today's prices plus the requested names (`build_panel(..., c_v=<fitted>)`) and serves it with the cached
model, so forecasts use today's features, and the model no longer depends on which other tickers a report requests
(it previously pooled them into the tree). `quantcio rsmt --market KR --no-model-cache` refits and stores the model
(weekly cron / launchd recommended). The report shows "[정보] RSMT-Q 시장 모델: …" with the fit span and data age.

**Research scripts**: `scripts/rsmt_ts.py` / `scripts/rsmt_history.py` ran under the pre-adoption configuration;
re-running them now starts from this configuration (W0 = the adopted model).

**Production fits (2026-10-01, final code, `quantcio rsmt --market <M> --no-model-cache`):**

| market | universe · span · rows | fit time | tree | sigma equation | CV gate (4 folds; 3/3 layouts) |
|---|---|---|---|---|---|
| KR | 339 entities · 1996-05-22 to 2026-09-29 · 423,329 daily rows | 347 s | 6 leaves (volume_z20, dd_250, rsi14), root p 0.040 (Simes), 123 bootstrap clusters of 60 sessions | log σ += −0.209·v −0.013·har_5 +0.427·har_66 +0.315·har_252 | distribution pass: CRPS 0.0611 vs twin 0.0629, gain bound +0.00147, cover80 0.812, VaR95 exceedance 0.049; direction fail |
| US | 146 entities · 2009-08-31 to 2026-09-30 · 251,320 daily rows | 183 s (918 s with a cold price load) | 3 leaves, root p 0.009, 72 clusters | log σ += +0.711·v +0.046·cal_s1 +0.092·cal_c1 +0.081·cal_s4 +… | distribution pass: CRPS 0.0388 vs 0.0390, bound +0.00010, cover80 0.809, VaR95 0.058; direction fail |

A report served from the cached models (4 names, no LLM) took 38 s end to end (KR / US serving panels 9-22 s).
Forward holdout `FTH-20261001` was registered after these fits (it supersedes `FTH-20260930`, which had no sample).

**Head-to-head audit of the adopted model** (`scripts/rsmt_adopted_check.py`, 2026-10-01; `reports/rsmt/adopted_check_<M>.json`):
the configuration served before (OLD: 7 years, every 5th session, half-life 504, no sigma terms, h-block clusters)
and the adopted one (NEW) scored on identical rows - every session of OLD's CV folds before the last 24 months (rows
present in both panels) and every session of the last 24 months. The 24 months were already looked at in sections
6-7 (an out-of-time check, not an untouched sample).

| market | CV CRPS OLD → NEW (layouts 3 / 4 / 5) | holdout CRPS OLD → NEW | holdout cover80 / exc95 OLD → NEW | holdout distribution gate OLD → NEW |
|---|---|---|---|---|
| KR | 0.056367 → 0.056051 **−0.56%** (−0.35 / −0.58 / −0.75%) | 0.080151 → 0.079795 **−0.44%** | 0.816 / 0.025 → 0.809 / 0.025 | fail → **pass** |
| US | 0.043232 → 0.042944 **−0.67%** (−0.44 / −0.73 / −0.82%) | 0.051651 → 0.051583 **−0.13%** | 0.775 / 0.062 → 0.760 / 0.065 | pass → pass |

NEW is better in every CV layout of both markets and on the KR holdout by more than the measured instability (~0.3%);
the US holdout gain is within it, and the US 2024-09+ intervals are a little too narrow in both models (the 2025 tariff
shock), slightly more so in NEW. Side effect: the RSMT-Q mean's rank IC on the KR holdout rose from +0.04 to +0.16
(AUC 0.499 → 0.528); it is not served while the stock-selection model exists.
