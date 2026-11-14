"""Candidate stock-selection models (docs/MODELING.md, "후보 모델").

Every candidate maps a (date, name) row to a SCORE; higher = more likely to do better than the date's other names
after beta (protocol v2 primary label: y_e above the date median). Only the ranking within a date is evaluated.
All hyper-parameter grids are small and pre-declared here; the grid point and the training-weight scheme
(``events.SCHEMES``) are chosen by inner cross-validation inside each outer fold (``modeling.selection``).

Baselines (no inner tuning):
- ``mom_12_1``: the 12-1 month momentum rank.
- ``factor_ic``: the factor-backtest agent's ``ic_weighted`` model (``backtest.model.fit_weights``: trailing
  36-month IC means, a-priori sign constraints, 50% shrink to equal weights) on the same six factors
  (E/P, B/P, traded-value trend, 3M / 6M momentum: sector-neutral ranks; sector momentum).
Trainable:
- ``factor_fm``: Fama-MacBeth - per-date OLS slopes of the excess return on the six factors plus 1W / 1M
  reversal; trailing mean slope shrunk toward ZERO by t^2 / (1 + t^2) (signs learned from the past only).
- ``ridge`` (L2 regression on the cross-sectional ranks), ``logit`` (L2 logistic on 1{y_e > date median}).
- ``hgb`` (HistGradientBoosting regressor), ``lgbm_reg`` / ``lgbm_clf`` (LightGBM regressor / classifier),
  ``lgbm_reg_w`` (the regressor on WEEKLY rows), ``lgbm_rank`` (LambdaRank with one query per date).
  Tree models also see the date-level regime / stress features (``r_*``), so they can learn interactions.
- ``rsmt_v3`` (2026-09-29, docs/RSMT.md): ``quant.rsmt.RSMTRanker`` stacked on the best inner-CV candidate's
  out-of-sample score (weekly rows) with the RSMT-Q v1.1 switches - Simes / BH split test, target
  y_dm / (vol60 * sqrt(h/252)) ranked by the standardised mean, leaf-local stacked-score gamma shrunk toward the
  parent (k = 50 blocks). Pre-declared, one fixed configuration (no tuning); ``STACKED`` = the stacked candidates
  (fitted after each context's base candidates, ``modeling.selection``). ``rsmt_v2`` was removed in v3 (it never
  split, so it was a copy of its stack base - docs/MODELING.md 15).
Candidates v2 (pre-declared, docs/MODELING.md 0.3):
- ``logit_w``: the same L2 logistic on WEEKLY rows. ``ridge_rg``: ridge on the per-date rank-gauss of y_e.
Candidates v3 (pre-declared, docs/MODELING.md 16.4):
- ``enet_logit`` (elastic-net logistic), ``rf`` / ``et`` (random forest / extra trees on the per-date rank-gauss
  target), ``xgb_reg`` / ``xgb_clf`` / ``xgb_clf_w`` / ``xgb_rank`` (XGBoost; the ranker uses per-date groups and
  per-group weights), ``cat_clf`` / ``cat_rank`` (CatBoost; YetiRank with per-date groups), ``mlp`` (a small
  neural net on the rank-gauss target).
- Diagnostics (``diagnostic=True``, never selected, never in a composite / stack base): ``logit_w_core`` and
  ``lgbm_clf_core`` - the same models on the v2 (pre-16.3) inputs only, to measure what the alternative data and
  the macro exposures add.
Composites (scores built from other candidates' scores, ``composite_rule``; no tuning of their own):
- ``ensemble`` / ``ensemble5``: rank mean of the top-3 / top-5 eligible candidates by inner-CV score;
- ``ensemble_w``: weighted rank mean of every eligible candidate, weight = max(inner score - no-skill value, 0)
  (protocol v4: inner top-quintile net excess; v3: inner AUC - 0.5);
- ``family_blend``: rank mean of the best eligible candidate (inner CV) of each family group (linear, boosting,
  bagging, neural net);
- ``blend``: the FIXED rank mean of ``BLEND_MEMBERS`` (logit + lgbm_clf, each tuned by its own inner CV).

Regression targets are per-date demeaned excess returns y_dm (the market move of the date removed); scheme
``winsor`` winsorises y_e per date first. Classifiers use 1{y_e > date median}; LambdaRank uses date quintiles;
``ridge_rg`` the per-date rank-gauss of y_e.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantcio.modeling.events import EventCalendar, scheme_weights, winsorised
from quantcio.modeling.features import rank_gauss
from quantcio.modeling.metrics import PRIMARY_NULL

log = logging.getLogger(__name__)

FACTOR_IC_MAP = {  # backtest factor name -> modeling column
    "value_ep": "sx_ep",
    "value_bp": "sx_bp",
    "volume_trend": "sx_volume_trend",
    "mom_3m": "sx_mom_3m",
    "mom_6m": "sx_mom_6m",
    "sector_mom_3m": "x_sector_mom_3m",
}
FM_FACTORS = [*FACTOR_IC_MAP.values(), "sx_ret_1m", "x_ret_1w"]
RSMT_XS = ["x_ret_1m", "x_bp", "x_vol60", "x_log_mcap"]
RSMT_REGIME = ["r_mkt_dd", "r_mkt_vol_ratio", "r_xs_disp", "r_vix_z"]
STACKED = ("rsmt_v3", "rsmt_v3_bo")  # stacked on a context's best inner-CV candidate (never a base candidate)
BLEND_MEMBERS = ("logit", "lgbm_clf")  # candidates v2: fixed linear + tree rank blend
FAMILY_GROUPS = {  # family_blend: one member per group (candidates v3)
    "linear": ("factor", "linear"),
    "boost": ("tree", "rank"),
    "bagging": ("bagging",),
    "nn": ("nn",),
}


@dataclass
class FitContext:
    xs_cols: list[str]
    regime_cols: list[str]
    events: EventCalendar | None
    cutoff: pd.Timestamp | None  # training cutoff (the test start): only events ended before it are "known"
    calendar: pd.DatetimeIndex
    horizon: int = 20
    seed: int = 0
    threads: int = 8
    cache_dir: Path | None = None
    backtest_shrink: float = 0.5
    backtest_lookback: int = 36
    core_xs_cols: list[str] | None = None  # the v2 inputs (diagnostic candidates); None = same as xs_cols
    core_regime_cols: list[str] | None = None


@dataclass
class Fitted:
    name: str
    params: dict[str, Any]
    scheme: str
    model: Any = None
    features: list[str] = field(default_factory=list)
    kind: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ inference
    def _matrix(self, df: pd.DataFrame, fill: bool) -> np.ndarray:
        x = df[self.features].to_numpy(dtype=np.float64)
        return np.nan_to_num(x, nan=0.0) if fill else x

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.array([])
        k = self.kind
        if k == "column":
            return df[self.features[0]].astype(float).fillna(0.0).to_numpy()
        if k == "weights":
            w = np.array([self.extra["weights"].get(c, 0.0) for c in self.features])
            return self._matrix(df, True) @ w
        if k == "ridge":
            return self.model.predict(self._matrix(df, True))
        if k == "logit":
            return self.model.decision_function(self._matrix(df, True))
        if k in ("hgb", "lgbm", "sk_tree", "xgb", "cat"):
            return np.asarray(self.model.predict(self._matrix(df, False)), dtype=float).ravel()
        if k in ("lgbm_clf", "xgb_clf", "cat_clf"):
            return np.asarray(self.model.predict_proba(self._matrix(df, False)))[:, 1]
        if k == "mlp":
            return np.asarray(self.model.predict(self._matrix(df, True)), dtype=float).ravel()
        if k == "rsmt":
            out = self.model.predict(df)
            return out.get("score", out["mu_e"])
        raise ValueError(f"unknown fitted kind {k!r}")

    def importance(self) -> dict[str, float]:
        k = self.kind
        if k == "weights":
            return {c: abs(float(self.extra["weights"].get(c, 0.0))) for c in self.features}
        if k in ("ridge", "logit"):
            coef = np.ravel(self.model.coef_)
            return {c: abs(float(v)) for c, v in zip(self.features, coef)}
        if k in ("lgbm", "lgbm_clf"):
            imp = self.model.booster_.feature_importance(importance_type="gain")
            return {c: float(v) for c, v in zip(self.features, imp)}
        if k in ("sk_tree", "xgb", "xgb_clf"):
            imp = np.asarray(self.model.feature_importances_, dtype=float)
            return {c: float(v) for c, v in zip(self.features, imp)}
        if k in ("cat", "cat_clf"):
            try:  # rankers default to LossFunctionChange (needs data): use the data-free prediction change
                imp = np.asarray(self.model.get_feature_importance(type="PredictionValuesChange"), dtype=float)
            except Exception:  # noqa: BLE001 - importance is display only
                return {}
            return {c: float(v) for c, v in zip(self.features, imp)}
        if k == "mlp":
            w = np.abs(np.asarray(self.model.coefs_[0], dtype=float)).sum(axis=1)  # first-layer weight mass
            return {c: float(v) for c, v in zip(self.features, w)}
        if k == "rsmt":
            return self.model.importance()
        if k == "column":
            return {self.features[0]: 1.0}
        return {}

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Additive per-feature contributions to the score (linear: coef * x; LightGBM: SHAP); None otherwise."""
        k = self.kind
        if len(df) == 0:
            return None
        if k == "weights":
            w = np.array([self.extra["weights"].get(c, 0.0) for c in self.features])
            return pd.DataFrame(self._matrix(df, True) * w, index=df.index, columns=self.features)
        if k in ("ridge", "logit"):
            coef = np.ravel(self.model.coef_)
            return pd.DataFrame(self._matrix(df, True) * coef, index=df.index, columns=self.features)
        if k in ("lgbm", "lgbm_clf"):
            c = self.model.booster_.predict(self._matrix(df, False), pred_contrib=True)
            return pd.DataFrame(np.asarray(c)[:, : len(self.features)], index=df.index, columns=self.features)
        if k in ("xgb", "xgb_clf"):
            import xgboost as xgb

            dm = xgb.DMatrix(self._matrix(df, False), feature_names=None)
            c = self.model.get_booster().predict(dm, pred_contribs=True)
            return pd.DataFrame(np.asarray(c)[:, : len(self.features)], index=df.index, columns=self.features)
        if k == "column":
            return pd.DataFrame({self.features[0]: df[self.features[0]].fillna(0.0).to_numpy()}, index=df.index)
        return None


# ------------------------------------------------------------------------------------------------ helpers


def row_weights(train: pd.DataFrame, scheme: str, ctx: FitContext) -> np.ndarray:
    """Per-date equal weights x the scheme multiplier, normalised to mean 1."""
    w = train["w_date"].fillna(0.0).to_numpy(dtype=float) * scheme_weights(train, scheme, ctx.events, ctx.cutoff)
    m = w.mean()
    return w / m if m > 0 else np.ones(len(train))


def reg_target(train: pd.DataFrame, scheme: str) -> np.ndarray:
    """Per-date demeaned excess return (winsorised per date first for scheme 'winsor')."""
    if scheme == "winsor":
        y = winsorised(train["y_e"], train["date"])
        return (y - y.groupby(train["date"]).transform("mean")).to_numpy(dtype=float)
    return train["y_dm"].to_numpy(dtype=float)


def per_date_ic_table(train: pd.DataFrame, cols: list[str], label: str = "y_e") -> pd.DataFrame:
    """Per-date Spearman IC of each column vs ``label`` (rows: dates)."""
    from quantcio.modeling.metrics import per_date_ic

    out = {c: per_date_ic(train.rename(columns={c: "_s"}), "_s", label) for c in cols}
    return pd.DataFrame(out).sort_index()


def fm_slopes(train: pd.DataFrame, cols: list[str], y: np.ndarray) -> pd.DataFrame:
    """Per-date cross-sectional OLS slopes of y on cols (NaN -> 0, intercept included)."""
    d = train[["date", *cols]].copy()
    d["_y"] = y
    out = {}
    for t, g in d.groupby("date", sort=True):
        g = g[np.isfinite(g["_y"])]
        if len(g) < len(cols) + 10:
            continue
        X = np.column_stack([np.ones(len(g)), np.nan_to_num(g[cols].to_numpy(dtype=float), nan=0.0)])
        beta, *_ = np.linalg.lstsq(X, g["_y"].to_numpy(dtype=float), rcond=None)
        out[t] = beta[1:]
    return pd.DataFrame.from_dict(out, orient="index", columns=cols).sort_index()


def _frac_leaf(n: int, frac: float) -> int:
    return max(50, int(frac * n))


# ------------------------------------------------------------------------------------------------ candidates


@dataclass
class Candidate:
    name: str
    family: str  # baseline | factor | linear | tree | rank | rsmt
    label_ko: str
    label_en: str
    rows: str = "M"  # training rows: "M" month-end, "W" weekly grid
    grid: list[dict[str, Any]] = field(default_factory=lambda: [{}])
    schemes: tuple[str, ...] = ("none",)
    tuned: bool = True  # False: baseline, no inner CV
    diagnostic: bool = False  # True: reported only - never selected, never a composite member / stack base
    feature_set: str = "all"  # "all" | "core" (the v2 inputs: docs/MODELING.md 16.4 diagnostics)
    fitter: str = ""  # name of the fitter in _FITTERS when it differs from ``name``
    # Bayesian (TPE) tuning (docs/MODELING.md 82): {param: (kind, *spec)} with kind int / int_log / float /
    # float_log / cat; "scheme" (cat) picks the training-weight scheme. Empty = the grid + scheme search above.
    search: dict[str, tuple] = field(default_factory=dict)
    n_trials: int = 0
    # the 1-SE rule's "simplest" order: (param, +1 larger = simpler | -1 smaller = simpler), compared in turn
    simplicity: tuple[tuple[str, int], ...] = ()

    def fit(self, train: pd.DataFrame, params: dict[str, Any], scheme: str, ctx: FitContext) -> Fitted:
        if self.feature_set == "core" and ctx.core_xs_cols is not None:
            ctx = replace(
                ctx,
                xs_cols=[c for c in ctx.core_xs_cols if c in train],
                regime_cols=[c for c in (ctx.core_regime_cols or ctx.regime_cols) if c in train],
            )
        f = _FITTERS[self.fitter or self.name](self, train, params, scheme, ctx)
        f.name = self.name
        return f


def _fit_mom(c: Candidate, train, params, scheme, ctx) -> Fitted:
    return Fitted(c.name, params, scheme, None, ["x_mom_12_1"], "column")


def _fit_factor_ic(c: Candidate, train, params, scheme, ctx) -> Fitted:
    from quantcio.backtest.model import fit_weights

    me = train[train["is_me"]]
    cols = list(FACTOR_IC_MAP.values())
    ic = per_date_ic_table(me, cols).iloc[-int(ctx.backtest_lookback) :]
    mean = ic.mean().fillna(0.0)
    mean.index = list(FACTOR_IC_MAP)
    w = fit_weights(mean, ctx.backtest_shrink, list(FACTOR_IC_MAP))
    weights = {FACTOR_IC_MAP[f]: float(v) for f, v in w.items()}
    return Fitted(c.name, params, scheme, None, cols, "weights", {"weights": weights, "n_dates": len(ic)})


def _fit_factor_fm(c: Candidate, train, params, scheme, ctx) -> Fitted:
    me = train[train["is_me"]]
    y = reg_target(me, scheme)
    slopes = fm_slopes(me, FM_FACTORS, y)
    lb = params.get("lookback")
    if lb:
        slopes = slopes.iloc[-int(lb) :]
    n = len(slopes)
    mean = slopes.mean()
    sd = slopes.std(ddof=1)
    t = mean / (sd / np.sqrt(max(n, 1)))
    shrink = (t * t / (1.0 + t * t)).fillna(0.0)
    weights = {k: float(v) for k, v in (mean.fillna(0.0) * shrink).items()}
    return Fitted(
        c.name, params, scheme, None, list(FM_FACTORS), "weights", {"weights": weights, "n_dates": n, "t": t.to_dict()}
    )


def _fit_ridge(c: Candidate, train, params, scheme, ctx) -> Fitted:
    from sklearn.linear_model import Ridge

    X = np.nan_to_num(train[ctx.xs_cols].to_numpy(dtype=np.float64), nan=0.0)
    y = reg_target(train, scheme)
    ok = np.isfinite(y)
    m = Ridge(alpha=float(params["a"]) * ok.sum()).fit(X[ok], y[ok], sample_weight=row_weights(train, scheme, ctx)[ok])
    return Fitted(c.name, params, scheme, m, list(ctx.xs_cols), "ridge")


def _fit_ridge_rg(c: Candidate, train, params, scheme, ctx) -> Fitted:
    """Ridge on the per-date rank-gauss of y_e (a rank target: per-date winsorisation cannot change it)."""
    from sklearn.linear_model import Ridge

    X = np.nan_to_num(train[ctx.xs_cols].to_numpy(dtype=np.float64), nan=0.0)
    y = rank_gauss(train["y_e"], train["date"]).to_numpy(dtype=float)
    ok = np.isfinite(y)
    m = Ridge(alpha=float(params["a"]) * ok.sum()).fit(X[ok], y[ok], sample_weight=row_weights(train, scheme, ctx)[ok])
    return Fitted(c.name, params, scheme, m, list(ctx.xs_cols), "ridge")


def _fit_logit(c: Candidate, train, params, scheme, ctx) -> Fitted:
    from sklearn.linear_model import LogisticRegression

    X = np.nan_to_num(train[ctx.xs_cols].to_numpy(dtype=np.float64), nan=0.0)
    y = train["y_cls"].to_numpy(dtype=float)
    ok = np.isfinite(y)
    m = LogisticRegression(C=float(params["C"]), max_iter=1000).fit(
        X[ok], y[ok].astype(int), sample_weight=row_weights(train, scheme, ctx)[ok]
    )
    return Fitted(c.name, params, scheme, m, list(ctx.xs_cols), "logit")


def _tree_cols(ctx: FitContext, train: pd.DataFrame) -> list[str]:
    """xs + regime columns that have at least one finite training value (an all-NaN column - e.g. a macro z
    before its history starts - carries no information and breaks some binners)."""
    cols = [*ctx.xs_cols, *ctx.regime_cols]
    return [c for c in cols if train[c].notna().any()]


def _fit_hgb(c: Candidate, train, params, scheme, ctx) -> Fitted:
    from sklearn.ensemble import HistGradientBoostingRegressor

    cols = _tree_cols(ctx, train)
    X = train[cols].to_numpy(dtype=np.float64)
    y = reg_target(train, scheme)
    ok = np.isfinite(y)
    m = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=15,
        max_bins=63,
        min_samples_leaf=_frac_leaf(int(ok.sum()), float(params["leaf_frac"])),
        l2_regularization=1.0,
        early_stopping=False,
        random_state=ctx.seed,
    ).fit(X[ok], y[ok], sample_weight=row_weights(train, scheme, ctx)[ok])
    return Fitted(c.name, params, scheme, m, cols, "hgb")


def _lgbm_common(params: dict[str, Any], n: int, ctx: FitContext, gain_scale: float = 1.0) -> dict[str, Any]:
    """LightGBM keyword arguments; the optional regularisation keys of the *_bo candidates (docs 82) default to the
    values every other candidate always used. ``min_gain`` is relative: x ``gain_scale`` (the target variance)."""
    return {
        "n_estimators": int(params.get("n_estimators", 300)),
        "learning_rate": float(params.get("lr", 0.03)),
        "num_leaves": int(params.get("leaves", 15)),
        "min_child_samples": _frac_leaf(n, float(params.get("leaf_frac", 0.01))),
        "subsample": float(params.get("bagging", 0.8)),
        "subsample_freq": 1,
        "colsample_bytree": float(params.get("feature_frac", 0.8)),
        "reg_lambda": float(params.get("l2", 1.0)),
        "reg_alpha": float(params.get("l1", 0.0)),
        "max_depth": int(params.get("max_depth", -1)),
        "min_split_gain": float(params.get("min_gain", 0.0)) * float(gain_scale),
        "n_jobs": int(ctx.threads),
        "random_state": int(ctx.seed),
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }


def _fit_lgbm_reg(c: Candidate, train, params, scheme, ctx) -> Fitted:
    import lightgbm as lgb

    cols = _tree_cols(ctx, train)
    y = reg_target(train, scheme)
    ok = np.isfinite(y)
    X = train[cols].to_numpy(dtype=np.float64)[ok]
    kw = _lgbm_common(params, int(ok.sum()), ctx, float(np.var(y[ok])) if ok.any() else 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = lgb.LGBMRegressor(objective="regression", **kw).fit(
            X, y[ok], sample_weight=row_weights(train, scheme, ctx)[ok]
        )
    return Fitted(c.name, params, scheme, m, cols, "lgbm")


def _fit_lgbm_clf(c: Candidate, train, params, scheme, ctx) -> Fitted:
    import lightgbm as lgb

    cols = _tree_cols(ctx, train)
    y = train["y_cls"].to_numpy(dtype=float)
    ok = np.isfinite(y)
    X = train[cols].to_numpy(dtype=np.float64)[ok]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = lgb.LGBMClassifier(objective="binary", **_lgbm_common(params, int(ok.sum()), ctx)).fit(
            X, y[ok].astype(int), sample_weight=row_weights(train, scheme, ctx)[ok]
        )
    return Fitted(c.name, params, scheme, m, cols, "lgbm_clf")


def _fit_lgbm_rank(c: Candidate, train, params, scheme, ctx) -> Fitted:
    import lightgbm as lgb

    cols = _tree_cols(ctx, train)
    d = train[np.isfinite(train["y_q"].to_numpy(dtype=float))].sort_values("date", kind="stable")
    groups = d.groupby("date", sort=True).size().to_numpy()
    kw = _lgbm_common(params, len(d), ctx, float(np.var(d["y_q"].to_numpy(dtype=float))) if len(d) else 1.0)
    kw["min_child_samples"] = _frac_leaf(len(d), float(params.get("leaf_frac", 0.01)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = lgb.LGBMRanker(objective="lambdarank", lambdarank_truncation_level=10_000, **kw).fit(
            d[cols].to_numpy(dtype=np.float64),
            d["y_q"].to_numpy(dtype=int),
            group=groups,
            sample_weight=row_weights(d, scheme, ctx),
        )
    return Fitted(c.name, params, scheme, m, cols, "lgbm")


def _fit_rsmt(c: Candidate, train, params, scheme, ctx) -> Fitted:
    """RSMT-Q v2 / v3 (``STACKED``): the grid point carries the v3 switches (absent = v2 behaviour)."""
    from quantcio.quant.rsmt import RSMTRanker

    d = train[np.isfinite(train["stack_z"].to_numpy(dtype=float)) & np.isfinite(train["y_dm"].to_numpy(dtype=float))]
    if d.empty:
        raise ValueError(f"{c.name}: no training row with a stacked out-of-sample score")
    xs = [x for x in RSMT_XS if x in d]
    rg = [x for x in RSMT_REGIME if x in d]
    ranker = RSMTRanker(
        xs,
        rg,
        alpha=float(params.get("alpha", 0.05)),
        max_depth=int(params.get("max_depth", 3)),
        min_leaf_blocks=int(params.get("min_leaf_blocks", 12)),
        mu_shrink_k=float(params.get("mu_shrink_k", 500.0)),
        half_life_days=float(params.get("half_life_days", 756)),
        block_sessions=int(ctx.horizon),
        cache_dir=ctx.cache_dir,
        seed=ctx.seed,
        multiple_testing=str(params.get("multiple_testing", "bonferroni")),
        pca_components=int(params.get("pca_components", 2)),
        target_scale=str(params.get("target_scale", "none")),
        gamma_mode=str(params.get("gamma_mode", "global")),
        gamma_shrink_k=float(params.get("gamma_shrink_k", 50.0)),
        horizon=int(ctx.horizon),
    ).fit(d, ctx.calendar, y_col="y_dm")
    return Fitted(c.name, params, scheme, ranker, [*xs, *rg, "stack_z"], "rsmt", {"n_train": len(d)})


def _fit_enet_logit(c: Candidate, train, params, scheme, ctx) -> Fitted:
    from sklearn.linear_model import LogisticRegression

    X = np.nan_to_num(train[ctx.xs_cols].to_numpy(dtype=np.float64), nan=0.0)
    y = train["y_cls"].to_numpy(dtype=float)
    ok = np.isfinite(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # saga ConvergenceWarning at max_iter: the iterate is still usable
        m = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=float(params.get("l1_ratio", 0.5)),
            C=float(params["C"]),
            max_iter=300,
            tol=1e-3,
        ).fit(X[ok], y[ok].astype(int), sample_weight=row_weights(train, scheme, ctx)[ok])
    return Fitted(c.name, params, scheme, m, list(ctx.xs_cols), "logit")


def _rg_target(train: pd.DataFrame, scheme: str) -> np.ndarray:
    """Per-date rank-gauss of the excess return (winsorising cannot change a rank target)."""
    return rank_gauss(train["y_e"], train["date"]).to_numpy(dtype=float)


def _fit_forest(c: Candidate, train, params, scheme, ctx) -> Fitted:
    """Random forest (``rf``) / extra trees (``et``) on the per-date rank-gauss target; NaN routed natively."""
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

    cols = _tree_cols(ctx, train)
    y = _rg_target(train, scheme)
    ok = np.isfinite(y)
    cls = ExtraTreesRegressor if params.get("kind") == "et" else RandomForestRegressor
    boot = bool(params.get("bootstrap", cls is RandomForestRegressor))  # et_bo may bootstrap (docs 82)
    depth = params.get("max_depth")
    m = cls(
        n_estimators=int(params.get("n_estimators", 300)),
        max_features=float(params.get("max_features", 0.33)),
        min_samples_leaf=_frac_leaf(int(ok.sum()), float(params["leaf_frac"])),
        max_depth=int(depth) if depth else None,
        max_samples=float(params.get("max_samples", 0.5)) if boot else None,
        bootstrap=boot,
        n_jobs=int(ctx.threads),
        random_state=int(ctx.seed),
    ).fit(train[cols].to_numpy(dtype=np.float64)[ok], y[ok], sample_weight=row_weights(train, scheme, ctx)[ok])
    return Fitted(c.name, params, scheme, m, cols, "sk_tree")


def _xgb_common(params: dict[str, Any], n: int, ctx: FitContext, hess: float = 1.0) -> dict[str, Any]:
    return {
        "n_estimators": int(params.get("n_estimators", 300)),
        "learning_rate": float(params.get("lr", 0.03)),
        "max_depth": int(params.get("depth", 3)),
        "min_child_weight": hess * _frac_leaf(n, float(params.get("leaf_frac", 0.01))),
        "subsample": float(params.get("bagging", 0.8)),  # the *_bo keys (docs 83) default to the old fixed values
        "colsample_bytree": float(params.get("feature_frac", 0.8)),
        "reg_lambda": float(params.get("l2", 1.0)),
        "reg_alpha": float(params.get("l1", 0.0)),
        "gamma": float(params.get("min_gain", 0.0)),
        "tree_method": "hist",
        "max_bin": 64,
        "n_jobs": int(ctx.threads),
        "random_state": int(ctx.seed),
        "verbosity": 0,
    }


def _fit_xgb_reg(c: Candidate, train, params, scheme, ctx) -> Fitted:
    import xgboost as xgb

    cols = _tree_cols(ctx, train)
    y = reg_target(train, scheme)
    ok = np.isfinite(y)
    m = xgb.XGBRegressor(objective="reg:squarederror", **_xgb_common(params, int(ok.sum()), ctx)).fit(
        train[cols].to_numpy(dtype=np.float64)[ok], y[ok], sample_weight=row_weights(train, scheme, ctx)[ok]
    )
    return Fitted(c.name, params, scheme, m, cols, "xgb")


def _fit_xgb_clf(c: Candidate, train, params, scheme, ctx) -> Fitted:
    import xgboost as xgb

    cols = _tree_cols(ctx, train)
    y = train["y_cls"].to_numpy(dtype=float)
    ok = np.isfinite(y)
    m = xgb.XGBClassifier(objective="binary:logistic", **_xgb_common(params, int(ok.sum()), ctx, 0.25)).fit(
        train[cols].to_numpy(dtype=np.float64)[ok], y[ok].astype(int), sample_weight=row_weights(train, scheme, ctx)[ok]
    )
    return Fitted(c.name, params, scheme, m, cols, "xgb_clf")


def _date_groups(d: pd.DataFrame, scheme: str, ctx: FitContext) -> tuple[np.ndarray, np.ndarray]:
    """(per-row query id, per-group weight = mean row weight of the date) of rows sorted by date."""
    qid = pd.factorize(d["date"], sort=True)[0]
    w = pd.Series(row_weights(d, scheme, ctx)).groupby(qid).mean().to_numpy(dtype=float)
    return qid, w


def _fit_xgb_rank(c: Candidate, train, params, scheme, ctx) -> Fitted:
    import xgboost as xgb

    cols = _tree_cols(ctx, train)
    d = train[np.isfinite(train["y_q"].to_numpy(dtype=float))].sort_values("date", kind="stable")
    qid, gw = _date_groups(d, scheme, ctx)
    kw = _xgb_common(params, len(d), ctx)
    kw["min_child_weight"] = 1.0  # pairwise hessians are not row counts
    m = xgb.XGBRanker(objective="rank:pairwise", **kw).fit(
        d[cols].to_numpy(dtype=np.float64), d["y_q"].to_numpy(dtype=int), qid=qid, sample_weight=gw
    )
    return Fitted(c.name, params, scheme, m, cols, "xgb")


def _cat_common(params: dict[str, Any], ctx: FitContext) -> dict[str, Any]:
    return {
        "iterations": int(params.get("iterations", 400)),
        "learning_rate": float(params.get("lr", 0.05)),
        "depth": int(params.get("depth", 6)),
        "l2_leaf_reg": float(params.get("l2", 3.0)),  # docs 83: cat_clf_bo tunes it (default = the old value)
        "rsm": float(params.get("feature_frac", 1.0)),
        "random_seed": int(ctx.seed),
        "thread_count": int(ctx.threads),
        "verbose": False,
        "allow_writing_files": False,
    }


def _fit_cat_clf(c: Candidate, train, params, scheme, ctx) -> Fitted:
    from catboost import CatBoostClassifier

    cols = _tree_cols(ctx, train)
    y = train["y_cls"].to_numpy(dtype=float)
    ok = np.isfinite(y)
    m = CatBoostClassifier(loss_function="Logloss", **_cat_common(params, ctx)).fit(
        train[cols].to_numpy(dtype=np.float64)[ok], y[ok].astype(int), sample_weight=row_weights(train, scheme, ctx)[ok]
    )
    return Fitted(c.name, params, scheme, m, cols, "cat_clf")


def _fit_cat_rank(c: Candidate, train, params, scheme, ctx) -> Fitted:
    from catboost import CatBoostRanker, Pool

    cols = _tree_cols(ctx, train)
    d = train[np.isfinite(train["y_q"].to_numpy(dtype=float))].sort_values("date", kind="stable")
    qid, _ = _date_groups(d, scheme, ctx)
    pool = Pool(d[cols].to_numpy(dtype=np.float64), d["y_q"].to_numpy(dtype=float), group_id=qid)
    m = CatBoostRanker(loss_function="YetiRank", **_cat_common(params, ctx)).fit(pool)
    return Fitted(c.name, params, scheme, m, cols, "cat")


def _fit_mlp(c: Candidate, train, params, scheme, ctx) -> Fitted:
    from sklearn.neural_network import MLPRegressor

    X = np.nan_to_num(train[ctx.xs_cols].to_numpy(dtype=np.float64), nan=0.0)
    y = _rg_target(train, scheme)
    ok = np.isfinite(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MLPRegressor(
            hidden_layer_sizes=tuple(int(x) for x in str(params.get("hidden", "64-32")).split("-")),
            alpha=float(params["alpha"]),
            learning_rate_init=float(params.get("lr_init", 1e-3)),
            batch_size=512,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=int(ctx.seed),
        ).fit(X[ok], y[ok], sample_weight=row_weights(train, scheme, ctx)[ok])
    return Fitted(c.name, params, scheme, m, list(ctx.xs_cols), "mlp")


_FITTERS = {
    "mom_12_1": _fit_mom,
    "factor_ic": _fit_factor_ic,
    "factor_fm": _fit_factor_fm,
    "ridge": _fit_ridge,
    "logit": _fit_logit,
    "logit_w": _fit_logit,
    "ridge_rg": _fit_ridge_rg,
    "hgb": _fit_hgb,
    "lgbm_reg": _fit_lgbm_reg,
    "lgbm_reg_w": _fit_lgbm_reg,
    "lgbm_clf": _fit_lgbm_clf,
    "lgbm_rank": _fit_lgbm_rank,
    "rsmt_v3": _fit_rsmt,
    "enet_logit": _fit_enet_logit,
    "rf": _fit_forest,
    "et": _fit_forest,
    "xgb_reg": _fit_xgb_reg,
    "xgb_clf": _fit_xgb_clf,
    "xgb_clf_w": _fit_xgb_clf,
    "xgb_rank": _fit_xgb_rank,
    "cat_clf": _fit_cat_clf,
    "cat_rank": _fit_cat_rank,
    "mlp": _fit_mlp,
}

_REG_SCHEMES = ("none", "event_down", "winsor", "magnitude")
# rank / class targets: per-date winsorisation cannot change them; magnitude weights can (protocol v4, docs 79)
_RANK_SCHEMES = ("none", "event_down", "magnitude")


BO_TRIALS = 40  # TPE trials per candidate and context (docs/MODELING.md 82)
_LGBM_SPACE: dict[str, tuple] = {
    "leaves": ("int_log", 7, 63),
    "leaf_frac": ("float_log", 0.002, 0.05),
    "feature_frac": ("float", 0.3, 1.0),
    "bagging": ("float", 0.5, 1.0),
    "l1": ("float_log", 1e-3, 10.0),
    "l2": ("float_log", 1e-3, 30.0),
    "max_depth": ("int", 3, 8),
    "min_gain": ("float", 0.0, 0.1),
    "lr": ("float_log", 0.01, 0.1),
    "n_estimators": ("int", 100, 600),
}
_LGBM_SIMPLE = (("leaf_frac", +1), ("leaves", -1))
_ET_SPACE: dict[str, tuple] = {
    "kind": ("cat", ("et",)),
    "max_features": ("float", 0.1, 0.7),
    "leaf_frac": ("float_log", 0.001, 0.05),
    "max_depth": ("int", 4, 24),
    "bootstrap": ("cat", (False, True)),
    "max_samples": ("float", 0.3, 0.9),
}


_SCHEMES_CLS = ("cat", ("none", "magnitude"))
_SCHEMES_REG = ("cat", ("none", "magnitude", "winsor"))
_XGB_SPACE: dict[str, tuple] = {
    "depth": ("int", 2, 6),
    "leaf_frac": ("float_log", 0.002, 0.05),
    "lr": ("float_log", 0.01, 0.1),
    "n_estimators": ("int", 100, 600),
    "bagging": ("float", 0.5, 1.0),
    "feature_frac": ("float", 0.3, 1.0),
    "l1": ("float_log", 1e-3, 10.0),
    "l2": ("float_log", 1e-3, 30.0),
    "min_gain": ("float", 0.0, 0.5),
    "scheme": _SCHEMES_CLS,
}
_CAT_SPACE: dict[str, tuple] = {
    "depth": ("int", 3, 7),
    "l2": ("float_log", 1.0, 30.0),
    "lr": ("float_log", 0.02, 0.15),
    "iterations": ("int", 200, 800),
    "feature_frac": ("float", 0.3, 1.0),
    "scheme": _SCHEMES_CLS,
}
_RSMT_V3 = {"multiple_testing": "simes", "target_scale": "vol", "gamma_mode": "leaf"}  # the v3 switches (fixed)
_RSMT_SPACE: dict[str, tuple] = {
    "alpha": ("float_log", 0.001, 0.2),
    "max_depth": ("int", 1, 4),
    "min_leaf_blocks": ("int", 6, 36),
    "mu_shrink_k": ("float_log", 50.0, 5000.0),
    "gamma_shrink_k": ("float_log", 5.0, 500.0),
    "half_life_days": ("int", 252, 2520),
    **{k: ("cat", (v,)) for k, v in _RSMT_V3.items()},
}


def default_candidates() -> list[Candidate]:
    """The pre-declared candidate set (order = report order)."""
    return [
        Candidate("mom_12_1", "baseline", "기준: 12-1 모멘텀 순위", "baseline: 12-1 momentum rank", tuned=False),
        Candidate(
            "factor_ic",
            "baseline",
            "기준: 백테스트 IC 가중 팩터 모델(사전 부호 제약)",
            "baseline: backtest IC-weighted factor model (a-priori signs)",
            tuned=False,
        ),
        Candidate(
            "factor_fm",
            "factor",
            "팩터 Fama–MacBeth(과거 추정 부호, 0으로 축소)",
            "factor Fama-MacBeth (signs learned, shrunk to 0)",
            grid=[{"lookback": 36}, {"lookback": 60}, {"lookback": 0}],
            schemes=_REG_SCHEMES,
            simplicity=(("lookback", +1),),  # 0 = the whole history, the longest window (selection._simple_value)
        ),
        Candidate(
            "ridge",
            "linear",
            "릿지 회귀(횡단면 순위)",
            "ridge on cross-sectional ranks",
            grid=[{"a": 0.01}, {"a": 0.1}, {"a": 1.0}],
            schemes=_REG_SCHEMES,
            simplicity=(("a", +1),),
        ),
        Candidate(
            "logit",
            "linear",
            "L2 로지스틱(날짜 중앙값 초과)",
            "L2 logistic (above date median)",
            grid=[{"C": 0.0005}, {"C": 0.005}, {"C": 0.05}],
            schemes=_RANK_SCHEMES,
            simplicity=(("C", -1),),
        ),
        Candidate(
            "logit_w",
            "linear",
            "L2 로지스틱(날짜 중앙값 초과, 주간 행)",
            "L2 logistic (above date median, weekly rows)",
            rows="W",
            grid=[{"C": 0.0005}, {"C": 0.005}, {"C": 0.05}],
            schemes=_RANK_SCHEMES,
            simplicity=(("C", -1),),
        ),
        Candidate(
            "ridge_rg",
            "linear",
            "릿지(날짜별 rank-gauss 초과수익 목표)",
            "ridge on the per-date rank-gauss excess return",
            grid=[{"a": 0.01}, {"a": 0.1}, {"a": 1.0}],
            schemes=_RANK_SCHEMES,
            simplicity=(("a", +1),),
        ),
        Candidate(
            "hgb",
            "tree",
            "HistGradientBoosting 회귀",
            "HistGradientBoosting regressor",
            grid=[{"leaf_frac": 0.005}, {"leaf_frac": 0.02}],
            schemes=_REG_SCHEMES,
            simplicity=(("leaf_frac", +1),),
        ),
        Candidate(
            "lgbm_reg",
            "tree",
            "LightGBM 회귀",
            "LightGBM regressor",
            grid=[
                {"leaves": 15, "leaf_frac": 0.005},
                {"leaves": 15, "leaf_frac": 0.02},
                {"leaves": 31, "leaf_frac": 0.005},
                {"leaves": 31, "leaf_frac": 0.02},
            ],
            schemes=_REG_SCHEMES,
            simplicity=_LGBM_SIMPLE,
        ),
        Candidate(
            "lgbm_reg_w",
            "tree",
            "LightGBM 회귀(주간 행)",
            "LightGBM regressor (weekly rows)",
            rows="W",
            grid=[{"leaves": 15, "leaf_frac": 0.01}],
            schemes=_REG_SCHEMES,
            simplicity=_LGBM_SIMPLE,
        ),
        Candidate(
            "lgbm_clf",
            "tree",
            "LightGBM 분류(날짜 중앙값 초과)",
            "LightGBM classifier (above date median)",
            grid=[{"leaves": 15, "leaf_frac": 0.01}, {"leaves": 31, "leaf_frac": 0.01}],
            schemes=_RANK_SCHEMES,
            simplicity=_LGBM_SIMPLE,
        ),
        Candidate(
            "lgbm_rank",
            "rank",
            "LightGBM LambdaRank(날짜별 순위)",
            "LightGBM LambdaRank (per date)",
            grid=[
                {"leaves": 15, "leaf_frac": 0.01, "n_estimators": 200, "lr": 0.05},
                {"leaves": 31, "leaf_frac": 0.01, "n_estimators": 200, "lr": 0.05},
            ],
            schemes=_RANK_SCHEMES,
            simplicity=_LGBM_SIMPLE,
        ),
        Candidate(
            "rsmt_v3",
            "rsmt",
            "RSMT-Q v3(Simes 검정, 변동성 표준화 목표, 리프별 γ, 주간)",
            "RSMT-Q v3 (Simes test, vol-standardised target, leaf gamma, weekly)",
            rows="W",
            grid=[
                {
                    "alpha": 0.05,
                    "max_depth": 3,
                    "half_life_days": 756,
                    "multiple_testing": "simes",
                    "target_scale": "vol",
                    "gamma_mode": "leaf",
                    "gamma_shrink_k": 50.0,
                }
            ],
            tuned=False,
        ),
        # ---- candidates v3 (docs/MODELING.md 16.4)
        Candidate(
            "enet_logit",
            "linear",
            "탄력망 로지스틱(날짜 중앙값 초과, l1 비율 0.5)",
            "elastic-net logistic (above date median, l1 ratio 0.5)",
            grid=[{"C": 0.005}, {"C": 0.05}],
            schemes=_RANK_SCHEMES,
            simplicity=(("C", -1),),
        ),
        Candidate(
            "rf",
            "bagging",
            "랜덤 포레스트(rank-gauss 목표)",
            "random forest (rank-gauss target)",
            grid=[{"kind": "rf", "leaf_frac": 0.005}, {"kind": "rf", "leaf_frac": 0.02}],
            schemes=_RANK_SCHEMES,
            simplicity=(("leaf_frac", +1),),
        ),
        Candidate(
            "et",
            "bagging",
            "엑스트라 트리(rank-gauss 목표)",
            "extra trees (rank-gauss target)",
            grid=[{"kind": "et", "leaf_frac": 0.005}, {"kind": "et", "leaf_frac": 0.02}],
            schemes=_RANK_SCHEMES,
            simplicity=(("leaf_frac", +1),),
        ),
        Candidate(
            "xgb_reg",
            "tree",
            "XGBoost 회귀",
            "XGBoost regressor",
            grid=[{"depth": 3}, {"depth": 5}],
            schemes=_REG_SCHEMES,
            simplicity=(("depth", -1),),
        ),
        Candidate(
            "xgb_clf",
            "tree",
            "XGBoost 분류(날짜 중앙값 초과)",
            "XGBoost classifier (above date median)",
            grid=[{"depth": 3}, {"depth": 5}],
            schemes=_RANK_SCHEMES,
            simplicity=(("depth", -1),),
        ),
        Candidate(
            "xgb_clf_w",
            "tree",
            "XGBoost 분류(주간 행)",
            "XGBoost classifier (weekly rows)",
            rows="W",
            grid=[{"depth": 3}, {"depth": 5}],
            schemes=_RANK_SCHEMES,
            simplicity=(("depth", -1),),
        ),
        Candidate(
            "xgb_rank",
            "rank",
            "XGBoost 순위(pairwise, 날짜별)",
            "XGBoost ranker (pairwise, per date)",
            grid=[{"depth": 3}, {"depth": 5}],
            schemes=_RANK_SCHEMES,
            simplicity=(("depth", -1),),
        ),
        Candidate(
            "cat_clf",
            "tree",
            "CatBoost 분류(날짜 중앙값 초과)",
            "CatBoost classifier (above date median)",
            grid=[{"depth": 4}, {"depth": 6}],
            schemes=_RANK_SCHEMES,
            simplicity=(("depth", -1),),
        ),
        Candidate(
            "cat_rank",
            "rank",
            "CatBoost 순위(YetiRank, 날짜별)",
            "CatBoost ranker (YetiRank, per date)",
            grid=[{"depth": 4}, {"depth": 6}],
            simplicity=(("depth", -1),),
        ),
        Candidate(
            "mlp",
            "nn",
            "신경망 MLP(64·32, rank-gauss 목표)",
            "neural net MLP (64-32, rank-gauss target)",
            grid=[{"alpha": 0.001}, {"alpha": 0.01}],
            schemes=_RANK_SCHEMES,
            simplicity=(("alpha", +1),),
        ),
        # ---- Bayesian-optimised, regularised boosting / bagging (docs/MODELING.md 82, protocol v4.1)
        Candidate(
            "lgbm_rank_bo",
            "rank",
            "LightGBM LambdaRank(베이지안 최적화·정규화)",
            "LightGBM LambdaRank (Bayesian-optimised, regularised)",
            grid=[{"leaves": 15, "leaf_frac": 0.02, "n_estimators": 200, "lr": 0.05}],
            fitter="lgbm_rank",
            search={**_LGBM_SPACE, "scheme": ("cat", ("none", "magnitude"))},
            n_trials=BO_TRIALS,
            simplicity=_LGBM_SIMPLE,
        ),
        Candidate(
            "lgbm_reg_bo",
            "tree",
            "LightGBM 회귀(베이지안 최적화·정규화)",
            "LightGBM regressor (Bayesian-optimised, regularised)",
            grid=[{"leaves": 15, "leaf_frac": 0.02}],
            fitter="lgbm_reg",
            search={**_LGBM_SPACE, "scheme": ("cat", ("none", "magnitude", "winsor"))},
            n_trials=BO_TRIALS,
            simplicity=_LGBM_SIMPLE,
        ),
        Candidate(
            "et_bo",
            "bagging",
            "엑스트라 트리(베이지안 최적화·정규화)",
            "extra trees (Bayesian-optimised, regularised)",
            grid=[{"kind": "et", "leaf_frac": 0.02}],
            fitter="et",
            search={**_ET_SPACE, "scheme": ("cat", ("none", "magnitude"))},
            n_trials=BO_TRIALS,
            simplicity=(("leaf_frac", +1), ("max_features", -1)),
        ),
        # ---- docs/MODELING.md 83 (protocol v4.2): the other families, Bayesian-optimised and regularised
        Candidate(
            "logit_bo",
            "linear",
            "탄력망 로지스틱(베이지안 최적화·정규화)",
            "elastic-net logistic (Bayesian-optimised, regularised)",
            grid=[{"C": 0.005, "l1_ratio": 0.5}],
            fitter="enet_logit",
            search={"C": ("float_log", 1e-4, 1.0), "l1_ratio": ("float", 0.0, 1.0), "scheme": _SCHEMES_CLS},
            n_trials=BO_TRIALS,
            simplicity=(("C", -1),),
        ),
        Candidate(
            "ridge_bo",
            "linear",
            "릿지 회귀(베이지안 최적화·정규화)",
            "ridge regression (Bayesian-optimised, regularised)",
            grid=[{"a": 0.1}],
            fitter="ridge",
            search={"a": ("float_log", 1e-3, 1e3), "scheme": _SCHEMES_REG},
            n_trials=BO_TRIALS,
            simplicity=(("a", +1),),
        ),
        Candidate(
            "rf_bo",
            "bagging",
            "랜덤 포레스트(베이지안 최적화·정규화)",
            "random forest (Bayesian-optimised, regularised)",
            grid=[{"kind": "rf", "leaf_frac": 0.02}],
            fitter="rf",
            search={
                "kind": ("cat", ("rf",)),
                "max_features": ("float", 0.1, 0.7),
                "leaf_frac": ("float_log", 0.001, 0.05),
                "max_depth": ("int", 4, 24),
                "max_samples": ("float", 0.3, 0.9),
                "scheme": _SCHEMES_CLS,
            },
            n_trials=BO_TRIALS,
            simplicity=(("leaf_frac", +1), ("max_features", -1)),
        ),
        Candidate(
            "xgb_clf_bo",
            "tree",
            "XGBoost 분류(베이지안 최적화·정규화)",
            "XGBoost classifier (Bayesian-optimised, regularised)",
            grid=[{"depth": 3}],
            fitter="xgb_clf",
            search=_XGB_SPACE,
            n_trials=BO_TRIALS,
            simplicity=(("leaf_frac", +1), ("depth", -1)),
        ),
        Candidate(
            "cat_clf_bo",
            "tree",
            "CatBoost 분류(베이지안 최적화·정규화)",
            "CatBoost classifier (Bayesian-optimised, regularised)",
            grid=[{"depth": 4}],
            fitter="cat_clf",
            search=_CAT_SPACE,
            n_trials=BO_TRIALS,
            simplicity=(("l2", +1), ("depth", -1)),
        ),
        Candidate(
            "mlp_bo",
            "nn",
            "신경망 MLP(베이지안 최적화·정규화)",
            "neural net MLP (Bayesian-optimised, regularised)",
            grid=[{"alpha": 0.01}],
            fitter="mlp",
            search={
                "alpha": ("float_log", 1e-4, 1.0),
                "hidden": ("cat", ("32", "64-32", "128-64")),
                "lr_init": ("float_log", 1e-4, 1e-2),
                "scheme": _SCHEMES_CLS,
            },
            n_trials=BO_TRIALS,
            simplicity=(("alpha", +1),),
        ),
        Candidate(
            "rsmt_v3_bo",
            "rsmt",
            "RSMT-Q v3(쌓기, 베이지안 최적화·정규화)",
            "RSMT-Q v3 (stacked, Bayesian-optimised, regularised)",
            rows="W",
            grid=[{"alpha": 0.05, "max_depth": 3, "half_life_days": 756, "gamma_shrink_k": 50.0, **_RSMT_V3}],
            fitter="rsmt_v3",
            tuned=False,  # never tuned as a base candidate: run_rsmt tunes it on the stacked score (docs 83)
            search=_RSMT_SPACE,
            n_trials=BO_TRIALS,
            simplicity=(("alpha", -1), ("max_depth", -1), ("min_leaf_blocks", +1)),
        ),
        # ---- diagnostics (never selected): the same models on the v2 inputs only
        Candidate(
            "logit_w_core",
            "linear",
            "진단: logit_w(v2 특징만)",
            "diagnostic: logit_w (v2 inputs only)",
            rows="W",
            grid=[{"C": 0.0005}, {"C": 0.005}, {"C": 0.05}],
            schemes=_RANK_SCHEMES,
            simplicity=(("C", -1),),
            diagnostic=True,
            feature_set="core",
            fitter="logit_w",
        ),
        Candidate(
            "lgbm_clf_core",
            "tree",
            "진단: lgbm_clf(v2 특징만)",
            "diagnostic: lgbm_clf (v2 inputs only)",
            grid=[{"leaves": 15, "leaf_frac": 0.01}, {"leaves": 31, "leaf_frac": 0.01}],
            schemes=_RANK_SCHEMES,
            simplicity=_LGBM_SIMPLE,
            diagnostic=True,
            feature_set="core",
            fitter="lgbm_clf",
        ),
    ]


ENSEMBLE = Candidate(
    "ensemble", "ensemble", "앙상블(내부 CV 상위 3개 순위 평균)", "ensemble (rank mean of inner top 3)", tuned=False
)
ENSEMBLE5 = Candidate(
    "ensemble5", "ensemble", "앙상블(내부 CV 상위 5개 순위 평균)", "ensemble (rank mean of inner top 5)", tuned=False
)
ENSEMBLE_W = Candidate(
    "ensemble_w",
    "ensemble",
    "가중 앙상블(전 후보, 가중 = 내부 점수)",
    "weighted ensemble (all candidates, weight = inner score)",
    tuned=False,
)
FAMILY_BLEND = Candidate(
    "family_blend",
    "ensemble",
    "계열 혼합(선형·부스팅·배깅·신경망 1위의 순위 평균)",
    "family blend (rank mean of the best linear / boosting / bagging / neural net)",
    tuned=False,
)
BLEND = Candidate(
    "blend",
    "ensemble",
    "고정 혼합(logit + lgbm_clf 순위 평균)",
    "fixed blend (rank mean of logit + lgbm_clf)",
    tuned=False,
)
COMPOSITE_CANDIDATES = (ENSEMBLE, ENSEMBLE5, ENSEMBLE_W, FAMILY_BLEND, BLEND)
COMPOSITES = tuple(c.name for c in COMPOSITE_CANDIDATES)  # scores built from other candidates' scores


def composite_rule(
    name: str, inner: dict[str, dict[str, Any]], cands: list[Candidate]
) -> tuple[list[str], list[float]]:
    """(members, weights) of composite ``name`` in one context.

    ``inner`` = {candidate: {"inner_auc", "inner_ic"}} of the context; eligible members are tuned, non-diagnostic,
    non-stacked candidates with a finite inner score, ranked by (inner score, inner IC). ``inner_auc`` holds the
    inner PRIMARY score (protocol v4: top-quintile net excess; v3: the AUC). ``blend`` uses its fixed members when
    they are present. Weights are 1 except ``ensemble_w`` (max(inner score - its no-skill value, 0); all-zero ->
    equal).
    """
    by = {c.name: c for c in cands}
    elig = [
        n
        for n, v in inner.items()
        if n in by
        and by[n].tuned
        and not by[n].diagnostic
        and n not in STACKED
        and n not in COMPOSITES
        and np.isfinite(v.get("inner_auc", np.nan))
    ]
    ranked = sorted(elig, key=lambda n: (-inner[n]["inner_auc"], -np.nan_to_num(inner[n].get("inner_ic", 0.0)), n))
    if name == BLEND.name:
        members = [m for m in BLEND_MEMBERS if m in inner]
        return (members, [1.0] * len(members)) if len(members) == len(BLEND_MEMBERS) else ([], [])
    if name == ENSEMBLE.name:
        m = ranked[:3]
    elif name == ENSEMBLE5.name:
        m = ranked[:5]
    elif name == ENSEMBLE_W.name:
        w = [max(float(inner[n]["inner_auc"]) - PRIMARY_NULL, 0.0) for n in ranked]
        if not any(w):
            return ranked, [1.0] * len(ranked)
        keep = [(n, x) for n, x in zip(ranked, w, strict=True) if x > 0]
        return [n for n, _ in keep], [x for _, x in keep]
    elif name == FAMILY_BLEND.name:
        m = []
        for fams in FAMILY_GROUPS.values():
            best = next((n for n in ranked if by[n].family in fams), None)
            if best is not None:
                m.append(best)
    else:
        raise ValueError(f"unknown composite {name!r}")
    return m, [1.0] * len(m)


__all__ = [
    "BLEND",
    "BLEND_MEMBERS",
    "COMPOSITES",
    "COMPOSITE_CANDIDATES",
    "ENSEMBLE",
    "ENSEMBLE5",
    "ENSEMBLE_W",
    "FAMILY_BLEND",
    "FAMILY_GROUPS",
    "FACTOR_IC_MAP",
    "FM_FACTORS",
    "RSMT_REGIME",
    "RSMT_XS",
    "STACKED",
    "Candidate",
    "FitContext",
    "Fitted",
    "composite_rule",
    "default_candidates",
    "fm_slopes",
    "reg_target",
    "row_weights",
]
