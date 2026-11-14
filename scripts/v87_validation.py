"""Stronger early stopping, period-blocked validation and regime-split models (docs/MODELING.md 87, pre-declared
2026-11-03). Research only: src/quantcio and the served models are not changed.

    .venv/bin/python scripts/v87_validation.py study --market KR
    .venv/bin/python scripts/v87_validation.py study --market KR --smoke   # last outer block only, own cache key

Inputs: arm A of 86 (the 84.6 frozen dataset, current inputs). Candidates:
- references: the grid LightGBM regressor (weekly rows) / LambdaRank / classifier, their 86 ``*_es`` twins, logit_w;
- strength ladder (87.2) ``*_es1se`` / ``*_strong`` / ``*_stump``: one validation tail (86's), the tree count by the
  per-date validation rank IC with the 1-standard-error rule toward fewer trees;
- period-blocked validation (87.3) ``*_cv`` / ``*_cvr``: annual blocks of the training window, each validated once by
  boosters trained on the other years (purged 2h sessions both sides), grown side by side; ``cvr`` picks the strength
  rung and the tree count by mean - SD of the per-block IC (1-SE toward the simplest);
- regime models (87.4) ``{lgbm_reg_w, logit_w}_rg_{stress, macro3}_{hard, soft}`` and the 50/50 ``*_mix`` composites.
Writes reports/modeling/<M>_v87.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v86_full_es as v86  # noqa: E402

from quantcio.config import load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

if __name__ == "__main__":  # checkpointed fitted objects pickle under the module name, loadable from other scripts
    sys.modules.setdefault("v87_validation", sys.modules[__name__])

OUT = v86.OUT
V87_VERSION = 1
LGBM_BASES = ("lgbm_reg_w", "lgbm_rank", "lgbm_clf")
REFS = (*LGBM_BASES, "logit_w")
OBJECTIVE = {"lgbm_reg_w": "regression", "lgbm_rank": "lambdarank", "lgbm_clf": "binary"}
RUNGS = {  # 87.2 (the "cur" rung = the base grid's first point)
    "strong": {
        "leaves": 7,
        "max_depth": 3,
        "leaf_frac": 0.05,
        "l2": 30.0,
        "feature_frac": 0.5,
        "bagging": 0.7,
        "lr": 0.02,
    },
    "stump": {
        "leaves": 2,
        "max_depth": 1,
        "leaf_frac": 0.05,
        "l2": 30.0,
        "feature_frac": 0.5,
        "bagging": 0.7,
        "lr": 0.05,
    },
}
SIMPLE = ("stump", "strong", "cur")  # 87.3: the simplest rung first
MODES = ("es1se", "strong", "stump", "cv", "cvr")
MODE_KO = {
    "es1se": "조기 종료(IC·1-SE)",
    "strong": "강한 제약 + 조기 종료",
    "stump": "그루터기 + 조기 종료",
    "cv": "연도 블록 검증",
    "cvr": "연도 블록 견고 검증 + 강도 자동",
}
N_MAX, PATIENCE, N_MIN, N_FALLBACK = 1000, 150, 10, 100
CHECKS = frozenset((*range(5, 51, 5), *range(60, 201, 10), *range(225, N_MAX + 1, 25)))
BLOCK_MONTHS = 12
RG_BASES = ("lgbm_reg_w", "logit_w")
POOLED = {"lgbm_reg_w": "lgbm_reg_w_strong", "logit_w": "logit_w"}  # 87.4: the pooled model of each regime family
KINDS = ("stress", "macro3")
WAYS = ("hard", "soft")
RG_MIN_MONTHS, SOFT_W = 24, 1.0 / 3.0
STRESS_VARS = (("r_mkt_dd", -1.0), ("r_mkt_vol20", 1.0), ("r_vix_z", -1.0), ("r_credit_z", -1.0))
MACRO_VARS = (
    "r_mkt_ret_3m",
    "r_mkt_dd",
    "r_mkt_vol20",
    "r_xs_disp",
    "r_vix_z",
    "r_credit_z",
    "r_curve_z",
    "r_score_region",
)
KMEANS_K, KMEANS_INIT = 3, 20
MIXES = {f"{n}_rg_{k}_mix": (POOLED[n], f"{n}_rg_{k}_hard") for n in RG_BASES for k in KINDS}
V86_REPRO = (*REFS, *(f"{n}_es" for n in LGBM_BASES))
LOG: list[Path] = []
_CHOICE: dict[tuple, tuple[str, int, dict]] = {}  # (base, mode, cutoff, rows, last date) -> the validated choice

say = v86.say


def _log(rec: dict) -> None:
    if LOG:
        with LOG[0].open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")


# ------------------------------------------------------------------------------------------------ validation IC


class _Val:
    """One fold's validation month-end rows: the matrix and per-date centred label ranks (87.1: the rank IC)."""

    def __init__(self, val: pd.DataFrame, cols: list[str]):
        v = val[val["is_me"].to_numpy(bool) & np.isfinite(val["y_e"].to_numpy(dtype=float))]
        v = v[v.groupby("date")["y_e"].transform("size").to_numpy() >= mt.MIN_NAMES]
        self.codes, uniq = pd.factorize(v["date"], sort=True)
        self.n_dates = len(uniq)
        self.X = v[cols].to_numpy(dtype=np.float64)
        self.b = _centred_ranks(v["y_e"].to_numpy(dtype=float), self.codes)
        self.bb = np.bincount(self.codes, self.b * self.b, minlength=self.n_dates)

    def ic(self, score: np.ndarray) -> np.ndarray:
        a = _centred_ranks(score, self.codes)
        ab = np.bincount(self.codes, a * self.b, minlength=self.n_dates)
        aa = np.bincount(self.codes, a * a, minlength=self.n_dates)
        ok = (aa > 0) & (self.bb > 0)
        out = np.zeros(self.n_dates)  # all-equal scores on a date = no view = IC 0
        out[ok] = ab[ok] / np.sqrt(aa[ok] * self.bb[ok])
        return out


def _centred_ranks(x: np.ndarray, codes: np.ndarray) -> np.ndarray:
    r = pd.Series(x).groupby(codes).rank(method="average")
    return (r - r.groupby(codes).transform("mean")).to_numpy(dtype=float)


# ------------------------------------------------------------------------------------------------ folds


def tail_folds(train: pd.DataFrame, ctx) -> tuple[list, dict]:
    """86's validation tail: the last 20% of the training months (3..12), fit rows purged h + 1 sessions before it."""
    split = v86.es_split(train, ctx)
    if split is None:
        return [], {}
    fit, val, info = split
    return [(fit, val)], info


def year_folds(train: pd.DataFrame, ctx) -> tuple[list, dict]:
    """87.3: full 12-month blocks of month-ends counted back from the window's end; each block validates boosters
    trained on every other row of the window except those within 2h sessions of the block (both sides)."""
    me = np.sort(train.loc[train["is_me"].to_numpy(bool), "date"].unique())
    cal = pd.DatetimeIndex(ctx.calendar)
    gap = 2 * int(ctx.horizon)
    dates = train["date"].to_numpy(dtype="datetime64[ns]")
    out, spans = [], []
    for j in range(len(me) // BLOCK_MONTHS):
        hi = len(me) - j * BLOCK_MONTHS
        blk = me[hi - BLOCK_MONTHS : hi]
        p0, p1 = int(cal.searchsorted(blk[0])), int(cal.searchsorted(blk[-1]))
        lo_cut, hi_cut = cal[max(p0 - gap, 0)], cal[min(p1 + gap, len(cal) - 1)]
        val = train[(dates >= blk[0]) & (dates <= blk[-1])]
        fit = train[(dates < np.datetime64(lo_cut, "ns")) | (dates > np.datetime64(hi_cut, "ns"))]
        if fit["date"].nunique() >= 3:
            out.append((fit, val))
            spans.append(f"{pd.Timestamp(blk[0]).date()}..{pd.Timestamp(blk[-1]).date()}")
    return out[::-1], {"blocks": len(out), "spans": spans[::-1]}


# ------------------------------------------------------------------------------------------------ lockstep boosting


def rung_params(base: str, rung: str) -> dict:
    if rung == "cur":
        by = {c.name: c for c in cmod.default_candidates()}
        p = dict(by[base].grid[0])
        p.pop("n_estimators", None)
        return p
    return dict(RUNGS[rung])


def _train_set(base: str, fit: pd.DataFrame, cols: list[str], ctx):
    """The LightGBM Dataset of one fold exactly as the base fitter builds its rows / target / weights."""
    import lightgbm as lgb

    group = None
    if base == "lgbm_rank":
        d = fit[np.isfinite(fit["y_q"].to_numpy(dtype=float))].sort_values("date", kind="stable")
        y = d["y_q"].to_numpy(dtype=int)
        group = d.groupby("date", sort=True).size().to_numpy()
        w = cmod.row_weights(d, "none", ctx)
    else:
        yall = fit["y_cls"].to_numpy(dtype=float) if base == "lgbm_clf" else cmod.reg_target(fit, "none")
        ok = np.isfinite(yall)
        d, y = fit[ok], yall[ok]
        w = cmod.row_weights(fit, "none", ctx)[ok]
    if len(d) < 100:
        return None
    return lgb.Dataset(d[cols].to_numpy(dtype=np.float64), y, weight=w, group=group, params={"verbose": -1}), len(d)


def _native(base: str, params: dict, n: int, ctx) -> dict:
    kw = cmod._lgbm_common(params, n, ctx)
    kw.pop("n_estimators")
    kw["objective"] = OBJECTIVE[base]
    if base == "lgbm_rank":
        kw["lambdarank_truncation_level"] = 10_000
    return kw


def lockstep(base: str, folds: list, params: dict, ctx, cols: list[str], robust: bool) -> list[dict]:
    """One booster per fold grown side by side; at every checkpoint the per-fold mean validation IC and the score
    (mean over folds, minus their SD when ``robust``). Stops PATIENCE trees after the best score (87.1)."""
    import lightgbm as lgb

    boosters, vals = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fit, val in folds:
            made, v = _train_set(base, fit, cols, ctx), _Val(val, cols)
            if made is None or v.n_dates == 0:
                continue
            boosters.append(lgb.Booster(params=_native(base, params, made[1], ctx), train_set=made[0]))
            vals.append(v)
        recs: list[dict] = []
        if not boosters:
            return recs
        best, best_n = -np.inf, 0
        for it in range(1, N_MAX + 1):
            for b in boosters:
                b.update()
            if it not in CHECKS:
                continue
            per = [v.ic(b.predict(v.X, raw_score=True)) for b, v in zip(boosters, vals, strict=True)]
            means = np.array([p.mean() for p in per])
            score = float(means.mean() - (means.std(ddof=1) if robust and len(means) > 1 else 0.0))
            pooled = np.concatenate(per)
            se = float(pooled.std(ddof=1) / np.sqrt(len(pooled))) if len(pooled) > 1 else 0.0
            recs.append({"n": it, "score": score, "se": se, "fold_means": [round(float(x), 5) for x in means]})
            if score > best:
                best, best_n = score, it
            elif it - best_n >= PATIENCE:
                break
    return recs


def one_se_pick(curves: dict[str, list[dict]]) -> tuple[str, int, dict] | None:
    """The best (rung, trees) by score; then, within one SE of it, the simplest rung (SIMPLE) and the fewest trees."""
    pts = [(r, x) for r, recs in curves.items() for x in recs if np.isfinite(x["score"])]
    if not pts:
        return None
    br, bx = max(pts, key=lambda t: t[1]["score"])
    near = [(r, x) for r, x in pts if x["score"] >= bx["score"] - bx["se"]]
    r, x = min(near, key=lambda t: (SIMPLE.index(t[0]), t[1]["n"]))
    rule = {
        "best_rung": br,
        "best_n": bx["n"],
        "best_score": round(bx["score"], 5),
        "se": round(bx["se"], 5),
        "picked_score": round(x["score"], 5),
        "stopped": {k: (v[-1]["n"] if v else None) for k, v in curves.items()},
    }
    return r, max(N_MIN, int(x["n"])), rule


def choose(base: str, mode: str, train: pd.DataFrame, ctx) -> tuple[str, int, dict]:
    """(rung, tree count, record) of one validated fit; memoised per training window (the regime models reuse
    ``lgbm_reg_w_strong``'s choice, 87.4)."""
    key = (base, mode, str(ctx.cutoff), len(train), str(train["date"].max()))
    if key in _CHOICE:
        return _CHOICE[key]
    cols = cmod._tree_cols(ctx, train)
    folds, info = year_folds(train, ctx) if mode in ("cv", "cvr") else tail_folds(train, ctx)
    rungs = SIMPLE if mode == "cvr" else (("cur",) if mode in ("es1se", "cv") else (mode,))
    curves = {r: lockstep(base, folds, rung_params(base, r), ctx, cols, mode == "cvr") for r in rungs} if folds else {}
    pick = one_se_pick(curves)
    if pick is None:
        out = (rungs[-1], N_FALLBACK, {**info, "fallback": True})
    else:
        out = (pick[0], pick[1], {**info, **pick[2]})
    _CHOICE[key] = out
    return out


def make_vfit(base: str, mode: str):
    def fit(c, train, params, scheme, ctx):
        rung, n, info = choose(base, mode, train, ctx)
        out = cmod._FITTERS[base](c, train, {**rung_params(base, rung), "n_estimators": n}, "none", ctx)
        out.extra["v87"] = {"rung": rung, "n": n, **{k: v for k, v in info.items() if k != "spans"}}
        _log(
            {
                "cand": c.name,
                "cutoff": str(pd.Timestamp(ctx.cutoff).date()) if ctx.cutoff is not None else None,
                "n_train": len(train),
                "rung": rung,
                "n": n,
                **info,
            }
        )
        return out

    return fit


# ------------------------------------------------------------------------------------------------ regimes


class RegimeMap:
    """87.4: a market / macro state per date from that date's own regime columns; the standardisation, the stress
    median and the k-means centres come from the training window's month-ends only."""

    def __init__(self, kind: str):
        self.kind = kind
        self.k = 2 if kind == "stress" else KMEANS_K

    def _cols(self, f: pd.DataFrame) -> list[str]:
        names = [c for c, _ in STRESS_VARS] if self.kind == "stress" else list(MACRO_VARS)
        return [c for c in names if c in f]

    def _stress(self, f: pd.DataFrame) -> np.ndarray:
        sign = dict(STRESS_VARS)
        z = np.column_stack([sign[c] * (f[c].to_numpy(dtype=float) - self.mu[c]) / self.sd[c] for c in self.cols])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmean(z, axis=1)

    def _z(self, f: pd.DataFrame) -> np.ndarray:
        z = np.column_stack([(f[c].to_numpy(dtype=float) - self.mu[c]) / self.sd[c] for c in self.cols])
        return np.nan_to_num(z, nan=0.0)

    def fit(self, train: pd.DataFrame) -> RegimeMap:
        me = train[train["is_me"].to_numpy(bool)]
        d = (me if len(me) else train).groupby("date")[self._cols(train)].first()
        self.cols = list(d.columns)
        self.mu = d.mean().to_dict()
        self.sd = {c: (float(s) if np.isfinite(s) and s > 0 else 1.0) for c, s in d.std(ddof=0).items()}
        self.mu = {c: (float(m) if np.isfinite(m) else 0.0) for c, m in self.mu.items()}
        if self.kind == "stress":
            self.thr = float(np.nanmedian(self._stress(d)))
            return self
        from sklearn.cluster import KMeans

        z = self._z(d)
        km = KMeans(n_clusters=self.k, n_init=KMEANS_INIT, random_state=0).fit(z)
        vol = self.cols.index("r_mkt_vol20") if "r_mkt_vol20" in self.cols else 0
        order = np.argsort(km.cluster_centers_[:, vol])  # label 0 = the calmest centre (lowest market volatility)
        self.centres = km.cluster_centers_[order]
        return self

    def assign(self, f: pd.DataFrame) -> np.ndarray:
        if self.kind == "stress":
            s = self._stress(f)
            return (np.nan_to_num(s, nan=-np.inf) > self.thr).astype(int)
        z = self._z(f)
        return np.argmin(((z[:, None, :] - self.centres[None, :, :]) ** 2).sum(axis=2), axis=1)

    def describe(self) -> dict:
        if self.kind == "stress":
            return {"threshold": round(self.thr, 4), "mu": self.mu, "sd": self.sd}
        cen = {
            str(g): {c: round(self.mu[c] + self.sd[c] * float(v), 4) for c, v in zip(self.cols, row, strict=True)}
            for g, row in enumerate(self.centres)
        }
        return {"centres": cen}


@dataclass
class RegimeFitted(cmod.Fitted):
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.array([])
        out = np.asarray(self.extra["pooled"].predict(df), dtype=float).copy()
        lab = self.extra["rmap"].assign(df)
        for g, mdl in self.extra["models"].items():
            m = lab == g
            if mdl is not None and m.any():
                out[m] = mdl.predict(df[m])
        return out

    def importance(self) -> dict[str, float]:
        return self.extra["pooled"].importance()

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame | None:
        return None


if __name__ == "__main__":
    RegimeMap.__module__ = RegimeFitted.__module__ = "v87_validation"


def make_rfit(base: str, kind: str, soft: bool):
    def fit(c, train, params, scheme, ctx):
        rmap = RegimeMap(kind).fit(train)
        lab = rmap.assign(train)
        if base == "lgbm_reg_w":  # the pooled model is lgbm_reg_w_strong: same rung and tree count (87.4)
            _, n, _ = choose(base, "strong", train, ctx)
            params, scheme = {**RUNGS["strong"], "n_estimators": n}, "none"
        fitter = cmod._FITTERS[base]
        pooled = fitter(c, train, params, scheme, ctx)
        me_dates = train["date"].to_numpy()[train["is_me"].to_numpy(bool)]
        me_lab = lab[train["is_me"].to_numpy(bool)]
        models, months = {}, {}
        for g in range(rmap.k):
            months[g] = int(pd.unique(me_dates[me_lab == g]).size)
            if months[g] < RG_MIN_MONTHS:
                models[g] = None
                continue
            if soft:
                sub = train.assign(w_date=train["w_date"].to_numpy(dtype=float) * np.where(lab == g, 1.0, SOFT_W))
            else:
                sub = train[lab == g]
            models[g] = fitter(c, sub, params, scheme, ctx)
        info = {"months": months, "used": [g for g, m in models.items() if m is not None], **rmap.describe()}
        _log(
            {
                "cand": c.name,
                "cutoff": str(pd.Timestamp(ctx.cutoff).date()) if ctx.cutoff is not None else None,
                "n_train": len(train),
                "regime": {"months": months, "used": info["used"]},
            }
        )
        extra = {"rmap": rmap, "pooled": pooled, "models": models, "v87": info}
        return RegimeFitted(c.name, params, scheme, None, pooled.features, "regime", extra)

    return fit


# ------------------------------------------------------------------------------------------------ candidates


def candidates() -> list[cmod.Candidate]:
    by = {c.name: c for c in cmod.default_candidates()}
    out = [by["mom_12_1"], *(by[n] for n in REFS)]
    for n in LGBM_BASES:
        cmod._FITTERS[f"{n}_es"] = v86._make_es_fitter(n)
        out.append(
            replace(
                by[n],
                name=f"{n}_es",
                fitter=f"{n}_es",
                label_ko=f"{by[n].label_ko} + 조기 종료",
                label_en=f"{by[n].label_en} + early stopping",
            )
        )
    fixed = {"grid": [{}], "schemes": ("none",), "tuned": False, "simplicity": ()}
    for n in LGBM_BASES:
        for mode in MODES:
            key = f"{n}_{mode}"
            cmod._FITTERS[key] = make_vfit(n, mode)
            out.append(
                replace(
                    by[n],
                    name=key,
                    fitter=key,
                    label_ko=f"{by[n].label_ko} + {MODE_KO[mode]}",
                    label_en=f"{by[n].label_en} + v87 {mode}",
                    **fixed,
                )
            )
    for n in RG_BASES:
        for kind in KINDS:
            for way in WAYS:
                key = f"{n}_rg_{kind}_{way}"
                cmod._FITTERS[key] = make_rfit(n, kind, way == "soft")
                extra = fixed if n == "lgbm_reg_w" else {}
                out.append(
                    replace(
                        by[n],
                        name=key,
                        fitter=key,
                        label_ko=f"{by[n].label_ko} + 국면 {kind}·{way}",
                        label_en=f"{by[n].label_en} + regime {kind} {way}",
                        **extra,
                    )
                )
    return out


def patch_selection() -> None:
    """The composites are the four pre-declared 50/50 mixes only (87.4); no ensemble / diagnostic pairs."""
    comps = tuple(
        cmod.Candidate(
            name, "ensemble", f"부분 공유: `{a}` + `{b}` 순위 평균", f"partial pooling: {a} + {b}", tuned=False
        )
        for name, (a, b) in MIXES.items()
    )
    orig = sel.composite_rule

    def rule(name, inner, cands):
        if name in MIXES:
            return list(MIXES[name]), [1.0, 1.0]
        return orig(name, inner, cands)

    sel.composite_rule = rule
    sel.COMPOSITE_CANDIDATES = comps
    sel.COMPOSITES = tuple(c.name for c in comps)
    sel.DIAGNOSTIC_PAIRS = ()


# ------------------------------------------------------------------------------------------------ analysis


def insample(res, ds, names: list[str]) -> dict:
    """Final-context models scored on their own training month-ends (the overfitting gap, as 86.3)."""
    data = sel._Data(ds)
    _, _, final, _ = sel.make_contexts(data)
    rows = data.train(final.train_end, "M")
    out = {}
    for n in names:
        f = res.final.fitted.get(n) if res.final is not None else None
        if f is None:
            continue
        fr = rows[["date", "entity", "y_e", "r"]].assign(score=f.predict(rows))
        info = f.extra.get("v87") or f.extra.get("es") or {}
        out[n] = {
            "ic": float(mt.per_date_ic(fr, "score").mean()),
            "choice": {k: info.get(k) for k in ("rung", "n", "n_final")},
        }
    return out


def choice_stats(path: Path, outer_starts: set[str]) -> dict:
    """Per candidate: the validated rung / tree count over its outer, holdout and final refits (87.1-87.3)."""
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    seen, out = set(), {}
    for r in recs:
        k = (r["cand"], r["cutoff"], r["n_train"])
        if "n" not in r or r["cutoff"] not in outer_starts or k in seen:
            continue
        seen.add(k)
        out.setdefault(r["cand"], []).append(r)
    stats = {}
    for name, rr in out.items():
        n = np.array([r["n"] for r in rr], dtype=float)
        rungs = pd.Series([r["rung"] for r in rr]).value_counts().to_dict()
        stats[name] = {
            "fits": len(rr),
            "median_n": float(np.median(n)),
            "range_n": [int(n.min()), int(n.max())],
            "rungs": rungs,
            "blocks": [r.get("blocks") for r in rr if r.get("blocks") is not None],
            "fallback": sum(1 for r in rr if r.get("fallback")),
        }
    return stats


def yearly(res, name: str, cost: float) -> dict:
    """87.5.2 (secondary): per calendar year mean outer PRIMARY - its SD across years and the worst year."""
    s = v86._per_date(res, name, "outer", cost)
    if s.empty:
        return {}
    y = s.groupby(pd.DatetimeIndex(s.index).year).mean()
    return {
        "sd": float(y.std(ddof=1)) if len(y) > 1 else None,
        "worst": float(y.min()),
        "worst_year": int(y.idxmin()),
        "years": {int(k): round(float(v), 5) for k, v in y.items()},
    }


def regime_diag(res, ds, cost: float) -> dict:
    """Descriptive only (87.4): the final window's regime definitions applied to the outer + holdout month-ends."""
    data = sel._Data(ds)
    parts = [x for x in (res.oos, res.holdout_preds) if x is not None]
    dates = pd.DatetimeIndex(sorted(pd.concat(parts)["date"].unique()))
    f = data.f
    st = f[f["date"].isin(dates) & f["is_me"].to_numpy(bool)].groupby("date").first()
    out = {}
    for kind in KINDS:
        fitted = res.final.fitted.get(f"lgbm_reg_w_rg_{kind}_hard") if res.final is not None else None
        if fitted is None:
            continue
        rmap = fitted.extra["rmap"]
        lab = pd.Series(rmap.assign(st), index=st.index)
        prim = {}
        for name in (
            "mom_12_1",
            "lgbm_reg_w_strong",
            f"lgbm_reg_w_rg_{kind}_hard",
            "logit_w",
            f"logit_w_rg_{kind}_hard",
        ):
            s = pd.concat([v86._per_date(res, name, "outer", cost), v86._per_date(res, name, "holdout", cost)])
            prim[name] = {
                str(g): (round(float(v), 5) if np.isfinite(v := s[s.index.isin(lab.index[lab == g])].mean()) else None)
                for g in range(rmap.k)
            }
        out[kind] = {
            "months": {str(g): int((lab == g).sum()) for g in range(rmap.k)},
            "primary": prim,
            **rmap.describe(),
        }
    return out


GROUPS = {
    "ref": {"es − 원래": [(f"{b}_es", b) for b in LGBM_BASES]},
    "ladder": {f"{m} − es": [(f"{b}_{m}", f"{b}_es") for b in LGBM_BASES] for m in ("es1se", "strong", "stump")},
    "valid": {f"{m} − es1se": [(f"{b}_{m}", f"{b}_es1se") for b in LGBM_BASES] for m in ("cv", "cvr")},
    "regime": {f"{k}·{w}": [(f"{n}_rg_{k}_{w}", POOLED[n]) for n in RG_BASES] for k in KINDS for w in (*WAYS, "mix")},
}
GROUP_KO = {"ref": "참고", "ladder": "A. 강도 사다리", "valid": "B. 시기별 검증", "regime": "C. 국면 모델"}


def _f(x, spec: str = "+.4f") -> str:
    return v86._f(x, spec)


def write_md(m: str, r: dict) -> Path:
    t, ins, gate = r["table"], r["insample"], r["gate"]
    lines = [
        f"# {m} 조기 종료 강도 · 시기별 검증 · 국면 모델 (docs/MODELING.md 87)",
        "",
        f"- 데이터셋 `{r['snapshot']}` (84.6 고정 자료, 갈래 A 입력), 비용 {r['cost_bps']:.0f}bp, "
        "지표: 월 상위 20% 베타 조정 순초과",
        f"- 선정: `{r['selected']}` (게이트 {'통과' if gate.get('passed') else '실패'}: 값 {_f(gate.get('value'))}, "
        f"하한 {_f(gate.get('lo'))}, 모멘텀 {_f(gate.get('baseline'))}), 홀드아웃 1위: `{r['holdout_best']}`",
        f"- 재현(86절 A와 같아야 함): {r['repro']}",
        "",
        "## 판정 (날짜별 1차 지표 차이: 외부 평균 [95%] (t) / 홀드아웃 평균)",
        "",
    ]
    for g, grp in r["judgments"].items():
        lines += [f"### {GROUP_KO[g]}", "", "| 비교 | 외부 [95%] | t | 홀드아웃 | 판정 |", "|---|---|---|---|---|"]
        for k, e in grp.items():
            mo, mh = e["mean_outer"], e["mean_holdout"]
            lines.append(
                f"| {k} | {_f(mo['mean'])} [{_f(mo['lo'])}, {_f(mo['hi'])}] | {_f(mo['t'], '.2f')} | "
                f"{_f(mh['mean'])} | {'**도움**' if e['helps'] else '도움 없음'} |"
            )
        lines.append("")
    lines += [
        "## 후보별 (외부·홀드아웃 월 1차 지표, 학습 구간 IC − 외부 IC, 연도별 흔들림)",
        "",
        "| 모델 | 외부 [95%] | 홀드아웃 | 외부 IC | 학습 IC | 차이 | 연도 SD | 최악 연도 "
        "| 고른 강도·나무 수(외부 재학습) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n, b in t.items():
        i, y, st = ins.get(n, {}), r["yearly"].get(n, {}), r["choices"].get(n) or {}
        gap = i["ic"] - b["outer_ic"] if i.get("ic") is not None and b.get("outer_ic") is not None else None
        ch = ""
        if st:
            rungs = ", ".join(f"{k} {v}" for k, v in st["rungs"].items())
            ch = f"{rungs}; 중앙값 {st['median_n']:.0f} ({st['range_n'][0]}~{st['range_n'][1]})"
        elif n in r["es86"]:
            ch = f"86절 ES 중앙값 {_f(r['es86'][n].get('outer_refit_median'), '.0f')}"
        worst = f"{_f(y.get('worst'))} ({y.get('worst_year')})" if y else "–"
        lines.append(
            f"| `{n}` | {_f(b['outer'])} [{_f(b['outer_lo'])}, {_f(b['outer_hi'])}] | {_f(b['holdout'])} | "
            f"{_f(b['outer_ic'], '+.3f')} | {_f(i.get('ic'), '+.3f')} | {_f(gap, '+.3f')} | {_f(y.get('sd'), '.4f')} | "
            f"{worst} | {ch} |"
        )
    lines += ["", "## 국면 진단 (서술 통계, 마지막 학습 구간의 국면 정의로 외부 + 홀드아웃 월말을 나눔)", ""]
    for kind, d in r["regime_diag"].items():
        ks = sorted(d["months"])
        lines += [
            f"### `{kind}`",
            "",
            "| 국면 | 월 수 | " + " | ".join(f"`{n}`" for n in d["primary"]) + " |",
            "|---|---|" + "---|" * len(d["primary"]),
        ]
        for g in ks:
            lines.append(
                f"| {g} | {d['months'][g]} | " + " | ".join(_f(v.get(g)) for v in d["primary"].values()) + " |"
            )
        if "centres" in d:
            lines.append("")
            for g, c in d["centres"].items():
                lines.append(f"- 국면 {g} 중심: " + ", ".join(f"`{k}` {v:+.3f}" for k, v in c.items()))
        else:
            lines.append(f"- 기준: 스트레스 지수 > {d['threshold']} (1 = 스트레스)")
        lines.append("")
    p = OUT / f"{m}_v87{r.get('suffix', '')}.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# ------------------------------------------------------------------------------------------------ run


def research_dir(s, m: str) -> Path:
    p = s.cache_path / "research" / "v87" / m / v86.SNAP[m][:12]
    p.mkdir(parents=True, exist_ok=True)
    return p


def smoke(m: str) -> None:
    """Every new fitter once on the last outer window: runs, chosen rung / trees, seconds (no performance shown)."""
    import time

    s = load_settings()
    ds = v86.load_frozen(m, s, DiskCache(s.cache_path))
    cands = candidates()
    LOG[:] = [research_dir(s, m) / "choice_smoke.jsonl"]
    data = sel._Data(ds)
    outer, _, _, _ = sel.make_contexts(data)
    ctx0 = outer[-1]
    test = data.eval_rows(ctx0.test_dates)
    for c in cands:
        if not (c.name.endswith(MODES) or "_rg_" in c.name):
            continue
        t0 = time.monotonic()
        tr = data.train(ctx0.train_end, c.rows)
        fitted = c.fit(tr, dict(c.grid[0]), "none", sel._ctx(data, ctx0.test_dates[0], s, 8))
        p = fitted.predict(test)
        info = fitted.extra.get("v87") or {}
        brief = {k: info.get(k) for k in ("rung", "n", "blocks", "best_rung", "best_n", "stopped", "months", "used")}
        say(f"{c.name}: {len(tr)} rows, finite {np.isfinite(p).mean():.3f}, {brief} ({time.monotonic() - t0:.1f}s)")


def study(m: str) -> None:
    import serve_override as so

    s = load_settings()
    cache = DiskCache(s.cache_path)
    ds = v86.load_frozen(m, s, cache)
    patch_selection()
    cands = candidates()
    rd = research_dir(s, m)
    LOG[:] = [rd / "choice_log.jsonl"]
    v86.ES_LOG[:] = [rd / "es86_log.jsonl"]
    v86.ARM[0] = "v87"
    snap0, cost = ds.snapshot, float(ds.cost_bps)
    data = sel._Data(ds)
    outer, holdout, final, _ = sel.make_contexts(data)
    outer_starts = {str(pd.Timestamp(c.test_dates[0]).date()) for c in [*outer, holdout, final]}
    del data
    ds.snapshot = f"{snap0}|v87|{V87_VERSION}"
    say(f"{m}: {len(ds.xs_cols)} features + {len(ds.regime_cols)} regime, {len(cands)} candidates + {len(MIXES)} mixes")
    res = sel.run_selection(ds, s, cache, candidates=cands, threads=8, use_cache=True, progress=say)
    d = res.to_dict()
    names = [c.name for c in cands if c.name != sel.BASELINE]
    table = v86.table(res)
    v86_a = json.loads((OUT / f"{m}_v86.json").read_text())["arm_a"]["table"]
    repro = {n: (round(table[n]["outer"], 6), round(v86_a[n]["outer"], 6)) for n in V86_REPRO}
    r = {
        "market": m,
        "snapshot": snap0,
        "cost_bps": cost,
        "settings": {
            "n_max": N_MAX,
            "patience": PATIENCE,
            "n_min": N_MIN,
            "rungs": RUNGS,
            "block_months": BLOCK_MONTHS,
            "regime_min_months": RG_MIN_MONTHS,
            "soft_weight": SOFT_W,
        },
        "selected": d["selected"],
        "gate": d["gate"],
        "holdout_best": so.holdout_best(res),
        "table": table,
        "insample": insample(res, ds, names),
        "choices": choice_stats(LOG[0], outer_starts),
        "es86": v86.es_stats(v86.ES_LOG[0], "v87", outer_starts),
        "yearly": {n: yearly(res, n, cost) for n in table},
        "judgments": {
            g: {k: v86.averaged(pairs, res, res, cost) for k, pairs in grp.items()} for g, grp in GROUPS.items()
        },
        "regime_diag": regime_diag(res, ds, cost),
        "repro": "같음" if all(abs(a - b) < 1e-9 for a, b in repro.values()) else f"다름 {repro}",
    }
    js = OUT / f"{m}_v87.json"
    js.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    say(f"wrote {write_md(m, r)} and {js}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["study", "smoke"])
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    a = ap.parse_args()
    if a.cmd == "smoke":
        smoke(a.market)
    else:
        study(a.market)


if __name__ == "__main__":
    main()
