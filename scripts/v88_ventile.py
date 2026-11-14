"""Ventile-monotone models (docs/MODELING.md 88, pre-declared 2026-11-04): a score whose 20 within-date quantiles
have mean excess returns rising in order, consistently across years. Research only: src/quantcio and the served
models are not changed.

    .venv/bin/python scripts/v88_ventile.py smoke --market KR   # every new fitter once on the last outer window
    .venv/bin/python scripts/v88_ventile.py study --market KR

Inputs: arm A of 86 (the 84.6 frozen dataset, current inputs). New candidates (88.3), all on weekly rows, validated
on the annual blocks of 87.3 by the ventile score (the mean over every validation month-end of every block of that
date's Spearman correlation between the ventile number and the ventile mean of the winsorised excess return; 1-SE
toward the simplest; 88.3 as amended before the run):
- ``vt_lin_eq``: equal-weight composite of the sign-aligned inputs whose training IC |t| >= tau;
- ``vt_lin``: the same inputs, non-negative ridge on the per-date robust z of y_e (tau, lambda);
- ``vt_gbm``: LightGBM on the robust z, each input monotone in its training IC direction (tau, trees);
- ``vt_gbm_free`` (no constraints), ``vt_gbm_rank`` (the per-date uniform rank target), ``vt_gbm_bag`` (vt_gbm's
  annual-block boosters averaged instead of the full-window refit).
Writes reports/modeling/<M>_v88.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v86_full_es as v86  # noqa: E402
import v87_validation as v87  # noqa: E402

from quantcio.config import load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.events import winsorised  # noqa: E402

if __name__ == "__main__":  # checkpointed fitted objects pickle under the module name, loadable from other scripts
    sys.modules.setdefault("v88_ventile", sys.modules[__name__])

OUT = v86.OUT
V88_VERSION = 1
K = 20  # ventiles
TAUS = (0.0, 1.0, 2.0)  # 88.3: inputs kept when |t| of the training per-date IC >= tau
LAMS = (0.01, 0.1, 1.0, 10.0, 100.0)  # non-negative ridge penalties (inputs ~ unit variance, weights sum to 1)
GBM = dict(v87.RUNGS["strong"])
GBM_MODES = {"vt_gbm": ("z", True), "vt_gbm_free": ("z", False), "vt_gbm_rank": ("rank", True)}  # target, monotone
NEW = ("vt_lin_eq", "vt_lin", "vt_gbm", "vt_gbm_free", "vt_gbm_rank", "vt_gbm_bag")
REFS = ("logit_w", "ridge_rg", "lgbm_reg_w", "lgbm_rank", "lgbm_reg_w_cv")
NEW_KO = {
    "vt_lin_eq": "단순 지표(방향 맞춘 입력 동일 가중)",
    "vt_lin": "지표(방향 고정 비음수 릿지)",
    "vt_gbm": "LightGBM 견고 z + 방향 제약",
    "vt_gbm_free": "LightGBM 견고 z(제약 없음)",
    "vt_gbm_rank": "LightGBM 순위 목표 + 방향 제약",
    "vt_gbm_bag": "LightGBM 견고 z + 방향 제약, 블록 모델 평균",
}
CONTRASTS = {  # 88.4.2
    "C1 목표(견고 z − 순위)": ("vt_gbm", "vt_gbm_rank"),
    "C2 방향 제약(있음 − 없음)": ("vt_gbm", "vt_gbm_free"),
    "C3 배깅(평균 − 재학습)": ("vt_gbm_bag", "vt_gbm"),
    "C4 선형 지표 − 나무": ("vt_lin", "vt_gbm"),
    "C5 지표 가중(릿지 − 동일)": ("vt_lin", "vt_lin_eq"),
}
PICK_REFS = ("lgbm_rank", "mom_12_1")  # 88.4.3
BOOT_SIMS, BOOT_BLOCK = mt.BOOT_SIMS, mt.BOOT_BLOCK
LOG: list[Path] = []
_WIN: dict = {}  # the current training window's arrays (one window at a time)
_GBM: dict = {}  # (mode, window key) -> the lockstep choice (+ the chosen tau's fold boosters for vt_gbm)
_LIN: dict = {}

say = v86.say


def _log(rec: dict) -> None:
    if LOG:
        with LOG[0].open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")


# ------------------------------------------------------------------------------------------------ ventile curves


def ventile_matrix(codes: np.ndarray, n_dates: int, score: np.ndarray, y: np.ndarray, k: int = K) -> np.ndarray:
    """(dates x k) mean of y per within-date score quantile (rank average, u = (rank - 0.5) / n). A date whose
    scores are all equal is "no view": its row stays NaN."""
    s = pd.Series(score)
    g = s.groupby(codes)
    u = ((g.rank(method="average") - 0.5) / g.transform("size")).to_numpy(dtype=float)
    b = np.clip(np.floor(u * k), 0, k - 1).astype(int)
    idx = codes * k + b
    sums = np.bincount(idx, y, minlength=n_dates * k).reshape(n_dates, k)
    cnt = np.bincount(idx, minlength=n_dates * k).reshape(n_dates, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = sums / cnt
    flat = (g.transform("nunique").to_numpy() <= 1).astype(float)
    m[np.bincount(codes, flat, minlength=n_dates) > 0] = np.nan
    return m


def rho(curve: np.ndarray) -> float:
    """Spearman correlation of the quantile number and the curve (0 when the curve has gaps / no spread)."""
    if curve.size < 2 or not np.isfinite(curve).all() or np.ptp(curve) == 0:
        return 0.0
    r = pd.Series(curve).rank(method="average").to_numpy()
    return float(np.corrcoef(np.arange(curve.size), r)[0, 1])


def _curve(m: np.ndarray) -> np.ndarray:
    ok = np.isfinite(m).all(axis=1)
    return m[ok].mean(axis=0) if ok.any() else np.full(m.shape[1], np.nan)


def _boot_idx(n: int, seed: int, n_boot: int = BOOT_SIMS, block: int = BOOT_BLOCK) -> np.ndarray:
    """Circular moving-block bootstrap row indices (as ``metrics.block_bootstrap_means``)."""
    b = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, int(math.ceil(n / b))))
    return ((starts[:, :, None] + np.arange(b)[None, None, :]) % n).reshape(n_boot, -1)[:, :n]


def mr_test(m: np.ndarray, seed: int = 0) -> dict:
    """Patton-Timmermann (2010) monotonic-relation test: J = min adjacent mean difference; p = share of centred
    block-bootstrap draws of the minimum at least J (H0: every step <= 0, least favourable = flat). A non-monotone
    curve is in neither hypothesis, so the minimum per-step t (``t_min``) is reported as well."""
    d = np.diff(m, axis=1)
    mu = d.mean(axis=0)
    boot = d[_boot_idx(len(d), seed)].mean(axis=1)
    jb = (boot - mu).min(axis=1)
    t = mu / boot.std(axis=0, ddof=1)  # 88.2: the intersection-union view, every step on its own
    return {"J": float(mu.min()), "p": float((jb >= mu.min()).mean()), "t_min": float(t.min())}


def _slopes(m: np.ndarray) -> np.ndarray:
    c = np.arange(m.shape[1]) - (m.shape[1] - 1) / 2.0
    return (m * c).sum(axis=1) / (c * c).sum()


def vmat(frame: pd.DataFrame | None, name: str, label: str = "y_e") -> tuple[pd.DatetimeIndex, np.ndarray]:
    """The ventile matrix of a candidate's OOS month-end scores (label y_e, or R = the simple total return)."""
    if frame is None or name not in frame:
        return pd.DatetimeIndex([]), np.zeros((0, K))
    d = frame[["date", name, "y_e", "r"]].dropna(subset=[name, "y_e"])
    d = d[d.groupby("date")["y_e"].transform("size").to_numpy() >= mt.MIN_NAMES]
    y = d["y_e"].to_numpy(dtype=float) if label == "y_e" else np.expm1(d["r"].to_numpy(dtype=float))
    codes, uniq = pd.factorize(d["date"], sort=True)
    m = ventile_matrix(codes, len(uniq), d[name].to_numpy(dtype=float), y)
    ok = np.isfinite(m).all(axis=1)
    return pd.DatetimeIndex(uniq)[ok], m[ok]


def vstats(dates: pd.DatetimeIndex, m: np.ndarray, seed: int = 0) -> dict:
    """88.2: the curve (date-demeaned, per month), rho20 [95%], slope x 19 [95%] t, rises of 19, MR p (20 / 5
    quantiles), per calendar year rho20."""
    if len(m) < 3:
        return {}
    m = m - m.mean(axis=1, keepdims=True)
    curve = m.mean(axis=0)
    draws = np.array([rho(m[i].mean(axis=0)) for i in _boot_idx(len(m), seed)])
    sl = mt.summarize(_slopes(m) * (K - 1), 0.0, seed + 1)
    q5 = m.reshape(len(m), 5, K // 5).mean(axis=2)
    years = pd.Series(range(len(dates)), index=dates).groupby(dates.year).apply(list)
    yr = {int(yy): round(rho(m[ix].mean(axis=0)), 3) for yy, ix in years.items()}
    return {
        "n": len(m),
        "curve": [round(float(x), 5) for x in curve],
        "rho": rho(curve),
        "rho_lo": float(np.quantile(draws, 0.025)),
        "rho_hi": float(np.quantile(draws, 0.975)),
        "rho_date": mt.summarize(date_rho(m), 0.0, seed + 4),
        "spread": sl,
        "ups": int((np.diff(curve) > 0).sum()),
        "mr20": mr_test(m, seed + 2),
        "mr5": mr_test(q5, seed + 3),
        "q5": [round(float(x), 5) for x in q5.mean(axis=0)],
        "years": yr,
        "years_pos": int(sum(v > 0 for v in yr.values())),
        "years_min": min(yr.values()) if yr else None,
    }


def paired(a: tuple, b: tuple, seed: int = 0) -> dict:
    """rho20 difference a - b on common dates (joint block bootstrap) and the per-date slope difference."""
    (da, ma), (db, mb) = a, b
    common = da.intersection(db)
    if len(common) < 3:
        return {}
    ia, ib = da.get_indexer(common), db.get_indexer(common)
    ma, mb = ma[ia], mb[ib]
    idx = _boot_idx(len(common), seed)
    draws = np.array([rho(ma[i].mean(axis=0)) - rho(mb[i].mean(axis=0)) for i in idx])
    return {
        "d_rho": rho(ma.mean(axis=0)) - rho(mb.mean(axis=0)),
        "lo": float(np.quantile(draws, 0.025)),
        "hi": float(np.quantile(draws, 0.975)),
        "d_spread": mt.summarize((_slopes(ma) - _slopes(mb)) * (K - 1), 0.0, seed + 1),
        "d_rho_date": mt.summarize(date_rho(ma) - date_rho(mb), 0.0, seed + 2),
    }


# ------------------------------------------------------------------------------------------------ window arrays


def _wkey(train: pd.DataFrame, ctx) -> tuple:
    return (str(ctx.cutoff), len(train), str(train["date"].max()))


def year_masks(dates: np.ndarray, is_me: np.ndarray, ctx) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    """87.3's annual blocks as row masks: (fit = outside the block and its 2h-session margins, validation = the
    block's rows)."""
    me = np.sort(np.unique(dates[is_me]))
    cal = pd.DatetimeIndex(ctx.calendar)
    gap = 2 * int(ctx.horizon)
    out, spans = [], []
    for j in range(len(me) // v87.BLOCK_MONTHS):
        hi = len(me) - j * v87.BLOCK_MONTHS
        blk = me[hi - v87.BLOCK_MONTHS : hi]
        p0, p1 = int(cal.searchsorted(blk[0])), int(cal.searchsorted(blk[-1]))
        lo_cut = np.datetime64(cal[max(p0 - gap, 0)], "ns")
        hi_cut = np.datetime64(cal[min(p1 + gap, len(cal) - 1)], "ns")
        val = (dates >= blk[0]) & (dates <= blk[-1])
        fit = (dates < lo_cut) | (dates > hi_cut)
        if np.unique(dates[fit]).size >= 3:
            out.append((fit, val))
            spans.append(f"{pd.Timestamp(blk[0]).date()}..{pd.Timestamp(blk[-1]).date()}")
    return out[::-1], spans[::-1]


def _ic_t(d: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per input: sign and |t| of the per-date Spearman IC vs y_e on these month-end rows (t = mean / SE)."""
    sign, t = np.ones(len(cols)), np.zeros(len(cols))
    for j, c in enumerate(cols):
        ic = mt.per_date_ic(d, c, "y_e").to_numpy(dtype=float)
        if len(ic) > 2 and ic.std(ddof=1) > 0:
            sign[j] = 1.0 if ic.mean() >= 0 else -1.0
            t[j] = abs(ic.mean()) / (ic.std(ddof=1) / np.sqrt(len(ic)))
    return sign, t


@dataclass
class Window:
    """One training window (weekly rows): inputs, targets, weights, validation labels and the annual folds; per
    fold and for the whole window the IC sign / |t| of every cross-sectional input (computed on that part only)."""

    key: tuple
    xs: list[str]
    reg: list[str]
    X: np.ndarray  # xs then regime columns
    target: dict
    w: np.ndarray
    ywin: np.ndarray
    codes: np.ndarray
    n_dates: int
    is_me: np.ndarray
    folds: list
    spans: list
    stats: list  # per fold (sign, t); stats_full for the window
    stats_full: tuple


def window(train: pd.DataFrame, ctx) -> Window:
    key = _wkey(train, ctx)
    if _WIN.get("key") == key:
        return _WIN["win"]
    _WIN.clear()
    cols = cmod._tree_cols(ctx, train)
    xs = [c for c in cols if c in ctx.xs_cols]
    reg = [c for c in cols if c not in ctx.xs_cols]
    d = train[["date", "y_e", "is_me", "w_date", *xs, *reg]].reset_index(drop=True)
    y = d["y_e"]
    yw = winsorised(y, d["date"])
    g = yw.groupby(d["date"])
    sd = g.transform("std").to_numpy(dtype=float)
    z = np.where(sd > 0, (yw - g.transform("mean")).to_numpy(dtype=float) / np.where(sd > 0, sd, 1.0), 0.0)
    u = (y.groupby(d["date"]).rank(method="average") - 0.5) / y.groupby(d["date"]).transform("size")
    codes, uniq = pd.factorize(d["date"], sort=True)
    is_me = d["is_me"].to_numpy(bool)
    dates = d["date"].to_numpy(dtype="datetime64[ns]")
    folds, spans = year_masks(dates, is_me, ctx)
    me = d[is_me]
    stats = [_ic_t(me[fit[is_me]], xs) for fit, _ in folds]
    win = Window(
        key,
        xs,
        reg,
        d[xs + reg].to_numpy(dtype=np.float64),
        {"z": z, "rank": ((u - 0.5) * np.sqrt(12.0)).to_numpy(dtype=float)},
        cmod.row_weights(d, "none", ctx),
        yw.to_numpy(dtype=float),
        codes,
        len(uniq),
        is_me,
        folds,
        spans,
        stats,
        _ic_t(me, xs),
    )
    _WIN.update(key=key, win=win)
    return win


def _kept(stats: tuple, tau: float) -> np.ndarray:
    return np.flatnonzero(stats[1] >= tau) if tau > 0 else np.arange(len(stats[1]))


class _Val:
    """One fold's validation month-ends (rows with >= MIN_NAMES per date): the per-date ventile rho of a score."""

    def __init__(self, win: Window, mask: np.ndarray):
        rows = np.flatnonzero(mask & win.is_me & np.isfinite(win.ywin))
        c = win.codes[rows]
        n = np.bincount(c, minlength=win.n_dates)
        rows = rows[n[c] >= mt.MIN_NAMES]
        self.rows = rows
        self.codes, uniq = pd.factorize(win.codes[rows], sort=True)
        self.n_dates = len(uniq)
        self.y = win.ywin[rows]

    def rho(self, score: np.ndarray) -> np.ndarray:
        if not self.n_dates:
            return np.zeros(0)
        return date_rho(ventile_matrix(self.codes, self.n_dates, score, self.y))


def date_rho(m: np.ndarray) -> np.ndarray:
    """Per row (date): Spearman correlation of the quantile number and that date's quantile means (no view = 0)."""
    from scipy.stats import rankdata

    ok = np.isfinite(m).all(axis=1)
    out = np.zeros(len(m))
    if ok.any():
        r = rankdata(m[ok], axis=1)
        r = r - r.mean(axis=1, keepdims=True)
        c = np.arange(m.shape[1]) - (m.shape[1] - 1) / 2.0
        den = np.sqrt((r * r).sum(axis=1) * (c * c).sum())
        out[ok] = np.where(den > 0, (r * c).sum(axis=1) / np.where(den > 0, den, 1.0), 0.0)
    return out


def pooled(per_fold: list[np.ndarray]) -> tuple[float, float]:
    """88.3 (22:28 amendment): the mean per-date rho20 over every validation date of every block; SE = SD / sqrt(n)."""
    x = np.concatenate(per_fold) if per_fold else np.zeros(0)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else -np.inf), 0.0
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def pick_1se(points: list[dict], simpler) -> dict | None:
    """Best score; within one SE of it, the simplest by ``simpler`` (a sort key, smaller = simpler)."""
    pts = [p for p in points if np.isfinite(p["score"])]
    if not pts:
        return None
    best = max(pts, key=lambda p: p["score"])
    near = [p for p in pts if p["score"] >= best["score"] - best["se"]]
    return {**min(near, key=simpler), "best": {k: best[k] for k in best if k != "fold"}}


# ------------------------------------------------------------------------------------------------ linear indicators


def _aligned(win: Window, stats: tuple) -> np.ndarray:
    return np.nan_to_num(win.X[:, : len(win.xs)], nan=0.0) * stats[0][None, :]


def nn_ridge(gram: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """argmin_{beta >= 0} beta' (G + lam I) beta - 2 b' beta (exact: NNLS on the Cholesky factor)."""
    from scipy.optimize import nnls

    a = gram + lam * np.eye(len(b))
    low = np.linalg.cholesky(a)
    beta, _ = nnls(low.T, np.linalg.solve(low, b), maxiter=50 * len(b))
    return beta


def _gram(xa: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = w / w.sum()
    xw = xa * w[:, None]
    return xw.T @ xa, xw.T @ y


def lin_weights(win: Window, stats: tuple, mode: str, tau: float, lam: float, gb=None) -> np.ndarray:
    """Weights on the sign-aligned inputs (zero outside the |t| >= tau set)."""
    keep = _kept(stats, tau)
    beta = np.zeros(len(win.xs))
    if not len(keep):
        return beta
    if mode == "eq":
        beta[keep] = 1.0 / len(keep)
        return beta
    gram, b = gb
    beta[keep] = nn_ridge(gram[np.ix_(keep, keep)], b[keep], lam)
    return beta


def choose_lin(mode: str, train: pd.DataFrame, ctx) -> dict:
    win = window(train, ctx)
    key = (mode, win.key)
    if key in _LIN:
        return _LIN[key]
    grid = [(t, lam) for t in TAUS for lam in (LAMS if mode == "ridge" else (0.0,))]
    per = {g: [] for g in grid}
    y, w = win.target["z"], win.w
    for (fit, val), stats in zip(win.folds, win.stats, strict=True):
        v = _Val(win, val)
        xa = _aligned(win, stats)
        ok = fit & np.isfinite(y)
        gb = _gram(xa[ok], y[ok], w[ok]) if mode == "ridge" else None
        xv = xa[v.rows]
        for tau, lam in grid:
            beta = lin_weights(win, stats, mode, tau, lam, gb)
            per[(tau, lam)].append(v.rho(xv @ beta) if beta.any() else np.zeros(v.n_dates))
    pts = []
    for (tau, lam), r in per.items():
        sc, se = pooled(r)
        pts.append({"tau": tau, "lam": lam, "score": sc, "se": se, "fold": [round(float(x.mean()), 3) for x in r]})
    pick = pick_1se(pts, lambda p: (-p["tau"], -p["lam"])) if win.folds else None
    out = pick or {"tau": 1.0, "lam": 10.0, "fallback": True}
    _LIN[key] = out
    return out


def make_lin(mode: str):
    def fit(c, train, params, scheme, ctx):
        win = window(train, ctx)
        ch = choose_lin(mode, train, ctx)
        y = win.target["z"]
        ok = np.isfinite(y)
        gb = _gram(_aligned(win, win.stats_full)[ok], y[ok], win.w[ok]) if mode == "ridge" else None
        beta = lin_weights(win, win.stats_full, mode, ch["tau"], ch["lam"], gb)
        weights = {col: float(s * b) for col, s, b in zip(win.xs, win.stats_full[0], beta, strict=True) if b != 0}
        info = {k: ch.get(k) for k in ("tau", "lam", "score", "se", "fold", "best", "fallback")}
        info.update(n_inputs=len(weights), blocks=len(win.folds))
        _log({"cand": c.name, "cutoff": _cut(ctx), "n_train": len(train), **info})
        return cmod.Fitted(c.name, params, scheme, None, list(win.xs), "weights", {"weights": weights, "v88": info})

    return fit


def _cut(ctx) -> str | None:
    return str(pd.Timestamp(ctx.cutoff).date()) if ctx.cutoff is not None else None


# ------------------------------------------------------------------------------------------------ monotone LightGBM


def _feat(win: Window, stats: tuple, tau: float, mono: bool) -> tuple[np.ndarray, list[int]]:
    """Column indices into ``win.X`` (kept xs, then every regime column) and their monotone constraints."""
    keep = _kept(stats, tau)
    idx = np.concatenate([keep, len(win.xs) + np.arange(len(win.reg))]).astype(int)
    cons = [int(stats[0][j]) if mono else 0 for j in keep] + [0] * len(win.reg)
    return idx, cons


def _native(n: int, ctx, cons: list[int]) -> dict:
    kw = cmod._lgbm_common(GBM, n, ctx)
    kw.pop("n_estimators")
    kw.update(objective="regression", monotone_constraints=cons, monotone_constraints_method="basic")
    return kw


def lockstep_vt(win: Window, target: str, tau: float, mono: bool, ctx) -> tuple[list[dict], list]:
    """87.1's lockstep boosting with the ventile score: one booster per annual fold, the pooled ventile score at
    every checkpoint, stop PATIENCE trees after the best. Returns the records and (columns, booster) per fold."""
    import lightgbm as lgb

    y = win.target[target]
    boosters, vals = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for (fit, val), stats in zip(win.folds, win.stats, strict=True):
            idx, cons = _feat(win, stats, tau, mono)
            ok = np.flatnonzero(fit & np.isfinite(y))
            v = _Val(win, val)
            if len(ok) < 100 or v.n_dates == 0:
                continue
            ds = lgb.Dataset(win.X[np.ix_(ok, idx)], y[ok], weight=win.w[ok], params={"verbose": -1})
            boosters.append((idx, lgb.Booster(params=_native(len(ok), ctx, cons), train_set=ds)))
            vals.append((v, win.X[np.ix_(v.rows, idx)]))
        recs: list[dict] = []
        best, best_n = -np.inf, 0
        for it in range(1, (v87.N_MAX + 1) if boosters else 1):
            for _, b in boosters:
                b.update()
            if it not in v87.CHECKS:
                continue
            per = [v.rho(b.predict(xv, raw_score=True)) for (_, b), (v, xv) in zip(boosters, vals, strict=True)]
            sc, se = pooled(per)
            recs.append({"n": it, "tau": tau, "score": sc, "se": se, "fold": [round(float(x.mean()), 3) for x in per]})
            if sc > best:
                best, best_n = sc, it
            elif it - best_n >= v87.PATIENCE:
                break
        for _, b in boosters:
            b.free_dataset()
    return recs, boosters


def choose_gbm(mode: str, train: pd.DataFrame, ctx) -> dict:
    win = window(train, ctx)
    key = (mode, win.key)
    if key in _GBM:
        return _GBM[key]
    target, mono = GBM_MODES[mode]
    pts, kept = [], {}
    for tau in TAUS:
        recs, boosters = lockstep_vt(win, target, tau, mono, ctx)
        pts += recs
        kept[tau] = boosters
    pick = pick_1se(pts, lambda p: (-p["tau"], p["n"])) if pts else None
    if pick is None:
        out = {"tau": 1.0, "n": v87.N_FALLBACK, "fallback": True, "boosters": []}
    else:
        out = {**pick, "n": max(v87.N_MIN, int(pick["n"]))}
        out["stopped"] = {str(t): max((p["n"] for p in pts if p["tau"] == t), default=None) for t in TAUS}
        out["boosters"] = kept[pick["tau"]] if mode == "vt_gbm" else []
    _GBM[key] = out
    return out


def _info(ch: dict, win: Window) -> dict:
    out = {k: ch.get(k) for k in ("tau", "n", "score", "se", "fold", "best", "stopped", "fallback")}
    out["blocks"] = len(win.folds)
    return out


def make_gbm(mode: str):
    def fit(c, train, params, scheme, ctx):
        import lightgbm as lgb

        win = window(train, ctx)
        ch = choose_gbm(mode, train, ctx)
        target, mono = GBM_MODES[mode]
        idx, cons = _feat(win, win.stats_full, ch["tau"], mono)
        y = win.target[target]
        ok = np.isfinite(y)
        cols = [(win.xs + win.reg)[i] for i in idx]
        kw = cmod._lgbm_common({**GBM, "n_estimators": ch["n"]}, int(ok.sum()), ctx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = lgb.LGBMRegressor(
                objective="regression", monotone_constraints=cons, monotone_constraints_method="basic", **kw
            ).fit(win.X[np.ix_(np.flatnonzero(ok), idx)], y[ok], sample_weight=win.w[ok])
        info = {**_info(ch, win), "n_inputs": len(idx) - len(win.reg)}
        info["signs"] = {cols[i]: cons[i] for i in range(len(cols)) if cons[i]}
        _log(
            {
                "cand": c.name,
                "cutoff": _cut(ctx),
                "n_train": len(train),
                **{k: v for k, v in info.items() if k != "signs"},
            }
        )
        return cmod.Fitted(c.name, params, scheme, m, cols, "lgbm", {"v88": info})

    return fit


@dataclass
class BagFitted(cmod.Fitted):
    """88.3 ``vt_gbm_bag``: the mean raw score of vt_gbm's annual-fold boosters at its chosen tree count."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.array([])
        names, n = self.extra["names"], self.extra["n"]
        x = df[names].to_numpy(dtype=np.float64)
        preds = [b.predict(x[:, idx], num_iteration=n, raw_score=True) for idx, b in self.extra["boosters"]]
        return np.mean(preds, axis=0)

    def importance(self) -> dict[str, float]:
        return {}

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame | None:
        return None


if __name__ == "__main__":
    BagFitted.__module__ = "v88_ventile"


def fit_bag(c, train, params, scheme, ctx):
    win = window(train, ctx)
    ch = choose_gbm("vt_gbm", train, ctx)
    if not ch["boosters"]:  # no folds: the full-window vt_gbm fit
        return make_gbm("vt_gbm")(c, train, params, scheme, ctx)
    info = {**_info(ch, win), "members": len(ch["boosters"])}
    _log({"cand": c.name, "cutoff": _cut(ctx), "n_train": len(train), **info})
    extra = {"names": win.xs + win.reg, "n": ch["n"], "boosters": ch["boosters"], "v88": info}
    return BagFitted(c.name, params, scheme, None, win.xs + win.reg, "bag", extra)


# ------------------------------------------------------------------------------------------------ candidates


def candidates() -> list[cmod.Candidate]:
    by = {c.name: c for c in cmod.default_candidates()}
    v87c = {c.name: c for c in v87.candidates()}  # registers v87's fitters (lgbm_reg_w_cv)
    out = [by["mom_12_1"], *(by[n] for n in REFS if n in by), v87c["lgbm_reg_w_cv"]]
    for n in NEW:
        if n.startswith("vt_lin"):
            cmod._FITTERS[n] = make_lin("eq" if n == "vt_lin_eq" else "ridge")
        elif n == "vt_gbm_bag":
            cmod._FITTERS[n] = fit_bag
        else:
            cmod._FITTERS[n] = make_gbm(n)
        out.append(
            cmod.Candidate(
                n,
                "linear" if n.startswith("vt_lin") else "tree",
                NEW_KO[n],
                f"v88 {n}",
                rows="W",
                tuned=False,
                fitter=n,
            )
        )
    return out


def patch_selection() -> None:
    """No composites / diagnostic pairs: the candidates are the pre-declared twelve (88.3)."""
    sel.COMPOSITE_CANDIDATES = ()
    sel.COMPOSITES = ()
    sel.DIAGNOSTIC_PAIRS = ()


# ------------------------------------------------------------------------------------------------ analysis


def insample(res, ds, names: list[str]) -> dict:
    """Final-context models on their own training month-ends: rho20 and IC (the in-sample gap)."""
    data = sel._Data(ds)
    _, _, final, _ = sel.make_contexts(data)
    rows = data.train(final.train_end, "M")
    out = {}
    for n in names:
        f = res.final.fitted.get(n) if res.final is not None else None
        if f is None:
            continue
        fr = rows[["date", "entity", "y_e", "r"]].assign(**{n: f.predict(rows)})
        dts, m = vmat(fr, n)
        out[n] = {"rho": rho((m - m.mean(axis=1, keepdims=True)).mean(axis=0)) if len(m) else None}
        out[n]["ic"] = float(mt.per_date_ic(fr, n).mean())
    return out


def choice_stats(path: Path, outer_starts: set[str]) -> dict:
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    seen, by = set(), {}
    for r in recs:
        k = (r["cand"], r["cutoff"], r["n_train"])
        if r["cutoff"] not in outer_starts or k in seen:
            continue
        seen.add(k)
        by.setdefault(r["cand"], []).append(r)
    out = {}
    for name, rr in by.items():
        out[name] = {
            "fits": len(rr),
            "tau": pd.Series([r.get("tau") for r in rr]).value_counts().sort_index().to_dict(),
            "lam": pd.Series([r.get("lam") for r in rr if r.get("lam") is not None]).value_counts().to_dict(),
            "n": [r.get("n") for r in rr if r.get("n") is not None],
            "inputs": [r.get("n_inputs") for r in rr if r.get("n_inputs") is not None],
            "blocks": [r.get("blocks") for r in rr],
            "fallback": sum(1 for r in rr if r.get("fallback")),
        }
    return out


def composition(res) -> dict:
    """The final (serving-window) indicators: vt_lin weights, vt_lin_eq inputs, vt_gbm constraint directions."""
    f = res.final.fitted if res.final is not None else {}
    out = {}
    for n in ("vt_lin", "vt_lin_eq"):
        if n in f:
            w = f[n].extra["weights"]
            out[n] = {"weights": dict(sorted(w.items(), key=lambda kv: -abs(kv[1]))), **f[n].extra["v88"]}
    if "vt_gbm" in f:
        out["vt_gbm"] = {k: v for k, v in f["vt_gbm"].extra["v88"].items() if k != "fold"}
        imp = f["vt_gbm"].importance()
        tot = sum(imp.values()) or 1.0
        out["vt_gbm"]["gain_share"] = {k: round(v / tot, 4) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:15]}
    return out


def _f(x, spec: str = "+.4f") -> str:
    return v86._f(x, spec)


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:+.2f}"


def write_md(m: str, r: dict) -> Path:
    gate, v = r["gate"], r["ventile"]
    lines = [
        f"# {m} 20분위 단조 모델 (docs/MODELING.md 88)",
        "",
        f"- 데이터셋 `{r['snapshot']}` (84.6 고정 자료, 갈래 A 입력), 20거래일 y_e, 곡선은 비용 전·날짜 평균을 뺀 값",
        f"- 20분위 모델 선정(88.4.3, 새 후보 중 외부 ρ20 1위): `{r['vt_pick']}`",
        f"- 돈 지표 선정(v4.3): `{r['selected']}` (게이트 {'통과' if gate.get('passed') else '실패'}: "
        f"값 {_f(gate.get('value'))}, 하한 {_f(gate.get('lo'))}, 모멘텀 {_f(gate.get('baseline'))})",
        f"- 87절 재현(`lgbm_reg_w_cv`·기준 후보 외부 1차 지표): {r['repro']}",
        "",
        "## 판정 1. 단조 통과 (외부 ρ20 하한 > 0, 외부 기울기 하한 > 0, 홀드아웃 ρ20 > 0, 홀드아웃 기울기 > 0; "
        "엄격 = + 5분위 MR p < 0.05)",
        "",
        "| 모델 | 외부 ρ20 [95%] | 날짜별 ρ20 (t) | D20−D1 맞춘 차이(월 %) [95%] t | 오른 칸 "
        "| MR p 20/5 · 5분위 최소 t | 연도 ρ20 양수/최소 | 홀드아웃 ρ20 | 홀드아웃 차이 | 외부 IC "
        "| 1차 지표 외부/홀드아웃 | 통과 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n, e in v.items():
        o, h = e.get("outer") or {}, e.get("holdout") or {}
        if not o:
            continue
        sp, hs = o["spread"], (h.get("spread") or {})
        t = r["table"].get(n, {})
        verdict = "**엄격 통과**" if e["strict"] else ("**통과**" if e["pass"] else "–")
        lines.append(
            f"| `{n}` | {o['rho']:+.3f} [{o['rho_lo']:+.3f}, {o['rho_hi']:+.3f}] | "
            f"{_f(o['rho_date']['mean'], '+.3f')} ({_f(o['rho_date']['t'], '.2f')}) | "
            f"{_pct(sp['mean'])} [{_pct(sp['lo'])}, {_pct(sp['hi'])}] {_f(sp['t'], '.2f')} | {o['ups']}/19 | "
            f"{o['mr20']['p']:.2f}/{o['mr5']['p']:.2f} · {o['mr5']['t_min']:+.2f} | "
            f"{o['years_pos']}/{len(o['years'])}, {o['years_min']:+.2f} | "
            f"{_f(h.get('rho'), '+.3f')} | {_pct(hs.get('mean'))} | {_f(t.get('outer_ic'), '+.3f')} | "
            f"{_f(t.get('outer'))} / {_f(t.get('holdout'))} | {verdict} |"
        )
    lines += ["", "## 판정 2·3. 대비와 선정 비교 (ρ20 차이, 짝지은 날짜 부트스트랩)", ""]
    lines += [
        "| 비교 | 외부 Δρ20 [95%] | 홀드아웃 Δρ20 | 외부 Δ날짜별 ρ20 (t) | 외부 Δ차이(월 %) t | 판정 |",
        "|---|---|---|---|---|---|",
    ]
    for k, e in r["judgments"].items():
        o, h = e.get("outer") or {}, e.get("holdout") or {}
        if not o:
            continue
        lines.append(
            f"| {k} | {o['d_rho']:+.3f} [{o['lo']:+.3f}, {o['hi']:+.3f}] | {_f(h.get('d_rho'), '+.3f')} | "
            f"{_f(o['d_rho_date']['mean'], '+.3f')} ({_f(o['d_rho_date']['t'], '.2f')}) | "
            f"{_pct(o['d_spread']['mean'])} {_f(o['d_spread']['t'], '.2f')} | "
            f"{'**도움**' if e['helps'] else '도움 없음'} |"
        )
    head = "| 모델 | 구간 | " + " | ".join(f"D{i}" for i in range(1, K + 1)) + " |"
    lines += ["", "## 20분위 곡선 (y_e, 날짜 평균 대비 월 %)", "", head, "|---|---|" + "---|" * K]
    for n, e in v.items():
        for part, ko in (("outer", "외부"), ("holdout", "홀드아웃")):
            c = (e.get(part) or {}).get("curve")
            if c:
                lines.append(f"| `{n}` | {ko} | " + " | ".join(f"{100 * x:+.2f}" for x in c) + " |")
    lines += ["", "## 5분위 곡선과 원 수익률(베타 조정 없음)", ""]
    lines += [
        "| 모델 | 외부 Q1..Q5 (월 %) | 홀드아웃 Q1..Q5 | 원 수익률 외부 ρ20 / 차이 | 연도별 ρ20 |",
        "|---|---|---|---|---|",
    ]
    for n, e in v.items():
        o, h, raw = e.get("outer") or {}, e.get("holdout") or {}, e.get("outer_raw") or {}
        if not o:
            continue
        q = " ".join(f"{100 * x:+.2f}" for x in o["q5"])
        qh = " ".join(f"{100 * x:+.2f}" for x in h.get("q5", []))
        yrs = ", ".join(f"{yy} {vv:+.2f}" for yy, vv in o["years"].items())
        lines.append(
            f"| `{n}` | {q} | {qh} | {_f(raw.get('rho'), '+.3f')} / "
            f"{_pct((raw.get('spread') or {}).get('mean'))} | {yrs} |"
        )
    lines += [
        "",
        "## 과적합 차이와 검증 선택 (외부 재학습)",
        "",
        "| 모델 | 학습 ρ20 | 외부 ρ20 | 학습 IC | 외부 IC | τ | λ | 나무 수 중앙값 | 입력 수 중앙값 |",
    ]
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for n, e in v.items():
        i, ch, t = r["insample"].get(n, {}), r["choices"].get(n, {}), r["table"].get(n, {})
        ns, ins = ch.get("n") or [], ch.get("inputs") or []
        lines.append(
            f"| `{n}` | {_f(i.get('rho'), '+.3f')} | {_f((e.get('outer') or {}).get('rho'), '+.3f')} | "
            f"{_f(i.get('ic'), '+.3f')} | {_f(t.get('outer_ic'), '+.3f')} | {ch.get('tau', '')} | "
            f"{ch.get('lam', '')} | "
            f"{np.median(ns) if ns else ''} | {np.median(ins) if ins else ''} |"
        )
    comp = r["composition"]
    lines += ["", "## 지표 구성 (마지막 학습 구간 = 서빙 구간)", ""]
    for n in ("vt_lin", "vt_lin_eq"):
        if n in comp:
            c = comp[n]
            ws = ", ".join(f"`{k}` {w:+.3f}" for k, w in list(c["weights"].items())[:20])
            lines.append(f"- `{n}` (τ {c.get('tau')}, λ {c.get('lam')}, 입력 {c.get('n_inputs')}개): {ws}")
    if "vt_gbm" in comp:
        c = comp["vt_gbm"]
        lines.append(
            f"- `vt_gbm` (τ {c.get('tau')}, 나무 {c.get('n')}, 입력 {c.get('n_inputs')}개) 기여 상위: "
            + ", ".join(f"`{k}` {w:.2f}" for k, w in c.get("gain_share", {}).items())
        )
    p = OUT / f"{m}_v88.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# ------------------------------------------------------------------------------------------------ run


def research_dir(s, m: str) -> Path:
    p = s.cache_path / "research" / "v88" / m / v86.SNAP[m][:12]
    p.mkdir(parents=True, exist_ok=True)
    return p


def smoke(m: str) -> None:
    """Every new fitter once on the last outer window: runs, choices, seconds (no out-of-sample result shown)."""
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
        if c.name not in NEW:
            continue
        t0 = time.monotonic()
        tr = data.train(ctx0.train_end, c.rows)
        fitted = c.fit(tr, {}, "none", sel._ctx(data, ctx0.test_dates[0], s, 8))
        p = fitted.predict(test)
        info = fitted.extra.get("v88") or {}
        brief = {k: info.get(k) for k in ("tau", "lam", "n", "n_inputs", "blocks", "stopped", "members", "fallback")}
        say(
            f"{c.name}: {len(tr)} rows, finite {np.isfinite(p).mean():.3f}, sd {np.nanstd(p):.4f}, {brief} "
            f"({time.monotonic() - t0:.1f}s)"
        )


def study(m: str) -> None:
    s = load_settings()
    cache = DiskCache(s.cache_path)
    ds = v86.load_frozen(m, s, cache)
    patch_selection()
    cands = candidates()
    rd = research_dir(s, m)
    LOG[:] = [rd / "choice_log.jsonl"]
    snap0 = ds.snapshot
    data = sel._Data(ds)
    outer, holdout, final, _ = sel.make_contexts(data)
    outer_starts = {str(pd.Timestamp(c.test_dates[0]).date()) for c in [*outer, holdout, final]}
    del data
    ds.snapshot = f"{snap0}|v88|{V88_VERSION}"
    say(f"{m}: {len(ds.xs_cols)} features + {len(ds.regime_cols)} regime, {len(cands)} candidates")
    res = sel.run_selection(ds, s, cache, candidates=cands, threads=8, use_cache=True, progress=say)
    d = res.to_dict()
    names = [c.name for c in cands]
    table = v86.table(res)
    v87_t = json.loads((OUT / f"{m}_v87.json").read_text())["table"]
    repro = {
        n: (round(table[n]["outer"], 6), round(v87_t[n]["outer"], 6)) for n in ("lgbm_reg_w_cv", *REFS) if n in v87_t
    }
    mats = {}
    ventile = {}
    for i, n in enumerate(names):
        mo, mh, mr = vmat(res.oos, n), vmat(res.holdout_preds, n), vmat(res.oos, n, "R")
        mats[n] = {"outer": mo, "holdout": mh}
        o, h = vstats(*mo, seed=10 * i), vstats(*mh, seed=10 * i + 5)
        ok = bool(o and h and o["rho_lo"] > 0 and o["spread"]["lo"] > 0 and h["rho"] > 0 and h["spread"]["mean"] > 0)
        ventile[n] = {
            "outer": o,
            "holdout": h,
            "outer_raw": {k: v for k, v in vstats(*mr, seed=10 * i + 7).items() if k in ("rho", "spread", "curve")},
            "pass": ok,
            "strict": bool(ok and o["mr5"]["p"] < 0.05 and min(np.diff(o["q5"])) > 0),
        }
    new_ok = [n for n in NEW if ventile[n]["outer"]]
    vt_pick = max(new_ok, key=lambda n: (ventile[n]["outer"]["rho"], ventile[n]["outer"]["spread"]["mean"]))
    pairs = {**CONTRASTS, **{f"선정 `{vt_pick}` − `{b}`": (vt_pick, b) for b in PICK_REFS}}
    judg = {}
    for j, (k, (a, b)) in enumerate(pairs.items()):
        o = paired(mats[a]["outer"], mats[b]["outer"], seed=1000 + 10 * j)
        h = paired(mats[a]["holdout"], mats[b]["holdout"], seed=1005 + 10 * j)
        judg[k] = {"a": a, "b": b, "outer": o, "holdout": h, "helps": bool(o and h and o["lo"] > 0 and h["d_rho"] >= 0)}
    r = {
        "market": m,
        "snapshot": snap0,
        "cost_bps": float(ds.cost_bps),
        "settings": {"k": K, "taus": TAUS, "lams": LAMS, "gbm": GBM, "n_max": v87.N_MAX, "patience": v87.PATIENCE},
        "selected": d["selected"],
        "gate": d["gate"],
        "table": table,
        "ventile": ventile,
        "vt_pick": vt_pick,
        "judgments": judg,
        "insample": insample(res, ds, names),
        "choices": choice_stats(LOG[0], outer_starts),
        "composition": composition(res),
        "repro": "같음" if all(abs(a - b) < 1e-9 for a, b in repro.values()) else f"다름 {repro}",
    }
    js = OUT / f"{m}_v88.json"
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
