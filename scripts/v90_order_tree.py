"""Ordering-loss trees and spline index (docs/MODELING.md 90, pre-declared 2026-11-05): the 89 ordering loss on
nonlinear models. Research only: src/quantcio and the served models are not changed.

    .venv/bin/python scripts/v90_order_tree.py smoke --market KR   # every new fitter once on the last outer window
    .venv/bin/python scripts/v90_order_tree.py study --market KR

- ``vg_up`` / ``vg_slope``: LightGBM boosted from the 89 start (the ridge index, standardised per date) with a custom
  objective - the per-row gradient of ``L_up`` / ``L_slope`` (torch autograd through the 89 straight-through curve),
  scaled to SD 1 per iteration, hessian 1; the tree count by the annual blocks of 87.3 (the same loss on the pooled
  out-of-fold curve, 1-SE toward fewer trees).
- ``vs_up``: the 89 ``L_up`` index on a spline basis (x and max(0, x - k), k = -0.84 / 0 / 0.84 per input).
Writes reports/modeling/<M>_v90.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v86_full_es as v86  # noqa: E402
import v88_ventile as v88  # noqa: E402
import v89_order_loss as v89  # noqa: E402

from quantcio.config import load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

if __name__ == "__main__":  # checkpointed fitted objects pickle under the module name, loadable from other scripts
    sys.modules.setdefault("v90_order_tree", sys.modules[__name__])

OUT = v86.OUT
V90_VERSION = 1
K = v88.K
TREE = {"leaves": 7, "max_depth": 3, "leaf_frac": 0.05, "l2": 30.0, "feature_frac": 0.5, "bagging": 0.7, "lr": 0.05}
T_CHECKS = (0, 10, 25, 50, 100, 200, 300, 500)
KNOTS = (-0.84, 0.0, 0.84)
TAU90 = 1.0 / K  # 90.3 (amended before the run): gradient-path edge width of the new models (89: 1/80)
TREE_LOSS = {"vg_up": "up", "vg_slope": "slope"}
NEW = ("vg_up", "vg_slope", "vs_up", "vl_up")
NEW_KO = {
    "vg_up": "순서 손실 나무(오른 칸 수)",
    "vg_slope": "기울기 손실 나무",
    "vs_up": "순서 손실 스플라인 지표",
    "vl_up": "순서 손실 선형 지표(부드러운 경계)",
}
CONTRASTS = {  # 90.2
    "E1 순서 나무 − 선형 시작점": ("vg_up", "vo_start"),
    "E2 순서 나무 − `lgbm_reg_w_cv`": ("vg_up", "lgbm_reg_w_cv"),
    "E3 나무: 순서 손실 − 기울기 손실": ("vg_up", "vg_slope"),
    "E4 스플라인 − 선형(같은 손실·경계)": ("vs_up", "vl_up"),
    "E5 나무 − 스플라인(둘 다 순서 손실)": ("vg_up", "vs_up"),
}
LOG: list[Path] = []
_WIN: dict = {}
_CHOICE: dict = {}

say = v86.say


def _log(rec: dict) -> None:
    if LOG:
        with LOG[0].open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")


# ------------------------------------------------------------------------------------------------ data


def zscore(x: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Per-date standardisation of a score (rank-preserving; the scale the trees add to)."""
    return v89.zret(x, codes, int(codes.max()) + 1 if len(codes) else 0)


def basis(x: np.ndarray) -> np.ndarray:
    """90.1 spline basis per input: x, max(0, x - k) for every knot."""
    return np.concatenate([x, *(np.maximum(x - k, 0.0) for k in KNOTS)], axis=1)


@dataclass
class TWin:
    """One training window of month-end rows: linear inputs (NaN = 0), tree inputs (xs + regime, NaN kept),
    standardised return, date codes, annual folds."""

    key: tuple
    xs: list[str]
    cols: list[str]
    XL: np.ndarray
    XT: np.ndarray
    r: np.ndarray
    codes: np.ndarray
    folds: list


def twindow(train: pd.DataFrame, ctx) -> TWin:
    key = v88._wkey(train, ctx)
    if _WIN.get("key") == key:
        return _WIN["win"]
    _WIN.clear()
    mw = v89.mwindow(train, ctx)  # the same month-end rows / standardised return / folds as 89
    d = train[train["is_me"].to_numpy(bool) & np.isfinite(train["y_e"].to_numpy(dtype=float))]
    d = d[d.groupby("date")["y_e"].transform("size").to_numpy() >= v88.mt.MIN_NAMES].reset_index(drop=True)
    cols = cmod._tree_cols(ctx, d)
    win = TWin(key, mw.xs, cols, mw.X, d[cols].to_numpy(dtype=np.float64), mw.r, mw.codes, mw.folds)
    _WIN.update(key=key, win=win)
    return win


# ---------------------------------------------------------------------------------------------- ordering-loss boosting


class Grad:
    """The custom objective of one booster: per-row gradient of the ordering loss of its current scores."""

    def __init__(self, r: np.ndarray, codes: np.ndarray, kind: str, tau: float = TAU90):
        self.b = v89.Batch(np.zeros((len(r), 1)), r, codes)
        self.kind, self.tau = kind, tau
        inv = np.argsort(self.b.order, kind="stable")  # input row -> position in the date-sorted order
        self.cs, self.pos = self.b.cs[inv], self.b.pos[inv]  # input row -> padded (date, name) cell

    def __call__(self, preds: np.ndarray, _ds) -> tuple[np.ndarray, np.ndarray]:
        torch = v89._torch()
        b, cs, pos = self.b, self.cs, self.pos
        s = torch.zeros(b.mask.shape, dtype=torch.float64)
        s[cs, pos] = torch.from_numpy(np.asarray(preds, dtype=np.float64))
        s.requires_grad_(True)
        loss = v89.loss_value(v89.curve_scores(s, b, self.tau), self.kind, torch)
        loss.backward()
        g = s.grad.numpy()[cs, pos]
        sd = g.std()
        return (g / sd if sd > 0 else g), np.ones_like(g)


def _params(n_rows: int, ctx) -> dict:
    return {
        "objective": "none",
        "learning_rate": TREE["lr"],
        "num_leaves": TREE["leaves"],
        "max_depth": TREE["max_depth"],
        "min_data_in_leaf": cmod._frac_leaf(n_rows, TREE["leaf_frac"]),
        "lambda_l2": TREE["l2"],
        "feature_fraction": TREE["feature_frac"],
        "bagging_fraction": TREE["bagging"],
        "bagging_freq": 1,
        "seed": int(ctx.seed),
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": int(ctx.threads),
        "verbose": -1,
    }


def _start(xl: np.ndarray, r: np.ndarray, codes: np.ndarray) -> np.ndarray:
    return v89.ridge_start(xl, r, codes)


def boost(win: TWin, fit: np.ndarray, kind: str, n_max: int, ctx, at=None):
    """One booster on the rows ``fit`` from the standardised ridge start; ``at(it, booster, theta)`` after every
    iteration in T_CHECKS (and at 0). Returns (theta, booster)."""
    import lightgbm as lgb

    theta = _start(win.XL[fit], win.r[fit], win.codes[fit])
    init = zscore(win.XL[fit] @ theta, pd.factorize(win.codes[fit], sort=True)[0])
    if at is not None:
        at(0, None, theta)
    if n_max == 0:
        return theta, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = lgb.Dataset(win.XT[fit], win.r[fit], init_score=init, params={"verbose": -1}, free_raw_data=False)
        bst = lgb.Booster(params=_params(int(fit.sum()), ctx), train_set=ds)
        obj = Grad(win.r[fit], win.codes[fit], kind)
        for it in range(1, n_max + 1):
            bst.update(fobj=obj)
            if at is not None and it in T_CHECKS:
                at(it, bst, theta)
        bst.free_dataset()
    return theta, bst


def tree_score(win: TWin, rows: np.ndarray, theta: np.ndarray, bst, n: int | None = None) -> np.ndarray:
    codes = pd.factorize(win.codes[rows], sort=True)[0]
    s = zscore(win.XL[rows] @ theta, codes)
    if bst is not None and (n is None or n > 0):
        s = s + bst.predict(win.XT[rows], num_iteration=n, raw_score=True)
    return s


def _pick(oof: dict, kind: str, seed: int) -> dict:
    """89.2's rule on the pooled out-of-fold curves {steps: [fold matrices]}."""
    pts = []
    for n, mats in sorted(oof.items()):
        m = np.concatenate(mats) if mats else np.zeros((0, K))
        if len(m) < 3:
            continue
        lv = float(v89.loss_value(m, kind, np))
        se = float(v89._boot_loss(m, kind, seed=seed + n, n_boot=v89.VAL_BOOT).std(ddof=1))
        pts.append({"n": n, "score": -lv, "se": se})
    pick = v88.pick_1se(pts, lambda p: p["n"]) if pts else None
    out = {"n": int(pick["n"]), **{k: pick[k] for k in ("score", "se", "best")}} if pick else {"n": 0, "fallback": True}
    out["curve"] = [{k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()} for p in pts]
    return out


def choose_tree(mode: str, train: pd.DataFrame, ctx) -> dict:
    win = twindow(train, ctx)
    key = (mode, win.key)
    if key in _CHOICE:
        return _CHOICE[key]
    kind = TREE_LOSS[mode]
    oof: dict[int, list] = {n: [] for n in T_CHECKS}
    for fit, val in win.folds:
        vrows = np.flatnonzero(val)
        vc = pd.factorize(win.codes[vrows], sort=True)

        def at(it, bst, theta, vrows=vrows, vc=vc):
            m = v88.ventile_matrix(vc[0], len(vc[1]), tree_score(win, vrows, theta, bst), win.r[vrows])
            oof[it].append(m[np.isfinite(m).all(axis=1)])

        boost(win, fit, kind, max(T_CHECKS), ctx, at)
    out = {**_pick(oof, kind, seed=0), "blocks": len(win.folds)}
    _CHOICE[key] = out
    return out


@dataclass
class OrderTreeFitted(cmod.Fitted):
    """90.1 ``vg_*``: the per-date standardised ridge index + the ordering-loss trees."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.array([])
        e = self.extra
        codes = pd.factorize(df["date"], sort=True)[0]
        x = np.nan_to_num(df[e["xs"]].to_numpy(dtype=np.float64), nan=0.0)
        s = zscore(x @ e["theta"], codes)
        if e["booster"] is not None:
            s = s + e["booster"].predict(df[e["cols"]].to_numpy(dtype=np.float64), raw_score=True)
        return s

    def importance(self) -> dict[str, float]:
        b = self.extra["booster"]
        if b is None:
            return {}
        return {c: float(v) for c, v in zip(self.extra["cols"], b.feature_importance("gain"), strict=True)}

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame | None:
        return None


@dataclass
class SplineFitted(cmod.Fitted):
    """90.1 ``vs_up`` / ``vl_up``: weights on the spline basis of the inputs / on the inputs (NaN = 0)."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.array([])
        x = np.nan_to_num(df[self.extra["xs"]].to_numpy(dtype=np.float64), nan=0.0)
        return (basis(x) if self.extra["spline"] else x) @ self.extra["theta"]

    def importance(self) -> dict[str, float]:
        th, xs = self.extra["theta"], self.extra["xs"]
        p = len(xs)
        return {c: float(np.abs(th[j::p]).sum()) for j, c in enumerate(xs)}

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame | None:
        return None


if __name__ == "__main__":
    OrderTreeFitted.__module__ = SplineFitted.__module__ = "v90_order_tree"


def make_tree(mode: str):
    def fit(c, train, params, scheme, ctx):
        win = twindow(train, ctx)
        ch = choose_tree(mode, train, ctx)
        allrows = np.ones(len(win.r), bool)
        theta, bst = boost(win, allrows, TREE_LOSS[mode], ch["n"], ctx)
        _log({"cand": c.name, "cutoff": v88._cut(ctx), "n_train": len(train), **ch})
        extra = {"xs": win.xs, "cols": win.cols, "theta": theta, "booster": bst, "v90": ch}
        return OrderTreeFitted(c.name, params, scheme, None, list(win.cols), "order_tree", extra)

    return fit


def _lin_x(mode: str, x: np.ndarray) -> np.ndarray:
    return basis(x) if mode == "vs_up" else x


def choose_lin(mode: str, train: pd.DataFrame, ctx) -> dict:
    """``vs_up`` (spline basis) / ``vl_up`` (the inputs): 89.2 with the 90 gradient edge width."""
    win = twindow(train, ctx)
    key = (mode, win.key)
    if key in _CHOICE:
        return _CHOICE[key]
    xb = _lin_x(mode, win.XL)
    oof: dict[int, list] = {n: [] for n in v89.CHECKS}
    for fit, val in win.folds:
        snaps, _ = v89.train_path(xb[fit], win.r[fit], win.codes[fit], "up", v89.CHECKS, tau=TAU90)
        for n in v89.CHECKS:
            oof[n].append(v89._vm(xb[val], win.r[val], win.codes[val], snaps[n]))
    out = {**_pick(oof, "up", seed=0), "blocks": len(win.folds)}
    _CHOICE[key] = out
    return out


def make_lin(mode: str):
    def fit(c, train, params, scheme, ctx):
        win = twindow(train, ctx)
        ch = choose_lin(mode, train, ctx)
        xb = _lin_x(mode, win.XL)
        snaps, _ = v89.train_path(xb, win.r, win.codes, "up", (ch["n"],) if ch["n"] else (1,), tau=TAU90)
        _log({"cand": c.name, "cutoff": v88._cut(ctx), "n_train": len(train), **ch})
        extra = {"xs": win.xs, "theta": snaps[ch["n"]], "spline": mode == "vs_up", "v90": ch}
        return SplineFitted(c.name, params, scheme, None, list(win.xs), "spline", extra)

    return fit


def candidates() -> list[cmod.Candidate]:
    base = {c.name: c for c in v89.candidates()}  # mom_12_1, lgbm_rank, lgbm_reg_w_cv, vo_* (registers their fitters)
    out = [base[n] for n in ("mom_12_1", "lgbm_rank", "lgbm_reg_w_cv", "vo_start", "vo_up")]
    for n in NEW:
        cmod._FITTERS[n] = make_lin(n) if n in ("vs_up", "vl_up") else make_tree(n)
        fam = "linear" if n in ("vs_up", "vl_up") else "tree"
        out.append(cmod.Candidate(n, fam, NEW_KO[n], f"v90 {n}", rows="M", tuned=False, fitter=n))
    return out


# ------------------------------------------------------------------------------------------------ run


def research_dir(s, m: str) -> Path:
    p = s.cache_path / "research" / "v90" / m / v86.SNAP[m][:12]
    p.mkdir(parents=True, exist_ok=True)
    return p


def smoke(m: str) -> None:
    """Every new fitter once on the last outer window: runs, chosen trees / steps, seconds (no out-of-sample result)."""
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
        info = fitted.extra.get("v90") or {}
        say(
            f"{c.name}: {len(tr)} rows, finite {np.isfinite(p).mean():.3f}, n {info.get('n')} "
            f"({time.monotonic() - t0:.1f}s)"
        )
        for pt in info.get("curve") or []:
            say(f"   {pt}")


def study(m: str) -> None:
    s = load_settings()
    cache = DiskCache(s.cache_path)
    ds = v86.load_frozen(m, s, cache)
    v88.patch_selection()
    cands = candidates()
    rd = research_dir(s, m)
    LOG[:] = [rd / "choice_log.jsonl"]
    snap0 = ds.snapshot
    data = sel._Data(ds)
    outer, holdout, final, _ = sel.make_contexts(data)
    outer_starts = {str(pd.Timestamp(c.test_dates[0]).date()) for c in [*outer, holdout, final]}
    del data
    ds.snapshot = f"{snap0}|v90|{V90_VERSION}"
    say(f"{m}: {len(ds.xs_cols)} features + {len(ds.regime_cols)} regime, {len(cands)} candidates")
    res = sel.run_selection(ds, s, cache, candidates=cands, threads=8, use_cache=True, progress=say)
    d = res.to_dict()
    names = [c.name for c in cands]
    table = v86.table(res)
    v89_t = json.loads((OUT / f"{m}_v89.json").read_text())["table"]
    refs = ("mom_12_1", "lgbm_rank", "lgbm_reg_w_cv", "vo_start", "vo_up")
    repro = {n: (round(table[n]["outer"], 6), round(v89_t[n]["outer"], 6)) for n in refs}
    mats, ventile = {}, {}
    for i, n in enumerate(names):
        mo, mh = v88.vmat(res.oos, n), v88.vmat(res.holdout_preds, n)
        zo, zh = v89.zmat(res.oos, n), v89.zmat(res.holdout_preds, n)
        mats[n] = {"outer": mo, "holdout": mh, "z_outer": zo, "z_holdout": zh}
        o, h = v88.vstats(*mo, seed=10 * i), v88.vstats(*mh, seed=10 * i + 5)
        ok = bool(o and h and o["rho_lo"] > 0 and o["spread"]["lo"] > 0 and h["rho"] > 0 and h["spread"]["mean"] > 0)
        ventile[n] = {
            "outer": o,
            "holdout": h,
            "loss_outer": v89.loss_stats(zo, 10 * i + 7),
            "loss_holdout": v89.loss_stats(zh, 10 * i + 8),
            "pass": ok,
            "strict": bool(ok and o["mr5"]["p"] < 0.05 and min(np.diff(o["q5"])) > 0),
        }
    judg = {}
    for j, (k, (a, b)) in enumerate(CONTRASTS.items()):
        o = v88.paired(mats[a]["outer"], mats[b]["outer"], seed=1000 + 10 * j)
        h = v88.paired(mats[a]["holdout"], mats[b]["holdout"], seed=1005 + 10 * j)
        judg[k] = {
            "a": a,
            "b": b,
            "outer": o,
            "holdout": h,
            "loss_outer": v89.paired_loss(mats[a]["z_outer"], mats[b]["z_outer"], seed=2000 + 10 * j),
            "loss_holdout": v89.paired_loss(mats[a]["z_holdout"], mats[b]["z_holdout"], seed=2005 + 10 * j),
            "helps": bool(o and h and o["lo"] > 0 and h["d_rho"] >= 0),
        }
    f = res.final.fitted if res.final is not None else {}
    comp = {}
    for n in (*NEW, "vo_up"):
        if n in f:
            imp = f[n].importance() if n != "vo_up" else {k: abs(v) for k, v in f[n].extra["weights"].items()}
            tot = sum(imp.values()) or 1.0
            comp[n] = {k: v / tot for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}
    choices = v89.choice_stats(LOG[0], outer_starts)
    r = {
        "market": m,
        "tag": "v90",
        "title": f"# {m} 순서 손실 나무·스플라인 지표 (docs/MODELING.md 90)",
        "snapshot": snap0,
        "cost_bps": float(ds.cost_bps),
        "settings": {"k": K, "tree": TREE, "t_checks": T_CHECKS, "knots": KNOTS, "spline_checks": v89.CHECKS},
        "selected": d["selected"],
        "gate": d["gate"],
        "table": table,
        "ventile": ventile,
        "judgments": judg,
        "insample": v89.insample(res, ds, names),
        "choices": choices,
        "composition": comp,
        "repro": "같음" if all(abs(a - b) < 1e-9 for a, b in repro.values()) else f"다름 {repro}",
    }
    js = OUT / f"{m}_v90.json"
    js.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    say(f"wrote {v89.write_md(m, r)} and {js}")


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
