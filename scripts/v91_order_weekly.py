"""Why does only the weekly mean-regression tree order the KR quantiles? (docs/MODELING.md 91, pre-declared
2026-11-05 after the 90 KR result). Research only: src/quantcio and the served models are not changed. KR only.

    .venv/bin/python scripts/v91_order_weekly.py smoke --market KR
    .venv/bin/python scripts/v91_order_weekly.py study --market KR

- ``vgw_up``: the 90 ordering-loss tree (``vg_up``) trained on the WEEKLY cross-sections (about 5x the month-end
  rows; validation still on the blocks' month-ends) - more data for the ordering loss.
- ``lgbm_reg_w_ord``: the weekly mean-regression tree (``lgbm_reg_w``, current grid point) with the tree count chosen by
  the 89 ordering loss on the pooled out-of-fold month-end curve (1-SE toward fewer trees) instead of 87's per-date IC.
Writes reports/modeling/<M>_v91.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v86_full_es as v86  # noqa: E402
import v87_validation as v87  # noqa: E402
import v88_ventile as v88  # noqa: E402
import v89_order_loss as v89  # noqa: E402
import v90_order_tree as v90  # noqa: E402

from quantcio.config import load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

if __name__ == "__main__":
    sys.modules.setdefault("v91_order_weekly", sys.modules[__name__])
    sys.modules.setdefault("v90_order_tree", v90)

OUT = v86.OUT
V91_VERSION = 1
K = v88.K
NEW = ("vgw_up", "lgbm_reg_w_ord")
NEW_KO = {"vgw_up": "순서 손실 나무(주간 행)", "lgbm_reg_w_ord": "주간 회귀 나무 + 순서 손실로 나무 수 선택"}
REFS = ("mom_12_1", "lgbm_reg_w_cv", "vg_up")
CONTRASTS = {  # 91.2
    "F1 순서 나무: 주간 − 월말": ("vgw_up", "vg_up"),
    "F2 주간 순서 나무 − `lgbm_reg_w_cv`": ("vgw_up", "lgbm_reg_w_cv"),
    "F3 회귀 나무: 순서 손실로 고른 나무 수 − IC로 고른 나무 수": ("lgbm_reg_w_ord", "lgbm_reg_w_cv"),
}
LOG: list[Path] = []
_WIN: dict = {}
_CHOICE: dict = {}

say = v86.say


def _log(rec: dict) -> None:
    if LOG:
        with LOG[0].open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")


class WWin(v90.TWin):
    """90's window on the weekly rows (+ the month-end flag, the per-date demeaned return and equal date weights)."""

    is_me: np.ndarray
    ydm: np.ndarray
    w: np.ndarray


def wwindow(train: pd.DataFrame, ctx) -> WWin:
    key = v88._wkey(train, ctx)
    if _WIN.get("key") == key:
        return _WIN["win"]
    _WIN.clear()
    d = train[np.isfinite(train["y_e"].to_numpy(dtype=float))]
    d = d[d.groupby("date")["y_e"].transform("size").to_numpy() >= v88.mt.MIN_NAMES].reset_index(drop=True)
    xs = [c for c in ctx.xs_cols if c in d and d[c].notna().any()]
    cols = cmod._tree_cols(ctx, d)
    codes, uniq = pd.factorize(d["date"], sort=True)
    y = d["y_e"].to_numpy(dtype=float)
    is_me = d["is_me"].to_numpy(bool)
    folds, _ = v88.year_masks(d["date"].to_numpy(dtype="datetime64[ns]"), is_me, ctx)
    cnt = np.bincount(codes)
    win = WWin(
        key,
        xs,
        cols,
        np.nan_to_num(d[xs].to_numpy(dtype=np.float64), nan=0.0),
        d[cols].to_numpy(dtype=np.float64),
        v89.zret(y, codes, len(uniq)),
        codes,
        folds,
    )
    win.is_me = is_me
    win.ydm = y - (np.bincount(codes, y) / cnt)[codes]
    w = 1.0 / cnt[codes]
    win.w = w / w.mean()
    _WIN.update(key=key, win=win)
    return win


def _val_rows(win: WWin, val: np.ndarray) -> tuple[np.ndarray, tuple]:
    rows = np.flatnonzero(val & win.is_me)
    return rows, pd.factorize(win.codes[rows], sort=True)


def _oof_add(oof: dict, it: int, win: WWin, rows: np.ndarray, vc: tuple, score: np.ndarray) -> None:
    m = v88.ventile_matrix(vc[0], len(vc[1]), score, win.r[rows])
    oof[it].append(m[np.isfinite(m).all(axis=1)])


# ------------------------------------------------------------------------------------------------ vgw_up


def choose_vgw(train: pd.DataFrame, ctx) -> dict:
    win = wwindow(train, ctx)
    key = ("vgw_up", win.key)
    if key in _CHOICE:
        return _CHOICE[key]
    oof: dict[int, list] = {n: [] for n in v90.T_CHECKS}
    for fit, val in win.folds:
        rows, vc = _val_rows(win, val)

        def at(it, bst, theta, rows=rows, vc=vc):
            _oof_add(oof, it, win, rows, vc, v90.tree_score(win, rows, theta, bst))

        v90.boost(win, fit, "up", max(v90.T_CHECKS), ctx, at)
    out = {**v90._pick(oof, "up", seed=0), "blocks": len(win.folds)}
    _CHOICE[key] = out
    return out


def fit_vgw(c, train, params, scheme, ctx):
    win = wwindow(train, ctx)
    ch = choose_vgw(train, ctx)
    theta, bst = v90.boost(win, np.ones(len(win.r), bool), "up", ch["n"], ctx)
    _log({"cand": c.name, "cutoff": v88._cut(ctx), "n_train": len(train), **ch})
    extra = {"xs": win.xs, "cols": win.cols, "theta": theta, "booster": bst, "v90": ch}
    return v90.OrderTreeFitted(c.name, params, scheme, None, list(win.cols), "order_tree", extra)


# ------------------------------------------------------------------------------------------------ lgbm_reg_w_ord

REG_CHECKS = tuple(sorted(v87.CHECKS))


def choose_reg(train: pd.DataFrame, ctx) -> dict:
    """Per annual fold the weekly regression booster (lgbm_reg_w's first grid point, target y_dm, equal date weights);
    the pooled out-of-fold month-end curve at every 87 checkpoint; L_up; 1-SE toward fewer trees (min 10)."""
    import lightgbm as lgb

    win = wwindow(train, ctx)
    key = ("lgbm_reg_w_ord", win.key)
    if key in _CHOICE:
        return _CHOICE[key]
    params = v87.rung_params("lgbm_reg_w", "cur")
    oof: dict[int, list] = {n: [] for n in REG_CHECKS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fit, val in win.folds:
            rows, vc = _val_rows(win, val)
            kw = v87._native("lgbm_reg_w", params, int(fit.sum()), ctx)
            ds = lgb.Dataset(win.XT[fit], win.ydm[fit], weight=win.w[fit], params={"verbose": -1})
            bst = lgb.Booster(params=kw, train_set=ds)
            for it in range(1, max(REG_CHECKS) + 1):
                bst.update()
                if it in REG_CHECKS:
                    _oof_add(oof, it, win, rows, vc, bst.predict(win.XT[rows], raw_score=True))
            bst.free_dataset()
    out = {**v90._pick(oof, "up", seed=0), "blocks": len(win.folds)}
    out["n"] = max(v87.N_MIN, int(out["n"]))
    _CHOICE[key] = out
    return out


def fit_reg(c, train, params, scheme, ctx):
    ch = choose_reg(train, ctx)
    p = {**v87.rung_params("lgbm_reg_w", "cur"), "n_estimators": ch["n"]}
    out = cmod._FITTERS["lgbm_reg_w"](c, train, p, "none", ctx)
    out.extra["v90"] = ch
    _log({"cand": c.name, "cutoff": v88._cut(ctx), "n_train": len(train), **ch})
    return out


def candidates() -> list[cmod.Candidate]:
    base = {c.name: c for c in v90.candidates()}  # registers 89 / 90 fitters (lgbm_reg_w_cv, vg_up, ...)
    out = [base[n] for n in REFS]
    cmod._FITTERS["vgw_up"] = fit_vgw
    cmod._FITTERS["lgbm_reg_w_ord"] = fit_reg
    for n in NEW:
        out.append(cmod.Candidate(n, "tree", NEW_KO[n], f"v91 {n}", rows="W", tuned=False, fitter=n))
    return out


# ------------------------------------------------------------------------------------------------ run


def research_dir(s, m: str) -> Path:
    p = s.cache_path / "research" / "v91" / m / v86.SNAP[m][:12]
    p.mkdir(parents=True, exist_ok=True)
    return p


def smoke(m: str) -> None:
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
            f"({time.monotonic() - t0:.0f}s)"
        )
        for pt in info.get("curve") or []:
            if pt["n"] in (*v90.T_CHECKS, 150, 250, 400, 600, 800, 1000):
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
    ds.snapshot = f"{snap0}|v91|{V91_VERSION}"
    say(f"{m}: {len(cands)} candidates")
    res = sel.run_selection(ds, s, cache, candidates=cands, threads=8, use_cache=True, progress=say)
    d = res.to_dict()
    names = [c.name for c in cands]
    table = v86.table(res)
    v90_t = json.loads((OUT / f"{m}_v90.json").read_text())["table"]
    repro = {n: (round(table[n]["outer"], 6), round(v90_t[n]["outer"], 6)) for n in REFS}
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
    for n in NEW:
        if n in f:
            imp = f[n].importance()
            tot = sum(imp.values()) or 1.0
            comp[n] = {k: v / tot for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}
    r = {
        "market": m,
        "tag": "v91",
        "title": f"# {m} 주간 행 순서 손실 나무 · 순서 손실로 고른 회귀 나무 (docs/MODELING.md 91)",
        "snapshot": snap0,
        "cost_bps": float(ds.cost_bps),
        "settings": {"t_checks": v90.T_CHECKS, "reg_checks": REG_CHECKS, "tree": v90.TREE, "tau": v90.TAU90},
        "selected": d["selected"],
        "gate": d["gate"],
        "table": table,
        "ventile": ventile,
        "judgments": judg,
        "insample": v89.insample(res, ds, names),
        "choices": v89.choice_stats(LOG[0], outer_starts),
        "composition": comp,
        "repro": "같음" if all(abs(a - b) < 1e-9 for a, b in repro.values()) else f"다름 {repro}",
    }
    js = OUT / f"{m}_v91.json"
    js.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    say(f"wrote {v89.write_md(m, r)} and {js}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["study", "smoke"])
    ap.add_argument("--market", choices=["KR"], required=True)
    a = ap.parse_args()
    if a.cmd == "smoke":
        smoke(a.market)
    else:
        study(a.market)


if __name__ == "__main__":
    main()
