"""Soft Decision Distribution Forest as an RSMT-Q risk-distribution candidate (docs/MODELING.md 60, pre-declared
2026-10-14). Research only - needs the optional ``research`` extra (torch).

A soft decision tree routes every row to EVERY leaf with a probability (sigmoid gates on standardised features,
heap-indexed paths); each leaf holds 19 monotone quantiles (lowest level + cumulative softplus increments), and the
row's quantiles are the probability-weighted mix (monotone by construction). Trees are trained with the pinball loss
on the 19 levels and averaged in a forest (seeds, bootstrap rows, feature subspaces). Scored against the served RSMT-Q
(P) and the 59 randomized distribution trees (E2) on the SAME test rows as docs/MODELING.md 34 / 59.

    .venv/bin/python scripts/soft_forest_study.py --market KR

Writes reports/rsmt/soft_dist_<M>.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rsmt_et_study as ets  # noqa: E402
import rsmt_np_study as nps  # noqa: E402

LEVELS = nps.LEVELS
VARIANTS = {
    # S1: the proposal's settings + the mandatory fixes (correct routing, monotone leaves, standardised inputs)
    "S1": {
        "depth": 3,
        "trees": 5,
        "frac": 1.0,
        "bootstrap": False,
        "batch": None,
        "epochs": 100,
        "wd": 0.0,
        "early_stop": False,
    },
    # S2: reinforced - deeper, a real forest with et-style randomness, mini-batches, weight decay, early stopping
    "S2": {
        "depth": 4,
        "trees": 10,
        "frac": 0.7,
        "bootstrap": True,
        "batch": 4096,
        "epochs": 40,
        "wd": 1e-4,
        "early_stop": True,
    },
}
LR = 0.01
VAL_FRAC = 0.15
GAP_SESSIONS = 20
PATIENCE = 4
NAMES = ("P", "E2", *VARIANTS)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================================================ model


def soft_tree_class():
    import torch
    from torch import nn

    class SoftTree(nn.Module):
        def __init__(self, n_in: int, depth: int, init_q: np.ndarray):
            super().__init__()
            k = 2**depth
            self.depth = depth
            self.gate = nn.Linear(n_in, k - 1)
            nn.init.normal_(self.gate.weight, std=0.1)
            nn.init.zeros_(self.gate.bias)
            inc = np.maximum(np.diff(init_q), 1e-6)
            inv_softplus = np.log(np.expm1(inc))
            self.base = nn.Parameter(torch.full((k,), float(init_q[0])))
            self.inc = nn.Parameter(torch.tensor(np.tile(inv_softplus, (k, 1)), dtype=torch.float32))
            idx = np.zeros((depth, k), dtype=np.int64)
            left = np.zeros((depth, k), dtype=bool)
            for leaf in range(k):  # heap indexing: node n -> children 2n+1 (left), 2n+2 (right)
                node = 0
                for d in range(depth):
                    bit = (leaf >> (depth - 1 - d)) & 1
                    idx[d, leaf], left[d, leaf] = node, bit == 0
                    node = 2 * node + 1 + bit
            self.register_buffer("idx", torch.tensor(idx))
            self.register_buffer("left", torch.tensor(left))

        def leaf_quantiles(self):
            steps = torch.nn.functional.softplus(self.inc)
            return self.base[:, None] + torch.cat([torch.zeros_like(self.base[:, None]), steps.cumsum(1)], dim=1)

        def forward(self, x):
            g = torch.sigmoid(self.gate(x))
            p = torch.ones(x.shape[0], self.idx.shape[1], dtype=x.dtype, device=x.device)
            for d in range(self.depth):
                gd = g[:, self.idx[d]]
                p = p * torch.where(self.left[d], gd, 1.0 - gd)
            return p @ self.leaf_quantiles(), p

    return SoftTree


def _pinball(q, y, tau):
    e = y[:, None] - q
    return (tau * e).maximum((tau - 1.0) * e).mean()


def fit_tree(xf, yf, xv, yv, cfg: dict, seed: int, init_q: np.ndarray):
    import torch

    torch.manual_seed(seed)
    model = soft_tree_class()(xf.shape[1], cfg["depth"], init_q)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=cfg["wd"])
    tau = torch.tensor(LEVELS, dtype=torch.float32)
    xf_t, yf_t = torch.tensor(xf, dtype=torch.float32), torch.tensor(yf, dtype=torch.float32)
    xv_t = torch.tensor(xv, dtype=torch.float32) if xv is not None else None
    yv_t = torch.tensor(yv, dtype=torch.float32) if yv is not None else None
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad = np.inf, None, 0
    for _ in range(cfg["epochs"]):
        model.train()
        if cfg["batch"]:
            perm = torch.randperm(len(yf_t), generator=gen)
            for i in range(0, len(perm), cfg["batch"]):
                b = perm[i : i + cfg["batch"]]
                opt.zero_grad()
                loss = _pinball(model(xf_t[b])[0], yf_t[b], tau)
                loss.backward()
                opt.step()
        else:
            opt.zero_grad()
            loss = _pinball(model(xf_t)[0], yf_t, tau)
            loss.backward()
            opt.step()
        if cfg["early_stop"] and xv_t is not None:
            model.eval()
            with torch.no_grad():
                v = float(_pinball(model(xv_t)[0], yv_t, tau))
            if v < best - 1e-7:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def soft_forest_quantiles(trn: pd.DataFrame, te: pd.DataFrame, feats: list[str], seed: int, cfg: dict):
    """(19 quantiles per test row, mean leaf-weight entropy) of one soft decision distribution forest."""
    import torch

    torch.set_num_threads(8)
    d = pd.to_datetime(trn["date"])
    keep = set(np.sort(d.unique())[:: nps.QR_STEP])
    t = trn[d.isin(keep).to_numpy()]
    t = t[np.isfinite(t["r"].to_numpy(dtype=float))]
    dates = np.sort(pd.to_datetime(t["date"]).unique())
    if cfg["early_stop"]:
        v0 = dates[int(len(dates) * (1 - VAL_FRAC))]
        cal = np.sort(d.unique())
        fit_end = cal[max(np.searchsorted(cal, v0) - GAP_SESSIONS, 0)]
        fit_m = (pd.to_datetime(t["date"]) < fit_end).to_numpy()
        val_m = (pd.to_datetime(t["date"]) >= v0).to_numpy()
    else:
        fit_m, val_m = np.ones(len(t), bool), np.zeros(len(t), bool)
    x_all = t[feats].to_numpy(dtype="float64")
    mu = np.nanmean(x_all[fit_m], axis=0)
    sd = np.nanstd(x_all[fit_m], axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)

    def prep(x):
        z = (x - mu) / sd
        return np.clip(np.nan_to_num(z, nan=0.0), -5, 5).astype(np.float32)

    x, y = prep(x_all), t["r"].to_numpy(dtype=np.float32)
    xt = prep(te[feats].to_numpy(dtype="float64"))
    init_q = np.quantile(y[fit_m], LEVELS)
    rng = np.random.default_rng(seed)
    qs, ent = [], []
    for m in range(cfg["trees"]):
        cols = np.sort(rng.choice(x.shape[1], max(1, int(round(cfg["frac"] * x.shape[1]))), replace=False))
        fi = np.where(fit_m)[0]
        if cfg["bootstrap"]:
            fi = rng.choice(fi, len(fi), replace=True)
        vi = np.where(val_m)[0]
        model = fit_tree(
            x[fi][:, cols],
            y[fi],
            x[vi][:, cols] if len(vi) else None,
            y[vi] if len(vi) else None,
            cfg,
            seed * 100 + m,
            init_q,
        )
        with torch.no_grad():
            q, p = model(torch.tensor(xt[:, cols]))
        qs.append(q.numpy())
        pn = p.numpy()
        ent.append(float(-(pn * np.log(np.clip(pn, 1e-12, None))).sum(1).mean()))
    return np.sort(np.mean(qs, axis=0), axis=1), float(np.mean(ent))


# ================================================================================================ study


def summarise(rows: pd.DataFrame) -> dict:
    out = {"n": int(len(rows)), "dates": int(rows["date"].nunique())}
    for v in NAMES:
        n_exc = int(rows[f"{v}_exc95"].sum())
        out[v] = {
            "qs": float(rows[f"{v}_qs"].mean()),
            "exc95": float(rows[f"{v}_exc95"].mean()),
            "kupiec_p": nps.kupiec(n_exc, len(rows)),
            "cover80": float(rows[f"{v}_in80"].mean()),
        }
    for v in NAMES[1:]:
        m, lo, hi = nps.block_ci((rows[f"{v}_qs"] - rows["P_qs"]).to_numpy(), rows["date"].to_numpy())
        out[f"{v}_minus_P"] = {"mean": m, "lo": lo, "hi": hi, "rel": m / out["P"]["qs"]}
    return out


def run(market: str) -> dict:
    import quantcio.quant.rsmt as R
    from quantcio import holdout
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache
    from quantcio.models import Market
    from quantcio.pipeline import training_panel
    from quantcio.quant.panel import build_panel

    t_all = time.perf_counter()
    s = load_settings()
    cache = DiskCache(s.cache_path)
    say(f"{market}: panel")
    price, _, _ = training_panel(Market(market), s, cache)
    price, _ = holdout.guard_panel(price, s)
    panel = build_panel(price, s)
    tr = panel.train
    feats = [c for c in tr.columns if c not in nps.NON_FEATURES and pd.api.types.is_numeric_dtype(tr[c])]
    grid_dates = pd.DatetimeIndex(sorted(tr.loc[tr["on_grid"], "date"].unique()))
    first_hold = grid_dates[grid_dates > grid_dates[-1] - pd.DateOffset(months=nps.HOLDOUT_MONTHS)][0]
    pre = tr[(pd.to_datetime(tr["t_end"]) < first_hold).to_numpy()].reset_index(drop=True)
    say(f"{market}: {len(tr)} rows, {len(feats)} features, holdout from {first_hold.date()}")
    entropy: dict[str, list[float]] = {v: [] for v in VARIANTS}

    def fold_rows(trn, te, seed):
        model = R.RSMTree.from_settings(s, panel.market).fit_frame(trn)
        p = nps.model_quantiles(model, panel, s, trn, te)
        r = te["r"].to_numpy(dtype="float64")
        out = pd.DataFrame({"date": pd.to_datetime(te["date"]).to_numpy(), "ticker": te["ticker"].to_numpy(), "r": r})
        qs = {"P": p["grid"], "E2": ets.et_quantiles(trn, te, feats, seed, ets.VARIANTS["E2"])[0]}
        for v, cfg in VARIANTS.items():
            qs[v], e = soft_forest_quantiles(trn, te, feats, seed, cfg)
            entropy[v].append(e)
        for name, q in qs.items():
            for k, val in nps.scores(r, q).items():
                out[f"{name}_{k}"] = val
        return out

    res: dict = {"market": market, "holdout_start": str(first_hold.date()), "variants": VARIANTS, "cv": {}}
    for k in nps.LAYOUTS:
        parts = []
        for fold in R.cv_splits(pre, k):
            trn, te = pre.iloc[fold.train_idx], pre.iloc[fold.test_idx]
            if len(te) == 0 or trn["on_grid"].sum() < 60:
                continue
            t0 = time.perf_counter()
            parts.append(fold_rows(trn, te, seed=100 * k + fold.fold))
            say(f"{market} layout {k} fold {fold.fold}: {len(te)} test rows, {time.perf_counter() - t0:.0f}s")
        res["cv"][str(k)] = summarise(pd.concat(parts, ignore_index=True))
    d = pd.to_datetime(tr["date"])
    trn = tr[((d < first_hold) & (pd.to_datetime(tr["t_end"]) < first_hold)).to_numpy()]
    te = tr[((d >= first_hold) & tr["on_grid"].to_numpy(dtype=bool)).to_numpy() & tr["r"].notna().to_numpy()]
    res["holdout"] = summarise(fold_rows(trn, te, seed=999))
    res["leaf_entropy"] = {v: float(np.mean(e)) for v, e in entropy.items()}
    res["verdict"] = {}
    for v in NAMES[1:]:
        cv_rel = float(np.mean([res["cv"][str(k)][f"{v}_minus_P"]["rel"] for k in nps.LAYOUTS]))
        better_cv = all(res["cv"][str(k)][f"{v}_minus_P"]["hi"] < 0 for k in nps.LAYOUTS) and cv_rel < -0.003
        better_hold = res["holdout"][f"{v}_minus_P"]["rel"] < -0.003
        kup = res["holdout"][v]["kupiec_p"] > 0.05
        res["verdict"][v] = {
            "cv_mean_rel": cv_rel,
            "better_cv": bool(better_cv),
            "better_holdout": bool(better_hold),
            "var95_ok_holdout": bool(kup),
            "improves": bool(better_cv and better_hold and kup),
        }
    res["seconds"] = round(time.perf_counter() - t_all, 1)
    return res


def write_md(res: dict) -> str:
    m = res["market"]
    lines = [f"# {m} 소프트 결정 분포 숲 (docs/MODELING.md 60)", ""]
    lines += ["| 구간 | 변형 | QS | P 대비 | VaR95 이탈 | Kupiec p | 80% 구간 |", "|---|---|---|---|---|---|---|"]
    blocks = [(f"교차검증 배치 {k}", res["cv"][str(k)]) for k in nps.LAYOUTS] + [("최근 24개월", res["holdout"])]
    for label, b in blocks:
        for v in NAMES:
            x = b[v]
            rel = b.get(f"{v}_minus_P", {}).get("rel")
            rel_s = "—" if rel is None else f"{rel:+.2%}"
            lines.append(
                f"| {label} | {v} | {x['qs']:.4f} | {rel_s} | {x['exc95']:.3f} | {x['kupiec_p']:.3f} | "
                f"{x['cover80']:.2f} |"
            )
    lines += ["", "| 변형 | 교차검증 평균 상대 | 교차검증 개선 | 최근 24개월 개선 | VaR95 정상 | 판정 |"]
    lines += ["|---|---|---|---|---|---|"]
    for v, x in res["verdict"].items():
        lines.append(
            f"| {v} | {x['cv_mean_rel']:+.2%} | {nps.yes(x['better_cv'])} | {nps.yes(x['better_holdout'])} | "
            f"{nps.yes(x['var95_ok_holdout'])} | **{'개선' if x['improves'] else '개선 아님'}** |"
        )
    lines += [
        "",
        "리프 도달 확률의 평균 엔트로피(0 = 한 리프로 확정, log K = 고르게 퍼짐): "
        + ", ".join(
            f"{v} {e:.2f} (최대 {np.log(2 ** VARIANTS[v]['depth']):.2f})" for v, e in res["leaf_entropy"].items()
        ),
    ]
    text = "\n".join(lines) + "\n"
    nps.OUT.mkdir(parents=True, exist_ok=True)
    (nps.OUT / f"soft_dist_{m}.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    res = run(ap.parse_args().market)
    (nps.OUT / f"soft_dist_{res['market']}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(write_md(res), flush=True)


if __name__ == "__main__":
    main()
