"""RSMT-Q distribution research (docs/MODELING.md 34, pre-declared 2026-10-09): does relaxing the parametric
(Student-t) leaf distribution help the RISK side? Three predictive distributions of the 20-session total log return
are scored on the SAME non-overlapping test rows (the on-grid rows of RSMT-Q's own expanding-window CV folds,
layouts 3/4/5, and of the last 24 months):

- P  = RSMT-Q as adopted (config/default.yaml): tree + sigma equation + Student-t legs, simulated paths;
- C  = P recalibrated: its quantile levels remapped by the empirical distribution of PIT values u = F(r) of EARLIER
       test rows only (earlier folds of the same layout; the holdout uses every CV fold of layout 5);
- Q  = LightGBM quantile regression (19 levels 0.05..0.95, fixed settings, no tuning) on the panel's features,
       trained on every 5th session of the same training rows; quantiles sorted per row (no crossing).

Scores per row on the 19-level grid: QS = (2/19) * sum_l pinball_l(r, q_l) (a CRPS approximation, the same for all
three), VaR95 exceedance (r < q_0.05; target 5%, Kupiec test), 80% interval cover (q_0.10 <= r <= q_0.90); plus the
exact CRPS of P. Research only - the served model is unchanged.

    .venv/bin/python scripts/rsmt_np_study.py --market KR
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LEVELS = np.round(np.arange(0.05, 0.951, 0.05), 2)  # 19 levels
DENSE = np.round(np.arange(0.005, 0.9951, 0.005), 3)  # model quantiles kept for the recalibration map
LAYOUTS = (3, 4, 5)
HOLDOUT_MONTHS = 24
N_PATHS = 1000
QR_STEP = 5  # quantile-regression training rows: every 5th session
QR_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 10.0,
    "verbose": -1,
}
NON_FEATURES = {"ticker", "date", "bench", "t_end", "y_e", "m", "r", "w", "on_grid", "block"}
OUT = Path(__file__).resolve().parents[1] / "reports" / "rsmt"


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pinball(r: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-row grid quantile score (2/K) * sum_l rho_l(r - q_l); q is (n, K) on LEVELS."""
    d = r[:, None] - q
    return (2.0 / len(LEVELS)) * np.sum(np.maximum(LEVELS * d, (LEVELS - 1.0) * d), axis=1)


def kupiec(n_exc: int, n: int, p: float = 0.05) -> float:
    """Kupiec proportion-of-failures test p-value."""
    if n == 0:
        return float("nan")
    x = n_exc / n
    if x in (0.0, 1.0):
        ll1 = 0.0
    else:
        ll1 = n_exc * math.log(x) + (n - n_exc) * math.log(1 - x)
    ll0 = n_exc * math.log(p) + (n - n_exc) * math.log(1 - p)
    return float(stats.chi2.sf(-2 * (ll0 - ll1), 1))


def model_quantiles(model, panel, settings, trn: pd.DataFrame, te: pd.DataFrame) -> dict[str, np.ndarray]:
    """P: simulated distribution of every test row (serving rules: plausibility clamp, row seeds)."""
    import quantcio.quant.rsmt as R
    from quantcio.quant.distribution import ReturnDistribution, path_seed, total_log_sd
    from quantcio.quant.panel import moments_from_settings

    h = panel.horizon
    moments = moments_from_settings(trn, settings, panel.market, h)
    prm = R._row_params(model, te, moments, h, True)
    r = te["r"].to_numpy(dtype="float64")
    n = len(te)
    dense = np.empty((n, len(DENSE)))
    grid = np.empty((n, len(LEVELS)))
    crps = np.empty(n)
    pit = np.empty(n)
    for i in range(n):
        beta = float(prm["beta"][i])
        sd = total_log_sd(
            beta,
            float(prm["sigma_e"][i]),
            float(prm["sigma_m"][i]),
            float(prm["rho"][i]),
            float(prm["nu"][i]),
            float(prm["nu_m"][i]),
            h,
        )
        clamp = R.plausibility_shift(beta * float(prm["mu_m"][i]) + float(prm["mu_e"][i]), sd)
        seed = path_seed(str(te["ticker"].iloc[i]), pd.Timestamp(te["date"].iloc[i]).date().isoformat())
        dist = ReturnDistribution(
            prm["mu_e"][i] + clamp,
            prm["sigma_e"][i],
            prm["mu_m"][i],
            prm["sigma_m"][i],
            prm["nu"][i],
            prm["nu_m"][i],
            prm["beta"][i],
            prm["rho"][i],
            h,
            N_PATHS,
            seed,
        )
        term = np.sort(np.asarray(dist.r_terminal, dtype="float64"))
        dense[i] = np.quantile(term, DENSE)
        grid[i] = np.quantile(term, LEVELS)
        crps[i] = dist.crps(r[i])
        pit[i] = (np.searchsorted(term, r[i], "right")) / term.size
    return {"grid": grid, "dense": dense, "crps": crps, "pit": pit}


def recalibrate(dense: np.ndarray, pits: np.ndarray) -> np.ndarray:
    """C: q_cal(p) = Q(G^-1(p)), G = empirical CDF of earlier PITs; Q interpolated on the dense level grid."""
    if pits.size < 200:
        return np.array([np.interp(LEVELS, DENSE, row) for row in dense])
    target = np.clip(np.quantile(pits, LEVELS), DENSE[0], DENSE[-1])
    return np.array([np.interp(target, DENSE, row) for row in dense])


def qr_quantiles(trn: pd.DataFrame, te: pd.DataFrame, feats: list[str], seed: int) -> np.ndarray:
    import lightgbm as lgb

    d = pd.to_datetime(trn["date"])
    sessions = np.sort(d.unique())
    keep = set(sessions[::QR_STEP])
    t = trn[d.isin(keep).to_numpy()]
    X, y = t[feats].to_numpy(dtype="float64"), t["r"].to_numpy(dtype="float64")
    ok = np.isfinite(y)
    Xt = te[feats].to_numpy(dtype="float64")
    out = np.empty((len(te), len(LEVELS)))
    for j, a in enumerate(LEVELS):
        m = lgb.LGBMRegressor(objective="quantile", alpha=float(a), random_state=seed, **QR_PARAMS)
        m.fit(X[ok], y[ok])
        out[:, j] = m.predict(Xt)
    return np.sort(out, axis=1)


def scores(r: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "qs": pinball(r, q),
        "exc95": (r < q[:, 0]).astype(float),
        "in80": ((q[:, 1] <= r) & (r <= q[:, -2])).astype(float),
    }


def block_ci(diff: np.ndarray, dates: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a per-row difference with a date-cluster bootstrap 95% interval."""
    df = pd.DataFrame({"d": dates, "x": diff}).groupby("d")["x"].agg(["sum", "count"])
    s, c = df["sum"].to_numpy(), df["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boots = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return float(s.sum() / c.sum()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def summarise(rows: pd.DataFrame) -> dict:
    out = {"n": int(len(rows)), "dates": int(rows["date"].nunique())}
    for v in ("P", "C", "Q"):
        n_exc = int(rows[f"{v}_exc95"].sum())
        out[v] = {
            "qs": float(rows[f"{v}_qs"].mean()),
            "exc95": float(rows[f"{v}_exc95"].mean()),
            "kupiec_p": kupiec(n_exc, len(rows)),
            "cover80": float(rows[f"{v}_in80"].mean()),
        }
    out["P"]["crps_exact"] = float(rows["P_crps"].mean())
    for v in ("C", "Q"):
        m, lo, hi = block_ci((rows[f"{v}_qs"] - rows["P_qs"]).to_numpy(), rows["date"].to_numpy())
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
    feats = [c for c in tr.columns if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(tr[c])]
    grid_dates = pd.DatetimeIndex(sorted(tr.loc[tr["on_grid"], "date"].unique()))
    first_hold = grid_dates[grid_dates > grid_dates[-1] - pd.DateOffset(months=HOLDOUT_MONTHS)][0]
    pre = tr[(pd.to_datetime(tr["t_end"]) < first_hold).to_numpy()].reset_index(drop=True)
    say(f"{market}: {len(tr)} rows, {len(feats)} QR features, holdout from {first_hold.date()}")

    def fold_rows(trn, te, pits_before, seed):
        model = R.RSMTree.from_settings(s, panel.market).fit_frame(trn)
        p = model_quantiles(model, panel, s, trn, te)
        r = te["r"].to_numpy(dtype="float64")
        qc = recalibrate(p["dense"], pits_before)
        qq = qr_quantiles(trn, te, feats, seed)
        out = pd.DataFrame({"date": pd.to_datetime(te["date"]).to_numpy(), "ticker": te["ticker"].to_numpy(), "r": r})
        for name, q in (("P", p["grid"]), ("C", qc), ("Q", qq)):
            for k, v in scores(r, q).items():
                out[f"{name}_{k}"] = v
        out["P_crps"] = p["crps"]
        out["pit"] = p["pit"]
        return out, int(model.n_splits)

    res: dict = {"market": market, "holdout_start": str(first_hold.date()), "features_qr": feats, "cv": {}}
    pits_l5 = []
    for k in LAYOUTS:
        parts, pits = [], np.array([])
        for fold in R.cv_splits(pre, k):
            trn, te = pre.iloc[fold.train_idx], pre.iloc[fold.test_idx]
            if len(te) == 0 or trn["on_grid"].sum() < 60:
                continue
            t0 = time.perf_counter()
            rows, splits = fold_rows(trn, te, pits, seed=100 * k + fold.fold)
            pits = np.concatenate([pits, rows["pit"].to_numpy()])
            parts.append(rows)
            say(
                f"{market} layout {k} fold {fold.fold}: {len(te)} test rows, {splits} splits, "
                f"{time.perf_counter() - t0:.0f}s"
            )
        allr = pd.concat(parts, ignore_index=True)
        res["cv"][str(k)] = summarise(allr)
        if k == 5:
            pits_l5 = pits
    d = pd.to_datetime(tr["date"])
    trn = tr[((d < first_hold) & (pd.to_datetime(tr["t_end"]) < first_hold)).to_numpy()]
    te = tr[((d >= first_hold) & tr["on_grid"].to_numpy(dtype=bool)).to_numpy() & tr["r"].notna().to_numpy()]
    t0 = time.perf_counter()
    rows, splits = fold_rows(trn, te, np.asarray(pits_l5), seed=999)
    res["holdout"] = summarise(rows)
    say(f"{market} holdout: {len(te)} rows, {splits} splits, {time.perf_counter() - t0:.0f}s")
    cv_mean = {v: float(np.mean([res["cv"][str(k)][f"{v}_minus_P"]["rel"] for k in LAYOUTS])) for v in ("C", "Q")}
    res["cv_mean_rel"] = cv_mean
    res["verdict"] = {}
    for v in ("C", "Q"):
        better_cv = all(res["cv"][str(k)][f"{v}_minus_P"]["hi"] < 0 for k in LAYOUTS) and cv_mean[v] < -0.003
        h = res["holdout"][f"{v}_minus_P"]
        better_hold = h["rel"] < -0.003
        kup = res["holdout"][v]["kupiec_p"] > 0.05
        res["verdict"][v] = {
            "better_cv": bool(better_cv),
            "better_holdout": bool(better_hold),
            "var95_ok_holdout": bool(kup),
            "improves": bool(better_cv and better_hold and kup),
        }
    res["seconds"] = round(time.perf_counter() - t_all, 1)
    return res


def yes(x: bool) -> str:
    return "예" if x else "아니오"


def write_md(res: dict) -> str:
    m = res["market"]
    lines = [
        f"# RSMT-Q 분포 완화 시험 — {m} (docs/MODELING.md 34)",
        "",
        f"- 시험 행: RSMT-Q 자체 CV(레이아웃 3·4·5)의 겹치지 않는 행 + 최근 24개월({res['holdout_start']}~). "
        "분위수 점수(QS) = 19개 분위 핀볼 손실(CRPS 근사, 낮을수록 좋음)",
        "",
        "| 구간 | 행 | P(현재) QS | C(사후 보정) QS | Q(분위수 회귀) QS | VaR95 이탈 P / C / Q (목표 5%) "
        "| 80% 구간 P / C / Q |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in [*((f"CV 레이아웃 {k}", res["cv"][str(k)]) for k in LAYOUTS), ("최근 24개월", res["holdout"])]:
        lines.append(
            f"| {name} | {s['n']} | {s['P']['qs']:.5f} | {s['C']['qs']:.5f} ({s['C_minus_P']['rel']:+.2%}) | "
            f"{s['Q']['qs']:.5f} ({s['Q_minus_P']['rel']:+.2%}) | "
            f"{s['P']['exc95']:.3f} / {s['C']['exc95']:.3f} / {s['Q']['exc95']:.3f} | "
            f"{s['P']['cover80']:.3f} / {s['C']['cover80']:.3f} / {s['Q']['cover80']:.3f} |"
        )
    h = res["holdout"]
    cv_crps = ", ".join(format(res["cv"][str(k)]["P"]["crps_exact"], ".5f") for k in LAYOUTS)
    lines += [
        "",
        f"- 최근 24개월 VaR95 이탈 Kupiec p: P {h['P']['kupiec_p']:.3f} · C {h['C']['kupiec_p']:.3f} · "
        f"Q {h['Q']['kupiec_p']:.3f} (p ≤ 0.05면 이탈 비율이 5%와 유의하게 다름)",
        f"- P의 정확한 CRPS: CV {cv_crps} · 최근 24개월 {h['P']['crps_exact']:.5f}",
        "",
    ]
    for v, ko in (("C", "사후 보정"), ("Q", "분위수 회귀")):
        vd = res["verdict"][v]
        lines.append(
            f"- **{v}({ko})**: CV 개선 {yes(vd['better_cv'])} · 최근 24개월 개선 {yes(vd['better_holdout'])} · "
            f"최근 24개월 VaR95 적정 {yes(vd['var95_ok_holdout'])} → "
            f"**{'채택 후보' if vd['improves'] else '채택 안 함'}**"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    a = ap.parse_args()
    res = run(a.market)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"np_dist_{a.market}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    md = write_md(res)
    (OUT / f"np_dist_{a.market}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
