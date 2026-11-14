"""RSMT-Q upgrade proposal (docs/MODELING.md 59, pre-declared 2026-10-14): randomized distribution-tree ensembles
(extra-trees splits + bootstrap, nonparametric leaf quantiles averaged across trees) scored against the served RSMT-Q
(P) and the LightGBM quantile regression (Q) on the SAME test rows as docs/MODELING.md 34. Research only.

    .venv/bin/python scripts/rsmt_et_study.py --market KR

Writes reports/rsmt/et_dist_<M>.{json,md}.
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
import rsmt_np_study as nps  # noqa: E402

LEVELS = nps.LEVELS
VARIANTS = {
    "E1": {"trees": 50, "depth": 6, "leaf": 20},  # the proposal's settings
    "E2": {"trees": 300, "depth": None, "leaf": "0.5%"},  # leaves >= 0.5% of the training rows (min 200)
}
NAMES = ("P", "Q", *VARIANTS)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def et_quantiles(
    trn: pd.DataFrame, te: pd.DataFrame, feats: list[str], seed: int, cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    """(19 sorted quantiles per test row, leaf CVaR5 per test row) of one randomized distribution-tree ensemble."""
    from sklearn.ensemble import ExtraTreesRegressor

    d = pd.to_datetime(trn["date"])
    keep = set(np.sort(d.unique())[:: nps.QR_STEP])
    t = trn[d.isin(keep).to_numpy()]
    x, y = t[feats].to_numpy(dtype="float64"), t["r"].to_numpy(dtype="float64")
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    leaf = cfg["leaf"] if isinstance(cfg["leaf"], int) else max(200, int(0.005 * len(y)))
    m = ExtraTreesRegressor(
        n_estimators=cfg["trees"],
        max_depth=cfg["depth"],
        min_samples_leaf=leaf,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=8,
        random_state=seed,
    ).fit(x, y)
    lt, le = m.apply(x), m.apply(te[feats].to_numpy(dtype="float64"))
    q = np.zeros((len(te), len(LEVELS)))
    cv = np.zeros(len(te))
    for j in range(lt.shape[1]):
        g = pd.DataFrame({"leaf": lt[:, j], "y": y}).groupby("leaf")["y"]
        qt = g.quantile(list(LEVELS)).unstack()
        tail = g.apply(lambda s: s[s <= s.quantile(0.05)].mean())
        q += qt.reindex(le[:, j]).to_numpy()
        cv += tail.reindex(le[:, j]).to_numpy()
    return np.sort(q / lt.shape[1], axis=1), cv / lt.shape[1]


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
    for v in VARIANTS:
        out[v]["cvar5_mean"] = float(rows[f"{v}_cvar5"].mean())
        r = rows["r"].to_numpy()
        tail = r < rows[f"{v}_q05"].to_numpy()
        out[v]["realised_cvar5"] = float(r[tail].mean()) if tail.any() else float("nan")
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

    def fold_rows(trn, te, seed):
        model = R.RSMTree.from_settings(s, panel.market).fit_frame(trn)
        p = nps.model_quantiles(model, panel, s, trn, te)
        r = te["r"].to_numpy(dtype="float64")
        out = pd.DataFrame({"date": pd.to_datetime(te["date"]).to_numpy(), "ticker": te["ticker"].to_numpy(), "r": r})
        qs = {"P": p["grid"], "Q": nps.qr_quantiles(trn, te, feats, seed)}
        for v, cfg in VARIANTS.items():
            qs[v], out[f"{v}_cvar5"] = et_quantiles(trn, te, feats, seed, cfg)
            out[f"{v}_q05"] = qs[v][:, 0]
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
    lines = [f"# {m} RSMT-Q 무작위 분포 나무 앙상블 (docs/MODELING.md 59)", ""]
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
    lines += [
        "",
        "| 변형 | 교차검증 평균 상대 | 교차검증 개선 | 최근 24개월 개선 | VaR95 정상 | 판정 |",
        "|---|---|---|---|---|---|",
    ]
    for v, x in res["verdict"].items():
        lines.append(
            f"| {v} | {x['cv_mean_rel']:+.2%} | {nps.yes(x['better_cv'])} | {nps.yes(x['better_holdout'])} | "
            f"{nps.yes(x['var95_ok_holdout'])} | **{'개선' if x['improves'] else '개선 아님'}** |"
        )
    h = res["holdout"]
    lines += ["", "리프 CVaR5(하위 5% 평균) 예측 평균 vs 실제(VaR95 이탈 행의 평균), 최근 24개월:"]
    for v in VARIANTS:
        lines.append(f"- {v}: 예측 {h[v]['cvar5_mean']:+.4f} · 실제 {h[v]['realised_cvar5']:+.4f}")
    text = "\n".join(lines) + "\n"
    nps.OUT.mkdir(parents=True, exist_ok=True)
    (nps.OUT / f"et_dist_{m}.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    res = run(ap.parse_args().market)
    (nps.OUT / f"et_dist_{res['market']}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(write_md(res), flush=True)


if __name__ == "__main__":
    main()
