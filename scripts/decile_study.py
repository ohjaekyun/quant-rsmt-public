"""Decile backtest of a simple multi-feature score (docs/MODELING.md 77, pre-declared 2026-10-24). Research only.

    .venv/bin/python scripts/decile_study.py --market KR

Features come from the 52 pool and are picked walk-forward (protocol v3 blocks, purged training ends): per month-end
Spearman IC with the label, |t| >= 2, pairwise |rho| <= 0.6, 3 to 10 features, sign = sign of t. Score = mean of
sign x the rank-gauss value (missing = 0). Each month-end the scored names are cut into deciles (D1 lowest .. D10
highest), held equal-weighted from the next close to the next rebalance (portfolio_backtest.simulate), gross and net
of the 47 cost model at the middle book size. Writes reports/strategy/deciles_<M>.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_select_study as fss  # noqa: E402
import portfolio_backtest as pb  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "strategy"
K_MAX, K_MIN, T_MIN, RHO_MAX, MIN_MONTHS, MIN_COVER, CANDIDATES = 10, 3, 2.0, 0.6, 24, 0.5, 40
DECILES = [f"D{i}" for i in range(1, 11)]
PART_KO = {"all": "전체", "outer": "표본 외", "holdout": "홀드아웃"}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ feature choice


def ic_by_date(me: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Month-end x feature Spearman IC with the label (fss._ic_t's per-date values; NaN rows left out)."""
    rows = {}
    for d, g in me.groupby("date"):
        y = g["y_e"].rank().to_numpy(dtype=float)
        if len(y) < 20:
            continue
        y = (y - y.mean()) / (y.std() or 1.0)
        xm = g[cols].rank().to_numpy(dtype=float)
        ok = np.isfinite(xm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu, sd = np.nanmean(xm, axis=0), np.nanstd(xm, axis=0)
        z = np.where(ok, (xm - mu) / np.where(sd > 0, sd, np.nan), 0.0)
        n = ok.sum(axis=0)
        r = (z * y[:, None]).sum(axis=0) / np.maximum(n, 1)
        r[n < 20] = np.nan
        rows[d] = r
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols).sort_index()


def feature_corr(me: pd.DataFrame, cols: list[str], dates: pd.Index) -> pd.DataFrame:
    """Mean per-date Spearman correlation between features over the given month-ends (NaN pairs count as 0)."""
    acc = np.zeros((len(cols), len(cols)))
    for d in dates:
        c = me.loc[me["date"] == d, cols].rank().corr(min_periods=20).to_numpy()
        acc += np.nan_to_num(c)
    return pd.DataFrame(acc / max(len(dates), 1), index=cols, columns=cols)


def choose(ic: pd.DataFrame, me: pd.DataFrame, train_end: pd.Timestamp) -> tuple[dict[str, float], dict[str, float]]:
    hist = ic[ic.index <= train_end]
    n = hist.notna().sum()
    t = (hist.mean() / (hist.std(ddof=1) / np.sqrt(n))).where(n >= MIN_MONTHS)
    cover = me.loc[me["date"] == hist.index[-1], list(ic.columns)].notna().mean()
    t = t[cover >= MIN_COVER].dropna()
    cand = list(t.abs().sort_values(ascending=False).index[:CANDIDATES])
    rho = feature_corr(me, cand, hist.index[-36:])
    chosen: list[str] = []
    for c in cand:  # |t| >= T_MIN first; if that gives fewer than K_MIN, fill by |t| (the rho rule still applies)
        if len(chosen) == K_MAX or (abs(t[c]) < T_MIN and len(chosen) >= K_MIN):
            break
        if all(abs(rho.loc[c, k]) <= RHO_MAX for k in chosen):
            chosen.append(c)
    return {c: float(np.sign(t[c])) for c in chosen}, {c: float(t[c]) for c in chosen}


def score(rows: pd.DataFrame, signs: dict[str, float]) -> np.ndarray:
    cols = list(signs)
    if not cols:
        return np.full(len(rows), np.nan)
    x = rows[cols].to_numpy(dtype=float)
    have = np.isfinite(x).sum(axis=1)
    s = (np.nan_to_num(x) * np.array([signs[c] for c in cols])).sum(axis=1) / len(cols)
    s[have < math.ceil(len(cols) / 2)] = np.nan
    return s


def walk_forward(ds) -> tuple[pd.DataFrame, list[dict]]:
    f = ds.frame
    xs = [c for c in ds.xs_cols if c in f]
    me_all = f[f["is_me"]].drop_duplicates(["date", "entity"])
    lab = me_all[me_all["labelled"]].dropna(subset=["y_e"])
    dates = pd.DatetimeIndex(sorted(lab["date"].unique()))
    hold, pre = dates[-sel.HOLDOUT_MONTHS :], dates[: -sel.HOLDOUT_MONTHS]
    tests = pre[sel.OUTER_MIN_TRAIN_MONTHS :]
    blocks = [("outer", tests[i : i + sel.OUTER_BLOCK_MONTHS]) for i in range(0, len(tests), sel.OUTER_BLOCK_MONTHS)]
    blocks.append(("holdout", hold))
    say(f"IC table: {len(xs)} features x {len(dates)} month-ends")
    ic = ic_by_date(lab, xs)
    parts, log = [], []
    for part, block in blocks:
        end = sel.purge_end(ds, block[0])
        signs, tvals = choose(ic, lab, end)
        rows = me_all[me_all["date"].isin(block)]
        parts.append(rows[["date", "entity", "log_mcap", "y_e"]].assign(score=score(rows, signs), part=part))
        log.append(
            {
                "part": part,
                "test": [str(block[0].date()), str(block[-1].date())],
                "train_end": str(end.date()),
                "features": {c: round(tvals[c], 2) for c in signs},
            }
        )
        say(f"{part} {block[0]:%Y-%m}..{block[-1]:%Y-%m}: {len(signs)} features")
    sc = pd.concat(parts, ignore_index=True)
    sc["mcap"] = np.exp(sc["log_mcap"].astype(float))
    return sc.drop(columns="log_mcap"), log


# ------------------------------------------------------------------------------------------------ portfolios

_base_targets = pb.targets


def targets(g: pd.DataFrame, kind: str) -> pd.Series:
    if not (kind.startswith("D") and kind[1:].isdigit()):
        return _base_targets(g, kind)
    g = g.dropna(subset=["score"])
    if g.empty:
        return pd.Series(dtype=float)
    dec = np.ceil(g["score"].rank(pct=True, method="first") * 10).clip(1, 10)
    s = g[dec.to_numpy() == int(kind[1:])]
    return pd.Series(1.0 / len(s), index=s["entity"]) if len(s) else pd.Series(dtype=float)


pb.targets = targets  # simulate() looks the name up in its module


def part_stats(per: pd.DataFrame, bper: pd.DataFrame) -> dict:
    out = {}
    for part in ("outer", "holdout"):
        m = (per["part"] == part).to_numpy()
        r, b = per["ret"].to_numpy()[m], bper["ret"].to_numpy()[m]
        if len(r) < 3:
            continue
        ex = r - b
        out[part] = {
            "months": int(len(r)),
            "mean_m": float(r.mean()),
            "ann": float(np.prod(1 + r) ** (12 / len(r)) - 1),
            "excess_ann": float(ex.mean() * 12),
            "excess_t": float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))) if ex.std(ddof=1) > 0 else None,
        }
    return out


def capm(per: pd.DataFrame, bper: pd.DataFrame) -> dict:
    """Monthly OLS of the decile's period return on the universe equal weight (B1): beta, alpha (annual) and its t."""
    r, b = per["ret"].to_numpy(dtype=float), bper["ret"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(b)), b])
    coef, *_ = np.linalg.lstsq(x, r, rcond=None)
    resid = r - x @ coef
    cov = (resid @ resid / max(len(r) - 2, 1)) * np.linalg.inv(x.T @ x)
    return {"beta": float(coef[1]), "alpha_ann": float(coef[0] * 12), "alpha_t": float(coef[0] / np.sqrt(cov[0, 0]))}


def label_view(sc: pd.DataFrame) -> tuple[dict, dict]:
    """What the features were picked for: the score's per-date Spearman IC with the label y_e (beta-adjusted
    excess log return over h sessions) and the mean label per decile (per-date decile means, averaged)."""
    lab = sc.dropna(subset=["score", "y_e"]).copy()
    lab["dec"] = lab.groupby("date")["score"].transform(
        lambda x: np.ceil(x.rank(pct=True, method="first") * 10).clip(1, 10)
    )
    ic, dec = {}, {}
    for part in ("outer", "holdout"):
        g = lab[lab["part"] == part]
        v = np.array([d["score"].rank().corr(d["y_e"].rank()) for _, d in g.groupby("date") if len(d) >= 20])
        v = v[np.isfinite(v)]
        if len(v) > 2:
            ic[part] = {
                "ic": float(v.mean()),
                "t": float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))),
                "months": len(v),
            }
        means = g.groupby(["date", "dec"])["y_e"].mean().unstack()
        dec[part] = {f"D{int(k)}": float(means[k].mean()) for k in means.columns}
    return ic, dec


def spread(hi: pd.DataFrame, lo: pd.DataFrame) -> dict:
    out = {}
    for part in ("all", "outer", "holdout"):
        m = np.ones(len(hi), bool) if part == "all" else (hi["part"] == part).to_numpy()
        d = hi["ret"].to_numpy()[m] - lo["ret"].to_numpy()[m]
        if len(d) < 3:
            continue
        sd = d.std(ddof=1)
        out[part] = {
            "mean_m": float(d.mean()),
            "t": float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else None,
            "sharpe": float(d.mean() / sd * np.sqrt(12)) if sd > 0 else None,
            "hit": float((d > 0).mean()),
        }
    return out


def run(market: str) -> dict:
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    uni, ds, info = fss.build_pool(market, settings, cache)
    sc, log = walk_forward(ds)
    cal = pd.DatetimeIndex(ds.calendar)
    cal = cal[cal >= pd.Timestamp(sc["date"].min())]
    w = pb.wide(uni.prices, sorted(sc["entity"].unique()), cal)
    mid = pb.cm.AUM[market][1]
    bnav, bper = pb.simulate(market, sc, w, "B1", mid, 1.0, False)
    res: dict = {"market": market, "pool": info.get("pool"), "blocks": log, "deciles": {}}
    pers = {}
    for k in DECILES:
        nav, per = pb.simulate(market, sc, w, k, mid, 1.0, False)
        nnav, nper = pb.simulate(market, sc, w, k, mid, 1.0, True)
        pers[k] = per
        st = pb.stats(nav, bnav, per, bper)
        st["net_cagr"] = pb.stats(nnav, bnav, nper, bper)["cagr"]
        st["avg_names"] = float(per["n"].mean())
        st["parts"] = part_stats(per, bper)
        st["capm"] = capm(per, bper)
        res["deciles"][k] = st
        say(f"{k}: CAGR {st['cagr']:+.1%} (net {st['net_cagr']:+.1%}), excess {st['excess_ann']:+.1%}")
    res["bench"] = {
        "cagr": pb.stats(bnav, bnav, bper, bper)["cagr"],
        "vol": float(bnav.pct_change().std() * np.sqrt(252)),
        "mdd": pb.mdd(bnav),
        "parts": part_stats(bper, bper),
    }
    res["spread_d10_d1"] = spread(pers["D10"], pers["D1"])
    mono = {}
    for part in ("all", "outer", "holdout"):
        means = [
            float(p["ret"].mean() if part == "all" else p.loc[p["part"] == part, "ret"].mean()) for p in pers.values()
        ]
        mono[part] = float(pd.Series(means).rank().corr(pd.Series(range(10)), method="spearman"))
    res["monotonic"] = mono
    res["score_ic"], res["label_by_decile"] = label_view(sc)  # 77.1: added after the first KR result
    res["period"] = [str(sc["date"].min().date()), str(sc["date"].max().date())]
    res["feature_count"] = dict(Counter(c for b in log for c in b["features"]).most_common())
    return res


# ------------------------------------------------------------------------------------------------ report


def _doc(c: str) -> str:
    return feat.FEATURE_DOCS.get(c.removeprefix("x_"), (c, c))[0]


def write_md(r: dict) -> str:
    p = pb.pct
    sp = r["spread_d10_d1"]
    lines = [
        f"# {r['market']} 여러 변수 합산 점수 10분위 백테스트 (docs/MODELING.md 77)",
        "",
        f"- 기간 {r['period'][0]} ~ {r['period'][1]} (월말 리밸런싱, 다음 거래일 종가 체결) · 후보 변수 {r['pool']} · "
        f"변수 고르기 = 걷기 방식(그 블록 이전 자료만)",
        f"- 비교(B1, 점수 있는 종목 같은 비중): 연 {p(r['bench']['cagr'])} · 최대낙폭 {p(r['bench']['mdd'])}",
        "",
        "| 분위 | 평균 종목 수 | 연수익(비용 전) | 연수익(비용 후) | 변동성 | 샤프 | 최대낙폭 | B1 대비 초과(연) | t |"
        " 회전율(연) | 표본 외 연 | 홀드아웃 연 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in DECILES:
        d = r["deciles"][k]
        o, h = d["parts"].get("outer", {}), d["parts"].get("holdout", {})
        lines.append(
            f"| {k}{' (상위)' if k == 'D10' else ' (하위)' if k == 'D1' else ''} | {d['avg_names']:.0f} | "
            f"{p(d['cagr'])} | {p(d['net_cagr'])} | {d['vol'] * 100:.1f}% | {d['sharpe']:.2f} | {p(d['mdd'])} | "
            f"{p(d['excess_ann'])} | {d['excess_t']:+.2f} | {d['turnover_yr']:.1f}배 | {p(o.get('ann'))} | "
            f"{p(h.get('ann'))} |"
        )
    lines += [
        "",
        "- **D10 − D1 월 스프레드**(비용 전): "
        + " · ".join(
            f"{PART_KO[k]} 월 {p(v['mean_m'], 2)} "
            f"(t {v['t']:+.2f}, 연 샤프 {v['sharpe']:.2f}, 플러스 달 {v['hit']:.0%})"
            for k, v in sp.items()
        ),
        "- **단조성**(분위 번호와 월평균 수익의 순위 상관, 1이면 완벽한 계단): "
        + " · ".join(f"{PART_KO[k]} {v:+.2f}" for k, v in r["monotonic"].items()),
        "",
        "## 베타를 빼고 보면 (77.1, 첫 한국 결과를 본 뒤 추가한 진단)",
        "",
        "변수는 **베타 조정 초과수익**(종목 수익 − 베타 × 지수 수익, 20거래일)을 맞히도록 골랐습니다. "
        "위 표는 원래 수익이라 베타가 낮은 분위가 상승장에서 불리합니다.",
        "",
        "- 점수의 IC(라벨 = 베타 조정 초과수익): "
        + " · ".join(
            f"{PART_KO[k]} {v['ic']:+.3f} (t {v['t']:+.2f}, {v['months']}개월)" for k, v in r["score_ic"].items()
        ),
        "",
        "| 분위 | 베타(B1 대비) | 알파(연) | 알파 t | 베타 조정 초과수익 평균(20일) 표본 외 | 홀드아웃 |",
        "|---|---|---|---|---|---|",
        *[
            f"| {k} | {r['deciles'][k]['capm']['beta']:.2f} | {p(r['deciles'][k]['capm']['alpha_ann'])} | "
            f"{r['deciles'][k]['capm']['alpha_t']:+.2f} | {p(r['label_by_decile'].get('outer', {}).get(k), 2)} | "
            f"{p(r['label_by_decile'].get('holdout', {}).get(k), 2)} |"
            for k in DECILES
        ],
        "",
        "## 블록별로 고른 변수 (괄호 = 학습 구간 IC t)",
        "",
    ]
    for b in r["blocks"]:
        feats = ", ".join(f"{_doc(c)}({t:+.1f})" for c, t in b["features"].items())
        lines.append(f"- {b['test'][0][:7]}~{b['test'][1][:7]} ({PART_KO[b['part']]}): {feats}")
    lines += ["", "## 자주 고른 변수", ""]
    nb = len(r["blocks"])
    for c, n in list(r["feature_count"].items())[:15]:
        lines.append(f"- {_doc(c)} (`{c}`): {n}/{nb} 블록")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    m = ap.parse_args().market
    OUT.mkdir(parents=True, exist_ok=True)
    r = run(m)
    text = write_md(r)
    (OUT / f"deciles_{m}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    (OUT / f"deciles_{m}.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
