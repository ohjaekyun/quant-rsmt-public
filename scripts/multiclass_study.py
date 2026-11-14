"""Multi-class targets (docs/MODELING.md 37, pre-declared 2026-10-10): the user's proposal - classify the 20-session
return into 4 classes (>= +20%, +10..20%, 0..10%, < 0) instead of the median split - tested against the same learner's
binary median model and the served model. Research only.

Score of a class model = expected class (0..3) from its probabilities; the protocol's within-date AUC ranks by it.
Tail metrics on the out-of-sample rows: for the top / bottom decile by score on each date, the mean 20-session excess
return and the share of rows with a return >= +10% (<= -10%) minus that date's base rate.

    .venv/bin/python scripts/multiclass_study.py --market KR

Writes reports/modeling/<MKT>_multiclass.{md,json} and reports/modeling/<MKT>_multiclass_summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import us_v6_study as v6  # noqa: E402
import v13_study as v13s  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
EDGES = (0.0, 0.10, 0.20)  # class k = number of edges the 20-session simple return reaches
SERVED = {"KR": "et", "US": "cat_rank"}
MODELS = ("mc_abs", "ord_abs", "mc_exc", "bin_med")
MIN_CLASS = 20


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def classes(log_ret: np.ndarray) -> np.ndarray:
    simple = np.expm1(log_ret)
    out = np.zeros(len(simple))
    for e in EDGES:
        out += simple >= e
    return np.where(np.isfinite(simple), out, np.nan)


class ExpectedClass:
    """Wraps a multi-class LightGBM: predict = sum_k k * P(class k) over the classes seen in training."""

    def __init__(self, model, labels: np.ndarray):
        self.model, self.labels = model, labels

    def predict(self, X):
        return np.asarray(self.model.predict_proba(X)) @ self.labels


class OrdinalSum:
    """Cumulative binary models P(class >= k), k = 1..3: predict = their sum (the expected class)."""

    def __init__(self, models: list):
        self.models = models

    def predict(self, X):
        return np.sum([np.asarray(m.predict_proba(X))[:, 1] for m in self.models], axis=0)


def _xy(train, cols, ctx, target_col: str | None, label_from: str | None):
    use = [x for x in [*cols, *ctx.regime_cols] if train[x].notna().any()]
    if label_from is not None:
        y = classes(train[label_from].to_numpy(dtype=float))
    else:
        y = train[target_col].to_numpy(dtype=float)
    ok = np.isfinite(y)
    return use, train[use].to_numpy(dtype=np.float64)[ok], y[ok], ok


def fitter_mc(cols: list[str], label_from: str):
    def fit(c, train, params, scheme, ctx):
        import lightgbm as lgb

        use, X, y, ok = _xy(train, cols, ctx, None, label_from)
        y = y.astype(int)
        seen = np.array(sorted(k for k in np.unique(y) if (y == k).sum() >= MIN_CLASS))
        y = np.clip(y, seen.min(), seen.max())  # merge a too-rare extreme class into its neighbour
        seen = np.unique(y)
        w = v6._weights(train, scheme, ctx, False)[ok]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = lgb.LGBMClassifier(objective="multiclass", **v6._tree_kw(params, len(y), ctx)).fit(
                X, y, sample_weight=w
            )
        return cmod.Fitted(c.name, params, scheme, ExpectedClass(m, seen.astype(float)), use, "hgb")

    return fit


def fitter_ord(cols: list[str], label_from: str):
    def fit(c, train, params, scheme, ctx):
        import lightgbm as lgb

        use, X, y, ok = _xy(train, cols, ctx, None, label_from)
        w = v6._weights(train, scheme, ctx, False)[ok]
        models = []
        for k in (1, 2, 3):
            yk = (y >= k).astype(int)
            if yk.sum() < MIN_CLASS or (1 - yk).sum() < MIN_CLASS:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                models.append(
                    lgb.LGBMClassifier(objective="binary", **v6._tree_kw(params, len(yk), ctx)).fit(
                        X, yk, sample_weight=w
                    )
                )
        return cmod.Fitted(c.name, params, scheme, OrdinalSum(models), use, "hgb")

    return fit


def fitter_bin(cols: list[str]):
    def fit(c, train, params, scheme, ctx):
        import lightgbm as lgb

        use, X, y, ok = _xy(train, cols, ctx, "y_cls", None)
        w = v6._weights(train, scheme, ctx, False)[ok]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = lgb.LGBMClassifier(objective="binary", **v6._tree_kw(params, len(y), ctx)).fit(
                X, y.astype(int), sample_weight=w
            )
        return cmod.Fitted(c.name, params, scheme, m, use, "lgbm_clf")

    return fit


def candidates(market: str, xs: list[str]) -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    out = [every["mom_12_1"], every["factor_ic"], every[SERVED[market]]]
    spec = [
        ("mc_abs", "LightGBM 4구간 분류 · 절대 수익(≥20%·10~20%·0~10%·<0)", fitter_mc(xs, "r")),
        ("ord_abs", "LightGBM 순서형(누적 이진 3개) · 절대 수익 같은 구간", fitter_ord(xs, "r")),
        ("mc_exc", "LightGBM 4구간 분류 · 초과수익(같은 구간)", fitter_mc(xs, "y_e")),
        ("bin_med", "LightGBM 이진 · 중앙값 초과(같은 설정, 비교 기준)", fitter_bin(xs)),
    ]
    for name, ko, fitter in spec:
        key = f"multiclass__{name}"
        cmod._FITTERS[key] = fitter
        out.append(cmod.Candidate(name, "tree", ko, name, grid=list(v6.DEPTH_GRID), schemes=("none",), fitter=key))
    return out


# ------------------------------------------------------------------------------------------------ tail metrics


def _boot(diff_by_date: pd.Series, n: int = 2000, seed: int = 0) -> dict:
    x = diff_by_date.dropna().to_numpy(float)
    if len(x) < 5:
        return {"mean": None, "lo": None, "hi": None, "n": int(len(x))}
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "lo": float(np.quantile(b, 0.025)),
        "hi": float(np.quantile(b, 0.975)),
        "n": len(x),
    }


def tail_metrics(pred: pd.DataFrame, vol: pd.Series, names: list[str]) -> dict:
    """Per candidate: top / bottom decile mean excess, +-10% hit rate minus the date base rate, tail AUC, vol corr."""
    d = pred.copy()
    d["R"] = np.expm1(d["r"].astype(float))
    d["big_up"] = (d["R"] >= 0.10).astype(float)
    d["big_dn"] = (d["R"] <= -0.10).astype(float)
    d["vol"] = vol.reindex(d.index).to_numpy() if vol is not None else np.nan
    out = {}
    for n in names:
        if n not in d:
            continue
        g = d[["date", n, "y_e", "big_up", "big_dn", "vol"]].dropna(subset=[n, "y_e"])
        q = g.groupby("date")[n].rank(pct=True)
        top, bot = g[q > 0.9], g[q <= 0.1]
        base_up = g.groupby("date")["big_up"].mean()
        base_dn = g.groupby("date")["big_dn"].mean()
        mean_ye = g.groupby("date")["y_e"].mean()
        tail = g.assign(score=g[n], y=g["big_up"])
        auc_up = []
        for _, x in tail.groupby("date"):
            if 5 <= x["y"].sum() <= len(x) - 5:
                from sklearn.metrics import roc_auc_score

                auc_up.append(roc_auc_score(x["y"], x["score"]))
        out[n] = {
            "top_excess": _boot(top.groupby("date")["y_e"].mean() - mean_ye),
            "bottom_excess": _boot(bot.groupby("date")["y_e"].mean() - mean_ye),
            "top_bigup_minus_base": _boot(top.groupby("date")["big_up"].mean() - base_up),
            "top_bigdn_minus_base": _boot(top.groupby("date")["big_dn"].mean() - base_dn),
            "bottom_bigdn_minus_base": _boot(bot.groupby("date")["big_dn"].mean() - base_dn),
            "tail_auc_bigup": float(np.mean(auc_up)) if auc_up else None,
            "corr_vol": float(
                g.groupby("date").apply(lambda x: x[n].corr(x["vol"], method="spearman"), include_groups=False).mean()
            )
            if g["vol"].notna().any()
            else None,
        }
    return out


# ------------------------------------------------------------------------------------------------ run


def write_summary(market: str) -> str:
    d = json.loads((OUT / f"{market}_multiclass.json").read_text())
    info = d["protocol"]["multiclass"]
    served = SERVED[market]

    def f(x, spec="+.4f"):
        return "–" if x is None else format(x, spec)

    lines = [
        f"# {market} 다중 분류 시험 (docs/MODELING.md 37)",
        "",
        f"- 구간 비율(학습 월말 행): 절대 수익 {info['class_share_abs']} · 초과수익 {info['class_share_exc']} "
        "(0 = <0%, 1 = 0~10%, 2 = 10~20%, 3 = ≥20%)",
        f"- 선정 `{d['selected']}` · 게이트 {'통과' if d['gate']['passed'] else '미달'}",
        "",
        "| 후보 | 외부 AUC [95% CI] | 외부 IC (t) | 홀드아웃 AUC |",
        "|---|---|---|---|",
    ]
    for c in d["candidates"]:
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        hm = f"{h['mean']:.4f}" if h.get("mean") is not None else "–"
        lines.append(
            f"| {c['name']} | {o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] | "
            f"{c['outer']['ic']['mean']:+.4f} ({c['outer']['ic']['t']:+.2f}) | {hm} |"
        )
    lines += ["", "## 질문 1: 순위 실력 (날짜별 AUC 차이)", ""]
    q1 = []
    for k, v in (d["protocol"].get("alt_value") or {}).items():
        o, h = v["outer"], v.get("holdout") or {}
        hh = f" · 홀드아웃 {f(h.get('mean'))} [{f(h.get('lo'))}, {f(h.get('hi'))}]" if h.get("mean") is not None else ""
        lines.append(f"- {k}: 외부 {f(o['mean'])} [{f(o['lo'])}, {f(o['hi'])}] (t {f(o.get('t'), '+.2f')}){hh}")
        if o["lo"] is not None and o["lo"] > 0 and (h.get("mean") is None or h["mean"] >= 0):
            q1.append(k)
    lines += ["", f"**질문 1 판정: {'개선 — ' + ', '.join(q1) if q1 else '개선 없음'}**", ""]
    lines += [
        "## 질문 2: 꼬리 실력 (외부 / 홀드아웃; 같은 날 평균 대비)",
        "",
        "| 후보 | 상위 10% 초과수익 | 상위 10% 중 +10% 이상 비율 − 기준 | 상위 10% 중 −10% 이하 비율 − 기준 "
        "| 하위 10% 초과수익 | +10% 이상 식별 AUC | 변동성과 순위상관 |",
        "|---|---|---|---|---|---|---|",
    ]
    q2 = []
    for n in [*MODELS, served]:
        o, h = info["tail_outer"].get(n), info["tail_holdout"].get(n)
        if not o:
            continue

        def ci(x):
            return f"{f(x['mean'])} [{f(x['lo'])}, {f(x['hi'])}]" if x and x.get("mean") is not None else "–"

        hold = h or {}
        lines.append(
            f"| {n} | {ci(o['top_excess'])} / {f((hold.get('top_excess') or {}).get('mean'))} | "
            f"{ci(o['top_bigup_minus_base'])} / {f((hold.get('top_bigup_minus_base') or {}).get('mean'))} | "
            f"{ci(o['top_bigdn_minus_base'])} / {f((hold.get('top_bigdn_minus_base') or {}).get('mean'))} | "
            f"{ci(o['bottom_excess'])} / {f((hold.get('bottom_excess') or {}).get('mean'))} | "
            f"{f(o['tail_auc_bigup'], '.4f')} / {f(hold.get('tail_auc_bigup'), '.4f')} | "
            f"{f(o['corr_vol'], '+.2f')} / {f(hold.get('corr_vol'), '+.2f')} |"
        )
        te, he = o["top_excess"], (hold.get("top_excess") or {})
        if (
            n in ("mc_abs", "ord_abs", "mc_exc")
            and te.get("lo") is not None
            and te["lo"] > 0
            and (he.get("mean") or 0) >= 0
        ):
            q2.append(n)
    lines += [
        "",
        f"**질문 2 판정(상위 10% 초과수익 외부 95% 하한 > 0 이고 홀드아웃 ≥ 0): "
        f"{', '.join(q2) if q2 else '해당 없음'}**",
    ]
    text = "\n".join(lines) + "\n"
    (OUT / f"{market}_multiclass_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    from quantcio.modeling.report import render_markdown

    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        print(write_summary(a.market))
        return
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    _, ds = v13s.load_inputs(a.market, settings, cache)
    f = ds.frame
    lab = f[f["labelled"] & f["is_me"]]
    share = {
        "class_share_abs": {
            int(k): round(float(v), 3)
            for k, v in pd.Series(classes(lab["r"].to_numpy(float))).value_counts(normalize=True).sort_index().items()
        },
        "class_share_exc": {
            int(k): round(float(v), 3)
            for k, v in pd.Series(classes(lab["y_e"].to_numpy(float))).value_counts(normalize=True).sort_index().items()
        },
    }
    say(f"{a.market} class shares {share}")
    sel.COMPOSITE_CANDIDATES = ()
    sel.COMPOSITES = ()
    served = SERVED[a.market]
    sel.DIAGNOSTIC_PAIRS = (
        ("mc_abs", "bin_med"),
        ("ord_abs", "bin_med"),
        ("mc_exc", "bin_med"),
        ("mc_abs", served),
        ("ord_abs", served),
        ("mc_exc", served),
    )
    ds.snapshot = f"{ds.snapshot}|multiclass"
    res = sel.run_selection(
        ds, settings, cache, candidates=candidates(a.market, list(ds.xs_cols)), threads=8, progress=say
    )
    vol = f["vol60"] if "vol60" in f else None
    names = [*MODELS, served]

    def keyed(p):
        if p is None:
            return None
        m = p.merge(f[["date", "entity"]].reset_index(), on=["date", "entity"], how="left").set_index("index")
        return m

    oos, hold = keyed(res.oos), keyed(res.holdout_preds)
    tails = {
        "tail_outer": tail_metrics(oos, vol, names),
        "tail_holdout": tail_metrics(hold, vol, names) if hold is not None else {},
    }
    res.protocol["multiclass"] = {"declared": "docs/MODELING.md 37", "edges": list(EDGES), **share, **tails}
    stem = OUT / f"{a.market}_multiclass"
    stem.with_suffix(".json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        f"# 종목 선택 모델 선정 — {a.market}", f"# 다중 분류 시험 (서빙 아님) — {a.market}", 1
    )
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    print(write_summary(a.market))


if __name__ == "__main__":
    main()
