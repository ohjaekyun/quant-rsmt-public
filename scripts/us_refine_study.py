"""US stock-selection refinement (docs/MODELING.md 35, pre-declared 2026-10-10): not new data but new ways to LEARN
from the v9 inputs - single-model refinement (lean inputs, fold-internal stability selection, IC-signed monotone
constraints) and target denoising (volatility-scaled residuals, a 20/40/60-session multi-horizon target), plus a fixed
blend of the target variants. Every variant is the same LightGBM regressor as ``single_all`` (docs/MODELING.md 33);
evaluation is protocol v3 (within-date AUC of the ORIGINAL 20-session excess return). Research only.

    .venv/bin/python scripts/us_refine_study.py

Writes reports/modeling/US_refine.{md,json} and reports/modeling/US_refine_summary.md.
"""

from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import us_v6_study as v6  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
EXTRA = ("y_rp_res", "y_vs_res", "y_mh", "t_end60", "vol60")
REF = "single_all"
VARIANTS = ("lean_core", "stable_fs", "mono_ic", "volscaled", "multih", "blend_t")
BLEND_MEMBERS = ("single_all", "volscaled", "multih")
STABLE_T, STABLE_SIGN, STABLE_MIN = 2.0, 0.7, 10  # stability selection: |t| >= 2 and the same sign in >= 70% of years
MONO_T = 2.0


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ targets


def forward_excess(frame: pd.DataFrame, prices: dict, bench: pd.Series, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k-session forward log excess return r_k - beta_s * m_k from close at t, and the date of session t + k
    (NaN / NaT when the stock has no close k sessions later - training targets only)."""
    sessions = pd.DatetimeIndex(bench.dropna().index)
    bpos = {d: i for i, d in enumerate(sessions)}
    bval = bench.reindex(sessions).to_numpy(float)
    out = np.full(len(frame), np.nan)
    tend = np.full(len(frame), np.datetime64("NaT"), dtype="datetime64[ns]")
    for ent, g in frame.groupby("entity", sort=False):
        px = prices.get(ent)
        if px is None or "close" not in px:
            continue
        c = px["close"].astype(float).reindex(sessions)
        cv = c.to_numpy(float)
        for ix, d, b in zip(g.index, g["date"], g["beta_s"], strict=True):
            i = bpos.get(pd.Timestamp(d))
            if i is None or i + k >= len(sessions):
                continue
            c0, c1 = cv[i], cv[i + k]
            if not (np.isfinite(c0) and np.isfinite(c1) and c0 > 0 and c1 > 0):
                continue
            j = frame.index.get_loc(ix)
            out[j] = math.log(c1 / c0) - float(b) * math.log(bval[i + k] / bval[i])
            tend[j] = sessions[i + k].to_datetime64()
    return out, tend


def residual_of(frame: pd.DataFrame, col: str) -> pd.Series:
    tmp = frame[["date", "x_log_mcap", "sector_group", "labelled"]].copy()
    tmp["y_e"] = frame[col]
    tmp["labelled"] = frame[col].notna()
    return v6.residual_target(tmp)


# ------------------------------------------------------------------------------------------------ fitters


class DataRefine(sel._Data):
    def __init__(self, ds):
        super().__init__(ds)
        for c in EXTRA:
            if c in ds.frame:
                self.f[c] = ds.frame.loc[self.f.index, c].to_numpy()


def _lgbm(train: pd.DataFrame, use: list[str], y: np.ndarray, params, scheme, ctx, mono=None):
    import lightgbm as lgb

    ok = np.isfinite(y)
    kw = v6._tree_kw(params, int(ok.sum()), ctx)
    if mono is not None:
        kw["monotone_constraints"] = mono
        kw["monotone_constraints_method"] = "advanced"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lgb.LGBMRegressor(objective="regression", **kw).fit(
            train[use].to_numpy(dtype=np.float64)[ok], y[ok], sample_weight=v6._weights(train, scheme, ctx, False)[ok]
        )


def _ic_table(train: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-date Spearman IC of each input vs y_e on the training month-ends (training rows only; a column with
    fewer than 30 values on a date gets NaN there)."""
    me = train[train["is_me"] & train["y_e"].notna()]
    cols = [c for c in cols if me[c].notna().sum() >= 100]
    out = {}
    for d, g in me.groupby("date", sort=True):
        x = g[cols].rank().to_numpy(dtype=float)
        y = g["y_e"].rank().to_numpy(dtype=float)
        ok = np.isfinite(x)
        n = ok.sum(axis=0)
        yy = np.where(ok, y[:, None], np.nan)
        xc = x - np.nanmean(x, axis=0)
        yc = yy - np.nanmean(yy, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.nansum(xc * yc, axis=0) / np.sqrt(np.nansum(xc**2, axis=0) * np.nansum(yc**2, axis=0))
        out[d] = np.where(n >= 30, r, np.nan)
    return pd.DataFrame.from_dict(out, orient="index", columns=cols).sort_index()


def _ic_stats(train: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    ic = _ic_table(train, cols)
    t = ic.mean() / (ic.std(ddof=1) / np.sqrt(ic.notna().sum()))
    yearly = ic.groupby(ic.index.year).mean()
    sign = np.sign(ic.mean())
    same = (np.sign(yearly) == sign).mean()
    return pd.DataFrame({"t": t, "same_sign": same})


def fitter_target(cols: list[str], target: str, purge60: bool = False):
    def fit(c, train, params, scheme, ctx):
        use = [x for x in [*cols, *ctx.regime_cols] if train[x].notna().any()]
        y = train[target].to_numpy(dtype=float)
        if purge60 and ctx.cutoff is not None:  # the 60-session label must end before the test block starts
            ok_end = pd.to_datetime(train["t_end60"]) < pd.Timestamp(ctx.cutoff)
            y = np.where(ok_end.to_numpy(), y, np.nan)
        return cmod.Fitted(c.name, params, scheme, _lgbm(train, use, y, params, scheme, ctx), use, "lgbm")

    return fit


def fitter_stable(cols: list[str]):
    def fit(c, train, params, scheme, ctx):
        st = _ic_stats(train, cols)
        keep = st[(st["t"].abs() >= STABLE_T) & (st["same_sign"] >= STABLE_SIGN)].index.tolist()
        if len(keep) < STABLE_MIN:
            keep = st["t"].abs().sort_values(ascending=False).index[:STABLE_MIN].tolist()
        use = [x for x in [*keep, *ctx.regime_cols] if train[x].notna().any()]
        y = train["y_rp_res"].to_numpy(dtype=float)
        f = cmod.Fitted(c.name, params, scheme, _lgbm(train, use, y, params, scheme, ctx), use, "lgbm")
        f.extra["selected"] = keep
        return f

    return fit


def fitter_mono(cols: list[str]):
    def fit(c, train, params, scheme, ctx):
        st = _ic_stats(train, cols)
        use = [x for x in [*cols, *ctx.regime_cols] if train[x].notna().any()]
        mono = []
        for x in use:
            t = float(st["t"].get(x, 0.0)) if x in st.index else 0.0
            mono.append(int(np.sign(t)) if abs(t) >= MONO_T else 0)
        y = train["y_rp_res"].to_numpy(dtype=float)
        return cmod.Fitted(c.name, params, scheme, _lgbm(train, use, y, params, scheme, ctx, mono), use, "lgbm")

    return fit


BLEND = cmod.Candidate(
    "blend_t",
    "ensemble",
    "고정 혼합(단일·변동성 조정·다기간 순위 평균)",
    "fixed blend of the target variants",
    tuned=False,
)
_orig_rule = sel.composite_rule


def composite_rule(name, inner, cands):
    if name == BLEND.name:
        return (list(BLEND_MEMBERS), [1.0] * len(BLEND_MEMBERS)) if all(m in inner for m in BLEND_MEMBERS) else ([], [])
    return _orig_rule(name, inner, cands)


def candidates(all_xs: list[str], core_xs: list[str]) -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    out = [every["mom_12_1"], every["factor_ic"], every["cat_rank"]]
    spec = [
        ("single_all", "LightGBM 회귀 · 잔차 순위 · 입력 전체(기준)", fitter_target(all_xs, "y_rp_res")),
        ("lean_core", "같은 학습기 · v3 핵심 입력만", fitter_target(core_xs, "y_rp_res")),
        ("stable_fs", "같은 학습기 · 학습 구간 안 안정 변수만(|t|≥2, 같은 부호 70%)", fitter_stable(all_xs)),
        ("mono_ic", "같은 학습기 · 학습 구간 IC 부호로 단조 제약", fitter_mono(all_xs)),
        ("volscaled", "같은 학습기 · 정답 = 잔차 ÷ 60일 변동성의 순위", fitter_target(all_xs, "y_vs_res")),
        ("multih", "같은 학습기 · 정답 = 20·40·60일 잔차 순위 평균", fitter_target(all_xs, "y_mh", purge60=True)),
    ]
    for name, ko, fitter in spec:
        key = f"refine__{name}"
        cmod._FITTERS[key] = fitter
        out.append(cmod.Candidate(name, "tree", ko, name, grid=list(v6.DEPTH_GRID), schemes=("none",), fitter=key))
    return out


# ------------------------------------------------------------------------------------------------ summary


def write_summary() -> str:
    d = json.loads((OUT / "US_refine.json").read_text())
    av = d["protocol"].get("alt_value") or {}

    def f(x, spec="+.4f"):
        return "–" if x is None else format(x, spec)

    lines = [
        "# 미국 종목 선택 정제·학습 방식 시험 (docs/MODELING.md 35)",
        "",
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
    lines += ["", f"## 기준 `{REF}` 대비 날짜별 AUC 차이 (단측 p, BH q = 0.10)", ""]
    rows = []
    for v in VARIANTS:
        k = f"{v} - {REF}"
        if k not in av:
            continue
        o, h = av[k]["outer"], av[k].get("holdout") or {}
        t = o.get("t")
        p = float(stats.norm.sf(t)) if t is not None and np.isfinite(t) else 1.0
        rows.append((v, o, h, p))
    m = len(rows)
    order = sorted(range(m), key=lambda i: rows[i][3])
    passed = set()
    for rank, i in enumerate(order, 1):  # Benjamini-Hochberg step-up
        if rows[i][3] <= 0.10 * rank / m:
            passed = {rows[j][0] for j in order[:rank]}
    for v, o, h, p in rows:
        hh = f" · 홀드아웃 {f(h.get('mean'))} [{f(h.get('lo'))}, {f(h.get('hi'))}]" if h.get("mean") is not None else ""
        win = v in passed and o["mean"] > 0 and (h.get("mean") is None or h["mean"] >= 0)
        lines.append(
            f"- {v}: 외부 {f(o['mean'])} [{f(o['lo'])}, {f(o['hi'])}] (t {f(o.get('t'), '+.2f')}, p {p:.3f}){hh}"
            f"{' → **개선(BH 통과, 홀드아웃 ≥ 0)**' if win else ''}"
        )
    ref = av.get(f"{REF} - cat_rank")
    if ref:
        lines += [
            "",
            f"- 참고 {REF} − cat_rank(서빙): 외부 {f(ref['outer']['mean'])} "
            f"[{f(ref['outer']['lo'])}, {f(ref['outer']['hi'])}]",
        ]
    sel_info = d["protocol"]["refine"].get("stable_selected_final")
    if sel_info:
        lines += ["", f"- stable_fs 최종 학습에서 고른 변수 {len(sel_info)}개: {', '.join(sel_info[:30])}"]
    text = "\n".join(lines) + "\n"
    (OUT / "US_refine_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    import us_v9_study as v9

    from quantcio.data.pit import load_pit_universe
    from quantcio.modeling.dataset import universe_params
    from quantcio.modeling.report import render_markdown
    from quantcio.models import Market

    if "--summary" in sys.argv:
        print(write_summary())
        return
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    n, start = universe_params(Market.US, settings)
    uni = load_pit_universe(Market.US, settings, cache, n, start)
    ds, _ = v9.build_inputs(settings, cache, universe=uni)
    f = ds.frame
    say("targets")
    res20 = v6.residual_target(f)
    f["y_rp_res"] = res20.groupby(f["date"]).rank(pct=True)
    f["vol60"] = f["vol60"] if "vol60" in f else np.nan
    f["y_vs_res"] = (res20 / f["vol60"].where(f["vol60"] > 0)).groupby(f["date"]).rank(pct=True)
    bench = uni.benchmarks["SPY"]["close"].astype(float)
    ranks = [f["y_rp_res"]]
    for k in (40, 60):
        f[f"_ex{k}"], tend = forward_excess(f, uni.prices, bench, k)
        if k == 60:
            f["t_end60"] = tend
        f[f"_rp{k}"] = residual_of(f, f"_ex{k}").groupby(f["date"]).rank(pct=True)
        ranks.append(f[f"_rp{k}"])
    f["y_mh"] = pd.concat(ranks, axis=1).mean(axis=1, skipna=False)
    all_xs, core_xs = list(ds.xs_cols), list(ds.core_xs_cols)
    lab = f[f["labelled"] & f["is_me"]]
    info = {
        "inputs_all": len(all_xs),
        "inputs_core": len(core_xs),
        "target_known": {c: round(float(lab[c].notna().mean()), 3) for c in ("y_rp_res", "y_vs_res", "y_mh")},
        "target_corr_with_ref": {
            c: round(float(lab[[c, "y_rp_res"]].corr(method="spearman").iloc[0, 1]), 3) for c in ("y_vs_res", "y_mh")
        },
    }
    say(f"targets: {info}")
    sel._Data = DataRefine
    sel.composite_rule = composite_rule
    sel.COMPOSITE_CANDIDATES = (BLEND,)
    sel.COMPOSITES = (BLEND.name,)
    sel.DIAGNOSTIC_PAIRS = (*((v, REF) for v in VARIANTS), (REF, "cat_rank"))
    ds.snapshot = f"{ds.snapshot}|refine"
    out = sel.run_selection(ds, settings, cache, candidates=candidates(all_xs, core_xs), threads=8, progress=say)
    fin = out.final.fitted if out.final is not None else {}
    stable = (fin.get("stable_fs").extra.get("selected") if fin and fin.get("stable_fs") is not None else None) or []
    out.protocol["refine"] = {"declared": "docs/MODELING.md 35", **info, "stable_selected_final": stable}
    (OUT / "US_refine.json").write_text(json.dumps(out.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(out, settings).replace(
        "# 종목 선택 모델 선정 — US", "# 미국 종목 선택 정제·학습 방식 시험 (서빙 아님) — US", 1
    )
    (OUT / "US_refine.md").write_text(md, encoding="utf-8")
    print(write_summary())


if __name__ == "__main__":
    main()
