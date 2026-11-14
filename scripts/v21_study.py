"""External suggestions (docs/MODELING.md 55, pre-declared 2026-10-13): the 54 pool + blocks I U (US) / Z F2 (KR),
the 52 candidates + PCA models, and a downside guardrail on the final model's portfolios. Research only.

    .venv/bin/python scripts/v21_study.py --market KR

Writes reports/modeling/<M>_v21.json and reports/modeling/<M>_v21_summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_select_study as fss  # noqa: E402
import portfolio_backtest as pb  # noqa: E402
import v19_study as v19s  # noqa: E402
import v20_study as v20s  # noqa: E402
import v21_data as v  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
_f = fss._f
_pd = v19s._pd


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ PCA candidates


def _fit_pca(c, train, params, scheme, ctx):
    x = np.nan_to_num(train[ctx.xs_cols].to_numpy(dtype=np.float64), nan=0.0)
    kind = "logit" if c.name == "pca_logit" else "ridge"
    if kind == "logit":
        y, reg = train["y_cls"].to_numpy(dtype=float), float(params["C"])
    else:
        y, reg = feat.rank_gauss(train["y_e"], train["date"]).to_numpy(dtype=float), float(params["a"])
    ok = np.isfinite(y)
    w = cmod.row_weights(train, scheme, ctx)[ok]
    model = v.PcaLinear(int(params["k"]), reg, kind, int(ctx.seed)).fit(x[ok], y[ok], w)
    return cmod.Fitted(c.name, params, scheme, model, list(ctx.xs_cols), kind)


def pca_candidates() -> list[cmod.Candidate]:
    cmod._FITTERS["pca_logit"] = _fit_pca
    cmod._FITTERS["pca_ridge"] = _fit_pca
    grid_l = [{"k": k, "C": c} for k in (10, 20, 40) for c in (0.005, 0.05)]
    grid_r = [{"k": k, "a": a} for k in (10, 20, 40) for a in (0.1, 1.0)]
    return [
        cmod.Candidate(
            "pca_logit", "linear", "PCA 차원 축소 + 로지스틱", "PCA + logistic", grid=grid_l, schemes=cmod._RANK_SCHEMES
        ),
        cmod.Candidate(
            "pca_ridge",
            "linear",
            "PCA 차원 축소 + 순위 릿지",
            "PCA + rank ridge",
            grid=grid_r,
            schemes=cmod._RANK_SCHEMES,
        ),
    ]


# ------------------------------------------------------------------------------------------------ blocks


def add_blocks(market: str, ds, settings, cache) -> tuple[list[str], list[str], dict]:
    f = ds.frame
    df, info = v.block(market, f, ds, settings, cache)
    xs, reg = [], []
    for c in df.columns:
        vals = df[c].astype(float)
        if vals.notna().sum() == 0 or vals.nunique() < 2:
            info.setdefault("empty", []).append(c)
            continue
        f[c] = vals.to_numpy()
        feat.FEATURE_DOCS[c] = v.DOCS.get(c, (c, c))
        if c in v.REGIME:
            reg.append(c)
        else:
            f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
            xs.append(feat.x_col(c))
    lab = f[f["labelled"] & f["is_me"]]
    info["missing_share"] = {c: round(float(lab[c].isna().mean()), 3) for c in df.columns if c in f}
    return xs, reg, info


# ------------------------------------------------------------------------------------------------ guardrail


def risky_mask(market: str, f: pd.DataFrame) -> pd.Series:
    """Per (date, entity) row of the frame: True = excluded by the 55.3 guardrail."""
    if market == "KR":
        zombie = f["zi_zombie"].fillna(0.0) >= 1 if "zi_zombie" in f else pd.Series(False, index=f.index)
        q90 = (
            f.groupby("date")["zi_rate_exposure"].transform(lambda s: s.quantile(0.9))
            if "zi_rate_exposure" in f
            else None
        )
        expo = (f["zi_rate_exposure"] >= q90) & f["zi_rate_exposure"].notna() if q90 is not None else False
        return zombie | expo
    if "b_int_cov" not in f:
        return pd.Series(False, index=f.index)
    return f["b_int_cov"].lt(0) & f["b_int_cov"].notna()


def _cvar5(per: pd.DataFrame) -> float:
    r = per["ret"].to_numpy(float)
    if not len(r):
        return float("nan")
    cut = np.quantile(r, 0.05)
    return float(r[r <= cut].mean())


def guardrail(market: str, res, uni, ds, name: str) -> dict:
    parts = [res.oos[["date", "entity", name]].assign(part="outer")]
    if res.holdout_preds is not None and name in res.holdout_preds:
        parts.append(res.holdout_preds[["date", "entity", name]].assign(part="holdout"))
    sc = pd.concat(parts, ignore_index=True).rename(columns={name: "score"})
    f = ds.frame
    flags = f[["date", "entity", "log_mcap"]].assign(risky=risky_mask(market, f).to_numpy(bool))
    flags = flags.drop_duplicates(["date", "entity"])
    sc = sc.merge(flags, on=["date", "entity"], how="left")
    sc["mcap"] = np.exp(sc["log_mcap"].astype(float))
    sc["risky"] = sc["risky"].fillna(False).astype(bool)
    guarded = sc.assign(score=sc["score"].where(~sc["risky"]))
    cal = pd.DatetimeIndex(ds.calendar)
    cal = cal[cal >= pd.Timestamp(sc["date"].min())]
    w = pb.wide(uni.prices, sorted(sc["entity"].unique()), cal)
    mid = pb.cm.AUM[market][1]
    out: dict = {"flagged_share": round(float(sc["risky"].mean()), 3)}
    for p, bk in pb.BENCH.items():
        bn, bp = pb.simulate(market, sc, w, bk, mid, 1.0, False)
        for tag, s in (("base", sc), ("guard", guarded)):
            nav, per = pb.simulate(market, s, w, p, mid, 1.0, True)
            st = pb.stats(nav, bn, per, bp)
            st["cvar5"] = _cvar5(per)
            st["stress"] = {k: pb.window(nav, bn, a, b) for k, (a, b) in pb.STRESS.items()}
            out[f"{p}|{tag}"] = st
    ok = []
    for p in pb.BENCH:
        b, g = out[f"{p}|base"], out[f"{p}|guard"]
        ok.append(g["mdd"] > b["mdd"] and g["cvar5"] > b["cvar5"] and g["excess_ann"] >= b["excess_ann"] - 0.01)
    out["helps"] = bool(all(ok))
    return out


# ------------------------------------------------------------------------------------------------ report


def write_summary(market: str, r: dict) -> str:
    d55, d54 = r["selection55"], r["selection54"]
    c55 = {c["name"]: c for c in d55["candidates"]}
    c54 = {c["name"]: c for c in d54["candidates"]}
    lines = [
        f"# {market} 외부 제안 반영 → 최종 모델 (docs/MODELING.md 55)",
        "",
        f"- 풀: 54절 {r['pool54']} → 55절 {r['pool55']}; 새 특징 {', '.join(r['new_xs']) or '없음'}",
        f"- 새 국면: {', '.join(r['new_regime']) or '없음'}",
        f"- 결측 누출 의심 제외: {', '.join(r['leak_suspect']) or '없음'}",
        f"- 자료: {r['block_info_short']}",
        "",
        "| 후보 | 55절 외부 AUC [95% CI] | 54절 외부 AUC | 55절 홀드아웃 | 54절 홀드아웃 | 55절 외부 IC (t) |",
        "|---|---|---|---|---|---|",
    ]
    for n, c in c55.items():
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        p = c54.get(n) or {}
        po = (p.get("outer") or {}).get("auc_median") or {}
        ph = ((p.get("holdout") or {}).get("auc_median")) or {}
        ic = c["outer"]["ic"]
        star = " **(선정)**" if n == d55["selected"] else ""
        lines.append(
            f"| {n}{star} | {_f(o['mean'], '.4f')} [{_f(o['lo'], '.4f')}, {_f(o['hi'], '.4f')}] | "
            f"{_f(po.get('mean'), '.4f')} | {_f(h.get('mean'), '.4f')} | {_f(ph.get('mean'), '.4f')} | "
            f"{_f(ic['mean'])} ({_f(ic['t'], '+.2f')}) |"
        )
    g = d55["gate"]
    lines += [
        "",
        f"**최종 모델: `{d55['selected']}`** · 게이트 {'통과' if g['passed'] else '미달'} · "
        f"54절 선정 `{d54['selected']}`",
        "",
    ]
    for k, x in r["vs"].items():
        o, h = x["outer"], x.get("holdout") or {}
        lines.append(
            f"- {k}: 외부 {_f(o['mean'])} [{_f(o['lo'])}, {_f(o['hi'])}] (t {_f(o['t'], '+.2f')}) · "
            f"홀드아웃 {_f(h.get('mean'))} [{_f(h.get('lo'))}, {_f(h.get('hi'))}]"
        )
    cb = r["contribution_55_54"]
    o, h = cb["mean_outer"], cb["mean_holdout"]
    lines += [
        "",
        f"- 같은 fs 모델 9개, 55 − 54 풀 차이(모델 평균): 외부 {_f(o['mean'])} [{_f(o['lo'])}, {_f(o['hi'])}] · "
        f"홀드아웃 {_f(h['mean'])} → **{'도움' if cb['helps'] else '도움 없음'}**",
        "",
        "## 새 특징 선택 비율",
        "",
    ]
    lines += [f"- {c}: {s:.0%}" for c, s in r["new_selected"]]
    gr = r.get("guardrail") or {}
    if gr:
        lines += [
            "",
            f"## 하방 가드레일 (제외 비율 {gr['flagged_share']:.1%}, 비용 후, 중간 규모)",
            "",
            "| 포트폴리오 | 연 수익 | 최대 낙폭 | 월 CVaR 5% | 기준 대비 연 초과 | 2020 코로나 급락 | 2022 긴축 |",
            "|---|---|---|---|---|---|---|",
        ]
        for k in ("P1|base", "P1|guard", "P2|base", "P2|guard"):
            s = gr[k]
            c19 = (s["stress"].get("코로나 급락") or {}).get("ret")
            c22 = (s["stress"].get("2022 긴축 약세장") or {}).get("ret")
            lines.append(
                f"| {k} | {_f(s['cagr'], '+.1%')} | {_f(s['mdd'], '+.1%')} | {_f(s['cvar5'], '+.1%')} | "
                f"{_f(s['excess_ann'], '+.1%')} | {_f(c19, '+.1%')} | {_f(c22, '+.1%')} |"
            )
        lines += ["", f"→ 가드레일 **{'도움' if gr['helps'] else '도움 없음'}** (55.3 기준)"]
    text = "\n".join(lines) + "\n"
    (OUT / f"{market}_v21_summary.md").write_text(text, encoding="utf-8")
    return text


def run_through_55(m: str, settings, cache) -> dict:
    """The 52-54 pools (v20_study.run_through_54) + 55.1 blocks + PCA candidates and the 55 selection run."""
    st = v20s.run_through_54(m, settings, cache)
    ds = st["ds"]
    xs54, reg54, snap54 = list(ds.xs_cols), list(ds.regime_cols), ds.snapshot
    new_xs, new_reg, binfo = add_blocks(m, ds, settings, cache)
    aucs, sus = v20s.leak_screen(ds, new_xs)
    new_xs = [x for x in new_xs if x not in set(sus)]
    ds.xs_cols, ds.regime_cols = [*xs54, *new_xs], [*reg54, *new_reg]
    ds.snapshot = f"{snap54}|v21|{len(new_xs)}|{len(new_reg)}|pca"
    say(f"55 pool: +{len(new_xs)} features, +{len(new_reg)} regime (dropped as leak suspects: {sus})")
    v19s._store(settings, m, ds.snapshot)
    cands = [*fss.candidates(m), *pca_candidates()]
    res55 = sel.run_selection(ds, settings, cache, candidates=cands, threads=8, progress=say)
    return {
        **st,
        "xs54": xs54,
        "reg54": reg54,
        "new_xs": new_xs,
        "new_reg": new_reg,
        "binfo": binfo,
        "aucs55": aucs,
        "sus55": sus,
        "cands55": cands,
        "res55": res55,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    m = ap.parse_args().market
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    st = run_through_55(m, settings, cache)
    uni, ds, res54, res55 = st["uni"], st["ds"], st["res54"], st["res55"]
    xs54, reg54, new_xs, new_reg, binfo = st["xs54"], st["reg54"], st["new_xs"], st["new_reg"], st["binfo"]
    aucs, sus = st["aucs55"], st["sus55"]
    d55, d54 = res55.to_dict(), res54.to_dict()
    core = f"{fss.SERVED[m]}_core"
    best = d55["selected"]
    vs = {}
    for ref, rres, tag in ((core, res55, "55"), (d54["selected"], res54, "54")):
        if ref == best and tag == "55":
            continue
        vs[f"{best}(55) − {ref}({tag})"] = {
            "outer": mt.paired_difference(_pd(res55, best, "outer"), _pd(rres, ref, "outer")),
            "holdout": mt.paired_difference(_pd(res55, best, "holdout"), _pd(rres, ref, "holdout")),
        }
    n55 = len(ds.xs_cols)
    fits = [x for k, x in fss.FS_LOG.items() if k.split("|")[2] == str(n55)]
    cnt = Counter(c for x in fits for c in x["xs"])
    short = {k: x for k, x in binfo.items() if k not in ("missing_share", "empty")}
    r = {
        "market": m,
        "pool54": {"xs": len(xs54), "regime": len(reg54)},
        "pool55": {"xs": n55, "regime": len(ds.regime_cols)},
        "new_xs": new_xs,
        "new_regime": new_reg,
        "block_info": binfo,
        "block_info_short": json.dumps(short, ensure_ascii=False, default=str)[:400],
        "missing_auc": aucs,
        "leak_suspect": sus,
        "selection54": d54,
        "selection55": d55,
        "vs": vs,
        "contribution_55_54": v19s.contribution(res55, res54),
        "new_selected": sorted(((c, cnt.get(c, 0) / max(len(fits), 1)) for c in new_xs), key=lambda t: -t[1]),
    }
    try:
        r["portfolio"] = fss.portfolio(m, res55, uni, ds, best)
        r["guardrail"] = guardrail(m, res55, uni, ds, best)
    except Exception as exc:  # noqa: BLE001 - the selection result stands without the portfolio tables
        say(f"portfolio / guardrail skipped: {type(exc).__name__}: {exc}")
    (OUT / f"{m}_v21.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    print(write_summary(m, r), flush=True)


if __name__ == "__main__":
    main()
