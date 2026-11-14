"""Score vs risk-engine overlap and the macro overlay in event windows (docs/MODELING.md 56.2 / 56.3, pre-declared
2026-10-13). Research only.

    .venv/bin/python scripts/risk_overlap_study.py run --market KR   # 56.2 sizing comparison + 56.3 overlay
    .venv/bin/python scripts/risk_overlap_study.py record           # weekly: rank corr(score, vol) of the served models

Outputs: reports/strategy/risk_overlap_<M>.{md,json}; ledger alt_ledger/monitor/score_risk.jsonl.
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
import portfolio_backtest as pb  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "strategy"
LEDGER = PROJECT_ROOT / "alt_ledger" / "monitor" / "score_risk.jsonl"
COST = 0.003  # per unit of one-way turnover (the selection report's 30 bp x turnover)
VOL_FLOOR = 0.10
SCHEMES = {"EW": "동일가중", "SCORE": "점수 비례", "REPORT": "리포트 규칙(점수 ÷ 변동성)", "INVVOL": "변동성 역수"}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ewma_vol(px: pd.DataFrame, lam: float = 0.94) -> pd.Series:
    """Annualised EWMA (RiskMetrics, lambda 0.94) volatility of daily log returns - the sizing fallback."""
    r = np.log(px["close"].astype(float)).diff()
    return np.sqrt((r**2).ewm(alpha=1 - lam, adjust=False).mean() * 252)


def _spearman_by_date(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    def one(g):
        g = g[[a, b]].dropna()
        return g[a].rank().corr(g[b].rank()) if len(g) >= 20 else np.nan

    return df.groupby("date")[[a, b]].apply(one).dropna()


def _book_stats(ret: pd.Series, bench: pd.Series) -> dict:
    nav = (1 + ret).cumprod()
    ex = ret - bench
    worst = ret[ret <= ret.quantile(0.05)]
    return {
        "ann_ret": float((nav.iloc[-1]) ** (12 / len(ret)) - 1),
        "sharpe": float(ret.mean() / ret.std(ddof=1) * np.sqrt(12)) if ret.std(ddof=1) > 0 else float("nan"),
        "mdd": float((nav / nav.cummax() - 1).min()),
        "cvar5": float(worst.mean()),
        "excess_ann": float(ex.mean() * 12),
        "excess_t": float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))) if len(ex) > 2 else float("nan"),
        "months": len(ret),
    }


def sizing(sc: pd.DataFrame) -> dict:
    """Top-quintile books under the four weighting schemes (monthly, 20-session forward returns, 30 bp turnover)."""
    out: dict = {}
    rets = {k: [] for k in SCHEMES}
    hv_share = {k: [] for k in SCHEMES}
    bench, dates = [], []
    prev = {k: pd.Series(dtype=float) for k in SCHEMES}
    sc = sc.assign(R=np.expm1(sc["r"].astype(float)))  # r is a LOG return: books and benchmark use simple returns
    for d, g in sc.groupby("date"):
        g = g.dropna(subset=["score", "R"])
        if len(g) < 25:
            continue
        z = (g["score"] - g["score"].mean()) / (g["score"].std(ddof=0) or 1.0)
        top = g[g["score"].rank(pct=True) > 0.8].copy()
        top["z"] = z.loc[top.index]
        vol = top["ewma"].fillna(top["vol60"]).clip(lower=VOL_FLOOR)
        hv = top["vol60"] >= g["vol60"].quantile(2 / 3)
        raw = {
            "EW": pd.Series(1.0, index=top.index),
            "SCORE": top["z"].clip(lower=1e-6),
            "REPORT": top["z"].clip(lower=1e-6) / vol,
            "INVVOL": 1.0 / vol,
        }
        for k, w in raw.items():
            w = w / w.sum()
            w.index = top["entity"].to_numpy()
            turn = 0.5 * w.sub(prev[k], fill_value=0.0).abs().sum()
            rets[k].append(float((w.to_numpy() * top["R"].to_numpy()).sum() - COST * turn))
            hv_share[k].append(float(w[hv.to_numpy()].sum()))
            prev[k] = w
        bench.append(float(g["R"].mean()))
        dates.append(d)
    b = pd.Series(bench, index=dates)
    for k in SCHEMES:
        r = pd.Series(rets[k], index=dates)
        out[k] = {**_book_stats(r, b), "high_vol_weight": float(np.mean(hv_share[k]))}
        out[k]["_ret"] = r
    for a, c in (("REPORT", "SCORE"), ("REPORT", "EW"), ("SCORE", "EW")):
        out[f"{a}-{c}"] = mt.summarize((out[a]["_ret"] - out[c]["_ret"]).to_numpy())
    for k in SCHEMES:
        out[k].pop("_ret")
    s, e = out["REPORT-SCORE"], out["REPORT-EW"]
    worse = (
        out["REPORT"]["sharpe"] < min(out["EW"]["sharpe"], out["SCORE"]["sharpe"])
        and out["REPORT"]["excess_ann"] < min(out["EW"]["excess_ann"], out["SCORE"]["excess_ann"])
        and (s.get("t") or 0) <= -2
    )
    out["verdict"] = "중복 벌점이 손해" if worse else "중복이 해롭다는 근거 없음"
    out["_t"] = {"report_vs_score_t": s.get("t"), "report_vs_ew_t": e.get("t")}
    return out


def overlay(market: str, sc: pd.DataFrame, uni, ds, settings, cache) -> dict:
    """56.3: P1 / P2 daily NAV x the month-end macro overlay multiplier of the market's region (rest cash at 0)."""
    from quantcio.macro import load_macro_series
    from quantcio.macro.regime import overlay_multiplier, regime_history

    hist = regime_history(settings, cache, series=load_macro_series(settings, cache))
    region = "KR" if market == "KR" else "US"
    h = hist[hist["region"].astype(str).str.upper() == region].set_index("date")["score"].sort_index()
    mult = h.map(
        lambda x: overlay_multiplier(None if pd.isna(x) else float(x), settings.macro.overlay_min_mult, warn=False)
    )
    cal = pd.DatetimeIndex(ds.calendar)
    cal = cal[cal >= pd.Timestamp(sc["date"].min())]
    w = pb.wide(uni.prices, sorted(sc["entity"].unique()), cal)
    mid = pb.cm.AUM[market][1]
    out: dict = {"overlay_share_below_1": float((mult < 0.999).mean()), "regions": sorted(hist["region"].unique())}
    for p, bk in pb.BENCH.items():
        bn, _ = pb.simulate(market, sc, w, bk, mid, 1.0, False)
        nav, _ = pb.simulate(market, sc, w, p, mid, 1.0, True)
        r = nav.pct_change().fillna(0.0)
        m = mult.reindex(r.index, method="ffill").shift(1).fillna(1.0)  # the month-end score acts from the next session
        nav_ov = (1 + r * m).cumprod() * nav.iloc[0]
        for tag, n in (("base", nav), ("overlay", nav_ov)):
            mo = n.resample("ME").last().pct_change().dropna()
            bm = bn.resample("ME").last().pct_change().reindex(mo.index).fillna(0.0)
            st = _book_stats(mo, bm)
            st["mdd_daily"] = pb.mdd(n)  # _book_stats' mdd is on month-end NAV
            st["stress"] = {k: pb.window(n, bn, a, b) for k, (a, b) in pb.STRESS.items()}
            out[f"{p}|{tag}"] = st
    ok = []
    for p in pb.BENCH:
        b, o = out[f"{p}|base"], out[f"{p}|overlay"]
        ok.append(o["mdd"] > b["mdd"] and o["cvar5"] > b["cvar5"] and o["ann_ret"] >= b["ann_ret"] - 0.01)
    out["helps"] = bool(all(ok))
    return out


def _f(x, spec="+.1%") -> str:
    return "–" if x is None or not np.isfinite(x) else format(x, spec)


def write_md(market: str, r: dict) -> str:
    lines = [f"# {market} 점수와 위험 엔진의 중복 · 사건 구간 오버레이 (docs/MODELING.md 56.2 / 56.3)", ""]
    c = r["corr"]
    lines += ["## 56.2 점수와 변동성의 날짜별 순위 상관 (외부 + 홀드아웃)", ""]
    for k, v in c.items():
        lines.append(
            f"- 점수 × {k}: 평균 {_f(v['mean'], '+.3f')} [{_f(v['lo'], '+.3f')}, {_f(v['hi'], '+.3f')}] "
            f"(t {_f(v['t'], '+.1f')}, {v['n']}개 날짜)"
        )
    s = r["sizing"]
    lines += [
        "",
        "## 56.2 상위 20% 안의 비중 방식 (월간, 20거래일 수익, 비용 30bp × 회전율)",
        "",
        "| 방식 | 연 수익 | 샤프 | 최대 낙폭 | 월 CVaR 5% | 기준 대비 연 초과 (t) | 고변동성 3분위 비중 |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, ko in SCHEMES.items():
        x = s[k]
        lines.append(
            f"| {ko} | {_f(x['ann_ret'])} | {_f(x['sharpe'], '.2f')} | {_f(x['mdd'])} | {_f(x['cvar5'])} | "
            f"{_f(x['excess_ann'])} ({_f(x['excess_t'], '+.2f')}) | {_f(x['high_vol_weight'], '.0%')} |"
        )
    for k in ("REPORT-SCORE", "REPORT-EW"):
        x = s[k]
        lines.append(
            f"- 월 차이 {k}: {_f(x['mean'], '+.2%')} [{_f(x['lo'], '+.2%')}, {_f(x['hi'], '+.2%')}] "
            f"(t {_f(x['t'], '+.2f')})"
        )
    lines += ["", f"→ **{s['verdict']}**", ""]
    o = r.get("overlay") or {}
    if o:
        lines += [
            f"## 56.3 매크로 오버레이 (오버레이가 1 미만인 달 {o['overlay_share_below_1']:.0%})",
            "",
            "| 포트폴리오 | 연 수익 | 최대 낙폭 | 월 CVaR 5% | 기준 대비 연 초과 "
            "| 코로나 급락 | 2022 긴축 | 2024-12 계엄 | 2025-04 관세 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for k in ("P1|base", "P1|overlay", "P2|base", "P2|overlay"):
            x = o[k]
            st = x["stress"]
            cells = [
                _f((st.get(w) or {}).get("ret"))
                for w in ("코로나 급락", "2022 긴축 약세장", "2024-12 계엄(한국)", "2025-04 관세 충격")
            ]
            lines.append(
                f"| {k} | {_f(x['ann_ret'])} | {_f(x['mdd'])} | {_f(x['cvar5'])} | {_f(x['excess_ann'])} | "
                + " | ".join(cells)
                + " |"
            )
        lines += ["", f"→ 오버레이 **{'도움' if o['helps'] else '도움 없음'}** (55.3절과 같은 기준)"]
    text = "\n".join(lines) + "\n"
    (OUT / f"risk_overlap_{market}.md").write_text(text, encoding="utf-8")
    return text


def run(market: str) -> None:
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    say(f"{market}: out-of-sample scores of the served model")
    sc, uni, ds = pb.oos_scores(market, settings, cache)
    f = ds.frame
    cols = [c for c in ("vol60", "ivol60", "dvol60", "r") if c in f]
    sc = sc.merge(f[["date", "entity", *cols]].drop_duplicates(["date", "entity"]), on=["date", "entity"], how="left")
    ew = []
    for e, g in sc.groupby("entity"):
        px = uni.prices.get(e)
        if px is None or len(px) < 30:
            continue
        v = ewma_vol(px.sort_index())
        ew.append(
            pd.DataFrame(
                {"date": g["date"].to_numpy(), "entity": e, "ewma": v.reindex(g["date"], method="ffill").to_numpy()}
            )
        )
    sc = sc.merge(pd.concat(ew, ignore_index=True), on=["date", "entity"], how="left")
    corr = {}
    for col, ko in (("vol60", "60일 변동성"), ("ivol60", "60일 고유 변동성"), ("ewma", "EWMA 변동성(사이징 대체값)")):
        if col in sc:
            corr[ko] = mt.summarize(_spearman_by_date(sc, "score", col).to_numpy())
    r = {"market": market, "corr": corr, "sizing": sizing(sc)}
    try:
        r["overlay"] = overlay(market, sc, uni, ds, settings, cache)
    except Exception as exc:  # noqa: BLE001 - the sizing result stands without the overlay table
        say(f"overlay skipped: {type(exc).__name__}: {exc}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"risk_overlap_{market}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    print(write_md(market, r), flush=True)


def record() -> None:
    """Weekly monitor: rank correlation of the served models' latest scores with their volatility columns."""
    from quantcio.modeling.serve import load_artifact
    from quantcio.models import Market

    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    today = str(pd.Timestamp.now(tz="Asia/Seoul").date())
    with LEDGER.open("a") as fh:
        for m in ("KR", "US"):
            art = load_artifact(Market(m), settings, cache)
            if art is None:
                continue
            lt = art.latest
            row = {"recorded": today, "market": m, "as_of": str(pd.Timestamp(art.as_of).date()), "model": art.selected}
            for c in ("vol20", "vol60", "ivol60", "dvol60"):
                if c in lt:
                    g = lt[["score", c]].dropna()
                    row[f"corr_{c}"] = round(float(g["score"].rank().corr(g[c].rank())), 4) if len(g) > 20 else None
            row["n"] = int(lt["score"].notna().sum())
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            say(f"{m}: {row}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--market", choices=["KR", "US"], default="KR")
    sub.add_parser("record")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.market)
    else:
        record()


if __name__ == "__main__":
    main()
