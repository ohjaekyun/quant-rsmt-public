"""Beta-hedged, beta-neutral long-short and low-volatility control portfolios of the 77 score (docs/MODELING.md 78,
pre-declared 2026-10-25). Research only.

    .venv/bin/python scripts/hedged_study.py --market KR

Same walk-forward score as scripts/decile_study.py. Each held name is hedged at rebalance with beta_s (the label's
shrunk ex-ante beta) times its own benchmark (KR KOSPI / KOSDAQ, US SPY), fixed to the next rebalance. H10 = top decile
+ hedge, LS = H10 - (bottom decile + hedge) with a borrow fee, LV10 = lowest vol60 decile + hedge, and H10 - LV10.
Writes reports/strategy/hedged_<M>.{md,json}.
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
import decile_study as dc  # noqa: E402  (also installs the decile targets in portfolio_backtest)
import fs_select_study as fss  # noqa: E402
import portfolio_backtest as pb  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling.dataset import PRIMARY_BENCH  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "strategy"
BORROW = {"KR": 0.02, "US": 0.005}  # annual fee on the short notional
KR_SHORT_BAN = [("2020-03-16", "2021-05-02"), ("2023-11-06", "2025-03-30")]
PART_KO = {"all": "전체", "outer": "표본 외", "holdout": "홀드아웃", "short_ok": "공매도 가능 달"}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bench_closes(uni, cal: pd.DatetimeIndex) -> dict[str, pd.Series]:
    out = {}
    for k, b in uni.benchmarks.items():
        if b is not None and not b.empty:
            c = b["close"].astype(float)
            out[k] = c.reindex(cal.union(c.index)).ffill(limit=3).reindex(cal)
    return out


def hedge_path(market: str, sc: pd.DataFrame, w: dict, kind: str, uni, bc: dict[str, pd.Series]) -> pd.DataFrame:
    """Per rebalance period: the same holdings as portfolio_backtest.simulate (same dates, execution, price filter,
    targets), their mean stock return, the hedge return mean(beta_i x benchmark_i) and the primary benchmark return."""
    cal = w["close"].index
    dates = sorted(sc["date"].unique())
    execs = [cal[min(cal.searchsorted(pd.Timestamp(d), "right"), len(cal) - 1)] for d in dates]
    by_date = {d: g for d, g in sc.groupby("date")}
    primary = bc[PRIMARY_BENCH[Market(market)]]
    rows = []
    for i, (d, ex) in enumerate(zip(dates, execs, strict=True)):
        g = by_date[d]
        ok = w["close"].loc[ex].reindex(g["entity"]).notna().to_numpy()
        tw = pb.targets(g[ok], kind)
        end = execs[i + 1] if i + 1 < len(execs) else cal[min(cal.searchsorted(ex) + 21, len(cal) - 1)]
        if not len(tw) or end <= ex:
            rows.append({"date": d, "part": g["part"].iloc[0], "r": 0.0, "hedge": 0.0, "mkt": 0.0, "beta": np.nan})
            continue
        ents = list(tw.index)
        c0, c1 = w["close"].loc[ex, ents], w["close"].loc[end, ents]
        r = (c1 / c0 - 1.0).fillna(0.0).to_numpy()
        gb = g.set_index("entity").loc[ents]
        beta = gb["beta_s"].astype(float).fillna(1.0).to_numpy()
        bret = np.array(
            [(lambda s: s.loc[end] / s.loc[ex] - 1.0)(bc.get(uni.entity_benchmark.get(e, ""), primary)) for e in ents],
            dtype=float,
        )
        bret = np.where(np.isfinite(bret), bret, 0.0)
        wts = tw.to_numpy(dtype=float)
        rows.append(
            {
                "date": d,
                "exec": ex,
                "part": g["part"].iloc[0],
                "r": float(wts @ r),
                "hedge": float(wts @ (beta * bret)),
                "mkt": float(primary.loc[end] / primary.loc[ex] - 1.0),
                "beta": float(wts @ beta),
            }
        )
    return pd.DataFrame(rows)


def mstats(r: np.ndarray, mkt: np.ndarray) -> dict:
    n = len(r)
    if n < 3:
        return {"months": n}
    sd = r.std(ddof=1)
    nav = np.cumprod(1 + r)
    return {
        "months": int(n),
        "ann": float(nav[-1] ** (12 / n) - 1),
        "vol": float(sd * np.sqrt(12)),
        "sharpe": float(r.mean() / sd * np.sqrt(12)) if sd > 0 else None,
        "t": float(r.mean() / (sd / np.sqrt(n))) if sd > 0 else None,
        "hit": float((r > 0).mean()),
        "mdd": float((nav / np.maximum.accumulate(nav) - 1).min()),
        "beta_ex_post": float(np.cov(r, mkt)[0, 1] / np.var(mkt, ddof=1)) if np.var(mkt) > 0 else None,
    }


def masks(per: pd.DataFrame, market: str) -> dict[str, np.ndarray]:
    m = {"all": np.ones(len(per), bool), "outer": (per["part"] == "outer").to_numpy()}
    m["holdout"] = (per["part"] == "holdout").to_numpy()
    if market == "KR":
        ex = pd.to_datetime(per["exec"])
        ban = np.zeros(len(per), bool)
        for a, b in KR_SHORT_BAN:
            ban |= ((ex >= a) & (ex <= b)).to_numpy()
        m["short_ok"] = ~ban
    return m


def label_steps(ds, sc: pd.DataFrame) -> dict:
    """78.1 steps 1-3 on the label window: mean per decile (per-date means, averaged) of the log label y_e, the
    simple-return version and the price-only simple version."""
    f = ds.frame[ds.frame["is_me"] & ds.frame["labelled"]].drop_duplicates(["date", "entity"])
    cols = ["date", "entity", "r", "m"] + (["r_price"] if "r_price" in f else [])
    lab = sc[["date", "entity", "score", "part", "beta_s", "y_e"]].merge(f[cols], on=["date", "entity"])
    lab = lab.dropna(subset=["score", "y_e"])
    lab["dec"] = lab.groupby("date")["score"].transform(
        lambda x: np.ceil(x.rank(pct=True, method="first") * 10).clip(1, 10)
    )
    b = lab["beta_s"].astype(float).fillna(1.0)
    lab["simple"] = np.expm1(lab["r"]) - b * np.expm1(lab["m"])
    lab["price"] = np.expm1(lab["r_price"] if "r_price" in lab else lab["r"]) - b * np.expm1(lab["m"])
    out = {}
    for col in ("y_e", "simple", "price"):
        for part in ("outer", "holdout"):
            g = lab[(lab["part"] == part) & lab["dec"].isin([1, 10])]
            mm = g.groupby(["date", "dec"])[col].mean().unstack()
            out[f"{col}|{part}"] = {"D10": float(mm[10].mean()), "D1": float(mm[1].mean())}
    return out


def same_close(sc: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Scores re-dated to the previous session, so the 'next session' execution of simulate / hedge_path is the
    scoring day's own close (78.1 step 4 - not tradable, a diagnostic)."""
    d = pd.DatetimeIndex(sc["date"])
    k = np.clip(cal.searchsorted(d, "left") - 1, 0, None)
    return sc.assign(date=cal[k])


def run(market: str) -> dict:
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    uni, ds, info = fss.build_pool(market, settings, cache)
    sc, log = dc.walk_forward(ds)
    me = ds.frame[ds.frame["is_me"]].drop_duplicates(["date", "entity"])[["date", "entity", "beta_s", "vol60"]]
    sc = sc.merge(me, on=["date", "entity"], how="left")
    lv = sc.assign(score=-sc["vol60"].astype(float))  # LV10 = the "D10" of minus volatility
    cal = pd.DatetimeIndex(ds.calendar)
    cal = cal[cal >= pd.Timestamp(sc["date"].min())]
    w = pb.wide(uni.prices, sorted(sc["entity"].unique()), cal)
    bc = bench_closes(uni, cal)
    mid = pb.cm.AUM[market][1]
    legs = {}
    for name, frame, kind in (("top", sc, "D10"), ("bottom", sc, "D1"), ("lowvol", lv, "D10")):
        _, per = pb.simulate(market, frame, w, kind, mid, 1.0, False)
        _, nper = pb.simulate(market, frame, w, kind, mid, 1.0, True)
        hp = hedge_path(market, frame, w, kind, uni, bc)
        gap = float(np.max(np.abs(hp["r"].to_numpy() - per["ret"].to_numpy())))
        say(f"{name}: holdings check |own leg return - simulate| max {gap:.2e}, mean beta {hp['beta'].mean():.2f}")
        legs[name] = hp.assign(
            gross=per["ret"].to_numpy(), cost=per["ret"].to_numpy() - nper["ret"].to_numpy(), turnover=per["turnover"]
        )
    t, b, lvl = legs["top"], legs["bottom"], legs["lowvol"]
    fee = BORROW[market] / 12
    series = {
        "H10": (t["gross"] - t["hedge"], t["gross"] - t["hedge"] - t["cost"]),
        "LS": (
            (t["gross"] - t["hedge"]) - (b["gross"] - b["hedge"]),
            (t["gross"] - t["hedge"] - t["cost"]) - (b["gross"] - b["hedge"]) - b["cost"] - fee,
        ),
        "LV10": (lvl["gross"] - lvl["hedge"], lvl["gross"] - lvl["hedge"] - lvl["cost"]),
    }
    series["H10−LV10"] = (series["H10"][0] - series["LV10"][0], series["H10"][1] - series["LV10"][1])
    mk = masks(t, market)
    mkt = t["mkt"].to_numpy()
    res: dict = {"market": market, "blocks": log, "pool": info.get("pool"), "portfolios": {}}
    for name, (g, n) in series.items():
        parts = {}
        for part, m in mk.items():
            if part == "short_ok" and name not in ("LS",):
                continue
            parts[part] = {"gross": mstats(g.to_numpy()[m], mkt[m]), "net": mstats(n.to_numpy()[m], mkt[m])}
        res["portfolios"][name] = parts
    steps = label_steps(ds, sc)
    full = pd.DatetimeIndex(ds.calendar)
    sh = same_close(sc, full)
    for tag, frame in (("same_close", sh), ("next_close", sc)):
        top = hedge_path(market, frame, w, "D10", uni, bc)
        bot = hedge_path(market, frame, w, "D1", uni, bc)
        for part in ("outer", "holdout"):
            m = (top["part"] == part).to_numpy()
            steps[f"{tag}|{part}"] = {
                "D10": float((top["r"] - top["hedge"]).to_numpy()[m].mean()),
                "D1": float((bot["r"] - bot["hedge"]).to_numpy()[m].mean()),
            }
    res["gap_steps"] = steps
    res["turnover_yr"] = {k: float(v["turnover"].mean() * 12) for k, v in legs.items()}
    res["mean_beta"] = {k: float(v["beta"].mean()) for k, v in legs.items()}
    res["period"] = [str(sc["date"].min().date()), str(sc["date"].max().date())]
    return res


def write_md(r: dict) -> str:
    p = pb.pct
    mb, tu = r["mean_beta"], r["turnover_yr"]
    lines = [
        f"# {r['market']} 베타 헤지·롱숏·저변동 대조 (docs/MODELING.md 78)",
        "",
        f"- 기간 {r['period'][0]} ~ {r['period'][1]} · 점수 = 77과 같은 걷기 방식 합산 점수 · "
        "헤지 = 종목별 사전 베타 × 자기 벤치마크 매도(선물 비용 0 가정)",
        f"- 평균 사전 베타: 상위 10% {mb['top']:.2f} · 하위 10% {mb['bottom']:.2f} · 저변동 10% {mb['lowvol']:.2f} · "
        f"연 회전율 상위 {tu['top']:.1f}배 · 하위 {tu['bottom']:.1f}배 · 저변동 {tu['lowvol']:.1f}배",
        f"- LS 비용 후에는 공매도 대여료 연 {BORROW[r['market']]:.1%} 포함 · LS는 매수 1 + 매도 1(총 2배) 기준",
        "",
        "| 포트폴리오 | 구간 | 연수익(비용 전) | 연수익(비용 후) | 변동성 | 샤프(비용 후) | 월평균 t(비용 후) |"
        " 최대낙폭(비용 후) | 플러스 달 | 사후 시장 베타 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    names = {
        "H10": "H10 상위 10% + 헤지",
        "LS": "LS 상위 − 하위(베타 중립)",
        "LV10": "LV10 저변동 10% + 헤지",
        "H10−LV10": "H10 − LV10",
    }
    for name, parts in r["portfolios"].items():
        for part, x in parts.items():
            g, n = x["gross"], x["net"]
            if n.get("months", 0) < 3:
                continue
            lines.append(
                f"| {names[name]} | {PART_KO[part]}({n['months']}개월) | {p(g['ann'])} | {p(n['ann'])} | "
                f"{n['vol'] * 100:.1f}% | {n['sharpe']:.2f} | {n['t']:+.2f} | {p(n['mdd'])} | {n['hit']:.0%} | "
                f"{n['beta_ex_post']:+.2f} |"
            )
    names = {
        "y_e": "① 라벨 그대로(로그·배당 포함·점수 날 종가 시작·20거래일)",
        "simple": "② 같은 창, 단순 수익률",
        "price": "③ 같은 창, 단순 수익률·배당 제외",
        "same_close": "④ 헤지 포트폴리오, 점수 날 종가 체결(실행 불가, 진단)",
        "next_close": "⑤ 헤지 포트폴리오, 다음 거래일 종가 체결(실제 가능)",
    }
    gs = r.get("gap_steps") or {}
    if gs:
        lines += [
            "",
            "## 라벨과 포트폴리오의 차이 분해 (78.1, 기간당 평균)",
            "",
            "| 단계 | 표본 외 D10 | 표본 외 D1 | 표본 외 D10 − D1 | 홀드아웃 D10 − D1 |",
            "|---|---|---|---|---|",
        ]
        for k, label in names.items():
            o, h = gs.get(f"{k}|outer"), gs.get(f"{k}|holdout")
            if not o:
                continue
            lines.append(
                f"| {label} | {p(o['D10'], 2)} | {p(o['D1'], 2)} | {p(o['D10'] - o['D1'], 2)} | "
                f"{p(h['D10'] - h['D1'], 2) if h else '–'} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    m = ap.parse_args().market
    OUT.mkdir(parents=True, exist_ok=True)
    r = run(m)
    text = write_md(r)
    (OUT / f"hedged_{m}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    (OUT / f"hedged_{m}.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
