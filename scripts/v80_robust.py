"""Robustness of the KR top-400 result of stage B (docs/MODELING.md 80.6, declared 2026-10-25). Research only.

    .venv/bin/python scripts/v80_robust.py

Reloads the `wide` variant run (checkpoints), takes its selected model and mom_12_1, and runs the top-quintile
equal-weight book through portfolio_backtest.simulate (47 per-stock cost model at the middle book size and at
twice the cost) on real price paths, with and without the per-name beta hedge of hedged_study. Reports outer /
holdout annual net excess vs the universe equal weight, year by year, the share of held names ranked 201-400 and
position size / 20-day traded value. Writes reports/modeling/KR_v80_robust.{md,json}.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hedged_study as hs  # noqa: E402  (imports decile_study / portfolio_backtest)
import portfolio_backtest as pb  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.data.pit import load_pit_universe  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.cli import guard_xsh  # noqa: E402
from quantcio.modeling.dataset import build_dataset, universe_params  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ann_stats(r: np.ndarray) -> dict:
    n = len(r)
    if n < 3:
        return {"months": n}
    sd = r.std(ddof=1)
    return {
        "months": n,
        "ann": float(np.prod(1 + r) ** (12 / n) - 1),
        "mean_ann": float(r.mean() * 12),
        "t": float(r.mean() / (sd / np.sqrt(n))) if sd > 0 else None,
        "hit": float((r > 0).mean()),
    }


def main() -> None:
    s = load_settings().model_copy(deep=True)
    s.backtest.kr_universe_n = 400
    cache = DiskCache(s.cache_path)
    mk = Market.KR
    ds = guard_xsh(build_dataset(mk, s, cache, use_cache=True, progress=say), s, say)
    ds.cost_bps = 50.0
    dsv = copy.copy(ds)
    dsv.snapshot = f"{dsv.snapshot}|v80b_wide"
    res = sel.run_selection(
        dsv, s, cache, candidates=cmod.default_candidates(), threads=8, use_cache=True, progress=say
    )
    chosen = res.selected
    say(f"wide selection: {chosen}")
    n, start = universe_params(mk, s)
    uni = load_pit_universe(mk, s, cache, n, start)
    me = ds.frame[ds.frame["is_me"]].drop_duplicates(["date", "entity"])
    meta = me[["date", "entity", "log_mcap", "beta_s"]]
    out: dict = {"selected": chosen, "models": {}}
    for name in (chosen, sel.BASELINE):
        parts = [res.oos[["date", "entity", name]].assign(part="outer")]
        parts.append(res.holdout_preds[["date", "entity", name]].assign(part="holdout"))
        sc = pd.concat(parts, ignore_index=True).rename(columns={name: "score"}).merge(meta, on=["date", "entity"])
        sc["mcap"] = np.exp(sc["log_mcap"].astype(float))
        sc["mcap_rank"] = sc.groupby("date")["mcap"].rank(ascending=False)
        cal = pd.DatetimeIndex(ds.calendar)
        cal = cal[cal >= pd.Timestamp(sc["date"].min())]
        w = pb.wide(uni.prices, sorted(sc["entity"].unique()), cal)
        bc = hs.bench_closes(uni, cal)
        mid = pb.cm.AUM["KR"][1]
        _, bper = pb.simulate("KR", sc, w, "B1", mid, 1.0, False)
        hp = hs.hedge_path("KR", sc, w, "P1", uni, bc)
        res_m: dict = {}
        for tag, scale, cost in (("gross", 1.0, False), ("net", 1.0, True), ("net_x2", 2.0, True)):
            _, per = pb.simulate("KR", sc, w, "P1", mid, scale, cost)
            ex = per["ret"].to_numpy() - bper["ret"].to_numpy()  # vs the universe equal weight
            hedged = per["ret"].to_numpy() - hp["hedge"].to_numpy()  # vs beta x own benchmark
            row = {}
            for part in ("outer", "holdout"):
                m = (per["part"] == part).to_numpy()
                row[part] = {"vs_B1": ann_stats(ex[m]), "hedged": ann_stats(hedged[m])}
            yrs = pd.Series(hedged, index=pd.to_datetime(per["date"])).groupby(lambda d: d.year)
            row["hedged_by_year"] = {int(y): float(np.prod(1 + g.to_numpy()) - 1) for y, g in yrs}
            row["turnover_yr"] = float(per["turnover"].mean() * 12)
            row["cost_yr"] = float(per["cost"].mean() * 12)
            res_m[tag] = row
            say(
                f"{name} {tag}: outer hedged {row['outer']['hedged'].get('ann', float('nan')):+.1%}/yr, "
                f"holdout {row['holdout']['hedged'].get('ann', float('nan')):+.1%}"
            )
        # composition and liquidity of the top-quintile book (the targets simulate uses)
        shares, liq = [], []
        dates = sorted(sc["date"].unique())
        for d in dates:
            g = sc[sc["date"] == d]
            book = pb.targets(g, "P1")
            if not len(book):
                continue
            ranks = g.set_index("entity").loc[book.index, "mcap_rank"]
            shares.append(float((ranks > 200).mean()))
            ex_d = cal[min(cal.searchsorted(pd.Timestamp(d), "right"), len(cal) - 1)]
            adv = w["adv"].loc[ex_d].reindex(book.index).astype(float)
            pos = mid * book
            liq += list((pos / adv).replace([np.inf, -np.inf], np.nan).dropna().to_numpy())
        res_m["share_rank_201_400"] = float(np.mean(shares)) if shares else None
        liq_arr = np.array(liq)
        res_m["position_over_adv"] = {
            "median": float(np.median(liq_arr)) if len(liq_arr) else None,
            "p95": float(np.quantile(liq_arr, 0.95)) if len(liq_arr) else None,
        }
        out["models"][name] = res_m

    def a(x):
        return "–" if x is None else f"{x:+.1%}"

    lines = [
        "# 한국 상위 400 결과의 견고성 (docs/MODELING.md 80.6)",
        "",
        f"- 모델 `{chosen}`(wide 변형 v4 선정)과 같은 유니버스의 `mom_12_1` · 상위 20% 같은 비중 · "
        "다음 거래일 종가 체결 · 47절 비용 모형(10억 원)",
        "- 헤지 = 종목별 사전 베타 × 자기 지수(선물 비용 0 가정) · B1 = 점수 있는 종목 같은 비중",
        "",
        "| 모델 | 비용 | 외부 헤지 순초과(연) | t | 외부 B1 대비(연) | 홀드아웃 헤지(연) | 홀드아웃 B1 대비(연) |"
        " 연 회전율 | 연 비용 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in out["models"].items():
        for tag, ko in (("gross", "비용 전"), ("net", "비용 후"), ("net_x2", "비용 2배")):
            x = r[tag]
            o, h = x["outer"], x["holdout"]
            t = o["hedged"].get("t")
            lines.append(
                f"| {name} | {ko} | {a(o['hedged'].get('ann'))} | {'–' if t is None else f'{t:+.2f}'} | "
                f"{a(o['vs_B1'].get('ann'))} | {a(h['hedged'].get('ann'))} | {a(h['vs_B1'].get('ann'))} | "
                f"{x['turnover_yr']:.1f}배 | {x['cost_yr']:.1%} |"
            )
    lines += ["", "## 연도별 헤지 순초과(비용 후)", "", "| 연도 | " + " | ".join(out["models"]) + " |"]
    lines.append("|---|" + "---|" * len(out["models"]))
    years = sorted({y for r in out["models"].values() for y in r["net"]["hedged_by_year"]})
    for y in years:
        cells = [a(r["net"]["hedged_by_year"].get(y)) for r in out["models"].values()]
        lines.append(f"| {y} | " + " | ".join(cells) + " |")
    lines += ["", "## 구성·유동성", ""]
    for name, r in out["models"].items():
        lq = r["position_over_adv"]
        lines.append(
            f"- `{name}`: 보유 종목 중 201~400위 비중 {r['share_rank_201_400']:.0%} · "
            f"종목당 보유 금액 ÷ 20일 평균 거래대금 중앙값 {lq['median']:.2%}, 상위 5% {lq['p95']:.2%}"
        )
    text = "\n".join(lines) + "\n"
    (OUT / "KR_v80_robust.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    (OUT / "KR_v80_robust.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    sys.exit(main())
