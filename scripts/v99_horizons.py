"""KR: models trained for 1 / 3 / 6 / 12-month horizons, compared on cost-aware D20 books (docs/MODELING.md 99,
pre-declared 2026-11-08). Research only: src/quantcio and the served models are not changed.

    .venv/bin/python scripts/v99_horizons.py build --h 1      # dataset for one horizon (cached vintages)
    .venv/bin/python scripts/v99_horizons.py study --h 1      # run the protocol for one horizon (checkpointed)
    .venv/bin/python scripts/v99_horizons.py analyse          # re-read the four cached runs (no fit) and compare

Horizons (months -> sessions): 1 -> 20, 3 -> 60, 6 -> 120, 12 -> 240. Candidates: rf_bo, rf, mom_12_1 (defaults).
Books (98 rules): month-end D20 cohorts held h in {1, 3, 6, 12} months (overlapping), monthly returns from the
20-session labels, missing -> universe mean, one-way cost x traded share (50 bp, 100 bp sensitivity); buffer (in at
D20, out below the top 20%). Common window: month-ends where every arm has OOS predictions, from its 12th month.
Rule (99): a long-horizon rf_bo helps if, at its own holding, its net monthly return minus the 1-month rf_bo held the
same way has t >= 2 and a non-negative sum over the last 24 window months.
Writes reports/modeling/KR_v99_horizons.{json,md}; state in reports/modeling/KR_v99_state.json.
"""

from __future__ import annotations

import os

for _k in ("KRX_ID", "KRX_PW"):  # before anything imports pykrx
    os.environ.pop(_k, None)

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v86_full_es as v86  # noqa: E402

from quantcio.config import load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.cli import guard_xsh  # noqa: E402
from quantcio.modeling.dataset import build_dataset  # noqa: E402
from quantcio.models import Market  # noqa: E402

M = "KR"
K = 20
V99_VERSION = 1
H = {1: 20, 3: 60, 6: 120, 12: 240}
NAMES = ("mom_12_1", "rf", "rf_bo")
HOLD = (1, 3, 6, 12)
COSTS = (0.005, 0.010)
OUT = v86.OUT
STATE = OUT / "KR_v99_state.json"
say = v86.say


def settings(h: int):
    return load_settings() if H[h] == 20 else load_settings(horizon_days=H[h])


def candidates() -> list[cmod.Candidate]:
    return [c for c in cmod.default_candidates() if c.name in NAMES]


def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def build(h: int) -> None:
    s = settings(h)
    cache = DiskCache(s.cache_path)
    ds = build_dataset(Market.KR, s, cache, use_cache=True, progress=say)
    dig = (ds.coverage.get("vintage") or {}).get("digest") or {}
    say(
        f"h={h}m ({H[h]} sessions): dataset {ds.snapshot} as_of {ds.as_of.date()} rows {len(ds.frame):,} "
        f"labelled month-ends {int((ds.frame['labelled'] & ds.frame['is_me']).sum()):,}; digest {dig.get('status')}"
    )
    # the digest identity includes the horizon: a new horizon's first build records it ("first"); its inputs are the
    # same frozen vintages as the 1-month build (which must "match"), checked below on the features
    if dig.get("status") != "match" and not (h != 1 and dig.get("status") == "first"):
        raise SystemExit(f"frozen-window digest is not a match: {dig}")
    st = state()
    if h != 1:
        one = cache.get("modeling_dataset", st["1"]["snapshot"], None)
        if one is None or list(one.xs_cols) != list(ds.xs_cols):
            raise SystemExit(f"h={h}: the 1-month dataset is missing or has different inputs")
        k = ["date", "entity"]
        a = one.frame[[*k, *one.xs_cols]].merge(ds.frame[[*k, *ds.xs_cols]], on=k, suffixes=("", "_h"))
        x = a[list(one.xs_cols)].to_numpy(float)
        y = a[[f"{c}_h" for c in one.xs_cols]].to_numpy(float)
        same = np.isclose(x, y, rtol=1e-6, atol=1e-9, equal_nan=True).mean()
        say(f"h={h}m: features vs the 1-month dataset on {len(a):,} shared rows: {100 * same:.4f}% equal")
        if len(a) < 0.95 * len(one.frame) or same < 0.999:
            raise SystemExit(f"h={h}: features differ from the 1-month dataset ({same:.6f}, {len(a)} rows)")
        dig = {**dig, "features_equal_to_1m": float(same)}
        del one, a, x, y
        gc.collect()
    st[str(h)] = {
        "snapshot": ds.snapshot,
        "as_of": str(ds.as_of.date()),
        "sessions": H[h],
        "digest": dig,
        "version": V99_VERSION,
    }
    STATE.write_text(json.dumps(st, indent=1, default=str))


def load(h: int, s, cache):
    st = state().get(str(h))
    if st is None:
        raise SystemExit(f"h={h}: run build first")
    ds = cache.get("modeling_dataset", st["snapshot"], None)
    if ds is None:
        raise SystemExit(f"h={h}: dataset {st['snapshot']} is not cached")
    if int(ds.horizon) != H[h]:
        raise SystemExit(f"h={h}: dataset horizon {ds.horizon} != {H[h]}")
    return guard_xsh(ds, s, say)


def run(h: int, no_fit: bool):
    s = settings(h)
    cache = DiskCache(s.cache_path)
    ds = load(h, s, cache)

    def guard(msg: str) -> None:
        if not no_fit:
            say(f"[{h}m] {msg}")
        elif "train <=" in msg:
            raise SystemExit(f"analyse: h={h} context is not cached ({msg})")

    snap0 = ds.snapshot
    ds.snapshot = f"{snap0}|v99|{V99_VERSION}"
    try:
        res = sel.run_selection(ds, s, cache, candidates=candidates(), threads=8, use_cache=True, progress=guard)
    finally:
        ds.snapshot = snap0
    return ds, res


def study(h: int) -> None:
    _, res = run(h, no_fit=False)
    say(f"h={h}m done: protocol selected {res.selected}")


# ------------------------------------------------------------------------------------------------ books


def stats(r: pd.Series, u: pd.Series) -> dict:
    x = r.fillna(0.0).to_numpy()
    nav = np.cumprod(1 + x)
    peak = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    e = (r - u).fillna(0.0)
    return {
        "ann": float(nav[-1] ** (12 / len(x)) - 1),
        "mdd": float((nav / peak - 1).min()),
        "t_vs_u": float(e.mean() / e.std(ddof=1) * np.sqrt(len(e))),
        "months": int(len(x)),
    }


def paired(a: pd.Series, b: pd.Series, last: int = 24) -> dict:
    e = (a - b).fillna(0.0)
    return {
        "mean_month": float(e.mean()),
        "t": float(e.mean() / e.std(ddof=1) * np.sqrt(len(e))) if e.std(ddof=1) > 0 else float("nan"),
        "last24_sum": float(e.iloc[-last:].sum()),
    }


class Books:
    """Cohort books on a fixed month-end grid with monthly returns R[(date, entity)] and universe means U[date]."""

    def __init__(self, dates, ret: pd.Series, uni: pd.Series):
        self.dates, self.ret, self.uni = list(dates), ret, uni
        self.missing = 0
        self.total = 0

    def _r(self, m, names) -> float:
        v = []
        for e in names:
            self.total += 1
            x = self.ret.get((m, e), np.nan)
            if not np.isfinite(x):
                self.missing += 1
                x = self.uni[m]
            v.append(x)
        return float(np.mean(v)) if v else np.nan

    def hold(self, books: dict, h: int) -> tuple[pd.Series, pd.Series]:
        d = self.dates
        r, tc = [], []
        for i, m in enumerate(d):
            live = [d[j] for j in range(max(0, i - h + 1), i + 1) if d[j] in books]
            r.append(np.mean([self._r(m, books[c]) for c in live]) if live else np.nan)
            new = books.get(m, set())
            old = books.get(d[i - h], set()) if i - h >= 0 else set()
            tc.append((1 / h) * (len(new - old) / len(new)) if new else 0.0)
        return pd.Series(r, index=d), pd.Series(tc, index=d)

    def buffer(self, books: dict, top20: dict) -> tuple[pd.Series, pd.Series, list[int]]:
        held, r, tc, lens, opened = set(), [], [], [], {}
        for i, m in enumerate(self.dates):
            keep = held & top20.get(m, set())
            for e in held - keep:
                lens.append(i - opened.pop(e))
            buy = books.get(m, set()) - keep
            for e in buy:
                opened[e] = i
            held = keep | buy
            tc.append(len(buy) / len(held) if held else 0.0)
            r.append(self._r(m, held) if held else np.nan)
        return pd.Series(r, index=self.dates), pd.Series(tc, index=self.dates), lens


def ventile_sets(frame: pd.DataFrame, name: str) -> tuple[dict, dict]:
    d = frame.dropna(subset=[name])
    d = d[d.groupby("date")[name].transform("nunique").to_numpy() > 1]
    g = d.groupby("date")
    u = (g[name].rank(method="average") - 0.5) / g[name].transform("size")
    b = np.clip(np.floor(u * K), 0, K - 1).astype(int) + 1
    top = {t: set(x) for t, x in d[b == K].groupby("date")["entity"]}
    top20 = {t: set(x) for t, x in d[b >= K - 3].groupby("date")["entity"]}
    return top, top20


def analyse() -> None:
    preds, ret, uni, latest, info = {}, None, None, {}, {}
    for h in H:
        ds, res = run(h, no_fit=True)
        f = pd.concat([res.oos, res.holdout_preds], ignore_index=True)
        f = f[f.groupby("date")["y_e"].transform("size") >= mt.MIN_NAMES]
        preds[h] = f[["date", "entity", *NAMES]].copy()
        info[h] = {
            "as_of": str(ds.as_of.date()),
            "oos": [str(f["date"].min().date()), str(f["date"].max().date())],
            "outer_last": str(res.oos["date"].max().date()),
            "holdout": [str(res.holdout_preds["date"].min().date()), str(res.holdout_preds["date"].max().date())],
            "protocol_selected": res.selected,
            "ic_oos": {n: float(mt.per_date_ic(f, n).mean()) for n in NAMES},
        }
        fin = res.final.predictions if res.final is not None else None
        if fin is not None:
            names = ds.frame.drop_duplicates("entity").set_index("entity")
            col = "name" if "name" in names.columns else ("ticker" if "ticker" in names.columns else None)
            z = fin.dropna(subset=["rf_bo"]).sort_values("rf_bo", ascending=False)
            n_top = int(np.ceil(len(z) / K))
            latest[h] = [
                {"entity": str(e), "label": str(names[col].get(e, e)) if col else str(e)}
                for e in z["entity"].head(n_top)
            ]
        if h == 1:
            lab = f.dropna(subset=["r"])
            ret = pd.Series(np.expm1(lab["r"].to_numpy(float)), index=pd.MultiIndex.from_frame(lab[["date", "entity"]]))
            ret = ret[~ret.index.duplicated()]
            uni = lab.assign(R=np.expm1(lab["r"].to_numpy(float))).groupby("date")["R"].mean()
        say(f"h={h}m: OOS {info[h]['oos']} ({f['date'].nunique()} month-ends), as_of {info[h]['as_of']}")
        del ds, res, f
        gc.collect()
    common = sorted(set.intersection(*(set(p["date"].unique()) for p in preds.values())) & set(uni.index))
    if len(common) < 36:
        raise SystemExit(f"common window too short: {len(common)} month-ends")
    start = 11
    w = common[start:]
    say(f"common formation grid {common[0].date()}..{common[-1].date()} ({len(common)}), evaluated {w[0].date()}..")
    bk = Books(common, ret, uni)
    u = uni.reindex(common)
    cells, series = {}, {}
    for h in H:
        for n in NAMES:
            top, top20 = ventile_sets(preds[h][preds[h]["date"].isin(common)], n)
            stay = [
                len(top[common[i]] & top[common[i - 1]]) / len(top[common[i]])
                for i in range(1, len(common))
                if common[i] in top and common[i - 1] in top
            ]
            for hold in HOLD:
                r, tc = bk.hold(top, hold)
                key = f"{n}@{h}m/hold{hold}"
                series[key] = {c: (r - c * tc) for c in COSTS}
                series[key]["gross"] = r
                cells[key] = {
                    "model": n,
                    "horizon_m": h,
                    "hold_m": hold,
                    "turnover": float(tc.loc[w].mean()),
                    "gross": stats(r.loc[w], u.loc[w]),
                    **{f"net{int(c * 1e4)}": stats((r - c * tc).loc[w], u.loc[w]) for c in COSTS},
                    "d20_stays_next_month": float(np.mean(stay)) if stay else None,
                }
            r, tc, lens = bk.buffer(top, top20)
            key = f"{n}@{h}m/buffer"
            series[key] = {c: (r - c * tc) for c in COSTS}
            series[key]["gross"] = r
            cells[key] = {
                "model": n,
                "horizon_m": h,
                "hold_m": "buffer",
                "turnover": float(tc.loc[w].mean()),
                "gross": stats(r.loc[w], u.loc[w]),
                **{f"net{int(c * 1e4)}": stats((r - c * tc).loc[w], u.loc[w]) for c in COSTS},
                "median_hold_months": float(np.median(lens)) if lens else None,
            }
    rule = {}
    for h in (3, 6, 12):
        a = series[f"rf_bo@{h}m/hold{h}"][COSTS[0]].loc[w]
        b = series[f"rf_bo@1m/hold{h}"][COSTS[0]].loc[w]
        p = paired(a, b)
        rule[f"{h}m"] = {**p, "helps": bool(p["t"] >= 2 and p["last24_sum"] >= 0)}
        for c in COSTS:
            rule[f"{h}m"][f"vs_monthly_1m_net{int(c * 1e4)}"] = paired(
                series[f"rf_bo@{h}m/hold{h}"][c].loc[w], series["rf_bo@1m/hold1"][c].loc[w]
            )
    o = {
        "market": M,
        "version": V99_VERSION,
        "horizons": {str(k): v for k, v in H.items()},
        "info": {str(k): v for k, v in info.items()},
        "common": [str(common[0].date()), str(common[-1].date())],
        "evaluated": [str(w[0].date()), str(w[-1].date())],
        "months": len(w),
        "universe_ann": stats(u.loc[w], u.loc[w])["ann"],
        "missing_share": bk.missing / bk.total if bk.total else 0.0,
        "cells": cells,
        "rule": rule,
        "latest_rf_bo_D20": {str(k): v for k, v in latest.items()},
    }
    (OUT / "KR_v99_horizons.json").write_text(
        json.dumps(o, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    write_md(o)
    say(f"wrote {OUT / 'KR_v99_horizons.json'}")


def write_md(o: dict) -> None:
    def pc(x: float) -> str:
        return f"{100 * x:+.1f}%"

    lines = [
        f"# KR 99: 예측 기간별 모델 — 거래비용 비교 (평가 {o['evaluated'][0]}~{o['evaluated'][1]}, {o['months']}개월)",
        "",
        f"- 전체(동일가중) 연 {pc(o['universe_ann'])}; 비용 편도 50bp(민감도 100bp); "
        f"행 없는 묶음 종목 비율 {100 * o['missing_share']:.1f}%",
        "",
        "| 모델 | 학습 기간 | 보유 | 연(비용 전) | 연(50bp) | 연(100bp) | 전체 대비 t(50bp) | 최대 낙폭 | 월 교체 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in o["cells"].values():
        hold = "완충" if c["hold_m"] == "buffer" else f"{c['hold_m']}개월"
        lines.append(
            f"| `{c['model']}` | {c['horizon_m']}개월 | {hold} | {pc(c['gross']['ann'])} | {pc(c['net50']['ann'])} | "
            f"{pc(c['net100']['ann'])} | {c['net50']['t_vs_u']:+.2f} | {100 * c['gross']['mdd']:.0f}% | "
            f"{100 * c['turnover']:.0f}% |"
        )
    lines += ["", "## 판정 (99, 사전 고정)", ""]
    for k, v in o["rule"].items():
        lines.append(
            f"- {k} `rf_bo` vs 1개월 `rf_bo` 같은 보유: 월 {100 * v['mean_month']:+.2f}%p, t {v['t']:+.2f}, "
            f"마지막 24개월 합 "
            f"{100 * v['last24_sum']:+.1f}%p → {'도움' if v['helps'] else '도움 없음'}"
        )
    (OUT / "KR_v99_horizons.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "study", "analyse"))
    ap.add_argument("--h", type=int, choices=sorted(H))
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.h)
    elif a.cmd == "study":
        study(a.h)
    else:
        analyse()


if __name__ == "__main__":
    main()
