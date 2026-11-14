"""KR 101 (docs/MODELING.md 101, pre-declared 2026-11-09 22:15): (1) 3- and 6-month models trained from 2002, checked
on the fresh 2005-2015 window; (2) rolling training windows (only the last N years of rows) vs the expanding window.
Research only: src/quantcio and the served models are not changed.

    .venv/bin/python scripts/v101_kr_window.py build --h 3          # 2002-start 60-session dataset (100's checks)
    .venv/bin/python scripts/v101_kr_window.py study --arm E        # E/F: 3/6-month from 2002 (100's protocol)
    .venv/bin/python scripts/v101_kr_window.py study --arm R5       # R3/R5/R8 on 100 A's data, S3/S5 on 99's 1-month
    .venv/bin/python scripts/v101_kr_window.py analyse --part ef    # rules 5-6 (+ the 99 recent-window cells)
    .venv/bin/python scripts/v101_kr_window.py analyse --part roll  # rules 7-8

Rolling (101.2): every fit (outer, inner tuning / BO, holdout, final) sees only labelled rows with
train_end - N years < date <= train_end; wrapped around selection._Data.train in this process only.
Writes reports/modeling/KR_v101_{ef,roll}.{json,md}; rolling span checks in reports/modeling/KR_v101_state.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v99_horizons as v99  # noqa: E402  (pops KRX_ID / KRX_PW before pykrx can be imported)
import v100_kr_long as v100  # noqa: E402

from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

V101_VERSION = 1
v100.SESS.update({3: 60, 6: 120})
v100.ARMS.update({"E": (3, None), "F": (6, None)})
OWN = {"A": 1, "E": 3, "F": 6, "C": 12}  # model horizon in months = its own holding
ROLL = {"R3": ("A", 3), "R5": ("A", 5), "R8": ("A", 8), "S3": ("S", 3), "S5": ("S", 5)}
T_ROLL = 2.4  # rule 7: three windows, Bonferroni
FRESH = ("2005-01-01", "2015-12-31")
HALVES = (("2005-01-01", "2015-12-31"), ("2016-01-01", "2026-12-31"))
RECENT = ("2019-01-01", "2026-12-31")  # 100 arm B's OOS
NAMES = v99.NAMES
COSTS = v99.COSTS
OUT = v99.OUT
STATE = OUT / "KR_v101_state.json"
say = v99.say
SPANS: list[tuple[pd.Timestamp, pd.Timestamp, int, int]] = []  # (train_end, first kept date, rows in, rows kept)


def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


@contextmanager
def rolling(years: int):
    """101.2: keep only the rows dated after train_end - `years` in every training call of this process."""
    orig = sel._Data.train

    def train(self, train_end, rows):
        tr = orig(self, train_end, rows)
        lo = np.datetime64(pd.Timestamp(train_end) - pd.DateOffset(years=years), "ns")
        kept = tr[tr["date"].to_numpy(dtype="datetime64[ns]") > lo]
        if len(kept):
            SPANS.append((pd.Timestamp(train_end), pd.Timestamp(kept["date"].min()), len(tr), len(kept)))
        return kept

    sel._Data.train = train
    try:
        yield
    finally:
        sel._Data.train = orig


def roll_prepared(arm: str):
    base, years = ROLL[arm]
    if base == "A":
        s = v100.settings(1)
        cache = DiskCache(s.cache_path)
        ds, tag = v100.prepared("A", s, cache)
    else:
        s = v99.settings(1)
        cache = DiskCache(s.cache_path)
        ds = v99.load(1, s, cache)
        tag = f"{ds.snapshot}|v99|{v99.V99_VERSION}"
    return ds, s, cache, f"{tag}|v101|{V101_VERSION}|roll{years}y"


def run_roll(arm: str, no_fit: bool):
    ds, s, cache, tag = roll_prepared(arm)

    def guard(msg: str) -> None:
        if not no_fit:
            say(f"[{arm}] {msg}")
        elif "train <=" in msg:
            raise SystemExit(f"analyse: arm {arm} context is not cached ({msg})")

    snap0 = ds.snapshot
    ds.snapshot = tag
    try:
        with rolling(ROLL[arm][1]):
            cands = v99.candidates()
            res = sel.run_selection(ds, s, cache, candidates=cands, threads=8, use_cache=True, progress=guard)
    finally:
        ds.snapshot = snap0
    return ds, res


def study(arm: str) -> None:
    if arm in ("E", "F"):
        v100.study(arm)
        return
    SPANS.clear()
    _, res = run_roll(arm, no_fit=False)
    years = ROLL[arm][1]
    bad = [(str(te.date()), str(lo.date())) for te, lo, _, _ in SPANS if lo <= te - pd.DateOffset(years=years)]
    span = max(((te - lo).days for te, lo, _, _ in SPANS), default=0)
    cut = sum(n1 < n0 for _, _, n0, n1 in SPANS)
    say(f"arm {arm}: {len(SPANS)} training calls, {cut} cut, longest span {span} days (limit {years} years)")
    if bad:
        raise SystemExit(f"arm {arm}: training rows older than {years} years: {bad[:5]}")
    st = state()
    st[arm] = {
        "calls": len(SPANS),
        "cut_calls": int(cut),
        "longest_span_days": int(span),
        "selected": res.selected,
        "version": V101_VERSION,
    }
    STATE.write_text(json.dumps(st, indent=1))
    say(f"arm {arm} done: protocol selected {res.selected}")


# ------------------------------------------------------------------------------------------------ analysis


def win(s: pd.Series, a: str, b: str) -> pd.Series:
    return s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]


def ann(x: pd.Series) -> float:
    x = x.fillna(0.0)
    return float((1 + x).prod() ** (12 / len(x)) - 1) if len(x) else float("nan")


def capm(y: pd.Series, m: pd.Series) -> tuple[float, float, float]:
    d = pd.concat([y, m], axis=1).dropna()
    yy, xx = d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)
    X = np.column_stack([np.ones(len(xx)), xx])
    b, *_ = np.linalg.lstsq(X, yy, rcond=None)
    e = yy - X @ b
    se = np.sqrt(np.diag(e @ e / (len(yy) - 2) * np.linalg.inv(X.T @ X)))
    return float(12 * b[0]), float(b[1]), float(b[0] / se[0])


def cell(r: pd.Series, tc: pd.Series, u: pd.Series) -> dict:
    out = {
        "span": [str(r.index[0].date()), str(r.index[-1].date())],
        "months": int(len(r)),
        "turnover": float(tc.mean()),
        "universe_ann": ann(u),
    }
    for c in (0.0, *COSTS):
        net = r - c * tc
        a, b, t = capm(net, u)
        out[f"net{int(round(c * 1e4))}"] = {**v99.stats(net, u), "alpha": a, "beta": b, "t_alpha": t}
    return out


def paired_t(x: pd.Series) -> dict:
    x = x.fillna(0.0)
    sd = x.std(ddof=1)
    return {"mean": float(x.mean()), "t": float(x.mean() / sd * np.sqrt(len(x))) if sd > 0 else float("nan")}


class Pool:
    """OOS + holdout predictions of several runs; monthly returns from the base run's 20-session labels."""

    def __init__(self) -> None:
        self.preds: dict[str, pd.DataFrame] = {}
        self.ret: pd.Series | None = None
        self.uni: pd.Series | None = None

    def add(self, key: str, res, base: bool = False) -> None:
        f = pd.concat([res.oos, res.holdout_preds], ignore_index=True)
        f = f[f.groupby("date")["y_e"].transform("size") >= mt.MIN_NAMES]
        self.preds[key] = f[["date", "entity", *NAMES]].copy()
        if base:
            lab = f.dropna(subset=["r"])
            R = np.expm1(lab["r"].to_numpy(float))
            ret = pd.Series(R, index=pd.MultiIndex.from_frame(lab[["date", "entity"]]))
            self.ret = ret[~ret.index.duplicated()]
            self.uni = lab.assign(R=R).groupby("date")["R"].mean()
        p = self.preds[key]
        say(f"{key}: OOS {p['date'].min().date()}..{p['date'].max().date()} ({p['date'].nunique()} month-ends)")

    def grid(self, *keys: str) -> list:
        s = set(self.uni.index)
        for k in keys:
            s &= set(self.preds[k]["date"].unique())
        return sorted(s)

    def books(self, key: str, n: str, grid: list, hold: int, cost: float | None = None):
        bk = v99.Books(grid, self.ret, self.uni)
        top, top20 = v99.ventile_sets(self.preds[key][self.preds[key]["date"].isin(grid)], n)
        r, tc = bk.hold(top, hold) if hold else bk.buffer(top, top20)[:2]
        return (r, tc) if cost is None else r - cost * tc

    def same_preds(self, a: str, b: str, months: set, n: str = "rf_bo") -> float:
        k = ["date", "entity"]
        x = self.preds[a][k + [n]].merge(self.preds[b][k + [n]], on=k, suffixes=("_a", "_b"))
        x = x[x["date"].isin(months)]
        return float(np.isclose(x[f"{n}_a"], x[f"{n}_b"], rtol=1e-9, atol=1e-12).mean()) if len(x) else float("nan")


def hold_key(hold: int) -> str:
    return "buffer" if hold == 0 else f"hold{hold}"


def analyse_ef() -> None:
    pool = Pool()
    for arm in ("A", "C", "E", "F"):
        _, res = v100.run(arm, no_fit=True)
        pool.add(arm, res, base=arm == "A")
        del res
        gc.collect()
    c50 = COSTS[0]
    out = {"version": V101_VERSION, "fresh": list(FRESH), "rule": {}, "cells": {}}
    for arm in ("E", "F"):  # rule 5
        g = pool.grid(arm)
        net = win(pool.books(arm, "rf_bo", g, OWN[arm], c50).loc[g[11:]], *FRESH)
        st = v99.stats(net, pool.uni.reindex(net.index))
        out["rule"][f"5_fresh_{arm}_rf_bo_hold{OWN[arm]}"] = {**st, "confirmed": bool(st["t_vs_u"] >= 2)}
    for arm in ("E", "F", "C"):  # rule 6
        h = OWN[arm]
        g = pool.grid(arm, "A")
        w = win(pd.Series(g[11:], index=g[11:]), *FRESH).index
        x = pool.books(arm, "rf_bo", g, h, c50).loc[w]
        a = pool.books("A", "rf_bo", g, h, c50).loc[w]
        p = v99.paired(x, a)
        out["rule"][f"6_fresh_{arm}_vs_A_hold{h}"] = {
            **p,
            "months": int(len(w)),
            "window": [str(w[0].date()), str(w[-1].date())],
            "ann_long": ann(x),
            "ann_1m_model": ann(a),
            "helps": bool(p["t"] >= 2),
        }
    for arm in ("A", "E", "F", "C"):  # descriptive, fresh window, from each arm's 12th OOS month
        g = pool.grid(arm)
        for n in NAMES:
            for hold in (1, 3, 6, 12, 0):
                r, tc = pool.books(arm, n, g, hold)
                idx = win(r.loc[g[11:]], *FRESH).index
                out["cells"][f"{n}@{arm}/{hold_key(hold)}"] = cell(r.loc[idx], tc.loc[idx], pool.uni.reindex(idx))
    p99 = OUT / "KR_v99_horizons.json"
    out["recent_99"] = json.loads(p99.read_text())["cells"] if p99.exists() else {}
    (OUT / "KR_v101_ef.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_ef_md(out)
    say(f"wrote {OUT / 'KR_v101_ef.json'}")


def cut_months(ds, years: int) -> tuple[set, set]:
    """Test months whose context's training window is cut by `years` (earliest labelled row <= train_end - years)."""
    data = sel._Data(ds)
    outer, hold, _, _ = sel.make_contexts(data)
    first = pd.Timestamp(data.date[data.lab].min())
    cut, keep = set(), set()
    for ctx in [*outer, hold]:
        (cut if first <= pd.Timestamp(ctx.train_end) - pd.DateOffset(years=years) else keep).update(ctx.test_dates)
    del data
    gc.collect()
    return cut, keep


def analyse_roll() -> None:
    c50 = COSTS[0]
    out = {"version": V101_VERSION, "rule": {}, "cells": {}, "checks": {}, "state": state()}
    pool = Pool()
    ds, res = v100.run("A", no_fit=True)
    pool.add("A", res, base=True)
    cuts = {arm: cut_months(ds, y) for arm, (b, y) in ROLL.items() if b == "A"}
    del ds, res
    _, res = v100.run("B", no_fit=True)
    pool.add("B", res)
    for arm in ("R3", "R5", "R8"):
        _, res = run_roll(arm, no_fit=True)
        pool.add(arm, res)
        del res
        gc.collect()
    for arm in ("R3", "R5", "R8"):  # rule 7
        g = pool.grid("A", arm)
        x, a = pool.books(arm, "rf_bo", g, 1, c50), pool.books("A", "rf_bo", g, 1, c50)
        p = v99.paired(x, a)
        halves = {f"{lo[:4]}-{hi[:4]}": paired_t(win(x - a, lo, hi)) for lo, hi in HALVES}
        cut, keep = cuts[arm]
        out["rule"][f"7_{arm}_vs_A"] = {
            **p,
            "months": len(g),
            "window": [str(g[0].date()), str(g[-1].date())],
            "halves": halves,
            "ann": ann(x),
            "ann_A": ann(a),
            "cut_months": len(cut & set(g)),
            "helps": bool(p["t"] >= T_ROLL and all(h["mean"] > 0 for h in halves.values())),
        }
        out["checks"][f"{arm}_same_as_A_uncut"] = pool.same_preds(arm, "A", keep)
        gb = pool.grid("B", arm)  # descriptive: vs the hand-picked 2016 start
        out["cells"][f"{arm}_vs_B_recent"] = {
            **v99.paired(pool.books(arm, "rf_bo", gb, 1, c50), pool.books("B", "rf_bo", gb, 1, c50)),
            "window": [str(gb[0].date()), str(gb[-1].date())],
        }
    for key in ("A", "B", "R3", "R5", "R8"):  # descriptive cells: fresh / recent windows, each key's own grid
        g = pool.grid(key)
        for n in NAMES:
            for hold in (1, 0):
                r, tc = pool.books(key, n, g, hold)
                for wn, (lo, hi) in (("fresh", FRESH), ("recent", RECENT)):
                    idx = win(r, lo, hi).index
                    if len(idx) >= 24:
                        out["cells"][f"{n}@{key}/{hold_key(hold)}/{wn}"] = cell(
                            r.loc[idx], tc.loc[idx], pool.uni.reindex(idx)
                        )
    del pool
    gc.collect()
    ps = Pool()  # serving data (99's 1-month, 2016 start, all features)
    ds, res = v99.run(1, no_fit=True)
    ps.add("X", res, base=True)
    cuts = {arm: cut_months(ds, y) for arm, (b, y) in ROLL.items() if b == "S"}
    del ds, res
    for arm in ("S3", "S5"):
        _, res = run_roll(arm, no_fit=True)
        ps.add(arm, res)
        del res
        gc.collect()
    for arm in ("S3", "S5"):  # rule 8 (confirmation)
        g = ps.grid("X", arm)
        cut, keep = cuts[arm]
        w = [d for d in g if d in cut]
        x, a = ps.books(arm, "rf_bo", g, 1, c50), ps.books("X", "rf_bo", g, 1, c50)
        p = paired_t((x - a).loc[w]) if w else {"mean": float("nan"), "t": float("nan")}
        out["rule"][f"8_{arm}_vs_99_1m"] = {
            **p,
            "months": len(w),
            "window": [str(w[0].date()), str(w[-1].date())] if w else None,
            "ann": ann(x.loc[w]) if w else None,
            "ann_99": ann(a.loc[w]) if w else None,
            "not_worse": bool(w and p["mean"] >= 0),
        }
        out["checks"][f"{arm}_same_as_99_uncut"] = ps.same_preds(arm, "X", keep)
    for key in ("X", "S3", "S5"):
        g = ps.grid(key)
        r, tc = ps.books(key, "rf_bo", g, 1)
        out["cells"][f"rf_bo@{key}/hold1/serving_oos"] = cell(r, tc, ps.uni.reindex(r.index))
    (OUT / "KR_v101_roll.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_roll_md(out)
    say(f"wrote {OUT / 'KR_v101_roll.json'}")


# ------------------------------------------------------------------------------------------------ markdown


def pc(x: float) -> str:
    return f"{100 * x:+.1f}%"


def write_ef_md(o: dict) -> None:
    r = o["rule"]
    lines = ["# KR 101.1: 3·6개월 모델의 옛 구간(2005~2015) 확인", "", "## 판정 (사전 고정, 비용 50bp)", ""]
    for k, v in r.items():
        if k.startswith("5_"):
            ok = "확인" if v["confirmed"] else "확인 안 됨"
            lines.append(f"- 5. `{k}`: 연 {pc(v['ann'])}, 전체 대비 t {v['t_vs_u']:+.2f} ({v['months']}개월) → {ok}")
        else:
            ok = "도움" if v["helps"] else "도움 없음"
            lines.append(
                f"- 6. `{k}`: 월 {100 * v['mean_month']:+.2f}%p, t {v['t']:+.2f}, 연 {pc(v['ann_long'])} 대 "
                f"1개월 모델 {pc(v['ann_1m_model'])} ({v['window'][0]}~{v['window'][1]}) → {ok}"
            )
    lines += [
        "",
        "## 옛 구간(2005~2015, 사전 고정 행)과 99절 최근 구간(2019-12~2025-10)",
        "",
        "| 칸 | 옛 50bp | 옛 100bp | 옛 t50 | 베타 | 알파 t | 최근 50bp | 최근 100bp | 최근 t50 | 월 교체(옛/최근) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    rec = o.get("recent_99", {})
    for arm, h in OWN.items():
        for n in ("rf_bo", "rf"):
            for hold in dict.fromkeys((1, h, 0)):
                c = o["cells"].get(f"{n}@{arm}/{hold_key(hold)}")
                q = rec.get(f"{n}@{h}m/{hold_key(hold)}")
                if c is None:
                    continue
                rq = (
                    f"{pc(q['net50']['ann'])} | {pc(q['net100']['ann'])} | {q['net50']['t_vs_u']:+.2f}"
                    if q
                    else "— | — | —"
                )
                tq = f"{100 * q['turnover']:.0f}%" if q else "—"
                lines.append(
                    f"| {h}개월 `{n}` {hold_key(hold)} | {pc(c['net50']['ann'])} | {pc(c['net100']['ann'])} | "
                    f"{c['net50']['t_vs_u']:+.2f} | {c['net50']['beta']:.2f} | {c['net50']['t_alpha']:+.2f} | {rq} | "
                    f"{100 * c['turnover']:.0f}% / {tq} |"
                )
    u = o["cells"].get("rf_bo@A/hold1")
    if u:
        lines += ["", f"옛 구간 전체(동일가중) 연 {pc(u['universe_ann'])}, {u['span'][0]}~{u['span'][1]}"]
    (OUT / "KR_v101_ef.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_roll_md(o: dict) -> None:
    lines = ["# KR 101.2: 최근 N년만 학습(롤링 창)", "", "## 판정 (사전 고정, `rf_bo` D20 매달, 50bp)", ""]
    for k, v in o["rule"].items():
        if k.startswith("7_"):
            hv = ", ".join(f"{h}: 월 {100 * x['mean']:+.2f}%p t {x['t']:+.2f}" for h, x in v["halves"].items())
            ok = "도움" if v["helps"] else "도움 없음"
            lines.append(
                f"- 7. `{k}`: 월 {100 * v['mean_month']:+.2f}%p, t {v['t']:+.2f} (기준 {T_ROLL}), 연 {pc(v['ann'])} 대 "
                f"A {pc(v['ann_A'])}; 절반 {hv}; 잘린 달 {v['cut_months']}/{v['months']} → {ok}"
            )
        else:
            ok = "나빠지지 않음" if v["not_worse"] else "나빠짐"
            lines.append(
                f"- 8. `{k}`: 잘린 달 {v['months']}개 {v['window']}, "
                f"월 {100 * v['mean']:+.2f}%p, t {v['t']:+.2f} → {ok}"
            )
    chk = ", ".join(f"{k} {v:.4f}" for k, v in o["checks"].items())
    lines += ["", f"점검(창이 안 잘린 달의 예측이 같은 비율): {chk}"]
    lines += ["", "| 칸 | 구간 | 50bp | 100bp | t50 | 베타 | 월 교체 |", "|---|---|---|---|---|---|---|"]
    for k, c in o["cells"].items():
        if "net50" not in c:
            continue
        lines.append(
            f"| `{k}` | {c['span'][0]}~{c['span'][1]} | {pc(c['net50']['ann'])} | {pc(c['net100']['ann'])} | "
            f"{c['net50']['t_vs_u']:+.2f} | {c['net50']['beta']:.2f} | {100 * c['turnover']:.0f}% |"
        )
    lines += [""] + [
        f"- `{k}`: 월 {100 * c['mean_month']:+.2f}%p, t {c['t']:+.2f} ({c['window'][0]}~{c['window'][1]})"
        for k, c in o["cells"].items()
        if k.endswith("_vs_B_recent")
    ]
    (OUT / "KR_v101_roll.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "study", "analyse"))
    ap.add_argument("--h", type=int, choices=(3, 6))
    ap.add_argument("--arm", choices=("E", "F", *ROLL))
    ap.add_argument("--part", choices=("ef", "roll"))
    a = ap.parse_args()
    if a.cmd == "build":
        v100.build(a.h)
    elif a.cmd == "study":
        study(a.arm)
    elif a.part == "ef":
        analyse_ef()
    else:
        analyse_roll()


if __name__ == "__main__":
    main()
