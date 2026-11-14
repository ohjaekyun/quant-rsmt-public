"""US: retrain every model family on the SEC-fixed data + ventile books (docs/MODELING.md 95, pre-declared 2026-11-07).
Research only: src/quantcio and the served models are not changed (no serving artifact is written).

    .venv/bin/python scripts/v95_us_multi.py study      # build the dataset (cached vintages), run, analyse
    .venv/bin/python scripts/v95_us_multi.py analyse    # re-read the cached run (no fit), analyse again

Dataset: ``build_dataset(US)`` through the vintage stores (identity db6486524f64, frozen cutoff 2026-10-31); the run
stops if the frozen-window digest does not match. Candidates: the 35 defaults (+ composites) and the ordering models
of 87-91 (``lgbm_reg_w_cv``, ``vg_up``, ``vgw_up``, ``lgbm_reg_w_ord``); the usual protocol (outer walk-forward,
24-month holdout, final context, 1-SE). Books (91.5 / 94.1 rules): month-end equal-weight ventiles held 20 trading
days, 10 bp, per bin annualised; D20 - universe t (outer / holdout), D20 without its best 10 months, max drawdown,
rho20, in-sample vs OOS IC. Display rule (95): outer t >= 2 and holdout mean excess > 0; up to 2 by outer t.
Writes reports/modeling/US_v95.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v86_full_es as v86  # noqa: E402
import v87_validation as v87  # noqa: E402
import v88_ventile as v88  # noqa: E402
import v89_order_loss as v89  # noqa: E402
import v90_order_tree as v90  # noqa: E402
import v91_order_weekly as v91  # noqa: E402

from quantcio.config import load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.cli import guard_xsh  # noqa: E402
from quantcio.modeling.dataset import build_dataset  # noqa: E402
from quantcio.models import Market  # noqa: E402

M = "US"
K = 20
V95_VERSION = 1
EXTRA = ("lgbm_reg_w_cv", "vg_up", "vgw_up", "lgbm_reg_w_ord")
T_MIN = 2.0  # 95: display rule
MAX_SHOWN = 2
OUT = v86.OUT
STATE = OUT / "US_v95_state.json"
say = v86.say


def candidates() -> list[cmod.Candidate]:
    base = list(cmod.default_candidates())
    have = {c.name for c in base}
    ext = {c.name: c for c in v91.candidates()}  # registers the 87-91 fitters
    return base + [ext[n] for n in EXTRA if n not in have]


def research_dir(s) -> Path:
    p = s.cache_path / "research" / "v95" / M
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logs(s) -> None:
    rd = research_dir(s)
    for mod in (v87, v88, v89, v90, v91):
        mod.LOG[:] = [rd / "choice_log.jsonl"]
    v86.ES_LOG[:] = [rd / "es_log.jsonl"]
    v86.ARM[0] = "v95"


def dataset(s, cache, rebuild: bool):
    if STATE.exists() and not rebuild:
        st = json.loads(STATE.read_text())
        ds = cache.get("modeling_dataset", st["snapshot"], None)
        if ds is None:
            raise SystemExit(f"dataset {st['snapshot']} is not cached")
    else:
        ds = build_dataset(Market.US, s, cache, use_cache=True, progress=say)
        dig = (ds.coverage.get("vintage") or {}).get("digest") or {}
        say(f"dataset {ds.snapshot} as_of {ds.as_of.date()} rows {len(ds.frame):,}; frozen digest {dig}")
        if dig.get("status") != "match":
            raise SystemExit(f"frozen-window digest is not a match: {dig}")
        STATE.write_text(
            json.dumps(
                {"snapshot": ds.snapshot, "as_of": str(ds.as_of.date()), "digest": dig, "version": V95_VERSION},
                indent=1,
                default=str,
            )
        )
    return guard_xsh(ds, s, say)


def run(ds, s, cache, cands, no_fit: bool):
    def guard(msg: str) -> None:
        say(msg)
        if no_fit and "train <=" in msg:
            raise SystemExit(f"analyse: a context is not cached ({msg})")

    snap0 = ds.snapshot
    ds.snapshot = f"{snap0}|v95|{V95_VERSION}"
    try:
        return sel.run_selection(ds, s, cache, candidates=cands, threads=8, use_cache=True, progress=guard)
    finally:
        ds.snapshot = snap0


# ------------------------------------------------------------------------------------------------ books


def bins(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Per month-end: the mean simple return R = e^r - 1 of every ventile (columns 1..K) and of the universe (0), the
    top book's one-way turnover (``turn``)."""
    d = frame[["date", "entity", name, "y_e", "r"]].dropna(subset=[name, "y_e", "r"])
    d = d[d.groupby("date")["y_e"].transform("size").to_numpy() >= mt.MIN_NAMES]
    d = d[d.groupby("date")[name].transform("nunique").to_numpy() > 1]
    g = d.groupby("date")
    u = (g[name].rank(method="average") - 0.5) / g[name].transform("size")
    d = d.assign(bin=np.clip(np.floor(u * K), 0, K - 1).astype(int) + 1, R=np.expm1(d["r"].to_numpy(float)))
    out = d.pivot_table(index="date", columns="bin", values="R", aggfunc="mean")
    out[0] = d.groupby("date")["R"].mean()
    books = d[d["bin"] == K].groupby("date")["entity"].apply(set).reindex(out.index)
    turn, prev = [], None
    for x in books:
        if not isinstance(x, set) or not x:
            turn.append(np.nan)
            prev = None
            continue
        turn.append(1.0 if prev is None else len(x - prev) / len(x))
        prev = x
    out["turn"] = turn
    out.index = pd.DatetimeIndex(out.index)
    return out


def comp(r: pd.Series) -> dict:
    x = r.fillna(0.0).to_numpy()
    nav = np.cumprod(1 + x)
    peak = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    return {"ann": float(nav[-1] ** (12 / len(x)) - 1), "cum": float(nav[-1] - 1), "mdd": float((nav / peak - 1).min())}


def tstat(x: pd.DataFrame) -> tuple[float, float]:
    e = (x[K] - x[0]).fillna(0.0)
    return float(e.mean()), float(e.mean() / e.std(ddof=1) * np.sqrt(len(e)))


def rho20(frame: pd.DataFrame, n: str) -> float:
    _, m = v88.vmat(frame, n)
    if len(m) < 3:
        return float("nan")
    return v88.rho((m - m.mean(axis=1, keepdims=True)).mean(axis=0))


def analyse(res, ds, cands) -> dict:
    cost = float(ds.cost_bps) / 1e4
    fr = {"outer": res.oos, "holdout": res.holdout_preds}
    names = [c for c in res.oos.columns if c not in ("date", "entity", "y_e", "r") and c in res.holdout_preds]
    fitted = res.final.fitted if res.final is not None else {}
    ins = v88.insample(res, ds, [n for n in names if n in fitted and n not in sel.STACKED])
    family = {c.name: c.family for c in cands}
    rows = {}
    for n in names:
        b = {w: bins(f, n) for w, f in fr.items()}
        b["all"] = pd.concat([b["outer"], b["holdout"]]).sort_index()
        a = b["all"]
        top = a[K].fillna(0.0).to_numpy()
        keep = np.ones(len(top), bool)
        keep[np.argsort(top)[-10:]] = False
        r = {"family": family.get(n, "composite")}
        r["months"] = {w: int(len(x)) for w, x in b.items()}
        r["ann"] = {w: {str(k): comp(x[k])["ann"] for k in range(0, K + 1) if k in x} for w, x in b.items()}
        r["D20"] = {**comp(a[K]), "worst": float(a[K].min()), "net_ann": comp(a[K] - a["turn"] * cost)["ann"]}
        r["D20_turnover"] = float(a["turn"].mean())
        r["D20_drop10"] = float(np.prod(1 + top[keep]) ** (12 / len(top)) - 1)
        for w in ("outer", "holdout"):
            r[f"ex_{w}"], r[f"t_{w}"] = tstat(b[w])
            r[f"rho_{w}"] = rho20(fr[w], n)
            r[f"ic_{w}"] = float(mt.per_date_ic(fr[w], n).mean())
        r["ic_in"] = ins.get(n, {}).get("ic")
        r["rho_in"] = ins.get(n, {}).get("rho")
        rows[n] = r
    ok = [n for n, r in rows.items() if r["t_outer"] >= T_MIN and r["ex_holdout"] > 0]
    shown = sorted(ok, key=lambda n: -rows[n]["t_outer"])[:MAX_SHOWN]
    d = res.to_dict()
    return {
        "market": M,
        "snapshot": ds.snapshot,
        "as_of": str(ds.as_of.date()),
        "cost_bps": float(ds.cost_bps),
        "outer": [str(res.oos["date"].min().date()), str(res.oos["date"].max().date())],
        "holdout": [str(res.holdout_preds["date"].min().date()), str(res.holdout_preds["date"].max().date())],
        "protocol_selected": d.get("selected"),
        "protocol_gate": d.get("gate"),
        "rule": {"t_min": T_MIN, "holdout_excess": "> 0", "max_shown": MAX_SHOWN},
        "passing": ok,
        "shown": shown,
        "models": rows,
    }


def write(o: dict) -> None:
    (OUT / "US_v95.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    rows = o["models"]
    order = sorted(rows, key=lambda n: -rows[n]["t_outer"])
    lines = [
        f"# US 95: 고친 데이터로 다시 학습 — 분위 장부 ({o['as_of']}, 비용 {o['cost_bps']:.0f}bp)",
        "",
        f"- 바깥 {o['outer'][0]}~{o['outer'][1]}, 확인 {o['holdout'][0]}~{o['holdout'][1]}",
        f"- 프로토콜 선택 `{o['protocol_selected']}` (서빙은 그대로)",
        f"- 표시 규칙(바깥 t ≥ {T_MIN}, 확인 초과 > 0) 통과: {o['passing'] or '없음'} → 표시 {o['shown'] or '없음'}",
        "",
        "| 모델 | 군 | D1 | D20 | 전체 | D20−전체 t 바깥/확인 | 최고 10개월 뺀 D20 | 최대 낙폭 | rho20 바깥/확인 "
        "| IC 학습→OOS |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in order:
        r = rows[n]
        a = r["ann"]["all"]
        ici = "—" if r["ic_in"] is None else f"{r['ic_in']:+.3f}"
        lines.append(
            f"| `{n}` | {r['family']} | {100 * a.get('1', np.nan):+.1f}% | {100 * a.get(str(K), np.nan):+.1f}% | "
            f"{100 * a['0']:+.1f}% | {r['t_outer']:+.1f} / {r['t_holdout']:+.1f} | {100 * r['D20_drop10']:+.1f}% | "
            f"{100 * r['D20']['mdd']:.0f}% | {r['rho_outer']:+.2f} / {r['rho_holdout']:+.2f} | "
            f"{ici} → {r['ic_outer']:+.3f} |"
        )
    (OUT / "US_v95.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"wrote {OUT / 'US_v95.json'} and .md; passing {o['passing']}; shown {o['shown']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["study", "analyse"])
    a = ap.parse_args()
    s = load_settings()
    cache = DiskCache(s.cache_path)
    setup_logs(s)
    cands = candidates()
    ds = dataset(s, cache, rebuild=a.cmd == "study" and not STATE.exists())
    say(f"{M}: {len(ds.xs_cols)} features, {len(cands)} candidates ({', '.join(EXTRA)} added)")
    res = run(ds, s, cache, cands, no_fit=a.cmd == "analyse")
    write(analyse(res, ds, cands))


if __name__ == "__main__":
    main()
