"""KR total-return labels (docs/MODELING.md 45, pre-declared 2026-10-12): the KR price (marcap base-price chain) omits
cash dividends, so the label of a row whose window (t, t_end] contains an ex-dividend date gets log(1 + dividend /
the previous Yahoo close) added. The same KR v3 inputs and candidates are selected twice - price labels (A) and
total-return labels (B) - so the two are comparable. Research only.

    .venv/bin/python scripts/kr_total_return_study.py fetch     # Yahoo dividends + closes of the KR universe
    .venv/bin/python scripts/kr_total_return_study.py run

Writes reports/modeling/KR_tr_{price,total}.json and reports/modeling/KR_tr_summary.md.
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
import keyed_data as kd  # noqa: E402
import v13_study as v13s  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
CANDS = ("logit_w", "lgbm_clf", "et")
DIAG_COLS = ["x_bp", "x_ep", "x_roe", "x_vol60"]


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def root(settings) -> Path:
    return kd.root(settings, "kr_tr")


def fetch(settings, cache) -> dict:
    """Per code: Yahoo daily Close (not dividend-adjusted) and Dividends (.KS, else .KQ)."""
    import yfinance as yf

    from quantcio.data.pit import load_pit_universe
    from quantcio.modeling.dataset import universe_params
    from quantcio.models import Market

    n, start = universe_params(Market.KR, settings)
    m = load_pit_universe(Market.KR, settings, cache, n, start).membership
    mk = m.sort_values("date").groupby("entity")["market"].last().to_dict()
    d = root(settings)
    got, none = 0, []
    for i, (code, market) in enumerate(sorted(mk.items()), 1):
        p = d / f"{code}.parquet"
        if p.exists():
            got += 1
            continue
        h = None
        for suf in (".KS", ".KQ") if market != "KOSDAQ" else (".KQ", ".KS"):
            try:
                h = yf.Ticker(code + suf).history(period="max", auto_adjust=False, actions=True)
            except Exception:  # noqa: BLE001 - delisted / unknown symbols return nothing
                h = None
            if h is not None and len(h):
                break
            time.sleep(0.3)
        if h is None or not len(h):
            none.append(code)
            pd.DataFrame(columns=["close", "div"]).to_parquet(p)
            continue
        idx = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
        pd.DataFrame({"close": h["Close"].to_numpy(), "div": h["Dividends"].to_numpy()}, index=idx).to_parquet(p)
        got += 1
        time.sleep(0.3)
        if i % 50 == 0:
            say(f"Yahoo KR {i}/{len(mk)}")
    return {"codes": len(mk), "with_data": got, "missing": len(none)}


def dividend_logs(f: pd.DataFrame, settings) -> tuple[np.ndarray, dict]:
    """Per row: sum of log(1 + D / previous close) over ex-dates in (t, t_end]."""
    out = np.zeros(len(f))
    pos = {ix: i for i, ix in enumerate(f.index)}
    n_ent, n_ev = 0, 0
    for ent, g in f.groupby("entity", sort=False):
        p = root(settings) / f"{ent}.parquet"
        if not p.exists():
            continue
        h = pd.read_parquet(p)
        if h.empty:
            continue
        h = h.sort_index()
        prev = h["close"].shift(1)
        ev = h[(h["div"] > 0) & (prev > 0)]
        if ev.empty:
            continue
        n_ent += 1
        n_ev += len(ev)
        ex = ev.index.to_numpy(dtype="datetime64[ns]")
        lg = np.log1p((ev["div"] / prev.loc[ev.index]).to_numpy(dtype=float).clip(0, 0.5))
        cum = np.concatenate([[0.0], np.cumsum(lg)])
        t0 = g["date"].to_numpy(dtype="datetime64[ns]")
        t1 = pd.to_datetime(g["t_end"]).to_numpy(dtype="datetime64[ns]")
        lo = np.searchsorted(ex, t0, "right")  # ex-date > t
        hi = np.searchsorted(ex, t1, "right")  # ex-date <= t_end
        val = cum[hi] - cum[lo]
        val[np.isnat(t1)] = 0.0
        rows = np.array([pos[i] for i in g.index])
        out[rows] = val
    return out, {"entities_with_dividends": n_ent, "dividend_events": n_ev}


def candidates() -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    return [every["mom_12_1"], every["factor_ic"], *(every[x] for x in CANDS)]


def ic_table(f: pd.DataFrame, label: str) -> dict:
    lab = f[f["labelled"] & f["is_me"]]
    out = {}
    for c in [*DIAG_COLS, "x_div_ttm"]:
        if c not in lab:
            continue
        s = mt.per_date_ic(lab.rename(columns={c: "_s"}), "_s", label)
        out[c] = mt.summarize(s, 0.0)
    return out


def trailing_yield(f: pd.DataFrame, settings) -> np.ndarray:
    """Diagnostic only: trailing 12-month Yahoo dividends / close as of t."""
    from quantcio.modeling import features as feat

    out = np.full(len(f), np.nan)
    pos = {ix: i for i, ix in enumerate(f.index)}
    for ent, g in f.groupby("entity", sort=False):
        p = root(settings) / f"{ent}.parquet"
        if not p.exists():
            continue
        h = pd.read_parquet(p)
        if h.empty:
            continue
        h = h.sort_index()
        dy = (h["div"].rolling("365D").sum() / h["close"]).where(h["close"] > 0)
        idx = h.index.to_numpy(dtype="datetime64[ns]")
        k = np.searchsorted(idx, g["date"].to_numpy(dtype="datetime64[ns]"), "right") - 1
        ok = k >= 0
        v = np.full(len(g), np.nan)
        v[ok] = dy.to_numpy()[k[ok]]
        out[np.array([pos[i] for i in g.index])] = v
    s = pd.Series(out, index=f.index)
    return feat.rank_gauss(s, f["date"]).to_numpy()


def run_one(tag: str, base, settings, cache, div: np.ndarray | None):
    from quantcio.modeling.dataset import add_date_targets

    ds = copy.copy(base)
    ds.frame = base.frame.copy()
    f = ds.frame
    if div is not None:
        f["y_e"] = f["y_e"] + np.where(f["labelled"], div, 0.0)
        f["r"] = f["r"] + np.where(f["labelled"], div, 0.0)
        add_date_targets(f)
    ds.snapshot = f"{base.snapshot}|tr45|{tag}"
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    sel.DIAGNOSTIC_PAIRS = ()
    res = sel.run_selection(ds, settings, cache, candidates=candidates(), threads=8, progress=say)
    (OUT / f"KR_tr_{tag}.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    return res, f


def write_summary(info: dict) -> str:
    rows = {}
    for tag in ("price", "total"):
        d = json.loads((OUT / f"KR_tr_{tag}.json").read_text())
        rows[tag] = {c["name"]: c for c in d["candidates"]}
    di = info["div"]
    lines = [
        "# KR 배당 포함 수익률 정답 (docs/MODELING.md 45)",
        "",
        f"- 배당이 있는 엔티티 {di['entities_with_dividends']}, 배당락 {di['dividend_events']}건",
        f"- 정답 창에 배당락이 든 월말 행 비율 {info['rows_with_div']:.3f}, "
        f"그 행들의 평균 배당 로그수익 {info['mean_div_log']:+.4f}",
        "",
        "| 후보 | 가격 정답 외부 AUC [95% CI] | 배당 포함 외부 AUC [95% CI] | 가격 홀드아웃 | 배당 포함 홀드아웃 |",
        "|---|---|---|---|---|",
    ]

    def ci(o):
        return f"{o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}]"

    for n in rows["price"]:
        a, b = rows["price"][n], rows["total"].get(n)
        if b is None:
            continue
        ha = ((a["holdout"] or {}).get("auc_median") or {}).get("mean")
        hb = ((b["holdout"] or {}).get("auc_median") or {}).get("mean")
        lines.append(f"| {n} | {ci(a['outer']['auc_median'])} | {ci(b['outer']['auc_median'])} | {ha:.4f} | {hb:.4f} |")
    lines += [
        "",
        "## 특징의 날짜별 IC (월말, 전체 기간)",
        "",
        "| 특징 | 가격 정답 IC (t) | 배당 포함 IC (t) |",
        "|---|---|---|",
    ]
    nan = float("nan")
    for c, v in info["ic_price"].items():
        w = info["ic_total"].get(c, {})
        lines.append(f"| {c} | {v['mean']:+.4f} ({v['t']:+.2f}) | {w.get('mean', nan):+.4f} ({w.get('t', nan):+.2f}) |")
    text = "\n".join(lines) + "\n"
    (OUT / "KR_tr_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    if sys.argv[1:2] == ["fetch"]:
        say(f"Yahoo KR dividends: {fetch(settings, cache)}")
        return
    _, base = v13s.load_inputs("KR", settings, cache)
    f = base.frame
    div, dinfo = dividend_logs(f, settings)
    f["x_div_ttm"] = trailing_yield(f, settings)  # diagnostic only, not a model input (not in xs_cols)
    lab = f["labelled"].to_numpy() & f["is_me"].to_numpy()
    info = {
        "div": dinfo,
        "rows_with_div": float((div[lab] > 0).mean()),
        "mean_div_log": float(div[lab][div[lab] > 0].mean()) if (div[lab] > 0).any() else 0.0,
        "ic_price": ic_table(f, "y_e"),
    }
    f["y_e_tr"] = f["y_e"] + np.where(f["labelled"], div, 0.0)
    info["ic_total"] = ic_table(f, "y_e_tr")
    say(f"dividends {json.dumps({k: v for k, v in info.items() if not k.startswith('ic')}, default=str)}")
    run_one("price", base, settings, cache, None)
    run_one("total", base, settings, cache, div)
    (OUT / "KR_tr_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str))
    print(write_summary(info), flush=True)


if __name__ == "__main__":
    main()
