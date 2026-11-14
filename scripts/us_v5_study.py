"""US v5 research runs (docs/MODELING.md section 19, pre-declared 2026-10-04): short-selling data, longer holding
horizons and the S&P 400 (all PIT members). Research only - the served model is not changed.

    .venv/bin/python scripts/us_v5_study.py R1        # one run (R1..R5), see RUNS
    .venv/bin/python scripts/us_v5_study.py summary   # Benjamini-Hochberg over the finished runs

Writes reports/modeling/US_v5_<run>.{md,json} and reports/modeling/US_v5_summary.md.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import us_v4_study as v4s  # noqa: E402
import us_v5_data as v5  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.data.pit import load_pit_fundamentals  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.dataset import build_dataset  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
RUNS = {
    "R1": {"universe": "sp500", "h": 20, "label": "S&P 500 · 20거래일 · 공매도 데이터 추가"},
    "R2": {"universe": "sp500", "h": 60, "label": "S&P 500 · 60거래일"},
    "R3": {"universe": "sp500", "h": 120, "label": "S&P 500 · 120거래일"},
    "R4": {"universe": "sp400", "h": 20, "label": "S&P 400 전체 구성원 · 20거래일"},
    "R5": {"universe": "sp400", "h": 60, "label": "S&P 400 전체 구성원 · 60거래일"},
}
BASE = ("logit_w", "enet_logit", "lgbm_clf", "et", "cat_rank")


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def block_months(h: int) -> int:
    return max(3, math.ceil(h / 21))


def patch_bootstrap(h: int) -> None:
    """Block bootstrap over >= ceil(h/21) months: month-end labels overlap when h > 21 (docs/MODELING.md 19.2)."""
    b = block_months(h)
    orig = mt.block_bootstrap_means

    def patched(x, block=mt.BOOT_BLOCK, n_boot=mt.BOOT_SIMS, seed=0):
        return orig(x, max(int(block), b), n_boot, seed)

    mt.block_bootstrap_means = patched


def candidates() -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    out = [every["mom_12_1"], every["factor_ic"], *(every[n] for n in BASE)]
    for name in ("logit_w_core", "lgbm_clf_core"):  # diagnostics: without the short-selling block
        d = every[name]
        d.label_ko = d.label_ko.replace("v2 특징만", "공매도 데이터 제외")
        d.label_en = d.label_en.replace("v2 inputs only", "without short-selling data")
        out.append(d)
    return out


def load_universe(spec: dict, settings, cache):
    if spec["universe"] == "sp500":
        return None, None
    from quantcio.data.pit.us_midcap import load_us_midcap_universe

    uni = load_us_midcap_universe(settings, cache, date(2012, 1, 1), None, include_sp500=True)
    pf = pd.Timestamp((uni.coverage or {}).get("pit_from") or "2017-07-31")
    uni.membership = uni.membership[uni.membership["date"] >= pf].reset_index(drop=True)
    return uni, pf


def build(run: str, settings, cache):
    spec = RUNS[run]
    uni, pf = load_universe(spec, settings, cache)
    if uni is None:
        ds = build_dataset(Market.US, settings, cache, progress=say)
        fund = None
    else:
        from quantcio.data.alt.load import load_alt_data

        ents = sorted(uni.membership["entity"].astype(str).unique())
        alt = load_alt_data(Market.US, settings, ents, date(2011, 1, 1), progress=say)
        ds = build_dataset(Market.US, settings, cache, universe=uni, alt=alt, progress=say)
        fund = load_pit_fundamentals(Market.US, settings, cache, ents)
    v3_xs, v3_reg = list(ds.xs_cols), list(ds.regime_cols)
    cov4 = v4s.add_v4_features(ds, settings, cache, universe=uni, fundamentals=fund)
    f = ds.frame
    say("short-selling features")
    root = settings.cache_path
    syms = set(f["ticker"].map(v5.norm_symbol))
    finra = v5.finra_panel(root, syms)
    ftd, cover = v5.ftd_panel(root, syms)
    bench_src = uni if uni is not None else None
    if bench_src is None:
        from quantcio.data.pit import load_pit_universe

        bench_src = load_pit_universe(Market.US, settings, cache, 500, date.fromisoformat(settings.backtest.us_start))
    sessions = pd.DatetimeIndex(bench_src.benchmarks["SPY"].index)
    sf = v5.short_features(f[["date", "ticker", "mcap", "close"]], finra, ftd, sessions, cover)
    for c in v5.FEATURES:
        f[c] = sf[c].to_numpy()
        f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
        feat.FEATURE_DOCS[c] = v5.DOCS[c]
    v4_xs = [*v3_xs, *[feat.x_col(c) for c in v4s.v4.FEATURES]]
    ds.xs_cols = [*v4_xs, *[feat.x_col(c) for c in v5.FEATURES]]
    ds.core_xs_cols, ds.core_regime_cols = v4_xs, v3_reg  # diagnostics: everything but the short-selling block
    ds.snapshot = f"{ds.snapshot}|us_v5|{run}"
    if uni is not None:
        ds.universe_desc = {
            **(ds.universe_desc or {}),
            "description": f"S&P 400 시점별 전체 구성원({pf.date()} 이후)",
            "description_en": f"S&P 400 all PIT members (from {pf.date()})",
        }
        # a cached dataset may carry the universe notes of an older loader text: use the loaded universe's notes
        stale = "[데이터] 미국 XSH 유니버스"
        fresh = [{"ko": f"[데이터] {x}", "en": f"[data] {x}"} for x in uni.notes[:4]]
        kept = [n for n in ds.notes if not str(n.get("ko", "")).startswith("[데이터]")]
        if any(str(n.get("ko", "")).startswith(stale) for n in ds.notes):
            ds.notes = [*fresh, *kept]
    ds.alt_raw = [*list(ds.alt_raw or []), *v4s.v4.FEATURES, *v5.FEATURES]
    lab = f[f["labelled"] & f["is_me"]]
    cov = {
        "v4": cov4["missing_share"],
        "short_missing_share": {c: round(float(f[c].isna().mean()), 3) for c in v5.FEATURES},
        "short_known_by_year": lab.groupby(lab["date"].dt.year)[v5.FEATURES]
        .apply(lambda g: g.notna().mean().round(2))
        .to_dict(orient="index"),
        "pit_from": str(pf.date()) if pf is not None else None,
        "entities": int(f["entity"].nunique()),
    }
    return ds, cov


def bootstrap_p(per_date: pd.Series, h: int) -> float:
    x = per_date.dropna().to_numpy(dtype=float)
    if len(x) < 6:
        return float("nan")
    means = mt.block_bootstrap_means(x - 0.5, block_months(h))
    return float((means <= 0).mean())


def run_one(run: str) -> None:
    from quantcio.modeling.report import render_markdown

    spec = RUNS[run]
    h = int(spec["h"])
    settings = load_settings() if h == 20 else load_settings(horizon_days=h)
    cache = DiskCache(settings.cache_path)
    v4s.patch_protocol()
    patch_bootstrap(h)
    if spec["universe"] == "sp400":  # the factor backtest scores S&P 500 rows only (no S&P 400 reference)
        sel._reference_backtest = lambda *a, **k: {"available": False, "reason": "S&P 400 research run"}
    say(f"{run}: {spec['label']}")
    ds, cov = build(run, settings, cache)
    say(f"{run}: coverage {json.dumps(cov['short_missing_share'])}")
    res = sel.run_selection(ds, settings, cache, candidates=candidates(), threads=8, progress=say)
    per_date = mt.per_date_primary(res.oos, res.selected)
    p = bootstrap_p(per_date, h)
    res.protocol["us_v5"] = {"run": run, **spec, "coverage": cov, "bootstrap_block_months": block_months(h), "p": p}
    (OUT / f"US_v5_{run}.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    md = render_markdown(res, settings).replace(
        "# 종목 선택 모델 선정 — US", f"# 미국 v5 연구 {run} — {spec['label']} (서빙 아님)", 1
    )
    (OUT / f"US_v5_{run}.md").write_text(md, encoding="utf-8")
    save_per_date(run, res)
    say(f"{run}: selected {res.selected}, gate passed {res.gate['passed']}, one-sided p {p:.4f}")


def per_date_path(run: str) -> Path:
    return PROJECT_ROOT / ".cache" / "quantcio" / "research" / f"us_v5_{run}_per_date.parquet"


def save_per_date(run: str, res) -> None:
    """Per-date primary AUC of every candidate (outer + holdout) for the post-hoc checks in ``summary``."""
    parts = []
    for part, frame in (("outer", res.oos), ("holdout", res.holdout_preds)):
        names = [c["name"] for c in res.candidates if c["name"] in frame]
        parts.append(pd.DataFrame({n: mt.per_date_primary(frame, n) for n in names}).assign(part=part))
    out = pd.concat(parts)
    out.index.name = "date"
    per_date_path(run).parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_parquet(per_date_path(run), index=False)


def summary() -> None:
    rows = []
    for run, spec in RUNS.items():
        p = OUT / f"US_v5_{run}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        c = next(x for x in d["candidates"] if x["name"] == d["selected"])
        o, hdo = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        av = d["protocol"].get("alt_value") or {}
        rows.append(
            {
                "run": run,
                "label": spec["label"],
                "selected": d["selected"],
                "auc": o["mean"],
                "lo": o["lo"],
                "hi": o["hi"],
                "ic_t": c["outer"]["ic"]["t"],
                "hold": hdo.get("mean"),
                "gate": bool(d["gate"]["passed"]),
                "p": d["protocol"]["us_v5"]["p"],
                "short_value": {k: (v.get("outer") or {}).get("mean") for k, v in av.items()},
                "short_value_t": {k: (v.get("outer") or {}).get("t") for k, v in av.items()},
            }
        )
    ps = np.array([r["p"] for r in rows], dtype=float)
    order = np.argsort(ps)
    m = len(ps)
    passed = np.zeros(m, dtype=bool)
    ok = ps[order] <= 0.10 * (np.arange(1, m + 1) / m)
    if ok.any():
        passed[order[: np.max(np.nonzero(ok)[0]) + 1]] = True
    lines = [
        "# 미국 v5 연구 요약 (docs/MODELING.md 19)",
        "",
        "| 실행 | 설정 | 선정 모델 | 외부 AUC(중앙값 초과) [95% CI] | IC t(겹침 미보정) "
        "| 홀드아웃 AUC | 게이트 | 단측 p | BH(q=0.10) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r, bh in zip(rows, passed, strict=True):
        hold = f"{r['hold']:.4f}" if r["hold"] is not None else "–"
        lines.append(
            f"| {r['run']} | {r['label']} | {r['selected']} | {r['auc']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] | "
            f"{r['ic_t']:+.2f} | {hold} | {'통과' if r['gate'] else '미달'} | {r['p']:.4f} | "
            f"{'유의' if bh else '아님'} |"
        )
    lines += ["", "## 공매도 데이터의 기여 (같은 모델에서 공매도 특징만 뺀 쌍과의 날짜별 AUC 차이)", ""]
    for r in rows:
        parts = [f"{k} {v:+.4f} (t {r['short_value_t'][k]:+.2f})" for k, v in r["short_value"].items() if v is not None]
        lines.append(f"- {r['run']}: " + " · ".join(parts))
    lines += [
        "",
        "### 사후 보조 확인 (사전 선언 아님): FINRA 공매도 비율이 있는 2019년 이후 평가일만",
        "",
    ]
    for r in rows:
        p = per_date_path(r["run"])
        if not p.exists():
            continue
        pd_ = pd.read_parquet(p)
        o = pd_[(pd_["part"] == "outer") & (pd_["date"] >= "2019-01-01")].set_index("date")
        parts = []
        for full, core in sel.DIAGNOSTIC_PAIRS:
            if full in o and core in o:
                x = (o[full] - o[core]).dropna().to_numpy(dtype=float)
                draws = mt.block_bootstrap_means(x, block_months(int(RUNS[r["run"]]["h"])))
                lo, hi = np.quantile(draws, [0.025, 0.975])
                parts.append(f"{full} − {core} {x.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] (날짜 {len(x)})")
        lines.append(f"- {r['run']}: " + " · ".join(parts))
    (OUT / "US_v5_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if arg == "summary":
        summary()
    elif arg in RUNS:
        run_one(arg)
    else:
        raise SystemExit(f"usage: {sys.argv[0]} R1..R5|summary")


if __name__ == "__main__":
    main()
