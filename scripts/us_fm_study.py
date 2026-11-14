"""US factor momentum on the monthly model (docs/MODELING.md 41, pre-declared 2026-10-12): rule scores that follow the
trailing performance of nine fixed characteristics (fm12 / fm1 / fm_sign) and the fixed blend cat_fm = rank mean of the
served-configuration cat_rank and fm12. Research only - the served model is not changed.

    .venv/bin/python scripts/us_fm_study.py

Writes reports/modeling/US_fm.{md,json} and reports/modeling/US_fm_summary.md.
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
import v13_study as v13s  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
CHARS = ["x_beta", "x_vol60", "x_ivol60", "x_log_mcap", "x_bp", "x_ep", "x_mom_12_1", "x_ret_1m", "x_roe"]
FM_COLS = {"fm12": "x_fm12", "fm1": "x_fm1", "fm_sign": "x_fm_sign"}
MIN_MONTHS = 6


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def factor_momentum(f: pd.DataFrame, chars: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per row: fm12 / fm1 / fm_sign scores (41.1) and the per-date weight table (for the report)."""
    me = f[f["is_me"] & f["labelled"] & f["y_e"].notna()]
    ic = pd.DataFrame({c: mt.per_date_ic(me.rename(columns={c: "_s"}), "_s", "y_e") for c in chars}).sort_index()
    t_end = me.groupby("date")["t_end"].max().reindex(ic.index)
    known = pd.DatetimeIndex(pd.to_datetime(t_end.to_numpy()))
    order = np.argsort(known.to_numpy())
    ic_k, known_k, month_k = ic.iloc[order], known[order], ic.index[order]
    dates = pd.DatetimeIndex(sorted(f["date"].unique()))
    w12, w1 = {}, {}
    for d in dates:
        usable = known_k < d  # labels that ended before the row date
        if not usable.any():
            continue
        m = month_k[usable]
        tab = ic_k[usable]
        recent = tab[(m > d - pd.Timedelta(365, "D"))]
        if len(recent.dropna(how="all")) >= MIN_MONTHS:
            w12[d] = recent.mean()
        last_m = m.max()
        if last_m >= d - pd.Timedelta(70, "D"):
            w1[d] = tab[m == last_m].iloc[-1]
    tab12 = pd.DataFrame(w12).T.reindex(columns=chars)
    tab1 = pd.DataFrame(w1).T.reindex(columns=chars)
    X = f[chars].astype(float).fillna(0.0).to_numpy()
    rows_d = pd.DatetimeIndex(f["date"])
    out = pd.DataFrame(index=f.index)
    for name, tab, sign in (("fm12", tab12, False), ("fm1", tab1, False), ("fm_sign", tab12, True)):
        wr = tab.reindex(rows_d).to_numpy(dtype=float)
        if sign:
            wr = np.sign(wr) / len(chars)
        ok = np.isfinite(wr).all(axis=1)
        s = np.where(ok, np.einsum("ij,ij->i", np.nan_to_num(wr), X), np.nan)
        out[FM_COLS[name]] = s
    return out, tab12


def rule_candidate(name: str, col: str, ko: str, en: str) -> cmod.Candidate:
    cmod._FITTERS[name] = lambda c, train, params, scheme, ctx, col=col: cmod.Fitted(
        c.name, params, scheme, None, [col], "column"
    )
    return cmod.Candidate(name, "baseline", ko, en, tuned=False)


CAT_FM = cmod.Candidate(
    "cat_fm",
    "ensemble",
    "고정 혼합(cat_rank + fm12 순위 평균)",
    "fixed blend (rank mean of cat_rank + fm12)",
    tuned=False,
)
_orig_rule = sel.composite_rule


def composite_rule(name, inner, cands):
    if name == CAT_FM.name:
        mem = [m for m in ("cat_rank", "fm12") if m in inner]
        return (mem, [1.0, 1.0]) if len(mem) == 2 else ([], [])
    return _orig_rule(name, inner, cands)


def candidates() -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    cat = copy.copy(every["cat_rank"])
    cat.feature_set = "core"  # the served configuration: v9 inputs only, never the factor-momentum scores
    return [
        every["mom_12_1"],
        every["factor_ic"],
        cat,
        rule_candidate("fm12", "x_fm12", "성향 따라가기(12개월 평균)", "factor momentum (12-month mean)"),
        rule_candidate("fm1", "x_fm1", "성향 따라가기(지난달)", "factor momentum (last month)"),
        rule_candidate("fm_sign", "x_fm_sign", "성향 따라가기(12개월 부호)", "factor momentum (12-month sign)"),
    ]


def per_year(res, names: list[str]) -> dict:
    out = {}
    for part, frame in (("outer", res.oos), ("holdout", res.holdout_preds)):
        if frame is None:
            continue
        for n in names:
            if n not in frame:
                continue
            s = mt.per_date_primary(frame, n)
            out.setdefault(part, {})[n] = {
                str(y): round(float(v.mean()), 4) for y, v in s.groupby(pd.DatetimeIndex(s.index).year)
            }
    return out


def write_summary() -> str:
    d = json.loads((OUT / "US_fm.json").read_text())
    info = d["protocol"]["fm"]
    cand = {c["name"]: c for c in d["candidates"]}
    lines = [
        "# US 성향 따라가기(팩터 모멘텀) — 월간 (docs/MODELING.md 41)",
        "",
        f"- 성향 9개: {', '.join(info['chars'])}",
        f"- 최신 12개월 가중(마지막 날짜): {info['latest_w12']}",
        "",
        "| 후보 | 외부 AUC [95% CI] | 외부 IC (t) | 홀드아웃 AUC | 상위 5분위 순초과(월, 외부) |",
        "|---|---|---|---|---|",
    ]
    for n, c in cand.items():
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        tn = c["outer"].get("top_net") or {}
        hm = f"{h['mean']:.4f}" if h.get("mean") is not None else "–"
        tnm = f"{tn['mean']:+.4%} (t {tn['t']:+.2f})" if tn.get("mean") is not None and tn.get("t") is not None else "–"
        lines.append(
            f"| {n} | {o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] | {c['outer']['ic']['mean']:+.4f} "
            f"({c['outer']['ic']['t']:+.2f}) | {hm} | {tnm} |"
        )
    av = d["protocol"].get("alt_value") or {}
    lines += ["", "## 판정 (41.2)", ""]
    pair = av.get("cat_fm - cat_rank")
    ok1 = False
    if pair:
        o, h = pair["outer"], pair.get("holdout") or {}
        ok1 = o["lo"] is not None and o["lo"] > 0 and (h.get("mean") is None or h["mean"] >= 0)
        lines.append(
            f"1. cat_fm − cat_rank: 외부 {o['mean']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}] (t {o['t']:+.2f})"
            f" · 홀드아웃 {h.get('mean', float('nan')):+.4f} → **{'나아짐' if ok1 else '나아지지 않음'}**"
        )
    fm = cand.get("fm12", {}).get("outer", {}).get("auc_median", {})
    ok2 = fm.get("lo") is not None and fm["lo"] > 0.5
    lines.append(
        f"2. fm12 자체: 외부 AUC {fm.get('mean', float('nan')):.4f} [{fm.get('lo', float('nan')):.4f}, "
        f"{fm.get('hi', float('nan')):.4f}] → **{'실력 있음' if ok2 else '실력 없음'}**"
    )
    lines += ["", "## 연도별 AUC(중앙값 초과) — 참고(홀드아웃은 이미 본 기간)", ""]
    yrs = info.get("per_year", {})
    for part in ("outer", "holdout"):
        for n, v in (yrs.get(part) or {}).items():
            lines.append(f"- {part} {n}: {v}")
    text = "\n".join(lines) + "\n"
    (OUT / "US_fm_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    if "--summary" in sys.argv:
        print(write_summary())
        return
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    _, ds = v13s.load_inputs("US", settings, cache)
    f = ds.frame
    base_xs, base_reg = list(ds.xs_cols), list(ds.regime_cols)
    say("factor momentum scores")
    sc, w12 = factor_momentum(f, CHARS)
    for c in sc.columns:
        f[c] = sc[c].to_numpy()
        feat.FEATURE_DOCS[c] = (c, c)
    lab = f[f["labelled"] & f["is_me"]]
    info = {
        "chars": CHARS,
        "missing_share": {c: round(float(lab[c].isna().mean()), 3) for c in sc.columns},
        "latest_w12": {k: round(float(v), 4) for k, v in w12.iloc[-1].items()} if len(w12) else {},
        "w12_dates": int(len(w12)),
    }
    say(f"fm info {json.dumps(info)}")
    ds.core_xs_cols, ds.core_regime_cols = base_xs, base_reg
    ds.xs_cols = [*base_xs, *sc.columns]
    ds.snapshot = f"{ds.snapshot}|fm41|{len(sc.columns)}"
    sel.composite_rule = composite_rule
    sel.COMPOSITE_CANDIDATES = (CAT_FM,)
    sel.COMPOSITES = (CAT_FM.name,)
    sel.DIAGNOSTIC_PAIRS = (("cat_fm", "cat_rank"), ("fm12", "mom_12_1"))
    res = sel.run_selection(ds, settings, cache, candidates=candidates(), threads=8, progress=say)
    info["per_year"] = per_year(res, ["cat_rank", "cat_fm", "fm12", "fm1", "fm_sign"])
    res.protocol["fm"] = {"declared": "docs/MODELING.md 41", **info}
    (OUT / "US_fm.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    try:
        from quantcio.modeling.report import render_markdown

        (OUT / "US_fm.md").write_text(render_markdown(res, settings), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - the summary below is the result; the full report is a convenience
        say(f"full report skipped: {type(exc).__name__}: {exc}")
    print(write_summary(), flush=True)


if __name__ == "__main__":
    main()
