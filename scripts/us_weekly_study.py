"""US one-week study (docs/MODELING.md 40, pre-declared 2026-10-12): h = 5-session labels evaluated on the last session
of every week, protocol v3 with the same calendar spans (104-week holdout, 52-week outer blocks, 156-week minimum
training, 2 x 52-week inner blocks, 13-week bootstrap blocks), v9 inputs + block W (short-horizon signals). Research
only - the served model is not changed.

    .venv/bin/python scripts/us_weekly_study.py

Writes reports/modeling/US_weekly.{md,json} and reports/modeling/US_weekly_summary.md.
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
import us_v4_data as v4  # noqa: E402
import v13_data as v13  # noqa: E402
import v13_study as v13s  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
H = 5
FRESH_DAYS = 7
W_DOCS = {
    "w_rev_sec_1w": ("1주 수익 − 섹터 평균", "1-week return minus the sector mean"),
    "w_res_1w": ("(1주 수익 − β·시장) − 섹터 평균", "(1-week return - beta x market) minus the sector mean"),
    "w_on_5": ("밤사이 수익 합(5일)", "overnight return sum (5 sessions)"),
    "w_id_5": ("장중 수익 합(5일)", "intraday return sum (5 sessions)"),
    "w_on_21": ("밤사이 수익 합(21일)", "overnight return sum (21 sessions)"),
    "w_id_21": ("장중 수익 합(21일)", "intraday return sum (21 sessions)"),
    "w_gap_1d": ("마지막 날 밤사이 수익", "last overnight return"),
    "w_vshock_5": ("거래량 충격(5일 ÷ 60일)", "volume shock (5d / 60d)"),
    "w_vshock_1": ("거래량 충격(마지막 날 ÷ 60일)", "volume shock (last day / 60d)"),
    "w_max_5": ("5일 최대 일간 수익", "max daily return (5 sessions)"),
    "w_earn_next": ("다음 5거래일 안 실적 발표 예상", "earnings expected in the next 5 sessions"),
    "w_earn_last": ("최근 5거래일 안 실적 발표", "earnings in the last 5 sessions"),
    "w_earn_days": ("마지막 실적 발표 이후 일수 ÷ 91", "days since the last earnings / 91"),
    "w_sec_1w": ("섹터 평균 1주 수익", "sector mean 1-week return"),
}
WEEKLY = {  # protocol v3 on weekly dates with the same calendar spans (40.1)
    "HOLDOUT_MONTHS": 104,
    "OUTER_MIN_TRAIN_MONTHS": 156,
    "OUTER_BLOCK_MONTHS": 52,
    "INNER_BLOCK_MONTHS": 52,
    "INNER_MIN_TRAIN_MONTHS": 52,
    "STACK_MIN_TRAIN_MONTHS": 104,
}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def daily_w(px: pd.DataFrame) -> pd.DataFrame:
    px = px.dropna(subset=["close"]).sort_index()
    c, o, v = px["close"].astype(float), px["open"].astype(float), px["volume"].astype(float)
    lc = np.log(c.where(c > 0))
    on = np.log(o.where(o > 0) / c.shift(1).where(c.shift(1) > 0))
    idr = np.log(c.where(c > 0) / o.where(o > 0))
    r1 = lc.diff()
    v60 = v.rolling(60, min_periods=40).mean().where(lambda x: x > 0)
    return pd.DataFrame(
        {
            "ret_1w": lc - lc.shift(5),
            "w_on_5": on.rolling(5).sum(),
            "w_id_5": idr.rolling(5).sum(),
            "w_on_21": on.rolling(21).sum(),
            "w_id_21": idr.rolling(21).sum(),
            "w_gap_1d": on,
            "w_vshock_5": np.log(v.rolling(5).mean().where(lambda x: x > 0) / v60),
            "w_vshock_1": np.log(v.where(v > 0) / v60),
            "w_max_5": r1.rolling(5).max(),
        }
    )


def earnings_features(react: list[pd.Timestamp], dates: np.ndarray, sessions: pd.DatetimeIndex) -> np.ndarray:
    """(next, last, days) per row date from reaction sessions <= date only."""
    out = np.full((len(dates), 3), np.nan)
    if not react:
        return out
    ev = np.array(sorted(react), dtype="datetime64[ns]")
    sess = sessions.to_numpy(dtype="datetime64[ns]")
    for i, d in enumerate(dates):
        k = np.searchsorted(ev, d, "right")
        if k == 0:
            out[i] = (0.0, 0.0, np.nan)
            continue
        last = ev[k - 1]
        p = np.searchsorted(sess, d, "right")  # sessions <= d
        win_start = sess[max(0, p - H)] if p > 0 else d
        out[i, 1] = float(last >= win_start)
        out[i, 2] = min(float((d - last) / np.timedelta64(1, "D")) / 91.0, 1.5)
        out[i, 0] = 0.0
        if k >= 2:
            gaps = np.diff(ev[max(0, k - 5) : k]).astype("timedelta64[D]").astype(float)
            gap = float(np.median(gaps[-4:]))
            if gap <= 120:
                nxt = last + np.timedelta64(int(round(gap)), "D")
                horizon_end = sess[min(len(sess) - 1, p + H - 1)] if p < len(sess) else d + np.timedelta64(7, "D")
                out[i, 0] = float(d < nxt <= horizon_end)
    return out


def block_w(ds, uni, settings, cache) -> tuple[pd.DataFrame, dict]:
    from us_v8_data import cik_chains

    from quantcio.data.pit.us_cik import Submissions, entity_cik
    from quantcio.data.pit.us_net import PitNet

    f = ds.frame
    n = len(f)
    cols = ["ret_1w", "w_on_5", "w_id_5", "w_on_21", "w_id_21", "w_gap_1d", "w_vshock_5", "w_vshock_1", "w_max_5"]
    arr = np.full((n, len(cols)), np.nan)
    pos = {ix: i for i, ix in enumerate(f.index)}
    for ent, g in f.groupby("entity", sort=False):
        px = uni.prices.get(ent)
        if px is None or len(px) < 80 or not {"open", "close", "volume"} <= set(px.columns):
            continue
        rows = np.array([pos[i] for i in g.index])
        arr[rows] = v13._asof_rows(daily_w(px), g["date"].to_numpy(dtype="datetime64[ns]"), FRESH_DAYS)
    out = pd.DataFrame(arr, index=f.index, columns=cols)
    bench = uni.benchmarks.get("SPY")
    m1w = (np.log(bench["close"].astype(float)) - np.log(bench["close"].astype(float)).shift(5)).to_frame("m")
    mkt = v13._asof_rows(m1w, f["date"].to_numpy(dtype="datetime64[ns]"), FRESH_DAYS)[:, 0]
    beta = f["beta"].astype(float).fillna(1.0).to_numpy() if "beta" in f else np.ones(n)
    keys = [f["date"], f["sector"]]
    sec = out["ret_1w"].groupby(keys).transform("mean")
    res = pd.Series(out["ret_1w"].to_numpy() - beta * mkt, index=f.index)
    out["w_sec_1w"] = sec
    out["w_rev_sec_1w"] = out["ret_1w"] - sec
    out["w_res_1w"] = res - res.groupby(keys).transform("mean")
    # earnings (8-K item 2.02 reaction sessions of every CIK of the entity's chain)
    chains = cik_chains(ds, settings, cache)
    subs = Submissions(PitNet(settings))
    sessions = pd.DatetimeIndex(ds.calendar)
    start = pd.Timestamp(f["date"].min()) - pd.Timedelta(400, "D")
    ev = np.full((n, 3), np.nan)
    n_ev = 0
    for ent, g in f.groupby("entity", sort=False):
        react = set()
        for c in chains.get(str(ent), {entity_cik(str(ent))}):
            try:
                react.update(v4.reaction_days(subs, c, sessions, start))
            except Exception:  # noqa: BLE001 - a missing submissions file leaves the entity without earnings dates
                continue
        n_ev += bool(react)
        rows = np.array([pos[i] for i in g.index])
        ev[rows] = earnings_features(sorted(react), g["date"].to_numpy(dtype="datetime64[ns]"), sessions)
    out["w_earn_next"], out["w_earn_last"], out["w_earn_days"] = ev[:, 0], ev[:, 1], ev[:, 2]
    return out.drop(columns=["ret_1w"]), {"entities_with_earnings": int(n_ev)}


def weekly_eval_rows(f: pd.DataFrame) -> np.ndarray:
    """True on the last grid session of every ISO week (non-overlapping 5-session labels)."""
    d = pd.DatetimeIndex(f["date"])
    iso = d.isocalendar()
    key = iso["year"].astype(int).to_numpy() * 100 + iso["week"].astype(int).to_numpy()
    last = pd.Series(d.to_numpy(), index=key).groupby(level=0).max()
    return (d.to_numpy() == last.reindex(key).to_numpy()).astype(bool)


def weights_candidate(name: str, weights: dict[str, float], ko: str, en: str) -> cmod.Candidate:
    cmod._FITTERS[name] = lambda c, train, params, scheme, ctx, w=weights: cmod.Fitted(
        c.name, params, scheme, None, list(w), "weights", {"weights": dict(w)}
    )
    return cmod.Candidate(name, "baseline", ko, en, tuned=False)


def candidates() -> list[cmod.Candidate]:
    every = {c.name: c for c in cmod.default_candidates()}
    out = [
        every["mom_12_1"],
        every["factor_ic"],
        weights_candidate("rev_1w", {"x_ret_1w": -1.0}, "규칙: 1주 되돌림", "rule: 1-week reversal"),
        weights_candidate(
            "rev_sec", {"x_w_rev_sec_1w": -1.0}, "규칙: 섹터 대비 1주 되돌림", "rule: sector-relative reversal"
        ),
        *(every[x] for x in ("cat_rank", "ridge", "logit_w", "lgbm_clf", "et")),
    ]
    for name in ("cat_rank", "logit_w", "lgbm_clf"):
        d = copy.copy(every[name])
        d.name, d.diagnostic, d.feature_set = f"{name}_noW", True, "core"
        d.fitter = every[name].fitter or every[name].name
        d.label_ko = every[name].label_ko + " · 진단: 묶음 W 제외"
        out.append(d)
    return out


def write_summary() -> str:
    d = json.loads((OUT / "US_weekly.json").read_text())
    info = d["protocol"]["weekly"]
    cand = {c["name"]: c for c in d["candidates"]}
    sel_name = d["selected"]
    leak = ", ".join(info.get("leak_suspect") or []) or "없음"
    lines = [
        "# US 1주 단위 시험 — h = 5거래일 (docs/MODELING.md 40)",
        "",
        f"- 평가 주 {info.get('eval_weeks')}개 · 묶음 W {len(info['features'])}개 · 결측 누수 의심: {leak}",
        f"- 결측률: {info.get('missing_share')}",
        f"- 선정 `{sel_name}` · 게이트 {'통과' if d['gate']['passed'] else '미달'}",
        "",
        "| 후보 | 외부 AUC [95% CI] | 외부 IC (t) | 홀드아웃 AUC | 순초과/주 외부 (t) | 홀드아웃 순초과/주 | 회전율 |",
        "|---|---|---|---|---|---|---|",
    ]

    nan = float("nan")

    def pct(x):
        return "–" if x is None else f"{x:+.3%}"

    def g(dct, k):
        v = dct.get(k)
        return nan if v is None else v

    for n, c in cand.items():
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        tn, htn = c["outer"].get("top_net") or {}, (c["holdout"] or {}).get("top_net") or {}
        hm = f"{h['mean']:.4f}" if h.get("mean") is not None else "–"
        tt = f" (t {tn['t']:+.2f})" if tn.get("t") is not None else ""
        lines.append(
            f"| {n} | {o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] | {c['outer']['ic']['mean']:+.4f} "
            f"({c['outer']['ic']['t']:+.2f}) | {hm} | {pct(tn.get('mean'))}{tt} | {pct(htn.get('mean'))} | "
            f"{c['outer'].get('turnover') or 0:.2f} |"
        )
    s = cand[sel_name]
    so, sh = s["outer"]["auc_median"], (s["holdout"] or {}).get("auc_median") or {}
    mom = cand["mom_12_1"]["outer"]["auc_median"]
    ok1 = so["lo"] > 0.5 and so["mean"] > mom["mean"] and (sh.get("mean") or 0) >= 0.5
    tn, htn = s["outer"].get("top_net") or {}, (s["holdout"] or {}).get("top_net") or {}
    ok2 = (tn.get("mean") or 0) > 0 and (tn.get("t") or 0) >= 2 and (htn.get("mean") or 0) >= 0
    av = d["protocol"].get("alt_value") or {}
    pair = av.get("cat_rank - cat_rank_noW") or {}
    po, ph = pair.get("outer") or {}, pair.get("holdout") or {}
    ok3 = po.get("lo") is not None and po["lo"] > 0 and (ph.get("mean") is None or ph["mean"] >= 0)
    lines += [
        "",
        "## 판정 (40.2)",
        "",
        f"1. 1주 실력(`{sel_name}`): 외부 {so['mean']:.4f} [{so['lo']:.4f}, {so['hi']:.4f}], 모멘텀 {mom['mean']:.4f}"
        f", 홀드아웃 {g(sh, 'mean'):.4f} → **{'통과' if ok1 else '미달'}**",
        f"2. 돈이 되는지(비용 10bp × 회전율 후): 외부 {pct(tn.get('mean'))}/주 (t {tn.get('t') or nan:+.2f})"
        f", 홀드아웃 {pct(htn.get('mean'))}/주 → **{'통과' if ok2 else '미달'}**",
        f"3. 묶음 W(`cat_rank` − `cat_rank_noW`): 외부 {g(po, 'mean'):+.4f} [{g(po, 'lo'):+.4f}, {g(po, 'hi'):+.4f}]"
        f" · 홀드아웃 {g(ph, 'mean'):+.4f} → **{'기여 있음' if ok3 else '기여 없음'}**",
        "",
        f"**종합: {'1주 운용 후보 — 전진 기록부터' if ok1 and ok2 else '1주 운용 후보 아님'}**",
    ]
    for k, v in av.items():
        if k != "cat_rank - cat_rank_noW":
            lines.append(
                f"- (참고) {k}: 외부 {v['outer']['mean']:+.4f} [{v['outer']['lo']:+.4f}, {v['outer']['hi']:+.4f}]"
            )
    text = "\n".join(lines) + "\n"
    (OUT / "US_weekly_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    if "--summary" in sys.argv:
        print(write_summary())
        return
    settings = load_settings(horizon_days=H)
    cache = DiskCache(settings.cache_path)
    uni, ds = v13s.load_inputs("US", settings, cache)
    f = ds.frame
    base_xs, base_reg = list(ds.xs_cols), list(ds.regime_cols)
    say("block W (short horizon)")
    w, info = block_w(ds, uni, settings, cache)
    raw = []
    for c in w.columns:
        vals = w[c].astype(float)
        if vals.notna().sum() == 0 or vals.nunique() < 2:
            info.setdefault("empty", []).append(c)
            continue
        f[c] = vals.to_numpy()
        f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
        feat.FEATURE_DOCS[c] = W_DOCS.get(c, (c, c))
        raw.append(c)
    f["is_me"] = weekly_eval_rows(f)
    lab = f[f["labelled"] & f["is_me"]]
    info["features"] = raw
    info["eval_weeks"] = int(lab["date"].nunique())
    info["missing_share"] = {c: round(float(lab[c].isna().mean()), 3) for c in raw}
    aucs = {c: v13s.missing_auc(f, c) for c in raw}
    info["missing_auc"] = aucs
    info["leak_suspect"] = sorted(c for c, a in aucs.items() if a is not None and abs(a - 0.5) > v13s.LEAK_AUC)
    say(f"block W: {raw}; {json.dumps(info, ensure_ascii=False, default=str)[:1500]}")
    ds.core_xs_cols, ds.core_regime_cols = base_xs, base_reg
    ds.xs_cols = [*base_xs, *[feat.x_col(c) for c in raw]]
    ds.snapshot = f"{ds.snapshot}|weekly40|{len(raw)}"
    for k, v in WEEKLY.items():
        setattr(sel, k, v)
    mt.PERIODS_PER_YEAR = 52
    mt.block_bootstrap_means.__defaults__ = (13, mt.BOOT_SIMS, 0)
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE,)
    sel.COMPOSITES = (cmod.ENSEMBLE.name,)
    sel.DIAGNOSTIC_PAIRS = tuple((x, f"{x}_noW") for x in ("cat_rank", "logit_w", "lgbm_clf"))
    res = sel.run_selection(ds, settings, cache, candidates=candidates(), threads=8, progress=say)
    res.protocol["weekly"] = {"declared": "docs/MODELING.md 40", "horizon": H, **info, "protocol": WEEKLY}
    (OUT / "US_weekly.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
    try:
        from quantcio.modeling.report import render_markdown

        (OUT / "US_weekly.md").write_text(render_markdown(res, settings), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - the summary below is the result; the full report is a convenience
        say(f"full report skipped: {type(exc).__name__}: {exc}")
    print(write_summary(), flush=True)


if __name__ == "__main__":
    main()
