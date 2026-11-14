"""New data -> feature selection -> model selection (docs/MODELING.md 53, pre-declared 2026-10-13). Research only.

1. The 52 pool (fs_select_study.build_pool) and its selection run (checkpoints of 52 reused when the data match).
2. + blocks A K M P G Y Q N (scripts/v19_data.py); features whose missingness alone has a within-date AUC beyond
   0.5 +- 0.02 are dropped BEFORE the run (53.2).
3. The same candidates (fs_* with per-fit feature selection, the served configuration, baselines, ensembles).
4. Best model = the selection rule's pick on the 53 pool; contribution = per-date AUC of every fs_* model on the 53
   pool minus the 52 pool, averaged over the 9 models (outer lower bound > 0 and holdout mean >= 0 to claim it).

    .venv/bin/python scripts/v19_study.py --market KR

Writes reports/modeling/<M>_v19.json and reports/modeling/<M>_v19_summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_select_study as fss  # noqa: E402
import v13_study as v13s  # noqa: E402
import v19_data as v  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
BLOCK_KO = {
    "A": "감사",
    "K": "파산·부실",
    "M": "인수합병",
    "P": "공시",
    "G": "투자 거장",
    "Y": "업력",
    "Q": "퇴사",
    "N": "ETF",
}
_f = fss._f


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_blocks(market: str, ds, settings, cache) -> tuple[list[str], list[str], dict]:
    f = ds.frame
    df, info = v.block(market, f, ds, settings, cache)
    raw, reg = [], []
    for c in df.columns:
        vals = df[c].astype(float)
        if vals.notna().sum() == 0 or vals.nunique() < 2:
            info.setdefault("empty", []).append(c)
            continue
        f[c] = vals.to_numpy()
        if c in v.REGIME:
            reg.append(c)
        else:
            f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
            raw.append(c)
        feat.FEATURE_DOCS[c] = v.DOCS.get(c, (c, c))
    lab = f[f["labelled"] & f["is_me"]]
    info["missing_share"] = {c: round(float(lab[c].isna().mean()), 3) for c in [*raw, *reg]}
    info["known_from"] = {
        c: str(lab.loc[lab[c].notna(), "date"].min().date()) if lab[c].notna().any() else None for c in [*raw, *reg]
    }
    aucs = {c: v13s.missing_auc(f, c) for c in raw}
    info["missing_auc"] = aucs
    info["leak_suspect"] = sorted(c for c, a in aucs.items() if a is not None and abs(a - 0.5) > v13s.LEAK_AUC)
    return raw, reg, info


def _store(settings, market: str, snapshot: str) -> None:
    fss.FS_STORE = settings.cache_path / "fs52" / f"{market}_{DiskCache.key('fs52', snapshot)}.json"
    fss.FS_STORE.parent.mkdir(parents=True, exist_ok=True)
    if fss.FS_STORE.exists():
        fss.FS_LOG.update(json.loads(fss.FS_STORE.read_text()))


def _pd(res, name: str, part: str) -> pd.Series:
    frame = res.oos if part == "outer" else res.holdout_preds
    return mt.per_date_primary(frame, name) if frame is not None and name in frame else pd.Series(dtype=float)


def contribution(res53, res52) -> dict:
    names = [f"fs_{m}" for m in fss.FS_MODELS]
    out: dict = {"per_model": {}}
    for part in ("outer", "holdout"):
        diffs = []
        for n in names:
            a, b = _pd(res53, n, part), _pd(res52, n, part)
            common = a.index.intersection(b.index)
            if len(common):
                diffs.append((a.reindex(common) - b.reindex(common)).rename(n))
                out["per_model"].setdefault(n, {})[part] = mt.paired_difference(a, b)
        avg = pd.concat(diffs, axis=1).mean(axis=1) if diffs else pd.Series(dtype=float)
        out[f"mean_{part}"] = mt.summarize(avg.to_numpy(), 0.0)
    o, h = out["mean_outer"], out["mean_holdout"]
    out["helps"] = bool(o.get("lo") is not None and o["lo"] > 0 and h.get("mean") is not None and h["mean"] >= 0)
    return out


def write_summary(market: str, r: dict) -> str:
    d53, d52 = r["selection53"], r["selection52"]
    c53 = {c["name"]: c for c in d53["candidates"]}
    c52 = {c["name"]: c for c in d52["candidates"]}
    core = f"{fss.SERVED[market]}_core"
    lines = [
        f"# {market} 새 자료(감사·파산·인수합병·공시·거장·업력·퇴사·ETF) → 특징 선택 → 모델 선정 (docs/MODELING.md 53)",
        "",
        f"- 풀: 52절 {r['pool52']} → 53절 {r['pool53']}",
        f"- 결측 누출 의심으로 실행 전에 뺀 특징: {', '.join(r['blocks']['leak_suspect']) or '없음'}",
        f"- 비어 있어 빠진 특징: {', '.join(r['blocks'].get('empty', [])) or '없음'}",
        "",
        "| 후보 | 53절 외부 AUC [95% CI] | 52절 외부 AUC | 53절 홀드아웃 | 52절 홀드아웃 | 53절 외부 IC (t) |",
        "|---|---|---|---|---|---|",
    ]
    for n, c in c53.items():
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        p = c52.get(n) or {}
        po = (p.get("outer") or {}).get("auc_median") or {}
        ph = ((p.get("holdout") or {}).get("auc_median")) or {}
        ic = c["outer"]["ic"]
        star = " **(선정)**" if n == d53["selected"] else ""
        lines.append(
            f"| {n}{star} | {_f(o['mean'], '.4f')} [{_f(o['lo'], '.4f')}, {_f(o['hi'], '.4f')}] | "
            f"{_f(po.get('mean'), '.4f')} | {_f(h.get('mean'), '.4f')} | {_f(ph.get('mean'), '.4f')} | "
            f"{_f(ic['mean'])} ({_f(ic['t'], '+.2f')}) |"
        )
    g = d53["gate"]
    lines += [
        "",
        f"**가장 괜찮은 모델(53절 풀): `{d53['selected']}`** · 게이트 {'통과' if g['passed'] else '미달'} · "
        f"52절 선정은 `{d52['selected']}`",
        "",
    ]
    for k, x in r["vs"].items():
        o, h = x["outer"], x.get("holdout") or {}
        lines.append(
            f"- {k}: 외부 {_f(o['mean'])} [{_f(o['lo'])}, {_f(o['hi'])}] (t {_f(o['t'], '+.2f')}) · "
            f"홀드아웃 {_f(h.get('mean'))} [{_f(h.get('lo'))}, {_f(h.get('hi'))}]"
        )
    cb = r["contribution"]
    o, h = cb["mean_outer"], cb["mean_holdout"]
    lines += [
        "",
        "## 새 자료의 기여 (같은 fs 모델 9개: 53절 풀 − 52절 풀, 날짜별 AUC 차이의 모델 평균)",
        "",
        f"- 외부 {_f(o['mean'])} [{_f(o['lo'])}, {_f(o['hi'])}] (t {_f(o['t'], '+.2f')}) · 홀드아웃 {_f(h['mean'])} "
        f"[{_f(h['lo'])}, {_f(h['hi'])}] → **{'도움' if cb['helps'] else '도움 없음'}**",
        "",
        "| 모델 | 외부 차이 [95% CI] | 홀드아웃 차이 |",
        "|---|---|---|",
    ]
    for n, x in cb["per_model"].items():
        o, h = x.get("outer") or {}, x.get("holdout") or {}
        lines.append(f"| {n} | {_f(o.get('mean'))} [{_f(o.get('lo'))}, {_f(o.get('hi'))}] | {_f(h.get('mean'))} |")
    lines += [
        "",
        "## 새 특징이 학습마다 선택된 비율 (상위 25)",
        "",
        "| 특징 | 묶음 | 선택 비율 | IC t(홀드아웃 전) |",
        "|---|---|---|---|",
    ]
    for c, share in r["new_selected"][:25]:
        lines.append(f"| {c} | {r['feature_block'].get(c, '')} | {share:.0%} | {_f(r['new_ic_t'].get(c), '+.2f')} |")
    lines += ["", "## 선정 모델의 신호 포트폴리오 (중간 규모, 47절 비용)", ""]
    lines += ["| 포트폴리오 | 연 수익 | MDD | 기준 대비 연 초과 | 월 초과 t |", "|---|---|---|---|---|"]
    for k, s in r.get("portfolio", {}).items():
        lines.append(
            f"| {k} | {_f(s['cagr'], '+.1%')} | {_f(s['mdd'], '+.1%')} | {_f(s['excess_ann'], '+.1%')} | "
            f"{_f(s['excess_t'], '+.2f')} |"
        )
    lines += ["", f"(서빙 구성 `{core}`)"]
    text = "\n".join(lines) + "\n"
    (OUT / f"{market}_v19_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    a = ap.parse_args()
    m = a.market
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE, cmod.ENSEMBLE_W)
    sel.COMPOSITES = (cmod.ENSEMBLE.name, cmod.ENSEMBLE_W.name)
    sel.DIAGNOSTIC_PAIRS = ()
    uni, ds, info52 = fss.build_pool(m, settings, cache)
    xs52, reg52, snap52 = list(ds.xs_cols), list(ds.regime_cols), ds.snapshot
    say(f"52 pool: {info52['pool']}")
    _store(settings, m, snap52)
    res52 = sel.run_selection(ds, settings, cache, candidates=fss.candidates(m), threads=8, progress=say)
    raw, reg, binfo = add_blocks(m, ds, settings, cache)
    keep = [c for c in raw if c not in set(binfo["leak_suspect"])]
    new_x = [feat.x_col(c) for c in keep]
    ds.xs_cols, ds.regime_cols = [*xs52, *new_x], [*reg52, *reg]
    ds.snapshot = f"{snap52}|v19|{len(new_x)}|{len(reg)}"
    say(f"53 pool: +{len(new_x)} features, +{len(reg)} regime (dropped as leak suspects: {binfo['leak_suspect']})")
    _store(settings, m, ds.snapshot)
    res53 = sel.run_selection(ds, settings, cache, candidates=fss.candidates(m), threads=8, progress=say)
    d53, d52 = res53.to_dict(), res52.to_dict()
    core = f"{fss.SERVED[m]}_core"
    best, best52 = d53["selected"], d52["selected"]
    vs = {}
    for n, ref, rres in ((best, core, res53), (best, best52, res52)):
        if n == ref and rres is res53:
            continue
        vs[f"{n}(53) − {ref}({'53' if rres is res53 else '52'})"] = {
            "outer": mt.paired_difference(_pd(res53, n, "outer"), _pd(rres, ref, "outer")),
            "holdout": mt.paired_difference(_pd(res53, n, "holdout"), _pd(rres, ref, "holdout")),
        }
    n53 = len(ds.xs_cols)
    fits = [x for k, x in fss.FS_LOG.items() if k.split("|")[2] == str(n53)]
    cnt = Counter(c for x in fits for c in x["xs"])
    new_sel = sorted(((c, cnt.get(c, 0) / max(len(fits), 1)) for c in new_x), key=lambda t: -t[1])
    f = ds.frame
    hold0 = pd.Timestamp(d53["protocol"]["holdout"][0])
    me = f[f["is_me"] & f["labelled"] & (f["date"] < hold0)]
    ic_t = fss._ic_t(me, new_x).round(2).to_dict()
    block_of = {feat.x_col(c): b for b, cols in v.BLOCK_COLS[m].items() for c in cols}
    r = {
        "market": m,
        "pool52": info52["pool"],
        "pool53": {"xs": n53, "regime": len(ds.regime_cols), "new_xs": len(new_x), "new_regime": len(reg)},
        "blocks": binfo,
        "selection52": d52,
        "selection53": d53,
        "vs": vs,
        "contribution": contribution(res53, res52),
        "new_selected": new_sel,
        "new_ic_t": ic_t,
        "feature_block": {c: f"{b} {BLOCK_KO[b]}" for c, b in block_of.items()},
        "fs_fits_53": len(fits),
    }
    try:
        r["portfolio"] = fss.portfolio(m, res53, uni, ds, best)
    except Exception as exc:  # noqa: BLE001 - the selection result stands without the portfolio table
        say(f"portfolio skipped: {type(exc).__name__}: {exc}")
    text = json.dumps(r, ensure_ascii=False, indent=1, default=str)
    (OUT / f"{m}_v19.json").write_text(text)
    print(write_summary(m, r), flush=True)


if __name__ == "__main__":
    main()
