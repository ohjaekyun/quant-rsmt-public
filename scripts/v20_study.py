"""Full pool -> feature selection -> model selection (docs/MODELING.md 54, pre-declared 2026-10-13). Research only.

The 53 pool exactly as scripts/v19_study.py builds it (52 and 53 selection runs come back from their checkpoints)
+ the blocks tested separately before: KR API-key block (23) and Naver search attention (27); US v10 (28), v11 (29),
v12 (30) and Wikipedia attention (27). New features whose missingness alone separates the label (|AUC - 0.5| > 0.02)
are dropped before the run. Best model = the selection rule's pick on this pool (the final answer per market).

    .venv/bin/python scripts/v20_study.py --market KR

Writes reports/modeling/<M>_v20.json and reports/modeling/<M>_v20_summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention_study as att_s  # noqa: E402
import fs_select_study as fss  # noqa: E402
import full_opt_study as fo  # noqa: E402
import v13_study as v13s  # noqa: E402
import v19_study as v19s  # noqa: E402

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
_f = fss._f
_pd = v19s._pd


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _attention(market: str, ds, settings, cache) -> list[str]:
    f = ds.frame
    raw = (att_s.us_attention if market == "US" else att_s.kr_attention)(ds, settings, cache, False)
    for c in raw:
        f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
        feat.FEATURE_DOCS[c] = att_s.DOCS[c]
    return [feat.x_col(c) for c in raw]


def extra_blocks(market: str, ds, uni, settings, cache) -> tuple[dict[str, list[str]], list[str], dict]:
    """block -> its x_ columns, the regime columns, info (each block exactly as its own study adds it)."""
    blocks: dict[str, list[str]] = {}
    reg: list[str] = []
    info: dict = {}
    if market == "KR":
        say("extra: API-key block (ECOS / KOSIS / KIPRIS)")
        kx, kreg, kinfo = fo.kr_blocks(ds, settings, cache)
        blocks["keyed"], info["keyed"] = kx, kinfo
        reg += kreg
        say("extra: Naver search attention")
        blocks["attention"] = _attention(market, ds, settings, cache)
    else:
        import us_v10_study as v10s
        import us_v11_study as v11s
        import us_v12_study as v12s

        say("extra: v10 (analysts / short interest / off-exchange / F-score / peers / put-call)")
        mx, mreg, minfo = v10s.add_block_m(ds, settings, cache, universe=uni)
        blocks["v10"], info["v10"] = mx, minfo
        reg += mreg
        say("extra: v11 (filing times / 10-K text / distance to default / pageview spikes / domains)")
        nx, ninfo = v11s.add_block_n(ds, settings, cache)
        blocks["v11"], info["v11"] = nx, ninfo
        say("extra: v12 (power / real estate / data centres)")
        px, preg, pinfo = v12s.add_block_p(ds, settings, cache)
        blocks["v12"], info["v12"] = px, pinfo
        reg += preg
        say("extra: Wikipedia attention")
        blocks["attention"] = _attention(market, ds, settings, cache)
    return blocks, reg, info


def leak_screen(ds, xs: list[str]) -> tuple[dict[str, float | None], list[str]]:
    """Missing-indicator AUC of every new feature (its raw column when present) and the suspects."""
    f = ds.frame
    aucs = {}
    for x in xs:
        raw = x[2:] if x.startswith("x_") and x[2:] in f else x
        aucs[x] = v13s.missing_auc(f, raw)
    sus = sorted(x for x, a in aucs.items() if a is not None and abs(a - 0.5) > v13s.LEAK_AUC)
    return aucs, sus


def write_summary(market: str, r: dict) -> str:
    d54, d53, d52 = r["selection54"], r["selection53"], r["selection52"]
    c54 = {c["name"]: c for c in d54["candidates"]}
    c53 = {c["name"]: c for c in d53["candidates"]}
    lines = [
        f"# {market} 전체 풀 → 특징 선택 → 모델 선정 (docs/MODELING.md 54)",
        "",
        f"- 풀: 53절 {r['pool53']} → 54절 {r['pool54']}",
        "- 추가 묶음: "
        + ", ".join(f"{b} {len(x)}개" for b, x in r["extra_blocks"].items())
        + f", 국면 +{len(r['extra_regime'])}",
        f"- 결측 누출 의심으로 실행 전에 뺀 특징: {', '.join(r['leak_suspect']) or '없음'}",
        "",
        "| 후보 | 54절 외부 AUC [95% CI] | 53절 외부 AUC | 54절 홀드아웃 | 53절 홀드아웃 | 54절 외부 IC (t) |",
        "|---|---|---|---|---|---|",
    ]
    for n, c in c54.items():
        o, h = c["outer"]["auc_median"], (c["holdout"] or {}).get("auc_median") or {}
        p = c53.get(n) or {}
        po = (p.get("outer") or {}).get("auc_median") or {}
        ph = ((p.get("holdout") or {}).get("auc_median")) or {}
        ic = c["outer"]["ic"]
        star = " **(선정)**" if n == d54["selected"] else ""
        lines.append(
            f"| {n}{star} | {_f(o['mean'], '.4f')} [{_f(o['lo'], '.4f')}, {_f(o['hi'], '.4f')}] | "
            f"{_f(po.get('mean'), '.4f')} | {_f(h.get('mean'), '.4f')} | {_f(ph.get('mean'), '.4f')} | "
            f"{_f(ic['mean'])} ({_f(ic['t'], '+.2f')}) |"
        )
    g = d54["gate"]
    lines += [
        "",
        f"**가장 괜찮은 모델(전체 풀): `{d54['selected']}`** · 게이트 {'통과' if g['passed'] else '미달'} · "
        f"53절 선정 `{d53['selected']}`, 52절 선정 `{d52['selected']}`",
        "",
    ]
    for k, x in r["vs"].items():
        o, h = x["outer"], x.get("holdout") or {}
        lines.append(
            f"- {k}: 외부 {_f(o['mean'])} [{_f(o['lo'])}, {_f(o['hi'])}] (t {_f(o['t'], '+.2f')}) · "
            f"홀드아웃 {_f(h.get('mean'))} [{_f(h.get('lo'))}, {_f(h.get('hi'))}]"
        )
    for tag, cb in (("54 − 53", r["contribution_54_53"]), ("54 − 52", r["contribution_54_52"])):
        o, h = cb["mean_outer"], cb["mean_holdout"]
        lines += [
            "",
            f"- 같은 fs 모델 9개, {tag} 풀 차이(모델 평균): 외부 {_f(o['mean'])} [{_f(o['lo'])}, {_f(o['hi'])}] "
            f"(t {_f(o['t'], '+.2f')}) · 홀드아웃 {_f(h['mean'])} → **{'도움' if cb['helps'] else '도움 없음'}**",
        ]
    lines += ["", "## 추가 특징이 학습마다 선택된 비율 (상위 20)", "", "| 특징 | 묶음 | 선택 비율 |", "|---|---|---|"]
    for c, share in r["new_selected"][:20]:
        lines.append(f"| {c} | {r['feature_block'].get(c, '')} | {share:.0%} |")
    lines += ["", "## 선정 모델의 신호 포트폴리오 (중간 규모, 47절 비용)", ""]
    lines += ["| 포트폴리오 | 연 수익 | MDD | 기준 대비 연 초과 | 월 초과 t |", "|---|---|---|---|---|"]
    for k, s in r.get("portfolio", {}).items():
        lines.append(
            f"| {k} | {_f(s['cagr'], '+.1%')} | {_f(s['mdd'], '+.1%')} | {_f(s['excess_ann'], '+.1%')} | "
            f"{_f(s['excess_t'], '+.2f')} |"
        )
    text = "\n".join(lines) + "\n"
    (OUT / f"{market}_v20_summary.md").write_text(text, encoding="utf-8")
    return text


def patch_selection() -> None:
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE, cmod.ENSEMBLE_W)
    sel.COMPOSITES = (cmod.ENSEMBLE.name, cmod.ENSEMBLE_W.name)
    sel.DIAGNOSTIC_PAIRS = ()


def run_through_54(m: str, settings, cache) -> dict:
    """The 52, 53 and 54 pools and their selection runs (52 / 53 exactly as v19_study.main: checkpoints)."""
    patch_selection()
    cands = fss.candidates(m)
    uni, ds, info52 = fss.build_pool(m, settings, cache)
    xs52, reg52, snap52 = list(ds.xs_cols), list(ds.regime_cols), ds.snapshot
    v19s._store(settings, m, snap52)
    res52 = sel.run_selection(ds, settings, cache, candidates=cands, threads=8, progress=say)
    raw, reg, binfo = v19s.add_blocks(m, ds, settings, cache)
    keep = [c for c in raw if c not in set(binfo["leak_suspect"])]
    new_x = [feat.x_col(c) for c in keep]
    ds.xs_cols, ds.regime_cols = [*xs52, *new_x], [*reg52, *reg]
    ds.snapshot = f"{snap52}|v19|{len(new_x)}|{len(reg)}"
    xs53, reg53, snap53 = list(ds.xs_cols), list(ds.regime_cols), ds.snapshot
    v19s._store(settings, m, snap53)
    res53 = sel.run_selection(ds, settings, cache, candidates=cands, threads=8, progress=say)
    # ---- 54: + the separately tested blocks
    blocks, xreg, xinfo = extra_blocks(m, ds, uni, settings, cache)
    add = [x for b in blocks.values() for x in b if x not in set(xs53)]
    add = list(dict.fromkeys(x for x in add if x in ds.frame and ds.frame[x].notna().any()))
    aucs, sus = leak_screen(ds, add)
    add = [x for x in add if x not in set(sus)]
    xreg = [c for c in dict.fromkeys(xreg) if c in ds.frame and c not in set(reg53) and ds.frame[c].notna().any()]
    ds.xs_cols, ds.regime_cols = [*xs53, *add], [*reg53, *xreg]
    ds.snapshot = f"{snap53}|v20|{len(add)}|{len(xreg)}"
    say(f"54 pool: +{len(add)} features, +{len(xreg)} regime (dropped as leak suspects: {sus})")
    v19s._store(settings, m, ds.snapshot)
    res54 = sel.run_selection(ds, settings, cache, candidates=cands, threads=8, progress=say)
    return {
        "uni": uni,
        "ds": ds,
        "res52": res52,
        "res53": res53,
        "res54": res54,
        "xs53": xs53,
        "reg53": reg53,
        "add": add,
        "xreg": xreg,
        "blocks": blocks,
        "xinfo": xinfo,
        "aucs": aucs,
        "sus": sus,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    m = ap.parse_args().market
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    st = run_through_54(m, settings, cache)
    uni, ds, res52, res53, res54 = st["uni"], st["ds"], st["res52"], st["res53"], st["res54"]
    xs53, reg53, add, xreg = st["xs53"], st["reg53"], st["add"], st["xreg"]
    blocks, xinfo, aucs, sus = st["blocks"], st["xinfo"], st["aucs"], st["sus"]
    d54, d53, d52 = res54.to_dict(), res53.to_dict(), res52.to_dict()
    core = f"{fss.SERVED[m]}_core"
    best = d54["selected"]
    vs = {}
    for ref, rres, tag in ((core, res54, "54"), (d53["selected"], res53, "53"), (d52["selected"], res52, "52")):
        if ref == best and tag == "54":
            continue
        vs[f"{best}(54) − {ref}({tag})"] = {
            "outer": mt.paired_difference(_pd(res54, best, "outer"), _pd(rres, ref, "outer")),
            "holdout": mt.paired_difference(_pd(res54, best, "holdout"), _pd(rres, ref, "holdout")),
        }
    n54 = len(ds.xs_cols)
    fits = [x for k, x in fss.FS_LOG.items() if k.split("|")[2] == str(n54)]
    cnt = Counter(c for x in fits for c in x["xs"])
    new_sel = sorted(((c, cnt.get(c, 0) / max(len(fits), 1)) for c in add), key=lambda t: -t[1])
    r = {
        "market": m,
        "pool53": {"xs": len(xs53), "regime": len(reg53)},
        "pool54": {"xs": n54, "regime": len(ds.regime_cols), "new_xs": len(add), "new_regime": len(xreg)},
        "extra_blocks": blocks,
        "extra_regime": xreg,
        "extra_info": xinfo,
        "missing_auc": aucs,
        "leak_suspect": sus,
        "selection52": d52,
        "selection53": d53,
        "selection54": d54,
        "vs": vs,
        "contribution_54_53": v19s.contribution(res54, res53),
        "contribution_54_52": v19s.contribution(res54, res52),
        "new_selected": new_sel,
        "feature_block": {x: b for b, xs in blocks.items() for x in xs},
        "fs_fits_54": len(fits),
    }
    try:
        r["portfolio"] = fss.portfolio(m, res54, uni, ds, best)
    except Exception as exc:  # noqa: BLE001 - the selection result stands without the portfolio table
        say(f"portfolio skipped: {type(exc).__name__}: {exc}")
    (OUT / f"{m}_v20.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    print(write_summary(m, r), flush=True)


if __name__ == "__main__":
    main()
