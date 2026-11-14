"""Markdown + JSON report of a model-selection run: ``reports/modeling/<market>_<as_of>.{md,json}``.

The Markdown is written in the configured report language (``quantcio.i18n.tr``) and always starts with one
Korean and one English summary line. The JSON is ``SelectionResult.to_dict()`` plus the serving summary.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from quantcio.config import Settings
from quantcio.i18n import tr
from quantcio.modeling.features import feature_label
from quantcio.modeling.metrics import PRIMARY, PRIMARY_AUC, PRIMARY_NULL, SECONDARY_AUC, auc_label, metric_label
from quantcio.modeling.selection import BASELINE, SelectionResult


def _f(x: Any, nd: int = 4, signed: bool = False) -> str:
    if x is None:
        return "–"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:  # NaN
        return "–"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def _n(x: Any) -> str:
    """Integer with thousands separators ('–' when unknown)."""
    return "–" if x is None else f"{int(x):,}"


def _pct(x: Any, nd: int = 1) -> str:
    return "–" if x is None else f"{100 * float(x):+.{nd}f}%"


def _ci(s: dict[str, Any] | None, nd: int = 4) -> str:
    if not s or s.get("mean") is None:
        return "–"
    return f"{_f(s['mean'], nd)} [{_f(s.get('lo'), nd)}, {_f(s.get('hi'), nd)}]"


def _ic(s: dict[str, Any] | None) -> str:
    if not s or s.get("mean") is None:
        return "–"
    return f"{_f(s['mean'], 4, True)} (t {_f(s.get('t'), 2, True)})"


def _top(m: dict[str, Any]) -> str:
    s = m.get("top_net") or {}
    if s.get("mean") is None:
        return "–"
    return f"{_pct(s['mean'] * 12)} [{_pct((s.get('lo') or 0) * 12)}, {_pct((s.get('hi') or 0) * 12)}]"


def _primary(res: SelectionResult) -> str:
    return str(res.gate.get("metric") or res.protocol.get("primary_metric") or "auc")


def _null(k: str) -> float:
    return PRIMARY_NULL if k == PRIMARY else 0.5


def _gate_ci_ok(g: dict[str, Any]) -> bool:
    return bool(g.get("lo95_above_null", g.get("auc_lo95_above_half")))


def _gate_word(g: dict[str, Any], lang: str) -> str:
    """Gate verdict incl. the post-hoc caveat (a post-hoc pass is served x0.35 until confirmed on untouched data)."""
    if not g.get("passed"):
        return "failed" if lang == "en" else "미달"
    if g.get("post_hoc"):
        return (
            "passed (post-hoc metric: x0.35 until confirmed on untouched data)"
            if lang == "en"
            else "통과(사후 지표: 미열람 자료로 확인 전까지 ×0.35)"
        )
    return "passed" if lang == "en" else "통과"


def summary_lines(res: SelectionResult) -> tuple[str, str]:
    """(Korean, English) one-line summary of the run (every AUC says which label it is computed against)."""
    c = res.candidate(res.selected)
    o, h = c["outer"], c["holdout"] or {}
    g = res.gate
    mm = res.market_model
    k = _primary(res)
    sec = PRIMARY_AUC if k != PRIMARY_AUC else SECONDARY_AUC
    sec_ko = f", {auc_label(sec, 'ko')} {_ci(o.get(sec))}" if sec else ""
    sec_en = f", {auc_label(sec, 'en')} {_ci(o.get(sec))}" if sec else ""
    ko = (
        f"{res.market} 종목 선택 모델: {res.selected}({c['label_ko']}) 선택 — 표본 외 {metric_label(k, 'ko')} "
        f"{_ci(o.get(k))}{sec_ko}, 순위 IC {_ic(o['ic'])}, 홀드아웃 {metric_label(k, 'ko')} {_ci(h.get(k))}; "
        "스킬 게이트 "
        f"{_gate_word(g, 'ko')}(모멘텀 초과 {'예' if g['beats_momentum'] else '아니오'}, "
        f"{metric_label(k, 'ko')} 하한>{_null(k):g} {'예' if _gate_ci_ok(g) else '아니오'}); 시장 방향 모델 "
        f"{'사용' if mm.get('use_model') else '미사용(기준률)'}."
    )
    en = (
        f"{res.market} stock-selection model: {res.selected} ({c['label_en']}) selected - OOS {metric_label(k, 'en')} "
        f"{_ci(o.get(k))}{sec_en}, rank IC {_ic(o['ic'])}, holdout {metric_label(k, 'en')} {_ci(h.get(k))}; skill gate "
        f"{_gate_word(g, 'en')} (beats momentum {g['beats_momentum']}, {metric_label(k, 'en')} lower bound > "
        f"{_null(k):g} {_gate_ci_ok(g)}); market-direction model "
        f"{'used' if mm.get('use_model') else 'not used (base rate)'}."
    )
    return ko, en


def _bo_section(res: SelectionResult, t: Callable[[str, str], str]) -> list[str]:
    """docs/MODELING.md 82: per Bayesian-optimised candidate and context - trials, the 1-SE pick vs the best inner
    score, and the in-sample fit of the chosen configuration (the overfitting gap against the outer OOS row)."""
    rows = [c for c in res.candidates if any((cfg or {}).get("bo") for cfg in (c.get("configs") or {}).values())]
    if not rows:
        return []

    def short(params: dict[str, Any] | None) -> str:
        return ", ".join(f"{k}={_f(v, 3) if isinstance(v, float) else v}" for k, v in (params or {}).items())

    out = [
        "",
        t(
            "### 베이지안 최적화 후보 — 과적합 점검 (docs/MODELING.md 82)",
            "### Bayesian-optimised candidates - overfitting check (docs/MODELING.md 82)",
        ),
        "",
        t(
            "| 후보 | 맥락 | 시도 | 1-SE 안 | 내부 최고(월) | 선택 내부(월) | 표본 내 AUC | 표본 내 1차(월) |"
            " 선택 구성 |",
            "| candidate | context | trials | within 1 SE | best inner (mo) | picked inner (mo) | in-sample AUC | "
            "in-sample primary (mo) | chosen |",
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in rows:
        for ctx, cfg in (c.get("configs") or {}).items():
            bo = (cfg or {}).get("bo") or {}
            if not bo:
                continue
            out.append(
                f"| {c['name']} | {ctx} | {bo.get('trials')} | {bo.get('within_1se')} | {_f(bo.get('best_inner'))} | "
                f"{_f(bo.get('picked_inner'))} | {_f(cfg.get('train_auc'), 3)} | {_f(cfg.get('train_primary'))} | "
                f"{cfg.get('scheme')}: {short(cfg.get('params'))} |"
            )
        o = c["outer"]
        out.append(
            f"| {c['name']} | {t('외부 표본 외', 'outer OOS')} | | | | {_f((o.get(PRIMARY) or {}).get('mean'))} | "
            f"{_f((o.get(PRIMARY_AUC) or {}).get('mean'), 3)} | | |"
        )
    return out


def render_markdown(res: SelectionResult, settings: Settings, serving: dict[str, Any] | None = None) -> str:
    t = lambda ko, en: tr(settings, ko, en)  # noqa: E731
    lang = settings.report.language
    ko, en = summary_lines(res)
    p = res.protocol
    k = _primary(res)
    ak, ae = metric_label(k, "ko"), metric_label(k, "en")
    sec2 = PRIMARY_AUC if k != PRIMARY_AUC else SECONDARY_AUC  # v4: the (v3 primary) AUC is the secondary column
    sk, se = auc_label(sec2, "ko"), auc_label(sec2, "en")
    lines: list[str] = []
    lines.append(
        t(
            f"# 종목 선택 모델 선정 — {res.market} ({res.as_of})",
            f"# Stock-selection model selection - {res.market} ({res.as_of})",
        )
    )
    lines += ["", f"- {ko}", f"- {en}", ""]
    ds = res.dataset
    cov = ds.get("coverage", {})
    uni = ds.get("universe", {})
    lines.append(t("## 데이터·프로토콜", "## Data and protocol"))
    lines.append(
        t(
            f"- 유니버스: {uni.get('description')} · 엔티티 {uni.get('entities')} · 행 {_n(cov.get('rows'))} "
            f"(레이블 {_n(cov.get('labelled_rows'))}, 월말 레이블 {_n(cov.get('labelled_me_rows'))} / "
            f"{cov.get('labelled_me_dates')}개 월말) · 기간 {cov.get('first_date')}~{cov.get('last_date')}",
            f"- Universe: {uni.get('description_en')} · {uni.get('entities')} entities · {_n(cov.get('rows'))} rows "
            f"({_n(cov.get('labelled_rows'))} labelled, {_n(cov.get('labelled_me_rows'))} month-end over "
            f"{cov.get('labelled_me_dates')} month-ends) · {cov.get('first_date')}~{cov.get('last_date')}",
        )
    )
    lines.append(
        t(
            f"- 목표: h={settings.horizon_days}거래일 베타 조정 초과수익(단순 수익률, 데이터셋 v5) "
            f"y_e = (e^r − 1) − β(e^m − 1). 주 지표(프로토콜 "
            f"v{p.get('protocol_version', 1)}): {ak}"
            + (" = 1{y_e > 날짜 중앙값}의 날짜 내 AUC" if k == PRIMARY_AUC else "")
            + f"; 보조: {sk}, 순위 IC, 상위 5분위−유니버스 초과(순, 거래비용 {ds.get('cost_bps')}bp×회전율). "
            "CI = 날짜 3개월 블록 부트스트랩 95%.",
            f"- Target: h={settings.horizon_days}-session beta-adjusted excess return (simple returns, dataset v5) "
            f"y_e = (e^r - 1) - beta*(e^m - 1). Primary "
            f"(protocol v{p.get('protocol_version', 1)}): {ae}"
            + (" = within-date AUC of 1{y_e > date median}" if k == PRIMARY_AUC else "")
            + f"; secondary: {se}, rank IC, top-quintile minus universe (net of {ds.get('cost_bps')}bp x turnover). "
            "CI = 95% 3-month date-block bootstrap.",
        )
    )
    if p.get("post_hoc"):
        lines.append(
            t(
                "- **사후 결정 경고**: 주 지표(프로토콜 v4: 단순 수익률 라벨의 상위 20% 순초과, docs/MODELING.md 79)는 "
                "이전 결과를 **본 뒤에** 정했습니다(post_hoc = True). 24개월 홀드아웃도 이미 봤습니다. 따라서 "
                "게이트를 통과해도 미열람 자료(전진 홀드아웃)에서 확인되기 전까지는 ×0.35로 서빙합니다.",
                "- **Post-hoc warning**: the primary metric (protocol v4: top-quintile net excess on the simple label, "
                "docs/MODELING.md 79) was chosen AFTER earlier results were seen (post_hoc = True), and the 24-month "
                "holdout was already seen. A gate pass is therefore served x0.35 until confirmed on untouched data "
                "(a forward holdout).",
            )
        )
    lines.append(
        t(
            f"- 외부 워크포워드 {p['outer_folds']}개 폴드(검증 {p['outer_test'][0]}~{p['outer_test'][1]}, 12개월 블록, "
            f"최소 학습 36개월), 퍼지+엠바고 {p['embargo_sessions']}거래일, 내부 CV 연간 블록 2개(같은 퍼지). "
            f"최종 홀드아웃 {p['holdout'][0]}~{p['holdout'][1]}({p['holdout_months']}개월, 선택에 미사용).",
            f"- Outer walk-forward: {p['outer_folds']} folds (test {p['outer_test'][0]}~{p['outer_test'][1]}, 12-month "
            f"blocks, >= 36 months of training), purge + {p['embargo_sessions']}-session embargo, inner CV "
            f"{p['inner']}. "
            f"Final holdout {p['holdout'][0]}~{p['holdout'][1]} ({p['holdout_months']} months, never used for "
            "choices).",
        )
    )
    lines.append(
        t(
            f"- 선택 규칙: 홀드아웃 이전 외부 표본 외 평균 {ak} 최대, 차이 < {p['rule']['tie_tolerance']}이면 "
            f"순위 IC, 다음 상위 5분위 순초과로 결정. 서빙 게이트: 모멘텀 기준보다 {ak}가 높고 그 95% 하한 > "
            f"{_null(k):g}.",
            f"- Selection rule: highest mean pre-holdout outer-OOS {ae}; differences < "
            f"{p['rule']['tie_tolerance']} are ties broken by rank IC, then top-quintile net excess. Serving gate: "
            f"{ae} above the momentum baseline AND its 95% lower bound > {_null(k):g}.",
        )
    )
    lines.append(
        t(
            f"- 다중검정: 후보 {p['candidates']}개(폴드마다 내부 CV로 튜닝한 구성 "
            f"{p['configurations_per_fold']}개)가 같은 홀드아웃 이전 날짜로 경쟁하므로 승자의 외부 주 지표는 "
            "낙관적으로 치우칩니다(winner's curse). "
            + (
                "24개월 홀드아웃은 이미 봤고 주 지표도 사후 결정이라 둘 다 미열람 확인이 아닙니다 — "
                "확인은 미열람 자료(전진 홀드아웃)가 합니다. "
                if p.get("post_hoc")
                else "홀드아웃이 선정 모델의 비편향 점검입니다. "
            )
            + "표의 CI는 개별 구간(다중성 보정 없음)입니다.",
            f"- {p['multiple_testing']} Table CIs are per-candidate (no multiplicity adjustment).",
        )
    )
    lines.append("")
    lines.append(t("## 후보별 결과", "## Candidates"))
    lines.append(
        t(
            f"| 후보 | 유형 | 외부 OOS {ak} [95% CI] | 외부 {sk} | 순위 IC (t) | 상위5분위 순초과(연) [CI] | "
            f"주 지표−모멘텀 (t) | 홀드아웃 {ak} [CI] | 홀드아웃 {sk} | 홀드아웃 IC (t) | 홀드아웃 상위5분위(연) |",
            f"| candidate | family | outer OOS {ae} [95% CI] | outer {se} | rank IC (t) | top-quintile net (ann.) "
            f"[CI] | primary - momentum (t) | holdout {ae} [CI] | holdout {se} | holdout IC (t) | holdout "
            "top-quintile (ann.) |",
        )
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in res.candidates:
        o, h = c["outer"], c["holdout"] or {}
        d = c.get("vs_momentum_auc") or {}
        diff = "–" if not d or d.get("mean") is None else f"{_f(d['mean'], 4, True)} ({_f(d.get('t'), 2, True)})"
        name = f"**{c['name']}**" if c["name"] == res.selected else c["name"]
        if c.get("diagnostic"):
            name += t(" (진단)", " (diagnostic)")
        label = c["label_en"] if lang == "en" else c["label_ko"]
        sec_o = _f((o.get(sec2) or {}).get("mean"))
        sec_h = _f((h.get(sec2) or {}).get("mean"))
        lines.append(
            f"| {name} | {label} | {_ci(o.get(k))} | {sec_o} | {_ic(o['ic'])} | {_top(o)} | {diff} | "
            f"{_ci(h.get(k))} | {sec_h} | {_ic(h.get('ic'))} | {_top(h) if h else '–'} |"
        )
    lines.append("")
    lines.append(
        t(
            "(홀드아웃 열은 선택에 쓰지 않았음 — 참고용. 단 v1에서 이미 본 구간이라 미열람 확인이 아님)",
            "(holdout columns were not used for selection - but the window was already seen in v1: not an untouched "
            "confirmation)",
        )
    )
    lines.append("")
    av = (res.protocol or {}).get("alt_value") or {}
    alt_feats = (res.protocol or {}).get("alt_features") or []
    if av or alt_feats:
        lines.append(
            t(
                "### 대안 데이터·매크로 노출의 기여 (진단 — 선정에 쓰지 않음)",
                "### What the alternative data and macro exposures add (diagnostic - not used for selection)",
            )
        )
        if alt_feats:
            lines.append(
                t(
                    f"- v3 특징 {len(alt_feats)}개: {', '.join(alt_feats)}",
                    f"- {len(alt_feats)} v3 features: {', '.join(alt_feats)}",
                )
            )
        for pair, v in av.items():
            o = v.get("outer") or {}
            h = v.get("holdout") or {}
            lines.append(
                t(
                    f"- {pair}: 외부 {ak} 날짜별 차이 {_f(o.get('mean'), 4, True)} "
                    f"[{_f(o.get('lo'), 4, True)}, {_f(o.get('hi'), 4, True)}] (t {_f(o.get('t'), 2, True)}, "
                    f"{o.get('n')}개 날짜) · 홀드아웃 {_f(h.get('mean'), 4, True)} (t {_f(h.get('t'), 2, True)})",
                    f"- {pair}: outer {ae} per-date difference {_f(o.get('mean'), 4, True)} "
                    f"[{_f(o.get('lo'), 4, True)}, {_f(o.get('hi'), 4, True)}] (t {_f(o.get('t'), 2, True)}, "
                    f"{o.get('n')} dates) · holdout {_f(h.get('mean'), 4, True)} (t {_f(h.get('t'), 2, True)})",
                )
            )
        cov = (res.protocol or {}).get("alt_coverage") or {}
        for src, c in cov.items():
            if isinstance(c, dict) and ("covered" in c or "mapped" in c):
                lines.append(
                    t(
                        f"- 수집 범위 {src}: " + ", ".join(f"{k} {v}" for k, v in c.items() if not isinstance(v, list)),
                        f"- coverage {src}: " + ", ".join(f"{k} {v}" for k, v in c.items() if not isinstance(v, list)),
                    )
                )
        lines.append("")
    g = res.gate
    gv = _f(g.get("value", g.get("auc")), 5)
    gb = _f(g.get("baseline", g.get("baseline_auc")), 5)
    gl = _f(g.get("lo", g.get("auc_lo")), 5)
    lines.append(t("## 선택 모델과 서빙 게이트", "## Selected model and serving gate"))
    lines.append(
        t(
            f"- 선택: **{res.selected}** · 외부 {ak} {gv} vs 모멘텀 {gb} · 95% 하한 {gl} → 게이트 "
            f"**{_gate_word(g, 'ko')}**" + ("" if g["passed"] else " (초과수익 기대치 ×0.35 축소, 스킬 미확인 표시)"),
            f"- Selected: **{res.selected}** · outer {ae} {gv} vs momentum {gb} · 95% lower bound {gl} "
            f"-> gate **{_gate_word(g, 'en')}**"
            + ("" if g["passed"] else " (expected excess shrunk x0.35, flagged unskilled)"),
        )
    )
    cal = res.calibration
    lines.append(
        t(
            f"- 보정: 날짜별 기울기(점수 z → 초과수익) {_f(cal.get('slope'), 5)} (t {_f(cal.get('t'), 2)}, "
            f"{cal.get('n_dates')}개 날짜) → 축소 t²/(1+t²) = {_f(cal.get('shrink'), 3)}. μ_excess = 축소 × 기울기 × "
            "z.",
            f"- Calibration: per-date slope (score z -> excess) {_f(cal.get('slope'), 5)} (t {_f(cal.get('t'), 2)}, "
            f"{cal.get('n_dates')} dates) -> shrink t^2/(1+t^2) = {_f(cal.get('shrink'), 3)}. mu_excess = shrink x "
            "slope x z.",
        )
    )
    sel = res.candidate(res.selected)
    cfgs = sel.get("configs") or {}
    shown = []
    for ctx, cfg in cfgs.items():
        if not cfg:
            continue
        if "members" in cfg:
            shown.append(f"{ctx}: {'+'.join(cfg['members'])}")
        else:
            shown.append(f"{ctx}: {cfg.get('params')} / {cfg.get('scheme')}")
    if shown:
        lines.append(t("- 폴드별 선택 구성: ", "- Chosen configuration per context: ") + "; ".join(shown))
    schemes: dict[str, int] = {}
    for c in res.candidates:
        for cfg in (c.get("configs") or {}).values():
            if cfg and "scheme" in cfg:
                schemes[cfg["scheme"]] = schemes.get(cfg["scheme"], 0) + 1
    if schemes:
        lines.append(
            t(
                "- 내부 CV가 고른 학습 가중 방식(모든 후보·폴드): "
                f"{', '.join(f'{k} {v}' for k, v in sorted(schemes.items()))}",
                f"- Training-weight schemes chosen by inner CV (all candidates x contexts): "
                f"{', '.join(f'{k} {v}' for k, v in sorted(schemes.items()))}",
            )
        )
    lines += _bo_section(res, t)
    if res.importances:
        lines.append("")
        lines.append(t("### 특징 중요도(최종 적합, 정규화)", "### Feature importance (final fit, normalised)"))
        for k, v in list(res.importances.items())[:15]:
            lines.append(f"- {k} ({feature_label(k, lang)}): {v:.3f}")
    lines.append("")
    mm = res.market_model
    lines.append(t("## 시장 방향 요소(별도 모델)", "## Market-direction component (separate model)"))
    for key, name in (("pre_holdout", t("홀드아웃 이전", "pre-holdout")), ("holdout", t("홀드아웃", "holdout"))):
        m = mm.get(key) or {}
        if not m.get("n"):
            continue
        lg = m.get("logloss_gain") or {}
        lines.append(
            t(
                f"- {name} {m['n']}개월({m.get('first')}~{m.get('last')}): 상승 비율 {_f(m.get('up_rate'), 3)}, "
                "로지스틱 AUC "
                f"{_f(m.get('auc_logit'), 3)} [{_f(m.get('auc_logit_lo'), 3)}, {_f(m.get('auc_logit_hi'), 3)}], "
                f"logloss 개선(기준률 대비) {_f(lg.get('mean'), 4, True)} [{_f(lg.get('lo'), 4, True)}, "
                f"{_f(lg.get('hi'), 4, True)}]",
                f"- {name} {m['n']} months ({m.get('first')}~{m.get('last')}): up rate {_f(m.get('up_rate'), 3)}, "
                "logistic "
                f"AUC {_f(m.get('auc_logit'), 3)} [{_f(m.get('auc_logit_lo'), 3)}, {_f(m.get('auc_logit_hi'), 3)}], "
                f"log-loss gain vs base rate {_f(lg.get('mean'), 4, True)} [{_f(lg.get('lo'), 4, True)}, "
                f"{_f(lg.get('hi'), 4, True)}]",
            )
        )
    lines.append(
        t(
            f"- 판정: {'모델 사용' if mm.get('use_model') else '기준률 사용(스킬 미확인)'} · 서빙 P(시장 상승) "
            f"{_f(mm.get('served_p_up'), 3)} (전체 기준률 {_f(mm.get('base_rate_all'), 3)})",
            f"- Verdict: {'model used' if mm.get('use_model') else 'base rate (no demonstrated skill)'} · served "
            f"P(market up) {_f(mm.get('served_p_up'), 3)} (overall base rate {_f(mm.get('base_rate_all'), 3)})",
        )
    )
    lines.append("")
    ev = res.events
    lines.append(
        t(
            "## 사건 구간 강건성(표본 외 전체: 외부 폴드 + 홀드아웃)",
            "## Event-window robustness (all OOS: outer + holdout)",
        )
    )
    lines.append(
        t(
            f"- 레이블 창이 사건 구간과 겹치는 날짜 {ev.get('n_event_dates')}개 vs 평시 {ev.get('n_calm_dates')}개",
            f"- {ev.get('n_event_dates')} dates whose label window overlaps an event vs {ev.get('n_calm_dates')} calm "
            "dates",
        )
    )
    ek = str(ev.get("auc_metric") or "auc")
    lines.append(
        t(
            f"| 후보 | 사건 {metric_label(ek, 'ko')} | 평시 {metric_label(ek, 'ko')} | 사건 IC | 평시 IC |",
            f"| candidate | event {metric_label(ek, 'en')} | calm {metric_label(ek, 'en')} | event IC | calm IC |",
        )
    )
    lines.append("|---|---|---|---|---|")
    for n, v in (ev.get("by_candidate") or {}).items():
        lines.append(
            f"| {n} | {_ci(v['event_auc'])} | {_ci(v['calm_auc'])} | {_ic(v['event_ic'])} | {_ic(v['calm_ic'])} |"
        )
    rows = ev.get("by_event") or []
    if rows:
        lines.append("")
        lines.append(
            t(
                f"| 사건 | 날짜 수 | {res.selected} {metric_label(ek, 'ko')} | {res.selected} IC | {BASELINE} "
                f"{metric_label(ek, 'ko')} | {BASELINE} IC |",
                f"| event | dates | {res.selected} {metric_label(ek, 'en')} | {res.selected} IC | {BASELINE} "
                f"{metric_label(ek, 'en')} | {BASELINE} IC |",
            )
        )
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            label = r["label_en"] if lang == "en" else r["label_ko"]
            lines.append(
                f"| {label} | {r['n_dates']} | {_f(r.get(f'auc_{res.selected}'), 3)} | "
                f"{_f(r.get(f'ic_{res.selected}'), 3, True)} | "
                f"{_f(r.get(f'auc_{BASELINE}'), 3)} | {_f(r.get(f'ic_{BASELINE}'), 3, True)} |"
            )
    lines.append("")
    ref = res.reference
    if ref.get("available"):
        m = ref["metrics"]
        lines.append(
            t(
                f"- 참고: 팩터 백테스트 에이전트의 캐시된 워크포워드 점수({ref.get('method')}) — 같은 지표로 {ak} "
                f"{_ci(m.get(k))}, {sk} {_ci(m.get(SECONDARY_AUC))}, "
                f"IC {_ic(m['ic'])} ({m.get('first_date')}~{m.get('last_date')}, 선택 대상 아님)",
                f"- Reference: the factor-backtest agent's cached walk-forward scores ({ref.get('method')}) - {ae} "
                f"{_ci(m.get(k))}, {se} {_ci(m.get(SECONDARY_AUC))}, IC {_ic(m['ic'])} "
                f"({m.get('first_date')}~{m.get('last_date')}, not a candidate)",
            )
        )
    if serving:
        lines.append("")
        lines.append(t("## 서빙 산출물", "## Serving artifact"))
        lines.append(
            t(
                f"- 기준일 {serving.get('as_of')} · 종목 {serving.get('n')}개 · 초과수익 배수 "
                f"{_f(serving.get('multiplier'), 3)} · "
                f"μ_excess 범위 {_pct(serving.get('mu_min'), 2)} ~ {_pct(serving.get('mu_max'), 2)} (h일 로그)",
                f"- As of {serving.get('as_of')} · {serving.get('n')} names · excess multiplier "
                f"{_f(serving.get('multiplier'), 3)} · mu_excess range {_pct(serving.get('mu_min'), 2)} ~ "
                f"{_pct(serving.get('mu_max'), 2)} (h-day log)",
            )
        )
        if serving.get("gate_passed") and serving.get("post_hoc"):
            lines.append(
                t(
                    f"- 게이트 통과(사후 라벨): 저장된 배수는 확인 전 값입니다. 점수화 시점에 "
                    f"`quantcio.holdout.xsh_confirmation`이 미열람 XSH 확인을 돌려줄 때만 배수 "
                    f"{_f(serving.get('multiplier_if_confirmed'), 3)}로 서빙합니다.",
                    f"- Gate passed (post-hoc label): the stored multiplier is the unconfirmed one. It is served at "
                    f"{_f(serving.get('multiplier_if_confirmed'), 3)} only when `quantcio.holdout.xsh_confirmation` "
                    "returns a confirmation on untouched XSH names at scoring time.",
                )
            )
    notes = [n.get("en") if lang == "en" else n.get("ko") for n in (ds.get("notes") or [])]
    if notes:
        lines.append("")
        lines.append(t("## 데이터 메모", "## Data notes"))
        lines += [f"- {n}" for n in notes if n]
    lines.append("")
    return "\n".join(lines)


def write_report(
    res: SelectionResult, settings: Settings, out_dir: str | Path | None = None, serving: dict[str, Any] | None = None
) -> tuple[Path, Path]:
    """Write ``<out_dir or reports/modeling>/<market>_<as_of>.md|json``; returns (md, json) paths."""
    root = Path(out_dir) if out_dir is not None else Path(settings.report.output_dir) / "modeling"
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{res.market}_{res.as_of}"
    md = root / f"{stem}.md"
    js = root / f"{stem}.json"
    md.write_text(render_markdown(res, settings, serving), encoding="utf-8")
    payload = res.to_dict()
    payload["summary_ko"], payload["summary_en"] = summary_lines(res)
    if serving:
        payload["serving"] = serving
    js.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return md, js


__all__ = ["render_markdown", "summary_lines", "write_report"]
