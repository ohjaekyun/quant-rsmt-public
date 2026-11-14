"""Korean markdown + JSON report of a PIT factor backtest (``quantcio backtest``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quantcio.backtest.candidates import CANDIDATES, LABELS
from quantcio.backtest.engine import BacktestResult
from quantcio.backtest.factors import FACTOR_LABELS, FACTORS

MARKET_LABELS = {"KR": "한국", "US": "미국"}


def _pct(x: Any, digits: int = 1) -> str:
    return "-" if x is None else f"{100 * float(x):+.{digits}f}%"


def _num(x: Any, digits: int = 2) -> str:
    return "-" if x is None else f"{float(x):.{digits}f}"


def _sig(r: dict[str, Any]) -> str:
    if not r.get("bh_significant"):
        return "-"
    return "유의(역방향)" if r.get("direction_ok") is False else "유의"


def factor_rows_md(result: dict[str, Any]) -> list[str]:
    lines = [
        "| 팩터 | 변형 | IC 평균 | IC t | IC>0 | 롱숏 연수익(비용 후) | 롱숏 샤프(비용 후) | 롱숏 MDD "
        "| 롱온리 초과(비용 후) | 회전율(롱) | BH q=0.10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in result["factor_table"]:
        ls = r.get("ls_net") or {}
        lo = r.get("lo_net") or {}
        lines.append(
            f"| {r['label']} | {r['variant_label']} | {_num(r.get('ic_mean'), 3)} | {_num(r.get('ic_t'))} | "
            f"{_pct(r.get('ic_hit'), 0).lstrip('+')} | {_pct(ls.get('ann_return'))} | {_num(ls.get('sharpe'))} | "
            f"{_pct(ls.get('mdd'))} | {_pct(lo.get('ann_return'))} | {_pct(r.get('turnover_long'), 0).lstrip('+')} | "
            f"{_sig(r)} |"
        )
    return lines


def _ic_cell(x: dict[str, Any] | None) -> str:
    x = x or {}
    return f"{_num(x.get('ic_mean'), 3)} ({_num(x.get('ic_t'))})"


def comparison_rows_md(model: dict[str, Any]) -> list[str]:
    """Candidate comparison table (pre-holdout months; holdout and full OOS columns) of ``model['selection']``."""
    sel = model.get("selection") or {}
    rows = sel.get("rows") or {}
    served = model.get("method")
    lines = [
        "| 후보 | 설명 | 적합 모수 | IC 평균 | IC t | IC>0 | 롱숏 연수익(비용 후) | 롱숏 샤프 "
        "| 상위 분위 초과(비용 후) | 회전율(롱) | 현행 대비 ΔIC (t) | 홀드아웃 IC (t) | 전체 OOS IC t |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CANDIDATES:
        r = rows.get(name)
        if r is None:
            continue
        label = f"**{name}** (선택)" if name == served else name
        if not r.get("available"):
            lines.append(f"| {label} | {LABELS.get(name, name)} | - | 사용 불가: {r.get('reason', '-')} |" + " |" * 9)
            continue
        pre = r.get("pre") or {}
        ls = pre.get("ls_net") or {}
        lo = pre.get("lo_net") or {}
        vs = r.get("pre_vs_reference")
        vs_txt = "기준" if vs is None else f"{_num(vs.get('mean_diff'), 3)} ({_num(vs.get('t'))})"
        lines.append(
            f"| {label} | {LABELS.get(name, name)} | {r.get('n_params')} | {_num(pre.get('ic_mean'), 3)} | "
            f"{_num(pre.get('ic_t'))} | {_pct(pre.get('ic_hit'), 0).lstrip('+')} | {_pct(ls.get('ann_return'))} | "
            f"{_num(ls.get('sharpe'))} | {_pct(lo.get('ann_return'))} | "
            f"{_pct(pre.get('turnover_long'), 0).lstrip('+')} | {vs_txt} | {_ic_cell(r.get('holdout'))} | "
            f"{_num((r.get('full_all_months') or {}).get('ic_t'))} |"
        )
    return lines


def comparison_section(model: dict[str, Any]) -> list[str]:
    sel = model.get("selection") or {}
    if not sel:
        return ["- 후보 비교 결과가 없습니다."]
    rows = sel.get("rows") or {}
    pre_p = sel.get("pre_period") or ["-", "-"]
    hold_p = sel.get("holdout_period") or ["-", "-"]
    lines = [
        f"- 사전 선언 후보 {len(sel.get('candidates') or [])}개(선택 규칙 버전 {sel.get('version')}, "
        "docs/BACKTEST_MACRO.md 3.1). 모든 후보는 t 이전 월만으로 가중치를 추정합니다.",
        f"- 공통 OOS {sel.get('common_months')}개월 = 선택 구간 {sel.get('pre_months')}개월({pre_p[0]}~{pre_p[1]}) "
        f"+ 홀드아웃 {sel.get('holdout_months')}개월({hold_p[0]}~{hold_p[1]}, 선택에 사용 안 함).",
        "- 규칙: 선택 구간의 월평균 OOS 순위 IC 최대 → 차이 < "
        f"{(sel.get('rule') or {}).get('tie_ic')}이면 IC t 큰 쪽 → t 차이 < {(sel.get('rule') or {}).get('tie_t')}이면 "
        "적합 모수가 적은 쪽.",
        "- 규칙 적용: " + " → ".join(sel.get("trace") or []),
        "",
        "표의 IC·롱숏·상위 분위 초과·회전율은 **선택 구간(홀드아웃 전)** 값입니다. 괄호는 t입니다.",
        "",
    ]
    lines += comparison_rows_md(model)
    lines += [
        "",
        "- **승자의 저주**: 여러 후보 중 선택 구간 최댓값을 골랐으므로 선택된 후보의 선택 구간 IC는 위로 치우칩니다. "
        "편향 없는 확인값은 홀드아웃 IC입니다. 스킬 게이트의 전체 OOS IC t도 선택 구간을 포함하므로 약간 낙관적입니다.",
        "- 후보 목록은 전체 기간 단일 팩터 결과를 본 뒤 설계했으므로, 홀드아웃은 선택에는 쓰이지 않았지만 연구자 "
        "자유도 면에서 완전히 깨끗하지는 않습니다.",
    ]
    ridge = (rows.get("ridge") or {}).get("diagnostics") or {}
    if ridge.get("alpha_counts"):
        lines.append(
            "- ridge: 날짜별로 고른 α(λ = α·N/3) 횟수 "
            + ", ".join(f"{a}: {n}" for a, n in ridge["alpha_counts"].items())
            + f", 최신 α {ridge.get('latest_alpha')}"
        )
    tree = (rows.get("rsmt_tree") or {}).get("diagnostics") or {}
    if tree:
        cnt = tree.get("split_variable_counts") or {}
        lines.append(
            f"- rsmt_tree: 분할 변수 {', '.join(tree.get('split_variables') or []) or '없음'}; 모델 날짜 "
            f"{tree.get('dates')}개 중 분할이 생긴 날짜 {tree.get('dates_with_split')}개"
            + (f" (변수별 {', '.join(f'{k} {v}' for k, v in cnt.items())})" if cnt else "")
            + f"; 최신 트리 잎 {tree.get('latest_leaves')}개."
        )
        lt = tree.get("latest_tree") or {}
        if lt.get("split"):
            sp = lt["split"]
            lines.append(
                f"  - 최신 트리 루트 분할: {sp['var']} ≤ {float(sp['threshold']):.4f} "
                f"(왼쪽 {lt['left']['n']}개월 / 오른쪽 {lt['right']['n']}개월, Simes p {lt.get('p_simes')})"
            )
        elif lt.get("tests"):
            lines.append(
                "  - 최신 트리 루트 검정 p: "
                + ", ".join(f"{v} {t.get('p')}" for v, t in lt["tests"].items())
                + f" → Simes p {lt.get('p_simes')} (분할 없음)"
            )
    mc = (rows.get("macro_conditional") or {}).get("diagnostics") or {}
    if mc:
        lines.append(
            f"- macro_conditional: 모델 날짜의 레짐 위험선호 {mc.get('months_risk_on')} / 위험회피 "
            f"{mc.get('months_risk_off')} / 미상 {mc.get('months_unknown')}개월."
        )
    return lines


def render_markdown(res: BacktestResult) -> str:
    d = res.to_dict()
    m = d["model"]
    oos = m.get("oos") or {}
    mkt = MARKET_LABELS.get(d["market"], d["market"])
    lines = [
        f"# {mkt} PIT 팩터 백테스트 · 멀티팩터 모델 ({d['period']['start']} ~ {d['period']['end']})",
        "",
        f"- 생성: {d['created_at']} · 설정 {d['settings_digest']} · 데이터 스냅샷 {d['data_snapshot'][:10]}",
        f"- 유니버스: {d['universe']['description']}, 월말 {d['n_months']}회, "
        f"평균 {d['universe']['avg_members']:.0f}종목, 편입 이력 {d['universe']['entities']}개 엔티티"
        f"(상장폐지 {d['universe'].get('delisted_entities')}개 포함)",
        f"- 거래비용 {d['cost_bps']:.0f}bp(왕복, 회전율 100%당) · 5분위 동일가중 · 보유 t+1 세션 종가 → 다음 월말",
        "",
        "## 1. 요약",
        "",
    ]
    table = d["factor_table"]
    sig = [r for r in table if r.get("bh_significant")]
    good = [r for r in sig if r.get("direction_ok") is not False]
    bad = [r for r in sig if r.get("direction_ok") is False]
    if good:
        lines.append(
            "- BH(q=0.10) 보정 후 유의하고 방향이 맞는(또는 사전 방향 없는) 팩터: "
            + ", ".join(f"{r['label']}({r['variant_label']}, IC t {_num(r['ic_t'])})" for r in good)
        )
    if bad:
        lines.append(
            "- BH 보정 후 유의하지만 **기대와 반대 방향**(반전)인 팩터: "
            + ", ".join(f"{r['label']}({r['variant_label']}, IC t {_num(r['ic_t'])})" for r in bad)
            + " — 부호 제약 후보(`ic_weighted` 계열·`macro_conditional`·`rsmt_tree`)는 이 팩터의 적합 가중치를"
            " 0으로 두고 동일가중 축소분을 사전 방향으로 남기며, 부호를 학습하는 후보"
            "(`ic_weighted_raw`·`fama_macbeth`·`ridge`)는 음(−)의 가중치를 줄 수 있습니다."
        )
    if not sig:
        lines.append(
            "- **BH(q=0.10) 보정 후 유의한 팩터가 없습니다.** 단일 팩터의 예측력은 통계적으로 확인되지 않았습니다."
        )
    skill = "통과" if m.get("skill_ok") else "미달"
    sel = m.get("selection") or {}
    srow = (sel.get("rows") or {}).get(m["method"]) or {}
    if sel:
        how = "규칙 선택" if sel.get("mode") == "auto" else "설정으로 고정"
        lines.append(
            f"- 모델 후보 {len(sel.get('candidates') or [])}개 비교 → **{m['method']}** ({how}; "
            f"선택 구간 IC {_ic_cell(srow.get('pre'))}, 홀드아웃 {sel.get('holdout_months')}개월 IC "
            f"{_ic_cell(srow.get('holdout'))}). 표는 3절."
        )
    lines += [
        f"- 워크포워드 모델({m['method']}): 표본 외 {m.get('oos_months', 0)}개월, "
        f"IC 평균 {_num(oos.get('ic_mean'), 3)}, IC t {_num(oos.get('ic_t'))} "
        f"→ 스킬 게이트(t ≥ {m.get('skill_min_ic_t')}) **{skill}**"
        + ("" if m.get("skill_ok") else " — 리포트 필러 기여 ×0.35"),
        f"- 모델 롱숏(비용 후) 연 {_pct((oos.get('ls_net') or {}).get('ann_return'))}, 샤프 "
        f"{_num((oos.get('ls_net') or {}).get('sharpe'))}; 롱온리 초과(상위 분위 − 유니버스, 비용 후) 연 "
        f"{_pct((oos.get('lo_net') or {}).get('ann_return'))}, 샤프 {_num((oos.get('lo_net') or {}).get('sharpe'))}",
    ]
    if m.get("macro"):
        mc = m["macro"]
        lines.append(
            f"- 매크로 조건부 후보: 전체 OOS IC t {_num(mc['conditional']['ic_t'])} vs 현행 무조건 "
            f"{_num(mc['unconditional']['ic_t'])} → {'선택됨' if mc.get('used') else '선택되지 않음'}"
        )
    else:
        lines.append("- 매크로 조건부 후보: 레짐 이력이 주어지지 않아 비교하지 않았습니다.")
    lines += ["", "## 2. 팩터별 결과", ""]
    lines += factor_rows_md(d)
    lines += [
        "",
        "- IC: 월별 스피어만 순위상관(신호 vs 다음 달 수익률). t = 평균/(표준편차/√월수).",
        "- 롱숏 = 최상위 분위 − 최하위 분위, 롱온리 초과 = 최상위 분위 − 유니버스 동일가중. 연수익은 기하 연환산.",
        "- 거래량 추세는 사전 방향이 없어 부호를 백테스트 IC로 판단합니다. 섹터 모멘텀은 전체 순위로만 평가합니다.",
        "",
        "### 분위별 월평균 수익률 (섹터 중립, 모멘텀은 전체 순위)",
        "",
        "| 팩터 | Q1(하위) | Q2 | Q3 | Q4 | Q5(상위) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in table:
        if r["variant"] != "sector_neutral" and r["factor"] != "sector_mom_3m":
            continue
        qm = r.get("quantile_mean_monthly") or {}
        lines.append(f"| {r['label']} | " + " | ".join(_pct(qm.get(f"q{k}"), 2) for k in range(1, 6)) + " |")
    lines += ["", "## 3. 모델 비교 (사전 선언 후보)", ""]
    lines += comparison_section(m)
    lines += ["", "## 4. 선택된 워크포워드 멀티팩터 모델", ""]
    lines.append(
        f"- 방법 {m['method']}: {LABELS.get(m['method'], m['method'])}. 가중치는 t 이전 월만으로 추정, |w| 합 1, "
        f"최소 {m.get('min_months')}개월 이력."
    )
    lines.append(f"- 첫 모델 월: {m.get('first_model_date')}, 최신 가중치({m.get('latest_date')} 이전 자료로 추정):")
    lines += ["", "| 팩터 | 가중치 |", "|---|---:|"]
    for f in FACTORS:
        lines.append(f"| {FACTOR_LABELS[f]} | {_num((m.get('latest_weights') or {}).get(f), 3)} |")
    ls = oos.get("ls_net") or {}
    lo = oos.get("lo_net") or {}
    lines += [
        "",
        "| 표본 외 지표 | 값 |",
        "|---|---:|",
        f"| 월수 | {oos.get('n_months')} |",
        f"| IC 평균 / t / IC>0 | {_num(oos.get('ic_mean'), 3)} / {_num(oos.get('ic_t'))} / "
        f"{_pct(oos.get('ic_hit'), 0).lstrip('+')} |",
        f"| 롱숏 연수익·변동성·샤프·MDD (비용 후) | {_pct(ls.get('ann_return'))} · "
        f"{_pct(ls.get('ann_vol')).lstrip('+')} · {_num(ls.get('sharpe'))} · {_pct(ls.get('mdd'))} |",
        f"| 롱온리 초과 연수익·샤프·MDD (비용 후) | {_pct(lo.get('ann_return'))} · {_num(lo.get('sharpe'))} · "
        f"{_pct(lo.get('mdd'))} |",
        f"| 월 승률(롱숏, 비용 후) | {_pct(ls.get('win_rate'), 0).lstrip('+')} |",
        f"| 평균 회전율(롱/숏, 편도) | {_pct(oos.get('turnover_long'), 0).lstrip('+')} / "
        f"{_pct(oos.get('turnover_short'), 0).lstrip('+')} |",
    ]
    if m.get("macro"):
        mc = m["macro"]
        lines += [
            "",
            "### 매크로 조건부 비교",
            "",
            f"- 비교 {mc['months_compared']}개월(위험선호 {mc['months_risk_on']}, 위험회피 {mc['months_risk_off']}, "
            f"미상 {mc['months_unknown']}), 레짐별 IC 평균은 전체 평균 쪽으로 {mc['shrink_months']:.0f}개월 가중 축소.",
            f"- 무조건 IC 평균 {_num(mc['unconditional']['ic_mean'], 3)} (t {_num(mc['unconditional']['ic_t'])}) vs "
            f"조건부 {_num(mc['conditional']['ic_mean'], 3)} (t {_num(mc['conditional']['ic_t'])}) → "
            + ("조건부 후보가 선택됨" if mc.get("used") else "조건부 후보는 선택되지 않음"),
        ]
    live = d.get("live") or {}
    lines += [
        "",
        "## 5. 서빙 스냅샷",
        "",
        f"- 기준일 {live.get('as_of')} (최근 리밸런싱 {live.get('rebalance_date')} 구성원 {live.get('n')}종목), "
        "종목별 점수·백분위는 리포트 모듈 2에 표시됩니다.",
        "",
        "## 6. 커버리지",
        "",
    ]
    cov = d["coverage"]
    lines.append(f"- 회원-월 {cov.get('member_months')}개, 날짜 {cov.get('dates')}개, 엔티티 {cov.get('entities')}개")
    lines.append(f"- 가격 상태: {cov.get('price_status')}")
    lines.append(f"- 다음 달 수익률 상태: {cov.get('fwd_status')}")
    lines.append(
        "- 팩터 가용 회원-월: "
        + ", ".join(f"{FACTOR_LABELS[f]} {cov.get('factor_available', {}).get(f)}" for f in FACTORS)
    )
    sec = cov.get("sectors") or {}
    lines.append(f"- 섹터 출처: {sec.get('sources')} · 섹터별 회원-월: {sec.get('sectors')}")
    lines += [
        "",
        "| 연도 | 회원-월 | 수익률 확보 | 보유 중 폐지 | 정지 후 체결 | 가격 결측 | E/P 가용 | B/P 가용 | 섹터 미분류 "
        "| 소형 섹터 병합 |",
    ]
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for y, v in (cov.get("by_year") or {}).items():
        lines.append(
            f"| {y} | {v['member_months']} | {v['fwd_ret_ok']} | {v['delisted_exits']} | {v.get('halted_exits', 0)} | "
            f"{v['price_missing']} | "
            f"{v['value_ep_available']} | {v['value_bp_available']} | {v['sector_unclassified']} | "
            f"{v['sector_merged_into_other']} |"
        )
    lines += ["", "## 7. 참고·한계", ""]
    lines += [f"- {n}" for n in d["notes"]]
    lines += [
        "- 과거 성과는 미래 수익을 보장하지 않습니다. 월별 IC와 분위 수익률은 거래 가능성(유동성·공매도 제약)을 "
        "단순화한 값입니다.",
        "",
    ]
    return "\n".join(lines)


def write_backtest_report(result: BacktestResult, out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``backtest_<market>_<as_of>.md`` (Korean) and ``.json`` into ``out_dir``; returns both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"backtest_{result.market}_{result.as_of}"
    md_path = out / f"{stem}.md"
    json_path = out / f"{stem}.json"
    md_path.write_text(render_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    return md_path, json_path
