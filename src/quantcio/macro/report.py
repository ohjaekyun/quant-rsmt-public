"""Korean markdown + JSON report of the current macro regimes and the macro backtest."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from quantcio.macro.backtest import MDD_MIN_GAIN, SHARPE_MIN_GAIN
from quantcio.macro.data import KST
from quantcio.macro.regime import INDICATORS_BY_KEY, LABELS, REGION_LABELS, STALE_SESSIONS
from quantcio.models import MacroIndicator, MacroRegime

_RETURN_UNITS = {"원", "달러", "지수", "비율"}


def _clean(obj: Any) -> Any:
    """JSON-safe copy: NaN / inf -> None, dates -> ISO strings."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def _num(v: float | None, fmt: str) -> str:
    return "—" if v is None or not math.isfinite(v) else format(v, fmt)


def fmt_value(ind: MacroIndicator) -> str:
    v = ind.value
    if v is None:
        return "—"
    if ind.unit == "%":
        return f"{v:+.1f}%" if ind.key.endswith("ma200") else f"{v:.2f}%"
    if ind.unit == "%p":
        return f"{v:+.2f}%p"
    if ind.unit == "원":
        return f"{v:,.1f}원"
    if ind.unit == "달러":
        return f"${v:,.2f}"
    if ind.unit == "비율":
        return f"{v:.3f}"
    return f"{v:,.2f}"


def fmt_change(ind: MacroIndicator) -> str:
    c = ind.change_3m
    if c is None:
        return "—"
    if ind.unit in _RETURN_UNITS:
        return f"{c * 100:+.1f}%"
    if ind.unit in ("%", "%p"):
        return f"{c:+.2f}%p"
    return f"{c:+.2f}"


def _pct(v: float | None, digits: int = 1) -> str:
    return "—" if v is None else f"{v * 100:+.{digits}f}%"


def _regime_table(regimes: list[MacroRegime]) -> list[str]:
    lines = ["| 지역 | 점수 | 판정 | 노출 배수 | 기준일 | 요약 |", "|---|---:|---|---:|---|---|"]
    for r in regimes:
        lines.append(
            f"| {REGION_LABELS.get(r.region, r.region)} | {_num(r.score, '+.2f')} | {r.label} | "
            f"{r.exposure_mult:.2f} | {r.as_of} | {r.summary} |"
        )
    return lines


def _indicator_tables(regimes: list[MacroRegime]) -> list[str]:
    lines: list[str] = []
    for r in regimes:
        lines += ["", f"### {REGION_LABELS.get(r.region, r.region)}", ""]
        lines += ["| 지표 | 값 | 3M 변화 | z (+위험선호) | 부호 | 기준일 | 출처 |", "|---|---:|---:|---:|---|---|---|"]
        for ind in r.indicators:
            spec = INDICATORS_BY_KEY.get(ind.key)
            sign = "—"
            if spec is not None:
                sign = ("+" if spec.sign > 0 else "−") + (" (단방향)" if spec.one_sided else "")
            z = _num(ind.z, "+.2f") if ind.z is not None else ("지연" if ind.key in r.stale else "—")
            lines.append(
                f"| {ind.label} | {fmt_value(ind)} | {fmt_change(ind)} | {z} | {sign} | "
                f"{ind.as_of or '—'} | {ind.source or '—'} |"
            )
        if r.stale:
            lines.append("")
            lines.append(f"- {STALE_SESSIONS}거래일 넘게 갱신되지 않아 점수에서 제외: {', '.join(r.stale)}")
    return lines


def _cond_rows(market: str, region_key: str, cond: dict[str, Any]) -> list[str]:
    rows = []
    region_name = REGION_LABELS.get(region_key, region_key)
    for lab in LABELS:
        st = cond["labels"].get(lab, {})
        if not st or not st.get("n"):
            continue
        rows.append(
            f"| {market} | {region_name} | {lab} | {st['n']} | {_pct(st.get('mean'), 2)} | "
            f"{_num(st.get('t'), '+.2f')} | {_num((st.get('hit_rate') or 0) * 100, '.0f')}% |"
        )
    a = cond["all"]
    rows.append(
        f"| {market} | {region_name} | 전체 | {a['n']} | {_pct(a.get('mean'), 2)} | "
        f"{_num(a.get('t'), '+.2f')} | {_num((a.get('hit_rate') or 0) * 100, '.0f')}% |"
    )
    d = cond["on_minus_off"]
    rows.append(
        f"| {market} | {region_name} | 위험선호−위험회피 | {d['n_a']}/{d['n_b']} | {_pct(d.get('diff'), 2)} | "
        f"{_num(d.get('t'), '+.2f')} (Welch) | — |"
    )
    return rows


def _backtest_section(bt: dict[str, Any]) -> list[str]:
    if not bt:
        return ["", "## 3. 백테스트", "", "백테스트 결과 없음."]
    s = bt.get("settings", {})
    lines = ["", "## 3. 백테스트", ""]
    lines.append(
        f"- 기간: {bt.get('window_start')} 이후 월말 신호(최근 {s.get('lookback_years')}년), "
        f"z 창 {s.get('z_window_years')}년, 임계값 {s.get('risk_off_threshold')}/{s.get('risk_on_threshold')}, "
        f"최소 배수 {s.get('overlay_min_mult')}"
    )
    lines.append(
        "- 신호는 월말(t) 데이터까지만 사용하고, 노출은 t 다음 세션 종가에 바꿉니다. 현금 수익률은 0으로 둡니다."
    )
    lines += ["", "### 3.1 월말 레짐별 다음 달 벤치마크 수익률", ""]
    lines += ["| 시장 | 레짐 기준 | 판정 | n | 평균 | t | 상승 비율 |", "|---|---|---|---:|---:|---:|---:|"]
    for mkt, res in bt.get("markets", {}).items():
        if "error" in res:
            lines.append(f"| {mkt} | — | 오류: {res['error']} | | | | |")
            continue
        for region_key, cond in res["conditional"].items():
            lines += _cond_rows(f"{mkt} ({res.get('benchmark')})", region_key, cond)
    lines += ["", "### 3.2 오버레이 vs 바이앤홀드", ""]
    lines += [
        "| 시장 | 전략 | 연수익(CAGR) | 변동성 | 샤프 | 최대낙폭 | 평균 노출 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    verdicts = []
    for mkt, res in bt.get("markets", {}).items():
        if "error" in res:
            continue
        ov = res["overlay"]
        for name, st, exp in (("바이앤홀드", ov["buy_hold"], 1.0), ("오버레이", ov["overlay"], ov.get("avg_exposure"))):
            lines.append(
                f"| {mkt} | {name} | {_pct(st.get('cagr'))} | {_num((st.get('vol') or 0) * 100, '.1f')}% | "
                f"{_num(st.get('sharpe'), '.2f')} | {_pct(st.get('mdd'))} | {_num(exp, '.2f')} |"
            )
        p = res["period"]
        verdicts.append(
            f"- **{mkt} 판정: {ov['verdict']['text']}** "
            f"(기간 {p['start']}~{p['end']}, 노출 축소 {ov['months_reduced']}/{p['months']}개월, "
            f"점수 NaN {ov['months_nan_score']}개월)"
        )
    lines += [""] + verdicts
    lines.append(
        f"- 판정 기준(사전 고정): 샤프 +{SHARPE_MIN_GAIN:.2f} 이상이면 수익 개선, "
        f"최대낙폭 {MDD_MIN_GAIN * 100:.0f}%p 이상 축소면 리스크 축소로 봅니다. "
        "둘 다 표본 내 수치이며 다중검정 보정 전입니다."
    )
    counts = bt.get("label_counts") or {}
    if counts:
        lines += ["", "### 3.3 기간 내 레짐 분포(월)", "", "| 지역 | " + " | ".join(LABELS) + " |"]
        lines.append("|---|" + "---:|" * len(LABELS))
        for region, c in counts.items():
            lines.append(
                f"| {REGION_LABELS.get(region, region)} | " + " | ".join(str(c.get(lab, 0)) for lab in LABELS) + " |"
            )
    return lines


def _method_section(bt: dict[str, Any]) -> list[str]:
    lines = ["", "## 4. 방법·한계", ""]
    lines += [
        "- 지표마다 부호를 사전에 고정했습니다(+ = 위험선호). 백테스트 결과로 부호나 가중치를 맞추지 않았습니다.",
        "- z = (값 − 과거 평균) / 과거 표준편차입니다. 기준일까지의 z 창 데이터만 쓰고"
        "(최소 252개 관측), ±3에서 자릅니다.",
        "- 지역 점수 = clip(z/2, −1, 1)의 평균입니다. 절반 이상이 결측·지연이면 점수를 내지 않습니다(데이터 부족).",
        "- 단방향 지표(유가 3M, 미 10년물 3M)는 급등할 때만 감점합니다. 급락은 원인이 모호해 0으로 둡니다.",
        "- 국고채 3년·10년은 키 없는 네이버 채권 API로 2023-09-26 이후만 받을 수 있습니다(약 3년).",
        "  그 이전 한국 점수는 나머지 4개 지표로 계산합니다. CD 91일물은 키 없는 소스가 없어 뺐습니다.",
        "- KOSPI는 가격지수(배당 미포함)이고 SPY는 배당 조정 가격입니다.",
        "- 선물(구리·금·WTI)은 근월 연결 시계열이라 롤 갭이 섞입니다.",
    ]
    for n in (bt or {}).get("notes", []):
        lines.append(f"- {n}")
    data = (bt or {}).get("data", {})
    if data.get("failures"):
        for k, v in data["failures"].items():
            lines.append(f"- 수집 실패 {k}: {v}")
    return lines


def render_markdown(regimes: list[MacroRegime], bt: dict[str, Any] | None) -> str:
    as_of = max((r.as_of for r in regimes), default=None)
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    lines = [f"# 매크로 레짐 리포트 (기준 {as_of or '—'})", "", f"생성 {now}", "", "## 1. 현재 레짐", ""]
    lines += _regime_table(regimes)
    lines.append("")
    lines.append(
        "노출 배수: 점수가 0 이상이면 1.0이고, −1에서 최소 배수까지 선형으로 줄어듭니다. "
        "한국 비중에는 한국 점수, 미국 비중에는 미국 점수를 씁니다. 글로벌 배수는 참고용입니다."
    )
    lines += ["", "## 2. 지표 상세"]
    lines += _indicator_tables(regimes)
    lines += _backtest_section(bt or {})
    lines += _method_section(bt or {})
    return "\n".join(lines) + "\n"


def write_macro_report(
    regimes: list[MacroRegime],
    bt: dict[str, Any] | None,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Write ``macro_<as_of>.md`` (Korean) and ``macro_<as_of>.json`` to ``out_dir``; returns both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    as_of = max((r.as_of for r in regimes), default=datetime.now(KST).date())
    stem = f"macro_{as_of:%Y%m%d}"
    md_path = out / f"{stem}.md"
    json_path = out / f"{stem}.json"
    md_path.write_text(render_markdown(regimes, bt), encoding="utf-8")
    payload = {
        "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "as_of": as_of.isoformat(),
        "regimes": [_clean(r.model_dump(mode="python")) for r in regimes],
        "backtest": _clean(bt or {}),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return md_path, json_path
