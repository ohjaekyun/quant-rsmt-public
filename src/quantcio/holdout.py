"""Holdout protocol (docs/HOLDOUT.md): freeze fingerprints, the append-only registry and the tuning guard.

Public API
----------
- ``code_fingerprint(root=None) -> str`` - hash of the signal-generating ``.py`` files under ``src/quantcio``.
  Excluded: ``report/``, ``cli.py``, ``evaluation.py``, ``holdout.py``, ``tracking.py``, ``xsh.py`` (they format,
  evaluate or record signals but never change one). Included: everything else, notably the data providers
  (``data/*.py``, ``data/pit/``) because a changed input changes the signal.
- ``git_state() -> GitState`` - HEAD commit and whether tracked files outside ``holdout/`` differ from it
  (``None``/``None`` when git or the repository is unavailable: never raises).
- ``Registry`` - ``holdout/registry.json`` (JSON schema version 1): load / validate / append-only atomic save.
  Entries are never changed after registration; the only permitted change is appending to an entry's
  ``looks`` (``Registry.save`` re-reads the file on disk and refuses anything else).
- ``register_forward(settings, markets, allow_dirty)`` / ``register_cross_sectional(settings, market, names,
  period, excluded_tuning_names, allow_dirty)`` - append a frozen FTH / XSH entry (config digest, code
  fingerprint, git state, horizon/hold days and the pre-registered ``primary_metrics`` block).
- ``status(settings)`` - per entry: frozen state now (digest + fingerprint match), looks so far, pristine flag.
- ``check_frozen(entry, settings)`` -> ``(ok, reasons)``; ``record_look(entry_id, settings, ...)`` appends a look.
- ``tuning_names(market, settings, cache)`` - every name used for tuning so far: the point-in-time
  modeling / factor-backtest universe (KR PIT top ``backtest.kr_universe_n`` since ``kr_start``, US PIT S&P 500
  since ``us_start``; entity ids AND the tickers they carried), today's lists used by ``evaluate`` (KR
  ``universe(150) ∪ universe(60)``, US the static list) and the tickers named in the docs / tests. XSH candidates
  must avoid all of them.
- ``xsh_primary_metrics_block(settings)`` - the pre-registered XSH verdict rules stored with every XSH entry;
  ``xsh_confirmation(market, settings)`` - the confirmed, pristine, still-frozen XSH look of a market (or None):
  the stock-selection model may be served at full weight only when its gate passed AND this confirms it.
- Model scope (XSH only): ``model_scope_fingerprint()`` hashes the files the stock-selection model depends on (the
  signal fingerprint minus ``llm/``, ``fusion/``, ``pipeline.py`` and ``config.py``) and
  ``model_scope_digest(settings)`` the config values outside ``llm`` / ``report`` / ``holdout``. New XSH entries
  store both; entries registered before get them through ``attest_model_scope`` (allowed only while the entry's FULL
  freeze still holds, append-only ``attestations``). ``model_scope_frozen(entry, settings)`` compares them with the
  current code / config, so an LLM or report change no longer voids the confirmation of the stock-selection model
  (the forward holdout keeps the full freeze: it evaluates the final signals).
- ``tuning_exclusions(settings) -> (names_by_market, forward_start)`` and ``guard_panel(panel, settings)
  -> (panel, dropped_counts)`` - remove the XSH names and every row dated on/after the earliest FTH
  ``start_date`` from a TUNING / EVALUATION panel (never used for production serving);
  ``drop_xsh_names(frame, market, settings)`` removes only the XSH names (runs that also serve, e.g. the
  stock-selection model's ``modeling select``).

Paths: a relative ``holdout.registry_path`` / ``holdout.ledger_dir`` resolves against the project root (the
registry and the ledger are git-tracked), or against ``$QUANTCIO_HOLDOUT_ROOT`` when that is set. While pytest
runs (``PYTEST_CURRENT_TEST``), writing to the repository's own ``holdout/`` folder is refused, so a test can
never append synthetic rows to the real append-only files; tests point the paths at a temporary folder.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantcio.config import PROJECT_ROOT, SECRET_DATA_KEYS, Settings
from quantcio.data.base import PricePanel, normalise_ticker
from quantcio.data.cache import DiskCache
from quantcio.models import Market

log = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")
SCHEMA_VERSION = 1
KINDS = ("forward", "cross_sectional")
PACKAGE_ROOT = Path(__file__).resolve().parent
HOLDOUT_ROOT_ENV = "QUANTCIO_HOLDOUT_ROOT"
# files / folders (relative to src/quantcio) that never change a signal: excluded from the fingerprint
FINGERPRINT_EXCLUDED_DIRS = ("report",)
FINGERPRINT_EXCLUDED_FILES = ("cli.py", "evaluation.py", "holdout.py", "tracking.py", "xsh.py")
# stock-selection model scope (XSH confirmation): additionally NOT part of the model. config.py is covered by value
# through model_scope_digest (every config value outside llm / report / holdout), not by its source text
MODEL_SCOPE_EXCLUDED_DIRS = ("llm", "fusion")
MODEL_SCOPE_EXCLUDED_FILES = ("pipeline.py", "config.py")
MODEL_SCOPE_SETTINGS_EXCLUDED = ("llm",)
# model scope v2 (2026-09-27): per market. Files only the OTHER market's path runs are not part of a market's model
# (checked: the KR stock-selection path never calls data/pit/us*, the US path never calls data/pit/kr*)
MODEL_SCOPE_VERSION = 2
MODEL_SCOPE_MARKET_ONLY: dict[str, tuple[str, ...]] = {"KR": ("data/pit/kr",), "US": ("data/pit/us",)}
ATTESTATION_FIELDS = (
    "entry_id",
    "at",
    "code_fingerprint",
    "config_digest",
    "model_scope_fingerprint",
    "model_scope_digest",
)
# tickers named in README/docs/tests (seen while the hyper-parameters were chosen): never XSH candidates
DOC_TICKERS: dict[str, tuple[str, ...]] = {
    "KR": ("005930", "000660", "247540", "035720", "035420", "005380", "373220"),
    "US": ("AAPL", "NVDA", "MSFT", "TSLA", "JPM", "BRK-B"),
}
# universe sizes used while tuning / evaluating (evaluate --universe 150 and the default 60; US static list)
TUNING_UNIVERSE_SIZES: dict[str, tuple[int, ...]] = {"KR": (150, 60), "US": (200,)}
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "common": (
        "id",
        "kind",
        "registered_at",
        "config_digest",
        "code_fingerprint",
        "git_commit",
        "git_dirty",
        "horizon_days",
        "hold_days",
        "primary_metrics",
        "looks",
    ),
    "forward": ("start_date", "markets"),
    "cross_sectional": ("market", "names", "period", "excluded_tuning_names"),
}
# pre-registered verdict rules (docs/HOLDOUT.md "사전 등록 판정 규칙"); stored verbatim in every entry
METRIC_SELECTION = "selection_ic"
METRIC_DIRECTION = "direction_auc_gain"
METRIC_RISK = "risk_calibration"
METRIC_GRADE = "signal_grade"
METRIC_LLM = "llm_pillar_ic"
PRIMARY_KEYS = (METRIC_SELECTION, METRIC_DIRECTION, METRIC_RISK, METRIC_GRADE, METRIC_LLM)
T_EVIDENCE = 2.0
# XSH (cross-sectional holdout) pre-registered metrics
XSH_METRIC_AUC = "xsh_auc_median"
XSH_METRIC_IC = "xsh_ic"
XSH_METRIC_VS_MOM = "xsh_auc_vs_momentum"
XSH_PRIMARY_KEYS = (XSH_METRIC_AUC, XSH_METRIC_IC, XSH_METRIC_VS_MOM)
XSH_MIN_NAMES_PER_DATE = 20
XSH_BOOT_BLOCK_MONTHS = 3
# confirmation needs the same bar as the serving gate: the 95% block-bootstrap lower bound of the primary AUC > 0.5
# (added 2026-09-27 after XSH-US-20260927 was "confirmed" by BH at one-sided p ~ 0.09 with a CI that includes 0.5)
XSH_CONFIRM_MIN_LO = 0.5
COVER80_BAND = (0.75, 0.85)
EXC95_BAND = (0.02, 0.08)
BOOT_DRAWS = 2000
MIN_NAMES_PER_DATE = 5


class HoldoutError(RuntimeError):
    """Registry / guard failure (Korean message)."""


# =============================================================================== paths


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    root = os.environ.get(HOLDOUT_ROOT_ENV)
    return (Path(root) if root else PROJECT_ROOT) / path


def registry_path(settings: Settings) -> Path:
    return _resolve(settings.holdout.registry_path)


def ledger_dir(settings: Settings) -> Path:
    return _resolve(settings.holdout.ledger_dir)


def ensure_writable(path: Path) -> None:
    """Refuse writing to the repository's own ``holdout/`` folder while pytest runs (append-only files).

    Raises:
        HoldoutError: under pytest with a path inside ``<project>/holdout``.
    """
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        return
    repo_holdout = (PROJECT_ROOT / "holdout").resolve()
    try:
        Path(path).resolve().relative_to(repo_holdout)
    except ValueError:
        return
    raise HoldoutError(
        f"테스트 실행 중에는 저장소의 추가 전용 파일({path})에 쓰지 않습니다 — "
        "holdout.registry_path / holdout.ledger_dir를 임시 폴더로 지정하세요"
    )


# =============================================================================== freeze


def fingerprint_files(root: Path | None = None) -> list[Path]:
    """The signal-generating ``.py`` files under ``root`` (default: the installed ``quantcio`` package)."""
    base = Path(root) if root is not None else PACKAGE_ROOT
    out = []
    for p in sorted(base.rglob("*.py")):
        rel = p.relative_to(base)
        if "__pycache__" in rel.parts:
            continue
        if rel.parts[0] in FINGERPRINT_EXCLUDED_DIRS:
            continue
        if len(rel.parts) == 1 and rel.name in FINGERPRINT_EXCLUDED_FILES:
            continue
        out.append(p)
    return out


def code_fingerprint(root: Path | None = None) -> str:
    """sha256 over (relative path, file sha256) of every signal-generating file; 16 hex chars.

    Raises:
        HoldoutError: no file found (wrong root).
    """
    base = Path(root) if root is not None else PACKAGE_ROOT
    files = fingerprint_files(base)
    if not files:
        raise HoldoutError(f"신호 코드 파일을 찾을 수 없습니다: {base}")
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(base).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def model_scope_files(root: Path | None = None, market: str | None = None) -> list[Path]:
    """The signal-generating files the stock-selection model depends on (see ``MODEL_SCOPE_*``). With ``market``
    (scope v2) the files only the other market's path runs (``MODEL_SCOPE_MARKET_ONLY``) are left out too."""
    base = Path(root) if root is not None else PACKAGE_ROOT
    other = [pre for m, pres in MODEL_SCOPE_MARKET_ONLY.items() if market and m != market for pre in pres]
    out = []
    for p in fingerprint_files(base):
        rel = p.relative_to(base)
        if rel.parts[0] in MODEL_SCOPE_EXCLUDED_DIRS:
            continue
        if len(rel.parts) == 1 and rel.name in MODEL_SCOPE_EXCLUDED_FILES:
            continue
        if any(rel.as_posix().startswith(pre) for pre in other):
            continue
        out.append(p)
    return out


def model_scope_fingerprint(root: Path | None = None, market: str | None = None) -> str:
    """sha256 over the model-scope files, like ``code_fingerprint``; 16 hex chars. ``market`` None = scope v1
    (both markets' data code), else scope v2 of that market.

    Raises:
        HoldoutError: no file found (wrong root).
    """
    base = Path(root) if root is not None else PACKAGE_ROOT
    files = model_scope_files(base, market)
    if not files:
        raise HoldoutError(f"모델 범위 코드 파일을 찾을 수 없습니다: {base}")
    h = hashlib.sha256(b"model-scope-v1\n" if market is None else f"model-scope-v2:{market}\n".encode())
    for p in files:
        h.update(p.relative_to(base).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def model_scope_digest(settings: Settings) -> str:
    """Hash of every config value the stock-selection model can read: ``Settings.digest`` minus the ``llm`` block."""
    exclude: dict[str, Any] = {"data": set(SECRET_DATA_KEYS), "report": True, "holdout": True}
    exclude.update({k: True for k in MODEL_SCOPE_SETTINGS_EXCLUDED})
    payload = settings.model_dump(exclude=exclude)
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


@dataclass(frozen=True)
class GitState:
    commit: str | None  # HEAD sha, None without git / repository
    dirty: bool | None  # tracked files outside holdout/ differ from HEAD; None when unknown
    note: str = ""  # Korean reason when unknown


def git_state(cwd: Path | None = None) -> GitState:
    """HEAD commit + dirty flag via ``git``; tolerant: missing git / not a repository -> (None, None, note).

    The dirty check ignores ``holdout/`` (the registry and ledger are appended to constantly) and
    untracked files.
    """
    where = Path(cwd) if cwd is not None else PROJECT_ROOT

    def run(*args: str) -> str:
        res = subprocess.run(["git", *args], cwd=where, capture_output=True, text=True, timeout=20, check=False)
        if res.returncode != 0:
            raise RuntimeError((res.stderr or res.stdout).strip()[:200] or f"git {args[0]} 실패")
        return res.stdout

    try:
        commit = run("rev-parse", "HEAD").strip()
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return GitState(None, None, f"git 정보 없음({type(exc).__name__}: {str(exc)[:120]})")
    try:
        out = run("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)holdout")
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return GitState(commit, None, f"git 상태 확인 실패({type(exc).__name__}: {str(exc)[:120]})")
    return GitState(commit, bool(out.strip()), "")


def primary_metrics_block(settings: Settings) -> dict[str, Any]:
    """The pre-registered verdict rules (docs/HOLDOUT.md), stored with every registry entry."""
    h = int(settings.horizon_days)
    return {
        "min_dates_for_verdict": int(settings.holdout.min_dates_for_verdict),
        "bh_q": float(settings.holdout.bh_q),
        "non_overlap": f"같은 종목의 예측이 {h}거래일 안에 여러 번 있으면 가장 이른 한 건만 판정 표본에 사용",
        "date_unit": "시장별 예측 기준일(as_of) 횡단면",
        "min_names_per_date": MIN_NAMES_PER_DATE,
        "bootstrap_draws": BOOT_DRAWS,
        "metrics": [
            {
                "key": METRIC_SELECTION,
                "label": "종목 선택: 날짜별 μ_excess 순위 IC",
                "rule": "날짜별 Spearman(mu_excess, y_e) 평균의 t ≥ 2",
                "t_min": T_EVIDENCE,
                "p_value": "단측 t검정(자유도 날짜 수 − 1)",
            },
            {
                "key": METRIC_DIRECTION,
                "label": "방향: P(상승) 날짜 군집 AUC 개선(루트 쌍둥이 대비)",
                "rule": "AUC(prob_up) − AUC(base_prob_up)의 날짜 블록 부트스트랩 5% 하한 > 0",
                "lower_quantile": 0.05,
                "p_value": "부트스트랩 개선 ≤ 0 비율(단측)",
            },
            {
                "key": METRIC_RISK,
                "label": "리스크 보정: 80% 구간 적중률·VaR95 초과율",
                "rule": "cover80 ∈ [0.75, 0.85] 그리고 exc95 ∈ [0.02, 0.08]",
                "cover80_band": list(COVER80_BAND),
                "exc95_band": list(EXC95_BAND),
                "p_value": "날짜 군집 표준오차의 동등성 검정(TOST) 두 지표 중 큰 p",
            },
            {
                "key": METRIC_GRADE,
                "label": "시그널 등급: Buy/Strong Buy − Hold/Sell 실현 초과수익",
                "rule": "날짜별 평균 차이의 날짜 군집 t ≥ 2",
                "t_min": T_EVIDENCE,
                "p_value": "단측 t검정(자유도 날짜 수 − 1)",
            },
            {
                "key": METRIC_LLM,
                "label": "LLM 필러: llm_pillar와 실현 초과수익의 순위 IC (LLM 실행 행만)",
                "rule": "날짜별 Spearman(llm_pillar, y_e) 평균의 t ≥ 2",
                "t_min": T_EVIDENCE,
                "p_value": "단측 t검정(자유도 날짜 수 − 1)",
            },
        ],
        "multiple_testing": "Benjamini–Hochberg(주 지표 5개, q = bh_q)",
    }


def xsh_primary_metrics_block(settings: Settings) -> dict[str, Any]:
    """The pre-registered XSH verdict rules (docs/HOLDOUT.md "XSH"), stored with every XSH entry.

    One look evaluates the SELECTED stock-selection model (frozen config + code) on the registered never-tuned
    names: trained walk-forward on the tuning universe only (the same outer / holdout contexts and purge as
    ``modeling.selection``), scored on the XSH names' month-end cross-sections inside the registered period.
    """
    h = int(settings.horizon_days)
    return {
        "kind": "cross_sectional",
        "model": (
            "modeling select 의 선택 모델(동결 설정·코드), 튜닝 유니버스로만 워크포워드 학습(outer·holdout 컨텍스트)"
        ),
        "date_unit": "월말 리밸런싱 날짜의 XSH 종목 횡단면 (등록 기간 안, 학습 가능한 컨텍스트가 있는 날짜)",
        "label": f"y_e = {h}거래일 베타조정 초과수익; 주 라벨 1{{y_e > 날짜 중앙값}}",
        "min_dates_for_verdict": int(settings.holdout.min_dates_for_verdict),
        "min_names_per_date": XSH_MIN_NAMES_PER_DATE,
        "bootstrap": f"날짜 순환 이동 블록 부트스트랩(블록 {XSH_BOOT_BLOCK_MONTHS}개월, {BOOT_DRAWS}회)",
        "bh_q": float(settings.holdout.bh_q),
        "target_auc": float(settings.holdout.target_auc),
        "metrics": [
            {
                "key": XSH_METRIC_AUC,
                "label": "날짜 내 AUC(중앙값 초과 라벨) > 0.5",
                "rule": "날짜별 AUC 평균의 블록 부트스트랩 단측 p (부트스트랩 평균 ≤ 0.5 비율)",
            },
            {
                "key": XSH_METRIC_IC,
                "label": "날짜별 순위 IC(점수, y_e) > 0",
                "rule": "날짜별 Spearman IC 평균의 단측 t검정(자유도 날짜 수 − 1), t ≥ 2 는 참고 기준",
            },
            {
                "key": XSH_METRIC_VS_MOM,
                "label": "12-1 모멘텀 기준 대비 AUC(중앙값 초과) 개선 > 0",
                "rule": "날짜별 AUC 차이 평균의 블록 부트스트랩 단측 p",
            },
        ],
        "multiple_testing": "Benjamini–Hochberg(주 지표 3개, q = bh_q)",
        "confirmed_rule": (
            f"{XSH_METRIC_AUC} 가 BH 보정 후 유의 AND AUC 95% 부트스트랩 하한 > {XSH_CONFIRM_MIN_LO} AND 평균 IC > 0 "
            "AND 날짜 수 ≥ min_dates_for_verdict → confirmed (서빙 게이트의 미열람 표본 확인으로 사용)"
        ),
        "confirm_min_lo": XSH_CONFIRM_MIN_LO,
        "target_rule": "평균 AUC(중앙값 초과) ≥ target_auc 이면 목표 달성(검정 아님, 신뢰구간과 함께 보고)",
        "sensitivity": (
            "등록 종목 외에 같은 순위 구간의 모든 시점별 행(튜닝 이력 종목 포함)도 참고로 보고(판정에 미사용)"
        ),
    }


# =============================================================================== registry


def _empty() -> dict[str, Any]:
    return {"version": SCHEMA_VERSION, "entries": []}


def validate_registry(data: Any) -> None:
    """Schema version 1: {"version": 1, "entries": [entry, ...]} with unique ids and the required fields.

    Raises:
        HoldoutError: any violation (Korean message).
    """
    if not isinstance(data, dict):
        raise HoldoutError("레지스트리 형식 오류: 최상위가 객체가 아님")
    if data.get("version") != SCHEMA_VERSION:
        raise HoldoutError(f"레지스트리 버전 불일치: {data.get('version')!r} (지원: {SCHEMA_VERSION})")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise HoldoutError("레지스트리 형식 오류: entries가 목록이 아님")
    seen: set[str] = set()
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise HoldoutError(f"레지스트리 항목 {i}: 객체가 아님")
        kind = e.get("kind")
        if kind not in KINDS:
            raise HoldoutError(f"레지스트리 항목 {e.get('id', i)}: 알 수 없는 kind {kind!r}")
        missing = [k for k in (*REQUIRED_FIELDS["common"], *REQUIRED_FIELDS[kind]) if k not in e]
        if missing:
            raise HoldoutError(f"레지스트리 항목 {e.get('id', i)}: 필수 필드 누락 {missing}")
        if e["id"] in seen:
            raise HoldoutError(f"레지스트리 항목 id 중복: {e['id']}")
        seen.add(e["id"])
        if not isinstance(e["looks"], list):
            raise HoldoutError(f"레지스트리 항목 {e['id']}: looks가 목록이 아님")
    att = data.get("attestations", [])
    if not isinstance(att, list):
        raise HoldoutError("레지스트리 형식 오류: attestations가 목록이 아님")
    for i, a in enumerate(att):
        if not isinstance(a, dict) or any(k not in a for k in ATTESTATION_FIELDS):
            raise HoldoutError(f"레지스트리 attestation {i}: 필수 필드 누락 {list(ATTESTATION_FIELDS)}")
        if a["entry_id"] not in seen:
            raise HoldoutError(f"레지스트리 attestation {i}: 없는 항목 {a['entry_id']}")


def _check_append_only(old: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    """Every entry of `old` must survive unchanged in `new` except for looks appended at the end."""
    by_id = {e["id"]: e for e in new["entries"]}
    for e in old["entries"]:
        n = by_id.get(e["id"])
        if n is None:
            raise HoldoutError(f"추가 전용 위반: 등록 항목 {e['id']} 삭제 시도")
        a = {k: v for k, v in e.items() if k != "looks"}
        b = {k: v for k, v in n.items() if k != "looks"}
        if a != b:
            changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            raise HoldoutError(f"추가 전용 위반: 등록 항목 {e['id']}의 필드 변경 시도 {changed}")
        if n["looks"][: len(e["looks"])] != e["looks"]:
            raise HoldoutError(f"추가 전용 위반: 등록 항목 {e['id']}의 기존 look 변경·삭제 시도")
    old_ids = [e["id"] for e in old["entries"]]
    new_ids = [e["id"] for e in new["entries"]]
    if new_ids[: len(old_ids)] != old_ids:
        raise HoldoutError("추가 전용 위반: 등록 항목 순서 변경 시도")
    old_att = list(old.get("attestations", []))
    if list(new.get("attestations", []))[: len(old_att)] != old_att:
        raise HoldoutError("추가 전용 위반: 기존 attestation 변경·삭제 시도")


class Registry:
    """``holdout/registry.json``: append-only (new entries, new looks); atomic writes."""

    def __init__(self, path: Path, data: dict[str, Any] | None = None):
        self.path = Path(path)
        self.data = data if data is not None else _empty()
        validate_registry(self.data)

    @classmethod
    def load(cls, path: Path) -> Registry:
        """Missing file -> an empty registry. A corrupt / invalid file raises (never silently reset)."""
        return cls(Path(path), cls._read(Path(path)))

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        if not path.exists():
            return _empty()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HoldoutError(f"레지스트리 읽기 실패({path}): {type(exc).__name__}: {exc}") from exc
        validate_registry(data)
        return data

    @property
    def entries(self) -> list[dict[str, Any]]:
        return self.data["entries"]

    def get(self, entry_id: str) -> dict[str, Any]:
        for e in self.entries:
            if e["id"] == entry_id:
                return e
        raise HoldoutError(f"등록된 홀드아웃 없음: {entry_id}")

    def ids(self) -> list[str]:
        return [e["id"] for e in self.entries]

    def forward_entries(self, market: str | None = None) -> list[dict[str, Any]]:
        out = [e for e in self.entries if e["kind"] == "forward"]
        if market is not None:
            out = [e for e in out if market in e.get("markets", [])]
        return out

    def active_forward(self, market: str | None = None) -> dict[str, Any] | None:
        """The latest forward registration (covering `market`, when given)."""
        fwd = self.forward_entries(market)
        return max(fwd, key=lambda e: str(e["registered_at"])) if fwd else None

    def unique_id(self, base: str) -> str:
        ids = set(self.ids())
        if base not in ids:
            return base
        k = 2
        while f"{base}-{k}" in ids:
            k += 1
        return f"{base}-{k}"

    def append_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        trial = {"version": SCHEMA_VERSION, "entries": [*self.entries, entry]}
        validate_registry(trial)
        self.entries.append(entry)
        return entry

    def append_look(self, entry_id: str, look: dict[str, Any]) -> dict[str, Any]:
        self.get(entry_id)["looks"].append(look)
        return look

    def save(self) -> Path:
        """Append-only check against the file on disk (re-read now), then an atomic replace.

        Raises:
            HoldoutError: an existing entry / look would change or disappear, or the write is refused.
        """
        validate_registry(self.data)
        ensure_writable(self.path)
        _check_append_only(self._read(self.path), self.data)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.data, ensure_ascii=False, indent=2, default=str) + "\n"
        fd, tmp = tempfile.mkstemp(prefix=".registry-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return self.path


def load_registry(settings: Settings) -> Registry:
    return Registry.load(registry_path(settings))


# =============================================================================== register


def _now() -> datetime:
    return datetime.now(KST).replace(microsecond=0)


def _frozen_fields(settings: Settings, allow_dirty: bool) -> dict[str, Any]:
    gs = git_state()
    if gs.dirty is None and not allow_dirty:
        raise HoldoutError(
            f"git 커밋을 확인할 수 없습니다({gs.note}) — 커밋 후 등록하거나 --allow-dirty로 등록하세요 "
            "(git_dirty=null로 기록됨)"
        )
    if gs.dirty and not allow_dirty:
        raise HoldoutError("작업 트리가 커밋과 다릅니다(dirty) — 커밋 후 등록하거나 --allow-dirty로 등록하세요")
    out: dict[str, Any] = {
        "config_digest": settings.digest(),
        "code_fingerprint": code_fingerprint(),
        "git_commit": gs.commit,
        "git_dirty": gs.dirty,
        "allow_dirty": bool(allow_dirty),
        "horizon_days": int(settings.horizon_days),
        "hold_days": int(settings.fusion.target_horizon_days),
        "primary_metrics": primary_metrics_block(settings),
    }
    if gs.note:
        out["git_note"] = gs.note
    return out


def _markets(markets: Iterable[str | Market]) -> list[str]:
    out: list[str] = []
    for m in markets:
        try:
            v = Market(str(getattr(m, "value", m)).strip().upper()).value
        except ValueError as exc:
            raise HoldoutError(f"시장은 KR 또는 US여야 합니다: {m}") from exc
        if v not in out:
            out.append(v)
    if not out:
        raise HoldoutError("등록할 시장이 없습니다")
    return out


def register_forward(
    settings: Settings, markets: Sequence[str | Market] = ("KR", "US"), allow_dirty: bool = False
) -> dict[str, Any]:
    """Append a forward (time) holdout starting now; returns the stored entry.

    Raises:
        HoldoutError: dirty / unknown git state without allow_dirty, bad market, registry violation.
    """
    reg = load_registry(settings)
    now = _now()
    entry = {
        "id": reg.unique_id(f"FTH-{now:%Y%m%d}"),
        "kind": "forward",
        "registered_at": now.isoformat(),
        "start_date": now.date().isoformat(),
        "markets": _markets(markets),
        **_frozen_fields(settings, allow_dirty),
        "looks": [],
    }
    reg.append_entry(entry)
    reg.save()
    log.info("holdout: registered %s (%s)", entry["id"], entry["markets"])
    return copy.deepcopy(entry)


def register_cross_sectional(
    settings: Settings,
    market: str | Market,
    names: Sequence[str],
    period: Sequence[str | date],
    excluded_tuning_names: Sequence[str],
    allow_dirty: bool = False,
    rank_band: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Append a cross-sectional holdout of never-tuned names (``rank_band``: KR PIT rank band, stored so the
    evaluation rebuilds the same universe whatever the config says later).

    Raises:
        HoldoutError: empty names, a name that is also a tuning name, a bad period or a registry violation.
    """
    mk = _markets([market])[0]
    clean = sorted({str(n).strip() for n in names if str(n).strip()})
    if not clean:
        raise HoldoutError("XSH 종목이 비어 있습니다")
    tuned = {_name_key(t) for t in excluded_tuning_names}
    overlap = sorted(n for n in clean if _name_key(n) in tuned)
    if overlap:
        raise HoldoutError(f"XSH 종목 {len(overlap)}개가 튜닝에 쓰인 종목입니다: {', '.join(overlap[:10])}")
    if len(period) != 2:
        raise HoldoutError(f"기간은 [시작, 끝] 두 날짜여야 합니다: {period}")
    p0, p1 = (pd.Timestamp(p).date() for p in period)
    if p0 >= p1:
        raise HoldoutError(f"기간 시작이 끝보다 늦습니다: {p0} ~ {p1}")
    reg = load_registry(settings)
    now = _now()
    entry = {
        "id": reg.unique_id(f"XSH-{mk}-{now:%Y%m%d}"),
        "kind": "cross_sectional",
        "registered_at": now.isoformat(),
        "market": mk,
        "names": clean,
        "period": [p0.isoformat(), p1.isoformat()],
        "excluded_tuning_names": sorted({str(t).strip() for t in excluded_tuning_names if str(t).strip()}),
        **({"rank_band": [int(x) for x in rank_band]} if rank_band else {}),
        **_frozen_fields(settings, allow_dirty),
        "looks": [],
    }
    entry["primary_metrics"] = xsh_primary_metrics_block(settings)
    entry["model_scope_version"] = MODEL_SCOPE_VERSION
    entry["model_scope_fingerprint"] = model_scope_fingerprint(market=mk)
    entry["model_scope_digest"] = model_scope_digest(settings)
    reg.append_entry(entry)
    reg.save()
    log.info("holdout: registered %s (%d names)", entry["id"], len(clean))
    return copy.deepcopy(entry)


# =============================================================================== frozen / looks / status


def check_frozen(entry: Mapping[str, Any], settings: Settings) -> tuple[bool, list[str]]:
    """Whether the current config digest and signal-code fingerprint equal the registered ones."""
    reasons: list[str] = []
    digest = settings.digest()
    if digest != entry.get("config_digest"):
        reasons.append(f"설정 다이제스트 변경: 등록 {entry.get('config_digest')} → 현재 {digest}")
    try:
        fp = code_fingerprint()
    except HoldoutError as exc:
        fp = None
        reasons.append(f"코드 지문 계산 실패: {exc}")
    if fp is not None and fp != entry.get("code_fingerprint"):
        reasons.append(f"신호 코드 지문 변경: 등록 {entry.get('code_fingerprint')} → 현재 {fp}")
    return not reasons, reasons


def _scope_expected(
    entry: Mapping[str, Any], reg: Registry | None, version: int | None = None
) -> tuple[str | None, str | None, str, int]:
    """(fingerprint, digest, source, scope version) of the entry's model scope - the highest version recorded (or
    exactly ``version``): stored in the entry, else in an attestation."""
    found: list[tuple[int, str, str, str]] = []
    if entry.get("model_scope_fingerprint") and entry.get("model_scope_digest"):
        v = int(entry.get("model_scope_version", 1))
        found.append((v, entry["model_scope_fingerprint"], entry["model_scope_digest"], "entry"))
    for a in reg.data.get("attestations", []) if reg is not None else []:
        if a["entry_id"] == entry["id"]:
            v = int(a.get("scope_version", 1))
            found.append((v, a["model_scope_fingerprint"], a["model_scope_digest"], f"attestation v{v}"))
    if version is not None:
        found = [f for f in found if f[0] == version]
    if not found:
        return None, None, "none", 0
    v, fp, dg, src = max(found, key=lambda f: f[0])
    return fp, dg, src, v


def _scope_market(entry: Mapping[str, Any], version: int) -> str | None:
    return str(entry.get("market")) if version >= 2 else None


def model_scope_frozen(
    entry: Mapping[str, Any], settings: Settings, reg: Registry | None = None, version: int | None = None
) -> tuple[bool, list[str]]:
    """Whether the stock-selection model scope (code + config values) still equals the entry's frozen one (the
    highest scope version recorded for it, or ``version``)."""
    reg = reg if reg is not None else load_registry(settings)
    fp0, dg0, src, ver = _scope_expected(entry, reg, version)
    if fp0 is None:
        return False, ["모델 범위 지문이 기록되지 않음(attest_model_scope 필요)"]
    reasons: list[str] = []
    try:
        fp = model_scope_fingerprint(market=_scope_market(entry, ver))
    except HoldoutError as exc:
        return False, [f"모델 범위 지문 계산 실패: {exc}"]
    if fp != fp0:
        reasons.append(f"모델 범위 코드 지문 변경({src}): {fp0} → {fp}")
    dg = model_scope_digest(settings)
    if dg != dg0:
        reasons.append(f"모델 범위 설정 다이제스트 변경({src}): {dg0} → {dg}")
    return not reasons, reasons


def attest_model_scope(entry_id: str, settings: Settings, version: int = MODEL_SCOPE_VERSION) -> dict[str, Any]:
    """Record the model-scope fingerprint / digest (scope ``version``) of an XSH entry that lacks it.

    Allowed only while the recorded state provably still holds, so the values describe exactly the registered code
    and config: the entry's FULL freeze (``check_frozen``), or - for scope v2, whose files are a subset of scope v1 -
    a recorded scope v1 that still matches (``model_scope_frozen(..., version=1)``). The basis is stored with the
    attestation. Append-only, once per entry and version.

    Raises:
        HoldoutError: not an XSH entry, already has this (or a newer) scope, or no basis holds any more.
    """
    reg = load_registry(settings)
    entry = reg.get(entry_id)
    if entry["kind"] != "cross_sectional":
        raise HoldoutError(f"{entry_id}: 횡단면 홀드아웃만 모델 범위를 증명할 수 있습니다")
    if _scope_expected(entry, reg)[3] >= int(version):
        raise HoldoutError(f"{entry_id}: 모델 범위(v{version} 이상)가 이미 기록되어 있습니다")
    ok, reasons = check_frozen(entry, settings)
    basis = "full_freeze"
    if not ok and int(version) >= 2:
        ok1, reasons1 = model_scope_frozen(entry, settings, reg, version=1)
        if ok1:
            ok, basis = True, "model_scope_v1"
        else:
            reasons = [*reasons, *reasons1]
    if not ok:
        raise HoldoutError(
            f"{entry_id}: 전체 동결·기록된 모델 범위가 모두 깨져 증명할 수 없습니다 — " + "; ".join(reasons)
        )
    mk = _scope_market(entry, int(version))
    att = {
        "entry_id": entry_id,
        "at": _now().isoformat(),
        "code_fingerprint": entry["code_fingerprint"],
        "config_digest": entry["config_digest"],
        "model_scope_fingerprint": model_scope_fingerprint(market=mk),
        "model_scope_digest": model_scope_digest(settings),
        "scope_version": int(version),
        "market": mk,
        "basis": basis,
    }
    reg.data.setdefault("attestations", []).append(att)
    reg.save()
    return copy.deepcopy(att)


def record_look(
    entry_id: str,
    settings: Settings,
    *,
    result_path: str | Path | None = None,
    headline: Mapping[str, Any] | None = None,
    forced: bool = False,
    note: str = "",
) -> dict[str, Any]:
    """Append one look (validity, pristine flag, where the result is) to the entry; returns the look.

    ``valid`` = digest and fingerprint match the registration; an invalid look is ``contaminated``.
    XSH: only the FIRST look can be pristine (and only when valid) - any earlier look, even a
    contaminated one, has already revealed the held-out result. FTH: ``pristine`` is null (the future
    is still unseen when it is looked at).
    """
    reg = load_registry(settings)
    entry = reg.get(entry_id)
    ok, reasons = check_frozen(entry, settings)
    xsh = entry["kind"] == "cross_sectional"
    look = {
        "at": _now().isoformat(),
        "config_digest": settings.digest(),
        "code_fingerprint": _safe_fingerprint(),
        "valid": ok,
        "contaminated": not ok,
        "forced": bool(forced),
        "pristine": (ok and not entry["looks"]) if xsh else None,
        "reasons": reasons,
        "result_path": str(result_path) if result_path is not None else None,
        "headline": _jsonable(dict(headline or {})),
    }
    if note:
        look["note"] = note
    reg.append_look(entry_id, look)
    reg.save()
    return copy.deepcopy(look)


def _safe_fingerprint() -> str | None:
    try:
        return code_fingerprint()
    except HoldoutError:
        return None


def status(settings: Settings) -> list[dict[str, Any]]:
    """One row per entry: frozen now?, reasons, looks so far, pristine flag (XSH: no look yet)."""
    reg = load_registry(settings)
    rows = []
    for e in reg.entries:
        ok, reasons = check_frozen(e, settings)
        looks = e["looks"]
        xsh = e["kind"] == "cross_sectional"
        scope_ok, scope_reasons = model_scope_frozen(e, settings, reg) if xsh else (None, [])
        first = looks[0] if looks else None
        rows.append(
            {
                "id": e["id"],
                "kind": e["kind"],
                "markets": [e["market"]] if xsh else list(e.get("markets", [])),
                "registered_at": e["registered_at"],
                "start_date": e.get("start_date"),
                "n_names": len(e.get("names", [])) if xsh else None,
                "frozen_ok": ok,
                "reasons": reasons,
                "model_scope_ok": scope_ok,
                "model_scope_reasons": scope_reasons,
                "n_looks": len(looks),
                "n_valid_looks": sum(1 for lk in looks if lk.get("valid")),
                # XSH: still pristine = never looked at; FTH: not applicable
                "pristine": (not looks) if xsh else None,
                "pristine_look": (first.get("result_path") if first and first.get("pristine") else None),
                "git_commit": e.get("git_commit"),
                "git_dirty": e.get("git_dirty"),
            }
        )
    return rows


# =============================================================================== tuning guard


def _name_key(name: str) -> str:
    """Comparable key: KR PIT entity '005930#1' -> '005930'; tickers canonicalised (BRK.B -> BRK-B)."""
    s = str(name).strip()
    if "#" in s:
        s = s.split("#", 1)[0]
    return normalise_ticker(s) if s else s


def pit_tuning_universe(market: str | Market, settings: Settings) -> tuple[int, date]:
    """(top-N, start) of the point-in-time universe the stock-selection model and the factor backtest were tuned
    on (``backtest.kr_universe_n`` / ``kr_start``, ``us_universe_n`` / ``us_start``). The RSMT-Q training universe
    (PIT top ``rsmt.markets.<M>.universe_size`` <= N over the last ``history_years``) lies inside it."""
    bt = settings.backtest
    if _markets([market])[0] == "KR":
        return int(bt.kr_universe_n), date.fromisoformat(bt.kr_start)
    return int(bt.us_universe_n), date.fromisoformat(bt.us_start)


def tuning_names(
    market: str | Market, settings: Settings, cache: DiskCache, *, pit_universe: Any = None, pit: bool = True
) -> list[str]:
    """Every name used for tuning so far in `market` (sorted, canonical keys; see ``_name_key``).

    - the PIT modeling / factor-backtest universe (``pit_tuning_universe``): every entity that was a member on
      any rebalance date, as its entity key AND every ticker it carried (``pit_universe`` injects a loaded
      ``PITUniverse``; ``pit=False`` skips it - tests only);
    - today's lists used by ``evaluate`` (KR ``universe(150) ∪ universe(60)``, US the static list: taken at
      registration time and stored in the XSH entry);
    - ``DOC_TICKERS`` (README / docs / tests);
    - the names of every EARLIER cross-sectional holdout of the market (each was looked at once, so it is no
      longer untouched - e.g. KR ranks 201-400 of XSH-KR-20260927 when the next XSH uses ranks 401-600).

    Raises:
        HoldoutError: a universe or the registry cannot be loaded (the exclusion would be incomplete).
    """
    from quantcio.data import base as data_base

    mk = _markets([market])[0]
    names = {normalise_ticker(t) for t in DOC_TICKERS[mk]}
    try:
        provider = data_base.get_provider(Market(mk), settings, cache)
        for n in TUNING_UNIVERSE_SIZES[mk]:
            names.update(normalise_ticker(t) for t in provider.universe(n))
        if pit:
            uni = pit_universe
            if uni is None:
                from quantcio.data.pit import load_pit_universe

                n, start = pit_tuning_universe(mk, settings)
                uni = load_pit_universe(Market(mk), settings, cache, n, start)
            mem = uni.membership
            names.update(_name_key(e) for e in mem["entity"].astype(str).unique())
            if "ticker" in mem:
                names.update(_name_key(t) for t in mem["ticker"].dropna().astype(str).unique() if t.strip())
    except Exception as exc:  # noqa: BLE001 - an incomplete exclusion list must stop the registration
        raise HoldoutError(f"{mk} 튜닝 유니버스 로드 실패({type(exc).__name__}: {str(exc)[:160]})") from exc
    try:
        for e in load_registry(settings).entries:
            if e.get("kind") == "cross_sectional" and e.get("market") == mk:
                names.update(_name_key(n) for n in e.get("names", []) or [])
    except Exception as exc:  # noqa: BLE001 - same rule: never register with an incomplete exclusion
        raise HoldoutError(f"{mk} 이전 XSH 종목 로드 실패({type(exc).__name__}: {str(exc)[:160]})") from exc
    names.discard("")
    return sorted(names)


def xsh_confirmation(market: str | Market, settings: Settings) -> dict[str, Any] | None:
    """The latest XSH entry of `market` whose FIRST look is valid, pristine and ``confirmed`` (see
    ``xsh_primary_metrics_block``) with a primary-AUC 95% lower bound > ``XSH_CONFIRM_MIN_LO`` (the serving gate's
    bar; also applied to entries registered before the rule said so) and whose frozen config digest + signal-code
    fingerprint still equal the current ones; None otherwise. Never raises (a serving path calls it): any failure
    means "not confirmed"."""
    try:
        mk = _markets([market])[0]
        reg = load_registry(settings)
        found = []
        for e in reg.entries:
            if e.get("kind") != "cross_sectional" or e.get("market") != mk or not e.get("looks"):
                continue
            first = e["looks"][0]
            head = first.get("headline") or {}
            if not (first.get("valid") and first.get("pristine") and head.get("confirmed") is True):
                continue
            lo = head.get("auc_median_lo")
            if not (isinstance(lo, int | float) and math.isfinite(lo) and lo > XSH_CONFIRM_MIN_LO):
                log.info("xsh_confirmation(%s): %s confirmed but AUC lower bound %s <= %s", mk, e["id"], lo, 0.5)
                continue
            ok, _ = check_frozen(e, settings)
            if not ok:
                ok, _ = model_scope_frozen(e, settings, reg)
            if ok:
                found.append((str(e["registered_at"]), e, first, head))
        if not found:
            return None
        _, e, first, head = max(found, key=lambda x: x[0])
        return {
            "id": e["id"],
            "looked_at": first.get("at"),
            "auc_median": head.get("auc_median"),
            "auc_median_lo": head.get("auc_median_lo"),
            "ic": head.get("ic"),
            "n_dates": head.get("n_dates"),
            "result_path": first.get("result_path"),
        }
    except Exception as exc:  # noqa: BLE001 - "not confirmed" is the safe answer
        log.warning("xsh_confirmation(%s) failed: %s", market, exc)
        return None


def tuning_exclusions(settings: Settings) -> tuple[dict[str, set[str]], date | None]:
    """({market: XSH names (keys)}, earliest FTH start_date or None) from the registry."""
    reg = load_registry(settings)
    names: dict[str, set[str]] = {}
    for e in reg.entries:
        if e["kind"] == "cross_sectional":
            names.setdefault(e["market"], set()).update(_name_key(n) for n in e["names"])
    starts = [pd.Timestamp(e["start_date"]).date() for e in reg.forward_entries()]
    return names, (min(starts) if starts else None)


def guard_panel(panel: Any, settings: Settings) -> tuple[Any, dict[str, int]]:
    """Remove the registered holdouts from a TUNING / EVALUATION panel (never call it for serving).

    - ``PricePanel``: drops the XSH names of the panel's market (price frames, benchmark map, membership) and
      truncates every price / benchmark frame and membership row to dates before the earliest FTH start, so
      no feature and no label can use a forward-holdout price.
    - ``DataFrame`` with ``date`` and ``ticker`` columns (e.g. ``PanelData.train``): drops the rows of XSH names
      and rows dated on/after the FTH start. Labels of earlier rows may still span the start: guard the
      ``PricePanel`` before ``build_panel`` whenever possible.

    Returns a new object (the input is not modified) and the counts ``{"xsh_names", "xsh_rows",
    "forward_rows", "benchmark_rows", "membership_rows"}``; non-zero counts are logged and, for a PricePanel,
    added to ``panel.notes`` in Korean.

    Raises:
        TypeError: an unsupported panel type.
    """
    names_by_market, start = tuning_exclusions(settings)
    counts = {"xsh_names": 0, "xsh_rows": 0, "forward_rows": 0, "benchmark_rows": 0, "membership_rows": 0}
    cut = pd.Timestamp(start) if start is not None else None
    if isinstance(panel, PricePanel):
        return _guard_price_panel(panel, names_by_market.get(panel.market.value, set()), cut, counts)
    if isinstance(panel, pd.DataFrame):
        if not {"date", "ticker"} <= set(panel.columns):
            raise TypeError("guard_panel: DataFrame에는 date, ticker 열이 필요합니다")
        xsh = set().union(*names_by_market.values()) if names_by_market else set()
        keys = panel["ticker"].astype(str).map(_name_key)
        is_x = keys.isin(xsh).to_numpy()
        counts["xsh_names"] = int(panel.loc[is_x, "ticker"].nunique())
        counts["xsh_rows"] = int(is_x.sum())
        late = (pd.to_datetime(panel["date"]) >= cut).to_numpy() if cut is not None else False
        counts["forward_rows"] = int((late & ~is_x).sum()) if cut is not None else 0
        out = panel.loc[~(is_x | late)].copy()
        _log_counts(counts, start)
        return out, counts
    raise TypeError(f"guard_panel: 지원하지 않는 패널 형식 {type(panel).__name__}")


def drop_xsh_names(frame: pd.DataFrame, market: str | Market, settings: Settings) -> tuple[pd.DataFrame, int]:
    """Rows of `frame` (``entity`` and / or ``ticker`` column) whose name is a registered XSH name of `market`
    removed; returns (frame, rows dropped). Dates are kept (use ``guard_panel`` for tuning-only runs)."""
    names_by_market, _ = tuning_exclusions(settings)
    xsh = names_by_market.get(_markets([market])[0], set())
    if not xsh or frame.empty:
        return frame, 0
    hit = np.zeros(len(frame), dtype=bool)
    for col in ("entity", "ticker"):
        if col in frame:
            hit |= frame[col].astype(str).map(_name_key).isin(xsh).to_numpy()
    if hit.any():
        log.warning("holdout guard: dropped %d rows of %d XSH names", int(hit.sum()), len(xsh))
    return frame.loc[~hit].copy(), int(hit.sum())


def _guard_price_panel(
    panel: PricePanel, xsh: set[str], cut: pd.Timestamp | None, counts: dict[str, int]
) -> tuple[PricePanel, dict[str, int]]:
    prices: dict[str, pd.DataFrame] = {}
    dropped_names: list[str] = []
    for t, px in panel.prices.items():
        if _name_key(t) in xsh:
            dropped_names.append(t)
            counts["xsh_rows"] += len(px)
            continue
        if cut is not None and len(px):
            keep = px.index < cut
            counts["forward_rows"] += int((~keep).sum())
            px = px.loc[keep]
        prices[t] = px
    counts["xsh_names"] = len(dropped_names)
    benchmarks: dict[str, pd.DataFrame] = {}
    for b, df in panel.benchmarks.items():
        if cut is not None and df is not None and len(df):
            keep = df.index < cut
            counts["benchmark_rows"] += int((~keep).sum())
            df = df.loc[keep]
        benchmarks[b] = df
    membership = panel.membership
    if membership is not None:
        mem = membership
        drop = mem["ticker"].astype(str).map(_name_key).isin(xsh).to_numpy()
        if cut is not None:
            drop = drop | (pd.to_datetime(mem["date"]) >= cut).to_numpy()
        counts["membership_rows"] = int(drop.sum())
        membership = mem.loc[~drop].reset_index(drop=True)
    notes = list(panel.notes)
    if counts["xsh_names"]:
        notes.append(
            f"[홀드아웃] 횡단면 홀드아웃(XSH) 종목 {counts['xsh_names']}개 제외 "
            f"({', '.join(sorted(dropped_names)[:8])}{' …' if len(dropped_names) > 8 else ''})"
        )
    if cut is not None and (counts["forward_rows"] or counts["benchmark_rows"] or counts["membership_rows"]):
        notes.append(
            f"[홀드아웃] 전진 홀드아웃(FTH) 시작 {cut.date()} 이후 가격 {counts['forward_rows']}행·"
            f"벤치마크 {counts['benchmark_rows']}행·멤버십 {counts['membership_rows']}행 제외"
        )
    out = replace(
        panel,
        prices=prices,
        benchmarks=benchmarks,
        ticker_benchmark={t: b for t, b in panel.ticker_benchmark.items() if t in prices},
        failed=dict(panel.failed),
        membership=membership,
        notes=notes,
    )
    _log_counts(counts, cut.date() if cut is not None else None)
    return out, counts


def _log_counts(counts: Mapping[str, int], start: date | None) -> None:
    if any(counts.values()):
        log.warning("holdout guard (FTH start %s): dropped %s", start, dict(counts))


# =============================================================================== helpers


def _jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple | set):
        return [_jsonable(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, str | int):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, pd.Timestamp | datetime | date):
        return x.isoformat()
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return v if math.isfinite(v) else None
