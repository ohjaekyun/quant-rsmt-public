"""Holdout protocol (docs/HOLDOUT.md): fingerprint, git state, append-only registry, frozen checks, looks, guard.

Every test points the registry at a temporary folder; the repository's own holdout/ files are never touched
(holdout.ensure_writable refuses them while pytest runs, which is also tested).
"""

from __future__ import annotations

import json
import subprocess
from datetime import date, datetime

import pandas as pd
import pytest
from conftest import make_prices

from quantcio import holdout
from quantcio.config import load_settings
from quantcio.data import base as data_base
from quantcio.data.base import PricePanel
from quantcio.models import Market

FIXED_NOW = datetime(2026, 6, 1, 15, 40, tzinfo=holdout.KST)


@pytest.fixture
def hs(tmp_path, monkeypatch):
    """Settings with a temporary registry / ledger and a clean, committed git state."""
    s = load_settings(
        data={"cache_dir": str(tmp_path / "cache")},
        holdout={"registry_path": str(tmp_path / "holdout" / "registry.json"), "ledger_dir": str(tmp_path / "led")},
    )
    monkeypatch.setattr(holdout, "git_state", lambda cwd=None: holdout.GitState("abc123", False))
    monkeypatch.setattr(holdout, "_now", lambda: FIXED_NOW)
    # other work may edit src/quantcio while the suite runs: pin the fingerprint (tested on temp trees above)
    monkeypatch.setattr(holdout, "code_fingerprint", lambda root=None: "a" * 16)
    return s


# ------------------------------------------------------------------------------ fingerprint


def _tree(root):
    files = {
        "__init__.py": "V = 1\n",
        "pipeline.py": "x = 1\n",
        "quant/rsmt.py": "def f():\n    return 1\n",
        "data/pit/kr.py": "N = 150\n",
        "data/kr.py": "SRC = 'naver'\n",
        "report/render.py": "HTML = '<p>'\n",
        "cli.py": "cmd = 1\n",
        "evaluation.py": "ev = 1\n",
        "holdout.py": "h = 1\n",
        "tracking.py": "t = 1\n",
        "quant/__pycache__/junk.py": "zzz\n",
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def test_fingerprint_changes_with_signal_code_not_with_report_or_cli(tmp_path):
    _tree(tmp_path)
    names = {p.relative_to(tmp_path).as_posix() for p in holdout.fingerprint_files(tmp_path)}
    assert names == {"__init__.py", "pipeline.py", "quant/rsmt.py", "data/pit/kr.py", "data/kr.py"}
    fp0 = holdout.code_fingerprint(tmp_path)
    assert len(fp0) == 16 and fp0 == holdout.code_fingerprint(tmp_path)

    for rel in (
        "report/render.py",
        "cli.py",
        "evaluation.py",
        "holdout.py",
        "tracking.py",
        "quant/__pycache__/junk.py",
    ):
        (tmp_path / rel).write_text("# changed\n", encoding="utf-8")
        assert holdout.code_fingerprint(tmp_path) == fp0, rel
    (tmp_path / "report" / "new_template.py").write_text("y = 2\n", encoding="utf-8")
    assert holdout.code_fingerprint(tmp_path) == fp0

    for rel in ("quant/rsmt.py", "data/pit/kr.py", "data/kr.py"):
        before = holdout.code_fingerprint(tmp_path)
        (tmp_path / rel).write_text("# signal changed\n", encoding="utf-8")
        assert holdout.code_fingerprint(tmp_path) != before, rel
    before = holdout.code_fingerprint(tmp_path)
    (tmp_path / "quant" / "new_feature.py").write_text("z = 3\n", encoding="utf-8")  # a new signal file counts
    assert holdout.code_fingerprint(tmp_path) != before

    with pytest.raises(holdout.HoldoutError):
        holdout.code_fingerprint(tmp_path / "missing")


def test_package_fingerprint_covers_providers_and_skips_non_signal_files():
    files = {p.relative_to(holdout.PACKAGE_ROOT).as_posix() for p in holdout.fingerprint_files()}
    assert {"pipeline.py", "data/kr.py", "data/us.py", "data/pit/__init__.py", "quant/rsmt.py"} <= files
    assert not {"cli.py", "evaluation.py", "holdout.py", "tracking.py"} & files
    assert not any(f.startswith("report/") for f in files)


def test_git_state_tolerates_no_repository_and_no_git(tmp_path, monkeypatch):
    gs = holdout.git_state(tmp_path)  # not a repository
    assert gs.commit is None and gs.dirty is None and gs.note.startswith("git 정보 없음")

    def no_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    gs = holdout.git_state(tmp_path)
    assert gs == holdout.GitState(None, None, gs.note) and "FileNotFoundError" in gs.note


# ------------------------------------------------------------------------------ registry


def test_register_forward_stores_the_frozen_state_and_rules(hs):
    e = holdout.register_forward(hs, ["KR", "US"])
    assert e["id"] == "FTH-20260601" and e["kind"] == "forward"
    assert e["start_date"] == "2026-06-01" and e["registered_at"] == FIXED_NOW.isoformat()
    assert e["config_digest"] == hs.digest() and e["code_fingerprint"] == "a" * 16
    assert (e["git_commit"], e["git_dirty"]) == ("abc123", False)
    assert (e["horizon_days"], e["hold_days"]) == (hs.horizon_days, hs.fusion.target_horizon_days)
    pm = e["primary_metrics"]
    assert pm["min_dates_for_verdict"] == 12 and pm["bh_q"] == 0.10
    assert [m["key"] for m in pm["metrics"]] == list(holdout.PRIMARY_KEYS)
    data = json.loads(holdout.registry_path(hs).read_text(encoding="utf-8"))
    assert data["version"] == 1 and data["entries"] == [e]
    # same day -> a distinct id; the first entry is untouched
    e2 = holdout.register_forward(hs, ["KR"])
    assert e2["id"] == "FTH-20260601-2"
    reg = holdout.load_registry(hs)
    assert reg.ids() == ["FTH-20260601", "FTH-20260601-2"] and reg.get("FTH-20260601") == e
    assert reg.active_forward("US")["id"] == "FTH-20260601"  # the second one covers KR only
    assert not list(holdout.registry_path(hs).parent.glob(".registry-*"))  # atomic write leaves no temp file
    with pytest.raises(holdout.HoldoutError, match="KR 또는 US"):
        holdout.register_forward(hs, ["JP"])


def test_register_requires_allow_dirty_for_dirty_or_unknown_git(hs, monkeypatch):
    monkeypatch.setattr(holdout, "git_state", lambda cwd=None: holdout.GitState("abc", True))
    with pytest.raises(holdout.HoldoutError, match="--allow-dirty"):
        holdout.register_forward(hs)
    e = holdout.register_forward(hs, allow_dirty=True)
    assert e["git_dirty"] is True and e["allow_dirty"] is True
    monkeypatch.setattr(holdout, "git_state", lambda cwd=None: holdout.GitState(None, None, "git 정보 없음(x)"))
    with pytest.raises(holdout.HoldoutError, match="git 커밋을 확인할 수 없습니다"):
        holdout.register_forward(hs)
    e = holdout.register_forward(hs, allow_dirty=True)
    assert e["git_commit"] is None and e["git_dirty"] is None and e["git_note"].startswith("git 정보 없음")


def test_registry_is_append_only(hs):
    holdout.register_forward(hs)
    path = holdout.registry_path(hs)

    reg = holdout.load_registry(hs)
    reg.entries[0]["config_digest"] = "tampered"
    with pytest.raises(holdout.HoldoutError, match="필드 변경"):
        reg.save()

    reg = holdout.load_registry(hs)
    reg.entries.clear()
    with pytest.raises(holdout.HoldoutError, match="삭제"):
        reg.save()

    reg = holdout.load_registry(hs)
    reg.append_look("FTH-20260601", {"at": "x", "valid": True})
    reg.save()
    reg = holdout.load_registry(hs)
    reg.entries[0]["looks"][0]["valid"] = False
    with pytest.raises(holdout.HoldoutError, match="기존 look"):
        reg.save()
    reg = holdout.load_registry(hs)
    reg.entries[0]["looks"].clear()
    with pytest.raises(holdout.HoldoutError, match="기존 look"):
        reg.save()
    assert json.loads(path.read_text(encoding="utf-8"))["entries"][0]["looks"] == [{"at": "x", "valid": True}]

    # a stale in-memory copy cannot overwrite an entry appended by another process
    stale = holdout.load_registry(hs)
    holdout.register_forward(hs, ["KR"])
    stale.append_look("FTH-20260601", {"at": "y"})
    with pytest.raises(holdout.HoldoutError, match="삭제"):
        stale.save()


def test_registry_validation_and_corrupt_file(hs):
    path = holdout.registry_path(hs)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(holdout.HoldoutError, match="읽기 실패"):
        holdout.load_registry(hs)
    path.write_text(json.dumps({"version": 2, "entries": []}), encoding="utf-8")
    with pytest.raises(holdout.HoldoutError, match="버전"):
        holdout.load_registry(hs)
    path.write_text(json.dumps({"version": 1, "entries": [{"id": "A", "kind": "forward"}]}), encoding="utf-8")
    with pytest.raises(holdout.HoldoutError, match="필수 필드"):
        holdout.load_registry(hs)
    assert holdout.Registry.load(path.parent / "absent.json").entries == []


def test_writes_to_the_repository_holdout_folder_are_refused_under_pytest(monkeypatch):
    monkeypatch.delenv(holdout.HOLDOUT_ROOT_ENV, raising=False)
    s = load_settings()
    assert holdout.registry_path(s) == holdout.PROJECT_ROOT / "holdout" / "registry.json"
    with pytest.raises(holdout.HoldoutError, match="테스트 실행 중"):
        holdout.ensure_writable(holdout.registry_path(s))
    monkeypatch.setenv(holdout.HOLDOUT_ROOT_ENV, "/tmp/elsewhere")
    assert str(holdout.registry_path(s)).startswith("/tmp/elsewhere")


# ------------------------------------------------------------------------------ frozen / looks / status


def test_check_frozen_detects_config_and_code_changes(hs, monkeypatch):
    e = holdout.register_forward(hs)
    assert holdout.check_frozen(e, hs) == (True, [])
    other = hs.model_copy(deep=True)
    other.fusion.min_buy_upside = 0.05
    ok, why = holdout.check_frozen(e, other)
    assert not ok and len(why) == 1 and why[0].startswith("설정 다이제스트 변경")
    monkeypatch.setattr(holdout, "code_fingerprint", lambda root=None: "0" * 16)
    ok, why = holdout.check_frozen(e, other)
    assert not ok and len(why) == 2 and why[1].startswith("신호 코드 지문 변경")


def test_xsh_register_looks_and_status(hs, monkeypatch):
    with pytest.raises(holdout.HoldoutError, match="튜닝에 쓰인 종목"):
        holdout.register_cross_sectional(hs, "KR", ["111111", "005930"], ["2018-01-01", "2026-05-29"], ["005930"])
    with pytest.raises(holdout.HoldoutError, match="기간"):
        holdout.register_cross_sectional(hs, "KR", ["111111"], ["2026-01-01", "2018-01-01"], [])
    e = holdout.register_cross_sectional(
        hs, Market.KR, ["222222", "111111", "111111"], [date(2018, 1, 1), "2026-05-29"], ["005930", "000660"]
    )
    assert e["id"] == "XSH-KR-20260601" and e["names"] == ["111111", "222222"]
    assert e["period"] == ["2018-01-01", "2026-05-29"] and e["excluded_tuning_names"] == ["000660", "005930"]
    st = {r["id"]: r for r in holdout.status(hs)}
    assert st[e["id"]]["pristine"] is True and st[e["id"]]["n_looks"] == 0 and st[e["id"]]["frozen_ok"]

    first = holdout.record_look(e["id"], hs, result_path="a.json", headline={"ic_t": 1.2})
    assert first["valid"] and first["pristine"] and not first["contaminated"]
    second = holdout.record_look(e["id"], hs, result_path="b.json")
    assert second["valid"] and second["pristine"] is False  # only the first look is pristine
    monkeypatch.setattr(holdout, "code_fingerprint", lambda root=None: "f" * 16)
    third = holdout.record_look(e["id"], hs, result_path="c.json", forced=True)
    assert not third["valid"] and third["contaminated"] and third["forced"] and third["pristine"] is False
    st = {r["id"]: r for r in holdout.status(hs)}[e["id"]]
    assert st["n_looks"] == 3 and st["n_valid_looks"] == 2 and st["pristine"] is False
    assert st["pristine_look"] == "a.json" and not st["frozen_ok"] and st["reasons"]
    assert [lk["result_path"] for lk in holdout.load_registry(hs).get(e["id"])["looks"]] == [
        "a.json",
        "b.json",
        "c.json",
    ]


def test_a_contaminated_first_look_makes_the_next_valid_look_non_pristine(hs, monkeypatch):
    e = holdout.register_cross_sectional(hs, "US", ["ZZZZ"], ["2018-01-01", "2026-05-29"], ["AAPL"])
    real = holdout.code_fingerprint
    monkeypatch.setattr(holdout, "code_fingerprint", lambda root=None: "e" * 16)
    assert holdout.record_look(e["id"], hs, forced=True)["contaminated"]
    monkeypatch.setattr(holdout, "code_fingerprint", real)
    look = holdout.record_look(e["id"], hs)
    assert look["valid"] and look["pristine"] is False
    fwd = holdout.register_forward(hs)
    assert holdout.record_look(fwd["id"], hs)["pristine"] is None  # not applicable to FTH


# ------------------------------------------------------------------------------ tuning names / guard


class _UniverseProvider:
    def __init__(self, market):
        self.market = market

    def universe(self, n):
        if self.market is Market.KR:
            return [f"{900000 + i}" for i in range(n)]
        return ["brk.b", "XOM", *(f"U{i:03d}" for i in range(105))]


class _PIT:
    """Minimal PIT universe stand-in: only ``membership`` is read by tuning_names."""

    def __init__(self, rows):
        self.membership = pd.DataFrame(rows, columns=["date", "entity", "ticker"])


def test_tuning_names_union(hs, monkeypatch):
    monkeypatch.setattr(data_base, "get_provider", lambda m, s, c: _UniverseProvider(m))
    kr = holdout.tuning_names("KR", hs, None, pit=False)
    assert len(kr) == 150 + len(holdout.DOC_TICKERS["KR"]) and "900149" in kr and "373220" in kr
    us = holdout.tuning_names(Market.US, hs, None, pit=False)
    assert "BRK-B" in us and "XOM" in us and "TSLA" in us and len(us) == 107 + 5  # BRK-B is in both lists

    # the PIT modeling universe: every entity ever a member (spell suffix dropped) and every ticker it carried
    pit = _PIT([("2016-01-29", "123456#1", "123456"), ("2020-01-31", "654321", "654321")])
    kr_pit = holdout.tuning_names("KR", hs, None, pit_universe=pit)
    assert {"123456", "654321"} <= set(kr_pit) and "123456#1" not in kr_pit
    us_pit = holdout.tuning_names(
        "US",
        hs,
        None,
        pit_universe=_PIT([("2013-01-31", "CIK0000000001", "OLD.A"), ("2020-01-31", "CIK0000000001", "NEWA")]),
    )
    assert {"CIK0000000001", "OLD-A", "NEWA"} <= set(us_pit)
    assert holdout.pit_tuning_universe("KR", hs) == (200, date(2016, 1, 1))
    assert holdout.pit_tuning_universe("US", hs) == (500, date(2012, 1, 1))

    def down(m, s, c):
        raise ConnectionError("naver down")

    monkeypatch.setattr(data_base, "get_provider", down)
    with pytest.raises(holdout.HoldoutError, match="튜닝 유니버스 로드 실패"):
        holdout.tuning_names("KR", hs, None, pit=False)
    monkeypatch.setattr(data_base, "get_provider", lambda m, s, c: _UniverseProvider(m))

    def pit_down(*a, **k):
        raise OSError("marcap unavailable")

    import quantcio.data.pit as pit_mod

    monkeypatch.setattr(pit_mod, "load_pit_universe", pit_down)
    with pytest.raises(holdout.HoldoutError, match="튜닝 유니버스 로드 실패"):  # an incomplete list never registers
        holdout.tuning_names("KR", hs, None)


def test_xsh_entries_store_the_xsh_rules_and_confirmation_needs_a_pristine_frozen_look(hs, monkeypatch):
    e = holdout.register_cross_sectional(hs, "KR", ["777777", "888888"], ["2019-09-01", "2026-09-01"], ["005930"])
    pm = e["primary_metrics"]
    assert pm["kind"] == "cross_sectional" and pm["target_auc"] == 0.55
    assert [m["key"] for m in pm["metrics"]] == list(holdout.XSH_PRIMARY_KEYS)
    assert holdout.xsh_confirmation("KR", hs) is None  # never looked at
    head = {"confirmed": True, "auc_median": 0.56, "auc_median_lo": 0.52, "ic": 0.1, "n_dates": 80}
    holdout.record_look(e["id"], hs, result_path="x.json", headline=head)
    conf = holdout.xsh_confirmation("KR", hs)
    assert conf is not None and conf["id"] == e["id"] and conf["auc_median"] == 0.56
    assert holdout.xsh_confirmation("US", hs) is None
    # non-model code changed (e.g. the LLM layer): the model scope still matches, the confirmation holds
    monkeypatch.setattr(holdout, "code_fingerprint", lambda root=None: "b" * 16)
    assert holdout.xsh_confirmation("KR", hs) is not None
    # the model code changed: the confirmation lapses (the served model is not the one that was checked)
    real_scope = holdout.model_scope_fingerprint
    monkeypatch.setattr(holdout, "model_scope_fingerprint", lambda root=None, market=None: "0" * 16)
    assert holdout.xsh_confirmation("KR", hs) is None
    monkeypatch.setattr(holdout, "model_scope_fingerprint", real_scope)
    monkeypatch.setattr(holdout, "code_fingerprint", lambda root=None: "a" * 16)
    # an unconfirmed first look never confirms, and a later look is never pristine
    e2 = holdout.register_cross_sectional(hs, "US", ["ZZZZ"], ["2019-09-01", "2026-09-01"], ["AAPL"])
    holdout.record_look(e2["id"], hs, headline={"confirmed": False})
    holdout.record_look(e2["id"], hs, headline={"confirmed": True})
    assert holdout.xsh_confirmation("US", hs) is None
    # "confirmed" by BH but with a lower bound <= 0.5 (weak evidence): never unlocks full serving
    e3 = holdout.register_cross_sectional(hs, "US", ["YYYY"], ["2019-09-01", "2026-09-01"], ["AAPL"])
    holdout.record_look(e3["id"], hs, headline={"confirmed": True, "auc_median": 0.508, "auc_median_lo": 0.496})
    assert holdout.xsh_confirmation("US", hs) is None
    # a broken registry is "not confirmed", never an exception on the serving path
    holdout.registry_path(hs).write_text("{broken", encoding="utf-8")
    assert holdout.xsh_confirmation("KR", hs) is None


def test_drop_xsh_names_keeps_dates(hs):
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-01", "2026-05-01", "2027-01-04"]),
            "entity": ["111111#1", "444444", "444444"],
            "ticker": ["111111", "444444", "444444"],
        }
    )
    same, n0 = holdout.drop_xsh_names(frame, "KR", hs)
    assert n0 == 0 and len(same) == 3
    holdout.register_cross_sectional(hs, "KR", ["111111"], ["2018-01-01", "2026-05-29"], [])
    holdout.register_forward(hs)
    kept, n = holdout.drop_xsh_names(frame, "KR", hs)
    assert n == 1 and kept["entity"].tolist() == ["444444", "444444"]  # the post-FTH row stays (serving runs)


def _panel() -> PricePanel:
    px = make_prices(n=60, end="2026-06-30")  # 2026-04-09 .. 2026-06-30 business days
    bench = make_prices(n=60, end="2026-06-30", seed=9)
    tickers = ["333333", "111111", "222222#1", "444444"]
    membership = pd.DataFrame(
        [(d, t) for d in pd.to_datetime(["2026-04-30", "2026-05-29", "2026-06-30"]) for t in tickers],
        columns=["date", "ticker"],
    )
    return PricePanel(
        Market.KR,
        {t: px.copy() for t in tickers},
        {"KOSPI": bench},
        {t: "KOSPI" for t in tickers},
        membership=membership,
        notes=["원래 노트"],
    )


def test_guard_panel_drops_xsh_names_and_forward_rows(hs):
    panel = _panel()
    same, counts = holdout.guard_panel(panel, hs)  # nothing registered: unchanged, zero counts
    assert not any(counts.values()) and set(same.prices) == set(panel.prices)

    holdout.register_cross_sectional(hs, "KR", ["111111", "222222"], ["2018-01-01", "2026-05-29"], [])
    holdout.register_cross_sectional(hs, "US", ["333333"], ["2018-01-01", "2026-05-29"], [])  # other market
    holdout.register_forward(hs)  # start 2026-06-01
    names, start = holdout.tuning_exclusions(hs)
    assert names == {"KR": {"111111", "222222"}, "US": {"333333"}} and start == date(2026, 6, 1)

    out, counts = holdout.guard_panel(panel, hs)
    n_late = int((panel.prices["333333"].index >= "2026-06-01").sum())
    assert n_late == 22
    assert set(out.prices) == {"333333", "444444"} and set(out.ticker_benchmark) == {"333333", "444444"}
    assert counts == {
        "xsh_names": 2,
        "xsh_rows": 120,
        "forward_rows": 2 * n_late,
        "benchmark_rows": n_late,
        "membership_rows": 2 * 3 + 2,  # 2 XSH names x 3 dates + the 2026-06-30 rows of the 2 kept names
    }
    assert all(px.index.max() < pd.Timestamp("2026-06-01") for px in out.prices.values())
    assert out.benchmarks["KOSPI"].index.max() < pd.Timestamp("2026-06-01")
    assert set(out.membership["ticker"]) == {"333333", "444444"} and out.membership["date"].max() < pd.Timestamp(
        "2026-06-01"
    )
    assert out.notes[0] == "원래 노트" and any("XSH" in n for n in out.notes) and any("FTH" in n for n in out.notes)
    assert len(panel.prices) == 4 and len(panel.prices["333333"]) == 60 and panel.notes == ["원래 노트"]  # untouched

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-01", "2026-05-01", "2026-06-02", "2026-06-02"]),
            "ticker": ["111111", "444444", "444444", "555555"],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    kept, c2 = holdout.guard_panel(frame, hs)
    assert kept["y"].tolist() == [2.0] and c2["xsh_rows"] == 1 and c2["forward_rows"] == 2
    with pytest.raises(TypeError):
        holdout.guard_panel([1, 2], hs)


# ------------------------------------------------------------------------------ model scope (XSH confirmation)


def test_model_scope_files_skip_llm_fusion_pipeline_and_config(tmp_path):
    _tree(tmp_path)
    for rel in ("llm/news.py", "fusion/signal.py", "pipeline.py", "config.py", "modeling/serve.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x = 1\n", encoding="utf-8")
    rels = {p.relative_to(tmp_path).as_posix() for p in holdout.model_scope_files(tmp_path)}
    assert {"modeling/serve.py", "data/pit/kr.py", "quant/rsmt.py"} <= rels
    assert not rels & {"llm/news.py", "fusion/signal.py", "pipeline.py", "config.py", "cli.py", "report/render.py"}
    fp = holdout.model_scope_fingerprint(tmp_path)
    (tmp_path / "llm" / "news.py").write_text("x = 2\n", encoding="utf-8")  # an LLM change: same model scope
    assert holdout.model_scope_fingerprint(tmp_path) == fp
    (tmp_path / "modeling" / "serve.py").write_text("x = 2\n", encoding="utf-8")
    assert holdout.model_scope_fingerprint(tmp_path) != fp


def test_model_scope_digest_ignores_llm_settings_only():
    base = load_settings()
    d0 = holdout.model_scope_digest(base)
    assert holdout.model_scope_digest(load_settings(llm={"effort": {"critic": "low"}})) == d0
    assert holdout.model_scope_digest(load_settings(backtest={"kr_universe_n": 150})) != d0
    assert holdout.model_scope_digest(load_settings(report={"language": "en"})) == d0


def test_attestation_keeps_the_xsh_confirmation_across_non_model_changes(hs, monkeypatch):
    monkeypatch.setattr(holdout, "model_scope_fingerprint", lambda root=None, market=None: "m" * 16)
    e = holdout.register_cross_sectional(hs, "KR", ["777777"], ["2019-09-01", "2026-09-01"], [])
    reg = holdout.load_registry(hs)
    raw = reg.get(e["id"])
    raw.pop("model_scope_fingerprint")  # an entry registered before model scopes were stored
    raw.pop("model_scope_digest")
    holdout.registry_path(hs).write_text(json.dumps(reg.data), encoding="utf-8")
    holdout.record_look(e["id"], hs, headline={"confirmed": True, "auc_median": 0.545, "auc_median_lo": 0.52})
    assert holdout.model_scope_frozen(holdout.load_registry(hs).get(e["id"]), hs)[0] is False  # nothing recorded
    # attesting needs the FULL freeze
    monkeypatch.setattr(holdout, "code_fingerprint", lambda root=None: "b" * 16)
    with pytest.raises(holdout.HoldoutError, match="전체 동결"):
        holdout.attest_model_scope(e["id"], hs)
    monkeypatch.setattr(holdout, "code_fingerprint", lambda root=None: "a" * 16)
    att = holdout.attest_model_scope(e["id"], hs)
    assert att["model_scope_fingerprint"] == "m" * 16 and att["code_fingerprint"] == "a" * 16
    with pytest.raises(holdout.HoldoutError, match="이미"):
        holdout.attest_model_scope(e["id"], hs)
    # an LLM / pipeline change: the full freeze breaks, the model scope holds -> still confirmed
    monkeypatch.setattr(holdout, "code_fingerprint", lambda root=None: "c" * 16)
    assert holdout.check_frozen(holdout.load_registry(hs).get(e["id"]), hs)[0] is False
    assert holdout.xsh_confirmation("KR", hs) is not None
    row = next(r for r in holdout.status(hs) if r["id"] == e["id"])
    assert row["frozen_ok"] is False and row["model_scope_ok"] is True
    # a model change voids it
    monkeypatch.setattr(holdout, "model_scope_fingerprint", lambda root=None, market=None: "n" * 16)
    assert holdout.xsh_confirmation("KR", hs) is None
    # a new XSH entry stores its model scope directly
    e2 = holdout.register_cross_sectional(hs, "US", ["ZZZZ"], ["2019-09-01", "2026-09-01"], [])
    assert e2["model_scope_fingerprint"] == "n" * 16 and e2["model_scope_digest"] == holdout.model_scope_digest(hs)
    # attestations are append-only and validated
    reg = holdout.load_registry(hs)
    reg.data["attestations"][0]["model_scope_fingerprint"] = "z" * 16
    with pytest.raises(holdout.HoldoutError, match="attestation"):
        reg.save()
    bad = holdout.load_registry(hs).data
    bad["attestations"].append({"entry_id": "NOPE"})
    with pytest.raises(holdout.HoldoutError, match="attestation"):
        holdout.validate_registry(bad)


def test_model_scope_v2_is_per_market_and_attested_through_a_matching_v1(tmp_path, hs, monkeypatch):
    _tree(tmp_path)
    for rel in ("data/pit/us.py", "data/pit/us_prices.py", "data/pit/kr_fund.py", "modeling/serve.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x = 1\n", encoding="utf-8")
    kr = {p.relative_to(tmp_path).as_posix() for p in holdout.model_scope_files(tmp_path, "KR")}
    us = {p.relative_to(tmp_path).as_posix() for p in holdout.model_scope_files(tmp_path, "US")}
    assert "data/pit/kr.py" in kr and "data/pit/us.py" not in kr and "data/pit/us_prices.py" not in kr
    assert "data/pit/us.py" in us and "data/pit/kr_fund.py" not in us and "modeling/serve.py" in us
    fp_kr = holdout.model_scope_fingerprint(tmp_path, "KR")
    (tmp_path / "data/pit/us.py").write_text("x = 2\n", encoding="utf-8")  # a US-only data fix
    assert holdout.model_scope_fingerprint(tmp_path, "KR") == fp_kr
    assert holdout.model_scope_fingerprint(tmp_path, "US") != holdout.model_scope_fingerprint(tmp_path, "KR")

    # chain: an entry with a v1 scope gets v2 while v1 still matches, even after the full freeze broke
    scopes = {None: "v1" * 8, "KR": "k2" * 8, "US": "u2" * 8}
    monkeypatch.setattr(holdout, "model_scope_fingerprint", lambda root=None, market=None: scopes[market])
    e = holdout.register_cross_sectional(hs, "KR", ["777777"], ["2019-09-01", "2026-09-01"], [])
    assert e["model_scope_version"] == 2 and e["model_scope_fingerprint"] == "k2" * 8  # new entries: v2 directly
    reg = holdout.load_registry(hs)
    raw = reg.get(e["id"])
    for k in ("model_scope_version", "model_scope_fingerprint", "model_scope_digest"):
        raw.pop(k)
    holdout.registry_path(hs).write_text(json.dumps(reg.data), encoding="utf-8")
    holdout.attest_model_scope(e["id"], hs, version=1)
    monkeypatch.setattr(holdout, "code_fingerprint", lambda root=None: "c" * 16)  # e.g. an LLM change
    att = holdout.attest_model_scope(e["id"], hs)  # v2 through the matching v1
    assert att["scope_version"] == 2 and att["market"] == "KR" and att["basis"] == "model_scope_v1"
    with pytest.raises(holdout.HoldoutError, match="이미"):
        holdout.attest_model_scope(e["id"], hs)
    scopes[None] = "zz" * 8  # a US data change: v1 breaks, the KR v2 scope holds
    ok, _ = holdout.model_scope_frozen(holdout.load_registry(hs).get(e["id"]), hs)
    assert ok
    scopes["KR"] = "kk" * 8  # a KR model change breaks it
    ok, reasons = holdout.model_scope_frozen(holdout.load_registry(hs).get(e["id"]), hs)
    assert not ok and "attestation v2" in reasons[0]
