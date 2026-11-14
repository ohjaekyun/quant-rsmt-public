"""`quantcio report` ticker selection: --tickers, --watchlist, the default watchlist and --top / --bottom model picks
(offline: the serving artifact and run_report are stubbed)."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

import quantcio.cli as cli
from quantcio.config import load_settings
from quantcio.modeling import serve
from quantcio.models import Market


@pytest.fixture
def ts(tmp_path, monkeypatch):
    s = load_settings(data={"cache_dir": str(tmp_path / "cache")})
    monkeypatch.setattr(cli, "load_settings", lambda config=None: s)
    monkeypatch.setattr(cli, "_setup_logging", lambda settings, verbose: tmp_path / "test.log")
    monkeypatch.setattr(cli, "WATCHLIST_DEFAULT", tmp_path / "watchlist.txt")  # absent unless a test writes it
    # `report --no-record` sets QUANTCIO_NO_RECORD in os.environ for the process: restore it after the test, or the
    # ledger tests that run later would see recording switched off
    # (setenv records the original state - even "unset" - so the undo restores it; delenv alone records nothing
    # when the variable is absent)
    monkeypatch.setenv("QUANTCIO_NO_RECORD", "")
    monkeypatch.delenv("QUANTCIO_NO_RECORD")
    return s


def _artifact(market: str, tickers: list[str], skill_ok: bool = True) -> SimpleNamespace:
    n = len(tickers)
    latest = pd.DataFrame(
        {
            "ticker": tickers,
            "name": [f"name-{t}" for t in tickers],
            "score": [float(n - i) for i in range(n)],  # already best-first
            "percentile": [(n - i) / n for i in range(n)],
        }
    ).sample(frac=1.0, random_state=0)  # the artifact order must not matter
    return SimpleNamespace(
        market=market, latest=latest, as_of=pd.Timestamp.today().normalize(), skill_ok=skill_ok, calibration={}
    )


@pytest.fixture
def arts(monkeypatch):
    table = {
        "KR": _artifact("KR", ["000001", "000002", "000003", "000004", "000005"]),
        "US": _artifact("US", ["AAA", "BRK.B", "CCC", "DDD"], skill_ok=False),
    }
    monkeypatch.setattr(serve, "load_artifact", lambda m, s, c: table.get(Market(m).value))
    monkeypatch.setattr(
        serve,
        "served_state",
        lambda art, s: {"skill_ok": art.skill_ok, "multiplier": 0.86 if art.skill_ok else 0.2},
    )
    return table


def test_read_watchlist_parses_commas_spaces_and_comments(tmp_path):
    f = tmp_path / "w.txt"
    f.write_text("# header\n005930, 000660  # memory\n\nAAPL msft\n  # only a comment\nbrk.b\n", encoding="utf-8")
    assert cli.read_watchlist(f) == ["005930", "000660", "AAPL", "msft", "brk.b"]
    with pytest.raises(cli.TickerSelectionError):
        cli.read_watchlist(tmp_path / "missing.txt")


def test_model_picks_top_and_bottom_per_market(ts, arts):
    picks, notes = cli.model_picks([Market.KR, Market.US], 2, 1, ts, None)
    kr = [(p["side"], p["ticker"]) for p in picks if p["market"] == "KR"]
    us = [(p["side"], p["ticker"]) for p in picks if p["market"] == "US"]
    assert kr == [("top", "000001"), ("top", "000002"), ("bottom", "000005")]
    assert us == [("top", "AAA"), ("top", "BRK-B"), ("bottom", "DDD")]  # tickers canonicalised
    assert len(notes) == 1 and "US" in notes[0] and "참고용" in notes[0]  # no-skill market is flagged
    # a small universe: bottom never repeats a top pick
    picks, _ = cli.model_picks([Market.US], 3, 3, ts, None)
    assert [p["ticker"] for p in picks] == ["AAA", "BRK-B", "CCC", "DDD"]


def test_missing_artifact_is_a_clear_error(ts, monkeypatch):
    monkeypatch.setattr(serve, "load_artifact", lambda m, s, c: None)
    with pytest.raises(cli.TickerSelectionError, match="modeling select"):
        cli.model_picks([Market.KR], 3, 0, ts, None)


def test_resolve_merges_explicit_watchlist_and_picks_in_order(ts, arts, tmp_path):
    w = tmp_path / "mine.txt"
    w.write_text("MSFT 000001\n", encoding="utf-8")
    out, picks, _ = cli.resolve_report_tickers("aapl,MSFT", 2, 0, "KR", w, ts)
    assert out == ["aapl", "MSFT", "000001", "000002"]  # de-duplicated by canonical ticker, first spelling kept
    assert [p["ticker"] for p in picks] == ["000001", "000002"]
    only_model, _, _ = cli.resolve_report_tickers(None, 1, 1, "KR,US", None, ts)
    assert only_model == ["000001", "000005", "AAA", "DDD"]


def test_default_watchlist_only_when_nothing_else_is_given(ts, tmp_path):
    with pytest.raises(cli.TickerSelectionError, match="종목을 지정하세요"):
        cli.resolve_report_tickers(None, 0, 0, "KR,US", None, ts)
    cli.WATCHLIST_DEFAULT.write_text("005930 AAPL\n", encoding="utf-8")
    out, _, notes = cli.resolve_report_tickers(None, 0, 0, "KR,US", None, ts)
    assert out == ["005930", "AAPL"] and any("관심 종목 파일" in n for n in notes)
    out, _, _ = cli.resolve_report_tickers("NVDA", 0, 0, "KR,US", None, ts)
    assert out == ["NVDA"]  # an explicit list never pulls in the default file
    cli.WATCHLIST_DEFAULT.write_text("# nothing yet\n", encoding="utf-8")
    with pytest.raises(cli.TickerSelectionError, match="종목이 없습니다"):
        cli.resolve_report_tickers(None, 0, 0, "KR,US", None, ts)


def test_bad_options(ts, arts):
    with pytest.raises(cli.TickerSelectionError, match="--top"):
        cli.resolve_report_tickers(None, 21, 0, "KR", None, ts)
    with pytest.raises(cli.TickerSelectionError, match="--bottom"):
        cli.resolve_report_tickers(None, 0, -1, "KR", None, ts)
    with pytest.raises(cli.TickerSelectionError, match="--markets"):
        cli.resolve_report_tickers(None, 2, 0, "JP", None, ts)
    many = ",".join(f"{100000 + i}" for i in range(ts.report.compact_max_names + 1))
    _, _, notes = cli.resolve_report_tickers(many, 0, 0, "KR", None, ts)
    assert any("쪽수를 늘려" in n for n in notes)  # auto-extended, not compacted
    ts.report.auto_extend = False
    _, _, notes = cli.resolve_report_tickers(many, 0, 0, "KR", None, ts)
    assert any("압축" in n for n in notes)


def test_cli_report_passes_the_resolved_list(ts, arts, monkeypatch):
    import quantcio.pipeline as pipeline

    seen = {}

    async def fake_run_report(tickers, settings, **kw):
        seen["tickers"] = list(tickers)
        raise pipeline.PipelineError("stop after the ticker list")

    monkeypatch.setattr(pipeline, "run_report", fake_run_report)
    runner = CliRunner()
    res = runner.invoke(cli.app, ["report", "--top", "2", "--markets", "KR", "-t", "AAPL", "--no-llm", "--no-record"])
    assert res.exit_code == 1 and seen["tickers"] == ["AAPL", "000001", "000002"]
    assert "모델 자동 선정 종목" in res.stdout and "리포트 종목 (3)" in res.stdout
    res = runner.invoke(cli.app, ["report", "--no-llm", "--no-record"])  # nothing given, no default file
    assert res.exit_code == 2 and "종목을 지정하세요" in res.output
    res = runner.invoke(cli.app, ["report", "--top", "30", "--no-record"])
    assert res.exit_code == 2
