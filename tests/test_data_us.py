"""Tests for quantcio.data.us.

Offline tests run on saved fixtures in tests/fixtures/us/. Real (trimmed) data: NVDA SEC submissions,
filing index, 8-K, XBRL companyfacts, the AAPL EX-99.1 and the AAPL yfinance .info. Synthetic but
structurally faithful: the 10-Q/10-K, the Yahoo chart/Nasdaq payloads, the news feeds, the Motley Fool
pages and Naver FX. An autouse guard fails any offline test that touches the network. Tests marked
``live`` hit Yahoo / SEC / Google / fool.com.
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from quantcio.data import us
from quantcio.data.base import clean_prices
from quantcio.data.cache import DiskCache
from quantcio.models import PRICE_COLUMNS, FilingItem, Instrument, Market

FIX = Path(__file__).parent / "fixtures" / "us"
LIVE_SEC_UA = "QuantCIO Research research@example.com"
OFFLINE_SEC_UA = "QuantCIO Offline Test offline@example.com"  # never sent: network is blocked offline
NOW = pd.Timestamp("2026-09-24T12:00:00Z")  # fixtures were captured on this day


def fx_bytes(name: str) -> bytes:
    return (FIX / name).read_bytes()


def fx_json(name: str):
    return json.loads((FIX / name).read_text())


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def nvda_instrument() -> Instrument:
    return Instrument(
        ticker="NVDA",
        market=Market.US,
        name="NVIDIA Corporation",
        exchange="NASDAQ",
        currency="USD",
        sector="Technology",
        yf_symbol="NVDA",
        benchmark="SPY",
        benchmark_name="S&P 500 (SPY)",
    )


def aapl_instrument() -> Instrument:
    return Instrument(
        ticker="AAPL",
        market=Market.US,
        name="Apple Inc.",
        exchange="NASDAQ",
        currency="USD",
        sector="Technology",
        yf_symbol="AAPL",
        benchmark="SPY",
        benchmark_name="S&P 500 (SPY)",
    )


@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):
    """Offline tests must never reach the network (fetchers are faked per test)."""
    if request.node.get_closest_marker("live"):
        return

    def boom(*args, **kwargs):
        raise AssertionError(f"offline test attempted network access: {args[1:2]}")

    monkeypatch.setattr(requests.sessions.Session, "request", boom)
    monkeypatch.setattr(us.yf, "Ticker", boom)


def _repair_offline(*args, **kwargs):
    raise RuntimeError("offline: repair source not faked")


@pytest.fixture
def provider(settings, tmp_path, monkeypatch) -> us.USProvider:
    settings.data.sec_user_agent = OFFLINE_SEC_UA
    # session-repair sources answer "unavailable" by default (class-level: covers every provider a test
    # builds); the repair tests patch them per instance
    monkeypatch.setattr(us.USProvider, "_repair_chart", _repair_offline)
    monkeypatch.setattr(us.USProvider, "_repair_nasdaq", _repair_offline)
    p = us.USProvider(settings, DiskCache(tmp_path / "cache"))
    # the earnings-calendar page is a network cross-check; tests of that path re-patch it
    monkeypatch.setattr(p, "_earnings_calendar", lambda sym, tz: {"upcoming": [], "reported": []})
    return p


# =============================================================================== utils / prices


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("aapl", "AAPL"),
        (" BRK.B ", "BRK-B"),
        ("BRK/B", "BRK-B"),
        ("brk-b", "BRK-B"),
        ("GOOGL", "GOOGL"),
    ],
)
def test_normalise_us_ticker(raw, expected):
    assert us.normalise_us_ticker(raw) == expected


@pytest.mark.parametrize(
    "now,expected",
    [
        ("2026-09-24T14:00:00Z", date(2026, 9, 23)),  # Thu 10:00 ET, session running
        ("2026-09-24T20:25:00Z", date(2026, 9, 24)),  # Thu 16:25 ET, closed + buffer
        ("2026-09-26T15:00:00Z", date(2026, 9, 25)),  # Saturday
        ("2026-09-28T13:00:00Z", date(2026, 9, 25)),  # Monday 09:00 ET
    ],
)
def test_last_completed_us_session(now, expected):
    assert us.last_completed_us_session(pd.Timestamp(now)) == expected


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_drop_unfinished_bar_simulated_clock():
    idx = pd.DatetimeIndex(["2026-09-22", "2026-09-23", "2026-09-24"])
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    ny = "America/New_York"
    assert us.drop_unfinished_bar(df, ny, now="2026-09-24T14:00:00Z").index[-1] == pd.Timestamp("2026-09-23")
    assert us.drop_unfinished_bar(df, ny, now="2026-09-24T20:05:00Z").index[-1] == pd.Timestamp("2026-09-23")
    assert us.drop_unfinished_bar(df, ny, now="2026-09-24T20:25:00Z").index[-1] == pd.Timestamp("2026-09-24")
    # half-day: regular session ends 13:00 ET -> final at 13:30 ET
    half = pd.Timestamp("2026-09-24T17:00:00Z")
    assert us.drop_unfinished_bar(df, ny, regular_end=half, now="2026-09-24T17:30:00Z").index[-1] == pd.Timestamp(
        "2026-09-24"
    )
    # regular_end given as epoch seconds (Yahoo chart meta)
    assert len(us.drop_unfinished_bar(df, ny, regular_end=int(half.timestamp()), now="2026-09-24T17:05:00Z")) == 2
    # last bar is not today -> untouched
    assert len(us.drop_unfinished_bar(df, ny, now="2026-09-25T14:00:00Z")) == 3


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_parse_yahoo_chart_adjusts_and_drops_live_bar():
    res = fx_json("yahoo_chart_aapl.json")
    df = us.parse_yahoo_chart(res, now="2026-09-24T15:00:00Z")  # 11:00 ET: 09-24 bar still forming
    assert list(df.columns) == PRICE_COLUMNS and (df.dtypes == "float64").all()
    assert df.index[-1] == pd.Timestamp("2026-09-23") and df.index.tz is None
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df.loc["2026-09-23", "close"] == pytest.approx(337.02 * 0.999)
    assert df.loc["2026-09-23", "open"] == pytest.approx(334.9 * 0.999)  # OHLC scaled by adjclose/close
    assert df.loc["2026-09-21", "volume"] == 0.0  # None volume -> 0, row kept
    assert df.attrs["adjusted"] is True and df.attrs["source"] == "yahoo_chart"

    after_close = us.parse_yahoo_chart(res, now="2026-09-24T20:30:00Z")
    assert after_close.index[-1] == pd.Timestamp("2026-09-24")
    raw = us.parse_yahoo_chart(res, now="2026-09-24T20:30:00Z", adjust=False)
    assert raw.loc["2026-09-23", "close"] == pytest.approx(337.02) and raw.attrs["adjusted"] is False
    splits = us.chart_splits(res)
    assert splits.to_dict() == {pd.Timestamp("2020-08-31"): 4.0}


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_frame_from_yf_history_contract():
    idx = pd.DatetimeIndex(pd.to_datetime(["2026-09-22", "2026-09-23", "2026-09-24"])).tz_localize("America/New_York")
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [1.5, 2.5, 3.5],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.2, 2.2, 3.2],
            "Volume": [10, 20, 30],
        },
        index=idx,
    )
    reg_end = pd.Timestamp("2026-09-24 16:00", tz="America/New_York")
    df = us.frame_from_yf_history(raw, reg_end, now="2026-09-24T15:00:00Z")
    assert list(df.columns) == PRICE_COLUMNS and (df.dtypes == "float64").all()
    assert df.index.tz is None and df.index[-1] == pd.Timestamp("2026-09-23")
    assert df.attrs["adjusted"] is True
    with pytest.raises(ValueError):
        us.frame_from_yf_history(raw.iloc[0:0])


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_parse_nasdaq_rows_is_flagged_unadjusted():
    df = us.parse_nasdaq_rows(fx_json("nasdaq_rows_aapl.json"), now="2026-09-24T15:00:00Z")
    assert df.index[0] == pd.Timestamp("2026-09-21") and df.index[-1] == pd.Timestamp("2026-09-23")
    assert df.loc["2026-09-23", "close"] == 337.02 and df.loc["2026-09-23", "volume"] == 31_658_820
    assert df.loc["2026-09-22", "volume"] == 0.0  # '--' -> 0
    assert df.attrs["adjusted"] is False and df.attrs["split_adjusted"] is True


def _frame(n=300, close=100.0, source="yfinance", adjusted=True):
    idx = pd.bdate_range(end="2026-09-23", periods=n)
    df = pd.DataFrame({c: np.full(n, close) for c in PRICE_COLUMNS}, index=idx).astype("float64")
    df.attrs.update(source=source, adjusted=adjusted)
    return df


def test_price_chain_falls_back_to_nasdaq_and_flags_it(provider, monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("HTTP 429 Too Many Requests")

    monkeypatch.setattr(provider, "_fetch_yf_frame", fail)
    monkeypatch.setattr(provider, "_fetch_chart_frame", fail)
    monkeypatch.setattr(
        provider,
        "_fetch_nasdaq_frame",
        lambda s, st, cls: _frame(source="nasdaq", adjusted=False),
    )
    inst = aapl_instrument()
    df = provider.prices(inst, date(2019, 9, 1))
    assert df.attrs["adjusted"] is False and df.attrs["source"] == "nasdaq"
    assert len(df.attrs["failed_sources"]) == 2
    assert clean_prices(df).attrs["adjusted"] is False  # survives the shared loader
    notes = provider.drain_notes("AAPL")
    assert any("Nasdaq" in n and "배당 미조정" in n for n in notes)
    # cached frame keeps the flag -> the note is raised again on a cache hit
    monkeypatch.setattr(provider, "_fetch_nasdaq_frame", fail)
    assert provider.prices(inst, date(2019, 9, 1)).attrs["adjusted"] is False
    assert any("Nasdaq" in n for n in provider.drain_notes("AAPL"))


def test_price_chain_all_sources_fail(provider, monkeypatch):
    for name in ("_fetch_yf_frame", "_fetch_chart_frame", "_fetch_nasdaq_frame"):
        monkeypatch.setattr(provider, name, lambda *a, **k: (_ for _ in ()).throw(ValueError("down")))
    with pytest.raises(RuntimeError, match="yfinance.*yahoo_chart.*nasdaq"):
        provider.prices(aapl_instrument(), date(2019, 9, 1))


def test_price_chain_primary_ok_no_notes(provider, monkeypatch):
    monkeypatch.setattr(us, "_now_utc", lambda: NOW)
    monkeypatch.setattr(provider, "_fetch_yf_frame", lambda s, st: _frame())
    df = provider.benchmark(aapl_instrument(), date(2019, 9, 1))
    assert df.attrs["source"] == "yfinance" and provider.drain_notes("AAPL") == []


def test_stale_price_data_is_noted(provider, monkeypatch):
    monkeypatch.setattr(us, "_now_utc", lambda: NOW)
    old = _frame()
    old.index = old.index - pd.Timedelta(30, "D")
    monkeypatch.setattr(provider, "_fetch_yf_frame", lambda s, st: old)
    provider.prices(aapl_instrument(), date(2019, 9, 1))
    assert any("최근 봉" in n for n in provider.drain_notes("AAPL"))


# =============================================================================== fundamentals / resolve


def test_fundamentals_from_info_unit_conversions():
    info = fx_json("yf_info_aapl.json")
    assert info["dividendYield"] == 0.32 and info["dividendRate"] == 1.08  # Yahoo: percent vs $/share
    f = us.fundamentals_from_info(info, as_of=date(2026, 9, 24))
    assert f.dividend_yield == pytest.approx(1.08 / 337.02)  # 0.32% as a decimal, not 0.32
    assert f.per == pytest.approx(info["trailingPE"]) and f.eps == info["trailingEps"]
    assert f.forward_per == pytest.approx(info["forwardPE"]) and f.forward_eps == info["forwardEps"]
    assert f.roe == pytest.approx(1.4875101)  # already a decimal (can exceed 1)
    assert f.operating_margin == pytest.approx(info["operatingMargins"])
    assert f.revenue_growth_yoy == pytest.approx(0.164) and f.earnings_growth_yoy == pytest.approx(0.287)
    assert f.analyst_target_price == pytest.approx(328.22205)
    assert f.analyst_target_high == 405.0 and f.analyst_target_low == 215.0 and f.analyst_count == 39
    assert f.analyst_rating == "2.20 매수 (1=강력매수)"
    assert f.market_cap and f.pbr and f.bps and f.ev_ebitda


@pytest.mark.parametrize(
    "info,expected",
    [
        ({"dividendYield": 0.44, "currentPrice": 225.51}, 0.0044),  # percent -> decimal
        (
            {"dividendRate": 1.0, "dividendYield": 0.44, "currentPrice": 225.51},
            1.0 / 225.51,
        ),
        ({"trailingAnnualDividendYield": 0.0012, "currentPrice": 10.0}, 0.0012),
        ({"trailingAnnualDividendRate": 0.0, "currentPrice": 10.0}, 0.0),
        ({"currentPrice": 10.0}, None),
    ],
)
def test_dividend_yield_from_info(info, expected):
    got = us.dividend_yield_from_info(info)
    assert got == (pytest.approx(expected) if expected is not None else None)


@pytest.mark.parametrize(
    "mean,key,expected",
    [
        (1.295, "strong_buy", "1.29 강력매수 (1=강력매수)"),
        (2.2, "buy", "2.20 매수 (1=강력매수)"),
        (3.0, None, "3.00 중립 (1=강력매수)"),
        (4.0, None, "4.00 비중축소 (1=강력매수)"),
        (4.8, None, "4.80 매도 (1=강력매수)"),
        (None, "hold", "중립"),
        ("Infinity", None, None),
        (None, None, None),
    ],
)
def test_recommendation_label(mean, key, expected):
    assert us.recommendation_label(mean, key) == expected


def test_loss_maker_gets_negative_per():
    f = us.fundamentals_from_info({"trailingEps": -2.0, "currentPrice": 10.0, "quoteType": "EQUITY"})
    assert f.per == pytest.approx(-5.0)


def test_instrument_from_info_and_quote_type_guard():
    inst = us.instrument_from_info("AAPL", fx_json("yf_info_aapl.json"))
    assert (inst.ticker, inst.name, inst.exchange, inst.currency, inst.sector) == (
        "AAPL",
        "Apple Inc.",
        "NASDAQ",
        "USD",
        "Technology",
    )
    assert (inst.benchmark, inst.benchmark_name, inst.yf_symbol, inst.market) == (
        "SPY",
        "S&P 500 (SPY)",
        "AAPL",
        Market.US,
    )
    with pytest.raises(us.UnsupportedInstrumentError):
        us.instrument_from_info("0P0001", {"quoteType": "MUTUALFUND", "longName": "Some Fund"})


def test_resolve_falls_back_to_chart_meta(provider, monkeypatch):
    def fail(sym):
        raise RuntimeError("HTTP 429")

    monkeypatch.setattr(provider, "_info", fail)
    monkeypatch.setattr(provider, "_chart_meta", lambda sym: fx_json("yahoo_chart_aapl.json")["meta"])
    inst = provider.resolve("aapl")
    assert inst.name == "Apple Inc." and inst.exchange == "NASDAQ" and inst.sector == "Technology"
    assert any("차트 메타데이터" in n for n in provider.drain_notes("AAPL"))


def test_resolve_share_class_uses_info(provider, monkeypatch):
    seen = []
    monkeypatch.setattr(
        provider,
        "_info",
        lambda sym: (
            seen.append(sym)
            or {
                "as_of": "2026-09-24",
                "info": {
                    "quoteType": "EQUITY",
                    "longName": "Berkshire Hathaway Inc.",
                    "exchange": "NYQ",
                    "currency": "USD",
                    "sector": "Financial Services",
                },
            }
        ),
    )
    inst = provider.resolve("BRK.B")
    assert seen == ["BRK-B"] and inst.ticker == "BRK-B" and inst.yf_symbol == "BRK-B" and inst.exchange == "NYSE"


def test_provider_fundamentals_fills_from_xbrl_and_sector_per(provider, monkeypatch):
    base = fx_json("yf_info_aapl.json")
    base.pop("revenueGrowth")
    peers = {
        "MSFT": 35.0,
        "NVDA": 30.0,
        "AVGO": 60.0,
        "ORCL": -5.0,
        "CSCO": 25.0,
        "AMD": 400.0,
        "CRM": 40.0,
    }

    def fake_info(sym):
        if sym == "AAPL":
            return {"as_of": "2026-09-24", "info": base}
        if sym in peers:
            return {"as_of": "2026-09-24", "info": {"trailingPE": peers[sym]}}
        raise RuntimeError("peer unavailable")

    monkeypatch.setattr(provider, "_info", fake_info)
    monkeypatch.setattr(provider, "_companyfacts", lambda t: fx_json("sec_companyfacts_nvda.json"))
    monkeypatch.setattr(us, "_now_utc", lambda: NOW)
    f = provider.fundamentals(aapl_instrument())
    assert f.revenue_growth_yoy == pytest.approx(1.0585, abs=1e-3)  # filled from XBRL (fixture = NVDA)
    assert f.earnings_growth_yoy == pytest.approx(0.287)  # Yahoo kept, XBRL only cross-checks
    assert "SEC XBRL 보완(revenue_growth_yoy)" in f.source
    assert f.sector_per == pytest.approx(float(np.median([35.0, 30.0, 60.0, 25.0, 40.0])))  # <=0 / >=300 dropped
    assert f.as_of == date(2026, 9, 24)


def test_provider_fundamentals_without_sec_ua_skips_xbrl(provider, monkeypatch):
    provider.settings.data.sec_user_agent = ""
    monkeypatch.setattr(
        provider,
        "_info",
        lambda sym: {"as_of": "2026-09-24", "info": fx_json("yf_info_aapl.json")},
    )
    monkeypatch.setattr(
        provider,
        "_companyfacts",
        lambda t: pytest.fail("XBRL must not be fetched without UA"),
    )
    monkeypatch.setattr(provider, "_sector_per", lambda t, s: None)
    f = provider.fundamentals(aapl_instrument())
    assert f.per and f.source == "yfinance"


def test_provider_fundamentals_info_failure_uses_xbrl_only(provider, monkeypatch):
    monkeypatch.setattr(provider, "_info", lambda sym: (_ for _ in ()).throw(RuntimeError("HTTP 404")))
    monkeypatch.setattr(provider, "_companyfacts", lambda t: fx_json("sec_companyfacts_nvda.json"))
    monkeypatch.setattr(provider, "_cached_prices", lambda s, st, c: _frame(close=200.0))
    monkeypatch.setattr(us, "_now_utc", lambda: NOW)
    f = provider.fundamentals(nvda_instrument())
    assert f.eps == pytest.approx(7.91) and f.per == pytest.approx(200.0 / 7.91)
    assert f.analyst_target_price is None and "XBRL" in f.source
    assert any("SEC XBRL 대체" in n for n in provider.drain_notes("NVDA"))


# =============================================================================== EDGAR text


def test_edgar_html_to_text_cleaning():
    t = us.edgar_html_to_text(fx_bytes("edgar_10q.htm"))
    assert "HIDDENFACT" not in t  # <ix:header> hidden facts dropped
    assert "Net revenue | $12,345 | $10,001" in t  # split '$' cells merged
    assert "Net income (loss) | (123) | $456" in t  # '(' 123 ')' merged
    assert "Gross margin | 61.2%" in t
    assert "Form 10-Q | 5" not in t and "\n7\n" not in t  # footer + page number dropped
    assert "\nItem 2\n" not in t  # repeated running header dropped
    assert "Management's Discussion" in t  # curly apostrophe normalised
    assert "<?xml" not in t and "p {margin" not in t


def test_extract_sections_10q_skips_toc():
    t = us.edgar_html_to_text(fx_bytes("edgar_10q.htm"))
    rf = us.extract_section(t, "10-Q", "risk_factors", None)
    assert rf.startswith("Item 1A. Risk Factors") and "| 22" not in rf
    assert "Unregistered Sales" not in rf and rf.rstrip().endswith("adversely affect our results.")
    assert len(rf) > 2000
    md = us.extract_section(t, "10-Q", "mdna", None)
    assert md.startswith("Item 2. Management's Discussion") and "Quantitative and Qualitative" not in md
    assert "Outlook: revenue for the third quarter" in md
    heads = us.risk_factor_headlines(rf)
    assert "Adverse economic conditions may harm our business." in heads
    short = us.extract_section(
        us.edgar_html_to_text(fx_bytes("edgar_10q_rf_nochange.htm")),
        "10-Q",
        "risk_factors",
        None,
    )
    assert 0 < len(short) < 2000 and "no material changes" in short


def test_extract_sections_10k_and_missing():
    t = us.edgar_html_to_text(fx_bytes("edgar_10k.htm"))
    rf = us.extract_section(t, "10-K", "risk_factors", None)
    assert rf.startswith("ITEM 1A. RISK FACTORS") and "Unresolved Staff Comments" not in rf
    md = us.extract_section(t, "10-K/A", "mdna", 800)
    assert md.startswith("Item 7. Management's Discussion") and len(md) <= 800 and "chars truncated" in md
    assert us.extract_section("no headings here", "10-Q", "mdna") == ""


def test_real_edgar_exhibit_and_inline_8k():
    ex = us.edgar_html_to_text(fx_bytes("sec_ex991_aapl.htm"))  # SGML <DOCUMENT> wrapper + Workiva html
    assert ex.startswith("Exhibit 99.1") and "<TYPE>" not in ex and len(ex) == 10_877
    assert "quarterly revenue of $109.4 billion, up 16 percent" in ex
    k8 = us.edgar_html_to_text(fx_bytes("sec_8k_nvda_801.htm"))  # <?xml encoding='ASCII'?> inline XBRL
    body = us.body_after_items(k8, 6000)
    assert body.startswith("Item 8.01") and "Hugging Face" in body
    assert not re.search(r"^SIGNATURES?$", body, re.M)


def test_truncate_head_tail_and_compose():
    text = "\n".join(f"line {i:04d} " + "x" * 50 for i in range(2000))
    out = us.truncate_head_tail(text, 5000)
    assert len(out) <= 5000 and out.startswith("line 0000") and out.rstrip().endswith("x" * 50)
    assert "line 1999" in out and "chars truncated" in out
    assert us.truncate_head_tail("short", 100) == "short"
    assert len(us.truncate_head_tail(text, 20)) <= 20
    composed = us.compose_sections(
        [("EX-99.1", text), ("EX-99.2", text[:3000]), ("empty", "")],
        8000,
        [0.62, 0.38, 0.0],
    )
    assert len(composed) <= 8000 and "[EX-99.1]" in composed and "[EX-99.2]" in composed and "[empty]" not in composed
    small = us.compose_sections([("a", "tiny"), ("b", text)], 1000)
    assert "[a]\ntiny" in small and len(small) <= 1000  # a short part keeps its full text


# =============================================================================== SEC parsing


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_parse_submissions_real_fixture():
    meta, df = us.parse_submissions(fx_json("sec_submissions_nvda.json"))
    assert meta["name"] == "NVIDIA CORP" and meta["fiscalYearEnd"]
    assert str(df["filed_utc"].dt.tz) == "UTC" and df["filed_utc"].is_monotonic_decreasing
    er = df[df["items"].str.contains(r"\b2\.02\b")].iloc[0]
    assert er["filed_utc"] == pd.Timestamp("2026-08-26T20:21:19Z")  # acceptanceDateTime is true UTC
    assert er["accepted_et"].hour == 16  # 16:21 ET -> after the close
    assert er["url"] == "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000073/nvda-20260826.htm"
    assert er["index_url"].endswith("/000104581026000073/0001045810-26-000073-index.htm")
    tenq = df[df["form"] == "10-Q"].iloc[0]
    assert tenq["filed_utc"] > er["filed_utc"]  # same day, accepted later
    assert df["items"].map(lambda v: isinstance(v, str)).all()


def test_parse_filing_index_and_exhibits():
    idx = us.parse_filing_index(fx_bytes("sec_index_nvda_202.htm"))
    main = idx[idx["type"] == "8-K"].iloc[0]
    assert "/ix?doc=" not in main["url"] and main["url"].endswith("/nvda-20260826.htm")
    ex = us.exhibits_99(idx)
    assert [x["type"] for x in ex] == ["EX-99.1", "EX-99.2"]
    assert ex[0]["url"].endswith("q2fy27pr.htm") and ex[1]["url"].endswith("q2fy27cfocommentary.htm")
    assert us.exhibits_99(idx.iloc[0:0]) == []


def test_parse_company_tickers_and_8k_items():
    m = us.parse_company_tickers(fx_json("sec_company_tickers.json"))
    assert m["NVDA"] == ("0001045810", "NVIDIA CORP") and m["BRK-B"][0] == "0001067983"
    assert us.parse_8k_items("2.02,9.01") == ["2.02", "9.01"] and us.parse_8k_items(None) == []
    assert us.item_severity(["4.02", "9.01"]) == -3 and us.item_severity(["8.01"]) == 0


# =============================================================================== XBRL


NVDA_SPLITS = pd.Series({pd.Timestamp("2021-07-20"): 4.0, pd.Timestamp("2024-06-10"): 10.0})


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_xbrl_eps_is_split_adjusted_per_fact():
    facts = fx_json("sec_companyfacts_nvda.json")
    raw = us.quarterly_series(facts, "eps_diluted")
    q4_fy23 = raw[raw["end"] == pd.Timestamp("2023-01-29")].iloc[0]
    assert q4_fy23["derived"] and q4_fy23["value"] < 0  # the bug: post-split FY minus pre-split 9M
    q = us.quarterly_series(facts, "eps_diluted", splits=NVDA_SPLITS)
    by_end = q.set_index("end")["value"]
    assert by_end[pd.Timestamp("2019-01-27")] == pytest.approx(0.92 / 40)  # pre both splits
    assert by_end[pd.Timestamp("2023-04-30")] == pytest.approx(0.82 / 10)  # filed 2024-05-29, pre 10:1
    assert by_end[pd.Timestamp("2023-01-29")] > 0  # derived Q4 now sane
    assert by_end[pd.Timestamp("2026-07-26")] == pytest.approx(2.46)
    ttm = us.ttm_series(q)
    assert ttm["ttm"].iloc[-1] == pytest.approx(1.30 + 1.76 + 2.39 + 2.46)
    assert ttm["available"].iloc[-1] == pd.Timestamp("2026-08-26")
    recent = ttm[ttm["end"] >= pd.Timestamp("2019-06-01")]["ttm"]
    ratio = (recent / recent.shift(1)).dropna()
    assert ratio.between(1 / 3, 3).all()  # no 4x / 10x split jumps


def test_xbrl_snapshot_and_merge():
    snap = us.xbrl_snapshot(fx_json("sec_companyfacts_nvda.json"), date(2026, 9, 24))
    assert snap["revenue_growth_yoy"] == pytest.approx(1.0585, abs=1e-3)  # press release: +106%
    assert snap["eps_ttm"] == pytest.approx(7.91) and 0.5 < snap["operating_margin"] < 0.8
    assert us.xbrl_snapshot(fx_json("sec_companyfacts_nvda.json"), date(2028, 1, 1)) == {}  # stale filer
    f = us.Fundamentals(revenue_growth_yoy=1.06)
    filled = us.merge_xbrl_snapshot(f, snap, price=200.0)
    assert set(filled) == {"earnings_growth_yoy", "operating_margin", "eps", "per"}
    assert f.revenue_growth_yoy == 1.06 and f.per == pytest.approx(200.0 / 7.91)


def _eps_facts(values: list[float], ends: list[str], lag_days: int = 30) -> dict:
    rows = []
    for v, e in zip(values, ends):
        end = pd.Timestamp(e)
        rows.append(
            {
                "start": (end - pd.Timedelta(90, "D")).date().isoformat(),
                "end": e,
                "val": v,
                "accn": f"a-{e}",
                "fy": end.year,
                "fp": "Q",
                "form": "10-Q",
                "filed": (end + pd.Timedelta(lag_days, "D")).date().isoformat(),
            }
        )
    return {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": rows}}}}}


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_trailing_pe_history_is_point_in_time():
    ends = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31"]
    facts = _eps_facts([1.0, 1.0, 1.0, 1.0, 2.0], ends)
    close = pd.Series(100.0, index=pd.bdate_range("2024-12-01", "2026-03-31"))
    vh = us.trailing_pe_history(facts, close, splits=None)
    first_ttm_filed = pd.Timestamp("2025-01-30")  # 4th quarter filed -> usable the NEXT day
    assert vh.index.min() == pd.Timestamp("2025-01-31")
    assert vh.loc["2025-01-31", "per"] == pytest.approx(25.0)
    assert first_ttm_filed not in vh.index  # never on the filing day itself
    assert vh.loc["2025-04-30", "per"] == pytest.approx(25.0)  # Q1'25 filed 04-30 -> not yet usable
    assert vh.loc["2025-05-01", "per"] == pytest.approx(20.0)  # TTM 5.0
    assert vh.index.max() <= pd.Timestamp("2025-05-01") + pd.Timedelta(200, "D")  # stale TTM -> NaN
    loss = us.trailing_pe_history(_eps_facts([-1.0, 0.1, 0.1, 0.1], ends[:4]), close, splits=None)
    assert loss is None  # TTM EPS <= 0 -> no P/E


def test_provider_valuation_history(provider, monkeypatch):
    close = pd.Series(100.0, index=pd.bdate_range("2019-09-01", "2026-09-23"))
    monkeypatch.setattr(provider, "_companyfacts", lambda t: fx_json("sec_companyfacts_nvda.json"))
    monkeypatch.setattr(
        provider,
        "_raw_close_and_splits",
        lambda s, st: {"close": close, "splits": NVDA_SPLITS, "source": "yfinance"},
    )
    vh = provider.valuation_history(nvda_instrument(), date(2019, 9, 1))
    assert list(vh.columns) == ["per"] and vh.index.min() >= pd.Timestamp("2019-09-01")
    jumps = (vh["per"] / vh["per"].shift(1)).dropna()
    assert jumps.between(1 / 3, 3).all()  # no split artefacts
    assert vh["per"].iloc[-1] == pytest.approx(100.0 / 7.91)
    provider.settings.data.sec_user_agent = ""
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        provider.valuation_history(nvda_instrument(), date(2019, 9, 1))


# =============================================================================== Motley Fool


def test_parse_fool_links_sorted_by_fiscal_period():
    links = us.parse_fool_links(fx_bytes("fool_quote_nvda.html").decode())
    assert [(d["fiscal_year"], d["fiscal_quarter"]) for d in links] == [
        (2027, 2),
        (2027, 1),
        (2025, 4),
    ]
    assert links[0]["published"] == "2026-08-31" and links[0]["url"].startswith("https://www.fool.com/earnings/")


def test_parse_fool_transcript_qa_split_skips_opening_operator():
    t = us.parse_fool_transcript(fx_bytes("fool_transcript_nvda.html").decode(), "u")
    assert t["prepared"].startswith("Operator: Good afternoon")  # opener mentions Q&A, stays in prepared
    assert t["qa"].startswith("Operator: We will now begin the question-and-answer session")
    assert "Vivek Arya" in t["qa"] and "Vivek Arya" not in t["prepared"]
    assert "NASDAQ:NVDA" not in t["transcript"] and "Should you invest" not in t["transcript"]
    assert t["takeaways"].startswith("- Revenue -- $96.2 billion")
    d = us.transcript_digest(t, 1500)
    assert len(d) <= 1500 and "[Prepared remarks]" in d and "[Q&A]" in d


# =============================================================================== news


def test_clean_company_name_and_query():
    assert us.clean_company_name("Apple Inc.") == "Apple"
    assert us.clean_company_name("Amazon.com, Inc.") == "Amazon"
    assert us.clean_company_name("JPMorgan Chase & Co.") == "JPMorgan Chase"
    assert us.clean_company_name("The Coca-Cola Company") == "Coca-Cola"
    assert us.clean_company_name("Procter & Gamble Company (The)") == "Procter & Gamble"
    assert us.clean_company_name("Berkshire Hathaway Inc. New") == "Berkshire Hathaway"
    assert us.clean_company_name("AT&T Inc.") == "AT&T"
    assert us.google_news_query("NVDA", "NVIDIA Corporation", 21) == '"NVIDIA" OR "NVDA stock" when:21d'


@pytest.mark.parametrize(
    "ticker,company,text,expected",
    [
        ("NVDA", "NVIDIA Corporation", "Nvidia unveils Rubin", True),
        ("NVDA", "NVIDIA Corporation", "Why Fastly Stock Jumped Today", False),
        ("T", "AT&T Inc.", "AT&T raises its dividend", True),
        ("T", "AT&T Inc.", "Don't miss these T-bills", False),
        ("T", "AT&T Inc.", "Shares of $T rose 2%", True),
        ("T", "AT&T Inc.", "Telecom giant (NYSE: T) beats", True),
        ("CAT", "Caterpillar Inc.", "funny cat videos", False),
        ("CAT", "Caterpillar Inc.", "CAT earnings beat", True),
        ("BRK-B", "Berkshire Hathaway Inc.", "BRK.B hits a record", True),
        ("META", "Meta Platforms, Inc.", "Facebook parent cuts costs", True),
        ("T", "T", "Don't miss these deals", False),  # name fell back to the ticker
        (
            "NVDA",
            "NVDA",
            "nvda options activity",
            False,
        ),  # ticker match stays case-sensitive
        ("NVDA", "NVDA", "NVDA options activity", True),
    ],
)
def test_news_relevance(ticker, company, text, expected):
    assert us.NewsRelevance(ticker, company).is_relevant(text) is expected


@pytest.mark.parametrize(
    "ticker,company,publisher,expected",
    [
        ("NVDA", "NVIDIA Corporation", "nvidia.com", True),
        ("NVDA", "NVIDIA Corporation", "developer.nvidia.com", True),
        ("NVDA", "NVIDIA Corporation", "NVIDIA Blog", True),
        ("NVDA", "NVIDIA Corporation", "Reuters", False),
        ("AAPL", "Apple Inc.", "apple.com", True),
        ("AAPL", "Apple Inc.", "Apple Newsroom", True),
        (
            "AAPL",
            "Apple Inc.",
            "AppleInsider",
            False,
        ),  # independent outlet named after the company
        ("AAPL", "Apple Inc.", "", False),
    ],
)
def test_self_published_matcher(ticker, company, publisher, expected):
    assert us.self_published_matcher(ticker, company)(publisher) is expected


def test_merge_news_drops_company_own_channels():
    row = {
        "title": "Nvidia GTC keynote recap",
        "summary": "",
        "url": "https://blogs.nvidia.com/x",
        "source": "NVIDIA Blog",
        "publisher": "NVIDIA Blog",
        "published_at": utc(2026, 9, 23, 22),
        "origin": "google",
    }
    items = us.merge_news(
        _nvda_rows() + [row],
        utc(2026, 9, 3),
        10,
        "NVDA",
        "NVIDIA Corporation",
        now=utc(2026, 9, 24),
    )
    assert "Nvidia GTC keynote recap" not in [i.title for i in items] and len(items) == 4


def _nvda_rows():
    rows = us.parse_yf_news(fx_json("yf_news_nvda.json"))
    rows += us.parse_rss(fx_bytes("google_news_nvda.xml"), "google")
    return rows


def test_parse_feeds_and_merge_news():
    rows = _nvda_rows()
    g = [r for r in rows if r["origin"] == "google"]
    assert g[0]["title"] == "Nvidia to acquire Hugging Face for $11.9 billion" and g[0]["source"] == "Reuters"
    assert all(r["published_at"].tzinfo is not None and r["published_at"].utcoffset() == timedelta(0) for r in rows)
    items = us.merge_news(
        rows,
        since=utc(2026, 9, 3),
        limit=10,
        ticker="NVDA",
        company="NVIDIA Corporation",
        now=utc(2026, 9, 24),
    )
    titles = [i.title for i in items]
    assert titles == [
        "NVIDIA shares hit record as Blackwell Ultra ramps",
        "Nvidia unveils Rubin Ultra roadmap at GTC Paris",
        "NVDA stock slips as chip export rules tighten",
        "Nvidia to acquire Hugging Face for $11.9 billion",
    ]
    dup = items[2]
    assert dup.summary.startswith("Shares of NVDA fell")  # google duplicate kept, summary lent by Yahoo
    assert all(i.lang == "en" and i.published_at.tzinfo is not None for i in items)
    assert len({i.id for i in items}) == len(items)
    assert [
        i.title for i in us.merge_news(rows, utc(2026, 9, 3), 2, "NVDA", "NVIDIA Corporation", now=utc(2026, 9, 24))
    ] == titles[:2]


def test_provider_news_rss_fallback_when_yfinance_empty(provider, monkeypatch):
    monkeypatch.setattr(provider, "_fetch_yf_news", lambda s: [])  # Yahoo 502 -> yfinance returns []
    monkeypatch.setattr(provider, "_fetch_yahoo_rss", lambda s: fx_bytes("yahoo_rss_nvda.xml"))
    queries = []
    monkeypatch.setattr(
        provider,
        "_fetch_google_news",
        lambda q: queries.append(q) or fx_bytes("google_news_nvda.xml"),
    )
    items = provider.news(nvda_instrument(), utc(2026, 9, 3), 30)
    titles = [i.title for i in items]
    assert "Nvidia stock: what analysts expect after earnings" in titles
    assert "3 dividend stocks to buy now" not in titles
    assert queries and '"NVIDIA" OR "NVDA stock"' in queries[0]
    assert provider.drain_notes("NVDA") == []  # RSS covers the same pool: no note


def test_provider_news_partial_and_total_failure(provider, monkeypatch):
    def fail(*a):
        raise RuntimeError("HTTP 503")

    monkeypatch.setattr(provider, "_fetch_yf_news", lambda s: fx_json("yf_news_nvda.json"))
    monkeypatch.setattr(provider, "_fetch_google_news", fail)
    items = provider.news(nvda_instrument(), utc(2026, 9, 3), 30)
    assert items and any("Google News" in n for n in provider.drain_notes("NVDA"))

    p2 = us.USProvider(provider.settings, DiskCache(provider.cache.root / "other"))
    for name in ("_fetch_yf_news", "_fetch_yahoo_rss", "_fetch_google_news"):
        monkeypatch.setattr(p2, name, fail)
    with pytest.raises(RuntimeError, match="모든 뉴스 소스 실패"):
        p2.news(nvda_instrument(), utc(2026, 9, 3), 30)


# =============================================================================== filings (offline end-to-end)


class FakeSec:
    """URL -> fixture bytes for the NVDA filing set; records every SEC request."""

    def __init__(self, tenq: str = "edgar_10q.htm", fail: tuple[str, ...] = ()):
        self.tenq, self.fail, self.calls = tenq, fail, []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if any(f in url for f in self.fail):
            raise requests.HTTPError("503 Server Error")
        routes = [
            ("company_tickers.json", "sec_company_tickers.json"),
            ("/submissions/CIK", "sec_submissions_nvda.json"),
            ("-index.htm", "sec_index_nvda_202.htm"),
            ("q2fy27pr.htm", "sec_ex991_aapl.htm"),
            ("q2fy27cfocommentary.htm", "sec_ex991_aapl.htm"),
            ("nvda-20260726.htm", self.tenq),
            ("nvda-20260125.htm", "edgar_10k.htm"),
        ]
        for pat, name in routes:
            if pat in url:
                return fx_bytes(name)
        if re.search(r"/(nvda-2026\d{4}|d48176d8k)\.htm$", url):
            return fx_bytes("sec_8k_nvda_801.htm")
        raise AssertionError(f"unexpected SEC url {url}")


def fake_fool(quote: str = "fool_quote_nvda.html"):
    calls = []

    def page(url: str):
        calls.append(url)
        if url == "https://www.fool.com/quote/nasdaq/nvda/":
            return fx_bytes(quote).decode()
        if "/2026/08/31/nvidia-nvda-q2-2027-earnings-call-transcript/" in url:
            return fx_bytes("fool_transcript_nvda.html").decode()
        return None

    page.calls = calls
    return page


SINCE = utc(2026, 5, 27)  # 120 days before 2026-09-24


def test_filings_require_sec_user_agent(provider):
    provider.settings.data.sec_user_agent = "  "
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        provider.filings(nvda_instrument(), SINCE, 6)


def test_filings_end_to_end_offline(provider, monkeypatch):
    sec, fool = FakeSec(), fake_fool()
    monkeypatch.setattr(provider, "_sec_get", sec)
    monkeypatch.setattr(provider, "_fool_page", fool)
    items = provider.filings(nvda_instrument(), SINCE, 6)
    assert len(items) == 6 and all(isinstance(i, FilingItem) for i in items)
    assert [i.filed_at for i in items] == sorted((i.filed_at for i in items), reverse=True)
    assert all(i.filed_at.utcoffset() == timedelta(0) and i.filed_at >= SINCE for i in items)
    max_chars = provider.settings.data.max_filing_chars
    assert all(0 < len(i.text) <= max_chars for i in items)

    er = [i for i in items if "Item 2.02" in i.title]
    assert len(er) == 1 and er[0].is_earnings and er[0].form_type == "8-K" and er[0].source == "SEC EDGAR"
    assert "[EX-99.1]" in er[0].text and "[EX-99.2]" in er[0].text and er[0].url.endswith("q2fy27pr.htm")
    assert er[0].filed_at == utc(2026, 8, 26, 20, 21, 19)
    tr = [i for i in items if i.form_type == "Earnings Call Transcript"]
    assert len(tr) == 1 and tr[0].is_earnings and tr[0].source == "Motley Fool" and "[Q&A]" in tr[0].text
    assert "FY2027 Q2" in tr[0].title
    rf = [i for i in items if "위험요인" in i.title]
    assert len(rf) == 1 and rf[0].form_type == "10-Q" and "Adverse economic conditions" in rf[0].text
    md = [i for i in items if "MD&A" in i.title]
    assert len(md) == 1 and "Outlook" in md[0].text and not md[0].is_earnings
    other = [i for i in items if i.title.startswith("8-K (")]
    assert [i.filed_at.date() for i in other] == [date(2026, 9, 3), date(2026, 8, 17)]
    assert "8.01 기타 주요 사항" in other[0].title and "1.01 중요 계약 체결" in other[1].title
    assert not any("Form4" in u or "xslF345" in u for u in sec.calls)  # insider forms never fetched
    assert provider.drain_notes("NVDA") == []

    # cached: no new requests, same result
    n_calls = len(sec.calls)
    again = provider.filings(nvda_instrument(), SINCE, 6)
    assert [i.id for i in again] == [i.id for i in items] and len(sec.calls) == n_calls


def test_filings_10q_without_material_changes_uses_10k(provider, monkeypatch):
    monkeypatch.setattr(provider, "_sec_get", FakeSec(tenq="edgar_10q_rf_nochange.htm"))
    monkeypatch.setattr(provider, "_fool_page", fake_fool())
    items = provider.filings(nvda_instrument(), SINCE, 6)
    rf = [i for i in items if "위험요인" in i.title][0]
    assert rf.form_type == "10-K" and "중요 변경 없음" in rf.title
    assert "[10-Q 위험요인 업데이트]" in rf.text and "no material changes" in rf.text
    assert rf.filed_at < SINCE  # the anchor 10-Q is in the window, the 10-K it points to is not
    assert rf.filed_at == utc(2026, 2, 25, 21, 42, 19)


def test_filings_stale_transcript_and_failed_document_are_reported(provider, tmp_path, monkeypatch):
    stale_quote = tmp_path / "quote.html"
    stale_quote.write_text('"/earnings/call-transcripts/2026/05/28/nvidia-nvda-q1-2027-earnings-call-transcript/"')
    monkeypatch.setattr(provider, "_sec_get", FakeSec(fail=("q2fy27pr.htm",)))

    def page(url):
        return stale_quote.read_text() if "/quote/nasdaq/nvda/" in url else None

    monkeypatch.setattr(provider, "_fool_page", page)
    items = provider.filings(nvda_instrument(), SINCE, 6)
    assert not any(i.is_earnings for i in items)
    assert len(items) == 6  # other candidates fill the slots
    notes = provider.drain_notes("NVDA")
    assert any("실적 보도자료(8-K 2.02) 수집 실패" in n for n in notes)
    assert any("녹취록 미게시" in n and "2026-05-28" in n for n in notes)


def test_filings_limit_prioritises_earnings(provider, monkeypatch):
    monkeypatch.setattr(provider, "_sec_get", FakeSec())
    monkeypatch.setattr(provider, "_fool_page", fake_fool())
    items = provider.filings(nvda_instrument(), SINCE, 2)
    assert len(items) == 2 and all(i.is_earnings for i in items)


# =============================================================================== FX / universe / plumbing


def test_pick_completed_fx_skips_live_bar():
    s = pd.Series(
        [1384.86, 1373.68, 1350.36, 1370.20],
        index=pd.DatetimeIndex(["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24"]),
    )
    assert us.pick_completed_fx(s, "Europe/London", date(2026, 9, 23), now=NOW) == (
        1350.36,
        date(2026, 9, 23),
    )
    assert us.pick_completed_fx(s, "Europe/London", date(2026, 9, 22), now=NOW) == (
        1373.68,
        date(2026, 9, 22),
    )
    assert us.pick_completed_fx(s, "Europe/London", date(2026, 9, 30), now="2026-09-26T12:00:00Z")[0] == 1370.20
    with pytest.raises(ValueError):
        us.pick_completed_fx(s * 100, "Europe/London", date(2026, 9, 23), now=NOW)


def test_fx_rate_yahoo_then_naver(provider, monkeypatch):
    monkeypatch.setattr(us, "_now_utc", lambda: NOW)
    s = pd.Series(
        [1373.68, 1350.36, 1370.20],
        index=pd.DatetimeIndex(["2026-09-22", "2026-09-23", "2026-09-24"]),
    )
    monkeypatch.setattr(provider, "_fetch_krw_series_yf", lambda: (s, "Europe/London"))
    assert provider.fx_rate("USD", "KRW") == pytest.approx(1350.36)
    assert provider.fx_rate("KRW", "USD") == pytest.approx(1 / 1350.36)
    assert provider.fx_rate("usd", "USD") == 1.0
    assert provider.drain_notes("AAPL") == []
    with pytest.raises(NotImplementedError):
        provider.fx_rate("USD", "JPY")

    p2 = us.USProvider(provider.settings, DiskCache(provider.cache.root / "fx2"))
    monkeypatch.setattr(
        p2,
        "_fetch_krw_series_yf",
        lambda: (_ for _ in ()).throw(RuntimeError("HTTP 429")),
    )
    monkeypatch.setattr(
        p2,
        "_fetch_usdkrw_naver",
        lambda: us.parse_naver_fx(fx_json("naver_fx_usdkrw.json")),
    )
    assert p2.fx_rate("USD", "KRW") == pytest.approx(1370.50)
    notes = p2.drain_notes("AAPL")
    assert any("네이버" in n for n in notes) and p2.drain_notes("NVDA") == notes  # provider-wide note


class _FakeKrwTicker:
    def __init__(self, hist: pd.DataFrame, calls: list, fail: bool):
        self._hist, self._calls, self._fail = hist, calls, fail

    def history(self, **kw) -> pd.DataFrame:
        self._calls.append(kw)
        if self._fail:
            raise RuntimeError("HTTP 429 Too Many Requests")
        return self._hist.copy()


class _FakeYFModule:
    """The slice of yfinance both providers use for KRW=X (US: module ``yf``; KR: ``kr._yf()``)."""

    def __init__(self, hist: pd.DataFrame, fail: bool = False):
        self.hist, self.fail, self.calls = hist, fail, []

    def Ticker(self, symbol: str) -> _FakeKrwTicker:  # noqa: N802 - mimics yfinance
        assert symbol == "KRW=X"
        return _FakeKrwTicker(self.hist, self.calls, self.fail)


def _krw_history() -> pd.DataFrame:
    """KRW=X as yfinance returns it: London-midnight bars; the 09-24 bar is live at 12:00Z."""
    idx = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="Europe/London") for d in ("2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24")]
    )
    return pd.DataFrame({"Close": [1384.86, 1373.68, 1350.36, 1399.99]}, index=idx)


def _both_providers(settings, tmp_path, monkeypatch, fake, shared_cache: bool):
    from quantcio.data import kr

    monkeypatch.setattr(us, "_now_utc", lambda: NOW)
    monkeypatch.setattr(us, "yf", fake)
    monkeypatch.setattr(us, "FX_YF_TRIES", 1)
    monkeypatch.setattr(kr, "_yf", lambda: fake)
    monkeypatch.setattr(kr, "_polite_wait", lambda url: None)
    us_cache = DiskCache(tmp_path / ("fx-shared" if shared_cache else "fx-us"))
    kr_cache = us_cache if shared_cache else DiskCache(tmp_path / "fx-kr")
    usp = us.USProvider(settings, us_cache)
    krp = kr.KRProvider(settings, kr_cache, now_fn=lambda: NOW.to_pydatetime())
    return usp, krp


def test_fx_one_definition_across_providers_exact_reciprocals(settings, tmp_path, monkeypatch):
    """KR and US must convert with the SAME USD/KRW: the last completed KRW=X bar aligned to the last
    completed US session (never the live bar), and KRW->USD must be exactly 1/(USD->KRW)."""
    fake = _FakeYFModule(_krw_history())
    usp, krp = _both_providers(settings, tmp_path, monkeypatch, fake, shared_cache=False)
    usd_krw = usp.fx_rate("USD", "KRW")
    assert usd_krw == 1350.36  # 09-23 (completed, = last completed US session), not the live 1399.99
    assert krp.fx_rate("USD", "KRW") == usd_krw  # identical, fetched independently (separate caches)
    krw_usd = krp.fx_rate("KRW", "USD")
    assert krw_usd == usp.fx_rate("KRW", "USD") == 1.0 / usd_krw  # exact reciprocal, both providers
    assert krw_usd * usd_krw == pytest.approx(1.0, rel=1e-15)
    assert usp.fx_rate("USD", "USD") == krp.fx_rate("KRW", "KRW") == 1.0
    for p in (usp, krp):  # one exception type (NotImplementedError AND ValueError) for both
        with pytest.raises(NotImplementedError):
            p.fx_rate("USD", "JPY")
        with pytest.raises(ValueError):
            p.fx_rate("JPY", "KRW")
    assert usp.data_warnings("AAPL") == [] and krp.data_warnings("005930") == []

    # one cache entry: a KR and a US ticker converted in one run read the same payload, fetched once
    fake2 = _FakeYFModule(_krw_history())
    usp2, krp2 = _both_providers(settings, tmp_path, monkeypatch, fake2, shared_cache=True)
    assert usp2.fx_rate("USD", "KRW") == krp2.fx_rate("USD", "KRW") == usd_krw
    assert krp2.fx_rate("KRW", "USD") == 1.0 / usp2.fx_rate("USD", "KRW")
    assert len(fake2.calls) == 1


def test_fx_naver_fallback_identical_and_flagged_in_both_providers(settings, tmp_path, monkeypatch):
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return fx_json("naver_fx_usdkrw.json")

    urls: list[str] = []

    def fake_request(self, method, url, *a, **k):
        assert "api.stock.naver.com/marketindex/exchange/FX_USDKRW/prices" in url, url
        urls.append(url)
        return _Resp()

    monkeypatch.setattr(requests.sessions.Session, "request", fake_request)
    fake = _FakeYFModule(_krw_history(), fail=True)
    usp, krp = _both_providers(settings, tmp_path, monkeypatch, fake, shared_cache=False)
    rate = usp.fx_rate("USD", "KRW")
    assert rate == krp.fx_rate("USD", "KRW") == 1370.50  # Hana Bank 09-24 (<= today KST)
    assert krp.fx_rate("KRW", "USD") == 1.0 / rate
    assert len(urls) == 2  # each provider went through the same fallback endpoint
    us_notes = [n for n in usp.data_warnings("AAPL") if "환율" in n]
    kr_notes = krp.data_warnings("__fx__")
    assert us_notes == kr_notes and len(kr_notes) == 1
    assert "네이버 고시환율" in kr_notes[0] and "1,370.50" in kr_notes[0]
    assert kr_notes[0] in krp.data_warnings("005930")  # provider-wide note reaches the KR ticker


def test_universe_static_sector_diverse():
    names = us.interleaved_universe()
    assert len(names) >= 80 and len(set(names)) == len(names)
    assert set(us.US_UNIVERSE) == set(us.GICS_TO_YAHOO) and len(us.US_UNIVERSE) == 11
    assert all(len(v) >= 8 for v in us.US_UNIVERSE.values())
    etfs = {
        "SPY",
        "QQQ",
        "IVV",
        "VOO",
        "VTI",
        "DIA",
        "IWM",
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "GLD",
        "TLT",
    }
    assert not etfs & set(names)
    sector_of = {t: s for s, ts in us.US_UNIVERSE.items() for t in ts}
    assert len({sector_of[t] for t in names[:11]}) == 11  # interleaved by sector


def test_provider_universe(provider):
    first = provider.universe(60)
    assert len(first) == 60 and first == provider.universe(60) and first[:3] == ["AAPL", "JPM", "LLY"]
    assert provider.universe(500) == us.interleaved_universe() and provider.universe(0) == []


def test_throttle_enforces_rate():
    t = us._Throttle(1 / 8)
    t0 = time.monotonic()
    for _ in range(9):
        t.wait()
    assert time.monotonic() - t0 >= 0.95  # 8 gaps x 125 ms
    assert us._SEC_THROTTLE.min_interval >= 1 / 10  # SEC fair access: <= 10 req/s


def test_sec_session_uses_global_throttle_and_ua(monkeypatch):
    waits, sent = [], []

    class Resp:
        status_code = 200

    monkeypatch.setattr(us, "_SEC_THROTTLE", type("T", (), {"wait": lambda self: waits.append(1)})())
    monkeypatch.setattr(
        requests.sessions.Session,
        "request",
        lambda self, method, url, *a, **k: sent.append(dict(self.headers)) or Resp(),
    )
    s = us.SecSession("QuantCIO Offline Test offline@example.com")
    s.get("https://data.sec.gov/submissions/CIK0001045810.json")
    s.get("https://www.sec.gov/files/company_tickers.json")
    assert len(waits) == 2 and sent[0]["User-Agent"] == "QuantCIO Offline Test offline@example.com"


def test_cached_dedupes_concurrent_fetches_and_survives_write_errors(provider, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    calls = []

    def slow_fetch():
        calls.append(1)
        time.sleep(0.2)
        return {"v": 1}

    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(lambda _: provider._cached("t", "k", 1.0, slow_fetch), range(8)))
    assert len(calls) == 1 and all(o == {"v": 1} for o in out)

    def broken_set(*a, **k):
        raise OSError("tmp file raced")

    monkeypatch.setattr(provider.cache, "set", broken_set)
    assert provider._cached("t", "k2", 1.0, lambda: [1, 2]) == [1, 2]


def test_google_query_share_class():
    assert (
        us.google_news_query("BRK-B", "Berkshire Hathaway Inc.", 21) == '"Berkshire Hathaway" OR "BRK.B stock" when:21d'
    )


def test_source_hygiene():
    src = Path(us.__file__).read_text()
    assert "Timedelta(days" not in src  # numpy 2.5: generic-unit deprecation
    assert not re.search(r"^\s*print\(", src, re.M)
    assert not re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", src)  # no hard-coded contact e-mail
    assert "research@example.com" not in src


# =============================================================================== review regressions


def test_parse_fool_links_new_slug_format_and_ticker_filter():
    # Live quote pages (2026-09): new calls use '...-earnings-transcript', and related-article links to
    # OTHER companies' transcripts sit on the same page (XOM's page carried an NRP transcript).
    page = (
        '"/earnings/call-transcripts/2026/08/07/microsoft-msft-q4-2026-earnings-call-transcript/" '
        '"/earnings/call-transcripts/2026/08/28/nvidia-nvda-q2-2027-earnings-call-transcript/" '
        '"/earnings/call-transcripts/2026/04/29/microsoft-msft-q3-2026-earnings-transcript/" '
        '"/earnings/call-transcripts/2025/11/04/natural-resource-partners-nrp-q3-2025-earnings-call-transcript/"'
    )
    links = us.parse_fool_links(page, "MSFT")
    assert [(d["fiscal_year"], d["fiscal_quarter"]) for d in links] == [
        (2026, 4),
        (2026, 3),
    ]
    assert links[1]["url"].endswith("/microsoft-msft-q3-2026-earnings-transcript/")
    assert all("nvidia" not in d["url"] and "natural-resource" not in d["url"] for d in links)
    assert len(us.parse_fool_links(page)) == 4  # unfiltered form still available
    brk = '"/earnings/call-transcripts/2026/05/04/berkshire-hathaway-brk-b-q1-2026-earnings-call-transcript/"'
    assert len(us.parse_fool_links(brk, "BRK.B")) == 1 and us.parse_fool_links(brk, "BRK-B")
    att = (
        '"/earnings/call-transcripts/2026/07/23/att-t-q2-2026-earnings-call-transcript/" '
        '"/earnings/call-transcripts/2026/07/24/at-home-tt-q2-2026-earnings-call-transcript/"'
    )
    assert [d["url"].split("/")[-2] for d in us.parse_fool_links(att, "T")] == [
        "att-t-q2-2026-earnings-call-transcript"
    ]


def test_filings_never_serve_another_companys_transcript(provider, monkeypatch):
    # NVDA's quote page also links a peer's call with a NEWER fiscal label (FY2028): it must be ignored.
    quote = (
        fx_bytes("fool_quote_nvda.html").decode() + '<a href="/earnings/call-transcripts/2026/09/01/'
        'advanced-micro-devices-amd-q1-2028-earnings-call-transcript/">x</a>'
    )
    fool = fake_fool()
    monkeypatch.setattr(provider, "_sec_get", FakeSec())
    monkeypatch.setattr(
        provider,
        "_fool_page",
        lambda url: quote if url.endswith("/quote/nasdaq/nvda/") else fool(url),
    )
    items = provider.filings(nvda_instrument(), SINCE, 6)
    tr = [i for i in items if i.form_type == "Earnings Call Transcript"]
    assert len(tr) == 1 and "nvidia-nvda-q2-2027" in tr[0].url and "FY2027 Q2" in tr[0].title
    assert not any("advanced-micro-devices" in u for u in fool.calls)


def test_transcript_published_before_late_8k_is_not_stale(provider, monkeypatch):
    # Item 2.02 may be furnished days after the call: a transcript dated 2 days BEFORE the 8-K
    # (2026-08-26) belongs to the same quarter and must be served, not reported as missing.
    early = "/earnings/call-transcripts/2026/08/24/nvidia-nvda-q2-2027-earnings-transcript/"

    def page(url):
        if url.endswith("/quote/nasdaq/nvda/"):
            return f'"{early}" "/earnings/call-transcripts/2026/05/28/nvidia-nvda-q1-2027-earnings-call-transcript/"'
        return fx_bytes("fool_transcript_nvda.html").decode() if early in url else None

    monkeypatch.setattr(provider, "_sec_get", FakeSec())
    monkeypatch.setattr(provider, "_fool_page", page)
    items = provider.filings(nvda_instrument(), SINCE, 6)
    tr = [i for i in items if i.form_type == "Earnings Call Transcript"]
    assert len(tr) == 1 and tr[0].url.endswith(early) and tr[0].filed_at == utc(2026, 8, 24, 4)
    assert not any("녹취록" in n for n in provider.drain_notes("NVDA"))


def test_10q_without_item_1a_is_not_reported_as_no_material_change(provider, monkeypatch):
    # XOM's 10-Q (2026-08-03) has no Item 1A at all. Falling back to the 10-K is fine, but the title
    # must not claim the 10-Q reported "no material changes", and the substitution must be noted.
    monkeypatch.setattr(provider, "_sec_get", FakeSec(tenq="edgar_10q_no_item1a.htm"))
    monkeypatch.setattr(provider, "_fool_page", fake_fool())
    items = provider.filings(nvda_instrument(), SINCE, 6)
    rf = [i for i in items if "위험요인" in i.title][0]
    assert rf.form_type == "10-K" and "중요 변경 없음" not in rf.title and "위험요인 항목 없음" in rf.title
    assert "[10-Q 위험요인 업데이트]" not in rf.text
    assert any("위험요인(Item 1A) 미추출" in n for n in provider.drain_notes("NVDA"))


def test_10q_short_update_that_is_not_no_change_keeps_neutral_title(provider, monkeypatch, tmp_path):
    # JPM style: the 10-Q Item 1A is a short cross-reference, not a "no material changes" statement.
    src = (
        fx_bytes("edgar_10q_rf_nochange.htm")
        .decode()
        .replace(
            "There have been no material changes to the risk factors disclosed in",
            "Refer to the discussion of risk factors in",
        )
    )
    assert "no material changes to the risk" not in src
    tenq = tmp_path / "edgar_10q_refer.htm"
    tenq.write_text(src)
    sec = FakeSec()
    monkeypatch.setattr(
        provider,
        "_sec_get",
        lambda url: tenq.read_bytes() if "nvda-20260726.htm" in url else sec(url),
    )
    monkeypatch.setattr(provider, "_fool_page", fake_fool())
    rf = [i for i in provider.filings(nvda_instrument(), SINCE, 6) if "위험요인" in i.title][0]
    assert rf.form_type == "10-K" and "중요 변경 없음" not in rf.title and "위험요인 항목" in rf.title
    assert "[10-Q 위험요인 업데이트]" in rf.text and "Refer to the discussion" in rf.text


@pytest.mark.parametrize(
    "ticker,company,publisher,expected",
    [
        (
            "NVDA",
            "NVIDIA Corporation",
            "NVIDIA",
            True,
        ),  # Google News label for nvidia.com (seen live)
        ("AAPL", "Apple Inc.", "Apple", True),
        ("AMZN", "Amazon.com, Inc.", "About Amazon", True),
        ("KO", "The Coca-Cola Company", "Coca-Cola", True),
        ("NVDA", "NVIDIA Corporation", "NVIDIA Newsroom", True),
        (
            "MSFT",
            "Microsoft Corporation",
            "Microsoft Source",
            True,
        ),  # news.microsoft.com/source (seen live)
        ("MSFT", "Microsoft Corporation", "Microsoft Azure", True),
        ("MSFT", "Microsoft Corporation", "Microsoft Watch", False),
        ("NVDA", "NVIDIA Corporation", "Nvidia Watcher Weekly", False),
        ("AAPL", "Apple Inc.", "AppleInsider", False),
        (
            "NVDA",
            "NVIDIA Corporation",
            "IonQ",
            False,
        ),  # a partner's release about NVIDIA is kept
    ],
)
def test_self_published_bare_company_name(ticker, company, publisher, expected):
    assert us.self_published_matcher(ticker, company)(publisher) is expected


def _mixed_unit_facts() -> dict:
    """A foreign filer tagging CNY plus a USD convenience translation of the latest quarter only."""

    def row(v, s, e):
        return {
            "start": s,
            "end": e,
            "val": v,
            "accn": f"a{e}{v}",
            "fy": 2025,
            "fp": "Q",
            "form": "10-Q",
            "filed": "2025-05-10",
        }

    q = [
        ("2024-01-01", "2024-03-31"),
        ("2024-04-01", "2024-06-30"),
        ("2024-07-01", "2024-09-30"),
        ("2024-10-01", "2024-12-31"),
        ("2025-01-01", "2025-03-31"),
    ]
    rev_cny = [row(700.0 + 70 * i, s, e) for i, (s, e) in enumerate(q)]
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [row(110.0, *q[-1])], "CNY": rev_cny}},
                "OperatingIncomeLoss": {"units": {"CNY": [row(70.0, s, e) for s, e in q]}},
                "NetIncomeLoss": {"units": {"CNY": [row(50.0, s, e) for s, e in q]}},
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [row(1.0, *q[-1])],
                        "CNY/shares": [row(7.0, s, e) for s, e in q],
                    }
                },
            }
        }
    }


def test_xbrl_never_mixes_units_and_pe_needs_usd_eps():
    facts = _mixed_unit_facts()
    rev = us.quarterly_series(facts, "revenue")
    assert set(rev["unit"]) == {"CNY"}  # one unit per series
    assert rev["yoy"].iloc[-1] == pytest.approx(980.0 / 700.0 - 1)  # CNY vs CNY, not USD vs CNY
    snap = us.xbrl_snapshot(facts, date(2025, 6, 1))
    assert "eps_ttm" not in snap  # CNY EPS is never a USD EPS
    assert snap["operating_margin"] == pytest.approx(280.0 / sum(700.0 + 70 * i for i in range(1, 5)))
    close = pd.Series(100.0, index=pd.bdate_range("2025-01-01", "2025-09-01"))
    assert us.trailing_pe_history(facts, close, splits=None) is None
    f = us.Fundamentals()
    assert "eps" not in us.merge_xbrl_snapshot(f, snap, price=100.0) and f.per is None
    # a USD filer with one stray non-USD fact keeps its USD series
    usd = _mixed_unit_facts()
    usd["facts"]["us-gaap"]["Revenues"]["units"] = {
        "USD": usd["facts"]["us-gaap"]["Revenues"]["units"]["CNY"],
        "EUR": [dict(usd["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0])],
    }
    assert set(us.quarterly_series(usd, "revenue")["unit"]) == {"USD"}


def test_fx_cache_rolls_over_when_the_session_bar_completes(provider, monkeypatch):
    # 17:30 ET Thu = 22:30 London: the Thu KRW=X bar is live -> Wed. At 19:30 ET (00:30 London Fri)
    # the Thu bar is complete: the cached Wed stand-in must not be served for the rest of the TTL.
    s = pd.Series(
        [1373.68, 1350.36, 1362.10],
        index=pd.DatetimeIndex(["2026-09-22", "2026-09-23", "2026-09-24"]),
    )
    fetches = []
    monkeypatch.setattr(
        provider,
        "_fetch_krw_series_yf",
        lambda: fetches.append(1) or (s, "Europe/London"),
    )
    monkeypatch.setattr(us, "_now_utc", lambda: pd.Timestamp("2026-09-24T21:30:00Z"))
    assert provider.fx_rate("USD", "KRW") == pytest.approx(1350.36)
    monkeypatch.setattr(us, "_now_utc", lambda: pd.Timestamp("2026-09-24T23:30:00Z"))
    assert provider.fx_rate("USD", "KRW") == pytest.approx(1362.10)
    assert provider.fx_rate("USD", "KRW") == pytest.approx(1362.10) and len(fetches) == 2  # then cached


def test_yahoo_session_holes_are_flagged_not_silent(provider, monkeypatch):
    # Live 2026-09-24: Yahoo returned close=None for 2026-09-22 on KO/XOM/GE/PG/CVX/LIN/NEE (Nasdaq had
    # the bars) and yfinance silently dropped the rows. A hole must be reported, and a missing LAST
    # session must say the latest price is stale.
    monkeypatch.setattr(us, "_now_utc", lambda: NOW)
    spy = _frame()
    ko = spy.drop(pd.Timestamp("2026-09-22"))
    monkeypatch.setattr(
        provider,
        "_fetch_yf_frame",
        lambda s, st: spy.copy() if s == "SPY" else ko.copy(),
    )
    df = provider.prices(aapl_instrument(), date(2019, 9, 1))
    assert df.attrs["missing_sessions"] == ["2026-09-22"]
    notes = provider.drain_notes("AAPL")
    assert any("1거래일 결측(2026-09-22)" in n for n in notes) and not any("최근 거래일" in n for n in notes)
    assert clean_prices(df).attrs["missing_sessions"] == ["2026-09-22"]  # survives the shared loader

    p2 = us.USProvider(provider.settings, DiskCache(provider.cache.root / "holes2"))
    stale = spy.iloc[:-1]
    monkeypatch.setattr(p2, "_fetch_yf_frame", lambda s, st: spy.copy() if s == "SPY" else stale.copy())
    p2.prices(aapl_instrument(), date(2019, 9, 1))
    assert any("최근 거래일 2026-09-23 봉 누락" in n for n in p2.drain_notes("AAPL"))
    # a clean frame raises nothing, and a listing that starts later is not "missing" its pre-IPO history
    p3 = us.USProvider(provider.settings, DiskCache(provider.cache.root / "holes3"))
    monkeypatch.setattr(
        p3,
        "_fetch_yf_frame",
        lambda s, st: spy.copy() if s == "SPY" else spy.iloc[100:].copy(),
    )
    assert p3.prices(aapl_instrument(), date(2019, 9, 1)).attrs["missing_sessions"] == []
    assert p3.drain_notes("AAPL") == []


def _ko_sources():
    """Real KO data captured 2026-09-24: the Yahoo chart has close=None on 2026-09-22 (the live hole)
    and an ex-dividend (0.53) on 2026-09-15; Nasdaq has every bar, unadjusted for dividends."""
    res = fx_json("yahoo_chart_ko_hole.json")
    chart = us.parse_yahoo_chart(res, now=NOW)
    nasdaq = us.parse_nasdaq_rows(fx_json("nasdaq_rows_ko.json"), now=NOW)
    return res, chart, nasdaq


def test_scale_missing_bars_on_real_ko_data_picks_the_dividend_side():
    res, chart, nasdaq = _ko_sources()
    events = us.chart_event_dates(res)
    assert list(events) == [pd.Timestamp("2026-09-15")]
    hole = pd.Timestamp("2026-09-22")
    assert hole not in chart.index and hole in nasdaq.index  # the live Yahoo hole

    # the real hole: Yahoo's chart has no bar either -> Nasdaq's bar (no dividend since -> ratio 1)
    bars, rej = us.scale_missing_bars(chart, [hole], chart, events)
    assert bars == {} and rej == {hole: "원천에도 봉 없음"}
    bars, rej = us.scale_missing_bars(chart, [hole], nasdaq, events)
    assert rej == {} and bars[hole][3] == pytest.approx(88.61, rel=1e-6)
    assert bars[hole][:3] == pytest.approx([87.60, 88.97, 87.455], rel=1e-6)
    assert bars[hole][4] == 17_273_400

    # knock out bars around the ex-date: Yahoo's own adjusted bar is the ground truth
    for d in ("2026-09-14", "2026-09-15"):  # day before the ex-date (factor 0.994) / the ex-date (1.0)
        day = pd.Timestamp(d)
        bars, rej = us.scale_missing_bars(chart.drop(day), [day], nasdaq, events)
        assert rej == {}
        truth = chart.loc[day]
        assert bars[day][3] == pytest.approx(truth["close"], rel=2e-6)  # the official close: exact
        # O/H/L come from Nasdaq's feed (Yahoo's intraday low differed by half a cent on 09-14)
        assert bars[day][:3] == pytest.approx(truth[["open", "high", "low"]].tolist(), rel=1e-4)
    # without the dividend dates the side cannot be decided -> not repaired (never guessed)
    day = pd.Timestamp("2026-09-14")
    bars, rej = us.scale_missing_bars(chart.drop(day), [day], nasdaq, None)
    assert bars == {} and rej == {day: "배당/분할 조정 비율 불명확"}
    # the ratio agreement alone suffices away from an ex-date
    bars, _ = us.scale_missing_bars(chart, [hole], nasdaq, None)
    assert bars[hole][3] == pytest.approx(88.61, rel=1e-6)


def test_scale_missing_bars_rejects_implausible_bars():
    _, chart, nasdaq = _ko_sources()
    hole = pd.Timestamp("2026-09-22")
    bad = nasdaq.copy()
    bad.loc[hole, ["open", "high", "low", "close"]] = [88.8, 89.3, 88.6, 140.0]  # close outside H/L
    assert us.scale_missing_bars(chart, [hole], bad, None)[1] == {hole: "OHLC 불일치"}
    bad.loc[hole, ["open", "high", "low", "close"]] = [150.0, 150.0, 150.0, 150.0]  # +53% vs neighbours
    assert us.scale_missing_bars(chart, [hole], bad, None)[1] == {hole: "인접일 대비 급변 — 검증 실패"}
    assert us.scale_missing_bars(chart, [hole], nasdaq.iloc[0:0], None)[1] == {hole: "원천 데이터 없음"}


def _ko_instrument() -> Instrument:
    return Instrument(
        ticker="KO",
        market=Market.US,
        name="The Coca-Cola Company",
        exchange="NYSE",
        currency="USD",
        yf_symbol="KO",
        benchmark="SPY",
        benchmark_name="S&P 500 (SPY)",
    )


def _ko_provider(provider, monkeypatch, chart_fn, nasdaq_fn):
    res, chart, _ = _ko_sources()
    chart.attrs.update(source="yfinance")  # yfinance serves the same Yahoo bars (and the same hole)
    spy = chart.reindex(chart.index.union([pd.Timestamp("2026-09-22")])).ffill()
    monkeypatch.setattr(us, "_now_utc", lambda: NOW)
    monkeypatch.setattr(provider, "_fetch_yf_frame", lambda s, st: spy.copy() if s == "SPY" else chart.copy())
    calls = {"chart": 0, "nasdaq": 0}

    def _chart(sym, start):
        calls["chart"] += 1
        assert sym == "KO" and start <= date(2026, 8, 23)  # a wide window, never the narrow one
        return chart_fn(res)

    def _nasdaq(sym, start, classes):
        calls["nasdaq"] += 1
        return nasdaq_fn()

    monkeypatch.setattr(provider, "_repair_chart", _chart)
    monkeypatch.setattr(provider, "_repair_nasdaq", _nasdaq)
    return calls


def test_prices_repairs_a_yahoo_hole_from_nasdaq_and_notes_it(provider, monkeypatch):
    _, _, nasdaq = _ko_sources()
    calls = _ko_provider(provider, monkeypatch, lambda res: res, lambda: nasdaq.copy())
    df = provider.prices(_ko_instrument(), date(2019, 9, 1))
    hole = pd.Timestamp("2026-09-22")
    assert df.attrs["missing_sessions"] == [] and df.attrs["repaired_sessions"] == {"2026-09-22": "nasdaq"}
    assert df.loc[hole, "close"] == pytest.approx(88.61, rel=1e-6) and df.index.is_unique
    assert df.index.is_monotonic_increasing and list(df.columns) == PRICE_COLUMNS
    assert df.attrs["source"] == "yfinance" and df.attrs["adjusted"] is True  # the series is not replaced
    notes = provider.data_warnings("KO")
    assert notes == df.attrs["warnings"]
    assert any("Yahoo 결측 1거래일 보정(2026-09-22 나스닥 종가)" in n for n in notes)
    assert not any("거래일 결측(" in n for n in notes)
    cleaned = clean_prices(df)
    assert hole in cleaned.index and cleaned.attrs["repaired_sessions"] == {"2026-09-22": "nasdaq"}
    assert calls == {"chart": 1, "nasdaq": 1}
    # a definitive repair is cached: no refetch on the next call (panel + report load the same name)
    provider.prices(_ko_instrument(), date(2019, 9, 1))
    assert calls == {"chart": 1, "nasdaq": 1}


def test_prices_repair_prefers_the_yahoo_chart_when_it_has_the_bar(provider, monkeypatch):
    _, chart, nasdaq = _ko_sources()
    res = fx_json("yahoo_chart_ko_hole.json")
    i = [pd.Timestamp(t, unit="s", tz="UTC").tz_convert(us.ET).date() for t in res["timestamp"]].index(
        date(2026, 9, 22)
    )
    q = res["indicators"]["quote"][0]
    q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i] = 87.6, 88.97, 87.455, 88.61, 1.7e7
    res["indicators"]["adjclose"][0]["adjclose"][i] = 88.61
    calls = _ko_provider(provider, monkeypatch, lambda _res: res, lambda: nasdaq.copy())
    df = provider.prices(_ko_instrument(), date(2019, 9, 1))
    assert df.attrs["repaired_sessions"] == {"2026-09-22": "yahoo_chart"} and calls["nasdaq"] == 0
    assert any("2026-09-22 Yahoo 차트 재조회" in n for n in df.attrs["warnings"])


def test_prices_unrepairable_hole_keeps_the_warning(provider, monkeypatch):
    _, _, nasdaq = _ko_sources()
    hole = pd.Timestamp("2026-09-22")
    calls = _ko_provider(provider, monkeypatch, lambda res: res, lambda: nasdaq.drop(hole))  # a halt
    df = provider.prices(_ko_instrument(), date(2019, 9, 1))
    assert df.attrs["missing_sessions"] == ["2026-09-22"] and df.attrs["repaired_sessions"] == {}
    notes = provider.data_warnings("KO")
    miss = [n for n in notes if "1거래일 결측(2026-09-22)" in n]
    assert len(miss) == 1 and "재조회 보정 불가" in miss[0] and "nasdaq: 원천에도 봉 없음" in miss[0]
    assert not any("보정(" in n for n in notes)
    provider.prices(_ko_instrument(), date(2019, 9, 1))
    assert calls == {"chart": 1, "nasdaq": 1}  # both sources answered -> the verdict is cached


def test_prices_repair_retries_after_a_transient_source_failure(provider, monkeypatch):
    _, _, nasdaq = _ko_sources()
    state = {"down": True}

    def nasdaq_fn():
        if state["down"]:
            raise RuntimeError("HTTP 503")
        return nasdaq.copy()

    calls = _ko_provider(provider, monkeypatch, lambda res: res, nasdaq_fn)
    df = provider.prices(_ko_instrument(), date(2019, 9, 1))
    assert df.attrs["missing_sessions"] == ["2026-09-22"]
    assert any("nasdaq 조회 실패" in n for n in df.attrs["warnings"])
    state["down"] = False  # not cached as a verdict: the next load repairs it
    df = provider.prices(_ko_instrument(), date(2019, 9, 1))
    assert df.attrs["repaired_sessions"] == {"2026-09-22": "nasdaq"} and calls["nasdaq"] == 2


def test_unadjusted_yahoo_chart_note_names_the_real_source(provider, monkeypatch):
    monkeypatch.setattr(us, "_now_utc", lambda: NOW)
    monkeypatch.setattr(
        provider,
        "_fetch_yf_frame",
        lambda s, st: (_ for _ in ()).throw(RuntimeError("HTTP 500")),
    )
    monkeypatch.setattr(
        provider,
        "_fetch_chart_frame",
        lambda s, st: _frame(source="yahoo_chart", adjusted=False),
    )
    df = provider.prices(aapl_instrument(), date(2019, 9, 1))
    notes = provider.drain_notes("AAPL")
    assert df.attrs["source"] == "yahoo_chart" and df.attrs["adjusted"] is False
    assert any("yahoo_chart" in n and "배당 미조정" in n for n in notes) and not any("Nasdaq" in n for n in notes)


# =============================================================================== earnings date


AAPL_NEXT_ER = 1793304000  # yf_info_aapl.json earningsTimestampStart: 2026-10-29 20:00Z = 16:00 EDT


def _cal_frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """Shape of yfinance get_earnings_dates (live 2026-09-24): ET-aware index, NaN = not reported yet."""
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="America/New_York") for t, _ in rows])
    return pd.DataFrame(
        {
            "EPS Estimate": [1.9] * len(rows),
            "Reported EPS": [r for _, r in rows],
            "Surprise(%)": [np.nan] * len(rows),
        },
        index=pd.Index(idx, name="Earnings Date"),
    )


def test_epoch_local_date_uses_the_exchange_timezone():
    # Yahoo date-only events sit at 20:00 UTC: the ET date is right; the machine-tz date that
    # yfinance's Ticker.calendar uses is one day late when run in Seoul (live: AAPL 10-29 -> 10-30)
    assert us.epoch_local_date(AAPL_NEXT_ER, us.ET) == date(2026, 10, 29)
    assert us.epoch_local_date(AAPL_NEXT_ER, "Asia/Seoul") == date(2026, 10, 30)
    assert us.epoch_local_date(1792499400, us.ET) == date(2026, 10, 20)  # KO 08:30 ET
    assert us.epoch_local_date(0, us.ET) is None and us.epoch_local_date(None, us.ET) is None
    assert us.epoch_local_date("garbage", us.ET) is None


def test_parse_earnings_calendar_splits_upcoming_and_reported():
    df = _cal_frame(
        [
            ("2026-10-29 16:00", np.nan),
            ("2026-07-30 16:00", 2.02),
            ("2026-08-08 16:00", np.nan),  # BRK-style stale duplicate, in the past
        ]
    )
    out = us.parse_earnings_calendar(df, us.ET)
    assert out["upcoming"] == [date(2026, 8, 8), date(2026, 10, 29)]
    assert out["reported"] == [date(2026, 7, 30)]
    assert us.parse_earnings_calendar(None, us.ET) == {"upcoming": [], "reported": []}


TODAY = date(2026, 9, 24)


def _info(**kw) -> dict:
    base = {
        "earningsTimestamp": 1785441600,  # 2026-07-30 (last release, in the past)
        "earningsTimestampStart": AAPL_NEXT_ER,
        "earningsTimestampEnd": AAPL_NEXT_ER,
        "isEarningsDateEstimate": False,
    }
    base.update(kw)
    return {k: v for k, v in base.items() if v is not None}


def _epoch(d: str) -> int:
    return int(pd.Timestamp(f"{d} 20:00", tz="UTC").timestamp())


@pytest.mark.parametrize(
    "info,cal,expected",
    [
        # confirmed by Yahoo, calendar page agrees
        (_info(), {"upcoming": [date(2026, 10, 29)]}, (date(2026, 10, 29), False)),
        # calendar page unavailable: Yahoo's own confirmation stands
        (_info(), None, (date(2026, 10, 29), False)),
        # sources disagree -> earliest, flagged estimate
        (_info(), {"upcoming": [date(2026, 10, 28)]}, (date(2026, 10, 28), True)),
        (_info(), {"upcoming": [date(2026, 11, 2)]}, (date(2026, 10, 29), True)),
        # Yahoo says it is an estimate (BRK-B live) / flag missing
        (_info(isEarningsDateEstimate=True), None, (date(2026, 10, 29), True)),
        (_info(isEarningsDateEstimate=None), None, (date(2026, 10, 29), True)),
        # a start..end window is not a confirmed date
        (_info(earningsTimestampEnd=_epoch("2026-11-03")), None, (date(2026, 10, 29), True)),
        # calendar page only (Yahoo info failed) -> estimate
        (None, {"upcoming": [date(2026, 10, 29)]}, (date(2026, 10, 29), True)),
        # everything in the past / nothing scheduled -> None (never guessed)
        (_info(earningsTimestampStart=1785441600, earningsTimestampEnd=1785441600), None, (None, None)),
        (None, None, (None, None)),
        # a year-ahead placeholder is not a schedule
        (
            _info(earningsTimestampStart=_epoch("2027-09-01"), earningsTimestampEnd=_epoch("2027-09-01")),
            None,
            (None, None),
        ),
    ],
)
def test_resolve_next_earnings(info, cal, expected):
    assert us.resolve_next_earnings(info, cal, TODAY, us.ET)[:2] == expected


def test_resolve_next_earnings_detail_names_the_disagreement():
    _, _, agree = us.resolve_next_earnings(_info(), {"upcoming": [date(2026, 10, 29)]}, TODAY, us.ET)
    _, _, clash = us.resolve_next_earnings(_info(), {"upcoming": [date(2026, 10, 28)]}, TODAY, us.ET)
    assert "DISAGREE" not in agree and "info=2026-10-29" in agree
    assert "DISAGREE" in clash and "calendar=2026-10-28" in clash


def test_resolve_next_earnings_drops_a_release_that_is_already_out():
    # release day, after the print: Yahoo's quote fields still say "today", the calendar page already
    # shows the reported EPS -> the next date is next quarter's (an estimate), not today
    today = date(2026, 10, 29)
    cal = {"upcoming": [date(2027, 1, 28)], "reported": [today]}
    assert us.resolve_next_earnings(_info(), cal, today, us.ET)[:2] == (
        date(2027, 1, 28),
        True,
    )
    # before the print the page has no reported row -> today, confirmed
    cal_before = {"upcoming": [today], "reported": []}
    assert us.resolve_next_earnings(_info(), cal_before, today, us.ET)[:2] == (today, False)


def test_provider_fundamentals_next_earnings_date(provider, monkeypatch):
    monkeypatch.setattr(us, "_now_utc", lambda: NOW)
    monkeypatch.setattr(
        provider,
        "_info",
        lambda sym: {"as_of": "2026-09-24", "info": fx_json("yf_info_aapl.json")},
    )
    monkeypatch.setattr(provider, "_companyfacts", lambda t: fx_json("sec_companyfacts_nvda.json"))
    monkeypatch.setattr(provider, "_sector_per", lambda t, s: None)
    seen = []
    monkeypatch.setattr(
        provider,
        "_earnings_calendar",
        lambda sym, tz: seen.append((sym, tz)) or {"upcoming": [date(2026, 10, 29)], "reported": []},
    )
    f = provider.fundamentals(aapl_instrument())
    assert (f.next_earnings_date, f.next_earnings_is_estimate) == (date(2026, 10, 29), False)
    assert seen == [("AAPL", "America/New_York")]
    assert not any("실적발표일" in n for n in provider.data_warnings("AAPL"))

    # calendar page down but Yahoo info has the date: served from info, only logged
    def down(sym, tz):
        raise RuntimeError("HTTP 503")

    monkeypatch.setattr(provider, "_earnings_calendar", down)
    f2 = provider.fundamentals(aapl_instrument())
    assert f2.next_earnings_date == date(2026, 10, 29) and f2.next_earnings_is_estimate is False
    assert not any("실적발표일" in n for n in provider.data_warnings("AAPL"))

    # calendar page down AND no usable date in info: unknown schedule is reported, not silent
    info = fx_json("yf_info_aapl.json")
    for k in ("earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd"):
        info.pop(k)
    monkeypatch.setattr(provider, "_info", lambda sym: {"as_of": "2026-09-24", "info": info})
    f3 = provider.fundamentals(aapl_instrument())
    assert f3.next_earnings_date is None and f3.next_earnings_is_estimate is None
    assert any("실적발표일" in n and "미확인" in n for n in provider.data_warnings("AAPL"))


def test_etf_has_no_earnings_lookup(provider, monkeypatch):
    monkeypatch.setattr(us, "_now_utc", lambda: NOW)
    monkeypatch.setattr(
        provider,
        "_info",
        lambda sym: {"as_of": "2026-09-24", "info": {"quoteType": "ETF", "longName": "SPDR S&P 500"}},
    )
    monkeypatch.setattr(provider, "_sector_per", lambda t, s: None)
    monkeypatch.setattr(provider, "_earnings_calendar", lambda s, tz: pytest.fail("no earnings for funds"))
    provider.settings.data.sec_user_agent = ""
    f = provider.fundamentals(aapl_instrument())
    assert f.next_earnings_date is None


def test_sec_ua_missing_with_missing_fields_is_noted(provider, monkeypatch):
    provider.settings.data.sec_user_agent = ""
    info = fx_json("yf_info_aapl.json")
    info.pop("revenueGrowth")
    monkeypatch.setattr(provider, "_info", lambda sym: {"as_of": "2026-09-24", "info": info})
    monkeypatch.setattr(provider, "_sector_per", lambda t, s: None)
    f = provider.fundamentals(aapl_instrument())
    assert f.revenue_growth_yoy is None
    assert any("SEC_USER_AGENT 미설정" in n and "revenue_growth_yoy" in n for n in provider.data_warnings("AAPL"))


def test_sector_per_failure_is_noted(provider, monkeypatch):
    monkeypatch.setattr(
        provider,
        "_info",
        lambda sym: {"as_of": "2026-09-24", "info": fx_json("yf_info_aapl.json")},
    )
    monkeypatch.setattr(provider, "_companyfacts", lambda t: fx_json("sec_companyfacts_nvda.json"))
    monkeypatch.setattr(provider, "_sector_per", lambda t, s: (_ for _ in ()).throw(RuntimeError("pool")))
    f = provider.fundamentals(aapl_instrument())
    assert f.sector_per is None
    assert any("업종 PER 산출 실패" in n for n in provider.data_warnings("AAPL"))


# =============================================================================== warnings plumbing


def test_data_warnings_is_non_destructive_and_drain_notes_clears(provider):
    provider._note("AAPL", "가격: 테스트 경고")
    provider._note(us._GLOBAL_NOTES, "환율: 네이버 대체")
    assert provider.data_warnings("aapl") == ["가격: 테스트 경고", "환율: 네이버 대체"]
    assert provider.data_warnings("AAPL") == provider.data_warnings("AAPL")  # repeatable
    assert provider.drain_notes("AAPL") == ["가격: 테스트 경고", "환율: 네이버 대체"]
    assert provider.data_warnings("AAPL") == ["환율: 네이버 대체"]  # provider-wide notes stay
    assert provider.data_warnings("BRK.B") == ["환율: 네이버 대체"]  # normalised key


def test_price_frames_carry_warnings_matching_the_notes(provider, monkeypatch):
    monkeypatch.setattr(us, "_now_utc", lambda: NOW)
    spy = _frame(source="nasdaq", adjusted=False)
    aapl = spy.drop(pd.Timestamp("2026-09-22"))
    monkeypatch.setattr(provider, "_fetch_yf_frame", lambda s, st: (_ for _ in ()).throw(RuntimeError("429")))
    monkeypatch.setattr(provider, "_fetch_chart_frame", lambda s, st: (_ for _ in ()).throw(RuntimeError("429")))
    monkeypatch.setattr(
        provider,
        "_fetch_nasdaq_frame",
        lambda s, st, cls: (spy if s == "SPY" else aapl).copy(),
    )
    inst = aapl_instrument()
    px = provider.prices(inst, date(2019, 9, 1))
    bench = provider.benchmark(inst, date(2019, 9, 1))
    assert isinstance(px.attrs["warnings"], list) and len(px.attrs["warnings"]) == 2
    assert any("Nasdaq" in w for w in px.attrs["warnings"])
    assert any("1거래일 결측" in w for w in px.attrs["warnings"])
    assert bench.attrs["warnings"] and all(w.startswith("SPY ") for w in bench.attrs["warnings"])
    notes = provider.data_warnings("AAPL")
    assert all(w in notes for w in px.attrs["warnings"])
    assert all(f"벤치마크: {w}" in notes for w in bench.attrs["warnings"])
    # a clean frame still carries an (empty) list
    p2 = us.USProvider(provider.settings, DiskCache(provider.cache.root / "clean"))
    monkeypatch.setattr(p2, "_fetch_yf_frame", lambda s, st: _frame())
    assert p2.prices(inst, date(2019, 9, 1)).attrs["warnings"] == []
    assert p2.benchmark(inst, date(2019, 9, 1)).attrs["warnings"] == []


def test_load_market_data_merges_us_warnings_once(provider, settings, monkeypatch):
    """Offline end-to-end through the shared loader: every degraded path reaches MarketData.warnings,
    and a note present both in frame attrs and data_warnings appears exactly once."""
    from quantcio.data import base

    monkeypatch.setattr(us, "_now_utc", lambda: NOW)
    monkeypatch.setattr(base, "get_provider", lambda m, s, c: provider)
    monkeypatch.setattr(
        provider,
        "_info",
        lambda sym: {"as_of": "2026-09-24", "info": fx_json("yf_info_aapl.json")},
    )
    monkeypatch.setattr(provider, "_companyfacts", lambda t: fx_json("sec_companyfacts_nvda.json"))
    monkeypatch.setattr(provider, "_sector_per", lambda t, s: None)
    monkeypatch.setattr(provider, "valuation_history", lambda inst, start: None)
    spy = _frame(n=400)
    monkeypatch.setattr(
        provider,
        "_fetch_yf_frame",
        lambda s, st: spy.copy() if s == "SPY" else spy.drop(pd.Timestamp("2026-09-22")),
    )
    monkeypatch.setattr(
        provider,
        "_fetch_krw_series_yf",
        lambda: (_ for _ in ()).throw(RuntimeError("429")),
    )
    monkeypatch.setattr(provider, "_fetch_usdkrw_naver", lambda: us.parse_naver_fx(fx_json("naver_fx_usdkrw.json")))
    md = base.load_market_data("AAPL", settings, provider.cache, base_currency="KRW", include_text=False)
    w = md.warnings
    assert len(w) == len(set(w))
    assert sum("1거래일 결측(2026-09-22)" in x for x in w) == 1  # attrs + data_warnings -> once
    assert any("네이버" in x and "환율" in x for x in w)  # provider-wide FX fallback reaches the ticker
    assert md.fundamentals.next_earnings_date == date(2026, 10, 29)
    assert isinstance(md.prices.attrs.get("warnings"), list)  # survives clean_prices


# =============================================================================== report language

HANGUL = re.compile("[가-힣]")


def in_lang(provider: us.USProvider, lang: str) -> us.USProvider:
    """The provider with a deep copy of its settings set to ``report.language = lang``."""
    s = provider.settings.model_copy(deep=True)
    s.report.language = lang
    provider.settings = s
    return provider


def assert_lang(texts: list[str], lang: str) -> None:
    assert texts
    for t in texts:
        if lang == "en":
            assert HANGUL.search(t) is None, t
        else:
            assert HANGUL.search(t) is not None, t


def test_module_helpers_follow_lang_keyword():
    _, chart, nasdaq = _ko_sources()
    hole = pd.Timestamp("2026-09-22")
    assert us.scale_missing_bars(chart, [hole], chart, None, lang="en")[1] == {hole: "no bar in the source either"}
    assert us.scale_missing_bars(chart, [hole], nasdaq.iloc[0:0], None, lang="en")[1] == {hole: "no source data"}
    assert us.scale_missing_bars(chart, [hole], nasdaq.iloc[0:0], None)[1] == {hole: "원천 데이터 없음"}
    with pytest.raises(us.UnsupportedInstrumentError) as exc:
        us.instrument_from_info("0P0001", {"quoteType": "MUTUALFUND", "longName": "Some Fund"}, lang="en")
    assert str(exc.value) == "0P0001: unsupported instrument type MUTUALFUND (stocks/ETFs only)"
    with pytest.raises(NotImplementedError) as exc:
        us.usdkrw_fx("USD", "JPY", lambda: 1300.0, lang="en")
    assert HANGUL.search(str(exc.value)) is None
    payload = {"rate": 1370.5, "date": "2026-09-24", "source": "naver"}
    assert us.usdkrw_fallback_note(payload) == (
        "환율: Yahoo(KRW=X) 실패 → 네이버 고시환율(하나은행) USD/KRW 1,370.50 (2026-09-24) 사용 — 정의 차이 최대 ~24원"
    )
    assert us.usdkrw_fallback_note(payload, lang="en") == (
        "FX: Yahoo (KRW=X) failed → using the Naver posted rate (Hana Bank) USD/KRW 1,370.50 (2026-09-24) "
        "— definitions differ by up to ~24 KRW"
    )
    with pytest.raises(RuntimeError) as exc:
        us.usdkrw_payload(
            lambda: (_ for _ in ()).throw(ValueError("down")),
            lambda: pd.Series(dtype="float64", index=pd.DatetimeIndex([])),
            now=NOW,
            lang="en",
        )
    assert str(exc.value).startswith("USD/KRW FX rate unavailable (yahoo: ValueError: down;")
    assert HANGUL.search(str(exc.value)) is None


@pytest.mark.parametrize("lang", ["ko", "en"])
def test_price_notes_follow_report_language(provider, monkeypatch, lang):
    in_lang(provider, lang)
    monkeypatch.setattr(us, "_now_utc", lambda: NOW)
    old = _frame(source="nasdaq", adjusted=False)
    old.index = old.index - pd.Timedelta(30, "D")
    monkeypatch.setattr(provider, "_fetch_yf_frame", lambda s, st: (_ for _ in ()).throw(RuntimeError("429")))
    monkeypatch.setattr(provider, "_fetch_chart_frame", lambda s, st: (_ for _ in ()).throw(RuntimeError("429")))
    monkeypatch.setattr(provider, "_fetch_nasdaq_frame", lambda s, st, cls: old.copy())
    df = provider.prices(aapl_instrument(), date(2019, 9, 1))
    b = provider.benchmark(aapl_instrument(), date(2019, 9, 1))
    notes = provider.drain_notes("AAPL")
    assert_lang(notes, lang)
    assert_lang(df.attrs["warnings"], lang)
    expect = {
        "ko": ["가격: Yahoo 실패로 Nasdaq 대체 — 분할만 반영, 배당 미조정 종가", "가격: 최근 봉", "벤치마크: SPY"],
        "en": ["Price: Nasdaq used after Yahoo failed — split-adjusted only", "Price: Latest bar", "Benchmark: SPY"],
    }[lang]
    assert all(any(n.startswith(e) for n in notes) for e in expect)
    prefix = {"ko": "벤치마크: ", "en": "Benchmark: "}[lang]
    assert all(f"{prefix}{w}" in notes for w in b.attrs["warnings"])  # == the loader's prefixed text


@pytest.mark.parametrize("lang", ["ko", "en"])
def test_session_repair_notes_follow_report_language(provider, monkeypatch, lang):
    _, _, nasdaq = _ko_sources()
    hole = pd.Timestamp("2026-09-22")
    _ko_provider(in_lang(provider, lang), monkeypatch, lambda res: res, lambda: nasdaq.drop(hole))
    df = provider.prices(_ko_instrument(), date(2019, 9, 1))
    notes = provider.data_warnings("KO")
    assert_lang(notes, lang)
    assert df.attrs["warnings"] == notes
    expect = {
        "ko": "가격: SPY 거래일 대비 1거래일 결측(2026-09-22)",
        "en": "Price: 1 session(s) missing vs the SPY calendar (2026-09-22)",
    }[lang]
    miss = [n for n in notes if n.startswith(expect)]
    assert len(miss) == 1
    why = {"ko": "nasdaq: 원천에도 봉 없음", "en": "nasdaq: no bar in the source either"}[lang]
    assert why in miss[0]


def test_resolve_fundamentals_news_fx_notes_are_english(provider, monkeypatch):
    in_lang(provider, "en")
    monkeypatch.setattr(provider, "_info", lambda sym: (_ for _ in ()).throw(RuntimeError("HTTP 429")))
    monkeypatch.setattr(provider, "_chart_meta", lambda sym: fx_json("yahoo_chart_aapl.json")["meta"])
    provider.resolve("aapl")
    monkeypatch.setattr(provider, "_companyfacts", lambda t: fx_json("sec_companyfacts_nvda.json"))
    monkeypatch.setattr(provider, "_cached_prices", lambda s, st, c: _frame(close=200.0))
    monkeypatch.setattr(us, "_now_utc", lambda: NOW)
    provider.fundamentals(nvda_instrument())
    monkeypatch.setattr(provider, "_fetch_yf_news", lambda s: fx_json("yf_news_nvda.json"))
    monkeypatch.setattr(provider, "_fetch_google_news", lambda *a: (_ for _ in ()).throw(RuntimeError("HTTP 503")))
    provider.news(nvda_instrument(), utc(2026, 9, 3), 30)
    monkeypatch.setattr(provider, "_fetch_krw_series_yf", lambda: (_ for _ in ()).throw(RuntimeError("HTTP 429")))
    monkeypatch.setattr(provider, "_fetch_usdkrw_naver", lambda: us.parse_naver_fx(fx_json("naver_fx_usdkrw.json")))
    provider.fx_rate("USD", "KRW")
    notes = provider.drain_notes("AAPL") + provider.drain_notes("NVDA")
    assert_lang(notes, "en")
    for e in (
        "Instrument info: Yahoo info failed (RuntimeError) → using chart metadata",
        "Fundamentals: Yahoo info failed (RuntimeError) → SEC XBRL fallback",
        "News: Google News fetch failed",
        "FX: Yahoo (KRW=X) failed",
    ):
        assert any(n.startswith(e) for n in notes), e

    p2 = us.USProvider(provider.settings, DiskCache(provider.cache.root / "en2"))
    for name in ("_fetch_yf_news", "_fetch_yahoo_rss", "_fetch_google_news"):
        monkeypatch.setattr(p2, name, lambda *a: (_ for _ in ()).throw(RuntimeError("HTTP 503")))
    with pytest.raises(RuntimeError, match="All news sources failed"):
        p2.news(nvda_instrument(), utc(2026, 9, 3), 30)
    p2.settings.data.sec_user_agent = ""
    with pytest.raises(RuntimeError) as exc:
        p2.filings(nvda_instrument(), SINCE, 6)
    assert str(exc.value).startswith("SEC_USER_AGENT not set") and HANGUL.search(str(exc.value)) is None


def test_filing_notes_are_english(provider, tmp_path, monkeypatch):
    stale_quote = tmp_path / "quote.html"
    stale_quote.write_text('"/earnings/call-transcripts/2026/05/28/nvidia-nvda-q1-2027-earnings-call-transcript/"')
    monkeypatch.setattr(in_lang(provider, "en"), "_sec_get", FakeSec(fail=("q2fy27pr.htm",)))
    monkeypatch.setattr(
        provider, "_fool_page", lambda url: stale_quote.read_text() if "/quote/nasdaq/nvda/" in url else None
    )
    provider.filings(nvda_instrument(), SINCE, 6)
    notes = provider.drain_notes("NVDA")
    assert_lang(notes, "en")
    assert any(n.startswith("Earnings release (8-K 2.02) fetch failed: ") for n in notes)
    assert any("not yet published" in n and "2026-05-28" in n for n in notes)


# =============================================================================== live


@pytest.fixture
def live_provider(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(settings.data, "sec_user_agent", LIVE_SEC_UA)
    return us.USProvider(settings, DiskCache(tmp_path / "live-cache"))


def _start() -> date:
    return date.today() - timedelta(days=int(365.25 * 7) + 30)


@pytest.mark.live
@pytest.mark.parametrize("ticker", ["AAPL", "NVDA"])
def test_live_resolve_prices_fundamentals(live_provider, ticker):
    p = live_provider
    inst = p.resolve(ticker)
    assert inst.ticker == ticker and inst.currency == "USD" and inst.exchange == "NASDAQ" and inst.sector
    px = clean_prices(p.prices(inst, _start()))
    assert len(px) >= 1500 and px.attrs["adjusted"] is True
    assert px.index[-1].date() <= us.last_completed_us_session()
    bench = clean_prices(p.benchmark(inst, _start()))
    assert len(bench) >= 1500 and bench.attrs["adjusted"] is True
    f = p.fundamentals(inst)
    assert f.per and f.per > 0 and f.eps and f.analyst_count and f.analyst_count > 5
    assert f.dividend_yield is not None and 0 <= f.dividend_yield < 0.05
    assert f.analyst_rating and "(1=강력매수)" in f.analyst_rating
    assert f.sector_per is None or 5 < f.sector_per < 200
    assert "SEC XBRL" in f.source
    assert "missing_sessions" in px.attrs
    # Yahoo session holes are a data issue (reported), anything else would be a degraded path
    assert [n for n in p.drain_notes(ticker) if "결측" not in n and "봉 누락" not in n] == []


@pytest.mark.live
@pytest.mark.parametrize("ticker", ["AAPL", "NVDA"])
def test_live_filings(live_provider, ticker):
    p = live_provider
    inst = p.resolve(ticker)
    since = datetime.now(UTC) - timedelta(days=120)
    items = p.filings(inst, since, 6)
    notes = p.drain_notes(ticker)
    assert len(items) >= 3
    er = [i for i in items if i.is_earnings and i.form_type.startswith("8-K")]
    assert er and "[EX-99.1]" in er[0].text
    assert all(len(i.text) <= p.settings.data.max_filing_chars for i in items)
    assert any("위험요인" in i.title for i in items) and any("MD&A" in i.title for i in items)
    has_tr = any(i.form_type == "Earnings Call Transcript" for i in items)
    assert has_tr or any("녹취록" in n for n in notes)  # missing transcript is never silent


@pytest.mark.live
@pytest.mark.parametrize("ticker", ["XOM", "MSFT", "KO"])
def test_live_fool_links_are_the_companys_own(live_provider, ticker):
    # quote pages link other companies' transcripts (XOM's page carried NRP on 2026-09-24)
    links = live_provider._fool_links(live_provider.resolve(ticker))
    assert links and all(f"-{ticker.lower()}-q" in d["url"] for d in links)
    newest = links[0]
    assert pd.Timestamp(newest["published"]) >= pd.Timestamp.now() - pd.Timedelta(200, "D")  # new slug format parsed


@pytest.mark.live
def test_live_news(live_provider):
    p = live_provider
    inst = p.resolve("AAPL")
    since = datetime.now(UTC) - timedelta(days=21)
    items = p.news(inst, since, 30)
    assert len(items) >= 5
    assert all(i.published_at >= since and i.published_at.utcoffset() == timedelta(0) for i in items)
    assert [i.published_at for i in items] == sorted((i.published_at for i in items), reverse=True)
    rel = us.NewsRelevance("AAPL", inst.name)
    assert all(rel.is_relevant(f"{i.title} {i.summary}") for i in items)


@pytest.mark.live
def test_live_valuation_history_nvda(live_provider):
    p = live_provider
    inst = p.resolve("NVDA")
    vh = p.valuation_history(inst, _start())
    assert vh is not None and len(vh) >= 250
    jumps = (vh["per"] / vh["per"].shift(1)).dropna()
    assert jumps.between(1 / 3, 3).all()  # 2021 4:1 and 2024 10:1 splits leave no artefacts
    trailing = p.fundamentals(inst).per
    assert vh["per"].iloc[-1] == pytest.approx(trailing, rel=0.4)


@pytest.mark.live
@pytest.mark.parametrize("ticker", ["AAPL", "NVDA"])
def test_live_next_earnings_date(live_provider, ticker):
    p = live_provider
    f = p.fundamentals(p.resolve(ticker))
    today = us.today_et()
    assert f.next_earnings_date is not None and isinstance(f.next_earnings_is_estimate, bool)
    assert today <= f.next_earnings_date <= today + timedelta(days=us.EARNINGS_MAX_AHEAD_DAYS)
    cal = p._earnings_calendar(ticker, us.ET)  # the cross-check source answers live
    assert cal["upcoming"] or cal["reported"]


@pytest.mark.live
@pytest.mark.parametrize("ticker", ["AAPL", "NVDA"])
def test_live_load_market_data_warnings(settings, tmp_path, monkeypatch, ticker):
    from quantcio.data.base import load_market_data

    monkeypatch.setattr(settings.data, "sec_user_agent", LIVE_SEC_UA)
    md = load_market_data(ticker, settings, DiskCache(tmp_path / "lmd"), base_currency="KRW")
    assert isinstance(md.warnings, list) and len(md.warnings) == len(set(md.warnings))
    assert isinstance(md.prices.attrs.get("warnings"), list)
    assert md.fundamentals.next_earnings_date is not None
    print(ticker, md.fundamentals.next_earnings_date, md.fundamentals.next_earnings_is_estimate, md.warnings)


@pytest.mark.live
def test_live_fx_identical_across_providers(live_provider, settings, tmp_path):
    from quantcio.data import kr

    krp = kr.KRProvider(settings, DiskCache(tmp_path / "kr-live"))  # separate cache: independent fetch
    usd_krw = live_provider.fx_rate("USD", "KRW")
    assert krp.fx_rate("USD", "KRW") == usd_krw
    assert krp.fx_rate("KRW", "USD") == live_provider.fx_rate("KRW", "USD") == 1.0 / usd_krw


@pytest.mark.live
@pytest.mark.parametrize("ticker", ["KO", "XOM", "AAPL"])
def test_live_recent_yahoo_holes_are_repaired_and_flagged(live_provider, ticker):
    # live 2026-09-24: 43/60 universe names (KO, XOM, ...) lacked 2026-09-22 on Yahoo; Nasdaq had it
    p = live_provider
    inst = p.resolve(ticker)
    px = p.prices(inst, _start())
    spy = p.benchmark(inst, _start())
    recent = set(spy.index[-60:])
    assert not [d for d in px.attrs["missing_sessions"] if pd.Timestamp(d) in recent]
    assert spy.index[-60:].difference(px.index).empty
    rep = px.attrs["repaired_sessions"]
    assert all(pd.Timestamp(d) in spy.index for d in rep)
    if rep:
        assert any("보정(" in n for n in px.attrs["warnings"])
        for d in rep:  # a repaired close sits within 15% of its neighbours
            i = px.index.get_loc(pd.Timestamp(d))
            window = px["close"].iloc[max(0, i - 1) : i + 2]
            assert (window / px["close"].iloc[i]).between(0.85, 1.15).all()
    print(ticker, "repaired", rep)


@pytest.mark.live
def test_live_fx_and_universe(live_provider):
    p = live_provider
    rate = p.fx_rate("USD", "KRW")
    assert 1000 < rate < 2000 and p.fx_rate("KRW", "USD") == pytest.approx(1 / rate)
    for t in us.interleaved_universe():  # every static name is a live SEC registrant
        p._cik(t)


# ============================================ QA regression: .info ratios priced at the live intraday quote


def _intraday_info() -> dict:
    """TSLA .info captured live 2026-09-24 10:38 ET (regular session in progress)."""
    return {
        "symbol": "TSLA",
        "quoteType": "EQUITY",
        "exchangeTimezoneName": "America/New_York",
        "sector": "Consumer Cyclical",
        "marketState": "REGULAR",
        "currentPrice": 378.175,
        "regularMarketPrice": 378.175,
        "previousClose": 380.12,
        "regularMarketPreviousClose": 380.12,
        "regularMarketTime": 1790260731,  # 2026-09-24 14:38:51 UTC
        "trailingPE": 343.79544,  # = 378.175 / 1.10 (live quote)
        "trailingEps": 1.1,
        "forwardPE": 172.11363,
        "forwardEps": 2.19728,
        "priceToBook": 17.19368,
        "bookValue": 21.995,
        "marketCap": 1493620097024,
    }


@pytest.mark.parametrize(
    "now,expect_prev",
    [
        ("2026-09-24T14:40:00Z", True),  # 10:40 ET, session in progress -> previous (completed) close
        ("2026-09-24T20:10:00Z", True),  # 16:10 ET, inside the 20-min buffer: the bar is not final yet
        ("2026-09-24T21:00:00Z", False),  # 17:00 ET: the quote is today's completed close
    ],
)
def test_session_anchor_price_uses_the_last_completed_close(now, expect_prev):
    anchor, quote = us.session_anchor_price(_intraday_info(), now=pd.Timestamp(now))
    assert quote == 378.175
    assert anchor == (380.12 if expect_prev else 378.175)


def test_session_anchor_price_pre_market_and_missing_fields():
    info = fx_json("yf_info_aapl.json")  # quote time 2026-09-23 16:00 ET, i.e. a completed close
    assert us.session_anchor_price(info, now=NOW) == (337.02, 337.02)
    info = _intraday_info()
    info.pop("regularMarketTime")
    assert us.session_anchor_price(info, now=pd.Timestamp("2026-09-24T14:40:00Z")) == (378.175, 378.175)


def test_fundamentals_from_info_reprices_ratios_to_the_anchor():
    info = _intraday_info()
    live = us.fundamentals_from_info(info)
    f = us.fundamentals_from_info(info, anchor=380.12)
    k = 380.12 / 378.175
    assert live.per == pytest.approx(343.79544)  # no anchor: unchanged (backward compatible)
    assert f.per == pytest.approx(380.12 / 1.1, rel=1e-6)  # completed close / TTM EPS
    assert f.forward_per == pytest.approx(380.12 / 2.19728, rel=1e-6)  # completed close / forward EPS
    assert f.pbr == pytest.approx(17.19368 * k)
    assert f.market_cap == pytest.approx(1493620097024 * k)
    assert f.eps == live.eps and f.bps == live.bps  # per-share figures do not depend on the price


def test_provider_fundamentals_reprices_an_intraday_quote(provider, monkeypatch):
    """Live bug: a US report run during the US session (22:30-05:00 KST) valued the stock at the live quote
    while its price series ended at the previous close, and two runs on one price snapshot disagreed."""
    peer = _intraday_info() | {"trailingPE": 30.0, "trailingEps": 378.175 / 30.0}  # consistent live P/E
    infos = {"TSLA": _intraday_info(), "GM": peer, "F": peer, "HD": peer, "MCD": peer, "NKE": peer}
    monkeypatch.setattr(provider, "_info", lambda s: {"as_of": "2026-09-24", "info": infos.get(s, peer)})
    monkeypatch.setattr(provider, "_companyfacts", lambda t: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(us, "_now_utc", lambda: pd.Timestamp("2026-09-24T14:40:00Z"))
    inst = Instrument(
        ticker="TSLA",
        market=Market.US,
        name="Tesla",
        currency="USD",
        yf_symbol="TSLA",
        benchmark="SPY",
        benchmark_name="S&P 500 (SPY)",
    )
    f = provider.fundamentals(inst)
    assert f.per == pytest.approx(380.12 / 1.1, rel=1e-6)
    # the sector-peer P/Es are re-priced the same way (every static Consumer Cyclical peer gets `peer`)
    assert provider._sector_per("TSLA", "Consumer Cyclical") == pytest.approx(30.0 * 380.12 / 378.175)
    assert f.sector_per == pytest.approx(30.0 * 380.12 / 378.175)
