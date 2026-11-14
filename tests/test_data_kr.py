"""Tests for quantcio.data.kr (KRProvider).

Offline tests replay real responses saved in tests/fixtures/kr (captured 2026-09-24 KST, the Chuseok
holiday; latest KRX session 2026-09-23) through a fake HTTP session and a fake yfinance module.
Tests marked ``live`` hit the real sources (run with ``-m live``).
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import pytest
import requests

from quantcio.data import kr, us
from quantcio.data.base import clean_prices
from quantcio.data.cache import DiskCache
from quantcio.data.kr import KST, KRProvider
from quantcio.models import (
    PRICE_COLUMNS,
    FilingItem,
    Fundamentals,
    Instrument,
    Market,
    NewsItem,
)

FIX = Path(__file__).parent / "fixtures" / "kr"
HOLIDAY_2PM = datetime(2026, 9, 24, 14, 0, tzinfo=KST)  # last bar = 2026-09-23 (final)


def fx_text(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def fx_json(name: str) -> Any:
    return json.loads(fx_text(name))


# ----------------------------------------------------------------------------- fakes


class FakeResponse:
    def __init__(
        self,
        body: bytes | str | dict | list,
        status: int = 200,
        content_type: str | None = None,
        encoding: str | None = None,
        url: str = "",
    ):
        if isinstance(body, dict | list):
            self.content = json.dumps(body, ensure_ascii=False).encode("utf-8")
            ctype = "application/json;charset=UTF-8"
        elif isinstance(body, str):
            self.content = body.encode("utf-8")
            ctype = "text/html; charset=utf-8"
        else:
            self.content = body
            ctype = "application/octet-stream"
        self.status_code = status
        self.headers = {"content-type": content_type or ctype}
        self.encoding = encoding or ("utf-8" if not isinstance(body, bytes) else None)
        self.url = url

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding or "utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)  # type: ignore[arg-type]


Route = FakeResponse | Callable[[str, dict, Any], FakeResponse]


class FakeSession:
    """Routes requests by regex on 'METHOD url?query'; unmatched requests get HTTP 404."""

    def __init__(self, routes: list[tuple[str, Route]]):
        self.routes = routes
        self.calls: list[str] = []

    def _dispatch(self, method: str, url: str, params: dict | None, data: Any) -> FakeResponse:
        full = f"{method} {url}" + (("?" + urlencode(params, doseq=True)) if params else "")
        self.calls.append(full)
        for pattern, resp in self.routes:
            if re.search(pattern, full):
                return resp(full, params or {}, data) if callable(resp) else resp
        return FakeResponse("not found", status=404)

    def get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        **kw: Any,
    ) -> FakeResponse:
        return self._dispatch("GET", url, params, None)

    def post(self, url: str, data: Any = None, headers: dict | None = None, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, None, data)


class FakeTicker:
    def __init__(
        self,
        symbol: str,
        histories: dict[str, pd.DataFrame],
        infos: dict[str, dict],
        fail: set[str],
    ):
        self.symbol, self._h, self._i, self._fail = symbol, histories, infos, fail

    def history(self, **kw: Any) -> pd.DataFrame:
        if self.symbol in self._fail:
            raise RuntimeError(f"yahoo down for {self.symbol}")
        return self._h.get(self.symbol, pd.DataFrame())

    @property
    def info(self) -> dict:
        if self.symbol in self._fail:
            raise RuntimeError(f"yahoo down for {self.symbol}")
        return self._i.get(self.symbol, {})


class FakeYF:
    def __init__(
        self,
        histories: dict | None = None,
        infos: dict | None = None,
        fail: set | None = None,
    ):
        self.histories, self.infos, self.fail = (
            histories or {},
            infos or {},
            fail or set(),
        )

    def Ticker(self, symbol: str) -> FakeTicker:  # noqa: N802 - mimics yfinance
        return FakeTicker(symbol, self.histories, self.infos, self.fail)


def base_routes() -> list[tuple[str, Route]]:
    """Routes for every KR endpoint, answered with the saved fixtures."""

    def cf4002(full: str, params: dict, data: Any) -> FakeResponse:
        return FakeResponse(
            fx_json(
                "wisereport_cf4002_quarter_005930.json"
                if params.get("frq") == "1"
                else "wisereport_cf4002_annual_005930.json"
            )
        )

    def dart_main(full: str, params: dict, data: Any) -> FakeResponse:
        rcp = params.get("rcpNo", "")
        return FakeResponse(
            fx_text("dart_main_20260730800077.html" if rcp[8:9] in ("8", "9") else "dart_main_20260922000524.html")
        )

    def dart_viewer(full: str, params: dict, data: Any) -> FakeResponse:
        if params.get("dtd") == "HTML":
            return FakeResponse(
                (FIX / "dart_viewer_20260730800077.html").read_bytes(),
                content_type="text/html; charset=MS949",
                encoding="MS949",
            )
        return FakeResponse(
            f"<html><body><p>섹션 {params.get('eleId')} 본문: 주요 내용 서술</p>"
            "<table><tr><td>항목</td><td>1,000</td></tr></table></body></html>"
        )

    def dart_search(full: str, params: dict, data: Any) -> FakeResponse:
        code = data.get("textCrpNm") if isinstance(data, dict) else ""
        return FakeResponse(fx_text(f"dart_search_{code}.html"))

    return [
        (
            r"fchart\.stock\.naver\.com.*symbol=KOSPI",
            FakeResponse(fx_text("fchart_KOSPI.xml")),
        ),
        (
            r"fchart\.stock\.naver\.com.*symbol=005930",
            FakeResponse(fx_text("fchart_005930.xml")),
        ),
        (
            r"finance\.daum\.net/api/charts/A005930/days",
            FakeResponse(fx_json("daum_chart_005930.json")),
        ),
        (
            r"finance\.daum\.net/api/quotes/A005930",
            FakeResponse(fx_json("daum_quote_005930.json")),
        ),
        (
            r"m\.stock\.naver\.com/api/stock/005930/basic",
            FakeResponse(fx_json("naver_basic_005930.json")),
        ),
        (
            r"m\.stock\.naver\.com/api/stock/247540/basic",
            FakeResponse(fx_json("naver_basic_247540.json")),
        ),
        (
            r"m\.stock\.naver\.com/api/stock/005930/integration",
            FakeResponse(fx_json("naver_integration_005930.json")),
        ),
        (
            r"m\.stock\.naver\.com/api/stock/247540/integration",
            FakeResponse(fx_json("naver_integration_247540.json")),
        ),
        (
            r"m\.stock\.naver\.com/api/stocks/industry/278",
            FakeResponse(fx_json("naver_industry_278.json")),
        ),
        (
            r"m\.stock\.naver\.com/api/stocks/industry/283",
            FakeResponse(fx_json("naver_industry_283.json")),
        ),
        (
            r"m\.stock\.naver\.com/api/stock/005930/finance/quarter",
            FakeResponse(fx_json("naver_finance_quarter_005930.json")),
        ),
        (
            r"m\.stock\.naver\.com/api/stock/005930/finance/annual",
            FakeResponse(fx_json("naver_finance_annual_005930.json")),
        ),
        (
            r"wisereport\.co\.kr/v2/company/c1010001\.aspx",
            FakeResponse(fx_text("wisereport_c1010001_005930.html")),
        ),
        (
            r"wisereport\.co\.kr/v2/company/c1040001\.aspx",
            FakeResponse(fx_text("wisereport_c1040001_excerpt.html")),
        ),
        (r"wisereport\.co\.kr/v2/company/cF4002\.aspx", cf4002),
        (
            r"m\.stock\.naver\.com/api/news/stock/\w+\?.*page=1\b",
            FakeResponse(fx_json("naver_news_247540.json")),
        ),
        (r"m\.stock\.naver\.com/api/news/stock/", FakeResponse([])),
        (
            r"news\.google\.com/rss/search",
            FakeResponse(fx_text("google_news_247540.xml")),
        ),
        (
            r"n\.news\.naver\.com/mnews/article/",
            FakeResponse(fx_text("naver_article.html")),
        ),
        (r"POST https://dart\.fss\.or\.kr/dsab007/detailSearch\.ax", dart_search),
        (r"dart\.fss\.or\.kr/dsaf001/main\.do", dart_main),
        (r"dart\.fss\.or\.kr/report/viewer\.do", dart_viewer),
        (
            r"m\.stock\.naver\.com/api/research/stock/",
            FakeResponse(fx_json("naver_research_list_005930.json")),
        ),
        (
            r"m\.stock\.naver\.com/api/research/company/",
            FakeResponse(fx_json("naver_research_detail.json")),
        ),
        (
            r"m\.stock\.naver\.com/api/stocks/marketValue/KOSPI\?.*page=1\b",
            FakeResponse(fx_json("naver_marketvalue_KOSPI.json")),
        ),
        (
            r"m\.stock\.naver\.com/api/stocks/marketValue/KOSDAQ\?.*page=1\b",
            FakeResponse(fx_json("naver_marketvalue_KOSDAQ.json")),
        ),
        (  # the shared USD/KRW fallback (quantcio.data.us) reads the daily /prices list (real capture)
            r"api\.stock\.naver\.com/marketindex/exchange/FX_USDKRW/prices",
            FakeResponse(fx_json("naver_fx_usdkrw_prices.json")),
        ),
    ]


def yf_raw_history(closes: dict[str, float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d).tz_localize("Asia/Seoul") for d in closes])
    c = np.array(list(closes.values()), dtype=float)
    return pd.DataFrame(
        {
            "Open": c,
            "High": c * 1.01,
            "Low": c * 0.99,
            "Close": c,
            "Adj Close": c * 0.98,
            "Volume": np.full(len(c), 1e6),
        },
        index=idx,
    )


@pytest.fixture
def make_provider(settings, tmp_path, monkeypatch):
    """Factory: provider wired to a FakeSession (fixture routes + overrides) and a FakeYF."""
    monkeypatch.setattr(kr, "_polite_wait", lambda url: None)
    monkeypatch.setattr(us, "FX_YF_TRIES", 1)  # a faked KRW=X failure must not sit in backoff sleeps
    settings.data.dart_api_key = ""

    def _make(
        routes: list | None = None,
        now: datetime = HOLIDAY_2PM,
        yf: FakeYF | None = None,
        cache: bool = True,
    ) -> tuple[KRProvider, FakeSession]:
        session = FakeSession([*(routes or []), *base_routes()])
        fake_yf = yf or FakeYF()
        monkeypatch.setattr(kr, "_yf", lambda: fake_yf)
        prov = KRProvider(
            settings,
            DiskCache(tmp_path / f"cache{len(list(tmp_path.iterdir()))}", enabled=cache),
            session_factory=lambda: session,
            now_fn=lambda: now,
        )
        return prov, session

    return _make


def samsung() -> Instrument:
    return Instrument(
        ticker="005930",
        market=Market.KR,
        name="삼성전자",
        exchange="KOSPI",
        currency="KRW",
        sector="반도체와반도체장비",
        yf_symbol="005930.KS",
        benchmark="KOSPI",
        benchmark_name="KOSPI",
    )


def ecopro() -> Instrument:
    return Instrument(
        ticker="247540",
        market=Market.KR,
        name="에코프로비엠",
        exchange="KOSDAQ",
        currency="KRW",
        sector="전기제품",
        yf_symbol="247540.KQ",
        benchmark="KOSDAQ",
        benchmark_name="KOSDAQ",
    )


# ============================================================================ pure helpers


def test_parse_number_and_korean_money():
    assert kr.parse_number("12.85배") == 12.85
    assert kr.parse_number("22,292원") == 22292.0
    assert kr.parse_number("-341") == -341.0
    assert kr.parse_number("N/A") is None and kr.parse_number(None) is None
    assert kr.parse_krw_korean("1,674조 9,588억") == pytest.approx(1.6749588e15)
    assert kr.parse_krw_korean("10조 3,211억") == pytest.approx(1.03211e13)
    assert kr.parse_krw_korean("460억") == pytest.approx(4.6e10)
    assert kr.parse_krw_korean("") is None


def test_check_code_rejects_us_and_accepts_alphanumeric():
    assert kr.check_code("005930.KS") == "005930"
    assert kr.check_code("0126z0") == "0126Z0"
    with pytest.raises(ValueError):
        kr.check_code("AAPL")


def test_truncate_head_tail_keeps_both_ends():
    text = "HEAD" + "x" * 5000 + "TAIL"
    out = kr.truncate_head_tail(text, 200)
    assert len(out) <= 200 and out.startswith("HEAD") and out.endswith("TAIL") and "중략" in out
    assert kr.truncate_head_tail("short", 200) == "short"


def test_opinion_and_consensus_normalisation():
    assert kr.normalize_opinion("StrongBuy") == "Strong Buy"
    assert kr.normalize_opinion("Trading  Buy") == "Trading Buy"
    assert kr.normalize_opinion("OutPerform") == "Buy"
    assert kr.normalize_opinion("없음") is None
    assert kr.normalize_opinion("투자의견없음") is None  # seen live (Naver research)
    assert kr.normalize_opinion("매수(유지)") == "Buy" and kr.normalize_opinion("N/R") is None
    assert kr.normalize_opinion("시장수익률") == "Hold"
    assert kr.recomm_mean_label(4.0) == "Buy (4.00/5)"
    assert kr.recomm_mean_label(4.6).startswith("Strong Buy")
    assert kr.recomm_mean_label(3.0).startswith("Hold")
    assert kr.recomm_mean_label(None) is None and kr.recomm_mean_label(0.0) is None


def test_html_to_text_flattens_tables():
    txt = kr.html_to_text(
        "<html><body><p>제목</p><table><tr><th>매출액</th><td>1,714,995</td></tr></table>"
        "<script>x=1</script></body></html>"
    )
    assert "매출액 | 1,714,995" in txt and "x=1" not in txt


def test_session_phase_and_partial_bar():
    df = pd.DataFrame(
        {c: [1.0, 2.0] for c in PRICE_COLUMNS},
        index=pd.DatetimeIndex(["2026-09-22", "2026-09-23"]),
    )
    at = lambda h, m: datetime(2026, 9, 23, h, m, tzinfo=KST)  # noqa: E731
    assert kr.session_phase(at(8, 10)) == "intraday"  # NXT pre-market counts as partial
    assert kr.session_phase(at(15, 50)) == "intraday"
    assert kr.session_phase(at(17, 0)) == "post_krx"
    assert kr.session_phase(at(20, 30)) == "post_nxt"
    assert len(kr.drop_partial_bar(df, at(11, 0))) == 1
    assert len(kr.drop_partial_bar(df, at(17, 0), confirmed_today=True)) == 2
    assert len(kr.drop_partial_bar(df, at(17, 0), confirmed_today=False)) == 1
    assert len(kr.drop_partial_bar(df, at(21, 0), confirmed_today=False)) == 2
    # a bar dated before today is never touched
    assert len(kr.drop_partial_bar(df, datetime(2026, 9, 24, 10, 0, tzinfo=KST))) == 2


# ================================================================================ prices


def test_parse_fchart_fixture_and_halt_rows():
    px = kr.parse_fchart(fx_text("fchart_005930.xml"), "005930")
    assert list(px.columns) == PRICE_COLUMNS and len(px) == 40
    assert px.index.is_monotonic_increasing and px.index.tz is None and px.index.is_unique
    assert (px.dtypes == "float64").all()
    assert px.loc["2026-09-23", "close"] == 286500.0  # Naver = KRX+NXT integrated close
    halted = kr.parse_fchart(
        '<item data="20180430|0|0|0|53000|0" />\n<item data="20180504|53000|53900|51800|51900|39565391" />',
        "X",
    )
    assert halted.loc["2018-04-30", ["open", "high", "low", "close"]].tolist() == [53000.0] * 4
    with pytest.raises(ValueError):
        kr.parse_fchart("<chartdata></chartdata>", "AAPL")


def test_daum_official_closes_are_next_day_base_price():
    off = kr.parse_daum_official_closes(fx_json("daum_chart_005930.json"))
    assert off.loc["2026-09-21"] == 274000.0 and off.loc["2026-09-22"] == 276500.0
    assert pd.Timestamp("2026-09-23") not in off.index  # last candle has no successor
    assert off.loc["2026-09-11"] == 259500.0  # FALL day (signed changePrice)
    # robust to an unsigned changePrice on a FALL day
    payload = {
        "data": [
            {
                "date": "2026-01-02",
                "tradePrice": 100.0,
                "changePrice": 0.0,
                "change": "EVEN",
            },
            {
                "date": "2026-01-05",
                "tradePrice": 95.0,
                "changePrice": 5.0,
                "change": "FALL",
            },
        ]
    }
    assert kr.parse_daum_official_closes(payload).iloc[0] == 100.0


def test_official_close_fix_on_real_nxt_data():
    px = kr.parse_fchart(fx_text("fchart_005930.xml"), "005930")
    off = kr.parse_daum_official_closes(fx_json("daum_chart_005930.json"))
    q = fx_json("daum_quote_005930.json")
    fixed, meta = kr.apply_official_close_fix(px, off, (pd.Timestamp(q["date"]), q["regularTradePrice"], "q"))
    # verified 2026-09-24: official KRX closes (WiseReport / fdr listing / Daum regularTradePrice)
    assert fixed.loc["2026-09-21", "close"] == 274000.0
    assert fixed.loc["2026-09-22", "close"] == 276500.0
    assert fixed.loc["2026-09-23", "close"] == 285500.0
    assert (fixed.loc[:"2026-08-31"] == px.loc[:"2026-08-31"]).all().all()  # pre-NXT bars untouched
    assert (fixed["high"] >= fixed["close"]).all() and (fixed["low"] <= fixed["close"]).all()
    assert meta["unverified"] == [] and meta["rejected"] == [] and meta["fixed"]["q"] == 1


def test_official_close_fix_reports_unverified_and_rejects_out_of_range():
    px = kr.parse_fchart(fx_text("fchart_005930.xml"), "005930")
    fixed, meta = kr.apply_official_close_fix(px, None, None)
    assert "2026-09-23" in meta["unverified"] and len(meta["unverified"]) == int((px.index >= kr.NXT_SINCE).sum())
    bogus = pd.Series({pd.Timestamp("2026-09-22"): 150000.0})  # far outside the day's [low, high]
    fixed, meta = kr.apply_official_close_fix(px, bogus, None)
    assert fixed.loc["2026-09-22", "close"] == 277500.0 and meta["rejected"] == ["2026-09-22"]


def test_prices_end_to_end_official_fix(make_provider):
    prov, sess = make_provider()
    df = prov.prices(samsung(), date(2026, 7, 1))
    assert list(df.columns) == PRICE_COLUMNS and df.index[-1] == pd.Timestamp("2026-09-23")
    assert df["close"].iloc[-1] == 285500.0 and df.loc["2026-09-22", "close"] == 276500.0
    assert df.attrs["warnings"] == [] and "공식종가" in df.attrs["source"]
    assert df.attrs["close_source"]["krx_official(daum_quote)"] == 1
    assert prov.data_warnings("005930") == []
    assert clean_prices(df).attrs["warnings"] == []  # attrs survive the shared cleaner
    n_calls = len(sess.calls)
    prov.prices(samsung(), date(2026, 7, 1))  # cached
    assert len(sess.calls) == n_calls


def test_prices_intraday_drops_partial_bar(make_provider):
    prov, sess = make_provider(now=datetime(2026, 9, 23, 11, 0, tzinfo=KST))
    df = prov.prices(samsung(), date(2026, 7, 1))
    assert df.index[-1] == pd.Timestamp("2026-09-22")
    assert df["close"].iloc[-1] == 276500.0  # official, from the Daum base price
    assert df.attrs["warnings"] == []
    assert not any("/quotes/" in c for c in sess.calls)  # last bar covered by the chart: no quote call


def test_prices_post_krx_keeps_bar_only_when_official_confirmed(make_provider):
    prov, _ = make_provider(now=datetime(2026, 9, 23, 17, 0, tzinfo=KST))
    assert prov.prices(samsung(), date(2026, 7, 1))["close"].iloc[-1] == 285500.0
    prov2, _ = make_provider(
        routes=[(r"finance\.daum\.net/api/quotes", FakeResponse("err", status=500))],
        now=datetime(2026, 9, 23, 17, 0, tzinfo=KST),
    )
    df = prov2.prices(samsung(), date(2026, 7, 1))
    assert df.index[-1] == pd.Timestamp("2026-09-22")
    # degraded (Daum quote + Yahoo both failed): the dropped session is reported, not only logged
    assert len(df.attrs["warnings"]) == 1 and "당일(2026-09-23) KRX 공식종가 미확인" in df.attrs["warnings"][0]
    assert any("당일 봉 제외" in w for w in prov2.data_warnings("005930"))


@pytest.mark.parametrize(
    "hh, mm, expect_last, expect_close",
    [
        (
            0,
            30,
            "2026-09-23",
            285500.0,
        ),  # after midnight: yesterday's bar is final, official via quote
        (8, 30, "2026-09-22", 276500.0),  # NXT pre-market: today's bar is partial
        (15, 59, "2026-09-22", 276500.0),  # KRX closing auction / Yahoo delay buffer
        (
            16,
            0,
            "2026-09-23",
            285500.0,
        ),  # KRX final, NXT still trading: official close from the quote
        (19, 59, "2026-09-23", 285500.0),
        (20, 0, "2026-09-23", 285500.0),
    ],
)
def test_prices_kst_clock_simulation(make_provider, hh, mm, expect_last, expect_close):
    """Same Naver/Daum data seen at different KST clock times of the trading day 2026-09-23 (09-24 for 00:30)."""
    day = 24 if hh == 0 else 23
    prov, _ = make_provider(now=datetime(2026, 9, day, hh, mm, tzinfo=KST))
    df = prov.prices(samsung(), date(2026, 7, 1))
    assert df.index[-1] == pd.Timestamp(expect_last) and df["close"].iloc[-1] == expect_close
    assert df.loc["2026-09-21", "close"] == 274000.0 and df.attrs["warnings"] == []
    # every NXT-era bar carries the official close (none of Naver's integrated closes survive)
    naver_integrated = {
        "2026-09-21": 275000.0,
        "2026-09-22": 277500.0,
        "2026-09-23": 286500.0,
    }
    assert all(df.loc[d, "close"] != v for d, v in naver_integrated.items() if pd.Timestamp(d) in df.index)


def test_price_cache_is_keyed_by_session_phase(settings, tmp_path, monkeypatch):
    """One provider + one disk cache while the KST clock moves: the intraday frame (09-22 last) must not be
    served after the KRX close; within a phase the cache is hit; an expired TTL refetches."""
    monkeypatch.setattr(kr, "_polite_wait", lambda url: None)
    monkeypatch.setattr(kr, "_yf", lambda: FakeYF())
    clock = {"now": datetime(2026, 9, 23, 11, 0, tzinfo=KST)}
    session = FakeSession(base_routes())
    cache = DiskCache(tmp_path / "c")
    prov = KRProvider(settings, cache, session_factory=lambda: session, now_fn=lambda: clock["now"])
    assert prov.prices(samsung(), date(2026, 7, 1)).index[-1] == pd.Timestamp("2026-09-22")
    n = len(session.calls)
    clock["now"] = datetime(2026, 9, 23, 15, 0, tzinfo=KST)
    prov.prices(samsung(), date(2026, 7, 1))
    assert len(session.calls) == n  # same phase -> cache hit
    clock["now"] = datetime(2026, 9, 23, 17, 0, tzinfo=KST)
    df = prov.prices(samsung(), date(2026, 7, 1))
    assert df.index[-1] == pd.Timestamp("2026-09-23") and df["close"].iloc[-1] == 285500.0
    assert len(session.calls) > n
    settings.data.price_ttl_hours = 0.0  # every entry is now expired
    n = len(session.calls)
    prov.prices(samsung(), date(2026, 7, 1))
    assert len(session.calls) > n


def test_prices_after_nxt_close_without_official_close_warns(make_provider):
    prov, _ = make_provider(
        routes=[(r"finance\.daum\.net/api/quotes", FakeResponse("err", status=500))],
        now=datetime(2026, 9, 23, 21, 0, tzinfo=KST),
    )
    df = prov.prices(samsung(), date(2026, 7, 1))
    assert df.index[-1] == pd.Timestamp("2026-09-23") and df["close"].iloc[-1] == 286500.0
    assert any("공식종가 미확인 1일" in w for w in df.attrs["warnings"])
    assert any("미확인" in w for w in prov.data_warnings("005930"))


def test_prices_daum_down_uses_guarded_yahoo_and_reports(make_provider):
    closes = {"2026-09-21": 274000.0, "2026-09-22": 276500.0, "2026-09-23": 285500.0}
    yf = FakeYF(histories={"005930.KS": yf_raw_history(closes)})
    prov, _ = make_provider(
        routes=[(r"finance\.daum\.net/api/", FakeResponse("forbidden", status=403))],
        yf=yf,
    )
    df = prov.prices(samsung(), date(2026, 7, 1))
    assert df.loc["2026-09-21", "close"] == 274000.0 and df["close"].iloc[-1] == 285500.0
    w = " ".join(df.attrs["warnings"])
    assert "Daum 공식종가 조회 실패" in w and "공식종가 미확인" in w  # 09-01..09-18 stay unverified


def test_prices_fallback_to_yahoo_never_splices(make_provider):
    closes = {"2026-09-21": 274000.0, "2026-09-22": 276500.0, "2026-09-23": 285500.0}
    yf = FakeYF(histories={"005930.KS": yf_raw_history(closes)})
    prov, _ = make_provider(routes=[(r"fchart\.stock\.naver\.com", FakeResponse("down", status=503))], yf=yf)
    df = prov.prices(samsung(), date(2026, 9, 1))
    assert len(df) == 3 and "yfinance" in df.attrs["source"]
    assert df["close"].tolist() == pytest.approx([c * 0.98 for c in closes.values()])  # Adj Close ratio
    assert df["open"].iloc[0] == pytest.approx(274000.0 * 0.98)
    assert any("yfinance" in w for w in prov.data_warnings("005930"))


def test_yahoo_adjusted_ohlcv_restores_ohlc_invariant():
    """Yahoo KR has stale closes outside the day's [low, high] (e.g. 005930 2024-10-14) and can report O/H/L=0."""
    h = pd.DataFrame(
        {
            "Open": [100.0, 0.0],
            "High": [101.0, 0.0],
            "Low": [99.0, 0.0],
            "Close": [103.0, 98.0],
            "Adj Close": [103.0, 98.0],
            "Volume": [1e5, 0.0],
        },
        index=pd.DatetimeIndex(["2024-10-14", "2024-10-15"]),
    )
    out = kr.yahoo_adjusted_ohlcv(h)
    assert out.loc["2024-10-14", "high"] == 103.0 and out.loc["2024-10-14", "low"] == 99.0
    assert out.loc["2024-10-15", ["open", "high", "low", "close"]].tolist() == [98.0] * 4
    assert (
        (out["high"] >= out[["open", "close"]].max(axis=1)) & (out["low"] <= out[["open", "close"]].min(axis=1))
    ).all()


def test_daum_down_yahoo_window_covers_whole_nxt_period(make_provider, monkeypatch):
    """The Yahoo fallback must be asked from the first NXT-era bar. It used to go back 'n_window + 20'
    CALENDAR days for n_window TRADING days, which falls short once the window is long (here: all 40 bars)."""
    idx = pd.bdate_range(end="2026-09-23", periods=300)
    xml = "\n".join(f'<item data="{d:%Y%m%d}|100|110|90|105|1000" />' for d in idx)
    monkeypatch.setattr(kr, "NXT_SINCE", idx[0])  # a 300-session NXT window
    first_bar = idx[0]
    starts: list[str] = []

    class RecordingYF(FakeYF):
        def Ticker(self, symbol: str) -> FakeTicker:  # noqa: N802
            t = super().Ticker(symbol)
            orig = t.history

            def history(**kw: Any) -> pd.DataFrame:
                starts.append(kw.get("start"))
                return orig(**kw)

            t.history = history  # type: ignore[method-assign]
            return t

    prov, _ = make_provider(
        routes=[
            (r"finance\.daum\.net/api/", FakeResponse("forbidden", status=403)),
            (
                r"fchart\.stock\.naver\.com.*symbol=005930",
                FakeResponse(f"<chartdata>{xml}</chartdata>"),
            ),
        ],
        yf=RecordingYF(),
    )
    df = prov.prices(samsung(), date(2025, 1, 1))
    assert len(df) == 300 and starts
    assert min(starts) <= first_bar.date().isoformat()  # old code: 2026-09-23 - 320 days = 2025-11-07
    assert any("공식종가 미확인 300일" in w for w in df.attrs["warnings"])  # no Yahoo data -> all reported


def test_prices_raise_when_every_source_fails(make_provider):
    prov, _ = make_provider(
        routes=[(r"fchart\.stock\.naver\.com", FakeResponse("down", status=503))],
        yf=FakeYF(fail={"005930.KS"}),
    )
    with pytest.raises(RuntimeError, match="시세 수집 실패"):
        prov.prices(samsung(), date(2026, 7, 1))


def test_benchmark_naver_and_yahoo_fallback(make_provider):
    prov, _ = make_provider()
    b = prov.benchmark(samsung(), date(2026, 9, 1))
    assert b["close"].iloc[-1] == pytest.approx(7080.92) and b.attrs["volume_unit"] == "1,000주"
    assert b.attrs["warnings"] == []
    hist = yf_raw_history({"2026-09-22": 7017.91, "2026-09-23": 7080.92})
    prov2, _ = make_provider(
        routes=[(r"fchart\.stock\.naver\.com", FakeResponse("down", status=503))],
        yf=FakeYF(histories={"^KS11": hist}),
    )
    b2 = prov2.benchmark(samsung(), date(2026, 9, 1))
    assert len(b2) == 2 and "yfinance" in b2.attrs["source"] and b2.attrs["warnings"]
    prov3, _ = make_provider(
        routes=[(r"fchart\.stock\.naver\.com", FakeResponse("down", status=503))],
        yf=FakeYF(fail={"^KS11"}),
    )
    with pytest.raises(RuntimeError):
        prov3.benchmark(samsung(), date(2026, 9, 1))
    # Yahoo rows without a usable close (^KS11 has NaN days) -> a clear error, never an empty frame
    nan_hist = yf_raw_history({"2026-09-22": 7017.91}).assign(Close=np.nan)
    prov4, _ = make_provider(
        routes=[(r"fchart\.stock\.naver\.com", FakeResponse("down", status=503))],
        yf=FakeYF(histories={"^KS11": nan_hist}),
    )
    with pytest.raises(RuntimeError, match="유효 종가 없음"):
        prov4.benchmark(samsung(), date(2026, 9, 1))


# =============================================================================== resolve


def test_resolve_kospi_and_kosdaq(make_provider):
    prov, _ = make_provider()
    s = prov.resolve("005930")
    assert (
        s.name,
        s.exchange,
        s.currency,
        s.yf_symbol,
        s.benchmark,
        s.benchmark_name,
    ) == ("삼성전자", "KOSPI", "KRW", "005930.KS", "KOSPI", "KOSPI")
    assert s.sector == "반도체와반도체장비" and s.market is Market.KR
    e = prov.resolve("247540.KQ")
    assert (e.name, e.exchange, e.yf_symbol, e.benchmark, e.sector) == (
        "에코프로비엠",
        "KOSDAQ",
        "247540.KQ",
        "KOSDAQ",
        "전기제품",
    )
    with pytest.raises(ValueError):
        prov.resolve("AAPL")


def test_resolve_falls_back_to_daum_and_reports(make_provider):
    prov, _ = make_provider(
        routes=[
            (
                r"m\.stock\.naver\.com/api/stock/005930/basic",
                FakeResponse({"code": "StockConflict"}, status=409),
            )
        ]
    )
    s = prov.resolve("005930")
    assert s.name == "삼성전자" and s.exchange == "KOSPI" and s.yf_symbol == "005930.KS"
    assert any("대체 소스" in w for w in prov.data_warnings("005930"))


# ========================================================================== fundamentals


def test_fundamentals_from_fixtures(make_provider):
    yf = FakeYF(
        infos={
            "005930.KS": {
                "targetHighPrice": 725000.0,
                "targetLowPrice": 290000.0,
                "numberOfAnalystOpinions": 36,
            }
        }
    )
    prov, _ = make_provider(yf=yf)
    f = prov.fundamentals(samsung())
    assert isinstance(f, Fundamentals)
    assert f.per == 12.81 and f.pbr == 3.32  # WiseReport: TTM on the KRX close
    assert f.eps == 22292.0 and f.bps == 86052.0
    assert f.forward_per == 5.98 and f.forward_eps == 47922.0
    assert f.sector_per == 9.62
    assert f.market_cap == pytest.approx(1.6749588e15)
    assert f.dividend_yield == pytest.approx(0.0058)
    assert f.roe == pytest.approx(22292 / 86052)
    # Q2'26 vs Q2'25 revenue 1,714,995 / 745,663 (matches the DART 잠정실적 +130.00%)
    assert f.revenue_growth_yoy == pytest.approx(1714995 / 745663 - 1)
    assert 0 < f.operating_margin < 1 and f.earnings_growth_yoy > 0
    assert f.analyst_target_price == 493864.0 and f.analyst_count == 22
    assert (f.analyst_target_high, f.analyst_target_low) == (725000.0, 290000.0)
    assert f.analyst_rating == "Buy (4.00/5)" and "PER=TTM" in f.source
    assert prov.data_warnings("005930") == []


def test_fundamentals_partial_and_total_failure(make_provider):
    prov, _ = make_provider(routes=[(r"wisereport", FakeResponse("down", status=503))])
    f = prov.fundamentals(samsung())
    assert f.per == 12.85 and f.sector_per is None  # Naver fallback, flagged
    assert any("WiseReport" in w for w in prov.data_warnings("005930"))
    prov2, _ = make_provider(
        routes=[
            (r"wisereport", FakeResponse("down", status=503)),
            (
                r"m\.stock\.naver\.com/api/stock/005930/integration",
                FakeResponse("x", status=500),
            ),
        ]
    )
    with pytest.raises(RuntimeError):
        prov2.fundamentals(samsung())


def samsung_pref() -> Instrument:
    return Instrument(
        ticker="005935",
        market=Market.KR,
        name="삼성전자우",
        exchange="KOSPI",
        currency="KRW",
        sector="반도체와반도체장비",
        yf_symbol="005935.KS",
        benchmark="KOSPI",
        benchmark_name="KOSPI",
    )


def pref_routes() -> list[tuple[str, Any]]:
    """005935 (삼성전자우): real Naver integration; company-level feeds answered like the issuer's (as live)."""
    return [
        (
            r"m\.stock\.naver\.com/api/stock/005935/integration",
            FakeResponse(fx_json("naver_integration_005935.json")),
        ),
        (
            r"m\.stock\.naver\.com/api/stock/005935/finance/quarter",
            FakeResponse(fx_json("naver_finance_quarter_005930.json")),
        ),
        (
            r"m\.stock\.naver\.com/api/stock/005935/finance/annual",
            FakeResponse(fx_json("naver_finance_annual_005930.json")),
        ),
        (
            r"fchart\.stock\.naver\.com.*symbol=005935",
            FakeResponse(fx_text("fchart_005930.xml")),
        ),
        (
            r"finance\.daum\.net/api/charts/A005935/days",
            FakeResponse(fx_json("daum_chart_005930.json")),
        ),
        (
            r"finance\.daum\.net/api/quotes/A005935",
            FakeResponse(fx_json("daum_quote_005930.json")),
        ),
    ]


def test_preferred_share_fundamentals_never_use_common_share_target(make_provider):
    """Regression (live 2026-09-24): for 005935 WiseReport serves the COMMON page (close 285,500, PER 12.81) and
    Naver/WiseReport consensus = the common's 493,864 target -> +124% fake upside vs the 220,000 preferred."""
    yf = FakeYF(infos={"005935.KS": {"targetHighPrice": 725000.0, "targetLowPrice": 290000.0}})
    prov, sess = make_provider(routes=pref_routes(), yf=yf)
    f = prov.fundamentals(samsung_pref())
    assert (f.per, f.pbr, f.forward_per) == (
        9.96,
        2.58,
        4.63,
    )  # Naver, on the preferred price
    assert f.market_cap == pytest.approx(178.1264e12)  # preferred class, not the company's
    assert f.dividend_yield == pytest.approx(0.0075)
    assert f.analyst_target_price is None and f.analyst_rating is None and f.analyst_count is None
    assert f.analyst_target_high is None and f.analyst_target_low is None
    assert f.eps == 22292.0 and f.sector_per == 9.62 and "우선주" in f.source
    assert any("우선주" in w and "목표가" in w for w in prov.data_warnings("005935"))
    assert any("c1010001.aspx?cmp_cd=005930" in c for c in sess.calls)  # issuer page, explicitly
    # the common share is unaffected
    assert prov.fundamentals(samsung()).analyst_target_price == 493864.0


def test_preferred_share_valuation_history_uses_issuer_knots(make_provider):
    prov, sess = make_provider(routes=pref_routes())
    v = prov.valuation_history(samsung_pref(), date(2026, 7, 1))
    assert v is not None and not v["per"].dropna().empty
    assert any("cF4002.aspx?cmp_cd=005930" in c for c in sess.calls)
    assert not any("cmp_cd=005935" in c for c in sess.calls)


def test_compose_fundamentals_growth_guards():
    fin = pd.DataFrame(
        {
            "202506": [100.0, -5.0],
            "202509": [110.0, 3.0],
            "202512": [120.0, 4.0],
            "202603": [130.0, 5.0],
            "202606": [150.0, 6.0],
        },
        index=["매출액", "지배주주순이익"],
    )
    assert kr.quarterly_yoy(fin, "매출액") == pytest.approx(0.5)
    assert kr.quarterly_yoy(fin, "지배주주순이익") is None  # negative base -> undefined


# ===================================================================== valuation history


def test_valuation_history_point_in_time(make_provider):
    prov, _ = make_provider()
    v = prov.valuation_history(samsung(), date(2026, 7, 1))
    assert list(v.columns) == ["per", "pbr"] and v.index.is_monotonic_increasing
    # knot 2026-06-30 + 45d = 2026-08-14 -> TTM EPS = Q3'25..Q2'26
    ttm = 1782.53438 + 2864.187 + 6992.85913 + 10717.84914
    assert v.loc["2026-09-23", "per"] == pytest.approx(285500 / ttm, rel=1e-6)  # ~12.77 (WiseReport 12.81)
    assert v.loc["2026-09-23", "pbr"] == pytest.approx(285500 / 86051.82511, rel=1e-6)
    ttm_prev = 732.52936 + 1782.53438 + 2864.187 + 6992.85913  # knot 03-31 + 45d
    close_0813 = prov.prices(samsung(), date(2026, 7, 1)).loc["2026-08-13", "close"]
    assert v.loc["2026-08-13", "per"] == pytest.approx(close_0813 / ttm_prev, rel=1e-6)


def test_build_valuation_history_negative_eps_is_nan():
    ann = pd.DataFrame(
        {"EPS": [-100.0], "BPS": [1000.0], "is_est": [False]},
        index=pd.DatetimeIndex(["2024-12-31"]),
    )
    qtr = pd.DataFrame({"EPS": [], "BPS": [], "is_est": []}, index=pd.DatetimeIndex([]))
    close = pd.Series(500.0, index=pd.bdate_range("2025-03-20", periods=20))
    v = kr.build_valuation_history(ann, qtr, close)
    assert v.index[0] == pd.Timestamp("2025-03-31") and v["per"].isna().all() and (v["pbr"] == 0.5).all()


# ================================================================================== news


def test_news_relevance_dedupe_and_utc(make_provider):
    prov, sess = make_provider()
    since = HOLIDAY_2PM.astimezone(UTC) - timedelta(days=21)
    items = prov.news(ecopro(), since, 30)
    assert items and all(isinstance(n, NewsItem) for n in items)
    assert all(n.published_at.tzinfo is not None and n.published_at.utcoffset() == timedelta(0) for n in items)
    assert [n.published_at for n in items] == sorted((n.published_at for n in items), reverse=True)
    assert all(n.published_at >= since for n in items)
    assert all("에코프로비엠" in (n.title + n.summary) or "에코프로BM" in (n.title + n.summary) for n in items)
    assert len({kr._norm_title(n.title) for n in items}) == len(items)
    assert all(" - " + n.source not in n.title for n in items if n.source)  # ' - 출처' stripped
    assert len({n.id for n in items}) == len(items) and all(len(n.summary) <= 600 for n in items)
    assert len(prov.news(ecopro(), since, 3)) == 3
    assert prov.data_warnings("247540") == []


def test_news_generic_fallback_is_flagged(make_provider):
    prov, _ = make_provider(
        routes=[
            (
                r"news\.google\.com",
                FakeResponse('<rss version="2.0"><channel></channel></rss>'),
            )
        ]
    )
    inst = ecopro().model_copy(update={"name": "없는회사명"})
    items = prov.news(inst, HOLIDAY_2PM - timedelta(days=21), 5)
    assert len(items) == 5
    assert any("일반 시황 기사" in w for w in prov.data_warnings("247540"))


def test_news_raises_when_all_sources_fail(make_provider):
    prov, _ = make_provider(
        routes=[
            (r"m\.stock\.naver\.com/api/news", FakeResponse("x", status=500)),
            (r"news\.google\.com", FakeResponse("x", status=500)),
        ]
    )
    with pytest.raises(RuntimeError):
        prov.news(ecopro(), HOLIDAY_2PM - timedelta(days=21), 30)


def test_article_parser():
    art = kr.parse_naver_article(fx_text("naver_article.html"))
    assert art["press"] == "디지털타임스" and len(art["body"]) > 200
    assert art["published_at"] == datetime(2026, 9, 24, 6, 42, 42, tzinfo=UTC)


# =============================================================================== filings


def test_dart_search_parse_and_pagination():
    rows, n_pages = kr.parse_dart_search(fx_text("dart_search_005930.html"))
    assert len(rows) == 17 and n_pages == 1
    first = rows[0]
    assert first["rcept_no"] == "20260821800837" and first["corp_code"] == "00126380" and first["is_krx"]
    _, n2 = kr.parse_dart_search(fx_text("dart_search_247540_paged.html"))
    assert n2 == 4
    assert kr.is_krx_filing("20260731900486", "코") and kr.is_krx_filing("20260730800077")
    assert not kr.is_krx_filing("20260814002535", "")


def test_select_filings_prioritises_earnings_and_prefers_originals():
    rows, _ = kr.parse_dart_search(fx_text("dart_search_005930.html"))
    sel = kr.select_filings(rows, 6)
    assert sel[0]["rcept_no"] == "20260730800077" and sel[0]["is_earnings"]  # original, not [기재정정]
    assert sel[0]["form_type"] == "영업(잠정)실적(공정공시)"
    assert [r["priority"] for r in sel] == sorted(r["priority"] for r in sel)
    assert any(r["report_nm"].startswith("주요사항보고서") for r in sel[:3])
    assert kr.classify_filing("반기보고서 (2026.06)")[:2] == (2, True)
    assert kr.classify_filing("[기재정정] 주요사항보고서(유상증자결정)")[0] == 1
    assert kr.classify_filing("기업설명회(IR)개최")[0] == 3


def test_select_filings_skips_attachment_only_correction_and_duplicate_separate_release():
    """Regression (live 035720, all versions): '[첨부정정] 주요사항보고서(회사합병결정)' has only a 570-char
    correction notice and used to replace the 08-21 decision; 연결 + 별도 잠정실적 of 08-06 took two slots."""
    rows, _ = kr.parse_dart_search(fx_text("dart_search_035720.html"))
    sel = kr.select_filings(rows, 6)
    nos = [r["rcept_no"] for r in sel]
    assert "20260821000052" in nos and "20260824000219" not in nos  # original merger decision
    earn = [r for r in sel if r["priority"] == 0]
    assert [r["rcept_no"] for r in earn] == ["20260806800010"]  # 연결 only
    assert kr.is_attachment_only_correction("[첨부정정] 주요사항보고서(회사합병결정)")
    assert kr.is_attachment_only_correction("[첨부추가]증권신고서(지분증권)")
    assert not kr.is_attachment_only_correction("[기재정정] 주요사항보고서(유상증자결정)")
    # an attachment-only correction is still used when its original is outside the window
    only = [r for r in rows if r["rcept_no"] == "20260824000219"]
    assert [r["rcept_no"] for r in kr.select_filings(only, 3)] == ["20260824000219"]
    # a separate-only filer keeps its separate release
    sep = [r for r in rows if r["rcept_no"] == "20260806800011"]
    assert kr.select_filings(sep, 3)[0]["is_earnings"]


def test_dart_web_search_requests_all_versions(make_provider):
    """finalReport=recent hid originals superseded by a correction (the Kakao merger decision)."""
    seen: list[dict] = []

    def search(full: str, params: dict, data: Any) -> FakeResponse:
        seen.append(dict(data))
        return FakeResponse(fx_text("dart_search_005930.html"))

    prov, _ = make_provider(routes=[(r"POST https://dart\.fss\.or\.kr/dsab007/detailSearch\.ax", search)])
    prov.filings(samsung(), HOLIDAY_2PM - timedelta(days=120), 6)
    assert seen and all("finalReport" not in d for d in seen)
    assert seen[0]["publicType"] == kr.DART_WEB_TYPES


def test_decode_dart_viewer_without_charset_header():
    raw = (FIX / "dart_viewer_20260730800077.html").read_bytes()  # cp949 bytes, <meta charset=euc-kr>
    for enc in (None, "ISO-8859-1"):  # requests' default without a charset
        text = kr.html_to_text(kr.decode_dart_viewer(raw, enc))
        assert "영업(잠정)실적" in text and "894,924" in text and "�" not in text
    bare = "<p>본문 1,000억원</p>"
    assert kr.decode_dart_viewer(bare.encode("cp949"), None) == bare  # no meta: utf-8 fails -> cp949
    assert kr.decode_dart_viewer(bare.encode("utf-8"), None) == bare


def test_dart_viewer_nodes_and_cp949_body():
    krx = kr.parse_dart_viewer_nodes(fx_text("dart_main_20260730800077.html"))
    assert krx == [(None, "11500896", "0", "0", "0", "HTML")]
    fss = kr.parse_dart_viewer_nodes(fx_text("dart_main_20260922000524.html"))
    assert len(fss) >= 3 and fss[0][5] == "dart4.xsd" and fss[1][0] == "1. 발행회사에 관한 사항"
    text = kr.html_to_text(kr.decode_dart_viewer((FIX / "dart_viewer_20260730800077.html").read_bytes(), "MS949"))
    assert "영업(잠정)실적" in text and "894,924" in text and "�" not in text


def test_filings_keyless_end_to_end(make_provider, settings):
    prov, sess = make_provider()
    since = HOLIDAY_2PM - timedelta(days=120)
    items = prov.filings(samsung(), since, 6)
    assert len(items) == 6 and all(isinstance(f, FilingItem) for f in items)
    assert [f.filed_at for f in items] == sorted((f.filed_at for f in items), reverse=True)
    assert all(f.filed_at.utcoffset() == timedelta(0) for f in items)
    earn = [f for f in items if f.form_type == "영업(잠정)실적(공정공시)"]
    assert len(earn) == 1 and earn[0].is_earnings and "894,924" in earn[0].text and earn[0].source == "DART"
    assert earn[0].filed_at.date() == date(2026, 7, 30)
    research = [f for f in items if f.source == "Naver Research"]
    assert len(research) == 2 and all(f.form_type == "증권사 리포트" and not f.is_earnings for f in research)
    assert "목표주가: 630,000원" in research[0].text
    assert all(len(f.text) <= settings.data.max_filing_chars for f in items)
    posts = [c for c in sess.calls if c.startswith("POST")]
    assert len(posts) == 1  # typed search: one page, no insider flood
    assert prov.data_warnings("005930") == []


def test_filings_research_only_when_dart_down(make_provider):
    prov, _ = make_provider(routes=[(r"POST https://dart\.fss\.or\.kr", FakeResponse("down", status=503))])
    items = prov.filings(samsung(), HOLIDAY_2PM - timedelta(days=60), 6)
    assert items and all(f.source == "Naver Research" for f in items)
    assert any("공시 소스 일부 실패" in w for w in prov.data_warnings("005930"))


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_opendart_zip_parsers():
    corp = _zip(
        {
            "CORPCODE.xml": (
                "<result><list><corp_code>00126380</corp_code><corp_name>삼성전자</corp_name>"
                "<stock_code>005930</stock_code></list><list><corp_code>00000001</corp_code>"
                "<stock_code> </stock_code></list></result>"
            ).encode()
        }
    )
    assert kr.parse_corp_code_zip(corp) == {"005930": "00126380"}
    doc = _zip(
        {
            "20260821000616.xml": '<?xml version="1.0" encoding="euc-kr"?><DOCUMENT><TABLE><TR><TE>취득예정주식</TE>'
            "<TU>1,000,000</TU></TR></TABLE><P>R&D 비용</P></DOCUMENT>".encode("euc-kr")
        }
    )
    text = kr.document_zip_to_text(doc)
    assert "취득예정주식 | 1,000,000" in text and "R&D" in text


def test_filings_with_opendart_key(make_provider, settings):
    corp = _zip(
        {
            "CORPCODE.xml": (
                b"<result><list><corp_code>00126380</corp_code><stock_code>005930</stock_code></list></result>"
            )
        }
    )

    def listing(full: str, params: dict, data: Any) -> FakeResponse:
        ty = params.get("pblntf_ty")
        if ty == "A":
            return FakeResponse(
                {
                    "status": "000",
                    "total_page": 1,
                    "list": [
                        {
                            "rcept_no": "20260814003699",
                            "rcept_dt": "20260814",
                            "report_nm": "반기보고서 (2026.06)",
                            "rm": "",
                            "flr_nm": "삼성전자",
                        }
                    ],
                }
            )
        if ty == "I":
            return FakeResponse(
                {
                    "status": "000",
                    "total_page": 1,
                    "list": [
                        {
                            "rcept_no": "20260730800077",
                            "rcept_dt": "20260730",
                            "rm": "유",
                            "flr_nm": "삼성전자",
                            "report_nm": "연결재무제표기준영업(잠정)실적(공정공시)",
                        }
                    ],
                }
            )
        return FakeResponse({"status": "013", "message": "조회된 데이타가 없습니다."})

    doc = _zip({"x.xml": "<DOCUMENT><P>II. 사업의 내용 반기 실적 설명</P></DOCUMENT>".encode()})
    routes = [
        (
            r"opendart\.fss\.or\.kr/api/corpCode\.xml",
            FakeResponse(corp, content_type="application/zip"),
        ),
        (r"opendart\.fss\.or\.kr/api/list\.json", listing),
        (
            r"opendart\.fss\.or\.kr/api/document\.xml",
            FakeResponse(doc, content_type="application/zip"),
        ),
    ]
    prov, sess = make_provider(routes=routes)
    settings.data.dart_api_key = "k" * 40
    items = prov.filings(samsung(), HOLIDAY_2PM - timedelta(days=120), 4)
    dart = [f for f in items if f.source == "DART OpenAPI"]
    assert {f.form_type for f in dart} == {"영업(잠정)실적(공정공시)", "반기보고서"}
    assert "894,924" in next(f.text for f in dart if f.is_earnings and "잠정" in f.form_type)  # KRX -> web viewer
    assert "반기 실적 설명" in next(f.text for f in dart if f.form_type == "반기보고서")  # OpenDART document
    assert not any(c.startswith("POST") for c in sess.calls)


def test_opendart_error_falls_back_to_web_with_note(make_provider, settings):
    routes = [
        (
            r"opendart\.fss\.or\.kr/api/",
            FakeResponse({"status": "010", "message": "등록되지 않은 인증키입니다."}),
        )
    ]
    prov, _ = make_provider(routes=routes)
    settings.data.dart_api_key = "k" * 40
    items = prov.filings(samsung(), HOLIDAY_2PM - timedelta(days=120), 6)
    assert any(f.source == "DART" for f in items)
    assert any("OpenDART 실패" in w for w in prov.data_warnings("005930"))


# =============================================================================== fx / universe


def test_fx_rate(make_provider):
    hist = pd.DataFrame(
        {"Close": [1360.0, 1366.7]},
        index=pd.DatetimeIndex(["2026-09-22", "2026-09-23"]),
    )
    prov, _ = make_provider(yf=FakeYF(histories={"KRW=X": hist}))
    assert prov.fx_rate("KRW", "KRW") == 1.0
    assert prov.fx_rate("USD", "KRW") == pytest.approx(1366.7)
    assert prov.fx_rate("KRW", "USD") == pytest.approx(1 / 1366.7)
    with pytest.raises(ValueError):
        prov.fx_rate("KRW", "JPY")
    prov2, session = make_provider(yf=FakeYF(fail={"KRW=X"}))
    assert prov2.fx_rate("KRW", "USD") == pytest.approx(1 / 1366.7)  # Naver fallback
    assert any("네이버 고시환율" in w for w in prov2.data_warnings("__fx__"))
    assert any("FX_USDKRW/prices" in c for c in session.calls)  # the shared (daily list) fallback


def test_fx_rate_skips_the_live_krw_bar_like_the_us_provider(make_provider):
    """Regression: KR used the last KRW=X close of history(period='10d'), i.e. the running (partial) bar,
    while US used the last COMPLETED bar aligned to the US session -> the two markets converted with
    different rates. 2026-09-24 21:00 KST = 13:00 London: the 09-24 bar is live."""
    london = [pd.Timestamp(d, tz="Europe/London") for d in ("2026-09-22", "2026-09-23", "2026-09-24")]
    hist = pd.DataFrame({"Close": [1373.68, 1350.36, 1399.99]}, index=pd.DatetimeIndex(london))
    evening = datetime(2026, 9, 24, 21, 0, tzinfo=KST)
    prov, _ = make_provider(now=evening, yf=FakeYF(histories={"KRW=X": hist}))
    usd_krw = prov.fx_rate("USD", "KRW")
    assert usd_krw == 1350.36  # the completed 09-23 bar (last completed US session), never 1399.99
    assert prov.fx_rate("KRW", "USD") == 1.0 / usd_krw  # exact reciprocal
    assert prov.data_warnings("__fx__") == []
    with pytest.raises(NotImplementedError):  # same exception type as the US provider (also ValueError)
        prov.fx_rate("USD", "JPY")


def test_universe_exclusion_rules():
    ok = {"end_type": "stock", "market": "KOSPI", "market_cap": 1e12}
    assert kr.universe_exclusion_reason({**ok, "code": "138040", "name": "메리츠금융지주"}) is None
    assert kr.universe_exclusion_reason({**ok, "code": "369370", "name": "블리츠웨이엔터테인먼트"}) is None
    assert kr.universe_exclusion_reason({**ok, "code": "005935", "name": "삼성전자우"}) == "preferred"
    assert kr.universe_exclusion_reason({**ok, "code": "00680K", "name": "미래에셋증권2우B"}) == "preferred"
    assert kr.universe_exclusion_reason({**ok, "code": "395400", "name": "SK리츠"}) == "reit"
    assert kr.universe_exclusion_reason({**ok, "code": "088260", "name": "이리츠코크렙"}) == "reit"
    assert kr.universe_exclusion_reason({**ok, "code": "0165X0", "name": "메리츠제2호스팩"}) == "spac"
    assert kr.universe_exclusion_reason({**ok, "code": "088980", "name": "맥쿼리인프라"}) == "listed_fund"
    assert kr.universe_exclusion_reason({**ok, "code": "069500", "name": "KODEX 200", "end_type": "etf"}) == "etf_etn"
    assert (
        kr.universe_exclusion_reason(
            {
                **ok,
                "code": "610063",
                "name": "메리츠 KIS CD금리투자 ETN",
                "end_type": "etn",
            }
        )
        == "etf_etn"
    )


def test_universe_top_n_deterministic(make_provider):
    prov, sess = make_provider()
    u = prov.universe(60)
    assert len(u) == 60 and len(set(u)) == 60
    assert all(c.endswith("0") for c in u) and u[:2] == ["005930", "000660"]
    rows = kr.parse_naver_market_value(fx_json("naver_marketvalue_KOSPI.json"), "KOSPI") + kr.parse_naver_market_value(
        fx_json("naver_marketvalue_KOSDAQ.json"), "KOSDAQ"
    )
    cap = {r["code"]: r["market_cap"] for r in rows}
    assert [cap[c] for c in u] == sorted((cap[c] for c in u), reverse=True)
    assert "138040" in u and not {"005935", "088980", "069500", "395400"} & set(u)
    assert "196170" in u  # KOSDAQ names compete on market cap
    n_calls = len(sess.calls)
    assert prov.universe(60) == u and len(sess.calls) == n_calls  # cached for the day


def test_universe_falls_back_to_fdr(make_provider, monkeypatch):
    import FinanceDataReader as fdr  # noqa: N813 - library's documented alias

    listing = pd.DataFrame(
        {
            "Code": ["005930", "005935", "000660", "091990"],
            "Name": ["삼성전자", "삼성전자우", "SK하이닉스", "셀트리온헬스케어"],
            "MarketId": ["STK", "STK", "STK", "KSQ"],
            "Marcap": [1.6e15, 1.7e14, 1.3e15, 1e13],
        }
    )
    monkeypatch.setattr(fdr, "StockListing", lambda market: listing)
    prov, _ = make_provider(routes=[(r"marketValue", FakeResponse("down", status=503))])
    assert prov.universe(3) == ["005930", "000660", "091990"]
    assert any("FinanceDataReader" in w for w in prov.data_warnings("__universe__"))


# ===================================================================== earnings schedule


def notice_text(rcp: str) -> str:
    """Real KRX 안내공시 viewer page (MS949 bytes, as served) -> the provider's text form."""
    raw = (FIX / f"dart_viewer_{rcp}.html").read_bytes()
    return kr.html_to_text(kr.decode_dart_viewer(raw, "MS949"))


@pytest.mark.parametrize(
    "rcp, status, kind, when",
    [
        ("20260710900467", "earnings", "preannounce", date(2026, 7, 31)),  # 결산실적공시예고 (KOSDAQ)
        ("20260710900470", "earnings", "ir", date(2026, 7, 31)),  # KOSDAQ IR: 2분기 경영실적 발표
        ("20260707800015", "earnings", "ir", date(2026, 7, 30)),  # KOSPI IR (Samsung conference call)
        ("20260904900599", "other", "ir", date(2026, 9, 16)),  # broker corporate day (NDR)
        ("20260918800063", "other", "ir", date(2026, 9, 21)),  # CLSA conference, Hong Kong
    ],
)
def test_parse_earnings_notice_real_layouts(rcp, status, kind, when):
    ev = kr.parse_earnings_notice(notice_text(rcp))
    assert (ev["status"], ev["kind"], ev["date"]) == (status, kind, when)


def test_parse_earnings_notice_roadshow_after_results_is_not_a_release():
    call = notice_text("20260707800015")
    assert kr.parse_earnings_notice(call)["status"] == "earnings"
    ndr = call.replace("2026년 2분기 경영실적 발표", "2026년 2분기 실적발표 후 해외 NDR")
    assert kr.parse_earnings_notice(ndr)["status"] == "other"
    kosdaq_call = notice_text("20260710900470")
    assert "온라인 컨퍼런스 콜" in kosdaq_call  # a conference CALL is still an earnings call
    assert kr.parse_earnings_notice(kosdaq_call)["status"] == "earnings"


def test_parse_earnings_notice_never_reads_decision_or_posting_dates():
    # item 1 (일시) is the event; 결정일자/개최확정일/게재일시 must never be used, and an unknown layout
    # is "unparsed" (reported by the caller), not guessed
    samsung_ir = notice_text("20260707800015")
    assert "7. 결정일자 | 2026-07-07" in samsung_ir
    no_item1 = re.sub(r"^\s*1\.\s*일시[^\n]*\n", "", samsung_ir, flags=re.M)
    assert kr.parse_earnings_notice(no_item1)["status"] == "unparsed"
    assert kr.parse_earnings_notice("영업(잠정)실적(공정공시) 본문")["status"] == "unparsed"
    no_date = notice_text("20260710900467").replace("2026-07-31", "미정")
    assert kr.parse_earnings_notice(no_date)["status"] == "unparsed"


def test_next_earnings_from_notices_rules():
    today = date(2026, 7, 20)

    def ev(d, status="earnings"):
        return {"status": status, "date": d}

    agree = [ev(date(2026, 7, 31)), ev(date(2026, 7, 31)), ev(date(2026, 7, 25), "other")]
    assert kr.next_earnings_from_notices(agree, today) == (date(2026, 7, 31), False)
    clash = [ev(date(2026, 7, 31)), ev(date(2026, 7, 30))]  # e.g. a [기재정정] moved the date
    assert kr.next_earnings_from_notices(clash, today) == (date(2026, 7, 30), True)
    assert kr.next_earnings_from_notices([ev(date(2026, 7, 10))], today) == (None, None)
    assert kr.next_earnings_from_notices([], today) == (None, None)
    # release day: the 잠정실적 is already filed -> not "next"
    assert kr.next_earnings_from_notices([ev(date(2026, 7, 20))], today, released=[date(2026, 7, 20)]) == (
        None,
        None,
    )


def notice_routes(overrides: dict[str, str] | None = None) -> list[tuple[str, Any]]:
    """DART main/viewer per rcpNo: saved notice fixtures; unknown rcpNos answer with a real NDR notice.
    ``overrides``: rcpNo -> text body served instead (for synthetic layouts)."""
    overrides = overrides or {}

    def main(full: str, params: dict, data: Any) -> FakeResponse:
        rcp = params.get("rcpNo", "")
        return FakeResponse(f'viewDoc("{rcp}", "{rcp}", "0", "0", "0", "HTML");')

    def viewer(full: str, params: dict, data: Any) -> FakeResponse:
        rcp = params.get("rcpNo", "")
        if rcp in overrides:
            return FakeResponse(overrides[rcp])
        path = FIX / f"dart_viewer_{rcp}.html"
        if not path.exists():
            path = FIX / "dart_viewer_20260904900599.html"
        return FakeResponse(path.read_bytes(), content_type="text/html; charset=MS949", encoding="MS949")

    return [
        (r"dart\.fss\.or\.kr/dsaf001/main\.do", main),
        (r"dart\.fss\.or\.kr/report/viewer\.do", viewer),
    ]


def test_fundamentals_next_earnings_from_kosdaq_preannouncement(make_provider):
    # 2026-07-20: 에코프로비엠 filed 결산실적공시예고 + an earnings-call IR (both 07-31) on 07-10 plus NDRs
    prov, sess = make_provider(routes=notice_routes(), now=datetime(2026, 7, 20, 14, 0, tzinfo=KST))
    f = prov.fundamentals(ecopro())
    assert (f.next_earnings_date, f.next_earnings_is_estimate) == (date(2026, 7, 31), False)
    assert not any("실적발표 일정" in w for w in prov.data_warnings("247540"))
    viewed = {re.search(r"rcpNo=(\d+)", c).group(1) for c in sess.calls if "viewer.do" in c}
    assert {"20260710900467", "20260710900470"} <= viewed
    assert not viewed & {"20260731900494", "20260814902052", "20260904900599"}  # filed after "today"
    assert len(viewed) <= kr.EARNINGS_NOTICE_MAX_DOCS


def test_fundamentals_next_earnings_from_kospi_ir_notice(make_provider):
    # Samsung pre-announces only the confirmed-results call (filed with the 잠정실적 on 07-07)
    prov, _ = make_provider(routes=notice_routes(), now=datetime(2026, 7, 15, 10, 0, tzinfo=KST))
    f = prov.fundamentals(samsung())
    assert (f.next_earnings_date, f.next_earnings_is_estimate) == (date(2026, 7, 30), False)
    # the preferred share uses the issuer's notices
    prov2, sess2 = make_provider(
        routes=[*notice_routes(), *pref_routes()], now=datetime(2026, 7, 15, 10, 0, tzinfo=KST)
    )
    assert prov2.fundamentals(samsung_pref()).next_earnings_date == date(2026, 7, 30)


def test_fundamentals_no_schedule_is_none_not_guessed(make_provider):
    # 2026-09-24: Samsung's Q3 잠정실적 (early Oct) is never pre-announced -> None, and not a warning
    prov, _ = make_provider(routes=notice_routes())
    f = prov.fundamentals(samsung())
    assert f.next_earnings_date is None and f.next_earnings_is_estimate is None
    assert prov.data_warnings("005930") == []


def test_fundamentals_conflicting_notices_use_earliest_and_flag_estimate(make_provider):
    moved = notice_text("20260710900470").replace("2026-07-31 | 2026-07-31", "2026-07-30 | 2026-07-30")
    prov, _ = make_provider(
        routes=notice_routes({"20260710900470": f"<pre>{moved}</pre>"}),
        now=datetime(2026, 7, 20, 14, 0, tzinfo=KST),
    )
    f = prov.fundamentals(ecopro())
    assert (f.next_earnings_date, f.next_earnings_is_estimate) == (date(2026, 7, 30), True)


def test_fundamentals_release_day_is_not_next(make_provider):
    # 07-31 15:00: the 07-31 잠정실적 is on DART -> the pre-announced 07-31 is no longer "next"
    prov, _ = make_provider(routes=notice_routes(), now=datetime(2026, 7, 31, 15, 0, tzinfo=KST))
    f = prov.fundamentals(ecopro())
    assert f.next_earnings_date is None and f.next_earnings_is_estimate is None


def test_fundamentals_schedule_failures_are_reported(make_provider):
    prov, _ = make_provider(routes=[(r"POST https://dart\.fss\.or\.kr", FakeResponse("down", status=503))])
    f = prov.fundamentals(samsung())
    assert f.per == 12.81 and f.next_earnings_date is None  # the rest of fundamentals is intact
    assert any("실적발표 일정 조회 실패" in w and "미확인" in w for w in prov.data_warnings("005930"))
    # notices found but none readable -> reported (a readable NDR alone would be a normal "none")
    garbled = {r: "<p>점검 중</p>" for r in ("20260710900467", "20260710900470")}
    prov2, _ = make_provider(
        routes=[
            *notice_routes(garbled),
            (r"dart\.fss\.or\.kr/report/viewer\.do", FakeResponse("x", status=500)),
        ],
        now=datetime(2026, 7, 20, 14, 0, tzinfo=KST),
    )
    f2 = prov2.fundamentals(ecopro())
    assert f2.next_earnings_date is None
    assert any("실적발표 일정 공시" in w and "확인 불가" in w for w in prov2.data_warnings("247540"))


def test_quarterly_financials_missing_is_noted():
    fin_a = kr.parse_naver_finance(fx_json("naver_finance_annual_005930.json"))
    snap = kr.parse_naver_integration(fx_json("naver_integration_005930.json"))
    fund, notes = kr.compose_fundamentals(snap, None, fin_a, None, None, date(2026, 9, 24))
    assert fund.revenue_growth_yoy is not None  # annual fallback
    assert any("분기 재무 수집 실패" in n and "연간 기준" in n for n in notes)


# ====================================================================== warnings plumbing


def test_data_warnings_include_provider_wide_fx_notes(make_provider):
    prov, _ = make_provider(yf=FakeYF(fail={"KRW=X"}))
    prov.fx_rate("KRW", "USD")
    fx = prov.data_warnings("__fx__")
    assert fx and all(n in prov.data_warnings("005930") for n in fx)
    assert prov.data_warnings("005930") == prov.data_warnings("005930")  # non-destructive
    assert prov.data_warnings("__universe__") == []


def test_benchmark_notes_match_the_loader_prefix_and_price_attrs(make_provider):
    hist = yf_raw_history({"2026-09-22": 7017.91, "2026-09-23": 7080.92})
    prov, _ = make_provider(
        routes=[(r"fchart\.stock\.naver\.com.*symbol=KOSPI", FakeResponse("down", status=503))],
        yf=FakeYF(histories={"^KS11": hist}),
    )
    b = prov.benchmark(samsung(), date(2026, 9, 1))
    assert b.attrs["warnings"] and not any(w.startswith("벤치마크") for w in b.attrs["warnings"])
    notes = prov.data_warnings("005930")
    assert all(f"벤치마크: {w}" in notes for w in b.attrs["warnings"])  # == loader's merged text
    px = prov.prices(samsung(), date(2026, 7, 1))
    assert px.attrs["split_adjusted"] is True and px.attrs["dividend_adjusted"] is False
    assert "adjusted" not in px.attrs  # Naver is the documented source, not a degraded fallback


def test_dart_search_truncation_is_noted(make_provider):
    prov, _ = make_provider(
        routes=[
            (
                r"POST https://dart\.fss\.or\.kr/dsab007/detailSearch\.ax",
                FakeResponse(fx_text("dart_search_247540_paged.html")),
            )
        ]
    )
    notes: list[str] = []
    prov._dart_web_rows("247540", date(2026, 5, 1), date(2026, 9, 24), max_pages=2, notes=notes)
    assert notes and "2/4페이지" in notes[0]


def test_load_market_data_merges_kr_warnings_once(make_provider, settings, monkeypatch):
    """Offline end-to-end through the shared loader: degraded price / benchmark / FX paths reach
    MarketData.warnings exactly once (frame attrs + data_warnings carry identical text)."""
    from quantcio.data import base

    idx = pd.bdate_range(end="2026-09-23", periods=400)
    xml = "\n".join(f'<item data="{d:%Y%m%d}|100|110|90|105|1000" />' for d in idx)
    hist = yf_raw_history({"2026-09-22": 7017.91, "2026-09-23": 7080.92})
    prov, _ = make_provider(
        routes=[
            (r"fchart\.stock\.naver\.com.*symbol=005930", FakeResponse(f"<chartdata>{xml}</chartdata>")),
            (r"fchart\.stock\.naver\.com.*symbol=KOSPI", FakeResponse("down", status=503)),
            (r"finance\.daum\.net/api/", FakeResponse("forbidden", status=403)),
        ],
        yf=FakeYF(histories={"^KS11": hist}, fail={"KRW=X"}),
    )
    monkeypatch.setattr(base, "get_provider", lambda m, s, c: prov)
    md = base.load_market_data("005930", settings, prov.cache, base_currency="USD", include_text=False)
    w = md.warnings
    assert len(w) == len(set(w)), w
    assert sum("공식종가 미확인" in x for x in w) == 1
    bench = [x for x in w if "yfinance ^KS11" in x]
    assert len(bench) == 1 and bench[0].startswith("벤치마크: ")
    assert any("네이버 고시환율" in x for x in w)  # FX fallback reaches the ticker
    assert isinstance(md.prices.attrs.get("warnings"), list)


# ============================================================================ report language

HANGUL = re.compile("[가-힣]")


def in_lang(prov: KRProvider, lang: str) -> KRProvider:
    """The provider with a deep copy of its settings set to ``report.language = lang``."""
    s = prov.settings.model_copy(deep=True)
    s.report.language = lang
    prov.settings = s
    return prov


def assert_lang(texts: list[str], lang: str) -> None:
    assert texts
    for t in texts:
        if lang == "en":
            assert HANGUL.search(t) is None, t
        else:
            assert HANGUL.search(t) is not None, t


def test_module_helpers_follow_lang_keyword():
    with pytest.raises(ValueError, match="6자리") as ko:
        kr.check_code("AAPL")
    with pytest.raises(ValueError, match="Not a 6-character KRX stock code") as en:
        kr.check_code("AAPL", lang="en")
    assert HANGUL.search(str(en.value)) is None and HANGUL.search(str(ko.value))
    with pytest.raises(ValueError) as exc:
        kr.parse_fchart("<chartdata></chartdata>", "AAPL", lang="en")
    assert HANGUL.search(str(exc.value)) is None
    fin_a = kr.parse_naver_finance(fx_json("naver_finance_annual_005930.json"))
    snap = kr.parse_naver_integration(fx_json("naver_integration_005930.json"))
    _, notes_en = kr.compose_fundamentals(snap, None, fin_a, None, None, date(2026, 9, 24), lang="en")
    assert_lang(notes_en, "en")
    assert any("Naver quarterly financials unavailable" in n for n in notes_en)
    _, notes_ko = kr.compose_fundamentals(snap, None, fin_a, None, None, date(2026, 9, 24))
    assert "네이버 분기 재무 수집 실패 → 성장률·영업이익률은 최근 연간 기준 (최근 분기 미반영)" in notes_ko
    with pytest.raises(RuntimeError) as exc:
        kr.compose_fundamentals(None, None, None, None, None, date(2026, 9, 24), lang="en")
    assert HANGUL.search(str(exc.value)) is None


@pytest.mark.parametrize("lang", ["ko", "en"])
def test_price_and_benchmark_warnings_follow_report_language(make_provider, lang):
    closes = {"2026-09-21": 274000.0, "2026-09-22": 276500.0, "2026-09-23": 285500.0}
    yf = FakeYF(histories={"005930.KS": yf_raw_history(closes)})
    prov, _ = make_provider(routes=[(r"fchart\.stock\.naver\.com", FakeResponse("down", status=503))], yf=yf)
    df = in_lang(prov, lang).prices(samsung(), date(2026, 9, 1))
    assert_lang(df.attrs["warnings"], lang)
    head = {
        "ko": "네이버 시세 실패(HTTPError: HTTP 503) → yfinance 005930.KS 배당조정 시계열로 대체",
        "en": "Naver prices failed (HTTPError: HTTP 503) → replaced by the yfinance 005930.KS",
    }
    assert df.attrs["warnings"][0].startswith(head[lang])

    prov2, _ = make_provider(routes=[(r"finance\.daum\.net/api/", FakeResponse("forbidden", status=403))], yf=yf)
    w = in_lang(prov2, lang).prices(samsung(), date(2026, 7, 1)).attrs["warnings"]
    assert_lang(w, lang)
    expect = {
        "ko": ["Daum 공식종가 조회 실패", "공식종가 미확인"],
        "en": ["Daum official close lookup failed", "official close unconfirmed"],
    }
    assert all(any(e in x for x in w) for e in expect[lang])

    hist = yf_raw_history({"2026-09-22": 7017.91, "2026-09-23": 7080.92})
    prov3, _ = make_provider(
        routes=[(r"fchart\.stock\.naver\.com.*symbol=KOSPI", FakeResponse("down", status=503))],
        yf=FakeYF(histories={"^KS11": hist}),
    )
    b = in_lang(prov3, lang).benchmark(samsung(), date(2026, 9, 1))
    assert_lang(b.attrs["warnings"], lang)
    prefix = {"ko": "벤치마크: ", "en": "Benchmark: "}[lang]
    notes = prov3.data_warnings("005930")
    assert all(f"{prefix}{x}" in notes for x in b.attrs["warnings"])
    assert_lang(notes, lang)


@pytest.mark.parametrize("lang", ["ko", "en"])
def test_resolve_fundamentals_news_notes_follow_report_language(make_provider, lang):
    prov, _ = make_provider(
        routes=[
            (r"m\.stock\.naver\.com/api/stock/005930/basic", FakeResponse({"code": "StockConflict"}, status=409)),
            (r"wisereport", FakeResponse("down", status=503)),
            (r"POST https://dart\.fss\.or\.kr", FakeResponse("down", status=503)),
        ]
    )
    in_lang(prov, lang)
    prov.resolve("005930")
    prov.fundamentals(samsung())
    notes = prov.data_warnings("005930")
    assert_lang(notes, lang)
    expect = {
        "ko": ["종목 식별 1차 소스(네이버) 실패 → 대체 소스 사용: ", "WiseReport 수집 실패", "실적발표 일정 조회 실패"],
        "en": ["Stock identification: primary source (Naver) failed", "WiseReport unavailable", "Earnings schedule"],
    }[lang]
    assert all(any(e in n for n in notes) for e in expect)

    empty_rss = FakeResponse('<rss version="2.0"><channel></channel></rss>')
    prov2, _ = make_provider(routes=[(r"news\.google\.com", empty_rss)])
    in_lang(prov2, lang).news(ecopro().model_copy(update={"name": "NoSuchCo"}), HOLIDAY_2PM - timedelta(days=21), 5)
    news_notes = prov2.data_warnings("247540")
    assert_lang(news_notes, lang)
    assert any({"ko": "일반 시황 기사", "en": "general market articles"}[lang] in n for n in news_notes)


def test_failures_fx_dart_and_universe_notes_are_english(make_provider, monkeypatch):
    prov, _ = make_provider(
        routes=[
            (r"wisereport", FakeResponse("down", status=503)),
            (r"m\.stock\.naver\.com/api/stock/005930/integration", FakeResponse("x", status=500)),
        ]
    )
    with pytest.raises(RuntimeError, match="Naver and WiseReport fundamentals both unavailable") as exc:
        in_lang(prov, "en").fundamentals(samsung())
    assert HANGUL.search(str(exc.value)) is None

    prov2, _ = make_provider(
        routes=[
            (r"m\.stock\.naver\.com/api/news", FakeResponse("x", status=500)),
            (r"news\.google\.com", FakeResponse("x", status=500)),
        ]
    )
    with pytest.raises(RuntimeError, match="News fetch failed") as exc:
        in_lang(prov2, "en").news(ecopro(), HOLIDAY_2PM - timedelta(days=21), 30)
    assert HANGUL.search(str(exc.value)) is None

    prov3, _ = make_provider(yf=FakeYF(fail={"KRW=X"}))
    in_lang(prov3, "en").fx_rate("KRW", "USD")
    fx = prov3.data_warnings("__fx__")
    assert_lang(fx, "en")
    assert fx[0].startswith("FX: Yahoo (KRW=X) failed → using the Naver posted rate")
    with pytest.raises(ValueError) as exc:
        prov3.fx_rate("KRW", "JPY")
    assert HANGUL.search(str(exc.value)) is None

    prov4, _ = make_provider(
        routes=[
            (
                r"POST https://dart\.fss\.or\.kr/dsab007/detailSearch\.ax",
                FakeResponse(fx_text("dart_search_247540_paged.html")),
            )
        ]
    )
    notes: list[str] = []
    in_lang(prov4, "en")._dart_web_rows("247540", date(2026, 5, 1), date(2026, 9, 24), max_pages=2, notes=notes)
    assert_lang(notes, "en")
    assert "page 2/4" in notes[0]

    import FinanceDataReader as fdr  # noqa: N813 - library's documented alias

    listing = pd.DataFrame({"Code": ["005930"], "Name": ["삼성전자"], "MarketId": ["STK"], "Marcap": [1.6e15]})
    monkeypatch.setattr(fdr, "StockListing", lambda market: listing)
    prov5, _ = make_provider(routes=[(r"marketValue", FakeResponse("down", status=503))])
    assert in_lang(prov5, "en").universe(3) == ["005930"]
    uni = prov5.data_warnings("__universe__")
    assert_lang(uni, "en")
    assert any("FinanceDataReader" in w for w in uni) and any("only 1/3" in w for w in uni)


# ============================================================================ English company names


class CountingYF(FakeYF):
    """FakeYF that counts ``.info`` reads per symbol."""

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self.calls: dict[str, int] = {}

    def Ticker(self, symbol: str) -> FakeTicker:  # noqa: N802 - mimics yfinance
        self.calls[symbol] = self.calls.get(symbol, 0) + 1
        return super().Ticker(symbol)


def test_clean_english_name_strips_legal_suffixes():
    cases = {
        "Samsung Electronics Co., Ltd.": "Samsung Electronics",
        "SAMSUNG BIOLOGICS Co.,Ltd.": "SAMSUNG BIOLOGICS",
        "Hyundai Mobis Co.,Ltd": "Hyundai Mobis",
        "Ecopro BM. Co., Ltd.": "Ecopro BM",
        "SK hynix Inc.": "SK hynix",
        "NAVER Corporation": "NAVER",
        "Kakao Corp.": "Kakao",
        "LG Chem, Ltd.": "LG Chem",
        "Samsung C&amp;T Corporation": "Samsung C&T",
        "  POSCO   Holdings Inc. ": "POSCO Holdings",
    }
    for raw, want in cases.items():
        assert kr.clean_english_name(raw, "005930") == want, raw
    for bad in (None, "", "   ", "삼성전자", "Co., Ltd.", "005930.KS", "005930", 12345, "0126Z0.KQ"):
        assert kr.clean_english_name(bad, "005930") is None, bad


def test_english_name_is_cached_for_30_days_and_failures_stay_quiet(make_provider, caplog):
    info = {"005930.KS": {"longName": "Samsung Electronics Co., Ltd.", "shortName": "SamsungElec"}}
    yf = CountingYF(infos=info)
    prov, _ = make_provider(yf=yf)
    assert prov.english_name(samsung()) == "Samsung Electronics"
    assert prov.english_name(samsung()) == "Samsung Electronics" and yf.calls["005930.KS"] == 1  # disk cache
    key = DiskCache.key("en_name", "005930.KS")
    assert prov.cache.get(kr.EN_NAME_NS, key, kr.EN_NAME_TTL_HOURS)["name"] == "Samsung Electronics"
    assert kr.EN_NAME_TTL_HOURS == 30 * 24
    # shortName is the fallback; no yf symbol -> no lookup at all
    prov2, _ = make_provider(yf=CountingYF(infos={"005930.KS": {"shortName": "SamsungElec"}}))
    assert prov2.english_name(samsung()) == "SamsungElec"
    assert prov2.english_name(samsung().model_copy(update={"yf_symbol": None})) is None
    # a failing lookup: None, DEBUG only (no warning noise, no report note), not cached -> retried next time
    down = CountingYF(infos=info, fail={"005930.KS"})
    prov3, _ = make_provider(yf=down)
    with caplog.at_level("DEBUG", logger="quantcio.data.kr"):
        assert prov3.english_name(samsung()) is None
    assert [r for r in caplog.records if r.name == "quantcio.data.kr" and "English name" in r.getMessage()]
    assert not [r for r in caplog.records if r.levelno >= 30]
    assert prov3.data_warnings("005930") == []
    down.fail.clear()
    assert prov3.english_name(samsung()) == "Samsung Electronics" and down.calls["005930.KS"] == 2
    # an info without any usable name is a miss cached for a day, then retried
    empty = CountingYF(infos={"005930.KS": {"longName": "005930.KS"}})
    prov4, _ = make_provider(yf=empty)
    assert prov4.english_name(samsung()) is None and prov4.english_name(samsung()) is None
    assert empty.calls["005930.KS"] == 1
    prov4.cache.set(kr.EN_NAME_NS, key, {"name": None, "at": 0.0})  # the miss is older than a day
    empty.infos["005930.KS"] = info["005930.KS"]
    assert prov4.english_name(samsung()) == "Samsung Electronics" and empty.calls["005930.KS"] == 2


@pytest.mark.parametrize("lang", ["ko", "en"])
def test_load_market_data_shows_the_english_name_only_in_english(make_provider, settings, monkeypatch, lang):
    from quantcio.data import base

    idx = pd.bdate_range(end="2026-09-23", periods=400)
    xml = "\n".join(f'<item data="{d:%Y%m%d}|100|110|90|105|1000" />' for d in idx)
    yf = CountingYF(infos={"005930.KS": {"longName": "Samsung Electronics Co., Ltd."}})
    prov, _ = make_provider(
        routes=[(r"fchart\.stock\.naver\.com.*symbol=005930", FakeResponse(f"<chartdata>{xml}</chartdata>"))],
        yf=yf,
    )
    s = settings.model_copy(deep=True)
    s.report.language = lang
    prov.settings = s
    monkeypatch.setattr(base, "get_provider", lambda m, st, c: prov)
    md = base.load_market_data("005930", s, prov.cache, base_currency="KRW", include_text=False)
    if lang == "en":
        assert md.instrument.name == "Samsung Electronics"
    else:
        assert md.instrument.name == prov.resolve("005930").name and HANGUL.search(md.instrument.name)
        # the Korean report never looks the name up
        assert prov.cache.get(kr.EN_NAME_NS, DiskCache.key("en_name", "005930.KS"), None) is None
    assert not any("Samsung" in w or "English name" in w for w in md.warnings)
    assert prov.resolve("005930").name != "Samsung Electronics"  # the cached instrument is never mutated
    # a failed lookup keeps the Korean name without a warning
    yf.fail.add("005930.KS")
    prov.cache.set(kr.EN_NAME_NS, DiskCache.key("en_name", "005930.KS"), {"name": None, "at": 0.0})
    md2 = base.load_market_data("005930", s, prov.cache, base_currency="KRW", include_text=False)
    assert HANGUL.search(md2.instrument.name) and md2.warnings == md.warnings


# ================================================================================== live


LIVE_START = date.today() - timedelta(days=int(365.25 * 7) + 30)


@pytest.fixture
def live_provider(settings, tmp_path):
    return KRProvider(settings, DiskCache(tmp_path / "live_cache"))


@pytest.mark.live
@pytest.mark.parametrize("code", ["005930", "247540"])
def test_live_end_to_end(live_provider, code):
    prov = live_provider
    inst = prov.resolve(code)
    assert inst.exchange == ("KOSPI" if code == "005930" else "KOSDAQ") and inst.name and inst.sector

    px = prov.prices(inst, LIVE_START)
    assert len(px) >= 1500 and list(px.columns) == PRICE_COLUMNS and px.index.is_unique
    last_day, last_close = px.index[-1], float(px["close"].iloc[-1])
    # independent official-close checks: WiseReport (FnGuide) and Yahoo (raw) for the same day
    wise = kr.parse_wisereport_snapshot(
        requests.get(
            f"{kr.WISE}/v2/company/c1010001.aspx",
            params={"cmp_cd": code},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        ).text
    )
    y = (
        kr._yf()
        .Ticker(inst.yf_symbol)
        .history(
            start=(last_day - pd.Timedelta(7, "D")).date().isoformat(),
            auto_adjust=False,
        )
    )
    y.index = kr._naive_daily_index(y.index)
    yahoo = float(y["Close"].get(last_day, np.nan)) if not y.empty else np.nan
    assert last_close in (wise["official_close"], yahoo), (
        last_day,
        last_close,
        wise["official_close"],
        yahoo,
    )
    assert not any("미확인" in w for w in px.attrs["warnings"]), px.attrs["warnings"]

    bench = prov.benchmark(inst, LIVE_START)
    assert len(bench) >= 1500

    f = prov.fundamentals(inst)
    assert f.per is not None and f.pbr is not None and f.market_cap and f.analyst_target_price

    news = prov.news(inst, datetime.now(UTC) - timedelta(days=21), 30)
    assert len(news) >= 1

    filings = prov.filings(inst, datetime.now(UTC) - timedelta(days=120), 6)
    assert isinstance(filings, list) and filings and all(isinstance(x, FilingItem) for x in filings)
    assert any(x.text for x in filings)

    if code == "005930":
        v = prov.valuation_history(inst, LIVE_START)
        assert v is not None and len(v) > 500 and v["per"].dropna().iloc[-1] > 0


@pytest.mark.live
@pytest.mark.parametrize("code", ["005930", "247540", "035720"])
def test_live_earnings_schedule(live_provider, code):
    nxt, est, notes = live_provider._earnings_schedule(code)
    today = datetime.now(KST).date()
    assert (nxt is None and est is None) or (nxt >= today and isinstance(est, bool))
    assert not any("확인 불가" in n for n in notes), notes  # every live notice layout is recognised
    print(code, nxt, est, notes)


@pytest.mark.live
@pytest.mark.parametrize("code", ["005930", "247540"])
def test_live_load_market_data_warnings(settings, tmp_path, code):
    from quantcio.data.base import load_market_data

    md = load_market_data(code, settings, DiskCache(tmp_path / "lmd"), base_currency="KRW")
    assert isinstance(md.warnings, list) and len(md.warnings) == len(set(md.warnings))
    assert isinstance(md.prices.attrs.get("warnings"), list)
    print(code, md.fundamentals.next_earnings_date, md.fundamentals.next_earnings_is_estimate, md.warnings)


@pytest.mark.live
def test_live_universe_and_fx(live_provider):
    u = live_provider.universe(60)
    assert len(u) == 60 and all(c.endswith("0") for c in u) and "005930" in u
    rate = live_provider.fx_rate("KRW", "USD")
    assert 1 / 3000 < rate < 1 / 500
