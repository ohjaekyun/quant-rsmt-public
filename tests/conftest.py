"""Shared synthetic fixtures (no network). Module tests should build on these."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quantcio.config import Settings, load_settings
from quantcio.models import (
    FilingItem,
    Fundamentals,
    Instrument,
    Market,
    MarketData,
    NewsItem,
    text_id,
)


def make_prices(
    n: int = 1800,
    start_price: float = 50_000.0,
    drift: float = 0.0003,
    vol: float = 0.018,
    seed: int = 0,
    end: str = "2026-09-23",
) -> pd.DataFrame:
    """Geometric random walk OHLCV on business days, satisfying the PRICE_COLUMNS contract."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n)
    rets = rng.normal(drift, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = close * np.exp(rng.normal(0, vol / 3, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.lognormal(mean=13, sigma=0.4, size=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx).astype(
        "float64"
    )


def make_instrument(market: Market = Market.KR) -> Instrument:
    if market is Market.KR:
        return Instrument(
            ticker="005930",
            market=Market.KR,
            name="삼성전자",
            exchange="KOSPI",
            currency="KRW",
            sector="반도체",
            yf_symbol="005930.KS",
            benchmark="KOSPI",
            benchmark_name="KOSPI",
        )
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


def make_news(n: int = 8, now: datetime | None = None) -> list[NewsItem]:
    now = now or datetime(2026, 9, 24, tzinfo=UTC)
    titles = [
        ("삼성전자, HBM4 엔비디아 퀄 테스트 통과…4분기 공급 본격화", "HBM4 12단 제품이 최종 품질 테스트를 통과했다."),
        ("삼성전자 3분기 영업이익 10조원 전망…시장 기대 상회", "증권가는 3분기 영업이익이 10조원을 넘길 것으로 본다."),
        ("외국인, 삼성전자 5거래일 연속 순매수", "외국인 투자자가 5거래일 연속 순매수했다."),
        ("삼성전자 파운드리 수율 부진 지속", "2나노 공정 수율이 기대에 못 미친다는 지적이 나온다."),
        ("美 반도체 수출 규제 강화 우려", "미국 정부가 추가 수출 규제를 검토 중이다."),
        ("삼성전자, 자사주 3조원 추가 매입 결정", "주주환원 강화를 위해 자사주를 추가 매입한다."),
        ("코스피, 외국인 매수에 2% 상승", "반도체 대형주가 지수 상승을 이끌었다."),
        ("삼성전자 노조 파업 예고", "노조가 임금 협상 결렬로 파업을 예고했다."),
    ]
    items = []
    for i in range(n):
        t, s = titles[i % len(titles)]
        ts = now - timedelta(days=i * 2, hours=3)
        items.append(
            NewsItem(
                id=text_id(t, ts.isoformat()),
                title=t,
                summary=s,
                url=f"https://example.com/{i}",
                source="테스트뉴스",
                published_at=ts,
                lang="ko",
            )
        )
    return items


def make_filings(now: datetime | None = None) -> list[FilingItem]:
    now = now or datetime(2026, 9, 24, tzinfo=UTC)
    text = (
        "연결재무제표 기준 영업(잠정)실적(공정공시)\n"
        "매출액 당기실적 79조 1,000억원, 전년동기 대비 12.3% 증가\n"
        "영업이익 당기실적 10조 4,000억원, 전년동기 대비 145.2% 증가\n"
        "당기순이익 8조 2,000억원\n"
    )
    return [
        FilingItem(
            id=text_id("잠정실적"),
            title="연결재무제표기준영업(잠정)실적(공정공시)",
            form_type="영업(잠정)실적(공정공시)",
            filed_at=now - timedelta(days=10),
            url="https://dart.fss.or.kr/x",
            text=text,
            is_earnings=True,
            source="DART",
        )
    ]


def make_market_data(market: Market = Market.KR, seed: int = 0, drift: float = 0.0004) -> MarketData:
    inst = make_instrument(market)
    px = make_prices(seed=seed, drift=drift, start_price=50_000.0 if market is Market.KR else 150.0)
    bench = make_prices(seed=seed + 100, drift=0.0002, vol=0.011, start_price=2500.0 if market is Market.KR else 4500.0)
    val_idx = px.index[::5]
    val_hist = pd.DataFrame(
        {"per": np.linspace(9, 18, len(val_idx)), "pbr": np.linspace(1.0, 1.8, len(val_idx))}, index=val_idx
    )
    fund = Fundamentals(
        source="synthetic",
        per=13.5,
        forward_per=11.0,
        pbr=1.4,
        eps=5500.0,
        bps=55000.0,
        roe=0.105,
        operating_margin=0.13,
        dividend_yield=0.02,
        revenue_growth_yoy=0.12,
        earnings_growth_yoy=0.45,
        sector_per=15.0,
        analyst_target_price=float(px["close"].iloc[-1]) * 1.2,
        analyst_target_high=float(px["close"].iloc[-1]) * 1.45,
        analyst_count=25,
        analyst_rating="Buy",
    )
    return MarketData(
        instrument=inst,
        prices=px,
        benchmark=bench,
        fundamentals=fund,
        valuation_history=val_hist,
        news=make_news(),
        filings=make_filings(),
        fx_to_base=1.0 if market is Market.KR else 1380.0,
    )


@pytest.fixture(autouse=True)
def _no_local_avoid_flag(monkeypatch, tmp_path):
    """Reports rendered in tests never read the developer's own US avoid-flag file (scripts/serve_avoid.py)."""
    from quantcio.report import render

    monkeypatch.setattr(render, "_avoid_file", lambda _settings: tmp_path / "no_avoid_US.json")


@pytest.fixture
def settings() -> Settings:
    return load_settings()


@pytest.fixture
def md_kr() -> MarketData:
    return make_market_data(Market.KR)


@pytest.fixture
def md_us() -> MarketData:
    return make_market_data(Market.US, seed=7)


@pytest.fixture(autouse=True)
def _no_network_pit_universe(request, monkeypatch):
    """Offline tests never build a PIT universe (it downloads marcap / SEC data): ``pipeline.training_panel`` then
    falls back to the providers' current list with its loud note. Tests that exercise the PIT path monkeypatch
    ``quantcio.data.pit.load_pit_universe`` themselves; ``live`` tests keep the real loader. KR dividends (Yahoo,
    total-return labels) are stubbed to "no data"."""
    if request.node.get_closest_marker("live") is not None:
        return

    def offline(*args, **kwargs):
        raise RuntimeError("PIT universe loading is disabled in offline tests")

    import quantcio.data.pit as pit
    import quantcio.data.pit.kr_dividends as kdiv

    monkeypatch.setattr(pit, "load_pit_universe", offline)
    monkeypatch.setattr(kdiv, "_yahoo_history", lambda symbol: None)  # KR labels stay price returns offline
