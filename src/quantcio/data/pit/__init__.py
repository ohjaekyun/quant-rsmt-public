"""Point-in-time (PIT) universes and fundamentals — SHARED CONTRACT for data/pit/kr.py and data/pit/us.py.

Why: the legacy training/evaluation universe is *today's* top-N (KR) or a static list of today's large caps (US),
which selects past winners with hindsight (survivorship bias, measured at +5.6%/yr US and +13.5%/yr KR). A PIT
universe decides membership at each rebalance date t using ONLY information available at t, and keeps names
that were later delisted/acquired together with their real terminal prices.

Conventions (every implementation MUST follow):
- `entity`: a stable security id that never changes when a ticker is renamed or reused.
    KR: "<6-digit code>" or "<code>#<spell>" when a code relisted after a gap (spell >= 1).
    US: "CIK<10-digit zero-padded>" (share classes of one issuer collapse to one entity, one priced class).
- `membership`: one row per (rebalance date, entity) for the top-N members on that date. A member on date d
    is eligible for every trading day t with d <= t < next rebalance date (hold from the next session is the
    caller's concern). Columns: date (tz-naive month-end session), entity, ticker (display ticker on that date),
    name, rank (1 = largest), rank_value (the ranking variable, local currency).
- `prices`: entity -> frame in the models.PRICE_COLUMNS contract (see data.base.clean_prices), split/rights
    adjusted. KR: KRX base-price adjusted, price-only (no dividends, like production Naver prices). US: Yahoo
    total-return adjusted where the series is validated, else split-adjusted (frame.attrs["adjusted"] = False).
    A delisted entity's frame ENDS at its last traded session (KR includes 정리매매 rows); never forward-filled.
    PIT frames also carry ``value``: the day's own traded value in local currency (KR: KRX close x volume; US:
    split-adjusted close x volume), which later splits / dividends never re-scale (docs/MODELING.md 84.3).
- `meta`: one row per entity: entity, ticker, name, market (KOSPI/KOSDAQ/NYSE/...), first_date, last_date,
    delisted (bool), delist_date, delist_reason, price_source, cik (US), notes.
- `coverage`: plain dict of counts, e.g. {"member_months": n, "member_months_with_prices": n, "by_year": {...},
    "no_price_entities": [...], "rejected_series": n}. Anything not covered MUST be counted here and summarised
    in `notes` (Korean) — never dropped silently.
- PIT fundamentals `panel`: one row per (entity, period_end, available_from); `available_from` is the FIRST
    session on which the values may be used (receipt date + 1 session, or the conservative fallback lag). Values
    are the ORIGINALLY filed ones where a later restatement exists (flag restated_later). Columns (NaN when
    unknown): entity, period_end, period_type ("Q"/"FY"), available_from, currency, revenue_ttm,
    op_income_ttm, ni_parent_ttm, equity_parent, shares_out, source, restated_later.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from quantcio.config import Settings
from quantcio.data.base import PricePanel
from quantcio.data.cache import DiskCache
from quantcio.models import Market

MEMBERSHIP_COLUMNS = ["date", "entity", "ticker", "name", "rank", "rank_value"]
FUNDAMENTAL_COLUMNS = [
    "entity",
    "period_end",
    "period_type",
    "available_from",
    "currency",
    "revenue_ttm",
    "op_income_ttm",
    "ni_parent_ttm",
    "equity_parent",
    "shares_out",
    "source",
    "restated_later",
]


@dataclass
class PITUniverse:
    market: Market
    n: int  # top-N per rebalance date
    membership: pd.DataFrame  # MEMBERSHIP_COLUMNS
    prices: dict[str, pd.DataFrame]  # entity -> PRICE_COLUMNS frame
    benchmarks: dict[str, pd.DataFrame]  # benchmark id -> PRICE_COLUMNS frame
    entity_benchmark: dict[str, str]  # entity -> benchmark id
    meta: pd.DataFrame
    coverage: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def members_on(self, t: date | pd.Timestamp) -> list[str]:
        """Entities eligible on day t (members of the latest rebalance date <= t)."""
        ts = pd.Timestamp(t)
        dates = self.membership["date"]
        past = dates[dates <= ts]
        if past.empty:
            return []
        return self.membership.loc[dates == past.max(), "entity"].tolist()

    def ever_members(self, start: date | None = None, end: date | None = None) -> list[str]:
        m = self.membership
        if start is not None:
            m = m[m["date"] >= pd.Timestamp(start)]
        if end is not None:
            m = m[m["date"] <= pd.Timestamp(end)]
        return sorted(m["entity"].unique().tolist())

    def to_price_panel(self, extra: PricePanel | None = None) -> PricePanel:
        """PricePanel whose `membership` restricts RSMT-Q rows to PIT members (entities with prices only).

        `extra` (e.g. the requested tickers loaded by the production providers) is merged in; its tickers are
        eligible on every date only if they are not already PIT entities (callers decide how to map them).
        """
        prices = {e: px for e, px in self.prices.items() if px is not None and len(px)}
        membership = self.membership[self.membership["entity"].isin(prices)][["date", "entity"]].rename(
            columns={"entity": "ticker"}
        )
        panel = PricePanel(
            market=self.market,
            prices=dict(prices),
            benchmarks=dict(self.benchmarks),
            ticker_benchmark={e: self.entity_benchmark[e] for e in prices if e in self.entity_benchmark},
            failed={},
            membership=membership.reset_index(drop=True),
            notes=list(self.notes),
        )
        if extra is not None:
            for t, px in extra.prices.items():
                panel.prices.setdefault(t, px)
                if t in extra.ticker_benchmark:
                    panel.ticker_benchmark.setdefault(t, extra.ticker_benchmark[t])
            for b, df in extra.benchmarks.items():
                panel.benchmarks.setdefault(b, df)
            panel.failed.update(extra.failed)
        return panel


@dataclass
class PITFundamentals:
    market: Market
    panel: pd.DataFrame  # FUNDAMENTAL_COLUMNS
    coverage: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def asof(self, entity: str, t: date | pd.Timestamp, max_staleness_days: int = 200) -> dict[str, Any] | None:
        """Latest row for `entity` with available_from <= t and period_end within max_staleness_days of t."""
        ts = pd.Timestamp(t)
        rows = self.panel[(self.panel["entity"] == entity) & (self.panel["available_from"] <= ts)]
        if rows.empty:
            return None
        row = rows.sort_values(["available_from", "period_end"]).iloc[-1]
        if (ts - pd.Timestamp(row["period_end"])).days > max_staleness_days:
            return None
        return row.to_dict()


def load_pit_universe(
    market: Market,
    settings: Settings,
    cache: DiskCache,
    n: int,
    start: date,
    end: date | None = None,
) -> PITUniverse:
    """Build (or load from cache) the PIT top-n universe of `market` for rebalance dates in [start, end]."""
    if market is Market.KR:
        from quantcio.data.pit.kr import load_kr_pit_universe

        return load_kr_pit_universe(settings, cache, n, start, end)
    from quantcio.data.pit.us import load_us_pit_universe

    return load_us_pit_universe(settings, cache, n, start, end)


def load_pit_xsh_universe(
    market: Market,
    settings: Settings,
    cache: DiskCache,
    start: date,
    end: date | None = None,
    exclude: Iterable[str] = (),
) -> PITUniverse:
    """Cross-sectional holdout (XSH) universe of `market`: names NEVER used for tuning, month-end dates in [start, end].

    US: S&P MidCap 400 members that were never PIT S&P 500 members since ``backtest.us_start`` (data/pit/us_midcap.py;
    read ``coverage['pit_from']`` - month-ends before it are SURVIVORSHIP-BIASED). `exclude`: extra tickers or entity
    ids dropped on every date (e.g. ``holdout.tuning_names``). KR: not implemented here yet (built from the KR PIT
    top-400 universe elsewhere).
    """
    if market is Market.KR:
        raise NotImplementedError("KR XSH 유니버스는 KR PIT 상위 400 유니버스에서 별도로 구성합니다 (미구현)")
    from quantcio.data.pit import us_midcap

    return us_midcap.load_us_midcap_universe(settings, cache, start, end, exclude)


def load_pit_fundamentals(
    market: Market,
    settings: Settings,
    cache: DiskCache,
    entities: list[str] | None = None,
) -> PITFundamentals:
    """PIT fundamentals panel for `entities` (None = every entity the builder knows)."""
    if market is Market.KR:
        from quantcio.data.pit.kr import load_kr_pit_fundamentals

        return load_kr_pit_fundamentals(settings, cache, entities)
    from quantcio.data.pit.us import load_us_pit_fundamentals

    return load_us_pit_fundamentals(settings, cache, entities)
