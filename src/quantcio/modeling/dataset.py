"""Point-in-time modeling panel of one market: PIT members x (weekly + month-end) grid, features, labels.

Universe: ``data.pit`` - KR top-N by market cap (``modeling.kr_universe_n``, from ``backtest.kr_start``), US S&P 500
members (``backtest.us_universe_n``, from ``backtest.us_start``); membership decided at each month-end rebalance date
d with information <= d and valid for d <= t < next rebalance. Delisted / removed names stay in with their real
terminal prices (KR: last traded close incl. 정리매매; US: ``data.pit.us.terminal_price``).

Grid: the last session of every week plus every month-end rebalance session (``is_me``) plus the latest session
(prediction only). Evaluation always uses month-end rows (label windows of h = 20 sessions barely overlap across
months); weekly rows are optional extra TRAINING rows whose overlap is handled by purging / embargo and
date-block clustering (``modeling.selection``, ``quant.rsmt.RSMTRanker``).

Labels (row at t, horizon h = ``settings.horizon_days`` market sessions, t_end = the h-th session after t):
    r   = log(C[t_end] / C[t])  - exit at the last close <= t_end; KR halted at t_end -> next traded close;
                                  delisted before t_end -> terminal price (never forward-filled);
                                  KR: + sum of log(1 + D / C_prev) over the ex-dividend sessions in (t, t_end]
                                  (``data.pit.kr_dividends``; the price return stays in ``r_price``)
    m   = log(B[t_end] / B[t])  - the entity's benchmark (KOSPI / KOSDAQ / SPY)
    beta = shrink_beta(252-day beta known at t, rsmt.beta_shrink) (1.0 when unknown - counted)
    y_e = (e^r - 1) - beta * (e^m - 1)
                                - SIMPLE beta-adjusted EXCESS return: the stock-selection target (dataset v5,
                                  protocol v4, docs/MODELING.md 79 - a mean of log returns penalises volatile names
                                  by ~sigma^2/2, which made "pick low volatility" look like skill: 78.1)
    y_e_log = r - beta * m      - the log label of dataset v4 and earlier, kept for reference
Per-date derived targets (only rows of that date): y_dm = y_e - mean_date(y_e), y_cls = 1{y_e > median_date}
(the PRIMARY within-date AUC label since protocol v2), y_q = quintile 0..4, y_beat = 1{y_e > 0} (the v1 AUC label,
now secondary).

Nothing is dropped silently: every excluded row is counted by reason in ``ModelingDataset.coverage``.

Inputs loaded here (PIT universe, macro series, KR dividends, PIT fundamentals - 84.5) go through frozen first-seen
vintages (``data.vintage``, docs/MODELING.md 84.2, ``modeling.vintage``): values dated up to the last month-end at
least ``modeling.vintage_settle_days`` old come from their first stored copy, so a source revising its history does
not change the panel; ``coverage["vintage"]`` counts the ignored revisions and holds the frozen-window digest check.
The dataset cache key and the digest identity include the market's loader modules (``LOADER_MODULES``).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantcio.config import Settings
from quantcio.data.cache import DiskCache
from quantcio.modeling import altfeatures as altf
from quantcio.modeling import features as feat
from quantcio.modeling.events import EventCalendar, build_calendar
from quantcio.models import Market

log = logging.getLogger(__name__)

DATASET_VERSION = 5  # v5: simple-return excess label y_e (79); v4: KR total-return labels (52.1); v3: alt data
CACHE_NS = "modeling_dataset"
CACHE_TTL_HOURS = 24.0 * 7
PRIMARY_BENCH = {Market.KR: "KOSPI", Market.US: "SPY"}
MACRO_LAG_DAYS = {Market.KR: 1, Market.US: 0}  # KR: US-hours macro closes are known the next KST morning
REGION = {Market.KR: "kr", Market.US: "us"}
MIN_NAMES_PER_DATE = 20  # a date with fewer labelled names is not usable for within-date metrics (counted)
OTHER_SECTOR = "기타"
KR_DIVIDEND_VINTAGE = "kr_dividends_v1"  # bump when kr_dividends.dividend_yields changes (a new vintage starts)
# w_date is label-derived (NaN until the row's label window ends), so it is hashed with the labels: in the feature
# digest the cutoff month-end's weights flipped once its labels completed (docs/MODELING.md 84.6, KR 2026-11-03)
LABEL_DIGEST_COLS = (
    "r",
    "r_price",
    "div_log",
    "m",
    "beta_s",
    "y_e",
    "y_e_log",
    "y_dm",
    "y_cls",
    "y_q",
    "y_beat",
    "w_date",
)


@dataclass
class ModelingDataset:
    """Modeling panel of one market (see module docstring)."""

    market: Market
    horizon: int
    frame: pd.DataFrame
    xs_cols: list[str]  # x_* and sx_* (cross-sectional model inputs)
    regime_cols: list[str]  # r_* (date-level stress / regime inputs)
    as_of: pd.Timestamp  # latest session (prediction date)
    calendar: pd.DatetimeIndex  # market sessions (primary benchmark)
    month_ends: pd.DatetimeIndex  # PIT rebalance dates
    events: EventCalendar
    dates_frame: pd.DataFrame = field(default_factory=pd.DataFrame)  # per grid date: r_* + m_primary, t_end, is_me
    core_xs_cols: list[str] = field(default_factory=list)  # the v2 (pre-16.3) cross-sectional inputs
    core_regime_cols: list[str] = field(default_factory=list)  # the v2 regime inputs
    alt_raw: list[str] = field(default_factory=list)  # raw v3 feature names (alternative data + macro betas)
    alt_coverage: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    notes: list[dict[str, str]] = field(default_factory=list)  # {"ko": ..., "en": ...}
    snapshot: str = ""
    universe_desc: dict[str, Any] = field(default_factory=dict)
    cost_bps: float = 0.0

    def pos(self, when: pd.Timestamp | np.ndarray) -> np.ndarray | int:
        """Calendar position of the last session <= ``when``."""
        return (
            np.searchsorted(self.calendar.to_numpy(dtype="datetime64[ns]"), np.asarray(when, "datetime64[ns]"), "right")
            - 1
        )

    def session_offset(self, when: pd.Timestamp, k: int) -> pd.Timestamp:
        """The session k positions after (k < 0: before) the last session <= ``when`` (clipped to the calendar)."""
        p = int(self.pos(np.datetime64(pd.Timestamp(when), "ns"))) + int(k)
        p = min(max(p, 0), len(self.calendar) - 1)
        return pd.Timestamp(self.calendar[p])

    @property
    def labelled(self) -> pd.DataFrame:
        return self.frame[self.frame["labelled"]]

    @property
    def model_cols(self) -> list[str]:
        return [*self.xs_cols, *self.regime_cols]


def _note(ko: str, en: str) -> dict[str, str]:
    return {"ko": ko, "en": en}


LOADER_MODULES = {  # the market's input loaders shape the panel too (docs/MODELING.md 84.5)
    "KR": (
        "pit/__init__.py",
        "pit/kr.py",
        "pit/kr_bulk.py",
        "pit/kr_dart.py",
        "pit/kr_delist.py",
        "pit/kr_dividends.py",
        "pit/kr_fund.py",
        "pit/kr_marcap.py",
        "alt/load.py",
        "alt/kr_dart_events.py",
        "alt/kr_jobs.py",
        "alt/kr_nps.py",
    ),
    "US": (
        "pit/__init__.py",
        "pit/us.py",
        "pit/us_calendar.py",
        "pit/us_cik.py",
        "pit/us_fund.py",
        "pit/us_membership.py",
        "pit/us_midcap.py",
        "pit/us_net.py",
        "pit/us_prices.py",
        "us.py",
        "alt/load.py",
        "alt/us_sec.py",
    ),
}


def _code_fingerprint(market: Market | str | None = None) -> str:
    """AST hash of the modules that define the panel (formatting-only edits keep the cache valid): the modeling
    modules, the vintages and (with ``market``) that market's loaders - a loader fix changes the data, so it must
    change the dataset cache key and the frozen-window digest identity too (docs/MODELING.md 84.5)."""
    import ast

    h = hashlib.sha1(str(DATASET_VERSION).encode())
    root = Path(__file__).resolve().parent
    data = root.parent / "data"
    loaders = LOADER_MODULES.get(Market(market).value, ()) if market is not None else ()
    for path in (
        *(root / n for n in ("dataset.py", "features.py", "events.py", "altfeatures.py")),
        data / "vintage.py",  # the frozen inputs shape the panel too (docs 84.2)
        *(data / n for n in loaders),
    ):
        try:
            h.update(ast.dump(ast.parse(path.read_text(encoding="utf-8"))).encode())
        except (OSError, SyntaxError):
            h.update(path.name.encode())
    return h.hexdigest()[:12]


def universe_params(market: Market, settings: Settings) -> tuple[int, date]:
    """(top-N, start date) of the modeling universe (the factor-backtest universe)."""
    bt = settings.backtest
    if Market(market) is Market.KR:  # the modeling universe (docs 80.8); the factor backtest keeps bt.kr_universe_n
        return int(settings.modeling.kr_universe_n), date.fromisoformat(bt.kr_start)
    return int(bt.us_universe_n), date.fromisoformat(bt.us_start)


def excess_labels(r: pd.Series, m: pd.Series, beta: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(y_e, y_e_log) from the log returns r (stock) and m (benchmark): y_e = (e^r - 1) - beta (e^m - 1), the simple
    beta-adjusted excess return (dataset v5, docs/MODELING.md 79); y_e_log = r - beta m (dataset v4 and earlier)."""
    return np.expm1(r) - beta * np.expm1(m), r - beta * m


def cost_bps_for(market: Market, settings: Settings) -> float:
    """Round-trip cost of the modeling book (KR: ``modeling.kr_cost_bps`` for the wider universe, docs 80.8)."""
    return float(settings.modeling.kr_cost_bps if Market(market) is Market.KR else settings.backtest.cost_bps_us)


def grid_dates(
    calendar: pd.DatetimeIndex, month_ends: pd.DatetimeIndex, start: pd.Timestamp, weekly: bool
) -> pd.DatetimeIndex:
    """Month-end rebalance sessions (+ the last session of every week when ``weekly``) + the latest session."""
    cal = calendar[calendar >= start]
    parts = [month_ends[month_ends >= start]]
    if weekly and len(cal):
        iso = cal.isocalendar()
        key = iso["year"].astype(int) * 100 + iso["week"].astype(int)
        last_of_week = pd.Series(cal, index=cal).groupby(key.to_numpy()).max()
        parts.append(pd.DatetimeIndex(last_of_week.to_numpy()))
    if len(cal):
        parts.append(pd.DatetimeIndex([cal[-1]]))
    out = pd.DatetimeIndex(np.unique(np.concatenate([p.to_numpy(dtype="datetime64[ns]") for p in parts if len(p)])))
    return out[out.isin(calendar) | out.isin(month_ends)]


def members_on_grid(membership: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """(date, rebalance, entity, ticker, name, rank) for every grid date: members of the latest rebalance <= date."""
    reb = pd.DatetimeIndex(sorted(membership["date"].unique()))
    pos = np.searchsorted(reb.to_numpy(dtype="datetime64[ns]"), grid.to_numpy(dtype="datetime64[ns]"), "right") - 1
    ok = pos >= 0
    link = pd.DataFrame({"date": grid[ok], "rebalance": reb[pos[ok]]})
    cols = [c for c in ("entity", "ticker", "name", "rank") if c in membership]
    mem = membership[["date", *cols]].rename(columns={"date": "rebalance"}).drop_duplicates(["rebalance", "entity"])
    out = link.merge(mem, on="rebalance", how="inner")
    return out.sort_values(["date", "entity"], kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ labels


def _labels_for_entity(
    rows_dates: np.ndarray,
    t_end: np.ndarray,
    feats: pd.DataFrame,
    bench_close: pd.Series,
    delisted: bool,
    terminal: tuple[pd.Timestamp, float] | None,
    halts: bool,
) -> dict[str, np.ndarray]:
    """Entry/exit prices, r, m and status labels of one entity's grid rows (see module docstring)."""
    idx = feats.index.to_numpy(dtype="datetime64[ns]")
    close = feats["close"].to_numpy(dtype=float)
    traded = feats["traded"].to_numpy(dtype=bool) if halts else np.ones(len(idx), dtype=bool)
    tr_pos = np.flatnonzero(traded)
    n = len(rows_dates)
    r = np.full(n, np.nan)
    status = np.full(n, "ok", dtype=object)
    i0 = np.searchsorted(idx, rows_dates, "right") - 1
    max_age = np.timedelta64(feat.PRICE_MAX_AGE_DAYS, "D")
    bidx = bench_close.index.to_numpy(dtype="datetime64[ns]")
    bval = bench_close.to_numpy(dtype=float)
    m = np.full(n, np.nan)
    for k in range(n):
        te = t_end[k]
        if np.isnat(te):
            status[k] = "unlabelled"
            continue
        i = i0[k]
        if i < 0 or rows_dates[k] - idx[i] > max_age:
            status[k] = "no_entry_price"
            continue
        if halts and idx[i] == rows_dates[k] and not traded[i]:
            status[k] = "halted_at_t"
            continue
        entry = close[i]
        j = int(np.searchsorted(idx, te, "right")) - 1
        exit_px = np.nan
        if j >= len(idx) - 1 and idx[-1] < te:
            # the frame ends before t_end
            if delisted:
                exit_px, status[k] = close[len(idx) - 1], "delisted"
                if halts and not traded[len(idx) - 1] and len(tr_pos):
                    exit_px = close[tr_pos[-1]]
                if terminal is not None:
                    td, tp = terminal
                    if np.datetime64(td, "ns") > idx[-1] and np.datetime64(td, "ns") <= te and np.isfinite(tp):
                        exit_px, status[k] = tp, "delisted_terminal"
            else:
                status[k] = "price_gap_end"
                continue
        else:
            if te - idx[j] > max_age:
                status[k] = "price_gap_end"
                continue
            exit_px = close[j]
            if halts and not traded[j] and j > i:
                q = int(np.searchsorted(tr_pos, j + 1))
                if q < len(tr_pos):
                    exit_px, status[k] = close[tr_pos[q]], "halted_exit_next_trade"
                elif delisted and len(tr_pos):
                    exit_px, status[k] = close[tr_pos[-1]], "delisted"
                else:
                    status[k] = "halted_unresolved"
                    continue
        if not (np.isfinite(entry) and entry > 0 and np.isfinite(exit_px) and exit_px > 0):
            status[k] = "bad_price"
            continue
        r[k] = np.log(exit_px / entry)
        b0 = np.searchsorted(bidx, rows_dates[k], "right") - 1
        b1 = np.searchsorted(bidx, te, "right") - 1
        if b0 >= 0 and b1 >= 0 and bval[b0] > 0 and bval[b1] > 0:
            m[k] = np.log(bval[b1] / bval[b0])
    return {"r": r, "m": m, "status": status}


def add_date_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """y_dm, y_cls, y_q, y_beat and the per-date equal weight ``w_date`` over labelled rows (in place)."""
    lab = frame["labelled"].to_numpy()
    d = frame["date"]
    y = frame["y_e"].where(lab)
    g = y.groupby(d)
    frame["y_dm"] = y - g.transform("mean")
    frame["y_cls"] = (y > g.transform("median")).astype(float).where(lab)
    u = feat.uniform_rank(y, d)
    frame["y_q"] = np.floor(u * 5).clip(0, 4).where(lab)
    frame["y_beat"] = (y > 0).astype(float).where(lab)
    n = y.notna().groupby(d).transform("sum")
    frame["w_date"] = (1.0 / n).where(lab)
    return frame


# ------------------------------------------------------------------------------------------------ build


def _add_kr_dividends(
    rows: pd.DataFrame,
    universe: Any,
    settings: Settings,
    cache: DiskCache,
    dividends: dict[str, pd.Series] | None,
    cov: dict[str, Any],
    notes: list[dict[str, str]],
    vstore: Any = None,
    vint: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """KR total-return labels: r += log(1 + D / C_prev) over the ex-dates in (t, t_end] (docs/MODELING.md 52.1).
    Loaded dividends go through the ``kr_dividends`` vintage when ``vstore`` is given (docs/MODELING.md 84.2)."""
    from quantcio.data.pit import kr_dividends as kdiv

    if dividends is None:
        mem = universe.membership
        mk = (
            mem.sort_values("date").groupby("entity")["market"].last().to_dict()
            if "market" in mem
            else dict.fromkeys(mem["entity"].unique())
        )
        dividends, dcov = kdiv.load_kr_dividends({str(e): m for e, m in mk.items()}, settings, cache)
        with_yahoo = dcov.pop("yahoo_codes", None)
        if vstore is not None:
            from quantcio.data import vintage as vt

            dividends, dinfo = vt.freeze_events(
                vstore, KR_DIVIDEND_VINTAGE, dividends, keys=[str(e) for e in mk], new_ok=with_yahoo
            )
            if vint is not None:
                vint["kr_dividends"] = dinfo
    else:
        dcov = {"codes": len(dividends), "injected": True}
    rows["r_price"] = rows["r"]
    div = np.zeros(len(rows))
    for ent, g in rows.groupby("entity", sort=False):
        y = dividends.get(str(ent))
        if y is None or not len(y):
            continue
        div[rows.index.get_indexer(g.index)] = kdiv.window_dividend_log(
            y, g["date"].to_numpy(dtype="datetime64[ns]"), g["t_end"].to_numpy(dtype="datetime64[ns]")
        )
    div = np.where(np.isfinite(rows["r"].to_numpy(dtype=float)), div, 0.0)
    rows["div_log"] = div
    rows["r"] = rows["r"] + div
    cov["kr_dividends"] = {**dcov, "rows_with_dividend": int((div > 0).sum())}
    if dcov.get("codes") and dcov.get("with_yahoo", dcov["codes"]) < dcov["codes"]:
        miss = dcov["codes"] - dcov.get("with_yahoo", 0)
        notes.append(
            _note(
                f"배당 포함 정답: Yahoo 배당 자료가 없는 {miss}개 종목(주로 상장폐지)은 가격 수익률 그대로",
                f"total-return labels: {miss} codes without Yahoo dividends (mostly delisted) keep price returns",
            )
        )
    return rows


def _load_macro(settings: Settings, cache: DiskCache, notes: list[dict[str, str]]) -> dict[str, pd.Series] | None:
    try:
        from quantcio.macro import load_macro_series

        series = load_macro_series(settings, cache)
    except Exception as exc:  # noqa: BLE001 - macro features become NaN, the note says so
        log.warning("modeling: macro series failed: %s", exc)
        notes.append(
            _note(
                f"매크로 시계열 로드 실패({type(exc).__name__}) — 매크로 레짐 특징 결측",
                f"macro series failed ({type(exc).__name__}) - macro regime features missing",
            )
        )
        return None
    if series.failures:
        notes.append(
            _note(
                f"매크로 시계열 일부 실패: {', '.join(sorted(series.failures))}",
                f"some macro series failed: {', '.join(sorted(series.failures))}",
            )
        )
    return series


def _vintage_tag(market: Market, n: int, start: date) -> str:
    """Vintage namespace tag of a PIT universe: market, size, start and the universe construction version."""
    if market is Market.KR:
        from quantcio.data.pit import kr as pit_mod

        version = pit_mod.UNIVERSE_VERSION
    else:
        from quantcio.data.pit import us as pit_mod

        version = pit_mod.BUILD_VERSION
    return f"{market.value}_n{int(n)}_{start.isoformat()}_u{version}"


def _freeze_macro(vstore: Any, series: Any, vint: dict[str, Any]) -> Any:
    """Loaded macro series through the macro vintages (docs/MODELING.md 84.2); provenance kept."""
    from quantcio.data import vintage as vt
    from quantcio.macro.data import SERIES_BY_KEY, MacroSeries

    chain = [k for k, spec in SERIES_BY_KEY.items() if spec.kind == "price"]
    frozen, infos = vt.freeze_macro(vstore, series, chain)
    vint.update(infos)
    out = MacroSeries(frozen)
    sources = dict(getattr(series, "sources", {}) or {})
    out.sources = {k: sources.get(k, "고정본(vintage)") for k in frozen}
    out.failures = {k: v for k, v in (getattr(series, "failures", {}) or {}).items() if k not in frozen}
    out.notes = list(getattr(series, "notes", []) or [])
    return out


FUND_VALUE_COLS = ("revenue_ttm", "op_income_ttm", "ni_parent_ttm", "equity_parent", "shares_out")


def _freeze_fundamentals(vstore: Any, fundamentals: Any, tag: str, vint: dict[str, Any]) -> Any:
    """The PIT fundamentals panel through the vintage ``fundamentals_<tag>`` (docs/MODELING.md 84.5). Only the
    contract columns (``FUNDAMENTAL_COLUMNS``, all the features read) are frozen; the market extras are dropped."""
    from dataclasses import replace

    from quantcio.data import vintage as vt
    from quantcio.data.pit import FUNDAMENTAL_COLUMNS

    p = fundamentals.panel.reindex(columns=FUNDAMENTAL_COLUMNS).copy()
    p["entity"] = p["entity"].astype(str)
    for c in ("period_end", "available_from"):
        p[c] = pd.to_datetime(p[c]).astype("datetime64[ns]")
    for c in ("period_type", "currency", "source"):
        p[c] = p[c].astype("string")
    for c in FUND_VALUE_COLS:
        p[c] = pd.to_numeric(p[c], errors="coerce").astype(float)
    p["restated_later"] = p["restated_later"].fillna(False).astype(bool)
    panel, vint["fundamentals"] = vt.freeze_panel(vstore, f"fundamentals_{tag}", p, compare=FUND_VALUE_COLS)
    return replace(fundamentals, panel=panel)


def _vintage_notes(vint: dict[str, Any], digest: dict[str, Any] | None) -> list[dict[str, str]]:
    """Korean / English summary of the frozen inputs and the ignored revisions (docs/MODELING.md 84.2)."""
    if not vint:
        return []
    cut = max(i.cutoff for i in vint.values())
    names = {
        "prices": ("가격", "prices"),
        "benchmarks": ("지수", "benchmarks"),
        "membership": ("구성 행", "membership rows"),
        "macro_chain": ("거시(가격형)", "macro (prices)"),
        "macro_level": ("거시(금리·수준)", "macro (yields, levels)"),
        "kr_dividends": ("배당", "dividends"),
        "fundamentals": ("재무", "fundamentals"),
    }
    created = [names.get(k, (k, k)) for k, i in vint.items() if i.created or i.restarted]
    parts_ko, parts_en = [], []
    for k, i in vint.items():
        ko, en = names.get(k, (k, k))
        if i.revised or i.added or i.dropped:
            mr = "" if not i.revised else (f", 최대 {i.max_rel:.1e}" if np.isfinite(i.max_rel) else ", 한쪽 결측")
            mr_en = (
                "" if not i.revised else (f", max {i.max_rel:.1e}" if np.isfinite(i.max_rel) else ", one side missing")
            )
            parts_ko.append(f"{ko} 수정 {i.revised:,}·추가 {i.added:,}·누락 {i.dropped:,}{mr}")
            parts_en.append(f"{en} revised {i.revised:,} / added {i.added:,} / missing {i.dropped:,}{mr_en}")
    ignored_ko = "; ".join(parts_ko) if parts_ko else "없음"
    ignored_en = "; ".join(parts_en) if parts_en else "none"
    status = (digest or {}).get("status")
    dg_ko = {"first": "첫 기록", "match": "일치", "mismatch": "불일치 " + ", ".join((digest or {}).get("mismatch", []))}
    dg_en = {
        "first": "first record",
        "match": "match",
        "mismatch": "MISMATCH " + ", ".join((digest or {}).get("mismatch", [])),
    }
    out = [
        _note(
            f"데이터 고정본: {cut}까지의 값은 처음 저장한 값을 씀(가격·지수·구성·거시·배당·재무)"
            + (f" — 새로 만듦: {', '.join(c[0] for c in created)}" if created else "")
            + f". 무시한 원천 수정: {ignored_ko}."
            + (f" 고정 구간 지문: {dg_ko.get(status, status)}." if status else ""),
            f"data vintage: values dated <= {cut} come from the first stored copy (prices, benchmarks, membership, "
            f"macro, dividends, fundamentals)"
            + (f" - created now: {', '.join(c[1] for c in created)}" if created else "")
            + f". Ignored source revisions: {ignored_en}."
            + (f" Frozen-window digest: {dg_en.get(status, status)}." if status else ""),
        )
    ]
    return out


def _terminal_map(meta: pd.DataFrame | None) -> dict[str, tuple[pd.Timestamp, float]]:
    out: dict[str, tuple[pd.Timestamp, float]] = {}
    if meta is None or meta.empty or "terminal_date" not in meta:
        return out
    for r in meta.itertuples(index=False):
        td, tp = getattr(r, "terminal_date", None), getattr(r, "terminal_price", None)
        basis = str(getattr(r, "terminal_basis", "frame") or "frame")
        if td is None or pd.isna(td) or tp is None or pd.isna(tp) or basis != "frame":
            continue
        out[str(r.entity)] = (pd.Timestamp(td), float(tp))
    return out


def build_dataset(
    market: Market,
    settings: Settings,
    cache: DiskCache,
    *,
    weekly: bool = True,
    use_cache: bool = True,
    universe: Any = None,
    fundamentals: Any = None,
    factor_panel: pd.DataFrame | None = None,
    macro_series: dict[str, pd.Series] | None = None,
    alt: Any = None,
    dividends: dict[str, pd.Series] | None = None,
    progress: Callable[[str], None] | None = None,
) -> ModelingDataset:
    """Build (or load from cache) the modeling panel of ``market``.

    ``universe`` / ``fundamentals`` / ``factor_panel`` / ``macro_series`` / ``alt`` (``data.alt.AltData``) are
    injection points (tests, callers that already loaded them). ``factor_panel`` defaults to
    ``backtest.compute_factor_panel`` at the rebalance dates (sector, PIT market cap); ``alt`` to
    ``data.alt.load.load_alt_data`` for the universe's entities (settings ``alt``; disabled -> no v3 alt features).
    ``dividends`` (KR: code -> ex-date dividend yields) defaults to ``data.pit.kr_dividends.load_kr_dividends``.
    """
    market = Market(market)
    say = progress or (lambda _m: None)
    t0 = time.monotonic()
    n, start = universe_params(market, settings)
    notes: list[dict[str, str]] = []
    own_universe = universe is None  # only inputs loaded here go through the vintages (docs/MODELING.md 84.2)
    own_macro = macro_series is None and settings.macro.enabled
    own_fund = fundamentals is None
    if universe is None:
        from quantcio.data.pit import load_pit_universe

        say("PIT universe")
        universe = load_pit_universe(market, settings, cache, n, start)
    ents = sorted(universe.membership["entity"].unique())
    if fundamentals is None:
        from quantcio.data.pit import load_pit_fundamentals

        say("PIT fundamentals")
        fundamentals = load_pit_fundamentals(market, settings, cache, ents)
    if macro_series is None and settings.macro.enabled:
        say("macro series")
        macro_series = _load_macro(settings, cache, notes)
    if alt is None and settings.alt.enabled:
        from quantcio.data.alt.load import load_alt_data

        say("alternative data")
        mem_names = {}
        if "name" in universe.membership:
            mem_names = universe.membership.groupby("entity")["name"].agg(lambda x: sorted(set(map(str, x.dropna()))))
        alt = load_alt_data(market, settings, ents, date(start.year - 1, 1, 1), names=dict(mem_names), progress=say)
    vstore: Any = None
    vint: dict[str, Any] = {}
    if settings.modeling.vintage:
        from quantcio.data import vintage as vt

        vstore = vt.VintageStore(
            settings.cache_path / "vintage", settle_days=int(settings.modeling.vintage_settle_days)
        )
        if own_universe:
            say(f"data vintage (universe, cutoff {vstore.cut.date()})")
            universe, infos = vt.freeze_universe(vstore, universe, tag=_vintage_tag(market, n, start))
            vint.update(infos)
        if own_fund:
            say("data vintage (fundamentals)")
            fundamentals = _freeze_fundamentals(vstore, fundamentals, _vintage_tag(market, n, start), vint)
        if own_macro and macro_series is not None:
            say("data vintage (macro)")
            macro_series = _freeze_macro(vstore, macro_series, vint)
    from quantcio.data.alt.load import nps_included

    with_nps = nps_included(alt)
    alt_names = altf.alt_feature_names(market.value, with_nps) if alt is not None else []
    beta_names = list(altf.MACRO_BETA_FEATURES) if settings.alt.enabled and settings.alt.macro_betas else []
    last_bar = max(
        (str(px.index[-1].date()) for px in universe.prices.values() if px is not None and len(px)), default=""
    )
    macro_sig = sorted((k, str(v.index[-1].date()), len(v)) for k, v in (macro_series or {}).items())
    alt_sig = (
        None
        if alt is None
        else [
            len(alt.events),
            len(alt.measures),
            str(alt.events["published"].max()) if len(alt.events) else "",
            sorted((k, len(v)) for k, v in alt.covered.items()),
            with_nps,
        ]
    )
    key = DiskCache.key(
        CACHE_NS,
        DATASET_VERSION,
        _code_fingerprint(market),
        market.value,
        n,
        start.isoformat(),
        bool(weekly),
        int(settings.horizon_days),
        float(settings.rsmt.beta_shrink),
        int(settings.pit.fundamentals.max_staleness_days),
        len(universe.membership),
        str(universe.membership["date"].max()),
        len(universe.prices),
        last_bar,
        len(fundamentals.panel),
        macro_sig,
        settings.backtest.model_dump(),
        alt_sig,
        beta_names,
        sorted((k, i.cutoff, i.keys) for k, i in vint.items()),
    )
    if use_cache and factor_panel is None:
        hit = cache.get(CACHE_NS, key, CACHE_TTL_HOURS)
        if isinstance(hit, ModelingDataset):
            say("dataset cache hit")
            return hit
    h = int(settings.horizon_days)
    bench_id = PRIMARY_BENCH[market]
    bench = universe.benchmarks.get(bench_id)
    if bench is None or bench.empty:
        raise ValueError(f"modeling {market.value}: primary benchmark {bench_id} missing")
    calendar = pd.DatetimeIndex(bench.index).normalize()
    month_ends = pd.DatetimeIndex(sorted(pd.to_datetime(universe.membership["date"].unique())))
    start_ts = max(pd.Timestamp(start), month_ends[0])
    grid = grid_dates(calendar, month_ends, start_ts, weekly)
    rows = members_on_grid(universe.membership, grid)
    rows["is_me"] = rows["date"].isin(month_ends)
    cov: dict[str, Any] = {
        "grid_dates": len(grid),
        "month_ends": int((month_ends >= start_ts).sum()),
        "member_rows": len(rows),
    }

    # ---- factor panel (sector, PIT market cap at the rebalance dates)
    if factor_panel is None:
        from quantcio.backtest import compute_factor_panel

        say("factor panel (sector, market cap)")
        reb = [d for d in month_ends if d >= start_ts]
        factor_panel = compute_factor_panel(universe, fundamentals, settings, reb, cache=cache)
    fp = factor_panel[["date", "entity", "sector", "sector_group", "market_cap", "close"]].rename(
        columns={"date": "rebalance", "close": "close_reb", "market_cap": "mcap_reb"}
    )
    rows = rows.merge(fp, on=["rebalance", "entity"], how="left")
    rows["sector"] = rows["sector"].fillna(OTHER_SECTOR)
    rows["sector_group"] = rows["sector_group"].fillna(OTHER_SECTOR)

    # ---- per-entity price features + labels
    say(f"price features + labels ({rows['entity'].nunique()} entities)")
    pos_grid = (
        np.searchsorted(
            calendar.to_numpy(dtype="datetime64[ns]"), rows["date"].to_numpy(dtype="datetime64[ns]"), "right"
        )
        - 1
    )
    te_pos = pos_grid + h
    t_end = np.where(
        te_pos < len(calendar),
        calendar.to_numpy(dtype="datetime64[ns]")[np.clip(te_pos, 0, len(calendar) - 1)],
        np.datetime64("NaT", "ns"),
    )
    rows["t_end"] = pd.DatetimeIndex(t_end)
    meta = universe.meta
    delisted = set()
    if meta is not None and len(meta) and "delisted" in meta:
        delisted = set(meta.loc[meta["delisted"].fillna(False).astype(bool), "entity"].astype(str))
    terminal = _terminal_map(meta)
    halts = market is Market.KR
    price_cols = [*feat.PRICE_FEATURES, "adv20", "close", *beta_names]
    feat_block = np.full((len(rows), len(price_cols)), np.nan)
    r_arr = np.full(len(rows), np.nan)
    m_arr = np.full(len(rows), np.nan)
    status = np.full(len(rows), "no_price", dtype=object)
    fresh_all = np.zeros(len(rows), dtype=bool)
    for ent, g in rows.groupby("entity", sort=False):
        ix = g.index.to_numpy()
        px = universe.prices.get(ent)
        if px is None or px.empty:
            continue
        b = universe.benchmarks.get(universe.entity_benchmark.get(ent, bench_id))
        if b is None or b.empty:
            b = bench
        feats = feat.entity_price_features(px, b)
        if beta_names:
            fac = altf.factor_changes(macro_series, market.value, pd.DatetimeIndex(feats.index), MACRO_LAG_DAYS[market])
            mb = altf.macro_betas(feats["close"], b["close"].astype(float), fac)
            for c in beta_names:
                feats[c] = mb[c].to_numpy()
        d = g["date"].to_numpy(dtype="datetime64[ns]")
        p, fresh = feat.sample_positions(pd.DatetimeIndex(feats.index), d)
        vals = feats[price_cols].to_numpy(dtype=float)[np.clip(p, 0, None)]
        vals[~fresh] = np.nan
        feat_block[ix] = vals
        fresh_all[ix] = fresh
        lab = _labels_for_entity(
            d,
            g["t_end"].to_numpy(dtype="datetime64[ns]"),
            feats,
            b["close"].astype(float),
            ent in delisted,
            terminal.get(ent),
            halts,
        )
        r_arr[ix] = lab["r"]
        m_arr[ix] = lab["m"]
        st = lab["status"]
        st[~fresh & (st != "unlabelled")] = "no_price_t"
        status[ix] = st
    for j, c in enumerate(price_cols):
        rows[c] = feat_block[:, j]
    rows["r"] = r_arr
    rows["m"] = m_arr
    rows["label_status"] = status
    cov["no_price_rows"] = int((~fresh_all).sum())
    cov["label_status"] = {str(k): int(v) for k, v in pd.Series(status).value_counts().items()}
    # rows without a usable price at t cannot be features or labels: dropped here (counted)
    rows = rows[fresh_all].reset_index(drop=True)

    # ---- market cap, size, fundamentals
    say("fundamentals")
    ratio = rows["close"] / rows["close_reb"]
    rows["mcap"] = (rows["mcap_reb"] * ratio).where(rows["mcap_reb"] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rows["log_mcap"] = np.log(rows["mcap"])
        rows["turnover"] = np.log(rows["adv20"] / rows["mcap"])
    fpanel = feat.add_prev_year(fundamentals.panel)
    fund = feat.fundamental_features(rows, fpanel, rows["mcap"], int(settings.pit.fundamentals.max_staleness_days))
    for c in feat.FUND_FEATURES:
        rows[c] = fund[c]
    cov["market_cap_missing"] = int(rows["mcap"].isna().sum())
    cov["fundamentals_missing"] = {c: int(rows[c].isna().sum()) for c in feat.FUND_FEATURES}

    # ---- features v3: alternative data (docs/MODELING.md 16.3; published < t)
    if alt_names:
        say(f"alternative-data features ({len(alt_names)})")
        af = altf.alt_features(rows[["date", "entity", "mcap", "close"]], alt, market.value, with_nps)
        for c in alt_names:
            rows[c] = af[c].to_numpy()
        for note in alt.notes:  # not `n`: that is the universe size used below
            notes.append(_note(f"[대안 데이터] {note}", f"[alt data] {note}"))

    # ---- sector momentum
    real = rows["sector"] != OTHER_SECTOR
    rows["sector_mom_3m"] = (
        rows["mom_3m"].where(real).groupby([rows["date"], rows["sector"]]).transform("mean").where(real)
    )
    cov["sector_other_rows"] = int((~real).sum())

    # ---- labels
    if market is Market.KR:
        say("KR cash dividends (total-return labels)")
        rows = _add_kr_dividends(rows, universe, settings, cache, dividends, cov, notes, vstore, vint)
    beta_s = (1.0 - settings.rsmt.beta_shrink) * rows["beta"] + settings.rsmt.beta_shrink
    cov["beta_default_rows"] = int(beta_s.isna().sum())
    rows["beta_s"] = beta_s.fillna(1.0)
    rows["y_e"], rows["y_e_log"] = excess_labels(rows["r"], rows["m"], rows["beta_s"])
    rows["labelled"] = rows["y_e"].notna() & rows["t_end"].notna()
    n_lab = rows[rows["labelled"]].groupby("date").size()
    thin = n_lab[n_lab < MIN_NAMES_PER_DATE].index
    if len(thin):
        rows.loc[rows["date"].isin(thin), "labelled"] = False
        cov["thin_dates_unlabelled"] = [str(d.date()) for d in thin]
    rows = add_date_targets(rows)

    # ---- coverage of every raw cross-sectional feature (a broken, e.g. all-NaN, feature is loud)
    raw_xs = [*feat.XS_FEATURES, *beta_names, *alt_names]
    cov["feature_missing_share"] = {c: round(float(rows[c].isna().mean()), 4) for c in raw_xs}
    dead = [c for c in raw_xs if rows[c].notna().sum() == 0]
    if dead:
        notes.append(_note(f"특징 전부 결측: {', '.join(dead)}", f"features entirely missing: {', '.join(dead)}"))

    # ---- normalisation
    say("normalisation")
    xs_cols: list[str] = []
    for c in feat.XS_FEATURES:
        rows[feat.x_col(c)] = feat.rank_gauss(rows[c], rows["date"])
        xs_cols.append(feat.x_col(c))
    for c in feat.SECTOR_RELATIVE:
        rows[feat.sx_col(c)] = feat.rank_gauss(rows[c], [rows["date"], rows["sector_group"]])
        xs_cols.append(feat.sx_col(c))
    core_xs_cols = list(xs_cols)
    for c in [*beta_names, *alt_names]:
        rows[feat.x_col(c)] = feat.rank_gauss(rows[c], rows["date"])
        xs_cols.append(feat.x_col(c))

    # ---- regime / stress (date level)
    say("regime features")
    dates = pd.DatetimeIndex(sorted(rows["date"].unique()))
    reg = feat.benchmark_regime(bench, dates)
    reg["xs_disp"] = rows.groupby("date")["ret_1m"].std().reindex(dates)
    reg["breadth"] = (
        (rows["ma200_dist"] > 0).where(rows["ma200_dist"].notna()).groupby(rows["date"]).mean().reindex(dates)
    )
    mac, mnotes = feat.macro_features(macro_series, settings, dates, REGION[market], MACRO_LAG_DAYS[market])
    for c in feat.MACRO_FEATURES:
        reg[c] = mac[c].to_numpy()
    for mn in mnotes:
        notes.append(_note(f"매크로: {mn}", f"macro: {mn}"))
    core_regime = [feat.r_col(c) for c in reg.columns]
    if settings.alt.enabled:
        reg["vix_term"] = feat.vix_term(macro_series, dates, MACRO_LAG_DAYS[market]).to_numpy()
    reg = reg.add_prefix("r_")
    regime_cols = list(reg.columns)
    rows = rows.merge(reg, left_on="date", right_index=True, how="left")
    # primary-benchmark h-session log return per date (the market-direction label)
    cal64 = calendar.to_numpy(dtype="datetime64[ns]")
    dpos = np.searchsorted(cal64, dates.to_numpy(dtype="datetime64[ns]"), "right") - 1
    bclose = feat.align_close(calendar, bench).to_numpy(dtype=float)
    epos = dpos + h
    ok_e = epos < len(calendar)
    m_primary = np.full(len(dates), np.nan)
    m_primary[ok_e] = np.log(bclose[epos[ok_e]] / bclose[dpos[ok_e]])
    dates_frame = reg.copy()
    dates_frame["m_primary"] = m_primary
    dates_frame["t_end"] = pd.DatetimeIndex(
        np.where(ok_e, cal64[np.clip(epos, 0, len(cal64) - 1)], np.datetime64("NaT", "ns"))
    )
    dates_frame["is_me"] = dates_frame.index.isin(month_ends)

    # ---- events (robustness flags; never model inputs)
    events = build_calendar(market, universe.benchmarks)
    rows["event"] = events.overlap_mask(rows["date"], rows["t_end"].fillna(rows["date"]))
    rows = rows.sort_values(["date", "entity"], kind="stable").reset_index(drop=True)

    lab = rows[rows["labelled"]]
    cov.update(
        {
            "rows": len(rows),
            "labelled_rows": len(lab),
            "labelled_me_rows": int(lab["is_me"].sum()),
            "labelled_me_dates": int(lab.loc[lab["is_me"], "date"].nunique()),
            "entities": int(rows["entity"].nunique()),
            "first_date": str(rows["date"].min().date()),
            "last_date": str(rows["date"].max().date()),
            "seconds": round(time.monotonic() - t0, 1),
        }
    )
    ls = cov["label_status"]
    status_txt = ", ".join(f"{k} {v}" for k, v in sorted(ls.items(), key=lambda kv: -kv[1]))
    notes.append(
        _note(
            f"PIT {market.value} 패널: 그리드 {len(grid)}일(월말 {cov['month_ends']}), "
            f"회원 행 {cov['member_rows']:,} → 가격 있는 행 {len(rows):,}"
            f"(가격 없음·지연 {cov['no_price_rows']:,} 제외), 레이블 {len(lab):,}행. 레이블 상태: {status_txt}",
            f"PIT {market.value} panel: {len(grid)} grid dates ({cov['month_ends']} month-ends), "
            f"{cov['member_rows']:,} member rows -> {len(rows):,} with a price "
            f"({cov['no_price_rows']:,} without/stale dropped), {len(lab):,} labelled. Label status: {status_txt}",
        )
    )
    if cov["beta_default_rows"]:
        notes.append(
            _note(
                f"베타 이력 부족 {cov['beta_default_rows']}행은 β=1로 초과수익 계산",
                f"{cov['beta_default_rows']} rows without a beta history use beta = 1 in the excess target",
            )
        )
    notes.extend(_note(f"[데이터] {x}", f"[data] {x}") for x in universe.notes[:4])
    digest = None
    if vstore is not None and "prices" in vint:
        from quantcio.data import vintage as vt

        say("frozen-window digest")
        identity = DiskCache.key(
            "vintage_digest",
            DATASET_VERSION,
            _code_fingerprint(market),
            market.value,
            n,
            start.isoformat(),
            bool(weekly),
            int(settings.horizon_days),
            float(settings.rsmt.beta_shrink),
            int(settings.pit.fundamentals.max_staleness_days),
            settings.backtest.model_dump(),
            settings.macro.model_dump(),
            settings.alt.model_dump(),
            settings.modeling.model_dump(),
            beta_names,
            alt_names,
            with_nps,
        )
        digest = vt.digest_check(
            vstore,
            identity,
            rows,
            features=[*xs_cols, *regime_cols],
            labels=LABEL_DIGEST_COLS,
            cut=pd.Timestamp(vint["prices"].cutoff),
        )
        if digest["status"] == "mismatch":
            log.warning("modeling %s: frozen-window digest mismatch %s", market.value, digest["mismatch"])
    if vint:
        cov["vintage"] = {k: i.as_dict() for k, i in vint.items()}
        cov["vintage"]["digest"] = digest
        notes.extend(_vintage_notes(vint, digest))
    ds = ModelingDataset(
        market=market,
        horizon=h,
        frame=rows,
        xs_cols=xs_cols,
        regime_cols=regime_cols,
        as_of=pd.Timestamp(calendar[-1]),
        calendar=calendar,
        month_ends=month_ends[month_ends >= start_ts],
        events=events,
        dates_frame=dates_frame,
        core_xs_cols=core_xs_cols,
        core_regime_cols=core_regime,
        alt_raw=[*beta_names, *alt_names],
        alt_coverage=(alt.coverage if alt is not None else {}),
        coverage=cov,
        notes=notes,
        snapshot=key,
        universe_desc={
            "n": n,
            "start": start.isoformat(),
            "entities": len(ents),
            "description": (
                f"KOSPI+KOSDAQ 시가총액 상위 {n}(시점별)" if market is Market.KR else "S&P 500 시점별 구성종목"
            ),
            "description_en": (
                f"KOSPI+KOSDAQ top {n} by market cap (PIT)" if market is Market.KR else "S&P 500 PIT members"
            ),
        },
        cost_bps=cost_bps_for(market, settings),
    )
    if use_cache:
        try:
            cache.set(CACHE_NS, key, ds)
        except Exception as exc:  # noqa: BLE001 - only costs a rebuild next time
            log.warning("modeling dataset cache write failed: %s", exc)
    log.info("modeling dataset %s: %s", market.value, {k: v for k, v in cov.items() if not isinstance(v, dict | list)})
    return ds


__all__ = [
    "CACHE_NS",
    "PRIMARY_BENCH",
    "ModelingDataset",
    "add_date_targets",
    "build_dataset",
    "cost_bps_for",
    "grid_dates",
    "members_on_grid",
    "universe_params",
]
