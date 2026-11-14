"""PIT factor backtest engine + serving (docs/BACKTEST_MACRO.md sections 2-3).

``run_factor_backtest`` loads the PIT universe and fundamentals of a market, builds the month-end factor panel,
evaluates each factor in two variants (``raw`` = ranked across the universe, ``sector_neutral`` = ranked within
sector) with monthly Spearman ICs and quintile portfolios before / after costs, applies Benjamini-Hochberg over all
factor x variant tests, fits the pre-declared walk-forward multi-factor candidates, selects one (docs/BACKTEST_MACRO.md
section 3.1) and scores the latest cross-section with the selected candidate's weights for serving. The result is
cached keyed by the PIT data snapshot + ``settings.digest()`` + the candidate set / selection version (and a
per-market pointer to the latest result, used by ``score_tickers`` so that report runs never recompute).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from quantcio.backtest.candidates import CANDIDATES, SELECTION_VERSION
from quantcio.backtest.factors import (
    EXPECTED_SIGN,
    FACTOR_LABELS,
    FACTORS,
    RAW_COLUMN,
    UNIVERSE_RANKED,
    compute_factor_panel,
    panel_coverage,
    rank_input,
)
from quantcio.backtest.model import (
    ModelResult,
    _jsonable,
    cross_sectional_ic,
    ic_summary,
    pillar_side_test,
    portfolio_summary,
    quantile_portfolios,
    walk_forward_model,
)
from quantcio.backtest.sectors import OTHER, sector_coverage
from quantcio.config import Settings
from quantcio.data.cache import DiskCache
from quantcio.data.pit import PITFundamentals, PITUniverse
from quantcio.i18n import tr
from quantcio.models import FactorModelScore, Market

log = logging.getLogger(__name__)

BACKTEST_VERSION = 2  # 2: candidate comparison + selection (the served model is the selected candidate)
BH_Q = 0.10  # Benjamini-Hochberg level over factor x variant tests (docs/BACKTEST_MACRO.md)
NAMESPACE = "factor_backtest"
LATEST_NAMESPACE = "factor_backtest_latest"
LIVE_MIN_SHARE = 0.8  # the serving snapshot uses the latest session on which >= 80% of members traded
SKILL_SHRINK = 0.35
EXPOSURE_KEYS = ["per", "pbr", "ep", "bp", "volume_trend", "mom_3m", "mom_6m", "sector_mom_3m"]
VARIANT_LABELS = {"raw": "전체 순위", "sector_neutral": "섹터 중립"}
MARKET_VAR_SESSIONS = 60  # rsmt_tree split variables: market return / realised vol over 60 sessions
PRIMARY_BENCHMARK = {Market.KR: "KOSPI", Market.US: "SPY"}


# ------------------------------------------------------------------------------------------------ result


@dataclass
class BacktestResult:
    market: str
    as_of: str  # last rebalance date of the panel
    period: dict[str, Any]
    n_months: int
    universe: dict[str, Any]
    factor_table: list[dict[str, Any]]
    model: dict[str, Any]
    curves: dict[str, Any]
    coverage: dict[str, Any]
    notes: list[str]
    settings_digest: str = ""
    data_snapshot: str = ""
    created_at: str = ""
    cost_bps: float = 0.0
    bh_q: float = BH_Q
    live: dict[str, Any] = field(default_factory=dict)  # serving snapshot summary (as_of, n, weights)
    # not serialised (kept in the pickle cache for serving / diagnostics)
    latest: pd.DataFrame | None = None
    panel: pd.DataFrame | None = None
    model_result: ModelResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "market": self.market,
                "as_of": self.as_of,
                "period": self.period,
                "n_months": self.n_months,
                "universe": self.universe,
                "factor_table": self.factor_table,
                "model": self.model,
                "curves": self.curves,
                "coverage": self.coverage,
                "notes": self.notes,
                "settings_digest": self.settings_digest,
                "data_snapshot": self.data_snapshot,
                "created_at": self.created_at,
                "cost_bps": self.cost_bps,
                "bh_q": self.bh_q,
                "live": self.live,
            }
        )


# ------------------------------------------------------------------------------------------------ helpers


def cost_bps_for(market: Market, settings: Settings) -> float:
    return float(settings.backtest.cost_bps_kr if Market(market) is Market.KR else settings.backtest.cost_bps_us)


def universe_n_for(market: Market, settings: Settings) -> int:
    return int(settings.backtest.kr_universe_n if Market(market) is Market.KR else settings.backtest.us_universe_n)


def start_for(market: Market, settings: Settings) -> date:
    return date.fromisoformat(settings.backtest.kr_start if Market(market) is Market.KR else settings.backtest.us_start)


def benjamini_hochberg(pvals: list[float], q: float) -> list[bool]:
    """Step-up BH rejections (NaN p-values are never rejected and do not count as tests)."""
    idx = [i for i, p in enumerate(pvals) if p is not None and np.isfinite(p)]
    m = len(idx)
    out = [False] * len(pvals)
    if m == 0:
        return out
    order = sorted(idx, key=lambda i: pvals[i])
    k_max = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= q * rank / m:
            k_max = rank
    for rank, i in enumerate(order, start=1):
        if rank <= k_max:
            out[i] = True
    return out


def _say(progress: Callable[[str], None] | None, msg: str) -> None:
    log.info("factor backtest: %s", msg)
    if progress is not None:
        try:
            progress(msg)
        except Exception:  # noqa: BLE001 - a UI callback must never break the backtest
            log.debug("progress callback failed", exc_info=True)


def _snapshot_id(universe: PITUniverse) -> str:
    mem = universe.membership
    last_bar = max((px.index[-1] for px in universe.prices.values() if px is not None and len(px)), default=None)
    cov = universe.coverage or {}
    return DiskCache.key(
        "pit_snapshot",
        str(universe.market),
        int(universe.n),
        str(mem["date"].min()) if len(mem) else None,
        str(mem["date"].max()) if len(mem) else None,
        len(mem),
        len(universe.prices),
        str(last_bar),
        cov.get("marcap_files"),
        cov.get("rebalance"),
    )


def _backtest_key(market: Market, settings: Settings, snapshot: str, regime: pd.DataFrame | None) -> str:
    reg = None
    if regime is not None and len(regime):
        reg = [len(regime), str(pd.to_datetime(regime["date"]).max()), float(regime["score"].sum())]
    return DiskCache.key(
        "factor_backtest",
        BACKTEST_VERSION,
        SELECTION_VERSION,
        list(CANDIDATES),
        str(market),
        snapshot,
        settings.digest(),
        reg,
    )


def _latest_key(market: Market, settings: Settings) -> str:
    return DiskCache.key(
        "factor_backtest_latest",
        BACKTEST_VERSION,
        SELECTION_VERSION,
        list(CANDIDATES),
        str(market),
        settings.backtest.model_dump(),
    )


def market_state_vars(universe: PITUniverse, dates: list[pd.Timestamp]) -> pd.DataFrame | None:
    """``mkt_ret_60d`` (close / close 60 sessions earlier - 1) and ``mkt_vol_60d`` (annualised std of the last 60
    daily log returns) of the market benchmark (KR KOSPI, US SPY) at each formation date, from closes <= t only
    (a close older than 7 days -> NaN). None when the universe has no benchmark."""
    bm = universe.benchmarks or {}
    frame = bm.get(PRIMARY_BENCHMARK.get(universe.market, ""))
    if frame is None or frame.empty:
        frame = next((f for f in bm.values() if f is not None and len(f)), None)
    if frame is None or frame.empty or "close" not in frame:
        return None
    close = frame["close"].astype(float)
    close = close[np.isfinite(close) & (close > 0)].sort_index()
    if close.empty:
        return None
    w = MARKET_VAR_SESSIONS
    ret = close / close.shift(w) - 1.0
    vol = np.log(close).diff().rolling(w, min_periods=w).std() * np.sqrt(252.0)
    idx = pd.DatetimeIndex(dates)
    pos = np.searchsorted(close.index.values, idx.values, side="right") - 1
    ok = pos >= 0
    ok[ok] &= (idx.values[ok] - close.index.values[pos[ok]]) <= np.timedelta64(7, "D")
    out = pd.DataFrame(np.nan, index=idx, columns=["mkt_ret_60d", "mkt_vol_60d"])
    out.loc[ok, "mkt_ret_60d"] = ret.to_numpy()[pos[ok]]
    out.loc[ok, "mkt_vol_60d"] = vol.to_numpy()[pos[ok]]
    return out


def _curve(series: pd.Series, ends: pd.Series) -> list[dict[str, Any]]:
    wealth = (1.0 + series.fillna(0.0)).cumprod()
    return [
        {"date": str(pd.Timestamp(d).date()), "value": round(float(v), 6)}
        for d, v in zip(ends, wealth, strict=False)
        if d is not None and not pd.isna(d)
    ]


# ------------------------------------------------------------------------------------------------ factor stats


def factor_table(
    panel: pd.DataFrame, q: int, cost_bps: float, bh_q: float = BH_Q
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows = factor x variant (raw / sector_neutral; sector momentum only raw) with IC and portfolio statistics,
    BH-adjusted significance; plus monthly cumulative L/S and long-only (net) curves."""
    rows: list[dict[str, Any]] = []
    curves: dict[str, Any] = {}
    work = panel.copy()
    for f in FACTORS:
        # the raw variant ranks the raw value across the universe; losses / negative book rank last
        work[f"rawrank_{f}"] = rank_input(work, f)
    for f in FACTORS:
        variants = [("raw", f"rawrank_{f}")]
        if f not in UNIVERSE_RANKED:
            variants.append(("sector_neutral", f"z_{f}"))
        for variant, col in variants:
            ic = cross_sectional_ic(work, col)
            ics = ic_summary(ic)
            qp = quantile_portfolios(work, col, q, cost_bps)
            ps = portfolio_summary(qp, q)
            sign = EXPECTED_SIGN[f]
            row = {
                "factor": f,
                "label": FACTOR_LABELS[f],
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "expected_sign": sign,
                **ics,
                "direction_ok": None
                if sign == 0 or not np.isfinite(ics["ic_mean"])
                else bool(np.sign(ics["ic_mean"]) == sign),
                "ls": ps.get("ls"),
                "ls_net": ps.get("ls_net"),
                "lo": ps.get("lo"),
                "lo_net": ps.get("lo_net"),
                "turnover_long": ps.get("turnover_long"),
                "turnover_short": ps.get("turnover_short"),
                "quantile_mean_monthly": ps.get("quantile_mean_monthly"),
                "months_with_portfolio": len(qp),
            }
            rows.append(row)
            if len(qp):
                ends = qp["end"]
                curves[f"{f}|{variant}|ls_net"] = _curve(qp["ls_net"], ends)
                curves[f"{f}|{variant}|lo_net"] = _curve(qp["lo_net"], ends)
    rej = benjamini_hochberg([r["ic_p"] for r in rows], bh_q)
    for r, flag in zip(rows, rej, strict=True):
        r["bh_significant"] = bool(flag)
        r["bh_tests"] = sum(1 for x in rows if x["ic_p"] is not None and np.isfinite(x["ic_p"]))
    return rows, curves


# ------------------------------------------------------------------------------------------------ live snapshot


def _live_date(universe: PITUniverse, members: list[str], t_last: pd.Timestamp) -> pd.Timestamp:
    """Latest session after ``t_last`` on which >= 80% of ``members`` have a bar (else ``t_last``)."""
    counts: dict[pd.Timestamp, int] = {}
    for e in members:
        px = universe.prices.get(e)
        if px is None or px.empty:
            continue
        for d in px.index[px.index > t_last]:
            counts[d] = counts.get(d, 0) + 1
    need = LIVE_MIN_SHARE * max(len(members), 1)
    ok = [d for d, c in counts.items() if c >= need]
    return max(ok) if ok else t_last


def score_frame(latest: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Composite score (sum w * z, missing z = 0), contributions and percentile within the cross-section."""
    out = latest.copy()
    zc = [f"z_{f}" for f in FACTORS]
    z = out[zc].to_numpy(dtype=float)
    w = np.array([float(weights.get(f, 0.0)) for f in FACTORS])
    contrib = np.nan_to_num(z, nan=0.0) * w
    for k, f in enumerate(FACTORS):
        out[f"c_{f}"] = contrib[:, k]
    s = contrib.sum(axis=1)
    s[np.isnan(z).all(axis=1)] = np.nan
    out["score"] = s
    n = out["score"].notna().sum()
    out["percentile"] = ((out["score"].rank(method="average") - 0.5) / max(n, 1)).clip(0, 1)
    return out


def _live_snapshot(
    universe: PITUniverse,
    fundamentals: PITFundamentals,
    settings: Settings,
    cache: DiskCache,
    panel: pd.DataFrame,
    weights: dict[str, float],
    scale_caps: bool = False,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Latest cross-section for serving: the last rebalance members re-measured on the latest common session
    (market cap: KR PIT marcap of that session; US / injected caps: last month-end cap x price ratio)."""
    t_last = pd.Timestamp(panel["date"].max())
    last = panel[panel["date"] == t_last]
    ents = last["entity"].tolist()
    live = _live_date(universe, ents, t_last)
    if live <= t_last:
        return score_frame(last, weights), t_last
    members = last[["entity", "ticker", "name"]].assign(date=live)
    sectors = last[["entity", "sector", "sector_source"]]
    mcaps = None
    if universe.market is Market.US or scale_caps:
        ratio = {}
        for e in ents:
            px = universe.prices.get(e)
            if px is None or px.empty:
                continue
            c0 = px["close"].asof(t_last)
            c1 = px["close"].asof(live)
            if pd.notna(c0) and pd.notna(c1) and c0 > 0 and px.index[px.index <= live][-1] > t_last:
                ratio[e] = c1 / c0
        cap = last.set_index("entity")["market_cap"]
        mcaps = pd.DataFrame({e: [cap.get(e, np.nan) * ratio.get(e, np.nan)] for e in ents}, index=[live])
    snap = compute_factor_panel(
        universe,
        fundamentals,
        settings,
        members=members,
        next_dates={},
        market_caps=mcaps,
        sectors=sectors,
        cache=cache,
    )
    return score_frame(snap, weights), live


# ------------------------------------------------------------------------------------------------ run


def _universe_summary(universe: PITUniverse, market: Market, panel: pd.DataFrame) -> dict[str, Any]:
    per_date = panel.groupby("date").size()
    desc = (
        f"KOSPI+KOSDAQ 시가총액 상위 {universe.n}종목(시점별)"
        if market is Market.KR
        else ("S&P 500 시점별 전 구성종목" if universe.n >= 500 else f"S&P 500 시가총액 상위 {universe.n}종목(시점별)")
    )
    return {
        "description": desc,
        "n": int(universe.n),
        "entities": int(panel["entity"].nunique()),
        "avg_members": float(per_date.mean()) if len(per_date) else 0.0,
        "min_members": int(per_date.min()) if len(per_date) else 0,
        "delisted_entities": int(universe.meta["delisted"].astype(bool).sum())
        if universe.meta is not None and "delisted" in universe.meta
        else None,
    }


def _coverage_notes(market: Market, cov: dict[str, Any], universe: PITUniverse) -> list[str]:
    notes = []
    fs = cov.get("fwd_status", {})
    ps = cov.get("price_status", {})
    total = max(int(cov.get("member_months", 0)), 1)
    no_px = ps.get("no_price", 0)
    if no_px:
        what = "일별 가격 없음(인수·상폐 후 Yahoo 삭제, FTD 월말가만 존재)" if market is Market.US else "가격 없음"
        notes.append(
            f"회원-월 {total}개 중 {no_px}개({100 * no_px / total:.1f}%)는 {what} → 신호·수익률 계산에서 제외(생존편향 "
            "가능성: 주로 이후 인수된 종목)."
        )
        worst = sorted(
            ((y, v["price_missing"] / max(v["member_months"], 1)) for y, v in cov.get("by_year", {}).items()),
            key=lambda x: -x[1],
        )[:3]
        if worst and worst[0][1] >= 0.05:
            notes.append(
                "가격 결측 비중이 큰 연도: "
                + ", ".join(f"{y} {100 * r:.0f}%" for y, r in worst)
                + " — 초기 구간 결과는 표본 선택 편향 가능성을 감안해 해석하세요."
            )
    stale = ps.get("stale", 0) + ps.get("no_bar_yet", 0)
    if stale:
        notes.append(f"리밸런싱일 가격이 7일 넘게 없는(거래정지 등) 회원-월 {stale}개 제외.")
    exits = fs.get("delisted", 0) + fs.get("delisted_terminal", 0)
    if exits:
        notes.append(
            f"보유 중 상장폐지·편출 종료 {exits}건은 마지막 거래가(정리매매 포함) 또는 종료 가격으로 수익률에 반영."
        )
    halted = fs.get("halted_exit_next_trade", 0)
    if halted:
        notes.append(
            f"월말에 거래정지 중이던 보유 {halted}건은 다음 체결가(거래 재개 또는 정리매매)로 평가해 "
            "그 달 수익률에 반영."
        )
    no_entry = fs.get("no_entry", 0) + fs.get("delisted_before_entry", 0) + fs.get("halted_no_entry", 0)
    gaps = fs.get("price_gap", 0) + no_entry + fs.get("bad_price", 0) + fs.get("halted_unresolved", 0)
    if gaps:
        notes.append(
            f"다음 달 수익률을 만들 수 없는 회원-월 {gaps}개(가격 공백 {fs.get('price_gap', 0)}, 진입 불가(거래정지·"
            f"폐지) {no_entry}, 정지 미해소 {fs.get('halted_unresolved', 0)}, 가격 오류 {fs.get('bad_price', 0)}) 제외."
        )
    fa = cov.get("factor_available", {})
    if fa:
        notes.append(
            "팩터별 사용 가능 회원-월: "
            + ", ".join(f"{FACTOR_LABELS[f]} {fa.get(f, 0)}({100 * fa.get(f, 0) / total:.0f}%)" for f in FACTORS)
            + "."
        )
    if cov.get("ep_negative"):
        notes.append(
            f"적자(E/P≤0) {cov['ep_negative']}건·자본잠식(B/P≤0) {cov.get('bp_negative', 0)}건은 PER/PBR을 비우고 "
            "해당 밸류 순위 최하위로 처리."
        )
    sec = cov.get("sectors", {})
    if sec:
        unc = sec.get("unclassified_member_months", 0)
        by_year = {y: v.get("unclassified", 0) for y, v in sec.get("by_year", {}).items() if v.get("unclassified")}
        msg = f"섹터 미분류('기타') 회원-월 {unc}개"
        if by_year:
            msg += " (연도별 " + ", ".join(f"{y} {n}" for y, n in by_year.items()) + ")"
        merged = cov.get("sector_merged_into_other", 0)
        msg += f", 구성원 {cov.get('min_names_per_sector')}개 미만 소형 섹터를 '기타'로 합친 회원-월 {merged}개."
        notes.append(msg)
    return notes


def run_factor_backtest(
    market: Market,
    settings: Settings,
    cache: DiskCache,
    *,
    use_cache: bool = True,
    progress: Callable[[str], None] | None = None,
    regime: pd.DataFrame | None = None,
    universe: PITUniverse | None = None,
    fundamentals: PITFundamentals | None = None,
    market_caps: pd.DataFrame | None = None,
) -> BacktestResult:
    """PIT factor backtest + walk-forward model for ``market`` (cached; see module docstring).

    ``regime`` (date, region, score) enables the macro-conditioned model comparison. ``universe`` / ``fundamentals``
    / ``market_caps`` (wide dates x entities) are injection points (tests, or callers that already loaded them).
    """
    from quantcio.data.pit import load_pit_fundamentals, load_pit_universe

    market = Market(market)
    t0 = time.monotonic()
    if universe is None:
        _say(progress, "PIT 유니버스 로드")
        universe = load_pit_universe(
            market, settings, cache, universe_n_for(market, settings), start_for(market, settings)
        )
    snapshot = _snapshot_id(universe)
    key = _backtest_key(market, settings, snapshot, regime)
    if use_cache:
        hit = cache.get(NAMESPACE, key, None)
        if isinstance(hit, BacktestResult):
            cache.set(LATEST_NAMESPACE, _latest_key(market, settings), key)
            return hit
    ents = sorted(universe.membership["entity"].unique())
    if fundamentals is None:
        _say(progress, f"PIT 재무 로드 ({len(ents)}개 엔티티)")
        fundamentals = load_pit_fundamentals(market, settings, cache, ents)
    _say(progress, "팩터 패널 계산")
    start = pd.Timestamp(start_for(market, settings))
    dates = sorted(d for d in universe.membership["date"].unique() if pd.Timestamp(d) >= start)
    panel = compute_factor_panel(universe, fundamentals, settings, dates, cache=cache, market_caps=market_caps)
    panel.attrs["region"] = "kr" if market is Market.KR else "us"
    q = int(settings.backtest.quantiles)
    cost = cost_bps_for(market, settings)
    _say(progress, "팩터별 IC·분위 포트폴리오")
    table, curves = factor_table(panel, q, cost)
    _say(progress, "워크포워드 멀티팩터 모델 후보 비교·선택")
    split_vars = market_state_vars(universe, sorted(pd.to_datetime(panel["date"].unique())))
    model = walk_forward_model(
        panel, settings, regime, region=panel.attrs["region"], cost_bps=cost, split_vars=split_vars
    )
    if len(model.portfolios):
        curves["model|ls_net"] = _curve(model.portfolios["ls_net"], model.portfolios["end"])
        curves["model|lo_net"] = _curve(model.portfolios["lo_net"], model.portfolios["end"])
        curves["model|universe"] = _curve(model.portfolios["univ"], model.portfolios["end"])
    _say(progress, "최신 단면 점수(서빙)")
    latest, live_date = _live_snapshot(
        universe, fundamentals, settings, cache, panel, model.latest_weights, scale_caps=market_caps is not None
    )

    cov = panel_coverage(panel)
    sec_cov = sector_coverage(panel[["date", "entity", "sector", "sector_source"]])
    cov["sectors"] = sec_cov
    cov["min_names_per_sector"] = int(settings.backtest.min_names_per_sector)
    cov["market_cap_info"] = panel.attrs.get("market_cap_info", {})
    ucov = universe.coverage or {}
    cov["universe"] = {
        k: ucov.get(k)
        for k in (
            "member_months",
            "member_months_with_prices",
            "member_months_ftd_only",
            "member_months_missing",
            "delisted_entities",
            "member_exits_before_next_rebalance",
            "no_price_entities",
            "rejected_series",
        )
        if k in ucov
    }
    if isinstance(ucov.get("no_price_entities"), list):
        cov["universe"]["no_price_entities"] = len(ucov["no_price_entities"])
    fcov = fundamentals.coverage or {}
    cov["fundamentals"] = {
        "rows": int(len(fundamentals.panel)),
        "entities": int(fundamentals.panel["entity"].nunique()) if len(fundamentals.panel) else 0,
        **{
            k: fcov[k]
            for k in ("entities_without_xbrl", "restated_later_rows")
            if k in fcov and not isinstance(fcov[k], list)
        },
    }
    notes = [
        f"기간 {panel['date'].min().date()}~{panel['date'].max().date()} 월말 {panel['date'].nunique()}회, "
        f"{_universe_summary(universe, market, panel)['description']}, "
        "신호는 t 이하 가격·available_from<=t 재무만 사용, 보유는 t+1 세션 종가~다음 월말.",
        f"거래비용: 회전율 × {cost:.0f}bp(왕복) 차감. 다중검정: 팩터×변형 {table[0]['bh_tests'] if table else 0}개 "
        f"검정에 Benjamini–Hochberg q={BH_Q:.2f}.",
        "섹터 모멘텀은 섹터 안에서 값이 같아 섹터 중립 순위가 무의미하므로 전체 순위로만 평가합니다.",
    ]
    notes += _coverage_notes(market, cov, universe)
    if market is Market.US:
        mci = cov["market_cap_info"]
        if mci.get("no_shares_member_months"):
            notes.append(
                f"XBRL 주식수가 없는 회원-월 {mci['no_shares_member_months']}개는 시가총액을 만들 수 없어 E/P·B/P 결측 "
                "(거래대금 추정치는 쓰지 않음)."
            )
        notes.append(
            "US 섹터: 현재 구성종목은 Wikipedia GICS, 편출 종목은 SEC SIC→GICS 근사(현재 분류로 과거를 대신함)."
        )
    else:
        notes.append(
            "KR 섹터: DART 정기보고서의 KSIC 업종 코드(보고 기준, 폐지 종목 포함)를 대분류로 묶음. 가격은 배당 미반영."
        )
    notes += [f"[데이터] {n}" for n in universe.notes[:6]]
    notes += [f"[재무] {n}" for n in fundamentals.notes[:3]]
    notes += model.notes

    oos = model.oos
    model_dict = model.to_dict()
    model_dict["skill_min_ic_t"] = float(settings.backtest.model.skill_min_ic_t)
    model_dict["lookback_months"] = int(settings.backtest.model.lookback_months)
    model_dict["min_months"] = int(settings.backtest.model.min_months)
    model_dict["shrink_to_equal"] = float(settings.backtest.model.shrink_to_equal)
    model_dict["holdout_months"] = int(settings.backtest.model.holdout_months)
    model_dict["configured_method"] = str(settings.backtest.model.method)
    model_dict["oos_months"] = int(oos.get("n_months") or 0)
    live_info = {
        "as_of": str(live_date.date()),
        "rebalance_date": str(pd.Timestamp(panel["date"].max()).date()),
        "n": int(latest["score"].notna().sum()),
        "weights": {k: float(v) for k, v in model.latest_weights.items()},
    }
    result = BacktestResult(
        market=str(market),
        as_of=str(pd.Timestamp(panel["date"].max()).date()),
        period={
            "start": str(pd.Timestamp(panel["date"].min()).date()),
            "end": str(pd.Timestamp(panel["date"].max()).date()),
            "first_model_date": model_dict.get("first_model_date"),
        },
        n_months=int(panel["date"].nunique()),
        universe=_universe_summary(universe, market, panel),
        factor_table=table,
        model=model_dict,
        curves=curves,
        coverage=cov,
        notes=notes,
        settings_digest=settings.digest(),
        data_snapshot=snapshot,
        created_at=datetime.now().isoformat(timespec="seconds"),
        cost_bps=cost,
        bh_q=BH_Q,
        live=live_info,
        latest=latest,
        panel=panel,
        model_result=model,
    )
    result.coverage["seconds"] = round(time.monotonic() - t0, 1)
    cache.set(NAMESPACE, key, result)
    cache.set(LATEST_NAMESPACE, _latest_key(market, settings), key)
    return result


def report_summary(result: BacktestResult) -> dict[str, Any]:
    """Compact JSON-safe summary for ``PortfolioReport.factor_backtest[market]`` (report module 2 table)."""

    def perf(x: dict[str, Any] | None) -> dict[str, Any]:
        x = x or {}
        return {k: x.get(k) for k in ("ann_return", "sharpe", "mdd")}

    m = result.model or {}
    oos = m.get("oos") or {}
    rows = [
        {
            "factor": r["factor"],
            "label": r["label"],
            "variant": r["variant"],
            "ic_mean": r.get("ic_mean"),
            "ic_t": r.get("ic_t"),
            "direction_ok": r.get("direction_ok"),
            "bh_significant": r.get("bh_significant"),
            "ls_net": perf(r.get("ls_net")),
            "lo_net": perf(r.get("lo_net")),
        }
        for r in result.factor_table
    ]
    macro = m.get("macro") or None
    sel = m.get("selection") or {}
    served = (sel.get("rows") or {}).get(m.get("method")) or {}
    candidates = [
        {
            "name": name,
            "available": bool(r.get("available")),
            "pre_ic_mean": (r.get("pre") or {}).get("ic_mean"),
            "pre_ic_t": (r.get("pre") or {}).get("ic_t"),
            "holdout_ic_mean": (r.get("holdout") or {}).get("ic_mean"),
            "holdout_ic_t": (r.get("holdout") or {}).get("ic_t"),
            "full_ic_t": (r.get("full_all_months") or {}).get("ic_t"),
        }
        for name, r in (sel.get("rows") or {}).items()
    ]
    return _jsonable(
        {
            "market": result.market,
            "as_of": result.as_of,
            "period": result.period,
            "n_months": result.n_months,
            "universe": result.universe.get("description"),
            "cost_bps": result.cost_bps,
            "bh_q": result.bh_q,
            "factors": rows,
            "model": {
                "method": m.get("method"),
                "oos_months": oos.get("n_months"),
                "oos_ic_mean": oos.get("ic_mean"),
                "oos_ic_t": oos.get("ic_t"),
                "skill_ok": bool(m.get("skill_ok")),
                "skill_min_ic_t": m.get("skill_min_ic_t"),
                "ls_net": perf(oos.get("ls_net")),
                "lo_net": perf(oos.get("lo_net")),
                "weights": m.get("latest_weights"),
                "macro_used": None if macro is None else bool(macro.get("used")),
                "selection_mode": sel.get("mode"),
                "selection_version": sel.get("version"),
                "rule_winner": sel.get("rule_winner"),
                "holdout_months": sel.get("holdout_months"),
                "pre_ic_mean": (served.get("pre") or {}).get("ic_mean"),
                "pre_ic_t": (served.get("pre") or {}).get("ic_t"),
                "holdout_ic_mean": (served.get("holdout") or {}).get("ic_mean"),
                "holdout_ic_t": (served.get("holdout") or {}).get("ic_t"),
                "candidates": candidates,
            },
            "live_as_of": (result.live or {}).get("as_of"),
            "settings_digest": result.settings_digest,
        }
    )


def load_cached_backtest(market: Market, settings: Settings, cache: DiskCache) -> BacktestResult | None:
    """The latest cached backtest of ``market`` for the current backtest config (no data loading, no recompute)."""
    key = cache.get(LATEST_NAMESPACE, _latest_key(Market(market), settings), None)
    if not key:
        return None
    hit = cache.get(NAMESPACE, key, None)
    return hit if isinstance(hit, BacktestResult) else None


# ------------------------------------------------------------------------------------------------ serving


def _norm_ticker(market: Market, t: str) -> str:
    s = str(t).strip().upper()
    if market is Market.KR:
        return s.split(".")[0]
    return s.replace(".", "-")


def _place(values: pd.Series, v: float) -> float:
    """Uniform z of ``v`` inside the distribution ``values`` (same convention as factors.uniform_rank)."""
    x = values.dropna()
    n = len(x)
    if n == 0 or v is None or (isinstance(v, float) and np.isnan(v)):
        return float("nan")
    below = float((x < v).sum())
    eq = float((x == v).sum())
    u = (below + 0.5 * eq + 0.5) / (n + 1)
    return float(2.0 * u - 1.0)


def _us_cik_for(settings: Settings, tickers: list[str]) -> dict[str, str]:
    """ticker -> "CIK##########" from SEC company_tickers.json (cached daily by the PIT US loader)."""
    try:
        from quantcio.data.pit.us_cik import cik_entity, current_company_tickers
        from quantcio.data.pit.us_net import PitNet

        rows = current_company_tickers(PitNet(settings))
    except Exception as exc:  # noqa: BLE001 - mapping falls back to the universe's display tickers
        log.warning("SEC company_tickers unavailable: %s", exc)
        return {}
    want = set(tickers)
    return {t: cik_entity(c) for t, c, _ in rows if t in want}


def _outside_exposures(
    market: Market,
    ticker: str,
    entity: str | None,
    settings: Settings,
    cache: DiskCache,
    as_of: pd.Timestamp,
    prices: pd.DataFrame | None,
) -> tuple[dict[str, Any], list[str]]:
    """Raw exposures of a requested ticker outside the PIT universe (best effort, every gap noted)."""
    notes: list[str] = []
    ex: dict[str, Any] = {k: None for k in EXPOSURE_KEYS}
    ex["sector"] = None
    px = prices
    if px is None:
        try:
            from quantcio.data.base import get_provider

            prov = get_provider(market, settings, cache)
            inst = prov.resolve(ticker)
            px = prov.prices(inst, (as_of - pd.Timedelta(400, "D")).date())
            ex["sector_hint"] = inst.sector
        except Exception as exc:  # noqa: BLE001
            notes.append(f"가격 로드 실패({type(exc).__name__}) → 가격 팩터 없음")
            px = None
    if px is not None and len(px):
        px = px[px.index <= as_of]
    if px is not None and len(px) > 20:
        close = px["close"].astype(float)
        w3, w6 = (int(x) for x in settings.backtest.momentum_windows[:2])
        vs, vl = (int(x) for x in settings.backtest.volume_windows[:2])
        if len(close) > w3:
            ex["mom_3m"] = float(close.iloc[-1] / close.iloc[-1 - w3] - 1)
        if len(close) > w6:
            ex["mom_6m"] = float(close.iloc[-1] / close.iloc[-1 - w6] - 1)
        dv = (close * px["volume"].astype(float)).fillna(0.0)
        if len(dv) >= vl:
            a, b = dv.iloc[-vs:].mean(), dv.iloc[-vl:].mean()
            if a > 0 and b > 0:
                ex["volume_trend"] = float(np.log(a / b))
        last_close = float(close.iloc[-1])
    else:
        last_close = None
    # value: PIT market cap + PIT fundamentals of the entity
    if entity is not None:
        try:
            from quantcio.backtest.factors import valuation_frame
            from quantcio.data.pit import load_pit_fundamentals

            fund = load_pit_fundamentals(market, settings, cache, [entity])
            mcap = None
            if market is Market.KR:
                from quantcio.data.pit.kr import pit_market_cap_kr

                mc = pit_market_cap_kr(settings, cache, [entity], [as_of])
                mcap = float(mc.iloc[0, 0]) if mc.shape[1] and pd.notna(mc.iloc[0, 0]) else None
            elif last_close is not None and len(fund.panel) and "shares_out" in fund.panel:
                p = fund.panel[(fund.panel["available_from"] <= as_of)].dropna(subset=["shares_out"])
                if len(p):
                    r = p.sort_values(["available_from", "period_end"]).iloc[-1]
                    stale = (as_of - pd.Timestamp(r["period_end"])).days
                    if stale <= int(settings.pit.fundamentals.max_staleness_days):
                        mcap = float(r["shares_out"]) * last_close
                        notes.append("시가총액 = 최근 XBRL 주식수 × 최근 종가(제출 후 분할은 미반영)")
            if mcap:
                v = valuation_frame(
                    fund,
                    pd.DataFrame({entity: [mcap]}, index=[as_of]),
                    int(settings.pit.fundamentals.max_staleness_days),
                )
                if len(v):
                    r = v.iloc[0]
                    for k in ("per", "pbr", "ep", "bp"):
                        ex[k] = None if pd.isna(r[k]) else float(r[k])
                    ex["ep_negative"] = bool(pd.notna(r["ep"]) and r["ep"] <= 0)
                    ex["bp_negative"] = bool(pd.notna(r["bp"]) and r["bp"] <= 0)
                    if ex["ep_negative"]:
                        ex["ep"] = None
                    if ex["bp_negative"]:
                        ex["bp"] = None
            else:
                notes.append("PIT 시가총액 없음 → PER·PBR 팩터 없음")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"PIT 재무 로드 실패({type(exc).__name__}) → 밸류 팩터 없음")
        try:
            if market is Market.KR:
                from quantcio.backtest.sectors import kr_bulk_industries, ksic_to_sector

                bulk = kr_bulk_industries(settings, cache)
                code = entity.split("#")[0]
                b = bulk[(bulk["code"] == code) & (bulk["period_end"] < as_of)]
                if len(b):
                    ex["sector"] = ksic_to_sector(b.iloc[-1]["ksic"])
            else:
                from quantcio.backtest.sectors import us_entity_sectors

                sec, _, _ = us_entity_sectors(settings, [entity])
                ex["sector"] = sec.get(entity)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"섹터 조회 실패({type(exc).__name__})")
    return ex, notes


def _score_outside(
    ex: dict[str, Any], latest: pd.DataFrame, weights: dict[str, float], min_names: int, sector_neutral: bool
) -> tuple[dict[str, float | None], float, float]:
    """(z by factor, score, percentile) of an out-of-universe name placed in the universe distributions."""
    sector = ex.get("sector") or OTHER
    groups = latest["sector_group"]
    in_group = latest[groups == sector] if sector in set(groups) else latest[groups == OTHER]
    if sector in set(latest["sector"]):
        ex["sector_mom_3m"] = float(latest.loc[latest["sector"] == sector, "mom_3m"].mean())
    z: dict[str, float | None] = {}
    for f in FACTORS:
        col = RAW_COLUMN[f]
        v = ex.get(col)
        if f == "value_ep" and ex.get("ep_negative"):
            v = -np.inf
        if f == "value_bp" and ex.get("bp_negative"):
            v = -np.inf
        if v is None:
            z[f] = None
            continue
        ref = latest if (f in UNIVERSE_RANKED or not sector_neutral or len(in_group) < min_names) else in_group
        vals = rank_input(ref, f)
        zz = _place(vals, float(v))
        z[f] = None if not np.isfinite(zz) else zz
    score = sum(float(weights.get(f, 0.0)) * (z[f] or 0.0) for f in FACTORS)
    if all(z[f] is None for f in FACTORS):
        return z, float("nan"), float("nan")
    pct = (_place(latest["score"], score) + 1.0) / 2.0
    return z, score, float(np.clip(pct, 0, 1))


def score_tickers(
    market: Market,
    tickers: list[str],
    settings: Settings,
    cache: DiskCache,
    as_of: date | None = None,
    *,
    prices: dict[str, pd.DataFrame] | None = None,
    build_if_missing: bool = False,
    result: BacktestResult | None = None,
) -> dict[str, FactorModelScore]:
    """FactorModelScore of each requested ticker from the latest cached backtest (seconds when warm).

    Tickers map to PIT entities (KR: 6-digit code == entity or '<code>#<spell>'; US: SEC CIK). Members of the latest
    cross-section use its values; others get their raw exposures placed into the universe distribution (noted).
    ``as_of`` (optional) selects the month-end cross-section of the cached panel on or before it instead of the
    latest serving snapshot. ``prices`` (ticker -> PRICE_COLUMNS frame) avoids a price download for outside names.
    With no cached backtest, ``build_if_missing`` runs the backtest (minutes; KR 1-10 min, US cold 30-45 min),
    else {} is returned.
    """
    market = Market(market)
    res = result or load_cached_backtest(market, settings, cache)
    if res is None and build_if_missing:
        res = run_factor_backtest(market, settings, cache)
    if res is None or res.latest is None:
        log.warning("no cached factor backtest for %s - run `quantcio backtest --market %s`", market, market)
        return {}
    weights = dict(res.live.get("weights") or {})
    snap = res.latest
    snap_date = pd.Timestamp(res.live.get("as_of") or res.as_of)
    base_notes: list[str] = []
    if as_of is not None and pd.Timestamp(as_of) < snap_date and res.panel is not None:
        ds = sorted(d for d in res.panel["date"].unique() if pd.Timestamp(d) <= pd.Timestamp(as_of))
        if ds:
            t = pd.Timestamp(ds[-1])
            mr = res.model_result
            if mr is not None and t in mr.weights.index:
                weights = mr.weights.loc[t].to_dict()
                snap = score_frame(res.panel[res.panel["date"] == t], weights)
                snap_date = t
            else:
                base_notes.append(f"{t.date()} 기준 모델 가중치 없음 → 최신 단면 사용")
    age = (pd.Timestamp.today().normalize() - snap_date).days
    if age > 10:
        base_notes.append(f"팩터 단면이 {age}일 전({snap_date.date()}) 기준 — 백테스트 캐시 갱신 필요")
    oos = res.model.get("oos") or {}
    skill_ok = bool(res.model.get("skill_ok"))
    mult = 1.0 if skill_ok else SKILL_SHRINK
    side = _side_test(res)
    min_names = int(settings.backtest.min_names_per_sector)
    by_ticker = {}
    for _, r in snap.iterrows():
        by_ticker.setdefault(_norm_ticker(market, str(r["ticker"])), r)
    by_entity = {str(r["entity"]): r for _, r in snap.iterrows()}
    want = [_norm_ticker(market, t) for t in tickers]
    unknown = [t for t in want if t not in by_ticker]
    ciks = _us_cik_for(settings, unknown) if market is Market.US and unknown else {}
    out: dict[str, FactorModelScore] = {}
    for raw_t, t in zip(tickers, want, strict=True):
        notes = list(base_notes)
        row = by_ticker.get(t)
        entity = None
        if row is None:
            if market is Market.KR:
                cands = [e for e in by_entity if e.split("#")[0] == t]
                entity = cands[-1] if cands else t
            else:
                entity = ciks.get(t)
            row = by_entity.get(entity) if entity else None
        if row is not None:
            z = {f: (None if pd.isna(row[f"z_{f}"]) else float(row[f"z_{f}"])) for f in FACTORS}
            exposures = {k: (None if pd.isna(row[k]) else float(row[k])) for k in EXPOSURE_KEYS}
            score = float(row["score"]) if pd.notna(row["score"]) else float("nan")
            pct = float(row["percentile"]) if pd.notna(row["percentile"]) else float("nan")
            sector = str(row["sector"])
            if row.get("sector_source") == "none":
                notes.append("섹터 미분류('기타')")
        else:
            ex, n2 = _outside_exposures(market, raw_t, entity, settings, cache, snap_date, (prices or {}).get(raw_t))
            notes += n2
            z, score, pct = _score_outside(ex, snap, weights, min_names, bool(settings.backtest.sector_neutral))
            exposures = {k: ex.get(k) for k in EXPOSURE_KEYS}
            sector = ex.get("sector") or OTHER
            notes.append("PIT 유니버스 밖 종목 — 유니버스 분포에 넣어 백분위 계산")
        if not np.isfinite(score) or not np.isfinite(pct):
            log.warning("factor model: no usable factor for %s", raw_t)
            continue
        contributions = {f: float(weights.get(f, 0.0)) * (z[f] or 0.0) for f in FACTORS}
        missing = [FACTOR_LABELS[f] for f in FACTORS if z[f] is None]
        if missing:
            notes.append(tr(settings, "결측 팩터(0으로 처리): ", "Missing factors (set to 0): ") + ", ".join(missing))
        pillar = float(np.clip(2.0 * (pct - 0.5), -1.0, 1.0) * mult)
        if not skill_ok:
            notes.append(
                tr(
                    settings,
                    f"표본 외 IC t가 기준({settings.backtest.model.skill_min_ic_t:.1f}) 미달 → 필러 ×{SKILL_SHRINK}",
                    f"OOS IC t below the {settings.backtest.model.skill_min_ic_t:.1f} gate → pillar ×{SKILL_SHRINK}",
                )
            )
        out[raw_t] = FactorModelScore(
            as_of=snap_date.date(),
            score=round(score, 6),
            percentile=round(float(np.clip(pct, 0, 1)), 6),
            pillar=round(pillar, 6),
            sector=sector,
            exposures=exposures,
            zscores=z,
            contributions={k: round(v, 6) for k, v in contributions.items()},
            weights={k: round(float(v), 6) for k, v in weights.items()},
            skill_ok=skill_ok,
            oos_ic_mean=oos.get("ic_mean"),
            oos_ic_t=oos.get("ic_t"),
            weak_side=side.get("weak_side"),
            side_test=side,
            notes=notes,
        )
    return out


def _side_test(res: BacktestResult) -> dict[str, Any]:
    """``pillar_side_test`` of the served model's walk-forward OOS scores ({} when unavailable = symmetric pillar)."""
    mr = res.model_result
    if mr is None or res.panel is None or mr.scores is None or mr.scores.empty:
        return {}
    period = (mr.selection or {}).get("holdout_period")
    try:
        out = pillar_side_test(mr.scores, res.panel, pd.Timestamp(period[0]) if period else None)
    except Exception as exc:  # noqa: BLE001 - a diagnostic: failure keeps the symmetric pillar, logged
        log.warning("factor pillar side test failed (%s): %s", res.market, exc)
        return {}
    return _jsonable(out)
