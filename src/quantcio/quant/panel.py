"""RSMT-Q pooled training panel (one market): regime features, offsets and forward labels per grid date.

`build_panel(PricePanel, settings)` turns the market universe (plus the requested tickers) into:
- `train`: one row per (ticker, grid date) with a KNOWN h-day label. The grid is the market's common
  calendar (union of stock dates) sampled every `sample_step` sessions counting backwards from the
  latest date, so forward windows do not overlap and the latest date is always on the grid.
  Overlapping samples (rsmt.overlap_step = k > 0, per market `rsmt.markets.<M>.overlap_step`): training rows every
  k sessions (k must divide sample_step), so neighbouring labels overlap. Every row carries `block` = its h-session
  block ((calendar position - latest position) // h; the fitted tree clusters on it, see quant.rsmt) and `on_grid`
  = the row is on the non-overlapping sample_step grid (`grid_dates`). CV folds, CV test rows and the market
  moments use the on_grid rows only, so every variant is evaluated on the same rows with the same market leg.
  Without overlap every row is on_grid and every block holds exactly one sample date (block = date cluster).
- `pred`: the latest row of every ticker (the rows the model serves).
- `scale_h` (train and pred): the horizon volatility known at t, offset_sigma_feature (annualised EWMA vol) *
  sqrt(h / 252) - the target scale of rsmt.target_scale = "ewma".

Labels: r = log(close[t+h]/close[t]); m = the same for the benchmark aligned to the stock calendar;
beta = shrink_beta(beta_252) known at t; y_e = r - beta*m (beta-adjusted excess log return).
Offsets: z = cross-sectional 12-1 momentum z (>= 10 names, winsorised at +-3); v = log(ewma_vol) - c_v.

Market drift mu_m (the simulated benchmark leg): never extrapolate a rally. With the time-decayed mean
of the benchmark h-returns mu_hist (each benchmark date once, weights w) and the long-run drift prior
prior = log(1 + rsmt.market_mu_prior_annual) * h / 252,
    mu_m = (1 - s) * mu_hist + s * prior,     s = rsmt.market_mu_shrink.
Live 2026-09-24: the KOSPI mu_hist was +2.9%/20d (the old 50%-toward-0 rule served +1.45%/20d, i.e.
~20%/yr of market drift into every KR name); with s = 0.8 and a 6%/yr prior it is +0.95%/20d.

No silent degradation: training rows with any NaN input are dropped AND counted (`dropped`);
prediction rows are never dropped - missing inputs are replaced explicitly and the row is marked
`degraded` with reasons in the report language (settings.report.language, e.g. "오프셋 입력 누락(z|v): xs_mom_z →
0으로 설정" / "offset input missing (z|v): xs_mom_z → set to 0"); `is_offset_missing` recognises both.

Point-in-time universe (``PricePanel.membership`` not None, pit.universe_mode = "pit"): a (ticker, date) row
is a TRAINING row only when the ticker was a member of the latest rebalance date <= date (members of d are
eligible for d <= t < next rebalance); the cross-sectional momentum z of a date is computed across that
date's members only; prediction rows are the members of the latest date plus every ticker that never
appears in the membership (the requested names added from the production providers). Rows outside the
membership are counted in ``PanelData.n_non_member`` (not "dropped": they are simply not in the universe).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from quantcio.config import RSMTMarketConfig, Settings
from quantcio.data.base import PricePanel
from quantcio.i18n import tr
from quantcio.models import Market
from quantcio.quant.distribution import (
    nu_from_excess_kurtosis,
    weighted_excess_kurtosis,
)
from quantcio.quant.features import (
    align_benchmark,
    ewma_vol,
    forward_log_return,
    regime_features,
    shrink_beta,
)

log = logging.getLogger(__name__)

XS_MIN_TICKERS = 10  # cross-sectional z needs at least this many names on the date
XS_WINSOR = 3.0  # |z| cap
STALE_SESSIONS = 3  # a ticker whose last bar lags the market calendar by >= this is flagged
LABEL_SPAN_TOL = 2  # a label window may exceed h market sessions by at most this (halts)
TRADING_DAYS = 252  # annual -> h-day scaling of the market drift prior
OFFSET_MISSING_REASON = "오프셋 입력 누락(z|v)"
OFFSET_MISSING_REASON_EN = "offset input missing (z|v)"
LABEL_COLUMNS = ("y_e", "m", "r", "t_end")


def is_offset_missing(reason: str) -> bool:
    """True for a degraded reason that says the offset inputs (z | v) were missing (either language)."""
    return str(reason).startswith((OFFSET_MISSING_REASON, OFFSET_MISSING_REASON_EN))


def market_config(settings: Settings, market: Market | str) -> RSMTMarketConfig:
    """Per-market RSMT-Q config (KR / US)."""
    key = market.value if isinstance(market, Market) else str(market)
    try:
        return settings.rsmt.markets[key]
    except KeyError as exc:
        raise ValueError(f"rsmt.markets has no entry for market {key!r}") from exc


# model switches that rsmt.markets.<M> may override (None there = the global rsmt.<key>)
RSMT_VARIANT_KEYS = (
    "overlap_step",
    "multiple_testing",
    "pca_components",
    "target_scale",
    "gamma_mode",
    "gamma_shrink_k",
    "sigma_terms",
)
# sigma-equation regressors (rsmt.sigma_terms): HAR log ratios per ticker, day-of-year harmonics per date
HAR_WINDOWS = (5, 66, 252)
CAL_TERMS = ("cal_s1", "cal_c1", "cal_s4", "cal_c4")
SIGMA_TERM_COLUMNS = (*(f"har_{k}" for k in HAR_WINDOWS), *CAL_TERMS)
SIGMA_TERM_MISSING_REASON = "변동성 식 입력 누락"
SIGMA_TERM_MISSING_REASON_EN = "sigma-equation input missing"


def calendar_terms(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Annual (k = 1) and quarterly (k = 4) harmonics of the day of year: sin / cos(2 pi k (doy - 1) / 365.25)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    frac = (d.dayofyear.to_numpy(dtype="float64") - 1.0) / 365.25
    out = {}
    for k in (1, 4):
        out[f"cal_s{k}"] = np.sin(2.0 * math.pi * k * frac)
        out[f"cal_c{k}"] = np.cos(2.0 * math.pi * k * frac)
    return pd.DataFrame(out)


def sigma_terms_of(settings: Settings, market: Market | str) -> list[str]:
    """The market's sigma-equation terms (rsmt.markets.<M>.sigma_terms, else rsmt.sigma_terms)."""
    return list(rsmt_option(settings, market, "sigma_terms") or [])


def rsmt_option(settings: Settings, market: Market | str, name: str) -> Any:
    """The effective value of an RSMT-Q model switch for one market: rsmt.markets.<M>.<name> when set, else
    rsmt.<name> (docs/RSMT.md).

    Raises:
        KeyError: ``name`` is not a per-market switch (RSMT_VARIANT_KEYS).
    """
    if name not in RSMT_VARIANT_KEYS:
        raise KeyError(f"{name!r} is not a per-market RSMT-Q switch {RSMT_VARIANT_KEYS}")
    local = getattr(market_config(settings, market), name, None)
    return getattr(settings.rsmt, name) if local is None else local


def overlap_factor(settings: Settings, market: Market | str) -> float:
    """Training rows per non-overlapping sample: sample_step / overlap_step with overlap, else 1.0."""
    k = int(rsmt_option(settings, market, "overlap_step") or 0)
    step = max(int(settings.rsmt.sample_step), int(settings.horizon_days))
    return step / k if 0 < k < step else 1.0


def on_grid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows on the non-overlapping sample grid (all rows when the frame has no `on_grid` flag)."""
    if "on_grid" not in frame.columns:
        return frame
    return frame[frame["on_grid"].to_numpy(dtype=bool)]


def market_mu_prior(prior_annual: float, horizon: int) -> float:
    """Long-run market drift prior of the h-day LOG return: log(1 + prior_annual) * h / 252.

    Raises:
        ValueError: prior_annual not finite or <= -1, or horizon < 1.
    """
    pa = float(prior_annual)
    if not math.isfinite(pa) or pa <= -1.0:
        raise ValueError(f"rsmt.market_mu_prior_annual must be finite and > -1, got {prior_annual!r}")
    if int(horizon) < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon!r}")
    return math.log1p(pa) * int(horizon) / TRADING_DAYS


@dataclass
class MarketMoments:
    """Benchmark h-return moments used for the simulated market leg.

    `mu` / `mu_by_bench` are the SERVED drifts (1 - shrink) * mu_hist + shrink * mu_prior; the
    time-decayed historical means before the shrink are kept in `mu_hist` / `mu_hist_by_bench`.
    """

    mu: float  # pooled served drift (prior-shrunk)
    nu: float  # pooled tail parameter, clipped to [nu_min, nu_max]
    mu_by_bench: dict[str, float] = field(default_factory=dict)
    nu_by_bench: dict[str, float] = field(default_factory=dict)
    n_obs: int = 0
    mu_hist: float = float("nan")  # pooled time-decayed mean of the benchmark h-returns
    mu_hist_by_bench: dict[str, float] = field(default_factory=dict)
    mu_prior: float = 0.0  # log(1 + prior_annual) * h / 252
    shrink: float = 0.0  # weight on mu_prior


def market_moments(
    train: pd.DataFrame,
    nu_min: float,
    nu_max: float,
    *,
    mu_prior: float,
    shrink: float,
) -> MarketMoments:
    """Time-decayed benchmark moments from the (bench, date)-deduplicated training rows.

    Each benchmark date counts once (not once per ticker). mu_hist = sum(w*m)/sum(w) and the served
    drift is mu = (1 - shrink) * mu_hist + shrink * mu_prior (see the module docstring: a recent
    rally must not be extrapolated). nu comes from the weighted excess kurtosis via
    nu = 4 + 6/kappa (nu_max when kappa <= 0), clipped. `mu_prior` / `shrink` are required keywords
    (`moments_from_settings` supplies them from the config), so no caller silently skips the prior.

    Raises:
        ValueError: empty frame, no finite benchmark label, shrink outside [0, 1] or a non-finite prior.
    """
    s, prior = float(shrink), float(mu_prior)
    if not (math.isfinite(s) and 0.0 <= s <= 1.0):
        raise ValueError(f"rsmt.market_mu_shrink must be in [0, 1], got {shrink!r}")
    if not math.isfinite(prior):
        raise ValueError(f"market drift prior must be finite, got {mu_prior!r}")
    if train.empty:
        raise ValueError("market_moments: empty training frame")
    d = train.drop_duplicates(["bench", "date"])[["bench", "m", "w"]]
    d = d[np.isfinite(d["m"]) & np.isfinite(d["w"])]
    if d.empty:
        raise ValueError("market_moments: no finite benchmark labels")

    def _one(frame: pd.DataFrame) -> tuple[float, float]:
        w = frame["w"].to_numpy(dtype="float64")
        m = frame["m"].to_numpy(dtype="float64")
        hist = float((w * m).sum() / w.sum())
        nu = nu_from_excess_kurtosis(weighted_excess_kurtosis(m, w), nu_min, nu_max)
        return hist, nu

    def _served(hist: float) -> float:
        return (1.0 - s) * hist + s * prior

    hist, nu = _one(d)
    hist_b: dict[str, float] = {}
    nu_b: dict[str, float] = {}
    for bench, frame in d.groupby("bench", sort=True):
        hist_b[str(bench)], nu_b[str(bench)] = _one(frame)
    return MarketMoments(
        mu=_served(hist),
        nu=nu,
        mu_by_bench={b: _served(v) for b, v in hist_b.items()},
        nu_by_bench=nu_b,
        n_obs=len(d),
        mu_hist=hist,
        mu_hist_by_bench=hist_b,
        mu_prior=prior,
        shrink=s,
    )


def moments_from_settings(train: pd.DataFrame, settings: Settings, market: Market | str, horizon: int) -> MarketMoments:
    """`market_moments` with the market's nu bounds and the configured drift prior / shrink (the ONE
    rule used by the panel, the CV folds and therefore serving). Only the rows on the non-overlapping grid
    (`on_grid_rows`) count, so overlapping training rows never change the market leg."""
    mcfg = market_config(settings, market)
    return market_moments(
        on_grid_rows(train),
        mcfg.nu_min,
        mcfg.nu_max,
        mu_prior=market_mu_prior(settings.rsmt.market_mu_prior_annual, horizon),
        shrink=settings.rsmt.market_mu_shrink,
    )


@dataclass
class PanelData:
    """Pooled RSMT-Q training / prediction set of ONE market (see module docstring)."""

    market: Market
    horizon: int
    train: pd.DataFrame
    pred: pd.DataFrame
    c_v: float
    market_mu_hist: float  # SERVED pooled market drift mu_m (prior-shrunk; the name predates the prior)
    market_nu: float
    dropped: dict[str, int]  # {"total": n, <column>: n rows with NaN in it}
    benchmark_ewma: dict[str, float]  # ticker -> benchmark EWMA vol (annualised) on its last date
    partition_features: list[str] = field(default_factory=list)
    offset_mu_feature: str = "xs_mom_z"
    offset_sigma_feature: str = "ewma_vol"
    as_of: pd.Timestamp | None = None  # latest grid date (= latest market date)
    grid_dates: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    n_candidates: int = 0  # training candidate rows before the NaN drop
    market_mu_by_bench: dict[str, float] = field(default_factory=dict)
    market_nu_by_bench: dict[str, float] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)  # ticker -> load/build error (never silent)
    n_tickers: int = 0  # tickers attempted = used + failed
    notes: list[str] = field(default_factory=list)
    n_non_member: int = 0  # PIT: grid rows of loaded tickers that were not universe members on that date
    pit: bool = False  # built with a PIT membership
    # market drift audit: time-decayed historical means BEFORE the prior shrink, the prior and its weight
    market_mu_raw: float = float("nan")
    market_mu_raw_by_bench: dict[str, float] = field(default_factory=dict)
    market_mu_prior: float = 0.0
    market_mu_shrink: float = 0.0
    overlap_step: int = 0  # training rows every overlap_step sessions (0 = the non-overlapping grid only)
    block_sessions: int = 0  # sessions per `block` cluster (= h)

    @property
    def failed_tickers(self) -> dict[str, str]:
        """Failed tickers (benchmark failures, keyed 'benchmark:<id>', excluded)."""
        return {k: v for k, v in self.failed.items() if not k.startswith("benchmark:")}

    @property
    def failed_ratio(self) -> float:
        return len(self.failed_tickers) / self.n_tickers if self.n_tickers else 0.0

    @property
    def tickers(self) -> list[str]:
        return list(self.pred["ticker"])


# ------------------------------------------------------------------------------ build


def _ticker_frame(
    ticker: str,
    px: pd.DataFrame,
    bench: pd.DataFrame | None,
    h: int,
    beta_shrink: float,
) -> pd.DataFrame:
    """All per-date columns of one ticker (features at t use data <= t; labels look forward h rows)."""
    feats = regime_features(px, bench if bench is not None else pd.DataFrame())
    close = px["close"]
    bclose = align_benchmark(px, bench if bench is not None else pd.DataFrame())
    out = feats.copy()
    lr = np.log(close).diff()
    with np.errstate(divide="ignore", invalid="ignore"):  # a flat window (halt) -> log 0 -> NaN (dropped / flagged)
        for k in HAR_WINDOWS:
            rv = np.sqrt(TRADING_DAYS * (lr * lr).rolling(k, min_periods=k).mean())
            out[f"har_{k}"] = np.log(rv / feats["ewma_vol"])
    out["beta"] = shrink_beta(feats["beta_252"], beta_shrink)
    if bench is not None and not bench.empty:
        bvol = ewma_vol(bench["close"])
        out["bench_vol"] = bvol.reindex(px.index.union(bvol.index)).ffill(limit=3).reindex(px.index)
    else:
        out["bench_vol"] = np.nan
    out["r"] = forward_log_return(close, h)
    out["m"] = forward_log_return(bclose, h)
    out["y_e"] = out["r"] - out["beta"] * out["m"]
    t_end = np.full(len(px), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    if len(px) > h:
        t_end[:-h] = px.index.values[h:]
    out["t_end"] = t_end
    num = out.select_dtypes("number").columns
    out[num] = out[num].replace([np.inf, -np.inf], np.nan)
    out.insert(0, "date", px.index)
    out.insert(0, "ticker", ticker)
    return out.reset_index(drop=True)


def _xs_zscore(wide: pd.DataFrame, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Row-wise z-score across tickers (>= XS_MIN_TICKERS finite names, else NaN), clipped to +-3.

    ``ref`` (PIT): the cross-section that defines the date's mean / sd (that date's members); every column of
    ``wide`` - members and the requested non-members alike - is standardised against it."""
    base = wide if ref is None else ref
    count = base.notna().sum(axis=1)
    mean = base.mean(axis=1)
    sd = base.std(axis=1, ddof=1).replace(0.0, np.nan)
    z = wide.sub(mean, axis=0).div(sd, axis=0)
    z[count < XS_MIN_TICKERS] = np.nan
    return z.clip(-XS_WINSOR, XS_WINSOR)


def member_mask(membership: pd.DataFrame, tickers: pd.Series, dates: pd.Series) -> np.ndarray:
    """True where ``tickers[i]`` is a member of the latest rebalance date <= ``dates[i]`` (PIT membership:
    columns date, ticker)."""
    reb = np.sort(pd.to_datetime(membership["date"]).unique())
    if reb.size == 0:
        return np.zeros(len(tickers), dtype=bool)
    d = pd.DatetimeIndex(dates).to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(reb.astype("datetime64[ns]"), d, "right") - 1
    ok = pos >= 0
    applicable = np.where(ok, reb.astype("datetime64[ns]")[np.clip(pos, 0, None)], np.datetime64("NaT", "ns"))
    pairs = set(
        zip(pd.to_datetime(membership["date"]).to_numpy(dtype="datetime64[ns]"), membership["ticker"].astype(str))
    )
    t = np.asarray(tickers, dtype=object).astype(str)
    return np.array([bool(o) and (a, tk) in pairs for o, a, tk in zip(ok, applicable, t)], dtype=bool)


def _as_timestamp(
    as_of: date | datetime | str | pd.Timestamp | None,
) -> pd.Timestamp | None:
    if as_of is None:
        return None
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def build_panel(
    panel: PricePanel,
    settings: Settings,
    as_of: date | datetime | str | pd.Timestamp | None = None,
    *,
    c_v: float | None = None,
) -> PanelData:
    """Build the pooled RSMT-Q panel of one market (see module docstring).

    Args:
        panel: prices of the market universe (+ requested tickers) and their benchmarks.
        settings: uses horizon_days and the `rsmt` block (+ the market's half_life/nu bounds).
        as_of: optional cutoff; every price/benchmark bar after it is ignored (backtests).
        c_v: the centering constant of v (log EWMA vol) of an already fitted model: a serving panel built with it
            gives the model the same v scale it was fitted on (`pipeline.fit_market`); None = computed here.

    Raises:
        ValueError: no usable ticker, or no training row survives the NaN filter.
    """
    h = int(settings.horizon_days)
    rcfg = settings.rsmt
    mcfg = market_config(settings, panel.market)
    step = max(int(rcfg.sample_step), h)
    overlap = int(rsmt_option(settings, panel.market, "overlap_step") or 0)
    if overlap >= step:
        overlap = 0  # a step >= the non-overlapping step adds nothing
    if overlap and step % overlap:
        raise ValueError(f"rsmt.overlap_step {overlap} must divide the sample step {step} (the grid must nest)")
    part = list(rcfg.partition_features)
    zname, vname = rcfg.offset_mu_feature, rcfg.offset_sigma_feature
    cutoff = _as_timestamp(as_of)

    failed: dict[str, str] = dict(panel.failed)
    notes: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    for ticker in sorted(panel.prices):
        px = panel.prices[ticker]
        if cutoff is not None:
            px = px.loc[:cutoff]
        if len(px) <= h + 1:
            failed[ticker] = tr(settings, f"이력 부족({len(px)}행)", f"history too short ({len(px)} rows)")
            continue
        bid = panel.ticker_benchmark.get(ticker)
        bench = panel.benchmarks.get(bid) if bid is not None else None
        if bench is not None and cutoff is not None:
            bench = bench.loc[:cutoff]
        if bench is None or bench.empty:
            notes.append(
                tr(
                    settings,
                    f"{ticker}: 벤치마크({bid}) 없음 — 시장 변수·시장 레이블 결측",
                    f"{ticker}: benchmark {bid} unavailable — market features and market labels absent",
                )
            )
            bench = None
        frame = _ticker_frame(ticker, px, bench, h, rcfg.beta_shrink)
        frame.insert(2, "bench", "" if bid is None else str(bid))
        frames[ticker] = frame
    if not frames:
        raise ValueError(
            tr(
                settings,
                f"RSMT-Q {panel.market.value}: 사용 가능한 종목 없음 (failed={failed})",
                f"RSMT-Q {panel.market.value}: no usable name (failed={failed})",
            )
        )

    calendar = pd.DatetimeIndex(sorted(set().union(*(f["date"] for f in frames.values()))))
    last = calendar[-1]
    grid = calendar[::-1][::step][::-1]
    fine = calendar[::-1][::overlap][::-1] if overlap else grid  # training grid (nests the sample grid)

    membership = panel.membership
    pit = membership is not None and len(membership) > 0
    if pit:
        membership = membership.copy()
        membership["date"] = pd.to_datetime(membership["date"]).dt.normalize()
        if cutoff is not None:
            membership = membership[membership["date"] <= cutoff]
        member_tickers = set(membership["ticker"].astype(str))
        for f in frames.values():
            f["member"] = member_mask(membership, f["ticker"], f["date"])

    # ---- offset z: cross-sectional momentum z on every calendar date (PIT: across that date's members)
    if zname == "xs_mom_z":
        wide = pd.DataFrame({t: f.set_index("date")["mom_12_1"] for t, f in frames.items()}).reindex(calendar)
        ref = None
        if pit:
            mem_wide = pd.DataFrame({t: f.set_index("date")["member"] for t, f in frames.items()}).reindex(calendar)
            ref = wide.where(mem_wide.eq(True))  # NaN (no bar that day) = not a member
        xs = _xs_zscore(wide, ref)
        for t, f in frames.items():
            f["xs_mom_z"] = xs[t].reindex(pd.DatetimeIndex(f["date"])).to_numpy()
    elif zname not in next(iter(frames.values())).columns:
        raise ValueError(f"offset_mu_feature {zname!r} is not produced by regime_features")

    long = pd.concat(frames.values(), ignore_index=True)
    long = pd.concat([long, calendar_terms(long["date"])], axis=1)
    sig_terms = sigma_terms_of(settings, panel.market)
    vraw = long[vname] if vname in long.columns else None
    if vraw is None:
        raise ValueError(f"offset_sigma_feature {vname!r} is not produced by regime_features")
    long["log_v"] = np.log(vraw.where(vraw > 0))
    long["z"] = long[zname]
    age = (last - pd.DatetimeIndex(long["date"])).days.to_numpy(dtype="float64")
    long["w"] = 0.5 ** (age / float(mcfg.half_life_days))

    # ---- training rows: grid dates whose label window ends inside the calendar
    pos = pd.Series(np.arange(len(calendar)), index=calendar)
    train_dates = fine[pos.reindex(fine).to_numpy() + h <= len(calendar) - 1]
    cand = long[long["date"].isin(train_dates)]
    n_non_member = 0
    if pit:
        on_grid = cand["member"].to_numpy(dtype=bool)
        n_non_member = int((~on_grid).sum())
        cand = cand[on_grid]
    required = [*part, "z", "log_v", *sig_terms, "beta", "bench_vol", "y_e", "m"]
    na = cand[required].isna()
    bad = na.any(axis=1).to_numpy()
    dropped = {("v" if c == "log_v" else c): int(na[c].sum()) for c in required if na[c].any()}
    # the label is h rows ahead on the ticker's OWN bars: across a trading halt that window spans far
    # more than h market sessions (a months-long "20-day" return) and overlaps the next grid sample
    span = (
        pos.reindex(pd.DatetimeIndex(cand["t_end"])).to_numpy() - pos.reindex(pd.DatetimeIndex(cand["date"])).to_numpy()
    )
    long_span = np.nan_to_num(span, nan=0.0) > h + LABEL_SPAN_TOL
    if long_span.any():
        dropped["label_span"] = int(long_span.sum())
        bad = bad | long_span
    dropped["total"] = int(bad.sum())
    train = cand.loc[~bad].copy()
    if train.empty:
        raise ValueError(
            tr(
                settings,
                f"RSMT-Q {panel.market.value}: 학습 행 없음 (후보 {len(cand)}행 전부 제외: {dropped})",
                f"RSMT-Q {panel.market.value}: no training row (all {len(cand)} candidate rows excluded: {dropped})",
            )
        )
    if dropped["total"]:
        log.info(
            "RSMT-Q %s panel: dropped %d/%d candidate rows %s",
            panel.market.value,
            dropped["total"],
            len(cand),
            dropped,
        )

    w = train["w"].to_numpy()
    if c_v is None:
        c_v = float((w * train["log_v"].to_numpy()).sum() / w.sum())
    train["v"] = train["log_v"] - c_v
    horizon_scale = math.sqrt(h / TRADING_DAYS)
    train["scale_h"] = np.exp(train["log_v"]) * horizon_scale
    train["on_grid"] = train["date"].isin(grid).to_numpy()
    # h-session blocks counted back from the latest date: increasing with the date, one sample date per block on
    # the non-overlapping grid (so clustering on it is clustering on dates there)
    train["block"] = (pos.reindex(pd.DatetimeIndex(train["date"])).to_numpy() - (len(calendar) - 1)) // h

    # ---- prediction rows: latest bar of every ticker, never dropped (PIT: members of the latest rebalance
    # date + tickers outside the membership, i.e. the requested names + past members that still trade - a
    # requested name may be a former member; delisted / ended PIT entities are not served)
    pred = long.groupby("ticker", sort=True).tail(1).copy().reset_index(drop=True)
    if pit:
        latest_members = set(membership.loc[membership["date"] == membership["date"].max(), "ticker"].astype(str))
        trading = pos.reindex(pd.DatetimeIndex(pred["date"])).to_numpy() >= len(calendar) - STALE_SESSIONS
        keep = [
            t in latest_members or t not in member_tickers or bool(live)
            for t, live in zip(pred["ticker"].astype(str), trading)
        ]
        pred = pred[np.asarray(keep, dtype=bool)].reset_index(drop=True)
    pred["v"] = pred["log_v"] - c_v
    flags: list[bool] = []
    reasons_col: list[list[str]] = []
    zfix, vfix, bfix = [], [], []
    for t in sig_terms:  # sigma-equation inputs: a missing one is set to 0 (neutral) and flagged, never dropped
        miss = ~np.isfinite(pred[t].to_numpy(dtype="float64"))
        pred.loc[miss, t] = 0.0
        pred[f"sigmiss_{t}"] = miss
    for row in pred.itertuples(index=False):
        reasons: list[str] = []
        miss_sig = [t for t in sig_terms if getattr(row, f"sigmiss_{t}")]
        if miss_sig:
            names = ", ".join(miss_sig)
            reasons.append(
                tr(
                    settings,
                    f"{SIGMA_TERM_MISSING_REASON}: {names} → 0(중립)으로 설정",
                    f"{SIGMA_TERM_MISSING_REASON_EN}: {names} → set to 0 (neutral)",
                )
            )
        missing_off = [name for name, val in ((zname, row.z), (vname, row.v)) if not math.isfinite(val)]
        if missing_off:
            names = ", ".join(missing_off)
            reasons.append(
                tr(
                    settings,
                    f"{OFFSET_MISSING_REASON}: {names} → 0으로 설정",
                    f"{OFFSET_MISSING_REASON_EN}: {names} → set to 0",
                )
            )
        zfix.append(row.z if math.isfinite(row.z) else 0.0)
        vfix.append(row.v if math.isfinite(row.v) else 0.0)
        miss_part = [c for c in part if not math.isfinite(getattr(row, c))]
        if miss_part:
            names = ", ".join(miss_part)
            reasons.append(
                tr(
                    settings,
                    f"분할 변수 누락: {names} → 다수 쪽 노드로 라우팅",
                    f"partition variables missing: {names} → routed to the larger child",
                )
            )
        if math.isfinite(row.beta):
            bfix.append(row.beta)
        else:
            bfix.append(1.0)
            reasons.append(tr(settings, "베타 누락 → β=1.0 가정", "beta missing → β=1.0 assumed"))
        if not math.isfinite(row.bench_vol):
            reasons.append(
                tr(
                    settings,
                    "벤치마크 변동성 누락 → 유니버스 중앙값 사용",
                    "benchmark volatility missing → universe median used",
                )
            )
        lag = int(pos[last] - pos[row.date])
        if lag >= STALE_SESSIONS:
            reasons.append(
                tr(
                    settings,
                    f"가격 지연: 최종 거래일 {row.date.date()} (시장 대비 {lag}거래일)",
                    f"stale price: last session {row.date.date()} ({lag} sessions behind the market)",
                )
            )
        flags.append(bool(reasons))
        reasons_col.append(reasons)
    pred["z"] = zfix
    pred["v"] = vfix
    pred["scale_h"] = np.exp(pred["v"].to_numpy(dtype="float64") + c_v) * horizon_scale  # v missing -> exp(c_v)
    pred["beta"] = bfix
    pred["degraded"] = flags
    pred["degraded_reasons"] = reasons_col
    n_deg = sum(flags)
    if n_deg:
        log.warning(
            "RSMT-Q %s panel: %d/%d prediction rows degraded",
            panel.market.value,
            n_deg,
            len(pred),
        )

    moments = moments_from_settings(train, settings, panel.market, h)
    benchmark_ewma = {str(r.ticker): float(r.bench_vol) for r in pred.itertuples(index=False)}

    non_feature = {"ticker", "date", "bench", "beta", "bench_vol", "member", *LABEL_COLUMNS}
    feat_cols = [c for c in next(iter(frames.values())).columns if c not in non_feature]
    train_cols = [
        "ticker",
        "date",
        "bench",
        *feat_cols,
        *CAL_TERMS,
        "z",
        "v",
        "beta",
        "bench_vol",
        "t_end",
        "y_e",
        "m",
        "r",
        "w",
        "scale_h",
        "on_grid",
        "block",
    ]
    pred_cols = [
        "ticker",
        "date",
        "bench",
        *feat_cols,
        *CAL_TERMS,
        "z",
        "v",
        "beta",
        "bench_vol",
        "w",
        "scale_h",
        "degraded",
        "degraded_reasons",
    ]
    train = train[train_cols].sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    pred = pred[pred_cols].reset_index(drop=True)

    n_attempted = len(frames) + len([k for k in failed if not k.startswith("benchmark:") and k not in frames])
    if pit:
        notes.append(
            tr(
                settings,
                f"PIT 유니버스: 학습 행은 각 날짜의 구성원만(비구성원 격자 행 {n_non_member}개 제외), "
                f"횡단면 모멘텀 z도 구성원 기준 · 예측 {len(pred)}종목(최신 구성원 + 요청 종목)",
                f"PIT universe: training rows are each date's members only ({n_non_member} non-member grid rows "
                f"excluded), cross-sectional momentum z across members · {len(pred)} names served (latest "
                "members + requested)",
            )
        )
    return PanelData(
        market=panel.market,
        horizon=h,
        train=train,
        pred=pred,
        c_v=c_v,
        market_mu_hist=moments.mu,
        market_nu=moments.nu,
        dropped=dropped,
        benchmark_ewma=benchmark_ewma,
        partition_features=part,
        offset_mu_feature=zname,
        offset_sigma_feature=vname,
        as_of=last,
        grid_dates=grid,
        n_candidates=int(len(cand)),
        market_mu_by_bench=moments.mu_by_bench,
        market_nu_by_bench=moments.nu_by_bench,
        failed=failed,
        n_tickers=n_attempted,
        notes=notes,
        market_mu_raw=moments.mu_hist,
        market_mu_raw_by_bench=moments.mu_hist_by_bench,
        market_mu_prior=moments.mu_prior,
        market_mu_shrink=moments.shrink,
        n_non_member=n_non_member,
        pit=pit,
        overlap_step=overlap,
        block_sessions=h,
    )
