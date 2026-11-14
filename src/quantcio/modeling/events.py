"""Historical market-event calendar: robustness evaluation and candidate training-weight schemes ONLY.

Why events are handled this way (docs/MODELING.md, "사건(이벤트) 처리"):

- Events stay in the data. A war, a tariff shock or a crash is part of the return distribution the model is
  served into; deleting those rows would make every risk number (sigma, the tail nu, VaR, barrier odds) too
  optimistic. The distribution / risk layer (RSMT-Q v1: sigma, nu, VaR, barrier) is always fitted on ALL rows.
- Event dummies are NEVER predictive inputs. Whether a war starts next month is unknowable at t; an
  "event" feature would only encode hindsight. What IS knowable at t is the stress state (VIX level, credit
  spreads, market drawdown, realised volatility, cross-sectional dispersion, macro regime scores) - those are
  features (``modeling.features``).
- For the STOCK-SELECTION model an event window is mostly a common market shock (removed by the beta-adjusted
  excess target and the within-date metrics) plus a burst of idiosyncratic noise. Two training-weight schemes
  are therefore offered as candidates and chosen (or rejected) by inner cross-validation only:
  ``event_down`` (rows whose label window overlaps an acute event window get weight ``EVENT_WEIGHT``) and
  ``winsor`` (per-date winsorised excess targets). ``none`` is the default. The outer folds never see the
  choice being made on their own data.
- ``magnitude`` (protocol v4, docs/MODELING.md 79 - not an event scheme, offered the same way): row weight =
  |y_e - date median| (winsorised per date, per-date mean 1). An above-median classifier trained with it targets
  "conditional mean above the median", i.e. the mean return instead of the probability of beating the median.
- Evaluation: every out-of-sample metric is also reported separately for dates whose label window overlaps an
  event window vs calm dates, and per named event (robustness, not selection).

Two kinds of windows:
- ``EVENTS``: a hand-written calendar of well-known episodes (acute windows used for weighting; long "episode"
  windows only for the robustness tables).
- detected shocks: benchmark days whose log return exceeds ``SHOCK_Z`` trailing standard deviations
  (``detect_shocks``; the trailing sd uses data strictly before the day, so the detection is point-in-time).

Down-weighting for a training cutoff T uses only windows that ENDED before T (known at T).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantcio.models import Market

SHOCK_Z = 5.0  # |daily log return| / trailing 252d sd (known before the day) above this = detected shock
SHOCK_MIN_OBS = 120
EVENT_WEIGHT = 0.25  # training weight of a row whose label window overlaps an acute event window (event_down)
WINSOR_Q = 0.025  # per-date winsorisation quantiles of the excess target (winsor)
SCHEMES = ("none", "event_down", "winsor", "magnitude")


@dataclass(frozen=True)
class Event:
    key: str
    start: str
    end: str
    markets: tuple[str, ...]  # "KR", "US"
    kind: str  # "acute" (weighting + robustness) | "episode" (robustness only)
    label_ko: str
    label_en: str

    @property
    def start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.start)

    @property
    def end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.end)


_BOTH = ("KR", "US")
EVENTS: tuple[Event, ...] = (
    Event(
        "cn_deval_2015", "2015-08-11", "2015-09-30", _BOTH, "acute", "2015 위안화 절하·글로벌 급락", "2015 CNY deval"
    ),
    Event("brexit_2016", "2016-06-23", "2016-07-08", _BOTH, "acute", "2016 브렉시트 투표", "2016 Brexit vote"),
    Event("volmageddon_2018", "2018-02-02", "2018-02-12", _BOTH, "acute", "2018-02 변동성 폭발", "2018-02 Volmageddon"),
    Event(
        "tariff_2018_03",
        "2018-03-22",
        "2018-04-06",
        _BOTH,
        "acute",
        "2018-03 미중 관세 발표",
        "2018-03 US-China tariffs",
    ),
    Event(
        "q4_selloff_2018", "2018-10-01", "2018-12-31", _BOTH, "acute", "2018 4분기 급락(관세·금리)", "2018 Q4 sell-off"
    ),
    Event(
        "tariff_2019_05", "2019-05-05", "2019-06-10", _BOTH, "acute", "2019-05 관세 확전", "2019-05 tariff escalation"
    ),
    Event(
        "tariff_2019_08", "2019-08-01", "2019-09-05", _BOTH, "acute", "2019-08 관세·환율 충돌", "2019-08 tariffs / CNY"
    ),
    Event(
        "us_china_trade_war",
        "2018-03-01",
        "2019-12-13",
        _BOTH,
        "episode",
        "2018~19 미중 무역분쟁(전체)",
        "2018-19 US-China trade war",
    ),
    Event("covid_2020", "2020-02-20", "2020-04-30", _BOTH, "acute", "2020 코로나 급락·반등", "2020 COVID crash"),
    Event(
        "russia_ukraine_2022",
        "2022-02-14",
        "2022-03-31",
        _BOTH,
        "acute",
        "2022-02-24 러시아 우크라이나 침공",
        "2022 Russia-Ukraine invasion",
    ),
    Event(
        "rate_shock_2022_06", "2022-06-01", "2022-06-30", _BOTH, "acute", "2022-06 CPI·금리 충격", "2022-06 rate shock"
    ),
    Event("rate_shock_2022_09", "2022-09-13", "2022-10-14", _BOTH, "acute", "2022-09 금리 충격", "2022-09 rate shock"),
    Event("rate_shock_2022", "2022-01-03", "2022-10-31", _BOTH, "episode", "2022 금리 급등기(전체)", "2022 rate shock"),
    Event("svb_2023", "2023-03-08", "2023-03-31", _BOTH, "acute", "2023-03 SVB 사태", "2023-03 SVB failure"),
    Event(
        "carry_unwind_2024",
        "2024-08-01",
        "2024-08-09",
        _BOTH,
        "acute",
        "2024-08 엔캐리 청산 급락",
        "2024-08 carry unwind",
    ),
    Event(
        "kr_martial_law_2024",
        "2024-12-03",
        "2024-12-31",
        ("KR",),
        "acute",
        "2024-12-03 계엄 사태",
        "2024-12 martial law",
    ),
    Event(
        "us_tariff_2025",
        "2025-04-02",
        "2025-04-30",
        _BOTH,
        "acute",
        "2025-04 미국 관세 충격",
        "2025-04 US tariff shock",
    ),
)


@dataclass
class EventCalendar:
    """Named events of one market + detected benchmark shocks, as closed date windows."""

    market: Market
    windows: pd.DataFrame  # key, start, end, kind ("acute" | "episode" | "shock"), label_ko, label_en
    shocks: pd.DataFrame = field(default_factory=pd.DataFrame)  # date, bench, ret, z

    def acute(self) -> pd.DataFrame:
        return self.windows[self.windows["kind"].isin(["acute", "shock"])]

    def overlap_mask(
        self,
        start: pd.Series | np.ndarray,
        end: pd.Series | np.ndarray,
        kinds: tuple[str, ...] = ("acute", "shock"),
        known_before: pd.Timestamp | None = None,
    ) -> np.ndarray:
        """True where [start, end] overlaps any window of ``kinds`` (optionally only windows ended before
        ``known_before`` - a training cutoff may only use events that were over by then)."""
        s = pd.DatetimeIndex(pd.to_datetime(np.asarray(start))).to_numpy(dtype="datetime64[ns]")
        e = pd.DatetimeIndex(pd.to_datetime(np.asarray(end))).to_numpy(dtype="datetime64[ns]")
        out = np.zeros(len(s), dtype=bool)
        w = self.windows[self.windows["kind"].isin(kinds)]
        if known_before is not None:
            w = w[w["end"] < pd.Timestamp(known_before)]
        valid = ~(np.isnat(s) | np.isnat(e))
        for ws, we in zip(w["start"].to_numpy(dtype="datetime64[ns]"), w["end"].to_numpy(dtype="datetime64[ns]")):
            out |= valid & (s <= we) & (e >= ws)
        return out

    def keys_overlapping(self, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
        w = self.windows
        hit = (w["start"] <= pd.Timestamp(end)) & (w["end"] >= pd.Timestamp(start))
        return w.loc[hit, "key"].tolist()


def detect_shocks(
    benchmarks: dict[str, pd.DataFrame], z: float = SHOCK_Z, min_obs: int = SHOCK_MIN_OBS
) -> pd.DataFrame:
    """Benchmark days with |log return| > z * trailing 252-session sd (sd from days strictly before)."""
    rows = []
    for bench, df in sorted(benchmarks.items()):
        if df is None or df.empty or "close" not in df:
            continue
        r = np.log(df["close"].astype(float)).diff()
        sd = r.rolling(252, min_periods=min_obs).std().shift(1)
        zz = r / sd
        hit = zz.abs() > z
        for d in r.index[hit.fillna(False).to_numpy()]:
            rows.append({"date": pd.Timestamp(d), "bench": str(bench), "ret": float(r[d]), "z": float(zz[d])})
    return pd.DataFrame(rows, columns=["date", "bench", "ret", "z"])


def build_calendar(market: Market, benchmarks: dict[str, pd.DataFrame] | None = None) -> EventCalendar:
    """Named events of ``market`` plus the detected shocks of its benchmarks (each shock = a one-day window)."""
    mk = Market(market).value
    rows = [
        {
            "key": ev.key,
            "start": ev.start_ts,
            "end": ev.end_ts,
            "kind": ev.kind,
            "label_ko": ev.label_ko,
            "label_en": ev.label_en,
        }
        for ev in EVENTS
        if mk in ev.markets
    ]
    shocks = detect_shocks(benchmarks or {})
    for d, g in shocks.groupby("date"):
        desc = ", ".join(f"{b} {100 * r:+.1f}%" for b, r in zip(g["bench"], g["ret"]))
        rows.append(
            {
                "key": f"shock_{pd.Timestamp(d).date().isoformat()}",
                "start": pd.Timestamp(d),
                "end": pd.Timestamp(d),
                "kind": "shock",
                "label_ko": f"{pd.Timestamp(d).date()} 지수 급변({desc})",
                "label_en": f"{pd.Timestamp(d).date()} index shock ({desc})",
            }
        )
    windows = pd.DataFrame(rows, columns=["key", "start", "end", "kind", "label_ko", "label_en"])
    windows = windows.sort_values(["start", "key"], kind="stable").reset_index(drop=True)
    return EventCalendar(market=Market(market), windows=windows, shocks=shocks)


def scheme_weights(
    frame: pd.DataFrame, scheme: str, calendar: EventCalendar | None, cutoff: pd.Timestamp | None
) -> np.ndarray:
    """Row multiplier of a training-weight scheme (``winsor`` changes the target, not the weight -> 1)."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown weighting scheme {scheme!r} (allowed {SCHEMES})")
    w = np.ones(len(frame))
    if scheme == "event_down" and calendar is not None and len(frame):
        hit = calendar.overlap_mask(frame["date"], frame["t_end"], known_before=cutoff)
        w[hit] = EVENT_WEIGHT
    if scheme == "magnitude" and len(frame):
        y = winsorised(frame["y_e"], frame["date"])
        dev = (y - y.groupby(frame["date"]).transform("median")).abs()
        mean = dev.groupby(frame["date"]).transform("mean")
        w = (dev / mean.where(mean > 0)).fillna(0.0).to_numpy(dtype=float)
    return w


def winsorised(y: pd.Series, dates: pd.Series, q: float = WINSOR_Q) -> pd.Series:
    """Per-date winsorisation of ``y`` at the q / 1-q quantiles of that date."""
    lo = y.groupby(dates).transform(lambda s: s.quantile(q))
    hi = y.groupby(dates).transform(lambda s: s.quantile(1.0 - q))
    return y.clip(lower=lo, upper=hi)


__all__ = [
    "EVENTS",
    "EVENT_WEIGHT",
    "SCHEMES",
    "SHOCK_Z",
    "Event",
    "EventCalendar",
    "build_calendar",
    "detect_shocks",
    "scheme_weights",
    "winsorised",
]
