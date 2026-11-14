"""Rules-based NYSE trading calendar (offline): regular holidays + unscheduled closures since 2001.

Used where no price series is at hand: ``available_from`` of SEC filings (filed + 1 session) and the price date of a
fails-to-deliver row (the session before its settlement date). Dates are tz-naive.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

_SPECIAL_CLOSURES = [
    "2001-09-11",
    "2001-09-12",
    "2001-09-13",
    "2001-09-14",
    "2004-06-11",  # Reagan funeral
    "2007-01-02",  # Ford funeral
    "2012-10-29",  # hurricane Sandy
    "2012-10-30",
    "2018-12-05",  # G.H.W. Bush funeral
    "2025-01-09",  # Carter funeral
]


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th (1-based) weekday (Mon=0) of a month; n=-1 = last."""
    if n > 0:
        first = date(year, month, 1)
        shift = (weekday - first.weekday()) % 7
        return date(year, month, 1 + shift + 7 * (n - 1))
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = pd.Timestamp(nxt) - pd.Timedelta(1, "D")
    shift = (last.weekday() - weekday) % 7
    return (last - pd.Timedelta(shift, "D")).date()


def _observed(d: date, saturday_to_friday: bool = True) -> date | None:
    wd = d.weekday()
    if wd == 5:
        return (pd.Timestamp(d) - pd.Timedelta(1, "D")).date() if saturday_to_friday else None
    if wd == 6:
        return (pd.Timestamp(d) + pd.Timedelta(1, "D")).date()
    return d


@lru_cache(maxsize=256)
def nyse_holidays(year: int) -> frozenset[date]:
    days: list[date | None] = [
        _observed(date(year, 1, 1), saturday_to_friday=False),  # no Friday close for a Saturday New Year
        _nth_weekday(year, 1, 0, 3) if year >= 1998 else None,  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        (pd.Timestamp(_easter(year)) - pd.Timedelta(2, "D")).date(),  # Good Friday
        _nth_weekday(year, 5, 0, -1),  # Memorial Day
        _observed(date(year, 6, 19)) if year >= 2022 else None,  # Juneteenth
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),
    ]
    special = [pd.Timestamp(s).date() for s in _SPECIAL_CLOSURES if s.startswith(str(year))]
    return frozenset(d for d in days + special if d is not None and d.year == year)


def nyse_sessions(start: date | pd.Timestamp | str, end: date | pd.Timestamp | str) -> pd.DatetimeIndex:
    """Trading days in [start, end] (weekdays minus NYSE holidays)."""
    days = pd.bdate_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
    if not len(days):
        return days
    hol = set().union(*(nyse_holidays(y) for y in range(days[0].year, days[-1].year + 1)))
    return days[~days.map(lambda d: d.date() in hol)]


def is_session(d: date | pd.Timestamp) -> bool:
    ts = pd.Timestamp(d)
    return ts.weekday() < 5 and ts.date() not in nyse_holidays(ts.year)


def next_session(d: date | pd.Timestamp, n: int = 1) -> pd.Timestamp:
    """The n-th session strictly after ``d``."""
    ts = pd.Timestamp(d).normalize()
    days = nyse_sessions(ts + pd.Timedelta(1, "D"), ts + pd.Timedelta(10 + 2 * n, "D"))
    return days[n - 1]


def prev_session(d: date | pd.Timestamp) -> pd.Timestamp:
    """The last session strictly before ``d``."""
    ts = pd.Timestamp(d).normalize()
    return nyse_sessions(ts - pd.Timedelta(12, "D"), ts - pd.Timedelta(1, "D"))[-1]


def next_sessions(dates: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Vectorised :func:`next_session` (first session strictly after each date)."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if not len(idx):
        return idx
    cal = nyse_sessions(idx.min(), idx.max() + pd.Timedelta(15, "D"))
    pos = cal.searchsorted(idx, side="right")
    return pd.DatetimeIndex(cal[np.minimum(pos, len(cal) - 1)])


def prev_sessions(dates: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Vectorised :func:`prev_session` (last session strictly before each date)."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if not len(idx):
        return idx
    cal = nyse_sessions(idx.min() - pd.Timedelta(15, "D"), idx.max())
    pos = cal.searchsorted(idx, side="left") - 1
    return pd.DatetimeIndex(cal[np.maximum(pos, 0)])


def month_end_sessions(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last session of every calendar month present in ``sessions``."""
    s = pd.Series(sessions, index=sessions)
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).max().to_numpy())
