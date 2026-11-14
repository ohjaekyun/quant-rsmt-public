"""Alternative data (docs/MODELING.md section 16): dated events and periodic measurements per PIT entity.

Every source (KR: OpenDART disclosures / insider / 5% holders / employee status, the National Pension Service
workplace file, job postings; US: SEC submissions forms / 8-K items and the Form 3/4/5 insider data sets) is
reduced to two long tables keyed by the modeling entity (KR 6-digit stock code, US ``CIK##########``):

``events``   one row per dated event
    entity      str        modeling entity id
    published   Timestamp  publication DATE in the exchange's local calendar (KR: KST receipt date; US: the
                           America/New_York date of the SEC acceptance time). Normalised (no time of day).
    kind        str        pre-declared event kind of the source (``KINDS`` of the loader module)
    value       float      size of the event (count 1.0, shares changed, ownership change in %p, ...)
    source      str        loader id (``dart_list``, ``dart_insider``, ``sec_forms``, ...)

``measures`` one row per periodic measurement
    entity, published, source   as above
    period_end  Timestamp  end of the measured period (report date / data month)
    item        str        pre-declared measurement name (``headcount``, ``nps_members``, ...)
    value       float

Point-in-time rule (one rule for every source): a row is usable for a grid date t only when
``published < t`` (STRICTLY earlier calendar date). A filing received during the session of t, or after its
close, is first used at the next grid date - never on the date it appeared. Loaders that only know an upper
bound of the publication date (e.g. a monthly file whose release day is unknown) must set ``published`` to a
conservative (late) date and say so in ``notes``.

Nothing is dropped silently: loaders count what they could not fetch / match in ``coverage`` and ``notes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["entity", "published", "kind", "value", "source"]
MEASURE_COLUMNS = ["entity", "published", "period_end", "item", "value", "source"]


def empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity": pd.Series(dtype=object),
            "published": pd.Series(dtype="datetime64[ns]"),
            "kind": pd.Series(dtype=object),
            "value": pd.Series(dtype=float),
            "source": pd.Series(dtype=object),
        }
    )


def empty_measures() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity": pd.Series(dtype=object),
            "published": pd.Series(dtype="datetime64[ns]"),
            "period_end": pd.Series(dtype="datetime64[ns]"),
            "item": pd.Series(dtype=object),
            "value": pd.Series(dtype=float),
            "source": pd.Series(dtype=object),
        }
    )


def normalise_events(df: pd.DataFrame | None) -> pd.DataFrame:
    """Coerce an events frame to ``EVENT_COLUMNS`` dtypes (dates normalised, value float, NaN dates dropped)."""
    if df is None or df.empty:
        return empty_events()
    out = df[EVENT_COLUMNS].copy()
    out["entity"] = out["entity"].astype(str)
    out["published"] = pd.to_datetime(out["published"]).dt.tz_localize(None).dt.normalize()
    out["kind"] = out["kind"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce").astype(float)
    out["source"] = out["source"].astype(str)
    out = out[out["published"].notna()]
    return out.sort_values(["entity", "published", "kind"], kind="stable").reset_index(drop=True)


def normalise_measures(df: pd.DataFrame | None) -> pd.DataFrame:
    """Coerce a measures frame to ``MEASURE_COLUMNS`` dtypes (dates normalised, value float, NaN dates dropped)."""
    if df is None or df.empty:
        return empty_measures()
    out = df[MEASURE_COLUMNS].copy()
    out["entity"] = out["entity"].astype(str)
    for c in ("published", "period_end"):
        out[c] = pd.to_datetime(out[c]).dt.tz_localize(None).dt.normalize()
    out["item"] = out["item"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce").astype(float)
    out["source"] = out["source"].astype(str)
    out = out[out["published"].notna() & out["period_end"].notna()]
    return out.sort_values(["entity", "published", "item"], kind="stable").reset_index(drop=True)


@dataclass
class AltData:
    """Alternative data of one market (see module docstring)."""

    market: str
    events: pd.DataFrame = field(default_factory=empty_events)
    measures: pd.DataFrame = field(default_factory=empty_measures)
    coverage: dict[str, Any] = field(default_factory=dict)  # per source: entities asked / matched / rows / failures
    notes: list[str] = field(default_factory=list)
    # per source (the ``source`` column value): entities whose data of that source was fetched COMPLETELY - for
    # them "no row" means "no event" (feature 0); entities outside get NaN features. A source missing here falls
    # back to "entities with at least one row of the source".
    covered: dict[str, list[str]] = field(default_factory=dict)

    def merged(self, other: AltData) -> AltData:
        """A new AltData with both sources' rows, coverage and notes."""
        return AltData(
            self.market,
            normalise_events(pd.concat([self.events, other.events], ignore_index=True)),
            normalise_measures(pd.concat([self.measures, other.measures], ignore_index=True)),
            {**self.coverage, **other.coverage},
            [*self.notes, *other.notes],
            {**self.covered, **other.covered},
        )

    def covered_through(self, source: str) -> pd.Timestamp | None:
        """Last publication date a source is complete for (``coverage[source]["covered_through"]``, e.g. the
        quarterly SEC insider data sets), None when complete up to the fetch."""
        v = (self.coverage.get(source) or {}).get("covered_through")
        try:
            return pd.Timestamp(v).normalize() if v else None
        except (ValueError, TypeError):
            return None

    def covered_entities(self, source: str) -> set[str]:
        """Entities with complete data of ``source`` (see ``covered``)."""
        if source in self.covered:
            return {str(e) for e in self.covered[source]}
        ev = self.events.loc[self.events["source"] == source, "entity"]
        me = self.measures.loc[self.measures["source"] == source, "entity"]
        return set(ev.astype(str)) | set(me.astype(str))

    @property
    def empty(self) -> bool:
        return self.events.empty and self.measures.empty


def usable(published: pd.Series | np.ndarray, t: pd.Timestamp | np.datetime64) -> np.ndarray:
    """The point-in-time rule: rows published STRICTLY before the calendar date of ``t``."""
    p = np.asarray(pd.to_datetime(published).values, dtype="datetime64[ns]")
    day = np.datetime64(pd.Timestamp(t).normalize(), "ns")
    return p < day


__all__ = [
    "EVENT_COLUMNS",
    "MEASURE_COLUMNS",
    "AltData",
    "empty_events",
    "empty_measures",
    "normalise_events",
    "normalise_measures",
    "usable",
]
