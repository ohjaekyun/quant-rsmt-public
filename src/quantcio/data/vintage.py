"""Frozen first-seen copies (vintages, 고정본) of the inputs that sources revise - docs/MODELING.md 84.2.

Sources rewrite history on every download: Yahoo re-scales its dividend-adjusted prices for each new dividend
(rounding noise in every past return - US features moved by ~1e-6 on almost every row from one day to the next), KR
prices are back-adjusted for every later split or bonus issue, Yahoo adds or corrects past dividends, and the macro
history starts at "today - 14 years". Revisions that small flipped noisy inner-CV choices, so the same code picked
another model the next day (docs/MODELING.md 83.3).

Rule: a value dated on or before the store's CUTOFF - the last calendar month-end at least ``SETTLE_DAYS`` days
before today (``cutoff``) - comes from the first stored copy; later values come from the fresh download. When the
cutoff moves to a new month-end, that month is stored as first seen then. After the stored part:

- frames (prices, benchmarks, macro series): ``chain`` columns (adjusted price levels) are the fresh values times
  stored / fresh of the anchor column on the last stored date the fresh frame also has - the fresh series' own
  returns linked to the stored level, so a later re-scaling of the whole fresh history changes nothing; ``inverse``
  columns (share volumes) are divided by that ratio (price x volume stays the traded value); the other columns
  (traded value, yields, index levels) are appended as they are;
- events (dividend yields): the stored events, then the fresh events after the cutoff - a stored key without
  events stays without them;
- table rows (PIT membership): the stored rows, then the fresh rows dated after the cutoff;
- fundamentals panel rows (``freeze_panel``, docs/MODELING.md 84.5): per entity like events - the stored rows usable
  on or before the cutoff, then the fresh rows after it (a filing the source published late is not used).

A key seen for the first time is stored as first seen. Every ignored revision is counted in ``FreezeInfo`` (never
silent). Store: ``<root>/<namespace>.parquet`` (long: key, date, columns) + ``<namespace>.json`` (cutoff, keys, sha1
of the parquet, history). An unreadable or inconsistent store is moved aside and a new vintage starts (logged,
flagged ``restarted``). ``digest_check`` is the tripwire for the inputs that are not frozen (alternative
data): a hash of the frozen window of the finished dataset, compared across builds of the same code.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantcio.models import PRICE_COLUMNS

log = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")
VINTAGE_VERSION = 1
SETTLE_DAYS = 5  # a month is frozen once its last day is at least this many calendar days old
REL_TOL = 1e-9  # a stored and a fresh value further apart than this (relative) count as a revision
PRICE_LEVELS = ("open", "high", "low", "close")  # chained by the close ratio
PRICE_INVERSE = ("volume",)  # share counts: divided by the close ratio (close x volume stays the traded value)
PRICE_FRAME_COLUMNS = (*PRICE_COLUMNS, "value")  # value: the day's own traded value (docs/MODELING.md 84.3)
DIGESTS = "digests.json"
MAX_DIGEST_CUTOFFS = 24  # records kept per dataset identity


def cutoff(today: date | pd.Timestamp | None = None, settle_days: int = SETTLE_DAYS) -> pd.Timestamp:
    """The last calendar month-end at least ``settle_days`` days before ``today`` (default: today in KST)."""
    t = pd.Timestamp(today if today is not None else datetime.now(KST).date()).normalize()
    u = t - pd.Timedelta(int(settle_days), "D") + pd.Timedelta(1, "D")
    return pd.Timestamp(u.to_period("M").start_time) - pd.Timedelta(1, "D")


@dataclass
class FreezeInfo:
    """What one freeze call did (dataset coverage / notes)."""

    namespace: str
    cutoff: str = ""  # the store's cutoff after the call
    created: bool = False  # there was no stored copy: the fresh data up to the cutoff became the vintage
    restarted: bool = False  # the stored copy was unreadable / inconsistent and was moved aside
    extended_from: str = ""  # the previous cutoff when this call stored a new month ("" = no extension)
    keys: int = 0  # keys returned
    new_keys: int = 0  # keys stored as first seen by this call
    store_only: int = 0  # stored keys the fresh data lacks (returned from the store)
    revised: int = 0  # stored rows / events whose fresh value differs (the stored value is used)
    max_rel: float = 0.0  # the largest relative difference among them (inf: one side missing)
    added: int = 0  # fresh rows / events dated <= the stored cutoff that the store does not have (ignored)
    dropped: int = 0  # stored rows / events dated <= the stored cutoff that the fresh data lacks (kept)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["max_rel"] = None if not np.isfinite(self.max_rel) else float(self.max_rel)
        return d

    def _diff(self, d: np.ndarray) -> None:
        bad = d > REL_TOL
        self.revised += int(bad.sum())
        if bad.any():
            self.max_rel = max(self.max_rel, float(d[bad].max()))


class VintageStore:
    """A folder of vintages (module docstring). ``today`` / ``settle_days`` fix this run's cutoff."""

    def __init__(self, root: Path, *, today: date | pd.Timestamp | None = None, settle_days: int = SETTLE_DAYS):
        self.root = Path(root)
        self.cut = cutoff(today, settle_days)

    def _paths(self, ns: str) -> tuple[Path, Path]:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", ns)
        return self.root / f"{safe}.parquet", self.root / f"{safe}.json"

    def read(self, ns: str) -> tuple[pd.DataFrame | None, dict[str, Any], bool]:
        """(stored long frame or None, manifest, restarted). A damaged store is moved aside (``*.bad-<time>``)."""
        data_p, meta_p = self._paths(ns)
        if not data_p.exists() and not meta_p.exists():
            return None, {}, False
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            blob = data_p.read_bytes()
            if meta.get("version") != VINTAGE_VERSION or meta.get("sha1") != hashlib.sha1(blob).hexdigest():
                raise ValueError("version or checksum mismatch")
            data = pd.read_parquet(io.BytesIO(blob))
        except Exception as exc:  # noqa: BLE001 - moved aside and restarted: logged and flagged, never silent
            stamp = time.strftime("%Y%m%d-%H%M%S")
            for p in (data_p, meta_p):
                if p.exists():
                    p.replace(p.with_name(f"{p.name}.bad-{stamp}"))
            log.warning(
                "vintage %s unreadable (%s: %s): moved aside, a new vintage starts", ns, type(exc).__name__, exc
            )
            return None, {}, True
        if "date" in data:
            data["date"] = pd.to_datetime(data["date"]).astype("datetime64[ns]")
        return data, meta, False

    def write(self, ns: str, data: pd.DataFrame, meta: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data_p, meta_p = self._paths(ns)
        buf = io.BytesIO()
        data.to_parquet(buf, index=False)
        blob = buf.getvalue()
        meta = {**meta, "version": VINTAGE_VERSION, "sha1": hashlib.sha1(blob).hexdigest(), "rows": len(data)}
        tag = f"{os.getpid()}.{time.monotonic_ns()}.tmp"
        tmp_d, tmp_m = data_p.with_name(f"{data_p.name}.{tag}"), meta_p.with_name(f"{meta_p.name}.{tag}")
        tmp_d.write_bytes(blob)
        tmp_m.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        os.replace(tmp_d, data_p)
        os.replace(tmp_m, meta_p)


# ------------------------------------------------------------------------------------------------ helpers


def _now() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def _meta(prev: dict[str, Any], cut: pd.Timestamp, keys: Iterable[str], event: dict[str, Any]) -> dict[str, Any]:
    hist = [*prev.get("history", []), {"at": _now(), **event}][-500:]
    return {
        "cutoff": str(pd.Timestamp(cut).date()),
        "keys": sorted({str(k) for k in keys}),
        "created": prev.get("created", _now()),
        "updated": _now(),
        "history": hist,
    }


def _stored_cut(meta: dict[str, Any]) -> pd.Timestamp:
    return pd.Timestamp(meta["cutoff"])


def _rel_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| / |b| (|a - b| where b == 0); both NaN -> 0, exactly one NaN -> inf."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.abs(a - b) / np.where(b != 0, np.abs(b), 1.0)
    na, nb = np.isnan(a), np.isnan(b)
    return np.where(na & nb, 0.0, np.where(na ^ nb, np.inf, d))


def _dt_index(idx: Any) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx)).astype("datetime64[ns]")


def _as_frame(f: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = f.reindex(columns=cols).astype("float64")
    out.index = _dt_index(out.index)
    return out


def _long(key: str, frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x.index = pd.DatetimeIndex(_dt_index(x.index), name="date")
    x = x.reset_index()
    x.insert(0, "key", str(key))
    return x


def _ratio(v: pd.DataFrame, f: pd.DataFrame, col: str) -> float:
    """stored / fresh ``col`` on the last stored date the fresh frame also has with both values > 0 (1.0 if none)."""
    common = v.index.intersection(f.index)
    if not len(common):
        return 1.0
    a = v.loc[common, col].to_numpy(dtype=float)
    b = f.loc[common, col].to_numpy(dtype=float)
    ok = np.flatnonzero(np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0))
    return float(a[ok[-1]] / b[ok[-1]]) if len(ok) else 1.0


# ------------------------------------------------------------------------------------------------ frames


def freeze_frames(
    store: VintageStore,
    ns: str,
    fresh: Mapping[str, pd.DataFrame | None],
    *,
    columns: Sequence[str],
    chain: Sequence[str] = (),
    inverse: Sequence[str] = (),
    anchor: str = "close",
) -> tuple[dict[str, pd.DataFrame], FreezeInfo]:
    """Daily frames per key (entity prices, benchmarks, macro series) through the vintage ``ns`` (module docstring).

    Returned frames have ``columns`` (a column the fresh frame lacks is NaN) and the fresh frame's attrs plus
    ``vintage_cutoff`` / ``vintage_ratio`` (the chain ratio applied after the stored part); a stored key the fresh
    data lacks is returned from the store.
    """
    cols = list(columns)
    data, meta, restarted = store.read(ns)
    info = FreezeInfo(ns, restarted=restarted)
    stored: dict[str, pd.DataFrame] = {}
    known: set[str] = set()
    if data is None:
        info.created = True
        s_cut = store.cut
    else:
        s_cut = _stored_cut(meta)
        known = set(meta.get("keys", []))
        for k, g in data.groupby("key", sort=False):
            stored[str(k)] = g.drop(columns="key").set_index("date").sort_index().reindex(columns=cols)
    new_cut = max(s_cut, store.cut)
    if not info.created and new_cut > s_cut:
        info.extended_from = str(s_cut.date())
    out: dict[str, pd.DataFrame] = {}
    add: list[pd.DataFrame] = []
    new_keys: list[str] = []
    for k in sorted({str(x) for x in fresh} | set(stored)):
        src = fresh.get(k)
        f = _as_frame(src, cols) if src is not None and len(src) else None
        if k not in known:  # first seen (or a new vintage): stored as it is now
            if f is None:
                continue
            new_keys.append(k)
            first = f[f.index <= new_cut]
            if len(first):
                add.append(_long(k, first))
            out[k] = _finish(f, src, new_cut if info.created else s_cut, 1.0)
            continue
        v = stored.get(k, pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"), dtype="float64"))
        if f is None:
            info.store_only += 1
            out[k] = _finish(v.copy(), None, s_cut, 1.0)
            continue
        r = _ratio(v, f, anchor) if len(v) and anchor in cols and (chain or inverse) else 1.0
        g = f.copy()
        for c in chain:
            g[c] = g[c] * r
        for c in inverse:
            g[c] = g[c] / r
        old = g[g.index <= s_cut]
        common = old.index.intersection(v.index)
        if len(common):
            info._diff(_rel_diff(old.loc[common, cols].to_numpy(), v.loc[common, cols].to_numpy()).max(axis=1))
        info.added += len(old.index.difference(v.index))
        info.dropped += len(v.index.difference(old.index))
        tail = g[g.index > s_cut]
        merged = pd.concat([v, tail]) if len(tail) else v.copy()
        if new_cut > s_cut and len(tail):
            ext = tail[tail.index <= new_cut]
            if len(ext):
                add.append(_long(k, ext))
        out[k] = _finish(merged, src, s_cut, r)
    info.new_keys = len(new_keys)
    info.keys = len(out)
    info.cutoff = str(new_cut.date())
    if add or new_keys or info.created or new_cut > s_cut:
        parts = [p for p in ([data] if data is not None else []) + add if len(p)]
        long = (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame({"key": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]")})
        )
        long = long.reindex(columns=["key", "date", *cols]).sort_values(["key", "date"], kind="stable")
        event = {"cutoff": info.cutoff, "new_keys": info.new_keys, "rows_added": int(sum(len(p) for p in add))}
        store.write(ns, long.reset_index(drop=True), _meta(meta, new_cut, known | set(new_keys), event))
    return out, info


def _finish(frame: pd.DataFrame, src: pd.DataFrame | None, cut: pd.Timestamp, ratio: float) -> pd.DataFrame:
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame.index = pd.DatetimeIndex(frame.index)
    frame.index.name = getattr(getattr(src, "index", None), "name", None)  # the fresh frame's (None from the store)
    frame.attrs = {k: (list(v) if isinstance(v, list) else v) for k, v in (getattr(src, "attrs", None) or {}).items()}
    frame.attrs.update(vintage_cutoff=str(pd.Timestamp(cut).date()), vintage_ratio=float(ratio))
    return frame


# ------------------------------------------------------------------------------------------------ events


def freeze_events(
    store: VintageStore,
    ns: str,
    fresh: Mapping[str, pd.Series],
    *,
    keys: Iterable[str],
    new_ok: Iterable[str] | None = None,
) -> tuple[dict[str, pd.Series], FreezeInfo]:
    """Dated events per key (KR ex-dividend yields) through the vintage ``ns``: the stored events dated <= the stored
    cutoff, then the fresh events after it.

    ``keys`` = every key looked up, with or without events, so a stored key without events stays without them;
    ``new_ok`` = the keys whose fresh answer may be stored as first seen (default all: a failed fetch must not be
    frozen as "no events"). Returns only keys with events, like the fresh mapping.
    """
    data, meta, restarted = store.read(ns)
    info = FreezeInfo(ns, restarted=restarted)
    stored: dict[str, pd.Series] = {}
    known: set[str] = set()
    if data is None:
        info.created = True
        s_cut = store.cut
    else:
        s_cut = _stored_cut(meta)
        known = set(meta.get("keys", []))
        for k, g in data.groupby("key", sort=False):
            stored[str(k)] = pd.Series(g["value"].to_numpy(dtype=float), index=_dt_index(g["date"])).sort_index()
    new_cut = max(s_cut, store.cut)
    if not info.created and new_cut > s_cut:
        info.extended_from = str(s_cut.date())
    look = {str(k) for k in keys} | {str(k) for k in fresh}
    allowed = look if new_ok is None else {str(k) for k in new_ok}
    out: dict[str, pd.Series] = {}
    add: list[pd.DataFrame] = []
    new_keys: list[str] = []
    for k in sorted(look):
        src = fresh.get(k)
        f = (
            pd.Series(src.to_numpy(dtype=float), index=_dt_index(src.index)).sort_index()
            if src is not None and len(src)
            else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        )
        if k not in known:
            if k not in allowed:
                if len(f):
                    out[k] = f
                continue
            new_keys.append(k)
            first = f[f.index <= new_cut]
            if len(first):
                add.append(_long(k, first.to_frame("value")))
            if len(f):
                out[k] = f
            continue
        v = stored.get(k, pd.Series(dtype=float, index=pd.DatetimeIndex([])))
        old = f[f.index <= s_cut]
        common = old.index.intersection(v.index)
        if len(common):
            info._diff(_rel_diff(old.loc[common].to_numpy(), v.loc[common].to_numpy()))
        info.added += len(old.index.difference(v.index))
        info.dropped += len(v.index.difference(old.index))
        tail = f[f.index > s_cut]
        merged = pd.concat([v, tail]) if len(tail) else v
        if new_cut > s_cut and len(tail):
            ext = tail[tail.index <= new_cut]
            if len(ext):
                add.append(_long(k, ext.to_frame("value")))
        if len(merged):
            out[k] = merged
    info.store_only = len(set(stored) - look)
    info.new_keys = len(new_keys)
    info.keys = len(out)
    info.cutoff = str(new_cut.date())
    if add or new_keys or info.created or new_cut > s_cut:
        parts = [p for p in ([data] if data is not None else []) + add if len(p)]
        long = (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame({"key": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]")})
        )
        long = long.reindex(columns=["key", "date", "value"]).sort_values(["key", "date"], kind="stable")
        event = {"cutoff": info.cutoff, "new_keys": info.new_keys, "rows_added": int(sum(len(p) for p in add))}
        store.write(ns, long.reset_index(drop=True), _meta(meta, new_cut, known | set(new_keys), event))
    return out, info


# ------------------------------------------------------------------------------------------------ table rows


def _plain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(object)
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    return out


def freeze_rows(
    store: VintageStore,
    ns: str,
    fresh: pd.DataFrame,
    *,
    key: Sequence[str] = ("date", "entity"),
    compare: Sequence[str] = (),
) -> tuple[pd.DataFrame, FreezeInfo]:
    """Dated table rows (PIT membership, column ``date``) through the vintage ``ns``: the stored rows (dated <= the
    stored cutoff), then the fresh rows dated after it. ``compare``: columns checked against the stored values on the
    rows both have (differences counted; the stored row is used)."""
    data, meta, restarted = store.read(ns)
    info = FreezeInfo(ns, restarted=restarted)
    f = _plain(fresh)
    keys = list(key)
    if data is None:
        info.created = True
        first = f[f["date"] <= store.cut]
        event = {"cutoff": str(store.cut.date()), "rows_added": len(first)}
        store.write(ns, first.reset_index(drop=True), _meta({}, store.cut, [], event))
        info.cutoff, info.keys = str(store.cut.date()), len(f)
        return fresh, info
    s_cut = _stored_cut(meta)
    new_cut = max(s_cut, store.cut)
    if new_cut > s_cut:
        info.extended_from = str(s_cut.date())
    v = data.reindex(columns=f.columns)
    old, tail = f[f["date"] <= s_cut], f[f["date"] > s_cut]
    vk, ok = v.set_index(keys), old.set_index(keys)
    vk, ok = vk[~vk.index.duplicated()], ok[~ok.index.duplicated()]
    common = vk.index.intersection(ok.index)
    info.added = len(ok.index.difference(vk.index))
    info.dropped = len(vk.index.difference(ok.index))
    if len(common):
        diff = np.zeros(len(common))
        for c in compare:
            if c not in vk or c not in ok:
                continue
            a, b = ok.loc[common, c], vk.loc[common, c]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                d = _rel_diff(a.to_numpy(dtype=float), b.to_numpy(dtype=float))
            else:
                d = np.where(a.astype(str).to_numpy() != b.astype(str).to_numpy(), np.inf, 0.0)
            diff = np.maximum(diff, d)
        info._diff(diff)
    out = pd.concat([v, tail], ignore_index=True) if len(tail) else v.copy()
    out = out.sort_values("date", kind="stable").reset_index(drop=True)
    for c in out.columns:  # the fresh dtypes where the stored copy came back wider (e.g. int -> float)
        if c in fresh and out[c].dtype != f[c].dtype and out[c].notna().all():
            try:
                out[c] = out[c].astype(f[c].dtype)
            except (TypeError, ValueError):
                pass
    info.cutoff, info.keys = str(new_cut.date()), len(out)
    if new_cut > s_cut:
        ext = tail[tail["date"] <= new_cut]
        long = pd.concat([data, ext.reindex(columns=data.columns)], ignore_index=True) if len(ext) else data
        event = {"cutoff": info.cutoff, "rows_added": len(ext)}
        store.write(ns, long.reset_index(drop=True), _meta(meta, new_cut, [], event))
    return out, info


def freeze_panel(
    store: VintageStore,
    ns: str,
    fresh: pd.DataFrame,
    *,
    date_col: str = "available_from",
    key: Sequence[str] = ("entity", "available_from", "period_end"),
    compare: Sequence[str] = (),
) -> tuple[pd.DataFrame, FreezeInfo]:
    """PIT fundamentals panel rows through the vintage ``ns`` (docs/MODELING.md 84.5): an entity seen before keeps its
    stored rows usable on or before the stored cutoff (``date_col``) and takes the fresh rows after it; an entity
    seen for the first time is stored as it is now. A row the source published late although it is usable on or
    before the cutoff (SEC companyfacts added the July 10-Qs of 29 S&P names only in late October 2026) is counted as
    added and not used; ``compare`` columns that changed on rows both have are counted (the stored row is used).
    Returns the panel with the fresh columns; a stored entity the fresh panel lacks keeps its stored rows."""
    data, meta, restarted = store.read(ns)
    info = FreezeInfo(ns, restarted=restarted)
    f = fresh.copy()
    f[date_col] = pd.to_datetime(f[date_col]).astype("datetime64[ns]")
    ent = f["entity"].astype(str)
    if data is None:
        info.created = True
        s_cut, known = store.cut, set()
    else:
        s_cut, known = _stored_cut(meta), set(meta.get("keys", []))
    new_cut = max(s_cut, store.cut)
    if not info.created and new_cut > s_cut:
        info.extended_from = str(s_cut.date())
    is_known = ent.isin(known).to_numpy()
    first, kf = f[~is_known], f[is_known]
    old, tail = kf[kf[date_col] <= s_cut], kf[kf[date_col] > s_cut]
    v = (data if data is not None else f.iloc[:0]).reindex(columns=f.columns)
    if data is not None:
        v[date_col] = pd.to_datetime(v[date_col]).astype("datetime64[ns]")
    keys = list(key)
    vk, ok = v.set_index(keys), old.set_index(keys)
    vk, ok = vk[~vk.index.duplicated()], ok[~ok.index.duplicated()]
    common = vk.index.intersection(ok.index)
    info.added = len(ok.index.difference(vk.index))
    info.dropped = len(vk.index.difference(ok.index))
    if len(common):
        diff = np.zeros(len(common))
        for c in compare:
            if c in vk and c in ok:
                diff = np.maximum(
                    diff, _rel_diff(ok.loc[common, c].to_numpy(dtype=float), vk.loc[common, c].to_numpy(dtype=float))
                )
        info._diff(diff)
    stored_ents = set(v["entity"].astype(str))
    info.store_only = len(stored_ents - set(ent))
    parts = [p for p in (v, tail, first) if len(p)]
    out = pd.concat(parts, ignore_index=True) if parts else f.iloc[:0].copy()
    out = out.sort_values(["entity", date_col], kind="stable").reset_index(drop=True)
    for c in out.columns:  # the fresh dtypes where the stored copy came back wider
        if c in f and out[c].dtype != f[c].dtype and out[c].notna().all():
            try:
                out[c] = out[c].astype(f[c].dtype)
            except (TypeError, ValueError):
                pass
    new_ents = sorted(set(first["entity"].astype(str)))
    info.new_keys = len(new_ents)
    info.keys = int(out["entity"].nunique()) if len(out) else 0
    info.cutoff = str(new_cut.date())
    add = [first[first[date_col] <= new_cut]]
    if new_cut > s_cut:
        add.append(tail[tail[date_col] <= new_cut])
    add = [a for a in add if len(a)]
    if add or new_ents or info.created or new_cut > s_cut:
        prev = [data] if data is not None and len(data) else []
        long = pd.concat(prev + add, ignore_index=True) if prev or add else f.iloc[:0].copy()
        event = {"cutoff": info.cutoff, "new_keys": info.new_keys, "rows_added": int(sum(len(a) for a in add))}
        store.write(ns, long.reset_index(drop=True), _meta(meta, new_cut, known | set(new_ents), event))
    return out, info


# ------------------------------------------------------------------------------------------------ inputs


def freeze_universe(store: VintageStore, universe: Any, *, tag: str) -> tuple[Any, dict[str, FreezeInfo]]:
    """A PIT universe (``data.pit.PITUniverse``) through the vintages ``membership_<tag>`` (rows; rank / rank value /
    ticker compared), ``prices_<tag>`` (levels chained, volume inverse, traded value as it is) and
    ``benchmarks_<tag>``."""
    mem, i_m = freeze_rows(
        store, f"membership_{tag}", universe.membership, compare=("rank", "rank_value", "ticker", "name")
    )
    px, i_p = freeze_frames(
        store,
        f"prices_{tag}",
        universe.prices,
        columns=PRICE_FRAME_COLUMNS,
        chain=PRICE_LEVELS,
        inverse=PRICE_INVERSE,
    )
    bench, i_b = freeze_frames(
        store,
        f"benchmarks_{tag}",
        universe.benchmarks,
        columns=PRICE_COLUMNS,
        chain=PRICE_LEVELS,
        inverse=PRICE_INVERSE,
    )
    out = replace(universe, membership=mem, prices=px, benchmarks=bench)
    return out, {"membership": i_m, "prices": i_p, "benchmarks": i_b}


def freeze_macro(
    store: VintageStore, series: Mapping[str, pd.Series], chain_keys: Iterable[str]
) -> tuple[dict[str, pd.Series], dict[str, FreezeInfo]]:
    """Macro series through the vintages ``macro_chain`` (price kinds: chained by their own returns) and
    ``macro_level`` (yields / index levels: as they are). Plain series named by key, with the fresh attrs."""
    chain = {str(k) for k in chain_keys}
    out: dict[str, pd.Series] = {}
    infos: dict[str, FreezeInfo] = {}
    for ns, keep, link in (("macro_chain", True, ("value",)), ("macro_level", False, ())):
        part = {k: s.to_frame("value") for k, s in series.items() if (str(k) in chain) == keep}
        frames, infos[ns] = freeze_frames(store, ns, part, columns=("value",), chain=link, anchor="value")
        for k, fr in frames.items():
            if (k in chain) != keep:  # a stored key of the other kind (cannot happen unless the kinds changed)
                continue
            s = fr["value"].copy()
            s.name = k
            s.index.name = "date"
            src = series.get(k)
            s.attrs = dict(getattr(src, "attrs", None) or {"source": "vintage", "notes": []})
            s.attrs.update(vintage_cutoff=fr.attrs.get("vintage_cutoff"), vintage_ratio=fr.attrs.get("vintage_ratio"))
            out[k] = s
    return out, infos


# ------------------------------------------------------------------------------------------------ tripwire


def frame_digest(frame: pd.DataFrame, columns: Sequence[str], mask: np.ndarray) -> tuple[str, int]:
    """Bit-exact sha1 (16 hex) of ``columns`` over the masked rows, ordered by (date, entity, is_me)."""
    keys = [c for c in ("date", "entity", "is_me") if c in frame]
    cols = sorted(c for c in columns if c in frame)
    x = frame.loc[np.asarray(mask, dtype=bool), [*keys, *cols]]
    x = x.sort_values(keys, kind="stable").reset_index(drop=True)
    h = pd.util.hash_pandas_object(x, index=False).to_numpy()
    return hashlib.sha1(h.tobytes()).hexdigest()[:16], len(x)


def digest_check(
    store: VintageStore,
    identity: str,
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    labels: Sequence[str],
    cut: pd.Timestamp,
) -> dict[str, Any]:
    """Tripwire for the inputs that are not frozen (docs/MODELING.md 84.2): hash the frozen window of a finished
    dataset - features of the rows dated <= ``cut``, labels of the labelled rows whose window ends <= ``cut`` - and
    compare with the record of the same dataset identity (code + settings). The first build of a cutoff is
    recorded; when the cutoff moved on, the latest earlier cutoff is checked too. Returns {status: first | match |
    mismatch, cutoff, checked: [...]} (mismatch lists the parts that differ)."""
    path = store.root / DIGESTS
    try:
        book = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        book = {}
    rec: dict[str, Any] = book.get(identity, {})
    d = pd.to_datetime(frame["date"]).to_numpy(dtype="datetime64[ns]")
    lab = frame["labelled"].to_numpy(dtype=bool) if "labelled" in frame else np.ones(len(frame), dtype=bool)
    t_end = pd.to_datetime(frame["t_end"]).to_numpy(dtype="datetime64[ns]") if "t_end" in frame else d

    def digests(c: pd.Timestamp) -> dict[str, Any]:
        c64 = np.datetime64(pd.Timestamp(c), "ns")
        fx, nf = frame_digest(frame, features, d <= c64)
        lx, nl = frame_digest(frame, labels, lab & ~np.isnat(t_end) & (t_end <= c64))
        return {"features": fx, "labels": lx, "rows_features": nf, "rows_labels": nl}

    cut_s = str(pd.Timestamp(cut).date())
    checks = [cut_s]
    earlier = sorted(k for k in rec if k < cut_s)
    if cut_s not in rec and earlier:
        checks.insert(0, earlier[-1])
    status, checked, bad = "first", [], []
    for c in checks:
        now = digests(pd.Timestamp(c))
        old = rec.get(c)
        if old is None:
            rec[c] = {**now, "first": _now(), "checks": 0}
            continue
        same = [p for p in ("features", "labels") if old.get(p) == now[p]]
        old["checks"] = int(old.get("checks", 0)) + 1
        old["last"] = _now()
        checked.append(c)
        if len(same) < 2:
            bad.extend(f"{c}:{p}" for p in ("features", "labels") if p not in same)
            old.setdefault("mismatches", []).append({"at": _now(), **now})
    if checked:
        status = "mismatch" if bad else "match"
    book[identity] = {k: rec[k] for k in sorted(rec)[-MAX_DIGEST_CUTOFFS:]}
    store.root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return {"status": status, "cutoff": cut_s, "checked": checked, "mismatch": bad}


__all__ = [
    "PRICE_FRAME_COLUMNS",
    "FreezeInfo",
    "VintageStore",
    "cutoff",
    "digest_check",
    "frame_digest",
    "freeze_events",
    "freeze_frames",
    "freeze_macro",
    "freeze_panel",
    "freeze_rows",
    "freeze_universe",
]
