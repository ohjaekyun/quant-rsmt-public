"""KR point-in-time universe, fundamentals and market caps (see the SHARED CONTRACT in data/pit/__init__.py).

Sources (keyless, probed and independently verified 2026-09-25; docs/sources/pit/kr-*.json):
- Universe / prices: FinanceData/marcap yearly parquet (daily KRX snapshots of every listed stock, incl. later
  delisted ones). Membership on month-end session d = top-n eligible commons by ``settings.pit.kr.rank_by`` using
  only rows <= d. Prices: KRX base-price chain adjusted OHLCV per (code, listing spell); a delisted spell ends at
  its last traded session (정리매매 included).
- Delisting reasons: FDR GitHub cache of KRX MDCSTAT23801 + KIND 상장폐지현황 (metadata only - look-ahead).
- Benchmarks: KOSPI / KOSDAQ from the production KR provider (Naver fchart, <= 3,000 bars); the part before
  Naver's window is back-filled with a marcap cap-weighted replica when it tracks the index (flagged in attrs).
- Fundamentals: OpenDART bulk zips + DART receipt dates (``kr_fund.py``).

Cache layout (``settings.cache_path / 'pit' / 'kr'``, gitignored): marcap/ (raw yearly files + manifest; the current
year is re-downloaded at most once per day), artefacts/<marcap signature>/ (sessions, spells, month-end shares,
daily market caps), universe/<marcap signature>/ (built universes), delist/ (delisting master, refreshed daily),
plus the fundamentals caches described in kr_fund.py. Old signatures are pruned when a new one is written.
"""

from __future__ import annotations

import logging
import pickle
import shutil
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantcio.config import Settings
from quantcio.data.base import clean_prices
from quantcio.data.cache import DiskCache
from quantcio.data.pit import PITFundamentals, PITUniverse
from quantcio.data.pit import kr_delist as kd
from quantcio.data.pit import kr_marcap as km
from quantcio.models import Instrument, Market

log = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")
UNIVERSE_VERSION = 2  # bump when the universe construction changes; 2: price frames carry ``value`` (84.3)
ARTEFACT_VERSION = 1
BENCHMARK_MARKETS = (("KOSPI", "STK"), ("KOSDAQ", "KSQ"))
REPLICA_MIN_CORR = 0.98  # the replica back-fill is used only when its daily returns track the index this well

__all__ = [
    "KRArtefacts",
    "load_kr_pit_universe",
    "load_kr_pit_fundamentals",
    "kr_artefacts",
    "pit_market_cap_kr",
    "pit_valuation_kr",
    "pit_valuation_frame_kr",
]


# ------------------------------------------------------------------------------------------------ helpers


def _now_kst(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(KST)
    return now.astimezone(KST) if now.tzinfo else now.replace(tzinfo=KST)


def kr_root(settings: Settings) -> Path:
    return settings.cache_path / "pit" / "kr"


def _write_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{time.monotonic_ns()}.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _read_pickle(path: Path) -> Any | None:
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:  # noqa: BLE001 - a corrupt cache entry is a miss
        return None


def _prune_siblings(keep: Path) -> None:
    """Delete sibling signature directories (the marcap current-year file changes daily)."""
    parent = keep.parent
    if not parent.exists():
        return
    for d in parent.iterdir():
        if d.is_dir() and d != keep:
            shutil.rmtree(d, ignore_errors=True)


def marcap_files(
    settings: Settings, now: datetime | None = None, session: Any = None, first_year: int | None = None
) -> km.MarcapFiles:
    """The marcap yearly files from pit.kr.start_year, or from `first_year` when that is earlier (the RSMT-Q history
    starts in 1995: rsmt.markets.KR.history_start) - other consumers keep the pit.kr.start_year file set."""
    now = _now_kst(now)
    first = (
        int(settings.pit.kr.start_year) if first_year is None else min(int(settings.pit.kr.start_year), int(first_year))
    )
    years = list(range(first, now.year + 1))
    return km.ensure_marcap_files(kr_root(settings) / "marcap", years, now, session=session)


def load_panel(files: km.MarcapFiles) -> pd.DataFrame:
    df, pref = km.read_marcap(files.paths)
    return km.prepare_commons(df, pref)


# ---------------------------------------------------------------------------------------------- artefacts


@dataclass
class KRArtefacts:
    """Small marcap-derived tables shared by the universe, fundamentals and valuation helpers."""

    signature: str
    sessions: pd.DatetimeIndex  # every KRX session in the loaded marcap years
    spells: pd.DataFrame  # km.spells_table
    month_end: pd.DataFrame  # Code, spell, Date (last session of each month), Stocks, Marcap, pref_marcap, MarketId
    mcap_path: Path | None  # daily market caps (parquet: Code, spell, Date, marcap, marcap_total)
    mcap_frame: pd.DataFrame | None = None  # in-memory copy when the disk cache is disabled


def _artefacts_from_panel(df: pd.DataFrame, signature: str) -> KRArtefacts:
    sessions = km.calendar(df)
    spells = km.spells_table(df)
    s = pd.Series(sessions, index=sessions)
    month_ends = pd.DatetimeIndex(s.groupby(sessions.to_period("M")).max().to_numpy())
    me = df.loc[df["Date"].isin(month_ends), ["Code", "spell", "Date", "Stocks", "Marcap", "pref_marcap", "MarketId"]]
    me = me.assign(Code=me["Code"].astype(str), MarketId=me["MarketId"].astype(str)).reset_index(drop=True)
    return KRArtefacts(signature=signature, sessions=sessions, spells=spells, month_end=me, mcap_path=None)


def _mcap_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Code": df["Code"],
            "spell": df["spell"],
            "Date": df["Date"],
            "marcap": df["Marcap"],
            "marcap_total": df["Marcap"] + df["pref_marcap"],
        }
    )


def _save_artefacts(art: KRArtefacts, df: pd.DataFrame, directory: Path) -> KRArtefacts:
    directory.mkdir(parents=True, exist_ok=True)
    _write_pickle(directory / "small.pkl", (ARTEFACT_VERSION, art.sessions, art.spells, art.month_end))
    mc = _mcap_frame(df)
    mc["Code"] = mc["Code"].cat.reorder_categories(sorted(mc["Code"].cat.categories))
    mc = mc.sort_values(["Code", "Date"], kind="stable")  # contiguous row groups per code -> fast filtered reads
    tmp = directory / f"mcap.{time.monotonic_ns()}.tmp"
    mc.to_parquet(tmp, index=False, row_group_size=250_000)
    tmp.replace(directory / "mcap.parquet")
    _prune_siblings(directory)
    art.mcap_path = directory / "mcap.parquet"
    return art


def kr_artefacts(
    settings: Settings,
    cache: DiskCache,
    files: km.MarcapFiles | None = None,
    panel: pd.DataFrame | None = None,
    now: datetime | None = None,
    session: Any = None,
) -> KRArtefacts:
    """Sessions / spells / month-end shares / daily market caps for the current marcap files (built once per
    marcap signature; the full panel is loaded only on a miss)."""
    files = files or marcap_files(settings, now, session)
    directory = kr_root(settings) / "artefacts" / files.signature
    if cache.enabled and panel is None:
        small = _read_pickle(directory / "small.pkl")
        if small is not None and small[0] == ARTEFACT_VERSION and (directory / "mcap.parquet").exists():
            _, sessions, spells, month_end = small
            return KRArtefacts(files.signature, sessions, spells, month_end, directory / "mcap.parquet")
    df = panel if panel is not None else load_panel(files)
    art = _artefacts_from_panel(df, files.signature)
    if cache.enabled:
        return _save_artefacts(art, df, directory)
    art.mcap_frame = _mcap_frame(df)
    return art


# ------------------------------------------------------------------------------------------ delisting info


def load_delist_info(
    settings: Settings, cache: DiskCache, now: datetime, session: Any = None, since: date | None = None
) -> kd.DelistInfo:
    """FDR + KIND delisting master, re-fetched at most once per day; stale copy (noted) when the fetch fails."""
    since = since or date(int(settings.pit.kr.start_year), 1, 1)
    path = kr_root(settings) / "delist" / "delist_info.pkl"
    cached = _read_pickle(path) if cache.enabled else None
    if isinstance(cached, kd.DelistInfo) and cached.fetched_at == now.date().isoformat():
        return cached
    try:
        if session is None:
            from quantcio.data.http import PoliteSession

            session = PoliteSession()
        info = kd.fetch_delist_info(session, since, now.date())
        if cache.enabled:
            _write_pickle(path, info)
        return info
    except Exception as exc:  # noqa: BLE001 - never fatal: reasons are metadata, regexes still exclude non-stocks
        err = f"{type(exc).__name__}: {str(exc)[:120]}"
        log.warning("KR delisting master fetch failed: %s", err)
        if isinstance(cached, kd.DelistInfo):
            cached.notes = [*cached.notes, f"상장폐지 목록 갱신 실패({err}) → {cached.fetched_at} 수집본 사용"]
            return cached
        return kd.empty_delist_info(
            f"상장폐지 목록 수집 실패({err}) → 폐지 사유 미기재, 비주권 코드는 이름 규칙으로만 제외"
        )


# --------------------------------------------------------------------------------------------- universe


@dataclass
class _UniverseCore:
    membership: pd.DataFrame
    prices: dict[str, pd.DataFrame]
    meta: pd.DataFrame
    entity_market: dict[str, str]  # entity -> 'KOSPI' | 'KOSDAQ' at its last membership
    replica: dict[str, pd.DataFrame]  # benchmark id -> replica index frame
    sessions: pd.DatetimeIndex
    coverage: dict[str, Any]
    notes: list[str]


def _entity_notes(row: dict[str, Any]) -> str:
    parts = []
    if row.get("delisted"):
        cat = kd.CATEGORY_LABELS.get(str(row.get("delist_category") or ""), "")
        when = row.get("delist_date")
        when_s = pd.Timestamp(when).date().isoformat() if when is not None and not pd.isna(when) else "날짜 미확인"
        parts.append(f"상장폐지 {when_s}" + (f"({cat})" if cat else "(사유 미확인)"))
    if row.get("spell", 0):
        parts.append(f"재상장 구간 #{row['spell']}")
    if row.get("reset_after_halt_rows"):
        parts.append(
            f"거래정지 후 기준가 재설정 {row['reset_after_halt_rows']}건(모호 {row['ambiguous_reset_rows']}건)"
        )
    if row.get("trimmed_trailing_halt_rows"):
        parts.append(f"마지막 거래일 이후 정지 {row['trimmed_trailing_halt_rows']}행 제외")
    return "; ".join(parts)


def build_universe_core(
    df: pd.DataFrame,
    n: int,
    start: date | pd.Timestamp,
    end: date | pd.Timestamp | None,
    rank_by: str,
    min_sessions: int,
    exclude_admin: bool,
    delist: kd.DelistInfo,
) -> _UniverseCore:
    """Prepared marcap panel -> membership, adjusted prices, meta, replica indices, coverage and notes."""
    sessions = km.calendar(df)
    spells = km.spells_table(df).set_index("entity")
    rebal = km.month_end_sessions(sessions, start, end)
    notes: list[str] = []
    if len(rebal) == 0:
        raise ValueError(f"리밸런싱 날짜 없음: {start}~{end} (marcap {sessions[0].date()}~{sessions[-1].date()})")
    rank_value = km.add_rank_value(df, rank_by)
    snap = km.snapshot(df, rebal, rank_value)
    reason = km.exclusion_reason(
        snap,
        min_sessions=min_sessions,
        exclude_admin=exclude_admin,
        exclude_codes=delist.nonstock_codes,
        arrant=delist.arrant,
    )
    membership, stats = km.select_membership(snap, reason, n)

    # price frames for every ever-member (code, spell)
    positions = df.groupby("Code", sort=False, observed=True).indices
    need_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "f", "spell", "reset_after_halt", "ambiguous_reset"]
    prices: dict[str, pd.DataFrame] = {}
    for entity in sorted(membership["entity"].unique()):
        code, spell = km.parse_entity(entity)
        rows = df.iloc[positions[code]][need_cols]
        rows = rows[rows["spell"] == spell]
        px = km.entity_price_frame(rows, active=bool(spells.loc[entity, "active"]))
        px.attrs["entity"] = entity
        prices[entity] = px
    no_price = sorted(e for e, px in prices.items() if px.empty)

    # meta
    last_member = membership.sort_values("date").groupby("entity").tail(1).set_index("entity")
    first_member = membership.groupby("entity")["date"].min()
    member_months = membership.groupby("entity").size()
    meta_rows = []
    for entity, px in prices.items():
        sp = spells.loc[entity]
        code, spell = km.parse_entity(entity)
        delisted = not bool(sp["active"])
        match = kd.match_delisting(delist.master, code, sp["last_row_date"]) if delisted else None
        row = {
            "entity": entity,
            "ticker": code,
            "name": str(sp["name"]),
            "market": km.MARKET_NAME.get(str(sp["market_id"]), str(sp["market_id"])),
            "first_date": sp["first_date"],
            "last_date": px.index[-1] if len(px) else pd.NaT,
            "delisted": delisted,
            "delist_date": match["delist_date"] if match else pd.NaT,
            "delist_reason": match["reason"] if match else None,
            "delist_category": match["category"] if match else None,
            "price_source": km.PRICE_SOURCE,
            "cik": None,
            "spell": spell,
            "last_row_date": sp["last_row_date"],
            "first_member_date": first_member.get(entity, pd.NaT),
            "last_member_date": last_member["date"].get(entity, pd.NaT),
            "member_months": int(member_months.get(entity, 0)),
            "reset_after_halt_rows": len(px.attrs.get("reset_after_halt", [])),
            "ambiguous_reset_rows": len(px.attrs.get("ambiguous_reset", [])),
            "trimmed_trailing_halt_rows": int(px.attrs.get("trimmed_trailing_halt_rows", 0)),
        }
        row["notes"] = _entity_notes(row)
        px.attrs["delisted"] = delisted
        if row["notes"]:
            px.attrs["warnings"] = [f"{entity}: {row['notes']}"]
        meta_rows.append(row)
    meta = pd.DataFrame(meta_rows)
    entity_market = {e: str(r["market"]) for e, r in last_member.iterrows()}

    replica = {bench: km.replica_index(df, mid) for bench, mid in BENCHMARK_MARKETS}

    # holding-period exits: a member whose frame ends before the next rebalance (delisted while held)
    nxt = pd.Series(list(rebal[1:]) + [sessions[-1] + pd.Timedelta(1, "D")], index=rebal)
    last_px = pd.Series({e: (px.index[-1] if len(px) else pd.NaT) for e, px in prices.items()})
    mem_last = membership["entity"].map(last_px)
    exits = membership[
        (mem_last < membership["date"].map(nxt)) & membership["entity"].isin(meta.loc[meta.delisted, "entity"])
    ]
    exit_cat = exits["entity"].map(meta.set_index("entity")["delist_category"]).fillna("unknown")

    delisted_meta = meta[meta["delisted"]]
    by_year: dict[str, dict[str, int]] = {}
    for y, g in membership.groupby(membership["date"].dt.year):
        ents = set(g["entity"])
        by_year[str(y)] = {
            "rebalances": int(g["date"].nunique()),
            "member_months": len(g),
            "entities": len(ents),
            "delisted_entities": int(delisted_meta["entity"].isin(ents).sum()),
        }
    reset_rows = int(meta["reset_after_halt_rows"].sum()) if len(meta) else 0
    ambiguous_rows = int(meta["ambiguous_reset_rows"].sum()) if len(meta) else 0
    coverage: dict[str, Any] = {
        "source": "FinanceData/marcap",
        "rank_by": rank_by,
        "n": int(n),
        "rebalance_dates": len(rebal),
        "first_rebalance": rebal[0].date().isoformat(),
        "last_rebalance": rebal[-1].date().isoformat(),
        "marcap_first_date": sessions[0].date().isoformat(),
        "marcap_last_date": sessions[-1].date().isoformat(),
        "member_months": len(membership),
        "member_months_expected": int(n) * len(rebal),
        "member_months_with_prices": int((~membership["entity"].isin(no_price)).sum()),
        "short_rebalances": stats["short_rebalances"],
        "empty_rebalances": stats["empty_rebalances"],
        "entities": len(prices),
        "no_price_entities": no_price,
        "delisted_entities": int(len(delisted_meta)),
        "delisted_by_category": {
            str(k): int(v) for k, v in delisted_meta["delist_category"].fillna("unknown").value_counts().items()
        },
        "delisted_without_reason": sorted(delisted_meta.loc[delisted_meta["delist_category"].isna(), "entity"]),
        "relisted_spell_entities": sorted(e for e in prices if "#" in e),
        "member_exits_before_next_rebalance": len(exits),
        "member_exits_by_category": {str(k): int(v) for k, v in exit_cat.value_counts().items()},
        "by_year": by_year,
        "excluded_candidates": stats["excluded_candidates"],
        "excluded_above_cutoff": stats["excluded_above_cutoff"],
        "reset_after_halt_rows": reset_rows,
        "ambiguous_reset_rows": ambiguous_rows,
        "trimmed_trailing_halt_rows": int(meta["trimmed_trailing_halt_rows"].sum()) if len(meta) else 0,
        "rejected_series": 0,
        "delisting_master": {
            "version": delist.version,
            "rows": int(len(delist.master)),
            "nonstock_codes": len(delist.nonstock_codes),
            "arrant_windows": int(len(delist.arrant)),
        },
    }

    rank_label = "당일 시가총액" if rank_by == "marcap" else "최근 60거래일 평균 시가총액"
    notes.append(
        f"KR 시점별 유니버스: FinanceData/marcap {sessions[0].date()}~{sessions[-1].date()}, "
        f"월말 리밸런싱 {len(rebal)}회({rebal[0].date()}~{rebal[-1].date()}), 상위 {n}종목({rank_label} 기준), "
        f"편입 이력 {len(prices)}개 종목(상장 구간 기준)."
    )
    if len(delisted_meta):
        notes.append(
            f"편입 이력 중 {len(delisted_meta)}개 종목은 이후 상장폐지 — 가격은 정리매매를 포함한 "
            f"마지막 거래일에서 끝남(보유 중 폐지 {len(exits)}건)."
        )
    above = stats["excluded_above_cutoff"]
    if above:
        detail = ", ".join(
            f"{km.EXCLUSION_LABELS.get(k, k)} {v}" for k, v in sorted(above.items(), key=lambda x: -x[1])
        )
        notes.append(f"상위 {n} 경계 안에서 적격성 규칙으로 제외된 행 {sum(above.values())}건: {detail}.")
    if stats["short_rebalances"] or stats["empty_rebalances"]:
        notes.append(
            f"적격 종목이 {n}개 미만인 리밸런싱 {len(stats['short_rebalances'])}회, 비어 있는 리밸런싱 "
            f"{len(stats['empty_rebalances'])}회(상장 {min_sessions}거래일 요건 등)."
        )
    if coverage["delisted_without_reason"]:
        notes.append(f"상장폐지 사유 미확인 {len(coverage['delisted_without_reason'])}개 종목(목록 매칭 실패).")
    if reset_rows:
        notes.append(
            f"거래정지 후 기준가 재설정 {reset_rows}행(주식수 변화 없는 모호한 재설정 {ambiguous_rows}행)은 "
            "KRX 기준가 체인을 그대로 사용 — 표시만 함."
        )
    notes.append(
        "잔여 편향: 가격 수익률(현금배당 미반영), 코스피 관리종목은 marcap에 표시되지 않아 제외하지 못함, "
        "상장폐지 사유는 사후 정보라 선정에 쓰지 않음."
    )
    notes.extend(delist.notes)
    return _UniverseCore(membership, prices, meta, entity_market, replica, sessions, coverage, notes)


def _splice_benchmark(
    primary: pd.DataFrame | None,
    replica: pd.DataFrame | None,
    need_start: pd.Timestamp,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Primary (Naver) index frame, back-filled before its first bar with the replica when it tracks the index."""
    info: dict[str, Any] = {}
    notes: list[str] = []
    if primary is None or primary.empty:
        if replica is None or replica.empty:
            raise RuntimeError(f"{label} 벤치마크 없음")
        out = replica.loc[need_start:].copy()
        out.attrs["source"] = "marcap 시가총액가중 복제지수"
        out.attrs["backfill_until"] = None
        info.update(source="replica", first=out.index[0].date().isoformat())
        notes.append(f"{label} 지수 수집 실패 → marcap 시가총액가중 복제지수로 대체(가격지수, 실제 지수와 소폭 차이).")
        return out, info, notes
    info.update(source=primary.attrs.get("source", ""), first=primary.index[0].date().isoformat())
    first = primary.index[0]
    if first <= need_start or replica is None or replica.empty:
        if first > need_start:
            notes.append(f"{label} 지수가 {first.date()}부터만 있음 → 그 이전 구간은 벤치마크 결측.")
        return primary, info, notes
    common = primary.index.intersection(replica.index)[:500]
    corr = np.nan
    if len(common) >= 60:
        a = np.log(primary.loc[common, "close"]).diff().dropna()
        b = np.log(replica.loc[common, "close"]).diff().dropna()
        corr = float(a.corr(b))
    info["replica_corr"] = None if np.isnan(corr) else round(corr, 4)
    if not (corr >= REPLICA_MIN_CORR):
        notes.append(
            f"{label} 지수가 {first.date()}부터만 있고 복제지수 일치도 부족(상관 {corr:.3f}) → 이전 구간 벤치마크 결측."
        )
        return primary, info, notes
    back = replica.loc[need_start:first].iloc[:-1] if first in replica.index else replica.loc[need_start:first]
    scale = float(primary["close"].iloc[0]) / float(replica["close"].asof(first))
    back = back.copy()
    for c in ("open", "high", "low", "close"):
        back[c] = back[c] * scale
    out = pd.concat([back, primary])
    out.attrs = dict(primary.attrs)
    out.attrs["backfill_until"] = first.date().isoformat()
    out.attrs["backfill_source"] = "marcap 시가총액가중 복제지수"
    info.update(backfill_rows=len(back), backfill_until=first.date().isoformat())
    notes.append(
        f"{label} 지수(네이버 {first.date()}~) 이전 {len(back)}거래일은 marcap 시가총액가중 복제지수로 보충 "
        f"(일간 수익률 상관 {corr:.3f})."
    )
    return out, info, notes


def _load_benchmarks(
    settings: Settings,
    cache: DiskCache,
    core: _UniverseCore,
    provider: Any = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], list[str]]:
    need_start = core.sessions[0]
    if provider is None:
        from quantcio.data.kr import KRProvider

        provider = KRProvider(settings, cache)
    out: dict[str, pd.DataFrame] = {}
    info: dict[str, Any] = {}
    notes: list[str] = []
    for bench, _mid in BENCHMARK_MARKETS:
        inst = Instrument(
            ticker=bench,
            market=Market.KR,
            name=bench,
            exchange=bench,
            currency="KRW",
            benchmark=bench,
            benchmark_name=bench,
        )
        primary = None
        try:
            raw = provider.benchmark(inst, need_start.date())
            primary = clean_prices(raw)
            primary.attrs = {k: v for k, v in raw.attrs.items() if k in ("source", "volume_unit", "warnings")}
            notes.extend(f"{bench}: {w}" for w in raw.attrs.get("warnings", []) or [])
        except Exception as exc:  # noqa: BLE001 - replaced by the replica, noted
            log.warning("KR PIT benchmark %s failed: %s", bench, exc)
            notes.append(f"{bench} 지수 수집 실패: {type(exc).__name__}: {str(exc)[:120]}")
        frame, binfo, bnotes = _splice_benchmark(primary, core.replica.get(bench), need_start, bench)
        out[bench] = frame
        info[bench] = binfo
        notes.extend(bnotes)
    return out, info, notes


def load_kr_pit_universe(
    settings: Settings,
    cache: DiskCache,
    n: int,
    start: date,
    end: date | None = None,
    *,
    session: Any = None,
    now: datetime | None = None,
    provider: Any = None,
    delist: kd.DelistInfo | None = None,
) -> PITUniverse:
    """PIT top-n KOSPI+KOSDAQ universe for month-end rebalance dates in [start, end] (cached per marcap signature).

    ``session`` (HTTP, for marcap / delisting downloads), ``now``, ``provider`` (benchmark source, default
    KRProvider) and ``delist`` (pre-built delisting info) are injection points for tests.
    """
    t0 = time.monotonic()
    now = _now_kst(now)
    cfg = settings.pit.kr
    # a universe starting before pit.kr.start_year loads the earlier yearly files too (from the start year: the marcap
    # files begin in 1995 and a missing past year is an error)
    files = marcap_files(settings, now, session, first_year=pd.Timestamp(start).year)
    delist = delist or load_delist_info(settings, cache, now, session)
    key = DiskCache.key(
        "kr_pit_universe",
        UNIVERSE_VERSION,
        files.signature,
        int(n),
        pd.Timestamp(start).date().isoformat(),
        pd.Timestamp(end).date().isoformat() if end is not None else None,
        cfg.model_dump(),
        delist.version,
    )
    udir = kr_root(settings) / "universe" / files.signature
    core = _read_pickle(udir / f"{key}.pkl") if cache.enabled else None
    cached = isinstance(core, _UniverseCore)
    if not cached:
        panel = load_panel(files)
        core = build_universe_core(
            panel, n, start, end, cfg.rank_by, int(cfg.min_sessions), bool(cfg.exclude_admin), delist
        )
        if cache.enabled:
            kr_artefacts(settings, cache, files=files, panel=panel)
            _write_pickle(udir / f"{key}.pkl", core)
            _prune_siblings(udir)
        del panel
    benchmarks, bench_info, bench_notes = _load_benchmarks(settings, cache, core, provider)
    coverage = dict(core.coverage)
    coverage["benchmarks"] = bench_info
    coverage["marcap_files"] = {"years": files.years, "signature": files.signature}
    coverage["from_cache"] = cached
    coverage["seconds"] = round(time.monotonic() - t0, 2)
    notes = [*files.notes, *core.notes, *bench_notes]
    membership = core.membership.copy()
    prices = {e: px.copy() for e, px in core.prices.items()}
    for e, px in core.prices.items():
        prices[e].attrs = {k: (list(v) if isinstance(v, list) else v) for k, v in px.attrs.items()}
    entity_benchmark = {e: core.entity_market.get(e, "KOSPI") for e in prices}
    return PITUniverse(
        market=Market.KR,
        n=int(n),
        membership=membership,
        prices=prices,
        benchmarks=benchmarks,
        entity_benchmark=entity_benchmark,
        meta=core.meta.copy(),
        coverage=coverage,
        notes=notes,
    )


# ------------------------------------------------------------------------------------ market caps / valuation


def pit_market_cap_kr(
    settings: Settings,
    cache: DiskCache,
    entities: list[str],
    dates: list | pd.DatetimeIndex,
    *,
    include_preferred: bool = True,
    now: datetime | None = None,
    session: Any = None,
    artefacts: KRArtefacts | None = None,
) -> pd.DataFrame:
    """Market cap (KRW) of each entity as of each date: the official marcap on the last session <= t of the SAME
    listing spell, NaN before its first row or after the last row of an ended spell (never carried past a
    delisting); a still-listed entity keeps its latest value for dates after the last marcap session.

    ``include_preferred`` adds the issuer's preferred lines (NI attributable to owners of the parent belongs to
    common AND preferred holders, so E/P and B/P should use the total). Returns a wide frame (index dates, columns
    entities). Use it to compute E/P, B/P etc. from PIT fundamentals - never EPS/BPS (as-reported, unadjusted).
    """
    art = artefacts or kr_artefacts(settings, cache, now=now, session=session)
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
    codes = sorted({km.parse_entity(e)[0] for e in entities})
    col = "marcap_total" if include_preferred else "marcap"
    if art.mcap_frame is not None:
        mc = art.mcap_frame[art.mcap_frame["Code"].isin(codes)]
    else:
        mc = pd.read_parquet(art.mcap_path, columns=["Code", "spell", "Date", col], filters=[("Code", "in", codes)])
    out = pd.DataFrame(index=idx, columns=list(entities), dtype=float)
    groups = {k: g for k, g in mc.groupby(["Code", "spell"], observed=True)}
    active = set(art.spells.loc[art.spells["active"].astype(bool), "entity"])
    order = np.argsort(idx.to_numpy())
    sorted_idx = idx[order]
    for e in entities:
        code, spell = km.parse_entity(e)
        g = groups.get((code, spell))
        if g is None or g.empty:
            continue
        s = g.set_index("Date")[col].sort_index()
        vals = s.reindex(sorted_idx, method="ffill").to_numpy(dtype=float)
        outside = sorted_idx < s.index[0]
        if e not in active:
            outside |= sorted_idx > s.index[-1]
        vals[outside] = np.nan
        res = np.empty(len(idx))
        res[order] = vals
        out[e] = res
    return out


def pit_valuation_frame_kr(
    fundamentals: PITFundamentals,
    market_caps: pd.DataFrame,
    max_staleness_days: int | None = None,
) -> pd.DataFrame:
    """Vectorised PIT valuation: for every (date, entity) cell of ``market_caps`` (pit_market_cap_kr output), the
    latest fundamentals row with available_from <= date (ties: latest period_end), dropped when older than
    ``max_staleness_days`` (period_end -> date). Columns: date, entity, market_cap, period_end, available_from,
    ep, bp, roe, op_margin, revenue_yoy, earnings_yoy, per, pbr, lookahead_risk.

    E/P = ni_parent_ttm / market cap, B/P = equity_parent / market cap, ROE = ni_parent_ttm / equity_parent,
    OP margin = op_income_ttm / revenue_ttm, YoY = latest standalone quarter vs the same quarter a year earlier
    (None when the base is <= 0). PER / PBR are NaN when earnings / book value are <= 0.
    """
    stale = int(max_staleness_days if max_staleness_days is not None else 200)
    long = (
        market_caps.rename_axis("date").reset_index().melt(id_vars="date", var_name="entity", value_name="market_cap")
    )
    long = long.dropna(subset=["market_cap"]).sort_values("date")
    panel = fundamentals.panel.dropna(subset=["available_from"])
    panel = panel[panel["entity"].isin(set(long["entity"]))].sort_values(["available_from", "period_end"])
    cols = [
        c
        for c in (
            "entity",
            "available_from",
            "period_end",
            "revenue_ttm",
            "op_income_ttm",
            "ni_parent_ttm",
            "equity_parent",
            "revenue_q",
            "revenue_q_prev_year",
            "ni_parent_q",
            "ni_parent_q_prev_year",
            "lookahead_risk",
        )
        if c in panel.columns
    ]
    if long.empty or panel.empty:
        return pd.DataFrame(columns=["date", "entity", "market_cap", "period_end", "available_from", "ep", "bp"])
    merged = pd.merge_asof(
        long,
        panel[cols].rename(columns={"available_from": "af"}).assign(available_from=lambda x: x["af"]),
        left_on="date",
        right_on="af",
        by="entity",
        direction="backward",
    ).drop(columns=["af"])
    too_old = (merged["date"] - merged["period_end"]).dt.days > stale
    fcols = [c for c in cols if c not in ("entity",)]
    for c in fcols:
        if merged[c].dtype == bool:
            merged[c] = merged[c].astype(object)
    merged.loc[too_old, fcols] = None
    mcap = merged["market_cap"]

    def ratio(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a / b).where(b.abs() > 0)

    merged["ep"] = ratio(merged["ni_parent_ttm"], mcap)
    merged["bp"] = ratio(merged["equity_parent"], mcap)
    merged["per"] = ratio(mcap, merged["ni_parent_ttm"]).where(merged["ni_parent_ttm"] > 0)
    merged["pbr"] = ratio(mcap, merged["equity_parent"]).where(merged["equity_parent"] > 0)
    merged["roe"] = ratio(merged["ni_parent_ttm"], merged["equity_parent"]).where(merged["equity_parent"] > 0)
    merged["op_margin"] = ratio(merged["op_income_ttm"], merged["revenue_ttm"]).where(merged["revenue_ttm"] > 0)
    if "revenue_q_prev_year" in merged:
        base = merged["revenue_q_prev_year"]
        merged["revenue_yoy"] = (merged["revenue_q"] / base - 1.0).where(base > 0)
    if "ni_parent_q_prev_year" in merged:
        base = merged["ni_parent_q_prev_year"]
        merged["earnings_yoy"] = (merged["ni_parent_q"] / base - 1.0).where(base > 0)
    return merged.reset_index(drop=True)


def pit_valuation_kr(
    fundamentals: PITFundamentals,
    entity: str,
    t: date | pd.Timestamp,
    market_cap: float,
    max_staleness_days: int | None = None,
) -> dict[str, Any] | None:
    """Single-cell convenience wrapper of pit_valuation_frame_kr (market_cap from pit_market_cap_kr at t)."""
    ts = pd.Timestamp(t).normalize()
    frame = pit_valuation_frame_kr(
        fundamentals, pd.DataFrame({entity: [market_cap]}, index=pd.DatetimeIndex([ts])), max_staleness_days
    )
    if frame.empty:
        return None
    row = frame.iloc[0].to_dict()
    return row if not pd.isna(row.get("period_end")) else None


def load_kr_pit_fundamentals(
    settings: Settings,
    cache: DiskCache,
    entities: list[str] | None = None,
    **kwargs: Any,
) -> PITFundamentals:
    """PIT fundamentals panel (OpenDART bulk values, DART receipt dates) - see kr_fund.build_kr_pit_fundamentals."""
    from quantcio.data.pit.kr_fund import build_kr_pit_fundamentals

    return build_kr_pit_fundamentals(settings, cache, entities, **kwargs)
