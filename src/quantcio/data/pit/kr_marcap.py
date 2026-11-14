"""KR point-in-time backbone: FinanceData/marcap daily KRX snapshots.

Ported from the verifier-fixed ``docs/sources/pit/reference/kr_pit_universe.py`` (probe 2026-09-25, gotchas in
``docs/sources/pit/kr-universe.json``). One row per (Date, Code) for EVERY stock listed on that date, incl. names
delisted later, with the official raw OHLC, Volume, Marcap, Stocks, PIT Name, MarketId and KOSDAQ 소속부 (Dept).

Pure functions (``prepare_panel`` ... ``replica_index``) take and return DataFrames so they can be tested on
synthetic frames; ``ensure_marcap_files`` / ``read_marcap`` do the IO.

Conventions:
- A security is keyed by (Code, spell): a new spell starts after a gap > 20 calendar days (relisting of the same
  issuer; codes are never reassigned to another issuer). Returns never cross spells.
- Returns come from the KRX base price (기준가 = Close - Changes): ret = Close / base - 1. The base already reflects
  splits, reverse splits, bonus and rights ex-dates (not cash dividends). ``f = base / prev_close`` is the event
  factor; the adjustment factor of day t is the product of f over the LATER days of the same spell, so adjusted
  levels are back-adjusted (use them for returns only, never as a point-in-time level).
- Eligibility on date t uses only row t and earlier rows of the same spell (``sess_in_spell``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantcio.models import PRICE_COLUMNS

log = logging.getLogger(__name__)

MARCAP_URL = "https://raw.githubusercontent.com/FinanceData/marcap/master/data/marcap-{year}.parquet"
STR_COLS = ("Code", "Name", "MarketId", "Dept")
NUM_COLS = ("Open", "High", "Low", "Close", "Changes", "Volume", "Marcap", "Stocks")
MARCAP_COLS = ("Date", *STR_COLS, *NUM_COLS)
MARKETS = ("STK", "KSQ")  # KONEX (KNX) is out of scope
MARKET_NAME = {"STK": "KOSPI", "KSQ": "KOSDAQ"}
GAP_DAYS = 20  # a gap longer than this (calendar days) starts a new listing spell
PRICE_SOURCE = "FinanceData/marcap (KRX 공식 종가 · 기준가 체인 수정주가, 배당 미반영)"

SPAC_RE = re.compile(r"스팩|기업인수목적|SPAC", re.I)
# ship funds (아시아N호, 동북아N호, 바다로N호, 하이골드N호, 코리아0N호), REIT / infra funds
# (맵스리얼티, KB발해인프라, 코크렙N호, 트러스제7호, 맥쿼리인프라), investment companies (베트남개발1,
# 한국패러랠, 하나니켈N호), oil fund (한국ANKOR유전). '\d+호$' also catches SPACs without '스팩' in the name
# (케이프이에스제4호). Checked on every 2013-2026 name: no false hits.
NONSTOCK_RE = re.compile(
    r"리츠$|^이리츠|REIT|부동산투자회사|인프라투융자|맥쿼리인프라|발해인프라|맵스리얼티|선박투자|유전"
    r"|\d+호$|베트남개발|한국패러랠|하나니켈",
    re.I,
)
ADMIN_DEPT = ("관리종목", "투자주의환기")  # KOSDAQ 소속부 only - KOSPI 관리종목 are NOT flagged in marcap
FOREIGN_DEPT = "외국기업"

# eligibility rules in evaluation order (the first failing rule is the reason counted for a row)
EXCLUSION_LABELS = {
    "foreign": "외국주권·KDR",
    "spac": "스팩",
    "nonstock": "리츠·인프라·선박펀드·투자회사",
    "admin": "관리종목·투자주의환기",
    "liquidation": "정리매매 기간",
    "halted": "거래정지(거래량 0)",
    "seasoning": "상장 거래일 부족",
    "no_rank_value": "순위 변수 결측",
}


# ------------------------------------------------------------------------------------------------ entities


def entity_id(code: str, spell: int) -> str:
    """'<code>' for the first listing spell in the data, '<code>#<spell>' for a relisting (spell >= 1)."""
    return str(code) if int(spell) == 0 else f"{code}#{int(spell)}"


def parse_entity(entity: str) -> tuple[str, int]:
    code, _, spell = str(entity).partition("#")
    return code, int(spell) if spell else 0


# ------------------------------------------------------------------------------------------------- files


@dataclass
class MarcapFiles:
    paths: list[Path]
    years: list[int]
    manifest: dict[str, Any]
    signature: str
    notes: list[str] = field(default_factory=list)


def _is_parquet(path: Path) -> bool:
    try:
        size = path.stat().st_size
        if size < 12:
            return False
        with path.open("rb") as fh:
            head = fh.read(4)
            fh.seek(size - 4)
            tail = fh.read(4)
        return head == b"PAR1" and tail == b"PAR1"
    except OSError:
        return False


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(root: Path) -> dict[str, Any]:
    p = root / "manifest.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    p = root / "manifest.json"
    tmp = p.with_name(f"manifest.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def _file_entry(path: Path, downloaded: datetime | None = None) -> dict[str, Any]:
    st = path.stat()
    when = downloaded or datetime.fromtimestamp(st.st_mtime)
    return {
        "downloaded": when.date().isoformat(),
        "downloaded_at": when.isoformat(timespec="seconds"),
        "size": st.st_size,
        "sha1": _sha1(path),
    }


def _needs_download(year: int, entry: dict[str, Any] | None, valid: bool, today: date) -> bool:
    """Current year: at most once per calendar day. Past year: re-fetch once if the local copy was taken before the
    year's data was final (downloaded <= Dec 31 + 7 days), still at most once per day."""
    if not valid or not entry:
        return True
    downloaded = date.fromisoformat(entry["downloaded"])
    if downloaded >= today:
        return False
    if year >= today.year:
        return True
    return downloaded <= date(year, 12, 31) + pd.Timedelta(7, "D").to_pytimedelta()


def ensure_marcap_files(
    root: Path,
    years: list[int],
    now: datetime,
    session: Any = None,
    pause: float = 0.3,
) -> MarcapFiles:
    """Make sure marcap-<year>.parquet exists for every year (download / refresh as needed, atomically).

    Raises RuntimeError when a PAST year cannot be obtained (the PIT universe would silently lose names); a missing
    CURRENT year (e.g. early January, not yet published) is skipped with a note.
    """
    root.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(root)
    today = now.date()
    notes: list[str] = []
    used: list[int] = []
    paths: list[Path] = []
    changed = False
    for y in years:
        if y > today.year:
            continue
        path = root / f"marcap-{y}.parquet"
        valid = path.exists() and _is_parquet(path)
        entry = manifest.get(str(y))
        if valid and (entry is None or entry.get("size") != path.stat().st_size):
            entry = _file_entry(path)  # seeded / externally copied file: trust its mtime as the download time
            manifest[str(y)] = entry
            changed = True
        if _needs_download(y, entry, valid, today):
            try:
                if session is None:
                    from quantcio.data.http import PoliteSession

                    session = PoliteSession(timeout=300)
                r = session.get(MARCAP_URL.format(year=y), timeout=300)
                r.raise_for_status()
                content = r.content
                if not (content[:4] == b"PAR1" and content[-4:] == b"PAR1"):
                    raise ValueError(f"parquet 아님 ({len(content)} B, {content[:40]!r})")
                tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                tmp.write_bytes(content)
                os.replace(tmp, path)
                entry = _file_entry(path, downloaded=now.replace(tzinfo=None))
                manifest[str(y)] = entry
                changed = True
                valid = True
                if pause:
                    time.sleep(pause)
            except Exception as exc:  # noqa: BLE001 - fall back to the local copy, never silently
                err = f"{type(exc).__name__}: {str(exc)[:120]}"
                if valid:
                    notes.append(f"marcap-{y} 갱신 실패({err}) → {entry.get('downloaded')} 다운로드본 사용")
                    log.warning("marcap-%s refresh failed: %s", y, err)
                elif y == today.year:
                    notes.append(f"marcap-{y} 파일 없음({err}) → {y}년 데이터 제외")
                    log.warning("marcap-%s unavailable: %s", y, err)
                    continue
                else:
                    raise RuntimeError(f"marcap-{y}.parquet 다운로드 실패: {err}") from exc
        used.append(y)
        paths.append(path)
    if changed:
        _write_manifest(root, manifest)
    sig_src = [(y, manifest[str(y)]["size"], manifest[str(y)]["sha1"]) for y in used]
    signature = hashlib.sha1(json.dumps(sig_src).encode()).hexdigest()[:16]
    return MarcapFiles(paths=paths, years=used, manifest=manifest, signature=signature, notes=notes)


def read_marcap(paths: list[Path], markets: tuple[str, ...] = MARKETS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Yearly parquet files -> (common-stock rows in Date order, preferred market cap per (common code, Date)).

    Column types differ between yearly files (Date us/ns, Stocks int/float), so every file is cast to one schema.
    String columns become pandas categoricals (8M rows of object strings would cost > 1 GB) and the preferred
    lines are aggregated in Arrow, so the pandas frame holds commons only.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    tables, prefs = [], []
    for p in paths:
        tb = pq.read_table(p, columns=list(MARCAP_COLS))
        tb = tb.filter(pc.is_in(tb["MarketId"], value_set=pa.array(list(markets))))
        date_col = pc.cast(tb["Date"], pa.timestamp("ns"))
        code = pc.cast(tb["Code"], pa.string())
        arrays = [date_col, code] + [pc.cast(tb[c], pa.string()) for c in STR_COLS[1:]]
        arrays += [pc.cast(tb[c], pa.float64()) for c in NUM_COLS]
        tb = pa.table(arrays, names=list(MARCAP_COLS))
        if len(tb) > 1 and not pc.all(pc.greater_equal(date_col[1:], date_col[:-1])).as_py():
            tb = tb.sort_by([("Date", "ascending")])
        common = pc.ends_with(tb["Code"], "0")
        pref = tb.filter(pc.invert(common))
        if len(pref):
            pcode = pc.binary_join_element_wise(pc.utf8_slice_codeunits(pref["Code"], 0, 5), "0", "")
            agg = pa.table({"Code": pcode, "Date": pref["Date"], "Marcap": pref["Marcap"]})
            prefs.append(agg.group_by(["Code", "Date"]).aggregate([("Marcap", "sum")]))
        tb = tb.filter(common)
        cols = [tb["Date"]] + [pc.dictionary_encode(tb[c]) for c in STR_COLS] + [tb[c] for c in NUM_COLS]
        tables.append(pa.table(cols, names=list(MARCAP_COLS)))
    if not tables:
        raise RuntimeError("marcap 파일 없음")
    table = pa.concat_tables(tables).unify_dictionaries()
    del tables
    df = table.to_pandas(self_destruct=True, split_blocks=True)
    del table
    for c in STR_COLS:
        if not isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype("category")
    pa.default_memory_pool().release_unused()  # return Arrow buffers to the OS (~1.5 GB otherwise kept)
    if prefs:
        pref_df = pa.concat_tables(prefs).to_pandas().rename(columns={"Marcap_sum": "Marcap"})
        pref_df = pref_df.groupby(["Code", "Date"], as_index=False)["Marcap"].sum()
    else:
        pref_df = pd.DataFrame({"Code": pd.Series(dtype=str), "Date": pd.Series(dtype="datetime64[ns]"), "Marcap": []})
    return df, pref_df


# ----------------------------------------------------------------------------------------- panel (pure)


def split_preferred(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw rows (commons + preferreds, any order) -> (commons in Date order, preferred cap per (common code, Date)).

    KRX commons end with '0'; preferred / class shares end with 5/7/9/K/L/M... and map to the common line by
    replacing the last character with '0' (005935 -> 005930, 00088K -> 000880)."""
    codes = raw["Code"].astype(str)
    common = codes.str.endswith("0")
    pref = raw.loc[~common, ["Date", "Marcap"]].assign(Code=codes[~common].str[:-1] + "0")
    pref_df = pref.groupby(["Code", "Date"], as_index=False)["Marcap"].sum()
    df = raw.loc[common].sort_values("Date", kind="stable").reset_index(drop=True)
    for c in STR_COLS:
        df[c] = df[c].astype(str).astype("category") if not isinstance(df[c].dtype, pd.CategoricalDtype) else df[c]
    df["Code"] = df["Code"].cat.remove_unused_categories()
    return df, pref_df


def prepare_panel(raw: pd.DataFrame, gap_days: int = GAP_DAYS) -> pd.DataFrame:
    """Raw marcap rows (MARCAP_COLS, commons and preferreds) -> prepared common-stock panel (see prepare_commons)."""
    raw = raw.loc[raw["MarketId"].astype(str).isin(MARKETS), list(MARCAP_COLS)]
    df, pref_df = split_preferred(raw)
    return prepare_commons(df, pref_df, gap_days)


def prepare_commons(df: pd.DataFrame, pref_df: pd.DataFrame, gap_days: int = GAP_DAYS) -> pd.DataFrame:
    """Common-stock rows in Date order (modified in place and returned) -> panel with:

    spell / sess_in_spell (0-based; the session count makes min_sessions work on one-day slices), ret (KRX base-price
    return, NaN on the first row of a spell), f (base-price event factor, 1.0 = no event), reset_after_halt (event
    right after a zero-volume day), ambiguous_reset (such an event with NO share-count change since the last traded
    day, e.g. SPAC-merger or re-appraisal resets whose holder return is the raw Close/prevClose), ret_alt (ret, but
    the raw return on ambiguous resets) and pref_marcap (market cap of the issuer's preferred lines that day).
    Group-wise shifts follow row order, so the frame must be in Date order (it is never re-sorted: saves a copy).
    """
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("marcap 패널은 날짜 순이어야 합니다")
    code = df["Code"]
    prev_date = df.groupby(code, sort=False, observed=True)["Date"].shift(1)
    dup = prev_date == df["Date"]
    if dup.any():
        log.warning("marcap: %d duplicated (Code, Date) rows dropped", int(dup.sum()))
        df = df.loc[~dup].reset_index(drop=True)
        code = df["Code"]
        prev_date = df.groupby(code, sort=False, observed=True)["Date"].shift(1)
    g = df.groupby(code, sort=False, observed=True)
    prev_close = g["Close"].shift(1)
    prev_vol = g["Volume"].shift(1)
    new_spell = prev_date.isna() | ((df["Date"] - prev_date).dt.days > gap_days)
    del prev_date
    spell = new_spell.astype(np.int16).groupby(code, sort=False, observed=True).cumsum() - 1
    df["spell"] = spell.astype(np.int16)
    del spell
    grp = [code, df["spell"]]
    df["sess_in_spell"] = df.groupby(grp, sort=False, observed=True).cumcount().astype(np.int32)

    base = df["Close"] - df["Changes"]
    ok_prev = prev_close.notna() & ~new_spell
    df["ret"] = (df["Close"] / base - 1.0).where((base > 0) & ok_prev)
    f = (base / prev_close).where((base > 0) & (prev_close > 0) & ok_prev)
    del base
    f = f.where((f - 1.0).abs() > 1e-6, 1.0).fillna(1.0)
    df["f"] = f.astype(float)
    df["reset_after_halt"] = (f != 1.0) & (prev_vol == 0) & ok_prev
    del f, prev_vol
    traded_shares = df["Stocks"].where(df["Volume"] > 0)
    shares_last_traded = traded_shares.groupby(grp, sort=False, observed=True).shift(1)
    shares_last_traded = shares_last_traded.groupby(grp, sort=False, observed=True).ffill()
    df["ambiguous_reset"] = df["reset_after_halt"] & (df["Stocks"] == shares_last_traded)
    del traded_shares, shares_last_traded
    raw_ret = (df["Close"] / prev_close - 1.0).where((prev_close > 0) & ok_prev)
    df["ret_alt"] = df["ret"].where(~df["ambiguous_reset"], raw_ret)
    del raw_ret, prev_close
    df = df.drop(columns=["Changes"])

    pref_col = np.zeros(len(df))
    if len(pref_df):
        positions = df.groupby("Code", sort=False, observed=True).indices
        dates = df["Date"].to_numpy()
        for c, s in pref_df.groupby("Code"):
            pos = positions.get(c)
            if pos is None:
                continue
            pref_col[pos] = s.set_index("Date")["Marcap"].reindex(dates[pos]).fillna(0.0).to_numpy()
    df["pref_marcap"] = pref_col
    return df


def spells_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (Code, spell): entity, first/last row date, first/last traded date, active (reaches the panel's
    last session), last Name / MarketId."""
    last_date = df["Date"].max()
    traded = df["Date"].where(df["Volume"] > 0)
    tmp = df[["Code", "spell", "Date", "Name", "MarketId"]].assign(traded=traded)
    g = tmp.groupby(["Code", "spell"], sort=True, observed=True)
    out = g.agg(
        first_date=("Date", "min"),
        last_row_date=("Date", "max"),
        first_traded=("traded", "min"),
        last_traded=("traded", "max"),
        name=("Name", "last"),
        market_id=("MarketId", "last"),
        rows=("Date", "size"),
    ).reset_index()
    out["Code"] = out["Code"].astype(str)
    out["name"] = out["name"].astype(str)
    out["market_id"] = out["market_id"].astype(str)
    out["active"] = out["last_row_date"] >= last_date
    out["entity"] = [entity_id(c, s) for c, s in zip(out["Code"], out["spell"], strict=True)]
    return out


def calendar(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.sort(df["Date"].unique()))


def month_end_sessions(
    sessions: pd.DatetimeIndex,
    start: date | pd.Timestamp | None = None,
    end: date | pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """Last session of every COMPLETE calendar month within [start, end].

    A month counts as complete only when the data holds a session of a later month (marcap lags ~1 session, so the
    last row of the current month is not yet known to be its last session). Deterministic given the data.
    """
    if len(sessions) == 0:
        return pd.DatetimeIndex([])
    s = pd.Series(sessions, index=sessions)
    ends = s.groupby(sessions.to_period("M")).max()
    last_period = sessions[-1].to_period("M")
    ends = ends[ends.index < last_period]
    out = pd.DatetimeIndex(ends.to_numpy())
    if start is not None:
        out = out[out >= pd.Timestamp(start)]
    if end is not None:
        out = out[out <= pd.Timestamp(end)]
    return out


def add_rank_value(df: pd.DataFrame, rank_by: str, window: int = 60) -> pd.Series:
    """Ranking variable per row: official Marcap on the date, or its trailing mean within the spell."""
    if rank_by == "marcap":
        return df["Marcap"]
    if rank_by != "marcap_mean60":
        raise ValueError(f"알 수 없는 rank_by: {rank_by!r}")
    key = df["Code"].cat.codes.astype(np.int64) * 1000 + df["spell"].astype(np.int64)
    roll = df["Marcap"].groupby(key, sort=False).rolling(window, min_periods=window // 2).mean()
    return roll.reset_index(level=0, drop=True).reindex(df.index)


def snapshot(df: pd.DataFrame, dates: pd.DatetimeIndex, rank_value: pd.Series) -> pd.DataFrame:
    """Rows on the given sessions with the columns eligibility and ranking need (plain strings, small)."""
    m = df["Date"].isin(dates)
    snap = df.loc[m, ["Date", "Code", "spell", "Name", "MarketId", "Dept", "Volume", "Marcap", "Close"]].copy()
    snap["sess_in_spell"] = df.loc[m, "sess_in_spell"].to_numpy()
    snap["rank_value"] = rank_value[m].to_numpy()
    for c in ("Code", "Name", "MarketId", "Dept"):
        snap[c] = snap[c].astype(str).replace({"nan": "", "None": ""})
    return snap.reset_index(drop=True)


def exclusion_reason(
    snap: pd.DataFrame,
    *,
    min_sessions: int,
    exclude_admin: bool,
    exclude_codes: set[str] | frozenset[str] = frozenset(),
    arrant: pd.DataFrame | None = None,
) -> pd.Series:
    """First failing eligibility rule per snapshot row ('' = eligible). Uses only the row itself (PIT)."""
    reason = pd.Series("", index=snap.index, dtype=object)

    def mark(mask: pd.Series, label: str) -> None:
        reason[(reason == "") & mask.fillna(False).astype(bool)] = label

    name = snap["Name"].astype(str)
    dept = snap["Dept"].astype(str)
    code = snap["Code"].astype(str)
    mark(code.str.startswith("9") | dept.str.startswith(FOREIGN_DEPT), "foreign")
    mark(name.str.contains(SPAC_RE) | dept.str.startswith("SPAC"), "spac")
    mark(name.str.contains(NONSTOCK_RE) | code.isin(set(exclude_codes)), "nonstock")
    if exclude_admin:
        mark(dept.str.startswith(ADMIN_DEPT), "admin")
    if arrant is not None and len(arrant):
        a = arrant.dropna(subset=["start"])
        a = a[a["code"].isin(set(code))]
        if len(a):
            j = snap[["Code", "Date"]].reset_index().merge(a, left_on="Code", right_on="code", how="inner")
            end = j["end"].fillna(j["start"] + pd.Timedelta(30, "D"))
            hit = j.loc[(j["Date"] >= j["start"]) & (j["Date"] <= end), "index"].unique()
            mark(pd.Series(snap.index.isin(hit), index=snap.index), "liquidation")
    mark(~(snap["Volume"] > 0), "halted")
    if min_sessions:
        mark(snap["sess_in_spell"] < min_sessions, "seasoning")
    mark(~(snap["rank_value"] > 0), "no_rank_value")
    return reason


def select_membership(snap: pd.DataFrame, reason: pd.Series, n: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Top-n eligible rows per date by rank_value (ties: code ascending) -> MEMBERSHIP_COLUMNS (+ market, spell,
    code). Also returns exclusion statistics: all excluded candidate rows and the ones ranked above the n-th member
    (they would have been members without the rule)."""
    s = snap.assign(reason=reason.to_numpy())
    s = s.sort_values(["Date", "rank_value", "Code"], ascending=[True, False, True], kind="mergesort")
    elig = s[s["reason"] == ""].copy()
    elig["rank"] = elig.groupby("Date").cumcount() + 1
    mem = elig[elig["rank"] <= n].copy()
    cutoff = mem.groupby("Date")["rank_value"].min()
    excl = s[s["reason"] != ""]
    above = excl[excl["rank_value"] >= excl["Date"].map(cutoff).fillna(np.inf)]
    counts = mem.groupby("Date").size()
    stats = {
        "excluded_candidates": {k: int(v) for k, v in excl["reason"].value_counts().items()},
        "excluded_above_cutoff": {k: int(v) for k, v in above["reason"].value_counts().items()},
        "short_rebalances": {d.date().isoformat(): int(c) for d, c in counts[counts < n].items()},
        "empty_rebalances": sorted(
            d.date().isoformat() for d in set(pd.DatetimeIndex(snap["Date"].unique())) - set(counts.index)
        ),
    }
    mem["entity"] = [entity_id(c, sp) for c, sp in zip(mem["Code"], mem["spell"], strict=True)]
    mem["market"] = mem["MarketId"].map(MARKET_NAME).fillna(mem["MarketId"])
    out = mem.rename(columns={"Date": "date", "Code": "ticker", "Name": "name"})[
        ["date", "entity", "ticker", "name", "rank", "rank_value", "market", "spell"]
    ]
    out["rank"] = out["rank"].astype(int)
    out["rank_value"] = out["rank_value"].astype(float)
    return out.reset_index(drop=True), stats


# ------------------------------------------------------------------------------------------ price frames


def entity_price_frame(rows: pd.DataFrame, active: bool) -> pd.DataFrame:
    """Rows of ONE (Code, spell), date-sorted -> PRICE_COLUMNS frame, back-adjusted with the base-price chain.

    Halt rows (Volume 0, O/H/L 0, stale Close) stay as flat bars with volume 0. An ENDED spell (delisted, or
    relisted later) is cut after its last traded session (정리매매 rows included); never forward-filled.
    attrs: reset_after_halt / ambiguous_reset dates, trimmed trailing halt rows.
    """
    rows = rows.sort_values("Date")
    trimmed = 0
    if not active:
        traded = np.flatnonzero(rows["Volume"].to_numpy() > 0)
        keep = traded[-1] + 1 if len(traded) else 0
        trimmed = len(rows) - keep
        rows = rows.iloc[:keep]
    f = rows["f"].to_numpy(dtype=float)
    adj = np.cumprod(f[::-1])[::-1] / f if len(f) else f
    close = rows["Close"].to_numpy(dtype=float)
    cols = {}
    for c in ("Open", "High", "Low"):
        v = rows[c].to_numpy(dtype=float)
        cols[c.lower()] = np.where(v > 0, v, close) * adj
    out = pd.DataFrame(
        {
            "open": cols["open"],
            "high": cols["high"],
            "low": cols["low"],
            "close": close * adj,
            "volume": rows["Volume"].to_numpy(dtype=float),
        },
        index=pd.DatetimeIndex(rows["Date"].to_numpy()),
    )
    ohlc = out[["open", "high", "low", "close"]]
    out["high"] = ohlc.max(axis=1)
    out["low"] = ohlc.min(axis=1)
    out = out[PRICE_COLUMNS].astype("float64")
    # the day's own traded value (KRX close x volume): close x volume above is back-adjusted for LATER events
    # (docs/MODELING.md 84.3)
    out["value"] = close * rows["Volume"].to_numpy(dtype=float)
    out = out[out["close"] > 0]
    out.attrs["source"] = PRICE_SOURCE
    out.attrs["split_adjusted"] = True
    out.attrs["dividend_adjusted"] = False
    out.attrs["reset_after_halt"] = [d.date().isoformat() for d in rows.loc[rows["reset_after_halt"], "Date"]]
    out.attrs["ambiguous_reset"] = [d.date().isoformat() for d in rows.loc[rows["ambiguous_reset"], "Date"]]
    out.attrs["trimmed_trailing_halt_rows"] = int(trimmed)
    out.attrs["warnings"] = []
    return out


# -------------------------------------------------------------------------------------- replica index


def replica_index(df: pd.DataFrame, market_id: str, base: float = 1000.0) -> pd.DataFrame:
    """Cap-weighted chain index of the market's common stocks (weights = previous-day Marcap, returns = base-price
    returns). Used only to back-fill a benchmark before Naver's 3,000-bar window; flat OHLC, volume in 1,000 shares
    (Naver index units)."""
    g = df.groupby(["Code", "spell"], sort=False, observed=True)
    w = g["Marcap"].shift(1)
    m = (df["MarketId"].astype(str) == market_id) & df["ret"].notna() & (w > 0)
    part = pd.DataFrame({"Date": df.loc[m, "Date"], "wr": (w * df["ret"])[m], "w": w[m]})
    agg = part.groupby("Date")[["wr", "w"]].sum()
    r = (agg["wr"] / agg["w"]).reindex(calendar(df)).fillna(0.0)
    level = base * np.cumprod(1.0 + r.to_numpy())
    vol = df.loc[df["MarketId"].astype(str) == market_id].groupby("Date")["Volume"].sum() / 1000.0
    out = pd.DataFrame({"close": level}, index=r.index)
    for c in ("open", "high", "low"):
        out[c] = out["close"]
    out["volume"] = vol.reindex(out.index).fillna(0.0)
    return out[PRICE_COLUMNS].astype("float64")
