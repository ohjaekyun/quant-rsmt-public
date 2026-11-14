"""Tests for the KR point-in-time universe / fundamentals (quantcio.data.pit.kr*).

Offline tests use small synthetic marcap-like frames, synthetic OpenDART bulk zips, a synthetic XBRL instance and
fake HTTP sessions (no network). Tests marked ``live`` build the real top-60 universe 2016-2026 and fundamentals for
5 entities incl. a delisted one (run with ``-m live``).
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from quantcio.config import load_settings
from quantcio.data.base import clean_prices
from quantcio.data.cache import DiskCache
from quantcio.data.pit import FUNDAMENTAL_COLUMNS, MEMBERSHIP_COLUMNS, PITFundamentals
from quantcio.data.pit import kr as kpit
from quantcio.data.pit import kr_bulk as kb
from quantcio.data.pit import kr_dart as kdart
from quantcio.data.pit import kr_delist as kd
from quantcio.data.pit import kr_fund as kf
from quantcio.data.pit import kr_marcap as km
from quantcio.models import PRICE_COLUMNS

KST = kpit.KST
DATES = pd.bdate_range("2019-01-01", periods=420)  # synthetic KRX calendar (2019-01-01 .. 2020-08-10)
N_MEMBERS = 4
MIN_SESSIONS = 20


# ------------------------------------------------------------------------------------------ synthetic marcap


def _code_rows(
    code: str,
    name: str,
    idx: np.ndarray,
    close: float | np.ndarray,
    stocks: float | np.ndarray,
    market: str = "STK",
    dept: str | list[str] | None = None,
    halted: np.ndarray | None = None,
    base: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Rows on DATES[idx]; `base` overrides the KRX base price on given positions (corporate actions / resets)."""
    n = len(idx)
    close = np.broadcast_to(np.asarray(close, float), (n,)).copy()
    stocks = np.broadcast_to(np.asarray(stocks, float), (n,)).copy()
    halted = np.zeros(n, bool) if halted is None else halted
    prev = np.r_[close[0], close[:-1]]
    b = prev.copy()
    for pos, v in (base or {}).items():
        b[pos] = v
    vol = np.where(halted, 0.0, 1000.0)
    ohl = np.where(halted, 0.0, close)
    depts = dept if isinstance(dept, list) else [dept] * n
    return pd.DataFrame(
        {
            "Date": DATES[idx],
            "Code": code,
            "Name": name,
            "MarketId": market,
            "Dept": depts,
            "Open": ohl,
            "High": ohl,
            "Low": ohl,
            "Close": close,
            "Changes": close - b,
            "Volume": vol,
            "Marcap": close * stocks,
            "Stocks": stocks,
        }
    )


def make_raw() -> pd.DataFrame:
    n = len(DATES)
    allx = np.arange(n)
    parts = []
    # A: large cap with a 2:1 split on day 200 (+ a preferred line)
    close_a = np.where(allx < 200, 50_000.0, 25_000.0) * (1 + 0.0005 * allx)
    stocks_a = np.where(allx < 200, 1e8, 2e8)
    parts.append(_code_rows("000010", "대형A", allx, close_a, stocks_a, base={200: close_a[199] / 2}))
    parts.append(_code_rows("000015", "대형A우", allx, 10_000.0, 1e7))
    # B: halted 150-169, then a base reset with no share-count change (ambiguous reset)
    close_b = np.where(allx < 170, 40_000.0, 31_000.0)
    halted_b = (allx >= 150) & (allx < 170)
    parts.append(_code_rows("000020", "중형B", allx, close_b, 5e7, market="KSQ", halted=halted_b, base={170: 30_000.0}))
    # C: delisted - 정리매매 crash on 251-255, trailing halt 256-258, no rows afterwards
    idx_c = np.arange(259)
    close_c = np.where(idx_c <= 250, 30_000.0, 1_000.0)
    close_c[256:] = close_c[255]
    halted_c = idx_c >= 256
    parts.append(_code_rows("000030", "폐지C", idx_c, close_c, 1e8, halted=halted_c))
    # D: relisted - spell 0 = 0..99, gap 100..139, spell 1 = 140..
    idx_d = np.r_[np.arange(100), np.arange(140, n)]
    parts.append(_code_rows("000040", "재상장D", idx_d, 25_000.0, 1e8, market="KSQ"))
    # non-stocks / excluded
    parts.append(_code_rows("000050", "하나35호스팩", allx, 9_000.0, 1e9, market="KSQ", dept="SPAC(소속부없음)"))
    parts.append(_code_rows("000060", "신한알파리츠", allx, 8_000.0, 1e9))
    parts.append(_code_rows("000070", "동북아13호", allx, 7_000.0, 1e9))
    parts.append(_code_rows("900080", "차이나그레이트", allx, 6_000.0, 1e9, market="KSQ", dept="외국기업(소속부없음)"))
    dept_e = ["관리종목(소속부없음)" if 100 <= i <= 200 else "우량기업부" for i in allx]
    parts.append(_code_rows("000090", "관리E", allx, 45_000.0, 1e8, market="KSQ", dept=dept_e))
    # F: IPO on day 300 (largest cap, needs MIN_SESSIONS sessions)
    idx_f = np.arange(300, n)
    parts.append(_code_rows("000100", "신규F", idx_f, 100_000.0, 1e8))
    # fillers
    for code, cap in (("000110", 1.0e12), ("000120", 0.9e12), ("000130", 0.8e12)):
        parts.append(_code_rows(code, f"소형{code[-2]}", allx, 10_000.0, cap / 10_000.0))
    # KONEX row (filtered out)
    parts.append(_code_rows("000140", "코넥스G", allx, 99_000.0, 1e9, market="KNX"))
    raw = pd.concat(parts, ignore_index=True).sort_values(["Date", "Code"]).reset_index(drop=True)
    return raw


def delist_info() -> kd.DelistInfo:
    fdr = pd.DataFrame(
        {
            "Symbol": ["000030", "123450", "000777"],
            "Name": ["폐지C", "아시아99호", "외국주식"],
            "Market": ["KOSPI", "KOSPI", "KOSDAQ"],
            "SecuGroup": ["주권", "선박투자회사", "외국주권"],
            "Kind": ["보통주", None, None],
            "ListingDate": pd.to_datetime(["2000-01-04", "2005-01-04", "2010-01-04"]),
            "DelistingDate": pd.to_datetime([DATES[259], DATES[100], DATES[50]]),
            "Reason": ["감사의견 거절(의견거절)", "존속기간 만료", "신청에 의한 상장폐지"],
            "ArrantEnforceDate": pd.to_datetime([DATES[251], pd.NaT, pd.NaT]),
            "ArrantEndDate": pd.to_datetime([DATES[258], pd.NaT, pd.NaT]),
        }
    )
    kind = pd.DataFrame(
        {
            "code": ["000040", "000020"],
            "name": ["재상장D", "중형B"],
            "delist_date": pd.to_datetime([DATES[100], DATES[60]]),
            "reason": ["피흡수합병", "유가증권시장 상장"],
            "note": ["", ""],
            "market": ["KOSDAQ", "KOSDAQ"],
        }
    )
    return kd.delist_info_from_parts(fdr, "2020-08-01", kind, date(2019, 1, 1), date(2020, 8, 12))


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return km.prepare_panel(make_raw())


def _core(panel: pd.DataFrame, df_override: pd.DataFrame | None = None) -> kpit._UniverseCore:
    return kpit.build_universe_core(
        panel if df_override is None else df_override,
        N_MEMBERS,
        DATES[0],
        None,
        "marcap",
        MIN_SESSIONS,
        True,
        delist_info(),
    )


# ---------------------------------------------------------------------------------------------- marcap panel


def test_prepare_panel_spells_returns_and_flags(panel):
    assert set(panel["Code"].astype(str)) == {
        "000010",
        "000020",
        "000030",
        "000040",
        "000050",
        "000060",
        "000070",
        "900080",
        "000090",
        "000100",
        "000110",
        "000120",
        "000130",
    }  # preferred line and KONEX removed
    assert panel["Date"].is_monotonic_increasing
    a = panel[panel["Code"] == "000010"].reset_index(drop=True)
    # 2:1 split: base = prev/2 -> event factor 0.5, the base-price return is the true (small) move
    assert a.loc[200, "f"] == pytest.approx(0.5)
    assert a.loc[200, "ret"] == pytest.approx(a.loc[200, "Close"] / (a.loc[199, "Close"] / 2) - 1)
    assert abs(a.loc[200, "ret"]) < 0.01
    assert (a.loc[a.index != 200, "f"] == 1.0).all()
    assert a["pref_marcap"].iloc[5] == pytest.approx(1e11)  # 000015 -> 000010
    d = panel[panel["Code"] == "000040"].reset_index(drop=True)
    assert d["spell"].tolist() == [0] * 100 + [1] * (len(d) - 100)
    assert d.loc[100, "sess_in_spell"] == 0 and np.isnan(d.loc[100, "ret"]) and d.loc[100, "f"] == 1.0
    b = panel[panel["Code"] == "000020"].reset_index(drop=True)
    assert bool(b.loc[170, "reset_after_halt"]) and bool(b.loc[170, "ambiguous_reset"])
    assert b.loc[170, "ret"] == pytest.approx(31_000 / 30_000 - 1)
    assert b.loc[170, "ret_alt"] == pytest.approx(31_000 / 40_000 - 1)  # raw holder return on ambiguous resets
    assert int(panel["reset_after_halt"].sum()) == 1


def test_spells_table_and_month_ends(panel):
    sp = km.spells_table(panel).set_index("entity")
    assert {"000040", "000040#1"} <= set(sp.index)
    assert not sp.loc["000040", "active"] and sp.loc["000040#1", "active"]
    assert not sp.loc["000030", "active"] and sp.loc["000030", "last_traded"] == DATES[255]
    me = km.month_end_sessions(km.calendar(panel))
    assert me[0] == pd.Timestamp("2019-01-31")
    assert me[-1] == pd.Timestamp("2020-07-31")  # August 2020 is incomplete in the data
    assert km.month_end_sessions(km.calendar(panel), "2019-03-01", "2019-05-15").tolist() == [
        pd.Timestamp("2019-03-29"),
        pd.Timestamp("2019-04-30"),
    ]


def test_exclusion_rules_are_point_in_time(panel):
    dates = km.month_end_sessions(km.calendar(panel))
    snap = km.snapshot(panel, dates, panel["Marcap"])
    info = delist_info()
    reason = km.exclusion_reason(
        snap,
        min_sessions=MIN_SESSIONS,
        exclude_admin=True,
        exclude_codes=info.nonstock_codes,
        arrant=info.arrant,
    )
    by = snap.assign(reason=reason.to_numpy())
    first = lambda code: set(by.loc[by["Code"] == code, "reason"])  # noqa: E731
    assert first("000050") == {"spac"}
    assert first("000060") == {"nonstock"} and first("000070") == {"nonstock"}
    assert first("900080") == {"foreign"}
    e = by[by["Code"] == "000090"]
    in_admin = e["Date"].between(DATES[100], DATES[200])
    assert set(e.loc[in_admin, "reason"]) == {"admin"} and set(e.loc[~in_admin, "reason"]) == {""}
    f = by[by["Code"] == "000100"]
    assert (f.loc[f["sess_in_spell"] < MIN_SESSIONS, "reason"] == "seasoning").all()
    assert (f.loc[f["sess_in_spell"] >= MIN_SESSIONS, "reason"] == "").all()
    b = by[by["Code"] == "000020"]
    assert set(b.loc[b["Date"].between(DATES[150], DATES[169]), "reason"]) == {"halted"}
    assert "123450" in info.nonstock_codes and "000777" in info.nonstock_codes and "000030" not in info.nonstock_codes


def test_membership_topn_entities_and_spells(panel):
    core = _core(panel)
    mem = core.membership
    assert list(mem.columns[: len(MEMBERSHIP_COLUMNS)]) == MEMBERSHIP_COLUMNS
    assert (mem.groupby("date").size() == N_MEMBERS).all()
    assert mem.groupby("date")["rank"].apply(lambda r: r.tolist() == [1, 2, 3, 4]).all()
    assert not mem["ticker"].isin(["000050", "000060", "000070", "900080", "000140", "000015"]).any()
    # relisted spell: '#1' entity only after it has MIN_SESSIONS sessions in the new spell
    d1 = mem[mem["entity"] == "000040#1"]
    assert len(d1) and d1["date"].min() >= DATES[140 + MIN_SESSIONS]
    assert mem.loc[mem["entity"] == "000040", "date"].max() < DATES[100]
    # delisted C is a member before its 정리매매 and never afterwards
    c = mem[mem["entity"] == "000030"]
    assert len(c) and c["date"].max() < DATES[251]
    # IPO F enters only once seasoned, then ranks first
    f = mem[mem["entity"] == "000100"]
    assert f["date"].min() >= DATES[300 + MIN_SESSIONS] and (f["rank"] == 1).all()
    # admin E excluded while flagged
    e = mem[mem["entity"] == "000090"]
    assert not e["date"].between(DATES[100], DATES[200]).any() and len(e)
    assert core.entity_market["000020"] == "KOSDAQ" and core.entity_market["000010"] == "KOSPI"


def test_membership_has_no_lookahead(panel):
    """Membership on or before date d must not change when rows after d change (or are absent)."""
    full = _core(panel).membership
    cut = DATES[260]
    future = panel["Date"] > cut
    perturbed = panel.copy()
    rng = np.random.default_rng(0)
    perturbed.loc[future, "Marcap"] = perturbed.loc[future, "Marcap"] * rng.uniform(0.01, 100, future.sum())
    perturbed.loc[future, "Volume"] = 0.0
    alt = _core(panel, perturbed).membership
    truncated = _core(panel, panel[~future].reset_index(drop=True)).membership
    last_complete = truncated["date"].max()
    for other in (alt, truncated):
        a = full[full["date"] <= last_complete].reset_index(drop=True)
        b = other[other["date"] <= last_complete].reset_index(drop=True)
        pd.testing.assert_frame_equal(a[["date", "entity", "rank"]], b[["date", "entity", "rank"]])


def test_price_frames_adjusted_spells_and_delisting(panel):
    core = _core(panel)
    px = core.prices
    for e, f in px.items():
        pd.testing.assert_frame_equal(f[PRICE_COLUMNS], clean_prices(f), check_freq=False)  # PRICE_COLUMNS contract
        assert list(f.columns) == [*PRICE_COLUMNS, "value"]  # + the day's own traded value (docs/MODELING.md 84.3)
    a = px["000010"]
    raw_a = panel[panel["Code"] == "000010"].set_index("Date")
    r = a["close"].pct_change().dropna()
    assert np.allclose(r.to_numpy(), raw_a["ret"].iloc[1:].to_numpy(), atol=1e-12)  # no -50% on the split day
    assert a["close"].iloc[-1] == pytest.approx(raw_a["Close"].iloc[-1])  # back-adjusted to the last day
    assert a["close"].iloc[0] == pytest.approx(raw_a["Close"].iloc[0] / 2)
    assert (a["volume"].to_numpy() == raw_a["Volume"].to_numpy()).all()  # volume raw
    # traded value = KRX close x volume of that day, never back-adjusted for the later 2:1 split
    assert np.allclose(a["value"].to_numpy(), (raw_a["Close"] * raw_a["Volume"]).to_numpy())
    assert a["value"].iloc[0] == pytest.approx(2 * a["close"].iloc[0] * a["volume"].iloc[0])
    # delisted: ends at the last TRADED session (정리매매 included), trailing halt rows cut
    c = px["000030"]
    assert c.index[-1] == DATES[255] and c["close"].iloc[-1] == pytest.approx(1_000.0)
    assert c.attrs["trimmed_trailing_halt_rows"] == 3
    # halts stay as flat zero-volume bars; the reset is flagged
    b = px["000020"]
    halt = b.loc[DATES[150] : DATES[169]]
    assert (halt["volume"] == 0).all() and (halt["open"] == halt["close"]).all()
    assert b.attrs["reset_after_halt"] == [DATES[170].date().isoformat()]
    assert b.attrs["ambiguous_reset"] == [DATES[170].date().isoformat()]
    # relisted code: two entities, returns never cross the gap
    d0, d1 = px["000040"], px["000040#1"]
    assert d0.index[-1] == DATES[99] and d1.index[0] == DATES[140]


def test_meta_coverage_and_notes(panel):
    core = _core(panel)
    meta = core.meta.set_index("entity")
    assert meta.loc["000030", "delisted"] and meta.loc["000030", "delist_category"] == "audit_opinion"
    assert meta.loc["000040", "delisted"] and meta.loc["000040", "delist_category"] == "merger"  # KIND-only row
    assert not meta.loc["000010", "delisted"] and pd.isna(meta.loc["000010", "delist_date"])
    assert meta.loc["000040#1", "spell"] == 1
    cov = core.coverage
    assert cov["member_months"] == cov["member_months_expected"] == N_MEMBERS * cov["rebalance_dates"]
    assert cov["member_months_with_prices"] == cov["member_months"] and cov["no_price_entities"] == []
    assert cov["delisted_entities"] == 2 and cov["delisted_without_reason"] == []
    assert cov["relisted_spell_entities"] == ["000040#1"]
    assert cov["member_exits_before_next_rebalance"] >= 1  # C delisted while held
    assert {"spac", "nonstock", "foreign", "admin", "seasoning", "halted"} <= set(cov["excluded_candidates"])
    assert cov["excluded_above_cutoff"]["spac"] > 0
    assert cov["reset_after_halt_rows"] == 1 and cov["ambiguous_reset_rows"] == 1
    assert sum(v["member_months"] for v in cov["by_year"].values()) == cov["member_months"]
    assert any("상장폐지" in n for n in core.notes) and all(isinstance(n, str) for n in core.notes)


def test_replica_index_is_cap_weighted(panel):
    rep = km.replica_index(panel, "KSQ")
    assert list(rep.columns) == PRICE_COLUMNS and len(rep) == len(DATES)
    day = DATES[170]  # only B moves on this day inside KOSDAQ (reset return)
    prev = DATES[169]
    ksq = panel[(panel["MarketId"] == "KSQ") & (panel["Date"] == prev)]
    ksq_today = panel[(panel["MarketId"] == "KSQ") & (panel["Date"] == day) & panel["ret"].notna()]
    w = ksq.set_index("Code")["Marcap"].reindex(ksq_today["Code"])
    expect = float((w.to_numpy() * ksq_today["ret"].to_numpy()).sum() / w.sum())
    assert rep.loc[day, "close"] / rep.loc[prev, "close"] - 1 == pytest.approx(expect)


# ------------------------------------------------------------------------------------------------ marcap files


class _Resp:
    def __init__(self, content: bytes = b"", status: int = 200, text: str | None = None, ctype: str = ""):
        self.content = content
        self.status_code = status
        self.headers = {"Content-Type": ctype}
        self._text = text

    @property
    def text(self) -> str:
        return self._text if self._text is not None else self.content.decode("utf-8", "replace")

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


class _MarcapSession:
    def __init__(self, payload: bytes, fail_years: set[int] | None = None):
        self.payload = payload
        self.fail_years = fail_years or set()
        self.calls: list[str] = []

    def get(self, url: str, **kw):
        self.calls.append(url)
        year = int(url.rsplit("-", 1)[1].split(".")[0])
        if year in self.fail_years:
            return _Resp(b"<html>not found</html>", status=404)
        return _Resp(self.payload)


def test_marcap_refresh_at_most_once_per_day(tmp_path):
    raw = make_raw().head(50)
    sess = _MarcapSession(_parquet_bytes(raw))
    day1 = datetime(2020, 3, 10, 9, 0, tzinfo=KST)
    f1 = km.ensure_marcap_files(tmp_path, [2019, 2020], day1, session=sess, pause=0)
    assert len(sess.calls) == 2 and f1.years == [2019, 2020]
    km.ensure_marcap_files(tmp_path, [2019, 2020], day1.replace(hour=18), session=sess, pause=0)
    assert len(sess.calls) == 2  # same day: nothing re-downloaded
    day2 = datetime(2020, 3, 11, 9, 0, tzinfo=KST)
    f3 = km.ensure_marcap_files(tmp_path, [2019, 2020], day2, session=sess, pause=0)
    assert sess.calls[2:] == [km.MARCAP_URL.format(year=2020)]  # only the current year, once per day
    assert f3.signature == f1.signature  # identical content -> identical signature
    # a past-year file downloaded before the year was final is refreshed once
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["2019"]["downloaded"] = "2019-12-30"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    km.ensure_marcap_files(tmp_path, [2019, 2020], day2, session=sess, pause=0)
    assert sess.calls[-1] == km.MARCAP_URL.format(year=2019)


def test_marcap_download_failures(tmp_path):
    sess = _MarcapSession(_parquet_bytes(make_raw().head(10)), fail_years={2020})
    now = datetime(2020, 1, 3, 9, 0, tzinfo=KST)
    files = km.ensure_marcap_files(tmp_path, [2019, 2020], now, session=sess, pause=0)
    assert files.years == [2019] and any("marcap-2020" in n for n in files.notes)  # current year skipped, noted
    with pytest.raises(RuntimeError, match="marcap-2018"):
        km.ensure_marcap_files(tmp_path, [2018], now, session=_MarcapSession(b"", fail_years={2018}), pause=0)
    # a corrupt local file is re-downloaded; if that fails for a past year the build stops (never silently)
    (tmp_path / "marcap-2019.parquet").write_bytes(b"PAR1 truncated")
    with pytest.raises(RuntimeError, match="marcap-2019"):
        km.ensure_marcap_files(tmp_path, [2019], now, session=_MarcapSession(b"", fail_years={2019}), pause=0)


# ------------------------------------------------------------------------------------------- universe loader


class _FakeProvider:
    """Benchmark source: the replica index scaled, starting on DATES[100] (like Naver's 3,000-bar window)."""

    def __init__(self, panel: pd.DataFrame, fail: bool = False):
        self.rep = {b: km.replica_index(panel, mid) for b, mid in kpit.BENCHMARK_MARKETS}
        self.fail = fail
        self.calls = 0

    def benchmark(self, inst, start):
        self.calls += 1
        if self.fail:
            raise RuntimeError("naver down")
        df = self.rep[inst.benchmark].loc[DATES[100] :].copy() * 2.0
        df.attrs["source"] = f"fake {inst.benchmark}"
        df.attrs["warnings"] = ["3000봉 한도"]
        return df


def _settings(tmp_path: Path):
    return load_settings(
        data={"cache_dir": str(tmp_path / "cache")},
        pit={"kr": {"start_year": 2019, "min_sessions": MIN_SESSIONS}},
    )


def _write_marcap(settings, raw: pd.DataFrame) -> None:
    root = kpit.kr_root(settings) / "marcap"
    root.mkdir(parents=True, exist_ok=True)
    for y, g in raw.groupby(raw["Date"].dt.year):
        g.to_parquet(root / f"marcap-{y}.parquet", index=False)


def test_load_kr_pit_universe_end_to_end_offline(tmp_path, panel):
    settings = _settings(tmp_path)
    _write_marcap(settings, make_raw())
    cache = DiskCache(settings.cache_path)
    now = datetime(2020, 8, 12, 10, 0, tzinfo=KST)
    provider = _FakeProvider(panel)
    u = kpit.load_kr_pit_universe(
        settings, cache, N_MEMBERS, date(2019, 1, 1), now=now, provider=provider, delist=delist_info()
    )
    assert u.n == N_MEMBERS and set(u.membership.columns) >= set(MEMBERSHIP_COLUMNS)
    assert u.coverage["from_cache"] is False and u.coverage["rebalance_dates"] == 19
    assert set(u.prices) == set(u.membership["entity"]) == set(u.meta["entity"])
    assert set(u.entity_benchmark.values()) <= {"KOSPI", "KOSDAQ"}
    for b in ("KOSPI", "KOSDAQ"):
        bench = u.benchmarks[b]
        assert bench.index[0] == DATES[0] and bench.attrs["backfill_until"] == DATES[100].date().isoformat()
        assert bench.loc[DATES[100], "close"] == pytest.approx(provider.rep[b].loc[DATES[100], "close"] * 2)
        assert u.coverage["benchmarks"][b]["replica_corr"] == pytest.approx(1.0)
    assert any("복제지수" in n for n in u.notes)
    panel_out = u.to_price_panel()
    assert set(panel_out.membership["ticker"]) <= set(panel_out.prices)
    first = u.membership["date"].min()
    assert first == pd.Timestamp("2019-01-31") and u.members_on(first - pd.Timedelta(1, "D")) == []
    assert u.members_on(DATES[25]) == u.membership.loc[u.membership["date"] == first, "entity"].tolist()
    # second call: served from the universe cache
    again = kpit.load_kr_pit_universe(
        settings, cache, N_MEMBERS, date(2019, 1, 1), now=now, provider=provider, delist=delist_info()
    )
    assert again.coverage["from_cache"] is True
    pd.testing.assert_frame_equal(again.membership, u.membership)
    # artefacts: market caps (incl. preferred), NaN outside the spell
    mc = kpit.pit_market_cap_kr(
        settings, cache, ["000010", "000030", "000040"], [DATES[10], DATES[258], DATES[300]], now=now
    )
    assert mc.loc[DATES[10], "000010"] == pytest.approx(
        panel.loc[(panel["Code"] == "000010") & (panel["Date"] == DATES[10]), "Marcap"].iloc[0] + 1e11
    )
    assert np.isnan(mc.loc[DATES[300], "000030"]) and np.isnan(mc.loc[DATES[300], "000040"])
    later = kpit.pit_market_cap_kr(settings, cache, ["000010", "000040#1"], [DATES[-1] + pd.Timedelta(3, "D")], now=now)
    assert later.notna().all(axis=None)  # still listed: latest value after the last marcap session
    mc_common = kpit.pit_market_cap_kr(settings, cache, ["000010"], [DATES[10]], include_preferred=False, now=now)
    assert mc_common.iloc[0, 0] == pytest.approx(mc.loc[DATES[10], "000010"] - 1e11)


def test_benchmark_falls_back_to_replica(tmp_path, panel):
    settings = _settings(tmp_path)
    _write_marcap(settings, make_raw())
    u = kpit.load_kr_pit_universe(
        settings,
        DiskCache(settings.cache_path, enabled=False),
        N_MEMBERS,
        date(2019, 6, 1),
        date(2019, 12, 31),
        now=datetime(2020, 8, 12, tzinfo=KST),
        provider=_FakeProvider(panel, fail=True),
        delist=delist_info(),
    )
    assert u.coverage["rebalance_dates"] == 7 and u.coverage["from_cache"] is False
    assert u.benchmarks["KOSPI"].attrs["source"].startswith("marcap")
    assert any("수집 실패" in n for n in u.notes)


# ---------------------------------------------------------------------------------------------- delisting


def test_delisting_master_categories_and_matching():
    info = delist_info()
    m = info.master
    assert set(m["src"]) == {"fdr_krx", "kind_only"}
    assert "123450" not in set(m["code"])  # non-주권 rows are not commons ...
    assert info.version.startswith("fdr:2020-08-01")
    cats = dict(zip(m["code"], m["category"], strict=True))
    assert cats["000030"] == "audit_opinion" and cats["000040"] == "merger" and cats["000020"] == "transfer"
    assert kd.match_delisting(m, "000020", DATES[59]) is None  # a market transfer is not an exit
    hit = kd.match_delisting(m, "000030", DATES[258])
    assert hit is not None and hit["delist_date"] == DATES[259]
    assert kd.delist_category("상장예비심사청구서 미제출") == "spac_expiry"
    kind_html = (
        "<table><tr><th>번호</th><th>회사명</th><th>종목코드</th><th>폐지일자</th><th>폐지사유</th><th>비고</th></tr>"
        "<tr><td>1</td><td>테스트</td><td>12340</td><td>2020-05-06</td><td>감사의견 거절</td><td></td></tr></table>"
    ).encode("euc-kr")
    t = kd.parse_kind_delisted(kind_html)
    assert t["code"].tolist() == ["012340"] and t["delist_date"].iloc[0] == pd.Timestamp("2020-05-06")


def test_delist_info_is_optional(tmp_path):
    class _Down:
        def get(self, *a, **k):
            raise requests.ConnectionError("down")

        post = get

    settings = _settings(tmp_path)
    info = kpit.load_delist_info(settings, DiskCache(settings.cache_path), datetime(2020, 8, 12, tzinfo=KST), _Down())
    assert info.version == "none" and info.nonstock_codes == frozenset() and "수집 실패" in info.notes[0]


# ------------------------------------------------------------------------------------------------- bulk zips

_BS_HEAD = ["", "당기 3분기말", "", "전기말", ""]
_PL_HEAD = ["", "당기 3분기 3개월", "당기 3분기 누적", "", "전기 3분기 3개월", "전기 3분기 누적"]
_CF_HEAD = ["", "당기 3분기", "", "전기 3분기", ""]


def _meta(code: str, name: str, ccy: str = "KRW") -> list[str]:
    return ["표", f"[{code}]", name, "유가증권시장상장법인", "264", "업종", "12", "2020-09-30", "3분기보고서", ccy]


def _line(code: str, name: str, item: str, label: str, values: list[str], ccy: str = "KRW") -> list[str]:
    return [*_meta(code, name, ccy), item, label, *values]


def _member(header_tail: list[str], lines: list[list[str]]) -> bytes:
    head = [
        "재무제표종류",
        "종목코드",
        "회사명",
        "시장구분",
        "업종",
        "업종명",
        "결산월",
        "결산기준일",
        "보고서종류",
        "통화",
        "항목코드",
        "항목명",
        *header_tail,
    ]
    text = "\n".join("\t".join(r) for r in [head, *lines]) + "\n"
    return text.encode("cp949")


def make_bulk_zip() -> bytes:
    bs_con = [
        _line("000010", "연결사", "ifrs-full_Assets", "자산총계", ["", "5,000", "", "9,999", ""]),
        _line("000010", "연결사", "ifrs-full_Equity", "자본총계", ["", "3,000", "", "", ""]),
        _line(
            "000010",
            "연결사",
            "ifrs-full_EquityAttributableToOwnersOfParent",
            "지배기업 소유주지분",
            ["", "2,800", "", "", ""],
        ),
        _line("000010", "연결사", "ifrs-full_NoncontrollingInterests", "비지배지분", ["", "200", "", "", ""]),
        _line("000050", "비지배확장", "ifrs-full_Equity", "자본총계", ["", "900", "", "", ""]),
        _line("000050", "비지배확장", "ifrs-full_NoncontrollingInterests", "비지배지분", ["", "100", "", "", ""]),
    ]
    bs_sep = [
        _line("000010", "연결사", "ifrs-full_Assets", "자산총계", ["", "4,000", "", "", ""]),
        _line("000020", "별도사", "ifrs-full_Assets", "자산총계", ["", "700", "", "", ""]),
        _line("000030", "달러사", "ifrs-full_Assets", "자산총계", ["", "10", "", "", ""], "USD"),
        _line("000060", "무비지배", "ifrs-full_Equity", "자본총계", ["", "500", "", "", ""]),
    ]
    pl_con = [
        _line("000010", "연결사", "ifrs-full_Revenue", "매출액", ["", "400", "1,000", "", "300", "900"]),
        _line("000010", "연결사", "dart_OperatingIncomeLoss", "영업이익", ["", "40", "100", "", "", ""]),
        _line("000010", "연결사", "ifrs-full_ProfitLoss", "당기순이익", ["", "30", "80", "", "", ""]),
        _line(
            "000010", "연결사", "ifrs-full_ProfitLossAttributableToOwnersOfParent", "지배", ["", "28", "75", "", "", ""]
        ),
        _line(
            "000010",
            "연결사",
            "ifrs-full_ProfitLossAttributableToNoncontrollingInterests",
            "비지배",
            ["", "2", "5", "", "", ""],
        ),
        # 000040: parent NI only under an extension code (the CI attribution reuses the label with another code)
        _line("000040", "확장사", "ifrs-full_ProfitLoss", "당기순이익", ["", "", "100", "", "", ""]),
        _line(
            "000040",
            "확장사",
            "ifrs-full_ProfitLossAttributableToNoncontrollingInterests",
            "비지배",
            ["", "", "30", "", "", ""],
        ),
        _line("000040", "확장사", "entity001_udf_IS_ProfitLoss", "지배기업의 소유주지분", ["", "", "70", "", "", ""]),
        _line(
            "000040", "확장사", "entity001_udf_CI_Comprehensive", "지배기업의 소유주지분", ["", "", "999", "", "", ""]
        ),
        # 000050: NCI profit line extension-coded and NCI equity present -> ni_parent must NOT fall back to ni
        _line("000050", "비지배확장", "ifrs-full_ProfitLoss", "당기순이익", ["", "", "60", "", "", ""]),
        _line("000050", "비지배확장", "entity002_udf_NCI", "비지배지분", ["", "", "6", "", "", ""]),
    ]
    pl_sep = [
        _line("000010", "연결사", "ifrs-full_Revenue", "매출액", ["", "300", "800", "", "", ""]),
        _line("000020", "별도사", "ifrs-full_Revenue", "매출액", ["", "50", "150", "", "", ""]),
        _line("000030", "달러사", "ifrs-full_Revenue", "매출액", ["", "1", "3", "", "", ""], "USD"),
        _line("000060", "무비지배", "ifrs-full_ProfitLoss", "당기순이익", ["", "5", "15", "", "", ""]),
    ]
    cf_con = [
        _line(
            "000010",
            "연결사",
            "ifrs-full_CashFlowsFromUsedInOperatingActivities",
            "영업활동현금흐름",
            ["", "90", "", "", ""],
        )
    ]
    members = {
        "2020_3분기보고서_01_재무상태표_연결_20240517.txt": _member(_BS_HEAD, bs_con),
        "2020_3분기보고서_01_재무상태표_20240517.txt": _member(_BS_HEAD, bs_sep),
        "2020_3분기보고서_02_손익계산서_연결_20240517.txt": _member(_PL_HEAD, pl_con),
        "2020_3분기보고서_02_손익계산서_20240517.txt": _member(_PL_HEAD, pl_sep),
        "2020_3분기보고서_04_현금흐름표_연결_20240517.txt": _member(_CF_HEAD, cf_con),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in members.items():
            z.writestr(name, content)
    return buf.getvalue()


def test_read_bulk_zip_and_extract_key_items():
    long, cons = kb.read_bulk_zip(make_bulk_zip())
    assert set(long["col"].str[:2]) == {"당기"}  # 전기 columns never read (position-based header)
    assert ("000010", "2020-09-30", "3분기보고서") in cons and ("000020", "2020-09-30", "3분기보고서") not in cons
    items = kb.extract_key_items(long, cons).set_index("stock_code")
    a = items.loc["000010"]
    assert a["fs"] == "CFS" and a["assets"] == 5000 and a["revenue"] == 1000 and a["revenue_3m"] == 400
    assert a["ni_parent"] == 75 and a["equity_parent"] == 2800 and a["cfo"] == 90
    assert items.loc["000020", "fs"] == "OFS" and items.loc["000020", "revenue"] == 150
    assert items.loc["000030", "currency"] == "USD"
    assert items.loc["000040", "ni_parent"] == 70  # extension code + label, not the 999 CI attribution
    assert np.isnan(items.loc["000050", "ni_parent"])  # NCI exists (equity) -> no ni fallback
    assert items.loc["000060", "ni_parent"] == 15 and items.loc["000060", "equity_parent"] == 500  # no NCI at all


def test_bulk_list_parsing_and_latest_files():
    html = (
        "download_ext002('2020','TQ', 'BS', '2020_3Q_BS_20240101000000.zip') "
        "download_ext002('2020','TQ', 'BS', '2020_3Q_BS_20250101000000.zip') "
        "download_ext002('2020','TQ', 'PL', '2020_3Q_PL_20240101000000.zip') "
        "download_ext002('2020','TQ', 'CF', '2020_3Q_CF_20240101000000.zip') "
        "download_ext002('2020','TQ', 'CE', '2020_3Q_CE_20240101000000.zip')"
    )
    files = kb.latest_files(kb.parse_bulk_list(html))
    assert len(files) == 1 and files.loc[0, "BS"] == "2020_3Q_BS_20250101000000.zip" and "CE" not in files
    with pytest.raises(ValueError):
        kb.parse_bulk_list("<html>changed</html>")


# --------------------------------------------------------------------------------------- items & versioning


def _q_items(code: str = "000010", years=(2017, 2018, 2019), qrev=None, currency: str = "KRW") -> pd.DataFrame:
    """Dec-FY company: standalone revenue per quarter = qrev[year]; YTD columns as filed."""
    qrev = qrev or {2017: 100.0, 2018: 110.0, 2019: 120.0}
    rows = []
    for y in years:
        for fq, (m, rep) in enumerate(
            ((3, "1분기보고서"), (6, "반기보고서"), (9, "3분기보고서"), (12, "사업보고서")), 1
        ):
            q = qrev[y]
            rows.append(
                {
                    "stock_code": code,
                    "period_end": pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0),
                    "보고서종류": rep,
                    "fs": "CFS",
                    "name": "테스트",
                    "market": "유가증권시장상장법인",
                    "currency": currency,
                    "fy_month": "12",
                    "revenue": q * fq,
                    "revenue_3m": q if fq < 4 else np.nan,
                    "op_income": q * fq / 10,
                    "op_income_3m": q / 10 if fq < 4 else np.nan,
                    "ni": q * fq / 20,
                    "ni_3m": np.nan,
                    "ni_parent": q * fq / 20,
                    "ni_parent_3m": np.nan,
                    "eps": 1.0,
                    "eps_3m": np.nan,
                    "assets": 1000.0 + y,
                    "liabilities": 400.0,
                    "equity": 600.0 + fq,
                    "equity_parent": 500.0 + fq,
                    "nci_equity": np.nan,
                    "cfo": q * fq / 5,
                    "cfo_3m": np.nan,
                    "eps_source": "std",
                    "eps_suspect": False,
                }
            )
    return pd.DataFrame(rows)


def test_quarterly_standalone_and_negative_q4():
    items = _q_items()
    items.loc[items["period_end"] == "2018-12-31", "revenue"] = 300.0  # FY below 3Q YTD (reclassification)
    q = kb.quarterly_standalone(items).set_index("period_end")
    assert q.loc["2017-12-31", "revenue_q"] == pytest.approx(100)  # derived Q4
    assert q.loc["2018-06-30", "revenue_q"] == pytest.approx(110)
    assert np.isnan(q.loc["2018-12-31", "revenue_q"])  # negative derived Q4 revenue -> NaN
    assert q.loc["2018-12-31", "ni_parent_q"] == pytest.approx(110 * 4 / 20 - 110 * 3 / 20)
    assert q.loc["2018-06-30", "ni_parent_q"] == pytest.approx(110 / 20)  # YTD difference when no 3-month column


SESSIONS = pd.bdate_range("2016-01-01", "2021-12-31")


def test_next_session_and_lag_rule():
    assert kdart.next_session_after(pd.Timestamp("2018-03-30"), SESSIONS) == pd.Timestamp("2018-04-02")  # Fri->Mon
    assert kdart.next_session_after(pd.Timestamp("2018-04-02"), SESSIONS) == pd.Timestamp(
        "2018-04-03"
    )  # strictly after
    assert pd.isna(kdart.next_session_after(None, SESSIONS)) and pd.isna(kdart.next_session_after(pd.NaT, SESSIONS))
    assert pd.isna(kdart.next_session_after(pd.Timestamp("2030-01-01"), SESSIONS))
    with pytest.raises(ValueError):
        kdart.next_session_after(pd.Timestamp("2015-06-01"), SESSIONS)
    # quarter: 2018-03-31 + 60 = 2018-05-30 (Wed, a session) -> next session 05-31
    assert kdart.lag_rule_available_from(pd.Timestamp("2018-03-31"), "분기보고서", SESSIONS, 60, 120) == pd.Timestamp(
        "2018-05-31"
    )
    # annual: 2018-12-31 + 120 = 2019-04-30 (Tue) -> 05-01
    assert kdart.lag_rule_available_from(pd.Timestamp("2018-12-31"), "사업보고서", SESSIONS, 60, 120) == pd.Timestamp(
        "2019-05-01"
    )
    # a weekend deadline rolls forward first: 2019-06-30 + 60 = 2019-08-29 (Thu); 2017-03-31 + 60 = 05-30 (Tue)
    ext, last = kdart.extend_sessions(SESSIONS, pd.Timestamp("2022-03-01"))
    assert last == SESSIONS[-1] and ext[-1] == pd.Timestamp("2022-03-01") and ext.is_monotonic_increasing


def _filing(no: str, nm: str, d: str) -> dict:
    return {"rcept_no": no, "report_nm": nm, "rcept_dt": date.fromisoformat(d)}


def test_publication_table_numeric_vs_attachment_corrections():
    rows = [
        _filing("20180330000001", "사업보고서 (2017.12)", "2018-03-30"),
        _filing("20180910000001", "[기재정정]사업보고서 (2017.12)", "2018-09-10"),
        _filing("20180515000001", "분기보고서 (2018.03)", "2018-05-15"),
        _filing("20180520000001", "[첨부정정]분기보고서 (2018.03)", "2018-05-20"),
        _filing("20180814000001", "반기보고서 (2018.06)", "2018-08-14"),
        _filing("20180817000001", "[기재정정]반기보고서 (2018.06)", "2018-08-17"),
        _filing("20180901000001", "주요사항보고서(자기주식취득결정)", "2018-09-01"),
    ]
    pt = kdart.publication_table(rows).set_index(["kind", "period_end"])
    fy = pt.loc[("사업보고서", pd.Timestamp("2017-12-31"))]
    assert fy["first_rcept_no"] == "20180330000001" and fy["last_rcept_dt"] == pd.Timestamp("2018-09-10")
    assert fy["n_numeric_corrections"] == 1
    q1 = pt.loc[("분기보고서", pd.Timestamp("2018-03-31"))]
    assert q1["last_rcept_dt"] == pd.Timestamp("2018-05-15") and q1["n_filings"] == 2  # 첨부정정 is not a restatement
    assert len(pt) == 3


def _pubs_for(items: pd.DataFrame, extra: list[dict] | None = None) -> pd.DataFrame:
    """Originals filed 45 days after quarter ends / 88 days after FY end."""
    rows = []
    for pe, rep in zip(items["period_end"], items["보고서종류"], strict=True):
        kind = kdart.BULK_TO_DART_KIND[rep]
        lag = 88 if kind == "사업보고서" else 45
        d = (pe + pd.Timedelta(lag, "D")).date()
        rows.append(_filing(f"{pe:%Y%m%d}000001", f"{kind} ({pe:%Y.%m})", d.isoformat()))
    return kdart.publication_table(rows + (extra or []))


def _stats() -> dict:
    keys = (
        "short_corrections_waited",
        "original_fs_mismatch",
        "restatement_rows",
        "restatement_rows_superseded",
        "no_available_from",
        "ttm_fs_mismatch",
        "ttm_missing_revenue",
        "ttm_missing_op_income",
        "ttm_missing_ni_parent",
    )
    return dict.fromkeys(keys, 0)


def _rows(items, pubs, originals=None, stats=None):
    q = kb.quarterly_standalone(items)
    stats = stats if stats is not None else _stats()
    out = pd.DataFrame(kf.build_code_rows(q, pubs, originals or {}, SESSIONS, (60, 120), stats))
    return out, stats


def test_ttm_available_from_and_as_filed_rows():
    items = _q_items()
    out, stats = _rows(items, _pubs_for(items))
    r = out.set_index("period_end")
    assert r.loc["2017-12-31", "revenue_ttm"] == 400 and r.loc["2017-12-31", "period_type"] == "FY"
    assert r.loc["2018-03-31", "revenue_ttm"] == pytest.approx(110 + 400 - 100)
    assert r.loc["2019-03-31", "revenue_ttm"] == pytest.approx(120 + 440 - 110)
    assert np.isnan(r.loc["2017-03-31", "revenue_ttm"]) and stats["ttm_missing_revenue"] >= 3  # no 2016 reports
    assert r.loc["2018-03-31", "revenue_q"] == 110 and r.loc["2018-03-31", "revenue_q_prev_year"] == 100
    # available_from = first session strictly after the receipt date (2018-05-15 Tue -> 05-16)
    assert r.loc["2018-03-31", "available_from"] == pd.Timestamp("2018-05-16")
    assert (out["available_from"] > pd.to_datetime(out["rcept_dt"])).all()
    assert (out["values_version"] == "as_filed").all() and not out["lookahead_risk"].any()
    assert not out["restated_later"].any() and (out["date_source"] == "dart_web").all()


def test_restatement_uses_original_values_until_correction():
    items = _q_items()
    extra = [
        _filing("20180910000001", "[기재정정]사업보고서 (2017.12)", "2018-09-10"),  # long gap, after next report
        _filing("20181220000001", "[기재정정]분기보고서 (2018.09)", "2018-12-20"),  # long gap, before FY2018
        _filing("20180817000001", "[기재정정]반기보고서 (2018.06)", "2018-08-17"),  # short gap -> wait for it
    ]
    pubs = _pubs_for(items, extra)
    same = {"op_income": 40.0, "ni_parent": 20.0, "equity_parent": 504.0}  # FY2017 values unchanged by the fix
    q3_orig = {"op_income": 33.0, "ni_parent": 16.5, "equity_parent": 503.0, "revenue": 320.0, "_fs": "CFS"}
    originals = {"20171231000001": {**same, "revenue": 390.0, "_fs": "CFS"}, "20180930000001": q3_orig}
    out, stats = _rows(items, pubs, originals)
    fy17 = out[out["period_end"] == "2017-12-31"]
    assert len(fy17) == 1 and fy17["values_version"].iloc[0] == "original_xbrl"  # restatement row superseded
    assert fy17["revenue_ttm"].iloc[0] == 390 and fy17["restated_later"].iloc[0]
    assert stats["restatement_rows_superseded"] >= 1
    q1 = out[out["period_end"] == "2018-03-31"].iloc[0]
    assert q1["revenue_ttm"] == pytest.approx(110 + 390 - 100)  # FY2017 correction not public yet on 2018-05-16
    assert not q1["lookahead_risk"]
    q3 = out[out["period_end"] == "2018-09-30"].sort_values("available_from")
    assert q3["values_version"].tolist() == ["original_xbrl", "restatement"]
    assert q3["available_from"].tolist() == [pd.Timestamp("2018-11-15"), pd.Timestamp("2018-12-21")]
    assert q3["revenue_ttm"].tolist() == [pytest.approx(320 + 400 - 300), pytest.approx(330 + 400 - 300)]
    h1 = out[out["period_end"] == "2018-06-30"].iloc[0]
    assert h1["available_from"] == pd.Timestamp("2018-08-20") and h1["values_version"] == "as_filed"
    assert stats["short_corrections_waited"] == 1
    # without the originals the bulk (restated) values are flagged, never silently mixed
    out2, _ = _rows(items, pubs, {})
    q1b = out2[out2["period_end"] == "2018-03-31"].iloc[0]
    assert q1b["revenue_ttm"] == pytest.approx(110 + 400 - 100) and q1b["lookahead_risk"]
    assert out2.loc[out2["period_end"] == "2017-12-31", "values_version"].iloc[0] == "restated_bulk"


def test_unknown_dates_use_lag_rule_and_are_flagged():
    items = _q_items(years=(2018,))
    out, _ = _rows(items, None)
    r = out.set_index("period_end")
    assert (out["date_source"] == "lag_rule").all() and out["restated_later"].all() and out["lookahead_risk"].all()
    assert r.loc["2018-03-31", "available_from"] == pd.Timestamp("2018-05-31")
    assert r.loc["2018-12-31", "available_from"] == pd.Timestamp("2019-05-01")


def test_currency_conversion_and_exclusion():
    items = pd.concat(
        [_q_items("000030", years=(2018,), currency="USD"), _q_items("000040", years=(2018,), currency="JPY")]
    )
    fx = pd.Series(1200.0, index=pd.bdate_range("2017-01-01", "2019-12-31"))
    notes, cov = [], {}
    out = kf.convert_currency(items, lambda c: fx if c == "USD" else None, notes, cov)
    assert set(out["stock_code"]) == {"000030"} and (out["currency"] == "KRW").all()
    assert (out["reporting_currency"] == "USD").all()
    fy = out[out["period_end"] == "2018-12-31"].iloc[0]
    assert fy["revenue"] == pytest.approx(440 * 1200) and fy["equity_parent"] == pytest.approx(504 * 1200)
    assert cov["fx_converted_rows"] == {"USD": 4} and cov["fx_excluded_rows"] == {"JPY": 4}
    assert any("환산" in n for n in notes) and any("제외" in n for n in notes)


_DUR = "<xbrli:startDate>2018-01-01</xbrli:startDate><xbrli:endDate>2018-12-31</xbrli:endDate>"
_SEP = (
    '<xbrli:segment><xbrldi:explicitMember dimension="a:ConsolidatedAndSeparateFinancialStatementsAxis">'
    "a:SeparateMember</xbrldi:explicitMember></xbrli:segment>"
)
_SEG = '<xbrli:segment><xbrldi:typedMember dimension="a:SegmentAxis"><a:s>1</a:s></xbrldi:typedMember></xbrli:segment>'


def _ctx(cid: str, period: str, dims: str = "") -> str:
    ent = f'<xbrli:entity><xbrli:identifier scheme="x">1</xbrli:identifier>{dims}</xbrli:entity>'
    return f'<xbrli:context id="{cid}">{ent}<xbrli:period>{period}</xbrli:period></xbrli:context>'


def _xbrl_zip(facts: list[tuple[str, str, str]]) -> bytes:
    ctx = (
        _ctx("cD", _DUR)
        + _ctx("cSep", _DUR, _SEP)
        + _ctx("iD", "<xbrli:instant>2018-12-31</xbrli:instant>")
        + _ctx("cSeg", _DUR, _SEG)
        + '<xbrli:unit id="KRW"><xbrli:measure>iso4217:KRW</xbrli:measure></xbrli:unit>'
    )
    body = "".join(f'<{c} contextRef="{ctx_id}" unitRef="KRW" decimals="0">{v}</{c}>' for c, ctx_id, v in facts)
    xml = (
        '<?xml version="1.0" encoding="utf-8"?><xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" '
        'xmlns:xbrldi="http://xbrl.org/2006/xbrldi" xmlns:ifrs-full="http://xbrl.ifrs.org/taxonomy/ifrs-full" '
        'xmlns:dart="http://dart.fss.or.kr" xmlns:a="http://a">' + ctx + body + "</xbrli:xbrl>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("entity_2018-12-31.xbrl", xml)
    return buf.getvalue()


def test_xbrl_key_items_consolidated_only():
    z = _xbrl_zip(
        [
            ("ifrs-full:Revenue", "cD", "1000"),
            ("ifrs-full:Revenue", "cSep", "700"),
            ("dart:OperatingIncomeLoss", "cSep", "-50"),  # separate only -> must NOT leak into a consolidated filing
            ("ifrs-full:ProfitLossAttributableToOwnersOfParent", "cSeg", "1"),  # typed segment context -> skipped
            ("ifrs-full:EquityAttributableToOwnersOfParent", "iD", "3000"),
        ]
    )
    out = kdart.xbrl_key_items(z, "2018-12-31")
    assert out["revenue"] == 1000 and out["equity_parent"] == 3000 and out["_fs"] == "CFS"
    assert "op_income" not in out and "ni_parent" not in out and out["_currency"] == "KRW"
    sep_only = kdart.xbrl_key_items(_xbrl_zip([("ifrs-full:Revenue", "cSep", "700")]), "2018-12-31")
    assert sep_only["revenue"] == 700 and sep_only["_fs"] == "OFS"


# -------------------------------------------------------------------------------------------- DART web + gate


def _dart_html(rows: list[tuple[str, str, str]]) -> str:
    trs = "".join(
        f"<tr><td>{i}</td><td><a href=\"javascript:openCorpInfoNew('00126380', 'w', '/x');\">가</a></td>"
        f'<td><a href="/dsaf001/main.do?rcpNo={no}">{nm}</a></td><td>가</td><td>{d}</td><td>유</td></tr>'
        for i, (no, nm, d) in enumerate(rows, 1)
    )
    return f"<table><tbody>{trs}</tbody></table><div class='pageInfo'>[1/1] [총 {len(rows)}건]</div>"


class _DartSession:
    def __init__(self, by_code: dict[str, list[tuple[str, str, str]]], fail: bool = False):
        self.by_code = by_code
        self.fail = fail
        self.posts: list[dict] = []

    def post(self, url, data=None, **kw):
        if self.fail:
            raise requests.ConnectionError("RemoteDisconnected")
        self.posts.append(data)
        ws = data["startDate"]
        rows = [r for r in self.by_code.get(data["textCrpNm"], []) if r[2].replace(".", "") >= ws]
        rows = [r for r in rows if r[2].replace(".", "") <= data["endDate"]]
        if not rows:
            return _Resp(text="<div>조회 결과가 없습니다.</div>")
        return _Resp(text=_dart_html(rows))


def test_dart_gate_budget_throttle_and_circuit_breaker(tmp_path):
    kdart.reset_circuit_breaker()
    slept: list[float] = []
    clock = iter(np.arange(0, 1000, 1.0))
    gate = kdart.DartGate(max_requests=2, min_interval=6.0, sleep=slept.append, clock=lambda: next(clock))
    sess = _DartSession({"000010": [("20190401000001", "사업보고서 (2018.12)", "2019.04.01")]})
    rows, fetched = kdart.cached_web_filings(tmp_path, sess, gate, "000010", date(2019, 6, 30))
    assert rows is not None and fetched == 2 and len(sess.posts) == 2 and slept  # two windows, throttled
    assert rows[0]["rcept_dt"] == date(2019, 4, 1)
    # cached: no network on the second call; the budget is exhausted anyway
    again, fetched2 = kdart.cached_web_filings(tmp_path, sess, gate, "000010", date(2019, 6, 30))
    assert fetched2 == 0 and again == rows and len(sess.posts) == 2
    refused, _ = kdart.cached_web_filings(tmp_path, sess, gate, "000020", date(2019, 6, 30))
    assert refused is None and gate.refused >= 1
    # a connection error trips the breaker for the whole process
    gate2 = kdart.DartGate(max_requests=10, min_interval=0)
    none, _ = kdart.cached_web_filings(tmp_path, _DartSession({}, fail=True), gate2, "000030", date(2019, 6, 30))
    assert none is None and gate2.blocked_error
    assert kdart.DartGate(max_requests=10, min_interval=0).blocked  # breaker is process-wide
    kdart.reset_circuit_breaker()
    assert not kdart.DartGate(max_requests=10, min_interval=0).blocked


def test_search_windows_are_fixed_and_contiguous():
    w = kdart.search_windows(date(2026, 9, 25))
    assert w[0] == (date(2014, 1, 1), date(2018, 12, 31), True)
    assert w[1] == (date(2019, 1, 1), date(2023, 12, 31), True)
    assert w[-1] == (date(2024, 1, 1), date(2026, 9, 25), False)
    assert all((b[0] - a[1]).days == 1 for a, b in zip(w, w[1:], strict=False))


# ------------------------------------------------------------------------------------ fundamentals end-to-end


def _artefacts() -> kpit.KRArtefacts:
    spells = pd.DataFrame(
        {
            "Code": ["000010", "000020", "000030"],
            "spell": [0, 0, 0],
            "first_date": pd.to_datetime(["2010-01-04", "2010-01-04", "2010-01-04"]),
            "last_row_date": pd.to_datetime(["2021-12-31", "2018-09-28", "2021-12-31"]),
            "first_traded": pd.to_datetime(["2010-01-04"] * 3),
            "last_traded": pd.to_datetime(["2021-12-31", "2018-09-28", "2021-12-31"]),
            "name": ["가", "나", "다"],
            "market_id": ["STK", "KSQ", "STK"],
            "rows": [100, 100, 100],
            "active": [True, False, True],
            "entity": ["000010", "000020", "000030"],
        }
    )
    me_dates = pd.DatetimeIndex(pd.Series(SESSIONS, index=SESSIONS).groupby(SESSIONS.to_period("M")).max().to_numpy())
    month_end = pd.DataFrame(
        {
            "Code": np.repeat(["000010", "000020", "000030"], len(me_dates)),
            "spell": 0,
            "Date": np.tile(me_dates, 3),
            "Stocks": 1e6,
            "Marcap": 1e9,
            "pref_marcap": 0.0,
            "MarketId": "STK",
        }
    )
    mcap = pd.DataFrame(
        {
            "Code": np.repeat(["000010", "000020"], len(SESSIONS)),
            "spell": 0,
            "Date": np.tile(SESSIONS, 2),
            "marcap": 1e4,
            "marcap_total": np.r_[np.full(len(SESSIONS), 2e4), np.full(len(SESSIONS), 1e4)],
        }
    )
    mcap = mcap[~((mcap["Code"] == "000020") & (mcap["Date"] > "2018-09-28"))]
    return kpit.KRArtefacts("sig", SESSIONS, spells, month_end, None, mcap_frame=mcap)


def test_build_kr_pit_fundamentals_offline(tmp_path):
    kdart.reset_circuit_breaker()
    settings = _settings(tmp_path)
    items = pd.concat(
        [_q_items("000010"), _q_items("000020", years=(2017, 2018, 2019)), _q_items("000030", currency="USD")],
        ignore_index=True,
    )
    web = {}
    for code in ("000010", "000020"):
        own = items[items["stock_code"] == code]
        rows = []
        for pe, rep in zip(own["period_end"], own["보고서종류"], strict=True):
            kind = kdart.BULK_TO_DART_KIND[rep]
            d = pe + pd.Timedelta(88 if kind == "사업보고서" else 45, "D")
            rows.append((f"{pe:%Y%m%d}{code}", f"{kind} ({pe:%Y.%m})", f"{d:%Y.%m.%d}"))
        web[code] = rows
    sess = _DartSession(web)
    fx = pd.Series(1000.0, index=pd.bdate_range("2016-01-01", "2020-12-31"))
    f = kpit.load_kr_pit_fundamentals(
        settings,
        DiskCache(settings.cache_path),
        ["000010", "000020", "000030", "999990"],
        session=sess,
        now=datetime(2020, 6, 1, 10, 0, tzinfo=KST),
        dart_gate=kdart.DartGate(max_requests=4, min_interval=0),
        fx_history=lambda c: fx,
        artefacts=_artefacts(),
        items=items,
    )
    assert isinstance(f, PITFundamentals)
    p = f.panel
    assert list(p.columns[: len(FUNDAMENTAL_COLUMNS)]) == FUNDAMENTAL_COLUMNS
    assert set(p["entity"]) == {"000010", "000020", "000030"}
    # 000020 was delisted 2018-09-28: rows only available afterwards are dropped and counted
    assert p.loc[p["entity"] == "000020", "available_from"].max() <= pd.Timestamp("2018-09-28")
    assert f.coverage["post_delisting_rows_dropped"] > 0
    # 000030: no DART dates within the budget (4 requests = 2 codes x 2 windows) -> lag rule, USD converted
    c = p[p["entity"] == "000030"]
    assert (c["date_source"] == "lag_rule").all() and (c["reporting_currency"] == "USD").all()
    assert c.loc[c["period_end"] == "2018-12-31", "revenue_ttm"].iloc[0] == pytest.approx(440 * 1000)
    assert f.coverage["dart_dates"]["without_dates"] == 1 and f.coverage["dart_dates"]["web_refused"] >= 1
    a = p[p["entity"] == "000010"].set_index("period_end")
    assert a.loc["2018-03-31", "available_from"] == pd.Timestamp("2018-05-16")
    assert a.loc["2018-03-31", "shares_out"] == 1e6 and (a["currency"] == "KRW").all()
    assert f.coverage["entities_unknown"] == ["999990"]
    assert any("접수일 미확인" in n for n in f.notes) and any("marcap에 없는" in n for n in f.notes)
    # asof (shared contract) picks the latest available row
    row = f.asof("000010", pd.Timestamp("2018-06-01"))
    assert row is not None and row["period_end"] == pd.Timestamp("2018-03-31")
    # valuation at t: market cap (incl. preferred) / fundamentals, never EPS
    mc = kpit.pit_market_cap_kr(
        settings,
        DiskCache(settings.cache_path),
        ["000010", "000020"],
        ["2018-06-01", "2019-06-03"],
        artefacts=_artefacts(),
    )
    assert mc.loc["2018-06-01", "000010"] == 2e4 and np.isnan(mc.loc["2019-06-03", "000020"])
    val = kpit.pit_valuation_frame_kr(f, mc).set_index(["date", "entity"])
    v = val.loc[(pd.Timestamp("2018-06-01"), "000010")]
    assert v["ep"] == pytest.approx(a.loc["2018-03-31", "ni_parent_ttm"] / 2e4)
    assert v["bp"] == pytest.approx(501 / 2e4) and v["roe"] == pytest.approx(a.loc["2018-03-31", "ni_parent_ttm"] / 501)
    assert v["revenue_yoy"] == pytest.approx(110 / 100 - 1) and v["op_margin"] == pytest.approx(0.1)
    one = kpit.pit_valuation_kr(f, "000010", "2018-06-01", 2e4)
    assert one is not None and one["ep"] == pytest.approx(v["ep"])
    assert kpit.pit_valuation_kr(f, "000010", "2016-01-05", 2e4) is None  # nothing available yet


def test_stale_fundamentals_are_dropped_from_valuation():
    panel = pd.DataFrame(
        {
            "entity": ["000010"],
            "period_end": [pd.Timestamp("2018-03-31")],
            "available_from": [pd.Timestamp("2018-05-16")],
            "ni_parent_ttm": [10.0],
            "equity_parent": [100.0],
            "revenue_ttm": [50.0],
            "op_income_ttm": [5.0],
        }
    )
    f = PITFundamentals(market=kpit.Market.KR, panel=panel)
    mc = pd.DataFrame({"000010": [1000.0, 1000.0]}, index=pd.to_datetime(["2018-06-01", "2019-06-03"]))
    val = kpit.pit_valuation_frame_kr(f, mc, max_staleness_days=200).set_index("date")
    assert val.loc["2018-06-01", "ep"] == pytest.approx(0.01) and np.isnan(val.loc["2019-06-03", "ep"])


# ------------------------------------------------------------------------------------------------------ live


@pytest.mark.live
def test_live_universe_top60_2016_2026():
    from quantcio.data.http import PoliteSession
    from quantcio.data.pit import kr_crosscheck as qa

    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    t0 = time.monotonic()
    u = kpit.load_kr_pit_universe(settings, cache, 60, date(2016, 1, 1))
    secs = time.monotonic() - t0
    cov = u.coverage
    print(
        f"\nKR PIT top-60: {cov['rebalance_dates']} rebalances {cov['first_rebalance']}..{cov['last_rebalance']}, "
        f"{cov['member_months']} member-months, {cov['entities']} entities, {cov['delisted_entities']} delisted "
        f"{cov['delisted_by_category']}, excluded above cutoff {cov['excluded_above_cutoff']}, {secs:.1f}s "
        f"(cache={cov['from_cache']})"
    )
    assert cov["rebalance_dates"] >= 120 and cov["member_months"] == 60 * cov["rebalance_dates"]
    assert cov["no_price_entities"] == [] and cov["delisted_entities"] >= 1
    assert all(len(px) > 0 for px in u.prices.values())
    for b in ("KOSPI", "KOSDAQ"):
        assert u.benchmarks[b].index[0] <= pd.Timestamp("2014-01-02")
    sess = PoliteSession()
    naver = qa.naver_fchart_spells(sess, "KOSPI")
    art = kpit.kr_artefacts(settings, cache)
    diff = qa.calendar_diff(art.sessions, pd.DatetimeIndex(naver.index))
    assert diff["only_marcap"] == 0 and diff["only_naver"] == 0 and diff["overlap"] > 2900
    agree = qa.adjusted_close_agreement(u.prices["005930"], qa.naver_fchart_spells(sess, "005930"))
    assert agree["within_tol"] > 0.99


@pytest.mark.live
def test_live_fundamentals_five_entities_incl_delisted():
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    entities = ["005930", "068270", "035420", "241560", "091990"]  # 091990 셀트리온헬스케어 delisted 2024-01
    t0 = time.monotonic()
    f = kpit.load_kr_pit_fundamentals(settings, cache, entities, max_dart_requests=120)
    secs = time.monotonic() - t0
    cov = f.coverage
    print(
        f"\nKR PIT fundamentals: {cov['rows']} rows, {cov['entities_with_rows']}/{len(entities)} entities, "
        f"dates {cov['date_source_rows']}, versions {cov['values_version_rows']}, fx {cov['fx_converted_rows']}, "
        f"DART {cov['dart_web_requests_total']} req, originals {cov['original_xbrl']}, {secs:.1f}s"
    )
    assert cov["entities_with_rows"] == len(entities)
    p = f.panel
    assert (p["available_from"] > p["period_end"]).all()
    dated = p[p["date_source"] != "lag_rule"]
    assert (dated["available_from"] > dated["rcept_dt"]).all()
    assert p.loc[p["entity"] == "091990", "available_from"].max() <= pd.Timestamp("2024-01-12")
    cell = p[(p["entity"] == "068270") & (p["period_end"] == "2018-12-31") & ~p["is_restatement_row"]]
    if cell["values_version"].iloc[0] == "original_xbrl":  # Celltrion FY2018 as first reported (2019-04-01)
        assert cell["op_income_ttm"].iloc[0] == pytest.approx(3386.9e8, rel=1e-3)
    assert (p.loc[p["entity"] == "241560", "reporting_currency"] == "USD").any()
