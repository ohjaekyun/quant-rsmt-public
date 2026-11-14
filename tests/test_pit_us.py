"""Tests for quantcio.data.pit.us (US point-in-time S&P 500 universe + SEC XBRL fundamentals).

Offline tests run on a small synthetic "world" served by a fake ``PitNet`` (no network): fja05680 snapshots and
Wikipedia pages (tests/fixtures/pit_us/*), SEC ticker files / submissions / fails-to-deliver zips, Yahoo chart JSON
and companyfacts generated in code. One real trimmed fixture (JPM companyfacts 2011-2013) checks the amendment rule.
Tests marked ``live`` build the real universe (2018-2026, top-100) and fundamentals for 5 CIKs.
"""

from __future__ import annotations

import gzip
import io
import json
import time
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from quantcio.config import load_settings
from quantcio.data.cache import DiskCache
from quantcio.data.pit import (
    FUNDAMENTAL_COLUMNS,
    MEMBERSHIP_COLUMNS,
    PITFundamentals,
    PITUniverse,
    us_cik,
    us_fund,
    us_membership,
    us_prices,
)
from quantcio.data.pit import us as pit_us
from quantcio.data.pit import us_calendar as cal
from quantcio.data.pit.us_net import PitNet, SecUserAgentError
from quantcio.models import PRICE_COLUMNS

FIX = Path(__file__).parent / "fixtures" / "pit_us"
UA = "QuantCIO Research research@example.com"
TODAY = pd.Timestamp("2020-06-30")
# the real objects the module-scoped `world` fixture replaces until the module ends: live tests restore them
_REAL = {
    "request": requests.Session.request,
    "today": pit_us._today,
    "count": us_membership.COUNT_RANGE,
    "hard": us_membership.COUNT_HARD_RANGE,
    "share": pit_us.MIN_SHARE_COVERAGE,
}


def _real_world(monkeypatch) -> None:
    """Undo this test's patches AND the module-scoped `world` patches (network block, 2020-06-30 'today', relaxed
    count ranges): otherwise a live test run after `world` checks today's S&P 500 against a 2020 calendar."""
    monkeypatch.undo()
    monkeypatch.setattr(requests.Session, "request", _REAL["request"])
    monkeypatch.setattr(pit_us, "_today", _REAL["today"])
    monkeypatch.setattr(us_membership, "COUNT_RANGE", _REAL["count"])
    monkeypatch.setattr(us_membership, "COUNT_HARD_RANGE", _REAL["hard"])
    monkeypatch.setattr(pit_us, "MIN_SHARE_COVERAGE", _REAL["share"])


SESS = cal.nyse_sessions("2017-01-03", TODAY)
SPLIT_AAA = pd.Timestamp("2019-07-01")


def settings_for(tmp_path: Path, ua: str = UA):
    return load_settings(data={"sec_user_agent": ua, "cache_dir": str(tmp_path / "cache")})


# =============================================================================== synthetic world


def walk(p0: float, seed: int, vol: float = 0.008) -> pd.Series:
    r = np.random.default_rng(seed).normal(0.0002, vol, len(SESS))
    return pd.Series(p0 * np.exp(np.cumsum(r)), index=SESS)


def _raw_prices() -> dict[str, pd.Series]:
    """Raw (as traded, unadjusted) closes of every synthetic security."""
    aaa = walk(100, 1)
    aaa[aaa.index >= SPLIT_AAA] /= 2.0  # 2:1 split
    return {
        "AAA": aaa,
        "BBB": walk(50, 2)[: pd.Timestamp("2020-02-28")],  # acquired, last session 2020-02-28
        "GOOGL": walk(1000, 3),
        "GOOG": walk(1000, 3) * 0.995,
        "META": walk(150, 4),  # FB until 2019-09-30, META after (same security, same CUSIP)
        "WRG": walk(30, 5),  # the member issuer (only in SEC FTD)
        "WRG_Y": walk(80, 55, vol=0.02),  # ANOTHER issuer that Yahoo shows under 'WRG'
        "OLD": walk(40, 6),
        "NEW": walk(60, 7),
        "FRC": walk(90, 9)[: pd.Timestamp("2020-03-19")],  # halted 2020-03-19, receivership
        "RSP": walk(100, 99),
        "SPY": walk(300, 98),
    }


RAW = _raw_prices()
SHARES = {"AAA": 100e6, "BBB": 50e6, "META": 600e6, "WRG": 40e6, "OLD": 80e6, "NEW": 70e6}  # AAA 200M after split
VOLUME = {"GOOGL": 5e6, "GOOG": 4e6}  # others 1e6 shares/day (raw)
CUSIP = {
    "AAA": "000001AA1",
    "BBB": "000002BB2",
    "GOOGL": "000003GL3",
    "GOOG": "000003GG3",
    "FB": "000004FB4",
    "META": "000004FB4",
    "WRG": "000005WR5",
    "OLD": "000007OL7",
    "NEW": "000006NW6",
    "FRC": "000009FR9",
    "FRCB": "000009FR9",
}
CIKS = {"AAA": 1001, "BBB": 1002, "GOOGL": 1003, "META": 1004, "WRG": 1005, "NEW": 1006, "OLD": 1007, "MIS": 1008}
CIKS["FRC"] = 1009


def _epoch(d: pd.Timestamp) -> int:
    return int((pd.Timestamp(d).tz_localize("America/New_York") + pd.Timedelta(570, "min")).timestamp())


def chart_json(symbol: str, raw: pd.Series, name: str, splits: dict | None = None, vol: float = 1e6) -> bytes:
    """Yahoo v8 chart payload: close split-adjusted, adjclose also 'dividend' adjusted, volume split-adjusted."""
    f = pd.Series(1.0, index=raw.index)
    for d, r in (splits or {}).items():
        f[f.index < d] *= r
    close = raw / f
    adj = close * np.linspace(0.97, 1.0, len(close))  # a small dividend drift
    res = {
        "meta": {
            "symbol": symbol,
            "currency": "USD",
            "exchangeName": "NYQ",
            "fullExchangeName": "NYSE",
            "instrumentType": "EQUITY",
            "firstTradeDate": _epoch(pd.Timestamp("2005-01-03")),
            "longName": name,
            "exchangeTimezoneName": "America/New_York",
        },
        "timestamp": [_epoch(d) for d in raw.index],
        "events": {
            "splits": {
                str(_epoch(d)): {"date": _epoch(d), "numerator": r, "denominator": 1} for d, r in (splits or {}).items()
            }
        },
        "indicators": {
            "quote": [
                {
                    "open": close.round(4).tolist(),
                    "high": (close * 1.01).round(4).tolist(),
                    "low": (close * 0.99).round(4).tolist(),
                    "close": close.round(4).tolist(),
                    "volume": (vol * f).round().tolist(),
                }
            ],
            "adjclose": [{"adjclose": adj.round(4).tolist()}],
        },
    }
    return json.dumps({"chart": {"result": [res], "error": None}}).encode()


def ftd_rows() -> list[tuple[pd.Timestamp, str, str, float]]:
    """(settle, cusip, symbol, price) — price = raw close of the session before settlement; ~70% of days."""
    rng = np.random.default_rng(11)
    rows = []

    def add(sym: str, px: pd.Series, lo: str | None = None, hi: str | None = None, cusip: str | None = None):
        s = px
        if lo:
            s = s[s.index >= pd.Timestamp(lo)]
        if hi:
            s = s[s.index <= pd.Timestamp(hi)]
        month_end = s.index.to_series().groupby(s.index.to_period("M")).transform("max")
        for d, p in s.items():
            if rng.random() < 0.7 or (month_end[d] - d).days <= 3:
                rows.append((cal.next_session(d), cusip or CUSIP[sym], sym, round(float(p), 2)))

    add("AAA", RAW["AAA"])
    add("BBB", RAW["BBB"])
    add("BBB", pd.Series(1.23, index=cal.nyse_sessions("2020-03-10", "2020-04-30")))  # rows persist after delisting
    add("GOOGL", RAW["GOOGL"])
    add("GOOG", RAW["GOOG"])
    add("FB", RAW["META"], hi="2019-09-30")
    add("META", RAW["META"], lo="2019-10-01")
    add("WRG", RAW["WRG"])
    add("OLD", RAW["OLD"])
    # NEW: no FTD prints at all -> its Yahoo series cannot be validated -> split-adjusted only
    add("FRC", RAW["FRC"])
    add("FRCB", pd.Series(0.33, index=cal.nyse_sessions("2020-04-03", "2020-04-20")))  # OTC after the failure
    return rows


def ftd_zips() -> dict[str, bytes]:
    by_key: dict[str, list[str]] = {}
    for settle, cusip, sym, px in ftd_rows():
        key = settle.strftime("%Y%m") + ("a" if settle.day <= 15 else "b")
        by_key.setdefault(key, []).append(
            f"{settle:%Y%m%d}|{cusip}|{sym}|{1000}|{sym} COMMON STOCK|CLASS A|{px}"
        )  # DESCRIPTION containing '|'
    out = {}
    for key, lines in by_key.items():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            body = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n" + "\n".join(lines) + "\n"
            z.writestr(f"cnsfails{key}", body)
        out[key] = buf.getvalue()
    return out


def fact(start, end, val, filed, form, accn) -> dict:
    f = {"end": str(pd.Timestamp(end).date()), "val": val, "accn": accn, "form": form, "filed": str(filed)}
    if start is not None:
        f["start"] = str(pd.Timestamp(start).date())
    return f


def companyfacts(
    cik: int, name: str, shares: float | None, split: tuple[str, float] | None = None, eps: bool = True
) -> dict:
    """Quarterly 10-Q/10-K facts 2016Q1..2020Q1 (filed end + 35/55 days): revenue, NI, diluted EPS, equity and the
    dei cover share count; after ``split`` (date, ratio) filings report post-split counts and EPS."""
    ends = pd.date_range("2016-03-31", "2020-03-31", freq="QE")
    g: dict[str, list] = {k: [] for k in ("rev", "ni", "eps", "eq", "cov")}
    for i, e in enumerate(ends):
        q4 = e.month == 12
        filed = (e + pd.Timedelta(55 if q4 else 35, "D")).date()
        form, accn = ("10-K" if q4 else "10-Q"), f"{cik:010d}-{e:%y%m}"
        start = (e - pd.offsets.QuarterEnd(1) + pd.Timedelta(1, "D")).normalize()
        ni, rev = 2e8 * (1 + 0.01 * i), 2e9 * (1 + 0.01 * i)
        mult = split[1] if split and pd.Timestamp(filed) >= pd.Timestamp(split[0]) else 1.0
        sh = shares * mult if shares else None
        g["rev"].append(fact(start, e, rev, filed, form, accn))
        g["ni"].append(fact(start, e, ni, filed, form, accn))
        if eps and sh:
            g["eps"].append(fact(start, e, round(ni / sh, 4), filed, form, accn))
            if i >= 4:  # prior-year comparative restated on the filing's share basis
                pe = ends[i - 4]
                ps = (pe - pd.offsets.QuarterEnd(1) + pd.Timedelta(1, "D")).normalize()
                g["eps"].append(fact(ps, pe, round(2e8 * (1 + 0.01 * (i - 4)) / sh, 4), filed, form, accn))
        if q4:
            fy_start = pd.Timestamp(f"{e.year}-01-01")
            g["rev"].append(fact(fy_start, e, rev * 4, filed, form, accn))
        g["eq"].append(fact(None, e, 5e9 + 1e8 * i, filed, form, accn))
        if sh:
            g["cov"].append(fact(None, pd.Timestamp(filed) - pd.Timedelta(5, "D"), sh, filed, form, accn))
    gaap = {
        "Revenues": {"units": {"USD": g["rev"]}},
        "NetIncomeLoss": {"units": {"USD": g["ni"]}},
        "StockholdersEquity": {"units": {"USD": g["eq"]}},
    }
    if g["eps"]:
        gaap["EarningsPerShareDiluted"] = {"units": {"USD/shares": g["eps"]}}
    facts = {"us-gaap": gaap}
    if g["cov"]:
        facts["dei"] = {"EntityCommonStockSharesOutstanding": {"units": {"shares": g["cov"]}}}
    return {"cik": cik, "entityName": name, "facts": facts}


def submissions(name: str, periodic: bool = True, former: list[str] | None = None) -> dict:
    dates = [str(d.date()) for d in pd.date_range("2017-02-15", "2020-05-15", freq="QS-FEB")] if periodic else []
    return {
        "name": name,
        "formerNames": [{"name": n} for n in former or []],
        "tickers": [],
        "filings": {
            "recent": {
                "form": ["10-Q"] * len(dates) if periodic else ["8-K"],
                "filingDate": dates if periodic else ["2020-03-01"],
                "accessionNumber": [f"a{i}" for i in range(max(len(dates), 1))],
            },
            "files": [],
        },
    }


def ticker_file(pairs: list[tuple[str, int, str]]) -> bytes:
    return json.dumps({str(i): {"cik_str": c, "ticker": t, "title": n} for i, (t, c, n) in enumerate(pairs)}).encode()


def build_routes() -> dict[str, bytes]:
    r: dict[str, bytes] = {}
    fja_name = "S&P 500 Historical Components & Changes (Updated).csv"
    r[us_membership.FJA_CONTENTS] = json.dumps(
        [{"name": "README.md"}, {"name": fja_name}, {"name": "S&P 500 Historical Components & Changes(12-30-2024).csv"}]
    ).encode()
    from urllib.parse import quote

    r[us_membership.FJA_RAW + quote(fja_name)] = (FIX / "fja_small.csv").read_bytes()
    r[us_membership.WIKI_CURRENT] = (FIX / "wiki_current_small.html").read_bytes()
    r[us_membership.WIKI_HIST] = (FIX / "wiki_changes_small.html").read_bytes()
    r["https://en.wikipedia.org/w/api.php"] = json.dumps({"query": {"pages": {"-1": {"title": "x"}}}}).encode()
    stamps = ["20190601000000", "20190901000000", "20200315000000"]
    r[us_cik.WAYBACK_CDX] = json.dumps([["timestamp", "digest", "length"]] + [[t, "x", "1"] for t in stamps]).encode()
    snap = [
        ("AAA", 1001, "Alpha Corp"),
        ("BBB", 1002, "Bravo Inc"),
        ("GOOGL", 1003, "Gamma Holdings"),
        ("WRG", 1005, "Wrong Target Co"),
        ("OLD", 1007, "Oldco Industries"),
        ("FRC", 1009, "First Failed Bank"),
    ]
    r[us_cik.WAYBACK_SNAP.format(ts=stamps[0])] = ticker_file(
        [*snap, ("FB", 1004, "Facebook Inc"), ("MIS", 1008, "Missing Data Corp")]
    )
    r[us_cik.WAYBACK_SNAP.format(ts=stamps[1])] = ticker_file(
        [*snap, ("FB", 1004, "Facebook Inc"), ("MIS", 4242, "Misfit Holdings")]  # MIS reused by a dead issuer
    )
    r[us_cik.WAYBACK_SNAP.format(ts=stamps[2])] = ticker_file([*snap, ("META", 1004, "Meta Platforms Inc")])
    r[us_cik.SEC_TICKERS] = ticker_file(
        [
            ("AAA", 1001, "Alpha Corp"),
            ("GOOGL", 1003, "Gamma Holdings"),
            ("GOOG", 1003, "Gamma Holdings"),
            ("META", 1004, "Meta Platforms Inc"),
            ("WRG", 1005, "Wrong Target Co"),
            ("NEW", 1006, "Newco Inc"),
            ("OLD", 1007, "Oldco Industries"),
        ]
    )
    r[us_cik.SEC_LOOKUP] = b"ALPHA CORP:0000001001:\nBRAVO INC:0000001002:\nMISSING DATA CORP:0000001008:\n"
    subs = {
        1002: submissions("Bravo Inc"),
        1004: submissions("Meta Platforms Inc", former=["Facebook Inc"]),
        1007: submissions("Oldco Industries"),
        1008: submissions("Missing Data Corp"),
        1009: submissions("First Failed Bank", periodic=False),  # FDIC filer: no 10-K/10-Q in EDGAR
        4242: submissions("Misfit Holdings", periodic=False),
    }
    for c, j in subs.items():
        r[f"{us_cik.SUBMISSIONS}CIK{c:010d}.json"] = json.dumps(j).encode()
    zips = ftd_zips()
    idx = "".join(f'<a href="/files/data/fails-deliver-data/cnsfails{k}.zip">x</a>\n' for k in sorted(zips))
    r[us_prices.FTD_INDEX] = idx.encode()
    for k, b in zips.items():
        r[f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{k}.zip"] = b
    charts = {
        "AAA": chart_json("AAA", RAW["AAA"], "Alpha Corp", {SPLIT_AAA: 2.0}),
        "GOOGL": chart_json("GOOGL", RAW["GOOGL"], "Gamma Holdings", vol=VOLUME["GOOGL"]),
        "GOOG": chart_json("GOOG", RAW["GOOG"], "Gamma Holdings", vol=VOLUME["GOOG"]),
        "META": chart_json("META", RAW["META"], "Meta Platforms Inc"),
        "WRG": chart_json("WRG", RAW["WRG_Y"], "Wrong Issuer Resources"),
        "OLD": chart_json("OLD", RAW["OLD"], "Oldco Industries"),
        "NEW": chart_json("NEW", RAW["NEW"], "Newco Inc"),
        "RSP": chart_json("RSP", RAW["RSP"], "Invesco S&P 500 Equal Weight ETF"),
    }
    for s, b in charts.items():
        r[us_prices.CHART_URL.format(symbol=s)] = b
    cfs = {
        1001: companyfacts(1001, "Alpha Corp", SHARES["AAA"], split=("2019-07-01", 2.0)),
        1002: companyfacts(1002, "Bravo Inc", SHARES["BBB"]),
        1003: companyfacts(1003, "Gamma Holdings", None, eps=False),  # multi-class: no count, no EPS
        1004: companyfacts(1004, "Meta Platforms Inc", SHARES["META"]),
        1005: companyfacts(1005, "Wrong Target Co", SHARES["WRG"]),
        1006: companyfacts(1006, "Newco Inc", SHARES["NEW"]),
        1007: companyfacts(1007, "Oldco Industries", SHARES["OLD"]),
    }
    for c, j in cfs.items():
        r[pit_us.COMPANYFACTS.format(cik=c)] = json.dumps(j).encode()
    return r


class FakeResponse:
    def __init__(self, status: int, body: bytes):
        self.status_code = status
        self.content = body
        self.headers = {"content-type": "application/json"}

    @property
    def text(self) -> str:
        return self.content.decode("latin-1")

    def json(self):
        return json.loads(self.content)


class FakeNet(PitNet):
    """PitNet serving the synthetic world; any other URL answers 404 (like Yahoo/SEC for unknown symbols)."""

    def __init__(self, settings, root: Path, routes: dict[str, bytes]):
        super().__init__(settings, root)
        self.routes = routes
        self.calls: list[str] = []

    def http_get(self, url, kind, params=None, headers=None, ok=(200,), tries=5):
        self.calls.append(url)
        if kind == "sec":
            self.sec_user_agent()
        body = self.routes.get(url)
        if body is None:
            resp = FakeResponse(404, b'{"chart": {"result": null, "error": {"description": "Not Found"}}}')
            if 404 in ok:
                return resp
            raise requests.HTTPError(f"404 {url}", response=resp)
        return FakeResponse(200, body)


def spy_frame(_start) -> pd.DataFrame:
    px = RAW["SPY"]
    return pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1e7}, index=px.index)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Offline tests must never reach the network (the fake PitNet serves everything)."""

    def boom(*a, **k):
        raise AssertionError("network access in an offline test")

    monkeypatch.setattr(requests.Session, "request", boom)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Full offline build of the synthetic world (2018-06 .. 2020-06 month-ends)."""
    tmp = tmp_path_factory.mktemp("pit_us_world")
    mp = pytest.MonkeyPatch()
    mp.setattr(requests.Session, "request", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    mp.setattr(us_membership, "COUNT_RANGE", (5, 9))
    mp.setattr(us_membership, "COUNT_HARD_RANGE", (3, 20))
    mp.setattr(pit_us, "MIN_SHARE_COVERAGE", 0.5)
    mp.setattr(pit_us, "_today", lambda: TODAY)
    settings = settings_for(tmp)
    net = FakeNet(settings, tmp / "pit", build_routes())
    cache = DiskCache(tmp / "dc", enabled=True)
    t0 = time.time()
    full = pit_us._full_build(settings, cache, net, pd.Timestamp("2018-06-01"), None, spy_loader=spy_frame, today=TODAY)
    secs = time.time() - t0
    yield {"full": full, "net": net, "settings": settings, "cache": cache, "secs": secs, "tmp": tmp}
    mp.undo()


# =============================================================================== calendar


def test_nyse_calendar_holidays_and_next_session():
    days = cal.nyse_sessions("2025-01-06", "2025-01-10")
    assert pd.Timestamp("2025-01-09") not in days  # Carter funeral
    assert pd.Timestamp("2026-04-03") not in cal.nyse_sessions("2026-03-30", "2026-04-06")  # Good Friday
    assert pd.Timestamp("2026-06-19") not in cal.nyse_sessions("2026-06-15", "2026-06-22")  # Juneteenth
    assert pd.Timestamp("2021-12-31") in cal.nyse_sessions("2021-12-27", "2022-01-04")  # Sat New Year: no Friday close
    assert cal.next_session("2026-07-02") == pd.Timestamp("2026-07-06")  # Fri 07-03 observed Independence Day
    assert cal.prev_session("2026-01-20") == pd.Timestamp("2026-01-16")  # MLK Monday
    nx = cal.next_sessions(pd.Series(pd.to_datetime(["2026-07-02", "2026-09-25"])))
    assert list(nx) == [pd.Timestamp("2026-07-06"), pd.Timestamp("2026-09-28")]
    me = cal.month_end_sessions(cal.nyse_sessions("2026-01-01", "2026-06-30"))
    assert pd.Timestamp("2026-05-29") in me and pd.Timestamp("2026-04-30") in me


# =============================================================================== membership


def test_pick_fja_file_prefers_updated_then_newest_date():
    names = [
        "README.md",
        "S&P 500 Historical Components & Changes.csv",
        "S&P 500 Historical Components & Changes(12-30-2024).csv",
        "S&P 500 Historical Components & Changes(01-17-2026).csv",
    ]
    assert us_membership.pick_fja_file(names).endswith("(01-17-2026).csv")  # not the lexically last one
    assert "Updated" in us_membership.pick_fja_file([*names, "S&P 500 Historical Components & Changes (Updated).csv"])
    with pytest.raises(us_membership.MembershipError):
        us_membership.pick_fja_file(["README.md"])


def test_membership_reconstruction_at_dates(monkeypatch):
    monkeypatch.setattr(us_membership, "COUNT_RANGE", (7, 8))  # the real 495-510 band scaled to the toy world
    monkeypatch.setattr(us_membership, "COUNT_HARD_RANGE", (3, 20))
    snaps = us_membership.parse_fja_csv((FIX / "fja_small.csv").read_bytes())
    ch = us_membership.parse_wiki_changes((FIX / "wiki_changes_small.html").read_bytes())
    assert "MIS" in set(ch["rem_t"].dropna())  # stray ' |' removed from the ticker cell
    sp, applied, warnings = us_membership.patch_spells(snaps, ch, TODAY)
    at = lambda d: set(us_membership.members_at(sp, d)["ticker"])  # noqa: E731
    assert at("2018-01-02") == {"AAA", "BBB", "FB", "FRC", "GOOGL", "MIS", "OLD", "WRG"}
    assert "GOOG" not in at("2018-05-31") and "GOOG" in at("2018-06-01")
    assert "MIS" in at("2019-05-31") and "MIS" not in at("2019-06-03")  # a spell ends on the FIRST day out
    assert "FB" in at("2019-09-30") and "META" in at("2019-10-01") and "FB" not in at("2019-10-01")
    # Wikipedia rows after fja's last row (2020-03-02) are applied; the announced 2020-07-15 change is not yet
    assert "FRC" in at("2020-03-31") and "FRC" not in at("2020-04-01")
    assert "OLD" in at("2020-04-30") and at("2020-05-01") >= {"NEW"} and "OLD" not in at("2020-05-01")
    assert at("2020-06-30") == {"AAA", "GOOG", "GOOGL", "META", "NEW", "WRG"}
    assert any("2020-07-15" in w for w in warnings) and len(applied) == 2
    cur = us_membership.parse_wiki_current((FIX / "wiki_current_small.html").read_bytes())
    assert list(cur["cik"]) == [1001, 1003, 1003, 1004, 1005, 1006]
    me = pd.DatetimeIndex(["2018-06-29", "2019-12-31", "2020-06-30"])
    info, notes = us_membership.membership_checks(sp, cur, me)
    assert info["current_equal_wikipedia"] and info["month_end_count_out_of_range"] == 2
    assert any("범위 밖" in n for n in notes)
    assert info["month_end_count_min"] == 6 and info["month_end_count_max"] == 9


def test_membership_checks_raise_on_structural_break():
    snaps = us_membership.parse_fja_csv((FIX / "fja_small.csv").read_bytes())
    sp = us_membership.spells_from_snapshots(snaps)  # without the Wikipedia patch: current set differs
    cur = us_membership.parse_wiki_current((FIX / "wiki_current_small.html").read_bytes())
    wrong = cur.assign(sym=["X1", "X2", "X3", "X4", "X5", "X6"])
    with pytest.raises(us_membership.MembershipError, match="불일치"):
        us_membership.membership_checks(sp, wrong, pd.DatetimeIndex([]))
    with pytest.raises(us_membership.MembershipError):
        us_membership.parse_wiki_current(
            "<table><tr><th>Ticker</th><th>Name</th></tr><tr><td>A</td><td>B</td></tr></table>"
        )


def test_manual_patches_split_add_drop_and_guards():
    def spell(t, s, e=None):
        return dict(ticker=t, start=pd.Timestamp(s), end=pd.Timestamp(e) if e else pd.NaT)

    base = pd.DataFrame(
        [
            spell("JCI", "1996-01-02"),
            spell("CCEP", "2016-01-04", "2016-05-31"),
            spell("AGN", "1996-01-02", "2020-05-12"),
        ]
    )
    for c in us_membership.SPELL_COLUMNS[3:]:
        base[c] = np.nan if c == "cik" else None
    base["patch"] = ""
    patches = tuple(p for p in us_membership.MANUAL_PATCHES if p.ticker in ("TYC", "JCI", "CCEP", "AGN"))
    sp, applied, skipped = us_membership.apply_manual_patches(base, patches)
    assert not skipped and len(applied) == len(patches)
    jci = sp[sp["ticker"] == "JCI"].sort_values("start")
    assert list(jci["end"].isna()) == [False, True] and jci["end"].iloc[0] == pd.Timestamp("2016-09-06")
    assert jci["cik"].iloc[0] == 53669 and pd.isna(jci["cik"].iloc[1])  # new part resolves as a current member
    tyc = sp[sp["ticker"] == "TYC"].iloc[0]
    assert tyc["cik"] == 833444 and tyc["yahoo_hint"] == "JCI"
    assert "CCEP" not in set(sp["ticker"])
    agn = sp[sp["ticker"] == "AGN"]
    assert sorted(agn["cik"].dropna().astype(int)) == [850693, 1578845] and len(agn) == 3
    again, applied2, skipped2 = us_membership.apply_manual_patches(sp, patches)  # idempotent: guarded
    assert len(again) == len(sp) and len(skipped2) == len(patches)


def test_manual_patches_backfilled_rename_gets_the_ticker_traded_then():
    """fja back-fills the ~2019 ticker (ARNC in 2014); the pre-rename part is priced/validated under AA, and a rename
    that came with a new registrant (BHI -> BHGE) becomes its own CIK entity."""
    base = pd.DataFrame(
        [
            dict(ticker="ARNC", start=pd.Timestamp("1996-01-02"), end=pd.Timestamp("2020-04-06")),
            dict(ticker="HWM", start=pd.Timestamp("2020-04-06"), end=pd.NaT),
            dict(ticker="BHGE", start=pd.Timestamp("1996-01-02"), end=pd.Timestamp("2019-10-18")),
        ]
    )
    for c in us_membership.SPELL_COLUMNS[3:]:
        base[c] = np.nan if c == "cik" else None
    base["patch"] = ""
    patches = tuple(p for p in us_membership.MANUAL_PATCHES if p.ticker in ("ARNC", "BHGE", "ANDV"))
    sp, applied, skipped = us_membership.apply_manual_patches(base, patches)
    assert len(applied) == 2 and len(skipped) == 1 and "ANDV" in skipped[0]  # no ANDV spell here: guarded
    arnc = sp[sp["ticker"] == "ARNC"].sort_values("start")
    assert list(arnc["end"]) == [pd.Timestamp("2016-11-01"), pd.Timestamp("2020-04-06")]
    assert arnc["ftd_sym"].iloc[0] == "AA" and arnc["cik"].iloc[0] == 4281 and pd.isna(arnc["cik"].iloc[1])
    assert set(us_membership.members_at(sp, "2016-10-31")["ticker"]) == {"ARNC", "BHGE"}  # counts unchanged
    bh = sp[sp["ticker"] == "BHGE"].sort_values("start").iloc[0]
    assert (bh["cik"], bh["ftd_sym"], bh["yahoo_hint"]) == (808362, "BHI", "BKR")
    ev = us_membership.patch_events(sp, pd.Timestamp("2014-01-01")).set_index("ticker")
    assert ev.loc["AA", "cat"] == "renamed" and ev.loc["BHI", "cat"] == "acquired"


def test_removal_events_rename_and_backfill_alias():
    snaps = pd.DataFrame(
        {
            "tickers": [
                frozenset({"AAA", "FB", "AABA"}),
                frozenset({"AAA", "META", "AABA"}),
                frozenset({"AAA", "META"}),
            ]
        },
        index=pd.to_datetime(["2016-01-04", "2016-06-09", "2017-06-19"]),
    )
    ch = pd.DataFrame(
        [
            dict(
                date=pd.Timestamp("2017-06-19"),
                add_t=None,
                add_n=None,
                add_link=None,
                rem_t="YHOO",
                rem_n="Yahoo! Inc.",
                rem_link="Yahoo!",
                reason="Verizon acquired Yahoo's operating business.",
            )
        ]
    )
    sp = us_membership.spells_from_snapshots(snaps)
    ev = us_membership.removal_events(sp, ch, pd.Timestamp("2016-01-01")).set_index("fja_ticker")
    assert ev.loc["FB", "rename_to"] == "META" and ev.loc["FB", "cat"] == "renamed"
    assert ev.loc["AABA", "ticker"] == "YHOO" and ev.loc["AABA", "name"] == "Yahoo! Inc."  # back-filled ticker alias


# =============================================================================== CIK mapping


def test_names_match_and_norm():
    assert us_cik.norm_name("CELGENE CORP /DE/") == "celgene"
    assert us_cik.names_match("QuintilesIMS", "Quintiles IMS Holdings Inc")
    assert us_cik.names_match("Kohl's", "KOHLS Corp")
    assert not us_cik.names_match("Authorize.Net Holdings", "Arista Networks")


def test_cik_for_event_rejects_stale_pair_and_falls_back(tmp_path):
    settings = settings_for(tmp_path)
    net = FakeNet(settings, tmp_path / "pit", build_routes())
    subs = us_cik.Submissions(net)
    pairs = us_cik.sec_ticker_history(net, TODAY)
    lookup = us_cik.cik_lookup_index(net)
    ev = pd.Series(
        dict(
            date=pd.Timestamp("2019-06-03"),
            fja_ticker="MIS",
            ticker="MIS",
            name="Missing Data Corp",
            link=None,
            rename_to=None,
        )
    )
    assert us_cik.cik_for_event(ev, pairs, lookup, {}, subs) == (1008, "m1_wayback")  # 4242 (dead) rejected
    frc = ev.copy()
    frc.update(dict(date=pd.Timestamp("2020-04-01"), fja_ticker="FRC", ticker="FRC", name="First Failed Bank"))
    assert us_cik.cik_for_event(frc, pairs, lookup, {}, subs) == (1009, "m1_unvalidated")  # FDIC filer
    nobody = ev.copy()
    nobody.update(dict(fja_ticker="QQQX", ticker="QQQX", name="Nobody Inc"))
    assert us_cik.cik_for_event(nobody, pairs, lookup, {}, subs) == (None, "unmapped")


# =============================================================================== FTD + Yahoo validation


def test_parse_ftd_text_keeps_pipe_in_description():
    txt = (
        "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20200302|123|ABC|10|ABC INC|CL A|12.5\nx|y\n"
    )
    df = us_prices.parse_ftd_text(txt)
    assert len(df) == 1 and df["price"].iloc[0] == 12.5 and df["symbol"].iloc[0] == "ABC"


def test_ftd_cross_check_rejects_wrong_issuer_and_accepts_right_one():
    raw = RAW["WRG"]
    f = pd.DataFrame(
        {"settle": cal.next_sessions(raw.index), "price": raw.round(2).to_numpy(), "symbol": "WRG", "key": "WRG"}
    )
    right = pd.DataFrame({"close": raw, "volume": 1.0}, index=raw.index)
    wrong = pd.DataFrame({"close": RAW["WRG_Y"], "volume": 1.0}, index=raw.index)
    n, share, _ = us_prices.ftd_consistency(us_prices.ftd_ratio(right, f))
    assert n > 100 and share > 0.95
    n, share, _ = us_prices.ftd_consistency(us_prices.ftd_ratio(wrong, f))
    assert share < 0.1
    spell = pd.Series(dict(ticker="WRG", start=pd.Timestamp("2018-01-02"), end=pd.NaT, yahoo_hint=None))
    charts = {"WRG": {"ok": True, "df": wrong, "meta": {"firstTradeDate": 0}}}
    sym, how, rej = us_prices.resolve_yahoo(
        spell, None, None, {}, {}, f, lambda s: charts.get(s, {"ok": False}), lambda c: [], TODAY, SESS[0]
    )
    assert sym is None and how == "none" and rej[0][1].startswith("ftd_issuer_mismatch")
    charts["WRG"]["df"] = right
    sym, how, _ = us_prices.resolve_yahoo(
        spell, None, None, {}, {}, f, lambda s: charts.get(s, {"ok": False}), lambda c: [], TODAY, SESS[0]
    )
    assert (sym, how) == ("WRG", "spell_ticker")


def test_partly_wrong_series_prefix_is_masked():
    raw = RAW["WRG"]
    mixed = raw.copy()
    head = mixed.index < pd.Timestamp("2018-06-01")
    mixed[head] = RAW["WRG_Y"][head]  # another issuer's history spliced in front (JCI/CB pre-2016 pattern)
    df = pd.DataFrame({"close": mixed, "volume": 1.0}, index=raw.index)
    f = pd.DataFrame({"settle": cal.next_sessions(raw.index), "price": raw.round(2).to_numpy()})
    bad, good = pit_us._month_flags(df, f, SESS[0], TODAY)
    assert bad and max(bad) < pd.Timestamp("2018-06-01") and good
    out, info = pit_us._mask(df, bad, good)
    assert info["policy"] == "prefix_dropped" and out.index[0] >= pd.Timestamp("2018-05-31")
    kept, info2 = pit_us._mask(df, {pd.Timestamp("2019-03-31")}, {pd.Timestamp("2019-01-31")})
    assert info2["policy"] == "suspect_kept" and len(kept) == len(df)  # one isolated month: false positive kept
    # SPGI 2016-04: one glitchy month at the START of the FTD evidence (the new symbol's first month) must not drop
    # the years of history before it that FTD never covered
    kept, info3 = pit_us._mask(df, {pd.Timestamp("2018-01-31")}, {pd.Timestamp("2018-03-31")})
    assert info3["policy"] == "suspect_kept" and len(kept) == len(df)


def test_stub_series_rejected_and_partial_accepted():
    """Yahoo keeps a 1-4 bar stub of some acquired names (AVB 2026): not a price series for the spell."""
    raw = RAW["WRG"]
    spell = pd.Series(dict(ticker="WRG", start=pd.Timestamp("2018-01-02"), end=pd.NaT, yahoo_hint=None))
    stub = pd.DataFrame({"close": raw.iloc[-3:], "volume": 1.0})
    charts = {"WRG": {"ok": True, "df": stub, "meta": {"firstTradeDate": 0}}}
    sym, how, rej = us_prices.resolve_yahoo(
        spell, None, None, {}, {}, None, lambda s: charts.get(s, {"ok": False}), lambda c: [], TODAY, SESS[0]
    )
    assert sym is None and how == "none" and rej == [("WRG", "stub_series(3)")]
    charts["WRG"]["df"] = pd.DataFrame({"close": raw.loc["2019-06-01":], "volume": 1.0})  # a real late start
    sym, how, rej = us_prices.resolve_yahoo(
        spell, None, None, {}, {}, None, lambda s: charts.get(s, {"ok": False}), lambda c: [], TODAY, SESS[0]
    )
    assert (sym, how, rej) == ("WRG", "spell_ticker_partial", [])


def test_ftd_prices_stop_before_the_first_day_out():
    """The removal date is the first day out: a symbol reused from that day on (Alcoa Corp took AA on 2016-11-01
    when Alcoa Inc became ARNC) never enters the old spell's month return or terminal mark."""
    f = pd.DataFrame(
        {
            "px_date": pd.to_datetime(["2016-09-30", "2016-10-28", "2016-10-31", "2016-11-01", "2016-11-02"]),
            "price": [30.0, 31.0, 32.0, 23.0, 23.5],  # 11-01/02: the new issuer under the same symbol
            "symbol": "AA",
            "key": "AA",
        }
    )
    end = pd.Timestamp("2016-11-01")
    r = us_prices.ftd_month_return(f, pd.Timestamp("2016-10-31"), pd.Timestamp("2016-11-30"), end)
    assert r is None  # nothing strictly inside (10-31, 11-01): no made-up -28% move
    r = us_prices.ftd_month_return(f, pd.Timestamp("2016-09-30"), pd.Timestamp("2016-10-31"), end)
    assert r == pytest.approx(32.0 / 30.0 - 1)
    assert us_prices.terminal_mark(f, end, failed=False) == (pd.Timestamp("2016-10-31"), 32.0, "ftd")


def test_terminal_marks_cap_at_removal_and_track_cusip():
    rows = pd.DataFrame(
        {
            "px_date": pd.to_datetime(["2020-02-26", "2020-02-27", "2020-03-12"]),
            "price": [48.0, 49.0, 1.23],
            "symbol": ["BBB", "BBB", "BBB"],
            "key": ["BBB", "BBB", "BBB"],
        }
    )
    assert us_prices.terminal_mark(rows, pd.Timestamp("2020-03-02"), failed=False) == (
        pd.Timestamp("2020-02-27"),
        49.0,
        "ftd",
    )  # the post-delisting print (1.23) is ignored
    frc = pd.DataFrame(
        {
            "px_date": pd.to_datetime(["2020-03-18", "2020-03-19", "2020-04-02"]),
            "price": [60.0, 55.0, 0.33],
            "symbol": ["FRC", "FRC", "FRCB"],
            "key": ["FRC", "FRC", "FRC"],
        }
    )
    assert us_prices.terminal_mark(frc, pd.Timestamp("2020-04-01"), failed=True)[1:] == (0.33, "ftd_otc")
    halted = frc[frc["symbol"] == "FRC"]
    assert us_prices.terminal_mark(halted, pd.Timestamp("2020-04-01"), failed=True)[1:] == (0.0, "failed_assumed_zero")


# =============================================================================== fundamentals (as-of)


def _cf(entity_facts: dict) -> us_fund.CompanyFacts:
    return us_fund.CompanyFacts.from_json({int(entity_facts["cik"]): entity_facts})


def test_asof_fundamentals_amendment_keeps_original_until_refiled():
    j = companyfacts(2001, "Amend Co", 100e6)
    ni = j["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"]
    q = next(f for f in ni if f["end"] == "2019-09-30")
    ni.append({**q, "val": q["val"] * 0.5, "form": "10-Q/A", "filed": "2019-12-16", "accn": "amend-1"})
    df, stats = us_fund.entity_panel("CIK0000002001", {2001: j})
    assert stats["rows"] == len(df) > 10 and set(FUNDAMENTAL_COLUMNS) <= set(df.columns)
    before = df[df["filed"] == pd.Timestamp("2019-11-04")].iloc[0]
    after = df[df["filed"] == pd.Timestamp("2019-12-16")].iloc[0]
    assert before["period_end"] == after["period_end"] == pd.Timestamp("2019-09-30")
    assert before["restated_later"] and not after["restated_later"]
    assert after["ni_parent_ttm"] == pytest.approx(before["ni_parent_ttm"] - q["val"] * 0.5)
    assert before["available_from"] == pd.Timestamp("2019-11-05")  # filed + 1 NYSE session
    assert after["available_from"] == pd.Timestamp("2019-12-17")
    fund = PITFundamentals(market=pit_us.Market.US, panel=df)
    assert fund.asof("CIK0000002001", "2019-12-16")["ni_parent_ttm"] == before["ni_parent_ttm"]  # not yet usable
    assert fund.asof("CIK0000002001", "2019-12-17")["ni_parent_ttm"] == after["ni_parent_ttm"]
    fy = df[df["period_end"] == pd.Timestamp("2018-12-31")].iloc[0]
    assert fy["period_type"] == "FY" and before["period_type"] == "Q"


def test_asof_fundamentals_split_basis_has_no_look_ahead():
    j = companyfacts(2002, "Split Co", 100e6, split=("2019-07-01", 2.0))
    yahoo_splits = pd.Series({pd.Timestamp("2019-07-01"): 2.0})
    df, _ = us_fund.entity_panel("CIK0000002002", {2002: j}, None, yahoo_splits, pd.Timestamp("2013-01-01"))
    pre = df[df["filed"] == pd.Timestamp("2019-05-05")].iloc[0]
    post = df[df["filed"] == pd.Timestamp("2019-08-04")].iloc[0]
    assert pre["shares_out"] == pytest.approx(100e6)  # the later split does NOT leak into the earlier row
    assert post["shares_out"] == pytest.approx(200e6)
    assert pre["eps_ttm"] == pytest.approx(pre["ni_parent_ttm"] / 100e6, rel=1e-3)
    assert post["eps_ttm"] == pytest.approx(post["ni_parent_ttm"] / 200e6, rel=1e-3)
    det = us_fund.detect_splits_xbrl(_cf(j))  # XBRL-only split detection from re-filed EPS
    assert len(det) == 1 and det["ratio"].iloc[0] == 2 and det["window_hi"].iloc[0] >= "2019-07-01"


def test_multiclass_share_count_is_nan_not_wrong():
    j = companyfacts(2003, "Two Class Co", 50e6, eps=False)  # cover count of ONE class, no EPS to check it
    df, _ = us_fund.entity_panel("CIK0000002003", {2003: j})
    assert df["shares_out"].isna().all() and df["ni_parent_ttm"].notna().any()


def test_split_windows_merge_with_any_overlapping_window():
    eps = []
    for (s, e), (v0, v1) in {
        ("2020-01-01", "2020-03-31"): (5.0, 1.0),
        ("2020-04-01", "2020-06-30"): (6.0, 1.2),
        ("2019-10-01", "2019-12-31"): (4.0, 0.8),
    }.items():
        eps.append({"start": s, "end": e, "val": v0, "filed": "2020-07-28", "form": "10-Q", "accn": "a"})
        eps.append({"start": s, "end": e, "val": v1, "filed": "2020-10-26", "form": "10-Q", "accn": "b"})
    eps.append({"start": "2019-10-01", "end": "2019-12-31", "val": 4.0, "filed": "2020-02-13", "form": "10-K"})
    eps.append({"start": "2019-10-01", "end": "2019-12-31", "val": 0.8, "filed": "2021-02-08", "form": "10-K"})
    for f in eps:
        f.setdefault("accn", "c")
    cf = us_fund.CompanyFacts(ciks=[1], gaap={"EarningsPerShareDiluted": {"USD/shares": eps}})
    det = us_fund.detect_splits_xbrl(cf)
    assert len(det) == 1 and det["ratio"].iloc[0] == 5  # the TSLA 2020 5:1 split once, not twice


def test_real_jpm_amendment_fixture():
    """JPM Q1-2012: 5.383B in the 10-Q (2012-05-10), 4.924B after the 10-Q/A (2012-08-09) — real SEC facts."""
    j = json.loads(gzip.decompress((FIX / "jpm_companyfacts_2011_2013.json.gz").read_bytes()))
    cf = _cf(j)
    q = {d: us_fund.pit_quarterly(cf, "net_income", d).set_index("end")["value"] for d in ("2012-05-10", "2012-08-09")}
    assert q["2012-05-10"][pd.Timestamp("2012-03-31")] == 5_383_000_000  # as originally filed
    assert q["2012-08-09"][pd.Timestamp("2012-03-31")] == 4_924_000_000  # after the 10-Q/A
    df, _ = us_fund.entity_panel("CIK0000019617", {19617: j}, since="2011-01-01")
    q1 = df[df["period_end"] == pd.Timestamp("2012-03-31")]
    assert len(q1) == 1 and q1["restated_later"].iloc[0] and q1["filed"].iloc[0] == pd.Timestamp("2012-05-10")
    fund = PITFundamentals(market=pit_us.Market.US, panel=df)
    early = fund.asof("CIK0000019617", "2012-06-29")
    late = fund.asof("CIK0000019617", "2012-08-10")
    assert early["period_end"] == pd.Timestamp("2012-03-31") and late["period_end"] == pd.Timestamp("2012-06-30")
    assert early["shares_out"] > 3e9 and early["equity_parent"] > 1e11


# =============================================================================== full offline build


def test_world_contract_and_cik_collapse(world):
    u = pit_us.select_top_n(world["full"], 500)
    assert isinstance(u, PITUniverse) and u.market.value == "US"
    assert list(u.membership.columns) == MEMBERSHIP_COLUMNS
    m = u.membership
    assert m["date"].min() == pd.Timestamp("2018-06-29") and m["date"].max() == pd.Timestamp("2020-06-30")
    assert m.groupby("date")["entity"].apply(lambda s: s.is_unique).all()
    # share classes collapse: GOOGL + GOOG -> one entity; FB -> META rename -> one entity with a changing ticker
    at = m[m["date"] == pd.Timestamp("2018-06-29")]
    assert sorted(at["entity"]) == sorted(f"CIK{c:010d}" for c in (1001, 1002, 1003, 1004, 1005, 1007, 1008, 1009))
    meta_rows = m[m["entity"] == "CIK0000001004"].set_index("date")["ticker"]
    assert meta_rows[pd.Timestamp("2019-09-30")] == "FB" and meta_rows[pd.Timestamp("2019-10-31")] == "META"
    assert m[m["entity"] == "CIK0000001003"]["ticker"].isin(["GOOGL", "GOOG"]).all()
    assert u.members_on("2020-06-15") == u.members_on(pd.Timestamp("2020-05-29"))
    assert "CIK0000001006" in u.members_on("2020-05-29") and "CIK0000001007" not in u.members_on("2020-05-29")
    # prices: contract frames ending at the last traded session, total-return adjusted where validated
    for e, px in u.prices.items():
        assert list(px.columns) == [*PRICE_COLUMNS, "value"]  # + the day's traded value (docs/MODELING.md 84.3)
        assert px.index.is_monotonic_increasing and px.index.is_unique
        assert px.index.tz is None and px.index[-1] <= TODAY
        assert px.attrs["adjusted"] is (e != "CIK0000001006") and px.attrs["validated"] is (e != "CIK0000001006")
    assert u.coverage["unvalidated_frames"] == ["CIK0000001006"]
    new = u.prices["CIK0000001006"]
    assert new["close"].iloc[-1] == pytest.approx(RAW["NEW"].iloc[-1], rel=1e-3)  # split-adjusted, no dividends
    assert "CIK0000001002" not in u.prices and "CIK0000001005" not in u.prices  # 404 / wrong issuer
    assert u.prices["CIK0000001004"].attrs["symbol"] == "META"
    aaa = u.prices["CIK0000001001"]
    jump = aaa["close"].pct_change().abs().max()
    assert jump < 0.2  # split-adjusted: no 50% drop at the 2019-07-01 split
    assert set(u.benchmarks) == {"SPY"} and set(u.entity_benchmark.values()) == {"SPY"}
    panel = u.to_price_panel()
    assert set(panel.membership["ticker"]) <= set(panel.prices)
    meta = u.meta.set_index("entity")
    assert {"entity", "ticker", "name", "market", "first_date", "last_date", "delisted", "delist_date"} <= set(
        u.meta.columns
    )
    assert {"delist_reason", "price_source", "cik", "notes"} <= set(u.meta.columns)
    assert meta.loc["CIK0000001005", "price_source"] == "ftd"
    assert meta.loc["CIK0000001008", "price_source"] == "none"


def test_world_ftd_rejection_fill_and_coverage(world):
    full = world["full"]
    u = pit_us.select_top_n(full, 500)
    cov = u.coverage
    assert cov["rejected_series"] >= 1 and any(x.startswith("WRG->WRG") for x in cov["rejected_examples"])
    ms = full.month_src.set_index(["date", "entity"])["src"]
    assert ms[(pd.Timestamp("2019-01-31"), "CIK0000001002")] == "ftd"  # acquired, no Yahoo: FTD-filled
    assert ms[(pd.Timestamp("2019-01-31"), "CIK0000001005")] == "ftd"  # wrong Yahoo issuer: FTD-filled
    assert ms[(pd.Timestamp("2019-01-31"), "CIK0000001008")] == "missing"  # nothing anywhere: counted, not dropped
    assert ms[(pd.Timestamp("2019-01-31"), "CIK0000001001")] == "yahoo"
    tot = cov["member_months"]
    assert tot == cov["member_months_with_prices"] + cov["member_months_ftd_only"] + cov["member_months_missing"]
    assert tot == len(u.membership)
    by_year = cov["by_year"]
    assert set(by_year) == {2018, 2019, 2020}
    assert sum(v["member_months"] for v in by_year.values()) == tot
    assert by_year[2019]["missing"] >= 5  # MIS 2019-01..05
    assert "CIK0000001002" in cov["no_price_entities"] and "CIK0000001008" in cov["no_price_entities"]
    assert cov["membership_checks"]["current_equal_wikipedia"]
    assert cov["cik_methods"]["m1_unvalidated"] >= 1  # the FDIC filer
    tc = cov["tracking_check"]
    assert tc["months"] >= 20 and isinstance(tc["ok"], bool) and np.isfinite(tc["te"])
    assert any("추적 점검" in n for n in u.notes) and any("회원-월" in n for n in u.notes)
    assert all(any("가" <= ch <= "힣" for ch in n) for n in u.notes)  # Korean user-facing notes


def test_world_terminal_prices(world):
    u = pit_us.select_top_n(world["full"], 500)
    bbb = pit_us.terminal_price(u, "CIK0000001002")
    assert bbb is not None and bbb[2] == "ftd" and bbb[0] <= pd.Timestamp("2020-03-02")
    assert bbb[1] == pytest.approx(round(float(RAW["BBB"].loc[bbb[0]]), 2))  # the 1.23 prints after delisting ignored
    frc = pit_us.terminal_price(u, "CIK0000001009")
    assert frc[1:] == (0.33, "ftd_otc")
    assert pit_us.terminal_price(u, "CIK0000001001") is None  # still trading
    assert pit_us.terminal_price(u, "CIK0000001007") is None  # demoted, still trading
    meta = u.meta.set_index("entity")
    assert bool(meta.loc["CIK0000001002", "delisted"]) and meta.loc["CIK0000001002", "delist_reason"] == "인수·합병"
    ms = world["full"].month_src.set_index(["date", "entity"])
    r = ms.loc[(pd.Timestamp("2020-03-31"), "CIK0000001009")]
    assert r["src"] == "ftd" and r["ret"] < -0.99  # the failure month uses the OTC mark (-99%+)


def test_world_rank_top_n_by_pit_market_cap(world):
    full = world["full"]
    u2 = pit_us.select_top_n(full, 2)
    per_date = u2.membership.groupby("date")["entity"].apply(set)
    assert (per_date == {"CIK0000001003", "CIK0000001004"}).all()  # GOOGL (imputed) and META
    mem = full.members.set_index(["date", "entity"])
    g = mem.loc[(pd.Timestamp("2019-01-31"), "CIK0000001003")]
    assert g["rank_basis"] == "adv_imputed_mcap" and g["rank"] == 1
    a = mem.loc[(pd.Timestamp("2019-01-31"), "CIK0000001001")]
    raw_close = RAW["AAA"].loc[:"2019-01-31"].iloc[-1]
    assert a["rank_value"] == pytest.approx(raw_close * 100e6, rel=0.01)  # pre-split shares x raw price
    a2 = mem.loc[(pd.Timestamp("2019-12-31"), "CIK0000001001")]
    assert a2["rank_value"] == pytest.approx(RAW["AAA"].loc[:"2019-12-31"].iloc[-1] * 200e6, rel=0.01)
    w = mem.loc[(pd.Timestamp("2019-01-31"), "CIK0000001005")]
    assert w["value_source"] == "ftd" and w["rank_value"] == pytest.approx(
        RAW["WRG"].loc[:"2019-01-31"].iloc[-1] * 40e6, rel=0.02
    )
    assert np.isnan(mem.loc[(pd.Timestamp("2019-01-31"), "CIK0000001009"), "rank_value"])  # no XBRL, no Yahoo
    assert u2.coverage["n"] == 2 and u2.coverage["rank_value_missing_outside_top_n"] >= 1
    assert set(u2.prices) <= {"CIK0000001003", "CIK0000001004"}


def test_world_fundamentals_loader_and_cache(world):
    settings, net = world["settings"], world["net"]
    fund = pit_us.load_us_pit_fundamentals(settings, world["cache"], None, net=net)  # entities of the last build
    assert isinstance(fund, PITFundamentals) and set(FUNDAMENTAL_COLUMNS) <= set(fund.panel.columns)
    ents = set(fund.panel["entity"])
    assert {"CIK0000001001", "CIK0000001004"} <= ents and "CIK0000001009" not in ents
    assert "CIK0000001009" in fund.coverage["entities_without_xbrl"]
    bbb = fund.panel[fund.panel["entity"] == "CIK0000001002"]
    assert bbb["filed"].max() <= pd.Timestamp("2020-02-28") + pd.Timedelta(pit_us.DELIST_GRACE_DAYS, "D")
    assert fund.coverage["rows_dropped_after_delisting"] >= 1
    assert (fund.panel["available_from"] > fund.panel["filed"]).all()
    assert any("XBRL" in n for n in fund.notes)
    calls = len(net.calls)
    pit_us.load_us_pit_fundamentals(settings, world["cache"], ["CIK0000001001"], net=net)
    assert len(net.calls) == calls  # warm: companyfacts + panels served from the pit cache


def test_world_build_is_cached(world):
    net = world["net"]
    calls = len(net.calls)
    again = pit_us._full_build(
        world["settings"], world["cache"], net, pd.Timestamp("2018-06-01"), None, spy_loader=spy_frame, today=TODAY
    )
    assert len(net.calls) == calls and len(again.members) == len(world["full"].members)


def test_fundamentals_incremental_refresh(tmp_path, monkeypatch):
    """A new filing recomputes only the snapshots from its filing day on; older rows stay byte-identical, while the
    forward-looking restated_later flag of an older row is refreshed."""
    monkeypatch.setattr(pit_us, "_today", lambda: TODAY)
    settings = settings_for(tmp_path)
    net = FakeNet(settings, tmp_path / "pit", build_routes())
    ent = ["CIK0000001002"]
    f1 = pit_us._fundamentals(settings, net, ent, {}, {}, {}, "2009-06-01", 1)
    assert f1.coverage["timing_sec"]["computed"] == 1 and not f1.panel["restated_later"].any()
    url = pit_us.COMPANYFACTS.format(cik=1002)
    j = json.loads(net.routes[url])
    ni = j["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"]
    q = next(f for f in ni if f["end"] == "2019-09-30")
    ni.append({**q, "val": q["val"] * 0.5, "form": "10-Q/A", "filed": "2020-06-01", "accn": "amend-2"})
    net.routes[url] = json.dumps(j).encode()
    net.path("xbrl", "facts/1002.json.gz").unlink()  # force the refetch of companyfacts
    f2 = pit_us._fundamentals(settings, net, ent, {}, {}, {}, "2009-06-01", 1)
    assert f2.coverage["timing_sec"]["incremental"] == 1
    cols = [c for c in f1.panel.columns if c != "restated_later"]
    old1 = f1.panel[f1.panel["filed"] < pd.Timestamp("2020-06-01")][cols].reset_index(drop=True)
    old2 = f2.panel[f2.panel["filed"] < pd.Timestamp("2020-06-01")][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(old1, old2)
    flag = f2.panel.set_index("filed")["restated_later"]
    assert flag[pd.Timestamp("2019-11-04")] and not flag[pd.Timestamp("2020-06-01")]
    f3 = pit_us._fundamentals(settings, net, ent, {}, {}, {}, "2009-06-01", 1)
    assert f3.coverage["timing_sec"]["computed"] == 0  # nothing new: served from the panel cache


def test_companyfacts_refetched_when_the_submissions_show_a_newer_periodic_report(tmp_path):
    """docs/MODELING.md 84.5: a companyfacts copy that lacks the newest 10-K / 10-Q of the (daily refreshed)
    submissions list is fetched again once it is 12 hours old - the weekly rule alone could leave the last filing
    before a vintage cutoff out of the frozen copy, and an inactive-looking stale copy was never refreshed."""
    import os

    url = pit_us.COMPANYFACTS.format(cik=1002)
    net = FakeNet(settings_for(tmp_path), tmp_path / "pit", {url: json.dumps(companyfacts(1002, "B", 50e6)).encode()})
    p = net.path("xbrl", "facts/1002.json.gz")

    def age(hours: float) -> None:
        t = time.time() - hours * 3600
        os.utime(p, (t, t))

    pit_us._companyfacts(net, 1002)  # last periodic fact filed 2020-05-05 (inactive today: weekly rule never fires)
    n = len(net.calls)
    newer = {"filings": {"recent": {"form": ["10-Q", "8-K"], "filingDate": ["2020-08-04", "2020-08-10"]}}}
    net.write_json("sec", "subs/1002.json.gz", newer)
    pit_us._companyfacts(net, 1002)
    assert len(net.calls) == n  # younger than 12 hours: kept
    age(13)
    pit_us._companyfacts(net, 1002)
    assert net.calls[n:] == [url]  # behind the submissions list: fetched again
    pit_us._companyfacts(net, 1002)
    assert len(net.calls) == n + 1  # the source still lacks it: tried again only after another 12 hours
    net.write_json(
        "sec", "subs/1002.json.gz", {"filings": {"recent": {"form": ["10-Q"], "filingDate": ["2020-05-05"]}}}
    )
    age(13)
    pit_us._companyfacts(net, 1002)
    assert len(net.calls) == n + 1  # nothing newer: the weekly / activity rule as before


def test_cik_chain_rules(tmp_path, monkeypatch):
    ts = pd.Timestamp
    pairs = pd.DataFrame(
        [
            ("HOLD", 3001, "Holdco Industries Inc", ts("2017-08-28"), ts("2019-03-01")),
            ("HOLD", 3002, "HOLDCO INDUSTRIES INC", ts("2019-04-01"), ts("2020-06-30")),  # new holdco registrant
            ("ACQ", 3003, "Target Corp", ts("2017-08-28"), ts("2020-01-01")),
            ("ACQ", 3004, "Acquirer Target Corp", ts("2020-02-01"), ts("2020-06-30")),  # established acquirer
            ("BKR", 3007, "Michael Baker Corp", ts("2017-08-28"), ts("2017-06-01")),
            ("BKR", 3008, "Baker Hughes Co", ts("2017-07-01"), ts("2020-06-30")),  # shared word only
            ("UNI", 3005, "Unity Co", ts("2017-08-28"), ts("2019-12-01")),
            ("UNI", 3006, "Unity Co", ts("2020-01-01"), ts("2020-06-30")),  # predecessor is its own S&P entity
        ],
        columns=["ticker", "cik", "title", "first_seen", "last_seen"],
    )
    monkeypatch.setattr(us_cik, "sec_ticker_history", lambda net, today: pairs)
    monkeypatch.setattr(pit_us, "_today", lambda: TODAY)
    net = FakeNet(settings_for(tmp_path), tmp_path / "pit", {})
    first = {3002: "2019-05-10", 3004: "2011-02-01", 3006: "2020-02-10", 3008: "2017-08-01", 1652044: "2015-10-29"}
    ents = [f"CIK{c:010d}" for c in first]
    chains, how = pit_us._chains(net, ents, first, universe_ciks={*first, 3005})
    assert chains["CIK0000003002"] == [3001, 3002] and how["CIK0000003002"] == "wayback_handover"
    assert chains["CIK0000003004"] == [3004]  # an acquirer taking the ticker is not a successor
    assert chains["CIK0000003008"] == [3008]  # 'Michael Baker' is not 'Baker Hughes'
    assert chains["CIK0000003006"] == [3006]
    assert chains["CIK0001652044"] == [1288776, 1652044] and how["CIK0001652044"] == "manual"


def test_ftd_fill_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(us_membership, "COUNT_RANGE", (5, 9))
    monkeypatch.setattr(us_membership, "COUNT_HARD_RANGE", (3, 20))
    monkeypatch.setattr(pit_us, "_today", lambda: TODAY)
    settings = load_settings(
        data={"sec_user_agent": UA, "cache_dir": str(tmp_path / "cache")}, pit={"us": {"ftd_fill": False}}
    )
    net = FakeNet(settings, tmp_path / "pit", build_routes())
    full = pit_us._full_build(
        settings, DiskCache(tmp_path / "dc"), net, pd.Timestamp("2019-06-01"), None, spy_loader=spy_frame, today=TODAY
    )
    ms = full.month_src.set_index(["date", "entity"])["src"]
    assert ms[(pd.Timestamp("2019-07-31"), "CIK0000001002")] == "missing"  # no FTD fill
    assert ms[(pd.Timestamp("2019-07-31"), "CIK0000001001")] == "yahoo"
    assert full.meta.set_index("entity").loc["CIK0000001002", "terminal_source"] is None
    assert full.coverage["rejected_series"] >= 1  # FTD still validates the Yahoo series
    assert any("ftd_fill=false" in n for n in full.notes)


def test_sec_user_agent_is_required(tmp_path):
    settings = settings_for(tmp_path, ua="")
    with pytest.raises(SecUserAgentError, match="SEC_USER_AGENT"):
        pit_us.load_us_pit_universe(settings, DiskCache(tmp_path / "dc"), 100, date(2018, 1, 1))
    with pytest.raises(SecUserAgentError):
        pit_us.load_us_pit_fundamentals(settings, DiskCache(tmp_path / "dc"), ["CIK0000000001"])


def test_fixture_files_are_small():
    for p in FIX.iterdir():
        assert p.stat().st_size < 300_000, p


# =============================================================================== live


@pytest.mark.live
def test_live_universe_2018_2026_top100(monkeypatch):
    _real_world(monkeypatch)  # re-enable the network and the real calendar
    settings = load_settings(data={"sec_user_agent": UA})
    t0 = time.time()
    u = pit_us.load_us_pit_universe(settings, DiskCache(settings.cache_path), 100, date(2018, 1, 1))
    secs = time.time() - t0
    cov = u.coverage
    print(f"\nlive universe 2018-2026 top-100: {secs:.0f}s", json.dumps(cov["by_year"], indent=0))
    print(
        json.dumps({k: cov[k] for k in ("member_months", "member_months_with_prices", "tracking_check")}, default=str)
    )
    for n in u.notes:
        print("NOTE", n)
    assert u.membership.groupby("date").size().eq(100).all()
    assert cov["membership_checks"]["current_equal_wikipedia"]
    assert (cov["member_months_with_prices"] + cov["member_months_ftd_only"]) / cov["member_months"] > 0.95
    assert len(u.prices) >= 100


@pytest.mark.live
def test_live_fundamentals_five_ciks_incl_delisted(monkeypatch):
    _real_world(monkeypatch)
    settings = load_settings(data={"sec_user_agent": UA})
    ents = ["CIK0000320193", "CIK0000019617", "CIK0001652044", "CIK0000816284", "CIK0001418091"]  # CELG, TWTR delisted
    t0 = time.time()
    fund = pit_us.load_us_pit_fundamentals(
        settings, DiskCache(settings.cache_path), ents, yahoo_splits={}, chart_from={}, delist={}
    )
    print(f"\nlive fundamentals 5 CIKs: {time.time() - t0:.1f}s rows={len(fund.panel)}")
    assert set(fund.panel["entity"]) == set(ents)
    jpm = fund.asof("CIK0000019617", "2012-06-29")
    assert jpm["period_end"] == pd.Timestamp("2012-03-31") and jpm["restated_later"]
    celg = fund.asof("CIK0000816284", "2019-06-28")
    assert celg is not None and celg["shares_out"] > 5e8
    assert fund.coverage["cik_chains"]["CIK0001652044"] == [1288776, 1652044]


def test_partial_intraday_bar_is_refetched_after_the_close():
    from quantcio.data.pit.us_prices import SESSION_FINAL_UTC, partial_last_bar

    day = pd.Timestamp("2026-09-25")
    close = (day + SESSION_FINAL_UTC).timestamp()
    assert partial_last_bar(day, fetched=close - 3600, now=close + 60)  # fetched intraday, now final
    assert not partial_last_bar(day, fetched=close - 3600, now=close - 60)  # the session is still open
    assert not partial_last_bar(day, fetched=close + 600, now=close + 3600)  # fetched after the close
