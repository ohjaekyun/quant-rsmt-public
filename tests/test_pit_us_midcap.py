"""Tests for quantcio.data.pit.us_midcap (US XSH universe: S&P 400 members never in the PIT S&P 500).

Offline only: a small synthetic world served by a fake ``PitNet`` (Wikipedia S&P 400 page fixture + month-end page
revisions, SEC ticker files / submissions / fails-to-deliver zips, Yahoo charts, companyfacts) and a stub PIT S&P 500
universe for the exclusion. Holdout hygiene: nothing here computes a return statistic of the universe.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from quantcio.config import load_settings
from quantcio.data.cache import DiskCache
from quantcio.data.pit import MEMBERSHIP_COLUMNS, PITUniverse, load_pit_xsh_universe, us_cik, us_membership, us_prices
from quantcio.data.pit import us as pit_us
from quantcio.data.pit import us_calendar as cal
from quantcio.data.pit import us_midcap as um
from quantcio.data.pit.us_net import PitNet
from quantcio.models import PRICE_COLUMNS, Market

FIX = Path(__file__).parent / "fixtures" / "pit_us_midcap"
FIX_US = Path(__file__).parent / "fixtures" / "pit_us"
UA = "QuantCIO Research research@example.com"
TODAY = pd.Timestamp("2020-06-30")
START = pd.Timestamp("2019-01-01")
SESS = cal.nyse_sessions("2017-01-03", TODAY)
RENAME = pd.Timestamp("2019-08-01")  # OLDT -> NEWT (same issuer, CIK 2004)
CIK = {
    "MIDA": 2001,
    "MIDB": 2002,
    "OLDM": 2003,
    "NEWT": 2004,
    "WRG": 2006,  # a mid cap using a ticker an S&P 500 member used (another issuer)
    "GONE": 2007,
    "MIDC": 2008,
    "LEFT": 2009,
    "PROM": 1001,  # promoted to the S&P 500 (the S&P 500 entity CIK0000001001)
    "NWC": 1006,  # in today's S&P 500 table (fixtures/pit_us/wiki_current_small.html: NEW, CIK 1006)
}
NAMES = {
    "MIDA": "Mida Corp",
    "MIDB": "Midb Inc",
    "OLDM": "Oldm Industries",
    "NEWT": "Newt Holdings",
    "WRG": "Wrongway Corp",
    "GONE": "Gone Corp",
    "MIDC": "Midc Co",
    "LEFT": "Left Corp",
    "PROM": "Promo Corp",
    "NWC": "Newco Inc",
}
SHARES = {"MIDA": 90e6, "MIDB": 60e6, "OLDM": 50e6, "NEWT": 40e6, "GONE": 30e6, "LEFT": 20e6, "WRG": 10e6}
LAST = {"OLDM": pd.Timestamp("2019-09-13")}  # acquired: last traded session
CUSIP = {k: f"{i:06d}XX{i % 10}" for i, k in enumerate(sorted(CIK), start=1)}


def settings_for(tmp_path: Path):
    return load_settings(data={"sec_user_agent": UA, "cache_dir": str(tmp_path / "cache")})


# =============================================================================== synthetic world


def walk(p0: float, seed: int) -> pd.Series:
    r = np.random.default_rng(seed).normal(0.0002, 0.01, len(SESS))
    return pd.Series(p0 * np.exp(np.cumsum(r)), index=SESS)


RAW = {k: walk(20 + 5 * i, i) for i, k in enumerate(sorted(CIK))}
RAW = {k: (v[v.index <= LAST[k]] if k in LAST else v) for k, v in RAW.items()}
RAW["SPY"] = walk(300, 99)


def _epoch(d: pd.Timestamp) -> int:
    return int((pd.Timestamp(d).tz_localize("America/New_York") + pd.Timedelta(570, "min")).timestamp())


def chart_json(symbol: str, raw: pd.Series, name: str) -> bytes:
    close = raw.round(4)
    res = {
        "meta": {
            "symbol": symbol,
            "currency": "USD",
            "exchangeName": "NYQ",
            "instrumentType": "EQUITY",
            "firstTradeDate": _epoch(pd.Timestamp("2005-01-03")),
            "longName": name,
            "exchangeTimezoneName": "America/New_York",
        },
        "timestamp": [_epoch(d) for d in raw.index],
        "events": {"splits": {}},
        "indicators": {
            "quote": [
                {
                    "open": close.tolist(),
                    "high": (close * 1.01).round(4).tolist(),
                    "low": (close * 0.99).round(4).tolist(),
                    "close": close.tolist(),
                    "volume": [1e6] * len(close),
                }
            ],
            "adjclose": [{"adjclose": (close * np.linspace(0.98, 1.0, len(close))).round(4).tolist()}],
        },
    }
    return json.dumps({"chart": {"result": [res], "error": None}}).encode()


def ftd_zips() -> dict[str, bytes]:
    """SEC fails-to-deliver prints (raw close of the session before settlement) of every name but MIDC; OLDT prints
    until the rename, NEWT from then on (same CUSIP)."""
    by_key: dict[str, list[str]] = {}
    for k, px in RAW.items():
        if k in ("MIDC", "SPY"):
            continue
        for d, p in px[px.index >= pd.Timestamp("2018-10-01")].items():
            sym = k
            if k == "NEWT":
                sym = "OLDT" if d < RENAME else "NEWT"
            settle = cal.next_session(d)
            key = settle.strftime("%Y%m") + ("a" if settle.day <= 15 else "b")
            by_key.setdefault(key, []).append(f"{settle:%Y%m%d}|{CUSIP[k]}|{sym}|1000|{sym} COMMON STOCK|{p:.2f}")
    out = {}
    for key, lines in by_key.items():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(
                f"cnsfails{key}", "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n" + "\n".join(lines)
            )
        out[key] = buf.getvalue()
    return out


def companyfacts(cik: int, name: str, shares: float) -> dict:
    ends = pd.date_range("2017-03-31", "2020-03-31", freq="QE")
    rev, ni, eps, eq, cov = [], [], [], [], []
    for i, e in enumerate(ends):
        q4 = e.month == 12
        filed = str((e + pd.Timedelta(55 if q4 else 35, "D")).date())
        form, accn = ("10-K" if q4 else "10-Q"), f"{cik:010d}-{e:%y%m}"
        start = str((e - pd.offsets.QuarterEnd(1) + pd.Timedelta(1, "D")).date())
        base = {"accn": accn, "form": form, "filed": filed}
        rev.append({**base, "start": start, "end": str(e.date()), "val": 5e8})
        ni.append({**base, "start": start, "end": str(e.date()), "val": 5e7})
        eps.append({**base, "start": start, "end": str(e.date()), "val": round(5e7 / shares, 4)})
        eq.append({**base, "end": str(e.date()), "val": 2e9})
        cov.append({**base, "end": str((pd.Timestamp(filed) - pd.Timedelta(5, "D")).date()), "val": shares})
    gaap = {
        "Revenues": {"units": {"USD": rev}},
        "NetIncomeLoss": {"units": {"USD": ni}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": eps}},
        "StockholdersEquity": {"units": {"USD": eq}},
    }
    return {
        "cik": cik,
        "entityName": name,
        "facts": {"us-gaap": gaap, "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": cov}}}},
    }


def submissions(name: str, former: list[str] | None = None) -> dict:
    dates = [str(d.date()) for d in pd.date_range("2017-02-15", "2020-05-15", freq="QS-FEB")]
    return {
        "name": name,
        "formerNames": [{"name": n} for n in former or []],
        "filings": {
            "recent": {
                "form": ["10-Q"] * len(dates),
                "filingDate": dates,
                "accessionNumber": [f"a{i}" for i in range(len(dates))],
            },
            "files": [],
        },
    }


def ticker_file(pairs: list[tuple[str, int, str]]) -> bytes:
    return json.dumps({str(i): {"cik_str": c, "ticker": t, "title": n} for i, (t, c, n) in enumerate(pairs)}).encode()


def world_page() -> bytes:
    """The fixture page without its two-share-class row (kept for the parser test only)."""
    html = (FIX / "wiki400_small.html").read_text()
    html = re.sub(r"<tr><td>UA/UAA</td>.*?</tr>\n", "", html).replace('rowspan="2">March', 'rowspan="1">March')
    return html.replace('<td rowspan="2">S&amp;P 500 constituent', "<td>S&amp;P 500 constituent").encode()


def truth_at(d: pd.Timestamp) -> list[str]:
    """Tickers listed by the page revision in force at month-end d (as editors kept the table then)."""
    t = ["MIDA", "MIDC", "WRG", "PROM", "OLDM", "LEFT"]
    if d >= pd.Timestamp("2019-02-01"):
        t.append("OLDT" if d < RENAME else "NEWT")
    if d >= pd.Timestamp("2019-03-18"):
        t = [x for x in t if x != "LEFT"] + ["MIDB"]
    if pd.Timestamp("2019-05-20") <= d < pd.Timestamp("2019-11-29"):
        t.append("GONE")  # the removal of GONE is missing from the change table
    if d >= pd.Timestamp("2019-09-16"):
        t.remove("OLDM")
    if d >= pd.Timestamp("2020-03-16"):
        t = [x for x in t if x != "PROM"] + ["NWC"]
    return t


def revision_html(tickers: list[str]) -> str:
    rows = "".join(f"<tr><td>{t}</td><td>{t} Co</td><td>x</td><td>y</td><td>reports</td></tr>" for t in tickers)
    return (
        '<table class="wikitable"><tr><th>Ticker symbol</th><th>Security</th><th>GICS Sector</th>'
        f"<th>GICS Sub Industry</th><th>SEC filings</th></tr>{rows}</table>"
    )


def revisions(stale_early: bool) -> dict[int, tuple[pd.Timestamp, list[str]]]:
    me = cal.month_end_sessions(cal.nyse_sessions("2018-11-01", TODAY))
    out = {}
    for i, d in enumerate(me, start=1000):
        t = truth_at(d)
        if stale_early and d <= pd.Timestamp("2019-02-28"):
            t = [x for x in t if x not in ("OLDT", "LEFT", "PROM")] + ["STALE1", "STALE2", "STALE3"]
        out[i] = (d + pd.Timedelta(12, "h"), t)
    return out


def build_routes() -> dict[str, bytes]:
    r: dict[str, bytes] = {
        um.WIKI_400: world_page(),
        us_membership.WIKI_CURRENT: (FIX_US / "wiki_current_small.html").read_bytes(),
    }
    stamps = ["20190115000000", "20190615000000", "20191015000000", "20200315000000"]
    r[us_cik.WAYBACK_CDX] = json.dumps([["timestamp", "digest", "length"]] + [[t, "x", "1"] for t in stamps]).encode()
    base = [(t, CIK[t], NAMES[t]) for t in ("MIDA", "MIDB", "WRG", "GONE", "MIDC", "LEFT", "PROM", "NWC")]
    r[us_cik.WAYBACK_SNAP.format(ts=stamps[0])] = ticker_file(
        [*base, ("OLDM", 2003, "Oldm Industries"), ("OLDT", 2004, "Oldt Holdings")]
    )
    r[us_cik.WAYBACK_SNAP.format(ts=stamps[1])] = ticker_file(
        [*base, ("OLDM", 2003, "Oldm Industries"), ("OLDT", 2004, "Oldt Holdings")]
    )
    r[us_cik.WAYBACK_SNAP.format(ts=stamps[2])] = ticker_file([*base, ("NEWT", 2004, "Newt Holdings")])
    r[us_cik.WAYBACK_SNAP.format(ts=stamps[3])] = ticker_file([*base, ("NEWT", 2004, "Newt Holdings")])
    r[us_cik.SEC_TICKERS] = ticker_file([*[b for b in base if b[0] != "GONE"], ("NEWT", 2004, "Newt Holdings")])
    r[us_cik.SEC_LOOKUP] = b"MIDA CORP:0000002001:\n"
    for t, c in CIK.items():
        former = ["Oldt Holdings"] if t == "NEWT" else None
        r[f"{us_cik.SUBMISSIONS}CIK{c:010d}.json"] = json.dumps(submissions(NAMES[t], former)).encode()
    zips = ftd_zips()
    r[us_prices.FTD_INDEX] = "".join(
        f'<a href="/files/data/fails-deliver-data/cnsfails{k}.zip">x</a>\n' for k in sorted(zips)
    ).encode()
    for k, b in zips.items():
        r[f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{k}.zip"] = b
    for k, px in RAW.items():
        if k != "SPY":
            r[us_prices.CHART_URL.format(symbol=k)] = chart_json(k, px, NAMES[k])
    for k, sh in SHARES.items():
        r[pit_us.COMPANYFACTS.format(cik=CIK[k])] = json.dumps(companyfacts(CIK[k], NAMES[k], sh)).encode()
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
    """PitNet serving the world; the MediaWiki API answers by ``action`` (revision list / revision text / pageprops)."""

    def __init__(self, settings, root: Path, routes: dict[str, bytes], revs: dict[int, tuple[pd.Timestamp, list[str]]]):
        super().__init__(settings, root)
        self.routes = routes
        self.revs = revs
        self.calls: list[str] = []

    def http_get(self, url, kind, params=None, headers=None, ok=(200,), tries=5):
        self.calls.append(url)
        if kind == "sec":
            self.sec_user_agent()
        if url == um.WIKI_API:
            p = params or {}
            if p.get("action") == "parse":
                ts, tick = self.revs[int(p["oldid"])]
                return FakeResponse(200, json.dumps({"parse": {"text": revision_html(tick)}}).encode())
            if p.get("prop") == "revisions":
                rv = [{"revid": k, "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ")} for k, (ts, _) in self.revs.items()]
                return FakeResponse(200, json.dumps({"query": {"pages": {"1": {"revisions": rv}}}}).encode())
            return FakeResponse(200, json.dumps({"query": {"pages": {"-1": {"title": "x"}}}}).encode())
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


def sp500_stub() -> PITUniverse:
    """The PIT S&P 500 tuning universe as the exclusion sees it: AAA (CIK 1001 = PROM, promoted 2020-03) and WRG
    (CIK 1005, another issuer than the mid cap WRG)."""
    dates = pd.DatetimeIndex(["2020-03-31", "2020-04-30", "2020-05-29", "2020-06-30"])
    rows = [
        (d, e, t, t, i + 1, 1e9)
        for d in dates
        for i, (e, t) in enumerate([("CIK0000001001", "AAA"), ("CIK0000001005", "WRG")])
    ]
    mem = pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS)
    meta = pd.DataFrame({"entity": ["CIK0000001001", "CIK0000001005"], "yahoo_symbol": ["AAA", "WRG"]})
    return PITUniverse(
        Market.US, 500, mem, {}, {}, {}, meta, coverage={"unmapped_spells": ["XYZ 2014-01-02~2015-03-02"]}
    )


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access in an offline test")

    monkeypatch.setattr(requests.Session, "request", boom)


def _small_world(mp: pytest.MonkeyPatch) -> None:
    mp.setattr(requests.Session, "request", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    mp.setattr(um, "MAX_CURRENT_ROWS", (3, 10))
    mp.setattr(um, "MIN_REVISION_ROWS", 3)
    mp.setattr(um, "REVISION_COMPLETE", (5, 12))
    mp.setattr(um, "COUNT_RANGE", (5, 8))
    mp.setattr(um, "COUNT_HARD_RANGE", (3, 12))
    mp.setattr(pit_us, "_today", lambda: TODAY)


def _build(tmp: Path, stale_early: bool = False, cache: DiskCache | None = None, net: FakeNet | None = None):
    settings = settings_for(tmp)
    net = net or FakeNet(settings, tmp / "pit", build_routes(), revisions(stale_early))
    cache = cache or DiskCache(tmp / "dc", enabled=True)
    full = um.midcap_build(
        settings, cache, net, START, None, today=TODAY, spy_loader=spy_frame, sp500_loader=sp500_stub
    )
    return full, net, settings, cache


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pit_us_midcap_world")
    mp = pytest.MonkeyPatch()
    _small_world(mp)
    full, net, settings, cache = _build(tmp)
    yield {"full": full, "net": net, "settings": settings, "cache": cache, "tmp": tmp}
    mp.undo()


# =============================================================================== parsing


def test_parse_wiki400_tables_links_rowspans_and_future_rows():
    cur, ch = um.parse_wiki400((FIX / "wiki400_small.html").read_bytes())
    assert list(cur["sym"]) == ["MIDA", "MIDB", "NEWT", "WRG", "NWC", "MIDC"]
    cik = cur.set_index("sym")["cik"]
    assert cik.isna().to_dict() == {
        "MIDA": False,
        "MIDB": True,
        "NEWT": True,
        "WRG": False,
        "NWC": False,
        "MIDC": False,
    }
    assert [int(cik[t]) for t in ("MIDA", "WRG", "NWC", "MIDC")] == [2001, 2006, 1006, 2008]  # numeric SEC links only
    assert cur.set_index("sym").loc["MIDA", "link"] == "Mida_Corp"
    assert len(ch) == 7 and ch["date"].is_monotonic_decreasing  # the unparseable 'Sometime 2018' row is skipped
    row = ch[ch["date"] == pd.Timestamp("2020-03-16")]
    assert len(row) == 2 and set(row["reason"]) == {"S&P 500 constituent Promo Corp moved to the S&P 500."}  # rowspan
    assert um._tickers("UA/UAA") == ["UA", "UAA"] and um._tickers("UAX/UA.B") == ["UAX", "UA-B"]
    assert ch.loc[ch["date"] == pd.Timestamp("2019-05-20"), "add_link"].iloc[0] == "Gone_Corp"  # footnote stripped
    assert ch.loc[ch["date"] == pd.Timestamp("2019-09-16"), "add_t"].isna().all()


def test_parse_wiki400_layout_change_raises():
    html = (FIX / "wiki400_small.html").read_text().replace("<th>SEC filings</th>", "<th>SEC filings</th><th>CIK</th>")
    with pytest.raises(us_membership.MembershipError):
        um.parse_wiki400(html)
    with pytest.raises(us_membership.MembershipError):
        um.parse_wiki400("<html><body><p>no tables</p></body></html>")


def _table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{body}</table>"


def test_parse_revision_tickers_old_column_names(monkeypatch):
    monkeypatch.setattr(um, "MIN_REVISION_ROWS", 2)
    old = _table(["Ticker Symbol", "Company"], [["JW.A", "x"], ["ABC", "y"]])
    assert um.parse_revision_tickers(old) == {"JW-A", "ABC"}
    new = _table(["Security", "Ticker symbol"], [["x", "UA/UAA"], ["y", "Q"]])
    assert um.parse_revision_tickers(new) == {"UA", "UAA", "Q"}
    monkeypatch.setattr(um, "MIN_REVISION_ROWS", 5)
    assert um.parse_revision_tickers(new) is None


def test_classify_reason():
    assert um.classify_reason("Acquisition.") == "acquired"
    assert um.classify_reason("Nasdaq delisted the company after its chapter 11 filing.") == "failed"
    assert um.classify_reason("Market capitalization change.") == "demoted"
    assert um.classify_reason(None) == "unknown"


# =============================================================================== reconstruction


def _resolver(table: dict[tuple[str, str], int]):
    def resolve(ticker, name, link, d, side):
        for (t, when), c in table.items():
            if t == ticker and (when == "*" or pd.Timestamp(when) == d):
                return c, "test"
        return None, "unmapped"

    return resolve


def _changes(rows):
    df = pd.DataFrame(rows, columns=["date", "add_t", "add_n", "rem_t", "rem_n", "reason"])
    df["date"] = pd.to_datetime(df["date"])
    df["add_link"] = None
    df["rem_link"] = None
    return df[um.CHANGE_COLUMNS].sort_values("date", ascending=False, ignore_index=True)


def test_reconstruct_renames_reuse_missing_rows_and_share_classes():
    cur = pd.DataFrame(
        {
            "sym": ["NEWT", "COHR", "KEEP", "UA", "UAA"],
            "security": ["Newt Holdings", "Two-Six", "Keep Co", "Under", "Under"],
            "link": None,
            "cik": pd.array([4, 26, 5, 7, 7], dtype="Int64"),
        }
    )
    ch = _changes(
        [
            ("2022-07-05", None, None, "COHR", "Coherent Old", "Acquisition."),  # reused ticker (another CIK)
            ("2021-01-04", "LOST", "Lost Co", "KEEP", "Keep Co", "Market cap"),  # LOST's removal missing; KEEP re-added
            ("2020-05-01", "OLDT", "Oldt Holdings", None, None, "x"),  # later renamed OLDT -> NEWT
            ("2019-09-26", "IIVI", "II-VI", None, None, "x"),  # II-VI took the COHR ticker later
            ("2019-01-02", "UA/UAA", "Under", "ZZ", "Zed", "x"),
            ("2031-01-01", "FUT", "Future", None, None, "announced"),
        ]
    )
    resolve = _resolver(
        {
            ("OLDT", "*"): 4,
            ("COHR", "2022-07-05"): 21,
            ("IIVI", "*"): 26,
            ("KEEP", "*"): 5,
            ("LOST", "*"): 9,
            ("ZZ", "*"): 8,
        }
    )
    sp, anomalies = um.reconstruct_spells(cur, ch, pd.Timestamp("2026-09-25"), resolve)
    kinds = sorted(a["kind"] for a in anomalies)
    assert kinds == ["missing_addition", "missing_removal", "rename_cik", "rename_cik"]

    def at(d: str) -> set[str]:
        return set(um.members_at(sp, pd.Timestamp(d))["ticker"])

    newt = sp[sp["ticker"] == "NEWT"].iloc[0]
    assert newt["start"] == pd.Timestamp("2020-05-01") and newt["alias"] == "OLDT" and newt["cik"] == 4
    cohr = sp[sp["ticker"] == "COHR"].set_index("cik")
    assert cohr.loc[21, "end"] == pd.Timestamp("2022-07-05") and cohr.loc[21, "start"] == um.START_UNKNOWN
    assert cohr.loc[26, "start"] == pd.Timestamp("2019-09-26") and cohr.loc[26, "alias"] == "IIVI"
    lost = sp[sp["ticker"] == "LOST"].iloc[0]
    assert lost["end_unknown"] and lost["start"] == pd.Timestamp("2021-01-04") and pd.isna(lost["end"])
    keep = sp[sp["ticker"] == "KEEP"].sort_values("start")
    assert len(keep) == 2 and keep.iloc[0]["end"] == pd.Timestamp("2021-01-04") and keep.iloc[1]["start_unknown"]
    assert "FUT" not in set(sp["ticker"])  # announced, not effective yet
    assert {"UA", "UAA"} <= at("2019-01-02") and "ZZ" in at("2019-01-01") and "ZZ" not in at("2019-01-02")
    assert at("2025-01-02") >= {"NEWT", "COHR", "UA", "UAA"}


def _sp(rows):
    base = dict(alias=None, end=pd.NaT, end_unknown=False, start_unknown=False, cik=None)
    return pd.DataFrame([{**base, **r} for r in rows])


def test_bridge_unknowns_with_revisions():
    sp = _sp(
        [
            dict(ticker="LOST", cik=9, start=pd.Timestamp("2021-01-04"), end_unknown=True),
            dict(ticker="NONE", start=pd.Timestamp("2021-01-04"), end_unknown=True),
            dict(ticker="KEEP", cik=5, start=pd.Timestamp("2021-01-04"), start_unknown=True),
        ]
    )
    rev = {
        pd.Timestamp("2021-01-29"): {"CIK5"},
        pd.Timestamp("2021-02-26"): {"CIK9"},
        pd.Timestamp("2021-03-31"): {"CIK9", "CIK5"},
        pd.Timestamp("2021-04-30"): {"CIK5"},
    }
    out, stats = um.bridge_unknowns(sp, rev)
    o = out.set_index("ticker")
    assert o.loc["LOST", "end"] == pd.Timestamp("2021-04-30")  # listed Feb-Mar (editor lag in Jan), gone in Apr
    assert o.loc["NONE", "end"] == pd.Timestamp("2021-01-29")  # no evidence: covers no month-end
    assert o.loc["KEEP", "start"] == pd.Timestamp("2021-01-29")
    assert stats == {"removal_bridged": 1, "removal_unresolved": 1, "addition_bridged": 1}


def test_trim_open_starts_needs_a_clear_absence():
    d = list(pd.date_range("2021-01-31", periods=8, freq="ME"))
    sp = _sp(
        [
            dict(ticker="SPIN", cik=1, start=um.START_UNKNOWN),  # a spin-off the change table never added
            dict(ticker="OLD", cik=2, start=um.START_UNKNOWN),  # listed throughout
            dict(ticker="FLICK", cik=3, start=um.START_UNKNOWN),  # one missing month: an editing artefact
        ]
    )
    rev = {x: {"CIK2", "CIK3"} | ({"CIK1"} if i >= 4 else set()) for i, x in enumerate(d)}
    rev[d[2]] = {"CIK2"}
    out, stats = um.trim_open_starts(sp, rev)
    o = out.set_index("ticker")["start"]
    assert o["SPIN"] == d[4] and o["OLD"] == um.START_UNKNOWN and o["FLICK"] == um.START_UNKNOWN
    assert stats == {"trimmed": 1, "kept": 2}


def test_revision_freshness_detects_stale_pages():
    events = [(pd.Timestamp("2020-01-10"), "CIK1", "add"), (pd.Timestamp("2020-01-10"), "CIK2", "remove")]
    rev = {
        pd.Timestamp("2020-01-31"): {"CIK1"},  # both reflected
        pd.Timestamp("2020-02-28"): {"CIK2"},  # neither: the page was not edited
        pd.Timestamp("2020-12-31"): {"CIK2"},  # nothing dated 7-90 days before: NaN
    }
    f = um.revision_freshness(events, rev)
    assert f[pd.Timestamp("2020-01-31")] == 1.0 and f[pd.Timestamp("2020-02-28")] == 0.0
    assert np.isnan(f[pd.Timestamp("2020-12-31")])
    assert um.usable_revisions(
        {pd.Timestamp("2020-01-31"): {str(i) for i in range(400)}, pd.Timestamp("2013-08-30"): {"x"}}
    ) == {pd.Timestamp("2020-01-31"): {str(i) for i in range(400)}}


def test_revision_supplements_add_names_the_change_table_missed():
    d = list(pd.date_range("2019-01-31", periods=6, freq="ME"))
    snaps = {x: {"KNWN": "Known Co", **({"MISS": "Missed Inc"} if 1 <= i <= 3 else {})} for i, x in enumerate(d)}
    snaps[d[5]]["ONCE"] = "Once Co"
    sp = _sp([dict(ticker="KNWN", cik=1, start=um.START_UNKNOWN, name="Known Co", alias_name=None)])
    pairs = pd.DataFrame(
        [("MISS", 77, "Missed Inc", d[0], d[4]), ("KNWN", 1, "Known Co", d[0], d[5]), ("ONCE", 78, "Once", d[0], d[5])],
        columns=["ticker", "cik", "title", "first_seen", "last_seen"],
    )
    out, stats = um.revision_supplements(sp.assign(supplement=False), snaps, um.RevisionKeys(sp, pairs, {}))
    assert list(out["ticker"]) == ["MISS"] and int(out["cik"].iloc[0]) == 77  # ONCE: a single listing is ignored
    assert out["start"].iloc[0] == d[1] and out["end"].iloc[0] == d[4] and bool(out["supplement"].iloc[0])
    assert stats == {"spells": 1, "issuers": 1}


def test_pit_from_rule_tolerates_isolated_dips_not_drift():
    d = pd.date_range("2020-01-31", periods=24, freq="ME")
    a = np.r_[np.full(8, 0.80), np.full(16, 0.99)]
    a[15] = 0.90  # one late-edited month-end
    agree = pd.DataFrame({"date": d, "agreement_lag": a})
    assert um.pit_from_date(agree) == d[8]
    a2 = a.copy()
    a2[20] = 0.80  # below the floor: the rule restarts after it
    assert um.pit_from_date(agree.assign(agreement_lag=a2)) == d[21]
    assert um.pit_from_date(agree.assign(agreement_lag=np.full(24, 0.5))) is None
    stale = np.r_[np.full(8, False), np.full(16, True)]  # stale revisions neither pass nor fail a month-end
    assert um.pit_from_date(agree.assign(agreement_lag=a2, fresh=~(np.arange(24) == 20))) == d[8]
    assert um.pit_from_date(agree.assign(fresh=stale)) == d[8]


def test_snapshot_agreement_is_keyed_by_issuer():
    sp = _sp(
        [
            dict(ticker="NEWT", cik=4, start=pd.Timestamp("2019-01-01")),
            dict(ticker="A", cik=1, start=pd.Timestamp("2019-01-01")),
        ]
    )
    snaps = {pd.Timestamp("2019-03-29"): {"OLDT": "Oldt", "A": "A Co"}, pd.Timestamp("2019-04-30"): {"NEWT": "Newt"}}
    keys = um.RevisionKeys(
        sp.assign(alias=["OLDT", None], name=["Newt", "A Co"]),
        pd.DataFrame(columns=["ticker", "cik", "title", "first_seen", "last_seen"]),
        {},
    )
    rk = um.revision_keys(snaps, keys)
    agree = um.snapshot_agreement(sp, rk, lags=1).set_index("date")
    assert agree.loc["2019-03-29", "agreement"] == 1.0  # OLDT (then) == NEWT (now): same issuer
    assert agree.loc["2019-04-30", "agreement"] == 0.5 and agree.loc["2019-04-30", "agreement_lag"] == 0.5


# =============================================================================== exclusion


def test_sp500_exclusion_sets():
    x = um.sp500_exclusion(sp500_stub())
    assert x["entities"] == {"CIK0000001001", "CIK0000001005"} and {1001, 1005} <= x["ciks"]
    assert {"AAA", "WRG", "XYZ"} <= x["tickers"]  # carried, Yahoo and unmapped-spell tickers
    assert x["first"]["CIK0000001001"] == pd.Timestamp("2020-03-31")


def test_exclude_keys_accepts_tickers_and_entities():
    ents, ticks = um._exclude_keys(["CIK0000002002", "brk.b", " ", "005930#1", "cik2001"])
    assert ents == {"CIK0000002002", "CIK0000002001"} and ticks == {"BRK-B", "005930"}


# =============================================================================== full offline build


def _tickers_at(uni: PITUniverse):
    def at(d: str) -> dict[str, str]:
        m = uni.membership[uni.membership["date"] == pd.Timestamp(d)]
        return dict(zip(m["entity"], m["ticker"], strict=False))

    return at


def test_world_contract_membership_and_rename(world):
    uni = um.to_universe(world["full"])
    assert isinstance(uni, PITUniverse) and uni.market is Market.US and uni.n == um.INDEX_SIZE
    assert list(uni.membership.columns) == MEMBERSHIP_COLUMNS
    ents = set(uni.membership["entity"])
    assert ents == {f"CIK{CIK[t]:010d}" for t in ("MIDA", "MIDB", "OLDM", "NEWT", "GONE", "MIDC", "LEFT")}
    at = _tickers_at(uni)
    newt = f"CIK{CIK['NEWT']:010d}"
    assert at("2019-01-31").get(newt) is None and at("2019-02-28")[newt] == "OLDT" and at("2019-08-30")[newt] == "NEWT"
    assert f"CIK{CIK['LEFT']:010d}" in at("2019-02-28") and f"CIK{CIK['LEFT']:010d}" not in at("2019-03-29")
    gone = f"CIK{CIK['GONE']:010d}"
    assert gone in at("2019-10-31") and gone not in at("2019-11-29")  # bridged by the page revisions
    assert f"CIK{CIK['OLDM']:010d}" in at("2019-08-30") and f"CIK{CIK['OLDM']:010d}" not in at("2019-09-30")
    # every entity's benchmark is SPY (the frozen model's excess-return basis)
    assert set(uni.benchmarks) == {"SPY"} and set(uni.entity_benchmark.values()) == {"SPY"}
    for e, px in uni.prices.items():
        assert list(px.columns) == [*PRICE_COLUMNS, "value"] and "adjusted" in px.attrs, e
    assert uni.prices[f"CIK{CIK['OLDM']:010d}"].index[-1] == LAST["OLDM"]  # ends at the last traded session
    assert uni.prices[f"CIK{CIK['MIDC']:010d}"].attrs["adjusted"] is False  # no FTD prints: unvalidated
    assert uni.prices[f"CIK{CIK['MIDA']:010d}"].attrs["adjusted"] is True
    cov = uni.coverage
    assert cov["membership_checks"]["rename_dates"] == {"ftd_last_print": 1}
    assert cov["membership_checks"]["bridged_by_revisions"] == {"removal_bridged": 1, "open_start_kept": 6}
    assert cov["member_months"] == len(uni.membership) and cov["member_months_missing"] == 0
    assert all(k not in uni.coverage for k in ("ret", "returns"))
    assert world["full"].month_src["ret"].isna().all()  # hygiene: no member-month returns are kept


def test_world_exclusion_counts_and_reasons(world):
    ex = world["full"].coverage["exclusion_sp500"]
    assert ex["excluded_by_reason"] == {"sp500_entity": 1, "sp500_ticker": 1, "sp500_current": 1}
    assert ex["excluded_sp500_entity_promoted_later"] == 1 and ex["excluded_ticker_only"] == ["WRG"]
    assert ex["retained_issuers"] == 7
    uni = um.to_universe(world["full"], exclude=["MIDB", "CIK0000002001", "NOPE"])
    assert uni.coverage["excluded"]["caller_entities_dropped"] == 2
    assert uni.coverage["excluded"]["by"] == {"ticker": 1, "entity": 1}
    assert not {f"CIK{CIK['MIDB']:010d}", "CIK0000002001"} & set(uni.membership["entity"])
    ranks = uni.membership.groupby("date")["rank"].apply(lambda s: list(s) == list(range(1, len(s) + 1)))
    assert ranks.all()  # re-ranked 1..n after the caller's exclusion


def test_world_rank_by_pit_market_cap(world):
    uni = um.to_universe(world["full"])
    m = uni.membership[uni.membership["date"] == pd.Timestamp("2020-05-29")].set_index("entity")
    assert world["full"].coverage["rank"]["dates_by_basis"].get("mcap", 0) >= 1
    mida, midb = f"CIK{CIK['MIDA']:010d}", f"CIK{CIK['MIDB']:010d}"
    px = {e: uni.prices[e]["close"].asof(pd.Timestamp("2020-05-29")) for e in (mida, midb)}
    assert m.loc[mida, "rank_value"] > 0 and m.loc[midb, "rank_value"] > 0
    assert (m.loc[mida, "rank"] < m.loc[midb, "rank"]) == (px[mida] * SHARES["MIDA"] > px[midb] * SHARES["MIDB"])


def test_world_pit_validated_no_survivorship_flag(world):
    cov = world["full"].coverage
    assert cov["pit_from"] is not None and pd.Timestamp(cov["pit_from"]) <= pd.Timestamp("2019-01-31")
    assert cov["survivorship_biased"] is False and cov["biased_dates"] == 0
    assert cov["membership_checks"]["agreement_min_since_pit_from"] == 1.0


def test_stale_early_revisions_flag_survivorship_bias(tmp_path, monkeypatch):
    _small_world(monkeypatch)
    full, *_ = _build(tmp_path, stale_early=True)
    cov = full.coverage
    assert cov["pit_from"] == "2019-03-29"
    assert cov["survivorship_biased"] is True and cov["biased_dates"] == 2
    assert cov["biased_range"] == ["2019-01-31", "2019-02-28"]
    uni = um.to_universe(full)
    assert any("SURVIVORSHIP-BIASED" in n for n in uni.notes)


def test_world_build_is_cached_and_key_tracks_inputs(world):
    net = world["net"]
    calls = len(net.calls)
    again, *_ = _build(world["tmp"], cache=world["cache"], net=net)
    assert len(net.calls) == calls and len(again.members) == len(world["full"].members)
    s = world["settings"]
    k0 = um.build_key(s, START, None, TODAY)
    assert k0 == um.build_key(s, START, None, TODAY)
    assert k0 != um.build_key(s, START, None, TODAY + pd.Timedelta(1, "D"))
    assert k0 != um.build_key(s, START, pd.Timestamp("2020-03-31"), TODAY)
    assert k0 != um.build_key(
        s.model_copy(update={"backtest": s.backtest.model_copy(update={"us_start": "2014-01-01"})}), START, None, TODAY
    )
    assert (world["tmp"] / "pit" / "build_v1" / f"midcap_{k0}.pkl").exists()
    assert not (world["tmp"] / "pit" / "build_v1" / "known_entities.json").exists()  # S&P 500 state untouched


def test_load_pit_xsh_universe_dispatch(monkeypatch, tmp_path):
    seen = {}

    def fake(settings, cache, start, end=None, exclude=()):
        seen.update(start=start, end=end, exclude=tuple(exclude))
        return "uni"

    monkeypatch.setattr(um, "load_us_midcap_universe", fake)
    s = settings_for(tmp_path)
    out = load_pit_xsh_universe(Market.US, s, DiskCache(tmp_path, enabled=False), date(2012, 1, 1), exclude=["AAPL"])
    assert out == "uni" and seen == {"start": date(2012, 1, 1), "end": None, "exclude": ("AAPL",)}
    with pytest.raises(NotImplementedError):
        load_pit_xsh_universe(Market.KR, s, DiskCache(tmp_path, enabled=False), date(2016, 1, 1))


def test_fixture_files_are_small():
    assert all(p.stat().st_size < 20_000 for p in FIX.iterdir())


def test_include_sp500_variant_skips_the_exclusion_and_caches_separately(monkeypatch, tmp_path):
    """The all-members variant (docs/MODELING.md 19) never calls the S&P 500 exclusion and uses its own cache name."""
    from quantcio.data.pit import us_midcap as um

    calls = []

    class FakeBuilder:
        def __init__(self, *args):
            calls.append(args[-1])  # include_sp500

        def run(self):
            return um.pit_us.FullBuild.__new__(um.pit_us.FullBuild)

    written = []

    class FakeNet:
        def read_pickle(self, src, name):
            return None

        def write_pickle(self, src, name, obj):
            written.append(name)

        def path(self, *a):
            return tmp_path

    monkeypatch.setattr(um, "_MidcapBuilder", FakeBuilder)
    monkeypatch.setattr(um, "build_key", lambda *a, **k: "K")
    import pandas as pd

    from quantcio.config import load_settings

    s = load_settings()
    cache = type("C", (), {"enabled": True})()
    um.midcap_build(
        s,
        cache,
        FakeNet(),
        pd.Timestamp("2017-01-01"),
        None,
        today=pd.Timestamp("2026-10-05"),
        sp500_loader=lambda: None,
        include_sp500=True,
    )
    um.midcap_build(
        s,
        cache,
        FakeNet(),
        pd.Timestamp("2017-01-01"),
        None,
        today=pd.Timestamp("2026-10-05"),
        sp500_loader=lambda: None,
    )
    assert calls == [True, False]
    assert written == ["midcap_all_K.pkl", "midcap_K.pkl"]


def test_summary_notes_describe_the_all_members_variant():
    membership = pd.DataFrame({"date": pd.to_datetime(["2020-01-31", "2020-02-28"])})
    cov = {
        "member_months": 800,
        "member_months_with_prices": 780,
        "member_months_ftd_only": 15,
        "member_months_missing": 5,
        "members_per_date": {"min": 400, "max": 400},
        "pit_from": "2017-07-31",
        "excluded": {"caller_entities_dropped": 0},
    }
    xsh = um._summary_notes(membership, cov)[0]
    every = um._summary_notes(membership, {**cov, "sp500_exclusion": "skipped (include_sp500)"})[0]
    assert "S&P 500 편입 이력(백테스트 시작 이후) 없는 종목" in xsh
    assert "전체 구성원(S&P 500 이력 종목 포함)" in every and "없는 종목" not in every
