"""Tests for quantcio.data.alt.us_sec (SEC submissions forms / 8-K items + Form 3/4/5 insider data sets).

Offline only: submissions JSON blocks and quarterly Form 345 zips are built in code and served by a fake ``PitNet``
(the same pattern as tests/test_pit_us.py); any real network access fails the test.
"""

from __future__ import annotations

import io
import json
import os
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests

from quantcio.config import load_settings
from quantcio.data.alt import EVENT_COLUMNS
from quantcio.data.alt import us_sec as us
from quantcio.data.pit.us_cik import SUBMISSIONS, Submissions
from quantcio.data.pit.us_net import PitNet

UA = "QuantCIO Research research@example.com"
T = pd.Timestamp


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access in an offline test")

    monkeypatch.setattr(requests.Session, "request", boom)


def settings_for(tmp_path: Path):
    return load_settings(data={"sec_user_agent": UA, "cache_dir": str(tmp_path / "cache")})


class FakeResponse:
    def __init__(self, status: int, body: bytes):
        self.status_code = status
        self.content = body
        self.headers = {"content-type": "application/octet-stream"}

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")

    def json(self):
        return json.loads(self.content)


class FakeNet(PitNet):
    """Serves ``routes``; any other URL answers 404."""

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
            resp = FakeResponse(404, b"{}")
            if 404 in ok:
                return resp
            raise requests.HTTPError(f"404 {url}", response=resp)
        return FakeResponse(200, body)


# =============================================================================== submissions


def block(rows: list[tuple]) -> dict:
    """[(form, filingDate, acceptanceDateTime, items, fileNumber)] -> a submissions block (column lists)."""
    out: dict[str, list] = {k: [] for k in ("accessionNumber", "form", "filingDate", "acceptanceDateTime")}
    out.update(items=[], fileNumber=[])
    for i, (form, fd, acc, items, fn) in enumerate(rows):
        out["accessionNumber"].append(f"0000000000-{fd[2:4]}-{abs(hash((form, fd, acc, i))) % 10**6:06d}")
        out["form"].append(form)
        out["filingDate"].append(fd)
        out["acceptanceDateTime"].append(acc)
        out["items"].append(items)
        out["fileNumber"].append(fn)
    return out


def test_kinds_are_predeclared_and_unique():
    assert len(set(us.KINDS)) == len(us.KINDS)
    assert {"form4", "8k", "8k_2.02", "insider_buy_value", "insider_sell_n"} <= set(us.KINDS)
    assert len(us.ITEM_KINDS) == 16


@pytest.mark.parametrize(
    "form,kind",
    [
        ("4", "form4"),
        ("4/A", "form4"),
        ("144/A", "form144"),
        ("SC 13D/A", "sc13d"),
        ("SCHEDULE 13D", "sc13d"),
        ("SCHEDULE 13G/A", "sc13g"),
        ("SC 13G", "sc13g"),
        ("NT 10-Q/A", "nt_late"),
        ("NT 10-K", "nt_late"),
        ("424B5", "prospectus_424b"),
        ("S-3ASR", "s3"),
        ("S-3/A", "s3"),
        ("8-K/A", "8k"),
        ("10-K", None),
        ("3", None),
        ("S-3MEF", None),
        ("8-K12B", None),
    ],
)
def test_form_kind(form, kind):
    assert us.form_kind(form) == kind


def test_published_is_new_york_acceptance_date_with_filing_date_fallback():
    got = us.published_dates(
        ["2024-03-01T22:30:00.000Z", "2024-03-02T03:00:00.000Z", "", None, "2024-07-01T03:30:00.000Z"],
        ["2024-03-04", "2024-03-01", "2024-02-15", "2024-02-16", "2024-07-01"],
    )
    assert list(got) == [T("2024-03-01"), T("2024-03-01"), T("2024-02-15"), T("2024-02-16"), T("2024-06-30")]


def test_submission_blocks_split_8k_items_and_filter_roles_and_window():
    recent = block(
        [
            ("8-K", "2024-03-04", "2024-03-01T22:30:00.000Z", "2.02,9.01", "001-36743"),  # 17:30 EST
            ("4", "2024-03-01", "2024-03-02T03:00:00.000Z", "", ""),  # 22:00 EST
            ("8-K/A", "2024-02-15", "", "5.02, 7.01,5.02", "001-36743"),  # no acceptance -> filingDate
            ("SC 13G/A", "2024-02-14", "2024-02-14T15:00:00.000Z", "", ""),  # filed BY this CIK: skipped
            ("SC 13G", "2024-02-13", "2024-02-13T15:00:00.000Z", "", "005-33632"),
            ("SCHEDULE 13D", "2025-03-03", "2025-03-03T15:00:00.000Z", "", "005-33632"),
            ("144", "2024-02-12", "2024-02-12T15:00:00.000Z", "", ""),  # seller of another issuer's stock
            ("144", "2024-02-12", "2024-02-12T16:00:00.000Z", "", "001-36743"),
            ("424B2", "2024-02-09", "2024-02-09T21:00:00.000Z", "", "333-1"),
            ("S-3ASR", "2024-02-08", "2024-02-08T21:00:00.000Z", "", "333-2"),
            ("NT 10-K", "2024-02-07", "2024-02-07T21:00:00.000Z", "", "001-36743"),
            ("10-K", "2024-02-06", "2024-02-06T21:00:00.000Z", "", "001-36743"),
            ("8-K", "2023-12-28", "2023-12-28T21:00:00.000Z", "8.01", "001-36743"),  # before start
        ]
    )
    ev, st = us.parse_submission_blocks([recent], "CIK0000320193", T("2024-01-01"), T("2024-12-31"))
    assert list(ev.columns) == EVENT_COLUMNS
    assert (ev["entity"] == "CIK0000320193").all() and (ev["source"] == "sec_forms").all()
    assert (ev["value"] == 1.0).all()
    got = sorted(zip(ev["published"].dt.strftime("%Y-%m-%d"), ev["kind"], strict=True))
    assert got == sorted(
        [
            ("2024-03-01", "8k"),
            ("2024-03-01", "8k_2.02"),  # 9.01 is not a declared item
            ("2024-03-01", "form4"),
            ("2024-02-15", "8k"),
            ("2024-02-15", "8k_5.02"),  # listed twice in one filing -> one event
            ("2024-02-15", "8k_7.01"),
            ("2024-02-13", "sc13g"),
            ("2024-02-12", "form144"),
            ("2024-02-09", "prospectus_424b"),
            ("2024-02-08", "s3"),
            ("2024-02-07", "nt_late"),
        ]
    )
    assert st["filer_role_skipped"] == 2
    assert st["acceptance_fallback"] == 1
    assert set(ev["kind"]) <= set(us.KINDS)


def _subs_routes(cik: int) -> dict[str, bytes]:
    recent = block([("4", f"20{y}-06-01", "", "", "") for y in (24, 22, 20)])
    page1 = block([("4", f"20{y}-06-01", "", "", "") for y in (19, 17, 15)])
    page2 = block([("4", f"20{y}-06-01", "", "", "") for y in (14, 12, 10)])
    main = {
        "cik": str(cik),
        "name": "TEST CO",
        "filings": {
            "recent": recent,
            "files": [
                {"name": f"CIK{cik:010d}-submissions-001.json", "filingFrom": "2015-06-01", "filingTo": "2019-06-01"},
                {"name": f"CIK{cik:010d}-submissions-002.json", "filingFrom": "2010-06-01", "filingTo": "2014-06-01"},
            ],
        },
    }
    return {
        f"{SUBMISSIONS}CIK{cik:010d}.json": json.dumps(main).encode(),
        f"{SUBMISSIONS}CIK{cik:010d}-submissions-001.json": json.dumps(page1).encode(),
        f"{SUBMISSIONS}CIK{cik:010d}-submissions-002.json": json.dumps(page2).encode(),
    }


def test_older_pages_are_read_only_until_start_is_covered(tmp_path):
    net = FakeNet(settings_for(tmp_path), tmp_path / "pit", _subs_routes(1234))
    blocks = us.submission_blocks(Submissions(net), 1234, T("2021-01-01"))
    assert len(blocks) == 1 and len(net.calls) == 1  # the recent block reaches back to 2020
    blocks = us.submission_blocks(Submissions(net), 1234, T("2016-01-01"))
    assert len(blocks) == 2 and net.calls[-1].endswith("-submissions-001.json")
    n = len(net.calls)  # main JSON and page 1 are cached on disk now: only page 2 is fetched
    blocks = us.submission_blocks(Submissions(net), 1234, T("2011-01-01"))
    assert len(blocks) == 3 and net.calls[n:] == [f"{SUBMISSIONS}CIK0000001234-submissions-002.json"]
    ev, _ = us.parse_submission_blocks(blocks, "CIK0000001234", T("2011-01-01"), T("2024-12-31"))
    assert ev["published"].dt.year.tolist() == [2012, 2014, 2015, 2017, 2019, 2020, 2022, 2024]


def test_a_recut_older_page_is_fetched_again(tmp_path):
    """EDGAR re-cuts the older pages as the recent block rolls over (docs/MODELING.md 84.5): a cached page that no
    longer matches its ``files`` entry left the filings between it and the new recent block out."""
    cik = 1234
    net = FakeNet(settings_for(tmp_path), tmp_path / "pit", _subs_routes(cik))
    us.submission_blocks(Submissions(net), cik, T("2011-01-01"))  # main JSON and both pages cached
    files = [
        ("001", 3, "2019-06-01", "2022-06-03", (22, 20, 19)),
        ("002", 5, "2010-06-01", "2017-06-01", (17, 15, 14, 12, 10)),
    ]
    main = {
        "cik": str(cik),
        "name": "TEST CO",
        "filings": {
            "recent": block([("4", f"20{y}-06-01", "", "", "") for y in (26, 25, 24)]),
            "files": [
                {"name": f"CIK{cik:010d}-submissions-{n}.json", "filingCount": c, "filingFrom": a, "filingTo": b}
                for n, c, a, b, _ in files
            ],
        },
    }
    net.routes = {f"{SUBMISSIONS}CIK{cik:010d}.json": json.dumps(main).encode()} | {
        f"{SUBMISSIONS}CIK{cik:010d}-submissions-{n}.json": json.dumps(
            block([("4", f"20{y}-06-01", "", "", "") for y in ys])
        ).encode()
        for n, *_, ys in files
    }
    net.path("sec", f"subs/{cik}.json.gz").unlink()  # the daily refresh of the main JSON
    blocks = us.submission_blocks(Submissions(net), cik, T("2011-01-01"))
    ev, _ = us.parse_submission_blocks(blocks, "CIK0000001234", T("2009-01-01"), T("2026-12-31"))
    assert ev["published"].dt.year.tolist() == [2010, 2012, 2014, 2015, 2017, 2019, 2020, 2022, 2024, 2025, 2026]
    n = len(net.calls)
    us.submission_blocks(Submissions(net), cik, T("2011-01-01"))  # the pages on disk match their entries now
    assert len(net.calls) == n


def test_page_matches_its_files_entry():
    from quantcio.data.pit.us_cik import page_matches

    page = {"accessionNumber": ["a", "b"], "filingDate": ["2017-06-12", "2024-06-21"]}
    meta = {"filingCount": 2, "filingFrom": "2017-06-12", "filingTo": "2024-06-23"}  # filingTo: the range boundary
    assert page_matches(page, meta) and page_matches(page, {})
    assert not page_matches(page, {**meta, "filingCount": 3})
    assert not page_matches(page, {**meta, "filingFrom": "2017-05-10"})
    assert not page_matches(page, {**meta, "filingTo": "2024-06-12"})
    assert page_matches({}, {"filingCount": 0}) and not page_matches({}, meta)
    # EDGAR's filingTo is often one day before the page's last filing (docs/MODELING.md 84.6 check 3)
    assert page_matches(page, {**meta, "filingTo": "2024-06-20"})
    assert not page_matches(page, {**meta, "filingTo": "2024-06-19"})


def _cut_routes(cik: int, recent: tuple, pages: list[tuple[str, tuple, tuple | None]]) -> dict[str, bytes]:
    """One EDGAR cut: recent years + older pages (name, years listed in the ``files`` entry, years on the page or
    None = the same)."""

    def b(ys):
        return block([("4", f"20{y:02d}-06-01", "", "", "") for y in ys])

    files = [
        {
            "name": f"CIK{cik:010d}-submissions-{n}.json",
            "filingCount": len(ys),
            "filingFrom": f"20{min(ys):02d}-06-01",
            "filingTo": f"20{max(ys):02d}-06-01",
        }
        for n, ys, _ in pages
    ]
    main = {"cik": str(cik), "name": "TEST CO", "filings": {"recent": b(recent), "files": files}}
    out = {f"{SUBMISSIONS}CIK{cik:010d}.json": json.dumps(main).encode()}
    for n, ys, on_page in pages:
        out[f"{SUBMISSIONS}CIK{cik:010d}-submissions-{n}.json"] = json.dumps(b(on_page or ys)).encode()
    return out


def test_pages_of_a_newer_cut_than_the_main_json_refetch_it(tmp_path):
    """docs/MODELING.md 92: the cached main JSON can be a day old; a page fetched now of EDGAR's newer cut next to a
    cached page of the old cut dropped the filings between them. The main JSON is fetched again instead."""
    cik = 1234
    old = _cut_routes(cik, (24, 22, 20), [("001", (19, 17, 15), None), ("002", (14, 12, 10), None)])
    net = FakeNet(settings_for(tmp_path), tmp_path / "pit", old)
    us.submission_blocks(Submissions(net), cik, T("2011-01-01"))  # main JSON and both pages cached
    net.path("sec", f"subs/CIK{cik:010d}-submissions-001.json.gz").unlink()  # page 1 must be fetched now
    net.routes = _cut_routes(cik, (26, 25, 24), [("001", (22, 20, 19), None), ("002", (17, 15, 14, 12, 10), None)])
    subs = Submissions(net)
    blocks = us.submission_blocks(subs, cik, T("2011-01-01"))
    ev, _ = us.parse_submission_blocks(blocks, "CIK0000001234", T("2009-01-01"), T("2026-12-31"))
    assert ev["published"].dt.year.tolist() == [2010, 2012, 2014, 2015, 2017, 2019, 2020, 2022, 2024, 2025, 2026]
    assert subs.refreshed == {cik} and not subs.mismatched


def test_pages_that_never_match_their_entry_leave_the_cik_uncovered(tmp_path):
    settings = settings_for(tmp_path)
    routes = _cut_routes(1234, (24, 22, 20), [("001", (19, 17, 15), (19, 17)), ("002", (14, 12, 10), None)])
    net = FakeNet(settings, tmp_path / "pit", routes)
    alt = us.load_us_sec_alt(settings, [1234], "2011-01-01", "2024-12-31", net=net, form345=False)
    assert alt.covered["sec_forms"] == []
    cov = alt.coverage["sec_forms"]
    assert cov["ciks_recut_unresolved"] == 1 and cov["main_refetched"] == 1
    assert any("re-cut them meanwhile" in n for n in alt.notes)


# ------------------------------------------------------------------------------- sec_forms vintage (92)


def _ev(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    from quantcio.data.alt import normalise_events

    df = pd.DataFrame(rows, columns=["entity", "published", "kind"]).assign(value=1.0, source=us.SOURCE_FORMS)
    return normalise_events(df)


def _rows(ev: pd.DataFrame) -> list[tuple[str, str, str]]:
    return [(e, str(p.date()), k) for e, p, k in zip(ev["entity"], ev["published"], ev["kind"], strict=True)]


def test_form_events_are_frozen_per_entity_as_first_seen(tmp_path):
    from quantcio.data.vintage import VintageStore

    root = tmp_path / "vintage"
    store = VintageStore(root, today=date(2026, 11, 6))  # cutoff 2026-10-31
    start = T("2025-01-01")
    fresh = _ev(
        [
            ("A", "2026-03-02", "form4"),
            ("A", "2026-03-02", "form4"),  # two filings on one day stay two
            ("A", "2026-11-02", "8k"),
            ("B", "2026-05-01", "form4"),
        ]
    )
    out, info, notes = us.freeze_form_events(store, fresh, start=start, asked=["A", "B", "C"], eligible=["A", "C"])
    assert _rows(out) == _rows(fresh)  # the first read is returned as it is
    assert info["created"] and info["new_keys"] == 2 and info["not_frozen"] == 1  # B: not read completely
    assert info["rows_added"] == 2  # A's two Forms 4 up to the cutoff; C has no filings (stored as none)
    assert any("not read completely" in n for n in notes)
    # EDGAR re-cut: one March filing of A is gone, an older one appears, C shows a filing now; B is read completely
    fresh2 = _ev(
        [
            ("A", "2026-01-05", "8k"),
            ("A", "2026-03-02", "form4"),
            ("A", "2026-11-02", "8k"),
            ("A", "2026-11-03", "form4"),
            ("B", "2026-05-01", "form4"),
            ("C", "2026-07-01", "sc13d"),
        ]
    )
    out, info, notes = us.freeze_form_events(
        store, fresh2, start=start, asked=["A", "B", "C"], eligible=["A", "B", "C"]
    )
    assert _rows(out) == [
        ("A", "2026-03-02", "form4"),
        ("A", "2026-03-02", "form4"),
        ("A", "2026-11-02", "8k"),
        ("A", "2026-11-03", "form4"),
        ("B", "2026-05-01", "form4"),
    ]
    assert info["added"] == 2 and info["dropped"] == 1 and info["new_keys"] == 1 and info["extended"] == 0
    assert any("ignored" in n for n in notes)
    out, _, _ = us.freeze_form_events(store, fresh2, start=start, asked=["B"], eligible=["B"])
    assert _rows(out) == [("B", "2026-05-01", "form4")]  # only the entities asked
    # the next month-end is frozen from the fresh read of that month
    later = VintageStore(root, today=date(2026, 12, 8))  # cutoff 2026-11-30
    out, info, _ = us.freeze_form_events(later, fresh2, start=start, asked=["A"], eligible=["A"])
    assert info["extended"] == 1 and info["rows_added"] == 2
    fresh3 = fresh2[~((fresh2["entity"] == "A") & (fresh2["kind"] == "8k"))]
    out, info, _ = us.freeze_form_events(later, fresh3, start=start, asked=["A"], eligible=["A"])
    assert _rows(out) == [
        ("A", "2026-03-02", "form4"),
        ("A", "2026-03-02", "form4"),
        ("A", "2026-11-02", "8k"),
        ("A", "2026-11-03", "form4"),
    ]
    assert info["dropped"] == 2 and info["extended"] == 0  # the second March Form 4 and the November 8-K


def test_load_us_sec_alt_freezes_the_forms_of_complete_ciks(tmp_path):
    settings = settings_for(tmp_path)
    net = FakeNet(settings, tmp_path / "pit", _subs_routes(1234))
    alt = us.load_us_sec_alt(settings, [1234], "2016-03-01", net=net, form345=False)
    years = alt.events["published"].dt.year.tolist()
    assert years == [2017, 2019, 2020, 2022, 2024]  # cut to the window asked after the Jan 1 read
    vint = alt.coverage["sec_forms"]["vintage"]
    assert vint["namespace"] == "us_sec_forms_v1_2016" and vint["new_keys"] == 1
    # EDGAR drops the 2022 filing from the recent block; the daily refresh fetches the main JSON again
    routes = _subs_routes(1234)
    main = json.loads(routes[f"{SUBMISSIONS}CIK0000001234.json"])
    main["filings"]["recent"] = block([("4", f"20{y}-06-01", "", "", "") for y in (24, 20)])
    routes[f"{SUBMISSIONS}CIK0000001234.json"] = json.dumps(main).encode()
    net.routes = routes
    net.path("sec", "subs/1234.json.gz").unlink()
    alt = us.load_us_sec_alt(settings, [1234], "2016-03-01", net=net, form345=False)
    assert alt.events["published"].dt.year.tolist() == years
    assert alt.coverage["sec_forms"]["vintage"]["dropped"] == 1
    assert alt.coverage["sec_forms"]["rows"] == 5


# =============================================================================== form 3/4/5 data sets

SUB_HDR = ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERNAME"]
TRANS_HDR = [
    "ACCESSION_NUMBER",
    "NONDERIV_TRANS_SK",
    "SECURITY_TITLE",
    "TRANS_DATE",
    "TRANS_CODE",
    "TRANS_SHARES",
    "TRANS_PRICEPERSHARE",
    "TRANS_ACQUIRED_DISP_CD",
]
OWNER_HDR = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP"]


def tsv(header: list[str], rows: list[list]) -> bytes:
    return ("\n".join("\t".join(map(str, r)) for r in [header, *rows]) + "\n").encode()


def form345_zip(subs: list[list], trans: list[list], owners: list[list], aff: list[str] | None = None) -> bytes:
    """A quarterly data-set zip with the three tables (``aff`` adds the 2023+ AFF10B5ONE column)."""
    sub_hdr = SUB_HDR + (["AFF10B5ONE"] if aff is not None else [])
    sub_rows = [[*r, aff[i]] for i, r in enumerate(subs)] if aff is not None else subs
    trans_rows = [[r[0], i + 1, "Common Stock", *r[1:]] for i, r in enumerate(trans)]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", tsv(sub_hdr, sub_rows))
        z.writestr("NONDERIV_TRANS.tsv", tsv(TRANS_HDR, trans_rows))
        z.writestr("REPORTINGOWNER.tsv", tsv(OWNER_HDR, owners))
        z.writestr("FOOTNOTES.tsv", b"ACCESSION_NUMBER\tFOOTNOTE_ID\tFOOTNOTE_TXT\n")
    return buf.getvalue()


def sample_zip() -> bytes:
    subs = [
        ["A1", "01-MAR-2024", "28-FEB-2024", "4", "0000320193", "Apple Inc."],
        ["A2", "01-MAR-2024", "28-FEB-2024", "4", "0000320193", "Apple Inc."],  # joint filing, two owners
        ["A3", "01-MAR-2024", "28-FEB-2024", "4", "0000320193", "Apple Inc."],
        ["A4", "05-MAR-2024", "28-FEB-2024", "4/A", "0000320193", "Apple Inc."],  # amends A3
        ["A5", "01-MAR-2024", "28-FEB-2024", "5", "0000320193", "Apple Inc."],  # Form 5: not used
        ["A6", "02-MAR-2024", "28-FEB-2024", "4", "0000789019", "Microsoft"],
    ]
    trans = [
        ["A1", "28-FEB-2024", "P", "100", "10", "A"],
        ["A1", "28-FEB-2024", "P", "50", "20", "A"],
        ["A2", "28-FEB-2024", "P", "10", "5", "A"],
        ["A3", "28-FEB-2024", "S", "200", "30", "D"],
        ["A3", "28-FEB-2024", "S", "0", "30", "D"],  # bad: zero shares
        ["A3", "28-FEB-2024", "S", "10", "", "D"],  # bad: no price
        ["A3", "28-FEB-2024", "M", "100", "1", "A"],  # option exercise: not open market
        ["A3", "28-FEB-2024", "P", "5", "5", "D"],  # purchase flagged disposed: contradictory
        ["A4", "28-FEB-2024", "S", "200", "30", "D"],  # restates A3's sale -> dropped
        ["A4", "29-FEB-2024", "S", "7", "30", "D"],  # new in the amendment -> kept on 03-05
        ["A5", "28-FEB-2024", "P", "1000", "10", "A"],
        ["A6", "01-MAR-2024", "S", "1", "1", "D"],
        ["A9", "01-MAR-2024", "S", "1", "1", "D"],  # no SUBMISSION row
    ]
    owners = [
        ["A1", "0000000111", "Insider One", "Director"],
        ["A2", "0000000222", "Fund LP", "TenPercentOwner"],
        ["A2", "0000000333", "Fund GP", "TenPercentOwner"],
        ["A3", "0000000111", "Insider One", "Director"],
        ["A4", "0000000111", "Insider One", "Director"],
        ["A5", "0000000111", "Insider One", "Director"],
        ["A6", "0000000444", "Insider Four", "Officer"],
    ]
    return form345_zip(subs, trans, owners, aff=["0", "false", "1", "", "0", "true"])


def test_form345_zip_keeps_open_market_form4_and_aggregates():
    rows, st = us.read_form345_zip(sample_zip())
    assert list(rows.columns) == us.ROW_COLUMNS
    assert set(rows["code"]) == {"P", "S"}  # the M row never reaches the cache
    assert st["no_submission"] == 1
    assert rows.loc[rows["accession"] == "A3", "aff10b5one"].eq(1.0).all()
    assert rows.loc[rows["accession"] == "A2", "owners"].iloc[0] == "222|333"
    ev, es = us.form345_events(rows, [320193], T("2024-01-01"), T("2024-12-31"))
    assert list(ev.columns) == EVENT_COLUMNS and (ev["source"] == "sec_form345").all()
    got = {(p.strftime("%m-%d"), k): v for p, k, v in zip(ev["published"], ev["kind"], ev["value"], strict=True)}
    assert got == {
        ("03-01", "insider_buy_value"): 100 * 10 + 50 * 20 + 10 * 5,
        ("03-01", "insider_buy_n"): 3,  # owner 111 + both co-filers of A2
        ("03-01", "insider_sell_value"): 200 * 30,
        ("03-01", "insider_sell_n"): 1,
        ("03-05", "insider_sell_value"): 7 * 30,
        ("03-05", "insider_sell_n"): 1,
    }
    assert set(ev["entity"]) == {"CIK0000320193"}
    assert es["bad_shares_or_price"] == 2
    assert es["code_ad_mismatch"] == 1
    assert es["amendment_restated"] == 1
    assert es["not_form4"] == 1
    # an accession posted again in the next quarter is counted once
    rows2 = pd.concat([rows.assign(quarter="2024q1"), rows.assign(quarter="2024q2")], ignore_index=True)
    ev2, es2 = us.form345_events(rows2, [320193], T("2024-01-01"), T("2024-12-31"))
    pd.testing.assert_frame_equal(ev2, ev)
    assert es2["duplicate_rows"] == len(rows[rows["issuer_cik"] == 320193])
    # a parquet round trip may return categorical columns: same events (no cartesian-product groupby)
    cat = rows.copy()
    for c in ("code", "owners", "doc_type", "ad", "accession"):
        cat[c] = cat[c].astype("category")
    ev4, _ = us.form345_events(cat, [320193], T("2024-01-01"), T("2024-12-31"))
    pd.testing.assert_frame_equal(ev4, ev)
    # window: published = FILING_DATE
    ev3, _ = us.form345_events(rows, [320193, 789019], T("2024-03-02"), T("2024-03-04"))
    assert ev3["entity"].unique().tolist() == ["CIK0000789019"]


def test_form345_tables_require_documented_columns():
    bad = pd.DataFrame({"ACCESSION_NUMBER": ["A"], "FILING_DATE": ["01-MAR-2024"]})
    with pytest.raises(ValueError, match="SUBMISSION"):
        us.parse_form345_tables(bad, pd.DataFrame(), pd.DataFrame())


def test_current_quarter_refresh_rule(tmp_path):
    now = time.time()
    day = 86400.0
    assert us.needs_download(None, False, now)
    assert us.needs_download(None, True, now)
    assert not us.needs_download(now - 400 * day, False, now)  # closed quarter: cached forever
    assert not us.needs_download(now - 6 * day, True, now)
    assert us.needs_download(now - 8 * day, True, now)  # latest listed quarter: refreshed after 7 days


def test_quarter_keys_and_index_links():
    assert us.quarter_keys(T("2013-02-01"), T("2013-10-01")) == ["2013q1", "2013q2", "2013q3", "2013q4"]
    assert us.quarter_end("2026q2") == T("2026-06-30")
    html = (
        '<a href="/files/datastandardsinnovation/data/insider-transactions-data-sets/2026q2_form345.zip" download>'
        '<a href="/files/structureddata/data/insider-transactions-data-sets/2026q1_form345.zip">'
        '<a href="/files/insider_transactions_readme.pdf">'
    )
    assert us.parse_form345_index(html) == {
        "2026q1": "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/2026q1_form345.zip",
        "2026q2": "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/"
        "2026q2_form345.zip",
    }


# =============================================================================== end to end (offline)

Q1 = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/2024q1_form345.zip"
Q2 = "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/2024q2_form345.zip"
INDEX_HTML = (
    f'<a href="{Q2.removeprefix("https://www.sec.gov")}">2024 Q2</a><a href="{Q1.removeprefix("https://www.sec.gov")}">'
).encode()


def world_routes(q2: bytes | None) -> dict[str, bytes]:
    recent = block(
        [
            ("8-K", "2024-05-02", "2024-05-02T20:30:00.000Z", "2.02,9.01", "001-36743"),
            ("4", "2024-03-01", "2024-03-01T21:00:00.000Z", "", ""),
        ]
    )
    main = {"cik": "320193", "name": "Apple Inc.", "filings": {"recent": recent, "files": []}}
    r = {
        f"{SUBMISSIONS}CIK0000320193.json": json.dumps(main).encode(),
        us.FORM345_INDEX: INDEX_HTML,
        Q1: sample_zip(),
    }
    if q2 is not None:
        r[Q2] = q2
    return r


def test_load_us_sec_alt_offline_counts_failures_and_caches(tmp_path):
    settings = settings_for(tmp_path)
    net = FakeNet(settings, tmp_path / "pit", world_routes(q2=None))  # 2024q2 zip missing -> failed quarter
    msgs: list[str] = []
    alt = us.load_us_sec_alt(settings, [320193, 999], "2024-01-01", "2024-08-15", net=net, progress=msgs.append)
    assert alt.market == "US"
    assert set(alt.events["source"]) == {"sec_forms", "sec_form345"}
    assert set(alt.events["kind"]) <= set(us.KINDS)
    forms = alt.coverage["sec_forms"]
    assert forms["ciks_asked"] == 2 and forms["ciks_with_events"] == 1 and forms["ciks_no_submissions"] == 1
    assert forms["rows_by_kind"] == {"8k": 1, "8k_2.02": 1, "form4": 1}
    f345 = alt.coverage["sec_form345"]
    assert f345["quarters_needed"] == 3 and f345["quarters_listed"] == 2
    assert f345["quarters_downloaded"] == 1 and f345["failed_quarters"] == ["2024q2"]
    assert f345["unlisted_quarters"] == ["2024q3"] and f345["covered_through"] == "2024-06-30"
    assert f345["rows"] == 6 and f345["dropped_amendment_restated"] == 1
    text = " ".join(alt.notes)
    assert "2024q2" in text and "covered through 2024-06-30" in text and "404" in text
    assert "still marked covered" in text
    assert alt.covered["sec_forms"] == ["CIK0000320193"]  # CIK 999 has no submissions JSON
    assert alt.covered["sec_form345"] == ["CIK0000000999", "CIK0000320193"]  # 2024q1 loaded (all issuers)
    assert any("2024q1" in m for m in msgs)
    folder = us.form345_cache_dir(settings)
    assert (folder / "2024q1.parquet").exists()
    assert not list(folder.glob("*.zip"))  # only the compact parquet is kept
    # second run: closed quarter and submissions come from the cache; the failed latest quarter is retried
    net2 = FakeNet(settings, tmp_path / "pit", world_routes(q2=sample_zip()))
    alt2 = us.load_us_sec_alt(settings, [320193], "2024-01-01", "2024-08-15", net=net2)
    assert Q1 not in net2.calls and f"{SUBMISSIONS}CIK0000320193.json" not in net2.calls
    assert net2.calls == [Q2]
    assert alt2.coverage["sec_form345"]["quarters_cached"] == 1
    assert alt2.coverage["sec_form345"]["quarters_downloaded"] == 1


def test_latest_quarter_is_refreshed_after_seven_days(tmp_path):
    settings = settings_for(tmp_path)
    routes = world_routes(q2=sample_zip())
    us.load_us_sec_alt(settings, [320193], "2024-01-01", "2024-06-30", net=FakeNet(settings, tmp_path / "pit", routes))
    folder = us.form345_cache_dir(settings)
    old = time.time() - 30 * 86400
    for key in ("2024q1", "2024q2"):
        os.utime(folder / f"{key}.parquet", (old, old))
    net = FakeNet(settings, tmp_path / "pit", routes)
    us.load_us_sec_alt(settings, [320193], "2024-01-01", "2024-06-30", net=net)
    assert net.calls == [Q2]  # 2024q2 is the latest listed quarter; 2024q1 is closed
    assert time.time() - (folder / "2024q2.parquet").stat().st_mtime < 60


def test_form345_can_be_switched_off(tmp_path):
    settings = settings_for(tmp_path)
    net = FakeNet(settings, tmp_path / "pit", world_routes(q2=None))
    alt = us.load_us_sec_alt(settings, [320193], "2024-01-01", "2024-08-15", net=net, form345=False)
    assert set(alt.events["source"]) == {"sec_forms"}
    assert alt.coverage["sec_form345"]["skipped"] is True
    assert alt.covered == {"sec_forms": ["CIK0000320193"], "sec_form345": []}
    assert any("form345=False" in n for n in alt.notes)
    assert us.FORM345_INDEX not in net.calls


def test_missing_older_page_keeps_events_but_not_covered(tmp_path):
    settings = settings_for(tmp_path)
    routes = _subs_routes(1234)
    del routes[f"{SUBMISSIONS}CIK0000001234-submissions-002.json"]
    net = FakeNet(settings, tmp_path / "pit", routes)
    alt = us.load_us_sec_alt(settings, [1234], "2011-01-01", "2024-12-31", net=net, form345=False)
    assert alt.covered["sec_forms"] == []
    assert alt.coverage["sec_forms"]["ciks_page_missing"] == 1
    assert len(alt.events) == 6  # recent + page 1 still used
    assert any("older submissions page" in n for n in alt.notes)
    alt = us.load_us_sec_alt(settings, [1234], "2016-01-01", "2024-12-31", net=net, form345=False)
    assert alt.covered["sec_forms"] == ["CIK0000001234"]  # page 2 is not needed for this window
