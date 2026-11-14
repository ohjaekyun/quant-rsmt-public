"""NPS workplace file (quantcio.data.alt.kr_nps): parsing, matching, point-in-time dates - no network."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quantcio.config import load_settings
from quantcio.data.alt import kr_nps
from quantcio.data.alt.kr_nps import (
    COMPACT_COLUMNS,
    NpsDownloadLimitError,
    NpsFile,
    conservative_published,
    download_csv,
    download_kr_nps_history,
    earliest_data_month,
    estimate_data_month,
    ingest_csv,
    load_kr_nps,
    match_rows,
    month_end,
    norm_company_name,
    parse_detail_info,
    parse_detail_pk,
    parse_hist_listing,
    parse_nps_csv,
    release_published,
)

FIX = Path(__file__).parent / "fixtures" / "alt_kr"
HEADER = (
    "자료생성년월,사업장명,사업자등록번호,사업장가입상태코드 1 등록 2 탈퇴,우편번호,사업장지번상세주소,"
    "사업장도로명상세주소,고객법정동주소코드,고객행정동주소코드,법정동주소광역시도코드,법정동주소광역시시군구코드,"
    "법정동주소광역시시군구읍면동코드,사업장형태구분코드 1 법인 2 개인,사업장업종코드,사업장업종코드명,적용일자,"
    "재등록일자,탈퇴일자,가입자수,당월고지금액,신규취득자수,상실가입자수"
)


def _row(month: str, name: str, bizr6: str, form: str, members: int, new: int, lost: int, status: str = "1") -> str:
    return (
        f"{month},{name},{bizr6},{status},16677,경기도 수원시 영통구 매탄동,경기도 수원시 영통구 삼성로,4111710300,"
        f"4111759000,41,117,103,{form},323001,반도체 제조업,1969-01-13,,,{members},{members * 300000},{new},{lost}"
    )


def write_csv(path: Path, rows: list[str], header: str = HEADER, encoding: str = "cp949") -> Path:
    path.write_bytes(("\n".join([header, *rows]) + "\n").encode(encoding))
    return path


SAMPLE_ROWS = [
    _row("2026-07", "삼성전자(주)", "124810", "1", 125000, 600, 500),
    _row("2026-07", "삼성전자(주)", "124810", "1", 720, 37, 14),  # second NPS unit of the same company: summed
    _row("2026-07", "SK네트웍스(주)", "124810", "1", 242, 0, 44),  # prefix collision, other company
    _row("2026-07", "삼성전자서비스(주)", "124810", "1", 50, 1, 1),  # contains the name, but an exact row exists
    _row("2026-07", "홍길동상회", "124810", "2", 3, 0, 0),  # 개인 사업장: dropped
    _row("2026-07", "에스케이하이닉스 주식회사", "126810", "1", 36777, 1340, 67),
    _row("2026-07", "주식회사에코프로비엠 오창공장", "638860", "1", 1049, 9, 19),  # only a contained match
    _row("2026-07", "(주)셀트리온제약", "133812", "1", 700, 5, 5),  # contained, but there is an exact row below
    _row("2026-07", "(주)셀트리온", "133812", "1", 3088, 43, 34),
]


# ------------------------------------------------------------------------------------------------ parsing


def test_parse_nps_csv_cp949_keeps_corporate_rows_and_compact_columns(tmp_path):
    df = parse_nps_csv(write_csv(tmp_path / "nps.csv", SAMPLE_ROWS))
    assert list(df.columns) == COMPACT_COLUMNS
    assert len(df) == len(SAMPLE_ROWS) - 1  # the 개인 row is gone
    assert "홍길동상회" not in set(df["name"])
    assert set(df["data_month"]) == {"2026-07"}
    sam = df[df["name"] == "삼성전자(주)"]
    assert sam["members"].tolist() == [125000.0, 720.0]
    assert sam["bizr6"].iloc[0] == "124810"
    assert df["status"].dtype == "int8"


def test_parse_nps_csv_utf8_and_old_month_format_and_leading_zeros(tmp_path):
    rows = [_row("201511", "(주)대신", "012810", "1", 10, 1, 0)]
    df = parse_nps_csv(write_csv(tmp_path / "old.csv", rows, encoding="utf-8"))
    assert df["data_month"].tolist() == ["2015-11"]
    assert df["bizr6"].tolist() == ["012810"]  # read as text: leading zero kept


def test_parse_nps_csv_rejects_unknown_layout(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_bytes("a,b,c\n1,2,3\n".encode("cp949"))
    with pytest.raises(ValueError, match="컬럼 없음"):
        parse_nps_csv(path)


# ------------------------------------------------------------------------------------------------ matching


def test_norm_company_name():
    assert norm_company_name("에스케이하이닉스 주식회사") == "에스케이하이닉스"
    assert norm_company_name("에스케이하이닉스(주)") == "에스케이하이닉스"
    assert norm_company_name("㈜ 셀트리온") == "셀트리온"
    assert norm_company_name("주식회사에코프로비엠") == "에코프로비엠"
    assert norm_company_name("NAVER Corp.") == "naver"
    assert norm_company_name("Princeton (주)") == "princeton"  # 'inc' inside a word is not a legal form


def test_match_rows_exact_sums_branches_and_excludes_prefix_collisions(tmp_path):
    rows = parse_nps_csv(write_csv(tmp_path / "nps.csv", SAMPLE_ROWS))
    got, quality, excluded = match_rows(rows, "124810", ["삼성전자(주)", "삼성전자"])
    assert quality == "exact"
    assert got["members"].sum() == 125720.0  # both NPS units of the company
    assert excluded == 2  # SK네트웍스 and 삼성전자서비스 (exact rows win over contained ones)


def test_match_rows_contained_only_without_exact_and_none():
    rows = pd.DataFrame(
        {
            "bizr6": ["638860", "638860", "133812", "133812", "999999"],
            "name": ["주식회사에코프로비엠 오창공장", "주식회사무무즈", "(주)셀트리온제약", "(주)셀트리온", "x"],
            "members": [1049.0, 43.0, 700.0, 3088.0, 1.0],
        }
    )
    got, quality, excluded = match_rows(rows, "638860", ["(주)에코프로비엠", "에코프로비엠"])
    assert (quality, got["members"].tolist(), excluded) == ("contained", [1049.0], 1)
    got, quality, _ = match_rows(rows, "133812", ["(주)셀트리온"])
    assert (quality, got["members"].tolist()) == ("exact", [3088.0])
    got, quality, excluded = match_rows(rows, "638860", ["삼성전자"])
    assert (quality, len(got), excluded) == ("none", 0, 2)
    assert match_rows(rows, "000000", ["a"])[1:] == ("none", 0)


# ------------------------------------------------------------------------------------------- point in time


def test_conservative_published_is_end_of_next_month():
    assert conservative_published("2026-08") == pd.Timestamp("2026-09-30")
    assert conservative_published("2025-12") == pd.Timestamp("2026-01-31")
    assert month_end("2024-02") == pd.Timestamp("2024-02-29")


def test_release_published_rules():
    assert release_published("2026-07", "2026-08-25", "2026-08-25") == (pd.Timestamp("2026-08-25"), "release")
    # modified later than registered: the downloadable bytes may be the modified version
    assert release_published("2025-09", "2025-10-31", "2025-11-19") == (pd.Timestamp("2025-11-19"), "release")
    # the 2019-08-05 batch registration of old months
    assert release_published("2015-11", "2019-08-05", "2019-08-06") == (
        pd.Timestamp("2019-08-06"),
        "backfill_registration",
    )
    assert release_published("2015-11", "2019-08-05", "2019-08-06", "month_after") == (
        pd.Timestamp("2015-12-31"),
        "backfill_month_after",
    )
    assert release_published("2026-08") == (pd.Timestamp("2026-09-30"), "unknown_month_after")


# ------------------------------------------------------------------------------------------- portal listing


def test_parse_hist_listing_fixture():
    rows = parse_hist_listing((FIX / "nps_hist_listing.html").read_text(encoding="utf-8"))
    assert len(rows) == 9
    assert rows[0] == {
        "detail_pk": "uddi:b5ac0771-a9e3-4ce0-9505-ad0439d97b79",
        "hist_sn": "1",
        "title": "국민연금공단_국민연금 가입 사업장 내역_20260825",
        "registered": "2026-08-25",
    }
    assert any(r["hist_sn"] == "2" for r in rows)  # revised entries carry their history serial
    last = rows[-1]
    assert (last["title"], last["registered"]) == ("국민연금 가입 사업장 내역 2015년 12월", "2019-08-05")


def test_parse_hist_listing_requires_the_tab():
    with pytest.raises(ValueError, match="tab-layer-file-05"):
        parse_hist_listing("<html><div id='tab-layer-file-04'></div></html>")


def test_parse_detail_info_and_page_pk_fixtures():
    info = parse_detail_info((FIX / "nps_detail_popup.html").read_text(encoding="utf-8"))
    assert info["atch_file_id"] == "FILE_000000007629828"
    assert info["file_sn"] == "1"  # the CSV (sn 2 is the JSON version)
    assert (info["registered"], info["modified"], info["next_due"]) == ("2026-08-25", "2026-08-25", "2026-09-28")
    pk = parse_detail_pk((FIX / "nps_page_snippet.html").read_text(encoding="utf-8"))
    assert pk == "uddi:8e0d590f-768e-4cbd-9560-c7168c8d13b6"


def test_estimate_data_month_and_earliest():
    assert estimate_data_month("국민연금공단_국민연금 가입 사업장 내역_20260825", "2026-08-25") == "2026-07"
    assert estimate_data_month("국민연금 가입 사업장 내역 2015년 12월", "2019-08-05") == "2015-11"  # verified
    files = [
        NpsFile("a", "1", "t", "2026-08-25", data_month="2026-07"),
        NpsFile("b", "1", "t", "2019-08-05", data_month="2015-11", month_estimated=False),
    ]
    assert earliest_data_month(files) == ("2015-11", True)
    assert earliest_data_month([]) == ("", False)
    assert files[1].backfilled and not files[0].backfilled
    assert files[0].url.endswith("fileDetailSn=1&insertDataPrcus=N")


# ------------------------------------------------------------------------------------------- download / store


class FakeResponse:
    def __init__(self, *, json_data=None, content=b"", headers=None, status=200):
        self._json = json_data
        self.content = content
        self.headers = headers or {}
        self.status_code = status
        self.text = content.decode("utf-8", "replace") if isinstance(content, bytes) else str(content)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakePortal:
    """check-limit + download of in-memory files; counts calls."""

    def __init__(self, files: dict[str, bytes], need_captcha: bool = False):
        self.files = files
        self.need_captcha = need_captcha
        self.downloads: list[str] = []

    def post(self, url, data=None, **kw):
        assert url == kr_nps.LIMIT_URL
        return FakeResponse(json_data={"needCaptcha": self.need_captcha})

    def get(self, url, params=None, **kw):
        assert url == kr_nps.DOWNLOAD_URL
        body = self.files[params["atchFileId"]]
        self.downloads.append(params["atchFileId"])
        return FakeResponse(content=body, headers={"Content-Length": str(len(body)), "Content-Type": "text/csv"})


class NoNetwork:
    def get(self, *a, **kw):
        raise AssertionError("network used")

    post = get


@pytest.fixture(autouse=True)
def _no_gap(monkeypatch):
    monkeypatch.setattr(kr_nps._gate, "gap", 0.0)


def test_download_csv_streams_and_detects_limit_and_html(tmp_path):
    body = "\n".join([HEADER, *SAMPLE_ROWS]).encode("cp949")
    f = NpsFile("pk", "1", "t", "2026-08-25", atch_file_id="FILE_1", data_month="2026-07")
    path = download_csv(FakePortal({"FILE_1": body}), f, tmp_path)
    assert path.read_bytes() == body and not (tmp_path / "FILE_1.csv.part").exists()
    with pytest.raises(NpsDownloadLimitError):
        download_csv(
            FakePortal({"FILE_2": body}, need_captcha=True),
            NpsFile("p", "1", "t", "x", atch_file_id="FILE_2"),
            tmp_path,
        )
    with pytest.raises(RuntimeError, match="HTML"):
        download_csv(
            FakePortal({"FILE_3": b"<html>error</html>"}), NpsFile("p", "1", "t", "x", atch_file_id="FILE_3"), tmp_path
        )
    assert not list(tmp_path.glob("*.part"))  # no partial file left behind


def _settings(tmp_path):
    return load_settings(data={"cache_dir": str(tmp_path / "cache")})


def test_bulk_download_is_resumable_and_never_downloads_a_file_twice(tmp_path):
    settings = _settings(tmp_path)
    body_jul = "\n".join([HEADER, *SAMPLE_ROWS]).encode("cp949")
    body_aug = "\n".join([HEADER, *(r.replace("2026-07", "2026-08") for r in SAMPLE_ROWS)]).encode("cp949")
    files = [
        NpsFile("p2", "1", "t", "2026-09-23", "2026-09-23", "FILE_AUG", data_month="2026-08", current=True),
        NpsFile("p1", "1", "t", "2026-08-25", "2026-08-25", "FILE_JUL", data_month="2026-07"),
        NpsFile("p0", "1", "t", "2019-08-05", "2019-08-06", "FILE_OLD", data_month="2015-11"),
    ]
    portal = FakePortal({"FILE_AUG": body_aug, "FILE_JUL": body_jul})
    seen: list[str] = []
    s1 = download_kr_nps_history(
        settings,
        date(2026, 8, 1),
        date(2026, 8, 31),
        session=portal,
        files=files[:1],
        progress=lambda i, n, m: seen.append(m),
    )
    assert (s1.downloaded, s1.failed) == (1, []) and seen and seen[0].startswith("2026-08")
    s2 = download_kr_nps_history(settings, date(2026, 7, 1), date(2026, 8, 31), session=portal, files=files)
    assert (s2.selected, s2.skipped_cached, s2.downloaded) == (2, 1, 1)  # 2015-11 is outside the range
    assert portal.downloads == ["FILE_AUG", "FILE_JUL"]
    root = kr_nps.nps_root(settings)
    assert sorted(p.name for p in root.glob("*.parquet")) == ["202607.parquet", "202608.parquet"]
    assert not list((root / "_tmp").glob("*.csv"))  # CSVs deleted after ingest
    assert json.loads((root / "_files.json").read_text()) == {"FILE_AUG": "2026-08", "FILE_JUL": "2026-07"}
    s3 = download_kr_nps_history(
        settings, None, None, session=FakePortal({}, need_captcha=True), files=files
    )  # only FILE_OLD left -> stops at the CAPTCHA answer
    assert s3.stopped and s3.downloaded == 0


def test_ingest_keeps_the_earlier_release_of_a_duplicated_month(tmp_path):
    root = tmp_path / "nps"
    body = [HEADER, *SAMPLE_ROWS]
    late = ingest_csv(root, write_csv(tmp_path / "a.csv", body[1:]), registered="2021-10-15", atch_file_id="A")
    assert late["atch_file_id"] == "A"
    kept = ingest_csv(root, write_csv(tmp_path / "b.csv", body[1:]), registered="2021-09-24", atch_file_id="B")
    assert kept["atch_file_id"] == "B" and kept["duplicates"][0]["atch_file_id"] == "A"
    again = ingest_csv(root, write_csv(tmp_path / "c.csv", body[1:]), registered="2022-01-01", atch_file_id="C")
    assert again["atch_file_id"] == "B"  # a later copy never replaces the earlier one
    assert json.loads((root / "_files.json").read_text()) == {"A": "2026-07", "B": "2026-07", "C": "2026-07"}
    assert not (tmp_path / "c.csv").exists()


# ------------------------------------------------------------------------------------------------ loader


def _seed_company_cache(settings, companies: dict[str, dict]) -> None:
    root = settings.cache_path / "alt" / "kr_dart_company"
    root.mkdir(parents=True, exist_ok=True)
    (root / "_corp_codes.json").write_text(json.dumps({e: c["corp_code"] for e, c in companies.items()}))
    for c in companies.values():
        (root / f"{c['corp_code']}.json").write_text(json.dumps({"status": "000", **c}, ensure_ascii=False))


def test_load_kr_nps_end_to_end_offline(tmp_path):
    settings = _settings(tmp_path)
    root = kr_nps.nps_root(settings)
    ingest_csv(
        root,
        write_csv(tmp_path / "jul.csv", SAMPLE_ROWS),
        registered="2026-08-25",
        modified="2026-08-25",
        atch_file_id="F7",
    )
    files = [NpsFile("p", "1", "t", "2026-08-25", "2026-08-25", "F7", data_month="2026-07")]
    (root / "_listing.json").write_text(
        json.dumps({"fetched_at": "2026-10-02T00:00:00+00:00", "files": [f.__dict__ for f in files]})
    )
    _seed_company_cache(
        settings,
        {
            "005930": {
                "corp_code": "00126380",
                "corp_name": "삼성전자(주)",
                "stock_name": "삼성전자",
                "bizr_no": "1248100998",
            },
            "000660": {
                "corp_code": "00164779",
                "corp_name": "에스케이하이닉스(주)",
                "stock_name": "SK하이닉스",
                "bizr_no": "1268103725",
            },
            "247540": {
                "corp_code": "01160363",
                "corp_name": "(주)에코프로비엠",
                "stock_name": "에코프로비엠",
                "bizr_no": "6388600376",
            },
            "111111": {
                "corp_code": "00999999",
                "corp_name": "(주)없는회사",
                "stock_name": "없는회사",
                "bizr_no": "5558100001",
            },
        },
    )
    alt = load_kr_nps(
        settings,
        ["005930", "000660", "247540", "111111", "222222"],
        date(2026, 7, 1),
        date(2026, 8, 31),
        download=False,
        session=NoNetwork(),
    )
    m = alt.measures
    assert alt.market == "KR" and set(m["item"]) == set(kr_nps.ITEMS) and set(m["source"]) == {"kr_nps"}
    val = m.set_index(["entity", "item"])["value"]
    assert val[("005930", "nps_members")] == 125720.0 and val[("005930", "nps_new")] == 637.0
    assert val[("000660", "nps_lost")] == 67.0
    assert val[("247540", "nps_members")] == 1049.0  # contained match
    assert (m["published"] == pd.Timestamp("2026-08-25")).all()
    assert (m["period_end"] == pd.Timestamp("2026-07-31")).all()
    cov = alt.coverage["kr_nps"]
    assert (cov["entities_asked"], cov["entities_matched"], cov["entities_unmatched"]) == (5, 3, 2)
    assert cov["unmatched"] == ["111111", "222222"]
    assert cov["match_quality"] == {"exact": 2, "contained": 1, "none": 1}
    assert cov["prefix_rows_excluded"] >= 2 and cov["earliest_month_available"] == "2026-07"
    assert cov["months_missing"] == []
    assert alt.covered == {"kr_nps": ["000660", "005930", "247540"]}  # complete range -> matched entities


def test_load_kr_nps_incomplete_range_covers_nobody(tmp_path):
    settings = _settings(tmp_path)
    root = kr_nps.nps_root(settings)
    ingest_csv(root, write_csv(tmp_path / "jul.csv", SAMPLE_ROWS), registered="2026-08-25", atch_file_id="F7")
    files = [
        NpsFile("p", "1", "t", "2026-08-25", "2026-08-25", "F7"),
        NpsFile("q", "1", "t", "2026-09-23", "2026-09-23", "F8", current=True),  # offered but not downloaded
    ]
    (root / "_listing.json").write_text(
        json.dumps({"fetched_at": "2026-10-02T00:00:00+00:00", "files": [f.__dict__ for f in files]})
    )
    _seed_company_cache(
        settings,
        {
            "005930": {
                "corp_code": "00126380",
                "corp_name": "삼성전자(주)",
                "stock_name": "삼성전자",
                "bizr_no": "1248100998",
            }
        },
    )
    alt = load_kr_nps(settings, ["005930"], date(2026, 7, 1), date(2026, 8, 31), download=False, session=NoNetwork())
    assert len(alt.measures) == 3
    assert alt.coverage["kr_nps"]["months_missing"] == ["2026-08"]
    assert alt.covered == {"kr_nps": []}
    assert any("미수집" in n for n in alt.notes)
