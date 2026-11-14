"""Job-posting snapshots (quantcio.data.alt.kr_jobs): API parsing, strict matching, append-only ledger - no network."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from quantcio.data.alt import kr_jobs
from quantcio.data.alt.kr_jobs import (
    SNAPSHOT_COLUMNS,
    JobsApiError,
    JobsQuotaError,
    append_snapshot,
    build_name_index,
    jobs_measures,
    parse_saramin,
    parse_work24,
    query_name,
    read_ledger,
    snapshot_job_postings,
)

FIX = Path(__file__).parent / "fixtures" / "alt_kr"
WHEN = date(2026, 10, 2)
SAMSUNG = {"005930": ["삼성전자(주)", "삼성전자"]}
HYNIX = {"000660": ["에스케이하이닉스(주)", "SK하이닉스"]}


def _fix(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


class Resp:
    def __init__(self, body: str, content_type: str):
        self.text = body
        self.content = body.encode("utf-8")
        self.headers = {"Content-Type": content_type}
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return json.loads(self.text)


class FakeApi:
    """Answers every call with a fixed body; records the query parameters."""

    def __init__(self, body: str, content_type: str):
        self.body, self.content_type = body, content_type
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, headers=None, **kw):
        self.calls.append((url, dict(params or {})))
        return Resp(self.body, self.content_type)


@pytest.fixture(autouse=True)
def _no_gap(monkeypatch):
    monkeypatch.setattr(kr_jobs, "MIN_GAP_S", {})


# ------------------------------------------------------------------------------------------------ parsing


def test_parse_saramin_fixture():
    total, posts = parse_saramin(json.loads(_fix("saramin_job_search.json")))
    assert total == 5 and len(posts) == 5
    p = posts[0]
    assert (p.company, p.bizr_no, p.active, p.posting_id) == ("삼성전자(주)", "1248100998", True, "50000001")
    assert p.posted == datetime(2026, 9, 30, 1, 0, tzinfo=UTC)  # 10:00 KST
    assert p.categories == ("IT개발·데이터", "연구·R&D")
    assert p.experience == "experienced"
    assert posts[1].bizr_no == "" and posts[1].experience == "new"


def test_parse_saramin_errors():
    with pytest.raises(JobsApiError, match="access-key") as e:
        parse_saramin(json.loads(_fix("saramin_error_invalid_key.json")))
    assert not isinstance(e.value, JobsQuotaError)
    with pytest.raises(JobsQuotaError):
        parse_saramin({"code": 4, "message": "일일 최대 요청 가능 횟수 초과"})


def test_parse_work24_fixture_and_errors():
    total, posts = parse_work24(_fix("work24_list_constructed.xml"))
    assert total == 3 and [p.company for p in posts][:2] == ["에스케이하이닉스(주)", "에스케이하이닉스 주식회사"]
    p = posts[0]
    assert p.bizr_no == "1268103725" and p.posted == datetime(2026, 9, 27, 15, 0, tzinfo=UTC)  # 26-09-28 KST
    assert p.expires == datetime(2026, 10, 12, 15, 0, tzinfo=UTC)  # closeDt 26-10-12 is still open that day
    assert posts[1].expires == datetime(2026, 12, 31, 15, 0, tzinfo=UTC)  # '채용시까지 26-12-31'
    assert [p.experience for p in posts] == ["experienced", "new", "any"]
    with pytest.raises(JobsApiError, match="authKey"):
        parse_work24(_fix("work24_error_no_key.xml"))
    with pytest.raises(JobsApiError, match="001"):
        parse_work24(_fix("worknet_legacy_error_no_key.xml"))


# ------------------------------------------------------------------------------------------------ matching


def test_name_index_drops_ambiguous_names_and_query_name():
    index, ambiguous = build_name_index({"A": ["삼성", "삼성A"], "B": ["(주)삼성"]})
    assert ambiguous == {"삼성"} and index == {"삼성a": "A"}
    assert query_name(["삼성전자(주)", "삼성전자"]) == "삼성전자"
    assert query_name(["주식회사 에코프로비엠"]) == "에코프로비엠"


def test_saramin_company_mode_matches_strictly():
    api = FakeApi(_fix("saramin_job_search.json"), "application/json")
    df = snapshot_job_postings(
        {**SAMSUNG, **HYNIX},
        WHEN,
        saramin_key="K",
        session=api,
        bizr_nos={"005930": "124-81-00998"},
        saramin_mode="company",
    )
    assert list(df.columns) == SNAPSHOT_COLUMNS
    s = df.set_index("entity").loc["005930"]
    assert s["source"] == "saramin" and s["date"] == "2026-10-02"
    assert (s["postings_matched"], s["active_postings"], s["new_7d"], s["excluded_other"]) == (3, 2, 1, 2)
    assert json.loads(s["categories"]) == {"연구·R&D": 2, "IT개발·데이터": 1}
    assert json.loads(s["experience"]) == {"experienced": 1, "new": 1}
    assert bool(s["complete"]) and s["query_hits"] == 5 and s["api_calls"] == 1
    h = df.set_index("entity").loc["000660"]  # every hit belongs to another company: excluded, zero postings
    assert (h["active_postings"], h["excluded_other"]) == (0, 5)
    url, params = api.calls[0]
    assert url == kr_jobs.SARAMIN_URL
    assert params["keywords"] == "삼성전자" and params["access-key"] == "K" and params["start"] == 0
    assert params["stock"] == "kospi,kosdaq,konex" and params["count"] == 110
    assert any("2026-10-02" in n for n in df.attrs["notes"])  # not today: the APIs only know the present


def test_saramin_sweep_mode_and_quota_stop():
    api = FakeApi(_fix("saramin_job_search.json"), "application/json")
    df = snapshot_job_postings({**SAMSUNG, **HYNIX}, WHEN, saramin_key="K", session=api, saramin_mode="sweep")
    assert len(api.calls) == 1 and "keywords" not in api.calls[0][1]
    assert set(df["entity"]) == {"005930", "000660"} and df["complete"].all()
    assert df.set_index("entity").loc["005930", "postings_matched"] == 3  # job 1 + 5 by name, job 2 by name
    quota = FakeApi(json.dumps({"code": 4, "message": "일일 최대 요청 가능 횟수 초과"}), "application/json")
    df = snapshot_job_postings({**SAMSUNG, **HYNIX}, WHEN, saramin_key="K", session=quota, saramin_mode="company")
    assert df.empty and len(quota.calls) == 1
    assert any("한도" in n for n in df.attrs["notes"])


def test_work24_snapshot_and_keyless_sources():
    api = FakeApi(_fix("work24_list_constructed.xml"), "application/xml")
    df = snapshot_job_postings(HYNIX, WHEN, work24_key="W", session=api, bizr_nos={"000660": "1268103725"})
    row = df.iloc[0]
    assert (row["source"], row["postings_matched"], row["active_postings"], row["new_7d"]) == ("work24", 2, 2, 1)
    assert row["excluded_other"] == 1 and row["query"] == "coNm=에스케이하이닉스"
    params = api.calls[0][1]
    assert (params["authKey"], params["callTp"], params["returnType"], params["startPage"]) == ("W", "L", "XML", 1)
    assert any("saramin 건너뜀" in n for n in df.attrs["notes"])
    none = snapshot_job_postings(HYNIX, WHEN, session=FakeApi("", "x"))
    assert none.empty and list(none.columns) == SNAPSHOT_COLUMNS
    assert sum("건너뜀" in n for n in none.attrs["notes"]) == 2


# ------------------------------------------------------------------------------------------------ ledger


def _snapshot(day: date) -> pd.DataFrame:
    api = FakeApi(_fix("saramin_job_search.json"), "application/json")
    return snapshot_job_postings({**SAMSUNG, **HYNIX}, day, saramin_key="K", session=api, saramin_mode="company")


def test_append_snapshot_is_append_only_and_idempotent(tmp_path):
    path = tmp_path / "alt_ledger" / "kr_jobs.jsonl"
    first = _snapshot(WHEN)
    assert append_snapshot(first, path) == 2
    before = path.read_bytes()
    assert append_snapshot(first, path) == 0  # same (date, entity, source): nothing written
    assert path.read_bytes() == before
    assert append_snapshot(_snapshot(date(2026, 10, 5)), path) == 2
    assert path.read_bytes().startswith(before)  # history never rewritten
    led = read_ledger(path)
    assert len(led) == 4 and list(led.columns) == SNAPSHOT_COLUMNS
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["entity"] == "005930"
    assert append_snapshot(pd.DataFrame(columns=SNAPSHOT_COLUMNS), path) == 0


def test_jobs_measures_from_ledger(tmp_path):
    path = tmp_path / "kr_jobs.jsonl"
    append_snapshot(_snapshot(WHEN), path)
    later = _snapshot(date(2026, 10, 5))
    later.loc[later["entity"] == "000660", "complete"] = False  # e.g. a page cap was hit
    append_snapshot(later, path)
    alt = jobs_measures(path)
    m = alt.measures
    assert alt.market == "KR" and set(m["item"]) == set(kr_jobs.ITEMS) and set(m["source"]) == {"kr_jobs_saramin"}
    assert len(m) == 2 * 3  # 3 complete (entity, date) rows x 2 items
    sam = m[(m["entity"] == "005930") & (m["item"] == "job_postings_active")].sort_values("published")
    assert sam["published"].tolist() == [pd.Timestamp("2026-10-02"), pd.Timestamp("2026-10-05")]
    assert (sam["published"] == sam["period_end"]).all()
    assert sam["value"].tolist() == [2.0, 2.0]
    assert alt.covered == {"kr_jobs_saramin": ["005930"], "kr_jobs_work24": []}
    cov = alt.coverage["kr_jobs_saramin"]
    assert (cov["snapshot_days"], cov["incomplete_rows"], cov["first_day"]) == (2, 1, "2026-10-02")
    assert any("미완료" in n for n in alt.notes) and any("12개월" in n for n in alt.notes)
    assert jobs_measures(tmp_path / "missing.jsonl").measures.empty
