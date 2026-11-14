"""KR OpenDART alternative data (quantcio.data.alt.kr_dart_events): parsing, mapping, quota / network handling."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
import requests

from quantcio.config import load_settings
from quantcio.data.alt import kr_dart_events as kd


def test_classify_report_names():
    assert kd.classify_report("주요사항보고서(자기주식취득결정)") == ("buyback", False)
    assert kd.classify_report("[기재정정]단일판매ㆍ공급계약체결") == ("supply_contract", True)
    assert kd.classify_report("단일판매ㆍ공급계약해지") == ("supply_contract_cancel", False)
    assert kd.classify_report("[첨부정정]주요사항보고서(전환사채권발행결정)") == ("convertible_issue", True)
    assert kd.classify_report("무상증자결정")[0] == "bonus_issue"
    assert kd.classify_report("유무상증자결정")[0] == "rights_issue"
    assert kd.classify_report("연결재무제표기준영업(잠정)실적(공정공시)")[0] == "earnings_prelim"
    assert kd.classify_report("투자주의환기종목지정")[0] == "admin_issue"
    assert kd.classify_report("감사보고서제출") == (None, False)
    assert kd.classify_report("[발행조건확정]증권신고서(지분증권)") == (None, False)  # a prefix that is no correction


def test_list_events_dedupes_and_counts_filings():
    rows = [
        {"rcept_no": "20240105000001", "rcept_dt": "20240105", "report_nm": "주요사항보고서(자기주식취득결정)"},
        {"rcept_no": "20240105000001", "rcept_dt": "20240105", "report_nm": "주요사항보고서(자기주식취득결정)"},
        {"rcept_no": "20240201000002", "rcept_dt": "20240201", "report_nm": "[기재정정]기업설명회(IR)개최"},
        {"rcept_no": "x", "rcept_dt": "bad", "report_nm": "기업설명회(IR)개최"},
    ]
    ev = kd.list_events("005930", rows)
    counts = ev.groupby("kind").size().to_dict()
    assert counts == {"filing": 2, "buyback": 1, "ir": 1, "correction": 1}
    assert set(ev["source"]) == {"kr_dart_list"}
    assert ev["published"].min() == pd.Timestamp("2024-01-05")


def test_ownership_events_signs_and_zero_changes():
    ins = kd.insider_events(
        "A",
        [
            {"rcept_dt": "2024-01-02", "sp_stock_lmp_irds_cnt": "1,000"},
            {"rcept_dt": "2024-01-03", "sp_stock_lmp_irds_cnt": "-250"},
            {"rcept_dt": "2024-01-04", "sp_stock_lmp_irds_cnt": "0"},
            {"rcept_dt": "2024-01-05", "sp_stock_lmp_irds_cnt": "-"},
        ],
    )
    k = ins.groupby("kind")["value"].sum().to_dict()
    assert k == {"insider_net_shares": 750.0, "insider_buy_n": 1.0, "insider_sell_n": 1.0}
    blk = kd.block_events("A", [{"rcept_dt": "2024-02-01", "stkrt_irds": "-1.25"}])
    assert blk.groupby("kind")["value"].sum().to_dict() == {"block_pct_change": -1.25, "block_down_n": 1.0}


def _emp(fo, sex, sm, cnt="-", ten="-", sal="-", rno="20250311000123", stlm="2024-12-31"):
    return {
        "rcept_no": rno,
        "fo_bbm": fo,
        "sexdstn": sex,
        "sm": sm,
        "cnttk_co": cnt,
        "avrg_cnwk_sdytrn": ten,
        "fyer_salary_totamt": sal,
        "stlm_dt": stlm,
    }


def test_employee_measures_use_total_rows_without_double_counting():
    # Samsung-style: division x sex detail rows + one 성별합계 row per sex (salary only on the totals)
    rows = [
        _emp("DX", "남", "38,291", "338", "16.9"),
        _emp("DX", "여", "12,520", "110", "13.8"),
        _emp("성별합계", "남", "94,913", "497", "13.4", "12,863,945,000,000"),
        _emp("성별합계", "여", "34,567", "137", "12.2", "3,407,173,000,000"),
    ]
    m = kd.employee_measures("005930", rows).set_index("item")["value"]
    assert m["headcount"] == 94913 + 34567
    assert m["contract_workers"] == 497 + 137
    assert m["salary_total"] == pytest.approx(12_863_945_000_000 + 3_407_173_000_000)
    assert m["avg_tenure_years"] == pytest.approx((13.4 * 94913 + 12.2 * 34567) / (94913 + 34567))
    pub = kd.employee_measures("005930", rows)["published"].iloc[0]
    assert pub == pd.Timestamp("2025-03-11")  # the receipt date inside rcept_no
    # a grand total (total in both fields) is used alone
    g = kd.employee_measures("X", [*rows, _emp("합계", "합계", "129,480")]).set_index("item")["value"]
    assert g["headcount"] == 129480
    # no total rows: detail rows are summed; division totals by sex column
    d = kd.employee_measures("X", [_emp("전사", "남", "2,607", "8"), _emp("전사", "여", "1,776", "34")])
    assert d.set_index("item")["value"]["headcount"] == 4383
    s = kd.employee_measures("X", [_emp("A", "합계", "100"), _emp("B", "합계", "50"), _emp("A", "남", "60")])
    assert s.set_index("item")["value"]["headcount"] == 150


def test_map_corp_codes_by_stock_code_then_unique_name():
    corps = pd.DataFrame(
        {
            "corp_code": ["001", "002", "003", "004"],
            "corp_name": ["삼성전자", "옛회사", "중복", "중복"],
            "stock_code": ["005930", "", "", ""],
            "modify_date": ["", "", "", ""],
        }
    )
    m, how = kd.map_corp_codes(
        ["005930", "123456", "654321", "999999"], corps, {"123456": ["(주)옛회사"], "654321": ["중복"]}
    )
    assert m == {"005930": "001", "123456": "002"}
    assert how == {"005930": "stock_code", "123456": "name", "654321": "ambiguous_name", "999999": "unmapped"}


class _Resp:
    def __init__(self, payload):
        self._p = payload
        self.headers = {"Content-Type": "application/json"}

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


class _Session:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return _Resp(a)


@pytest.fixture
def dsettings(tmp_path):
    s = load_settings(data={"cache_dir": str(tmp_path / "c")})
    s.data.dart_api_key = "SECRETKEY"
    return s


def test_quota_stops_and_key_is_redacted(dsettings, tmp_path):
    sess = _Session([{"status": "020", "message": "limit"}])
    cl = kd.DartAltClient(dsettings, session=sess, root=tmp_path / "d", today=date(2026, 10, 3))
    with pytest.raises(kd.DartQuotaError):
        cl.whole_history("elestock", "00126380", 7)
    assert cl.quota_hit
    with pytest.raises(kd.DartQuotaError):  # no further network call after the quota
        cl.get("majorstock", corp_code="1")
    assert len(sess.calls) == 1
    assert cl.redact("https://x?crtfc_key=SECRETKEY&a=1") == "https://x?crtfc_key=***&a=1"


def test_network_patience_then_stop(dsettings, tmp_path):
    err = requests.ConnectionError("Failed to resolve host ... crtfc_key=SECRETKEY")
    sess = _Session([err] * 50)
    cl = kd.DartAltClient(dsettings, session=sess, root=tmp_path / "d", today=date(2026, 10, 3))
    slept = []
    cl.sleep = slept.append
    with pytest.raises(kd.DartNetworkError) as ei:
        cl.get("elestock", corp_code="1")
    assert "SECRETKEY" not in str(ei.value) and sum(slept) >= kd.NETWORK_PATIENCE_S
    assert cl.network_down
    # recovery inside the patience window is transparent
    sess2 = _Session([err, err, {"status": "000", "list": []}])
    cl2 = kd.DartAltClient(dsettings, session=sess2, root=tmp_path / "d2", today=date(2026, 10, 3))
    cl2.sleep = lambda _s: None
    assert cl2.get("elestock", corp_code="1")["status"] == "000"


def test_cache_reuse_and_incremental_list_refresh(dsettings, tmp_path):
    page = {"status": "000", "total_page": 1, "list": [{"rcept_no": "1", "rcept_dt": "20240101", "report_nm": "a"}]}
    sess = _Session([page])
    cl = kd.DartAltClient(dsettings, session=sess, root=tmp_path / "d", today=date(2026, 10, 1))
    assert len(cl.disclosures("C", "B", 7)) == 1
    assert len(cl.disclosures("C", "B", 7)) == 1 and len(sess.calls) == 1  # cached
    newer = {"status": "000", "total_page": 1, "list": [{"rcept_no": "2", "rcept_dt": "20261008", "report_nm": "b"}]}
    sess.answers.append(newer)
    cl.today = date(2026, 10, 9)
    rows = cl.disclosures("C", "B", 7)
    assert [r["rcept_no"] for r in rows] == ["1", "2"]
    assert sess.calls[-1][1]["bgn_de"] == "20260924"  # from the last fetch - 7 days, not from 2013
    assert "SECRETKEY" == sess.calls[-1][1]["crtfc_key"]


def test_employee_year_finality(dsettings, tmp_path):
    sess = _Session([{"status": "013"}, {"status": "013"}])
    cl = kd.DartAltClient(dsettings, session=sess, root=tmp_path / "d", today=date(2026, 3, 1))
    assert cl.employees("C", 2025) is None  # before the deadline: not yet available, ask again later
    assert cl.employees("C", 2026) is None and len(sess.calls) == 1  # the year has not ended: no call
    cl.today = date(2026, 7, 15)
    assert cl.employees("C", 2025) == []  # after the deadline: final no-data
    assert cl.employees("C", 2025) == [] and len(sess.calls) == 2  # cached as final
