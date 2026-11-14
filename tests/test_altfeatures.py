"""Features v3 (quantcio.modeling.altfeatures): point-in-time windows, coverage, employee / NPS measures, betas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcio.data.alt import AltData, normalise_events, normalise_measures, usable
from quantcio.modeling import altfeatures as af


def _ev(rows):
    return normalise_events(pd.DataFrame(rows, columns=["entity", "published", "kind", "value", "source"]))


def test_usable_is_strictly_before_the_grid_date():
    p = pd.Series(pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-06"]))
    assert usable(p, pd.Timestamp("2024-01-05 15:30")).tolist() == [True, False, False]


def test_event_window_is_point_in_time_and_coverage_decides_nan_vs_zero():
    ev = _ev(
        [
            ("A", "2024-01-10", "buyback", 1.0, "kr_dart_list"),
            ("A", "2024-03-01", "buyback", 1.0, "kr_dart_list"),
            ("A", "2024-03-01", "filing", 1.0, "kr_dart_list"),
            ("A", "2024-02-01", "insider_net_shares", 500.0, "kr_dart_insider"),
        ]
    )
    alt = AltData("KR", ev, covered={"kr_dart_list": ["A", "B"], "kr_dart_insider": ["A"]})
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-03-01", "2024-03-02", "2024-08-01", "2024-03-02", "2024-03-02"]),
            "entity": ["A", "A", "A", "B", "C"],
            "mcap": [1e6, 1e6, 1e6, 1e6, 1e6],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    every = [*af.KR_EVENT_SPECS, *af.KR_FORWARD_ONLY_SPECS]
    specs = [s for s in every if s.name in ("dart_buyback_180", "insider_net_90", "dart_filings_90")]
    out = af.event_features(rows, alt, specs)
    # the 2024-03-01 filing is NOT usable on 2024-03-01 itself, only from the next grid date
    assert out["dart_buyback_180"].tolist()[:3] == [1.0, 2.0, 1.0]  # 2024-08-01: the January one left the window
    assert out["dart_filings_90"].iloc[0] == 0.0 and out["dart_filings_90"].iloc[1] == pytest.approx(np.log(2.0))
    # insider net shares / shares outstanding (mcap / close = 10,000 shares)
    assert out["insider_net_90"].iloc[1] == pytest.approx(500 / 10_000)
    # B: covered without events -> 0; C: not covered -> NaN; B has no insider coverage -> NaN
    assert out.loc[3, "dart_buyback_180"] == 0.0 and np.isnan(out.loc[4, "dart_buyback_180"])
    assert np.isnan(out.loc[3, "insider_net_90"])


def test_minus_kinds_and_mcap_scaling():
    ev = _ev(
        [
            ("X", "2024-01-02", "insider_buy_value", 2e6, "sec_form345"),
            ("X", "2024-01-03", "insider_sell_value", 5e6, "sec_form345"),
            ("X", "2024-01-03", "insider_buy_n", 2.0, "sec_form345"),
            ("X", "2024-01-04", "insider_sell_n", 1.0, "sec_form345"),
        ]
    )
    alt = AltData("US", ev, covered={"sec_form345": ["X"], "sec_forms": ["X"]})
    rows = pd.DataFrame({"date": [pd.Timestamp("2024-02-01")], "entity": ["X"], "mcap": [1e9], "close": [10.0]})
    out = af.event_features(rows, alt, af.US_EVENT_SPECS)
    assert out["insider_buy_180"].iloc[0] == pytest.approx(2e-3)
    assert out["insider_sell_180"].iloc[0] == pytest.approx(5e-3)
    assert out["insider_breadth_90"].iloc[0] == 1.0
    assert out["sec_8k_90"].iloc[0] == 0.0  # covered by sec_forms, no 8-K


def _me(entity, published, period_end, **items):
    return [(entity, published, period_end, k, v, "kr_dart_emp") for k, v in items.items()]


def test_employee_features_yoy_staleness_and_first_version():
    rows_m = [
        *_me("A", "2023-03-20", "2022-12-31", headcount=100, contract_workers=10, avg_tenure_years=5, salary_total=5e9),
        *_me(
            "A", "2024-03-18", "2023-12-31", headcount=120, contract_workers=6, avg_tenure_years=5.5, salary_total=7.2e9
        ),
        # a later correction of FY2023 must not replace the first version
        *_me("A", "2024-06-01", "2023-12-31", headcount=999, contract_workers=0, avg_tenure_years=9, salary_total=1e9),
    ]
    me = normalise_measures(
        pd.DataFrame(rows_m, columns=["entity", "published", "period_end", "item", "value", "source"])
    )
    alt = AltData("KR", measures=me, covered={"kr_dart_emp": ["A"]})
    rows = pd.DataFrame(
        {"date": pd.to_datetime(["2024-03-18", "2024-03-19", "2024-07-01", "2025-08-01"]), "entity": ["A"] * 4}
    )
    out = af.employee_features(rows, alt)
    # 2024-03-18: the FY2023 report is not usable yet -> FY2022 (no YoY partner)
    assert np.isnan(out["emp_growth"].iloc[0]) and out["emp_tenure"].iloc[0] == 5
    assert out["emp_growth"].iloc[1] == pytest.approx(0.2)
    assert out["emp_pay_growth"].iloc[1] == pytest.approx((7.2e9 / 120) / (5e9 / 100) - 1)
    assert out["emp_contract_share"].iloc[1] == pytest.approx(0.05)
    assert out["emp_growth"].iloc[2] == pytest.approx(0.2)  # the correction is ignored
    assert out.iloc[3].isna().all()  # FY2023 ended > 550 days before 2025-08-01: stale


def test_nps_features():
    rows_m = []
    for i, m in enumerate(pd.period_range("2023-01", "2024-03", freq="M")):
        pe = m.to_timestamp(how="end").normalize()
        pub = (m + 1).to_timestamp(how="end").normalize()  # conservative: end of the next month
        rows_m += [
            ("A", pub, pe, "nps_members", 100.0 + 10 * i, "kr_nps"),
            ("A", pub, pe, "nps_new", 5.0, "kr_nps"),
            ("A", pub, pe, "nps_lost", 2.0, "kr_nps"),
        ]
    me = normalise_measures(
        pd.DataFrame(rows_m, columns=["entity", "published", "period_end", "item", "value", "source"])
    )
    alt = AltData("KR", measures=me, covered={"kr_nps": ["A"]})
    rows = pd.DataFrame({"date": [pd.Timestamp("2024-04-01")], "entity": ["A"]})
    out = af.nps_features(rows, alt)
    # usable on 2024-04-01: data months up to 2024-02 (published 2024-03-31); 2024-02 is index 13 -> 230 members
    assert out["nps_growth_3m"].iloc[0] == pytest.approx(230 / 200 - 1)
    assert out["nps_growth_12m"].iloc[0] == pytest.approx(230 / 110 - 1)
    assert out["nps_hire_3m"].iloc[0] == pytest.approx(15 / 220)
    assert out["nps_quit_3m"].iloc[0] == pytest.approx(6 / 220)


def test_macro_betas_recover_a_planted_exposure():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=400)
    m = rng.normal(0, 0.01, len(idx))
    f = rng.normal(0, 0.005, len(idx))
    r = 1.0 * m + 0.8 * f + rng.normal(0, 0.002, len(idx))
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    bench = pd.Series(100 * np.exp(np.cumsum(m)), index=idx)
    fac = pd.DataFrame({c: np.nan for c in af.MACRO_BETA_FEATURES}, index=idx)
    fac["beta_oil"] = f
    out = af.macro_betas(close, bench, fac)
    assert out["beta_oil"].iloc[-1] == pytest.approx(0.8, abs=0.05)
    assert np.isnan(out["beta_oil"].iloc[50]) and out["beta_fx"].isna().all()


def test_factor_changes_use_the_lagged_level():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    wti = pd.Series([70.0, 77.0, 84.7], index=pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"]))
    fc = af.factor_changes({"wti": wti}, "KR", idx, lag_days=1)
    # KR 2024-01-03 sees the US close of 2024-01-02 (77) vs 2024-01-01 (70)
    assert fc["beta_oil"].iloc[1] == pytest.approx(np.log(77 / 70))
    assert fc["beta_oil"].iloc[2] == pytest.approx(np.log(84.7 / 77))
    assert fc["beta_fx"].isna().all()


def test_window_past_the_source_coverage_end_is_nan():
    ev = _ev([("X", "2026-05-02", "insider_buy_value", 1e6, "sec_form345")])
    alt = AltData("US", ev, coverage={"sec_form345": {"covered_through": "2026-06-30"}}, covered={"sec_form345": ["X"]})
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-30", "2026-07-01", "2026-07-02"]),
            "entity": ["X"] * 3,
            "mcap": [1e9] * 3,
            "close": [10.0] * 3,
        }
    )
    spec = [s for s in af.US_EVENT_SPECS if s.name == "insider_buy_180"]
    out = af.event_features(rows, alt, spec)["insider_buy_180"].tolist()
    assert out[0] == pytest.approx(1e-3) and out[1] == pytest.approx(1e-3) and np.isnan(out[2])
