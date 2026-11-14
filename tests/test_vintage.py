"""Frozen first-seen vintages of revised inputs (docs/MODELING.md 84.2) and the traded-value fix (84.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcio.data import vintage as vt
from quantcio.data.pit import PITUniverse
from quantcio.data.pit.us_prices import price_frame
from quantcio.modeling import features as feat
from quantcio.models import PRICE_COLUMNS, Market

TODAY = pd.Timestamp("2026-10-29")  # cutoff 2026-09-30
DAYS = pd.bdate_range("2026-08-03", "2026-10-28")


def _px(scale: float = 1.0, days: pd.DatetimeIndex = DAYS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    vol = rng.integers(1_000, 5_000, len(days)).astype(float)
    df = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": vol}, index=days
    )
    df["value"] = df["close"] * df["volume"]  # the day's own traded value (basis-free)
    for c in ("open", "high", "low", "close"):
        df[c] *= scale
    df["volume"] /= scale
    return df


def _rescale(df: pd.DataFrame, k: float) -> pd.DataFrame:
    out = df.copy()
    for c in ("open", "high", "low", "close"):
        out[c] *= k
    out["volume"] /= k
    return out


def _store(tmp_path, today=TODAY) -> vt.VintageStore:
    return vt.VintageStore(tmp_path / "vintage", today=today)


def _freeze(store, fresh):
    return vt.freeze_frames(
        store, "prices_t", fresh, columns=vt.PRICE_FRAME_COLUMNS, chain=vt.PRICE_LEVELS, inverse=vt.PRICE_INVERSE
    )


def test_cutoff_is_the_last_settled_month_end():
    assert vt.cutoff("2026-10-29") == pd.Timestamp("2026-09-30")
    assert vt.cutoff("2026-10-05") == pd.Timestamp("2026-09-30")  # 5 days after the month-end: frozen
    assert vt.cutoff("2026-10-04") == pd.Timestamp("2026-08-31")
    assert vt.cutoff("2026-03-06", 5) == pd.Timestamp("2026-02-28")
    assert vt.cutoff("2026-11-01", 0) == pd.Timestamp("2026-10-31")


def test_a_rescaled_history_changes_nothing_and_new_days_chain(tmp_path):
    a = _px()
    out, info = _freeze(_store(tmp_path), {"X": a})
    assert info.created and info.cutoff == "2026-09-30" and info.new_keys == 1
    pd.testing.assert_frame_equal(out["X"], a, check_freq=False)
    # the source re-scales the whole history (a later dividend / split) and adds a day
    b = _rescale(a, 0.5)
    b.loc[pd.Timestamp("2026-10-29")] = b.loc["2026-10-28"] * 1.02
    out, info = _freeze(_store(tmp_path), {"X": b})
    x = out["X"]
    assert not info.created and info.revised == 0 and info.added == 0 and info.dropped == 0
    pd.testing.assert_frame_equal(x.loc[:"2026-10-28"], a, check_freq=False, rtol=1e-12)  # same as first seen
    assert x["close"].iloc[-1] / x["close"].iloc[-2] == pytest.approx(1.02)  # the fresh return, chained
    assert x["value"].iloc[-1] == pytest.approx(b["value"].iloc[-1])  # traded value appended as it is
    assert x.attrs["vintage_cutoff"] == "2026-09-30" and x.attrs["vintage_ratio"] == pytest.approx(2.0)


def test_corrections_before_the_cutoff_are_ignored_and_counted(tmp_path):
    a = _px()
    _freeze(_store(tmp_path), {"X": a})
    b = a.copy()
    b.loc["2026-09-15", "close"] *= 1.05  # before the cutoff: ignored
    b.loc["2026-10-15", "close"] *= 1.05  # after it: used
    b = b.drop(pd.Timestamp("2026-08-10"))  # a stored day the source lost: kept
    out, info = _freeze(_store(tmp_path), {"X": b})
    x = out["X"]
    assert x.loc["2026-09-15", "close"] == a.loc["2026-09-15", "close"]
    assert x.loc["2026-10-15", "close"] == pytest.approx(b.loc["2026-10-15", "close"])
    assert pd.Timestamp("2026-08-10") in x.index
    assert info.revised == 1 and info.max_rel == pytest.approx(0.05) and info.dropped == 1


def test_new_keys_are_stored_first_seen_and_lost_keys_come_from_the_store(tmp_path):
    a, c = _px(seed=1), _px(seed=2)
    _freeze(_store(tmp_path), {"A": a})
    out, info = _freeze(_store(tmp_path), {"C": c})
    assert info.new_keys == 1 and info.store_only == 1
    pd.testing.assert_frame_equal(out["A"].loc[:"2026-09-30"], a.loc[:"2026-09-30"], check_freq=False)
    out, info = _freeze(_store(tmp_path), {"A": a, "C": c * 1.1})  # C's history now frozen as first seen
    assert out["C"].loc["2026-09-01", "close"] == pytest.approx(c.loc["2026-09-01", "close"])
    assert info.revised > 0 and info.new_keys == 0


def test_a_new_month_is_stored_as_first_seen(tmp_path):
    a = _px()
    _freeze(_store(tmp_path), {"X": a})
    later = pd.Timestamp("2026-11-06")  # cutoff moves to 2026-10-31
    b = _px(scale=3.0)
    out, info = _freeze(_store(tmp_path, later), {"X": b})
    assert info.extended_from == "2026-09-30" and info.cutoff == "2026-10-31"
    pd.testing.assert_frame_equal(out["X"], a, check_freq=False, rtol=1e-12)
    c = b.copy()
    c.loc["2026-10-20", "close"] *= 0.9  # October is frozen now
    out, info = _freeze(_store(tmp_path, later), {"X": c})
    assert out["X"].loc["2026-10-20", "close"] == pytest.approx(a.loc["2026-10-20", "close"])
    assert info.revised == 1 and not info.extended_from


def test_events_keep_the_stored_dividends(tmp_path):
    s = _store(tmp_path)
    a = pd.Series([0.01, 0.02], index=pd.to_datetime(["2026-03-31", "2026-09-15"]))
    out, info = vt.freeze_events(s, "div", {"A": a}, keys=["A", "B"])
    assert info.created and set(out) == {"A"}
    fresh_a = pd.Series(
        [0.011, 0.005, 0.02, 0.03], index=pd.to_datetime(["2026-03-31", "2026-06-30", "2026-09-15", "2026-10-15"])
    )
    fresh_b = pd.Series([0.04], index=pd.to_datetime(["2026-08-31"]))  # a late-added past dividend
    out, info = vt.freeze_events(_store(tmp_path), "div", {"A": fresh_a, "B": fresh_b}, keys=["A", "B"])
    assert out["A"].to_dict() == {
        pd.Timestamp("2026-03-31"): 0.01,
        pd.Timestamp("2026-09-15"): 0.02,
        pd.Timestamp("2026-10-15"): 0.03,
    }
    assert "B" not in out  # B was stored without events: it stays without them
    assert info.revised == 1 and info.added == 2


def test_events_never_freeze_a_failed_fetch(tmp_path):
    out, info = vt.freeze_events(_store(tmp_path), "div", {}, keys=["C"], new_ok=[])
    assert info.new_keys == 0
    late = pd.Series([0.03], index=pd.to_datetime(["2026-05-29"]))
    out, info = vt.freeze_events(_store(tmp_path), "div", {"C": late}, keys=["C"], new_ok=["C"])
    assert info.new_keys == 1 and out["C"].iloc[0] == 0.03


def test_membership_rows_are_frozen(tmp_path):
    mem = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-31", "2026-08-31", "2026-09-30", "2026-10-28"]),
            "entity": ["A", "B", "A", "A"],
            "ticker": ["a", "b", "a", "a"],
            "name": ["A", "B", "A", "A"],
            "rank": [1, 2, 1, 1],
            "rank_value": [10.0, 5.0, 11.0, 12.0],
        }
    )
    out, info = vt.freeze_rows(_store(tmp_path), "mem", mem, compare=("rank_value",))
    assert info.created
    fresh = mem.copy()
    fresh.loc[0, "rank_value"] = 10.000001  # US market caps move by ~1e-6 between downloads
    fresh = pd.concat([fresh, mem.iloc[[1]].assign(date=pd.to_datetime(["2026-09-30"]))], ignore_index=True)
    fresh.loc[3, "rank_value"] = 13.0  # after the cutoff: used
    out, info = vt.freeze_rows(_store(tmp_path), "mem", fresh, compare=("rank_value",))
    assert out.loc[0, "rank_value"] == 10.0 and len(out) == 4  # B on 09-30 is a fresh row before the cutoff
    assert out["rank_value"].iloc[-1] == 13.0 and out["rank"].dtype == mem["rank"].dtype
    assert info.revised == 1 and info.added == 1


def _panel(rows: list[tuple]) -> pd.DataFrame:
    """[(entity, period_end, available_from, ni)] -> a PIT fundamentals panel (contract columns)."""
    df = pd.DataFrame(rows, columns=["entity", "period_end", "available_from", "ni_parent_ttm"])
    for c in ("period_end", "available_from"):
        df[c] = pd.to_datetime(df[c])
    return df.assign(equity_parent=100.0, period_type="Q", restated_later=False)


def test_fundamentals_panel_keeps_first_seen_rows_and_new_entities(tmp_path):
    first = _panel([("A", "2026-03-31", "2026-04-30", 10.0), ("A", "2026-06-30", "2026-07-30", 11.0)])
    out, info = vt.freeze_panel(_store(tmp_path), "fund", first, compare=("ni_parent_ttm",))
    assert info.created and info.new_keys == 1 and len(out) == 2
    fresh = _panel(
        [
            ("A", "2026-03-31", "2026-04-30", 10.0),
            ("A", "2026-06-30", "2026-07-30", 11.5),  # a changed value before the cutoff: stored one used
            ("A", "2026-06-30", "2026-08-14", 11.2),  # a late-published row before the cutoff: not used
            ("A", "2026-09-30", "2026-10-28", 12.0),  # after the cutoff: used
            ("B", "2026-06-30", "2026-07-29", 5.0),  # an entity seen for the first time: stored as it is
        ]
    )
    out, info = vt.freeze_panel(_store(tmp_path), "fund", fresh, compare=("ni_parent_ttm",))
    a = out[out["entity"] == "A"].set_index("available_from")["ni_parent_ttm"]
    assert a.to_dict() == {
        pd.Timestamp("2026-04-30"): 10.0,
        pd.Timestamp("2026-07-30"): 11.0,
        pd.Timestamp("2026-10-28"): 12.0,
    }
    assert out.loc[out["entity"] == "B", "ni_parent_ttm"].tolist() == [5.0]
    assert info.revised == 1 and info.added == 1 and info.new_keys == 1
    assert out["available_from"].dtype == fresh["available_from"].dtype
    # B's stored row is final too; a fetch failure (entity missing) keeps the stored rows
    out, info = vt.freeze_panel(_store(tmp_path), "fund", fresh[fresh["entity"] == "A"], compare=("ni_parent_ttm",))
    assert info.store_only == 1 and out.loc[out["entity"] == "B", "ni_parent_ttm"].tolist() == [5.0]
    # the cutoff moves to 2026-10-31: October's fresh rows are stored as first seen then
    later = _store(tmp_path, pd.Timestamp("2026-11-06"))
    out, info = vt.freeze_panel(later, "fund", fresh, compare=("ni_parent_ttm",))
    assert info.extended_from == "2026-09-30" and info.cutoff == "2026-10-31"
    revised = fresh.copy()
    revised.loc[3, "ni_parent_ttm"] = 99.0
    out, info = vt.freeze_panel(_store(tmp_path, pd.Timestamp("2026-11-06")), "fund", revised)
    assert out.loc[(out["entity"] == "A") & (out["available_from"] == "2026-10-28"), "ni_parent_ttm"].item() == 12.0


def test_a_damaged_store_is_moved_aside(tmp_path):
    _freeze(_store(tmp_path), {"X": _px()})
    (tmp_path / "vintage" / "prices_t.parquet").write_bytes(b"broken")
    out, info = _freeze(_store(tmp_path), {"X": _px(scale=2.0)})
    assert info.restarted and info.created
    assert any(p.name.startswith("prices_t.parquet.bad-") for p in (tmp_path / "vintage").iterdir())


def test_universe_and_macro_wrappers(tmp_path):
    mem = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-31"]),
            "entity": ["A"],
            "ticker": ["a"],
            "name": ["A"],
            "rank": [1],
            "rank_value": [1.0],
        }
    )
    bench = _px(seed=5)[PRICE_COLUMNS]
    uni = PITUniverse(Market.KR, 1, mem, {"A": _px()}, {"KOSPI": bench}, {"A": "KOSPI"}, pd.DataFrame())
    out, infos = vt.freeze_universe(_store(tmp_path), uni, tag="t")
    assert set(infos) == {"membership", "prices", "benchmarks"} and all(i.created for i in infos.values())
    uni2 = PITUniverse(Market.KR, 1, mem, {"A": _px(scale=0.25)}, {"KOSPI": bench * 2}, {"A": "KOSPI"}, pd.DataFrame())
    out, infos = vt.freeze_universe(_store(tmp_path), uni2, tag="t")
    pd.testing.assert_frame_equal(out.prices["A"], _px(), check_freq=False, rtol=1e-12)
    assert out.benchmarks["KOSPI"]["close"].iloc[0] == pytest.approx(bench["close"].iloc[0])
    hyg = _px(seed=7)["close"].rename("hyg")
    hyg.attrs.update(source="yahoo", kind="price")
    vix = pd.Series(20.0, index=DAYS, name="vix")
    vt.freeze_macro(_store(tmp_path), {"hyg": hyg, "vix": vix}, chain_keys=["hyg"])
    vix2 = vix.copy()
    vix2.loc["2026-09-01"] = 25.0  # a revised level before the cutoff: ignored
    hyg2 = hyg * 0.97
    hyg2.attrs = dict(hyg.attrs)
    m, infos = vt.freeze_macro(_store(tmp_path), {"hyg": hyg2, "vix": vix2}, chain_keys=["hyg"])
    assert np.allclose(m["hyg"].to_numpy(), hyg.to_numpy(), rtol=1e-12) and m["hyg"].attrs["source"] == "yahoo"
    assert m["vix"].loc["2026-09-01"] == 20.0 and infos["macro_level"].revised == 1


def test_digest_tripwire(tmp_path):
    s = _store(tmp_path)
    n = 6
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-30"] * 3 + ["2026-10-28"] * 3)[:n],
            "entity": list("ABCABC"),
            "is_me": [True] * n,
            "labelled": [True, True, True, False, False, False],
            "t_end": pd.to_datetime(["2026-09-25"] * 3 + [None] * 3),
            "x_a": np.arange(n, dtype=float),
            "y_e": np.linspace(-0.1, 0.1, n),
        }
    )
    cut = pd.Timestamp("2026-09-30")
    kw = dict(features=["x_a"], labels=["y_e"], cut=cut)
    assert vt.digest_check(s, "id", frame, **kw)["status"] == "first"
    later = frame.copy()
    later.loc[5, "x_a"] = 99.0  # after the cutoff
    assert vt.digest_check(s, "id", later, **kw)["status"] == "match"
    bad = frame.copy()
    bad.loc[1, "x_a"] = 99.0
    res = vt.digest_check(s, "id", bad, **kw)
    assert res["status"] == "mismatch" and res["mismatch"] == ["2026-09-30:features"]


def test_digest_label_weight_on_cutoff_date(tmp_path):
    """The cutoff month-end's label window ends after the cutoff: its per-date weight (NaN until the label completes)
    is hashed with the labels, so a later build where the label completed still matches (docs/MODELING.md 84.6)."""
    from quantcio.modeling.dataset import LABEL_DIGEST_COLS

    assert "w_date" in LABEL_DIGEST_COLS
    s = _store(tmp_path)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-31"] * 2 + ["2026-09-30"] * 2),
            "entity": list("ABAB"),
            "is_me": [True] * 4,
            "labelled": [True, True, False, False],
            "t_end": pd.to_datetime(["2026-09-29"] * 2 + ["2026-10-30"] * 2),
            "x_a": np.arange(4, dtype=float),
            "y_e": [0.01, -0.02, np.nan, np.nan],
            "w_date": [0.5, 0.5, np.nan, np.nan],
        }
    )
    kw = dict(features=["x_a"], labels=["y_e", "w_date"], cut=pd.Timestamp("2026-09-30"))
    assert vt.digest_check(s, "id", frame, **kw)["status"] == "first"
    done = frame.copy()
    done.loc[2:, ["labelled", "y_e", "w_date"]] = [True, 0.03, 0.5]
    assert vt.digest_check(s, "id", done, **kw)["status"] == "match"
    old = dict(features=["x_a", "w_date"], labels=["y_e"], cut=kw["cut"])  # the weight hashed as a feature: trips
    assert vt.digest_check(s, "id2", frame, **old)["status"] == "first"
    assert vt.digest_check(s, "id2", done, **old)["status"] == "mismatch"


def test_traded_value_ignores_later_dividends_and_splits():
    idx = pd.bdate_range("2026-01-02", periods=60)
    rng = np.random.default_rng(3)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx))))
    vol = rng.integers(10_000, 50_000, len(idx)).astype(float)
    chart = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": vol}, index=idx)
    a = price_frame(chart.assign(adjclose=close))
    b = price_frame(chart.assign(adjclose=close * 0.97))  # a later dividend re-scales the adjusted history
    assert list(a.columns) == [*PRICE_COLUMNS, "value"]
    assert np.allclose(a["value"], close * vol) and np.allclose(b["value"], a["value"])
    assert not np.allclose(b["close"] * b["volume"], a["close"] * a["volume"])  # the old, re-scaled definition
    fa, fb = feat.entity_price_features(a, None), feat.entity_price_features(b, None)
    for c in ("adv20", "amihud", "volume_trend"):
        pd.testing.assert_series_equal(fa[c], fb[c], check_names=False)
