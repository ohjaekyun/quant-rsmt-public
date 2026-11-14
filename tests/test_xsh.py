"""Cross-sectional holdout (quantcio.xsh): candidate names, the one-time evaluation and its verdict (offline).

Synthetic tuning / XSH datasets share a calendar; the XSH names are disjoint from the tuning names. With a PLANTED
signal in both, the frozen selected model (trained on tuning rows only) must be confirmed on the XSH names; with no
signal in the XSH names it must not be.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantcio import holdout, xsh
from quantcio.config import load_settings
from quantcio.data.cache import DiskCache
from quantcio.data.pit import PITUniverse
from quantcio.modeling import events as ev
from quantcio.modeling import selection as sel
from quantcio.modeling.candidates import FACTOR_IC_MAP
from quantcio.modeling.dataset import ModelingDataset, add_date_targets, grid_dates
from quantcio.models import Market

H = 20
XS = ["x_sig", "x_noise", "x_mom_12_1"]


@pytest.fixture
def xs_settings(tmp_path):
    return load_settings(
        data={"cache_dir": str(tmp_path / "cache")},
        macro={"enabled": False},
        holdout={"registry_path": str(tmp_path / "holdout" / "registry.json"), "ledger_dir": str(tmp_path / "led")},
    )


def _dataset(prefix: int, n_names: int, signal: float, seed: int, weekly: bool) -> ModelingDataset:
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2015-01-01", periods=252 * 8 + 40)
    s = pd.Series(cal, index=cal)
    month_ends = pd.DatetimeIndex(s.groupby([cal.year, cal.month]).max().to_numpy())[:-1]
    grid = grid_dates(cal, month_ends, cal[0], weekly=weekly)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    ents = [f"{prefix + i:06d}" for i in range(n_names)]
    rows = []
    for d in grid:
        p = int(pos[d])
        te = cal[p + H] if p + H < len(cal) else pd.NaT
        common = rng.normal(0.0, 0.05)
        x_sig = rng.normal(size=n_names)
        y = signal * x_sig + common + rng.normal(0.0, 0.08, n_names)
        for i, e in enumerate(ents):
            lab = not pd.isna(te)
            rows.append(
                {
                    "date": d,
                    "entity": e,
                    "ticker": e,
                    "is_me": d in month_ends,
                    "t_end": te,
                    "y_e": y[i] if lab else np.nan,
                    "r": y[i] + 0.004 if lab else np.nan,
                    "vol60": 0.3,
                    "x_sig": x_sig[i],
                    "x_noise": rng.normal(),
                    "x_mom_12_1": rng.normal(),
                }
            )
    f = pd.DataFrame(rows)
    f["labelled"] = f["y_e"].notna()
    f = add_date_targets(f)
    # columns the factor candidates read directly (noise here; the signal lives in x_sig)
    extra = [*FACTOR_IC_MAP.values(), "sx_ret_1m", "x_ret_1w", "x_ret_1m", "x_bp", "x_vol60", "x_log_mcap"]
    for c in extra:
        f[c] = rng.normal(size=len(f))
    dates = pd.DatetimeIndex(sorted(f["date"].unique()))
    reg = pd.DataFrame({"r_mkt_dd": rng.uniform(-0.3, 0.0, len(dates))}, index=dates)
    f = f.merge(reg, left_on="date", right_index=True)
    dfr = reg.copy()
    dfr["t_end"] = f.groupby("date")["t_end"].first().reindex(dates)
    dfr["is_me"] = dates.isin(month_ends)
    return ModelingDataset(
        market=Market.KR,
        horizon=H,
        frame=f.sort_values(["date", "entity"]).reset_index(drop=True),
        xs_cols=[*XS, *dict.fromkeys(c for c in extra if c not in XS)],
        regime_cols=["r_mkt_dd"],
        as_of=pd.Timestamp(grid[-1]),
        calendar=cal,
        month_ends=month_ends,
        events=ev.build_calendar(Market.KR, {}),
        dates_frame=dfr,
        snapshot=f"xsh-test-{prefix}-{seed}-{signal}",
        cost_bps=30.0,
    )


def _selection(settings, selected: str = "logit") -> dict:
    return {
        "selected": selected,
        "created_at": "2026-09-27T00:00:00",
        "as_of": "2023-01-31",
        "protocol": {"fingerprint": sel._fingerprint()},
        "settings_digest": settings.digest(),
        "candidates": [{"name": selected, "outer": {"auc_median": {"mean": 0.6}}, "holdout": {}}],
    }


def _entry(settings, xds: ModelingDataset, drop: int = 3) -> dict:
    names = sorted(xds.frame["entity"].unique())[drop:]  # a few band names are "tuned elsewhere": not registered
    return {
        "id": "XSH-KR-TEST",
        "kind": "cross_sectional",
        "market": "KR",
        "names": names,
        "period": ["2015-01-01", "2030-01-01"],
        "primary_metrics": holdout.xsh_primary_metrics_block(settings),
    }


@pytest.fixture(scope="module")
def tuning_ds():
    return _dataset(100000, 40, 0.03, 0, weekly=True)


def test_planted_signal_is_confirmed_on_never_tuned_names(xs_settings, tuning_ds, tmp_path):
    xds = _dataset(500000, 40, 0.03, 1, weekly=False)
    entry = _entry(xs_settings, xds)
    out = xsh.run_xsh_evaluation(
        entry,
        xs_settings,
        DiskCache(xs_settings.cache_path),
        threads=2,
        out_dir=tmp_path / "out",
        selection=_selection(xs_settings),
        tuning_ds=tuning_ds,
        xsh_ds=xds,
    )
    h = out["headline"]
    assert h["confirmed"] is True and h["auc_median"] > 0.6 and h["auc_median_lo"] > 0.5
    assert h["n_names"] == 37 and h["n_dates"] >= 40 and h["bh_reject"][holdout.XSH_METRIC_AUC]
    assert h["target_met"] is (h["auc_median"] >= 0.55) and h["vs_momentum"] > 0.05
    res = out["result"]
    # every scored date belongs to a context whose training ended >= 2h sessions before it (purge + embargo)
    ctxs = {c["name"]: pd.Timestamp(c["train_end"]) for c in res["contexts"]}
    assert set(ctxs) >= {"holdout"} and res["coverage"]["dates_without_context"] > 0  # the first 36 months
    for c in res["by_context"]:
        assert pd.Timestamp(c["first"]) > ctxs[c["context"]]
    assert res["sensitivity_all_band_rows"]["n_names"] == 40  # includes the unregistered names
    js = json.loads(open(out["result_path"], encoding="utf-8").read())
    assert js["headline"]["confirmed"] is True
    md = open(out["md_path"], encoding="utf-8").read()
    assert "횡단면 홀드아웃(XSH) 평가" in md and "민감도" in md and "사후" in md


def test_no_signal_in_the_holdout_names_is_not_confirmed(xs_settings, tuning_ds, tmp_path):
    xds = _dataset(500000, 40, 0.0, 2, weekly=False)
    xds.coverage = {"pit_from": "2021-01-01"}  # e.g. US mid caps: membership point-in-time only from here
    out = xsh.run_xsh_evaluation(
        _entry(xs_settings, xds),
        xs_settings,
        DiskCache(xs_settings.cache_path),
        threads=2,
        out_dir=tmp_path,
        selection=_selection(xs_settings),
        tuning_ds=tuning_ds,
        xsh_ds=xds,
    )
    h = out["headline"]
    assert h["confirmed"] is False and h["target_met"] is False and abs(h["auc_median"] - 0.5) < 0.04
    cov = out["result"]["coverage"]
    assert cov["pit_from"] == "2021-01-01" and cov["dates_before_pit_from"] > 0
    assert min(pd.Timestamp(c["first"]) for c in out["result"]["by_context"]) >= pd.Timestamp("2021-01-01")
    assert "생존 편향이 있어 평가에서 제외" in open(out["md_path"], encoding="utf-8").read()


@pytest.mark.parametrize("selected", ["blend", "ensemble"])
def test_composite_selections_score_their_members(xs_settings, tuning_ds, tmp_path, selected):
    xds = _dataset(500000, 30, 0.03, 4, weekly=False)
    out = xsh.run_xsh_evaluation(
        _entry(xs_settings, xds),
        xs_settings,
        DiskCache(xs_settings.cache_path),
        threads=2,
        out_dir=tmp_path,
        selection=_selection(xs_settings, selected),
        tuning_ds=tuning_ds,
        xsh_ds=xds,
    )
    members = {tuple(c["members"]) for c in out["result"]["contexts"]}
    if selected == "blend":
        assert members == {("logit", "lgbm_clf")}
    else:
        assert all(len(m) == 3 for m in members)
    assert out["headline"]["auc_median"] > 0.55


def test_refuses_stale_selection_and_overlapping_names(xs_settings, tuning_ds, tmp_path):
    xds = _dataset(500000, 25, 0.03, 3, weekly=False)
    cache = DiskCache(xs_settings.cache_path)
    stale = {**_selection(xs_settings), "protocol": {"fingerprint": "0" * 12}}
    kw = {"threads": 2, "out_dir": tmp_path, "tuning_ds": tuning_ds, "xsh_ds": xds}
    with pytest.raises(xsh.XSHError, match="코드 지문"):
        xsh.run_xsh_evaluation(_entry(xs_settings, xds), xs_settings, cache, selection=stale, **kw)
    other = {**_selection(xs_settings), "settings_digest": "zzz"}
    with pytest.raises(xsh.XSHError, match="설정 다이제스트"):
        xsh.run_xsh_evaluation(_entry(xs_settings, xds), xs_settings, cache, selection=other, **kw)
    with pytest.raises(xsh.XSHError, match="선택 결과가 없습니다"):
        xsh.run_xsh_evaluation(_entry(xs_settings, xds), xs_settings, cache, **kw)
    bad = {**_entry(xs_settings, xds), "names": ["100001", "500004"]}  # 100001 is a tuning name
    with pytest.raises(xsh.XSHError, match="튜닝 데이터셋"):
        xsh.run_xsh_evaluation(bad, xs_settings, cache, selection=_selection(xs_settings), **kw)
    assert not list(tmp_path.glob("xsh_*"))  # nothing written, nothing to look at


def test_verdict_rules():
    s = load_settings()
    idx = pd.date_range("2019-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(0)

    def m(auc, mom, ic):
        from quantcio.modeling import metrics as mt

        a = pd.Series(auc, index=idx)
        return {
            "model": {"auc_median": mt.summarize(a, 0.5), "ic": {"mean": float(np.mean(ic))}},
            "_series": {"auc_median": a, "auc_median_mom": pd.Series(mom, index=idx), "ic": pd.Series(ic, index=idx)},
        }

    good = xsh.verdict(
        m(0.56 + rng.normal(0, 0.03, 60), 0.5 + rng.normal(0, 0.03, 60), 0.1 + rng.normal(0, 0.1, 60)), s
    )
    assert good["confirmed"] and good["target_met"] and all(good["bh_reject"].values())
    weak = xsh.verdict(m(0.5 + rng.normal(0, 0.03, 60), 0.5 + rng.normal(0, 0.03, 60), rng.normal(0, 0.1, 60)), s)
    assert not weak["confirmed"] and not weak["target_met"]
    # BH at q = 0.10 can reject one-sided p ~ 0.09 (seen on XSH-US-20260927): without a 95% lower bound > 0.5 the
    # result is weak evidence and never "confirmed"
    borderline = m(0.5082 + rng.normal(0, 0.06, 60), 0.4935 + rng.normal(0, 0.06, 60), 0.02 + rng.normal(0, 0.1, 60))
    b = xsh.verdict(borderline, s)
    assert borderline["model"]["auc_median"]["lo"] <= 0.5 and not b["confirmed"]
    few = xsh.verdict(
        {
            "model": {"auc_median": {"mean": 0.6}, "ic": {"mean": 0.1}},
            "_series": {k: pd.Series([0.6] * 5, index=idx[:5]) for k in ("auc_median", "auc_median_mom", "ic")},
        },
        s,
    )
    assert not few["confirmed"] and "판정 불가" in few["text"]


def _pit(ranks: dict[str, list[int]]) -> PITUniverse:
    dates = pd.to_datetime(["2020-01-31", "2020-02-28"])
    mem = pd.DataFrame(
        [(d, e, e, f"n{e}", r[i], 1.0) for e, r in ranks.items() for i, d in enumerate(dates)],
        columns=["date", "entity", "ticker", "name", "rank", "rank_value"],
    )
    px = pd.DataFrame({"close": [1.0, 1.0]}, index=dates)
    meta = pd.DataFrame({"entity": list(ranks), "delisted": False})
    return PITUniverse(Market.KR, 400, mem, {e: px for e in ranks}, {"KOSPI": px}, {e: "KOSPI" for e in ranks}, meta)


def test_band_universe_and_candidate_names(xs_settings, monkeypatch):
    full = _pit({"000001": [150, 180], "000002": [210, 199], "000003": [250, 300], "000004#1": [390, 420]})
    band = xsh.band_universe(full, 201, 400)
    assert set(band.membership["entity"]) == {"000002", "000003", "000004#1"} and len(band.membership) == 4
    assert set(band.prices) == {"000002", "000003", "000004#1"} and set(band.meta["entity"]) == set(band.prices)
    assert band.coverage["xsh_band"] == [201, 400] and "XSH 순위 구간" in band.notes[0]

    # 000002 was in the tuned top 200 on 2020-02-28 (a tuning name); 000004's code is tuned via another ticker
    monkeypatch.setattr(holdout, "tuning_names", lambda m, s, c: ["000001", "000002", "000004"])
    monkeypatch.setattr(xsh, "xsh_universe", lambda m, s, c, exclude=(): band)
    assert xsh.xsh_candidate_names("KR", xs_settings, None) == ["000003"]
    monkeypatch.setattr(holdout, "tuning_names", lambda m, s, c: ["000002", "000003", "000004"])
    with pytest.raises(holdout.HoldoutError, match="후보가 없습니다"):
        xsh.xsh_candidate_names("KR", xs_settings, None)


def test_rank_band_must_not_overlap_the_tuning_universe(tmp_path):
    s = load_settings(data={"cache_dir": str(tmp_path)}, holdout={"xsh_kr_rank_band": [150, 400]})
    with pytest.raises(xsh.XSHError, match="겹칩니다"):
        xsh.xsh_universe("KR", s, DiskCache(s.cache_path))


def test_cli_hooks_exist():
    from quantcio import evaluation

    assert callable(evaluation.xsh_candidate_names) and callable(evaluation.run_xsh_evaluation)
    assert "xsh.py" in holdout.FINGERPRINT_EXCLUDED_FILES


def test_modeling_select_drops_registered_xsh_names(xs_settings, monkeypatch):
    from quantcio.modeling.cli import guard_xsh

    monkeypatch.setattr(holdout, "git_state", lambda cwd=None: holdout.GitState("abc", False))
    monkeypatch.setattr(holdout, "code_fingerprint", lambda root=None: "a" * 16)
    ds = _dataset(100000, 20, 0.0, 5, weekly=False)
    same = guard_xsh(ds, xs_settings)
    assert same is ds and same.snapshot.endswith("-0.0")  # nothing registered: untouched
    holdout.register_cross_sectional(xs_settings, "KR", ["100003"], ["2016-01-01", "2026-01-01"], [])
    n_rows = int((ds.frame["entity"] == "100003").sum())
    out = guard_xsh(ds, xs_settings)
    assert "100003" not in set(out.frame["entity"]) and out.snapshot.endswith(f"|xsh-{n_rows}")
    assert any("XSH" in n["ko"] for n in out.notes)
