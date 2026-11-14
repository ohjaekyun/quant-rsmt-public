"""Tests for quant/factors.py (synthetic, offline)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from conftest import make_market_data, make_prices

from quantcio.models import PRICE_COLUMNS, FactorScore, Fundamentals, MarketData, Technicals
from quantcio.quant.factors import (
    FACTOR_LABELS,
    FACTOR_ORDER,
    MOM_12_1,
    MOM_52W_HIGH,
    TREND_RSI,
    VAL_PER_REL_SECTOR,
    VOL_ABS,
    VOL_BETA_DEV,
    VOL_MDD,
    Logistic,
    PiecewiseLinear,
    _ro,
    compute_factors,
    factor_composite,
    factor_pillar,
    momentum_subscores,
    own_percentile,
    trend_subscores,
)
from quantcio.quant.indicators import compute_technicals

BASE_TECH = Technicals(
    last_close=100.0,
    atr=2.0,
    atr_pct=0.02,
    rsi14=55.0,
    ma50=95.0,
    ma200=90.0,
    high_52w=110.0,
    low_52w=80.0,
    realized_vol_60d=0.25,
    beta_1y=1.0,
    max_drawdown_1y=-0.20,
    ret_1m=0.02,
    ret_3m=0.05,
    ret_12m_ex1m=0.15,
    rel_strength_3m=0.02,
    support=95.0,
    resistance=110.0,
)


def _by_name(factors: list[FactorScore]) -> dict[str, FactorScore]:
    return {f.name: f for f in factors}


def _score(md: MarketData, tech: Technicals, settings, name: str) -> float | None:
    return _by_name(compute_factors(md, tech, settings))[name].score


def _non_decreasing(xs: list[float]) -> bool:
    return all(b >= a - 1e-12 for a, b in zip(xs, xs[1:]))


# ------------------------------------------------------------------------ mappings


def test_anchor_points_match_spec():
    assert MOM_52W_HIGH(0.0) == pytest.approx(80.0)
    assert MOM_52W_HIGH(-0.30) == pytest.approx(20.0)
    assert VAL_PER_REL_SECTOR(math.log(1.0)) == pytest.approx(50.0)
    assert VAL_PER_REL_SECTOR(math.log(0.6)) == pytest.approx(80.0)
    assert VAL_PER_REL_SECTOR(math.log(1.5)) == pytest.approx(25.0, abs=0.5)
    assert VOL_ABS(0.15) == pytest.approx(85.0) and VOL_ABS(0.60) == pytest.approx(15.0)
    assert VOL_MDD(-0.10) == pytest.approx(80.0) and VOL_MDD(-0.50) == pytest.approx(15.0)
    assert MOM_12_1(0.0) == pytest.approx(50.0)


@pytest.mark.parametrize("fn", [MOM_12_1, MOM_52W_HIGH, VAL_PER_REL_SECTOR, VOL_ABS, VOL_BETA_DEV, VOL_MDD])
def test_logistic_bounded_monotone_and_nan_safe(fn: Logistic):
    xs = np.concatenate([[-1e9, -1e3], np.linspace(-5, 5, 2001), [1e3, 1e9]])
    ys = [fn(x) for x in xs]
    assert all(0.0 <= y <= 100.0 for y in ys)
    assert _non_decreasing(ys if fn.s > 0 else ys[::-1])
    assert fn(None) is None and fn(float("nan")) is None and fn(float("inf")) is None


def test_logistic_rejects_zero_scale():
    with pytest.raises(ValueError):
        Logistic(c=0.0, s=0.0)


def test_rsi_map_is_unimodal_with_asymmetric_penalties():
    grid = np.linspace(0, 100, 1001)
    ys = [TREND_RSI(x) for x in grid]
    assert all(0 <= y <= 100 for y in ys)
    up, flat, down = grid <= 50, (grid >= 50) & (grid <= 65), grid >= 65
    assert _non_decreasing([y for y, m in zip(ys, up) if m])
    assert max(ys) == TREND_RSI(57.5) == TREND_RSI(50) == TREND_RSI(65)
    assert len({round(y, 9) for y, m in zip(ys, flat) if m}) == 1
    assert _non_decreasing([y for y, m in zip(ys, down) if m][::-1])
    # overbought (>75) penalised harder than oversold (<30) at the same distance from the 50..65 band
    assert TREND_RSI(85) < TREND_RSI(30) and TREND_RSI(95) < TREND_RSI(20)
    assert TREND_RSI(None) is None
    assert PiecewiseLinear(((0, 0), (1, 100)))(5) == 100.0


def test_own_percentile_mid_rank():
    hist = [1.0, 2.0, 3.0, 4.0]
    assert own_percentile(2.5, hist) == 0.5
    assert own_percentile(2.0, hist) == pytest.approx(1.5 / 4)
    assert own_percentile(0.0, hist) == 0.0 and own_percentile(9.0, hist) == 1.0
    assert own_percentile(None, hist) is None and own_percentile(1.0, []) is None
    assert own_percentile(2.5, [1.0, np.nan, 4.0]) == 0.5


def test_particle():
    assert _ro("RSI 73") == "으로" and _ro("RSI 72") == "로" and _ro("베타 1.30") == "으로"
    assert _ro("+34%") == "로" and _ro("0.85배") == "로" and _ro("위(+3%)") == "로" and _ro("+5%p") == "로"


# --------------------------------------------------------------------- end to end


def test_all_factors_order_labels_bounds_and_determinism(md_kr, settings):
    tech = compute_technicals(md_kr.prices, md_kr.benchmark, settings.risk.atr_period)
    factors = compute_factors(md_kr, tech, settings)
    assert [f.name for f in factors] == list(FACTOR_ORDER)
    assert [f.label for f in factors] == ["모멘텀", "밸류에이션", "변동성(안정성)", "퀄리티", "추세"]
    assert [f.label for f in factors] == [FACTOR_LABELS[n] for n in FACTOR_ORDER]
    for f in factors:
        assert f.score is not None and 0.0 <= f.score <= 100.0
        assert f.rationale and "로" in f.rationale
        assert all(v is None or math.isfinite(v) for v in f.metrics.values())
    assert compute_factors(md_kr, tech, settings) == factors  # deterministic
    by = _by_name(factors)
    assert set(by["momentum"].metrics) == {"ret_12m_ex1m", "ret_3m", "rel_strength_3m", "dist_52w_high"}
    assert set(by["value"].metrics) == {"per", "pbr", "per_pct_own", "pbr_pct_own", "per_rel_sector", "analyst_upside"}
    assert by["value"].metrics["per_rel_sector"] == pytest.approx(13.5 / 15.0)
    assert by["value"].metrics["analyst_upside"] == pytest.approx(0.2)


def test_rationale_matches_spec_example(md_kr, settings):
    tech = BASE_TECH.model_copy(
        update=dict(
            ret_12m_ex1m=0.34,
            ret_3m=0.15,
            rel_strength_3m=0.10,
            last_close=97.0,
            high_52w=100.0,
        )
    )
    mom = _by_name(compute_factors(md_kr, tech, settings))["momentum"]
    assert mom.rationale == "12-1M 수익률 +34%로 강세, 52주 고점 대비 -3%"


@pytest.mark.parametrize(
    "rsi_value, expected",
    [
        (65.3, "RSI 65로 적정"),  # regression: 65.3 > 65 used to read "RSI 65로 과열 경계"
        (49.6, "RSI 50으로 적정"),  # regression: used to read "RSI 50으로 중립"
        (75.2, "RSI 75로 과열 경계"),  # regression: used to read "RSI 75로 과매수" (spec: > 75)
        (75.6, "RSI 76으로 과매수"),
        (29.6, "RSI 30으로 중립"),
        (29.4, "RSI 29로 과매도"),
    ],
)
def test_rsi_tone_follows_the_printed_value(md_kr, rsi_value, expected):
    sub = trend_subscores(md_kr, BASE_TECH.model_copy(update={"rsi14": rsi_value}))[-1]
    assert sub.key == "rsi14" and f"{sub.text}{_ro(sub.text)} {sub.tone}" == expected
    assert sub.score == pytest.approx(TREND_RSI(rsi_value))  # the score still uses the raw value


def test_low_beta_is_not_called_unstable(md_kr, settings):
    calm = dict(realized_vol_60d=0.12, max_drawdown_1y=-0.05)
    low = _by_name(compute_factors(md_kr, BASE_TECH.model_copy(update={**calm, "beta_1y": 0.10}), settings))
    assert "베타 0.10으로 시장 연동 낮음" in low["volatility"].rationale
    assert "불안정" not in low["volatility"].rationale
    high = _by_name(compute_factors(md_kr, BASE_TECH.model_copy(update={**calm, "beta_1y": 2.0}), settings))
    assert "베타 2.00으로 불안정" in high["volatility"].rationale
    # the penalty itself is unchanged and symmetric (spec: |beta - 0.8|)
    assert low["volatility"].score == pytest.approx(
        _by_name(compute_factors(md_kr, BASE_TECH.model_copy(update={**calm, "beta_1y": 1.5}), settings))[
            "volatility"
        ].score
    )


def test_quality_logistic_centers_and_scales():
    from quantcio.quant.factors import QUAL_EARN_G, QUAL_OPM, QUAL_REV_G, QUAL_ROE

    one_s = 100.0 / (1.0 + math.exp(-1.0))
    for fn, c, s in [
        (QUAL_ROE, 0.10, 0.06),
        (QUAL_OPM, 0.10, 0.07),
        (QUAL_EARN_G, 0.0, 0.20),
        (QUAL_REV_G, 0.05, 0.10),
    ]:
        assert fn(c) == pytest.approx(50.0) and fn(c + s) == pytest.approx(one_s)
        assert fn(c - s) == pytest.approx(100.0 - one_s)


def test_value_percentile_direction_matches_independent_rank(md_kr, settings):
    tech = compute_technicals(md_kr.prices, md_kr.benchmark)
    hist = md_kr.valuation_history["per"].to_numpy()
    for per in (9.2, 12.0, 17.5):
        md_kr.fundamentals = md_kr.fundamentals.model_copy(update={"per": per})
        value = _by_name(compute_factors(md_kr, tech, settings))["value"]
        expected = ((hist < per).sum() + 0.5 * (hist == per).sum()) / hist.size
        assert value.metrics["per_pct_own"] == pytest.approx(expected)
    cheap = md_kr.fundamentals.model_copy(update={"per": 5.0, "pbr": 0.5})
    dear = md_kr.fundamentals.model_copy(update={"per": 40.0, "pbr": 3.0})
    md_kr.fundamentals = cheap
    s_cheap = _by_name(compute_factors(md_kr, tech, settings))["value"]
    md_kr.fundamentals = dear
    s_dear = _by_name(compute_factors(md_kr, tech, settings))["value"]
    assert s_cheap.metrics["per_pct_own"] == 0.0 and s_dear.metrics["per_pct_own"] == 1.0
    assert s_cheap.score > 50 > s_dear.score


def test_flat_prices_do_not_score_as_overbought(settings):
    md = make_market_data()
    c = np.full(400, 50_000.0)
    md.prices = pd.DataFrame(
        {"open": c, "high": c, "low": c, "close": c, "volume": np.ones(400)},
        index=pd.bdate_range(end="2026-09-23", periods=400),
    )
    tech = compute_technicals(md.prices, md.benchmark)
    trend = _by_name(compute_factors(md, tech, settings))["trend"]
    assert trend.metrics["rsi14"] is None and "과매수" not in trend.rationale
    assert trend.score == pytest.approx(50.0) and "4/5 지표 반영" in trend.rationale


def test_rationale_mixed_tones_and_partial_coverage(md_kr, settings):
    tech = BASE_TECH.model_copy(update=dict(ret_12m_ex1m=0.40, ret_3m=None, rel_strength_3m=-0.20))
    mom = _by_name(compute_factors(md_kr, tech, settings))["momentum"]
    assert mom.rationale.startswith("12-1M 수익률 +40%로 강세, 벤치마크 대비 3개월 -20%p로 약세")
    assert "(3/4 지표 반영)" in mom.rationale


# ------------------------------------------------------------------ monotonicity


@pytest.mark.parametrize("field", ["ret_12m_ex1m", "ret_3m", "rel_strength_3m", "last_close"])
def test_momentum_non_decreasing_in_each_input(md_kr, settings, field):
    grid = np.linspace(-0.6, 0.8, 29) if field != "last_close" else np.linspace(40.0, 110.0, 29)
    scores = [_score(md_kr, BASE_TECH.model_copy(update={field: float(x)}), settings, "momentum") for x in grid]
    assert _non_decreasing(scores) and scores[-1] > scores[0]


def test_momentum_non_decreasing_when_all_inputs_rise():
    rng = np.random.default_rng(0)
    for _ in range(200):
        base = rng.normal(0, 0.3, 3)
        bump = np.abs(rng.normal(0, 0.1, 3))
        lo = BASE_TECH.model_copy(update=dict(ret_12m_ex1m=base[0], ret_3m=base[1], rel_strength_3m=base[2]))
        hi = lo.model_copy(
            update=dict(ret_12m_ex1m=base[0] + bump[0], ret_3m=base[1] + bump[1], rel_strength_3m=base[2] + bump[2])
        )
        s_lo = np.mean([s.score for s in momentum_subscores(lo)])
        s_hi = np.mean([s.score for s in momentum_subscores(hi)])
        assert s_hi >= s_lo - 1e-12


def test_value_monotone(md_kr, settings):
    tech = compute_technicals(md_kr.prices, md_kr.benchmark)

    def val(**fund_update):
        md = make_market_data()
        md.fundamentals = md_kr.fundamentals.model_copy(update=fund_update)
        return _score(md, tech, settings, "value")

    pers = [val(per=p) for p in np.linspace(5, 40, 15)]
    assert _non_decreasing(pers[::-1]) and pers[0] > pers[-1]  # higher PER -> cheaper score falls
    pbrs = [val(pbr=p) for p in np.linspace(0.5, 3.0, 15)]
    assert _non_decreasing(pbrs[::-1])
    targets = [val(analyst_target_price=tech.last_close * k) for k in np.linspace(0.6, 1.8, 15)]
    assert _non_decreasing(targets) and targets[-1] > targets[0]


def test_volatility_and_quality_monotone(md_kr, settings):
    vols = [
        _score(md_kr, BASE_TECH.model_copy(update={"realized_vol_60d": v}), settings, "volatility")
        for v in np.linspace(0.05, 1.2, 20)
    ]
    assert _non_decreasing(vols[::-1]) and vols[0] > vols[-1]
    mdds = [
        _score(md_kr, BASE_TECH.model_copy(update={"max_drawdown_1y": d}), settings, "volatility")
        for d in np.linspace(-0.9, 0.0, 20)
    ]
    assert _non_decreasing(mdds)
    devs = [
        _score(md_kr, BASE_TECH.model_copy(update={"beta_1y": 0.8 + d}), settings, "volatility")
        for d in np.linspace(0.0, 2.0, 20)
    ]
    neg = [
        _score(md_kr, BASE_TECH.model_copy(update={"beta_1y": 0.8 - d}), settings, "volatility")
        for d in np.linspace(0.0, 2.0, 20)
    ]
    assert _non_decreasing(devs[::-1]) and devs == pytest.approx(neg)  # symmetric |beta - 0.8| penalty

    tech = BASE_TECH
    for field in ("roe", "operating_margin", "earnings_growth_yoy", "revenue_growth_yoy"):
        scores = []
        for x in np.linspace(-0.5, 0.8, 14):
            md = make_market_data()
            md.fundamentals = md.fundamentals.model_copy(update={field: float(x)})
            scores.append(_score(md, tech, settings, "quality"))
        assert _non_decreasing(scores) and scores[-1] > scores[0], field


def test_trend_monotone_in_ma_distance(md_kr, settings):
    scores = [
        _score(md_kr, BASE_TECH.model_copy(update={"last_close": c}), settings, "trend")
        for c in np.linspace(60, 140, 20)
    ]
    assert _non_decreasing(scores)


# ------------------------------------------------------------------- missing data


def test_missing_fundamentals_is_explicit(settings):
    md = make_market_data()
    md.fundamentals = Fundamentals(source="unavailable")
    md.valuation_history = None
    tech = compute_technicals(md.prices, md.benchmark)
    notes: list[str] = []
    by = _by_name(compute_factors(md, tech, settings, warnings=notes))
    assert by["value"].score is None and by["quality"].score is None
    assert by["value"].rationale.startswith("산출 불가") and "PER 미제공/적자" in by["value"].rationale
    assert by["quality"].rationale.startswith("산출 불가")
    assert all(v is None for v in by["quality"].metrics.values())
    assert by["momentum"].score is not None and by["volatility"].score is not None and by["trend"].score is not None
    assert any(n.startswith("밸류에이션 팩터 산출 불가: ") for n in notes)
    assert any(n.startswith("퀄리티 팩터 산출 불가: ") for n in notes)
    assert not any(n.count("산출 불가") > 1 for n in notes)
    compute_factors(md, tech, settings, warnings=notes)
    assert len(notes) == len(set(notes))  # deduplicated


@pytest.mark.parametrize("per", [-8.0, 0.0, None, float("nan")])
def test_loss_making_skips_per_metrics(md_kr, settings, per):
    md_kr.fundamentals = md_kr.fundamentals.model_copy(update={"per": per})
    tech = compute_technicals(md_kr.prices, md_kr.benchmark)
    value = _by_name(compute_factors(md_kr, tech, settings))["value"]
    assert value.metrics["per_pct_own"] is None and value.metrics["per_rel_sector"] is None
    assert value.metrics["pbr_pct_own"] is not None and value.metrics["analyst_upside"] is not None
    assert value.score is not None and 0 <= value.score <= 100
    assert "PER 지표 제외" in value.rationale
    assert ("적자(PER≤0)" in value.rationale) == (per is not None and per == per and per <= 0)


def test_valuation_history_rules(md_kr, settings):
    tech = compute_technicals(md_kr.prices, md_kr.benchmark)
    as_of = md_kr.prices.index[-1]
    hist = md_kr.valuation_history
    base = _by_name(compute_factors(md_kr, tech, settings))["value"]

    # future rows (after t) with extreme values are ignored
    future_idx = pd.bdate_range(as_of + pd.Timedelta(1, "D"), periods=50)
    md_kr.valuation_history = pd.concat([hist, pd.DataFrame({"per": 1000.0, "pbr": 1000.0}, index=future_idx)])
    assert _by_name(compute_factors(md_kr, tech, settings))["value"] == base

    # < 1 year of history -> percentile metrics skipped and noted
    md_kr.valuation_history = hist.loc[hist.index > as_of - pd.Timedelta(200, "D")]
    short = _by_name(compute_factors(md_kr, tech, settings))["value"]
    assert short.metrics["per_pct_own"] is None and short.metrics["pbr_pct_own"] is None
    assert "1년 미만" in short.rationale and short.score is not None

    # non-positive / NaN history values are excluded, string dates are accepted
    messy = hist.copy()
    messy.iloc[::3, 0] = -5.0
    messy.iloc[1::3, 1] = np.nan
    messy.index = messy.index.strftime("%Y-%m-%d")
    md_kr.valuation_history = messy
    v = _by_name(compute_factors(md_kr, tech, settings))["value"]
    assert v.metrics["per_pct_own"] is not None and 0 <= v.metrics["per_pct_own"] <= 1


def test_short_history_and_empty_benchmark(settings):
    md = make_market_data()
    md.prices = make_prices(n=150, seed=3)
    md.benchmark = pd.DataFrame(columns=PRICE_COLUMNS, dtype="float64")
    tech = compute_technicals(md.prices, md.benchmark)
    by = _by_name(compute_factors(md, tech, settings))
    assert by["momentum"].metrics["ret_12m_ex1m"] is None and by["momentum"].metrics["rel_strength_3m"] is None
    assert by["volatility"].metrics["vol_pct_3y"] is None and by["volatility"].metrics["beta_1y"] is None
    assert by["trend"].metrics["close_vs_ma200"] is None and by["trend"].metrics["ma200_slope_20d"] is None
    for f in by.values():
        assert f.score is not None and 0 <= f.score <= 100


def test_nan_technicals_are_treated_as_missing(md_kr, settings):
    tech = BASE_TECH.model_copy(update=dict(ret_12m_ex1m=float("nan"), rsi14=float("nan"), beta_1y=float("inf")))
    by = _by_name(compute_factors(md_kr, tech, settings))
    assert by["momentum"].metrics["ret_12m_ex1m"] is None and by["trend"].metrics["rsi14"] is None
    assert by["volatility"].metrics["beta_1y"] is None
    assert all(f.score is not None and 0 <= f.score <= 100 for f in by.values())


# -------------------------------------------------------------------- look-ahead


def test_look_ahead_invariance(settings):
    full = make_market_data()
    t_cut = full.prices.index[1400]
    as_of_md = make_market_data()
    as_of_md.prices = full.prices.loc[:t_cut]
    as_of_md.benchmark = full.benchmark.loc[:t_cut]
    as_of_md.valuation_history = full.valuation_history.loc[:t_cut]
    tech = compute_technicals(as_of_md.prices, as_of_md.benchmark)
    expected = compute_factors(as_of_md, tech, settings)

    leaky = make_market_data()
    leaky.prices = full.prices.loc[:t_cut]
    bench = full.benchmark.copy()
    bench.loc[bench.index > t_cut, "close"] *= 5.0  # wild future benchmark
    leaky.benchmark = bench
    vh = full.valuation_history.copy()
    vh.loc[vh.index > t_cut] = 0.01  # wild future valuation
    leaky.valuation_history = vh
    tech_leaky = compute_technicals(leaky.prices, leaky.benchmark)
    assert tech_leaky == tech
    assert compute_factors(leaky, tech_leaky, settings) == expected


# ---------------------------------------------------------------- composite / pillar


def _fs(name: str, score: float | None) -> FactorScore:
    return FactorScore(name=name, label=FACTOR_LABELS[name], score=score)


def test_factor_composite_renormalises(settings):
    weights = {"momentum": 0.30, "value": 0.25, "volatility": 0.15, "quality": 0.15, "trend": 0.15}
    factors = [
        _fs("momentum", 80.0),
        _fs("value", None),
        _fs("volatility", 50.0),
        _fs("quality", 50.0),
        _fs("trend", 20.0),
    ]
    expected = (0.30 * 80 + 0.15 * 50 + 0.15 * 50 + 0.15 * 20) / (0.30 + 0.15 * 3)
    assert factor_composite(factors, weights) == pytest.approx(expected)
    # scaling all weights does not change the result
    assert factor_composite(factors, {k: 10 * v for k, v in weights.items()}) == pytest.approx(expected)
    # a factor without a (positive) weight is ignored
    assert factor_composite(factors, {**weights, "trend": 0.0}) == pytest.approx(
        (0.30 * 80 + 0.15 * 50 + 0.15 * 50) / 0.60
    )
    assert factor_composite([_fs(n, None) for n in FACTOR_ORDER], weights) is None
    assert factor_composite([], weights) is None
    all_100 = [_fs(n, 100.0) for n in FACTOR_ORDER]
    assert factor_composite(all_100, settings.factors.weights) == pytest.approx(100.0)


def test_factor_pillar():
    assert factor_pillar(None) == 0.0 and factor_pillar(float("nan")) == 0.0
    assert factor_pillar(50.0) == 0.0
    assert factor_pillar(75.0) == pytest.approx(0.5)
    assert factor_pillar(0.0) == -1.0 and factor_pillar(100.0) == 1.0
    assert factor_pillar(130.0) == 1.0 and factor_pillar(-20.0) == -1.0


# ------------------------------------------------------------------ English output


def test_factors_english_labels_and_rationales(md_kr, settings):
    import re

    from quantcio.quant.factors import FACTOR_LABELS_EN

    en = settings.model_copy(deep=True)
    en.report.language = "en"
    tech = compute_technicals(md_kr.prices, md_kr.benchmark, settings.risk.atr_period)
    ko_f = compute_factors(md_kr, tech, settings)
    en_f = compute_factors(md_kr, tech, en)
    assert [f.label for f in en_f] == [FACTOR_LABELS_EN[f.name] for f in en_f]
    for a, b in zip(ko_f, en_f, strict=True):
        assert a.score == b.score and a.metrics == b.metrics  # language never changes a number
        assert b.rationale and not re.search("[가-힣]", b.rationale), b.rationale
    md_kr.fundamentals.per = None
    md_kr.valuation_history = None
    v = next(f for f in compute_factors(md_kr, tech, en) if f.name == "value")
    assert "PER missing/loss-making: PER metrics excluded" in v.rationale
