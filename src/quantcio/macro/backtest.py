"""Macro regime validation: conditional next-month benchmark returns and the exposure overlay.

Timing (the same rule as the factor backtest, docs/BACKTEST_MACRO.md section 2):

- The regime of month-end ``t`` uses data <= t (calendar day t, including US closes of that day).
- Positions change at the close of the first session **after** t (``t1``). That session's return is
  still earned at the old exposure, so no same-day information is used; the KR market closes before
  the US session of day t has even started.
- "Next-month return" = close(last session <= next month-end) / close(t1) - 1.
- Overlay: exposure ``m(score_t)`` in [overlay_min_mult, 1] is held from the close of t1 until the next
  change; the rest sits in cash earning 0. Buy-and-hold is 100% invested over the same days. The
  Sharpe ratio uses rf = 0 for both, so they are comparable. A NaN score keeps exposure at 1.0; such
  months are counted.

Benchmarks: KOSPI (KR provider: Naver fchart price index, dividends NOT included) with the Korean
regime, SPY (US provider: yfinance adjusted, dividends included) with the US regime. Both are also
split by the global regime.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from quantcio.config import Settings
from quantcio.data.cache import DiskCache
from quantcio.macro.data import KST, load_macro_series
from quantcio.macro.regime import (
    INDICATORS,
    INDICATORS_BY_KEY,
    LABEL_NA,
    LABEL_RISK_OFF,
    LABEL_RISK_ON,
    LABELS,
    build_indicator_frames,
    overlay_multiplier,
    regime_history,
)

log = logging.getLogger(__name__)

TRADING_DAYS = 252
SHARPE_MIN_GAIN = 0.10  # overlay Sharpe - buy&hold Sharpe needed to call it a return improvement
MDD_MIN_GAIN = 0.03  # 3%p shallower max drawdown needed to call it a risk reduction
VERDICT_NO_EDGE = "리스크 관리용(수익 개선 근거 없음)"

MARKETS = {
    "KR": {"series": "kospi", "region": "kr", "name": "KOSPI"},
    "US": {"series": "spy", "region": "us", "name": "S&P 500 (SPY)"},
}


# ----------------------------------------------------------------------------------- statistics


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def mean_t(x: pd.Series | np.ndarray) -> dict[str, float | int | None]:
    """n, mean, std (ddof=1), t = mean / (std / sqrt(n)), hit rate (share > 0)."""
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "t": None, "hit_rate": None}
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else None
    t = mean / (std / math.sqrt(n)) if std and std > 0 else None
    return {"n": n, "mean": mean, "std": std, "t": _f(t), "hit_rate": float((a > 0).mean())}


def welch_t(a: np.ndarray, b: np.ndarray) -> dict[str, float | int | None]:
    """Difference of means a - b with the Welch t-statistic."""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    out: dict[str, float | int | None] = {"n_a": int(a.size), "n_b": int(b.size), "diff": None, "t": None}
    if a.size == 0 or b.size == 0:
        return out
    out["diff"] = float(a.mean() - b.mean())
    if a.size > 1 and b.size > 1:
        se = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
        out["t"] = _f(out["diff"] / se) if se > 0 else None
    return out


def conditional_stats(returns: pd.Series, labels: pd.Series) -> dict[str, Any]:
    """Returns grouped by regime label (aligned on the index): per label n / mean / std / t / hit rate,
    all months together, and 위험선호 minus 위험회피 (Welch t)."""
    df = pd.concat([returns.rename("ret"), labels.rename("label")], axis=1, join="inner").dropna(subset=["ret"])
    df["label"] = df["label"].fillna(LABEL_NA)
    by_label = {lab: mean_t(df.loc[df["label"] == lab, "ret"]) for lab in LABELS}
    on = df.loc[df["label"] == LABEL_RISK_ON, "ret"].to_numpy()
    off = df.loc[df["label"] == LABEL_RISK_OFF, "ret"].to_numpy()
    rest = df.loc[(df["label"] != LABEL_RISK_OFF) & (df["label"] != LABEL_NA), "ret"].to_numpy()
    return {
        "labels": by_label,
        "all": mean_t(df["ret"]),
        "on_minus_off": welch_t(on, off),
        "off_minus_rest": welch_t(off, rest),
    }


def entry_exit(sessions: pd.DatetimeIndex, signal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """For each signal date t (sorted): entry = first session > t, exit = last session <= next signal
    date. The last signal has no exit (NaT)."""
    sig = pd.DatetimeIndex(signal_dates).sort_values()
    s = pd.DatetimeIndex(sessions)
    p_entry = s.searchsorted(sig, side="right")
    entry = [s[p] if p < len(s) else pd.NaT for p in p_entry]
    nxt = list(sig[1:]) + [pd.NaT]
    exit_ = []
    for e, n in zip(entry, nxt, strict=True):
        if pd.isna(n) or pd.isna(e):
            exit_.append(pd.NaT)
            continue
        p = s.searchsorted(n, side="right") - 1
        x = s[p] if p >= 0 else pd.NaT
        exit_.append(x if not pd.isna(x) and x > e else pd.NaT)
    return pd.DataFrame({"entry": entry, "exit": exit_}, index=sig)


def next_period_returns(close: pd.Series, signal_dates: pd.DatetimeIndex) -> pd.Series:
    """close(exit) / close(entry) - 1 per signal date (see :func:`entry_exit`)."""
    ee = entry_exit(close.index, signal_dates)
    vals = []
    for _, row in ee.iterrows():
        if pd.isna(row["entry"]) or pd.isna(row["exit"]):
            vals.append(np.nan)
        else:
            vals.append(float(close.loc[row["exit"]] / close.loc[row["entry"]] - 1.0))
    return pd.Series(vals, index=ee.index, name="next_ret")


def perf_stats(daily: pd.Series) -> dict[str, float | int | None]:
    """CAGR, annualised vol, Sharpe (rf = 0), max drawdown, total return of a daily simple-return series."""
    r = daily.dropna()
    n = len(r)
    if n < 2:
        return {"n_days": n, "cagr": None, "vol": None, "sharpe": None, "mdd": None, "total_return": None}
    eq = (1.0 + r).cumprod()
    years = n / TRADING_DAYS
    total = float(eq.iloc[-1] - 1.0)
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if eq.iloc[-1] > 0 else -1.0
    sd = float(r.std(ddof=1))
    vol = sd * math.sqrt(TRADING_DAYS)
    sharpe = float(r.mean()) / sd * math.sqrt(TRADING_DAYS) if sd > 0 else None
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq.to_numpy()]))
    mdd = float((np.concatenate([[1.0], eq.to_numpy()]) / peak - 1.0).min())
    return {"n_days": n, "cagr": cagr, "vol": vol, "sharpe": _f(sharpe), "mdd": mdd, "total_return": total}


def overlay_returns(close: pd.Series, mult: pd.Series) -> pd.DataFrame:
    """Daily buy&hold vs overlay returns. ``mult`` is indexed by signal date; each exposure is set at the
    close of the first session after its date and applies from the following session on."""
    sessions = pd.DatetimeIndex(close.index)
    r = close.pct_change()
    ee = entry_exit(sessions, pd.DatetimeIndex(mult.index))
    set_at = pd.Series(mult.to_numpy(dtype="float64"), index=pd.DatetimeIndex(ee["entry"]))
    set_at = set_at[set_at.index.notna()]
    set_at = set_at[~set_at.index.duplicated(keep="last")]
    held = set_at.reindex(sessions).ffill().shift(1)  # exposure held during each session's return
    df = pd.DataFrame({"bench": r, "weight": held})
    df = df.dropna(subset=["weight", "bench"])
    df["overlay"] = df["weight"] * df["bench"]
    return df


def verdict(bh: dict, ov: dict) -> dict[str, Any]:
    """Plain-Korean verdict with fixed, pre-set thresholds (no tuning on the result)."""
    s_gain = (ov["sharpe"] - bh["sharpe"]) if ov.get("sharpe") is not None and bh.get("sharpe") is not None else None
    m_gain = (ov["mdd"] - bh["mdd"]) if ov.get("mdd") is not None and bh.get("mdd") is not None else None
    sharpe_ok = s_gain is not None and s_gain >= SHARPE_MIN_GAIN
    mdd_ok = m_gain is not None and m_gain >= MDD_MIN_GAIN
    if s_gain is None or m_gain is None:
        text = "판정 불가(표본 부족)"
    elif sharpe_ok:
        text = f"위험조정수익 개선(샤프 {s_gain:+.2f}, MDD {m_gain * 100:+.1f}%p) — 표본 내 결과이므로 추적 필요"
    elif mdd_ok:
        text = f"{VERDICT_NO_EDGE} — 샤프 {s_gain:+.2f}, 최대낙폭만 {m_gain * 100:.1f}%p 축소"
    else:
        text = f"{VERDICT_NO_EDGE} — 샤프 {s_gain:+.2f}, MDD {m_gain * 100:+.1f}%p로 낙폭 개선도 미미"
    return {
        "text": text,
        "sharpe_gain": _f(s_gain),
        "mdd_gain": _f(m_gain),
        "sharpe_improved": bool(sharpe_ok),
        "mdd_improved": bool(mdd_ok),
        "thresholds": {"sharpe_min_gain": SHARPE_MIN_GAIN, "mdd_min_gain": MDD_MIN_GAIN},
    }


# ----------------------------------------------------------------------------------- driver


def _labels_scores(hist: pd.DataFrame, region: str) -> tuple[pd.Series, pd.Series]:
    h = hist[hist["region"] == region].set_index("date").sort_index()
    return h["label"], h["score"].astype("float64")


def market_backtest(
    close: pd.Series,
    hist: pd.DataFrame,
    region: str,
    settings: Settings,
    window_start: pd.Timestamp,
) -> dict[str, Any]:
    """Conditional stats (own region + global) and the overlay test for one benchmark."""
    close = close.dropna().sort_index()
    labels, scores = _labels_scores(hist, region)
    g_labels, _ = _labels_scores(hist, "global")
    dates = labels.index[(labels.index >= window_start) & (labels.index <= close.index[-1])]
    nxt = next_period_returns(close, dates)
    cond = {
        region: conditional_stats(nxt, labels.reindex(dates)),
        "global": conditional_stats(nxt, g_labels.reindex(dates)),
    }
    sc = scores.reindex(dates)
    mult = pd.Series([overlay_multiplier(s, settings.macro.overlay_min_mult, warn=False) for s in sc], index=dates)
    daily = overlay_returns(close, mult)
    bh = perf_stats(daily["bench"])
    ov = perf_stats(daily["overlay"])
    months = pd.DataFrame({"score": sc, "mult": mult, "label": labels.reindex(dates), "next_ret": nxt})
    eq = (1.0 + daily[["bench", "overlay"]]).cumprod()
    monthly_eq = eq.groupby(eq.index.to_period("M")).tail(1)  # last session of each month
    return {
        "period": {
            "start": daily.index[0].date().isoformat() if len(daily) else None,
            "end": daily.index[-1].date().isoformat() if len(daily) else None,
            "months": int(len(dates)),
            "months_with_return": int(nxt.notna().sum()),
        },
        "conditional": cond,
        "overlay": {
            "buy_hold": bh,
            "overlay": ov,
            "avg_exposure": _f(daily["weight"].mean()) if len(daily) else None,
            "months_reduced": int((mult < 1.0).sum()),
            "months_nan_score": int(sc.isna().sum()),
            "verdict": verdict(bh, ov),
        },
        "months": [
            {
                "date": d.date().isoformat(),
                "score": _f(r["score"]),
                "label": r["label"] if isinstance(r["label"], str) else LABEL_NA,
                "mult": _f(r["mult"]),
                "next_ret": _f(r["next_ret"]),
            }
            for d, r in months.iterrows()
        ],
        "curves": {
            "dates": [d.date().isoformat() for d in monthly_eq.index],
            "buy_hold": [_f(v) for v in monthly_eq["bench"]],
            "overlay": [_f(v) for v in monthly_eq["overlay"]],
        },
    }


def macro_backtest(
    settings: Settings,
    cache: DiskCache,
    *,
    series: dict[str, pd.Series] | None = None,
    today: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """(a) next-month KOSPI / SPY returns by month-end regime label and (b) the overlay vs buy&hold
    over ``macro.lookback_years``. JSON-serialisable; NaN -> None."""
    data = series if series is not None else load_macro_series(settings, cache)
    hist = regime_history(settings, cache, series=data)
    now = pd.Timestamp(today) if today is not None else pd.Timestamp(datetime.now(KST).date())
    window_start = now - pd.DateOffset(years=settings.macro.lookback_years)
    notes: list[str] = []
    markets: dict[str, Any] = {}
    for mkt, cfg in MARKETS.items():
        key = cfg["series"]
        if key not in data or len(data[key]) < 300:
            reason = (getattr(data, "failures", {}) or {}).get(key, "시계열 없음/부족")
            notes.append(f"{mkt} 벤치마크({cfg['name']}) 없음 — {reason}")
            markets[mkt] = {"error": reason}
            continue
        try:
            res = market_backtest(data[key], hist, cfg["region"], settings, window_start)
        except Exception as exc:  # noqa: BLE001 - reported, never silent
            log.warning("macro backtest %s failed: %s", mkt, exc)
            notes.append(f"{mkt} 매크로 백테스트 실패: {type(exc).__name__}: {str(exc)[:120]}")
            markets[mkt] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        res["benchmark"] = cfg["name"]
        res["benchmark_source"] = str(
            (getattr(data, "sources", {}) or {}).get(key) or data[key].attrs.get("source", "")
        )
        res["region"] = cfg["region"]
        markets[mkt] = res
    coverage = _indicator_coverage(hist, data, settings)
    notes.extend(coverage["notes"])
    return {
        "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "window_start": window_start.date().isoformat(),
        "settings": {
            "lookback_years": settings.macro.lookback_years,
            "z_window_years": settings.macro.z_window_years,
            "risk_off_threshold": settings.macro.risk_off_threshold,
            "risk_on_threshold": settings.macro.risk_on_threshold,
            "overlay_min_mult": settings.macro.overlay_min_mult,
            "cash_return": 0.0,
            "rebalance": "월말 신호 → 다음 세션 종가에 노출 변경",
        },
        "markets": markets,
        "label_counts": _label_counts(hist, window_start),
        "coverage": coverage["first_z"],
        "data": {
            "sources": dict(getattr(data, "sources", {}) or {}),
            "failures": dict(getattr(data, "failures", {}) or {}),
            "notes": list(getattr(data, "notes", []) or []),
        },
        "notes": notes,
    }


def _label_counts(hist: pd.DataFrame, window_start: pd.Timestamp) -> dict[str, dict[str, int]]:
    h = hist[hist["date"] >= window_start]
    return {
        region: {lab: int((g["label"] == lab).sum()) for lab in LABELS} for region, g in h.groupby("region", sort=False)
    }


def _indicator_coverage(hist: pd.DataFrame, data: dict[str, pd.Series], settings: Settings) -> dict[str, Any]:
    """First month-end at which each indicator has a z (so the reader knows what the history rests on)."""
    frames = build_indicator_frames(data, settings)
    first: dict[str, str | None] = {}
    notes: list[str] = []
    for spec in INDICATORS:
        fr = frames[spec.key]
        z = fr.z_adj.dropna() if fr.z_adj is not None else pd.Series(dtype="float64")
        first[spec.key] = z.index[0].date().isoformat() if len(z) else None
        if fr.error:
            notes.append(f"지표 {spec.label}: {fr.error} — 백테스트 전 기간 제외")
    for key in ("kr_ktb3y_3m", "kr_us_spread"):
        if first.get(key):
            notes.append(
                f"{INDICATORS_BY_KEY[key].label}: {first[key]}부터만 z 산출(네이버 채권 이력 약 3년) — "
                "그 이전 한국 점수는 나머지 지표로 계산"
            )
    return {"first_z": first, "notes": notes}
