"""RSMT-Q time-series extensions - research fork and pre-declared comparison (docs/RSMT.md section 6).

Usage:
    .venv/bin/python scripts/rsmt_ts.py --market KR [--out reports/rsmt] [--paths 1000]

Never imported by `quantcio`: forward holdout FTH-20260930 freezes the signal code and the config values, so this
module subclasses `quant.rsmt.RSMTree` at runtime (`RSMTreeX`, installed into the module for the duration of a run
by `installed()`) and adds columns to the panel after `build_panel` (`augment`). Settings are changed in memory only.

Extensions (switches read from `CONFIG`, set per variant; the baseline twin - no offsets, root only - never uses them):
1. sigma equation (`sigma_terms`): log sigma_e = log sigma_leaf + eta*v + sum_j theta_j V_j. The coefficients are fitted
   jointly with the leaves inside the offset loop (`fit_offsets_multi`, a vector version of `rsmt.fit_offsets`) on the
   training rows of every fold; the tree then works with the composite log-scale index v_c = eta*v + theta'vm (and a
   unit eta), and `predict_params` rebuilds v_c from the test frame.
2. extra univariate partition candidates: plain columns added to `settings.rsmt.partition_features` in memory.
3. oblique groups (`groups`): every group of 2-3 inputs becomes a set of projection columns c'x~ over a FIXED direction
   grid (x~ standardised on the training rows; pairs 8 angles, triples the 13 cubic-lattice directions). The group is
   ONE test: statistic = max over its directions of the sup-LM, reference = the max over the same directions under the
   same cluster multipliers (`joint_max_bootstrap`), so the direction search is paid for. Tests are adjusted over
   (univariate variables + groups); a selected group splits on its best direction.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import quantcio.quant.rsmt as R
from quantcio.data.base import PricePanel
from quantcio.quant.features import ewma_vol, regime_features
from quantcio.quant.panel import PanelData

log = logging.getLogger("rsmt_ts")

_BaseTree = R.RSMTree  # the production class (captured before any install)

# ----------------------------------------------------------------------------------------------- features

CAL_COLUMNS = ("cal_s1", "cal_c1", "cal_s4", "cal_c4")
HAR_COLUMNS = ("har_5", "har_66", "har_252")
LEV_COLUMNS = ("lev",)
TS_PARTITION = ("t_years", "d20_vol_ratio", "d20_mkt_vol", "d20_dist_ma200")
EXTRA_COLUMNS = (*CAL_COLUMNS, *HAR_COLUMNS, *LEV_COLUMNS, *TS_PARTITION)
DIFF_LAG = 20  # sessions
T_ORIGIN = pd.Timestamp("2000-01-01")


def calendar_columns(dates: pd.Series) -> pd.DataFrame:
    """Annual (k=1) and quarterly (k=4) harmonics of the day of year, and calendar time in years."""
    d = pd.to_datetime(dates)
    frac = (d.dt.dayofyear.to_numpy(dtype="float64") - 1.0) / 365.25
    out = {}
    for k in (1, 4):
        out[f"cal_s{k}"] = np.sin(2.0 * math.pi * k * frac)
        out[f"cal_c{k}"] = np.cos(2.0 * math.pi * k * frac)
    out["t_years"] = ((d - T_ORIGIN).dt.days / 365.25).to_numpy(dtype="float64")
    return pd.DataFrame(out, index=dates.index)


def ticker_extras(px: pd.DataFrame, bench: pd.DataFrame | None) -> pd.DataFrame:
    """Per-date HAR log-vol ratios and 20-session changes of regime variables of one ticker (data <= t only)."""
    close = px["close"]
    lr = np.log(close).diff()
    ew = ewma_vol(close)
    out = pd.DataFrame(index=px.index)
    with np.errstate(divide="ignore", invalid="ignore"):  # a flat window (halt) gives log 0 -> NaN, dropped + counted
        for k in (5, 66, 252):
            rv = np.sqrt(252.0 * (lr * lr).rolling(k, min_periods=k).mean())
            out[f"har_{k}"] = np.log(rv / ew)
    feats = regime_features(px, bench if bench is not None else pd.DataFrame())
    out["d20_vol_ratio"] = feats["vol_ratio_20_120"] - feats["vol_ratio_20_120"].shift(DIFF_LAG)
    out["d20_mkt_vol"] = feats["mkt_vol_60"] - feats["mkt_vol_60"].shift(DIFF_LAG)
    out["d20_dist_ma200"] = feats["dist_ma200"] - feats["dist_ma200"].shift(DIFF_LAG)
    return out.replace([np.inf, -np.inf], np.nan)


def augment(
    panel: PanelData, prices: PricePanel, events: pd.DataFrame | None = None, market: str | None = None
) -> tuple[PanelData, dict[str, int]]:
    """The panel with EXTRA_COLUMNS (and, with `events`, EVENT_COLUMNS) merged by (ticker, date); training rows with a
    missing extra are dropped (the same rows for every variant) and counted."""
    parts = []
    for ticker, px in prices.prices.items():
        bid = prices.ticker_benchmark.get(ticker)
        bench = prices.benchmarks.get(bid) if bid is not None else None
        e = ticker_extras(px, bench)
        e.insert(0, "date", e.index)
        e.insert(0, "ticker", ticker)
        parts.append(e.reset_index(drop=True))
    extras = pd.concat(parts, ignore_index=True)

    def merge(frame: pd.DataFrame) -> pd.DataFrame:
        f = frame.merge(extras, on=["ticker", "date"], how="left", validate="many_to_one")
        f = pd.concat([f, calendar_columns(f["date"])], axis=1)
        f["lev"] = np.minimum(f["ret_1m"].to_numpy(dtype="float64"), 0.0)
        if events is not None:
            f = pd.concat([f, event_columns(f["date"], str(market), events)], axis=1)
        return f

    train = merge(panel.train)
    cols = [*EXTRA_COLUMNS, *(EVENT_COLUMNS if events is not None else ())]
    bad = ~np.isfinite(train[cols].to_numpy(dtype="float64")).all(axis=1)
    counts = {"rows_before": int(len(train)), "dropped_missing_extra": int(bad.sum())}
    train = train[~bad].reset_index(drop=True)
    return replace(panel, train=train, pred=merge(panel.pred)), counts


# ----------------------------------------------------------------------------------------------- event risk
# docs/RSMT.md section 7.1: observable, point-in-time event-risk variables (date-level)

EVENT_COLUMNS = ("ev_vix", "ev_vrp", "ev_gpr", "ev_epu")
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.dta"
EPU_URL = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"
GPR_LAG_DAYS = 8  # weekly publication: an 8-calendar-day lag is always published
EPU_LAG_DAYS = 2  # daily publication
REL_WINDOW = "365D"  # relative level: log(7-day mean) - log(trailing 365-day median)
REFRESH_DAYS = 3


def _download(url: str, path: Path) -> Path:
    import requests

    if path.exists() and (time.time() - path.stat().st_mtime) < REFRESH_DAYS * 86400:
        return path
    r = requests.get(url, timeout=120, headers={"User-Agent": "quantcio-research"})
    r.raise_for_status()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(r.content)
    tmp.replace(path)
    return path


def _relative(daily: pd.Series) -> pd.Series:
    """log(7-day mean) - log(trailing 365-day median of the 7-day mean) on a gap-free daily calendar (data <= d)."""
    s = daily.asfreq("D").interpolate(limit=3)
    m7 = s.rolling("7D", min_periods=4).mean()
    return np.log(m7) - np.log(m7.rolling(REL_WINDOW, min_periods=180).median())


def load_events(settings: Any, cache: Any) -> pd.DataFrame:
    """Daily event-risk inputs (index = date): vix / vrp by US session, gpr_rel / epu_rel by calendar day (unlagged;
    `event_columns` applies the publication lags)."""
    from datetime import date as _date

    from quantcio.macro.data import fetch_yahoo

    root = Path(settings.cache_path) / "research"
    root.mkdir(parents=True, exist_ok=True)
    gpr = pd.read_stata(_download(GPR_URL, root / "gpr_daily.dta"))
    gpr = pd.Series(gpr["GPRD"].to_numpy(dtype="float64"), index=pd.to_datetime(gpr["date"])).sort_index()
    epu = pd.read_csv(_download(EPU_URL, root / "epu_daily.csv"))
    epu = pd.Series(
        epu["daily_policy_index"].to_numpy(dtype="float64"),
        index=pd.to_datetime(dict(year=epu["year"], month=epu["month"], day=epu["day"])),
    ).sort_index()
    vix, _, _ = fetch_yahoo("^VIX", _date(1990, 1, 1), settings, cache)
    spy, _, _ = fetch_yahoo("SPY", _date(1993, 1, 1), settings, cache)
    lr = np.log(spy).diff()
    rv21 = np.sqrt(252.0 * (lr * lr).rolling(21, min_periods=21).mean())
    us = pd.DataFrame({"vix": np.log(vix)}).join(pd.DataFrame({"rv21": rv21}), how="left")
    us["vrp"] = us["vix"] - np.log(100.0) - np.log(us["rv21"])
    out = {
        "us": us[["vix", "vrp"]].dropna(how="all"),
        "gpr": _relative(gpr).dropna().to_frame("gpr"),
        "epu": _relative(epu).dropna().to_frame("epu"),
    }
    return pd.concat({k: v for k, v in out.items()}, axis=1)


def _asof(dates: pd.Series, series: pd.DataFrame, col: str, shift_days: int, strict: bool) -> np.ndarray:
    s = series[col].dropna()
    left = pd.DataFrame({"t": pd.to_datetime(dates).to_numpy() - np.timedelta64(shift_days, "D")})
    left["_i"] = np.arange(len(left))
    right = pd.DataFrame({"t": s.index.to_numpy(), "v": s.to_numpy()})
    m = pd.merge_asof(left.sort_values("t"), right, on="t", allow_exact_matches=not strict)
    return m.sort_values("_i")["v"].to_numpy(dtype="float64")


def event_columns(dates: pd.Series, market: str, events: pd.DataFrame) -> pd.DataFrame:
    """EVENT_COLUMNS at each row date t, point in time: VIX / VRP of the last US session <= t (US) or < t (KR: the US
    session of t's date closes after Seoul); GPR lagged GPR_LAG_DAYS, EPU EPU_LAG_DAYS calendar days."""
    strict = market != "US"
    return pd.DataFrame(
        {
            "ev_vix": _asof(dates, events["us"], "vix", 0, strict),
            "ev_vrp": _asof(dates, events["us"], "vrp", 0, strict),
            "ev_gpr": _asof(dates, events["gpr"], "gpr", GPR_LAG_DAYS, False),
            "ev_epu": _asof(dates, events["epu"], "epu", EPU_LAG_DAYS, False),
        },
        index=dates.index,
    )


# ----------------------------------------------------------------------------------------------- offsets


def fit_offsets_multi(
    y: np.ndarray,
    z: np.ndarray,
    vm: np.ndarray,
    w: np.ndarray,
    leaf: np.ndarray,
    gamma0: float,
    eta0: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[float, np.ndarray, int, bool]:
    """`rsmt.fit_offsets` with a vector sigma equation log s = vm @ eta (column 0 = the production v): the same joint
    (mu_L, gamma) block, then a bounded (componentwise-clipped) Newton step for eta on
    l(eta) = sum w [-vm eta - r^2 / (2 sigma_L^2 e^{2 vm eta})], halved while the likelihood does not improve."""
    leaf = np.asarray(leaf, dtype=np.int64)
    nl = int(leaf.max()) + 1
    gamma, eta = float(gamma0), np.asarray(eta0, dtype="float64").copy()
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    sw_leaf = np.bincount(leaf, w, nl)

    def loglik(g: float, e: np.ndarray, mu_l: np.ndarray, sig2_l: np.ndarray) -> float:
        lv = vm @ e
        r2 = (y - g * z - mu_l[leaf]) ** 2
        return float((w * (-lv - 0.5 * r2 * np.exp(-2.0 * lv) / sig2_l[leaf])).sum())

    for it in range(1, max_iter + 1):
        lv = vm @ eta
        s2 = np.exp(2.0 * lv)
        ytil = y - gamma * z
        a = w / s2
        sa = np.bincount(leaf, a, nl)
        mu_l = np.bincount(leaf, a * ytil, nl) / sa
        res = ytil - mu_l[leaf]
        sig2_l = np.maximum(np.bincount(leaf, w * res * res / s2, nl) / sw_leaf, 1e-300)
        aa = w / (sig2_l[leaf] * s2)
        zc = z - (np.bincount(leaf, a * z, nl) / sa)[leaf]
        yc = y - (np.bincount(leaf, a * y, nl) / sa)[leaf]
        den = float((aa * zc * zc).sum())
        gamma_new = float((aa * zc * yc).sum() / den) if den > 0 else gamma
        mu_l = np.bincount(leaf, a * (y - gamma_new * z), nl) / sa
        r2 = (y - gamma_new * z - mu_l[leaf]) ** 2
        q = r2 * np.exp(-2.0 * lv) / sig2_l[leaf]
        grad = vm.T @ (w * (q - 1.0))
        hess = -2.0 * (vm * (w * q)[:, None]).T @ vm
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.zeros_like(eta)
        base = loglik(gamma_new, eta, mu_l, sig2_l)
        eta_new = np.clip(eta - step, lo, hi)
        for _ in range(20):
            if loglik(gamma_new, eta_new, mu_l, sig2_l) >= base - 1e-12:
                break
            eta_new = eta + 0.5 * (eta_new - eta)
        change = abs(gamma_new - gamma) + float(np.abs(eta_new - eta).sum())
        gamma, eta = gamma_new, eta_new
        if change < tol:
            return gamma, eta, it, True
    log.warning("fit_offsets_multi: no convergence in %d iterations (gamma=%.6g, eta=%s)", max_iter, gamma, eta)
    return gamma, eta, max_iter, False


# ----------------------------------------------------------------------------------------------- oblique groups


def direction_grid(dim: int) -> np.ndarray:
    """Unit directions (rows) of the fixed grid: 8 half-circle angles for pairs, 13 cubic-lattice directions for
    triples (a direction and its negative define the same split)."""
    if dim == 2:
        ang = np.arange(8) * math.pi / 8.0
        return np.column_stack([np.cos(ang), np.sin(ang)])
    if dim == 3:
        raw = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]
        raw += [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)]
        d = np.array(raw, dtype="float64")
        return d / np.linalg.norm(d, axis=1, keepdims=True)
    raise ValueError(f"oblique groups have 2 or 3 inputs, got {dim}")


BOOT_FLOATS = 2_000_000  # working-set bound of the joint bootstrap (as rsmt.BOOT_BATCH_FLOATS)


def joint_max_bootstrap(
    x: np.ndarray,
    s: np.ndarray,
    b_inv_sqrt: np.ndarray,
    h_mu: np.ndarray,
    h_sigma: np.ndarray,
    clusters: np.ndarray,
    trim: float,
    n_boot: int,
    seed: int,
    stat_obs: float,
) -> tuple[float, np.ndarray]:
    """p-value of max_j supLM(x[:, j]) against the multiplier bootstrap of the SAME maximum: every draw applies one
    set of cluster multipliers to all columns (`rsmt.cluster_bootstrap_null` per column, joined). Returns (p, draws)."""
    n, ncol = x.shape
    codes = np.asarray(clusters, dtype=np.int64).ravel()
    n_clu = int(codes.max()) + 1
    lo, hi = math.ceil(trim * n - 1e-9), math.floor((1.0 - trim) * n + 1e-9)
    sum_mu, sum_sig = float(h_mu.sum()), float(h_sigma.sum())
    cluster_tot = np.column_stack([np.bincount(codes, s[:, 0], n_clu), np.bincount(codes, s[:, 1], n_clu)])
    b00, b01, b10, b11 = (float(v) for v in np.asarray(b_inv_sqrt).ravel())
    prep = []
    for j in range(ncol):
        order = np.argsort(x[:, j], kind="stable")
        xs = x[order, j]
        k = np.nonzero(xs[1:] > xs[:-1])[0] + 1
        k = k[(k >= lo) & (k <= hi)]
        if k.size == 0:
            continue
        t = k / n
        s_sorted, c_sorted = s[order], codes[order]
        # few clusters x boundaries: per-cluster partial sums once, one matmul per batch (as the production null);
        # otherwise per-row multipliers and a cumulative sum per batch (bounded memory on long panels)
        mk = None
        if n * n_clu <= BOOT_FLOATS and k.size * n_clu <= BOOT_FLOATS:
            mk = []
            for col in (0, 1):
                cum = np.zeros((n, n_clu))
                cum[np.arange(n), c_sorted] = s_sorted[:, col]
                mk.append(np.cumsum(cum, axis=0)[k - 1].T.copy())
        prep.append(
            (
                mk,
                k,
                s_sorted,
                c_sorted,
                (np.cumsum(h_mu[order])[k - 1] / sum_mu)[None, :],
                (np.cumsum(h_sigma[order])[k - 1] / sum_sig)[None, :],
                n * t * (1.0 - t),
            )
        )
    if not prep:
        return 1.0, np.empty(0)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, 0x0B1]).generate_state(2))
    draws = []
    done = 0
    cap = max(1, BOOT_FLOATS // max(1, 2 * n))
    while done < n_boot:
        b = min(128, cap, n_boot - done)
        g = rng.standard_normal((b, n_clu))
        tot = g @ cluster_tot
        best = np.full(b, -np.inf)
        for mk, k, s_sorted, c_sorted, a_mu, a_sig, denom in prep:
            if mk is not None:
                c0 = g @ mk[0]
                c1 = g @ mk[1]
            else:
                g_rows = g[:, c_sorted]
                c0 = np.cumsum(g_rows * s_sorted[:, 0], axis=1)[:, k - 1]
                c1 = np.cumsum(g_rows * s_sorted[:, 1], axis=1)[:, k - 1]
            c0 = c0 - a_mu * tot[:, :1]
            c1 = c1 - a_sig * tot[:, 1:]
            u0 = c0 * b00 + c1 * b10
            u1 = c0 * b01 + c1 * b11
            best = np.maximum(best, ((u0 * u0 + u1 * u1) / denom).max(axis=1))
        draws.append(best)
        done += b
    d = np.sort(np.concatenate(draws))
    p = (1.0 + float((d >= stat_obs).sum())) / (d.size + 1.0)
    return p, d


# ----------------------------------------------------------------------------------------------- tree

CONFIG: dict[str, Any] = {"sigma_terms": (), "groups": ()}
THETA_BOUNDS = (-3.0, 3.0)


@contextlib.contextmanager
def installed(sigma_terms: Sequence[str] = (), groups: Sequence[Sequence[str]] = ()) -> Iterator[None]:
    """Run the production CV / holdout code with `RSMTreeX` and the given switches (restored afterwards)."""
    old = dict(CONFIG)
    CONFIG.update(sigma_terms=tuple(sigma_terms), groups=tuple(tuple(g) for g in groups))
    R.RSMTree = RSMTreeX  # type: ignore[misc]
    try:
        yield
    finally:
        R.RSMTree = _BaseTree  # type: ignore[misc]
        CONFIG.clear()
        CONFIG.update(old)


class RSMTreeX(_BaseTree):
    """`RSMTree` + the switches of `CONFIG` (see the module doc). Without switches every method is the parent's."""

    def __init__(self, partition_features: Sequence[str], **kw: Any):
        super().__init__(partition_features, **kw)
        on = self.use_offsets  # the baseline twin (no offsets, root only) never uses the extensions
        self.x_sigma_terms: tuple[str, ...] = tuple(CONFIG["sigma_terms"]) if on else ()
        self.groups: tuple[tuple[str, ...], ...] = tuple(CONFIG["groups"]) if on else ()
        if self.groups and self.multiple_testing == "pca":
            raise ValueError("oblique groups are not combined with multiple_testing = 'pca'")
        self.x_theta = np.zeros(len(self.x_sigma_terms))
        self.eta_v = float("nan")
        self._obl: list[dict[str, np.ndarray]] | None = None
        self._xv2: np.ndarray | None = None
        self._frame: pd.DataFrame | None = None

    @property
    def extended(self) -> bool:
        return bool(self.x_sigma_terms or self.groups)

    # ------------------------------------------------------------------ inputs / split variables

    @property
    def inputs(self) -> list[str]:
        hidden = [c for g in self.groups for c in g if c not in self.partition_features]
        return [*self.partition_features, *dict.fromkeys(hidden)]

    @property
    def split_features(self) -> list[str]:
        if not self.groups:
            return super().split_features
        names = list(self.partition_features)
        for g in self.groups:
            for d in direction_grid(len(g)):
                names.append("(" + " ".join(f"{c:+.2f}·{v}" for c, v in zip(d, g, strict=True)) + ")")
        return names

    def _matrix(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.groups:
            return super()._matrix(x)
        if not isinstance(x, pd.DataFrame):
            raise TypeError("RSMTreeX with oblique groups needs a DataFrame")
        return x[self.inputs].to_numpy(dtype="float64")

    def _fit_group_scales(self, raw: np.ndarray) -> None:
        cols = self.inputs
        self._obl = []
        for g in self.groups:
            idx = [cols.index(c) for c in g]
            sub = raw[:, idx]
            sd = sub.std(axis=0)
            self._obl.append(
                {
                    "idx": np.array(idx),
                    "mean": sub.mean(axis=0),
                    "sd": np.where(sd > 1e-12, sd, 1.0),
                    "dirs": direction_grid(len(g)),
                }
            )

    def _split_matrix(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.groups:
            return super()._split_matrix(raw)
        if self._obl is None:
            raise RuntimeError("oblique standardisation not fitted")
        missing = ~np.isfinite(raw).all(axis=1)
        cols = [raw[:, : len(self.partition_features)]]
        for spec in self._obl:
            zs = (raw[:, spec["idx"]] - spec["mean"]) / spec["sd"]
            cols.append(zs @ spec["dirs"].T)
        return np.column_stack(cols), missing

    def _group_columns(self) -> list[np.ndarray]:
        start = len(self.partition_features)
        out = []
        for g in self.groups:
            d = len(direction_grid(len(g)))
            out.append(np.arange(start, start + d))
            start += d
        return out

    # ------------------------------------------------------------------ tests

    def _uni_test(self, j: int, name: str, stat: float, times: np.ndarray, bn: R.BootNull | None) -> R.VariableTest:
        null: np.ndarray | None = None
        if times.size == 0:
            return R.VariableTest(name, stat, 1.0, float("nan"), 0, "none", float("nan"))
        if bn is not None:
            null, kind, p = bn.draws, "bootstrap", bn.p
        elif times.size <= R.DISCRETE_NULL_MAX_POINTS:
            null, kind = R.discrete_null(times, R.N_SCORE, self.null_sims, self.null_seed), "discrete"
            p = R.p_value(null, stat)
        else:
            null, kind = self._null, "continuum"
            p = R.p_value(null, stat)
        return R.VariableTest(name, stat, p, float("nan"), int(times.size), kind, float(np.quantile(null, 0.99)))

    def _test_variables(
        self, xn: np.ndarray, u: np.ndarray, boot: dict[str, Any] | None = None
    ) -> list[R.VariableTest]:
        if not self.groups:
            return super()._test_variables(xn, u, boot)
        names = self.split_features
        n_uni = len(self.partition_features)
        observed = [R.sup_lm(xn[:, j], u, self.trim) for j in range(xn.shape[1])]
        boot_null: list[R.BootNull | None] = [None] * n_uni
        if boot is not None:
            boot_null = R.cluster_bootstrap_null(
                xn[:, :n_uni],
                boot["s"],
                boot["b_ih"],
                boot["h_mu"],
                boot["h_sigma"],
                boot["clusters"],
                self.trim,
                self.boot_sims,
                boot["seed"],
                stats_obs=[observed[j][0] for j in range(n_uni)],
            )
        tests = [self._uni_test(j, names[j], *observed[j], boot_null[j]) for j in range(n_uni)]
        family_p = [t.p for t in tests]
        group_best: list[tuple[np.ndarray, int, float, float]] = []
        for gi, cols in enumerate(self._group_columns()):
            stats_g = np.array([observed[c][0] for c in cols])
            best = int(cols[int(np.argmax(stats_g))])
            stat = float(stats_g.max())
            if not any(observed[c][1].size for c in cols):
                p, q99 = 1.0, float("nan")
            elif boot is not None:
                seed = int(np.random.SeedSequence([boot["seed"], 0x6A, gi]).generate_state(1)[0])
                p, draws = joint_max_bootstrap(
                    xn[:, cols],
                    boot["s"],
                    boot["b_ih"],
                    boot["h_mu"],
                    boot["h_sigma"],
                    boot["clusters"],
                    self.trim,
                    self.boot_sims,
                    seed,
                    stat,
                )
                q99 = float(np.quantile(draws, 0.99)) if draws.size else float("nan")
            else:  # unclustered fits (tests only): Bonferroni over the directions
                p_dir = [R.p_value(self._null, observed[c][0]) for c in cols]
                p, q99 = min(1.0, len(cols) * min(p_dir)), float(np.quantile(self._null, 0.99))
            family_p.append(p)
            group_best.append((cols, best, p, q99))
        adj = self.adjust_p(family_p)
        for t, pa in zip(tests, adj[:n_uni], strict=True):
            t.p_adj = float(pa)
        for (cols, best, p, q99), pa in zip(group_best, adj[n_uni:], strict=True):
            for c in cols:
                stat, times = observed[c]
                if c == best:
                    tests.append(R.VariableTest(names[c], stat, p, float(pa), int(times.size), "joint-bootstrap", q99))
                else:  # the group is one test: only its best direction can be selected
                    tests.append(R.VariableTest(names[c], stat, 1.0, 1.0, int(times.size), "joint-bootstrap", q99))
        return tests

    # ------------------------------------------------------------------ fit / predict

    def fit_frame(self, df: pd.DataFrame, y_col: str = "y_e") -> RSMTreeX:
        self._xv2 = df[list(self.x_sigma_terms)].to_numpy(dtype="float64") if self.x_sigma_terms else None
        self._frame = df if self.groups else None
        return super().fit_frame(df, y_col)  # type: ignore[return-value]

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series,
        z: np.ndarray | pd.Series | None = None,
        v: np.ndarray | pd.Series | None = None,
        w: np.ndarray | pd.Series | None = None,
        m: np.ndarray | pd.Series | None = None,
        cluster: np.ndarray | pd.Series | None = None,
        scale: np.ndarray | pd.Series | None = None,
        boot_cluster: np.ndarray | pd.Series | None = None,
    ) -> RSMTreeX:
        """The parent's `fit` with (a) the projection columns of the oblique groups in the split matrix and (b) the
        vector sigma equation; without switches it IS the parent's fit."""
        if not self.extended:
            return super().fit(X, y, z, v, w, m, cluster, scale, boot_cluster)  # type: ignore[return-value]
        if self.groups and isinstance(X, pd.DataFrame) and not set(self.inputs) <= set(X.columns):
            X = self._frame  # fit_frame passes only the partition columns
        raw = self._matrix(X)
        ya = np.asarray(y, dtype="float64").ravel()
        n = ya.size
        wa = np.ones(n) if w is None else np.asarray(w, dtype="float64").ravel()
        if self.scaled:
            ya = ya / np.asarray(scale, dtype="float64").ravel()
        self.pca, self._split_names = None, None
        if self.groups:
            self._fit_group_scales(raw)
            xa = self._split_matrix(raw)[0]
        else:
            xa = raw
        if not np.isfinite(xa).all() or not np.isfinite(ya).all():
            raise ValueError("RSMTreeX.fit: non-finite inputs")
        za = self._check_offset("z", None if z is None else np.asarray(z, dtype="float64"), wa)
        va = self._check_offset("v", None if v is None else np.asarray(v, dtype="float64"), wa)
        vm = va[:, None]
        if self.x_sigma_terms:
            v2 = self._xv2
            if v2 is None or v2.shape[0] != n or not np.isfinite(v2).all():
                raise ValueError("RSMTreeX.fit: sigma terms missing or non-finite")
            vm = np.column_stack([va, v2])
        ma = None if m is None else np.asarray(m, dtype="float64").ravel()
        codes = None
        if cluster is not None:
            codes = pd.factorize(pd.Series(np.asarray(cluster).ravel()), sort=True)[0].astype(np.int64)
        clustered = codes is not None and int(codes.max()) + 1 < n
        weighted = bool(np.ptp(wa) > 1e-12 * float(wa.max()))
        use_boot = self.null_method == "bootstrap" or (self.null_method == "auto" and (clustered or weighted))
        self.null_used = "bootstrap" if use_boot else "bb"
        self._date_codes = codes if codes is not None else np.arange(n, dtype=np.int64)
        boot_codes = self._date_codes
        if boot_cluster is not None:  # production: runs of boot_block_horizons h-blocks (quant.rsmt.fit_frame)
            boot_codes = pd.factorize(pd.Series(np.asarray(boot_cluster).ravel()), sort=True)[0].astype(np.int64)
        self._clusters = boot_codes if use_boot else None
        self.n_clusters = int(codes.max()) + 1 if codes is not None else n
        self._null = R.sup_lm_null(2, self.trim, self.null_sims, self.n_grid, self.null_seed, self.cache_dir)
        self.n_train = n
        gamma = self.gamma0
        eta = np.zeros(vm.shape[1])
        eta[0] = self.eta0
        bounds = [tuple(self.eta_bounds)] + [THETA_BOUNDS] * (vm.shape[1] - 1)
        for _round in range(self.outer_rounds):
            self._round = _round
            ytil = ya - gamma * za
            s2 = np.exp(2.0 * (vm @ eta))
            self._next_id = 0
            root = self._grow(np.arange(n), 0, (), xa, ytil, s2, wa)
            leaves = self._collect_leaves(root)
            leaf_index = np.empty(n, dtype=np.int64)
            for i, node in enumerate(leaves):
                leaf_index[node.rows] = i
            if vm.shape[1] == 1:  # exactly the production offsets
                off = R.fit_offsets(
                    ya,
                    za,
                    va,
                    wa,
                    leaf_index,
                    gamma,
                    float(eta[0]),
                    self.offset_max_iter,
                    self.offset_tol,
                    self.eta_bounds,
                )
                gamma, eta = off.gamma, np.array([off.eta])
                self.offset_fit = off
            else:
                gamma, eta, it, ok = fit_offsets_multi(ya, za, vm, wa, leaf_index, gamma, eta, bounds)
                self.offset_fit = R.OffsetFit(gamma, float(eta[0]), it, ok)
        self.root = root
        self.gamma = float(gamma)
        self.eta_v, self.x_theta = float(eta[0]), np.asarray(eta[1:], dtype="float64")
        vc = vm @ eta  # composite log-scale index; the parent's code below sees it with a unit eta
        self.eta = 1.0
        self._leaves = self._collect_leaves(root)
        for i, node in enumerate(self._leaves, start=1):
            node.leaf_id = i
        if self.gamma_mode == "leaf":
            g_row = self._fit_node_gammas(root, ya, za, vc, wa)
        else:
            g_row = np.full(n, self.gamma)
            stack = [root]
            while stack:
                node = stack.pop()
                node.gamma = self.gamma
                if not node.is_leaf:
                    stack.extend([node.left, node.right])
        self._finalize(root, None, ya - g_row * za, np.exp(2.0 * vc), wa, ma)
        self._frame = None
        return self

    @property
    def eta_equivalent(self) -> float:
        if not self.extended:
            return super().eta_equivalent
        return self.eta_v + (1.0 if self.scaled else 0.0)

    def predict_params(
        self,
        X: pd.DataFrame | np.ndarray,
        z: np.ndarray | pd.Series | None = None,
        v: np.ndarray | pd.Series | None = None,
        scale: np.ndarray | pd.Series | None = None,
    ) -> dict[str, np.ndarray]:
        if not self.extended:
            return super().predict_params(X, z, v, scale)
        va = np.asarray(v, dtype="float64").ravel()
        vc = self.eta_v * va
        if self.x_sigma_terms:
            if not isinstance(X, pd.DataFrame):
                raise TypeError("RSMTreeX with sigma terms needs a DataFrame")
            vc = vc + X[list(self.x_sigma_terms)].to_numpy(dtype="float64") @ self.x_theta
        return super().predict_params(X, z, vc, scale)

    def sigma_equation(self) -> dict[str, float]:
        out = {"eta_v": self.eta_v}
        out.update({t: float(c) for t, c in zip(self.x_sigma_terms, self.x_theta, strict=True)})
        return out


# ----------------------------------------------------------------------------------------------- comparison

LAYOUTS = (3, 4, 5)
HOLDOUT_MONTHS = 24
TIE_REL = 0.001
COVER80_BAND = (0.75, 0.85)
EXC95_BAND = (0.02, 0.08)
GROUPS = (
    ("cal_s4", "cal_c4"),
    ("cal_s1", "cal_c1"),
    ("mkt_vol_60", "mkt_ret_60"),
    ("vol_ratio_20_120", "dd_250", "ret_1m"),
)
VARIANTS: dict[str, dict[str, Any]] = {
    "W0": {"components": 0, "label": "current (KR V6 / US V1)"},
    "W1": {"components": 1, "label": "+HAR sigma terms", "sigma": HAR_COLUMNS},
    "W2": {"components": 2, "label": "+HAR +leverage", "sigma": (*HAR_COLUMNS, *LEV_COLUMNS)},
    "W3": {
        "components": 3,
        "label": "+HAR +leverage +seasonal sigma",
        "sigma": (*HAR_COLUMNS, *LEV_COLUMNS, *CAL_COLUMNS),
    },
    "W4": {"components": 1, "label": "+time-series split candidates (Simes)", "partition": TS_PARTITION},
    "W5": {"components": 2, "label": "W4 +oblique groups", "partition": TS_PARTITION, "groups": GROUPS},
    "W6": {
        "components": 5,
        "label": "W3 + W5",
        "sigma": (*HAR_COLUMNS, *LEV_COLUMNS, *CAL_COLUMNS),
        "partition": TS_PARTITION,
        "groups": GROUPS,
    },
    "W7": {"components": 1, "label": "+seasonal sigma only", "sigma": CAL_COLUMNS},
    # section 7.1 (event risk, proposal A with observable data)
    "E1": {"components": 1, "label": "+event-risk sigma terms", "sigma": EVENT_COLUMNS, "events": True},
    "E2": {
        "components": 1,
        "label": "+event-risk split candidates (Simes)",
        "partition": EVENT_COLUMNS,
        "events": True,
    },
    "E3": {
        "components": 2,
        "label": "E1 + E2",
        "sigma": EVENT_COLUMNS,
        "partition": EVENT_COLUMNS,
        "events": True,
    },
}
ORDER = ("W0", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "E1", "E2", "E3")


def variant_settings(base: Any, market: str, name: str) -> Any:
    s = base.model_copy(deep=True)
    spec = VARIANTS[name]
    if spec.get("partition"):
        s.rsmt.partition_features = [*s.rsmt.partition_features, *spec["partition"]]
        s.rsmt.markets[market].multiple_testing = "simes"
    return s


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def select(rows: dict[str, dict[str, Any]]) -> tuple[str, list[str], str]:
    eligible = [
        n
        for n, r in rows.items()
        if r["crps"] is not None
        and r["cover80"] is not None
        and r["exc95"] is not None
        and COVER80_BAND[0] <= r["cover80"] <= COVER80_BAND[1]
        and EXC95_BAND[0] <= r["exc95"] <= EXC95_BAND[1]
    ]
    if not eligible:
        return "W0", [], "no variant inside both coverage bands -> W0 stays"
    best = min(rows[n]["crps"] for n in eligible)
    tied = [n for n in eligible if (rows[n]["crps"] - best) / best < TIE_REL]
    tied.sort(key=lambda n: (VARIANTS[n]["components"], ORDER.index(n)))
    lowest = min(eligible, key=lambda n: rows[n]["crps"])
    return (
        tied[0],
        tied,
        f"eligible {eligible}; lowest {lowest} ({best:.6f}); tied within {TIE_REL:.1%}: {tied} -> {tied[0]}",
    )


def _metrics_row(m: dict[str, Any]) -> dict[str, Any]:
    keys = ("crps", "base_crps", "crps_gain_p05", "cover80", "exc95", "auc", "base_auc", "ic", "auc_rel", "logloss")
    return {k: _f(m.get(k)) for k in keys}


def run(market_code: str, out_dir: Path, n_paths: int, only: Sequence[str] | None = None) -> dict[str, Any]:
    from quantcio import holdout
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache
    from quantcio.models import Market
    from quantcio.pipeline import training_panel
    from quantcio.quant.panel import build_panel

    t_all = time.perf_counter()
    market = Market(market_code)
    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    price, notes, _ = training_panel(market, settings, cache)
    price, dropped = holdout.guard_panel(price, settings)
    panel = build_panel(price, settings)
    panel, counts = augment(panel, price)
    tr_all = panel.train
    grid = pd.DatetimeIndex(sorted(tr_all.loc[tr_all["on_grid"], "date"].unique()))
    first_hold = grid[grid > grid[-1] - pd.DateOffset(months=HOLDOUT_MONTHS)][0]
    pre = replace(panel, train=tr_all[(pd.to_datetime(tr_all["t_end"]) < first_hold).to_numpy()].reset_index(drop=True))
    info = {
        "market": market.value,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_digest": settings.digest(),
        "holdout_guard": dropped,
        "augment": counts,
        "n_train": int(len(tr_all)),
        "n_on_grid": int(tr_all["on_grid"].sum()),
        "n_grid_dates": int(len(grid)),
        "first_date": str(grid[0].date()),
        "holdout_start": str(first_hold.date()),
        "pre_rows": int(len(pre.train)),
        "n_paths": n_paths,
    }
    log.info("%s: %s", market.value, info)
    rows: dict[str, dict[str, Any]] = {}
    for name, spec in VARIANTS.items():
        if spec.get("events"):  # section 7.1 variants run in scripts/rsmt_history.py
            continue
        if only and name not in only:  # smoke runs only; the comparison always runs every variant
            continue
        s = variant_settings(settings, market.value, name)
        t = time.perf_counter()
        with installed(spec.get("sigma", ()), spec.get("groups", ())):
            runs = R.cross_validate_layouts(pre, s, list(LAYOUTS), n_paths)
            tree = R.RSMTree.from_settings(s, market).fit_frame(pre.train)
        per = {
            str(k): {
                **_metrics_row(r.metrics),
                "splits": [int(f.get("n_splits", 0)) for f in r.folds if "skipped" not in f],
                "dist_gate": bool(r.distribution_ok),
            }
            for k, r in runs.items()
        }

        def mean(key: str, per: dict[str, Any] = per) -> float | None:
            vals = [per[str(k)][key] for k in LAYOUTS if per[str(k)][key] is not None]
            return float(np.mean(vals)) if len(vals) == len(LAYOUTS) else None

        rows[name] = {
            "label": spec["label"],
            "components": spec["components"],
            **{
                k: mean(k)
                for k in ("crps", "base_crps", "cover80", "exc95", "auc", "base_auc", "ic", "auc_rel", "logloss")
            },
            "dist_gate_layouts": sum(per[str(k)]["dist_gate"] for k in LAYOUTS),
            "splits_final": int(tree.n_splits),
            "tree": tree.describe("en"),
            "sigma_equation": tree.sigma_equation()
            if isinstance(tree, RSMTreeX) and tree.extended
            else {"eta_v": tree.eta},
            "gamma": float(tree.gamma),
            "seconds": round(time.perf_counter() - t, 1),
            "layouts": per,
        }
        r = rows[name]
        log.info(
            "%s %s: CRPS %s cover80 %s exc95 %s AUC %s IC %s splits %s (%.1fs)",
            market.value,
            name,
            r["crps"],
            r["cover80"],
            r["exc95"],
            r["auc"],
            r["ic"],
            r["splits_final"],
            r["seconds"],
        )
    winner, tied, why = select(rows)
    log.info("%s: winner %s (%s)", market.value, winner, why)
    hold: dict[str, Any] = {}
    tr = panel.train
    train_mask = (pd.to_datetime(tr["t_end"]) < first_hold).to_numpy()
    test_mask = (pd.to_datetime(tr["date"]) >= first_hold).to_numpy()
    for name in dict.fromkeys(["W0", winner]) if not only else ():  # smoke runs never touch the holdout
        spec = VARIANTS[name]
        s = variant_settings(settings, market.value, name)
        with installed(spec.get("sigma", ()), spec.get("groups", ())):
            m, model, _ = R.evaluate_split(panel, s, train_mask, test_mask, n_paths)
        hold[name] = {
            **_metrics_row(m),
            "n_oos": int(m.get("n_oos", 0)),
            "n_oos_dates": int(m.get("n_oos_dates", 0)),
            "splits": int(model.n_splits),
        }
        log.info("%s holdout %s: %s", market.value, name, hold[name])
    res = {
        "info": info,
        "variants": rows,
        "winner": winner,
        "tied": tied,
        "why": why,
        "holdout": hold,
        "seconds": round(time.perf_counter() - t_all, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"ts_compare_{market.value}.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (out_dir / f"ts_compare_{market.value}.md").write_text(render(res), encoding="utf-8")
    return res


def _n(x: Any, d: int = 4) -> str:
    v = _f(x)
    return "—" if v is None else f"{v:.{d}f}"


def render(res: dict[str, Any]) -> str:
    info, rows, hold = res["info"], res["variants"], res["holdout"]
    base = rows["W0"]["crps"] if "W0" in rows else None
    md = [
        f"# RSMT-Q time-series extensions — {info['market']} (pre-declared, docs/RSMT.md §6)",
        "",
        f"- generated {info['generated_at']} · config digest `{info['config_digest']}` · "
        f"{info['n_train']:,} training rows ({info['n_on_grid']:,} on the grid, {info['n_grid_dates']} sample dates "
        f"from {info['first_date']}); augment {info['augment']}",
        f"- final holdout from {info['holdout_start']} (not used for the choice); CV on {info['pre_rows']:,} "
        "pre-holdout rows, layouts 3/4/5",
        "",
        "| variant | CRPS | vs W0 | base CRPS | cover80 | exc95 | AUC (twin) | μ IC | dist. gate | splits | "
        "sigma equation | s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        rel = "—" if r["crps"] is None or base is None else f"{(r['crps'] - base) / base:+.2%}"
        eq = ", ".join(f"{k} {v:+.3f}" for k, v in r["sigma_equation"].items() if v is not None)
        md.append(
            f"| **{name}** {r['label']} | {_n(r['crps'], 6)} | {rel} | {_n(r['base_crps'], 6)} | "
            f"{_n(r['cover80'], 3)} | {_n(r['exc95'], 3)} | {_n(r['auc'])} ({_n(r['base_auc'])}) | {_n(r['ic'])} | "
            f"{r['dist_gate_layouts']}/3 | {r['splits_final']} | {eq} | {r['seconds']} |"
        )
    md += [
        "",
        f"**Winner: {res['winner']}** — {res['why']}",
        "",
        "## Final holdout (once)",
        "",
        "| variant | CRPS | base CRPS | gain bound | cover80 | exc95 | AUC | base AUC | μ IC | splits |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, h in hold.items():
        md.append(
            f"| {name} | {_n(h['crps'], 6)} | {_n(h['base_crps'], 6)} | {_n(h['crps_gain_p05'], 6)} | "
            f"{_n(h['cover80'], 3)} | {_n(h['exc95'], 3)} | {_n(h['auc'])} | {_n(h['base_auc'])} | {_n(h['ic'])} | "
            f"{h['splits']} |"
        )
    md += ["", "## Trees (final pre-holdout fit)", ""]
    for name, r in rows.items():
        md += [f"**{name}** (γ {r['gamma']:+.4f})", "", *[f"    {ln}" for ln in r["tree"]], ""]
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--market", required=True, choices=["KR", "US"])
    ap.add_argument("--out", default="reports/rsmt")
    ap.add_argument("--paths", type=int, default=R.CV_PATHS)
    ap.add_argument("--only", nargs="*", default=None, help="smoke run: these variants only (not the comparison)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    res = run(args.market, Path(args.out), args.paths, args.only)
    print(f"{args.market}: winner {res['winner']} ({res['why']}) in {res['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
