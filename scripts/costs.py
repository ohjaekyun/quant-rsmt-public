"""Trading-cost model for the research portfolio backtests (docs/MODELING.md 47, pre-declared 2026-10-12).

Per side of a trade (fraction of the traded value):

    fixed_side                                      half of the existing round-trip cost (KR 15bp incl. tax, US 5bp)
  + half_spread = 0.5 x median_20d(Corwin-Schultz)  the daily high-low spread estimator (Corwin & Schultz 2012), 0..5%
  + impact      = Y x sigma_d x sqrt(value / ADV20) square-root law, Y = 1, sigma_d = 60-day daily volatility

Everything is computed from data up to the trade date. Research code - the served / evaluation cost rule is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FIXED_SIDE = {"KR": 0.0015, "US": 0.0005}
IMPACT_Y = 1.0
MAX_HALF_SPREAD = 0.05
AUM = {"KR": (1e8, 1e9, 1e10), "US": (1e6, 1e7, 1e8)}  # small / middle / large book
_K = 3.0 - 2.0 * np.sqrt(2.0)


def corwin_schultz(px: pd.DataFrame) -> pd.Series:
    """Two-day Corwin-Schultz spread estimates (negative estimates set to 0), indexed by the second day."""
    h, lo = px["high"].astype(float), px["low"].astype(float)
    ok = (h > 0) & (lo > 0) & (h >= lo)
    h, lo = h.where(ok), lo.where(ok)
    hl = np.log(h / lo) ** 2
    beta = hl + hl.shift(1)
    gamma = np.log(np.maximum(h, h.shift(1)) / np.minimum(lo, lo.shift(1))) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return s.clip(lower=0.0)


def cost_inputs(px: pd.DataFrame) -> pd.DataFrame:
    """Daily (half_spread, sigma_d, adv) known at each date's close (rolling windows ending that day)."""
    px = px.sort_index()
    c = px["close"].astype(float)
    hs = (0.5 * corwin_schultz(px).rolling(20, min_periods=10).median()).clip(0.0, MAX_HALF_SPREAD)
    sig = np.log(c.where(c > 0)).diff().rolling(60, min_periods=30).std()
    adv = (c * px["volume"].astype(float)).rolling(20, min_periods=10).mean()
    return pd.DataFrame({"half_spread": hs, "sigma_d": sig, "adv": adv})


def side_cost(value: np.ndarray, market: str, half_spread: np.ndarray, sigma_d: np.ndarray, adv: np.ndarray,
              scale: float = 1.0) -> np.ndarray:  # fmt: skip
    """Cost fraction of the traded value per side; missing inputs fall back to conservative values."""
    hs = np.where(np.isfinite(half_spread), half_spread, 0.005)
    sd = np.where(np.isfinite(sigma_d), sigma_d, 0.03)
    av = np.where(np.isfinite(adv) & (adv > 0), adv, np.nan)
    part = np.where(np.isfinite(av), np.sqrt(np.clip(value, 0, None) / av), 0.3)
    return scale * (FIXED_SIDE[market] + hs + IMPACT_Y * sd * part)
