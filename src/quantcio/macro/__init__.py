"""Macro regime module: global / US / Korea regime scores, exposure overlay and its backtest.

See docs/BACKTEST_MACRO.md section 4 and :mod:`quantcio.macro.regime` for the indicator table.
"""

from quantcio.macro.backtest import conditional_stats, macro_backtest, next_period_returns, perf_stats
from quantcio.macro.data import MacroSeries, load_macro_series
from quantcio.macro.regime import (
    INDICATORS,
    LABEL_NA,
    LABEL_NEUTRAL,
    LABEL_RISK_OFF,
    LABEL_RISK_ON,
    MacroWarning,
    current_regimes,
    exposure_multiplier,
    regime_history,
    regime_label,
)
from quantcio.macro.report import write_macro_report

__all__ = [
    "INDICATORS",
    "LABEL_NA",
    "LABEL_NEUTRAL",
    "LABEL_RISK_OFF",
    "LABEL_RISK_ON",
    "MacroSeries",
    "MacroWarning",
    "conditional_stats",
    "current_regimes",
    "exposure_multiplier",
    "load_macro_series",
    "macro_backtest",
    "next_period_returns",
    "perf_stats",
    "regime_history",
    "regime_label",
    "write_macro_report",
]
