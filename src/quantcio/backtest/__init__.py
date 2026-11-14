"""PIT factor backtest + walk-forward multi-factor model (docs/BACKTEST_MACRO.md sections 1-3).

Public API:
- ``compute_factor_panel(universe, fundamentals, settings, dates)`` -> month-end factor panel
- ``run_factor_backtest(market, settings, cache, *, use_cache=True, progress=None)`` -> ``BacktestResult``
- ``walk_forward_model(panel, settings, regime=None)`` -> ``ModelResult``
- ``score_tickers(market, tickers, settings, cache, as_of=None)`` -> ``{ticker: FactorModelScore}``
- ``write_backtest_report(result, out_dir)`` -> (markdown path, json path)
- ``load_cached_backtest(market, settings, cache)`` / ``report_summary(result)`` -> cached result / compact dict for
  ``PortfolioReport.factor_backtest[market]``
"""

from __future__ import annotations

from quantcio.backtest.engine import (
    BacktestResult,
    load_cached_backtest,
    report_summary,
    run_factor_backtest,
    score_tickers,
)
from quantcio.backtest.factors import EXPECTED_SIGN, FACTOR_LABELS, FACTORS, compute_factor_panel
from quantcio.backtest.model import ModelResult, walk_forward_model
from quantcio.backtest.report import write_backtest_report

__all__ = [
    "EXPECTED_SIGN",
    "FACTORS",
    "FACTOR_LABELS",
    "BacktestResult",
    "ModelResult",
    "compute_factor_panel",
    "load_cached_backtest",
    "report_summary",
    "run_factor_backtest",
    "score_tickers",
    "walk_forward_model",
    "write_backtest_report",
]
