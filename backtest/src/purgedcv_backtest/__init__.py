"""Overfitting-aware backtesting on top of purgedcv.

Layer 1 (this release): the statistics that decide whether a backtest can be
believed, and a trial log that makes counting trials the default.
"""

from .stats import (
    PBOResult,
    SharpeStats,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
    sharpe_stats,
)
from .trials import Trial, TrialLog

__version__ = "0.0.1"

__all__ = [
    "PBOResult",
    "SharpeStats",
    "Trial",
    "TrialLog",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "min_track_record_length",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "sharpe_ratio",
    "sharpe_stats",
]
