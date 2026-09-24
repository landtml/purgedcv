"""Overfitting-aware backtesting on top of purgedcv.

Layer 1: the statistics that decide whether a backtest can be believed, and
a trial log that makes counting trials the default.

Layer 2: the engine core -- calendars and instruments, cost/carry/sizing
interfaces, the vectorised engine, CPCV path returns, and the falsification
harness. See DESIGN.md for the failure patterns each piece guards against.
"""

from .costs import (
    BorrowCarry,
    CarryModel,
    CompositeCost,
    CostModel,
    FundingCarry,
    LinearCost,
    SquareRootImpact,
    ZeroCost,
)
from .engine import PathResults, SimResult, cpcv_path_returns, simulate
from .falsify import (
    CostSensitivity,
    Finding,
    LagProfile,
    PermutationResult,
    Status,
    Verdict,
    cost_sensitivity,
    falsify,
    lag_profile,
    permutation_test,
)
from .market import (
    CRYPTO_DAILY,
    CRYPTO_HOURLY,
    EQUITY_DAILY,
    FUTURES_DAILY,
    AssetClass,
    Calendar,
    Instrument,
)
from .report import SCHEMA, build_report, write_report
from .sizing import ProbabilitySizer, SignSizer, Sizer
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

__version__ = "0.0.2"

__all__ = [
    "SCHEMA",
    "build_report",
    "write_report",
    "AssetClass",
    "BorrowCarry",
    "CRYPTO_DAILY",
    "CRYPTO_HOURLY",
    "Calendar",
    "CarryModel",
    "CompositeCost",
    "CostModel",
    "CostSensitivity",
    "EQUITY_DAILY",
    "FUTURES_DAILY",
    "Finding",
    "FundingCarry",
    "Instrument",
    "LagProfile",
    "LinearCost",
    "PathResults",
    "PermutationResult",
    "ProbabilitySizer",
    "SignSizer",
    "SimResult",
    "Sizer",
    "SquareRootImpact",
    "Status",
    "Verdict",
    "ZeroCost",
    "cost_sensitivity",
    "cpcv_path_returns",
    "falsify",
    "lag_profile",
    "permutation_test",
    "simulate",
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
