"""Market descriptions: calendars and instruments.

These describe *what* is traded; how trading it costs money lives in
:mod:`purgedcv_backtest.costs`. Asset-class differences (24/7 crypto, exchange
sessions, futures multipliers) belong here and in the cost/carry models, never
in the engine.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

__all__ = [
    "AssetClass",
    "Calendar",
    "Instrument",
    "EQUITY_DAILY",
    "FUTURES_DAILY",
    "CRYPTO_DAILY",
    "CRYPTO_HOURLY",
]


class AssetClass(enum.Enum):
    EQUITY = "equity"
    FUTURE = "future"
    PERPETUAL = "perpetual"
    CRYPTO_SPOT = "crypto_spot"
    FX = "fx"
    OPTION = "option"
    OTHER = "other"


@dataclass(frozen=True)
class Calendar:
    """How many bars make a year; used only to annualise for display."""

    name: str
    periods_per_year: float

    def __post_init__(self) -> None:
        if not self.periods_per_year > 0:
            raise ValueError(f"periods_per_year must be > 0, got {self.periods_per_year!r}.")


EQUITY_DAILY = Calendar("equity-daily", 252)
FUTURES_DAILY = Calendar("futures-daily", 252)
CRYPTO_DAILY = Calendar("crypto-daily", 365)
CRYPTO_HOURLY = Calendar("crypto-hourly", 365 * 24)


@dataclass(frozen=True)
class Instrument:
    """A tradable instrument.

    ``fee_bps`` is the proportional fee per unit of traded notional, in basis
    points (commission plus half the typical spread, for example). Cost
    models can read it via :meth:`LinearCost.from_instruments`.
    """

    symbol: str
    asset_class: AssetClass
    calendar: Calendar
    currency: str = "USD"
    fee_bps: float | None = None
    multiplier: float = 1.0
