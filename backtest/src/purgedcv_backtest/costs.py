"""Cost and carry models: what trading and holding positions costs.

All charges are in the engine's units: a fraction of equity per bar, the same
units as the portfolio return they are subtracted from.

A model is configured with its own data (volatility, volume, funding rates,
...) and *bound* to a panel's dates and instruments once per simulation. The
bound function then works on plain arrays, so re-running a strategy
thousands of times (permutation tests) costs no pandas overhead.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from .market import Calendar, Instrument

__all__ = [
    "CostModel",
    "CarryModel",
    "ZeroCost",
    "LinearCost",
    "SquareRootImpact",
    "CompositeCost",
    "FundingCarry",
    "BorrowCarry",
]

Array = npt.NDArray[np.float64]
Bound = Callable[[Array], Array]


@runtime_checkable
class CostModel(Protocol):
    """Charge for trading. ``bind(...)(trades)`` maps the ``(T, A)`` weight
    changes to a ``(T,)`` per-bar charge."""

    def bind(self, index: pd.Index, columns: pd.Index) -> Bound: ...


@runtime_checkable
class CarryModel(Protocol):
    """Charge for holding. ``bind(...)(held)`` maps the ``(T, A)`` held
    weights to a ``(T,)`` per-bar charge (negative means income)."""

    def bind(self, index: pd.Index, columns: pd.Index) -> Bound: ...


def _per_instrument(value: float | Mapping[str, float], columns: pd.Index, what: str) -> Array:
    if isinstance(value, Mapping):
        missing = [c for c in columns if c not in value]
        if missing:
            raise ValueError(f"{what} missing for instruments {missing}.")
        return np.array([float(value[c]) for c in columns])
    return np.full(len(columns), float(value))


def _panel(df: pd.DataFrame, index: pd.Index, columns: pd.Index, what: str) -> Array:
    """``df`` aligned to the simulation panel; refuses to invent missing data."""
    missing_cols = columns.difference(df.columns)
    if len(missing_cols):
        raise ValueError(f"{what} has no data for instruments {list(missing_cols)}.")
    aligned = df.reindex(index=index, columns=columns)
    if aligned.isna().to_numpy().any():
        raise ValueError(
            f"{what} does not cover every bar of the simulation panel; "
            f"supply it for the full index (forward-fill explicitly if intended)."
        )
    return aligned.to_numpy(dtype=np.float64)


@dataclass(frozen=True)
class ZeroCost:
    """Free trading. Allowed, but reported as a finding by :func:`falsify`."""

    def bind(self, index: pd.Index, columns: pd.Index) -> Bound:
        return lambda trades: np.zeros(trades.shape[0])


@dataclass(frozen=True)
class LinearCost:
    """Proportional cost: ``bps / 1e4`` per unit of traded weight.

    ``bps`` is one rate for every instrument or a mapping per symbol.
    """

    bps: float | Mapping[str, float]

    @classmethod
    def from_instruments(cls, instruments: Iterable[Instrument]) -> LinearCost:
        rates = {}
        for inst in instruments:
            if inst.fee_bps is None:
                raise ValueError(f"instrument {inst.symbol!r} has no fee_bps.")
            rates[inst.symbol] = inst.fee_bps
        return cls(rates)

    def bind(self, index: pd.Index, columns: pd.Index) -> Bound:
        rate = _per_instrument(self.bps, columns, "bps") / 1e4
        if (rate < 0).any():
            raise ValueError("bps must be >= 0.")
        return lambda trades: np.abs(trades) @ rate


@dataclass(frozen=True)
class SquareRootImpact:
    """Half-spread plus square-root market impact.

    Per instrument and bar, trading ``|dw|`` of equity costs
    ``|dw| * (half_spread + coef * sigma * sqrt(|dw| * capital / adv))``.
    ``sigma`` is per-bar return volatility and ``adv`` the traded notional
    per bar, both panels over the simulation's dates. Capacity enters
    through ``capital``: the same strategy costs more when it trades more
    money.
    """

    volatility: pd.DataFrame
    adv: pd.DataFrame
    capital: float
    coef: float = 1.0
    half_spread_bps: float | Mapping[str, float] = 0.0

    def bind(self, index: pd.Index, columns: pd.Index) -> Bound:
        if not self.capital > 0:
            raise ValueError(f"capital must be > 0, got {self.capital!r}.")
        sigma = _panel(self.volatility, index, columns, "volatility")
        adv = _panel(self.adv, index, columns, "adv")
        if (adv <= 0).any():
            raise ValueError("adv must be > 0 everywhere.")
        spread = _per_instrument(self.half_spread_bps, columns, "half_spread_bps") / 1e4
        coef, capital = self.coef, self.capital

        def cost(trades: Array) -> Array:
            size = np.abs(trades)
            impact = coef * sigma * np.sqrt(size * capital / adv)
            return (size * (spread + impact)).sum(axis=1)

        return cost


@dataclass(frozen=True)
class CompositeCost:
    """Sum of several cost models (e.g. fees plus impact)."""

    models: tuple[CostModel, ...] = field(default_factory=tuple)

    def __init__(self, *models: CostModel) -> None:
        object.__setattr__(self, "models", tuple(models))

    def bind(self, index: pd.Index, columns: pd.Index) -> Bound:
        bound = [m.bind(index, columns) for m in self.models]
        return lambda trades: sum((b(trades) for b in bound), np.zeros(trades.shape[0]))


@dataclass(frozen=True)
class FundingCarry:
    """Perpetual-swap funding: per-bar rates, paid by longs, received by shorts.

    ``rates`` is a ``(T, A)`` panel of the funding rate applying to each bar
    (already scaled to the bar, e.g. an 8-hourly rate spread over its bars).
    """

    rates: pd.DataFrame

    def bind(self, index: pd.Index, columns: pd.Index) -> Bound:
        rates = _panel(self.rates, index, columns, "funding rates")
        return lambda held: (held * rates).sum(axis=1)


@dataclass(frozen=True)
class BorrowCarry:
    """Short-borrow cost: shorts pay ``annual_rate`` pro rata per bar."""

    annual_rate: float | Mapping[str, float]
    calendar: Calendar

    def bind(self, index: pd.Index, columns: pd.Index) -> Bound:
        per_bar = _per_instrument(self.annual_rate, columns, "annual_rate") / self.calendar.periods_per_year
        return lambda held: np.clip(-held, 0.0, None) @ per_bar
