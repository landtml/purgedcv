"""Vectorised backtest engine.

Conventions (DESIGN.md, section 3):

* Row ``t`` of every panel is the bar ending at ``t``. A target weight
  decided with information up to the end of bar ``t`` is held from bar
  ``t + lag`` on, with ``lag >= 1`` -- trading on the bar whose return you
  earn (``lag = 0``) is refused, not warned about (failure pattern F1).
* Weights are signed fractions of equity, returns are simple per-bar
  returns, and every cost and carry charge is in the same units, so
  ``net = sum(held * returns) - costs - carry``.
* Held weights are reset to target every bar. Turnover is
  ``sum(|held_t - held_{t-1}|)``, starting from flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .costs import Bound, CarryModel, CostModel, ZeroCost
from .market import Calendar
from .stats import SharpeStats, sharpe_stats

__all__ = ["SimResult", "PathResults", "simulate", "cpcv_path_returns"]

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SimResult:
    """Per-bar outcome of one simulation, plus the assumptions behind it."""

    net: pd.Series
    gross: pd.Series
    costs: pd.Series
    carry: pd.Series
    turnover: pd.Series
    held: pd.DataFrame
    calendar: Calendar
    assumptions: dict[str, Any] = field(default_factory=dict)

    @property
    def stats(self) -> SharpeStats:
        return sharpe_stats(self.net.to_numpy())

    @property
    def sharpe(self) -> float:
        """Per-period Sharpe ratio of net returns."""
        return self.stats.sharpe

    @property
    def annualized_sharpe(self) -> float:
        return self.sharpe * float(np.sqrt(self.calendar.periods_per_year))

    def equity(self) -> pd.Series:
        """Compounded equity curve, starting at 1."""
        return (1.0 + self.net).cumprod()


# --------------------------------------------------------------------------- #
# The kernel: pure arrays, bound models                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Kernel:
    """A panel with models bound once, re-runnable on many weight arrays."""

    returns: Array
    cost: Bound
    carry: Bound | None
    lag: int

    def run(self, weights: Array) -> tuple[Array, Array, Array, Array, Array, Array]:
        T = weights.shape[0]
        held = np.zeros_like(weights)
        if self.lag < T:
            held[self.lag :] = weights[: T - self.lag]
        exposed = held != 0
        if np.isnan(self.returns[exposed]).any():
            raise ValueError("returns are NaN on a bar where a position is held.")
        gross = np.where(exposed, held * np.nan_to_num(self.returns), 0.0).sum(axis=1)
        trades = np.diff(held, axis=0, prepend=0.0)
        costs = self.cost(trades)
        carry = self.carry(held) if self.carry is not None else np.zeros(T)
        net = gross - costs - carry
        return net, gross, costs, carry, np.abs(trades).sum(axis=1), held


def _check_panel(weights: pd.DataFrame, returns: pd.DataFrame) -> None:
    if not isinstance(weights, pd.DataFrame) or not isinstance(returns, pd.DataFrame):
        raise TypeError("weights and returns must be pandas DataFrames (bars x instruments).")
    if not weights.index.equals(returns.index) or not weights.columns.equals(returns.columns):
        raise ValueError(
            "weights and returns must share the same index and columns; align them "
            "explicitly (the engine does not reindex, so nothing is silently filled)."
        )
    if not weights.index.is_monotonic_increasing or not weights.index.is_unique:
        raise ValueError("the panel index must be sorted ascending and unique.")
    if not np.isfinite(weights.to_numpy(dtype=np.float64)).all():
        raise ValueError("weights contain NaN or infinite values; use 0 for no position.")


def _kernel(returns: pd.DataFrame, costs: CostModel, carry: CarryModel | None, lag: int) -> _Kernel:
    if not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValueError(
            f"lag must be an integer >= 1, got {lag!r}. lag=0 trades on the bar "
            f"whose return it earns: same-bar look-ahead (failure pattern F1)."
        )
    if not isinstance(costs, CostModel):
        raise TypeError("costs must be a CostModel; pass ZeroCost() to trade for free explicitly.")
    index, columns = returns.index, returns.columns
    return _Kernel(
        returns=returns.to_numpy(dtype=np.float64),
        cost=costs.bind(index, columns),
        carry=carry.bind(index, columns) if carry is not None else None,
        lag=int(lag),
    )


def simulate(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    costs: CostModel,
    calendar: Calendar,
    carry: CarryModel | None = None,
    lag: int = 1,
) -> SimResult:
    """Run target ``weights`` against ``returns`` (both bars x instruments).

    ``costs`` and ``calendar`` have no defaults: free trading has to be asked
    for (``ZeroCost()``), and is then reported by :func:`falsify`.
    """
    _check_panel(weights, returns)
    kernel = _kernel(returns, costs, carry, lag)
    net, gross, cost, carry_, turnover, held = kernel.run(weights.to_numpy(dtype=np.float64))
    index = returns.index
    return SimResult(
        net=pd.Series(net, index=index, name="net"),
        gross=pd.Series(gross, index=index, name="gross"),
        costs=pd.Series(cost, index=index, name="costs"),
        carry=pd.Series(carry_, index=index, name="carry"),
        turnover=pd.Series(turnover, index=index, name="turnover"),
        held=pd.DataFrame(held, index=index, columns=returns.columns),
        calendar=calendar,
        assumptions={
            "lag": int(lag),
            "costs": repr(costs),
            "zero_cost": isinstance(costs, ZeroCost),
            "carry": repr(carry) if carry is not None else None,
        },
    )


# --------------------------------------------------------------------------- #
# CPCV paths (failure pattern F6)                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PathResults:
    """One :class:`SimResult` per CPCV backtest path.

    The paths share training data and overlap in time, so their Sharpe
    ratios are correlated: the spread across paths is *not* a confidence
    interval. Use it to see path dependence; use the trial log's DSR and
    PBO for significance.
    """

    paths: tuple[SimResult, ...]

    @property
    def net(self) -> pd.DataFrame:
        return pd.concat({f"path_{p}": r.net for p, r in enumerate(self.paths)}, axis=1)

    @property
    def sharpes(self) -> npt.NDArray[np.float64]:
        return np.array([r.sharpe for r in self.paths])

    def summary(self) -> dict[str, float]:
        s = self.sharpes
        return {
            "n_paths": len(s),
            "sharpe_mean": float(s.mean()),
            "sharpe_min": float(s.min()),
            "sharpe_median": float(np.median(s)),
            "sharpe_max": float(s.max()),
            "share_positive": float(np.mean(s > 0)),
        }


def cpcv_path_returns(
    paths,
    sim_weights: npt.ArrayLike,
    returns: pd.DataFrame,
    *,
    costs: CostModel,
    calendar: Calendar,
    carry: CarryModel | None = None,
    lag: int = 1,
) -> PathResults:
    """Assemble each CPCV path's positions, then simulate each path on its own.

    ``paths`` is a ``purgedcv.CPCVPaths``. ``sim_weights`` has shape
    ``(n_obs, n_sims, n_instruments)``, or ``(n_obs, n_sims)`` for one
    instrument: the target weights each simulation's model produced. Only a
    simulation's test rows are ever read.

    Returns and costs are computed per path from that path's own position
    series, including the trades where one simulation's segment hands over
    to the next. Returns are never computed once and reused across paths,
    which is the double-counting bug this layering exists to prevent.
    """
    w = np.asarray(sim_weights, dtype=np.float64)
    if w.ndim == 2:
        w = w[:, :, None]
    n_obs, n_sims, n_inst = w.shape
    if (n_obs, n_sims) != paths.is_test.shape:
        raise ValueError(
            f"sim_weights has (n_obs, n_sims) = {(n_obs, n_sims)} but the path map "
            f"is {paths.is_test.shape}."
        )
    if returns.shape != (n_obs, n_inst):
        raise ValueError(f"returns has shape {returns.shape}, expected {(n_obs, n_inst)}.")
    if not np.isfinite(w[paths.is_test]).all():
        raise ValueError("sim_weights contain NaN or infinite values on test rows.")
    kernel = _kernel(returns, costs, carry, lag)
    rows = np.arange(n_obs)
    results = []
    for p in range(paths.n_paths):
        path_w = w[rows, paths.paths[:, p], :]
        net, gross, cost, carry_, turnover, held = kernel.run(path_w)
        index = returns.index
        results.append(
            SimResult(
                net=pd.Series(net, index=index, name=f"path_{p}"),
                gross=pd.Series(gross, index=index),
                costs=pd.Series(cost, index=index),
                carry=pd.Series(carry_, index=index),
                turnover=pd.Series(turnover, index=index),
                held=pd.DataFrame(held, index=index, columns=returns.columns),
                calendar=calendar,
                assumptions={"lag": int(lag), "costs": repr(costs),
                             "zero_cost": isinstance(costs, ZeroCost),
                             "carry": repr(carry) if carry is not None else None, "path": p},
            )
        )
    return PathResults(tuple(results))
