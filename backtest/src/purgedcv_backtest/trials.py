"""Trial accounting: every configuration tried counts against the result."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import numpy as np
import numpy.typing as npt

from .stats import (
    PBOResult,
    SharpeStats,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_stats,
)

__all__ = ["Trial", "TrialLog"]


@dataclass(frozen=True)
class Trial:
    """One recorded configuration and its out-of-sample returns (read-only)."""

    number: int
    name: str
    params: Mapping[str, Any]
    returns: npt.NDArray[np.float64]
    stats: SharpeStats

    @property
    def sharpe(self) -> float:
        return self.stats.sharpe


class TrialLog:
    """Append-only record of every strategy configuration evaluated.

    The deflated Sharpe ratio and PBO are only as honest as the trial count
    behind them, so this log has no way to remove, replace or edit a trial:
    record everything you evaluate, including the configurations you
    abandoned. Recorded returns are copied and frozen.

    Trials are often correlated (neighbouring parameter values), which makes
    the effective number of independent trials smaller than ``len(log)``.
    Using the raw count is therefore conservative: it can only make the
    deflated Sharpe ratio harsher, never more flattering.

    DSR also depends on the dispersion of the trials' Sharpe ratios, which is
    re-estimated from the log, so one extra trial can nudge a DSR up if it
    shrinks that dispersion. At fixed dispersion, every trial lowers it.
    """

    def __init__(self) -> None:
        self._trials: list[Trial] = []

    # -- recording --------------------------------------------------------- #
    def record(self, returns, name: str | None = None, **params: Any) -> Trial:
        """Record a trial's out-of-sample returns and the parameters that made them."""
        r = np.array(returns, dtype=np.float64)  # copy, then freeze
        stats = sharpe_stats(r)
        r.setflags(write=False)
        trial = Trial(
            number=len(self._trials),
            name=name if name is not None else f"trial_{len(self._trials)}",
            params=MappingProxyType(dict(params)),
            returns=r,
            stats=stats,
        )
        self._trials.append(trial)
        return trial

    # -- read access ------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._trials)

    def __iter__(self) -> Iterator[Trial]:
        return iter(tuple(self._trials))

    def __getitem__(self, number: int) -> Trial:
        return self._trials[number]

    def __repr__(self) -> str:
        return f"TrialLog(n_trials={len(self)})"

    @property
    def n_trials(self) -> int:
        return len(self._trials)

    def _require(self, minimum: int, what: str) -> None:
        if len(self._trials) < minimum:
            raise ValueError(f"{what} needs at least {minimum} recorded trial(s), have {len(self)}.")

    def sharpe_variance(self) -> float:
        """Cross-sectional variance of the recorded trials' Sharpe ratios."""
        self._require(1, "sharpe_variance")
        if len(self._trials) == 1:
            return 0.0
        return float(np.var([t.sharpe for t in self._trials], ddof=1))

    def best(self) -> Trial:
        """The trial with the highest Sharpe ratio -- the one selection would pick."""
        self._require(1, "best")
        return max(self._trials, key=lambda t: t.sharpe)

    # -- the honest statistics --------------------------------------------- #
    def expected_max_sharpe(self) -> float:
        """The Sharpe ratio the best of these trials would reach by luck alone."""
        return expected_max_sharpe(len(self), self.sharpe_variance())

    def deflated_sharpe(self, trial: Trial | int | None = None) -> float:
        """DSR of ``trial`` (default: the best), deflated by every recorded trial."""
        self._require(1, "deflated_sharpe")
        t = self._resolve(trial)
        return deflated_sharpe_ratio(
            t.stats, n_trials=len(self), sr_variance=self.sharpe_variance()
        )

    def pbo(self, n_splits: int = 16) -> PBOResult:
        """Probability of backtest overfitting across all recorded trials.

        Requires every trial's returns to cover the same observations.
        """
        self._require(2, "pbo")
        lengths = {t.returns.size for t in self._trials}
        if len(lengths) != 1:
            raise ValueError(
                f"pbo needs every trial on the same observations; got return "
                f"lengths {sorted(lengths)}."
            )
        matrix = np.column_stack([t.returns for t in self._trials])
        return probability_of_backtest_overfitting(matrix, n_splits=n_splits)

    def summary(self, trial: Trial | int | None = None, alpha: float = 0.05) -> dict[str, Any]:
        """Headline numbers for ``trial`` (default: the best), all per period."""
        self._require(1, "summary")
        t = self._resolve(trial)
        return {
            "trial": t.name,
            "params": dict(t.params),
            "n_trials": len(self),
            "n_obs": t.stats.n_obs,
            "sharpe": t.sharpe,
            "psr": probabilistic_sharpe_ratio(t.stats),
            "expected_max_sharpe": self.expected_max_sharpe(),
            "dsr": self.deflated_sharpe(t),
            "min_track_record_length": min_track_record_length(t.stats, alpha=alpha),
        }

    def _resolve(self, trial: Trial | int | None) -> Trial:
        if trial is None:
            return self.best()
        if isinstance(trial, Trial):
            if trial.number >= len(self._trials) or self._trials[trial.number] is not trial:
                raise ValueError("trial was not recorded in this log.")
            return trial
        return self._trials[trial]
