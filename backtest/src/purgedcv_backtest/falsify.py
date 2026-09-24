"""Falsification harness: try to break a strategy before believing it.

Each check targets a failure pattern from DESIGN.md:

* F3 (spurious edge): :func:`permutation_test`
* F2 (leaky feature): :func:`lag_profile`
* F5 (unrealistic costs): :func:`cost_sensitivity`, plus findings for free
  trading and unfunded shorts

:func:`falsify` runs them all and returns a :class:`Verdict`.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from .costs import CarryModel, CostModel, LinearCost, ZeroCost
from .engine import SimResult, _check_panel, _kernel, simulate
from .market import Calendar
from .stats import probabilistic_sharpe_ratio

__all__ = [
    "Status",
    "Finding",
    "Verdict",
    "PermutationResult",
    "LagProfile",
    "CostSensitivity",
    "permutation_test",
    "lag_profile",
    "cost_sensitivity",
    "falsify",
]

Array = npt.NDArray[np.float64]


def _sharpe(x: Array) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else 0.0


# --------------------------------------------------------------------------- #
# F3: circular-shift permutation test                                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PermutationResult:
    observed: float
    null: Array
    p_value: float


def permutation_test(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    costs: CostModel,
    carry: CarryModel | None = None,
    lag: int = 1,
    n_permutations: int = 1000,
    min_shift: int | None = None,
    seed: int | None = 0,
) -> PermutationResult:
    """Is the strategy's Sharpe ratio more than its signal's alignment with
    returns could produce by chance?

    Each permutation rotates the whole weight panel in time by a random
    offset against fixed returns. That keeps each series' own autocorrelation,
    turnover and cross-instrument structure, and breaks only the timing link
    between the signal and returns, which is the thing a real edge needs.
    ``min_shift`` (default ``n_obs // 10``) keeps rotations far enough apart
    that a persistent signal cannot stay aligned with itself.

    The p-value is ``(1 + #{null >= observed}) / (1 + n_permutations)``.
    """
    _check_panel(weights, returns)
    kernel = _kernel(returns, costs, carry, lag)
    w = weights.to_numpy(dtype=np.float64)
    T = w.shape[0]
    min_shift = max(1, T // 10) if min_shift is None else int(min_shift)
    if not 1 <= min_shift <= T - min_shift:
        raise ValueError(f"min_shift={min_shift} leaves no admissible rotation for {T} bars.")
    observed = _sharpe(kernel.run(w)[0])
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, T - min_shift + 1, size=n_permutations)
    null = np.array([_sharpe(kernel.run(np.roll(w, s, axis=0))[0]) for s in shifts])
    p = (1 + int(np.sum(null >= observed))) / (1 + n_permutations)
    return PermutationResult(observed=observed, null=null, p_value=p)


# --------------------------------------------------------------------------- #
# F2: lag cliff                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LagProfile:
    """Sharpe ratio by execution lag.

    Real information decays gradually with delay. A Sharpe ratio that is
    significant at the base lag and has mostly vanished one bar later is the
    signature of information that was not really available at decision time,
    such as a feature built from the bar it predicts. Fast, genuine signals can
    also decay quickly, so a cliff is a warning to verify data timing, not a
    rejection.
    """

    lags: tuple[int, ...]
    sharpes: tuple[float, ...]
    base_psr: float
    cliff_ratio: float
    flagged: bool


def lag_profile(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    costs: CostModel,
    carry: CarryModel | None = None,
    lag: int = 1,
    extra_lags: tuple[int, ...] = (1, 2, 4, 9),
    cliff_threshold: float = 0.25,
) -> LagProfile:
    _check_panel(weights, returns)
    lags = (lag,) + tuple(lag + e for e in extra_lags)
    w = weights.to_numpy(dtype=np.float64)
    nets = [_kernel(returns, costs, carry, L).run(w)[0] for L in lags]
    sharpes = tuple(_sharpe(n) for n in nets)
    base = sharpes[0]
    base_psr = probabilistic_sharpe_ratio(nets[0]) if nets[0].std() > 0 else 0.5
    ratio = sharpes[1] / base if base > 0 else math.nan
    flagged = base > 0 and base_psr > 0.95 and ratio < cliff_threshold
    return LagProfile(lags, sharpes, base_psr, ratio, bool(flagged))


# --------------------------------------------------------------------------- #
# F5: cost sensitivity                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostSensitivity:
    """Net Sharpe ratio as a flat proportional cost is raised.

    ``break_even_bps`` is where the net Sharpe ratio crosses zero, linearly
    interpolated on the grid: ``inf`` if the strategy survives the whole grid,
    ``0`` if it loses money even when trading is free.
    """

    bps: tuple[float, ...]
    sharpes: tuple[float, ...]
    break_even_bps: float


def cost_sensitivity(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    carry: CarryModel | None = None,
    lag: int = 1,
    bps_grid: tuple[float, ...] = (0, 1, 2, 5, 10, 20, 50, 100),
) -> CostSensitivity:
    _check_panel(weights, returns)
    w = weights.to_numpy(dtype=np.float64)
    grid = tuple(float(b) for b in sorted(bps_grid))
    sharpes = tuple(_sharpe(_kernel(returns, LinearCost(b), carry, lag).run(w)[0]) for b in grid)
    if sharpes[0] <= 0:
        be = 0.0
    else:
        be = math.inf
        for (b0, s0), (b1, s1) in zip(zip(grid, sharpes), zip(grid[1:], sharpes[1:])):
            if s1 <= 0 < s0:
                be = b0 + (b1 - b0) * s0 / (s0 - s1)
                break
    return CostSensitivity(grid, sharpes, be)


# --------------------------------------------------------------------------- #
# The verdict                                                                 #
# --------------------------------------------------------------------------- #
class Status(enum.Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class Finding:
    check: str
    status: Status
    detail: str


@dataclass(frozen=True)
class Verdict:
    """Outcome of :func:`falsify`: the findings first, the evidence after."""

    findings: tuple[Finding, ...]
    result: SimResult
    permutation: PermutationResult
    lags: LagProfile
    cost_curve: CostSensitivity
    extras: dict = field(default_factory=dict)

    @property
    def survived(self) -> bool:
        """No check failed. Warnings still need a human answer."""
        return all(f.status is not Status.FAIL for f in self.findings)

    def __str__(self) -> str:
        head = "SURVIVED" if self.survived else "FALSIFIED"
        lines = [f"Verdict: {head}"]
        lines += [f"  [{f.status.value.upper():4}] {f.check}: {f.detail}" for f in self.findings]
        return "\n".join(lines)


def falsify(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    costs: CostModel,
    calendar: Calendar,
    carry: CarryModel | None = None,
    lag: int = 1,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> Verdict:
    """Run every falsification check on one strategy and return the verdict.

    This is one trial's verdict. Selection across trials (F4) is judged by the
    :class:`~purgedcv_backtest.TrialLog` the trial is recorded in.
    """
    result = simulate(weights, returns, costs=costs, calendar=calendar, carry=carry, lag=lag)
    perm = permutation_test(
        weights, returns, costs=costs, carry=carry, lag=lag,
        n_permutations=n_permutations, seed=seed,
    )
    lags = lag_profile(weights, returns, costs=costs, carry=carry, lag=lag)
    curve = cost_sensitivity(weights, returns, carry=carry, lag=lag)

    findings = [
        Finding("F1 execution timing", Status.PASS, f"positions held from {lag} bar(s) after the decision"),
        Finding(
            "F3 permutation test",
            Status.PASS if perm.p_value < alpha else Status.FAIL,
            f"p = {perm.p_value:.4f} over {n_permutations} time rotations "
            f"(observed per-period Sharpe {perm.observed:.4f})",
        ),
        Finding(
            "F2 lag cliff",
            Status.WARN if lags.flagged else Status.PASS,
            (f"Sharpe keeps {lags.cliff_ratio:.0%} of its value with one more bar of delay; "
             f"verify every input was known at decision time")
            if lags.flagged
            else "Sharpe decays smoothly with execution delay",
        ),
    ]
    if isinstance(costs, ZeroCost):
        findings.append(Finding("F5 costs", Status.WARN, "costs are zero; supply a cost model"))
    net_sr = result.sharpe
    findings.append(
        Finding(
            "F5 net of costs",
            Status.PASS if net_sr > 0 else Status.FAIL,
            f"net per-period Sharpe {net_sr:.4f}; flat-cost break-even "
            f"{curve.break_even_bps:.1f} bps per unit turnover",
        )
    )
    if carry is None and (result.held.to_numpy() < 0).any():
        findings.append(
            Finding("F5 carry", Status.WARN,
                    "short positions held with no borrow or funding carry modelled")
        )
    return Verdict(tuple(findings), result, perm, lags, curve)
