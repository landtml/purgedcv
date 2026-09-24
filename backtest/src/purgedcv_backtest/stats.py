"""Overfitting-aware performance statistics.

Every Sharpe ratio here is **per period** (not annualised) and every sample
size is a count of return observations, because that is the scale the
formulas below are derived on. Annualise only for display.

References
----------
* Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier",
  Journal of Risk -- probabilistic Sharpe ratio, minimum track record length.
* Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", Journal of
  Portfolio Management -- expected maximum Sharpe ratio, DSR.
* Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest
  Overfitting", Journal of Computational Finance -- CSCV and PBO.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

__all__ = [
    "SharpeStats",
    "PBOResult",
    "sharpe_stats",
    "sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "min_track_record_length",
    "probability_of_backtest_overfitting",
]

_NORMAL = NormalDist()
EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Moments                                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SharpeStats:
    """What every Sharpe-based test needs from a return series.

    ``kurtosis`` is Pearson (non-excess) kurtosis: 3 for a normal distribution.
    """

    sharpe: float
    n_obs: int
    skew: float
    kurtosis: float

    @property
    def variance_factor(self) -> float:
        """``1 - skew * SR + (kurtosis - 1) / 4 * SR**2``.

        ``(n_obs - 1)`` times the asymptotic variance of the Sharpe estimate.
        Negative skew and fat tails inflate it, which is how PSR, DSR and
        MinTRL penalise strategies that look smooth but crash.
        """
        sr = self.sharpe
        return 1.0 - self.skew * sr + (self.kurtosis - 1.0) / 4.0 * sr * sr


def _as_returns(returns) -> npt.NDArray[np.float64]:
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError(f"returns must be one-dimensional, got shape {r.shape}.")
    if r.size < 2:
        raise ValueError(f"need at least 2 return observations, got {r.size}.")
    if not np.isfinite(r).all():
        raise ValueError("returns contain NaN or infinite values; clean them first.")
    return r


def sharpe_stats(returns) -> SharpeStats:
    """Per-period Sharpe ratio, sample size, skewness and Pearson kurtosis."""
    r = _as_returns(returns)
    centred = r - r.mean()
    m2 = float(np.mean(centred**2))
    if m2 == 0.0:
        raise ValueError("returns are constant; the Sharpe ratio is undefined.")
    sd = float(r.std(ddof=1))
    return SharpeStats(
        sharpe=float(r.mean()) / sd,
        n_obs=r.size,
        skew=float(np.mean(centred**3)) / m2**1.5,
        kurtosis=float(np.mean(centred**4)) / m2**2,
    )


def sharpe_ratio(returns) -> float:
    """Per-period Sharpe ratio, ``mean / std`` (sample std, ``ddof=1``)."""
    return sharpe_stats(returns).sharpe


# --------------------------------------------------------------------------- #
# PSR, DSR, MinTRL                                                            #
# --------------------------------------------------------------------------- #
def _stats(x: SharpeStats | npt.ArrayLike) -> SharpeStats:
    return x if isinstance(x, SharpeStats) else sharpe_stats(x)


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """Probability that the true Sharpe ratio exceeds ``sr_benchmark``.

    ``returns`` may be a return series or precomputed :class:`SharpeStats`.
    Accounts for sample length, skewness and kurtosis, but not for how many
    strategies were tried before this one -- that is what
    :func:`deflated_sharpe_ratio` adds.
    """
    s = _stats(returns)
    factor = s.variance_factor
    if factor <= 0.0:
        raise ValueError(
            f"Sharpe variance factor is {factor:.4g} <= 0 (skew={s.skew:.3g}, "
            f"kurtosis={s.kurtosis:.3g}); the sample moments are too extreme "
            f"for the asymptotic approximation."
        )
    z = (s.sharpe - sr_benchmark) * math.sqrt(s.n_obs - 1) / math.sqrt(factor)
    return _NORMAL.cdf(z)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe ratio among ``n_trials`` skill-less strategies.

    ``sr_variance`` is the cross-sectional variance of the trials' per-period
    Sharpe ratios. This is the Sharpe ratio pure selection would produce, and
    so the bar the chosen strategy has to clear.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials!r}.")
    if sr_variance < 0:
        raise ValueError(f"sr_variance must be >= 0, got {sr_variance!r}.")
    if n_trials == 1:
        return 0.0  # nothing was selected
    return math.sqrt(sr_variance) * (
        (1.0 - EULER_GAMMA) * _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
        + EULER_GAMMA * _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe_ratio(returns, *, n_trials: int, sr_variance: float) -> float:
    """Probability that the selected strategy's true Sharpe ratio is positive,
    after correcting for the ``n_trials`` strategies tried to find it.

    ``n_trials`` and ``sr_variance`` are keyword-only and have no defaults on
    purpose: leaving out the trials that were tried is the most common way a
    backtest overstates its result. :class:`~purgedcv_backtest.TrialLog`
    supplies both from its record of every trial.
    """
    return probabilistic_sharpe_ratio(returns, expected_max_sharpe(n_trials, sr_variance))


def min_track_record_length(
    returns, sr_benchmark: float = 0.0, alpha: float = 0.05
) -> float:
    """Observations needed for ``PSR(sr_benchmark) >= 1 - alpha``.

    Returns ``inf`` when the observed Sharpe ratio does not exceed the
    benchmark: no amount of history makes it significant. Compare with the
    series' own ``n_obs`` -- a shorter series has not earned its Sharpe ratio.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}.")
    s = _stats(returns)
    if s.sharpe <= sr_benchmark:
        return math.inf
    z = _NORMAL.inv_cdf(1.0 - alpha)
    return 1.0 + s.variance_factor * (z / (s.sharpe - sr_benchmark)) ** 2


# --------------------------------------------------------------------------- #
# PBO via combinatorially symmetric cross-validation                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PBOResult:
    """Outcome of CSCV over a ``(n_obs, n_trials)`` matrix of trial returns.

    Attributes
    ----------
    pbo : float
        Share of in-/out-of-sample splits where the in-sample winner ranks at
        or below the out-of-sample median. About 0.5 means selection is
        noise; near 0 means the in-sample winner reliably holds up.
    logits : ndarray
        Per split, ``log(w / (1 - w))`` of the winner's relative
        out-of-sample rank ``w``. ``pbo`` is the share ``<= 0``.
    is_sharpe, oos_sharpe : ndarray
        Per split, the in-sample winner's Sharpe ratio in and out of sample:
        the data for a performance-degradation plot.
    """

    pbo: float
    logits: npt.NDArray[np.float64]
    is_sharpe: npt.NDArray[np.float64]
    oos_sharpe: npt.NDArray[np.float64]

    @property
    def prob_oos_loss(self) -> float:
        """Share of splits where the in-sample winner loses money out of sample."""
        return float(np.mean(self.oos_sharpe < 0))


def probability_of_backtest_overfitting(
    trial_returns, n_splits: int = 16
) -> PBOResult:
    """Probability of backtest overfitting by CSCV (Bailey et al., 2017).

    ``trial_returns`` is ``(n_obs, n_trials)``: one column per configuration
    tried, all on the same dates. The rows are cut into ``n_splits``
    contiguous blocks, and every choice of half of them serves once as the
    in-sample set, with the other half out of sample. That is
    ``C(n_splits, n_splits / 2)`` splits.

    Sharpe ratios for every split come from per-block sums and sums of
    squares, so the cost does not grow with ``n_obs``.
    """
    m = np.asarray(trial_returns, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"trial_returns must be 2-D (n_obs, n_trials), got shape {m.shape}.")
    n_obs, n_trials = m.shape
    if n_trials < 2:
        raise ValueError(f"PBO compares trials; need at least 2, got {n_trials}.")
    if n_splits < 2 or n_splits % 2:
        raise ValueError(f"n_splits must be an even number >= 2, got {n_splits!r}.")
    if n_obs < 2 * n_splits:
        raise ValueError(
            f"need at least 2 observations per block: n_obs={n_obs} < 2 * n_splits={2 * n_splits}."
        )
    if not np.isfinite(m).all():
        raise ValueError("trial_returns contain NaN or infinite values.")

    blocks = np.array_split(np.arange(n_obs), n_splits)
    count = np.array([len(b) for b in blocks], dtype=np.float64)[:, None]
    total = np.stack([m[b].sum(axis=0) for b in blocks])
    total_sq = np.stack([(m[b] ** 2).sum(axis=0) for b in blocks])

    combos = np.array(list(itertools.combinations(range(n_splits), n_splits // 2)))
    in_sample = np.zeros((len(combos), n_splits), dtype=bool)
    in_sample[np.arange(len(combos))[:, None], combos] = True

    def sharpe(mask):
        w = mask.astype(np.float64)
        n, s, ss = w @ count, w @ total, w @ total_sq
        var = (ss - s * s / n) / (n - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(var > 0, (s / n) / np.sqrt(var), 0.0)

    is_sr, oos_sr = sharpe(in_sample), sharpe(~in_sample)
    rows = np.arange(len(combos))
    best = is_sr.argmax(axis=1)
    best_oos = oos_sr[rows, best][:, None]
    # Average rank for ties, 1 = worst: count strictly worse, half the ties.
    rank = 1.0 + (oos_sr < best_oos).sum(axis=1) + 0.5 * ((oos_sr == best_oos).sum(axis=1) - 1)
    w = rank / (n_trials + 1)
    logits = np.log(w / (1.0 - w))
    return PBOResult(
        pbo=float(np.mean(logits <= 0)),
        logits=logits,
        is_sharpe=is_sr[rows, best],
        oos_sharpe=best_oos[:, 0],
    )
