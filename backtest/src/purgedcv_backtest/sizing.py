"""Sizers: model predictions to target weights."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

__all__ = ["Sizer", "SignSizer", "ProbabilitySizer"]


@runtime_checkable
class Sizer(Protocol):
    def __call__(self, predictions: pd.DataFrame) -> pd.DataFrame: ...


def _normalise_gross(w: pd.DataFrame, gross: float) -> pd.DataFrame:
    total = w.abs().sum(axis=1)
    scale = np.where(total > 0, gross / total.where(total > 0, 1.0), 0.0)
    return w.mul(scale, axis=0)


@dataclass(frozen=True)
class SignSizer:
    """Long where the prediction is positive, short where negative, equally
    weighted, with gross exposure ``gross`` on every bar that has a view."""

    gross: float = 1.0

    def __call__(self, predictions: pd.DataFrame) -> pd.DataFrame:
        return _normalise_gross(np.sign(predictions).astype(float), self.gross)


@dataclass(frozen=True)
class ProbabilitySizer:
    """Bet sizing from a predicted probability of an up move (AFML ch. 10).

    With ``p`` the probability the next move is up, the test statistic
    ``z = (p - 0.5) / sqrt(p (1 - p))`` gives a size ``2 * Phi(z) - 1`` in
    ``(-1, 1)``: zero for ``p = 0.5``, approaching full size only for
    confident predictions. Each instrument's weight is that size times
    ``max_weight``.
    """

    max_weight: float = 1.0

    def __call__(self, predictions: pd.DataFrame) -> pd.DataFrame:
        p = predictions.to_numpy(dtype=np.float64)
        if ((p < 0) | (p > 1)).any():
            raise ValueError("ProbabilitySizer expects probabilities in [0, 1].")
        p = np.clip(p, 1e-12, 1 - 1e-12)
        z = (p - 0.5) / np.sqrt(p * (1 - p))
        size = _erf(z / math.sqrt(2.0))  # == 2 * Phi(z) - 1
        return pd.DataFrame(
            size * self.max_weight, index=predictions.index, columns=predictions.columns
        )


_erf = np.vectorize(math.erf, otypes=[np.float64])
