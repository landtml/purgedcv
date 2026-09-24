"""Experimental Rust-backed drop-in for :class:`purgedcv.CombinatorialPurgedCV`.

Only the per-split purge/embargo loop runs in Rust. Input validation, ``t1``
resolution, ``build_paths`` and the sklearn protocol are inherited unchanged
from the pure-Python class, so both produce identical splits and raise
identical errors.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Iterator

import numpy as np
import numpy.typing as npt
import pandas as pd
import purgedcv
from purgedcv import CPCVPaths, make_t1
from purgedcv._splitter import _end_positions, _sample_index

from ._engine import SplitPlan, num_threads

__all__ = ["CombinatorialPurgedCV", "CPCVPaths", "SplitPlan", "make_t1"]

# Below this many samples a split costs less than dispatching it to the thread
# pool (measured crossover ~15-20k on 4 cores), so split() stays serial.
_PARALLEL_MIN_SAMPLES = 20_000


class CombinatorialPurgedCV(purgedcv.CombinatorialPurgedCV):
    __doc__ = purgedcv.CombinatorialPurgedCV.__doc__

    def _plan(self, n: int, end_pos: npt.NDArray[np.int_]) -> SplitPlan:
        return SplitPlan(
            np.ascontiguousarray(end_pos, dtype=np.int64),
            self.n_groups,
            self._embargo_size(n),
            self.embargo_anchor == "label_end",
        )

    def _combos(self) -> Iterator[tuple[int, ...]]:
        return itertools.combinations(range(self.n_groups), self.n_test_groups)

    def _check_train_size(self, combo, train, n, end_pos) -> None:
        # Same message as the pure-Python splitter, so callers can't tell the
        # engines apart by their errors.
        if len(train) >= self.min_train_size:
            return
        horizon = int((end_pos - np.arange(n)).max())
        raise ValueError(
            f"split with test groups {combo} retains {len(train)} training "
            f"observation(s) after purge and embargo, below "
            f"min_train_size={self.min_train_size}. n_samples={n}, "
            f"longest label horizon={horizon} bars, "
            f"embargo={self._embargo_size(n)} bars. "
            f"Lengthen the sample, shorten the label horizon, reduce "
            f"embargo_pct or n_test_groups, or pass min_train_size=0 to "
            f"allow degenerate folds."
        )

    def _iter_splits(
        self, n: int, index: pd.Index, end_pos: npt.NDArray[np.int_]
    ) -> Iterator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
        plan = self._plan(n, end_pos)
        combos = self._combos()
        if n < _PARALLEL_MIN_SAMPLES:
            for combo in combos:
                train, test = plan.split(combo)
                self._check_train_size(combo, train, n, end_pos)
                yield train, test
            return
        # Lazy but parallel: compute a batch of splits across the thread pool,
        # yield it, then compute the next. At most one batch is held in memory,
        # and a degenerate fold still raises at the same split as before.
        batch_size = 2 * num_threads()
        while batch := list(itertools.islice(combos, batch_size)):
            for combo, (train, test) in zip(batch, plan.split_many(batch)):
                self._check_train_size(combo, train, n, end_pos)
                yield train, test

    def split_all(
        self, X, y=None, groups=None, t1: pd.Series | None = None
    ) -> list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
        """Every split at once, computed in parallel across threads.

        Trades the lazy generator of :meth:`split` for throughput: all
        ``C(N, k)`` train/test arrays are held in memory together.
        """
        if groups is not None:
            warnings.warn(
                "groups is accepted for sklearn API compatibility but is never "
                "used: CombinatorialPurgedCV forms its own contiguous time "
                "groups from n_groups. Pass t1 to control label lifespans.",
                UserWarning,
                stacklevel=2,
            )
        t1 = self.t1 if t1 is None else t1
        index = _sample_index(X)
        n = len(index)
        self._validate_n(n)
        end_pos = _end_positions(index, t1)
        combos = list(self._combos())
        out = self._plan(n, end_pos).split_many(combos)
        for combo, (train, _) in zip(combos, out):
            self._check_train_size(combo, train, n, end_pos)
        return out
