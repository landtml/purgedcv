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

from ._engine import SplitPlan, build_paths as _build_paths, num_threads

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

    def _check_train_size(self, combo, n_train: int, n, end_pos) -> None:
        # Same message as the pure-Python splitter, so callers can't tell the
        # engines apart by their errors.
        if n_train >= self.min_train_size:
            return
        horizon = int((end_pos - np.arange(n)).max())
        raise ValueError(
            f"split with test groups {combo} retains {n_train} training "
            f"observation(s) after purge and embargo, below "
            f"min_train_size={self.min_train_size}. n_samples={n}, "
            f"longest label horizon={horizon} bars, "
            f"embargo={self._embargo_size(n)} bars. "
            f"Lengthen the sample, shorten the label horizon, reduce "
            f"embargo_pct or n_test_groups, or pass min_train_size=0 to "
            f"allow degenerate folds."
        )

    def _resolve(self, X, groups, t1) -> tuple[int, npt.NDArray[np.int_]]:
        """What :meth:`split` validates and resolves before its first fold."""
        if groups is not None:
            warnings.warn(
                "groups is accepted for sklearn API compatibility but is never "
                "used: CombinatorialPurgedCV forms its own contiguous time "
                "groups from n_groups. Pass t1 to control label lifespans.",
                UserWarning,
                stacklevel=3,  # past _resolve and the public method
            )
        t1 = self.t1 if t1 is None else t1
        index = _sample_index(X)
        n = len(index)
        self._validate_n(n)
        return n, _end_positions(index, t1)

    def _iter_splits(
        self, n: int, index: pd.Index, end_pos: npt.NDArray[np.int_]
    ) -> Iterator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
        plan = self._plan(n, end_pos)
        combos = self._combos()
        if n < _PARALLEL_MIN_SAMPLES:
            for combo in combos:
                train, test = plan.split(combo)
                self._check_train_size(combo, len(train), n, end_pos)
                yield train, test
            return
        # Lazy but parallel: compute a batch of splits across the thread pool,
        # yield it, then compute the next. At most one batch is held in memory,
        # and a degenerate fold still raises at the same split as before.
        batch_size = 2 * num_threads()
        while batch := list(itertools.islice(combos, batch_size)):
            for combo, (train, test) in zip(batch, plan.split_many(batch)):
                self._check_train_size(combo, len(train), n, end_pos)
                yield train, test

    def split_all(
        self, X, y=None, groups=None, t1: pd.Series | None = None
    ) -> list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
        """Every split at once, computed in parallel across threads.

        Trades the lazy generator of :meth:`split` for throughput: all
        ``C(N, k)`` train/test arrays are held in memory together.
        """
        n, end_pos = self._resolve(X, groups, t1)
        combos = list(self._combos())
        out = self._plan(n, end_pos).split_many(combos)
        for combo, (train, _) in zip(combos, out):
            self._check_train_size(combo, len(train), n, end_pos)
        return out

    def masks(
        self, X, y=None, groups=None, t1: pd.Series | None = None
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Boolean ``(n_samples, n_sims)`` train and test membership matrices.

        Column ``c`` corresponds to the ``c``-th split yielded by :meth:`split`:
        ``np.flatnonzero(train[:, c])`` equals its ``train_idx``. Fixed-shape
        masks suit batched training across every split at once (e.g. JAX
        ``vmap`` over sample weights) where variable-length index arrays do not.
        Columns are built in parallel and returned Fortran-ordered, so each
        split's column is contiguous. Degenerate folds raise as in :meth:`split`.
        """
        n, end_pos = self._resolve(X, groups, t1)
        combos = list(self._combos())
        train, test = self._plan(n, end_pos).masks(combos)
        train = train.reshape(len(combos), n).T
        test = test.reshape(len(combos), n).T
        for combo, n_train in zip(combos, train.sum(axis=0)):
            self._check_train_size(combo, int(n_train), n, end_pos)
        return train, test

    def build_paths(self, X, t1: pd.Series | None = None) -> CPCVPaths:
        index = _sample_index(X)
        n = len(index)
        self._validate_n(n)
        combos = list(self._combos())
        n_sims, n_paths = len(combos), self.get_n_paths()
        is_test, paths, path_folds = _build_paths(n, self.n_groups, combos, n_paths)
        return CPCVPaths(
            is_test=is_test.reshape(n, n_sims),
            paths=paths.reshape(n, n_paths),
            path_folds=path_folds.reshape(self.n_groups, n_paths),
            index=index,
            n_sims=n_sims,
            n_paths=n_paths,
        )

    build_paths.__doc__ = purgedcv.CombinatorialPurgedCV.build_paths.__doc__
