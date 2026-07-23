"""Internal implementation of Combinatorial Purged Cross-Validation.

Import from :mod:`purgedcv` (the package's public ``__init__.py``), not this
module directly — this file is the private implementation detail, kept
separate from the public API so it can be reorganized without breaking users.

A clean, fully-vectorized implementation of the cross-validation scheme from
Marcos Lopez de Prado, *Advances in Financial Machine Learning* (AFML),
Chapter 7 (snippets 7.1-7.6).

Scope
-----
This module is a **splitter + path-map only**. It produces:

* purged, embargoed ``(train_idx, test_idx)`` splits (sklearn-compatible), and
* the combinatorial **path map** needed to assemble ``N`` out-of-sample
  backtest paths from the per-simulation model predictions.

It deliberately performs **no return / PnL accounting**. Translating predictions
into strategy returns belongs to the downstream backtest engine. Keeping that
out is what makes this component immune to the overlapping-return double-counting
mistake that plagues naive CPCV blog implementations.

Design notes
------------
* Purge and embargo are expressed as a single vectorized *interval-overlap*
  test in positional space, so cost is a handful of boolean ops per split.
* The splitter is stateless and picklable: every call to :meth:`split` is
  independent and deterministic, so the expensive part of a real run -- fitting
  the model ``C(N, k)`` times -- can be fanned across cores with
  ``joblib.Parallel`` without any shared-state hazards.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import numpy.typing as npt
import pandas as pd

try:  # sklearn compatibility is nice-to-have, not required to import this module.
    from sklearn.model_selection import BaseCrossValidator as _Base
except Exception:  # pragma: no cover - sklearn always present per requirements.
    _Base = object  # type: ignore[assignment, misc]

__all__ = ["CombinatorialPurgedCV", "CPCVPaths", "make_t1"]


# --------------------------------------------------------------------------- #
# Event-lifespan helper                                                       #
# --------------------------------------------------------------------------- #
def make_t1(index: pd.Index, horizon: int) -> pd.Series:
    """Build a de Prado ``t1`` (event-lifespan) Series for a fixed holding period.

    Each observation ``index[i]`` is mapped to the timestamp ``horizon`` bars
    later, ``index[i + horizon]``. The final ``horizon`` observations clip to the
    last available bar (their event simply ends at the end of the sample).

    Parameters
    ----------
    index : pd.Index
        The (ascending, unique) sample index, e.g. a ``DatetimeIndex``.
    horizon : int
        Holding period in bars (e.g. ``21`` for a 21-day forward label).

    Returns
    -------
    pd.Series
        ``index``-aligned series whose values are the event-end timestamps.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon!r}.")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending.")
    if not index.is_unique:
        raise ValueError("index must be unique.")
    n = len(index)
    end_pos = np.minimum(np.arange(n) + horizon, n - 1)
    return pd.Series(index[end_pos], index=index, name="t1")


# --------------------------------------------------------------------------- #
# Internal vectorized primitives (pure, independently testable)               #
# --------------------------------------------------------------------------- #
def _make_group_labels(n_samples: int, n_groups: int) -> npt.NDArray[np.int_]:
    """Assign each of ``n_samples`` positions to one of ``n_groups`` contiguous blocks.

    The trailing remainder is folded into the last group (the AFML convention),
    so groups are contiguous in time and cover the timeline exactly once.
    """
    block = n_samples // n_groups
    labels = np.arange(n_samples) // block
    labels[labels >= n_groups] = n_groups - 1
    return labels


def _end_positions(index: pd.Index, t1: pd.Series | None) -> npt.NDArray[np.int_]:
    """Map every observation's event-end time (``t1``) to a positional index.

    With ``t1 is None`` each observation spans a single bar (``end_pos == i``).
    Otherwise ``end_pos[i]`` is the position in ``index`` of ``t1[i]`` (located by
    binary search, so non-exact / holiday-adjusted timestamps still resolve).
    An event can never end before it starts.

    Raises
    ------
    ValueError
        If ``index`` is not strictly increasing/unique, or if ``t1`` does not
        provide an event-end for every observation in ``index`` (silent
        misalignment is a leakage hazard, so it is rejected, not patched).
    """
    n = len(index)
    if t1 is None:
        return np.arange(n)
    if not index.is_monotonic_increasing:
        raise ValueError("X.index must be sorted ascending for purge/embargo.")
    if not index.is_unique:
        raise ValueError("X.index must be unique for purge/embargo.")

    aligned = t1.reindex(index)
    if aligned.isna().any():
        raise ValueError(
            "t1 must provide an event-end time for every observation in X.index; "
            "found unaligned/missing entries after reindexing onto X.index."
        )
    pos = index.searchsorted(aligned.to_numpy(), side="left")
    end_pos = np.clip(pos, 0, n - 1).astype(np.int64)
    # An event end cannot precede its own start position.
    return np.maximum(end_pos, np.arange(n))


def _contiguous_blocks(group_labels: npt.NDArray[np.int_], groups: tuple[int, ...]
                       ) -> list[tuple[int, int]]:
    """Return the ``[start, end]`` position spans (inclusive) of the given groups.

    Each group is a single contiguous block by construction; adjacent selected
    groups are merged into one span so the overlap test runs once per run.
    """
    in_test = np.isin(group_labels, groups)
    if not in_test.any():
        return []
    # Boundaries where membership flips identify contiguous runs.
    edges = np.flatnonzero(np.diff(in_test.astype(np.int8)))
    starts = np.r_[0, edges + 1]
    ends = np.r_[edges, len(in_test) - 1]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if in_test[s]]


def _train_mask(
    start_pos: npt.NDArray[np.int_],
    end_pos: npt.NDArray[np.int_],
    test_blocks: list[tuple[int, int]],
    embargo: int,
    anchor: str,
) -> npt.NDArray[np.bool_]:
    """Boolean mask of training observations surviving purge + forward embargo.

    For each contiguous test block ``[b0, b1]`` with label envelope
    ``label_end = max(end_pos[b0..b1])``:

    * **Purge** removes any train observation ``[s, e]`` whose label overlaps the
      test label window, in either direction: ``(e >= b0) and (s <= label_end)``.
      Because the test block's label intervals form a *gapless* union
      ``[b0, label_end]`` (consecutive integer starts, each ``e_j >= j``), this
      envelope test is provably equivalent to checking every individual test
      observation, and it covers all three AFML 7.1 cases (train starts within /
      ends within / fully envelops the test window).
    * **Embargo** (forward only) additionally removes a buffer of ``embargo`` bars
      *after* an anchor position. ``anchor='label_end'`` (default, the more
      conservative reading) measures the buffer from the end of the test *labels*;
      ``anchor='test_end'`` measures it from the last test *index* ``b1``
      (the literal mlfinlab/``PurgedKFold`` convention, weaker whenever the embargo
      is shorter than the label horizon). ``embargo == 0`` reduces to a pure purge.
    """
    remove = np.zeros(start_pos.shape, dtype=bool)
    for b0, b1 in test_blocks:
        label_end = int(end_pos[b0 : b1 + 1].max())
        remove |= (end_pos >= b0) & (start_pos <= label_end)          # purge
        if embargo:
            anchor_pos = label_end if anchor == "label_end" else b1
            remove |= (start_pos > anchor_pos) & (start_pos <= anchor_pos + embargo)
    return ~remove


# --------------------------------------------------------------------------- #
# Path map                                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CPCVPaths:
    """Combinatorial path map produced by :meth:`CombinatorialPurgedCV.build_paths`.

    Read column-wise. ``is_test[:, c]`` is the test mask of simulation ``c``;
    ``paths[:, p]`` records, for each observation, which simulation supplies its
    prediction in backtest path ``p``.
    """

    is_test: npt.NDArray[np.bool_]      # (n_samples, n_sims)
    paths: npt.NDArray[np.int_]         # (n_samples, n_paths)
    path_folds: npt.NDArray[np.int_]    # (n_groups, n_paths)
    index: pd.Index                     # original index, preserved for downstream
    n_sims: int
    n_paths: int

    def combine(self, pred: npt.NDArray) -> npt.NDArray:
        """Gather per-simulation predictions into per-path predictions.

        ``pred`` has shape ``(n_samples, n_sims)`` (e.g. ``pred[t, c]`` is the
        prediction for observation ``t`` from simulation ``c``). The result has
        shape ``(n_samples, n_paths)`` with ``out[t, p] = pred[t, paths[t, p]]``.

        This is a pure index gather -- no returns, no compounding -- so it cannot
        reproduce the overlapping-return inflation bug.
        """
        pred = np.asarray(pred)
        expected = (self.paths.shape[0], self.n_sims)
        if pred.ndim != 2 or pred.shape != expected:
            raise ValueError(
                f"pred must be a 2-D array of shape {expected}, got "
                f"shape {pred.shape} (ndim={pred.ndim})."
            )
        rows = np.arange(pred.shape[0])[:, None]
        return pred[rows, self.paths]

    def to_frame(self, pred: npt.NDArray) -> pd.DataFrame:
        """Same gather as :meth:`combine`, returned as an index-aligned DataFrame.

        Carrying the original (datetime) index downstream is what lets
        performance tooling annualize and compute drawdowns correctly.
        """
        cols = [f"path_{p}" for p in range(self.n_paths)]
        return pd.DataFrame(self.combine(pred), index=self.index, columns=cols)


# --------------------------------------------------------------------------- #
# The cross-validator                                                         #
# --------------------------------------------------------------------------- #
class CombinatorialPurgedCV(_Base):
    """Combinatorial Purged Cross-Validation with an embargo (AFML ch. 7).

    The timeline is split into ``n_groups`` contiguous groups. Every combination
    of ``n_test_groups`` groups forms one test set (a *simulation*), giving
    ``C(n_groups, n_test_groups)`` splits. Training observations whose labels
    overlap the test window are *purged*, and a forward *embargo* of
    ``ceil(embargo_pct * n_samples)`` bars is removed after each test block.

    Parameters
    ----------
    n_groups : int
        Number of contiguous groups ``N`` to partition the timeline into.
    n_test_groups : int
        Groups held out as test per combination ``k`` (``1 <= k < N``).
    embargo_pct : float, default 0.0
        Embargo size as a fraction of total bars (de Prado convention).
    t1 : pd.Series, optional
        Default event-lifespan series (see :func:`make_t1`). When supplied here,
        purging works even through sklearn helpers (``cross_val_score``,
        ``GridSearchCV``) that call ``split(X, y, groups)`` and cannot pass ``t1``
        themselves. A ``t1`` passed directly to :meth:`split` overrides this.
    embargo_anchor : {'label_end', 'test_end'}, default 'label_end'
        Where the forward embargo buffer is measured from (see :func:`_train_mask`).

    Notes
    -----
    The number of fully-sampled backtest paths is ``C(N, k) * k // N``
    (equivalently ``C(N-1, k-1)``), the number of simulations in which any single
    group serves as a test fold.

    Groups are equal-sized by count, with any remainder folded into the last
    group; if ``n_groups`` does not divide ``n_samples`` the final group is larger.
    """

    _VALID_ANCHORS = ("label_end", "test_end")

    def __init__(
        self,
        n_groups: int,
        n_test_groups: int,
        embargo_pct: float = 0.0,
        t1: pd.Series | None = None,
        embargo_anchor: str = "label_end",
    ):
        if n_groups < 2:
            raise ValueError(f"n_groups must be >= 2, got {n_groups!r}.")
        if not 1 <= n_test_groups < n_groups:
            raise ValueError(
                f"n_test_groups must satisfy 1 <= k < n_groups; "
                f"got k={n_test_groups!r}, n_groups={n_groups!r}."
            )
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError(f"embargo_pct must be in [0, 1), got {embargo_pct!r}.")
        if embargo_anchor not in self._VALID_ANCHORS:
            raise ValueError(
                f"embargo_anchor must be one of {self._VALID_ANCHORS}, "
                f"got {embargo_anchor!r}."
            )
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct
        self.t1 = t1
        self.embargo_anchor = embargo_anchor

    # -- introspection ----------------------------------------------------- #
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of simulations, ``C(n_groups, n_test_groups)``."""
        return math.comb(self.n_groups, self.n_test_groups)

    def get_n_paths(self) -> int:
        """Number of fully-sampled backtest paths, ``C(N, k) * k // N``."""
        return self.get_n_splits() * self.n_test_groups // self.n_groups

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_groups={self.n_groups}, "
            f"n_test_groups={self.n_test_groups}, embargo_pct={self.embargo_pct}, "
            f"embargo_anchor={self.embargo_anchor!r}, "
            f"t1={'set' if self.t1 is not None else None})"
        )

    # -- internal ---------------------------------------------------------- #
    def _validate_n(self, n_samples: int) -> None:
        if n_samples < self.n_groups:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_groups ({self.n_groups})."
            )

    def _embargo_size(self, n_samples: int) -> int:
        return math.ceil(self.embargo_pct * n_samples) if self.embargo_pct else 0

    # -- the splits -------------------------------------------------------- #
    def split(
        self, X, y=None, groups=None, t1: pd.Series | None = None
    ) -> Iterator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
        """Yield ``(train_idx, test_idx)`` positional index arrays per simulation.

        Parameters
        ----------
        X : array-like or DataFrame
            Samples; only its length and (if ``t1`` is given) its ``.index`` are used.
        y, groups : ignored
            Present for sklearn API compatibility (``split(X, y, groups)``).
        t1 : pd.Series, optional
            Event-lifespan series mapping each sample's start time to its
            event-end time (see :func:`make_t1`). Falls back to the ``t1`` given to
            the constructor. If both are ``None``, each observation spans a single
            bar and purging only removes the test indices themselves.
        """
        t1 = self.t1 if t1 is None else t1
        index = X.index if hasattr(X, "index") else pd.RangeIndex(len(X))
        n = len(index)
        self._validate_n(n)

        group_labels = _make_group_labels(n, self.n_groups)
        start_pos = np.arange(n)
        end_pos = _end_positions(index, t1)
        embargo = self._embargo_size(n)
        all_idx = np.arange(n)

        for combo in itertools.combinations(range(self.n_groups), self.n_test_groups):
            test_mask = np.isin(group_labels, combo)
            blocks = _contiguous_blocks(group_labels, combo)
            train_mask = _train_mask(start_pos, end_pos, blocks, embargo, self.embargo_anchor)
            train_mask &= ~test_mask  # never train on a test observation
            yield all_idx[train_mask], all_idx[test_mask]

    # -- the path map ------------------------------------------------------ #
    def build_paths(self, X, t1: pd.Series | None = None) -> CPCVPaths:
        """Construct the combinatorial path map (``is_test`` / ``paths`` / ``path_folds``).

        ``t1`` is accepted for API symmetry; the path map depends only on the
        group geometry, not on the event lifespans.
        """
        index = X.index if hasattr(X, "index") else pd.RangeIndex(len(X))
        n = len(index)
        self._validate_n(n)

        N, k = self.n_groups, self.n_test_groups
        group_labels = _make_group_labels(n, N)
        combos = list(itertools.combinations(range(N), k))
        n_sims = len(combos)
        n_paths = self.get_n_paths()

        is_test = np.zeros((n, n_sims), dtype=bool)
        is_test_group = np.zeros((N, n_sims), dtype=bool)
        for c, combo in enumerate(combos):
            is_test_group[list(combo), c] = True
            is_test[np.isin(group_labels, combo), c] = True

        # Stitch paths with a consume-as-you-go counter: for each path, take the
        # first not-yet-used simulation in which a group is a test fold, then mark
        # it used. The invariant -- each group is a test fold in exactly n_paths
        # simulations -- guarantees every (group, path) cell gets a real sim.
        work = is_test_group.copy()
        path_folds = np.full((N, n_paths), -1, dtype=np.int64)
        for p in range(n_paths):
            for g in range(N):
                if not work[g].any():  # would silently default argmax() to 0
                    raise RuntimeError(
                        f"path-stitch invariant violated for group {g}: "
                        "ran out of test simulations."
                    )
                sim = int(work[g].argmax())
                path_folds[g, p] = sim
                work[g, sim] = False

        # Broadcast each group's per-path simulation across the group's ticks.
        paths = np.full((n, n_paths), -1, dtype=np.int64)
        for g in range(N):
            paths[group_labels == g, :] = path_folds[g, :]

        return CPCVPaths(
            is_test=is_test,
            paths=paths,
            path_folds=path_folds,
            index=index,
            n_sims=n_sims,
            n_paths=n_paths,
        )
