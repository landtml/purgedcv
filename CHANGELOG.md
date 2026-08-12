# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-12

Two silent-failure paths closed, and the documented claims brought back in line
with what the code and the test suite actually do.

### Fixed

- **Reject a `t1` built for a denser sample than `X`.** Purging happens in
  positional space, so resolving a `t1` against an index it was not built for
  silently rescaled every label horizon: a 21-bar label over a halved index
  spans about 10 positions, and purging then removed far less than it should.
  Reusing a `t1` after resampling or `dropna()` produced hundreds to thousands
  of overlapping train/test label pairs with no error. Extra `t1` timestamps
  falling inside the span of `X.index` now raise `ValueError`. Contiguous
  slices and longer histories resolve identically to a rebuilt `t1` and are
  still accepted.
- **Raise on degenerate training folds.** Purge and embargo can consume an
  entire training set — 6 months of daily bars with a 21-day horizon and a 1%
  embargo is enough. Such folds were yielded silently, became `NaN` scores
  inside scikit-learn, and `np.nanmean` then reported a confident number
  computed from whichever folds survived. See `min_train_size` below.
- **Lists and tuples are no longer mistaken for labelled objects.** `list.index`
  is a method, so `hasattr(X, "index")` is `True` and a plain list failed with
  an error about a builtin method. They now split positionally, as documented.
- Replaced the path-stitch `RuntimeError` with the regular-degree precondition
  it was meant to protect. The old check sat inside the stitch loop, where row
  independence made it unreachable.

### Added

- `min_train_size` (default `1`): the minimum number of training observations a
  fold must retain after purge and embargo. `split` raises below it, naming the
  combination, sample size, horizon and embargo. Pass `0` for the previous
  behaviour.
- `tests/test_sklearn.py`: 14 tests against a real scikit-learn install —
  `cross_val_score`, `cross_validate`, `GridSearchCV`, `RandomizedSearchCV`,
  `Pipeline`, `learning_curve`, `permutation_test_score`, `n_jobs=2` and a
  pickle roundtrip. The suite previously imported scikit-learn nowhere and
  would have passed unchanged had the integration been broken.
- `py.typed`, so the existing annotations reach type checkers downstream.
- A clear `ValueError` when `X.index` and `t1` disagree on timezone-awareness,
  and a `TypeError` when `t1` is not a `Series`.
- A `UserWarning` when `groups=` is passed: it is accepted for API
  compatibility but never consulted, unlike `GroupKFold`.
- CI now covers Python 3.13 and 3.14, exercises the declared dependency floors
  on 3.10/3.11 (the newest Pythons those floors have wheels for), and builds
  the package on every run.

### Changed

- The version is read from `purgedcv.__version__` instead of being duplicated
  in `pyproject.toml`.
- The sdist now ships `PROOF.md` and `verify_leakage.py`, both linked from the
  README and previously absent from the published archive.
- README no longer claims the splitter works anywhere scikit-learn accepts a
  cross-validator: `cross_val_predict` requires partitioning folds and CPCV's
  overlap by construction. `build_paths` is the documented replacement.
- Corrected the test count in README and PROOF.md, labelled the abridged
  certificate as abridged, and rewrote PROOF.md §5 and §7 to match the code.

### Notes

`clone()` on the splitter still fails, exactly as it does for scikit-learn's own
`KFold` and `TimeSeriesSplit`: `BaseCrossValidator` deliberately does not
inherit `BaseEstimator`, and splitters are passed by reference rather than
cloned. The only practical consequence is that the splitter's own parameters
cannot be grid-searched. This is pinned by a test rather than worked around.

## [0.1.0] - 2026-07-23

Initial release: the CPCV splitter with purge and embargo, the combinatorial
path map, `PROOF.md`, and `verify_leakage.py`.

[0.2.0]: https://github.com/landtml/purgedcv/releases/tag/v0.2.0
[0.1.0]: https://github.com/landtml/purgedcv/releases/tag/v0.1.0
