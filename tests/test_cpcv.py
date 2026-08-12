"""Theory-driven tests for the CPCV splitter and path map.

Each test pins a property the AFML ch. 7 scheme must satisfy. Together they form
the regression net that keeps the implementation honest -- notably the
path-stitch test, which fails if the consume-as-you-go counter is "simplified"
into copying the mask per path, a degenerate implementation that silently
duplicates predictions across paths instead of stitching them correctly.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from purgedcv import CombinatorialPurgedCV, CPCVPaths, make_t1
from purgedcv._splitter import _end_positions, _make_group_labels

# (n_groups, n_test_groups) configurations exercised across the suite.
CONFIGS = [(6, 2), (8, 2), (10, 3), (5, 2), (4, 1)]


def _daily_index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=n)


def _frame(n: int) -> pd.DataFrame:
    idx = _daily_index(n)
    rng = np.random.default_rng(0)
    return pd.DataFrame({"f": rng.standard_normal(n)}, index=idx)


# --------------------------------------------------------------------------- #
# 1. Counts                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_groups,k", CONFIGS)
def test_split_and_path_counts(n_groups, k):
    cv = CombinatorialPurgedCV(n_groups, k)
    assert cv.get_n_splits() == math.comb(n_groups, k)
    assert cv.get_n_paths() == math.comb(n_groups, k) * k // n_groups
    # Identity: C(N,k)*k/N == C(N-1, k-1).
    assert cv.get_n_paths() == math.comb(n_groups - 1, k - 1)


@pytest.mark.parametrize("n_groups,k", CONFIGS)
def test_split_yields_n_splits(n_groups, k):
    cv = CombinatorialPurgedCV(n_groups, k)
    X = _frame(400)
    assert sum(1 for _ in cv.split(X)) == cv.get_n_splits()


# --------------------------------------------------------------------------- #
# 2. No leakage (the core purge guarantee)                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_groups,k", CONFIGS)
@pytest.mark.parametrize("horizon", [1, 21])
def test_no_label_overlap_between_train_and_test(n_groups, k, horizon):
    X = _frame(500)
    t1 = make_t1(X.index, horizon)
    cv = CombinatorialPurgedCV(n_groups, k, embargo_pct=0.0)

    start = np.arange(len(X))
    end = _end_positions(X.index, t1)

    for train_idx, test_idx in cv.split(X, t1=t1):
        ts0, te0 = start[test_idx], end[test_idx]
        tr0, tr1 = start[train_idx], end[train_idx]
        # Brute-force every train x test interval pair: none may overlap.
        overlap = (tr1[:, None] >= ts0[None, :]) & (tr0[:, None] <= te0[None, :])
        assert not overlap.any(), "purge failed: a train label overlaps a test label"


# --------------------------------------------------------------------------- #
# 3. Embargo                                                                  #
# --------------------------------------------------------------------------- #
def test_embargo_removes_following_bars():
    n, n_groups, k = 600, 6, 2
    embargo_pct = 0.02
    h = math.ceil(embargo_pct * n)
    X = _frame(n)
    cv = CombinatorialPurgedCV(n_groups, k, embargo_pct=embargo_pct)
    labels = _make_group_labels(n, n_groups)

    for train_idx, test_idx in cv.split(X):  # t1=None -> event ends at own bar
        train = set(train_idx.tolist())
        # Last position of each contiguous test run.
        in_test = np.isin(labels, np.unique(labels[test_idx]))
        edges = np.flatnonzero(np.diff(in_test.astype(np.int8)))
        ends = np.r_[edges, n - 1][np.r_[in_test[edges], in_test[-1]]]
        for b1 in ends:
            for off in range(1, h + 1):
                pos = b1 + off
                if pos < n and not in_test[pos]:
                    assert pos not in train, f"embargoed bar {pos} leaked into train"


def test_zero_embargo_is_noop_and_does_not_raise():
    # Pins the latent NameError the reference code hit when step == 0.
    X = _frame(120)
    cv = CombinatorialPurgedCV(6, 2, embargo_pct=0.0)
    splits = list(cv.split(X, t1=make_t1(X.index, 5)))
    assert len(splits) == cv.get_n_splits()


# --------------------------------------------------------------------------- #
# 4. Disjointness                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_groups,k", CONFIGS)
def test_train_test_disjoint(n_groups, k):
    X = _frame(450)
    cv = CombinatorialPurgedCV(n_groups, k, embargo_pct=0.01)
    for train_idx, test_idx in cv.split(X, t1=make_t1(X.index, 10)):
        assert set(train_idx).isdisjoint(set(test_idx))


# --------------------------------------------------------------------------- #
# 5. Path-stitch regression (rejects the degenerate ".copy()" behaviour)      #
# --------------------------------------------------------------------------- #
def test_path_folds_canonical_6_2():
    X = _frame(120)
    paths = CombinatorialPurgedCV(6, 2).build_paths(X)
    expected = np.array(
        [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [1, 5, 9, 10, 11],
            [2, 6, 9, 12, 13],
            [3, 7, 10, 12, 14],
            [4, 8, 11, 13, 14],
        ]
    )
    np.testing.assert_array_equal(paths.path_folds, expected)


@pytest.mark.parametrize("n_groups,k", CONFIGS)
def test_path_folds_are_non_degenerate(n_groups, k):
    X = _frame(400)
    cv = CombinatorialPurgedCV(n_groups, k)
    pf = cv.build_paths(X).path_folds
    n_paths = cv.get_n_paths()

    # Every path (column) is distinct -> the mask was consumed, not copied.
    cols = {tuple(pf[:, p]) for p in range(n_paths)}
    assert len(cols) == n_paths
    # Each group (row) uses each of its test simulations exactly once.
    for row in pf:
        assert len(set(row.tolist())) == n_paths
    # Each simulation is assigned to exactly its k member groups across paths.
    counts = np.bincount(pf.ravel(), minlength=cv.get_n_splits())
    assert (counts == k).all()


# --------------------------------------------------------------------------- #
# 6. Coverage / path completeness                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_groups,k", CONFIGS)
def test_every_observation_tested_n_paths_times(n_groups, k):
    X = _frame(400)
    paths = CombinatorialPurgedCV(n_groups, k).build_paths(X)
    # Each observation is a test point in exactly n_paths simulations.
    assert (paths.is_test.sum(axis=1) == paths.n_paths).all()
    # Each backtest path assigns a simulation to every observation (no gaps).
    assert (paths.paths >= 0).all()


@pytest.mark.parametrize("n_groups,k", CONFIGS)
def test_each_path_uses_only_valid_simulations(n_groups, k):
    X = _frame(400)
    paths = CombinatorialPurgedCV(n_groups, k).build_paths(X)
    # For path p, group g is served by a simulation that actually tests group g.
    for p in range(paths.n_paths):
        for g in range(n_groups):
            sim = paths.path_folds[g, p]
            assert paths.is_test[:, sim][_make_group_labels(len(X), n_groups) == g].all()


# --------------------------------------------------------------------------- #
# 7. Determinism (no shared state across calls -> parallel-safe)              #
# --------------------------------------------------------------------------- #
def test_split_is_deterministic_across_calls():
    X = _frame(300)
    t1 = make_t1(X.index, 15)
    cv = CombinatorialPurgedCV(6, 2, embargo_pct=0.01)
    first = [(tr.copy(), te.copy()) for tr, te in cv.split(X, t1=t1)]
    second = list(cv.split(X, t1=t1))
    for (a_tr, a_te), (b_tr, b_te) in zip(first, second):
        np.testing.assert_array_equal(a_tr, b_tr)
        np.testing.assert_array_equal(a_te, b_te)


def test_build_paths_is_deterministic():
    X = _frame(300)
    cv = CombinatorialPurgedCV(6, 2)
    np.testing.assert_array_equal(
        cv.build_paths(X).path_folds, cv.build_paths(X).path_folds
    )


# --------------------------------------------------------------------------- #
# combine() gather + index preservation (Bug #4 hygiene)                      #
# --------------------------------------------------------------------------- #
def test_combine_gathers_correct_simulation():
    X = _frame(120)
    paths = CombinatorialPurgedCV(6, 2).build_paths(X)
    # pred[t, c] == c lets us check the gather picks paths[t, p] verbatim.
    pred = np.tile(np.arange(paths.n_sims), (len(X), 1)).astype(float)
    out = paths.combine(pred)
    np.testing.assert_array_equal(out, paths.paths.astype(float))

    df = paths.to_frame(pred)
    assert df.index.equals(X.index)          # datetime index survives downstream
    assert list(df.columns) == [f"path_{p}" for p in range(paths.n_paths)]


def test_combine_rejects_wrong_shape():
    X = _frame(120)
    paths = CombinatorialPurgedCV(6, 2).build_paths(X)
    with pytest.raises(ValueError):
        paths.combine(np.zeros((len(X), paths.n_sims + 1)))


# --------------------------------------------------------------------------- #
# make_t1 + constructor validation                                            #
# --------------------------------------------------------------------------- #
def test_make_t1_shape_and_tail_clip():
    idx = _daily_index(50)
    t1 = make_t1(idx, 21)
    assert len(t1) == 50
    assert t1.index.equals(idx)
    assert (t1.iloc[-21:] == idx[-1]).all()   # tail events clip to the last bar
    assert (t1.to_numpy() >= idx.to_numpy()).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_groups=1, n_test_groups=1),
        dict(n_groups=6, n_test_groups=0),
        dict(n_groups=6, n_test_groups=6),
        dict(n_groups=6, n_test_groups=2, embargo_pct=1.0),
        dict(n_groups=6, n_test_groups=2, embargo_pct=-0.1),
    ],
)
def test_constructor_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(**kwargs)


def test_split_rejects_too_few_samples():
    cv = CombinatorialPurgedCV(6, 2)
    with pytest.raises(ValueError):
        list(cv.split(_frame(4)))


def test_unsorted_index_with_t1_raises():
    X = _frame(100).iloc[::-1]  # descending index, like the reference CSV
    cv = CombinatorialPurgedCV(6, 2)
    with pytest.raises(ValueError):
        list(cv.split(X, t1=pd.Series(X.index, index=X.index)))


# --------------------------------------------------------------------------- #
# Envelope-purge == per-observation brute force (locks the equivalence proof)  #
# --------------------------------------------------------------------------- #
def _bruteforce_train_idx(index, t1, n_groups, combo):
    """Reference purge: drop a train obs iff it overlaps ANY individual test obs."""
    n = len(index)
    start = np.arange(n)
    end = _end_positions(index, t1)
    labels = _make_group_labels(n, n_groups)
    test_mask = np.isin(labels, combo)
    ts0, te0 = start[test_mask], end[test_mask]
    keep = []
    for i in range(n):
        if test_mask[i]:
            continue
        if np.any((end[i] >= ts0) & (start[i] <= te0)):
            continue  # overlaps some test observation -> purge
        keep.append(i)
    return np.array(keep, dtype=int)


@pytest.mark.parametrize("n_groups,k", CONFIGS)
@pytest.mark.parametrize("horizon", [1, 10, 21])
def test_envelope_purge_equals_bruteforce(n_groups, k, horizon):
    X = _frame(360)
    t1 = make_t1(X.index, horizon)
    cv = CombinatorialPurgedCV(n_groups, k, embargo_pct=0.0)
    combos = list(itertools.combinations(range(n_groups), k))
    for (train_idx, _), combo in zip(cv.split(X, t1=t1), combos):
        np.testing.assert_array_equal(
            train_idx, _bruteforce_train_idx(X.index, t1, n_groups, combo)
        )


# --------------------------------------------------------------------------- #
# Fix 1: t1 in constructor (purges through the sklearn split signature)        #
# --------------------------------------------------------------------------- #
def test_t1_in_constructor_purges_via_sklearn_signature():
    X = _frame(500)
    t1 = make_t1(X.index, 21)
    ctor = CombinatorialPurgedCV(6, 2, t1=t1)
    # split(X, y, groups) -- the call sklearn helpers make -- still purges.
    for (a_tr, a_te), (b_tr, b_te) in zip(ctor.split(X, None, None),
                                          CombinatorialPurgedCV(6, 2).split(X, t1=t1)):
        np.testing.assert_array_equal(a_tr, b_tr)
        np.testing.assert_array_equal(a_te, b_te)
    # ...and it genuinely removes more than the no-t1 (test-indices-only) case.
    no_t1 = list(CombinatorialPurgedCV(6, 2).split(X))
    assert any(len(a_tr) < len(b_tr)
               for (a_tr, _), (b_tr, _) in zip(ctor.split(X, None, None), no_t1))


def test_split_t1_overrides_constructor_t1():
    X = _frame(360)
    cv = CombinatorialPurgedCV(6, 2, t1=make_t1(X.index, 1))
    long_purge = sum(len(tr) for tr, _ in cv.split(X, t1=make_t1(X.index, 21)))
    short_purge = sum(len(tr) for tr, _ in cv.split(X))  # uses ctor t1 (horizon 1)
    assert long_purge < short_purge


# --------------------------------------------------------------------------- #
# Fix 2: embargo_anchor                                                        #
# --------------------------------------------------------------------------- #
def test_embargo_anchor_label_end_removes_at_least_test_end():
    X = _frame(600)
    t1 = make_t1(X.index, 21)
    le = CombinatorialPurgedCV(6, 2, embargo_pct=0.02, embargo_anchor="label_end")
    te = CombinatorialPurgedCV(6, 2, embargo_pct=0.02, embargo_anchor="test_end")
    for (le_tr, _), (te_tr, _) in zip(le.split(X, t1=t1), te.split(X, t1=t1)):
        assert set(le_tr).issubset(set(te_tr))  # label_end is the conservative one


def test_invalid_embargo_anchor_raises():
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(6, 2, embargo_anchor="nope")


# --------------------------------------------------------------------------- #
# Fix 3: input validation (alignment / uniqueness)                             #
# --------------------------------------------------------------------------- #
def test_misaligned_t1_raises():
    X = _frame(200)
    t1 = make_t1(X.index, 5).iloc[:-10]  # missing event-ends for the tail
    with pytest.raises(ValueError):
        list(CombinatorialPurgedCV(6, 2).split(X, t1=t1))


def test_nonunique_index_with_t1_raises():
    base = _daily_index(100)
    dup = base.append(base[[-1]])  # last timestamp duplicated (still non-decreasing)
    X = pd.DataFrame({"f": np.zeros(len(dup))}, index=dup)
    t1 = pd.Series(dup, index=dup)
    with pytest.raises(ValueError):
        list(CombinatorialPurgedCV(6, 2).split(X, t1=t1))


def test_make_t1_rejects_unsorted_and_duplicate_index():
    with pytest.raises(ValueError):
        make_t1(_daily_index(50)[::-1], 5)
    base = _daily_index(50)
    with pytest.raises(ValueError):
        make_t1(base.append(base[[-1]]), 5)


# --------------------------------------------------------------------------- #
# Stale t1: a t1 built for a different X must never be applied silently        #
# --------------------------------------------------------------------------- #
# A t1 is a map from event start to event end, but purging works in *positional*
# space. Resolving a t1 against an index it was not built for silently rescales
# every label horizon -- a 21-bar label over a halved index spans ~10 positions --
# so purge removes far less than it should. That is exactly the leakage this
# library exists to prevent, so a mismatched t1 is rejected rather than reindexed.
def _leaking_pairs(X, t1, cv):
    """Count train/test label-interval overlaps, judged against the true t1."""
    start = np.arange(len(X))
    end = _end_positions(X.index, t1)
    total = 0
    for train_idx, test_idx in cv.split(X, t1=t1):
        ts0, te0 = start[test_idx], end[test_idx]
        tr0, tr1 = start[train_idx], end[train_idx]
        total += int(((tr1[:, None] >= ts0[None, :]) & (tr0[:, None] <= te0[None, :])).sum())
    return total


def test_stale_ctor_t1_on_subsampled_X_raises():
    # The dangerous case: t1 built on the full index, split on a resampled subset.
    X = _frame(400)
    cv = CombinatorialPurgedCV(6, 2, t1=make_t1(X.index, 21))
    with pytest.raises(ValueError, match="t1"):
        list(cv.split(X.iloc[::2]))


def test_stale_ctor_t1_after_dropna_raises():
    # dropna() is ordinary preprocessing; reusing a pre-dropna t1 leaves interior
    # gaps in t1.index, which is the resampling signature the guard rejects.
    X = _frame(300)
    t1 = make_t1(X.index, 21)
    X_clean = X.drop(X.index[50:80])
    with pytest.raises(ValueError, match="t1"):
        list(CombinatorialPurgedCV(6, 2, t1=t1).split(X_clean))


def test_stale_t1_would_have_leaked_but_correct_t1_does_not():
    # The guard's justification: rebuilt-for-the-subset t1 leaks nothing.
    X = _frame(400).iloc[::2]
    cv = CombinatorialPurgedCV(6, 2, embargo_pct=0.0)
    assert _leaking_pairs(X, make_t1(X.index, 21), cv) == 0


# Contiguous slices are provably safe: shared timestamps keep their relative
# positions and tail labels clip identically, so end_pos is bit-for-bit equal to
# a rebuilt t1. Rejecting them would break nested CV over contiguous folds.
@pytest.mark.parametrize("sl", [slice(None, 200), slice(100, 300), slice(200, None)])
def test_contiguous_slice_with_original_t1_is_accepted(sl):
    X = _frame(400)
    t1 = make_t1(X.index, 21)
    X_sub = X.iloc[sl]
    got = _end_positions(X_sub.index, t1)
    want = _end_positions(X_sub.index, make_t1(X_sub.index, 21))
    np.testing.assert_array_equal(got, want)


def test_superset_t1_is_still_accepted():
    # A t1 covering a longer history than X is unambiguous on X's own grid.
    X = _frame(400)
    t1_big = make_t1(_daily_index(600), 21)
    splits = list(CombinatorialPurgedCV(6, 2).split(X, t1=t1_big))
    assert len(splits) == 15


# --------------------------------------------------------------------------- #
# Degenerate folds: purge + embargo can consume the entire training set        #
# --------------------------------------------------------------------------- #
# An empty training fold is never a meaningful simulation. Yielded silently it
# becomes a NaN score inside sklearn, and np.nanmean then reports a confident
# number computed from whichever folds happened to survive.
def test_degenerate_fold_raises_by_default():
    # 6 months of daily bars with a 21-day horizon -- an ordinary configuration.
    X = _frame(126)
    cv = CombinatorialPurgedCV(6, 2, embargo_pct=0.01)
    with pytest.raises(ValueError, match="training observation"):
        list(cv.split(X, t1=make_t1(X.index, 21)))


def test_degenerate_fold_message_names_the_knobs():
    X = _frame(126)
    cv = CombinatorialPurgedCV(6, 2, embargo_pct=0.01)
    with pytest.raises(ValueError) as exc:
        list(cv.split(X, t1=make_t1(X.index, 21)))
    msg = str(exc.value)
    for token in ("test groups", "embargo", "min_train_size"):
        assert token in msg, f"error message should mention {token!r}: {msg}"


def test_min_train_size_zero_restores_legacy_behaviour():
    X = _frame(126)
    cv = CombinatorialPurgedCV(6, 2, embargo_pct=0.01, min_train_size=0)
    sizes = [len(tr) for tr, _ in cv.split(X, t1=make_t1(X.index, 21))]
    assert min(sizes) == 0          # the degenerate fold is still yielded
    assert len(sizes) == cv.get_n_splits()


def test_min_train_size_enforces_a_floor():
    X = _frame(500)
    t1 = make_t1(X.index, 21)
    ok = [len(tr) for tr, _ in CombinatorialPurgedCV(6, 2).split(X, t1=t1)]
    cv = CombinatorialPurgedCV(6, 2, min_train_size=min(ok) + 1)
    with pytest.raises(ValueError, match="training observation"):
        list(cv.split(X, t1=t1))


def test_healthy_config_is_unaffected():
    X = _frame(2000)
    cv = CombinatorialPurgedCV(6, 2, embargo_pct=0.01)
    sizes = [len(tr) for tr, _ in cv.split(X, t1=make_t1(X.index, 21))]
    assert len(sizes) == cv.get_n_splits()
    assert min(sizes) > 0


def test_min_train_size_rejects_negative():
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(6, 2, min_train_size=-1)


# --------------------------------------------------------------------------- #
# Fix 4: combine() shape hardening                                             #
# --------------------------------------------------------------------------- #
def test_combine_rejects_non_2d():
    X = _frame(120)
    paths = CombinatorialPurgedCV(6, 2).build_paths(X)
    with pytest.raises(ValueError):
        paths.combine(np.zeros(len(X)))  # 1-D


# --------------------------------------------------------------------------- #
# yfinance integration (auto-skips offline)                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_yfinance_real_data_no_leakage():
    yf = pytest.importorskip("yfinance")
    df = yf.download("AAPL", start="2018-01-01", end="2022-01-01",
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        pytest.skip("yfinance returned no data (offline?)")
    df = df.dropna()
    t1 = make_t1(df.index, 21)
    cv = CombinatorialPurgedCV(6, 2, embargo_pct=0.01)

    start = np.arange(len(df))
    end = _end_positions(df.index, t1)
    n_test_seen = np.zeros(len(df), dtype=int)

    for train_idx, test_idx in cv.split(df, t1=t1):
        assert set(train_idx).isdisjoint(set(test_idx))
        ts0, te0 = start[test_idx], end[test_idx]
        tr0, tr1 = start[train_idx], end[train_idx]
        overlap = (tr1[:, None] >= ts0[None, :]) & (tr0[:, None] <= te0[None, :])
        assert not overlap.any()
        n_test_seen[test_idx] += 1

    assert (n_test_seen == cv.get_n_paths()).all()
    assert cv.build_paths(df).index.equals(df.index)
