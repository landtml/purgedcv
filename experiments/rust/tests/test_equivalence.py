"""The Rust engine must be indistinguishable from the pure-Python splitter."""

import pickle

import numpy as np
import pandas as pd
import pytest

import purgedcv

purgedcv_rs = pytest.importorskip("purgedcv_rs")


def _assert_same_splits(py_splits, rs_splits):
    py_splits, rs_splits = list(py_splits), list(rs_splits)
    assert len(py_splits) == len(rs_splits)
    for (py_tr, py_te), (rs_tr, rs_te) in zip(py_splits, rs_splits):
        assert rs_tr.dtype == py_tr.dtype and rs_te.dtype == py_te.dtype
        np.testing.assert_array_equal(rs_tr, py_tr)
        np.testing.assert_array_equal(rs_te, py_te)


def _random_t1(rng, index, kind):
    n = len(index)
    if kind == "none":
        return None
    if kind == "fixed":
        return purgedcv.make_t1(index, int(rng.integers(1, max(2, n // 4))))
    # Arbitrary per-event horizons: end_pos is not monotone, which exercises
    # the engine's linear-scan fallback rather than the binary-search path.
    ends = np.minimum(np.arange(n) + rng.integers(0, max(2, n // 3), n), n - 1)
    return pd.Series(index[ends], index=index)


@pytest.mark.parametrize("seed", range(300))
def test_matches_python_engine(seed):
    rng = np.random.default_rng(seed)
    n_groups = int(rng.integers(2, 9))
    n_test = int(rng.integers(1, n_groups))
    n = int(rng.integers(n_groups, 400))
    index = pd.bdate_range("2000-01-03", periods=n)
    X = pd.DataFrame({"x": np.zeros(n)}, index=index)
    kwargs = dict(
        n_groups=n_groups,
        n_test_groups=n_test,
        embargo_pct=float(rng.choice([0.0, 0.01, 0.05, 0.2])),
        embargo_anchor=str(rng.choice(["label_end", "test_end"])),
        t1=_random_t1(rng, index, rng.choice(["none", "fixed", "random"])),
        min_train_size=0,
    )
    py = purgedcv.CombinatorialPurgedCV(**kwargs)
    rs = purgedcv_rs.CombinatorialPurgedCV(**kwargs)
    _assert_same_splits(py.split(X), rs.split(X))
    _assert_same_splits(py.split(X), rs.split_all(X))


def test_same_error_on_degenerate_fold():
    X = pd.DataFrame({"x": np.zeros(126)}, index=pd.bdate_range("2020", periods=126))
    t1 = purgedcv.make_t1(X.index, 21)
    errors = []
    for engine in (purgedcv, purgedcv_rs):
        cv = engine.CombinatorialPurgedCV(6, 2, embargo_pct=0.01, t1=t1)
        with pytest.raises(ValueError) as exc:
            list(cv.split(X))
        errors.append(str(exc.value))
    assert errors[0] == errors[1]
    with pytest.raises(ValueError, match="retains 0 training"):
        purgedcv_rs.CombinatorialPurgedCV(6, 2, embargo_pct=0.01, t1=t1).split_all(X)


def test_bad_t1_still_raises_eagerly():
    X = pd.DataFrame({"x": np.zeros(100)}, index=pd.bdate_range("2020", periods=100))
    t1 = purgedcv.make_t1(X.index, 5)
    with pytest.raises(ValueError, match="denser sample"):
        purgedcv_rs.CombinatorialPurgedCV(4, 2, t1=t1).split(X.iloc[::2])


def test_plan_rejects_broken_preconditions():
    with pytest.raises(ValueError, match="violates"):
        purgedcv_rs.SplitPlan(np.array([1, 0, 2]), 2, 0)
    plan = purgedcv_rs.SplitPlan(np.arange(10), 2, 0)
    with pytest.raises(ValueError, match="strictly increasing"):
        plan.split([1, 0])


def test_picklable_and_sklearn_compatible():
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), index=pd.bdate_range("2020", periods=300))
    y = pd.Series(rng.normal(size=300), index=X.index)
    cv = purgedcv_rs.CombinatorialPurgedCV(6, 2, embargo_pct=0.01, t1=purgedcv.make_t1(X.index, 5))
    cv = pickle.loads(pickle.dumps(cv))
    scores = cross_val_score(Ridge(), X, y, cv=cv, n_jobs=2)
    assert len(scores) == cv.get_n_splits() and np.isfinite(scores).all()


@pytest.mark.parametrize(
    "X",
    [np.zeros((50, 2)), list(range(50)), pd.Series(np.zeros(50))],
    ids=["ndarray", "list", "series-rangeindex"],
)
@pytest.mark.parametrize("n_groups,n_test", [(2, 1), (5, 4), (50, 49), (50, 1)])
def test_edge_shapes_match(X, n_groups, n_test):
    kw = dict(n_groups=n_groups, n_test_groups=n_test, embargo_pct=0.5, min_train_size=0)
    py = purgedcv.CombinatorialPurgedCV(**kw)
    rs = purgedcv_rs.CombinatorialPurgedCV(**kw)
    _assert_same_splits(py.split(X), rs.split(X))


def test_huge_embargo_on_raw_plan_saturates():
    plan = purgedcv_rs.SplitPlan(np.arange(10), 2, 2**64 - 1)
    train, test = plan.split([0])
    assert train.size == 0
    np.testing.assert_array_equal(test, np.arange(5))


@pytest.mark.parametrize("n", [100, 25_000])  # below and above the parallel cutoff
def test_lazy_split_is_lazy_and_raises_at_the_same_fold(n):
    X = pd.DataFrame({"x": np.zeros(n)}, index=pd.bdate_range("1990", periods=n))
    cv = purgedcv_rs.CombinatorialPurgedCV(10, 3, embargo_pct=0.01, t1=purgedcv.make_t1(X.index, 5))
    it = cv.split(X)
    first = next(it)
    _assert_same_splits([first], [next(purgedcv.CombinatorialPurgedCV(
        10, 3, embargo_pct=0.01, t1=purgedcv.make_t1(X.index, 5)).split(X))])
    it.close()
    # A fold goes degenerate partway through: both engines must yield the same
    # prefix and then raise the same error. Require exactly the first fold's
    # training size, so it passes and a later, more heavily purged one fails.
    t1 = purgedcv.make_t1(X.index, n // 10)
    sizes = [len(tr) for tr, _ in purgedcv.CombinatorialPurgedCV(10, 3, t1=t1).split(X)]
    assert min(sizes) < sizes[0]
    kw = dict(n_groups=10, n_test_groups=3, t1=t1, min_train_size=sizes[0])
    outcomes = []
    for engine in (purgedcv, purgedcv_rs):
        got = []
        with pytest.raises(ValueError) as exc:
            for split in engine.CombinatorialPurgedCV(**kw).split(X):
                got.append(split)
        outcomes.append((got, str(exc.value)))
    assert len(outcomes[0][0]) == len(outcomes[1][0]) > 0
    _assert_same_splits(outcomes[0][0], outcomes[1][0])
    assert outcomes[0][1] == outcomes[1][1]


@pytest.mark.parametrize("seed", range(40))
def test_masks_match_python_splits(seed):
    rng = np.random.default_rng(1000 + seed)
    n_groups = int(rng.integers(2, 8))
    n = int(rng.integers(n_groups, 300))
    index = pd.bdate_range("2000-01-03", periods=n)
    X = pd.DataFrame({"x": np.zeros(n)}, index=index)
    kw = dict(
        n_groups=n_groups,
        n_test_groups=int(rng.integers(1, n_groups)),
        embargo_pct=float(rng.choice([0.0, 0.05])),
        embargo_anchor=str(rng.choice(["label_end", "test_end"])),
        t1=_random_t1(rng, index, rng.choice(["none", "fixed", "random"])),
        min_train_size=0,
    )
    train, test = purgedcv_rs.CombinatorialPurgedCV(**kw).masks(X)
    py_splits = list(purgedcv.CombinatorialPurgedCV(**kw).split(X))
    assert train.shape == test.shape == (n, len(py_splits))
    assert train.dtype == test.dtype == np.bool_
    assert train.flags.f_contiguous and test.flags.f_contiguous
    for c, (py_tr, py_te) in enumerate(py_splits):
        np.testing.assert_array_equal(np.flatnonzero(train[:, c]), py_tr)
        np.testing.assert_array_equal(np.flatnonzero(test[:, c]), py_te)
    # CPCVPaths.is_test is the Python engine's own test-membership matrix.
    paths = purgedcv.CombinatorialPurgedCV(**kw).build_paths(X)
    np.testing.assert_array_equal(test, paths.is_test)


def test_masks_raise_on_degenerate_fold_and_warn_at_caller():
    X = pd.DataFrame({"x": np.zeros(126)}, index=pd.bdate_range("2020", periods=126))
    cv = purgedcv_rs.CombinatorialPurgedCV(6, 2, embargo_pct=0.01, t1=purgedcv.make_t1(X.index, 21))
    with pytest.raises(ValueError, match="retains 0 training"):
        cv.masks(X)
    with pytest.warns(UserWarning, match="groups is accepted") as rec:
        purgedcv_rs.CombinatorialPurgedCV(6, 2).masks(X, groups=np.zeros(126))
    assert rec[0].filename == __file__


@pytest.mark.parametrize("n_groups,n_test", [(2, 1), (3, 2), (6, 2), (8, 3), (10, 5), (12, 11)])
@pytest.mark.parametrize("n_extra", [0, 7, 250])
def test_build_paths_matches_python(n_groups, n_test, n_extra):
    n = n_groups + n_extra
    X = pd.DataFrame({"x": np.zeros(n)}, index=pd.bdate_range("2001", periods=n))
    py = purgedcv.CombinatorialPurgedCV(n_groups, n_test).build_paths(X)
    rs = purgedcv_rs.CombinatorialPurgedCV(n_groups, n_test).build_paths(X)
    assert isinstance(rs, purgedcv.CPCVPaths)
    for field in ("is_test", "paths", "path_folds"):
        a, b = getattr(py, field), getattr(rs, field)
        assert a.shape == b.shape and a.dtype == b.dtype, field
        assert b.flags.c_contiguous, field
        np.testing.assert_array_equal(a, b, err_msg=field)
    assert (rs.n_sims, rs.n_paths) == (py.n_sims, py.n_paths)
    assert rs.index.equals(py.index)
    pred = np.random.default_rng(0).normal(size=(n, py.n_sims))
    np.testing.assert_array_equal(rs.combine(pred), py.combine(pred))
    pd.testing.assert_frame_equal(rs.to_frame(pred), py.to_frame(pred))


def test_build_paths_rejects_bad_input_like_python():
    for engine in (purgedcv, purgedcv_rs):
        with pytest.raises(ValueError, match="must be >= n_groups"):
            engine.CombinatorialPurgedCV(6, 2).build_paths(np.zeros(5))
    with pytest.raises(RuntimeError, match="precondition violated"):
        purgedcv_rs._engine.build_paths(10, 3, [[0], [0, 1]], 1)
