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
