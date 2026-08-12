"""Integration tests against a real scikit-learn install.

The rest of the suite exercises the splitter directly and never imports sklearn,
so it would pass unchanged even if the sklearn integration were completely
broken -- ``_splitter.py`` silently falls back to ``_Base = object`` when the
import fails. These tests pin the compatibility the README advertises.

``cross_val_predict`` is deliberately absent: it requires the test folds to
partition the sample, and CPCV's folds overlap by construction (each observation
is tested ``n_paths`` times). ``build_paths``/``CPCVPaths.combine`` is the
supported way to assemble out-of-sample predictions -- see
:func:`test_cross_val_predict_is_unsupported_by_construction`.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from purgedcv import CombinatorialPurgedCV, make_t1

sklearn = pytest.importorskip("sklearn", reason="sklearn integration tests")

from sklearn.base import clone  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    learning_curve,
    permutation_test_score,
)
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

N_SAMPLES = 500
N_SPLITS = 15  # C(6, 2)


@pytest.fixture
def data():
    idx = pd.bdate_range("2015-01-01", periods=N_SAMPLES)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.standard_normal(N_SAMPLES),
                      "b": rng.standard_normal(N_SAMPLES)}, index=idx)
    y = pd.Series(rng.standard_normal(N_SAMPLES), index=idx)
    return X, y, make_t1(idx, 21)


@pytest.fixture
def cv(data):
    _, _, t1 = data
    return CombinatorialPurgedCV(6, 2, embargo_pct=0.01, t1=t1)


# --------------------------------------------------------------------------- #
# The splitter is accepted by sklearn's cross-validation helpers               #
# --------------------------------------------------------------------------- #
def test_cross_val_score(data, cv):
    X, y, _ = data
    scores = cross_val_score(Ridge(), X, y, cv=cv)
    assert scores.shape == (N_SPLITS,)
    assert np.isfinite(scores).all(), "no fold may score NaN on a healthy config"


def test_cross_validate(data, cv):
    X, y, _ = data
    out = cross_validate(Ridge(), X, y, cv=cv, return_train_score=True)
    assert len(out["test_score"]) == N_SPLITS
    assert np.isfinite(out["train_score"]).all()


def test_grid_search(data, cv):
    X, y, _ = data
    gs = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0, 10.0]}, cv=cv)
    gs.fit(X, y)
    assert gs.best_params_["alpha"] in (0.1, 1.0, 10.0)
    assert len(gs.cv_results_["mean_test_score"]) == 3


def test_randomized_search(data, cv):
    X, y, _ = data
    rs = RandomizedSearchCV(
        Ridge(), {"alpha": [0.1, 1.0, 10.0]}, n_iter=2, cv=cv, random_state=0
    )
    rs.fit(X, y)
    assert hasattr(rs, "best_estimator_")


def test_pipeline(data, cv):
    X, y, _ = data
    pipe = make_pipeline(StandardScaler(), Ridge())
    assert cross_val_score(pipe, X, y, cv=cv).shape == (N_SPLITS,)


def test_learning_curve(data, cv):
    X, y, _ = data
    sizes, train, test = learning_curve(Ridge(), X, y, cv=cv, train_sizes=[0.5, 1.0])
    assert train.shape[1] == N_SPLITS and test.shape[1] == N_SPLITS


def test_permutation_test_score(data, cv):
    X, y, _ = data
    score, perm, pvalue = permutation_test_score(
        Ridge(), X, y, cv=cv, n_permutations=3, random_state=0
    )
    assert 0.0 <= pvalue <= 1.0


# --------------------------------------------------------------------------- #
# Parallel execution: the splitter is stateless and picklable                  #
# --------------------------------------------------------------------------- #
def test_parallel_cross_val_score_matches_serial(data, cv):
    X, y, _ = data
    np.testing.assert_allclose(
        cross_val_score(Ridge(), X, y, cv=cv, n_jobs=2),
        cross_val_score(Ridge(), X, y, cv=cv),
    )


def test_grid_search_parallel(data, cv):
    X, y, _ = data
    gs = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=cv, n_jobs=2)
    gs.fit(X, y)
    assert hasattr(gs, "best_estimator_")


def test_splitter_survives_pickle_roundtrip(data, cv):
    X, _, t1 = data
    restored = pickle.loads(pickle.dumps(cv))
    for (a_tr, a_te), (b_tr, b_te) in zip(cv.split(X), restored.split(X)):
        np.testing.assert_array_equal(a_tr, b_tr)
        np.testing.assert_array_equal(a_te, b_te)


# --------------------------------------------------------------------------- #
# Documented limits                                                            #
# --------------------------------------------------------------------------- #
def test_cross_val_predict_is_unsupported_by_construction(data, cv):
    # CPCV test folds overlap on purpose, so no partition exists for sklearn to
    # assemble. build_paths()/combine() is the supported replacement.
    X, y, _ = data
    with pytest.raises(ValueError, match="partition"):
        cross_val_predict(Ridge(), X, y, cv=cv)


def test_build_paths_is_the_cross_val_predict_replacement(data, cv):
    X, y, _ = data
    paths = cv.build_paths(X)
    pred = np.zeros((len(X), paths.n_sims))
    for c, (train_idx, test_idx) in enumerate(cv.split(X)):
        model = Ridge().fit(X.iloc[train_idx], y.iloc[train_idx])
        pred[test_idx, c] = model.predict(X.iloc[test_idx])
    frame = paths.to_frame(pred)
    assert frame.shape == (len(X), paths.n_paths)
    assert frame.index.equals(X.index)
    assert np.isfinite(frame.to_numpy()).all()


def test_clone_fails_like_sklearns_own_splitters(data, cv):
    # BaseCrossValidator deliberately does not inherit BaseEstimator, so splitters
    # are passed by reference rather than cloned. KFold behaves identically; this
    # pins the fact that we are no worse than the stdlib splitters.
    from sklearn.model_selection import KFold

    for splitter in (KFold(5), cv):
        with pytest.raises(TypeError):
            clone(splitter)


def test_splitter_is_reusable_inside_a_cloned_search(data, cv):
    # The realistic path: the *estimator* is cloned, the cv object is shared.
    X, y, _ = data
    gs = GridSearchCV(Ridge(), {"alpha": [1.0]}, cv=cv)
    clone(gs).fit(X, y)
    assert gs.get_params()["cv"] is cv
