"""Layer 2 engine: conventions, cost/carry/sizing models, CPCV path returns."""

import numpy as np
import pandas as pd
import pytest

pb = pytest.importorskip("purgedcv_backtest")
purgedcv = pytest.importorskip("purgedcv")

CAL = pb.EQUITY_DAILY


def panel(values, cols=("A", "B")):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.DataFrame(np.asarray(values, dtype=float), index=idx, columns=list(cols))


# --------------------------------------------------------------------------- #
# Conventions                                                                 #
# --------------------------------------------------------------------------- #
def test_hand_computed_simulation():
    w = panel([[1.0, 0.0], [0.5, -0.5], [0.0, 0.0], [0.0, 1.0]])
    r = panel([[0.10, 0.20], [0.01, 0.02], [-0.02, 0.04], [0.03, -0.01]])
    res = pb.simulate(w, r, costs=pb.LinearCost(10), calendar=CAL, lag=1)
    held = np.array([[0, 0], [1, 0], [0.5, -0.5], [0, 0]])
    np.testing.assert_allclose(res.held.to_numpy(), held)
    gross = (held * r.to_numpy()).sum(axis=1)
    turnover = np.abs(np.diff(held, axis=0, prepend=0)).sum(axis=1)
    np.testing.assert_allclose(res.gross, gross)
    np.testing.assert_allclose(res.turnover, turnover)
    np.testing.assert_allclose(res.costs, turnover * 10 / 1e4)
    np.testing.assert_allclose(res.net, gross - turnover * 1e-3)
    np.testing.assert_allclose(res.equity(), np.cumprod(1 + res.net))
    assert res.annualized_sharpe == pytest.approx(res.sharpe * np.sqrt(252))


def test_lag_shifts_positions():
    w = panel(np.eye(5, 2))
    r = panel(np.ones((5, 2)) * 0.01)
    held = pb.simulate(w, r, costs=pb.ZeroCost(), calendar=CAL, lag=3).held.to_numpy()
    np.testing.assert_array_equal(held[3:], w.to_numpy()[:2])
    assert (held[:3] == 0).all()


def test_f1_same_bar_execution_is_refused():
    w = r = panel(np.zeros((5, 2)))
    with pytest.raises(ValueError, match="look-ahead"):
        pb.simulate(w, r, costs=pb.ZeroCost(), calendar=CAL, lag=0)


def test_no_silent_defaults_or_alignment():
    w = r = panel(np.zeros((5, 2)))
    with pytest.raises(TypeError):
        pb.simulate(w, r, calendar=CAL)  # no cost model
    with pytest.raises(TypeError):
        pb.simulate(w, r, costs=pb.ZeroCost())  # no calendar
    with pytest.raises(ValueError, match="same index and columns"):
        pb.simulate(w, r.iloc[1:], costs=pb.ZeroCost(), calendar=CAL)
    with pytest.raises(ValueError, match="same index and columns"):
        pb.simulate(w, r[["B", "A"]], costs=pb.ZeroCost(), calendar=CAL)
    bad = w.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        pb.simulate(bad, r, costs=pb.ZeroCost(), calendar=CAL)


def test_nan_returns_only_allowed_where_flat():
    w = panel([[0.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    r = panel([[np.nan, 0.01], [np.nan, 0.01], [np.nan, 0.01]])
    # Asset A is not yet listed: fine while flat...
    pb.simulate(w.iloc[:2], r.iloc[:2], costs=pb.ZeroCost(), calendar=CAL)
    # ...but holding it over a NaN return is an error, not a silent zero.
    with pytest.raises(ValueError, match="NaN on a bar where a position is held"):
        pb.simulate(panel([[1.0, 0], [1.0, 0], [1.0, 0]]), r, costs=pb.ZeroCost(), calendar=CAL)


# --------------------------------------------------------------------------- #
# Cost, carry and sizing models                                               #
# --------------------------------------------------------------------------- #
def test_per_instrument_fees_from_instruments():
    insts = [
        pb.Instrument("A", pb.AssetClass.EQUITY, CAL, fee_bps=2),
        pb.Instrument("B", pb.AssetClass.PERPETUAL, pb.CRYPTO_HOURLY, fee_bps=5),
    ]
    cost = pb.LinearCost.from_instruments(insts).bind(pd.RangeIndex(1), pd.Index(["A", "B"]))
    assert cost(np.array([[1.0, -2.0]]))[0] == pytest.approx(2e-4 + 10e-4)
    with pytest.raises(ValueError, match="missing"):
        pb.LinearCost({"A": 1}).bind(pd.RangeIndex(1), pd.Index(["A", "B"]))


def test_square_root_impact_formula_and_capacity():
    idx, cols = pd.RangeIndex(2), pd.Index(["A"])
    vol = pd.DataFrame({"A": [0.02, 0.02]})
    adv = pd.DataFrame({"A": [1e6, 1e6]})
    small = pb.SquareRootImpact(vol, adv, capital=1e5, coef=0.5, half_spread_bps=1).bind(idx, cols)
    big = pb.SquareRootImpact(vol, adv, capital=1e7, coef=0.5, half_spread_bps=1).bind(idx, cols)
    trade = np.array([[0.1], [0.0]])
    expected = 0.1 * (1e-4 + 0.5 * 0.02 * np.sqrt(0.1 * 1e5 / 1e6))
    assert small(trade)[0] == pytest.approx(expected)
    assert small(trade)[1] == 0
    assert big(trade)[0] > small(trade)[0]  # capacity: more capital, more impact
    with pytest.raises(ValueError, match="does not cover"):
        pb.SquareRootImpact(vol, adv, capital=1e5).bind(pd.RangeIndex(3), cols)


def test_carry_models():
    idx, cols = pd.RangeIndex(1), pd.Index(["A", "B"])
    held = np.array([[0.5, -0.5]])
    funding = pb.FundingCarry(pd.DataFrame({"A": [1e-4], "B": [1e-4]})).bind(idx, cols)
    assert funding(held)[0] == pytest.approx(0.0)  # long pays, short receives
    assert funding(np.array([[1.0, 0.0]]))[0] == pytest.approx(1e-4)
    borrow = pb.BorrowCarry(0.0252, CAL).bind(idx, cols)
    assert borrow(held)[0] == pytest.approx(0.5 * 0.0252 / 252)  # shorts only
    composite = pb.CompositeCost(pb.LinearCost(1), pb.LinearCost(2)).bind(idx, cols)
    assert composite(np.array([[1.0, 0.0]]))[0] == pytest.approx(3e-4)


def test_sizers():
    pred = panel([[0.3, -0.1], [0.0, 0.0], [2.0, 5.0]])
    w = pb.SignSizer(gross=2.0)(pred)
    np.testing.assert_allclose(w.to_numpy(), [[1, -1], [0, 0], [1, 1]])
    prob = panel([[0.5, 0.9], [0.1, 0.99]])
    size = pb.ProbabilitySizer(max_weight=1.0)(prob).to_numpy()
    assert size[0, 0] == 0
    assert size[0, 1] == pytest.approx(-size[1, 0])  # symmetric around 0.5
    assert 0 < size[0, 1] < size[1, 1] < 1  # monotone, bounded
    with pytest.raises(ValueError):
        pb.ProbabilitySizer()(panel([[1.5, 0.5]]))


# --------------------------------------------------------------------------- #
# F6: CPCV path returns                                                       #
# --------------------------------------------------------------------------- #
def test_f6_each_path_is_its_own_simulation():
    rng = np.random.default_rng(0)
    n, n_inst = 240, 2
    r = panel(rng.normal(0, 0.01, (n, n_inst)))
    cv = purgedcv.CombinatorialPurgedCV(6, 2)
    paths = cv.build_paths(r)
    sim_w = rng.normal(0, 1, (n, paths.n_sims, n_inst))
    res = pb.cpcv_path_returns(paths, sim_w, r, costs=pb.LinearCost(5), calendar=CAL)
    assert len(res.paths) == paths.n_paths == cv.get_n_paths()
    for p, path_res in enumerate(res.paths):
        # The path's positions come from the simulation the path map names...
        expected_w = sim_w[np.arange(n), paths.paths[:, p], :]
        direct = pb.simulate(pd.DataFrame(expected_w, index=r.index, columns=r.columns), r,
                             costs=pb.LinearCost(5), calendar=CAL)
        # ...and its returns and costs, including hand-over trades between
        # segments, equal a direct simulation of that position series.
        np.testing.assert_allclose(path_res.net, direct.net)
        np.testing.assert_allclose(path_res.costs, direct.costs)
    assert res.net.shape == (n, paths.n_paths)
    assert set(res.summary()) >= {"sharpe_mean", "sharpe_min", "share_positive"}


def test_f6_only_test_rows_are_read():
    n = 120
    r = panel(np.random.default_rng(1).normal(0, 0.01, (n, 1)), cols=("A",))
    paths = purgedcv.CombinatorialPurgedCV(4, 2).build_paths(r)
    sim_w = np.where(paths.is_test, 1.0, np.nan)  # training rows poisoned
    res = pb.cpcv_path_returns(paths, sim_w, r, costs=pb.ZeroCost(), calendar=CAL)
    assert all(np.isfinite(p.net).all() for p in res.paths)
    with pytest.raises(ValueError, match="n_sims"):
        pb.cpcv_path_returns(paths, sim_w[:, :-1], r, costs=pb.ZeroCost(), calendar=CAL)
