"""Planted-flaw tests: each guard must catch its flaw and pass the clean twin."""

import numpy as np
import pandas as pd
import pytest

pb = pytest.importorskip("purgedcv_backtest")
CAL = pb.EQUITY_DAILY


def frame(x, name="A"):
    idx = pd.date_range("2000-01-03", periods=len(x), freq="B")
    return pd.DataFrame({name: np.asarray(x, dtype=float)}, index=idx)


def slow_signal_market(T=2000, beta=0.0015, phi=0.95, seed=0):
    """Returns genuinely predictable from a persistent, honestly-timed signal:
    r_t = beta * x_{t-1} + noise. Weight at t is x_t, known at t."""
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + np.sqrt(1 - phi**2) * rng.standard_normal()
    r = np.zeros(T)
    r[1:] = beta * x[:-1] + 0.01 * rng.standard_normal(T - 1)
    return frame(np.clip(x / 2, -1, 1)), frame(r)


def noise_market(T=500, seed=0):
    rng = np.random.default_rng(seed)
    x = np.convolve(rng.standard_normal(T + 20), np.ones(20) / 20, mode="valid")[:T]
    return frame(np.sign(x)), frame(0.01 * rng.standard_normal(T))


# --------------------------------------------------------------------------- #
# F3: permutation test                                                        #
# --------------------------------------------------------------------------- #
def test_f3_noise_strategies_get_uniform_p_values():
    ps = []
    for seed in range(60):
        w, r = noise_market(seed=seed)
        ps.append(pb.permutation_test(w, r, costs=pb.ZeroCost(), n_permutations=199, seed=seed).p_value)
    ps = np.array(ps)
    assert np.mean(ps < 0.05) <= 0.15
    assert 0.35 <= ps.mean() <= 0.65


def test_f3_real_edge_is_significant():
    w, r = slow_signal_market()
    res = pb.permutation_test(w, r, costs=pb.LinearCost(1), n_permutations=499)
    assert res.p_value < 0.01
    assert res.observed > np.quantile(res.null, 0.99)


# --------------------------------------------------------------------------- #
# F2: lag cliff                                                               #
# --------------------------------------------------------------------------- #
def test_f2_leaky_feature_is_flagged():
    rng = np.random.default_rng(1)
    r = 0.01 * rng.standard_normal(2000)
    # The planted flaw: a "feature" at t that already contains r_{t+1}.
    leak = np.append(r[1:], 0.0) + 0.01 * rng.standard_normal(2000)
    prof = pb.lag_profile(frame(np.sign(leak)), frame(r), costs=pb.ZeroCost())
    assert prof.sharpes[0] > 0.3
    assert prof.flagged


def test_f2_slow_genuine_signal_is_not_flagged():
    w, r = slow_signal_market()
    prof = pb.lag_profile(w, r, costs=pb.ZeroCost())
    assert prof.base_psr > 0.99
    assert not prof.flagged
    assert prof.cliff_ratio > 0.8  # information decays with the signal's persistence


def test_f2_noise_is_not_flagged():
    w, r = noise_market(T=2000, seed=3)
    assert not pb.lag_profile(w, r, costs=pb.ZeroCost()).flagged


# --------------------------------------------------------------------------- #
# F5: costs                                                                   #
# --------------------------------------------------------------------------- #
def high_turnover_edge(T=20000, beta=0.0005, seed=2):
    """A real but thin edge that flips position about every other bar."""
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], T)
    r = np.zeros(T)
    r[1:] = beta * s[:-1] + 0.005 * rng.standard_normal(T - 1)
    return frame(s), frame(r)


def test_f5_break_even_cost_is_located():
    w, r = high_turnover_edge()
    curve = pb.cost_sensitivity(w, r)
    # Edge 5 bps per bar, turnover ~1 per bar: break-even near 5 bps.
    assert curve.sharpes[0] > 0
    assert 3.0 < curve.break_even_bps < 7.5
    assert list(curve.sharpes) == sorted(curve.sharpes, reverse=True)


def test_f5_edge_that_dies_at_realistic_cost_is_falsified():
    w, r = high_turnover_edge()
    free = pb.falsify(w, r, costs=pb.ZeroCost(), calendar=CAL, n_permutations=99)
    assert any(f.check == "F5 costs" and f.status is pb.Status.WARN for f in free.findings)
    costly = pb.falsify(w, r, costs=pb.LinearCost(10), calendar=CAL, n_permutations=99)
    net = next(f for f in costly.findings if f.check == "F5 net of costs")
    assert net.status is pb.Status.FAIL
    assert not costly.survived


def test_f5_unfunded_shorts_are_flagged():
    w, r = slow_signal_market()
    bare = pb.falsify(w, r, costs=pb.LinearCost(1), calendar=CAL, n_permutations=99)
    assert any(f.check == "F5 carry" for f in bare.findings)
    funded = pb.falsify(w, r, costs=pb.LinearCost(1), calendar=CAL, n_permutations=99,
                        carry=pb.BorrowCarry(0.01, CAL))
    assert not any(f.check == "F5 carry" for f in funded.findings)


# --------------------------------------------------------------------------- #
# The verdict                                                                 #
# --------------------------------------------------------------------------- #
def test_verdicts():
    w, r = slow_signal_market()
    good = pb.falsify(w, r, costs=pb.LinearCost(1), calendar=CAL,
                      carry=pb.BorrowCarry(0.01, CAL), n_permutations=199)
    assert good.survived
    assert all(f.status is pb.Status.PASS for f in good.findings), str(good)
    assert str(good).startswith("Verdict: SURVIVED")

    w, r = noise_market(T=2000, seed=11)
    bad = pb.falsify(w, r, costs=pb.LinearCost(1), calendar=CAL,
                     carry=pb.BorrowCarry(0.01, CAL), n_permutations=199)
    assert not bad.survived
    assert str(bad).startswith("Verdict: FALSIFIED")
