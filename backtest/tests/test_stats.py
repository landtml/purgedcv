"""Layer 1: the statistics must be correct, calibrated, and hard to misuse."""

import dataclasses
import math

import numpy as np
import pytest

pb = pytest.importorskip("purgedcv_backtest")


# --------------------------------------------------------------------------- #
# Known values                                                                #
# --------------------------------------------------------------------------- #
def test_dsr_reproduces_published_example():
    # Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", numerical
    # example: annualised SR 2.5 over 5 years of daily data (1250 obs), skew
    # -3, kurtosis 10, 100 trials with annualised SR variance 0.5 -> DSR 0.9004.
    s = pb.SharpeStats(sharpe=2.5 / math.sqrt(250), n_obs=1250, skew=-3.0, kurtosis=10.0)
    dsr = pb.deflated_sharpe_ratio(s, n_trials=100, sr_variance=0.5 / 250)
    assert dsr == pytest.approx(0.9004, abs=5e-4)


def test_moments_of_a_normal_sample():
    r = np.random.default_rng(0).normal(0.01, 1.0, 200_000)
    s = pb.sharpe_stats(r)
    assert s.sharpe == pytest.approx(0.01, abs=0.005)
    assert s.skew == pytest.approx(0.0, abs=0.02)
    assert s.kurtosis == pytest.approx(3.0, abs=0.05)
    assert s.n_obs == r.size


def test_expected_max_sharpe_matches_simulated_maxima():
    rng = np.random.default_rng(1)
    for n_trials in (10, 100, 1000):
        simulated = rng.standard_normal((5000, n_trials)).max(axis=1).mean()
        assert pb.expected_max_sharpe(n_trials, 1.0) == pytest.approx(simulated, rel=0.03)
    assert pb.expected_max_sharpe(1, 1.0) == 0.0
    assert pb.expected_max_sharpe(50, 4.0) == pytest.approx(2 * pb.expected_max_sharpe(50, 1.0))


# --------------------------------------------------------------------------- #
# Calibration: the tests must flag noise at their nominal rate               #
# --------------------------------------------------------------------------- #
def test_psr_is_calibrated_under_the_null():
    rng = np.random.default_rng(2)
    psr = [pb.probabilistic_sharpe_ratio(r) for r in rng.normal(0, 1, (4000, 250))]
    assert 0.035 <= np.mean(np.array(psr) > 0.95) <= 0.065


def test_dsr_removes_selection_bias_that_psr_misses():
    # 50 skill-less strategies per experiment; pick the best, as a researcher
    # would. Naive PSR calls it significant most of the time. DSR, fed the
    # full trial log, should do so at roughly the nominal 5% rate.
    rng = np.random.default_rng(3)
    naive, deflated = [], []
    for _ in range(400):
        log = pb.TrialLog()
        for r in rng.normal(0, 1, (50, 250)):
            log.record(r)
        naive.append(pb.probabilistic_sharpe_ratio(log.best().stats) > 0.95)
        deflated.append(log.deflated_sharpe() > 0.95)
    assert np.mean(naive) > 0.6
    assert np.mean(deflated) < 0.08


def test_dsr_still_recognises_real_skill():
    rng = np.random.default_rng(4)
    log = pb.TrialLog()
    for r in rng.normal(0, 1, (49, 1000)):
        log.record(r, kind="noise")
    log.record(rng.normal(0.15, 1, 1000), kind="edge")
    assert log.best().params["kind"] == "edge"
    assert log.deflated_sharpe() > 0.99


# --------------------------------------------------------------------------- #
# Minimum track record length                                                 #
# --------------------------------------------------------------------------- #
def test_min_trl_is_the_length_where_psr_crosses_the_threshold():
    s = pb.sharpe_stats(np.random.default_rng(5).normal(0.08, 1, 500))
    trl = pb.min_track_record_length(s, alpha=0.05)
    assert math.isfinite(trl)
    above = dataclasses.replace(s, n_obs=math.ceil(trl))
    below = dataclasses.replace(s, n_obs=math.floor(trl))
    assert pb.probabilistic_sharpe_ratio(above) >= 0.95 > pb.probabilistic_sharpe_ratio(below)


def test_min_trl_penalises_negative_skew_and_is_infinite_without_edge():
    base = pb.SharpeStats(sharpe=0.1, n_obs=500, skew=0.0, kurtosis=3.0)
    crashy = dataclasses.replace(base, skew=-2.0, kurtosis=9.0)
    assert pb.min_track_record_length(crashy) > pb.min_track_record_length(base)
    assert pb.min_track_record_length(dataclasses.replace(base, sharpe=0.0)) == math.inf
    assert pb.min_track_record_length(base, sr_benchmark=0.2) == math.inf


# --------------------------------------------------------------------------- #
# PBO                                                                         #
# --------------------------------------------------------------------------- #
def test_pbo_of_pure_noise_is_about_one_half():
    m = np.random.default_rng(6).normal(0, 1, (2000, 30))
    assert 0.3 <= pb.probability_of_backtest_overfitting(m).pbo <= 0.7


def test_pbo_of_a_genuinely_better_trial_is_near_zero():
    rng = np.random.default_rng(7)
    m = rng.normal(0, 1, (2000, 30))
    m[:, 11] += 0.2
    res = pb.probability_of_backtest_overfitting(m, n_splits=10)
    assert res.pbo < 0.05
    assert res.prob_oos_loss < 0.05


def test_pbo_split_sharpes_match_direct_computation():
    rng = np.random.default_rng(8)
    m = rng.normal(0.01, 1, (97, 5))  # 97 rows: unequal blocks
    res = pb.probability_of_backtest_overfitting(m, n_splits=4)
    blocks = np.array_split(np.arange(97), 4)
    # First combination is blocks (0, 1) in sample.
    is_rows = np.concatenate([blocks[0], blocks[1]])
    oos_rows = np.concatenate([blocks[2], blocks[3]])
    is_sr = [pb.sharpe_ratio(m[is_rows, j]) for j in range(5)]
    best = int(np.argmax(is_sr))
    assert res.is_sharpe[0] == pytest.approx(is_sr[best], rel=1e-9)
    assert res.oos_sharpe[0] == pytest.approx(pb.sharpe_ratio(m[oos_rows, best]), rel=1e-9)
    assert len(res.logits) == math.comb(4, 2)


def test_pbo_input_validation():
    m = np.zeros((100, 3))
    with pytest.raises(ValueError, match="even"):
        pb.probability_of_backtest_overfitting(m, n_splits=5)
    with pytest.raises(ValueError, match="at least 2"):
        pb.probability_of_backtest_overfitting(m[:, :1])
    with pytest.raises(ValueError, match="2 observations per block"):
        pb.probability_of_backtest_overfitting(m[:20], n_splits=16)
    bad = m.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        pb.probability_of_backtest_overfitting(bad)


# --------------------------------------------------------------------------- #
# Hard to misuse                                                              #
# --------------------------------------------------------------------------- #
def test_dsr_has_no_default_trial_count():
    with pytest.raises(TypeError):
        pb.deflated_sharpe_ratio(np.random.default_rng(0).normal(size=100))


@pytest.mark.parametrize(
    "returns,match",
    [([1.0], "at least 2"), ([1.0, 1.0, 1.0], "constant"), ([0.1, np.nan], "NaN"), (np.zeros((3, 2)), "one-dimensional")],
)
def test_return_validation(returns, match):
    with pytest.raises(ValueError, match=match):
        pb.sharpe_stats(returns)


def test_trial_log_is_append_only_and_frozen():
    log = pb.TrialLog()
    src = np.random.default_rng(9).normal(size=100)
    t = log.record(src, name="a", lookback=20)
    src[:] = 0.0  # mutating the caller's array must not change the record
    assert t.returns.std() > 0
    with pytest.raises(ValueError):
        t.returns[0] = 1.0
    with pytest.raises(TypeError):
        t.params["lookback"] = 5
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.name = "b"
    for attr in ("remove", "pop", "clear", "__delitem__", "__setitem__"):
        assert not hasattr(log, attr)


def test_more_trials_raise_the_bar_at_fixed_dispersion():
    bars = [pb.expected_max_sharpe(n, 0.002) for n in range(1, 1001)]
    assert all(a < b for a, b in zip(bars, bars[1:]))


def test_every_extra_trial_makes_the_same_result_harder_to_believe():
    # With the dispersion re-estimated from the log, DSR is not strictly
    # monotone trial by trial (the variance estimate moves), but piling on
    # trials must drag it down.
    rng = np.random.default_rng(10)
    log = pb.TrialLog()
    star = log.record(rng.normal(0.15, 1, 750), name="star")
    for r in rng.normal(0, 1, (10, 750)):
        log.record(r)
    few = log.deflated_sharpe(star)
    for r in rng.normal(0, 1, (190, 750)):
        log.record(r)
    many = log.deflated_sharpe(star)
    assert many < few
    summary = log.summary(star)
    assert summary["n_trials"] == 201 and summary["dsr"] == many
    assert summary["expected_max_sharpe"] == log.expected_max_sharpe()


def test_trial_log_guards():
    log = pb.TrialLog()
    with pytest.raises(ValueError, match="at least 1"):
        log.best()
    log.record(np.arange(10.0))
    log.record(np.arange(12.0))
    with pytest.raises(ValueError, match="same observations"):
        log.pbo()
    foreign = pb.TrialLog().record(np.arange(10.0))
    with pytest.raises(ValueError, match="not recorded in this log"):
        log.deflated_sharpe(foreign)
