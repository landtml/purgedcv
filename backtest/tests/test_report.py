"""The report is the contract with the dashboard: strict JSON, verdict first."""

import json

import numpy as np
import pandas as pd
import pytest

pb = pytest.importorskip("purgedcv_backtest")
purgedcv = pytest.importorskip("purgedcv")


def market(T=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=T)
    x = np.cumsum(rng.standard_normal(T)) / 10
    r = np.append(0.0, 0.002 * np.sign(x[:-1]) + 0.01 * rng.standard_normal(T - 1))
    return pd.DataFrame({"A": np.sign(x)}, index=idx), pd.DataFrame({"A": r}, index=idx)


@pytest.fixture(scope="module")
def pieces():
    w, r = market()
    v = pb.falsify(w, r, costs=pb.LinearCost(1), calendar=pb.EQUITY_DAILY, n_permutations=99)
    log = pb.TrialLog()
    for k in range(4):
        log.record(pb.simulate(w.shift(k, fill_value=0.0), r, costs=pb.LinearCost(1),
                               calendar=pb.EQUITY_DAILY).net.to_numpy(), lag=k)
    paths = purgedcv.CombinatorialPurgedCV(4, 2).build_paths(r)
    sim_w = np.repeat(w.to_numpy()[:, None, :], paths.is_test.shape[1], axis=1)
    pr = pb.cpcv_path_returns(paths, sim_w, r, costs=pb.LinearCost(1), calendar=pb.EQUITY_DAILY)
    return v, log, pr


def test_full_report_is_strict_json(pieces, tmp_path):
    v, log, pr = pieces
    rep = pb.build_report(verdict=v, trials=log, paths=pr, title="t")
    out = pb.write_report(tmp_path / "r.json", rep)
    back = json.loads(out.read_text(), parse_constant=lambda c: pytest.fail(f"non-strict token {c}"))
    assert back["schema"] == pb.SCHEMA
    assert back["verdict"]["survived"] == v.survived
    assert [f["check"] for f in back["verdict"]["findings"]] == [f.check for f in v.findings]
    assert back["trials"]["n_trials"] == 4 and back["trials"]["best"] == log.best().number
    assert back["paths"]["summary"]["n_paths"] == pr.summary()["n_paths"]
    assert len(back["paths"]["equity"]) == len(pr.paths)
    assert back["assumptions"]["lag"] == 1


def test_series_are_thinned_but_keep_the_last_point(pieces):
    v, _, _ = pieces
    long_idx = pd.bdate_range("2000-01-03", periods=5000)
    w = pd.DataFrame({"A": np.ones(5000)}, index=long_idx)
    r = pd.DataFrame({"A": np.full(5000, 1e-4)}, index=long_idx)
    v = pb.falsify(w, r, costs=pb.ZeroCost(), calendar=pb.EQUITY_DAILY, n_permutations=19)
    s = pb.build_report(verdict=v)["series"]
    assert len(s["index"]) <= 1600
    assert s["index"][-1].startswith(str(long_idx[-1].date()))
    assert s["equity"][-1] == pytest.approx(float(v.result.equity().iloc[-1]), rel=1e-5)


def test_non_finite_numbers_become_null(pieces):
    v, _, _ = pieces
    # A losing strategy has infinite MinTRL; it must not leak as Infinity.
    w, r = market(seed=3)
    bad = pb.falsify(-w, r, costs=pb.LinearCost(50), calendar=pb.EQUITY_DAILY, n_permutations=19)
    rep = pb.build_report(verdict=bad)
    assert rep["performance"]["min_track_record_length"] is None
    json.dumps(rep, allow_nan=False)


def test_sections_are_optional_but_not_all(pieces):
    _, log, _ = pieces
    rep = pb.build_report(trials=log)
    assert rep["verdict"] is None and rep["trials"] is not None
    with pytest.raises(ValueError, match="at least one"):
        pb.build_report()
