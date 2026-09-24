"""Build a demo report for the dashboard from a synthetic market.

    python backtest/examples/demo_report.py [out.json]

The market has a real, honestly timed edge: returns follow a persistent
signal observed with noise. A small parameter search (signal smoothing and
a dead zone) is recorded in a TrialLog so the dashboard can show what the
search cost in deflated Sharpe ratio and PBO. The data is synthetic; the
numbers illustrate the report, not any real strategy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import purgedcv
import purgedcv_backtest as pb

ASSETS = ("ES", "NQ", "ZN", "CL")
T = 2520  # ten years of trading days


def market(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns r_t = beta * x_{t-1} + noise; the strategy sees x through noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=T)
    x = np.zeros((T, len(ASSETS)))
    for t in range(1, T):
        x[t] = 0.97 * x[t - 1] + np.sqrt(1 - 0.97**2) * rng.standard_normal(len(ASSETS))
    r = np.zeros_like(x)
    r[1:] = 0.0009 * x[:-1] + 0.011 * rng.standard_normal((T - 1, len(ASSETS)))
    observed = x + 1.2 * rng.standard_normal(x.shape)
    return (pd.DataFrame(observed, index=idx, columns=ASSETS),
            pd.DataFrame(r, index=idx, columns=ASSETS))


def weights(signal: pd.DataFrame, span: int, band: float) -> pd.DataFrame:
    """Smooth the noisy signal, stay flat inside a +-band dead zone, then
    size equal-weight long/short at gross exposure 1."""
    smooth = signal.ewm(span=span, adjust=False).mean()
    return pb.SignSizer(gross=1.0)(smooth.where(smooth.abs() > band, 0.0))


def main(out: Path) -> None:
    signal, returns = market()
    costs = pb.LinearCost(2.0)
    carry = pb.BorrowCarry(0.01, pb.FUTURES_DAILY)
    cal = pb.FUTURES_DAILY

    # The search: every configuration tried is recorded, not just the winner.
    log = pb.TrialLog()
    for span in (1, 3, 5, 10, 20, 40, 80):
        for band in (0.0, 0.2, 0.5):
            w = weights(signal, span, band)
            res = pb.simulate(w, returns, costs=costs, calendar=cal, carry=carry)
            log.record(res.net.to_numpy(), name=f"ewm{span}-band{band}", span=span, band=band)

    best = log.best()
    chosen = weights(signal, best.params["span"], best.params["band"])
    verdict = pb.falsify(chosen, returns, costs=costs, calendar=cal, carry=carry,
                         n_permutations=500)

    # CPCV: each simulation re-runs the span search on its training rows only
    # and trades the winner on its test rows, so the paths show how much the
    # selection itself varies.
    cv = purgedcv.CombinatorialPurgedCV(8, 3, embargo_pct=0.01)
    paths = cv.build_paths(returns)
    sim_w = np.zeros((T, paths.is_test.shape[1], len(ASSETS)))
    candidates = {(t.params["span"], t.params["band"]):
                  weights(signal, t.params["span"], t.params["band"]).to_numpy() for t in log}
    fwd = returns.shift(-1).fillna(0.0).to_numpy()
    for s, (train, test) in enumerate(cv.split(returns)):
        def train_sharpe(w):
            pnl = (w[train] * fwd[train]).sum(axis=1)
            return pnl.mean() / pnl.std()
        pick = max(candidates, key=lambda k: train_sharpe(candidates[k]))
        sim_w[test, s, :] = candidates[pick][test]
    path_res = pb.cpcv_path_returns(paths, sim_w, returns, costs=costs, calendar=cal, carry=carry)

    report = pb.build_report(verdict=verdict, trials=log, paths=path_res,
                             title=f"Futures trend demo ({best.name})")
    pb.write_report(out, report)
    print(verdict)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[1] / "dashboard" / "public" / "report.json"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
