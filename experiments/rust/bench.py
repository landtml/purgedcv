"""Benchmark the Rust engine against the pure-Python splitter.

    python experiments/rust/bench.py

Timings are best-of-``REPEATS`` wall clock for materialising every split
(the generator is fully consumed), then one end-to-end ``cross_val_score``
to show how much of a real run the splitter accounts for.
"""

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

import purgedcv
import purgedcv_rs

REPEATS = 5


def best_of(fn, repeats=REPEATS):
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def frame(n, cols=1):
    index = pd.date_range("1970-01-01", periods=n, freq="min")
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(n, cols)), index=index)


def random_t1(index, max_horizon, seed=0):
    rng = np.random.default_rng(seed)
    n = len(index)
    ends = np.minimum(np.arange(n) + rng.integers(1, max_horizon, n), n - 1)
    return pd.Series(index[ends], index=index)


def splits_table():
    print("Materialise every split (ms, best of 5)")
    header = f"{'n':>10} {'N,k':>6} {'t1':>7} {'splits':>6} | {'python':>9} {'rust':>9} {'rust ||':>9} | {'speedup':>7} {'||':>6}"
    print(header)
    print("-" * len(header))
    for n, N, k in [
        (5_000, 10, 2),
        (100_000, 10, 2),
        (1_000_000, 10, 2),
        (100_000, 16, 4),
    ]:
        X = frame(n)
        for kind in ("fixed", "random"):
            t1 = purgedcv.make_t1(X.index, 21) if kind == "fixed" else random_t1(X.index, 42)
            kw = dict(n_groups=N, n_test_groups=k, embargo_pct=0.01, t1=t1)
            py = purgedcv.CombinatorialPurgedCV(**kw)
            rs = purgedcv_rs.CombinatorialPurgedCV(**kw)
            # t1 resolution is shared Python code in both engines; time it once
            # so the table isolates what the engine actually changes.
            t_py = best_of(lambda: list(py.split(X)))
            t_rs = best_of(lambda: list(rs.split(X)))
            t_par = best_of(lambda: rs.split_all(X))
            print(
                f"{n:>10,} {f'{N},{k}':>6} {kind:>7} {py.get_n_splits():>6} | "
                f"{t_py * 1e3:>9.1f} {t_rs * 1e3:>9.1f} {t_par * 1e3:>9.1f} | "
                f"{t_py / t_rs:>6.1f}x {t_py / t_par:>5.1f}x"
            )


def overhead_table():
    print("\nWhere the time goes at n=1,000,000, N=10, k=2 (ms)")
    X = frame(1_000_000)
    t1 = purgedcv.make_t1(X.index, 21)
    index = X.index
    t_resolve = best_of(lambda: purgedcv._splitter._end_positions(index, t1))
    end_pos = purgedcv._splitter._end_positions(index, t1)
    rs = purgedcv_rs.CombinatorialPurgedCV(10, 2, embargo_pct=0.01, t1=t1)
    t_kernel = best_of(lambda: list(rs._iter_splits(len(index), index, end_pos)))
    print(f"  t1 -> end_pos resolution (shared Python/pandas): {t_resolve * 1e3:8.1f}")
    print(f"  45 splits in the Rust kernel                   : {t_kernel * 1e3:8.1f}")


def masks_table():
    print("\nTrain + test masks, (n_samples, n_sims) bool, t1=make_t1(21) (ms)")
    for n, N, k in [(100_000, 10, 2), (1_000_000, 10, 2), (100_000, 16, 4)]:
        X = frame(n)
        kw = dict(n_groups=N, n_test_groups=k, embargo_pct=0.01, t1=purgedcv.make_t1(X.index, 21))
        py = purgedcv.CombinatorialPurgedCV(**kw)
        rs = purgedcv_rs.CombinatorialPurgedCV(**kw)

        def python_masks():
            splits = list(py.split(X))
            train = np.zeros((n, len(splits)), dtype=bool, order="F")
            test = np.zeros((n, len(splits)), dtype=bool, order="F")
            for c, (tr, te) in enumerate(splits):
                train[tr, c] = True
                test[te, c] = True

        t_py = best_of(python_masks, repeats=3)
        t_rs = best_of(lambda: rs.masks(X), repeats=3)
        print(f"  n={n:>9,} N,k={N},{k}: python {t_py * 1e3:8.1f}  rust {t_rs * 1e3:7.1f}  ({t_py / t_rs:.0f}x)")


def end_to_end():
    print("\nEnd-to-end cross_val_score(Ridge), n=100,000 x 20 features, N=10, k=2 (s)")
    X = frame(100_000, cols=20)
    y = pd.Series(np.random.default_rng(1).normal(size=len(X)), index=X.index)
    t1 = purgedcv.make_t1(X.index, 21)
    for name, engine in (("python", purgedcv), ("rust", purgedcv_rs)):
        cv = engine.CombinatorialPurgedCV(10, 2, embargo_pct=0.01, t1=t1)
        t = best_of(lambda: cross_val_score(Ridge(), X, y, cv=cv), repeats=3)
        print(f"  {name:>6}: {t:6.2f}")


if __name__ == "__main__":
    splits_table()
    overhead_table()
    masks_table()
    end_to_end()
