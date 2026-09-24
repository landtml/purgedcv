# Experiment: a Rust engine behind a thin Python wrapper

**Status: experiment, not part of the published package.** Nothing under
`experiments/` is imported by `purgedcv`, and the root test suite skips these
tests unless the extension is built.

The question: is it worth moving the splitting engine to Rust? This directory
is a working answer. `purgedcv_rs.CombinatorialPurgedCV` is a drop-in for
`purgedcv.CombinatorialPurgedCV` that produces identical splits.

## Build

```bash
pip install maturin
pip install -e .                      # the pure-Python package (repo root)
pip install -e experiments/rust       # builds the extension (release, LTO)
pytest experiments/rust/tests         # 304 equivalence tests
python experiments/rust/bench.py
```

## Design

| Layer | Where | What it does |
|---|---|---|
| Validation, `t1` resolution, `build_paths`, sklearn protocol, pickling | inherited from `purgedcv` | unchanged; same errors, same messages |
| `_iter_splits` / `split_all` | `python/purgedcv_rs/__init__.py` | ~100 lines: builds a `SplitPlan`, loops over combinations |
| `SplitPlan` | `src/lib.rs` | per-split purge + embargo, returned as NumPy arrays |

Choices worth noting:

- **The wrapper only replaces the hot loop.** Subclassing keeps a single source of
  truth for validation and the path map, so the correctness proof in
  `PROOF.md` still describes the inputs the kernel receives.
- **Interval arithmetic, not masks.** Every removal the Python version applies
  is an interval `[b0, label_end]` or `(anchor, anchor + embargo]`, except the
  "left purge" of training labels reaching into a test block. When `end_pos` is
  non-decreasing (always true for `make_t1` and `t1=None`) that set is a suffix
  of `[0, b0)` found by binary search, so a split costs `O(k log n)` plus
  writing its output. An arbitrary `t1` falls back to a linear scan over a byte
  mask.
- **Zero-copy output.** Index `Vec<i64>`s are handed to NumPy with
  `into_pyarray`; the dtype is `int64`, as in the Python engine.
- **GIL released** during computation (`py.detach`), so other Python threads
  keep running.
- **`split_all`** computes every split in parallel with rayon. It is eager, so
  all `C(N, k)` index arrays are held in memory together; `split` stays lazy.
- **abi3 wheel** (`abi3-py310`): one wheel per platform covers CPython 3.10+.
- **Preconditions re-checked at the boundary.** `SplitPlan::new` verifies
  `i <= end_pos[i] < n`, which is what lets the kernel skip per-element checks
  inside `[b0, label_end]`. A bad input raises `ValueError`, it can't cause UB.

## Results

Linux container, 4 cores, Python 3.11, NumPy 2.4. `fixed` is `make_t1(index, 21)`
(binary-search path); `random` has per-event horizons of 1-41 bars
(linear-scan path). Embargo 1%.

Every split materialised, ms, best of 5:

| n | N,k | t1 | splits | python | rust | rust parallel | speedup | parallel speedup |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 10,2 | fixed | 45 | 9.1 | 1.2 | 0.7 | 7.7x | 13.1x |
| 5,000 | 10,2 | random | 45 | 8.8 | 1.2 | 1.1 | 7.2x | 8.1x |
| 100,000 | 10,2 | fixed | 45 | 88.4 | 21.5 | 11.5 | 4.1x | 7.7x |
| 100,000 | 10,2 | random | 45 | 88.3 | 33.5 | 17.8 | 2.6x | 5.0x |
| 1,000,000 | 10,2 | fixed | 45 | 974.0 | 215.3 | 98.5 | 4.5x | 9.9x |
| 1,000,000 | 10,2 | random | 45 | 1022.2 | 325.1 | 152.0 | 3.1x | 6.7x |
| 100,000 | 16,4 | fixed | 1820 | 5265.6 | 764.8 | 144.1 | 6.9x | 36.5x |
| 100,000 | 16,4 | random | 1820 | 5307.8 | 1158.0 | 210.6 | 4.6x | 25.2x |

At n = 1,000,000 the Rust kernel spends about 3.6 ms per split, which is
about the cost of writing ~7 MB of fresh `int64` indices. It is limited by
memory bandwidth, not computation, so no algorithm can do much better while
the API returns index arrays.

End to end, `cross_val_score(Ridge(), X, y, cv=cv)` with n = 100,000 x 20
features, N = 10, k = 2:

| engine | seconds |
|---|---:|
| python | 0.90 |
| rust | 0.86 |

## Verdict

The engine is 3-7x faster serially and up to 36x in parallel, and it
gives identical results. A real cross-validation run barely notices: with a
model as cheap as Ridge the splitter was already only ~5% of the wall clock.
Any heavier model (gradient boosting, a neural net) makes that share smaller.

What Rust would cost the project:

- a compiled build (maturin + a Rust toolchain for source installs) and a
  wheel matrix (Linux/macOS/Windows x x86-64/arm64) in CI;
- a second implementation that `PROOF.md` and `verify_leakage.py` would have
  to cover, and that fewer Python users can read to audit;
- more places a leakage bug can hide, in a library whose selling point is that
  there is nowhere for one to hide.

**Recommendation:** keep the library pure Python. This would be worth
revisiting only if a workload spends most of its time generating splits, e.g.
very large `C(N, k)` with a near-free model. Even then, `split_all`'s parallel
speedup is the part that matters, and part of the serial gain is available in
NumPy alone: the Python engine rebuilds full-length boolean masks per split
where slices and a binary search would do.
