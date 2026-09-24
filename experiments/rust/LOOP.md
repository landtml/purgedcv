# Self-iterating improvement loop: Rust engine

This file is the charter for an autonomous improvement loop on
`experiments/rust/`. Each iteration reads this file, does one backlog item end
to end, and records the outcome in the log below.

## Status

**Paused after iteration 5** at the user's direction, to build the
backtest statistics layer (`backtest/`) first. Resume with `/loop` and the
same prompt; the unchecked items below are still valid.

## Goal

Make `purgedcv_rs` the best Rust engine it can be, with parallelism as the
headline feature, behind a thin Python wrapper that keeps `purgedcv`'s API
conventions: sklearn `BaseCrossValidator` protocol, same constructor
parameters, same `int64` index arrays, same errors and messages, picklable,
lazy `split()`.

## Hard rules (never broken)

1. **Correctness first.** The equivalence suite against the pure-Python engine
   must pass on every commit. A speedup that changes a single index is a bug.
   New kernels get a brute-force oracle test, not only an equivalence test.
2. **Scope.** Only touch `experiments/rust/` and, if needed, a workflow file
   that only runs for `experiments/rust/**`. Never modify `src/purgedcv/`,
   `tests/`, `PROOF.md`, `verify_leakage.py` or the main CI job.
3. **Gate before commit:** `cargo fmt --check`, `cargo clippy --all-targets
   -- -D warnings`, `cargo test`, rebuild with `pip install -e experiments/rust`,
   `pytest experiments/rust/tests`, and the root `pytest`. All green, or the
   change is not committed.
4. **Performance changes need numbers.** Run `bench.py` before and after. A
   change that regresses any row by more than 10% is reverted unless it is a
   correctness fix.
5. **No `unsafe`** unless a benchmark proves a >20% win and a comment states
   the invariant; prefer safe code.
6. **One item per iteration, one commit per item**, pushed to
   `claude/busy-dirac-qpjyzb`. No pull request unless the user asks.
7. **Python wrapper stays thin.** Logic belongs in Rust; the wrapper only
   adapts types and preserves API conventions.

## Stop condition

Stop the loop when the backlog has no item with expected value left, or after
three consecutive iterations that produce no committed improvement. Then
update `README.md` (results + verdict) and report to the user.

## Backlog (highest expected value first; iterations may add items)

- [x] **Bug hunt:** property-based Rust tests (`proptest`) against a naive
      O(n^2) brute-force oracle of the purge/embargo definition; edge cases
      (n == n_groups, k == N-1, embargo >= n, horizon 0, numpy `X`).
- [x] **Efficiency, non-monotone path:** only `i` in `[b0 - H, b0)` can have
      `end_pos[i] >= b0`, where `H = max(end_pos[i] - i)`. Scan that window
      instead of `[0, b0)` and emit intervals, dropping the byte mask.
- [x] **Parallelism, lazy:** a bounded-prefetch parallel iterator so
      `split()` stays lazy and memory-bounded while computing ahead on a
      rayon pool (`split(X, prefetch=...)`-free: keep signature, add a
      constructor-free module setting or `split_iter(n_ahead)`).
- [~] **Parallelism, intra-split:** for very large n, write one split's output
      with parallel chunked fills. *Skipped (iter 4): iteration 3 already
      parallelises across splits, and output writing is memory-bandwidth bound;
      it would only help when C(N, k) < cores at very large n.*
- [x] **Feature:** `masks(X, t1)` (shipped as one call returning train + test) returning
      `(n_samples, n_sims)` bool matrices, built in parallel (for JAX `vmap`
      and weighted-fit users).
- [x] **Feature:** `build_paths` in Rust (parallel), verified equal to Python.
- [ ] **Feature:** a purged K-fold (`PurgedKFold`, AFML 7.3) sharing the kernel.
- [ ] **Extension:** thread-count control (`n_threads` / `RAYON_NUM_THREADS`
      respected), `.pyi` stubs + `py.typed`.
- [ ] **CI:** workflow building the extension with maturin and running the
      equivalence suite, path-filtered to `experiments/rust/**`.
- [ ] **Docs:** keep `README.md` results and verdict current.

## Iteration log

| # | Item | Outcome | Commit |
|---|------|---------|--------|
| 0 | Initial engine, wrapper, equivalence suite, benchmark | 3-7x serial, up to 36x parallel; ~4% end to end | see git log |
| 1 | Bug hunt: proptest vs brute-force pairwise oracle, edge shapes | **Fixed a real bug:** `SplitPlan` embargo add wrapped in release builds (`embargo=2**64-1` gave *no* embargo, i.e. leakage); now saturating. Oracle mutation-tested (3/3 planted bugs caught). Construction/validation moved to pure Rust so it's testable without Python. | see git log |
| 2 | Efficiency: windowed left purge, O(k) label envelopes, exact-size output | Serial rust (ms, before -> after): 1M random 319 -> 211, 100k N16k4 random 1204 -> 606, fixed 772 -> 615, 5k 1.2 -> 0.5. Parallel N16k4 207 -> 83 (now 73x over Python). No row regressed. Byte mask removed; one code path (interval complement) for both t1 shapes. | see git log |
| 3 | Parallelism, lazy: `split()` computes batches of 2x threads via `split_many`, serial below 20k samples (measured crossover) | Lazy `split()` (ms, iter 2 -> 3): 1M fixed 198 -> 79, 1M random 211 -> 104, 100k N16k4 615 -> 140 / 606 -> 120 (now ~40x over Python, still lazy, memory bounded to one batch). 5k within noise (0.5 -> 0.6). First try regressed small n 2.4x via pool overhead + batching loop; fixed with the cutoff and a plain serial loop. Tests: prefix + error parity when a fold goes degenerate mid-stream, both sides of the cutoff. | see git log |
| 4 | Feature: `masks(X)` -> `(train, test)` bool `(n_samples, n_sims)` F-ordered, columns filled in parallel, zero-copy reshape | 15x (100k, 1M) to 44x (N16k4) over building the same masks from Python splits. Test masks equal Python's `build_paths().is_test`; oracle proptest covers the mask painter. `_resolve` shared by `split_all`/`masks`; `groups` warning stacklevel verified to point at the caller. Intra-split parallelism skipped with rationale. No split-row regression. | see git log |
| 5 | Feature: `build_paths` in Rust: per-group rows broadcast in parallel into C-ordered `is_test` / `paths` | 28x (100k), 16x (1M), 23x (N16k4) over Python. All fields equal in value, dtype, shape and C-contiguity across 18 geometries; `combine` / `to_frame` identical; same errors. A 1M split row read +21% once; 15-run re-measure (min 70.4 ms vs 79 before) showed noise. | see git log |
