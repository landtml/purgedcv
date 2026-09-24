# Self-iterating improvement loop: Rust engine

This file is the charter for an autonomous improvement loop on
`experiments/rust/`. Each iteration reads this file, does one backlog item end
to end, and records the outcome in the log below.

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
- [ ] **Efficiency, non-monotone path:** only `i` in `[b0 - H, b0)` can have
      `end_pos[i] >= b0`, where `H = max(end_pos[i] - i)`. Scan that window
      instead of `[0, b0)` and emit intervals, dropping the byte mask.
- [ ] **Parallelism, lazy:** a bounded-prefetch parallel iterator so
      `split()` stays lazy and memory-bounded while computing ahead on a
      rayon pool (`split(X, prefetch=...)`-free: keep signature, add a
      constructor-free module setting or `split_iter(n_ahead)`).
- [ ] **Parallelism, intra-split:** for very large n, write one split's output
      with parallel chunked fills.
- [ ] **Feature:** `train_masks(X, t1)` / `test_masks` returning
      `(n_samples, n_sims)` bool matrices, built in parallel (for JAX `vmap`
      and weighted-fit users).
- [ ] **Feature:** `build_paths` in Rust (parallel), verified equal to Python.
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
