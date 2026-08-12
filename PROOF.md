# purgedcv: Proof of No Data Leakage and Verification Report

This document gives (1) a mathematical proof that the splitter cannot leak
information from the test set into the training set, and (2) an empirical
certificate from a large-scale run over real market data. It also states the
confidence level and the scope/limits of the guarantee.

Component under test: [`src/purgedcv/_splitter.py`](src/purgedcv/_splitter.py)
· `CombinatorialPurgedCV` (AFML, López de Prado, ch. 7).

---

## 1. Definitions

Let the sample have `n` observations at positions `0 … n-1`. Each observation
`i` carries an **information interval** `I_i = [s_i, e_i]` in position space, where
`s_i = i` (the bar at which the feature is known) and `e_i = end_pos(t1_i) ≥ i`
(the position at which its label/event resolves). This is exactly de Prado's
`t1` "event lifespan".

> **Leakage** is defined as a *kept training* observation `i` whose interval
> overlaps the interval of *any test* observation `j`:
> `I_i ∩ I_j ≠ ∅  ⟺  (e_i ≥ s_j) ∧ (s_i ≤ e_j)`.

A correct purge + embargo must guarantee the kept training set has **zero**
leakage against the test set, for every split.

The timeline is partitioned into `N` contiguous groups; a test set is the union
of `k` chosen groups, i.e. a union of contiguous **blocks** of positions.

---

## 2. Lemma (test-block label-union is gapless)

*For one contiguous test block occupying positions `b0 … b1`, the union of its
observations' intervals is the single contiguous interval `[b0, m]`, where
`m = max_{b0 ≤ j ≤ b1} e_j`.*

**Proof.** Each test obs contributes `[j, e_j]` with `e_j ≥ j`.
- Every integer `p ∈ [b0, b1]` is a test start, so `p ∈ [p, e_p] ⊆ union`.
- Every integer `p ∈ (b1, m]`: let `j★ = argmax e_j`, so `e_{j★} = m ≥ p` and
  `j★ ≤ b1 < p`, hence `p ∈ [j★, e_{j★}] ⊆ union`.
Thus `[b0, m] ⊆ union`. Conversely every `[j, e_j] ⊆ [b0, m]`, so the union is
exactly `[b0, m]`, with no gaps. ∎

---

## 3. Theorem 1 (envelope purge ≡ per-observation purge)

*A training interval `[s_i, e_i]` overlaps the test block iff it overlaps the
envelope `[b0, m]`, i.e. iff `(e_i ≥ b0) ∧ (s_i ≤ m)`.*

**Proof.** Overlap with the block means overlap with the union of its members'
intervals. By the Lemma that union is `[b0, m]`. Two integer intervals
`[s_i, e_i]` and `[b0, m]` overlap iff `e_i ≥ b0` and `s_i ≤ m`. ∎

This is the exact condition implemented in `_train_mask` (purge clause
`(end_pos >= b0) & (start_pos <= label_end)`). It covers all three AFML 7.1
cases in one expression: train *starts within*, *ends within*, or *fully
envelops* the test window. The implementation therefore matches the
per-observation definition with **no approximation**; Theorem 1 is what licenses
using the cheap envelope test instead of an `O(n_train·n_test)` scan.

---

## 4. Theorem 2 (the kept training set is leakage-free)

*Let a training observation be **kept** by `split()` for a given combination.
Then it does not overlap any test observation in any selected block.*

**Proof.** `split()` removes observation `i` from training if, for some test
block `[b0, b1]` with envelope end `m`, the purge clause
`(e_i ≥ b0) ∧ (s_i ≤ m)` holds (plus, separately, the embargo clause and the
explicit removal of the test indices themselves). A kept training obs therefore
falsifies the purge clause for **every** block: for each block
`(e_i < b0) ∨ (s_i > m)`.
- If `e_i < b0`: the train interval ends strictly before the block's union
  begins ⇒ disjoint from every member interval `[j, e_j] ⊆ [b0, m]`.
- If `s_i > m`: the train interval starts strictly after the union ends ⇒
  disjoint from every member interval.
Either way `I_i ∩ I_j = ∅` for all test `j` in that block, for all blocks. Hence
no kept training observation overlaps any test observation. ∎

**Embargo (forward buffer).** When `embargo = ⌈embargo_pct·n⌉ > 0`, an extra band
of `embargo` positions immediately after each block's anchor
(`label_end` by default, or the last test index `b1` with
`embargo_anchor="test_end"`) is also removed. This only *enlarges* the removed
set, so Theorem 2 still holds; the embargo additionally drops serially-correlated
neighbours that survive pure purging. `embargo = 0` reduces exactly to the purge.

---

## 5. Coverage / path invariant

Each group serves as a test fold in exactly `C(N-1, k-1)` simulations, which
equals `n_paths = C(N, k)·k / N`. The path stitcher consumes one such simulation
per group per path, so every observation is predicted out-of-sample in exactly
`n_paths` simulations and every backtest path tiles the timeline once.

The stitch is total by construction: it clears the consumed simulation from one
group's row only, so rows never interact, and each row starts with exactly
`n_paths` candidates and hands out one per path. `build_paths` checks that
regular-degree precondition before stitching -- that is the assumption the
argument rests on, and it is what would break if the combination geometry ever
changed. (Through v0.1.0 the check sat *inside* the stitch loop, where the
row-independence argument above made it unreachable.)

Distinctness is what separates this from the naive implementation that copies
one simulation's predictions across every path: `test_path_folds_are_non_degenerate`
asserts every path column is distinct and every simulation is used exactly `k`
times, which fails immediately under that shortcut.

---

## 6. Empirical certificate

Two **independent** leakage oracles were run and agreed everywhere:

* **cert**: kept set vs each test block's label-union range `[b0, m]` (Thm 1/2).
* **brute**: kept set vs **every individual** test observation interval
  (the raw definition in §1), with no reference to the envelope shortcut.
* **equiv**: library purge output compared element-for-element to a from-scratch
  per-observation brute-force purge (embargo-free).

Reproduce with `python verify_leakage.py`. Result of the run on this machine:

Configuration grid behind the numbers below: `(N,k)` in `(6,2) (8,2) (10,3)
(8,3)`; horizons `1, 5, 21, 63` bars; `embargo_pct` in `0.0, 0.02`;
`embargo_anchor` in `label_end, test_end`. Tickers: `^GSPC ^DJI IBM KO GE XOM
JNJ PG MSFT AAPL`, each verified over its full history and a recent window.

```
======================================================================
PURGEDCV LEAKAGE VERIFICATION CERTIFICATE
======================================================================
  datasets verified      : 20 tickers
  calendar span          : 1927-12-30 -> 2026-05-19
  total bars (obs)        : 191,160
  scenarios (cfg x h x e) : 1,280
  train/test splits       : 70,080
  envelope assertions     : 977,027,536
  brute-force pair checks : 107,693,946,180
  purge-equivalence checks: 8,760
----------------------------------------------------------------------
  RESULT: PASS
  D disjoint | L no-leakage (cert+brute) | E embargo | Q purge==bruteforce | C coverage
  Zero leakage detected across every split, scenario, and dataset.
======================================================================
```

This is the exact output `python verify_leakage.py` prints (see
[verify_leakage.py](verify_leakage.py)), not a paraphrase. The run above was
produced on v0.1.0; v0.2.0 changed only validation and error paths, leaving the
purge and embargo arithmetic the theorems describe untouched.

Unit suite: `pytest` → **115 tests** (incl. the live-yfinance integration test,
which skips offline), covering counts, no-leakage, embargo, disjointness, the
canonical (6,2) path-stitch regression, coverage, determinism, the
`envelope == brute-force` equivalence, all input-validation paths, and -- in
`tests/test_sklearn.py` -- the scikit-learn integration against a real install.

---

## 7. Confidence & scope

**Confidence: very high for bar data.** The no-leakage property is *proved*
(§2–§4), not merely tested; the proof's one shortcut (envelope vs per-observation)
is itself a theorem and was independently re-checked ~1.08×10¹¹ times against the
raw definition on ~98 years of real data across 20 datasets and 70,080 splits,
with zero violations. The proof is constructive and dtype/calendar-agnostic
(positions are `int64`; irregular/holiday gaps resolve through `searchsorted`).

**The guarantee is conditional on the splitter's contract**, and the code
enforces that contract instead of assuming it holds:
- the index must be **unique and monotonically increasing** (else `ValueError`);
- `t1` must cover every observation (mis-alignment raises `ValueError` instead
  of silently spanning to the end of the sample);
- `t1` must have been built for **the sample being split**. Purging works in
  positional space, so a `t1` built on a denser grid silently rescales every
  label horizon and under-purges; since v0.2.0 that raises `ValueError` rather
  than proceeding. Contiguous slices and longer histories resolve identically
  to a rebuilt `t1` and remain permitted;
- `X.index` and `t1` must agree on timezone-awareness (else `ValueError`);
- leakage is defined w.r.t. the **`t1` you supply**: an understated `t1`
  (label lifespan shorter than reality) is a *modelling* error the splitter
  cannot detect.

**Degenerate folds are rejected, not emitted.** Purge and embargo can consume an
entire training set; such a fold is not a simulation, and downstream it becomes
a `NaN` score that averaging can hide. Since v0.2.0 `split` raises when a fold
falls below `min_train_size` (default 1); pass `min_train_size=0` to opt out.

**Out of scope, deliberately:** return/PnL accounting, feature
scaling/leakage inside the user's model pipeline, and metric annualisation. The
component yields indices + a pure path map only.
