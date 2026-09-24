//! Experimental Rust engine for purgedcv's combinatorial purged splits.
//!
//! Computes the same `(train_idx, test_idx)` pairs as
//! `purgedcv._splitter.CombinatorialPurgedCV._iter_splits`, but as interval
//! arithmetic instead of full-length boolean masks.
//!
//! Every removal the Python version applies is an interval in positional space,
//! except one: the "left purge" (train observations starting before a test block
//! whose label reaches into it). When `end_pos` is non-decreasing -- always true
//! for `make_t1` and for `t1=None` -- that set is a suffix of `[0, b0)` found by
//! binary search, so a split costs `O(k log n)` plus writing its output. For an
//! arbitrary `t1` the left purge falls back to a linear scan over a byte mask.
//!
//! Preconditions guaranteed by `purgedcv._splitter._end_positions` (and
//! re-checked in `SplitPlan::new`): `i <= end_pos[i] < n` for every `i`.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;

/// Half-open position interval `[lo, hi)`.
type Span = (usize, usize);

/// A maximal run of adjacent test groups, `[lo, hi)`, with its label envelope:
/// the last position any of its labels reaches.
#[derive(Clone, Copy)]
struct Block {
    lo: usize,
    hi: usize,
    label_end: usize,
}

type Split = (Vec<i64>, Vec<i64>);

/// Everything a split needs that does not depend on the test combination.
///
/// Built once per `split()` call, then queried once per combination, so the
/// `end_pos` copy, the group geometry and the monotonicity check are shared by
/// all `C(N, k)` splits.
#[pyclass(frozen, module = "purgedcv_rs._engine")]
struct SplitPlan {
    end_pos: Vec<usize>,
    /// `n_groups + 1` group boundaries: group `g` is `[bounds[g], bounds[g + 1])`.
    bounds: Vec<usize>,
    embargo: usize,
    anchor_label_end: bool,
    monotone: bool,
    /// Longest label horizon, `max(end_pos[i] - i)`. Only `i >= b0 - H` can
    /// have a label reaching position `b0`, which bounds every left purge.
    max_horizon: usize,
    /// Per group, the last position any of its labels reaches.
    group_label_end: Vec<usize>,
}

impl SplitPlan {
    fn n(&self) -> usize {
        self.end_pos.len()
    }

    fn n_groups(&self) -> usize {
        self.bounds.len() - 1
    }

    /// Validate and build a plan; see the Python constructor for the contract.
    fn build(
        raw: &[i64],
        n_groups: usize,
        embargo: usize,
        anchor_label_end: bool,
    ) -> Result<Self, String> {
        let n = raw.len();
        if n_groups < 2 || n < n_groups {
            return Err(format!(
                "need 2 <= n_groups <= n_samples; got n_groups={n_groups}, n_samples={n}"
            ));
        }
        let mut end = Vec::with_capacity(n);
        for (i, &e) in raw.iter().enumerate() {
            match usize::try_from(e) {
                Ok(e) if i <= e && e < n => end.push(e),
                _ => {
                    return Err(format!(
                        "end_pos[{i}] = {e} violates i <= end_pos[i] < n_samples={n}"
                    ));
                }
            }
        }
        let block = n / n_groups;
        let mut bounds: Vec<usize> = (0..n_groups).map(|g| g * block).collect();
        bounds.push(n);
        let monotone = end.windows(2).all(|w| w[0] <= w[1]);
        let max_horizon = end.iter().enumerate().map(|(i, &e)| e - i).max().unwrap();
        // Groups are non-empty since n >= n_groups.
        let group_label_end = bounds
            .windows(2)
            .map(|w| end[w[0]..w[1]].iter().copied().max().unwrap())
            .collect();
        Ok(Self {
            end_pos: end,
            bounds,
            embargo,
            anchor_label_end,
            monotone,
            max_horizon,
            group_label_end,
        })
    }

    fn check_combo(&self, combo: &[usize]) -> Result<(), String> {
        let sorted = combo.windows(2).all(|w| w[0] < w[1]);
        if combo.is_empty() || !sorted || combo[combo.len() - 1] >= self.n_groups() {
            return Err(format!(
                "combo must be non-empty, strictly increasing and < n_groups={}; got {combo:?}",
                self.n_groups()
            ));
        }
        Ok(())
    }

    /// Merge the selected groups into maximal contiguous test blocks.
    ///
    /// A block's label envelope is the max over its groups' precomputed
    /// envelopes, so this is O(k) regardless of block length.
    fn test_blocks(&self, combo: &[usize]) -> Vec<Block> {
        let mut blocks: Vec<Block> = Vec::with_capacity(combo.len());
        for &g in combo {
            let (lo, hi) = (self.bounds[g], self.bounds[g + 1]);
            let label_end = self.group_label_end[g];
            match blocks.last_mut() {
                Some(last) if last.hi == lo => {
                    last.hi = hi;
                    last.label_end = last.label_end.max(label_end);
                }
                _ => blocks.push(Block { lo, hi, label_end }),
            }
        }
        blocks
    }

    /// Removal intervals for one block.
    ///
    /// * `[b0, label_end]` -- the test block and every observation starting
    ///   inside its label window. Each such `i` has `end_pos[i] >= i >= b0`, so
    ///   the purge condition holds without looking at `end_pos`.
    /// * Left purge -- `i < b0` with `end_pos[i] >= b0`. Only the window
    ///   `[b0 - max_horizon, b0)` can qualify. With non-decreasing `end_pos`
    ///   the qualifying set is a suffix of that window (binary search);
    ///   otherwise it is scanned and emitted as runs.
    /// * `(anchor, anchor + embargo]` -- the forward embargo.
    fn block_removals(&self, block: Block, out: &mut Vec<Span>) {
        let n = self.n();
        let Block {
            lo: b0,
            hi: b1,
            label_end,
        } = block;
        let from = b0.saturating_sub(self.max_horizon);
        let window = &self.end_pos[from..b0];
        if self.monotone {
            out.push((from + window.partition_point(|&e| e < b0), label_end + 1));
        } else {
            out.push((b0, label_end + 1));
            let mut run_start = None;
            for (i, &e) in (from..b0).zip(window) {
                match (e >= b0, run_start) {
                    (true, None) => run_start = Some(i),
                    (false, Some(s)) => {
                        out.push((s, i));
                        run_start = None;
                    }
                    _ => {}
                }
            }
            if let Some(s) = run_start {
                out.push((s, b0));
            }
        }
        if self.embargo > 0 {
            let anchor = if self.anchor_label_end {
                label_end
            } else {
                b1 - 1
            };
            let lo = anchor + 1;
            // Saturating: `embargo` comes from Python unchecked, and a wrapping
            // add in release builds would silently shrink the embargo window.
            let hi = lo.saturating_add(self.embargo).min(n);
            if lo < hi {
                out.push((lo, hi));
            }
        }
    }

    fn compute(&self, combo: &[usize]) -> Split {
        let blocks = self.test_blocks(combo);
        let test_len = blocks.iter().map(|b| b.hi - b.lo).sum();
        let mut test = Vec::with_capacity(test_len);
        for b in &blocks {
            test.extend(b.lo as i64..b.hi as i64);
        }
        let mut removals: Vec<Span> = Vec::with_capacity(3 * blocks.len());
        for &block in &blocks {
            self.block_removals(block, &mut removals);
        }
        (complement(removals, self.n()), test)
    }
}

/// Positions in `[0, n)` not covered by any of `spans`, in ascending order.
fn complement(mut spans: Vec<Span>, n: usize) -> Vec<i64> {
    spans.sort_unstable();
    // Gaps between the merged spans; sized exactly so NumPy owns no slack.
    let mut gaps: Vec<Span> = Vec::with_capacity(spans.len() + 1);
    let mut cursor = 0;
    for (lo, hi) in spans {
        if lo > cursor {
            gaps.push((cursor, lo));
        }
        cursor = cursor.max(hi);
    }
    if cursor < n {
        gaps.push((cursor, n));
    }
    let mut out = Vec::with_capacity(gaps.iter().map(|&(lo, hi)| hi - lo).sum());
    for (lo, hi) in gaps {
        out.extend(lo as i64..hi as i64);
    }
    out
}

fn to_py<'py>(py: Python<'py>, (train, test): Split) -> PyResult<Bound<'py, PyTuple>> {
    // into_pyarray hands the Vec's buffer to NumPy: no copy on the way out.
    PyTuple::new(py, [train.into_pyarray(py), test.into_pyarray(py)])
}

#[pymethods]
impl SplitPlan {
    /// Build a plan from resolved label end positions.
    ///
    /// Groups follow `purgedcv._splitter._make_group_labels`: `n // n_groups`
    /// observations each, with the remainder folded into the last group.
    #[new]
    #[pyo3(signature = (end_pos, n_groups, embargo, anchor_label_end = true))]
    fn new(
        end_pos: PyReadonlyArray1<'_, i64>,
        n_groups: usize,
        embargo: usize,
        anchor_label_end: bool,
    ) -> PyResult<Self> {
        Self::build(end_pos.as_slice()?, n_groups, embargo, anchor_label_end)
            .map_err(PyValueError::new_err)
    }

    /// Whether the O(k log n) interval path applies (non-decreasing end_pos).
    #[getter]
    fn monotone(&self) -> bool {
        self.monotone
    }

    /// `(train_idx, test_idx)` for one combination of test groups.
    fn split<'py>(&self, py: Python<'py>, combo: Vec<usize>) -> PyResult<Bound<'py, PyTuple>> {
        self.check_combo(&combo).map_err(PyValueError::new_err)?;
        let result = py.detach(|| self.compute(&combo));
        to_py(py, result)
    }

    /// All combinations at once, computed in parallel across threads.
    fn split_many<'py>(
        &self,
        py: Python<'py>,
        combos: Vec<Vec<usize>>,
    ) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        combos
            .iter()
            .try_for_each(|c| self.check_combo(c))
            .map_err(PyValueError::new_err)?;
        let results: Vec<Split> =
            py.detach(|| combos.par_iter().map(|c| self.compute(c)).collect());
        results.into_iter().map(|r| to_py(py, r)).collect()
    }
}

/// Worker threads in the pool `split_many` runs on (honours `RAYON_NUM_THREADS`).
#[pyfunction]
fn num_threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SplitPlan>()?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn complement_merges_overlaps() {
        assert_eq!(
            complement(vec![(5, 7), (1, 3), (2, 4)], 9),
            vec![0, 4, 7, 8]
        );
        assert_eq!(complement(vec![(0, 9)], 9), Vec::<i64>::new());
        assert_eq!(complement(vec![], 3), vec![0, 1, 2]);
    }

    #[test]
    fn huge_embargo_saturates_instead_of_wrapping() {
        let end: Vec<i64> = (0..10).collect();
        let plan = SplitPlan::build(&end, 2, usize::MAX, true).unwrap();
        let (train, test) = plan.compute(&[0]);
        assert!(train.is_empty());
        assert_eq!(test, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn build_rejects_broken_preconditions() {
        assert!(SplitPlan::build(&[0, 1, 2], 4, 0, true).is_err());
        assert!(SplitPlan::build(&[0, 1, 2], 1, 0, true).is_err());
        assert!(SplitPlan::build(&[1, 0, 2], 2, 0, true).is_err());
        assert!(SplitPlan::build(&[0, 1, 3], 2, 0, true).is_err());
        assert!(SplitPlan::build(&[-1, 1, 2], 2, 0, true).is_err());
        let plan = SplitPlan::build(&[0, 1, 2, 3], 2, 0, true).unwrap();
        assert!(plan.check_combo(&[]).is_err());
        assert!(plan.check_combo(&[1, 0]).is_err());
        assert!(plan.check_combo(&[0, 0]).is_err());
        assert!(plan.check_combo(&[2]).is_err());
        assert!(plan.check_combo(&[0, 1]).is_ok());
    }

    /// Brute-force reference, written from the leakage definition rather than
    /// from the kernel: an observation is test if its group is selected;
    /// otherwise it is dropped if its label interval `[i, end[i]]` intersects
    /// any test observation's label interval (purge, checked pair by pair --
    /// no envelope shortcut), or if it starts inside a block's forward embargo
    /// `(anchor, anchor + embargo]`.
    fn oracle(
        end: &[usize],
        n_groups: usize,
        combo: &[usize],
        embargo: usize,
        anchor_label_end: bool,
    ) -> Split {
        let n = end.len();
        let block = n / n_groups;
        let group = |i: usize| (i / block).min(n_groups - 1);
        let is_test: Vec<bool> = (0..n).map(|i| combo.contains(&group(i))).collect();

        // Maximal runs of test positions, inclusive.
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for i in (0..n).filter(|&i| is_test[i]) {
            match runs.last_mut() {
                Some(r) if r.1 + 1 == i => r.1 = i,
                _ => runs.push((i, i)),
            }
        }

        let mut train = Vec::new();
        for i in (0..n).filter(|&i| !is_test[i]) {
            let purged = (0..n)
                .filter(|&j| is_test[j])
                .any(|j| i <= end[j] && j <= end[i]);
            let embargoed = runs.iter().any(|&(b0, b1)| {
                let label_end = (b0..=b1).map(|j| end[j]).max().unwrap();
                let anchor = if anchor_label_end { label_end } else { b1 };
                i > anchor && i - anchor <= embargo
            });
            if !purged && !embargoed {
                train.push(i as i64);
            }
        }
        let test = (0..n).filter(|&i| is_test[i]).map(|i| i as i64).collect();
        (train, test)
    }

    /// A random problem: end positions (monotone or arbitrary), a group
    /// count, one valid combination of test groups, embargo and anchor.
    fn problem() -> impl Strategy<Value = (Vec<usize>, usize, Vec<usize>, usize, bool)> {
        (2usize..8, 0usize..60, any::<bool>())
            .prop_flat_map(|(n_groups, extra, monotone)| {
                let n = n_groups + extra;
                let horizons = prop::collection::vec(0usize..=n, n);
                let combo =
                    prop::sample::subsequence((0..n_groups).collect::<Vec<_>>(), 1..n_groups);
                let embargo = prop_oneof![Just(0usize), 0usize..=n + 2, Just(usize::MAX)];
                (
                    Just(n),
                    Just(n_groups),
                    horizons,
                    Just(monotone),
                    combo,
                    embargo,
                    any::<bool>(),
                )
            })
            .prop_map(
                |(n, n_groups, horizons, monotone, combo, embargo, anchor)| {
                    let mut end: Vec<usize> =
                        (0..n).map(|i| (i + horizons[i]).min(n - 1)).collect();
                    if monotone {
                        for i in 1..n {
                            end[i] = end[i].max(end[i - 1]);
                        }
                    }
                    (end, n_groups, combo, embargo, anchor)
                },
            )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4000))]

        #[test]
        fn kernel_matches_bruteforce_oracle(
            (end, n_groups, combo, embargo, anchor) in problem()
        ) {
            let raw: Vec<i64> = end.iter().map(|&e| e as i64).collect();
            let plan = SplitPlan::build(&raw, n_groups, embargo, anchor).unwrap();
            prop_assert_eq!(
                plan.compute(&combo),
                oracle(&end, n_groups, &combo, embargo, anchor)
            );
        }
    }
}
