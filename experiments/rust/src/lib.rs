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
        Ok(Self {
            end_pos: end,
            bounds,
            embargo,
            anchor_label_end,
            monotone,
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
    fn test_blocks(&self, combo: &[usize]) -> Vec<Span> {
        let mut blocks: Vec<Span> = Vec::with_capacity(combo.len());
        for &g in combo {
            let (lo, hi) = (self.bounds[g], self.bounds[g + 1]);
            match blocks.last_mut() {
                Some(last) if last.1 == lo => last.1 = hi,
                _ => blocks.push((lo, hi)),
            }
        }
        blocks
    }

    /// Label envelope of a block: the last position any of its labels reaches.
    fn label_end(&self, (b0, b1): Span) -> usize {
        if self.monotone {
            self.end_pos[b1 - 1]
        } else {
            // Non-empty: every group holds >= 1 observation since n >= n_groups.
            self.end_pos[b0..b1].iter().copied().max().unwrap()
        }
    }

    /// Removal intervals for one block, excluding the non-monotone left purge.
    ///
    /// * `[b0, label_end]` -- the test block and every observation starting
    ///   inside its label window. Each such `i` has `end_pos[i] >= i >= b0`, so
    ///   the purge condition holds without looking at `end_pos`.
    /// * `(anchor, anchor + embargo]` -- the forward embargo.
    fn block_removals(&self, block: Span, out: &mut Vec<Span>) -> usize {
        let n = self.n();
        let (b0, b1) = block;
        let label_end = self.label_end(block);
        out.push((b0, label_end + 1));
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
        label_end
    }

    fn compute(&self, combo: &[usize]) -> Split {
        let n = self.n();
        let blocks = self.test_blocks(combo);

        let test: Vec<i64> = blocks
            .iter()
            .flat_map(|&(lo, hi)| lo as i64..hi as i64)
            .collect();

        let mut removals: Vec<Span> = Vec::with_capacity(3 * blocks.len());
        for &block in &blocks {
            self.block_removals(block, &mut removals);
        }

        if self.monotone {
            // Left purge: positions i < b0 with end_pos[i] >= b0, a suffix of
            // [0, b0) because end_pos is non-decreasing.
            for &(b0, _) in &blocks {
                let first = self.end_pos[..b0].partition_point(|&e| e < b0);
                if first < b0 {
                    removals.push((first, b0));
                }
            }
            (complement(removals, n), test)
        } else {
            let mut removed = vec![false; n];
            for &(lo, hi) in &removals {
                removed[lo..hi].fill(true);
            }
            for &(b0, _) in &blocks {
                for (flag, &e) in removed[..b0].iter_mut().zip(&self.end_pos[..b0]) {
                    *flag |= e >= b0;
                }
            }
            let train = removed
                .iter()
                .enumerate()
                .filter(|&(_, &r)| !r)
                .map(|(i, _)| i as i64)
                .collect();
            (train, test)
        }
    }
}

/// Positions in `[0, n)` not covered by any of `spans`, in ascending order.
fn complement(mut spans: Vec<Span>, n: usize) -> Vec<i64> {
    spans.sort_unstable();
    let mut out = Vec::with_capacity(n);
    let mut cursor = 0;
    for (lo, hi) in spans {
        if lo > cursor {
            out.extend(cursor as i64..lo as i64);
        }
        cursor = cursor.max(hi);
    }
    if cursor < n {
        out.extend(cursor as i64..n as i64);
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

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SplitPlan>()?;
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
