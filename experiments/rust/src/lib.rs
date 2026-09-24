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

    fn check_combo(&self, combo: &[usize]) -> PyResult<()> {
        let sorted = combo.windows(2).all(|w| w[0] < w[1]);
        if combo.is_empty() || !sorted || combo[combo.len() - 1] >= self.n_groups() {
            return Err(PyValueError::new_err(format!(
                "combo must be non-empty, strictly increasing and < n_groups={}; got {combo:?}",
                self.n_groups()
            )));
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
            let hi = (lo + self.embargo).min(n);
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
        let raw = end_pos.as_slice()?;
        let n = raw.len();
        if n_groups < 2 || n < n_groups {
            return Err(PyValueError::new_err(format!(
                "need 2 <= n_groups <= n_samples; got n_groups={n_groups}, n_samples={n}"
            )));
        }
        let mut end = Vec::with_capacity(n);
        for (i, &e) in raw.iter().enumerate() {
            match usize::try_from(e) {
                Ok(e) if i <= e && e < n => end.push(e),
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "end_pos[{i}] = {e} violates i <= end_pos[i] < n_samples={n}"
                    )));
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

    /// Whether the O(k log n) interval path applies (non-decreasing end_pos).
    #[getter]
    fn monotone(&self) -> bool {
        self.monotone
    }

    /// `(train_idx, test_idx)` for one combination of test groups.
    fn split<'py>(&self, py: Python<'py>, combo: Vec<usize>) -> PyResult<Bound<'py, PyTuple>> {
        self.check_combo(&combo)?;
        let result = py.detach(|| self.compute(&combo));
        to_py(py, result)
    }

    /// All combinations at once, computed in parallel across threads.
    fn split_many<'py>(
        &self,
        py: Python<'py>,
        combos: Vec<Vec<usize>>,
    ) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        combos.iter().try_for_each(|c| self.check_combo(c))?;
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

    #[test]
    fn complement_merges_overlaps() {
        assert_eq!(
            complement(vec![(5, 7), (1, 3), (2, 4)], 9),
            vec![0, 4, 7, 8]
        );
        assert_eq!(complement(vec![(0, 9)], 9), Vec::<i64>::new());
        assert_eq!(complement(vec![], 3), vec![0, 1, 2]);
    }
}
