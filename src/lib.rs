use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray1, IntoPyArray, ndarray::{ArrayView1, ArrayView2}};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

// --- Float Wrapper for Heap ---
#[derive(Debug, Clone, Copy, PartialEq)]
struct Float(f64);

impl Eq for Float {}

impl PartialOrd for Float {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for Float {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// --- Candidate Struct for Heap ---
#[derive(PartialEq, Eq)]
struct Candidate {
    index: usize,
    mmr_score: Float,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.mmr_score.cmp(&other.mmr_score)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// --- The Python Class ---
#[pyclass]
struct MMRSelector {
    target_k: usize,
    lambda_val: f64,
}

#[pymethods]
impl MMRSelector {
    #[new]
    fn new(target_k: usize, lambda_val: f64) -> Self {
        MMRSelector { target_k, lambda_val }
    }

    /// The Zero-Copy Selection Interface
    fn select<'py>(
        &self,
        py: Python<'py>,
        fitness: PyReadonlyArray1<f64>,
        descriptors: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<usize>>> {

        let fit_view = fitness.as_array();
        let desc_view = descriptors.as_array();

        let n_samples = fit_view.len();
        if desc_view.shape()[0] != n_samples {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Mismatch: Fitness and Descriptor arrays must have same length",
            ));
        }

        let selected_indices = self.run_lazy_greedy(fit_view, desc_view);

        Ok(PyArray1::from_vec(py, selected_indices))
    }
}

impl MMRSelector {
    /// Internal Pure Rust implementation working on ndarray Views
    fn run_lazy_greedy(
        &self,
        fitness: ArrayView1<f64>,
        descriptors: ArrayView2<f64>
    ) -> Vec<usize> {
        let n = fitness.len();
        if n <= self.target_k {
            return (0..n).collect();
        }

        let mut selected_indices = Vec::with_capacity(self.target_k);
        let mut archive_indices: Vec<usize> = Vec::with_capacity(self.target_k);

        // 1. Seed with Best Fitness
        let mut best_idx = 0;
        let mut max_fit = f64::NEG_INFINITY;

        for (i, &val) in fitness.iter().enumerate() {
            if val > max_fit {
                max_fit = val;
                best_idx = i;
            }
        }

        selected_indices.push(best_idx);
        archive_indices.push(best_idx);

        // 2. Initialize Priority Queue
        let mut pq = BinaryHeap::new();
        let seed_desc = descriptors.row(best_idx);

        for i in 0..n {
            if i == best_idx { continue; }

            let current_desc = descriptors.row(i);

            // Optimization: Iterator-based distance (No Allocation)
            let d = current_desc.iter()
                .zip(seed_desc.iter())
                .fold(0.0, |acc, (a, b)| acc + (a - b).powi(2))
                .sqrt();

            let score = (1.0 - self.lambda_val) * fitness[i] + self.lambda_val * d;

            pq.push(Candidate {
                index: i,
                mmr_score: Float(score),
            });
        }

        // 3. Lazy Greedy Loop
        while selected_indices.len() < self.target_k {
            if let Some(top) = pq.pop() {
                let cand_idx = top.index;
                let cand_desc = descriptors.row(cand_idx);

                // LAZY CHECK
                let mut current_d_min = f64::INFINITY;

                for &archived_idx in &archive_indices {
                    let archive_desc = descriptors.row(archived_idx);

                    // Optimization: Iterator-based distance (No Allocation)
                    let d = cand_desc.iter()
                        .zip(archive_desc.iter())
                        .fold(0.0, |acc, (a, b)| acc + (a - b).powi(2))
                        .sqrt();

                    if d < current_d_min {
                        current_d_min = d;
                    }
                }

                let new_score = (1.0 - self.lambda_val) * fitness[cand_idx] + self.lambda_val * current_d_min;
                
                // Peek next best
                let threshold = pq.peek().map(|c| c.mmr_score.0).unwrap_or(f64::NEG_INFINITY);

                if new_score >= threshold {
                    // Accepted
                    selected_indices.push(cand_idx);
                    archive_indices.push(cand_idx);
                } else {
                    // Rejected - push back with updated score
                    pq.push(Candidate {
                        index: cand_idx,
                        mmr_score: Float(new_score),
                    });
                }
            } else {
                break;
            }
        }

        selected_indices
    }
}

#[pymodule]
#[pyo3(name = "mmr_elites_rs")]
fn mmr_elites_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MMRSelector>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{arr1, arr2};

    #[test]
    fn test_small_basic_case() {
        let selector = MMRSelector::new(2, 0.5);
        let fitness = arr1(&[1.0, 0.5, 0.5]);
        let descriptors = arr2(&[
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 0.0]
        ]);

        let indices = selector.run_lazy_greedy(fitness.view(), descriptors.view());
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0], 0); // Best fitness
        assert_eq!(indices[1], 2); // Most diverse
    }

    #[test]
    fn test_n_less_than_k() {
        let selector = MMRSelector::new(10, 0.5);
        let fitness = arr1(&[1.0, 2.0]);
        let descriptors = arr2(&[[0.0], [1.0]]);
        let indices = selector.run_lazy_greedy(fitness.view(), descriptors.view());
        assert_eq!(indices.len(), 2);
    }

    #[test]
    fn test_identical_descriptors() {
        let selector = MMRSelector::new(3, 0.5);
        let fitness = arr1(&[10.0, 8.0, 9.0, 5.0]);
        // All identical - logic should fall back to pure fitness ranking
        let descriptors = arr2(&[
            [0.0], [0.0], [0.0], [0.0]
        ]);
        let indices = selector.run_lazy_greedy(fitness.view(), descriptors.view());
        // Should select 10.0, 9.0, 8.0 -> indices 0, 2, 1
        assert_eq!(indices, vec![0, 2, 1]);
    }
}
