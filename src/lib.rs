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
    /// fitness: 1D numpy array (N,)
    /// descriptors: 2D numpy array (N, D)
    /// Returns: 1D numpy array of INDICES to keep
    fn select<'py>(
        &self,
        py: Python<'py>,
        fitness: PyReadonlyArray1<f64>,
        descriptors: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<usize>>> {
        
        // 1. Convert to Rust Views (Zero Copy)
        let fit_view = fitness.as_array();
        let desc_view = descriptors.as_array();

        // 2. Safety Checks
        let n_samples = fit_view.len();
        if desc_view.shape()[0] != n_samples {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Mismatch: Fitness and Descriptor arrays must have same length",
            ));
        }

        // 3. Execute Pure Rust Logic
        let selected_indices = self.run_lazy_greedy(fit_view, desc_view);

        // 4. Return as Numpy Array
        // PyArray1::from_vec returns a Bound<'py, ...> in newer numpy/pyo3 versions
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
            
            // Initial distance is strictly to the seed
            let d = (&current_desc - &seed_desc)
                .mapv(|x| x.powi(2))
                .sum()
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
                
                // LAZY CHECK
                let cand_desc = descriptors.row(cand_idx);
                let mut current_d_min = f64::INFINITY;

                for &archived_idx in &archive_indices {
                    let archive_desc = descriptors.row(archived_idx);
                    let d = (&cand_desc - &archive_desc)
                        .mapv(|x| x.powi(2))
                        .sum()
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
                    // Rejected
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

/// The module definition
#[pymodule]
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
        assert_eq!(indices[0], 0); // First is always best fitness
        assert_eq!(indices[1], 2); // Second should be the far one
    }

    #[test]
    fn test_n_less_than_k() {
        let selector = MMRSelector::new(10, 0.5);
        let fitness = arr1(&[1.0, 2.0, 3.0]);
        let descriptors = arr2(&[
            [0.0], [1.0], [2.0]
        ]);

        let indices = selector.run_lazy_greedy(fitness.view(), descriptors.view());
        assert_eq!(indices.len(), 3); // Should return all
    }

    #[test]
    fn test_identical_descriptors() {
        let selector = MMRSelector::new(3, 0.5);
        let fitness = arr1(&[10.0, 8.0, 9.0, 5.0]);
        let descriptors = arr2(&[
            [0.0], [0.0], [0.0], [0.0]
        ]);

        let indices = selector.run_lazy_greedy(fitness.view(), descriptors.view());
        
        // Should pick best fitnesses since diversity is 0 for all
        // Order: 10.0 -> 9.0 -> 8.0 -> ...
        // Indices: 0, 2, 1
        assert_eq!(indices, vec![0, 2, 1]);
    }
}
