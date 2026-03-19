use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray1, ndarray::{ArrayView1, ArrayView2}};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use rayon::prelude::*;

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

// --- Optimized Candidate Struct ---
// Removed #[derive(PartialEq, Eq)] because f64 (cached_d_min) breaks it.
struct Candidate {
    index: usize,
    mmr_score: Float,
    // CACHE: The d_min valid against the archive of size 'checked_count'
    cached_d_min: f64,     
    // TIMESTAMP: How many elites were in the archive when we last updated this?
    checked_count: usize,  
}

// Manually implement PartialEq to satisfy Eq, ignoring the raw f64 fields
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.mmr_score == other.mmr_score
    }
}

impl Eq for Candidate {}

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

    /// The High-Performance Selection Interface
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

        // Release GIL for expensive computation (Parallelism!)
        let indices = py.allow_threads(|| {
            self.run_optimized(fit_view, desc_view)
        });

        Ok(PyArray1::from_vec(py, indices))
    }
}

impl MMRSelector {
    fn run_optimized(
        &self,
        fitness: ArrayView1<f64>,
        descriptors: ArrayView2<f64>
    ) -> Vec<usize> {
        let n = fitness.len();
        if n <= self.target_k {
            return (0..n).collect();
        }

        // Normalize fitness to [0, 1] so λ trades off comparable scales
        let mut f_min = f64::INFINITY;
        let mut f_max = f64::NEG_INFINITY;
        for &val in fitness.iter() {
            if val < f_min { f_min = val; }
            if val > f_max { f_max = val; }
        }
        let f_range = f_max - f_min;
        let f_norm: Vec<f64> = if f_range > 1e-10 {
            fitness.iter().map(|&v| (v - f_min) / f_range).collect()
        } else {
            vec![0.5; n]
        };

        let mut selected_indices = Vec::with_capacity(self.target_k);
        let mut archive_indices: Vec<usize> = Vec::with_capacity(self.target_k);

        // 1. Seed with Best Fitness (Serial is fine for O(N))
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

        // 2. Parallel Initialization (Rayon)
        let seed_desc = descriptors.row(best_idx);
        let lambda = self.lambda_val;

        // Build candidates in parallel
        let candidates_vec: Vec<Candidate> = (0..n).into_par_iter()
            .map(|i| {
                if i == best_idx {
                    return None;
                }

                let current_desc = descriptors.row(i);
                // Zero-Alloc Distance
                let d = current_desc.iter()
                    .zip(seed_desc.iter())
                    .fold(0.0, |acc, (a, b)| acc + (a - b).powi(2))
                    .sqrt();

                let score = (1.0 - lambda) * f_norm[i] + lambda * d;

                Some(Candidate {
                    index: i,
                    mmr_score: Float(score),
                    cached_d_min: d,
                    checked_count: 1, // Valid against the 1 initial elite
                })
            })
            .flatten()
            .collect();

        // Heapify O(N)
        let mut pq: BinaryHeap<Candidate> = BinaryHeap::from(candidates_vec);

        // 3. Incremental Lazy Greedy Loop
        while selected_indices.len() < self.target_k {
            if let Some(mut top) = pq.pop() {
                
                let current_archive_len = archive_indices.len();
                
                // CHECK: Is this candidate stale?
                if top.checked_count < current_archive_len {
                    let cand_idx = top.index;
                    let cand_desc = descriptors.row(cand_idx);
                    
                    let mut new_d_min = top.cached_d_min;

                    // OPTIMIZATION: Only scan the NEW elites added since we last checked
                    for i in top.checked_count..current_archive_len {
                        let elite_idx = archive_indices[i];
                        let elite_desc = descriptors.row(elite_idx);

                        let d = cand_desc.iter()
                            .zip(elite_desc.iter())
                            .fold(0.0, |acc, (a, b)| acc + (a - b).powi(2))
                            .sqrt();
                        
                        if d < new_d_min {
                            new_d_min = d;
                        }
                    }

                    // Update Candidate State
                    top.cached_d_min = new_d_min;
                    top.checked_count = current_archive_len;
                    
                    // Recalculate Score (using normalized fitness)
                    let new_score = (1.0 - lambda) * f_norm[cand_idx] + lambda * new_d_min;
                    top.mmr_score = Float(new_score);

                    // PEEK Strategy
                    let threshold = pq.peek().map(|c| c.mmr_score.0).unwrap_or(f64::NEG_INFINITY);

                    if new_score >= threshold {
                        // Winner
                        selected_indices.push(cand_idx);
                        archive_indices.push(cand_idx);
                    } else {
                        // Stale: Push back with updated score
                        pq.push(top);
                    }

                } else {
                    // Candidate is fresh, wins immediately
                    selected_indices.push(top.index);
                    archive_indices.push(top.index);
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
    fn test_small_logic_preserved() {
        let selector = MMRSelector::new(2, 0.5);
        let fitness = arr1(&[1.0, 0.5, 0.5]);
        let descriptors = arr2(&[
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 0.0]
        ]);

        let indices = selector.run_optimized(fitness.view(), descriptors.view());
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0], 0); 
        assert_eq!(indices[1], 2); 
    }
}
