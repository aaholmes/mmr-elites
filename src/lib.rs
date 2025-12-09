use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray1, ToPyArray};
use ndarray::{ArrayView1, ArrayView2, Axis};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

// --- Float Wrapper for Heap (Same as before) ---
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
// Note: We no longer store the descriptor data here. Just the index.
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
struct MuseSelector {
    target_k: usize,
    lambda: f64,
}

#[pymethods]
impl MuseSelector {
    #[new]
    fn new(target_k: usize, lambda: f64) -> Self {
        MuseSelector { target_k, lambda }
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
    ) -> PyResult<&'py PyArray1<usize>> {
        
        // 1. Convert to Rust Views (Zero Copy)
        // These are safe slices into Python memory.
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
        // We delegate to a pure Rust function to keep the pyo3 logic clean
        let selected_indices = self.run_lazy_greedy(fit_view, desc_view);

        // 4. Return as Numpy Array
        Ok(PyArray1::from_vec(py, selected_indices))
    }
}

impl MuseSelector {
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
        // We store indices of selected items to lookup their descriptors later
        let mut archive_indices: Vec<usize> = Vec::with_capacity(self.target_k);

        // 1. Seed with Best Fitness
        // Helper to find argmax without dealing with NaNs
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
        
        // Pre-fetch the descriptor of the seed to compute initial distances
        let seed_desc = descriptors.row(best_idx);

        for i in 0..n {
            if i == best_idx { continue; }
            
            let current_desc = descriptors.row(i);
            
            // Initial distance is strictly to the seed
            // Optimizing Euclidean dist using ndarray
            let d = (&current_desc - &seed_desc)
                .mapv(|x| x.powi(2))
                .sum()
                .sqrt();

            let score = (1.0 - self.lambda) * fitness[i] + self.lambda * d;
            
            pq.push(Candidate {
                index: i,
                mmr_score: Float(score),
            });
        }

        // 3. Lazy Greedy Loop
        while selected_indices.len() < self.target_k {
            if let Some(mut top) = pq.pop() {
                let cand_idx = top.index;
                
                // LAZY CHECK
                // Recalculate d_min against the ENTIRE current archive.
                // Note: In production code, you can optimize this further by storing 
                // the 'last_d_min' for each candidate and only checking against 
                // newly added elites. For now, we scan the archive (Size K).
                
                let cand_desc = descriptors.row(cand_idx);
                let mut current_d_min = f64::INFINITY;

                for &archived_idx in &archive_indices {
                    let archive_desc = descriptors.row(archived_idx);
                    // dist calculation
                    let d = (&cand_desc - &archive_desc)
                        .mapv(|x| x.powi(2))
                        .sum()
                        .sqrt();
                    
                    if d < current_d_min {
                        current_d_min = d;
                    }
                }

                let new_score = (1.0 - self.lambda) * fitness[cand_idx] + self.lambda * current_d_min;

                // Peek next best
                let threshold = pq.peek().map(|c| c.mmr_score.0).unwrap_or(f64::NEG_INFINITY);

                if new_score >= threshold {
                    // Accepted
                    selected_indices.push(cand_idx);
                    archive_indices.push(cand_idx);
                } else {
                    // Rejected (for now), push back with updated score
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
