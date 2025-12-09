use std::collections::BinaryHeap;
use std::cmp::Ordering;

// Wrapper to allow f64 in BinaryHeap
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

#[derive(Clone, Debug)]
pub struct Individual {
    pub id: usize,
    pub fitness: f64,
    pub descriptor: Vec<f64>,
}

/// A candidate wrapper for the Priority Queue
#[derive(PartialEq, Eq)]
struct Candidate {
    index: usize,
    mmr_score: Float, 
}

// We want a Max-Heap, so we derive standard ordering based on mmr_score
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

pub struct MuseSelector {
    target_k: usize,
    lambda: f64, // Mixing parameter (0.0 = Fitness only, 1.0 = Diversity only)
}

impl MuseSelector {
    pub fn new(target_k: usize, lambda: f64) -> Self {
        Self { target_k, lambda }
    }

    fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
    }

    /// The Lazy Greedy Selection Loop
    /// Returns the indices of the selected survivors
    pub fn select(&self, population: &Vec<Individual>) -> Vec<usize> {
        if population.len() <= self.target_k {
            // If pop size < K, keep everyone (or handle as error)
            return (0..population.len()).collect();
        }

        let mut selected_indices = Vec::with_capacity(self.target_k);
        // Keep track of actual values added to archive for distance computations
        let mut archive_descriptors: Vec<&Vec<f64>> = Vec::with_capacity(self.target_k);
        
        // 1. Initialization: Find Best Fitness
        // We start with the highest fitness individual to anchor the distribution.
        let mut best_idx = 0;
        let mut max_fit = f64::NEG_INFINITY;

        for (i, ind) in population.iter().enumerate() {
            if ind.fitness > max_fit {
                max_fit = ind.fitness;
                best_idx = i;
            }
        }

        selected_indices.push(best_idx);
        archive_descriptors.push(&population[best_idx].descriptor);

        // 2. Initialize Heap
        // Initially, d_min is infinity (or max possible dist).
        // Score = (1-L)*Fit + L*Infinity.
        // Optimization: We can just use Fitness as the initial sorting key if we assume 
        // infinite diversity for the first step, or calculate d_min to the single seed.
        
        let mut pq = BinaryHeap::new();

        // Pre-calculate normalized fitness or raw fitness logic here.
        // For simplicity, we assume fitness is already scaled or we use raw.
        
        // Initial population of the queue
        for (i, ind) in population.iter().enumerate() {
            if i == best_idx { continue; }
            
            // Initial distance is to the ONE item currently in archive
            let d = Self::euclidean_dist(&ind.descriptor, archive_descriptors[0]);
            let score = (1.0 - self.lambda) * ind.fitness + self.lambda * d;
            
            pq.push(Candidate {
                index: i,
                mmr_score: Float(score),
            });
        }

        // 3. Main Loop
        while selected_indices.len() < self.target_k {
            if let Some(mut top) = pq.pop() {
                let candidate_idx = top.index;
                let candidate_ind = &population[candidate_idx];

                // LAZY CHECK:
                // The score in 'top' was calculated based on an archive of size N.
                // The archive is now size N + M.
                // d_min can ONLY decrease. Therefore, current 'top.mmr_score' is an UPPER BOUND.
                
                // Re-calculate exact d_min against the current full archive
                // (We only need to check dist against the *newly added* elites since the last update
                // of this node, but for simplicity here we check all. 
                // A true lazy implementation stores a 'last_updated' timestamp.)
                
                let mut current_d_min = f64::INFINITY;
                for existing_desc in &archive_descriptors {
                    let d = Self::euclidean_dist(&candidate_ind.descriptor, existing_desc);
                    if d < current_d_min {
                        current_d_min = d;
                    }
                }

                let new_score = (1.0 - self.lambda) * candidate_ind.fitness + self.lambda * current_d_min;

                // Peek at the NEXT best in the queue
                let threshold = pq.peek().map(|c| c.mmr_score.0).unwrap_or(f64::NEG_INFINITY);

                if new_score >= threshold {
                    // Even after reduction, this is still better than the upper bound of the next best.
                    // SELECT IT.
                    selected_indices.push(candidate_idx);
                    archive_descriptors.push(&candidate_ind.descriptor);
                } else {
                    // It degraded too much. Push it back with the new score.
                    // It will likely sink in the heap.
                    pq.push(Candidate {
                        index: candidate_idx,
                        mmr_score: Float(new_score), // Update with new valid score
                    });
                }
            } else {
                break; // Queue empty
            }
        }

        selected_indices
    }
}
