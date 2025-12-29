import numpy as np
import mmr_elites_rs
import pytest

def naive_greedy_mmr(fitness: np.ndarray, descriptors: np.ndarray, target_k: int, lambda_val: float):
    """
    A naive, non-lazy implementation of the MMR selection logic for verification.
    Complexity: O(K * N * K) ~ O(NK^2)
    """
    n_samples = fitness.shape[0]
    if n_samples <= target_k:
        return np.arange(n_samples)

    selected_indices = []
    remaining_indices = list(range(n_samples))

    # 1. Seed with Best Fitness
    best_idx = np.argmax(fitness)
    selected_indices.append(best_idx)
    remaining_indices.remove(best_idx)

    # 2. Greedy Loop
    while len(selected_indices) < target_k:
        best_candidate_idx = -1
        max_score = -np.inf

        # Check every remaining candidate
        for idx in remaining_indices:
            # Calculate d_min to CURRENT archive
            # d_min = min(dist(candidate, arch) for arch in archive)
            
            cand_desc = descriptors[idx]
            archive_descs = descriptors[selected_indices]
            
            # Vectorized distance calc for this candidate against all archive members
            dists = np.linalg.norm(archive_descs - cand_desc, axis=1)
            d_min = np.min(dists)
            
            score = (1.0 - lambda_val) * fitness[idx] + lambda_val * d_min
            
            if score > max_score:
                max_score = score
                best_candidate_idx = idx
        
        # Add best to archive
        if best_candidate_idx != -1:
            selected_indices.append(best_candidate_idx)
            remaining_indices.remove(best_candidate_idx)
        else:
            break
            
    return np.array(selected_indices)

def test_correctness_against_naive():
    """
    Verifies that the Optimized Rust Lazy Greedy implementation produces 
    the EXACT same subset as a Naive Python implementation.
    """
    np.random.seed(42)
    
    # Parameters
    n_samples = 100
    target_k = 10
    dim = 5
    lambda_val = 0.5
    
    # Generate Data
    fitness = np.random.rand(n_samples).astype(np.float64)
    descriptors = np.random.rand(n_samples, dim).astype(np.float64)
    
    # 1. Run Naive Python
    naive_selection = naive_greedy_mmr(fitness, descriptors, target_k, lambda_val)
    
    # 2. Run Rust Lazy Greedy
    selector = mmr_elites_rs.MMRSelector(target_k, lambda_val)
    rust_selection = selector.select(fitness, descriptors)
    
    print(f"\nNaive Selection ({len(naive_selection)}): {naive_selection}")
    print(f"Rust Selection  ({len(rust_selection)}): {rust_selection}")
    
    # 3. Assert Identical Sets
    # Note: Order might differ if scores are identical, but with random floats it's unlikely.
    # However, the greedy process is deterministic in order too.
    # Let's check exact equality of indices first.
    
    np.testing.assert_array_equal(
        naive_selection, 
        rust_selection, 
        err_msg="Rust selection did not match Naive selection exactly."
    )
    
    print("Test Passed: Rust Lazy Greedy matches Naive implementation exactly.")

if __name__ == "__main__":
    # Manually run if executed as script
    try:
        test_correctness_against_naive()
    except AssertionError as e:
        print(f"FAILED: {e}")
        exit(1)
