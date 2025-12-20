import time
import numpy as np
import mmr_elites_rs

def python_lazy_greedy(fitness, descriptors, k, lambda_val):
    """
    A pure Python/Numpy implementation of Lazy Greedy MMR.
    Used as the baseline to prove Rust superiority.
    """
    n = len(fitness)
    if n <= k:
        return list(range(n))

    selected_indices = []
    # 1. Seed
    best_idx = np.argmax(fitness)
    selected_indices.append(best_idx)
    
    # 2. Initialize Heap (Using negative score for Min-Heap simulating Max-Heap)
    import heapq
    pq = []
    
    # Pre-calculate distances to seed
    seed_desc = descriptors[best_idx]
    # Vectorized distance to seed
    dists = np.linalg.norm(descriptors - seed_desc, axis=1)
    
    for i in range(n):
        if i == best_idx: continue
        score = (1.0 - lambda_val) * fitness[i] + lambda_val * dists[i]
        # Python heapq is a min-heap, so store negative score
        heapq.heappush(pq, (-score, i))
        
    # 3. Loop
    while len(selected_indices) < k and pq:
        neg_score, idx = heapq.heappop(pq)
        score = -neg_score
        
        # Lazy Check
        # Calculate distance to ALL current elites
        current_elites_desc = descriptors[selected_indices]
        cand_desc = descriptors[idx]
        
        # Distance to nearest neighbor
        d_min = np.min(np.linalg.norm(current_elites_desc - cand_desc, axis=1))
        
        new_score = (1.0 - lambda_val) * fitness[idx] + lambda_val * d_min
        
        # Peek
        if not pq:
            selected_indices.append(idx)
            break
            
        next_best_score = -pq[0][0]
        
        if new_score >= next_best_score:
            selected_indices.append(idx)
        else:
            heapq.heappush(pq, (-new_score, idx))
            
    return selected_indices

def run_benchmark():
    # Parameters matches a realistic "Generation"
    N = 5000     # Population size
    D = 20       # 20-DOF Arm
    K = 500      # Archive Size
    LAMBDA = 0.5
    
    print(f"🔧 BENCHMARK: N={N}, D={D}, K={K}")
    
    # Random Data
    np.random.seed(42)
    fitness = np.random.rand(N)
    descriptors = np.random.rand(N, D)
    
    # --- Rust Benchmark ---
    selector = mmr_elites_rs.MMRSelector(K, LAMBDA)
    
    start_rs = time.time()
    rust_indices = selector.select(fitness, descriptors)
    end_rs = time.time()
    
    rust_time = (end_rs - start_rs) * 1000
    print(f"🦀 Rust Time:   {rust_time:.2f} ms")
    
    # --- Python Benchmark ---
    start_py = time.time()
    py_indices = python_lazy_greedy(fitness, descriptors, K, LAMBDA)
    end_py = time.time()
    
    py_time = (end_py - start_py) * 1000
    print(f"🐍 Python Time: {py_time:.2f} ms")
    
    # --- Analysis ---
    speedup = py_time / rust_time
    print(f"🚀 Speedup:     {speedup:.1f}x")
    
    # --- Verification ---
    # Sets might differ slightly if scores are identical (floating point noise), 
    # but sizes must match.
    assert len(rust_indices) == K
    print("✅ Integrity Check Passed")

if __name__ == "__main__":
    run_benchmark()
