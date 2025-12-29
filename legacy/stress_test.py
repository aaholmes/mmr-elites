import time
import numpy as np
import mmr_elites_rs
import heapq

# --- Python Implementation (Baseline) ---
def python_lazy_greedy(fitness, descriptors, k, lambda_val):
    n = len(fitness)
    if n <= k: return list(range(n))

    selected_indices = []
    best_idx = np.argmax(fitness)
    selected_indices.append(best_idx)
    
    pq = []
    seed_desc = descriptors[best_idx]
    dists = np.linalg.norm(descriptors - seed_desc, axis=1)
    
    for i in range(n):
        if i == best_idx: continue
        score = (1.0 - lambda_val) * fitness[i] + lambda_val * dists[i]
        heapq.heappush(pq, (-score, i))
        
    while len(selected_indices) < k and pq:
        neg_score, idx = heapq.heappop(pq)
        
        # Optimization: Batch distance calculation in Numpy is actually faster 
        # than a loop in pure Python, so we give Python a fighting chance here.
        current_elites_desc = descriptors[selected_indices]
        cand_desc = descriptors[idx]
        d_min = np.min(np.linalg.norm(current_elites_desc - cand_desc, axis=1))
        
        new_score = (1.0 - lambda_val) * fitness[idx] + lambda_val * d_min
        
        if not pq:
            selected_indices.append(idx)
            break
            
        next_best_score = -pq[0][0]
        
        if new_score >= next_best_score:
            selected_indices.append(idx)
        else:
            heapq.heappush(pq, (-new_score, idx))
            
    return selected_indices

# --- The Stress Test Suite ---
def run_scenario(name, N, D, K):
    print(f"\n🧪 SCENARIO: {name}")
    print(f"   Configs: N={N:,}, D={D}, K={K}")
    
    # Generate Data
    np.random.seed(42)
    fitness = np.random.rand(N)
    descriptors = np.random.rand(N, D)
    lambda_val = 0.5
    
    # 1. Rust Run
    selector = mmr_elites_rs.MMRSelector(K, lambda_val)
    start = time.perf_counter()
    _ = selector.select(fitness, descriptors)
    rust_time = (time.perf_counter() - start) * 1000
    
    # 2. Python Run
    start = time.perf_counter()
    _ = python_lazy_greedy(fitness, descriptors, K, lambda_val)
    py_time = (time.perf_counter() - start) * 1000
    
    # 3. Report
    speedup = py_time / rust_time
    print(f"   🦀 Rust:   {rust_time:8.2f} ms")
    print(f"   🐍 Python: {py_time:8.2f} ms")
    print(f"   🚀 SPEEDUP: {speedup:.1f}x")

if __name__ == "__main__":
    print("🔥 IGNITING STRESS TEST ENGINE 🔥")
    print("-----------------------------------")
    
    # Scenario 1: The "Warm Up" (Previous Test)
    run_scenario("Warm Up", N=5000, D=20, K=500)
    
    # Scenario 2: "High Dimensions" (Where Numpy allocation hurts)
    # Rust iterator should shine here.
    run_scenario("High Dimensions", N=10_000, D=100, K=500)
    
    # Scenario 3: "Heavy Traffic" (Large Population)
    # This tests the Heap/Loop efficiency.
    run_scenario("Heavy Traffic", N=50_000, D=20, K=1000)
    
    # Scenario 4: "The Grid Killer" (Massive Archive)
    # The O(K * N) logic hits hard here.
    run_scenario("Massive Archive", N=20_000, D=50, K=5000)
