import numpy as np
import mmr_elites_rs
from tasks.arm_20 import Arm20Task
import time
import pickle

# --- Config ---
GENERATIONS = 2000
BATCH_SIZE = 200     # New offspring per generation
ARCHIVE_SIZE = 1000  # K elites
LAMBDA = 0.5         # 50% Fitness, 50% Diversity
SIGMA = 0.05         # Mutation Strength (Gaussian)

def main():
    print(f"🚀 LAUNCHING MUSE-QD: 20-DOF ARM TRAP")
    print(f"   Configs: Gen={GENERATIONS}, K={ARCHIVE_SIZE}, λ={LAMBDA}")
    
    # 1. Init Task & Engine
    task = Arm20Task()
    selector = mmr_elites_rs.MMRSelector(ARCHIVE_SIZE, LAMBDA)
    
    # 2. Random Start
    # Archive shape: (K, 20)
    archive = np.random.uniform(-np.pi, np.pi, (ARCHIVE_SIZE, 20))
    fit, descriptors = task.evaluate(archive)
    
    # Initial Selection (Pruning random garbage)
    indices = selector.select(fit, archive)
    archive = archive[indices]
    fit = fit[indices]
    
    # Metrics History
    history = {
        "gen": [],
        "max_fit": [],
        "avg_fit": [],
        "archive_size": [],
        "coverage_metric": [] # Mean Nearest Neighbor Dist
    }
    
    start_time = time.time()
    
    # 3. The Loop
    for gen in range(1, GENERATIONS + 1):
        # --- A. Mutation (Simple Gaussian for now) ---
        # Pick parents randomly from elites
        parent_indices = np.random.randint(0, len(archive), BATCH_SIZE)
        parents = archive[parent_indices]
        noise = np.random.normal(0, SIGMA, (BATCH_SIZE, 20))
        offspring = parents + noise
        
        # --- B. Evaluation ---
        off_fit, off_desc = task.evaluate(offspring)
        
        # --- C. Survival (Rust Engine) ---
        # Pool = Old Elites + New Offspring
        pool_genes = np.vstack([archive, offspring])
        pool_fit = np.concatenate([fit, off_fit])
        
        # The Rust Magic:
        survivor_idx = selector.select(pool_fit, pool_genes)
        
        # Update Archive
        archive = pool_genes[survivor_idx]
        fit = pool_fit[survivor_idx]
        
        # --- D. Logging ---
        if gen % 50 == 0:
            max_f = np.max(fit)
            elapsed = time.time() - start_time
            print(f"Gen {gen:4d} | Max Fit: {max_f:.4f} | Size: {len(archive)} | Time: {elapsed:.1f}s")
            
            history["gen"].append(gen)
            history["max_fit"].append(max_f)
            history["avg_fit"].append(np.mean(fit))
            history["archive_size"].append(len(archive))

    # 4. Save Artifacts
    print("✅ Experiment Complete.")
    
    # Get final descriptors for plotting
    final_fit, final_desc = task.evaluate(archive)
    
    data = {
        "history": history,
        "final_archive": final_desc, # (K, 20) - End Effector positions? 
        # Wait, task.evaluate returns (fit, tips). 
        # For the plot we want TIPS (2D) to show the coverage, 
        # but the algorithm used JOINTS (20D) for diversity.
        # This proves the "High-D -> Low-D Projection" capability.
        "final_tips": final_desc
    }
    
    with open("muse_results.pkl", "wb") as f:
        pickle.dump(data, f)
    print("💾 Results saved to muse_results.pkl")

if __name__ == "__main__":
    main()
