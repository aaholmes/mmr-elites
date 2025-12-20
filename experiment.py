import numpy as np
import mmr_elites_rs
from tasks.arm_20 import Arm20Task
import time
import pickle

# --- Config ---
GENERATIONS = 3000
BATCH_SIZE = 500
ARCHIVE_SIZE = 1000  
LAMBDA = 0.95        # High Diversity
SIGMA = 0.1          

def get_novelty(candidate, archive):
    """
    Compute distance to the nearest neighbor in the archive.
    (Simple Brute Force for Debugging)
    """
    dists = np.linalg.norm(archive - candidate, axis=1)
    # Filter out 0.0 (distance to self if candidate is in archive)
    dists = dists[dists > 1e-6]
    if len(dists) == 0: return 0.0
    return np.min(dists)

def main():
    print(f"🚀 LAUNCHING MUSE-QD: DIVERSITY CHECK")
    task = Arm20Task()
    selector = mmr_elites_rs.MMRSelector(ARCHIVE_SIZE, LAMBDA)
    
    # Init
    archive = np.random.uniform(-np.pi, np.pi, (ARCHIVE_SIZE, 20))
    fit, desc = task.evaluate(archive)
    indices = selector.select(fit, desc)
    archive = archive[indices]
    fit = fit[indices]
    desc = desc[indices]
    
    history = {"gen": [], "max_fit": []}
    start_time = time.time()
    
    for gen in range(1, GENERATIONS + 1):
        # Mutate
        parents = archive[np.random.randint(0, len(archive), BATCH_SIZE)]
        offspring = parents + np.random.normal(0, SIGMA, (BATCH_SIZE, 20))
        off_fit, off_desc = task.evaluate(offspring)
        
        # Pool
        pool_genes = np.vstack([archive, offspring])
        pool_fit = np.concatenate([fit, off_fit])
        pool_desc = np.vstack([desc, off_desc])
        
        # Select
        survivor_idx = selector.select(pool_fit, pool_desc)
        archive = pool_genes[survivor_idx]
        fit = pool_fit[survivor_idx]
        desc = pool_desc[survivor_idx]
        
        if gen % 50 == 0:
            max_f = np.max(fit)
            
            # --- DIVERSITY CHECK ---
            # Get indices of Top 3 Fitness
            top_3_idx = np.argsort(fit)[-3:][::-1]
            
            print(f"Gen {gen:4d} | Max Fit: {max_f:.4f} | Size: {len(archive)}")
            print(f"   Top 3 Elites:")
            for i, idx in enumerate(top_3_idx):
                f_val = fit[idx]
                # Calculate how unique this elite is (Distance to nearest neighbor)
                nov_val = get_novelty(desc[idx], desc)
                print(f"     #{i+1}: Fit={f_val:.4f} | Novelty (Dist to NN)={nov_val:.4f}")
            
            history["gen"].append(gen)
            history["max_fit"].append(max_f)

            if max_f > 0.98:
                print(f"✅ CONVERGED. (Fit: {max_f:.4f})")
                break

    print(f"✅ Done in {time.time() - start_time:.2f}s")
    
    data = {"history": history, "final_tips": desc}
    with open("muse_results.pkl", "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()
