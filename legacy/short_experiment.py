import numpy as np
import mmr_elites_rs
from tasks.arm_20 import Arm20Task
import time

# --- Config ---
GENERATIONS = 500
BATCH_SIZE = 200
ARCHIVE_SIZE = 500 
LAMBDA = 0.5       
SIGMA = 0.05       # Precision mode

def run_sanity_check():
    print(f"🧪 SANITY CHECK v3: Correct Descriptor Passing")
    task = Arm20Task()
    selector = mmr_elites_rs.MMRSelector(ARCHIVE_SIZE, LAMBDA)
    
    # 1. Init Population
    archive = np.random.uniform(-np.pi, np.pi, (ARCHIVE_SIZE, 20))
    fit, desc = task.evaluate(archive) # Get Fitness AND Descriptors (Tips)
    
    # 2. Init Selection (Pass DESC, not ARCHIVE)
    indices = selector.select(fit, desc)
    archive = archive[indices]
    fit = fit[indices]
    desc = desc[indices] # <--- KEEP DESCRIPTORS SYNCED
    
    start_time = time.time()
    
    for gen in range(1, GENERATIONS + 1):
        # A. Mutate
        parents_idx = np.random.randint(0, len(archive), BATCH_SIZE)
        parents = archive[parents_idx]
        offspring = parents + np.random.normal(0, SIGMA, (BATCH_SIZE, 20))
        
        # B. Evaluate
        off_fit, off_desc = task.evaluate(offspring)
        
        # C. Create Pools
        pool_genes = np.vstack([archive, offspring])
        pool_fit = np.concatenate([fit, off_fit])
        pool_desc = np.vstack([desc, off_desc]) # <--- STACK DESCRIPTORS
        
        # D. Selection (The Fix: Pass pool_desc!)
        survivor_idx = selector.select(pool_fit, pool_desc)
        
        # E. Update State
        archive = pool_genes[survivor_idx]
        fit = pool_fit[survivor_idx]
        desc = pool_desc[survivor_idx] # <--- Track descriptors for next gen
        
        if gen % 50 == 0:
            max_f = np.max(fit)
            print(f"Gen {gen:3d} | Max Fit: {max_f:.4f} | Size: {len(archive)}")
            
            # If we pass 0.99, the engine is proven.
            if max_f > 0.99:
                print(f"\n✅ SUCCESS! Reached Target (Fit: {max_f:.4f})")
                print(f"   Time: {time.time() - start_time:.2f}s")
                return

    print("\n❌ FAILURE: Did not reach target.")

if __name__ == "__main__":
    run_sanity_check()
