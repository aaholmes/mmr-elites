import numpy as np
import mmr_elites_rs
from tasks.arm_20 import Arm20Task
import time
import pickle

# --- Config ---
SEED = 1337          # <--- REPRODUCIBILITY LOCK
GENERATIONS = 3000
BATCH_SIZE = 500
ARCHIVE_SIZE = 1000
LAMBDA = 0.95
SIGMA = 0.1

def main():
    # 1. Set Seed
    np.random.seed(SEED)
    print(f"🚀 LAUNCHING MUSE-QD: DATA RUN (Seed={SEED})")
    
    task = Arm20Task()
    selector = mmr_elites_rs.MMRSelector(ARCHIVE_SIZE, LAMBDA)
    
    # 2. Init
    archive = np.random.uniform(-np.pi, np.pi, (ARCHIVE_SIZE, 20))
    fit, desc = task.evaluate(archive)
    indices = selector.select(fit, desc)
    archive = archive[indices]
    fit = fit[indices]
    desc = desc[indices]
    
    # 3. History Storage
    history = {"gen": [], "max_fit": []}
    tips_snapshots = {} 
    
    start_time = time.time()
    
    # 4. Evolution Loop
    for gen in range(1, GENERATIONS + 1):
        parents = archive[np.random.randint(0, len(archive), BATCH_SIZE)]
        offspring = parents + np.random.normal(0, SIGMA, (BATCH_SIZE, 20))
        off_fit, off_desc = task.evaluate(offspring)
        
        pool_genes = np.vstack([archive, offspring])
        pool_fit = np.concatenate([fit, off_fit])
        pool_desc = np.vstack([desc, off_desc])
        
        survivor_idx = selector.select(pool_fit, pool_desc)
        archive = pool_genes[survivor_idx]
        fit = pool_fit[survivor_idx]
        desc = pool_desc[survivor_idx]
        
        # LOGGING (Every 10 Gens)
        if gen % 10 == 0:
            max_f = np.max(fit)
            tips_snapshots[gen] = desc.copy()
            history["gen"].append(gen)
            history["max_fit"].append(max_f)
            
            if gen % 100 == 0:
                print(f"Gen {gen:4d} | Max: {max_f:.4f} | Size: {len(archive)}")
            
            if max_f > 0.98:
                print(f"✅ SOLVED at Gen {gen}!")
                break

    print(f"✅ Done in {time.time() - start_time:.2f}s")
    
    data = {
        "history": history,
        "snapshots": tips_snapshots,
        "final_tips": desc
    }
    with open("muse_results.pkl", "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()
