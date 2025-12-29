import numpy as np
import mmr_elites_rs
import time
import pickle
from tasks.ant import AntTask

# --- CONFIG ---
SEED = 1337
GENERATIONS = 1000
BATCH_SIZE = 200     # Same as baseline
ARCHIVE_SIZE = 10    # Reduced for fast testing
LAMBDA = 0.5         # Balanced: 50% Fitness, 50% Diversity
SIGMA = 0.02         # Mutation Power (Standard for Neural Nets)
INIT_SIGMA = 0.1     # Initialization Range

def main():
    np.random.seed(SEED)
    print(f"🚀 LAUNCHING MUSE-QD: ANT BENCHMARK")
    print(f"   Archive: {ARCHIVE_SIZE} | Lambda: {LAMBDA}")
    
    # Initialize the Parallel Task Adapter
    task = AntTask(workers=30) 
    
    # Initialize Rust Selector
    selector = mmr_elites_rs.MMRSelector(ARCHIVE_SIZE, LAMBDA)
    
    # 1. Initialize Population
    print("   Initializing Population...")
    archive = np.random.uniform(-INIT_SIGMA, INIT_SIGMA, (ARCHIVE_SIZE, task.genome_size))
    # Initial seed 0 for population init
    fit, desc = task.evaluate_batch(archive, current_gen_seed=0)
    
    # Initial Select (Fill archive with best initial randoms)
    indices = selector.select(fit, desc)
    archive = archive[indices]
    fit = fit[indices]
    desc = desc[indices]
    
    history = {"gen": [], "max_fit": [], "archive_size": []}
    
    start_time = time.time()
    
    for gen in range(1, GENERATIONS + 1):
        # 2. Mutate (Standard Evolutionary Strategy)
        # Pick parents uniformly from the archive
        parents_idx = np.random.randint(0, len(archive), BATCH_SIZE)
        parents = archive[parents_idx]
        offspring = parents + np.random.normal(0, SIGMA, parents.shape)
        
        # 3. Evaluate Offspring (Passing gen as seed)
        off_fit, off_desc = task.evaluate_batch(offspring, current_gen_seed=gen)
        
        # 4. Pool & Select (The MMR Step)
        # Combine [Archive + Offspring] -> Select Best K
        pool_genes = np.vstack([archive, offspring])
        pool_fit = np.concatenate([fit, off_fit])
        pool_desc = np.vstack([desc, off_desc])
        
        survivor_idx = selector.select(pool_fit, pool_desc)
        
        archive = pool_genes[survivor_idx]
        fit = pool_fit[survivor_idx]
        desc = pool_desc[survivor_idx]
        
        # 5. Logging
        if gen % 10 == 0:
            max_f = np.max(fit)
            print(f"Gen {gen:4d} | Archive: {len(archive):4d} | Max Fit: {max_f:.2f}")
            history["gen"].append(gen)
            history["max_fit"].append(max_f)
            history["archive_size"].append(len(archive))
            
            # Auto-save every 100 gens just in case
            if gen % 100 == 0:
                with open("muse_ant_results.pkl", "wb") as f:
                    data = {"history": history, "final_descriptors": desc, "final_fitness": fit}
                    pickle.dump(data, f)

    total_time = time.time() - start_time
    print(f"✅ Done in {total_time:.2f}s")
    task.close()
    
    # Final Save
    data = {
        "history": history,
        "final_descriptors": desc,
        "final_fitness": fit
    }
    with open("muse_ant_results.pkl", "wb") as f:
        pickle.dump(data, f)
    print("💾 Saved results to muse_ant_results.pkl")

if __name__ == "__main__":
    main()
