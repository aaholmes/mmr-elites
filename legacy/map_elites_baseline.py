import numpy as np
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from tasks.ant import AntTask, eval_one_ant

# --- CONFIG ---
SEED = 1337
GENERATIONS = 1000  # Adjust based on patience
BATCH_SIZE = 200    # Eval batch size
SIGMA = 0.02        # Mutation power (Neural nets need small sigma)
GRID_RES = 50       # 50x50 grid
RANGE_X = [-30, 30] # Expected range of Ant
RANGE_Y = [-30, 30]
WORKERS = 30        # Utilize your 7950x!

def get_grid_idx(descriptor):
    """Discretize continuous (x,y) into grid index (i, j)."""
    # Normalize to [0, 1]
    norm_x = (descriptor[0] - RANGE_X[0]) / (RANGE_X[1] - RANGE_X[0])
    norm_y = (descriptor[1] - RANGE_Y[0]) / (RANGE_Y[1] - RANGE_Y[0])
    
    # Clip
    norm_x = np.clip(norm_x, 0, 0.999)
    norm_y = np.clip(norm_y, 0, 0.999)
    
    idx_x = int(norm_x * GRID_RES)
    idx_y = int(norm_y * GRID_RES)
    return (idx_x, idx_y)

def main():
    np.random.seed(SEED)
    task = AntTask()
    print(f"🐜 Launching MAP-Elites Baseline on 7950x ({WORKERS} threads)")
    print(f"   Genome Size: {task.param_count}")
    
    # Init Population (Random Weights)
    # Using smaller init range for Tanh stability
    current_batch = np.random.uniform(-0.1, 0.1, (BATCH_SIZE, task.param_count))
    
    # The Archive: Dict mapping (i,j) -> {genome, fitness, desc}
    archive = {}
    
    history_gen = []
    history_coverage = []
    history_max_fit = []
    
    start_time = time.time()
    
    # Process Pool for Parallel Eval
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        
        for gen in range(1, GENERATIONS + 1):
            # 1. Evaluate Batch (Parallel)
            results = list(executor.map(eval_one_ant, current_batch))
            
            # 2. Add to Archive
            for i, (fit, desc) in enumerate(results):
                genome = current_batch[i]
                idx = get_grid_idx(desc)
                
                # If cell empty OR new guy is better -> Replace
                if idx not in archive or fit > archive[idx]["fitness"]:
                    archive[idx] = {
                        "genome": genome,
                        "fitness": fit,
                        "desc": desc
                    }
            
            # 3. Select & Mutate (Generate next batch)
            # Pick random elites from archive
            keys = list(archive.keys())
            if len(keys) > 0:
                parent_indices = np.random.randint(0, len(keys), BATCH_SIZE)
                parents = [archive[keys[k]]["genome"] for k in parent_indices]
                parents = np.array(parents)
                
                # Gaussian Mutation
                noise = np.random.normal(0, SIGMA, parents.shape)
                current_batch = parents + noise
            else:
                # If archive empty (unlikely after Gen 1), restart random
                current_batch = np.random.uniform(-0.5, 0.5, (BATCH_SIZE, task.param_count))
            
            # 4. Logging
            if gen % 10 == 0:
                max_f = max(d["fitness"] for d in archive.values()) if archive else 0
                coverage = len(archive)
                print(f"Gen {gen:4d} | Archive: {coverage:4d} cells | Max Fit: {max_f:.2f}")
                
                history_gen.append(gen)
                history_coverage.append(coverage)
                history_max_fit.append(max_f)
    
    total_time = time.time() - start_time
    print(f"✅ Done in {total_time:.2f}s")
    
    # Save Data
    # Extract just the descriptors for plotting later
    all_tips = np.array([d["desc"] for d in archive.values()])
    
    data = {
        "history_gen": history_gen,
        "history_coverage": history_coverage,
        "final_descriptors": all_tips,
        "archive_size": len(archive)
    }
    
    with open("map_elites_ant_results.pkl", "wb") as f:
        pickle.dump(data, f)
    print("💾 Saved results to map_elites_ant_results.pkl")

if __name__ == "__main__":
    main()
