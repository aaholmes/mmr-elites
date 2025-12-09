import numpy as np
from time import time
import mmr_elites_rs  # This is your compiled Rust library
from mmr_qd.tasks.arm20 import Arm20DOF

# Configuration
K_ELITES = 1000
GENERATIONS = 1000
BATCH_SIZE = 100
MUTATION_POWER = 0.1
LAMBDA = 0.5 # Balance between Fitness and Diversity

def run_mmr_elites():
    # 1. Setup
    task = Arm20DOF(target_pos=(15, 0)) # Target far to the right
    
    # The Rust Selector
    selector = mmr_elites_rs.MMRSelector(target_k=K_ELITES, lambda_val=LAMBDA)
    
    # Initial Population
    # Randomly initialize K genomes
    archive_genomes = np.random.uniform(-np.pi, np.pi, (K_ELITES, 20))
    archive_fitness, archive_desc = task.evaluate(archive_genomes)
    
    # We maintain the archive as a list of dicts or parallel arrays
    
    print(f"Starting MMR-Elites (K={K_ELITES}, 20-Dim Behavior Space)...")
    
    for gen in range(GENERATIONS):
        t0 = time()
        
        # 2. Reproduction (Simple Gaussian Mutation)
        # Select parents (randomly from current archive for now)
        parent_indices = np.random.randint(0, len(archive_genomes), size=BATCH_SIZE)
        parents = archive_genomes[parent_indices]
        
        noise = np.random.normal(0, MUTATION_POWER, size=parents.shape)
        offspring_genomes = parents + noise
        
        # Clip to bounds
        np.clip(offspring_genomes, -np.pi, np.pi, out=offspring_genomes)
        
        # 3. Evaluation
        off_fit, off_desc = task.evaluate(offspring_genomes)
        
        # 4. Construct Pool for Selection
        # Pool = Current Archive + Offspring
        pool_genomes = np.vstack([archive_genomes, offspring_genomes])
        pool_fit = np.concatenate([archive_fitness, off_fit])
        pool_desc = np.vstack([archive_desc, off_desc])
        
        # 5. Rust Selection (The Heavy Lifting)
        # We need to format this for the Rust API we designed.
        survivor_indices = selector.select(pool_fit, pool_desc)
        
        # 6. Update Archive
        archive_genomes = pool_genomes[survivor_indices]
        archive_fitness = pool_fit[survivor_indices]
        archive_desc = pool_desc[survivor_indices]
        
        # Logging
        if gen % 10 == 0:
            avg_fit = np.mean(archive_fitness)
            max_fit = np.max(archive_fitness)
            print(f"Gen {gen}: Max Fit {max_fit:.4f} | Avg Fit {avg_fit:.4f} | Archive Size {len(archive_genomes)}")

    return archive_genomes

if __name__ == "__main__":
    run_mmr_elites()