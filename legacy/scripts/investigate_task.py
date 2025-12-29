"""
Diagnostic script to investigate fitness distribution and collision rates.
"""

import numpy as np
import matplotlib.pyplot as plt
from mmr_elites.tasks.arm import ArmTask

def investigate_task(n_dof=20, n_samples=10000):
    task = ArmTask(n_dof=n_dof, use_highdim_descriptor=True)
    
    # Sample random genomes
    genomes = np.random.uniform(-np.pi, np.pi, (n_samples, n_dof))
    
    # Evaluate
    fitness, descriptors = task.evaluate(genomes)
    
    # Calculate collision rate
    collision_rate = np.mean(fitness == 0)
    print(f"Task: {n_dof}-DOF Arm")
    print(f"Collision Rate: {collision_rate:.2%}")
    
    # Fitness distribution of non-colliding
    valid_fitness = fitness[fitness > 0]
    if len(valid_fitness) > 0:
        print(f"Mean Fitness (valid): {np.mean(valid_fitness):.4f}")
        print(f"Max Fitness: {np.max(valid_fitness):.4f}")
        print(f"90th percentile: {np.percentile(valid_fitness, 90):.4f}")
    else:
        print("No valid solutions found!")

    # Plot fitness distribution
    plt.figure(figsize=(10, 6))
    plt.hist(fitness, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.title(f"Fitness Distribution ({n_dof}-DOF Arm, {n_samples} samples)")
    plt.xlabel("Fitness")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"results/fitness_dist_{n_dof}d.png")
    # plt.show()

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    investigate_task(n_dof=5)
    investigate_task(n_dof=20)
    investigate_task(n_dof=50)
