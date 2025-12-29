"""
Random Search baseline.

Maintains archive of K best-diverse solutions without any selection pressure.
Used to verify that algorithms actually improve over random.
"""

import numpy as np
import time
from typing import Dict, Tuple
from .base import QDAlgorithm, QDResult, ExperimentConfig


class RandomSearch(QDAlgorithm):
    """
    Random search baseline that maintains top-K solutions.
    
    Generates random solutions and keeps the K with highest fitness.
    This is the simplest possible QD algorithm.
    """
    
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.n_dof = None
    
    def initialize(self, task, seed: int):
        """Initialize with random solutions."""
        np.random.seed(seed)
        
        self.n_dof = getattr(task, 'n_dof', getattr(task, 'n_dim', 20))
        
        # Generate initial population
        self.archive = np.random.uniform(
            -np.pi, np.pi, (self.config.archive_size, self.n_dof)
        )
        self.fitness, self.descriptors = task.evaluate(self.archive)
    
    def step(self, task) -> Dict[str, float]:
        """Generate random solutions and keep best K."""
        # Generate new random solutions
        new_solutions = np.random.uniform(
            -np.pi, np.pi, (self.config.batch_size, self.n_dof)
        )
        new_fit, new_desc = task.evaluate(new_solutions)
        
        # Pool with current archive
        pool_genes = np.vstack([self.archive, new_solutions])
        pool_fit = np.concatenate([self.fitness, new_fit])
        pool_desc = np.vstack([self.descriptors, new_desc])
        
        # Keep top K by fitness
        top_k_idx = np.argsort(pool_fit)[-self.config.archive_size:]
        
        self.archive = pool_genes[top_k_idx]
        self.fitness = pool_fit[top_k_idx]
        self.descriptors = pool_desc[top_k_idx]
        
        # Compute metrics
        from mmr_elites.metrics.qd_metrics import compute_all_metrics
        return compute_all_metrics(
            self.fitness, self.descriptors, self.config.archive_size
        )
    
    def get_archive(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return current archive state."""
        return self.archive, self.fitness, self.descriptors


def run_random_search(
    task,
    archive_size: int = 1000,
    generations: int = 1000,
    batch_size: int = 200,
    seed: int = 42,
    log_interval: int = 100,
) -> Dict:
    """Functional interface for Random Search."""
    config = ExperimentConfig(
        archive_size=archive_size,
        generations=generations,
        batch_size=batch_size,
        log_interval=log_interval,
    )
    
    alg = RandomSearch(config)
    result = alg.run(task, seed)
    
    return {
        "algorithm": result.algorithm,
        "seed": result.seed,
        "runtime": result.runtime,
        "final_metrics": result.final_metrics,
        "history": result.history,
        "final_genomes": result.final_genomes,
        "final_fitness": result.final_fitness,
        "final_descriptors": result.final_descriptors,
    }