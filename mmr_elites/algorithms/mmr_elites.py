"""
MMR-Elites: Maximum Marginal Relevance Selection of Elites.

Core contribution of the GECCO paper.
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple
from .base import QDAlgorithm, QDResult, ExperimentConfig

try:
    import mmr_elites_rs
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


class MMRElites(QDAlgorithm):
    """
    MMR-Elites algorithm using Rust backend.
    
    Selection criterion:
        Score(x) = (1 - λ) · fitness(x) + λ · d_min(x, Archive)
    
    Properties:
        - Fixed archive size K
        - O(K log K) selection via lazy greedy
        - Explicit diversity optimization
    """
    
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        if not RUST_AVAILABLE:
            raise RuntimeError(
                "Rust backend required for MMR-Elites. "
                "Run: maturin develop --release"
            )
        self.selector = mmr_elites_rs.MMRSelector(
            config.archive_size, 
            config.lambda_val
        )
        self.n_dof = None
    
    def initialize(self, task, seed: int):
        """Initialize archive with random solutions."""
        np.random.seed(seed)
        
        self.n_dof = getattr(task, 'n_dof', getattr(task, 'n_dim', 20))
        
        # Random initial population
        self.archive = np.random.uniform(
            -np.pi, np.pi, (self.config.archive_size, self.n_dof)
        )
        self.fitness, self.descriptors = task.evaluate(self.archive)
        
        # Initial selection to get diverse starting set
        idx = self.selector.select(self.fitness, self.descriptors)
        self.archive = self.archive[idx]
        self.fitness = self.fitness[idx]
        self.descriptors = self.descriptors[idx]
    
    def step(self, task) -> Dict[str, float]:
        """Perform one generation."""
        # Mutation
        parent_idx = np.random.randint(0, len(self.archive), self.config.batch_size)
        parents = self.archive[parent_idx]
        offspring = parents + np.random.normal(
            0, self.config.mutation_sigma, 
            (self.config.batch_size, self.n_dof)
        )
        offspring = np.clip(offspring, -np.pi, np.pi)
        
        # Evaluation
        off_fit, off_desc = task.evaluate(offspring)
        
        # Pool and select
        pool_genes = np.vstack([self.archive, offspring])
        pool_fit = np.concatenate([self.fitness, off_fit])
        pool_desc = np.vstack([self.descriptors, off_desc])
        
        survivor_idx = self.selector.select(pool_fit, pool_desc)
        
        self.archive = pool_genes[survivor_idx]
        self.fitness = pool_fit[survivor_idx]
        self.descriptors = pool_desc[survivor_idx]
        
        # Compute metrics
        from mmr_elites.metrics.qd_metrics import compute_all_metrics
        return compute_all_metrics(
            self.fitness, self.descriptors, self.config.archive_size
        )
    
    def get_archive(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return current archive state."""
        return self.archive, self.fitness, self.descriptors


def run_mmr_elites(
    task,
    archive_size: int = 1000,
    generations: int = 1000,
    batch_size: int = 200,
    lambda_val: float = 0.5,
    mutation_sigma: float = 0.1,
    seed: int = 42,
    log_interval: int = 100,
) -> Dict:
    """
    Functional interface for MMR-Elites.
    
    Args:
        task: Task object with evaluate(genomes) method
        archive_size: Number of solutions to maintain (K)
        generations: Number of generations
        batch_size: Offspring per generation
        lambda_val: Diversity weight λ ∈ [0, 1]
        mutation_sigma: Gaussian mutation std
        seed: Random seed
        log_interval: How often to log metrics
    
    Returns:
        Dictionary with results and history
    """
    config = ExperimentConfig(
        archive_size=archive_size,
        generations=generations,
        batch_size=batch_size,
        lambda_val=lambda_val,
        mutation_sigma=mutation_sigma,
        log_interval=log_interval,
    )
    
    alg = MMRElites(config)
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