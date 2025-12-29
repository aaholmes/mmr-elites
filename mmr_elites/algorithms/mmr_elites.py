"""
MMR-Elites: Maximum Marginal Relevance Selection of Elites.

Core contribution of the paper.
"""

import numpy as np
import time
from typing import Dict, Optional, List
from .base import QDAlgorithm, QDResult
from ..metrics.qd_metrics import compute_all_metrics

try:
    import mmr_elites_rs
    RUST_AVAILABLE = True
except ImportError:
    # Try local build import if installed in development mode or in path
    try:
        import src.mmr_elites_rs as mmr_elites_rs
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
    
    def __init__(self, archive_size: int, lambda_val: float = 0.5):
        """
        Initialize MMR-Elites.
        
        Args:
            archive_size: Target archive size K
            lambda_val: Diversity weight λ ∈ [0, 1]
        """
        if not RUST_AVAILABLE:
            raise RuntimeError(
                "Rust backend not available. Please compile the Rust extension.\n"
                "Ensure 'mmr_elites_rs' is importable."
            )
        
        self.archive_size = archive_size
        self.lambda_val = lambda_val
        self.selector = mmr_elites_rs.MMRSelector(archive_size, lambda_val)
        
        # Archive state
        self.archive: Optional[np.ndarray] = None
        self.fitness: Optional[np.ndarray] = None
        self.descriptors: Optional[np.ndarray] = None
    
    def run(
        self,
        task,
        generations: int,
        batch_size: int,
        mutation_sigma: float,
        seed: int,
        log_interval: int = 100,
    ) -> QDResult:
        """Run MMR-Elites evolution."""
        np.random.seed(seed)
        
        # Infer genome dimension from task
        # Try different attributes to find dimension
        if hasattr(task, 'genome_dim'):
            n_dof = task.genome_dim
        elif hasattr(task, 'n_dof'):
            n_dof = task.n_dof
        elif hasattr(task, 'n_dim'):
            n_dof = task.n_dim
        else:
            # Fallback: evaluate one genome to find out
            temp_genome = np.zeros((1, 1)) # Dummy, task likely expects correct dim
            # This is risky if we don't know dim. Let's assume standard 20 if not specified
            # or try to get bounds
            n_dof = 20
        
        # Initialize archive with random solutions
        # We start with archive_size random solutions to fill the buffer
        # But actually we should start with a small population and grow? 
        # No, MMR-Elites typically selects K from a pool.
        # Standard QD loop: initialization -> selection -> archive
        
        # Initialize with MORE than archive size to get good initial coverage
        init_size = max(batch_size, self.archive_size)
        init_pop = np.random.uniform(-np.pi, np.pi, (init_size, n_dof))
        
        init_fit, init_desc = task.evaluate(init_pop)
        
        # Initial selection
        idx = self.selector.select(init_fit, init_desc)
        self.archive = init_pop[idx]
        self.fitness = init_fit[idx]
        self.descriptors = init_desc[idx]
        
        # History tracking
        history = {
            "generation": [],
            "qd_score": [],
            "qd_score_at_budget": [],
            "max_fitness": [],
            "mean_fitness": [],
            "mean_pairwise_distance": [],
            "uniformity_cv": [],
            "archive_size": [],
        }
        
        start_time = time.time()
        
        for gen in range(1, generations + 1):
            # Mutation
            # Sample parents uniformly from current archive
            if len(self.archive) > 0:
                parent_idx = np.random.randint(0, len(self.archive), batch_size)
                parents = self.archive[parent_idx]
                offspring = parents + np.random.normal(0, mutation_sigma, (batch_size, n_dof))
                offspring = np.clip(offspring, -np.pi, np.pi)
                
                # Evaluation
                off_fit, off_desc = task.evaluate(offspring)
                
                # Pool and select
                # We combine current archive + offspring and select K best
                pool_genes = np.vstack([self.archive, offspring])
                pool_fit = np.concatenate([self.fitness, off_fit])
                pool_desc = np.vstack([self.descriptors, off_desc])
            else:
                # Should not happen if initialized correctly
                continue
            
            survivor_idx = self.selector.select(pool_fit, pool_desc)
            
            self.archive = pool_genes[survivor_idx]
            self.fitness = pool_fit[survivor_idx]
            self.descriptors = pool_desc[survivor_idx]
            
            # Logging
            if gen % log_interval == 0:
                metrics = compute_all_metrics(
                    self.fitness, self.descriptors, budget_k=self.archive_size
                )
                history["generation"].append(gen)
                for key in history:
                    if key != "generation" and key in metrics:
                        history[key].append(metrics[key])
        
        runtime = time.time() - start_time
        
        return QDResult(
            algorithm="MMR-Elites",
            seed=seed,
            runtime=runtime,
            final_metrics=compute_all_metrics(
                self.fitness, self.descriptors, budget_k=self.archive_size
            ),
            history=history,
            final_genomes=self.archive,
            final_fitness=self.fitness,
            final_descriptors=self.descriptors,
        )
    
    def get_archive_size(self) -> int:
        return len(self.fitness) if self.fitness is not None else 0


def run_mmr_elites(
    task,
    archive_size: int,
    generations: int,
    batch_size: int,
    lambda_val: float,
    mutation_sigma: float,
    seed: int,
    log_interval: int = 100,
) -> Dict:
    """Functional interface for MMR-Elites."""
    alg = MMRElites(archive_size, lambda_val)
    result = alg.run(task, generations, batch_size, mutation_sigma, seed, log_interval)
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
