"""
MMR-Elites: Maximum Marginal Relevance Selection of Elites.

Core algorithm implementation.
"""

from typing import Dict, Tuple

import numpy as np

from .base import ExperimentConfig, QDAlgorithm, QDResult

try:
    from mmr_elites import mmr_elites_rs

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
        - Exact greedy MMR selection via a lazy priority queue
          (empirically fast: candidates are re-scored only 2-5 times on
          average; worst case O(N*K) distance evaluations)
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
            config.archive_size, config.lambda_val
        )
        self.n_dof = None

    def initialize(self, task, seed: int):
        """Initialize archive by MMR-selecting K from an oversampled pool."""
        np.random.seed(seed)

        self.n_dof = getattr(task, "n_dof", getattr(task, "n_dim", 20))
        lo, hi = getattr(task, "genome_bounds", (-np.pi, np.pi))

        # Oversample so the initial MMR selection actually filters
        # (selecting K from a pool of exactly K would be a no-op).
        init_size = self.config.archive_size + self.config.batch_size
        pool = np.random.uniform(lo, hi, (init_size, self.n_dof))
        pool_fit, pool_desc = task.evaluate(pool)

        idx = self.selector.select(pool_fit, pool_desc)
        self.archive = pool[idx]
        self.fitness = pool_fit[idx]
        self.descriptors = pool_desc[idx]

    def step(self, task) -> Dict[str, float]:
        """Perform one generation."""
        # Mutation
        parent_idx = np.random.randint(0, len(self.archive), self.config.batch_size)
        parents = self.archive[parent_idx]
        offspring = parents + np.random.normal(
            0, self.config.mutation_sigma, (self.config.batch_size, self.n_dof)
        )
        lo, hi = getattr(task, "genome_bounds", (-np.pi, np.pi))
        offspring = np.clip(offspring, lo, hi)

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

        # Metrics are computed by QDAlgorithm.run on logging generations
        return {}

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
