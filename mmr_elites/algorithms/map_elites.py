"""
MAP-Elites: Multi-dimensional Archive of Phenotypic Elites.

Baseline algorithm for comparison.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import ExperimentConfig, QDAlgorithm, QDResult


class MAPElites(QDAlgorithm):
    """
    MAP-Elites with sparse grid storage.

    Uses dictionary for archive to handle high-dimensional behavior spaces
    where a full grid would be infeasible (e.g., 3^20 = 3.5B cells).
    """

    def __init__(self, config: ExperimentConfig, bins_per_dim: int = 3):
        super().__init__(config)
        self.bins_per_dim = bins_per_dim
        self.archive: Dict[tuple, Tuple[np.ndarray, float, np.ndarray]] = {}
        self.n_dof = None

    def _get_cell(self, descriptor: np.ndarray) -> tuple:
        """Map descriptor to grid cell."""
        # Descriptor should be in [0, 1]
        idx = (descriptor * self.bins_per_dim).astype(int)
        idx = np.clip(idx, 0, self.bins_per_dim - 1)
        return tuple(idx)

    def _add_to_archive(
        self, genome: np.ndarray, fitness: float, descriptor: np.ndarray
    ) -> bool:
        """Try to add solution to archive. Returns True if added."""
        cell = self._get_cell(descriptor)
        if cell not in self.archive or fitness > self.archive[cell][1]:
            self.archive[cell] = (genome.copy(), fitness, descriptor.copy())
            return True
        return False

    def _sample_parents(self, n: int) -> np.ndarray:
        """Sample n parents uniformly from archive."""
        keys = list(self.archive.keys())
        indices = np.random.randint(0, len(keys), n)
        return np.array([self.archive[keys[i]][0] for i in indices])

    def initialize(self, task, seed: int):
        """Initialize archive with random solutions."""
        np.random.seed(seed)

        self.n_dof = getattr(task, "n_dof", getattr(task, "n_dim", 20))
        self.archive = {}

        # Initialize with larger population
        init_size = self.config.batch_size * 5
        init_pop = np.random.uniform(-np.pi, np.pi, (init_size, self.n_dof))
        fitness, descriptors = task.evaluate(init_pop)

        for i in range(init_size):
            self._add_to_archive(init_pop[i], fitness[i], descriptors[i])

    def step(self, task) -> Dict[str, float]:
        """Perform one generation."""
        if not self.archive:
            return {}

        # Sample and mutate
        parents = self._sample_parents(self.config.batch_size)
        offspring = parents + np.random.normal(
            0, self.config.mutation_sigma, parents.shape
        )
        offspring = np.clip(offspring, -np.pi, np.pi)

        # Evaluate and add to archive
        fitness, descriptors = task.evaluate(offspring)
        for i in range(len(offspring)):
            self._add_to_archive(offspring[i], fitness[i], descriptors[i])

        # Compute metrics
        all_fit, all_desc = self._get_archive_arrays()
        from mmr_elites.metrics.qd_metrics import compute_all_metrics

        return compute_all_metrics(all_fit, all_desc, self.config.archive_size)

    def _get_archive_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract fitness and descriptors as arrays."""
        if not self.archive:
            return np.array([]), np.array([])
        fitness = np.array([v[1] for v in self.archive.values()])
        descriptors = np.array([v[2] for v in self.archive.values()])
        return fitness, descriptors

    def get_archive(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return current archive state."""
        if not self.archive:
            return np.array([]), np.array([]), np.array([])
        genomes = np.array([v[0] for v in self.archive.values()])
        fitness = np.array([v[1] for v in self.archive.values()])
        descriptors = np.array([v[2] for v in self.archive.values()])
        return genomes, fitness, descriptors


def run_map_elites(
    task,
    generations: int = 1000,
    batch_size: int = 200,
    bins_per_dim: int = 3,
    mutation_sigma: float = 0.1,
    seed: int = 42,
    log_interval: int = 100,
) -> Dict:
    """
    Functional interface for MAP-Elites.

    Args:
        task: Task object with evaluate(genomes) method
        generations: Number of generations
        batch_size: Offspring per generation
        bins_per_dim: Grid resolution per dimension
        mutation_sigma: Gaussian mutation std
        seed: Random seed
        log_interval: How often to log metrics

    Returns:
        Dictionary with results and history
    """
    config = ExperimentConfig(
        generations=generations,
        batch_size=batch_size,
        bins_per_dim=bins_per_dim,
        mutation_sigma=mutation_sigma,
        log_interval=log_interval,
    )

    alg = MAPElites(config, bins_per_dim=bins_per_dim)
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
