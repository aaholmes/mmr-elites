"""
CVT-MAP-Elites: Centroidal Voronoi Tessellation MAP-Elites.

Uses k-means clustering to create fixed-size archive with good coverage.
Main baseline for comparison with MMR-Elites.
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .base import ExperimentConfig, QDAlgorithm, QDResult

# Check for sklearn
try:
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class CVTMAPElites(QDAlgorithm):
    """
    CVT-MAP-Elites algorithm.

    Properties:
        - Fixed archive size (n_niches centroids)
        - Uses k-means for centroid placement
        - Better coverage than grid-based MAP-Elites
    """

    def __init__(self, config: ExperimentConfig, n_niches: Optional[int] = None):
        super().__init__(config)
        self.n_niches = n_niches or config.archive_size
        self.centroids = None
        self.tree = None
        self.archive: Dict[int, Tuple[np.ndarray, float, np.ndarray]] = {}
        self.n_dof = None
        self.desc_dim = None

    def _compute_centroids(self, desc_dim: int, seed: int) -> np.ndarray:
        """Compute CVT centroids using k-means."""
        np.random.seed(seed)

        # Sample points in descriptor space [0, 1]^D
        n_samples = min(50000, self.n_niches * 100)
        samples = np.random.uniform(0, 1, (n_samples, desc_dim))

        if SKLEARN_AVAILABLE:
            kmeans = KMeans(
                n_clusters=self.n_niches, random_state=seed, n_init=1, max_iter=100
            )
            kmeans.fit(samples)
            return kmeans.cluster_centers_
        else:
            # Fallback: random selection (worse but works)
            print("⚠️ sklearn not available, using random centroids")
            idx = np.random.choice(len(samples), self.n_niches, replace=False)
            return samples[idx]

    def _get_niche(self, descriptor: np.ndarray) -> int:
        """Find nearest centroid for descriptor."""
        _, idx = self.tree.query(descriptor)
        return int(idx)

    def _add_to_archive(
        self, genome: np.ndarray, fitness: float, descriptor: np.ndarray
    ) -> bool:
        """Try to add solution to archive. Returns True if added."""
        niche = self._get_niche(descriptor)
        if niche not in self.archive or fitness > self.archive[niche][1]:
            self.archive[niche] = (genome.copy(), fitness, descriptor.copy())
            return True
        return False

    def initialize(self, task, seed: int):
        """Initialize archive with CVT centroids."""
        np.random.seed(seed)

        self.n_dof = getattr(task, "n_dof", getattr(task, "n_dim", 20))
        self.desc_dim = getattr(task, "desc_dim", self.n_dof)

        # Compute centroids
        self.centroids = self._compute_centroids(self.desc_dim, seed)
        self.tree = cKDTree(self.centroids)
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

        # Sample parents
        keys = list(self.archive.keys())
        parent_indices = np.random.randint(0, len(keys), self.config.batch_size)
        parents = np.array([self.archive[keys[i]][0] for i in parent_indices])

        # Mutate
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

    def coverage(self) -> float:
        """Return fraction of niches filled."""
        return len(self.archive) / self.n_niches


def run_cvt_map_elites(
    task,
    n_niches: int = 1000,
    generations: int = 1000,
    batch_size: int = 200,
    mutation_sigma: float = 0.1,
    seed: int = 42,
    log_interval: int = 100,
) -> Dict:
    """
    Functional interface for CVT-MAP-Elites.

    Args:
        task: Task object with evaluate(genomes) method
        n_niches: Number of CVT centroids (archive capacity)
        generations: Number of generations
        batch_size: Offspring per generation
        mutation_sigma: Gaussian mutation std
        seed: Random seed
        log_interval: How often to log metrics

    Returns:
        Dictionary with results and history
    """
    config = ExperimentConfig(
        archive_size=n_niches,
        generations=generations,
        batch_size=batch_size,
        mutation_sigma=mutation_sigma,
        log_interval=log_interval,
    )

    alg = CVTMAPElites(config, n_niches=n_niches)
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
        "coverage": len(alg.archive) / alg.n_niches,
    }
