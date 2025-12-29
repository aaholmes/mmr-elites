"""
CVT-MAP-Elites: Centroidal Voronoi Tessellation MAP-Elites

This is the PRIMARY COMPETITOR baseline for MUSE-QD.

Reference: 
    Vassiliades, Chatzilygeroudis & Mouret (2016)
    "Using Centroidal Voronoi Tessellations to Scale Up the 
    Multi-dimensional Archive of Phenotypic Elites Algorithm"

CVT-MAP-Elites addresses the curse of dimensionality by using
K pre-computed centroids instead of a grid. Each solution is
assigned to its nearest centroid.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional, Dict, List
import time


class CVTArchive:
    """
    Archive based on Centroidal Voronoi Tessellation.
    
    Instead of a grid (which scales exponentially with dimension),
    CVT uses K pre-computed centroids. Each individual is assigned
    to its nearest centroid, creating K "niches".
    
    This gives CVT-MAP-Elites a fixed memory footprint of K,
    similar to MUSE-QD.
    """
    
    def __init__(
        self, 
        n_niches: int,
        descriptor_dim: int,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        seed: int = 42,
        cvt_samples: int = 100000
    ):
        """
        Initialize CVT archive.
        
        Args:
            n_niches: Number of centroids/niches (K)
            descriptor_dim: Dimensionality of behavior descriptors
            bounds_min: Lower bounds of descriptor space
            bounds_max: Upper bounds of descriptor space
            seed: Random seed for centroid computation
            cvt_samples: Number of samples for k-means initialization
        """
        self.n_niches = n_niches
        self.descriptor_dim = descriptor_dim
        self.bounds_min = np.asarray(bounds_min)
        self.bounds_max = np.asarray(bounds_max)
        
        # Compute CVT centroids using k-means
        print(f"  Computing {n_niches} CVT centroids in {descriptor_dim}D space...", end=" ")
        self.centroids = self._compute_cvt_centroids(seed, cvt_samples)
        print("Done.")
        
        # Build KD-tree for fast nearest-neighbor lookup
        self.tree = cKDTree(self.centroids)
        
        # Archive storage: niche_index -> (genome, fitness, descriptor)
        self.archive: Dict[int, Tuple[np.ndarray, float, np.ndarray]] = {}
    
    def _compute_cvt_centroids(self, seed: int, n_samples: int) -> np.ndarray:
        """
        Compute CVT centroids using Lloyd's algorithm (k-means).
        
        This is the standard approach from the CVT-MAP-Elites paper.
        """
        rng = np.random.default_rng(seed)
        
        # Sample uniform points in the descriptor space
        samples = rng.uniform(
            self.bounds_min, 
            self.bounds_max, 
            size=(n_samples, self.descriptor_dim)
        )
        
        # Use k-means to find centroids
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(
                n_clusters=self.n_niches, 
                random_state=seed, 
                n_init=1,
                max_iter=100
            )
            kmeans.fit(samples)
            return kmeans.cluster_centers_
        except ImportError:
            # Fallback: simple k-means implementation
            return self._simple_kmeans(samples, seed)
    
    def _simple_kmeans(self, samples: np.ndarray, seed: int, max_iter: int = 50) -> np.ndarray:
        """Simple k-means fallback if sklearn not available."""
        rng = np.random.default_rng(seed)
        
        # Initialize centroids randomly from samples
        indices = rng.choice(len(samples), self.n_niches, replace=False)
        centroids = samples[indices].copy()
        
        for _ in range(max_iter):
            # Assign samples to nearest centroid
            tree = cKDTree(centroids)
            _, assignments = tree.query(samples)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(self.n_niches)
            
            for i, c in enumerate(assignments):
                new_centroids[c] += samples[i]
                counts[c] += 1
            
            # Avoid division by zero
            counts = np.maximum(counts, 1)
            new_centroids /= counts[:, np.newaxis]
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        return centroids
    
    def get_niche(self, descriptor: np.ndarray) -> int:
        """Find the nearest centroid (niche) for a descriptor."""
        _, idx = self.tree.query(descriptor)
        return int(idx)
    
    def add(self, genome: np.ndarray, fitness: float, descriptor: np.ndarray) -> bool:
        """
        Try to add an individual to the archive.
        
        Returns True if added/replaced an existing elite.
        """
        niche = self.get_niche(descriptor)
        
        if niche not in self.archive or fitness > self.archive[niche][1]:
            self.archive[niche] = (genome.copy(), fitness, descriptor.copy())
            return True
        return False
    
    def sample_parents(self, n: int) -> Optional[np.ndarray]:
        """Sample n parents uniformly from archive."""
        if not self.archive:
            return None
        
        keys = list(self.archive.keys())
        indices = np.random.randint(0, len(keys), n)
        return np.array([self.archive[keys[i]][0] for i in indices])
    
    def get_all_genomes(self) -> np.ndarray:
        """Get all genomes in the archive."""
        if not self.archive:
            return np.array([])
        return np.array([v[0] for v in self.archive.values()])
    
    def get_all_fitness(self) -> np.ndarray:
        """Get all fitness values in the archive."""
        if not self.archive:
            return np.array([])
        return np.array([v[1] for v in self.archive.values()])
    
    def get_all_descriptors(self) -> np.ndarray:
        """Get all descriptors in the archive."""
        if not self.archive:
            return np.array([])
        return np.array([v[2] for v in self.archive.values()])
    
    def coverage(self) -> float:
        """Fraction of niches filled."""
        return len(self.archive) / self.n_niches
    
    def __len__(self):
        return len(self.archive)


def run_cvt_map_elites(
    task,
    n_niches: int,
    generations: int,
    batch_size: int,
    mutation_sigma: float,
    seed: int,
    log_interval: int = 100,
    descriptor_dim: int = 20,
    bounds_min: Optional[np.ndarray] = None,
    bounds_max: Optional[np.ndarray] = None,
) -> Dict:
    """
    Run CVT-MAP-Elites algorithm.
    
    Args:
        task: Task object with evaluate(genomes) method
        n_niches: Number of CVT centroids (archive capacity)
        generations: Number of generations
        batch_size: Offspring per generation
        mutation_sigma: Gaussian mutation std
        seed: Random seed
        log_interval: How often to log metrics
        descriptor_dim: Dimension of behavior descriptors
        bounds_min: Lower bounds (default: zeros)
        bounds_max: Upper bounds (default: ones)
    
    Returns:
        Dictionary with results and history
    """
    np.random.seed(seed)
    
    # Default bounds for normalized descriptors
    if bounds_min is None:
        bounds_min = np.zeros(descriptor_dim)
    if bounds_max is None:
        bounds_max = np.ones(descriptor_dim)
    
    # Initialize archive
    archive = CVTArchive(
        n_niches=n_niches,
        descriptor_dim=descriptor_dim,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        seed=seed
    )
    
    # Initial population
    init_pop = np.random.uniform(-np.pi, np.pi, (batch_size * 5, 20))
    fit, desc = task.evaluate(init_pop)
    
    for i in range(len(init_pop)):
        archive.add(init_pop[i], fit[i], desc[i])
    
    # History tracking
    history = {
        "generation": [],
        "qd_score": [],
        "max_fitness": [],
        "mean_fitness": [],
        "mean_pairwise_distance": [],
        "archive_size": [],
        "coverage": [],
    }
    
    start_time = time.time()
    
    for gen in range(1, generations + 1):
        # Sample parents from archive
        parents = archive.sample_parents(batch_size)
        if parents is None:
            continue
        
        # Gaussian mutation
        offspring = parents + np.random.normal(0, mutation_sigma, parents.shape)
        offspring = np.clip(offspring, -np.pi, np.pi)
        
        # Evaluate offspring
        off_fit, off_desc = task.evaluate(offspring)
        
        # Add to archive
        for i in range(len(offspring)):
            archive.add(offspring[i], off_fit[i], off_desc[i])
        
        # Logging
        if gen % log_interval == 0:
            all_fit = archive.get_all_fitness()
            all_desc = archive.get_all_descriptors()
            
            if len(all_fit) > 0:
                # Import metrics
                try:
                    from metrics import mean_pairwise_distance
                    mpd = mean_pairwise_distance(all_desc)
                except ImportError:
                    from scipy.spatial.distance import cdist
                    if len(all_desc) > 1:
                        dists = cdist(all_desc, all_desc)
                        mpd = float(np.mean(dists[np.triu_indices(len(all_desc), k=1)]))
                    else:
                        mpd = 0.0
                
                history["generation"].append(gen)
                history["qd_score"].append(float(np.sum(all_fit)))
                history["max_fitness"].append(float(np.max(all_fit)))
                history["mean_fitness"].append(float(np.mean(all_fit)))
                history["mean_pairwise_distance"].append(mpd)
                history["archive_size"].append(len(archive))
                history["coverage"].append(archive.coverage())
    
    runtime = time.time() - start_time
    
    # Final metrics
    all_fit = archive.get_all_fitness()
    all_desc = archive.get_all_descriptors()
    
    try:
        from metrics import mean_pairwise_distance
        mpd = mean_pairwise_distance(all_desc)
    except ImportError:
        from scipy.spatial.distance import cdist
        if len(all_desc) > 1:
            dists = cdist(all_desc, all_desc)
            mpd = float(np.mean(dists[np.triu_indices(len(all_desc), k=1)]))
        else:
            mpd = 0.0
    
    final_metrics = {
        "qd_score": float(np.sum(all_fit)) if len(all_fit) > 0 else 0.0,
        "max_fitness": float(np.max(all_fit)) if len(all_fit) > 0 else 0.0,
        "mean_fitness": float(np.mean(all_fit)) if len(all_fit) > 0 else 0.0,
        "mean_pairwise_distance": mpd,
        "archive_size": len(archive),
        "coverage": archive.coverage(),
    }
    
    return {
        "algorithm": "CVT-MAP-Elites",
        "seed": seed,
        "runtime": runtime,
        "final_metrics": final_metrics,
        "history": history,
        "final_genomes": archive.get_all_genomes(),
        "final_descriptors": all_desc,
        "final_fitness": all_fit,
    }
