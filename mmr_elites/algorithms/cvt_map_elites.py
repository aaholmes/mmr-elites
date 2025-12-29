"""
CVT-MAP-Elites: Centroidal Voronoi Tessellation MAP-Elites
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional, List
from scipy.spatial import cKDTree
from .base import QDAlgorithm, QDResult
from ..metrics.qd_metrics import compute_all_metrics


class CVTArchive:
    """
    Archive based on Centroidal Voronoi Tessellation.
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
        self.n_niches = n_niches
        self.descriptor_dim = descriptor_dim
        self.bounds_min = np.asarray(bounds_min)
        self.bounds_max = np.asarray(bounds_max)
        
        self.centroids = self._compute_cvt_centroids(seed, cvt_samples)
        self.tree = cKDTree(self.centroids)
        self.archive: Dict[int, Tuple[np.ndarray, float, np.ndarray]] = {}
    
    def _compute_cvt_centroids(self, seed: int, n_samples: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        samples = rng.uniform(
            self.bounds_min, 
            self.bounds_max, 
            size=(n_samples, self.descriptor_dim)
        )
        
        try:
            from sklearn.cluster import KMeans
            # Suppress sklearn warnings if possible, or just let them be
            kmeans = KMeans(
                n_clusters=self.n_niches, 
                random_state=seed, 
                n_init=1,
                max_iter=20
            )
            kmeans.fit(samples)
            return kmeans.cluster_centers_
        except ImportError:
            return self._simple_kmeans(samples, seed)
    
    def _simple_kmeans(self, samples: np.ndarray, seed: int, max_iter: int = 50) -> np.ndarray:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(samples), self.n_niches, replace=False)
        centroids = samples[indices].copy()
        
        for _ in range(max_iter):
            tree = cKDTree(centroids)
            _, assignments = tree.query(samples)
            
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(self.n_niches)
            
            for i, c in enumerate(assignments):
                new_centroids[c] += samples[i]
                counts[c] += 1
            
            counts = np.maximum(counts, 1)
            new_centroids /= counts[:, np.newaxis]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        return centroids
    
    def get_niche(self, descriptor: np.ndarray) -> int:
        _, idx = self.tree.query(descriptor)
        return int(idx)
    
    def add(self, genome: np.ndarray, fitness: float, descriptor: np.ndarray) -> bool:
        niche = self.get_niche(descriptor)
        if niche not in self.archive or fitness > self.archive[niche][1]:
            self.archive[niche] = (genome.copy(), fitness, descriptor.copy())
            return True
        return False
    
    def sample_parents(self, n: int) -> Optional[np.ndarray]:
        if not self.archive:
            return None
        keys = list(self.archive.keys())
        indices = np.random.randint(0, len(keys), n)
        return np.array([self.archive[keys[i]][0] for i in indices])

    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.archive:
            return np.array([]), np.array([]), np.array([])
        
        # Ensure consistent ordering
        keys = sorted(list(self.archive.keys()))
        genomes = np.array([self.archive[k][0] for k in keys])
        fitness = np.array([self.archive[k][1] for k in keys])
        descriptors = np.array([self.archive[k][2] for k in keys])
        return genomes, fitness, descriptors

    def __len__(self):
        return len(self.archive)


class CVTMapElites(QDAlgorithm):
    """CVT-MAP-Elites wrapper."""
    
    def __init__(
        self, 
        n_niches: int, 
        descriptor_dim: int = 20,
        bounds_min: Optional[np.ndarray] = None, 
        bounds_max: Optional[np.ndarray] = None
    ):
        self.n_niches = n_niches
        self.descriptor_dim = descriptor_dim
        self.bounds_min = bounds_min if bounds_min is not None else np.zeros(descriptor_dim)
        self.bounds_max = bounds_max if bounds_max is not None else np.ones(descriptor_dim)
        self.archive_obj = None

    def run(
        self,
        task,
        generations: int,
        batch_size: int,
        mutation_sigma: float,
        seed: int,
        log_interval: int = 100,
    ) -> QDResult:
        np.random.seed(seed)
        
        # Initialize archive
        self.archive_obj = CVTArchive(
            n_niches=self.n_niches,
            descriptor_dim=self.descriptor_dim,
            bounds_min=self.bounds_min,
            bounds_max=self.bounds_max,
            seed=seed
        )
        
        if hasattr(task, 'genome_dim'):
            n_dof = task.genome_dim
        elif hasattr(task, 'n_dof'):
            n_dof = task.n_dof
        else:
            n_dof = 20 # Fallback
            
        # Initial population
        init_pop = np.random.uniform(-np.pi, np.pi, (batch_size * 2, n_dof))
        fit, desc = task.evaluate(init_pop)
        
        for i in range(len(init_pop)):
            self.archive_obj.add(init_pop[i], fit[i], desc[i])
            
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
            parents = self.archive_obj.sample_parents(batch_size)
            if parents is None:
                continue
                
            offspring = parents + np.random.normal(0, mutation_sigma, parents.shape)
            offspring = np.clip(offspring, -np.pi, np.pi)
            
            off_fit, off_desc = task.evaluate(offspring)
            
            for i in range(len(offspring)):
                self.archive_obj.add(offspring[i], off_fit[i], off_desc[i])
                
            if gen % log_interval == 0:
                genomes, fitness, descriptors = self.archive_obj.get_all_data()
                metrics = compute_all_metrics(fitness, descriptors, budget_k=self.n_niches)
                
                history["generation"].append(gen)
                for key in history:
                    if key != "generation" and key in metrics:
                        history[key].append(metrics[key])
        
        runtime = time.time() - start_time
        genomes, fitness, descriptors = self.archive_obj.get_all_data()
        
        return QDResult(
            algorithm="CVT-MAP-Elites",
            seed=seed,
            runtime=runtime,
            final_metrics=compute_all_metrics(fitness, descriptors, budget_k=self.n_niches),
            history=history,
            final_genomes=genomes,
            final_fitness=fitness,
            final_descriptors=descriptors,
        )

    def get_archive_size(self) -> int:
        return len(self.archive_obj) if self.archive_obj else 0

def run_cvt_map_elites(
    task,
    n_niches: int,
    generations: int,
    batch_size: int,
    mutation_sigma: float,
    seed: int,
    log_interval: int = 100,
    descriptor_dim: int = 20,
) -> Dict:
    alg = CVTMapElites(n_niches, descriptor_dim)
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
