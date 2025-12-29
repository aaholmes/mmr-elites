"""
MAP-Elites: Multi-dimensional Archive of Phenotypic Elites.

Standard grid-based QD algorithm.
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional, List, Union
from .base import QDAlgorithm, QDResult
from ..metrics.qd_metrics import compute_all_metrics


class MAPElites(QDAlgorithm):
    """
    Standard MAP-Elites with grid-based archive.
    """
    
    def __init__(
        self, 
        bins_per_dim: int, 
        descriptor_bounds: List[Tuple[float, float]]
    ):
        self.bins_per_dim = bins_per_dim
        self.descriptor_bounds = descriptor_bounds
        self.n_dims = len(descriptor_bounds)
        self.archive: Dict[Tuple[int, ...], Tuple[np.ndarray, float, np.ndarray]] = {}
        
    def _get_cell(self, descriptor: np.ndarray) -> Tuple[int, ...]:
        """Map descriptor to grid cell."""
        indices = []
        for i in range(self.n_dims):
            min_val, max_val = self.descriptor_bounds[i]
            # Normalize to [0, 1]
            val = (descriptor[i] - min_val) / (max_val - min_val)
            # Clip and map to bin
            idx = int(np.clip(val * self.bins_per_dim, 0, self.bins_per_dim - 1))
            indices.append(idx)
        return tuple(indices)
    
    def _add_to_archive(self, genome: np.ndarray, fitness: float, descriptor: np.ndarray) -> bool:
        cell = self._get_cell(descriptor)
        
        if cell not in self.archive or fitness > self.archive[cell][1]:
            self.archive[cell] = (genome.copy(), fitness, descriptor.copy())
            return True
        return False
        
    def _sample_parents(self, n: int) -> Optional[np.ndarray]:
        if not self.archive:
            return None
        keys = list(self.archive.keys())
        indices = np.random.randint(0, len(keys), n)
        return np.array([self.archive[keys[i]][0] for i in indices])

    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.archive:
            return np.array([]), np.array([]), np.array([])
        
        keys = sorted(list(self.archive.keys()))
        genomes = np.array([self.archive[k][0] for k in keys])
        fitness = np.array([self.archive[k][1] for k in keys])
        descriptors = np.array([self.archive[k][2] for k in keys])
        return genomes, fitness, descriptors

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
        
        if hasattr(task, 'genome_dim'):
            n_dof = task.genome_dim
        elif hasattr(task, 'n_dof'):
            n_dof = task.n_dof
        else:
            n_dof = 20
            
        # Initial population
        init_pop = np.random.uniform(-np.pi, np.pi, (batch_size * 2, n_dof))
        fit, desc = task.evaluate(init_pop)
        
        for i in range(len(init_pop)):
            self._add_to_archive(init_pop[i], fit[i], desc[i])
            
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
        
        # Max capacity for coverage calculation
        max_capacity = self.bins_per_dim ** self.n_dims
        
        for gen in range(1, generations + 1):
            parents = self._sample_parents(batch_size)
            if parents is None:
                continue
                
            offspring = parents + np.random.normal(0, mutation_sigma, parents.shape)
            offspring = np.clip(offspring, -np.pi, np.pi)
            
            off_fit, off_desc = task.evaluate(offspring)
            
            for i in range(len(offspring)):
                self._add_to_archive(offspring[i], off_fit[i], off_desc[i])
                
            if gen % log_interval == 0:
                genomes, fitness, descriptors = self.get_all_data()
                # Use standard metrics
                # Note: coverage in metrics.py is calculated differently than simple cell count for MAP-Elites
                # but we use the shared one for consistency
                metrics = compute_all_metrics(fitness, descriptors, budget_k=1000) # Arbitrary budget for comparison
                
                history["generation"].append(gen)
                for key in history:
                    if key != "generation" and key in metrics:
                        history[key].append(metrics[key])
        
        runtime = time.time() - start_time
        genomes, fitness, descriptors = self.get_all_data()
        
        return QDResult(
            algorithm="MAP-Elites",
            seed=seed,
            runtime=runtime,
            final_metrics=compute_all_metrics(fitness, descriptors, budget_k=1000),
            history=history,
            final_genomes=genomes,
            final_fitness=fitness,
            final_descriptors=descriptors,
        )

    def get_archive_size(self) -> int:
        return len(self.archive)

def run_map_elites(
    task,
    generations: int,
    batch_size: int,
    bins_per_dim: int,
    mutation_sigma: float,
    seed: int,
    log_interval: int = 100,
    descriptor_bounds: Optional[List[Tuple[float, float]]] = None
) -> Dict:
    # Infer bounds from task if not provided
    if descriptor_bounds is None:
        # Default to [0, 1] for typical tasks
        # If task is 20D arm, descriptor might be 20D or 2D
        # We need to know descriptor dim
        if hasattr(task, 'desc_dim'):
            dim = task.desc_dim
        else:
            dim = 20 # Guess
        descriptor_bounds = [(0.0, 1.0) for _ in range(dim)]
        
    alg = MAPElites(bins_per_dim, descriptor_bounds)
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
