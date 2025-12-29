# experiments/mmr_custom_distance.py
"""
MMR-Elites with custom distance functions.

This is a Python implementation for experimenting with different
distance functions. For production, integrate into Rust backend.
"""

import numpy as np
from typing import Callable, Tuple
import heapq


class MMRSelectorCustomDistance:
    """
    MMR selector with pluggable distance function.
    
    Slower than Rust but allows experimenting with different distances.
    """
    
    def __init__(
        self, 
        target_k: int, 
        lambda_val: float,
        distance_fn: Callable[[np.ndarray, np.ndarray], float],
    ):
        self.target_k = target_k
        self.lambda_val = lambda_val
        self.distance_fn = distance_fn
    
    def select(
        self, 
        fitness: np.ndarray, 
        descriptors: np.ndarray
    ) -> np.ndarray:
        """
        Select K solutions using MMR with custom distance.
        
        Args:
            fitness: Fitness values (N,)
            descriptors: Behavior descriptors (N, D)
        
        Returns:
            Indices of selected solutions (K,)
        """
        n = len(fitness)
        if n <= self.target_k:
            return np.arange(n)
        
        # Normalize fitness to [0, 1]
        f_min, f_max = fitness.min(), fitness.max()
        if f_max - f_min > 1e-10:
            f_norm = (fitness - f_min) / (f_max - f_min)
        else:
            f_norm = np.ones(n) * 0.5
        
        selected = []
        remaining = set(range(n))
        
        # Seed with best fitness
        best_idx = int(np.argmax(fitness))
        selected.append(best_idx)
        remaining.remove(best_idx)
        
        # Cache distances to selected set
        # d_min[i] = min distance from i to any selected
        d_min = np.full(n, np.inf)
        
        # Initialize d_min to distance from seed
        seed_desc = descriptors[best_idx]
        for i in remaining:
            d_min[i] = self.distance_fn(descriptors[i], seed_desc)
        
        # Greedy selection
        while len(selected) < self.target_k and remaining:
            # Find best candidate
            best_score = -np.inf
            best_idx = None
            
            for i in remaining:
                score = (1 - self.lambda_val) * f_norm[i] + self.lambda_val * d_min[i]
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx is None:
                break
            
            # Add to selected
            selected.append(best_idx)
            remaining.remove(best_idx)
            
            # Update d_min for remaining candidates
            new_desc = descriptors[best_idx]
            for i in remaining:
                d_new = self.distance_fn(descriptors[i], new_desc)
                d_min[i] = min(d_min[i], d_new)
        
        return np.array(selected)


def run_mmr_elites_custom_distance(
    task,
    distance_fn: Callable,
    archive_size: int = 1000,
    generations: int = 1000,
    batch_size: int = 200,
    lambda_val: float = 0.5,
    mutation_sigma: float = 0.1,
    seed: int = 42,
    log_interval: int = 100,
) -> dict:
    """
    Run MMR-Elites with custom distance function.
    """
    import time
    from mmr_elites.metrics.qd_metrics import compute_all_metrics
    
    np.random.seed(seed)
    
    selector = MMRSelectorCustomDistance(archive_size, lambda_val, distance_fn)
    
    # Get task dimensions
    n_dof = getattr(task, 'n_dof', getattr(task, 'n_dim', 20))
    
    # Initialize
    archive = np.random.uniform(-np.pi, np.pi, (archive_size, n_dof))
    fit, desc = task.evaluate(archive)
    
    idx = selector.select(fit, desc)
    archive, fit, desc = archive[idx], fit[idx], desc[idx]
    
    history = {
        "generation": [],
        "qd_score": [],
        "qd_score_at_budget": [],
        "max_fitness": [],
        "mean_fitness": [],
        "mean_pairwise_distance": [],
        "uniformity_cv": [],
    }
    
    start_time = time.time()
    
    for gen in range(1, generations + 1):
        # Mutation
        parents_idx = np.random.randint(0, len(archive), batch_size)
        parents = archive[parents_idx]
        offspring = np.clip(
            parents + np.random.normal(0, mutation_sigma, (batch_size, n_dof)),
            -np.pi, np.pi
        )
        
        off_fit, off_desc = task.evaluate(offspring)
        
        # Pool and select
        pool = np.vstack([archive, offspring])
        pool_fit = np.concatenate([fit, off_fit])
        pool_desc = np.vstack([desc, off_desc])
        
        idx = selector.select(pool_fit, pool_desc)
        archive, fit, desc = pool[idx], pool_fit[idx], pool_desc[idx]
        
        if gen % log_interval == 0:
            metrics = compute_all_metrics(fit, desc, archive_size)
            history["generation"].append(gen)
            for k in metrics:
                if k in history:
                    history[k].append(metrics[k])
    
    runtime = time.time() - start_time
    final_metrics = compute_all_metrics(fit, desc, archive_size)
    
    return {
        "algorithm": f"MMR-Elites ({distance_fn.__name__})",
        "seed": seed,
        "runtime": runtime,
        "final_metrics": final_metrics,
        "history": history,
        "final_genomes": archive,
        "final_fitness": fit,
        "final_descriptors": desc,
    }
