"""
MMR-Elites with pluggable distance functions.

This Python implementation allows experimenting with different distance
functions before potentially integrating the best into Rust.
"""

import time
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from experiments.distance_functions import (
    auto_sigma,
    auto_sigma_diagonal,
    euclidean_distance,
    exponential_saturating,
    gaussian_saturating,
    normalized_euclidean,
)


class MMRSelectorCustomDistance:
    """
    MMR selector with pluggable distance function.

    Implements the core MMR selection:
        score(x) = (1 - λ) * fitness(x) + λ * d_min(x, Archive)

    where d_min uses the provided distance function.
    """

    def __init__(
        self,
        target_k: int,
        lambda_val: float,
        distance_fn: Callable[[np.ndarray, np.ndarray], float],
    ):
        """
        Args:
            target_k: Archive size
            lambda_val: Diversity weight λ ∈ [0, 1]
            distance_fn: Function (b1, b2) -> distance in [0, 1]
        """
        self.target_k = target_k
        self.lambda_val = lambda_val
        self.distance_fn = distance_fn

    def select(self, fitness: np.ndarray, descriptors: np.ndarray) -> np.ndarray:
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

        # Normalize fitness to [0, 1] for fair combination with distance
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

        # Cache d_min for each candidate
        # d_min[i] = min distance from i to any selected solution
        d_min = np.full(n, np.inf)

        # Initialize d_min to distance from seed
        seed_desc = descriptors[best_idx]
        for i in remaining:
            d_min[i] = self.distance_fn(descriptors[i], seed_desc)

        # Greedy selection loop
        while len(selected) < self.target_k and remaining:
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


def run_mmr_elites_custom(
    task,
    distance_fn: Callable,
    distance_name: str = "custom",
    archive_size: int = 1000,
    generations: int = 1000,
    batch_size: int = 200,
    lambda_val: float = 0.5,
    mutation_sigma: float = 0.1,
    seed: int = 42,
    log_interval: int = 100,
) -> Dict:
    """
    Run MMR-Elites with custom distance function.

    Args:
        task: Task with evaluate(genomes) method
        distance_fn: Distance function (b1, b2) -> float
        distance_name: Name for logging
        archive_size: Archive size K
        generations: Number of generations
        batch_size: Offspring per generation
        lambda_val: Diversity weight
        mutation_sigma: Mutation std
        seed: Random seed
        log_interval: Logging frequency

    Returns:
        Results dictionary
    """
    np.random.seed(seed)

    selector = MMRSelectorCustomDistance(archive_size, lambda_val, distance_fn)

    # Get dimensions from task
    n_dof = getattr(task, "n_dof", getattr(task, "n_dim", 20))

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
            -np.pi,
            np.pi,
        )

        off_fit, off_desc = task.evaluate(offspring)

        # Pool and select
        pool = np.vstack([archive, offspring])
        pool_fit = np.concatenate([fit, off_fit])
        pool_desc = np.vstack([desc, off_desc])

        idx = selector.select(pool_fit, pool_desc)
        archive, fit, desc = pool[idx], pool_fit[idx], pool_desc[idx]

        # Logging
        if gen % log_interval == 0:
            from mmr_elites.metrics.qd_metrics import compute_all_metrics

            metrics = compute_all_metrics(fit, desc, archive_size)
            history["generation"].append(gen)
            for k in metrics:
                if k in history:
                    history[k].append(metrics[k])

    runtime = time.time() - start_time

    from mmr_elites.metrics.qd_metrics import compute_all_metrics

    final_metrics = compute_all_metrics(fit, desc, archive_size)

    return {
        "algorithm": f"MMR-Elites ({distance_name})",
        "distance": distance_name,
        "seed": seed,
        "runtime": runtime,
        "final_metrics": final_metrics,
        "history": history,
        "final_genomes": archive,
        "final_fitness": fit,
        "final_descriptors": desc,
    }
