"""
QD Metrics Module for MMR-Elites
================================

Implements standard Quality-Diversity metrics for scientific comparison.

References:
- Mouret & Clune (2015) "Illuminating search spaces by mapping elites"
- Pugh et al. (2016) "Quality Diversity: A New Frontier for Evolutionary Computation"
- Fontaine et al. (2020) "Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space"
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def qd_score(fitness: np.ndarray) -> float:
    """
    QD-Score: Sum of all fitness values in the archive.

    This measures the total "quality" accumulated across all discovered niches.
    Higher is better.

    Args:
        fitness: Array of fitness values (N,)

    Returns:
        Sum of fitness values
    """
    return float(np.sum(fitness))


def qd_score_at_budget(fitness: np.ndarray, budget_k: int) -> float:
    """
    Fair QD-Score: sum of top-K fitness values.

    Used to compare algorithms with different archive sizes (e.g. fixed K vs unbounded).

    Args:
        fitness: Array of fitness values (N,)
        budget_k: Maximum number of solutions to count

    Returns:
        Sum of top-K fitness values
    """
    if len(fitness) == 0:
        return 0.0

    if len(fitness) <= budget_k:
        return float(np.sum(fitness))

    # Sort and take top K
    return float(np.sum(np.sort(fitness)[-budget_k:]))


def archive_coverage(
    descriptors: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    grid_resolution: int = 100,
) -> float:
    """
    Coverage: Fraction of behavior space cells occupied.

    Discretizes the behavior space into a grid and counts unique cells.
    For high-dimensional spaces, this uses a hash-based approach.

    Args:
        descriptors: Behavior descriptors (N, D)
        bounds_min: Lower bounds of behavior space (D,)
        bounds_max: Upper bounds of behavior space (D,)
        grid_resolution: Number of bins per dimension

    Returns:
        Coverage ratio in [0, 1]
    """
    if len(descriptors) == 0:
        return 0.0

    n_samples, n_dims = descriptors.shape

    # Normalize to [0, 1]
    normalized = (descriptors - bounds_min) / (bounds_max - bounds_min + 1e-10)
    normalized = np.clip(normalized, 0, 0.9999)

    # Discretize
    indices = (normalized * grid_resolution).astype(int)

    # Count unique cells using tuple hashing
    unique_cells = set(map(tuple, indices))

    # Total possible cells (capped for high dimensions)
    if n_dims <= 6:
        total_cells = grid_resolution**n_dims
    else:
        # For high-D, report raw count (coverage ratio meaningless)
        return float(len(unique_cells))

    return len(unique_cells) / total_cells


def mean_pairwise_distance(descriptors: np.ndarray) -> float:
    """
    Mean Pairwise Distance: Average Euclidean distance between all pairs.

    Measures overall spread/diversity of the archive.
    Complexity: O(N^2 * D)

    Args:
        descriptors: Behavior descriptors (N, D)

    Returns:
        Mean pairwise distance
    """
    if len(descriptors) < 2:
        return 0.0

    # Use cdist for efficiency
    dists = cdist(descriptors, descriptors, metric="euclidean")

    # Extract upper triangle (excluding diagonal)
    n = len(descriptors)
    upper_tri = dists[np.triu_indices(n, k=1)]

    return float(np.mean(upper_tri))


def archive_uniformity(descriptors: np.ndarray, k: int = 5) -> float:
    """
    Uniformity: Measures how evenly spread the archive is.

    Computes the coefficient of variation (CV) of k-NN distances.
    Lower CV = more uniform distribution.

    Args:
        descriptors: Behavior descriptors (N, D)
        k: Number of nearest neighbors

    Returns:
        Coefficient of variation of k-NN distances (lower = more uniform)
    """
    if len(descriptors) <= k:
        return 0.0

    tree = cKDTree(descriptors)

    # Query k+1 neighbors (first is self)
    distances, _ = tree.query(descriptors, k=k + 1)

    # Average distance to k nearest neighbors (excluding self)
    mean_knn_dists = np.mean(distances[:, 1:], axis=1)

    # Coefficient of variation
    cv = np.std(mean_knn_dists) / (np.mean(mean_knn_dists) + 1e-10)

    return float(cv)


def max_fitness(fitness: np.ndarray) -> float:
    """Maximum fitness in the archive."""
    if len(fitness) == 0:
        return 0.0
    return float(np.max(fitness))


def mean_fitness(fitness: np.ndarray) -> float:
    """Mean fitness in the archive."""
    if len(fitness) == 0:
        return 0.0
    return float(np.mean(fitness))


def compute_all_metrics(
    fitness: np.ndarray,
    descriptors: np.ndarray,
    budget_k: Optional[int] = None,
    bounds_min: Optional[np.ndarray] = None,
    bounds_max: Optional[np.ndarray] = None,
    grid_resolution: int = 50,
    knn_k: int = 5,
) -> Dict[str, float]:
    """
    Compute all standard QD metrics.

    Args:
        fitness: Fitness values (N,)
        descriptors: Behavior descriptors (N, D)
        budget_k: Optional budget for fair QD-Score comparison
        bounds_min: Lower bounds (defaults to data min)
        bounds_max: Upper bounds (defaults to data max)
        grid_resolution: Resolution for coverage calculation
        knn_k: K for uniformity calculation

    Returns:
        Dictionary of metric names to values
    """
    if len(fitness) == 0:
        return {
            "qd_score": 0.0,
            "qd_score_at_budget": 0.0,
            "max_fitness": 0.0,
            "mean_fitness": 0.0,
            "archive_size": 0,
            "coverage": 0.0,
            "mean_pairwise_distance": 0.0,
            "uniformity_cv": 0.0,
        }

    if bounds_min is None:
        bounds_min = np.min(descriptors, axis=0) - 1e-6
    if bounds_max is None:
        bounds_max = np.max(descriptors, axis=0) + 1e-6

    metrics = {
        "qd_score": qd_score(fitness),
        "max_fitness": max_fitness(fitness),
        "mean_fitness": mean_fitness(fitness),
        "archive_size": len(fitness),
        "coverage": archive_coverage(
            descriptors, bounds_min, bounds_max, grid_resolution
        ),
        "mean_pairwise_distance": mean_pairwise_distance(descriptors),
        "uniformity_cv": archive_uniformity(descriptors, knn_k),
    }

    if budget_k is not None:
        metrics["qd_score_at_budget"] = qd_score_at_budget(fitness, budget_k)
        # Also compute mean fitness of top K
        if len(fitness) > budget_k:
            top_k_fit = np.sort(fitness)[-budget_k:]
            metrics["mean_fitness_at_budget"] = float(np.mean(top_k_fit))
        else:
            metrics["mean_fitness_at_budget"] = metrics["mean_fitness"]

    return metrics
