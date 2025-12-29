"""
QD Metrics Module for MUSE-QD
=============================

Implements standard Quality-Diversity metrics for scientific comparison.

References:
- Mouret & Clune (2015) "Illuminating search spaces by mapping elites"
- Pugh et al. (2016) "Quality Diversity: A New Frontier for Evolutionary Computation"
- Fontaine et al. (2020) "Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space"
"""

import numpy as np
from typing import Tuple, Optional, Dict
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


def archive_coverage(
    descriptors: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    grid_resolution: int = 100
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
        total_cells = grid_resolution ** n_dims
    else:
        # For high-D, report raw count (coverage ratio meaningless)
        return len(unique_cells)
    
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
    dists = cdist(descriptors, descriptors, metric='euclidean')
    
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
    distances, _ = tree.query(descriptors, k=k+1)
    
    # Average distance to k nearest neighbors (excluding self)
    mean_knn_dists = np.mean(distances[:, 1:], axis=1)
    
    # Coefficient of variation
    cv = np.std(mean_knn_dists) / (np.mean(mean_knn_dists) + 1e-10)
    
    return float(cv)


def max_fitness(fitness: np.ndarray) -> float:
    """Maximum fitness in the archive."""
    return float(np.max(fitness))


def mean_fitness(fitness: np.ndarray) -> float:
    """Mean fitness in the archive."""
    return float(np.mean(fitness))


def compute_all_metrics(
    fitness: np.ndarray,
    descriptors: np.ndarray,
    bounds_min: Optional[np.ndarray] = None,
    bounds_max: Optional[np.ndarray] = None,
    grid_resolution: int = 50,
    knn_k: int = 5
) -> Dict[str, float]:
    """
    Compute all standard QD metrics.
    
    Args:
        fitness: Fitness values (N,)
        descriptors: Behavior descriptors (N, D)
        bounds_min: Lower bounds (defaults to data min)
        bounds_max: Upper bounds (defaults to data max)
        grid_resolution: Resolution for coverage calculation
        knn_k: K for uniformity calculation
    
    Returns:
        Dictionary of metric names to values
    """
    if bounds_min is None:
        bounds_min = np.min(descriptors, axis=0) - 1e-6
    if bounds_max is None:
        bounds_max = np.max(descriptors, axis=0) + 1e-6
    
    return {
        "qd_score": qd_score(fitness),
        "max_fitness": max_fitness(fitness),
        "mean_fitness": mean_fitness(fitness),
        "archive_size": len(fitness),
        "coverage": archive_coverage(descriptors, bounds_min, bounds_max, grid_resolution),
        "mean_pairwise_distance": mean_pairwise_distance(descriptors),
        "uniformity_cv": archive_uniformity(descriptors, knn_k),
    }


# =============================================================================
# High-Dimensional Coverage Metrics (For D > 10)
# =============================================================================

def epsilon_coverage(
    descriptors: np.ndarray,
    epsilon: float,
    reference_samples: int = 10000,
    bounds_min: Optional[np.ndarray] = None,
    bounds_max: Optional[np.ndarray] = None,
    seed: int = 42
) -> float:
    """
    Epsilon-Coverage: Fraction of random samples within epsilon of an archive member.
    
    For high-dimensional spaces where grid-based coverage is meaningless.
    Samples random points uniformly and checks if each is "covered" by the archive.
    
    Args:
        descriptors: Behavior descriptors (N, D)
        epsilon: Coverage radius
        reference_samples: Number of random points to sample
        bounds_min: Lower bounds
        bounds_max: Upper bounds
        seed: Random seed for reproducibility
    
    Returns:
        Fraction of reference samples within epsilon of archive
    """
    n_dims = descriptors.shape[1]
    
    if bounds_min is None:
        bounds_min = np.min(descriptors, axis=0) - 0.1
    if bounds_max is None:
        bounds_max = np.max(descriptors, axis=0) + 0.1
    
    rng = np.random.default_rng(seed)
    
    # Sample reference points uniformly
    reference = rng.uniform(bounds_min, bounds_max, size=(reference_samples, n_dims))
    
    # Build KD-tree for fast queries
    tree = cKDTree(descriptors)
    
    # Query for nearest neighbor distance
    dists, _ = tree.query(reference, k=1)
    
    # Count covered points
    covered = np.sum(dists <= epsilon)
    
    return covered / reference_samples


def sum_of_knn_distances(descriptors: np.ndarray, k: int = 1) -> float:
    """
    Sum of k-NN Distances: Total spread metric.
    
    For each archive member, sum the distance to its k-th nearest neighbor.
    Higher = more spread out archive.
    
    This is related to the Dispersion metric used in some QD papers.
    
    Args:
        descriptors: Behavior descriptors (N, D)
        k: Which neighbor to use (1 = nearest)
    
    Returns:
        Sum of k-NN distances
    """
    if len(descriptors) <= k:
        return 0.0
    
    tree = cKDTree(descriptors)
    distances, _ = tree.query(descriptors, k=k+1)
    
    # k-th neighbor is at index k (0 is self)
    return float(np.sum(distances[:, k]))


# =============================================================================
# Experiment Aggregation Utilities
# =============================================================================

def aggregate_runs(
    metrics_list: list[Dict[str, float]]
) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate metrics across multiple runs.
    
    Args:
        metrics_list: List of metric dictionaries from multiple runs
    
    Returns:
        Dictionary of metric names to (mean, std) tuples
    """
    if not metrics_list:
        return {}
    
    keys = metrics_list[0].keys()
    result = {}
    
    for key in keys:
        values = [m[key] for m in metrics_list]
        result[key] = (float(np.mean(values)), float(np.std(values)))
    
    return result


def format_metrics_table(
    aggregated: Dict[str, Tuple[float, float]],
    precision: int = 4
) -> str:
    """
    Format aggregated metrics as a publication-ready table.
    
    Args:
        aggregated: Output from aggregate_runs()
        precision: Decimal places
    
    Returns:
        Formatted string table
    """
    lines = ["| Metric | Mean ± Std |", "|--------|------------|"]
    
    for key, (mean, std) in aggregated.items():
        lines.append(f"| {key} | {mean:.{precision}f} ± {std:.{precision}f} |")
    
    return "\n".join(lines)
