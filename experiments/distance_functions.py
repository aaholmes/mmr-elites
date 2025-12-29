# experiments/distance_functions.py
"""
Distance functions for MMR-Elites comparison.
"""

import numpy as np
from typing import Callable


def euclidean_distance(b1: np.ndarray, b2: np.ndarray) -> float:
    """Standard Euclidean distance."""
    return np.linalg.norm(b1 - b2)


def saturating_distance(b1: np.ndarray, b2: np.ndarray, sigma: float = 0.2) -> float:
    """
    Saturating distance: 1 - exp(-||b1-b2||² / σ²)
    
    Properties:
    - Range: [0, 1]
    - Strong gradient for small distances
    - Saturates at 1 for large distances
    - σ controls "different enough" threshold
    """
    d_sq = np.sum((b1 - b2) ** 2)
    return 1.0 - np.exp(-d_sq / (sigma ** 2))


def cosine_distance(b1: np.ndarray, b2: np.ndarray) -> float:
    """Cosine distance: 1 - cos(θ)."""
    norm1 = np.linalg.norm(b1)
    norm2 = np.linalg.norm(b2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 1.0
    return 1.0 - np.dot(b1, b2) / (norm1 * norm2)


def normalized_euclidean(b1: np.ndarray, b2: np.ndarray, max_dist: float = None) -> float:
    """
    Euclidean distance normalized to [0, 1].
    
    max_dist: Maximum possible distance (e.g., sqrt(D) for [0,1]^D)
    """
    d = np.linalg.norm(b1 - b2)
    if max_dist is None:
        max_dist = np.sqrt(len(b1))  # Assumes [0,1]^D
    return min(d / max_dist, 1.0)


# Vectorized versions for efficiency
def euclidean_distance_matrix(descriptors: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances."""
    from scipy.spatial.distance import cdist
    return cdist(descriptors, descriptors, metric='euclidean')


def saturating_distance_matrix(descriptors: np.ndarray, sigma: float = 0.2) -> np.ndarray:
    """Compute pairwise saturating distances."""
    from scipy.spatial.distance import cdist
    d_sq = cdist(descriptors, descriptors, metric='sqeuclidean')
    return 1.0 - np.exp(-d_sq / (sigma ** 2))


def get_distance_function(name: str, **kwargs) -> Callable:
    """Get distance function by name."""
    functions = {
        'euclidean': euclidean_distance,
        'saturating': lambda b1, b2: saturating_distance(b1, b2, kwargs.get('sigma', 0.2)),
        'cosine': cosine_distance,
        'normalized': lambda b1, b2: normalized_euclidean(b1, b2, kwargs.get('max_dist')),
    }
    return functions[name]
