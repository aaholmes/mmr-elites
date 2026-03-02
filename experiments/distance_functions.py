"""
Distance functions for MMR-Elites experiments.

Key insight: Exponential saturation maintains gradient at small distances,
unlike Gaussian which has a "dead zone" near zero.
"""

from typing import Callable, Optional

import numpy as np
from scipy.spatial.distance import cdist, pdist

# =============================================================================
# Core Distance Functions
# =============================================================================


def euclidean_distance(b1: np.ndarray, b2: np.ndarray) -> float:
    """Standard Euclidean distance (unbounded)."""
    return np.linalg.norm(b1 - b2)


def normalized_euclidean(
    b1: np.ndarray, b2: np.ndarray, max_dist: Optional[float] = None
) -> float:
    """Euclidean normalized to [0, 1]."""
    d = np.linalg.norm(b1 - b2)
    if max_dist is None:
        max_dist = np.sqrt(len(b1))  # Diagonal of [0,1]^D
    return min(d / max_dist, 1.0)


def gaussian_saturating(b1: np.ndarray, b2: np.ndarray, sigma: float) -> float:
    """
    Gaussian saturating distance: 1 - exp(-||b1-b2||² / 2σ²)

    Problem: Near-zero gradient at small distances (dead zone).
    """
    d_sq = np.sum((b1 - b2) ** 2)
    return 1.0 - np.exp(-d_sq / (2 * sigma**2))


def exponential_saturating(b1: np.ndarray, b2: np.ndarray, sigma: float) -> float:
    """
    Exponential saturating distance: 1 - exp(-||b1-b2|| / σ)

    Advantages:
    - Linear gradient at small distances (no dead zone)
    - Interpretable σ: at d=σ, output ≈ 0.63
    - Matches Laplacian kernel from ML literature

    Args:
        b1, b2: Behavior descriptors
        sigma: Characteristic scale ("different enough" threshold)

    Returns:
        Distance in [0, 1)
    """
    d = np.linalg.norm(b1 - b2)
    return 1.0 - np.exp(-d / sigma)


# =============================================================================
# Automatic Sigma Selection
# =============================================================================


def auto_sigma(descriptors: np.ndarray, percentile: float = 50) -> float:
    """
    Automatically compute σ from data distribution.

    Sets σ to the median (or other percentile) of pairwise distances,
    so that "different enough" is calibrated to the actual data spread.

    Args:
        descriptors: (N, D) array of behavior descriptors
        percentile: Which percentile of distances to use (default: median)

    Returns:
        Recommended σ value
    """
    if len(descriptors) < 2:
        return 0.1  # Fallback

    # Sample if too large (for efficiency)
    if len(descriptors) > 1000:
        idx = np.random.choice(len(descriptors), 1000, replace=False)
        descriptors = descriptors[idx]

    distances = pdist(descriptors)
    return float(np.percentile(distances, percentile))


def auto_sigma_diagonal(dim: int, fraction: float = 0.2) -> float:
    """
    Simple σ estimate based on behavior space diagonal.

    For [0,1]^D, diagonal length is sqrt(D).
    σ = fraction * sqrt(D) means "different enough" is ~20% of max distance.

    Args:
        dim: Dimensionality of behavior space
        fraction: Fraction of diagonal (default: 0.2 = 20%)

    Returns:
        Recommended σ value
    """
    return fraction * np.sqrt(dim)


# =============================================================================
# Vectorized Distance Matrices (for efficiency)
# =============================================================================


def euclidean_distance_matrix(descriptors: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances."""
    return cdist(descriptors, descriptors, metric="euclidean")


def exponential_saturating_matrix(descriptors: np.ndarray, sigma: float) -> np.ndarray:
    """Pairwise exponential saturating distances."""
    d = cdist(descriptors, descriptors, metric="euclidean")
    return 1.0 - np.exp(-d / sigma)


def gaussian_saturating_matrix(descriptors: np.ndarray, sigma: float) -> np.ndarray:
    """Pairwise Gaussian saturating distances."""
    d_sq = cdist(descriptors, descriptors, metric="sqeuclidean")
    return 1.0 - np.exp(-d_sq / (2 * sigma**2))


# =============================================================================
# Distance Function Factory
# =============================================================================


def get_distance_function(
    name: str,
    sigma: Optional[float] = None,
    auto_sigma_percentile: float = 50,
    auto_sigma_fraction: float = 0.2,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Get distance function by name.

    Args:
        name: One of 'euclidean', 'normalized', 'gaussian', 'exponential'
        sigma: Scale parameter (required for gaussian/exponential, or auto-computed)
        auto_sigma_percentile: Percentile for auto sigma (if sigma=None)
        auto_sigma_fraction: Fraction of diagonal for simple auto sigma

    Returns:
        Distance function (b1, b2) -> float
    """
    from functools import partial

    if name == "euclidean":
        return euclidean_distance

    elif name == "normalized":
        return normalized_euclidean

    elif name == "gaussian":
        if sigma is None:
            raise ValueError(
                "sigma required for gaussian distance (or use 'gaussian_auto')"
            )
        return partial(gaussian_saturating, sigma=sigma)

    elif name == "exponential":
        if sigma is None:
            raise ValueError(
                "sigma required for exponential distance (or use 'exponential_auto')"
            )
        return partial(exponential_saturating, sigma=sigma)

    else:
        raise ValueError(f"Unknown distance function: {name}")
