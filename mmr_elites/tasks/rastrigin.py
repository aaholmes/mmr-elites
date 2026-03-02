"""
Rastrigin Function Task.
"""

from typing import Tuple

import numpy as np

from .base import Task


class RastriginTask(Task):
    """
    Rastrigin function optimization.
    Common benchmark for optimization algorithms.
    """

    def __init__(self, n_dim: int = 10, A: float = 10.0):
        self.n_dim = n_dim
        self.A = A
        self.genome_bounds = [(-5.12, 5.12)] * n_dim

    @property
    def genome_dim(self) -> int:
        return self.n_dim

    @property
    def desc_dim(self) -> int:
        return self.n_dim

    def evaluate(self, genomes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Map genomes from [-pi, pi] to [-5.12, 5.12]
        # Assuming input is in range [-pi, pi] from evolution loop
        scale = 5.12 / np.pi
        x = genomes * scale

        # Rastrigin function
        n = self.n_dim
        sum_sq = np.sum(x**2 - self.A * np.cos(2 * np.pi * x), axis=1)
        val = self.A * n + sum_sq

        # Maximize fitness (inverted Rastrigin)
        # Rastrigin min is 0 at origin.
        # We want to maximize, so we can do 1 / (1 + val) or -val
        # Or just bounded inverse:
        fitness = 100.0 / (1.0 + val)

        # Descriptor is just the genome itself (search space = behavior space)
        # Normalize to [0, 1] for standard metrics
        descriptors = (x + 5.12) / 10.24
        descriptors = np.clip(descriptors, 0, 1)

        return fitness, descriptors
