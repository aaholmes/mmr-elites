"""
Abstract base class for Tasks.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class Task(ABC):
    """
    Abstract base class for benchmark tasks.
    """

    @abstractmethod
    def evaluate(self, genomes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a batch of genomes.

        Args:
            genomes: Genomes (batch_size, genome_dim)

        Returns:
            fitness: Fitness values (batch_size,)
            descriptors: Behavior descriptors (batch_size, descriptor_dim)
        """
        pass

    @property
    def genome_dim(self) -> int:
        """Dimensionality of the genome."""
        raise NotImplementedError

    @property
    def desc_dim(self) -> int:
        """Dimensionality of the behavior descriptor."""
        raise NotImplementedError
