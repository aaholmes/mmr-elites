"""
Abstract base class for QD algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class QDResult:
    """Result from a QD algorithm run."""
    algorithm: str
    seed: int
    runtime: float
    final_metrics: Dict[str, float]
    history: Dict[str, List[float]]
    final_genomes: np.ndarray
    final_fitness: np.ndarray
    final_descriptors: np.ndarray


class QDAlgorithm(ABC):
    """Abstract base class for Quality-Diversity algorithms."""
    
    @abstractmethod
    def run(
        self,
        task,
        generations: int,
        batch_size: int,
        mutation_sigma: float,
        seed: int,
        log_interval: int = 100,
    ) -> QDResult:
        """
        Run the QD algorithm.
        
        Args:
            task: Task with evaluate(genomes) method
            generations: Number of generations
            batch_size: Offspring per generation
            mutation_sigma: Gaussian mutation std
            seed: Random seed
            log_interval: Logging frequency
            
        Returns:
            QDResult with final state and history
        """
        pass
    
    @abstractmethod
    def get_archive_size(self) -> int:
        """Return current archive size."""
        pass
