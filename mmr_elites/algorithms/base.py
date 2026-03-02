"""
Abstract base class for Quality-Diversity algorithms.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import unified config
from mmr_elites.utils.config import ExperimentConfig


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
    config: Optional[ExperimentConfig] = None


class QDAlgorithm(ABC):
    """Abstract base class for Quality-Diversity algorithms."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.archive = None
        self.fitness = None
        self.descriptors = None

    @abstractmethod
    def initialize(self, task, seed: int):
        """Initialize the archive."""
        pass

    @abstractmethod
    def step(self, task) -> Dict[str, float]:
        """Perform one generation step. Returns metrics."""
        pass

    @abstractmethod
    def get_archive(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (genomes, fitness, descriptors)."""
        pass

    def run(self, task, seed: int) -> QDResult:
        """Run the full evolution loop."""
        np.random.seed(seed)

        self.initialize(task, seed)

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

        for gen in range(1, self.config.generations + 1):
            metrics = self.step(task)

            if gen % self.config.log_interval == 0:
                history["generation"].append(gen)
                for key in history:
                    if key != "generation" and key in metrics:
                        history[key].append(metrics[key])

        runtime = time.time() - start_time
        genomes, fitness, descriptors = self.get_archive()

        # Final metrics
        from mmr_elites.metrics.qd_metrics import compute_all_metrics

        final_metrics = compute_all_metrics(
            fitness, descriptors, self.config.archive_size
        )

        return QDResult(
            algorithm=self.__class__.__name__,
            seed=seed,
            runtime=runtime,
            final_metrics=final_metrics,
            history=history,
            final_genomes=genomes,
            final_fitness=fitness,
            final_descriptors=descriptors,
            config=self.config,
        )
