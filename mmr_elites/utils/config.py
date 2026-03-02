"""
Unified configuration for MMR-Elites experiments.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ExperimentConfig:
    """
    Complete configuration for QD experiments.

    This is the single source of truth for experiment configuration.
    All algorithms and experiment scripts should use this class.
    """

    # Task configuration
    task: str = "arm"
    n_dof: int = 20

    # Algorithm selection
    algorithm: str = "mmr_elites"

    # Archive settings (used by MMR-Elites, CVT-MAP-Elites, Random)
    archive_size: int = 1000
    n_niches: int = 1000  # Alias for CVT-MAP-Elites

    # MMR-Elites specific
    lambda_val: float = 0.5

    # MAP-Elites specific
    bins_per_dim: int = 3

    # Evolution settings
    generations: int = 1000
    batch_size: int = 200
    mutation_sigma: float = 0.1

    # Execution settings
    seed: int = 42
    log_interval: int = 100

    # Output settings
    output_dir: str = "results"
    exp_name: str = "experiment"

    def __post_init__(self):
        """Ensure n_niches matches archive_size by default."""
        if self.n_niches != self.archive_size:
            self.n_niches = self.archive_size

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict

        return asdict(self)
