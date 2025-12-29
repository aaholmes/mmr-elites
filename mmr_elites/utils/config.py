from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class ExperimentConfig:
    """Configuration for QD experiments."""
    task: str = "arm"  # arm, rastrigin, ant
    algorithm: str = "mmr_elites"  # mmr_elites, map_elites, cvt_map_elites, random
    
    # Task params
    n_dof: int = 20
    
    # Algorithm params
    generations: int = 1000
    batch_size: int = 100
    archive_size: int = 1000  # For MMR, Random
    n_niches: int = 1000     # For CVT
    bins_per_dim: int = 5    # For MAP-Elites
    lambda_val: float = 0.5  # For MMR
    mutation_sigma: float = 0.1
    
    # Execution
    seed: int = 42
    log_interval: int = 10
    output_dir: str = "results"
    exp_name: str = "experiment"
