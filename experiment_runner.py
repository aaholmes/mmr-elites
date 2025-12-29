"""
Experiment Runner for MUSE-QD
==============================

Runs reproducible experiments with multiple seeds and proper statistical analysis.
Designed for GECCO-quality experimental methodology.

Usage:
    python -m mmr_qd.experiment_runner --config configs/arm20_benchmark.yaml
"""

import numpy as np
import json
import pickle
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Callable, Dict, List, Any
from concurrent.futures import ProcessPoolExecutor

# These would be your actual imports
# import mmr_elites_rs
# from tasks.arm_20 import Arm20Task
# from mmr_qd.metrics import compute_all_metrics, aggregate_runs


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    
    # Identification
    name: str
    algorithm: str  # "muse_qd" or "map_elites" or "cvt_map_elites"
    task: str       # "arm20" or "ant"
    
    # Algorithm Hyperparameters
    archive_size: int = 1000
    lambda_val: float = 0.5  # Only for MUSE-QD
    
    # Evolution Parameters
    generations: int = 1000
    batch_size: int = 200
    mutation_sigma: float = 0.1
    
    # Statistical Parameters
    n_seeds: int = 10
    seed_offset: int = 0  # Starting seed
    
    # Task-Specific
    task_kwargs: Optional[Dict] = None
    
    # Logging
    log_interval: int = 50
    snapshot_interval: int = 100
    
    # Output
    output_dir: str = "results"
    
    def __post_init__(self):
        if self.task_kwargs is None:
            self.task_kwargs = {}


@dataclass
class RunResult:
    """Results from a single run."""
    seed: int
    final_metrics: Dict[str, float]
    history: Dict[str, List[float]]  # metric_name -> values over generations
    runtime_seconds: float
    config: ExperimentConfig


def run_muse_qd(config: ExperimentConfig, seed: int) -> RunResult:
    """
    Run MUSE-QD with given configuration and seed.
    
    This is the core experiment loop - adapt to your actual implementation.
    """
    import mmr_elites_rs
    from tasks.arm_20 import Arm20Task
    from mmr_qd.metrics import compute_all_metrics
    
    np.random.seed(seed)
    
    # Initialize
    task = Arm20Task(**config.task_kwargs)
    selector = mmr_elites_rs.MMRSelector(config.archive_size, config.lambda_val)
    
    # Initial population
    archive = np.random.uniform(-np.pi, np.pi, (config.archive_size, 20))
    fit, desc = task.evaluate(archive)
    
    # Initial selection
    indices = selector.select(fit, desc)
    archive = archive[indices]
    fit = fit[indices]
    desc = desc[indices]
    
    # History tracking
    history = {
        "generation": [],
        "qd_score": [],
        "max_fitness": [],
        "mean_fitness": [],
        "mean_pairwise_distance": [],
        "uniformity_cv": [],
    }
    
    start_time = time.time()
    
    # Evolution loop
    for gen in range(1, config.generations + 1):
        # Mutation
        parents_idx = np.random.randint(0, len(archive), config.batch_size)
        parents = archive[parents_idx]
        offspring = parents + np.random.normal(0, config.mutation_sigma, (config.batch_size, 20))
        offspring = np.clip(offspring, -np.pi, np.pi)
        
        # Evaluation
        off_fit, off_desc = task.evaluate(offspring)
        
        # Pool and select
        pool_genes = np.vstack([archive, offspring])
        pool_fit = np.concatenate([fit, off_fit])
        pool_desc = np.vstack([desc, off_desc])
        
        survivor_idx = selector.select(pool_fit, pool_desc)
        
        archive = pool_genes[survivor_idx]
        fit = pool_fit[survivor_idx]
        desc = pool_desc[survivor_idx]
        
        # Logging
        if gen % config.log_interval == 0:
            metrics = compute_all_metrics(fit, desc)
            
            history["generation"].append(gen)
            history["qd_score"].append(metrics["qd_score"])
            history["max_fitness"].append(metrics["max_fitness"])
            history["mean_fitness"].append(metrics["mean_fitness"])
            history["mean_pairwise_distance"].append(metrics["mean_pairwise_distance"])
            history["uniformity_cv"].append(metrics["uniformity_cv"])
    
    runtime = time.time() - start_time
    
    # Final metrics
    final_metrics = compute_all_metrics(fit, desc)
    
    return RunResult(
        seed=seed,
        final_metrics=final_metrics,
        history=history,
        runtime_seconds=runtime,
        config=config
    )


def run_map_elites(config: ExperimentConfig, seed: int) -> RunResult:
    """
    Run MAP-Elites baseline with given configuration and seed.
    
    Uses sparse grid (dictionary) for high-dimensional behavior spaces.
    """
    from tasks.arm_20 import Arm20Task
    from mmr_qd.metrics import compute_all_metrics
    
    np.random.seed(seed)
    
    # Configuration for grid
    BINS_PER_DIM = 5  # 5^20 is impossibly large, demonstrating the problem
    
    task = Arm20Task(**config.task_kwargs)
    
    def get_grid_key(descriptor: np.ndarray) -> tuple:
        """Discretize descriptor to grid cell."""
        # Normalize to [0, 1] assuming descriptor is angles in [-pi, pi]
        normalized = (descriptor + np.pi) / (2 * np.pi)
        normalized = np.clip(normalized, 0, 0.9999)
        indices = (normalized * BINS_PER_DIM).astype(int)
        return tuple(indices)
    
    # Archive: dict mapping grid_key -> (genome, fitness, descriptor)
    archive = {}
    
    # Initial population
    init_pop = np.random.uniform(-np.pi, np.pi, (config.batch_size, 20))
    fit, desc = task.evaluate(init_pop)
    
    for i in range(len(init_pop)):
        key = get_grid_key(init_pop[i])
        if key not in archive or fit[i] > archive[key][1]:
            archive[key] = (init_pop[i].copy(), fit[i], desc[i].copy())
    
    # History tracking
    history = {
        "generation": [],
        "qd_score": [],
        "max_fitness": [],
        "mean_fitness": [],
        "mean_pairwise_distance": [],
        "uniformity_cv": [],
        "archive_size": [],
    }
    
    start_time = time.time()
    
    for gen in range(1, config.generations + 1):
        # Select parents from archive
        if archive:
            keys = list(archive.keys())
            parent_keys = [keys[np.random.randint(len(keys))] for _ in range(config.batch_size)]
            parents = np.array([archive[k][0] for k in parent_keys])
        else:
            parents = np.random.uniform(-np.pi, np.pi, (config.batch_size, 20))
        
        # Mutation
        offspring = parents + np.random.normal(0, config.mutation_sigma, parents.shape)
        offspring = np.clip(offspring, -np.pi, np.pi)
        
        # Evaluation
        off_fit, off_desc = task.evaluate(offspring)
        
        # Add to archive
        for i in range(len(offspring)):
            key = get_grid_key(offspring[i])
            if key not in archive or off_fit[i] > archive[key][1]:
                archive[key] = (offspring[i].copy(), off_fit[i], off_desc[i].copy())
        
        # Logging
        if gen % config.log_interval == 0:
            # Extract arrays from archive
            all_fit = np.array([v[1] for v in archive.values()])
            all_desc = np.array([v[2] for v in archive.values()])
            
            metrics = compute_all_metrics(all_fit, all_desc)
            
            history["generation"].append(gen)
            history["qd_score"].append(metrics["qd_score"])
            history["max_fitness"].append(metrics["max_fitness"])
            history["mean_fitness"].append(metrics["mean_fitness"])
            history["mean_pairwise_distance"].append(metrics["mean_pairwise_distance"])
            history["uniformity_cv"].append(metrics["uniformity_cv"])
            history["archive_size"].append(len(archive))
    
    runtime = time.time() - start_time
    
    # Final metrics
    all_fit = np.array([v[1] for v in archive.values()])
    all_desc = np.array([v[2] for v in archive.values()])
    final_metrics = compute_all_metrics(all_fit, all_desc)
    final_metrics["archive_size"] = len(archive)
    
    return RunResult(
        seed=seed,
        final_metrics=final_metrics,
        history=history,
        runtime_seconds=runtime,
        config=config
    )


def run_experiment(config: ExperimentConfig, parallel: bool = True) -> List[RunResult]:
    """
    Run a full experiment with multiple seeds.
    
    Args:
        config: Experiment configuration
        parallel: Whether to run seeds in parallel
    
    Returns:
        List of results for each seed
    """
    seeds = list(range(config.seed_offset, config.seed_offset + config.n_seeds))
    
    # Select algorithm runner
    if config.algorithm == "muse_qd":
        runner = run_muse_qd
    elif config.algorithm == "map_elites":
        runner = run_map_elites
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")
    
    print(f"Running {config.name} with {config.n_seeds} seeds...")
    print(f"  Algorithm: {config.algorithm}")
    print(f"  Task: {config.task}")
    print(f"  Generations: {config.generations}")
    
    results = []
    
    if parallel and config.n_seeds > 1:
        # Note: This won't work with Rust backend due to GIL issues
        # For now, run sequentially
        pass
    
    # Sequential execution
    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i+1}/{config.n_seeds})...")
        result = runner(config, seed)
        results.append(result)
        print(f"    Done in {result.runtime_seconds:.1f}s, "
              f"QD-Score: {result.final_metrics['qd_score']:.2f}")
    
    return results


def save_results(results: List[RunResult], output_dir: Path):
    """Save experiment results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results as pickle
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Save summary as JSON
    summary = {
        "config": asdict(results[0].config),
        "n_runs": len(results),
        "final_metrics": {}
    }
    
    # Aggregate metrics
    for key in results[0].final_metrics.keys():
        values = [r.final_metrics[key] for r in results]
        summary["final_metrics"][key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {output_dir}")


def print_comparison_table(
    results_dict: Dict[str, List[RunResult]],
    metrics: List[str] = None
):
    """
    Print a publication-ready comparison table.
    
    Args:
        results_dict: Dictionary mapping algorithm names to results
        metrics: Metrics to include (default: all)
    """
    if metrics is None:
        metrics = ["qd_score", "max_fitness", "mean_fitness", 
                   "mean_pairwise_distance", "archive_size"]
    
    # Header
    header = "| Metric |"
    separator = "|--------|"
    for name in results_dict.keys():
        header += f" {name} |"
        separator += "------------|"
    
    print(header)
    print(separator)
    
    # Rows
    for metric in metrics:
        row = f"| {metric} |"
        for name, results in results_dict.items():
            values = [r.final_metrics.get(metric, 0) for r in results]
            mean = np.mean(values)
            std = np.std(values)
            row += f" {mean:.2f}±{std:.2f} |"
        print(row)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run MUSE-QD experiments")
    parser.add_argument("--algorithm", choices=["muse_qd", "map_elites", "both"],
                        default="both", help="Algorithm to run")
    parser.add_argument("--task", default="arm20", help="Task to run")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--generations", type=int, default=1000, help="Generations")
    parser.add_argument("--archive-size", type=int, default=1000, help="Archive size (MUSE)")
    parser.add_argument("--lambda", type=float, default=0.5, dest="lambda_val",
                        help="Lambda value (MUSE)")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    base_config = {
        "task": args.task,
        "generations": args.generations,
        "n_seeds": args.seeds,
        "archive_size": args.archive_size,
        "lambda_val": args.lambda_val,
        "output_dir": args.output_dir,
    }
    
    all_results = {}
    
    if args.algorithm in ["muse_qd", "both"]:
        config = ExperimentConfig(
            name="muse_qd_experiment",
            algorithm="muse_qd",
            **base_config
        )
        results = run_experiment(config)
        save_results(results, Path(args.output_dir) / "muse_qd")
        all_results["MUSE-QD"] = results
    
    if args.algorithm in ["map_elites", "both"]:
        config = ExperimentConfig(
            name="map_elites_experiment",
            algorithm="map_elites",
            **base_config
        )
        results = run_experiment(config)
        save_results(results, Path(args.output_dir) / "map_elites")
        all_results["MAP-Elites"] = results
    
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("COMPARISON TABLE")
        print("="*60)
        print_comparison_table(all_results)


if __name__ == "__main__":
    main()
