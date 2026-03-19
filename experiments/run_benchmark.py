"""
Main experiment runner for MMR-Elites.
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mmr_elites.algorithms.cvt_map_elites import run_cvt_map_elites
from mmr_elites.algorithms.map_elites import run_map_elites
from mmr_elites.algorithms.mmr_elites import run_mmr_elites
from mmr_elites.algorithms.random_search import run_random_search
from mmr_elites.tasks.arm import ArmTask
from mmr_elites.tasks.rastrigin import RastriginTask
from mmr_elites.utils.config import ExperimentConfig


def run_experiment(config: ExperimentConfig):
    """Run a single experiment based on config."""
    print(f"Running {config.algorithm} on {config.task} (seed={config.seed})...")

    # Setup task
    if config.task == "arm":
        task = ArmTask(n_dof=config.n_dof, use_highdim_descriptor=True)
    elif config.task == "rastrigin":
        task = RastriginTask(n_dim=config.n_dof)
    elif config.task == "ant":
        from mmr_elites.tasks.ant import AntTask

        task = AntTask()
    else:
        raise ValueError(f"Unknown task: {config.task}")

    # Setup output
    out_path = (
        Path(config.output_dir) / config.exp_name / f"{config.algorithm}_s{config.seed}"
    )
    out_path.mkdir(parents=True, exist_ok=True)

    # Run algorithm
    if config.algorithm == "mmr_elites":
        result = run_mmr_elites(
            task=task,
            archive_size=config.archive_size,
            generations=config.generations,
            batch_size=config.batch_size,
            lambda_val=config.lambda_val,
            mutation_sigma=config.mutation_sigma,
            seed=config.seed,
            log_interval=config.log_interval,
        )
    elif config.algorithm == "map_elites":
        result = run_map_elites(
            task=task,
            generations=config.generations,
            batch_size=config.batch_size,
            bins_per_dim=config.bins_per_dim,
            mutation_sigma=config.mutation_sigma,
            seed=config.seed,
            log_interval=config.log_interval,
        )
    elif config.algorithm == "cvt_map_elites":
        result = run_cvt_map_elites(
            task=task,
            n_niches=config.n_niches,
            generations=config.generations,
            batch_size=config.batch_size,
            mutation_sigma=config.mutation_sigma,
            seed=config.seed,
            log_interval=config.log_interval,
        )
    elif config.algorithm == "random":
        result = run_random_search(
            task=task,
            archive_size=config.archive_size,
            generations=config.generations,
            batch_size=config.batch_size,
            seed=config.seed,
            log_interval=config.log_interval,
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    # Save results
    with open(out_path / "results.pkl", "wb") as f:
        pickle.dump(result, f)

    # Save summary as JSON (metrics only)
    summary = {
        "algorithm": result["algorithm"],
        "seed": result["seed"],
        "runtime": result["runtime"],
        "final_metrics": result["final_metrics"],
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Done. QD-Score: {result['final_metrics']['qd_score']:.2f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="arm")
    parser.add_argument("--algo", type=str, default="mmr_elites")
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dof", type=int, default=20)
    args = parser.parse_args()

    config = ExperimentConfig(
        task=args.task,
        algorithm=args.algo,
        generations=args.gens,
        seed=args.seed,
        n_dof=args.dof,
        exp_name="quick_test",
    )

    run_experiment(config)
