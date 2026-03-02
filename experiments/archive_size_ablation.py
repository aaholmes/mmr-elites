"""
Archive Size Ablation Experiment
================================

Tests how archive size K affects MMR-Elites performance.

Hypothesis: Larger archives capture finer-grained diversity but may
dilute fitness pressure. There should be diminishing returns beyond
a certain K for a given evaluation budget.
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional

from mmr_elites.tasks.arm import ArmTask
from mmr_elites.algorithms import run_mmr_elites


def run_archive_size_ablation(
    archive_sizes: Optional[List[int]] = None,
    n_seeds: int = 5,
    generations: int = 2000,
    n_dof: int = 20,
    batch_size: int = 200,
    lambda_val: float = 0.5,
    mutation_sigma: float = 0.1,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Run archive size ablation experiment.

    Args:
        archive_sizes: List of archive sizes K to test
        n_seeds: Number of random seeds
        generations: Generations per run
        n_dof: Arm degrees of freedom
        batch_size: Offspring per generation
        lambda_val: MMR-Elites diversity weight
        mutation_sigma: Mutation standard deviation
        output_dir: Where to save results

    Returns:
        Dictionary with all results
    """
    if archive_sizes is None:
        archive_sizes = [100, 500, 1000, 2000, 5000]

    if output_dir is None:
        output_dir = Path("results/archive_size_ablation")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task = ArmTask(n_dof=n_dof, use_highdim_descriptor=True)

    all_results = {k: [] for k in archive_sizes}

    print("Archive Size Ablation Experiment")
    print(f"Task: {n_dof}-DOF Arm")
    print(f"Archive sizes: {archive_sizes}")
    print(f"Seeds: {n_seeds}")

    for k in archive_sizes:
        print(f"\n{'='*50}")
        print(f"K = {k}")
        print("="*50)

        for seed in range(n_seeds):
            print(f"  Seed {seed + 1}/{n_seeds}...", end=" ", flush=True)

            result = run_mmr_elites(
                task,
                archive_size=k,
                generations=generations,
                batch_size=batch_size,
                lambda_val=lambda_val,
                mutation_sigma=mutation_sigma,
                seed=seed,
            )
            all_results[k].append(result)

            print(f"QD@K={result['final_metrics']['qd_score_at_budget']:.2f}, "
                  f"CV={result['final_metrics']['uniformity_cv']:.3f}, "
                  f"MeanFit={result['final_metrics']['mean_fitness']:.4f}")

    # Save results
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    # Generate and save summary
    summary = {}
    for k in archive_sizes:
        metrics = {}
        for metric in ["qd_score_at_budget", "mean_fitness", "max_fitness",
                        "uniformity_cv", "mean_pairwise_distance"]:
            values = [r["final_metrics"][metric] for r in all_results[k]]
            metrics[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        summary[str(k)] = metrics

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    print(f"{'K':<8} | {'QD@K':<18} | {'Mean Fit':<18} | {'Uniformity CV':<18}")
    print("-"*90)

    for k in archive_sizes:
        qd = summary[str(k)]["qd_score_at_budget"]
        mf = summary[str(k)]["mean_fitness"]
        cv = summary[str(k)]["uniformity_cv"]
        print(f"{k:<8} | {qd['mean']:>7.2f} +/- {qd['std']:<6.2f} | "
              f"{mf['mean']:>7.4f} +/- {mf['std']:<6.4f} | "
              f"{cv['mean']:>7.4f} +/- {cv['std']:<6.4f}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Archive Size Ablation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (2 seeds, 200 generations)")
    parser.add_argument("--full", action="store_true",
                        help="Full experiment (10 seeds, 2000 generations)")
    args = parser.parse_args()

    if args.quick:
        run_archive_size_ablation(
            archive_sizes=[100, 500, 1000],
            n_seeds=2,
            generations=200,
        )
    elif args.full:
        run_archive_size_ablation(
            archive_sizes=[100, 500, 1000, 2000, 5000],
            n_seeds=10,
            generations=2000,
        )
    else:
        run_archive_size_ablation()
