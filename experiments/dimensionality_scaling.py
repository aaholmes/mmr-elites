"""
Dimensionality Scaling Experiment
=================================

THE KEY EXPERIMENT for the paper.

Tests how algorithms scale as behavior descriptor dimension increases.
Expected result: MAP-Elites degrades, MMR-Elites maintains performance.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from mmr_elites.algorithms import (
    run_cvt_map_elites,
    run_map_elites,
    run_mmr_elites,
    run_random_search,
)
from mmr_elites.tasks.arm import ArmTask


def run_dimensionality_scaling(
    dimensions: List[int] = None,
    n_seeds: int = 10,
    generations: int = 2000,
    archive_size: int = 1000,
    batch_size: int = 200,
    lambda_val: float = 0.5,
    mutation_sigma: float = 0.1,
    output_dir: Path = None,
) -> Dict:
    """
    Run dimensionality scaling experiment.

    Args:
        dimensions: List of dimensions to test (default: [5, 10, 20, 50, 100])
        n_seeds: Number of random seeds
        generations: Generations per run
        archive_size: Archive size K
        batch_size: Offspring per generation
        lambda_val: MMR-Elites diversity weight
        mutation_sigma: Mutation standard deviation
        output_dir: Where to save results

    Returns:
        Dictionary with all results
    """
    if dimensions is None:
        dimensions = [5, 10, 20, 50, 100]

    if output_dir is None:
        output_dir = Path("results/dimensionality_scaling")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    algorithms = {
        "MMR-Elites": lambda task, seed: run_mmr_elites(
            task,
            archive_size=archive_size,
            generations=generations,
            batch_size=batch_size,
            lambda_val=lambda_val,
            mutation_sigma=mutation_sigma,
            seed=seed,
        ),
        "MAP-Elites": lambda task, seed: run_map_elites(
            task,
            generations=generations,
            batch_size=batch_size,
            bins_per_dim=3,
            mutation_sigma=mutation_sigma,
            seed=seed,
        ),
        "CVT-MAP-Elites": lambda task, seed: run_cvt_map_elites(
            task,
            n_niches=archive_size,
            generations=generations,
            batch_size=batch_size,
            mutation_sigma=mutation_sigma,
            seed=seed,
        ),
        "Random": lambda task, seed: run_random_search(
            task,
            archive_size=archive_size,
            generations=generations,
            batch_size=batch_size,
            seed=seed,
        ),
    }

    all_results = {dim: {alg: [] for alg in algorithms} for dim in dimensions}

    for dim in dimensions:
        print(f"\n{'='*60}")
        print(f"DIMENSION: {dim}")
        print("=" * 60)

        task = ArmTask(n_dof=dim, use_highdim_descriptor=True)

        for seed in range(n_seeds):
            print(f"\n  Seed {seed + 1}/{n_seeds}")

            for alg_name, alg_fn in algorithms.items():
                print(f"    {alg_name}...", end=" ", flush=True)
                try:
                    result = alg_fn(task, seed)
                    all_results[dim][alg_name].append(result)
                    print(
                        f"QD@K={result['final_metrics']['qd_score_at_budget']:.2f}, "
                        f"MeanFit={result['final_metrics']['mean_fitness']:.4f}"
                    )
                except Exception as e:
                    print(f"ERROR: {e}")

    # Save results
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    # Generate summary
    summary = generate_summary(all_results, dimensions, list(algorithms.keys()))
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print_summary_table(summary, dimensions, list(algorithms.keys()))

    # Add statistical analysis
    from mmr_elites.utils.statistics import compute_all_statistics

    print("\n" + "=" * 100)
    print("STATISTICAL ANALYSIS")
    print("=" * 100)

    for dim in dimensions:
        print(f"\n--- Dimension {dim} ---")
        stats = compute_all_statistics(all_results[dim], baseline="Random")

        for alg, s in stats.items():
            if alg.startswith("_"):
                continue
            print(
                f"{alg}: {s['mean']:.2f} ± {s['std']:.2f} (95% CI: [{s['ci_95_low']:.2f}, {s['ci_95_high']:.2f}])"
            )

        if "_comparisons" in stats:
            print("\nSignificance vs Random:")
            for alg, comp in stats["_comparisons"].items():
                sig = (
                    "***"
                    if comp["significant_001"]
                    else ("*" if comp["significant_005"] else "")
                )
                print(
                    f"  {alg}: d={comp['cohens_d']:.2f}, p={comp['p_value']:.4f} {sig}"
                )

    return all_results


def generate_summary(
    results: Dict, dimensions: List[int], algorithms: List[str]
) -> Dict:
    """Generate summary statistics."""
    summary = {}

    for dim in dimensions:
        summary[dim] = {}
        for alg in algorithms:
            if results[dim][alg]:
                metrics = {}
                for metric in [
                    "qd_score_at_budget",
                    "mean_fitness",
                    "uniformity_cv",
                    "archive_size",
                ]:
                    values = [
                        r["final_metrics"].get(metric, 0) for r in results[dim][alg]
                    ]
                    metrics[metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                    }
                summary[dim][alg] = metrics

    return summary


def print_summary_table(summary: Dict, dimensions: List[int], algorithms: List[str]):
    """Print publication-ready summary table."""
    print("\n" + "=" * 100)
    print("SUMMARY: QD-Score @ Budget (mean ± std)")
    print("=" * 100)

    header = f"{'Dim':<6}"
    for alg in algorithms:
        header += f" | {alg:<20}"
    print(header)
    print("-" + "=" * 99)

    for dim in dimensions:
        row = f"{dim:<6}"
        for alg in algorithms:
            if alg in summary.get(dim, {}):
                m = summary[dim][alg]["qd_score_at_budget"]
                row += f" | {m['mean']:>8.2f} ± {m['std']:<7.2f}"
            else:
                row += f" | {'N/A':^20}"
        print(row)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    if args.quick:
        run_dimensionality_scaling(
            dimensions=[5, 20],
            n_seeds=2,
            generations=200,
        )
    elif args.full:
        run_dimensionality_scaling(
            dimensions=[5, 10, 20, 50, 100],
            n_seeds=10,
            generations=2000,
        )
    else:
        run_dimensionality_scaling()
