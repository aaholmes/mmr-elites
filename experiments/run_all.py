#!/usr/bin/env python3
"""
MMR-Elites Full Experiment Suite
=================================

Run all experiments for the paper.

Usage:
    python experiments/run_all.py --quick      # Quick test (2 seeds, 200 gens)
    python experiments/run_all.py --full       # Full experiments (10 seeds, 2000 gens)
    python experiments/run_all.py --scaling    # Only dimensionality scaling
    python experiments/run_all.py --ablation   # Only lambda ablation
    python experiments/run_all.py --distance   # Only distance comparison
    python experiments/run_all.py --archsize   # Only archive size ablation
"""

import argparse
import pickle
import json
from pathlib import Path
from datetime import datetime

from experiments.dimensionality_scaling import run_dimensionality_scaling
from experiments.lambda_ablation import run_lambda_ablation
from experiments.distance_comparison import run_distance_comparison
from experiments.archive_size_ablation import run_archive_size_ablation


def main():
    parser = argparse.ArgumentParser(description="MMR-Elites Experiment Suite")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--full", action="store_true", help="Full experiment mode")
    parser.add_argument("--scaling", action="store_true", help="Run dimensionality scaling")
    parser.add_argument("--ablation", action="store_true", help="Run lambda ablation")
    parser.add_argument("--distance", action="store_true", help="Run distance comparison")
    parser.add_argument("--archsize", action="store_true", help="Run archive size ablation")
    parser.add_argument("--output-dir", default="results", help="Output directory")

    args = parser.parse_args()

    # Determine configuration
    if args.quick:
        n_seeds = 2
        generations = 200
    elif args.full:
        n_seeds = 10
        generations = 2000
    else:
        n_seeds = 5
        generations = 1000

    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # If no specific experiment selected, run all
    run_specific = args.scaling or args.ablation or args.distance or args.archsize
    run_all = not run_specific

    print("="*70)
    print("MMR-Elites Experiment Suite")
    print("="*70)
    print(f"Seeds: {n_seeds}")
    print(f"Generations: {generations}")
    print(f"Output: {run_dir}")
    print("="*70)

    results = {}

    # 1. Dimensionality scaling (THE key result)
    if run_all or args.scaling:
        print("\n[1/4] Running Dimensionality Scaling Experiment...")
        if args.quick:
            dims = [5, 20]
        else:
            dims = [5, 10, 20, 50, 100]
        results["dimensionality_scaling"] = run_dimensionality_scaling(
            dimensions=dims,
            n_seeds=n_seeds,
            generations=generations,
            output_dir=run_dir / "dimensionality_scaling",
        )

    # 2. Lambda ablation
    if run_all or args.ablation:
        print("\n[2/4] Running Lambda Ablation Experiment...")
        if args.quick:
            lambdas = [0.0, 0.5, 1.0]
        else:
            lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        results["lambda_ablation"] = run_lambda_ablation(
            lambda_values=lambdas,
            n_seeds=n_seeds,
            generations=generations,
            output_dir=run_dir / "lambda_ablation",
        )

    # 3. Distance function comparison
    if run_all or args.distance:
        print("\n[3/4] Running Distance Function Comparison...")
        if args.quick:
            dist_dims = [5, 20]
            dist_gens = min(generations, 200)
        else:
            dist_dims = [5, 10, 20, 50]
            dist_gens = min(generations, 1000)
        results["distance_comparison"] = run_distance_comparison(
            dimensions=dist_dims,
            n_seeds=n_seeds,
            generations=dist_gens,
            output_dir=run_dir / "distance_comparison",
        )

    # 4. Archive size ablation
    if run_all or args.archsize:
        print("\n[4/4] Running Archive Size Ablation...")
        if args.quick:
            sizes = [100, 500, 1000]
        else:
            sizes = [100, 500, 1000, 2000, 5000]
        results["archive_size_ablation"] = run_archive_size_ablation(
            archive_sizes=sizes,
            n_seeds=n_seeds,
            generations=generations,
            output_dir=run_dir / "archive_size_ablation",
        )

    # Save combined results
    print("\nSaving combined results...")
    with open(run_dir / "all_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\nAll experiments complete! Results in: {run_dir}")


if __name__ == "__main__":
    main()
