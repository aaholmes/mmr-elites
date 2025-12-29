#!/usr/bin/env python3
"""
MMR-Elites GECCO Experiment Suite
=================================

Run all experiments for the GECCO paper.

Usage:
    python experiments/run_all.py --quick      # Quick test (2 seeds, 200 gens)
    python experiments/run_all.py --full       # Full experiments (10 seeds, 2000 gens)
    python experiments/run_all.py --scaling    # Only dimensionality scaling
    python experiments/run_all.py --ablation   # Only lambda ablation
"""

import argparse
import pickle
import json
from pathlib import Path
from datetime import datetime

from experiments.dimensionality_scaling import run_dimensionality_scaling
from experiments.lambda_ablation import run_lambda_ablation


def main():
    parser = argparse.ArgumentParser(description="MMR-Elites Experiment Suite")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--full", action="store_true", help="Full experiment mode")
    parser.add_argument("--scaling", action="store_true", help="Run dimensionality scaling")
    parser.add_argument("--ablation", action="store_true", help="Run lambda ablation")
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
    
    print("="*70)
    print("MMR-Elites GECCO Experiment Suite")
    print("="*70)
    print(f"Seeds: {n_seeds}")
    print(f"Generations: {generations}")
    print(f"Output: {run_dir}")
    print("="*70)
    
    results = {}
    
    # Run experiments
    if args.scaling or (not args.scaling and not args.ablation):
        print("\n📊 Running Dimensionality Scaling Experiment...")
        results["dimensionality_scaling"] = run_dimensionality_scaling(
            n_seeds=n_seeds,
            generations=generations,
            output_dir=run_dir / "dimensionality_scaling",
        )
    
    if args.ablation or (not args.scaling and not args.ablation):
        print("\n📊 Running Lambda Ablation Experiment...")
        results["lambda_ablation"] = run_lambda_ablation(
            n_seeds=n_seeds,
            generations=generations,
            output_dir=run_dir / "lambda_ablation",
        )
    
    # Save summary
    print("\n💾 Saving results...")
    with open(run_dir / "all_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print(f"\n✅ All experiments complete! Results in: {run_dir}")


if __name__ == "__main__":
    main()
