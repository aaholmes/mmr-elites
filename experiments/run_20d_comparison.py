#!/usr/bin/env python3
"""
20D Arm comparison: MMR-Elites vs MAP-Elites vs CVT-MAP-Elites vs Random.

Usage:
    python experiments/run_20d_comparison.py --seeds 5 --gens 1000
    python experiments/run_20d_comparison.py --seeds 2 --gens 100  # quick test
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from mmr_elites.algorithms import (
    run_cvt_map_elites,
    run_map_elites,
    run_mmr_elites,
    run_random_search,
)
from mmr_elites.tasks.arm import ArmTask


def main():
    parser = argparse.ArgumentParser(description="20D Arm Comparison")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--gens", type=int, default=1000)
    parser.add_argument("--archive-size", type=int, default=1000)
    parser.add_argument("--lambda-val", type=float, default=0.5)
    parser.add_argument("--output-dir", default="results/20d_comparison")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    task = ArmTask(n_dof=20, use_highdim_descriptor=True)
    all_results = {"MMR-Elites": [], "MAP-Elites": [], "CVT-MAP-Elites": [], "Random": []}

    for seed in range(args.seeds):
        print(f"\n--- Seed {seed + 1}/{args.seeds} ---")

        r = run_mmr_elites(task, archive_size=args.archive_size, generations=args.gens,
                           lambda_val=args.lambda_val, seed=seed)
        all_results["MMR-Elites"].append(r)
        print(f"  MMR-Elites:     QD={r['final_metrics']['qd_score']:.2f}")

        r = run_map_elites(task, generations=args.gens, seed=seed)
        all_results["MAP-Elites"].append(r)
        print(f"  MAP-Elites:     QD={r['final_metrics']['qd_score']:.2f}")

        r = run_cvt_map_elites(task, n_niches=args.archive_size, generations=args.gens, seed=seed)
        all_results["CVT-MAP-Elites"].append(r)
        print(f"  CVT-MAP-Elites: QD={r['final_metrics']['qd_score']:.2f}")

        r = run_random_search(task, archive_size=args.archive_size, generations=args.gens, seed=seed)
        all_results["Random"].append(r)
        print(f"  Random:         QD={r['final_metrics']['qd_score']:.2f}")

    with open(output_path / "all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    # Print summary
    print(f"\n{'Algorithm':<20} {'QD-Score':>12}")
    print("-" * 35)
    for alg, runs in all_results.items():
        vals = [r["final_metrics"]["qd_score"] for r in runs]
        print(f"{alg:<20} {np.mean(vals):>7.1f} +/- {np.std(vals):.1f}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
