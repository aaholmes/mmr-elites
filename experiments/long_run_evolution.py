"""
Long-run evolution test: does diversity in the selection step improve
pure optimization performance over many generations?

Setup follows the classic MAP-Elites arm task (Mouret & Clune 2015 /
Cully et al. 2015): a 20-DOF planar arm reaching a target behind an
obstacle, with the 2-D end-effector position as the behavior descriptor.
The obstacle zeroes fitness on collision, making the landscape deceptive:
greedy fitness selection can converge on one side of the obstacle.

The key comparison is the lambda ablation. lambda=0 is pure top-K-by-fitness
evolution with the identical mutation operator and evaluation budget, so any
long-run gain in MAX fitness at lambda>0 is attributable to diversity in the
selection step alone (stepping stones). MAP-Elites on a 2-D grid is included
as the classic reference.

Usage:
    python experiments/long_run_evolution.py [--seeds 5] [--generations 2000]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mmr_elites.algorithms.map_elites import run_map_elites
from mmr_elites.algorithms.mmr_elites import run_mmr_elites
from mmr_elites.tasks.arm import ArmTask

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def make_task():
    # 2-D end-effector descriptor (the original MAP-Elites arm setup),
    # obstacle on (default): target at (0.8, 0) behind a wall at x=[0.5,0.55].
    return ArmTask(n_dof=20, use_highdim_descriptor=False)


def run(n_seeds: int, generations: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    configs = [(f"MMR lambda={lam}", ("mmr", lam)) for lam in LAMBDAS]
    configs.append(("MAP-Elites (32x32 grid)", ("map", None)))

    for name, (kind, lam) in configs:
        runs = []
        for seed in range(n_seeds):
            task = make_task()
            if kind == "mmr":
                r = run_mmr_elites(
                    task,
                    archive_size=1000,
                    generations=generations,
                    batch_size=200,
                    lambda_val=lam,
                    seed=seed,
                    log_interval=50,
                )
            else:
                # 32x32 = 1024 cells, comparable to K=1000
                r = run_map_elites(
                    task,
                    generations=generations,
                    batch_size=200,
                    bins_per_dim=32,
                    seed=seed,
                    log_interval=50,
                )
            runs.append(
                {
                    "seed": seed,
                    "final_max_fitness": r["final_metrics"]["max_fitness"],
                    "final_qd_at_budget": r["final_metrics"].get(
                        "qd_score_at_budget", r["final_metrics"]["qd_score"]
                    ),
                    "history_generation": r["history"]["generation"],
                    "history_max_fitness": r["history"]["max_fitness"],
                }
            )
            print(
                f"  {name:26s} seed={seed} "
                f"max_fitness={runs[-1]['final_max_fitness']:.4f}"
            )
        results[name] = runs

    with open(output_dir / "long_run_evolution.json", "w") as f:
        json.dump(results, f, indent=1)

    print("\n=== Final max fitness (mean +/- std over seeds) ===")
    for name, runs in results.items():
        vals = np.array([r["final_max_fitness"] for r in runs])
        print(f"{name:26s} {vals.mean():.4f} +/- {vals.std():.4f}")

    # Time-to-threshold: generations needed to reach 95% of the best
    # max-fitness any method achieved (NaN if never reached).
    best = max(r["final_max_fitness"] for runs in results.values() for r in runs)
    threshold = 0.95 * best
    print(f"\n=== Generations to reach {threshold:.4f} (95% of best) ===")
    for name, runs in results.items():
        gens = []
        for r in runs:
            hit = [
                g
                for g, mf in zip(r["history_generation"], r["history_max_fitness"])
                if mf >= threshold
            ]
            gens.append(hit[0] if hit else float("nan"))
        gens = np.array(gens, dtype=float)
        reached = np.sum(~np.isnan(gens))
        mean = np.nanmean(gens) if reached else float("nan")
        print(f"{name:26s} reached {reached}/{len(gens)} seeds, mean gen {mean:.0f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument(
        "--output", type=Path, default=Path("results/long_run_evolution")
    )
    args = parser.parse_args()
    run(args.seeds, args.generations, args.output)
