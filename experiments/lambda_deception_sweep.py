"""
Lambda x deception sweep: how does the best fitness/diversity balance shift
as a task becomes more deceptive?

This is a direct, continuous-knob test of the "Abandoning Objectives"
hypothesis (Lehman & Stanley 2011): when the objective is a poor compass,
searching for behavioral novelty finds the objective faster than optimizing
for it. MMR-Elites' lambda makes that a dial rather than a dichotomy --
lambda=0 is pure fitness (objective-driven), lambda=1 is fitness-blind
novelty (plus retention of the single best-so-far individual).

Task: the 20-DOF planar arm reaching a target behind a wall (2-D
end-effector descriptor). Deception is controlled by the wall half-height:
a taller wall makes the greedy "go straight at the target" path a deeper
trap, because the arm must detour further before it can arc over.

For each (wall height, lambda) we report final max fitness averaged over
seeds, and identify lambda* = the lambda maximizing it. The prediction:
lambda* increases toward 1 as deception grows; with no wall (non-deceptive)
a lower lambda that keeps some fitness pressure should do at least as well.

Usage:
    python experiments/lambda_deception_sweep.py [--seeds 5] [--generations 2000]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mmr_elites.algorithms.mmr_elites import run_mmr_elites
from mmr_elites.tasks.arm import ArmTask

LAMBDAS = [0.0, 0.5, 0.8, 0.9, 0.95, 1.0]
# Wall half-heights. None = no obstacle (non-deceptive control).
WALL_HALF_HEIGHTS = [None, 0.1, 0.25, 0.4]


def make_task(wall_half_height):
    if wall_half_height is None:
        obstacle = None
    else:
        # Wall spans x in [0.5, 0.55], y in [-h, h]; target at (0.8, 0).
        obstacle = (0.5, 0.55, -wall_half_height, wall_half_height)
    return ArmTask(
        n_dof=20,
        target_pos=(0.8, 0.0),
        use_highdim_descriptor=False,
        obstacle=obstacle,
    )


def run(n_seeds: int, generations: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = {}  # (wall, lambda) -> [max_fitness per seed]

    for wall in WALL_HALF_HEIGHTS:
        for lam in LAMBDAS:
            vals = []
            for seed in range(n_seeds):
                task = make_task(wall)
                r = run_mmr_elites(
                    task,
                    archive_size=1000,
                    generations=generations,
                    batch_size=200,
                    lambda_val=lam,
                    seed=seed,
                    log_interval=200,
                )
                vals.append(float(r["final_metrics"]["max_fitness"]))
            grid[f"{wall}|{lam}"] = vals
            print(
                f"  wall={str(wall):>5} lambda={lam:<4} "
                f"max_fitness={np.mean(vals):.4f} +/- {np.std(vals):.4f}"
            )

    with open(output_dir / "lambda_deception_sweep.json", "w") as f:
        json.dump(
            {"lambdas": LAMBDAS, "walls": WALL_HALF_HEIGHTS, "grid": grid},
            f,
            indent=1,
        )

    # Summary table + lambda* per wall
    print("\n=== Final max fitness by (wall half-height, lambda) ===")
    header = "wall      " + "".join(f"λ={l:<5}" for l in LAMBDAS) + " | λ*"
    print(header)
    for wall in WALL_HALF_HEIGHTS:
        means = [np.mean(grid[f"{wall}|{l}"]) for l in LAMBDAS]
        best_lambda = LAMBDAS[int(np.argmax(means))]
        label = "none " if wall is None else f"{wall:<5}"
        row = f"{label}    " + "".join(f"{m:<7.3f}" for m in means)
        print(f"{row} | {best_lambda}")

    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument(
        "--output", type=Path, default=Path("results/lambda_deception_sweep")
    )
    args = parser.parse_args()
    run(args.seeds, args.generations, args.output)
