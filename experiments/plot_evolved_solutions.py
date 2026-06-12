"""
Visualize the best evolved arm at each wall height.

For each obstacle size we evolve with lambda=1 (diversity-driven, which
solves the deceptive task) and lambda=0 (pure fitness, which gets stuck),
then draw the best arm from each against the wall and target. This makes
the lambda-deception result concrete: you can see the fitness-only arm
pinned flat against the wall while the diversity-driven arm reaches over it.

Usage:
    python experiments/plot_evolved_solutions.py [--generations 2000] [--seed 0]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from mmr_elites.algorithms.mmr_elites import run_mmr_elites
from mmr_elites.tasks.arm import ArmTask

WALLS = [
    (None, "No wall"),
    (0.1, "Short wall"),
    (0.25, "Medium wall"),
    (0.4, "Tall wall"),
]
TARGET = (0.8, 0.0)


def make_task(half_height):
    obstacle = None if half_height is None else (0.5, 0.55, -half_height, half_height)
    return ArmTask(
        n_dof=20, target_pos=TARGET, use_highdim_descriptor=False, obstacle=obstacle
    )


def best_arm(task, lam, generations, seed):
    """Evolve and return joint coords (including origin) of the best solution."""
    r = run_mmr_elites(
        task,
        archive_size=1000,
        generations=generations,
        batch_size=200,
        lambda_val=lam,
        seed=seed,
        log_interval=generations,
    )
    genomes, fitness = r["final_genomes"], r["final_fitness"]
    best = genomes[np.argmax(fitness)]
    coords = task.forward_kinematics_batch(best[None])[0]  # (n_dof, 2)
    coords = np.vstack([[0.0, 0.0], coords])  # prepend the origin
    return coords, float(np.max(fitness))


def main(generations, seed, output):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for ax, (half_height, title) in zip(axes.flat, WALLS):
        task = make_task(half_height)
        arm_div, fit_div = best_arm(task, 1.0, generations, seed)
        arm_fit, fit_fit = best_arm(task, 0.0, generations, seed)

        # Wall
        if half_height is not None:
            ax.add_patch(
                mpatches.Rectangle(
                    (0.5, -half_height),
                    0.05,
                    2 * half_height,
                    facecolor="0.4",
                    edgecolor="black",
                    zorder=1,
                )
            )

        # Arms
        ax.plot(
            arm_fit[:, 0],
            arm_fit[:, 1],
            "-",
            color="#d62728",
            lw=2,
            zorder=3,
            label=f"λ=0 fitness-only (fit={fit_fit:.2f})",
        )
        ax.plot(
            arm_div[:, 0],
            arm_div[:, 1],
            "-",
            color="#2ca02c",
            lw=2,
            zorder=4,
            label=f"λ=1 diversity-driven (fit={fit_div:.2f})",
        )
        # Base and target
        ax.plot(0, 0, "ks", ms=8, zorder=5)
        ax.plot(*TARGET, "*", color="gold", ms=20, mec="black", zorder=5)

        ax.set_title(title)
        ax.set_xlim(-0.2, 1.0)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect("equal")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Best evolved arm by wall height: fitness-only stalls at the wall,\n"
        "diversity-driven selection reaches over it "
        "(target = gold star, base = black square)",
        fontsize=12,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/figures/evolved_solutions.png")
    )
    args = parser.parse_args()
    main(args.generations, args.seed, args.output)
