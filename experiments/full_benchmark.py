#!/usr/bin/env python3
"""
MMR-Elites: Full Benchmark Suite
=================================================

Paper: "Quality-Diversity as Information Retrieval: Overcoming the Curse of
        Dimensionality with Maximum Marginal Relevance Selection of Elites"

This script runs all experiments needed for the paper:
1. Dimensionality scaling (5, 10, 20, 50, 100-DOF arms)
2. Standard QD benchmarks (Arm, Rastrigin)
3. Ablation studies (λ, K)
4. Runtime comparison

Hardware: AMD 7950X + 128GB RAM (GPU optional)

Usage:
    python full_benchmark.py --experiment all        # Everything (overnight)
    python full_benchmark.py --experiment scaling    # Just dimensionality scaling
    python full_benchmark.py --experiment ablation   # Just ablations
    python full_benchmark.py --quick                 # Quick sanity check
"""

import argparse
import json
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

# Try imports
try:
    import mmr_elites_rs

    RUST_AVAILABLE = True
except ImportError:
    print("⚠️  Rust backend not found. Run: maturin develop --release")
    RUST_AVAILABLE = False

try:
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  sklearn not found. CVT-MAP-Elites will use fallback.")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""

    name: str
    n_seeds: int = 10
    generations: int = 2000
    archive_size: int = 1000
    batch_size: int = 200
    mutation_sigma: float = 0.1
    lambda_val: float = 0.5
    log_interval: int = 100
    output_dir: str = "results"


# =============================================================================
# Metrics (Corrected for Fair Comparison)
# =============================================================================


def compute_metrics(
    fitness: np.ndarray, descriptors: np.ndarray, budget_k: int = 1000
) -> Dict[str, float]:
    """
    Compute QD metrics with proper normalization.

    Key insight: QD-Score alone is misleading when archive sizes differ.
    We add budget-normalized metrics for fair comparison.
    """
    n = len(fitness)
    if n == 0:
        return {
            k: 0.0
            for k in [
                "qd_score",
                "qd_score_at_budget",
                "max_fitness",
                "mean_fitness",
                "mean_pairwise_distance",
                "uniformity_cv",
                "archive_size",
                "coverage_efficiency",
            ]
        }

    # Sort fitness for budget-constrained metrics
    sorted_fit = np.sort(fitness)[::-1]

    # Basic metrics
    qd_score = float(np.sum(fitness))
    qd_at_budget = float(np.sum(sorted_fit[:budget_k]))  # Best K solutions
    max_fit = float(np.max(fitness))
    mean_fit = float(np.mean(fitness))

    # Diversity metrics
    if n > 1:
        # Mean pairwise distance
        dists = cdist(descriptors, descriptors)
        mpd = float(np.mean(dists[np.triu_indices(n, k=1)]))

        # Uniformity (CV of k-NN distances) - LOWER IS BETTER
        k = min(5, n - 1)
        if k > 0:
            tree = cKDTree(descriptors)
            knn_dists, _ = tree.query(descriptors, k=k + 1)
            mean_knn = np.mean(knn_dists[:, 1:], axis=1)
            uniformity_cv = float(np.std(mean_knn) / (np.mean(mean_knn) + 1e-10))
        else:
            uniformity_cv = 0.0
    else:
        mpd = 0.0
        uniformity_cv = 0.0

    # Coverage efficiency: QD-Score per solution (higher = more efficient)
    coverage_efficiency = qd_score / n if n > 0 else 0.0

    return {
        "qd_score": qd_score,
        "qd_score_at_budget": qd_at_budget,  # Fair comparison metric
        "max_fitness": max_fit,
        "mean_fitness": mean_fit,
        "mean_pairwise_distance": mpd,
        "uniformity_cv": uniformity_cv,  # Lower = more uniform
        "archive_size": n,
        "coverage_efficiency": coverage_efficiency,
    }


# =============================================================================
# Tasks
# =============================================================================


class ArmTask:
    """
    N-DOF Planar Arm with configurable dimensionality.

    This is the KEY task for demonstrating MMR-Elites' advantage.
    By varying DOF, we show how MAP-Elites degrades with dimensionality.
    """

    def __init__(
        self,
        n_dof: int = 20,
        target_pos: Tuple[float, float] = (0.8, 0.0),
        use_highdim_descriptor: bool = True,
    ):
        self.n_dof = n_dof
        self.link_length = 1.0 / n_dof
        self.target_pos = np.array(target_pos)
        self.use_highdim_descriptor = use_highdim_descriptor

        # Obstacle
        self.box_x = [0.5, 0.55]
        self.box_y = [-0.25, 0.25]

        # Descriptor bounds
        self.desc_dim = n_dof if use_highdim_descriptor else 2
        self.desc_min = np.zeros(self.desc_dim)
        self.desc_max = np.ones(self.desc_dim)

    def forward_kinematics_batch(self, joints: np.ndarray) -> np.ndarray:
        """Compute joint positions for batch of configurations."""
        angles = np.cumsum(joints, axis=1)
        dx = self.link_length * np.cos(angles)
        dy = self.link_length * np.sin(angles)
        x = np.cumsum(dx, axis=1)
        y = np.cumsum(dy, axis=1)
        return np.stack([x, y], axis=2)

    def check_collisions_batch(self, joint_coords: np.ndarray) -> np.ndarray:
        """Check collisions with obstacle."""
        batch_size = joint_coords.shape[0]
        origin = np.zeros((batch_size, 1, 2))
        points = np.concatenate([origin, joint_coords], axis=1)

        # Point inside box
        p_in_x = (points[:, :, 0] > self.box_x[0]) & (points[:, :, 0] < self.box_x[1])
        p_in_y = (points[:, :, 1] > self.box_y[0]) & (points[:, :, 1] < self.box_y[1])
        any_inside = np.any(p_in_x & p_in_y, axis=1)

        # Line segment intersections
        A, B = points[:, :-1, :], points[:, 1:, :]
        Ax, Ay, Bx, By = A[:, :, 0], A[:, :, 1], B[:, :, 0], B[:, :, 1]
        dx, dy = Bx - Ax, By - Ay
        dx = np.where(np.abs(dx) < 1e-9, 1e-9, dx)
        dy = np.where(np.abs(dy) < 1e-9, 1e-9, dy)

        def check_v(wx, ymin, ymax):
            t = (wx - Ax) / dx
            return (t >= 0) & (t <= 1) & (Ay + t * dy >= ymin) & (Ay + t * dy <= ymax)

        def check_h(wy, xmin, xmax):
            t = (wy - Ay) / dy
            return (t >= 0) & (t <= 1) & (Ax + t * dx >= xmin) & (Ax + t * dx <= xmax)

        hit = (
            check_v(self.box_x[0], *self.box_y)
            | check_v(self.box_x[1], *self.box_y)
            | check_h(self.box_y[0], *self.box_x)
            | check_h(self.box_y[1], *self.box_x)
        )

        return any_inside | np.any(hit, axis=1)

    def evaluate(self, genomes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate batch of genomes."""
        joint_coords = self.forward_kinematics_batch(genomes)
        tips = joint_coords[:, -1, :]

        # Fitness: distance to target
        dists = np.linalg.norm(tips - self.target_pos, axis=1)
        fitness = np.maximum(1.0 - dists, 0.0)

        # Collision penalty
        fitness[self.check_collisions_batch(joint_coords)] = 0.0

        # Descriptor: joint angles (high-D) or end-effector (2D)
        if self.use_highdim_descriptor:
            descriptors = (genomes + np.pi) / (2 * np.pi)  # Normalize to [0,1]
        else:
            # 2D: end-effector position normalized
            descriptors = (tips + 1.5) / 3.0  # Assuming range [-1.5, 1.5]
            descriptors = np.clip(descriptors, 0, 1)

        return fitness, descriptors


class RastriginTask:
    """
    Rastrigin function - standard optimization benchmark.

    Good for testing because:
    - Scalable to any dimension
    - Known global optimum
    - Many local optima (tests diversity)
    """

    def __init__(self, n_dim: int = 20, bounds: float = 5.12):
        self.n_dim = n_dim
        self.bounds = bounds
        self.desc_min = np.zeros(n_dim)
        self.desc_max = np.ones(n_dim)

    def evaluate(self, genomes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate Rastrigin function.

        f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
        Global minimum: f(0) = 0
        """
        # Scale genomes from [-π, π] to [-bounds, bounds]
        x = genomes * (self.bounds / np.pi)

        # Rastrigin function (we negate for maximization)
        A = 10
        f = A * self.n_dim + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)

        # Convert to fitness (higher is better), normalize roughly to [0, 1]
        max_val = A * self.n_dim + self.n_dim * (self.bounds**2 + A)
        fitness = 1.0 - (f / max_val)
        fitness = np.clip(fitness, 0, 1)

        # Descriptor: normalized position in search space
        descriptors = (genomes + np.pi) / (2 * np.pi)

        return fitness, descriptors


# =============================================================================
# Algorithms
# =============================================================================


def run_mmr_elites(
    task, config: ExperimentConfig, seed: int, lambda_val: Optional[float] = None
) -> Dict:
    """Run MMR-Elites algorithm."""
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust backend required for MMR-Elites")

    lam = lambda_val if lambda_val is not None else config.lambda_val
    np.random.seed(seed)

    selector = mmr_elites_rs.MMRSelector(config.archive_size, lam)

    # Initialize
    n_dof = task.n_dof if hasattr(task, "n_dof") else task.n_dim
    archive = np.random.uniform(-np.pi, np.pi, (config.archive_size, n_dof))
    fit, desc = task.evaluate(archive)

    idx = selector.select(fit, desc)
    archive, fit, desc = archive[idx], fit[idx], desc[idx]

    history = {
        k: []
        for k in [
            "generation",
            "qd_score",
            "qd_score_at_budget",
            "max_fitness",
            "mean_pairwise_distance",
            "uniformity_cv",
            "archive_size",
        ]
    }

    start = time.time()

    for gen in range(1, config.generations + 1):
        # Mutation
        parents = archive[np.random.randint(0, len(archive), config.batch_size)]
        offspring = np.clip(
            parents
            + np.random.normal(0, config.mutation_sigma, (config.batch_size, n_dof)),
            -np.pi,
            np.pi,
        )

        off_fit, off_desc = task.evaluate(offspring)

        # Selection
        pool = np.vstack([archive, offspring])
        pool_fit = np.concatenate([fit, off_fit])
        pool_desc = np.vstack([desc, off_desc])

        idx = selector.select(pool_fit, pool_desc)
        archive, fit, desc = pool[idx], pool_fit[idx], pool_desc[idx]

        # Logging
        if gen % config.log_interval == 0:
            m = compute_metrics(fit, desc, config.archive_size)
            history["generation"].append(gen)
            for k in history:
                if k != "generation":
                    history[k].append(m.get(k, 0))

    return {
        "algorithm": "MMR-Elites",
        "seed": seed,
        "runtime": time.time() - start,
        "final_metrics": compute_metrics(fit, desc, config.archive_size),
        "history": history,
        "final_descriptors": desc,
        "final_fitness": fit,
    }


def run_map_elites(
    task, config: ExperimentConfig, seed: int, bins_per_dim: int = 3
) -> Dict:
    """Run sparse MAP-Elites."""
    np.random.seed(seed)

    n_dof = task.n_dof if hasattr(task, "n_dof") else task.n_dim
    desc_dim = task.desc_dim if hasattr(task, "desc_dim") else n_dof

    def get_cell(desc):
        idx = (desc * bins_per_dim).astype(int)
        return tuple(np.clip(idx, 0, bins_per_dim - 1))

    archive = {}

    # Initialize
    init_pop = np.random.uniform(-np.pi, np.pi, (config.batch_size * 5, n_dof))
    fit, desc = task.evaluate(init_pop)

    for i in range(len(init_pop)):
        cell = get_cell(desc[i])
        if cell not in archive or fit[i] > archive[cell][1]:
            archive[cell] = (init_pop[i].copy(), fit[i], desc[i].copy())

    history = {
        k: []
        for k in [
            "generation",
            "qd_score",
            "qd_score_at_budget",
            "max_fitness",
            "mean_pairwise_distance",
            "uniformity_cv",
            "archive_size",
        ]
    }

    start = time.time()

    for gen in range(1, config.generations + 1):
        keys = list(archive.keys())
        parents = np.array(
            [
                archive[keys[np.random.randint(len(keys))]][0]
                for _ in range(config.batch_size)
            ]
        )

        offspring = np.clip(
            parents + np.random.normal(0, config.mutation_sigma, parents.shape),
            -np.pi,
            np.pi,
        )

        off_fit, off_desc = task.evaluate(offspring)

        for i in range(len(offspring)):
            cell = get_cell(off_desc[i])
            if cell not in archive or off_fit[i] > archive[cell][1]:
                archive[cell] = (offspring[i].copy(), off_fit[i], off_desc[i].copy())

        if gen % config.log_interval == 0:
            all_fit = np.array([v[1] for v in archive.values()])
            all_desc = np.array([v[2] for v in archive.values()])
            m = compute_metrics(all_fit, all_desc, config.archive_size)
            history["generation"].append(gen)
            for k in history:
                if k != "generation":
                    history[k].append(m.get(k, 0))

    all_fit = np.array([v[1] for v in archive.values()])
    all_desc = np.array([v[2] for v in archive.values()])

    return {
        "algorithm": "MAP-Elites",
        "seed": seed,
        "runtime": time.time() - start,
        "final_metrics": compute_metrics(all_fit, all_desc, config.archive_size),
        "history": history,
        "final_descriptors": all_desc,
        "final_fitness": all_fit,
    }


def run_cvt_map_elites(task, config: ExperimentConfig, seed: int) -> Dict:
    """Run CVT-MAP-Elites."""
    np.random.seed(seed)

    n_dof = task.n_dof if hasattr(task, "n_dof") else task.n_dim
    desc_dim = task.desc_dim if hasattr(task, "desc_dim") else n_dof

    # Compute CVT centroids
    samples = np.random.uniform(0, 1, (50000, desc_dim))
    if SKLEARN_AVAILABLE:
        centroids = (
            KMeans(
                n_clusters=config.archive_size, random_state=seed, n_init=1, max_iter=50
            )
            .fit(samples)
            .cluster_centers_
        )
    else:
        idx = np.random.choice(len(samples), config.archive_size, replace=False)
        centroids = samples[idx]

    tree = cKDTree(centroids)
    archive = {}

    def get_niche(desc):
        _, idx = tree.query(desc)
        return int(idx)

    # Initialize
    init_pop = np.random.uniform(-np.pi, np.pi, (config.batch_size * 5, n_dof))
    fit, desc = task.evaluate(init_pop)

    for i in range(len(init_pop)):
        niche = get_niche(desc[i])
        if niche not in archive or fit[i] > archive[niche][1]:
            archive[niche] = (init_pop[i].copy(), fit[i], desc[i].copy())

    history = {
        k: []
        for k in [
            "generation",
            "qd_score",
            "qd_score_at_budget",
            "max_fitness",
            "mean_pairwise_distance",
            "uniformity_cv",
            "archive_size",
        ]
    }

    start = time.time()

    for gen in range(1, config.generations + 1):
        keys = list(archive.keys())
        if not keys:
            continue

        parents = np.array(
            [
                archive[keys[np.random.randint(len(keys))]][0]
                for _ in range(config.batch_size)
            ]
        )

        offspring = np.clip(
            parents + np.random.normal(0, config.mutation_sigma, parents.shape),
            -np.pi,
            np.pi,
        )

        off_fit, off_desc = task.evaluate(offspring)

        for i in range(len(offspring)):
            niche = get_niche(off_desc[i])
            if niche not in archive or off_fit[i] > archive[niche][1]:
                archive[niche] = (offspring[i].copy(), off_fit[i], off_desc[i].copy())

        if gen % config.log_interval == 0:
            all_fit = np.array([v[1] for v in archive.values()])
            all_desc = np.array([v[2] for v in archive.values()])
            m = compute_metrics(all_fit, all_desc, config.archive_size)
            history["generation"].append(gen)
            for k in history:
                if k != "generation":
                    history[k].append(m.get(k, 0))

    all_fit = np.array([v[1] for v in archive.values()])
    all_desc = np.array([v[2] for v in archive.values()])

    return {
        "algorithm": "CVT-MAP-Elites",
        "seed": seed,
        "runtime": time.time() - start,
        "final_metrics": compute_metrics(all_fit, all_desc, config.archive_size),
        "history": history,
        "final_descriptors": all_desc,
        "final_fitness": all_fit,
    }


def run_random_search(task, config: ExperimentConfig, seed: int) -> Dict:
    """Random search baseline (sanity check)."""
    np.random.seed(seed)

    n_dof = task.n_dof if hasattr(task, "n_dof") else task.n_dim

    history = {
        k: []
        for k in [
            "generation",
            "qd_score",
            "qd_score_at_budget",
            "max_fitness",
            "mean_pairwise_distance",
            "uniformity_cv",
            "archive_size",
        ]
    }

    all_genomes = []
    all_fit = []
    all_desc = []

    start = time.time()

    for gen in range(1, config.generations + 1):
        # Generate random solutions
        new_genomes = np.random.uniform(-np.pi, np.pi, (config.batch_size, n_dof))
        fit, desc = task.evaluate(new_genomes)

        all_genomes.extend(new_genomes)
        all_fit.extend(fit)
        all_desc.extend(desc)

        if gen % config.log_interval == 0:
            # Keep best K
            fit_arr = np.array(all_fit)
            desc_arr = np.array(all_desc)

            if len(fit_arr) > config.archive_size:
                top_idx = np.argsort(fit_arr)[-config.archive_size :]
                fit_arr = fit_arr[top_idx]
                desc_arr = desc_arr[top_idx]

            m = compute_metrics(fit_arr, desc_arr, config.archive_size)
            history["generation"].append(gen)
            for k in history:
                if k != "generation":
                    history[k].append(m.get(k, 0))

    # Final: keep best K
    fit_arr = np.array(all_fit)
    desc_arr = np.array(all_desc)
    top_idx = np.argsort(fit_arr)[-config.archive_size :]

    return {
        "algorithm": "Random",
        "seed": seed,
        "runtime": time.time() - start,
        "final_metrics": compute_metrics(
            fit_arr[top_idx], desc_arr[top_idx], config.archive_size
        ),
        "history": history,
        "final_descriptors": desc_arr[top_idx],
        "final_fitness": fit_arr[top_idx],
    }


# =============================================================================
# Experiments
# =============================================================================


def run_dimensionality_scaling(
    output_dir: Path, n_seeds: int = 10, quick: bool = False
):
    """
    Experiment 1: How do algorithms scale with behavior space dimensionality?

    This is THE key experiment for the paper.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Dimensionality Scaling")
    print("=" * 70)

    dimensions = [5, 10, 20, 50, 100] if not quick else [5, 20]
    generations = 500 if quick else 2000
    seeds = 2 if quick else n_seeds

    results = {
        d: {"MMR-Elites": [], "MAP-Elites": [], "CVT-MAP-Elites": [], "Random": []}
        for d in dimensions
    }

    for n_dof in dimensions:
        print(f"\n--- {n_dof}-DOF Arm ---")

        task = ArmTask(n_dof=n_dof, use_highdim_descriptor=True)
        config = ExperimentConfig(
            name=f"arm_{n_dof}dof",
            n_seeds=seeds,
            generations=generations,
            archive_size=1000,
            log_interval=50 if quick else 100,
        )

        for seed in range(seeds):
            print(f"  Seed {seed+1}/{seeds}:")

            # MMR-Elites
            if RUST_AVAILABLE:
                print("    MMR-Elites...", end=" ", flush=True)
                r = run_mmr_elites(task, config, seed)
                results[n_dof]["MMR-Elites"].append(r)
                print(f"QD@K={r['final_metrics']['qd_score_at_budget']:.1f}")

            # MAP-Elites
            print("    MAP-Elites...", end=" ", flush=True)
            r = run_map_elites(task, config, seed, bins_per_dim=3)
            results[n_dof]["MAP-Elites"].append(r)
            print(
                f"QD@K={r['final_metrics']['qd_score_at_budget']:.1f}, "
                f"Size={r['final_metrics']['archive_size']:.0f}"
            )

            # CVT-MAP-Elites
            print("    CVT-MAP-Elites...", end=" ", flush=True)
            r = run_cvt_map_elites(task, config, seed)
            results[n_dof]["CVT-MAP-Elites"].append(r)
            print(f"QD@K={r['final_metrics']['qd_score_at_budget']:.1f}")

            # Random
            print("    Random...", end=" ", flush=True)
            r = run_random_search(task, config, seed)
            results[n_dof]["Random"].append(r)
            print(f"QD@K={r['final_metrics']['qd_score_at_budget']:.1f}")

    # Save
    with open(output_dir / "dimensionality_scaling.pkl", "wb") as f:
        pickle.dump(results, f)

    # Print summary table
    print("\n" + "=" * 90)
    print("DIMENSIONALITY SCALING RESULTS (QD-Score @ Budget K=1000)")
    print("=" * 90)

    header = f"{'DOF':<8}"
    for alg in ["MMR-Elites", "MAP-Elites", "CVT-MAP-Elites", "Random"]:
        header += f" | {alg:<18}"
    print(header)
    print("-" * 90)

    for d in dimensions:
        row = f"{d:<8}"
        for alg in ["MMR-Elites", "MAP-Elites", "CVT-MAP-Elites", "Random"]:
            if results[d][alg]:
                vals = [
                    r["final_metrics"]["qd_score_at_budget"] for r in results[d][alg]
                ]
                row += f" | {np.mean(vals):>7.1f} ± {np.std(vals):<6.1f}"
            else:
                row += f" | {'N/A':^18}"
        print(row)

    return results


def run_lambda_ablation(output_dir: Path, n_seeds: int = 5, quick: bool = False):
    """
    Experiment 2: Effect of λ parameter on MMR-Elites.

    λ = 0: Pure fitness (greedy)
    λ = 1: Pure diversity (novelty search)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Lambda Ablation")
    print("=" * 70)

    lambda_values = (
        [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] if not quick else [0.0, 0.5, 1.0]
    )
    generations = 500 if quick else 1000
    seeds = 2 if quick else n_seeds

    if not RUST_AVAILABLE:
        print("⚠️  Rust backend required for ablation study")
        return {}

    task = ArmTask(n_dof=20, use_highdim_descriptor=True)
    config = ExperimentConfig(
        name="lambda_ablation",
        n_seeds=seeds,
        generations=generations,
        archive_size=1000,
        log_interval=50,
    )

    results = {lam: [] for lam in lambda_values}

    for lam in lambda_values:
        print(f"\n--- λ = {lam} ---")

        for seed in range(seeds):
            print(f"  Seed {seed+1}/{seeds}...", end=" ", flush=True)
            r = run_mmr_elites(task, config, seed, lambda_val=lam)
            results[lam].append(r)
            print(
                f"QD={r['final_metrics']['qd_score']:.1f}, "
                f"MaxFit={r['final_metrics']['max_fitness']:.4f}, "
                f"Uniformity={r['final_metrics']['uniformity_cv']:.3f}"
            )

    # Save
    with open(output_dir / "lambda_ablation.pkl", "wb") as f:
        pickle.dump(results, f)

    # Print summary
    print("\n" + "=" * 80)
    print("LAMBDA ABLATION RESULTS")
    print("=" * 80)
    print(f"{'λ':<8} | {'QD-Score':<18} | {'Max Fitness':<18} | {'Uniformity CV':<18}")
    print("-" * 80)

    for lam in lambda_values:
        qd = [r["final_metrics"]["qd_score"] for r in results[lam]]
        mf = [r["final_metrics"]["max_fitness"] for r in results[lam]]
        uv = [r["final_metrics"]["uniformity_cv"] for r in results[lam]]
        print(
            f"{lam:<8.1f} | {np.mean(qd):>7.1f} ± {np.std(qd):<6.1f} | "
            f"{np.mean(mf):>7.4f} ± {np.std(mf):<6.4f} | "
            f"{np.mean(uv):>7.4f} ± {np.std(uv):<6.4f}"
        )

    return results


def run_archive_size_ablation(output_dir: Path, n_seeds: int = 5, quick: bool = False):
    """
    Experiment 3: Effect of archive size K.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Archive Size Ablation")
    print("=" * 70)

    archive_sizes = [100, 500, 1000, 2000, 5000] if not quick else [100, 1000]
    generations = 500 if quick else 1000
    seeds = 2 if quick else n_seeds

    task = ArmTask(n_dof=20, use_highdim_descriptor=True)

    results = {k: {"MMR-Elites": [], "CVT-MAP-Elites": []} for k in archive_sizes}

    for k in archive_sizes:
        print(f"\n--- K = {k} ---")

        config = ExperimentConfig(
            name=f"archive_size_{k}",
            n_seeds=seeds,
            generations=generations,
            archive_size=k,
            log_interval=50,
        )

        for seed in range(seeds):
            print(f"  Seed {seed+1}/{seeds}:")

            if RUST_AVAILABLE:
                print("    MMR-Elites...", end=" ", flush=True)
                r = run_mmr_elites(task, config, seed)
                results[k]["MMR-Elites"].append(r)
                print(f"QD={r['final_metrics']['qd_score']:.1f}")

            print("    CVT-MAP-Elites...", end=" ", flush=True)
            r = run_cvt_map_elites(task, config, seed)
            results[k]["CVT-MAP-Elites"].append(r)
            print(f"QD={r['final_metrics']['qd_score']:.1f}")

    with open(output_dir / "archive_size_ablation.pkl", "wb") as f:
        pickle.dump(results, f)

    return results


def run_runtime_comparison(output_dir: Path):
    """
    Experiment 4: Runtime comparison across algorithms.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Runtime Comparison")
    print("=" * 70)

    task = ArmTask(n_dof=20, use_highdim_descriptor=True)
    config = ExperimentConfig(
        name="runtime",
        n_seeds=1,
        generations=100,
        archive_size=1000,
        log_interval=100,
    )

    results = {}

    if RUST_AVAILABLE:
        print("MMR-Elites...", end=" ", flush=True)
        r = run_mmr_elites(task, config, seed=0)
        results["MMR-Elites"] = r["runtime"]
        print(f"{r['runtime']:.2f}s")

    print("MAP-Elites...", end=" ", flush=True)
    r = run_map_elites(task, config, seed=0)
    results["MAP-Elites"] = r["runtime"]
    print(f"{r['runtime']:.2f}s")

    print("CVT-MAP-Elites...", end=" ", flush=True)
    r = run_cvt_map_elites(task, config, seed=0)
    results["CVT-MAP-Elites"] = r["runtime"]
    print(f"{r['runtime']:.2f}s")

    print("Random...", end=" ", flush=True)
    r = run_random_search(task, config, seed=0)
    results["Random"] = r["runtime"]
    print(f"{r['runtime']:.2f}s")

    with open(output_dir / "runtime_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="MMR-Elites Full Benchmark Suite")
    parser.add_argument(
        "--experiment",
        choices=["all", "scaling", "lambda", "archive", "runtime"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument(
        "--output-dir", default="results/full_benchmark", help="Output directory"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MMR-ELITES FULL BENCHMARK SUITE")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Quick mode: {args.quick}")
    print(f"Rust available: {RUST_AVAILABLE}")
    print("=" * 70)

    if args.experiment in ["all", "scaling"]:
        run_dimensionality_scaling(output_dir, args.seeds, args.quick)

    if args.experiment in ["all", "lambda"]:
        run_lambda_ablation(output_dir, args.seeds, args.quick)

    if args.experiment in ["all", "archive"]:
        run_archive_size_ablation(output_dir, args.seeds, args.quick)

    if args.experiment in ["all", "runtime"]:
        run_runtime_comparison(output_dir)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
