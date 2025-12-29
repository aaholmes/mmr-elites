#!/usr/bin/env python3
"""
MUSE-QD vs MAP-Elites Benchmark
================================

A complete, self-contained benchmark comparing MUSE-QD to MAP-Elites
on the 20-DOF Arm task with proper statistical methodology.

This script produces publication-ready results for GECCO submission.

Usage:
    # Quick test (2 seeds, 200 generations)
    python benchmark_arm20.py --quick
    
    # Full experiment (10 seeds, 1000 generations)
    python benchmark_arm20.py --full
    
    # Custom configuration
    python benchmark_arm20.py --seeds 5 --generations 500 --lambda 0.7
"""

import numpy as np
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import pickle

# Import your modules (adjust paths as needed)
try:
    import mmr_elites_rs
    RUST_AVAILABLE = True
except ImportError:
    print("⚠️  Rust backend not found. Run: maturin develop --release")
    RUST_AVAILABLE = False


# =============================================================================
# Task Definition (Self-Contained)
# =============================================================================

class Arm20Task:
    """20-DOF Planar Arm with obstacle."""
    
    def __init__(self, target_pos=(0.8, 0.0)):
        self.dof = 20
        self.link_length = 1.0 / self.dof
        self.target_pos = np.array(target_pos)
        
        # Obstacle (wall)
        self.box_x = [0.5, 0.55]
        self.box_y = [-0.25, 0.25]
    
    def forward_kinematics_batch(self, joints):
        """Compute all joint positions for a batch of configurations."""
        angles = np.cumsum(joints, axis=1)
        dx = self.link_length * np.cos(angles)
        dy = self.link_length * np.sin(angles)
        x = np.cumsum(dx, axis=1)
        y = np.cumsum(dy, axis=1)
        return np.stack([x, y], axis=2)
    
    def check_collisions_batch(self, joint_coords):
        """Check which configurations collide with the obstacle."""
        batch_size = joint_coords.shape[0]
        
        # Add origin to create full path
        origin = np.zeros((batch_size, 1, 2))
        points = np.concatenate([origin, joint_coords], axis=1)
        
        # Check if any point is inside the box
        p_in_x = (points[:, :, 0] > self.box_x[0]) & (points[:, :, 0] < self.box_x[1])
        p_in_y = (points[:, :, 1] > self.box_y[0]) & (points[:, :, 1] < self.box_y[1])
        any_inside = np.any(p_in_x & p_in_y, axis=1)
        
        # Check line segment intersections
        A = points[:, :-1, :]
        B = points[:, 1:, :]
        
        Ax, Ay = A[:, :, 0], A[:, :, 1]
        Bx, By = B[:, :, 0], B[:, :, 1]
        dx, dy = Bx - Ax, By - Ay
        
        dx = np.where(np.abs(dx) < 1e-9, 1e-9, dx)
        dy = np.where(np.abs(dy) < 1e-9, 1e-9, dy)
        
        def check_vertical(wall_x, y_min, y_max):
            t = (wall_x - Ax) / dx
            y_at_t = Ay + t * dy
            return (t >= 0) & (t <= 1) & (y_at_t >= y_min) & (y_at_t <= y_max)
        
        def check_horizontal(wall_y, x_min, x_max):
            t = (wall_y - Ay) / dy
            x_at_t = Ax + t * dx
            return (t >= 0) & (t <= 1) & (x_at_t >= x_min) & (x_at_t <= x_max)
        
        hit = (
            check_vertical(self.box_x[0], self.box_y[0], self.box_y[1]) |
            check_vertical(self.box_x[1], self.box_y[0], self.box_y[1]) |
            check_horizontal(self.box_y[0], self.box_x[0], self.box_x[1]) |
            check_horizontal(self.box_y[1], self.box_x[0], self.box_x[1])
        )
        any_intersection = np.any(hit, axis=1)
        
        return any_inside | any_intersection
    
    def evaluate(self, genomes):
        """
        Evaluate a batch of genomes.
        
        Returns:
            fitness: (N,) array of fitness values
            descriptors: (N, 2) array of end-effector positions
        """
        joint_coords = self.forward_kinematics_batch(genomes)
        tips = joint_coords[:, -1, :]
        
        # Fitness: negative distance to target
        dists = np.linalg.norm(tips - self.target_pos, axis=1)
        fitness = 1.0 - dists
        fitness = np.maximum(fitness, 0.0)
        
        # Penalize collisions
        collides = self.check_collisions_batch(joint_coords)
        fitness[collides] = 0.0
        
        return fitness, tips


# =============================================================================
# Metrics (Self-Contained)
# =============================================================================

def qd_score(fitness):
    return float(np.sum(fitness))

def mean_pairwise_distance(descriptors):
    if len(descriptors) < 2:
        return 0.0
    from scipy.spatial.distance import cdist
    dists = cdist(descriptors, descriptors)
    n = len(descriptors)
    upper = dists[np.triu_indices(n, k=1)]
    return float(np.mean(upper))

def compute_metrics(fitness, descriptors):
    return {
        "qd_score": qd_score(fitness),
        "max_fitness": float(np.max(fitness)),
        "mean_fitness": float(np.mean(fitness)),
        "archive_size": len(fitness),
        "mean_pairwise_distance": mean_pairwise_distance(descriptors),
    }


# =============================================================================
# Algorithms
# =============================================================================

def run_muse_qd(
    task: Arm20Task,
    generations: int,
    archive_size: int,
    batch_size: int,
    lambda_val: float,
    mutation_sigma: float,
    seed: int,
    log_interval: int = 50,
) -> Dict:
    """Run MUSE-QD algorithm."""
    
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust backend required for MUSE-QD")
    
    np.random.seed(seed)
    
    selector = mmr_elites_rs.MMRSelector(archive_size, lambda_val)
    
    # Initialize
    archive = np.random.uniform(-np.pi, np.pi, (archive_size, 20))
    fit, desc = task.evaluate(archive)
    
    indices = selector.select(fit, desc)
    archive = archive[indices]
    fit = fit[indices]
    desc = desc[indices]
    
    history = {"generation": [], "qd_score": [], "max_fitness": [], 
               "mean_pairwise_distance": []}
    
    start_time = time.time()
    
    for gen in range(1, generations + 1):
        # Mutation
        parents = archive[np.random.randint(0, len(archive), batch_size)]
        offspring = parents + np.random.normal(0, mutation_sigma, (batch_size, 20))
        offspring = np.clip(offspring, -np.pi, np.pi)
        
        # Evaluation
        off_fit, off_desc = task.evaluate(offspring)
        
        # Selection
        pool_genes = np.vstack([archive, offspring])
        pool_fit = np.concatenate([fit, off_fit])
        pool_desc = np.vstack([desc, off_desc])
        
        survivor_idx = selector.select(pool_fit, pool_desc)
        
        archive = pool_genes[survivor_idx]
        fit = pool_fit[survivor_idx]
        desc = pool_desc[survivor_idx]
        
        # Logging
        if gen % log_interval == 0:
            metrics = compute_metrics(fit, desc)
            history["generation"].append(gen)
            history["qd_score"].append(metrics["qd_score"])
            history["max_fitness"].append(metrics["max_fitness"])
            history["mean_pairwise_distance"].append(metrics["mean_pairwise_distance"])
    
    runtime = time.time() - start_time
    final_metrics = compute_metrics(fit, desc)
    
    return {
        "algorithm": "MUSE-QD",
        "seed": seed,
        "runtime": runtime,
        "final_metrics": final_metrics,
        "history": history,
        "final_archive": archive,
        "final_descriptors": desc,
        "final_fitness": fit,
    }


def run_map_elites(
    task: Arm20Task,
    generations: int,
    batch_size: int,
    mutation_sigma: float,
    seed: int,
    bins_per_dim: int = 50,  # For 2D descriptor space
    log_interval: int = 50,
) -> Dict:
    """Run MAP-Elites algorithm."""
    
    np.random.seed(seed)
    
    # Grid bounds for end-effector positions
    bounds = [(-1.5, 1.5), (-1.5, 1.5)]
    
    def get_cell(descriptor):
        indices = []
        for i, (low, high) in enumerate(bounds):
            norm = (descriptor[i] - low) / (high - low)
            norm = np.clip(norm, 0, 0.9999)
            idx = int(norm * bins_per_dim)
            indices.append(idx)
        return tuple(indices)
    
    # Archive: cell -> (genome, fitness, descriptor)
    archive = {}
    
    # Initialize
    init_pop = np.random.uniform(-np.pi, np.pi, (batch_size * 5, 20))
    fit, desc = task.evaluate(init_pop)
    
    for i in range(len(init_pop)):
        cell = get_cell(desc[i])
        if cell not in archive or fit[i] > archive[cell][1]:
            archive[cell] = (init_pop[i].copy(), fit[i], desc[i].copy())
    
    history = {"generation": [], "qd_score": [], "max_fitness": [], 
               "mean_pairwise_distance": [], "archive_size": []}
    
    start_time = time.time()
    
    for gen in range(1, generations + 1):
        # Select parents
        keys = list(archive.keys())
        if not keys:
            continue
        
        parent_keys = [keys[np.random.randint(len(keys))] for _ in range(batch_size)]
        parents = np.array([archive[k][0] for k in parent_keys])
        
        # Mutation
        offspring = parents + np.random.normal(0, mutation_sigma, parents.shape)
        offspring = np.clip(offspring, -np.pi, np.pi)
        
        # Evaluation
        off_fit, off_desc = task.evaluate(offspring)
        
        # Update archive
        for i in range(len(offspring)):
            cell = get_cell(off_desc[i])
            if cell not in archive or off_fit[i] > archive[cell][1]:
                archive[cell] = (offspring[i].copy(), off_fit[i], off_desc[i].copy())
        
        # Logging
        if gen % log_interval == 0:
            all_fit = np.array([v[1] for v in archive.values()])
            all_desc = np.array([v[2] for v in archive.values()])
            metrics = compute_metrics(all_fit, all_desc)
            
            history["generation"].append(gen)
            history["qd_score"].append(metrics["qd_score"])
            history["max_fitness"].append(metrics["max_fitness"])
            history["mean_pairwise_distance"].append(metrics["mean_pairwise_distance"])
            history["archive_size"].append(len(archive))
    
    runtime = time.time() - start_time
    
    # Final state
    all_genomes = np.array([v[0] for v in archive.values()])
    all_fit = np.array([v[1] for v in archive.values()])
    all_desc = np.array([v[2] for v in archive.values()])
    
    final_metrics = compute_metrics(all_fit, all_desc)
    final_metrics["archive_size"] = len(archive)
    
    return {
        "algorithm": "MAP-Elites",
        "seed": seed,
        "runtime": runtime,
        "final_metrics": final_metrics,
        "history": history,
        "final_archive": all_genomes,
        "final_descriptors": all_desc,
        "final_fitness": all_fit,
    }


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiment(
    n_seeds: int = 10,
    generations: int = 1000,
    archive_size: int = 1000,
    batch_size: int = 200,
    lambda_val: float = 0.5,
    mutation_sigma: float = 0.1,
    output_dir: str = "benchmark_results",
    quick_mode: bool = False,
):
    """Run the complete benchmark experiment."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    task = Arm20Task(target_pos=(0.8, 0.0))
    
    print("=" * 60)
    print("MUSE-QD vs MAP-Elites Benchmark")
    print("=" * 60)
    print(f"Seeds: {n_seeds}")
    print(f"Generations: {generations}")
    print(f"Archive Size (MUSE): {archive_size}")
    print(f"Lambda: {lambda_val}")
    print("=" * 60)
    
    all_results = {"MUSE-QD": [], "MAP-Elites": []}
    
    # Run MUSE-QD
    if RUST_AVAILABLE:
        print("\n🚀 Running MUSE-QD...")
        for seed in range(n_seeds):
            print(f"  Seed {seed + 1}/{n_seeds}...", end=" ", flush=True)
            result = run_muse_qd(
                task, generations, archive_size, batch_size,
                lambda_val, mutation_sigma, seed
            )
            all_results["MUSE-QD"].append(result)
            print(f"QD-Score: {result['final_metrics']['qd_score']:.2f}")
    else:
        print("\n⚠️  Skipping MUSE-QD (Rust not available)")
    
    # Run MAP-Elites
    print("\n📊 Running MAP-Elites...")
    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}...", end=" ", flush=True)
        result = run_map_elites(
            task, generations, batch_size, mutation_sigma, seed
        )
        all_results["MAP-Elites"].append(result)
        print(f"QD-Score: {result['final_metrics']['qd_score']:.2f}, "
              f"Archive: {result['final_metrics']['archive_size']}")
    
    # ==========================================================================
    # Results Summary
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    metrics_to_report = ["qd_score", "max_fitness", "mean_pairwise_distance", "archive_size"]
    
    print(f"\n{'Metric':<25} | {'MUSE-QD':<20} | {'MAP-Elites':<20}")
    print("-" * 70)
    
    summary = {}
    
    for metric in metrics_to_report:
        row = f"{metric:<25} |"
        summary[metric] = {}
        
        for alg in ["MUSE-QD", "MAP-Elites"]:
            if all_results[alg]:
                values = [r["final_metrics"].get(metric, 0) for r in all_results[alg]]
                mean = np.mean(values)
                std = np.std(values)
                row += f" {mean:>8.2f} ± {std:<6.2f} |"
                summary[metric][alg] = {"mean": mean, "std": std}
            else:
                row += f" {'N/A':^18} |"
        
        print(row)
    
    # Save results
    with open(output_path / "all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to {output_path}")
    
    # ==========================================================================
    # Generate Plots
    # ==========================================================================
    
    try:
        import matplotlib.pyplot as plt
        
        # Learning curves
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        metrics_plot = ["qd_score", "max_fitness", "mean_pairwise_distance"]
        
        for ax, metric in zip(axes, metrics_plot):
            for alg, color in [("MUSE-QD", "#0072B2"), ("MAP-Elites", "#D55E00")]:
                if not all_results[alg]:
                    continue
                
                # Get all runs' histories
                all_gens = all_results[alg][0]["history"]["generation"]
                all_values = np.array([r["history"][metric] for r in all_results[alg]])
                
                mean = np.mean(all_values, axis=0)
                std = np.std(all_values, axis=0)
                
                ax.fill_between(all_gens, mean - std, mean + std, alpha=0.2, color=color)
                ax.plot(all_gens, mean, label=alg, color=color, linewidth=2)
            
            ax.set_xlabel("Generation")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "learning_curves.png", dpi=150)
        plt.savefig(output_path / "learning_curves.pdf")
        print(f"📈 Saved learning curves to {output_path}")
        
        # Behavior space comparison
        if all_results["MUSE-QD"] and all_results["MAP-Elites"]:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            for ax, alg, color in zip(axes, ["MUSE-QD", "MAP-Elites"], 
                                       ["#0072B2", "#D55E00"]):
                desc = all_results[alg][0]["final_descriptors"]
                ax.scatter(desc[:, 0], desc[:, 1], c=color, alpha=0.5, s=20)
                ax.set_title(f"{alg} (N={len(desc)})")
                ax.set_xlabel("End-Effector X")
                ax.set_ylabel("End-Effector Y")
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                ax.set_aspect('equal')
                
                # Draw target
                ax.scatter([0.8], [0.0], c='green', s=200, marker='*', zorder=10)
                
                # Draw obstacle
                from matplotlib.patches import Rectangle
                rect = Rectangle((0.5, -0.25), 0.05, 0.5, color='red', alpha=0.5)
                ax.add_patch(rect)
            
            plt.tight_layout()
            plt.savefig(output_path / "behavior_space.png", dpi=150)
            plt.savefig(output_path / "behavior_space.pdf")
            print(f"📊 Saved behavior space plots to {output_path}")
        
    except ImportError:
        print("⚠️  Matplotlib not available, skipping plots")
    
    print("\n✅ Benchmark complete!")
    
    return all_results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MUSE-QD vs MAP-Elites Benchmark")
    
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (2 seeds, 200 generations)")
    parser.add_argument("--full", action="store_true",
                        help="Full experiment (10 seeds, 1000 generations)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--generations", type=int, default=500, help="Generations")
    parser.add_argument("--archive-size", type=int, default=1000, help="Archive size")
    parser.add_argument("--lambda", type=float, default=0.5, dest="lambda_val",
                        help="Lambda (diversity weight)")
    parser.add_argument("--output-dir", default="benchmark_results",
                        help="Output directory")
    
    args = parser.parse_args()
    
    if args.quick:
        run_experiment(n_seeds=2, generations=200, output_dir=args.output_dir)
    elif args.full:
        run_experiment(n_seeds=10, generations=1000, output_dir=args.output_dir)
    else:
        run_experiment(
            n_seeds=args.seeds,
            generations=args.generations,
            archive_size=args.archive_size,
            lambda_val=args.lambda_val,
            output_dir=args.output_dir,
        )
