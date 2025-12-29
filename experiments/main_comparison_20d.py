#!/usr/bin/env python3
"""
Main Experiment: MUSE-QD vs MAP-Elites vs CVT-MAP-Elites
========================================================

This is THE experiment for the GECCO paper.

Task: 20-DOF Planar Arm with 20D behavior descriptors (joint angles)

This setup demonstrates the curse of dimensionality:
- MAP-Elites with 3 bins/dim = 3^20 ≈ 3.5 billion cells → degrades to random search
- CVT-MAP-Elites: K pre-computed centroids, fixed memory
- MUSE-QD: K solutions selected via MMR, fixed memory

Expected results:
- MUSE-QD > CVT-MAP-Elites > MAP-Elites on QD-Score
- MUSE-QD achieves better uniformity (lower CV of k-NN distances)
- MAP-Elites archive grows unboundedly but with low-quality solutions

Usage:
    python main_comparison_20d.py --quick      # Quick test (2 seeds)
    python main_comparison_20d.py --full       # Full experiment (10 seeds)
    python main_comparison_20d.py --seeds 5    # Custom seeds
"""

import numpy as np
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List
import time
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Try to import Rust backend
try:
    import mmr_elites_rs
    RUST_AVAILABLE = True
except ImportError:
    print("⚠️  Rust backend not found. Run: maturin develop --release")
    RUST_AVAILABLE = False


# =============================================================================
# Task Definition
# =============================================================================

class Arm20HighDimTask:
    """20-DOF Arm with 20D behavior descriptor (joint angles)."""
    
    def __init__(self, target_pos=(0.8, 0.0)):
        self.dof = 20
        self.link_length = 1.0 / self.dof
        self.target_pos = np.array(target_pos)
        self.box_x = [0.5, 0.55]
        self.box_y = [-0.25, 0.25]
    
    def forward_kinematics_batch(self, joints):
        angles = np.cumsum(joints, axis=1)
        dx = self.link_length * np.cos(angles)
        dy = self.link_length * np.sin(angles)
        return np.stack([np.cumsum(dx, axis=1), np.cumsum(dy, axis=1)], axis=2)
    
    def check_collisions_batch(self, joint_coords):
        batch_size = joint_coords.shape[0]
        origin = np.zeros((batch_size, 1, 2))
        points = np.concatenate([origin, joint_coords], axis=1)
        
        p_in_x = (points[:, :, 0] > self.box_x[0]) & (points[:, :, 0] < self.box_x[1])
        p_in_y = (points[:, :, 1] > self.box_y[0]) & (points[:, :, 1] < self.box_y[1])
        any_inside = np.any(p_in_x & p_in_y, axis=1)
        
        A, B = points[:, :-1, :], points[:, 1:, :]
        Ax, Ay, Bx, By = A[:,:,0], A[:,:,1], B[:,:,0], B[:,:,1]
        dx, dy = Bx - Ax, By - Ay
        dx = np.where(np.abs(dx) < 1e-9, 1e-9, dx)
        dy = np.where(np.abs(dy) < 1e-9, 1e-9, dy)
        
        def check_v(wx, ymin, ymax):
            t = (wx - Ax) / dx
            return (t >= 0) & (t <= 1) & (Ay + t*dy >= ymin) & (Ay + t*dy <= ymax)
        def check_h(wy, xmin, xmax):
            t = (wy - Ay) / dy
            return (t >= 0) & (t <= 1) & (Ax + t*dx >= xmin) & (Ax + t*dx <= xmax)
        
        hit = check_v(self.box_x[0], *self.box_y) | check_v(self.box_x[1], *self.box_y) | \
              check_h(self.box_y[0], *self.box_x) | check_h(self.box_y[1], *self.box_x)
        return any_inside | np.any(hit, axis=1)
    
    def evaluate(self, genomes):
        jc = self.forward_kinematics_batch(genomes)
        tips = jc[:, -1, :]
        fitness = np.maximum(1.0 - np.linalg.norm(tips - self.target_pos, axis=1), 0.0)
        fitness[self.check_collisions_batch(jc)] = 0.0
        descriptors = (genomes + np.pi) / (2 * np.pi)  # Normalize to [0,1]
        return fitness, descriptors


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(fitness: np.ndarray, descriptors: np.ndarray) -> Dict[str, float]:
    """Compute QD metrics."""
    from scipy.spatial.distance import cdist
    from scipy.spatial import cKDTree
    
    n = len(fitness)
    if n == 0:
        return {"qd_score": 0, "max_fitness": 0, "mean_fitness": 0, 
                "mean_pairwise_distance": 0, "uniformity_cv": 0, "archive_size": 0}
    
    # Basic metrics
    qd_score = float(np.sum(fitness))
    max_fit = float(np.max(fitness))
    mean_fit = float(np.mean(fitness))
    
    # Diversity metrics
    if n > 1:
        dists = cdist(descriptors, descriptors)
        mpd = float(np.mean(dists[np.triu_indices(n, k=1)]))
        
        # Uniformity (CV of k-NN distances)
        k = min(5, n - 1)
        if k > 0:
            tree = cKDTree(descriptors)
            knn_dists, _ = tree.query(descriptors, k=k+1)
            mean_knn = np.mean(knn_dists[:, 1:], axis=1)
            cv = np.std(mean_knn) / (np.mean(mean_knn) + 1e-10)
        else:
            cv = 0.0
    else:
        mpd = 0.0
        cv = 0.0
    
    return {
        "qd_score": qd_score,
        "max_fitness": max_fit,
        "mean_fitness": mean_fit,
        "mean_pairwise_distance": mpd,
        "uniformity_cv": cv,
        "archive_size": n,
    }


# =============================================================================
# Algorithms
# =============================================================================

def run_muse_qd(task, generations, archive_size, batch_size, mutation_sigma, 
                lambda_val, seed, log_interval=100) -> Dict:
    """Run MUSE-QD algorithm."""
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust backend required")
    
    np.random.seed(seed)
    selector = mmr_elites_rs.MMRSelector(archive_size, lambda_val)
    
    # Initialize
    archive = np.random.uniform(-np.pi, np.pi, (archive_size, 20))
    fit, desc = task.evaluate(archive)
    idx = selector.select(fit, desc)
    archive, fit, desc = archive[idx], fit[idx], desc[idx]
    
    history = {"generation": [], "qd_score": [], "max_fitness": [], 
               "mean_pairwise_distance": [], "uniformity_cv": []}
    
    start = time.time()
    
    for gen in range(1, generations + 1):
        parents = archive[np.random.randint(0, len(archive), batch_size)]
        offspring = np.clip(parents + np.random.normal(0, mutation_sigma, (batch_size, 20)), 
                           -np.pi, np.pi)
        off_fit, off_desc = task.evaluate(offspring)
        
        pool = np.vstack([archive, offspring])
        pool_fit = np.concatenate([fit, off_fit])
        pool_desc = np.vstack([desc, off_desc])
        
        idx = selector.select(pool_fit, pool_desc)
        archive, fit, desc = pool[idx], pool_fit[idx], pool_desc[idx]
        
        if gen % log_interval == 0:
            m = compute_metrics(fit, desc)
            for k in history:
                if k != "generation":
                    history[k].append(m[k])
            history["generation"].append(gen)
    
    return {
        "algorithm": "MUSE-QD",
        "seed": seed,
        "runtime": time.time() - start,
        "final_metrics": compute_metrics(fit, desc),
        "history": history,
        "final_descriptors": desc,
        "final_fitness": fit,
    }


def run_map_elites(task, generations, batch_size, mutation_sigma, seed, 
                   bins_per_dim=3, log_interval=100) -> Dict:
    """Run sparse MAP-Elites (demonstrates curse of dimensionality)."""
    np.random.seed(seed)
    
    def get_cell(desc):
        idx = (desc * bins_per_dim).astype(int)
        return tuple(np.clip(idx, 0, bins_per_dim - 1))
    
    archive = {}
    
    # Initialize
    init_pop = np.random.uniform(-np.pi, np.pi, (batch_size * 5, 20))
    fit, desc = task.evaluate(init_pop)
    for i in range(len(init_pop)):
        cell = get_cell(desc[i])
        if cell not in archive or fit[i] > archive[cell][1]:
            archive[cell] = (init_pop[i].copy(), fit[i], desc[i].copy())
    
    history = {"generation": [], "qd_score": [], "max_fitness": [], 
               "mean_pairwise_distance": [], "uniformity_cv": [], "archive_size": []}
    
    start = time.time()
    
    for gen in range(1, generations + 1):
        keys = list(archive.keys())
        parents = np.array([archive[keys[np.random.randint(len(keys))]][0] 
                           for _ in range(batch_size)])
        offspring = np.clip(parents + np.random.normal(0, mutation_sigma, parents.shape),
                           -np.pi, np.pi)
        off_fit, off_desc = task.evaluate(offspring)
        
        for i in range(len(offspring)):
            cell = get_cell(off_desc[i])
            if cell not in archive or off_fit[i] > archive[cell][1]:
                archive[cell] = (offspring[i].copy(), off_fit[i], off_desc[i].copy())
        
        if gen % log_interval == 0:
            all_fit = np.array([v[1] for v in archive.values()])
            all_desc = np.array([v[2] for v in archive.values()])
            m = compute_metrics(all_fit, all_desc)
            for k in history:
                if k not in ["generation", "archive_size"]:
                    history[k].append(m[k])
            history["generation"].append(gen)
            history["archive_size"].append(len(archive))
    
    all_fit = np.array([v[1] for v in archive.values()])
    all_desc = np.array([v[2] for v in archive.values()])
    
    return {
        "algorithm": "MAP-Elites",
        "seed": seed,
        "runtime": time.time() - start,
        "final_metrics": compute_metrics(all_fit, all_desc),
        "history": history,
        "final_descriptors": all_desc,
        "final_fitness": all_fit,
    }


def run_cvt_map_elites(task, n_niches, generations, batch_size, mutation_sigma, 
                       seed, log_interval=100) -> Dict:
    """Run CVT-MAP-Elites."""
    from scipy.spatial import cKDTree
    
    np.random.seed(seed)
    
    # Compute CVT centroids
    print(f"    Computing {n_niches} CVT centroids...", end=" ", flush=True)
    samples = np.random.uniform(0, 1, (50000, 20))
    try:
        from sklearn.cluster import KMeans
        centroids = KMeans(n_clusters=n_niches, random_state=seed, n_init=1).fit(samples).cluster_centers_
    except ImportError:
        # Simple fallback
        idx = np.random.choice(len(samples), n_niches, replace=False)
        centroids = samples[idx]
    print("Done.")
    
    tree = cKDTree(centroids)
    archive = {}
    
    def get_niche(desc):
        _, idx = tree.query(desc)
        return idx
    
    # Initialize
    init_pop = np.random.uniform(-np.pi, np.pi, (batch_size * 5, 20))
    fit, desc = task.evaluate(init_pop)
    for i in range(len(init_pop)):
        niche = get_niche(desc[i])
        if niche not in archive or fit[i] > archive[niche][1]:
            archive[niche] = (init_pop[i].copy(), fit[i], desc[i].copy())
    
    history = {"generation": [], "qd_score": [], "max_fitness": [], 
               "mean_pairwise_distance": [], "uniformity_cv": [], "archive_size": []}
    
    start = time.time()
    
    for gen in range(1, generations + 1):
        keys = list(archive.keys())
        parents = np.array([archive[keys[np.random.randint(len(keys))]][0] 
                           for _ in range(batch_size)])
        offspring = np.clip(parents + np.random.normal(0, mutation_sigma, parents.shape),
                           -np.pi, np.pi)
        off_fit, off_desc = task.evaluate(offspring)
        
        for i in range(len(offspring)):
            niche = get_niche(off_desc[i])
            if niche not in archive or off_fit[i] > archive[niche][1]:
                archive[niche] = (offspring[i].copy(), off_fit[i], off_desc[i].copy())
        
        if gen % log_interval == 0:
            all_fit = np.array([v[1] for v in archive.values()])
            all_desc = np.array([v[2] for v in archive.values()])
            m = compute_metrics(all_fit, all_desc)
            for k in history:
                if k not in ["generation", "archive_size"]:
                    history[k].append(m[k])
            history["generation"].append(gen)
            history["archive_size"].append(len(archive))
    
    all_fit = np.array([v[1] for v in archive.values()])
    all_desc = np.array([v[2] for v in archive.values()])
    
    return {
        "algorithm": "CVT-MAP-Elites",
        "seed": seed,
        "runtime": time.time() - start,
        "final_metrics": compute_metrics(all_fit, all_desc),
        "history": history,
        "final_descriptors": all_desc,
        "final_fitness": all_fit,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(
    n_seeds: int = 10,
    generations: int = 2000,
    archive_size: int = 1000,
    batch_size: int = 200,
    lambda_val: float = 0.5,
    mutation_sigma: float = 0.1,
    output_dir: str = "results/main_comparison_20d",
):
    """Run the complete comparison experiment."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    task = Arm20HighDimTask()
    
    print("="*70)
    print("MAIN EXPERIMENT: 20D Behavior Descriptor Comparison")
    print("="*70)
    print(f"Seeds: {n_seeds}")
    print(f"Generations: {generations}")
    print(f"Archive Size: {archive_size}")
    print(f"Lambda (MUSE): {lambda_val}")
    print("="*70)
    
    all_results = {"MUSE-QD": [], "MAP-Elites": [], "CVT-MAP-Elites": []}
    
    for seed in range(n_seeds):
        print(f"\n{'='*50}")
        print(f"SEED {seed + 1}/{n_seeds}")
        print("="*50)
        
        if RUST_AVAILABLE:
            print("  Running MUSE-QD...", end=" ", flush=True)
            r = run_muse_qd(task, generations, archive_size, batch_size, 
                           mutation_sigma, lambda_val, seed)
            all_results["MUSE-QD"].append(r)
            print(f"QD={r['final_metrics']['qd_score']:.2f}, "
                  f"MaxFit={r['final_metrics']['max_fitness']:.4f}")
        
        print("  Running MAP-Elites...", end=" ", flush=True)
        r = run_map_elites(task, generations, batch_size, mutation_sigma, seed)
        all_results["MAP-Elites"].append(r)
        print(f"QD={r['final_metrics']['qd_score']:.2f}, "
              f"Archive={r['final_metrics']['archive_size']}")
        
        print("  Running CVT-MAP-Elites...")
        r = run_cvt_map_elites(task, archive_size, generations, batch_size, 
                              mutation_sigma, seed)
        all_results["CVT-MAP-Elites"].append(r)
        print(f"    QD={r['final_metrics']['qd_score']:.2f}, "
              f"Archive={r['final_metrics']['archive_size']}")
    
    # Save results
    with open(output_path / "all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    # Print summary
    print("\n" + "="*90)
    print("FINAL RESULTS (mean ± std)")
    print("="*90)
    
    metrics = ["qd_score", "max_fitness", "mean_pairwise_distance", 
               "uniformity_cv", "archive_size"]
    
    header = f"{'Metric':<25}"
    for alg in ["MUSE-QD", "MAP-Elites", "CVT-MAP-Elites"]:
        header += f" | {alg:<20}"
    print(header)
    print("-"*90)
    
    summary = {}
    for m in metrics:
        row = f"{m:<25}"
        summary[m] = {}
        for alg in ["MUSE-QD", "MAP-Elites", "CVT-MAP-Elites"]:
            if all_results[alg]:
                vals = [r["final_metrics"].get(m, 0) for r in all_results[alg]]
                mean, std = np.mean(vals), np.std(vals)
                row += f" | {mean:>8.2f} ± {std:<7.2f}"
                summary[m][alg] = {"mean": mean, "std": std}
            else:
                row += f" | {'N/A':^20}"
        print(row)
    
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Generate plots
    try:
        import matplotlib.pyplot as plt
        
        # Learning curves
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        metrics_plot = ["qd_score", "max_fitness", "mean_pairwise_distance", "archive_size"]
        colors = {"MUSE-QD": "#0072B2", "MAP-Elites": "#D55E00", "CVT-MAP-Elites": "#009E73"}
        
        for ax, metric in zip(axes, metrics_plot):
            for alg in ["MUSE-QD", "MAP-Elites", "CVT-MAP-Elites"]:
                if not all_results[alg]:
                    continue
                if metric not in all_results[alg][0]["history"]:
                    continue
                
                gens = all_results[alg][0]["history"]["generation"]
                vals = np.array([r["history"][metric] for r in all_results[alg]])
                mean, std = np.mean(vals, axis=0), np.std(vals, axis=0)
                
                ax.fill_between(gens, mean - std, mean + std, alpha=0.2, color=colors[alg])
                ax.plot(gens, mean, label=alg, color=colors[alg], linewidth=2)
            
            ax.set_xlabel("Generation")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "learning_curves.png", dpi=150)
        plt.savefig(output_path / "learning_curves.pdf")
        print(f"\n📈 Saved plots to {output_path}")
        
    except ImportError:
        print("\n⚠️  Matplotlib not available, skipping plots")
    
    print(f"\n✅ Results saved to {output_path}")
    
    return all_results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MUSE-QD Main Comparison Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick test (2 seeds, 500 gens)")
    parser.add_argument("--full", action="store_true", help="Full experiment (10 seeds, 2000 gens)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--generations", type=int, default=1000, help="Generations")
    parser.add_argument("--archive-size", type=int, default=1000, help="Archive size")
    parser.add_argument("--lambda", type=float, default=0.5, dest="lambda_val")
    parser.add_argument("--output-dir", default="results/main_comparison_20d")
    
    args = parser.parse_args()
    
    if args.quick:
        run_experiment(n_seeds=2, generations=500, output_dir=args.output_dir)
    elif args.full:
        run_experiment(n_seeds=10, generations=2000, output_dir=args.output_dir)
    else:
        run_experiment(
            n_seeds=args.seeds,
            generations=args.generations,
            archive_size=args.archive_size,
            lambda_val=args.lambda_val,
            output_dir=args.output_dir,
        )
