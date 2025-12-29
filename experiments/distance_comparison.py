# experiments/distance_comparison.py
"""
Experiment: Comparing Distance Functions in MMR-Elites

This experiment tests the hypothesis that saturating distance functions
provide better interior coverage than Euclidean distance.
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List
from functools import partial
import argparse

from mmr_elites.tasks.arm import ArmTask
from mmr_elites.metrics.qd_metrics import compute_all_metrics
from experiments.distance_functions import (
    euclidean_distance,
    saturating_distance,
    cosine_distance,
    normalized_euclidean,
)
from experiments.mmr_custom_distance import run_mmr_elites_custom_distance


def compute_coverage_metrics(descriptors: np.ndarray) -> Dict[str, float]:
    """
    Compute detailed coverage metrics.
    
    Beyond standard QD metrics, we compute:
    - Interior coverage: fraction of points not on boundary
    - Boundary ratio: fraction of points near behavior space boundary
    - Spread uniformity: how evenly distributed are the points
    """
    n, d = descriptors.shape
    
    # Boundary detection (points within 10% of boundary)
    boundary_threshold = 0.1
    near_min = np.any(descriptors < boundary_threshold, axis=1)
    near_max = np.any(descriptors > 1 - boundary_threshold, axis=1)
    on_boundary = near_min | near_max
    boundary_ratio = np.mean(on_boundary)
    interior_ratio = 1 - boundary_ratio
    
    # Spread uniformity via binning
    n_bins = 5
    bins = np.linspace(0, 1, n_bins + 1)
    
    # Count points in each hypercolumn (marginal distribution per dimension)
    marginal_uniformity = []
    for dim in range(d):
        hist, _ = np.histogram(descriptors[:, dim], bins=bins)
        # CV of bin counts (lower = more uniform)
        cv = np.std(hist) / (np.mean(hist) + 1e-10)
        marginal_uniformity.append(cv)
    
    mean_marginal_cv = np.mean(marginal_uniformity)
    
    # Convex hull volume (only for low-D)
    if d <= 3 and n > d + 1:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(descriptors[:, :min(d, 3)])
            hull_volume = hull.volume
        except:
            hull_volume = 0.0
    else:
        hull_volume = None
    
    return {
        "boundary_ratio": float(boundary_ratio),
        "interior_ratio": float(interior_ratio),
        "marginal_uniformity_cv": float(mean_marginal_cv),
        "hull_volume": hull_volume,
    }


def run_distance_comparison(
    dimensions: List[int] = None,
    sigma_values: List[float] = None,
    n_seeds: int = 10,
    generations: int = 1000,
    archive_size: int = 1000,
    batch_size: int = 200,
    lambda_val: float = 0.5,
    mutation_sigma: float = 0.1,
    output_dir: Path = None,
) -> Dict:
    """
    Run distance function comparison experiment.
    
    Args:
        dimensions: Behavior space dimensions to test
        sigma_values: Sigma values for saturating distance
        n_seeds: Number of random seeds
        generations: Generations per run
        archive_size: Archive size K
        batch_size: Offspring per generation
        lambda_val: MMR diversity weight
        mutation_sigma: Mutation std
        output_dir: Output directory
    
    Returns:
        Dictionary with all results
    """
    if dimensions is None:
        dimensions = [5, 20, 50]
    if sigma_values is None:
        sigma_values = [0.1, 0.2, 0.3, 0.5]
    if output_dir is None:
        output_dir = Path("results/distance_comparison")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define distance functions to compare
    distance_configs = [
        ("Euclidean", euclidean_distance),
        ("Normalized", lambda b1, b2: normalized_euclidean(b1, b2)),
        ("Cosine", cosine_distance),
    ]
    
    # Add saturating distances with different sigma
    for sigma in sigma_values:
        distance_configs.append(
            (f"Saturating(σ={sigma})", 
             partial(saturating_distance, sigma=sigma))
        )
    
    results = {}
    
    for dim in dimensions:
        print(f"\n{'='*60}")
        print(f"DIMENSION: {dim}")
        print("="*60)
        
        results[dim] = {name: [] for name, _ in distance_configs}
        task = ArmTask(n_dof=dim, use_highdim_descriptor=True)
        
        for seed in range(n_seeds):
            print(f"\n  Seed {seed + 1}/{n_seeds}")
            
            for name, dist_fn in distance_configs:
                print(f"    {name}...", end=" ", flush=True)
                
                result = run_mmr_elites_custom_distance(
                    task=task,
                    distance_fn=dist_fn,
                    archive_size=archive_size,
                    generations=generations,
                    batch_size=batch_size,
                    lambda_val=lambda_val,
                    mutation_sigma=mutation_sigma,
                    seed=seed,
                    log_interval=generations // 10,
                )
                
                # Add coverage metrics
                coverage = compute_coverage_metrics(result["final_descriptors"])
                result["coverage_metrics"] = coverage
                
                results[dim][name].append(result)
                
                print(f"QD@K={result['final_metrics']['qd_score_at_budget']:.1f}, "
                      f"CV={result['final_metrics']['uniformity_cv']:.3f}, "
                      f"Interior={coverage['interior_ratio']:.2%}")
    
    # Save results
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Generate summary
    summary = generate_summary(results, dimensions, [name for name, _ in distance_configs])
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary tables
    print_summary_tables(summary, dimensions, [name for name, _ in distance_configs])
    
    return results


def generate_summary(results: Dict, dimensions: List[int], dist_names: List[str]) -> Dict:
    """Generate summary statistics."""
    summary = {}
    
    metrics_to_summarize = [
        "qd_score_at_budget",
        "mean_fitness", 
        "uniformity_cv",
        "interior_ratio",
        "boundary_ratio",
        "marginal_uniformity_cv",
    ]
    
    for dim in dimensions:
        summary[str(dim)] = {}
        for name in dist_names:
            if results[dim][name]:
                metrics = {}
                for metric in metrics_to_summarize:
                    if metric in ["interior_ratio", "boundary_ratio", "marginal_uniformity_cv"]:
                        values = [r["coverage_metrics"][metric] for r in results[dim][name]]
                    else:
                        values = [r["final_metrics"].get(metric, 0) for r in results[dim][name]]
                    
                    metrics[metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                    }
                summary[str(dim)][name] = metrics
    
    return summary


def print_summary_tables(summary: Dict, dimensions: List[int], dist_names: List[str]):
    """Print publication-ready summary tables."""
    
    # Table 1: QD-Score and Fitness
    print("\n" + "="*100)
    print("TABLE 1: Quality Metrics (mean ± std)")
    print("="*100)
    
    for dim in dimensions:
        print(f"\n--- {dim}D ---")
        print(f"{ 'Distance':<20} | {'QD@K':<18} | {'Mean Fitness':<18}")
        print("-"*60)
        
        for name in dist_names:
            if name in summary.get(str(dim), {}):
                qd = summary[str(dim)][name]["qd_score_at_budget"]
                mf = summary[str(dim)][name]["mean_fitness"]
                print(f"{name:<20} | {qd['mean']:>7.1f} ± {qd['std']:<6.1f} | "
                      f"{mf['mean']:>7.4f} ± {mf['std']:<6.4f}")
    
    # Table 2: Coverage Metrics
    print("\n" + "="*100)
    print("TABLE 2: Coverage Metrics (mean ± std)")
    print("="*100)
    
    for dim in dimensions:
        print(f"\n--- {dim}D ---")
        print(f"{ 'Distance':<20} | {'Uniformity CV ↓':<18} | {'Interior Ratio ↑':<18}")
        print("-"*60)
        
        for name in dist_names:
            if name in summary.get(str(dim), {}):
                cv = summary[str(dim)][name]["uniformity_cv"]
                ir = summary[str(dim)][name]["interior_ratio"]
                print(f"{name:<20} | {cv['mean']:>7.4f} ± {cv['std']:<6.4f} | "
                      f"{ir['mean']:>7.2%} ± {ir['std']:<6.2%}")


def run_sigma_sensitivity(
    sigma_values: List[float] = None,
    n_seeds: int = 5,
    generations: int = 500,
    dim: int = 20,
    output_dir: Path = None,
) -> Dict:
    """
    Sensitivity analysis for sigma parameter in saturating distance.
    
    Find optimal sigma for different behavior space scales.
    """
    if sigma_values is None:
        sigma_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
    
    if output_dir is None:
        output_dir = Path("results/sigma_sensitivity")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task = ArmTask(n_dof=dim, use_highdim_descriptor=True)
    
    results = {sigma: [] for sigma in sigma_values}
    
    print("Sigma Sensitivity Analysis")
    print(f"Dimension: {dim}")
    print(f"Sigma values: {sigma_values}")
    
    for sigma in sigma_values:
        print(f"\n  σ = {sigma}")
        dist_fn = partial(saturating_distance, sigma=sigma)
        
        for seed in range(n_seeds):
            result = run_mmr_elites_custom_distance(
                task=task,
                distance_fn=dist_fn,
                generations=generations,
                seed=seed,
                log_interval=generations // 5,
            )
            coverage = compute_coverage_metrics(result["final_descriptors"])
            result["coverage_metrics"] = coverage
            results[sigma].append(result)
        
        # Print summary for this sigma
        qd_vals = [r["final_metrics"]["qd_score_at_budget"] for r in results[sigma]]
        cv_vals = [r["final_metrics"]["uniformity_cv"] for r in results[sigma]]
        ir_vals = [r["coverage_metrics"]["interior_ratio"] for r in results[sigma]]
        
        print(f"    QD@K: {np.mean(qd_vals):.1f} ± {np.std(qd_vals):.1f}")
        print(f"    CV: {np.mean(cv_vals):.4f} ± {np.std(cv_vals):.4f}")
        print(f"    Interior: {np.mean(ir_vals):.2%} ± {np.std(ir_vals):.2%}")
    
    # Save
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Find optimal sigma (maximize QD while minimizing CV)
    print("\n" + "="*60)
    print("OPTIMAL SIGMA ANALYSIS")
    print("="*60)
    
    for sigma in sigma_values:
        qd = np.mean([r["final_metrics"]["qd_score_at_budget"] for r in results[sigma]])
        cv = np.mean([r["final_metrics"]["uniformity_cv"] for r in results[sigma]])
        # Combined score: high QD, low CV
        combined = qd * (1 - cv)  # Higher is better
        print(f"σ={sigma:<4}: QD={qd:>7.1f}, CV={cv:.4f}, Combined={combined:.1f}")
    
    return results


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test")
    parser.add_argument("--full", action="store_true", help="Full experiment")
    parser.add_argument("--sigma", action="store_true", help="Sigma sensitivity only")
    args = parser.parse_args()
    
    if args.sigma:
        run_sigma_sensitivity()
    elif args.quick:
        run_distance_comparison(
            dimensions=[5, 20],
            sigma_values=[0.2, 0.5],
            n_seeds=2,
            generations=200,
        )
    elif args.full:
        run_distance_comparison(
            dimensions=[5, 10, 20, 50],
            sigma_values=[0.1, 0.2, 0.3, 0.5],
            n_seeds=10,
            generations=1000,
        )
    else:
        # Default run
        run_distance_comparison()
