"""
Distance Function Comparison Experiment
=======================================

Compares Euclidean, Gaussian saturating, and Exponential saturating distances
across multiple behavior space dimensions.

Key hypotheses:
1. Exponential > Gaussian (no dead zone at small distances)
2. Saturating > Euclidean (better interior coverage)
3. Effect more pronounced at higher dimensions
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional
from functools import partial
from dataclasses import dataclass

from mmr_elites.tasks.arm import ArmTask
from mmr_elites.metrics.qd_metrics import compute_all_metrics

from experiments.distance_functions import (
    euclidean_distance,
    normalized_euclidean,
    exponential_saturating,
    gaussian_saturating,
    auto_sigma,
    auto_sigma_diagonal,
)
from experiments.mmr_custom_distance import run_mmr_elites_custom


# =============================================================================
# Coverage Metrics
# =============================================================================

def compute_coverage_metrics(descriptors: np.ndarray) -> Dict[str, float]:
    """
    Compute detailed coverage metrics beyond standard QD metrics.
    
    Args:
        descriptors: (N, D) behavior descriptors in [0, 1]^D
    
    Returns:
        Dictionary with:
        - interior_ratio: Fraction of points not near boundary
        - boundary_ratio: Fraction near boundary (within 10%)
        - marginal_uniformity_cv: CV of marginal distributions
    """
    n, d = descriptors.shape
    
    # Boundary detection: points within 10% of any boundary
    threshold = 0.1
    near_min = np.any(descriptors < threshold, axis=1)
    near_max = np.any(descriptors > 1 - threshold, axis=1)
    on_boundary = near_min | near_max
    
    boundary_ratio = float(np.mean(on_boundary))
    interior_ratio = 1.0 - boundary_ratio
    
    # Marginal uniformity: check distribution along each dimension
    n_bins = 5
    marginal_cvs = []
    for dim in range(d):
        hist, _ = np.histogram(descriptors[:, dim], bins=n_bins, range=(0, 1))
        if np.mean(hist) > 0:
            cv = np.std(hist) / np.mean(hist)
            marginal_cvs.append(cv)
    
    marginal_uniformity_cv = float(np.mean(marginal_cvs)) if marginal_cvs else 0.0
    
    return {
        "interior_ratio": interior_ratio,
        "boundary_ratio": boundary_ratio,
        "marginal_uniformity_cv": marginal_uniformity_cv,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_distance_comparison(
    dimensions: Optional[List[int]] = None,
    n_seeds: int = 10,
    generations: int = 1000,
    archive_size: int = 1000,
    batch_size: int = 200,
    lambda_val: float = 0.5,
    mutation_sigma: float = 0.1,
    output_dir: Optional[Path] = None,
    sigma_method: str = "auto",  # "auto" or "diagonal"
) -> Dict:
    """
    Run distance function comparison experiment.
    
    Args:
        dimensions: List of behavior space dimensions to test
        n_seeds: Number of random seeds per configuration
        generations: Generations per run
        archive_size: Archive size K
        batch_size: Offspring per generation
        lambda_val: MMR diversity weight
        mutation_sigma: Mutation std
        output_dir: Where to save results
        sigma_method: How to compute sigma ("auto" from data, "diagonal" from space)
    
    Returns:
        Dictionary with all results
    """
    if dimensions is None:
        dimensions = [5, 10, 20, 50]
    
    if output_dir is None:
        output_dir = Path("results/distance_comparison")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for dim in dimensions:
        print(f"\n{'='*60}")
        print(f"DIMENSION: {dim}")
        print("="*60)
        
        task = ArmTask(n_dof=dim, use_highdim_descriptor=True)
        
        # Compute sigma for this dimension
        if sigma_method == "diagonal":
            sigma = auto_sigma_diagonal(dim, fraction=0.2)
        else:
            # Compute from sample data
            sample_genomes = np.random.uniform(-np.pi, np.pi, (1000, dim))
            _, sample_desc = task.evaluate(sample_genomes)
            sigma = auto_sigma(sample_desc, percentile=50)
        
        print(f"  Using σ = {sigma:.4f}")
        
        # Define distance functions to compare
        distance_configs = [
            ("Euclidean", euclidean_distance),
            ("Normalized", normalized_euclidean),
            ("Gaussian (σ={:.2f})".format(sigma), 
             partial(gaussian_saturating, sigma=sigma)),
            ("Exponential (σ={:.2f})".format(sigma), 
             partial(exponential_saturating, sigma=sigma)),
        ]
        
        results[dim] = {name: [] for name, _ in distance_configs}
        results[dim]["sigma"] = sigma
        
        for seed in range(n_seeds):
            print(f"\n  Seed {seed + 1}/{n_seeds}")
            
            for name, dist_fn in distance_configs:
                print(f"    {name}...", end=" ", flush=True)
                
                result = run_mmr_elites_custom(
                    task=task,
                    distance_fn=dist_fn,
                    distance_name=name,
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
                
                qd = result["final_metrics"]["qd_score_at_budget"]
                cv = result["final_metrics"]["uniformity_cv"]
                interior = coverage["interior_ratio"]
                
                print(f"QD@K={qd:.1f}, CV={cv:.3f}, Interior={interior:.1%}")
    
    # Save results
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Generate summary
    summary = generate_summary(results, dimensions)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary tables
    print_summary_tables(results, dimensions)
    
    return results


def generate_summary(results: Dict, dimensions: List[int]) -> Dict:
    """Generate summary statistics across seeds."""
    summary = {}
    
    metrics = [
        ("qd_score_at_budget", "final_metrics"),
        ("uniformity_cv", "final_metrics"),
        ("mean_fitness", "final_metrics"),
        ("interior_ratio", "coverage_metrics"),
    ]
    
    for dim in dimensions:
        summary[str(dim)] = {"sigma": results[dim].get("sigma", None)}
        
        for name in results[dim]:
            if name == "sigma":
                continue
            
            summary[str(dim)][name] = {}
            
            for metric, source in metrics:
                if source == "final_metrics":
                    values = [r["final_metrics"].get(metric, 0) 
                             for r in results[dim][name]]
                else:
                    values = [r["coverage_metrics"].get(metric, 0) 
                             for r in results[dim][name]]
                
                summary[str(dim)][name][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
    
    return summary

def print_summary_tables(results: Dict, dimensions: List[int]):
    """Print publication-ready summary tables."""
    
    print("\n" + "="*100)
    print("SUMMARY: Distance Function Comparison")
    print("="*100)
    
    for dim in dimensions:
        print(f"\n--- {dim}D (σ = {results[dim].get('sigma', 'N/A'):.3f}) ---")
        print(f"{ 'Distance':<25} | {'QD@K':<16} | {'CV ↓':<16} | {'Interior ↑':<16}")
        print("-"*80)
        
        for name in results[dim]:
            if name == "sigma":
                continue
            
            runs = results[dim][name]
            
            qd_vals = [r["final_metrics"]["qd_score_at_budget"] for r in runs]
            cv_vals = [r["final_metrics"]["uniformity_cv"] for r in runs]
            int_vals = [r["coverage_metrics"]["interior_ratio"] for r in runs]
            
            qd_str = f"{np.mean(qd_vals):>7.1f} ± {np.std(qd_vals):<5.1f}"
            cv_str = f"{np.mean(cv_vals):>7.3f} ± {np.std(cv_vals):<5.3f}"
            int_str = f"{np.mean(int_vals)*100:>6.1f}% ± {np.std(int_vals)*100:<4.1f}%"
            
            print(f"{name:<25} | {qd_str} | {cv_str} | {int_str}")
    
    # Winner summary
    print("\n" + "="*100)
    print("WINNERS BY METRIC")
    print("="*100)
    
    for dim in dimensions:
        print(f"\n{dim}D:")
        
        best_qd = ("", -np.inf)
        best_cv = ("", np.inf)
        best_int = ("", -np.inf)
        
        for name in results[dim]:
            if name == "sigma":
                continue
            
            runs = results[dim][name]
            qd_mean = np.mean([r["final_metrics"]["qd_score_at_budget"] for r in runs])
            cv_mean = np.mean([r["final_metrics"]["uniformity_cv"] for r in runs])
            int_mean = np.mean([r["coverage_metrics"]["interior_ratio"] for r in runs])
            
            if qd_mean > best_qd[1]:
                best_qd = (name, qd_mean)
            if cv_mean < best_cv[1]:
                best_cv = (name, cv_mean)
            if int_mean > best_int[1]:
                best_int = (name, int_mean)
        
        print(f"  QD@K:     {best_qd[0]} ({best_qd[1]:.1f})")
        print(f"  CV:       {best_cv[0]} ({best_cv[1]:.3f})")
        print(f"  Interior: {best_int[0]} ({best_int[1]*100:.1f}%)")


# =============================================================================
# Sigma Sensitivity Experiment
# =============================================================================

def run_sigma_sensitivity(
    sigma_multipliers: Optional[List[float]] = None,
    dim: int = 20,
    n_seeds: int = 5,
    generations: int = 500,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Test sensitivity to sigma parameter.
    
    Uses exponential distance with various sigma values relative to auto-computed.
    
    Args:
        sigma_multipliers: Multipliers of auto-sigma to test
        dim: Behavior space dimension
        n_seeds: Seeds per configuration
        generations: Generations per run
        output_dir: Where to save results
    
    Returns:
        Results dictionary
    """
    if sigma_multipliers is None:
        sigma_multipliers = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    
    if output_dir is None:
        output_dir = Path("results/sigma_sensitivity")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task = ArmTask(n_dof=dim, use_highdim_descriptor=True)
    
    # Get base sigma
    sample_genomes = np.random.uniform(-np.pi, np.pi, (1000, dim))
    _, sample_desc = task.evaluate(sample_genomes)
    base_sigma = auto_sigma(sample_desc)
    
    print(f"Base σ (auto) = {base_sigma:.4f}")
    print(f"Testing multipliers: {sigma_multipliers}")
    
    results = {}
    
    for mult in sigma_multipliers:
        sigma = base_sigma * mult
        print(f"\n  σ = {sigma:.4f} ({mult}× base)")
        
        results[mult] = {"sigma": sigma, "runs": []}
        
        dist_fn = partial(exponential_saturating, sigma=sigma)
        
        for seed in range(n_seeds):
            result = run_mmr_elites_custom(
                task=task,
                distance_fn=dist_fn,
                distance_name=f"Exp(σ={sigma:.3f})",
                generations=generations,
                seed=seed,
                log_interval=generations // 5,
            )
            coverage = compute_coverage_metrics(result["final_descriptors"])
            result["coverage_metrics"] = coverage
            results[mult]["runs"].append(result)
        
        # Summary
        qd_vals = [r["final_metrics"]["qd_score_at_budget"] for r in results[mult]["runs"]]
        cv_vals = [r["final_metrics"]["uniformity_cv"] for r in results[mult]["runs"]]
        
        print(f"    QD@K: {np.mean(qd_vals):.1f} ± {np.std(qd_vals):.1f}")
        print(f"    CV:   {np.mean(cv_vals):.3f} ± {np.std(cv_vals):.3f}")
    
    # Save
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Distance Function Comparison")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick test (2 seeds, 200 generations)")
    parser.add_argument("--full", action="store_true",
                       help="Full experiment (10 seeds, 1000 generations)")
    parser.add_argument("--sigma-test", action="store_true",
                       help="Run sigma sensitivity analysis only")
    parser.add_argument("--dimensions", type=int, nargs="+", default=None,
                       help="Dimensions to test")
    parser.add_argument("--seeds", type=int, default=5,
                       help="Number of seeds")
    parser.add_argument("--generations", type=int, default=500,
                       help="Generations per run")
    
    args = parser.parse_args()
    
    if args.sigma_test:
        run_sigma_sensitivity(
            n_seeds=args.seeds,
            generations=args.generations,
        )
    elif args.quick:
        run_distance_comparison(
            dimensions=[5, 20],
            n_seeds=2,
            generations=200,
        )
    elif args.full:
        run_distance_comparison(
            dimensions=[5, 10, 20, 50],
            n_seeds=10,
            generations=1000,
        )
    else:
        run_distance_comparison(
            dimensions=args.dimensions or [5, 10, 20],
            n_seeds=args.seeds,
            generations=args.generations,
        )
