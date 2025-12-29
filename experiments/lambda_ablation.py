"""
Lambda Ablation Experiment
==========================

Tests the effect of λ parameter on MMR-Elites performance.

λ = 0: Pure fitness selection (no diversity)
λ = 1: Pure diversity selection (no fitness)
λ = 0.5: Balanced (default)
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List

from mmr_elites.tasks.arm import ArmTask
from mmr_elites.algorithms import run_mmr_elites


def run_lambda_ablation(
    lambda_values: List[float] = None,
    n_seeds: int = 10,
    generations: int = 2000,
    n_dof: int = 20,
    archive_size: int = 1000,
    batch_size: int = 200,
    mutation_sigma: float = 0.1,
    output_dir: Path = None,
) -> Dict:
    """
    Run lambda ablation experiment.
    
    Args:
        lambda_values: List of λ values to test
        n_seeds: Number of random seeds
        generations: Generations per run
        n_dof: Arm degrees of freedom
        archive_size: Archive size K
        batch_size: Offspring per generation
        mutation_sigma: Mutation standard deviation
        output_dir: Where to save results
    
    Returns:
        Dictionary with all results
    """
    if lambda_values is None:
        lambda_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    if output_dir is None:
        output_dir = Path("results/lambda_ablation")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task = ArmTask(n_dof=n_dof, use_highdim_descriptor=True)
    
    all_results = {lam: [] for lam in lambda_values}
    
    print("Lambda Ablation Experiment")
    print(f"Task: {n_dof}-DOF Arm")
    print(f"Lambda values: {lambda_values}")
    print(f"Seeds: {n_seeds}")
    
    for lam in lambda_values:
        print(f"\n{'='*50}")
        print(f"λ = {lam}")
        print("="*50)
        
        for seed in range(n_seeds):
            print(f"  Seed {seed + 1}/{n_seeds}...", end=" ", flush=True)
            
            result = run_mmr_elites(
                task,
                archive_size=archive_size,
                generations=generations,
                batch_size=batch_size,
                lambda_val=lam,
                mutation_sigma=mutation_sigma,
                seed=seed,
            )
            all_results[lam].append(result)
            
            print(f"QD@K={result['final_metrics']['qd_score_at_budget']:.2f}, "
                  f"CV={result['final_metrics']['uniformity_cv']:.3f}")
    
    # Save results
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    # Generate and save summary
    summary = {}
    for lam in lambda_values:
        metrics = {}
        for metric in ["qd_score_at_budget", "mean_fitness", "max_fitness", "uniformity_cv"]:
            values = [r["final_metrics"][metric] for r in all_results[lam]]
            metrics[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        summary[str(lam)] = metrics
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'lambda':<8} | {'QD@K':<18} | {'Max Fit':<18} | {'Uniformity CV':<18}")
    print("-"*80)
    
    for lam in lambda_values:
        qd = summary[str(lam)]["qd_score_at_budget"]
        mf = summary[str(lam)]["max_fitness"]
        cv = summary[str(lam)]["uniformity_cv"]
        print(f"{lam:<8.1f} | {qd['mean']:>7.2f} ± {qd['std']:<6.2f} | "
              f"{mf['mean']:>7.4f} ± {mf['std']:<6.4f} | "
              f"{cv['mean']:>7.4f} ± {cv['std']:<6.4f}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    if args.quick:
        run_lambda_ablation(
            lambda_values=[0.0, 0.5, 1.0],
            n_seeds=2,
            generations=200,
        )
    else:
        run_lambda_ablation()
