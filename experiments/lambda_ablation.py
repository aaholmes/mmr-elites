"""
Lambda Ablation Study.
Effect of λ on fitness and diversity.
"""

import numpy as np
import matplotlib.pyplot as plt
from mmr_elites.utils.config import ExperimentConfig
from experiments.run_benchmark import run_experiment
from mmr_elites.utils.visualization import set_publication_style, save_figure


def main():
    set_publication_style()
    
    lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    seeds = [42, 43, 44]
    
    results = {lam: [] for lam in lambdas}
    
    for lam in lambdas:
        print(f"\n--- Testing Lambda λ={lam} ---")
        for seed in seeds:
            config = ExperimentConfig(
                task="arm",
                algorithm="mmr_elites",
                n_dof=20,
                lambda_val=lam,
                generations=300,
                seed=seed,
                exp_name=f"ablation_lam{lam}"
            )
            res = run_experiment(config)
            results[lam].append(res["final_metrics"])

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [
        ("qd_score_at_budget", "QD-Score (Top 1000)"),
        ("max_fitness", "Max Fitness"),
        ("mean_pairwise_distance", "Diversity (Mean Dist)")
    ]
    
    for i, (m_key, m_name) in enumerate(metrics):
        ax = axes[i]
        means = [np.mean([r[m_key] for r in results[lam]]) for lam in lambdas]
        stds = [np.std([r[m_key] for r in results[lam]]) for lam in lambdas]
        
        ax.errorbar(lambdas, means, yerr=stds, marker='o', capsize=4, linewidth=2)
        ax.set_xlabel("λ (Diversity Weight)")
        ax.set_ylabel(m_name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "paper/figures/lambda_ablation")
    plt.show()

if __name__ == "__main__":
    main()
