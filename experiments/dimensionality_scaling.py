"""
Dimensionality Scaling Experiment.
Compares MMR-Elites vs MAP-Elites vs CVT-MAP-Elites as D increases.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mmr_elites.utils.config import ExperimentConfig
from experiments.run_benchmark import run_experiment
from mmr_elites.utils.visualization import set_publication_style, save_figure


def main():
    set_publication_style()
    
    # Dimensions to test
    dims = [2, 5, 10, 20, 50]
    algorithms = ["mmr_elites", "map_elites", "cvt_map_elites", "random"]
    seeds = [42, 43, 44]
    
    results = {alg: {d: [] for d in dims} for alg in algorithms}
    
    for d in dims:
        print(f"\n--- Testing Dimension D={d} ---")
        for alg in algorithms:
            # Skip MAP-Elites for high D (it will OOM or be too slow)
            if alg == "map_elites" and d > 10:
                print(f"Skipping MAP-Elites for D={d}")
                continue
                
            for seed in seeds:
                config = ExperimentConfig(
                    task="arm",
                    algorithm=alg,
                    n_dof=d,
                    generations=200,
                    batch_size=100,
                    seed=seed,
                    exp_name=f"scaling_d{d}"
                )
                try:
                    res = run_experiment(config)
                    results[alg][d].append(res["final_metrics"]["qd_score_at_budget"])
                except Exception as e:
                    print(f"Error running {alg} at D={d}: {e}")

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for alg in algorithms:
        means = []
        stds = []
        valid_dims = []
        for d in dims:
            if results[alg][d]:
                means.append(np.mean(results[alg][d]))
                stds.append(np.std(results[alg][d]))
                valid_dims.append(d)
        
        if valid_dims:
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(valid_dims, means, marker='o', label=alg, linewidth=2)
            ax.fill_between(valid_dims, means - stds, means + stds, alpha=0.2)

    ax.set_xlabel("Behavior Space Dimension (D)")
    ax.set_ylabel("QD-Score (Top 1000)")
    ax.set_title("Dimensionality Scaling Comparison")
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    save_figure(fig, "paper/figures/dimensionality_scaling")
    plt.show()

if __name__ == "__main__":
    main()
