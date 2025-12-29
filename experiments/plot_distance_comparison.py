# experiments/plot_distance_comparison.py
"""
Visualizations for distance function comparison experiment.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path


def set_style():
    """Set publication-quality style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
    })


# Color scheme
COLORS = {
    "Euclidean": "#D55E00",
    "Normalized": "#CC79A7", 
    "Cosine": "#009E73",
    "Saturating(σ=0.1)": "#0072B2",
    "Saturating(σ=0.2)": "#56B4E9",
    "Saturating(σ=0.3)": "#E69F00",
    "Saturating(σ=0.5)": "#F0E442",
}


def plot_coverage_comparison(results: dict, dim: int, save_path: Path = None):
    """
    Side-by-side scatter plots showing archive distribution for each distance.
    """
    set_style()
    
    dist_names = list(results[dim].keys())
    n_dist = len(dist_names)
    
    fig, axes = plt.subplots(1, min(n_dist, 4), figsize=(4 * min(n_dist, 4), 4))
    if n_dist == 1:
        axes = [axes]
    
    # Use first seed for visualization
    for ax, name in zip(axes, dist_names[:4]):
        if not results[dim][name]:
            continue
        
        desc = results[dim][name][0]["final_descriptors"]
        
        # Project to 2D if needed
        if desc.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            proj = pca.fit_transform(desc)
        else:
            proj = desc[:, :2]
        
        color = COLORS.get(name, "gray")
        ax.scatter(proj[:, 0], proj[:, 1], c=color, alpha=0.5, s=15, edgecolors='none')
        
        # Compute metrics
        cv = results[dim][name][0]["final_metrics"]["uniformity_cv"]
        ir = results[dim][name][0]["coverage_metrics"]["interior_ratio"]
        
        ax.set_title(f"{name}\nCV={cv:.3f}, Interior={ir:.0%}")
        ax.set_xlabel("PC1" if desc.shape[1] > 2 else "Dim 1")
        ax.set_ylabel("PC2" if desc.shape[1] > 2 else "Dim 2")
    
    plt.suptitle(f"Archive Distribution Comparison ({dim}D)", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / f"coverage_comparison_{dim}d.pdf")
        plt.savefig(save_path / f"coverage_comparison_{dim}d.png")
    
    return fig


def plot_metrics_comparison(results: dict, save_path: Path = None):
    """
    Bar chart comparing key metrics across distance functions.
    """
    set_style()
    
    dimensions = sorted([d for d in results.keys() if isinstance(d, int)])
    dist_names = list(results[dimensions[0]].keys())
    
    metrics = ["qd_score_at_budget", "uniformity_cv", "interior_ratio"]
    metric_labels = ["QD-Score @ K", "Uniformity CV (↓)", "Interior Ratio (↑)"]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    
    x = np.arange(len(dimensions))
    width = 0.8 / len(dist_names)
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        for i, name in enumerate(dist_names):
            means = []
            stds = []
            
            for dim in dimensions:
                if metric == "interior_ratio":
                    vals = [r["coverage_metrics"][metric] for r in results[dim][name]]
                else:
                    vals = [r["final_metrics"].get(metric, 0) for r in results[dim][name]]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            
            color = COLORS.get(name, "gray")
            offset = (i - len(dist_names)/2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, label=name,
                   color=color, capsize=2, alpha=0.8)
        
        ax.set_ylabel(label)
        ax.set_xlabel("Behavior Space Dimension")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}D" for d in dimensions])
        
        if ax == axes[-1]:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / "metrics_comparison.pdf", bbox_inches='tight')
        plt.savefig(save_path / "metrics_comparison.png", bbox_inches='tight')
    
    return fig


def plot_sigma_sensitivity(results: dict, save_path: Path = None):
    """
    Line plot showing effect of sigma on different metrics.
    """
    set_style()
    
    sigmas = sorted(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    metrics = ["qd_score_at_budget", "uniformity_cv", "interior_ratio"]
    labels = ["QD-Score @ K", "Uniformity CV (↓)", "Interior Ratio (↑)"]
    
    for ax, metric, label in zip(axes, metrics, labels):
        means = []
        stds = []
        
        for sigma in sigmas:
            if metric == "interior_ratio":
                vals = [r["coverage_metrics"][metric] for r in results[sigma]]
            else:
                vals = [r["final_metrics"].get(metric, 0) for r in results[sigma]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        
        ax.errorbar(sigmas, means, yerr=stds, marker='o', linewidth=2,
                   markersize=8, capsize=4, color="#0072B2")
        ax.set_xlabel("σ (Saturation Scale)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        
        # Highlight optimal region
        if metric == "uniformity_cv":
            ax.axhline(y=min(means), color='green', linestyle='--', alpha=0.5)
    
    plt.suptitle("Effect of σ on Saturating Distance Performance", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / "sigma_sensitivity.pdf")
        plt.savefig(save_path / "sigma_sensitivity.png")
    
    return fig


def generate_all_figures(results_dir: Path, output_dir: Path):
    """Generate all figures for the paper."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load distance comparison results
    dist_results_file = results_dir / "distance_comparison" / "results.pkl"
    if dist_results_file.exists():
        with open(dist_results_file, "rb") as f:
            dist_results = pickle.load(f)
        
        for dim in dist_results.keys():
            plot_coverage_comparison(dist_results, dim, output_dir)
        
        plot_metrics_comparison(dist_results, output_dir)
    
    # Load sigma sensitivity results
    sigma_results_file = results_dir / "sigma_sensitivity" / "results.pkl"
    if sigma_results_file.exists():
        with open(sigma_results_file, "rb") as f:
            sigma_results = pickle.load(f)
        
        plot_sigma_sensitivity(sigma_results, output_dir)
    
    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    generate_all_figures(
        results_dir=Path("results"),
        output_dir=Path("figures/distance_comparison")
    )
