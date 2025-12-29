#!/usr/bin/env python3
"""
MMR-Elites: Publication-Quality Visualizations
===============================================

Generates all figures for the GECCO paper:
"Quality-Diversity as Information Retrieval: Overcoming the Curse of 
Dimensionality with Maximum Marginal Relevance Selection of Elites"

Figures:
1. The Problem (Curse of Dimensionality)
2. Main Comparison (Learning Curves)
3. Dimensionality Scaling
4. Arm Repertoire Visualization
5. Lambda Ablation
6. Uniformity Comparison
"""

import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Style Configuration (Publication Quality)
# =============================================================================

# Color palette (colorblind-friendly, Wong 2011)
COLORS = {
    "MMR-Elites": "#0072B2",       # Blue
    "MAP-Elites": "#D55E00",       # Orange/Vermillion
    "CVT-MAP-Elites": "#009E73",   # Green
    "Random": "#CC79A7",           # Pink
    "CMA-ME": "#F0E442",           # Yellow
}

MARKERS = {
    "MMR-Elites": "o",
    "MAP-Elites": "s",
    "CVT-MAP-Elites": "^",
    "Random": "x",
}

LINESTYLES = {
    "MMR-Elites": "-",
    "MAP-Elites": "--",
    "CVT-MAP-Elites": "-.",
    "Random": ":",
}

def set_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.rcParams.update({
        # Font
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        
        # Figure
        'figure.figsize': (7, 4),  # GECCO column width ~3.5in, full width ~7in
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        
        # Axes
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        
        # Grid
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        
        # Legend
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'gray',
    })


# =============================================================================
# Figure 1: The Problem (Curse of Dimensionality)
# =============================================================================

def plot_curse_of_dimensionality(save_path: Optional[Path] = None):
    """
    Illustrate exponential growth of grid cells with dimensionality.
    """
    set_publication_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    
    # Left: Exponential growth
    dims = np.arange(2, 25)
    bins_list = [3, 5, 10]
    
    for bins in bins_list:
        cells = bins ** dims
        ax1.semilogy(dims, cells, label=f'{bins} bins/dim', linewidth=2)
    
    # Memory limit line
    ax1.axhline(y=1e9, color='red', linestyle='--', alpha=0.7, label='1 GB limit')
    ax1.axhline(y=1e12, color='darkred', linestyle='--', alpha=0.7, label='1 TB limit')
    
    ax1.fill_between(dims, 1e9, 1e20, alpha=0.1, color='red')
    ax1.text(15, 1e15, 'Infeasible\nRegion', fontsize=10, ha='center', color='darkred')
    
    ax1.set_xlabel('Behavior Space Dimension')
    ax1.set_ylabel('Number of Grid Cells')
    ax1.set_title('(a) MAP-Elites Grid Scaling')
    ax1.legend(loc='lower right')
    ax1.set_xlim(2, 24)
    ax1.set_ylim(1, 1e20)
    
    # Right: Comparison of approaches
    algorithms = ['MAP-Elites\n(Grid)', 'CVT-MAP-Elites\n(Centroids)', 'MMR-Elites\n(Ours)']
    memory_20d = [
        3**20,  # MAP-Elites with 3 bins
        1000,   # CVT (fixed)
        1000,   # MMR (fixed)
    ]
    
    colors = [COLORS["MAP-Elites"], COLORS["CVT-MAP-Elites"], COLORS["MMR-Elites"]]
    bars = ax2.bar(algorithms, memory_20d, color=colors, edgecolor='black', linewidth=0.5)
    
    ax2.set_yscale('log')
    ax2.set_ylabel('Archive Size (20-D behavior space)')
    ax2.set_title('(b) Memory Requirements')
    
    # Add value labels
    for bar, val in zip(bars, memory_20d):
        if val > 1e6:
            label = f'{val:.0e}'
        else:
            label = f'{val}'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
                label, ha='center', va='bottom', fontsize=8)
    
    ax2.set_ylim(100, 1e15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / "fig1_curse_of_dimensionality.pdf")
        plt.savefig(save_path / "fig1_curse_of_dimensionality.png")
        print(f"Saved: {save_path}/fig1_curse_of_dimensionality.pdf")
    
    return fig


# =============================================================================
# Figure 2: Main Comparison (Learning Curves)
# =============================================================================

def plot_learning_curves(
    results: Dict[str, List[Dict]], 
    save_path: Optional[Path] = None,
    metrics: List[str] = None,
    smooth_sigma: float = 1.0
):
    """
    Plot learning curves with confidence intervals.
    """
    set_publication_style()
    
    if metrics is None:
        metrics = ["qd_score_at_budget", "max_fitness", "uniformity_cv"]
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 3))
    if n_metrics == 1:
        axes = [axes]
    
    titles = {
        "qd_score": "QD-Score (Total)",
        "qd_score_at_budget": "QD-Score @ K=1000",
        "max_fitness": "Maximum Fitness",
        "mean_fitness": "Mean Fitness",
        "uniformity_cv": "Uniformity (CV, ↓ better)",
        "mean_pairwise_distance": "Mean Pairwise Distance",
        "archive_size": "Archive Size",
    }
    
    for ax, metric in zip(axes, metrics):
        for alg_name, alg_results in results.items():
            if not alg_results:
                continue
            if metric not in alg_results[0]["history"]:
                continue
            
            color = COLORS.get(alg_name, "gray")
            linestyle = LINESTYLES.get(alg_name, "-")
            
            gens = alg_results[0]["history"]["generation"]
            values = np.array([r["history"][metric] for r in alg_results])
            
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0)
            
            # Smooth
            if smooth_sigma > 0:
                mean = gaussian_filter1d(mean, smooth_sigma)
                std = gaussian_filter1d(std, smooth_sigma)
            
            # 95% CI
            n = len(alg_results)
            ci = 1.96 * std / np.sqrt(n)
            
            ax.fill_between(gens, mean - ci, mean + ci, alpha=0.2, color=color)
            ax.plot(gens, mean, color=color, linestyle=linestyle, 
                   linewidth=2, label=alg_name)
        
        ax.set_xlabel("Generation")
        ax.set_ylabel(titles.get(metric, metric))
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / "fig2_learning_curves.pdf")
        plt.savefig(save_path / "fig2_learning_curves.png")
        print(f"Saved: {save_path}/fig2_learning_curves.pdf")
    
    return fig


# =============================================================================
# Figure 3: Dimensionality Scaling (Key Result)
# =============================================================================

def plot_dimensionality_scaling(
    results: Dict[int, Dict[str, List[Dict]]],
    save_path: Optional[Path] = None
):
    """
    Plot how algorithms scale with behavior space dimensionality.
    
    This is THE key figure for the paper.
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    
    dimensions = sorted(results.keys())
    metrics = ["qd_score_at_budget", "uniformity_cv", "coverage_efficiency"]
    titles = ["QD-Score @ K=1000 (↑)", "Uniformity CV (↓)", "Coverage Efficiency (↑)"]
    
    for ax, metric, title in zip(axes, metrics, titles):
        for alg in ["MMR-Elites", "CVT-MAP-Elites", "MAP-Elites", "Random"]:
            means = []
            stds = []
            
            for d in dimensions:
                if alg in results[d] and results[d][alg]:
                    vals = [r["final_metrics"][metric] for r in results[d][alg]]
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            
            means = np.array(means)
            stds = np.array(stds)
            
            color = COLORS.get(alg, "gray")
            marker = MARKERS.get(alg, "o")
            
            ax.errorbar(dimensions, means, yerr=stds, 
                       color=color, marker=marker, 
                       linewidth=2, markersize=8, capsize=4,
                       label=alg)
        
        ax.set_xlabel("Behavior Space Dimension")
        ax.set_ylabel(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        # Log scale for dimensions
        ax.set_xscale('log')
        ax.set_xticks(dimensions)
        ax.set_xticklabels([str(d) for d in dimensions])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / "fig3_dimensionality_scaling.pdf")
        plt.savefig(save_path / "fig3_dimensionality_scaling.png")
        print(f"Saved: {save_path}/fig3_dimensionality_scaling.pdf")
    
    return fig


# =============================================================================
# Figure 4: Arm Repertoire Visualization
# =============================================================================

def plot_arm_repertoire(
    results: Dict[str, Dict],
    n_samples: int = 30,
    save_path: Optional[Path] = None
):
    """
    Visualize arm configurations from each algorithm's archive.
    """
    set_publication_style()
    
    algorithms = ["MMR-Elites", "CVT-MAP-Elites", "MAP-Elites"]
    n_algs = len([a for a in algorithms if a in results and results[a]])
    
    fig, axes = plt.subplots(1, n_algs, figsize=(4 * n_algs, 4))
    if n_algs == 1:
        axes = [axes]
    
    def draw_arm(ax, genome, color, alpha=0.3):
        """Draw a single arm configuration."""
        n_dof = len(genome)
        link_len = 1.0 / n_dof
        
        angles = np.cumsum(genome)
        dx = link_len * np.cos(angles)
        dy = link_len * np.sin(angles)
        
        x = np.concatenate([[0], np.cumsum(dx)])
        y = np.concatenate([[0], np.cumsum(dy)])
        
        ax.plot(x, y, color=color, alpha=alpha, linewidth=1)
        ax.plot(x[-1], y[-1], 'o', color=color, alpha=alpha, markersize=3)
    
    ax_idx = 0
    for alg in algorithms:
        if alg not in results or not results[alg]:
            continue
        
        ax = axes[ax_idx]
        color = COLORS.get(alg, "gray")
        
        # Get archive from first seed
        if isinstance(results[alg], list):
            archive = results[alg][0]
        else:
            archive = results[alg]
        
        # We need genomes - reconstruct from descriptors if needed
        # For now, use fitness to color
        descriptors = archive.get("final_descriptors", np.array([]))
        fitness = archive.get("final_fitness", np.array([]))
        
        if len(descriptors) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax_idx += 1
            continue
        
        # If descriptors are normalized joint angles, denormalize
        if descriptors.shape[1] >= 5:  # Likely joint angles
            genomes = descriptors * 2 * np.pi - np.pi
        else:
            # Can't visualize 2D descriptors as arms
            ax.text(0.5, 0.5, "2D descriptors\n(not visualizable)", 
                   ha='center', va='center')
            ax_idx += 1
            continue
        
        # Sample configurations
        if len(genomes) > n_samples:
            idx = np.random.choice(len(genomes), n_samples, replace=False)
            genomes = genomes[idx]
            fitness = fitness[idx] if len(fitness) > 0 else fitness
        
        # Draw arms
        for i, genome in enumerate(genomes):
            draw_arm(ax, genome, color, alpha=0.4)
        
        # Draw obstacle
        obstacle = plt.Rectangle((0.5, -0.25), 0.05, 0.5, 
                                  color='red', alpha=0.5)
        ax.add_patch(obstacle)
        
        # Draw target
        ax.scatter([0.8], [0.0], c='green', s=150, marker='*', 
                  zorder=10, edgecolors='black')
        
        # Draw base
        ax.plot(0, 0, 'ko', markersize=10)
        
        ax.set_xlim(-0.5, 1.2)
        ax.set_ylim(-0.8, 0.8)
        ax.set_aspect('equal')
        ax.set_title(f"{alg}\n(N={len(archive.get('final_fitness', []))})")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        
        ax_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / "fig4_arm_repertoire.pdf")
        plt.savefig(save_path / "fig4_arm_repertoire.png")
        print(f"Saved: {save_path}/fig4_arm_repertoire.pdf")
    
    return fig


# =============================================================================
# Figure 5: Lambda Ablation
# =============================================================================

def plot_lambda_ablation(
    results: Dict[float, List[Dict]],
    save_path: Optional[Path] = None
):
    """
    Plot effect of λ parameter on MMR-Elites.
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3))
    
    lambdas = sorted(results.keys())
    metrics = ["qd_score", "max_fitness", "uniformity_cv"]
    titles = ["QD-Score", "Max Fitness", "Uniformity CV (↓ better)"]
    
    for ax, metric, title in zip(axes, metrics, titles):
        means = []
        stds = []
        
        for lam in lambdas:
            vals = [r["final_metrics"][metric] for r in results[lam]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        
        ax.errorbar(lambdas, means, yerr=stds,
                   color=COLORS["MMR-Elites"], marker='o',
                   linewidth=2, markersize=8, capsize=4)
        
        ax.set_xlabel("λ (Diversity Weight)")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        
        # Highlight optimal region
        if metric == "uniformity_cv":
            ax.axhspan(0, 0.15, alpha=0.1, color='green')
            ax.text(0.5, 0.08, 'Good uniformity', fontsize=8, ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / "fig5_lambda_ablation.pdf")
        plt.savefig(save_path / "fig5_lambda_ablation.png")
        print(f"Saved: {save_path}/fig5_lambda_ablation.pdf")
    
    return fig


# =============================================================================
# Figure 6: Uniformity Comparison (Archive Distribution)
# =============================================================================

def plot_uniformity_comparison(
    results: Dict[str, Dict],
    save_path: Optional[Path] = None
):
    """
    Compare archive uniformity across algorithms using 2D projections.
    """
    set_publication_style()
    
    from sklearn.decomposition import PCA
    
    algorithms = ["MMR-Elites", "CVT-MAP-Elites", "MAP-Elites"]
    n_algs = len([a for a in algorithms if a in results and results[a]])
    
    fig, axes = plt.subplots(1, n_algs, figsize=(4 * n_algs, 4))
    if n_algs == 1:
        axes = [axes]
    
    ax_idx = 0
    for alg in algorithms:
        if alg not in results or not results[alg]:
            continue
        
        ax = axes[ax_idx]
        color = COLORS.get(alg, "gray")
        
        if isinstance(results[alg], list):
            archive = results[alg][0]
        else:
            archive = results[alg]
        
        descriptors = archive.get("final_descriptors", np.array([]))
        fitness = archive.get("final_fitness", np.array([]))
        
        if len(descriptors) == 0:
            ax_idx += 1
            continue
        
        # Project to 2D if needed
        if descriptors.shape[1] > 2:
            pca = PCA(n_components=2)
            proj = pca.fit_transform(descriptors)
        else:
            proj = descriptors
        
        # Color by fitness
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=fitness, cmap='viridis',
                       s=20, alpha=0.6, edgecolors='none')
        
        # Compute uniformity
        from scipy.spatial import cKDTree
        if len(descriptors) > 5:
            tree = cKDTree(descriptors)
            dists, _ = tree.query(descriptors, k=6)
            mean_knn = np.mean(dists[:, 1:], axis=1)
            cv = np.std(mean_knn) / np.mean(mean_knn)
        else:
            cv = 0
        
        ax.set_title(f"{alg}\nUniformity CV: {cv:.3f}")
        ax.set_xlabel("PC1" if descriptors.shape[1] > 2 else "Dim 1")
        ax.set_ylabel("PC2" if descriptors.shape[1] > 2 else "Dim 2")
        
        ax_idx += 1
    
    # Colorbar
    plt.colorbar(sc, ax=axes, label="Fitness", shrink=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / "fig6_uniformity_comparison.pdf")
        plt.savefig(save_path / "fig6_uniformity_comparison.png")
        print(f"Saved: {save_path}/fig6_uniformity_comparison.pdf")
    
    return fig


# =============================================================================
# Generate All Figures
# =============================================================================

def generate_all_figures(results_dir: Path, output_dir: Path):
    """Generate all paper figures from experiment results."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Always generate (no data needed)
    print("Generating Figure 1: Curse of Dimensionality...")
    plot_curse_of_dimensionality(output_dir)
    
    # Try to load results
    results_files = {
        "dimensionality": results_dir / "dimensionality_scaling.pkl",
        "lambda": results_dir / "lambda_ablation.pkl",
        "archive": results_dir / "archive_size_ablation.pkl",
        "main": results_dir / "all_results.pkl",
    }
    
    # Figure 2 & 3: From dimensionality scaling
    if results_files["dimensionality"].exists():
        print("Loading dimensionality scaling results...")
        with open(results_files["dimensionality"], "rb") as f:
            dim_results = pickle.load(f)
        
        # Figure 3: Dimensionality scaling
        print("Generating Figure 3: Dimensionality Scaling...")
        plot_dimensionality_scaling(dim_results, output_dir)
        
        # Figure 2: Learning curves (use 20-DOF results)
        if 20 in dim_results:
            print("Generating Figure 2: Learning Curves...")
            plot_learning_curves(dim_results[20], output_dir)
            
            # Figure 4: Arm repertoire
            print("Generating Figure 4: Arm Repertoire...")
            plot_arm_repertoire(dim_results[20], save_path=output_dir)
            
            # Figure 6: Uniformity comparison
            print("Generating Figure 6: Uniformity Comparison...")
            try:
                plot_uniformity_comparison(dim_results[20], output_dir)
            except ImportError:
                print("  Skipping (sklearn required)")
    
    # Figure 5: Lambda ablation
    if results_files["lambda"].exists():
        print("Loading lambda ablation results...")
        with open(results_files["lambda"], "rb") as f:
            lambda_results = pickle.load(f)
        
        print("Generating Figure 5: Lambda Ablation...")
        plot_lambda_ablation(lambda_results, output_dir)
    
    # Try main comparison results
    if results_files["main"].exists():
        print("Loading main comparison results...")
        with open(results_files["main"], "rb") as f:
            main_results = pickle.load(f)
        
        print("Generating Figure 2 (from main results)...")
        plot_learning_curves(main_results, output_dir)
    
    print(f"\n✅ All figures saved to: {output_dir}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results-dir", type=Path, default=Path("results/full_benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--figure", type=int, help="Generate specific figure (1-6)")
    
    args = parser.parse_args()
    
    if args.figure == 1:
        plot_curse_of_dimensionality(args.output_dir)
    else:
        generate_all_figures(args.results_dir, args.output_dir)
