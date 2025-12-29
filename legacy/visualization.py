"""
Visualization Module for MUSE-QD
=================================

Generates publication-quality figures for GECCO/EvoStar papers.

Style guidelines:
- Use colorblind-friendly palettes
- Include confidence intervals for all curves
- Label everything clearly
- Export as PDF for vector graphics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# Style Configuration
# =============================================================================

# Colorblind-friendly palette (Wong, 2011)
COLORS = {
    "muse_qd": "#0072B2",      # Blue
    "map_elites": "#D55E00",   # Orange
    "cvt_map_elites": "#009E73", # Green
    "random": "#CC79A7",       # Pink
}

LINESTYLES = {
    "muse_qd": "-",
    "map_elites": "--",
    "cvt_map_elites": "-.",
    "random": ":",
}

# Publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def set_publication_style():
    """Apply publication-ready matplotlib style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
    })


# =============================================================================
# Core Plotting Functions
# =============================================================================

def plot_learning_curves(
    results_dict: Dict[str, List],  # algorithm_name -> list of RunResult
    metric: str = "qd_score",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    smooth_sigma: float = 2.0,
    show_individual: bool = False,
) -> plt.Axes:
    """
    Plot learning curves with confidence intervals.
    
    Args:
        results_dict: Dictionary mapping algorithm names to lists of results
        metric: Metric to plot ("qd_score", "max_fitness", etc.)
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        ylabel: Y-axis label
        smooth_sigma: Gaussian smoothing sigma (0 for no smoothing)
        show_individual: Whether to show individual run traces
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    for alg_name, results in results_dict.items():
        color = COLORS.get(alg_name.lower().replace("-", "_"), "gray")
        linestyle = LINESTYLES.get(alg_name.lower().replace("-", "_"), "-")
        
        # Extract histories
        generations = results[0].history["generation"]
        metric_values = np.array([r.history[metric] for r in results])
        
        # Compute mean and confidence interval
        mean = np.mean(metric_values, axis=0)
        std = np.std(metric_values, axis=0)
        
        # 95% confidence interval (assuming normal distribution)
        n_runs = len(results)
        ci = 1.96 * std / np.sqrt(n_runs)
        
        # Optional smoothing
        if smooth_sigma > 0:
            mean = gaussian_filter1d(mean, smooth_sigma)
            ci = gaussian_filter1d(ci, smooth_sigma)
        
        # Plot individual runs (faint)
        if show_individual:
            for run_values in metric_values:
                if smooth_sigma > 0:
                    run_values = gaussian_filter1d(run_values, smooth_sigma)
                ax.plot(generations, run_values, color=color, alpha=0.15, linewidth=0.5)
        
        # Plot mean with CI
        ax.fill_between(generations, mean - ci, mean + ci, color=color, alpha=0.2)
        ax.plot(generations, mean, color=color, linestyle=linestyle, 
                linewidth=2, label=alg_name)
    
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel or metric.replace("_", " ").title())
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    
    return ax


def plot_final_metrics_bars(
    results_dict: Dict[str, List],
    metrics: List[str] = None,
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Bar chart comparing final metrics across algorithms.
    
    Args:
        results_dict: Dictionary mapping algorithm names to lists of results
        metrics: List of metrics to plot
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ["qd_score", "max_fitness", "mean_pairwise_distance"]
    
    n_metrics = len(metrics)
    n_algorithms = len(results_dict)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    x = np.arange(n_algorithms)
    width = 0.6
    
    alg_names = list(results_dict.keys())
    colors = [COLORS.get(name.lower().replace("-", "_"), "gray") for name in alg_names]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        means = []
        stds = []
        
        for alg_name, results in results_dict.items():
            values = [r.final_metrics[metric] for r in results]
            means.append(np.mean(values))
            stds.append(np.std(values))
        
        bars = ax.bar(x, means, width, yerr=stds, capsize=4, 
                      color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(alg_names, rotation=15, ha='right')
        
        # Add significance markers if applicable
        # (Would need statistical tests - simplified here)
    
    plt.tight_layout()
    return fig


def plot_behavior_space_comparison(
    results_dict: Dict[str, np.ndarray],  # algorithm_name -> descriptors (N, 2)
    bounds: Tuple[float, float, float, float] = None,  # (xmin, xmax, ymin, ymax)
    titles: Dict[str, str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Side-by-side scatter plots of final archive positions in 2D behavior space.
    
    Args:
        results_dict: Dictionary mapping algorithm names to 2D descriptor arrays
        bounds: Axis bounds (xmin, xmax, ymin, ymax)
        titles: Custom titles for each subplot
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    n_algorithms = len(results_dict)
    fig, axes = plt.subplots(1, n_algorithms, figsize=figsize)
    
    if n_algorithms == 1:
        axes = [axes]
    
    for i, (alg_name, descriptors) in enumerate(results_dict.items()):
        ax = axes[i]
        color = COLORS.get(alg_name.lower().replace("-", "_"), "gray")
        
        # Scatter plot
        ax.scatter(descriptors[:, 0], descriptors[:, 1], 
                   c=color, alpha=0.5, s=20, edgecolors='none')
        
        # Styling
        title = titles.get(alg_name, alg_name) if titles else alg_name
        ax.set_title(f"{title}\n(N={len(descriptors)})")
        ax.set_xlabel("Behavior Dim 1")
        ax.set_ylabel("Behavior Dim 2")
        ax.set_aspect('equal')
        
        if bounds:
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
    
    plt.tight_layout()
    return fig


def plot_arm_configurations(
    genomes: np.ndarray,
    n_samples: int = 10,
    target_pos: Tuple[float, float] = (0.8, 0.0),
    obstacle_box: Tuple[float, float, float, float] = (0.5, 0.55, -0.25, 0.25),
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Visualize sampled arm configurations from the archive.
    
    Args:
        genomes: Array of arm joint angles (N, 20)
        n_samples: Number of configurations to show
        target_pos: Target position for end effector
        obstacle_box: Obstacle bounding box (x_min, x_max, y_min, y_max)
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sample configurations
    if len(genomes) > n_samples:
        indices = np.random.choice(len(genomes), n_samples, replace=False)
        samples = genomes[indices]
    else:
        samples = genomes
    
    # Draw obstacle
    ox_min, ox_max, oy_min, oy_max = obstacle_box
    obstacle = plt.Rectangle((ox_min, oy_min), ox_max - ox_min, oy_max - oy_min,
                              color='red', alpha=0.5, label='Obstacle')
    ax.add_patch(obstacle)
    
    # Draw target
    ax.scatter([target_pos[0]], [target_pos[1]], c='green', s=200, 
               marker='*', zorder=10, label='Target')
    
    # Draw arms
    dof = 20
    link_len = 1.0 / dof
    
    for genome in samples:
        angles = np.cumsum(genome)
        dx = link_len * np.cos(angles)
        dy = link_len * np.sin(angles)
        
        x = np.concatenate([[0], np.cumsum(dx)])
        y = np.concatenate([[0], np.cumsum(dy)])
        
        ax.plot(x, y, 'b-', linewidth=1, alpha=0.4)
        ax.plot(x[-1], y[-1], 'bo', markersize=4, alpha=0.6)
    
    # Draw origin
    ax.plot(0, 0, 'ko', markersize=8, label='Base')
    
    ax.set_xlim(-0.5, 1.2)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Archive Arm Configurations (N={len(samples)})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_lambda_ablation(
    lambda_values: List[float],
    qd_scores: List[Tuple[float, float]],  # (mean, std) for each lambda
    max_fitnesses: List[Tuple[float, float]],
    diversities: List[Tuple[float, float]],
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Ablation study: Effect of lambda on different metrics.
    
    Args:
        lambda_values: List of lambda values tested
        qd_scores: (mean, std) tuples for QD-Score at each lambda
        max_fitnesses: (mean, std) tuples for max fitness
        diversities: (mean, std) tuples for diversity metric
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    metrics = [
        ("QD-Score", qd_scores),
        ("Max Fitness", max_fitnesses),
        ("Diversity (Mean Dist)", diversities),
    ]
    
    for ax, (name, values) in zip(axes, metrics):
        means = [v[0] for v in values]
        stds = [v[1] for v in values]
        
        ax.errorbar(lambda_values, means, yerr=stds, 
                    marker='o', capsize=4, linewidth=2, color=COLORS["muse_qd"])
        ax.set_xlabel("λ (Diversity Weight)")
        ax.set_ylabel(name)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Convenience Functions
# =============================================================================

def save_figure(fig: plt.Figure, path: str, formats: List[str] = None):
    """
    Save figure in multiple formats.
    
    Args:
        fig: Matplotlib figure
        path: Base path (without extension)
        formats: List of formats (default: ["pdf", "png"])
    """
    if formats is None:
        formats = ["pdf", "png"]
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", format=fmt, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}.{fmt}")


def create_all_figures(results_dir: Path, output_dir: Path):
    """
    Generate all standard figures from experiment results.
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save figures
    """
    set_publication_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_dict = {}
    
    for alg_dir in results_dir.iterdir():
        if alg_dir.is_dir() and (alg_dir / "results.pkl").exists():
            with open(alg_dir / "results.pkl", "rb") as f:
                results = pickle.load(f)
            results_dict[alg_dir.name] = results
    
    if not results_dict:
        print("No results found!")
        return
    
    print(f"Loaded results for: {list(results_dict.keys())}")
    
    # 1. Learning curves
    for metric in ["qd_score", "max_fitness", "mean_pairwise_distance"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_learning_curves(results_dict, metric=metric, ax=ax)
        save_figure(fig, output_dir / f"learning_curve_{metric}")
        plt.close(fig)
    
    # 2. Final metrics bars
    fig = plot_final_metrics_bars(results_dict)
    save_figure(fig, output_dir / "final_metrics_comparison")
    plt.close(fig)
    
    print(f"All figures saved to {output_dir}")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"),
                        help="Directory to save figures")
    
    args = parser.parse_args()
    create_all_figures(args.results_dir, args.output_dir)
