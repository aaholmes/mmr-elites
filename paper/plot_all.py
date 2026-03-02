#!/usr/bin/env python3
"""
Publication Figure Generation Pipeline
=======================================

Generates all figures for the MMR-Elites paper from experiment results.

Usage:
    python paper/plot_all.py                          # Use latest results
    python paper/plot_all.py --results-dir results/run_20260301_120000
    python paper/plot_all.py --only scaling           # Generate only one figure
"""

import argparse
import pickle
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

# Publication style
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
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
})

# Colorblind-friendly palette (Wong, 2011)
COLORS = {
    "MMR-Elites": "#0072B2",
    "MAP-Elites": "#D55E00",
    "CVT-MAP-Elites": "#009E73",
    "Random": "#CC79A7",
}

MARKERS = {
    "MMR-Elites": "o",
    "MAP-Elites": "s",
    "CVT-MAP-Elites": "^",
    "Random": "D",
}

LINESTYLES = {
    "MMR-Elites": "-",
    "MAP-Elites": "--",
    "CVT-MAP-Elites": "-.",
    "Random": ":",
}

FIGURE_DIR = Path("paper/figures")


def save_fig(fig, name):
    """Save figure as PDF and PNG."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        path = FIGURE_DIR / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight')
    print(f"  Saved: {FIGURE_DIR / name}.{{pdf,png}}")
    plt.close(fig)


# =============================================================================
# Figure 1: Dimensionality Scaling (THE key figure)
# =============================================================================

def plot_dimensionality_scaling(results_dir: Path):
    """
    Three-panel figure: QD-Score@K, Uniformity CV, Mean Fitness vs dimension.
    """
    print("Generating Figure 1: Dimensionality Scaling...")

    summary_path = results_dir / "dimensionality_scaling" / "summary.json"
    if not summary_path.exists():
        print(f"  WARNING: {summary_path} not found, skipping")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    dimensions = sorted([int(d) for d in summary.keys()])
    algorithms = [a for a in ["MMR-Elites", "MAP-Elites", "CVT-MAP-Elites", "Random"]
                  if a in summary[str(dimensions[0])]]

    metrics = [
        ("qd_score_at_budget", "QD-Score @ $K$"),
        ("uniformity_cv", "Uniformity CV $\\downarrow$"),
        ("mean_fitness", "Mean Fitness"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (metric, ylabel) in zip(axes, metrics):
        for alg in algorithms:
            means = []
            stds = []
            valid_dims = []
            for dim in dimensions:
                if alg in summary[str(dim)] and metric in summary[str(dim)][alg]:
                    means.append(summary[str(dim)][alg][metric]["mean"])
                    stds.append(summary[str(dim)][alg][metric]["std"])
                    valid_dims.append(dim)

            if not valid_dims:
                continue

            means = np.array(means)
            stds = np.array(stds)

            ax.errorbar(valid_dims, means, yerr=stds,
                        color=COLORS.get(alg, "gray"),
                        marker=MARKERS.get(alg, "o"),
                        linestyle=LINESTYLES.get(alg, "-"),
                        linewidth=2, markersize=6, capsize=3,
                        label=alg)

        ax.set_xlabel("Behavior Space Dimension")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_xticks(dimensions)
        ax.set_xticklabels([str(d) for d in dimensions])
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best")
    fig.tight_layout()
    save_fig(fig, "dimensionality_scaling")


# =============================================================================
# Figure 2: Lambda Ablation (Pareto frontier)
# =============================================================================

def plot_lambda_ablation(results_dir: Path):
    """
    Scatter plot: mean fitness vs uniformity CV, each point is a lambda value.
    """
    print("Generating Figure 2: Lambda Ablation...")

    summary_path = results_dir / "lambda_ablation" / "summary.json"
    if not summary_path.exists():
        print(f"  WARNING: {summary_path} not found, skipping")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 5))

    lambdas = sorted([float(l) for l in summary.keys()])

    mean_fitness = []
    uniformity_cv = []
    labels = []

    for lam in lambdas:
        key = str(lam)
        if key not in summary:
            continue
        mf = summary[key]["mean_fitness"]["mean"]
        cv = summary[key]["uniformity_cv"]["mean"]
        mean_fitness.append(mf)
        uniformity_cv.append(cv)
        labels.append(f"$\\lambda={lam}$")

    # Color by lambda value
    scatter = ax.scatter(mean_fitness, uniformity_cv,
                         c=lambdas, cmap="viridis",
                         s=100, edgecolors="black", linewidths=0.5, zorder=5)

    # Connect points with line
    ax.plot(mean_fitness, uniformity_cv, 'k--', alpha=0.3, linewidth=1)

    # Label each point
    for i, label in enumerate(labels):
        ax.annotate(label, (mean_fitness[i], uniformity_cv[i]),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=8)

    cbar = plt.colorbar(scatter, ax=ax, label="$\\lambda$")
    ax.set_xlabel("Mean Fitness")
    ax.set_ylabel("Uniformity CV $\\downarrow$")
    ax.set_title("Fitness--Diversity Tradeoff ($d=20$)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "lambda_ablation")


# =============================================================================
# Figure 3: Learning Curves
# =============================================================================

def plot_learning_curves(results_dir: Path):
    """
    QD-Score@K vs generation for all algorithms at dim=20.
    """
    print("Generating Figure 3: Learning Curves...")

    pkl_path = results_dir / "dimensionality_scaling" / "results.pkl"
    if not pkl_path.exists():
        print(f"  WARNING: {pkl_path} not found, skipping")
        return

    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    # Use dim=20 if available
    target_dim = 20
    if target_dim not in results:
        target_dim = list(results.keys())[0]
        print(f"  Using dim={target_dim} (20 not available)")

    dim_results = results[target_dim]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    metrics_to_plot = [
        ("qd_score_at_budget", "QD-Score @ $K$"),
        ("uniformity_cv", "Uniformity CV $\\downarrow$"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics_to_plot):
        for alg_name, runs in dim_results.items():
            if not runs:
                continue

            color = COLORS.get(alg_name, "gray")
            linestyle = LINESTYLES.get(alg_name, "-")

            # Extract history
            all_gens = []
            all_vals = []
            for r in runs:
                hist = r["history"] if isinstance(r, dict) else r.history
                if metric in hist and len(hist[metric]) > 0:
                    all_gens.append(hist["generation"])
                    all_vals.append(hist[metric])

            if not all_vals:
                continue

            # Find common generations
            min_len = min(len(v) for v in all_vals)
            gens = all_gens[0][:min_len]
            vals = np.array([v[:min_len] for v in all_vals])

            mean = np.mean(vals, axis=0)
            std = np.std(vals, axis=0)
            ci = 1.96 * std / np.sqrt(len(vals))

            ax.fill_between(gens, mean - ci, mean + ci, color=color, alpha=0.2)
            ax.plot(gens, mean, color=color, linestyle=linestyle,
                    linewidth=2, label=alg_name)

        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best")
    fig.suptitle(f"Learning Curves ($d = {target_dim}$)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "learning_curves")


# =============================================================================
# Figure 4: Distance Function Comparison
# =============================================================================

def plot_distance_comparison(results_dir: Path):
    """
    Multi-panel comparing distance functions across dimensions.
    """
    print("Generating Figure 4: Distance Comparison...")

    summary_path = results_dir / "distance_comparison" / "summary.json"
    if not summary_path.exists():
        print(f"  WARNING: {summary_path} not found, skipping")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    dimensions = sorted([int(d) for d in summary.keys()])

    # Get distance function names (excluding 'sigma')
    dist_names = [k for k in summary[str(dimensions[0])].keys() if k != "sigma"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    metrics = [
        ("uniformity_cv", "Uniformity CV $\\downarrow$"),
        ("interior_ratio", "Interior Ratio $\\uparrow$"),
    ]

    dist_colors = plt.cm.Set2(np.linspace(0, 1, len(dist_names)))

    for ax, (metric, ylabel) in zip(axes, metrics):
        x = np.arange(len(dimensions))
        width = 0.8 / len(dist_names)

        for i, name in enumerate(dist_names):
            means = []
            stds = []
            for dim in dimensions:
                if name in summary[str(dim)] and metric in summary[str(dim)][name]:
                    means.append(summary[str(dim)][name][metric]["mean"])
                    stds.append(summary[str(dim)][name][metric]["std"])
                else:
                    means.append(0)
                    stds.append(0)

            offset = (i - len(dist_names)/2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   label=name, color=dist_colors[i],
                   edgecolor='black', linewidth=0.5, capsize=2)

        ax.set_xlabel("Behavior Space Dimension")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dimensions])
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].legend(loc="best", fontsize=7)
    fig.tight_layout()
    save_fig(fig, "distance_comparison")


# =============================================================================
# Figure 5: Archive Size Ablation
# =============================================================================

def plot_archive_size_ablation(results_dir: Path):
    """
    Line plot: metrics vs archive size K.
    """
    print("Generating Figure 5: Archive Size Ablation...")

    summary_path = results_dir / "archive_size_ablation" / "summary.json"
    if not summary_path.exists():
        print(f"  WARNING: {summary_path} not found, skipping")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    sizes = sorted([int(k) for k in summary.keys()])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    metrics = [
        ("qd_score_at_budget", "QD-Score @ $K$"),
        ("uniformity_cv", "Uniformity CV $\\downarrow$"),
        ("mean_fitness", "Mean Fitness"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        means = [summary[str(k)][metric]["mean"] for k in sizes]
        stds = [summary[str(k)][metric]["std"] for k in sizes]

        ax.errorbar(sizes, means, yerr=stds,
                    color=COLORS["MMR-Elites"], marker="o",
                    linewidth=2, markersize=6, capsize=3)

        ax.set_xlabel("Archive Size $K$")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Archive Size Sensitivity ($d = 20$)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "archive_size_ablation")


# =============================================================================
# Main
# =============================================================================

def find_latest_results(base_dir: Path = Path("results")) -> Optional[Path]:
    """Find the most recent run directory."""
    run_dirs = sorted(base_dir.glob("run_*"))
    if run_dirs:
        return run_dirs[-1]

    # Fall back to individual result directories
    if (base_dir / "dimensionality_scaling").exists():
        return base_dir
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to results directory")
    parser.add_argument("--only", type=str, default=None,
                        choices=["scaling", "lambda", "curves", "distance", "archsize"],
                        help="Generate only one figure")
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = find_latest_results()
        if results_dir is None:
            print("ERROR: No results directory found. Run experiments first.")
            print("  python experiments/run_all.py --quick")
            return

    print(f"Using results from: {results_dir}")
    print(f"Output to: {FIGURE_DIR}")
    print()

    figure_fns = {
        "scaling": plot_dimensionality_scaling,
        "lambda": plot_lambda_ablation,
        "curves": plot_learning_curves,
        "distance": plot_distance_comparison,
        "archsize": plot_archive_size_ablation,
    }

    if args.only:
        figure_fns[args.only](results_dir)
    else:
        for name, fn in figure_fns.items():
            try:
                fn(results_dir)
            except Exception as e:
                print(f"  ERROR generating {name}: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
