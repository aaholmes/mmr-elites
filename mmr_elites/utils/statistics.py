"Statistical analysis utilities for QD experiments."

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


def compute_confidence_interval(
    data: np.ndarray, 
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval. 
    
    Args:
        data: Array of values
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        (mean, ci_lower, ci_upper)
    """
    n = len(data)
    if n < 2:
        return np.mean(data), np.mean(data), np.mean(data)
    mean = np.mean(data)
    se = stats.sem(data)
    ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - ci, mean + ci


def wilcoxon_signed_rank_test(
    x: np.ndarray, 
    y: np.ndarray,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Non-parametric paired test for comparing algorithms.
    
    Args:
        x, y: Paired samples from two algorithms
        alternative: "two-sided", "greater", or "less"
    
    Returns:
        (statistic, p_value)
    """
    stat, p = stats.wilcoxon(x, y, alternative=alternative)
    return float(stat), float(p)


def mann_whitney_u_test(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Non-parametric unpaired test for comparing algorithms.
    
    Args:
        x, y: Samples from two algorithms (can be different sizes)
        alternative: "two-sided", "greater", or "less"
    
    Returns:
        (statistic, p_value)
    """
    stat, p = stats.mannwhitneyu(x, y, alternative=alternative)
    return float(stat), float(p)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    
    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def compute_all_statistics(
    results: Dict[str, List[Dict]],
    metric: str = "qd_score_at_budget",
    baseline: str = "Random"
) -> Dict:
    """
    Compute comprehensive statistics for experiment results.
    
    Args:
        results: Dict mapping algorithm name to list of result dicts
        metric: Metric to analyze
        baseline: Algorithm to compare against
    
    Returns:
        Dictionary with summary statistics and significance tests
    """
    summary = {}
    
    # Extract metric values for each algorithm
    values = {}
    for alg, runs in results.items():
        vals = []
        for r in runs:
            # Handle both QDResult and dict
            metrics = r.final_metrics if hasattr(r, 'final_metrics') else r.get("final_metrics", {})
            val = metrics.get(metric, metrics.get("qd_score", 0))
            vals.append(val)
        values[alg] = np.array(vals)
    
    # Compute summary stats for each algorithm
    for alg, vals in values.items():
        if len(vals) == 0:
            continue
        mean, ci_low, ci_high = compute_confidence_interval(vals)
        summary[alg] = {
            "mean": mean,
            "std": float(np.std(vals)),
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": len(vals),
        }
    
    # Pairwise comparisons against baseline
    if baseline in values and len(values[baseline]) > 0:
        baseline_vals = values[baseline]
        comparisons = {}
        
        for alg, vals in values.items():
            if alg == baseline or len(vals) == 0:
                continue
            
            # Effect size
            d = cohens_d(vals, baseline_vals)
            
            # Significance test (use Mann-Whitney for robustness)
            try:
                stat, p = mann_whitney_u_test(vals, baseline_vals, alternative="greater")
            except ValueError: # If all values are identical
                stat, p = 0.0, 1.0
            
            comparisons[alg] = {
                "vs": baseline,
                "cohens_d": d,
                "mann_whitney_stat": stat,
                "p_value": p,
                "significant_005": p < 0.05,
                "significant_001": p < 0.01,
            }
        
        summary["_comparisons"] = comparisons
    
    return summary


def format_results_table(
    results: Dict[str, List[Dict]],
    metrics: List[str] = None
) -> str:
    """
    Format results as a publication-ready table.
    
    Args:
        results: Dict mapping algorithm name to list of result dicts
        metrics: List of metrics to include
    
    Returns:
        Formatted string table
    """
    if metrics is None:
        metrics = ["qd_score_at_budget", "mean_fitness", "uniformity_cv"]
    
    lines = []
    
    # Header
    header = f"{ 'Algorithm':<20}"
    for m in metrics:
        header += f" | {m:<20}"
    lines.append(header)
    lines.append("-" * len(header))
    
    # Data rows
    for alg, runs in results.items():
        if not runs:
            continue
        row = f"{alg:<20}"
        for m in metrics:
            vals = []
            for r in runs:
                metrics_dict = r.final_metrics if hasattr(r, 'final_metrics') else r.get("final_metrics", {})
                vals.append(metrics_dict.get(m, 0))
            mean, std = np.mean(vals), np.std(vals)
            row += f" | {mean:>8.2f} ± {std:<8.2f}"
        lines.append(row)
    
    return "\n".join(lines)
