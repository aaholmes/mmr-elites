#!/usr/bin/env python3
"""
MMR-Elites Command Line Interface.

Usage:
    mmr-elites run --task arm --algorithm mmr_elites --generations 1000
    mmr-elites benchmark --quick
    mmr-elites compare --dimensions 5 10 20 50
    mmr-elites demo
"""

import json
import pickle
from datetime import datetime
from pathlib import Path

import click
import numpy as np


@click.group()
@click.version_option(version="0.1.0", prog_name="MMR-Elites")
def main():
    """MMR-Elites: Quality-Diversity via Maximum Marginal Relevance."""
    pass


@main.command()
@click.option(
    "--task",
    type=click.Choice(["arm", "rastrigin"]),
    default="arm",
    help="Benchmark task to run",
)
@click.option(
    "--algorithm",
    type=click.Choice(["mmr_elites", "map_elites", "cvt_map_elites", "random"]),
    default="mmr_elites",
    help="Algorithm to run",
)
@click.option(
    "--generations", "-g", type=int, default=1000, help="Number of generations"
)
@click.option("--archive-size", "-k", type=int, default=1000, help="Archive size K")
@click.option(
    "--batch-size", "-b", type=int, default=200, help="Offspring per generation"
)
@click.option(
    "--lambda-val",
    "-l",
    type=float,
    default=0.5,
    help="Diversity weight λ (MMR-Elites only)",
)
@click.option(
    "--n-dof", "-d", type=int, default=20, help="Degrees of freedom (arm task)"
)
@click.option("--seed", "-s", type=int, default=42, help="Random seed")
@click.option(
    "--output", "-o", type=click.Path(), default=None, help="Output directory"
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def run(
    task,
    algorithm,
    generations,
    archive_size,
    batch_size,
    lambda_val,
    n_dof,
    seed,
    output,
    quiet,
):
    """Run a single QD algorithm experiment."""
    from mmr_elites.algorithms import (
        run_cvt_map_elites,
        run_map_elites,
        run_mmr_elites,
        run_random_search,
    )
    from mmr_elites.tasks.arm import ArmTask
    from mmr_elites.tasks.rastrigin import RastriginTask

    # Setup task
    if task == "arm":
        task_obj = ArmTask(n_dof=n_dof, use_highdim_descriptor=True)
    else:
        task_obj = RastriginTask(n_dim=n_dof)

    if not quiet:
        click.echo(f"🚀 Running {algorithm} on {task} ({n_dof}D)")
        click.echo(
            f"   Generations: {generations}, Archive: {archive_size}, Seed: {seed}"
        )

    # Run algorithm
    algorithms = {
        "mmr_elites": lambda: run_mmr_elites(
            task_obj,
            archive_size=archive_size,
            generations=generations,
            batch_size=batch_size,
            lambda_val=lambda_val,
            seed=seed,
        ),
        "map_elites": lambda: run_map_elites(
            task_obj, generations=generations, batch_size=batch_size, seed=seed
        ),
        "cvt_map_elites": lambda: run_cvt_map_elites(
            task_obj,
            n_niches=archive_size,
            generations=generations,
            batch_size=batch_size,
            seed=seed,
        ),
        "random": lambda: run_random_search(
            task_obj,
            archive_size=archive_size,
            generations=generations,
            batch_size=batch_size,
            seed=seed,
        ),
    }

    result = algorithms[algorithm]()

    # Display results
    metrics = (
        result.final_metrics
        if hasattr(result, "final_metrics")
        else result["final_metrics"]
    )
    runtime = result.runtime if hasattr(result, "runtime") else result["runtime"]

    if not quiet:
        click.echo(f"\n📊 Results:")
        click.echo(f"   QD-Score:     {metrics['qd_score']:.2f}")
        click.echo(
            f"   QD@Budget:    {metrics.get('qd_score_at_budget', metrics['qd_score']):.2f}"
        )
        click.echo(f"   Max Fitness:  {metrics['max_fitness']:.4f}")
        click.echo(f"   Mean Fitness: {metrics['mean_fitness']:.4f}")
        click.echo(
            f"   Uniformity:   {metrics.get('uniformity_cv', 0):.4f} (lower=better)"
        )
        click.echo(f"   Archive Size: {metrics.get('archive_size', archive_size)}")
        click.echo(f"   Runtime:      {runtime:.2f}s")

    # Save results
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "results.pkl", "wb") as f:
            pickle.dump(result, f)

        summary = {
            "algorithm": (
                result.algorithm
                if hasattr(result, "algorithm")
                else result["algorithm"]
            ),
            "seed": seed,
            "runtime": runtime,
            "final_metrics": {k: float(v) for k, v in metrics.items()},
        }
        with open(output_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if not quiet:
            click.echo(f"\n💾 Saved to {output_path}")


@main.command()
@click.option("--quick", is_flag=True, help="Quick test (2 seeds, 200 generations)")
@click.option(
    "--full", is_flag=True, help="Full benchmark (10 seeds, 2000 generations)"
)
@click.option("--seeds", "-n", type=int, default=5, help="Number of random seeds")
@click.option("--generations", "-g", type=int, default=1000, help="Generations per run")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="results/benchmark",
    help="Output directory",
)
def benchmark(quick, full, seeds, generations, output):
    """Run complete benchmark comparing all algorithms."""
    from mmr_elites.algorithms import (
        run_cvt_map_elites,
        run_map_elites,
        run_mmr_elites,
        run_random_search,
    )
    from mmr_elites.tasks.arm import ArmTask

    if quick:
        seeds, generations = 2, 200
    elif full:
        seeds, generations = 10, 2000

    output_path = Path(output) / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 60)
    click.echo("MMR-Elites Benchmark Suite")
    click.echo("=" * 60)
    click.echo(f"Seeds: {seeds}, Generations: {generations}")
    click.echo(f"Output: {output_path}")
    click.echo("=" * 60)

    task = ArmTask(n_dof=20, use_highdim_descriptor=True)

    algorithms = {
        "MMR-Elites": lambda s: run_mmr_elites(task, generations=generations, seed=s),
        "MAP-Elites": lambda s: run_map_elites(task, generations=generations, seed=s),
        "CVT-MAP-Elites": lambda s: run_cvt_map_elites(
            task, generations=generations, seed=s
        ),
        "Random": lambda s: run_random_search(task, generations=generations, seed=s),
    }

    results = {name: [] for name in algorithms}

    for seed in range(seeds):
        click.echo(f"\n--- Seed {seed + 1}/{seeds} ---")
        for name, alg_fn in algorithms.items():
            click.echo(f"  {name}...", nl=False)
            try:
                r = alg_fn(seed)
                results[name].append(r)

                metrics = (
                    r.final_metrics
                    if hasattr(r, "final_metrics")
                    else r["final_metrics"]
                )
                qd = metrics.get("qd_score_at_budget", metrics["qd_score"])
                click.echo(f" QD@K={qd:.1f}")
            except Exception as e:
                click.echo(f" ERROR: {e}")
                import traceback

                traceback.print_exc()

    # Save and display summary
    with open(output_path / "all_results.pkl", "wb") as f:
        pickle.dump(results, f)

    click.echo("\n" + "=" * 60)
    click.echo("RESULTS SUMMARY")
    click.echo("=" * 60)

    summary = {}
    for name, runs in results.items():
        if runs:
            qd_scores = []
            for r in runs:
                metrics = (
                    r.final_metrics
                    if hasattr(r, "final_metrics")
                    else r["final_metrics"]
                )
                qd_scores.append(metrics.get("qd_score_at_budget", metrics["qd_score"]))

            summary[name] = {
                "qd_score_mean": float(np.mean(qd_scores)),
                "qd_score_std": float(np.std(qd_scores)),
            }
            click.echo(
                f"{name:20s}: {np.mean(qd_scores):8.2f} ± {np.std(qd_scores):.2f}"
            )

    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    click.echo(f"\n✅ Results saved to {output_path}")


@main.command()
@click.option(
    "--dimensions",
    "-d",
    multiple=True,
    type=int,
    default=[5, 10, 20, 50],
    help="Dimensions to test (can specify multiple)",
)
@click.option("--seeds", "-n", type=int, default=5, help="Seeds per dimension")
@click.option("--generations", "-g", type=int, default=1000, help="Generations per run")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="results/scaling",
    help="Output directory",
)
def compare(dimensions, seeds, generations, output):
    """Run dimensionality scaling comparison."""
    from experiments.dimensionality_scaling import run_dimensionality_scaling

    output_path = Path(output)

    click.echo("=" * 60)
    click.echo("Dimensionality Scaling Experiment")
    click.echo("=" * 60)
    click.echo(f"Dimensions: {list(dimensions)}")
    click.echo(f"Seeds: {seeds}, Generations: {generations}")

    run_dimensionality_scaling(
        dimensions=list(dimensions),
        n_seeds=seeds,
        generations=generations,
        output_dir=output_path,
    )


@main.command()
@click.option("--port", "-p", type=int, default=8501, help="Port for Streamlit server")
def demo(port):
    """Launch interactive demo (requires streamlit)."""
    import subprocess
    import sys

    demo_path = Path(__file__).parent.parent / "demo" / "app.py"

    if not demo_path.exists():
        click.echo("❌ Demo not found. Creating demo scaffold...")
        create_demo_scaffold()
        demo_path = Path(__file__).parent.parent / "demo" / "app.py"

    click.echo(f"🚀 Launching demo on port {port}...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(demo_path),
            "--server.port",
            str(port),
        ]
    )


def create_demo_scaffold():
    """Create the demo directory and files if they don't exist."""
    demo_dir = Path(__file__).parent.parent / "demo"
    demo_dir.mkdir(exist_ok=True)

    app_content = '''"""
MMR-Elites Interactive Demo
"""
import streamlit as st

st.set_page_config(page_title="MMR-Elites Demo", layout="wide")
st.title("🤖 MMR-Elites: Quality-Diversity via Maximum Marginal Relevance")
st.markdown("Demo coming soon! Run `mmr-elites benchmark --quick` to test the algorithms.")
'''

    with open(demo_dir / "app.py", "w") as f:
        f.write(app_content)


if __name__ == "__main__":
    main()
