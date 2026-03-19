"""
Integration tests for experiments/run_benchmark.py.

Verifies that run_experiment() completes for each algorithm
and returns the expected result format.
"""

import tempfile
from pathlib import Path

import pytest

from mmr_elites.utils.config import ExperimentConfig


@pytest.fixture
def small_config(tmp_path):
    """Minimal config for fast testing."""
    return ExperimentConfig(
        task="arm",
        n_dof=5,
        generations=5,
        batch_size=50,
        archive_size=50,
        seed=42,
        log_interval=5,
        output_dir=str(tmp_path),
        exp_name="test_benchmark",
    )


REQUIRED_KEYS = {"algorithm", "seed", "runtime", "final_metrics", "history"}
REQUIRED_METRICS = {"qd_score", "mean_fitness", "max_fitness"}


class TestRunBenchmark:
    """Tests for run_experiment() with each algorithm."""

    def test_mmr_elites(self, small_config):
        from experiments.run_benchmark import run_experiment

        small_config.algorithm = "mmr_elites"
        result = run_experiment(small_config)

        assert REQUIRED_KEYS.issubset(set(result.keys()))
        assert REQUIRED_METRICS.issubset(set(result["final_metrics"].keys()))

    def test_map_elites(self, small_config):
        from experiments.run_benchmark import run_experiment

        small_config.algorithm = "map_elites"
        result = run_experiment(small_config)

        assert REQUIRED_KEYS.issubset(set(result.keys()))
        assert REQUIRED_METRICS.issubset(set(result["final_metrics"].keys()))

    def test_cvt_map_elites(self, small_config):
        from experiments.run_benchmark import run_experiment

        small_config.algorithm = "cvt_map_elites"
        result = run_experiment(small_config)

        assert REQUIRED_KEYS.issubset(set(result.keys()))
        assert REQUIRED_METRICS.issubset(set(result["final_metrics"].keys()))

    def test_random(self, small_config):
        from experiments.run_benchmark import run_experiment

        small_config.algorithm = "random"
        result = run_experiment(small_config)

        assert REQUIRED_KEYS.issubset(set(result.keys()))
        assert REQUIRED_METRICS.issubset(set(result["final_metrics"].keys()))
