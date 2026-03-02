"""Tests for CLI module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from mmr_elites.cli import create_demo_scaffold, main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_result():
    """Create a mock QDResult-like object."""
    from mmr_elites.algorithms.base import QDResult

    return QDResult(
        algorithm="MMRElites",
        seed=42,
        runtime=1.23,
        final_metrics={
            "qd_score": 100.0,
            "qd_score_at_budget": 95.0,
            "max_fitness": 0.99,
            "mean_fitness": 0.5,
            "uniformity_cv": 0.1,
            "archive_size": 100,
        },
        history={"generation": [1], "qd_score": [50.0]},
        final_genomes=np.zeros((10, 5)),
        final_fitness=np.ones(10),
        final_descriptors=np.random.rand(10, 5),
    )


def _patch_algorithms(mock_result):
    """Return a context manager that patches all algorithm run functions."""
    return (
        patch("mmr_elites.algorithms.run_mmr_elites", return_value=mock_result),
        patch("mmr_elites.algorithms.run_map_elites", return_value=mock_result),
        patch("mmr_elites.algorithms.run_cvt_map_elites", return_value=mock_result),
        patch("mmr_elites.algorithms.run_random_search", return_value=mock_result),
    )


class TestMainGroup:
    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "MMR-Elites" in result.output


class TestRunCommand:
    def test_run_mmr_elites(self, runner, mock_result):
        p1, p2, p3, p4 = _patch_algorithms(mock_result)
        with p1, p2, p3, p4:
            result = runner.invoke(
                main,
                [
                    "run",
                    "--algorithm",
                    "mmr_elites",
                    "--generations",
                    "10",
                    "--seed",
                    "0",
                ],
            )
            assert result.exit_code == 0, result.output
            assert "QD-Score" in result.output

    def test_run_map_elites(self, runner, mock_result):
        p1, p2, p3, p4 = _patch_algorithms(mock_result)
        with p1, p2, p3, p4:
            result = runner.invoke(
                main,
                ["run", "--algorithm", "map_elites", "--generations", "10"],
            )
            assert result.exit_code == 0, result.output

    def test_run_cvt_map_elites(self, runner, mock_result):
        p1, p2, p3, p4 = _patch_algorithms(mock_result)
        with p1, p2, p3, p4:
            result = runner.invoke(
                main,
                ["run", "--algorithm", "cvt_map_elites", "--generations", "10"],
            )
            assert result.exit_code == 0, result.output

    def test_run_random(self, runner, mock_result):
        p1, p2, p3, p4 = _patch_algorithms(mock_result)
        with p1, p2, p3, p4:
            result = runner.invoke(
                main,
                ["run", "--algorithm", "random", "--generations", "10"],
            )
            assert result.exit_code == 0, result.output

    def test_run_rastrigin_task(self, runner, mock_result):
        p1, p2, p3, p4 = _patch_algorithms(mock_result)
        with p1, p2, p3, p4:
            result = runner.invoke(
                main,
                ["run", "--task", "rastrigin", "--generations", "10"],
            )
            assert result.exit_code == 0, result.output

    def test_run_quiet_mode(self, runner, mock_result):
        p1, p2, p3, p4 = _patch_algorithms(mock_result)
        with p1, p2, p3, p4:
            result = runner.invoke(
                main,
                ["run", "--quiet", "--generations", "10"],
            )
            assert result.exit_code == 0, result.output
            assert "QD-Score" not in result.output

    def test_run_with_output(self, runner, mock_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3, p4 = _patch_algorithms(mock_result)
            with p1, p2, p3, p4:
                result = runner.invoke(
                    main,
                    ["run", "--generations", "10", "--output", tmpdir],
                )
                assert result.exit_code == 0, result.output
                assert (Path(tmpdir) / "results.pkl").exists()
                assert (Path(tmpdir) / "summary.json").exists()

                with open(Path(tmpdir) / "summary.json") as f:
                    summary = json.load(f)
                assert "final_metrics" in summary

    def test_run_with_dict_result(self, runner):
        """Test that dict-style results are handled correctly."""
        dict_result = {
            "algorithm": "MMRElites",
            "seed": 42,
            "runtime": 1.0,
            "final_metrics": {
                "qd_score": 50.0,
                "max_fitness": 0.8,
                "mean_fitness": 0.4,
            },
        }
        p1, p2, p3, p4 = _patch_algorithms(dict_result)
        with p1, p2, p3, p4:
            result = runner.invoke(
                main,
                ["run", "--generations", "10"],
            )
            assert result.exit_code == 0, result.output

    def test_run_with_all_options(self, runner, mock_result):
        p1, p2, p3, p4 = _patch_algorithms(mock_result)
        with p1, p2, p3, p4:
            result = runner.invoke(
                main,
                [
                    "run",
                    "--task",
                    "arm",
                    "--algorithm",
                    "mmr_elites",
                    "--generations",
                    "10",
                    "--archive-size",
                    "50",
                    "--batch-size",
                    "20",
                    "--lambda-val",
                    "0.7",
                    "--n-dof",
                    "5",
                    "--seed",
                    "123",
                ],
            )
            assert result.exit_code == 0, result.output


class TestBenchmarkCommand:
    def test_benchmark_quick(self, runner, mock_result):
        p1, p2, p3, p4 = _patch_algorithms(mock_result)
        with p1, p2, p3, p4:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = runner.invoke(
                    main,
                    ["benchmark", "--quick", "--output", tmpdir],
                )
                assert result.exit_code == 0, result.output
                assert "Benchmark Suite" in result.output

    def test_benchmark_full_flag(self, runner, mock_result):
        p1, p2, p3, p4 = _patch_algorithms(mock_result)
        with p1, p2, p3, p4:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = runner.invoke(
                    main,
                    ["benchmark", "--full", "--output", tmpdir],
                )
                assert result.exit_code == 0, result.output

    def test_benchmark_error_handling(self, runner):
        """Test that benchmark handles algorithm errors gracefully."""
        with patch(
            "mmr_elites.algorithms.run_mmr_elites",
            side_effect=RuntimeError("test error"),
        ), patch(
            "mmr_elites.algorithms.run_map_elites",
            side_effect=RuntimeError("test error"),
        ), patch(
            "mmr_elites.algorithms.run_cvt_map_elites",
            side_effect=RuntimeError("test error"),
        ), patch(
            "mmr_elites.algorithms.run_random_search",
            side_effect=RuntimeError("test error"),
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = runner.invoke(
                    main,
                    ["benchmark", "--quick", "--output", tmpdir],
                )
                assert result.exit_code == 0, result.output
                assert "ERROR" in result.output


class TestCompareCommand:
    def test_compare(self, runner):
        with patch(
            "experiments.dimensionality_scaling.run_dimensionality_scaling"
        ) as mock_fn:
            result = runner.invoke(
                main,
                ["compare", "--dimensions", "5", "--seeds", "1", "--generations", "10"],
            )
            assert result.exit_code == 0, result.output
            mock_fn.assert_called_once()


class TestDemoCommand:
    def test_demo_with_existing_app(self, runner):
        with patch("subprocess.run") as mock_run:
            # demo/app.py exists in the repo
            result = runner.invoke(main, ["demo"])
            # subprocess.run will be called with streamlit
            mock_run.assert_called_once()

    def test_demo_custom_port(self, runner):
        with patch("subprocess.run") as mock_run:
            result = runner.invoke(main, ["demo", "--port", "9000"])
            call_args = mock_run.call_args[0][0]
            assert "9000" in call_args


class TestCreateDemoScaffold:
    def test_creates_demo_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch Path(__file__) to use temp dir
            mock_path = MagicMock()
            mock_path.parent.parent = Path(tmpdir)
            with patch("mmr_elites.cli.Path", return_value=mock_path) as MockPath:
                MockPath.return_value.parent.parent = Path(tmpdir)
                create_demo_scaffold()
                demo_dir = Path(tmpdir) / "demo"
                assert demo_dir.exists()
                assert (demo_dir / "app.py").exists()
